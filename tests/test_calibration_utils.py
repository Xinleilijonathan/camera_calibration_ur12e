"""Config validation, transforms, atomic writes, and data separation."""
import math

import numpy as np
import pytest
import yaml

import calibration_utils as utils
from calibration_utils import (CameraPaths, ConfigError, camera_paths,
                               error_statistics, invert_transform,
                               is_placeholder_serial, load_calibration_config,
                               load_cameras_config, load_safety_config,
                               make_transform, matrix_to_pose, matrix_to_rotvec,
                               median_absolute_deviation, normalize_min_max,
                               pose_difference, pose_to_matrix,
                               require_board_verified, resolve_camera,
                               rotation_angle_deg, rotvec_to_matrix, save_csv,
                               save_yaml, to_builtin)


class TestShippedConfigs:
    """The files in config/ must load and validate as shipped."""

    def test_cameras_yaml_loads(self):
        config = load_cameras_config()
        assert set(config["cameras"]) == set(utils.VALID_CAMERA_NAMES)

    def test_camera_serials_are_distinct_and_present(self):
        """Two cameras sharing a serial is the one identity error nothing catches.

        This used to assert the opposite -- that the serials were still
        placeholders -- back when config/ was a blank template. It now holds
        the real serials for the ur12e-flexlab rig, so the useful invariant is
        no longer "unfilled" but "filled in, and filled in DISTINCTLY".

        A duplicated serial resolves two logical cameras to one physical
        device. Both calibrations then succeed, agree with themselves, and
        pass every reprojection and hold-out check, while describing the wrong
        camera. Placeholders are still rejected, but at the point of use, by
        find_device_for_serial().
        """
        config = load_cameras_config()
        serials = {name: str(entry["serial"]).strip()
                   for name, entry in config["cameras"].items()}
        for name, serial in serials.items():
            assert serial, f"{name} has an empty serial"
        duplicates = {s for s in serials.values()
                      if list(serials.values()).count(s) > 1}
        assert not duplicates, (
            f"cameras share serial(s) {sorted(duplicates)}: {serials}")

    def test_every_camera_declares_its_handeye_mode(self):
        """This rig mixes mountings, so the mode cannot come from a default."""
        config = load_cameras_config()
        for name, entry in config["cameras"].items():
            assert entry.get("handeye_mode") in ("eye_in_hand", "eye_to_hand"), (
                f"{name} has no valid handeye_mode")

    def test_calibration_yaml_loads(self):
        config = load_calibration_config()
        assert config["apriltag_grid"]["tag_family"] == "tag36h11"
        assert config["waypoint_collection"]["target_count"] == 30
        assert config["waypoint_collection"]["final_count"] == 20

    def test_selection_weights_sum_to_one(self):
        weights = load_calibration_config()["selection"]["weights"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_safety_yaml_loads_and_ships_locked_down(self):
        config = load_safety_config()
        connection = config["connection"]
        assert connection["robot_ip"] == "127.0.0.1"
        assert connection["allow_physical_robot"] is False
        assert connection["allow_motion"] is False

    def test_shipped_speeds_are_conservative(self):
        motion = load_safety_config()["motion"]
        assert motion["joint_speed_rad_s"] <= 0.25
        assert motion["tcp_speed_m_s"] <= 0.05

    def test_shipped_board_is_marked_unverified(self):
        with pytest.raises(ConfigError, match="not confirmed"):
            require_board_verified(load_calibration_config())

    def test_require_board_verified_passes_when_set(self):
        config = load_calibration_config()
        config["apriltag_grid"]["verified_by_user"] = True
        require_board_verified(config)


class TestConfigValidation:
    def _write(self, tmp_path, name, data):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return path

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            utils.load_yaml(tmp_path / "absent.yaml")

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ConfigError, match="empty"):
            utils.load_yaml(path)

    def test_millimetre_tag_size_is_caught(self, tmp_path):
        path = self._write(tmp_path, "c.yaml", {"apriltag_grid": {
            "tag_family": "tag36h11", "rows": 6, "columns": 6,
            "tag_size_m": 30.0, "tag_spacing_m": 9.0}})
        with pytest.raises(ConfigError, match="METRES"):
            load_calibration_config(path)

    def test_zero_tag_size_is_rejected(self, tmp_path):
        path = self._write(tmp_path, "c.yaml", {"apriltag_grid": {
            "tag_family": "tag36h11", "rows": 6, "columns": 6,
            "tag_size_m": 0.0, "tag_spacing_m": 0.009}})
        with pytest.raises(ConfigError, match="tag_size_m"):
            load_calibration_config(path)

    def test_non_integer_rows_is_rejected(self, tmp_path):
        path = self._write(tmp_path, "c.yaml", {"apriltag_grid": {
            "tag_family": "tag36h11", "rows": 6.5, "columns": 6,
            "tag_size_m": 0.03, "tag_spacing_m": 0.009}})
        with pytest.raises(ConfigError, match="positive"):
            load_calibration_config(path)

    def test_unknown_backend_is_rejected(self, tmp_path):
        path = self._write(tmp_path, "cams.yaml", {"cameras": {
            "camera_1": {"backend": "gopro", "serial": "123"}}})
        with pytest.raises(ConfigError, match="unknown backend"):
            load_cameras_config(path)

    def test_inverted_joint_limits_are_rejected(self, tmp_path):
        safety = yaml.safe_load(utils.SAFETY_CONFIG.read_text())
        safety["joint_limits"]["elbow"] = {"min": 1.0, "max": -1.0}
        path = self._write(tmp_path, "s.yaml", safety)
        with pytest.raises(ConfigError, match="min >= max"):
            load_safety_config(path)

    def test_unknown_camera_name_lists_the_known_ones(self):
        with pytest.raises(ConfigError, match="camera_1"):
            resolve_camera("camera_9", load_cameras_config())

    @pytest.mark.parametrize("serial,expected", [
        ("REPLACE_WITH_SERIAL", True), ("SERIAL_NUMBER", True),
        ("XXXXXXXX", True), ("", True), (None, True), ("  ", True),
        ("943222071234", False), ("200901010001", False),
    ])
    def test_placeholder_detection(self, serial, expected):
        assert is_placeholder_serial(serial) is expected


class TestCameraSeparation:
    """Camera datasets must never overlap -- section AL."""

    def test_camera_subtrees_are_disjoint(self, tmp_path):
        paths = [camera_paths(name, tmp_path) for name in utils.VALID_CAMERA_NAMES]
        for path in paths:
            path.ensure()
        roots = [p.root.resolve() for p in paths]
        assert len(set(roots)) == 3
        for a in roots:
            for b in roots:
                if a != b:
                    assert not str(a).startswith(str(b) + "/")

    def test_every_result_path_is_camera_specific(self, tmp_path):
        one = camera_paths("camera_1", tmp_path)
        two = camera_paths("camera_2", tmp_path)
        for attribute in ("intrinsics_result", "final_result", "preliminary_result",
                          "waypoints_file", "waypoint_scores", "initial_center"):
            assert getattr(one, attribute) != getattr(two, attribute)

    def test_path_traversal_in_camera_name_is_rejected(self):
        for bad in ("../escape", "a/b", ".hidden", ""):
            with pytest.raises(ConfigError):
                camera_paths(bad)

    def test_ensure_creates_the_full_layout(self, tmp_path):
        paths = camera_paths("camera_1", tmp_path)
        paths.ensure()
        for directory in paths.all_dirs():
            assert directory.is_dir()


class TestTransforms:
    def test_rotvec_matrix_roundtrip(self):
        for rotvec in ([0.1, -0.2, 0.3], [0, 0, 0], [math.pi, 0, 0], [0.0, 2.5, -1.1]):
            matrix = rotvec_to_matrix(rotvec)
            assert utils.is_rotation_matrix(matrix)
            assert np.allclose(rotvec_to_matrix(matrix_to_rotvec(matrix)), matrix,
                               atol=1e-9)

    def test_pose_matrix_roundtrip(self):
        pose = [0.4, -0.15, 0.32, 0.8, -1.2, 0.35]
        assert np.allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-9)

    def test_invert_transform_is_a_true_inverse(self):
        transform = pose_to_matrix([0.1, 0.2, 0.3, 0.4, -0.5, 0.6])
        assert np.allclose(transform @ invert_transform(transform), np.eye(4), atol=1e-12)

    def test_rotation_angle_is_exact_for_a_known_rotation(self):
        matrix = rotvec_to_matrix([0, 0, math.radians(37.0)])
        assert rotation_angle_deg(matrix) == pytest.approx(37.0, abs=1e-9)

    def test_rotation_angle_is_stable_at_the_extremes(self):
        assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)
        assert rotation_angle_deg(rotvec_to_matrix([math.pi, 0, 0])) == pytest.approx(180.0)

    def test_pose_difference_reports_translation_and_rotation(self):
        a = [0.0, 0.0, 0.0, 0, 0, 0]
        b = [0.03, 0.04, 0.0, 0, 0, math.radians(10)]
        translation, rotation = pose_difference(a, b)
        assert translation == pytest.approx(0.05)
        assert rotation == pytest.approx(10.0, abs=1e-9)

    def test_make_transform_composes_correctly(self):
        rotation = rotvec_to_matrix([0, 0, math.pi / 2])
        transform = make_transform(rotation, [1, 2, 3])
        point = transform @ np.array([1, 0, 0, 1.0])
        assert np.allclose(point[:3], [1, 3, 3], atol=1e-9)

    def test_rotvec_to_rpy_matches_a_pure_yaw(self):
        rpy = utils.rotvec_to_rpy_deg([0, 0, math.radians(30)])
        assert rpy[2] == pytest.approx(30.0, abs=1e-6)
        assert rpy[0] == pytest.approx(0.0, abs=1e-6)

    def test_ur_rotation_vector_is_not_treated_as_rpy(self):
        """Guards the section-R warning: rx,ry,rz are axis-angle, not Euler."""
        rotvec = [1.2, -0.8, 0.5]
        assert not np.allclose(utils.rotvec_to_rpy_deg(rotvec), np.degrees(rotvec))


class TestNumerics:
    def test_normalize_min_max_maps_to_unit_range(self):
        result = normalize_min_max([2.0, 4.0, 6.0])
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_normalize_handles_identical_values_without_nan(self):
        result = normalize_min_max([5.0, 5.0, 5.0])
        assert np.allclose(result, 0.0)
        assert np.isfinite(result).all()

    def test_normalize_handles_empty_input(self):
        assert normalize_min_max([]).size == 0

    def test_normalize_pushes_non_finite_to_worst(self):
        result = normalize_min_max([1.0, 2.0, float("nan")])
        assert result[2] == 1.0

    def test_median_absolute_deviation(self):
        median, mad = median_absolute_deviation([1, 2, 3, 4, 100])
        assert median == 3
        assert mad > 0

    def test_error_statistics(self):
        stats = error_statistics([1.0, 2.0, 3.0])
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(2.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["rms"] == pytest.approx(math.sqrt(14 / 3))

    def test_error_statistics_on_empty_input(self):
        assert error_statistics([])["count"] == 0

    def test_error_statistics_ignores_non_finite(self):
        assert error_statistics([1.0, float("nan"), 3.0])["count"] == 2


class TestAtomicWrites:
    def test_save_yaml_roundtrips_numpy(self, tmp_path):
        path = tmp_path / "out.yaml"
        save_yaml(path, {"matrix": np.eye(3), "count": np.int64(7),
                         "value": np.float64(1.5), "flag": np.bool_(True)})
        data = yaml.safe_load(path.read_text())
        assert data["matrix"] == [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        assert data["count"] == 7 and isinstance(data["count"], int)
        assert data["flag"] is True

    def test_save_yaml_writes_a_header(self, tmp_path):
        path = tmp_path / "out.yaml"
        save_yaml(path, {"a": 1}, header="line one\nline two")
        text = path.read_text()
        assert text.startswith("# line one\n# line two\n")

    def test_save_yaml_creates_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "out.yaml"
        save_yaml(path, {"a": 1})
        assert path.is_file()

    def test_save_yaml_leaves_no_temp_files(self, tmp_path):
        save_yaml(tmp_path / "out.yaml", {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["out.yaml"]

    def test_failed_write_does_not_destroy_the_previous_file(self, tmp_path):
        path = tmp_path / "out.yaml"
        save_yaml(path, {"good": 1})
        class Unserializable:
            pass
        with pytest.raises(Exception):
            save_yaml(path, {"bad": Unserializable()})
        assert yaml.safe_load(path.read_text()) == {"good": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["out.yaml"]

    def test_non_finite_floats_become_null_not_nan(self, tmp_path):
        path = tmp_path / "out.yaml"
        save_yaml(path, {"a": float("nan"), "b": float("inf")})
        data = yaml.safe_load(path.read_text())
        assert data["a"] is None and data["b"] is None

    def test_save_csv(self, tmp_path):
        import csv
        path = tmp_path / "scores.csv"
        save_csv(path, [{"id": 1, "score": np.float64(0.5)},
                        {"id": 2, "score": 0.25}], ["id", "score"])
        rows = list(csv.DictReader(path.open()))
        assert [r["id"] for r in rows] == ["1", "2"]
        assert rows[0]["score"] == "0.5"

    def test_save_csv_with_no_rows_writes_only_a_header(self, tmp_path):
        path = tmp_path / "empty.csv"
        save_csv(path, [], ["a", "b"])
        assert path.read_text().strip() == "a,b"

    def test_to_builtin_handles_nesting(self):
        result = to_builtin({"a": [np.int64(1), {"b": np.array([1.0, 2.0])}]})
        assert result == {"a": [1, {"b": [1.0, 2.0]}]}


class TestJointOrdering:
    def test_joint_names_are_in_controller_order(self):
        assert utils.JOINT_NAMES == ("base", "shoulder", "elbow",
                                     "wrist_1", "wrist_2", "wrist_3")

    def test_priority_is_distal_first(self):
        """Section I: wrist joints first, base last."""
        assert utils.JOINT_PRIORITY[0] == "wrist_3"
        assert utils.JOINT_PRIORITY[-1] == "base"
        assert set(utils.JOINT_PRIORITY) == set(utils.JOINT_NAMES)
