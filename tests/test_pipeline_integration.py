"""End-to-end pipeline test: the real scripts, run as subprocesses.

Builds a complete synthetic dataset for a camera whose hand-eye transform we
choose, then runs

    analyze_waypoints -> select_best_waypoints -> solve_handeye -> verify

exactly as the operator will, and checks the pipeline recovers the transform
we started from. This is the test that would catch a broken script long before
anyone stands next to a real UR12e.

Both mountings are covered, because this rig uses both.
"""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PYTHON = sys.executable

from apriltag_detector import AprilGridDetector, GridSpec  # noqa: E402
from calibration_utils import (invert_transform, make_transform,  # noqa: E402
                               matrix_to_pose, to_builtin,
                               transform_difference)

BOARD = {"tag_family": "tag36h11", "rows": 4, "columns": 5,
         "tag_size_m": 0.030, "tag_spacing_m": 0.009, "first_tag_id": 0,
         "minimum_tags_required": 12, "minimum_corners_required": 48,
         "verified_by_user": True}
K = np.array([[905.0, 0.0, 641.0], [0.0, 903.0, 359.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.02, -0.05, 0.0005, -0.0003, 0.01])
IMAGE_SIZE = (1280, 720)
SERIAL = "SYNTH12345"

TRUE_X = {
    "eye_in_hand": make_transform(
        cv2.Rodrigues(np.array([0.03, -0.04, 1.571]))[0], [0.041, -0.058, 0.093]),
    "eye_to_hand": make_transform(
        cv2.Rodrigues(np.array([2.05, 0.18, 0.26]))[0], [0.78, -0.35, 0.58]),
}
TRUE_CONSTANT = {
    "eye_in_hand": make_transform(
        cv2.Rodrigues(np.array([0.03, 0.02, 0.35]))[0], [0.44, 0.08, 0.10]),
    "eye_to_hand": make_transform(
        cv2.Rodrigues(np.array([0.04, 3.09, 0.03]))[0], [0.010, 0.025, 0.152]),
}


def transform(rotvec, translation):
    return make_transform(cv2.Rodrigues(np.asarray(rotvec, float).reshape(3, 1))[0],
                          translation)


def build_dataset(root: Path, mode: str, count: int = 30,
                  noise_px: float = 0.08, seed: int = 11) -> dict:
    """Write a full camera_1 dataset: intrinsics plus `count` waypoints."""
    rng = np.random.default_rng(seed)
    spec = GridSpec(BOARD["tag_family"], BOARD["rows"], BOARD["columns"],
                    BOARD["tag_size_m"], BOARD["tag_spacing_m"])
    detector = AprilGridDetector(spec, {"minimum_tags_required": 1})
    object_points = np.asarray(detector.board.getObjPoints(),
                               dtype=np.float64).reshape(-1, 3)
    tag_ids = list(range(spec.tag_count))

    camera = root / "camera_1"
    (camera / "intrinsics" / "observations").mkdir(parents=True, exist_ok=True)
    (camera / "intrinsics" / "images").mkdir(parents=True, exist_ok=True)
    handeye = camera / "handeye"
    for sub in ("images", "observations", "waypoints", "selection", "verification"):
        (handeye / sub).mkdir(parents=True, exist_ok=True)

    (camera / "intrinsics" / "result.yaml").write_text(yaml.safe_dump(to_builtin({
        "camera_name": "camera_1", "camera_serial": SERIAL,
        "timestamp": "2026-09-14T00:00:00.000+00:00",
        "image_width": IMAGE_SIZE[0], "image_height": IMAGE_SIZE[1],
        "observation_count": 30, "distortion_model": "standard",
        "camera_matrix": K.tolist(), "distortion_coefficients": DIST.tolist(),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "rms_reprojection_error_px": 0.17,
        "mean_reprojection_error_px": 0.15,
        "median_reprojection_error_px": 0.14,
        "max_reprojection_error_px": 0.55,
        "per_observation": [], "warnings": [],
    })))

    # Centre the board on its own origin so a board pose is easy to keep in view.
    object_points = object_points - np.array(
        [spec.width_m / 2, spec.height_m / 2, 0.0])

    X = TRUE_X[mode]
    constant = TRUE_CONSTANT[mode]
    blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 90, dtype=np.uint8)
    written = 0
    index = 0
    while written < count and index < count * 8:
        index += 1
        # Generate the BOARD-IN-CAMERA pose first, so the board is guaranteed
        # to be in front of the camera and in frame, then derive the robot pose
        # that the ground-truth chain requires. Going the other way round makes
        # it very easy to place the board behind the camera.
        board = transform(
            np.array([0.30 * np.sin(index * 0.8),
                      0.28 * np.cos(index * 1.1),
                      0.45 * np.sin(index * 0.55)]) + rng.normal(0, 0.05, 3),
            np.array([0.055 * np.cos(index * 0.7),
                      0.045 * np.sin(index * 0.9),
                      0.46 + 0.10 * np.sin(index * 0.4)]) + rng.normal(0, 0.004, 3))

        if mode == "eye_in_hand":
            # board = inv(X) @ inv(robot) @ constant  =>  robot = constant @ inv(board) @ inv(X)
            robot = constant @ invert_transform(board) @ invert_transform(X)
        else:
            # board = inv(X) @ robot @ constant  =>  robot = X @ board @ inv(constant)
            robot = X @ board @ invert_transform(constant)

        rvec, _ = cv2.Rodrigues(board[:3, :3])
        tvec = board[:3, 3]
        if tvec[2] < 0.20:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, DIST)
        image_points = projected.reshape(-1, 2)
        if (image_points[:, 0].min() < 30 or image_points[:, 0].max() > IMAGE_SIZE[0] - 30
                or image_points[:, 1].min() < 30
                or image_points[:, 1].max() > IMAGE_SIZE[1] - 30):
            continue
        image_points = image_points + rng.normal(0, noise_px, image_points.shape)

        written += 1
        name = f"waypoint_{written:03d}"
        cv2.imwrite(str(handeye / "images" / f"{name}.png"), blank)
        # Robot "TCP pose" is the flange transform, as UR reports it.
        tcp = matrix_to_pose(robot)
        # Joints are not used by the solver; plausible values keep the
        # diversity and replay code paths exercised.
        joints = (np.array([0.1, -1.2, 1.1, -1.4, -1.5, 0.2])
                  + rng.normal(0, 0.05, 6)).tolist()
        residual = float(np.sqrt(np.mean(
            np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1) ** 2)))
        (handeye / "observations" / f"{name}.yaml").write_text(yaml.safe_dump(to_builtin({
            "camera_name": "camera_1", "camera_serial": SERIAL,
            "waypoint_number": written, "waypoint_name": name,
            "timestamp": "2026-09-14T00:00:00.000+00:00",
            "image": f"{name}.png",
            "actual_joints_rad": joints,
            "actual_joints_deg": np.degrees(joints).tolist(),
            "actual_joint_velocity_rad_s": [0.0] * 6,
            "actual_tcp_vector": tcp.tolist(),
            "actual_tcp": {"x": tcp[0], "y": tcp[1], "z": tcp[2],
                           "rx": tcp[3], "ry": tcp[4], "rz": tcp[5]},
            "detected_tag_ids": tag_ids,
            "number_of_tags": len(tag_ids),
            "number_of_corners": len(image_points),
            "detected_tag_corners": image_points.reshape(-1, 4, 2).tolist(),
            "object_points": object_points.tolist(),
            "image_points": image_points.tolist(),
            "board_pose_camera": {
                "rvec": rvec.reshape(3).tolist(), "tvec": tvec.tolist(),
                "distance_m": float(np.linalg.norm(tvec)),
                "tilt_deg": 12.0, "pnp_reprojection_px": residual,
                "pnp_max_reprojection_px": residual * 2.5},
            "detection_quality": {
                "tags_detected": len(tag_ids), "corners_detected": len(image_points),
                "visible_fraction": 1.0, "missing_tag_count": 0,
                "clipped_fraction": 0.0, "border_margin_px": 40.0,
                "sharpness": 850.0, "valid": True, "reasons": []},
            "intrinsics_file": str(camera / "intrinsics" / "result.yaml"),
            "settle_time_s": 0.5, "robot_mode": 7, "safety_mode": 1,
        })))
    assert written == count, f"only generated {written}/{count} waypoints"
    return {"X": X, "constant": constant, "count": written}


def build_config(config_dir: Path, mode: str) -> None:
    """A config directory pointing camera_1 at the synthetic dataset."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "cameras.yaml").write_text(yaml.safe_dump({
        "cameras": {
            "camera_1": {"name": "camera_1", "backend": "realsense",
                         "serial": SERIAL, "model": "D405",
                         "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1],
                         "fps": 30, "handeye_mode": mode},
            "camera_2": {"name": "camera_2", "backend": "realsense",
                         "serial": "OTHER222", "handeye_mode": "eye_to_hand"},
            "camera_3": {"name": "camera_3", "backend": "realsense",
                         "serial": "OTHER333", "handeye_mode": "eye_to_hand"},
        },
        "capture": {"warmup_frames": 2, "flush_frames": 1},
    }))
    source = yaml.safe_load((ROOT / "config" / "calibration.yaml").read_text())
    source["apriltag_grid"] = dict(BOARD)
    source["handeye"]["mode"] = mode
    source["handeye"]["verified_by_user"] = True
    (config_dir / "calibration.yaml").write_text(yaml.safe_dump(source))
    safety = yaml.safe_load((ROOT / "config" / "safety.yaml").read_text())
    (config_dir / "safety.yaml").write_text(yaml.safe_dump(safety))


def run_script(name: str, arguments: list[str], environment: dict,
               expect_success: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [PYTHON, str(SCRIPTS / name), *arguments],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, **environment})
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"{name} failed with code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result


@pytest.fixture(params=["eye_in_hand", "eye_to_hand"])
def pipeline(request, tmp_path_factory):
    """Build a dataset and run the full pipeline once per mounting mode."""
    mode = request.param
    base = tmp_path_factory.mktemp(f"pipeline_{mode}")
    data_dir = base / "data"
    config_dir = base / "config"
    truth = build_dataset(data_dir, mode)
    build_config(config_dir, mode)
    environment = {
        "CAMERA_CALIBRATION_DATA_DIR": str(data_dir),
        "CAMERA_CALIBRATION_CONFIG_DIR": str(config_dir),
        "CAMERA_CALIBRATION_LOG_DIR": str(base / "logs"),
        "PYTHONPATH": str(ROOT / "src"),
    }
    outputs = {
        "analyze": run_script("analyze_waypoints.py",
                              ["--camera", "camera_1"], environment),
        "select": run_script("select_best_waypoints.py",
                             ["--camera", "camera_1", "--count", "20"], environment),
        "solve": run_script("solve_handeye.py",
                            ["--camera", "camera_1", "--selection", "best20"],
                            environment),
        "verify": run_script("verify_calibration.py",
                             ["--camera", "camera_1"], environment),
    }
    return {"mode": mode, "truth": truth, "data": data_dir,
            "config": config_dir, "env": environment, "out": outputs}


class TestFullPipeline:
    def test_all_stages_succeed(self, pipeline):
        for stage, result in pipeline["out"].items():
            assert result.returncode == 0, f"{stage} failed:\n{result.stdout}"

    def test_expected_files_are_written(self, pipeline):
        handeye = pipeline["data"] / "camera_1" / "handeye"
        for relative in ("preliminary_result_all_30.yaml",
                         "final_result_best_20.yaml",
                         "selection/waypoint_scores.csv",
                         "selection/selected_20.yaml",
                         "selection/rejected_10.yaml",
                         "verification/verification_report.yaml"):
            assert (handeye / relative).is_file(), f"missing {relative}"

    def test_final_transform_matches_ground_truth(self, pipeline):
        """The whole point: does the pipeline recover the transform we chose?"""
        final = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye"
             / "final_result_best_20.yaml").read_text())
        solved = np.asarray(final["transform"], dtype=np.float64)
        translation, rotation = transform_difference(solved, pipeline["truth"]["X"])
        assert translation * 1000 < 5.0, (
            f"{pipeline['mode']}: translation off by {translation * 1000:.2f} mm")
        assert rotation < 0.5, (
            f"{pipeline['mode']}: rotation off by {rotation:.3f} deg")

    def test_preliminary_used_all_thirty(self, pipeline):
        preliminary = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye"
             / "preliminary_result_all_30.yaml").read_text())
        assert preliminary["observation_count"] == 30

    def test_final_used_exactly_twenty(self, pipeline):
        final = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye"
             / "final_result_best_20.yaml").read_text())
        assert final["observation_count"] == 20
        assert len(final["holdout_numbers"]) == 10

    def test_preliminary_is_not_overwritten_by_the_final_solve(self, pipeline):
        """Section AA: the all-30 result must survive."""
        preliminary = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye"
             / "preliminary_result_all_30.yaml").read_text())
        assert preliminary["observation_count"] == 30
        assert "preliminary" in preliminary["label"]

    def test_all_thirty_observations_survive_selection(self, pipeline):
        """Section U: selection never deletes anything."""
        observations = list((pipeline["data"] / "camera_1" / "handeye"
                             / "observations").glob("waypoint_*.yaml"))
        images = list((pipeline["data"] / "camera_1" / "handeye"
                       / "images").glob("waypoint_*.png"))
        assert len(observations) == 30
        assert len(images) == 30

    def test_selected_and_rejected_partition_the_set(self, pipeline):
        selection = pipeline["data"] / "camera_1" / "handeye" / "selection"
        selected = yaml.safe_load((selection / "selected_20.yaml").read_text())
        rejected = yaml.safe_load((selection / "rejected_10.yaml").read_text())
        chosen = set(selected["selected_numbers"])
        assert len(chosen) == 20
        reasons = {k: v for k, v in rejected.items()
                   if k.startswith("waypoint_")}
        assert len(reasons) == 10
        assert not chosen & {int(name.split("_")[1]) for name in reasons}

    def test_every_rejection_has_a_reason(self, pipeline):
        rejected = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye" / "selection"
             / "rejected_10.yaml").read_text())
        for name, entry in rejected.items():
            if name.startswith("waypoint_"):
                assert entry.get("reason"), f"{name} has no reason"

    def test_scores_csv_has_every_column_and_row(self, pipeline):
        import csv
        path = (pipeline["data"] / "camera_1" / "handeye" / "selection"
                / "waypoint_scores.csv")
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 30
        from waypoint_quality import SCORE_COLUMNS
        assert set(rows[0]) == set(SCORE_COLUMNS)
        assert sum(int(r["selected"]) for r in rows) == 20

    def test_selected_waypoints_are_geometrically_diverse(self, pipeline):
        """Not twenty near-identical poses (sections L, X3, AV)."""
        from waypoint_recorder import load_waypoints
        from calibration_utils import camera_paths, pose_to_matrix
        selected = set(yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye" / "selection"
             / "selected_20.yaml").read_text())["selected_numbers"])
        records = [r for r in load_waypoints(
            camera_paths("camera_1", pipeline["data"])) if r.number in selected]
        translations, rotations = [], []
        for index, first in enumerate(records):
            for second in records[index + 1:]:
                translation, rotation = transform_difference(
                    pose_to_matrix(first.actual_tcp), pose_to_matrix(second.actual_tcp))
                translations.append(translation * 1000)
                rotations.append(rotation)
        assert max(translations) > 40.0, "selected set has no translation spread"
        assert max(rotations) > 10.0, "selected set has no rotation spread"

    def test_verification_passes_on_clean_data(self, pipeline):
        report = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye" / "verification"
             / "verification_report.yaml").read_text())
        assert report["passed"], report
        assert report["holdout"]["count"] > 0
        assert report["holdout"]["mean"] < 2.0

    def test_holdout_error_is_comparable_to_fit_error(self, pipeline):
        """If the hold-out is far worse, the fit was overfitted."""
        report = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye" / "verification"
             / "verification_report.yaml").read_text())
        assert report["holdout"]["mean"] < 3.0 * report["fitted"]["mean"] + 0.5

    def test_solve_output_names_the_mounting_mode(self, pipeline):
        stdout = pipeline["out"]["solve"].stdout
        assert pipeline["mode"] in stdout
        assert "HOLD-OUT VALIDATION" in stdout
        assert "RECOMMENDED" in stdout

    def test_comparison_reports_both_candidates(self, pipeline):
        final = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye"
             / "final_result_best_20.yaml").read_text())
        comparison = final["comparison_with_all"]
        assert comparison["recommendation"] in ("ALL-30", "BEST-20")
        assert comparison["reasons"]

    def test_method_cross_check_agrees(self, pipeline):
        final = yaml.safe_load(
            (pipeline["data"] / "camera_1" / "handeye"
             / "final_result_best_20.yaml").read_text())
        spread = final["cross_check"]["spread"]
        assert spread["max_translation_difference_mm"] < 10.0

    def test_replay_defaults_to_dry_run(self, pipeline):
        """Section AH: replay must never move anything by default."""
        result = run_script("replay_waypoints.py", ["--camera", "camera_1"],
                            pipeline["env"])
        assert "DRY RUN" in result.stdout
        assert "nothing was sent" in result.stdout.lower()

    def test_replay_with_enable_motion_still_blocked_by_config(self, pipeline):
        """allow_motion is false, so even --enable-motion must refuse."""
        result = run_script("replay_waypoints.py",
                            ["--camera", "camera_1", "--enable-motion"],
                            pipeline["env"], expect_success=False)
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert ("MOTION IS DISABLED" in combined
                or "REFUSING" in combined), combined


class TestPipelineGuards:
    """Guards that protect the operator from silently wrong results."""

    def test_unverified_board_blocks_the_solve(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("unverified")
        build_dataset(base / "data", "eye_in_hand", count=12)
        build_config(base / "config", "eye_in_hand")
        config = yaml.safe_load((base / "config" / "calibration.yaml").read_text())
        config["apriltag_grid"]["verified_by_user"] = False
        (base / "config" / "calibration.yaml").write_text(yaml.safe_dump(config))
        result = run_script("analyze_waypoints.py", ["--camera", "camera_1"], {
            "CAMERA_CALIBRATION_DATA_DIR": str(base / "data"),
            "CAMERA_CALIBRATION_CONFIG_DIR": str(base / "config"),
            "CAMERA_CALIBRATION_LOG_DIR": str(base / "logs"),
            "PYTHONPATH": str(ROOT / "src")}, expect_success=False)
        assert result.returncode != 0
        assert "not confirmed" in (result.stdout + result.stderr)

    def test_unverified_handeye_mode_blocks_the_solve(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("unverified_mode")
        build_dataset(base / "data", "eye_in_hand", count=12)
        build_config(base / "config", "eye_in_hand")
        config = yaml.safe_load((base / "config" / "calibration.yaml").read_text())
        config["handeye"]["verified_by_user"] = False
        (base / "config" / "calibration.yaml").write_text(yaml.safe_dump(config))
        result = run_script("analyze_waypoints.py", ["--camera", "camera_1"], {
            "CAMERA_CALIBRATION_DATA_DIR": str(base / "data"),
            "CAMERA_CALIBRATION_CONFIG_DIR": str(base / "config"),
            "CAMERA_CALIBRATION_LOG_DIR": str(base / "logs"),
            "PYTHONPATH": str(ROOT / "src")}, expect_success=False)
        assert "mounting is not confirmed" in (result.stdout + result.stderr).lower()

    def test_serial_mismatch_between_cameras_is_refused(self, tmp_path_factory):
        """Section AK/AL: one camera must never use another's data."""
        base = tmp_path_factory.mktemp("serial_mismatch")
        build_dataset(base / "data", "eye_in_hand", count=12)
        build_config(base / "config", "eye_in_hand")
        cameras = yaml.safe_load((base / "config" / "cameras.yaml").read_text())
        cameras["cameras"]["camera_1"]["serial"] = "A_DIFFERENT_CAMERA"
        (base / "config" / "cameras.yaml").write_text(yaml.safe_dump(cameras))
        result = run_script("analyze_waypoints.py", ["--camera", "camera_1"], {
            "CAMERA_CALIBRATION_DATA_DIR": str(base / "data"),
            "CAMERA_CALIBRATION_CONFIG_DIR": str(base / "config"),
            "CAMERA_CALIBRATION_LOG_DIR": str(base / "logs"),
            "PYTHONPATH": str(ROOT / "src")}, expect_success=False)
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "serial" in combined.lower()

    def test_camera_datasets_stay_separate(self, tmp_path_factory):
        """Running camera_1 must not read or write camera_2's tree."""
        base = tmp_path_factory.mktemp("separation")
        build_dataset(base / "data", "eye_in_hand", count=12)
        build_config(base / "config", "eye_in_hand")
        two = base / "data" / "camera_2" / "handeye" / "observations"
        two.mkdir(parents=True, exist_ok=True)
        sentinel = two / "waypoint_001.yaml"
        sentinel.write_text("do not touch\n")
        run_script("analyze_waypoints.py", ["--camera", "camera_1"], {
            "CAMERA_CALIBRATION_DATA_DIR": str(base / "data"),
            "CAMERA_CALIBRATION_CONFIG_DIR": str(base / "config"),
            "CAMERA_CALIBRATION_LOG_DIR": str(base / "logs"),
            "PYTHONPATH": str(ROOT / "src")})
        assert sentinel.read_text() == "do not touch\n"
        assert not (base / "data" / "camera_2" / "handeye"
                    / "preliminary_result_all_30.yaml").exists()
