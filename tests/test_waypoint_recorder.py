"""Tests for the section-P record procedure and the waypoint record format.

No camera and no robot are opened here. The camera, detector and robot are
fakes whose whole purpose is to fail on demand, because almost everything
worth testing in this module is a *refusal*: the recorder's job is to write a
waypoint only when the board was valid and the arm was genuinely still, and to
leave nothing behind when it was not.

The failure that motivates most of this file is silent. An image saved while
the arm was still ringing pairs a sharp, self-consistent board detection with
the wrong joint angles; every downstream reprojection check still passes and
the hand-eye result is simply wrong. So the checks are asserted here rather
than trusted to show up later.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apriltag_detector import Detection  # noqa: E402
from calibration_utils import (JOINT_NAMES, camera_paths,  # noqa: E402
                               invert_transform, load_yaml, save_yaml)
from camera_interface import Frame  # noqa: E402
from robot_interface import RobotState  # noqa: E402
from waypoint_recorder import (RecordingRejected, WaypointRecord,  # noqa: E402
                               WaypointRecorder, load_waypoints,
                               save_initial_center)

CENTER_Q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
CENTER_TCP = np.array([0.40, 0.10, 0.30, 0.0, 3.14, 0.0])


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeCamera:
    """Records how it was read, so flushing behaviour can be asserted."""

    def __init__(self, name="camera_1", serial="SYNTH0001"):
        self.name = name
        self.serial = serial
        self.reads: list[int | None] = []
        self.index = 0
        self.fail_on_read = False

    def read(self, flush=None):
        if self.fail_on_read:
            raise RuntimeError("camera exploded")
        self.reads.append(flush)
        self.index += 1
        # Encode the read number in the pixels so the saved image can be
        # traced back to the read that produced it.
        image = np.full((48, 64, 3), self.index, dtype=np.uint8)
        return Frame(image=image, timestamp=100.0 + self.index, index=self.index)

    def describe(self):
        return {"camera_name": self.name, "camera_serial": self.serial,
                "backend": "fake"}


def detection(valid=True, reasons=(), with_pose=True, tags=12):
    ids = np.arange(tags, dtype=np.int32)
    corners = np.tile(np.array([[10.0, 10.0], [20.0, 10.0],
                                [20.0, 20.0], [10.0, 20.0]]), (tags, 1, 1))
    result = Detection(
        ids=ids, corners=corners, image_size=(64, 48),
        tags_detected=tags, corners_detected=tags * 4,
        border_margin_px=15.0, visible_fraction=1.0, clipped_fraction=0.0,
        board_area_fraction=0.25, sharpness=120.0, center_px=(32.0, 24.0),
        valid=valid, reasons=list(reasons))
    if with_pose:
        result.rvec = np.array([0.10, -0.20, 0.30])
        result.tvec = np.array([0.02, -0.01, 0.45])
        result.pnp_reprojection_px = 0.21
        result.pnp_max_reprojection_px = 0.60
        result.object_points = np.random.default_rng(3).normal(
            0, 0.05, (tags * 4, 3))
        result.image_points = np.random.default_rng(4).uniform(
            0, 64, (tags * 4, 2))
    return result


class FakeDetector:
    """Returns a queued Detection per call, repeating the last one."""

    def __init__(self, results=None):
        self.results = list(results) if results else [detection()]
        self.calls = 0

    def process(self, image, camera_matrix, dist_coeffs, require_pose=False):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def state(qd=0.0, connected=True, q=None, tcp=None):
    return RobotState(
        connected=connected,
        q=np.asarray(CENTER_Q if q is None else q, dtype=np.float64),
        qd=np.full(6, qd, dtype=np.float64),
        tcp=np.asarray(CENTER_TCP if tcp is None else tcp, dtype=np.float64),
        robot_mode=7, safety_mode=1)


class FakeRobot:
    """Replays a queue of RobotStates, repeating the last one."""

    def __init__(self, states=None):
        self.states = list(states) if states else [state()]
        self.calls = 0

    def read_state(self):
        result = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return result

    def deltas_from_center(self, robot_state):
        delta = robot_state.q - CENTER_Q
        return {
            "joint_delta_rad": delta.tolist(),
            "joint_delta_deg": np.degrees(delta).tolist(),
            "tcp_translation_delta_mm": 12.5,
            "tcp_rotation_delta_deg": 7.5,
            "dominant_joint_change": JOINT_NAMES[int(np.argmax(np.abs(delta)))],
        }


CONFIG = {"waypoint_collection": {
    "settle_time_s": 0.0,
    "stationary_velocity_threshold": 0.005,
    "stationary_consecutive_samples": 3,
    "stationary_sample_interval_s": 0.0,
    "stationary_timeout_s": 0.5,
}}


@pytest.fixture
def paths(tmp_path):
    result = camera_paths("camera_1", data_root=tmp_path)
    result.ensure()
    return result


def make_recorder(paths, camera=None, detector=None, robot=None, config=None):
    return WaypointRecorder(
        paths=paths,
        camera=camera or FakeCamera(),
        detector=detector or FakeDetector(),
        robot=robot or FakeRobot(),
        config=config or CONFIG,
        camera_matrix=np.array([[900.0, 0, 32.0], [0, 900.0, 24.0], [0, 0, 1.0]]),
        dist_coeffs=np.zeros((1, 5)),
        intrinsics_info={"timestamp": "2026-09-14T00:00:00+00:00"})


def written(paths):
    """(image names, observation names) currently on disk."""
    return (sorted(p.name for p in paths.handeye_images.glob("*.png")),
            sorted(p.name for p in paths.handeye_observations.glob("*.yaml")))


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_record_writes_image_and_observation(paths):
    recorder = make_recorder(paths)
    record = recorder.record(1)

    assert record.name == "waypoint_001"
    assert written(paths) == (["waypoint_001.png"], ["waypoint_001.yaml"])
    assert recorder.records == [record]


def test_saved_observation_holds_actual_measured_state(paths):
    measured_q = CENTER_Q + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25])
    measured_tcp = CENTER_TCP + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot = FakeRobot([state(q=measured_q, tcp=measured_tcp)])
    make_recorder(paths, robot=robot).record(4)

    data = load_yaml(paths.handeye_observations / "waypoint_004.yaml")
    assert data["actual_joints_rad"] == pytest.approx(measured_q.tolist())
    assert data["actual_tcp_vector"] == pytest.approx(measured_tcp.tolist())
    assert data["actual_joints"]["wrist_3"] == pytest.approx(measured_q[5])
    assert data["actual_joints_deg"][5] == pytest.approx(np.degrees(measured_q[5]))
    assert data["dominant_joint_change"] == "wrist_3"
    assert data["robot_mode"] == 7 and data["safety_mode"] == 1


def test_saved_observation_labels_the_rotation_convention(paths):
    """rx/ry/rz are axis-angle. A reader who assumes rpy gets silent nonsense."""
    make_recorder(paths).record(1)
    data = load_yaml(paths.handeye_observations / "waypoint_001.yaml")
    assert "NOT rpy" in data["actual_tcp"]["rotation_representation"]
    assert "axis-angle" in data["board_pose_camera"]["frame"]


def test_saved_settle_time_is_the_one_actually_waited(paths):
    config = {"waypoint_collection": dict(CONFIG["waypoint_collection"],
                                          settle_time_s=0.0)}
    recorder = make_recorder(paths, config=config)
    recorder.settle_time_s = 0.4          # as if configured to 0.4 s
    recorder.record(1)
    data = load_yaml(paths.handeye_observations / "waypoint_001.yaml")
    assert data["settle_time_s"] == pytest.approx(0.4)


def test_capture_flushes_but_preview_does_not(paths):
    """The saved frame must be flushed; a buffered frame shows the old pose."""
    camera = FakeCamera()
    make_recorder(paths, camera=camera).record(1)

    assert camera.reads[0] == 0            # preview: cheap, no flush
    assert camera.reads[-1] is None        # capture: camera's own flush count
    image = cv2.imread(str(paths.handeye_images / "waypoint_001.png"))
    assert int(image[0, 0, 0]) == camera.index   # the LAST frame read, not the first


def test_detection_is_rechecked_on_the_frame_that_gets_saved(paths):
    """Preview validity does not transfer to a different, later frame."""
    detector = FakeDetector([detection(valid=True),
                             detection(valid=False, reasons=["board clipped"])])
    with pytest.raises(RecordingRejected, match="post-capture"):
        make_recorder(paths, detector=detector).record(1)
    assert written(paths) == ([], [])


# --------------------------------------------------------------------------
# Refusals. Every one of these must write nothing.
# --------------------------------------------------------------------------

def test_invalid_board_is_rejected_before_the_robot_is_read(paths):
    robot = FakeRobot()
    detector = FakeDetector([detection(valid=False, reasons=["only 4 tags"])])
    with pytest.raises(RecordingRejected, match="only 4 tags"):
        make_recorder(paths, detector=detector, robot=robot).record(1)
    assert robot.calls == 0
    assert written(paths) == ([], [])


def test_moving_robot_is_rejected_and_reports_the_speed(paths):
    robot = FakeRobot([state(qd=0.2)])
    with pytest.raises(RecordingRejected, match="not stationary"):
        make_recorder(paths, robot=robot).record(1)
    assert written(paths) == ([], [])


def test_stationary_needs_consecutive_quiet_samples(paths):
    """One quiet sample proves nothing: velocity passes through zero on reversal."""
    robot = FakeRobot([state(qd=0.0), state(qd=0.2),
                       state(qd=0.0), state(qd=0.2), state(qd=0.2)])
    with pytest.raises(RecordingRejected, match="not stationary"):
        make_recorder(paths, robot=robot).record(1)

    calm = FakeRobot([state(qd=0.0), state(qd=0.2)] + [state(qd=0.0)] * 5)
    make_recorder(paths, robot=calm).record(1)
    assert written(paths) == (["waypoint_001.png"], ["waypoint_001.yaml"])


def test_movement_during_the_capture_is_rejected(paths):
    """Quiet through the settle check, then moving when the shutter opened."""
    robot = FakeRobot([state(qd=0.0)] * 3 + [state(qd=0.4)])
    with pytest.raises(RecordingRejected, match="moved during capture"):
        make_recorder(paths, robot=robot).record(1)
    assert written(paths) == ([], [])


def test_lost_connection_while_settling_is_rejected(paths):
    robot = FakeRobot([state(qd=0.0), RobotState(connected=False)])
    with pytest.raises(RecordingRejected, match="connection lost"):
        make_recorder(paths, robot=robot).record(1)
    assert written(paths) == ([], [])


def test_lost_connection_during_capture_is_rejected(paths):
    robot = FakeRobot([state(qd=0.0)] * 3 + [RobotState(connected=False)])
    with pytest.raises(RecordingRejected, match="lost the robot connection"):
        make_recorder(paths, robot=robot).record(1)
    assert written(paths) == ([], [])


def test_missing_board_pose_is_rejected(paths):
    detector = FakeDetector([detection(), detection(with_pose=False)])
    with pytest.raises(RecordingRejected, match="no board pose"):
        make_recorder(paths, detector=detector).record(1)
    assert written(paths) == ([], [])


def test_metadata_failure_removes_the_orphan_image(paths, monkeypatch):
    """An image with no metadata looks recorded to a human, invisible to loaders."""
    import waypoint_recorder

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(waypoint_recorder, "save_yaml", explode)
    with pytest.raises(OSError):
        make_recorder(paths).record(1)
    assert written(paths) == ([], [])


def test_a_rejected_attempt_does_not_consume_the_number(paths):
    detector = FakeDetector([detection(valid=False, reasons=["too dark"]),
                             detection()])
    recorder = make_recorder(paths, detector=detector)
    with pytest.raises(RecordingRejected):
        recorder.record(1)
    assert recorder.records == []
    recorder.record(1)
    assert [r.name for r in recorder.records] == ["waypoint_001"]


# --------------------------------------------------------------------------
# Undo and the master list
# --------------------------------------------------------------------------

def test_undo_removes_image_and_metadata_together(paths):
    recorder = make_recorder(paths)
    recorder.record(1)
    recorder.record(2)

    assert recorder.undo_last() == "waypoint_002"
    assert written(paths) == (["waypoint_001.png"], ["waypoint_001.yaml"])
    assert [r.number for r in recorder.records] == [1]


def test_undo_with_nothing_recorded_returns_none(paths):
    assert make_recorder(paths).undo_last() is None


def test_save_master_lists_every_record(paths):
    recorder = make_recorder(paths)
    recorder.record(1)
    recorder.record(2)
    path = recorder.save_master(CENTER_Q, CENTER_TCP, extra={"mode": "eye_in_hand"})

    data = load_yaml(path)
    assert data["count"] == 2
    assert data["camera_serial"] == "SYNTH0001"
    assert data["mode"] == "eye_in_hand"
    assert data["calibration_center_q"] == pytest.approx(CENTER_Q.tolist())
    assert [w["name"] for w in data["waypoints"]] == ["waypoint_001", "waypoint_002"]
    for entry in data["waypoints"]:
        assert (paths.waypoints_file.parent / entry["image"]).resolve().exists()
        assert (paths.waypoints_file.parent / entry["observation"]).resolve().exists()


def test_save_master_tolerates_no_center(paths):
    data = load_yaml(make_recorder(paths).save_master(None, None))
    assert data["calibration_center_q"] is None
    assert data["count"] == 0


# --------------------------------------------------------------------------
# The record format itself
# --------------------------------------------------------------------------

def test_record_round_trips_through_yaml(paths):
    original = make_recorder(paths).record(7)
    restored = WaypointRecord.from_dict(
        load_yaml(paths.handeye_observations / "waypoint_007.yaml"))

    assert restored.number == original.number
    assert restored.name == "waypoint_007"
    assert restored.camera_serial == original.camera_serial
    assert restored.actual_q == pytest.approx(original.actual_q)
    assert restored.actual_tcp == pytest.approx(original.actual_tcp)
    assert restored.actual_qd == pytest.approx(original.actual_qd)
    assert restored.tag_ids == original.tag_ids
    assert restored.object_points == pytest.approx(original.object_points)
    assert restored.image_points == pytest.approx(original.image_points)
    assert restored.board_transform == pytest.approx(original.board_transform)
    assert restored.dominant_joint_change == original.dominant_joint_change


def test_board_transform_matches_the_stored_rvec_tvec(paths):
    record = make_recorder(paths).record(1)
    transform = record.board_transform
    expected, _ = cv2.Rodrigues(np.array([0.10, -0.20, 0.30]).reshape(3, 1))

    assert transform.shape == (4, 4)
    assert transform[:3, :3] == pytest.approx(expected)
    assert transform[:3, 3] == pytest.approx([0.02, -0.01, 0.45])
    assert transform[3] == pytest.approx([0, 0, 0, 1])
    # It is a real rigid transform, so inverting it is exact.
    assert invert_transform(transform) @ transform == pytest.approx(np.eye(4), abs=1e-12)


def test_board_transform_is_none_without_a_pose():
    record = WaypointRecord(
        number=1, camera_name="camera_1", camera_serial="S", timestamp="",
        image_name="", actual_q=np.zeros(6), actual_tcp=np.zeros(6),
        actual_qd=np.zeros(6), detection={}, tag_ids=[],
        tag_corners=np.zeros((0, 4, 2)), object_points=np.zeros((0, 3)),
        image_points=np.zeros((0, 2)))
    assert record.board_transform is None


def test_tcp_transform_matches_the_ur_pose(paths):
    record = make_recorder(paths).record(1)
    transform = record.tcp_transform
    expected, _ = cv2.Rodrigues(CENTER_TCP[3:].reshape(3, 1))
    assert transform[:3, :3] == pytest.approx(expected)
    assert transform[:3, 3] == pytest.approx(CENTER_TCP[:3])


# --------------------------------------------------------------------------
# Loading a stored session
# --------------------------------------------------------------------------

def test_load_waypoints_returns_them_in_numeric_order(paths):
    recorder = make_recorder(paths)
    for number in (10, 2, 1):
        recorder.record(number)

    loaded = load_waypoints(paths)
    assert [r.number for r in loaded] == [1, 2, 10]


def test_load_waypoints_on_an_empty_session(paths):
    assert load_waypoints(paths) == []


def test_load_waypoints_refuses_to_skip_a_corrupt_file(paths):
    """Silently dropping a waypoint would shrink the dataset without anyone noticing."""
    recorder = make_recorder(paths)
    recorder.record(1)
    recorder.record(2)
    (paths.handeye_observations / "waypoint_002.yaml").write_text("not: [valid")

    with pytest.raises(RuntimeError, match="waypoint_002"):
        load_waypoints(paths)


def test_load_waypoints_ignores_unrelated_files(paths):
    make_recorder(paths).record(1)
    save_yaml(paths.handeye_observations / "notes.yaml", {"anything": 1})
    assert [r.number for r in load_waypoints(paths)] == [1]


# --------------------------------------------------------------------------
# The session centre
# --------------------------------------------------------------------------

def test_save_initial_center_records_named_and_raw_state(paths):
    path = save_initial_center(paths, FakeCamera(), state())
    data = load_yaml(path)

    assert data["camera"] == "camera_1"
    assert data["actual_q_rad"] == pytest.approx(CENTER_Q.tolist())
    assert data["actual_q_deg"] == pytest.approx(np.degrees(CENTER_Q).tolist())
    assert set(data["actual_q_named"]) == set(JOINT_NAMES)
    assert data["actual_q_named"]["shoulder"] == pytest.approx(CENTER_Q[1])
    assert "NOT rpy" in data["actual_tcp_named"]["rotation_representation"]
    assert data["environment"]


def test_save_initial_center_includes_the_safety_envelope(paths):
    class FakeEnvelope:
        def describe(self):
            return {"allow_motion": False}

    data = load_yaml(save_initial_center(paths, FakeCamera(), state(),
                                         envelope=FakeEnvelope()))
    assert data["safety"] == {"allow_motion": False}
