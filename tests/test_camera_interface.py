"""Camera enumeration and identity resolution. Opens no real camera."""
import numpy as np
import pytest

import camera_interface as ci
from calibration_utils import ConfigError
from camera_interface import (BaseCamera, CameraError, CameraInfo,
                              find_device_for_serial, format_camera_table,
                              open_camera)


def make_info(serial="ABC123", backend="v4l2", device="/dev/video0",
              usable=True, usb_path="", modes=None):
    return CameraInfo(backend=backend, name="Test Cam", serial=serial,
                      device=device, usable=usable, usb_path=usb_path,
                      modes=modes if modes is not None else
                      [{"format": "MJPG", "width": 1280, "height": 720, "fps": 30.0}])


def patch_enumeration(monkeypatch, v4l2=(), realsense=()):
    """Patch BOTH per-backend enumerators.

    find_device_for_serial() calls enumerate_v4l2_cameras() /
    enumerate_realsense_cameras() directly rather than the combined
    enumerate_cameras(), so that a RealSense configured as backend: v4l2 still
    resolves despite enumerate_cameras() hiding its v4l2 shadows. Patching
    both keeps these tests off the real hardware whichever backend is asked
    for -- without it, a test requesting "realsense" would shell out to udev.
    """
    monkeypatch.setattr(ci, "enumerate_v4l2_cameras", lambda: list(v4l2))
    monkeypatch.setattr(ci, "enumerate_realsense_cameras", lambda: list(realsense))


class FakeCamera(BaseCamera):
    """In-memory camera used to exercise BaseCamera's lifecycle rules."""

    def __init__(self, config, frames=None, fail_after=None):
        super().__init__(config)
        self.frames = frames
        self.opens = self.closes = self.reads = 0
        self.fail_after = fail_after

    def _open(self):
        self.opens += 1
        self.info = make_info()

    def _read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return None
        return np.full((480, 640, 3), 40, dtype=np.uint8)

    def _close(self):
        self.closes += 1


BASE_CONFIG = {"name": "camera_1", "serial": "ABC123", "backend": "v4l2",
               "width": 640, "height": 480, "fps": 30,
               "capture": {"warmup_frames": 3, "flush_frames": 2,
                           "read_timeout_s": 0.05}}


class TestCombinedEnumeration:
    """A RealSense must be reported once, not once per /dev/video node."""

    def rs(self, serial, device):
        return CameraInfo(backend="realsense", name="RealSense D435IF",
                          serial=serial, device=device)

    def shadow(self, device):
        """A v4l2 node of a RealSense: named as one, and serial-less."""
        return CameraInfo(backend="v4l2", serial="", device=device,
                          name="Intel(R) RealSense(TM) Depth Ca")

    def test_v4l2_shadows_of_a_realsense_are_hidden(self, monkeypatch):
        monkeypatch.setattr(ci, "enumerate_realsense_cameras",
                            lambda: [self.rs("327122073926", "/dev/video12")])
        monkeypatch.setattr(ci, "enumerate_v4l2_cameras", lambda: [
            self.shadow("/dev/video12"), self.shadow("/dev/video13"),
            self.shadow("/dev/video14"), self.shadow("/dev/video15")])

        found = ci.enumerate_cameras()
        assert [c.backend for c in found] == ["realsense"]
        assert found[0].serial == "327122073926"

    def test_a_real_webcam_is_still_listed(self, monkeypatch):
        """De-duplication must not swallow unrelated v4l2 cameras."""
        monkeypatch.setattr(ci, "enumerate_realsense_cameras",
                            lambda: [self.rs("327122073926", "/dev/video12")])
        monkeypatch.setattr(ci, "enumerate_v4l2_cameras", lambda: [
            self.shadow("/dev/video13"),
            make_info("200901010001", device="/dev/video0")])

        found = ci.enumerate_cameras()
        assert sorted(c.serial for c in found) == ["200901010001", "327122073926"]

    def test_shadows_are_kept_when_the_sdk_finds_nothing(self, monkeypatch):
        """Without pyrealsense2 the v4l2 nodes are the only evidence at all."""
        monkeypatch.setattr(ci, "enumerate_realsense_cameras", lambda: [])
        monkeypatch.setattr(ci, "enumerate_v4l2_cameras",
                            lambda: [self.shadow("/dev/video12")])

        found = ci.enumerate_cameras()
        assert len(found) == 1 and found[0].backend == "v4l2"


class TestSerialResolution:
    """Identity must come from the serial, never from the device index."""

    def test_matches_by_serial(self, monkeypatch):
        patch_enumeration(monkeypatch, v4l2=[
            make_info("AAA", device="/dev/video0"),
            make_info("BBB", device="/dev/video2")])
        assert find_device_for_serial("BBB", "v4l2").device == "/dev/video2"

    def test_device_index_alone_never_matches(self, monkeypatch):
        """The whole point: /dev/videoN is not an identity."""
        patch_enumeration(monkeypatch, v4l2=[make_info("AAA")])
        with pytest.raises(CameraError, match="No v4l2 camera with serial"):
            find_device_for_serial("/dev/video0", "v4l2")

    def test_missing_serial_error_lists_what_is_connected(self, monkeypatch):
        patch_enumeration(monkeypatch, v4l2=[make_info("AAA")])
        with pytest.raises(CameraError, match="AAA"):
            find_device_for_serial("ZZZ", "v4l2")

    def test_placeholder_serial_is_rejected_before_any_hardware_access(self, monkeypatch):
        def explode():
            raise AssertionError("must not enumerate for a placeholder serial")
        monkeypatch.setattr(ci, "enumerate_cameras", explode)
        monkeypatch.setattr(ci, "enumerate_v4l2_cameras", explode)
        monkeypatch.setattr(ci, "enumerate_realsense_cameras", explode)
        with pytest.raises(ConfigError, match="placeholder"):
            find_device_for_serial("REPLACE_WITH_SERIAL", "v4l2")

    def test_backend_is_part_of_the_identity(self, monkeypatch):
        patch_enumeration(monkeypatch, v4l2=[make_info("SHARED", backend="v4l2")])
        with pytest.raises(CameraError):
            find_device_for_serial("SHARED", "realsense")

    def test_unusable_nodes_are_not_matched(self, monkeypatch):
        """A metadata-only node must never be chosen as the capture device."""
        patch_enumeration(monkeypatch, v4l2=[
            make_info("AAA", device="/dev/video1", usable=False, modes=[]),
            make_info("AAA", device="/dev/video0")])
        assert find_device_for_serial("AAA", "v4l2").device == "/dev/video0"

    def test_usb_path_is_the_fallback_for_serial_less_cameras(self, monkeypatch):
        patch_enumeration(monkeypatch, v4l2=[
            make_info("", usb_path="pci-0000:00:14.0-usb-0:3:1.0")])
        found = find_device_for_serial("pci-0000:00:14.0-usb-0:3:1.0", "v4l2")
        assert found.device == "/dev/video0"

    def test_multiple_nodes_of_one_camera_pick_the_lowest(self, monkeypatch):
        patch_enumeration(monkeypatch, v4l2=[
            make_info("AAA", device="/dev/video4"),
            make_info("AAA", device="/dev/video2")])
        assert find_device_for_serial("AAA", "v4l2").device == "/dev/video2"


class TestLifecycle:
    def test_open_warms_up_then_read_flushes(self):
        camera = FakeCamera(BASE_CONFIG)
        camera.open()
        assert camera.opens == 1
        assert camera.reads == 3            # warmup_frames
        camera.read()
        assert camera.reads == 3 + 2 + 1    # flush + the returned frame

    def test_read_before_open_is_an_error(self):
        with pytest.raises(CameraError, match="not open"):
            FakeCamera(BASE_CONFIG).read()

    def test_flush_override_is_respected(self):
        camera = FakeCamera(BASE_CONFIG).open()
        before = camera.reads
        camera.read(flush=0)
        assert camera.reads == before + 1

    def test_context_manager_closes_on_exception(self):
        camera = FakeCamera(BASE_CONFIG)
        with pytest.raises(RuntimeError):
            with camera:
                raise RuntimeError("boom")
        assert camera.closes == 1

    def test_double_open_does_not_reopen(self):
        camera = FakeCamera(BASE_CONFIG).open()
        camera.open()
        assert camera.opens == 1

    def test_close_is_idempotent(self):
        camera = FakeCamera(BASE_CONFIG).open()
        camera.close()
        camera.close()
        assert camera.closes == 1

    def test_dead_camera_raises_rather_than_returning_a_stale_frame(self):
        camera = FakeCamera(BASE_CONFIG, fail_after=3).open()
        with pytest.raises(CameraError, match="no frame within"):
            camera.read()

    def test_failed_warmup_closes_the_device(self):
        camera = FakeCamera(BASE_CONFIG, fail_after=0)
        with pytest.raises(CameraError):
            camera.open()
        assert camera.closes == 1
        assert not camera.is_open

    def test_frame_reports_its_own_resolution(self):
        frame = FakeCamera(BASE_CONFIG).open().read()
        assert frame.resolution == (640, 480)
        assert frame.index >= 1

    def test_describe_records_provenance(self):
        described = FakeCamera(BASE_CONFIG).open().describe()
        assert described["camera_name"] == "camera_1"
        assert described["camera_serial"] == "ABC123"
        assert described["actual_resolution"] == [640, 480]


class TestOpenCamera:
    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown camera backend"):
            open_camera({"name": "camera_1", "backend": "gopro", "serial": "1"})

    def test_realsense_backend_reports_the_missing_dependency_clearly(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyrealsense2":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(CameraError, match="pyrealsense2 is not installed"):
            open_camera({**BASE_CONFIG, "backend": "realsense"})


class TestEnumerationHelpers:
    def test_best_mode_prefers_resolution_then_fps(self):
        info = make_info(modes=[
            {"format": "MJPG", "width": 640, "height": 480, "fps": 60.0},
            {"format": "MJPG", "width": 1280, "height": 720, "fps": 15.0},
            {"format": "MJPG", "width": 1280, "height": 720, "fps": 30.0}])
        best = info.best_mode()
        assert (best["width"], best["fps"]) == (1280, 30.0)

    def test_best_mode_is_none_without_modes(self):
        assert make_info(modes=[]).best_mode() is None

    def test_table_renders_empty_and_populated(self):
        assert "No cameras" in format_camera_table([])
        table = format_camera_table([make_info("XYZ")])
        assert "XYZ" in table and "1280x720" in table

    def test_realsense_enumeration_is_empty_without_the_binding(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyrealsense2":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert ci.enumerate_realsense_cameras() == []
