"""
Tests for StreamManager and CameraConfig.
Tests the data structures and manager logic — does NOT open real cameras.
"""
import time

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.stream_manager import CameraConfig, CameraStream, StreamManager


class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig(camera_id="cam1", source="test.mp4")
        assert cfg.camera_id == "cam1"
        assert cfg.source == "test.mp4"
        assert cfg.name == "Camera"
        assert cfg.gps_lat == 0.0
        assert cfg.gps_lng == 0.0

    def test_full_config(self):
        cfg = CameraConfig(
            camera_id="cam2", source="rtsp://192.168.1.1/stream",
            name="Main Gate", gps_lat=28.61, gps_lng=77.20,
        )
        assert cfg.name == "Main Gate"
        assert cfg.gps_lat == pytest.approx(28.61)

    def test_webcam_source(self):
        cfg = CameraConfig(camera_id="cam0", source=0)
        assert cfg.source == 0


class TestCameraStream:
    def test_init_video_file(self):
        cfg = CameraConfig(camera_id="cam1", source="video.mp4")
        stream = CameraStream(cfg)
        assert stream.camera_id == "cam1"
        assert stream._is_video_file is True
        assert stream._is_rtsp is False
        assert stream.is_running is False

    def test_init_rtsp(self):
        cfg = CameraConfig(camera_id="cam2", source="rtsp://192.168.1.1/stream")
        stream = CameraStream(cfg)
        assert stream._is_rtsp is True
        assert stream._is_video_file is False

    def test_init_webcam(self):
        cfg = CameraConfig(camera_id="cam0", source=0)
        stream = CameraStream(cfg)
        assert stream._is_video_file is False
        assert stream._is_rtsp is False

    def test_read_returns_none_initially(self):
        cfg = CameraConfig(camera_id="cam1", source="video.mp4")
        stream = CameraStream(cfg)
        assert stream.read() is None

    def test_stop_sets_running_false(self):
        cfg = CameraConfig(camera_id="cam1", source="video.mp4")
        stream = CameraStream(cfg)
        stream._running = True
        stream.stop()
        assert stream._running is False

    def test_frame_size_default(self):
        cfg = CameraConfig(camera_id="cam1", source="video.mp4")
        stream = CameraStream(cfg)
        w, h = stream.frame_size
        assert w == 0
        assert h == 0


class TestStreamManager:
    def test_add_camera(self):
        sm = StreamManager()
        cfg = CameraConfig(camera_id="cam1", source="test.mp4")
        sm.add_camera(cfg)
        assert "cam1" in sm.streams
        assert "cam1" in sm.configs
        assert sm.camera_count == 1

    def test_add_multiple_cameras(self):
        sm = StreamManager()
        for i in range(3):
            sm.add_camera(CameraConfig(camera_id=f"cam{i}", source=f"v{i}.mp4"))
        assert sm.camera_count == 3
        assert sorted(sm.get_camera_ids()) == ["cam0", "cam1", "cam2"]

    def test_get_config(self):
        sm = StreamManager()
        cfg = CameraConfig(camera_id="cam1", source="test.mp4", name="Test Camera")
        sm.add_camera(cfg)
        retrieved = sm.get_config("cam1")
        assert retrieved.name == "Test Camera"

    def test_get_config_nonexistent(self):
        sm = StreamManager()
        assert sm.get_config("nonexistent") is None

    def test_grab_latest_empty(self):
        sm = StreamManager()
        sm.add_camera(CameraConfig(camera_id="cam1", source="test.mp4"))
        frames = sm.grab_latest()
        assert frames == {}

    def test_all_stopped_initially(self):
        sm = StreamManager()
        sm.add_camera(CameraConfig(camera_id="cam1", source="test.mp4"))
        assert sm.all_stopped() is True


class TestNightVision:
    dark = np.full((360, 640, 3), 25, dtype=np.uint8)
    bright = np.full((360, 640, 3), 170, dtype=np.uint8)

    def test_auto_mode_dark_scene_at_noon(self):
        s = CameraStream(CameraConfig(camera_id="c", source="x.mp4"))
        assert s.evaluate_night_mode(self.dark, 0, hour=12) is True
        assert s.is_night is True

    def test_auto_mode_bright_scene(self):
        s = CameraStream(CameraConfig(camera_id="c", source="x.mp4"))
        assert s.evaluate_night_mode(self.bright, 0, hour=12) is False
        assert s.evaluate_night_mode(self.bright, 0, hour=23) is True   # clock window

    def test_luminance_hysteresis(self):
        s = CameraStream(CameraConfig(camera_id="c", source="x.mp4"), night_vision={"mode": "luminance"})
        assert s.evaluate_night_mode(self.dark, 0) is True
        mid = np.full((360, 640, 3), 80, dtype=np.uint8)   # between enter (70) and exit (90)
        assert s.evaluate_night_mode(mid, 15) is True
        assert s.evaluate_night_mode(self.bright, 30) is False

    def test_clock_and_off_modes(self):
        s = CameraStream(CameraConfig(camera_id="c", source="x.mp4"), night_vision={"mode": "clock"})
        assert s.evaluate_night_mode(self.dark, 0, hour=12) is False
        s = CameraStream(CameraConfig(camera_id="c", source="x.mp4"), night_vision={"mode": "off"})
        assert s.evaluate_night_mode(self.dark, 0, hour=2) is False

    def test_enhancement_brightens_dark_frames(self):
        s = CameraStream(CameraConfig(camera_id="c", source="x.mp4"), night_vision={"mode": "luminance"})
        rng = np.random.RandomState(0)
        dark = rng.randint(0, 50, (360, 640, 3)).astype(np.uint8)
        out = s._enhance_night_vision(dark)
        assert out.std() > dark.std()

    def test_manager_passes_settings_and_restart(self):
        sm = StreamManager(night_vision={"mode": "off"})
        sm.add_camera(CameraConfig(camera_id="cam1", source="missing.mp4"))
        assert sm.streams["cam1"].night_cfg["mode"] == "off"
        assert sm.is_night("cam1") is False
        assert sm.is_night("nope") is False
        assert sm.restart_camera("nope") is False


def test_per_camera_night_override():
    s = CameraStream(CameraConfig(camera_id="c", source="x.mp4", night_mode="on"))
    assert s.evaluate_night_mode(np.full((360, 640, 3), 200, dtype=np.uint8), 0, hour=12) is True
    s = CameraStream(CameraConfig(camera_id="c", source="x.mp4", night_mode="off"))
    assert s.evaluate_night_mode(np.full((360, 640, 3), 5, dtype=np.uint8), 0, hour=2) is False


class TestLiveSources:
    """Phone apps (IP Webcam: http://<ip>:8080/video), webcams and RTSP are live sources."""

    @pytest.mark.parametrize("src", ["http://192.168.1.5:8080/video", "https://cam.local/stream",
                                     "rtsp://192.168.1.1/stream", "rtmp://x/live", "HTTP://10.0.0.2:8080/video"])
    def test_network_urls_are_live(self, src):
        s = CameraStream(CameraConfig(camera_id="p", source=src))
        assert s._is_network and s._is_live and not s._is_video_file

    @pytest.mark.parametrize("src", [0, 1, "0", " 1 "])
    def test_webcam_index_int_or_string(self, src):
        s = CameraStream(CameraConfig(camera_id="w", source=src))
        assert s._is_webcam and s._is_live and isinstance(s.source, int)

    def test_file_is_not_live(self):
        s = CameraStream(CameraConfig(camera_id="f", source="videos/a.mp4"))
        assert s._is_video_file and not s._is_live

    def test_live_frame_handed_out_once(self):
        s = CameraStream(CameraConfig(camera_id="p", source="http://x/video"))
        frame = np.zeros((4, 4, 3), np.uint8)
        with s._lock:
            s._latest, s._fresh = (frame, 1.0), True
        assert s.read() is not None
        assert s.read() is None            # same frame is not re-analysed
        with s._lock:
            s._latest, s._fresh = (frame, 2.0), True
        assert s.read()[1] == 2.0

    def test_live_source_retries_instead_of_exiting(self):
        s = CameraStream(CameraConfig(camera_id="p", source="http://127.0.0.1:9/video"), reconnect_delay=0.05)
        attempts = []
        s._open_capture = lambda: attempts.append(1) or False
        s.start()
        time.sleep(0.4)
        assert s.is_running and len(attempts) >= 3
        s.stop(); s.join(timeout=1)


def test_video_file_loops_when_configured(tmp_path):
    import cv2
    path = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for _ in range(3):
        w.write(np.zeros((32, 32, 3), np.uint8))
    w.release()
    s = CameraStream(CameraConfig(camera_id="v", source=path, loop=True))
    s.start()
    got, t = 0, time.time()
    while got < 8 and time.time() - t < 5:
        if s.read() is not None:
            got += 1
        time.sleep(0.005)
    assert got >= 8 and s.is_running   # 3-frame clip replayed past its end
    s.stop(); s.join(2)


@pytest.mark.parametrize("rot,expected", [(0, (40, 20)), (90, (20, 40)), (180, (40, 20)), (270, (20, 40))])
def test_rotate_option(tmp_path, rot, expected):
    import cv2
    path = str(tmp_path / "r.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 20))
    frame = np.zeros((20, 40, 3), np.uint8); frame[0, 0] = 255  # marker top-left
    for _ in range(2):
        w.write(frame)
    w.release()
    s = CameraStream(CameraConfig(camera_id="r", source=path, rotate=rot))
    s.start()
    t, got = time.time(), None
    while got is None and time.time() - t < 3:
        got = s.read(); time.sleep(0.005)
    s.stop(); s.join(2)
    assert (got[0].shape[1], got[0].shape[0]) == expected
    assert s.frame_size == expected


def _write_clip(path, n, fps=30):
    import cv2
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 32))
    for i in range(n):
        w.write(np.full((32, 32, 3), i % 255, np.uint8))
    w.release()


def test_realtime_file_plays_at_native_speed_and_drops(tmp_path):
    path = str(tmp_path / "rt.avi"); _write_clip(path, 30, fps=30)   # 1 s of video
    s = CameraStream(CameraConfig(camera_id="v", source=path, realtime=True))
    t = time.time(); s.start(); s.join(timeout=5)
    assert 0.8 <= time.time() - t <= 2.0          # paced at ~30 fps, nobody consumed frames
    assert s.latest_display()[2] == 30


def test_display_copy_is_not_consumed(tmp_path):
    path = str(tmp_path / "d.avi"); _write_clip(path, 3)
    s = CameraStream(CameraConfig(camera_id="v", source=path))
    s.start()
    t = time.time()
    while s.latest_display() is None and time.time() - t < 3:
        time.sleep(0.005)
    assert s.read() is not None                    # analysis consumes...
    assert s.latest_display() is not None          # ...the display copy stays
    s.stop(); s.join(2)
