"""
Tests for StreamManager and CameraConfig.
Tests the data structures and manager logic — does NOT open real cameras.
"""
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
