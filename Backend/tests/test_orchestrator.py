"""
Tests for the Orchestrator — integration test with all sub-components mocked.
Verifies config wiring and initialization logic.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from core.stream_manager import CameraConfig


@pytest.fixture
def mock_all_components():
    """Mock all heavy ML components so Orchestrator can be instantiated."""
    with patch("core.orchestrator.SecurityDetector") as mock_det, \
         patch("core.orchestrator.ReIDEncoder") as mock_reid, \
         patch("core.orchestrator.ANPREngine") as mock_anpr, \
         patch("core.orchestrator.FaceAuthEngine") as mock_face, \
         patch("core.orchestrator.GlobalMatcher") as mock_gm, \
         patch("core.orchestrator.StreamManager") as mock_sm, \
         patch("core.orchestrator.SingleCameraTracker") as mock_tracker, \
         patch("core.orchestrator.RiskAnalyzer") as mock_risk, \
         patch("core.orchestrator.cv2") as mock_cv2:

        # Make _probe_resolution return a known value
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 1920, 4: 1080}.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

        yield {
            "detector": mock_det,
            "reid": mock_reid,
            "anpr": mock_anpr,
            "face": mock_face,
            "global_matcher": mock_gm,
            "stream_manager": mock_sm,
            "tracker": mock_tracker,
            "risk": mock_risk,
        }


class TestOrchestratorInit:
    def test_init_without_config(self, mock_all_components):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        orch = Orchestrator(cameras=cameras, device="cpu", display=False)
        assert orch.device == "cpu"
        assert orch.display is False

    def test_init_with_config(self, mock_all_components, sample_config):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        orch = Orchestrator(cameras=cameras, device="cpu", display=False, config=sample_config)
        assert orch.config == sample_config

    def test_detector_receives_config(self, mock_all_components, sample_config):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        Orchestrator(cameras=cameras, device="cpu", display=False, config=sample_config)

        det_call = mock_all_components["detector"].call_args
        assert det_call[1]["imgsz"] == 1280
        assert det_call[1]["high_conf"] == 0.40
        assert det_call[1]["low_conf"] == 0.07
        assert det_call[1]["nms_iou"] == 0.75

    def test_reid_receives_config(self, mock_all_components, sample_config):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        Orchestrator(cameras=cameras, device="cpu", display=False, config=sample_config)

        reid_call = mock_all_components["reid"].call_args
        assert reid_call[1]["model_name"] == "osnet_x1_0"
        assert reid_call[1]["input_width"] == 128
        assert reid_call[1]["input_height"] == 256

    def test_tracker_receives_config(self, mock_all_components, sample_config):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        Orchestrator(cameras=cameras, device="cpu", display=False, config=sample_config)

        tracker_call = mock_all_components["tracker"].call_args
        assert tracker_call[1]["max_miss"] == 25
        assert tracker_call[1]["lost_buffer_sec"] == 5.0
        assert tracker_call[1]["embedding_pool_k"] == 10
        assert tracker_call[1]["sim_thr_live"] == 0.45
        assert tracker_call[1]["sim_thr_lost"] == 0.40

    def test_global_matcher_receives_config(self, mock_all_components, sample_config):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        Orchestrator(cameras=cameras, device="cpu", display=False, config=sample_config)

        gm_call = mock_all_components["global_matcher"].call_args
        assert gm_call[1]["gallery_threshold"] == 0.70
        assert gm_call[1]["margin_gate"] == 0.08
        assert gm_call[1]["ema_alpha"] == 0.10

    def test_multiple_cameras(self, mock_all_components):
        from core.orchestrator import Orchestrator
        cameras = [
            CameraConfig(camera_id="cam1", source="v1.mp4"),
            CameraConfig(camera_id="cam2", source="v2.mp4"),
        ]
        orch = Orchestrator(cameras=cameras, device="cpu", display=False)
        # Tracker and RiskAnalyzer should be created for each camera
        assert mock_all_components["tracker"].call_count == 2
        assert mock_all_components["risk"].call_count == 2

    def test_default_config_used_when_none(self, mock_all_components):
        from core.orchestrator import Orchestrator
        cameras = [CameraConfig(camera_id="cam1", source="test.mp4")]
        orch = Orchestrator(cameras=cameras, device="cpu", display=False, config=None)

        det_call = mock_all_components["detector"].call_args
        assert det_call[1]["imgsz"] == 640  # Default, not config value
        assert det_call[1]["low_conf"] == 0.10
