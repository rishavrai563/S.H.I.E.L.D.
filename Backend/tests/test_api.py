"""
Tests for the FastAPI SHIELD API.
Uses FastAPI TestClient with the entire ML pipeline mocked.
"""
import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    """Mock the ML pipeline so the API can start without GPU/models."""
    with patch("api.ml_worker") as mock_worker, \
         patch("api.load_config") as mock_load_cfg, \
         patch("api.load_camera_configs") as mock_load_cams:

        mock_load_cfg.return_value = {
            "system": {"device": "cpu"},
            "stream": {"jpeg_quality": 70},
            "api": {"host": "0.0.0.0", "port": 8000},
        }
        mock_load_cams.return_value = [
            MagicMock(
                camera_id="cam1", source="test.mp4",
                name="Test Camera", gps_lat=28.61, gps_lng=77.20,
            ),
        ]
        # Don't actually start the ML thread
        mock_worker.return_value = None

        yield {
            "worker": mock_worker,
            "load_config": mock_load_cfg,
            "load_cameras": mock_load_cams,
        }


@pytest.fixture
def client(mock_pipeline):
    """FastAPI TestClient with mocked backend."""
    from api import app, state
    state.start_time = time.time()
    state.running = True
    state.frame_count = 100

    with TestClient(app) as c:
        yield c

    state.running = False


class TestSystemEndpoints:
    def test_system_status(self, client):
        resp = client.get("/api/v1/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("operational", "stopped")
        assert "uptime_seconds" in data
        assert "fps" in data
        assert "frame_count" in data

    def test_system_status_has_camera_count(self, client):
        resp = client.get("/api/v1/system/status")
        data = resp.json()
        assert "total_cameras" in data


class TestStreamEndpoints:
    def test_active_streams(self, client):
        resp = client.get("/api/v1/streams/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "cameras" in data

    def test_snapshot_no_frame(self, client):
        resp = client.get("/api/v1/streams/cam1/snapshot")
        assert resp.status_code == 404

    def test_snapshot_with_frame(self, client):
        from api import state
        import cv2
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", frame)
        state._frame_buffer["cam1"] = (jpeg.tobytes(), time.time())

        resp = client.get("/api/v1/streams/cam1/snapshot")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_control_no_pipeline(self, client):
        from api import state
        state.orchestrator = None
        resp = client.post("/api/v1/streams/cam1/control", json={"action": "stop"})
        assert resp.status_code == 503


class TestIdentityEndpoints:
    def test_identities_empty(self, client):
        resp = client.get("/api/v1/identities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_identity_not_found(self, client):
        from api import state
        mock_orch = MagicMock()
        mock_orch.global_matcher.get_identity.return_value = None
        state.orchestrator = mock_orch

        resp = client.get("/api/v1/identities/999")
        assert resp.status_code == 404

    def test_register_no_pipeline(self, client):
        from api import state
        state.orchestrator = None
        resp = client.post("/api/v1/identities/register", json={
            "name": "Alice", "image": "aW52YWxpZA=="
        })
        assert resp.status_code == 503


class TestAlertEndpoints:
    def test_alerts_empty(self, client):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alerts"] == []

    def test_alerts_with_data(self, client):
        from api import state
        state.recent_alerts.append({"type": "FACE_MATCH", "threat": "WANTED", "timestamp": time.time()})
        state.recent_alerts.append({"type": "VEHICLE_DETECTED", "plate": "MH01AB1234", "timestamp": time.time()})

        resp = client.get("/api/v1/alerts")
        data = resp.json()
        assert len(data["alerts"]) == 2

    def test_alerts_filter_by_type(self, client):
        from api import state
        state.recent_alerts.clear()
        state.recent_alerts.append({"type": "FACE_MATCH", "timestamp": time.time()})
        state.recent_alerts.append({"type": "VEHICLE_DETECTED", "timestamp": time.time()})

        resp = client.get("/api/v1/alerts?alert_type=FACE_MATCH")
        data = resp.json()
        assert all(a["type"] == "FACE_MATCH" for a in data["alerts"])

    def test_alerts_limit(self, client):
        from api import state
        state.recent_alerts.clear()
        for i in range(20):
            state.recent_alerts.append({"type": "TEST", "timestamp": time.time()})

        resp = client.get("/api/v1/alerts?limit=5")
        data = resp.json()
        assert len(data["alerts"]) == 5


class TestRiskEndpoints:
    def test_risk_summary_empty(self, client):
        resp = client.get("/api/v1/risk/summary")
        assert resp.status_code == 200


class TestFenceEndpoints:
    def test_fence_no_pipeline(self, client):
        from api import state
        state.orchestrator = None
        resp = client.post("/api/v1/fence/config", json={
            "points": [[100, 200], [800, 200]]
        })
        assert resp.status_code == 503

    def test_fence_invalid_points(self, client):
        from api import state
        state.orchestrator = MagicMock()
        state.orchestrator.risk_analyzers = {"cam1": MagicMock()}

        resp = client.post("/api/v1/fence/config", json={
            "points": [[100, 200]]  # Only 1 point
        })
        assert resp.status_code == 400

    def test_fence_update_all_cameras(self, client):
        from api import state
        mock_analyzer = MagicMock()
        state.orchestrator = MagicMock()
        state.orchestrator.risk_analyzers = {"cam1": mock_analyzer}

        resp = client.post("/api/v1/fence/config", json={
            "points": [[100, 200], [800, 200]]
        })
        assert resp.status_code == 200
        mock_analyzer.update_fence.assert_called_once()

    def test_fence_update_specific_camera(self, client):
        from api import state
        mock_analyzer = MagicMock()
        state.orchestrator = MagicMock()
        state.orchestrator.risk_analyzers = {"cam1": mock_analyzer}

        resp = client.post("/api/v1/fence/config", json={
            "points": [[100, 200], [800, 200]],
            "camera_id": "cam1",
        })
        assert resp.status_code == 200
        mock_analyzer.update_fence.assert_called_once()
