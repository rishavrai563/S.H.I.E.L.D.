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


# ═══════════════════════════════════════════════════════════
#  FRS / ANPR / Event-log endpoints
# ═══════════════════════════════════════════════════════════
import base64
import cv2
from types import SimpleNamespace


def b64_image(img):
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def analytics(client, tmp_path):
    """Attach real FRS (fake nets), ANPR (fake OCR) and AlertManager to the API state."""
    from api import state
    from core.alerts import AlertManager
    from core.anpr import ANPREngine
    from tests.test_face_auth import make_engine
    from tests.test_anpr import FakeReader

    face = make_engine(tmp_path)
    anpr = ANPREngine(reader=FakeReader("MHO1AB1234", 0.9), watchlist_path=str(tmp_path / "wl.json"))
    alerts = AlertManager(events_log_path=str(tmp_path / "events.jsonl"))
    state.orchestrator = SimpleNamespace(face_auth=face, anpr_engine=anpr, alert_manager=alerts)
    yield state.orchestrator
    state.orchestrator = None


RED = np.full((200, 160, 3), (20, 20, 230), np.uint8)


class TestFaceEndpoints:
    def test_enroll_requires_pipeline(self, client):
        from api import state
        state.orchestrator = None
        assert client.post("/api/v1/faces/enroll", json={"name": "A", "image": b64_image(RED)}).status_code == 503

    def test_enroll_list_search_delete(self, client, analytics):
        r = client.post("/api/v1/faces/enroll", json={"name": "Suspect", "image": b64_image(RED),
                                                     "category": "wanted", "notes": "n"})
        assert r.status_code == 200, r.text
        ident = r.json()["identity"]
        assert ident["category"] == "WANTED" and ident["references_added"] == 1

        g = client.get("/api/v1/faces/gallery").json()
        assert g["total"] == 1 and g["identities"][0]["thumbnail"]

        s = client.post("/api/v1/faces/search", json={"image": b64_image(RED)}).json()
        assert s["faces"][0]["match"]["name"] == "Suspect"

        assert client.delete(f"/api/v1/faces/{ident['identity_id']}").status_code == 200
        assert client.delete(f"/api/v1/faces/{ident['identity_id']}").status_code == 404

    def test_enroll_data_url_and_multiple_images(self, client, analytics):
        data_url = "data:image/jpeg;base64," + b64_image(RED)
        r = client.post("/api/v1/faces/enroll", json={"name": "A", "images": [data_url, b64_image(RED)]})
        assert r.json()["identity"]["num_references"] == 2

    def test_enroll_validation_errors(self, client, analytics):
        assert client.post("/api/v1/faces/enroll", json={"name": "A"}).status_code == 400
        assert client.post("/api/v1/faces/enroll", json={"name": "A", "image": "@@@"}).status_code == 400
        black = b64_image(np.zeros((100, 100, 3), np.uint8))
        assert client.post("/api/v1/faces/enroll", json={"name": "A", "image": black}).status_code == 422
        bad_cat = client.post("/api/v1/faces/enroll", json={"name": "A", "image": b64_image(RED), "category": "X"})
        assert bad_cat.status_code == 422


class TestAnprEndpoints:
    def test_watchlist_crud(self, client, analytics):
        before = client.get("/api/v1/anpr/watchlist").json()["total"]
        r = client.post("/api/v1/anpr/watchlist", json={"plate": "gj-05 ab 7777", "tag": "stolen", "notes": "x"})
        assert r.status_code == 200 and r.json()["plate"] == "GJ05AB7777"
        items = {w["plate"]: w for w in client.get("/api/v1/anpr/watchlist").json()["watchlist"]}
        assert len(items) == before + 1 and items["GJ05AB7777"]["tag"] == "STOLEN"
        assert client.delete("/api/v1/anpr/watchlist/GJ05AB7777").status_code == 200
        assert client.delete("/api/v1/anpr/watchlist/GJ05AB7777").status_code == 404
        assert client.post("/api/v1/anpr/watchlist", json={"plate": "A1"}).status_code == 400

    def test_recognize(self, client, analytics):
        from tests.test_anpr import synthetic_vehicle
        r = client.post("/api/v1/anpr/recognize", json={"image": b64_image(synthetic_vehicle())})
        assert r.status_code == 200
        assert r.json()["result"]["plate"] == "MH01AB1234"

    def test_recent_plates(self, client, analytics):
        from tests.test_anpr import synthetic_vehicle
        car = synthetic_vehicle()
        analytics.anpr_engine.extract_plate_with_voting(car, "cam1:1", now=0)
        analytics.anpr_engine.extract_plate_with_voting(car, "cam1:1", now=1, camera_id="cam1")
        plates = client.get("/api/v1/anpr/plates").json()["plates"]
        assert plates[0]["plate"] == "MH01AB1234" and plates[0]["camera_id"] == "cam1"

    def test_anpr_disabled(self, client, analytics):
        analytics.anpr_engine = None
        assert client.get("/api/v1/anpr/watchlist").status_code == 503


class TestEventLog:
    def test_events_endpoint(self, client, analytics):
        am = analytics.alert_manager
        am.publish([am.make_alert("INTRUSION", "cam1", "CRITICAL", "x"), am.make_alert("LOITERING", "cam1", "WARNING", "y")])
        ev = client.get("/api/v1/events").json()
        assert ev["total"] == 2 and ev["events"][0]["type"] == "LOITERING"
        only = client.get("/api/v1/events?alert_type=INTRUSION").json()
        assert [e["type"] for e in only["events"]] == ["INTRUSION"]


class TestDisplayCompositor:
    def test_overlay_drawn_on_newer_frame(self):
        import threading, time as _t
        from unittest.mock import MagicMock
        import cv2
        from api import ShieldState, display_worker
        st = ShieldState()
        st.config = {"stream": {"display_fps": 50, "display_max_dim": 2000}}
        raw = np.zeros((40, 60, 3), np.uint8)
        ann = raw.copy(); ann[5:10, 5:10] = (0, 0, 255)          # a red "box"
        newer = np.full((40, 60, 3), 100, np.uint8)               # camera moved on
        stream = MagicMock(); stream.latest_display.return_value = (newer, _t.time(), 1)
        st.orchestrator = MagicMock(); st.orchestrator.stream_manager.streams = {"cam1": stream}
        st.set_overlay("cam1", raw, ann)
        st.running = True
        th = threading.Thread(target=display_worker, args=(st,), daemon=True); th.start()
        deadline = _t.time() + 3
        while st.get_latest_frame("cam1") is None and _t.time() < deadline:
            _t.sleep(0.01)
        st.running = False; th.join(2)
        img = cv2.imdecode(np.frombuffer(st.get_latest_frame("cam1")[0], np.uint8), cv2.IMREAD_COLOR)
        assert img[7, 7, 2] > 200 and img[7, 7, 0] < 60         # overlay copied
        assert abs(int(img[30, 40, 0]) - 100) < 10               # rest is the newer frame


class TestDrawnFenceAPI:
    def _orch(self, tmp_path):
        from api import state
        from core.behavior_risk import RiskAnalyzer
        from core.orchestrator import Orchestrator
        orch = MagicMock()
        orch.risk_analyzers = {"cam1": RiskAnalyzer(frame_width=1280, frame_height=720)}
        orch._fences_path = str(tmp_path / "fences.json")
        orch.ignore_zones = {}
        orch.save_fences = lambda: Orchestrator.save_fences(orch)
        state.orchestrator = orch
        return orch

    def test_zone_saved_read_back_and_persisted(self, client, tmp_path):
        import json
        orch = self._orch(tmp_path)
        pts = [[100, 100], [600, 120], [500, 600], [150, 500]]
        r = client.post("/api/v1/fence/config", json={"camera_id": "cam1", "type": "zone", "points": pts})
        assert r.status_code == 200
        got = client.get("/api/v1/fence/cam1").json()
        assert got["type"] == "zone" and got["points"] == pts and got["frame"] == {"width": 1280, "height": 720}
        assert json.load(open(orch._fences_path))["cam1"]["points"] == pts

    def test_zone_needs_three_points(self, client, tmp_path):
        self._orch(tmp_path)
        r = client.post("/api/v1/fence/config", json={"camera_id": "cam1", "type": "zone", "points": [[0, 0], [9, 9]]})
        assert r.status_code == 400

    def test_bad_type_and_unknown_camera(self, client, tmp_path):
        self._orch(tmp_path)
        assert client.post("/api/v1/fence/config", json={"type": "circle", "points": [[0, 0], [5, 5]]}).status_code == 400
        assert client.get("/api/v1/fence/nope").status_code == 404

    def test_empty_points_disables(self, client, tmp_path):
        self._orch(tmp_path)
        assert client.post("/api/v1/fence/config", json={"camera_id": "cam1", "points": []}).status_code == 200
        assert client.get("/api/v1/fence/cam1").json()["points"] == []


def test_ignore_zone_api(client, tmp_path):
    from api import state
    from core.behavior_risk import RiskAnalyzer
    from core.orchestrator import Orchestrator
    orch = MagicMock()
    orch.risk_analyzers = {"cam1": RiskAnalyzer(frame_width=1280, frame_height=720)}
    orch.ignore_zones = {}
    orch._fences_path = str(tmp_path / "f.json")
    orch.set_ignore_zones = lambda c, z: Orchestrator.set_ignore_zones(orch, c, z)
    orch.save_fences = lambda: Orchestrator.save_fences(orch)
    state.orchestrator = orch
    zone = [[400, 0], [880, 0], [880, 110], [400, 110]]
    assert client.post("/api/v1/fence/ignore", json={"camera_id": "cam1", "zones": [zone]}).status_code == 200
    assert client.get("/api/v1/fence/cam1").json()["ignore"] == [zone]
    assert client.post("/api/v1/fence/ignore", json={"camera_id": "cam1", "zones": [[[0, 0], [1, 1]]]}).status_code == 400
    assert client.post("/api/v1/fence/ignore", json={"camera_id": "cam1", "zones": []}).status_code == 200
    assert client.get("/api/v1/fence/cam1").json()["ignore"] == []
