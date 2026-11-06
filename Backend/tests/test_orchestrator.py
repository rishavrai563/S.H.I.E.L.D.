"""
Tests for the Orchestrator — integration test with all sub-components mocked.
Verifies config wiring and initialization logic.
"""
import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from core.stream_manager import CameraConfig


@pytest.fixture
def mock_all_components(tmp_path, monkeypatch):
    """Mock all heavy ML components so Orchestrator can be instantiated."""
    monkeypatch.chdir(tmp_path)
    with patch("core.orchestrator.SecurityDetector") as mock_det, \
         patch("core.orchestrator.ReIDEncoder") as mock_reid, \
         patch("core.orchestrator.ANPREngine") as mock_anpr, \
         patch("core.orchestrator.FaceRecognitionEngine") as mock_face, \
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


class TestOrchestratorNewModules:
    def test_face_and_anpr_receive_config(self, mock_all_components):
        from core.orchestrator import Orchestrator
        cfg = {"face_recognition": {"match_distance_threshold": 0.5, "confirm_hits": 3, "backend": "sface"},
               "anpr": {"temporal_votes_required": 4, "plate_format": "", "plate_aspect_ratio": [2.0, 6.0]},
               "risk": {"loiter_seconds": 12, "loiter_displacement_px": 25}}
        Orchestrator(cameras=[CameraConfig(camera_id="cam1", source="t.mp4")], device="cpu",
                     display=False, config=cfg)
        face_kw = mock_all_components["face"].call_args[1]
        assert face_kw["match_distance_threshold"] == 0.5 and face_kw["confirm_hits"] == 3
        assert face_kw["backend"] == "sface"
        anpr_kw = mock_all_components["anpr"].call_args[1]
        assert anpr_kw["temporal_votes_required"] == 4 and anpr_kw["aspect_ratio_range"] == (2.0, 6.0)
        risk_kw = mock_all_components["risk"].call_args[1]
        assert risk_kw["loiter_seconds"] == 12 and risk_kw["loiter_displacement_px"] == 25

    def test_anpr_can_be_disabled(self, mock_all_components):
        from core.orchestrator import Orchestrator
        orch = Orchestrator(cameras=[CameraConfig(camera_id="cam1", source="t.mp4")], device="cpu",
                            display=False, config={"anpr": {"enabled": False}})
        assert orch.anpr_engine is None
        mock_all_components["anpr"].assert_not_called()


# ═══════════════════════════════════════════════════════════
#  End-to-end pipeline: real trackers / risk / ANPR logic,
#  fake neural networks.
# ═══════════════════════════════════════════════════════════

from core.detector import Detection
from core.face_auth import FaceMatch


class FakeOCR:
    def readtext(self, image, **kw):
        return [([[0, 0], [10, 0], [10, 5], [0, 5]], "DLO1AB1234", 0.9)]  # OCR confuses 0/O


class FakeFaceEngine:
    """Confirms the suspect on the 2nd check of any track."""
    available = True
    gallery_size = 1

    def __init__(self, *a, **k):
        self.calls = {}

    def recognize_track(self, crop, key, now=None):
        n = self.calls[key] = self.calls.get(key, 0) + 1
        if n < 2:
            return None, False
        return FaceMatch("F1", "Wanted Man", "WANTED", 0.8, 0.2, bbox=(1, 1, 5, 5)), n == 2

    def prune_tracks(self, keys):
        pass

    def get_stats(self):
        return {}


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.anpr import ANPREngine
    rng = np.random.RandomState(0)
    person_emb = rng.randn(512).astype(np.float32)
    person_emb /= np.linalg.norm(person_emb)

    def make_anpr(**kw):
        kw.update(reader=FakeOCR(), watchlist={"DL01AB1234": {"tag": "STOLEN"}}, watchlist_path=None)
        kw.pop("use_gpu", None)
        return ANPREngine(**kw)

    with patch("core.orchestrator.SecurityDetector") as det_cls, \
         patch("core.orchestrator.ReIDEncoder") as enc_cls, \
         patch("core.orchestrator.ANPREngine", side_effect=make_anpr), \
         patch("core.orchestrator.FaceRecognitionEngine", FakeFaceEngine):
        enc_cls.return_value.batch_extract.side_effect = lambda crops: np.stack([person_emb] * len(crops))
        from core.orchestrator import Orchestrator
        cfg = {"anpr": {"ocr_interval_sec": 0.0},
               "tracker": {"n_confirm": 2},
               "night_vision": {"mode": "off"},
               "logging": {"events_log": str(tmp_path / "events.jsonl")}}
        orch = Orchestrator(cameras=[CameraConfig(camera_id="cam1", source="none.mp4")], device="cpu",
                            display=False, config=cfg)
        orch.risk_analyzers["cam1"].update_fence([(0, 600), (1920, 600)])
        yield orch, det_cls.return_value, tmp_path


def run_scenario(orch, detector, n_frames=30, night=False):
    orch.stream_manager.is_night = lambda cam: night
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    all_alerts, last = [], None
    for i in range(n_frames):
        y2 = 450 + i * 12           # person walks down across the fence at y=600
        person = Detection(bbox=(900.0, y2 - 300.0, 1000.0, float(y2)), confidence=0.9, class_id=0, label="person")
        car = Detection(bbox=(200.0 + i, 500.0, 620.0 + i, 800.0), confidence=0.9, class_id=2, label="car")
        detector.batch_detect.return_value = [([person, car], [])]
        last, _ = orch.process_frames({"cam1": (frame, time.time())})
        all_alerts.extend(last["alerts"])
    return all_alerts, last


class TestPipelineEndToEnd:
    def test_all_alert_types_raised(self, pipeline):
        orch, detector, _ = pipeline
        alerts, _ = run_scenario(orch, detector)
        types = {a["type"] for a in alerts}
        assert {"INTRUSION", "VEHICLE_DETECTED", "ANPR_WATCHLIST_HIT", "FACE_MATCH_SUSPECT"} <= types

    def test_alerts_fire_once_per_incident(self, pipeline):
        orch, detector, _ = pipeline
        alerts, _ = run_scenario(orch, detector)
        count = lambda t: sum(a["type"] == t for a in alerts)
        assert count("FACE_MATCH_SUSPECT") == 1
        assert count("VEHICLE_DETECTED") == 1
        assert count("ANPR_WATCHLIST_HIT") == 1
        assert count("INTRUSION") == 1

    def test_alert_content(self, pipeline):
        orch, detector, _ = pipeline
        alerts, _ = run_scenario(orch, detector)
        by = {a["type"]: a for a in alerts}
        face = by["FACE_MATCH_SUSPECT"]
        assert face["risk_level"] == "CRITICAL" and "CRITICAL_THREAT: FACE_MATCH_SUSPECT" in face["message"]
        plate = by["ANPR_WATCHLIST_HIT"]
        assert plate["risk_level"] == "CRITICAL" and plate["plate"] == "DL01AB1234"
        assert plate["watchlist"]["tag"] == "STOLEN"
        assert by["VEHICLE_DETECTED"]["vehicle_class"] == "car"
        assert by["INTRUSION"]["risk_level"] == "CRITICAL"

    def test_telemetry_payload(self, pipeline):
        orch, detector, _ = pipeline
        _, last = run_scenario(orch, detector)
        cam = last["streams"]["cam1"]
        assert cam["camera_name"] and cam["night_mode"] is False
        assert cam["vehicles"][0]["class_name"] == "car" and cam["vehicles"][0]["plate"] == "DL01AB1234"
        assert cam["face_matches"][0]["name"] == "Wanted Man"
        assert cam["counts"]["persons"] == 1 and cam["counts"]["vehicles"] == 1
        assert cam["scene_status"] == "BREACH_DETECTED"

    def test_events_persisted(self, pipeline):
        orch, detector, tmp_path = pipeline
        alerts, _ = run_scenario(orch, detector)
        lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == len(alerts)
        assert orch.alert_manager.read_events(alert_type="INTRUSION")

    def test_night_movement(self, pipeline):
        orch, detector, _ = pipeline
        alerts, _ = run_scenario(orch, detector, n_frames=8, night=True)
        night = [a for a in alerts if a["type"] == "NIGHT_MOVEMENT"]
        assert len(night) == 1 and night[0]["risk_level"] == "WARNING"

    def test_alerts_master_switch(self, pipeline):
        orch, detector, _ = pipeline
        orch.alerts_enabled = False
        alerts, _ = run_scenario(orch, detector)
        assert alerts == []

    def test_returns_annotated_frames(self, pipeline):
        orch, detector, _ = pipeline
        detector.batch_detect.return_value = [([], [])]
        tel, frames = orch.process_frames({"cam1": (np.zeros((1080, 1920, 3), np.uint8), 0.0)})
        assert frames["cam1"].shape == (1080, 1920, 3)
        assert tel["frame"] == 1


class TestAsyncANPR:
    """anpr.async_ocr: OCR runs on a background thread; plates are applied when ready."""

    @staticmethod
    def _enable_async(orch):
        import threading
        orch._anpr_async = True
        threading.Thread(target=orch._anpr_worker, daemon=True).start()

    @staticmethod
    def _run(orch, detector, n_frames=30):
        orch.stream_manager.is_night = lambda cam: False
        frame = np.full((1080, 1920, 3), 90, np.uint8)
        alerts, last = [], None
        for i in range(n_frames):
            car = Detection(bbox=(200.0 + i, 500.0, 620.0 + i, 800.0), confidence=0.9, class_id=2, label="car")
            detector.batch_detect.return_value = [([car], [])]
            last, _ = orch.process_frames({"cam1": (frame, time.time())})
            alerts.extend(last["alerts"])
            time.sleep(0.02)  # give the worker time to OCR
        return alerts, last

    def test_plate_confirmed_and_watchlist_hit(self, pipeline):
        orch, detector, _ = pipeline
        self._enable_async(orch)
        try:
            alerts, last = self._run(orch, detector)
        finally:
            orch._stop_event.set()
        count = lambda t: sum(a["type"] == t for a in alerts)
        assert count("VEHICLE_DETECTED") == 1
        assert count("ANPR_WATCHLIST_HIT") == 1
        assert last["streams"]["cam1"]["vehicles"][0]["plate"] == "DL01AB1234"
        assert not orch._anpr_pending

    def test_full_queue_does_not_block(self, pipeline):
        orch, detector, _ = pipeline
        orch._anpr_async = True  # no worker started: queue fills, main loop must keep going
        orch._anpr_jobs = __import__("queue").Queue(maxsize=1)
        alerts, last = self._run(orch, detector, n_frames=5)
        assert last["frame"] == 5
        assert orch._anpr_jobs.qsize() == 1


class TestPerformance:
    def test_metrics_in_telemetry(self, pipeline):
        orch, detector, _ = pipeline
        detector.batch_detect.return_value = [([], [])]
        frame = np.zeros((720, 1280, 3), np.uint8)
        for _ in range(6):
            tel, _ = orch.process_frames({"cam1": (frame, time.time() - 0.05)})
            time.sleep(0.02)
        assert tel["processing_fps"] > 0
        cam = tel["streams"]["cam1"]
        assert cam["analysis_fps"] > 0 and cam["latency_ms"] >= 50
        assert orch.performance()["cameras"]["cam1"]["analysis_fps"] == cam["analysis_fps"]

    def test_stalled_camera_reports_zero(self, pipeline):
        orch, detector, _ = pipeline
        detector.batch_detect.return_value = [([], [])]
        frame = np.zeros((720, 1280, 3), np.uint8)
        for _ in range(3):
            orch.process_frames({"cam1": (frame, time.time())})
        q = orch._cam_times["cam1"]
        for i in range(len(q)):
            q[i] -= 10  # pretend the last frame was 10 s ago
        orch._loop_times = __import__("collections").deque(t - 10 for t in orch._loop_times)
        perf = orch.performance()
        assert perf["processing_fps"] == 0.0 and perf["cameras"]["cam1"]["analysis_fps"] == 0.0

    def test_precomputed_detections_skip_yolo(self, pipeline):
        orch, detector, _ = pipeline
        detector.batch_detect.reset_mock()
        orch.process_frames({"cam1": (np.zeros((720, 1280, 3), np.uint8), time.time())},
                            detections=[([], [])])
        detector.batch_detect.assert_not_called()

    def test_pipelined_run_processes_all_frames(self, pipeline):
        orch, detector, _ = pipeline
        detector.batch_detect.side_effect = lambda frames: [([], []) for _ in frames]
        frame = np.zeros((720, 1280, 3), np.uint8)
        supply = [{"cam1": (frame, time.time())} for _ in range(5)]
        orch.stream_manager.grab_latest = lambda: supply.pop(0) if supply else {}
        orch.stream_manager.all_stopped = lambda: not supply
        orch.stream_manager.start_all = lambda: None
        seen = []
        orch._on_frames = lambda ann: seen.append(1) or True
        orch._pipelined = True
        t = __import__("threading").Thread(target=orch.run, daemon=True)
        t.start(); t.join(timeout=5)
        assert not t.is_alive() and len(seen) == 5
        assert detector.batch_detect.call_count == 5


class TestOCRBudget:
    def test_token_bucket_limits_rate(self, pipeline):
        orch, _, _ = pipeline
        orch._ocr_rate, orch._ocr_tokens, orch._ocr_refill_at = 4.0, 0.0, time.time()
        time.sleep(0.5)
        orch._refill_ocr_tokens()
        assert 1.5 <= orch._ocr_tokens <= 2.5             # ~4/s * 0.5 s
        time.sleep(2.0); orch._refill_ocr_tokens()
        assert orch._ocr_tokens == 4.0                    # capped at one second of budget

    def test_unlimited_when_zero(self, pipeline):
        orch, _, _ = pipeline
        orch._ocr_rate = 0
        orch._refill_ocr_tokens()
        assert orch._ocr_tokens == float("inf")


def test_handoff_alert_when_person_moves_camera(pipeline):
    """Same appearance seen on cam1, then a NEW track on cam2 -> same G-number + handoff event."""
    orch, detector, _ = pipeline
    from core.stream_manager import CameraConfig
    from core.tracker import SingleCameraTracker
    from core.behavior_risk import RiskAnalyzer
    from core.vehicle_tracker import VehicleTracker
    orch.camera_names["cam2"] = "Phone"; orch.camera_names["cam1"] = "Laptop"
    orch.trackers["cam2"] = SingleCameraTracker(camera_id="cam2", frame_w=1920, frame_h=1080, n_confirm=2)
    orch.risk_analyzers["cam2"] = RiskAnalyzer(camera_id="cam2", frame_width=1920, frame_height=1080) \
        if "camera_id" in RiskAnalyzer.__init__.__code__.co_varnames else orch.risk_analyzers["cam1"]
    orch.vehicle_trackers["cam2"] = VehicleTracker()
    orch.global_matcher.diff_cam_min = 0.0
    orch._xcam_threshold, orch._global_rematch_min_age = 0.7, 3
    orch.stream_manager.is_night = lambda cam: False
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    person = Detection(bbox=(900.0, 200.0, 1000.0, 500.0), confidence=0.9, class_id=0, label="person")
    alerts = []
    for _ in range(12):
        detector.batch_detect.return_value = [([person], [])]
        tel, _ = orch.process_frames({"cam1": (frame, time.time())}); alerts += tel["alerts"]
    g1 = orch.trackers["cam1"].local_to_global[next(iter(orch.trackers["cam1"].local_to_global))]
    for _ in range(25):
        detector.batch_detect.return_value = [([person], [])]
        tel, _ = orch.process_frames({"cam2": (frame, time.time())}); alerts += tel["alerts"]
    g2 = set(orch.trackers["cam2"].local_to_global.values())
    assert g2 == {g1}
    handoff = [a for a in alerts if a["type"] == "PERSON_HANDOFF"]
    assert len(handoff) == 1 and "Laptop → Phone" in handoff[0]["message"]
    assert [c for c, _, _ in orch.global_matcher.get_journey(g1)][-2:] == ["cam1", "cam2"]


def test_fences_persist_across_restart(pipeline, tmp_path):
    orch, _, _ = pipeline
    orch._fences_path = str(tmp_path / "f.json")
    orch.risk_analyzers["cam1"].update_fence([(10, 10), (500, 10), (500, 400)], "zone")
    orch.save_fences()
    orch.risk_analyzers["cam1"].update_fence([], "line")
    orch.load_fences()
    assert orch.risk_analyzers["cam1"].fence_type == "zone"
    assert len(orch.risk_analyzers["cam1"].virtual_fence_coords) == 3


class TestIgnoreZones:
    def test_poster_people_dropped_real_person_kept(self, pipeline):
        orch, detector, _ = pipeline
        orch.set_ignore_zones("cam1", [[[400, 0], [880, 0], [880, 110], [400, 110]]])   # the poster
        poster_people = [Detection(bbox=(520.0 + 30 * i, 5.0, 545.0 + 30 * i, 60.0), confidence=0.6,
                                   class_id=0, label="person") for i in range(8)]
        me = Detection(bbox=(438.0, 354.0, 1020.0, 716.0), confidence=0.92, class_id=0, label="person")
        in_front = Detection(bbox=(450.0, 20.0, 700.0, 700.0), confidence=0.9, class_id=0, label="person")
        kept = orch._drop_ignored("cam1", poster_people + [me, in_front])
        assert kept == [me, in_front]

    def test_ignore_zones_persist(self, pipeline, tmp_path):
        orch, _, _ = pipeline
        orch._fences_path = str(tmp_path / "f.json")
        orch.set_ignore_zones("cam1", [[[0, 0], [100, 0], [100, 100]]])
        orch.save_fences()
        orch.ignore_zones.clear()
        orch.load_fences()
        assert orch.ignore_zones["cam1"]["points"] == [[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]][0:1] or \
            len(orch.ignore_zones["cam1"]["points"]) == 1

    def test_invalid_zone_skipped(self, pipeline):
        orch, _, _ = pipeline
        assert orch.set_ignore_zones("cam1", [[[0, 0], [5, 5]], [[0, 0], [10, 0], [20, 0]]]) == 0
        assert "cam1" not in orch.ignore_zones


class TestFenceEventsPeopleAndVehicles:
    def _run(self, orch, detector, dets_per_frame):
        orch.stream_manager.is_night = lambda cam: False
        frame = np.full((1080, 1920, 3), 90, np.uint8)
        alerts = []
        for dets in dets_per_frame:
            detector.batch_detect.return_value = [(dets, [])]
            tel, _ = orch.process_frames({"cam1": (frame, time.time())})
            alerts += tel["alerts"]
        return alerts

    def test_vehicle_crossing_tripwire_alerts_once(self, pipeline):
        orch, detector, _ = pipeline
        orch.risk_analyzers["cam1"].update_fence([(0, 700), (1920, 700)], "line")
        frames = [[Detection(bbox=(800.0, y - 200.0, 1100.0, float(y)), confidence=0.9, class_id=2, label="car")]
                  for y in range(560, 900, 30)]
        alerts = self._run(orch, detector, frames)
        hits = [a for a in alerts if a["type"] == "VEHICLE_FENCE_CROSSING"]
        assert len(hits) == 1 and hits[0]["risk_level"] == "CRITICAL"

    def test_person_and_vehicle_zone_entry_and_exit(self, pipeline):
        orch, detector, _ = pipeline
        orch.risk_analyzers["cam1"].update_fence([(600, 300), (1300, 300), (1300, 900), (600, 900)], "zone")
        def frame(x):
            return [Detection(bbox=(x - 50.0, 400.0, x + 50.0, 700.0), confidence=0.9, class_id=0, label="person"),
                    Detection(bbox=(x - 150.0, 500.0, x + 150.0, 800.0), confidence=0.9, class_id=2, label="car")]
        xs = list(range(300, 1800, 60))               # walks in from the left and out the right
        alerts = self._run(orch, detector, [frame(x) for x in xs])
        entries = [a for a in alerts if a["type"] == "RESTRICTED_ZONE_ENTRY"]
        exits = [a for a in alerts if a["type"] == "RESTRICTED_ZONE_EXIT"]
        assert sorted(a["object_class"] for a in entries) == ["person", "vehicle"]
        assert sorted(a["object_class"] for a in exits) == ["person", "vehicle"]
        assert all(a["risk_level"] == "WARNING" and "Suspicious" in a["message"] for a in entries)
