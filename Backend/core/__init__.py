# ═══════════════════════════════════════════════════════════
#  SHIELD Core — Multi-Camera Person Tracking & Re-ID System
# ═══════════════════════════════════════════════════════════
#
#  Processing Pipeline:
#    StreamManager → SecurityDetector → ReIDEncoder
#      → SingleCameraTracker(s) → GlobalMatcher → RiskAnalyzer
#
#  Specialized Modules:
#    ANPREngine    — Automatic Number Plate Recognition
#    FaceRecognitionEngine — Face detection + deep recognition (FRS)
#    VehicleTracker — Vehicle tracking / classification for ANPR voting
#    AlertManager   — Alert cooldowns, event log, C&C webhook
# ═══════════════════════════════════════════════════════════

# ── Local model caches ────────────────────────────────────
# Every model download (OSNet via torchreid, EasyOCR, Ultralytics assets,
# optional FaceNet / HuggingFace) goes to Backend/.cache inside this checkout,
# never to the user's home drive. Set before any heavy library is imported,
# because torch/easyocr read these variables at import time.
import os as _os

_CACHE_ROOT = _os.environ.get(
    "SHIELD_CACHE_DIR",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".cache"),
)
for _var, _sub in (("TORCH_HOME", "torch"), ("EASYOCR_MODULE_PATH", "easyocr"),
                   ("YOLO_CONFIG_DIR", "ultralytics"), ("HF_HOME", "huggingface"),
                   ("MPLCONFIGDIR", "matplotlib")):
    _os.environ.setdefault(_var, _os.path.join(_CACHE_ROOT, _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

# Lazy imports — allow partial loading when heavy deps
# (ultralytics, torch, torchreid, easyocr) are not installed.
# This lets lightweight modules (kalman, global_matcher, etc.)
# be imported independently, e.g. for unit testing.

_IMPORTS = {
    "core.kalman":         ["KalmanFilter8"],
    "core.stream_manager": ["StreamManager", "CameraStream", "CameraConfig"],
    "core.global_matcher": ["GlobalMatcher", "GlobalIdentity", "MatchResult"],
    "core.behavior_risk":  ["RiskAnalyzer"],
    "core.detector":       ["SecurityDetector", "Detection"],
    "core.reid_encoder":   ["ReIDEncoder"],
    "core.tracker":        ["SingleCameraTracker", "Track", "TrackEvent", "TrackState"],
    "core.anpr":           ["ANPREngine"],
    "core.face_auth":      ["FaceRecognitionEngine", "FaceAuthEngine"],
    "core.vehicle_tracker": ["VehicleTracker"],
    "core.alerts":         ["AlertManager"],
    "core.orchestrator":   ["Orchestrator"],
}

_loaded = {}

for _mod, _names in _IMPORTS.items():
    try:
        import importlib as _il
        _m = _il.import_module(_mod)
        for _n in _names:
            _obj = getattr(_m, _n, None)
            if _obj is not None:
                _loaded[_n] = _obj
                globals()[_n] = _obj
    except (ImportError, ModuleNotFoundError):
        pass  # skip modules whose deps are missing

__all__ = [
    # Detection
    "SecurityDetector",
    "Detection",
    # Re-Identification
    "ReIDEncoder",
    # Tracking
    "SingleCameraTracker",
    "Track",
    "TrackEvent",
    "TrackState",
    "KalmanFilter8",
    # Streams
    "StreamManager",
    "CameraStream",
    "CameraConfig",
    # Global Identity
    "GlobalMatcher",
    "GlobalIdentity",
    "MatchResult",
    # Analytics
    "RiskAnalyzer",
    "ANPREngine",
    "FaceRecognitionEngine",
    "FaceAuthEngine",
    "VehicleTracker",
    "AlertManager",
    # Orchestrator
    "Orchestrator",
]
