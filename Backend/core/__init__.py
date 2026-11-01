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
#    FaceAuthEngine — Face detection & watchlist matching
# ═══════════════════════════════════════════════════════════

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
    "core.face_auth":      ["FaceAuthEngine"],
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
    "FaceAuthEngine",
    # Orchestrator
    "Orchestrator",
]
