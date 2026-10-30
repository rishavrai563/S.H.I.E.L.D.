"""
Shared fixtures for SHIELD test suite.

Pre-mocks heavy third-party modules (ultralytics, torch, torchreid, easyocr)
so that core.* modules can be imported even when those packages are not installed.
Adds the Backend directory to sys.path so `from core.xxx import ...` works.
"""
import os
import sys
import types
from unittest.mock import MagicMock

# ── Ensure Backend/ is on sys.path ──────────────────────────
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ── Pre-mock heavy third-party modules if not installed ─────
# This lets `from core.detector import Detection` (etc.) work
# even without ultralytics / torch / torchreid / easyocr.

def _ensure_module(name, attrs=None):
    """Install a stub module into sys.modules if not already importable."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except (ImportError, ModuleNotFoundError):
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod


# ultralytics — detector.py does `from ultralytics import YOLO`
_ensure_module("ultralytics", {"YOLO": MagicMock})

# torch — reid_encoder.py does `import torch`
_mock_torch = types.ModuleType("torch")
_mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
_mock_torch.device = MagicMock
_mock_torch.float32 = "float32"
_mock_torch.Tensor = MagicMock
_mock_torch.from_numpy = MagicMock(return_value=MagicMock())
_mock_torch.stack = MagicMock(return_value=MagicMock())
_mock_torch.cuda = MagicMock()
_mock_torch.cuda.is_available = MagicMock(return_value=False)
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401 — use real torch if available
    except (ImportError, ModuleNotFoundError):
        sys.modules["torch"] = _mock_torch
        # Also need torch.nn, torch.nn.functional etc.
        _mock_torch_nn = types.ModuleType("torch.nn")
        _mock_torch_nn.Module = MagicMock
        sys.modules["torch.nn"] = _mock_torch_nn
        _mock_torch_nn_functional = types.ModuleType("torch.nn.functional")
        sys.modules["torch.nn.functional"] = _mock_torch_nn_functional

# torchvision — sometimes imported transitively
_ensure_module("torchvision")
_ensure_module("torchvision.transforms", {
    "Compose": MagicMock,
    "Resize": MagicMock,
    "ToTensor": MagicMock,
    "Normalize": MagicMock,
})

# torchreid — reid_encoder.py does `import torchreid`
_mock_torchreid = types.ModuleType("torchreid")
# torchreid.utils.FeatureExtractor is what's typically used, but the code uses
# torchreid.models.build_model() style calls. Mock the whole tree.
_mock_torchreid.models = MagicMock()
_mock_torchreid.utils = MagicMock()
if "torchreid" not in sys.modules:
    try:
        import torchreid  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        sys.modules["torchreid"] = _mock_torchreid
        sys.modules["torchreid.models"] = _mock_torchreid.models
        sys.modules["torchreid.utils"] = _mock_torchreid.utils

# easyocr — anpr.py does `import easyocr`
_ensure_module("easyocr", {"Reader": MagicMock})


# ── Now safe to import numpy and pytest ─────────────────────
import pytest
import numpy as np


# ── Reusable Fixtures ───────────────────────────────────────

@pytest.fixture
def random_embedding():
    """Returns a factory for random L2-normalized 512-dim embeddings."""
    def _make(seed=None):
        rng = np.random.RandomState(seed)
        vec = rng.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec
    return _make


@pytest.fixture
def similar_embeddings(random_embedding):
    """Returns two embeddings with cosine similarity > 0.90."""
    base = random_embedding(seed=42)
    noise = np.random.RandomState(99).randn(512).astype(np.float32) * 0.05
    perturbed = base + noise
    perturbed /= np.linalg.norm(perturbed)
    return base, perturbed


@pytest.fixture
def dissimilar_embeddings(random_embedding):
    """Returns two embeddings with low cosine similarity."""
    a = random_embedding(seed=1)
    b = random_embedding(seed=9999)
    return a, b


@pytest.fixture
def sample_bbox():
    """A standard bounding box (x1, y1, x2, y2)."""
    return (100.0, 200.0, 180.0, 400.0)


@pytest.fixture
def sample_config():
    """A minimal config dict matching config.yaml structure."""
    return {
        "system": {"device": "cpu", "num_cameras": 2},
        "detection": {
            "model_path": "models/yolo11m.pt",
            "high_conf": 0.40,
            "low_conf": 0.07,
            "min_area": 120,
            "nms_iou": 0.75,
            "imgsz": 1280,
        },
        "reid": {
            "model_name": "osnet_x1_0",
            "model_path": "models/osnet_x1_0_msmt17.pth",
            "embedding_dim": 512,
            "input_width": 128,
            "input_height": 256,
        },
        "tracker": {
            "n_confirm": 3,
            "tentative_max_miss": 3,
            "max_miss": 25,
            "lost_buffer_sec": 5.0,
            "gate_dist_px": 200,
            "mahal_gate": 9.48,
            "cost_threshold_high": 0.65,
            "cost_threshold_low": 0.70,
            "weights_normal": [0.50, 0.40, 0.10],
            "weights_crowd": [0.20, 0.60, 0.20],
            "embedding_pool_k": 10,
            "sim_threshold_live": 0.45,
            "sim_threshold_lost": 0.40,
        },
        "global_matcher": {
            "match_threshold": 0.70,
            "margin_threshold": 0.08,
            "registration_threshold": 0.70,
            "ema_alpha": 0.10,
        },
        "stream": {"target_fps": 30, "jpeg_quality": 70},
        "risk": {"alert_threshold": 90},
        "api": {"host": "0.0.0.0", "port": 8000},
    }
