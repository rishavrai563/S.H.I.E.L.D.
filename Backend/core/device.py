"""Compute-device resolution and GPU setup shared by every ML component."""
from __future__ import annotations


def resolve_device(requested: str = "cuda") -> str:
    """Return "cuda" if requested and usable, else "cpu" (with a loud warning)."""
    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        print("  [GPU] torch not installed - falling back to CPU")
        return "cpu"
    if not torch.cuda.is_available():
        print(
            "\n  [GPU] WARNING: CUDA requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}). Falling back to CPU.\n"
            "        You probably have the CPU-only PyTorch wheel. Install the CUDA build:\n"
            "        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
        )
        return "cpu"
    torch.backends.cudnn.benchmark = True  # fixed input sizes -> faster convs
    props = torch.cuda.get_device_properties(0)
    print(f"  [GPU] Using {props.name} ({props.total_memory / 2**30:.1f} GB), torch {torch.__version__}")
    return "cuda"
