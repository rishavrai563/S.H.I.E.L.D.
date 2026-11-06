"""Verify the GPU stack SHIELD needs:  python scripts/check_gpu.py"""
import sys
import torch

print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA NOT available. Install the CUDA wheel:\n"
             "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.half)
print("FP16 matmul OK:", (x @ x).shape)
import easyocr  # noqa: F401
print("OK - ready: python run_shield.py --api --device cuda")
