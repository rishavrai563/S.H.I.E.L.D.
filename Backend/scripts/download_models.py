"""
Download the model weights SHIELD needs that are not committed to git.

    cd Backend
    python scripts/download_models.py           # face models (YuNet + SFace)
    python scripts/download_models.py --check   # only verify what is present

Files are verified with SHA-256 so a truncated download is never used.
YOLOv11m (models/yolo11m.pt) is fetched automatically by ultralytics on first
run if missing; OSNet weights are fetched by torchreid; EasyOCR fetches its
own weights into ~/.EasyOCR on first use.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = {
    "models/face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "models/face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    # Fast detector (default profile); yolo11m.pt ships with the repo
    "models/yolo11s.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    ),
    # Person re-identification (cross-camera tracking), from the torchreid author
    "models/osnet_ain_x1_0_msmt17.pth": (
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
        "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
        "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
    ),
    "models/osnet_x1_0_msmt17.pth": (
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727",
    ),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only, do not download")
    args = ap.parse_args()

    ok = True
    for rel, (url, digest) in MODELS.items():
        path = os.path.join(BACKEND, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path) and sha256(path) == digest:
            print(f"  ✅ {rel}")
            continue
        if args.check:
            print(f"  ❌ {rel} missing or corrupt")
            ok = False
            continue
        print(f"  ⬇️  {rel}")
        try:
            download(url, path)
        except Exception as e:
            print(f"  ❌ download failed: {e}")
            ok = False
            continue
        if sha256(path) != digest:
            print(f"  ❌ checksum mismatch for {rel} — delete it and retry")
            ok = False
        else:
            print(f"  ✅ {rel}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
