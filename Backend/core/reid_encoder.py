"""
Person Re-Identification Encoder — OSNet_x1_0 wrapper.

Converts cropped person images into 512-dimensional identity vectors (embeddings).
Two people with high cosine similarity between their embeddings are likely the same person.

Key design decisions:
- OSNet_x1_0 (not x0_25): 4× wider network = significantly more discriminative
- MSMT17 pretrained: largest multi-camera ReID dataset, best cross-camera generalization
- Batch extraction: all crops from all cameras → single GPU call
- L2 normalization: required for cosine similarity via dot product
- High-conf only: we NEVER extract embeddings for low-conf crops (garbage in = garbage out)

Model weights:
- If model_path exists on disk, loads from there.
- Otherwise, torchreid auto-downloads the pretrained weights on first run.
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

# TORCH_HOME (checkpoint cache) is pointed at Backend/.cache/torch in core/__init__.py.

import torch
import torchreid


class ReIDEncoder:
    """
    OSNet_x1_0 person re-identification encoder.
    
    Usage:
        encoder = ReIDEncoder()
        
        # Single crop
        embedding = encoder.extract(crop_bgr)  # → (512,)
        
        # Batch of crops (efficient)
        embeddings = encoder.batch_extract([crop1, crop2, ...])  # → (N, 512)
    """

    def __init__(self, model_name: str = "osnet_x1_0",
                 model_path: str = "models/osnet_x1_0_msmt17.pth",
                 device: str = "cuda",
                 input_width: int = 128,
                 input_height: int = 256):
        """
        Args:
            model_name:   Model architecture name (for torchreid).
            model_path:   Path to pretrained weights. If file doesn't exist,
                          torchreid will auto-download MSMT17 weights.
            device:       "cuda" or "cpu".
            input_width:  Model input width (128 for OSNet).
            input_height: Model input height (256 for OSNet).
        """
        self.device = device
        self.input_size = (input_width, input_height)  # (W, H) for PIL resize

        # Re-ID needs weights trained on person re-identification (MSMT17). If they are
        # missing, torchreid falls back to ImageNet classification weights, which work
        # but match people across cameras far less reliably -- warn loudly.
        effective_path = model_path if os.path.isfile(model_path) else ''

        if effective_path:
            print(f"  [ReID] Loading re-ID weights from {model_path}")
        else:
            print(f"  [ReID] WARNING: {model_path} not found -- using ImageNet {model_name} weights. "
                  f"Cross-camera matching will be weak. Run: python scripts/download_models.py")
        
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=effective_path,
            device=device,
        )

        # Fast path: batched GPU preprocessing + FP16 inference on the underlying model
        # (torchreid's extractor converts every crop to PIL and normalises on the CPU).
        self._model = self.extractor.model.eval()
        self._torch_device = torch.device(device)
        self._half = str(device).startswith("cuda") and torch.cuda.is_available()
        if self._half:
            self._model.half()
        dtype = torch.float16 if self._half else torch.float32
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self._torch_device, dtype=dtype).view(1, 3, 1, 1) * 255
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self._torch_device, dtype=dtype).view(1, 3, 1, 1) * 255

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-dim embedding from a single BGR person crop.
        
        Args:
            crop_bgr: BGR person crop (any size).
            
        Returns:
            L2-normalized (512,) numpy array, or None if crop is invalid.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        
        result = self.batch_extract([crop_bgr])
        if result is not None and len(result) > 0:
            return result[0]
        return None

    def batch_extract(self, crops_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract embeddings for a batch of BGR person crops in one GPU call.
        
        Args:
            crops_bgr: List of BGR person crops (any size).
            
        Returns:
            (N, 512) numpy array with L2-normalized embeddings.
            Invalid crops get a zero vector. Returns None only if ALL crops invalid.
            
            IMPORTANT: The returned array has the SAME length as crops_bgr,
            preserving index alignment with the caller's crop_map.
        """
        if not crops_bgr:
            return None
        
        n_total = len(crops_bgr)
        
        # Filter out empty/invalid crops, tracking their original index
        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0 and crop.shape[0] > 2 and crop.shape[1] > 2:
                valid_crops.append(crop)
                valid_indices.append(i)
        
        if not valid_crops:
            return None

        # Convert BGR → RGB and resize to (256, 128) — H×W format for cv2
        processed = []
        for crop in valid_crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.input_size[0], self.input_size[1]))  # (W, H)
            processed.append(resized)
        
        # Single GPU forward pass: one upload, normalise on the GPU, FP16 on CUDA
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(processed)).to(self._torch_device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).to(self._mean.dtype)
            x = (x - self._mean) / self._std
            features = self._model(x)

        features = features.float().cpu().numpy().astype(np.float32)
        
        # L2 normalize each embedding
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        features = features / norms
        
        # Build full-sized output array preserving index alignment
        # Invalid crops get zero vectors (will be ignored by tracker since
        # cosine_sim(zero_vec, anything) ≈ 0 after L2 normalization)
        if len(valid_indices) == n_total:
            return features  # All valid — no remapping needed
        
        full_output = np.zeros((n_total, features.shape[1]), dtype=np.float32)
        for out_idx, orig_idx in enumerate(valid_indices):
            full_output[orig_idx] = features[out_idx]
        
        return full_output
