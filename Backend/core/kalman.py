"""
8-State Kalman Filter for Person Tracking.

State:  [cx, cy, w, h, vx, vy, vw, vh]
    cx, cy  = bounding box center
    w, h    = bounding box width, height
    vx, vy  = velocity of center
    vw, vh  = velocity of size (handles person walking toward/away from camera)

Observation:  [cx, cy, w, h]

Model: Constant-velocity.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ─────────────────── Geometry Helpers ────────────────────

def xyxy_to_cxcywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h)."""
    w = max(2.0, bbox[2] - bbox[0])
    h = max(2.0, bbox[3] - bbox[1])
    cx = bbox[0] + w * 0.5
    cy = bbox[1] + h * 0.5
    return cx, cy, w, h


def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    w = max(2.0, float(w))
    h = max(2.0, float(h))
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def clamp_bbox(x1: float, y1: float, x2: float, y2: float,
               W: int, H: int) -> Tuple[float, float, float, float]:
    """Clamp bbox to frame boundaries, ensuring minimum 2px size."""
    x1 = max(0.0, min(float(x1), W - 1))
    y1 = max(0.0, min(float(y1), H - 1))
    x2 = max(0.0, min(float(x2), W - 1))
    y2 = max(0.0, min(float(y2), H - 1))
    if x2 <= x1:
        x2 = min(x1 + 2.0, float(W - 1))
    if y2 <= y1:
        y2 = min(y1 + 2.0, float(H - 1))
    return x1, y1, x2, y2


def bbox_iou(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get center point of (x1, y1, x2, y2) bbox."""
    return (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Get area of (x1, y1, x2, y2) bbox."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


# ─────────────────── Kalman Filter ──────────────────────

class KalmanFilter8:
    """
    8-state Kalman filter for bounding box tracking.
    
    Constant-velocity model predicting both position and size velocity.
    This handles people walking toward/away from the camera (size changes).
    
    Process noise is set higher for size velocity (vw, vh) per v3 plan:
    size doesn't change fast in typical surveillance, so we trust
    position velocity more and let size velocity be less influential.
    """

    # Transition matrix F: state_{t} = F @ state_{t-1}
    # Maps [cx, cy, w, h, vx, vy, vw, vh] → next step
    _F = np.eye(8, dtype=np.float32)
    _F[0, 4] = _F[1, 5] = _F[2, 6] = _F[3, 7] = 1.0  # pos += velocity

    # Observation matrix H: observation = H @ state
    # We observe [cx, cy, w, h] directly (no velocity)
    _H = np.zeros((4, 8), dtype=np.float32)
    _H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0

    def __init__(self, init_bbox: Tuple[float, float, float, float]) -> None:
        """
        Initialize Kalman filter from an (x1, y1, x2, y2) bounding box.
        
        Args:
            init_bbox: Initial bounding box in (x1, y1, x2, y2) format.
        """
        cx, cy, w, h = xyxy_to_cxcywh(init_bbox)
        
        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float32)

        # Process noise Q: how much we expect each state to change per step
        # Position velocities (vx, vy) are trusted more (lower noise)
        # Size velocities (vw, vh) are trusted less (higher noise) — v3 fix
        self.Q = np.diag([
            1.0, 1.0,     # cx, cy — moderate position process noise  
            4.0, 4.0,     # w, h — size can jitter more
            0.5, 0.5,     # vx, vy — velocity is fairly smooth
            2.0, 2.0,     # vw, vh — size velocity should be dampened (v3)
        ]).astype(np.float32)

        # Measurement noise R: how noisy our YOLO detections are
        self.R = np.diag([
            4.0, 4.0,     # cx, cy — position measurements fairly accurate
            16.0, 16.0,   # w, h — size measurements noisier
        ]).astype(np.float32)

        # Initial state covariance — large uncertainty at birth
        self.P = np.diag([
            50., 50.,     # cx, cy position uncertainty
            100., 100.,   # w, h size uncertainty
            100., 100.,   # vx, vy velocity unknown
            50., 50.,     # vw, vh size velocity unknown
        ]).astype(np.float32)

    def predict(self) -> None:
        """
        Predict next state using constant-velocity model.
        
        Updates self.x and self.P in-place.
        Call this BEFORE processing new detections each frame.
        """
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        """
        Update state estimate with a new measurement (detected bbox).
        
        Args:
            bbox: Measured bounding box (x1, y1, x2, y2).
        """
        cx, cy, w, h = xyxy_to_cxcywh(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Innovation: difference between measurement and prediction
        y = z - self._H @ self.x

        # Innovation covariance
        S = self._H @ self.P @ self._H.T + self.R

        # Kalman gain
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Si = np.linalg.pinv(S)
        K = self.P @ self._H.T @ Si

        # Update state and covariance
        self.x = self.x + K @ y
        # Joseph form: numerically stable covariance update
        # Prevents P from going negative-definite over long tracks
        IKH = np.eye(8, dtype=np.float32) - K @ self._H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def mahalanobis(self, bbox: Tuple[float, float, float, float]) -> float:
        """
        Compute Mahalanobis distance between predicted state and candidate bbox.
        
        This is a statistically-weighted distance that accounts for the
        uncertainty in our prediction. Returns a chi-squared-like value.
        Lower = better match.
        
        Args:
            bbox: Candidate bounding box (x1, y1, x2, y2).
            
        Returns:
            Mahalanobis distance (0 = perfect match, typically <9.48 for good match).
        """
        cx, cy, w, h = xyxy_to_cxcywh(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float32)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self.R
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Si = np.linalg.pinv(S)
        d = float(y @ Si @ y)
        return d if math.isfinite(d) else 1e9

    def predicted_center(self) -> Tuple[float, float]:
        """Get the predicted center (cx, cy) of the bounding box."""
        return float(self.x[0]), float(self.x[1])

    def predicted_bbox(self) -> Tuple[float, float, float, float]:
        """Get predicted bounding box as (x1, y1, x2, y2) — PRE-update estimate."""
        cx = float(self.x[0])
        cy = float(self.x[1])
        w = max(2.0, float(self.x[2]))
        h = max(2.0, float(self.x[3]))
        # Aspect ratio guard: width should never exceed height for persons
        if w > h:
            w = h * 0.5
        return cxcywh_to_xyxy(cx, cy, w, h)

    def updated_bbox(self) -> Tuple[float, float, float, float]:
        """
        Get corrected bounding box as (x1, y1, x2, y2) — POST-update estimate.
        Call AFTER update() to get the Kalman-corrected position (blends
        prediction + measurement). After update() is called, self.x already
        contains the corrected state, so this is equivalent to predicted_bbox().
        """
        return self.predicted_bbox()
