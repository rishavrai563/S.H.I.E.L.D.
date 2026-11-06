"""
Multi-Class Security Detector — YOLOv11m wrapper for Border Surveillance.
Detects:
  - Person (Class 0)
  - Car (Class 2)
  - Motorcycle (Class 3)
  - Bus (Class 5)
  - Truck (Class 7)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from ultralytics import YOLO


# Mapping COCO class IDs to operational categories
TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CLASS_IDS = list(TARGET_CLASSES.keys())  # [0, 2, 3, 5, 7]


@dataclass
class Detection:
    """A single object detection from YOLO."""
    bbox: Tuple[float, float, float, float]    # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    label: str

    def center(self) -> Tuple[float, float]:
        """Get center (cx, cy) of the bounding box."""
        return (self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5

    def area(self) -> float:
        """Get area of the bounding box."""
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def is_vehicle(self) -> bool:
        """Helper to check if detection is a vehicle."""
        return self.class_id in [2, 3, 5, 7]


class SecurityDetector:
    """
    Multi-class detector handling human and vehicle surveillance simultaneously.
    """

    def __init__(self,
                 model_path: str = "models/yolo11m.pt",
                 device: str = "cuda",
                 high_conf: float = 0.40,
                 low_conf: float = 0.10,
                 min_area: float = 120.0,
                 nms_iou: float = 0.60,
                 imgsz: int = 640):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        # FP16 inference on GPU: ~2x faster, half the VRAM. Ultralytics >= 8.4
        # replaced `half=True` with `quantize=16`; pick whichever this version knows.
        self._precision_kwargs = {}
        if device == "cuda":
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT
                new_api = "quantize" in DEFAULT_CFG_DICT
            except Exception:
                new_api = False
            self._precision_kwargs = {"quantize": 16} if new_api else {"half": True}
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.min_area = min_area
        self.nms_iou = nms_iou
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """
        Detect persons and vehicles in a single frame.
        Returns:
            (high_confidence_detections, low_confidence_detections)
        """
        return self.batch_detect([frame])[0]

    def batch_detect(self, frames: List[np.ndarray]) -> List[Tuple[List[Detection], List[Detection]]]:
        """
        Batched inference across multiple camera streams.
        """
        if not frames:
            return []

        # Frames of different shapes in one batch are all padded to an imgsz x imgsz
        # square; same-shape groups keep a tight rectangle (~1.7x less work for 16:9).
        groups: Dict[Tuple[int, int], List[int]] = {}
        for i, f in enumerate(frames):
            groups.setdefault(tuple(f.shape[:2]), []).append(i)

        all_dets: List[Tuple[List[Detection], List[Detection]]] = [([], []) for _ in frames]
        for idx in groups.values():
            results = self.model.predict(
                [frames[i] for i in idx],
                classes=CLASS_IDS,
                conf=self.low_conf,
                iou=self.nms_iou,
                imgsz=self.imgsz,
                device=self.device,
                **self._precision_kwargs,
                verbose=False,
            )
            for i, result in zip(idx, results):
                all_dets[i] = self._parse(result)
        return all_dets

    def _parse(self, result) -> Tuple[List[Detection], List[Detection]]:
        """Split one YOLO result into (high, low) detections.

        All boxes are copied GPU->CPU in one transfer: rows of [x1, y1, x2, y2, conf, cls].
        """
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []
        data = result.boxes.data
        rows = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        for x1, y1, x2, y2, conf, cls_id in rows.tolist():
            det = Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=int(cls_id),
                            label=TARGET_CLASSES.get(int(cls_id), "unknown"))
            if det.area() < self.min_area:
                continue
            (high_dets if conf >= self.high_conf else low_dets).append(det)
        return high_dets, low_dets