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
        results = self.model.predict(
            frame,
            classes=CLASS_IDS,         # Ingest persons + all vehicle classes
            conf=self.low_conf,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        high_dets: List[Detection] = []
        low_dets: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                bbox = tuple(box.xyxy[0].cpu().tolist())
                conf = float(box.conf.cpu())
                cls_id = int(box.cls.cpu().item())
                label = TARGET_CLASSES.get(cls_id, "unknown")

                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    label=label
                )

                if det.area() < self.min_area:
                    continue

                if conf >= self.high_conf:
                    high_dets.append(det)
                else:
                    low_dets.append(det)

        return high_dets, low_dets

    def batch_detect(self, frames: List[np.ndarray]) -> List[Tuple[List[Detection], List[Detection]]]:
        """
        Batched inference across multiple camera streams.
        """
        if not frames:
            return []

        results = self.model.predict(
            frames,
            classes=CLASS_IDS,
            conf=self.low_conf,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        all_dets: List[Tuple[List[Detection], List[Detection]]] = []

        for result in results:
            high_dets: List[Detection] = []
            low_dets: List[Detection] = []

            for box in result.boxes:
                bbox = tuple(box.xyxy[0].cpu().tolist())
                conf = float(box.conf.cpu())
                cls_id = int(box.cls.cpu().item())
                label = TARGET_CLASSES.get(cls_id, "unknown")

                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    label=label
                )

                if det.area() < self.min_area:
                    continue

                if conf >= self.high_conf:
                    high_dets.append(det)
                else:
                    low_dets.append(det)

            all_dets.append((high_dets, low_dets))

        return all_dets