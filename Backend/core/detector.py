"""
Person Detector — YOLOv11m wrapper with batch detection and confidence splitting.

Key design decisions:
- Batch detection: 4 frames → 1 GPU call (instead of 4 separate calls)
- Confidence split: High (≥0.40) for full matching, Low (0.10-0.40) for occlusion recovery
- Person class only (COCO class 0): no wasted computation on other objects
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """A single person detection from YOLO."""
    bbox: Tuple[float, float, float, float]    # (x1, y1, x2, y2)
    confidence: float
    
    def center(self) -> Tuple[float, float]:
        """Get center (cx, cy) of the bounding box."""
        return (self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5
    
    def area(self) -> float:
        """Get area of the bounding box."""
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


class PersonDetector:
    """
    YOLOv11m person detector with batch detection support.
    
    Usage:
        detector = PersonDetector("models/yolo11m.pt")
        
        # Single frame
        high_dets, low_dets = detector.detect(frame)
        
        # Batch of 4 frames (one GPU call)
        all_results = detector.batch_detect([frame1, frame2, frame3, frame4])
        for high_dets, low_dets in all_results:
            ...
    """

    def __init__(self, model_path: str = "models/yolo11m.pt",
                 device: str = "cuda",
                 high_conf: float = 0.40,
                 low_conf: float = 0.10,
                 min_area: float = 400.0,
                 nms_iou: float = 0.60,
                 imgsz: int = 640):
        """
        Args:
            model_path: Path to YOLOv11m weights.
            device:     "cuda" or "cpu".
            high_conf:  Threshold for high-confidence detections.
            low_conf:   Threshold for low-confidence detections (occlusion recovery).
            min_area:   Minimum bbox area in px² to accept (filters out tiny noise).
            nms_iou:    NMS IOU threshold for YOLO post-processing.
            imgsz:      Inference resolution passed to YOLO.
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.min_area = min_area
        self.nms_iou = nms_iou
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """
        Detect persons in a single frame.
        
        Args:
            frame: BGR image (numpy array).
            
        Returns:
            (high_confidence_detections, low_confidence_detections)
            High: conf >= high_conf
            Low:  low_conf <= conf < high_conf
        """
        results = self.model.predict(
            frame,
            classes=[0],              # Person class only
            conf=self.low_conf,       # Use low threshold, split afterwards
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
                det = Detection(bbox=bbox, confidence=conf)
                
                # Filter tiny detections
                if det.area() < self.min_area:
                    continue
                
                if conf >= self.high_conf:
                    high_dets.append(det)
                else:
                    low_dets.append(det)

        return high_dets, low_dets

    def batch_detect(self, frames: List[np.ndarray]) -> List[Tuple[List[Detection], List[Detection]]]:
        """
        Detect persons in multiple frames with a single GPU call.
        
        This is the preferred method for multi-camera processing:
        batch all 4 camera frames → one inference → split results per camera.
        Gives ~2-3× speedup over calling detect() 4 times.
        
        Args:
            frames: List of BGR images.
            
        Returns:
            List of (high_dets, low_dets) tuples, one per input frame.
        """
        if not frames:
            return []

        results = self.model.predict(
            frames,
            classes=[0],
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
                det = Detection(bbox=bbox, confidence=conf)

                if det.area() < self.min_area:
                    continue

                if conf >= self.high_conf:
                    high_dets.append(det)
                else:
                    low_dets.append(det)

            all_dets.append((high_dets, low_dets))

        return all_dets
