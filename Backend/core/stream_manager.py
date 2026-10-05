"""
Stream Manager — Threaded camera capture for SHIELD.
Upgraded with Auto-Night-Time Enhancement (CLAHE) for low-light border surveillance.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Camera configuration loaded from cameras.yaml."""
    camera_id: str
    source: Union[int, str]               # 0 (webcam), "rtsp://...", "video.mp4"
    name: str = "Camera"
    building: str = ""
    floor: int = 0
    gps_lat: float = 0.0
    gps_lng: float = 0.0


class CameraStream(threading.Thread):
    """
    One per camera. Continuously captures frames in a background thread.
    Only the latest frame is stored — no memory buildup, no backlog.
    """

    def __init__(self, config: CameraConfig, reconnect_delay: float = 2.0):
        super().__init__(daemon=True, name=f"CameraStream-{config.camera_id}")
        self.config = config
        self.camera_id = config.camera_id
        self.source = config.source
        self.reconnect_delay = reconnect_delay

        # Detect source type
        self._is_video_file = isinstance(self.source, str) and not self.source.startswith("rtsp")
        self._is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp")

        # Single-slot frame buffer (lock-protected)
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._lock = threading.Lock()

        # State
        self._running = False
        self._connected = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._fps: float = 30.0
        self._frame_w: int = 0
        self._frame_h: int = 0

        # Stats
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0
        self._consumed = True  

        # Initialize CLAHE for Night-Time Vision Enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def _open_capture(self) -> bool:
        """Open the video capture with optimal settings."""
        try:
            if self._is_rtsp:
                # RTSP: use FFMPEG backend with TCP for reliability
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap or not self._cap.isOpened():
                return False

            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._connected = True
            return True

        except Exception as e:
            print(f"  [WARN] {self.camera_id}: Failed to open {self.source}: {e}")
            return False
            
    def _enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the Lightness channel if it is night time."""
        current_hour = datetime.now().hour
        # Activate enhancement between 6:00 PM (18) and 6:00 AM (6)
        if current_hour >= 18 or current_hour < 6:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel to enhance contrast without altering colors too much
            enhanced_l = self._clahe.apply(l_channel)
            
            # Merge back and convert to BGR
            merged = cv2.merge((enhanced_l, a, b))
            enhanced_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return enhanced_frame
        return frame

    def run(self) -> None:
        """Main capture loop — runs in daemon thread."""
        self._running = True
        self._start_time = time.time()

        if not self._open_capture():
            print(f"  [ERROR] {self.camera_id}: Cannot open source: {self.source}")
            if not self._is_rtsp:
                self._running = False
                return

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if self._is_rtsp:
                    print(f"  [INFO] {self.camera_id}: Reconnecting to {self.source}...")
                    time.sleep(self.reconnect_delay)
                    self._open_capture()
                    continue
                else:
                    break

            ret, frame = self._cap.read()
            if not ret:
                if self._is_video_file:
                    with self._lock:
                        self._latest = None
                    self._connected = False
                    self._running = False
                    break
                elif self._is_rtsp:
                    self._connected = False
                    with self._lock:
                        self._latest = None
                    self._cap.release()
                    self._cap = None
                    continue
                else:
                    time.sleep(0.01)
                    continue
            
            # --- NEW: Process Night-Time Frame Enhancement ---
            enhanced_frame = self._enhance_night_vision(frame)

            now = time.time()
            self._frame_count += 1
            self._last_frame_time = now

            if self._is_video_file:
                while not self._consumed and self._running:
                    time.sleep(0.001)

            with self._lock:
                self._latest = (enhanced_frame, now)
                if self._is_video_file:
                    self._consumed = False

        self._connected = False

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            data = self._latest
            if self._is_video_file and data is not None:
                self._latest = None   
                self._consumed = True  
            return data

    def stop(self) -> None:
        self._running = False

    def release(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_alive_stream(self) -> bool:
        return self._running and self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_w, self._frame_h


class StreamManager:
    def __init__(self):
        self.streams: Dict[str, CameraStream] = {}
        self.configs: Dict[str, CameraConfig] = {}

    def add_camera(self, config: CameraConfig, reconnect_delay: float = 2.0) -> None:
        stream = CameraStream(config, reconnect_delay=reconnect_delay)
        self.streams[config.camera_id] = stream
        self.configs[config.camera_id] = config

    def start_all(self) -> None:
        for cam_id, stream in self.streams.items():
            print(f"  [STREAM] Starting {cam_id}: {stream.source}")
            stream.start()

        time.sleep(0.5)

        for cam_id, stream in self.streams.items():
            if stream.is_alive_stream:
                w, h = stream.frame_size
                print(f"  [STREAM] {cam_id} ✓ connected ({w}×{h} @ {stream.fps:.0f}fps)")
            else:
                print(f"  [STREAM] {cam_id} ✗ not connected")

    def grab_latest(self) -> Dict[str, Tuple[np.ndarray, float]]:
        frames: Dict[str, Tuple[np.ndarray, float]] = {}
        for cam_id, stream in self.streams.items():
            data = stream.read()
            if data is not None:
                frames[cam_id] = data
        return frames

    def stop_all(self) -> None:
        for stream in self.streams.values():
            stream.stop()
        for stream in self.streams.values():
            stream.join(timeout=2.0)
        for stream in self.streams.values():
            stream.release()

    def get_camera_ids(self) -> List[str]:
        return list(self.streams.keys())

    def get_config(self, camera_id: str) -> Optional[CameraConfig]:
        return self.configs.get(camera_id)

    def all_stopped(self) -> bool:
        return all(not s.is_running for s in self.streams.values())

    @property
    def camera_count(self) -> int:
        return len(self.streams)