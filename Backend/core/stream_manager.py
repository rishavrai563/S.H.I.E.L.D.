"""
Stream Manager — Threaded camera capture for multi-camera tracking.

Supports:
  - Webcams (int): 0, 1, 2...
  - RTSP streams (str): "rtsp://..."
  - Video files (str): "video.mp4"

Design:
  - Each camera runs in its own daemon thread
  - Single-slot buffer: only the latest frame is kept (no queue backlog)
  - Lock-protected reads: grab_latest() never blocks processing
  - Auto-reconnect for RTSP streams on failure
  - Graceful stop with thread join
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
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

        # Video file: frame-by-frame delivery (no dropping)
        # For live streams: single-slot overwrite (lowest latency)
        self._consumed = True  # True = capture thread may write next frame

    def _open_capture(self) -> bool:
        """Open the video capture with optimal settings."""
        try:
            if self._is_rtsp:
                # RTSP: use FFMPEG backend with TCP for reliability
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if self._cap.isOpened():
                    # Set buffer size to 1 for minimum latency
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap or not self._cap.isOpened():
                return False

            # Read stream properties
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._connected = True
            return True

        except Exception as e:
            print(f"  [WARN] {self.camera_id}: Failed to open {self.source}: {e}")
            return False

    def run(self) -> None:
        """Main capture loop — runs in daemon thread."""
        self._running = True
        self._start_time = time.time()

        if not self._open_capture():
            print(f"  [ERROR] {self.camera_id}: Cannot open source: {self.source}")
            if not self._is_rtsp:
                # Non-RTSP source: don't retry
                self._running = False
                return

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                # Reconnect (RTSP only)
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
                    # Video file ended — clear buffer so grab_latest returns None
                    with self._lock:
                        self._latest = None
                    self._connected = False
                    self._running = False
                    break
                elif self._is_rtsp:
                    # RTSP lost connection — clear buffer, will reconnect
                    self._connected = False
                    with self._lock:
                        self._latest = None
                    self._cap.release()
                    self._cap = None
                    continue
                else:
                    # Webcam error — brief retry
                    time.sleep(0.01)
                    continue

            now = time.time()
            self._frame_count += 1
            self._last_frame_time = now

            # For video files: wait until previous frame was consumed
            # This ensures EVERY frame is processed (no dropping)
            if self._is_video_file:
                while not self._consumed and self._running:
                    time.sleep(0.001)

            # Store latest frame (single-slot overwrite for live, blocking for video)
            with self._lock:
                self._latest = (frame, now)
                if self._is_video_file:
                    self._consumed = False

        # Loop exited: stream is no longer connected.
        self._connected = False

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the most recent frame and timestamp.
        Returns None if no frame is available yet.
        Thread-safe, never blocks.
        For video files: consumes the frame (clears buffer + signals capture thread).
        For live streams: returns latest frame without consuming (single-slot).
        """
        with self._lock:
            data = self._latest
            if self._is_video_file and data is not None:
                self._latest = None   # Clear so same frame isn't read twice
                self._consumed = True  # Signal capture thread to read next frame
            return data

    def stop(self) -> None:
        """Signal the capture thread to stop."""
        self._running = False

    def release(self) -> None:
        """Stop thread and release video capture resources."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_alive_stream(self) -> bool:
        """Check if the stream is still producing frames."""
        return self._running and self._connected

    @property
    def is_running(self) -> bool:
        """Check if stream thread is still intended to run (including reconnect loops)."""
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        return self._frame_w, self._frame_h


class StreamManager:
    """
    Manages all camera streams for the multi-camera tracking system.

    Usage:
        manager = StreamManager()
        manager.add_camera(CameraConfig("cam1", 0, "Main Gate"))
        manager.add_camera(CameraConfig("cam2", "rtsp://...", "Corridor"))
        manager.start_all()

        # In processing loop:
        frames = manager.grab_latest()
        for cam_id, (frame, timestamp) in frames.items():
            ...

        manager.stop_all()
    """

    def __init__(self):
        self.streams: Dict[str, CameraStream] = {}
        self.configs: Dict[str, CameraConfig] = {}

    def add_camera(self, config: CameraConfig,
                   reconnect_delay: float = 2.0) -> None:
        """Add a camera to the manager."""
        stream = CameraStream(config, reconnect_delay=reconnect_delay)
        self.streams[config.camera_id] = stream
        self.configs[config.camera_id] = config

    def start_all(self) -> None:
        """Start all camera capture threads."""
        for cam_id, stream in self.streams.items():
            print(f"  [STREAM] Starting {cam_id}: {stream.source}")
            stream.start()

        # Wait briefly for cameras to produce first frame
        time.sleep(0.5)

        # Report status
        for cam_id, stream in self.streams.items():
            if stream.is_alive_stream:
                w, h = stream.frame_size
                print(f"  [STREAM] {cam_id} ✓ connected ({w}×{h} @ {stream.fps:.0f}fps)")
            else:
                print(f"  [STREAM] {cam_id} ✗ not connected")

    def grab_latest(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Grab the latest frame from ALL cameras at once.

        Returns a dict of {camera_id: (frame, timestamp)}.
        Only includes cameras that have a frame available.
        This is the input to the batched detection pipeline.
        """
        frames: Dict[str, Tuple[np.ndarray, float]] = {}
        for cam_id, stream in self.streams.items():
            data = stream.read()
            if data is not None:
                frames[cam_id] = data
        return frames

    def stop_all(self) -> None:
        """Stop all camera threads and release resources."""
        for stream in self.streams.values():
            stream.stop()

        # Wait for threads to finish (with timeout)
        for stream in self.streams.values():
            stream.join(timeout=2.0)

        for stream in self.streams.values():
            stream.release()

    def get_camera_ids(self) -> List[str]:
        """Get list of all camera IDs."""
        return list(self.streams.keys())

    def get_config(self, camera_id: str) -> Optional[CameraConfig]:
        """Get camera config by ID."""
        return self.configs.get(camera_id)

    def all_stopped(self) -> bool:
        """
        Check if all streams have fully stopped.

        Important: RTSP streams may be temporarily disconnected while reconnecting.
        They are not considered stopped unless their run loop has exited.
        """
        return all(not s.is_running for s in self.streams.values())

    @property
    def camera_count(self) -> int:
        return len(self.streams)
