"""
Stream Manager — Threaded camera capture for SHIELD.
Includes automatic night-time / low-light enhancement for border surveillance.

Night mode (per camera) is decided by `night_vision.mode`:
  "auto"      — night if the local clock is in the night window OR the scene is dark
  "clock"     — night only inside the night window (default 18:00–06:00)
  "luminance" — night only when the scene is dark (mean luma, with hysteresis)
  "on"        — always night (fixed IR / floodlit night cameras)
  "off"       — never
While in night mode, CLAHE is applied to the L channel in LAB space before
the frame reaches the detectors, and `is_night` is exposed for analytics.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# URL schemes treated as live network streams (IP cameras, NVRs, phone camera apps)
NETWORK_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://")


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
    night_mode: str = ""                  # "" = use global night_vision.mode; on | off | auto | clock | luminance
    width: int = 0                        # webcam only: requested capture size (0 = driver default)
    height: int = 0
    loop: bool = False                    # video files only: restart at the end (demo mode)
    rotate: int = 0                       # 0 | 90 | 180 | 270 degrees clockwise (sideways phone feeds)
    realtime: bool = False                # video files: play at native speed like a live camera
                                          # (analysis samples the newest frame; video stays smooth)


DEFAULT_NIGHT_VISION = {
    "mode": "auto",
    "night_start_hour": 18,
    "night_end_hour": 6,
    "dark_luma_enter": 70.0,     # mean luma below this → dark
    "dark_luma_exit": 90.0,      # must rise above this to leave dark state
    "luma_check_every_n": 15,    # measure brightness every N frames
    "clahe_clip_limit": 2.5,
}


class CameraStream(threading.Thread):
    """
    One per camera. Continuously captures frames in a background thread.
    Only the latest frame is stored — no memory buildup, no backlog.
    """

    def __init__(self, config: CameraConfig, reconnect_delay: float = 2.0,
                 night_vision: Optional[dict] = None):
        super().__init__(daemon=True, name=f"CameraStream-{config.camera_id}")
        self.night_cfg = {**DEFAULT_NIGHT_VISION, **(night_vision or {})}
        override = str(getattr(config, "night_mode", "") or "").lower()
        if override:  # per-camera override, e.g. a fixed IR / floodlit night camera
            self.night_cfg["mode"] = override
        self._is_dark = False
        self._is_night = False
        self._mean_luma: float = -1.0
        self.config = config
        self.camera_id = config.camera_id
        self.source = config.source
        self.reconnect_delay = reconnect_delay

        # Detect source type. A numeric source ("0", 1) is a local webcam index;
        # rtsp/http(s)/rtmp/udp/tcp URLs are network streams (IP cameras, phone apps
        # such as IP Webcam: http://<phone-ip>:8080/video); anything else is a file.
        if isinstance(self.source, str) and self.source.strip().isdigit():
            self.source = int(self.source.strip())
        src = self.source.strip().lower() if isinstance(self.source, str) else ""
        self._is_webcam = isinstance(self.source, int)
        self._is_rtsp = src.startswith(("rtsp://", "rtsps://"))
        self._is_network = src.startswith(NETWORK_PREFIXES)
        self._is_live = self._is_webcam or self._is_network       # reconnects, never "ends"
        self._is_video_file = not self._is_live
        self._fresh = False  # live sources: has the latest frame been handed out yet?
        # Real-time files behave like live cameras (drop frames, never block the reader)
        self._realtime = self._is_video_file and bool(getattr(config, 'realtime', False))
        self._drops_frames = self._is_live or self._realtime
        # Display copy: newest frame for the dashboard video, never consumed by analysis
        self._display: Optional[Tuple[np.ndarray, float]] = None
        self._display_seq = 0
        self.frame_ready: Optional[threading.Event] = None  # set by StreamManager
        rot = int(getattr(config, "rotate", 0) or 0) % 360
        self._rotate_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                             270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)

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
        # Signals instead of 1 ms polling loops (polling threads fight YOLO for the GIL)
        self._consumed_evt = threading.Event(); self._consumed_evt.set()

        # Initialize CLAHE for Night-Time Vision Enhancement
        self._clahe = cv2.createCLAHE(clipLimit=float(self.night_cfg["clahe_clip_limit"]), tileGridSize=(8, 8))

    def _open_capture(self) -> bool:
        """Open the video capture with optimal settings."""
        try:
            if self._is_rtsp:
                # RTSP over TCP: avoids smeared frames from UDP packet loss on Wi-Fi
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            if self._is_network:
                # Timeouts so a phone that drops off Wi-Fi can't hang the capture thread
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000,
                ])
                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self._is_webcam and sys.platform == "win32":
                # DirectShow opens laptop webcams in ~1 s (MSMF can take 10-30 s)
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self.source)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap or not self._cap.isOpened():
                return False

            if self._is_webcam and self.config.width and self.config.height:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = fps if fps and fps > 0 else 30.0  # DirectShow/MJPEG streams report -1/0
            self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self._rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                self._frame_w, self._frame_h = self._frame_h, self._frame_w
            self._connected = True
            return True

        except Exception as e:
            print(f"  [WARN] {self.camera_id}: Failed to open {self.source}: {e}")
            return False
            
    def _in_night_window(self, hour: Optional[int] = None) -> bool:
        hour = datetime.now().hour if hour is None else hour
        start, end = int(self.night_cfg["night_start_hour"]), int(self.night_cfg["night_end_hour"])
        return (hour >= start or hour < end) if start > end else (start <= hour < end)

    def _update_darkness(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        self._mean_luma = float(gray.mean())
        if self._is_dark:
            self._is_dark = self._mean_luma < float(self.night_cfg["dark_luma_exit"])
        else:
            self._is_dark = self._mean_luma < float(self.night_cfg["dark_luma_enter"])

    def evaluate_night_mode(self, frame: np.ndarray, frame_index: int = 0, hour: Optional[int] = None) -> bool:
        mode = str(self.night_cfg.get("mode", "auto")).lower()
        if mode == "on":
            self._is_night = True
            return True
        if mode == "off":
            self._is_night = False
            return False
        if mode in ("auto", "luminance") and frame_index % max(1, int(self.night_cfg["luma_check_every_n"])) == 0:
            self._update_darkness(frame)
        clock = self._in_night_window(hour)
        if mode == "clock":
            self._is_night = clock
        elif mode == "luminance":
            self._is_night = self._is_dark
        else:
            self._is_night = clock or self._is_dark
        return self._is_night

    @property
    def is_night(self) -> bool:
        return self._is_night

    @property
    def mean_luma(self) -> float:
        return self._mean_luma

    def _enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the Lightness channel while in night mode."""
        if self.evaluate_night_mode(frame, self._frame_count):
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
            if not self._is_live:
                self._running = False
                return
            print(f"  [INFO] {self.camera_id}: will keep retrying every {self.reconnect_delay:.0f}s")

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if self._is_live:
                    # Phone app restarted / Wi-Fi blip / webcam busy: keep retrying quietly
                    time.sleep(self.reconnect_delay)
                    if self._running and self._open_capture():
                        print(f"  [STREAM] {self.camera_id} ✓ reconnected ({self._frame_w}×{self._frame_h})")
                    continue
                else:
                    break

            ret, frame = self._cap.read()
            if not ret:
                if self._is_video_file and self.config.loop and self._frame_count > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # demo: replay from the start
                    continue
                if self._is_video_file:
                    with self._lock:
                        self._latest = None
                    self._connected = False
                    self._running = False
                    break
                # Live source dropped: release and fall through to the reconnect path
                print(f"  [WARN] {self.camera_id}: stream lost, reconnecting...")
                self._connected = False
                with self._lock:
                    self._latest = None
                self._cap.release()
                self._cap = None
                continue
            
            if self._rotate_code is not None:
                frame = cv2.rotate(frame, self._rotate_code)  # upright before any analysis

            # --- NEW: Process Night-Time Frame Enhancement ---
            enhanced_frame = self._enhance_night_vision(frame)

            now = time.time()
            self._frame_count += 1
            self._last_frame_time = now

            if self._realtime:
                # Pace a file at its native frame rate, like a camera would deliver it
                next_due = getattr(self, "_next_due", now) + 1.0 / max(1.0, self._fps)
                self._next_due = max(next_due, now - 0.5)  # don't "catch up" after a stall
                delay = self._next_due - time.time()
                if delay > 0:
                    time.sleep(delay)
                now = time.time()
            elif self._is_video_file:
                while not self._consumed_evt.wait(0.1):
                    if not self._running:
                        break

            with self._lock:
                self._latest = (enhanced_frame, now)
                self._display = (enhanced_frame, now)
                self._display_seq += 1
                if self.frame_ready is not None:
                    self.frame_ready.set()
                if self._drops_frames:
                    self._fresh = True  # overwrite (drop stale frames), never block
                else:
                    self._consumed = False
                    self._consumed_evt.clear()

        self._connected = False

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        with self._lock:
            data = self._latest
            if not self._drops_frames and data is not None:
                self._latest = None
                self._consumed = True
                self._consumed_evt.set()
            elif self._drops_frames:
                # Hand each live frame out once, so a frozen camera isn't re-analysed
                if not self._fresh:
                    return None
                self._fresh = False
            return data

    def latest_display(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Newest captured frame for the live video view: (frame, timestamp, sequence).
        Unlike read(), this never consumes the frame."""
        with self._lock:
            if self._display is None:
                return None
            return self._display[0], self._display[1], self._display_seq

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
    def __init__(self, night_vision: Optional[dict] = None):
        self.streams: Dict[str, CameraStream] = {}
        self.configs: Dict[str, CameraConfig] = {}
        self.night_vision = night_vision or {}
        self._reconnect_delay: Dict[str, float] = {}
        self.frame_ready = threading.Event()  # any camera produced a frame

    def add_camera(self, config: CameraConfig, reconnect_delay: float = 2.0) -> None:
        stream = CameraStream(config, reconnect_delay=reconnect_delay, night_vision=self.night_vision)
        stream.frame_ready = self.frame_ready
        self.streams[config.camera_id] = stream
        self.configs[config.camera_id] = config
        self._reconnect_delay[config.camera_id] = reconnect_delay

    def restart_camera(self, camera_id: str) -> bool:
        """Re-create and start a stopped camera stream with its original settings."""
        config = self.configs.get(camera_id)
        if config is None:
            return False
        old = self.streams.get(camera_id)
        if old is not None and old.is_running:
            return False
        stream = CameraStream(config, reconnect_delay=self._reconnect_delay.get(camera_id, 2.0),
                              night_vision=self.night_vision)
        stream.frame_ready = self.frame_ready
        self.streams[camera_id] = stream
        stream.start()
        return True

    def is_night(self, camera_id: str) -> bool:
        stream = self.streams.get(camera_id)
        return bool(stream is not None and stream.is_night)

    def start_all(self) -> None:
        for cam_id, stream in self.streams.items():
            print(f"  [STREAM] Starting {cam_id}: {stream.source}")
            stream.start()

        # Files open instantly; webcams / phones over Wi-Fi can take a few seconds
        deadline = time.time() + (5.0 if any(s._is_live for s in self.streams.values()) else 0.5)
        while time.time() < deadline and not all(s.is_alive_stream or not s.is_running
                                                 for s in self.streams.values()):
            time.sleep(0.1)

        for cam_id, stream in self.streams.items():
            if stream.is_alive_stream:
                w, h = stream.frame_size
                print(f"  [STREAM] {cam_id} ✓ connected ({w}×{h} @ {stream.fps:.0f}fps)")
            else:
                print(f"  [STREAM] {cam_id} ✗ not connected")

    def wait_for_frame(self, timeout: float = 0.1) -> bool:
        """Block until some camera has produced a new frame (no busy polling)."""
        ok = self.frame_ready.wait(timeout)
        self.frame_ready.clear()
        return ok

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