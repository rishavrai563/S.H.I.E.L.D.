"""
SHIELD — Production ANPR Engine v3
==================================
Software-only Automatic Number Plate Recognition for ordinary CCTV.

  Stage 1  Plate localisation
           Bilateral filter → Sobel-X gradient → Otsu threshold → morphological
           close → contour filtering (aspect 2.5:1 – 5:1, area, lower-body bias).

  Stage 2  Adaptive pre-processing
           Upscale to OCR height → bilateral denoise → CLAHE → deskew
           (cv2.minAreaRect on edge pixels).

  Stage 3  Multi-pass OCR (EasyOCR, character allow-list)
           Pass 1  lower 50 % "bumper" crop
           Pass 2  localised plate region(s)
           Pass 3  full vehicle crop (fallback)
           Segments are merged row-by-row so two-line plates (trucks, bikes)
           read correctly. The best candidate across passes wins; a pass that
           yields a format-valid, high-confidence read exits early.

  Stage 4  Format correction (Indian plates, configurable)
           Position-aware fixes for OCR confusions: letters expected in the
           state/series slots, digits in the district/number slots
           (e.g. "MHO1AB1234" → "MH01AB1234"). Supports BH-series.

  Stage 5  Temporal voting
           Per-vehicle-track accumulator. A plate is CONFIRMED only after
           N (default 2) concordant reads, eliminating one-frame hallucinations.

  Stage 6  Watchlist matching
           OCR-confusion-tolerant weighted edit distance (0↔O, 1↔I, 8↔B, 5↔S,
           2↔Z, 6↔G …) against a persistent tactical watchlist.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr
except ImportError:  # pragma: no cover — handled at runtime
    easyocr = None

logging.getLogger("easyocr").setLevel(logging.ERROR)
logger = logging.getLogger("shield.anpr")

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Indian formats: standard "MH01AB1234", older "DL3CAB1234", "MH121234"; BH "22BH1234AB"
_IN_STANDARD = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")
_IN_BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

# OCR confusion pairs (symmetric)
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1",
             "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}
_CONFUSABLE = {frozenset(p) for p in [("0", "O"), ("0", "D"), ("0", "Q"), ("1", "I"), ("1", "L"),
                                       ("8", "B"), ("5", "S"), ("2", "Z"), ("6", "G"), ("7", "T"),
                                       ("4", "A"), ("I", "L"), ("O", "D"), ("O", "Q")]}
CONFUSION_COST = 0.3


# ═══════════════════════════════════════════════════════════
#  Plate text utilities
# ═══════════════════════════════════════════════════════════

def normalize_plate(text: str) -> str:
    return _NON_ALNUM.sub("", (text or "").upper())


def is_valid_indian_plate(text: str) -> bool:
    if _IN_BH.match(text):
        return True
    return 8 <= len(text) <= 10 and bool(_IN_STANDARD.match(text))


def correct_indian_plate(text: str) -> str:
    """Position-aware OCR correction for Indian plates. Returns text unchanged if it can't fit."""
    t = normalize_plate(text)
    if len(t) < 8 or len(t) > 10:
        return t

    def as_digits(s):
        return "".join(_TO_DIGIT.get(c, c) for c in s)

    def as_letters(s):
        return "".join(_TO_LETTER.get(c, c) for c in s)

    # BH series: NN BH NNNN SS
    if len(t) in (9, 10) and as_letters(t[2:4]) in ("BH",) and as_digits(t[:2]).isdigit():
        cand = as_digits(t[:2]) + "BH" + as_digits(t[4:8]) + as_letters(t[8:])
        if _IN_BH.match(cand):
            return cand

    head = as_letters(t[:2])
    tail = as_digits(t[-4:])
    rest = t[2:-4]
    # district code is 1–2 digits; series letters (0–3) follow
    dist_len = 2 if len(rest) >= 2 and as_digits(rest[:2]).isdigit() else 1
    cand = head + as_digits(rest[:dist_len]) + as_letters(rest[dist_len:]) + tail
    return cand if is_valid_indian_plate(cand) else t


def plate_distance(a: str, b: str) -> float:
    """Weighted Levenshtein distance; OCR-confusable substitutions cost CONFUSION_COST."""
    a, b = normalize_plate(a), normalize_plate(b)
    if a == b:
        return 0.0
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)] + [0.0] * len(b)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                sub = 0.0
            elif frozenset((ca, cb)) in _CONFUSABLE:
                sub = CONFUSION_COST
            else:
                sub = 1.0
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub)
        prev = cur
    return prev[-1]


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    """Result from a single-frame plate extraction attempt."""
    text: str = ""
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None   # plate region inside the crop (x, y, w, h)
    method: str = "none"                               # bumper | localized | full_crop
    valid_format: bool = False
    raw_text: str = ""

    @property
    def score(self) -> float:
        return self.confidence + (0.25 if self.valid_format else 0.0)


@dataclass
class WatchlistHit:
    watch_plate: str
    tag: str
    notes: str
    distance: float
    exact: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"watch_plate": self.watch_plate, "tag": self.tag, "notes": self.notes,
                "distance": round(self.distance, 2), "exact": self.exact}


@dataclass
class ConfirmedPlate:
    """A temporally validated plate."""
    text: str
    confidence: float
    camera_id: str = ""
    track_id: int = -1
    watchlist_tag: str = ""
    watchlist_hit: Optional[WatchlistHit] = None
    read_count: int = 0
    valid_format: bool = False
    timestamp: float = field(default_factory=time.time)
    vehicle_class: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"plate": self.text, "confidence": round(self.confidence, 3), "camera_id": self.camera_id,
                "track_id": self.track_id, "read_count": self.read_count, "valid_format": self.valid_format,
                "timestamp": self.timestamp, "vehicle_class": self.vehicle_class,
                "watchlist": self.watchlist_hit.to_dict() if self.watchlist_hit else None}


@dataclass
class _VoteAccumulator:
    votes: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    attempts: int = 0
    last_attempt: float = float("-inf")
    confirmed: Optional[ConfirmedPlate] = None


# ═══════════════════════════════════════════════════════════
#  ANPR Engine
# ═══════════════════════════════════════════════════════════

class ANPREngine:
    def __init__(
        self,
        use_gpu: bool = False,
        min_plate_chars: int = 5,
        max_plate_chars: int = 11,
        ocr_confidence_threshold: float = 0.20,
        temporal_votes_required: int = 2,
        temporal_window_size: int = 12,
        ocr_interval_sec: float = 0.25,
        min_vehicle_width_px: int = 60,
        early_accept_confidence: float = 0.60,
        aspect_ratio_range: Tuple[float, float] = (2.5, 5.0),
        plate_format: str = "IN",
        require_valid_format: bool = False,
        watchlist_max_distance: float = 1.0,
        watchlist: Optional[Dict[str, Any]] = None,
        watchlist_path: Optional[str] = "data/plate_watchlist.json",
        reader: Any = None,
    ):
        print("🚗 Initializing Production ANPR Engine v3 (localisation + multi-pass OCR + voting)...")
        if reader is not None:
            self.reader = reader
        elif easyocr is not None:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        else:
            self.reader = None
            print("   ⚠️ easyocr not installed — ANPR disabled (pip install easyocr)")

        self.min_chars = int(min_plate_chars)
        self.max_chars = int(max_plate_chars)
        self.ocr_conf_threshold = float(ocr_confidence_threshold)
        self.votes_required = max(1, int(temporal_votes_required))
        self.window_size = max(self.votes_required, int(temporal_window_size))
        self.ocr_interval_sec = float(ocr_interval_sec)
        self.min_vehicle_width = int(min_vehicle_width_px)
        self.early_accept = float(early_accept_confidence)
        self.aspect_min, self.aspect_max = float(aspect_ratio_range[0]), float(aspect_ratio_range[1])
        self.plate_format = (plate_format or "").upper()
        self.require_valid_format = bool(require_valid_format)
        self.watchlist_max_distance = float(watchlist_max_distance)
        self.watchlist_path = (watchlist_path if not watchlist_path or os.path.isabs(watchlist_path)
                               else os.path.join(BACKEND_DIR, watchlist_path))

        self._lock = threading.RLock()
        self._ocr_lock = threading.Lock()
        self._track_votes: Dict[str, _VoteAccumulator] = {}
        self.recent_plates: Deque[ConfirmedPlate] = deque(maxlen=500)
        self.stats = {"ocr_calls": 0, "confirmed": 0, "watchlist_hits": 0}

        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

        self.watchlist: Dict[str, Dict[str, Any]] = {}
        if watchlist is not None:
            self.update_watchlist(watchlist, persist=False)
        elif not self._load_watchlist():
            self.update_watchlist(self._default_watchlist(), persist=False)

        print(f"   ✅ Voting: {self.votes_required} concordant reads | Watchlist: {len(self.watchlist)} plates"
              f" | Format: {self.plate_format or 'any'}")

    # ──────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self.reader is not None

    def extract_plate(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        """Single-frame extraction (backward compatible). Returns (text, confidence)."""
        r = self._extract_single_frame(vehicle_crop)
        return r.text, r.confidence

    def read_plate(self, vehicle_crop: np.ndarray) -> PlateResult:
        """Single-frame extraction with full detail."""
        return self._extract_single_frame(vehicle_crop)

    def extract_plate_with_voting(self, vehicle_crop: np.ndarray, track_key: str,
                                  now: Optional[float] = None, camera_id: str = "",
                                  track_id: int = -1, vehicle_class: str = "") -> Optional[ConfirmedPlate]:
        """
        Temporally validated extraction for a tracked vehicle.
        Returns a ConfirmedPlate exactly once — on the read that reaches consensus.
        """
        now = time.time() if now is None else now
        with self._lock:
            acc = self._track_votes.setdefault(track_key, _VoteAccumulator())
            if acc.confirmed is not None:
                return None
            if now - acc.last_attempt < self.ocr_interval_sec:
                return None
            acc.last_attempt = now
        if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
            return None
        if vehicle_crop.shape[1] < self.min_vehicle_width:
            return None

        result = self._extract_single_frame(vehicle_crop)
        with self._lock:
            acc.attempts += 1
            if result.text:
                acc.votes[result.text].append(result.confidence)
                reads = acc.votes[result.text]
                if len(reads) >= self.votes_required:
                    hit = self.check_watchlist_detailed(result.text)
                    confirmed = ConfirmedPlate(
                        text=result.text, confidence=float(np.mean(reads)), camera_id=camera_id,
                        track_id=track_id, read_count=len(reads), valid_format=result.valid_format,
                        watchlist_tag=hit.tag if hit else "", watchlist_hit=hit, timestamp=time.time(),
                        vehicle_class=vehicle_class)
                    acc.confirmed = confirmed
                    self.recent_plates.append(confirmed)
                    self.stats["confirmed"] += 1
                    if hit:
                        self.stats["watchlist_hits"] += 1
                    return confirmed
            if acc.attempts >= self.window_size:
                # window expired without consensus — start a fresh window
                self._track_votes[track_key] = _VoteAccumulator(last_attempt=now)
        return None

    def get_confirmed(self, track_key: str) -> Optional[ConfirmedPlate]:
        acc = self._track_votes.get(track_key)
        return acc.confirmed if acc else None

    def clear_track(self, track_key: str) -> None:
        with self._lock:
            self._track_votes.pop(track_key, None)

    def recognize_image(self, image: np.ndarray) -> Dict[str, Any]:
        """One-shot recognition for an uploaded image (API)."""
        r = self._extract_single_frame(image)
        hit = self.check_watchlist_detailed(r.text) if r.text else None
        return {"plate": r.text, "raw_text": r.raw_text, "confidence": round(r.confidence, 3),
                "method": r.method, "valid_format": r.valid_format,
                "bbox": list(r.bbox) if r.bbox else None, "watchlist": hit.to_dict() if hit else None}

    # ──────────────────────────────────────────────────────
    #  Watchlist
    # ──────────────────────────────────────────────────────

    def check_watchlist(self, plate_text: str) -> str:
        hit = self.check_watchlist_detailed(plate_text)
        return hit.tag if hit else ""

    def check_watchlist_detailed(self, plate_text: str) -> Optional[WatchlistHit]:
        plate = normalize_plate(plate_text)
        if not plate:
            return None
        with self._lock:
            entries = list(self.watchlist.items())
        best: Optional[WatchlistHit] = None
        for watch_plate, meta in entries:
            if watch_plate == plate:
                return WatchlistHit(watch_plate, meta["tag"], meta.get("notes", ""), 0.0, True)
            if min(len(plate), len(watch_plate)) < 6 or abs(len(plate) - len(watch_plate)) > 1:
                continue
            d = plate_distance(plate, watch_plate)
            if d <= self.watchlist_max_distance and (best is None or d < best.distance):
                best = WatchlistHit(watch_plate, meta["tag"], meta.get("notes", ""), d, False)
        return best

    def update_watchlist(self, entries: Dict[str, Any], persist: bool = True) -> None:
        """Merge entries. Values may be a tag string or {"tag":..., "notes":...}."""
        with self._lock:
            for plate, value in entries.items():
                key = normalize_plate(plate)
                if not key:
                    continue
                meta = value if isinstance(value, dict) else {"tag": str(value)}
                self.watchlist[key] = {"tag": str(meta.get("tag", "BLACKLISTED")).upper(),
                                       "notes": str(meta.get("notes", "")),
                                       "added_at": float(meta.get("added_at", time.time()))}
            if persist:
                self._save_watchlist()

    def remove_from_watchlist(self, plate: str) -> bool:
        with self._lock:
            removed = self.watchlist.pop(normalize_plate(plate), None) is not None
            if removed:
                self._save_watchlist()
        return removed

    def list_watchlist(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"plate": p, **m} for p, m in sorted(self.watchlist.items())]

    def _save_watchlist(self) -> None:
        if not self.watchlist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.watchlist_path), exist_ok=True)
            tmp = self.watchlist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.watchlist, f, indent=2)
            os.replace(tmp, self.watchlist_path)
        except Exception as e:
            logger.warning("Could not save plate watchlist: %s", e)

    def _load_watchlist(self) -> bool:
        if not self.watchlist_path or not os.path.isfile(self.watchlist_path):
            return False
        try:
            with open(self.watchlist_path) as f:
                self.update_watchlist(json.load(f), persist=False)
            return True
        except Exception as e:
            logger.warning("Could not load plate watchlist: %s", e)
            return False

    @staticmethod
    def _default_watchlist() -> Dict[str, Dict[str, str]]:
        """Demo entries; replace via the API or data/plate_watchlist.json."""
        return {
            "DL01AB1234": {"tag": "BLACKLISTED", "notes": "Demo: stolen vehicle"},
            "HR26DK1234": {"tag": "BLACKLISTED", "notes": "Demo: smuggling suspect"},
            "UP32GH5678": {"tag": "RESTRICTED", "notes": "Demo: restricted zone access"},
            "RJ14CD9999": {"tag": "BLACKLISTED", "notes": "Demo: wanted vehicle"},
            "MH01AV0001": {"tag": "VIP", "notes": "Demo: convoy"},
        }

    # ──────────────────────────────────────────────────────
    #  Stage 1 — plate localisation
    # ──────────────────────────────────────────────────────

    def localize_plates(self, image: np.ndarray, max_candidates: int = 2) -> List[Tuple[int, int, int, int]]:
        """
        Return up to max_candidates plate-like regions (x, y, w, h), best first.

        Two complementary cues are fused:
          (a) character-edge density: Sobel-X → Otsu → morphological close.
              The blob covers the characters, so the plate box is estimated by
              growing its height (characters fill ~65 % of plate height).
          (b) plate background: bright (white/yellow) rectangle on darker body.
        Every candidate must pass the plate aspect gate (default 2.5:1 – 5:1),
        an area gate, and is scored with a lower-body bias.
        """
        h, w = image.shape[:2]
        if h < 20 or w < 40:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        min_area, max_area = w * h * 0.004, w * h * 0.40
        candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []

        def add(x, y, bw, bh, weight):
            x, y = max(0, int(x)), max(0, int(y))
            bw, bh = min(w - x, int(bw)), min(h - y, int(bh))
            if bw <= 0 or bh <= 0:
                return
            aspect, area = bw / bh, bw * bh
            if self.aspect_min <= aspect <= self.aspect_max and min_area <= area <= max_area:
                cy = y + bh / 2
                candidates.append((weight * area * (0.5 + cy / h), (x, y, bw, bh)))

        # (a) edge-density cue
        sobel = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3))
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kw = max(17, int(w * 0.06)) | 1
        kernel = self._close_kernel if kw == 17 else cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(cv2.erode(closed, self._open_kernel), self._open_kernel)
        for cnt in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            fill = cv2.contourArea(cnt) / max(1.0, bw * bh)
            if fill < 0.35:
                continue
            plate_h = bh / 0.65
            add(x - bw * 0.05, y + bh / 2 - plate_h / 2, bw * 1.10, plate_h, 0.8 + 0.4 * fill)

        # (b) bright plate-background cue
        _, bright = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for cnt in cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_fill = cv2.contourArea(cnt) / max(1.0, bw * bh)
            if rect_fill >= 0.75:
                add(x, y, bw, bh, 1.0 + 0.5 * rect_fill)

        candidates.sort(key=lambda c: c[0], reverse=True)
        regions: List[Tuple[int, int, int, int]] = []
        for _, (x, y, bw, bh) in candidates:
            box = (x, y, x + bw, y + bh)
            if any(_box_iou(box, (rx, ry, rx + rw, ry + rh)) > 0.4 for rx, ry, rw, rh in regions):
                continue
            px, py = int(bw * 0.05), int(bh * 0.12)
            x1, y1 = max(0, x - px), max(0, y - py)
            x2, y2 = min(w, x + bw + px), min(h, y + bh + py)
            regions.append((x1, y1, x2 - x1, y2 - y1))
            if len(regions) >= max_candidates:
                break
        return regions

    # ──────────────────────────────────────────────────────
    #  Stage 2 — preprocessing
    # ──────────────────────────────────────────────────────

    def preprocess(self, crop: np.ndarray, target_h: int = 64) -> np.ndarray:
        h, w = crop.shape[:2]
        if h < 5 or w < 10:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if h < target_h:
            scale = target_h / h
            gray = cv2.resize(gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = self._clahe.apply(denoised)
        return self._deskew(enhanced)

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 50, 200)
        pts = cv2.findNonZero(edges)
        if pts is None or len(pts) < 20:
            return gray
        (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
        if rw < rh:
            angle -= 90.0
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) < 0.5 or abs(angle) > 20:
            return gray
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # ──────────────────────────────────────────────────────
    #  Stage 3 — multi-pass OCR
    # ──────────────────────────────────────────────────────

    def _extract_single_frame(self, vehicle_crop: np.ndarray) -> PlateResult:
        if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0 or self.reader is None:
            return PlateResult()
        try:
            h, w = vehicle_crop.shape[:2]
            best = PlateResult()

            def consider(res: PlateResult) -> bool:
                nonlocal best
                if res.text and res.score > best.score:
                    best = res
                return bool(res.text and res.valid_format and res.confidence >= self.early_accept)

            # Pass 1: lower 50 % bumper region
            y0 = int(h * 0.5)
            bumper = vehicle_crop[y0:, :]
            if bumper.size and consider(self._ocr_region(bumper, "bumper", (0, y0, w, h - y0))):
                return best

            # Pass 2: localised plate region(s) (searched in the whole crop, lower-biased)
            for (x, y, bw, bh) in self.localize_plates(vehicle_crop):
                region = vehicle_crop[y:y + bh, x:x + bw]
                if region.size and consider(self._ocr_region(region, "localized", (x, y, bw, bh))):
                    return best

            # Pass 3: full crop fallback
            if not best.text:
                consider(self._ocr_region(vehicle_crop, "full_crop", (0, 0, w, h)))
            return best
        except Exception as e:
            logger.warning("ANPR extraction error: %s", e)
            return PlateResult()

    def _ocr_region(self, region: np.ndarray, method: str, bbox) -> PlateResult:
        processed = self.preprocess(region)
        with self._ocr_lock:
            self.stats["ocr_calls"] += 1
            results = self.reader.readtext(processed, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)
        res = self._select_text(results or [])
        if res.text:
            res.method, res.bbox = method, tuple(int(v) for v in bbox)
        return res

    def _postprocess(self, raw: str) -> Tuple[str, bool]:
        text = normalize_plate(raw)
        if self.plate_format == "IN":
            text = correct_indian_plate(text)
            return text, is_valid_indian_plate(text)
        return text, False

    def _accept(self, text: str, valid: bool) -> bool:
        if not (self.min_chars <= len(text) <= self.max_chars):
            return False
        return valid or not self.require_valid_format

    def _select_text(self, results: List) -> PlateResult:
        best = PlateResult()
        segments = []
        for item in results:
            try:
                box, text, conf = item
            except (TypeError, ValueError):
                continue
            conf = float(conf)
            if conf < self.ocr_conf_threshold:
                continue
            segments.append((box, text, conf))
            plate, valid = self._postprocess(text)
            cand = PlateResult(text=plate, confidence=conf, valid_format=valid, raw_text=str(text))
            if self._accept(plate, valid) and cand.score > best.score:
                best = cand

        # Merge segments row-by-row (two-line plates, split reads)
        if len(segments) >= 2:
            merged_raw, merged_conf = self._merge_segments(segments)
            plate, valid = self._postprocess(merged_raw)
            cand = PlateResult(text=plate, confidence=merged_conf, valid_format=valid, raw_text=merged_raw)
            if self._accept(plate, valid) and (cand.score > best.score or (valid and not best.valid_format)):
                best = cand
        return best

    @staticmethod
    def _merge_segments(segments) -> Tuple[str, float]:
        def geom(box):
            pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            return float(pts[:, 0].min()), float(pts[:, 1].mean()), float(np.ptp(pts[:, 1]) or 1.0)

        items = sorted(((geom(b), t, c) for b, t, c in segments), key=lambda s: s[0][1])
        rows: List[List] = []
        for it in items:
            (x, cy, hgt), _, _ = it
            if rows and abs(cy - rows[-1][0][0][1]) < 0.6 * max(hgt, rows[-1][0][0][2]):
                rows[-1].append(it)
            else:
                rows.append([it])
        text = "".join("".join(t for _, t, _ in sorted(r, key=lambda s: s[0][0])) for r in rows)
        conf = float(np.mean([c for _, _, c in items]))
        return text, conf

    # ──────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            pending = sum(1 for v in self._track_votes.values() if v.confirmed is None)
        return {**self.stats, "pending_tracks": pending, "watchlist_size": len(self.watchlist),
                "votes_required": self.votes_required, "available": self.available}
