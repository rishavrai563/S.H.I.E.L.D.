"""
SHIELD Facial Recognition System (FRS)
======================================
Software-only facial recognition for existing CCTV feeds — no FRS hardware.

Pipeline
--------
  1. Face detection
       YuNet (OpenCV DNN, ONNX, ~230 KB) — fast, landmark-aware, CPU friendly.
       Falls back to the Haar cascade bundled with OpenCV if the model is absent.
  2. Alignment
       5-point landmark alignment to the canonical 112x112 ArcFace template
       (via cv2.FaceRecognizerSF.alignCrop) when landmarks are available.
  3. Deep embedding
       Pluggable backends:
         - "sface"   : OpenCV SFace (ArcFace-family margin loss, 128-d, ONNX, CPU)
         - "facenet" : facenet-pytorch InceptionResnetV1 / VGGFace2 (512-d)
       "auto" picks facenet if installed, else sface.
  4. Gallery matching
       Cosine distance (1 - cosine similarity) against every enrolled
       reference embedding. A match requires distance <= threshold (default 0.6)
       and a margin over the runner-up identity, so look-alikes don't trigger.
  5. Temporal confirmation (live video)
       A track must hit the same identity in >= confirm_hits checks before an
       alert is raised, which suppresses single-frame false positives.

The gallery is persisted to disk (JSON) so enrolments survive restarts.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("shield.frs")

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"
DEFAULT_GALLERY_PATH = "data/face_gallery.json"

# Categories that should raise a CRITICAL alert when matched.
THREAT_CATEGORIES = {"SUSPECT", "WANTED", "BLACKLISTED", "SMUGGLER", "INTRUDER"}
VALID_CATEGORIES = THREAT_CATEGORIES | {"MISSING", "AUTHORIZED", "VIP"}


def _resolve(path: str) -> str:
    """Resolve a path relative to the Backend directory."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(BACKEND_DIR, path)


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class FaceDetection:
    """A detected face in image coordinates."""
    bbox: Tuple[int, int, int, int]          # x, y, w, h
    score: float
    landmarks: Optional[np.ndarray] = None   # (5, 2) or None
    raw: Optional[np.ndarray] = None         # YuNet row (15 values) for alignCrop

    @property
    def area(self) -> int:
        return int(self.bbox[2] * self.bbox[3])


@dataclass
class FaceMatch:
    """Result of comparing one face against the gallery."""
    identity_id: str
    name: str
    category: str
    similarity: float
    distance: float
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    is_match: bool = True

    @property
    def is_threat(self) -> bool:
        return self.category in THREAT_CATEGORIES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "name": self.name,
            "category": self.category,
            "similarity": round(float(self.similarity), 4),
            "distance": round(float(self.distance), 4),
            "bbox": [int(v) for v in self.bbox],
            "is_match": bool(self.is_match),
            "is_threat": self.is_threat,
        }


@dataclass
class EnrolledIdentity:
    identity_id: str
    name: str
    category: str = "SUSPECT"
    notes: str = ""
    enrolled_at: float = field(default_factory=time.time)
    embeddings: List[np.ndarray] = field(default_factory=list)
    thumbnail_b64: str = ""
    backend: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "name": self.name,
            "category": self.category,
            "notes": self.notes,
            "enrolled_at": self.enrolled_at,
            "num_references": len(self.embeddings),
            "thumbnail": self.thumbnail_b64,
            "backend": self.backend,
        }


@dataclass
class _TrackFaceState:
    last_check: float = float("-inf")
    candidate_id: str = ""
    hits: int = 0
    confirmed: Optional[FaceMatch] = None
    confirmed_at: float = 0.0


# ═══════════════════════════════════════════════════════════
#  Face detectors
# ═══════════════════════════════════════════════════════════

class _YuNetDetector:
    name = "yunet"

    def __init__(self, model_path: str, score_threshold: float = 0.8):
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        out: List[FaceDetection] = []
        if faces is None:
            return out
        for row in faces:
            x, y, fw, fh = [int(round(v)) for v in row[:4]]
            lms = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            out.append(FaceDetection(bbox=(x, y, fw, fh), score=float(row[14]),
                                     landmarks=lms, raw=np.asarray(row, dtype=np.float32)))
        return out


class _HaarDetector:
    name = "haar"

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._det = cv2.CascadeClassifier(path)
        if self._det.empty():
            raise RuntimeError("Haar cascade could not be loaded")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        faces = self._det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        return [FaceDetection(bbox=(int(x), int(y), int(w), int(h)), score=1.0) for (x, y, w, h) in faces]


# ═══════════════════════════════════════════════════════════
#  Embedding backends
# ═══════════════════════════════════════════════════════════

class _SFaceBackend:
    """OpenCV SFace — ArcFace-family recogniser, 128-d embeddings."""
    name = "sface"
    dim = 128

    def __init__(self, model_path: str):
        self._rec = cv2.FaceRecognizerSF.create(model_path, "")

    def embed(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        if face.raw is not None:
            aligned = self._rec.alignCrop(image, face.raw)
        else:
            aligned = cv2.resize(_crop_with_margin(image, face.bbox, 0.15), (112, 112))
        return _l2(self._rec.feature(aligned))


class _FaceNetBackend:
    """facenet-pytorch InceptionResnetV1 (VGGFace2) — 512-d embeddings."""
    name = "facenet"
    dim = 512

    def __init__(self, device: str = "cpu"):
        import torch  # noqa: F401
        from facenet_pytorch import InceptionResnetV1
        self._torch = torch
        self._device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)

    def embed(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        crop = cv2.resize(_crop_with_margin(image, face.bbox, 0.20), (160, 160))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = self._torch.from_numpy((rgb - 127.5) / 128.0).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            emb = self._model(tensor).cpu().numpy()[0]
        return _l2(emb)


def _crop_with_margin(image: np.ndarray, bbox: Tuple[int, int, int, int], margin: float) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else image


# ═══════════════════════════════════════════════════════════
#  Facial Recognition Engine
# ═══════════════════════════════════════════════════════════

class FaceRecognitionEngine:
    """
    Detect → align → embed → match faces against an enrolled watchlist gallery.

    Thread-safe: the API thread may enrol while the ML thread recognises.
    """

    def __init__(
        self,
        backend: str = "auto",
        detector_model: str = DEFAULT_DETECTOR_MODEL,
        recognizer_model: str = DEFAULT_SFACE_MODEL,
        gallery_path: Optional[str] = DEFAULT_GALLERY_PATH,
        match_distance_threshold: float = 0.60,
        margin: float = 0.05,
        min_face_px: int = 28,
        detector_score_threshold: float = 0.80,
        check_interval_sec: float = 0.5,
        confirm_hits: int = 2,
        recheck_sec: float = 30.0,
        device: str = "cpu",
        enabled: bool = True,
    ):
        print("👤 Initializing SHIELD Facial Recognition System (FRS)...")
        self.match_distance_threshold = float(match_distance_threshold)
        self.margin = float(margin)
        self.min_face_px = int(min_face_px)
        self.check_interval_sec = float(check_interval_sec)
        self.confirm_hits = max(1, int(confirm_hits))
        self.recheck_sec = float(recheck_sec)
        self.gallery_path = _resolve(gallery_path) if gallery_path else None

        self._lock = threading.RLock()
        self._identities: Dict[str, EnrolledIdentity] = {}
        self._track_state: Dict[str, _TrackFaceState] = {}
        self.stats = {"faces_detected": 0, "faces_recognized": 0, "checks": 0}

        self.detector = None
        self.embedder = None
        if not enabled:
            print("   ⚠️ FRS disabled by configuration")
            return

        # ── Detector ──
        det_path = _resolve(detector_model)
        try:
            if det_path and os.path.isfile(det_path) and hasattr(cv2, "FaceDetectorYN"):
                self.detector = _YuNetDetector(det_path, detector_score_threshold)
            else:
                self.detector = _HaarDetector()
                print(f"   ⚠️ YuNet model not found at {det_path} — using Haar cascade "
                      f"(run scripts/download_models.py)")
        except Exception as e:  # pragma: no cover — depends on OpenCV build
            logger.warning("Face detector init failed: %s", e)
            self.detector = None

        # ── Embedding backend ──
        self.embedder = self._build_backend(backend, _resolve(recognizer_model), device)

        self._load_gallery()
        det_name = getattr(self.detector, "name", "none")
        emb_name = getattr(self.embedder, "name", "none")
        print(f"   ✅ Detector: {det_name} | Embedder: {emb_name} "
              f"({self.embedding_dim}-d) | Gallery: {len(self._identities)} identities")

    # ──────────────────────────────────────────────────────
    #  Setup helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _build_backend(backend: str, sface_path: str, device: str):
        order = {"auto": ["facenet", "sface"], "facenet": ["facenet", "sface"], "sface": ["sface"]}
        for name in order.get(backend, ["sface"]):
            try:
                if name == "facenet":
                    return _FaceNetBackend(device)
                if name == "sface":
                    if not (sface_path and os.path.isfile(sface_path)):
                        raise FileNotFoundError(sface_path)
                    return _SFaceBackend(sface_path)
            except Exception as e:
                if backend != "auto" or name != "facenet":
                    logger.warning("FRS backend %s unavailable: %s", name, e)
        print("   ⚠️ No face embedding model available — FRS runs in detection-only mode "
              "(run scripts/download_models.py)")
        return None

    @property
    def available(self) -> bool:
        """True when full recognition (not just detection) is possible."""
        return self.detector is not None and self.embedder is not None

    @property
    def embedding_dim(self) -> int:
        return int(getattr(self.embedder, "dim", 0))

    @property
    def backend_name(self) -> str:
        return getattr(self.embedder, "name", "none")

    # ──────────────────────────────────────────────────────
    #  Core primitives
    # ──────────────────────────────────────────────────────

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces, largest first. Filters faces smaller than min_face_px."""
        if self.detector is None or image is None or getattr(image, "size", 0) == 0:
            return []
        try:
            with self._lock:
                faces = self.detector.detect(image)
        except Exception as e:
            logger.debug("face detection failed: %s", e)
            return []
        faces = [f for f in faces if min(f.bbox[2], f.bbox[3]) >= self.min_face_px]
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def extract_embedding(self, image: np.ndarray, face: FaceDetection) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        try:
            with self._lock:
                return self.embedder.embed(image, face)
        except Exception as e:
            logger.debug("embedding failed: %s", e)
            return None

    def match_embedding(self, embedding: np.ndarray) -> Optional[FaceMatch]:
        """
        Compare an embedding with the gallery. Returns the best candidate
        (is_match=False if it fails the distance or margin gate), or None
        when the gallery is empty.
        """
        emb = _l2(embedding)
        scored: List[Tuple[float, EnrolledIdentity]] = []
        with self._lock:
            for ident in self._identities.values():
                refs = [r for r in ident.embeddings if r.shape == emb.shape]
                if refs:
                    scored.append((float(np.max(np.stack(refs) @ emb)), ident))
        if not scored:
            return None
        scored.sort(key=lambda s: s[0], reverse=True)
        sim, ident = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        distance = 1.0 - sim
        is_match = distance <= self.match_distance_threshold and (sim - second) >= self.margin
        return FaceMatch(identity_id=ident.identity_id, name=ident.name, category=ident.category,
                         similarity=sim, distance=distance, is_match=is_match)

    def identify(self, image: np.ndarray, max_faces: int = 10) -> List[Dict[str, Any]]:
        """Detect every face in an image and match each against the gallery."""
        results = []
        for face in self.detect_faces(image)[:max_faces]:
            entry: Dict[str, Any] = {"bbox": [int(v) for v in face.bbox], "score": round(face.score, 3), "match": None}
            emb = self.extract_embedding(image, face)
            if emb is not None:
                m = self.match_embedding(emb)
                if m is not None:
                    m.bbox = face.bbox
                    entry["match"] = m.to_dict()
            results.append(entry)
        return results

    # ──────────────────────────────────────────────────────
    #  Enrollment / gallery management
    # ──────────────────────────────────────────────────────

    def enroll_face(self, person_name: str, image, category: str = "SUSPECT",
                    notes: str = "", identity_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enrol a reference photo. `image` may be a BGR ndarray or a list of them.
        Adds to an existing identity when identity_id (or an identical name) exists.
        Raises ValueError when no usable face is found.
        """
        if not self.available:
            raise RuntimeError("Face recognition models are not loaded (run scripts/download_models.py)")
        name = (person_name or "").strip()
        if not name:
            raise ValueError("person_name is required")
        category = (category or "SUSPECT").strip().upper()
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")

        images = image if isinstance(image, (list, tuple)) else [image]
        new_embs: List[np.ndarray] = []
        thumb = ""
        rejected = 0
        for img in images:
            if img is None or getattr(img, "size", 0) == 0:
                rejected += 1
                continue
            faces = self.detect_faces(img)
            if not faces:
                rejected += 1
                continue
            face = faces[0]  # largest face is the subject
            emb = self.extract_embedding(img, face)
            if emb is None:
                rejected += 1
                continue
            new_embs.append(emb)
            if not thumb:
                thumb = self._thumbnail(img, face)

        if not new_embs:
            raise ValueError("No clear face found in the supplied image(s)")

        with self._lock:
            ident = self._identities.get(identity_id) if identity_id else None
            if ident is None:
                ident = next((i for i in self._identities.values() if i.name.lower() == name.lower()), None)
            if ident is None:
                ident = EnrolledIdentity(identity_id=identity_id or f"F{uuid.uuid4().hex[:8].upper()}",
                                         name=name, category=category, notes=notes,
                                         thumbnail_b64=thumb, backend=self.backend_name)
                self._identities[ident.identity_id] = ident
            else:
                ident.category = category
                if notes:
                    ident.notes = notes
                if not ident.thumbnail_b64:
                    ident.thumbnail_b64 = thumb
            ident.embeddings.extend(new_embs)
            ident.embeddings = ident.embeddings[-20:]  # cap references per identity
            self._save_gallery()
            summary = ident.summary()

        summary.update({"references_added": len(new_embs), "images_rejected": rejected})
        return summary

    def remove_identity(self, identity_id: str) -> bool:
        with self._lock:
            removed = self._identities.pop(identity_id, None) is not None
            if removed:
                for st in self._track_state.values():
                    if st.confirmed and st.confirmed.identity_id == identity_id:
                        st.confirmed = None
                self._save_gallery()
        return removed

    def list_identities(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [i.summary() for i in self._identities.values()]

    @property
    def gallery_size(self) -> int:
        return len(self._identities)

    @staticmethod
    def _thumbnail(image: np.ndarray, face: FaceDetection) -> str:
        crop = _crop_with_margin(image, face.bbox, 0.25)
        crop = cv2.resize(crop, (96, 96))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    # ──────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────

    def _save_gallery(self) -> None:
        if not self.gallery_path:
            return
        try:
            os.makedirs(os.path.dirname(self.gallery_path), exist_ok=True)
            payload = {"version": 1, "identities": [
                {**{k: v for k, v in i.summary().items() if k != "num_references"},
                 "embeddings": [e.astype(float).round(6).tolist() for e in i.embeddings]}
                for i in self._identities.values()
            ]}
            tmp = self.gallery_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.gallery_path)
        except Exception as e:
            logger.warning("Could not save face gallery: %s", e)

    def _load_gallery(self) -> None:
        if not self.gallery_path or not os.path.isfile(self.gallery_path):
            return
        try:
            with open(self.gallery_path) as f:
                payload = json.load(f)
            for item in payload.get("identities", []):
                embs = [_l2(np.asarray(e, dtype=np.float32)) for e in item.get("embeddings", [])]
                if self.embedding_dim and embs and embs[0].shape[0] != self.embedding_dim:
                    logger.warning("Skipping %s: enrolled with a different face model (%s)",
                                   item.get("name"), item.get("backend"))
                    continue
                ident = EnrolledIdentity(
                    identity_id=item["identity_id"], name=item["name"],
                    category=item.get("category", "SUSPECT"), notes=item.get("notes", ""),
                    enrolled_at=item.get("enrolled_at", time.time()), embeddings=embs,
                    thumbnail_b64=item.get("thumbnail", ""), backend=item.get("backend", ""))
                self._identities[ident.identity_id] = ident
        except Exception as e:
            logger.warning("Could not load face gallery: %s", e)

    # ──────────────────────────────────────────────────────
    #  Live-video recognition (per tracked person)
    # ──────────────────────────────────────────────────────

    def _head_region(self, person_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """Upper body region, upscaled when small so distant faces are detectable."""
        h, w = person_crop.shape[:2]
        head = person_crop[: max(1, int(h * 0.5)), :]
        scale = 1.0
        if w < 200:
            scale = min(3.0, 200.0 / max(1, w))
            head = cv2.resize(head, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return head, scale

    def recognize_track(self, person_crop: np.ndarray, track_key: str,
                        now: Optional[float] = None) -> Tuple[Optional[FaceMatch], bool]:
        """
        Throttled, temporally-confirmed recognition for a tracked person.

        Returns (match, newly_confirmed):
          match           — the confirmed FaceMatch for this track (or None)
          newly_confirmed — True only on the check that confirms the identity,
                            so callers raise exactly one alert per sighting.
        """
        now = time.time() if now is None else now
        st = self._track_state.setdefault(track_key, _TrackFaceState())

        if st.confirmed is not None and now - st.confirmed_at < self.recheck_sec:
            return st.confirmed, False
        if now - st.last_check < self.check_interval_sec:
            return st.confirmed, False
        st.last_check = now

        if not self.available or self.gallery_size == 0:
            return None, False
        if person_crop is None or getattr(person_crop, "size", 0) == 0:
            return st.confirmed, False
        ph, pw = person_crop.shape[:2]
        if ph < 48 or pw < 20:
            return st.confirmed, False

        self.stats["checks"] += 1
        head, scale = self._head_region(person_crop)
        faces = self.detect_faces(head)
        if not faces:
            return st.confirmed, False
        self.stats["faces_detected"] += 1

        face = faces[0]
        emb = self.extract_embedding(head, face)
        match = self.match_embedding(emb) if emb is not None else None
        if match is None or not match.is_match:
            st.candidate_id, st.hits = "", 0
            return st.confirmed, False

        x, y, w, h = face.bbox
        match.bbox = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        if match.identity_id == st.candidate_id:
            st.hits += 1
        else:
            st.candidate_id, st.hits = match.identity_id, 1

        if st.hits >= self.confirm_hits:
            already = st.confirmed is not None and st.confirmed.identity_id == match.identity_id
            st.confirmed, st.confirmed_at = match, now
            if not already:
                self.stats["faces_recognized"] += 1
            return match, not already
        return st.confirmed, False

    def clear_track(self, track_key: str) -> None:
        self._track_state.pop(track_key, None)

    def prune_tracks(self, active_keys) -> None:
        """Drop per-track state for tracks that no longer exist."""
        active = set(active_keys)
        for key in [k for k in self._track_state if k not in active]:
            del self._track_state[key]

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, "gallery_size": self.gallery_size, "backend": self.backend_name,
                "detector": getattr(self.detector, "name", "none"), "embedding_dim": self.embedding_dim,
                "threshold": self.match_distance_threshold}


# Backward-compatible name used by older imports.
FaceAuthEngine = FaceRecognitionEngine
