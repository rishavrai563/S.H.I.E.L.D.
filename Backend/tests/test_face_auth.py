"""
Tests for the Facial Recognition System (core/face_auth.py).

Unit tests inject a fake detector/embedder so they run anywhere.
Integration tests use the real YuNet + SFace ONNX models when they are present
in Backend/models/ (python scripts/download_models.py) and are skipped otherwise.
"""
import os
import json

import cv2
import numpy as np
import pytest

from core.face_auth import (
    FaceRecognitionEngine, FaceAuthEngine, FaceDetection, FaceMatch, THREAT_CATEGORIES,
)

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
HAS_MODELS = all(os.path.isfile(os.path.join(MODELS, f)) for f in
                 ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))


# ── Fakes ────────────────────────────────────────────────

class FakeDetector:
    """'Detects' one face per image; the image's mean colour encodes identity."""
    name = "fake"

    def detect(self, image):
        if image.mean() < 1:          # all-black image → no face
            return []
        h, w = image.shape[:2]
        return [FaceDetection(bbox=(w // 4, h // 4, w // 2, h // 2), score=0.99)]


class FakeEmbedder:
    name = "fake"
    dim = 8

    def embed(self, image, face):
        b, g, r = [float(c) for c in image.reshape(-1, 3).mean(axis=0)]
        v = np.array([b, g, r, b - g, g - r, r - b, 1.0, (b + g + r) / 3], dtype=np.float32)
        return v / np.linalg.norm(v)


def make_engine(tmp_path, **kw):
    eng = FaceRecognitionEngine(enabled=False, gallery_path=str(tmp_path / "gallery.json"), **kw)
    eng.detector, eng.embedder = FakeDetector(), FakeEmbedder()
    return eng


def solid(color, size=(200, 160)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = color
    return img


RED, BLUE = (20, 20, 230), (230, 20, 20)


# ── Unit tests ───────────────────────────────────────────

class TestEnrollment:
    def test_alias_kept_for_backward_compat(self):
        assert FaceAuthEngine is FaceRecognitionEngine

    def test_enroll_and_list(self, tmp_path):
        eng = make_engine(tmp_path)
        res = eng.enroll_face("Suspect A", solid(RED), category="wanted", notes="test")
        assert res["name"] == "Suspect A"
        assert res["category"] == "WANTED"
        assert res["num_references"] == 1
        assert res["thumbnail"]
        assert eng.gallery_size == 1

    def test_enroll_multiple_images_adds_references(self, tmp_path):
        eng = make_engine(tmp_path)
        res = eng.enroll_face("A", [solid(RED), solid((25, 25, 225))])
        assert res["num_references"] == 2
        again = eng.enroll_face("a", solid(RED))  # same name, case-insensitive → same identity
        assert again["identity_id"] == res["identity_id"]
        assert again["num_references"] == 3

    def test_enroll_rejects_image_without_face(self, tmp_path):
        eng = make_engine(tmp_path)
        with pytest.raises(ValueError):
            eng.enroll_face("Nobody", np.zeros((100, 100, 3), np.uint8))

    def test_enroll_rejects_bad_category_and_name(self, tmp_path):
        eng = make_engine(tmp_path)
        with pytest.raises(ValueError):
            eng.enroll_face("X", solid(RED), category="ALIEN")
        with pytest.raises(ValueError):
            eng.enroll_face("   ", solid(RED))

    def test_enroll_without_models_raises_runtime_error(self, tmp_path):
        eng = FaceRecognitionEngine(enabled=False, gallery_path=None)
        assert not eng.available
        with pytest.raises(RuntimeError):
            eng.enroll_face("A", solid(RED))

    def test_remove_identity(self, tmp_path):
        eng = make_engine(tmp_path)
        ident = eng.enroll_face("A", solid(RED))
        assert eng.remove_identity(ident["identity_id"]) is True
        assert eng.gallery_size == 0
        assert eng.remove_identity("missing") is False

    def test_gallery_persists_across_instances(self, tmp_path):
        eng = make_engine(tmp_path)
        eng.enroll_face("Persisted", solid(RED), category="SUSPECT")
        data = json.loads((tmp_path / "gallery.json").read_text())
        assert data["identities"][0]["name"] == "Persisted"

        eng2 = make_engine(tmp_path)
        eng2._load_gallery()
        assert eng2.gallery_size == 1
        assert eng2.list_identities()[0]["name"] == "Persisted"


class TestMatching:
    def test_match_same_identity(self, tmp_path):
        eng = make_engine(tmp_path)
        eng.enroll_face("Red", solid(RED))
        res = eng.identify(solid((22, 22, 228)))
        assert res[0]["match"]["name"] == "Red"
        assert res[0]["match"]["is_match"] is True
        assert res[0]["match"]["distance"] <= 0.6

    def test_no_match_for_different_identity(self, tmp_path):
        eng = make_engine(tmp_path)
        eng.enroll_face("Red", solid(RED))
        m = eng.match_embedding(FakeEmbedder().embed(solid(BLUE), None))
        assert m is not None and m.is_match is False

    def test_empty_gallery_returns_none(self, tmp_path):
        eng = make_engine(tmp_path)
        assert eng.match_embedding(np.ones(8, np.float32)) is None

    def test_margin_gate_rejects_ambiguous(self, tmp_path):
        eng = make_engine(tmp_path, margin=0.5)
        eng.enroll_face("Twin1", solid(RED))
        eng.enroll_face("Twin2", solid((21, 21, 229)))
        m = eng.match_embedding(FakeEmbedder().embed(solid(RED), None))
        assert m.is_match is False   # both twins are too close to separate

    def test_threshold_controls_strictness(self, tmp_path):
        eng = make_engine(tmp_path, match_distance_threshold=0.0)
        eng.enroll_face("Red", solid(RED))
        m = eng.match_embedding(FakeEmbedder().embed(solid((40, 40, 200)), None))
        assert m.is_match is False

    def test_face_match_threat_flag(self):
        assert FaceMatch("F1", "x", "SUSPECT", 0.9, 0.1).is_threat
        assert not FaceMatch("F1", "x", "AUTHORIZED", 0.9, 0.1).is_threat
        assert "WANTED" in THREAT_CATEGORIES


class TestTrackRecognition:
    def test_requires_confirm_hits_before_alert(self, tmp_path):
        eng = make_engine(tmp_path, confirm_hits=2, check_interval_sec=0.5)
        eng.enroll_face("Red", solid(RED))
        crop = solid(RED, (260, 120))
        m, new = eng.recognize_track(crop, "cam1:1", now=0.0)
        assert m is None and new is False
        m, new = eng.recognize_track(crop, "cam1:1", now=1.0)
        assert m is not None and m.name == "Red" and new is True
        # subsequent calls return the cached match without re-alerting
        m, new = eng.recognize_track(crop, "cam1:1", now=2.0)
        assert m.name == "Red" and new is False

    def test_rate_limited(self, tmp_path):
        eng = make_engine(tmp_path, confirm_hits=2, check_interval_sec=10)
        eng.enroll_face("Red", solid(RED))
        crop = solid(RED, (260, 120))
        eng.recognize_track(crop, "k", now=0.0)
        m, new = eng.recognize_track(crop, "k", now=1.0)  # throttled
        assert m is None and new is False
        assert eng.stats["checks"] == 1

    def test_tiny_or_empty_crops_skipped(self, tmp_path):
        eng = make_engine(tmp_path)
        eng.enroll_face("Red", solid(RED))
        assert eng.recognize_track(None, "a", now=0) == (None, False)
        assert eng.recognize_track(solid(RED, (30, 10)), "b", now=0) == (None, False)

    def test_no_gallery_no_checks(self, tmp_path):
        eng = make_engine(tmp_path)
        assert eng.recognize_track(solid(RED, (260, 120)), "a", now=0) == (None, False)
        assert eng.stats["checks"] == 0

    def test_prune_tracks(self, tmp_path):
        eng = make_engine(tmp_path, confirm_hits=1)
        eng.enroll_face("Red", solid(RED))
        eng.recognize_track(solid(RED, (260, 120)), "cam1:1", now=0)
        eng.recognize_track(solid(RED, (260, 120)), "cam1:2", now=0)
        eng.prune_tracks(["cam1:2"])
        assert set(eng._track_state) == {"cam1:2"}

    def test_removed_identity_clears_confirmed_tracks(self, tmp_path):
        eng = make_engine(tmp_path, confirm_hits=1)
        ident = eng.enroll_face("Red", solid(RED))
        m, _ = eng.recognize_track(solid(RED, (260, 120)), "t", now=0)
        assert m is not None
        eng.remove_identity(ident["identity_id"])
        assert eng._track_state["t"].confirmed is None


# ── Integration with the real ONNX models ────────────────

def synthetic_face(seed: int, size: int = 220) -> np.ndarray:
    """Procedurally drawn cartoon face; YuNet is not guaranteed to fire on it."""
    rng = np.random.RandomState(seed)
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (70, 90), 0, 0, 360, tuple(int(c) for c in rng.randint(120, 220, 3)), -1)
    return img


@pytest.mark.skipif(not HAS_MODELS, reason="face models not downloaded")
class TestRealModels:
    def test_models_load(self, tmp_path):
        eng = FaceRecognitionEngine(backend="sface", gallery_path=str(tmp_path / "g.json"))
        assert eng.available
        assert eng.backend_name == "sface"
        assert eng.embedding_dim == 128
        assert eng.detector.name == "yunet"

    def test_no_face_in_blank_image(self, tmp_path):
        eng = FaceRecognitionEngine(backend="sface", gallery_path=str(tmp_path / "g.json"))
        assert eng.detect_faces(np.zeros((240, 320, 3), np.uint8)) == []
        with pytest.raises(ValueError):
            eng.enroll_face("Nobody", np.zeros((240, 320, 3), np.uint8))

    def test_embedding_is_normalized(self, tmp_path):
        eng = FaceRecognitionEngine(backend="sface", gallery_path=str(tmp_path / "g.json"))
        face = FaceDetection(bbox=(40, 40, 120, 120), score=1.0)
        emb = eng.extract_embedding(synthetic_face(1), face)
        assert emb.shape == (128,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-4)


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "face_astronaut.jpg")


@pytest.mark.skipif(not (HAS_MODELS and os.path.isfile(FIXTURE)), reason="face models or fixture missing")
class TestRealFaceRecognition:
    """End-to-end on a real photo (NASA public-domain portrait)."""

    def _engine(self, tmp_path):
        return FaceRecognitionEngine(backend="sface", gallery_path=str(tmp_path / "g.json"))

    def test_detects_real_face(self, tmp_path):
        faces = self._engine(tmp_path).detect_faces(cv2.imread(FIXTURE))
        assert len(faces) == 1
        assert faces[0].landmarks is not None

    def test_enroll_then_recognize_under_variation(self, tmp_path):
        eng = self._engine(tmp_path)
        img = cv2.imread(FIXTURE)
        eng.enroll_face("Test Subject", img, category="SUSPECT")
        # mirrored, darker, down-scaled — as a CCTV camera would see it
        probe = cv2.convertScaleAbs(cv2.resize(cv2.flip(img, 1), None, fx=0.6, fy=0.6), alpha=0.7, beta=-10)
        res = eng.identify(probe)
        assert res and res[0]["match"]["name"] == "Test Subject"
        assert res[0]["match"]["is_match"] and res[0]["match"]["distance"] <= 0.6

    def test_unknown_person_not_matched(self, tmp_path, random_embedding):
        eng = self._engine(tmp_path)
        from core.face_auth import EnrolledIdentity
        eng._identities["FX"] = EnrolledIdentity("FX", "Someone Else",
                                                 embeddings=[random_embedding(3)[:128] / np.linalg.norm(random_embedding(3)[:128])])
        res = eng.identify(cv2.imread(FIXTURE))
        assert res[0]["match"]["is_match"] is False

    def test_live_track_raises_single_confirmation(self, tmp_path):
        eng = self._engine(tmp_path)
        img = cv2.imread(FIXTURE)
        eng.enroll_face("Test Subject", img)
        # person-shaped crop with a ~100 px face at the top, like a CCTV track box
        person = np.full((320, 130, 3), 70, np.uint8)
        person[0:130, 0:130] = cv2.resize(img, (130, 130))
        results = [eng.recognize_track(person, "cam1:5", now=t) for t in (0.0, 1.0, 2.0, 3.0)]
        assert sum(new for _, new in results) == 1
        assert results[-1][0].name == "Test Subject"
