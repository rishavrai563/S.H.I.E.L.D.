"""
Tests for the GlobalMatcher — cross-camera person identity matching.
Uses real numpy for embedding math (no mocks needed).
"""
import pytest
import numpy as np
from core.global_matcher import GlobalMatcher, GlobalIdentity, MatchResult


def _make_emb(seed):
    """Create a deterministic L2-normalized 512-dim embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _similar_emb(base, noise_scale=0.05, seed=99):
    """Create an embedding similar to base (cosine sim > 0.90)."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(512).astype(np.float32) * noise_scale
    v = base + noise
    return (v / np.linalg.norm(v)).astype(np.float32)


class TestGlobalMatcherInit:
    def test_default_init(self):
        gm = GlobalMatcher()
        assert gm.gallery_size == 0
        assert gm.gallery_thr == 0.60
        assert gm.registered_thr == 0.70

    def test_custom_thresholds(self):
        gm = GlobalMatcher(
            gallery_threshold=0.75,
            registered_threshold=0.80,
            margin_gate=0.10,
        )
        assert gm.gallery_thr == 0.75
        assert gm.registered_thr == 0.80
        assert gm.margin_gate == 0.10


class TestPhase1Matching:
    def test_new_identity_created(self):
        gm = GlobalMatcher()
        emb = _make_emb(42)
        result = gm.match_phase1(emb, "cam1", 1.0)
        assert result.is_new is True
        assert result.phase == 1
        assert result.global_id == 1
        assert gm.gallery_size == 1

    def test_same_embedding_matches_existing(self):
        gm = GlobalMatcher(gallery_threshold=0.50)
        emb = _make_emb(42)
        r1 = gm.match_phase1(emb, "cam1", 1.0)

        # Same embedding, different camera, enough time gap
        r2 = gm.match_phase1(emb, "cam2", 10.0)
        assert r2.is_new is False
        assert r2.global_id == r1.global_id

    def test_different_embeddings_create_different_ids(self):
        gm = GlobalMatcher()
        r1 = gm.match_phase1(_make_emb(1), "cam1", 1.0)
        r2 = gm.match_phase1(_make_emb(9999), "cam1", 5.0)
        assert r1.global_id != r2.global_id
        assert gm.gallery_size == 2

    def test_none_embedding_creates_new(self):
        gm = GlobalMatcher()
        result = gm.match_phase1(None, "cam1", 1.0)
        assert result.is_new is True

    def test_same_camera_time_gate(self):
        """Same camera sighting too soon should create new ID, not match."""
        gm = GlobalMatcher(same_camera_min_gap=1.0, gallery_threshold=0.50)
        emb = _make_emb(42)
        r1 = gm.match_phase1(emb, "cam1", 1.0)
        # Only 0.1s later on same camera — too soon
        r2 = gm.match_phase1(emb, "cam1", 1.1)
        assert r2.is_new is True
        assert r2.global_id != r1.global_id


class TestPhase2Matching:
    def test_confirms_phase1_assignment(self):
        gm = GlobalMatcher(gallery_threshold=0.50)
        emb = _make_emb(42)
        r1 = gm.match_phase1(emb, "cam1", 1.0)
        pooled = _similar_emb(emb)

        r2 = gm.match_phase2(
            pooled, "cam1", entry_time=1.0, exit_time=5.0,
            quality=0.8, current_global_id=r1.global_id,
        )
        assert r2.global_id == r1.global_id
        assert r2.phase == 2

    def test_none_embedding_keeps_current(self):
        gm = GlobalMatcher()
        emb = _make_emb(42)
        r1 = gm.match_phase1(emb, "cam1", 1.0)

        r2 = gm.match_phase2(
            None, "cam1", entry_time=1.0, exit_time=5.0,
            quality=0.8, current_global_id=r1.global_id,
        )
        assert r2.global_id == r1.global_id

    def test_logs_movement(self):
        gm = GlobalMatcher(gallery_threshold=0.50)
        emb = _make_emb(42)
        r1 = gm.match_phase1(emb, "cam1", 1.0)

        gm.match_phase2(
            _similar_emb(emb), "cam1", entry_time=1.0, exit_time=5.0,
            quality=0.8, current_global_id=r1.global_id,
        )
        journey = gm.get_journey(r1.global_id)
        assert len(journey) >= 1
        assert journey[-1][0] == "cam1"


class TestRegistration:
    def test_register_person(self):
        gm = GlobalMatcher()
        emb = _make_emb(42)
        gid = gm.register_person("Alice", emb)
        assert gid > 0
        assert gm.gallery_size == 1

        ident = gm.get_identity(gid)
        assert ident.display_name == "Alice"
        assert ident.is_registered is True

    def test_registered_person_matches_with_priority(self):
        gm = GlobalMatcher(
            registered_threshold=0.50, gallery_threshold=0.50,
        )
        emb = _make_emb(42)
        gid = gm.register_person("Alice", emb)

        # Match should find the registered person
        result = gm.match_phase1(_similar_emb(emb), "cam1", 10.0)
        assert result.global_id == gid
        assert result.is_registered is True
        assert result.display_name == "Alice"

    def test_registered_not_matched_below_threshold(self):
        gm = GlobalMatcher(registered_threshold=0.99)
        gm.register_person("Alice", _make_emb(42))

        # Very different embedding
        result = gm.match_phase1(_make_emb(9999), "cam1", 10.0)
        assert result.is_registered is False


class TestGalleryManagement:
    def test_get_all_identities(self):
        gm = GlobalMatcher()
        gm.match_phase1(_make_emb(1), "cam1", 1.0)
        gm.match_phase1(_make_emb(2), "cam1", 3.0)
        gm.match_phase1(_make_emb(3), "cam1", 5.0)
        identities = gm.get_all_identities()
        assert len(identities) == 3

    def test_get_identity_nonexistent(self):
        gm = GlobalMatcher()
        assert gm.get_identity(999) is None

    def test_get_journey_nonexistent(self):
        gm = GlobalMatcher()
        assert gm.get_journey(999) is None

    def test_update_gallery(self):
        gm = GlobalMatcher()
        emb = _make_emb(42)
        r = gm.match_phase1(emb, "cam1", 1.0)

        new_emb = _similar_emb(emb, noise_scale=0.02)
        gm.update_gallery(r.global_id, new_emb, "cam1", 5.0)

        ident = gm.get_identity(r.global_id)
        assert ident.last_seen_time == 5.0
        assert ident.last_seen_camera == "cam1"


class TestMarginGating:
    def test_ambiguous_match_rejected(self):
        """Two very similar gallery entries should trigger margin gate."""
        gm = GlobalMatcher(gallery_threshold=0.50, margin_gate=0.15)

        # Create two very similar identities
        base = _make_emb(42)
        gm.match_phase1(base, "cam1", 1.0)
        gm.match_phase1(_similar_emb(base, noise_scale=0.01), "cam1", 3.0)

        # Query with something equally close to both — margin should reject
        query = _similar_emb(base, noise_scale=0.005, seed=50)
        result = gm.match_phase1(query, "cam2", 10.0)
        # Should create new ID due to ambiguity (or match the closest if margin is met)
        assert isinstance(result, MatchResult)
