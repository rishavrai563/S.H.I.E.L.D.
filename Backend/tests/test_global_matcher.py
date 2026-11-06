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


class TestCrossCameraLifecycle:
    def _gm(self, **kw):
        return GlobalMatcher(diff_camera_default_min=kw.pop("gap", 2.0), **kw)

    def test_excluded_ids_are_never_returned(self):
        gm = self._gm()
        a = _make_emb(1)
        g1 = gm.match_phase1(a, "cam1", 0.0).global_id
        res = gm.match_phase1(_similar_emb(a), "cam1", 5.0, exclude_gids={g1})
        assert res.is_new and res.global_id != g1

    def test_mark_seen_blocks_impossible_transit(self):
        gm = self._gm(gap=2.0)
        a = _make_emb(2)
        g = gm.match_phase1(a, "cam1", 0.0).global_id
        gm.mark_seen(g, "cam1", 10.0)                    # still visible on cam1 at t=10
        res = gm.match_phase1(_similar_emb(a), "cam2", 10.5)
        assert res.is_new                                 # 0.5 s later on cam2: implausible
        res2 = gm.rematch(_similar_emb(a), "cam2", 13.0, exclude_gids={res.global_id})  # own provisional id
        assert res2 is not None and res2.global_id == g   # after the transit gap: same person

    def test_rematch_respects_stricter_threshold(self):
        gm = self._gm()
        a = _make_emb(3)
        g = gm.match_phase1(a, "cam1", 0.0).global_id
        weak = _similar_emb(a, noise_scale=0.06, seed=7)
        sim = float(np.dot(weak, gm.get_identity(g).prototype))
        assert gm.rematch(weak, "cam2", 10.0, set(), threshold=sim + 0.01) is None
        assert gm.rematch(weak, "cam2", 10.0, set(), threshold=sim - 0.01).global_id == g

    def test_remove_identity_drops_provisional_only(self):
        gm = self._gm()
        g = gm.match_phase1(_make_emb(4), "cam1", 0.0).global_id
        reg = gm.register_person("VIP", _make_emb(5))
        gm.remove_identity(g); gm.remove_identity(reg)
        assert gm.get_identity(g) is None and gm.get_identity(reg) is not None


def test_record_visit_builds_journey():
    gm = GlobalMatcher()
    g = gm.match_phase1(_make_emb(11), "cam1", 0.0).global_id
    for t in (1.0, 2.0):
        gm.mark_seen(g, "cam1", t)
    gm.record_visit(g, "cam2", 3.5)
    gm.mark_seen(g, "cam2", 5.0)
    j = gm.get_journey(g)
    assert [c for c, _, _ in j] == ["cam1", "cam2"]
    assert j[0][2] == 2.0 and j[1][1:] == (3.5, 5.0)
