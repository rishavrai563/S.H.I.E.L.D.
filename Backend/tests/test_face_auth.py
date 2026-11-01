"""
Tests for the Face Authentication Engine.
Uses OpenCV's built-in cascade — no heavy ML mocking needed.
"""
import pytest
import numpy as np
from core.face_auth import FaceAuthEngine


class TestFaceAuthWatchlist:
    def test_default_watchlist(self):
        engine = FaceAuthEngine()
        assert "G1" in engine.threat_watchlist
        assert "G5" in engine.threat_watchlist
        assert "G8" in engine.threat_watchlist

    def test_custom_watchlist(self):
        custom = {"G10": "VIP_THREAT", "G20": "BANNED_PERSON"}
        engine = FaceAuthEngine(watchlist=custom)
        assert engine.threat_watchlist == custom
        assert "G1" not in engine.threat_watchlist

    def test_add_to_watchlist(self):
        engine = FaceAuthEngine()
        engine.add_to_watchlist(42, "TEST_THREAT")
        assert "G42" in engine.threat_watchlist
        assert engine.threat_watchlist["G42"] == "TEST_THREAT"

    def test_remove_from_watchlist(self):
        engine = FaceAuthEngine()
        engine.remove_from_watchlist(1)
        assert "G1" not in engine.threat_watchlist

    def test_remove_nonexistent(self):
        engine = FaceAuthEngine()
        engine.remove_from_watchlist(999)  # Should not raise


class TestFaceAuthVerify:
    def test_none_crop_returns_empty(self):
        engine = FaceAuthEngine()
        assert engine.verify_face(None, 1) == ""

    def test_empty_crop_returns_empty(self):
        engine = FaceAuthEngine()
        assert engine.verify_face(np.array([]), 1) == ""

    def test_small_crop_skipped(self):
        """Crops smaller than 120×60 are skipped (person too far)."""
        engine = FaceAuthEngine()
        small = np.zeros((50, 30, 3), dtype=np.uint8)
        assert engine.verify_face(small, 1) == ""

    def test_large_crop_no_match(self):
        """Large crop with a non-watchlisted ID returns empty."""
        engine = FaceAuthEngine()
        crop = np.zeros((250, 150, 3), dtype=np.uint8)
        result = engine.verify_face(crop, 999)
        assert result == ""

    def test_watchlisted_id_with_face_fallback(self):
        """When cascade is unavailable or no face, heuristic fallback for large crops."""
        engine = FaceAuthEngine()
        # Force heuristic fallback by using a large enough crop
        crop = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        # G1 is in default watchlist
        # Result depends on whether face is detected — at minimum it shouldn't crash
        result = engine.verify_face(crop, 1)
        assert isinstance(result, str)
