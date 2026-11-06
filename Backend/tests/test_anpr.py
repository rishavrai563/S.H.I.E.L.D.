"""
Tests for ANPR Engine. Mocks EasyOCR to avoid heavy model downloads.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestANPREngine:
    @patch("core.anpr.easyocr.Reader")
    def test_init(self, mock_reader_cls):
        from core.anpr import ANPREngine
        engine = ANPREngine(use_gpu=False)
        mock_reader_cls.assert_called_once_with(['en'], gpu=False)

    @patch("core.anpr.easyocr.Reader")
    def test_extract_plate_valid(self, mock_reader_cls):
        from core.anpr import ANPREngine
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0,0],[100,0],[100,30],[0,30]], "MH01AB1234", 0.92),
        ]
        mock_reader_cls.return_value = mock_reader

        engine = ANPREngine(use_gpu=False)
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        text, conf = engine.extract_plate(crop)
        assert text == "MH01AB1234"
        assert conf == pytest.approx(0.92)

    @patch("core.anpr.easyocr.Reader")
    def test_extract_plate_filters_short(self, mock_reader_cls):
        from core.anpr import ANPREngine
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0,0],[50,0],[50,20],[0,20]], "AB", 0.95),
        ]
        mock_reader_cls.return_value = mock_reader

        engine = ANPREngine(use_gpu=False)
        text, conf = engine.extract_plate(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == ""

    @patch("core.anpr.easyocr.Reader")
    def test_extract_plate_none_crop(self, mock_reader_cls):
        from core.anpr import ANPREngine
        mock_reader_cls.return_value = MagicMock()
        engine = ANPREngine(use_gpu=False)
        text, conf = engine.extract_plate(None)
        assert text == ""
        assert conf == 0.0

    @patch("core.anpr.easyocr.Reader")
    def test_extract_plate_empty_crop(self, mock_reader_cls):
        from core.anpr import ANPREngine
        mock_reader_cls.return_value = MagicMock()
        engine = ANPREngine(use_gpu=False)
        text, conf = engine.extract_plate(np.array([]))
        assert text == ""

    @patch("core.anpr.easyocr.Reader")
    def test_extract_plate_no_results(self, mock_reader_cls):
        from core.anpr import ANPREngine
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_reader_cls.return_value = mock_reader

        engine = ANPREngine(use_gpu=False)
        text, conf = engine.extract_plate(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == ""
        assert conf == 0.0

    @patch("core.anpr.easyocr.Reader")
    def test_picks_best_confidence(self, mock_reader_cls):
        from core.anpr import ANPREngine
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0,0],[1,0],[1,1],[0,1]], "KA09EE1234", 0.70),
            ([[0,0],[1,0],[1,1],[0,1]], "KA09EE5678", 0.95),
        ]
        mock_reader_cls.return_value = mock_reader

        engine = ANPREngine(use_gpu=False)
        text, conf = engine.extract_plate(np.zeros((100, 200, 3), dtype=np.uint8))
        assert text == "KA09EE5678"
        assert conf == pytest.approx(0.95)


# ═══════════════════════════════════════════════════════════
#  ANPR v3 — localisation, correction, voting, watchlist
# ═══════════════════════════════════════════════════════════
import cv2
from core.anpr import (
    ANPREngine, correct_indian_plate, is_valid_indian_plate, normalize_plate, plate_distance,
)


class FakeReader:
    """Returns a scripted OCR result (or a sequence of them, one per call)."""
    def __init__(self, text="MH01AB1234", conf=0.8, sequence=None):
        self.text, self.conf, self.sequence, self.calls = text, conf, list(sequence or []), 0

    def readtext(self, image, **kwargs):
        self.calls += 1
        if self.sequence:
            t = self.sequence.pop(0)
            return [] if t is None else [([[0, 0], [10, 0], [10, 5], [0, 5]], t, self.conf)]
        return [([[0, 0], [10, 0], [10, 5], [0, 5]], self.text, self.conf)]


def engine(reader=None, **kw):
    kw.setdefault("watchlist_path", None)
    return ANPREngine(reader=reader or FakeReader(), **kw)


def synthetic_vehicle(w=420, h=300, plate=(130, 215, 160, 40), angle=0, color=(240, 240, 240)):
    img = np.full((h, w, 3), (90, 90, 140), np.uint8)
    cv2.rectangle(img, (int(w * .07), int(h * .13)), (int(w * .93), int(h * .55)), (60, 60, 100), -1)
    px, py, pw, ph = plate
    cv2.rectangle(img, (px, py), (px + pw, py + ph), color, -1)
    cv2.putText(img, "MH01AB1234", (px + 5, py + int(ph * .75)), cv2.FONT_HERSHEY_SIMPLEX, pw / 260, (10, 10, 10), 2)
    if angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0, min(ax2, bx2) - max(a[0], b[0])) * max(0, min(ay2, by2) - max(a[1], b[1]))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


class TestPlateText:
    @pytest.mark.parametrize("raw,expected", [
        ("MHO1AB1234", "MH01AB1234"),   # O in district → 0
        ("MH01A81234", "MH01AB1234"),   # 8 in series → B
        ("KAO9EEI234", "KA09EE1234"),   # I in number → 1
        ("2Z8H1234AB", "22BH1234AB"),   # BH series
        ("DL3CAB1234", "DL3CAB1234"),   # single-digit district untouched
        ("mh-01 ab 1234", "MH01AB1234"),
    ])
    def test_indian_correction(self, raw, expected):
        assert correct_indian_plate(raw) == expected
        assert is_valid_indian_plate(expected)

    def test_correction_leaves_unfittable_text(self):
        assert correct_indian_plate("HELLO") == "HELLO"

    def test_normalize(self):
        assert normalize_plate(" ab-12 cd ") == "AB12CD"

    def test_confusion_weighted_distance(self):
        assert plate_distance("DL01AB1234", "DL01AB1234") == 0
        assert plate_distance("DL01AB1234", "DLO1AB1234") == pytest.approx(0.3)   # 0↔O
        assert plate_distance("DL01AB1234", "DL01A81234") == pytest.approx(0.3)   # B↔8
        assert plate_distance("DL01AB1234", "DL01AB1235") == pytest.approx(1.0)   # real error
        assert plate_distance("DL01AB1234", "MH12XY9999") > 5


class TestLocalisation:
    @pytest.mark.parametrize("kw", [
        {}, {"w": 640, "h": 420, "plate": (240, 320, 180, 42)}, {"angle": 6},
        {"color": (40, 200, 230)},  # yellow commercial plate
    ])
    def test_finds_plate(self, kw):
        img = synthetic_vehicle(**kw)
        plate = kw.get("plate", (130, 215, 160, 40))
        regions = engine().localize_plates(img)
        assert regions, "no plate region found"
        assert max(iou_xywh(plate, r) for r in regions) > 0.5

    def test_aspect_filter(self):
        regions = engine().localize_plates(synthetic_vehicle())
        for (_, _, w, h) in regions:
            assert 2.0 <= w / h <= 5.5  # padded box stays plate-shaped

    def test_tiny_crop(self):
        assert engine().localize_plates(np.zeros((10, 10, 3), np.uint8)) == []

    def test_preprocess_upscales_and_grays(self):
        out = engine().preprocess(np.full((20, 80, 3), 128, np.uint8))
        assert out.ndim == 2 and out.shape[0] == 64


class TestMultiPassOCR:
    def test_bumper_pass_first_and_early_exit(self):
        reader = FakeReader("MH01AB1234", 0.9)
        r = engine(reader).read_plate(synthetic_vehicle())
        assert r.text == "MH01AB1234" and r.method == "bumper" and r.valid_format
        assert reader.calls == 1

    def test_falls_through_to_localized(self):
        reader = FakeReader(sequence=[None, "MH01AB1234"])
        r = engine(reader).read_plate(synthetic_vehicle())
        assert r.method == "localized" and r.text == "MH01AB1234"

    def test_full_crop_fallback(self):
        e = engine(FakeReader(sequence=[None, "MH01AB1234"]))
        blank = np.full((120, 200, 3), 100, np.uint8)
        assert e.localize_plates(blank) == []          # no plate-like region → bumper, then full crop
        r = e.read_plate(blank)
        assert r.method == "full_crop" and r.text == "MH01AB1234"

    def test_correction_applied_to_ocr(self):
        r = engine(FakeReader("MHO1A81234", 0.7)).read_plate(synthetic_vehicle())
        assert r.text == "MH01AB1234" and r.raw_text == "MHO1A81234"

    def test_two_row_plate_merged(self):
        e = engine()
        segs = [([[0, 0], [80, 0], [80, 20], [0, 20]], "MH01", 0.8),
                ([[0, 25], [100, 25], [100, 45], [0, 45]], "AB1234", 0.7)]
        r = e._select_text(segs)
        assert r.text == "MH01AB1234" and r.valid_format

    def test_require_valid_format(self):
        e = engine(FakeReader("ZZZZZZ", 0.9), require_valid_format=True)
        assert e.read_plate(synthetic_vehicle()).text == ""

    def test_reader_missing_returns_empty(self):
        e = engine()
        e.reader = None
        assert e.extract_plate(synthetic_vehicle()) == ("", 0.0)


class TestTemporalVoting:
    def test_needs_two_concordant_reads(self):
        e = engine(FakeReader("MH01AB1234", 0.8))
        car = synthetic_vehicle()
        assert e.extract_plate_with_voting(car, "cam1:1", now=0.0) is None
        c = e.extract_plate_with_voting(car, "cam1:1", now=1.0, camera_id="cam1", track_id=1, vehicle_class="car")
        assert c is not None and c.text == "MH01AB1234" and c.read_count == 2
        assert c.vehicle_class == "car" and e.recent_plates[-1] is c
        # emitted once only
        assert e.extract_plate_with_voting(car, "cam1:1", now=2.0) is None
        assert e.get_confirmed("cam1:1").text == "MH01AB1234"

    def test_one_off_hallucination_is_rejected(self):
        e = engine(FakeReader(sequence=["MH01AB1234", "KA05XY9876", "GJ01ZZ0001"] * 2), early_accept_confidence=2)
        car = synthetic_vehicle()
        results = [e.extract_plate_with_voting(car, "t", now=float(i)) for i in range(3)]
        assert all(r is None for r in results)

    def test_rate_limit(self):
        reader = FakeReader("MH01AB1234", 0.9)
        e = engine(reader, ocr_interval_sec=1.0)
        car = synthetic_vehicle()
        e.extract_plate_with_voting(car, "t", now=0.0)
        e.extract_plate_with_voting(car, "t", now=0.5)  # throttled
        assert reader.calls == 1

    def test_small_vehicle_skipped(self):
        reader = FakeReader()
        e = engine(reader, min_vehicle_width_px=100)
        assert e.extract_plate_with_voting(np.zeros((40, 50, 3), np.uint8), "t", now=0) is None
        assert reader.calls == 0

    def test_clear_track(self):
        e = engine(FakeReader("MH01AB1234", 0.9))
        car = synthetic_vehicle()
        e.extract_plate_with_voting(car, "t", now=0)
        e.extract_plate_with_voting(car, "t", now=1)
        e.clear_track("t")
        assert e.get_confirmed("t") is None


class TestWatchlist:
    def test_exact_and_fuzzy_hits(self):
        e = engine(watchlist={"DL01AB1234": {"tag": "BLACKLISTED", "notes": "stolen"}})
        exact = e.check_watchlist_detailed("DL01AB1234")
        assert exact.exact and exact.tag == "BLACKLISTED"
        fuzzy = e.check_watchlist_detailed("DLO1A81234")   # two OCR confusions
        assert fuzzy is not None and not fuzzy.exact and fuzzy.distance == pytest.approx(0.6)
        assert e.check_watchlist("MH12XY9999") == ""

    def test_one_real_error_tolerated_two_not(self):
        e = engine(watchlist={"DL01AB1234": "BLACKLISTED"})
        assert e.check_watchlist("DL01AB1235") == "BLACKLISTED"
        assert e.check_watchlist("DL01AB1299") == ""

    def test_short_plates_need_exact(self):
        e = engine(watchlist={"AB123": "VIP"})
        assert e.check_watchlist("AB124") == ""
        assert e.check_watchlist("AB123") == "VIP"

    def test_confirmed_plate_carries_hit(self):
        e = engine(FakeReader("DLO1AB1234", 0.9), watchlist={"DL01AB1234": "STOLEN"})
        car = synthetic_vehicle()
        e.extract_plate_with_voting(car, "t", now=0)
        c = e.extract_plate_with_voting(car, "t", now=1)
        assert c.watchlist_tag == "STOLEN" and c.watchlist_hit.exact   # corrected before matching

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "wl.json")
        e = engine(watchlist_path=path)
        e.update_watchlist({"hr-26 dk 1234": {"tag": "wanted", "notes": "n"}})
        assert e.remove_from_watchlist("UP32GH5678")
        e2 = engine(watchlist_path=path)
        plates = {w["plate"]: w for w in e2.list_watchlist()}
        assert plates["HR26DK1234"]["tag"] == "WANTED"
        assert "UP32GH5678" not in plates

    def test_recognize_image_api(self):
        e = engine(FakeReader("RJ14CD9999", 0.9))
        out = e.recognize_image(synthetic_vehicle())
        assert out["plate"] == "RJ14CD9999" and out["watchlist"]["tag"] == "BLACKLISTED"
