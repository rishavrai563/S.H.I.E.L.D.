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
