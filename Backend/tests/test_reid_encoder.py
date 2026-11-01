"""
Tests for ReIDEncoder. Mocks torchreid to avoid model downloads.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def _fake_tensor(rows, cols):
    """Create a numpy array that mimics a torch.Tensor for test mocking."""
    arr = np.random.randn(rows, cols).astype(np.float32)
    mock = MagicMock()
    mock.cpu.return_value = mock
    mock.detach.return_value = mock
    mock.numpy.return_value = arr
    mock.shape = (rows, cols)
    mock.__len__ = lambda s: rows
    return mock


class TestReIDEncoder:
    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_init_with_existing_weights(self, mock_fe_cls, tmp_path):
        from core.reid_encoder import ReIDEncoder
        # Create a fake weights file
        weights = tmp_path / "model.pth"
        weights.write_bytes(b"fake")

        mock_fe_cls.return_value = MagicMock()
        encoder = ReIDEncoder(model_path=str(weights), device="cpu")
        call_kwargs = mock_fe_cls.call_args[1]
        assert call_kwargs["model_path"] == str(weights)

    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_init_missing_weights_auto_download(self, mock_fe_cls):
        from core.reid_encoder import ReIDEncoder
        mock_fe_cls.return_value = MagicMock()
        encoder = ReIDEncoder(model_path="nonexistent.pth", device="cpu")
        call_kwargs = mock_fe_cls.call_args[1]
        assert call_kwargs["model_path"] == ""  # triggers auto-download

    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_batch_extract_returns_normalized(self, mock_fe_cls):
        from core.reid_encoder import ReIDEncoder
        mock_extractor = MagicMock()
        raw = _fake_tensor(3, 512)
        mock_extractor.return_value = raw
        mock_fe_cls.return_value = mock_extractor

        encoder = ReIDEncoder(model_path="x.pth", device="cpu")
        crops = [np.zeros((128, 64, 3), dtype=np.uint8) for _ in range(3)]
        result = encoder.batch_extract(crops)

        assert result.shape == (3, 512)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_batch_extract_preserves_index(self, mock_fe_cls):
        """Invalid crops get zero vectors; valid crops keep correct indices."""
        from core.reid_encoder import ReIDEncoder
        mock_extractor = MagicMock()
        mock_extractor.return_value = _fake_tensor(1, 512)
        mock_fe_cls.return_value = mock_extractor

        encoder = ReIDEncoder(model_path="x.pth", device="cpu")
        crops = [
            None,
            np.zeros((128, 64, 3), dtype=np.uint8),
            np.array([]),
        ]
        result = encoder.batch_extract(crops)
        assert result.shape == (3, 512)
        assert np.allclose(result[0], 0.0)  # None → zero
        assert np.allclose(result[2], 0.0)  # empty → zero
        assert np.linalg.norm(result[1]) > 0  # valid crop

    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_batch_extract_empty_list(self, mock_fe_cls):
        from core.reid_encoder import ReIDEncoder
        mock_fe_cls.return_value = MagicMock()
        encoder = ReIDEncoder(model_path="x.pth", device="cpu")
        assert encoder.batch_extract([]) is None

    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_extract_single(self, mock_fe_cls):
        from core.reid_encoder import ReIDEncoder
        mock_extractor = MagicMock()
        mock_extractor.return_value = _fake_tensor(1, 512)
        mock_fe_cls.return_value = mock_extractor

        encoder = ReIDEncoder(model_path="x.pth", device="cpu")
        crop = np.zeros((256, 128, 3), dtype=np.uint8)
        result = encoder.extract(crop)
        assert result is not None
        assert result.shape == (512,)

    @patch("core.reid_encoder.torchreid.utils.FeatureExtractor")
    def test_extract_none_returns_none(self, mock_fe_cls):
        from core.reid_encoder import ReIDEncoder
        mock_fe_cls.return_value = MagicMock()
        encoder = ReIDEncoder(model_path="x.pth", device="cpu")
        assert encoder.extract(None) is None
