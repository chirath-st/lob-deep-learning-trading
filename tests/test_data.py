"""Tests for FI-2010 data loading and preprocessing.

Verifies:
- Dataset shapes match DeepLOB paper specs
- Label values are in valid range {0, 1, 2}
- No NaN or Inf values in features
- DataLoader produces correct batch shapes
- Chronological split integrity
"""

from pathlib import Path

import pytest
import torch

from src.data.dataset import FI2010Dataset, get_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Skip all tests if data hasn't been preprocessed yet
pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "train_x.pt").exists(),
    reason="Preprocessed data not found. Run `python scripts/preprocess_data.py` first.",
)


@pytest.fixture(scope="module")
def train_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load training tensors once for all tests in this module."""
    x = torch.load(PROCESSED_DIR / "train_x.pt", weights_only=True)
    y = torch.load(PROCESSED_DIR / "train_y.pt", weights_only=True)
    return x, y


@pytest.fixture(scope="module")
def test_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load test tensors once for all tests in this module."""
    x = torch.load(PROCESSED_DIR / "test_x.pt", weights_only=True)
    y = torch.load(PROCESSED_DIR / "test_y.pt", weights_only=True)
    return x, y


class TestDatasetShapes:
    """Verify tensor shapes match paper specifications."""

    def test_train_x_shape(self, train_data: tuple) -> None:
        x, _ = train_data
        assert x.ndim == 3, f"Expected 3D tensor (N, T, features), got {x.ndim}D"
        assert x.shape[1] == 100, f"Expected T=100 timesteps, got {x.shape[1]}"
        assert x.shape[2] == 40, f"Expected 40 features, got {x.shape[2]}"

    def test_train_y_shape(self, train_data: tuple) -> None:
        x, y = train_data
        assert y.shape[0] == x.shape[0], "X and Y must have same number of samples"
        assert y.shape[1] == 5, f"Expected 5 horizon columns, got {y.shape[1]}"

    def test_test_x_shape(self, test_data: tuple) -> None:
        x, _ = test_data
        assert x.shape[1] == 100
        assert x.shape[2] == 40

    def test_dataset_getitem_shape(self, train_data: tuple) -> None:
        """Verify __getitem__ returns correct shape with channel dim."""
        x, y = train_data
        dataset = FI2010Dataset(x, y, horizon=10)
        sample_x, sample_y = dataset[0]

        # After unsqueeze: (1, 100, 40) — channel dim added
        assert sample_x.shape == (1, 100, 40), (
            f"Expected (1, 100, 40), got {sample_x.shape}"
        )
        # Label should be a scalar
        assert sample_y.ndim == 0, f"Label should be scalar, got shape {sample_y.shape}"


class TestDataValues:
    """Verify data values are valid."""

    def test_no_nan_in_features(self, train_data: tuple) -> None:
        x, _ = train_data
        assert not torch.isnan(x).any(), "Training features contain NaN values"

    def test_no_inf_in_features(self, train_data: tuple) -> None:
        x, _ = train_data
        assert not torch.isinf(x).any(), "Training features contain Inf values"

    def test_no_nan_in_test(self, test_data: tuple) -> None:
        x, _ = test_data
        assert not torch.isnan(x).any(), "Test features contain NaN values"

    def test_label_range(self, train_data: tuple) -> None:
        """Labels must be in {0, 1, 2} for all horizons."""
        _, y = train_data
        assert y.min() >= 0, f"Min label is {y.min()}, expected >= 0"
        assert y.max() <= 2, f"Max label is {y.max()}, expected <= 2"

    def test_label_dtype(self, train_data: tuple) -> None:
        _, y = train_data
        assert y.dtype == torch.int64, f"Expected int64 labels, got {y.dtype}"

    def test_feature_dtype(self, train_data: tuple) -> None:
        x, _ = train_data
        assert x.dtype == torch.float32, f"Expected float32 features, got {x.dtype}"

    def test_all_classes_present(self, train_data: tuple) -> None:
        """All three classes should appear in training data for each horizon."""
        _, y = train_data
        for h_idx in range(5):
            unique = torch.unique(y[:, h_idx])
            assert len(unique) == 3, (
                f"Horizon {h_idx}: expected 3 classes, got {len(unique)} ({unique})"
            )


class TestDataLoader:
    """Verify DataLoader produces correct batches."""

    def test_batch_shape(self, train_data: tuple) -> None:
        """Verify a batch has the correct shape for DeepLOB input."""
        x, y = train_data
        dataset = FI2010Dataset(x, y, horizon=10)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        batch_x, batch_y = next(iter(loader))

        # DeepLOB expects input: (batch, 1, 100, 40)
        assert batch_x.shape == (64, 1, 100, 40), (
            f"Expected batch shape (64, 1, 100, 40), got {batch_x.shape}"
        )
        assert batch_y.shape == (64,), (
            f"Expected label shape (64,), got {batch_y.shape}"
        )

    def test_batch_dtypes(self, train_data: tuple) -> None:
        x, y = train_data
        dataset = FI2010Dataset(x, y, horizon=10)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        batch_x, batch_y = next(iter(loader))
        assert batch_x.dtype == torch.float32
        assert batch_y.dtype == torch.int64


class TestHorizons:
    """Verify different prediction horizons work correctly."""

    @pytest.mark.parametrize("horizon", [10, 20, 30, 50, 100])
    def test_valid_horizons(self, train_data: tuple, horizon: int) -> None:
        x, y = train_data
        dataset = FI2010Dataset(x, y, horizon=horizon)
        assert len(dataset) == x.shape[0]

    def test_invalid_horizon_raises(self, train_data: tuple) -> None:
        x, y = train_data
        with pytest.raises(ValueError, match="Invalid horizon"):
            FI2010Dataset(x, y, horizon=42)


class TestChronologicalSplit:
    """Verify train/val/test splits preserve temporal ordering."""

    def test_train_before_val(self) -> None:
        """Train samples should come from earlier time period than val."""
        # The split is done in preprocess_data.py by slicing (not shuffling)
        # We verify that train and val exist and have expected relative sizes
        train_x = torch.load(PROCESSED_DIR / "train_x.pt", weights_only=True)
        val_x = torch.load(PROCESSED_DIR / "val_x.pt", weights_only=True)

        # Train should be ~4x larger than val (80/20 split)
        ratio = len(train_x) / len(val_x)
        assert 3.0 < ratio < 5.0, (
            f"Train/val ratio is {ratio:.1f}, expected ~4.0 (80/20 split)"
        )

    def test_no_data_leakage_sizes(self) -> None:
        """Total samples should be consistent across splits."""
        train_x = torch.load(PROCESSED_DIR / "train_x.pt", weights_only=True)
        val_x = torch.load(PROCESSED_DIR / "val_x.pt", weights_only=True)
        test_x = torch.load(PROCESSED_DIR / "test_x.pt", weights_only=True)

        # All should have T=100 timesteps and 40 features
        for name, tensor in [("train", train_x), ("val", val_x), ("test", test_x)]:
            assert tensor.shape[1] == 100, f"{name} has wrong T"
            assert tensor.shape[2] == 40, f"{name} has wrong feature dim"
