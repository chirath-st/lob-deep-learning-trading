"""FI-2010 Dataset class for PyTorch.

This module provides the FI2010Dataset class that loads preprocessed
LOB data and serves it to PyTorch DataLoaders for training.

📚 Study this on Desktop: PyTorch Dataset and DataLoader pattern — a Dataset
   defines HOW to access individual samples (via __getitem__), while a
   DataLoader handles batching, shuffling, and parallel loading.

Usage
-----
    from src.data.dataset import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data/processed",
        horizon=10,
        batch_size=64,
    )

    for x, y in train_loader:
        # x shape: (batch_size, 1, 100, 40) — 1-channel "image" of LOB states
        # y shape: (batch_size,) — class labels {0: down, 1: stationary, 2: up}
        pass
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


# Map prediction horizon k to column index in the label tensor
HORIZON_TO_IDX = {10: 0, 20: 1, 30: 2, 50: 3, 100: 4}


class FI2010Dataset(Dataset):
    """PyTorch Dataset for preprocessed FI-2010 limit order book data.

    Each sample is a window of T=100 consecutive LOB snapshots, shaped as
    a 1-channel "image" (1, 100, 40) for the CNN to process.

    📚 Study this on Desktop: Why reshape to (1, 100, 40)?
       Conv2d expects input as (batch, channels, height, width).
       We treat the LOB window as a single-channel image where:
       - Height = 100 timesteps (like rows of an image)
       - Width = 40 features (like columns of an image)
       The convolutions then learn spatial patterns across time and LOB levels.

    Parameters
    ----------
    x : torch.Tensor
        Feature tensor of shape (N, 100, 40).
    y : torch.Tensor
        Label tensor of shape (N, 5) — labels for all 5 horizons.
    horizon : int
        Prediction horizon k. One of {10, 20, 30, 50, 100}.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, horizon: int = 10) -> None:
        if horizon not in HORIZON_TO_IDX:
            raise ValueError(
                f"Invalid horizon {horizon}. Must be one of {list(HORIZON_TO_IDX.keys())}"
            )

        self.x = x  # shape: (N, 100, 40)
        # Select the label column for the chosen horizon
        self.y = y[:, HORIZON_TO_IDX[horizon]]  # shape: (N,)
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single (input, label) pair.

        Returns
        -------
        x : torch.Tensor
            Shape (1, 100, 40) — unsqueezed to add channel dimension.
            The "1" is the channel dim that Conv2d expects.
        y : torch.Tensor
            Scalar int64 tensor — class label {0: down, 1: stationary, 2: up}.
        """
        # Add channel dimension: (100, 40) → (1, 100, 40)
        # 📚 Study this on Desktop: unsqueeze(0) adds a dimension at position 0.
        #    This is like going from a 2D matrix to a single-channel image.
        return self.x[idx].unsqueeze(0), self.y[idx]


def get_dataloaders(
    data_dir: str = "data/processed",
    horizon: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    project_root: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir : str
        Path to processed data directory (relative to project root).
    horizon : int
        Prediction horizon k. One of {10, 20, 30, 50, 100}.
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of worker processes for data loading. 0 = main process only.
    project_root : Path, optional
        Project root directory. If None, auto-detected.

    Returns
    -------
    train_loader : DataLoader
        Training data loader. NOTE: shuffle=True for training.
    val_loader : DataLoader
        Validation data loader.
    test_loader : DataLoader
        Test data loader.

    Notes
    -----
    📚 Study this on Desktop: Why shuffle training but NOT val/test?
       Shuffling during training helps the model generalize (it doesn't
       memorize the order of samples). But for validation/test, we keep
       chronological order so results are reproducible and we can analyze
       performance over time.

       The official DeepLOB code also shuffles training data. While this
       technically breaks strict chronological order within the training
       set, the train/val/test SPLIT is still chronological (no future
       data leaks into training).
    """
    if project_root is None:
        # Assume we're called from the project root or find it
        project_root = Path(__file__).resolve().parent.parent.parent

    processed_dir = project_root / data_dir

    # Load preprocessed tensors
    print(f"Loading preprocessed data from {processed_dir}...")
    train_x = torch.load(processed_dir / "train_x.pt", weights_only=True)
    train_y = torch.load(processed_dir / "train_y.pt", weights_only=True)
    val_x = torch.load(processed_dir / "val_x.pt", weights_only=True)
    val_y = torch.load(processed_dir / "val_y.pt", weights_only=True)
    test_x = torch.load(processed_dir / "test_x.pt", weights_only=True)
    test_y = torch.load(processed_dir / "test_y.pt", weights_only=True)

    # Create datasets
    train_dataset = FI2010Dataset(train_x, train_y, horizon=horizon)
    val_dataset = FI2010Dataset(val_x, val_y, horizon=horizon)
    test_dataset = FI2010Dataset(test_x, test_y, horizon=horizon)

    print(f"  Horizon: k={horizon}")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Print class distribution for selected horizon
    for name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        counts = torch.bincount(ds.y, minlength=3)
        pcts = counts.float() / len(ds) * 100
        print(
            f"  {name:5s} classes: down={counts[0]:6d} ({pcts[0]:5.1f}%), "
            f"stat={counts[1]:6d} ({pcts[1]:5.1f}%), "
            f"up={counts[2]:6d} ({pcts[2]:5.1f}%)"
        )

    # Create DataLoaders
    # NOTE: Only shuffle training data. Val/test stay in chronological order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Official code also shuffles training
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
