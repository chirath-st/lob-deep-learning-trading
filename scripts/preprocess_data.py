"""Preprocess FI-2010 raw data into PyTorch-ready tensors.

This script loads the raw .txt files downloaded by download_data.py,
extracts the LOB features and labels, and saves them as .pt tensor files
for fast loading during training.

📚 Study this on Desktop: What is a tensor? Think of it as a
   multi-dimensional array. A 2D tensor = matrix, 3D tensor = "cube" of numbers.
   PyTorch uses tensors as its core data structure, similar to NumPy arrays
   but with GPU support and automatic differentiation.

Raw data format (from FI-2010 dataset):
    Shape: (149, N) where N = number of timestamps
    - Rows 0-39:   LOB features (10 levels × 4 features each)
    - Rows 40-143: Hand-crafted features (NOT used by DeepLOB)
    - Rows 144-148: Labels for 5 prediction horizons (k=10, 20, 30, 50, 100)

Output format:
    Saved to data/processed/ as PyTorch tensors:
    - train_x.pt: (num_samples, 100, 40) float32 — LOB feature windows
    - train_y.pt: (num_samples, 5) int64 — labels for all 5 horizons
    - test_x.pt, test_y.pt: same format for test set

    The horizon is selected at dataset loading time, not here.
    This way we preprocess once and can train on any horizon.
"""

import sys
from pathlib import Path

import numpy as np
import torch


# Map horizon names to column indices in the label rows
# The raw labels have 5 rows: k=10, 20, 30, 50, 100 (indices 0-4)
HORIZON_MAP = {10: 0, 20: 1, 30: 2, 50: 3, 100: 4}

# Map normalization names to filename components
NORM_MAP = {
    "DecPre": "DecPre",
    "ZScore": "ZScore",
    "MinMax": "MinMax",
}


def load_raw_data(
    raw_dir: Path, normalization: str = "DecPre"
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw FI-2010 text files for Setup 2.

    Setup 2 (anchored forward validation):
        - Train: single file containing days 1-7
        - Test: three separate files for days 8, 9, 10

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw .txt files.
    normalization : str
        Normalization type: 'DecPre', 'ZScore', or 'MinMax'.

    Returns
    -------
    train_data : np.ndarray
        Training data, shape (149, N_train).
    test_data : np.ndarray
        Test data, shape (149, N_test).
    """
    norm = NORM_MAP.get(normalization)
    if norm is None:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            f"Choose from: {list(NORM_MAP.keys())}"
        )

    # Training data: days 1-7 combined in one file
    train_file = raw_dir / f"Train_Dst_NoAuction_{norm}_CF_7.txt"
    if not train_file.exists():
        print(f"ERROR: Training data not found at {train_file}")
        print("Run `python scripts/download_data.py` first.")
        sys.exit(1)

    print(f"Loading training data: {train_file.name}...")
    train_data = np.loadtxt(str(train_file))
    print(f"  Shape: {train_data.shape}")
    # Shape should be (149, ~N) where 149 = 40 LOB + 104 features + 5 labels

    # Test data: days 8, 9, 10 in separate files
    test_files = [
        raw_dir / f"Test_Dst_NoAuction_{norm}_CF_{day}.txt"
        for day in [7, 8, 9]  # File indices 7,8,9 correspond to test days 8,9,10
    ]

    test_arrays = []
    for tf in test_files:
        if not tf.exists():
            print(f"ERROR: Test data not found at {tf}")
            sys.exit(1)
        print(f"Loading test data: {tf.name}...")
        arr = np.loadtxt(str(tf))
        print(f"  Shape: {arr.shape}")
        test_arrays.append(arr)

    # Concatenate test days along the time axis (axis=1)
    # 📚 Study this on Desktop: np.hstack joins arrays horizontally (along columns).
    #    Since our data is (features × timestamps), hstack concatenates timestamps.
    test_data = np.hstack(test_arrays)
    print(f"  Combined test shape: {test_data.shape}")

    return train_data, test_data


def extract_features_and_labels(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract LOB features and labels from raw FI-2010 data.

    Parameters
    ----------
    data : np.ndarray
        Raw data of shape (149, N).

    Returns
    -------
    features : np.ndarray
        LOB features, shape (N, 40). Transposed so each row = one timestep.
    labels : np.ndarray
        Labels for all 5 horizons, shape (N, 5). Values are 0, 1, 2.
    """
    # First 40 rows = LOB features (price/volume at 10 levels)
    # Transpose: (40, N) → (N, 40) so each row is one timestep
    features = data[:40, :].T  # shape: (N, 40)

    # Last 5 rows = labels for horizons k=10, 20, 30, 50, 100
    # Raw labels are encoded as 1, 2, 3 in the dataset
    # We convert to 0, 1, 2 to match standard PyTorch convention
    # 📚 Study this on Desktop: PyTorch's CrossEntropyLoss expects class indices
    #    starting from 0, not 1. That's why we subtract 1.
    labels = data[-5:, :].T  # shape: (N, 5)
    labels = labels.astype(int) - 1  # Convert 1,2,3 → 0,1,2

    return features, labels


def create_sliding_windows(
    features: np.ndarray, labels: np.ndarray, T: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of T consecutive LOB states.

    For each window, we take T consecutive timesteps as input (x) and
    the label at the LAST timestep as the target (y).

    📚 Study this on Desktop: Sliding windows — this is how we turn a long
       time series into individual training samples. Each sample is a "window"
       of T=100 consecutive snapshots. The model looks at these 100 snapshots
       to predict the next price movement direction.

    Parameters
    ----------
    features : np.ndarray
        Shape (N, 40) — all timesteps.
    labels : np.ndarray
        Shape (N, 5) — labels for all 5 horizons at each timestep.
    T : int
        Window size (lookback period). Default: 100.

    Returns
    -------
    X : np.ndarray
        Shape (N - T + 1, T, 40) — sliding windows of features.
    Y : np.ndarray
        Shape (N - T + 1, 5) — labels at the end of each window.

    Example
    -------
    If N=1000 and T=100:
        Window 0: features[0:100]   → label at timestep 99
        Window 1: features[1:101]   → label at timestep 100
        ...
        Window 900: features[900:1000] → label at timestep 999
        Total: 901 windows
    """
    N = features.shape[0]
    num_windows = N - T + 1

    # Pre-allocate array for all windows
    # 📚 Study this on Desktop: We pre-allocate instead of appending in a loop
    #    because NumPy array operations are much faster when the memory is
    #    allocated upfront (avoids repeated memory copying).
    X = np.zeros((num_windows, T, 40), dtype=np.float32)
    for i in range(num_windows):
        X[i] = features[i : i + T]

    # Labels at the end of each window (all 5 horizons)
    Y = labels[T - 1 : N]  # shape: (num_windows, 5)

    return X, Y


def preprocess_fi2010(
    normalization: str = "DecPre",
    lookback: int = 100,
    val_fraction: float = 0.2,
) -> None:
    """Full preprocessing pipeline for FI-2010 dataset.

    Parameters
    ----------
    normalization : str
        Which normalization to use ('DecPre', 'ZScore', 'MinMax').
    lookback : int
        Window size T for sliding windows.
    val_fraction : float
        Fraction of training data to use as validation (from the end,
        preserving chronological order).
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    print("=" * 60)
    print("Step 1: Loading raw FI-2010 data")
    print("=" * 60)
    train_data, test_data = load_raw_data(raw_dir, normalization)

    # Step 2: Extract features and labels
    print("\n" + "=" * 60)
    print("Step 2: Extracting features and labels")
    print("=" * 60)
    train_features, train_labels = extract_features_and_labels(train_data)
    test_features, test_labels = extract_features_and_labels(test_data)

    print(f"Train features: {train_features.shape}, labels: {train_labels.shape}")
    print(f"Test features:  {test_features.shape}, labels: {test_labels.shape}")

    # Verify label values
    unique_labels = np.unique(train_labels)
    print(f"Unique label values: {unique_labels}")
    assert set(unique_labels).issubset({0, 1, 2}), (
        f"Expected labels in {{0, 1, 2}}, got {unique_labels}"
    )

    # Step 3: Create sliding windows
    print("\n" + "=" * 60)
    print(f"Step 3: Creating sliding windows (T={lookback})")
    print("=" * 60)
    train_X, train_Y = create_sliding_windows(train_features, train_labels, lookback)
    test_X, test_Y = create_sliding_windows(test_features, test_labels, lookback)

    print(f"Train windows: X={train_X.shape}, Y={train_Y.shape}")
    print(f"Test windows:  X={test_X.shape}, Y={test_Y.shape}")

    # Step 4: Split training data into train and validation
    # 📚 Study this on Desktop: We split chronologically, NOT randomly!
    #    In time series, random splits cause "data leakage" — the model
    #    could see future data during training. Always split by time.
    print("\n" + "=" * 60)
    print(f"Step 4: Chronological train/val split ({1-val_fraction:.0%}/{val_fraction:.0%})")
    print("=" * 60)
    n_train = train_X.shape[0]
    split_idx = int(n_train * (1 - val_fraction))

    val_X = train_X[split_idx:]
    val_Y = train_Y[split_idx:]
    train_X = train_X[:split_idx]
    train_Y = train_Y[:split_idx]

    print(f"Train: X={train_X.shape}, Y={train_Y.shape}")
    print(f"Val:   X={val_X.shape}, Y={val_Y.shape}")
    print(f"Test:  X={test_X.shape}, Y={test_Y.shape}")

    # Step 5: Convert to PyTorch tensors and save
    print("\n" + "=" * 60)
    print("Step 5: Saving as PyTorch tensors")
    print("=" * 60)

    tensors = {
        "train_x": torch.from_numpy(train_X).float(),
        "train_y": torch.from_numpy(train_Y).long(),
        "val_x": torch.from_numpy(val_X).float(),
        "val_y": torch.from_numpy(val_Y).long(),
        "test_x": torch.from_numpy(test_X).float(),
        "test_y": torch.from_numpy(test_Y).long(),
    }

    for name, tensor in tensors.items():
        path = processed_dir / f"{name}.pt"
        torch.save(tensor, path)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Saved {name}.pt: {tensor.shape} ({size_mb:.1f} MB)")

    # Step 6: Print summary statistics
    print("\n" + "=" * 60)
    print("Step 6: Summary statistics")
    print("=" * 60)

    # Class distribution for each horizon (using horizon k=10 as primary example)
    for h_name, h_idx in HORIZON_MAP.items():
        print(f"\n  Horizon k={h_name}:")
        for split_name, labels_tensor in [
            ("train", tensors["train_y"]),
            ("val", tensors["val_y"]),
            ("test", tensors["test_y"]),
        ]:
            y = labels_tensor[:, h_idx].numpy()
            counts = np.bincount(y, minlength=3)
            pcts = counts / len(y) * 100
            print(
                f"    {split_name:5s}: down={counts[0]:6d} ({pcts[0]:5.1f}%), "
                f"stat={counts[1]:6d} ({pcts[1]:5.1f}%), "
                f"up={counts[2]:6d} ({pcts[2]:5.1f}%)"
            )

    # Feature value ranges
    print(f"\n  Feature value ranges (train):")
    train_tensor = tensors["train_x"]
    print(f"    Min: {train_tensor.min().item():.4f}")
    print(f"    Max: {train_tensor.max().item():.4f}")
    print(f"    Mean: {train_tensor.mean().item():.4f}")
    print(f"    Std: {train_tensor.std().item():.4f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(train_tensor).any().item()
    has_inf = torch.isinf(train_tensor).any().item()
    print(f"    NaN values: {has_nan}")
    print(f"    Inf values: {has_inf}")

    if has_nan or has_inf:
        print("  WARNING: Data contains NaN or Inf values!")
    else:
        print("\n  Data quality check passed!")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    preprocess_fi2010()
