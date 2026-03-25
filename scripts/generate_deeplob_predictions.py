"""Generate per-sample test predictions for the original DeepLOB model.

The original train.py only saved aggregated metrics (loss, accuracy) but not
per-sample predictions. The extension and baseline training scripts added this.
This script loads each saved DeepLOB checkpoint and runs test inference to
produce predictions in the same format as the other models.

Saves: test_predictions (torch.Tensor) and test_labels (torch.Tensor)
into each experiment's history.pt file.

Usage:
    python scripts/generate_deeplob_predictions.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FI2010Dataset
from src.models.deeplob import DeepLOB
from src.training.trainer import load_checkpoint

HORIZONS = [10, 20, 30, 50, 100]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def generate_predictions(horizon: int) -> None:
    """Load DeepLOB model for given horizon and generate test predictions."""
    exp_dir = PROJECT_ROOT / "experiments" / f"k{horizon}"
    model_path = exp_dir / "best_model.pt"
    history_path = exp_dir / "history.pt"

    if not model_path.exists():
        print(f"  WARNING: No model found at {model_path}, skipping")
        return

    # Load model (trained on CUDA, loading on MPS/CPU)
    model = DeepLOB()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Load test data
    test_x = torch.load(PROJECT_ROOT / "data/processed/test_x.pt", weights_only=True)
    test_y = torch.load(PROJECT_ROOT / "data/processed/test_y.pt", weights_only=True)
    dataset = FI2010Dataset(test_x, test_y, horizon=horizon)

    # Run inference in batches
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch)

    predictions = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # Verify accuracy matches saved value
    accuracy = (predictions == labels).float().mean().item()
    history = torch.load(history_path, weights_only=False)
    saved_acc = history.get("test_accuracy", None)

    print(f"  k={horizon}: accuracy={accuracy:.4f} (saved={saved_acc:.4f})", end="")
    if saved_acc and abs(accuracy - saved_acc) > 0.001:
        print(" ⚠️  MISMATCH")
    else:
        print(" ✓")

    # Update history with predictions
    history["test_predictions"] = predictions
    history["test_labels"] = labels
    torch.save(history, history_path)


def main():
    print(f"Device: {DEVICE}")
    print(f"Generating DeepLOB test predictions for all horizons...\n")

    for horizon in HORIZONS:
        generate_predictions(horizon)

    print("\nDone! All history.pt files updated with test_predictions and test_labels.")


if __name__ == "__main__":
    main()
