"""Generate softmax probability vectors for all models.

Our training scripts only saved argmax class indices (0/1/2), not the full
probability distribution. For confidence-based trading strategies, we need
the probabilities — e.g., P(Up) = 0.92 is a much stronger signal than
P(Up) = 0.37 even though both predict "Up".

This script:
1. Loads each saved DL model, runs test inference, saves softmax probabilities
2. Retrains XGBoost and LogReg (fast, <1min each) to get predict_proba() outputs

Saves: experiments/**/test_probabilities.pt (or .npy for baselines)
    Shape: (139488, 3) — probability for [Down, Stationary, Up] per sample

Usage:
    python scripts/generate_probabilities.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FI2010Dataset
from src.models.deeplob import DeepLOB
from src.models.extension import DeepLOBAttention, DeepLOBCNNOnly, DeepLOBCNNAttention

HORIZONS = [10, 20, 30, 50, 100]
HORIZON_TO_IDX = {10: 0, 20: 1, 30: 2, 50: 3, 100: 4}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Deep learning models
# ---------------------------------------------------------------------------

def build_model(model_name: str) -> torch.nn.Module:
    """Instantiate model by name with correct config."""
    if model_name == "deeplob":
        return DeepLOB()
    elif model_name == "deeplob_attention":
        return DeepLOBAttention(
            d_model=192, n_heads=4, n_encoder_layers=2,
            dim_feedforward=256, dropout=0.1, pooling="mean",
        )
    elif model_name == "cnn_only":
        return DeepLOBCNNOnly()
    elif model_name == "cnn_attention":
        return DeepLOBCNNAttention(
            n_heads=4, n_encoder_layers=2,
            dim_feedforward=128, dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_dl_probabilities(model_name: str, exp_dir: Path, horizon: int) -> None:
    """Run inference for one DL model and save softmax probabilities."""
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        print(f"    WARNING: No model at {model_path}, skipping")
        return

    # Load model
    model = build_model(model_name)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Load test data
    test_x = torch.load(PROJECT_ROOT / "data/processed/test_x.pt", weights_only=True)
    test_y = torch.load(PROJECT_ROOT / "data/processed/test_y.pt", weights_only=True)
    dataset = FI2010Dataset(test_x, test_y, horizon=horizon)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # Run inference and collect softmax probabilities
    all_probs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1).cpu()
            all_probs.append(probs)

    probabilities = torch.cat(all_probs)  # (N, 3)

    # Verify argmax matches saved predictions
    history = torch.load(exp_dir / "history.pt", weights_only=False)
    saved_preds = history["test_predictions"]
    our_preds = probabilities.argmax(dim=1)
    match_rate = (our_preds == saved_preds).float().mean().item()

    # Save probabilities
    torch.save(probabilities, exp_dir / "test_probabilities.pt")
    print(f"    k={horizon}: saved {probabilities.shape}, pred match={match_rate:.4f}")


def generate_all_dl_probabilities():
    """Generate probabilities for all 4 DL models across all horizons."""
    dl_models = {
        "deeplob": "experiments/k{h}",
        "deeplob_attention": "experiments/extension/k{h}",
        "cnn_only": "experiments/ablation/cnn_only/k{h}",
        "cnn_attention": "experiments/ablation/cnn_attention/k{h}",
    }

    for model_name, path_template in dl_models.items():
        display_name = {
            "deeplob": "DeepLOB",
            "deeplob_attention": "DL-Attention",
            "cnn_only": "CNN-Only",
            "cnn_attention": "CNN+Attention",
        }[model_name]
        print(f"\n  {display_name}:")
        for h in HORIZONS:
            exp_dir = PROJECT_ROOT / path_template.format(h=h)
            generate_dl_probabilities(model_name, exp_dir, h)


# ---------------------------------------------------------------------------
# Baseline models (retrain to get predict_proba)
# ---------------------------------------------------------------------------

def generate_baseline_probabilities():
    """Retrain XGBoost and LogReg to get probability outputs."""
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    print("\n  Loading data for baselines...")
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_x = torch.load(data_dir / "train_x.pt", weights_only=True)
    train_y = torch.load(data_dir / "train_y.pt", weights_only=True)
    val_x = torch.load(data_dir / "val_x.pt", weights_only=True)
    val_y = torch.load(data_dir / "val_y.pt", weights_only=True)
    test_x = torch.load(data_dir / "test_x.pt", weights_only=True)
    test_y = torch.load(data_dir / "test_y.pt", weights_only=True)

    # Flatten for sklearn
    trainval_x = np.concatenate([
        train_x.reshape(train_x.shape[0], -1).numpy(),
        val_x.reshape(val_x.shape[0], -1).numpy(),
    ], axis=0)
    trainval_y_all = torch.cat([train_y, val_y], dim=0)
    test_x_flat = test_x.reshape(test_x.shape[0], -1).numpy()

    baselines_dir = PROJECT_ROOT / "experiments" / "baselines"

    # --- Logistic Regression ---
    print("\n  LogReg:")
    for h in HORIZONS:
        y_train = trainval_y_all[:, HORIZON_TO_IDX[h]].numpy()
        y_test = test_y[:, HORIZON_TO_IDX[h]].numpy()

        start = time.time()
        model = LogisticRegression(
            max_iter=200, solver="saga", random_state=42, n_jobs=-1, tol=1e-3
        )
        model.fit(trainval_x, y_train)
        elapsed = time.time() - start

        probs = model.predict_proba(test_x_flat)  # (N, 3)
        preds = probs.argmax(axis=1)

        # Verify matches saved predictions
        saved = torch.load(baselines_dir / f"logistic_regression_k{h}.pt", weights_only=False)
        match_rate = (preds == saved["predictions"]).mean()

        np.save(baselines_dir / f"logistic_regression_k{h}_probabilities.npy", probs)
        print(f"    k={h}: saved {probs.shape}, pred match={match_rate:.4f}, time={elapsed:.1f}s")

    # --- XGBoost ---
    print("\n  XGBoost:")
    for h in HORIZONS:
        y_train = trainval_y_all[:, HORIZON_TO_IDX[h]].numpy()
        y_test = test_y[:, HORIZON_TO_IDX[h]].numpy()

        start = time.time()
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective="multi:softprob",  # softprob instead of softmax for probabilities
            num_class=3, tree_method="hist", random_state=42,
            n_jobs=-1, verbosity=0,
        )
        model.fit(trainval_x, y_train)
        elapsed = time.time() - start

        probs = model.predict_proba(test_x_flat)  # (N, 3)
        preds = probs.argmax(axis=1)

        # Verify matches saved predictions
        saved = torch.load(baselines_dir / f"xgboost_k{h}.pt", weights_only=False)
        match_rate = (preds == saved["predictions"]).mean()

        np.save(baselines_dir / f"xgboost_k{h}_probabilities.npy", probs)
        print(f"    k={h}: saved {probs.shape}, pred match={match_rate:.4f}, time={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("Generating softmax probabilities for all models")
    print("=" * 60)

    print("\n--- Deep Learning Models ---")
    generate_all_dl_probabilities()

    print("\n--- Baseline Models (retraining) ---")
    generate_baseline_probabilities()

    print("\n" + "=" * 60)
    print("Done! All probability files saved.")
    print("  DL models:  experiments/**/test_probabilities.pt")
    print("  Baselines:  experiments/baselines/*_probabilities.npy")


if __name__ == "__main__":
    main()
