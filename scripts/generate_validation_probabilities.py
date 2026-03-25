"""Generate validation-set probabilities and logits for temperature scaling & stacking.

Only generates for k=10 (our backtest horizon).
- DL models: load saved model, run inference on val_x, save logits + probabilities
- Baselines: retrain on train_x only (not train+val), predict on val_x

Why train-only for baselines?
    Our test-set baseline probabilities come from models trained on train+val.
    But for stacking/temperature-scaling, we need quasi-out-of-sample validation
    predictions. DL models used val for early stopping (quasi-OOS). Baselines
    trained on train+val would have seen val data → in-sample. So we retrain
    baselines on train-only for valid validation predictions.

Usage:
    python scripts/generate_validation_probabilities.py
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

HORIZON = 10
HORIZON_IDX = 0  # k=10 is index 0 in the label tensor
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def generate_dl_validation_probs():
    """Run DL model inference on validation data, save logits + probabilities."""
    print("\n  Loading validation data...")
    val_x = torch.load(PROJECT_ROOT / "data/processed/val_x.pt", weights_only=True)
    val_y = torch.load(PROJECT_ROOT / "data/processed/val_y.pt", weights_only=True)
    dataset = FI2010Dataset(val_x, val_y, horizon=HORIZON)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    print(f"  Val samples: {len(dataset)}")

    dl_models = {
        "deeplob": PROJECT_ROOT / f"experiments/k{HORIZON}",
        "deeplob_attention": PROJECT_ROOT / f"experiments/extension/k{HORIZON}",
        "cnn_only": PROJECT_ROOT / f"experiments/ablation/cnn_only/k{HORIZON}",
        "cnn_attention": PROJECT_ROOT / f"experiments/ablation/cnn_attention/k{HORIZON}",
    }

    for model_name, exp_dir in dl_models.items():
        model_path = exp_dir / "best_model.pt"
        if not model_path.exists():
            print(f"  WARNING: {model_path} not found, skipping")
            continue

        start = time.time()
        model = build_model(model_name)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()

        all_logits = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(DEVICE)
                logits = model(x_batch).cpu()
                all_logits.append(logits)

        logits_all = torch.cat(all_logits)  # (N_val, 3)
        probs_all = F.softmax(logits_all, dim=1)
        elapsed = time.time() - start

        save_path = exp_dir / "val_probabilities.pt"
        torch.save({"logits": logits_all, "probabilities": probs_all}, save_path)
        print(f"  {model_name}: {probs_all.shape}, time={elapsed:.1f}s → {save_path}")


def generate_baseline_validation_probs():
    """Retrain baselines on train-only data, predict on validation."""
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    print("\n  Loading data...")
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_x = torch.load(data_dir / "train_x.pt", weights_only=True)
    train_y = torch.load(data_dir / "train_y.pt", weights_only=True)
    val_x = torch.load(data_dir / "val_x.pt", weights_only=True)

    train_x_flat = train_x.reshape(train_x.shape[0], -1).numpy()
    val_x_flat = val_x.reshape(val_x.shape[0], -1).numpy()
    y_train = train_y[:, HORIZON_IDX].numpy()

    baselines_dir = PROJECT_ROOT / "experiments" / "baselines"

    # Logistic Regression (train-only)
    print("\n  LogReg (train-only)...")
    start = time.time()
    lr_model = LogisticRegression(
        max_iter=200, solver="saga", random_state=42, n_jobs=-1, tol=1e-3
    )
    lr_model.fit(train_x_flat, y_train)
    lr_probs = lr_model.predict_proba(val_x_flat)
    elapsed = time.time() - start
    np.save(baselines_dir / "logistic_regression_k10_val_probabilities.npy", lr_probs)
    print(f"    saved {lr_probs.shape}, time={elapsed:.1f}s")

    # XGBoost (train-only)
    print("\n  XGBoost (train-only)...")
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=3, tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(train_x_flat, y_train)
    xgb_probs = xgb_model.predict_proba(val_x_flat)
    elapsed = time.time() - start
    np.save(baselines_dir / "xgboost_k10_val_probabilities.npy", xgb_probs)
    print(f"    saved {xgb_probs.shape}, time={elapsed:.1f}s")


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print(f"Generating VALIDATION probabilities for k={HORIZON}")
    print("=" * 60)

    print("\n--- Deep Learning Models ---")
    generate_dl_validation_probs()

    print("\n--- Baseline Models (train-only retraining) ---")
    generate_baseline_validation_probs()

    print("\n" + "=" * 60)
    print("Done! Validation probability files saved.")
    print("  DL models:  experiments/**/val_probabilities.pt")
    print("  Baselines:  experiments/baselines/*_val_probabilities.npy")


if __name__ == "__main__":
    main()
