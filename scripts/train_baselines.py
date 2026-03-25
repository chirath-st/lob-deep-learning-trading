"""Train all baseline models (LogReg, MLP, XGBoost) across all 5 horizons.

Saves results to experiments/baselines/.
Run from project root: python scripts/train_baselines.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HORIZONS = [10, 20, 30, 50, 100]
HORIZON_TO_IDX = {10: 0, 20: 1, 30: 2, 50: 3, 100: 4}
CLASS_NAMES = ["Down", "Stationary", "Up"]
BASELINES_DIR = PROJECT_ROOT / "experiments" / "baselines"
BASELINES_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
data_dir = PROJECT_ROOT / "data" / "processed"
train_x = torch.load(data_dir / "train_x.pt", weights_only=True)
train_y = torch.load(data_dir / "train_y.pt", weights_only=True)
val_x = torch.load(data_dir / "val_x.pt", weights_only=True)
val_y = torch.load(data_dir / "val_y.pt", weights_only=True)
test_x = torch.load(data_dir / "test_x.pt", weights_only=True)
test_y = torch.load(data_dir / "test_y.pt", weights_only=True)

# Flatten
train_x_flat = train_x.reshape(train_x.shape[0], -1).numpy()
val_x_flat = val_x.reshape(val_x.shape[0], -1).numpy()
test_x_flat = test_x.reshape(test_x.shape[0], -1).numpy()

# Combined train+val for models without early stopping
trainval_x_flat = np.concatenate([train_x_flat, val_x_flat], axis=0)
trainval_y = torch.cat([train_y, val_y], dim=0)

print(f"Train: {train_x_flat.shape}, Val: {val_x_flat.shape}, Test: {test_x_flat.shape}")
print(f"Combined train+val: {trainval_x_flat.shape}")


def get_labels(y_tensor, horizon):
    return y_tensor[:, HORIZON_TO_IDX[horizon]].numpy()


def save_result(model_name, horizon, acc, f1_w, preds, labels, train_time):
    torch.save(
        {
            "model": model_name,
            "horizon": horizon,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "predictions": preds,
            "labels": labels,
            "train_time": train_time,
        },
        BASELINES_DIR / f"{model_name}_k{horizon}.pt",
    )


# ── 1. Logistic Regression ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

for h in HORIZONS:
    print(f"\n  k={h}...", end=" ", flush=True)
    y_train = get_labels(trainval_y, h)
    y_test = get_labels(test_y, h)

    start = time.time()
    model = LogisticRegression(
        max_iter=200, solver="saga", random_state=SEED, n_jobs=-1, tol=1e-3
    )
    model.fit(trainval_x_flat, y_train)
    elapsed = time.time() - start

    preds = model.predict(test_x_flat)
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted")
    print(f"acc={acc*100:.2f}%, F1={f1_w*100:.2f}%, time={elapsed:.1f}s")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))

    save_result("logistic_regression", h, acc, f1_w, preds, y_test, elapsed)


# ── 2. MLP ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MLP (2-layer)")
print("=" * 60)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim=4000, hidden1=256, hidden2=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


for h in HORIZONS:
    print(f"\n  k={h}...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train = torch.from_numpy(train_x_flat).float()
    y_train = train_y[:, HORIZON_TO_IDX[h]].long()
    X_val = torch.from_numpy(val_x_flat).float()
    y_val = val_y[:, HORIZON_TO_IDX[h]].long()
    X_test = torch.from_numpy(test_x_flat).float()
    y_test_t = test_y[:, HORIZON_TO_IDX[h]].long()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=512, shuffle=True
    )

    model = MLPBaseline()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = 10

    start = time.time()
    for epoch in range(1, 101):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - start

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1).numpy()

    y_test = y_test_t.numpy()
    acc = accuracy_score(y_test, test_preds)
    f1_w = f1_score(y_test, test_preds, average="weighted")
    print(f"    Test: acc={acc*100:.2f}%, F1={f1_w*100:.2f}%, time={elapsed:.1f}s")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))

    save_result("mlp", h, acc, f1_w, test_preds, y_test, elapsed)


# ── 3. XGBoost ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

for h in HORIZONS:
    print(f"\n  k={h}...", end=" ", flush=True)
    y_train = get_labels(trainval_y, h)
    y_test = get_labels(test_y, h)

    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=3,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(trainval_x_flat, y_train)
    elapsed = time.time() - start

    preds = model.predict(test_x_flat)
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted")
    print(f"acc={acc*100:.2f}%, F1={f1_w*100:.2f}%, time={elapsed:.1f}s")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))

    save_result("xgboost", h, acc, f1_w, preds, y_test, elapsed)


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Load DeepLOB results
deeplob_accs = {}
for h in HORIZONS:
    history = torch.load(
        PROJECT_ROOT / "experiments" / f"k{h}" / "history.pt",
        map_location="cpu",
        weights_only=False,
    )
    deeplob_accs[h] = history["test_accuracy"]

# Print summary table
print(f"\n{'Horizon':>8s} {'DeepLOB':>8s} {'LogReg':>8s} {'MLP':>8s} {'XGBoost':>8s}")
print("-" * 44)
for h in HORIZONS:
    lr = torch.load(BASELINES_DIR / f"logistic_regression_k{h}.pt", weights_only=False)
    mlp = torch.load(BASELINES_DIR / f"mlp_k{h}.pt", weights_only=False)
    xgb_r = torch.load(BASELINES_DIR / f"xgboost_k{h}.pt", weights_only=False)
    print(
        f"  k={h:>3d}  {deeplob_accs[h]*100:>7.2f}% {lr['accuracy']*100:>7.2f}% "
        f"{mlp['accuracy']*100:>7.2f}% {xgb_r['accuracy']*100:>7.2f}%"
    )

# Save summary JSON
summary = {}
for h in HORIZONS:
    lr = torch.load(BASELINES_DIR / f"logistic_regression_k{h}.pt", weights_only=False)
    mlp = torch.load(BASELINES_DIR / f"mlp_k{h}.pt", weights_only=False)
    xgb_r = torch.load(BASELINES_DIR / f"xgboost_k{h}.pt", weights_only=False)
    summary[f"k{h}"] = {
        "deeplob_acc": round(deeplob_accs[h] * 100, 2),
        "logistic_regression_acc": round(lr["accuracy"] * 100, 2),
        "logistic_regression_f1": round(lr["f1_weighted"] * 100, 2),
        "mlp_acc": round(mlp["accuracy"] * 100, 2),
        "mlp_f1": round(mlp["f1_weighted"] * 100, 2),
        "xgboost_acc": round(xgb_r["accuracy"] * 100, 2),
        "xgboost_f1": round(xgb_r["f1_weighted"] * 100, 2),
    }

with open(BASELINES_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll results saved to {BASELINES_DIR}/")
print("Done!")
