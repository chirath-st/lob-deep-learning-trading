"""Retrain MLP baseline with better hyperparameters.

Key fixes:
- StandardScaler on input (DecPre normalization has varying scales)
- Smaller architecture (128, 64) to reduce overfitting risk
- lr=0.01 with ReduceLROnPlateau to escape majority-class minimum
- Weight decay for regularization

Run: python scripts/retrain_mlp.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HORIZONS = [10, 20, 30, 50, 100]
HORIZON_TO_IDX = {10: 0, 20: 1, 30: 2, 50: 3, 100: 4}
CLASS_NAMES = ["Down", "Stationary", "Up"]
BASELINES_DIR = PROJECT_ROOT / "experiments" / "baselines"
SEED = 42


class MLPBaseline(nn.Module):
    def __init__(self, input_dim=4000, hidden1=128, hidden2=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Load data
print("Loading data...")
data_dir = PROJECT_ROOT / "data" / "processed"
train_x = torch.load(data_dir / "train_x.pt", weights_only=True)
train_y = torch.load(data_dir / "train_y.pt", weights_only=True)
val_x = torch.load(data_dir / "val_x.pt", weights_only=True)
val_y = torch.load(data_dir / "val_y.pt", weights_only=True)
test_x = torch.load(data_dir / "test_x.pt", weights_only=True)
test_y = torch.load(data_dir / "test_y.pt", weights_only=True)

train_x_flat = train_x.reshape(train_x.shape[0], -1).numpy()
val_x_flat = val_x.reshape(val_x.shape[0], -1).numpy()
test_x_flat = test_x.reshape(test_x.shape[0], -1).numpy()

# Standardize input
print("Standardizing input...")
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x_flat).astype(np.float32)
val_x_scaled = scaler.transform(val_x_flat).astype(np.float32)
test_x_scaled = scaler.transform(test_x_flat).astype(np.float32)

for h in HORIZONS:
    print(f"\n{'='*50}")
    print(f"MLP — k={h}")
    print(f"{'='*50}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train = torch.from_numpy(train_x_scaled)
    y_train = train_y[:, HORIZON_TO_IDX[h]].long()
    X_val = torch.from_numpy(val_x_scaled)
    y_val = val_y[:, HORIZON_TO_IDX[h]].long()
    X_test = torch.from_numpy(test_x_scaled)
    y_test_t = test_y[:, HORIZON_TO_IDX[h]].long()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

    model = MLPBaseline()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = 20

    start = time.time()
    for epoch in range(1, 151):
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

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}: val_acc={val_acc:.4f} (best={best_val_acc:.4f}) lr={lr_now:.6f}")

        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch} (best={best_val_acc:.4f})")
            break

    elapsed = time.time() - start

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1).numpy()

    y_test = y_test_t.numpy()
    acc = accuracy_score(y_test, test_preds)
    f1_w = f1_score(y_test, test_preds, average="weighted")
    print(f"\n  Test: acc={acc*100:.2f}%, F1={f1_w*100:.2f}%, time={elapsed:.1f}s")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))

    torch.save(
        {
            "model": "mlp",
            "horizon": h,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "predictions": test_preds,
            "labels": y_test,
            "train_time": elapsed,
        },
        BASELINES_DIR / f"mlp_k{h}.pt",
    )

print("\nDone! MLP results updated.")
