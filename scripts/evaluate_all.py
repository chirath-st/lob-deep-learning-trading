"""Evaluate all 5 DeepLOB horizons on the test set and generate plots.

Usage:
    python -u scripts/evaluate_all.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.deeplob import DeepLOB

HORIZONS = [10, 20, 30, 50, 100]
CLASS_NAMES = ["Down", "Stationary", "Up"]
DEVICE = torch.device("cpu")


def main():
    results = {}

    for h in HORIZONS:
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating k={h}", flush=True)
        print(f"{'='*60}", flush=True)

        # Load model
        config = yaml.safe_load(open(PROJECT_ROOT / f"experiments/k{h}/config.yaml"))
        mc = config["model"]
        model = DeepLOB(
            mc["num_classes"], mc["conv_filters"], mc["inception_filters"],
            mc["lstm_hidden"], mc["lstm_layers"], mc["leaky_relu_slope"],
        )
        ckpt = torch.load(
            PROJECT_ROOT / f"experiments/k{h}/best_model.pt",
            map_location=DEVICE, weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Load test data
        _, _, test_loader = get_dataloaders(
            "data/processed", horizon=h, batch_size=512, project_root=PROJECT_ROOT,
        )

        # Inference
        all_preds, all_labels = [], []
        n_batches = len(test_loader)
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                logits = model(x)
                all_preds.append(logits.argmax(dim=1).numpy())
                all_labels.append(y.numpy())
                if (i + 1) % 50 == 0:
                    print(f"  batch {i+1}/{n_batches}", flush=True)

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        results[h] = {"preds": preds, "labels": labels}

        acc = accuracy_score(labels, preds)
        print(f"\n  Test accuracy: {acc*100:.2f}%", flush=True)
        print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4), flush=True)

        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print("  Confusion matrix (row-normalized):", flush=True)
        for i, name in enumerate(CLASS_NAMES):
            row = "  ".join(f"{cm_norm[i,j]:.3f}" for j in range(3))
            print(f"    {name:12s}: {row}", flush=True)

    # Generate plots
    print("\nGenerating plots...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, h in zip(axes, HORIZONS):
        cm = confusion_matrix(results[h]["labels"], results[h]["preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, vmin=0, vmax=1, cbar=False,
        )
        acc = accuracy_score(results[h]["labels"], results[h]["preds"])
        ax.set_title(f"k={h} (acc={acc*100:.1f}%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label" if h == 10 else "")
        ax.set_xlabel("Predicted Label")
    fig.suptitle(
        "Confusion Matrices — Normalized by True Label (Recall)",
        fontsize=15, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "experiments/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved experiments/confusion_matrices.png", flush=True)

    # F1 by horizon
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Down": "#e74c3c", "Stationary": "#95a5a6", "Up": "#2ecc71"}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        f1s = [
            f1_score(results[h]["labels"], results[h]["preds"], average=None)[cls_idx] * 100
            for h in HORIZONS
        ]
        ax.plot(HORIZONS, f1s, marker="o", linewidth=2, label=cls_name,
                color=colors[cls_name], markersize=8)
    weighted_f1s = [
        f1_score(results[h]["labels"], results[h]["preds"], average="weighted") * 100
        for h in HORIZONS
    ]
    ax.plot(HORIZONS, weighted_f1s, marker="s", linewidth=2, linestyle="--",
            label="Weighted F1", color="black", markersize=8)
    ax.set_xlabel("Prediction Horizon (k)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Per-Class F1 Score by Prediction Horizon", fontsize=14, fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "experiments/f1_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved experiments/f1_by_horizon.png", flush=True)

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
