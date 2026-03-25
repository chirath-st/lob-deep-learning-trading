"""Train DeepLOB-Attention (or ablation variants) on FI-2010 dataset.

Extends scripts/train.py with:
- Model factory: builds model from config model.name field
- Extended test evaluation: collects predictions, computes weighted F1
- Learning rate warmup scheduler
- history.pt includes test_f1_weighted, test_predictions, test_labels

Usage:
    python scripts/train_extension.py                          # defaults
    python scripts/train_extension.py --horizon 20             # different horizon
    python scripts/train_extension.py --config path/to/cfg.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.training.trainer import (
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
    select_device,
    train,
    train_one_epoch,
    validate,
)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> nn.Module:
    """Build model from config. Supports multiple model architectures.

    📚 Study this on Desktop: Factory pattern — a function that creates
       different objects based on a parameter. Here, model.name in the
       config determines which architecture to instantiate.
    """
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name == "deeplob":
        from src.models.deeplob import DeepLOB

        return DeepLOB(
            num_classes=model_cfg["num_classes"],
            conv_filters=model_cfg["conv_filters"],
            inception_filters=model_cfg["inception_filters"],
            lstm_hidden=model_cfg.get("lstm_hidden", 64),
            lstm_layers=model_cfg.get("lstm_layers", 1),
            leaky_relu_slope=model_cfg["leaky_relu_slope"],
        )

    elif name == "deeplob_attention":
        from src.models.extension import DeepLOBAttention

        return DeepLOBAttention(
            num_classes=model_cfg["num_classes"],
            conv_filters=model_cfg["conv_filters"],
            inception_filters=model_cfg["inception_filters"],
            leaky_relu_slope=model_cfg["leaky_relu_slope"],
            d_model=model_cfg.get("d_model", 192),
            n_heads=model_cfg.get("n_heads", 4),
            n_encoder_layers=model_cfg.get("n_encoder_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            pooling=model_cfg.get("pooling", "mean"),
            max_seq_len=model_cfg.get("max_seq_len", 120),
        )

    elif name == "cnn_only":
        from src.models.extension import DeepLOBCNNOnly

        return DeepLOBCNNOnly(
            num_classes=model_cfg["num_classes"],
            conv_filters=model_cfg["conv_filters"],
            inception_filters=model_cfg.get("inception_filters", 64),
            leaky_relu_slope=model_cfg["leaky_relu_slope"],
        )

    elif name == "cnn_attention":
        from src.models.extension import DeepLOBCNNAttention

        return DeepLOBCNNAttention(
            num_classes=model_cfg["num_classes"],
            conv_filters=model_cfg["conv_filters"],
            inception_filters=model_cfg.get("inception_filters", 64),
            leaky_relu_slope=model_cfg["leaky_relu_slope"],
            n_heads=model_cfg.get("n_heads", 4),
            n_encoder_layers=model_cfg.get("n_encoder_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 128),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", 120),
        )

    else:
        raise ValueError(
            f"Unknown model: '{name}'. "
            "Must be one of: deeplob, deeplob_attention, cnn_only, cnn_attention"
        )


@torch.no_grad()
def evaluate_test_extended(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """Evaluate model on test set with extended metrics.

    Returns:
        (test_loss, test_accuracy, test_f1_weighted, test_predictions, test_labels)
        Predictions and labels are CPU tensors of shape (N,).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        total += x.size(0)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / total
    accuracy = (all_preds == all_labels).float().mean().item()
    f1_w = f1_score(all_labels.numpy(), all_preds.numpy(), average="weighted")

    return avg_loss, accuracy, f1_w, all_preds, all_labels


def train_with_warmup(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    max_epochs: int = 200,
    learning_rate: float = 0.001,
    adam_epsilon: float = 1e-8,
    warmup_epochs: int = 5,
    patience: int = 20,
    device: torch.device | None = None,
    save_dir: Path | None = None,
    seed: int = 42,
) -> TrainingHistory:
    """Training loop with LR warmup, early stopping, and checkpointing.

    📚 Study this on Desktop: Learning rate warmup.
       Transformers are sensitive to large initial updates because the
       attention weights are randomly initialized. Warmup gradually
       increases the LR from 0 to the target value, letting the model
       stabilize before taking full-size steps.
    """
    import time

    torch.manual_seed(seed)
    if device is None:
        device = select_device()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        eps=adam_epsilon,
    )

    # LR warmup scheduler
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    history = TrainingHistory()
    epochs_without_improvement = 0
    checkpoint_path = save_dir / "best_model.pt" if save_dir else None

    print(f"\n{'='*60}")
    print(f"Training {model.__class__.__name__} — {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}")
    print(f"LR: {learning_rate}, Adam epsilon: {adam_epsilon}, Warmup: {warmup_epochs} epochs")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if checkpoint_path:
        print(f"Checkpoints: {checkpoint_path}")
    print(f"{'='*60}\n")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # --- Validate ---
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # --- LR scheduler step ---
        if scheduler is not None:
            scheduler.step()

        # --- Record history ---
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)
        history.epoch_times.append(epoch_time)

        # --- Early stopping check ---
        if val_acc > history.best_val_accuracy:
            history.best_val_accuracy = val_acc
            history.best_epoch = epoch
            epochs_without_improvement = 0

            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)

            marker = " ★ new best"
        else:
            epochs_without_improvement += 1
            marker = ""

        # --- Epoch log ---
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"{epoch_time:.1f}s{marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            print(f"Best val accuracy: {history.best_val_accuracy:.4f} at epoch {history.best_epoch}")
            break
    else:
        print(f"\nReached max epochs ({max_epochs})")
        print(f"Best val accuracy: {history.best_val_accuracy:.4f} at epoch {history.best_epoch}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train DeepLOB extension on FI-2010")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/extension_fi2010.yaml",
        help="Path to config file",
    )
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon k")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/mps/cuda)")
    args = parser.parse_args()

    # --- Load config ---
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    print(f"Config loaded from {config_path}")

    # Apply CLI overrides
    if args.horizon is not None:
        config["data"]["horizon"] = args.horizon
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs

    horizon = config["data"]["horizon"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (forced)")
    else:
        device = select_device()

    # --- Seed ---
    torch.manual_seed(training_cfg["seed"])

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config["data"].get("data_dir", "data/processed"),
        horizon=horizon,
        batch_size=training_cfg["batch_size"],
        project_root=PROJECT_ROOT,
    )

    # --- Model ---
    model = build_model(config)
    print(f"\nModel: {model_cfg['name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Experiment directory ---
    save_dir = PROJECT_ROOT / config["logging"]["save_dir"] / f"k{horizon}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoint for reproducibility
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- Train ---
    history = train_with_warmup(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=training_cfg["max_epochs"],
        learning_rate=training_cfg["learning_rate"],
        adam_epsilon=training_cfg.get("adam_epsilon", 1e-8),
        warmup_epochs=training_cfg.get("warmup_epochs", 0),
        patience=training_cfg["early_stopping_patience"],
        device=device,
        save_dir=save_dir,
        seed=training_cfg["seed"],
    )

    # --- Evaluate best model on test set ---
    print(f"\n{'='*60}")
    print("Loading best model for test evaluation...")
    checkpoint_path = save_dir / "best_model.pt"
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model)
        model = model.to(device)
    else:
        print("No checkpoint found — using final model state")
        model = model.to(device)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_test_extended(
        model, test_loader, device
    )
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1 (wtd): {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"{'='*60}")

    # --- Save history (extended format) ---
    torch.save(
        {
            # Standard fields (compatible with DeepLOB history.pt)
            "train_loss": history.train_loss,
            "train_accuracy": history.train_accuracy,
            "val_loss": history.val_loss,
            "val_accuracy": history.val_accuracy,
            "epoch_times": history.epoch_times,
            "best_epoch": history.best_epoch,
            "best_val_accuracy": history.best_val_accuracy,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            # Extended fields
            "test_f1_weighted": test_f1,
            "test_predictions": test_preds,
            "test_labels": test_labels,
        },
        save_dir / "history.pt",
    )
    print(f"\nHistory saved to {save_dir / 'history.pt'}")


if __name__ == "__main__":
    main()
