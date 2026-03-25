"""Train DeepLOB on FI-2010 dataset.

Usage:
    python scripts/train.py                          # defaults from config
    python scripts/train.py --horizon 20             # different horizon
    python scripts/train.py --config path/to/cfg.yaml  # custom config

📚 Study this on Desktop: YAML configs for ML experiments.
   Instead of hardcoding hyperparameters, we store them in a YAML file.
   This makes it easy to track what settings produced which results,
   and to reproduce experiments exactly.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.models.deeplob import DeepLOB
from src.training.trainer import (
    TrainingHistory,
    load_checkpoint,
    select_device,
    train,
    validate,
)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on test set. Returns (loss, accuracy)."""
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train DeepLOB on FI-2010")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deeplob_fi2010.yaml",
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
    model = DeepLOB(
        num_classes=model_cfg["num_classes"],
        conv_filters=model_cfg["conv_filters"],
        inception_filters=model_cfg["inception_filters"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        leaky_relu_slope=model_cfg["leaky_relu_slope"],
    )

    # --- Experiment directory ---
    save_dir = PROJECT_ROOT / config["logging"]["save_dir"] / f"k{horizon}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoint for reproducibility
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- Train ---
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=training_cfg["max_epochs"],
        learning_rate=training_cfg["learning_rate"],
        adam_epsilon=training_cfg["adam_epsilon"],
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
        test_loss, test_acc = evaluate_test(model, test_loader, device)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Paper target (k={horizon}): ~84.47% for k=10")
        print(f"{'='*60}")
    else:
        print("No checkpoint found — using final model state")
        model = model.to(device)
        test_loss, test_acc = evaluate_test(model, test_loader, device)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # --- Save history ---
    torch.save(
        {
            "train_loss": history.train_loss,
            "train_accuracy": history.train_accuracy,
            "val_loss": history.val_loss,
            "val_accuracy": history.val_accuracy,
            "epoch_times": history.epoch_times,
            "best_epoch": history.best_epoch,
            "best_val_accuracy": history.best_val_accuracy,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
        },
        save_dir / "history.pt",
    )
    print(f"\nHistory saved to {save_dir / 'history.pt'}")


if __name__ == "__main__":
    main()
