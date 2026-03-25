"""Train DeepLOB with cost-aware loss functions.

Supports four loss types that address the accuracy-profitability gap:

1. weighted_ce  -- Cross-entropy with up-weighted stationary class
2. focal        -- Focal loss (focuses on hard examples)
3. turnover     -- CE + turnover penalty (smooth predictions)
4. sharpe       -- Differentiable Sharpe ratio (direct profit optimization)

All losses are in src/losses/cost_aware.py with detailed explanations.

Usage:
    # Weighted CE with stationary weight = 3.0
    python scripts/train_cost_aware.py \\
        --config configs/cost_aware/weighted_ce.yaml --loss-param 3.0

    # Focal loss with gamma = 2.0
    python scripts/train_cost_aware.py \\
        --config configs/cost_aware/focal.yaml --loss-param 2.0

    # Turnover penalty with lambda = 0.1
    python scripts/train_cost_aware.py \\
        --config configs/cost_aware/turnover.yaml --loss-param 0.1

    # Sharpe loss with warm-start from pre-trained DeepLOB
    python scripts/train_cost_aware.py \\
        --config configs/cost_aware/sharpe.yaml \\
        --warmstart experiments/k10/best_model.pt --loss-param 0.01
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FI2010Dataset, get_dataloaders
from src.losses.cost_aware import (
    DifferentiableSharpeLoss,
    FocalLoss,
    TurnoverPenalizedLoss,
)
from src.training.trainer import (
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
    select_device,
)

# Reuse model factory from extension training
from scripts.train_extension import build_model


# ---------------------------------------------------------------------------
# Config and loss factory
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_loss(config: dict, device: torch.device) -> tuple[nn.Module, bool]:
    """Build loss function from config.

    Returns:
        (criterion, needs_prices): The loss module and whether it requires
        mid-price/spread data (True only for Sharpe loss).
    """
    loss_cfg = config["loss"]
    loss_type = loss_cfg["type"]

    if loss_type == "weighted_ce":
        w_stat = loss_cfg.get("stationary_weight", 2.0)
        weights = torch.tensor([1.0, w_stat, 1.0], device=device)
        print(f"  Class weights: [1.0, {w_stat}, 1.0]")
        return nn.CrossEntropyLoss(weight=weights), False

    elif loss_type == "focal":
        gamma = loss_cfg.get("gamma", 2.0)
        alpha = loss_cfg.get("alpha", None)
        if alpha is not None:
            alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        print(f"  Focal gamma={gamma}, alpha={alpha}")
        return FocalLoss(gamma=gamma, alpha=alpha), False

    elif loss_type == "turnover":
        lam = loss_cfg.get("lambda_turnover", 0.1)
        print(f"  Turnover lambda={lam}")
        return TurnoverPenalizedLoss(lambda_turnover=lam), False

    elif loss_type == "sharpe":
        cm = loss_cfg.get("cost_multiplier", 0.5)
        gc = loss_cfg.get("gamma_cost", 0.5)
        lt = loss_cfg.get("lambda_turnover", 0.01)
        print(f"  Sharpe: cost_mult={cm}, gamma_cost={gc}, lambda_turnover={lt}")
        return DifferentiableSharpeLoss(
            cost_multiplier=cm, gamma_cost=gc, lambda_turnover=lt
        ), True

    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            "Must be one of: weighted_ce, focal, turnover, sharpe"
        )


def loss_param_string(loss_cfg: dict) -> str:
    """Generate a descriptive directory name from loss config."""
    t = loss_cfg["type"]
    if t == "weighted_ce":
        return f"w{loss_cfg.get('stationary_weight', 2.0)}"
    elif t == "focal":
        return f"g{loss_cfg.get('gamma', 2.0)}"
    elif t == "turnover":
        return f"lam{loss_cfg.get('lambda_turnover', 0.1)}"
    elif t == "sharpe":
        gc = loss_cfg.get("gamma_cost", 0.5)
        lt = loss_cfg.get("lambda_turnover", 0.01)
        return f"gc{gc}_lt{lt}"
    return "default"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_sequential_dataloaders(
    config: dict, horizon: int, project_root: Path
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders with shuffle=False for turnover/Sharpe losses.

    Turnover and Sharpe losses need consecutive-in-time samples to compute
    meaningful penalties. Shuffling destroys temporal ordering, making the
    turnover computation nonsensical.

    The FI-2010 data is already stored in chronological order within each
    stock's segment. With shuffle=False, consecutive batch samples ARE
    consecutive in time.
    """
    data_dir = config["data"].get("data_dir", "data/processed")
    batch_size = config["training"]["batch_size"]
    processed_dir = project_root / data_dir

    print(f"Loading preprocessed data from {processed_dir}...")
    train_x = torch.load(processed_dir / "train_x.pt", weights_only=True)
    train_y = torch.load(processed_dir / "train_y.pt", weights_only=True)
    val_x = torch.load(processed_dir / "val_x.pt", weights_only=True)
    val_y = torch.load(processed_dir / "val_y.pt", weights_only=True)
    test_x = torch.load(processed_dir / "test_x.pt", weights_only=True)
    test_y = torch.load(processed_dir / "test_y.pt", weights_only=True)

    train_ds = FI2010Dataset(train_x, train_y, horizon=horizon)
    val_ds = FI2010Dataset(val_x, val_y, horizon=horizon)
    test_ds = FI2010Dataset(test_x, test_y, horizon=horizon)

    print(f"  Horizon: k={horizon}")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print("  Sequential loading (shuffle=False) for turnover/Sharpe loss")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Price extraction
# ---------------------------------------------------------------------------


def extract_prices_from_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract mid-prices and spreads from LOB input tensor.

    FI-2010 features are interleaved by level:
        x[:, 0, -1, 0] = ask_price_1 (best ask)
        x[:, 0, -1, 2] = bid_price_1 (best bid)
    The last timestep (index -1) gives the "current" LOB state.

    Args:
        x: Input tensor, shape (B, 1, 100, 40).

    Returns:
        mid_prices: (B,) — (best_ask + best_bid) / 2
        spreads: (B,) — best_ask - best_bid
    """
    best_ask = x[:, 0, -1, 0]  # (B,)
    best_bid = x[:, 0, -1, 2]  # (B,)
    mid_prices = (best_ask + best_bid) / 2.0
    spreads = best_ask - best_bid
    return mid_prices, spreads


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------


def train_one_epoch_cost_aware(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    needs_prices: bool = False,
    loss_type: str = "weighted_ce",
) -> tuple[float, float, dict]:
    """Train one epoch with a cost-aware loss function.

    The key difference from standard training is that some losses (turnover,
    sharpe) need extra information beyond (logits, targets):
    - Turnover loss: returns (loss, components_dict) instead of just loss
    - Sharpe loss: takes (logits, mid_prices, spreads) instead of (logits, targets)

    Returns:
        (avg_loss, accuracy, extra_metrics)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    extra_accum = {}  # Accumulate loss component metrics across batches

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)

        # --- Compute loss (dispatch by type) ---
        if needs_prices:
            # Sharpe loss needs price data extracted from input
            mid_prices, spreads = extract_prices_from_batch(x)
            loss, components = criterion(logits, mid_prices, spreads)
            for k, v in components.items():
                extra_accum[k] = extra_accum.get(k, 0.0) + v
        elif loss_type == "turnover":
            # Turnover loss returns (loss, components_dict)
            loss, components = criterion(logits, y)
            for k, v in components.items():
                extra_accum[k] = extra_accum.get(k, 0.0) + v
        else:
            # Standard loss (weighted CE, focal): just (logits, targets)
            loss = criterion(logits, y)

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Track classification metrics (useful even for Sharpe loss) ---
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    # Average extra metrics over batches
    n_batches = len(loader)
    for k in extra_accum:
        extra_accum[k] /= n_batches

    return total_loss / total, correct / total, extra_accum


@torch.no_grad()
def validate_cost_aware(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    needs_prices: bool = False,
    loss_type: str = "weighted_ce",
) -> tuple[float, float, dict]:
    """Validate with a cost-aware loss function.

    Same structure as training but without gradient computation.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    extra_accum = {}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if needs_prices:
            mid_prices, spreads = extract_prices_from_batch(x)
            loss, components = criterion(logits, mid_prices, spreads)
            for k, v in components.items():
                extra_accum[k] = extra_accum.get(k, 0.0) + v
        elif loss_type == "turnover":
            loss, components = criterion(logits, y)
            for k, v in components.items():
                extra_accum[k] = extra_accum.get(k, 0.0) + v
        else:
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    n_batches = len(loader)
    for k in extra_accum:
        extra_accum[k] /= n_batches

    return total_loss / total, correct / total, extra_accum


@torch.no_grad()
def evaluate_test_extended(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """Evaluate model on test set with classification metrics.

    Always uses standard CE for test evaluation regardless of training loss,
    so we can fairly compare classification quality across all methods.

    Returns:
        (test_loss, accuracy, f1_weighted, predictions, labels)
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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLOB with cost-aware losses"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="Override horizon k"
    )
    parser.add_argument(
        "--loss-param",
        type=float,
        default=None,
        help="Override primary loss hyperparameter "
        "(w_stat for weighted_ce, gamma for focal, "
        "lambda for turnover, lambda_turnover for sharpe)",
    )
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Path to pretrained checkpoint for warm-starting",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force device (cpu/mps/cuda)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override max epochs"
    )
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

    # Override primary loss hyperparameter
    loss_cfg = config["loss"]
    loss_type = loss_cfg["type"]
    if args.loss_param is not None:
        if loss_type == "weighted_ce":
            loss_cfg["stationary_weight"] = args.loss_param
        elif loss_type == "focal":
            loss_cfg["gamma"] = args.loss_param
        elif loss_type == "turnover":
            loss_cfg["lambda_turnover"] = args.loss_param
        elif loss_type == "sharpe":
            loss_cfg["lambda_turnover"] = args.loss_param

    horizon = config["data"]["horizon"]
    training_cfg = config["training"]

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (forced)")
    else:
        device = select_device()

    # --- Seed everything for reproducibility ---
    seed = training_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Data ---
    # Turnover and Sharpe losses need sequential (non-shuffled) loading
    needs_sequential = loss_type in ("turnover", "sharpe")
    if needs_sequential:
        train_loader, val_loader, test_loader = get_sequential_dataloaders(
            config, horizon, PROJECT_ROOT
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=config["data"].get("data_dir", "data/processed"),
            horizon=horizon,
            batch_size=training_cfg["batch_size"],
            project_root=PROJECT_ROOT,
        )

    # --- Model ---
    model = build_model(config)
    print(f"\nModel: {config['model']['name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Warm-start from pretrained checkpoint ---
    warmstart_path = args.warmstart or training_cfg.get("warmstart_from")
    if warmstart_path:
        warmstart_path = PROJECT_ROOT / warmstart_path
        if warmstart_path.exists():
            print(f"\nWarm-starting from {warmstart_path}")
            print(
                "  Loading pre-trained weights. The CNN backbone already knows"
            )
            print(
                "  good LOB features -- fine-tuning for the new objective."
            )
            ckpt = torch.load(
                warmstart_path, weights_only=False, map_location="cpu"
            )
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        else:
            print(f"WARNING: Warm-start path not found: {warmstart_path}")

    model = model.to(device)

    # --- Loss function ---
    print(f"\nLoss type: {loss_type}")
    criterion, needs_prices = build_loss(config, device)

    # --- Experiment directory ---
    param_str = loss_param_string(loss_cfg)
    base_save_dir = config.get("logging", {}).get(
        "save_dir", f"experiments/cost_aware/{loss_type}"
    )
    save_dir = PROJECT_ROOT / base_save_dir / param_str / f"k{horizon}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside checkpoint for reproducibility
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- Optimizer ---
    lr = training_cfg["learning_rate"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        eps=training_cfg.get("adam_epsilon", 1e-8),
    )

    # --- LR warmup scheduler ---
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: min(1.0, (e + 1) / warmup_epochs)
        )
    else:
        scheduler = None

    # --- Early stopping config ---
    max_epochs = training_cfg["max_epochs"]
    patience = training_cfg["early_stopping_patience"]

    # For Sharpe loss: use val loss for early stopping (lower = better Sharpe)
    # For classification losses: use val accuracy (higher = better)
    use_loss_for_es = loss_type == "sharpe"
    es_metric_name = "val_loss" if use_loss_for_es else "val_accuracy"

    # --- Print training summary ---
    print(f"\n{'='*60}")
    print(f"Training {config['model']['name']} with {loss_type} loss")
    print(f"Device: {device}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}")
    print(f"LR: {lr}, Warmup: {warmup_epochs} epochs")
    print(f"Early stopping on: {es_metric_name}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    # --- Training loop ---
    history = TrainingHistory()
    best_val_metric = float("inf") if use_loss_for_es else 0.0
    best_epoch = 0
    epochs_no_improve = 0
    checkpoint_path = save_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- Train ---
        train_loss, train_acc, train_extra = train_one_epoch_cost_aware(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            needs_prices=needs_prices,
            loss_type=loss_type,
        )

        # --- Validate ---
        val_loss, val_acc, val_extra = validate_cost_aware(
            model,
            val_loader,
            criterion,
            device,
            needs_prices=needs_prices,
            loss_type=loss_type,
        )

        epoch_time = time.time() - t0

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
        if use_loss_for_es:
            improved = val_loss < best_val_metric
            if improved:
                best_val_metric = val_loss
        else:
            improved = val_acc > best_val_metric
            if improved:
                best_val_metric = val_acc

        if improved:
            best_epoch = epoch
            history.best_epoch = epoch
            history.best_val_accuracy = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            marker = " * new best"
        else:
            epochs_no_improve += 1
            marker = ""

        # --- Epoch log ---
        lr_now = optimizer.param_groups[0]["lr"]
        extra_str = ""
        if train_extra:
            parts = [f"{k}={v:.4f}" for k, v in train_extra.items()]
            extra_str = " | " + ", ".join(parts)

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train L: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val L: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"LR: {lr_now:.6f} | "
            f"{epoch_time:.1f}s{extra_str}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break
    else:
        print(f"\nReached max epochs ({max_epochs})")

    print(
        f"Best epoch: {best_epoch}, "
        f"{es_metric_name}: {best_val_metric:.4f}"
    )

    # --- Load best model for test evaluation ---
    print(f"\n{'='*60}")
    print("Loading best model for test evaluation...")
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model)
        model = model.to(device)
    else:
        print("No checkpoint found -- using final model state")

    test_loss, test_acc, test_f1, test_preds, test_labels = (
        evaluate_test_extended(model, test_loader, device)
    )
    print(f"Test Results ({loss_type} loss, param={loss_param_string(loss_cfg)}):")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 (wtd): {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  CE Loss:  {test_loss:.4f}")
    print(f"{'='*60}")

    # --- Save history (compatible with Phase 5 format) ---
    history_data = {
        # Standard fields (compatible with existing analysis code)
        "train_loss": history.train_loss,
        "train_accuracy": history.train_accuracy,
        "val_loss": history.val_loss,
        "val_accuracy": history.val_accuracy,
        "epoch_times": history.epoch_times,
        "best_epoch": best_epoch,
        "best_val_accuracy": history.best_val_accuracy,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        # Extended fields
        "test_f1_weighted": test_f1,
        "test_predictions": test_preds,
        "test_labels": test_labels,
        # Cost-aware metadata
        "loss_type": loss_type,
        "loss_config": dict(loss_cfg),
    }
    torch.save(history_data, save_dir / "history.pt")
    print(f"\nHistory saved to {save_dir / 'history.pt'}")


if __name__ == "__main__":
    main()
