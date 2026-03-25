"""Training loop for DeepLOB.

📚 Study this on Desktop: The training loop pattern in PyTorch.
   Unlike Keras' model.fit(), PyTorch requires you to write the loop yourself.
   This gives more control but means you handle: batching, loss computation,
   gradient computation (backward), parameter updates (optimizer.step()),
   and zeroing gradients between batches.

The training loop has this structure each epoch:
   1. TRAIN phase: iterate batches, compute loss, backprop, update weights
   2. VALIDATE phase: iterate batches, compute loss + accuracy (no gradients)
   3. Check early stopping: did validation accuracy improve?
   4. Save checkpoint if it's the best model so far

📚 Study this on Desktop: Early stopping — a regularization technique.
   If validation accuracy doesn't improve for `patience` epochs, we stop
   training to prevent overfitting. The model memorizes training data but
   gets worse on unseen data — early stopping catches this inflection point.
"""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """Stores per-epoch metrics for analysis and plotting.

    This is like a lightweight version of what W&B tracks — we keep a local
    copy so we can inspect results without needing an external service.
    """

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0


def select_device() -> torch.device:
    """Pick the best available device: MPS (Apple Silicon GPU) > CUDA > CPU.

    📚 Study this on Desktop: MPS (Metal Performance Shaders) is Apple's
       GPU compute framework. PyTorch added MPS backend so you can train
       on your MacBook's GPU. It's ~2-5x faster than CPU for typical models.
       CUDA is NVIDIA's GPU framework — used on Linux/Windows with NVIDIA GPUs.
    """
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch: forward pass → loss → backward → update weights.

    📚 Study this on Desktop: Why do we call optimizer.zero_grad()?
       PyTorch ACCUMULATES gradients by default. If you don't zero them,
       each batch's gradients add to the previous batch's — giving wrong
       updates. zero_grad() clears the slate before each batch.

    Returns:
        (avg_loss, accuracy) for this epoch
    """
    model.train()  # Enable dropout, BatchNorm in training mode

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Forward pass: input → model → logits
        logits = model(x)

        # Compute loss: how wrong are our predictions?
        loss = criterion(logits, y)

        # Backward pass: compute gradients (d_loss / d_weights)
        optimizer.zero_grad()
        loss.backward()

        # Update weights: w = w - lr * gradient
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * x.size(0)  # Undo mean reduction
        preds = logits.argmax(dim=1)  # Predicted class = highest logit
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()  # Disable gradient computation for validation (saves memory)
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation: same as training but NO gradient updates.

    📚 Study this on Desktop: model.eval() vs model.train()
       - eval() disables dropout and uses running stats for BatchNorm
       - train() enables dropout and computes batch stats for BatchNorm
       Using the wrong mode gives misleading validation metrics!

    @torch.no_grad() is a decorator that disables gradient tracking.
    This saves GPU memory and speeds up computation since we don't need
    gradients during validation.

    Returns:
        (avg_loss, accuracy) for the validation set
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    path: Path,
) -> None:
    """Save model + optimizer state to disk.

    📚 Study this on Desktop: What's in a checkpoint?
       - model.state_dict(): all learned weights and biases
       - optimizer.state_dict(): momentum buffers, learning rate state
       - epoch + accuracy: so we know WHEN this was the best model
       Saving the optimizer lets you resume training exactly where you left off.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Returns the full checkpoint dict so caller can read epoch/accuracy.
    """
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 200,
    learning_rate: float = 0.01,
    adam_epsilon: float = 1.0,
    patience: int = 20,
    device: Optional[torch.device] = None,
    save_dir: Optional[Path] = None,
    seed: int = 42,
) -> TrainingHistory:
    """Full training loop with validation and early stopping.

    📚 Study this on Desktop: Adam optimizer hyperparameters.
       - lr=0.01: learning rate — how big each weight update step is.
         0.01 is relatively large; many use 0.001. The paper uses 0.01.
       - epsilon=1.0: numerical stability term in Adam's denominator.
         Default is 1e-8, but the paper uses 1.0 — this effectively
         dampens the adaptive learning rate, making Adam behave more
         like SGD with momentum. It's an unusual choice that the authors
         found works well for this specific problem.

    Args:
        model: The DeepLOB model
        train_loader: Training data
        val_loader: Validation data
        max_epochs: Maximum epochs before stopping
        learning_rate: Adam learning rate
        adam_epsilon: Adam epsilon parameter (paper uses 1.0)
        patience: Early stopping patience (epochs without improvement)
        device: Compute device (auto-detected if None)
        save_dir: Directory for checkpoints (None = no saving)
        seed: Random seed for reproducibility

    Returns:
        TrainingHistory with all per-epoch metrics
    """
    # Setup — seed all RNGs for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is None:
        device = select_device()

    model = model.to(device)

    # CrossEntropyLoss: combines softmax + negative log-likelihood.
    # It expects raw logits (which our model outputs) and integer labels.
    # 📚 Study this on Desktop: CrossEntropyLoss — why we don't put softmax
    #    in the model. Combining them is numerically more stable than
    #    doing softmax first then log-loss separately.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        eps=adam_epsilon,  # Paper's unusual epsilon=1.0
    )

    history = TrainingHistory()
    epochs_without_improvement = 0

    checkpoint_path = save_dir / "best_model.pt" if save_dir else None

    print(f"\n{'='*60}")
    print(f"Training DeepLOB — {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}")
    print(f"LR: {learning_rate}, Adam epsilon: {adam_epsilon}")
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

            # Save best checkpoint
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)

            marker = " ★ new best"
        else:
            epochs_without_improvement += 1
            marker = ""

        # --- Epoch log ---
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"{epoch_time:.1f}s{marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            print(f"Best val accuracy: {history.best_val_accuracy:.4f} at epoch {history.best_epoch}")
            break

    else:
        # Loop completed without early stopping
        print(f"\nReached max epochs ({max_epochs})")
        print(f"Best val accuracy: {history.best_val_accuracy:.4f} at epoch {history.best_epoch}")

    return history
