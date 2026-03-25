"""
Tests for the training loop.

Verifies:
- 1-epoch overfit on a tiny batch (loss decreases)
- Early stopping triggers when validation stalls
- Checkpoint save/load roundtrip preserves weights
- Device selection logic works
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.deeplob import DeepLOB
from src.training.trainer import (
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
    select_device,
    train,
    train_one_epoch,
    validate,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_model():
    """Small DeepLOB model for fast tests."""
    return DeepLOB(conv_filters=8, inception_filters=8, lstm_hidden=8)


@pytest.fixture
def tiny_loaders():
    """Tiny DataLoaders (32 samples) for fast training tests.

    Creates synthetic data shaped like FI-2010: (N, 1, 100, 40) inputs,
    integer labels in {0, 1, 2}.
    """
    torch.manual_seed(42)
    n = 32
    x = torch.randn(n, 1, 100, 40)
    y = torch.randint(0, 3, (n,))

    dataset = TensorDataset(x, y)
    # Use same dataset for train and val (overfitting test)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader


# =============================================================================
# Overfit tests — can the model learn ANYTHING?
# =============================================================================


class TestOverfit:
    """Verify the model can overfit a tiny batch.

    This is a critical sanity check: if the model can't even memorize
    32 samples, something is wrong with the forward pass, loss computation,
    or gradient updates.
    """

    def test_loss_decreases_after_training(self, tiny_model, tiny_loaders):
        """Loss should decrease after multiple passes over a tiny dataset."""
        train_loader, val_loader = tiny_loaders
        device = torch.device("cpu")
        model = tiny_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Measure initial loss
        initial_loss, _ = validate(model, train_loader, criterion, device)

        # Train for several epochs
        for _ in range(20):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Measure final loss
        final_loss, _ = validate(model, train_loader, criterion, device)

        assert final_loss < initial_loss, (
            f"Loss didn't decrease: {initial_loss:.4f} → {final_loss:.4f}. "
            "Model failed to learn from tiny dataset."
        )

    def test_accuracy_improves_after_training(self, tiny_model, tiny_loaders):
        """Accuracy should improve after training on a tiny dataset."""
        train_loader, val_loader = tiny_loaders
        device = torch.device("cpu")
        model = tiny_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Measure initial accuracy (should be ~33% for 3 classes)
        _, initial_acc = validate(model, train_loader, criterion, device)

        # Train
        for _ in range(30):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, final_acc = validate(model, train_loader, criterion, device)

        assert final_acc > initial_acc, (
            f"Accuracy didn't improve: {initial_acc:.4f} → {final_acc:.4f}."
        )

    def test_full_train_function_runs(self, tiny_model, tiny_loaders):
        """The full train() function completes without errors."""
        train_loader, val_loader = tiny_loaders

        history = train(
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=5,
            learning_rate=0.001,
            adam_epsilon=1.0,
            patience=100,  # Don't early stop during this test
            device=torch.device("cpu"),
            save_dir=None,  # No checkpointing
            seed=42,
        )

        assert isinstance(history, TrainingHistory)
        assert len(history.train_loss) == 5
        assert len(history.val_accuracy) == 5
        assert history.best_val_accuracy > 0


# =============================================================================
# Early stopping tests
# =============================================================================


class TestEarlyStopping:
    """Verify early stopping triggers when validation stalls."""

    def test_stops_before_max_epochs(self, tiny_model, tiny_loaders):
        """Training should stop early when patience is exhausted.

        We use a very small patience so it's likely to trigger even on
        a tiny dataset where val accuracy may plateau quickly.
        """
        train_loader, val_loader = tiny_loaders

        history = train(
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=200,  # Would take forever without early stopping
            learning_rate=0.001,
            adam_epsilon=1.0,
            patience=3,  # Very impatient — stop quickly
            device=torch.device("cpu"),
            save_dir=None,
            seed=42,
        )

        # Should have stopped well before 200 epochs
        epochs_run = len(history.train_loss)
        assert epochs_run < 200, (
            f"Ran all 200 epochs — early stopping didn't trigger with patience=3"
        )
        # Should have run at least patience+1 epochs (need patience consecutive non-improvements)
        assert epochs_run >= 4, (
            f"Only ran {epochs_run} epochs — should run at least patience+1"
        )

    def test_best_epoch_is_recorded(self, tiny_model, tiny_loaders):
        """TrainingHistory should record which epoch had best val accuracy."""
        train_loader, val_loader = tiny_loaders

        history = train(
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=10,
            learning_rate=0.001,
            adam_epsilon=1.0,
            patience=100,
            device=torch.device("cpu"),
            save_dir=None,
            seed=42,
        )

        assert 1 <= history.best_epoch <= 10
        # Best val accuracy should match the actual best in the list
        assert history.best_val_accuracy == max(history.val_accuracy)


# =============================================================================
# Checkpoint save/load tests
# =============================================================================


class TestCheckpoints:
    """Verify checkpoint save and load preserves model state."""

    def test_save_load_roundtrip(self, tiny_model, tmp_path):
        """Loaded model should produce identical outputs to saved model."""
        # Run a forward pass to get "before" outputs
        torch.manual_seed(42)
        x = torch.randn(4, 1, 100, 40)
        tiny_model.eval()
        with torch.no_grad():
            expected_output = tiny_model(x)

        # Save checkpoint
        optimizer = torch.optim.Adam(tiny_model.parameters())
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(tiny_model, optimizer, epoch=5, val_accuracy=0.85, path=checkpoint_path)

        # Create fresh model and load checkpoint
        fresh_model = DeepLOB(conv_filters=8, inception_filters=8, lstm_hidden=8)
        checkpoint = load_checkpoint(checkpoint_path, fresh_model)

        # Verify metadata
        assert checkpoint["epoch"] == 5
        assert checkpoint["val_accuracy"] == 0.85

        # Verify outputs match
        fresh_model.eval()
        with torch.no_grad():
            loaded_output = fresh_model(x)

        assert torch.allclose(expected_output, loaded_output, atol=1e-6), (
            "Loaded model produces different outputs than saved model"
        )

    def test_checkpoint_saves_during_training(self, tiny_model, tiny_loaders, tmp_path):
        """train() should save a checkpoint when save_dir is provided."""
        train_loader, val_loader = tiny_loaders

        train(
            model=tiny_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=3,
            learning_rate=0.001,
            adam_epsilon=1.0,
            patience=100,
            device=torch.device("cpu"),
            save_dir=tmp_path,
            seed=42,
        )

        checkpoint_path = tmp_path / "best_model.pt"
        assert checkpoint_path.exists(), "Checkpoint file was not created"

        # Verify it's a valid checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_accuracy" in checkpoint


# =============================================================================
# Device selection tests
# =============================================================================


class TestDeviceSelection:
    """Verify device selection logic."""

    def test_returns_valid_device(self):
        """select_device() should return a valid torch.device."""
        device = select_device()
        assert isinstance(device, torch.device)
        # Should be one of the known types
        assert device.type in ("cpu", "mps", "cuda")

    def test_model_runs_on_selected_device(self, tiny_model):
        """Model should work on the selected device."""
        device = select_device()
        model = tiny_model.to(device)
        x = torch.randn(2, 1, 100, 40).to(device)
        output = model(x)
        assert output.shape == (2, 3)
        assert output.device.type == device.type
