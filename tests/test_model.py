"""
Tests for DeepLOB model architecture.

Verifies:
- Forward pass produces correct output shape
- Intermediate shapes at each block boundary
- Parameter count matches expectations (~144K for official code spec)
- No NaN/Inf in outputs for reasonable inputs
- Model works on different batch sizes
- Gradient flow (all parameters receive gradients)
"""

import pytest
import torch

from src.models.deeplob import DeepLOB


@pytest.fixture
def model():
    """Create a DeepLOB model with default (official code) settings."""
    return DeepLOB()


@pytest.fixture
def sample_input():
    """Random input mimicking LOB data: (batch=4, channels=1, T=100, features=40)."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 100, 40)


# =============================================================================
# Forward pass tests
# =============================================================================

class TestForwardPass:
    """Test that the full forward pass works correctly."""

    def test_output_shape(self, model, sample_input):
        """Output should be (batch, 3) logits."""
        output = model(sample_input)
        assert output.shape == (4, 3), f"Expected (4, 3), got {output.shape}"

    def test_output_is_logits_not_probabilities(self, model, sample_input):
        """Output should be raw logits (can be negative, don't sum to 1)."""
        output = model(sample_input)
        # Logits can be negative — softmax hasn't been applied
        # They should NOT sum to 1 (that would indicate softmax was applied)
        sums = output.sum(dim=1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=0.01), (
            "Output sums to ~1 for all samples — looks like softmax was applied. "
            "DeepLOB should output raw logits (CrossEntropyLoss applies softmax)."
        )

    def test_no_nan_in_output(self, model, sample_input):
        """No NaN values in output."""
        output = model(sample_input)
        assert not torch.isnan(output).any(), "NaN detected in model output"

    def test_no_inf_in_output(self, model, sample_input):
        """No Inf values in output."""
        output = model(sample_input)
        assert not torch.isinf(output).any(), "Inf detected in model output"

    def test_batch_size_1(self, model):
        """Model works with batch size 1."""
        x = torch.randn(1, 1, 100, 40)
        output = model(x)
        assert output.shape == (1, 3)

    def test_batch_size_64(self, model):
        """Model works with training batch size (64)."""
        x = torch.randn(64, 1, 100, 40)
        output = model(x)
        assert output.shape == (64, 3)


# =============================================================================
# Shape tests at block boundaries
# =============================================================================

class TestBlockShapes:
    """Verify tensor shapes at each block boundary.

    These catch dimension mismatches early — the most common bug
    when implementing CNN architectures from papers.
    """

    def test_conv_block1_shape(self, model, sample_input):
        """Block 1: (B, 1, 100, 40) → (B, 32, 102, 20)."""
        out = model.conv_block1(sample_input)
        assert out.shape == (4, 32, 102, 20), f"Block 1: expected (4,32,102,20), got {out.shape}"

    def test_conv_block2_shape(self, model, sample_input):
        """Block 2: → (B, 32, 104, 10)."""
        x = model.conv_block1(sample_input)
        out = model.conv_block2(x)
        assert out.shape == (4, 32, 104, 10), f"Block 2: expected (4,32,104,10), got {out.shape}"

    def test_conv_block3_shape(self, model, sample_input):
        """Block 3: → (B, 32, 106, 1)."""
        x = model.conv_block1(sample_input)
        x = model.conv_block2(x)
        out = model.conv_block3(x)
        assert out.shape == (4, 32, 106, 1), f"Block 3: expected (4,32,106,1), got {out.shape}"

    def test_inception_paths_shape(self, model, sample_input):
        """Each inception path: (B, 64, 106, 1)."""
        x = model.conv_block1(sample_input)
        x = model.conv_block2(x)
        x = model.conv_block3(x)

        for name, path in [("A", model.inception_path_a),
                           ("B", model.inception_path_b),
                           ("C", model.inception_path_c)]:
            out = path(x)
            assert out.shape == (4, 64, 106, 1), (
                f"Inception path {name}: expected (4,64,106,1), got {out.shape}"
            )

    def test_inception_concat_shape(self, model, sample_input):
        """After concat: (B, 192, 106, 1)."""
        x = model.conv_block1(sample_input)
        x = model.conv_block2(x)
        x = model.conv_block3(x)

        pa = model.inception_path_a(x)
        pb = model.inception_path_b(x)
        pc = model.inception_path_c(x)
        concat = torch.cat([pa, pb, pc], dim=1)

        assert concat.shape == (4, 192, 106, 1), (
            f"Inception concat: expected (4,192,106,1), got {concat.shape}"
        )


# =============================================================================
# Parameter count tests
# =============================================================================

class TestParameterCount:
    """Verify parameter count matches expectations.

    Official code spec (32 filters, BatchNorm, 64 inception filters):
    ~144K total parameters.
    """

    def test_total_parameter_count(self, model):
        """Total params should be ~144K for official code spec."""
        total = sum(p.numel() for p in model.parameters())
        assert 140_000 < total < 150_000, (
            f"Expected ~144K params (official code spec), got {total:,}"
        )

    def test_all_parameters_trainable(self, model):
        """All parameters should be trainable (no frozen layers)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable, (
            f"{total - trainable} parameters are frozen — all should be trainable"
        )

    def test_lstm_is_largest_component(self, model):
        """LSTM should have the most parameters (weight sharing advantage)."""
        lstm_params = sum(p.numel() for p in model.lstm.parameters())
        total = sum(p.numel() for p in model.parameters())
        assert lstm_params > total * 0.4, (
            f"LSTM has {lstm_params:,} params ({lstm_params/total:.0%} of total). "
            "Expected >40% — LSTM should be the largest component."
        )


# =============================================================================
# Gradient flow tests
# =============================================================================

class TestGradientFlow:
    """Verify gradients flow through all layers."""

    def test_all_parameters_receive_gradients(self, model, sample_input):
        """Every parameter should get a gradient after backward pass."""
        output = model(sample_input)
        loss = output.sum()  # Simple scalar loss for testing
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_loss_backward_works(self, model, sample_input):
        """CrossEntropyLoss + backward should work without errors."""
        targets = torch.randint(0, 3, (4,))
        output = model(sample_input)
        loss = torch.nn.functional.cross_entropy(output, targets)
        loss.backward()
        assert loss.item() > 0, "Loss should be positive"


# =============================================================================
# Configuration tests
# =============================================================================

class TestConfiguration:
    """Test model with different configurations."""

    def test_custom_num_classes(self):
        """Model works with different number of classes."""
        model = DeepLOB(num_classes=5)
        x = torch.randn(2, 1, 100, 40)
        output = model(x)
        assert output.shape == (2, 5)

    def test_paper_spec_16_filters(self):
        """Model works with paper spec (16 filters, 32 inception)."""
        model = DeepLOB(conv_filters=16, inception_filters=32)
        x = torch.randn(2, 1, 100, 40)
        output = model(x)
        assert output.shape == (2, 3)

        # Paper says ~60K params. With BatchNorm (not in paper but in our impl),
        # it's ~62K. Should be well under the official code's ~144K.
        total = sum(p.numel() for p in model.parameters())
        assert 55_000 < total < 70_000, f"Paper spec should be ~62K params, got {total:,}"

    def test_deeper_lstm(self):
        """Model works with multi-layer LSTM."""
        model = DeepLOB(lstm_layers=2)
        x = torch.randn(2, 1, 100, 40)
        output = model(x)
        assert output.shape == (2, 3)
