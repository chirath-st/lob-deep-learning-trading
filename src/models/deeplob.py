"""
DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
================================================================
Zhang, Zohren, Roberts (2019) — IEEE Transactions on Signal Processing

This follows the OFFICIAL CODE implementation (not paper specs):
- 32 conv filters (paper says 16)
- BatchNorm after each conv layer (paper omits this)
- Tanh activation in conv block 2 (paper says all LeakyReLU)
- 64 inception filters → 192 LSTM input (paper: 32 → 96)

Architecture overview (verified tensor shapes):
    Input (B, 1, 100, 40)
    → CNN Block 1: pairs price+volume                  → (B, 32, 102, 20)
    → CNN Block 2: pairs ask+bid sides                 → (B, 32, 104, 10)
    → CNN Block 3: integrates all 10 levels into 1     → (B, 32, 106, 1)
    → Inception: 3 parallel temporal convolutions       → (B, 192, 106, 1)
    → LSTM: sequence modeling over 106 timesteps        → (B, 64)
    → FC: classification into 3 classes                 → (B, 3)

Note: Time dimension grows from 100→106 because kernel_size=4 with
padding=2 adds +1 per temporal conv. This matches the official code.

ADR: See docs/architecture_decisions.md — chose official code specs for
reproducibility and compatibility with extension papers.
"""

import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """
    DeepLOB model for limit order book mid-price movement prediction.

    The model processes LOB snapshots through three stages:
    1. CNN extracts spatial features (price/volume relationships across levels)
    2. Inception captures multi-scale temporal patterns
    3. LSTM models sequential dependencies

    Args:
        num_classes: Number of output classes (default: 3 for down/stationary/up)
        conv_filters: Number of filters in each conv layer (default: 32)
        inception_filters: Number of filters per inception path (default: 64)
        lstm_hidden: LSTM hidden state size (default: 64)
        lstm_layers: Number of stacked LSTM layers (default: 1)
        leaky_relu_slope: Negative slope for LeakyReLU (default: 0.01)
    """

    def __init__(
        self,
        num_classes: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.inception_filters = inception_filters
        self.lstm_hidden = lstm_hidden

        # =====================================================================
        # CNN BLOCK 1: Price-Volume Pairing
        # =====================================================================
        # Input: (B, 1, 100, 40)
        #   40 features = 10 levels × {ask_price, ask_vol, bid_price, bid_vol}
        #
        # Conv1 uses kernel (1,2) stride (1,2) to pair adjacent features:
        #   ask_price + ask_vol → ask imbalance feature
        #   bid_price + bid_vol → bid imbalance feature
        # This halves the feature dimension: 40 → 20
        #
        # Conv2, Conv3 use kernel (4,1) to look at 4 consecutive timesteps.
        # This is a temporal convolution — like a learned moving average.
        # Padding (2,0) adds 2 rows to BOTH top and bottom (PyTorch symmetric).
        # With kernel=4: output = input + 2*2 - 4 + 1 = input + 1
        # So each temporal conv adds +1 to the time dimension (100→101→102).
        #
        # 📚 Study this on Desktop: How 2D convolutions work on non-image data.
        #    Here the "height" is time (100 steps) and "width" is LOB features.
        # =====================================================================
        self.conv_block1 = nn.Sequential(
            # Conv1: pair price+volume features
            nn.Conv2d(1, conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            # (B, 1, 100, 40) → (B, 32, 100, 20)
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            # Conv2: temporal convolution
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1), padding=(2, 0)),
            # (B, 32, 100, 20) → (B, 32, 101, 20)  [+1 from padding quirk]
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            # Conv3: another temporal convolution
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1), padding=(2, 0)),
            # (B, 32, 101, 20) → (B, 32, 102, 20)  [+1 again]
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
        )
        # After block 1: (B, 32, 102, 20)

        # =====================================================================
        # CNN BLOCK 2: Ask-Bid Pairing
        # =====================================================================
        # Same structure as Block 1, but now:
        # Conv4 with kernel (1,2) stride (1,2) pairs ask+bid features:
        #   ask_imbalance + bid_imbalance → spread/micro-price features
        # This halves features again: 20 → 10 (one feature per LOB level)
        #
        # IMPORTANT: This block uses Tanh activation (official code differs
        # from paper here). Tanh bounds outputs to [-1, 1], which stabilizes
        # inputs to the Inception module.
        # =====================================================================
        self.conv_block2 = nn.Sequential(
            # Conv4: pair ask+bid sides
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            # (B, 32, 102, 20) → (B, 32, 102, 10)
            nn.Tanh(),  # Official code uses Tanh here, not LeakyReLU
            nn.BatchNorm2d(conv_filters),
            # Conv5: temporal convolution
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1), padding=(2, 0)),
            # (B, 32, 102, 10) → (B, 32, 103, 10)
            nn.Tanh(),
            nn.BatchNorm2d(conv_filters),
            # Conv6: temporal convolution
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1), padding=(2, 0)),
            # (B, 32, 103, 10) → (B, 32, 104, 10)
            nn.Tanh(),
            nn.BatchNorm2d(conv_filters),
        )
        # After block 2: (B, 32, 104, 10)

        # =====================================================================
        # CNN BLOCK 3: Level Integration
        # =====================================================================
        # Conv7 uses kernel (1,10) to integrate all 10 LOB levels into a
        # single feature. This is like computing a weighted average across
        # all price levels — the network learns WHICH levels matter most.
        #
        # After this, spatial dimension collapses to 1, and we're left with
        # a temporal sequence of learned LOB features.
        # =====================================================================
        self.conv_block3 = nn.Sequential(
            # Conv7: integrate all 10 levels
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 10)),
            # (B, 32, 104, 10) → (B, 32, 104, 1)
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            # Conv8: temporal convolution
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1), padding=(2, 0)),
            # (B, 32, 104, 1) → (B, 32, 105, 1)
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            # Conv9: temporal convolution
            nn.Conv2d(conv_filters, conv_filters, kernel_size=(4, 1), padding=(2, 0)),
            # (B, 32, 105, 1) → (B, 32, 106, 1)
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
        )
        # After block 3: (B, 32, 106, 1)

        # =====================================================================
        # INCEPTION MODULE
        # =====================================================================
        # Inspired by GoogLeNet's Inception module, but adapted for 1D
        # temporal data. Three parallel paths capture patterns at different
        # time scales:
        #
        # Path A: 1×1 → 3×1  (short-term patterns, ~3 timestep window)
        # Path B: 1×1 → 5×1  (medium-term patterns, ~5 timestep window)
        # Path C: MaxPool3×1 → 1×1  (local max features, ~3 timestep window)
        #
        # The 1×1 convs act as "bottleneck" layers that reduce computation
        # while mixing channel information.
        #
        # All three outputs are concatenated: 64 + 64 + 64 = 192 channels.
        #
        # 📚 Study this on Desktop: Inception/GoogLeNet architecture — why
        #    parallel paths at different scales capture richer features than
        #    a single conv layer.
        # =====================================================================

        # Path A: 1×1 conv → 3×1 conv (short-term temporal patterns)
        self.inception_path_a = nn.Sequential(
            nn.Conv2d(conv_filters, inception_filters, kernel_size=(1, 1)),
            # (B, 32, T', 1) → (B, 64, T', 1)
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
            nn.Conv2d(inception_filters, inception_filters, kernel_size=(3, 1), padding=(1, 0)),
            # (B, 64, T', 1) → (B, 64, T', 1)  [padding keeps T' same]
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
        )

        # Path B: 1×1 conv → 5×1 conv (medium-term temporal patterns)
        self.inception_path_b = nn.Sequential(
            nn.Conv2d(conv_filters, inception_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
            nn.Conv2d(inception_filters, inception_filters, kernel_size=(5, 1), padding=(2, 0)),
            # (B, 64, T', 1) → (B, 64, T', 1)  [padding keeps T' same]
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
        )

        # Path C: MaxPool → 1×1 conv (local max features)
        self.inception_path_c = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            # (B, 32, T', 1) → (B, 32, T', 1)  [stride=1, padding keeps size]
            nn.Conv2d(conv_filters, inception_filters, kernel_size=(1, 1)),
            # (B, 32, T', 1) → (B, 64, T', 1)
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
        )

        # After concatenation: (B, 192, T', 1)
        # Reshape to: (B, 192, T') for LSTM input

        # =====================================================================
        # LSTM BLOCK
        # =====================================================================
        # The LSTM processes the 100-timestep sequence of 192-dimensional
        # feature vectors. It captures long-range temporal dependencies
        # that the local CNN/Inception convolutions cannot.
        #
        # We take ONLY the last timestep's hidden state as output.
        # This is the model's "summary" of the entire 100-step input.
        #
        # Why LSTM over a fully-connected layer?
        # - FC on 192×100 = 19,200 inputs → massive parameter count
        # - LSTM shares weights across timesteps → only ~60K total params
        # - LSTM naturally handles sequential data
        #
        # 📚 Study this on Desktop: LSTM architecture — gates (forget, input,
        #    output), cell state, why it handles vanishing gradients better
        #    than vanilla RNNs.
        # =====================================================================
        lstm_input_size = inception_filters * 3  # 64 × 3 = 192
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,  # 192
            hidden_size=lstm_hidden,      # 64
            num_layers=lstm_layers,        # 1
            batch_first=True,
        )
        # Input: (B, T', 192) → Output: (B, T', 64), hidden: (1, B, 64)

        # =====================================================================
        # FULLY CONNECTED OUTPUT
        # =====================================================================
        # Maps LSTM's 64-dim hidden state to 3 class logits.
        # No softmax here — PyTorch's CrossEntropyLoss applies it internally.
        # =====================================================================
        self.fc = nn.Linear(lstm_hidden, num_classes)
        # (B, 64) → (B, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepLOB.

        Args:
            x: Input tensor of shape (B, 1, 100, 40)
               B = batch size
               1 = single channel (like grayscale image)
               100 = lookback window (timesteps)
               40 = LOB features (10 levels × 4 features each)

        Returns:
            Logits of shape (B, 3) — one score per class (down/stationary/up).
            Apply softmax to get probabilities, or use with CrossEntropyLoss directly.
        """
        # === CNN Feature Extraction ===

        # Block 1: pair price+volume
        x = self.conv_block1(x)
        # (B, 1, 100, 40) → (B, 32, 102, 20)

        # Block 2: pair ask+bid
        x = self.conv_block2(x)
        # (B, 32, 102, 20) → (B, 32, 104, 10)

        # Block 3: integrate all levels
        x = self.conv_block3(x)
        # (B, 32, 104, 10) → (B, 32, 106, 1)

        # === Inception Module ===
        # Three parallel paths on the same input
        path_a = self.inception_path_a(x)   # (B, 64, 106, 1)
        path_b = self.inception_path_b(x)   # (B, 64, 106, 1)
        path_c = self.inception_path_c(x)   # (B, 64, 106, 1)

        # Concatenate along channel dimension
        x = torch.cat([path_a, path_b, path_c], dim=1)
        # (B, 192, 106, 1)

        # Remove the last spatial dimension (it's 1)
        x = x.squeeze(-1)
        # (B, 192, 106)

        # === LSTM ===
        # LSTM expects (B, T, features) with batch_first=True
        x = x.permute(0, 2, 1)
        # (B, 106, 192)

        x, _ = self.lstm(x)
        # (B, 106, 64) — hidden state at every timestep

        # Take only the LAST timestep's output
        x = x[:, -1, :]
        # (B, 64)

        # === Classification ===
        x = self.fc(x)
        # (B, 3)

        return x
