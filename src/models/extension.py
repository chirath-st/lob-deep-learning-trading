"""
DeepLOB-Attention: Transformer-based extension of DeepLOB.
==========================================================
Replaces LSTM temporal modeling with multi-head self-attention
(Transformer encoder). CNN + Inception feature extraction is
reused identically from DeepLOB via composition.

Architecture overview (verified tensor shapes):
    Input (B, 1, 100, 40)
    → CNN Block 1-3 + Inception  [identical to DeepLOB]  → (B, 192, 106, 1)
    → Squeeze + Permute                                   → (B, 106, 192)
    → Learnable Positional Encoding                        → (B, 106, 192)
    → 2× TransformerEncoderLayer(d=192, heads=4, ff=256)   → (B, 106, 192)
    → Mean Pool over sequence                              → (B, 192)
    → FC                                                   → (B, 3)

📚 Study this on Desktop: Self-attention vs LSTM for sequence modeling.
   LSTM processes tokens sequentially — token 106 must wait for tokens 1-105.
   Self-attention computes ALL pairwise interactions in parallel.
   For LOB data, this means the model can directly compare the current LOB
   state (position 106) with ANY historical state (position 1-105) without
   the information passing through intermediate LSTM states.

   Trade-off: Attention is O(L²·d) vs LSTM's O(L·d²). With L=106 and d=192,
   attention wins because L < d. For very long sequences (L >> d), LSTM
   or SSMs (Mamba) would be more efficient.

ADR: See docs/architecture_decisions.md — ADR-003
Reference: Zhang et al. 2019 for base architecture.
"""

import torch
import torch.nn as nn

from src.models.deeplob import DeepLOB


class DeepLOBAttention(nn.Module):
    """
    DeepLOB with Transformer encoder replacing LSTM.

    CNN blocks 1-3 and Inception module are reused identically from DeepLOB
    via composition. The LSTM is replaced by:
    1. Learnable positional encoding
    2. N Transformer encoder layers (self-attention + FFN + LayerNorm)
    3. Pooling (mean/last/cls) over the sequence

    📚 Study this on Desktop: Why composition over inheritance?
       We instantiate a DeepLOB object and reference its CNN/Inception layers.
       This is cleaner than inheriting — we don't carry the LSTM/FC from the
       parent, and changes to DeepLOB's forward() don't silently break us.

    Args:
        num_classes: Number of output classes (default: 3)
        conv_filters: CNN filter count (default: 32, matches DeepLOB)
        inception_filters: Inception filter count (default: 64, matches DeepLOB)
        leaky_relu_slope: LeakyReLU slope (default: 0.01, matches DeepLOB)
        d_model: Transformer model dimension (default: 192 = 3×inception_filters)
        n_heads: Number of attention heads (default: 4, 192/4=48 dims per head)
        n_encoder_layers: Number of Transformer encoder layers (default: 2)
        dim_feedforward: FFN hidden dimension (default: 256)
        dropout: Dropout rate for attention and FFN (default: 0.1)
        pooling: Sequence pooling strategy — "mean", "last", or "cls" (default: "mean")
        max_seq_len: Max sequence length for positional encoding (default: 120)
    """

    def __init__(
        self,
        num_classes: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 64,
        leaky_relu_slope: float = 0.01,
        d_model: int = 192,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
        max_seq_len: int = 120,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.pooling = pooling

        # =====================================================================
        # REUSE CNN + INCEPTION FROM DEEPLOB
        # =====================================================================
        # Compose a DeepLOB instance and reference its feature extraction layers.
        # The LSTM and FC from DeepLOB are not used (garbage collected).
        # =====================================================================
        base = DeepLOB(
            num_classes=num_classes,
            conv_filters=conv_filters,
            inception_filters=inception_filters,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.conv_block1 = base.conv_block1
        self.conv_block2 = base.conv_block2
        self.conv_block3 = base.conv_block3
        self.inception_path_a = base.inception_path_a
        self.inception_path_b = base.inception_path_b
        self.inception_path_c = base.inception_path_c

        # Inception output channels
        inception_out = inception_filters * 3  # 64 × 3 = 192

        # Optional projection if d_model != inception output
        if d_model != inception_out:
            self.input_projection = nn.Linear(inception_out, d_model)
        else:
            self.input_projection = None

        # =====================================================================
        # POSITIONAL ENCODING (Learnable)
        # =====================================================================
        # 📚 Study this on Desktop: Why positional encoding?
        #    Unlike LSTM, self-attention has no notion of order — it treats
        #    the sequence as a SET. Without positional encoding, the model
        #    can't distinguish "bid volume at timestep 1" from "bid volume
        #    at timestep 100". The learnable PE adds a unique vector to each
        #    position so the model knows WHERE each token is in the sequence.
        #
        #    We use learnable (not sinusoidal) PE because our sequence length
        #    is always exactly 106 — no need for generalization to unseen lengths.
        # =====================================================================
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # =====================================================================
        # CLS TOKEN (only if pooling == "cls")
        # =====================================================================
        # 📚 Study this on Desktop: CLS token — a learnable "query" vector
        #    prepended to the sequence. After self-attention, the CLS token
        #    has attended to all other tokens, making it a global summary.
        #    This is the BERT approach to classification.
        # =====================================================================
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # =====================================================================
        # TRANSFORMER ENCODER
        # =====================================================================
        # 📚 Study this on Desktop: TransformerEncoderLayer internals:
        #    1. Multi-Head Self-Attention: Q, K, V all come from the SAME input.
        #       Each head learns different attention patterns (e.g., one head
        #       might focus on recent timesteps, another on bid-ask imbalance).
        #    2. Add & LayerNorm (residual connection + normalization)
        #    3. FFN: two linear layers with ReLU (expands then contracts dims)
        #    4. Add & LayerNorm again
        #
        #    PyTorch uses batch_first=True so input is (B, L, d_model).
        # =====================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,  # Post-LN (original Transformer style)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
        )

        # =====================================================================
        # FULLY CONNECTED OUTPUT
        # =====================================================================
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepLOB-Attention.

        Args:
            x: Input tensor of shape (B, 1, 100, 40)

        Returns:
            Logits of shape (B, 3)
        """
        # === CNN Feature Extraction (identical to DeepLOB) ===
        x = self.conv_block1(x)   # (B, 1, 100, 40) → (B, 32, 102, 20)
        x = self.conv_block2(x)   # → (B, 32, 104, 10)
        x = self.conv_block3(x)   # → (B, 32, 106, 1)

        # === Inception Module (identical to DeepLOB) ===
        path_a = self.inception_path_a(x)   # (B, 64, 106, 1)
        path_b = self.inception_path_b(x)   # (B, 64, 106, 1)
        path_c = self.inception_path_c(x)   # (B, 64, 106, 1)
        x = torch.cat([path_a, path_b, path_c], dim=1)  # (B, 192, 106, 1)

        # === Reshape for Transformer ===
        x = x.squeeze(-1)          # (B, 192, 106)
        x = x.permute(0, 2, 1)     # (B, 106, 192)

        # Optional projection
        if self.input_projection is not None:
            x = self.input_projection(x)  # (B, 106, d_model)

        seq_len = x.size(1)

        # === Positional Encoding ===
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.pos_dropout(x)
        # (B, 106, d_model)

        # === CLS Token (if using cls pooling) ===
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 107, d_model)

        # === Transformer Encoder ===
        x = self.transformer_encoder(x)
        # (B, 106, d_model) or (B, 107, d_model) with CLS

        # === Pooling ===
        if self.pooling == "mean":
            x = x.mean(dim=1)       # (B, d_model)
        elif self.pooling == "last":
            x = x[:, -1, :]         # (B, d_model) — last token, like LSTM
        elif self.pooling == "cls":
            x = x[:, 0, :]          # (B, d_model) — CLS token
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # === Classification ===
        x = self.fc(x)  # (B, 3)

        return x


class DeepLOBCNNOnly(nn.Module):
    """
    Ablation variant: CNN blocks only, no Inception, no temporal modeling.

    📚 Study this on Desktop: Why this ablation?
       By removing both Inception and temporal modeling, we isolate the
       contribution of the CNN spatial feature extraction alone. If this
       model performs poorly, it confirms that temporal modeling (LSTM or
       attention) is critical for LOB prediction.

    Architecture:
        Input (B, 1, 100, 40)
        → CNN Block 1-3          → (B, 32, 106, 1)
        → Global Avg Pool (time) → (B, 32)
        → FC                     → (B, 3)
    """

    def __init__(
        self,
        num_classes: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 64,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()

        base = DeepLOB(
            num_classes=num_classes,
            conv_filters=conv_filters,
            inception_filters=inception_filters,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.conv_block1 = base.conv_block1
        self.conv_block2 = base.conv_block2
        self.conv_block3 = base.conv_block3

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN blocks, global pooling, and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input LOB tensor of shape (B, 1, 100, 40).

        Returns
        -------
        torch.Tensor
            Class logits of shape (B, 3).
        """
        x = self.conv_block1(x)   # (B, 32, 102, 20)
        x = self.conv_block2(x)   # (B, 32, 104, 10)
        x = self.conv_block3(x)   # (B, 32, 106, 1)
        x = self.global_avg_pool(x)  # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 32)
        x = self.fc(x)               # (B, 3)
        return x


class DeepLOBCNNAttention(nn.Module):
    """
    Ablation variant: CNN blocks + Transformer, skip Inception.

    📚 Study this on Desktop: Why this ablation?
       By using attention directly on CNN output (d=32) without Inception,
       we test whether Inception's multi-scale temporal features are
       necessary, or if attention alone can learn those patterns.

    Architecture:
        Input (B, 1, 100, 40)
        → CNN Block 1-3                                    → (B, 32, 106, 1)
        → Squeeze + Permute                                → (B, 106, 32)
        → Positional Encoding + TransformerEncoder(d=32)   → (B, 106, 32)
        → Mean Pool                                        → (B, 32)
        → FC                                               → (B, 3)
    """

    def __init__(
        self,
        num_classes: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 64,
        leaky_relu_slope: float = 0.01,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 120,
    ):
        super().__init__()

        d_model = conv_filters  # 32

        base = DeepLOB(
            num_classes=num_classes,
            conv_filters=conv_filters,
            inception_filters=inception_filters,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.conv_block1 = base.conv_block1
        self.conv_block2 = base.conv_block2
        self.conv_block3 = base.conv_block3

        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN blocks, Transformer encoder, and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input LOB tensor of shape (B, 1, 100, 40).

        Returns
        -------
        torch.Tensor
            Class logits of shape (B, 3).
        """
        x = self.conv_block1(x)   # (B, 32, 102, 20)
        x = self.conv_block2(x)   # (B, 32, 104, 10)
        x = self.conv_block3(x)   # (B, 32, 106, 1)

        x = x.squeeze(-1)          # (B, 32, 106)
        x = x.permute(0, 2, 1)     # (B, 106, 32)

        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.pos_dropout(x)

        x = self.transformer_encoder(x)  # (B, 106, 32)
        x = x.mean(dim=1)                # (B, 32)
        x = self.fc(x)                    # (B, 3)
        return x
