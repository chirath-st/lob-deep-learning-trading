# DeepLOB Paper Reference (Zhang, Zohren, Roberts 2019)

## Paper: arXiv:1808.03668v6, IEEE Transactions on Signal Processing
## GitHub: github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

---

## Input Specification
- Shape: `(batch, 100, 40)` — 100 most recent LOB states × 40 features
- Feature ordering per timestep: `{p_a^(i), v_a^(i), p_b^(i), v_b^(i)}` for i=1..10
  - Ask price, Ask volume, Bid price, Bid volume × 10 levels
- FI-2010 uses z-score normalized data (pre-normalized in dataset)

## Architecture (Figure 3 in paper)

### Block 1: Convolutional Feature Extraction
```
Input: (batch, 1, 100, 40)  # treat as 1-channel image

Conv1: 16 filters, kernel (1,2), stride (1,2), padding=(0,0)
  → (batch, 16, 100, 20)  # pairs {price, volume} at each level
  LeakyReLU(0.01)
  
Conv2: 16 filters, kernel (4,1), stride (1,1), padding=(2,0)  # note: paper uses zero-pad to keep time dim
  → (batch, 16, 100, 20)
  LeakyReLU(0.01)

Conv3: 16 filters, kernel (4,1), stride (1,1), padding=(2,0)
  → (batch, 16, 100, 20)
  LeakyReLU(0.01)

Conv4: 16 filters, kernel (1,2), stride (1,2)
  → (batch, 16, 100, 10)  # combines ask+bid at each level → micro-prices
  LeakyReLU(0.01)

Conv5: 16 filters, kernel (4,1), stride (1,1), padding=(2,0)
  → (batch, 16, 100, 10)
  LeakyReLU(0.01)

Conv6: 16 filters, kernel (4,1), stride (1,1), padding=(2,0)
  → (batch, 16, 100, 10)
  LeakyReLU(0.01)

Conv7: 16 filters, kernel (1,10), stride (1,1)
  → (batch, 16, 100, 1)  # integrates all 10 levels
  LeakyReLU(0.01)

Conv8: 16 filters, kernel (4,1), stride (1,1), padding=(2,0)
  → (batch, 16, 100, 1)
  LeakyReLU(0.01)

Conv9: 16 filters, kernel (4,1), stride (1,1), padding=(2,0)
  → (batch, 16, 100, 1)
  LeakyReLU(0.01)
```

### Block 2: Inception Module (Figure 4)
Input to inception: (batch, 16, 100, 1) → squeeze to (batch, 16, 100)

Three parallel paths:
```
Path A: Conv1x1@32 → Conv3x1@32 → output (batch, 32, 100)
Path B: Conv1x1@32 → Conv5x1@32 → output (batch, 32, 100)
Path C: MaxPool3x1(stride=1, pad=1) → Conv1x1@32 → output (batch, 32, 100)

Concatenate → (batch, 96, 100)
```

### Block 3: LSTM
```
Input: (100, batch, 96)  # time_steps × batch × features
LSTM: 64 hidden units, 1 layer
Output: take LAST timestep → (batch, 64)
```

### Block 4: Output
```
FC: 64 → 3 (softmax)
Classes: {down: 0, stationary: 1, up: 2}
```

## Training Hyperparameters
- Loss: Categorical cross-entropy
- Optimizer: Adam (lr=0.01, epsilon=1)
- Batch size: 32 (paper) or 64 (official code)
- Early stopping: patience=20 epochs on validation accuracy
- ~100 epochs for FI-2010, ~40 for LSE
- Activation: LeakyReLU(negative_slope=0.01) everywhere
- No dropout mentioned in paper (but official code may vary)

## FI-2010 Dataset Details
- Source: etsin.fairdata.fi or bundled in official GitHub repo
- 5 Finnish stocks, 10 trading days, Nasdaq Nordic
- Pre-normalized (z-score, min-max, decimal precision available)
- 10 LOB levels × 4 features = 40 features per timestamp
- ~4 million samples total
- Labels: 5 horizons (k = 10, 20, 30, 50, 100)
- Label method: smoothed mid-price direction with threshold α

### Setup 2 (our primary benchmark):
- Train: Days 1-7
- Test: Days 8-10
- No separate validation in original — we add early stopping on day 7

## Target Results to Reproduce (Table II, Setup 2)

| Horizon | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| k=10 | 84.47% | 84.00% | 84.47% | 83.40% |
| k=20 | 74.85% | 74.06% | 74.85% | 72.82% |
| k=50 | 80.51% | 80.38% | 80.51% | 80.35% |

**Realistic expectation: within 1-3% of these numbers is a successful replication.**

## Labelling Method
Mid-price: p_t = (p_a^(1)(t) + p_b^(1)(t)) / 2

FI-2010 uses Equation 3 (forward smoothing only):
- m+(t) = mean of next k mid-prices
- l_t = (m+(t) - p_t) / p_t
- If l_t > α → up (+1), if l_t < -α → down (-1), else → stationary (0)

## Key Insight: Why Convolutions Work Here
- First conv (1×2, stride 2): pairs price+volume → learns imbalance-like features
- Second conv (1×2, stride 2): pairs ask+bid → learns micro-price (Eq. 7)
- Large conv (1×10): integrates all levels → full LOB summary
- Inception (3×1, 5×1): multiple temporal scales → like different moving averages
- This is equivalent to learned FIR filters (Eq. 5) with data-adaptive coefficients

## Parameter Count
- DeepLOB: ~60K parameters (vs CNN-I's 768K)
- LSTM saves 10× parameters vs fully-connected layer
- Forward pass: 0.253ms (feasible for HFT)
