# DeepLOB Revisited: When the Best Classifier is the Worst Trader

## Abstract

We reproduce and extend DeepLOB, a convolutional neural network for limit order book (LOB) mid-price movement prediction, on the FI-2010 benchmark dataset. Our reproduction achieves accuracies within 2--6 percentage points of the published results across all five prediction horizons. We propose DeepLOB-Attention, which replaces the LSTM temporal module with a Transformer encoder, and conduct a systematic ablation study with two additional variants: CNN-Only and CNN+Attention. A surprising finding emerges: the CNN-Only model (approximately 1,200 parameters) outperforms the full DeepLOB (approximately 144,000 parameters) on four of five horizons, suggesting that the Inception module and LSTM contribute minimally on FI-2010. We then bridge classification accuracy to trading profitability through a realistic backtesting framework with transaction costs. All four deep learning models lose money when trading naively on every prediction, and the best classifier (CNN+Attention, 78.98% average accuracy) produces the worst trading losses. Through 60 cost-aware training experiments spanning four loss functions and five horizons, we demonstrate that the accuracy-profitability gap is structural and cannot be closed by loss function design alone. The most profitable strategy remains a simple post-hoc approach: a logistic regression stacking ensemble with a holding period filter.

## 1. Introduction

Limit order book (LOB) prediction has attracted significant attention at the intersection of deep learning and quantitative finance. The limit order book records all outstanding buy and sell orders for a financial instrument at multiple price levels, providing a high-dimensional, high-frequency view of market microstructure. Predicting the direction of mid-price movement from LOB data is a classification problem with direct implications for algorithmic trading.

Zhang, Zohren, and Roberts (2019) introduced DeepLOB, a convolutional neural network architecture that processes raw LOB data through spatial convolutions, an Inception module for multi-scale temporal feature extraction, and an LSTM for sequence modeling. DeepLOB achieved state-of-the-art results on the FI-2010 benchmark and has since become a standard baseline in the LOB prediction literature.

This work makes four contributions:

1. **Reproduction with honest reporting.** We reproduce DeepLOB on FI-2010 (Setup 2) and report results that fall 2--6 percentage points below the published figures, documenting the gap transparently rather than claiming exact replication.

2. **Architectural extension and ablation.** We replace the LSTM with a Transformer encoder (DeepLOB-Attention) and conduct an ablation study revealing that a CNN-Only variant with approximately 1,200 parameters matches or exceeds the full 144,000-parameter DeepLOB on most horizons.

3. **Backtesting with transaction costs.** We implement a realistic backtesting framework that converts classification predictions to trading positions and evaluates profitability after bid-ask spread costs. We find that all deep learning models lose money when trading on every prediction.

4. **Cost-aware training at scale.** We run 60 experiments across four loss function families (Weighted Cross-Entropy, Focal Loss, Turnover-Penalized CE, and Differentiable Sharpe Ratio) and five prediction horizons, demonstrating that the accuracy-profitability gap is structural rather than a training signal problem.

The remainder of this report is organized as follows. Section 2 reviews related work. Section 3 describes our methods, including model architectures and the backtesting framework. Section 4 presents experimental results. Section 5 discusses key findings. Section 6 concludes.

## 2. Related Work

Since the introduction of DeepLOB (Zhang et al., 2019), which established CNN-LSTM as the dominant architecture for limit order book mid-price prediction on the FI-2010 benchmark, numerous extensions have explored attention-based alternatives. Wallbridge (2020) was the first to apply Transformers to LOB data, reporting substantial gains, though a recent 15-model benchmark study by Prata et al. (2024) found that TransLOB's claimed results could not be reproduced (59% vs. 87% F1), while DeepLOB proved one of the more robust baselines. Axial-LOB (Kisiel & Gorse, 2022) introduced a fully attention-based architecture with axial attention factorization, and TLOB (Berti & Kasneci, 2025) proposed dual spatial-temporal attention with a bias-corrected labeling method, claiming new state-of-the-art F1 scores. Zhang and Zohren (2021) extended DeepLOB to multi-horizon forecasting with encoder-decoder attention. Despite these architectural advances, reproducible gains over DeepLOB remain modest (typically 1--3 percentage points), and Prata et al. (2024) found that all models degrade by approximately 20% F1 when tested on out-of-sample NASDAQ data, raising questions about the generalizability of FI-2010 benchmark improvements.

A growing body of work highlights the disconnect between classification accuracy and trading profitability. Briola et al. (2025) demonstrated that "high forecasting power does not necessarily correspond to actionable trading signals" and that microstructural stock properties (tick size, spread) matter more than model architecture for prediction quality. Kolm et al. (2023) showed that derived order flow features outperform raw LOB states and that the effective forecasting horizon is limited to approximately two average price changes, helping explain why frequent trading erodes profits. On the training side, Moody et al. (1998) proposed the differential Sharpe ratio as a differentiable objective for directly optimizing risk-adjusted returns, a concept we implement in our cost-aware training experiments. However, as our results demonstrate, trading-aware loss functions collapse at longer prediction horizons and fail to consistently outperform post-hoc trade reduction strategies, confirming the structural nature of the accuracy-profitability gap.

## 3. Method

### 3.1 DeepLOB Architecture

The DeepLOB model processes LOB snapshots of shape (batch, 1, 100, 40), where 100 is the lookback window in timesteps and 40 represents 10 price levels times 4 features per level (ask price, ask volume, bid price, bid volume), interleaved by level.

The architecture consists of four stages:

**CNN Feature Extraction (Blocks 1--3).** Three convolutional blocks progressively reduce the spatial dimension of the LOB input. Block 1 uses a (1, 2) kernel with stride (1, 2) to pair price and volume features, halving the feature dimension from 40 to 20. Block 2 applies the same structure to pair ask and bid sides, reducing from 20 to 10. Block 3 uses a (1, 10) kernel to integrate all 10 price levels into a single spatial feature. Each block includes two additional temporal convolutions with (4, 1) kernels that act as learned moving averages. All blocks use BatchNorm; Blocks 1 and 3 use LeakyReLU activation while Block 2 uses Tanh (following the official code implementation rather than the paper specification).

**Inception Module.** Three parallel convolutional paths operate on the CNN output: Path A applies 3x1 convolutions (short-term patterns), Path B applies 5x1 convolutions (medium-term patterns), and Path C applies max pooling followed by 1x1 convolutions (local extrema). Each path produces 64 channels, and their concatenation yields a 192-channel temporal feature sequence.

**LSTM.** A single-layer LSTM with 64 hidden units processes the 106-timestep sequence (the time dimension grows from 100 to 106 due to padding in temporal convolutions). Only the final hidden state is used for classification.

**Fully Connected Layer.** A linear layer maps the 64-dimensional LSTM output to 3 class logits (down, stationary, up). No softmax is applied; PyTorch's CrossEntropyLoss handles this internally.

The model has approximately 144,000 trainable parameters. We follow the official code specification (32 filters per convolution, BatchNorm, Tanh in Block 2) rather than the paper specification (16 filters, no BatchNorm, LeakyReLU throughout), as this is what produced the published results and is the basis for all follow-up work in the literature.

### 3.2 DeepLOB-Attention Extension

DeepLOB-Attention replaces the LSTM with a Transformer encoder while reusing the CNN and Inception blocks identically. The motivation is to compare sequential processing (LSTM) against parallel self-attention for temporal modeling of LOB data.

After the Inception module produces a sequence of 192-dimensional vectors (one per timestep), we add a learnable positional encoding (since self-attention has no inherent notion of order) and pass the sequence through 2 Transformer encoder layers, each with 4 attention heads, a model dimension of 192, and a feedforward dimension of 256. Dropout of 0.1 is applied within the attention and feedforward sublayers. The sequence is then mean-pooled across the temporal dimension and passed to a linear classification layer.

DeepLOB-Attention has approximately 597,000 parameters --- 4.1 times more than DeepLOB --- primarily due to the query, key, and value projection matrices in the attention layers.

**Design rationale.** We use composition rather than inheritance: a DeepLOB instance is created and its CNN/Inception layers are referenced directly. This ensures exact feature extraction parity and avoids carrying unused LSTM/FC layers.

### 3.3 Ablation Variants

To isolate the contribution of each component, we define two ablation architectures:

**CNN-Only** (approximately 1,200 parameters). CNN Blocks 1--3 followed by global average pooling over the temporal dimension and a linear classifier. No Inception module, no temporal modeling. This tests whether the CNN spatial features alone are sufficient for LOB prediction.

**CNN+Attention** (approximately 67,000 parameters). CNN Blocks 1--3 followed by a Transformer encoder operating on the 32-dimensional CNN output (without Inception expansion to 192 dimensions). Uses 4 attention heads, 2 encoder layers, and a feedforward dimension of 128. This tests whether attention can learn temporal patterns directly from CNN features without the Inception module.

### 3.4 Backtesting Framework

Classification accuracy alone does not determine trading profitability. We implement a backtesting framework that converts model predictions to trading positions and evaluates financial performance with realistic transaction costs.

**Signal-to-trade conversion.** Predictions are mapped to positions: Down (class 0) maps to Short (-1), Stationary (class 1) to Flat (0), and Up (class 2) to Long (+1).

**Transaction costs.** Each position change incurs a cost proportional to the bid-ask spread. Specifically, the cost at timestep *t* is:

    cost_t = |pos_t - pos_{t-1}| * spread_t * 0.5

where the factor of 0.5 reflects the half-spread cost per leg of a trade. Going from flat to long costs half the spread; going from short to long costs the full spread (two units of position change).

**PnL computation.** At each timestep, gross PnL equals the current position multiplied by the subsequent price change. Net PnL subtracts transaction costs.

**Metrics.** We report total net PnL, Sharpe ratio (mean of per-step net returns divided by their standard deviation), maximum drawdown (worst peak-to-trough decline in cumulative PnL), number of trades, and win rate.

**Post-hoc strategies.** Beyond the naive approach of trading on every prediction, we implement several strategies to improve profitability:
- *Confidence filtering*: only trade when the maximum softmax probability exceeds a threshold (default 0.7).
- *Holding period*: once a directional signal is generated, hold the position for a minimum number of timesteps before re-evaluating.
- *Ensemble consensus*: average probabilities across models and require minimum agreement for trade signals.
- *Stacking*: train a logistic regression meta-learner on the probability outputs of multiple models.

### 3.5 Cost-Aware Training

To test whether the accuracy-profitability gap can be closed through modified training objectives, we implement four cost-aware loss functions:

**Weighted Cross-Entropy.** Standard CE with increased weight on the stationary class: w = [1.0, w_stat, 1.0]. By making stationary misclassification more costly, the model becomes more conservative about generating directional predictions. We sweep w_stat in {2.0, 3.0, 5.0}.

**Focal Loss** (Lin et al., 2017). Adds a modulating factor (1 - p_t)^gamma to the CE loss, down-weighting easy/confident predictions. This shifts gradient signal from the dominant stationary class to the harder boundary cases. We sweep gamma in {1.0, 2.0, 3.0}.

**Turnover-Penalized CE** (inspired by Lim, Zohren, and Roberts, 2019). Augments CE with a penalty on prediction instability between consecutive timesteps. The penalty is computed on "soft positions" derived from the softmax output: pos_t = P(Up) - P(Down). The loss is L_CE + lambda * mean(|pos_t - pos_{t-1}|). This requires sequential (non-shuffled) data loading. We sweep lambda in {0.01, 0.1, 0.5}.

**Differentiable Sharpe Ratio** (Moody and Saffell, 2001; Wood, Roberts, and Zohren, 2025). Bypasses classification entirely by converting model outputs to continuous positions, simulating returns with transaction costs, and optimizing the Sharpe ratio via backpropagation. We employ the "half-cost trick" (gamma_cost = 0.5), which avoids the inertia trap where full-cost training causes the model to never trade. This method requires warm-starting from a pre-trained CE model and a lower learning rate (0.0001). We sweep lambda_turnover in {0.01, 0.1, 0.5}.

All cost-aware experiments use the DeepLOB architecture for comparability with the Phase 3 baseline.

## 4. Experiments

### 4.1 Dataset

We use the FI-2010 benchmark dataset (Ntakaris et al., 2018), which contains limit order book data for 5 Finnish stocks traded on Nasdaq Nordic over 10 trading days. The dataset provides 10 LOB levels (4 features each: ask price, ask volume, bid price, bid volume) for a total of 40 features per timestep. Labels are defined at 5 prediction horizons (k = 10, 20, 30, 50, 100 events) using the smoothed mid-price direction with a threshold parameter.

We use Setup 2 (anchored): days 1--7 for training and days 8--10 for testing. We carve a validation set from the end of the training period for early stopping. Data is DecPre (decimal precision) normalized, which is a linear scaling that preserves relative price comparisons. All models use a lookback window of 100 timesteps and a batch size of 64.

### 4.2 Reproduction Results

Table 1 compares our DeepLOB reproduction with the published results across all five prediction horizons.

**Table 1: DeepLOB Reproduction — Test Accuracy (%)**

| Horizon | Paper | Ours | Gap |
|---------|-------|------|-----|
| k=10 | 84.47 | 81.88 | -2.59 |
| k=20 | 77.76 | 73.77 | -3.99 |
| k=30 | 79.46 | 76.14 | -3.32 |
| k=50 | 82.18 | 77.34 | -4.84 |
| k=100 | 84.44 | 78.19 | -6.25 |

Our results fall 2--6 percentage points below the published figures. We attribute this gap to three factors: (1) we allocate a portion of the training data for validation-based early stopping, whereas the paper trains on all 7 days; (2) we use the official code specification (32 filters, BatchNorm) rather than the paper specification (16 filters, no BatchNorm); (3) hyperparameter sensitivity with a single random seed (42). The gap is widest at k=100, where the longest prediction horizon amplifies small modeling differences.

Training used Adam with learning rate 0.01 and epsilon=1 (matching the paper), with early stopping at patience 20 on validation accuracy. All experiments were run on Rutgers University's Amarel HPC cluster using NVIDIA RTX 3090 and A100 GPUs.

### 4.3 Baseline Results

We compare DeepLOB against three traditional baselines: Logistic Regression, a 2-layer MLP (128 to 64 hidden units with BatchNorm and StandardScaler), and XGBoost (100 trees, depth 6).

**Table 2: Model Comparison — Test Accuracy (%) by Horizon**

| Model | k=10 | k=20 | k=30 | k=50 | k=100 | Avg |
|-------|------|------|------|------|-------|-----|
| DeepLOB | 81.88 | 73.77 | 76.14 | 77.34 | 78.19 | 77.46 |
| XGBoost | 80.16 | 71.99 | 72.89 | 71.82 | 66.53 | 72.68 |
| LogReg | 72.20 | 63.67 | 57.90 | 50.23 | 47.62 | 58.32 |
| MLP | 70.73 | 62.79 | 56.75 | 51.16 | 52.18 | 58.72 |

XGBoost is the strongest baseline, nearly matching DeepLOB at k=10 (80.16% vs. 81.88%, a gap of 1.72 percentage points). However, DeepLOB's advantage grows substantially with the prediction horizon, reaching 11.66 percentage points at k=100. This confirms the value of convolutional and temporal inductive biases for capturing longer-range LOB dynamics. Logistic Regression degrades to near-random performance at longer horizons (47.62% at k=100), demonstrating that linear models cannot capture the non-linear structure of LOB data. The MLP collapsed to majority-class prediction at short horizons despite tuning, indicating that the 4,000-dimensional flattened input is too challenging without convolutional structure.

### 4.4 Extension and Ablation Results

**Table 3: All Models — Test Accuracy (%) by Horizon**

| Model | Params | k=10 | k=20 | k=30 | k=50 | k=100 | Avg |
|-------|--------|------|------|------|------|-------|-----|
| CNN+Attention | ~67K | 83.84 | 75.22 | 77.46 | 78.97 | 79.41 | **78.98** |
| CNN-Only | ~1.2K | 83.53 | 74.96 | 76.30 | 78.14 | 77.71 | 78.13 |
| DL-Attention | ~597K | 82.44 | 71.49 | 76.93 | 79.21 | 78.77 | 77.77 |
| DeepLOB | ~144K | 81.88 | 73.77 | 76.14 | 77.34 | 78.19 | 77.46 |

[FIGURE: Bar chart comparing model accuracy across horizons, from notebooks/04_extension.ipynb]

Several notable patterns emerge:

1. **CNN+Attention is the best classifier overall** with 78.98% average accuracy, outperforming DeepLOB by 1.52 percentage points while using less than half the parameters.

2. **CNN-Only beats DeepLOB on 4 of 5 horizons** despite having approximately 120 times fewer parameters (1,200 vs. 144,000). It achieves 78.13% average accuracy versus 77.46% for DeepLOB.

3. **DL-Attention provides marginal improvement over DeepLOB** (+0.31 percentage points average) with 4.1 times more parameters. The Transformer encoder is not substantially better than the LSTM for this task.

4. **Inception adds complexity without benefit.** Both models without Inception (CNN-Only, CNN+Attention) outperform both models with Inception (DeepLOB, DL-Attention).

**Caveat.** The CNN-Only and CNN+Attention models used a learning rate of 0.001 (with standard Adam epsilon), while DeepLOB used 0.01 (with epsilon=1) to match the paper's hyperparameters. All results use a single seed (42). These differences limit the strength of architectural conclusions, though the magnitude of the CNN-Only result (beating a 120x-larger model) is difficult to explain by hyperparameter effects alone.

### 4.5 Backtest Results

We evaluate all four deep learning models through our backtesting framework on the k=10 horizon, where predictions are densest and transaction costs most impactful.

**Table 4: Naive Backtest Results (k=10, Trading on Every Prediction)**

| Model | Net PnL | Trades | Sharpe |
|-------|---------|--------|--------|
| DeepLOB | -7.91 | 37,115 | -0.0933 |
| DL-Attention | -8.09 | 38,178 | -0.0922 |
| CNN-Only | -7.99 | 37,126 | -0.1365 |
| CNN+Attention | -8.48 | 37,469 | -0.1222 |

All models lose money when trading naively on every prediction. The core problem is excessive trading: the DeepLOB generates over 37,000 trades on the test set, and each position change incurs the bid-ask spread as a transaction cost. The gross PnL (before costs) is positive for all models, but transaction costs overwhelm the signal.

**Post-hoc strategies.** We test several methods to reduce trading frequency while preserving signal quality.

**Table 5: Best Post-Hoc Strategies (k=10)**

| Strategy | Config | Net PnL | Trades |
|----------|--------|---------|--------|
| LR-Stack | h=200 | **+0.6880** | 1,087 |
| Hold (LogReg) | h=100 | +0.6144 | 1,085 |
| Hold (DeepLOB) | h=200 | +0.4757 | 1,081 |
| Naive DeepLOB | — | -7.9121 | 37,115 |

The best strategy is a stacking meta-learner: a logistic regression model trained on the softmax probability outputs of all four deep learning models, combined with a holding period of 200 timesteps. This achieves a net PnL of +0.6880 with only 1,087 trades --- a 34-fold reduction from the naive approach. Notably, the simpler LogReg baseline with a holding period of 100 (+0.6144) nearly matches the stacking approach, suggesting that the primary driver of profitability is trade reduction, not classifier sophistication.

[FIGURE: Cumulative PnL curves for naive vs. post-hoc strategies, from notebooks/05_backtest.ipynb]

### 4.6 Cost-Aware Training Results

We conduct 60 cost-aware training experiments: 12 configurations (4 loss types times 3 hyperparameter settings) evaluated across all 5 prediction horizons.

**Table 6: Cost-Aware Results at k=10**

| Loss | Param | Acc% | Naive PnL | h=200 PnL |
|------|-------|------|-----------|-----------|
| DeepLOB (CE) | baseline | 81.88 | -8.02 | — |
| weighted_ce | w=2.0 | 82.65 | -7.76 | -0.36 |
| weighted_ce | w=3.0 | 82.11 | -7.25 | -0.18 |
| weighted_ce | w=5.0 | 81.80 | -7.88 | -1.16 |
| focal | gamma=1.0 | 82.40 | -8.62 | -0.61 |
| focal | gamma=2.0 | 82.18 | -8.43 | +0.15 |
| focal | gamma=3.0 | 81.54 | -7.86 | +0.02 |
| turnover | lambda=0.01 | 73.10 | -3.14 | -0.00 |
| turnover | lambda=0.1 | 72.22 | -4.10 | -0.47 |
| turnover | lambda=0.5 | 71.81 | -3.00 | -0.33 |
| sharpe | lt=0.01,gc=0.5 | 40.51 | -5.70 | -0.65 |
| sharpe | lt=0.1,gc=0.5 | 40.98 | -3.03 | -0.57 |
| sharpe | lt=0.5,gc=0.5 | 41.34 | -4.89 | +0.62 |

**Table 7: Multi-Horizon Robustness — Average Accuracy (%) Across All Horizons**

| Loss | Param | k=10 | k=20 | k=30 | k=50 | k=100 | Avg |
|------|-------|------|------|------|------|-------|-----|
| DeepLOB (CE) | baseline | 81.88 | 73.77 | 76.14 | 77.34 | 78.19 | 77.46 |
| weighted_ce | w=2.0 | 82.65 | 73.77 | 75.64 | 76.90 | 76.71 | 77.13 |
| focal | gamma=1.0 | 82.40 | 73.77 | 75.46 | 77.63 | 76.71 | 77.19 |
| focal | gamma=2.0 | 82.18 | 73.13 | 75.72 | 76.36 | 76.10 | 76.70 |
| turnover | lambda=0.01 | 73.10 | 35.39 | 55.04 | 43.48 | 41.26 | 49.65 |
| turnover | lambda=0.1 | 72.22 | 59.94 | 55.09 | 46.75 | 45.85 | 55.97 |
| sharpe | lt=0.01 | 40.51 | 18.26 | 21.19 | 30.69 | 34.07 | 28.94 |
| sharpe | lt=0.5 | 41.34 | 51.53 | 21.01 | 30.50 | 34.02 | 35.68 |

[FIGURE: Heatmap of 13 configs times 5 horizons accuracy, from notebooks/06_cost_aware.ipynb]

Key findings from the cost-aware experiments:

1. **No cost-aware model beats the LR-Stack champion** (+0.6880 PnL). The closest is Sharpe lt=0.5 with h=200 at +0.6214, which sacrifices accuracy to 41.34%.

2. **Focal and Weighted CE are robust across horizons** (76--77% average accuracy), closely tracking the CE baseline. These are safe, classification-preserving alternatives, but they do not improve profitability.

3. **Turnover loss collapses on longer horizons.** From approximately 72% at k=10, accuracy drops to 35--61% at k=20--100. The sequential penalty overfits to k=10 label dynamics and does not transfer.

4. **Sharpe loss collapses even more severely** (18--62% on k=20--100). Even warm-starting from horizon-specific DeepLOB checkpoints does not prevent this collapse.

5. **Turnover penalty is most effective at reducing naive losses.** Turnover lambda=0.5 reduces the trade count from 37,116 to 9,998 (73% reduction) and improves naive PnL from -8.02 to -3.00 (63% loss reduction). However, this still loses money.

6. **Stronger cost-aware parameters degrade more.** Higher gamma (focal), higher w (weighted CE), and stronger lambda (turnover) all reduce accuracy, especially at longer horizons.

### 4.7 Advanced Trading Strategies (Phase 8)

Building on the finding that the accuracy-profitability gap is structural, we implement five post-hoc strategies that exploit the existing model predictions more intelligently, without retraining. All strategies use pre-computed softmax probabilities from 6 models (4 DL + 2 baselines) across 5 horizons.

**Strategy 1: Ensemble Disagreement.** We treat agreement across 6 structurally diverse models (spanning CNN, LSTM, Transformer, logistic regression, and gradient-boosted trees) as a confidence signal. When all 6 models agree on a class, accuracy is 88.8% (vs. 83.8% overall), and 72.9% of samples achieve this unanimous consensus. Gating trades to require agreement of at least 6/6 models with a holding period of 200 yields +0.6825 PnL.

**Strategy 2: Multi-Horizon Arbitrage.** Predictions at 5 horizons (k=10,20,30,50,100) form a "term structure" of directional expectations. When all horizons agree on direction, the signal is stronger. Combining model agreement (>=6/6) with cross-horizon agreement (>=4/5 horizons) produces a "double filter" achieving +0.6798 PnL.

**Strategy 3: Regime Detection.** We classify market regimes using rolling volatility (low/medium/high terciles) and Gaussian HMMs. The volatility filter is more informative than the HMM, which assigns 99.8% of samples to a single state. Trading only in medium+high volatility periods with a holding period of 100 achieves +0.4708 PnL.

**Strategy 4: Kelly Position Sizing.** Instead of binary positions {-1,0,+1}, we use the Kelly criterion with P(Up)-P(Down) as a continuous edge signal. Quarter-Kelly with h=200 achieves only +0.0373 PnL, substantially underperforming discrete strategies. The continuous positions generate excessive position-change costs without sufficient edge to compensate.

**Strategy 5: Meta-Labeling (Lopez de Prado, 2018).** A secondary GradientBoosting classifier predicts whether each directional trade will be profitable after costs. Using walk-forward validation (60/40 split of the test set), the meta-labeler achieves AUC=0.83 on the evaluation period. The top features are spread (28.2%), volume imbalance (11.1%), and mid-return (8.3%). On the evaluation period, meta-label gating with threshold 0.55 and h=100 achieves +0.3963 PnL versus +0.1349 for the ensemble baseline.

**Strategy 6: Combined Layered Strategy.** Layering all filters --- ensemble agreement >= 6, cross-horizon agreement >= 3, low+medium volatility regime, holding period 200 --- achieves the highest overall PnL of **+0.7063** with only 461 trades. This marginally beats the Phase 6 champion (LR-Stack h=200 at +0.6880) while using fewer trades and achieving a higher Sharpe ratio (0.0047 vs. 0.0037).

**Table 8: Phase 8 Strategy Comparison (k=10)**

| Strategy | Net PnL | Sharpe | Trades |
|----------|---------|--------|--------|
| Combined: E>=6 H>=3 V=L+M h=200 | **+0.7063** | **0.0047** | **461** |
| S1: agree>=6 h=200 | +0.6825 | 0.0045 | 631 |
| S2: E>=6+H>=4 h=200 | +0.6798 | 0.0045 | 625 |
| Phase 6 Best: LR-Stack h=200 | +0.6880 | 0.0037 | 1,087 |
| S3: Vol M+H h=100 | +0.4708 | 0.0026 | 1,648 |
| S4: Quarter-Kelly h=200 | +0.0373 | 0.0014 | 1,003 |

## 5. Discussion

### 5.1 The Accuracy-Profitability Gap

The central finding of this work is the disconnect between classification accuracy and trading profitability. CNN+Attention achieves the highest accuracy (78.98%) but, like all deep learning models tested, loses money when trading naively. The best trading strategy (LR-Stack with holding period, +0.6880 PnL) uses a simple logistic regression meta-learner, not the most accurate classifier.

This gap has three causes. First, cross-entropy loss treats all misclassifications equally, but in trading, predicting "up" when the price drops costs money, while predicting "stationary" when the price rises merely misses an opportunity at zero cost. Second, many FI-2010 labels correspond to price movements smaller than the bid-ask spread; these are correctly classified but untradable. Third, acting on every prediction at tick frequency generates catastrophic turnover (over 37,000 trades), overwhelming any directional signal with transaction costs.

This finding is consistent with the broader literature. Prata et al. (2024) tested 15 state-of-the-art LOB prediction models and concluded that profitability is "far from guaranteed." Berti and Kasneci (2025) achieved 92.8% F1 on FI-2010 with TLOB but found that "performance deterioration underscores the complexity of translating trend classification into profitable trading strategies."

### 5.2 Less Is More: Parameter Efficiency on FI-2010

The ablation results challenge the assumption that more complex models produce better predictions. The CNN-Only model (approximately 1,200 parameters) outperforms the full DeepLOB (approximately 144,000 parameters) on 4 of 5 horizons. This implies that:

- The three CNN blocks already capture the essential spatial structure of the LOB (price-volume pairing, ask-bid pairing, and level integration).
- The Inception module's multi-scale temporal convolutions add complexity without consistently improving predictions.
- The LSTM's sequential modeling provides negligible benefit when global average pooling of CNN features suffices.

This result is specific to FI-2010, which contains 10 highly correlated Finnish stocks with relatively low-frequency updates. On noisier or higher-frequency datasets, temporal modeling may prove more valuable. Nevertheless, the finding underscores the importance of ablation studies: without testing the CNN-Only variant, one might incorrectly attribute DeepLOB's performance to its Inception and LSTM components.

### 5.3 Cost-Aware Training Does Not Solve It

Sixty cost-aware training experiments across four loss function families and five horizons provide strong evidence that the accuracy-profitability gap cannot be closed through loss function modification alone.

Classification-preserving losses (Focal Loss, Weighted CE) maintain accuracy at 76--77% average but do not make the models profitable. They are safe alternatives to standard CE but address the wrong problem: the issue is not class imbalance or hard-example learning but rather the structural mismatch between classification objectives and trading objectives.

Trading-aware losses (Differentiable Sharpe, Turnover Penalty) directly optimize financial metrics but collapse catastrophically on horizons beyond k=10. The turnover penalty overfits to the label transition dynamics at k=10 (where labels change frequently) and fails at k=50--100 (where label sequences are smoother). The Sharpe loss, despite warm-starting from pre-trained models, reduces accuracy below 50% on most horizons --- effectively worse than random.

The most telling comparison: Sharpe lt=0.5 achieves +0.6214 PnL at k=10 with the h=200 holding period, nearly matching the LR-Stack champion (+0.6880). But it does so at 41.34% accuracy. The holding period, not the loss function, is doing the work. This confirms that the accuracy-profitability gap is structural, arising from the mismatch between per-sample classification and sequential trading with costs.

### 5.4 Post-Hoc Beats End-to-End

The most effective profitability strategies in our experiments are all post-hoc: applied after model training rather than incorporated into the training objective. The holding period filter alone transforms a -7.91 PnL loss into a +0.48 profit for DeepLOB. The stacking meta-learner adds further improvement by combining the strengths of multiple classifiers. The Phase 8 combined layered strategy extends this further to +0.7063 PnL by stacking three independent filter dimensions (model agreement, cross-horizon consistency, regime conditioning) before applying a holding period.

This suggests a multi-stage approach for LOB trading: (1) train diverse classifiers to maximize accuracy using standard objectives, then (2) combine predictions via ensemble averaging, then (3) apply multiple independent quality filters, and finally (4) reduce trade frequency with holding periods. Each stage operates on a different dimension of the problem: the first exploits model diversity, the second improves signal quality, the third removes context-dependent noise, and the fourth manages transaction costs.

The Phase 8 results reinforce this insight: the combined strategy achieves the highest PnL (+0.7063) with only 461 trades --- an 80-fold reduction from the naive baseline. Notably, continuous position sizing (Kelly criterion) underperforms discrete gating, suggesting that the binary trade/no-trade decision is more important than fine-grained capital allocation on this dataset.

### 5.5 Intellectual Honesty and Negative Results

Several of our results are negative: the reproduction falls below published figures, the Transformer extension barely improves on the LSTM, and 60 cost-aware experiments fail to beat a simple post-hoc strategy. We view these as strengths of the work. Negative results are underreported in machine learning research, creating a publication bias toward architectural novelty. By documenting what does not work --- and quantifying why --- we contribute to a more accurate understanding of the LOB prediction problem.

### Limitations

This work has several limitations that should be noted:

- **Single seed.** All experiments use seed 42. Variance across seeds could change some close comparisons (e.g., DL-Attention vs. DeepLOB at +0.31 percentage points).
- **Learning rate differences.** DeepLOB uses LR=0.01 (paper specification); extension and ablation models use LR=0.001. A grid search over learning rates for all models would strengthen architectural conclusions.
- **FI-2010 specificity.** FI-2010 contains 10 days of data from 5 correlated Finnish stocks with relatively low activity. Results may not generalize to modern, higher-frequency, or multi-asset LOB datasets.
- **Normalized prices.** PnL values are computed on DecPre-normalized prices. While Sharpe ratios and relative comparisons between models are valid, absolute PnL figures are not directly interpretable as monetary values.

## 6. Conclusion

We have presented a comprehensive study of deep learning for limit order book prediction, spanning reproduction, extension, ablation, backtesting, and cost-aware training. Our key findings are:

1. A CNN-Only model with approximately 1,200 parameters matches or exceeds the full DeepLOB (approximately 144,000 parameters) on the FI-2010 benchmark, suggesting that the Inception module and LSTM contribute minimally to this dataset.

2. All deep learning models lose money when trading naively on every prediction, and the best classifier (CNN+Attention, 78.98% accuracy) is the worst trader. The accuracy-profitability gap is fundamental, not incidental.

3. Sixty cost-aware training experiments across four loss functions and five horizons confirm that this gap cannot be closed through loss function design alone. Trading-aware losses (Sharpe, turnover) sacrifice accuracy without achieving profitability; classification-preserving losses (focal, weighted CE) maintain accuracy without improving profitability.

4. The most profitable strategy uses multi-dimensional post-hoc filtering: ensemble agreement across 6 diverse models, cross-horizon consistency, and volatility regime conditioning, combined with a 200-step holding period. This achieves +0.7063 net PnL with only 461 trades, marginally beating the simpler LR-Stack approach (+0.6880) while trading 58% less frequently.

5. Meta-labeling (Lopez de Prado, 2018) shows promise for learning context-dependent trade quality, with a secondary classifier achieving AUC=0.83 on predicting trade profitability. The most informative features are microstructural (spread, volume imbalance) rather than model confidence, suggesting that market state matters more than prediction certainty.

These results point to several directions for future work: (1) multi-seed experiments and hyperparameter-matched comparisons to strengthen architectural conclusions, (2) evaluation on modern LOB datasets (e.g., LOBSTER) with higher frequency and greater diversity, (3) reinforcement learning approaches that optimize trading policies end-to-end rather than through loss function modification, (4) spread-aware relabeling that aligns classification thresholds with transaction costs, and (5) meta-labeling with richer features and walk-forward retraining.

## References

[1] Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. IEEE Transactions on Signal Processing, 67(11), 3001--3012.

[2] Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods. Journal of Forecasting, 37(8), 852--866.

[3] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

[4] Lim, B., Zohren, S., & Roberts, S. (2019). Enhancing time-series momentum strategies using deep neural networks. Journal of Financial Data Science, 1(4), 19--33.

[5] Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement. IEEE Transactions on Neural Networks, 12(4), 875--889.

[6] Wood, M., Roberts, S., & Zohren, S. (2025). DeePM: Deep momentum strategies. arXiv:2601.05975.

[7] Prata, M., et al. (2024). LOBCAST: Limit order book forecasting benchmark. Artificial Intelligence Review.

[8] Berti, L., & Kasneci, G. (2025). TLOB: A Transformer model for limit order book prediction. arXiv:2502.15757.

[9] Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep learning for portfolio optimization. Journal of Financial Data Science, 2(4), 8--20.

[10] Yin, J., & Wong, W. K. (2022). Deep LOB trading with fractional Kelly criterion. Expert Systems with Applications.

[11] Wallbridge, J. (2020). Transformers for limit order books. arXiv:2003.00130.

[12] Kisiel, M., & Gorse, D. (2022). Axial-LOB: High-frequency trading with axial attention. arXiv:2212.01807.

[13] Briola, A., Bartolucci, S., & Aste, T. (2025). Deep limit order book forecasting: A microstructural guide. Quantitative Finance, Taylor & Francis.

[14] Kolm, P. N., Turiel, J., & Westray, N. (2023). Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book. Mathematical Finance.

[15] Moody, J., Wu, L., Liao, Y., & Saffell, M. (1998). Performance functions and reinforcement learning for trading systems and portfolios. Journal of Forecasting.
