# Experiment Log

> All training runs and their results. Every experiment MUST have a W&B run link.

## Naming Convention
`exp{NNN}_{model}_{horizon}_{description}`
Example: `exp001_deeplob_k10_baseline`

## Results Table

| Exp ID | Model | Horizon | Acc% | F1% | W&B Link | Notes |
|--------|-------|---------|------|-----|----------|-------|
| paper | DeepLOB | k=10 | 84.47 | 83.40 | N/A | Paper Table II target |
| paper | DeepLOB | k=20 | 77.76 | 72.82 | N/A | Paper Table II target |
| paper | DeepLOB | k=30 | 79.46 | — | N/A | Paper Table II target |
| paper | DeepLOB | k=50 | 82.18 | 80.35 | N/A | Paper Table II target |
| paper | DeepLOB | k=100 | 84.44 | — | N/A | Paper Table II target |
| exp001 | DeepLOB | k=10 | 81.88 | — | N/A | Amarel HPC, gap -2.59% |
| exp002 | DeepLOB | k=20 | 73.77 | — | N/A | Amarel HPC, gap -3.99% |
| exp003 | DeepLOB | k=30 | 76.14 | — | N/A | Amarel HPC, gap -3.32% |
| exp004 | DeepLOB | k=50 | 77.34 | — | N/A | Amarel HPC, gap -4.84% |
| exp005 | DeepLOB | k=100 | 78.19 | — | N/A | Amarel HPC, gap -6.25% |
| exp006 | LogReg | k=10 | 72.20 | 62.79 | N/A | saga solver, max_iter=200, gap -9.68% vs DeepLOB |
| exp007 | LogReg | k=20 | 63.67 | 54.56 | N/A | gap -10.10% |
| exp008 | LogReg | k=30 | 57.90 | 53.63 | N/A | gap -18.24% |
| exp009 | LogReg | k=50 | 50.23 | 50.40 | N/A | gap -27.11% |
| exp010 | LogReg | k=100 | 47.62 | 40.58 | N/A | gap -30.57% |
| exp011 | MLP | k=10 | 70.73 | 58.90 | N/A | 128→64, BN+StandardScaler, collapsed to majority class |
| exp012 | MLP | k=20 | 62.79 | 51.53 | N/A | collapsed to majority class |
| exp013 | MLP | k=30 | 56.75 | 55.11 | N/A | gap -19.39% |
| exp014 | MLP | k=50 | 51.16 | 51.56 | N/A | gap -26.18% |
| exp015 | MLP | k=100 | 52.18 | 51.01 | N/A | gap -26.01% |
| exp016 | XGBoost | k=10 | 80.16 | 77.45 | N/A | 100 trees, depth=6, hist, gap -1.72% vs DeepLOB |
| exp017 | XGBoost | k=20 | 71.99 | 68.20 | N/A | gap -1.78% |
| exp018 | XGBoost | k=30 | 72.89 | 71.12 | N/A | gap -3.25% |
| exp019 | XGBoost | k=50 | 71.82 | 71.38 | N/A | gap -5.52% |
| exp020 | XGBoost | k=100 | 66.53 | 66.54 | N/A | gap -11.66% |
| exp021 | DL-Attention | k=10 | 82.44 | 81.07 | N/A | CNN+Inc+Transformer, 597K params, best_epoch=22, 42 epochs |
| exp022 | DL-Attention | k=20 | 71.49 | 67.66 | N/A | best_epoch=33, 53 epochs, WORSE than DeepLOB (-2.28pp) |
| exp023 | DL-Attention | k=30 | 76.93 | 76.32 | N/A | best_epoch=15, 35 epochs |
| exp024 | DL-Attention | k=50 | 79.21 | 79.11 | N/A | best_epoch=11, 31 epochs, first run hung on Quadro M6000, resubmitted |
| exp025 | DL-Attention | k=100 | 78.77 | 78.82 | N/A | best_epoch=8, 28 epochs |
| exp026 | CNN-Only | k=10 | 83.53 | 82.21 | N/A | Ablation: CNN blocks only, ~1.2K params, best_epoch=20, 40 epochs |
| exp027 | CNN-Only | k=20 | 74.96 | 72.74 | N/A | best_epoch=18, 38 epochs |
| exp028 | CNN-Only | k=30 | 76.30 | 75.84 | N/A | best_epoch=19, 39 epochs |
| exp029 | CNN-Only | k=50 | 78.14 | 78.12 | N/A | best_epoch=13, 33 epochs |
| exp030 | CNN-Only | k=100 | 77.71 | 77.78 | N/A | best_epoch=9, 29 epochs |
| exp031 | CNN+Attention | k=10 | 83.84 | 82.81 | N/A | Ablation: CNN+Transformer(d=32), ~67K params, best_epoch=20, 40 epochs |
| exp032 | CNN+Attention | k=20 | 75.22 | 73.51 | N/A | best_epoch=11, 31 epochs |
| exp033 | CNN+Attention | k=30 | 77.46 | 76.87 | N/A | best_epoch=17, 37 epochs |
| exp034 | CNN+Attention | k=50 | 78.97 | 78.86 | N/A | best_epoch=14, 34 epochs |
| exp035 | CNN+Attention | k=100 | 79.41 | 79.47 | N/A | best_epoch=10, 30 epochs |
| | | | | | | |

## Hyperparameter Tracker

| Exp ID | LR | Batch | Epochs | Optimizer | Notes |
|--------|-----|-------|--------|-----------|-------|
| paper | 0.01 | 32 | ~100 | Adam(eps=1) | Paper settings |
| exp001-005 | 0.01 | 64 | 200 (ES@20) | Adam(eps=1) | Amarel HPC, all 5 horizons |
| exp006-010 | — | — | 200 (saga) | LBFGS-like | LogReg, flattened 4000 features |
| exp011-015 | 1e-3 | 256 | 150 (ES@20) | Adam | MLP 128→64, BN, StandardScaler, ReduceLROnPlateau |
| exp016-020 | 0.1 | — | 100 trees | XGBoost | max_depth=6, hist method |
| exp021-025 | 1e-3 | 64 | 200 (ES@20) | Adam(eps=1e-8) | DL-Attention, 5-epoch warmup, dropout=0.1, 2 enc layers |
| exp026-030 | 1e-3 | 64 | 200 (ES@20) | Adam(eps=1e-8) | CNN-Only ablation, no warmup, ~1.2K params |
| exp031-035 | 1e-3 | 64 | 200 (ES@20) | Adam(eps=1e-8) | CNN+Attention ablation, 5-epoch warmup, d=32, ~67K params |
| | | | | | |

## Key Findings

### DeepLOB (Phase 3)
- All 5 horizons trained on Amarel HPC (2026-03-21), early stopping with patience=20
- Results are 2-6% below paper targets, which is expected because:
  - We use a validation split from training data (paper uses all 7 days for training)
  - Official code specs (32 filters, BatchNorm, Tanh in block 2) vs paper specs (16 filters, no BN)
  - Batch size 64 (official code) vs 32 (paper)
- k=10 closest to target (-2.59%), k=100 largest gap (-6.25%)

### Baselines (Phase 4)
- **XGBoost** is the strongest baseline, nearly matching DeepLOB at k=10 (80.16% vs 81.88%)
- DeepLOB's advantage grows with horizon: +1.72pp at k=10 → +11.66pp at k=100
- **LogReg** degrades to near-random at longer horizons (47.62% at k=100) — linear model cannot capture non-linear LOB dynamics
- **MLP** collapsed to majority-class prediction for k=10-20 despite tuning (StandardScaler, BatchNorm, LR scheduling) — 4000-dim flattened input too challenging without convolutional inductive bias
- DeepLOB's temporal modeling (LSTM) is most valuable at longer horizons where sequential patterns matter

### DeepLOB-Attention (Phase 5) — COMPLETE (2026-03-22)
- Extension: Replace LSTM with 2× TransformerEncoderLayer (d=192, 4 heads, ff=256)
- CNN+Inception blocks reused identically from DeepLOB
- ~597K params (vs 144K) — 4.1x more parameters
- LR=0.001 (lower than DeepLOB's 0.01), standard Adam epsilon, 5-epoch warmup
- **Results:** Wins on 4/5 horizons, but by small margins (+0.56 to +1.87pp)
  - k=10: 82.44% (+0.56), k=20: 71.49% (-2.28), k=30: 76.93% (+0.79), k=50: 79.21% (+1.87), k=100: 78.77% (+0.58)
  - Average: 77.77% vs DeepLOB 77.46% — only +0.31pp with 4.1x more parameters
- **Key finding:** Temporal model choice (LSTM vs Attention) matters less than the CNN+Inception feature extraction
- **Issues encountered:** numpy/pandas binary incompatibility on Amarel (fixed with version pinning), k=50 job hung on Quadro M6000 GPU (resubmitted to different node)

### Ablation Study (Phase 5 — Complete) — 2026-03-23
- **CNN-Only** (~1.2K params): CNN blocks 1-3 → Global Avg Pool → FC. No Inception, no temporal modeling.
- **CNN+Attention** (~67K params): CNN blocks 1-3 → Transformer(d=32, 4 heads, 2 layers) → FC. No Inception.
- Trained on Amarel HPC, 10 SLURM jobs (5 horizons × 2 models)
- **Issues:** Quadro M6000 GPUs on gpu006 caused hangs; stdout buffering on SLURM made jobs appear stuck when they were actually training — check experiment directories (best_model.pt timestamps) instead of log files

**Results — Average accuracy across 5 horizons:**

| Model | Params | Avg Acc | vs DeepLOB |
|-------|--------|---------|------------|
| CNN-Only | ~1.2K | 78.13% | +0.67pp |
| CNN+Attention | ~67K | 78.98% | +1.52pp |
| DeepLOB | ~144K | 77.46% | baseline |
| DL-Attention | ~597K | 77.77% | +0.31pp |

**Key findings:**
1. **CNN-Only beats full DeepLOB on 4/5 horizons** — the CNN spatial features alone capture most of the predictive signal
2. **CNN+Attention is the best model overall** (78.98% avg) — attention on raw CNN features (d=32) works better than Inception→LSTM or Inception→Attention
3. **Inception adds complexity without consistent benefit** — all models with Inception perform worse than their non-Inception counterparts
4. **More parameters ≠ better accuracy on FI-2010** — CNN-Only is ~500x more parameter-efficient than DL-Attention
5. **Caveat:** CNN-Only/CNN+Attention used LR=0.001 while DeepLOB used LR=0.01 (paper settings). Single seed (42) for all.

### Cost-Aware Training (Phase 6e — Complete) — 2026-03-23
- 12 experiments on Amarel HPC (RTX 3090 / A100), all at k=10, seed=42
- 4 loss types: Weighted CE, Focal Loss, Turnover-Penalized CE, Differentiable Sharpe Ratio

| Exp ID | Loss | Param | Acc% | Naive PnL | Trades | h=200 PnL | Best Epoch |
|--------|------|-------|------|-----------|--------|-----------|------------|
| exp036 | weighted_ce | w=2.0 | 82.65 | -7.7606 | 35,951 | -0.3588 | 26 |
| exp037 | weighted_ce | w=3.0 | 82.11 | -7.2484 | 34,509 | -0.1769 | 40 |
| exp038 | weighted_ce | w=5.0 | 81.80 | -7.8757 | 36,906 | -1.1631 | 89 |
| exp039 | focal | γ=1.0 | 82.40 | -8.6177 | 38,174 | -0.6137 | 22 |
| exp040 | focal | γ=2.0 | 82.18 | -8.4260 | 37,110 | +0.1494 | 26 |
| exp041 | focal | γ=3.0 | 81.54 | -7.8566 | 36,830 | +0.0242 | 27 |
| exp042 | turnover | λ=0.01 | 73.10 | -3.1351 | 9,078 | -0.0006 | 6 |
| exp043 | turnover | λ=0.1 | 72.22 | -4.1021 | 16,322 | -0.4701 | 13 |
| exp044 | turnover | λ=0.5 | 71.81 | -2.9982 | 9,998 | -0.3267 | 6 |
| exp045 | sharpe | lt=0.01,gc=0.5 | 40.51 | -5.7044 | 27,544 | -0.6530 | 13 |
| exp046 | sharpe | lt=0.1,gc=0.5 | 40.98 | -3.0328 | 16,442 | -0.5701 | 6 |
| exp047 | sharpe | lt=0.5,gc=0.5 | 41.34 | -4.8862 | 28,135 | +0.6214 | 6 |

**Baseline comparison:** DeepLOB (CE) = 81.88% acc, -8.02 naive PnL, 37,116 trades

**Key findings:**
1. **No cost-aware model beats LR-Stack h=200 (+0.6880)** — the post-hoc two-stage approach remains best
2. **Turnover penalty is most effective at reducing naive losses** — Turn λ=0.5 improved from -8.02 to -2.99 PnL (63% reduction) with 73% fewer trades (9,998 vs 37,116)
3. **Sharpe lt=0.5 is closest to the champion** — +0.6214 with h=200, just 0.0666 behind LR-Stack
4. **Focal γ=2 is the only classification-preserving method that goes positive** — +0.1494 with h=200, maintaining 82.18% accuracy
5. **Weighted CE doesn't help** — all 3 variants lose money even with h=200. Blunt class weighting is too crude.
6. **Sharpe models sacrifice accuracy dramatically** (~41%) for trading objective — accuracy and profitability remain fundamentally decoupled
7. **Turnover models sacrifice accuracy moderately** (~72%) but achieve the best naive trade reduction
8. **w=5.0 took 89 epochs** — heavy stationary weighting delays convergence without improving outcomes
9. **Central insight confirmed:** The accuracy-profitability gap is NOT fully addressable through loss function design alone. Post-hoc trade reduction (holding period) remains essential.

### Cost-Aware Training — Multi-Horizon (Phase 6e — Complete) — 2026-03-23
- 48 additional experiments on Amarel HPC across k=20,30,50,100 (12 configs × 4 horizons)
- 3 jobs failed due to transient GPU issues (gpuk008 OOM, gpuk011 driver error); resubmitted successfully
- 1 job preempted on gpu017; resubmitted successfully
- All 48 completed. Total: 60 cost-aware experiments (12 k=10 + 48 multi-horizon)

**Results — Test Accuracy (%) by horizon:**

| Loss | Param | k=10 | k=20 | k=30 | k=50 | k=100 | Avg |
|------|-------|------|------|------|------|-------|-----|
| DeepLOB (CE) | baseline | 81.88 | 73.77 | 76.14 | 77.34 | 78.19 | 77.46 |
| weighted_ce | w=2.0 | 82.65 | 73.77 | 75.64 | 76.90 | 76.71 | 77.13 |
| weighted_ce | w=3.0 | 82.11 | 73.52 | 75.67 | 76.16 | 75.87 | 76.67 |
| weighted_ce | w=5.0 | 81.80 | 73.11 | 74.56 | 75.22 | 73.70 | 75.68 |
| focal | γ=1.0 | 82.40 | 73.77 | 75.46 | 77.63 | 76.71 | 77.19 |
| focal | γ=2.0 | 82.18 | 73.13 | 75.72 | 76.36 | 76.10 | 76.70 |
| focal | γ=3.0 | 81.54 | 72.59 | 74.84 | 74.85 | 74.69 | 75.70 |
| turnover | λ=0.01 | 73.10 | 35.39 | 55.04 | 43.48 | 41.26 | 49.65 |
| turnover | λ=0.1 | 72.22 | 59.94 | 55.09 | 46.75 | 45.85 | 55.97 |
| turnover | λ=0.5 | 71.81 | 61.30 | 39.14 | 42.11 | 45.29 | 51.93 |
| sharpe | lt=0.01 | 40.51 | 18.26 | 21.19 | 30.69 | 34.07 | 28.94 |
| sharpe | lt=0.1 | 40.98 | 62.36 | 21.85 | 30.25 | 34.35 | 37.96 |
| sharpe | lt=0.5 | 41.34 | 51.53 | 21.01 | 30.50 | 34.02 | 35.68 |

**Key findings (multi-horizon):**
1. **Focal & Weighted CE are robust across horizons** — average accuracy (76-77%) matches or slightly trails DeepLOB CE baseline (77.46%), confirming these are safe classification-preserving alternatives
2. **Turnover loss collapses on non-k=10 horizons** — from ~72% at k=10 to 35-61% at k=20-100. The sequential penalty overfits to k=10 label dynamics
3. **Sharpe loss collapses even worse** — 18-62% accuracy on k=20-100 (vs already-poor 41% at k=10). Warm-starting from k-specific DeepLOB checkpoints doesn't help
4. **Best config per horizon:** k=10: weighted_ce w=2.0 (82.65%), k=20: weighted_ce w=2.0 / focal γ=1.0 (73.77%), k=30: focal γ=2.0 (75.72%), k=50: focal γ=1.0 (77.63%), k=100: focal γ=1.0 / weighted_ce w=2.0 (76.71%)
5. **No cost-aware loss consistently beats standard CE** — the DeepLOB CE baseline (77.46% avg) is within noise of the best cost-aware methods (focal γ=1.0 at 77.19%)
6. **Stronger cost-aware parameters degrade more** — higher γ (focal), higher w (WCE), and stronger λ (turnover) all reduce accuracy, especially at longer horizons
7. **Trading-aware losses (Sharpe, turnover) are fragile** — they learn k=10-specific patterns that don't transfer. Classification-focused losses (focal, WCE) remain robust because they optimize the same universal objective
8. **Central conclusion:** Cost-aware training through loss function modification cannot solve the accuracy-profitability gap. The gap is structural (arising from transaction costs and trade frequency), not a training signal problem
