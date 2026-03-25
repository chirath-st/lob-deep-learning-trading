# DeepLOB: Reproducing and Extending Deep Learning for Limit Order Books

**A complete reproduction of the DeepLOB paper with Transformer extension, ablation study, and realistic backtesting on FI-2010.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project reproduces [Zhang, Zohren & Roberts (2019)](https://arxiv.org/abs/1808.03668) from scratch, extends it with a Transformer-based architecture, runs a full ablation study across 4 model variants, evaluates all models with a realistic backtesting pipeline that includes transaction costs, and implements advanced post-hoc trading strategies. 95 experiments were trained on an HPC cluster (RTX 3090 / A100 GPUs).

**The central finding: accuracy does not equal profitability.** The best classifier (CNN+Attention, 78.98%) is the *worst* trader. The best trading strategy layers ensemble agreement, cross-horizon consistency, and regime conditioning to achieve +0.71 PnL with only 461 trades.

---

## Key Results

### Model Comparison (Average accuracy across 5 horizons, FI-2010 Setup 2)

| Model | Params | k=10 | k=20 | k=30 | k=50 | k=100 | Avg Acc |
|-------|--------|------|------|------|------|-------|---------|
| **CNN+Attention** | **67K** | **83.84** | **75.22** | **77.46** | **78.97** | **79.41** | **78.98%** |
| CNN-Only | 1.2K | 83.53 | 74.96 | 76.30 | 78.14 | 77.71 | 78.13% |
| DL-Attention | 597K | 82.44 | 71.49 | 76.93 | 79.21 | 78.77 | 77.77% |
| DeepLOB (reproduced) | 144K | 81.88 | 73.77 | 76.14 | 77.34 | 78.19 | 77.46% |
| XGBoost | -- | 80.16 | 71.99 | 72.89 | 71.82 | 66.53 | 72.68% |
| MLP | -- | 70.73 | 62.79 | 56.75 | 51.16 | 52.18 | 58.72% |
| LogReg | -- | 72.20 | 63.67 | 57.90 | 50.23 | 47.62 | 58.32% |

### Backtest Results (k=10 horizon, with transaction costs)

| Strategy | Net PnL | Trades | Profitable? |
|----------|---------|--------|-------------|
| **Combined layered** (E>=6, H>=3, V=L+M, h=200) | **+0.71** | **461** | **Yes** |
| Ensemble agree>=6 h=200 | +0.68 | 631 | Yes |
| LR-Stack (meta-learner), h=200 | +0.69 | 1,087 | Yes |
| LogReg hold, h=100 | +0.61 | 1,085 | Yes |
| DeepLOB hold, h=200 | +0.48 | 1,081 | Yes |
| DeepLOB naive (no filtering) | -7.91 | 37,115 | No |
| CNN+Attention naive | -8.48 | 37,469 | No |

---

## Surprising Findings

1. **A 1,200-parameter CNN beats the full 144K-parameter DeepLOB on 4/5 horizons.** The CNN spatial features alone capture most of the predictive signal in FI-2010. Inception modules and temporal modeling (LSTM/Attention) add complexity without consistent benefit.

2. **All deep learning models lose money after transaction costs.** Every model generates 35,000+ trades on the test set. At realistic spread-based costs, the PnL is deeply negative (-7 to -9). Only aggressive trade filtering (holding periods of 100-200 steps, reducing trades by ~35x) makes any strategy profitable.

3. **The best classifier is the worst trader.** CNN+Attention achieves the highest accuracy (78.98%) but generates the most trades and loses the most money. A simple logistic regression stacking ensemble with holding-period filtering is the most profitable strategy (+0.69 PnL).

4. **Cost-aware training cannot close the gap.** Across 60 experiments with 4 custom loss functions (Focal Loss, Weighted CE, Turnover Penalty, Differentiable Sharpe), no training-time approach consistently beats post-hoc trade filtering. The accuracy-profitability gap is structural, not a training signal problem.

5. **Multi-dimensional filtering is the answer.** Layering ensemble agreement (6/6 models), cross-horizon consistency (3/5 horizons), and regime conditioning (low+medium volatility) with holding periods achieves the best result: +0.71 PnL with only 461 trades --- an 80x reduction from naive trading.

---

## Project Structure

```
deeplob-project/
├── src/
│   ├── models/
│   │   ├── deeplob.py          # Original DeepLOB (CNN + Inception + LSTM)
│   │   └── extension.py        # DL-Attention, CNN-Only, CNN+Attention
│   ├── losses/
│   │   └── cost_aware.py       # Focal, Weighted CE, Turnover, Sharpe losses
│   ├── data/
│   │   └── dataset.py          # FI-2010 Dataset and DataLoader
│   ├── training/
│   │   └── trainer.py          # Training loop with early stopping
│   └── backtest.py             # Signal-to-trade, PnL, transaction costs
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset structure, distributions, LOB visualization
│   ├── 02_evaluation.ipynb         # DeepLOB reproduction results, confusion matrices
│   ├── 03_baselines.ipynb          # LogReg, MLP, XGBoost comparison
│   ├── 04_extension.ipynb          # Transformer extension + ablation study
│   ├── 05_backtest.ipynb           # Full backtesting analysis (19 sections)
│   ├── 06_cost_aware.ipynb         # Cost-aware training experiments (60 results)
│   └── 07_advanced_strategies.ipynb # Advanced post-hoc trading strategies
├── scripts/
│   ├── train.py                    # DeepLOB training
│   ├── train_extension.py          # Extension model training
│   ├── train_baselines.py          # Baseline model training
│   ├── train_cost_aware.py         # Cost-aware training (4 loss types)
│   ├── slurm/                      # SLURM job scripts for HPC
│   └── generate_*.py              # Prediction and probability generation
├── configs/
│   ├── deeplob_fi2010.yaml         # DeepLOB hyperparameters
│   ├── extension_fi2010.yaml       # Transformer extension config
│   ├── ablation_*.yaml             # Ablation study configs
│   └── cost_aware/                 # 4 cost-aware loss configs
├── experiments/                    # All saved results and checkpoints
├── tests/                          # Unit tests (model, data, training)
├── docs/                           # Architecture decisions, experiment log, guides
└── data/                           # FI-2010 dataset (not tracked in git)
```

---

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/<your-username>/deeplob-project.git
cd deeplob-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download FI-2010 dataset
python scripts/download_data.py

# Train DeepLOB on horizon k=10
python scripts/train.py --config configs/deeplob_fi2010.yaml

# Train extension models (Transformer, CNN-Only, CNN+Attention)
python scripts/train_extension.py --config configs/extension_fi2010.yaml

# Run tests
pytest tests/ -v
```

---

## Models

### DeepLOB (Baseline Reproduction)
CNN (3 conv blocks) -> Inception Module -> LSTM (64 units) -> FC.
~144K parameters. Reproduces the original paper within 2-6% of reported accuracy.

### DL-Attention (Extension)
Replaces LSTM with 2-layer Transformer encoder (d=192, 4 heads, ff=256).
~597K parameters. Wins on 4/5 horizons, but only +0.31pp average improvement over DeepLOB -- 4x more parameters for marginal gain.

### CNN-Only (Ablation)
CNN blocks only -> Global Average Pooling -> FC. No Inception, no temporal model.
~1.2K parameters. Beats full DeepLOB on 4/5 horizons, demonstrating that spatial CNN features dominate on FI-2010.

### CNN+Attention (Ablation -- Best Classifier)
CNN blocks -> Transformer (d=32, 4 heads, 2 layers) -> FC. No Inception.
~67K parameters. Best overall accuracy (78.98% average), showing that attention on raw CNN features works better than the full DeepLOB pipeline.

---

## Experiments

**95 total experiments** trained on Rutgers Amarel HPC (RTX 3090 / A100 GPUs):

| Category | Count | Description |
|----------|-------|-------------|
| DeepLOB reproduction | 5 | All 5 horizons (k=10, 20, 30, 50, 100) |
| Baselines | 15 | LogReg + MLP + XGBoost, all 5 horizons |
| DL-Attention | 5 | Transformer extension, all horizons |
| CNN-Only ablation | 5 | Minimal model, all horizons |
| CNN+Attention ablation | 5 | Best classifier, all horizons |
| Cost-aware training | 60 | 4 loss types x 3 params x 5 horizons |

All experiments use seed=42, FI-2010 Setup 2 (days 1-7 train, days 8-10 test), and early stopping with patience=20.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01 -- Data Exploration](notebooks/01_data_exploration.ipynb) | FI-2010 dataset structure, LOB feature visualization, label distributions across horizons |
| [02 -- Evaluation](notebooks/02_evaluation.ipynb) | DeepLOB reproduction results, confusion matrices, per-class metrics, comparison to paper Table II |
| [03 -- Baselines](notebooks/03_baselines.ipynb) | LogReg, MLP, XGBoost comparison, accuracy bar charts, DeepLOB advantage analysis |
| [04 -- Extension](notebooks/04_extension.ipynb) | Transformer extension results, ablation study (CNN-Only, CNN+Attention), parameter efficiency analysis |
| [05 -- Backtest](notebooks/05_backtest.ipynb) | Full backtesting pipeline: naive trading, holding periods, confidence gating, temperature scaling, cross-horizon consensus, stacking meta-learner (19 sections) |
| [06 -- Cost-Aware Training](notebooks/06_cost_aware.ipynb) | 60 cost-aware experiments: Weighted CE, Focal Loss, Turnover Penalty, Differentiable Sharpe, multi-horizon robustness analysis |
| [07 -- Advanced Strategies](notebooks/07_advanced_strategies.ipynb) | Post-hoc trading strategies: ensemble disagreement, multi-horizon arbitrage, regime detection, Kelly sizing, meta-labeling, combined layered strategy |

---

## References

- **DeepLOB Paper:** Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing, 67*(11), 3001-3012. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- **FI-2010 Dataset:** Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods. *Journal of Forecasting, 37*(8), 852-866.
- **Official DeepLOB Code:** [github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

---

## License

MIT
