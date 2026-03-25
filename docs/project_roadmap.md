# Project Roadmap

## Status Legend: ⬜ Not Started | 🟡 In Progress | ✅ Done | ❌ Blocked

---

## Phase 1: Data & Environment Setup [✅]
- [x] Initialize git repo, push to GitHub
- [x] Set up Python 3.13 virtual environment
- [x] Install dependencies (requirements.txt)
- [x] Download FI-2010 dataset
- [x] Verify dataset: shapes, value ranges, label distributions
- [x] Build `FI2010Dataset` class (PyTorch Dataset)
- [x] Build DataLoader with proper chronological splits (Setup 2: days 1-7 train, 8-10 test)
- [x] Create data exploration notebook with visualizations
- [ ] Set up W&B project
- [ ] Verify all Makefile targets work

## Phase 2: Model Implementation [✅]
- [x] Implement CNN feature extraction block (9 conv layers)
- [x] Implement Inception Module (3 parallel paths + concat)
- [x] Implement LSTM block (64 units)
- [x] Implement full DeepLOB model class
- [x] Write shape assertion tests (input→output at each block)
- [x] Verify parameter count (~144K official code spec)
- [x] Test forward pass with random data

## Phase 3: Training & Reproduction [✅]
- [x] Implement training loop with early stopping
- [x] Train on k=10 horizon first (easiest to debug)
- [x] Compare to paper Table II numbers
- [x] Train all 5 horizons (k=10, 20, 30, 50, 100) — Amarel HPC, 2026-03-21
- [x] Generate confusion matrices — notebooks/02_evaluation.ipynb
- [x] Generate classification reports (precision, recall, F1) — notebooks/02_evaluation.ipynb
- [ ] Log everything to W&B

## Phase 4: Baselines [✅]
- [x] Logistic Regression baseline — saga solver, all 5 horizons
- [x] MLP (2-layer) baseline — 128→64, BatchNorm, StandardScaler, ReduceLROnPlateau
- [x] XGBoost baseline — 100 trees, depth=6, hist method, all 5 horizons
- [x] Comparison table: all models × all horizons — notebooks/03_baselines.ipynb
- [x] Accuracy comparison bar chart + DeepLOB advantage heatmap
- [x] Results saved to experiments/baselines/

## Phase 5: Extension [✅]
- [x] Decision: DeepLOB-Attention — Transformer encoder replaces LSTM (ADR-003)
- [x] Implement extension architecture — `src/models/extension.py` (DeepLOBAttention, CNN-Only, CNN+Attention)
- [x] Training script + SLURM jobs — `scripts/train_extension.py`, `scripts/slurm_train_extension*.sh`
- [x] Config — `configs/extension_fi2010.yaml` (LR=0.001, warmup=5, 2 enc layers)
- [x] Analysis notebook — `notebooks/04_extension.ipynb` (loads saved .pt files, comparison + ablation)
- [x] Train all 5 horizons on Amarel HPC — DL-Attention (exp021-025)
- [x] Ablation training — CNN-Only (exp026-030), CNN+Attention (exp031-035), 10 SLURM jobs
- [x] scp results to `experiments/` locally, verified all 35 experiments
- [x] Notebook polished: 10 sections, ablation analysis, bar charts, discussion with caveats
- [ ] Additional dataset test (Bitcoin LOB or other) — deferred to future work

## Phase 6: Realistic Backtest [✅]
- [x] Implement signal→trade logic (src/backtest.py)
- [x] Add transaction costs (spread-based, half-spread per leg)
- [x] Calculate: Sharpe ratio, max drawdown, cumulative PnL
- [x] Compare models on financial metrics (not just F1) — notebooks/05_backtest.ipynb
- [x] Post-hoc strategies: confidence gating, holding period, ensemble, stacking
- [x] Temperature scaling and calibration analysis
- [x] Cross-horizon consensus strategies

## Phase 6e: Cost-Aware Training [✅]
- [x] Implement 4 cost-aware loss functions (src/losses/cost_aware.py)
- [x] Train 12 experiments at k=10 (exp036-047 on Amarel HPC)
- [x] Train 48 multi-horizon experiments (k=20,30,50,100 × 12 configs)
- [x] Notebook analysis: notebooks/06_cost_aware.ipynb (heatmaps, line charts, takeaways)
- [x] Central conclusion: accuracy-profitability gap is structural, not fixable by loss design

## Phase 7: Write-up & Polish [✅]
- [x] Paper-style report (docs/report_draft.md, ~3,500 words, 7 tables)
- [x] Blog post (docs/blog_post_draft.md, ~2,400 words)
- [x] README with results table and quick-start
- [x] Clean code: docstrings, type hints, remove dead code
- [x] Unit tests passing (49/49)
- [x] PDFs generated (report, blog, readme)
- [ ] Have someone else reproduce from README
- [ ] Dashboard/website (Gemini + Figma, NOT Claude)

## Phase 8: Advanced Trading Strategies [⬜]
- [ ] Research complete: `docs/advanced_trading_research.md` (8 approaches, key papers)
- [ ] Notebook: `notebooks/07_advanced_strategies.ipynb`
- [ ] Ensemble disagreement filter (7-model agreement as confidence signal)
- [ ] Multi-horizon arbitrage (cross-horizon signal taxonomy)
- [ ] Regime detection (HMM/volatility-based trade suppression)
- [ ] Kelly position sizing (calibrated softmax → continuous capital allocation)
- [ ] Meta-labeling (Lopez de Prado: secondary model predicts trade profitability)
- [ ] Combined layered strategy (ensemble + regime + sizing + meta-labeling)
- [ ] Compare all strategies against current best (+0.69 PnL)
- [ ] Update report, blog, README with Phase 8 results

---

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-11 | Use DecPre normalization | Matches official code (ADR-004) |
| 2026-03-21 | Follow official code specs (32 filters, BatchNorm, Tanh) | Reproducibility + extension compatibility (ADR-005) |
| 2026-03-21 | k=10 reproduction: 82.89% test acc (target 84.47%) | Close enough for first pass; may tune later |
| 2026-03-21 | All 5 horizons trained on Amarel HPC | k=10: 81.88%, k=20: 73.77%, k=30: 76.14%, k=50: 77.34%, k=100: 78.19% — within 2-6% of paper |
| 2026-03-22 | Phase 4 baselines complete | XGBoost strongest baseline (80.16% k=10), DeepLOB advantage grows with horizon (+1.7pp k=10 → +11.7pp k=100) |
| 2026-03-22 | Extension: DeepLOB-Attention | Replace LSTM with 2× TransformerEncoder (d=192, 4 heads), ~597K params, ready for Amarel training |

## Blockers
| Issue | Status | Resolution |
|-------|--------|------------|
| | | |
