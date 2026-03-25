# Cost-Aware Training for LOB Prediction: Research Summary

> Synthesized from deep research on methods to bridge the accuracy-profitability gap.
> Date: 2026-03-23 | Phase 6e of the DeepLOB project

---

## The Core Problem

Our Phase 6 backtest revealed a stark disconnect: **the best classifier (CNN+Attention, 78.98% accuracy) is the worst trader.** All 4 DL models lose money when trading naively on every prediction. The root cause is threefold:

1. **Cross-entropy is misaligned with trading objectives.** CE treats all misclassifications equally, but in trading: predicting "up" when price drops = realized loss, predicting "up" when price is flat = unnecessary spread cost, predicting "flat" when price rises = missed opportunity (zero cost). CE has no incentive to distinguish these.

2. **FI-2010 labels include untradable movements.** The labeling threshold (theta=0.002) doesn't account for the bid-ask spread. Many "up"/"down" labels correspond to movements smaller than transaction costs.

3. **Acting on every prediction generates catastrophic turnover.** Our naive DeepLOB produces ~37,000 trades, while all profitable strategies require ~35x trade reduction.

This is not a bug in our implementation — it's the central unsolved problem of the LOB prediction literature. LOBCAST (Prata et al., 2024) tested 15 SOTA models and concluded profitability is "far from guaranteed." Even TLOB (Berti & Kasneci, 2025), achieving 92.8% F1 on FI-2010, explicitly found that "performance deterioration underscores the complexity of translating trend classification into profitable trading strategies."

---

## Methods Ranked by Feasibility for Our Project

Criteria: (a) implementation complexity, (b) published evidence of impact, (c) educational value for learning DL, (d) feasibility on pre-processed FI-2010 data with PyTorch + Amarel HPC.

### Tier 1: Implement These (Selected for Phase 6e)

| Rank | Method | Effort | Impact | Educational Value | Why Selected |
|------|--------|--------|--------|-------------------|--------------|
| 1 | **Weighted Cross-Entropy** | ~10 lines | Moderate | High (loss function basics) | Simplest baseline; shows limits of class weighting |
| 2 | **Focal Loss** | ~40 lines | Moderate | Very High (key DL concept) | Beautiful math; used everywhere in DL |
| 3 | **Turnover-Penalized CE** | ~50 lines | Moderate-High | High (composite losses) | Directly attacks our #1 problem (excessive trades) |
| 4 | **Differentiable Sharpe Ratio** | ~80 lines | Very High | Very High (differentiable finance) | Most radical; bypasses classification entirely |

### Tier 2: Worth Exploring Later

| Rank | Method | Effort | Impact | Why Deferred |
|------|--------|--------|--------|--------------|
| 5 | Selective Prediction (Deep Gamblers) | ~100 lines | High | Adds "abstain" class; needs careful tuning |
| 6 | Hybrid CE + Sharpe Loss | ~60 lines | High | Variant of #4; explore if pure Sharpe fails |
| 7 | Band Turnover Regularization | ~30 lines | Moderate | Refinement of #3; worth trying if #3 works |
| 8 | Frozen Encoder + DQN Agent | ~300 lines + gym env | Very High | Full RL pipeline; major implementation effort |

### Tier 3: Beyond Project Scope

| Method | Why Not Now |
|--------|------------|
| End-to-end Deep Hedging | Requires architectural change, complex reward shaping |
| TLOB/LOBERT architecture | Different architecture entirely; separate project |
| LOBSTER data migration | FI-2010 is our benchmark; consistency matters |
| Spread-aware relabeling | Requires reprocessing FI-2010 labels from scratch |

---

## Detailed Method Descriptions

### Method 1: Weighted Cross-Entropy

**Idea:** Standard CE with per-class weights. Up-weight the stationary class so the model is more conservative about predicting trades.

**How it works:**
- Standard CE: `L = -sum(y_c * log(p_c))` for all classes c
- Weighted CE: `L = -sum(w_c * y_c * log(p_c))` where w_c is the class weight
- We set w = [1.0, w_stat, 1.0] with w_stat > 1 to penalize stationary misclassifications

**Why it helps:** By making stationary misclassification more expensive, the model becomes more reluctant to predict directional moves unless confident. This naturally reduces trade count.

**Sweep:** w_stat in {2.0, 3.0, 5.0}

**Literature:** Khan et al. (2018, IEEE TNNLS) showed 20-40% fewer false signals with cost-sensitive weighting. This is the simplest possible modification.

**Limitations:** Uniform weighting affects ALL samples in a class equally, regardless of how confident the model is. Doesn't distinguish "barely stationary" from "strongly stationary."

### Method 2: Focal Loss

**Idea:** Down-weight easy/confident predictions to focus learning on hard examples.

**The math:**
```
L_FL = -(1 - p_t)^gamma * log(p_t)
```
Where p_t is the probability assigned to the true class, and gamma controls focusing:
- gamma = 0: Standard CE
- gamma = 2: Easy examples (p_t=0.9) down-weighted 100x; hard examples barely affected
- gamma = 5: Very aggressive focusing

**Why it helps for LOB trading:**
1. Stationary class dominates FI-2010 (~50-60% at short horizons). Standard CE spends most gradient signal on these "easy" predictions.
2. Focal loss shifts attention to hard examples — which are often the boundary cases where UP/DOWN meets STATIONARY. Better learning at these boundaries means more reliable directional signals.
3. Acts as an implicit form of class balancing without explicitly specifying weights.

**Sweep:** gamma in {1.0, 2.0, 3.0}

**Literature:** Lin et al. (2017, ICCV) for object detection. Yin & Wong (2022, Expert Systems with Applications) showed focal loss combined with fractional Kelly criterion achieved significant profits after transaction costs on Chinese A-shares.

### Method 3: Turnover-Penalized Composite Loss

**Idea:** CE + explicit penalty for prediction instability between consecutive timesteps.

**The math:**
```
L = L_CE + lambda * L_turnover
L_turnover = mean(|pos_t - pos_{t-1}|)
pos_t = P(Up)_t - P(Down)_t   (soft position in [-1, 1])
```

**Why it works:**
Adjacent LOB snapshots (t and t+1) differ by only one new event. Their information content is ~99% overlapping. So the model's predictions SHOULD be smooth over time. But standard CE has zero incentive for temporal smoothness — it treats each sample independently.

The turnover penalty creates a direct incentive: "if you want to change your prediction, that change better be worth the penalty." This is exactly analogous to transaction costs in real trading.

**Requires:** Sequential (non-shuffled) data loading so consecutive batch samples are consecutive in time.

**Sweep:** lambda in {0.01, 0.1, 0.5}

**Literature:** Lim, Zohren & Roberts (2019, JFDS) pioneered turnover regularization for deep momentum networks, achieving 2x Sharpe improvement. Khubiev et al. (2025) introduced band turnover regularization (only penalize changes above a threshold).

### Method 4: Differentiable Sharpe Ratio Loss

**Idea:** Bypass classification entirely. Convert model outputs to continuous trading positions, simulate returns with transaction costs, and optimize the Sharpe ratio directly via backpropagation.

**The math:**
```
pos_t = P(Up)_t - P(Down)_t              # Soft position from softmax
R_t = pos_t * (price_{t+1} - price_t)    # Gross return
TC_t = gamma * spread_t * |pos_t - pos_{t-1}|  # Transaction cost
NR_t = R_t - TC_t                        # Net return
Sharpe = mean(NR) / std(NR)              # Risk-adjusted return
Loss = -Sharpe + lambda * turnover       # Minimize negative Sharpe
```

**The half-cost trick (Wood et al., 2025):**
Training with the full transaction cost (gamma=1.0) causes an "inertia trap" — the model learns to never trade, collapsing to zero positions. Training with zero cost produces excessive turnover. gamma=0.5 hits a sweet spot: cost-aware but not paralyzed. This improved net Sharpe by ~50% over full-cost training.

**Requires:**
- Sequential data loading (for consecutive price changes)
- Warm-starting from a pre-trained CE model (random weights → noisy Sharpe signal)
- Lower learning rate (0.0001) to avoid destroying learned features
- Mid-prices and spreads extracted from input tensor during training

**Sweep:** gamma_cost in {0.5}, lambda_turnover in {0.01, 0.1}

**Literature:** Moody & Saffell (2001, IEEE TNN) established differentiable Sharpe. Zhang, Zohren & Roberts (2020, JFDS) applied to deep learning. Lim et al. (2019, JFDS) showed 2x Sharpe improvement with turnover regularization. This is the most evidence-backed approach.

---

## Experiment Plan

All experiments at horizon k=10 (our primary evaluation horizon). DeepLOB architecture (144K params) for comparability with Phase 5-6 baselines.

### Experiments (12 total)

| Exp ID | Loss Type | Primary Param | Other Details |
|--------|-----------|---------------|---------------|
| exp036 | weighted_ce | w_stat=2.0 | Same training as Phase 3 DeepLOB |
| exp037 | weighted_ce | w_stat=3.0 | |
| exp038 | weighted_ce | w_stat=5.0 | |
| exp039 | focal | gamma=1.0 | |
| exp040 | focal | gamma=2.0 | |
| exp041 | focal | gamma=3.0 | |
| exp042 | turnover | lambda=0.01 | Sequential loading, LR=0.01 |
| exp043 | turnover | lambda=0.1 | Sequential loading, LR=0.01 |
| exp044 | turnover | lambda=0.5 | Sequential loading, LR=0.01 |
| exp045 | sharpe | lt=0.01, gc=0.5 | Sequential, warm-start, LR=0.0001 |
| exp046 | sharpe | lt=0.1, gc=0.5 | Sequential, warm-start, LR=0.0001 |
| exp047 | sharpe | lt=0.5, gc=0.5 | Sequential, warm-start, LR=0.0001 |

### Evaluation Plan

Each experiment will be evaluated on:
1. **Classification metrics:** Accuracy, weighted F1 (comparison with Phase 5)
2. **Naive backtest:** Direct PnL from predictions (no post-hoc strategies)
3. **With holding period:** h=100 and h=200 (our best post-hoc strategies)
4. **Trade statistics:** Number of trades, fraction flat, prediction stability

### Success Criteria

A cost-aware model is "successful" if it achieves ANY of:
- Higher naive PnL than standard CE DeepLOB (-7.91 is the bar to beat)
- Higher PnL than standard DeepLOB + holding period (+0.4757 at h=200)
- Fewer trades with comparable accuracy
- Comparable PnL with fewer trades (lower risk)

The ultimate benchmark is our current best: **LR-Stack h=200 at +0.6880 PnL.**

---

## Key References

1. Lim, Zohren & Roberts (2019). "Deep Momentum Networks." Journal of Financial Data Science. — Sharpe + turnover regularization framework
2. Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV. — Focal loss
3. Wood, Roberts & Zohren (2025). "DeePM." arXiv:2601.05975. — Half-cost trick
4. Prata et al. (2024). "LOBCAST." Artificial Intelligence Review. — Benchmark showing profitability gap
5. Briola et al. (2025). "LOBFrame." Quantitative Finance. — Sequential prediction matters
6. Berti & Kasneci (2025). "TLOB." arXiv:2502.15757. — Spread-aware labeling challenge
7. Yin & Wong (2022). "Deep LOB Trading." Expert Systems with Applications. — Focal + Kelly
8. Khubiev et al. (2025). "Finance-Grounded Optimization." arXiv:2509.04541. — Band turnover
9. Moody & Saffell (2001). IEEE TNN. — Original differentiable Sharpe ratio
10. Zhang, Zohren & Roberts (2020). JFDS. — Differentiable Sharpe for deep learning
