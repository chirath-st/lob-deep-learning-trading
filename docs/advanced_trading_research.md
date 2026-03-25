# Advanced Trading Strategy Research for LOB Predictions

> Research conducted 2026-03-23. Eight approaches to bridge the accuracy-profitability gap.
> Source: Deep research on exploiting pre-computed LOB predictions with real-world trading techniques.

## Summary

The gap between 78% classification accuracy and profitable trading is a **strategy design problem**, not a modeling failure. The current best result (+0.69 PnL from LR-Stack with holding period) proves the core signal has value. Eight advanced approaches were evaluated, organized into three tiers by feasibility.

## Implementation Priority

| Priority | Approach | Complexity | Post-hoc? | P(beats +0.69) | Expected PnL |
|----------|----------|------------|-----------|-----------------|--------------|
| 1 | Meta-labeling | Low-Med | Yes | 55–70% | +1.0 to +1.5 |
| 2 | Ensemble disagreement | Low | Yes | 50–65% | +0.77 to +0.90 |
| 3 | Regime detection | Low-Med | Yes | 65–80% | +1.0 to +1.5 |
| 4 | Kelly position sizing | Low | Yes | 60–75% | +0.85 to +1.0 |
| 5 | Multi-horizon arbitrage | Low | Yes | 40–55% | +0.72 to +0.80 |
| 6 | RL execution | High | Partial | 40–55% | +0.5 to +2.0 |
| 7 | Spread-aware relabeling | Med-High | No | 45–60% | +0.8 to +1.1 |
| 8 | Optimal execution | Med-High | Partial | 25–40% | +0.7 to +1.2 |

## Area 1: Kelly Criterion & Position Sizing

**Concept:** Transform binary trade/no-trade into continuous capital allocation using softmax probabilities.

**Key formula:** f ≈ μ/(μ₂ − μ²) where μ is expected return, μ₂ is second moment. Half-Kelly (0.5×) retains ~75% of growth with dramatically reduced variance.

**Prerequisites:** Softmax probabilities must be well-calibrated (temperature scaling first).

**Key references:**
- Kelly (1956), "A New Interpretation of Information Rate," Bell System Technical Journal
- Thorp (2007), "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
- Busseti, Ryu & Boyd (2016), Risk-constrained Kelly via bisection (Stanford)
- Guo et al. (2017), calibration of modern deep networks

**Implementation:** ~15 lines of code. Entirely post-hoc. Expected improvement: 20–50% over binary gating.

## Area 2: Regime Detection

**Concept:** Identify market regimes and suppress trading during unfavorable ones.

**Regime-prediction mapping:**
- Low volatility, high liquidity → predictions less useful, spread too tight
- Moderate volatility, trending → predictions most useful
- Very high volatility / structural breaks → predictions fail
- Strong order flow imbalance → higher predictability

**Methods:**
- Simple volatility threshold (~10 lines)
- Gaussian HMM via hmmlearn (2-3 states, ~50-100 lines)
- Online Bayesian Change-Point Detection (~100 lines)

**Key references:**
- Tsaknaki, Lillo & Mazzarisi (2024), "Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection," Quantitative Finance
- Adams & MacKay (2007), arXiv:0710.3742 (BOCPD foundation)
- Nystrup, Madsen & Lindström (2018), "Dynamic Portfolio Optimization across Hidden Market Regimes," Quantitative Finance
- Hamilton (1989), "A New Approach to Nonstationary Time Series," Econometrica

**Implementation:** Fully post-hoc. Expected improvement: +1.0 to +1.5 PnL.

## Area 3: Reinforcement Learning for Execution

**Concept:** RL agent takes predictions + LOB state as input, learns optimal trading policy accounting for costs.

**Key references:**
- Schnaubelt et al. (2023), "Asynchronous Deep Double Dueling Q-Learning for Trading-Signal Execution in LOB Markets," Frontiers in AI
- Spooner et al. (2018), "Market Making via Reinforcement Learning," AAMAS
- Ning, Lin & Jaimungal (2022), "Double Deep Q-Learning for Optimal Execution," Applied Mathematical Finance
- Sadighian (2019), arXiv:1911.08647 (A2C/PPO framework, GitHub: sadighian/crypto-rl)

**Critical constraint:** FI-2010 has only 10 trading days (~395K events) — far too little for RL. Would need LOBSTER or similar dataset. High complexity (4-8 weeks).

## Area 4: Meta-Labeling (Lopez de Prado)

**Concept:** Secondary binary classifier predicts whether each primary signal will be profitable. Decouples *side* (direction) from *size* (whether/how much to bet).

**Implementation steps:**
1. Triple barrier method: for each prediction, determine if resulting trade would have been profitable after spread costs
2. Build features: softmax probabilities (all 7 models), LOB microstructure (spread, imbalance, depth), volatility, rolling accuracy
3. Train XGBoost/LightGBM binary classifier to predict profitability
4. Use calibrated probability for position sizing

**Key references:**
- Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapter 3
- Joubert (2022), "Meta-Labeling: Theory and Framework," JFDS
- Meyer, Joubert & Alfeus (2022), "Meta-Labeling Architecture," JFDS
- Meyer, Barziy & Joubert (2023), "Meta-Labeling: Calibration and Position Sizing," JFDS
- Thumm, Barucca & Joubert (2023), "Ensemble Meta-Labeling," JFDS
- Library: mlfinlab (Hudson & Thames)

**Note:** No publication exists applying meta-labeling to HFT/LOB specifically — this is a research gap.

**Implementation:** 1-2 weeks. Fully post-hoc. This should be first priority given low risk and direct applicability.

## Area 5: Spread-Aware Relabeling

**Concept:** Relabel FI-2010 with threshold θ = average bid-ask spread, so "up"/"down" only applies when movement exceeds transaction costs.

**Key references:**
- Berti & Kasneci (2025), "TLOB," arXiv:2502.15757 — tested this, performance deterioration as expected
- Briola, Bartolucci & Aste (2025), "Deep Limit Order Book Forecasting: A Microstructural Guide," Quantitative Finance
- Zaznov et al. (2022), "Predicting Stock Price Changes Based on LOB: A Survey," Mathematics

**Problem:** Requires full model retraining. Raw bid/ask prices not available in pre-processed FI-2010. Needs LOBSTER or equivalent. Better as future research direction.

## Area 6: Optimal Execution

**Concept:** Limit order vs. market order decision to reduce spread costs.

**Key references:**
- Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions," Journal of Risk
- Cont & Kukanov (2017), "Optimal Order Placement in Limit Order Markets," Quantitative Finance
- Nevmyvaka, Feng & Kearns (2006), "Reinforcement Learning for Optimized Trade Execution," ICML
- Schnaubelt (2022), "Deep RL for Optimal Placement of Cryptocurrency Limit Orders," EJOR

**Assessment:** At single-lot scale, market impact is negligible. Limit order simulation needs data not in FI-2010. Not recommended as priority.

## Area 7: Ensemble Disagreement

**Concept:** Use agreement across 7 structurally different models as uncertainty signal. When all models agree, confidence is higher than any single model's softmax.

**Metrics to compute:**
- Unanimous agreement (all 7 agree)
- Majority strength (5/7, 6/7, 7/7)
- Ensemble entropy
- Jensen-Shannon divergence between softmax vectors
- Oracle analysis (accuracy when at least one model is correct)

**Key references:**
- Lakshminarayanan, Pritzel & Blundell (2017), "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS
- Zhang, Zohren & Roberts (2018), "BDLOB: Bayesian Deep CNNs for LOB," NeurIPS Workshop — used MC-dropout uncertainty for position sizing on LSE data
- Krogh & Vedelsby (1995), ambiguity decomposition (ensemble error = avg error - avg disagreement)
- Kuncheva & Whitaker (2003), "Measures of Diversity in Classifier Ensembles," Machine Learning

**Implementation:** ~30 lines. Entirely post-hoc. Expected improvement: 10-30% over single-model gating.

## Area 8: Multi-Horizon Arbitrage

**Concept:** Exploit the "term structure" of directional expectations across k=10,20,30,50,100.

**Signal taxonomy:**
- Unanimous cross-horizon agreement → strongest trend signal
- Short-long disagreement (k=10,20 UP, k=50,100 DOWN) → reversal signal
- Confidence gradient (conviction increasing/decreasing across horizons) → momentum building/fading

**Key references:**
- Zhang & Zohren (2021), "Multi-Horizon Forecasting for Limit Order Books," Quantitative Finance (arXiv:2105.10430)
- Kolm, Turiel & Westray (2023), "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from LOB," Mathematical Finance
- Bates & Granger (1969), forecast combination; Claeskens et al. (2016), combination puzzle

**Implementation:** ~40 lines. Entirely post-hoc. Expected improvement: 5-15% alone, 15-30% combined with ensemble disagreement.

## Recommended Combined Strategy

Layer Areas 7+8+2+1+4:
1. **Ensemble agreement × cross-horizon consistency** as double filter (Areas 7+8)
2. **Regime conditioning** — only trade in favorable regimes (Area 2)
3. **Position sizing** via meta-labeling or Kelly criterion (Areas 4+1)
4. **Calibrate all probabilities** before sizing (temperature scaling)

Target: +1.5 to +3.0 PnL by trading fewer but higher-conviction positions with optimized sizing.

**Core insight:** The path from 78% accuracy to profitability runs through selectivity, not through better classification — choosing *when*, *how much*, and *under what conditions* to act on predictions that are already good enough.
