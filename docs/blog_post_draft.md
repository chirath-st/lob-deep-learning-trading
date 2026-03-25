# The Best Classifier I Built Was the Worst Trader

*I reproduced a deep learning paper for limit order book prediction, ran 60+ experiments on a GPU cluster, and learned that accuracy is the wrong metric for finance.*

---

The punchline of six weeks of work is this: my best deep learning model predicted stock price direction with 79% accuracy, and it lost money on every single trade.

That is not a bug. It is the central unsolved problem of applying machine learning to financial markets, and I had to build the entire pipeline myself before I truly understood why.

This is the story of reproducing a published deep learning paper, extending it with my own architecture, discovering that a model with 1,200 parameters beats one with 144,000, running 60 experiments to try to fix the profitability problem, and ultimately learning that a logistic regression with a holding period outperforms every neural network I trained.

## What I Set Out to Do

For my portfolio project at Rutgers, I decided to reproduce DeepLOB (Zhang, Zohren & Roberts, 2019), a convolutional neural network designed to predict price movements from limit order book (LOB) data. The paper was published in IEEE Transactions on Signal Processing and has become a benchmark in financial ML.

A limit order book is the queue of buy and sell orders waiting to be filled at a stock exchange. At any moment, you can see the best prices people are willing to buy at (bids) and sell at (asks), along with the volumes at multiple price levels. This stream of data updates thousands of times per second and contains signals about where the price is headed next.

DeepLOB takes a snapshot of the 10 best bid and ask levels over the last 100 timesteps -- a 100x40 matrix -- and predicts whether the mid-price will go up, down, or stay flat over the next k timesteps. The architecture is a pipeline: three blocks of convolutional layers extract spatial features from the order book, an Inception module captures patterns at multiple temporal scales, and an LSTM processes the sequence to make a final prediction.

My goal was not just to run someone else's code. I wanted to implement the model from the paper specifications, reproduce the reported accuracy numbers, then extend the architecture by replacing the LSTM with a Transformer encoder and run a full ablation study. After that, I planned to backtest everything against realistic trading conditions -- something the original paper does not do.

I trained everything on Rutgers' Amarel HPC cluster using SLURM jobs on RTX 3090 and A100 GPUs.

## How Close I Got: Reproduction Results

The paper reports 84.47% accuracy at horizon k=10 (predicting 10 steps ahead) on the FI-2010 benchmark dataset, which contains limit order book data from five Finnish stocks over ten trading days.

I hit 81.88%. A gap of 2.59 percentage points.

That gap is expected and honest. The paper trains on all seven days of training data without a validation split, while I held out part of the training set for early stopping. I also followed the official code specifications (32 convolutional filters, BatchNorm) rather than the paper text (16 filters, no BatchNorm) -- a common discrepancy in ML research where the code that produced the results does not exactly match what the paper describes.

Across all five prediction horizons:

| Horizon | Paper | Mine | Gap |
|---------|-------|------|-----|
| k=10 | 84.47% | 81.88% | -2.59 |
| k=20 | 77.76% | 73.77% | -3.99 |
| k=30 | 79.46% | 76.14% | -3.32 |
| k=50 | 82.18% | 77.34% | -4.84 |
| k=100 | 84.44% | 78.19% | -6.25 |

Close enough to confirm the architecture works. The gap widens at longer horizons, which makes sense -- with less training data and a validation split, the model has fewer examples to learn from.

## The Surprise: A 1,200-Parameter Model Beats a 144,000-Parameter One

After reproducing DeepLOB, I built my extension: DeepLOB-Attention, which replaces the LSTM with a two-layer Transformer encoder (d=192, 4 heads). This is the approach taken by several follow-up papers (TransLOB, TLOB). At 597,000 parameters -- four times larger than DeepLOB -- it won on four out of five horizons. But only by tiny margins: +0.31 percentage points on average.

Four times the parameters. Four times the compute. One-third of a percentage point.

That made me wonder: how much of DeepLOB's performance comes from the CNN feature extraction versus the temporal modeling (LSTM/Inception)? So I ran an ablation study with two stripped-down variants:

- **CNN-Only**: Just the three CNN blocks, global average pooling, and a linear classifier. Around 1,200 parameters.
- **CNN+Attention**: The CNN blocks plus a small Transformer (d=32, 4 heads). Around 67,000 parameters. No Inception module.

The results were humbling:

| Model | Parameters | Avg Accuracy |
|-------|-----------|-------------|
| CNN-Only | ~1,200 | 78.13% |
| CNN+Attention | ~67,000 | 78.98% |
| DeepLOB | ~144,000 | 77.46% |
| DL-Attention | ~597,000 | 77.77% |

CNN-Only, with 0.8% of DeepLOB's parameters, outperforms the full model on four out of five horizons. CNN+Attention is the best classifier overall.

The Inception module and LSTM -- the components that make DeepLOB architecturally interesting -- contribute almost nothing on the FI-2010 dataset. The convolutional layers that pair prices with volumes and aggregate across order book levels do nearly all the work.

There is an important caveat: the ablation models used a learning rate of 0.001 while DeepLOB used the paper's prescribed 0.01, and everything ran with a single random seed (42). A rigorous comparison would need multiple seeds and a learning rate sweep. But the direction of the result -- that most of the signal is in the spatial CNN features, not the temporal modeling -- is robust enough to be interesting.

## The Wake-Up Call: Every Model Loses Money

This is where the project took a turn I did not expect.

I built a backtesting framework that converts each model's predictions into trading signals, applies realistic transaction costs based on the bid-ask spread, and calculates profit and loss. The setup is straightforward: when the model predicts "up," you buy; when it predicts "down," you sell; when it predicts "stationary," you go flat.

Every single model loses money. Not a little -- catastrophically.

| Model | Accuracy | Naive PnL | Trades |
|-------|----------|-----------|--------|
| DeepLOB | 81.88% | -7.91 | 37,115 |
| CNN-Only | 83.53% | -7.99 | 37,126 |
| DL-Attention | 82.44% | -8.09 | 38,178 |
| CNN+Attention | 83.84% | -8.48 | 37,469 |

The best classifier (CNN+Attention at 83.84% on k=10) is the worst trader at -8.48 PnL. Higher accuracy does not help -- all models generate roughly 37,000+ trades and the transaction costs overwhelm any directional signal.

The problem is structural. Acting on every prediction generates roughly 37,000 trades across the test period. The bid-ask spread -- the cost of entering and exiting a position -- compounds with each trade. Even with 80%+ accuracy, the small average profit per correct prediction cannot overcome the costs of 37,000 round trips.

This is not unique to my implementation. A recent benchmark paper (LOBCAST, Prata et al. 2024) tested 15 state-of-the-art LOB models and concluded that profitability is "far from guaranteed." Even TLOB, which achieves 92.8% F1 on FI-2010, found that "performance deterioration underscores the complexity of translating trend classification into profitable trading strategies."

## 60 Experiments to Fix It

I was not ready to accept this. If the problem is that models trade too often, maybe I could train them to be more selective.

I implemented four cost-aware training methods, each attacking the problem from a different angle:

1. **Weighted Cross-Entropy** -- penalize the model more heavily for misclassifying the "stationary" class, making it more conservative about predicting trades.
2. **Focal Loss** -- down-weight easy predictions to focus learning on the hard boundary cases between stationary and directional movements.
3. **Turnover-Penalized Loss** -- add an explicit penalty for changing predictions between consecutive timesteps, analogous to transaction costs.
4. **Differentiable Sharpe Ratio** -- bypass classification entirely and optimize the risk-adjusted return directly through backpropagation.

I ran 12 experiments at horizon k=10 (three hyperparameter settings per method), then expanded to all five horizons for a total of 60 experiments on Amarel.

The results were sobering. At k=10, the best cost-aware model (Sharpe loss with turnover penalty 0.5) achieved +0.62 PnL with a holding period of 200 -- close to but still below the simple logistic regression stacking approach (+0.69). Focal loss with gamma=2.0 was the only classification-preserving method that went positive (+0.15).

But the real lesson came from the multi-horizon analysis. Turnover-penalized loss collapsed from 72% accuracy at k=10 to 35-61% at longer horizons. The differentiable Sharpe ratio was even worse: 18-62% accuracy on horizons beyond k=10. These trading-aware losses learned patterns specific to k=10's label dynamics and could not generalize.

Only the classification-preserving methods (Weighted CE, Focal Loss) were robust across horizons, maintaining accuracy within a few points of the standard cross-entropy baseline. But being robust was not the same as being profitable -- none of them consistently made money.

The central conclusion after 60 experiments: the accuracy-profitability gap is structural. It arises from transaction costs and trade frequency, not from how you design the loss function. You cannot train your way out of it.

## What Actually Worked

The strategy that finally made money was embarrassingly simple.

Instead of modifying the training process, I applied post-hoc trade reduction: take any model's predictions and simply hold each position for a minimum number of timesteps before allowing a change. This filters out the rapid flip-flopping that generates excessive transaction costs.

A logistic regression model with a holding period of 100 timesteps achieved +0.61 PnL. A stacking meta-learner (logistic regression trained on the softmax outputs of all four deep learning models) with a holding period of 200 timesteps hit +0.69.

The best overall strategy: combine the predictions of models that individually lose money, then trade infrequently.

All profitable strategies require roughly a 35x reduction in trade count compared to the naive baseline. At that level of trade reduction, the model's accuracy matters far less than its signal stability.

## Going Further: Advanced Trading Strategies

After finding that the simple LR-Stack with holding period was the best strategy at +0.69 PnL, I wanted to push further. Could I beat it without retraining any models?

I implemented five post-hoc strategies that exploit the existing predictions more intelligently:

1. **Ensemble Disagreement** -- when all 6 of my structurally different models (CNN, LSTM, Transformer, logistic regression, XGBoost) agree on a prediction, the accuracy jumps from 84% to 89%. Trading only when models agree with a holding period cut trades from 37,000 to 631 and achieved +0.68 PnL.

2. **Multi-Horizon Arbitrage** -- I have predictions at 5 different time horizons. When all horizons predict the same direction, the signal is much stronger. Combining model agreement with cross-horizon consistency created a "double filter" worth +0.68 PnL.

3. **Regime Detection** -- not all market conditions are equally predictable. I classified periods into low/medium/high volatility regimes. Suppressing trades during low-volatility periods (where spreads eat all the profit) and combining with other filters added value.

4. **Kelly Position Sizing** -- instead of binary go/no-go decisions, I tried continuous position sizing using calibrated probabilities. This actually performed *worse* than simple gating -- the fractional positions generated excessive position-change costs without enough edge.

5. **Meta-Labeling** -- following Lopez de Prado's framework, I trained a secondary classifier to predict whether each trade would be profitable. It achieved AUC=0.83, and the most important features were spread and volume imbalance -- market microstructure matters more than model confidence.

The best result came from layering everything together: ensemble agreement (6/6 models) + cross-horizon consistency (3/5 horizons) + regime conditioning (low+medium volatility) + holding period of 200. This achieved **+0.71 PnL with only 461 trades** -- beating the Phase 6 champion while trading 58% less.

The core insight: the path from 79% accuracy to profitability runs through *selectivity*, not through better classification. Each filter removes a different type of bad trade, and their combination is more powerful than any single approach.

## What I Learned

**1. Accuracy is the wrong metric for trading.** Classification metrics evaluate each prediction independently. Trading is sequential: the cost of changing your mind compounds over time. A model that predicts correctly 80% of the time but changes its prediction every timestep will lose money. A model that predicts correctly 65% of the time but only changes its prediction when it is confident can be profitable.

**2. Start with the simplest model that could work.** My 1,200-parameter CNN-Only model outperformed the full DeepLOB architecture. This is not always the case -- the CNN captures the specific spatial structure of the FI-2010 order book very efficiently. But it is a reminder that architectural complexity should be justified by results, not by intuition.

**3. Negative results are the most valuable results.** If I had only reported the classification accuracy of my models, this project would tell a misleading story: "Deep learning achieves 82% accuracy on limit order book prediction." The backtesting phase, which showed that every model loses money, is the most honest and informative part of the entire project.

**4. Post-hoc beats end-to-end (sometimes).** I spent weeks implementing differentiable trading losses, only to find that a simple holding period applied after training outperforms all of them. End-to-end optimization is elegant in theory, but in practice, the training signal from a Sharpe ratio loss is noisy and fragile. Sometimes the right approach is to let the model learn to classify, then apply domain knowledge as a separate filter. The Phase 8 layered strategy -- stacking independent filters on top of accurate classifiers -- confirms this: +0.71 PnL from post-hoc processing alone.

**5. The gap between ML research and trading reality is enormous.** Most academic papers on LOB prediction report accuracy and F1 scores. Very few include backtests with transaction costs. The reason is obvious: the results look much worse. But this gap is exactly where the interesting problems are.

## What I Would Do Differently

If I started over, I would build the backtesting framework first and evaluate models on financial metrics from day one. I spent the first four phases optimizing for accuracy before discovering that accuracy was the wrong target.

I would also invest in multi-seed experiments early. Running everything with a single seed (42) saved GPU hours but limits the statistical claims I can make. The ablation results are directionally interesting but not publication-ready without confidence intervals.

Finally, I would explore the labeling problem more deeply. The FI-2010 labels define "up" and "down" using a threshold that does not account for transaction costs. Relabeling the data with a spread-aware threshold could potentially align the classification objective with profitability from the start.

## Looking Ahead

This project started as a paper reproduction and became a case study in why ML metrics do not translate to financial performance. The DeepLOB architecture works as advertised for classification. But classification is not trading.

The most promising direction I see is not better models but better problem formulations: spread-aware labeling, selective prediction (learning when *not* to trade), and reinforcement learning approaches that optimize cumulative PnL directly. The tools exist. The dataset exists. The gap between "predicts well" and "trades well" is where the real research frontier lies.

All code, configs, and experiment results are available in the project repository. I documented everything -- including the failures -- because I believe honest engineering is more impressive than cherry-picked results.

---

*This project was completed as an independent portfolio project at Rutgers University. It involved 95+ experiments on the Amarel HPC cluster, reproduction of a published IEEE paper, and a full backtesting pipeline with transaction costs. The full codebase includes DeepLOB reproduction, Transformer extension, ablation study, backtesting module, cost-aware training experiments, and advanced post-hoc trading strategies.*
