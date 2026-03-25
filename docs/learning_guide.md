# DeepLOB Learning Guide — For Claude Desktop Sessions

> Upload this file + the DeepLOB paper PDF to a Claude Desktop Project.
> Use Desktop for conceptual understanding. Use Claude Code for implementation.

## Custom Instructions for Desktop Project
Paste this into your Claude Desktop project's custom instructions:

---
You are an ML research tutor helping a data science student replicate and extend the DeepLOB paper. The student understands Python and basic ML but is learning deep learning architecture design, financial market microstructure, and research methodology as they go.

Rules:
- Explain concepts at graduate level but start from fundamentals when introducing new topics
- Use analogies and visual descriptions — the student learns well from concrete examples
- When the student shares code, analyze it against the paper's specifications
- Reference the uploaded paper when answering architecture questions
- When a concept needs hands-on implementation, say: "🔧 Implement this in Claude Code"
- Track what has been explained in previous messages — build on it, don't repeat
- The student's project state is tracked in their docs/ files — ask them to share current state if needed
---

## Topics to Study (by Phase)

### Phase 1: Foundations
- What is a Limit Order Book? Bid/ask, levels, market microstructure
- How do prices form? Mid-price, micro-price, order flow imbalance
- Why is financial data hard for ML? Non-stationarity, low SNR, regime shifts
- What is z-score normalization and why does it matter?
- Time-series train/test splits: why you can't shuffle

### Phase 2: Architecture Concepts
- Convolutional Neural Networks: kernels, strides, padding, feature maps
- WHY convolutions work for LOB: FIR filters, equivariance to translation
- The stride trick: how (1,2) stride pairs price+volume → imbalance features
- Inception Modules: multiple temporal scales, like different moving averages
- LSTM: memory cells, gates, why it captures temporal dependencies
- BatchNorm vs no BatchNorm (DeepLOB doesn't use it — why?)

### Phase 3: Training Concepts
- Cross-entropy loss for classification
- Adam optimizer: why epsilon=1? (numerical stability for financial data)
- Early stopping: overfitting in financial data
- Class imbalance: why F1 matters more than accuracy
- Learning rate schedules (the paper uses fixed LR — modern alternatives?)

### Phase 5: Extension Concepts
- **If MambaLOB**: State-space models, selective scan, why O(L) matters for HFT
- **If Transformer**: Self-attention, positional encoding, why attention maps are interpretable
- **If Bayesian**: MC Dropout, epistemic vs aleatoric uncertainty, calibration
- Transfer learning: why features might be "universal" across instruments

### Phase 6: Finance Concepts
- Sharpe ratio: risk-adjusted returns
- Maximum drawdown: worst peak-to-trough decline
- Transaction costs: bid-ask spread, market impact, slippage
- Why classification accuracy ≠ trading profitability
- Mid-price simulation assumptions and limitations

## Questions to Ask Desktop (examples)
- "Explain how the Inception Module in DeepLOB works like different moving averages"
- "Why does the paper use stride (1,2) in the first conv layer instead of (1,1)?"
- "What is the micro-price formula and why does it matter for prediction?"
- "Help me understand LSTM gates — how does the forget gate work?"
- "The paper says DeepLOB has 60K params vs CNN-I's 768K. Why is smaller better here?"
- "Explain the LIME sensitivity analysis results in Figure 9"
