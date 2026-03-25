# Architecture Decision Records

> Log ALL significant design choices here. Both Claude Code and Desktop reference this.

---

## ADR-001: PyTorch over TensorFlow
- **Date**: Project start
- **Decision**: Use PyTorch
- **Rationale**: 85%+ of recent LOB papers use PyTorch. Official DeepLOB repo has PyTorch version. All follow-up papers (TransLOB, TLOB, AxialLOB) are PyTorch. Better MPS support for M4 Pro dev work. Industry standard for quant research.

## ADR-002: FI-2010 Setup 2 as Primary Benchmark
- **Date**: Project start
- **Decision**: Use Setup 2 (7-day train / 3-day test) as primary
- **Rationale**: More training data for deep model. Standard in literature since 2017. All comparison papers use this split. Setup 1 (anchored forward) as secondary for robustness.

## ADR-003: Extension Architecture — DeepLOB-Attention
- **Date**: 2026-03-22
- **Options Considered**:
  1. **MambaLOB** — Replace LSTM with Mamba state-space model. Genuine novelty, O(L) complexity. Risk: newer library, less community support, unproven for LOB.
  2. **Transformer Attention** — Replace LSTM with self-attention. Well-documented (TransLOB, TLOB). Proven improvement pathway.
  3. **Bayesian Uncertainty** — MC Dropout on existing DeepLOB. Easiest but less of an architectural extension.
  4. **Combination** — MambaLOB + Bayesian.
- **Decision**: Option 2 — **Transformer encoder replacing LSTM** (DeepLOB-Attention)
- **Rationale**:
  1. Cleanly replaces one component (LSTM → Attention), enabling clear ablation study
  2. CNN+Inception blocks reused identically — isolates the temporal modeling change
  3. Well-documented in LOB literature (TransLOB, TLOB papers use similar approach)
  4. PyTorch `nn.TransformerEncoder` is battle-tested and GPU-optimized
  5. Self-attention computes direct pairwise interactions between all timesteps (vs LSTM's sequential processing)
- **Architecture**: CNN blocks 1-3 + Inception (identical to DeepLOB) → Learnable PE → 2× TransformerEncoderLayer(d=192, 4 heads, ff=256) → Mean pool → FC
- **Hyperparameters**: LR=0.001, adam_epsilon=1e-8, warmup=5 epochs, dropout=0.1
- **Parameters**: ~597K (vs DeepLOB's 144K)
- **📚 Study on Desktop**: Self-attention mechanism, positional encoding, Transformer architecture

## ADR-004: Normalization Strategy
- **Date**: Project start
- **Decision**: Use FI-2010's pre-normalized z-score data for benchmark reproduction. For any extension datasets, use rolling 5-day z-score (as paper Section III.C describes for LSE).
- **Rationale**: Fair comparison requires same normalization. Rolling z-score avoids look-ahead bias.

---

## ADR-005: Official Code Specs over Paper Specs
- **Date**: 2026-03-21
- **Decision**: Follow the official GitHub code (32 filters, BatchNorm, Tanh in block 2)
- **Rationale**:
  1. Official code is what produced the reported results — papers often simplify
  2. BatchNorm stabilizes training significantly (easier convergence)
  3. Tanh in block 2 bounds values before Inception, prevents exploding activations
  4. All extension papers (TransLOB, TLOB, AxialLOB) build from official code
  5. Can always ablate to 16 filters in Phase 4 experiments
- **Implications**: ~144K params (vs ~62K paper spec). LSTM input is 192 (not 96).

---

## Pending Decisions
- [x] ADR-003: Extension architecture choice → DeepLOB-Attention (2026-03-22)
- [ ] Crypto dataset selection (if doing cross-asset transfer)
- [ ] Dashboard tech stack details (Gemini + Figma confirmed for frontend)
