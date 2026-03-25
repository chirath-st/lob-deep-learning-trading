# Project Goals & Philosophy

> This file captures the overarching goals for this project beyond just "making it work."
> Every decision should be evaluated against these priorities.

## Priority Order

1. **Knowledge & Learning** — Deeply understand every component. Don't copy-paste code
   without understanding it. Ask "why" at every step. Each phase should teach new concepts
   in deep learning, financial ML, and software engineering.

2. **Resume & Portfolio Quality** — This project should demonstrate:
   - Independent reproduction of a published paper (not just running someone's code)
   - Extension with a novel architecture comparison (LSTM vs Transformer)
   - Full ML pipeline: data → model → training → evaluation → backtesting → write-up
   - HPC experience (SLURM, GPU training on Amarel)
   - Clean code, documentation, and reproducibility

3. **Rigor Over Speed** — Not in a rush. Prefer doing things properly:
   - Real ablation studies with actual numbers, not hand-waving
   - Statistical awareness (single-seed limitations, variance)
   - Financial realism (transaction costs, not just accuracy)
   - Clear documentation of what worked and what didn't

## What Makes This Project Stand Out

- **Full reproduction** of a real ML finance paper, not a toy tutorial
- **Multiple model comparisons**: DeepLOB vs baselines vs attention extension
- **Financial evaluation** (Phase 6): bridging ML metrics to trading profitability
- **HPC training**: real-world experience with GPU clusters
- **Clean engineering**: configs, tests, experiment tracking, documentation
- **Honest analysis**: documenting negative results (e.g., attention didn't help much)
