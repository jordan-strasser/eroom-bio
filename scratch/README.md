# scratch/ — non-canonical experiments

Everything under `scratch/` is **one-off, non-canonical** experiment and diagnostic
code. It is NOT part of the build/predict/eval path and the canonical results do
NOT depend on it.

In particular, the `*_RESULTS.md` numbers under `docs/dev/reports/` were originally
produced by untracked one-offs in `scratch/diagnostics/` that consumed a frozen
prediction dump (`onco_graph_preds.json`). Those scripts are kept for provenance
but are **not** the reproduction path.

## Canonical reproduction (tracked, no scratch dependency)

```bash
python -m scripts.reproduce_frontier
```

This runs the two tracked harnesses directly from the built graph, seeded for
exact reproducibility:
- `scripts.eval_holdout_kfold` → holdout AUROC + in-sample
- `scripts.eval_baselines --with-graph` → ΔAUROC vs the 4-field design baseline (paired DeLong)

Frontier numbers (seed 42): holdout AUROC **0.701**, in-sample **0.768**, acc **0.713**.
