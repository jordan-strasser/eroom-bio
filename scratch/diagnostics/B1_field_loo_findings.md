# B1 Step 1 — BioLORD field honest LOO + effective reuse (real `multi_500`)

The (s,t) BioLORD field is built but its out-of-sample number was never run.
This closes that open item with the SAME leakage discipline as
`eval_holdout_kfold.py:105`, made fully honest for the field: per fold,
re-attribute the `initial` EXCLUDING the fold (clean scalar) AND re-materialize
the field on that clean graph (anchors + marginal both exclude the fold), then
predict. Harness: `scratch/diagnostics/field_holdout.py`. Cost ≈ 5 min/fold
(re-attr 6s + materialize ~25s + 2000-sample prediction); 5 folds ≈ 25 min.

## Honest holdout AUROC (n=221, success=147, failure=74)

| | AUROC | binary acc |
|---|---:|---:|
| scalar holdout (discrete-edge baseline) | **0.565** | 0.661 |
| BioLORD field holdout | **0.561** | 0.643 |
| field − scalar | **−0.004** | −0.018 |

- The scalar reproduces the documented baseline **0.565 to the digit** → the
  harness is a faithful control.
- The field is **statistically indistinguishable from the scalar (−0.004)** — it
  does **not** beat 0.565. (Per-fold: f1 0.636/0.693, and the spread averages out
  to ≈0 over all 221.) The earlier `eval_holdout_kfold --field` anchor-drop LOO
  looked better only because its marginal fallback still carried the held-out
  trial's scalar on singleton edges (leaky); the honest re-materialization removes
  that and the lift vanishes.

## Why: the field is a PER-EDGE surface, not cross-node sharing

`belief_field.query()` sums only over a SINGLE edge's anchor list. It separates a
*shared* edge's evidence by (s,t); it cannot let a singleton biology node borrow
from a *different* node's edge. So after LOO, a held-out trial's singleton-biology
chain edge has no training anchors → falls back to the scalar marginal. Confirmed
by the effective-reuse probe (`b1_substrate_probe.py`):

| metric (biology nodes, n=212) | value |
|---|---:|
| discrete reuse (#host trials): median / %≥8 | 1.0 / **5.2%** |
| singletons (reuse ≤ 1) | **71%** |
| field within-edge eff sample-size ÷ trial count | **1.10×** |
| biology-node field eff reuse %≥8 | **9.9%** (vs discrete 5.2%) |

The field lifts effective reuse by ~10% — it localizes existing within-edge
evidence but does **not** manufacture cross-node transfer, so the singleton
biology layer stays starved (well below the reuse-8 recovery bar).

## Per-target safety transfer (Option-2 metric 3) — B1-invariant

Within-target AE posterior SD = **0.048** (n=45 targets, = the P9 baseline). AE
edges carry no field and key on target identity (already merged), so the biology
re-representation leaves the safety layer untouched. The per-target safety
decomposition is solid and shippable **today**, independent of the biology fix.

**Verdict:** the field neither beats the discrete-edge holdout nor lifts biology
effective reuse. It is not the B1 lever. See `B1_DECISION.md`.
