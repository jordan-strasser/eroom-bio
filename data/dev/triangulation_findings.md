# Triangulation (outcome-driven edge-weight inference) — HONEST findings

**Date:** 2026-06-11 · branch `arch/triangulation-edge-weights`
**Approach:** NO LLM beliefs — the LLM only resolves nodes; edge weights are latent,
inferred from outcome patterns across overlapping paths by L2-logistic
(`logit P(success)=Σ_{e∈path} w_e`; coefficients = edge weights). Path = the owner's
**maximal composition** (full fanned subgraph). Observation = trial-level.
**Evaluation:** the BLESSED honest harness in `eval_baselines.py` — IID k-fold +
leave-one-INDICATION-out + leave-one-TARGET-out, identical folds, paired DeLong vs the
deployed graph. (The first fast pass used same-corpus LOO at p≫n — an artifact; see §4.)

## 1. Honest out-of-sample AUROC (n=250 → 107 scorable, base rate 0.692)

| model | IID k-fold | leave-1-TARGET-out | leave-1-INDICATION-out |
|---|---|---|---|
| deployed graph (Beta+softmin) | **0.534** | — | — |
| base: per-target / per-indication | 0.59 / 0.51 | (artifact*) | (artifact*) |
| logreg:mech(b) — categorical nodes | 0.648 | 0.559 | 0.534 |
| xgb:mech(b) — categorical nodes | 0.649 | 0.514 | 0.503 |
| **logreg:edges(comp)** — granular triangulation | **0.560** | 0.518 | 0.421 |
| logreg:edges+filt — + line/bio×line | 0.568 | 0.542 | 0.450 |
| **logreg:edges(coarse)** — fan-out collapsed | **0.615** | **0.715** | 0.509 |
| cond:bio×line — the conditional probe | 0.624 | 0.423 | 0.532 |
| xgb:design(plan) — NON-mechanistic | 0.675 | 0.684 | 0.698 |
| xgb:all(safe) — mech + design | 0.695 | 0.679 | 0.677 |

*base-rate models on pooled leave-group-out produce AUROC artifacts — read their
Brier/Acc, not AUROC (the harness prints the caveat).

**Paired DeLong vs the deployed graph (n=107):** every difference is NON-significant —
`edges(comp)` Δ−0.026 (p=0.73), `edges(coarse)` Δ−0.081 (p=0.32), `cond:bio×line`
Δ−0.091 (p=0.29), `logreg:mech(b)` Δ−0.115 (p=0.11). At this n nothing beats anything.

## 2. What the honest numbers say (calibrated)

1. **At n=250, no approach significantly beats the deployed graph or each other.** All
   models cluster 0.53–0.65 with overlapping ±0.10–0.12 CIs and DeLong p>0.1. n=250
   (107 scorable) lacks the power to declare a winner. **n=500 is the real test.**
2. **Coarsening the fan-out is the clearest signal.** `edges(coarse)` (collapse the
   mechanism-pathway fan-out → compound→target→biology→indication) beats `edges(comp)`
   everywhere (0.615 vs 0.560 k-fold; **0.715 vs 0.518 leave-one-TARGET-out**, with
   POSITIVE Brier skill +0.053 — the only mechanistic model with real skill there). When
   an entire target is held out, the coarse composition generalizes via SHARED biology —
   the structured-transfer the thesis predicts. The granular pathway edges are noise
   (see §3). Validates BOTH the owner's composition idea AND the diagnosis that the
   mechanism fan-out is the noisy layer.
3. **The conditional biology×line "0.64–0.71" was LOO-OPTIMISM.** On the honest harness
   it is 0.624 (k-fold) / 0.532 (leave-indication) / 0.423 (leave-target) — NOT robustly
   above the pack. It was over-reported earlier from same-corpus leave-one-trial-out;
   the documented LOO-optimism trap bit again. Honest correction: the conditional lift is
   real but modest (~0.55–0.62), not a breakthrough.
4. **The uncomfortable benchmark: non-mechanistic DESIGN features win numerically**
   (xgb:design 0.675–0.698 across all modes). Much of trial success is predictable from
   operational facts (enrollment, single-arm, #compounds, rationale) the graph doesn't
   model — and no mechanistic model (graph, triangulation, conditional) clearly beats
   them at n=250. This bounds how much mechanism alone can add (BENCHMARK.md Q1).

## 3. Interpretability — the entanglement is the story (`triangulation_weights.md`)

The owner's "report entanglement" prediction is exactly what dominates. The pathway
fan-out makes a gene's ~8 Reactome pathways ALWAYS co-occur, so their edges share an
identical traversal set → **individually unidentifiable** (L2 splits the weight evenly):
- Bevacizumab→VEGFA→{9 angiogenesis pathways}, succ 1.00, split **+0.104** each.
- {8 EGFR signaling pathways}→cell-growth-inhibition, succ 0.00, split **−0.194** each.
- {11 insulin-secretion pathways}→diabetes (a sulfonylurea), succ 1.00, split +0.077.

So the granular `edges(comp)` spends its capacity on entangled, unidentifiable pathway
edges — precisely why collapsing them (`edges(coarse)`) recovers signal. The weight SIGNS
are domain-sensible (anti-VEGF positive; the failed EGFR trials negative), and a
prediction decomposes cleanly: NCT00085839 predicted-fail 0.143 ← paclitaxel→tubulin
(−0.265) + mitotic-arrest→PFS (−0.237) + EGFR→growth-inhibition (−0.194).

## 4. The p≫n anomaly — explained (owed from the fast first pass)

The first pass reported in-sample AUROC 1.000 and LOO 0.22–0.58. Cause: the fan-out gives
**1892 granular edge-features for 142 trials (p≫n)** → the design matrix is wider than
tall → perfectly separable → in-sample 1.000, and same-corpus leave-one-trial-out in that
regime has unstable held-out coefficient signs, so AUROC can fall BELOW 0.5 (not a bug —
the degenerate p≫n + LOO regime). The honest fix is structural: (a) score with k-fold /
leave-group-out, not same-corpus LOO; (b) reduce p — coarsening (`edges(coarse)`,
p≈few-hundred < n) gives stable 0.615; inner-CV-regularized granular gives a stable 0.560.
No sub-0.5 honest AUROC remains.

## 5. Bottom line + next

The triangulation architecture is sound and the COARSE composition is the most promising
mechanistic signal (0.715 leave-one-target-out with real skill) — but at n=250 it is
statistically indistinguishable from the deployed graph and is beaten numerically by
non-mechanistic design features. **Decision gate: rebuild a clean fan-out at n=500 and
re-run `eval_baselines` (k-fold + leave-group-out + paired DeLong).** If `edges(coarse)`
separates from the graph AND the design baseline with significance, it justifies replacing
the Beta+softmin predictor with the coarse-composition logistic. If not, the honest read
is that mechanism-at-this-altitude is at its ceiling and the lever is new (design/
operational) signal or finer data, not a better edge-weight estimator.

## Files
- `scripts/eval_baselines.py` — added `logreg:edges(comp/coarse/+filt)` + `cond:bio×line`
  models + `Trial.comp_edges` / `_composition` (maximal composition from the subgraph).
- `scripts/triangulation_experiment.py` — refocused to the edge-weight explainer +
  entanglement report (`data/dev/triangulation_weights.md`).
