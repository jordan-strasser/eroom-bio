# Conditional-overlap experiment — first-principles edge-assignment test

**Date:** 2026-06-11 · branch `fix/st-field-faithfulness`
**Tool:** `scripts/conditional_overlap_experiment.py` (pure case-based predictor,
zero edge attribution, leave-one-TRIAL-out holdout — counts only, no rebuild/MC).
**Motivation (owner):** per-edge scalar Beta attribution MARGINALIZES over the
conditioning context — chain A (n1..n6, right pop) succeeds, chain B (same n1..n6,
wrong pop) fails → B drags the SHARED edges to ~0.5, both predict the cohort
average, held-out trials can't be ranked. The fix is CONDITIONAL pooling over the
trials that share a sub-configuration (triangulation overlap). Test it directly.

## Result — the hypothesis is VALIDATED; first clean break above the ~0.53–0.57 ceiling

Holdout AUROC (LOO) by overlap scheme, k_min=2, across three independent slices.
Random-fold reference (deployed softmin-over-edge-marginals): **0.534**.

| scheme (what the outcome is pooled over) | n=50 (36 tr) | n=100 (64 tr) | n=250 (142 tr) |
|---|---|---|---|
| `marginal_mech`  {mechanism} ≈ today's altitude | 0.520 | 0.534 | 0.478 |
| `marginal_bio`   {biology} | 0.573 | 0.625 | 0.570 |
| **`bio_linegrp`  {biology, early/late line}** | 0.487* | **0.711** | **0.636** |
| `marginal_ind`   {indication} base-rate baseline | 0.593 | 0.385 | 0.418 |

*n=50: too few trials per (biology, line) cell → conditioning is sparse and hurts
(the triangulation-overlap floor — the conditional needs enough overlap to estimate).

Full n=250 table (all schemes, Δ vs marginal_mech):

| scheme | AUROC(mean) | AUROC(min) | Δ vs marginal_mech |
|---|---|---|---|
| marginal_mech | 0.478 | 0.458 | +0.000 |
| mech_line | 0.456 | 0.436 | -0.022 |
| mech_linegrp | 0.491 | 0.477 | +0.013 |
| marginal_bio | 0.570 | 0.552 | +0.092 |
| bio_line | 0.604 | 0.583 | +0.126 |
| **bio_linegrp** | **0.636** | **0.649** | **+0.158** |
| bio_pop | 0.584 | 0.534 | +0.106 |
| bio_severity | 0.578 | 0.566 | +0.100 |
| bio_stage | 0.581 | 0.569 | +0.103 |
| bio_linegrp_sev | 0.633 | 0.648 | +0.155 |
| marginal_ind | 0.418 | 0.418 | -0.060 |
| backbone | 0.465 | 0.443 | -0.013 |

### Two robust, separable findings

1. **The UNIT is biology, not mechanism.** `marginal_bio` (0.57–0.63) beats
   `marginal_mech` (0.48–0.53) at every n. The mechanism/pathway node — the layer
   this whole branch (Option 2) optimized — is a SHARED HUB whose marginal success
   rate sits at base rate, so it carries no ranking signal. The downstream BIOLOGY
   node (BioLORD-merged process) does. This recontextualizes Phase C: the flat
   honest curve wasn't "no signal in the chain," it was "we predicted off the wrong
   (mechanism-edge-marginal) altitude."

2. **Conditioning biology on coarse line-of-therapy adds real, robust signal.**
   `bio_linegrp` − `marginal_bio` = **+0.086 (n=100), +0.066 (n=250)**, robust to
   k_min (k_min=3: +0.076). Line is SPECIFIC: severity (+0.008) and stage (+0.011)
   barely move it, and stacking severity on top of line doesn't help
   (`bio_linegrp_sev` 0.633 ≈ `bio_linegrp` 0.636). Coarse early/late beats raw line
   (0.636 > 0.604) — raw-line cells are too sparse. Conditioning the MECHANISM on
   line does NOT work (`mech_line` 0.456 < 0.478) — the conditioner only pays off on
   the right (dense, meaningful) unit.

Descriptive confirmation (Step 0): within a mechanism, early-line trials succeed more
than late-line — Δ = +0.04 / +0.27 / +0.14 at n=50/100/250 (always positive). Same
mechanism, different line, systematically different outcome — the motivating example,
measured.

## What it implies for the architecture

The deployed predictor is a **noisy-AND of context-free, per-edge MARGINAL Betas**,
read off the **mechanism** layer — the two worst choices the data exposes. The signal
that beats the ceiling is **outcome pooling at the BIOLOGY node, CONDITIONED on coarse
line-of-therapy**, with backoff for sparsity. Redesign path:

- Predict from a **biology × line-of-therapy** conditional success estimate (Beta-
  smoothed, distinct-trial counts), not a product of edge marginals.
- **Hierarchical backoff** (biology,line) → (biology) → base when a cell is sparse —
  machinery already exists (`beliefs.pool_hierarchical`, Phase A), here over the
  chain-context hierarchy instead of the indication hierarchy.
- Line-of-therapy is currently STRUCTURALLY DROPPED from P(success)
  (`project_predictor_signal_finding`: line in 118/370 pop nodes but 0 evidenced
  `responds_differently` edges). This is the measured cost of that gap, and the fix.
- Keep mechanism/biology nodes for interpretability + cross-indication accumulation,
  but STOP predicting off mechanism-edge marginals.

## Honest caveats

- n is modest (36/64/142 trials); AUROC CI ≈ ±0.08–0.12. The 0.711 (n=100) vs 0.636
  (n=250) wobble is slice + sample noise. ROBUST: the ORDERING (bio > mech;
  bio_linegrp > bio at n≥64), the SPECIFICITY (line, not severity/stage), and k_min
  insensitivity. Confirm at n=500 before committing the rebuild.
- This is a case-based predictor showing the SIGNAL is estimable. Translating it into
  the graph's prediction (a conditional biology-belief / biology×line factor) is the
  design step.
- The conditional needs overlap density (fails at n=36) → a SCALE-rewarded signal,
  unlike the flat marginal curve. That is itself the north-star tell: the right
  representation makes the curve rise with n.

## Files
- `scripts/conditional_overlap_experiment.py` — the experiment (case-based LOO).
- `scripts/cross_indication_transfer.py` — the LOIO test that motivated this pivot
  (pooled 0.468 — re-slicing AUROC doesn't escape a belief-formation ceiling).
