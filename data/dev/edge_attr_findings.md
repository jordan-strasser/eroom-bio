# TASK 2 — per-edge attribution-math experiment (findings)

**Date:** 2026-06-10 · branch `fix/st-field-faithfulness`
**Harness:** `scripts/edge_attr_experiment.py` — re-attributes `neff100_initial.json`
(n=100, pre-attribution, DB-only) EFFICACY-ONLY under each mode, no AE/Anthropic.
**Knobs added (off by default, behavior-preserving):**
`EROOM_EDGE_ATTR` ∈ {explain_away (default), symmetric_full, symmetric_uniform,
symmetric_explain}; `EROOM_EDGE_EFFECT` ∈ {off (default), on}.

## 2a — symmetry experiment: the per-edge split is SECOND-ORDER

In-sample `min`-over-stated-chains AUROC (n=60 scorable, 40 success). IN-SAMPLE so
the absolute (~0.95) is the leakage ceiling — the RELATIVE ranking is the signal.

| mode | in-sample AUROC | affects mean | affects **std** | affects strength |
|---|---|---|---|---|
| explain_away (default) | 0.959 | 0.720 | **0.159** | 13.2 |
| symmetric_full | 0.970 | 0.701 | **0.155** | 14.3 |
| symmetric_uniform | 0.943 | 0.718 | **0.159** | 11.5 |
| symmetric_explain | 0.950 | 0.722 | **0.162** | 11.0 |

**What the mode changes — and doesn't.**
- **Discrimination: barely.** All four sit in 0.943–0.970, inside MC noise of the
  ~0.95 in-sample leakage ceiling. The split rule does not change in-sample separation.
- **Spread (the differentiation that matters): unchanged.** `affects` std is
  0.155–0.162 across all four. This corroborates **T1**: the failure-mass split shapes
  CONFIDENCE/accumulation, not the cross-compound spread.
- **Where it DOES bite — downstream means + evidence strength.** `symmetric_full`
  forces every backbone edge to eat the FULL contradict on a failure, so the
  downstream edges absorb more pessimism: `mechanism_affects` mean 0.529 vs 0.552,
  `biology_drives` 0.541 vs 0.552, and evidence strength accumulates fastest
  (modulates_via 24.2 vs 21.9). `symmetric_uniform`/`explain` split mass thinly →
  lowest downstream strength (biology_drives strength 1.2–1.3 vs 2.5).

**Decision: keep `explain_away` (default).** It is best-or-tied on the (uninformative)
in-sample AUROC AND it is the only mode consistent with the project's
`feedback_trial_failure_not_falsification` axiom — a single failure can't pinpoint the
weak link, so the modest failure mass is steered toward the currently-uncertain edges
and curated high-belief edges self-protect. `symmetric_full`'s marginally-higher
in-sample number comes precisely from DROPPING that self-protection (every edge eats the
full contradict — the new test `test_symmetric_full_breaks_self_protection_on_failure`
pins this), which the axiom argues against. No data reason to switch; one principled
reason not to. The honest out-of-sample test (Phase C, default mode) is the real lever;
the modes don't change in-sample separation, so they aren't expected to move it.

## 2b — effect_size / p_value: the DATA isn't there (the important finding)

Owner: "def need this." Measured the inputs first:

- **`effect_size` is unusable as extracted.** Present on 85% of trials, but it is a
  bare first-number parse of a free-text string. Range **−1.06e5 … 2.43e6, median 6**.
  Samples (all the SAME field):
  - `HR 0.56 (95% CI: 0.45-0.70)` → 0.56  (a real ratio)
  - `41.3% vs 54.0% pCR` → 41.3  (a percentage)
  - `3.8 point improvement in SF-36 MCS` → 3.8  (a point difference)
  - `11 discontinuations (immediate) vs 10 (delayed)` → 11  (a raw count)

  There is **no single scale** on which one float means the same thing across trials.
  Any magnitude rule (even a ratio-plausibility gate) mislabels a `2.2%` response rate
  as a 2.2× ratio. Folding it in is **actively wrong, not merely noisy** → the
  modulation `_effect_modulation` accepts the arg but DELIBERATELY ignores it.
- **`p_value` is the only safe quantitative signal** — semantically uniform (smaller =
  stronger regardless of metric) — but present on **only 26%** of trials
  (median 0.024). So a p_value-only modulation can touch ≤¼ of updates.
- **Measured impact (p_value-only, `explain_away+effect`):** in-sample AUROC 0.956 vs
  0.959, affects mean/std identical (0.722/0.160). Negligible — exactly what 26%
  coverage predicts. Downstream means tick up slightly (mechanism 0.564 vs 0.552) where
  significant-p trials land.

**Decision: keep `EROOM_EDGE_EFFECT` OFF by default.** The shipped p_value-only
modulation is principled and ready, but its ceiling is low until coverage improves.
**The real fix is upstream (root-cause #1/#2, ingestion / data→node mapping):** the
extractor must emit a STRUCTURED effect — `{metric_type ∈ HR/OR/RR/%/difference,
direction-normalized magnitude, CI}` — at which point magnitude → finer support bucket
becomes sign-safe and the existing `_hr_support_bucket` machinery (HR + CI → bucket)
generalizes to it. That is an extractor-prompt + schema change, not an attribution-math
change, and is the prerequisite this experiment surfaced.

## Files
- `src/annotation/attributor.py` — `EROOM_EDGE_ATTR`/`EROOM_EDGE_EFFECT` knobs,
  `_per_edge_fracs`, `_effect_modulation` (default path byte-identical).
- `scripts/edge_attr_experiment.py` — the harness.
- `tests/test_outcome_conditioning.py::TestEdgeAttrModes` — 8 new tests.
