# Next session — pickup notes

Last touched: 2026-05-05. Branch `main` is clean and pushed.

## What landed this session

Principled Beta-Binomial belief updates replacing the old
`weight × quality × magnitude` heuristic. Two commits on `main`:

- `22996b9` — Principled belief updates + accumulated branch work
- `a434e73` — Ground rubric in observable checks; rescale N_eff for
  clinical dominance

### The new update

```
α_post = α_prior + N_eff · p_obs
β_post = β_prior + N_eff · (1 − p_obs)
```

- `N_eff` per evidence class lives in `src/inference/beliefs.py`
  (`EVIDENCE_TYPE_N_EFF`). Phase 3 = 15, MR = 10, P2 = 6, GWAS = 4,
  P1 / in vivo = 2, in vitro = 1, computational = 0.3, literature = 0.2.
- `p_obs` per `SupportBucket` lives in the same file (`BUCKET_TO_P_OBS`).
  Seven buckets: strong / moderate / weak × support / contradict, plus
  ambiguous = 0.5. Symmetric around 0.5; floors at 0.05 / 0.95.
- `EvidenceRecord.support` is the bucket name (string for serialization,
  enum at runtime). `quality_score` ∈ [0, 1] discounts `N_eff` and is
  fed by the trial-level classifier confidence rubric. Defaults to 1.0
  for non-LLM evidence streams (LINCS, GWAS).
- Apply via `src/inference/beliefs.apply_virtual_evidence(belief, n_eff,
  p_obs)`. The store / attributor / LINCS adapter all go through this.

### The rubric

`src/annotation/prompts/classification_system.txt` now grounds the
support buckets in observable features. Strong/moderate/weak support
all reference a five-item alternative-explanations checklist (placebo
inflation, regression to the mean, dose-response inconsistency,
natural disease fluctuation, concomitant meds). None apply →
strong_support; one applies → moderate_support; two or more →
weak_support. Every item is a feature the LLM can pull from the trial
report rather than a judgment call.

### Tests

347 non-integration tests passing. New `tests/test_beliefs.py` covers
table invariants (monotonicity, symmetry around AMBIGUOUS, positivity)
and conjugate-update properties (no-mutation, exact +N_eff growth in
evidence_strength, ambiguous evidence preserves the mean).

## Key design decisions (don't relitigate)

- Categorical 7-bucket emission, not free-form 0–1 confidence floats.
  LLMs are notoriously miscalibrated on continuous probability outputs;
  bucketed emissions are repeatable and calibratable later.
- `EvidenceDirection` retained as a derived property of the bucket
  (filtering / display only) — the bucket is the source of truth for
  the update.
- Trial-level `confidence_overall` modulates `N_eff`, not the per-edge
  update. Low classification confidence → fewer effective virtual trials.
- Endpoint→indication priors (Beta(3,1) for OS, etc.) stay as-is —
  they're priors, not evidence. New machinery sits cleanly on top.
- Single `quality_score` field replaces the old `magnitude` ×
  `quality_score` product. No more compounded underspecified floats.

## Open follow-ups

### Calibration harness (the big one)

`src/inference/calibration.py` is a stub. Once ≥50 annotated trials
with known indication outcomes exist, fit `EVIDENCE_TYPE_N_EFF` and
`BUCKET_TO_P_OBS` jointly to minimize Brier on held-out trials.
Constraints: `N_eff > 0`, `p_obs` monotone in bucket strength,
symmetric around AMBIGUOUS = 0.5. Module docstring sketches the
recipe. Return refitted tables rather than mutating module-level
constants so old vs. new is diff-able.

### Contradict-side rubric symmetry

The five-item checklist only tightened the support side per user ask.
The contradict side still says "with mitigating factors: underpowered,
possible confound, or only secondary endpoint missed." If we want
parity, list the symmetric concrete checks: underpowered design,
toxicity-truncated dosing, wrong dose / wrong timeframe, high placebo
masking, dropout / non-compliance, wrong population. Skipped this
session — user only asked for support side.

### N_eff scaling with actual trial size

Current `N_eff` is one-per-evidence-class (Phase 3 = 15 always).
A registrational Phase 3 with N=2000 carries more information than
one with N=80. Could fold trial sample size in as
`N_eff_class × log(N / N_baseline)` or similar. Deferred — adds
LLM-output complexity and the calibration loop will absorb most of
this anyway.

### Predictor integration

`src/prediction/path_query.py` consumes edge beliefs but I didn't
revisit it this session. Worth checking that `weighted_geomean` and
`product` aggregations still behave sensibly given the larger `N_eff`
values (edges now accumulate evidence faster, posteriors will tighten
sooner — may shift the calibration curve).

### Backtest rerun

The 2026-04-30 method-comparison run on n=100 trials was interrupted
mid-flight. With the new belief math, that AUC / calibration baseline
is no longer apples-to-apples — reruns become the new baseline. Most
caches still warm (`data/annotations/`, `data/cache/endpoint_*`,
`data/cache/mechanism_*`, `data/cache/population_*`). Open Targets
disease-association cache was on the prior TODO; check if it was added.

## Where to start next time

1. Sanity-check: `pytest -m "not integration"` should be 347 passing.
2. If picking up belief work: `src/inference/beliefs.py` is the entry
   point. Update tables, calibration, or conjugate-update math there.
3. If picking up rubric work: `src/annotation/prompts/classification_system.txt`.
4. If picking up backtest work: `src/validation/backtest.py` (per the
   prior session's notes — verify the OT cache and timeout-handling
   TODOs first).

## What was already addressed (no action needed)

- `_call_messages_with_backoff` now catches `APITimeoutError` and
  `APIConnectionError` (the prior NEXT_SESSION item 1).
- Combinatorial therapy + clinical subgroup handling: done (per user,
  prior to this session).
- ClinicalTrials.gov "biological" vs "drug" terminology: done.
