# A3 + A4 — reason-routed EM with censoring + principled responsibility

**Branch:** `arch/triangulation-edge-weights` · **Flag:** `EROOM_ROUTING` (default OFF)
**Corpus:** `multi_500` · **Initial:** `data/exports/multi_500_initial.json`
**Method:** both snapshots re-attributed from the SAME `initial.json` so the only
difference is the flag. Baseline = `EROOM_ROUTING=0` (reproduces the deployed
behavior); Routed = `EROOM_ROUTING=1`.

> **Expectation set up front (per architecture-v2 §A "what A buys"):** the primary
> success criterion here is **clean beliefs** (#2 contamination removed, #3
> efficacy spread de-centered), NOT a big AUROC jump. Pillar A cleans
> contamination but cannot break the substrate ceiling (Disease 2: 1.24
> trials/edge, 71% singleton biology). A modest AUROC move with #2/#3 moving
> correctly is the expected, informative result — it hands the baton to Pillar B.

---

## Headline

| metric | baseline (off) | routed (on) | Δ |
|---|---|---|---|
| **#2** failure-trials touching efficacy spine | **92.3%** (215/233) | **30.9%** (72/233) | **−61 pts** |
| **#3** efficacy-edge E[p] mean (base rate 0.658) | 0.620 | **0.634** | +0.014 (toward base rate) |
| **#3** efficacy-edge E[p] sd | 0.150 | 0.150 | ~0 |
| holdout AUROC (TRUE out-of-sample) | **0.565** | **0.589** | **+0.024** |
| in-sample AUROC (leaky upper bound) | 0.795 | 0.784 | −0.011 |
| in-sample → holdout **memorization gap** | +0.230 | **+0.195** | **−0.035** |

**Both primary criteria moved correctly. AUROC moved modestly (+0.024) and the
memorization gap shrank (+0.230 → +0.195).** This is exactly the predicted shape:
contamination removed, beliefs de-centered and a touch more honest, ceiling
unmoved. Baton → Pillar B (substrate).

---

## #2 — contamination removed (the core win)

`probe_routing_metrics.py`. "Touches the efficacy spine" = the trial landed a
trial-sourced record on any of `affects / modulates_via / mechanism_affects /
biology_drives`. Failure population = `trial_outcome=failure` (n=233, the FINDINGS
P4 baseline set).

```
                                baseline   routed
failure-trials touching spine   92.3%      30.9%
```

Routing-branch breakdown over the 233 failure-trials (theoretical, from the
classifier's primary mode → `routing_branch_for`):

```
operational   153 (66%)   → CENSORED (was downvoting the spine)
measurement    64 (27%)   → responsibility update (touches spine)
efficacy       10 ( 4%)   → responsibility update (touches spine)
safety          3 ( 1%)   → CENSORED (was downvoting at full weight)
unknown         3 ( 1%)   → legacy full-spread fallback (touches spine)
```

Predicted spine-touch under routing = efficacy+measurement+unknown ≈ **33%**;
measured **30.9%**. The ~2-pt gap is failure-trials whose efficacy/measurement
chains didn't resolve a *live* spine edge (UNKNOWN-placeholder ids), so they touch
nothing even when not censored. The 156 safety+operational failures (67%) that
used to contaminate the spine now apply **zero** virtual evidence to it — this is
the competing-risks censoring, not a down-weight.

This directly removes the "contaminate-in-training, exclude-in-scoring" asymmetry
FINDINGS P4 identified: the held-out 0.565 was scored on the clean efficacy subset
while the beliefs were trained on ~139 contaminating non-efficacy failures.

## #3 — efficacy beliefs de-centered; safety spread widened

Posterior E[p] over evidenced edges (`evidence_strength > 0`), by class. Global
mechanistic base success rate = **0.658**.

```
                baseline                       routed
class        n     mean    sd    med        n     mean    sd    med
efficacy     4567  0.620  0.150  0.575      4513  0.634  0.150  0.586
measurement  1670  0.533  0.111  0.509      1462  0.554  0.111  0.575
safety       3986  0.565  0.111  0.500      3992  0.479  0.176  0.500
modulation    331  0.522  0.129  0.510       331  0.522  0.129  0.510
```

- **Efficacy:** mean **0.620 → 0.634**, moving UP toward the 0.658 base rate — the
  below-base-rate centering FINDINGS P8 flagged is reduced (the spine is no longer
  dragged down by 156 non-efficacy failures). Median 0.575 → 0.586. The sd held at
  0.150 — the centering moved correctly; the spread did not widen materially at
  this n (the spread ceiling is a substrate/Pillar-B problem, not a contamination
  one).
- **Measurement:** mean 0.533 → 0.554 (same de-contamination effect).
- **Safety:** mean 0.565 → 0.479, **sd 0.111 → 0.176**. The safety-gate SURVIVAL
  credit (b += w on readout-reaching trials) is doing real work: gates that
  genuinely fire stay high, gates that always survive drop, so the distribution
  *widens* (more discrimination — the P9 "on-target tox transfers" signal gets
  sharper) while the mean falls (most gates correctly → "doesn't halt"). **Caveat
  below** — the magnitude is aggressive.
- Edge counts fall slightly (efficacy 4567 → 4513, measurement 1670 → 1462): a few
  edges touched *only* by now-censored failures revert to Beta(1,1) and drop out
  of the evidenced set. Expected, consistent with decontamination.

## Holdout AUROC — modest lift, gap shrinks (the expected result)

`scripts/eval_holdout_kfold.py`, K=5, `md5(nct)%k` folds, per-fold re-attribution
of `initial.json` with the fold excluded.

```
                            baseline (off)    routed (on)
AUROC in-sample             0.795             0.784
AUROC holdout (TRUE OOS)    0.565             0.589
in-sample → holdout gap     +0.230            +0.195
holdout binary acc          0.661             0.668
  (TP, TN, FP, FN)          (128,18,56,19)    (138, 9,65, 8)
scorable n                  221 (147s/74f)    220 (146s/74f)
```

- Holdout AUROC **0.565 → 0.589 (+0.024)** — a real but modest lift, exactly as
  pre-stated. The baseline reproduces the documented 0.565 to the digit, so this
  is an apples-to-apples flag flip.
- **Memorization gap shrank +0.230 → +0.195**: in-sample dropped (less
  trial-specific memorization) and holdout rose (cleaner cross-trial signal) — both
  point the same way.
- Binary accuracy ≈ base rate either way (the model still predicts "success" for
  ~84% of trials); this is the substrate ceiling, untouched by A — as predicted.
- Scorable n differs by 1 (220 vs 221): one success trial's chain reverted to
  all-empty-Beta under routing, so the predictor drops it. Folds are otherwise
  identical (same `md5(nct)%k`).

## Leakage guard — verified intact

`eval_holdout_kfold.py:105` pre-marks the held-out fold in
`applied_attribution_trial_ids` so the idempotency guard skips it before the
attribution loop. **All routing logic lives inside `attribute()` /
`attribute_adverse_events`, which are never called for excluded trials** — so a
held-out trial contributes zero evidence regardless of the flag. Confirmed: both
runs printed `Excluding N NCT(s) from attribution` for all 5 folds, with identical
fold membership. No new code path runs outside the guarded attribution for
excluded trials.

---

## Caveats / follow-ups

1. **Safety-survival-credit magnitude is aggressive.** `b += w` with `w = w_base`
   (the full per-trial evidence weight, ~12–37 for a Phase 3) applied to *every*
   existing `causes_ae` gate of the trial's compounds, on *every* readout-reaching
   trial, over-pushes some genuine on-target liabilities (e.g.
   `methotrexate → platelet_count_decreased` 0.84 → 0.29;
   `gemcitabine → abdominal_pain` 0.73 → 0.14 — myelosuppression / GI tox are real
   here). The *aggregate* effect is healthy (safety sd 0.111 → 0.176 = more
   discrimination, holdout AUROC up), but individual edges over-correct. This is a
   **tuning knob, not a correctness bug** — the cleanest follow-up is to down-weight
   the survival count (e.g. a fixed small `n_eff` per gate, matching the EM doc's
   literal "+= 1" rather than "+= w"). Left at spec value for this A/B.
2. **AUROC barely moved — and that's the point.** Per architecture-v2 §A: A makes
   beliefs honest; only B/C move the ceiling. The binding constraint is now
   Disease 2 (substrate starvation). `#2`/`#3` moving while AUROC moves only
   modestly is the informative result that hands off to Pillar B.
3. **Scope held:** only the attribution/update path changed. `_aggregate_samples`,
   the softmin (A1), and the triangle factor (A2) are untouched. Flag OFF is
   byte-identical (full suite 1394 passed; `test_flag_off_is_identity`).

## Artifacts

- Map: `src/annotation/taxonomy.py` (`RoutingBranch`, `FAILURE_MODE_BRANCH`,
  `routing_branch_for`; `python -m src.annotation.taxonomy` prints it).
- Update path: `src/annotation/attributor.py` (`_routing_enabled`,
  `_apply_responsibility_update`, `_credit_safety_survival`,
  routing block in `_condition_chain_on_outcomes`).
- Tests: `scratch/diagnostics/test_routing.py` (19 pass).
- Snapshots: `data/exports/multi_500_{baseline,routed}_reattr.json` (regenerable).
- Logs: `scratch/diagnostics/a34_{baseline,routed}_reattr.log`,
  `a34_metrics.txt`, `a34_kfold_{baseline,routed}.txt`.
- Runners: `scratch/diagnostics/run_a34_eval.sh`, `run_a34_kfold.sh`,
  `probe_routing_metrics.py`.
