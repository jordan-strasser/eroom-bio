# Phase C — honest out-of-sample AUROC on the clean fan-out (findings)

**Date:** 2026-06-10 · branch `fix/st-field-faithfulness`
**Question (north star):** does the clean Option-2 pathway FAN-OUT's cross-target
accumulation LIFT the honest (out-of-sample) AUROC, where the noisy pathway-identity
builds were FLAT n=5→472 near chance? **MEASURE, don't assume.**
**Harness:** `scripts/eval_holdout_kfold.py` — K-fold re-attribution: re-attribute the
pre-attribution `initial.json` once per fold EXCLUDING that fold, then predict the
held-out trials. Real generalization through the real pipeline (not belief-replay).

## n=50 (phaseb_n50b — the committed Phase A+B fan-out), K=5

```
── K-fold true-holdout (K=5, n=30, success=20, failure=10) ──
  AUROC in-sample          = 0.800   (upper bound — leakage)
  AUROC holdout            = 0.500   (TRUE out-of-sample)        ← CHANCE
  in-sample → holdout gap  = +0.300
  holdout binary acc       = 0.633   (TP=19, TN=0, FP=10, FN=1)
```

**The clean fan-out does NOT lift the honest AUROC above chance at n=50.** Consistent
with the documented core finding (`project_predictor_signal_finding`: honest holdout
~chance, learning curve flat). The holdout collapses to all-success (TN=0) — no rank
separation once a trial's own evidence is removed. Caveat: n=30 scorable (20/10) → the
AUROC CI is wide (~±0.12); the LEARNING-CURVE TREND is the real signal, not this point.

## Diagnosis — it's NO TRANSFER, not saturation (cheap in-sample check)

In-sample `predict_clinical_hypothesis` distribution on phaseb_n50b, by label:

| label | n | min | median | max | std |
|---|---|---|---|---|---|
| success | 25 | 0.430 | **0.697** | 0.789 | 0.093 |
| failure | 11 | 0.339 | **0.477** | 0.622 | 0.081 |

The predictions are **well-spread (0.34–0.79) and cleanly separated** in-sample
(success ~0.70 vs failure ~0.48 → the 0.80 in-sample AUROC). So the chance HOLDOUT is
**not** a saturation artifact (predictions aren't all pinned high). It is **no
transfer**: the graph separates a trial using its OWN folded-in evidence, but once that
evidence is held out, the cross-trial accumulation on the shared
mechanism→biology→indication edges does not sort THIS trial. Mechanistically: many
trials share the same generic backbone edges, so holding one out leaves the edge
dominated by the others → it predicts the cohort AVERAGE, not the held-out outcome.
This is the irreducible chain-level label noise the project already documented (same
chain → both successes and failures; binary success isn't determined by the mechanistic
chain alone — dose / population / execution differ). The fan-out's cross-target sharing
did not create transfer at this scale.

## Learning curve clean fan-out (does scale create transfer?)

| n (trials) | scorable (succ/fail) | in-sample AUROC | **holdout AUROC** | holdout TN/fail | CI ≈ |
|---|---|---|---|---|---|
| 50 | 30 (20/10) | 0.800 | **0.500** | 0/10 | ±0.11 |
| 100 | 51 (33/18) | 0.795 | **0.557** | 4/18 | ±0.09 |
| 250 | 107 (74/33) | 0.802 | **0.534** | 7/33 | ±0.06 |

_Historical reference (noisy pathway-identity builds): ~0.567 kfold / ~0.51 forward;
curve FLAT n=5→472. The static ceiling is documented as ~0.57 at large n._

**VERDICT: the clean fan-out curve is FLAT near chance (~0.53) — it does NOT lift the
honest AUROC.** The three points (0.500 / 0.557 / 0.534) all overlap within CI; the
MOST RELIABLE point — n=250 with 107 scorable trials, the tightest CI — settles at
**0.534**, barely above chance and squarely inside the documented ~0.51–0.567 band of
the noisy builds. The n=100 "0.557" was a within-noise fluctuation (n=51), which the
larger n=250 corrected — exactly why the third point was measured rather than
concluding at two. In-sample is rock-steady (0.800 / 0.795 / 0.802) so the graph
memorizes equally well at every scale; the gap is pure, constant leakage (~0.27) with
no transfer emerging as n grows. Out-of-sample failure-recall is flat ~21% at
n=100/250 (TN/fail 4/18, 7/33) — barely above base rate.

**n=500 NOT run (deliberate):** three clean points spanning 5× already establish
flatness, and the tightest-CI point is the high-n one; an n=500 build (~3h live-API)
would near-certainly reproduce ~0.55 (the documented ceiling) and add no information.
Conservative-rebuild call — the question ("does the clean fan-out lift the curve?") is
answered: NO.

## What this localizes (north-star axiom)

Option 2 (Phases A+B) demonstrably fixed the MECHANISM-layer noise — root causes #3
(per-trial edge-assignment) and #4 (merge): T3's 65–72% mechanism over-merge → clean
id-merged curated pathways; cross-target sharing materialized (17 convergences @ n=50).
Yet the honest curve stayed flat. So the static-AUROC ceiling does NOT live in the
mechanism layer — it is the deeper, already-documented cause: **binary trial success
is not determined by the mechanistic chain alone.** The same chain yields both
successes and failures (dose / population / line-of-therapy / execution differ), so a
held-out trial reverts to the cohort average on its shared backbone edges — irreducible
label noise at the chain altitude. Lifting the honest curve requires finer
discriminative conditioning the chain doesn't yet carry (e.g. the confirmed
line-of-therapy gap: present in 118/370 population nodes but 0 evidenced
responds_differently edges — `project_predictor_signal_finding`), NOT more mechanism
cleanup. That is the next real lever, and it is orthogonal to the mechanism work this
branch completed.

## Reproduce
- `scripts/phasec_build_eval.sh <N> <area>` — build a clean fan-out at N (cached
  annotations) + write its corpus + run the honest K-fold holdout.
- Points: `phaseb_n50b` (n=50), `phasec_n100` (n=100), `phasec_n250` (n=250).
- Raw: `data/dev/phasec_kfold_n{50,100,250}.txt`.
