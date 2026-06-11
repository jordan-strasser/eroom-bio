# Eroom Bio — findings summary (graph-fix arc → diagnosis → triangulation)

**Date:** 2026-06-10/11 · branches `fix/st-field-faithfulness`, `arch/triangulation-edge-weights`
**Scope:** what we tried, what we found, the evidence, and the reasoning — across the
graph-fix work (Phases A/B/C, T2, T5), the architecture diagnosis, and the new
triangulation/conditional experiments. Confidence scale: **HIGH ≥80% · MED-HIGH 65–80%
· MED 50–65% · LOW <50%**.

---

## 0. The goal and what "working" means

**Goal.** Decompose trials into mechanistic causal-chain belief updates on a shared
graph so a mechanism learned in one indication informs another → compositional,
interpretable P(success) that RISES as the corpus grows.

**"Working" (measurable).** Honest out-of-sample AUROC that (a) beats base-rate ranking
and (b) rises with n; plus (c) cross-indication transfer. The leakage-free measurement is
k-fold re-attribution and leave-one-group-out (NOT same-corpus leave-one-trial-out, which
is optimistic — this trap recurs below).

---

## 1. Headline

- **The deployed predictor does not yet meet the goal.** Honest holdout AUROC is flat
  near chance (~0.53) while in-sample is ~0.80 — it memorizes each trial from its own
  evidence and does not generalize. (HIGH)
- **The failure is REPRESENTATIONAL, not data-cleanliness and not "impossible goal."**
  We cleaned ingestion/mapping/merge (causes #1/#2/#4) and the honest curve did not move;
  but changing the prediction REPRESENTATION moved the honest number by ~0.08–0.18. So the
  binding constraint is the predictor (#6) + belief-formation (#3). (HIGH)
- **But no approach yet SIGNIFICANTLY beats the deployed graph at our scale.** At n=250
  (107 scorable) every method clusters 0.53–0.65 with overlapping CIs and paired-DeLong
  p>0.1. The promising mechanistic signal (coarse composition, 0.715 leave-one-target-out)
  and the "conditional 0.64–0.71" both need n=500 to be confirmed or killed — and the
  conditional already shrank to ~0.53–0.62 once evaluated honestly. (HIGH on
  "underpowered + conditional was optimistic")
- **Uncomfortable benchmark:** non-mechanistic DESIGN features (enrollment, single-arm,
  #compounds, rationale) predict success NUMERICALLY BEST (~0.68–0.70) and no mechanistic
  model clearly beats them. This bounds the program. (MED-HIGH)

---

## 2. The evidence arc (each step: what, evidence, reasoning)

### 2.1 Graph-fix work — cleaned the data layers; the honest curve did NOT move
- **Phase A** (`2fe4f4c`): hierarchical backoff partial pooling (sparse indication leaf
  borrows its parent instead of being shadowed). Cross-indication strength-borrow fires
  (her2_breast 0.51→0.79).
- **Phase B** (`991a25b`): mechanism = curated Reactome pathway FAN-OUT, id-merge.
  Collapsed the T3-measured 65–72% mechanism over-merge to clean id-merge; cross-target
  pathway convergence materializes (17 @ n=50).
- **Phase C honest holdout** (`11a9b0c`): k-fold re-attribution on clean fan-out builds.
  **Evidence:** holdout AUROC 0.500 / 0.557 / 0.534 at n=50/100/250; in-sample steady
  ~0.80. **Reasoning:** memorize-but-no-transfer; cleaning the mechanism layer left the
  honest curve flat → the ceiling is NOT in the mechanism/merge layer. (HIGH)
- **T2** (`3fd9268`): per-edge attribution-math experiment. **Evidence:** the success/
  failure SPLIT rule is 2nd-order (in-sample AUROC 0.94–0.97 across modes; spread
  unchanged); `effect_size` is unusable as extracted (one float conflating HR/%/count,
  range −1e5…2.4e6). **Reasoning:** tuning the edge-split doesn't help → the problem isn't
  the per-edge update knob.
- **T5** (`20dcaf4`): entity-merge SapBERT verification. **Evidence:** Indication/
  Endpoint/Population tiers verified safe at 0.80 (0 synonym misses, siblings separated);
  Target id-merge validated (paralog siblings would over-merge under SapBERT); AdverseEvent
  SapBERT tier was inert + a re-merge footgun → removed.

### 2.2 Cross-indication transfer — re-slicing the SAME objective doesn't escape the ceiling
- **Leave-one-INDICATION-out** (`scripts/cross_indication_transfer.py`): hold out a whole
  indication, predict it from the rest. **Evidence:** pooled AUROC **0.468**. **Reasoning:**
  the holdout split changes, the belief-formation ceiling doesn't — confirmed the owner's
  intuition that this is still an AUROC problem, not a new lever. (MED — the LOIO had an
  empty-edge confound, later subsumed by the honest harness numbers in 2.4.)

### 2.3 Conditional-overlap probe — looked like a breakthrough, WAS leave-one-trial-out optimism
- **Case-based pooling** (`scripts/conditional_overlap_experiment.py`), leave-one-trial-out:
  biology ≫ mechanism as the unit (marginal_bio 0.57–0.63 vs marginal_mech 0.48–0.53), and
  biology × coarse line-of-therapy reached **0.64–0.71**. **Reasoning at the time:** the
  signal lives at the biology node conditioned on line-of-therapy, not the mechanism edge
  marginal. **CORRECTION (2.4):** the 0.64–0.71 was same-corpus LOO — optimistic. On the
  honest k-fold / leave-group-out harness the SAME model is 0.624 / 0.532 / 0.423. The
  lift is real but modest, not a breakthrough. (HIGH on the correction.) *Methodological
  lesson: the documented LOO-optimism trap recurred; only k-fold/leave-group-out is honest.*

### 2.4 Triangulation — the owner's "no LLM beliefs" architecture, evaluated honestly
- **Model:** `logit P(success)=Σ_{e∈maximal-composition} w_e`, L2-logistic; coefficients
  ARE the edge weights, triangulated jointly. Evaluated on the blessed harness (k-fold +
  leave-one-indication/target-out + paired DeLong vs the deployed graph), n=250.
- **Evidence (honest AUROC):** deployed graph 0.534; granular `edges(comp)` 0.560;
  **coarse `edges(coarse)` 0.615 k-fold / 0.715 leave-one-TARGET-out (+0.053 Brier
  skill)**; conditional 0.624/0.532/0.423; categorical logreg:mech 0.648; design xgb
  0.675–0.698. **Paired DeLong vs graph: ALL p>0.1** (edges(comp) p=0.73, edges(coarse)
  p=0.32, cond p=0.29, mech p=0.11).
- **Reasoning:**
  1. **Coarsening the fan-out is the clearest mechanistic signal** — collapsing the
     mechanism-pathway fan-out (compound→target→biology→indication) beats the granular
     version everywhere, and on leave-one-TARGET-out reaches 0.715 with real skill: when a
     whole target is unseen, the coarse composition generalizes via SHARED biology — the
     structured transfer the thesis predicts. (MED-HIGH that coarse>granular; MED that the
     0.715 holds at scale — n=107.)
  2. **The granular pathway edges are entangled NOISE.** A gene's ~8 Reactome pathways
     always co-occur → their edges share an identical traversal set → individually
     unidentifiable (Bevacizumab→VEGFA's 9 edges split +0.104 each; EGFR's 8 split −0.194).
     This is exactly the owner's predicted entanglement, and it explains coarse>granular.
     (HIGH — directly measured.)
  3. **At n=250 nothing is significant.** 107 scorable, ±0.10–0.12 CIs, all paired p>0.1.
     The honest verdict is "underpowered," not "winner found." (HIGH)
  4. **Design features are the benchmark to beat.** (MED-HIGH)

---

## 3. The six root causes — what's ruled out, what's open

| cause | status | confidence |
|---|---|---|
| #1 ingestion noise | cleaned over prior sessions; not the binding constraint | HIGH |
| #2 data→node mapping | cleaned (codename/canon/target resolution); not binding | HIGH |
| #3 per-trial edge-assignment | PART of the problem — beliefs are context-free MARGINALS; conditioning helps modestly | MED-HIGH |
| #4 node merge | T3 found mechanism over-merge (real) → Option 2 fixed it; fixing it did NOT move the curve ⇒ not binding | HIGH |
| #5 graph structure / empty edges | empty edges drag the softmin product; partial (the composition model bypasses this) | MED |
| **#6 predictor query / representation** | **the binding constraint** — noisy-AND of context-free per-edge marginals off the mechanism hub; changing it moved the honest number | HIGH |

---

## 4. What carries signal (honest, calibrated)

- **Biology > mechanism as the unit** (marginal_bio 0.57–0.63 vs marginal_mech 0.48–0.53,
  and coarse>granular). The mechanism/pathway layer is a shared hub at base rate. (HIGH)
- **The coarse composition shows structured transfer** (0.715 leave-one-target-out, real
  skill). (MED — needs n=500.)
- **Line-of-therapy conditions outcome within a mechanism** (early-line beats late-line
  Δ+0.14 descriptively) but adds only a modest honest predictive lift. (MED)
- **Design/operational features carry the most predictable variance** — and the graph
  doesn't model them. (MED-HIGH)
- **What does NOT carry signal:** the deployed Beta+softmin off mechanism edges (0.534);
  the granular pathway-edge triangulation (entangled); per-edge attribution-split tuning;
  re-slicing the holdout (cross-indication 0.468). (HIGH)

---

## 5. Methodological lessons (load-bearing)

1. **Same-corpus leave-one-trial-out is optimistic** — it inflated the conditional probe
   from an honest ~0.55 to a mirage 0.64–0.71. Use k-fold re-attribution / leave-group-out.
2. **p≫n is a trap.** The fan-out gave 1892 edge-features for 142 trials → in-sample 1.000,
   unstable held-out signs (AUROC fell below 0.5). Fixed by honest eval + coarsening.
3. **Reuse the blessed harness** (`eval_baselines.py`) so every model is scored on
   IDENTICAL folds with paired DeLong — the only valid comparison at n≈107.
4. **At our n, read CIs and paired tests, not point estimates.** Most "wins" this arc were
   within noise.

---

## 6. Open levers (ranked)

1. **Confirm the coarse composition at n=500** (paired DeLong vs graph + design). If it
   separates with significance → replace Beta+softmin with the coarse-composition logistic.
   This is the single cleanest next experiment. (the decision gate)
2. **Model design/operational features explicitly** (the benchmark the mechanism can't yet
   beat) — and ask whether mechanism adds anything ON TOP via paired tests.
3. **Conditional structure done right** — bake biology × coarse-line conditioning into the
   chosen predictor (hierarchical backoff = Phase A `pool_hierarchical`), measured honestly.
4. **Accept a reframed metric** if n=500 shows no mechanistic separation: the graph's value
   may be interpretable decomposition + bottleneck attribution, not binary-success AUROC.

---

## 7. Artifacts
- Diagnosis: `data/dev/architecture_diagnosis.md`. Honest holdout curve:
  `data/dev/phasec_honest_holdout_findings.md`. Conditional probe:
  `data/dev/conditional_overlap_findings.md`. Triangulation (honest):
  `data/dev/triangulation_findings.md` + weights `data/dev/triangulation_weights.md`.
- Tools: `scripts/{eval_baselines, eval_holdout_kfold, conditional_overlap_experiment,
  cross_indication_transfer, triangulation_experiment, phasec_build_eval}.py`.
- Branches: `fix/st-field-faithfulness` (graph-fix, T2, T5, Phase C — pushed),
  `arch/triangulation-edge-weights` (triangulation — pushed). No PRs (owner reviews).
