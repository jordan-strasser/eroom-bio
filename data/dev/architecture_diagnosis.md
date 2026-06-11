# Architecture diagnosis — why the current system does not meet the north-star goal

**Date:** 2026-06-11 · branch `fix/st-field-faithfulness`
**Purpose:** a clear-eyed post-mortem with exact definitions and calibrated confidence
on each diagnosis. Confidence scale: **HIGH ≥80%** (strong, replicated, direct
evidence) · **MED-HIGH 65–80%** · **MED 50–65%** · **LOW <50%**.

---

## 0. TL;DR

- **The DEPLOYED architecture does not meet the goal.** Honest out-of-sample AUROC is
  flat near chance (~0.53) while in-sample is ~0.80 → the system MEMORIZES each trial
  from its own evidence and does not GENERALIZE. (HIGH)
- **The goal is NOT falsified.** A case-based probe shows the predictive signal exists,
  is estimable, and is scale-rewarded — at a DIFFERENT altitude than we built on
  (BIOLOGY node × coarse line-of-therapy), reaching honest LOO AUROC 0.64–0.71, the
  first clean break above the project's long-standing ~0.53–0.57 ceiling. (MED-HIGH)
- **So the diagnosis is "wrong representation," not "impossible goal," and not the
  layers we spent sessions cleaning.** The failure localizes to root causes **#6
  (predictor query/representation)** and **#3 (per-trial belief-formation)** — NOT
  #1 ingestion / #2 mapping / #4 merge, which were cleaned and did NOT move the
  honest number when fixed. (HIGH on the localization)

---

## 1. Exact definitions

**North-star goal.** Decompose each trial into mechanistic causal-chain belief updates
on a shared graph so that a mechanism learned in one indication informs another →
compositional, interpretable P(success) that IMPROVES as the corpus grows.

**"Working" (measurable).** (a) HONEST out-of-sample AUROC > base-rate ranking and
RISING as n=10→N; (b) cross-indication transfer (a belief learned in indication X
ranks indication Y's trials); (c) interpretable bottleneck attribution. This document
is about (a)+(b); (c) was never the failing part.

**Deployed architecture (the object under diagnosis).**
- *Belief.* Every graph EDGE carries one scalar Beta(α,β). A trial's per-arm OUTCOME
  conditions its whole chain by edge id: SUCCESS pushes every backbone edge up;
  FAILURE spreads a modest contradict across the chain (explaining-away). The edge
  belief is therefore the **MARGINAL** success-association of that edge, averaged over
  EVERY context the edge ever appeared in.
- *Prediction.* P(success) = **softmin (weakest-link noisy-AND)** over the chain's edge
  Betas (MC-sampled) × (1 − safety_penalty). The mechanism-layer edges
  (target→mechanism, mechanism→biology) sit on a MechanismNode = a curated Reactome
  pathway (Option 2), which is SHARED across many drugs/trials.

**In-sample vs honest holdout.** *In-sample* scores a trial against a graph that
INCLUDES that trial's own evidence (leakage → optimistic). *Honest holdout* (K-fold
re-attribution, `eval_holdout_kfold`) removes a trial's evidence, re-attributes, then
predicts it — real generalization. The GAP between them is the leakage/memorization.

**Marginal vs conditional belief.** *Marginal* P(success | edge) averages over context.
*Conditional* P(success | edge, context=c) keeps the interaction (e.g. mechanism ×
line-of-therapy). The deployed system can represent only the marginal.

---

## 2. The failure, stated precisely

| measurement | in-sample AUROC | HONEST holdout AUROC | source |
|---|---|---|---|
| chain prediction, n=50 | 0.80 | **0.500** | Phase C kfold |
| chain prediction, n=100 | 0.80 | **0.557** | Phase C kfold |
| chain prediction, n=250 | 0.80 | **0.534** | Phase C kfold |

- **In-sample is high and flat (~0.80); honest holdout is near chance and flat
  (~0.53) across a 5× scale span.** The ~0.27 gap is pure leakage. (HIGH)
- The deployed predictor on held-out trials collapses toward predicting the cohort
  AVERAGE — it ranks almost everything one class (n=50: TN=0/10 failures caught).
- This is **memorization without transfer**, and crucially it **did not improve when
  the merge/mechanism layers were cleaned** (Option 2 fixed 65–72% mechanism
  over-merge; the honest curve did not move). So the binding constraint was never the
  data-cleanliness causes. (HIGH)

---

## 3. Root-cause diagnoses (with confidence)

### D1 — Prediction reads off the WRONG UNIT (mechanism hub, not biology). **HIGH (~85%)**
*Definition.* P(success) is composed largely off the MechanismNode (curated Reactome
pathway). A pathway is a HUB shared by many drugs/indications, so its MARGINAL
success-association ≈ the base rate ("in everything" → belief ~uninformative). The
discriminative signal lives one layer down, at the BIOLOGY node (the specific
downstream process).
*Evidence.* Case-based holdout: pooling outcomes at `{mechanism}` gives AUROC 0.48–0.53;
at `{biology}` gives 0.57–0.63 — a +0.09 gap, replicated at n=100 and n=250.
*Why confidence isn't higher:* the unit advantage is measured on a case-based probe,
not yet in the graph's own composition.

### D2 — Edge beliefs are context-free MARGINALS; success is CONDITIONAL. **HIGH that conditioning helps (~80%); MED-HIGH that it's a primary cause (~70%)**
*Definition.* Each edge is one scalar averaged over all contexts. The SAME mechanism/
biology succeeds early-line and fails late-line; the marginal averages these to the
middle and erases the discriminative interaction. The architecture has NO slot for the
mechanism × context interaction. Line-of-therapy is in fact STRUCTURALLY DROPPED from
P(success) today (line present in 118/370 population nodes but 0 evidenced
`responds_differently` edges).
*Evidence.* (i) Descriptive: within a single mechanism, early-line trials beat late-line
by Δ +0.04 / +0.27 / +0.14 (n=50/100/250) — same mechanism, different line, different
outcome. (ii) Predictive: conditioning the biology pool on coarse line-of-therapy lifts
honest LOO AUROC +0.07–0.09 over the biology marginal (`bio_linegrp` 0.711 @ n=100,
0.636 @ n=250), robust to the smoothing threshold and SPECIFIC to line (severity +0.008,
stage +0.011 do not help). *This is the most direct evidence the failure is
representational, not irreducible.*

### D3 — Composition = softmin noisy-AND of INDEPENDENT edges. **MED-HIGH (~70%)**
*Definition.* P(success) = product/softmin over edges assumes the links are independent
and that the weakest gates success. Consequences: (a) an EMPTY Beta(1,1)≈0.5 edge (or a
hub edge) drags the product and can dominate the min; (b) the form cannot express "the
chain works EXCEPT when context = x" — there is no interaction term, only a product of
per-edge marginals.
*Evidence.* The case-based predictor uses NO composition (just conditional pooling) and
beats the deployed softmin. The leave-one-INDICATION-out test was actively confounded by
empty indication edges washing out the upstream signal under min (pooled 0.468). The
softmin is entangled with D1/D2, so it is hard to isolate its independent contribution —
hence not HIGH.

### D4 — Per-trial attribution spreads outcome over EDGES, not CONTEXTS. **MED (~60%)**
*Definition.* A trial's outcome updates every backbone edge; explaining-away decides
WHICH edge in the chain absorbs a failure, never WHICH CONTEXT the chain fails in. So a
wrong-population failure down-votes shared mechanism/biology edges that are mechanistically
fine — the owner's motivating example.
*Evidence.* T2 measured the per-edge SPLIT rule (explain-away vs symmetric variants) as
SECOND-ORDER for separation — i.e. tuning the edge split doesn't help, consistent with
the real problem being the MARGINAL/context axis (D2), not the edge axis. Bypassing
attribution entirely (case-based pooling) does better. Confidence is only MED because D4
is largely a restatement of D2 from the attribution side, not an independent lever.

### D5 — A large IRREDUCIBLE non-mechanistic component of trial success. **MED (~55%), and FALLING**
*Definition.* Trials fail for dose, enrollment, operational, competitive, statistical-power
reasons orthogonal to the mechanism; the chain cannot encode these, imposing a ceiling on
ANY chain-based predictor.
*Evidence.* Even the best conditional probe (0.64–0.71) is far from 0.9; the whole project
sits in a 0.50–0.71 band. BUT the conditional result LOWERED my confidence here: a
representation change moved the honest number ~0.10–0.18, which a dominant-irreducible-noise
world would not allow. So an irreducible component is real but is NOT the dominant ceiling
at current scale. (This is the diagnosis I am least certain about.)

### What it is NOT (the binding constraint). **HIGH (~85%)**
Causes #1 (ingestion noise), #2 (data→node mapping), #4 (node merge) were each cleaned
over prior sessions AND this one (T5 verified the entity merges; Option 2 + T3 fixed the
65–72% mechanism over-merge). Fixing #4 did NOT move the honest curve. So while these had
real defects, they were not the binding constraint on the north-star metric. The remaining
binding constraints are #6 (predictor/representation) and #3 (belief-formation).

---

## 4. Condensed recap — what was tried and found (this arc)

| work | what | finding |
|---|---|---|
| Phase A | hierarchical backoff partial pooling (indication/pop) | cross-indication strength-borrow works; honest delta deferred |
| Phase B | mechanism = curated Reactome pathway FAN-OUT (id-merge) | fixed mechanism over-merge (65–72%→clean); cross-target sharing real |
| Phase C | honest K-fold holdout on clean fan-out, n=50/100/250 | **flat ~0.53** (0.500/0.557/0.534); in-sample steady 0.80 → memorize, no transfer; n=500 skipped (flat established) |
| T2 | attribution-math experiment (split modes; effect_size/p) | split is 2nd-order; `effect_size` unusable as extracted (use p_value only) |
| T5 | entity-merge SapBERT verification | Indication/Endpoint/Population safe; Target id-merge correct; AE SapBERT tier inert+footgun → removed |
| cross-indication | leave-one-INDICATION-out transfer test | pooled **0.468** — re-slicing AUROC doesn't escape a belief-formation ceiling |
| conditional-overlap | case-based pooling, no edge attribution | **biology >> mechanism (0.57–0.63 vs 0.48–0.53); biology × coarse line = 0.64–0.71** — first break above ceiling |

Numbers to anchor on: deployed honest holdout **~0.53**; conditional biology×line probe
**0.64–0.71** (honest LOO, n=64/142); chance 0.50; in-sample ceiling ~0.80.

---

## 5. What the evidence says about the GOAL (honest)

- The north-star goal is **not falsified.** The compounding-knowledge signal EXISTS, is
  ESTIMABLE out-of-sample, and is SCALE-REWARDED (the conditional probe FAILS at n=36 —
  too little overlap — and WORKS at n≥64; the right representation makes the curve rise
  with n, which is the literal north-star tell). (MED-HIGH)
- What is (provisionally) FALSIFIED is the specific DEPLOYED representation: **a
  noisy-AND of context-free per-edge marginal Betas read off the mechanism hub.** That
  representation cannot encode the two things that carry the signal — the right unit
  (biology) and the dominant conditioner (line-of-therapy). (HIGH)
- Honest ceiling caveat: the best measured honest number is ~0.64–0.71 at modest n, not
  0.9. Some irreducible non-mechanistic noise (D5) is real. The achievable target is
  "meaningfully above base rate and rising with n," not "near-perfect." (MED)

---

## 6. Implied direction (the user decides; stated for completeness)

Re-architect prediction from "product of mechanism-edge marginals" to **conditional
outcome-pooling at the biology node, conditioned on coarse line-of-therapy, with
hierarchical backoff for sparsity** (the `beliefs.pool_hierarchical` machinery from
Phase A, applied to the chain-context hierarchy instead of the indication hierarchy).
Keep the mechanism/biology nodes for interpretability + cross-indication accumulation,
but stop COMPOSING P(success) off mechanism-edge marginals. Confirm the lift at n=500
before committing the rebuild (current n is modest; CI ≈ ±0.08–0.12).

*Tools that produced this: `scripts/{eval_holdout_kfold, cross_indication_transfer,
conditional_overlap_experiment}.py`; findings `data/dev/{phasec_honest_holdout,
cross_indication_transfer, conditional_overlap}_findings.md`.*
