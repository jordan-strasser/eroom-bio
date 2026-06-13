# eroom.bio — Architecture v2 strategy

Built on the diagnostic (`FINDINGS.md`, honest holdout AUROC **0.565**, in-sample 0.795, gap +0.231 = memorization). Maps each confirmed root cause to a concrete change, sequenced cheapest-highest-value first, with explicit notes on what each change **will and will not** buy.

## The frame: two diseases, two cures

| | Disease 1 — **contamination** | Disease 2 — **substrate starvation** |
|---|---|---|
| what | one outcome bit smeared across an undifferentiated, class-blind backbone; non-efficacy failures train the efficacy edges | 1.24 trials/edge (efficacy), 71% singleton biology nodes; nothing to transfer through |
| evidence | P1, P2, P4, P8 | P6, P3, P7 gap |
| cure | inference: class-split + reason-routed EM with censoring | representation + statistics: coarser canonicalization + hierarchical pooling; ultimately more/curated data |
| ceiling effect | removes active poisoning, recovers the *clean* ceiling of current data | **raises** the ceiling |

Critical point: **Pillar A cleans contamination but cannot break the substrate ceiling.** Expect A to move 0.565 up modestly and, more importantly, to make the beliefs *honest*. Only B and C move the ceiling. Do A first anyway — it's nearly free, it stops active harm, and it tells you the true clean baseline before you spend on data.

---

## Pillar A — Reason-routed EM with censoring + class split
*Fixes Disease 1. No new data. Highest value/effort ratio.*

**A1. Split efficacy from measurement.** Today `_chain_backbone_edges` (`attributor.py:257-263`) and `_collect_edges` (`path_query.py:1091`) treat `affects/modulates_via/mechanism_affects/biology_drives` (efficacy) and `reflects_biology/endpoint_captures` (measurement) identically — same conjugate update, same softmin. Add an `edge_class ∈ {efficacy, measurement, safety}` field to `EdgeBeliefState` (`models.py:824`) and let the aggregator use the structured form:
```
P(success) = Φ (efficacy AND)  ×  D (detection)  ×  s (safety survival)  ×  ℓ
Φ = softmin/∏ over efficacy spine     (keep weakest-link if it ablates better)
D = v · w     (triangle factor v × population edge w)
s = ∏(1 − q_k)   (already ~built: path_query.py:655, capped max → make it the product)
```
This replaces the single softmin over efficacy+measurement (`path_query.py:293-333`).

**A2. Collapse the triangle to one factor.** The biology–endpoint–indication triangle is currently 3 independent Betas (1566 of them), double-counted under the geomean branch. Replace with one clique factor `v` indexed by the (biology, endpoint, indication) triple. Removes the double-count and the false independence. This is the correct version of the triangulation change from last night.

**A3. Route training updates by failure reason (the untested lever).** The 13-category taxonomy (`taxonomy.py:16-29`) is extracted then discarded except for a leaky binary operational gate (`taxonomy.py:475`, 0.2 vs 1.0). Wire the reason into the update as **competing-risks censoring** (EM doc §3.2):

| reason | safety AE | efficacy spine | measurement | leak |
|---|---|---|---|---|
| success | survived (b+1) | held (α+1) | held (α+1) | ok |
| safety / DLT | fired, split by ρ_k | **censor** | **censor** | censor |
| efficacy / futility | survived (b+1) | blame within Φ | blame within Φ | ok |
| commercial / underpowered / insufficient-info | censor | **censor** | **censor** | (leak) |
| unknown | full ρ (§3.1) | full ρ | full ρ | full ρ |

The immediate win: stop training the spine on the **68% of failures that aren't efficacy failures**. This directly removes the "contaminate-in-training, exclude-in-scoring" asymmetry (`eval_holdout_compose.py:482-487`).

**A4. Generalize explaining-away into principled responsibility.** The existing `u_i = 1 − E[p_i]` split (`attributor.py:942-958`) is a heuristic version of `ρ_a = (1−r_a)/(1−π)`. Swap in the normalized responsibility with the branch-local denominator (1−Φ for efficacy deaths, 1−s for safety). Same code shape, correct math, and now class-aware.

**What A buys:** honest, un-poisoned beliefs; safety no longer invisible to the metric; the triangle stops double-counting. **What it won't buy:** transfer that isn't in the data. If holdout barely moves after A, that's not failure — it's A telling you Disease 2 is now the binding constraint.

---

## Pillar B — Substrate: canonicalization + hierarchical pooling
*Fixes Disease 2. Representation + statistics, still no new data needed for the first two moves.*

**B1. Re-canonicalize biology to an ontology (biggest single lever on sparsity).** Biology nodes are `bio:<content-hash>` → 71% singletons → edges can't recur → nothing transfers. Mechanism works precisely because it's Reactome-id-merged (1% trial-scoped). Map biology to a controlled vocabulary (GO biological process, MONDO/EFO disease-axis, or Reactove-adjacent) so the same biology across trials lands on the same node. This alone should lift efficacy reuse off 1.24.

**B2. Hierarchical partial pooling (mandatory at this sparsity).** A single-trial edge reverting to the global base rate is the memorization mode. Put a hierarchy on it: `edge | context | class`, so a leaf with one observation shrinks toward its context/class siblings instead of toward the cohort mean. With 1.24 trials/edge this isn't an enhancement, it's the only way leaves get informative posteriors. Implement as class- and context-level Beta hyperpriors feeding the per-edge `α,β`.

**B3. Context-condition the pooling (folds in P3).** Only `mechanism_affects` is tissue-conditioned at query time (`path_query.py:1125`); everything else pools context-free, so 86% of reused edges mix ≥2 indications and 40% pool mixed outcomes under one Beta (e.g. `DNA→dna_damage`: 35 trials, 27 indications, one number). Add indication/modality as the grouping level in B2's hierarchy rather than collapsing it.

**What B buys:** raises the ceiling A exposed. **What it won't buy:** if biology re-canonicalization over-merges, you trade sparsity for context collapse — watch the P3 heterogeneity metric as you coarsen.

---

## Pillar C — Re-target to where signal and substrate coexist
*Strategic, not code. Decides what to collect and what to claim.*

**C1. Safety-via-shared-target is the strongest demonstrated signal, and it's corpus-starved.** P9: within-target AE posterior SD 0.048 (vs 0.134 efficacy) — on-target tox genuinely transfers. But the corpus has **3 DLT failures**, and `_resolve_label` routes safety-driven failures to "ambiguous" and drops them from scoring. So: (a) stop excluding safety failures from the label space, and (b) enrich the next data pull for safety-stopped / black-box / DLT-terminated trials. This is the cleanest place to *prove* cross-trial learning works.

**C2. Make the decomposition the product, not the scalar.** Binary accuracy already equals the base rate (predicts "success" 84% of the time). At n≈500 the honest near-term claim isn't "we predict win/lose" — it's "we attribute *which* risk dominates (biology vs endpoint vs population vs target-tox) with calibrated, pooled, cross-trial evidence." Expose `Φ̂, D̂, ŝ` and the top blamed edges. That's defensible value the AUROC can't capture and the federated-data thesis can be priced on.

**C3. Scale n with intent.** Sparsity is partly just n≈500. But scale *toward reuse* — more trials on shared targets/mechanisms/indications beats more breadth. Curated depth on a few target families will identify edges that broad sampling never will.

---

## Sequencing

1. **A3 + A4 first** (reason-routing + censoring + responsibility). Cheapest, stops active poisoning, recovers the clean baseline. One day of work; the taxonomy already exists.
2. **A1 + A2** (class split + triangle factor). Structural correctness; modest lift.
3. **Synthetic harness in parallel** (EM doc §8): planted `r_a, q_k, ℓ` with deliberate edge-sharing. This is the control that separates "model wrong" from "data can't identify it" — essential before trusting any real number, and the only way to know if B is working.
4. **B1 + B2** (canonicalization + pooling). The ceiling-raisers. Re-measure P6 reuse and P3 heterogeneity after.
5. **C1 corpus enrichment** for safety, once A/B have maximized the current data.

## Expectation-setting (say this out loud before running)
- A makes beliefs **honest**; it may move 0.565 only a little. That is a *success*, not a disappointment — it means the contamination was masking, and the substrate is now the constraint.
- B is what moves the number, and it's gated by whether biology re-canonicalization recovers reuse without over-collapsing context.
- If, after A+B, efficacy holdout is still weak but the synthetic harness shows EM recovers planted reliabilities, the conclusion is clean and defensible: **the method is sound; n≈500 efficacy-dominated data is the limit**, and the roadmap is C (safety enrichment + decomposition product + targeted scale).
