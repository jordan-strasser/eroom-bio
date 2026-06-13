# Eroom Bio — why is binary trial-success prediction at chance?

**Investigation, not fix.** All measurements on `data/exports/multi_500_annotated.json`
(n=472 in-graph trials, 3851 nodes, 12734 edges, 25561 evidence records) unless noted.
Scripts under `scratch/diagnostics/` (`_common.py`, `probe_*.py`). Honest out-of-sample
number this session: **k-fold re-attribution holdout AUROC = 0.565** (in-sample 0.795,
gap +0.231), n=221 scorable — consistent with the repo's documented flat ~0.53 band.

---

## Phase 0 — Pipeline map

| stage | where | note |
|---|---|---|
| trial → chain decomposition | `src/annotation/extractor.py` (LLM extract) → `src/graph/populate.py` (`_populate_trial_*`, fan-out) → read-only resolver `scripts/eval_holdout_compose.py:402 resolve_chain`. Backbone edge set: `src/prediction/path_query.py:30-42` (`_CAUSAL_CHAIN`+`_AUXILIARY_EDGES`); `src/annotation/attributor.py:228 _chain_backbone_edges` | a trial = arm×subgroup×endpoint×pathway fan-out of chains |
| node canonicalize / merge | `src/graph/node_merge.py`; biology = `bio:<content-hash>` ids; mechanism = Reactome `R-HSA*` id-merge | mechanism merges well (1% trial-scoped); **biology 71% singletons** |
| LLM "votes" | `src/annotation/classifier.py` emits `trial_outcome` + `failure_modes` + `operational_failure`. **Per-edge up/down votes were REMOVED** — `attributor.py:716-724, 736-740`: "a single trial cannot pinpoint WHICH edge failed" | the only signal that conditions the backbone is the coarse arm/trial OUTCOME |
| Beta-Binomial update | `src/inference/beliefs.py:541 apply_virtual_evidence` → `α += n_eff·p_obs`, `β += n_eff·(1−p_obs)`. `p_obs` from bucket (`beliefs.py:65`), `n_eff` from evidence type (`beliefs.py:100,245`). Outcome p_obs: success **0.80** (`attributor.py:99`), failure **0.20** (`attributor.py:101`), explaining-away split `attributor.py:942-958` | NOT "β+=1"; a conjugate update whose direction is set by the outcome |
| aggregation (edge→score) | `src/prediction/path_query.py:293 _aggregate_samples` = **softmin** (weakest-link, default `EROOM_AGG`, line 318/331). Edges from `path_query.py:1064 _collect_edges` (`_CAUSAL_CHAIN`+`_AUXILIARY_EDGES`; empty Beta(1,1) dropped at line 1134). Safety folded SEPARATELY: `overall = efficacy × (1 − safety_penalty)` (`path_query.py:655-656`), `_compute_safety_penalty` cap 0.60 (`path_query.py:774-889`) | one undifferentiated softmin over efficacy+measurement; safety is a separate capped drag |
| eval harness | `scripts/eval_holdout_kfold.py`: fold = `md5(nct)%k` (`:59`); `graph_holdout_predictions` re-attributes `initial.json` per fold with the fold **excluded from attribution** (`:105`) then predicts. Label: `eval_holdout_compose.py:463 _resolve_label` | leakage-free; honest |

### Single most important structural fact
**The code distinguishes SAFETY from the rest, but does NOT distinguish efficacy-spine
from measurement/validity edges.** Proof:

- *Safety is separated* — own update path (`attributor.py:1416 attribute_adverse_events`,
  incidence deltas) and own aggregation factor (`path_query.py:655 overall = efficacy ×
  (1 − safety_penalty)`, cap 0.60).
- *Efficacy and measurement are identical* — `_chain_backbone_edges` (`attributor.py:257-263`)
  conditions `affects, modulates_via, mechanism_affects, biology_drives` (efficacy) **and**
  `reflects_biology, endpoint_captures` (measurement) with the **same** conjugate update;
  `_collect_edges` (`path_query.py:1091`) feeds **both** into the **same** softmin
  (`path_query.py:606`). There is no `edge_class` field; `EdgeBeliefState` (`models.py:824`)
  is one Beta for every type. So the central hypothesis is **half right**: safety is routed
  apart, efficacy↔measurement are not.

---

### P1 — Update-rule routing across edge classes
Verdict: **CONFIRMED (substance); literal form REFUTED**
Evidence: The update is not "β+=1 on all" — on FAILURE every backbone edge gets a *modest*
contradict (p_obs=0.20) with the trial mass `w_base` SPLIT by explaining-away
`u_i = 1 − E[p_i]` (`attributor.py:942-958`), so already-weak edges absorb the blame and
curated high-belief edges self-protect. On SUCCESS every backbone edge gets the FULL upvote
(p_obs=0.80, `_per_edge_fracs` returns `[1.0]*n` — `attributor.py:163-166`). BUT the smear
is real and class-blind: the backbone is efficacy+measurement, the failure REASON never
routes (P4), and AE edges are absent from this path — so a toxicity death downvotes the
(possibly-correct) mechanism/biology spine, never the AE edge. P1.3 (`probe_labels.py`):
Pearson(E[p], host-trial base success rate) = **+0.44 efficacy, +0.67 measurement**, −0.15
safety; **70% of efficacy edges have exactly one host trial** (E[p] is then a pure echo of
that one outcome: success→0.676, failure→0.514). Mismatch (P1.2, `probe_p12_examples.py`):
e.g. NCT00282347 (high-placebo-response, a *measurement* failure) downvoted **58** efficacy-
spine edges at full gate weight.
Code: `src/annotation/attributor.py:818-1011 _condition_chain_on_outcomes`; `:154 _per_edge_fracs`
Notes: explaining-away is a genuine partial mitigation the hypothesis didn't credit — but it
redistributes blame *within* the efficacy+measurement backbone, it does not route *across*
classes. AE edges have their own incidence path, so the one class that "shouldn't be smeared"
isn't — the efficacy spine is.

### P2 — Aggregation function
Verdict: **PARTIALLY REFUTED** (it is NOT a flat product; it is structured-but-wrong)
Evidence: aggregation = **softmin / weakest-link** over efficacy+measurement (`path_query.py:318,331`),
times a SEPARATE `(1 − safety_penalty)` with safety aggregated by **max** over AEs, capped 0.60
(`path_query.py:888-889`). So safety already resembles the hypothesis's `∏(1−q)` term (as a
capped max), but there is **no distinct detection term** — measurement edges are pooled into
the same weakest-link as the efficacy AND. The triangle is **3 independent Betas**, not a joint
factor: 1566 biology→endpoint→indication + biology→indication "triangles" exist
(`probe_struct.py`), each contributing 3 independent inputs (and double-counting biology +
indication under the legacy `EROOM_AGG=geomean`/product branch). Score distribution
(`probe_pred.py`): compressed to **[0.31, 0.80]**, mean 0.625 sd 0.123. corr(score, chain-length)
= **+0.011** (so it does NOT track length — that sub-hypothesis is refuted; length is ~constant
at 5.6 because the fan-out collapses to a fixed backbone). Calibration is monotone but
mis-sloped (pred 0.45→obs 0.19; pred 0.75→obs 0.86).
Code: `src/prediction/path_query.py:293-333, 655-656, 1064-1144`
Notes: weakest-link is dominated by `mechanism_affects` (132/221) and `modulates_via` (43) —
the efficacy spine sets the bottleneck; measurement bottlenecks 39/221.

### P3 — Node merging / context collapse / loops
Verdict: **CONFIRMED (context collapse); cycles REFUTED**
Evidence (`probe_struct.py`, `probe_p3_context.py`): the belief-edge graph is a **DAG — 0
directed cycles**. But context collapse is real: of edges with ≥3 host trials, **86%
(487/565) span ≥2 distinct indications** and **40% (226/565) pool genuinely mixed outcomes**
(host success rate in (0.2,0.8)) under one context-free Beta. The single most-reused edge,
`DNA→dna_damage` (modulates_via), pools **35 trials across 27 indications**. Plus 1566
non-independent triangles (above). Mechanism-merge itself is healthy (only 7/507 trial-scoped),
so the collapse is *intended* pooling without context conditioning, not a merge bug — except
`mechanism_affects` is tissue-conditioned at query time (`path_query.py:1125`), nothing else is.
Code: `src/graph/node_merge.py`; `path_query.py:1124-1130`
Notes: the diabetes/metabolic mechanisms dominate the high-degree edges — broad shared
pathways sitting at base rate, exactly the "shared hub at base rate" the prior arc found.

### P4 — Failure-reason routing
Verdict: **CONFIRMED — the reason exists and is thrown away**
Evidence: a mechanistic 13-category failure taxonomy IS extracted (`taxonomy.py:16-29
FailureMode`) and maps almost 1:1 to edge classes, but the attributor reads it **only for a
counter** (`attributor.py:736-740`) — "the backbone is no longer name-matched from it." The
ONLY routing is a coarse binary `operational_failure` gate → weight 0.2 vs 1.0
(`taxonomy.py:475 gate_weight_for`; explicitly "NOT the 13-category failure mode",
`taxonomy.py:505`). Coverage (`probe_labels.py`, 233 trial_outcome=failure): only **10/233
(4%) implicate the efficacy spine**, yet all 233 downvote it; **68% (159/233) failed for a
NON-efficacy/measurement reason** (insufficient_information 93, underpowered 38, commercial 19,
manufacturing 3, DLT 3, …) and still smear the spine. The gate is leaky: of the 60
operational/business failures only **35 (58%) actually had the gate fire**; the 3 DLT (safety)
failures stayed at full weight (gate 1.0). The label resolver already routes DLT/commercial/
underpowered/insufficient-info to "ambiguous" and drops them from SCORING
(`eval_holdout_compose.py:482-487`) — so the honest 0.565 is computed on the clean efficacy
subset **while the beliefs that score it were trained on the 139 dropped failures**. That
asymmetry (contaminate in training, exclude in scoring) is the heart of P4.
Code: `src/annotation/taxonomy.py:471-515`; `attributor.py:894,901`; `eval_holdout_compose.py:463`
Notes: P4.3 "AUC on efficacy-only failures" is already the deployed metric (0.565) — routing
the *training* updates by reason is the untested lever, not re-scoring.

### P5 — LLM attribution reliability
Verdict: **REFUTED for the mechanism in use; CAN'T-TELL on cross-run stability**
Evidence (`probe_p5.py`): per-edge LLM votes were abandoned (P0), so "confabulated edge votes"
doesn't apply. The signal that IS used — the coarse outcome — is internally consistent: the
extractor's `primary_endpoint_met` and the classifier's `trial_outcome` **agree on 289/290
(100%)** comparable trials. BUT classifier confidence is low (mean 0.541, **41% < 0.5**),
**66% are flagged needs_expert_review**, and **30% of all trials are `insufficient_information`**
(the LLM declaring it couldn't extract clean efficacy) yet still condition the chain. Cross-run
vote/decomposition stability is not measurable — one cached run per trial; re-running ~15
trials needs paid API spend (declined for a diagnosis).
Code: `src/annotation/classifier.py:340-380`
Notes: the reliability problem isn't a noisy outcome bit — it's that a reliable *single bit* is
spread across a whole chain (P1) and that 30% of trials carry no usable efficacy signal.

### P6 — Sparsity of shared structure
Verdict: **CONFIRMED — transfer is structurally weak at this scale**
Evidence (`probe_struct.py`, edge reuse = #distinct host NCTs whose outcome touched the edge):
mean trials/edge = **efficacy 1.24, measurement 0.36, safety 0.90**. Efficacy: 43% touched by
0 host trials (curated-only), 37% by exactly 1, only 11% by ≥2. Measurement: 68% by 0 hosts,
29% by 1. Node reuse: BiologyNode **71% singletons**, EndpointNode 49% used by 0 chains,
IndicationNode 49% by 0. So the cross-trial substrate the thesis needs barely exists — most
edges that carry trial evidence are single-trial echoes, which is why held-out trials revert
to the cohort average.
Code: graph evidence provenance via `EvidenceRecord.source_id` (`models.py:765`)
Notes: this is the structural twin of P7's memorize-no-transfer.

### P7 — Eval hygiene / leakage
Verdict: **No leakage in the holdout path; in-sample number IS leaky (and known)**
Evidence: the split is entity-disjoint by `md5(nct)%k` (`eval_holdout_kfold.py:59`) and the
held-out fold is **excluded from attribution** before its trials are scored (`:105`), so a
test trial's own outcome never touches the weights that score it — verified the `initial.json`
base carries **0 trial-outcome records** on the backbone (curated facts only; `annotated`
adds 9505). Train vs test: **in-sample 0.795 vs honest holdout 0.565, gap +0.231**
(`honest_holdout_n472.txt`). Holdout binary accuracy 0.665 = the base rate (TP=129, TN=18,
FP=56, FN=18 → it predicts "success" for ~84% of trials).
Code: `scripts/eval_holdout_kfold.py:59-112`
Notes: the in-sample 0.80 quoted elsewhere is the leaky upper bound; the deployed reality is
0.565. The gap is memorization, not a split bug.

### P8 — Calibration vs discrimination
Verdict: **PARTIALLY CONFIRMED — measurement/safety collapsed; efficacy compressed**
Evidence (`probe_struct.py`, evidenced edges only): posterior E[p] by class — efficacy
mean 0.620 sd 0.150 (some spread), **measurement mean 0.533 sd 0.113**, **safety mean 0.565
sd 0.111 with MEDIAN 0.500** (half the AE edges sit exactly at the prior). Global base rate
0.658. So efficacy carries discriminable spread but is centered *below* base rate; measurement
and safety are clustered at 0.5 = near-zero discrimination. Predictions inherit this: compressed
to [0.31, 0.80] (P2).
Code: `EdgeBeliefState.expected_probability` (`models.py:838`)
Notes: not a total collapse — the efficacy layer is where what little discrimination exists
lives, consistent with the prior arc's "biology>mechanism, coarse>granular."

### P9 — Transfer asymmetry by edge class (where real signal may be)
Verdict: **9.1/9.2 CONFIRMED; 9.3 CAN'T-TELL**
Evidence (`probe_struct.py`): on-target safety transfers structurally — within-target posterior
spread (targets with ≥3 edges): `target_associated_ae` **SD 0.048** (n=45 targets) vs `affects`
SD 0.134 — a target's AE liability is far more consistent than its binding/efficacy. AE is also
the most-instantiated class (3286 causes_ae + 700 target_associated_ae). 9.3 (AUROC for
predicting *safety-driven* failure from AE edges) is **not computable**: the corpus contains only
**3 DLT failures**, and `_resolve_label` routes safety-driven failures to "ambiguous" (excluded
from scoring); `safety_penalty` alone over the scored (efficacy) set gives AUROC 0.484 with only
19/221 trials carrying any penalty (`probe_pred.py`). So "the genuine predictive value is on
safety via shared targets" remains plausible and structurally supported, but **unmeasured** —
this corpus barely contains the failures it would predict.
Code: `path_query.py:776-889 _compute_safety_penalty/_penalty_from_risks`; `attributor.py:1416 attribute_adverse_events`
Notes: to test P9 you need a corpus enriched for safety-stopped trials; the current oncology/
metabolic mix is efficacy-failure dominated.

---

## Top 3 most likely root causes, ranked by evidence strength

1. **One outcome bit is smeared across an undifferentiated, context-free, base-rate-echoing
   efficacy+measurement weakest-link (causes #6 predictor + #3 belief-formation).** [HIGH]
   The failure *reason* is discarded (P4: only 4% of failures implicate the spine, 68% mis-route
   onto it; gate fires 58%); efficacy and measurement share one update and one softmin (P0/P2);
   the resulting marginals echo their host-trial base rate (P1.3: r=0.44 efficacy, 0.67
   measurement) and 70% are single-trial echoes (P1). This is the binding constraint: it both
   contaminates the beliefs (training on failures that aren't efficacy failures) and discards the
   one cross-class signal — safety — into a separate path that the efficacy-only metric can't see.

2. **The cross-trial substrate barely exists at n≈500, so the system memorizes instead of
   transferring (causes #6 + #5).** [HIGH] Edge reuse is 1.24 (efficacy) / 0.36 (measurement)
   trials/edge; 71% of biology nodes are singletons (P6). In-sample 0.795 collapses to honest
   0.565 (P7). A held-out trial lands on edges touched by ≤1 other trial and reverts to the
   cohort average.

3. **Context collapse + a near-empty measurement/validity layer (causes #4 pooling + #5 empty
   edges).** [MED-HIGH] High-degree edges pool 86% across multiple indications / 40% across mixed
   outcomes under one context-free Beta (P3); the entire population dimension is inert
   (`responds_differently` 100% empty), `reflects_biology` 46% empty, and the measurement class
   clusters at 0.5 (P8). The "detection/validity" half of the topology contributes almost nothing,
   so the prediction is effectively a context-free efficacy-spine marginal.

## Measurements I could not make (and why)
- **P9.3 — safety-driven-failure AUROC from AE edges.** Only 3 DLT failures in the corpus, and
  `_resolve_label` excludes safety-driven failures from scoring. Needs a safety-enriched corpus.
- **P5 — cross-run LLM vote / node-decomposition stability.** One cached run per trial; measuring
  requires paid re-extraction of ~15 trials (not justified for a diagnosis). Used internal-
  consistency (extractor vs classifier, 100% agree) as a proxy instead.
- **Honest holdout of the (s,t) belief-field representation at scale** — orthogonal to this
  investigation; the field LOO needs per-fold re-materialization and was not run.
