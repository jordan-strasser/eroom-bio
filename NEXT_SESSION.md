# NEXT_SESSION.md

Handoff for the next debugging pass on Eroom Bio's predictive accuracy.
Last session ended 2026-05-17 with a damning out-of-sample result and a working architectural hypothesis for why.

---

## TL;DR

- **In-sample AUROC** (post-round-14 fixes, n=86): **0.683**
- **Holdout AUROC** (n=19 OOS trials): **0.243** — worse than random, predictions inverted relative to outcomes
- **Working diagnosis**: the graph predicts `P(compound × indication efficacy)`, but our labels measure `P(this specific trial succeeds)`. Pembrolizumab in MSS prostate fails for trial/population reasons, but the graph aggregates ALL pembro evidence (mostly from melanoma successes) and predicts high regardless.
- **The Beta-Binomial mechanics and LLM signal quality are probably fine**. The architecture is predicting a different question than the labels measure.

---

## What we did this session (chronological)

### Rounds 6–13 (architecture + corpus hygiene)
| Round | Status | What |
|---|---|---|
| 6 | Deferred | Vaccine biology (DC-receptor heuristic) — not yet started |
| 8/9 | Shipped | Arm-id alignment via classifier prompt menu (CT.gov `group_id` verbatim) — replaced fragile LLM slugs |
| 10 | Shipped | CMP-001 codename + compound canonicalization (split combos, strip parentheticals) |
| 11 | Shipped | TargetType expansion via ChEMBL REST (DNA, RNA, microtubule via CHEBI ids) |
| 12 | Shipped | Compound-id iteration fix + cell-therapy heuristic + separator-agnostic matcher |
| 13 | Shipped | ChEMBL-driven diagnostic filter (conservative: `indication_class` contains "diagnostic" OR (`therapeutic_flag=False` AND `max_phase<1`)). Plus T-VEC/ADI-PEG-20/GSK2132231a entries in `_CELL_THERAPY_COMPONENT_TARGETS` |

User pushback that shaped these:
- "non_therapeutic_intervention_patterns does not seem generalizable" → reverted aggressive string-match filter to ChEMBL-attribute rule
- "are you misdiagnosing intentional compound nodes?" → check if combo components exist singly in other arms before dropping
- "we can just use whatever names come from CT.gov, right?" → killed LLM arm slugs entirely

### The pivot to evaluation
Scaled corpus to n=100 (melanoma_145). Built `scripts/eval_predictions.py` (in-sample), then `eval_predictions_loo.py` (leave-one-out via `_strip_trial_evidence`), then `eval_holdout.py` (true OOS on melanoma_145 NCTs not in trained slice).

Found: `primary_endpoint_met` alone labels only 60/100; combining with `trial_outcome` fallback yields 89/100. Built `_resolve_label` helper.

### Root-cause diagnosis (3-agent parallel)
Spawned 3 agents to diagnose why AUROC was hovering ~0.5. Key findings:
1. **LINCS over-weighting** — 89.8% of evidence records were LINCS, drowning out clinical signal
2. **Drop-zero-trust edges rule** — `path_query` was dropping `Beta(1,1)` edges entirely, so predictions depended on whichever surviving (often LINCS-loaded) edge held the chain
3. **AMBIGUOUS-at-high-n_eff** — failed trials sharpen variance without shifting mean, so failure evidence didn't move predictions
4. **Affects prior too strong** — `Beta(4.0, 1.0)` OT-direct priors made fresh edges look like strong successes

### Round 14: 5 prediction fixes (all shipped)
| # | File | Change |
|---|---|---|
| 1 | `src/ingestion/lincs.py` | LINCS `STRONG_SUPPORT` → `MODERATE_SUPPORT` |
| 2 | `src/prediction/path_query.py` | Added `_TRUST_FLOOR = 0.10`; `_trust_weight` returns `max(_TRUST_FLOOR, raw)` — no more dropping unknown edges |
| 3 | `src/annotation/prompts/classification_system.txt` | Failed trials WITH target engagement emit `weak_contradict` on `mechanism_affects` (not `ambiguous`) |
| 4 | `src/graph/populate.py` | OT-direct affects prior `Beta(4.0, 1.0)` → `Beta(2.0, 1.0)` |
| 5 | `src/prediction/path_query.py` | Weakest-link picker: composite score `(1−E[p])·trust + 0.20·(1−trust)` |

Tests updated to match new semantics:
- `test_uniform_prior_gets_zero_trust` → `test_uniform_prior_gets_floor_trust`
- `test_uniform_priors_have_zero_bottleneck_score` → `test_data_gap_bottleneck_score`
- `test_one_strong_edge_dominates` range adjusted
- `test_adds_evidence_when_consistent` now expects `moderate_support`

### The four eval runs
| Run | n | AUROC | Notes |
|---|---|---|---|
| In-sample (pre-round-14) | 86 | 0.596 | Baseline |
| LOO (pre-round-14) | 86 | 0.481 | First sign of trouble — near-chance |
| In-sample (post-round-14) | 86 | **0.683** | Fixes work for in-sample |
| Holdout (post-round-14) | 19 | **0.243** | OOS collapse |

---

## The OOS finding in detail

`scripts/eval_holdout.py --max-fresh 0` ran on 50 melanoma_145 NCTs not in the trained slice. Output:

```
Predicted: 19  |  Skipped: 31
AUROC            │ 0.243 │ 1.0=perfect, 0.5=random
Brier            │ 0.465 │ base-rate baseline (0.26): 0.194
Mean P | label=1 │ 0.724
Mean P | label=0 │ 0.775   ← HIGHER for failures
```

### Worst misses (illustrative)
- **NCT04032704**: pembrolizumab → prostate_cancer, P=0.831, **failed** (MSS prostate is known pembro flop)
- 6× nivolumab → melanoma trials all predicted 0.76–0.78, **all failed** (likely later-line / post-checkpoint)
- **NCT02967692**: dabrafenib → melanoma, P=0.776, **failed**

### Why 31 skipped
Compounds not in trained graph (novel TCRs, new small molecules). The 19 that DID predict are all well-known drugs (pembro/nivo/dabra) with extensive melanoma success evidence — so all predict high regardless of specific trial context.

### Caveats on the number
- n=19 is small; 95% CI on AUROC is roughly 0.05–0.50
- "Worse than random" is observed but noisy
- Sample is biased toward high-evidence compounds — the trials where the graph is MOST confident

---

## Working architectural diagnosis

The graph encodes `compound → target → mechanism → biology → indication` chains and updates `(compound, indication)` beliefs. But trial outcomes depend on **trial-specific covariates** the chain doesn't model:

- Line of therapy (1L treatment-naive vs post-checkpoint refractory)
- Population enrichment (PD-L1-high vs MSS prostate)
- Dose / schedule
- Comparator strength (placebo vs SoC)
- Trial design (randomized vs single-arm)

Pembro works in melanoma 1L; pembro fails in MSS prostate. Both feed the same `pembrolizumab → indication` belief, pulled toward "pembro works." The chain can't distinguish them.

**The architecture predicts drug-class efficacy. The labels measure trial-specific success.** That's a unit mismatch, not a math bug.

---

## Recommendations / decision points for next session

Three live options. User explicitly hasn't picked yet.

### Option A: Ship round-14 as-is and move on
Round-14 fixes are real improvements (in-sample 0.596 → 0.683, real bugs in LINCS weighting + drop-zero-trust logic). They don't fix OOS but they don't make it worse either — the OOS failure is architectural, not algorithmic.

**Action**: commit + merge round-14 branch. Document holdout = 0.243 as a known limitation. Move to corpus expansion.

### Option B: Re-frame the prediction unit (the cleanest pivot)
Stop predicting per-trial outcomes. Predict `(compound, indication)`-level success rate, evaluate against indication-aggregated outcomes (e.g., "fraction of pembro-prostate trials that succeed in primary endpoint").

**Pros**: matches what the architecture actually computes. Honest about scope.
**Cons**: lose granular per-trial evaluation. Aggregation is itself non-trivial (how to weight different trial sizes).

### Option C: Enrich the chain with covariates (the big architectural lift)
Add line-of-therapy, population, and dose as edge attributes or chain branches. `responds_differently` partially supports this but isn't pervasive.

**Pros**: matches the per-trial prediction target.
**Cons**: large architectural surface. Would need to break the v0.1.0 lock. Adds prompt + extraction complexity. Probably 2–3 rounds of work.

### My read
Option B is the highest-information next move. It tests the diagnosis cheaply — if indication-aggregated AUROC is decent, the architecture is fine and we just had the wrong evaluation target. If it's still bad at the aggregate level, then Option C is needed. Don't do C until B is run.

---

## Files touched (round-14, not yet committed)

- `src/ingestion/lincs.py` — support bucket demotion
- `src/prediction/path_query.py` — trust floor + composite picker
- `src/annotation/prompts/classification_system.txt` — weak_contradict on engaged-but-failed
- `src/graph/populate.py` — affects prior rebalance
- `scripts/eval_predictions.py` — `_resolve_label` helper
- `scripts/eval_predictions_loo.py` — NEW, LOO evaluator
- `scripts/eval_holdout.py` — NEW, true OOS evaluator
- `tests/test_prediction.py` — 4 tests updated for new semantics

Branch state: round-14 fixes live on a branch (not main). Tests pass. Not merged pending user decision on Options A/B/C.

---

## Pipeline state at session end

- 217 nodes, 533 edges, 666 tests passing
- main at `bc79466` (classifier per-arm Phase B merge)
- n=100 melanoma snapshot built and cached
- Holdout fetches cached for 19/50 NCTs in `data/annotations/` (the rest were skipped, not fetched)

---

## Round-17 candidate: node enrichment for cross-trial learning

Surfaced during round-16 audit (2026-05-19): nodes are id+name+1–3 categorical fields. The rich data we extract per trial (effect sizes, p-values, dose info, biomarker observations, subgroup findings, eligibility criteria, classifier reasoning, evidence quotes) gets used once for edge updates then discarded. Each node ends up as a thin shell that says "this thing exists" with almost nothing about how it has *behaved* across the corpus.

User's framing: `ORR_melanoma → melanoma` doesn't tell us much. The endpoint node should accumulate a cross-trial profile (threshold distribution, observed-value distribution, met-rate, concordance with OS, breakdown by compound class / line of therapy). Same idea applies to every node type.

**Why this matters**: future cross-trial ML wants each node as a feature vector. Today the graph forces an embedding-of-name approach because there's nothing else on the node. Properly enriched nodes become structured features the model can learn from.

**Round-17 scope** (rough):

1. Add a `profile: dict[str, Any]` (or typed `NodeProfile` model) on each node type.
2. End-of-populate aggregation pass: walk every trial extraction + trial_subgraph, accumulate per-node statistics into the profile field.
3. Per-node-type profile schemas:
   - **EndpointNode** — thresholds_set distribution, observed values distribution, met_rate, concordance with adjacent endpoints, breakdown by compound class / line
   - **PopulationNode** — response rate distribution, biomarker prevalence, n trials sampled, mean enrollment, by-compound success rate
   - **InterventionNode** — per-indication outcome distribution, common combo partners, dose ranges, observed AE incidence aggregated, line-of-therapy distribution
   - **TargetNode** — list of compound classes engaging it, mechanism downstream, n_trials touching, druggability proxy from observed binding diversity
   - **BiologyNode** — upstream mechanism list, downstream indication list, observed evidence_strength distribution
   - **IndicationNode** — already has observed_variants + qualifier_axes; extend with base success rate, n_trials, line distribution
   - **TrialNode** — already has phase/status/sponsor/enrollment; extend with outcome, primary_endpoint_met, study design fields from CT.gov we currently drop
   - **AdverseEventNode** — incidence distribution across trials, severity distribution, common compound co-occurrences
4. Persist on snapshot export; round-15 prune still removes zero-coverage ghosts but enriched survivors carry their profile.

**Caveat**: profile aggregation runs on the FULL trial set every build. If corpus scales to 10k trials, the aggregation cost matters; for melanoma_145 it's instant.

Related to [[project-biolord-embeddings]] (long-term semantic similarity layer) — that layer would replace heuristic alias-pulling, but enriched node profiles are needed regardless to give the embeddings something structured to attend over.

---

## Round-15 candidate: graph hygiene (post-eval-refactor)

Surfaced by the round-14 holdout audit (2026-05-19):

**Ghost indication nodes**: 29 of 41 IndicationNodes in the trained graph are unused by any chain. `prostate_cancer`, `prostate_cancer__unselected`, `other_prostate_cancer` are the canonical examples — they survived from earlier builds (basket trials like NCT04032704 with 9 conditions canonicalized into separate IndicationNodes) and were never pruned when the corpus narrowed to melanoma_145.

The new holdout-v2 eval filters them out at scoring time via a `_TRAINING_USED_NODES` set, but the underlying populator/snapshot still accumulates them.

Two follow-ups:

1. **Rebuild-from-scratch mode for `scripts/build_graph.py`**: today the populator typically bootstraps from an existing snapshot via `import_snapshot`. Add a `--fresh` flag (or make it the default) that starts from an empty `GraphStore` so the graph only contains what the current corpus produces.

2. **Prune unused nodes at end of populate**: drop any node not referenced by ≥1 chain AND with no edge of `evidence_strength > 0`. Keep companion structural nodes (subtype hierarchy parents, e.g. `melanoma` as parent of `cutaneous_melanoma`) via an explicit allowlist.

**Caveat that shaped the design**: legitimate basket trials (one trial, multiple indications it's actually testing — breast + melanoma + lung) should still produce IndicationNodes and chains for each indication. The bug isn't basket trials creating cross-indication nodes; it's that those nodes survive after the source trial gets dropped from a later corpus.

---

## Open questions to resolve early next session

1. **Pick Option A, B, or C** (or some combination)
2. If B: define the aggregation function for indication-level outcome (count-based? weighted by sample size? success threshold?)
3. If C: which covariate first? Line-of-therapy is highest-signal but population enrichment is closer to existing `responds_differently` machinery
4. **Re-verify holdout with fresh extractions** for the 31 skipped trials before treating 0.243 as load-bearing. Could discover the skipped trials are mostly the "interesting" ones the graph has no opinion on, which would make the 19-trial result even less representative.
5. **Investigate the 35 trials with `primary_endpoint_met=None`** — labeling completeness is still a known gap (#35).

---

## What I'd do first (if I'm picking this up alone)

1. Re-run `eval_holdout.py` with `--max-fresh 50` to get fresh extractions on the 31 skipped trials. Get a real n.
2. Build the indication-aggregate evaluator (Option B preview). Cheap, ~50 lines.
3. Bring those two numbers to the user before doing anything destructive.

Don't merge round-14 yet — user might want to bisect later if Option B reveals the fixes themselves are part of the issue.
