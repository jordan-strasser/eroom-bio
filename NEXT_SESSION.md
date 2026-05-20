# NEXT_SESSION.md

Handoff for the next session on Eroom Bio. Last session (round 18, 2026-05-20) ended with cross-trial learning working across 5 indications on a n=50 smoke corpus + canonicalization fix shipped + two big architectural items queued for next pickup.

---

## TL;DR — where we are at end of round 18

**Shipped this session** (commits 9534dec → cece593 on main):
- Round 14/15/16: prediction refactor (B(1,1) edges dropped, no trust floor, compound optional), classifier prompt always-emits `responds_differently` + `endpoint_captures`, populator structural fix for unselected populations, canonicalization persisting chembl_id + OT aliases on compound nodes, build orchestrator safety nets, eval refactor with three-bucket labels, 13 regression tests.
- Round 17: LLM extraction of population features (line of therapy, prior tx, biomarker, mutation) from eligibility criteria; populations went 114 → 283 (2.5x); 92% of trials moved off `__unselected`.
- Round 18 scaffolding: multi-indication corpus + case-study audit infrastructure across 5 indications (melanoma + alzheimers + colorectal + atherosclerosis/hypercholesterolemia + NSCLC + thyroid). CT.gov 302→429 rate-limit bug found and fixed (`follow_redirects=True` + per-attempt exponential backoff in `get_study`, lowered corpus_concurrency 8 → 4).
- Round 18 codename-canonicalization followup: curated CODENAME_TO_INN dict + one-off snapshot migration script. azd6244 → selumetinib applied; selumetinib node now carries AZD6244 / ARRY-142886 / ARRY-886 / AZD-6244 as aliases.

**Eval headline numbers** (n=50 smoke graph, end of round 18):
- In-sample AUROC on clean-mechanistic-failure subset (round-17): **0.873**
- 5-case-study audit verdict, direction-only: 3/5 match (nivolumab success ✓ at 0.84, solanezumab failure ✓ at 0.48, bevacizumab AVANT failure ✓ at 0.48, torcetrapib miss — safety-not-chain, selumetinib weak — sparse evidence).

**Two architectural fixes blocked before the full n=281 build** (see round 19 + 20 below):
1. **Round 19 — incremental build mode**: avoid wiping the snapshot on every rebuild. Sequential trial addition without double-counting attribution. Important infrastructure before the corpus expands further.
2. **Round 20 — safety in P(success)**: integrate `causes_ae` + `target_associated_ae` into `overall_probability`. The torcetrapib audit exposed the v0.1.0 architectural decoupling as wrong; trials fail for safety reasons that the chain can't currently see.

**User said explicitly**: do round 19 + round 20 before the next full-scale (n=281) build.

---

## How to pick this up at the next session

**Read these files in this order to get oriented**:
1. This file (sections below — round 19 plan, then round 20 plan)
2. `audit/cross-trial-learning/key_trials.md` — the 5 case-study failure modes from literature
3. `audit/cross-trial-learning/results.md` — the round-18 audit output
4. Memory at `~/.claude/projects/.../memory/MEMORY.md` for project history

**The exact place to start**: round 19. See its detailed plan below — it's "add `--base-snapshot` + `--add-trials` flags + attribution idempotency + tests" and lands cleanly in a few files (`scripts/build_graph.py`, `src/annotation/attributor.py`, `src/graph/store.py`, `tests/test_build_graph.py` + `tests/test_attributor.py`). Estimated half-day to full day.

After round 19 ships: round 20 (safety in P(success)). Detailed plan below; ~4–6 hours including test surface updates.

After both: run the full n=281 multi_indication build using the new incremental mode (don't re-do the n=50; add the extra ~230 trials to the existing n=50 snapshot).

**Useful commands**:
```bash
# Existing n=50 snapshot lives at:
data/exports/multi_indication_50_annotated.json

# Existing n=50 corpus list:
data/corpora/multi_indication_50.txt    # 50 NCTs + 5 case studies
data/corpora/multi_indication.txt       # the bigger 281-NCT list

# Run the case-study audit on the n=50 graph at any point:
.venv/bin/python -m scripts.case_study_audit \
  --graph data/exports/multi_indication_50_annotated.json

# Sanity-check a graph:
.venv/bin/python -m scripts.multi_indication_sanity \
  --graph data/exports/multi_indication_50_annotated.json

# Apply codename → INN canonicalization to a snapshot:
.venv/bin/python -m scripts.canonicalize_codenames \
  --in PATH --out PATH
```

**Open known issues** (none blocking, all queued):
- Case-study audit's `verdict` logic is too strict — calls "PARTIAL" any time `weakest_link` doesn't exactly match the literature's expected edge, even when prediction direction is right. After round 20 ships, refine verdict to also accept MATCH-VIA-SAFETY for safety-driven failures and direction-only matches when bottleneck is close.
- `src/graph/compound_codenames.py` has a curated dict but isn't wired into the populator's compound-creation step yet. Future builds will still produce code-form ids until that wiring lands. Tracked for later — the migration script (`scripts/canonicalize_codenames.py`) handles it retroactively for existing snapshots.
- The melanoma-only OOS holdout AUROC story (round 14's 0.243) was eclipsed by the round-15 / 16 / 17 architectural work. The current "in-sample on clean-mechanistic-failure subset" AUROC 0.873 is the headline. A clean OOS measurement requires either (a) more trials per indication so we have a real held-out set per indication, or (b) the incremental-build mode (round 19) so we can add trials to test against without rebuilding.

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

## Round-20 — integrate safety into P(success) (DO NEXT SESSION, after round 19)

Surfaced by the round-18 torcetrapib case study (2026-05-20).
`predict_clinical_hypothesis` currently treats AE/safety as a separate
output (`SafetyRisk` list) — it does NOT factor into
`overall_probability`. Comment on `PredictionResult` (line 204 in
`src/prediction/path_query.py`): *"efficacy and safety are scored
independently."* This was a v0.1.0 architectural choice that the
torcetrapib audit just exposed as wrong.

ILLUMINATE (NCT00134264) failed for off-target hypertension causing
excess deaths. The mechanism CHAIN worked (CETP inhibition raised HDL
dramatically). Without safety in the prediction, the system rates
torcetrapib at P=0.65 modest-success, missing the actual failure mode.

**Concrete plan when picking this up**:

1. **New helper `_compute_safety_penalty` in `src/prediction/path_query.py`**:

   ```python
   def _compute_safety_penalty(
       self, chain: CausalChain,
       *,
       min_belief: float = 0.5,
       min_evidence: float = 1.0,
   ) -> float:
       """Combine compound-specific + on-mechanism AE evidence into a
       [0, ~0.4] penalty applied to the mechanism-only P(success)."""
       # Pull both edge types via the existing _collect_safety_risks
       # logic — already does the union of causes_ae + target_associated_ae
       # with min_belief / min_evidence filters.
       risks = self._collect_safety_risks(
           chain, min_belief=min_belief, min_evidence=min_evidence,
       )
       if not risks:
           return 0.0
       # Per-AE contribution: belief * log(1 + n_eff) / log(50)
       # (mirrors the round-15 _trust_weight log-saturation curve so
       # the AE penalty caps gracefully under heavy evidence accumulation).
       contributions = []
       for r in risks:
           t = min(1.0, math.log(r.evidence_strength + 1) / math.log(50))
           contributions.append(r.belief_probability * t)
       # Aggregate via soft-or so multiple moderate AEs accumulate but
       # we don't run away on long AE tails.
       penalty = 1.0
       for c in contributions:
           penalty *= (1.0 - c)
       return min(0.4, 1.0 - penalty)  # cap to keep the architecture honest
   ```

   Source for both edge types is already implemented in
   `_collect_safety_risks` (lines 312–397). Pulls from
   `graph.get_neighboring_edges(compound_id, edge_types=[EdgeType.CAUSES_AE])`
   AND `graph.get_neighboring_edges(target_id, edge_types=[EdgeType.TARGET_ASSOCIATED_AE])`.
   Round-20's `_compute_safety_penalty` reuses that retrieval, just
   converts the SafetyRisk list to a single scalar.

2. **Modify `PredictionResult`** to expose all three numbers:
   ```python
   class PredictionResult(BaseModel):
       efficacy_probability: float       # mechanism-only chain geomean
       safety_penalty: float             # [0, 0.4] subtractive
       overall_probability: float        # efficacy * (1 - safety_penalty)
       ...
   ```
   Existing callers that read `overall_probability` get the new
   combined semantics; the breakdown is available for introspection.

3. **Modify `PredictionEngine.predict`**: compute the efficacy chain as
   today (call it `efficacy_probability`), then call
   `_compute_safety_penalty(chain)`, then set
   `overall_probability = efficacy_probability * (1 - safety_penalty)`.
   Update CI computation similarly (apply the safety_penalty
   multiplicatively to CI bounds too).

4. **Tests** (`tests/test_prediction.py` extension):
   - `test_safety_penalty_zero_when_no_ae`: chain with no AE edges →
     overall_probability == efficacy_probability.
   - `test_safety_penalty_pulls_overall_down`: chain with one
     compound-specific severe AE → overall < efficacy by the penalty.
   - `test_target_class_ae_contributes_for_novel_compound`: compound
     has no causes_ae edges, but its target has target_associated_ae →
     penalty is non-zero (the on-mechanism class signal).
   - `test_safety_penalty_caps_at_0_4`: enormous AE evidence still
     keeps overall_probability above efficacy * 0.6.
   - Existing `test_overall_probability_*` tests need updating to use
     the new field names where they expect mechanism-only behavior.

5. **Update eval scripts** (`scripts/eval_holdout_v2.py`,
   `scripts/eval_in_sample.py`, `scripts/case_study_audit.py`):
   - When reporting per-trial output, show all three numbers:
     `P(success) = 0.65 = efficacy 0.78 × (1 - safety 0.17)`.
   - Verdict logic in `case_study_audit.py`: if `safety_penalty > 0.10`
     AND the trial's literature failure mode is safety-driven (torce,
     CAR-T-CRS, anti-CTLA-4-irAE-stop), tag the verdict as
     MATCH-VIA-SAFETY rather than DISAGREE.

**Tunables to revisit after first run**:
- The `min_belief` + `min_evidence` thresholds in `_compute_safety_penalty`
  (currently 0.5 and 1.0; could shift to be stricter)
- The 0.4 penalty cap (currently caps at making efficacy 60% of itself;
  could tighten to 0.3 or relax to 0.5)
- The `log(50)` saturation point (matches `_trust_weight`)

**Caveat that shaped the design**: torcetrapib has zero AE evidence in the smoke corpus (one trial, no posted results). Even with integration, that specific case won't change for THAT trial until more torcetrapib trials (or CETP inhibitor class trials) accumulate. But for well-evidenced compounds (nivo, ipi, bevacizumab, the BRAF/MEK families) it would meaningfully shift predictions.

**Files most relevant to start**:
- `src/prediction/path_query.py` — engine + helpers (look at `_collect_safety_risks` near line 312 first)
- `src/graph/models.py` — `EdgeType.CAUSES_AE` and `EdgeType.TARGET_ASSOCIATED_AE` definitions
- `tests/test_prediction.py` — extend `TestWeightedGeomeanPredict` + add a `TestSafetyPenalty` class

**Estimated effort**: 4–6 hours including test surface updates.

Caveat: torcetrapib has 0 AE evidence in the smoke corpus (one trial,
no posted results). Even with integration, that specific case won't
change until more torcetrapib (or CETP inhibitor class) trials
accumulate. But for well-evidenced compounds (nivo, ipi, bevacizumab,
the BRAF/MEK families) it would meaningfully shift predictions.

---

## Round-19 — incremental graph build (DO NEXT SESSION, before full n=281 build)

Surfaced during the round-18 multi-indication build (2026-05-20). Every
`build_graph.py` run today wipes the export snapshot and re-marches the
populator + attributor across all trials in the corpus. LLM calls are
amortized to zero (extractor / classifier hit per-trial caches), but
wall-clock and orchestration cost grows linearly with corpus size. For
sequential trial addition ("add 30 more breast cancer trials to the
existing multi_indication graph") this is the wrong shape. We need an
incremental add mode that doesn't double-count.

**Target CLI shape**:

```bash
# Add new trials by NCT id list (case-study use case)
python -m scripts.build_graph --base-snapshot data/exports/multi_indication_annotated.json \
  --add-trials NCT00112918,NCT00134264,...

# Or: extend with a new indication via search
python -m scripts.build_graph --base-snapshot data/exports/multi_indication_annotated.json \
  --add-corpus breast_cancer_30   # reads data/corpora/breast_cancer_30.txt
```

**Concrete steps when picking this up**:

1. **Add a `--base-snapshot` CLI flag in `scripts/build_graph.py`** (next to `--corpus`). When present, skip the `Step 0: wipe_outputs` block — both initial and annotated snapshots remain on disk. Load the existing annotated snapshot at the start instead of starting from an empty `GraphStore`.

2. **Add a `--add-trials` flag** that accepts a comma-separated NCT list. Plus `--add-corpus <name>` for the indication-search case. Both bypass the existing `--corpus` requirement; they short-circuit `fetch_trials()` to fetch ONLY the named NCT ids.

3. **Make the populator's step 2 (Canonicalize conditions + seed Indication / Population nodes) skip trials whose subgraph already exists**. Today it iterates all trials in `trials=[...]`. Wrap with a check: `if trial.nct_id in graph.trial_subgraphs: continue`. The existing `add_node` / `add_edge` ops are already idempotent on dup keys; this is just to avoid recomputing the LLM canonicalization + population_features calls.

4. **Add attribution idempotency** (the critical safety check). Today the attributor reads `data/annotations/*_classification.json`, calls `attributor.attribute(...)` for every trial with a sidecar TrialSubgraph, and applies updates. Without a guard, running it twice double-counts evidence on each edge. Implementation:
   - Add a `graph.applied_attribution_trial_ids: set[str]` field on `GraphStore` (serialized in the snapshot JSON).
   - In `_main` of `attributor.py`, before calling `attributor.attribute(...)` on a given trial, check `if trial_id in graph.applied_attribution_trial_ids: console.print(f"  skipping already-attributed {trial_id}"); continue`.
   - After `apply_updates` succeeds, `graph.applied_attribution_trial_ids.add(trial_id)`.
   - On `import_snapshot`, restore the set from JSON (empty default for old snapshots).
   - On `export_snapshot`, serialize the set.

5. **Tests** (`tests/test_build_graph.py` extension):
   - `test_incremental_add_doesnt_double_count`: build a tiny graph with 1 trial, capture an edge's `evidence_strength`, run the incremental add of the SAME trial again, assert evidence_strength didn't change.
   - `test_incremental_add_appends_new_trials`: build with trial A, incremental add of trial B, assert trial B's NCT is in `trial_subgraphs` AND trial A's subgraph is preserved AND the snapshot file timestamp updated.
   - `test_base_snapshot_skips_wipe`: assert that `wipe_outputs` is NOT called when `--base-snapshot` is passed.

**Engineering risk**: getting attribution idempotency right is subtle. The Beta-Binomial updates aren't trivially undo-able, so the guard MUST run BEFORE `apply_updates`. The set must be transactional (set add + apply commit together).

**Files most relevant to start**:
- `scripts/build_graph.py` (CLI + orchestration)
- `src/annotation/attributor.py` (the `_main` function + Attributor class)
- `src/graph/store.py` (the GraphStore needs the new field + snapshot persistence)
- `tests/test_build_graph.py` + `tests/test_attributor.py`

**Estimated effort**: half-day to a day. Most of the complexity is the idempotency dance + getting tests right.

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
