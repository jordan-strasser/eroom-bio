# Current graph-fix tasks (2026-06-09) — compaction-proof handoff

Purpose: durable task list so a fresh/compacted session can pick up the
graph-fundamentals work. Branch **`fix/st-field-faithfulness`**. North-star lens:
the n=10→N AUROC should rise; if not, it localizes to one of the 6 causes
(ingestion / data→node mapping / per-trial edge-assignment / **merging** /
graph structure / predictor query). The work below targets **merging (#4)** and
**per-trial edge-assignment (#3)** — the two most likely roots of current noise.

## Session status snapshot (branch `fix/st-field-faithfulness`, 1370 tests green)
COMMITTED (3 commits):
- `db4736f` — Bug B (`applied_n_eff` persistence → field/LOO replay faithful, affects
  gap 118→0), #2 pre-merge-coordinate (s,t) localization, **field→PUBLIC** (open-core),
  bottom-up merge **batch-encoder** perf fix (n=500 merge no longer hangs).
- `6d0656b` — **T1** merge faithfulness: content-dedup + `applied_weights` replay.
- `34c6a28` — **T4a** merge-tier config (BioLORD mech/bio, SapBERT indication/pop/endpoint).
- `#1` line-of-therapy: diagnosed INERT (structured re-test); no cosmetic fix.

NEXT LEVER = **T4b** (mechanism description-identity flip) — the #1 lever; not yet
started (deferred for a focused pass). **multi_500 is REBUILT** (multi_500_annotated.json,
3851 nodes/497 trials, Bug B + public field; predates T1/T4a so it still over-counts +
has the broken mechanism merge). Build gotcha: `--max-trials` defaults to **10** — always
pass `--max-trials 500`. Field measurement on multi_500 (scalar_vs_field) still un-run
(lower priority — the field is a band-aid over the mechanism noise T3 exposed).
- **Re-attribution harness**: `data/exports/neff100_initial.json` is the n=100
  **merged, pre-attribution** graph (0 applied_attribution_trial_ids, DB-only edge
  evidence). It can be re-attributed N ways cheaply (no re-populate/re-merge) via
  `attributor._main(annotations_dir, graph_path, output_path, ...)`.

---

## TASK 1 — Merge faithfulness fixes (CONFIRMED bugs; START HERE) ⭐
Edge merge lives in `node_merge._merge_belief_data` → `biology_merge._replay_belief`
(`src/graph/biology_merge.py:177-187`) + `_merge_belief_data` (node_merge.py ~329 /
biology_merge ~190) + `_redirect_edges` (node_merge ~362). Three confirmed defects:

**1a. Dedup over-counts replicated DATABASE facts.** Dedup key is
`(source_id, timestamp)`. Each trial resolved in isolation (bottom-up Phase 1)
stamps its OWN copy of an OT/DB fact with its OWN `datetime.now()`, so copies have
DIFFERENT timestamps → NOT deduped. CONFIRMED on `neff100_initial`:
`paclitaxel → ENSG00000188229` = Beta(80.8, 5.2), E[p]=0.940, from **7 records all
`opentargets:CHEMBL428647`** (7 timestamps, 1 source). One binding fact (n_eff 12,
p_obs 0.95) should give Beta(12.4,1.6) E[p]≈0.886; replicated 7× → E[p] 0.94 with
evidence_strength ~85 vs ~14 — spuriously near-certain. **The more trials reference
a compound, the more its `affects` belief inflates** → this is the handoff's
unexplained "affects beliefs uniformly ~0.9, undifferentiated" = SATURATION, and
it's REPLICATION not biology. (`placebo→target` rows compound this with the
wrong-ChEMBL mis-resolution, CHEMBL521=ibuprofen on placebo → 24 copies.)

**1b. `_replay_belief` re-derives with NOMINAL n_eff only.** It calls
`effective_n_for_evidence(source_type, quality_score)` and `BUCKET_TO_P_OBS[support]`
— dropping (a) **`applied_n_eff`** (the explaining-away split — THIRD instance of
Bug B, missed when fixing the field+LOO paths), (b) **`n_obs`/√N precision**, (c)
the **redundancy/cluster discount** (the exact mechanism meant to stop 1a — never
called). Consequences: fresh build (merge is PRE-attribution → DB-only) gets the 1a
over-count; **re-merge over ATTRIBUTED beliefs** (incremental `--add-trials`,
`biology_merge` on an existing graph, `assemble_v2 --merge`) re-derives the
trial-outcome records nominally too → **UNDOES explaining-away self-protection**.
Latent now (fresh builds), bites the moment the n→N loop goes incremental.

**1c. Dedup key is type-wrong.** Idempotent DB facts SHOULD collapse to one;
per-arm trial-outcome records (same NCT, maybe same timestamp) must NOT — but the
`(source_id, timestamp)` key does the opposite on both.

### Fix plan (cheapest first)
1. **Content-dedup idempotent DB records at merge**: key DATABASE_* records by
   `(source_id, support, edge)` WITHOUT timestamp so replicas collapse to one; keep
   CLINICAL/trial records keyed to include `arm_id` (context["arm_id"]) so distinct
   arms survive. In `_merge_belief_data` (biology_merge ~190 + node_merge ~329).
2. **Make `_replay_belief` faithful**: use a shared `applied_weights(record)` helper
   (prefers `applied_n_eff`/`applied_p_obs`, else nominal) + apply `redundancy_factor`
   over the deduped set. Fixes both the over-count AND the explaining-away-undo, and
   FINISHES Bug B (its 4th replay site). `src/inference/beliefs.py` already has the
   building blocks; consider adding `applied_weights` there and reusing in
   `provenance._replay_records`, `holdout_thesis_analysis`, `materialize_belief_field`.
3. **[Architectural, later]** Attach DB/molecular facts to the canonical edge ONCE
   (a property of the (compound,target) pair), separating the molecular manifold from
   the outcome manifold.

### Measurement
Re-attribute (or re-merge) `neff100`, expect `affects` E[p] ~0.94 → ~0.87 and —
more important — **regain SPREAD across compounds (de-saturation)**. Then LOO/forward
AUROC delta. Tools: `scripts/compound_target_merge_audit.py`, `scalar_vs_field_auroc.py`.

Files: `src/graph/biology_merge.py`, `src/graph/node_merge.py`, `src/inference/beliefs.py`.

### T1 RESULT (2026-06-10) — DONE + 1370 tests green; hypothesis partly WRONG
Implemented: `beliefs.applied_weights()` helper; `_replay_belief` uses it (4th Bug-B
site faithful); `_evidence_dedup_key = (source_id, support, arm_id)` content-dedup in
BOTH `_merge_belief_data` (biology_merge + node_merge). Verified on neff100:
paclitaxel→ENSG **7→1** records, E[p] 0.940→0.886, **evidence_strength 86→14**;
placebo 24→1, strength 290→14. **BUT de-saturation hypothesis was WRONG**: affects
E[p] MEAN 0.727→0.721 (barely moves), std 0.164→0.159 (slightly DROPS). The over-count
inflated CONFIDENCE, not the MEAN — one OT binding (p_obs 0.95) already gives E[p]
0.886, so `affects` is inherently ~0.89 (validated-target ≠ trial-success ceiling).
Real value: correctness; **LEARNABILITY** (over-counted Beta(290,15) edges were FROZEN
to new evidence → likely contributor to the flat learning curve #3); faithful
explaining-away on re-merge/incremental; removes spurious popularity-correlated spread.
NOT the static-AUROC lever. OPEN: does un-freezing edges lift the LEARNING CURVE?
(rebuild + `scripts/evidence_learning_curve.py`). NOTE: the running n=500 build predates
this fix → its graph still over-counts (minor); a T1-clean rebuild would be needed to
test the learning-curve hypothesis.

---

## TASK 2 — Per-edge success/failure update-math experimentation
Site: `attributor._condition_chain_on_outcomes` (`src/annotation/attributor.py`
~820-878). Current (asymmetric): SUCCESS conjunctive → every edge `frac=1.0` at
`p_obs=0.80`; FAILURE/PARTIAL → explaining-away `frac_i=(1-E[p_i])/Σ` at
`p_obs=0.20/0.35`. Weight `n_eff_i = w_base·frac_i`, `w_base =
effective_n_for_evidence(phase,quality) × f_N(√N from n_obs) × gate_weight`.

**2a. Symmetry experiment** (owner wants to SEE how it changes the graph). Add env
flag `EROOM_EDGE_ATTR` with modes: `explain_away` (default, current) |
`symmetric_full` (every edge full w_base both directions = pure per-edge frequency) |
`symmetric_uniform` (split 1/L both directions) | `symmetric_explain` (explaining-away
weighting for BOTH success and failure). Re-attribute `neff100_initial` each mode,
compare resultant edge-belief distributions (mean, SPREAD/differentiation) + LOO
softmin AUROC. The asymmetry rationale = conjunctive noisy-AND (success ⟹ all links
true; failure ⟹ ≥1 weak, unknown which) — principled but worth testing vs symmetric.

**2b. Incorporate effect_size + p_value into the weights (owner: "def need this").**
Today `effect_size`/`p_value` are RECORDED on the EvidenceRecord but the BACKBONE
update uses only the coarse 3-way outcome → fixed `p_obs` (0.80/0.35/0.20); `n_eff`
uses sample size (f_N) but not effect/p. PARTIAL machinery already exists:
`_hr_support_bucket` (attributor ~380) maps HR + CI → support bucket with a
significance gate (CI spanning 1.0 → AMBIGUOUS), and `_ae_support_bucket` for AEs.
EXTEND it: effect MAGNITUDE → finer support bucket / continuous p_obs (large-effect
success → STRONG_SUPPORT 0.95; marginal → WEAK 0.65); p_value/precision → n_eff
scaling. CAVEAT: needs sign/direction normalization (HR<1 good for survival, OR>1
good for response, etc.) — the extractor's effect_size isn't consistently normalized.

Files: `src/annotation/attributor.py`, `src/inference/beliefs.py` (BUCKET_TO_P_OBS,
effective_n_for_evidence). Harness: `attributor._main` on copies of
`neff100_initial.json` per mode → compare.

---

## TASK 3 — Instrument the merge (MechanismNode chain-description divergence)
Quantify the #2 (mapping) + #4 (merge) noise directly. SapBERT-NAME cosine merges
MechanismNodes (config: `enable_sapbert`, `sapbert_node_types=("MechanismNode",)`);
the Reactome pathway-ranker LABELS them. Observed noise: chains with
`mechanism_description="DNA cross-linking"` sit on node `R-HSA-512988` labeled
"receptor agonism".

Build `scripts/instrument_mechanism_merge.py`: for each MechanismNode, dump
{node id, node name/description (the Reactome label), set of DISTINCT chain
`mechanism_description`s that landed on it (from `graph.trial_subgraphs[*].chains`
where `chain.mechanism_id == node`), count}. Then quantify:
- **Over-merge**: nodes carrying ≥2 SEMANTICALLY DISTINCT chain-descriptions
  (BioLORD cosine between chain-descs low) → SapBERT-name merged unlike things.
- **Under-merge**: the same/near chain-description split across multiple nodes.
- **Label divergence**: cosine(node description, chain descriptions) — how often the
  canonical Reactome label diverges from the trial's stated mechanism.
Output a ranked report + summary rates. (Also worth: same instrument for BiologyNode
[BioLORD-description merge] and a check of whether Population/Indication SHOULD use
geometric merge — current recommendation: NO by default, over-merge risk: sibling
diseases / adjacent axis levels are cosine-close; measure fragmentation first.)

Files: read `data/exports/<area>_annotated.json` + trial_subgraphs; new script.

### T3 RESULT (2026-06-10) — DONE; MAJOR FINDING (likely the dominant AUROC noise)
`scripts/instrument_mechanism_merge.py` on neff100. **MechanismNode merge is BROKEN;
BiologyNode is CLEAN — the contrast names the fix.**
- MECHANISM (SapBERT on pathway NAME): 244 nodes, 81 multi-desc. OVER-MERGE **53/81
  (65%)** min intra-node cosine <0.5 — e.g. `R-HSA-416476 "G alpha (q) signalling"`
  carries "B-cell depletion" + "HDL elevation via niacin" + "beta-adrenergic blockade"
  + "calcium sensitization". UNDER-MERGE: **"receptor antagonism" on 33 nodes**,
  "enzyme inhibition" 29, "microtubule stabilization" 26, "kinase inhibition" 26,
  "DNA synthesis inhibition" 25. The mechanism nodes are garbage buckets.
- BIOLOGY (BioLORD on DESCRIPTION): 67 nodes, 10 multi-desc, only 2/10 borderline
  over-merge; label-divergence mean **0.967**; merges PARAPHRASES correctly
  ("incretin hormone regulation"≈"...stabilization" 0.89).
- ROOT: mechanism IDENTITY = the noisy Reactome/GO **pathway-ranker** (same desc →
  different pathways = under-merge; distinct desc → same generic pathway = over-merge)
  + merge via SapBERT on the GENERIC pathway NAME. Biology uses the trial's
  DESCRIPTION embedding for identity+merge → clean.
- **FIX (high priority, likely biggest lever found): make MechanismNode like
  BiologyNode** — identity + merge by the trial's mechanism DESCRIPTION embedding
  (tuned threshold to keep PD-1-blockade ≠ CTLA4-blockade), with Reactome/GO as
  METADATA only, not identity. `mechanism_affects` + `modulates_via` are built on
  these garbage nodes → explains flat chain AUROC; the (s,t) field helping
  `modulates_via` confirms it's routing around the wrong node identity.
  CONFIRMED AT n=500 (multi_500): OVER-MERGE **193/268 (72%)** — WORSE than n=100's
  65%; `R-HSA-416476 "G alpha (q) signalling"` carries **37** distinct mechanisms;
  "enzyme inhibition" on **71 nodes**, "kinase inhibition" 67. The mechanism noise
  GROWS with n (relevant to the n→N AUROC-rise diagnostic — it's actively dragging).
  Files: `src/graph/pathway_ranker.py` (the ranker), `src/graph/populate*.py`
  (mechanism node id assignment), `src/graph/node_merge.py` (cfg:
  sapbert_node_types MechanismNode → switch to description-embedding tier).

---

## TASK 4 — Merge-tier redesign by node semantics (owner-directed, HIGH leverage) ⭐
Principle (owner): embedding model should match what the node IS. SapBERT = biomedical
ENTITY-LINKER (UMLS synonym normalization); BioLORD = definition/DESCRIPTION embedder.
Current `MergeConfig` (populate_bottomup ~230): `sapbert_node_types=("MechanismNode",)`,
`biolord_node_types=("BiologyNode",)`, everything else id-only. T3 proved this is
backwards for mechanism. Redesign:

| node | identity | geometric merge |
|---|---|---|
| Compound | ChEMBL id | id/chembl |
| Target | Ensembl/gene id | id |
| Indication | id | **SapBERT** (disease synonyms; entity-linker keeps breast≠ovarian) |
| Population | axes id | **SapBERT** |
| Endpoint | id | **SapBERT** (endpoint-term synonyms; also helps the over-collapse) |
| **Mechanism** | **→ DESCRIPTION content-address** (revert pathway-identity) | **BioLORD** |
| Biology | description | BioLORD (already correct) |

### 4a. Config change (lower risk): MergeConfig in `populate_bottomup` ~230 →
`biolord_node_types=("MechanismNode","BiologyNode")`,
`sapbert_node_types=("IndicationNode","PopulationNode","EndpointNode")`. Needs a
rebuild to take effect. NOTE: mechanism BioLORD-merge is only effective WITH 4b (the
node description must BE the trial's mechanism, not the pathway desc).

### 4b. Mechanism identity flip (the real fix; bigger, test-heavy):
`populate._ensure_mechanism` (~2672) currently sets MechanismNode id+name = the
Reactome pathway (from `_resolve_gene_pathways` + pathway_ranker), fanning out ONE
chain per pathway (~2743). The content-address fn `_mechanism_id_from_description`
(populate:654) is SUPERSEDED but kept. FLIP: id = `_mechanism_id_from_description(desc)`,
name/description = the trial's mechanism_description, Reactome pathway → node.metadata
(interpretability only). Collapses the pathway fan-out to one mechanism per stated
action (matches the stated-chain eval, removes the over-merge). BROAD test impact
(many tests assert R-HSA mechanism ids) + needs rebuild + re-instrument
(`scripts/instrument_mechanism_merge.py`) to confirm over-merge 65%→low.
Files: `src/graph/populate.py` (_ensure_mechanism, the gene-pathway fan-out,
mechanism node creation), `src/graph/pathway_ranker.py` (demote to metadata),
`src/graph/node_merge.py` cfg.
BioLORD-vs-SapBERT for mechanism (owner): BioLORD — descriptions are the substrate;
sibling over-merge (PD-1 vs CTLA4 blockade) is mitigated because the TARGET node already
separates them upstream + a tuned biolord_threshold.

### T4b RESULT (2026-06-10) — DONE; over-merge collapsed; fan-out → metadata (by design)
Implemented the identity flip: `populate._populate_trial_mechanisms` now sets MechanismNode
id = `_mechanism_id_from_description(mech_desc)` (content-address of the trial's STATED
action), name/description = the action, and the gene's Reactome footprint → `pathway_ids` +
`metadata.reactome_pathways` (interpretability only; ONE mechanism per action, NO pathway
fan-out). Model: added `pathway_ids` + `metadata` to MechanismNode. Tests: rewrote the 5
pathway-identity asserts in test_populate.py to description-identity; **1370 green**.
INSTRUMENT (t4b_n50_initial, 37 trials): MechanismNodes **244→26**; OVER-MERGE
**65-72% → 2 of 4** multi-desc nodes — and both residuals are the **enum-slug fallback**
buckets (`enzyme_inhibition` swallowing CETP/HMG-CoA, `receptor agonism`), not the clean
`mech:` description nodes. LABEL-DIVERGENCE mean 0.955.
ROOT of the residual: identity-time `mech_desc` lookup did EXACT `(nct,arm,key)` only while
the description-backfill (`_lookup_chain_intervention_desc`) had cross-arm + salt-form
fallbacks → a specific action whose entry sat under another arm fell to the generic enum
slug. FIXED: factored `_match_intervention_entry` (shared resolver) so identity == backfill.
Re-build `t4b_n50b` in flight to confirm the 2/4 drops further.

### T4b DESIGN DECISION (owner-raised) — Reactome breadth: METADATA, not nodes
Owner asked: did we lose the Reactome fan-out (wanted breadth for cross-trial triangulation,
less noise)? Reasoned from north star (cross-trial accumulation → prediction):
- Prediction = softmin over the trial's STATED chain. A node helps ONLY if the chain walks
  it. Pathway nodes hung off the mechanism (composed_of) are OFF-chain → feed P(success)
  nothing without an added belief-pooling step → complexity w/o payoff. **Rejected new
  pathway nodes.**
- The breadth that MATTERS for prediction (cross-INDICATION mechanism transfer — the literal
  north-star sentence) is what T4b REPAIRS: pathway-identity fragmented "kinase inhibition"
  across 26 nodes AND polluted generic pathways with 37 unrelated actions; description-
  identity makes it one clean node that pools across indications.
- Cross-TARGET convergence (different targets → same downstream) already happens on-chain at
  the clean BIOLOGY layer (label-match 0.967). Polypharmacology footprint is preserved as
  node metadata (not lost, just not structural).
- Empirical tell: fan-out breadth was present in every prior build; learning curve FLAT
  n=5→472 while mechanism noise GREW with n. Breadth-as-implemented = noise, not accumulation.
DECISION: keep footprint as metadata; **let PREDICTION arbitrate** — measure clean-vs-noisy
accumulation (evidence_learning_curve.py + honest holdout). Add pathway-convergence pooling
ONLY if it adds signal beyond biology, and then as a pooling fn over metadata / the (s,t)
field, NEVER as off-chain nodes.

## TASK 6 — Clean pathway fan-out as its own rung (owner-directed ARCHITECTURE) ⭐⭐
Owner decision (2026-06-10): T4b cleaned the mechanism noise but REMOVED the
multi-candidate credit-assignment substrate the fan-out provided. Data confirms
(same-corpus n=50, mechanism_affects evidence/edge): NOISY fan-out 3.07/edge
(max 23) vs CLEAN T4b 1.02/edge (max 2) — the fan-out pooled ~3× more cross-trial
evidence on mechanism→biology edges (the "success/failure up/down-votes shared
mechanistic edges" substrate), but much of it was over-merged garbage (65-72%).
KEY INSIGHT: the noise was NOT the fan-out — it was (1) the pathway RANKER mapping
stated-action→one pathway under weak context, (2) SapBERT merge on generic pathway
NAME. A fan-out from CURATED Reactome membership + R-HSA id-merge (deterministic) is
clean: hubs self-neutralize (in everything → belief ~0.5 → softmin ignores),
discriminative pathways carry signal.

**CHOSEN ARCHITECTURE (owner, 2026-06-10) — OPTION 2: clean pathway fan-out AS the mechanism.**
`Compound → Target → Mechanism(= curated Reactome pathway, FAN-OUT) → Biology → Indication/Pop/Endpoint`
- The mechanism node IS the target's downstream Reactome pathway (RAF/MAP cascade, PI3K/AKT),
  from CURATED membership (NOT the ranker), one target → its footprint (capped, + BioLORD
  `semantic_relevance_floor` to drop off-context leaves like the TUBB4B "flagellated sperm
  motility" problem). MERGE BY R-HSA ID (deterministic Tier-1) — NOT SapBERT-on-name, NOT BioLORD.
- Pathways SHARED across targets (EGFR+BRAF+MEK → RAF/MAP cascade) ⇒ outcomes up/down-vote the
  shared pathway→biology edges = cross-trial credit assignment. Hubs (in everything) self-
  neutralize (belief→0.5 → softmin ignores); discriminative pathways carry signal.
- This REVERTS T4b's stated-action identity (mechanism = pathway again) but fixes the ORIGINAL
  noise (membership not ranker; id-merge not name). Stated action + MechanismCategory → queryable
  node/edge METADATA (keeps polypharmacology / mutation-specific / off-target info). Over/under-
  merge metrics MOOT (drugs sharing a pathway is the point). Why NOT the 4-rung (action rung +
  pathway): action ≈ target×direction×pathway (redundant; direction already on modulates_via),
  extra rung adds empty Beta(1,1) edges that drag the weakest-link softmin, and "action
  constrains fan-out" needs the ranker. Full discussion: [[project_t4b_mechanism_identity_flip]].

### Phased plan (each phase: branch + tests green + audit before next) — see NEXT_SESSION.md
**Phase A — backoff RULE fix (prereq; precision-weighted, not "any leaf evidence wins").**
`path_query._resolve_indication_edge` (line 434) returns the FIRST ancestor with
`evidence_strength > 0` → ONE uveal_melanoma trial OVERRIDES 50-trial melanoma (no combine,
no precision weight). Fix = hierarchical partial pooling: parent belief as PRIOR, leaf
evidence updates it (1 leaf trial barely moves a rich parent; 20 leaf trials dominate).
Applies to indication + population NOW; prereq for any hierarchy backoff. Files:
`src/prediction/path_query.py` (_resolve_indication_edge + population backoff),
`src/inference/beliefs.py` (combine util). Cheap, testable w/o rebuild.

### PHASE A RESULT (2026-06-10) — DONE; 1379 tests green; cross-indication borrow MEASURED
Added `beliefs.pool_hierarchical` (+ `_cap_concentration`, `_POOL_PRIOR_STRENGTH=20` τ, env
`EROOM_POOL_PRIOR_STRENGTH`): fixed-concentration hierarchical Beta-Binomial partial pooling.
Coarsest evidenced level seeds the prior; each finer level updates a τ-CAPPED copy of the running
pool with its own evidence mass (α−1, β−1). Both resolvers (`_resolve_indication_edge`,
`_resolve_responds_differently`) now collect ALL evidenced levels and pool, returning
(most-specific-evidenced-id, pooled_belief) instead of the first-evidenced level outright.
- DISJOINTNESS verified: leaf-anchored chains attribute each trial to its leaf indication / own
  population slug; SUBTYPE_OF + axis-subset parents are structural (roll-up deferred to predict) →
  pooling does NOT double-count.
- SELF-EXCLUSION: pooled belief carries the LEAF level's evidence (leaf enters UNCAPPED), so
  `provenance._belief_excluding`'s delta-adjust removes a held-out trial's leaf contribution
  EXACTLY while the capped ancestor mass is a constant prior offset. When the held-out trial is the
  leaf's only evidence, self-excluding it now LEAVES the parent borrow (instead of dropping the
  edge) → the honest holdout MEASURES cross-indication transfer. Fixed the 2 test_provenance
  regressions this surfaced. Single-level pooling = exact no-op.
- τ=20 calibrated by SCALE not holdout (n=500 parent biology_drives strength p90 ~22) — see
  memory `tuning_pool_prior_strength`.
- AUDIT (`scripts/backoff_pooling_diagnostic.py`, deterministic, no rebuild, multi_500_annotated):
  128 firing indication pairs (biology_drives 73; +deeper ancestors & endpoint_captures = 128);
  responds_differently 0 (all evidenced pop edges single-axis → population backoff inert NOW).
  Prediction shift OLD leaf-only → NEW pooled: |Δmean| mean 0.048 / median 0.025 / p90 0.104 /
  max 0.279; 45/128 > 0.05. her2_positive_breast 0.51→0.79, type_1_diabetes 0.49→0.70,
  st_elevation_MI 0.54→0.35 (parent more pessimistic — correct). Full note:
  `data/dev/phase_a_backoff_pooling_findings.md`. Honest AUROC delta → Phase C (needs rebuild).
  Tests: `tests/test_beliefs.py::TestPoolHierarchical` (8) + resolver/self-exclusion regressions.
**Phase B — mechanism = pathway fan-out (revert T4b identity, CLEAN).** In
`populate._populate_trial_mechanisms`: mechanism node id = R-HSA pathway (from `_resolve_gene_pathways`
CURATED membership + relevance floor), fan out one chain per pathway; stated action +
MechanismCategory → node metadata. Merge cfg: MechanismNode → id-merge only (REMOVE from BioLORD/
SapBERT tiers in `populate_bottomup` MergeConfig). Re-instrument; rebuild. Migrate tests
(the 5 T4b description-identity tests flip back toward pathway-identity + metadata-action).

### PHASE B RESULT (2026-06-10) — DONE; committed `991a25b` + `ade0fac`; 1381 tests green
Implemented exactly as specced. `_populate_trial_mechanisms` fans out one MechanismNode +
one chain per curated Reactome/GO pathway (id = stable_id; `_resolve_gene_pathways` re-rank
ORDERS the fan-out instead of picking one; GO relevance floor active). `_ensure_mechanism`
accumulates stated action + category as node metadata (`stated_actions` /
`mechanism_categories`). MergeConfig: MechanismNode → id-merge only (removed from BioLORD).
Tests: `git checkout 10a622f^ -- tests/test_populate.py` restored the 5 pathway-identity tests
(T4b only touched those; nothing since), updated the merge-tier test for id-only, added a
metadata-accumulation test. Model docstring refreshed.
- AUDIT (phaseb_n50, 50 trials multi_500 head): 50→500 chains (fan-out); 148/155 mech nodes are
  real R-HSA/GO pathways (7 enum-slug fallback, inflated by 26 Reactome rate-limit failures);
  17 pathways converge across ≥2 targets (RAF/MAP, PI3K/AKT, DNA replication); cross-trial
  pooling works (RAF/MAP referenced by 4 trials → 1 node). `test_shared_pathway_is_unambiguous`
  confirms EGFR(inhibitor)+IL2RA(agonist) → 1 RAF/MAP node, drug-agnostic, per-drug direction.
- HONEST: mechanism→biology evidence/edge **2.77** (fan-out) vs **3.06** (t4b_n50 ref, n=37) —
  does NOT reproduce the memory's predicted 3.07-vs-1.02 pooling gap (different trial sets +
  metric counts LINCS/DB records). The fan-out's value is cross-target SHARING; whether it LIFTS
  prediction is **Phase C** (learning curve was FLAT — MEASURE, don't assume). Full note:
  `data/dev/phase_b_pathway_fanout_findings.md`.
- BUG FOUND + FIXED (`ade0fac`): 2/155 mech nodes kept `#NCT` scope — Option 2's R-HSA namespace
  collides with the BiologyNode Reactome fallback; a transient orphan biology node blocked
  `_canonicalize_ids` then got pruned. Fix: prune orphan biology BEFORE canonicalize in
  `build_bottomup`. Regression test + source-order guard.
**Phase C — predictor credit-assignment over the fan-out + MEASURE.** softmin/aggregate over the
fan-out's pathway→biology edges (marginalize over multiple pathways for the PRODUCT). Measure
holdout AUROC + learning curve clean-vs-noisy: did the fan-out accumulation LIFT prediction?
(Prior: learning curve FLAT n=5→472 ⇒ don't assume more candidates = signal; the corpus must
actually sort them — MEASURE, don't assume.)
NOTE new prediction math + identity change = architecture-level (CLAUDE.md): branch + green +
rebuild before merge. T4b committed `10a622f`. Same-corpus noisy-vs-clean comparison DONE
(mechanism_affects evidence/edge: noisy fan-out 3.07/max23 vs clean-T4b 1.02/max2).

### PHASE C measure-first RESULT (2026-06-10) — marginalize NOT needed; honest AUROC + curve remain
`predict_clinical_hypothesis` returns the `min` overall_probability over a compound's stated chains.
The fan-out makes that a min over MANY pathway-chains (mean 11.3/trial, max 49), so the Phase-C
hypothesis was "min penalizes broad footprints → marginalize over pathways." MEASURED first
(`scripts/phasec_aggregation_diagnostic.py`, phaseb_n50b, 33 trials, IN-SAMPLE so relative ranking
is the signal): **min 0.952 > mean 0.913 > median 0.861 > max 0.826 > best_evid 0.817**. `min` is
the BEST separator; every marginalization alternative is WORSE (a failing trial reliably has ≥1 weak
pathway-chain `min` catches). **DECISION: keep `min`; do NOT implement the speculative marginalize
change** (committed `d38efd1` diagnostic; finding `data/dev/phase_c_aggregation_findings.md`).
REMAINING (heavy compute, focused follow-up): (1) HONEST out-of-sample AUROC via an
exclude-from-attribution / kfold re-attribution fan-out build (the n=50 number is in-sample),
compared to the historical ~0.567 kfold / ~0.51 forward; (2) learning curve clean-vs-noisy across
n=10/50/100/250/500 (prior FLAT n=5→472 — MEASURE). GOTCHAS: `--keep-annotations` loads cache (0
re-extract cost); NCT00282308 has corrupt cache (non-fatal skip); merged-graph chains can reference
pruned compound nodes → `engine.predict` KeyError (skip).

## TASK 5 — Full merge verification: generalized noise checker (owner: VERIFY, don't assume) ⭐
The 4a/AE SapBERT tiers (Indication / Population / Endpoint / **AdverseEvent**, added
2026-06-10) are **config-only and UNVERIFIED** — SapBERT could over-merge related-but-
distinct entities or mis-threshold. Build the verification BEFORE trusting it.
`scripts/instrument_mechanism_merge.py` checks DESCRIPTION nodes (chain-descriptions on a
node). GENERALIZE for ENTITY nodes:
- **UNDER-merge**: pairs of entity nodes (Indication/Population/Endpoint/AE/Compound/Target)
  whose NAME/aliases are SapBERT-synonyms (cosine > threshold) but are SEPARATE nodes ⇒
  should have merged ("NSCLC" vs "non-small-cell lung cancer"; "MI" vs "myocardial infarction").
- **OVER-merge**: a node whose accumulated names / `metadata.merged_from` aliases span LOW
  SapBERT cosine ⇒ merged DISTINCT entities (breast vs ovarian; a MedDRA term swallowing
  unrelated PTs).
Run the FULL sweep (ALL node types) on a 4a+AE rebuild → per-type under/over-merge rates.
This is the gate that answers the owner's question: "is SapBERT working for indication/
population/endpoint/AE?" (Right now: unknown — current multi_500 predates 4a, so those are
still id-only merged.) Files: extend `scripts/instrument_mechanism_merge.py` or sibling
`scripts/instrument_entity_merge.py`; REQUIRES a rebuild with 4a+AE first.

## Deferred / context threads (don't lose)
- **n=500 field measurement**: when the rebuild finishes, run
  `scalar_vs_field_auroc --graph multi_500_annotated.json --field multi_500_annotated.json
  --bandwidth 0.25` (field is now IN the public snapshot). Compare to pre-Bug-B
  (softmin 0.648/field 0.550). Expect field ≈ scalar (faithful), modulates_via +~0.16.
- **Populations/indications geometric merge** (owner question): recommend NOT by
  default (over-merge risk); the id+EFO-SUBTYPE_OF-hierarchy backoff is safer.
  Measure synonym fragmentation first (Task 3 extension).
- **Prediction architecture** (discussed, not yet actioned): topology-completion for
  bare hypotheses is greedy-argmax (fragile to one strong-wrong edge) — marginalize
  over paths for the PRODUCT; not in the current holdout eval (which uses stated
  chains, so it doesn't pin current AUROC). MC sampling is for CI + non-linear softmin
  uncertainty propagation — principled, second-order for AUROC.

## Reproduction quick-refs
- venv: `source .venv/bin/activate`
- merge over-count check: load `data/exports/neff100_initial.json`, inspect an
  `affects` edge's evidence records (same source_id × N timestamps).
- re-attribute a mode: copy `neff100_initial.json` → tmp, `EROOM_EDGE_ATTR=<mode>`,
  run `python -m scripts.build_graph`-style attribution OR `attributor._main`.
