# Hyperparameter registry — every numerical knob the graph is built & predicted with

Status: **catalogue** (2026-06-05). Companion to `BENCHMARK.md` (the nested-CV tuning design) and `SCALING.md`. Built by an 8-agent exhaustive sweep of `src/` + `scripts/`; this is the *tuning surface* the nested-CV inner loop selects over.

## How to read this

Every knob is classed:
- **TUNE** — a genuine hyperparameter that changes beliefs / predictions / graph shape. These are the rows that matter.
- **STRUCT** — definitional/structural constant (enum sizes, regex shapes, id formats, unit conversions). Not tuned. Listed in §5 for completeness only.
- **PERF** — performance/IO only (batch sizes, timeouts, retries, concurrency). §6.
- **DERIVED** — computed from other constants at runtime (not independently set).

The TUNE knobs are split by **WHEN they take effect**, because that decides how expensive they are to sweep — the single most important fact for the harness:

| Tier | Re-evaluate cost | How the harness sweeps it |
|---|---|---|
| **§1 Predict-time** | ~seconds (re-run `predict` on a fixed graph) | Inner CV loop — full grid, cheap |
| **§2 Attribution-time** | ~minutes (re-attribute a snapshot from cached extractions; K-fold already does this per fold) | Inner CV loop — moderate grid; **n_eff/p_obs are replay-able from preserved `EvidenceRecord`s without a rebuild** (the key efficiency seam, see §7) |
| **§3 Build-time** | ~10min–3h (full populate/merge/geometry rebuild) | Outer coarse grid only; do NOT put in the inner loop |

---

## §1 — Predict-time knobs (cheap; the inner-loop workhorses)

These apply at `predict()` time on a frozen graph. Sweeping them needs no rebuild → put the whole grid here.

| Knob | file:line | Current | Env override | Controls | Range to sweep |
|---|---|---|---|---|---|
| `_SOFTMIN_T` | path_query.py:220 | 0.10 | `EROOM_SOFTMIN_T` | Softmin temperature — how sharply the weakest link dominates the chain | 0.03–0.30 log-spaced (+ check 0.50) |
| `EROOM_AGG` mode | path_query.py:294 | `softmin` | `EROOM_AGG` | Aggregation: softmin/geomean/harmonic/product/min | discrete {softmin, geomean, harmonic, product, min} |
| `_INFORMED_PRIOR_MEAN` | path_query.py:227 | 0.75 | `EROOM_PRIOR_MEAN` | Mean an under-evidenced edge defers to ("probably operative") | 0.50–0.85 |
| `_INFORMED_PRIOR_STRENGTH` | path_query.py:228 | 2.0 | `EROOM_PRIOR_STRENGTH` | Pseudo-obs strength of that informed prior | 0.5–5.0 |
| informed-prior on/off | path_query.py:236 | ON | `EROOM_INFORMED_PRIOR` | Re-prior weak edges to ~0.75 vs Beta(1,1) coin-flip | {0,1} |
| `_TRUST_LOG_SAT` | path_query.py:218 | log(50)≈3.91 | — | Evidence-strength → trust saturation point (reused in safety trust_factor) | strength 20–100 |
| `_SAFETY_PENALTY_CAP` | path_query.py:643 | 0.60 | — | Max safety drag (efficacy floors at 1-cap). *Hand-fit 0.4→0.6 on 5-trial audit (round-20)* | 0.30–0.80 |
| `_SAFETY_PENALTY_MIN_BELIEF` | path_query.py:634 | 0.55 | — | Min Beta-mean for an AE to enter penalty (must stay > 0.5) | 0.50–0.65 |
| `_SAFETY_PENALTY_MIN_EVIDENCE` | path_query.py:635 | 1.0 | — | Min evidence_strength for an AE to penalize | 0.5–4.0 |
| `_SEVERITY_GRADE_TO_WEIGHT` | path_query.py:52-56 | {1:.05,2:.05,3:.15,4:.30,5:.50} | — | CTCAE grade → penalty weight | scale family; grade-3 0.10–0.25 |
| `_UNKNOWN_GRADE_WEIGHT` | path_query.py:61 | 0.10 | — | Penalty weight when grade unobserved | 0.05–0.20 |
| `_SERIOUS_FLOOR_WEIGHT` | path_query.py:69 | 0.15 (=grade-3) | — | Round-29 floor for `serious=True` ungraded AEs | 0.10–0.30 |
| `_SAFETY_DLT_FLOOR` | path_query.py:136 | 0.15 | — | Floor share for AEs that merely occurred vs dose-limiting | 0.05–0.30 |
| DLT gate on/off | path_query.py:142 | ON | `EROOM_SAFETY_DLT_GATE` | Gate penalty on failure-causing tox vs occurrence | {0,1} |
| `belief_factor` slope | path_query.py:729 | (p-0.5)/0.5 | — | Three-gate belief scaling for AE penalty | family/exponent |
| field `DEFAULT_BANDWIDTH` | belief_field.py:43 | 0.25 | — (query-overridable via `--bandwidth`) | Cosine-kernel locality of the (s,t) field | 0.10–0.50 |
| field `FALLBACK_STRENGTH` | belief_field.py:44 | 2.0 | — | Pull of far-from-anchor queries back to scalar marginal | 0.5–8.0 |
| `off_tissue_weight` | store.py:110 | 0.3 | — | Query-time n_eff multiplier for off-tissue LINCS evidence | 0.1–0.7 |
| `find_paths` max_length | store.py:241 | 6 | — | Max path length in graph traversal queries | 3–8 (STRUCT-ish) |
| MC `n_samples` (predict) | path_query.py:425 etc | 10_000 | — | Monte-Carlo samples (variance vs speed; not accuracy) | PERF-ish |

**Eval-harness gates (decide the scored set, not the prediction — report sensitivity):**

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| `--min-overlap` | eval_holdout_compose.py:906 / kfold:130 | 5 | Min resolved chain-nodes (of 7) to score a trial | 3–7 (report AUROC-vs-coverage) |
| `--k` | eval_holdout_kfold.py:129 | 5 | K-fold split count | 5 (outer); inner derived |
| `--n-samples` (eval) | both | 2000 | MC samples in eval | PERF |

---

## §2 — Attribution-time knobs (re-attribute from cached extractions; K-fold already does this)

These take effect when `EvidenceRecord`s are folded into Beta posteriors. **n_eff & p_obs are replay-able from preserved records** (§7) — exploit that to sweep them without re-running the LLM attribution.

### Evidence weights — `EVIDENCE_TYPE_N_EFF` (beliefs.py:101-176) — the highest-leverage block

| Tier | line | n_eff | Note |
|---|---|---|---|
| CLINICAL_PHASE3 | 101 | 15.0 | anchor; comment: "refit against the labeled set to minimize Brier" |
| CLINICAL_PHASE2 | 102 | 6.0 | |
| CLINICAL_PHASE1 | 103 | 2.0 | safety/PK-dosed → below Phase 2 |
| GENETIC_MR | 104 | 10.0 | ≈ Phase-2 causal equiv |
| GENETIC_GWAS | 105 | 4.0 | ~50% replication |
| PRECLINICAL_IN_VIVO | 106 | 2.0 | |
| PRECLINICAL_IN_VITRO | 107 | 1.0 | the 1× baseline unit |
| DATABASE_OT_DIRECT | 135 | 12.0 | round-28 bump 3→12 (binding = curated fact) |
| DATABASE_CHEMBL | 136 | 10.0 | round-28 bump 3→10 |
| DATABASE_MAB_TABLE | 137 | 10.0 | round-28 bump 3→10 |
| DATABASE_OT_ASSOCIATION | 145 | 2.0 | aggregate/interpretive |
| DATABASE_ENDPOINT_PRIOR | 146 | 2.0 | FDA/ICH consensus |
| DATABASE_REACTOME_GO | 152 | 1.5 | between in-vitro & in-vivo |
| DATABASE_LINCS | 157 | 1.0 | in-vitro tier |
| DATABASE_INDICATION_TAXONOMY | 162 | 1.0 | |
| DATABASE_FALLBACK | 168 | 0.5 | double-counting halve |
| DATABASE_CROSS_REFERENCE | 173 | 0.3 | string overlap only |
| COMPUTATIONAL | 175 | 0.3 | |
| LITERATURE | 176 | 0.2 | lowest |

**Sweep strategy:** hold IN_VITRO=1.0 fixed, sweep the *clinical:in-vitro ratio* (PHASE3 ∈ 5–25) and the *binding-tier:clinical ratio* (OT_DIRECT/CHEMBL/MAB ∈ 3–15) — these two ratios are what the round-28 work actually contested.

### Per-observation strengths — `BUCKET_TO_P_OBS` (beliefs.py:66-72)

| Bucket | line | p_obs | | Bucket | line | p_obs |
|---|---|---|---|---|---|---|
| STRONG_SUPPORT | 66 | 0.95 | | WEAK_CONTRADICT | 70 | 0.35 |
| MODERATE_SUPPORT | 67 | 0.80 | | MODERATE_CONTRADICT | 71 | 0.20 |
| WEAK_SUPPORT | 68 | 0.65 | | STRONG_CONTRADICT | 72 | 0.05 |
| AMBIGUOUS | 69 | 0.50 | | | | |

Comment flags these as "miscalibrated LLM-bucket targets; calibration is a pending follow-up." Sweep as a symmetric family: extreme floor 0.02–0.10, AMBIGUOUS pinned 0.50.

### Outcome-conditioning p_obs (attributor.py:98-101) — folded onto every chain edge

| Knob | line | Current | Range |
|---|---|---|---|
| `_SUCCESS_P_OBS` | 98 | 0.80 | 0.70–0.90 |
| `_FAILURE_P_OBS` | 100 | 0.20 | 0.10–0.35 ("deliberately MODEST — failure ≠ falsification") |
| `_PARTIAL_SUPPORT_P_OBS` | 99 | 0.65 | 0.55–0.75 |
| `_PARTIAL_CONTRADICT_P_OBS` | 101 | 0.35 | 0.30–0.45 |

### Other attribution-time knobs

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| `OPERATIONAL_GATE_WEIGHT` | taxonomy.py:471 | 0.2 | How much operational (non-mech) failures move mechanism beliefs | 0.05–0.5 |
| `VALID_TEST_GATE_WEIGHT` | taxonomy.py:472 | 1.0 | Valid-test gate weight | fix at 1.0 |
| `_hr_support_bucket` HR cuts | attributor.py:399-409 | 1.5/1.25 (strong/mod), CI∋1→AMB | Literature-AE HR→bucket. *"first-pass; refit downstream of the calibration harness"* | strong 1.3–1.8, mod 1.15–1.35 |
| `_ae_support_bucket` rate cuts | attributor.py:462-471 | Δ20pp/RR3, 10/2, 5/1.5 | Per-arm AE incidence→bucket ("deliberately coarse") | strong-Δ 10–25pp, RR 2–4 |
| AE absolute-count gates | attributor.py:480-493 | ≥3/≥5 affected | Min affected patients to keep AE support | 2–8 |
| RR denominator floor | attributor.py:460 | max(c,0.5) | Control-rate floor for finite RR | 0.25–1.0 |
| arm-differential delta→bucket | attributor.py:204-213 | ±2→STRONG,±1→MOD | Modulation support from arm-pair outcome jump | discrete |
| classifier low-conf review | classifier.py:291 | <0.7 | Confidence below which a classification is flagged/downweighted | 0.5–0.8 |
| classifier ambiguous gap | classifier.py:295,298 | 0.15 | Top-2 failure-mode separation → ambiguous flag | 0.05–0.25 |

### AE class-effect propagation (ae_propagation.py — runs at build/attribution)

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| `_VOTE_N_EFF_PER_COMPOUND` | 98 | 4.0 | Pseudo-count each sibling compound adds to target_associated_ae | 1.0–8.0 |
| `_MIN_COMPOUNDS_FOR_TARGET_AE` | 105 | 2 | Min siblings to materialize a class-AE edge | {2,3,4} |
| `_MIN_EVIDENCE_STRENGTH_FOR_VOTE` | 102 | 1.0 | Min accumulated counts for a compound to vote | 0.5–4.0 |
| DLT-vote majority | 455 | 0.5 | Fraction failure-causing for a vote to count dose-limiting | 0.3–0.7 |
| support-bucket cutoffs | 385-391 | 0.75/0.55/0.40/0.25 | causes_ae posterior → vote strength | coupled set |

### Curated-DB score→bucket (curated_evidence.py)

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| OT assoc score cuts | 109-115 | 0.75/0.5/0.25/0.05 | OT target-disease score → support bucket | tune w/ §3 OT min_score |
| `ot_score_quality` floor / saturation | 129 | 0.2 / ÷5.0 | Single-source floor; quality saturates at 5+ sources | floor 0.1–0.3, sat 3–8 |
| `endpoint_class_to_bucket` | 142-149 | OS→STRONG, PFS/ORR→MOD, else WEAK | Endpoint class → bucket | discrete |

---

## §3 — Build-time knobs (full rebuild to evaluate; outer coarse grid only)

Changing any of these requires re-populate / re-merge / re-geometry. Each eval costs minutes–hours. **Keep these OUT of the inner loop** — pick a config per scaling rung, sweep coarsely if at all. Many were hand-set; flagged for one-time re-validation, not per-fold tuning.

### Merge / canonicalization

| Knob | file:line | Current | Env | Controls | Range |
|---|---|---|---|---|---|
| `biolord_threshold` | node_merge.py:80 | 0.85 | `EROOM_MERGE_COSINE` | Tier-3 BioLORD cosine-on-description merge | 0.82–0.97 |
| `sapbert_threshold` | node_merge.py:83 | 0.80 | — | Tier-2b SapBERT cosine-on-name merge | 0.78–0.90 |
| `_COMPOUND_EMBEDDING_SIMILARITY_THRESHOLD` | populate.py:3647 | 0.80 | — | Compound-node dedup cosine. *Calibrated 2026-05-20 on n=11* | 0.74–0.88 |
| `biolord_node_types` / `sapbert_node_types` | node_merge.py:91-92 | bottom-up: BioLORD→Biology, SapBERT→Mechanism | — | Which node types each embedding tier merges (comment: BioLORD over-merges PD-1 vs CTLA4=1.000) | categorical |
| `embedding_merge_pairs` thr | biology_merge.py:351 | 0.95 | `EROOM_EMBEDDING_MERGE` (OFF) | Biology semantic-twin merge. *0.92→0.95, best F1 ~0.53; flag stays OFF* | 0.93–0.985 if enabled |
| name-match symbol min-len | populate.py:3586 | ≥4 + `\b` | — | round-27: 3→4 (APP/MET alias English words) | {3,4,5} |
| name-match name min-len | populate.py:3592 | ≥5 + `\b` | — | Long-name cross-ref AFFECTS gate | {4,5,6} |

### Geometry (is-a / box embeddings)

| Knob | file:line | Current | Env | Controls | Range |
|---|---|---|---|---|---|
| `_HI` (box contain) | box_embeddings.py:37 | 0.90 | `EROOM_BOX_CONTAIN_HI` | is-a containment cutoff (the 11570→791 regime) | 0.80–0.97 |
| `_LO` (box overlap) | box_embeddings.py:38 | 0.30 | `EROOM_BOX_OVERLAP_LO` | sibling-vs-unrelated cutoff | 0.15–0.50 |
| `leaf_half_width` | box_embeddings.py:152 etc | 0.05 | — | Leaf-cube size (volume floor for sparse nodes) | 0.01–0.15 |
| `margin` | box_embeddings.py:153/191/443 | 0.05 / 0.10 (inconsistent!) | — | Parent-box padding → containment fraction | 0.0–0.25 (unify first) |

### Mechanism / pathway assignment

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| `_MECHANISM_PATHWAY_CAP` | populate.py:2886 | 8 | Reactome pathways → MechanismNodes per (constituent,target) | 2–12 |
| `_BIOLOGY_PATHWAY_CAP` | populate.py:2874 | 1 | Reactome pathways → BiologyNodes per (target,mech). *round-3.4: cap=3 just split signal* | {1,2,3} |
| `MECHANISM_PATHWAY_TOKENS` | pathway_ranker.py:68-83 | 13 mech→token sets | Re-rank vocab picking each chain's mechanism/biology identity | vocab coverage |
| `_MIN_TOKEN_LEN` | pathway_ranker.py:54 | 4 | Min token len for overlap (vegf↔vegfa works, il↔iliac doesn't) | {3,4,5} |
| `_overlap_score` norm | pathway_ranker.py:149 | matches/len(tokens) | Name-length normalization | raw/norm/sqrt |
| GO-vs-Reactome win gate | populate.py:2851 | strict `>` | GO replaces Reactome only on strict score improvement | strict/≥/margin |
| mechanism-validity name gates | mechanism_validity.py:29-65 | regex/token sets | Prune non-mechanism Reactome/GO (423→402, AUROC +0.014) | curated |

### Population / endpoint identity (← Phase 2 enrichment target)

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| `_DEFAULT_POPULATION_AXES` | populate.py:142-147 | {line,stage,extent,setting} | Parent-pop coarsening whitelist (cross-trial pooling vs fragmentation) | 4-of-7 axis subsets |
| extent-vs-stage split | population_features.py:150 | {metastatic,resectable,unresectable,locally_advanced}→extent | Which stage levels stay in the coarse parent id | curated |
| `max_eligibility_chars` | population_features.py:221 | 4000 | Eligibility-text truncation into the LLM prompt | 3000–8000 |
| `ENDPOINT_CLASS_BUCKETS` | populate.py:313-333 | 16-entry class→bucket | Endpoint class → endpoint_captures prior | discrete |
| `_ENDPOINT_KEYWORDS` | populate.py:210-242 | ~11 ordered regex→class | Deterministic endpoint classifier | curated |

### Ingestion sourcing (priors & gates on external data)

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| OT `min_score` (general) | opentargets.py:268 | 0.1 | biology_drives association floor | 0.05–0.6 |
| OT `min_disease_score` (pathway→indication) | lincs.py:1103 | 0.3 | **inconsistent with the 0.1 above — harmonize** | 0.05–0.6 |
| LINCS support bucket | lincs.py:794 | MODERATE | mechanism_affects strength. *round-14 downgrade from STRONG (drowned clinical)* | {WEAK,MOD,STRONG} |
| LINCS `min_consistency` | lincs.py:651 | 0.5 | Fraction of cell lines hitting a pathway to make evidence | 0.3–0.8 |
| LINCS `min_genes_per_sig` | lincs.py:600/626 | 3 | Min perturbed genes for a pathway hit | 2–5 |
| LINCS `min_signatures` | lincs.py:652 | 2 | Min consensus signatures per compound | {1,2,3} |
| `_MODULATES_VIA_PRIOR_ALPHA/BETA` | lincs.py:56-57 | 5.0 / 1.0 | Beta prior (mean 0.83) on LINCS target→mechanism edge | α 2–10 |
| ChEMBL diagnostic gate | chembl.py:137-145 | max_phase<1 & ¬therapeutic_flag | Therapeutic-vs-diagnostic corpus filter (FDA diagnostics w/ phase=4 slip) | phase 0–2 |
| default Beta prior | models.py:710-711 | Beta(1,1) | Global uninformative edge prior | 0.3–2.0 (Jeffreys 0.5?) |
| `EvidenceRecord.quality_score` default | models.py:669 | 1.0 | [0,1] n_eff discount per record | — |

### Build-abort gates (silent-truncation guards)

| Knob | file:line | Current | Controls | Range |
|---|---|---|---|---|
| `min_classify_success_rate` | build_graph.py:878 | 0.80 | Fraction that must classify or build aborts | 0.70–0.90 |
| `min_subgraph_success_rate` | build_graph.py:915 | 0.75 | Fraction that must yield a subgraph | 0.70–0.90 |

---

## §4 — Environment-variable index (runtime overrides)

| Env var | Default | Knob | Tier |
|---|---|---|---|
| `EROOM_AGG` | softmin | aggregation method | predict |
| `EROOM_SOFTMIN_T` | 0.10 | softmin temperature | predict |
| `EROOM_PRIOR_MEAN` | 0.75 | informed-prior mean | predict |
| `EROOM_PRIOR_STRENGTH` | 2.0 | informed-prior strength | predict |
| `EROOM_INFORMED_PRIOR` | ON | informed prior on/off | predict |
| `EROOM_SAFETY_DLT_GATE` | ON | DLT safety gate | predict |
| `EROOM_BELIEF_FIELD` | OFF | materialize/use (s,t) field | build/predict |
| `EROOM_MERGE_COSINE` | 0.85 | biolord merge threshold | build |
| `EROOM_EMBEDDING_MERGE` | OFF | biology embedding-merge augment | build |
| `EROOM_BOX_CONTAIN_HI` | 0.90 | is-a containment cutoff | build |
| `EROOM_BOX_OVERLAP_LO` | 0.30 | sibling cutoff | build |
| `EROOM_NEFF_PRECISION` | OFF | precision/redundancy-aware n_eff path | attribution |
| `EROOM_PRIVATE_ROOT` | ~/.eroom/private | private-artifact root | IO |
| `EROOM_GRAPH_SNAPSHOT` | oncology_annotated.json | API startup graph | IO |

**Note:** the precision-aware n_eff path (`_REDUNDANCY_RHO`=0.5, `_N_REF_ANCHOR`=350, `_PRECISION_EXPONENT`=0.5, floor 0.5, ceil 2.5 — beliefs.py:196-287) is **gated OFF** by `EROOM_NEFF_PRECISION` and currently inert. BUT the attributor calls `_precision_multiplier` **directly** (attributor.py:779), bypassing the flag — so trial enrollment N already modulates outcome-conditioning weight regardless. Worth reconciling.

---

## §5 — Structural constants (reference; not tuned)

Definitional values found but **not** tuning targets, grouped by file. Listed so "every numerical value" is accounted for.

- **Taxonomy/enums:** 13-category `FailureMode` (taxonomy.py:16-29); 27 MedDRA SOCs (meddra_hierarchy.py); 7-tuple `CHAIN_BACKBONE` (populate_groundup.py:33); 7 `ALL_CHAIN_TYPES` (assemble_v2.py:25); `GENE_LEVELS`/`NON_GENE_AXES`/`_LINE_LEVELS`/`_STAGE_LEVELS` (subgroup_taxonomy.py, population_features.py).
- **Pydantic validators / id formats:** `_ENSG_PATTERN` ≥6 digits, `_GO_PATTERN` 7 digits, `_DRUGBANK_PATTERN`, `_CL_PATTERN` (models.py:827-837); credible_interval default 0.95 (models.py:745); slug truncations [:30]/[:60]/[:120]/[:12] (multiple).
- **Regex/format gates:** `VARIANT_PATTERN`, `HUGO_PATTERN` (subgroup_taxonomy.py); `_MAB_RE`/`_MAB_SUFFIXES` (antibody_target_resolver.py); `_NORM_RE`/`_INN_SHAPED_RE` (codename_resolver.py — min-len 4 is a borderline TUNE); dose/freq/route strippers (clinicaltrials.py).
- **Curated lookup tables (content, not scalars):** `CODENAME_TO_INN` (28), `MAB_COMPOUND_TO_TARGET_GENE` (~40), `NL_TARGET_TO_GENE` (~70), `_VACCINE_COMPONENT_TARGETS` (~24), `_CELL_THERAPY_COMPONENT_TARGETS` (~20), `_MOA_TEXT_TO_MECHANISM` (14), `_INDICATION_HIERARCHY`, `_QUALIFIER_PATTERNS` (~60). These are *content* expansions — grow them, don't sweep them.
- **Numerical guards:** `_LOG_FLOOR` 1e-12, alpha/beta clamps 1e-6, span floors 1e-9, zero-norm cosine→0.0, CI percentiles 2.5/97.5, RNG seed 42.
- **LLM model ids:** Sonnet (extractor/classifier), Haiku (meddra/populate/descriptions/population_features), temperature=0 everywhere (deterministic — STRUCT, not a tuned temperature).

## §6 — Perf / IO knobs (not belief-affecting)

Batch/timeout/retry/concurrency. Tune for speed/cost, never reported as a model hyperparameter.
- **Concurrency:** build_graph 5 (extract/classify/describe), fetch 4, LINCS Reactome 10, OT 8, PubMed 3 (10 w/ key), descriptions 5.
- **Timeouts:** 30s (CT.gov/OT/ChEMBL/PubMed/PubChem/RxNorm/QuickGO), 60s (LLM clients/HGNC/LINCS).
- **Retries/backoff:** extractor 5 (base 10s ×2^n), classifier/extractor JSON 3, CT.gov 4 (base 1s ×2).
- **Token caps:** extractor 4096 (16384 doc), classifier 4096, Haiku calls 10–800.
- **MC samples:** predict 10_000, eval 2_000.
- **Page sizes:** CT.gov ≤1000, OT assoc 500, PubMed `max_pmids` 3.
- **Caches:** `data/cache/*.json` (biolord/sapbert/chembl/reactome/hgnc/rxnorm/pubchem/meddra).
- **Field dedup:** anchor-vector index (store.py:389) — the 474MB→28MB scale fix.

---

## §7 — Tuning discipline (read before sweeping)

1. **NEVER tune on the headline holdout.** Nested CV: outer test fold selects nothing; inner CV (inside outer-train) picks knobs. The round-25/28 anti-pattern (`[[project_round28_sicko_mode]]`, `[[project_round25_evidence_records]]`) was hand-fitting on the 5-trial holdout — don't repeat it.

2. **Tune for calibration first, AUROC second.** The 0.566-acc < 0.71-base-rate gap is a pessimism/calibration artifact. Brier/ECE on edges is the bedrock; a Platt/isotonic recalibration + revisiting `_FAILURE_P_OBS`/softmin may lift both.

3. **The replay seam makes §2 cheap.** `EvidenceRecord`s are preserved on edges and `_replay_belief` reconstructs the Beta posterior from them (used by `store.get_edge_belief_conditioned` for off-tissue reweighting). So **n_eff tiers and p_obs buckets can be re-evaluated by replaying records with a parameterized weight map — no LLM re-attribution, no rebuild.** This is the efficiency unlock for putting the §2 grid in the inner loop. (Caveat from NEXT_SESSION: the merge-replay and attribution-replay formulas differ — verify replay fidelity before trusting it as ground truth; fall back to true re-attribution per fold if they diverge.)

4. **Already hand-fit → re-validate, don't trust:** `_SAFETY_PENALTY_CAP` (5-trial audit), `_COMPOUND_EMBEDDING_SIMILARITY_THRESHOLD` (n=11), `_hr_support_bucket`/`_ae_support_bucket` (self-described "first-pass, refit downstream of the calibration harness"), round-28 n_eff binding bumps (calibrated by "source character," not data).

5. **Two-stage sweep:** (a) inner-loop full grid over §1 + the replay-able parts of §2 on the frozen n=500 graph; (b) outer coarse grid over a handful of §3 knobs (biolord_threshold, _MECHANISM_PATHWAY_CAP, _HI/_LO, OT min_score), one rebuild each. Don't cross them.

6. **Inconsistencies to fix while here:** OT min_score 0.1 vs 0.3 (§3); box `margin` 0.05 vs 0.10 (§3); precision-multiplier flag bypassed in attributor (§4). Each is a latent un-tuned divergence.

7. **Record every change** via the `/tuning-log` skill → persistent memory, so a future session can audit whether a value was principled or overfit.

## §8 — Deferred calibration ideas

- **Evidence concentration-capping (owner idea, 2026-06-05; deferred until after edge-weight tuning).** Edges grow unbounded (observed Beta(1228, 447), 238 records) ⇒ near-zero variance ⇒ overconfident MC samples ⇒ the softmin can lock onto an overconfident *low* edge. Fix = when `α+β > C`, scale both by `C/(α+β)` — **preserves the mean `α/(α+β)` exactly, caps confidence**. (NOT a per-value sigmoid: squashing α,β individually distorts the mean and `(<1,<1)` is bimodal.) Targets *overconfidence*, complementary to the *mean-pessimism* fix (p_obs/prior). Same disease as `_REDUNDANCY_RHO` (OFF) cures at the source. A new predict-time or post-attribution knob `EROOM_EVIDENCE_CAP` (C ∈ ~20–200). Measure: does capping lift holdout Brier/ECE beyond the p_obs fix?
