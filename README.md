# Eroom Bio

**A meta-learning system for medicine.**

Eroom Bio decomposes clinical trials into mechanistic causal chains, accumulates evidence across trials on a shared knowledge graph, and surfaces patterns that no single trial can reveal—which mechanisms translate, which endpoints capture clinical benefit, and where the real scientific disagreements are.

Named after [Eroom's Law](https://www.nature.com/articles/nrd3681): the cost of developing a new drug has doubled every nine years since 1950. Unlike chip fabrication—where every production run updates a collective yield-learning model—biotech has never had a centralized system that accumulates mechanistic knowledge from every trial and builds on it over time. This is that system.

---

## What it does

Every clinical trial tests a causal hypothesis across seven canonical node types — **Compound, Target, Mechanism, Biology, Endpoint, Indication, Population** — connected by typed directed edges:

```
Compound → Target → Mechanism → Biology → Endpoint → Indication → Population
```

*"This drug, binding this target, through this mechanism, will change this biology, captured by this endpoint, treating this indication in this patient population."*

Each edge carries an independent Beta-distributed belief — so a trial can validate one part of the hypothesis while contradicting another.

Most systems record whether trials pass or fail. Eroom Bio records **where in the chain** they passed or failed, and accumulates that evidence across trials that share the same targets, mechanisms, and biology. P(success) is decomposed into a mechanism-only **efficacy** term and a severity-weighted **safety** drag, integrated as `overall = efficacy × (1 − safety_penalty)`.

### Three things this enables

**Cross-trial knowledge accumulation.** When multiple independent trials test the same mechanistic link, evidence accumulates on the shared edge. Edge beliefs become structured features rather than per-trial point estimates.

**Conflict detection.** When trials disagree about the same mechanistic link — same compound, same target, contradicting evidence — the system flags it. These are the interesting questions, not the settled ones.

**Structured failure decomposition.** When a trial fails, the system classifies *where* in the causal chain it broke—using a 13-category mechanistic taxonomy:

| Failure mode | What it means | Which edges update |
|---|---|---|
| No target engagement | Drug didn't bind/modulate target | Weakens `affects`, `modulates_via` |
| Target engaged, biology not moved | Hit target, pathway didn't respond | Weakens `mechanism_affects` |
| Biology moved, endpoint flat | Biomarker changed, clinical outcome didn't | Weakens `endpoint_captures` |
| Efficacy in subgroup only | Worked in subgroup, diluted in full population | Strengthens subgroup `responds_differently` |
| Dose-limiting toxicity | Can't reach therapeutic dose | Compound-specific `causes_ae` |
| Wrong population | Mechanism valid, patients lack driving biology | Strengthens mechanism, weakens population |

A single failed trial produces *opposing* updates on different edges—strengthening the links that worked, weakening the one that broke.

---

## Current evaluation (2026-05-20)

The most-evaluated experiment in the project to date — a 52-trial multi-indication training corpus + 5 well-known case studies held out for direction-prediction.

### Setup

- **Training:** 52 NCT ids across 5 indications (melanoma, Alzheimer's, colorectal, atherosclerosis/hypercholesterolemia, thyroid). 12 dropped as non-therapeutic (behavioral, device, diagnostic, procedure studies). 40 training subgraphs produced.
- **Holdout:** 5 case studies, added via the round 19 incremental-build mode:
  - **nivolumab CheckMate-067** (success, melanoma)
  - **solanezumab EXPEDITION** (failure, Alzheimer's — chain works, biology→outcome breaks)
  - **bevacizumab AVANT** (failure, colorectal — wrong context: works in metastatic, fails in adjuvant)
  - **torcetrapib ILLUMINATE** (failure, cardiovascular — off-target hypertension, not mechanism)
  - **selumetinib thyroid Ho 2013** (success, niche — works in BRAF-mutant subset)
- **Final snapshot:** 777 nodes, 1,459 edges, 91% chain coverage on the 178 chains.

### Result: 4 of 5 direction-correct

| Case | Lit outcome | Efficacy | Safety | **P(success)** | Direction |
|---|---|---:|---:|---:|---|
| nivolumab | success | 0.856 | 0.067 | **0.799** | ✓ |
| solanezumab | failure | 0.481 | 0.019 | **0.472** | ✓ <0.5 |
| bevacizumab AVANT | failure | 0.482 | 0.038 | **0.464** | ✓ <0.5 |
| torcetrapib | failure (safety-driven) | 0.652 | 0.000 | **0.652** | ✗ |
| selumetinib thyroid | success | 0.723 | 0.000 | **0.723** | ✓ |

### Edge decompositions correctly reflect literature failure modes

The most informative case is **bevacizumab AVANT**. Literature says it fails in the adjuvant setting (not metastatic). The system's chain decomposition shows:

```
[UP ↑] affects                bevacizumab → VEGFA          E[p]=0.52  n_eff=19
[UP ↑] modulates_via          VEGFA → angiogenesis_inhib   E[p]=0.55  n_eff=20
[UP ↑] mechanism_affects      angiogenesis → biology       E[p]=0.50  n_eff=18
[DN ↓] biology_drives         angiogenesis → colorectal    E[p]=0.40  n_eff=21
[DN ↓] responds_differently   adjuvant_stage_iii → CRC     E[p]=0.29  n_eff=10  ← weakest
```

The system correctly identifies `responds_differently` for the adjuvant-stage-III population as the weakest link — exactly the failure mode AVANT demonstrated. Cross-trial learning surfaced the population/context bottleneck without being told to look there.

### Where it misses, and why

**Torcetrapib (the one miss).** The chain decomposition correctly shows torcetrapib's mechanism works (CETP inhibition raised HDL — literature-true). The failure was off-target cardiovascular safety. With zero torcetrapib-specific or CETP-class adverse-event evidence in the 52-trial corpus, `safety_penalty=0.000`. The safety architecture is in place (`causes_ae` and `target_associated_ae` edges both feed the penalty); the data is missing. This is a corpus coverage gap, queued for a deeper round 24 audit.

### Trust calibration

This is the most-evaluated state of the system, but it's **5 case studies, not a statistical sample**. Older claims about specific n=145 melanoma OOS AUROCs and "production-quality" cross-trial learning predate the v0.3.0 prediction-math rework and the round 19-22 architecture and shouldn't be over-interpreted. The honest scope:

- The pipeline mechanically works end-to-end on a fresh multi-indication corpus
- Cross-trial learning produces directional signal on 4 of 5 well-known case studies
- The one miss is a known data-coverage gap, not an architectural failure
- The result needs the [round 24 audit](audit/round_24_holdout_eval_audit_questions.md) (leakage check, edge provenance, monoclonal antibody resolver, torcetrapib safety propagation diagnostic) before it's load-bearing for any larger claim

---

## Architecture

### The knowledge graph

Typed nodes (Compound, Target, Mechanism, Biology, Indication, Endpoint, Population, Biomarker, AdverseEvent) connected by typed directed edges. Every edge carries a **Beta(α, β) belief state**—a probability distribution representing both the expected probability and the strength of evidence.

- `Beta(1, 1)` = no evidence, total ignorance
- `Beta(18, 3)` = strong evidence supporting this link (probability ~0.86)
- `Beta(4, 11)` = evidence mostly against this link (probability ~0.27)
- `Beta(15, 12)` = substantial evidence, genuinely conflicted (probability ~0.56, high conflict score)

Every update records its source trial, evidence type, and support bucket—full provenance on every belief.

### Compound canonicalization

Multiple CT.gov spellings of the same drug (`"5-Fluorouracil"` vs `"Fluorouracil"`, `"Imatinib"` vs `"Imatinib mesylate"`) used to fragment evidence across duplicate nodes. The current populator resolves an incoming compound via:

1. **`CODENAME_TO_INN`** curated dict (deterministic, handles development codes like AZD6244 → selumetinib)
2. **ChEMBL `stable_id` match** against existing compound nodes (the authoritative signal — same ChEMBL id = same drug)
3. **SapBERT embedding cosine ≥ 0.80** with chembl-id non-conflict gate (catches spelling + salt-form variants that fall through the first two layers)

Threshold validated on a 30-pair labeled set: max similar-class-distinct pair (paclitaxel/docetaxel) scored 0.72, well below the 0.80 cutoff. No false-merge candidates at threshold.

### Evidence weighting

Updates use a Beta-Binomial conjugate model. Each evidence record contributes N_eff virtual observations based on evidence type:

| Evidence type | N_eff | Rationale |
|---|---|---|
| Phase III RCT | 15.0 | Definitive clinical evidence |
| Genetic (Mendelian randomization) | 10.0 | Causal direction established |
| Phase II | 6.0 | Clinical signal, not definitive |
| GWAS | 4.0 | Association, not causation |
| Phase I | 2.0 | Safety/PK, limited efficacy data |
| Preclinical in vivo | 2.0 | Animal model, variable translation |
| Preclinical in vitro | 1.0 | Cell line data |
| Computational | 0.3 | Predicted, not measured |

### Prediction (round 20: efficacy + safety)

P(success) decomposes into two components:

- **`efficacy_probability`** — trust-weighted geometric mean of chain edge beliefs (the mechanism-only view)
- **`safety_penalty`** — soft-or aggregation over compound-specific (`causes_ae`) and on-mechanism (`target_associated_ae`) AE evidence, each AE contributing `severity_weight × belief_factor × trust_factor` (capped at 0.6)
- **`overall_probability = efficacy_probability × (1 − safety_penalty)`**

Severity weighting comes from the AE node's max observed CTCAE grade: Grade 1-2 = 0.05, Grade 3 = 0.15, Grade 4 = 0.30, Grade 5 = 0.50. Three-gate modulation (severity × belief × evidence) prevents AMBIGUOUS-bucket AEs from saturating the cap on well-evidenced drugs.

### Build modes (round 19)

- **Fresh build:** `--corpus <name>` rebuilds from scratch under a frozen NCT-id list.
- **Incremental:** `--base-snapshot path --add-trials NCT1,NCT2` extends an existing annotated snapshot. The `applied_attribution_trial_ids` set provides idempotency — re-running attribution on existing trials is a no-op.

### Data sources

| Source | What it populates |
|---|---|
| ClinicalTrials.gov | Trial records, compounds, indications, endpoints |
| Open Targets | Target-disease associations (`biology_drives` priors) |
| ChEMBL | Drug metadata (stable_id, diagnostic filter, target inference for chemo) |
| LINCS L1000 | Perturbation signatures (`mechanism_affects` evidence) |
| Reactome + QuickGO | Pathway / biology canonicalization |
| HGNC | Gene-symbol alias resolution |
| MedDRA (via Haiku) | Adverse-event term normalization |
| HuggingFace SapBERT | Compound name embeddings for canonicalization |
| Anthropic Claude (Sonnet) | Structured trial extraction, failure classification, mechanism inference |

---

## Quickstart

```bash
git clone https://github.com/eroom-bio/eroom.git
cd eroom
pip install -e ".[dev,sapbert]"
```

Set environment variables:
```bash
export ANTHROPIC_API_KEY=your_key
export CLUE_API_KEY=your_key  # optional, for LINCS data
```

Reproduce the n=52 multi-indication build:
```bash
python -m scripts.build_graph \
  --corpus multi_indication_52_train \
  --max-trials 52 \
  --area multi_indication_52 \
  --include-terminated \
  --allow-partial-subgraphs
```

Add the 5 holdout case studies incrementally:
```bash
python -m scripts.build_graph \
  --base-snapshot data/exports/multi_indication_52_annotated.json \
  --add-trials NCT01844505,NCT01127633,NCT00112918,NCT00134264,NCT00970359 \
  --area multi_indication_52 \
  --allow-partial-subgraphs
```

Run the case-study audit:
```bash
python -m scripts.case_study_audit \
  --graph data/exports/multi_indication_52_annotated.json
```

Verify the SapBERT canonicalization layer:
```bash
python -m scripts.verify_sapbert
```

Run the API:
```bash
uvicorn src.api.main:app --host 0.0.0.0
```

---

## Project structure

```
src/
  graph/        # Knowledge graph schema, store, population pipeline,
                # SapBERT canonicalization, codename dict
  ingestion/    # ClinicalTrials.gov, Open Targets, ChEMBL, LINCS adapters
  annotation/   # AI-powered extraction, classification, attribution
  inference/    # Belief updates + AE propagation
  prediction/   # Compositional path queries + safety integration
  api/          # FastAPI service
data/
  corpora/      # Frozen NCT-id lists for reproducible builds
  annotations/  # Per-trial structured annotations (cached)
  cache/        # API caches (OT, ChEMBL, SapBERT, etc.)
  exports/      # Graph snapshots
  dev/          # Per-build observability logs (dropped trials,
                # unrouted updates, unmapped subgroup features)
audit/          # Per-round audit notes + case-study reports (gitignored)
tests/          # 852 tests
scripts/        # Build + analysis tools
docs/           # Architecture spec
```

---

## What's novel

Existing tools predict trial outcomes as black-box classifiers. Eroom Bio produces:

1. **Edge-level Bayesian beliefs** with full evidence provenance — every update records source NCT, evidence type, and support bucket
2. **A mechanistic failure taxonomy** — 13 categories that map failure modes to specific edges in the causal chain
3. **Cross-trial knowledge accumulation** — shared edges that compound evidence from independent trials
4. **Decomposed P(success)** — efficacy (mechanism chain) × (1 − safety drag), with the safety drag computed from `causes_ae` and `target_associated_ae` evidence at the chain's target
5. **Scientific conflict detection** — edges where trials disagree, flagged automatically

---

## Current status

- 852 tests passing on Python 3.12
- 18 atomic rounds shipped to main (rounds 14-22)
- Latest build: n=52 multi-indication + 5-trial holdout (2026-05-20). Result: 4/5 direction-correct, edge decompositions match literature failure modes on 4 of 5, one safety-driven miss attributed to corpus data gap. Round 24 audit queued to verify this isn't memorization, expose trial provenance on edges, fix monoclonal antibody target resolution, and diagnose the torcetrapib `target_associated_ae` propagation.

---

## Contributing

The most valuable contributions are:

- **Trial annotations**: Run the pipeline on trials in your therapeutic area and submit the structured annotations.
- **Failure taxonomy extensions**: Propose new failure mode categories with edge update rules.
- **Data source adapters**: Connect new evidence sources (RxNorm, PubChem, DrugBank, FDA review documents).
- **Entity resolution improvements**: Better mapping of free-text drug/target/endpoint names to canonical IDs — especially monoclonal antibodies.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

Apache 2.0

---

## Citation

If you use Eroom Bio in your research, please cite:

```
@software{eroom_bio,
  title={Eroom Bio: A Meta-Learning System for Medicine},
  url={https://github.com/eroom-bio/eroom},
  year={2026}
}
```

---

*The chip-fab industry built infrastructure for compounding knowledge. Drug development never did. Eroom Bio is that infrastructure.*
