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

## Current evaluation (round 30)

Five well-known case studies anchor direction-prediction: nivolumab CheckMate-067 (success), solanezumab EXPEDITION (failure), bevacizumab AVANT (failure, adjuvant), torcetrapib ILLUMINATE (failure, off-target safety), selumetinib thyroid (success). They're scored two ways: **in-sample** (the trial's own evidence is attributed into the graph — an algorithmic-correctness gate; it *should* be called right) and **true holdout** (excluded from attribution, predicted only from other trials — the generalization test).

- **In-sample: 5/5** direction-correct — the algorithm reproduces the outcomes of trials whose own evidence is in the graph.
- **True holdout: 3/5**, including both successes (nivolumab, selumetinib). Three mechanisms make this work: a decisive weak link can veto the chain (bevacizumab's `responds_differently` for the adjuvant population), an informed prior keeps ignorance from sinking the chain, and tolerated toxicity no longer sinks an effective drug (nivolumab's irAEs).
- Across **54 labeled trials**: binary accuracy 0.69, AUROC 0.65 — lifting true successes without lifting failures.

**The two holdout misses are data gaps, not predictor flaws:**
- **torcetrapib** — its off-target cardiac safety isn't in CT.gov (no posted results); a PubMed ingester is the fix.
- **bevacizumab** — `responds_differently` is population-only, so adjuvant-CRC chemo successes dilute the anti-VEGF-adjuvant failure on the shared edge; the structural fix is mechanism-conditioned population sub-regions.

**Honest scope:** these 5 case studies + 54 labeled trials are a *directional* signal, not statistical validation — in-sample is self-consistency, the holdout is the real (small) generalization test; the predictor's knobs are pre-calibration.

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

Round 25 split these into per-source curated-database tiers; **round 30** makes N_eff *precision-grounded* — the value above is the anchor for a median-N trial, scaled by the trial's actual patient count, with an independence/redundancy discount so correlated evidence (same study/sponsor) doesn't compound as if independent (flag-gated via `EROOM_NEFF_PRECISION`).

### Prediction (round 30: weakest-link + informed prior, failure-causing safety)

`overall = efficacy × (1 − safety_penalty)`.

- **Efficacy — weakest-link, not geometric mean.** A causal chain is only as strong as its weakest *well-evidenced* link, so efficacy is a **soft-min** over per-edge Beta samples (`P(success) ≈ P(weakest link)`) — replacing the earlier trust-weighted geometric mean, which diluted a decisive weak link by the n-th root. Under-evidenced edges sample under a **weak informed prior** (mean ~0.75), so ignorance defers to a plausible base rate instead of producing low samples that spuriously become the minimum. The Bayesian posterior's concentration thus replaces the old evidence-count "trust weight" (sparse → near prior; abundant → near observed). Env-tunable (`EROOM_SOFTMIN_T`, `EROOM_PRIOR_MEAN`/`EROOM_PRIOR_STRENGTH`); `EROOM_AGG=geomean` restores the legacy mean.
- **Safety — failure-causing toxicity, not occurrence.** `causes_ae` / `target_associated_ae` measure AE *incidence*, but an effective drug with tolerated toxicity (e.g. nivolumab's irAEs in trials that *succeeded*) must not be penalized like one whose toxicity was dose-limiting. Each AE's penalty is gated by the fraction of its evidence from **dose-limiting-toxicity failures** vs tolerated/successful trials — soft-or of `severity × belief × trust × failure-causing` (capped). A serious AE floors at the grade-3 weight when CTCAE grade is missing (the common case). `EROOM_SAFETY_DLT_GATE=0` restores occurrence-only.

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

- 1065 tests passing (non-integration) on Python 3.12
- **Round 30** (branch `neff-precision-v1`): the predictor was rebuilt — weakest-link soft-min + informed prior efficacy, failure-causing-gated safety, and precision-grounded N_eff (trial sample size + an independence/redundancy discount) — all flag-gated, validated, then made default. Results under *Current evaluation* above.
- Pending: a `modulates_via` demonstrated-vs-assumed classifier rule; the PubMed ingester (torcetrapib safety + N_eff p-values); the BioLORD node substrate (semantic redundancy + mechanism-conditioned populations); LOO calibration of the predictor knobs.

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
