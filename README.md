# Eroom Bio

**A meta-learning system for medicine.**

Eroom Bio decomposes clinical trials into mechanistic causal chains, accumulates evidence across trials on a shared knowledge graph, and surfaces patterns that no single trial can reveal—which mechanisms translate, which endpoints capture clinical benefit, and where the real scientific disagreements are.

Named after [Eroom's Law](https://www.nature.com/articles/nrd3681): the cost of developing a new drug has doubled every nine years since 1950. Unlike chip fabrication—where every production run updates a collective yield-learning model—biotech has never had a centralized system that accumulates mechanistic knowledge from every trial and builds on it over time. This is that system.

---

## What it does

Every clinical trial tests a causal hypothesis:

```
Compound → Target → Mechanism → Biology → Endpoint
```

*"This drug, binding this target, through this mechanism, will change this biology, detectable by this endpoint, in this patient population, for this disease."*

Most systems record whether trials pass or fail. Eroom Bio records **where in the chain** they passed or failed, and accumulates that evidence across trials that share the same targets, mechanisms, and biology.

### Three things this enables

**Cross-trial knowledge accumulation.** When multiple independent trials test the same mechanistic link, evidence accumulates on the shared edge. After processing 145 melanoma trials:

- **PD-1 → checkpoint blockade** (`modulates_via`): 7 independent trials, all concordant support, posterior belief 0.50 → 0.88. The system learned that PD-1 inhibition reliably produces checkpoint blockade—from data, not from a hard-coded rule.
- **Nivolumab → anemia** (`causes_ae`): 9 trials contributing to the same adverse event edge. Cross-trial AE signal that no single trial could establish.
- 77 edges with 3+ contributing trials. 15 edges with 5+ trials.

**Conflict detection.** When trials disagree about the same mechanistic link, the system flags it:

- **ORR → clinical benefit in cutaneous melanoma** (`endpoint_captures`): 2 trials, both contradicting. ORR doesn't reliably translate to clinical benefit in this subtype—a finding with direct implications for endpoint selection in future trials.
- **Checkpoint blockade → melanoma biology** (`mechanism_affects`): 8 trials, 7 supporting, 1 contradicting (NCT02752074). A real scientific outlier worth investigating.
- 7 disagreement edges total at n=145, with 6 representing genuine scientific conflict.

**Structured failure decomposition.** When a trial fails, the system classifies *where* in the causal chain it broke—using a 13-category mechanistic taxonomy:

| Failure mode | What it means | Which edges update |
|---|---|---|
| No target engagement | Drug didn't bind/modulate target | Weakens `binds_to`, `modulates_via` |
| Target engaged, biology not moved | Hit target, pathway didn't respond | Weakens `mechanism_affects` |
| Biology moved, endpoint flat | Biomarker changed, clinical outcome didn't | Weakens `endpoint_captures` |
| Efficacy in subgroup only | Worked in subgroup, diluted in full population | Strengthens subgroup edges |
| Dose-limiting toxicity | Can't reach therapeutic dose | Compound-specific |
| Wrong population | Mechanism valid, patients lack driving biology | Strengthens mechanism, weakens population |

A single failed trial produces *opposing* updates on different edges—strengthening the links that worked, weakening the one that broke.

---

## Architecture

### The knowledge graph

Typed nodes (Compound, Target, Mechanism, Biology, Indication, Endpoint, Population, Biomarker, AdverseEvent) connected by typed directed edges. Every edge carries a **Beta(α, β) belief state**—a probability distribution representing both the expected probability and the strength of evidence.

- `Beta(1, 1)` = no evidence, total ignorance
- `Beta(18, 3)` = strong evidence supporting this link (probability ~0.86)
- `Beta(4, 11)` = evidence mostly against this link (probability ~0.27)
- `Beta(15, 12)` = substantial evidence, genuinely conflicted (probability ~0.56, high conflict score)

Every update records its source trial, evidence type, and support bucket—full provenance on every belief.

### Evidence weighting

Updates use a principled Beta-Binomial conjugate model. Each evidence record contributes N_eff virtual observations based on evidence type:

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

The support direction (how much the evidence supports or contradicts the edge) is classified into 7 buckets by the LLM using a rubric grounded in observable features—not subjective confidence scores.

### Data sources

| Source | What it populates |
|---|---|
| ClinicalTrials.gov | Trial records, compounds, indications, endpoints |
| Open Targets | Target-disease associations (`biology_drives` priors) |
| LINCS L1000 | Perturbation signatures (`mechanism_affects` evidence) |
| Anthropic Claude (Sonnet) | Structured extraction, failure classification, mechanism inference |

### Prediction

Given a therapeutic hypothesis (compound, target, mechanism, biology, indication, endpoint, population), the system computes P(success) as a weighted geometric mean of edge beliefs along the causal chain. Each edge's contribution is weighted by a trust score reflecting evidence strength. The prediction identifies the **weakest link**—the edge most likely to cause failure—and explains why.

Current OOS discrimination (n=145 melanoma, temporal split): modest. The prediction layer improves with data density and will be the focus of scaling work. The knowledge accumulation, conflict detection, and failure decomposition are production-quality now.

---

## Quickstart

```bash
git clone https://github.com/eroom-bio/eroom.git
cd eroom
pip install -e .
```

Set environment variables:
```bash
export ANTHROPIC_API_KEY=your_key
export CLUE_API_KEY=your_key  # optional, for LINCS data
```

Build a melanoma knowledge graph:
```bash
python scripts/build_graph.py --indication melanoma --max-trials 50
```

Analyze the graph:
```bash
python scripts/analyze_run.py --snapshot data/exports/melanoma_n50.json
```

Run the API:
```bash
uvicorn src.api.main:app --host 0.0.0.0
```

Query a prediction:
```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"compound": "pembrolizumab", "target": "PDCD1", "mechanism": "checkpoint_blockade", "indication": "melanoma", "endpoint": "OS"}'
```

---

## Project structure

```
src/
  graph/        # Knowledge graph schema, store, population pipeline
  ingestion/    # ClinicalTrials.gov, Open Targets, LINCS adapters
  annotation/   # AI-powered extraction, classification, attribution
  prediction/   # Compositional path queries
  api/          # FastAPI service (6 endpoints)
data/
  corpora/      # Frozen trial lists for reproducible builds
  annotations/  # Per-trial structured annotations
  exports/      # Graph snapshots
tests/          # 261 tests
scripts/        # Build and analysis tools
docs/           # Architecture spec
```

---

## What's novel

Existing tools predict trial outcomes as black-box classifiers. Eroom Bio produces:

1. **Edge-level Bayesian beliefs** with evidence provenance—not a single probability, but a decomposed view of where the evidence is strong and where it's weak.
2. **A mechanistic failure taxonomy**—13 categories that map failure modes to specific edges in the causal chain. This taxonomy does not exist elsewhere.
3. **Cross-trial knowledge accumulation**—shared edges that compound evidence from independent trials. 77 edges with 3+ contributing trials after 145 melanoma trials.
4. **Scientific conflict detection**—edges where trials disagree, flagged automatically. These are the interesting questions, not the settled ones.

---

## Current status

- 145 melanoma trials processed with full causal chain decomposition
- 1,950 evidence-carrying edges, 190 with multi-trial evidence
- 7 conflict edges surfaced (including ORR translatability in cutaneous melanoma)
- 100% trial coverage with frozen corpus
- Architecture locked. Scaling to additional indications and data sources is the next phase.

---

## Contributing

The most valuable contributions are:

- **Trial annotations**: Run the pipeline on trials in your therapeutic area and submit the structured annotations.
- **Failure taxonomy extensions**: Propose new failure mode categories with edge update rules.
- **Data source adapters**: Connect new evidence sources (FDA review documents, ChEMBL, DepMap).
- **Entity resolution improvements**: Better mapping of free-text drug/target/endpoint names to canonical IDs.

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
