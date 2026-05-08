# Eroom Bio

## What this is
Eroom Bio is an open knowledge system that learns from clinical trials. It decomposes trial outcomes into mechanistic causal-chain updates, maintains Bayesian beliefs on a knowledge graph, and produces interpretable, compositional predictions for therapeutic success.

Named after Eroom's Law (Moore's Law backwards)—the 80x decline in drug approvals per $B of R&D since 1950. Unlike chip fabrication, biotech has never had a centralized system that accumulates mechanistic knowledge from every trial and builds on it over time. This is that system.

## Architecture (4 layers, built in order)
1. **Knowledge Graph** (`src/graph/`)—Typed nodes (Compound, Target, Mechanism, Biology, Biomarker, Population, Endpoint, Indication, AdverseEvent, Trial) and typed directed edges (binds_to, modulates_via, mechanism_affects, biology_drives, reflects_biology, endpoint_captures, responds_differently, causes_ae, target_associated_ae, composed_of). Every edge carries a Beta distribution belief state with evidence provenance.
2. **Ingestion** (`src/ingestion/`)—Adapters for ClinicalTrials.gov, Open Targets, LINCS L1000. Maps external data to graph nodes and edges.
3. **Annotation** (`src/annotation/`)—AI-powered pipeline that extracts structured trial data (Sonnet), classifies failure modes (13-category mechanistic taxonomy), normalizes adverse-event terms (Haiku), and attributes evidence to specific edge updates. Beta-Binomial conjugate updates live alongside in `src/inference/beliefs.py`.
4. **Prediction** (`src/prediction/`)—Compositional path queries: P(success) is the **trust-weighted geometric mean** of per-edge Beta samples along the causal chain (not a raw product), so unobserved Beta(1,1) edges don't drag the prediction down. Monte Carlo sampling for uncertainty. Identifies bottleneck edges (weighted by trust, so unknown ≠ weak).

## v0.1.0 architecture lock
The prediction math, trust-weight function (log-scaled, saturation at evidence_strength=49), edge priors, aggregation method, and edge topology are **frozen at v0.1.0** until the corpus expands beyond melanoma. Architectural PRs touching those should wait for the cross-indication scaling phase. Data, adapter, and prompt PRs are unblocked.

## Key design decisions
- NetworkX (MultiDiGraph) for graph storage, not Neo4j—keeps everything in-process for now
- Pydantic v2 for all models—strict validation everywhere
- Beta distributions for edge beliefs—conjugate prior, cheap updates, explicit uncertainty
- Evidence-weighted updates (`EVIDENCE_TYPE_N_EFF` in `src/inference/beliefs.py`): Phase 3 = 15.0, Genetic MR = 10.0, Phase 2 = 6.0, GWAS = 4.0, Phase 1 = 2.0, Preclinical in vivo = 2.0, Preclinical in vitro = 1.0, Computational = 0.3, Literature = 0.2
- Failure taxonomy has 13 mechanistic categories (not operational categories like "lack of efficacy")
- Each trial is represented as a subgraph (an arm × subgroup × endpoint fan-out of causal chains), not a flat feature vector
- Frozen-corpus mechanism: a corpus file at `data/corpora/<name>.txt` pins the exact NCT-id list used to build a snapshot, making rebuilds reproducible across runs

## Repo structure
```
eroom/
  src/
    graph/           # Schema, models, store, population pipeline
    ingestion/       # ClinicalTrials.gov, Open Targets, LINCS adapters
    annotation/      # Extractor, classifier, attributor, taxonomy, MedDRA, prompts/
    inference/       # Beta-Binomial belief updates, AE propagation
    prediction/      # Path queries, sampler, explainer
    api/             # FastAPI service
  data/
    corpora/         # Frozen NCT-id lists for reproducible builds
    annotations/     # Per-trial extraction + classification JSONs
    exports/         # Graph snapshots (oncology_initial.json, oncology_annotated.json)
  scripts/           # build_graph, analyze_run, eval_predictions
  tests/             # 428 tests; integration tests hit live APIs
  docs/              # Architecture spec
```

## Pipeline driver
`scripts/build_graph.py` is the single end-to-end entry point: fetch → populate → annotate → attribute. Pass `--corpus <name>` to build against a frozen NCT-id list (or write one on first invocation). Pass `--keep-annotations` to preserve cached extract+classify outputs across rebuilds.

## Testing
- pytest for all tests
- Integration tests that hit external APIs marked with `@pytest.mark.integration`
- Run: `pytest tests/` or `pytest tests/test_populate.py` for a specific module

## Style
- Python 3.11+
- Type hints on everything
- Async where I/O is involved (API calls, batch processing)
- Models in Pydantic, not dataclasses
- Keep functions focused—one function, one job
