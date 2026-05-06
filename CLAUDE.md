# Eroom Bio

## What this is
Eroom Bio is an open knowledge system that learns from clinical trials. It decomposes trial outcomes into mechanistic causal chain updates, maintains Bayesian beliefs on a knowledge graph, and produces interpretable, compositional predictions for therapeutic success.

Named after Eroom's Law (Moore's Law backwards) — the 80x decline in drug approvals per $B of R&D since 1950. Unlike chip fabrication, biotech has never had a centralized system that accumulates mechanistic knowledge from every trial and builds on it over time. This is that system.

## Architecture (5 layers, built in order)
1. **Knowledge Graph** (`src/graph/`) — Typed nodes (Compound, Target, Mechanism, Biology, Biomarker, Population, Endpoint, Indication) and typed directed edges (binds_to, modulates_via, mechanism_affects, biology_drives, reflects_biology, endpoint_captures, responds_differently). Every edge carries a Beta distribution belief state with evidence provenance.
2. **Ingestion** (`src/ingestion/`) — Adapters for ClinicalTrials.gov, Open Targets, DrugBank. Maps external data to graph nodes and edges.
3. **Annotation** (`src/annotation/`) — AI-powered pipeline that extracts structured trial data, classifies failure modes (13-category mechanistic taxonomy), and attributes failures to specific edge updates. Uses Anthropic API (Claude Sonnet).
4. **Inference** (`src/inference/`) — Bayesian updating of edge beliefs with evidence-type weighting (clinical > genetic > preclinical > in vitro). Damped belief propagation across graph neighbors.
5. **Prediction** (`src/prediction/`) — Compositional path queries: P(success) is the **trust-weighted geometric mean** of per-edge Beta samples along the causal chain (not a raw product), so unobserved Beta(1,1) edges don't drag the prediction down. Monte Carlo sampling for uncertainty. Identifies bottleneck edges (weighted by trust, so unknown ≠ weak). See `onboarding/how_success_is_predicted.md` for the full breakdown.

## Key design decisions
- NetworkX (MultiDiGraph) for graph storage, not Neo4j — keeps everything in-process for now
- Pydantic v2 for all models — strict validation everywhere
- Beta distributions for edge beliefs — conjugate prior, cheap updates, explicit uncertainty
- Evidence-weighted updates: Phase 3 trial = 5.0x, genetic MR = 4.0x, GWAS = 2.5x, in vivo = 1.5x, in vitro = 1.0x
- Failure taxonomy has 13 mechanistic categories (not operational categories like "lack of efficacy")
- Each trial is represented as a subgraph (a path through the causal chain), not a flat feature vector

## Repo structure
```
eroom/
  src/
    graph/           # Schema, models, store, population
    ingestion/       # ClinicalTrials.gov, Open Targets, DrugBank
    annotation/      # Extractor, classifier, attributor, taxonomy, prompts/
    inference/       # Beliefs, updater, propagation
    prediction/      # Path queries, sampler, explainer
    api/             # FastAPI service
  data/
    schemas/         # JSON Schema exports
    annotations/     # Per-trial annotation JSONs
    exports/         # Graph snapshots
  tests/
```

## Testing
- pytest for all tests
- Integration tests that hit external APIs marked with @pytest.mark.integration
- Run: `pytest tests/` or `pytest tests/test_models.py` for specific module

## Style
- Python 3.11+
- Type hints on everything
- Async where I/O is involved (API calls, batch processing)
- Models in Pydantic, not dataclasses
- Keep functions focused — one function, one job
