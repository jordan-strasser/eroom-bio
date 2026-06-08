# Eroom Bio

## What this is
Eroom Bio is an open knowledge system that learns from clinical trials. It decomposes trial outcomes into mechanistic causal-chain updates, maintains Bayesian beliefs on a knowledge graph, and produces interpretable, compositional predictions for therapeutic success.

Named after Eroom's Law (Moore's Law backwards) — the ~80x decline in drug approvals per $B of R&D since 1950. Unlike chip fabrication, biotech has never had a centralized system that accumulates mechanistic knowledge from every trial and builds on it over time. This is an attempt at one.

## Architecture (4 layers, built in order)
1. **Knowledge Graph** (`src/graph/`) — Typed nodes (Compound, Target, Mechanism, Biology, Biomarker, Population, Endpoint, Indication, AdverseEvent, Trial) and typed directed edges (`binds_to`, `modulates_via`, `mechanism_affects`, `biology_drives`, `reflects_biology`, `endpoint_captures`, `responds_differently`, `causes_ae`, `target_associated_ae`, `composed_of`). Every edge carries a **Beta-distribution belief** with evidence provenance.
2. **Ingestion** (`src/ingestion/`) — Adapters for ClinicalTrials.gov, Open Targets, ChEMBL, and LINCS L1000. Maps external data to graph nodes and edges.
3. **Annotation** (`src/annotation/`) — An LLM pipeline that extracts structured trial data, classifies failure modes against a mechanistic taxonomy, normalizes adverse-event terms, and attributes evidence to specific edge updates. Beta-Binomial conjugate updates live in `src/inference/beliefs.py`.
4. **Prediction** (`src/prediction/`) — Compositional path queries: P(success) is composed from the per-edge Beta beliefs along a causal chain, with Monte-Carlo sampling for uncertainty and identification of the bottleneck (lowest-belief) edges.

## Key design decisions
- **NetworkX** (MultiDiGraph) for graph storage — keeps everything in-process.
- **Pydantic v2** for all models — strict validation everywhere.
- **Beta distributions** for edge beliefs — conjugate prior, cheap updates, explicit uncertainty. Updates are **evidence-weighted**: stronger evidence types (e.g. a curated molecular binding fact, or a Phase 3 readout) move a belief more than weak ones (e.g. a single preclinical assay).
- **A trial is a subgraph**, not a flat feature vector — an arm × subgroup × endpoint fan-out of causal chains. This is what lets a mechanism learned in one indication inform another.
- **Failure taxonomy is mechanistic** (why the biology/chain failed), not operational ("lack of efficacy").
- **Bottom-up build**: each trial is resolved into trial-scoped nodes, then merged into the shared graph — so canonicalization and cross-trial pooling apply uniformly, and trials can be appended incrementally.
- **Frozen corpora**: a file at `data/corpora/<name>.txt` pins the exact NCT-id list for a snapshot, making rebuilds reproducible.

## Repo structure
```
eroom/
  src/
    graph/        # schema, models, store, population pipeline
    ingestion/    # ClinicalTrials.gov, Open Targets, ChEMBL, LINCS adapters
    annotation/   # extractor, classifier, attributor, taxonomy, prompts/
    inference/    # Beta-Binomial belief updates, AE propagation
    prediction/   # path queries, sampler, explainer
    api/          # FastAPI service
  data/
    corpora/      # frozen NCT-id lists for reproducible builds
    annotations/  # per-trial extraction + classification JSON (LLM outputs)
    exports/      # graph snapshots: <name>_initial.json, <name>_annotated.json
  scripts/        # build_graph, eval, visualize
  tests/          # pytest; integration tests hit live APIs
```

## Building the graph
`scripts/build_graph.py` is the end-to-end entry point: **fetch → populate → annotate → attribute**. Common flags:
- `--corpus <name>` — build against the frozen NCT-id list at `data/corpora/<name>.txt` (or write one on first run).
- `--area <name>` — output prefix for the snapshots written to `data/exports/`.
- `--max-trials N` — cap the number of trials.
- `--keep-annotations` — reuse cached LLM extractions/classifications across rebuilds (so a re-populate doesn't re-pay for extraction).

```bash
python -m scripts.build_graph --corpus my_corpus --area my_corpus --max-trials 50 --keep-annotations
```

Visualize a built graph: `python -m scripts.visualize_graph --area <name> --mode layered` (a left→right Compound→…→Population causal-chain view) or `--mode graph` (a force-directed overview).

## Testing
- `pytest tests/` runs the suite; `pytest -m "not integration"` skips tests that hit external APIs.
- Integration tests are marked `@pytest.mark.integration`.

## Style
- Python 3.11+, type hints on everything.
- Async where I/O is involved (API calls, batch processing).
- Models in Pydantic, not dataclasses.
- Keep functions focused — one function, one job.

## Contributing
This repo is the open core. See `CONTRIBUTING.md`. Architecture-level changes (new node/edge types, new prediction math) should land on a branch with the test suite green and a graph rebuild before merge, so regressions are visible.
