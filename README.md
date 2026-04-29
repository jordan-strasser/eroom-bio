# Eroom Bio

Eroom Bio is an open knowledge system that accumulates structured mechanistic understanding from clinical trials. It decomposes trial outcomes into causal chain updates, maintains Bayesian beliefs on a knowledge graph, and produces interpretable, compositional predictions.

Named after Eroom's Law (Moore's Law backwards) — the 80x decline in drug approvals per billion dollars of R&D since 1950.

## Architecture

1. **Knowledge Graph** (`src/graph/`) — Typed nodes and directed edges with Beta distribution belief states
2. **Ingestion** (`src/ingestion/`) — Adapters for ClinicalTrials.gov, Open Targets, DrugBank
3. **Annotation** (`src/annotation/`) — AI-powered trial failure classification and causal attribution
4. **Inference** (`src/inference/`) — Bayesian updating with evidence-type weighting and belief propagation
5. **Prediction** (`src/prediction/`) — Compositional path queries with Monte Carlo uncertainty estimation

## Setup

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/
```
