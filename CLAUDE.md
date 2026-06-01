# Eroom Bio

## What this is
Eroom Bio is an open knowledge system that learns from clinical trials. It decomposes trial outcomes into mechanistic causal-chain updates, maintains Bayesian beliefs on a knowledge graph, and produces interpretable, compositional predictions for therapeutic success.

Named after Eroom's Law (Moore's Law backwards)—the 80x decline in drug approvals per $B of R&D since 1950. Unlike chip fabrication, biotech has never had a centralized system that accumulates mechanistic knowledge from every trial and builds on it over time. This is that system.

## Architecture (4 layers, built in order)
1. **Knowledge Graph** (`src/graph/`)—Typed nodes (Compound, Target, Mechanism, Biology, Biomarker, Population, Endpoint, Indication, AdverseEvent, Trial) and typed directed edges (binds_to, modulates_via, mechanism_affects, biology_drives, reflects_biology, endpoint_captures, responds_differently, causes_ae, target_associated_ae, composed_of). Every edge carries a Beta distribution belief state with evidence provenance.
2. **Ingestion** (`src/ingestion/`)—Adapters for ClinicalTrials.gov, Open Targets, LINCS L1000. Maps external data to graph nodes and edges.
3. **Annotation** (`src/annotation/`)—AI-powered pipeline that extracts structured trial data (Sonnet), classifies failure modes (13-category mechanistic taxonomy), normalizes adverse-event terms (Haiku), and attributes evidence to specific edge updates. Beta-Binomial conjugate updates live alongside in `src/inference/beliefs.py`.
4. **Prediction** (`src/prediction/`)—Compositional path queries: P(success) defaults to a **weakest-link softmin** over per-edge Beta samples along the causal chain (round-30; temperature `_SOFTMIN_T≈0.10`), with zero-evidence Beta(1,1) edges dropped upstream so unobserved edges don't drag the prediction down. The legacy **trust-weighted geometric mean** is opt-in via `EROOM_AGG=geomean` (also `product`/`min`/`harmonic`). Monte Carlo sampling for uncertainty. Identifies bottleneck edges (weighted by trust, so unknown ≠ weak).

## v0.2.0 — architecture lock lifted (round 28)
The v0.1.0 architecture lock was lifted in round 28. The cross-indication corpus (n=30 per-indication holdout build, multiple indications) is large enough that the previously-frozen subsystems need to evolve together. Changes that landed in v0.2.0:

- **Prediction math:** trust-weighted geomean at v0.2.0; **round-30 (later) switched the default to weakest-link softmin** (`EROOM_AGG`; geomean now opt-in) — a sparse-but-decisive weak edge should set the chain probability rather than be averaged away. Mechanism direction (`activating` / `inhibiting` / `modulating`) is now surfaced as metadata on `MechanismNode` so audits can distinguish "edge operativity" from "patient-benefit direction" without changing the underlying math.
- **Edge priors:** sourced exclusively via `EvidenceRecord` per round 25 (no hand-set populator priors). Round 28 raised `DATABASE_OT_DIRECT` / `DATABASE_CHEMBL` / `DATABASE_MAB_TABLE` n_eff values to reflect that molecular binding is a curated fact, not a probabilistic claim — so AMBIGUOUS trial classifications no longer dilute the AFFECTS-edge molecular truth.
- **Aggregation method:** unchanged *at v0.2.0* (round-30 later made weakest-link softmin the default — see Prediction math above). `target_associated_ae` propagation now rolls up to MedDRA SOC tier so sibling-compound safety signals aggregate even when individual extractions land at disjoint PT-level terms.
- **Edge topology:** unchanged.
- **Population canonicalization:** parent population_id uses coarse axes `{line, stage, extent}` by default; rare axes (histology, prior_tx, biomarker, age_group) demote to subgroup-only. Cross-trial `responds_differently` evidence can now share a population node across e.g. NSABP C-08 + bev AVANT.

What's locked at v0.2.0: nothing further — the system is in an "evolve as the corpus scales" mode now. Architectural PRs should still come with rebuild + holdout audit results so regressions are visible.

## Key design decisions
- NetworkX (MultiDiGraph) for graph storage, not Neo4j—keeps everything in-process for now
- Pydantic v2 for all models—strict validation everywhere
- Beta distributions for edge beliefs—conjugate prior, cheap updates, explicit uncertainty
- Evidence-weighted updates (`EVIDENCE_TYPE_N_EFF` in `src/inference/beliefs.py`): Phase 3 = 15.0, DB-OT-direct = 12.0, DB-ChEMBL/mAb-table = 10.0, Genetic MR = 10.0, Phase 2 = 6.0, GWAS = 4.0, Phase 1 = 2.0, Preclinical in vivo = 2.0, DB-OT-association/endpoint-prior = 2.0, DB-Reactome-GO = 1.5, Preclinical in vitro = 1.0, DB-LINCS = 1.0, DB-indication-taxonomy = 1.0, DB-fallback = 0.5, DB-cross-reference = 0.3, Computational = 0.3, Literature = 0.2. Per-source DB tiers added in round 25 — populator no longer hand-sets non-Beta(1,1) priors; every curated-DB fact is an EvidenceRecord with provenance. Round 28 raised the binding tier (OT-direct / ChEMBL / mAb-table) to reflect that molecular binding is a cross-checked fact, not a noisy clinical signal — so AMBIGUOUS trial classifications no longer dominate the AFFECTS-edge molecular truth.
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
