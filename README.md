# Eroom Bio

[![CI](https://github.com/jordan-strasser/eroom-bio/actions/workflows/ci.yml/badge.svg)](https://github.com/jordan-strasser/eroom-bio/actions/workflows/ci.yml)

**A meta-learning system for medicine.** It learns from clinical trials the way chip fabs learn from production runs — by accumulating mechanistic knowledge across every trial instead of treating each as a one-off.

Named after [Eroom's Law](https://www.nature.com/articles/nrd3681) (Moore's Law backwards): the cost to develop a drug has roughly doubled every nine years since 1950. Unlike chip fabrication, biotech has never had a shared system that compounds what each trial teaches. This is an attempt at one.

## The idea

Every trial tests a causal hypothesis along a chain of typed nodes:

```
Compound → Target → Mechanism → Biology → Endpoint → Indication → Population
```

> *"This drug, hitting this target, through this mechanism, changes this biology, captured by this endpoint, in this indication and patient population."*

Each edge carries its own **Beta-distributed belief**, so a trial can validate one link while contradicting another. Evidence accumulates on the **shared** links across trials that test the same targets, mechanisms, and biology — so a mechanism learned in one indication can inform another. Instead of only recording pass/fail, Eroom Bio records *where in the chain* a trial succeeded or failed, and composes predictions from the per-edge beliefs (`overall = efficacy × (1 − safety_penalty)`), flagging the bottleneck link.

## Current state

Eroom Bio is an active research prototype. Results are reported under leak-free protocols (k-fold holdout with per-fold re-attribution; leave-target / leave-indication-out) — never same-corpus leave-one-out.

- **What works today:** interpretable **failure decomposition** (which link the evidence implicates) and **per-target safety attribution** (e.g. EGFR→rash, insulin→hypoglycemia).
- **Predictive accuracy:** honest out-of-sample AUROC is ≈0.53–0.57 on a broad multi-indication corpus and ≈0.70 on a concentrated oncology one. The ceiling is **cross-trial reuse** — how many trials share each mechanistic link — so the active focus is building a denser, higher-reuse corpus to lift it. The interpretable decomposition is the near-term differentiator, not a single accuracy number.
- The success/failure label is LLM-extracted from each trial's results text, so treat headline numbers as conditioned on label fidelity.

The full modeling algorithm is frozen in one file: **`src/config.py`**.

## How to use it

```bash
git clone https://github.com/jordan-strasser/eroom-bio.git
cd eroom-bio
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...        # required for the extract + classify steps
```

**Build a graph** from a frozen, reproducible NCT-id corpus (fetch → extract → populate → annotate → attribute):

```bash
python -m scripts.build_graph --corpus <name> --area <name> --max-trials 50
```

**Predict** a clinical hypothesis (`compound` may be `None` to ask about a novel compound on a known target):

```python
from src.prediction.path_query import predict_clinical_hypothesis
r = predict_clinical_hypothesis(graph, "nivolumab", "melanoma")
r.overall_probability        # efficacy × (1 − safety_penalty), with the bottleneck edge flagged
```

**Evaluate** honestly — k-fold holdout that re-attributes the graph with each fold excluded:

```bash
python -m scripts.eval_holdout_kfold \
  --initial data/exports/<name>_initial.json \
  --annotated data/exports/<name>_annotated.json --corpus <name> --k 5
```

**Reproduce** the headline oncology numbers in one seeded command, or **inspect** causal chains visually:

```bash
python -m scripts.reproduce_frontier
python scripts/visualize_graph.py --area <name> --mode chain
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding trials and data sources, and `docs/` for architecture detail.

## Project layout

```
src/graph/        knowledge-graph schema, store, build pipeline
src/ingestion/    ClinicalTrials.gov, Open Targets, ChEMBL, LINCS adapters
src/annotation/   LLM extraction, failure classification, attribution
src/inference/    Beta-Binomial belief updates
src/prediction/   compositional path queries + safety
src/api/          FastAPI service
```

## License

[Apache 2.0](LICENSE).
