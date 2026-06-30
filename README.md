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

**Structured failure decomposition.** A single trial can't reliably pinpoint *which* link broke—attributing failure to one edge from one trial is premature falsification. So the trial's **outcome conditions the whole chain**: on a failure, contradictory evidence is distributed across the chain's links by *explaining-away*—it flows to the under-evidenced links while curated molecular facts (a confirmed binding) self-protect; on a success, every link is reinforced. The **overlap across many trials** then triangulates the responsible edge—a link that recurs in failed chains and rarely in successful ones is the one the evidence implicates. The failure classifier's job is a coarse **operational-vs-mechanistic gate** (did the trial actually *test* the chain, or fail for recruitment / dosing / funding reasons?), not per-trial edge attribution. This keeps each trial's evidence honest and lets the responsible pathway emerge from the corpus rather than from a single noisy guess.

---

## Current evaluation

Eroom Bio is an active research prototype; everything below is reported under honest, leak-free protocols — k-fold holdout with **per-fold re-attribution** (each test trial's evidence is removed from the graph before it is scored), plus leave-target-out and leave-indication-out — never same-corpus leave-one-out (which flatters the model).

- **In-sample** (a trial's own evidence is in the graph — a correctness gate, *not* generalization): AUROC ≈ 0.77.
- **Honest holdout, broad multi-indication corpus (~500 trials across 5 diseases):** AUROC ≈ 0.53–0.57, roughly flat with corpus size — sparse cross-trial reuse means beliefs largely echo each trial's own evidence rather than transferring.
- **Honest holdout, concentrated oncology corpus (`onco_scale_500`):** AUROC ≈ 0.70, robust to leave-indication/leave-target-out — but the lift is **localized to targeted therapies** and absent in the cytotoxic backbone.
- **vs. baselines:** on a single binary-success number the graph currently **ties** strong non-mechanistic trial-design features (enrollment, single-arm, #compounds, rationale; AUROC ≈ 0.68–0.70). Incremental tests (likelihood-ratio + CV-stacked DeLong) show the graph adds *complementary* signal on top of design — localized to targeted therapies — rather than beating it outright.

**What ships today is the interpretable layer, not novel-entity prediction:** **structured failure decomposition** (which edge in the causal chain the corpus implicates for a given outcome) and **per-target safety attribution** (validated on EGFR→rash and INSR→hypoglycemia). Predicting risk for *unseen* compounds/targets remains at chance — gated by **reuse-per-edge** (how many trials share each link), the project's single binding constraint, not by the inference method. (Five classic case studies — nivolumab, solanezumab, bevacizumab, torcetrapib, selumetinib — remain wired as a direction sanity-check in `scripts/eval_holdout_compose.py`.)

**Ground-truth caveat:** the success/failure label is the **LLM-extracted `primary_endpoint_met`** (read from the trial's results text by the extraction model), with a classifier-`trial_outcome` fallback — *not* a structured ClinicalTrials.gov efficacy field. The labels therefore inherit any extraction error; treat the headline AUROC as conditioned on label fidelity, not an absolute.

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

Round 25 split these into per-source curated-database tiers — the shipped behavior. (An experimental refinement that further scaled N_eff by trial size with a redundancy discount was prototyped and removed; the fixed tiers above are what runs.)

### Prediction (weakest-link + informed prior, failure-causing safety)

`overall = efficacy × (1 − safety_penalty)`.

- **Efficacy — weakest-link, not geometric mean.** A causal chain is only as strong as its weakest *well-evidenced* link, so efficacy is a **soft-min** over per-edge Beta samples (`P(success) ≈ P(weakest link)`) — replacing the earlier trust-weighted geometric mean, which diluted a decisive weak link by the n-th root. Under-evidenced edges sample under a **weak informed prior** (mean ~0.75), so ignorance defers to a plausible base rate instead of producing low samples that spuriously become the minimum. The Bayesian posterior's concentration thus replaces the old evidence-count "trust weight" (sparse → near prior; abundant → near observed). These parameters are baked in `src/config.py` (`softmin_t`, `prior_mean`, `prior_strength`); the geometric-mean / product / hard-min variants were removed — soft-min is the sole aggregation path.
- **Safety — failure-causing toxicity, not occurrence.** `causes_ae` / `target_associated_ae` measure AE *incidence*, but an effective drug with tolerated toxicity (e.g. nivolumab's irAEs in trials that *succeeded*) must not be penalized like one whose toxicity was dose-limiting. Each AE's penalty is gated by the fraction of its evidence from **dose-limiting-toxicity failures** vs tolerated/successful trials — soft-or of `severity × belief × trust × failure-causing` (capped). A serious AE floors at the grade-3 weight when CTCAE grade is missing (the common case). The DLT gate is baked on (`src/config.py`).

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

### Running it — the entry points that matter

The repo has many scripts and flags; in practice you use a small set. This is that subset.

**Build the graph.** One driver — `scripts/build_graph.py` (fetch → populate → annotate → attribute). **Invoke it as a module**: `python scripts/build_graph.py` silently breaks `--assemble`.

```bash
python -m scripts.build_graph \
  --corpus multi_indication_52_train \   # frozen NCT-id list → reproducible build
  --max-trials 60 \
  --bottom-up \                          # chains-first build — already the default (flag kept for back-compat)
  --assemble \                           # materialize geometry + the (s,t) belief field
  --allow-partial-classify --allow-partial-subgraphs
```

The flags that actually matter (the rest are rarely needed):

| Flag | When you reach for it |
|---|---|
| `--bottom-up` | The production build — per-trial isolated subgraphs → a re-runnable merge. **It is already the default**; the flag is a no-op kept for back-compat. Pass `--top-down` to opt into the legacy shared-store path. |
| `--assemble` | Materialize the geometry (boxes, is-a) and the per-`(s,t)` belief field. Requires `python -m`. |
| `--corpus <name>` | Pin a frozen NCT-id list (written on first use) — reproducible builds. |
| `--reannotate NCT,NCT…` | **Iterate on prompts.** Deletes only those trials' caches, re-runs just them, preserves every other annotation — append-only. This is *not* a fresh build. |
| `--keep-annotations` | Reuse cached extract+classify (populate-only iteration). **Footgun:** without it, a fresh build WIPES `data/annotations/`. |
| `--base-snapshot … --add-trials …` | Extend an existing snapshot (attribution is idempotent). |

**Predict a clinical hypothesis.** `compound_id` may be `None` — pass `target_id` to predict a *novel compound on a familiar target* (the chain is walked from the target onward):

```python
from src.prediction.path_query import predict_clinical_hypothesis
r = predict_clinical_hypothesis(g, "nivolumab", "melanoma")                         # familiar compound
r = predict_clinical_hypothesis(g, None, "melanoma", target_id="ENSG00000188389")   # novel anti-PD-1
r.overall_probability   # efficacy × (1 − safety_penalty)
```

**Evaluate on holdouts — compose-and-scan** (`scripts/eval_holdout_compose.py`), the honest true-holdout: it predicts test trials *without building them into the graph*, anchors on the target, scores only where the chain lands, and returns honest "unknown" when the corpus has no knowledge to generalize from — rather than fabricating a number. Three modes:

```bash
python -m scripts.eval_holdout_compose --graph data/exports/<area>_annotated.json          # classic-5 direction (default)
python -m scripts.eval_holdout_compose --graph G --field <field.json>                       # + (s,t)-field-localized vs scalar
python -m scripts.eval_holdout_compose --graph G --auroc --corpus <name> [--field <f>]       # AUROC + accuracy over a corpus
```

For a TRUE out-of-sample AUROC (the corpus `--auroc` above is in-sample — each
trial's evidence is baked into the edges scoring it), use the K-fold holdout,
which re-attributes the graph with each fold excluded (exact; replay-based
masking is unfaithful under node-merge + AE-propagation):

```bash
python -m scripts.eval_holdout_kfold --initial data/exports/<area>_initial.json \
  --annotated data/exports/<area>_annotated.json --corpus <name> --k 5
```

**Reproduce the headline oncology numbers.** One tracked, seeded command — it runs the honest k-fold holdout *and* the paired-DeLong comparison against the design baseline, against the frozen frontier snapshots:

```bash
python -m scripts.reproduce_frontier        # → holdout AUROC ≈0.70, in-sample ≈0.77 (seed 42)
```

The full modeling config (routing, ontology keying, informed prior, soft-min, safety gate, n_eff tiers) is frozen in **`src/config.py`** — the single source of algorithm truth. There are no `EROOM_*` algorithm flags; only deployment paths (`EROOM_GRAPH_SNAPSHOT`, `EROOM_PRIVATE_ROOT`) and API keys are read from the environment.

**Inspect causal chains** (columns by node type, per-row before-merge vs converged after-merge) — also wired as the `/graph-inspect` skill:

```bash
python scripts/visualize_graph.py --area <area> --mode chain
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
tests/          # ~1,390 tests (non-integration)
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

- ~1,390 tests passing (non-integration) on Python 3.11+.
- **One baked algorithm config** (`src/config.py`) — reason-routed attribution, biology→GO-BP ontology keying, native direction, informed prior, safety DLT-gate, weakest-link soft-min. There are **no production feature flags**; `python -m scripts.reproduce_frontier` reproduces the headline oncology numbers from the frozen snapshots.
- **Chains-first build (on `main`).** Per-trial isolated subgraphs → a re-runnable merge; the abstraction ladder sits at true scale (Target → Mechanism = Reactome signaling pathway → Biology = general process; drug-class is metadata on the compound, operativity on the `modulates_via` edge); and **attribution is outcome-conditioning** — the trial outcome conditions the whole chain by explaining-away, the failure classifier is a coarse operational gate, and cross-trial overlap triangulates the failing edge.
- **Validated surfaces:** interpretable failure decomposition + per-target safety attribution. **Gated on denser data:** novel-entity risk and binary-success prediction — reuse-per-edge is the binding constraint.
- **Pending:** a PubMed safety ingester (off-target safety not posted to CT.gov); predictor calibration; the BioLORD node substrate.

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
