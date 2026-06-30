# Contributing to Eroom Bio

The most valuable contributions are **trial annotations** that extend the
knowledge graph into new indications, mechanisms, and trial designs. Every
trial run through the pipeline produces a structured annotation that
accumulates evidence on the shared causal-chain edges—the more diverse
the corpus, the more cross-trial signal the system can surface.

This guide covers running the pipeline on new trials, submitting
annotations, and the conventions we keep stable across PRs.

---

## Quick start

```bash
git clone https://github.com/jordan-strasser/eroom-bio.git
cd eroom
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=your_key       # required for extract + classify
export CLUE_API_KEY=your_key            # optional, for LINCS signatures
```

Sanity-check the install:

```bash
pytest tests/ -q                        # the non-integration suite should pass (~1,390 tests)
```

Build the existing melanoma graph from the frozen corpus:

```bash
python -m scripts.build_graph \
    --condition melanoma --max-trials 200 \
    --keep-annotations --corpus melanoma_145
```

(The `melanoma_145` corpus file at `data/corpora/melanoma_145.txt` pins the
exact 145 NCT ids used for the v0.1.0 baseline. Cached annotations under
`data/annotations/` mean this rebuild is fast and free of API spend.)

---

## Running the pipeline on new trials

The four-phase pipeline—**fetch → populate → annotate → attribute** —
lives in `scripts/build_graph.py`. The simplest way to add new trials is
to define a frozen corpus.

### Option 1: extend an existing indication

1. Pick or create a corpus file under `data/corpora/`. For example to add
   trials to the melanoma graph, append NCT ids to a new corpus
   `melanoma_extended.txt`:

   ```
   # Melanoma corpus extended with adjuvant Phase III trials
   NCT01844505
   NCT02388906
   ...
   ```

2. Run the pipeline against the new corpus. The `--corpus` flag tells the
   script to fetch trials by NCT id from the file (skipping the
   ClinicalTrials.gov search query—this is what makes builds
   reproducible across runs):

   ```bash
   python -m scripts.build_graph \
       --condition melanoma --max-trials 500 \
       --keep-annotations --corpus melanoma_extended
   ```

3. Inspect what landed in the graph:

   ```bash
   python -m scripts.analyze_run \
       --graph data/exports/oncology_annotated.json \
       --annotations data/annotations
   ```

### Option 2: a brand-new indication

The same flow, but with a fresh condition filter at first fetch (the
search query path runs when the corpus file does NOT exist yet—the
script writes the resulting NCT list back to that path so future runs are
reproducible):

```bash
python -m scripts.build_graph \
    --condition nsclc --max-trials 200 \
    --corpus nsclc_v1            # writes data/corpora/nsclc_v1.txt on first run
```

After the first run, every subsequent invocation with `--corpus nsclc_v1`
loads the same NCT ids deterministically.

### What gets generated

Each trial produces two files under `data/annotations/`:

- `<NCT>_extraction.json`—structured extraction (compound, target, arms,
  endpoints, results, AEs, subgroups). Cached on disk; deterministic re-runs
  hit cache and skip the Anthropic API.
- `<NCT>_classification.json`—failure-mode classification + edge updates
  (one per causal-chain edge the trial provides evidence for). Cached the
  same way.

If you want to **force a re-classify** after editing prompts in
`src/annotation/prompts/`, delete the relevant `_classification.json` files
(or all of them for a wholesale refresh) and re-run with
`--keep-annotations` so cached extractions are preserved.

---

## Submitting annotations

Annotations are checked in. To contribute:

1. **Run the pipeline locally** on the trials you want to add (instructions
   above). Confirm the build is clean—no `Skipped NCT*: no trial_subgraph
   in sidecar` lines, or if there are, document the reasons.

2. **Inspect what changed.** Run `analyze_run.py` against the new snapshot
   and capture the cross-trial accumulation diff (which edges gained
   contributing trials, which conflict edges surfaced).

3. **Open a PR with**:
   - The new corpus file under `data/corpora/`
   - The new `*_extraction.json` and `*_classification.json` files under
     `data/annotations/`
   - An updated graph snapshot at `data/exports/oncology_annotated.json`
     (or a new snapshot path if you're scoping to a separate indication)
   - A short description of what trials were added, what edges accumulated
     evidence, and any new conflicts surfaced

4. **PRs that change the prompts** in `src/annotation/prompts/` should also
   delete and regenerate every cached classification, since the cached
   files reflect the old prompt's reasoning.

PRs that only add a corpus file (without committing the regenerated
annotations) are also welcome—annotations are reproducible from the
corpus + an Anthropic API key, just slower for reviewers to verify.

---

## Other ways to contribute

### Failure taxonomy extensions

The 13 mechanistic failure modes live in `src/annotation/taxonomy.py` —
each one declares which edge types it weakens / strengthens / leaves
neutral. New categories should be added with explicit edge-update rules
and at least one reference trial that exemplifies the mode.

### New data source adapters

Adapters for ClinicalTrials.gov, Open Targets, and LINCS live in
`src/ingestion/`. New sources should:

- Map to existing graph nodes where possible (CompoundNode, TargetNode,
  IndicationNode, etc.)
- Emit `EvidenceRecord` objects with a clearly-classified `EvidenceType`
  so the inference layer's N_eff weighting handles them correctly
- Preserve provenance (`source_id`, `provenance_url`) so every belief
  update remains auditable

Add the new evidence type to `EVIDENCE_TYPE_N_EFF` in
`src/inference/beliefs.py` with a defensible relative scaling.

### Entity resolution improvements

Compound resolution already layers a curated codename dict
(`CODENAME_TO_INN`), ChEMBL `stable_id` matching, and SapBERT embedding
similarity. The remaining gaps are trade names, indication aliases, and
monoclonal-antibody target resolution; better alias tables or HUGO/MeSH
lookups would move the corpus-coverage ceiling above its current ~80% on
noisy indications.

---

## Conventions we keep stable

- **Python 3.11+**, type hints on every public function
- **Pydantic v2** for all models (no dataclasses)
- **Async** for any I/O—Anthropic, OT, CT.gov
- **NetworkX** MultiDiGraph for graph storage (in-process, not Neo4j)
- **Beta(α, β) beliefs** with full evidence-record provenance on every edge
- **Tests live in `tests/`**; integration tests that hit external APIs are
  marked `@pytest.mark.integration`

Stability matters here: the prediction algorithm is **baked into one frozen
config**, `src/config.py` — reason-routing, biology→GO ontology keying, native
direction, the informed prior, the safety DLT-gate, weakest-link **softmin**
aggregation, and the n_eff evidence tiers. There are no production feature flags;
changing any of those modeling choices is an architectural change. Architectural
PRs that touch them should wait for the cross-indication scaling phase to land
first; data and adapter PRs are unblocked.

---

## Reporting issues

- Bugs in the pipeline: GitHub issue with a minimal reproducer (a
  corpus file + the failing build log)
- Questions about a specific trial's classification: open an issue
  referencing the NCT id and what you expected vs. what the system
  produced—disagreements with classifier output are how the prompts
  get sharper
- Architectural proposals: open a discussion before opening a PR; the
  baked baseline in `src/config.py` is intentionally locked
