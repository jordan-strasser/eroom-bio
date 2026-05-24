# The public / private boundary

Eroom is **open core**. This repo (Apache 2.0) is the public core: the graph
schema, all algorithms, the ingestion + annotation pipeline, and snapshots
carrying the scalar `Beta(α, β)` edge beliefs with full provenance. The
*enterprise* layer — fine-tuned model weights, the per-region belief field, and
the outcome-conditioned manifold learner — lives in a **separate private repo**
(`eroom-enterprise`) and is never published here.

This document is the operational contract. The strategy behind it is in
`future_ideas/manifold_learning.md` (the three-manifold model).

## What is public vs private

| Layer | Public (this repo) | Private (`eroom-enterprise` + artifact store) |
|---|---|---|
| **Code / schema** | All of it — algorithms, embedding code, box-embedding schema, merge logic | Manifold-2 field math, manifold-3 learner |
| **Manifold 1** (concepts + hierarchy) | Embedding *code*; node descriptions | Fine-tuned BioLORD weights, trained box params |
| **Manifold 2** (edge beliefs) | Scalar `Beta(α, β)` marginal + evidence | Full per-region belief field |
| **Manifold 3** (outcome-conditioned) | — (referenced as the paid product) | Entire learner + its snapshots |
| **Data** | Trial annotations/outcomes (from public registries), gold-pair eval set | Proprietary non-registry data fed back by partners |

Rule of thumb (from `manifold_learning.md`): **trace the primary refinement
signal.** Trial *outcomes* train manifold 3 and nothing else → private.
Anything observational (ontologies, descriptions, evidence) → public-eligible.

## Why this is enforced in code, not `.gitignore`

`data/exports/` is **tracked**, and `GraphStore.export_snapshot` serializes
*every field on every node and edge*. The moment a node gains a fine-tuned
`embedding` or an edge gains a `belief_field`, that value would flow straight
into a committed public JSON — and a tracked file's history is permanent.
`.gitignore` does nothing for a tracked path. So the boundary is three layers
of code (`src/boundary.py`):

1. **Default-safe export.** `export_snapshot` runs `strip_private()` before
   writing — the public artifact is clean by construction even when the
   in-memory graph holds private values (it does, during a combined build).
   `export_private_snapshot` writes the full payload, and **refuses any path
   outside `EROOM_PRIVATE_ROOT`**, which itself **refuses to resolve inside
   this repo**.
2. **Fail-loud audit.** `scripts/check_public_snapshots.py` re-scans on-disk
   snapshots and exits non-zero on any private value. Run it in CI / pre-commit.
3. **Naming convention.** A field is private if its name is in
   `PRIVATE_FIELD_NAMES` or ends with `_embedding`, `_field`, `_box`,
   `_anchors`, `_weights`, … . New private fields inherit protection for free
   by following the convention. Field *names/schemas* are public; only
   populated *values* are stripped (a declared-but-`None` field is always safe).

## Doing manifold work without leaking

- **Add private fields to public models** (e.g. `embedding`, `belief_field`)
  as optional, default `None`, named per the convention. The schema being
  public is fine; the boundary strips the values on public export.
- **Write the heavy private logic in `eroom-enterprise`**, not gitignored files
  inside this repo. (This corrects the earlier kickoff-doc plan that placed
  `belief_field.py` gitignored under `src/inference/` — gitignore is not the
  boundary.) The private package imports `eroom` as a dependency.
- **Embeddings are a cache artifact, not graph state.** They live in
  `data/cache/` (gitignored) and are rehydrated on load
  (`populate._rehydrate_compound_embedding`), so the public snapshot never
  needs them. Base-model vectors are recomputable by anyone; fine-tuned vectors
  are the moat — the snapshot strips *both*, so the rule is simply "no embedding
  vectors in public snapshots."
- **Private artifacts** (weights, box params, belief-field snapshots) go under
  `EROOM_PRIVATE_ROOT` (default `~/.eroom/private`) or the `eroom-enterprise`
  repo — never `data/exports/`.

## If `check_public_snapshots.py` reports a leak

A committed snapshot carries private values. Re-export it through
`export_snapshot` (which strips), or delete it if it was an experiment
artifact. As of this writing several round-30/52 snapshots carry base-SapBERT
`embedding` vectors from before the boundary existed — harmless (public model)
but bloat, and they must be scrubbed before any fine-tuned vector is produced.
