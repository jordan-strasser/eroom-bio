# Merge plan — harmonizing the outstanding branches (READ-ONLY proposal)

**Status: proposal for human review. Nothing here has been executed.** No merge,
rebase, cherry-pick, commit, or branch edit was performed. This document inspects
the git DAG + tree and recommends a sequence. The merge itself is a separate,
approved step.

---

## TL;DR (read this first — the premise needs one correction)

1. **The branches are NOT four divergent builds. They are one linear stack.**
   `git merge-base --is-ancestor` proves
   `fix/endpoint-deorphan-punchlist ⊂ fix/st-field-faithfulness ⊂ arch/triangulation-edge-weights`.
   `arch/triangulation-edge-weights` (local `c094f96`) already transitively contains
   every other build. **`main` is an ancestor of `arch` → merging `arch` into `main`
   is a fast-forward with ZERO file conflicts.** "Recent takes precedence" is already
   baked into commit order; there is nothing to arbitrate at the code level.

2. **So the real work is not conflict-resolution — it is hygiene + boundary.** Two
   problems dominate:
   - **~2 GB of belief-state blobs** were swept into the two most recent commits
     (`2eddfcc`, `c094f96`) by an `add -A` (e.g. `multi_500_premerge.json` 526 MB,
     `onco_scale_500_enr_premerge.json` 544 MB). Fast-forwarding `main` to `arch`
     would put all of it in `main`'s permanent history.
   - **The public/private boundary as built does not match the rule in this task.**
     The repo treats the materialized **belief-state snapshots as PUBLIC**
     (`BOUNDARY.md:5`, `src/boundary.py:40-41`); the task rule says the belief-state
     **is the private $500/query product.** This is the "boundary built for a past
     architecture" flag, and it is a business-model decision for the human.

3. **All code is public under both rules.** The 14 changed `src/` files are
   algorithm/schema only; none is belief-state code. Nothing needs to move
   public→private *in code*. The moves are all in `data/`.

---

## Step 1 — Branch inventory

### Branches with commits not in `main` (everything else is already an ancestor of `main`)

| branch | tip | commits ahead of main | newest commit | in the stack? |
|---|---|---:|---|---|
| **arch/triangulation-edge-weights** (local) | `c094f96` | **35** | 2026-06-12 | **stack HEAD (contains the other two)** |
| fix/st-field-faithfulness | `44d16de` | 31 | 2026-06-11 | yes — depth 31 of the stack |
| fix/endpoint-deorphan-punchlist | `5b1ad14` | 12 | 2026-06-09 | yes — depth 12 of the stack |
| archive/round-4-sub-chains | `a00519b` | 1 | (round 4, old) | **no — abandoned dead-end, exclude** |

`redesign/node-orthogonality` and `bench/tuning-and-enrichment` are **already merged
into `main`** (branch tip = merge-base with main; 0 commits ahead) — not outstanding.
All the `round-*` / `arch-*` legacy branches are likewise ancestors of `main`.

### The single linear stack (the four build arcs, oldest→newest)

```
main 240264d (2026-06-08)
 │
 ├─[A] 178db2f…5b1ad14  (+12, →06-09)  = fix/endpoint-deorphan-punchlist TIP
 │      edge-completeness de-orphan + prune, multi_500 (n=500) data, 2021–26 holdout
 │      corpus+extractions, "predict the STATED chain" fixes, calibration_bands tool
 │
 ├─[B] db4736f…44d16de  (+19 more =31)  = fix/st-field-faithfulness TIP
 │      faithful PUBLIC (s,t) belief field; merge-tier refactor (T4a/T4b/T5,
 │      mechanism = Reactome pathway fan-out); hierarchical backoff pooling;
 │      Phase A/B/C measure-first; edge-attribution knobs; architecture-diagnosis docs
 │
 ├─[C] 9f9048f,1773c0f  (+2 =33)  = origin/arch/triangulation-edge-weights TIP (pushed)
 │      triangulation: outcome-driven edge-weight inference + capstone findings
 │
 └─[D] 2eddfcc,c094f96  (+2 =35)  = arch LOCAL HEAD (NOT pushed to origin/arch)
        2eddfcc feat(inference): A3/A4 reason-routed EM (EROOM_ROUTING) + B1 ontology (EROOM_BIO_ONTOLOGY)
        c094f96 feat(safety):    domain-manifold AE borrowing/decomposition (EROOM_SAFETY_MANIFOLD)
```

**Recency order (the merge order, since it is a stack): A → B → C → D.** Newest is
arc **D** (the local `arch` commits, 06-11/06-12). Three feature flags ship across the
stack, all default-OFF: `EROOM_ROUTING`, `EROOM_BIO_ONTOLOGY`, `EROOM_SAFETY_MANIFOLD`.

> Note on arc D: it is **two un-pushed local commits** that an `add -A` bundled
> together — `2eddfcc` mixes **two** independent flagged features (A34 routing + B1
> ontology) plus B1 export blobs; `c094f96` mixes the safety feature with ~2 GB of
> unrelated premerge blobs, `data/dev/` notes, and other arcs' exports. They are
> functionally fine but **not cleanly scoped** (see Step 5 hygiene).

### Per-build one-liner intent

- **A — endpoint-deorphan:** make every trial's chain edge-complete (de-orphan
  endpoints/biomarkers, prune ghosts); land the n=500 corpus + a real 2021–26 holdout.
- **B — st-field-faithfulness:** make the (s,t) belief field faithful + public, and
  refactor node-merge so mechanism = curated Reactome fan-out (the reuse substrate).
- **C — triangulation:** infer edge weights from outcomes (collapse the biology–
  endpoint–indication triangle), honestly evaluated.
- **D — inference+substrate+safety:** reason-routed/censored belief updates (A34),
  biology→GO-BP ids (B1), and domain-manifold safety borrowing + on/off-target
  decomposition. The three current flags.

---

## Step 2 — Conflict map

**There are no cross-branch file conflicts.** Because the branches are a linear
stack (not parallel forks), a merge of `arch → main` fast-forwards; git has nothing
to reconcile. "Recent takes precedence" is automatic — the latest commit to touch a
file is its state on `arch`.

The honest analogue of a conflict map for a stack is **which files evolved across
multiple build arcs** (so a reviewer knows the final state reflects the newest arc,
and an older arc's intermediate version is superseded):

| file | # commits in stack | arcs involved | final state owner |
|---|---:|---|---|
| `src/graph/populate_bottomup.py` | 7 | A, B | newest wins (B) |
| `src/graph/populate.py` | 4 | A, B, D | newest wins (D — B1 flag branch) |
| `scripts/holdout_thesis_analysis.py` | 4 | A, B | newest wins |
| `src/graph/models.py` | 3 | A, B | newest wins |
| `src/prediction/path_query.py` | 2 | B | — |
| `src/inference/beliefs.py` | 2 | B | — |
| `src/annotation/attributor.py` | 2 | B, D | newest wins (D — A34 routing) |
| `scripts/build_graph.py` | 2 | A, B | newest wins |
| `src/boundary.py` | 1 | B | **edited by arc B** (`db4736f` "PUBLIC (s,t) field") |

None of these is a true conflict; each is already resolved by commit order. The one
worth a reviewer's eye is **`src/boundary.py`**, which arc B edited to declare the
(s,t) field public — relevant to Step 3.

---

## Step 3 — Current public/private boundary (and where it no longer fits)

### How the split is enforced today (code, not `.gitignore`)

The boundary is **field-level value-stripping at serialization time**, not a
directory or repo split. Quoted mechanism:

- `src/boundary.py:81-116` — a **naming convention**: any serialized field whose name
  ends in `_embedding / _field / _box / _anchors / _weights …` (`PRIVATE_FIELD_SUFFIXES`,
  `:81`) or is in `PRIVATE_FIELD_NAMES` (`:92`) is a **private value**, with an explicit
  public override `PUBLIC_FIELD_NAMES = {"belief_field"}` (`:116`).
- `src/boundary.py:159-176` `strip_private()` — deep-copies a payload removing private
  keys; run by `GraphStore.export_snapshot` (`src/graph/store.py:365`) so the public
  artifact is clean by construction.
- `src/boundary.py:200-216` `assert_public_safe()` — fail-loud audit; CI entry point
  `scripts/check_public_snapshots.py`.
- `src/boundary.py:233-256` `private_root()` — out-of-tree `EROOM_PRIVATE_ROOT` that
  **refuses to resolve inside the repo**; guards `export_private_snapshot`
  (`store.py:389`, `require_under_private_root` `boundary.py:259`).

### What is public vs private **today** (per `BOUNDARY.md`)

| layer | today's classification | source |
|---|---|---|
| all code / schema / algorithms / harnesses | **PUBLIC** | `BOUNDARY.md:17` |
| scalar `Beta(α,β)` edge beliefs + provenance, **in committed snapshots** | **PUBLIC** | `BOUNDARY.md:5,19`; `boundary.py:40` |
| manifold-2 (s,t) belief **field** anchors + values | **PUBLIC** (the "field-public move") | `boundary.py:41,47-50`; arc B `db4736f` |
| fine-tuned embeddings, trained boxes (manifold-1 values) | private | `boundary.py:39` |
| manifold-3 outcome-conditioned learner + its snapshots | private (separate `eroom-enterprise` repo) | `boundary.py:42`; `BOUNDARY.md:7-8` |
| trial annotations/outcomes from public registries | PUBLIC-eligible | `BOUNDARY.md:21,25` |
| proprietary partner (non-registry) data | private | `BOUNDARY.md:21` |

### Where it no longer matches the task's rule — **the key flag**

The task's rule:
> Public = the algorithm/code. **Private = the corpus-derived belief-STATE (the
> trained/materialized graph), the hosted endpoint, customer instances, the raw
> pooled corpus.** The $500/query product is access to the private endpoint +
> belief-state, not a secret algorithm.

The repo's boundary was built on the **opposite** monetization thesis —
`boundary.py:45-50`: *"The open core is meant to be a top-shelf predictor, so the
manifold-2 belief field ships in the PUBLIC snapshot… The enterprise moat is private
DATA integration (pharma deals), NOT the prediction math."* So:

- **MISMATCH 1 (the big one): the materialized belief-state is PUBLIC today, PRIVATE
  under the task rule.** The committed `data/exports/*_initial.json` /
  `*_annotated.json` / the (s,t) field *are* the corpus-derived belief-state — the
  thing the task says is the paid product. The boundary mechanism has **no concept of
  "withhold the whole snapshot"**; it only scrubs private *fields within* a snapshot
  it assumes is publishable. Under the task rule, these snapshots should not be in the
  public repo at all. **Decision for the human: which thesis governs?**
- **MISMATCH 2: the (s,t) field was just made *more* public** (arc B, `db4736f`,
  `PUBLIC_FIELD_NAMES={"belief_field"}`). That is correct under the repo's thesis and
  **wrong under the task rule** (the field is part of the belief-state). If the task
  rule wins, this override should be reverted and the field treated as private state.
- **No reverse mismatch:** no algorithm code is sitting private. All `src/` is public,
  which both rules agree on. The safety-manifold + synth code is in `scratch/`
  (exploratory), not in private — also fine.

---

## Step 4 — Classify each build's changes (per the task rule)

| change (by arc) | files | task-rule class | note |
|---|---|---|---|
| edge-completeness, de-orphan, merge refactors, hierarchical backoff, triangulation math, A34 routing/censoring, B1 ontology id, safety borrow/decompose **logic** | 14 `src/` + 26 `scripts/` + 9 `tests/` | **PUBLIC** | algorithm/harness — ships to public `main`. Matches both rules. |
| safety-manifold implementation | `scratch/safety_manifold/*.py` | **PUBLIC** (but exploratory) | currently in `scratch/`, **not productized into `src/`**. Decide: promote to `src/` or keep as research code. Not a boundary risk. |
| synth EM harness | `scratch/synth/*` | PUBLIC | validation harness. |
| **materialized graphs** `*_initial.json`, `*_annotated.json`, `*_reattr.json`, `*_premerge.json`, b1 exports | **54 files in `data/exports/`** | **PRIVATE (belief-state)** under task rule — *currently public* | **the central conflict.** Also ~2 GB of bloat. |
| (s,t) belief field values inside those snapshots | within the 54 | **PRIVATE** under task rule | repo currently PUBLIC (Mismatch 2). |
| trial extractions + classifications | **828 files in `data/annotations/`** | **AMBIGUOUS** | registry-derived (public-eligible, `BOUNDARY.md:21`) **but** the task rule names "the raw pooled corpus" private. Owner habit = commit them (paid LLM output). **Flag.** |
| corpus NCT-id lists | 4 in `data/corpora/` | PUBLIC-eligible | frozen registry id lists; reproducibility. Low risk. |
| working notes / viz / caches | 58 in `data/dev`, `data/viz`, `data/cache` | not product | should not enter `main` regardless (regenerable clutter). |
| result docs (`B1_*.md`, `SAFETY_MANIFOLD_*.md`, `docs/branch-notes/*`, diagnosis md) | 37 `*.md` | PUBLIC | methodology/results — safe to publish; they describe the algorithm + honest nulls, not the belief-state. |

### Explicit ambiguous items for the human (with recommendations)

1. **The prediction/query path (the case the task named).** `src/prediction/path_query.py`
   (the compositional query algorithm) is **public**; but it *loads and serves the
   belief-state* (private under the task rule). **Recommendation: keep the prediction
   CODE public; keep the materialized belief-state it reads PRIVATE.** The open thing
   is "how to compose a P(success) from edge betas"; the paid thing is "the edge betas
   trained on the pooled corpus." This is the open-core seam and it is clean as long
   as no real belief-state snapshot is committed public.

2. **The hosted endpoint.** `src/api/` (FastAPI service) is **public code** and was
   **not changed by any of these branches** (0 files). Under the task rule the *running
   endpoint + the instance's belief-state* is the private product, but the *serving
   code* is public algorithm. **Recommendation: API code public; deployment config,
   instance, and loaded belief-state private.** No action needed in this merge.

3. **`data/annotations/` (828 files).** Registry-derived but the bulk of the merge and
   named "raw pooled corpus" by the rule. **Recommendation: keep public for now**
   (they are reproducible from public registries + paid extraction, and the repo's
   thesis treats them as public), **but flag** that if the "belief-state is private"
   thesis is adopted, the *pooled, processed* corpus may follow it private. Human call.

---

## Step 5 — The plan (do not execute)

### 5.1 Recency-ordered merge sequence

Because the four arcs are a linear stack, there is **one** integration: bring
`arch/triangulation-edge-weights` to `main`. The arc order A→B→C→D is already the
commit order. Recommended sequence:

1. **Exclude `archive/round-4-sub-chains`** — abandoned, 1 ancient commit, not in the
   stack. Do not merge.
2. **Resolve arc D's hygiene first** (see 5.2) — it is two un-pushed local commits that
   bundle features with ~2 GB of blobs. Clean these **before** they enter `main`'s
   permanent history.
3. **Decide the boundary thesis** (Step 3 Mismatch 1) — this gates whether
   `data/exports/` snapshots may land in `main` at all.
4. **Fast-forward / merge `arch` → `main`** once 2–3 are settled. Mechanically trivial
   (fast-forward, zero conflicts); the gating is policy, not git.
5. Push, open a single PR (the owner's convention is branch-first, owner-reviewed).

### 5.2 Per-overlap resolution

There are **no version conflicts to resolve** (linear stack — Step 2). The only
"who wins" calls are hygiene, and recency already decided the code:

- For every multi-arc file in the Step-2 table, **the newest arc's version is already
  the state on `arch`** — keep it. No older flag-gated capability is dropped by a newer
  commit (A34/B1/safety are *additive*, default-OFF; triangulation and the merge
  refactor are *retained*, not reverted). Nothing to call out as a regression.
- **`src/boundary.py`**: arc B's edit (field→public) is the newest state. **Hold it**
  pending the Step-3 thesis decision — it is the one line that flips with the rule.

### 5.3 Hygiene actions required before merge (recommended; each is its own step)

These are the real "harmonization" work. None is destructive to code:

- **Strip the ~2 GB belief-state blobs out of arc D** before they reach `main`. They
  are in `2eddfcc`/`c094f96` and earlier data commits. Options for the human:
  (a) `git rm --cached` the large `data/exports/*premerge*.json` + re-commit arc D
  cleanly; (b) rewrite the two un-pushed local commits to exclude them. They are
  regenerable (`build_graph.py … --assemble`) — losing them from history is safe.
- **Add `.gitignore` rules** for `data/exports/*premerge*.json` and the large
  `*_annotated.json` so they cannot be `add -A`-ed again (the mechanism that caused
  this).
- **Optionally split arc D** into `feat(inference): A34 routing`, `feat(graph): B1
  ontology`, `feat(safety): manifold` for reviewable history — currently `2eddfcc`
  bundles two flagged features.
- **Run `scripts/check_public_snapshots.py`** over any snapshot that *does* stay public,
  as the existing field-value leak gate (separate from the Step-3 thesis question).

### 5.4 Resulting public/private partition (after merge, under the task rule)

| stays PUBLIC in `main` | moves/relocates to PRIVATE (or stays out of public history) |
|---|---|
| all `src/` algorithm + schema (incl. prediction/query code, API serving code) | the materialized belief-state snapshots `data/exports/*_{initial,annotated,reattr,premerge}.json` |
| `scripts/` harnesses, `tests/`, result `*.md` docs | the (s,t) belief-**field** values (revert `PUBLIC_FIELD_NAMES` override if the task thesis wins) |
| `data/corpora/` id lists; the three OFF-by-default flags | the hosted endpoint instance + its loaded belief-state (deployment, not in repo) |
| `src/boundary.py` mechanism itself | **(decision)** `data/annotations/` pooled corpus — *if* the private-belief-state thesis extends to processed corpus |

**Moves needed to make the repo match the task rule** (if adopted): (1) relocate
`data/exports/` materialized snapshots out of the public repo into the private
artifact store / `eroom-enterprise`; (2) extend `src/boundary.py` with a notion of
"whole-snapshot private" (it only does field-stripping today); (3) reconsider the
`belief_field`-is-public override. None of these is required to merge the *code*; they
are required to make the repo's *data posture* match the new rule.

### 5.5 Open questions for the human (decide before executing)

1. **Which monetization thesis governs?** Repo today: open-core predictor, belief-state
   snapshots PUBLIC, moat = private partner data. Task rule: belief-state is the PRIVATE
   product. This single decision drives whether `data/exports/` may enter `main` and
   whether the field-public override stands. *Everything else is downstream of this.*
2. **The ~2 GB of blobs in arc D** — strip from history before merge (recommended) or
   accept them in `main`? They are not pushed yet, so this is the cheap moment to fix.
3. **`data/annotations/` (828 files)** — public (registry-derived, current habit) or
   private (the "pooled corpus")? Bulk of the merge.
4. **Split arc D's bundled commit** (`2eddfcc` = A34 + B1) into per-feature commits for
   review, or merge as-is?
5. **Promote `scratch/safety_manifold/` into `src/`** (productize) or leave it as
   research code excluded from the shipped surface?
6. **Confirm "the four branches"** — this plan reads them as the linear stack
   (endpoint-deorphan ⊂ st-field ⊂ triangulation ⊂ +A34/B1/safety) and excludes the
   abandoned `archive/round-4-sub-chains`. Correct?

---

## Stop

Plan only. The merge, any history rewrite, `.gitignore` change, and the boundary-thesis
decision are separate, approved steps. Awaiting human review.
