# Eroom Bio — the merge/pooling map (how sharing is actually implemented)

**Investigation, not fix. Read-only.** Goal: before the metric layer is rewritten
around *effective reuse* and per-branch / per-target recovery, map the existing
sharing machinery precisely so the new scoreboard hooks into what is implemented —
both the scalar Beta edge beliefs and the (s,t) field.

All numbers on `data/exports/multi_500_annotated.json` (n=472 in-graph trials, 3851
nodes, 12734 edges; this snapshot carries the materialized `belief_field` + the
`_belief_field_vectors` dedup table). Probes: `scratch/diagnostics/merge_pooling_probe.py`
(P1/P2/P4/P5/P6), `b1_substrate_probe.py` (effective reuse), `field_holdout.py` +
`scripts/scalar_vs_field_auroc.py` (honest field OOS). Cross-checks against
`FINDINGS.md` (P6 reuse, P9 AE SD) are exact.

---

## P0 — How many belief representations, and which one predicts?

**There are two stores. The deployed predictor reads the scalar only.**

1. **Scalar `EdgeBeliefState`** — `src/graph/models.py:824`. One `Beta(alpha, beta)`
   per edge identity; `expected_probability = α/(α+β)` (`models.py:837`). This is the
   public marginal. Updated by `apply_virtual_evidence` (`src/inference/beliefs.py:541`):
   `α += n_eff·p_obs`, `β += n_eff·(1−p_obs)`.

2. **(s,t) belief field** — `belief_field: dict | None` on the *same* `EdgeBeliefState`
   (`models.py:835`), holding a serialized `BeliefField` (`src/inference/belief_field.py:184`):
   a sparse per-edge anchor list over the joint (source-desc, target-desc) BioLORD space.
   Flag-gated: `EROOM_BELIEF_FIELD`, **default OFF** (`belief_field.py:32`).

**Deployed predictor read path.** The API `/predict` (`src/api/main.py:247`) calls
`path_query.PredictionEngine.predict`. `_collect_edges` (`src/prediction/path_query.py:1064`)
retrieves beliefs via `graph.get_edge_belief(src_id, tgt_id, edge_type)` and reads
`belief.expected_probability` / `evidence_strength` — **the scalar**. `path_query.py`
imports neither `belief_field` nor `field_prediction` (grep: zero hits). The field is
read only by `src/prediction/field_prediction.py`, which is used exclusively in eval
scripts (`scalar_vs_field_auroc.py`, `eval_holdout_kfold.py --field`, `field_holdout.py`).

**Field status: experimental / shadow.** It is materialized post-hoc
(`scripts/materialize_belief_field.py`) onto private snapshots; nothing in the
prediction service or the default holdout path reads it.

> **Correction to the task premise.** FINDINGS said the field's honest OOS "was not
> run." It *has since been run this session*, two ways, and the field does **not** beat
> the scalar (P4). The premise that the field is an unmeasured upside no longer holds.

**Verdict:** two stores, one live (scalar). The field is a derived refinement, off by
default, measured to be OOS-neutral-to-worse.

---

## P1 — Node identity / merge, per node type

Merge runs in tiers over a union-find (`src/graph/node_merge.py:210 _classes_for_type`):
**Tier 1** ontology/id (`_ontology_key`, `node_merge.py:110`) + **Tier 1b** ChEMBL
`stable_id` for compounds (`_chembl_key:123`); **Tier 2** SapBERT canonical `name_id`
(`_name_key:118`); **Tier 3** BioLORD cosine ≥ `biolord_threshold` (default 0.85,
`MergeConfig:80`) / box. The id *minted at populate time* is what Tier 1 keys on, so it
decides everything.

Integer reuse = # distinct host trials whose chain referenced the node (AE via its
`causes_ae`/`target_associated_ae` edges). `%==0` = present but unused (curated/ghost);
`%==1` = singleton.

| node type | id / merge key (file:line) | median | mean | %unobs | %singleton | %≥8 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **Mechanism** | Reactome `R-HSA-*` / GO stable_id (id-merge; `populate.py:654` content-hash SUPERSEDED) | 2 | 4.89 | 0% | 35% | **18.7%** | **merges well — densest unit** |
| **Target** | gene symbol / Open Targets ontology id; ChEMBL `stable_id` for compounds | 1 | 2.69 | 33% | 28% | 10.0% | merges decently (2nd densest) |
| **AdverseEvent** | `AE:<MedDRA-normalized term>` | 1 | 2.64 | 3% | 61% | 5.7% | partial — singleton-heavy |
| **Biology** | `bio:<sha1(norm desc)[:12]>` content-hash (`populate.py:636,651`) | 1 | 2.65 | 0% | **71%** | 5.2% | **collapses to singletons** |
| **Indication** | indication slug / EFO | 1 | 1.80 | 49% | 30% | 4.1% | half ghost; long tail singleton |
| **Population** | subgroup id (line/stage feature slug) | 1 | 1.20 | 21% | **69%** | 2.2% | collapses to singletons |
| **Intervention** | normalized drug name, merged on ChEMBL id (`_chembl_key:123`) | 1 | 1.40 | 36% | 40% | 2.4% | collapses to singletons |
| **Endpoint** | `{class}` / `{class}_{measure_slug}`, disease-agnostic (`populate.py:378`) | 1 | 0.75 | **49%** | 47% | 1.2% | mostly ghost/singleton |

**Which types create real sharing, and why:** sharing tracks the id scheme's
*determinism*. Reactome-id (mechanism) and HGNC/ChEMBL-id (target, intervention-as-ChEMBL)
are controlled vocabularies → same concept lands on same node → mechanism (median 2,
18.7%≥8) and target (10%≥8) are the only two layers with real recurrence. Everything
keyed on free-text identity collapses: biology's `bio:<hash>` makes 71% singletons (a
paraphrase is a new node), endpoint's measure-slug + indication leave ~49% of nodes
*unused by any chain* (ghosts). This is exactly architecture-v2 **B1**: biology is the
single biggest sparsity lever, and the fix is the id scheme (controlled vocabulary),
not the merge tiers (which already run BioLORD Tier 3 at 0.85 and still can't rescue a
content-hash identity).

---

## P2 — Edge identity & scalar pooling

**Edge identity tuple = (source canonical id, target canonical id, edge_type)** — the
NetworkX `MultiDiGraph` `(u, v, key)`. Pooling onto one Beta is enforced at retrieval:
`get_edge_belief(src_id, tgt_id, edge_type)` returns the single `EdgeBeliefState` for
that triple (`_collect_edges`, `path_query.py:1130`); all evidence with the same triple
accumulates onto one `Beta` via `apply_virtual_evidence` (`beliefs.py:541`).

**Scalar pooling is exact-identity only — no similarity component.** Fuzzy/geometry
matching lives at *assembly time on node ids* (`node_merge.py` Tiers 2–3), which makes
edges share identity by sharing endpoints; the scalar belief store itself does no
similarity pooling. Two retrieval-time *hierarchical* exceptions exist on the backbone
(this is partial B2/B3 already shipped): `responds_differently` backs off to its is-a
ancestor (`path_query.py:1103`), `biology_drives`/`endpoint_captures` back off to the
`subtype_of` indication parent (`:1116`), and `mechanism_affects` is tissue-conditioned
(`:1125`). Everything else is context-free exact-identity.

**Edge integer reuse (# distinct host NCTs) — cross-checks FINDINGS P6 exactly:**

| class | edges | mean trials/edge | reuse 0 | reuse 1 | reuse ≥2 | max |
|---|---:|---:|---:|---:|---:|---:|
| efficacy | 4872 | **1.24** | 43% | 37% | 20% | 35 |
| measurement | 2656 | **0.36** | 68% | 29% | 3% | 13 |
| safety | 3986 | **0.90** | 18% | 77% | 6% | 6 |
| modulation | 331 | 1.07 | 0% | 94% | 6% | 3 |

By type, all the efficacy reuse is in two edges: `modulates_via` (mean 2.29, 37%≥2) and
`mechanism_affects` (1.78, 28%≥2) — the mechanism-incident, Reactome-keyed edges.
`affects` 0.63, `biology_drives` 0.39, `reflects_biology` 0.63, `endpoint_captures` 0.38
barely recur; **`responds_differently` = 0.00 (100% empty)** and `target_associated_ae`
= 0.00 trial-sourced (curated-only). One Beta per identity, confirmed.

**Verdict:** exact-identity pooling; reuse is structurally starved (efficacy 1.24) and
concentrated on the two mechanism-keyed edges.

---

## P3 — The (s,t) field: full anatomy

- **s, t** = BioLORD-2023 embeddings (`FremyCompany/BioLORD-2023`, **768-dim**,
  `src/graph/biolord_embeddings.py:37`) of the **source/target descriptions** — per-chain
  typed descriptions (mechanism/biology/population) where present, node description
  otherwise (`field_prediction.build_st_desc_map:82`). Defined on the **7 backbone edge
  types** (`materialize_belief_field.EDGE_SPECS:77`). **No field on AE edges.**
- **Representation:** a **sparse anchor list per edge** — *kernel over observed pairs*
  (Nadaraya–Watson-style), **not** a grid / GP / inducing points. Each evidence record →
  one `FieldAnchor(s_i, t_i, α_i, β_i, nct)` (`belief_field.py:147`) with
  `(α_i, β_i) = (n_eff·p_obs, n_eff·(1−p_obs))` — the *same increment the scalar applies*.
  Snapshot totals: 5881 edges carry a field, **20644 anchors**, deduped to 2034 distinct
  (s,t) vectors.
- **Full-vector, not per-dimension.** `_cosine` (`belief_field.py:246`) runs over the
  whole 768-vector; s and t cosines are summed in the kernel. No per-dim decomposition.
- **Borrowing / kernel:** `w_i = exp((cos(s,s_i) + cos(t,t_i) − 2) / bandwidth)`,
  **bandwidth default 0.25** (`belief_field.py:43`, `query:282`). A query for edge (A,B)
  at (s′,t′) returns `(fallback_a + Σ w_i α_i, fallback_b + Σ w_i β_i)`. Far from anchors
  → `fallback_prior` = `fallback_strength`(=2.0) units of mass at the **scalar mean**
  (`belief_field.py:237`). So the field is a strict *refinement of the scalar*: local
  where there is nearby evidence, pooled-scalar otherwise.
- **The decisive structural fact:** `query()` sums **only that one edge's own anchors**
  (`belief_field.py:296`). There is **no cross-edge / cross-node borrowing** — evidence
  spreads across an edge's (s,t) surface, never to a different edge. A singleton biology
  node cannot borrow from anything.
- **Write path:** `materialize_belief_field.py` replays each edge's *existing scalar
  evidence records* at (s,t)=BioLORD(per-chain descriptions); a trial with K chain (s,t)
  pairs splits the record's `n_eff` evenly across them (`:1-18`). Post-hoc — **not** in
  the live attributor.
- **Materialization cost (measured this session, `field_holdout_all.log`):** re-attr 6s +
  **materialize 23s/fold** (5919 edges, 19040 anchors); ~298s/fold all-in incl. prediction;
  full honest 5-fold ≈ 25 min. The `without_trial` additive LOO (`belief_field.py:221`)
  gives *exact* anchor-drop leave-one-out with **no** re-materialization (the kernel is
  additive over anchors) — but its marginal fallback still carries the held-out trial's
  scalar, leaky on singletons, which is why `field_holdout.py` does the full per-fold
  re-materialization for the clean number.
- **Status in path_query:** nothing reads it. Parallel/unused in the deployed predictor.

---

## P4 — Relationship between scalar and field

- **Derived, not parallel.** The field is built by *replaying the scalar's own evidence
  records* (P3 write path); its `marginal_alpha/beta` **are** the scalar Beta (the
  fallback). It is a refinement, not an independent representation.
- **They agree almost perfectly.** Over 3064 backbone edges with ≥2 host trials:
  **corr(scalar E[p], field E[p]) = +0.990**, **mean |Δ| = 0.025**; the field moves >0.05
  from the scalar on only **11%** of edges (`merge_pooling_probe.py`).
- **Honest OOS — the field does not beat the scalar:**
  - `field_holdout.py` (fully honest, per-fold re-materialized clean graph, n=221):
    **scalar 0.565 vs FIELD 0.561 (Δ −0.004)**.
  - `scalar_vs_field_auroc.py` (self-excluded per-edge): field strictly ≤ scalar at every
    bandwidth — trial-softmin **scalar 0.648 vs field 0.545**, learned-logistic 0.766 vs
    0.643. The *only* edge type the field improves is `modulates_via` (+0.144). And the
    **bandwidth sweep 0.02 → 0.6 is identical** (soft_fld 0.545, learn_fld 0.643 at every
    bw) — bandwidth is inert, because after self-exclusion there are no nearby cross-trial
    anchors at any width and the query falls back to the (diluted) scalar.
- **Which store the new scoreboard should read:** **the scalar.** Both expose per-edge
  posteriors uniformly (scalar α,β directly; field via `query()→(a,b)`), so the scoreboard
  *can* hang on either — but at corpus reuse the field is corr-0.99 to the scalar and
  OOS-neutral-to-worse, so wiring it buys nothing until reuse rises.

---

## P5 — Effective reuse (the bridge to the new scoreboard)

- **integer reuse** = # distinct host trials (P1/P2).
- **effective reuse (field)** = kernel-weighted effective sample size at a query point,
  `Σ_j w(q, anchor_j)` over the edge's own anchors, with the field's real kernel +
  bandwidth (0.25). Two honest variants, because the answer hinges on which you use:
  - **(a) within-edge, self-inclusive** (`b1_substrate_probe.py`): field eff ÷ trial-count
    = **1.10× mean**; biology-node %≥8 rises 5.2% → 9.9%. **But this counts a trial's own
    fan-out** (one trial puts multiple chains' anchors on the same merged edge), not new
    trials — it is not cross-trial reuse.
  - **(b) cross-trial, self-trial excluded** (`merge_pooling_probe.py`, the honest one):
    **median(eff/integer) = 0.17×** for efficacy; field-effX %≥8 = **4.6% vs integer 6.7%**
    (efficacy), 0.2% vs 0.1% (measurement). The field manufactures *less* cross-trial
    reuse than the raw integer count, because different trials' (s,t) sit far apart in
    cosine → kernel weight ≪ 1.

| | integer %≥8 | field eff %≥8 | multiplier |
|---|---:|---:|---:|
| efficacy edges | 6.7% | 4.6% (cross-trial) | **0.17×** |
| measurement edges | 0.1% | 0.2% | ~0× |
| biology nodes (max over edges) | 4.6% | 9.9% (self-incl, fanout) / ≤4.6% (cross-trial) | 1.1× / ≤1× |

**Verdict:** the field does **not** manufacture the reuse the synth says we need.
Self-inclusive it adds ~10% (fan-out, not transfer); cross-trial it *subtracts*. The
fraction of biology/efficacy structure clearing **cross-trial effective reuse ≥ 8 is
essentially unchanged** from the ~0–7% integer baseline. Bandwidth is proven inert
(P4 sweep). The reuse lever is not the field; it is Pillar B (id re-canonicalization).

---

## P6 — Per-branch & per-target breakdown

**By edge class (integer reuse):** efficacy 1.24, measurement 0.36, safety 0.90 (P2).
Sharing is strongest on the **mechanism-incident efficacy edges** (`modulates_via` 2.29,
`mechanism_affects` 1.78 — Reactome-keyed) and on **safety/AE** (target-keyed).
Effective reuse adds nothing on top (P5).

**Per target (the densest-sharing unit after mechanism):** targets with ≥3 anchored
edges (n=147) → distinct-trial reuse median **2.0**, mean **4.0**, max 35. Within-target
`target_associated_ae` posterior E[p] **SD = 0.048 (n=45 targets)** — confirms FINDINGS
P9 to the digit. On-target tox is the most *consistent* (lowest-variance) cross-trial
signal in the graph, precisely because AE keys on the well-merged target id and bypasses
the field entirely. Targets are confirmed as the densest-sharing unit after mechanism
(target node reuse mean 2.69 / 10% ≥8, P1).

---

## The sharing map

| node type | id scheme | integer reuse (median / %singleton / %≥8) | merges well? |
|---|---|---|---|
| Mechanism | Reactome `R-HSA-*` / GO id | 2 / 35% / 18.7% | **yes — densest** |
| Target | HGNC gene / ChEMBL id | 1 / 28% / 10.0% | yes — 2nd densest |
| AdverseEvent | `AE:<MedDRA term>` | 1 / 61% / 5.7% | partial (target-keyed → consistent) |
| Biology | `bio:<sha1(desc)>` content-hash | 1 / 71% / 5.2% | **no — collapses to singletons** |
| Indication | slug / EFO | 1 / 30% / 4.1% | weak (49% ghost) |
| Population | subgroup feature slug | 1 / 69% / 2.2% | **no — singletons** |
| Intervention | drug name → ChEMBL id | 1 / 40% / 2.4% | no — singletons |
| Endpoint | `{class}_{measure}` disease-agnostic | 1 / 47% / 1.2% | no — 49% ghost |

## Scalar vs field

- **Live:** the **scalar** (`path_query`/API). The field is off-by-default, read only by
  eval scripts.
- **Agree?** Yes — corr **0.990**, mean |Δ| 0.025; field moves >0.05 on 11% of edges.
- **Effective-reuse multiplier the field provides:** ≈**1.1×** (within-edge, fan-out
  inflated) and **0.17×** (honest cross-trial). The field localizes existing evidence; it
  does **not** pool across nodes/edges, so it cannot lift a singleton.
- **Honest OOS:** field 0.561 vs scalar 0.565 (−0.004); strictly ≤ scalar self-excluded;
  bandwidth-invariant.

## The one decision input

**At the field's current bandwidth (0.25), the share of biology/efficacy structure that
clears cross-trial effective reuse ≥ 8 is essentially unchanged from the integer baseline
(~0–7%) — the field is *not* close, and bandwidth is not the knob.** Three independent
confirmations: (i) `query()` sums one edge's anchors only — no cross-node pooling exists
to widen; (ii) the cross-trial effective/integer multiplier is 0.17× (different trials'
(s,t) are too far apart in BioLORD cosine to borrow); (iii) the self-excluded AUROC is
identical from bandwidth 0.02 to 0.6. **The geometry is too sparse regardless.** The lever
that raises reuse is Pillar B (re-canonicalize biology onto a controlled vocabulary so
edges recur), not field bandwidth tuning. The new scoreboard should read the scalar
posteriors and measure reuse on node/edge identity, not on the field surface.
