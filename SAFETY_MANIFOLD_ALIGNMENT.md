# Safety manifold — Phases 0–1 (the alignment gate)

**Thesis.** The exact-target on-target safety signal already works and is shippable
(within-target `target_associated_ae` posterior SD **0.048**, n=45 — the most
consistent cross-trial signal in the graph; MERGE_POOLING_MAP P6). This work adds
**cross-node borrowing** so that (1) the safety branch gets effective reuse above the
exact-id integer count, (2) novel compounds/targets inherit a calibrated liability
from neighbors, and (3) we can decompose an observed AE into **on-target vs
off-target**. The borrowing rides on geometry that is *reliability-aligned* —
chemical structure → off-target tox, target homology/pathway → on-target tox — i.e.
real pharmacology, not the text proximity the BioLORD field used (and which gave a
0.17× cross-trial multiplier, B1_field_loo_findings.md).

This is the safety analog of the B1 gate: **build a manifold's borrowing only if
proximity on it predicts shared AEs.** If it doesn't, pooling over it would be wrong
— skip it and keep exact-id. Everything below is read-only over
`data/exports/multi_500_annotated.json` (n=472 in-graph trials, 3851 nodes).
Harness: `scratch/safety_manifold/{geometry.py, phase1_alignment.py}`.

---

## Phase 0 — Safety substrate + geometry inventory

### AE substrate (the thing we borrow over)

| edge | direction | count | distinct sources | distinct AEs | belief path |
|---|---|---:|---:|---:|---|
| `causes_ae` | compound → AE (PT) | 3286 | 268 compounds | 676 | per-trial conjugate update (`attributor.attribute_adverse_events`) |
| `target_associated_ae` | target → AE (PT+SOC) | 700 | 53 targets | 163 | rebuilt by `ae_propagation.propagate_to_target_associated_ae` from sibling compounds binding the same target |

AdverseEventNode total 697 (PT 676, SOC 21). The deployed read path is
`path_query._collect_safety_risks` → `_compute_safety_penalty`: it pulls the
compound's `causes_ae` and the target's `target_associated_ae` edges, severity-weights
each, three-gate modulates (belief × evidence × DLT-fraction), and takes **max over
AEs** capped at `_SAFETY_PENALTY_CAP`. Both AE edge families key on **exact node id**
— `causes_ae` on the ChEMBL-merged compound, `target_associated_ae` on the
HGNC-merged target. **Neither carries a belief field; neither borrows across nodes
today.** The propagation already does one kind of cross-compound pooling — sibling
compounds binding the *same exact target* vote on that target's AE — but it requires
*exact* `affects`-target identity; there is no structure- or pathway-similarity
borrowing anywhere.

### Geometry available without downloads

| manifold | geometry | source | buildable? |
|---|---|---|---|
| **Compound** | Morgan/ECFP4 fingerprint + Tanimoto | RDKit (installed from pythonhosted) over **SMILES** | SMILES are **0% populated** on `InterventionNode`, but `chembl_id` covers 55% (74% of AE-bearing compounds). SMILES resolved once from ChEMBL REST by `chembl_id` → cached at `data/cache/chembl_smiles.json` (204/265 resolved; biologics correctly have none). |
| **Target** | Reactome/GO pathway co-membership (Jaccard) | **in-graph, no download** | The `MechanismNode` **id is the pathway id** (`R-HSA-*`/`GO:*`); a target's pathway set = the mechanism ids reached via `modulates_via`. |
| Target (upgrade) | sequence homology | — | **not vendored** — no protein sequences in `data/`. Pathway co-membership stands in; homology is a later upgrade only if sequences are added. |

**Coverage on the entities that actually carry an AE profile:**

| manifold | entities with AE edges | with manifold representation | profile size (median / mean) |
|---|---:|---:|---:|
| Compound (Morgan fp) | 268 | **150 (56%)** | 10 / 12.1 AEs |
| Target (pathway set)  | 53  | **43 (81%)** | pathway-set 5 / 5.9 |

The 118 uncovered compounds are almost all biologics (Protein/Antibody/Gene/Enzyme/
Oligosaccharide — no SMILES by nature) plus 71 with no `chembl_id`. **Structure
borrowing is inherently a small-molecule concept; biologics correctly fall back to
exact-id.** What's missing and why, stated exactly: per-compound SMILES (absent from
the graph, recovered from ChEMBL by id — done), and protein sequences (absent, not
recoverable offline — pathway co-membership used instead).

---

## Phase 1 — Alignment gate (does geometry predict AE sharing?)

For every entity pair we correlate **geometric similarity** with **AE-profile
similarity** (idf-weighted Jaccard over the observed AE sets, discounting ubiquitous
nausea/fatigue; plain Jaccard and cosine agree). AEs count only with ≥1 evidence
pseudo-count; an entity needs ≥2 such AEs to enter a pair. Significance is a
permutation null (shuffle AE profiles across entities, 200 perms).

### Compound manifold — Tanimoto (ECFP4) vs AE-profile (132 compounds, 8646 pairs)

```
geo-sim bin       n      mean AE-sim (idf-Jaccard)
[0.00,0.10)    4885      0.0267
[0.10,0.20)    3587      0.0337
[0.20,0.40)     142      0.0821
[0.40,0.60)      21      0.1027
[0.60,1.01)      11      0.1207
```
- Overall Pearson **+0.097**, Spearman +0.085. **Permutation null p = 0.005** (null
  mean ≈ 0) — the association is real, not chance.
- The overall r is *diluted by design*: 98% of drug pairs are structurally unrelated
  (Tanimoto < 0.2), where structure says nothing and a kernel assigns ~0 weight. The
  quantity a kernel actually uses is `E[AE-sim | high geo-sim]`, and that **rises
  monotonically to ~4.5× the baseline** at the near neighbors it borrows from.
- **Genuine off-target signal:** restricting to *different-target* pairs (8513),
  the trend survives (0.026 → 0.033 → 0.058 across Tanimoto bins) — structure
  predicts AE sharing even controlling for shared target, which is exactly the
  chemistry-specific (scaffold-escapable) liability Phase 3 attributes off-target.
- **Neighbor availability (can it borrow at this n?):** 50% of fp-compounds have ≥1
  neighbor at Tanimoto ≥ 0.3, 36% at ≥ 0.4. So the manifold helps roughly half the
  small-molecule compounds and falls back to exact-id for the rest.

### Target manifold — pathway co-membership vs on-target AE (34 targets, 561 pairs)

```
geo-sim bin       n      mean AE-sim (idf-Jaccard)
[0.00,0.10)     545      0.0464
[0.10,0.20)       6      0.2858
[0.20,0.40)       9      0.1660
[0.40,0.60)       1      0.0926   (single pair — noisy tail)
```
- Overall Pearson **+0.207**, Spearman +0.162. **Permutation null p = 0.010.**
- Any pathway overlap lifts on-target AE-sharing from 0.046 to 0.17–0.29 — a **~4–6×
  contrast** between pathway-neighbors and pathway-strangers. (The extreme-similarity
  bins are sparse, so the robust signal is the near-vs-far jump, not the tail shape.)
- **Neighbor availability:** 82% of profiled targets have ≥1 pathway-neighbor at
  Jaccard ≥ 0.05, 53% at ≥ 0.1. And there is a large *borrowing pool*: 147 targets
  carry pathways but only 34 carry an AE profile — so a novel/sparse target can
  inherit a liability from the profiled pathway-neighbors (the novel-entity value
  prop, tested in Phase 4).

### Why this is the borrowing the text field couldn't do

The BioLORD (s,t) field gave a **0.17× cross-trial multiplier** (it *subtracted*
reuse) because different trials' text embeddings sit far apart in cosine — the
geometry was not reliability-aligned, so near-in-text did not mean near-in-outcome.
Here the opposite holds: **near-in-structure and near-in-pathway demonstrably mean
near-in-AE-profile** (4–6× lift at the neighbors, permutation-significant). The
precondition the field failed, both manifolds pass.

---

## The gate decision

Bar (per manifold, all three required):
1. **Permutation-significant** (p < 0.05) — the alignment is real.
2. **Monotone near-vs-far lift ≥ 2×** — high-similarity pairs share materially more AEs.
3. **Non-trivial neighbor availability** (≥⅓ of entities have a borrowable neighbor).

| manifold | (1) p | (2) near/far lift | (3) neighbor coverage | **decision** |
|---|---:|---:|---:|---|
| **Target (pathway)** | 0.010 | ~6× | 82% @ J≥0.05 | **BUILD — primary lever** |
| **Compound (ECFP4)** | 0.005 | ~4.5× | 50% @ Tan≥0.3 | **BUILD — secondary, n-limited** |
| Target (homology) | — | — | — | **SKIP — sequences not vendored**; revisit if added |

**Both manifolds clear the gate → build both** (Phase 2). The target manifold is the
stronger and better-covered lever and carries the cleaner on-target signal; the
compound manifold is real and permutation-significant but helps ~half the
small-molecules at n=472 and is the source of the off-target signal. Borrowing layers
**on top of** exact-id (a node's own observed AEs dominate; neighbors fill the tail),
behind `EROOM_SAFETY_MANIFOLD` (default OFF), A/B vs exact-id. If a manifold's Phase-2
effective-reuse multiplier fails to beat 1×, drop that borrowing and ship exact-id —
downside is bounded.
