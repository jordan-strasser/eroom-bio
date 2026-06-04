# Scaling design decisions (deferred)

Forward-looking design choices that are **decided but intentionally not built yet**,
because they only pay off past the current corpus size. Each has a trigger. Until
the trigger, the interim mechanism (noted) is sufficient. Build no earlier than the
trigger — it's pure 200K-runway, no benefit at today's n.

North star: process ~200K trials incrementally (append, never full rebuild). See
`CLAUDE.md` and memory `project_scale_architecture`.

---

## 1. Merge candidate-generation: hybrid ANN, O(n log n)  ·  trigger: > ~1k trials

**Problem.** The geometric merge tiers (`node_merge._classes_for_type`, BioLORD on
biology descriptions + SapBERT on mechanism names) score **every pair** — O(n²·d),
d=768. At the 200K north star (biggest type ~100K nodes) that's ~4×10¹⁰ cosines and a
160 GB similarity matrix that doesn't fit. Infeasible on both compute and memory.

**Decision — "Half 1" hybrid (the O(n log n) option we agreed on).** Keep embeddings
**fixed** (BioLORD/SapBERT already place each node correctly; do NOT relax/"settle"
them — that adds iterative cost, non-determinism, geometry distortion, and loses
reversibility-by-projection). Replace all-pairs with a **spatial index**
(HNSW / faiss / sklearn BallTree):
- build index over the node embeddings: O(n log n) one-time;
- each node compares only against its ~k nearest neighbors: O(n·k·d).

Result: **~3,000× fewer cosines at 200K** (~1.3×10⁷ vs ~4×10¹⁰ at k=64), 160 GB → ~1 GB.
Complexity lands at **O(n log n)** (k≈const). Append becomes O(new · log n) (index
query) vs the interim O(new · total).

**Implementation seam (already in place).** `node_merge._pair_indices(keys, new_ids)`
is the candidate-pair generator that feeds union-find. It already has two strategies:
- `new_ids is None` → all pairs (fresh build),
- `new_ids` set → only pairs touching a just-added node (incremental append, shipped).

The ANN upgrade is a **third strategy** — yield index-neighbor pairs — localized to
that one generator. **No change** to union-find, lossless belief union/replay, chain-id
rewrite, or the authoritative exact-key tiers (id / chembl / name_id stay hard, O(n);
only the geometric description-identity tiers use the index).

**Trade-off.** ANN is approximate — ~95–99% recall (HNSW). A missed neighbor = safe
**under**-merge (a duplicate node survives), tunable via ef/M. We trade ~1–5% merge
recall for 3 orders of magnitude.

**Crossover (pure-Python pairwise, rough):** n≈5k ~seconds (fine) · n≈20k ~minutes ·
n≈50k ~tens of minutes (wall). So wire the index around rung 3–4.

**Interim until trigger (shipped):** the `new_ids` restriction (append scores only
new-involving pairs, O(new · total)) + batched encoders (one `model.encode`, one cache
round-trip). Full-rebuild merge is still O(n²) — fine below ~1k, that's the trigger.

---

## 2. Field bandwidth tuning for optimal evidence sharing  ·  trigger: alongside §1

**What.** The (s,t) belief field gives each backbone edge anchors at
`(s,t)=BioLORD(per-arm descriptions)`; the **bandwidth** controls how far evidence at one
anchor bleeds to nearby (s,t) queries — i.e. how aggressively cross-context evidence is
**shared** vs kept local. Today it's a fixed default (~0.25, set in
`node_merge._merge_belief_data` / `BeliefField`). See memory `project_st_field_architecture`.

**Decision.** Tune bandwidth for **optimal sharing**, validated on holdout P(success)
(calibration + AUROC) with proper CV. This is the **"anchor softer instead of merge
harder"** lever: bandwidth is the continuous soft-boundary analog of the hard merge
radius — looser merge + tuned bandwidth can pool evidence while preserving sub-concept
resolution via separate anchors. So **tune the merge boundary and the field bandwidth
together**, not independently.

**Guardrail (same as the merge boundary).** P(success) is a **validator, not the sole
optimizer**. Over-wide bandwidth → over-sharing → regression toward the base rate (a
metric can improve for the wrong reason). Set sharing primarily by a concept-identity /
coverage criterion; use holdout P(success) to check it didn't hurt. Don't tune on the
5-trial holdout (overfit trap — see `project_round28_*`, `feedback_premature_classification`).

**Trigger.** Once there's enough cross-context data to estimate optimal sharing — i.e.
alongside the §1 merge work, post-~1k trials.

---

*Both items are accuracy/efficiency levers on the geometric layer; the authoritative
id/chembl tiers and the chains-first build are unaffected. Efficiency (how neighbors are
found) and accuracy (how big the boundary / how wide the bandwidth) are orthogonal and
compose.*
