# SYNTH_REPORT_B1 — identity vs similarity for the singleton biology layer

**Question (Pillar B1).** The substrate is starved: efficacy edge reuse 1.24
trials/edge, 71% singleton biology (`SYNTH_REPORT.md`: recovery is near-zero
below reuse ≈ 8). B1's job is to manufacture sharing so the biology layer stops
reverting to prior. Two mechanisms exist — **identity** (discrete ontology
canonicalization: same concept → same node → hard-pooled Beta) and **similarity**
(the BioLORD (s,t) field: nearby concepts borrow via a kernel). This synth plants
known biology geometry and asks: *at the corpus reuse rate, does similarity clear
the recovery bar identity can't — and at what geometry_alignment does each stop
helping?*

**Bottom line.**
- **Identity (discrete) is the only potentially transformative lever** — at corpus
  reuse it lifts composite recovery `corr(π̂,π_true)` from 0.13 (no sharing) to
  **0.60** *if the grouping tracks reliability* (alignment → 1). But it is
  high-variance: at low alignment it **context-collapses to 0.02, worse than no
  sharing at all**. The crossover where it overtakes the field is **α_cross ≈ 0.4**.
- **Similarity (field) is robust but low-ceiling** — it never collapses (degrades
  gracefully to ≈ no-sharing at alignment 0) but even at perfect alignment caps at
  **0.27** at corpus reuse, less than half of discrete's 0.60. The kernel's soft
  borrowing + marginal fallback is far less sample-efficient than hard pooling.
- **The per-endpoint (s,t) factorization the real field uses BEATS a joint kernel
  at every cell** — `field_perdim ≥ field_full` throughout. The real field's
  `cos(s)+cos(t)` additive form is not a liability; the joint kernel over-couples
  and dilutes.
- **Routing (A3/A4) is on for all representations** — it composes with each; the
  geometry result is orthogonal to the routing result.
- **Decisive caveat (the estimator-power control):** at reuse 1.24 the
  geometry_alignment of real data is **unmeasurable** — the Step-3 estimator reads
  ~0.01 even for a *planted* alignment of 0.9, because edge reliabilities at reuse
  ~2 are themselves near-prior noise. So we cannot verify whether real biology
  geometry clears α_cross; committing to a representation now is a blind bet.

Self-contained under `scratch/synth/` (`geometry.py`, `alignment_estimator_power.py`,
building on `generate.py` + `recover.py`). Touches no `src/`, reads no real corpus.

---

## 1. The plant (geometry.py)

Extends `generate.make_ground_truth`. K=10 latent biology classes; each must-hold
edge gets a class, a biology-side embedding `T_a` and mechanism-side `S_a`
(unit vectors near the class centroid, within-class `scatter`=0.35). The knob:

> **geometry_alignment** ∈ [0,1] = how well embedding neighborhood predicts
> reliability. `r_a = Beta(6,2)-quantile(Φ(z_a))`,
> `z_a = align·z_class[c(a)] + √(1−align²)·ε_a`. At align=1 every edge in a class
> shares one reliability (proximity ⇒ identical r); at align=0 reliability is
> independent of the embedding. The Beta(6,2) marginal is held fixed across
> alignment — only the geometry↔reliability *coupling* moves.

Safety gates carry no geometry (the real safety layer keys on target identity,
already merged — confirmed invariant on real data, §4).

Four representations, all on the routed (mode-b) EM:
- **none** — per-node routed EM, class-mean backoff (the current substrate).
- **discrete** — KMeans-cluster the biology embeddings into K nodes, pool each
  cluster's trials, routed EM on merged nodes (identity; n_clusters = true K, the
  best case for discrete).
- **field_perdim** — per-node EM, then cross-node kernel smoothing with the real
  field's per-endpoint additive factorization `exp((cos S + cos T − 2)/bw)`.
- **field_full** — same, but a joint cosine over the concatenated `[S;T]`.

---

## 2. The recovery surface — composite corr(π̂, π_true), reuse × alignment

Corpus reuse row (1.24), 8 seeds:

| align | none | discrete | field_perdim | field_full | winner |
|---:|---:|---:|---:|---:|:--|
| 0.0 | 0.125 | **0.019** | 0.104 | 0.065 | none |
| 0.2 | 0.124 | 0.078 | 0.111 | 0.071 | none |
| 0.4 | 0.126 | 0.174 | 0.128 | 0.086 | **discrete** |
| 0.5 | 0.127 | 0.239 | 0.143 | 0.100 | discrete |
| 0.6 | 0.125 | 0.285 | 0.156 | 0.113 | discrete |
| 0.8 | 0.126 | 0.431 | 0.199 | 0.151 | discrete |
| 1.0 | 0.131 | **0.595** | 0.268 | 0.216 | discrete |

**α_cross = 0.4** (discrete overtakes the best field at corpus reuse). Below
align ≈ 0.3, *neither* sharing method beats no-sharing; discrete is actively
harmful (collapse). The same shape holds at reuse 2/4/8 (full grid in
`geometry_sweep.txt`): discrete's collapse-floor rises slowly with reuse (0.019 →
0.135 at reuse 8, align 0) but its high-alignment ceiling stays ~0.54–0.60;
field_perdim's ceiling rises with reuse (0.27 → 0.38 at reuse 8).

Two structural reads:
- **Hard pooling >> soft borrowing when the grouping is right.** Discrete merges
  ~50 same-reliability trials into one Beta (effective reuse past the reuse-8 bar →
  recovery), while the field's kernel weights (<1) plus marginal fallback never
  concentrate that much mass. Hence discrete 0.60 vs field 0.27 at align 1.
- **Soft borrowing is safe when the grouping is wrong.** The field's fallback to
  the pooled marginal means a bad neighbor contributes little; discrete's hard
  merge commits fully and averages unrelated reliabilities → collapse.

## 3. Effective reuse (% biology nodes ≥ 8) at corpus reuse 1.24

| representation | % nodes eff-reuse ≥ 8 |
|---|---:|
| none (discrete per-node) | 0% |
| discrete-canonical | 100% (hard pool into K clusters) |
| field_perdim | 29% (selective kernel mass) |
| field_full | 100% (indiscriminate — borrows from everything) |

`field_full` clears 8 for 100% of nodes yet recovers *worst* — proof that raw
effective-reuse count is necessary but not sufficient: indiscriminate borrowing
(wide kernel) inflates the count while averaging in irrelevant evidence. Discrete
also clears 100% but only *recovers* when alignment is high. Effective reuse must
be read together with whether the borrowed mass is on-reliability.

## 4. Per-dim vs full kernel; per-branch; safety

- **per-dim vs full:** `field_perdim` ≥ `field_full` at every (reuse, alignment).
  The per-endpoint additive factorization the real field already uses is the
  better choice — the joint kernel couples the (mechanism, biology) endpoints and
  dilutes the biology signal that carries the reliability. *Keep the factorization.*
- **per-branch:** the geometry is planted on the must-hold (biology) layer; gate
  (safety) recovery is unchanged across representations — biology re-representation
  doesn't touch safety, mirroring the real-data invariant (within-target AE SD
  0.048 unchanged, §Step-1 probe).

## 5. Can we even measure real alignment at this reuse? (the control)

`alignment_estimator_power.py` plants a known alignment, recovers reliabilities at
a given reuse, and runs the Step-3 estimator (LOO kNN reliability-prediction corr)
on the *recovered* (noisy) reliabilities:

| reuse | planted align | est on TRUE r | est on RECOVERED r |
|---:|---:|---:|---:|
| 1.24 | 0.0 | −0.02 | −0.02 |
| 1.24 | 0.5 | 0.25 | **0.00** |
| 1.24 | 0.9 | 0.66 | **0.01** |
| 2.0 | 0.9 | 0.64 | 0.01 |
| 8.0 | 0.9 | 0.65 | 0.02 |
| 64.0 | 0.9 | 0.28 | 0.21 |

**At reuse ≤ 8 the estimator is blind** — it reads ~0.01 even when the true
alignment is 0.9, because reuse-2 reliabilities are near-prior noise. It only
recovers signal at reuse ≈ 64. So Step 3's real ≈ 0.05 is **measurement-floored,
not a verified misalignment**: at this substrate the geometry's quality is
unknowable from outcome data alone.

## 6. Verdict for B1

1. **The field (similarity) is not the lever.** Even idealized cross-node, its
   ceiling at corpus reuse is ~0.27 vs discrete's ~0.60; and the *deployed* field
   is per-edge (within-edge only — §Step-1), so it can't cross-node-share at all
   and its honest holdout does not beat the scalar.
2. **Discrete (identity) canonicalization is the only transformative option** at
   the current reuse — but its entire payoff rides on a grouping with alignment
   > α_cross ≈ 0.4, and it context-collapses (worse than nothing) below it.
3. **The grouping must be externally validated, not embedding-clustered.** BioLORD
   alignment is unmeasurable at reuse 1.24, and embedding-clustering at low/unknown
   alignment is the collapse regime. The safe analog is the **Reactome-for-mechanism
   pattern**: map biology to a *curated* controlled vocabulary (GO biological
   process / MONDO), whose grouping is functionally validated independent of the
   sparse outcome data — then re-measure alignment as reuse rises (Pillar C).
4. **If the field is used at all, use the per-endpoint factorization** (measured
   better than the joint kernel), as a soft prior for the singleton tail that
   doesn't map cleanly — the junior partner in a hybrid, never the lever.

Reproduce:
```bash
cd scratch/synth
../../.venv/bin/python -m scratch.synth.geometry --seeds 8 \
    --reuses 1.24 2.0 4.0 8.0 --alignments 0.0 0.2 0.4 0.5 0.6 0.8 1.0
../../.venv/bin/python -m scratch.synth.alignment_estimator_power
```
