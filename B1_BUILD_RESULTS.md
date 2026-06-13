# B1 build results — biology re-canonicalized onto GO-biological-process

Phase 1 said **GO** (`B1_DECISION.md`). This is the built A/B: the only difference
between the two snapshots is the biology id scheme, behind `EROOM_BIO_ONTOLOGY`.

## What was built

- **`src/graph/biology_ontology.py`** — flag-gated (`EROOM_BIO_ONTOLOGY`, default OFF)
  desc→GO-BP id lookup. Maps a biology description to `bio:GO:xxxxxxx` (nearest GO
  biological-process term, BioLORD cosine ≥ `EROOM_BIO_ONTOLOGY_GATE`, default 0.60),
  else falls back to the `bio:<sha1>` content hash (no node lost). Map precomputed at
  `data/cache/biology_ontology_map.json` (346 descriptions; `b1_build_ontology_map.py`).
- **`populate.py:_biology_id_from_description`** — consults the helper when the flag is
  on, so a fresh `--corpus` build mints GO-keyed biology ids (mirrors Reactome for
  mechanism). Default path (flag off) is byte-identical to before.
- **A/B harness** — `b1_rekey.py` stamps GO `ontology_id` on the biology nodes of
  `multi_500_initial.json` and runs the biology-only Tier-1 `node_merge.assemble`
  projection; both arms re-attribute from the same annotations (`run_b1_ab.sh`).

Re-key (gate 0.60): **192/241 biology nodes mapped, 241→181 nodes (60 merged), 789
chain biology-id references rewritten.**

## Phase 3 scoreboard (probe: `b1_phase3_scoreboard.py`)

### (1) Biology node trial-reuse — the headline (target: %≥8 → mechanism's 18.7%)

| | nodes | %singleton | %≥4 | **%≥8** | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| baseline (`bio:<sha1>`) | 241 | 74.3 | 11.6 | **4.6** | 2.33 | 53 |
| **B1 (GO-BP)** | 181 | **61.3** | **19.3** | **8.8** | 3.10 | 54 |

**The headline metric moves materially: %≥8 reuse nearly doubles (4.6→8.8%), %singleton
−13pp, %≥4 +66%.** This is the success criterion the task defined ("biology %≥8 reuse
climbs materially"). It does not reach mechanism's 18.7% — expected: top GO groups pool
~5 nodes, and the GO-unmappable physiological tail stays singleton.

### (1b) Biology-incident EDGE reuse — the honest limitation

| edge type | baseline mean / %≥2 | B1 mean / %≥2 |
|---|---:|---:|
| mechanism_affects | 1.78 / 28% | 1.85 / 31% |
| biology_drives | 0.37 / 5% | 0.38 / 5% |
| reflects_biology | 0.60 / 5% | 0.61 / 5% |

### (2) Per-branch edge-class reuse

| class | baseline mean / %≥2 | B1 mean / %≥2 |
|---|---:|---:|
| efficacy | 1.22 / 19% | 1.24 / 20% |
| measurement | 0.34 / 2% | 0.34 / 3% |
| safety | 0.90 / 6% | 0.90 / 6% |

**Node reuse roughly doubles at the top, but edge reuse barely moves.** A biology node
merge only collapses an incident edge when the *other* endpoint (mechanism / indication
/ endpoint) is also shared — and those stay diverse, so `biology_drives` /
`reflects_biology` (biology→indication/endpoint) are essentially flat, while
`mechanism_affects` (mechanism→biology, Reactome-dense source) gains a little. **B1's
node-identity fix is necessary but not sufficient**: converting node reuse into the
*edge* recurrence the predictor consumes requires B2/B3 (hierarchical partial pooling +
context conditioning). This is the architecture-v2 sequencing, confirmed empirically.

### (3) Context-collapse guard (the thing to watch when coarsening)

| | reuse≥2 nodes | mean #indications/node | %pooling mixed succ+fail | mean outcome-entropy |
|---|---:|---:|---:|---:|
| baseline | 62 | 3.69 | 42% | 0.561 |
| B1 (GO-BP) | 70 | 3.87 | 46% | **0.626** |

B1 **mildly** raises within-node outcome heterogeneity (mixed-outcome +4pp, entropy
+0.065). Not a severe spike — the widest-pooling nodes (`DNA-damage-induced apoptosis`:
41 trials / 28 indications) already exist in the baseline content-hash graph; B1 adds
~8 reuse≥2 nodes at similar spread. But the signal is real and in the warned-of
direction: **merging trades some singletons for mixed-context pooling**, so a build
should hold the gate conservative (0.60–0.65) and treat **context-conditioned
hierarchical pooling (B2/B3) as a required follow-up**, not optional — otherwise a
GO-merged node pools reliable-in-one-indication biology with another under one Beta.

### (5) Safety invariance

| | targets (≥3 AE) | within-target `target_associated_ae` E[p] SD |
|---|---:|---:|
| baseline | 45 | 0.0479 |
| B1 (GO-BP) | 45 | **0.0479** |

**Identical** — B1 keys on biology and leaves the target-keyed safety layer untouched
(confirms the P9 baseline 0.048 to the digit). The per-target safety decomposition is
shippable today, independent of B1.

### (4) Trust gate — honest holdout AUROC  _(secondary; null expected)_

Honest 5-fold holdout (re-attribute `--initial` per fold excluding the fold,
`eval_holdout_kfold.py:105`), same discipline both arms, n=221 (147 succ / 74 fail):

| | in-sample | **holdout AUROC** | gap | holdout acc |
|---|---:|---:|---:|---:|
| baseline | 0.797 | **0.567** | +0.230 | 0.661 |
| B1 (GO-BP) | 0.799 | **0.569** | +0.230 | 0.647 |

**Δ AUROC = +0.002 — a clean null.** The baseline reproduces the documented 0.565–0.567
to the digit (faithful control). As predicted: prediction reads the scalar *edge*
beliefs, whose per-branch reuse is essentially unchanged (efficacy 1.22→1.24), so the
binary AUROC does not move. The task states this up front ("binary AUROC may move only
modestly; that's expected and not the criterion"). The conditional-validity gain B1
targets is gated on B2/B3 converting node reuse into edge reuse.

## Verdict

B1's identity lever **works as predicted**: a curated GO-BP id roughly doubles biology
%≥8 node reuse and cuts singletons 13pp, with **no safety disturbance** and only a
**mild** context-collapse cost. But the gain stops at the node level — **edge reuse,
the predictor's actual input, is flat** — so B1 alone will not move the AUROC. The
honest reading matches the strategy doc: **B1 is the necessary substrate fix; B2/B3
(context-conditioned hierarchical pooling) and Pillar C (reuse-dense data) are what
convert it into predictive lift.** Ship B1 behind the flag as the substrate; sequence
B2/B3 next.
