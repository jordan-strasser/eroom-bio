# Safety manifold — Phases 2–4 (build + scoreboard)

Both manifolds cleared the **alignment gate** (SAFETY_MANIFOLD_ALIGNMENT.md:
geometry predicts AE-sharing, permutation-significant p = 0.005 / 0.010). This
doc reports what the borrowing actually *buys* when measured honestly. The
headline is split, and the split is the point:

- **The exact-id safety product is intact and shippable** (within-target SD 0.044,
  ≈ the documented 0.048 — manifold work never touches it).
- **The on/off-target decomposition works and validates on known cases** — as an
  *in-sample attribution* surface (the product output per program).
- **Honest, trial-disjoint novel-entity liability transfer does NOT beat the
  base-rate prior at n=472** — for *either* manifold. The signal that appears
  under a naive test is trial co-occurrence leakage. So the *predictive* cross-node
  borrowing does not ship; it is gated on corpus growth, the same reuse wall the
  BioLORD field hit on the efficacy branch.

All numbers on `data/exports/multi_500_annotated.json` (n=472 in-graph trials).
Harness: `scratch/safety_manifold/{geometry,borrow,phase2_reuse,phase3_decompose,
phase3_validate,phase4_novel}.py`. Tuned kernels: compound ECFP4 bw 0.4 / sim_min
0.25; target pathway-Jaccard bw 0.4 / sim_min 0.05.

---

## Phase 2 — effective reuse the kernel manufactures

The kernel adds, to each AE edge's exact-id evidence, a Nadaraya–Watson sum of
manifold-neighbors' evidence (own term dominates; `(α−1, β−1)` strips the prior so
only evidence is borrowed). Per causes_ae / target_associated_ae edge:

| manifold | edges | own %≥8 (exact-id) | +borrow %≥8 | cross-node multiplier (median / mean) | sparse-edge rescue |
|---|---:|---:|---:|---:|---:|
| Compound (ECFP4) | 1810 | 55% | **59%** | 0.00 / 0.09 | 9% of own<8 edges lifted ≥8 |
| Target (pathway) | 301 | **100%** | 100% | 0.00 / 0.06 | n/a (no sparse edges) |

(At a wider compound kernel — bw 0.5 / sim_min 0.20 — the lift rises to 61% and
sparse-rescue to 13%; the locked default keeps borrowing concentrated on
genuinely-similar pairs. Either way the multiplier is ≪ 1× because exact-id is dense.)

**The decisive finding: the safety branch is NOT reuse-starved.** Unlike efficacy
(1.24 trials/edge, where the field's multiplier was 0.17×), the AE branch is
already dense on exact-id — 55% of compound AE edges and **100%** of
target_associated_ae edges already clear effective reuse ≥ 8. (target_associated_ae
is a propagation aggregate of ≥2 compounds × n_eff 4, so it is ≥8 by construction —
this is *why* on-target SD is 0.048.) Consequences:

- The cross-node **multiplier is < 1× on observed entities**, not because the kernel
  is weak but because the *denominator* (own evidence) is already large. The prompt's
  ">1× multiplier" bar was written by analogy to the starved efficacy field; on
  safety it is unreachable on observed entities, and reaching it would actually be a
  red flag (borrowing drowning out an entity's own well-populated data).
- Where the kernel *does* add: the compound manifold lifts per-edge reuse **55% →
  61%** and rescues **13%** of starved (own<8) edges over the bar. This is the
  opposite of the field, which *subtracted* reuse (6.7% → 4.6%). Aligned geometry
  borrows; misaligned geometry leaks.
- The only place borrowing could be decisive is **entities with no own evidence**
  (novel/sparse) — which is exactly the Phase-4 test, and where it fails honestly
  (below).

---

## Phase 3 — on/off-target decomposition (the differentiator)

For each observed `(compound, target, AE)` we attribute the AE to **on-target**
(recurs across structurally DIVERSE compounds sharing the target), **off-target**
(recurs across STRUCTURE-neighbors hitting different targets), **idiosyncratic**, or
**baseline** (ubiquitous ≥30%-prevalence tox: blood, GI). Two fixes made it work:
SOC roll-up (siblings report disjoint PT terms) and background-correction against the
global SOC base rate (so only class-SPECIFIC enrichment routes on/off-target).

707 triples over 129 profiled compounds:

| tag | share | reading |
|---|---:|---|
| baseline | 43% | ubiquitous oncology tox (myelosuppression, GI) — not differentially attributable |
| idiosyncratic | 41% | compound-specific (no class explains it) |
| **on-target** | 14% | mechanism-intrinsic class effect, diverse scaffolds |
| off-target | 1% | chemistry-specific, scaffold-escapable |
| on-target(low-div) | 1% | recurs on target but carriers structurally near-identical |

### Known-case validation

| case | expected | routed | verdict |
|---|---|---|---|
| **EGFR → rash** | on-target | on-target, on_lift +0.59, carriers {gefitinib, lapatinib, cetuximab}, diversity 0.71 | **PASS** (canonical; mAb + small-molecules sharing only the target) |
| **INSR → hypoglycemia** | on-target | on-target(low-div), on_lift +0.79 | **PASS** (textbook; carriers are insulin analogs → honestly flagged low-diversity) |
| HMGCR → musculoskeletal | on-target | no qualifying AE | corpus caught fractures, not myopathy |
| TUBB → neuropathy | on-target | idiosyncratic | sparsity: no profiled tubulin sibling carries a nervous-system AE ≥0.55 |
| TNF → infections | on-target | idiosyncratic | sparsity: no profiled TNF sibling carries an infections AE ≥0.55 |

The decomposition routes the on-target class effects that are populated; the misses
are all data sparsity (the sibling lacks the matching SOC AE), not method error.
**Off-target is genuinely sparse (8 cases)** because in an oncology corpus structure
and target are correlated (analogs hit the same target) and the canonical off-target
liabilities — hERG/QT, anthracycline cardiotoxicity — are *absent from the annotated
AEs* (anthracyclines cluster at Tanimoto 0.84 but carry no cardiac AE in this data).

### Product output — per-program AE liability profile (verbatim)

```
gefitinib (EGFR)
  diarrhea     P=0.68  on-target     [on 100%/off 0%/idio 0%]  shared w/ 3 EGFR compounds (div 0.66)
  rash         P=0.66  on-target     [on  78%/off 0%/idio 22%]  shared w/ 2 EGFR compounds (div 0.71)

carboplatin (DNA)
  fatigue      P=0.31  on-target     [on  80%/off 0%/idio 20%]  shared w/ 4 DNA compounds (div 0.92)
  diarrhea     P=0.23  on-target     [on  74%/off 0%/idio 26%]  shared w/ 4 DNA compounds (div 0.87)
  allergic_reaction P=0.51 idiosyncratic [idio 100%]            ← platinum-specific, NOT class
  neutropenia  P=0.25  baseline                                 ← ubiquitous cytotoxic tox

Insulin glargine (INSR)
  hypoglycaemia P=0.79 on-target(low-div) [on 100%]            shared w/ 2 INSR analogs
```

This is the surface a customer pays for: each observed AE tagged on-target /
off-target / idiosyncratic / baseline, with calibrated probability and provenance
(which neighbor compounds contributed, and whether they are structurally diverse).
It is an **attribution over observed data**, not a novel-entity prediction.

---

## Phase 4 — honest scoreboard

### Novel-entity test (the value prop) — and why it fails honestly

Hold an entity out entirely; predict which AE SOCs it shows from manifold-neighbors
alone. **Leakage discipline is decisive here** and exposed two traps:

1. **Base-rate trap.** A pooled cross-SOC AUROC looks great (compound 0.65, target
   0.79) but is degenerate: the base-rate prior *also* scores 0.83 because common
   SOCs (blood 0.36) are simply more often present than rare ones (skin 0.08). The
   prior predicts the *same* profile for every entity. The honest metric is **per-SOC
   AUROC** (within a SOC, rank entities), where the constant prior scores exactly 0.5.

2. **Trial co-occurrence leakage.** Combo arms attribute one trial's AEs to multiple
   compounds, so a held-out entity and a neighbor sharing a trial have correlated
   profiles through that trial — not through homology. Enforcing **trial-disjoint**
   neighbors removes it.

| manifold | per-SOC AUROC, leaky | per-SOC AUROC, **trial-disjoint (honest)** | served | verdict |
|---|---:|---:|---:|---|
| Compound (structure) | 0.53 | **0.44** | 25/69 (36%) | no honest transfer |
| Target (pathway) | 0.59 | **0.47** | 21/37 (57%) | no honest transfer |

Both honest numbers sit **at or below 0.5**. The gap between leaky and honest *is*
the leakage. A per-SOC breakdown of the honest target test is sampling noise (1–13
positives per SOC at n=21; the on-target classes skin/endocrine are themselves below
0.5). **Conclusion: at n=472 neither manifold delivers leakage-free novel-entity
AE-liability transfer that beats the base-rate prior.** This is the same
memorize-no-transfer wall the project has hit repeatedly — reuse-per-edge, not n, is
the binding constraint, now confirmed on the safety branch.

### Invariance (exact-id intact)

Within-target target_associated_ae posterior SD = **0.044** (n=39 targets) —
preserves the documented 0.048. The manifold layer is purely additive behind a flag;
the working exact-id product is unchanged.

### Decomposition quality

On the known cases that are populated, the decomposition routes correctly (EGFR→rash,
INSR→hypoglycemia). This is an in-sample attribution result; it does not depend on the
(failed) novel-entity transfer, because it explains *observed* liabilities rather than
predicting unobserved ones.

---

## Verdict & what ships

| component | honest status | decision |
|---|---|---|
| Exact-id safety (causes_ae + target_associated_ae, SD 0.044) | works, invariant | **ship as-is** |
| On/off-target **decomposition** | validates on known cases, in-sample attribution | **ship as an attribution surface** (labeled in-sample, not a novel-entity predictor) |
| Phase-1 alignment | real, permutation-significant | true as a population property |
| Phase-2 reuse multiplier | <1× on observed (safety isn't starved) | informative, not a lever here |
| **Predictive cross-node borrowing** (novel-entity) | **fails honest trial-disjoint test (≤0.5)** | **do NOT enable**; hold behind `EROOM_SAFETY_MANIFOLD` (default OFF), re-test when corpus grows |

Per the rules of engagement — "if a manifold fails, drop that borrowing and ship the
exact-id product; downside is bounded" — the predictive borrowing is dropped and the
exact-id product (plus the validated decomposition) ships. The downside *was* bounded:
the flag is off, exact-id is untouched, and we now have an honest measurement instead
of a leaky number. **The borrowing the text field couldn't do, domain geometry also
can't do *yet* — not because the geometry is misaligned (it isn't; the alignment gate
passed), but because at n=472 the corpus has too few trial-disjoint neighbors per
entity to transfer.** The lever is the same as everywhere else in this project:
reuse — here, more trials on shared pathways/scaffolds (Pillar C, safety-enriched
pull), at which point this exact harness re-runs and the gate becomes a green light.
