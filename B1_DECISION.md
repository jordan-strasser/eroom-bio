# B1 decision — re-canonicalize biology onto a controlled vocabulary?

**Frame (Option 2).** The deliverable is per-branch risk decomposition + per-target
safety transfer, not binary-success AUROC. B1's job is to raise *effective reuse* so
the singleton-dominated biology layer (71% singletons, 1.24 trials/edge) stops
reverting to prior. This decision supersedes the prior structural-bet version with
**measured** Phase 0 (descriptor audit) + Phase 1 (collapse gate) numbers.

All numbers on `data/exports/multi_500_annotated.json` (n=472, 212 BiologyNodes).
Probes: `scratch/diagnostics/b1_descriptor_audit.py`, `b1_phase1_cosine.py`,
`b1_phase1_govocab.py`, `b1_phase1_reuse_preview.py`. Reuse control reproduces
`MERGE_POOLING_MAP` P1 exactly (median 1, 70.8% singleton, 5.2% ≥8).

> **Headline: GO — proceed to Phase 2.** The "wrong-string" descriptor artifact is
> refuted (Phase 0). A curated GO-biological-process id collapses **~40% of singleton
> biology into biologically-coherent groups** and **roughly doubles the headline
> scoreboard metric (%≥8 reuse 5.2% → ~10%)**, well clear of the STOP condition. The
> win is *real but moderate*, not transformative: reuse rises ~2× but stays below
> mechanism's 18.7% and below the synth's reuse-8 recovery bar, so **Pillar C
> (reuse-dense data) remains the binding dependency** for the AUROC to move. Build
> GO-BP-primary with a content-hash fallback for the ~⅓ physiological-outcome tail,
> behind `EROOM_BIO_ONTOLOGY`, and watch the context-collapse guard.

---

## Phase 0 — descriptor provenance audit (full writeup: `B1_DESCRIPTOR_AUDIT.md`)

The four consumers — id hash (`populate.py:650`), SapBERT `name_id`
(`node_merge.py:118`), BioLORD Tier-3 `_node_text` (`node_merge.py:134,293`), the
(s,t) field (`field_prediction.build_st_desc_map:82`) — **all compare the same
normalized phrase**: `id == sha1(norm(description))` 212/212, `norm(description) ==
norm(name)` 212/212. So Tier-3 is comparing the correct, consistent claim.

**The "compare the right string / re-normalize" fix is ruled out** — it would change
nothing. The descriptors are merely *impoverished*: 4–5 word functional-outcome
phrases (`blood pressure reduction`, `mitotic arrest`), often compound and
direction-laden. That is a genuine-semantics problem, so Phase 1 (collapsibility) is
the correct gate. Format/session drift is mild (no markdown, no version metadata,
84% lowercase) and is normalized away — not a confounder.

## Phase 1A — raw BioLORD cosine among singletons (`b1_phase1_cosine.py`)

Nearest-other-node cosine for the 150 singletons: **66% have a neighbor ≥0.70, 46%
≥0.75, 20% ≥0.80, 0% ≥0.85** (0.85 = the live Tier-3 bar, so 0% by construction).
The band is **contaminated**: distinct-mechanism pairs sit *above* true paraphrases
— `HER2-mediated growth inhibition ↔ EGFR-mediated growth inhibition` at **0.845**
(distinct receptors) outranks `anti-inflammatory immune modulation ↔ suppression of
immune-mediated inflammation` at 0.843. **No single cosine bar cleanly separates
collapse from context-collapse** — this is the synth's warning, and it rules out
"just lower the Tier-3 threshold" / embedding-cluster ids as the fix.

## Phase 1B — curated GO-BP collapse (`b1_phase1_govocab.py`, `_reuse_preview.py`)

Mapping each biology description to its nearest term in a corpus-relevant GO-BP
vocabulary (2367 terms from the local QuickGO gene cache) by BioLORD cosine
— direction-robust and network-free, after OLS lexical search proved too brittle
(SSL timeouts, drug/CHEBI false hits). Eyeball-verified correct in the ≥0.65 band.

| metric | gate 0.55 | gate 0.60 | gate 0.65 |
|---|---:|---:|---:|
| **coverage** (nodes mapped) | 95% | 82% | 66% |
| **singleton collapse** (share a GO term) | **47%** | **39%** | 28% |
| merge-group quality | looser | balanced | clean |

The merges are biologically right: `mitotic arrest`+`cell-cycle arrest` → *negative
regulation of cell cycle*; `amyloid plaque clearance`+`amyloid beta peptide
reduction` → *amyloid-beta clearance*; `B-cell depletion`+`B-cell survival
suppression` → *negative regulation of B cell proliferation*.

**Headline scoreboard preview — biology TRIAL-reuse re-keyed onto GO-BP:**

| | nodes | %singleton | %≥4 | **%≥8** | max |
|---|---:|---:|---:|---:|---:|
| current (`bio:<sha1>`) | 212 | 70.8 | 13.2 | **5.2** | 53 |
| GO-re-keyed @0.55 | 153 | 52.3 | 22.9 | **10.5** | 54 |
| GO-re-keyed @0.60 | 161 | 56.5 | 21.7 | **9.9** | 54 |
| GO-re-keyed @0.65 | 174 | 62.1 | 20.1 | **8.6** | 53 |

(mechanism, the dense reference, is 18.7% ≥8.)

## The decision gate

| gate branch (task rule) | met? |
|---|---|
| **High collapse (>~40% of singletons merge) → B1 pays off, build** | **YES** (47%@0.55, 39%@0.60) |
| Descriptor artifact → re-test normalized first | **NO** — refuted in Phase 0 |
| Low collapse (<15%) AND low coverage → STOP, lever is Pillar C | **NO** — coverage 66–82%, collapse ≫15% |

→ **GO. Proceed to Phase 2.** B1's identity lever is real: a curated GO-BP id
collapses ~40% of singletons coherently and roughly doubles %≥8 reuse. This is the
first branch of the gate, not the STOP branch.

## What it will and won't buy (state up front)

- **Will:** %singleton 71→~53–56%, %≥8 reuse 5.2→~10% (≈2×), %≥4 13→~22%; cleaner
  per-branch evidence on the mechanism-process subset (apoptosis, cell-cycle,
  angiogenesis, immune, amyloid).
- **Won't:** clear the synth's reuse-8 recovery bar for most biology (top GO groups
  pool ~5 trials, not ≥8), so binary AUROC should move only modestly — *expected and
  not the criterion*. The cardio/metabolic/neuro **physiological-outcome tail (~⅓ of
  biology) does not map to GO-BP** (`blood pressure reduction`, `glycemic control`)
  and stays singleton under a content-hash fallback — those need EFO/MONDO/physiology
  axes (a later move), not GO.

## Recommendation for the build

1. **Mint biology ids from the nearest GO-BP term** (the Reactome/mechanism pattern),
   at cosine gate **0.60–0.65** (clean merges; re-check the P3 heterogeneity guard and
   back off toward 0.65 if within-node outcome heterogeneity spikes — the synth's
   low-alignment collapse regime). **Content-hash fallback** (`bio:<sha1>`) for
   sub-gate biology so no node is lost.
2. **Gate behind `EROOM_BIO_ONTOLOGY` (default OFF)**; re-merge + re-attribute from the
   same `multi_500_initial.json` so the only difference is the flag (clean A/B).
3. **Keep the BioLORD field as a soft tail-prior only** — Step-1 LOO showed it neither
   beats the scalar (0.561 vs 0.565) nor lifts reuse (1.10×); it is within-edge and
   cannot cross-node-share (`B1_field_loo_findings.md`). Never the lever.
4. **Pillar C is the gating dependency, not optional.** B1 lifts reuse ~2× but the
   substrate stays thin at n≈500; only reuse-dense data on shared targets/mechanisms/
   indications pushes biology past the reuse-8 regime where recovery becomes real and
   where the ontology's granularity can be validated rather than chosen by eyeball.

## Ship-now, B1-independent
Per-target safety transfer (within-target `target_associated_ae` posterior SD
**0.048**) keys on the already-merged target id and is invariant to B1
(`B1_field_loo_findings.md`) — shippable today as the "which risk dominates"
decomposition while the biology backbone is rebuilt.
