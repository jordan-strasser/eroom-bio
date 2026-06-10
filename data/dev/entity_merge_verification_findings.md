# TASK 5 — entity-merge SapBERT verification (findings)

**Date:** 2026-06-10 · branch `fix/st-field-faithfulness`
**Tool:** `scripts/instrument_entity_merge.py` (new) — SapBERT name-cosine
under/over-merge checker for ENTITY nodes, the counterpart to
`instrument_mechanism_merge.py` (BioLORD description nodes).
**Graph:** `data/exports/phaseb_n50b_annotated.json` (n=50, the 4a+AE merge config).
**Model/threshold:** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, the same the
merge uses; under-flag ≥0.80 (the merge threshold), over-flag <0.50.

This is the gate that answers the owner's question — *"is SapBERT working for
indication / population / endpoint / AE?"* — with measured rates instead of trust.

## Summary

| type | nodes | under-flags | over-flags | verdict |
|---|---|---|---|---|
| Indication | 48 | 0 | 0 | ✅ SapBERT working + safe |
| Population | 32 | 0 | 0 | ✅ no defects (weak check — compositional names) |
| Endpoint | 47 | 0 | 2* | ✅ SapBERT clean (*2 = endpoint id-slug collapse, not SapBERT) |
| **AdverseEvent** | 173 | 25† | 0 | ⚠️ **tier INERT + re-merge footgun → REMOVED from SapBERT tier** |
| Compound/Intervention | 86 | 11* | 27* | ✅ flags are curated ChEMBL aliases / regimen strings, not SapBERT |
| Target | 54 | 13† | 0 | ✅ id-merge correct — the 13 are paralog siblings SapBERT WOULD fuse |

## Verdicts

### Indication / Endpoint / Population — VERIFIED working + safe (trust at 0.80)
These nodes are created during **populate** (before the merge), so they DO pass
through the SapBERT tier. Result: **0 synonym misses** across 48+47+32 nodes, and
the known sibling-disease false-positive stays below threshold (breast vs ovarian
≈ 0.65 < 0.80), so true synonyms ("NSCLC"≡"non-small-cell lung cancer" 0.81) merge
while distinct diseases don't. The entity-linker is doing exactly its job. The 2
Endpoint "over" flags are NOT SapBERT — they're endpoint-id slug collapses lumping
safety/biomarker descriptors ("OL; Number of Participants…" ← safety/biomarker
slugs, min-cos 0.35); a separate endpoint-id-construction follow-up, logged.

### AdverseEvent — the headline: tier was INERT and a latent over-merge footgun → FIXED
Two independent confirmations the AdverseEventNode SapBERT entry (added `b88b02e`)
should NOT be there:

1. **Inert on fresh builds.** AE nodes are created in `attributor._main`
   (`add_node(AdverseEventNode(...))`, step 4 = attribution), which runs AFTER the
   populate+merge pass (step 2). The merge's SapBERT AE tier never sees them. Proof:
   `Cardiac disorders` and `Cardiac disorder` sit as SEPARATE nodes at cosine
   **0.996** — impossible if a 0.80 SapBERT tier had processed them. (They are in
   fact a SOC `AE:soc:cardiac_disorders` vs PT `AE:cardiac_disorder` — correctly
   distinct MedDRA levels; the 0.996 is my checker's false-positive, harmless.)
2. **Active harm on re-merge.** AdverseEventNode IS in `MergeConfig.node_types`, so
   any re-merge over a graph that already HAS AE nodes (incremental `--add-trials`,
   `assemble_v2 --merge`) WOULD apply SapBERT 0.80 to them. SapBERT name-cosine
   cannot separate true AE synonyms from clinically-distinct siblings at any single
   threshold — the under-flags split into both:
   - SHOULD merge: `Cardiac ischaemia`~`Myocardial ischaemia` **0.944**.
   - MUST NOT merge: `Alanine aminotransferase increased`~`Aspartate aminotransferase
     increased` **0.901** (ALT vs AST — different labs); `Lymphopenia`~`Leukopenia`
     **0.891** (different cytopenias); `Transaminase increased`~`ALT increased` 0.896.

   A 0.80 (or even 0.88) threshold fuses ALT with AST. **SapBERT is the wrong tool
   for AE consolidation; MedDRA's curated synonymy is the right one.**

**FIX (shipped):** removed `AdverseEventNode` from `sapbert_node_types` in
`populate_bottomup` (+ updated `test_bottomup_mergeconfig_merge_tiers_by_node_semantics`).
Safe — inert on fresh builds, removes the re-merge footgun. **Follow-up (separate,
larger):** the genuine AE synonym misses (Cardiac/Myocardial ischaemia) want a
MedDRA synonym-normalization table in `ae_node_id`, not geometric name-merge.

### Target — id-merge VALIDATED (the "under" flags prove SapBERT would be WRONG here)
Targets merge by Ensembl id, NOT SapBERT. The 13 high-cosine pairs are paralogous
genes with near-identical names that MUST stay separate and do:
`FLT1`/`FLT3`/`FLT4` ("fms related receptor tyrosine kinase…" 0.91–0.95),
`TOP2A`/`TOP2B` ("DNA topoisomerase II alpha/beta" 0.909), `CSF1R`/`CSF3R`
("colony stimulating factor 1/3 receptor" 0.901), tubulin isotypes 0.89–0.93. If
Target were a SapBERT tier these would all over-merge — so id-merge is the correct
design, now with evidence.

### Compound / Intervention — flags are expected curated aliases, not SapBERT errors
The 27 Compound "over" flags are ChEMBL brand/code aliases with naturally-low cosine
to the generic (`Losartan` ← `Angizaar`[brand], `Dup-89`[code]; `Mesna` ← `Mesnex`).
That's curated alias breadth, not a geometric over-merge. The 11 Intervention "under"
flags are regimen-string variants on the RAW intervention layer ("carboplatin" vs
"Combination carboplatin"), not the canonical CompoundNode tier. **Minor follow-up:**
`Allisartan` (a distinct ARB) appears in `Losartan`'s alias list (cosine 0.21) — a
possible ChEMBL alias-pull bleed worth a spot-check.

## Files
- `scripts/instrument_entity_merge.py` — the checker (read-only; runs on any 4a+AE build).
- `src/graph/populate_bottomup.py` — removed AdverseEventNode from `sapbert_node_types`.
- `tests/test_populate.py` — merge-tier test now asserts AE is EXCLUDED + why.
