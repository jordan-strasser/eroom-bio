# Cross-round issue checklist

Persistent tracker for issues surfaced by the iterative audit cycle. Updated as items close. New items get appended; closed items stay in place with their closing reference so the history is preserved.

**Convention**: `[x]` closed in commit / round, `[ ]` open, `[~]` in progress this round.

---

## Tier 1 — Top priority (concrete, actionable, surfaced-and-not-addressed)

- [x] **1. Full corpus re-classification.** 62 of 145 trials had zero chain coverage because round-2 prompt fix was masked by cached pre-fix classifications. _Closed: round 3 build (full re-classify at n=100). Coverage 45% → 80%._
- [~] **2. NCT00509496 partial coverage outlier** (2/16 chains touched even after re-classify). Trial-level inspection needed. _Partially addressed in round 3.1 (Pattern B prompt extension closed the slug-mismatch part; trial is now at 2/16 with 3 routed edges instead of the prior misroute). Remaining 14/16 silent chains may be genuine missing data, not a bug — folded into checklist #14 for triage._
- [x] **3. Endpoint slug source-side leak.** Classifier emits non-canonical endpoint sources (`CR_cancer`, `ORR_refractory_melanoma`, `ORR_unresected_stage_iiib_to_ivm1c_melanoma`) even with round-2 prompt fix. _Closed: round 3.1. Classifier prompt now applies the canonical-id rule symmetrically to source AND target. Pattern B trials NCT02302339, NCT01248936, NCT02366195 went from 0 → full/partial coverage._
- [x] **4. Untyped "unknown" nodes from non-drug interventions** (17 nodes: diagnostics, radiation, devices + 4 real drugs slipping through the filter). _Closed: round 3.1. `build_arms` now drops intervention names explicitly typed as non-drug in `trial.interventions`. After fix: 0 unknown-type nodes in the snapshot. New side-effect tracked as checklist #13: 4 cell-therapy/radiation/device-only trials lose all chains._
- [x] **5. IndicationNode near-duplicates** (e.g. brain_metastases vs brain_metastasis). One-time canonicalization cache sweep. _Closed: round 3.1. `slugify_disease_name` now normalizes plural disease nouns (tumors→tumor, metastases→metastasis, cancers→cancer, etc.) and resolves a known-aliases dict. Cache swept; 5 entries rewritten. IndicationNode count went 39 → 37._

## Tier 2 — Dev log noise (already covered by prompt rules but pre-fix caches don't)

- [ ] **6. "Final analysis" / "Primary completion" still in extraction subgroups.** Prompt rule rejects them but cached pre-fix extractions still emit them. Same self-heal pattern as MedDRA cache (commit `1cc06ea`) would close this.
- [ ] **7. HAHA (human anti-human antibody) positive/negative not in vocab.** Real immunogenicity stratifier for biologics; currently axis="other" → unmapped log.
- [ ] **8. Likert change-from-baseline labels** ("missing", "better", "worse", "no change", 4× each in today's log). Either map to a new axis or filter as non-stratifying.

## Tier 3 — Scaling readiness (not urgent for melanoma-only; required before multi-indication)

- [ ] **9. Subtype-collapse consistency.** `Cutaneous Melanoma → melanoma` (modifier stripped) vs `Uveal Melanoma → uveal_melanoma` (subtype preserved). Needs a consistent rule before adding lung/breast/etc.
- [ ] **10. LLM canonicalization prompt smoke test on non-oncology.** Docstring claims it's domain-agnostic; never verified against immunology / infectious / neurology trials.
- [ ] **11. No explicit disease hierarchy.** `uveal_melanoma` and `melanoma` are sibling nodes today. Will want `subtype_of` edges or MeSH/EFO-backed hierarchy once corpus crosses ~3 indications.

## Tier 4 — Emergent (surfaced by round 3 re-classify; not in original 11)

- [x] **12. Failure trials with `confidence_overall <0.5` emit zero edges_to_update.** 9 of 12 zero-coverage trials. The confidence rubric and the failure-trial-MUST-emit rule conflict; LLM resolves it the wrong way. _Closed: round 3.1. Prompt strengthened to mark the failure-emit rule as overriding the confidence tier; defensive backstop in `Attributor` auto-emits `biology_drives weak_contradict` on the parent chain when a failure trial returns zero edges. After fix: 0 zero-coverage trials._

## Tier 6 — Emergent (surfaced by round 3.3 verification)

- [x] **15. Reactome biology-pathway fan-out (`_BIOLOGY_PATHWAY_CAP=3`) multiplies chain counts ~3×.** _Closed: round 3.4. Dropped cap to 1; full Reactome pathway list preserved as `pathway_ids` metadata on the BiologyNode for cross-pathway query later. NCT01844505: 24 → 8 chains. Corpus-wide chains 829 → 363 (−56%); BiologyNodes 149 → 107. Re-splitting per pathway is round 4.0 work (when pathway-level biomarker data starts to discriminate)._
- [x] **16. Primary heuristic misses regimens where the hypothesis text names supportive drugs.** _Closed: round 3.4. Added `_COMMONLY_SUPPORTIVE_COMPOUNDS` allowlist (lymphodepletion chemo, cytokine support, common adjuvants) + `_SUPPORTIVE_CONTEXT_PHRASES` ("with lymphodepletion", "with cytokine support", "preconditioning", etc.) — both demote candidates only when at least one non-supportive primary survives, so cyclophosphamide-as-monotherapy trials aren't silenced. NCT00670748: 32 → 4 chains, 4/4 covered. Full extractor-side `primary_compounds` / `supportive_compounds` schema change deferred to round 4.0._

## Tier 5 — Emergent (surfaced by round 3.1 verification)

- [~] **13. Non-drug therapeutic trials get filtered to 0 chains.** _Partially closed: round 3.3. CompoundNode renamed to InterventionNode with `intervention_type` enum mirroring CT.gov (DRUG / BIOLOGICAL / RADIATION / DEVICE / PROCEDURE / DIAGNOSTIC_TEST / BEHAVIORAL / COMBINATION / OTHER / UNKNOWN). `binds_to` renamed to `affects` so the edge no longer presumes drug-binding chemistry. The 4 trials (NCT00587964 radiation, NCT01350401 cell therapy, NCT01473004 device, NCT00472459 PDT) still have no chains because non-drug interventions don't get chain backbones yet — proper edge semantics for radiation/cell-therapy/device chain backbones is round 3.4 work._
- [x] **14. Partial-coverage trials need per-trial classification (correct silence vs missed emission).** _Closed: round 3.3. Diagnosed: 66-chain trials weren't over-forking endpoints/subgroups but were per-constituent-decomposing every drug in the arm, including supportive infrastructure (lymphodepletion chemo, growth-factor support). Fix: `primary_intervention_ids` on TrialSubgraph; chain fan-out filtered to compounds named in `therapeutic_hypothesis.compound`. Verification: NCT01218867 went 66 chains / 0 touched → 11 chains / 11 touched. Corpus chain count dropped 972 → 829 (−15%), coverage 81% → 89% (+8pp)._

---

## Round → checklist crosswalk

| Round | Items closed by this round |
|---|---|
| 1 (pre-fixes baseline) | — |
| 2 (`fixes_round2.md`) | — (prompt fixes shipped but masked by cache until round 3.0 re-classify) |
| 3.0 (full re-classify) | #1 |
| 3.1 (5 round-3 priorities) | **#3, #4, #5, #12** closed; #2 partial (folded into #14) |
| 3.2 (planned: dev-log cleanup) | #6, #7, #8 |
| 3.3 (Intervention rename + primary filter) | **#14** closed; **#13** partial (4 non-drug trials still chainless until non-drug edge semantics ship) |
| 3.4 (pathway-cap collapse + primary heuristic upgrade) | **#15, #16** closed |
| 4.0 (planned: sequential hypothesis chains — `audit/round_4_design.md`) | architecture pass; addresses #13 fully + makes supportive-chain learning first-class |
| Future (multi-indication scaling) | #9, #10, #11 |
