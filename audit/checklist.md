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

## Tier 5 — Emergent (surfaced by round 3.1 verification)

- [ ] **13. Non-drug therapeutic trials get filtered to 0 chains.** Fix #4 (round 3.1) correctly drops procedures/diagnostics/devices from arm intervention names — but it also drops trials whose only "interventions" are radiation therapy (NCT00587964), engineered cell therapy (NCT01350401), radioactive devices (NCT01473004), or topical PDT (NCT00472459). 4 trials at n=100. These are real therapeutics CT.gov just doesn't type as DRUG/BIOLOGICAL. Either extend `DRUG_LIKE_INTERVENTION_TYPES` to include RADIATION + select OTHER cases, or introduce a typed `NonDrugTherapeuticNode` so they participate in chains without polluting the CompoundNode namespace.
- [ ] **14. Partial-coverage trials need per-trial classification (correct silence vs missed emission).** 17 trials have chains but only some are touched. Many are correct (trial has 66 chains across arms × subgroups × endpoints but only reports overall data — most chains rightly stay silent). Some are likely classifier-emission misses (NCT00509496 at 2/16 is the suspicious end). Per-trial inspection needed to triage; consider adding a "chains-with-data" denominator separate from "chains-built" to the coverage metric so we measure against what the trial actually reported.

---

## Round → checklist crosswalk

| Round | Items closed by this round |
|---|---|
| 1 (pre-fixes baseline) | — |
| 2 (`fixes_round2.md`) | — (prompt fixes shipped but masked by cache until round 3.0 re-classify) |
| 3.0 (full re-classify) | #1 |
| 3.1 (5 round-3 priorities) | **#3, #4, #5, #12** closed; #2 partial (folded into #14) |
| 3.2 (planned: dev-log cleanup) | #6, #7, #8 |
| 3.3 (planned: round-3.1 emergent) | #13, #14 |
| Future (multi-indication scaling) | #9, #10, #11 |
