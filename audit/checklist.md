# Cross-round issue checklist

Persistent tracker for issues surfaced by the iterative audit cycle. Updated as items close. New items get appended; closed items stay in place with their closing reference so the history is preserved.

**Convention**: `[x]` closed in commit / round, `[ ]` open, `[~]` in progress this round.

---

## Tier 1 — Top priority (concrete, actionable, surfaced-and-not-addressed)

- [x] **1. Full corpus re-classification.** 62 of 145 trials had zero chain coverage because round-2 prompt fix was masked by cached pre-fix classifications. _Closed: round 3 build (full re-classify at n=100). Coverage 45% → 80%._
- [ ] **2. NCT00509496 partial coverage outlier** (2/16 chains touched even after re-classify). Trial-level inspection needed. _Pattern in `audit/fixes_round3.md` #3._
- [ ] **3. Endpoint slug source-side leak.** Classifier emits non-canonical endpoint sources (`CR_cancer`, `ORR_refractory_melanoma`, `ORR_unresected_stage_iiib_to_ivm1c_melanoma`) even with round-2 prompt fix. _Now part of round 3 Pattern B; `audit/fixes_round3.md` #2._
- [ ] **4. Untyped "unknown" nodes from non-drug interventions** (17 nodes: diagnostics, radiation, devices + 4 real drugs slipping through the filter — calcitriol, polyiclc, resiquimod, ifa). _`audit/fixes_round3.md` #4._
- [ ] **5. IndicationNode near-duplicates** (e.g. brain_metastases vs brain_metastasis). One-time canonicalization cache sweep. _`audit/fixes_round3.md` #5._

## Tier 2 — Dev log noise (already covered by prompt rules but pre-fix caches don't)

- [ ] **6. "Final analysis" / "Primary completion" still in extraction subgroups.** Prompt rule rejects them but cached pre-fix extractions still emit them. Same self-heal pattern as MedDRA cache (commit `1cc06ea`) would close this.
- [ ] **7. HAHA (human anti-human antibody) positive/negative not in vocab.** Real immunogenicity stratifier for biologics; currently axis="other" → unmapped log.
- [ ] **8. Likert change-from-baseline labels** ("missing", "better", "worse", "no change", 4× each in today's log). Either map to a new axis or filter as non-stratifying.

## Tier 3 — Scaling readiness (not urgent for melanoma-only; required before multi-indication)

- [ ] **9. Subtype-collapse consistency.** `Cutaneous Melanoma → melanoma` (modifier stripped) vs `Uveal Melanoma → uveal_melanoma` (subtype preserved). Needs a consistent rule before adding lung/breast/etc.
- [ ] **10. LLM canonicalization prompt smoke test on non-oncology.** Docstring claims it's domain-agnostic; never verified against immunology / infectious / neurology trials.
- [ ] **11. No explicit disease hierarchy.** `uveal_melanoma` and `melanoma` are sibling nodes today. Will want `subtype_of` edges or MeSH/EFO-backed hierarchy once corpus crosses ~3 indications.

## Tier 4 — Emergent (surfaced by round 3 re-classify; not in original 11)

- [ ] **12. Failure trials with `confidence_overall <0.5` emit zero edges_to_update.** 9 of 12 zero-coverage trials. The confidence rubric and the failure-trial-MUST-emit rule conflict; LLM resolves it the wrong way. _`audit/fixes_round3.md` #1._

---

## Round → checklist crosswalk

| Round | Items closed by this round |
|---|---|
| 1 (pre-fixes baseline) | — |
| 2 (`fixes_round2.md`) | — (prompt fixes shipped but masked by cache until round 3.0 re-classify) |
| 3.0 (full re-classify) | #1 |
| 3.1 (planned: 5 round-3 priorities) | #2, #3, #4, #5, #12 |
| 3.2 (planned: dev-log cleanup) | #6, #7, #8 |
| Future (multi-indication scaling) | #9, #10, #11 |
