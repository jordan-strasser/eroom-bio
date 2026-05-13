# Round 6 — Clean house: revert the round-3.4 primary-vs-supportive heuristic

Date: 2026-05-13. Done on `main` directly. Round-4-sub-chains archived as `archive/round-4-sub-chains` (pushed to origin).

## Why this round

Round 5's four-layer audit exposed two HIGH-severity bugs that the chain-coverage KPI hid:

- **`_identify_primary_intervention_ids` over-filtered**: in NCT00019682 (Schwartzentruber 2011 — Phase 3 gp100 ± high-dose IL-2 vs IL-2 alone), the entire aldesleukin monotherapy arm produced zero chains; in NCT00003222, aldesleukin and sargramostim were dropped from a 5-compound peptide+IL-2+GM-CSF combo regimen.
- **`predict_clinical_hypothesis` crashes on UNKNOWN-target hypothesis** (NCT00003509 antineoplaston therapy).

When the user reviewed, they identified the underlying issue: round 3.4's primary-vs-supportive distinction was an architectural change introduced before the system was ready for it. Round 4 (parked on a branch) was building further on the same premature distinction. The right move was to roll back the heuristic and converge on a simpler baseline.

## What shipped

### 1. Archived round-4-sub-chains

Renamed `round-4-sub-chains` → `archive/round-4-sub-chains` locally and pushed to origin. Branch tip: `a00519b "Round 4: sequential hypothesis sub-chains"`. Stash `stash@{0}` (round-4 WIP — annotations + playbook edit) preserved untouched. Memory updated at [[project-round4-status]] and a new [[feedback-premature-classification]] entry documents the directional decision.

### 2. Full revert of round 3.4's primary/supportive compound filter

| File | Change |
|---|---|
| `src/graph/populate.py` | Removed `_COMMONLY_SUPPORTIVE_COMPOUNDS` (frozenset of 18 compounds), `_SUPPORTIVE_CONTEXT_PHRASES` (12 phrases), `_identify_primary_intervention_ids` function, call sites in `build_arms` and `seed_responds_differently_from_extractions`, and the now-dead `if primary_ids:` branch in the per-constituent chain rebuild. |
| `src/graph/models.py` | Removed `primary_intervention_ids: list[str]` field from `TrialSubgraph`. Updated docstring to reflect that every constituent compound of every arm fans out a chain; primary-vs-supportive distinction (if added later) belongs at the extractor schema layer, not the populator. |
| `tests/test_populate.py` | Deleted `TestIdentifyPrimaryInterventions` class (8 tests). |

559 tests pass (was 567; the 8 deleted tests were specific to the reverted heuristic).

## KPI movement

| | post3.final | post5 | **post6** |
|---|---|---|---|
| Chains in slice | 27/34 covered (79%) | 34/34 (100%) | **47/48 (98%)** |
| Trials full / partial / zero | 5 / 2 / 3 | 10 / 0 / 0 | **9 / 1 / 0** |
| Zero-coverage trials | 3 | 0 | **0** ✓ (user's stated goal) |
| Chain count (slice) | 34 | 34 | **48** (+41% — un-filtered constituents) |

The "47/48 chains touched" number is slightly below post5's 34/34 only because removing the filter creates more chains than the classifier has emissions for. The single untouched chain (`NCT03618641 / cmp_001`, target=UNKNOWN, TLR9 agonist codename) is a known limitation of the UNKNOWN-target archetype, not a regression from this round.

## Per-trial change summary (slice)

Edges-routed counts measured by `support buckets:` entries in each inspection file.

| Trial | Chains post5 → post6 | Edges routed post5 → post6 | Notes |
|---|---|---|---|
| NCT00003222 | 6 → **11** | 45 → 45 | Added aldesleukin, sargramostim, ifa chains. Edge routing unchanged because classifier emits combo names (see "What did NOT change" below). |
| NCT00003509 | 1 → 1 | 6 → 6 | Single-arm antineoplaston trial — no constituents to un-filter. Still crashes prediction (Finding A from round 5). |
| **NCT00019682** | **1 → 4** | 6 → 6 | **Aldesleukin monotherapy arm now has a chain.** Phase 3 IL-2 evidence is no longer silently dropped at the populator. |
| NCT00072189 | 1 → 1 | 10 → 10 | Single compound. Unchanged. |
| NCT00084656 | 6 → 6 | 24 → 24 | Already 2 compounds × 3 chains each. Unchanged. |
| NCT00109005 | 2 → 2 | 5 → 5 | 2-dose-cohort revlimid monotherapy. Unchanged. |
| NCT01844505 | 8 → 8 | 16 → 16 | Standard set. Unchanged. |
| NCT01950390 | 3 → 3 | 25 → 25 | Standard set. Unchanged. |
| NCT03484923 | 5 → 10 | 45 → 45 | Standard set — added 5 more constituent chains (combo trial). Edge routing unchanged. |
| **NCT03618641** | **1 → 2** | 24 → 24 | **Standard set** — added `cmp_001` chain (TLR9 codename, UNKNOWN target). **Hypothesis-selection now picks cmp_001 → triggers Finding A prediction crash.** Regression in prediction availability, even though chain coverage is fine. |

## What did NOT change (and why)

`Edges routed` is unchanged across the board even on trials with more chains. Mechanically:

- The classifier emits edges keyed on combo regimen names like `aldesleukin+gp100_antigen+incomplete_freund_s_adjuvant+sargramostim+tetanus_peptide_melanoma_vaccine+tyrosinase_peptide → ENSG00000134460`.
- The attributor matches `source_entity` against `chain.compound_id`.
- A new per-constituent `aldesleukin` chain has `compound_id="aldesleukin"`, which does NOT match the combo source name.
- So the new chains exist but the classifier's per-trial edges still route the same way they always did — concentrated on whichever single-compound or combo chain happens to match.

This is the per-constituent decomposition issue (`NEXT_SESSION.md` bucket B approach 1) — orthogonal to round 6's heuristic revert. Solving it requires either: (a) extractor-side schema change emitting per-constituent edges from the LLM, (b) populator-side combo→constituent matching at attribution time, or (c) classifier prompt revision to emit per-constituent edges explicitly.

## Findings (deferred — surfaced or persistent after round 6)

### Finding A (from round 5, persisting; **now affects standard set**)

`predict_clinical_hypothesis` crashes on hypothesis chains with `target_id=UNKNOWN`.

- Round 5: 1 trial affected (NCT00003509).
- Round 6: **2 trials affected — NCT00003509 + NCT03618641** (the standard-set TLR9 codename trial). The revert exposed this because cmp_001 wasn't previously getting its own chain (silently filtered as UNKNOWN-target combo constituent — same root cause that was hiding Finding A on NCT03618641 previously).
- Fix is a ~2-line guard in `predict_clinical_hypothesis`: when traversing, skip nodes with id `UNKNOWN` and start the path at `mechanism` instead.

This is the highest-leverage next patch — small, isolated, restores prediction for the standard-set trial. **Recommend shipping as round 6.1 before any other work.**

### Finding (combo-constituent attribution gap)

Classifier-emitted edges using combo regimen names don't route to per-constituent chains. Edge-routing counts unchanged in this round despite +41% chains. This is the underlying problem that the round-3.4 heuristic was an indirect attempt to mask. Real fix lives at the extractor-schema or attributor-matching layer.

### Carryover

| | Status |
|---|---|
| Reactome top-1 ranking returns disease-irrelevant pathways (round 5 Finding C — CRBN→SARS, CHEK1→GPVI) | Deferred per user direction; separate round. |
| Peptide-vaccine target heuristic (gp100, tyrosinase, MART-1 → curated Ensembl ids) | Was queued for "round 6 bucket B" before clean-house; now deferred. Would reduce UNKNOWN-target chain count (currently 15/48 chains, 5 trials). |
| `automate_node_debug.md` doc update for append-only `unrouted_attribution_updates.jsonl` (round 5 Finding) | Trivial; bundle with any next commit. |
| Subgroup population anchors on parent disease (round 5 Finding D) | Document; no fix needed. |

## Artifacts

- `audit/inspection_*_post6.txt` (10 per-trial + 1 extremes)
- `audit/fixes_round6.md` (this file)
- Code changes (uncommitted):
  - `src/graph/populate.py` (removed ~130 lines)
  - `src/graph/models.py` (removed 1 field + updated docstring)
  - `tests/test_populate.py` (removed 1 test class, 8 tests)
- Branch operation: `round-4-sub-chains` → `archive/round-4-sub-chains` (pushed to origin)
- Memory: [[feedback-premature-classification]] new; [[project-round4-status]] updated.

## Recommended next move

Patch Finding A (`predict_clinical_hypothesis` UNKNOWN-target guard) so the prediction crash on NCT03618641 — a standard-set trial — is resolved. The 2-line guard is independent of any further architecture decisions and is fully reversible.

After that, the next strategic question is whether to invest in fixing the combo-constituent attribution gap (the real reason this round's revert didn't move `edges_routed`) or to expand the corpus first and see if the issue self-resolves at scale.
