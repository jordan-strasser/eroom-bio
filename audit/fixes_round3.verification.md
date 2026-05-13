# Round-3 verification — post-wrap audit on main

**Date**: 2026-05-13
**Branch**: `main` (HEAD `7ced85d`, tip of post-round-3 wrap `a6f4411` plus today's tooling + parent-anchoring fixes)
**Build**: `--corpus melanoma_145 --max-trials 10 --include NCT01844505,NCT01950390,NCT03484923,NCT03618641 --keep-annotations`

**Headline KPI progression** (same 10-trial slice across runs):

| Build | chains | trials | unrouted (today's records) |
|---|---|---|---|
| Initial post3.1-on-main rebuild (stale cache) | 27/34 (79%) | 5 full, 2 partial, 3 zero | 30 |
| **After parent-anchoring + targeted re-classify** | **32/34 (94%)** | **8 full, 0 partial, 2 zero** | **11** |

The 2 remaining zero-coverage trials use single-arm peptide-vaccine compounds with unresolved targets (finding #2 below); not affected by the indication-slug fix.

## Why this exists

Round 3 wrapped at `a6f4411` (subtype consistency, SUBTYPE_OF hierarchy, non-oncology smoke test). The last per-trial inspection snapshots were `audit/inspection_*_post3.1.txt`, taken **before** rounds 3.2 / 3.3 / 3.4 / wrap landed. Today we generated `audit/inspection_*_post3.final.txt` and diffed.

Tooling needed to do the audit was missing on main (`scripts/inspect_trial.py` was only on `round-4-sub-chains`; `--max-trials` slicing didn't honor the standard inspection set). Both fixed in `6d88b74` and `862f0ed`.

## What rounds 3.2 / 3.3 / 3.4 / wrap actually delivered (verified)

| Fix | Verification |
|---|---|
| Round 3.3 `Compound → Intervention`, `binds_to → affects` | Edge updates now show `affects:` (e.g. `affects: nivolumab → ENSG00000188389` in NCT01844505 post3.final). |
| Round 3.3 filter chains to primaries | NCT01844505 chain count 24 → 8; NCT01950390 also dropped sharply. |
| Round 3.4 Reactome cap → 1 | Per-chain biology is now a single Reactome id rather than the multi-id fan-out visible in post3.1. |
| Previously-silent trials | NCT03484923 (5-target combo): `0` edges in post3.1 → `43` in post3.final. NCT03618641 (TLR9+PD-1 codename CMP-001): `0` → `20`. Largest single delta in this audit. |

These all show up cleanly in the diffs.

## New findings (surfaced by this audit)

### 1. Classifier-populator indication-slug mismatch — FIXED (HIGH → resolved)

**Original finding** — the initial post3.1-on-main rebuild used cached classifications from May-7 v0.1.0 that emitted subtype-qualified indications (`unresectable_or_metastatic_melanoma`, `intraocular_melanoma`, `recurrent_melanoma`, …) and synthetic biology slugs anchored on those. Populator chains used a mix of parent (`melanoma`) and subtype slugs depending on the trial. Mismatch ⇒ classifier backbone edges silently dropped. NCT01844505's mechanistic backbone (modulates_via, mechanism_affects, biology_drives, reflects_biology, endpoint_captures) was almost entirely unevidenced.

**Fix** (commit `e35678b`): anchor chain `indication_id` on the parent disease via `_root_indication(canonical_id)` at every chain-construction site. Endpoint node ids also keyed on parent so a single `PFS_melanoma` node serves every melanoma subtype. Subtype IndicationNodes and SUBTYPE_OF edges stay intact for cross-rollup queries; population slugs still encode the subtype via `compose_id`.

**Verification** (post-fix rebuild after targeted re-classify of standard set):

| Metric | Before | After |
|---|---|---|
| Chain coverage | 27/34 (79%) | 32/34 (94%) |
| Full / partial / zero | 5 / 2 / 3 | 8 / 0 / 2 |
| Unrouted records in dev log | 30 | 11 |
| NCT01844505 backbone edges (affects + modulates_via + mechanism_affects + biology_drives + endpoint_captures) | 1 | 9 |
| Standard-set trials in unrouted log | 4/4 | **0/4** |

**Same-shape note**: this resolution mirrors round-2's "use canonical entity ids" fix (`audit/fixes_round2.md`). Round 2 fixed entity-slug emission; round-3 verification fixes indication-slug emission. Same principle, one architectural level up.

### 2. UNKNOWN entities still appearing in classifier output (MEDIUM)

4 of the 11 remaining unrouted records (`no_chain_match`) are emissions where compound→target resolution failed upstream. Trials affected are peptide-vaccine combos (e.g. `aldesleukin+gp100_antigen+...`), small-molecule codename trials, and Lymphoseek (diagnostic radiopharmaceutical). The classifier emits `UNKNOWN` rather than refusing, which gets logged as `no_chain_match`.

**Severity**: MEDIUM — persistent gap, partially addressed across rounds 2 and 3. Most-affected archetypes are not the standard-set trials.

### 3. Stale-cache risk in `--keep-annotations` (LOW, already documented)

A subtle finding: the initial post3.1-on-main rebuild loaded cached classifications from `42e233c` (v0.1.0 baseline, May 7 — pre-round-3 entirely). `--keep-annotations` skipped the LLM and silently used pre-round-3 classifier output. Only deleting the targeted cache files exposed the current classifier prompt.

The playbook already says "if you changed a classifier or extractor prompt, you must delete the affected cache files first" (`automate_node_debug.md:43`), but the practical risk is higher than it reads: a round wrap that subsumes multiple prompt changes (3.2 + 3.3 + 3.4 + wrap) is easy to forget about.

**Severity**: LOW. No code change recommended — the playbook guidance is sufficient if followed. Worth a reminder in the next round-closing review.

### 3. Tooling gap on main (FIXED today)

`scripts/inspect_trial.py` lived only on `round-4-sub-chains`, so the post-round-3 audit couldn't be done on main. `--max-trials N` sliced the first N alphabetically, so the standard inspection set was unreachable at small N. Both fixed in this session: `6d88b74` (ported inspect_trial) and `862f0ed` (`--include` flag).

## Round-over-round edge-update counts (standard set)

| Trial | post3.1 | post3.final (post-fix) | Note |
|---|---|---|---|
| NCT01844505 (3-arm combo) | 16 | 16 | Same count, but composition is correct now: full mechanistic backbone (`affects`, `modulates_via`, `mechanism_affects`, `biology_drives`, `endpoint_captures`, `responds_differently`) lands instead of just AE edges + 1 backbone |
| NCT01950390 (failed bev+ipi) | 29 | 25 | Dedup from round-3.3 chain-filtering accounts for the small drop |
| NCT03484923 (5-target combo) | **0 (silent)** | 45 | Round 3.3 primary-filter fix verified |
| NCT03618641 (TLR9+PD-1 codename) | **0 (silent)** | 24 | Round 3.3 primary-filter fix verified |

Note: post3.1 was on a 96-trial graph; post3.final is on a 10-trial graph. Edge-update *counts per trial* are comparable (a trial's evidence emissions are independent of corpus size), but prediction sections are not apples-to-apples and weren't compared.

## Priority list (post-fix state)

| # | Bug | Status | Severity | Next |
|---|---|---|---|---|
| 1 | Classifier-populator indication-slug mismatch | **Resolved this session** (`e35678b`) | — | Re-classify the rest of the corpus at next round closeout to clear the remaining 7 `entity_not_in_trial` records from non-standard-set trials |
| 2 | UNKNOWN entities for peptide-vaccine / codename trials | Open | MEDIUM | Compound resolver follow-up — out of scope for this verification pass |
| 3 | Stale-cache risk on round wraps | Open (no code change) | LOW | Reminder at next round opening |

## What to commit from this session

Already on `main`:
- `6d88b74` Restore `scripts/inspect_trial.py` on main (round-3 compatible)
- `6d1c38d` Note conservative-rebuild guidance in debug playbook
- `862f0ed` Add `--include` flag to `build_graph.py` for the debug loop
- `e35678b` Anchor chain `indication_id` on parent disease
- `7ced85d` Surface Reactome biology name + pathway count in `inspect_trial`

Audit artifacts to optionally commit:
- `audit/inspection_01844505_post3.final.txt`
- `audit/inspection_01950390_post3.final.txt`
- `audit/inspection_03484923_post3.final.txt`
- `audit/inspection_03618641_post3.final.txt`
- `audit/fixes_round3.verification.md` (this file)

Per `automate_node_debug.md`'s ground rules, the `audit/` dir is committed selectively per round closeout — these are the verification round, separate from the original `fixes_round3.md`.
