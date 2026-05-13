# Round 5 — Bucket A cleanup + inspect_trial race fix

Date: 2026-05-13. Skipping round-4 numbering: that work is parked on `round-4-sub-chains` and pre-existing `audit/inspection_*_post4.txt` files belong to that branch's exploration. This round is round 5 on `main` and diffs against `audit/inspection_*_post3.final.txt`.

## Goal

Execute the Bucket A work queued in `audit/NEXT_SESSION.md`: clear the 7 stale-cache classifications for non-standard-set trials whose pre-round-3 classifier output emitted subtype-qualified indication slugs (`stage_iv_melanoma`, `recurrent_melanoma`, `CR_stage_iv_skin_melanoma`, …) and synthetic biology slugs (`checkpoint_blockade__intraocular_melanoma`, `protein_degradation__melanoma`, `receptor_agonism__recurrent_melanoma`). The post-round-3 populator anchors chains on the parent disease (`melanoma`), so these slugs route nowhere.

## What ran

```bash
for nct in NCT00003222 NCT00003509 NCT00019682 NCT00072189 NCT00084656 NCT00109005 NCT00110019; do
  rm -f data/annotations/${nct}_classification.json
done
.venv/bin/python scripts/build_graph.py \
  --corpus melanoma_145 --max-trials 10 \
  --include "NCT01844505,NCT01950390,NCT03484923,NCT03618641" \
  --keep-annotations
```

7 Sonnet classify calls. No extractor calls.

## Headline KPIs

| | post3.final | post5 |
|---|---|---|
| Chain coverage | 32/34 (94%) | **34/34 (100%)** |
| Trials full / partial / zero | 8 / 0 / 2 | **10 / 0 / 0** |
| Standard set in unrouted log | 0 / 4 | 0 / 4 (unchanged) |

Both zero-coverage trials (`NCT00003509`, `NCT00019682`) flipped to full coverage. `NEXT_SESSION.md` predicted these were UNKNOWN-target archetypes (bucket B); the rebuild revealed they were actually stale-cache (bucket A). After re-classify with the current prompts, every chain in the 10-trial slice has at least one supporting update from its own trial. Final snapshot: `nodes=216 edges=491`.

## Post-rebuild unrouted breakdown

The `data/dev/unrouted_attribution_updates.jsonl` log is **append-only across runs**. Raw `wc -l` jumped 11 → 14, which on first read suggested regression. It isn't. Grouped by timestamp:

| Logged at | Records | Origin |
|---|---|---|
| `2026-05-13T14:10:53Z` | 11 | Pre-rebuild leftovers (the records `NEXT_SESSION.md` was tracking). Already-classified state at that timestamp. |
| `2026-05-13T19:14:50Z` | **3** | Today's rebuild. |

The 3 fresh records are all the **same bucket-B archetype** — peptide-vaccine compounds with no Ensembl gene resolution:

| Trial | Edge | Source | Target |
|---|---|---|---|
| `NCT00003222` | `affects` | `gp100_antigen` | `UNKNOWN` |
| `NCT00003222` | `affects` | `tyrosinase_peptide` | `UNKNOWN` |
| `NCT00019682` | `affects` | `gp100_antigen` | `UNKNOWN` |

Zero stale-slug emissions remain in fresh records. The classifier prompt's "no qualifiers" rule (`classification_system.txt:118`) is now reflected in every output.

## Standard-set diffs (post3.final → post5)

Small numerical drift on Beta strengths and predictions; no structural changes.

| Trial | Lines changed (+ / −) | Edge updates | P(success) WITH | Δ from post3.final |
|---|---|---|---|---|
| NCT01844505 | +11 / −11 | 16 | 0.8473 | +0.0002 |
| NCT01950390 | +5 / −5 | 25 | 0.8097 | +0.0002 |
| NCT03484923 | +5 / −5 | 45 | **0.7789** | **+0.0258** |
| NCT03618641 | +2 / −2 | 24 | 0.8361 | −0.0002 |

NCT03484923 (the 5-target combo) shifted the most because it benefits most from the upstream evidence flowing in from the 7 newly-routed trials in the slice. The "weakest link" annotation and the support-bucket assignments are unchanged on all 4 trials — the drift is in alpha/beta strengths, not in shape of the inference.

## What this round found (and fixed)

### 1. `scripts/inspect_trial.py` tmp-file race under parallel invocation

**Pattern**: First post5 inspection generation traceback'd:

```
FileNotFoundError: '/tmp/_inspect_strip_NCT03618641.json'
```

**Root cause**: `_strip_trial_evidence` writes a snapshot to `/tmp/_inspect_strip_<NCT>.json` (`scripts/inspect_trial.py:371`), calls `clone.import_snapshot`, then unlinks. When the standard-set inspections were launched in parallel and `--best 2 --worst 2` independently selected `NCT03618641` as a best trial, two processes wrote to the same path; one unlinked before the other finished `read_text()`.

**Severity**: LOW (only surfaces under parallel `inspect_trial.py` invocation, which the playbook implicitly encourages because there are 5 inspections per round).

**Fix shipped**: replaced the hand-rolled path with `tempfile.NamedTemporaryFile(...)` and a `try / finally` for cleanup. Now each process gets a unique path regardless of NCT overlap.

```python
with tempfile.NamedTemporaryFile(
    mode="w", suffix=f"_inspect_strip_{nct_id}.json", delete=False
) as fh:
    fh.write(snap_text)
    tmp_path = fh.name
try:
    clone.import_snapshot(tmp_path)
finally:
    Path(tmp_path).unlink(missing_ok=True)
```

567 tests still pass. Re-ran the two affected inspections in parallel post-fix — both completed cleanly.

### 2. `tests/test_attributor.py` was unlinking the real `data/dev/unrouted_attribution_updates.jsonl`

**Pattern**: After running `pytest tests/ -q` to verify the inspect_trial fix, `git status` showed `D data/dev/unrouted_attribution_updates.jsonl` — the real working-directory dev log file was deleted by the test suite.

**Root cause**: `tests/test_attributor.py:131` defined an `autouse=True` fixture `_clean_unrouted_log` that called `_UNROUTED_LOG_PATH.unlink()` on setup AND teardown. Because the production module's `_UNROUTED_LOG_PATH` is hardcoded to `Path("data/dev/unrouted_attribution_updates.jsonl")` (relative to cwd), every test run was destroying the real audit log.

**Severity**: MEDIUM. Silently destroys round-over-round audit data. Not catastrophic (the log is reproducible by re-running build), but exactly the kind of test-side state leak that hides as long as no one notices.

**Fix shipped**: replaced the autouse fixture with one that takes `tmp_path` + `monkeypatch` and redirects `_UNROUTED_LOG_PATH` to a per-test temp file:

```python
@pytest.fixture(autouse=True)
def _isolated_unrouted_log(tmp_path, monkeypatch):
    log_path = tmp_path / "unrouted_attribution_updates.jsonl"
    monkeypatch.setattr(_attributor_module, "_UNROUTED_LOG_PATH", log_path)
    yield log_path
```

Test references that previously did `_UNROUTED_LOG_PATH.read_text()` now go through `_attributor_module._UNROUTED_LOG_PATH` so they see the monkey-patched value. Verified: pre-pytest md5 of the real log equals post-pytest md5; 18 attributor tests still pass; full suite 567/567 passes.

### 3. (Observational, not a fix) Append-only unrouted log can mislead diagnosis

The `data/dev/unrouted_attribution_updates.jsonl` file accumulates across runs by design (mode `"a"` in `src/annotation/attributor.py:249`). A 11→14 line count looks like regression at first read; only grouping by `logged_at` revealed that the 11 pre-rebuild lines were still there and only 3 fresh entries came from today.

**Severity**: LOW. The log content is correct; the per-line read is the misleading part.

**Suggested options** (not shipped):
- Document the append behavior in `automate_node_debug.md` near "Watch for [these dev logs] after every build" so the next session reads `logged_at` first. **Lowest-effort, highest-leverage.**
- Or add a `run_id` field per record (a UUID stamped at build start) so `grep run_id=<latest>` isolates a single run.

Truncating at build start would lose round-over-round history and is not recommended.

## Bucket B — UNKNOWN-target archetype (not closed this round)

3 unrouted records, all peptide-vaccine compounds:

| Compound | Resolution attempt | Result |
|---|---|---|
| `gp100_antigen` | Not in Open Targets at the intervention level (gp100 is a melanoma differentiation antigen; the gene is `PMEL` / `ENSG00000185664`). | `UNKNOWN` |
| `tyrosinase_peptide` | Tyrosinase peptide vaccine; gene is `TYR` / `ENSG00000077942`. | `UNKNOWN` |

`NEXT_SESSION.md` already proposes three approaches (per-constituent target resolution for combos; peptide-vaccine heuristic; mechanism-only fallback chain) and recommends approach 2 first. Nothing changed there this round — recommendation stands.

---

## Four-layer pipeline audit on the 6 re-classified trials

`NEXT_SESSION.md` framed round 5 as a KPI win (32/34 → 34/34) but the standard set was already at 0/4 unrouted before the rebuild — its diffs were always going to be tiny. The actual round-5 work happened on the 6 newly re-classified non-standard-set trials, which never had per-trial inspections in any earlier round. Generating `audit/inspection_*_post5.txt` for those trials (NCT00003222, 00003509, 00019682, 00072189, 00084656, 00109005) and walking each through the 4-layer audit per `automate_node_debug.md §3` surfaced four new findings the headline KPI hid.

### Finding A — `predict_clinical_hypothesis` crashes on UNKNOWN-target hypothesis chains

**Pattern**: `NCT00003509` (antineoplaston therapy, alternative cancer therapy not in Open Targets). The hypothesis is `compound=antineoplaston_therapy_atengenal_astugenal indication=melanoma endpoint=ORR_melanoma`. The chain's `target_id=UNKNOWN`, the chain shows 6 routed edge updates (full coverage) — but the prediction section reports:

```
predict KeyError: "Node 'UNKNOWN' not found"
```

for BOTH `WITH` and `WITHOUT` variants. The compound has no traversable path because `predict_clinical_hypothesis` dereferences `target_id` as a node id without checking for the `UNKNOWN` sentinel.

**Severity**: HIGH. The chain-coverage KPI shows full coverage but the actual prediction the trial is supposed to support is a silent crash. This is currently invisible at the aggregate level.

**Scope**: 11/34 chains (32.4%) in this slice have `target_id=UNKNOWN`. Hypothesis crashes happen whenever the picked compound's chain has that property. NCT00003509 hits it; NCT00003222 (`gp100_antigen` hypothesis), NCT00019682 (`gp100_antigen`), and NCT00084656 (`ipilimumab` — has a non-UNKNOWN sibling chain that the path query found) don't crash because either the hypothesis chain has a known target or the path query falls through to a sibling.

**Suggested fix**: skip nodes equal to `UNKNOWN` (or use a sentinel constant) in the path-query traversal and treat the chain as biology-only (Δ from biology onwards). Two-line guard in `src/prediction/path_query.py`.

### Finding B — `_COMMONLY_SUPPORTIVE_COMPOUNDS` over-filters when supportive compound has its own monotherapy arm

**Pattern**: `NCT00019682` (Schwartzentruber 2011 — Phase 3 gp100 vaccine ± high-dose IL-2 vs IL-2 alone in advanced melanoma):

| Arm | compound_ids | Has a chain? |
|---|---|---|
| `arm_i_aldesleukin` | `['aldesleukin']` | **No** |
| `arm_ii_gp100_antigen_in_montanide_ida_51_and_aldesleukin` | `['aldesleukin', 'gp100_antigen', 'montanide_isa_51_vg']` | Yes — but only for `gp100_antigen` |

The entire IL-2 monotherapy arm — a published Phase 3 result for a foundational melanoma cytokine therapy — produces zero chains. Same pattern in NCT00003222 (5-compound peptide+IL-2+GM-CSF combo): no chain for aldesleukin even though the classifier emits `aldesleukin → ENSG00000134460` (IL-2R alpha) binding.

**Root cause**: `_identify_primary_intervention_ids` (`src/graph/populate.py:2341-2343`) demotes any compound in `_COMMONLY_SUPPORTIVE_COMPOUNDS` (aldesleukin, lymphodepletion chemo, common adjuvants) whenever any non-supportive candidate survives at the **trial** level. It doesn't check whether the demoted compound has its own monotherapy arm or is being studied as a comparator. Round 3.4's stopgap heuristic was correct for the cyclophosphamide-as-supportive case but is too aggressive for the "vs supportive-as-monotherapy comparator" case.

**Severity**: HIGH. Phase 3 evidence is silently dropped. The classifier's per-constituent edge emissions for aldesleukin / IL-2 land in the unrouted log instead of strengthening real edges.

**Suggested fix**: in `_identify_primary_intervention_ids`, before applying the supportive-allowlist demotion, scan arms for a monotherapy arm containing the candidate (`len(arm.compound_ids) == 1 and arm.compound_ids[0] == cid`). If yes, keep the compound as primary regardless of the allowlist. ~5-line patch. Add a regression test that NCT00019682 produces a chain for aldesleukin.

### Finding C — Reactome top-1 ranking returns disease-irrelevant pathways

**Pattern**: two cases in this slice where Reactome's default top-1 pathway has nothing to do with the trial's indication:

| Trial | Compound | Target | Reactome top-1 (what we use) | What it should be |
|---|---|---|---|---|
| `NCT00109005` | revlimid / lenalidomide | CRBN (ENSG00000113851) | `R-HSA-9679191` "Potential therapeutics for SARS" | CRBN E3 ligase / IKZF1-3 degradation / immune-modulatory pathway |
| `NCT00072189` | 7-hydroxystaurosporine | CHEK1 (ENSG00000140992) | `R-HSA-114604` "GPVI-mediated activation cascade" (platelet signaling) | Cell-cycle / DNA damage response / kinase inhibition pathway |

**Root cause**: `_BIOLOGY_PATHWAY_CAP = 1` (set in round 3.4) takes Reactome's default ranking. The 19 alternative pathways for CHEK1 and the 0 alternatives for CRBN (CRBN has *only* one Reactome pathway, which is the COVID-19 one) are sitting in `BiologyNode.pathway_ids` metadata, unused for biology_drives edges.

**Severity**: MEDIUM. Affects the biology_drives evidence routing but does not crash anything. Downstream effects: NCT00109005's "weakest link" flips between WITH/WITHOUT (the trial correctly weakens the wrong biology edge, so it becomes the bottleneck post-update). Confounds prediction interpretation.

**Suggested fix** (not for round 6; needs design): re-rank Reactome candidates by either (a) curated mechanism→pathway mapping, (b) pathway-indication co-occurrence in literature, or (c) fall back to the `{mechanism}__{indication}` synthetic slug when no pathway in the candidate list contains the indication's MeSH/disease ontology context. Coverage gap is in Reactome data, not in our code.

### Finding D — Subgroup populations anchor on parent disease, not trial indication

**Pattern**: `NCT00084656` is an intraocular melanoma trial (`parent_population_id=intraocular_melanoma__unselected`), but its antibody-status subgroup chains use `population_id=melanoma__antibody_status_positive` / `__negative` — anchored on the parent disease (`melanoma`), not the trial's actual indication.

**Severity**: LOW. Probably intentional cross-indication learning (antibody status is disease-axis-independent), but it's not documented anywhere. Worth flagging because a user querying `intraocular_melanoma__antibody_status_positive` would miss this trial's contribution.

**Suggested fix**: either (a) document the parent-disease anchor convention in `automate_node_debug.md` so reviewers expect it, or (b) double-write to both `melanoma__*` and `intraocular_melanoma__*` so subtype queries find the trial. (a) is the lighter touch.

## Priority list for round 6 (revised after the 4-layer audit)

| # | Bug | Severity | Effort | Reference |
|---|---|---|---|---|
| 1 | **`predict_clinical_hypothesis` crashes on UNKNOWN-target hypothesis chains.** | HIGH | Small (2-line guard in path query traversal + 1 test). | Finding A |
| 2 | **`_COMMONLY_SUPPORTIVE_COMPOUNDS` filter silently drops monotherapy/comparator arms** (aldesleukin in NCT00019682, NCT00003222). | HIGH | Small (~5-line check for monotherapy-arm existence + 1 test). | Finding B |
| 3 | Peptide-vaccine compounds resolve to `target=UNKNOWN` (gp100, tyrosinase, MART-1). | MEDIUM | Small (curated dict + intervention-type/name pattern matcher). | NEXT_SESSION.md bucket B + this round's UNKNOWN-target prevalence (11/34 chains) |
| 4 | Reactome top-1 ranking returns disease-irrelevant pathways (CRBN→SARS, CHEK1→GPVI). | MEDIUM | Medium (design pass on re-ranking; alternative: defer to round 7). | Finding C |
| 5 | Per-constituent target resolution for combos (classifier emits combined edges; populator decomposes inconsistently). | MEDIUM | Medium (populator change + chain backbone fan-out). | NEXT_SESSION.md bucket B approach 1 |
| 6 | Document `unrouted_attribution_updates.jsonl` append behavior in `automate_node_debug.md §3b`. | LOW | Trivial. | Round 5 finding |
| 7 | Document parent-disease subgroup-population anchor convention. | LOW | Trivial. | Finding D |
| 8 | Audit other tests for `.unlink()` / `rmtree` on production paths. | LOW | Small. | Round 5 commit `020cbe4` |

**Recommended round-6 scope**: #1 + #2 + #3. Combined effort is ~half a session and they unblock the most evidence flow. Findings A and B are both HIGH and were invisible to all prior rounds because the chain-coverage KPI rolls them up as "covered". #3 (peptide vaccines) was already queued for round 6 and benefits from being landed alongside #2 since both touch the compound-resolution layer.

#4 (Reactome relevance) and #5 (combo per-constituent decomp) need architectural design discussion and probably belong on a branch per `feedback_architecture_branches`.

#6, #7, #8 are housekeeping; bundle them into a single small commit at round closeout.

## Artifacts from this round

- `audit/inspection_01844505_post5.txt`
- `audit/inspection_01950390_post5.txt`
- `audit/inspection_03484923_post5.txt`
- `audit/inspection_03618641_post5.txt`
- `audit/inspection_extremes_post5.txt`
- `audit/fixes_round5.md` (this file)
- Code changes (uncommitted):
  - `scripts/inspect_trial.py` — tempfile race fix
  - `tests/test_attributor.py` — fixture monkeypatches `_UNROUTED_LOG_PATH` instead of unlinking the real file
- Build log: `/tmp/round5_build.log` (transient)

## Recommended next move

The KPI is at 100% on the 10-trial slice but the 4-layer audit shows that "100% chain coverage" is a misleading top-line — Findings A and B are both *invisible* at the KPI level because the chain exists and is touched by ≥1 update, even though prediction crashes (A) or whole arms were dropped before chain creation (B).

Round 6 should land #1 + #2 + #3 from the priority table. All three are small patches; combined they unblock 32% of the slice's chains (the UNKNOWN-target population) and recover an entire Phase 3 IL-2 monotherapy arm. Recommend a branch — `round-6-coverage-fixes` — since #2 touches the populator's chain-construction logic and that's the kind of change `feedback_architecture_branches` says to review on a branch first.
