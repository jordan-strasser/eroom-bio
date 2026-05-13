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

## Priority list for round 6

| # | Bug | Severity | Effort | Quick fix? |
|---|---|---|---|---|
| 1 | Peptide-vaccine compounds resolve to `target=UNKNOWN` (gp100, tyrosinase, MART-1, …). | MEDIUM | Small (curated dict + intervention-type/name pattern matcher). | Yes — `NEXT_SESSION.md` bucket B, approach 2. |
| 2 | Per-constituent target resolution for combos (classifier already emits per-constituent edges; populator builds combined compound nodes only). | MEDIUM | Medium (populator change + chain backbone fan-out). | No — needs design pass. |
| 3 | Document `data/dev/unrouted_attribution_updates.jsonl` append behavior in `automate_node_debug.md` step 3b. | LOW | Trivial. | Yes. |
| 4 | Audit other tests for the same anti-pattern as #2 above: are any other tests `.unlink()`ing real production paths during setup/teardown? | LOW | Small (`grep -rn "unlink\|rmtree" tests/`). | Yes. |

Items #3 and #4 are both <30 min of work and prevent future regressions; should ship alongside any bucket B work.

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

Land the inspect_trial fix as its own small commit (one-line change, one regression-free verification), then either:
- **(a)** open bucket B work on a branch (`peptide-vaccine-heuristic`) per `feedback_architecture_branches`, **or**
- **(b)** ship the playbook documentation edit (#3 above) as a tiny commit and defer bucket B until a fresh session.

The KPI is at 100% on the 10-trial slice. There is no urgent corpus-quality reason to push further before user review.
