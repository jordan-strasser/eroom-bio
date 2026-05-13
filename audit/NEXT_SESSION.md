# NEXT_SESSION — continue the round-3 verification cleanup

## Where we left off (2026-05-13)

Today's session ran a post-round-3-wrap audit on `main` and shipped the structural fix that surfaced. Branch state at session close: `main` is **8 commits ahead of `origin/main`** with everything green (567 tests). The round-4 sub-chain branch (`round-4-sub-chains`) is still parked unmerged; today's stash `stash@{0}` holds its WIP and should be popped when you switch back.

### What landed on main today

| Commit | Purpose |
|---|---|
| `6d88b74` | Restore `scripts/inspect_trial.py` on main (was only on `round-4-sub-chains`). |
| `6d1c38d` | Conservative-rebuild guidance in the debug playbook. |
| `862f0ed` | `--include` flag in `build_graph.py` so `--max-trials N` slices retain the standard inspection set. |
| `e35678b` | **Anchor chain `indication_id` on the parent disease** (the structural fix). |
| `7ced85d` | Surface Reactome name + pathway count in `inspect_trial` (audit visibility). |

### Headline KPI movement (10-trial slice, standard set)

| | Before fixes | After fixes |
|---|---|---|
| Chain coverage | 27/34 (79%) | **32/34 (94%)** |
| Trials full / partial / zero | 5 / 2 / 3 | **8 / 0 / 2** |
| Unrouted records | 30 | 11 |
| Standard set in unrouted | 4 / 4 | **0 / 4** |
| NCT01844505 backbone edges | 1 | **9** (full mechanistic backbone) |

Full write-up: `audit/fixes_round3.verification.md`. Snapshots: `audit/inspection_*_post3.final.txt`.

---

## What's left

Three concrete buckets. Tackle in this order — they get harder as you go.

### A. Clear the remaining 11 unrouted records (LOW effort, cheap Sonnet)

7 of 11 are stale-cache artifacts: non-standard-set trials whose classifications were generated against pre-round-3 indication/endpoint slugs (`stage_iv_melanoma`, `recurrent_melanoma`, `CR_stage_iv_skin_melanoma`, `ORR_stage_iv_melanoma`, synthetic biology slugs like `checkpoint_blockade__intraocular_melanoma`, `protein_degradation__melanoma`, `receptor_agonism__recurrent_melanoma`). They route nowhere because the post-fix populator anchors chains on `melanoma`.

The classifier prompt already says "no qualifiers" (`classification_system.txt:118`) — the only reason these still appear is the cache. Delete + rebuild for the affected trials:

```bash
# Trials with stale-cache unrouted records (today's slice):
for nct in NCT00003222 NCT00003509 NCT00019682 NCT00072189 NCT00084656 NCT00109005 NCT00110019; do
  rm -f data/annotations/${nct}_classification.json
done
.venv/bin/python scripts/build_graph.py \
  --corpus melanoma_145 --max-trials 10 \
  --include "NCT01844505,NCT01950390,NCT03484923,NCT03618641" \
  --keep-annotations
```

Then regenerate inspections, re-grep the dev jsonl, expect the stale-cache records to vanish. Cost: ~7 Sonnet calls. Coverage should rise to 33–34/34.

The remaining ~4 records will be the UNKNOWN-target archetype (B).

### B. Resolve UNKNOWN entities + the 2 zero-coverage trials (HIGH effort, real fix)

These three findings have one root cause: the compound→target resolver returns `UNKNOWN` for some intervention archetypes, and a chain with `target_id=UNKNOWN` can't take a `binds_to`/`affects` edge, can't reach a real Reactome biology, and degrades downstream.

**Today's zero-coverage trials in the slice:**

| Trial | Compound | Why UNKNOWN |
|---|---|---|
| `NCT00003509` | `antineoplaston_therapy_atengenal_astugenal` | Alternative/niche cancer therapy not in Open Targets. Mechanism falls back to `other` and biology to synthetic `other__melanoma`. |
| `NCT00019682` | `gp100_antigen` | Peptide vaccine; gp100 is a melanoma differentiation antigen with no clean Ensembl gene id at the intervention level. Mechanism = `immune_costimulation`, biology = synthetic `immune_costimulation__melanoma`. |

**Same archetype outside the slice (from `unrouted` log):**

- `NCT00003222`, `NCT00019682`: classifier emits `binds_to: aldesleukin+gp100_antigen+... → ENSG00000134460` (IL-2R alpha) but the chain has `target=UNKNOWN` because the multi-component combo compound doesn't resolve to a single ENSG. Result: `no_chain_match`.

**Suggested approach** (any one of these would help):

1. **Per-constituent target resolution for combos**: when the regimen is a combo, resolve each constituent compound separately and attach a `binds_to` per (compound, target) pair on the chain. The classifier already emits per-constituent edges; the populator just isn't building per-constituent target lookups for combos.
2. **Peptide-vaccine target heuristic**: trials with intervention type "biological" + name pattern matching `*_antigen|*_peptide|*_vaccine|*_idiotype` route to a known immunogenic-vaccine archetype with a curated default target (gp100→PMEL/ENSG00000185664, MART-1→MLANA, etc.) rather than UNKNOWN.
3. **Mechanism-only fallback chain**: when target genuinely can't be resolved, let the chain skip `binds_to`/`affects` and start at `mechanism_affects` so the rest of the backbone still has a place to land. Today the whole chain is unrouteable past target=UNKNOWN.

Approach 1 is the most architecturally aligned with the project goal (compositional decomposition) but takes the most work. Approach 2 is a 50-line patch and clears the peptide-vaccine archetype. Approach 3 is the smallest change and unlocks the most coverage immediately.

**Recommended first move**: approach 2 for the peptide-vaccine archetype (covers ~3 trials in the corpus), then approach 1 for combos.

### C. Re-classify the rest of the corpus (deferred, big-batch)

The full melanoma_145 corpus still has 135 trials' worth of pre-round-3 cached classifications. Most are fine — round-3 changes only affect a subset of edges — but a final closeout pass at round 3 done would delete all classifications and re-run end-to-end against the current prompts.

Per `feedback_conservative_rebuilds`: don't do this until you genuinely need it (round-3 final closeout, scaling readiness check, or before bumping the corpus to a different indication). One full corpus rebuild = ~145 Sonnet extract calls + ~145 Sonnet classify calls.

---

## Mechanical bootstrapping for the next session

```bash
# Verify clean state and recall what's queued:
git status --short
git log main --oneline -10
git stash list                                  # stash@{0} from round-4 still parked

# If returning to round-4:
git checkout round-4-sub-chains && git stash pop

# If staying on main and starting bucket (A):
git checkout main
# (follow the rm+rebuild commands in section A above)
```

The audit history (`audit/fixes_*.md`, `audit/inspection_*_post*.txt`, this file) is the long-form memory across sessions. Read `audit/fixes_round3.verification.md` first — it has the full per-trial table and the priority list.
