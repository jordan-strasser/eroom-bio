# automate_node_debug — iterative pipeline-correctness loop

## Project goal

A structured knowledge graph for clinical trials where each trial is decomposed into a causal hypothesis chain with probabilistic beliefs populating each edge, so that we can learn across **indications, endpoints, populations, mechanisms, targets, biology, compounds, and adverse events.**

The graph is only as useful as the fidelity of its layer-to-layer mapping. This document defines a repeatable audit cycle that surfaces pipeline errors at every stage, fixes them, and proves the fixes shipped.

This playbook is **corpus-agnostic** — it applies the same way whether the working corpus is melanoma trials, NSCLC trials, immunology trials, or the full CT.gov universe. The current corpus configuration is captured at the bottom under "Current configuration" and is the only thing that changes when the working corpus shifts.

## Where audit artifacts live

All audit artifacts — inspection files, fix manifests, round-over-round handbacks — go under the `audit/` directory at the repo root. The repo root itself stays clean. Specifically:

- `audit/inspection_<NCT>_postN.txt` — per-trial inspection snapshots for round N
- `audit/inspection_extremes_postN.txt` — best/worst extremes for round N
- `audit/fixes_roundN.md` — audit findings + priority list for round N
- `audit/` is fine to leave untracked while iterating; commit selectively per round closeout.

Pre-fix inspection files (round 1 before any fixes landed) live in `audit/` too without a `_postN` suffix — they're the original ground-truth snapshots that subsequent rounds diff against.

---

## The cycle, one pass

Each iteration is called a **round**. After round N completes (`fixes_roundN.md` written), increment N and start again. The cycle is the same regardless of which round you're in.

```
       ┌─────────────────────────────────────────────────────┐
       │                                                     │
       ▼                                                     │
  1. BUILD ─→ 2. INSPECT ─→ 3. AUDIT ─→ 4. FIX ─→ 5. VERIFY ─┘
```

### 1. Build the graph

```bash
python scripts/build_graph.py --corpus <CORPUS> --max-trials <N> \
    --include <NCT_A>,<NCT_B>,<NCT_C>,<NCT_D> \
    --keep-annotations
```

- `--corpus <CORPUS>` pins a frozen NCT id list so successive runs are reproducible. The file at `data/corpora/<CORPUS>.txt` is the source of truth for which trials this corpus contains.
- `--max-trials <N>` caps how many of those trials are processed per build. **The exact number is not significant**; pick a value that makes the build fast enough that audit cadence is the bottleneck, not build time. Bigger numbers exercise more diversity at the cost of slower iteration.
  - **Be conservative with `--max-trials` during the debug loop.** This playbook is meant to run many rounds, and each round costs real Sonnet/Haiku tokens for any trial that needs (re-)annotation. Default to a small N (e.g. `--max-trials 10`) for intermediate verification builds; reserve larger N for round closeout or when you specifically need more archetype diversity than the small slice provides. Treat re-builds the same way: only rebuild when there's a concrete reason (prompt change, fix to verify, KPI check), not as a routine "let's see what it looks like now."
- `--include <NCT_IDS>` pins the standard inspection set (or any other must-have NCTs) to the head of the corpus order so they survive the `--max-trials` cap. Without this, `--max-trials 10` just slices the first 10 ids alphabetically — and the round-over-round diff anchor (the standard set) won't be in the slice. Pair `--include` with the standard inspection set listed in "Current configuration" below.
- `--keep-annotations` reuses cached Sonnet `_extraction.json` / `_classification.json`. **If you changed a classifier or extractor prompt, you must delete the affected cache files first** (see "Targeted re-classification" below) — otherwise the new prompt isn't being tested.
- Watch the tail of the build output. The chain-coverage line is the headline KPI:

  ```
  chain coverage: 718/1593 chains (45%) touched by ≥1 update from their own trial
                  67/145 trials full, 16 partial, 62 zero
  ```

  Round-over-round, **the number of zero-coverage trials should monotonically decrease** if your fixes are working. If it doesn't, the fix didn't ship — investigate before continuing.

### 2. Inspect a stable trial set

The audit uses the same trials every round so diffs are meaningful. Pick them once when starting a new corpus, then keep them stable across all rounds in that corpus. The standard set should include each of these archetypes, drawn from the current corpus:

| Archetype | Why it's in the standard set |
|---|---|
| **Combo trial** | Stress-tests per-constituent chain decomposition, multi-target binding, subgroup forking on patient strata. |
| **Failed trial** | Stress-tests failure-mode classification, biology_drives contradicts, and whether the system learns from negative evidence. |
| **N-way multi-target combo** (3+ active drugs) | Stress-tests classifier per-constituent emission and target lookup under heavier load. |
| **Codename / non-canonical compound trial** | Stress-tests compound-id resolution under names that don't match standard databases (e.g. internal sponsor IDs, code numbers). |
| **`--best 2 --worst 2`** | Auto-finds whichever trials currently sit at prediction extremes — surfaces what the system is most/least confident about, and whether that confidence is justified. |

Generate inspections to round-numbered file names under `audit/`:

```bash
python scripts/inspect_trial.py <TRIAL_A> > audit/inspection_<TRIAL_A>_postN.txt
python scripts/inspect_trial.py <TRIAL_B> > audit/inspection_<TRIAL_B>_postN.txt
python scripts/inspect_trial.py <TRIAL_C> > audit/inspection_<TRIAL_C>_postN.txt
python scripts/inspect_trial.py <TRIAL_D> > audit/inspection_<TRIAL_D>_postN.txt
python scripts/inspect_trial.py --best 2 --worst 2 > audit/inspection_extremes_postN.txt
```

Where `N` is the current round number (post1, post2, post3…). **Do not overwrite earlier rounds' inspections.** Each `_postN.txt` is a snapshot for diffing.

### 3. Audit the four pipeline layers

Walk each trial through the layers in order. For every gap between layers, note it as a bug. The questions below are the same regardless of indication or therapy area.

#### 3a. Raw text → extraction JSON
- Open the inspection file's `RAW TEXT` section, then the `EXTRACTION JSON` section.
- Did the extractor capture every arm? Every reported subgroup? Every per-arm AE rate? The primary endpoint result?
- Watch for:
  - Hallucinated subgroups (extractor inventing strata that aren't in the trial)
  - Outcome categories (e.g. RECIST response states like CR/PR/SD/PD, analysis time points) emitted as `subgroups` rather than as endpoint stratifications
  - Per-arm AE rates flattened into one tx/ctrl pair when the trial has 3+ arms
  - Primary endpoint met / not met inverted from the source
  - Combo arms mis-attributing results to constituent monotherapies
  - Compound names that the extractor couldn't normalize (internal codenames preserved as-is)

#### 3b. Extraction JSON → node mapping
- Compare the `EXTRACTION JSON` section to the `NODE MAPPING` section.
- Does every arm in extraction map to a TrialArm with the right compound_ids? Every subgroup to a PopulationNode? Every endpoint to an EndpointNode?
- Check these dev logs after every build:
  - `data/dev/unmapped_subgroup_features.jsonl` — features the populator couldn't canonicalize. Each entry is something to investigate (vocab gap, axis miscategorization, or "this isn't really a subgroup").
  - `data/dev/unrouted_attribution_updates.jsonl` — classifier-emitted edges that didn't land. `entity_not_in_trial` typically means slug mismatch (classifier emitted a more specific or differently-qualified id than the populator chose). `no_chain_match` means entities exist but the (compound, target, mechanism, …) combination isn't a chain in this trial.
- Are the indication and endpoint slugs the classifier emits exactly the canonical ids shown in the trial subgraph? Any qualifier the classifier added (stage, refractory, line-of-therapy, anatomical site) that the canonicalizer stripped will silently drop the edge.

#### 3c. Node mapping → edge updates
- Compare the `NODE MAPPING` section to the `EDGE UPDATES` section.
- For each chain in the trial, is there at least one corresponding edge update? Silent chains are the largest source of evidence loss.
- For each edge update, is the `support` bucket consistent with the trial outcome and the failure-mode taxonomy? (A success trial emitting `weak_contradict` on a primary edge, or a failure emitting `strong_support` on `binds_to`, is internally inconsistent.)
- Are evidence types correct? The evidence type drives `N_eff` (Phase 3 = 15, Phase 2 = 6, Phase 1 = 2, Genetic MR = 10, etc.); a Phase 3 trial routed as Phase 2 evidence dilutes the update by 2.5×.

#### 3d. Edge updates → prediction
- Check the `PREDICTION` section's `WITH` vs `WITHOUT` P(success) deltas.
- **|Δ P| ≤ 0.01 means the trial contributed no informative signal.** That trial belongs on the list of round-N failures.
- For successful trials, Δ should be positive; for failures, negative. Sign mismatches are red flags (often: the classifier picked the wrong support direction, or the trial's chain doesn't reach the prediction's hypothesis).
- Inspect the "weakest link" annotation — if it points at an edge that should have been strengthened by this trial but wasn't, the classifier missed an emission.

### 4. Record findings in `audit/fixes_roundN.md`

Group findings by pipeline layer, with severity and a priority list at the end. Each finding entry:

```markdown
## N. <one-line summary>

**Pattern** (concrete examples from the inspection files, with NCT ids):
- ...

**Root cause** (which code path, which file:line):
- ...

**Severity**: HIGH / MEDIUM / LOW

**Suggested fix**:
- ...
```

End with a priority table:

```markdown
## Priority list

| # | Bug | Severity | Effort | Quick fix? |
|---|---|---|---|---|
| 1 | ... | HIGH | Small (prompt) | Yes |
```

Use absolute dates if relative ones come up (the file may be re-read months later). Cite file:line references where the root cause lives so the next reader can jump straight there.

### 5. Fix, verify, loop

For each fix:

1. **Write the code change** with a regression test that captures the bug before the fix and passes after.
2. **Run the test suite**: `python -m pytest tests/ -q --ignore=tests/integration`. Should be green.
3. **Targeted re-classification if the fix is in a Sonnet prompt** (classifier or extractor):
   ```bash
   rm data/annotations/<NCT_ID>_classification.json   # or _extraction.json
   python scripts/build_graph.py --corpus <CORPUS> --max-trials <N> --keep-annotations
   ```
   Only the deleted file(s) hit Sonnet again. Use this when verifying classifier-prompt fixes against the standard inspection set — fully re-classifying every trial in the corpus should be reserved for the final round closeout, not every intermediate fix.
4. **Re-run inspections** (step 2 above, incrementing N to N+1).
5. **Diff to verify**:
   ```bash
   diff -u audit/inspection_<TRIAL_A>_postN.txt audit/inspection_<TRIAL_A>_post(N+1).txt | less
   ```
   The diff should show:
   - The specific bug pattern from round N **gone** in round N+1.
   - The chain-coverage KPI improved.
   - No regressions in trials that weren't part of the fix.
6. **Commit** with a message that names the fix and links the bug entry in `audit/fixes_roundN.md`. One commit per logical fix; group fixes that touch the same file with one larger commit only when they're inseparable.

When all round-N priorities are closed, the loop has converged for this round. Start round N+1 by re-running step 1.

---

## Switching corpora

When you swap the working corpus (e.g. from a single-indication starter set to a broader cross-indication mix):

1. Write the new corpus file at `data/corpora/<NEW_CORPUS>.txt`, one NCT id per line.
2. Pick a new standard inspection set per the archetypes in step 2 above, drawn from the new corpus.
3. Update the "Current configuration" section at the bottom of this file.
4. Treat the first round on the new corpus as round 1 of its own series. Either branch off a clean numbering or use `audit/fixes_<CORPUS>_round1.md` to keep histories separate. Earlier-corpus rounds remain the historical record for that corpus.

The cycle steps themselves don't change. What changes is the choice of trials and the kind of edge cases that will surface — a multi-indication corpus, for example, will stress-test cross-indication slug canonicalization and disease hierarchy in ways that single-indication audits never could.

---

## Ground rules

- **Never overwrite a previous round's inspection file or `audit/fixes_roundN.md`.** They are the diff baseline. New rounds go to `audit/inspection_*_post(N+1).txt` and `audit/fixes_round(N+1).md`.
- **Never `git add` the contents of `audit/`, the dev jsonl logs, or the snapshot exports by default.** They are working artifacts. The user decides what gets committed and when.
- **Test before commit.** A fix without a regression test makes the next round more expensive.
- **The chain-coverage KPI is the headline.** If it didn't move, the round didn't deliver — investigate why before declaring done.
- **Targeted re-classification, not bulk.** Sonnet tokens are real money; re-classify only the trials you're actively verifying, except at round closeout.
- **Cache pollution masks prompt fixes.** If a classifier or extractor prompt changed, the corresponding cache file must be deleted for the affected trial. Older runs may have written stale mappings; the meddra cache self-heal in `1cc06ea` is the canonical example.
- **A round produces three durable outputs**: `audit/fixes_roundN.md` (audit findings), git commits (the fixes themselves), and `audit/inspection_*_postN.txt` (the diffable evidence). Anything else is scratch.

---

## Standard round structure (paste into a session)

```
Goal: round N of the automate_node_debug loop.

Steps:
1. Build the graph: <CORPUS>, --max-trials <N>, --keep-annotations.
2. Generate the standard inspection files at audit/inspection_*_postN.txt.
3. Read audit/fixes_round(N-1).md to see what was meant to be closed.
4. Verify each round-(N-1) fix actually landed (grep + diff against audit/*_postN-1 files).
5. Audit the four pipeline layers per the standard set; record new findings.
6. Write audit/fixes_roundN.md with priority list.
7. Hand back: which round-(N-1) fixes verified, top-3 round-N findings, what to commit.
```

The user picks scope from the hand-back. Don't proceed past the audit without alignment.

---

## Current configuration

This is the only section that changes when the working corpus shifts. Update it at the start of every round.

| Setting | Value |
|---|---|
| Working corpus | `melanoma_145` (`data/corpora/melanoma_145.txt`, 145 NCT ids) |
| Build size | `--max-trials 100` |
| Standard inspection set | NCT01844505 (3-arm checkpoint combo); NCT01950390 (failed bev+ipi); NCT03484923 (5-target combo); NCT03618641 (TLR9+PD-1 codename CMP-001) |
| Extremes inspection | `--best 2 --worst 2` |

### History

| Round | Headline | Commits |
|---|---|---|
| 1 (`audit/fixes.md`) | 9 evidence-routing bugs from the first pipeline audit (per-arm chains, indication canonicalization, MedDRA filter, failed-trial efficacy, …) | `dc1ab95`, `1cc06ea`, `8310f99` |
| 2 (`audit/fixes_round2.md`) | Classifier emitted non-canonical entity slugs (67 trials silent); RECIST states treated as subgroups; chain-coverage metric not surfaced | `00748b8`, `3b95683`, `73c24ca` |
| 3 (pending) | Run the loop again to surface what round-2 didn't catch | — |

A round that doesn't find anything new is the signal that the pipeline is stable enough to grow the corpus. Until then, keep looping.
