# NEXT_SESSION — verify fixes.md fixes on a fresh build

The previous session implemented every item in `fixes.md` (#1–#9 plus
the "real biology nodes" priority-6 follow-up). The fixes are committed
at `dc1ab95` — `git show --stat dc1ab95` for the file list, `fixes.md`
itself for the original bug catalog, and the commit body for the
fix-by-fix recap. **Read both before starting.**

The on-disk graph snapshot at `data/exports/oncology_annotated.json`
was built **before** those fixes landed, and the inspection files we
have on disk (`inspection_01844505.txt`, `inspection_01950390.txt`,
`inspection_extremes.txt`) capture that pre-fix state. Your job is to
regenerate the snapshot + inspections with the new code, diff them
against the originals, and audit whether every fix actually shipped.

## 0. Don't overwrite the originals

The original pre-fix inspection files are the ground truth for
comparison. **Do not let any command write to those paths.** Write the
new captures to `*_post.txt`:

```
inspection_01844505.txt      ← keep   (pre-fix, ground truth)
inspection_01950390.txt      ← keep   (pre-fix, ground truth)
inspection_extremes.txt      ← keep   (pre-fix, ground truth)
inspection_01844505_post.txt ← write  (fresh run)
inspection_01950390_post.txt ← write  (fresh run)
inspection_extremes_post.txt ← write  (fresh run)
```

If the originals get clobbered the audit becomes much harder — `git
checkout` will only recover them if they were ever tracked, and they
aren't.

## 1. Rebuild the graph end-to-end against the frozen corpus

```
python scripts/build_graph.py --corpus melanoma_145 --keep-annotations
```

- `--corpus melanoma_145` pins the 145-trial set so the rebuild is
  reproducible. The corpus file is `data/corpora/melanoma_145.txt`.
- `--keep-annotations` reuses cached extract+classify outputs in
  `data/annotations/` instead of re-running every Sonnet call. The
  fixes we care about live in the populate + classifier-prompt +
  attributor layers; rerunning extraction would cost a lot of tokens
  without changing the inputs to those layers.
- Expect the snapshot at `data/exports/oncology_annotated.json` to be
  overwritten — that's fine. (If you want to keep the old one for
  side-by-side prediction, copy it to a `_pre.json` first.)

## 2. Re-run the three inspections (writing to new files)

```
python scripts/inspect_trial.py NCT01844505 > inspection_01844505_post.txt
python scripts/inspect_trial.py NCT01950390 > inspection_01950390_post.txt
python scripts/inspect_trial.py --best 2 --worst 2 > inspection_extremes_post.txt
```

For reference, the pre-fix originals are:

```
inspection_01844505.txt: 839 lines / 49 015 bytes
inspection_01950390.txt: 583 lines / 31 552 bytes
inspection_extremes.txt: 2406 lines / 133 658 bytes
```

`diff -u inspection_01844505.txt inspection_01844505_post.txt | less`
is the easiest way to inspect the changes. The diffs will be huge —
don't dump them to the conversation; summarize by section
(extraction / classification / node mapping / edges / prediction).

## 3. Verify each fix actually landed

For each item below, check the new inspection files for the predicted
new behavior. List anything that DIDN'T change as expected — that's a
sign the fix didn't fire.

### #1 + #2 — Per-constituent chains (NCT01844505)
- `arm_c_ipilimumab_*` chain target should be `ENSG00000163599` (CTLA-4),
  not `ENSG00000188389` (PD-1).
- `arm_b_nivolumab_ipilimumab_*` should now appear as TWO chains in
  the node mapping — one with `compound_id=nivolumab, target=PD-1`,
  one with `compound_id=ipilimumab, target=CTLA-4`. (Pre-fix: one
  chain with `compound_id=ipilimumab+nivolumab, target=PD-1`.)
- Edge updates should now include `binds_to: ipilimumab → ENSG00000163599`.
- Same edge should NOT be updated twice in the same trial (dedup).

### #4 — Indication canonicalization
- NCT01844505 `indication_id` and NCT01950390 `indication_id` should
  both be `melanoma` (or whatever the canonicalizer chose) — not
  `unresectable_or_metastatic_melanoma` vs
  `stage_iiic_cutaneous_melanoma_ajcc_v7`.
- The trial's parent population should be a qualified slug like
  `melanoma__histology_cutaneous__stage_iii`, not `melanoma__unselected`.
- Endpoint id should match: `OS_melanoma`, `PFS_melanoma`. The
  classifier's `endpoint_captures` source/target should agree
  (no more `OS_unresectable_melanoma → stage_iiic_*_v7` slug-bridge).
- Open the canonicalization cache (`data/cache/indication_canonicalizations.json`)
  to see what variants collapsed to what base disease.

### #6 — Failed trial efficacy edges (NCT01950390)
- Pre-fix: only 1 entry in `edges_to_update`, all 22 graph updates
  are `causes_ae`. Post-fix: should see at least one `biology_drives`
  contradict (weak_contradict or moderate_contradict) AND probably an
  `endpoint_captures` ambiguous/weak_contradict.
- The classifier's reasoning should reflect the new prompt rule that
  a missed-endpoint failure with no PD data still informs the
  upstream chain.

### #7 + #8 — MedDRA pre-filter and British→American
- Search for `AE:unspecified_adverse_event` in either inspection
  file's edge-update section — should be **gone**. (Pre-fix:
  ipilimumab → AE:unspecified_adverse_event with P ≈ 0.93.)
- Search for `AE:anaemia` — should be gone, collapsed into
  `AE:anemia`. Same for `haemo*` variants.
- Look in `data/cache/meddra_terms.json` for empty-`preferred_term`
  entries — those are the rejected meta rows.

### #5 — PD-L1 vocab collapse
- NCT01844505 node mapping should show subgroup populations only at
  `cd274_positive` and `cd274_negative`, NOT `cd274_high`, `cd274_low`,
  or a mix. (Pre-fix had all three.)

### Priority-6 follow-up — Real Reactome biology
- Every chain's `biology_id` should be either `R-HSA-*` (real
  Reactome pathway) or a slug tagged with
  `metadata.unresolved_biology = True`.
- NCT01844505 nivolumab chains should point at `R-HSA-389948`
  (Co-inhibition by PD-1); ipilimumab chains at `R-HSA-389513`
  (Co-inhibition by CTLA4) or `R-HSA-389356` (Co-stim by CD28).
- Count the slug-fallback chains vs Reactome chains — that ratio
  tells us how much real-biology coverage we have.

### #3 — Subgroup-relevance gate
- NCT01844505 ipi-only arm chains should NOT be forked across PD-L1
  subgroup populations anymore. Pre-fix: `arm_c_ipilimumab_*` had 4
  chains across `unselected / cd274_low / cd274_positive / cd274_high`.
  Post-fix: `arm_c` should only appear at the parent (qualified)
  population.
- Nivolumab chains SHOULD still fork across PD-L1 subgroups
  (PDCD1 and CD274 share Reactome pathways).

### #9 — AE absolute-count gate
- In NCT01950390's edge updates, bevacizumab → AE edges for AEs at
  ~1.2% incidence (gait disturbance, sudden death, erythema multiforme,
  pruritus, skin ulceration) should now be **AMBIGUOUS**, not
  moderate_support. With trial enrollment 169 and 2 arms, abs_count
  ≈ 1 patient, which the new gate drops to AMBIGUOUS.

## 4. Re-do the raw-text → JSON → node-embedding audit

This is the open-ended part. Re-run the same kind of analysis fixes.md
captured originally — pick a handful of trials (the four in fixes.md
plus a couple from the `--best/--worst` extremes) and trace each
through the four pipeline layers, looking for new failure modes:

1. **Raw text vs extraction JSON** — does the extractor pick up arms,
   subgroups, and AEs accurately? Watch for: hallucinated subgroups,
   per-arm result misattribution, AE term that the LLM normalized
   inconsistently across spelling/punctuation variants the
   pre-filter didn't catch.
2. **Extraction JSON vs node mapping** — does every arm/subgroup/AE
   in the extraction resolve to a canonical graph node? Look in
   `data/dev/unmapped_subgroup_features.jsonl` and
   `data/dev/unrouted_attribution_updates.jsonl` for items that
   should have routed but didn't.
3. **Node mapping vs edge updates** — for each chain in a trial,
   is there at least one corresponding edge update? Chains with
   zero updates are silent failures (the classifier didn't emit
   anything that routed there). Conversely, edge updates with
   `evidence_type` not matching `_PHASE_TO_EVIDENCE[trial.phase]`
   are a sign that phase routing is broken.
4. **Edge updates vs prediction** — when WITHOUT-this-trial P(success)
   barely moves from WITH, the trial isn't carrying any informative
   signal. List those — they're the new "worst" bucket.

Write the audit results to a fresh `fixes_round2.md` at the repo
root (don't overwrite `fixes.md` — it documents the round-1 bugs that
are now fixed). Group findings the same way fixes.md does — by
pipeline stage, with a priority list at the end.

## 5. Hand-back

When you're done, leave a short summary in this turn's response:

- which inspection files you wrote (paths)
- which of the eight fixes verified vs which didn't
- top 3 issues from the round-2 audit (if any), with severity

Don't `git add` or commit anything by default — let the user pick the
scope (same as last session). The new `inspection_*_post.txt` files
and `fixes_round2.md` should stay untracked unless explicitly
requested.
