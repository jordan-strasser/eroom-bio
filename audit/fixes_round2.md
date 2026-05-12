# Round-2 audit — fresh build, post round-1 fixes

Built against `data/corpora/melanoma_145.txt`, fresh extractions/classifications for the four fixes.md trials (NCT01844505, NCT01950390, NCT03484923, NCT03618641) and round-1 commits at `dc1ab95` + the per-arm AE fix at `8310f99` + the MedDRA cache self-heal at `1cc06ea`.

Final snapshot: **1414 nodes / 4750 edges / 145 trial_subgraphs**. Inspections at `inspection_*_post.txt`.

Same triage shape as round-1: by pipeline stage, severity at the end.

## 1. Classifier emits non-canonical indication slugs — 67 trials (46%) contribute zero efficacy evidence

**The single biggest evidence-loss bug in the system right now.** Round-1's #4 collapsed `unresectable_or_metastatic_melanoma`, `stage_iiic_cutaneous_melanoma_ajcc_v7`, etc. into one `melanoma` IndicationNode in the populator — but the classifier prompt didn't get the same memo, so it keeps emitting indication-qualified slugs that nothing in the trial subgraph matches.

Pattern, observed in `data/dev/unrouted_attribution_updates.jsonl`:

| Slug the classifier emitted | Times seen across corpus |
|---|---|
| `melanoma` (canonical) | 64 |
| `ORR_melanoma` | 15 |
| `safety_melanoma` | 14 |
| `metastatic_melanoma` | 10 |
| `malignant_melanoma` | 7 |
| `melanoma_skin` | 6 |
| `unresectable_melanoma` | 5 |
| `recurrent_melanoma` | 3 |
| `clinical_stage_iv_cutaneous_melanoma_ajcc_v8` | 3 |
| `stage_iii_melanoma`, `advanced_melanoma` | 2 each |
| + ~30 more variants | … |

Concrete example, NCT00670748 (170 chains in the subgraph, 0 efficacy updates landed):

```
classifier edges_to_update: [
  mechanism_affects: immune_costimulation → immune_costimulation__metastatic_melanoma
  endpoint_captures: CR_metastatic_cancer → metastatic_melanoma
]
```

Trial subgraph has `indication_id = melanoma` and biology nodes like `R-HSA-*__melanoma`. Both classifier edges drop as `entity_not_in_trial`.

**Root cause**: `src/annotation/prompts/classification_system.txt:106` lists *both* `melanoma` AND `metastatic_melanoma` as valid target examples for `responds_differently` — the prompt itself authorises the non-canonical slug. The classifier renderer at `classifier.py:106` *does* show the canonical indication, but the system prompt's example overrides it for any indication that isn't a perfect string match.

**Counts**:
- Total chains: 2327. Touched by ≥1 edge update from their own trial: **827 (36%)**.
- Trials with chains but **zero** updates landing: **67 (46% of 145)**.
- 14 trials get partial coverage (some chains updated, some silent).
- Only 64 trials have every chain touched.

**Severity**: HIGH. This is the largest single hole.

**Suggested fix**: 
1. Remove `"metastatic_melanoma"` from the system-prompt example at line 106. Replace with `(e.g. "melanoma" — use the exact id shown in "Shared trial-level entities" above)`.
2. Strengthen the renderer at `classifier.py:104–110` to repeat the canonical indication string verbatim at the top of `edges_to_update` instructions ("Use the indication id **`{sample.indication_id}`** for all *_drives, endpoint_captures, and responds_differently targets — do not add stage / metastatic / refractory qualifiers").
3. Optional: validate before logging — if the LLM emits an indication-shaped slug, normalise via the same canonicalizer the populator uses before attempting `entity_not_in_trial`.

## 2. Endpoint slug fragmentation — same bug, different field

`PFS_melanoma`, `ORR_melanoma`, `OS_melanoma`, `safety_melanoma` are correct. But the classifier also emits:

- `ORR_refractory_melanoma` (5)
- `ORR_stage_iv_melanoma` (3)
- `safety_metastatic_melanoma` (3)
- `safety_unresectable_melanoma`
- `OS_stage_iv_skin_melanoma`
- `composite_response_uveal_melanoma`
- `CR_metastatic_cancer`
- `CR_stage_iv_skin_melanoma`

All `entity_not_in_trial` drops. Same root cause as #1 (classifier picks specific over canonical), but a different prompt area — endpoint-slug guidance.

Worth noting one of the extremes inspections (composite_response_uveal_melanoma case) shows Δ P(success) = +0.0002 — uninformative trial because its endpoint slug doesn't match anything in the graph.

**Severity**: HIGH (compounds with #1).

**Suggested fix**: Same as #1 — strengthen renderer + remove misleading examples. Optionally, the populator's endpoint canonicaliser (`EndpointClass.*` prefix logic at `src/graph/populate.py` per CLAUDE.md description) could be extended to recognise these as aliases of the canonical endpoint when they hit the dev log.

## 3. Extraction emits RECIST response categories as "subgroups" — 683 unmapped descriptors across 18 trials

Top of `data/dev/unmapped_subgroup_features.jsonl`:

| Times | Raw "subgroup" |
|---|---|
| 42 | Final analysis |
| 41 | Progressive Disease |
| 41 | Stable Disease |
| 30 | Complete Response |
| 30 | Partial Response |
| 26 | CD8 T cells per mm² day 22 |
| 22 | Not Evaluable |
| 16 | Primary completion |
| 16 | CD8 T cells per mm2 in tumor pre-vaccine (day 0) |
| 16 | CD4 T cells per mm2 of tumor prevaccine (day 0) |
| 12 | HAHA positive |
| 12 | Not Evaluated |

The extractor is treating three different things as `subgroups`:

1. **Outcome categories** (RECIST: CR/PR/SD/PD/NE) — these stratify the *results* table, they are not patient subgroups.
2. **Analysis time points** (Final analysis / Primary completion) — when, not who.
3. **Continuous biomarker measurements** (CD8/CD4 per mm² at day N) — these need a stratifier definition the extractor doesn't have.

None of these resolve to a `PopulationNode`. They land in the dev log and contribute nothing.

**Root cause**: The extraction system prompt (`extraction_system.txt`) treats anything in CT.gov's `outcomeMeasures[*].classes[*].title` as a subgroup descriptor. CT.gov uses that field for both real subgroups and result/timepoint stratifiers. Likely also worth checking the Sonnet-extracted `subgroups` field for the same noise pattern.

**Severity**: MEDIUM. The graph is unaffected — these never become nodes — but it's wasted extractor work and the log is loud.

**Suggested fix**:
- In the extractor prompt, list explicit categories to reject: RECIST response categories (CR/PR/SD/PD/NE), analysis timepoints, raw continuous biomarker measurements.
- Optionally pre-filter at the populator before logging — if `raw_descriptor` matches one of these patterns, drop silently without logging.

## 4. Subgroup over-forking on trials with non-stratifying biomarkers

NCT00670748 has **170 chains** because the populator forked each arm × subgroup combination across 5 extracted subgroups (mostly CD8/CD4 timepoint measurements that aren't real stratifiers). The classifier emits 2 edges and nothing routes to those chains.

Round-1's fix #3 added a "subgroup-relevance gate" that successfully prevented PD-L1 forks on the ipilimumab arm in NCT01844505 (good). But the gate is target-aware (PDCD1 / CD274), not type-aware — it doesn't catch the case where the "subgroup" isn't a stratifier at all (RECIST states, biomarker timepoints).

**Severity**: LOW-MEDIUM. Mostly downstream of #3 — if extraction stops emitting these descriptors, the forking stops too. But worth a defensive check: if a "subgroup" doesn't produce a normalizable `SubgroupFeature`, don't fork.

**Suggested fix**: Tie subgroup forking to subgroup-feature resolvability — if the extractor's subgroup didn't normalize to a real population axis, don't create the chain.

## 5. Chain coverage by trial: a usable diagnostic, not surfaced today

Built this number for the audit; recommending it as a permanent metric:

```
Total chains across corpus:    2327
Chains touched by ≥1 update:    827  (36%)
Trials with full coverage:       64  (44%)
Trials with partial coverage:    14  (10%)
Trials with zero coverage:       67  (46%)
```

**Severity**: This isn't a bug; it's a measurement. Add to `scripts/analyze_run.py` so future audits don't need an ad-hoc script.

---

## Priority list

| # | Bug | Severity | Effort | Quick fix? |
|---|---|---|---|---|
| 1 | Classifier emits non-canonical indication slugs (67 trials silent) | HIGH | Small (prompt + renderer) | Yes |
| 2 | Classifier emits non-canonical endpoint slugs | HIGH | Small (paired with #1) | Yes |
| 3 | Extractor emits RECIST/timepoints as "subgroups" (683 dev-log entries) | MEDIUM | Small (prompt) | Yes |
| 4 | Subgroup over-forking on non-stratifying biomarkers | LOW-MEDIUM | Medium (populator change) | Defensive add-on after #3 |
| 5 | Chain coverage metric not exposed | LOW (instrumentation) | Trivial | N/A |

**Order to fix**: #1 and #2 together (single classifier-prompt PR) would close the biggest evidence-loss gap. #3 is the next-cheapest cleanup. #4 follows #3 naturally. #5 is a script change worth doing alongside.

---

## Notes / non-issues

- **Per-arm AE attribution is working as intended.** NCT01844505 anaemia now reads as expected (per-compound pooled rates vs comparator), and the 86% noise drop from the ≥3-affected threshold removed bevacizumab's noisy 1-patient AEs from NCT01950390 (verified by #9 staying ambiguous).
- **MedDRA cache self-healed** — 46 stale `Unspecified adverse event` entries rewrote to the empty sentinel on first hit during this build. Zero `AE:unspecified` references remain in any post inspection.
- **All round-1 fixes hold post-rebuild** — see the verdict table in the session handback. Fixes #2 and #6 (classifier-prompt) needed a targeted re-classification to surface; that's a process detail worth amending into `NEXT_SESSION.md` for future audits ("if testing classifier-prompt fixes, delete affected `*_classification.json` first").
