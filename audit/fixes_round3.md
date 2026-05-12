# Round-3 audit — fresh build, full re-classify at n=100

Built against `data/corpora/melanoma_145.txt` capped to 100 trials. All 94 cached classifications deleted before the build, so every classifier output reflects the round-2 prompt fixes (commits `00748b8`, `3b95683`, `73c24ca`). Pre-existing fixes from rounds 1 and 2 are now actually being tested at corpus scale rather than masked by stale caches.

Final snapshot: **1022 nodes / 2980 edges / 100 trial_subgraphs.**

Chain coverage: **838/1049 chains (80%) touched** — up from 36% pre-round-2-fixes and 45% pre-full-reclassify. 74/100 trials full coverage, 14 partial, **12 zero**.

The 12 remaining zero-coverage trials are real round-3 bugs, not cache artifacts. They split cleanly into two patterns.

---

## 1. Failure trials with low confidence_overall emit zero `edges_to_update` (9 of 12 silent trials)

**Pattern:**

| Trial | confidence_overall | edges_emitted |
|---|---|---|
| NCT02027935 | 0.3 | 0 |
| NCT01689974 | 0.4 | 0 |
| NCT01692691 | 0.4 | 0 |
| NCT02050321 | 0.3 | 0 |
| NCT00521001 | 0.4 | 0 |
| NCT01259284 | 0.4 | 0 |
| NCT01942993 | 0.3 | 0 |
| NCT02225366 | 0.4 | 0 |
| NCT00680225 | 0.3 | 0 |

Every one is `trial_outcome=failure` with `confidence_overall` in 0.3-0.4 (the <0.5 tier reserved for "insufficient information").

Sample reasoning, NCT02027935: "No biomarker data available to confirm target engagement for any of the four components ... No pathway biomarkers or pharmacodynamic data..."

**Root cause** (`src/annotation/prompts/classification_system.txt`):

Two rules in the system prompt are in tension and the LLM is resolving it the wrong way:

- Lines 66-69 (round-1 #6 fix): "Failure trials MUST emit at least one upstream causal-chain edge update when the primary efficacy endpoint was measured AND missed... Do NOT skip these updates because `insufficient_information` was picked."
- Lines 239-247 (the confidence rubric): `<0.5  Insufficient information — reserved for trials with no usable efficacy readout. Set the primary failure_mode to "insufficient_information"... Do NOT use this tier just because biomarker / PD data is absent — a clearly missed primary endpoint is itself informative and belongs at 0.5-0.7.`

The LLM:
- Sees a failure trial with no PD/biomarker data
- Goes to the 0.3-0.4 confidence tier instead of the 0.5-0.7 tier where the rule says it belongs
- Treats the "MUST emit at least one" directive as overridden by the low-confidence tier
- Returns `edges_to_update: []`

**Severity**: HIGH. 9 of 100 trials (9%) silenced by this alone. Every one is a Phase 2/3 failure with usable outcome data being ignored.

**Suggested fix** — combination of prompt + populator guard:
1. Strengthen the failure-trial directive at lines 66-69: explicitly enumerate that confidence_overall <0.5 is **not** a license to skip edges. Add a worked example showing a 0.3-confidence missed-endpoint trial that still emits `biology_drives weak_contradict`.
2. Add an audit assertion: when a failure trial returns zero `edges_to_update`, log a warning and (optionally) auto-emit a default `biology_drives weak_contradict` on the trial's parent chain so the failure isn't completely silent. Defensive code-side backstop for the prompt rule.

---

## 2. Classifier still emits non-canonical entity slugs (3 of 12 silent trials)

**Pattern**:

| Trial | Non-canonical slug emitted |
|---|---|
| NCT02302339 | `ORR_refractory_melanoma → melanoma` — endpoint **source** uses refractory qualifier |
| NCT01248936 | `melanoma__performance_good → melanoma` — `responds_differently` source isn't a real population node in this trial |
| NCT02366195 | `other → other__unresected_stage_iiib_to_ivm1c_melanoma`; `ORR_unresected_stage_iiib_to_ivm1c_melanoma → unresected_stage_iiib_to_ivm1c_melanoma` — multiple non-canonical slugs (mechanism, biology, endpoint source) |

**Root cause**: The round-2 prompt fix focused the renderer + system-prompt directive on **target-side** ids (the indication targets of biology_drives, endpoint_captures, responds_differently). It didn't cover **source-side** slugs symmetrically. Specifically:
- Endpoint sources in `endpoint_captures` (e.g. `ORR_melanoma` is canonical; `ORR_refractory_melanoma` is not)
- Population sources in `responds_differently` (the classifier sometimes invents `melanoma__performance_good` even when no such node was extracted)
- Mechanism / biology slugs in `mechanism_affects` / `biology_drives` chains

**Severity**: MEDIUM. 3 of 100 trials silenced; same root cause as round-2 #1+#2 but on the other side of the edge.

**Suggested fix**: Extend the round-2 prompt rule to source-side ids. The renderer at `classifier.py:104-110` already shows endpoint ids verbatim; the system prompt needs the same "use these exact ids on both source AND target" framing. Could also add a code-side guard that validates classifier output against the trial subgraph's id set and drops emissions that reference non-existent nodes with a logged warning (already partially done via `_log_unrouted`, but a stricter pre-check would surface these as classifier bugs rather than silent drops).

---

## 3. NCT00509496 stays at 2/16 chain coverage even after re-classify

**Pattern**: 16 chains in subgraph, only 2 touched. The classifier emitted 5 `edges_to_update` — none of them route to 14 of the chains. Not in either of the patterns above.

**Root cause** (suspected, needs trial-level inspection): The trial likely has many subgroup forks (perhaps multiple biomarker strata) and the classifier emits backbone-only edges that route to the unselected population chain but not to any of the subgroup chains. So `responds_differently` doesn't fire for those subgroups, and the unselected backbone gets a few edges while the subgroup forks stay silent.

**Severity**: LOW-MEDIUM. One trial; probably indicative of a broader "subgroup forks rarely get covered" issue but the size is hard to measure without a per-subgroup coverage metric.

**Suggested fix**: First open `audit/inspection_NCT00509496_post3.txt` and confirm the diagnosis. If correct, this is the same bug class as round-2 #4 (over-forking on non-stratifying features) but for cases where the subgroup IS a real stratifier and the classifier just didn't emit `responds_differently`. Tighten the classifier-prompt directive to emit one `responds_differently` per extracted subgroup that the trial reports a result for, not just when the LLM judges the effect size differs.

---

## 4. 17 "unknown"-type nodes from non-drug interventions

`build_graph` summary shows `'unknown': 17` in the node-type counter. Inspection:

```
quality_of_life_assessment       laboratory_biomarker_analysis    biopsy
questionnaire_administration     pharmacological_study            photographs
immunoenzyme_technique           total_body_irradiation_tbi       calcitriol
stereotactic_body_radiation_therapy   stereotactic_radiosurgery   polyiclc
radiation_therapy                stereotactic_body_radiation_therapy_sbrt
oncosec_medical_system_oms       ifa                              resiquimod
```

Five categories:
- **Procedures / diagnostics** (8): assessments, biomarker analysis, biopsy, questionnaire, etc. — these are CT.gov "intervention" entries that aren't therapeutic agents.
- **Radiation therapy** (5): TBI, SBRT, SRS, radiation_therapy. Real therapeutic modalities that don't fit `CompoundNode`.
- **Devices** (1): oncosec electroporation system.
- **Adjuvants / dropped drugs** (3): calcitriol, polyiclc, resiquimod, ifa — these ARE drugs but their CT.gov intervention type is probably `OTHER` rather than `DRUG`/`BIOLOGICAL`, so they slipped through the `DRUG_LIKE_INTERVENTION_TYPES` filter at `src/ingestion/clinicaltrials.py:54`.

**Root cause**: These nodes are getting created (probably by chain-build code that adds a placeholder for every arm intervention name) but without a `node_type` field. The graph then has them as orphans that can't participate in causal chains.

**Severity**: LOW. 17 of 1022 nodes (1.7%); doesn't break anything that's currently in coverage but pollutes the graph and hurts compound-id resolution for legitimate drugs (calcitriol, polyiclc, resiquimod, ifa).

**Suggested fix**:
- Audit `_PLACEHOLDER_INTERVENTION_PATTERNS` / `is_drug_like` rules in ingestion. Expand `DRUG_LIKE_INTERVENTION_TYPES` to include `OTHER` when the name looks drug-like (or relax the filter to type-aware: treat names with `-mab` / `-nib` / known generic suffixes as drugs regardless of declared type).
- For real non-drug interventions (radiation, biopsy, devices), either skip them at ingestion or create a typed `OtherInterventionNode` so they're not silent orphans.

---

## 5. 39 IndicationNodes for a melanoma-only corpus

At n=100 the snapshot has 39 IndicationNodes:

```
melanoma (canonical, 64 trials)
uveal_melanoma, intraocular_melanoma, iris_melanoma, ocular_melanoma,
choroidal_melanoma, mucosal_melanoma — distinct biological subtypes ✓
breast_cancer (?), lung_cancer (?), colon_cancer (?), ... — co-occurring conditions
brain_metastases / brain_metastasis — duplicate? likely should collapse
```

**Severity**: LOW (correctness) / MEDIUM (scaling readiness). Most are correct distinct subtypes. The likely-collapsible ones (singular/plural, near-synonym) are an LLM canonicalizer asymmetry — see fixes_round2.md scaling concerns.

**Suggested fix**: One-time cache sweep — read `data/cache/indication_canonicalizations.json`, find canonical ids that differ only by trivial morphology (plural vs singular, "metastases" vs "metastasis"), and pick one. Lock in a rule: the LLM canonicalizer prompt should normalize to singular forms.

---

## Priority list

| # | Bug | Severity | Effort | Quick fix? |
|---|---|---|---|---|
| 1 | Failure trials with low confidence emit 0 edges (9 silent trials) | HIGH | Small (prompt + code guard) | Yes |
| 2 | Source-side non-canonical slugs (3 silent trials) | MEDIUM | Small (prompt extension) | Yes |
| 3 | NCT00509496-style subgroup coverage gap | LOW-MEDIUM | Trial inspection + prompt rule | Yes (start with inspection) |
| 4 | Non-drug intervention nodes orphaned as "unknown" | LOW | Medium (ingestion filter audit) | Yes for the 4 drugs; defer the rest |
| 5 | IndicationNode near-duplicates (brain_metastases / brain_metastasis) | LOW | Trivial (one-time cache sweep) | Yes |

**Order to fix**: #1 has the biggest coverage impact (9 trials → coverage). #2 closes the remaining 3 silent trials and tightens the same prompt rule from the other side. #3 unblocks the partial-coverage tier next. #4 and #5 are cleanup that can ride along.

---

## Notes / non-issues

- **Round-2 prompt fix is fully verified at corpus scale** — chain coverage went from 36% (round 1) → 45% (round 2 with cached classifications masking the win) → 80% (round 2 fixes plus this re-classify). The round-2 fixes are correct; we just needed fresh classifier output to see it.
- **NCT01844505 (24/24) and NCT01950390 (9/9)** — the two standard-set combo + failed trials are now at full coverage. Round-1 priorities are locked in.
- **The MedDRA cache self-heal kept working** — zero `AE:unspecified_adverse_event` in the new snapshot.
- The same 50 trials that remained `--keep-annotations`'d (not re-classified, indices 100+) still hold pre-fix classifications. They're not part of this build but live in the same `data/annotations/` directory. They'll need re-classification before any future corpus expansion exercises them.
