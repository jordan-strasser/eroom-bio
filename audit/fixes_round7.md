# Round 7 — Re-inspect after the round-6 cleanup + round-6.1 crash guard

Date: 2026-05-13. Same 10-trial slice (`--corpus melanoma_145 --max-trials 10 --include <standard set>`). Snapshot unchanged structurally from round 6 (`nodes=216 edges=491`, chain coverage `47/48 (98%)`, `9/10` trials full / `1` partial / `0` zero). 6.1 only touched prediction code.

## Status of round-6 round-6.1 fixes

| Fix | Status | Evidence |
|---|---|---|
| Round 6 revert (drop primary/supportive heuristic) | **Verified.** | NCT00019682's aldesleukin monotherapy arm has Chain 1 (`aldesleukin → IL2RA → receptor_agonism → R-HSA-5673001 → melanoma`). NCT00003222 has chains for all 6 compounds in the regimen, including aldesleukin + sargramostim. Zero zero-coverage trials. |
| Round 6.1 prediction guard | **Verified.** | All 10 slice trials produce P(success) values. NCT00003509 (antineoplaston): WITH=0.6510, WITHOUT=0.6635. NCT03618641 (cmp_001 hypothesis): WITH=0.4975, WITHOUT≈0.49 (degenerate but no crash). |

## Slice-wide prediction snapshot

| Trial | Hypothesis (post7) | P(success) | Notes |
|---|---|---|---|
| NCT00003222 | aldesleukin | 0.5804 [0.358, 0.785] | Combo failure (peptide vaccine + IL-2 + GM-CSF + adjuvants). |
| NCT00003509 | antineoplaston_therapy_atengenal_astugenal | 0.6510 [0.413, 0.853] | Single-arm alt therapy; UNKNOWN target. |
| NCT00019682 | aldesleukin | 0.6612 [0.452, 0.837] | Hypothesis flipped from `gp100_antigen` (post6). |
| NCT00072189 | 7_hydroxystaurosporine | 0.6294 [0.440, 0.794] | Pan-kinase inhibitor monotherapy. |
| NCT00084656 | ipilimumab | 0.8266 [0.736, 0.901] | Intraocular melanoma. |
| NCT00109005 | revlimid | 0.5576 [0.377, 0.725] | Lenalidomide melanoma trial. |
| NCT01844505 | nivolumab | 0.8456 [0.774, 0.907] | Standard. |
| NCT01950390 | ipilimumab | 0.8100 [0.714, 0.889] | Standard. |
| NCT03484923 | pdr001 | 0.7782 [0.686, 0.858] | Standard. |
| NCT03618641 | **cmp_001** | **0.4975 [0.027, 0.975]** | **Hypothesis flipped from `nivolumab` (post5: 0.8361).** Round-6 revert added the cmp_001 chain, and `_pick_treatment_chain` chooses it. The trial's actual evidence (24 edge updates on the nivolumab chain) still exists but isn't headlined. |

## Findings

### 1. inspect_trial's hypothesis-picker is degenerate when an UNKNOWN-target chain sorts first

**Pattern**: NCT03618641 has 2 chains: `cmp_001` (target=UNKNOWN, immune_costimulation) and `nivolumab` (target=PDCD1, full backbone). `_pick_treatment_chain` in `scripts/inspect_trial.py:395` walks `ts.chains` in order and returns the first whose `compound_id` is non-placebo and non-UNKNOWN. It does **not** check `target_id`. cmp_001 sorts first per populator insertion order, so the trial's "headline" prediction is the degenerate one (CI [0.027, 0.975]) — even though the nivolumab chain has 24 routed edge updates and would yield P(success) ≈ 0.84.

**Severity**: MEDIUM. Doesn't break the graph; just hides the trial's real signal in the inspector. NCT00019682 and NCT00003222 also flipped hypothesis between post6 and post7 for the same reason (chain insertion order is non-deterministic across rebuilds in some cases).

**Suggested fix**: in `_pick_treatment_chain`, prefer chains with `target_id != "UNKNOWN"`; fall back to UNKNOWN-target only if no other chain exists. ~3-line patch. No test infra change needed — the function is local to inspect_trial.

### 2. Reactome top-1 biology mapping is wrong on ~50% of compounds in the slice

Full slice audit (one row per distinct (compound, target, mechanism, biology) tuple in the graph):

| Compound | Target gene | Mechanism | Reactome biology id | Reactome name | Verdict |
|---|---|---|---|---|---|
| aldesleukin | IL2RA | receptor_agonism | R-HSA-5673001 | RAF/MAP kinase cascade | **✗** (IL-2 signals via JAK/STAT5, not MAPK) |
| sargramostim | CSF2RA | receptor_agonism | R-HSA-512988 | IL-3, IL-5, GM-CSF signaling | ✓ |
| 7-OH-staurosporine | PDPK1 | kinase_inhibition | R-HSA-114604 | GPVI-mediated activation cascade | **✗** (platelet signaling, unrelated) |
| ipilimumab | CTLA4 | checkpoint_blockade | R-HSA-389356 | Co-stimulation by CD28 | ✓ |
| revlimid | CRBN | protein_degradation | R-HSA-9679191 | Potential therapeutics for SARS | **✗** (COVID-era curation) |
| nivolumab | PDCD1 | checkpoint_blockade | R-HSA-389948 | Co-inhibition by PD-1 | ✓ |
| bevacizumab | VEGFA | angiogenesis_inhibition | R-HSA-114608 | Platelet degranulation | **✗** (wrong VEGFA pathway) |
| pdr001 | PDCD1 | checkpoint_blockade | R-HSA-389948 | Co-inhibition by PD-1 | ✓ |
| lag525 | LAG3 | checkpoint_blockade | R-HSA-2132295 | MHC class II antigen presentation | ~ (related but not the checkpoint biology) |
| inc280 | MET | kinase_inhibition | R-HSA-1257604 | PIP3 activates AKT signaling | ✓ |
| lee011 | CDK6 | kinase_inhibition | R-HSA-2559580 | Oxidative Stress Induced Senescence | ~ (a downstream consequence, not the mechanism) |

**5 correct / 4 wrong / 2 dubious.** This is the same `_BIOLOGY_PATHWAY_CAP=1` issue surfaced in round 5 Finding C; round 7 just makes the scope clearer. The wrong cases all share a pattern: Reactome's default ranking puts the gene's most-cited pathway first, regardless of whether that pathway is relevant to the trial's mechanism + indication.

**Severity**: HIGH at scale. Every wrong biology_drives edge is an edge that classifier evidence flows into and **doesn't reflect actual biology**. At 1000 trials, that's compounded false-positive learning.

**Suggested fix** (none small; this is a design exercise):
- (a) Curated `(mechanism, target) → preferred_pathway` lookup for common cases.
- (b) Re-rank Reactome candidates by token overlap between pathway name and `{mechanism_name, indication_name}`.
- (c) Fall back to the synthetic `{mechanism}__{indication}` slug when no candidate has a relevance signal.

Recommendation: prototype (b) on a branch before changing populate.py. (a) is the highest-confidence fix but doesn't scale across indications.

### 3. The classifier is hypothesis-anchored, not arm-anchored — supportive arms contribute no updates

**Pattern**: NCT00019682's classifier output emits 6 `edges_to_update`:

- `affects: gp100_antigen → UNKNOWN` (and the same for tyrosinase, both bucket B — unrouted)
- `mechanism_affects: immune_costimulation → immune_costimulation__melanoma`
- `biology_drives: immune_costimulation__melanoma → melanoma`
- `reflects_biology: immune_costimulation__melanoma → composite_response_melanoma`
- `endpoint_captures: composite_response_melanoma → melanoma`

The LLM is decomposing the trial's **mechanistic hypothesis** ("gp100 augments immune costimulation in melanoma") into edge updates. Aldesleukin's chain (`aldesleukin → IL2RA → receptor_agonism → R-HSA-5673001 → melanoma`) gets **zero** trial-specific edge updates from this classification, even though arm I is aldesleukin monotherapy with its own outcome data.

So adding the aldesleukin chain post-revert structurally exists but accumulates no evidence from this trial.

**Severity**: MEDIUM-HIGH for compositional learning. The system is supposed to "decompose trial outcomes into mechanistic causal-chain updates" (CLAUDE.md). Currently it only decomposes into ONE chain's worth of updates (the trial's primary hypothesis), even when the trial has multiple mechanistically-distinct arms.

**Suggested fix** (design-level):
- (a) Modify the classifier prompt to emit per-arm edge updates. Add `affecting_arm_id` field to `edges_to_update` so the attributor can route each update to the right arm's chains.
- (b) Run the classifier N times per trial, once per arm.
- (c) Accept current behavior and only ingest hypothesis-mechanism evidence (back-compat).

Recommendation: (a). Single classifier call still; just richer output. Touches the classifier prompt + attributor schema. Real work, real value.

### 4. Per-compound AE attribution creates dense false co-occurrences in combo trials

**Pattern**: NCT00003222 has 41 `causes_ae` edge updates from a single trial. Every reported AE (7 of them: alkaline_phosphatase_increased, blood_creatinine_increased, dyspnea, hyperglycemia, insomnia, palpitations, pyrexia) gets attributed to every compound in the regimen (6 compounds × 7 AEs ≈ 41 updates; some not all combos emit).

This is the round-1 per-compound AE attribution fix doing exactly what it was designed to do: in single-arm combo trials, you can't disambiguate which compound caused which AE, so you attribute each AE to each compound. But at scale across the corpus, this creates lots of low-confidence noise edges. Aldesleukin appearing in many vaccine combos will accumulate spurious AE attributions across all of them.

**Severity**: LOW for now (each individual edge is weak, evidence_strength accumulates slowly). Could become HIGH at 1000 trials if combos dominate the corpus.

**Suggested fix** (defer until scale problem appears):
- (a) When AE rates are arm-level (multi-arm trial), attribute to arm compounds, not regimen compounds.
- (b) When trial is single-arm combo, attribute with reduced weight (e.g. `N_eff / arm.n_compounds`) so per-compound updates are softer.
- (c) Don't attribute AE edges for compounds appearing in combos with > N constituents (filter out vehicle/adjuvant/cytokine noise).

### 5. Three peptide-vaccine unrouted records persist (carryover bucket B)

```
NCT00003222 [entity_not_in_trial] affects gp100_antigen      -> UNKNOWN
NCT00003222 [entity_not_in_trial] affects tyrosinase_peptide -> UNKNOWN
NCT00019682 [entity_not_in_trial] affects gp100_antigen      -> UNKNOWN
```

The classifier honestly emits `target=UNKNOWN` because it doesn't know the target. The attributor (`src/annotation/attributor.py:906`) skips chain (src,tgt) pairs where either side is UNKNOWN — so a chain with `compound_id=gp100_antigen, target_id=UNKNOWN` doesn't accept this update either. Logged as `entity_not_in_trial`.

**Fix**: peptide-vaccine heuristic (gp100→PMEL/ENSG00000185664, tyrosinase→TYR/ENSG00000077942, MART-1→MLANA, etc.). Curated dict. ~30-line patch in `src/graph/populate.py` at the compound→target resolution point.

## Priority list for round 8

| # | Issue | Severity | Effort | Round |
|---|---|---|---|---|
| 1 | inspect_trial picks UNKNOWN-target chains as hypothesis when they sort first. | MEDIUM | Trivial (~3 lines + 1 test). | 8 — quick win, fix first. |
| 2 | Peptide-vaccine target heuristic (gp100, tyrosinase, MART-1 → curated ENSG ids). | MEDIUM | Small. | 8 — unblocks 3 unrouted + ~5 UNKNOWN-target chains. |
| 3 | Reactome biology relevance (50% wrong in slice; will scale badly). | HIGH | Medium (design + curated/heuristic re-ranking). | 8 if appetite, else 9. Highest learning impact. |
| 4 | Classifier is hypothesis-anchored, not arm-anchored — supportive arms get no evidence. | MEDIUM-HIGH | Medium (prompt + attributor schema). | 9. |
| 5 | Per-compound AE attribution density at combo scale. | LOW now, HIGH at 1000 trials. | Medium (arm-aware AE attribution). | Defer — let scale evidence drive priority. |
| 6 | Document subgroup population anchoring (round 5 Finding D). | LOW | Trivial. | Bundle with any round-8 commit. |

**Recommended round-8 scope**: #1 + #2 + start #3 on a branch. Together these make the slice's predictions usable (no more degenerate cmp_001 headlining) and address the biggest quality issue (Reactome relevance) before scaling. #4 is the bigger architectural conversation that should wait until after the corpus scales past melanoma — at scale the missing arm-level evidence will be more visible.

## Path to 1000 trials

Per the playbook (`automate_node_debug.md`), the slice should grow only when each round finds nothing new. Round 7 found 4 new findings + verified 5 carryovers. **Not ready to scale yet.**

After round 8 (which should close #1 + #2 + #3), candidate next steps for scaling:
- Bump slice to `--max-trials 30` and re-run the audit loop. Bigger archetype variety (different combos, more failure modes).
- Then `--max-trials 100` against melanoma_145.
- Then add a second indication's corpus (e.g. NSCLC_N) and run side-by-side audits per the playbook's "Switching corpora" section.
- Only after that does 1000 trials become a sensible target — and at that point the major architectural decisions (arm-level classifier, Reactome re-ranking, AE attribution policy) should already be made on smaller evidence.

## Artifacts

- `audit/inspection_*_post7.txt` (10 per-trial + 1 extremes)
- `audit/fixes_round7.md` (this file)
- No code changes this round — audit only.
