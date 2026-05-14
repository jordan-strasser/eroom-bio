# NEXT_SESSION — round 8: write the conditioning architecture design doc

## Where we left off (2026-05-13 end of session)

`main` is synced to `origin/main` at commit `f40f503` (Round 6.1: guard `predict_clinical_hypothesis` against UNKNOWN target). 560/560 tests pass. The pipeline structurally works on the 10-trial melanoma slice:

- Chain coverage: 47/48 chains (98%)
- Trials: 9 full / 1 partial / **0 zero-coverage** ✓
- All 10 trials produce P(success) predictions (no crashes)
- Snapshot at `data/exports/oncology_annotated.json`: 216 nodes, 491 edges

Round 7 was audit-only (no code changes). It produced `audit/fixes_round7.md` and identified the headline architectural question that round 8 must resolve before further shipping.

## The blocker for round 8

**Combo trials encode conditional dependence between constituents; the current per-constituent chain decomposition cannot represent it.**

Concrete example: NCT00019682 (Schwartzentruber gp100 + IL-2 vs IL-2 alone) produces independent chains for `aldesleukin`, `gp100_antigen`, `montanide_isa_51_vg`. The trial's actual scientific finding is the **differential between arm I and arm II** — conditional information about gp100 *given* a high-dose IL-2 backbone. That conditioning is invisible to the inference today.

Many smaller open issues from round 7 (Reactome biology relevance, classifier hypothesis-anchoring, headline chain picker) **depend on this decision** because the right shape of biology / chain selection / classifier output changes by architectural path. So they're paused.

## Round 8's deliverable

A written design doc at `audit/round_8_architecture_design.md` covering three paths and recommending one for implementation:

### Path 1 — Arm-level chains
One chain per (arm × subgroup × endpoint) cell instead of per constituent. Chain compound = `arm.regimen_compound_id` (combo `CompoundNode` already created today). Mechanism / target / biology resolved at the arm level.

### Path 2 — Context-conditional Beta beliefs
Keep per-constituent chains. Add a `co_compounds` context dimension to evidence records. Extends the existing `get_edge_belief_conditioned` machinery (already conditions mechanism→biology on tissue).

### Path 3 — Explicit combination edges
Keep per-constituent chains. Add new edge type `modulates_efficacy_of` between compound pairs. Combo trial evidence updates the combination edge based on arm differentials.

Full architectural analysis with pros/cons, v0.1.0 lock surface, sparsity behavior, and how each path would handle NCT00019682 + a cross-indication scenario: see the memory file `project_conditioning_question.md` in the user's Claude project memory dir.

### What the design doc must contain

For each path:
1. Concrete schema diff (which Pydantic models, which `EdgeType` values, which fields).
2. Worked example: how NCT00019682's chains, edges, and Beta updates look under the path.
3. Worked example: how `ipi+nivo` combo learning in melanoma transfers to a hypothetical `ipi+nivo` NSCLC trial under the path. (This is the cross-indication test for whether compositional learning survives.)
4. v0.1.0 architecture-lock surface touched (per CLAUDE.md).
5. Estimated implementation cost (rough — lines of code touched, new tests, migration concerns).

Then a **decision section** picking one path with reasoning. Then a branch plan (probably `arch-conditioning-v0.2-{path}`).

## Procedural notes

- **Don't ship code in round 8.** Decision first, branch second, implementation third.
- The headline-picker question that round 7 surfaced (`scripts/inspect_trial.py:401`, `eval_predictions.py:118`, `attributor.py:611`) is downstream of this decision. Don't fix it as an isolated quick-win — see memory `feedback_picker_is_symptom.md`.
- Defer-to-shippable items that are orthogonal to the conditioning decision:
  - Peptide-vaccine target heuristic (~30 lines; gp100→PMEL, tyrosinase→TYR, MART-1→MLANA). Reduces UNKNOWN-target chains, unrelated to architecture.
  - Documentation: append-only behavior of `data/dev/unrouted_attribution_updates.jsonl` in `automate_node_debug.md §3b`.
  - Documentation: subgroup population anchoring (round 5 Finding D).

## Mechanical bootstrap

```bash
git status --short                              # expect clean
git log main --oneline -5                       # expect f40f503 at top
git branch -a                                   # archive/round-4-sub-chains preserved
```

Memory to read in order:
1. `CLAUDE.md` (project goals, v0.1.0 lock)
2. `project_round7_state.md` (current state, what's safe to ship)
3. `project_conditioning_question.md` (the architectural decision; the three paths in detail)
4. `feedback_picker_is_symptom.md` (don't fix the picker in isolation)
5. `audit/fixes_round7.md` (the full round-7 audit findings)
6. This file

## Standing context the playbook assumes

- v0.1.0 architecture lock per CLAUDE.md: math, trust-weight, edge priors, aggregation, **edge topology** all frozen until corpus expands beyond melanoma. Paths 1 and 2 touch this surface significantly; Path 3 less so.
- Per `feedback_architecture_branches`: large architecture changes land on a branch first.
- Per `feedback_premature_classification`: don't add classification layers to trial decomposition before the architecture is ready. (Path 1 specifically risks this if shipped without scaffolding.)
- Per `feedback_simple_faithful`: prefer the simplest change that serves the thesis (mechanistic chain decomposition, evidence flow, compositional predictions). This is the lens for evaluating the three paths.

## Why this round matters for scaling

The user wants to scale to 1000 trials. The conditioning gap is the architectural ceiling on combo-trial learning, which dominates the cross-indication phase. Resolving it now — before the v0.1.0 lock comes off — sets up the rest of the scaling work to inherit the right structure rather than retrofitting later.

Round 7's audit explicitly said: "After round 8 (which should close the architecture decision), candidate next steps for scaling: bump slice to --max-trials 30, then 100, then add a second indication's corpus." Round 8 is the gate.
