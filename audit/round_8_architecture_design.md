# Round 8 — Architecture design: representing conditional dependence in combo trials

Date: 2026-05-14. No code in this round. Round 7's audit (`audit/fixes_round7.md`) surfaced the headline architectural question; this doc evaluates three paths and picks one.

## The question

Combo trials encode **conditional dependence** between constituents. Today's per-constituent chain decomposition (`src/graph/populate.py:1211-1251`) treats each constituent's chain as independent: `aldesleukin → IL2RA → … → melanoma` and `gp100_antigen → UNKNOWN → … → melanoma` accumulate evidence side-by-side as if the trial reported on each in isolation. The trial's actual scientific finding — the **arm-to-arm differential** — has nowhere to live.

Concrete: NCT00019682 (Schwartzentruber 2011) compares arm I (high-dose IL-2 alone) to arm II (high-dose IL-2 + gp100 + montanide). The differential is evidence about gp100 *given* an IL-2 backbone. The graph today cannot represent "gp100 helped because the IL-2 backbone existed," only "gp100 helped in this trial" and "IL-2 helped in this trial" as two independent claims. Cross-indication transfer of combo learning is the architectural ceiling on the scaling phase.

This decision is upstream of several smaller round-7 findings:
- The chain-picker question (`scripts/inspect_trial.py:395`, `scripts/eval_predictions.py:118`, `src/annotation/attributor.py:611`).
- Whether Reactome biology is resolved per-constituent or per-regimen.
- Whether the classifier emits per-arm or per-hypothesis edge updates.

All of those should be decided *after* the path here is chosen — see `feedback_picker_is_symptom`.

## v0.1.0 architecture lock (from CLAUDE.md)

> The prediction math, trust-weight function (log-scaled, saturation at evidence_strength=49), edge priors, aggregation method, and **edge topology** are frozen at v0.1.0 until the corpus expands beyond melanoma.

Round 8 is also part of the lock-removal decision. The conditioning gap is exactly the kind of architectural blocker the lock was designed to defer until a deliberate review. This doc *is* that review.

---

## Path 1 — Arm-level chains

**Idea.** Replace per-constituent chain fan-out with one chain per (arm × subgroup × endpoint) cell. The chain's `compound_id` is the arm's `regimen_compound_id` (the synthesized combo `InterventionNode` already created in `synthesize_combo_compounds`, `src/graph/populate.py:1907`). Target, mechanism, and biology are resolved at the arm level — one LLM inference per arm.

### Schema diff

`src/graph/populate.py:1211-1251` (the chain rebuild loop in `_populate_trial_mechanisms`):
- Remove the per-constituent fan-out. For every (arm × subgroup × endpoint) cell emit exactly one chain whose `compound_id = arm.regimen_compound_id`.
- Drop the `is_constituent` / `regimen_id` chain metadata; the chain *is* the regimen.

`src/graph/models.py:482-496` (TrialArm):
- No schema change. `compound_ids` is preserved as the list of constituents.

`src/graph/models.py:499-524` (CausalChain):
- No schema change. `compound_id` field range now legitimately includes synthesized combo ids — which it nominally already does for mono arms whose `regimen_compound_id == compound_ids[0]`.

`synthesize_combo_compounds` (`src/graph/populate.py:1907-1980`):
- Combo `InterventionNode` already inherits `affects` edges from each constituent at `Beta(3, 1.5)`. Keep that — it's the regimen-level target attachment point.
- Need a per-regimen `MechanismNode` and `BiologyNode` resolution step. New code path: take the dominant constituent's mechanism (e.g. the most-cited primary target), or run an LLM mechanism-inference at the regimen level. The latter is fuzzy ("what is the mechanism of `aldesleukin + gp100 + montanide`?").

AE attribution (`src/annotation/attributor.py:652+`) stays per-constituent — that mechanism is independent.

### How NCT00019682 looks

```
arm I (aldesleukin alone):
  chain: aldesleukin → IL2RA → receptor_agonism → R-HSA-... → melanoma

arm II (aldesleukin + gp100 + montanide):
  chain: aldesleukin+gp100_antigen+montanide_isa_51_vg
         → ???-target → ???-mechanism → ???-biology → melanoma
```

The arm-level differential is now directly representable: arm I and arm II are two distinct chains with different `regimen_compound_id`s. Trial-level scoring is well-defined per chain.

But the arm II chain's middle nodes are degenerate. There is no single "target" of `aldesleukin + gp100 + montanide`. An LLM might pick `PMEL` (the gp100 antigen) and label the mechanism `immune_costimulation`, but then aldesleukin's `IL2RA → receptor_agonism` contribution is invisible in the arm II chain — it lives only in arm I's chain.

### Cross-indication transfer: ipi+nivo melanoma → ipi+nivo NSCLC

Today's per-constituent decomposition: melanoma's ipi+nivo trial updates `ipilimumab → CTLA4 → checkpoint_blockade → … → melanoma` and `nivolumab → PDCD1 → checkpoint_blockade → … → melanoma`. A first NSCLC ipi+nivo trial inherits the upstream beliefs on `CTLA4 → checkpoint_blockade` and `PDCD1 → checkpoint_blockade` — those edges already accumulated evidence from the melanoma trial.

Path 1: melanoma's ipi+nivo trial updates a chain rooted at `ipilimumab+nivolumab → ???-target → … → melanoma`. The NSCLC trial's chain is rooted at the **same** `ipilimumab+nivolumab` regimen node, so the upstream edges from that combo node transfer. **But** the per-constituent `CTLA4 → checkpoint_blockade` edge does not get updated by the combo trial — it only gets evidence from the (rarer) mono ipi trials. The compositional thesis breaks: ipi-alone NSCLC inherits less ipilimumab evidence because the melanoma combo trial bypassed ipilimumab's edges entirely.

### v0.1.0 lock surface — HIGH

- Edge topology semantically changes — chain backbones now route through combo regimen nodes.
- `affects` edge semantics: regimen → "primary target of regimen" is a new semantic, not just inheritance.
- The frozen mechanism (single chain backbone per (compound, target, mechanism, biology, indication) tuple) loses one of its axes (compound is now sometimes a regimen).

### Pros / cons

| Pro | Con |
|---|---|
| Honest about what trials actually measure (arm-level outcomes). | **Breaks compositional learning** — the project's core thesis. Cross-indication transfer of combo evidence to mono constituents collapses. |
| Eval scoring is well-defined (per-arm). | Combo regimen nodes are unique (`A+B+C` rarely repeats exactly across trials) → weak Betas forever for combo chains. |
| Backbone construction simpler (1 chain per cell). | The "primary target of a regimen" step is fuzzy — what is the target of `aldesleukin + gp100 + montanide`? |
| | Loses constituent-level AE/efficacy connection (regimens don't share targets the way constituents do). |

### Implementation cost

- `_populate_trial_mechanisms`: rewrite the chain rebuild loop (~50 lines down to ~25 lines, but plus a new regimen-target resolver).
- Regimen-level mechanism/target/biology resolver: ~100-200 lines (new LLM inference path or a primary-constituent heuristic; both have problems).
- Attributor routing: the `_route_to_chain_edge` logic (`src/annotation/attributor.py:860+`) needs to handle classifier output that may reference constituent compound names against chains keyed by regimen ids.
- Migration: every existing trial subgraph in `data/exports/oncology_annotated.json` needs re-population. Cached classifications survive; chain re-build is a 1-time cost.
- New tests: ~5-10 (chain shape on combo arms, target resolution, attribution routing on combo arms).

---

## Path 2 — Context-conditional Beta beliefs

**Idea.** Keep per-constituent chains. Add a context dimension to `EvidenceRecord`: when applying evidence from a combo trial, tag it with `co_compounds: list[str]` — the other compounds present in the same arm. At sample time, partition evidence by co-compound context and let the caller specify which context applies.

The infrastructure already half-exists: `GraphStore.get_edge_belief_conditioned` (`src/graph/store.py:85`) already conditions `mechanism_affects` on indication tissues by re-playing evidence records with per-record weight based on `EvidenceRecord.context["tissue"]`. Extending the same machinery to `co_compounds` is a generalization, not a new mechanism.

### Schema diff

`src/graph/models.py:406-434` (EvidenceRecord):
- `context` already exists as `dict[str, Any]`. No schema change needed — we just start populating `context["co_compounds"]` in the attributor.

`src/annotation/attributor.py:540-547` (EvidenceRecord construction):
- When emitting evidence from a combo trial, set `evidence.context["co_compounds"] = sorted(arm.compound_ids - {emitting_compound_id})`.

`src/graph/store.py:85-132` (`get_edge_belief_conditioned`):
- Generalize the signature: accept an arbitrary context filter, not just `relevant_tissues`. New signature roughly `get_edge_belief_conditioned(src, tgt, edge_type, context_filter: dict[str, Any], off_context_weight: float = 0.3)`.
- The replay loop becomes: for each evidence record, compute `weight = 1.0` if the record's context matches the filter on every specified axis, else `off_context_weight`.
- Backwards-compat: the `relevant_tissues` parameter can stay as a sugar over `context_filter={"tissue": tissues}`.

`src/prediction/path_query.py:380-413` (`_collect_edges`):
- For combo predictions, the caller specifies which co-compounds are present; the engine uses the conditioned belief.
- `predict_clinical_hypothesis` (`src/prediction/path_query.py:523`) gains an optional `context` param.

### How NCT00019682 looks

aldesleukin's `mechanism_affects: receptor_agonism → R-HSA-... ` edge:
- From arm I outcome: 1 evidence record with `context["co_compounds"] = []`
- From arm II outcome: 1 evidence record with `context["co_compounds"] = ["gp100_antigen", "montanide_isa_51_vg"]`

gp100's `mechanism_affects: immune_costimulation → ...` edge:
- From arm II outcome: 1 evidence record with `context["co_compounds"] = ["aldesleukin", "montanide_isa_51_vg"]`

When predicting "aldesleukin monotherapy in melanoma", the engine samples aldesleukin's edges with filter `{"co_compounds": []}` — only the arm I record contributes at full weight; arm II's record contributes at `off_context_weight` (e.g. 0.3).

### Cross-indication transfer: ipi+nivo melanoma → ipi+nivo NSCLC

Per-constituent chains preserved. Melanoma's ipi+nivo trial updates:
- `ipilimumab → CTLA4 → checkpoint_blockade → … → melanoma` with `context["co_compounds"] = ["nivolumab"]`
- `nivolumab → PDCD1 → checkpoint_blockade → … → melanoma` with `context["co_compounds"] = ["ipilimumab"]`

NSCLC's ipi+nivo trial then inherits the upstream beliefs on `CTLA4 → checkpoint_blockade` and `PDCD1 → checkpoint_blockade` (no indication-specific filter on those edges in the current schema). Prediction for ipi+nivo NSCLC samples those edges with `{"co_compounds": ["nivolumab"]}` and `{"co_compounds": ["ipilimumab"]}` filters — melanoma's records match at full weight, mono-ipi-in-melanoma records contribute at `off_context_weight`.

Compositional learning is preserved on paper. In practice, most context cells are empty: `aldesleukin given [gp100, montanide]` matches no other trial's evidence at full weight, so the conditioned belief falls back to marginal-or-near-marginal almost always.

### v0.1.0 lock surface — HIGH

- The math doesn't change shape (still Beta-Binomial conjugate), but the **aggregation** function gains a context-filter argument — the caller now specifies which partition of evidence to sample.
- Trust-weight calculation: if the conditioned belief has fewer effective records than the marginal, `evidence_strength` drops, which drops trust weight. This is correct (less evidence under context → less trust) but changes prediction behavior in ways that need new tests.
- Backwards-compat is straightforward: empty `context_filter` reproduces today's `get_edge_belief` exactly. Existing tests should pass unchanged.

### Pros / cons

| Pro | Con |
|---|---|
| **Compositional learning preserved** — chains stay per-constituent; cross-indication transfer of mono evidence works the same as today. | **Sparsity nightmare.** Most conditional cells are empty; predictions fall back to marginals or near-marginals constantly. Conditional structure exists but rarely fires. |
| Conditioning is explicit and queryable. | Combinatorial explosion: `gp100 given aldesleukin` vs `gp100 given aldesleukin + montanide` vs `gp100 given aldesleukin + ifa` — each is a different cell. |
| Extends machinery that already exists (`get_edge_belief_conditioned`). | Smoothing rules between partitions need design. Off-context weight 0.3 is a guess. |
| Extensible to line-of-therapy, biomarker status, prior treatment, etc. — generalizes beyond combos. | Serialization of contexted evidence is non-trivial at scale (every record carries a dict; lots of redundant keys). |
| | Prediction API gets messier (caller must specify context). |

### Implementation cost

- Generalize `get_edge_belief_conditioned`: ~30 lines refactor.
- Attributor: populate `evidence.context["co_compounds"]` at emission time. ~20 lines.
- Path query: thread context filter through `_collect_edges` and `predict_clinical_hypothesis`. ~30 lines + decisions about defaults.
- Tests: ~15 (new conditioned-retrieval cases, prediction-API regression).
- Migration: existing evidence records have empty `context`; they're treated as context-free and apply at full weight to every filter, which is the conservative back-compat behavior.

---

## Path 3 — Explicit combination edges (layer-aware)

**Idea.** Keep per-constituent chains. Add a new edge type — `MODULATES_EFFICACY_OF` — between **nodes in the causal chain**, not just between compounds. The edge's Beta represents "compound A's presence in the regimen modulates the efficacy of node X in compound B's chain." When a combo trial includes a monotherapy comparator (arm I = `A` vs arm II = `A + B`), the arm-to-arm differential updates a modulation edge at whatever chain layer biology says the interaction operates. When a combo trial has no monotherapy comparator, per-constituent updates flow as today plus a weaker update to the modulation edge(s).

**Why layer-aware, not compound-to-compound.** Real combo pharmacology operates at biologically distinct layers:
- **Compound → Target** edges: PK interactions, competitive binding, displacement, exposure changes ("aldesleukin enhances gp100's binding effectiveness at PMEL").
- **Target → Mechanism** edges: signaling crosstalk at the target-mechanism interface ("CTLA4 blockade primes the T-cell pool that PDCD1 blockade can then unleash").
- **Mechanism → Biology** edges: convergent pathway modulation ("checkpoint blockade and IL-2 receptor agonism converge on antigen-driven cytotoxicity").

A compound-to-compound edge collapses all three into a single black-box "did adding B help?" signal. That works operationally but discards the mechanistic information the rest of the graph is built to carry. The layer-aware version places each modulation at the chain node where the biology actually happens — which is also where cross-indication transfer is sharpest (a `mechanism_affects → biology` modulation transfers to every indication that uses that biology).

**Empirical expectation.** Most modulation edges will likely land at `compound → target` (PK/binding) and `mechanism → biology` (pathway convergence); `target → mechanism` will be rarer and trickier to identify.

### Schema diff

`src/graph/models.py:146-180` (EdgeType):
- Add `MODULATES_EFFICACY_OF = "modulates_efficacy_of"`.
- Edge endpoints typed across `{Compound, Target, Mechanism, Biology}` (not pinned to compound-compound). Validation rule: src and tgt may be heterogeneous (the typical case is `Compound → Target` or `Mechanism → Biology`); same-type edges are allowed.
- Direction convention: source is the modulator; target is the chain node whose efficacy is modulated. For symmetric same-layer cases (e.g. compound-compound when no chain layer is identified yet), store one direction by lex order of src/tgt ids and read symmetrically.

`src/graph/models.py:472-477` (GraphEdge):
- No schema change. Standard `EdgeBeliefState` works.

`src/annotation/attributor.py`:
- New code path for emitting modulation edges. Trigger condition: trial has ≥2 arms where one is a strict subset of another (`A` vs `A + B`). Compute the differential: arm II outcome bucket minus arm I outcome bucket, mapped to a support bucket.
- **v0.2.0 emission default**: lacking a layer-resolution signal, emit at compound-compound. This is the operational fallback; the schema reserves room for the principled version.
- Even without a monotherapy comparator (single-arm combo trial), emit weaker `ambiguous`/`weak_*` updates to each pairwise modulation edge.
- All modulation evidence carries `EvidenceRecord.context["indication"]` so future indication-conditioning is non-breaking.

`src/prediction/path_query.py`:
- Treat each modulation edge as a standard edge contribution alongside `affects`, `mechanism_affects`, etc. The aggregation in `_aggregate_samples` (`src/prediction/path_query.py:63`) already handles arbitrary edge counts via trust-weighted geomean — no new math.
- When predicting on a regimen, pull in modulation edges whose src/tgt both appear in the regimen's union of constituent chain nodes.

### Resolution: where does the edge land?

The hard architectural question is **how the attributor decides which chain layer the modulation belongs at**. Three flavors, in order of mechanistic ambition. The branch plan stages them so we can ship a working v0.2 without locking in the harder decisions.

1. **Compound-compound default (v0.2.0).** No layer resolution. Every arm-differential modulation lands at the compound-compound layer. Honest about what we know from arm outcomes alone. Schema is already general — these edges live in the same table as later, sharper edges.
2. **Heuristic resolver (v0.2.1).** If both constituents' chains converge on a shared downstream node (e.g. both resolve to `immune_costimulation` mechanism, or both biology nodes share a Reactome ancestor), promote the modulation edge to that deepest shared node. Cheap, no LLM, but only fires when chains are well-resolved and biologically adjacent. **Depends on round-7 Finding #5 (peptide-vaccine target heuristic) being landed first** — without resolved targets for vaccine constituents, there's nothing for the resolver to anchor to in trials like NCT00019682.
3. **LLM modulation-mapping classifier (v0.3.0).** Add an extractor step that, given a combo arm's constituents and their chains, proposes layer-specific modulation hypotheses with cited biology. Output schema: `{modulator_compound_id, modulated_node_id, modulated_layer, direction, hypothesis, citation}`. Richest, generalizes to any chain layer, but expands the LLM's mechanistic-claim surface area — per `feedback_premature_classification`, this is the kind of layer that should land deliberately with its own audit loop, not bundled.

The schema (one new edge type, layer-general) is the same across all three stages. What changes is the attributor's emission logic.

### How NCT00019682 looks (under v0.2.0)

aldesleukin chain: unchanged. Receives evidence from both arm I (direct) and arm II (per-constituent fan-out, as today).
gp100 chain: unchanged. Receives evidence from arm II.
**NEW (v0.2.0)**: `gp100_antigen modulates_efficacy_of aldesleukin` edge at compound-compound layer gets an update based on the arm I vs arm II outcome differential. If arm II OS > arm I OS → `moderate_support`; if arm II OS = arm I OS → `ambiguous`; if arm II OS < arm I OS → `contradict`.

Under v0.2.1, if both chains resolved to `immune_costimulation`, the edge would be promoted to `aldesleukin_mechanism → immune_costimulation` or similar (the exact promotion rule needs design).

Under v0.3.0, the LLM might emit two more precise edges: `aldesleukin modulates_efficacy_of (gp100_antigen → PMEL)` (IL-2 enhances antigen presentation, sharpening binding effectiveness) and `gp100_immune_costim modulates_efficacy_of (IL2RA → receptor_agonism)` (antigen presence amplifies IL-2-driven T-cell expansion).

When predicting `aldesleukin + gp100 + montanide → melanoma`, the engine folds all applicable modulation edges into the trust-weighted geomean alongside per-constituent chain edges. When predicting `aldesleukin → melanoma` (mono), only aldesleukin's chain contributes — modulation edges aren't pulled in.

### Cross-indication transfer: ipi+nivo melanoma → ipi+nivo NSCLC

Per-constituent chains preserved. Melanoma's ipi+nivo trial under v0.2.0:
- Updates `ipilimumab → CTLA4 → … → melanoma` (per-constituent, as today)
- Updates `nivolumab → PDCD1 → … → melanoma` (per-constituent, as today)
- Updates `nivolumab modulates_efficacy_of ipilimumab` (compound-compound modulation edge)

NSCLC ipi+nivo trial inherits:
- ipi's `CTLA4 → checkpoint_blockade` upstream belief (same as today's compositional transfer).
- nivo's `PDCD1 → checkpoint_blockade` upstream belief (same as today).
- The **modulation edge belief** from melanoma — this is the architectural payoff. "ipi+nivo synergy" learning transfers across indications, even though no NSCLC ipi+nivo trial has been observed.

Under v0.2.1 / v0.3.0, the modulation might land on `checkpoint_blockade → T-cell-effector` biology instead of compound-compound. That's *sharper* transfer: any future regimen whose constituents converge on the same biology — not just ipi+nivo specifically — would inherit the modulation. Compound-compound learning *and* mechanism/biology-level modulation learning both transfer, and they transfer separately.

### Caveats

- **Indication-conditioning deferred.** Modulation edges are indication-agnostic in v0.2.0. That's a strong assumption — "ipi+nivo synergy" might genuinely differ between tumor microenvironments. The honest version applies Path 2's context mechanism to this one edge type: each evidence record carries `context["indication"]`, and prediction samples conditioned on the queried indication. v0.2.0 tags the context but doesn't condition on it; lifting the condition is non-breaking.
- **Pairwise restriction.** 3-way combos (ipi + nivo + relatlimab) get decomposed into pairs. Genuinely 3-way non-decomposable interactions can't be represented. Same reductionist tradeoff as the rest of the system.
- **Heuristic resolver assumes well-resolved chains.** Won't fire on chains with `UNKNOWN_target` — peptide-vaccine target heuristic is the structural prerequisite.

### v0.1.0 lock surface — MEDIUM

- New edge type added; existing edges keep their semantics.
- Edge typing widens to allow heterogeneous endpoints across `{Compound, Target, Mechanism, Biology}` — bigger than a pure compound-compound edge, but no existing edge's behavior changes.
- Prediction math: aggregation (trust-weighted geomean) unchanged. The new edge type slots into the existing edge contribution list.
- Edge topology is technically frozen, but additive — closer to a new layer than to a rewrite.

### Pros / cons

| Pro | Con |
|---|---|
| **Compositional learning preserved** (per-constituent chains untouched). | Pairwise restriction — 3-way interactions can't be expressed natively. |
| **Mechanistically faithful.** Modulation lands where biology says it operates, not at a black-box compound pair. | Layer resolution is the hard part — v0.2.0 ships the easy fallback; the principled version waits on v0.2.1 / v0.3.0. |
| **Cross-indication transfer is sharpest at mechanism/biology layers.** Layer-aware edges generalize beyond the specific compound pair. | LLM modulation classifier (v0.3.0) expands hallucination surface — per `feedback_premature_classification`, needs its own audit. |
| Combo evidence has a natural, queryable home — "what biology layers does checkpoint blockade modulate?" becomes a graph query. | Modulation edges are also sparse, but bounded by chain-node count and structurally tractable. |
| Smallest v0.1.0 lock surface that still admits the principled architecture — one new edge type, schema general. | Same-direction convention needed for symmetric (compound-compound) cases. |
| Aggregation reuses existing trust-weighted geomean. | If a combo trial has no monotherapy comparator, modulation edges get only weak signal. |
| Forward-compatible: v0.2.0's compound-compound edges live in the same schema as v0.2.1's promoted edges and v0.3.0's LLM-extracted edges. No migration required between stages. | |

### Implementation cost (staged)

**v0.2.0 (this branch)**:
- Add `EdgeType.MODULATES_EFFICACY_OF` + heterogeneous-endpoint validation + lex-order canonicalization helper: ~15 lines.
- Attributor: arm-differential emission at compound-compound layer. ~80-120 lines.
- Attributor: weaker single-arm combo emission. ~30 lines.
- Attributor: tag `context["indication"]` on every modulation evidence record. ~5 lines.
- Path query: include modulation edges in `_collect_edges` for regimen predictions. ~30 lines.
- Tests: ~10-15.

**v0.2.1 (follow-up branch)**:
- Heuristic resolver: shared-downstream-node detection across constituent chains. ~60 lines.
- Promotion rule (lift compound-compound edge to deepest shared chain node, or emit new layer-specific edge). Design pending.
- Tests: ~10.

**v0.3.0 (later, gated)**:
- LLM modulation-mapping extractor step. ~150-300 lines plus prompt + audit harness.
- Wire output into attributor emission. ~50 lines.
- Tests + eval slice.

**Migration**: existing trial subgraphs have no modulation edges. Optional retro-fit over `oncology_annotated.json` from cached classifications; could defer to next full rebuild.

---

## Scoring matrix

| Dimension | Path 1 (arm chains) | Path 2 (context Betas) | Path 3 (layer-aware modulation edges) |
|---|---|---|---|
| v0.1.0 lock impact | HIGH | HIGH (math signature) | MEDIUM (new edge type only) |
| Compositional preservation | **Lost** | Preserved on paper | Preserved |
| Cross-indication transfer of mono evidence | Poor | Strong | Strong |
| Cross-indication transfer of combo evidence | Poor | Weak (sparse cells) | Strong (modulation edges transfer; sharpest at mechanism/biology layers) |
| Conditioning honesty | Excellent | Excellent (in theory) | Good (v0.2.0) → Excellent (v0.3.0 LLM mapping) |
| Combo trial scoring | Natural | Complex (caller specifies context) | Natural |
| Sparsity behavior | Very high (regimens unique) | Highest (combinatorial cells) | Bounded; sharper at higher chain layers |
| Implementation complexity | Medium-High (regimen target resolver is fuzzy) | High (API surface change) | Medium v0.2.0; staged for v0.2.1 / v0.3.0 |
| Generalizes beyond combos | No | Yes (line of therapy, biomarker, etc.) | Partially (modulation edges on biology layer subsume some Path-2 cases) |

---

## Decision

**Path 3 (layer-aware modulation edges), staged.** Reasoning:

1. **Compositional thesis preserved.** Per-constituent chains stay intact; modulation is an additive layer, not a rewrite. Path 1 sacrifices the thesis; Path 2 nominally preserves it but combinatorial-cell sparsity collapses it in practice.
2. **Mechanistically faithful, not black-box.** Compound-compound is a fallback, not the target. The schema reserves the right shape — modulation lands at the chain node where biology actually operates (most often `Compound → Target` or `Mechanism → Biology`). This is more faithful to CLAUDE.md's "decompose trial outcomes into mechanistic causal-chain updates" than a flat pair edge.
3. **Smallest v0.1.0 lock perturbation that still admits the principled architecture.** One new edge type, no math change, no signature change to `predict_clinical_hypothesis`. Paths 1 and 2 each touch the math or the chain-construction loop.
4. **Cross-indication transfer of combo learning is the architectural payoff** — and it's sharpest at the higher chain layers. A modulation on `Mechanism → Biology` transfers to every regimen that converges on that biology, not just to the same compound pair. Paths 1 and 2 don't deliver this.
5. **Bounded sparsity.** Modulation edges are sparse but bounded by chain-node count, and at mechanism/biology layers many edges will see repeat evidence across regimens that share mechanisms.
6. **Stageable without architectural debt.** v0.2.0 ships compound-compound emission with the general schema. v0.2.1 adds a heuristic resolver. v0.3.0 adds an LLM modulation-mapping classifier. All three stages write to the same edge type; no migration between stages.
7. **Faithful to `feedback_simple_faithful` and `feedback_premature_classification`.** The simplest change that serves the thesis at v0.2.0, with the LLM classifier deliberately deferred to its own audit loop rather than bundled with the architectural change.

**The honest caveats** (known followups, not blockers):
- Pairwise restriction. Acceptable — every existing combo trial in the slice decomposes into pairs.
- Indication-conditioning deferred. `context["indication"]` is tagged on every modulation evidence record from v0.2.0 forward; lifting the condition later (Path 2 mechanism applied to one edge type) is non-breaking.
- Layer resolution is the real architectural work. v0.2.0 admits we don't have it yet and emits at compound-compound; v0.2.1 attempts a heuristic; v0.3.0 puts the LLM to work.
- Same-layer (compound-compound, or chain-node-of-same-type) edges need a canonical direction — use lex order of node ids and read symmetrically.

## Branch plan

### Prerequisite on `main` (separate, ships first)

**Round-7 Finding #5: peptide-vaccine target heuristic** (~30 lines). gp100→PMEL, tyrosinase→TYR, MART-1→MLANA. Lands on `main` independently of the architecture branch. Why first: the v0.2.1 heuristic resolver needs resolved targets to anchor against. Without this, NCT00019682's gp100 chain stays `UNKNOWN_target` and the resolver has nothing to work with. Tests + small slice rebuild + verify reduced UNKNOWN-target count.

### Branch: `arch-conditioning-v0.2-modulation-edges`

Sequencing (each item a separate commit; verify slice predictions after each):

1. **Add `EdgeType.MODULATES_EFFICACY_OF`** with heterogeneous-endpoint typing across `{Compound, Target, Mechanism, Biology}` + lex-order canonicalization helper. Enum validation tests. (~15 lines.)
2. **Attributor: arm-differential emission** at compound-compound layer for trials with a monotherapy comparator. Tests on NCT00019682 (arm I = `aldesleukin`, arm II = `aldesleukin + gp100 + montanide` → emit one modulation edge per non-aldesleukin constituent, lex-ordered). (~80 lines + tests.)
3. **Attributor: single-arm-combo weaker emission.** Tests on NCT00003222 (single-arm 6-compound combo → 15 pair updates at `ambiguous`/`weak_*`). (~30 lines + tests.)
4. **Tag `context["indication"]` on modulation-edge evidence** so a future indication-conditioning refinement is non-breaking. (~5 lines.)
5. **Path query: include modulation edges** in `_collect_edges` when the queried compound is a regimen or when the caller passes a combo context. Tests on combo prediction. (~40 lines + tests.)
6. **Rebuild on the 10-trial slice; re-run audit loop.** Verify: (a) NCT00019682's modulation edges have meaningful beliefs, (b) per-constituent chain coverage stays at 47/48, (c) all 10 predictions still succeed, (d) UNKNOWN-target count is lower thanks to the peptide heuristic. No new audit findings on the slice.
7. **Bring back the headline-picker question.** With the architecture settled, the inspect_trial chain choice rule becomes well-defined (prefer the chain whose `compound_id` matches the trial's primary hypothesis OR the regimen-level prediction wired through modulation edges). Small follow-up off the same branch or `main`.

**Merge gate**: slice audit clean, all tests passing, snapshot diff inspected, user reviews on the branch before merge per `feedback_architecture_branches`.

### Follow-up branches (deferred, not blocking v0.2.0)

- **`arch-conditioning-v0.2.1-layer-resolver`**: heuristic resolver promotes compound-compound edges to deepest shared chain node when constituent chains converge. Depends on biology resolution quality (round-7 Finding #2 may be a partial prerequisite).
- **`arch-conditioning-v0.3-modulation-classifier`**: LLM modulation-mapping extractor step. Its own audit loop per `feedback_premature_classification`. Output schema: `{modulator_compound_id, modulated_node_id, modulated_layer, direction, hypothesis, citation}`. Wires into attributor emission alongside (not replacing) v0.2.0's arm-differential path.

## What this doc does NOT decide

These are downstream of Path 3 and become tractable once v0.2.0 lands:

- **Reactome biology re-ranking** (round-7 Finding #2). v0.2.0 keeps biology per-constituent, so re-ranking can proceed without architectural blockage. May become a partial prerequisite for v0.2.1's heuristic resolver depending on biology-node quality.
- **Classifier per-arm edge emission** (round-7 Finding #3). v0.2.0 derives modulation edges from arm outcome data structurally, so per-arm classifier output is not required. Round 10+ candidate, scale-driven. (v0.3.0 picks this back up as the LLM modulation-mapping classifier.)
- **Per-compound AE attribution density** (round-7 Finding #4). Orthogonal to conditioning. Defer until scale evidence demands.
- **Indication-conditioning of modulation edges.** Tagged but not applied in v0.2.0. Lifting the condition is a small follow-up applying Path 2's mechanism to one edge type — bounded scope, clear payoff.

## Why this round matters

The conditioning gap is the architectural ceiling on combo-trial learning, which dominates the cross-indication scaling phase. Path 3 (layer-aware, staged) resolves it with the smallest possible perturbation to v0.1.0's frozen surface — one new edge type with schema room for layer-specific landings — and unlocks the bump-to-30-trials → bump-to-100-trials → second-indication trajectory described in `audit/fixes_round7.md`'s "Path to 1000 trials" section. The v0.2.0 stage ships a working, schema-correct version immediately; v0.2.1 and v0.3.0 sharpen the mechanistic resolution without architectural rework. Round 8 is the gate; this is the call.
