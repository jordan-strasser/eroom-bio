# Round 4.0 design — Sequential hypothesis chains

Major architectural change deferred from round 3.3 follow-ups. Goal: model trial interventions as **a series of sub-hypotheses that combine multiplicatively** into trial-level success probability, rather than collapsing supportive interventions to "not a hypothesis" as round 3.3 does.

This is a real architecture pass, not a tweak. Probably ~1000+ LOC across schema, populate, prediction, attribution, and prompts. Worth designing on paper before coding.

## Motivation

Round 3.3 treats supportive interventions (preconditioning chemo, growth-factor support, premedications) as infrastructure: they appear in arm rosters and accumulate AE evidence, but **don't generate causal chains**. The graph can't learn anything about them. Concretely:

- Lymphodepletion (cyclophosphamide + fludarabine) appears in dozens of cell-therapy trials. Its contribution to engraftment success is invisible to the graph.
- IL-2 / aldesleukin cytokine support: same story.
- Premedications across many immunotherapy trials: same.
- Whether the combination *interaction* matters (combo > sum of monos? antagonism?) — also invisible at the edge level.

Project goal says "each trial is decomposed into a causal hypothesis chain with probabilistic beliefs populating each edge". A trial of "TCR-T cells + lymphodepletion + IL-2 cytokine support" is really three sub-hypotheses chained together, not one.

## Target architecture

Each trial decomposes into N **sub-chains**, each with its own causal backbone + Beta beliefs on each edge:

```
Supportive sub-chain A:    A → target_A → mech_A → bio_A → enables(?)
Supportive sub-chain B:    B → target_B → mech_B → bio_B → enables(?)
Active sub-chain X:        X → target_X → mech_X → bio_X → indication → endpoint

P(trial success) = P(active works) × Π P(supportive_i holds)
```

Each sub-chain's P is computed the same way today's chains compute P(success) — trust-weighted geometric mean of per-edge Beta samples. Trial-level P(success) is the product across sub-chains, under an independence assumption (refineable later).

## Schema changes

### New enum
```python
class ChainRole(str, Enum):
    ACTIVE = "active"        # The investigational intervention's chain.
    SUPPORTIVE = "supportive" # Preconditioning, cytokine support, premeds.
    ENABLER = "enabler"      # An intervention that explicitly enables
                             # another (e.g. radiation as priming for
                             # immune therapy). Subset of supportive
                             # where the mechanism is "enable downstream
                             # chain to work" specifically.
```

### CausalChain extension
```python
class CausalChain(BaseModel):
    ...
    chain_role: ChainRole = ChainRole.ACTIVE
    # When chain_role is SUPPORTIVE or ENABLER, this points at the
    # active chain in the same trial whose success this chain supports.
    # None for ACTIVE chains.
    supports_chain_id: str | None = None
```

### New edge type (maybe)
```python
class EdgeType(str, Enum):
    ...
    # Sub-chain → sub-chain linkage. The supportive chain's biology
    # "enables" the active chain's biology to function. Beta belief =
    # P(this enablement relationship is required for active to work).
    ENABLES = "enables"
```

Open question: do we need `ENABLES` as a first-class edge, or can the product-across-chains math at prediction time imply the relationship implicitly? Tentatively yes — explicit edges let us learn cross-trial "X enables Y" patterns (e.g. "lymphodepletion enables CAR-T efficacy" becomes a queryable hypothesis).

## Extraction schema changes

Replace `therapeutic_hypothesis.compound: str` (free-text regimen description) with:

```python
class TherapeuticHypothesis(BaseModel):
    primary_compounds: list[str]      # The investigational drugs/cells/etc.
    supportive_compounds: list[str]   # Preconditioning, cytokines, premeds.
    indication: str
    claimed_target: str
    proposed_mechanism: str
    intended_biology: str
    target_population: str
    primary_endpoint: str
    rationale_strength: str
```

Extractor prompt change: explicit instruction to separate primary vs supportive. Example wordings to look for as supportive markers:
- "with lymphodepletion"
- "following conditioning chemotherapy"
- "premedication with X"
- "with growth factor support"
- "preceded by X"
- "concurrent X to enable"

## Populate changes

`build_arms` + `_resolve_mechanism_per_trial` need to:
1. Iterate every intervention in each arm, not just primaries
2. Build a sub-chain per intervention (typed with ChainRole)
3. Wire supports_chain_id from supportive chains to the active chain in the same arm
4. Add `enables` edges (one per supportive → active pairing, scoped to the trial)

Chain count goes UP per trial (supportive chains are no longer dropped). NCT01218867 example: at round 3.3 it has 11 chains (CAR-T only). Round 4: 11 + 3 × 11 (lymphodepletion + IL-2 chains per arm) = ~44 chains. The cost is more graph, the benefit is supportive-drug learnability.

The chain-count metric needs a tier breakdown: "X chains total, Y active, Z supportive". The KPI shifts: rather than minimizing chains, we minimize *silent* chains within each role.

## Prediction changes

`predict()` currently walks one chain. New behavior:
1. Group the trial's chains by arm (or by hypothesis-cluster).
2. For each group: identify the active chain + its supportive chains.
3. Compute P(active) and P(supportive_i) separately using the existing per-edge Beta-sampled chain product.
4. Combine: `P(trial success) = P(active) × Π P(supportive_i)`.
5. CI propagation: same Monte Carlo sampling, just over more chains per draw.

The trust-weighted geometric mean (per CLAUDE.md spec) applies WITHIN each sub-chain. The multiplicative combination across sub-chains is the new step.

Open question: independence assumption. If lymphodepletion fails AND CAR-T engraftment fails, are those independent events? Probably not — they share patient-level confounders. For v1, assume independence; future work could add cross-chain covariance.

## Attribution changes

Classifier emits edges per sub-chain. The prompt needs to express the trial-level decomposition: "here's the active sub-chain, here are the supportive sub-chains, emit edges for each". The Resolved Graph Entities block in the user prompt gets one block per sub-chain.

Failure backstop logic (round 3.1) needs to fire per-sub-chain on a failure trial — both active and supportive sub-chains should at minimum get a `biology_drives` weak_contradict if the trial as a whole failed.

## Migration strategy

1. **Schema**: ship in models.py with backwards-compat defaults (`chain_role=ACTIVE`, `supports_chain_id=None`) so existing trial subgraphs in snapshots still validate.
2. **Extraction**: extractor prompt change requires re-extracting every trial. Bulk-delete `_extraction.json` and rebuild. Round 3.0 cost was ~$5; expect similar.
3. **Populate**: update arm-build + chain-build to honor the new `primary_compounds`/`supportive_compounds` lists.
4. **Prediction**: extend `predict()` to walk sub-chains and multiply. Add unit tests for the multiplicative combination.
5. **Attribution**: classifier prompt changes; re-classify every trial. Another ~$5.

Total budget guess: ~$10-15 in Sonnet tokens, ~2 sessions of focused work, ~5-day calendar pass.

## What this does *not* address

- Adaptive ChainRole tagging from raw text (relies on the extractor prompt being clear about the distinction)
- Quantifying combination synergy explicitly (multi-arm combo-vs-mono comparisons — separate hypothesis class)
- Patient-level confounders / cross-chain covariance (v1 assumes independence)
- Pathway-cap fan-out — that's the round 3.4 #15 separate fix

## Decision points before coding

Before round 4 lands, agree on:

1. **`enables` edge: yes or no?** First-class graph edge or implicit via product-of-chains. Recommendation: yes, first-class — lets cross-trial supportive patterns become queryable.
2. **Independence assumption explicit or hidden?** The math assumes independent sub-chains. Probably worth a doc + tests rather than a code comment.
3. **Chain count budget**: do we want a per-trial chain cap to prevent the fan-out from exploding once we add supportive chains? E.g. a 4-drug regimen with 3 subgroups × 2 endpoints = 24 active + 72 supportive chains is plausible. Cap at K total per trial?
4. **Prediction-time chain grouping**: by arm? By trial-level hypothesis cluster? The grouping determines what gets multiplied together.

These are the things to settle before any code lands. None are unrecoverable later, but each lock-in affects downstream design.
