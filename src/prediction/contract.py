"""Public prediction contract — the open-core surface the build, the API, and any
consumer can import WITHOUT pulling in the (private) prediction engine.

Two kinds of thing live here, both pure schema with no engine dependency:

1. **Causal-chain edge spec** — ``_CAUSAL_CHAIN`` / ``_AUXILIARY_EDGES`` and the
   derived ``CONSUMED_BACKBONE_EDGE_TYPES``. A BUILD-TIME contract: the populator
   must PRODUCE every edge type the prediction consumes, and
   ``scripts/build_graph.py`` asserts it on every build (the phantom-edge guard).

2. **Prediction output schema** — ``EdgeContribution`` / ``SafetyRisk`` /
   ``PredictionResult``. The shape a consumer sees (the API response model, a result
   renderer, the website's frozen sample outputs), independent of the engine.

Per the governing write-path/read-path boundary (``docs/dev/reports/BOUNDARY.md``)
the prediction ENGINE is private (relocating to ``eroom-enterprise``); this contract
stays public so the build, the public API, and result-rendering keep a public symbol
to import. Dependencies flow ONE WAY: the engine imports this module; this module
imports nothing from the engine (``path_query`` / ``field_prediction`` / the
``provenance`` frontier).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.graph.models import EdgeBeliefState, EdgeType

# ── Causal-chain edge spec (build-time contract) ──────────────────────────────
# The canonical causal chain edges in order. Each entry maps a pair of
# CausalChain field names to its edge type.
_CAUSAL_CHAIN: list[tuple[str, str, EdgeType]] = [
    ("compound_id", "target_id", EdgeType.AFFECTS),
    ("target_id", "mechanism_id", EdgeType.MODULATES_VIA),
    ("mechanism_id", "biology_id", EdgeType.MECHANISM_AFFECTS),
    ("biology_id", "indication_id", EdgeType.BIOLOGY_DRIVES),
]

# Auxiliary edges that contribute to the prediction
_AUXILIARY_EDGES: list[tuple[str, str, EdgeType]] = [
    ("biology_id", "endpoint_id", EdgeType.REFLECTS_BIOLOGY),
    ("endpoint_id", "indication_id", EdgeType.ENDPOINT_CAPTURES),
    ("subgroup_population_id", "indication_id", EdgeType.RESPONDS_DIFFERENTLY),
]

# Every edge type the prediction CONSUMES along/around the chain. The populator
# must PRODUCE each of these — a type consumed here but instantiated by no
# producer is a phantom edge (the reflects_biology gap: defined + walked by the
# attributor + in the field EDGE_SPECS, but created by no populate method, so it
# never carried a belief). ``build_graph`` checks this set is non-empty per type
# on every build so the gap can't silently reopen.
CONSUMED_BACKBONE_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {et for *_, et in _CAUSAL_CHAIN} | {et for *_, et in _AUXILIARY_EDGES}
)


# ── Prediction output schema ──────────────────────────────────────────────────


class EdgeContribution(BaseModel):
    """One edge's contribution to the prediction."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    belief: EdgeBeliefState
    sampled_mean: float
    bottleneck_score: float = Field(
        description="1 - belief.mean; higher = weaker link"
    )


class SafetyRisk(BaseModel):
    """One adverse-event risk surfaced in a prediction.

    ``source`` distinguishes compound-specific risk (this drug has caused
    this AE in past trials) from target-class risk (other drugs binding
    the same target have caused this AE—likely on-mechanism). The two
    travel together so the consumer can decide whether the risk is
    chemistry-related (changeable) or mechanism-related (intrinsic).

    Round-29: ``severity_range`` carries the grade-token string used to
    derive the severity weight. For ``causes_ae`` (compound source)
    risks this comes from the PT-level AE node. For
    ``target_associated_ae`` (target_class source) risks it comes from
    the EDGE — target-scoped — because the SOC-tier AE node is shared
    across targets and a global severity_range there would leak grade
    data from one target class to another.
    """

    ae_id: str
    ae_name: str
    source: str  # "compound" | "target_class"
    belief_probability: float
    evidence_strength: float
    severity_range: str = ""
    # Round-29: did any contributing trial flag this AE as serious? For
    # ``causes_ae`` risks this is the PT node's OR-merged ``serious``
    # field. For ``target_associated_ae`` risks this is the per-target
    # SOC-tier OR-aggregation stored on the edge. Acts as a grade-3
    # severity floor in the safety-penalty math when CTCAE grade is
    # missing (the majority case).
    serious: bool = False
    # Round-30 DLT-gate: fraction of this AE's evidence from trials whose
    # failure was dose-limiting toxicity (vs AEs that merely occurred in
    # tolerated/successful trials). 1.0 = no gating (default / back-compat).
    failure_causing_fraction: float = 1.0
    contributing_compound_ids: list[str] = Field(default_factory=list)


class PredictionResult(BaseModel):
    """Full prediction for a trial hypothesis.

    Round-20: efficacy and safety are integrated into a single
    ``overall_probability``. The torcetrapib audit (ILLUMINATE,
    NCT00134264) exposed the v0.1.0 decoupling as wrong — the mechanism
    chain worked (CETP inhibition raised HDL) but the trial failed for
    off-target hypertension. Without folding safety into the headline
    number, the system scored torcetrapib at modest success and missed
    the actual failure mode.

    ``overall_probability`` is now ``efficacy_probability * (1 -
    safety_penalty)``. The breakdown is exposed so consumers can see
    where the drag came from.
    """

    trial_hypothesis: str
    # Mechanism-only chain geomean (the round-15 trust-weighted
    # aggregation). What ``overall_probability`` used to mean
    # before round 20.
    efficacy_probability: float
    # [0, 0.4] subtractive — soft-or aggregation of compound-specific +
    # target-class AE evidence. Capped to keep efficacy as the
    # dominant signal; safety acts as a drag, not the whole story.
    safety_penalty: float
    overall_probability: float
    ci_lower: float
    ci_upper: float
    edge_contributions: list[EdgeContribution]
    weakest_link: EdgeContribution | None
    n_samples: int
    # Full list of AE risks (under the more inclusive display threshold)
    # for introspection. The safety_penalty math uses a stricter
    # threshold; see ``_compute_safety_penalty``.
    safety_risks: list[SafetyRisk] = Field(default_factory=list)
