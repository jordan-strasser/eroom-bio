"""Phase D — validate trust-weight + edge-drop semantics under round 25.

The round-25 architecture stops hand-setting non-Beta(1,1) priors in the
populator and instead emits EvidenceRecords with DATABASE_* source types.
This file tests that the existing drop rule (evidence_strength <= 0 in
src/prediction/path_query.py:_collect_edges) still does the right thing
under those semantics:

  - heuristic-seed edges (no records, Beta(1,1)) → drop
  - curated-only edges (one DATABASE_* record) → kept, lower trust
  - trial-evidence edges → kept, higher trust
  - mixed → trial evidence dominates

These are integration-style tests that build small graphs and run
predict_clinical_hypothesis directly, so a future refactor of the drop
rule has to keep these semantics or change them on purpose.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.graph.curated_evidence import (
    belief_from_curated_record,
    make_curated_record,
)
from src.graph.models import (
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointClass,
    EndpointNode,
    EndpointType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    IndicationNode,
    MechanismCategory,
    MechanismNode,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
)
from src.graph.store import GraphStore
from src.inference.beliefs import (
    SupportBucket,
    apply_virtual_evidence,
    p_obs_for_bucket,
)
from src.graph.models import CausalChain
from src.prediction.path_query import (
    PredictionEngine,
    _trust_weight,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def empty_chain_graph() -> GraphStore:
    """A graph with the 5 chain nodes wired in, but every edge starts
    Beta(1, 1) with no evidence records. Caller is expected to attach
    edges (curated, trial, or both) per test scenario."""
    g = GraphStore()
    g.add_node(CompoundNode(
        id="cmpd", name="TestDrug", modality=Modality.SMALL_MOLECULE,
    ))
    g.add_node(TargetNode(
        id="tgt", name="TestTarget", gene_symbol="TGT",
    ))
    from src.graph.models import MechanismType
    g.add_node(MechanismNode(
        id=MechanismCategory.KINASE_INHIBITION.value,
        name="kinase inhibition",
        mechanism_type=MechanismType.INHIBITION,
    ))
    from src.graph.models import BiologyNode
    g.add_node(BiologyNode(id="bio", name="test pathway"))
    g.add_node(IndicationNode(id="ind", name="test indication"))
    g.add_node(EndpointNode(
        id="end", name="overall survival",
        endpoint_class=EndpointClass.OS,
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(
        id="ind__unselected", name="ind unselected",
    ))
    return g


def _heuristic_edge(src: str, tgt: str, etype: EdgeType) -> GraphEdge:
    """An edge with Beta(1, 1) and no evidence records — equivalent to
    a round-25 heuristic seed."""
    return GraphEdge(
        source_id=src, target_id=tgt, edge_type=etype,
        belief=EdgeBeliefState(),
    )


def _curated_edge(
    src: str, tgt: str, etype: EdgeType,
    *, bucket: SupportBucket, source_type: EvidenceType,
) -> GraphEdge:
    """An edge whose belief carries one DATABASE_* curated record."""
    record = make_curated_record(
        source_id=f"{source_type.value}:{src}__{tgt}",
        bucket=bucket,
        source_type=source_type,
    )
    return GraphEdge(
        source_id=src, target_id=tgt, edge_type=etype,
        belief=belief_from_curated_record(record),
    )


def _trial_edge(
    src: str, tgt: str, etype: EdgeType,
    *, bucket: SupportBucket = SupportBucket.STRONG_SUPPORT,
    source_type: EvidenceType = EvidenceType.CLINICAL_PHASE3,
    n_trials: int = 1,
) -> GraphEdge:
    """An edge built from n_trials trial-attributed records via the
    real apply_virtual_evidence path."""
    belief = EdgeBeliefState()
    records: list[EvidenceRecord] = []
    for i in range(n_trials):
        from src.inference.beliefs import effective_n_for_evidence
        rec = EvidenceRecord(
            source_id=f"NCT{i:08d}",
            source_type=source_type,
            support=bucket.value,
            quality_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )
        n_eff = effective_n_for_evidence(source_type, rec.quality_score)
        p_obs = p_obs_for_bucket(bucket)
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
        records.append(rec)
    belief.evidence = records
    return GraphEdge(
        source_id=src, target_id=tgt, edge_type=etype, belief=belief,
    )


def _attach_chain(graph: GraphStore, *edges: GraphEdge) -> None:
    for e in edges:
        graph.add_edge(e)


# ── Tests ───────────────────────────────────────────────────────────────


class TestHeuristicSeedEdgesDrop:
    """Heuristic-seed edges (Beta(1,1), no records) should not contribute
    to the trust-weighted geomean. Round-25 verifies the existing drop
    rule (evidence_strength <= 0) handles this case correctly without
    any code change."""

    def test_heuristic_edge_evidence_strength_zero(self):
        e = _heuristic_edge("cmpd", "tgt", EdgeType.AFFECTS)
        assert e.belief.evidence_strength == 0.0
        # And trust weight is exactly 0 for such an edge.
        assert _trust_weight(e.belief) == 0.0

    def test_chain_with_only_heuristic_edges_drops_all(
        self, empty_chain_graph: GraphStore,
    ):
        # Build the entire chain with heuristic seeds only. Every edge
        # should be skipped at _collect_edges time.
        bio_id = "bio"
        _attach_chain(
            empty_chain_graph,
            _heuristic_edge("cmpd", "tgt", EdgeType.AFFECTS),
            _heuristic_edge(
                "tgt", MechanismCategory.KINASE_INHIBITION.value,
                EdgeType.MODULATES_VIA,
            ),
            _heuristic_edge(
                MechanismCategory.KINASE_INHIBITION.value, bio_id,
                EdgeType.MECHANISM_AFFECTS,
            ),
            _heuristic_edge(bio_id, "ind", EdgeType.BIOLOGY_DRIVES),
            _heuristic_edge("end", "ind", EdgeType.ENDPOINT_CAPTURES),
            _heuristic_edge(
                "ind__unselected", "ind", EdgeType.RESPONDS_DIFFERENTLY,
            ),
        )

        # Predict — engine drops all edges, falls back to AMBIGUOUS 0.5.
        engine = PredictionEngine(empty_chain_graph)
        chain = CausalChain(
            arm_id="arm_a",
            compound_id="cmpd", target_id="tgt",
            mechanism_id=MechanismCategory.KINASE_INHIBITION.value,
            biology_id=bio_id, indication_id="ind", endpoint_id="end",
            subgroup_population_id="ind__unselected",
        )
        result = engine.predict(chain)
        # No edges survive → wide CI / midpoint efficacy.
        assert result.efficacy_probability == pytest.approx(0.5)
        # All edge_contributions list should be empty
        assert result.edge_contributions == []


class TestCuratedOnlyEdges:
    """Edges carrying a single DATABASE_* record (no trial evidence) get
    kept and contribute trust proportional to the tier's n_eff."""

    def test_ot_direct_record_yields_non_zero_evidence_strength(self):
        e = _curated_edge(
            "cmpd", "tgt", EdgeType.AFFECTS,
            bucket=SupportBucket.STRONG_SUPPORT,
            source_type=EvidenceType.DATABASE_OT_DIRECT,
        )
        # Round-28: n_eff=12 for OT_DIRECT.
        assert e.belief.evidence_strength == pytest.approx(12.0)
        assert _trust_weight(e.belief) > 0.0

    def test_cross_reference_record_has_minimal_trust(self):
        e = _curated_edge(
            "cmpd", "tgt", EdgeType.AFFECTS,
            bucket=SupportBucket.WEAK_SUPPORT,
            source_type=EvidenceType.DATABASE_CROSS_REFERENCE,
        )
        # n_eff=0.3 for CROSS_REFERENCE — barely above zero.
        assert e.belief.evidence_strength == pytest.approx(0.3)
        # Trust weight is positive but very small.
        tw = _trust_weight(e.belief)
        assert 0.0 < tw < 0.1

    def test_curated_tiers_strictly_ordered_by_trust(self):
        """Higher-tier curated records should always carry more trust
        than lower-tier ones (with the same bucket)."""
        tiers = [
            EvidenceType.DATABASE_OT_DIRECT,        # 12.0  (round-28)
            EvidenceType.DATABASE_OT_ASSOCIATION,   # 2.0
            EvidenceType.DATABASE_REACTOME_GO,      # 1.5
            EvidenceType.DATABASE_LINCS,            # 1.0
            EvidenceType.DATABASE_FALLBACK,         # 0.5
            EvidenceType.DATABASE_CROSS_REFERENCE,  # 0.3
        ]
        prev_trust = float("inf")
        for tier in tiers:
            e = _curated_edge(
                "cmpd", "tgt", EdgeType.AFFECTS,
                bucket=SupportBucket.STRONG_SUPPORT, source_type=tier,
            )
            tw = _trust_weight(e.belief)
            assert tw < prev_trust, (
                f"{tier.value} trust {tw:.3f} not strictly below "
                f"previous tier {prev_trust:.3f}"
            )
            prev_trust = tw


class TestPhase3StillOutweighsCurated:
    """Round-28 raised the binding-tier n_eff (OT-direct = 12,
    ChEMBL / mAb-table = 10) to reflect that molecular binding is a
    cross-checked fact. But a primary clinical RCT should still carry
    MORE evidence strength than a single curated binding record — the
    Phase-3 trial is N=hundreds of patients, randomized, blinded; the
    OT-direct entry aggregates upstream curators."""

    def test_phase3_trial_above_curated_record(self):
        e_curated = _curated_edge(
            "cmpd", "tgt", EdgeType.AFFECTS,
            bucket=SupportBucket.STRONG_SUPPORT,
            source_type=EvidenceType.DATABASE_OT_DIRECT,
        )
        e_trial = _trial_edge(
            "cmpd", "tgt", EdgeType.AFFECTS,
            bucket=SupportBucket.STRONG_SUPPORT,
            source_type=EvidenceType.CLINICAL_PHASE3, n_trials=1,
        )
        # Phase-3 evidence strength > OT-direct, but they're now in the
        # same order of magnitude (15 vs 12). The ordering is the
        # invariant; the ratio is data-defined.
        assert (
            e_trial.belief.evidence_strength
            > e_curated.belief.evidence_strength
        )
        # Trust weight ordering is preserved.
        assert _trust_weight(e_trial.belief) > _trust_weight(e_curated.belief)


class TestRound25ArchInvariant:
    """Catch-all: confirm the round-25 invariant that the trust-weighted
    geomean produces an interpretable answer in all combinations of
    {only curated, only trial, mixed, none}."""

    @pytest.mark.parametrize("scenario", [
        "only_curated", "only_trial", "mixed", "no_edges",
    ])
    def test_predict_always_returns_valid_probability(
        self, empty_chain_graph: GraphStore, scenario: str,
    ):
        bio_id = "bio"
        mech_id = MechanismCategory.KINASE_INHIBITION.value

        if scenario == "only_curated":
            _attach_chain(
                empty_chain_graph,
                _curated_edge(
                    "cmpd", "tgt", EdgeType.AFFECTS,
                    bucket=SupportBucket.STRONG_SUPPORT,
                    source_type=EvidenceType.DATABASE_OT_DIRECT,
                ),
                _curated_edge(
                    "tgt", mech_id, EdgeType.MODULATES_VIA,
                    bucket=SupportBucket.STRONG_SUPPORT,
                    source_type=EvidenceType.DATABASE_CHEMBL,
                ),
                _curated_edge(
                    mech_id, bio_id, EdgeType.MECHANISM_AFFECTS,
                    bucket=SupportBucket.MODERATE_SUPPORT,
                    source_type=EvidenceType.DATABASE_REACTOME_GO,
                ),
                _curated_edge(
                    bio_id, "ind", EdgeType.BIOLOGY_DRIVES,
                    bucket=SupportBucket.MODERATE_SUPPORT,
                    source_type=EvidenceType.DATABASE_OT_ASSOCIATION,
                ),
                _curated_edge(
                    "end", "ind", EdgeType.ENDPOINT_CAPTURES,
                    bucket=SupportBucket.STRONG_SUPPORT,
                    source_type=EvidenceType.DATABASE_ENDPOINT_PRIOR,
                ),
            )
        elif scenario == "only_trial":
            _attach_chain(
                empty_chain_graph,
                _trial_edge("cmpd", "tgt", EdgeType.AFFECTS),
                _trial_edge("tgt", mech_id, EdgeType.MODULATES_VIA),
                _trial_edge(
                    mech_id, bio_id, EdgeType.MECHANISM_AFFECTS,
                ),
                _trial_edge(bio_id, "ind", EdgeType.BIOLOGY_DRIVES),
                _trial_edge("end", "ind", EdgeType.ENDPOINT_CAPTURES),
            )
        elif scenario == "mixed":
            _attach_chain(
                empty_chain_graph,
                _curated_edge(
                    "cmpd", "tgt", EdgeType.AFFECTS,
                    bucket=SupportBucket.STRONG_SUPPORT,
                    source_type=EvidenceType.DATABASE_OT_DIRECT,
                ),
                _trial_edge(
                    "tgt", mech_id, EdgeType.MODULATES_VIA,
                    n_trials=2,
                ),
            )
        # else "no_edges" — leave the graph as-is.

        chain = CausalChain(
            arm_id="arm_a",
            compound_id="cmpd", target_id="tgt", mechanism_id=mech_id,
            biology_id=bio_id, indication_id="ind", endpoint_id="end",
            subgroup_population_id="ind__unselected",
        )
        engine = PredictionEngine(empty_chain_graph)
        result = engine.predict(chain)
        # Always returns a valid probability in [0, 1].
        assert 0.0 <= result.efficacy_probability <= 1.0
        # CIs are sensible.
        assert 0.0 <= result.ci_lower <= result.ci_upper <= 1.0
