"""Tests for the Phase C compound-consolidation pass.

The critical invariant: when two compound nodes get merged by
stable_id, every unique piece of evidence from both sides ends up on
the canonical node EXACTLY ONCE (no double-counting), and every
TrialSubgraph that referenced the duplicate id now references the
canonical id.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.consolidate_compounds import (
    _pick_canonical,
    _replay_belief_from_evidence,
    consolidate_by_chembl,
)
from src.graph.models import (
    BiologyNode,
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    TargetNode,
    TrialArm,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore


def _make_evidence(
    source_id: str, support: str = "moderate_support",
    source_type: EvidenceType = EvidenceType.CLINICAL_PHASE3,
) -> EvidenceRecord:
    return EvidenceRecord(
        source_id=source_id,
        source_type=source_type,
        support=support,
        quality_score=1.0,
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )


def _seed_duplicate_compound_graph() -> GraphStore:
    """Build a small graph with TWO compound nodes that share a
    stable_id — the round-19 smoke case. Both have edges to the same
    target. The canonical node has more evidence on that edge than
    the duplicate; merging should preserve all unique evidence."""
    g = GraphStore()
    # Canonical: more evidence accumulated.
    g.add_node(CompoundNode(
        id="fluorouracil_canonical",
        name="Fluorouracil",
        modality=Modality.SMALL_MOLECULE,
        stable_id="CHEMBL185",
        embedding=[1.0, 0.0, 0.0],
        aliases=["5-FU"],
    ))
    # Duplicate: different CT.gov spelling, same chembl.
    g.add_node(CompoundNode(
        id="5_fluorouracil",
        name="5-Fluorouracil",
        modality=Modality.SMALL_MOLECULE,
        stable_id="CHEMBL185",
        embedding=[0.999, 0.001, 0.0],
        aliases=["Adrucil"],
    ))
    # Target
    g.add_node(TargetNode(id="t_dna", name="DNA", gene_symbol="DNA"))
    # AE
    from src.graph.models import AdverseEventNode
    g.add_node(AdverseEventNode(id="AE:nausea", name="Nausea"))

    # Canonical's edge to target — 3 evidence records. More accumulated
    # evidence than the duplicate so _pick_canonical picks it.
    canonical_edge = GraphEdge(
        source_id="fluorouracil_canonical",
        target_id="t_dna",
        edge_type=EdgeType.AFFECTS,
        belief=EdgeBeliefState(),
    )
    g.add_edge(canonical_edge)
    g.update_edge_belief(
        "fluorouracil_canonical", "t_dna", EdgeType.AFFECTS,
        _make_evidence("NCT001"),
    )
    g.update_edge_belief(
        "fluorouracil_canonical", "t_dna", EdgeType.AFFECTS,
        _make_evidence("NCT002"),
    )
    g.update_edge_belief(
        "fluorouracil_canonical", "t_dna", EdgeType.AFFECTS,
        _make_evidence("NCT003"),
    )

    # Duplicate's edge to target — 1 evidence record, NEW source not
    # already on canonical.
    duplicate_edge = GraphEdge(
        source_id="5_fluorouracil",
        target_id="t_dna",
        edge_type=EdgeType.AFFECTS,
        belief=EdgeBeliefState(),
    )
    g.add_edge(duplicate_edge)
    g.update_edge_belief(
        "5_fluorouracil", "t_dna", EdgeType.AFFECTS,
        _make_evidence("NCT00068692"),  # new trial, only on duplicate
    )

    # Duplicate also has a causes_ae edge that canonical doesn't —
    # should get copied over as a fresh edge.
    dup_ae_edge = GraphEdge(
        source_id="5_fluorouracil",
        target_id="AE:nausea",
        edge_type=EdgeType.CAUSES_AE,
        belief=EdgeBeliefState(),
    )
    g.add_edge(dup_ae_edge)
    g.update_edge_belief(
        "5_fluorouracil", "AE:nausea", EdgeType.CAUSES_AE,
        _make_evidence("NCT00068692", support="strong_support"),
    )
    return g


# ── _pick_canonical ─────────────────────────────────────────────────────


class TestPickCanonical:
    def test_more_evidence_wins(self):
        g = _seed_duplicate_compound_graph()
        canonical = _pick_canonical(g, ["fluorouracil_canonical", "5_fluorouracil"])
        assert canonical == "fluorouracil_canonical"

    def test_shortest_id_breaks_tie(self):
        # Both have zero evidence — choose by id length.
        g = GraphStore()
        g.add_node(CompoundNode(
            id="abc_long", name="abc_long",
            modality=Modality.SMALL_MOLECULE, stable_id="CHEMBL1",
        ))
        g.add_node(CompoundNode(
            id="abc", name="abc",
            modality=Modality.SMALL_MOLECULE, stable_id="CHEMBL1",
        ))
        canonical = _pick_canonical(g, ["abc_long", "abc"])
        assert canonical == "abc"


# ── Belief replay ───────────────────────────────────────────────────────


class TestReplayBelief:
    def test_replay_matches_sequential_updates(self):
        """Replaying N evidence records through Beta(1,1) must give the
        same alpha+beta as N sequential update_edge_belief calls."""
        from src.graph.models import EdgeBeliefState
        evs = [
            _make_evidence(f"NCT{i}", support="moderate_support")
            for i in range(3)
        ]
        replayed = _replay_belief_from_evidence(evs)

        # Reference: same updates in sequence.
        ref = EdgeBeliefState(alpha=1.0, beta=1.0)
        from src.inference.beliefs import (
            SupportBucket, apply_virtual_evidence,
            effective_n_for_evidence, p_obs_for_bucket,
        )
        for ev in evs:
            n_eff = effective_n_for_evidence(ev.source_type, ev.quality_score)
            p_obs = p_obs_for_bucket(SupportBucket(ev.support))
            ref = apply_virtual_evidence(ref, n_eff=n_eff, p_obs=p_obs)
        assert replayed.alpha == pytest.approx(ref.alpha)
        assert replayed.beta == pytest.approx(ref.beta)
        assert len(replayed.evidence) == 3


# ── Full consolidation ─────────────────────────────────────────────────


class TestConsolidate:
    def test_merge_preserves_all_unique_evidence(self):
        g = _seed_duplicate_compound_graph()
        summary = consolidate_by_chembl(g)

        assert summary["duplicate_groups"] == 1
        assert summary["compounds_consolidated"] == 1

        # Duplicate is gone.
        with pytest.raises(KeyError):
            g.get_node("5_fluorouracil")

        # Canonical edge has BOTH original evidence records (NCT001,
        # NCT002, NCT003) AND the duplicate's (NCT00068692). Four
        # total.
        merged_belief = g.get_edge_belief(
            "fluorouracil_canonical", "t_dna", EdgeType.AFFECTS,
        )
        source_ids = sorted(ev.source_id for ev in merged_belief.evidence)
        # Alphabetical sort puts NCT00068692 first because "0" < "1".
        assert source_ids == ["NCT00068692", "NCT001", "NCT002", "NCT003"]

    def test_no_double_count_when_same_evidence_on_both(self):
        """If a piece of evidence happens to exist on BOTH the canonical
        and the duplicate's edge (e.g. some prior build's bug), the
        merge dedups by (source_id, source_type, support, notes) so it
        appears once in the merged belief, not twice."""
        g = GraphStore()
        # Use ids of equal length so the tiebreak is deterministic
        # (alphabetical → "node_a" picks over "node_b").
        g.add_node(CompoundNode(
            id="node_a", name="node_a", modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL1",
        ))
        g.add_node(CompoundNode(
            id="node_b", name="node_b", modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL1",
        ))
        g.add_node(TargetNode(id="t1", name="t1", gene_symbol="T1"))

        # Same evidence record on both — simulates a buggy prior build.
        for src in ("node_a", "node_b"):
            g.add_edge(GraphEdge(
                source_id=src, target_id="t1",
                edge_type=EdgeType.AFFECTS,
                belief=EdgeBeliefState(),
            ))
            g.update_edge_belief(
                src, "t1", EdgeType.AFFECTS,
                _make_evidence("NCT_SHARED"),
            )

        consolidate_by_chembl(g)
        # node_a is canonical (alphabetical tiebreak on equal evidence).
        merged = g.get_edge_belief("node_a", "t1", EdgeType.AFFECTS)
        # ONLY ONE evidence record — dedup worked.
        ev_sources = [ev.source_id for ev in merged.evidence]
        assert ev_sources == ["NCT_SHARED"]
        # node_b is gone from the graph.
        with pytest.raises(KeyError):
            g.get_node("node_b")

    def test_duplicate_only_edge_gets_copied(self):
        """When the duplicate has an edge the canonical doesn't, the
        edge is copied over to canonical's namespace (no collision to
        merge)."""
        g = _seed_duplicate_compound_graph()
        consolidate_by_chembl(g)
        # canonical now has a causes_ae edge to AE:nausea (originally
        # on the duplicate).
        ae_belief = g.get_edge_belief(
            "fluorouracil_canonical", "AE:nausea", EdgeType.CAUSES_AE,
        )
        assert len(ae_belief.evidence) == 1
        assert ae_belief.evidence[0].source_id == "NCT00068692"

    def test_aliases_unioned_onto_canonical(self):
        g = _seed_duplicate_compound_graph()
        consolidate_by_chembl(g)
        canon = g.get_node("fluorouracil_canonical")
        # Canonical originally had ["5-FU"]; duplicate had ["Adrucil"]
        # plus name="5-Fluorouracil" plus id="5_fluorouracil".
        assert set(canon["aliases"]) >= {
            "5-FU", "Adrucil", "5-Fluorouracil", "5_fluorouracil",
        }

    def test_trial_subgraph_compound_refs_updated(self):
        g = _seed_duplicate_compound_graph()
        # Add a TrialSubgraph that references the DUPLICATE id.
        g.add_node(IndicationNode(id="crc", name="Colorectal cancer"))
        g.add_node(MechanismNode(
            id="dna_damage", name="DNA damage",
            mechanism_type=MechanismType.INHIBITION,
        ))
        g.add_node(BiologyNode(id="b1", name="b1"))
        g.add_node(PopulationNode(id="crc__unselected", name="all crc"))
        from src.graph.models import EndpointNode, EndpointType, RegulatoryStatus
        g.add_node(EndpointNode(
            id="OS_crc", name="OS",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        arm = TrialArm(
            arm_id="arm_a",
            compound_ids=["5_fluorouracil"],  # references duplicate
            regimen_compound_id="5_fluorouracil",
        )
        chain = CausalChain(
            arm_id="arm_a", compound_id="5_fluorouracil",
            subgroup_population_id="crc__unselected",
            target_id="t_dna", mechanism_id="dna_damage",
            biology_id="b1", indication_id="crc",
            endpoint_id="OS_crc", outcome=TrialOutcome.UNKNOWN,
        )
        ts = TrialSubgraph(
            trial_id="NCT_MERGE_TEST", phase="3", arms=[arm],
            chains=[chain], parent_population_id="crc__unselected",
        )
        g.set_trial_subgraph(ts)

        consolidate_by_chembl(g)

        updated = g.get_trial_subgraph_by_id("NCT_MERGE_TEST")
        assert updated.arms[0].compound_ids == ["fluorouracil_canonical"]
        assert updated.arms[0].regimen_compound_id == "fluorouracil_canonical"
        assert updated.chains[0].compound_id == "fluorouracil_canonical"

    def test_no_duplicate_groups_returns_zero(self):
        g = GraphStore()
        g.add_node(CompoundNode(
            id="x", name="x", modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL1",
        ))
        g.add_node(CompoundNode(
            id="y", name="y", modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL2",
        ))
        summary = consolidate_by_chembl(g)
        assert summary["duplicate_groups"] == 0
        assert summary["compounds_consolidated"] == 0

    def test_compounds_without_stable_id_are_ignored(self):
        """Nodes with stable_id=None can't be canonicalized by ChEMBL —
        they get left as-is (a Phase C.5 embedding-based pass could
        merge them later, but this v1 doesn't)."""
        g = GraphStore()
        g.add_node(CompoundNode(
            id="x", name="x", modality=Modality.SMALL_MOLECULE,
            stable_id=None,
        ))
        g.add_node(CompoundNode(
            id="y", name="y", modality=Modality.SMALL_MOLECULE,
            stable_id=None,
        ))
        summary = consolidate_by_chembl(g)
        assert summary["duplicate_groups"] == 0
