"""Tests for the ground-up (chains-first) build + comparison.

A tiny two-trial top-down graph that share a target/mechanism/biology is exploded
into trial-scoped instances and reassembled; Tier-1 merge must reconstruct the
shared concept set faithfully (bottom-up ≡ top-down dedup).
"""

from __future__ import annotations

from src.graph.models import (
    BiologyNode, CausalChain, EndpointNode, EndpointType, IndicationNode,
    InterventionNode, MechanismNode, MechanismType, PopulationNode,
    RegulatoryStatus, TargetNode, TrialSubgraph,
)
from src.graph.populate_groundup import build_groundup, explode_to_chains_first
from src.graph.store import GraphStore


def _shared_topdown() -> GraphStore:
    """Two trials whose chains share EGFR / kinase_inhibition / a biology node —
    the overlap top-down created by exact-id match."""
    g = GraphStore()
    g.add_node(InterventionNode(id="drugA", name="Drug A"))
    g.add_node(InterventionNode(id="drugB", name="Drug B"))
    g.add_node(TargetNode(id="EGFR", gene_symbol="EGFR", name="EGFR"))
    g.add_node(MechanismNode(id="kinase_inhibition", name="kinase inhibition",
                             mechanism_type=MechanismType.INHIBITION))
    g.add_node(BiologyNode(id="R-1", name="EGFR signaling"))
    g.add_node(IndicationNode(id="nsclc", name="NSCLC"))
    g.add_node(EndpointNode(id="os_nsclc", name="overall survival",
                            endpoint_type=EndpointType.PRIMARY,
                            regulatory_status=RegulatoryStatus.ACCEPTED))
    g.add_node(PopulationNode(id="nsclc__unselected", name="all NSCLC"))

    def chain(compound):
        return CausalChain(
            arm_id="a1", compound_id=compound, subgroup_population_id="nsclc__unselected",
            target_id="EGFR", mechanism_id="kinase_inhibition", biology_id="R-1",
            indication_id="nsclc", endpoint_id="os_nsclc",
        )
    for nct, comp in (("NCT1", "drugA"), ("NCT2", "drugB")):
        g.set_trial_subgraph(TrialSubgraph(
            trial_id=nct, parent_population_id="nsclc__unselected", chains=[chain(comp)],
        ))
    return g


def test_explode_creates_trial_scoped_instances():
    g = _shared_topdown()
    gu = explode_to_chains_first(g, gen_name_id=False)
    # EGFR is shared by both trials → two trial-scoped instances pre-merge
    assert "EGFR#NCT1" in gu._graph and "EGFR#NCT2" in gu._graph  # noqa: SLF001
    assert gu._graph.nodes["EGFR#NCT1"]["ontology_id"] == "EGFR"  # noqa: SLF001


def test_groundup_reconstructs_topdown_concepts_tier1():
    g = _shared_topdown()
    gu, cmp = build_groundup(g)  # Tier-1 only
    # 8 distinct concepts top-down; ground-up explodes shared ones then merges back
    assert cmp.topdown_concepts == cmp.groundup_concepts == 8
    assert cmp.groundup_instances > cmp.groundup_concepts   # there was real overlap
    assert cmp.consolidated == 0                            # faithful, no extra collapse
    # the shared EGFR instances collapsed back to one concept
    egfr = [n for n in gu._graph.nodes  # noqa: SLF001
            if (gu._graph.nodes[n].get("ontology_id") == "EGFR")]
    assert len(egfr) == 1


def test_groundup_chain_edges_match_topdown():
    g = _shared_topdown()
    _, cmp = build_groundup(g)
    assert cmp.topdown_backbone_edges == cmp.groundup_backbone_edges
