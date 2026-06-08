"""Unit tests for the four edge-completeness passes.

`scripts/edge_completeness_audit.py` checks the expected-edge contract; these
verify the populate-side passes that satisfy it. The two de-orphan passes run
PRE-merge on a per-trial graph (g_t); the two prune passes run POST-merge.
"""
from src.graph.models import (
    AdverseEventNode, BiologyNode, CausalChain, CompoundNode, EdgeBeliefState,
    EdgeType, EndpointNode, EndpointType, GraphEdge, IndicationNode,
    InterventionNode, MechanismNode, PopulationNode, RegulatoryStatus,
    TargetNode, TrialNode, TrialSubgraph,
)
from src.graph.populate import (
    _UNKNOWN, deorphan_nonchain_endpoints, fan_biology_drives_to_coconditions,
    prune_disconnected_noise, prune_graph_topology, prune_orphan_targets,
)
from src.graph.store import GraphStore


def _ep(eid: str) -> EndpointNode:
    return EndpointNode(id=eid, name=eid, endpoint_type=EndpointType.PRIMARY,
                        regulatory_status=RegulatoryStatus.ACCEPTED)


def _chain(**kw):
    base = dict(arm_id="a", compound_id="c", subgroup_population_id=_UNKNOWN,
                target_id="t", mechanism_id="m", biology_id="bio:1",
                indication_id="ind", endpoint_id="PFS")
    base.update(kw)
    return CausalChain(**base)


class TestDeorphanNonchainEndpoints:
    def _g(self) -> GraphStore:
        g = GraphStore()
        g.add_node(BiologyNode(id="bio:1", name="angiogenesis"))
        g.add_node(_ep("PFS"))          # chain endpoint
        g.add_node(_ep("safety_tox"))   # non-chain (CT.gov primary safety outcome)
        g.add_node(_ep("biomarker_x"))  # non-chain
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN, chains=[_chain()]))
        return g

    def test_connects_only_nonchain_endpoints(self):
        g = self._g()
        assert deorphan_nonchain_endpoints(g) == 2
        rb = EdgeType.REFLECTS_BIOLOGY.value
        assert g._graph.has_edge("bio:1", "safety_tox", key=rb)
        assert g._graph.has_edge("bio:1", "biomarker_x", key=rb)
        # the chain endpoint is ensure_reflects_biology_edges' job, not this pass
        assert not g._graph.has_edge("bio:1", "PFS", key=rb)

    def test_idempotent(self):
        g = self._g()
        deorphan_nonchain_endpoints(g)
        assert deorphan_nonchain_endpoints(g) == 0

    def test_noop_without_resolved_biology(self):
        g = GraphStore()
        g.add_node(_ep("PFS"))
        g.add_node(_ep("safety_tox"))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(biology_id=_UNKNOWN)]))
        assert deorphan_nonchain_endpoints(g) == 0


class TestFanBiologyDrivesToCoconditions:
    def _g(self) -> GraphStore:
        g = GraphStore()
        g.add_node(BiologyNode(id="bio:1", name="dna damage response"))
        for i in ("ind_primary", "ind_secondary", "ind_tertiary"):
            g.add_node(IndicationNode(id=i, name=i))
        g.add_edge(GraphEdge(source_id="bio:1", target_id="ind_primary",
                             edge_type=EdgeType.BIOLOGY_DRIVES,
                             belief=EdgeBeliefState(alpha=8.0, beta=2.0)))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(indication_id="ind_primary")]))
        return g

    def test_fans_to_other_conditions_with_borrowed_strength(self):
        g = self._g()
        assert fan_biology_drives_to_coconditions(g) == 2
        bd = EdgeType.BIOLOGY_DRIVES.value
        for ind in ("ind_secondary", "ind_tertiary"):
            assert g._graph.has_edge("bio:1", ind, key=bd)
            data = g._graph.get_edge_data("bio:1", ind, key=bd)
            b = EdgeBeliefState.model_validate(data["belief"])
            assert (b.alpha, b.beta) == (8.0, 2.0)  # borrowed strength
            assert data["metadata"]["co_condition"] is True
            assert data["metadata"]["trial_id"] == "NCT1"
        # primary edge untouched
        prim = g._graph.get_edge_data("bio:1", "ind_primary", key=bd)
        assert EdgeBeliefState.model_validate(prim["belief"]).alpha == 8.0

    def test_idempotent(self):
        g = self._g()
        fan_biology_drives_to_coconditions(g)
        assert fan_biology_drives_to_coconditions(g) == 0


class TestPruneOrphanTargets:
    def test_prunes_family_isoform_not_in_chain(self):
        g = GraphStore()
        g.add_node(CompoundNode(id="paclitaxel", name="Paclitaxel"))
        g.add_node(TargetNode(id="ENSG_chain", name="t", gene_symbol="TUBB"))
        g.add_node(TargetNode(id="ENSG_orphan", name="iso", gene_symbol="TUBA3C"))
        for t in ("ENSG_chain", "ENSG_orphan"):
            g.add_edge(GraphEdge(source_id="paclitaxel", target_id=t,
                                 edge_type=EdgeType.AFFECTS))
        g.add_edge(GraphEdge(source_id="ENSG_chain", target_id="m",
                             edge_type=EdgeType.MODULATES_VIA))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(compound_id="paclitaxel", target_id="ENSG_chain")]))
        assert prune_orphan_targets(g) == 1
        assert "ENSG_orphan" not in g._graph
        assert "ENSG_chain" in g._graph

    def test_keeps_target_with_mechanism_even_if_not_in_chain(self):
        g = GraphStore()
        g.add_node(CompoundNode(id="d", name="D"))
        g.add_node(TargetNode(id="ENSG_x", name="x", gene_symbol="X"))
        g.add_edge(GraphEdge(source_id="d", target_id="ENSG_x",
                             edge_type=EdgeType.AFFECTS))
        g.add_edge(GraphEdge(source_id="ENSG_x", target_id="m",
                             edge_type=EdgeType.MODULATES_VIA))
        assert prune_orphan_targets(g) == 0


class TestPruneDisconnectedNoise:
    def test_prunes_degree0_nontrial_keeps_trial(self):
        g = GraphStore()
        g.add_node(InterventionNode(id="placebo_tablet", name="Placebo tablet"))
        g.add_node(AdverseEventNode(id="AE:soc:empty", name="Investigations"))
        g.add_node(TrialNode(id="NCT1", name="t"))  # degree-0 BY DESIGN
        g.add_node(BiologyNode(id="bio:1", name="b"))
        g.add_node(_ep("PFS"))
        g.add_edge(GraphEdge(source_id="bio:1", target_id="PFS",
                             edge_type=EdgeType.REFLECTS_BIOLOGY))
        assert prune_disconnected_noise(g) == 2
        assert "NCT1" in g._graph          # container kept
        assert "bio:1" in g._graph
        assert "placebo_tablet" not in g._graph
        assert "AE:soc:empty" not in g._graph


class TestPruneGraphTopology:
    """Final fixpoint sweep: a vestigial zero-chain-trial subgraph (ungrounded
    biology + its cardio indications/endpoints) cascades away; the grounded chain
    and the degree-0-by-design TrialNode survive."""

    def test_cascade_removes_degree1_island_keeps_grounded(self):
        g = GraphStore()
        # grounded chain: mech --mechanism_affects--> bio_g, chain references bio_g
        g.add_node(MechanismNode(id="m", name="m"))
        g.add_node(BiologyNode(id="bio_g", name="grounded"))
        g.add_node(IndicationNode(id="ind_g", name="g"))
        g.add_node(_ep("PFS"))
        g.add_edge(GraphEdge(source_id="m", target_id="bio_g",
                             edge_type=EdgeType.MECHANISM_AFFECTS))
        g.add_edge(GraphEdge(source_id="bio_g", target_id="ind_g",
                             edge_type=EdgeType.BIOLOGY_DRIVES))
        g.add_edge(GraphEdge(source_id="bio_g", target_id="PFS",
                             edge_type=EdgeType.REFLECTS_BIOLOGY))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(biology_id="bio_g", indication_id="ind_g")]))
        g.add_node(TrialNode(id="NCT1", name="t"))  # degree-0 BY DESIGN
        # vestige: ungrounded biology + a degree-1 ISLAND that degree-0 pruning
        # can't break (endpoint --endpoint_captures--> indication hold each other
        # up; indication also has responds_differently from a population).
        g.add_node(BiologyNode(id="bio_vestige", name="cardioprotection"))
        g.add_node(IndicationNode(id="cardiomegaly", name="Cardiomegaly"))
        g.add_node(_ep("safety_lvef"))
        g.add_node(PopulationNode(id="lvef_low", name="LVEF low"))
        g.add_edge(GraphEdge(source_id="bio_vestige", target_id="cardiomegaly",
                             edge_type=EdgeType.BIOLOGY_DRIVES))
        g.add_edge(GraphEdge(source_id="bio_vestige", target_id="safety_lvef",
                             edge_type=EdgeType.REFLECTS_BIOLOGY))
        g.add_edge(GraphEdge(source_id="safety_lvef", target_id="cardiomegaly",
                             edge_type=EdgeType.ENDPOINT_CAPTURES))
        g.add_edge(GraphEdge(source_id="lvef_low", target_id="cardiomegaly",
                             edge_type=EdgeType.RESPONDS_DIFFERENTLY))
        g.add_node(AdverseEventNode(id="AE:soc:investigations", name="Inv"))

        totals = prune_graph_topology(g)

        # the whole island cascades away
        for gone in ("bio_vestige", "cardiomegaly", "safety_lvef", "lvef_low",
                     "AE:soc:investigations"):
            assert gone not in g._graph, gone
        # grounded subgraph + container intact
        for keep in ("m", "bio_g", "ind_g", "PFS", "NCT1"):
            assert keep in g._graph
        assert totals["biology"] >= 1
        assert totals["endpoints"] >= 1
        assert totals["indications"] >= 1

    def test_prunes_dead_end_mechanism_keeps_chain_mechanism(self):
        # receptor_agonism: no chain uses it + no mechanism_affects-out → prune.
        # chain_mech: a chain uses it (even missing an upstream target) → keep.
        g = GraphStore()
        g.add_node(TargetNode(id="ENSG_x", name="x", gene_symbol="X"))
        g.add_node(MechanismNode(id="receptor_agonism", name="receptor agonism"))
        g.add_node(MechanismNode(id="chain_mech", name="kept"))
        g.add_node(BiologyNode(id="bio_g", name="grounded"))
        # dead-end: target -> receptor_agonism, but no mechanism_affects out, no chain
        g.add_edge(GraphEdge(source_id="ENSG_x", target_id="receptor_agonism",
                             edge_type=EdgeType.MODULATES_VIA))
        # chain mechanism: referenced by a chain + has biology downstream
        g.add_edge(GraphEdge(source_id="chain_mech", target_id="bio_g",
                             edge_type=EdgeType.MECHANISM_AFFECTS))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(mechanism_id="chain_mech", biology_id="bio_g")]))
        totals = prune_graph_topology(g)
        assert "receptor_agonism" not in g._graph
        assert "chain_mech" in g._graph
        assert "ENSG_x" not in g._graph   # orphaned target cascades after its only mech goes
        assert totals["mechanisms"] >= 1

    def test_keeps_other_class_endpoint_with_reflects_biology(self):
        # an "other"-class endpoint legitimately lacks endpoint_captures but HAS
        # reflects_biology (de-orphan) — it must NOT be pruned.
        g = GraphStore()
        g.add_node(MechanismNode(id="m", name="m"))
        g.add_node(BiologyNode(id="bio_g", name="grounded"))
        g.add_node(_ep("other_treatment_completion_rate"))
        g.add_edge(GraphEdge(source_id="m", target_id="bio_g",
                             edge_type=EdgeType.MECHANISM_AFFECTS))
        g.add_edge(GraphEdge(source_id="bio_g",
                             target_id="other_treatment_completion_rate",
                             edge_type=EdgeType.REFLECTS_BIOLOGY))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(biology_id="bio_g")]))
        totals = prune_graph_topology(g)
        assert "other_treatment_completion_rate" in g._graph
        assert totals["endpoints"] == 0

    def test_idempotent_on_clean_graph(self):
        g = GraphStore()
        g.add_node(MechanismNode(id="m", name="m"))
        g.add_node(BiologyNode(id="bio_g", name="grounded"))
        g.add_edge(GraphEdge(source_id="m", target_id="bio_g",
                             edge_type=EdgeType.MECHANISM_AFFECTS))
        g.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT1", parent_population_id=_UNKNOWN,
            chains=[_chain(biology_id="bio_g")]))
        totals = prune_graph_topology(g)
        assert sum(totals.values()) == 0
