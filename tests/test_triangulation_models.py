"""Unit tests for the triangulation / conditional models added to eval_baselines.

Covers (offline, no API): the maximal-composition edge extraction from a trial
subgraph, and that the edge-incidence + conditional model closures return
well-formed probabilities.
"""
from __future__ import annotations

from scripts.eval_baselines import (
    Trial,
    _comp_feats,
    _composition,
    _cond_model,
    _edge_model,
)
from src.graph.models import (
    BiologyNode,
    CausalChain,
    CompoundNode,
    EndpointNode,
    EndpointType,
    IndicationNode,
    MechanismNode,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialArm,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore


def _seed_graph_with_trial(nct: str = "NCT_T") -> GraphStore:
    g = GraphStore()
    g.add_node(CompoundNode(id="drugx", name="DrugX", modality=Modality.ANTIBODY))
    g.add_node(TargetNode(id="ENSG1", name="PD-1", gene_symbol="PDCD1"))
    g.add_node(MechanismNode(id="R-HSA-1", name="checkpoint pathway"))
    g.add_node(BiologyNode(id="bio:abc", name="immune activation"))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(id="PFS", name="PFS", endpoint_type=EndpointType.PRIMARY,
                            regulatory_status=RegulatoryStatus.ACCEPTED))
    g.add_node(PopulationNode(id="pop1", name="line: second · stage: iii"))
    chain = CausalChain(
        arm_id="solo", compound_id="drugx", subgroup_population_id="pop1",
        target_id="ENSG1", mechanism_id="R-HSA-1", biology_id="bio:abc",
        indication_id="melanoma", endpoint_id="PFS", outcome=TrialOutcome.UNKNOWN,
    )
    ts = TrialSubgraph(trial_id=nct, phase="3",
                       arms=[TrialArm(arm_id="solo", compound_ids=["drugx"],
                                      regimen_compound_id="drugx")],
                       chains=[chain], parent_population_id="pop1")
    g.set_trial_subgraph(ts)
    return g


def test_composition_extracts_edges_and_coarse():
    g = _seed_graph_with_trial()
    comp = _composition(g, "NCT_T")
    # granular backbone edges present
    assert "affects|drugx->ENSG1" in comp["edges"]
    assert "mechanism_affects|R-HSA-1->bio:abc" in comp["edges"]
    assert "biology_drives|bio:abc->melanoma" in comp["edges"]
    # coarse collapses the mechanism rung → a synthesized target→biology edge
    assert "t2b|ENSG1->bio:abc" in comp["coarse"]
    assert not any("R-HSA-1" in e for e in comp["coarse"])  # mechanism gone in coarse
    # fan-in conditioners
    assert comp["bios"] == {"bio:abc"}
    assert comp["lines"] == {"late"}              # "second" → late
    assert ("bio:abc", "late") in comp["bio_lines"]


def test_composition_missing_trial_is_empty():
    g = _seed_graph_with_trial()
    comp = _composition(g, "NCT_ABSENT")
    assert comp["edges"] == set() and comp["bios"] == set()


def _mk(nct, y, edges=(), bios=(), bio_lines=()):
    return Trial(nct=nct, y=y, label="success" if y else "failure", is_eval=True,
                 target="t", indication="i", feats={}, chain={},
                 comp_edges=set(edges), bios=set(bios), bio_lines=set(bio_lines))


def test_edge_model_returns_valid_probabilities():
    train = [_mk("a", 1, edges=["E1", "E2"]), _mk("b", 0, edges=["E2", "E3"]),
             _mk("c", 1, edges=["E1"]), _mk("d", 0, edges=["E3"])]
    test = [_mk("e", 1, edges=["E1"]), _mk("f", 0, edges=["E3"])]
    probs = _edge_model(_comp_feats)(train, test, seed=0)
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_edge_model_single_class_train_returns_rate():
    train = [_mk("a", 1, edges=["E1"]), _mk("b", 1, edges=["E2"])]
    test = [_mk("c", 1, edges=["E1"])]
    assert _edge_model(_comp_feats)(train, test, seed=0) == [1.0]


def test_cond_model_pools_biology_line_cells():
    # biology b1 at late line fails (3 trials), b2 at early line succeeds (3 trials)
    train = [_mk(f"l{i}", 0, bios=["b1"], bio_lines=[("b1", "late")]) for i in range(3)]
    train += [_mk(f"e{i}", 1, bios=["b2"], bio_lines=[("b2", "early")]) for i in range(3)]
    test = [_mk("tl", 0, bios=["b1"], bio_lines=[("b1", "late")]),
            _mk("te", 1, bios=["b2"], bio_lines=[("b2", "early")])]
    probs = _cond_model(k_min=2)(train, test, seed=0)
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    # the late-line/b1 test trial should score lower than the early-line/b2 one
    assert probs[0] < probs[1]
