"""Tests for cross-indication provenance analysis (src/prediction/provenance.py)."""

from __future__ import annotations

from datetime import datetime

import pytest

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
    TargetNode,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket
from src.prediction.provenance import (
    EdgeProvenance,
    HoldoutTrace,
    canonical_indication,
    deciding_cross_indications,
    find_biology_bridges,
    find_mechanism_bridges,
    id_scheme,
    is_non_disease,
    is_oncology,
    therapeutic_area,
    trace_holdout,
)
from src.prediction.path_query import EdgeContribution
from src.prediction.provenance import _edge_provenance, _replay_records

_TS = datetime(2020, 1, 1)


def _rec(nct: str, support: SupportBucket = SupportBucket.STRONG_SUPPORT) -> EvidenceRecord:
    return EvidenceRecord(
        source_id=nct,
        source_type=EvidenceType.CLINICAL_PHASE3,
        support=support.value,
        timestamp=_TS,
    )


def _belief(alpha: float, beta: float, ncts: list[str]) -> EdgeBeliefState:
    return EdgeBeliefState(
        alpha=alpha, beta=beta, evidence=[_rec(n) for n in ncts]
    )


def _biology_drives(bio_id: str, ind_id: str, belief: EdgeBeliefState) -> GraphEdge:
    return GraphEdge(
        source_id=bio_id,
        target_id=ind_id,
        edge_type=EdgeType.BIOLOGY_DRIVES,
        belief=belief,
    )


# ── pure helpers ─────────────────────────────────────────────────────────


def test_is_oncology():
    assert is_oncology("non_small_cell_lung_cancer")
    assert is_oncology("acute_myeloid_leukemia")
    assert is_oncology("uveal_melanoma")
    assert not is_oncology("hypercholesterolemia")
    assert not is_oncology("alzheimers_disease")


def test_therapeutic_area():
    assert therapeutic_area("breast_cancer") == "oncology"
    assert therapeutic_area("hypercholesterolemia") == "cardiometabolic"
    assert therapeutic_area("alzheimers_disease") == "neuro"
    assert therapeutic_area("some_unmapped_disease") == "other"


def test_is_non_disease():
    assert is_non_disease("chemotherapeutic_agent_toxicity")
    assert is_non_disease("i_cannot_determine_a_specific_disease_blah")
    assert not is_non_disease("hypercholesterolemia")


def test_canonical_indication_rolls_up_subtypes():
    assert canonical_indication("uveal_melanoma") == "melanoma"
    assert canonical_indication("hypercholesterolemia") == "hypercholesterolemia"


def test_id_scheme():
    assert id_scheme("R-HSA-109581") == "reactome"
    assert id_scheme("GO:0006915") == "go"
    assert id_scheme("bio:3de92c5f913e") == "embedded"
    assert id_scheme("angiogenesis__melanoma") == "synthetic"
    assert id_scheme("ENSG00000119535") == "other"


# ── biology bridges ──────────────────────────────────────────────────────


def _store_with_cholesterol_bridge() -> GraphStore:
    """A BiologyNode driving 3 indications across 3 areas, strong on the
    lipid side and contradicted on the neuro side."""
    s = GraphStore()
    s.add_node(BiologyNode(id="GO:0006695", name="cholesterol synthesis inhibition"))
    for ind in ("hypercholesterolemia", "alzheimers_disease", "melanoma"):
        s.add_node(IndicationNode(id=ind, name=ind.replace("_", " ")))
    s.add_edge(_biology_drives("GO:0006695", "hypercholesterolemia",
                               _belief(8, 2, ["NCT00000001"])))  # E[p]=0.80
    s.add_edge(_biology_drives("GO:0006695", "alzheimers_disease",
                               _belief(2, 8, ["NCT00000002"])))  # E[p]=0.20
    s.add_edge(_biology_drives("GO:0006695", "melanoma",
                               _belief(3, 3, ["NCT00000003"])))  # E[p]=0.50
    return s


def test_biology_bridge_detected_with_spread_and_area():
    s = _store_with_cholesterol_bridge()
    bridges = find_biology_bridges(s)
    assert len(bridges) == 1
    b = bridges[0]
    assert b.biology_id == "GO:0006695"
    assert b.id_scheme == "go"
    assert b.n_indications == 3
    assert b.n_areas == 3
    assert b.spans_oncology_and_other is True
    assert b.belief_spread == pytest.approx(0.6, abs=1e-6)
    assert b.strongest.indication_id == "hypercholesterolemia"
    assert b.weakest.indication_id == "alzheimers_disease"
    areas = {side.indication_id: side.area for side in b.sides}
    assert areas["hypercholesterolemia"] == "cardiometabolic"
    assert areas["alzheimers_disease"] == "neuro"
    assert areas["melanoma"] == "oncology"
    # provenance: each side carries its source NCT
    by_ind = {side.indication_id: side for side in b.sides}
    assert by_ind["hypercholesterolemia"].source_ncts == ["NCT00000001"]


def test_single_indication_is_not_a_bridge():
    s = GraphStore()
    s.add_node(BiologyNode(id="GO:0006695", name="cholesterol synthesis inhibition"))
    s.add_node(IndicationNode(id="melanoma", name="melanoma"))
    s.add_edge(_biology_drives("GO:0006695", "melanoma", _belief(5, 2, ["NCT00000001"])))
    assert find_biology_bridges(s) == []


def test_target_sourced_biology_drives_excluded():
    """biology_drives also fires from TargetNode when biology collapses to
    its target — those must NOT count as biology bridges."""
    s = _store_with_cholesterol_bridge()
    # a target driving two indications: a target bridge, not a biology one
    s.add_node(TargetNode(id="ENSG00000113161", name="HMGCR", gene_symbol="HMGCR"))
    s.add_edge(_biology_drives("ENSG00000113161", "hypercholesterolemia",
                               _belief(6, 2, ["NCT00000004"])))
    s.add_edge(_biology_drives("ENSG00000113161", "melanoma",
                               _belief(4, 4, ["NCT00000005"])))
    bridges = find_biology_bridges(s)
    assert {b.biology_id for b in bridges} == {"GO:0006695"}


def test_subtype_collapse_is_not_a_spurious_bridge():
    """A biology driving melanoma + uveal_melanoma is ONE canonical
    indication (uveal rolls up to melanoma), not a bridge."""
    s = GraphStore()
    s.add_node(BiologyNode(id="GO:0006915", name="apoptosis"))
    s.add_node(IndicationNode(id="melanoma", name="melanoma"))
    s.add_node(IndicationNode(id="uveal_melanoma", name="uveal melanoma"))
    s.add_edge(_biology_drives("GO:0006915", "melanoma", _belief(5, 2, ["NCT00000001"])))
    s.add_edge(_biology_drives("GO:0006915", "uveal_melanoma", _belief(3, 2, ["NCT00000002"])))
    assert find_biology_bridges(s) == []


def test_spread_floor_ignores_unobserved_side():
    """A near-prior (Beta(1,1)-ish) side shouldn't be read as a
    contradiction in the spread; it's unobserved, not weak."""
    s = GraphStore()
    s.add_node(BiologyNode(id="GO:0006695", name="cholesterol synthesis inhibition"))
    s.add_node(IndicationNode(id="hypercholesterolemia", name="hc"))
    s.add_node(IndicationNode(id="melanoma", name="melanoma"))
    # melanoma side is essentially unobserved: evidence_strength ~ 0.1 < floor
    s.add_edge(_biology_drives("GO:0006695", "hypercholesterolemia",
                               _belief(8, 2, ["NCT00000001"])))
    s.add_edge(_biology_drives("GO:0006695", "melanoma",
                               _belief(1.05, 1.05, ["NCT00000002"])))
    b = find_biology_bridges(s, spread_evidence_floor=1.0)[0]
    # only the lipid side clears the floor → no two-sided contrast
    assert b.belief_spread == 0.0


# ── mechanism bridges ────────────────────────────────────────────────────


def _chain(indication_id: str) -> CausalChain:
    return CausalChain(
        arm_id="arm1", compound_id="c1", subgroup_population_id="p1",
        target_id="t1", mechanism_id="m1", biology_id="b1",
        indication_id=indication_id, endpoint_id="e1",
    )


def test_mechanism_bridge_via_evidence_provenance():
    s = GraphStore()
    s.add_node(MechanismNode(id="mech:apopto", name="apoptosis induction"))
    s.add_node(BiologyNode(id="GO:0006915", name="apoptosis"))
    # one mechanism_affects belief, co-updated by trials from two diseases
    s.add_edge(GraphEdge(
        source_id="mech:apopto", target_id="GO:0006915",
        edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=_belief(6, 3, ["NCT00000010", "NCT00000011"]),
    ))
    s.trial_subgraphs["NCT00000010"] = TrialSubgraph(
        trial_id="NCT00000010", parent_population_id="melanoma__unselected",
        chains=[_chain("melanoma")],
    )
    s.trial_subgraphs["NCT00000011"] = TrialSubgraph(
        trial_id="NCT00000011", parent_population_id="hc__unselected",
        chains=[_chain("hypercholesterolemia")],
    )
    bridges = find_mechanism_bridges(s)
    assert len(bridges) == 1
    m = bridges[0]
    assert m.mechanism_name == "apoptosis induction"
    assert m.indications == ["hypercholesterolemia", "melanoma"]
    assert m.spans_oncology_and_other is True
    assert m.ncts_by_indication["melanoma"] == ["NCT00000010"]
    assert m.ncts_by_indication["hypercholesterolemia"] == ["NCT00000011"]


def test_mechanism_single_indication_not_a_bridge():
    s = GraphStore()
    s.add_node(MechanismNode(id="mech:apopto", name="apoptosis induction"))
    s.add_node(BiologyNode(id="GO:0006915", name="apoptosis"))
    s.add_edge(GraphEdge(
        source_id="mech:apopto", target_id="GO:0006915",
        edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=_belief(6, 3, ["NCT00000010", "NCT00000012"]),
    ))
    # both source trials are the same indication → not a bridge
    for nct in ("NCT00000010", "NCT00000012"):
        s.trial_subgraphs[nct] = TrialSubgraph(
            trial_id=nct, parent_population_id="melanoma__unselected",
            chains=[_chain("melanoma")],
        )
    assert find_mechanism_bridges(s) == []


# ── per-holdout trace ────────────────────────────────────────────────────


def test_edge_provenance_splits_evidence_by_source_indication():
    """The core trace primitive: an edge's evidence split into self /
    same-indication / cross-indication trial sources + a DB count."""
    s = GraphStore()
    s.add_node(BiologyNode(id="GO:0006915", name="apoptosis"))
    s.add_node(IndicationNode(id="melanoma", name="melanoma"))
    belief = EdgeBeliefState(
        alpha=4, beta=3,
        evidence=[
            _rec("NCT00000001"),  # self
            _rec("NCT00000002"),  # other trial, same indication
            _rec("NCT00000003"),  # other trial, DIFFERENT indication
            EvidenceRecord(source_id="DATABASE_OT", source_type=EvidenceType.DATABASE_OT_DIRECT,
                           support="strong_support", timestamp=_TS),  # non-trial
        ],
    )
    contribution = EdgeContribution(
        source_id="GO:0006915", target_id="melanoma",
        edge_type=EdgeType.BIOLOGY_DRIVES, belief=belief,
        sampled_mean=0.57, bottleneck_score=0.4,
    )
    nct_index = {
        "NCT00000001": {"melanoma"},
        "NCT00000002": {"melanoma"},
        "NCT00000003": {"hypercholesterolemia"},
    }
    ep = _edge_provenance(
        contribution, store=s, nct_index=nct_index,
        self_nct="NCT00000001", self_indications={"melanoma"}, is_deciding=True,
    )
    assert ep.self_ncts == ["NCT00000001"]
    assert ep.same_indication_ncts == ["NCT00000002"]
    assert ep.cross_indication_ncts == {"hypercholesterolemia": ["NCT00000003"]}
    assert ep.n_database_records == 1
    assert ep.source_name == "apoptosis" and ep.target_name == "melanoma"


def _cb(specs: list[tuple[str, SupportBucket]], edge_type: EdgeType) -> EdgeBeliefState:
    """A belief whose stored alpha/beta are CONSISTENT with replaying its
    records (mirrors a real graph, where the delta-adjust self-exclusion is
    exact). Each spec is (source_nct, support bucket)."""
    recs = [_rec(n, sup) for n, sup in specs]
    b = _replay_records(recs, edge_type.value)
    return EdgeBeliefState(alpha=b.alpha, beta=b.beta, evidence=recs)


def _predictable_chain_store() -> GraphStore:
    """Minimal graph with a full compound→indication chain so
    predict_clinical_hypothesis resolves and predicts it. The modulates_via
    edge is the weakest link and carries a cross-indication record."""
    s = GraphStore()
    s.add_node(CompoundNode(id="drugx", name="drugx"))
    s.add_node(TargetNode(id="ENSG00000999999", name="GENEX", gene_symbol="GENEX"))
    s.add_node(MechanismNode(id="bio:mechx", name="mechanism x"))
    s.add_node(BiologyNode(id="GO:0006915", name="apoptosis"))
    s.add_node(IndicationNode(id="melanoma", name="melanoma"))
    s.add_edge(GraphEdge(source_id="drugx", target_id="ENSG00000999999",
                         edge_type=EdgeType.AFFECTS,
                         belief=_cb([("NCT00000001", SupportBucket.STRONG_SUPPORT)], EdgeType.AFFECTS)))
    # weakest link (mixed self-contradict + cross-support) + cross-indication trial:
    s.add_edge(GraphEdge(source_id="ENSG00000999999", target_id="bio:mechx",
                         edge_type=EdgeType.MODULATES_VIA,
                         belief=_cb([("NCT00000001", SupportBucket.MODERATE_CONTRADICT),
                                     ("NCT00000003", SupportBucket.MODERATE_SUPPORT)],
                                    EdgeType.MODULATES_VIA)))
    s.add_edge(GraphEdge(source_id="bio:mechx", target_id="GO:0006915",
                         edge_type=EdgeType.MECHANISM_AFFECTS,
                         belief=_cb([("NCT00000001", SupportBucket.STRONG_SUPPORT)], EdgeType.MECHANISM_AFFECTS)))
    s.add_edge(GraphEdge(source_id="GO:0006915", target_id="melanoma",
                         edge_type=EdgeType.BIOLOGY_DRIVES,
                         belief=_cb([("NCT00000001", SupportBucket.STRONG_SUPPORT)], EdgeType.BIOLOGY_DRIVES)))
    # the predicted trial (melanoma) + a cross-indication trial (hypercholesterolemia)
    s.trial_subgraphs["NCT00000001"] = TrialSubgraph(
        trial_id="NCT00000001", parent_population_id="melanoma__unselected",
        chains=[CausalChain(
            arm_id="a", compound_id="drugx", subgroup_population_id="p",
            target_id="ENSG00000999999", mechanism_id="bio:mechx",
            biology_id="GO:0006915", indication_id="melanoma", endpoint_id="e",
        )],
    )
    s.trial_subgraphs["NCT00000003"] = TrialSubgraph(
        trial_id="NCT00000003", parent_population_id="hc__unselected",
        chains=[_chain("hypercholesterolemia")],
    )
    return s


def test_trace_holdout_end_to_end():
    s = _predictable_chain_store()
    t = trace_holdout(s, "NCT00000001", n_samples=500)
    assert t.nct == "NCT00000001"
    assert t.indications == ["melanoma"]
    assert t.compound_id == "drugx"
    # the modulates_via edge carries the cross-indication record
    mv = next(e for e in t.edges if e.edge_type == "modulates_via")
    assert mv.self_ncts == ["NCT00000001"]
    assert mv.cross_indication_ncts == {"hypercholesterolemia": ["NCT00000003"]}
    # it's the weakest link → the deciding edge, and it spans areas
    assert t.deciding_edge is not None and t.deciding_edge.edge_type == "modulates_via"
    assert t.deciding_edge_has_cross_indication is True
    real, areas, spans = deciding_cross_indications(t)
    assert real == ["hypercholesterolemia"]
    assert spans is True  # melanoma (oncology) + hypercholesterolemia (cardiometabolic)
    # self-exclusion: the deciding edge keeps the cross-indication record, so it
    # is still supported without this trial; edges with only self-evidence drop.
    assert mv.self_excluded_evidence_strength > 0.0
    affects = next(e for e in t.edges if e.edge_type == "affects")  # only self evidence
    assert affects.self_excluded_evidence_strength == 0.0
    assert t.self_excluded_efficacy is not None  # the modulates_via edge survives
    assert t.self_excluded_n_edges == 1


def test_self_exclusion_drops_everything_when_only_self_evidence():
    """A trial whose every edge is supported ONLY by itself has no
    cross-trial transfer → self-excluded efficacy is None."""
    s = _predictable_chain_store()
    # rewrite the modulates_via edge to drop the cross-indication record so
    # EVERY edge is supported only by the trial itself
    s._graph.remove_edge("ENSG00000999999", "bio:mechx", key=EdgeType.MODULATES_VIA.value)
    s.add_edge(GraphEdge(
        source_id="ENSG00000999999", target_id="bio:mechx",
        edge_type=EdgeType.MODULATES_VIA,
        belief=_cb([("NCT00000001", SupportBucket.MODERATE_SUPPORT)], EdgeType.MODULATES_VIA),
    ))
    t = trace_holdout(s, "NCT00000001", n_samples=500)
    assert t.self_excluded_efficacy is None
    assert t.self_excluded_n_edges == 0


def test_trace_holdout_missing_trial_raises():
    s = _predictable_chain_store()
    with pytest.raises(KeyError):
        trace_holdout(s, "NCT99999999")
