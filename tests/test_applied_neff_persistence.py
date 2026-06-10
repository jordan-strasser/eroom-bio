"""Regression: each conjugate update persists the EXACT (n_eff, p_obs) it applied
on the EvidenceRecord, so a replay (the (s,t) field materializer, the LOO
self-exclusion) reconstructs the scalar Beta exactly — including the attributor's
explaining-away override that a nominal ``effective_n_for_evidence`` recompute
ignores. This is the bug that dragged high-belief ``affects`` edges 0.90→0.65 in
the field (data/dev/field_scalar_divergence_findings.md)."""
from datetime import datetime

from src.graph.store import GraphStore
from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    InterventionNode,
    TargetNode,
)
from src.inference.beliefs import effective_n_for_evidence
from src.prediction.provenance import _replay_records


def _binding_edge_store() -> GraphStore:
    g = GraphStore()
    g.add_node(InterventionNode(id="drugx", name="Drug X"))
    g.add_node(TargetNode(id="ENSG1", name="Target", gene_symbol="TGT"))
    g.add_edge(GraphEdge(
        source_id="drugx", target_id="ENSG1",
        edge_type=EdgeType.AFFECTS, belief=EdgeBeliefState(),  # Beta(1,1)
    ))
    return g


def _rec(source_id: str, source_type: EvidenceType, support: str, **kw) -> EvidenceRecord:
    return EvidenceRecord(
        source_id=source_id, source_type=source_type, support=support,
        timestamp=datetime(2026, 1, 1), **kw,
    )


def test_update_persists_applied_override():
    g = _binding_edge_store()
    fail = _rec("NCT_FAIL", EvidenceType.CLINICAL_PHASE3, "moderate_contradict", n_obs=400)
    g.update_edge_belief(
        "drugx", "ENSG1", EdgeType.AFFECTS, fail,
        n_eff_override=0.5, p_obs_override=0.20,
    )
    stored = g.get_edge_belief("drugx", "ENSG1", EdgeType.AFFECTS).evidence[-1]
    assert stored.applied_n_eff == 0.5
    assert stored.applied_p_obs == 0.20
    # the nominal recompute (what the field used to do) is ~10x larger — it would
    # re-apply the full contradiction the explaining-away split away.
    nominal = effective_n_for_evidence(fail.source_type, fail.quality_score, n_obs=fail.n_obs)
    assert nominal > 5 * stored.applied_n_eff


def test_replay_reconstructs_scalar_exactly():
    g = _binding_edge_store()
    # 1) curated binding fact (nominal weighting path stamps applied_n_eff too)
    g.update_edge_belief(
        "drugx", "ENSG1", EdgeType.AFFECTS,
        _rec("opentargets:CHEMBLX", EvidenceType.DATABASE_OT_DIRECT, "strong_support"),
    )
    # 2) failed trial, explaining-away self-protects the high-belief edge
    post = g.update_edge_belief(
        "drugx", "ENSG1", EdgeType.AFFECTS,
        _rec("NCT_FAIL", EvidenceType.CLINICAL_PHASE3, "moderate_contradict", n_obs=400),
        n_eff_override=0.5, p_obs_override=0.20,
    )
    # Replaying the stored evidence reconstructs the scalar EXACTLY (applied weights).
    replayed = _replay_records(post.evidence, EdgeType.AFFECTS.value)
    assert abs(replayed.alpha - post.alpha) < 1e-6
    assert abs(replayed.beta - post.beta) < 1e-6
    # And the binding edge self-protected: one failure barely moved it.
    assert post.alpha / (post.alpha + post.beta) > 0.85
