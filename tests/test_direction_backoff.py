"""Phase 0 GATE — direction wired as a hierarchical-backoff grouping key.

Proves, with concrete numeric assertions, that the direction backoff FIRES in the
canonical predict path (predict_clinical_hypothesis → PredictionEngine.predict →
_collect_edges → direction_resolved_belief), so a null DRD2 result is
interpretable. See docs/dev/reports/DRD2_DIRECTION_LOIO_RESULTS.md (Phase 0).

Three claims:
  1. rich agonist + zero antagonist  → an ANTAGONIST query shrinks toward the
     pooled parent (graceful fallback), NOT the Beta(1,1) collapse of
     same-direction-only.
  2. a rich-BOTH edge → each direction uses its own child, numerically distinct
     from the parent and from the other direction (opposite predictions off the
     shared node).
  3. predict_clinical_hypothesis actually invokes the direction-conditional
     belief — agonist vs antagonist compounds get different predictions under
     backoff, identical under flat (so the machinery is not dead code).
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.graph.models import (
    BiologyNode,
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.prediction.path_query import (
    DIRECTION_BACKOFF,
    DIRECTION_FLAT,
    DIRECTION_SAME_ONLY,
    direction_resolved_belief,
    predict_clinical_hypothesis,
    seed_prediction_rng,
)

_T = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _rec(direction: str, n_eff: float, p_obs: float) -> EvidenceRecord:
    """One trial evidence record carrying (direction, applied n_eff/p_obs) — the
    exact fields _direction_filtered subtracts on."""
    return EvidenceRecord(
        source_id=f"trial_{direction or 'agnostic'}_{n_eff}_{p_obs}",
        source_type=EvidenceType.CLINICAL_PHASE3,
        support="strong_support" if p_obs > 0.5 else "contradict",
        timestamp=_T,
        context=({"direction": direction} if direction else {}),
        applied_n_eff=n_eff,
        applied_p_obs=p_obs,
    )


def _chain(direction: str) -> CausalChain:
    return CausalChain(
        arm_id="a", compound_id="c1", subgroup_population_id="UNKNOWN",
        target_id="t1", mechanism_id="m1", biology_id="b1",
        indication_id="i1", endpoint_id="e1",
        outcome=TrialOutcome.UNKNOWN, direction=direction,
    )


_ET = EdgeType.MECHANISM_AFFECTS  # a _DIRECTION_PARTITIONED edge type


# ── Claim 1: graceful fallback to the pooled parent ───────────────────────
def test_thin_direction_falls_back_to_pooled_parent_not_collapse():
    # Edge: agnostic mass (n=4 @ 0.5) + rich AGONIST success (n=10 @ 0.9), zero
    # antagonist. alpha = 1 + (10*0.9) + (4*0.5) = 12 ; beta = 1 + 1 + 2 = 4.
    recs = [_rec("agonist", 10, 0.9), _rec("", 4, 0.5)]
    parent = EdgeBeliefState(alpha=12.0, beta=4.0, evidence=recs)
    antag = _chain("antagonist")

    backoff = direction_resolved_belief(parent, _ET, antag, DIRECTION_BACKOFF)
    same_only = direction_resolved_belief(parent, _ET, antag, DIRECTION_SAME_ONLY)

    # same-direction-only strips the (opposite) agonist mass → Beta(3,3), mean 0.5.
    assert abs(same_only.expected_probability - 0.5) < 1e-9
    # backoff shrinks the thin antagonist child TOWARD the pooled parent: mean
    # well above the 0.5 collapse, close to (just under) the parent's 0.75.
    assert backoff.expected_probability > same_only.expected_probability + 0.1
    assert 0.5 < backoff.expected_probability < parent.expected_probability
    assert abs(backoff.expected_probability - parent.expected_probability) < 0.1  # near the parent, not the prior


# ── Claim 2: rich-both edge — each child distinct from parent & each other ──
def test_rich_both_directions_use_own_child_distinct_from_parent():
    # Balanced parent: agonist success (n=10@0.9) + antagonist failure (n=10@0.1)
    # + agnostic (n=4@0.5). alpha = 1+9+1+2 = 13 ; beta = 1+1+9+2 = 13 → mean 0.5.
    recs = [_rec("agonist", 10, 0.9), _rec("antagonist", 10, 0.1), _rec("", 4, 0.5)]
    parent = EdgeBeliefState(alpha=13.0, beta=13.0, evidence=recs)

    ag = direction_resolved_belief(parent, _ET, _chain("agonist"), DIRECTION_BACKOFF)
    an = direction_resolved_belief(parent, _ET, _chain("antagonist"), DIRECTION_BACKOFF)

    # Opposite predictions off the SAME shared belief.
    assert ag.expected_probability > parent.expected_probability > an.expected_probability
    # Each is numerically distinct from the direction-agnostic parent…
    assert ag.expected_probability - parent.expected_probability > 0.05
    assert parent.expected_probability - an.expected_probability > 0.05
    # …and from each other by a wide margin.
    assert ag.expected_probability - an.expected_probability > 0.15
    # flat mode ignores direction → identical to the parent for both.
    ag_flat = direction_resolved_belief(parent, _ET, _chain("agonist"), DIRECTION_FLAT)
    an_flat = direction_resolved_belief(parent, _ET, _chain("antagonist"), DIRECTION_FLAT)
    assert ag_flat.expected_probability == an_flat.expected_probability == parent.expected_probability


# ── Claim 3: the canonical entry invokes it (not dead code) ────────────────
def _shared_edge_graph() -> GraphStore:
    """Two compounds (agonist c_ag, antagonist c_an) with stated chains through the
    SAME indication and the SAME +/- edge (m1→b1). Identical topology/params — the
    ONLY difference is the chain direction, so any prediction gap is the direction
    machinery alone."""
    g = GraphStore()
    g.add_node(CompoundNode(id="c_ag", name="AgoDrug", modality=Modality.SMALL_MOLECULE))
    g.add_node(CompoundNode(id="c_an", name="AntDrug", modality=Modality.SMALL_MOLECULE))
    g.add_node(TargetNode(id="t1", name="DRD2", gene_symbol="DRD2"))
    g.add_node(MechanismNode(id="m1", name="D2 signaling",
                             mechanism_type=MechanismType.MODULATION))
    g.add_node(BiologyNode(id="b1", name="dopaminergic tone"))
    g.add_node(IndicationNode(id="i1", name="MovementOrPsych"))
    g.add_node(EndpointNode(id="e1", name="Ep", endpoint_type=EndpointType.PRIMARY,
                            regulatory_status=RegulatoryStatus.ACCEPTED))
    g.add_node(PopulationNode(id="i1__unselected", name="P"))

    for c in ("c_ag", "c_an"):
        g.add_edge(GraphEdge(source_id=c, target_id="t1", edge_type=EdgeType.AFFECTS,
                             belief=EdgeBeliefState(alpha=5.0, beta=1.0)))
    g.add_edge(GraphEdge(source_id="t1", target_id="m1",
                         edge_type=EdgeType.MODULATES_VIA,
                         belief=EdgeBeliefState(alpha=4.0, beta=1.0)))
    # THE shared +/- edge: mechanism→biology, both directions, opposite signs.
    recs = [_rec("agonist", 12, 0.9), _rec("antagonist", 12, 0.1), _rec("", 3, 0.5)]
    g.add_edge(GraphEdge(source_id="m1", target_id="b1",
                         edge_type=EdgeType.MECHANISM_AFFECTS,
                         belief=EdgeBeliefState(alpha=14.0, beta=14.0, evidence=recs)))
    g.add_edge(GraphEdge(source_id="b1", target_id="i1",
                         edge_type=EdgeType.BIOLOGY_DRIVES,
                         belief=EdgeBeliefState(alpha=4.0, beta=2.0)))

    # stated chains so predict_clinical_hypothesis takes the faithful path
    for nct, cid, d in (("NCT_AG", "c_ag", "agonist"), ("NCT_AN", "c_an", "antagonist")):
        g.trial_subgraphs[nct] = TrialSubgraph(
            trial_id=nct, parent_population_id="i1__unselected",
            chains=[CausalChain(
                arm_id="a", compound_id=cid, subgroup_population_id="i1__unselected",
                target_id="t1", mechanism_id="m1", biology_id="b1",
                indication_id="i1", endpoint_id="e1",
                outcome=TrialOutcome.UNKNOWN, direction=d)])
    return g


def test_predict_clinical_hypothesis_invokes_direction_backoff():
    g = _shared_edge_graph()

    def _p(cid, mode):
        seed_prediction_rng(42)  # identical MC draws → isolate the belief effect
        return predict_clinical_hypothesis(
            g, cid, "i1", direction_mode=mode
        ).overall_probability

    ag = _p("c_ag", DIRECTION_BACKOFF)
    an = _p("c_an", DIRECTION_BACKOFF)
    ag_flat = _p("c_ag", DIRECTION_FLAT)
    an_flat = _p("c_an", DIRECTION_FLAT)

    # Same shared edge, opposite directions → agonist success > antagonist failure.
    assert ag > an + 0.05, (
        f"direction backoff not firing in canonical path: "
        f"agonist={ag:.3f} antagonist={an:.3f}"
    )
    # Flat: identical topology + identical (direction-ignored) beliefs + same seed
    # → bit-identical. Proves the ONLY driver of the backoff gap is direction.
    assert ag_flat == an_flat, (
        f"flat mode should erase the direction difference: {ag_flat} vs {an_flat}"
    )
    # Backoff must actually move off flat — proof the conditional belief is
    # consumed in the canonical entry, not dead code.
    assert abs(ag - ag_flat) > 0.02 and abs(an - an_flat) > 0.02
