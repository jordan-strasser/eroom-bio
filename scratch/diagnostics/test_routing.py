"""Unit tests for Pillar A — reason-routed EM with competing-risks censoring
(A3) and the principled normalized responsibility (A4).

All behavior is gated behind ``EROOM_ROUTING`` (default OFF). These tests pin
the routed contract from ``eroom-architecture-v2.md`` (Pillar A table) and
``eroom-em-derivation.md`` (§3.2 censoring table / §4 M-step):

  1. test_safety_death_censors_backbone       — DLT failure leaves every
     efficacy/measurement edge (α,β) untouched; AE occurrence still moves via
     the separate attribute_adverse_events path.
  2. test_operational_censors_backbone         — underpowered / insufficient-
     information failures leave the backbone untouched.
  3. test_responsibility_protects_reliable_edges — efficacy failure: the β-delta
     is monotone in (1 − r_a); a reliable edge gets ≈0 blame, an unreliable one
     absorbs it.
  4. test_efficacy_failure_credits_safety_survival — AE gates get a did-not-fire
     (β += w) count on an efficacy death.
  5. test_flag_off_is_identity                 — with the flag off the legacy
     explaining-away path runs unchanged (no routing markers).

Run:  .venv/bin/python -m pytest scratch/diagnostics/test_routing.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from src.annotation.attributor import Attributor
from src.annotation.taxonomy import (
    ChainResult,
    ExtractedArm,
    FailureClassification,
    FailureMode,
    RoutingBranch,
    routing_branch_for,
    StructuredAE,
    TrialExtraction,
)
from src.graph.models import (
    AdverseEventNode,
    BiologyNode,
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    GraphEdge,
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


# ── Fixtures ────────────────────────────────────────────────────────────────


# Default per-edge backbone beliefs. Distinct E[p] so the responsibility
# ordering (A4) is observable: AFFECTS reliable (0.95), MECHANISM_AFFECTS at
# the prior (0.50), BIOLOGY_DRIVES unreliable (0.20).
def _seed_graph(
    *,
    affects: EdgeBeliefState | None = None,
    modulates_via: EdgeBeliefState | None = None,
    mechanism_affects: EdgeBeliefState | None = None,
    biology_drives: EdgeBeliefState | None = None,
    reflects_biology: EdgeBeliefState | None = None,
    endpoint_captures: EdgeBeliefState | None = None,
    causes_ae: EdgeBeliefState | None = None,
) -> tuple[GraphStore, TrialSubgraph]:
    """One-arm one-chain melanoma graph with the full backbone live, plus an
    optional causes_ae gate on the treatment compound."""
    g = GraphStore()
    g.add_node(CompoundNode(id="drugx", name="DrugX", modality=Modality.ANTIBODY))
    g.add_node(TargetNode(id="ENSG00000188389", name="PD-1", gene_symbol="PDCD1"))
    g.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade"))
    g.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 signaling"))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(
        id="PFS_melanoma", name="PFS (Melanoma)",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="melanoma__unselected", name="All patients"))

    def _b(state, a, bb):
        return state or EdgeBeliefState(alpha=a, beta=bb)

    g.add_edge(GraphEdge(source_id="drugx", target_id="ENSG00000188389",
                         edge_type=EdgeType.AFFECTS,
                         belief=_b(affects, 19.0, 1.0)))          # E[p]=0.95
    g.add_edge(GraphEdge(source_id="ENSG00000188389", target_id="checkpoint_blockade",
                         edge_type=EdgeType.MODULATES_VIA,
                         belief=_b(modulates_via, 2.0, 1.0)))     # E[p]≈0.667
    g.add_edge(GraphEdge(source_id="checkpoint_blockade", target_id="R-HSA-389948",
                         edge_type=EdgeType.MECHANISM_AFFECTS,
                         belief=_b(mechanism_affects, 1.0, 1.0)))  # E[p]=0.50
    g.add_edge(GraphEdge(source_id="R-HSA-389948", target_id="melanoma",
                         edge_type=EdgeType.BIOLOGY_DRIVES,
                         belief=_b(biology_drives, 1.0, 4.0)))     # E[p]=0.20
    g.add_edge(GraphEdge(source_id="R-HSA-389948", target_id="PFS_melanoma",
                         edge_type=EdgeType.REFLECTS_BIOLOGY,
                         belief=_b(reflects_biology, 2.0, 1.0)))
    g.add_edge(GraphEdge(source_id="PFS_melanoma", target_id="melanoma",
                         edge_type=EdgeType.ENDPOINT_CAPTURES,
                         belief=_b(endpoint_captures, 2.0, 1.0)))
    if causes_ae is not None:
        g.add_node(AdverseEventNode(id="AE:nausea", name="Nausea"))
        g.add_edge(GraphEdge(source_id="drugx", target_id="AE:nausea",
                             edge_type=EdgeType.CAUSES_AE, belief=causes_ae))

    arm = TrialArm(arm_id="solo", compound_ids=["drugx"], regimen_compound_id="drugx")
    chain = CausalChain(
        arm_id="solo", compound_id="drugx",
        subgroup_population_id="melanoma__unselected",
        target_id="ENSG00000188389", mechanism_id="checkpoint_blockade",
        biology_id="R-HSA-389948", indication_id="melanoma",
        endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
    )
    ts = TrialSubgraph(
        trial_id="NCT_ROUTE", phase="3", arms=[arm], chains=[chain],
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


def _clf(mode: FailureMode, *, outcome: str = "failure",
         confidence: float = 0.8) -> FailureClassification:
    clf = FailureClassification(
        trial_id="NCT_ROUTE", primary_failure_mode=mode, confidence=confidence,
    )
    clf._raw = {"trial_outcome": outcome, "edges_to_update": []}  # type: ignore[attr-defined]
    return clf


def _ext(outcome: str = "failure", *, sample_size: int | None = 350,
         adverse_events: list[StructuredAE] | None = None) -> TrialExtraction:
    return TrialExtraction(
        trial_id="NCT_ROUTE", sample_size=sample_size,
        arms=[ExtractedArm(arm_id="solo", compounds=["drugx"])],
        results_by_chain=[ChainResult(arm_id="solo", outcome=outcome, endpoint="PFS")],
        adverse_events=adverse_events or [],
    )


_BACKBONE = [
    ("drugx", "ENSG00000188389", EdgeType.AFFECTS),
    ("ENSG00000188389", "checkpoint_blockade", EdgeType.MODULATES_VIA),
    ("checkpoint_blockade", "R-HSA-389948", EdgeType.MECHANISM_AFFECTS),
    ("R-HSA-389948", "melanoma", EdgeType.BIOLOGY_DRIVES),
    ("R-HSA-389948", "PFS_melanoma", EdgeType.REFLECTS_BIOLOGY),
    ("PFS_melanoma", "melanoma", EdgeType.ENDPOINT_CAPTURES),
]


def _backbone_ab(g: GraphStore) -> dict[tuple, tuple[float, float]]:
    out = {}
    for s, t, et in _BACKBONE:
        b = g.get_edge_belief(s, t, et)
        out[(s, t, et)] = (b.alpha, b.beta)
    return out


# ── Step 1: the reason→branch map ────────────────────────────────────────────


class TestBranchMap:
    def test_all_failure_modes_mapped(self):
        # Every enum member resolves to a concrete branch (no silent gaps).
        for mode in FailureMode:
            assert isinstance(routing_branch_for(mode), RoutingBranch)

    def test_intended_semantics(self):
        assert routing_branch_for(FailureMode.NO_TARGET_ENGAGEMENT) == RoutingBranch.EFFICACY
        assert routing_branch_for(FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED) == RoutingBranch.EFFICACY
        assert routing_branch_for(FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT) == RoutingBranch.MEASUREMENT
        assert routing_branch_for(FailureMode.HIGH_PLACEBO_RESPONSE) == RoutingBranch.MEASUREMENT
        assert routing_branch_for(FailureMode.WRONG_POPULATION) == RoutingBranch.MEASUREMENT
        assert routing_branch_for(FailureMode.EFFICACY_IN_SUBGROUP_ONLY) == RoutingBranch.MEASUREMENT
        assert routing_branch_for(FailureMode.WRONG_TIMEFRAME) == RoutingBranch.MEASUREMENT
        assert routing_branch_for(FailureMode.DOSE_LIMITING_TOXICITY) == RoutingBranch.SAFETY
        assert routing_branch_for(FailureMode.UNDERPOWERED) == RoutingBranch.OPERATIONAL
        assert routing_branch_for(FailureMode.INSUFFICIENT_INFORMATION) == RoutingBranch.OPERATIONAL
        assert routing_branch_for(FailureMode.COMMERCIAL_NOT_SCIENTIFIC) == RoutingBranch.OPERATIONAL
        assert routing_branch_for(FailureMode.MANUFACTURING_OR_DELIVERY) == RoutingBranch.OPERATIONAL
        assert routing_branch_for(FailureMode.MULTIPLE_FACTORS) == RoutingBranch.UNKNOWN

    def test_none_defaults_unknown(self):
        assert routing_branch_for(None) == RoutingBranch.UNKNOWN


# ── Test 1: safety death censors the backbone ────────────────────────────────


class TestSafetyDeathCensorsBackbone:
    def test_dlt_failure_leaves_backbone_unchanged(self, monkeypatch):
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph()
        before = _backbone_ab(g)
        updates = Attributor(g).attribute(
            _clf(FailureMode.DOSE_LIMITING_TOXICITY), ts, _ext("failure"),
        )
        after = _backbone_ab(g)
        # Censored: every efficacy/measurement edge is byte-for-byte unchanged.
        assert after == before
        # And no backbone update was emitted (no down/upvote on the spine).
        assert not [u for u in updates
                    if (u.source_id, u.target_id, u.edge_type) in set(_BACKBONE)]

    def test_dlt_does_not_credit_ae_survival(self, monkeypatch):
        # The gate FIRED on a safety death → no did-not-fire survival credit.
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph(causes_ae=EdgeBeliefState(alpha=3.0, beta=2.0))
        pre = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        Attributor(g).attribute(
            _clf(FailureMode.DOSE_LIMITING_TOXICITY), ts, _ext("failure"),
        )
        post = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        assert (post.alpha, post.beta) == (pre.alpha, pre.beta)

    def test_ae_occurrence_still_moves_under_censor(self, monkeypatch, tmp_path):
        """The censor only blocks the BACKBONE; the existing
        attribute_adverse_events occurrence path still moves the AE edge."""
        from src.annotation.meddra import MeddraCache

        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph()
        before = _backbone_ab(g)
        cache = MeddraCache(tmp_path / "meddra.json")
        cache.set("nausea", {"preferred_term": "Nausea",
                             "system_organ_class": "Gastrointestinal disorders"})
        ext = _ext("failure", adverse_events=[
            StructuredAE(term="nausea", incidence_treatment_pct=40.0,
                         incidence_control_pct=5.0, serious=True),
        ])
        attributor = Attributor(g)
        # Backbone censored by attribute(); AE occurrence lands separately.
        attributor.attribute(_clf(FailureMode.DOSE_LIMITING_TOXICITY), ts, ext)
        ae_updates = asyncio.run(attributor.attribute_adverse_events(
            ts, ext, client=None, meddra_cache=cache,
            classification=_clf(FailureMode.DOSE_LIMITING_TOXICITY),
        ))
        # The AE edge moved (occurrence), the backbone did not (censored).
        assert any(u.edge_type == EdgeType.CAUSES_AE for u in ae_updates)
        assert _backbone_ab(g) == before


# ── Test 2: operational failure censors the backbone ─────────────────────────


class TestOperationalCensorsBackbone:
    @pytest.mark.parametrize("mode", [
        FailureMode.UNDERPOWERED,
        FailureMode.INSUFFICIENT_INFORMATION,
        FailureMode.COMMERCIAL_NOT_SCIENTIFIC,
        FailureMode.MANUFACTURING_OR_DELIVERY,
    ])
    def test_operational_failure_leaves_backbone_unchanged(self, monkeypatch, mode):
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph()
        before = _backbone_ab(g)
        Attributor(g).attribute(_clf(mode), ts, _ext("failure"))
        assert _backbone_ab(g) == before

    def test_operational_does_not_credit_ae_survival(self, monkeypatch):
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph(causes_ae=EdgeBeliefState(alpha=3.0, beta=2.0))
        pre = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        Attributor(g).attribute(_clf(FailureMode.UNDERPOWERED), ts, _ext("failure"))
        post = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        assert (post.alpha, post.beta) == (pre.alpha, pre.beta)


# ── Test 3: principled responsibility protects reliable edges ────────────────


class TestResponsibilityProtectsReliableEdges:
    def test_beta_delta_monotone_in_one_minus_r(self, monkeypatch):
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph()  # AFFECTS 0.95, MECHANISM_AFFECTS 0.50, BIOLOGY_DRIVES 0.20
        updates = Attributor(g).attribute(
            _clf(FailureMode.NO_TARGET_ENGAGEMENT), ts, _ext("failure"),
        )
        by_edge = {u.edge_type: u for u in updates}

        def beta_delta(et):
            u = by_edge[et]
            return u.post_update_belief.beta - u.pre_update_belief.beta

        def alpha_delta(et):
            u = by_edge[et]
            return u.post_update_belief.alpha - u.pre_update_belief.alpha

        reliable = beta_delta(EdgeType.AFFECTS)          # r_a = 0.95
        mid = beta_delta(EdgeType.MECHANISM_AFFECTS)     # r_a = 0.50
        unreliable = beta_delta(EdgeType.BIOLOGY_DRIVES)  # r_a = 0.20

        # β-delta = w·(1 − r_a)/(1 − M): strictly monotone in (1 − r_a).
        assert unreliable > mid > reliable > 0
        # The reliable edge collects ≈no blame; it is mostly CREDITED (α-delta)
        # for having survived (P(f_a=1 | fail) ≈ 1).
        assert reliable < 0.1 * unreliable
        assert alpha_delta(EdgeType.AFFECTS) > alpha_delta(EdgeType.BIOLOGY_DRIVES)
        # Each must-hold edge receives exactly its full weight w (α-share +
        # β-share = w), the responsibility partition.
        w = updates[0].evidence.context["n_eff_applied"]
        for et in (EdgeType.AFFECTS, EdgeType.MECHANISM_AFFECTS, EdgeType.BIOLOGY_DRIVES):
            assert alpha_delta(et) + beta_delta(et) == pytest.approx(w, rel=1e-9)

    def test_measurement_failure_also_uses_responsibility(self, monkeypatch):
        # MEASUREMENT shares the responsibility path (blame within M_t).
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph()
        updates = Attributor(g).attribute(
            _clf(FailureMode.HIGH_PLACEBO_RESPONSE), ts, _ext("failure"),
        )
        routed = [u for u in updates if u.evidence.context.get("routed")]
        assert routed
        assert all(u.evidence.context["routing_branch"] == "measurement" for u in routed)

    def test_degenerate_all_reliable_chain_is_skipped(self, monkeypatch):
        # Every must-hold at E[p]=1.0 → M=1, (1−M)=0 → no information → skip.
        # (A merely-high chain like 0.999^6≈0.994 does NOT trip the guard and
        # correctly still updates — the guard is only for the exact 1−M→0 limit.)
        monkeypatch.setenv("EROOM_ROUTING", "1")
        certain = EdgeBeliefState(alpha=50.0, beta=0.0)  # E[p]=1.0 exactly
        g, ts = _seed_graph(
            affects=certain, modulates_via=certain, mechanism_affects=certain,
            biology_drives=certain, reflects_biology=certain,
            endpoint_captures=certain,
        )
        before = _backbone_ab(g)
        Attributor(g).attribute(
            _clf(FailureMode.NO_TARGET_ENGAGEMENT), ts, _ext("failure"),
        )
        assert _backbone_ab(g) == before


# ── Test 4: efficacy failure credits safety survival ─────────────────────────


class TestEfficacyFailureCreditsSafetySurvival:
    def test_ae_gate_gets_did_not_fire_count(self, monkeypatch):
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph(causes_ae=EdgeBeliefState(alpha=3.0, beta=2.0))
        pre = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        updates = Attributor(g).attribute(
            _clf(FailureMode.NO_TARGET_ENGAGEMENT), ts, _ext("failure"),
        )
        post = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        surv = [u for u in updates
                if u.edge_type == EdgeType.CAUSES_AE
                and u.evidence.context.get("safety_survival")]
        assert len(surv) == 1
        w = surv[0].evidence.context["n_eff_applied"]
        # b += w (all mass to β), α unchanged.
        assert post.alpha == pytest.approx(pre.alpha, rel=1e-9)
        assert post.beta - pre.beta == pytest.approx(w, rel=1e-9)
        assert w > 0

    def test_success_also_credits_survival(self, monkeypatch):
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g, ts = _seed_graph(causes_ae=EdgeBeliefState(alpha=3.0, beta=2.0))
        pre = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        Attributor(g).attribute(
            _clf(FailureMode.INSUFFICIENT_INFORMATION, outcome="success"),
            ts, _ext("success"),
        )
        post = g.get_edge_belief("drugx", "AE:nausea", EdgeType.CAUSES_AE)
        # Survived (success) → β credit even though the reason maps to operational.
        assert post.beta > pre.beta
        assert post.alpha == pytest.approx(pre.alpha, rel=1e-9)


# ── Test 5: flag off is identity to the legacy path ──────────────────────────


class TestFlagOffIsIdentity:
    def test_flag_off_runs_legacy_explain_away(self, monkeypatch):
        monkeypatch.delenv("EROOM_ROUTING", raising=False)
        g, ts = _seed_graph()
        updates = Attributor(g).attribute(
            _clf(FailureMode.DOSE_LIMITING_TOXICITY), ts, _ext("failure"),
        )
        # Legacy markers present, routing markers absent.
        assert updates
        for u in updates:
            assert "explain_away_weight" in u.evidence.context
            assert "routed" not in u.evidence.context
        # Legacy invariant: the explaining-away fractions partition the mass.
        fracs = [u.evidence.context["explain_away_weight"] for u in updates]
        assert abs(sum(fracs) - 1.0) < 1e-9

    def test_flag_off_downvotes_backbone_on_dlt_unlike_routed(self, monkeypatch):
        """Same DLT failure: flag OFF downvotes the whole backbone (legacy,
        class-blind), flag ON censors it. The flag is what changes behavior."""
        # Flag OFF — legacy contradict reaches the spine.
        monkeypatch.delenv("EROOM_ROUTING", raising=False)
        g_off, ts_off = _seed_graph()
        before_off = _backbone_ab(g_off)
        Attributor(g_off).attribute(
            _clf(FailureMode.DOSE_LIMITING_TOXICITY), ts_off, _ext("failure"),
        )
        assert _backbone_ab(g_off) != before_off  # legacy moved the backbone

        # Flag ON — censored.
        monkeypatch.setenv("EROOM_ROUTING", "1")
        g_on, ts_on = _seed_graph()
        before_on = _backbone_ab(g_on)
        Attributor(g_on).attribute(
            _clf(FailureMode.DOSE_LIMITING_TOXICITY), ts_on, _ext("failure"),
        )
        assert _backbone_ab(g_on) == before_on  # routed left it untouched

    def test_flag_off_is_deterministic(self, monkeypatch):
        # Two flag-off runs produce identical (α,β) on every backbone edge.
        monkeypatch.delenv("EROOM_ROUTING", raising=False)
        out = []
        for _ in range(2):
            g, ts = _seed_graph()
            Attributor(g).attribute(
                _clf(FailureMode.NO_TARGET_ENGAGEMENT), ts, _ext("failure"),
            )
            out.append(_backbone_ab(g))
        assert out[0] == out[1]
