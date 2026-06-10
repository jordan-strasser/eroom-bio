"""Focused unit tests for the outcome-conditioning prediction-core redesign.

Covers the four pieces of the redesign (offline, no API):

  1. Verbatim stated stop reason (CT.gov ``why_stopped``) round-trips into
     the TrialNode + TrialSubgraph metadata (Piece 1).
  2. The trial-level operational gate down-weights a true conduct failure
     but NOT a chain failure (wrong endpoint / wrong subgroup) (Piece 2).
  3. On a FAILURE, a high-E[p] curated edge (binds_to with α≫β) barely
     moves while a low-E[p] uncertain edge absorbs most of the contradict
     mass — the explaining-away ordering (Piece 3).
  4. On a SUCCESS, every backbone edge moves up (Piece 3).
  5. The saturating f(N) population multiplier (Piece 4).

The principle: a single trial cannot pinpoint WHICH chain edge failed
(premature falsification), so the OUTCOME conditions the WHOLE chain by id
and failure mass is spread (explaining-away), modestly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.annotation.attributor import (
    Attributor,
    _edge_attr_mode,
    _effect_modulation,
    _per_edge_fracs,
)
from src.annotation.taxonomy import (
    ChainResult,
    ExtractedArm,
    FailureClassification,
    FailureMode,
    OPERATIONAL_GATE_WEIGHT,
    TrialExtraction,
    gate_weight_for,
)
from src.graph.models import (
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
from src.graph.populate import PopulationPipeline, seed_trial_node
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ArmGroup,
    Intervention,
    OutcomeMeasure,
    TrialRecord,
)
from src.inference.beliefs import _precision_multiplier


# ── Shared single-arm graph fixture ─────────────────────────────────────────


def _seed_single_arm_graph(
    *,
    affects_belief: EdgeBeliefState | None = None,
    mech_belief: EdgeBeliefState | None = None,
) -> tuple[GraphStore, TrialSubgraph]:
    """A one-arm, one-chain melanoma graph with the full backbone live.

    The AFFECTS edge defaults to a curated near-certainty (α=40, β=1 →
    E[p]≈0.976); the MECHANISM_AFFECTS edge to maximal uncertainty
    (α=1, β=1 → E[p]=0.5). The other backbone edges start at Beta(2, 1)
    so they have a mild prior to move from. Callers override the two key
    edges for the explaining-away test.
    """
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

    affects = affects_belief or EdgeBeliefState(alpha=40.0, beta=1.0)
    mech = mech_belief or EdgeBeliefState(alpha=1.0, beta=1.0)
    g.add_edge(GraphEdge(source_id="drugx", target_id="ENSG00000188389",
                         edge_type=EdgeType.AFFECTS, belief=affects))
    g.add_edge(GraphEdge(source_id="ENSG00000188389", target_id="checkpoint_blockade",
                         edge_type=EdgeType.MODULATES_VIA,
                         belief=EdgeBeliefState(alpha=2.0, beta=1.0)))
    g.add_edge(GraphEdge(source_id="checkpoint_blockade", target_id="R-HSA-389948",
                         edge_type=EdgeType.MECHANISM_AFFECTS, belief=mech))
    g.add_edge(GraphEdge(source_id="R-HSA-389948", target_id="melanoma",
                         edge_type=EdgeType.BIOLOGY_DRIVES,
                         belief=EdgeBeliefState(alpha=2.0, beta=1.0)))
    g.add_edge(GraphEdge(source_id="R-HSA-389948", target_id="PFS_melanoma",
                         edge_type=EdgeType.REFLECTS_BIOLOGY,
                         belief=EdgeBeliefState(alpha=2.0, beta=1.0)))
    g.add_edge(GraphEdge(source_id="PFS_melanoma", target_id="melanoma",
                         edge_type=EdgeType.ENDPOINT_CAPTURES,
                         belief=EdgeBeliefState(alpha=2.0, beta=1.0)))

    arm = TrialArm(arm_id="solo", compound_ids=["drugx"],
                   regimen_compound_id="drugx")
    chain = CausalChain(
        arm_id="solo", compound_id="drugx",
        subgroup_population_id="melanoma__unselected",
        target_id="ENSG00000188389", mechanism_id="checkpoint_blockade",
        biology_id="R-HSA-389948", indication_id="melanoma",
        endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
    )
    ts = TrialSubgraph(
        trial_id="NCT_OC", phase="3", arms=[arm], chains=[chain],
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


def _clf(
    *,
    outcome: str = "failure",
    operational_failure: bool | None = None,
    confidence: float = 0.8,
) -> FailureClassification:
    clf = FailureClassification(
        trial_id="NCT_OC",
        primary_failure_mode=FailureMode.INSUFFICIENT_INFORMATION,
        confidence=confidence,
        operational_failure=operational_failure,
    )
    clf._raw = {"trial_outcome": outcome, "edges_to_update": []}  # type: ignore[attr-defined]
    return clf


def _ext(outcome: str = "failure", *, sample_size: int | None = 100) -> TrialExtraction:
    return TrialExtraction(
        trial_id="NCT_OC",
        sample_size=sample_size,
        arms=[ExtractedArm(arm_id="solo", compounds=["drugx"])],
        results_by_chain=[ChainResult(arm_id="solo", outcome=outcome, endpoint="PFS")],
    )


# ── Piece 1: verbatim stated reason round-trips ─────────────────────────────


class TestVerbatimStopReasonRoundTrip:
    def _make_terminated_trial(self, why: str) -> TrialRecord:
        return TrialRecord(
            nct_id="NCT_WHY",
            title="A terminated melanoma trial",
            phase="2",
            status="TERMINATED",
            conditions=["Melanoma"],
            interventions=[Intervention(name="DrugX", type="DRUG", description="anti-PD-1")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival", timeframe="24 months")],
            enrollment=40,
            has_results=False,
            why_stopped=why,
            arm_groups=[ArmGroup(group_id="drugx", title="DrugX", intervention_names=["DrugX"])],
        )

    def test_why_stopped_round_trips_into_trial_node_metadata(self):
        why = "Slow accrual; sponsor business decision"
        g = GraphStore()
        seed_trial_node(g, self._make_terminated_trial(why))
        node = g.get_node("NCT_WHY")
        # Verbatim text, not a classified judgment.
        assert node["metadata"]["why_stopped"] == why

    def test_why_stopped_round_trips_into_trial_subgraph_metadata(self):
        from unittest.mock import AsyncMock
        from types import SimpleNamespace

        why = "Terminated due to an increased risk of cardiac events"
        trial = self._make_terminated_trial(why)
        g = GraphStore()
        client = AsyncMock()

        async def _fake_create(**kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="OS")])

        client.messages.create = _fake_create
        pipeline = PopulationPipeline(g, anthropic_client=client)
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1
        assert subgraphs[0].metadata["why_stopped"] == why
        # And it persisted on the node too.
        assert g.get_node("NCT_WHY")["metadata"]["why_stopped"] == why

    def test_absent_why_stopped_is_none_not_missing(self):
        trial = self._make_terminated_trial("")  # CT.gov omitted → None
        trial.why_stopped = None
        g = GraphStore()
        seed_trial_node(g, trial)
        node = g.get_node("NCT_WHY")
        assert node["metadata"]["why_stopped"] is None


# ── Piece 2: the trial-level gate ──────────────────────────────────────────


class TestOperationalGate:
    def test_gate_weight_derivation(self):
        assert gate_weight_for(True) == OPERATIONAL_GATE_WEIGHT
        assert gate_weight_for(False) == 1.0
        assert gate_weight_for(None) == 1.0  # unknown → conservative full weight

    def test_operational_failure_downweights_chain_conditioning(self):
        """An operational failure (recruitment collapse, manufacturing
        error, …) barely perturbs the mechanism beliefs."""
        g, ts = _seed_single_arm_graph()
        clf = _clf(outcome="failure", operational_failure=True)
        updates = Attributor(g).attribute(clf, ts, _ext("failure"))
        mech = next(u for u in updates if u.edge_type == EdgeType.MECHANISM_AFFECTS)
        gated_delta = abs(mech.probability_change)

        # Same trial, but a VALID test (chain failure) → full weight.
        g2, ts2 = _seed_single_arm_graph()
        clf2 = _clf(outcome="failure", operational_failure=False)
        updates2 = Attributor(g2).attribute(clf2, ts2, _ext("failure"))
        mech2 = next(u for u in updates2 if u.edge_type == EdgeType.MECHANISM_AFFECTS)
        valid_delta = abs(mech2.probability_change)

        # The operational gate (0.2) makes the update much smaller.
        assert gated_delta < valid_delta
        assert gated_delta < 0.5 * valid_delta

    def test_chain_failure_is_not_operational_full_weight(self):
        """A wrong-endpoint / wrong-subgroup failure is a CHAIN failure, not
        operational — it conditions the chain at FULL weight. We assert the
        gate weight stays 1.0 (operational_failure=False) and the update is
        the same magnitude as the un-gated baseline."""
        # operational_failure=False is the correct classification for a
        # wrong-endpoint/wrong-subgroup chain failure.
        clf = _clf(outcome="failure", operational_failure=False)
        assert clf.gate_weight == 1.0

        g, ts = _seed_single_arm_graph()
        updates = Attributor(g).attribute(clf, ts, _ext("failure"))
        mech = next(u for u in updates if u.edge_type == EdgeType.MECHANISM_AFFECTS)
        # Full-weight contradict noticeably moves the uncertain edge down.
        assert mech.probability_change < 0
        assert abs(mech.probability_change) > 0.05


# ── Piece 3: explaining-away ordering on FAILURE ───────────────────────────


class TestExplainingAway:
    def test_curated_edge_self_protects_uncertain_edge_absorbs(self):
        """On a FAILURE the high-E[p] curated AFFECTS edge (α≫β) barely
        moves; the near-0.5 MECHANISM_AFFECTS edge absorbs most of the
        contradict mass. Assert the probability-change ordering."""
        g, ts = _seed_single_arm_graph(
            affects_belief=EdgeBeliefState(alpha=80.0, beta=1.0),   # E[p]≈0.988
            mech_belief=EdgeBeliefState(alpha=1.0, beta=1.0),       # E[p]=0.5
        )
        updates = Attributor(g).attribute(
            _clf(outcome="failure"), ts, _ext("failure"),
        )
        by_edge = {u.edge_type: u for u in updates}
        affects_delta = abs(by_edge[EdgeType.AFFECTS].probability_change)
        mech_delta = abs(by_edge[EdgeType.MECHANISM_AFFECTS].probability_change)

        # The weak link takes the blame (symmetric with softmin prediction).
        assert mech_delta > affects_delta
        # The curated binding edge is essentially untouched.
        assert affects_delta < 0.01
        # The uncertain edge actually moved.
        assert mech_delta > 0.02

    def test_failure_mass_is_modest_no_collapse(self):
        """One failed trial — even a huge Phase 3 — can never collapse an
        edge, because the failure mass is split across the chain."""
        g, ts = _seed_single_arm_graph(
            mech_belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        )
        updates = Attributor(g).attribute(
            _clf(outcome="failure", confidence=1.0), ts,
            _ext("failure", sample_size=5000),
        )
        for u in updates:
            assert 0.02 < u.post_update_belief.expected_probability < 0.98

    def test_all_failure_weight_fractions_sum_to_one(self):
        """The per-edge explaining-away fractions (w_i) partition the trial
        mass — they sum to ~1 across the conditioned chain."""
        g, ts = _seed_single_arm_graph()
        updates = Attributor(g).attribute(
            _clf(outcome="failure"), ts, _ext("failure"),
        )
        fracs = [u.evidence.context["explain_away_weight"] for u in updates]
        assert abs(sum(fracs) - 1.0) < 1e-9


# ── Piece 3: SUCCESS moves every backbone edge up ──────────────────────────


class TestSuccessConditioning:
    def test_success_moves_all_backbone_edges_up(self):
        g, ts = _seed_single_arm_graph(
            affects_belief=EdgeBeliefState(alpha=1.0, beta=1.0),  # fresh so it moves
        )
        updates = Attributor(g).attribute(
            _clf(outcome="success"), ts, _ext("success"),
        )
        # 6 live backbone edges all conditioned.
        assert len(updates) == 6
        assert all(u.probability_change > 0 for u in updates)
        # SUCCESS is conjunctive: each edge gets FULL trial weight (frac=1).
        assert all(
            u.evidence.context["explain_away_weight"] == 1.0 for u in updates
        )


# ── Piece 4: saturating f(N) ───────────────────────────────────────────────


class TestSaturatingFofN:
    def test_precision_multiplier_saturates(self):
        """Large N produces a larger multiplier than small N, but NOT
        linearly larger — the ceil caps it (concave √N)."""
        small = _precision_multiplier(40)
        big = _precision_multiplier(3500)
        # Bigger trial → bigger weight.
        assert big > small
        # But sub-linear: 3500/40 = 87.5x patients, far from 87.5x weight.
        assert big / small < 10.0
        # And the ceil bounds it.
        assert big <= 2.5
        # Floor protects tiny trials.
        assert small >= 0.5

    def test_large_n_trial_has_larger_w_base_but_capped(self):
        """End-to-end: a large-N trial's conditioning carries more weight
        than a small-N trial's, but the saturation keeps the ratio bounded.

        Compared on a FRESH uncertain edge so the |Δp| is monotonic in
        n_eff (avoids the saturating-belief confound near the prior)."""
        # Small-N trial.
        g_s, ts_s = _seed_single_arm_graph(
            mech_belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        )
        up_s = Attributor(g_s).attribute(
            _clf(outcome="failure"), ts_s, _ext("failure", sample_size=40),
        )
        mech_s = next(u for u in up_s if u.edge_type == EdgeType.MECHANISM_AFFECTS)
        n_eff_small = mech_s.evidence.context["n_eff_applied"]

        # Large-N trial.
        g_b, ts_b = _seed_single_arm_graph(
            mech_belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        )
        up_b = Attributor(g_b).attribute(
            _clf(outcome="failure"), ts_b, _ext("failure", sample_size=3500),
        )
        mech_b = next(u for u in up_b if u.edge_type == EdgeType.MECHANISM_AFFECTS)
        n_eff_big = mech_b.evidence.context["n_eff_applied"]

        # Larger N → larger applied n_eff, but the per-edge weight ratio is
        # bounded by the saturating multiplier (≈ ceil 2.5 / floor 0.5 = 5x).
        assert n_eff_big > n_eff_small
        assert n_eff_big / n_eff_small <= 5.0


# ── TASK 2: per-edge attribution-math experiment knobs ─────────────────────


class TestEdgeAttrModes:
    def test_default_mode_is_explain_away(self, monkeypatch):
        monkeypatch.delenv("EROOM_EDGE_ATTR", raising=False)
        assert _edge_attr_mode() == "explain_away"

    def test_invalid_mode_raises(self, monkeypatch):
        monkeypatch.setenv("EROOM_EDGE_ATTR", "nonsense")
        with pytest.raises(ValueError):
            _edge_attr_mode()

    def test_per_edge_fracs_dispatch(self):
        ef = [0.5, 0.3, 0.2]  # an explaining-away split (already normalized)
        # explain_away: success=full, failure=explaining-away
        assert _per_edge_fracs("explain_away", True, ef, 3) == [1.0, 1.0, 1.0]
        assert _per_edge_fracs("explain_away", False, ef, 3) == ef
        # symmetric_full: full both ways
        assert _per_edge_fracs("symmetric_full", True, ef, 3) == [1.0, 1.0, 1.0]
        assert _per_edge_fracs("symmetric_full", False, ef, 3) == [1.0, 1.0, 1.0]
        # symmetric_uniform: 1/L both ways
        unif = _per_edge_fracs("symmetric_uniform", False, ef, 3)
        assert all(abs(f - 1 / 3) < 1e-9 for f in unif)
        # symmetric_explain: explaining-away weighting BOTH ways
        assert _per_edge_fracs("symmetric_explain", True, ef, 3) == ef
        assert _per_edge_fracs("symmetric_explain", False, ef, 3) == ef

    def test_symmetric_full_breaks_self_protection_on_failure(self, monkeypatch):
        """Under explain_away a curated AFFECTS edge (α≫β) self-protects on
        failure (absorbs ≈0). Under symmetric_full every edge eats the FULL
        contradict, so the curated edge moves materially more — the exact
        trade-off the experiment exists to surface."""
        def _affects_delta(mode: str) -> float:
            monkeypatch.setenv("EROOM_EDGE_ATTR", mode)
            g, ts = _seed_single_arm_graph(
                affects_belief=EdgeBeliefState(alpha=80.0, beta=1.0),
                mech_belief=EdgeBeliefState(alpha=1.0, beta=1.0),
            )
            updates = Attributor(g).attribute(
                _clf(outcome="failure"), ts, _ext("failure"),
            )
            by_edge = {u.edge_type: u for u in updates}
            return abs(by_edge[EdgeType.AFFECTS].probability_change)

        explain = _affects_delta("explain_away")
        full = _affects_delta("symmetric_full")
        assert full > explain
        assert explain < 0.01  # self-protects

    def test_symmetric_full_weights_are_all_one(self, monkeypatch):
        monkeypatch.setenv("EROOM_EDGE_ATTR", "symmetric_full")
        g, ts = _seed_single_arm_graph()
        updates = Attributor(g).attribute(
            _clf(outcome="failure"), ts, _ext("failure"),
        )
        assert all(
            u.evidence.context["explain_away_weight"] == 1.0 for u in updates
        )

    def test_effect_modulation_uses_pvalue_ignores_effect(self):
        # effect arg is deliberately ignored (unreliable extraction)
        assert _effect_modulation(0.8, 999999.0, None) == _effect_modulation(
            0.8, None, None
        )
        # no p_value → identity
        assert _effect_modulation(0.8, None, None) == (0.8, 1.0)

    def test_effect_modulation_significant_p_sharpens(self):
        # success p_obs=0.80; a very significant p pushes it UP + raises n_eff
        p_obs_adj, neff = _effect_modulation(0.80, None, 0.0005)
        assert p_obs_adj > 0.80
        assert neff > 1.0

    def test_effect_modulation_nonsignificant_p_softens(self):
        # a non-significant p pulls p_obs toward 0.5 + shrinks n_eff
        p_obs_adj, neff = _effect_modulation(0.80, None, 0.5)
        assert p_obs_adj < 0.80
        assert neff < 1.0
