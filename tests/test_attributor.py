"""Tests for chain-aware attribution.

The attributor takes a classifier-emitted edge update with free-text
source/target entity names and routes it to the specific CausalChain
whose canonical ids match. The canonical example: in CheckMate 067 the
classifier emits both ``Nivolumab → PD-1`` and ``Ipilimumab → CTLA-4``
binds_to updates; the attributor must route each to the right chain
rather than blindly using the first compound/target tuple.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from src.annotation import attributor as _attributor_module
from src.annotation.attributor import (
    AppliedEdgeUpdate,
    Attributor,
    _PHASE_TO_EVIDENCE,
    _norm_name,
)
from src.annotation.taxonomy import (
    ChainResult,
    ExtractedArm,
    FailureClassification,
    FailureMode,
    ModulationEntry,
    TrialExtraction,
)
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
    TrialArm,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket


# ── Helpers ─────────────────────────────────────────────────────────────


def _seed_combo_trial_graph() -> tuple[GraphStore, TrialSubgraph]:
    """Set up a CheckMate-067-shaped graph + trial subgraph.

    Three arms (nivo mono, ipi mono, combo). One subgroup population.
    Two distinct binds_to edges: nivolumab → PD-1, ipilimumab → CTLA-4.
    """
    g = GraphStore()
    g.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
    g.add_node(CompoundNode(id="ipilimumab", name="Ipilimumab", modality=Modality.ANTIBODY))
    g.add_node(CompoundNode(id="ipilimumab+nivolumab", name="ipi+nivo", modality=Modality.OTHER))
    g.add_node(TargetNode(id="ENSG00000188389", name="Programmed cell death 1", gene_symbol="PD-1"))
    g.add_node(TargetNode(id="ENSG00000163599", name="Cytotoxic T-lymphocyte protein 4", gene_symbol="CTLA4"))
    g.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade", mechanism_type=MechanismType.ANTAGONISM))
    g.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 signaling"))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(
        id="PFS_melanoma", name="PFS (Melanoma)",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="melanoma__unselected", name="All patients (Melanoma)"))

    # Per-compound binds_to edges with their own beliefs.
    g.add_edge(GraphEdge(source_id="nivolumab", target_id="ENSG00000188389",
                        edge_type=EdgeType.AFFECTS,
                        belief=EdgeBeliefState(alpha=4.0, beta=1.0)))
    g.add_edge(GraphEdge(source_id="ipilimumab", target_id="ENSG00000163599",
                        edge_type=EdgeType.AFFECTS,
                        belief=EdgeBeliefState(alpha=4.0, beta=1.0)))

    arms = [
        TrialArm(arm_id="nivo_only", compound_ids=["nivolumab"],
                 regimen_compound_id="nivolumab"),
        TrialArm(arm_id="combo", compound_ids=["nivolumab", "ipilimumab"],
                 regimen_compound_id="ipilimumab+nivolumab", is_combination=True),
        TrialArm(arm_id="ipi_only", compound_ids=["ipilimumab"],
                 regimen_compound_id="ipilimumab"),
    ]
    target_for_arm = {
        "nivo_only": "ENSG00000188389",
        "combo": "ENSG00000188389",
        "ipi_only": "ENSG00000163599",
    }
    chains = [
        CausalChain(
            arm_id=a.arm_id, compound_id=a.regimen_compound_id,
            subgroup_population_id="melanoma__unselected",
            target_id=target_for_arm[a.arm_id], mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        )
        for a in arms
    ]
    ts = TrialSubgraph(
        trial_id="NCT_TEST", phase="3", arms=arms, chains=chains,
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


def _make_classification(
    raw_edges: list[dict],
    *,
    trial_outcome: str | None = None,
    operational_failure: bool | None = None,
    confidence: float = 0.7,
) -> FailureClassification:
    clf = FailureClassification(
        trial_id="NCT_TEST",
        primary_failure_mode=FailureMode.EFFICACY_IN_SUBGROUP_ONLY,
        confidence=confidence,
        evidence_quotes=["test"],
        operational_failure=operational_failure,
    )
    raw: dict = {"edges_to_update": raw_edges}
    if trial_outcome is not None:
        raw["trial_outcome"] = trial_outcome
    clf._raw = raw  # type: ignore[attr-defined]
    return clf


def _outcome_extraction(
    arm_outcomes: dict[str, str],
    *,
    sample_size: int | None = 100,
    arm_compounds: dict[str, list[str]] | None = None,
) -> TrialExtraction:
    """Build a TrialExtraction whose results_by_chain + arms drive the
    outcome-conditioning attributor.

    ``arm_outcomes`` maps the (LLM) arm_id → outcome string. ``arm_compounds``
    maps the same arm_id → its constituent compound names so
    ``_map_extraction_arms_to_graph`` can match the extraction arm onto the
    graph arm by compound set. Defaults to the CheckMate-067 fixture arms.
    """
    if arm_compounds is None:
        arm_compounds = {
            "nivo_only": ["nivolumab"],
            "ipi_only": ["ipilimumab"],
            "ipi_mono": ["ipilimumab"],
            "combo": ["nivolumab", "ipilimumab"],
        }
    arms = [
        ExtractedArm(arm_id=arm_id, compounds=arm_compounds.get(arm_id, []))
        for arm_id in arm_outcomes
    ]
    results = [
        ChainResult(arm_id=arm_id, outcome=outcome, endpoint="PFS")
        for arm_id, outcome in arm_outcomes.items()
    ]
    return TrialExtraction(
        trial_id="NCT_TEST",
        sample_size=sample_size,
        arms=arms,
        results_by_chain=results,
    )


@pytest.fixture(autouse=True)
def _isolated_unrouted_log(tmp_path, monkeypatch):
    """Redirect the unrouted-attribution audit log to a per-test tmp
    file so tests never read or unlink the real `data/dev/...jsonl`.
    """
    log_path = tmp_path / "unrouted_attribution_updates.jsonl"
    monkeypatch.setattr(_attributor_module, "_UNROUTED_LOG_PATH", log_path)
    yield log_path


@pytest.fixture(autouse=True)
def _isolated_unrouted_mod_log(tmp_path, monkeypatch):
    """Same idea for the v0.3.0 modulation unrouted log."""
    log_path = tmp_path / "unrouted_modulation_entries.jsonl"
    monkeypatch.setattr(_attributor_module, "_UNROUTED_MOD_LOG_PATH", log_path)
    yield log_path


# ── Phase mapping (preserved from original test) ────────────────────────


class TestPhaseMapping:
    def test_phase3_maps_to_clinical_phase3(self):
        assert _PHASE_TO_EVIDENCE["3"] == EvidenceType.CLINICAL_PHASE3

    def test_phase2_maps_to_clinical_phase2(self):
        assert _PHASE_TO_EVIDENCE["2"] == EvidenceType.CLINICAL_PHASE2

    def test_phase1_maps_to_clinical_phase1(self):
        assert _PHASE_TO_EVIDENCE["1"] == EvidenceType.CLINICAL_PHASE1

    def test_phase2_3_maps_to_phase3(self):
        assert _PHASE_TO_EVIDENCE["2/3"] == EvidenceType.CLINICAL_PHASE3


# ── Name normalization (the hyphen-strip fix that unblocks CTLA-4) ──────


class TestNameNormalization:
    def test_strips_hyphens(self):
        assert _norm_name("CTLA-4") == _norm_name("CTLA4") == "ctla4"

    def test_strips_spaces_and_lowercases(self):
        assert _norm_name("PD L1") == _norm_name("PD-L1") == _norm_name("pdl1") == "pdl1"

    def test_empty_returns_empty(self):
        assert _norm_name("") == ""


# ── Outcome-conditions-the-chain backbone (replaces name-matched routing) ──
#
# Under the outcome-conditioning redesign the backbone is no longer
# attributed by name-matching a classifier-emitted failing edge to a chain
# edge. The trial's per-arm OUTCOME conditions the WHOLE chain by edge id.
# These tests replace the old TestChainAwareRouting / TestAffectingArmIdRouting
# name-match coverage with the outcome-conditioning semantics that supersede
# it. (Off-trial-entity hallucination + sparse-chain unrouted logging only
# applied to the deleted name-match path and no longer have a backbone analog;
# the modulation paths still guard their own entity resolution and keep their
# unrouted-log tests below.)


class TestOutcomeConditioning:
    def test_failure_conditions_each_arms_own_chain(self):
        """A per-arm FAILURE conditions every live backbone edge of that
        arm's chain — by edge id, with no name-matching. Nivolumab's chain
        (nivo_only arm) gets contradict evidence on its AFFECTS edge."""
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([], trial_outcome="failure")
        ext = _outcome_extraction({"nivo_only": "failure"})
        updates = Attributor(g).attribute(clf, ts, ext)
        affects = [
            u for u in updates
            if u.edge_type == EdgeType.AFFECTS
            and u.source_id == "nivolumab"
            and u.target_id == "ENSG00000188389"
        ]
        assert len(affects) == 1
        # Failure → the edge belief moved DOWN (contradict mass).
        assert affects[0].probability_change < 0
        # Every record is tagged outcome-conditioned for provenance.
        assert affects[0].evidence.context.get("outcome_conditioned") is True

    def test_success_moves_every_backbone_edge_up(self):
        """SUCCESS is conjunctive: every backbone edge on the arm's chain
        gets a SUPPORT update. No edge moves DOWN; edges that start below the
        support p_obs (0.80) move strictly UP."""
        g, ts = _seed_combo_trial_graph()
        # Seed the rest of the backbone edges (all fresh α=1,β=1 → E[p]=0.5)
        # so the whole chain is live.
        _seed_full_backbone(g)
        clf = _make_classification([], trial_outcome="success")
        ext = _outcome_extraction({"nivo_only": "success"})
        updates = Attributor(g).attribute(clf, ts, ext)
        nivo_updates = [u for u in updates if u.evidence.context.get("arm_id") == "nivo_only"]
        # AFFECTS, MODULATES_VIA, MECHANISM_AFFECTS, BIOLOGY_DRIVES,
        # REFLECTS_BIOLOGY, ENDPOINT_CAPTURES = 6 live edges.
        assert len(nivo_updates) == 6
        # Whole chain starts fresh (Beta(1,1)); a success support update
        # moves every backbone edge strictly UP.
        assert all(u.probability_change > 0 for u in nivo_updates)

    def test_each_arm_conditions_independently(self):
        """In a combo trial the nivo_only and ipi_only arms condition their
        OWN per-constituent AFFECTS edges — ipi failure lands on ipi→CTLA4,
        not nivo→PD-1 (the old misrouting bug, now structural)."""
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([], trial_outcome="failure")
        ext = _outcome_extraction(
            {"nivo_only": "failure", "ipi_only": "failure"}
        )
        updates = Attributor(g).attribute(clf, ts, ext)
        routes = {
            (u.source_id, u.target_id)
            for u in updates if u.edge_type == EdgeType.AFFECTS
        }
        assert ("nivolumab", "ENSG00000188389") in routes
        assert ("ipilimumab", "ENSG00000163599") in routes

    def test_no_outcome_at_all_conditions_nothing(self):
        """When NEITHER a per-arm NOR a trial-level outcome is known
        (extraction empty AND trial_outcome unknown/absent), the backbone
        gets no conditioning evidence."""
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([])  # no trial_outcome, no extraction
        updates = Attributor(g).attribute(clf, ts)
        backbone = [
            u for u in updates
            if u.edge_type in (
                EdgeType.AFFECTS, EdgeType.MODULATES_VIA,
                EdgeType.MECHANISM_AFFECTS, EdgeType.BIOLOGY_DRIVES,
            )
        ]
        assert backbone == []

    def test_trial_level_outcome_conditions_when_per_arm_missing(self):
        """When per-arm outcomes don't resolve but the classifier reports a
        TRIAL-LEVEL outcome, it conditions every arm's chain (the coarsest
        valid signal — recovers chains per-arm conditioning would skip)."""
        g, ts = _seed_combo_trial_graph()
        # trial_outcome=failure, but NO extraction → per-arm unresolved.
        clf = _make_classification([], trial_outcome="failure")
        updates = Attributor(g).attribute(clf, ts)
        affects = [u for u in updates if u.edge_type == EdgeType.AFFECTS]
        # Both per-constituent AFFECTS edges conditioned (nivo + ipi arms).
        routes = {(u.source_id, u.target_id) for u in affects}
        assert ("nivolumab", "ENSG00000188389") in routes
        assert ("ipilimumab", "ENSG00000163599") in routes
        assert all(u.probability_change < 0 for u in affects)  # failure → down

    def test_curated_affects_edge_barely_moves_on_failure(self):
        """Explaining-away: a high-E[p] curated AFFECTS edge (α≫β) absorbs
        almost none of one trial's failure mass, while a near-0.5 uncertain
        mechanism_affects edge absorbs most. The |Δp| ordering proves the
        weak link takes the blame."""
        g, ts = _seed_combo_trial_graph()
        # Curated, near-certain AFFECTS edge (already α≫β in the fixture:
        # alpha=4, beta=1 → E[p]=0.8). Push it higher to make the contrast
        # stark.
        g.add_edge(GraphEdge(
            source_id="checkpoint_blockade", target_id="R-HSA-389948",
            edge_type=EdgeType.MECHANISM_AFFECTS,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0),  # E[p]=0.5, uncertain
        ))
        # Strengthen the AFFECTS edge to near-certainty.
        g._graph.edges["nivolumab", "ENSG00000188389", "affects"]["belief"] = (
            EdgeBeliefState(alpha=50.0, beta=1.0).model_dump(mode="json")
        )
        clf = _make_classification([], trial_outcome="failure")
        ext = _outcome_extraction({"nivo_only": "failure"})
        updates = Attributor(g).attribute(clf, ts, ext)
        by_edge = {u.edge_type: u for u in updates if u.evidence.context.get("arm_id") == "nivo_only"}
        affects_delta = abs(by_edge[EdgeType.AFFECTS].probability_change)
        mech_delta = abs(by_edge[EdgeType.MECHANISM_AFFECTS].probability_change)
        # The uncertain mechanism edge absorbs MORE of the failure than the
        # curated near-certain binding edge.
        assert mech_delta > affects_delta

    def test_one_trial_cannot_collapse_an_edge(self):
        """The total failure mass w_base is SPLIT across the chain, so a
        single trial can never drive any edge to a degenerate belief."""
        g, ts = _seed_combo_trial_graph()
        _seed_full_backbone(g)
        clf = _make_classification([], trial_outcome="failure", confidence=1.0)
        ext = _outcome_extraction({"nivo_only": "failure"}, sample_size=3500)
        updates = Attributor(g).attribute(clf, ts, ext)
        for u in updates:
            post = u.post_update_belief
            assert post.expected_probability > 0.02
            assert post.expected_probability < 0.98


def _seed_full_backbone(g: GraphStore, *, reset_affects: bool = True) -> None:
    """Add the MODULATES_VIA / MECHANISM_AFFECTS / BIOLOGY_DRIVES /
    REFLECTS_BIOLOGY / ENDPOINT_CAPTURES edges that complete the
    CheckMate-067 fixture chain (the fixture only seeds the two AFFECTS
    edges).

    ``reset_affects`` (default) resets the two fixture AFFECTS edges to a
    fresh Beta(1, 1) so the WHOLE chain starts uncertain — handy for tests
    that assert every backbone edge moves on a clean outcome (the fixture
    seeds them at α=4, β=1 = E[p]=0.8, which a 0.80-p_obs support update
    leaves unmoved)."""
    if reset_affects:
        for src, tgt in [
            ("nivolumab", "ENSG00000188389"),
            ("ipilimumab", "ENSG00000163599"),
        ]:
            if g._graph.has_edge(src, tgt, key=EdgeType.AFFECTS.value):
                g._graph.edges[src, tgt, EdgeType.AFFECTS.value]["belief"] = (
                    EdgeBeliefState(alpha=1.0, beta=1.0).model_dump(mode="json")
                )
    for src, tgt, et in [
        ("ENSG00000188389", "checkpoint_blockade", EdgeType.MODULATES_VIA),
        ("ENSG00000163599", "checkpoint_blockade", EdgeType.MODULATES_VIA),
        ("checkpoint_blockade", "R-HSA-389948", EdgeType.MECHANISM_AFFECTS),
        ("R-HSA-389948", "melanoma", EdgeType.BIOLOGY_DRIVES),
        ("R-HSA-389948", "PFS_melanoma", EdgeType.REFLECTS_BIOLOGY),
        ("PFS_melanoma", "melanoma", EdgeType.ENDPOINT_CAPTURES),
    ]:
        if not g._graph.has_edge(src, tgt, key=et.value):
            g.add_edge(GraphEdge(
                source_id=src, target_id=tgt, edge_type=et,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
            ))


# ── Phase B: per-arm outcome conditioning (replaces name-match arm routing) ──


def _seed_per_constituent_combo_graph() -> tuple[GraphStore, TrialSubgraph]:
    """Two-arm trial where ipilimumab has PER-CONSTITUENT chains in both
    arms: an ipi_only mono arm and a combo arm. This mirrors what the
    real populator produces for combo trials — the combo arm contributes
    one chain per constituent compound, not one chain per regimen slug.
    """
    g, _ = _seed_combo_trial_graph()
    arms = [
        TrialArm(arm_id="ipi_mono", compound_ids=["ipilimumab"],
                 regimen_compound_id="ipilimumab"),
        TrialArm(arm_id="combo", compound_ids=["nivolumab", "ipilimumab"],
                 regimen_compound_id="ipilimumab+nivolumab", is_combination=True),
    ]
    chains = [
        # ipi_only arm: single ipi chain.
        CausalChain(
            arm_id="ipi_mono", compound_id="ipilimumab",
            subgroup_population_id="melanoma__unselected",
            target_id="ENSG00000163599", mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        ),
        # Combo arm: one chain per constituent.
        CausalChain(
            arm_id="combo", compound_id="nivolumab",
            subgroup_population_id="melanoma__unselected",
            target_id="ENSG00000188389", mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        ),
        CausalChain(
            arm_id="combo", compound_id="ipilimumab",
            subgroup_population_id="melanoma__unselected",
            target_id="ENSG00000163599", mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        ),
    ]
    ts = TrialSubgraph(
        trial_id="NCT_TEST", phase="3", arms=arms, chains=chains,
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


class TestAffectingArmIdRouting:
    def test_unmappable_extraction_arm_falls_back_to_per_arm_skip(self):
        """An extraction arm whose id/compounds match no graph arm yields no
        per-arm mapping. With NO trial-level outcome either, the backbone
        gets no conditioning (the outcome-conditioning analog of the old
        invalid-arm-slug drop)."""
        g, ts = _seed_per_constituent_combo_graph()
        clf = _make_classification([])  # no trial-level outcome
        ext = _outcome_extraction(
            {"phantom_arm": "failure"},
            arm_compounds={"phantom_arm": ["not_a_real_compound"]},
        )
        updates = Attributor(g).attribute(clf, ts, ext)
        backbone = [
            u for u in updates
            if u.edge_type in (
                EdgeType.AFFECTS, EdgeType.MODULATES_VIA,
                EdgeType.MECHANISM_AFFECTS, EdgeType.BIOLOGY_DRIVES,
            )
        ]
        assert backbone == []

    def test_per_arm_outcomes_condition_their_own_chains(self):
        """Per-constituent combo graph: ipilimumab has chains in BOTH the
        ipi_mono arm and the combo arm. Conditioning the ipi_mono and combo
        arms separately produces TWO evidence records on the
        ipilimumab→CTLA4 AFFECTS edge (one per arm), keyed by arm_id."""
        g, ts = _seed_per_constituent_combo_graph()
        clf = _make_classification([], trial_outcome="partial")
        ext = _outcome_extraction(
            {"ipi_mono": "failure", "combo": "success"},
            arm_compounds={
                "ipi_mono": ["ipilimumab"],
                "combo": ["nivolumab", "ipilimumab"],
            },
        )
        updates = Attributor(g).attribute(clf, ts, ext)
        ctla4 = [
            u for u in updates
            if u.edge_type == EdgeType.AFFECTS
            and u.source_id == "ipilimumab"
            and u.target_id == "ENSG00000163599"
        ]
        # Two records — one per arm (dedupe key includes arm_id).
        assert len(ctla4) == 2
        arms = {u.evidence.context.get("arm_id") for u in ctla4}
        assert arms == {"ipi_mono", "combo"}

    def test_same_arm_two_chains_dont_double_apply(self):
        """Two chains of the SAME arm that share a backbone edge condition
        it only once (the applied_edges dedupe keyed by
        (edge_type, src, tgt, arm_id))."""
        g, ts = _seed_per_constituent_combo_graph()
        # The combo arm has two chains (nivo + ipi constituents) that share
        # the same MECHANISM_AFFECTS edge (checkpoint_blockade → R-HSA-389948).
        g.add_edge(GraphEdge(
            source_id="checkpoint_blockade", target_id="R-HSA-389948",
            edge_type=EdgeType.MECHANISM_AFFECTS,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        ))
        clf = _make_classification([], trial_outcome="failure")
        ext = _outcome_extraction(
            {"combo": "failure"},
            arm_compounds={"combo": ["nivolumab", "ipilimumab"]},
        )
        updates = Attributor(g).attribute(clf, ts, ext)
        mech = [
            u for u in updates
            if u.edge_type == EdgeType.MECHANISM_AFFECTS
            and u.evidence.context.get("arm_id") == "combo"
        ]
        # Shared edge conditioned once for the combo arm, not twice.
        assert len(mech) == 1


# ── Failure-trial: outcome-conditioning replaces the round-14 backstop ──────


class TestFailureNeverSilent:
    """The round-14 failure-trial backstop (auto-emit a default
    biology_drives weak_contradict when the classifier returned zero edges)
    is removed under the outcome-conditioning redesign: a failure trial with
    a known arm outcome ALWAYS conditions every live backbone edge of its
    chain, so it can never be silent. These tests pin that guarantee."""

    def _seed_with_biology_drives(self):
        g, ts = _seed_combo_trial_graph()
        # Production graphs always have this edge (populate.py builds it).
        # The combo-trial fixture omits it, so add it here.
        g.add_edge(GraphEdge(
            source_id="R-HSA-389948", target_id="melanoma",
            edge_type=EdgeType.BIOLOGY_DRIVES,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        ))
        return g, ts

    def test_failure_conditions_biology_drives_among_others(self):
        """A failure trial conditions the chain's biology_drives edge (the
        structural guarantee the old backstop hand-rolled) plus every other
        live backbone edge — all with contradict mass."""
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([], trial_outcome="failure")
        ext = _outcome_extraction({"nivo_only": "failure"})
        updates = Attributor(g).attribute(clf, ts, ext)
        bio = [
            u for u in updates
            if u.edge_type == EdgeType.BIOLOGY_DRIVES
            and u.source_id == "R-HSA-389948"
            and u.target_id == "melanoma"
        ]
        assert len(bio) == 1
        assert bio[0].probability_change < 0  # contradict moves it down

    def test_trial_level_failure_conditions_without_extraction(self):
        """A trial-level failure conditions the chain even with no extraction
        (per-arm outcomes unresolved) — the trial-level-outcome fallback is
        the redesign's replacement for the round-14 backstop: a failure is
        never silent as long as SOME outcome (per-arm or trial-level) is
        known."""
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([], trial_outcome="failure")
        updates = Attributor(g).attribute(clf, ts)  # no extraction
        bio = [
            u for u in updates if u.edge_type == EdgeType.BIOLOGY_DRIVES
        ]
        # The shared biology_drives edge is conditioned once per arm (the
        # combo fixture has 3 arms; the trial-level outcome applies to all).
        assert len(bio) >= 1
        assert all(u.probability_change < 0 for u in bio)  # failure → down

    def test_no_outcome_anywhere_is_silent(self):
        """Genuine silence: no per-arm outcome AND no trial-level outcome.
        The redesign deliberately doesn't manufacture a signal when no
        outcome is known at all."""
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([])  # no trial_outcome, no extraction
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []

    def test_success_conditions_support_not_contradict(self):
        """A success trial conditions biology_drives UP (support), the
        mirror of the failure case."""
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([], trial_outcome="success")
        ext = _outcome_extraction({"nivo_only": "success"})
        updates = Attributor(g).attribute(clf, ts, ext)
        bio = [
            u for u in updates
            if u.edge_type == EdgeType.BIOLOGY_DRIVES
        ]
        assert len(bio) == 1
        assert bio[0].probability_change > 0


# ── Arm-differential modulation emission (round 8 v0.2.0) ───────────────


def _seed_subset_arm_pair(
    arm_a_outcome: TrialOutcome,
    arm_b_outcome: TrialOutcome,
    arm_a_compounds: tuple[str, ...] = ("aldesleukin",),
    arm_b_compounds: tuple[str, ...] = (
        "aldesleukin", "gp100_antigen", "montanide_isa_51_vg",
    ),
) -> tuple[GraphStore, TrialSubgraph]:
    """NCT00019682-shaped 2-arm subset comparison fixture."""
    g = GraphStore()
    for cid in set(arm_a_compounds) | set(arm_b_compounds):
        g.add_node(CompoundNode(
            id=cid, name=cid.replace("_", " "), modality=Modality.OTHER,
        ))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(
        id="RR_melanoma", name="Response rate (Melanoma)",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="melanoma__unselected", name="All patients"))

    arms = [
        TrialArm(
            arm_id="arm_a",
            compound_ids=list(arm_a_compounds),
            regimen_compound_id="+".join(sorted(arm_a_compounds))
            if len(arm_a_compounds) > 1 else arm_a_compounds[0],
            is_combination=len(arm_a_compounds) > 1,
        ),
        TrialArm(
            arm_id="arm_b",
            compound_ids=list(arm_b_compounds),
            regimen_compound_id="+".join(sorted(arm_b_compounds))
            if len(arm_b_compounds) > 1 else arm_b_compounds[0],
            is_combination=len(arm_b_compounds) > 1,
        ),
    ]
    chains = []
    for arm, outcome in [(arms[0], arm_a_outcome), (arms[1], arm_b_outcome)]:
        chains.append(CausalChain(
            arm_id=arm.arm_id,
            compound_id=arm.regimen_compound_id,
            subgroup_population_id="melanoma__unselected",
            target_id="UNKNOWN", mechanism_id="UNKNOWN",
            biology_id="UNKNOWN", indication_id="melanoma",
            endpoint_id="RR_melanoma",
            outcome=outcome,
        ))
    ts = TrialSubgraph(
        trial_id="NCT_SUBSET", phase="3", arms=arms, chains=chains,
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


class TestArmDifferentialModulation:
    def test_failure_to_success_emits_strong_support(self):
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        # Two edges: aldesleukin↔gp100, aldesleukin↔montanide.
        # (gp100×montanide is within `added` set, not emitted.)
        assert len(mod_updates) == 2
        for u in mod_updates:
            assert u.evidence.support == SupportBucket.STRONG_SUPPORT.value

    def test_endpoints_are_lex_canonicalized(self):
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        # aldesleukin < gp100_antigen, aldesleukin < montanide_isa_51_vg
        for u in mod_updates:
            assert u.source_id == "aldesleukin"
            assert u.target_id in {"gp100_antigen", "montanide_isa_51_vg"}

    def test_equal_outcomes_emit_ambiguous(self):
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.SUCCESS,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert len(mod_updates) == 2
        for u in mod_updates:
            assert u.evidence.support == SupportBucket.AMBIGUOUS.value

    def test_success_to_failure_emits_strong_contradict(self):
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.SUCCESS,
            arm_b_outcome=TrialOutcome.FAILURE,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert len(mod_updates) == 2
        for u in mod_updates:
            assert u.evidence.support == SupportBucket.STRONG_CONTRADICT.value

    def test_unknown_arm_outcome_skips_emission(self):
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.UNKNOWN,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert mod_updates == []

    def test_no_subset_relation_skips_emission(self):
        # Two disjoint mono arms (no subset, no combo): neither the
        # differential path nor the single-arm-combo path fires.
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
            arm_a_compounds=("drug_x",),
            arm_b_compounds=("drug_y",),
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert mod_updates == []

    def test_three_arm_chain_emits_all_pairs(self):
        """A vs A+B vs A+B+C: emit (A,B), (A,C), (B,C) deduped across pairs."""
        g = GraphStore()
        for cid in ("A", "B", "C"):
            g.add_node(CompoundNode(id=cid, name=cid, modality=Modality.OTHER))
        g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        g.add_node(EndpointNode(
            id="RR", name="RR",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.ACCEPTED,
        ))
        g.add_node(PopulationNode(id="melanoma__unselected", name="All"))

        arms = [
            TrialArm(arm_id="a", compound_ids=["A"], regimen_compound_id="A"),
            TrialArm(arm_id="ab", compound_ids=["A", "B"],
                     regimen_compound_id="A+B", is_combination=True),
            TrialArm(arm_id="abc", compound_ids=["A", "B", "C"],
                     regimen_compound_id="A+B+C", is_combination=True),
        ]
        outcomes = {
            "a": TrialOutcome.FAILURE,
            "ab": TrialOutcome.PARTIAL,
            "abc": TrialOutcome.SUCCESS,
        }
        chains = [
            CausalChain(
                arm_id=arm.arm_id, compound_id=arm.regimen_compound_id,
                subgroup_population_id="melanoma__unselected",
                target_id="UNKNOWN", mechanism_id="UNKNOWN",
                biology_id="UNKNOWN", indication_id="melanoma",
                endpoint_id="RR",
                outcome=outcomes[arm.arm_id],
            )
            for arm in arms
        ]
        ts = TrialSubgraph(
            trial_id="NCT_3ARM", phase="3", arms=arms, chains=chains,
            parent_population_id="melanoma__unselected",
        )
        g.set_trial_subgraph(ts)
        clf = _make_classification([])

        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        # Three distinct pairs: (A,B), (A,C), (B,C). Deduped per-trial,
        # so 3 emissions total.
        pairs = {(u.source_id, u.target_id) for u in mod_updates}
        assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}

    def test_idempotent_within_single_attribute_call(self):
        """The applied_edges dedup should prevent multi-emission across
        arm comparisons that name the same pair."""
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        pairs = [(u.source_id, u.target_id) for u in mod_updates]
        assert len(pairs) == len(set(pairs))


# ── Single-arm combo modulation (round 8 v0.2.0) ────────────────────────


def _seed_single_arm_combo(
    compounds: tuple[str, ...],
    outcome: TrialOutcome,
) -> tuple[GraphStore, TrialSubgraph]:
    """NCT00003222-shaped fixture: one combo arm with N constituents."""
    g = GraphStore()
    for cid in compounds:
        g.add_node(CompoundNode(
            id=cid, name=cid.replace("_", " "), modality=Modality.OTHER,
        ))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(
        id="OS", name="OS",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="melanoma__unselected", name="All"))

    arm = TrialArm(
        arm_id="combo",
        compound_ids=list(compounds),
        regimen_compound_id="+".join(sorted(compounds)),
        is_combination=True,
    )
    chain = CausalChain(
        arm_id=arm.arm_id, compound_id=arm.regimen_compound_id,
        subgroup_population_id="melanoma__unselected",
        target_id="UNKNOWN", mechanism_id="UNKNOWN",
        biology_id="UNKNOWN", indication_id="melanoma",
        endpoint_id="OS", outcome=outcome,
    )
    ts = TrialSubgraph(
        trial_id="NCT_SINGLE_ARM_COMBO", phase="2", arms=[arm], chains=[chain],
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


class TestSingleArmComboModulation:
    def test_six_compound_combo_emits_15_pairs(self):
        compounds = ("c1", "c2", "c3", "c4", "c5", "c6")
        g, ts = _seed_single_arm_combo(compounds, TrialOutcome.SUCCESS)
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        # C(6, 2) = 15 pairs.
        assert len(mod_updates) == 15
        for u in mod_updates:
            assert u.evidence.support == SupportBucket.WEAK_SUPPORT.value

    def test_failure_outcome_emits_weak_contradict(self):
        g, ts = _seed_single_arm_combo(("a", "b", "c"), TrialOutcome.FAILURE)
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert len(mod_updates) == 3
        for u in mod_updates:
            assert u.evidence.support == SupportBucket.WEAK_CONTRADICT.value

    def test_partial_outcome_emits_ambiguous(self):
        g, ts = _seed_single_arm_combo(("a", "b"), TrialOutcome.PARTIAL)
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert len(mod_updates) == 1
        assert mod_updates[0].evidence.support == SupportBucket.AMBIGUOUS.value

    def test_unknown_outcome_skips(self):
        g, ts = _seed_single_arm_combo(("a", "b"), TrialOutcome.UNKNOWN)
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert mod_updates == []

    def test_skipped_when_subset_comparator_exists(self):
        """If any other arm's compound set is a strict subset of this
        combo arm's set, the differential path handled it — single-arm
        emission must not double-emit on top."""
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        # Differential emits 2 (aldesleukin × gp100, aldesleukin × montanide).
        # Single-arm path must not add the (gp100, montanide) pair on top
        # since arm A's {aldesleukin} is a strict subset of arm B's set.
        pairs = {(u.source_id, u.target_id) for u in mod_updates}
        assert pairs == {
            ("aldesleukin", "gp100_antigen"),
            ("aldesleukin", "montanide_isa_51_vg"),
        }

    def test_mono_arm_skipped(self):
        """Single-compound arms have nothing to pair, so they emit
        nothing."""
        g, ts = _seed_single_arm_combo(("solo",), TrialOutcome.SUCCESS)
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert mod_updates == []


# ── Modulation evidence context (indication tagging) ────────────────────


class TestModulationEvidenceContext:
    def test_indication_tagged_on_differential_emission(self):
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert mod_updates
        for u in mod_updates:
            assert u.evidence.context.get("indication") == "melanoma"

    def test_indication_tagged_on_single_arm_emission(self):
        g, ts = _seed_single_arm_combo(("a", "b"), TrialOutcome.SUCCESS)
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert mod_updates
        for u in mod_updates:
            assert u.evidence.context.get("indication") == "melanoma"


# ── Extraction-driven per-arm outcomes (NCT00019682-shaped) ─────────────


class TestExtractionDrivenArmOutcomes:
    def test_extraction_outcomes_drive_emission_when_chain_outcomes_unknown(
        self,
    ):
        """Chain.outcome is UNKNOWN in current populate flows. The
        extraction-path reads per-arm outcomes from results_by_chain
        and maps LLM arm_ids to graph arm_ids via compound-set match."""
        # Graph arm_ids ("arm_i_aldesleukin" / "arm_ii_combo") differ from
        # extraction arm_ids ("aldesleukin_alone" / "combination").
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.UNKNOWN,
            arm_b_outcome=TrialOutcome.UNKNOWN,
        )
        # Re-label graph arm_ids to make the mismatch realistic.
        from src.graph.models import TrialSubgraph as TSG
        new_arms = [
            ts.arms[0].model_copy(update={"arm_id": "arm_i_real_id"}),
            ts.arms[1].model_copy(update={"arm_id": "arm_ii_real_id"}),
        ]
        new_chains = [
            ts.chains[0].model_copy(update={"arm_id": "arm_i_real_id"}),
            ts.chains[1].model_copy(update={"arm_id": "arm_ii_real_id"}),
        ]
        ts2 = TSG(
            trial_id=ts.trial_id, phase=ts.phase,
            arms=new_arms, chains=new_chains,
            parent_population_id=ts.parent_population_id,
        )
        g.set_trial_subgraph(ts2)

        extraction = TrialExtraction(
            trial_id=ts.trial_id,
            therapeutic_hypothesis="test",
            arms=[
                ExtractedArm(arm_id="ext_arm_a", compounds=["aldesleukin"]),
                ExtractedArm(
                    arm_id="ext_arm_b",
                    compounds=[
                        "aldesleukin", "gp100 antigen", "montanide ISA 51 VG",
                    ],
                ),
            ],
            results_by_chain=[
                ChainResult(
                    arm_id="ext_arm_a", subgroup_descriptor=None,
                    endpoint="OS", outcome="failure",
                ),
                ChainResult(
                    arm_id="ext_arm_b", subgroup_descriptor=None,
                    endpoint="OS", outcome="success",
                ),
            ],
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts2, extraction)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert len(mod_updates) == 2
        for u in mod_updates:
            assert u.evidence.support == SupportBucket.STRONG_SUPPORT.value

    def test_extraction_arm_with_no_compound_match_is_ignored(self):
        """An extraction arm whose compounds don't match any graph arm
        gets dropped; modulation emission falls back to other paths."""
        g, ts = _seed_subset_arm_pair(
            arm_a_outcome=TrialOutcome.FAILURE,
            arm_b_outcome=TrialOutcome.SUCCESS,
        )
        # Extraction declares an arm with a compound the graph doesn't know.
        extraction = TrialExtraction(
            trial_id=ts.trial_id,
            therapeutic_hypothesis="test",
            arms=[
                ExtractedArm(arm_id="ghost_arm", compounds=["mystery_drug"]),
            ],
            results_by_chain=[
                ChainResult(
                    arm_id="ghost_arm", subgroup_descriptor=None,
                    endpoint="OS", outcome="success",
                ),
            ],
        )
        clf = _make_classification([])
        updates = Attributor(g).attribute(clf, ts, extraction)
        mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        # Extraction yielded no usable mapping; chain-outcome fallback
        # produces the standard 2 edges.
        assert len(mod_updates) == 2


# ── AppliedEdgeUpdate ───────────────────────────────────────────────────


class TestAppliedEdgeUpdate:
    def test_probability_change(self):
        pre = EdgeBeliefState(alpha=2.0, beta=2.0)
        post = EdgeBeliefState(alpha=2.0, beta=5.0)
        update = AppliedEdgeUpdate(
            source_id="a", target_id="b",
            edge_type=EdgeType.AFFECTS,
            evidence=EvidenceRecord(
                source_id="trial1",
                source_type=EvidenceType.CLINICAL_PHASE3,
                support=SupportBucket.MODERATE_CONTRADICT.value,
                quality_score=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
            pre_update_belief=pre,
            post_update_belief=post,
        )
        # E[p] went from 0.5 to 2/7 ≈ 0.286 → Δ ≈ -0.214
        assert update.probability_change < 0


# ── v0.3.0 LLM modulation emission (Phase B) ────────────────────────────


def _layer_extraction(
    modulator: str = "ipilimumab",
    primary: str = "nivolumab",
    layer: str = "biology",
    direction: str = "amplifies",
    confidence: float = 0.85,
) -> TrialExtraction:
    return TrialExtraction(
        trial_id="NCT_TEST",
        modulation_entries=[ModulationEntry(
            modulator_compound_id=modulator,
            primary_compound_id=primary,
            affects_layer=layer,
            direction=direction,
            confidence=confidence,
            hypothesis="test hypothesis",
            citation="test citation",
        )],
    )


class TestLLMModulationEmission:
    def test_biology_layer_anchors_at_chain_biology_node(self):
        """LLM names primary + biology layer; populator routes to that
        primary's chain.biology_id (R-HSA-389948 for nivolumab in the
        seeded combo trial)."""
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = _layer_extraction(
            modulator="ipilimumab", primary="nivolumab", layer="biology",
        )
        clf = _make_classification([])

        updates = attributor.attribute(clf, ts, extraction)
        llm_mod_updates = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
            and "LLM modulation" in (u.evidence.notes or "")
        ]
        assert len(llm_mod_updates) == 1
        u = llm_mod_updates[0]
        assert u.source_id == "ipilimumab"
        # The seeded chain for nivolumab has biology_id=R-HSA-389948
        assert u.target_id == "R-HSA-389948"
        ctx = u.evidence.context
        assert ctx["primary_compound"] == "nivolumab"
        assert ctx["affects_layer"] == "biology"
        assert ctx["modulation_direction"] == "amplifies"
        assert u.evidence.support == SupportBucket.STRONG_SUPPORT.value

    def test_target_layer_anchors_at_chain_target_node(self):
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = _layer_extraction(
            modulator="ipilimumab", primary="nivolumab", layer="target",
        )
        clf = _make_classification([])

        updates = attributor.attribute(clf, ts, extraction)
        llm_mods = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
            and "LLM modulation" in (u.evidence.notes or "")
        ]
        assert len(llm_mods) == 1
        # Nivo's chain target is ENSG00000188389 (PD-1)
        assert llm_mods[0].target_id == "ENSG00000188389"
        assert llm_mods[0].evidence.context["affects_layer"] == "target"

    def test_mechanism_layer_anchors_at_chain_mechanism_node(self):
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = _layer_extraction(
            modulator="ipilimumab", primary="nivolumab", layer="mechanism",
        )
        clf = _make_classification([])

        updates = attributor.attribute(clf, ts, extraction)
        llm_mods = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
            and "LLM modulation" in (u.evidence.notes or "")
        ]
        assert len(llm_mods) == 1
        assert llm_mods[0].target_id == "checkpoint_blockade"

    def test_neutral_modulation_creates_ambiguous_evidence(self):
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = _layer_extraction(direction="neutral", confidence=0.85)
        clf = _make_classification([])

        updates = attributor.attribute(clf, ts, extraction)
        llm_mods = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
            and "LLM modulation" in (u.evidence.notes or "")
        ]
        assert len(llm_mods) == 1
        # neutral → AMBIGUOUS regardless of confidence per
        # feedback_trial_failure_not_falsification.
        assert llm_mods[0].evidence.support == SupportBucket.AMBIGUOUS.value

    def test_unrouted_when_modulator_unknown(self, _isolated_unrouted_mod_log):
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = _layer_extraction(modulator="some_fake_compound_xyz")
        clf = _make_classification([])

        attributor.attribute(clf, ts, extraction)
        rows = [
            json.loads(line)
            for line in _isolated_unrouted_mod_log.read_text().splitlines()
            if line.strip()
        ]
        assert any(r["reason"] == "modulator_not_in_graph" for r in rows)

    def test_unrouted_when_primary_not_in_graph(self, _isolated_unrouted_mod_log):
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = _layer_extraction(primary="some_fake_primary_xyz")
        clf = _make_classification([])

        attributor.attribute(clf, ts, extraction)
        rows = [
            json.loads(line)
            for line in _isolated_unrouted_mod_log.read_text().splitlines()
            if line.strip()
        ]
        assert any(r["reason"] == "primary_not_in_graph" for r in rows)

    def test_unrouted_when_primary_not_in_trial(self, _isolated_unrouted_mod_log):
        """Primary compound exists in graph but isn't in this trial's
        chains/arms — modulation can't be anchored to a chain layer."""
        g, ts = _seed_combo_trial_graph()
        # Add a compound to the graph but NOT to the trial.
        g.add_node(CompoundNode(
            id="bevacizumab", name="Bevacizumab", modality=Modality.ANTIBODY,
        ))
        attributor = Attributor(g)
        extraction = _layer_extraction(
            modulator="ipilimumab", primary="bevacizumab", layer="biology",
        )
        clf = _make_classification([])

        attributor.attribute(clf, ts, extraction)
        rows = [
            json.loads(line)
            for line in _isolated_unrouted_mod_log.read_text().splitlines()
            if line.strip()
        ]
        assert any(r["reason"] == "primary_chain_not_in_trial" for r in rows)

    def test_combo_constituent_resolves_via_arm_membership(self):
        """The primary may be one constituent of a combo regimen — the
        chain's compound_id is the combo slug, not the constituent. The
        resolver falls back to arm membership when chain.compound_id
        doesn't match directly."""
        g, ts = _seed_combo_trial_graph()
        # Add the biology_drives edge so the routing succeeds.
        attributor = Attributor(g)
        # In the seeded combo trial, the "combo" arm has compound_ids
        # ['nivolumab', 'ipilimumab'] but the chain's compound_id is the
        # combo slug. Using "ipilimumab" as primary still routes because
        # arm membership picks up the ipi-only chain.
        extraction = _layer_extraction(
            modulator="nivolumab", primary="ipilimumab", layer="biology",
        )
        clf = _make_classification([])

        updates = attributor.attribute(clf, ts, extraction)
        llm_mods = [
            u for u in updates
            if u.edge_type == EdgeType.MODULATES_EFFICACY_OF
            and "LLM modulation" in (u.evidence.notes or "")
        ]
        assert len(llm_mods) == 1
        # ipi's chain has biology_id=R-HSA-389948 in the seed
        assert llm_mods[0].target_id == "R-HSA-389948"

    def test_empty_modulation_entries_is_noop(self):
        g, ts = _seed_combo_trial_graph()
        attributor = Attributor(g)
        extraction = TrialExtraction(trial_id="NCT_TEST", modulation_entries=[])
        clf = _make_classification([])

        updates = attributor.attribute(clf, ts, extraction)
        llm_mods = [
            u for u in updates
            if "LLM modulation" in (u.evidence.notes or "")
        ]
        assert len(llm_mods) == 0


# ── Round-16: drop-rate counters ─────────────────────────────────────────


class TestDropCounters:
    """Round-16 observability: the Attributor exposes
    `last_attempted_updates` and `last_dropped_updates` after each
    attribute() call.

    Outcome-conditioning redesign: the backbone is conditioned by edge id
    (no name-match step that can miss), so backbone conditioning never
    increments the drop counter. ``last_attempted_updates`` still reflects
    the classifier's raw ``edges_to_update`` length (kept for back-compat
    observability), and the counters still reset per call."""

    def test_attempted_counter_reflects_raw_edges_length(self):
        graph, ts = _seed_combo_trial_graph()
        attributor = Attributor(graph)
        clf = _make_classification(
            [
                {"edge_type": "affects", "source_entity": "x",
                 "target_entity": "y", "support": "moderate_support"},
                {"edge_type": "affects", "source_entity": "p",
                 "target_entity": "q", "support": "weak_support"},
            ],
            trial_outcome="failure",
        )
        ext = _outcome_extraction({"nivo_only": "failure"})
        attributor.attribute(clf, ts, ext)
        assert attributor.last_attempted_updates == 2

    def test_outcome_conditioning_does_not_increment_drops(self):
        """Conditioning the chain by id never 'drops' — there is no
        name-match that can miss. The drop counter stays 0 even though the
        classifier's raw edges are no longer used for the backbone."""
        graph, ts = _seed_combo_trial_graph()
        attributor = Attributor(graph)
        clf = _make_classification([], trial_outcome="failure")
        ext = _outcome_extraction({"nivo_only": "failure"})
        attributor.attribute(clf, ts, ext)
        assert attributor.last_attempted_updates == 0
        assert attributor.last_dropped_updates == 0

    def test_counters_reset_per_call(self):
        """Counters reflect the most recent attribute() call only."""
        graph, ts = _seed_combo_trial_graph()
        attributor = Attributor(graph)
        ext = _outcome_extraction({"nivo_only": "failure"})

        # First call: 2 raw edges.
        clf_first = _make_classification(
            [
                {"edge_type": "affects", "source_entity": "x",
                 "target_entity": "y", "support": "ambiguous"},
                {"edge_type": "affects", "source_entity": "nivolumab",
                 "target_entity": "ENSG00000188389",
                 "support": "moderate_support"},
            ],
            trial_outcome="failure",
        )
        attributor.attribute(clf_first, ts, ext)
        assert attributor.last_attempted_updates == 2

        # Second call: 0 raw edges — counters reset.
        clf_second = _make_classification([], trial_outcome="failure")
        attributor.attribute(clf_second, ts, ext)
        assert attributor.last_attempted_updates == 0
        assert attributor.last_dropped_updates == 0


# ── Round-19: attribution idempotency ───────────────────────────────────


class TestAENodeHierarchyWiring:
    """Round-28: ``_ensure_ae_node`` looks up MedDRA hierarchy parents
    when creating an AdverseEventNode and populates the round-28
    soc_id / soc_name / hlt_id / hlgt_id fields. SOC fallback via the
    free-text SOC string keeps unknown PTs from losing their parent."""

    def _new_attributor(self) -> tuple[Attributor, GraphStore]:
        g = GraphStore()
        return Attributor(g), g

    def test_known_pt_gets_soc_from_hierarchy(self):
        attr, g = self._new_attributor()
        attr._ensure_ae_node(
            "AE:atrial_fibrillation", "Atrial fibrillation",
            "Cardiac disorders", "grade_3",
        )
        node = g.get_node("AE:atrial_fibrillation")
        assert node["soc_id"] == "cardiac_disorders"
        assert node["soc_name"] == "Cardiac disorders"

    def test_pt_outside_curated_table_falls_back_via_soc_name(self):
        attr, g = self._new_attributor()
        # A made-up cardiac AE the LLM tagged as Cardiac disorders but
        # which isn't in the curated PT→SOC map.
        attr._ensure_ae_node(
            "AE:fictional_arrhythmia", "Fictional arrhythmia",
            "Cardiac disorders", "grade_2",
        )
        node = g.get_node("AE:fictional_arrhythmia")
        # Fallback via the LLM-emitted SOC name lands the AE under
        # cardiac_disorders so target-class roll-up still aggregates.
        assert node["soc_id"] == "cardiac_disorders"
        assert node["soc_name"] == "Cardiac disorders"

    def test_unknown_pt_unknown_soc_keeps_empty(self):
        attr, g = self._new_attributor()
        attr._ensure_ae_node(
            "AE:thoroughly_unknown", "Thoroughly Unknown",
            "Not a real SOC", "",
        )
        node = g.get_node("AE:thoroughly_unknown")
        assert node["soc_id"] == ""
        # soc_name falls back to the LLM's free-text string so renderers
        # have something to show even when the hierarchy can't resolve it.
        assert node["soc_name"] == "Not a real SOC"

    def test_repeated_call_doesnt_clobber_existing_soc(self):
        attr, g = self._new_attributor()
        attr._ensure_ae_node(
            "AE:atrial_fibrillation", "Atrial fibrillation",
            "Cardiac disorders", "grade_2",
        )
        # Second call with a different "grade" should extend severity_range
        # without resetting hierarchy fields.
        attr._ensure_ae_node(
            "AE:atrial_fibrillation", "Atrial fibrillation",
            "Cardiac disorders", "grade_3",
        )
        node = g.get_node("AE:atrial_fibrillation")
        assert "grade_2" in node["severity_range"]
        assert "grade_3" in node["severity_range"]
        assert node["soc_id"] == "cardiac_disorders"

    def test_legacy_node_without_hierarchy_gets_backfilled(self):
        from src.graph.models import AdverseEventNode
        attr, g = self._new_attributor()
        # Simulate a node created by a pre-round-28 snapshot (no SOC ids).
        g.add_node(AdverseEventNode(
            id="AE:atrial_fibrillation",
            name="Atrial fibrillation",
            system_organ_class="Cardiac disorders",
            severity_range="grade_2",
        ))
        node = g.get_node("AE:atrial_fibrillation")
        assert node["soc_id"] == ""
        # Now _ensure_ae_node should backfill on re-attribution.
        attr._ensure_ae_node(
            "AE:atrial_fibrillation", "Atrial fibrillation",
            "Cardiac disorders", "grade_3",
        )
        node = g.get_node("AE:atrial_fibrillation")
        assert node["soc_id"] == "cardiac_disorders"
        assert node["soc_name"] == "Cardiac disorders"


class TestAttributionIdempotency:
    """Round-19 incremental-build safety net. Re-running attribution on a
    trial whose updates already landed must be a no-op — otherwise every
    incremental --add-trials run would double-count Beta-Binomial
    evidence on every edge the trial originally touched."""

    def _write_annotation_pair(
        self, annotations_dir, nct_id: str,
    ) -> None:
        """Drop the minimal extraction + classification JSON the
        attributor's `_main` needs to attempt an update on the
        seeded combo-trial graph."""
        extraction = {
            "nct_id": nct_id,
            "trial_outcome": "failure",
            "title": "Test",
            "phase": "3",
            "sample_size": 100,
            "compounds": ["Nivolumab"],
            "arms": [
                {
                    "arm_id": "nivo_only",
                    "compounds": ["nivolumab"],
                    "label": "nivo monotherapy",
                    "n": 100,
                },
            ],
            # Outcome-conditioning needs a per-arm outcome to fold onto the
            # chain. A failure conditions nivolumab→PD-1 (the edge these
            # idempotency tests assert on).
            "results_by_chain": [
                {"arm_id": "nivo_only", "outcome": "failure",
                 "endpoint": "PFS"},
            ],
            "adverse_events": [],
            "modulation_entries": [],
            "primary_endpoint_met": False,
        }
        classification = {
            "nct_id": nct_id,
            "trial_outcome": "partial",
            "failure_modes": [
                {"mode": "efficacy_in_subgroup_only", "confidence": 0.7},
            ],
            "confidence_overall": 0.7,
            "reasoning": "test",
            "edges_to_update": [
                {
                    "edge_type": "affects",
                    "source_entity": "Nivolumab",
                    "target_entity": "PD-1",
                    "support": "moderate_support",
                    "affecting_arm_id": "nivo_only",
                },
            ],
        }
        (annotations_dir / f"{nct_id}_extraction.json").write_text(
            json.dumps(extraction)
        )
        (annotations_dir / f"{nct_id}_classification.json").write_text(
            json.dumps(classification)
        )

    @pytest.mark.asyncio
    async def test_second_run_doesnt_double_count(self, tmp_path):
        from src.annotation.attributor import _main as attributor_main

        # Seed a graph + trial subgraph, persist to disk so _main can
        # import_snapshot from it.
        graph, ts = _seed_combo_trial_graph()
        # _main looks up the trial by clf_data["nct_id"] which we set to
        # NCT_TEST in _seed_combo_trial_graph.
        graph_path = tmp_path / "graph_initial.json"
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()
        out_path_1 = tmp_path / "graph_annotated_1.json"
        out_path_2 = tmp_path / "graph_annotated_2.json"

        graph.export_snapshot(str(graph_path))
        self._write_annotation_pair(annotations_dir, "NCT_TEST")

        # First run: applies the affects update. Capture the edge's
        # evidence count after.
        await attributor_main(
            str(annotations_dir), str(graph_path), str(out_path_1),
        )
        after_first = GraphStore()
        after_first.import_snapshot(str(out_path_1))
        belief_1 = after_first.get_edge_belief(
            "nivolumab", "ENSG00000188389", EdgeType.AFFECTS,
        )
        assert "NCT_TEST" in after_first.applied_attribution_trial_ids
        assert len(belief_1.evidence) == 1, (
            "First run should add exactly one evidence record"
        )

        # Second run starts from the first run's output — applied set is
        # already populated. Idempotency guard MUST skip the trial.
        await attributor_main(
            str(annotations_dir), str(out_path_1), str(out_path_2),
        )
        after_second = GraphStore()
        after_second.import_snapshot(str(out_path_2))
        belief_2 = after_second.get_edge_belief(
            "nivolumab", "ENSG00000188389", EdgeType.AFFECTS,
        )
        assert len(belief_2.evidence) == 1, (
            "Second run must NOT add a duplicate evidence record"
        )
        assert belief_2.alpha == belief_1.alpha
        assert belief_2.beta == belief_1.beta

    @pytest.mark.asyncio
    async def test_new_trial_in_same_run_still_attributed(self, tmp_path):
        """Idempotency guard skips only the already-attributed trial,
        not subsequent NEW trials whose annotations are sitting in the
        same directory."""
        from src.annotation.attributor import _main as attributor_main

        graph, ts_a = _seed_combo_trial_graph()

        # Add a second trial subgraph (NCT_NEW) sharing the same nodes —
        # a separate arm_id but referencing the same compound/target.
        new_arm = TrialArm(
            arm_id="nivo_only_new", compound_ids=["nivolumab"],
            regimen_compound_id="nivolumab",
        )
        new_chain = CausalChain(
            arm_id="nivo_only_new", compound_id="nivolumab",
            subgroup_population_id="melanoma__unselected",
            target_id="ENSG00000188389", mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        )
        ts_b = TrialSubgraph(
            trial_id="NCT_NEW", phase="3", arms=[new_arm],
            chains=[new_chain],
            parent_population_id="melanoma__unselected",
        )
        graph.set_trial_subgraph(ts_b)

        # Pretend NCT_TEST is already attributed.
        graph.applied_attribution_trial_ids.add("NCT_TEST")

        graph_path = tmp_path / "graph_initial.json"
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()
        out_path = tmp_path / "graph_annotated.json"

        graph.export_snapshot(str(graph_path))
        self._write_annotation_pair(annotations_dir, "NCT_TEST")
        # NCT_NEW classification uses its own arm_id.
        new_ext = {
            "nct_id": "NCT_NEW", "trial_outcome": "failure",
            "title": "Test", "phase": "3", "sample_size": 100,
            "compounds": ["Nivolumab"],
            "arms": [{
                "arm_id": "nivo_only_new",
                "compounds": ["nivolumab"],
                "label": "nivo monotherapy",
                "n": 100,
            }],
            "results_by_chain": [
                {"arm_id": "nivo_only_new", "outcome": "failure",
                 "endpoint": "PFS"},
            ],
            "adverse_events": [],
            "modulation_entries": [], "primary_endpoint_met": False,
        }
        new_clf = {
            "nct_id": "NCT_NEW", "trial_outcome": "partial",
            "failure_modes": [
                {"mode": "efficacy_in_subgroup_only", "confidence": 0.7},
            ],
            "confidence_overall": 0.7, "reasoning": "test",
            "edges_to_update": [{
                "edge_type": "affects",
                "source_entity": "Nivolumab",
                "target_entity": "PD-1",
                "support": "moderate_support",
                "affecting_arm_id": "nivo_only_new",
            }],
        }
        (annotations_dir / "NCT_NEW_extraction.json").write_text(
            json.dumps(new_ext)
        )
        (annotations_dir / "NCT_NEW_classification.json").write_text(
            json.dumps(new_clf)
        )

        await attributor_main(
            str(annotations_dir), str(graph_path), str(out_path),
        )

        result = GraphStore()
        result.import_snapshot(str(out_path))
        assert "NCT_TEST" in result.applied_attribution_trial_ids
        assert "NCT_NEW" in result.applied_attribution_trial_ids

        # NCT_NEW's update lands; the original NCT_TEST update doesn't.
        belief = result.get_edge_belief(
            "nivolumab", "ENSG00000188389", EdgeType.AFFECTS,
        )
        ev_sources = [e.source_id for e in belief.evidence]
        assert "NCT_NEW" in ev_sources
        assert "NCT_TEST" not in ev_sources

    @pytest.mark.asyncio
    async def test_exclude_from_attribution_skips_holdout_evidence(
        self, tmp_path,
    ):
        """Round-26: NCTs passed in exclude_from_attribution have their
        subgraph populated but their evidence is NOT folded into edge
        beliefs. This is what makes a true holdout possible — the
        chain structure is available for prediction, but the holdout's
        own trial outcomes don't leak into the graph it's being scored
        against.
        """
        from src.annotation.attributor import _main as attributor_main

        graph, ts = _seed_combo_trial_graph()
        graph_path = tmp_path / "graph_initial.json"
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()
        out_path = tmp_path / "graph_annotated.json"

        graph.export_snapshot(str(graph_path))
        self._write_annotation_pair(annotations_dir, "NCT_TEST")

        # NCT_TEST is in the annotations dir but listed as excluded.
        # Subgraph stays in the graph (chain prediction can still run
        # on it); evidence is never folded into the AFFECTS edge.
        await attributor_main(
            str(annotations_dir), str(graph_path), str(out_path),
            exclude_from_attribution=["NCT_TEST"],
        )

        result = GraphStore()
        result.import_snapshot(str(out_path))

        # The excluded NCT is marked attributed (so future re-runs of
        # attribute also skip it), but no evidence record landed on the
        # edge.
        assert "NCT_TEST" in result.applied_attribution_trial_ids
        belief = result.get_edge_belief(
            "nivolumab", "ENSG00000188389", EdgeType.AFFECTS,
        )
        ev_sources = [e.source_id for e in belief.evidence]
        assert "NCT_TEST" not in ev_sources, (
            "Holdout NCT must not appear in any edge's evidence list"
        )
        # Subgraph still present for prediction.
        assert "NCT_TEST" in {
            ts.trial_id for ts in result.trial_subgraphs.values()
        }
