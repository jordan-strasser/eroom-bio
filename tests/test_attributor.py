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
    FailureClassification,
    FailureMode,
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


def _make_classification(raw_edges: list[dict]) -> FailureClassification:
    clf = FailureClassification(
        trial_id="NCT_TEST",
        primary_failure_mode=FailureMode.EFFICACY_IN_SUBGROUP_ONLY,
        confidence=0.7,
        evidence_quotes=["test"],
    )
    clf._raw = {"edges_to_update": raw_edges}  # type: ignore[attr-defined]
    return clf


@pytest.fixture(autouse=True)
def _isolated_unrouted_log(tmp_path, monkeypatch):
    """Redirect the unrouted-attribution audit log to a per-test tmp
    file so tests never read or unlink the real `data/dev/...jsonl`.
    """
    log_path = tmp_path / "unrouted_attribution_updates.jsonl"
    monkeypatch.setattr(_attributor_module, "_UNROUTED_LOG_PATH", log_path)
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


# ── Chain-aware routing ─────────────────────────────────────────────────


class TestChainAwareRouting:
    def test_nivo_pd1_update_routes_to_nivo_chain(self):
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Nivolumab",
             "target_entity": "PD-1", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        assert len(updates) == 1
        assert updates[0].source_id == "nivolumab"
        assert updates[0].target_id == "ENSG00000188389"

    def test_ipi_ctla4_update_routes_to_ipi_chain_not_pd1(self):
        """The original misrouting bug: Ipilimumab→CTLA-4 must NOT land on
        the nivolumab→PD-1 binds_to edge in a multi-arm trial."""
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Ipilimumab",
             "target_entity": "CTLA-4", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        assert len(updates) == 1
        assert updates[0].source_id == "ipilimumab"
        assert updates[0].target_id == "ENSG00000163599"

    def test_both_binds_updates_route_independently(self):
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Nivolumab",
             "target_entity": "PD-1", "support": "moderate_support"},
            {"edge_type": "binds_to", "source_entity": "Ipilimumab",
             "target_entity": "CTLA-4", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        routes = {(u.source_id, u.target_id) for u in updates}
        assert ("nivolumab", "ENSG00000188389") in routes
        assert ("ipilimumab", "ENSG00000163599") in routes

    def test_off_trial_entity_is_dropped_as_hallucination(self):
        """Classifier emits an entity (Pembrolizumab) that is nowhere in
        the trial subgraph. Candidate (compound, target) pairs exist but
        none match the classifier names—graph-build trusts only
        trial-derived entities, so this is rejected as a hallucination
        rather than misrouted to whichever pair happened to be present.
        """
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Pembrolizumab",
             "target_entity": "PD-1", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []
        assert _attributor_module._UNROUTED_LOG_PATH.exists()
        records = [
            json.loads(line)
            for line in _attributor_module._UNROUTED_LOG_PATH.read_text().splitlines()
        ]
        hallucinations = [r for r in records if r["source_entity"] == "Pembrolizumab"]
        assert hallucinations, "expected the off-trial entity to be logged"
        assert all(r["reason"] == "entity_not_in_trial" for r in hallucinations)

    def test_sparse_chain_logs_no_chain_match_not_hallucination(self):
        """When the trial subgraph has UNKNOWN placeholders so no
        candidate (src, tgt) pairs can be formed for the requested edge
        type, the update is still dropped—but logged as
        ``no_chain_match`` (chain too sparse to verify) rather than
        ``entity_not_in_trial`` (classifier hallucinated). The
        distinction matters: hallucination is a model-quality signal,
        sparse chains are an ingestion-coverage signal.
        """
        g = GraphStore()
        g.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        g.add_node(PopulationNode(id="melanoma__unselected", name="All patients"))
        # Note: NO TargetNode and NO mechanism_affects-relevant nodes —
        # the trial chain will reference "UNKNOWN" for those.

        arms = [TrialArm(arm_id="solo", compound_ids=["nivolumab"],
                         regimen_compound_id="nivolumab")]
        chains = [
            CausalChain(
                arm_id="solo", compound_id="nivolumab",
                subgroup_population_id="melanoma__unselected",
                target_id="UNKNOWN", mechanism_id="UNKNOWN",
                biology_id="UNKNOWN", indication_id="melanoma",
                endpoint_id="UNKNOWN", outcome=TrialOutcome.UNKNOWN,
            )
        ]
        ts = TrialSubgraph(
            trial_id="NCT_SPARSE", phase="3", arms=arms, chains=chains,
            parent_population_id="melanoma__unselected",
        )
        g.set_trial_subgraph(ts)

        clf = FailureClassification(
            trial_id="NCT_SPARSE",
            primary_failure_mode=FailureMode.EFFICACY_IN_SUBGROUP_ONLY,
            confidence=0.7, evidence_quotes=["test"],
        )
        clf._raw = {"edges_to_update": [
            {"edge_type": "binds_to", "source_entity": "Nivolumab",
             "target_entity": "PD-1", "support": "moderate_support"},
        ]}  # type: ignore[attr-defined]

        updates = Attributor(g).attribute(clf, ts)
        assert updates == []
        records = [
            json.loads(line)
            for line in _attributor_module._UNROUTED_LOG_PATH.read_text().splitlines()
        ]
        assert records
        assert all(r["reason"] == "no_chain_match" for r in records)

    def test_composed_of_updates_skipped(self):
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "composed_of", "source_entity": "ipi+nivo",
             "target_entity": "Nivolumab", "support": "moderate_support"},
        ])
        # composed_of is a structural edge—classifier-driven updates on it
        # are dropped silently (not applied, not logged as unrouted).
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []


# ── Failure-trial backstop ──────────────────────────────────────────────


class TestFailureBackstop:
    """When a classifier returns zero `biology_drives` on a failure trial,
    the attributor auto-emits a default weak_contradict on the parent
    chain so the failure isn't completely silent. The classifier prompt
    rule says this is mandatory, but the LLM ignores it at low confidence;
    this is the defensive code-side enforcement."""

    def _seed_with_biology_drives(self):
        g, ts = _seed_combo_trial_graph()
        # Production graphs always have this edge (populate.py builds it).
        # The combo-trial fixture omits it, so add it here for backstop tests.
        g.add_edge(GraphEdge(
            source_id="R-HSA-389948", target_id="melanoma",
            edge_type=EdgeType.BIOLOGY_DRIVES,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        ))
        return g, ts

    def test_zero_edges_on_failure_emits_backstop_biology_drives(self):
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([])  # zero edges
        clf._raw["trial_outcome"] = "failure"
        updates = Attributor(g).attribute(clf, ts)
        assert len(updates) == 1
        u = updates[0]
        assert u.edge_type == EdgeType.BIOLOGY_DRIVES
        # Should target the parent chain's biology→indication edge.
        parent_chain = ts.chains[0]
        assert u.source_id == parent_chain.biology_id
        assert u.target_id == parent_chain.indication_id
        assert u.evidence.support == SupportBucket.WEAK_CONTRADICT.value
        assert "backstop" in u.evidence.notes.lower()

    def test_backstop_not_emitted_when_classifier_already_emitted_biology_drives(self):
        """If the classifier did its job, the backstop must NOT fire — the
        graph would otherwise get two contradict updates on the same edge
        from one trial."""
        g, ts = self._seed_with_biology_drives()
        parent = ts.chains[0]
        clf = _make_classification([{
            "edge_type": "biology_drives",
            "source_entity": parent.biology_id,
            "target_entity": parent.indication_id,
            "support": "moderate_contradict",
        }])
        clf._raw["trial_outcome"] = "failure"
        updates = Attributor(g).attribute(clf, ts)
        # Exactly one biology_drives update — the classifier's emission,
        # not the backstop. Bucket is moderate (not the backstop's weak).
        biology_drives_updates = [
            u for u in updates if u.edge_type == EdgeType.BIOLOGY_DRIVES
        ]
        assert len(biology_drives_updates) == 1
        assert biology_drives_updates[0].evidence.support == SupportBucket.MODERATE_CONTRADICT.value

    def test_backstop_not_emitted_on_success(self):
        """A success trial returning zero edges is suspect but not the
        backstop's problem — only failure trials get the rule."""
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([])
        clf._raw["trial_outcome"] = "success"
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []

    def test_backstop_not_emitted_on_partial(self):
        """Partial-outcome trials don't get the backstop either — the
        failure signal is too ambiguous to justify a structural default."""
        g, ts = self._seed_with_biology_drives()
        clf = _make_classification([])
        clf._raw["trial_outcome"] = "partial"
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []


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
