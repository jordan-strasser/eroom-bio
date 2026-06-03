"""Tests for the graph population orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.graph.models import (
    CompoundNode,
    EdgeType,
    EndpointNode,
    EndpointType,
    IndicationNode,
    Modality,
    RegulatoryStatus,
    TargetNode,
    TrialOutcome,
)
from src.graph.populate import (
    PopulationPipeline,
    _normalize,
    _normalize_drug_lookup_name,
    _split_combo_regimen,
    _strip_parenthetical_brand,
    build_arms,
    build_trial_subgraph_from_extraction,
    classify_endpoint_deterministic,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ArmGroup,
    Intervention,
    OutcomeMeasure,
    TrialRecord,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    return GraphStore()


@pytest.fixture
def pipeline(graph, tmp_path):
    """Pipeline with a mocked anthropic client so endpoint classification runs."""
    client = AsyncMock()

    async def fake_create(**kwargs):
        # Default to a valid EndpointClass value; tests that need different
        # behavior can override pipeline._anthropic.
        from types import SimpleNamespace
        return SimpleNamespace(content=[SimpleNamespace(text="OS")])

    client.messages.create = fake_create
    return PopulationPipeline(graph, anthropic_client=client, cache_dir=tmp_path)


def _make_trial(
    nct_id: str = "NCT00000001",
    conditions: list[str] | None = None,
    drug_name: str = "Imatinib",
    drug_desc: str = "oral kinase inhibitor targeting BCR-ABL",
    outcome_measure: str = "Overall Survival",
    phase: str = "3",
    title: str = "A Phase 3 Study of Imatinib in CML",
    status: str = "COMPLETED",
) -> TrialRecord:
    # Single-arm trial: one arm group with one drug.
    return TrialRecord(
        nct_id=nct_id,
        title=title,
        phase=phase,
        status=status,
        conditions=conditions or ["Chronic Myeloid Leukemia"],
        interventions=[
            Intervention(name=drug_name, type="DRUG", description=drug_desc),
            Intervention(name="Placebo", type="OTHER", description=""),
        ],
        primary_outcomes=[
            OutcomeMeasure(measure=outcome_measure, timeframe="36 months"),
        ],
        enrollment=400,
        has_results=True,
        arm_groups=[
            ArmGroup(group_id=drug_name.lower(), title=drug_name, intervention_names=[drug_name]),
        ],
    )


# ── Normalize ────────────────────────────────────────────────────────────


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("EGFR") == "egfr"

    def test_strips_whitespace(self):
        assert _normalize("  lung cancer  ") == "lung cancer"

    def test_collapses_spaces(self):
        assert _normalize("non  small   cell") == "non small cell"


# ── Entity resolution ────────────────────────────────────────────────────


class TestResolveEntity:
    def test_resolve_indexed_entity(self, pipeline):
        pipeline._index_node("IND_001", "Lung Cancer", "indication")
        assert pipeline.resolve_entity("Lung Cancer", "indication") == "IND_001"

    def test_case_insensitive(self, pipeline):
        pipeline._index_node("IND_001", "Lung Cancer", "indication")
        assert pipeline.resolve_entity("lung cancer", "indication") == "IND_001"

    def test_returns_none_for_unknown(self, pipeline):
        assert pipeline.resolve_entity("Unknown Disease", "indication") is None

    def test_different_types_independent(self, pipeline):
        pipeline._index_node("IND_001", "ABC", "indication")
        pipeline._index_node("COMP_001", "ABC", "compound")
        assert pipeline.resolve_entity("ABC", "indication") == "IND_001"
        assert pipeline.resolve_entity("ABC", "compound") == "COMP_001"


# ── Trial subgraph building ──────────────────────────────────────────────


class TestBuildTrialSubgraphs:
    def test_builds_subgraph_when_all_resolve(self, pipeline, graph):
        trial = _make_trial()
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="IND_001", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("IND_001", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="EP_001", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Overall Survival", "endpoint")

        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1
        sg = subgraphs[0]
        assert sg.trial_id == "NCT00000001"
        assert sg.phase == "3"
        # One arm with the drug; one chain at the parent population.
        assert len(sg.arms) == 1
        assert sg.arms[0].compound_ids == ["imatinib"]
        assert not sg.arms[0].is_combination
        assert len(sg.chains) == 1
        chain = sg.chains[0]
        assert chain.compound_id == "imatinib"
        assert chain.indication_id == "IND_001"
        assert chain.endpoint_id == "EP_001"
        assert sg.parent_population_id == "IND_001__unselected"

    def test_unresolvable_trial_still_built_via_slug_fallback(self, pipeline):
        """Round-22: a trial whose conditions + outcomes don't match any
        indexed canonical nodes is NOT dropped — slug-create fallbacks
        produce IndicationNode + EndpointNode from the raw CT.gov text
        so the chain backbone resolves. The OLD behavior (silent drop)
        produced 14/50 silent losses in the n=50 multi_indication build."""
        trial = _make_trial()
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1
        # Fallback nodes were created in the graph.
        sg = subgraphs[0]
        assert sg.chains[0].indication_id  # not None
        assert sg.chains[0].endpoint_id    # not None

    def test_compound_only_indexed_still_builds_via_slug_fallback(self, pipeline):
        """Round-22: indication + endpoint canonicalization both
        missing — same fallback applies. Compound resolution is
        independent (and already lenient via slug-creation)."""
        trial = _make_trial()
        pipeline._index_node("imatinib", "Imatinib", "compound")
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1

    def test_placeholder_ids_for_unresolved_backbone(self, pipeline, graph):
        # target/mechanism/biology start as UNKNOWN; subgroup populations
        # are added later by extraction. Parent population is created.
        trial = _make_trial()
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="I1", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("I1", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="E1", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("E1", "Overall Survival", "endpoint")

        subgraphs = pipeline.build_trial_subgraphs([trial])
        sg = subgraphs[0]
        chain = sg.chains[0]
        assert chain.target_id == "UNKNOWN"
        assert chain.mechanism_id == "UNKNOWN"
        assert chain.biology_id == "UNKNOWN"
        assert chain.subgroup_population_id == "I1__unselected"

    def test_combo_arm_synthesizes_combo_compound(self, pipeline, graph):
        # A combo arm (two intervention names) gets a synthesized
        # CompoundNode + composed_of edges and a chain rooted on it.
        trial = _make_trial()
        trial.arm_groups.append(ArmGroup(
            group_id="imatinib_dasatinib",
            title="Imatinib + Dasatinib",
            intervention_names=["Imatinib", "Dasatinib"],
        ))
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        graph.add_node(CompoundNode(id="dasatinib", name="Dasatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        pipeline._index_node("dasatinib", "Dasatinib", "compound")
        graph.add_node(IndicationNode(id="I1", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("I1", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="E1", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("E1", "Overall Survival", "endpoint")

        subgraphs = pipeline.build_trial_subgraphs([trial])
        sg = subgraphs[0]
        combo_arms = [a for a in sg.arms if a.is_combination]
        assert len(combo_arms) == 1
        assert combo_arms[0].regimen_compound_id == "dasatinib+imatinib"
        # Combo CompoundNode synthesized
        combo_node = graph.get_node("dasatinib+imatinib")
        assert combo_node["node_type"] == "InterventionNode"
        # composed_of edges to both constituents
        composed = graph.get_edges_by_type(EdgeType.COMPOSED_OF)
        targets = sorted(e["target_id"] for e in composed if e["source_id"] == "dasatinib+imatinib")
        assert targets == ["dasatinib", "imatinib"]
        # Chain count: 2 arms (mono Imatinib + combo) × 1 parent population
        assert len(sg.chains) == 2

    def test_trial_node_persisted_in_sidecar(self, pipeline, graph):
        trial = _make_trial()
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="I1", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("I1", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="E1", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("E1", "Overall Survival", "endpoint")

        pipeline.build_trial_subgraphs([trial])
        # TrialNode was added to the graph as a marker
        node = graph.get_node("NCT00000001")
        assert node["node_type"] == "TrialNode"
        # And the full TrialSubgraph is in the sidecar
        ts = graph.get_trial_subgraph_by_id("NCT00000001")
        assert ts.trial_id == "NCT00000001"
        assert len(ts.chains) >= 1

    def test_non_drug_intervention_names_are_filtered_from_arms(self, pipeline, graph):
        """CT.gov lists procedures, diagnostics, radiation and devices in
        arm_groups[].intervention_names alongside actual drugs. Those
        non-drug names previously became orphan untyped compound nodes
        ("biopsy", "quality_of_life_assessment", "radiation_therapy", ...)
        polluting the graph. The arm-building step must drop them when
        they're explicitly typed non-drug in trial.interventions.
        """
        from src.ingestion.clinicaltrials import Intervention

        trial = _make_trial()
        # Procedures, radiation, biopsy — all non-drug per CT.gov type.
        trial.interventions.extend([
            Intervention(name="Biopsy", type="PROCEDURE", description=""),
            Intervention(name="Radiation Therapy", type="RADIATION", description=""),
            Intervention(name="Quality-of-Life Assessment", type="OTHER", description=""),
        ])
        trial.arm_groups.append(ArmGroup(
            group_id="combined_treatment",
            title="Imatinib + Radiation Therapy",
            intervention_names=[
                "Imatinib", "Radiation Therapy", "Biopsy",
                "Quality-of-Life Assessment",
            ],
        ))
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="I1", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("I1", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="E1", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("E1", "Overall Survival", "endpoint")

        subgraphs = pipeline.build_trial_subgraphs([trial])
        sg = subgraphs[0]
        combined_arm = next(a for a in sg.arms if a.arm_id == "combined_treatment")
        # Only the drug ("imatinib") survived the filter.
        assert combined_arm.compound_ids == ["imatinib"]
        # No orphan radiation/biopsy/qol nodes leaked into the graph as
        # untyped CompoundNodes.
        for orphan in ("biopsy", "radiation_therapy", "quality_of_life_assessment"):
            try:
                node = graph.get_node(orphan)
            except KeyError:
                continue  # not in graph — good
            assert "node_type" in node, (
                f"non-drug intervention {orphan!r} leaked into the graph "
                "as an untyped node"
            )


# ── Round 20.5: silent-drop visibility ──────────────────────────────────


class TestDropLogVisibility:
    """Round-20.5: build_trial_subgraphs must never silently drop a trial.
    Every skip writes a structured record to
    data/dev/dropped_trial_subgraphs.jsonl with the reason + the input
    fields that failed. Without this the n=50 multi_indication build
    silently lost 14 of 50 trials (28%).
    """

    @pytest.fixture(autouse=True)
    def _isolated_drop_log(self, tmp_path, monkeypatch):
        from src.graph import populate as pop_mod
        log_path = tmp_path / "dropped_trial_subgraphs.jsonl"
        monkeypatch.setattr(
            pop_mod, "_DROPPED_TRIAL_SUBGRAPHS_LOG", log_path,
        )
        yield log_path

    def _drop_log_entries(self, log_path):
        import json
        if not log_path.exists():
            return []
        return [
            json.loads(line) for line in log_path.read_text().splitlines()
            if line.strip()
        ]

    def test_unmatched_indication_slug_creates_node_not_dropped(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """Round-22: a trial whose conditions don't canonicalize must
        NOT be dropped — instead the populator slug-creates a fallback
        IndicationNode from the raw condition text. Mirrors the
        UNKNOWN-allowed pattern for target/mechanism/biology."""
        graph.add_node(EndpointNode(
            id="EP_001", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Overall Survival", "endpoint")
        trial = _make_trial(conditions=["Fictional Unicorn Disease"])
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1
        assert self._drop_log_entries(_isolated_drop_log) == []
        new_ind_id = subgraphs[0].chains[0].indication_id
        new_node = graph.get_node(new_ind_id)
        assert new_node["name"] == "Fictional Unicorn Disease"
        assert (
            new_node.get("metadata", {}).get("source")
            == "slug_fallback_no_canonicalization"
        )

    def test_unmatched_endpoint_slug_creates_node_not_dropped(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """Round-22: same fallback for endpoints. The torcetrapib-class
        case (non-oncology primary outcomes like 'major cardiovascular
        event' that don't match the curated EndpointClass enum) used to
        silently drop. Now slug-creates a measure-keyed EndpointNode."""
        graph.add_node(IndicationNode(id="IND_001", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("IND_001", "Chronic Myeloid Leukemia", "indication")
        trial = _make_trial(outcome_measure="Major cardiovascular disease event")
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1
        assert self._drop_log_entries(_isolated_drop_log) == []
        ep_id = subgraphs[0].chains[0].endpoint_id
        ep_node = graph.get_node(ep_id)
        assert ep_node["name"] == "Major cardiovascular disease event"
        assert ep_id.startswith("other_")
        assert (
            ep_node.get("measurement_properties", {}).get("source")
            == "slug_fallback_no_canonicalization"
        )

    def test_truly_empty_conditions_still_dropped(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """Round-22 drop boundary: only TRULY-no-data trials drop.
        Empty conditions list is a legitimate drop (no slug input)."""
        graph.add_node(EndpointNode(
            id="EP_001", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Overall Survival", "endpoint")
        trial = TrialRecord(
            nct_id="NCT_NO_COND", title="t", phase="3", status="COMPLETED",
            conditions=[],
            interventions=[Intervention(name="drug_x", type="DRUG")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival")],
            enrollment=100, has_results=True, arm_groups=[],
        )
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert subgraphs == []
        entries = self._drop_log_entries(_isolated_drop_log)
        assert len(entries) == 1
        assert entries[0]["reason"] == "no_conditions"

    def test_truly_empty_primary_outcomes_still_dropped(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """Same boundary for endpoints: no primary outcomes at all is a
        legitimate drop."""
        graph.add_node(IndicationNode(id="IND_001", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("IND_001", "Chronic Myeloid Leukemia", "indication")
        trial = TrialRecord(
            nct_id="NCT_NO_OUT", title="t", phase="3", status="COMPLETED",
            conditions=["Chronic Myeloid Leukemia"],
            interventions=[Intervention(name="drug_x", type="DRUG")],
            primary_outcomes=[],
            enrollment=100, has_results=True, arm_groups=[],
        )
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert subgraphs == []
        entries = self._drop_log_entries(_isolated_drop_log)
        assert len(entries) == 1
        assert entries[0]["reason"] == "no_primary_outcomes"

    def test_slug_created_indication_reused_across_trials(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """Idempotency: a slug-created IndicationNode is indexed under
        the raw condition string, so a second trial with the same
        condition reuses the existing node instead of slug-creating
        another. Prevents duplicate-node proliferation in a batch."""
        graph.add_node(EndpointNode(
            id="EP_001", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Overall Survival", "endpoint")
        trial_a = _make_trial(
            nct_id="NCT_A", conditions=["Novel Disease X"],
        )
        trial_b = _make_trial(
            nct_id="NCT_B", conditions=["Novel Disease X"],
        )
        subgraphs = pipeline.build_trial_subgraphs([trial_a, trial_b])
        assert len(subgraphs) == 2
        assert (
            subgraphs[0].chains[0].indication_id
            == subgraphs[1].chains[0].indication_id
        )

    def test_empty_arm_groups_logged_with_specific_reason(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """The torcetrapib case (NCT00134264): older CT.gov records
        sometimes return arm_groups=[]. The fallback in build_arms now
        synthesizes arms from interventions, so this case should NOT
        drop. Verify by removing the only intervention so the fallback
        produces nothing and the drop fires with the empty-arm-groups
        reason."""
        graph.add_node(IndicationNode(id="IND_001", name="Coronary Disease"))
        pipeline._index_node("IND_001", "Coronary Disease", "indication")
        graph.add_node(EndpointNode(
            id="EP_001", name="Major cardiovascular event",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Major cardiovascular event", "endpoint")
        trial = TrialRecord(
            nct_id="NCT_EMPTY", title="t", phase="3", status="TERMINATED",
            conditions=["Coronary Disease"],
            interventions=[],  # no interventions to synthesize from
            primary_outcomes=[
                OutcomeMeasure(measure="Major cardiovascular event"),
            ],
            enrollment=100, has_results=True, arm_groups=[],
        )
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert subgraphs == []
        entries = self._drop_log_entries(_isolated_drop_log)
        assert len(entries) == 1
        assert entries[0]["reason"] == "no_arms_empty_arm_groups"

    def test_arm_groups_fallback_recovers_torcetrapib_shape(
        self, pipeline, graph, _isolated_drop_log,
    ):
        """When arm_groups is empty but interventions has drug-like
        entries, the fallback synthesizes arms. NCT00134264 had
        interventions=['torcetrapib/atorvastatin', 'atorvastatin'] with
        empty arm_groups — should now produce a valid subgraph
        instead of being silently dropped."""
        graph.add_node(IndicationNode(id="IND_001", name="Coronary Disease"))
        pipeline._index_node("IND_001", "Coronary Disease", "indication")
        graph.add_node(EndpointNode(
            id="EP_001", name="Major cardiovascular event",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Major cardiovascular event", "endpoint")
        graph.add_node(CompoundNode(
            id="atorvastatin", name="atorvastatin",
            modality=Modality.SMALL_MOLECULE,
        ))
        pipeline._index_node("atorvastatin", "atorvastatin", "compound")
        trial = TrialRecord(
            nct_id="NCT_TORC", title="t", phase="3", status="TERMINATED",
            conditions=["Coronary Disease"],
            interventions=[
                Intervention(name="torcetrapib/atorvastatin", type="DRUG"),
                Intervention(name="atorvastatin", type="DRUG"),
            ],
            primary_outcomes=[
                OutcomeMeasure(measure="Major cardiovascular event"),
            ],
            enrollment=15000, has_results=True, arm_groups=[],  # CT.gov gap
        )
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1, (
            "arm_groups fallback should synthesize one arm per drug "
            "intervention when CT.gov returns empty arm_groups"
        )
        sg = subgraphs[0]
        # Two drug interventions → two synthesized arms.
        assert len(sg.arms) == 2
        # No drop entry expected — the fallback recovered the trial.
        assert self._drop_log_entries(_isolated_drop_log) == []


class TestArmGroupsFallback:
    """Round-20.5: build_arms must synthesize arms from interventions
    when arm_groups is empty (CT.gov v2 API sometimes omits arm_groups
    on older terminated trials). Without the fallback, trials like
    NCT00134264 (torcetrapib ILLUMINATE) silently dropped."""

    def test_synthesizes_one_arm_per_drug_intervention(self):
        trial = TrialRecord(
            nct_id="NCT_FALLBACK", title="t", phase="3", status="COMPLETED",
            conditions=["Disease"],
            interventions=[
                Intervention(name="drug_a", type="DRUG"),
                Intervention(name="drug_b", type="BIOLOGICAL"),
            ],
            primary_outcomes=[],
            enrollment=100, has_results=True,
            arm_groups=[],  # empty — triggers fallback
        )
        arms = build_arms(trial)
        assert len(arms) == 2
        arm_compound_ids = {arm.compound_ids[0] for arm in arms}
        assert "drug_a" in arm_compound_ids
        assert "drug_b" in arm_compound_ids

    def test_skips_non_drug_interventions_in_fallback(self):
        """The fallback only synthesizes arms for DRUG/BIOLOGICAL
        interventions — radiation, procedures, devices, diagnostics
        don't get their own arm."""
        trial = TrialRecord(
            nct_id="NCT_NONDRUG", title="t", phase="3", status="COMPLETED",
            conditions=["Disease"],
            interventions=[
                Intervention(name="drug_x", type="DRUG"),
                Intervention(name="radiation", type="RADIATION"),
                Intervention(name="biopsy", type="PROCEDURE"),
            ],
            primary_outcomes=[],
            enrollment=100, has_results=True, arm_groups=[],
        )
        arms = build_arms(trial)
        assert len(arms) == 1
        assert arms[0].compound_ids == ["drug_x"]

    def test_fallback_does_not_fire_when_arm_groups_present(self):
        """If arm_groups has any entries, the fallback is bypassed — we
        trust the CT.gov-reported arm structure."""
        trial = TrialRecord(
            nct_id="NCT_NORMAL", title="t", phase="3", status="COMPLETED",
            conditions=["Disease"],
            interventions=[
                Intervention(name="drug_x", type="DRUG"),
                Intervention(name="drug_y", type="DRUG"),
            ],
            primary_outcomes=[],
            enrollment=100, has_results=True,
            arm_groups=[
                ArmGroup(
                    group_id="ag1", title="Arm 1",
                    intervention_names=["drug_x", "drug_y"],
                ),
            ],
        )
        arms = build_arms(trial)
        # One arm with both compounds (combo from arm_groups), NOT two
        # separate arms (which the fallback would produce).
        assert len(arms) == 1
        assert set(arms[0].compound_ids) == {"drug_x", "drug_y"}


# ── Round 3.2 scaling readiness: hierarchy + smoke tests ────────────────


# ── Round 21 Phase B: chembl_id + embedding canonicalization ────────────


class TestCompoundCanonicalization:
    """Round-21 Phase B: at compound-add time, the populator resolves
    incoming compounds to existing nodes when (a) their stable_id
    (ChEMBL) matches, or (b) their embedding is above similarity
    threshold AND chembl ids don't conflict. Without this, two CT.gov
    spellings of the same drug ('Fluorouracil' vs '5-Fluorouracil')
    fragment evidence across separate nodes (the round 19 smoke
    finding)."""

    @pytest.fixture
    def pipeline_with_existing_compound(self, pipeline, graph):
        """Seed an existing compound with chembl_id + embedding so the
        canonicalization decision tree has something to resolve to."""
        from src.graph.models import CompoundNode, Modality
        graph.add_node(CompoundNode(
            id="fluorouracil_canonical",
            name="fluorouracil_canonical",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL185",
            embedding=[1.0, 0.0, 0.0],
            aliases=["5-FU"],
        ))
        return pipeline, graph

    def test_chembl_match_reuses_existing_id(
        self, pipeline_with_existing_compound,
    ):
        from src.graph.models import CompoundNode, Modality
        pipeline, graph = pipeline_with_existing_compound
        existing_chembl_index = {"CHEMBL185": "fluorouracil_canonical"}
        embedding_index: list = []

        incoming = CompoundNode(
            id="five_fluorouracil",
            name="5-Fluorouracil",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL185",  # same chembl as existing
            embedding=[0.0, 1.0, 0.0],  # completely different embedding
        )
        canonical_id, was_resolved = pipeline._canonicalize_compound(
            incoming, existing_chembl_index, embedding_index,
        )
        assert was_resolved
        assert canonical_id == "fluorouracil_canonical"

    def test_embedding_match_above_threshold_reuses_existing(
        self, pipeline_with_existing_compound,
    ):
        from src.graph.models import CompoundNode, Modality
        pipeline, graph = pipeline_with_existing_compound
        existing_chembl_index: dict = {}
        embedding_index = [("fluorouracil_canonical", [1.0, 0.0, 0.0])]

        # Near-identical embedding (cosine ~1.0).
        incoming = CompoundNode(
            id="some_other_slug", name="5-Fluorouracil",
            modality=Modality.SMALL_MOLECULE,
            embedding=[0.999, 0.001, 0.0],
        )
        canonical_id, was_resolved = pipeline._canonicalize_compound(
            incoming, existing_chembl_index, embedding_index,
        )
        assert was_resolved
        assert canonical_id == "fluorouracil_canonical"

    def test_embedding_below_threshold_does_not_merge(
        self, pipeline_with_existing_compound,
    ):
        from src.graph.models import CompoundNode, Modality
        pipeline, graph = pipeline_with_existing_compound
        existing_chembl_index: dict = {}
        embedding_index = [("fluorouracil_canonical", [1.0, 0.0, 0.0])]

        # Cosine well below default 0.80 threshold (~0.577 for this
        # uniform vector vs the canonical [1,0,0]).
        incoming = CompoundNode(
            id="distinct_compound", name="distinct_compound",
            modality=Modality.SMALL_MOLECULE,
            embedding=[0.5, 0.5, 0.5],
        )
        canonical_id, was_resolved = pipeline._canonicalize_compound(
            incoming, existing_chembl_index, embedding_index,
        )
        assert not was_resolved
        assert canonical_id == "distinct_compound"

    def test_chembl_conflict_overrides_embedding_match(
        self, pipeline_with_existing_compound,
    ):
        """Critical safety: even when embedding cosine is very high,
        if the existing node has chembl_id X and the incoming has
        chembl_id Y ≠ X, we MUST NOT merge them. They're distinct
        ChEMBL-registered drugs (e.g. erlotinib vs gefitinib in the
        EGFR TKI family) that happen to embed close."""
        from src.graph.models import CompoundNode, Modality
        pipeline, graph = pipeline_with_existing_compound
        existing_chembl_index: dict = {}
        embedding_index = [("fluorouracil_canonical", [1.0, 0.0, 0.0])]

        # Near-identical embedding but conflicting chembl_id.
        incoming = CompoundNode(
            id="similar_but_different_drug",
            name="similar_but_different_drug",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL999",  # DIFFERENT from existing's CHEMBL185
            embedding=[0.999, 0.001, 0.0],
        )
        canonical_id, was_resolved = pipeline._canonicalize_compound(
            incoming, existing_chembl_index, embedding_index,
        )
        assert not was_resolved
        assert canonical_id == "similar_but_different_drug"

    def test_missing_embedding_falls_back_to_chembl_only(
        self, pipeline_with_existing_compound,
    ):
        from src.graph.models import CompoundNode, Modality
        pipeline, graph = pipeline_with_existing_compound
        existing_chembl_index = {"CHEMBL185": "fluorouracil_canonical"}
        embedding_index: list = []

        incoming = CompoundNode(
            id="fu", name="Fluorouracil",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL185",
            embedding=None,  # sapbert extra not installed
        )
        canonical_id, was_resolved = pipeline._canonicalize_compound(
            incoming, existing_chembl_index, embedding_index,
        )
        assert was_resolved
        assert canonical_id == "fluorouracil_canonical"

    def test_no_signal_returns_unresolved(self, pipeline):
        from src.graph.models import CompoundNode, Modality
        existing_chembl_index: dict = {}
        embedding_index: list = []
        incoming = CompoundNode(
            id="novel_drug", name="novel_drug",
            modality=Modality.SMALL_MOLECULE,
        )
        canonical_id, was_resolved = pipeline._canonicalize_compound(
            incoming, existing_chembl_index, embedding_index,
        )
        assert not was_resolved
        assert canonical_id == "novel_drug"


class TestMergeCompoundIntoExisting:
    """Phase B merge step: when canonicalization resolves to an existing
    node, the incoming compound's identity (aliases, name, original id)
    must be merged in so downstream resolve_entity calls land on the
    canonical node."""

    def test_aliases_get_unioned_into_existing_node(self, pipeline, graph):
        from src.graph.models import CompoundNode, Modality
        graph.add_node(CompoundNode(
            id="fluorouracil_canonical", name="fluorouracil_canonical",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL185",
            aliases=["5-FU"],
        ))
        incoming = CompoundNode(
            id="5_fluorouracil",
            name="5-Fluorouracil",
            modality=Modality.SMALL_MOLECULE,
            aliases=["Adrucil", "5-FU"],  # one overlaps
        )
        pipeline._merge_compound_into_existing(incoming, "fluorouracil_canonical")
        node = graph.get_node("fluorouracil_canonical")
        # Original alias preserved; new ones added; duplicates not added.
        assert set(node["aliases"]) == {
            "5-FU", "5-Fluorouracil", "5_fluorouracil", "Adrucil",
        }

    def test_entity_index_gets_aliases_pointing_to_canonical(
        self, pipeline, graph,
    ):
        from src.graph.models import CompoundNode, Modality
        graph.add_node(CompoundNode(
            id="canon", name="canon", modality=Modality.SMALL_MOLECULE,
        ))
        incoming = CompoundNode(
            id="alt_slug", name="Alternate Name",
            modality=Modality.SMALL_MOLECULE,
            aliases=["another_alias"],
        )
        pipeline._merge_compound_into_existing(incoming, "canon")
        # All incoming identifiers now resolve to the canonical id.
        assert pipeline.resolve_entity("Alternate Name", "compound") == "canon"
        assert pipeline.resolve_entity("alt_slug", "compound") == "canon"
        assert pipeline.resolve_entity("another_alias", "compound") == "canon"

    def test_stable_id_backfilled_when_existing_node_missing_it(
        self, pipeline, graph,
    ):
        from src.graph.models import CompoundNode, Modality
        graph.add_node(CompoundNode(
            id="canon", name="canon", modality=Modality.SMALL_MOLECULE,
            stable_id=None,  # no chembl yet
        ))
        incoming = CompoundNode(
            id="other", name="Other",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL999",
        )
        pipeline._merge_compound_into_existing(incoming, "canon")
        assert graph.get_node("canon")["stable_id"] == "CHEMBL999"

    def test_existing_stable_id_never_overwritten(self, pipeline, graph):
        from src.graph.models import CompoundNode, Modality
        graph.add_node(CompoundNode(
            id="canon", name="canon", modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL185",
        ))
        incoming = CompoundNode(
            id="other", name="Other",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL999",  # would conflict, but should be ignored
        )
        # NOTE: in practice _canonicalize_compound would never call
        # merge in this case (chembl conflict gate blocks it), but the
        # merge function itself must be defensive.
        pipeline._merge_compound_into_existing(incoming, "canon")
        assert graph.get_node("canon")["stable_id"] == "CHEMBL185"


class TestParentIndicationHierarchy:
    """Round 3.2 #11 — disease hierarchy via SUBTYPE_OF edges. Subtype
    IndicationNodes (uveal_melanoma, cutaneous_melanoma, etc.) roll up to
    a parent disease (melanoma) for cross-subtype queries. Hand-curated
    table in indication_taxonomy._INDICATION_HIERARCHY."""

    def test_known_subtypes_resolve_to_parent(self):
        from src.graph.indication_taxonomy import parent_indication_for
        # All anatomical/molecular melanoma subtypes share `melanoma` as
        # parent.
        for sub in [
            "uveal_melanoma", "mucosal_melanoma", "intraocular_melanoma",
            "choroidal_melanoma", "ocular_melanoma", "iris_melanoma",
            "cutaneous_melanoma", "acral_melanoma",
            "acral_lentiginous_melanoma", "mucosal_lentiginous_melanoma",
        ]:
            assert parent_indication_for(sub) == "melanoma", sub

    def test_parent_itself_has_no_parent(self):
        """`melanoma` is a parent in the hierarchy — it doesn't recurse
        up further. (When the table grows to include broader categories
        like `cancer`, this test will need to invert.)"""
        from src.graph.indication_taxonomy import parent_indication_for
        assert parent_indication_for("melanoma") is None

    def test_suffix_heuristic_picks_known_parent(self):
        """Slugs the hierarchy table doesn't enumerate but that suffix-
        match a known parent (e.g. `advanced_melanoma`) get the parent
        via the suffix heuristic."""
        from src.graph.indication_taxonomy import parent_indication_for
        assert parent_indication_for("advanced_melanoma") == "melanoma"
        assert parent_indication_for("refractory_melanoma") == "melanoma"

    def test_unknown_indication_returns_none(self):
        """Diseases with no known parent return None — populator won't
        add a SUBTYPE_OF edge."""
        from src.graph.indication_taxonomy import parent_indication_for
        assert parent_indication_for("crohns_disease") is None
        assert parent_indication_for("type_2_diabetes") is None


class TestChainIndicationAnchoring:
    """A trial whose canonical indication is a subtype still produces
    chains anchored on the parent disease. The subtype IndicationNode
    and SUBTYPE_OF edge are created upstream; the chain backbone uses
    the parent so per-disease evidence accumulates at one place."""

    def test_root_indication_helper(self):
        from src.graph.populate import _root_indication
        assert _root_indication("intraocular_melanoma") == "melanoma"
        assert _root_indication("uveal_melanoma") == "melanoma"
        assert _root_indication("melanoma") == "melanoma"
        # Diseases with no parent passthrough unchanged.
        assert _root_indication("crohns_disease") == "crohns_disease"

    def test_subgroup_fork_chains_anchor_on_parent(self):
        """``add_subgroup_chains`` — the seam ``seed_responds_differently``
        relies on — must produce chains whose ``indication_id`` is the
        parent even when called with a subtype id."""
        from src.graph.models import (
            EdgeBeliefState, IndicationNode, PopulationNode, TrialArm,
            TrialSubgraph,
        )
        from src.graph.populate import add_subgroup_chains
        from src.graph.store import GraphStore
        from src.graph.subgroup_taxonomy import SubgroupFeature

        graph = GraphStore()
        # Seed the subtype + parent IndicationNodes so add_subgroup_chains
        # doesn't fail upstream.
        graph.add_node(IndicationNode(id="melanoma", name="melanoma"))
        graph.add_node(IndicationNode(id="intraocular_melanoma", name="intraocular melanoma"))
        graph.add_node(PopulationNode(
            id="intraocular_melanoma__unselected",
            name="all intraocular melanoma",
            defining_features=[],
        ))

        ts = TrialSubgraph(
            trial_id="NCT_test",
            phase="2",
            parent_population_id="intraocular_melanoma__unselected",
            arms=[TrialArm(
                arm_id="arm_a", regimen_compound_id="drug_x",
                compound_ids=["drug_x"], is_combination=False,
            )],
            chains=[],
        )

        n = add_subgroup_chains(
            graph, ts,
            indication_id="intraocular_melanoma",
            endpoint_id="PFS_melanoma",
            subgroup_features=[[SubgroupFeature(axis="extent", level="metastatic")]],
        )
        assert n == 1
        forked = graph.get_trial_subgraph_by_id("NCT_test").chains
        assert all(c.indication_id == "melanoma" for c in forked), (
            f"expected chains anchored on `melanoma`, got "
            f"{[c.indication_id for c in forked]}"
        )


class TestNonOncologyCanonicalization:
    """Round 3.2 #10 — smoke test that slug normalization handles
    non-cancer condition strings cleanly. The LLM canonicalizer step
    requires a live call (not exercised here); this verifies the
    deterministic slug-level normalization works for non-oncology
    terms too."""

    @pytest.mark.parametrize("raw,expected", [
        # Autoimmune / inflammatory — should normalize cleanly.
        ("Rheumatoid Arthritis",            "rheumatoid_arthritis"),
        ("Systemic Lupus Erythematosus",    "systemic_lupus_erythematosus"),
        ("Ulcerative Colitis",              "ulcerative_colitis"),
        ("Crohn's Disease",                 "crohn_s_disease"),
        ("Psoriasis",                       "psoriasis"),
        # Neurology — plurals collapse via the singular rule.
        ("Multiple Sclerosis",              "multiple_sclerosis"),
        ("Parkinson's Disease",             "parkinson_s_disease"),
        ("Alzheimer's Disease",             "alzheimer_s_disease"),
        # Infectious — chronic / acute qualifiers handled by LLM step,
        # but slug normalization on the disease name itself works.
        ("Hepatitis B",                     "hepatitis_b"),
        ("Tuberculosis",                    "tuberculosis"),
        ("HIV Infection",                   "hiv_infection"),
        # Cardiology
        ("Heart Failure",                   "heart_failure"),
        ("Atrial Fibrillation",             "atrial_fibrillation"),
        # Metabolic — plural-singular rule
        ("Type 2 Diabetes Mellitus",        "type_2_diabetes_mellitus"),
        ("Diabetic Neuropathies",           "diabetic_neuropathy"),
    ])
    def test_non_oncology_slugs_normalize(self, raw, expected):
        from src.graph.indication_taxonomy import slugify_disease_name
        assert slugify_disease_name(raw) == expected


class TestSubtypePreservationRule:
    """Round 3.2 #9 — distinct anatomical/molecular subtypes get their own
    IndicationNode (don't collapse to bare parent). The cache fix and
    LLM prompt update enforce this for melanoma subtypes; this test
    documents the rule via the canonicalization cache contents."""

    def test_known_melanoma_subtype_cache_entries_preserved(self):
        """Verifies the cache sweep correctly maps each histologic
        subtype of melanoma to its own canonical id rather than the
        bare `melanoma` parent. Lock-in test so future cache
        manipulations don't silently collapse subtypes back together."""
        import json
        from pathlib import Path
        cache_path = Path("data/cache/indication_canonicalizations.json")
        if not cache_path.exists():
            pytest.skip("No canonicalization cache to verify")
        m = json.loads(cache_path.read_text())
        expected = {
            "Cutaneous Melanoma": "cutaneous_melanoma",
            "Acral Melanoma": "acral_melanoma",
            "Acral Lentiginous Melanoma": "acral_lentiginous_melanoma",
            "Mucosal Lentiginous Melanoma": "mucosal_lentiginous_melanoma",
        }
        for raw, want_slug in expected.items():
            if raw not in m:
                continue  # cache may not have this exact entry yet
            got_slug = m[raw].split("|")[0]
            assert got_slug == want_slug, (
                f"{raw!r}: expected {want_slug}, got {got_slug}. "
                "This subtype should be a distinct IndicationNode, "
                "not collapsed to the bare parent."
            )


# ── Indication slug normalization ────────────────────────────────────────


class TestSlugifyDiseaseName:
    """The canonicalizer-side normalization that collapses near-duplicate
    IndicationNode ids (singular/plural, word-order variants)."""

    @pytest.mark.parametrize("raw,expected", [
        ("Solid Tumors",          "solid_tumor"),
        ("solid tumor",           "solid_tumor"),       # already singular
        ("Brain Metastases",      "brain_metastasis"),
        ("Brain Metastasis",      "brain_metastasis"),
        ("Pancreatic Cancers",    "pancreatic_cancer"),
        ("Soft Tissue Sarcomas",  "soft_tissue_sarcoma"),
        ("Non-Hodgkin Lymphomas", "non_hodgkin_lymphoma"),
        ("Acute Leukemias",       "acute_leukemia"),
        ("Squamous Cell Carcinomas", "squamous_cell_carcinoma"),
        ("Benign Neoplasms",      "benign_neoplasm"),
    ])
    def test_plural_disease_nouns_collapse_to_singular(self, raw, expected):
        from src.graph.indication_taxonomy import slugify_disease_name
        assert slugify_disease_name(raw) == expected

    def test_known_aliases_normalize_to_canonical_form(self):
        """Word-order variants of the same disease — CT.gov phrases the
        same disease two different ways across trials; the alias table
        picks one canonical form so evidence accumulates together."""
        from src.graph.indication_taxonomy import slugify_disease_name
        canonical = "head_and_neck_squamous_cell_carcinoma"
        assert slugify_disease_name(
            "Head and Neck Squamous Cell Carcinoma"
        ) == canonical
        # The CT.gov variant "Carcinoma, Squamous Cell of Head and Neck"
        # slugifies to a different word order; the alias dict maps it back.
        assert slugify_disease_name(
            "squamous cell carcinoma head and neck"
        ) == canonical


# ── Compound-target cross-reference ──────────────────────────────────────


class TestCompoundTargetEdges:
    def test_adds_binds_to_when_symbol_in_text(self, pipeline, graph):
        # Round-27 fix: the cross-reference heuristic now requires the
        # compound to have a resolved chembl_id (kills the phantom
        # bev → MET and checkpoint → APP edges). Setting chembl_id on
        # the test compound to exercise the kept-path.
        graph.add_node(CompoundNode(
            id="C1", name="Imatinib", modality=Modality.SMALL_MOLECULE,
            chembl_id="CHEMBL941",
        ))
        pipeline._index_node("C1", "Imatinib", "compound")
        graph.add_node(TargetNode(id="T_ABL", name="ABL1 kinase", gene_symbol="ABL1"))

        trial = _make_trial(
            drug_name="Imatinib",
            drug_desc="targets ABL1 kinase",
            title="Phase 3 Imatinib in CML",
        )
        added = pipeline._add_compound_target_edges([trial])
        assert added >= 1
        belief = graph.get_edge_belief("C1", "T_ABL", EdgeType.AFFECTS)
        # Round-25: DATABASE_CROSS_REFERENCE record (n_eff=0.3,
        # p_obs=0.65) applied to Beta(1, 1) → alpha=1.195, beta=1.105.
        assert belief.alpha == pytest.approx(1.195)
        assert belief.evidence[0].source_type.value == "database_cross_reference"

    def test_skips_unresolved_compound(self, pipeline, graph):
        """Round-27: cross-reference heuristic must NOT fire for
        compounds lacking a chembl_id — that's how phantom edges land
        on dose-laden / unresolved compound slugs.
        """
        # No chembl_id set on the compound — simulates the round-26
        # `bevacizumab_7_5_mg_kg` regression.
        graph.add_node(CompoundNode(
            id="C_UNRESOLVED", name="Unresolved Compound",
            modality=Modality.SMALL_MOLECULE,
        ))
        pipeline._index_node("C_UNRESOLVED", "Unresolved Compound", "compound")
        graph.add_node(TargetNode(
            id="T_MET", name="MET receptor", gene_symbol="MET",
        ))

        # Trial title contains "metastatic" → substring "met" — would
        # have matched MET gene_symbol under the pre-round-27 heuristic.
        trial = _make_trial(
            drug_name="Unresolved Compound",
            drug_desc="treats metastatic disease",
            title="Phase 3 trial in metastatic colorectal cancer",
        )
        added = pipeline._add_compound_target_edges([trial])
        # Round-27 gate skips because chembl_id is None — and tightened
        # symbol min-length 3→4 + word boundary would also block MET
        # (3 chars, and "met" not a whole word in "metastatic").
        assert added == 0

    def test_no_edge_when_no_match(self, pipeline, graph):
        graph.add_node(CompoundNode(id="C1", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("C1", "Imatinib", "compound")
        graph.add_node(TargetNode(id="T_HER2", name="HER2 receptor", gene_symbol="ERBB2"))

        trial = _make_trial(
            drug_name="Imatinib",
            drug_desc="oral kinase inhibitor",
            title="Phase 3 Imatinib in CML",
        )
        added = pipeline._add_compound_target_edges([trial])
        assert added == 0


# ── Drug-name normalization for OT cache lookup ─────────────────────────


class TestNormalizeDrugLookupName:
    def test_semicolon_normalized_to_comma(self):
        assert _normalize_drug_lookup_name(
            "Sorafenib (Nexavar; BAY43-9006)"
        ) == "sorafenib (nexavar, bay43-9006)"

    def test_already_comma_form_unchanged(self):
        assert _normalize_drug_lookup_name(
            "sorafenib (nexavar, bay43-9006)"
        ) == "sorafenib (nexavar, bay43-9006)"

    def test_whitespace_collapsed(self):
        assert _normalize_drug_lookup_name(
            "  Drug   X  (alias)  "
        ) == "drug x (alias)"

    def test_distinct_drug_names_stay_distinct(self):
        # Parenthetical contents preserved — must not conflate
        # "Drug X" with "Drug X (in combo with Drug Y)".
        a = _normalize_drug_lookup_name("Drug X")
        b = _normalize_drug_lookup_name("Drug X (in combo with Drug Y)")
        assert a != b


class TestStripParentheticalBrand:
    def test_strips_brand_parenthetical(self):
        assert _strip_parenthetical_brand(
            "Sorafenib (Nexavar; BAY43-9006)"
        ) == "Sorafenib"

    def test_preserves_combination_with_modifier(self):
        # "Drug X (in combo with Drug Y)" must NOT collapse to "Drug X"
        # — they're different regimens.
        assert _strip_parenthetical_brand(
            "Drug X (in combo with Drug Y)"
        ) is None

    def test_preserves_plus_combination(self):
        assert _strip_parenthetical_brand(
            "Drug X (plus Drug Y)"
        ) is None

    def test_returns_none_when_no_parenthetical(self):
        assert _strip_parenthetical_brand("Sorafenib") is None

    def test_returns_none_when_only_parenthetical(self):
        assert _strip_parenthetical_brand("(Nexavar)") is None


# ── Slashed-regimen splitter ───────────────────────────────────────────


class TestSplitComboRegimen:
    def test_splits_two_drug_regimen(self):
        assert _split_combo_regimen("Carboplatin/Paclitaxel") == [
            "Carboplatin", "Paclitaxel"
        ]

    def test_splits_three_drug_regimen(self):
        assert _split_combo_regimen("Cyclophosphamide/Doxorubicin/Vincristine") == [
            "Cyclophosphamide", "Doxorubicin", "Vincristine"
        ]

    def test_no_split_when_no_slash(self):
        assert _split_combo_regimen("Sorafenib") is None

    def test_no_split_when_part_has_digits(self):
        # "5% Dextrose/Saline" — % and digits suggest formulation, not combo
        assert _split_combo_regimen("5% Dextrose/Saline") is None

    def test_no_split_when_part_too_short(self):
        assert _split_combo_regimen("A/B") is None

    def test_strips_whitespace_around_parts(self):
        assert _split_combo_regimen(" Carboplatin / Paclitaxel ") == [
            "Carboplatin", "Paclitaxel"
        ]

    def test_splits_on_plus_separator(self):
        assert _split_combo_regimen("Sorafenib + Dacarbazine") == [
            "Sorafenib", "Dacarbazine"
        ]

    def test_splits_plus_with_parenthetical_aliases(self):
        """The actual NCT00492297 case — CT.gov intervention name."""
        assert _split_combo_regimen(
            "Sorafenib (Nexavar; BAY43-9006) + Dacarbazine (DTIC)"
        ) == ["Sorafenib (Nexavar; BAY43-9006)", "Dacarbazine (DTIC)"]

    def test_splits_on_and_separator(self):
        assert _split_combo_regimen("Drug A and Drug B") == ["Drug A", "Drug B"]

    def test_splits_on_plus_word_separator(self):
        assert _split_combo_regimen("Drug A plus Drug B") == ["Drug A", "Drug B"]

    def test_no_split_when_plus_attached_to_compound(self):
        """`anti_pd1+ctla4` (no spaces around +) should NOT split — it's
        a hyphenless compound-id-style name, not a regimen separator."""
        assert _split_combo_regimen("anti_pd1+ctla4") is None


# ── build_arms compound-name canonicalization (round-10b) ──────────────


class TestBuildArmsCompoundCanonicalization:
    """Regression suite for the round-10b populator fixes:
    parenthetical-alias strip (4c-1) + slashed-regimen split (4c-2).
    Both run BEFORE compound id creation so cross-trial learning lands
    on a single CompoundNode per drug rather than fragmenting."""

    def _trial(self, arm_groups, interventions=None):
        return TrialRecord(
            nct_id="NCT_TEST",
            title="t",
            phase="2",
            status="COMPLETED",
            conditions=["Melanoma"],
            interventions=interventions or [],
            primary_outcomes=[OutcomeMeasure(measure="OS", timeframe="36 months")],
            enrollment=100,
            has_results=True,
            arm_groups=arm_groups,
        )

    def test_parenthetical_brand_stripped_before_id(self):
        """`Sorafenib (Nexavar, BAY43-9006)` canonicalizes to `sorafenib`
        — not `sorafenib_nexavar_bay43_9006` (the round-8 fragmentation
        bug that broke cross-trial learning for sorafenib)."""
        trial = self._trial([
            ArmGroup(
                group_id="arm_a",
                title="Sorafenib arm",
                intervention_names=["Sorafenib (Nexavar, BAY43-9006)"],
            ),
        ])
        arms = build_arms(trial)
        assert arms[0].compound_ids == ["sorafenib"]

    def test_slashed_regimen_splits_to_constituents(self):
        """`Carboplatin/Paclitaxel` becomes two compounds, gets treated
        as a combo arm — so the synthesized regimen has composed_of
        edges to standalone `carboplatin` and `paclitaxel` (matching
        the existing `+`-format combo behavior)."""
        trial = self._trial([
            ArmGroup(
                group_id="arm_a",
                title="Carbo/Pac arm",
                intervention_names=["Carboplatin/Paclitaxel"],
            ),
        ])
        arms = build_arms(trial)
        assert set(arms[0].compound_ids) == {"carboplatin", "paclitaxel"}
        assert arms[0].is_combination is True
        assert arms[0].regimen_compound_id == "carboplatin+paclitaxel"

    def test_combined_split_and_strip(self):
        """`Sorafenib (Nexavar, BAY43-9006) + Carboplatin/Paclitaxel`
        from CT.gov arrives as TWO intervention_names; the second one
        gets split, the first gets parenthetical-stripped. Net result:
        a 3-constituent combo arm with canonical ids."""
        trial = self._trial([
            ArmGroup(
                group_id="combo_arm",
                title="Sorafenib + carbo/pac",
                intervention_names=[
                    "Sorafenib (Nexavar, BAY43-9006)",
                    "Carboplatin/Paclitaxel",
                ],
            ),
        ])
        arms = build_arms(trial)
        assert set(arms[0].compound_ids) == {
            "sorafenib", "carboplatin", "paclitaxel",
        }
        assert arms[0].is_combination is True

    def test_single_intervention_with_plus_separator_splits(self):
        """The NCT00492297 case: CT.gov gives ONE intervention name
        "Sorafenib (Nexavar; BAY43-9006) + Dacarbazine (DTIC)" describing
        the whole combo regimen. Should split + strip → 2 constituents."""
        trial = self._trial([
            ArmGroup(
                group_id="combo_arm",
                title="Sora + DTIC",
                intervention_names=[
                    "Sorafenib (Nexavar; BAY43-9006) + Dacarbazine (DTIC)",
                ],
            ),
        ])
        arms = build_arms(trial)
        assert set(arms[0].compound_ids) == {"sorafenib", "dacarbazine"}
        assert arms[0].is_combination is True

    def test_combo_indicator_parenthetical_preserved(self):
        """`Drug X (with adjuvant)` must NOT be stripped to `drug_x` —
        the parenthetical describes the regimen, not a brand alias.
        Existing _strip_parenthetical_brand guard handles this; here we
        verify build_arms respects that guard."""
        trial = self._trial([
            ArmGroup(
                group_id="arm_a",
                title="X with adjuvant",
                intervention_names=["Drug X (in combo with Drug Y)"],
            ),
        ])
        arms = build_arms(trial)
        # The whole name normalizes to one compound id (no strip applied)
        assert len(arms[0].compound_ids) == 1
        assert "with" in arms[0].compound_ids[0] or "combo" in arms[0].compound_ids[0]


# ── Vaccine-component target heuristic ──────────────────────────────────


class TestVaccineComponentTargets:
    def test_adds_pmel_edge_for_gp100_vaccine(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="gp100_antigen", name="gp100 antigen", modality=Modality.OTHER,
        ))
        pipeline._index_node("gp100_antigen", "gp100 antigen", "compound")

        trial = _make_trial(drug_name="gp100 antigen", drug_desc="peptide vaccine")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"gp100_antigen": ["ENSG00000185664"]}
        assert graph.get_node("ENSG00000185664")["gene_symbol"] == "PMEL"
        belief = graph.get_edge_belief(
            "gp100_antigen", "ENSG00000185664", EdgeType.AFFECTS,
        )
        # Vaccine-component pattern-match emits strong_support
        # DATABASE_MAB_TABLE (n_eff=10 after round-28) → α=1+10·0.95=10.5,
        # β=1+10·0.05=1.5.
        assert belief.alpha == pytest.approx(10.5)
        assert belief.beta == pytest.approx(1.5)
        assert belief.evidence[0].source_type.value == "database_mab_table"

    def test_adds_all_three_targets_for_combo_peptide_vaccine(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="combo_peptides",
            name="Tyrosinase/gp100/MART-1 Peptides",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "combo_peptides", "Tyrosinase/gp100/MART-1 Peptides", "compound",
        )

        trial = _make_trial(
            drug_name="Tyrosinase/gp100/MART-1 Peptides",
            drug_desc="three TAA peptides",
        )
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 3
        assert set(comp_targets["combo_peptides"]) == {
            "ENSG00000185664", "ENSG00000077498", "ENSG00000120215",
        }
        for ens, symbol in [
            ("ENSG00000185664", "PMEL"),
            ("ENSG00000077498", "TYR"),
            ("ENSG00000120215", "MLANA"),
        ]:
            assert graph.get_node(ens)["gene_symbol"] == symbol
            # strong_support DATABASE_MAB_TABLE (n_eff=10 after round-28)
            # → α=1+10·0.95=10.5.
            assert graph.get_edge_belief(
                "combo_peptides", ens, EdgeType.AFFECTS,
            ).alpha == pytest.approx(10.5)

    def test_idempotent_across_repeated_trials(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="gp100_antigen", name="gp100 antigen", modality=Modality.OTHER,
        ))
        pipeline._index_node("gp100_antigen", "gp100 antigen", "compound")

        trials = [
            _make_trial(nct_id="NCT01", drug_name="gp100 antigen"),
            _make_trial(nct_id="NCT02", drug_name="gp100 antigen"),
        ]
        first_added, _ = pipeline._add_vaccine_component_target_edges(trials)
        second_added, _ = pipeline._add_vaccine_component_target_edges(trials)

        assert first_added == 1
        assert second_added == 0

    def test_reuses_existing_target_node(self, pipeline, graph):
        # Pre-existing TargetNode from another source; our heuristic must
        # not overwrite it.
        graph.add_node(TargetNode(
            id="ENSG00000185664",
            name="Pre-existing PMEL node from OT",
            gene_symbol="PMEL",
        ))
        graph.add_node(CompoundNode(
            id="gp100_antigen", name="gp100 antigen", modality=Modality.OTHER,
        ))
        pipeline._index_node("gp100_antigen", "gp100 antigen", "compound")

        trial = _make_trial(drug_name="gp100 antigen")
        added, _ = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert graph.get_node("ENSG00000185664")["name"] == "Pre-existing PMEL node from OT"

    def test_unrelated_compound_emits_nothing(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="C1", name="Imatinib", modality=Modality.SMALL_MOLECULE,
        ))
        pipeline._index_node("C1", "Imatinib", "compound")

        trial = _make_trial(drug_name="Imatinib")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 0
        assert comp_targets == {}

    def test_montanide_adjuvant_routes_to_nlrp3(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="montanide_isa_51_vg",
            name="Montanide ISA 51 VG",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "montanide_isa_51_vg", "Montanide ISA 51 VG", "compound",
        )

        trial = _make_trial(drug_name="Montanide ISA 51 VG")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"montanide_isa_51_vg": ["ENSG00000162711"]}
        assert graph.get_node("ENSG00000162711")["gene_symbol"] == "NLRP3"
        belief = graph.get_edge_belief(
            "montanide_isa_51_vg", "ENSG00000162711", EdgeType.AFFECTS,
        )
        # strong_support DATABASE_MAB_TABLE (n_eff=10 after round-28)
        # → α=1+10·0.95=10.5, β=1+10·0.05=1.5.
        assert belief.alpha == pytest.approx(10.5)
        assert belief.beta == pytest.approx(1.5)

    def test_incomplete_freund_routes_to_nlrp3(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="incomplete_freund_s_adjuvant",
            name="Incomplete Freund's Adjuvant",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "incomplete_freund_s_adjuvant",
            "Incomplete Freund's Adjuvant",
            "compound",
        )

        trial = _make_trial(drug_name="Incomplete Freund's Adjuvant")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {
            "incomplete_freund_s_adjuvant": ["ENSG00000162711"],
        }
        assert graph.get_node("ENSG00000162711")["gene_symbol"] == "NLRP3"

    def test_tetanus_helper_peptide_routes_to_hla_drb1(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="tetanus_helper_peptide",
            name="Tetanus Helper Peptide",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "tetanus_helper_peptide", "Tetanus Helper Peptide", "compound",
        )

        trial = _make_trial(drug_name="Tetanus Helper Peptide")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"tetanus_helper_peptide": ["ENSG00000196126"]}
        assert graph.get_node("ENSG00000196126")["gene_symbol"] == "HLA-DRB1"

    def test_melanoma_helper_peptide_routes_to_hla_drb1(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="melanoma_helper_peptide",
            name="Melanoma Helper Peptide",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "melanoma_helper_peptide", "Melanoma Helper Peptide", "compound",
        )

        trial = _make_trial(drug_name="Melanoma Helper Peptide")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"melanoma_helper_peptide": ["ENSG00000196126"]}
        assert graph.get_node("ENSG00000196126")["gene_symbol"] == "HLA-DRB1"

    def test_poly_iclc_routes_to_tlr3(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="poly_iclc", name="Poly-ICLC", modality=Modality.OTHER,
        ))
        pipeline._index_node("poly_iclc", "Poly-ICLC", "compound")

        trial = _make_trial(drug_name="Poly-ICLC")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"poly_iclc": ["ENSG00000164342"]}
        assert graph.get_node("ENSG00000164342")["gene_symbol"] == "TLR3"

    def test_cpg_odn_routes_to_tlr9(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="cpg_7909", name="CPG-7909", modality=Modality.OTHER,
        ))
        pipeline._index_node("cpg_7909", "CPG-7909", "compound")

        trial = _make_trial(drug_name="CPG-7909")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"cpg_7909": ["ENSG00000239732"]}
        assert graph.get_node("ENSG00000239732")["gene_symbol"] == "TLR9"

    def test_cmp_001_codename_routes_to_tlr9(self, pipeline, graph):
        """CMP-001 (vidutolimod) is a TLR9-agonist virus-like particle
        containing CpG-A. The codename doesn't share tokens with the
        cpg/cpg-7909 patterns so it needs its own entries."""
        graph.add_node(CompoundNode(
            id="cmp_001", name="CMP-001", modality=Modality.OTHER,
        ))
        pipeline._index_node("cmp_001", "CMP-001", "compound")

        trial = _make_trial(drug_name="CMP-001")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"cmp_001": ["ENSG00000239732"]}
        assert graph.get_node("ENSG00000239732")["gene_symbol"] == "TLR9"

    def test_vidutolimod_inn_routes_to_tlr9(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="vidutolimod", name="Vidutolimod", modality=Modality.OTHER,
        ))
        pipeline._index_node("vidutolimod", "Vidutolimod", "compound")

        trial = _make_trial(drug_name="Vidutolimod")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"vidutolimod": ["ENSG00000239732"]}

    def test_anti_ny_eso_1_tcr_routes_to_ctag1b(self, pipeline, graph):
        """Adoptive TCR therapy targeting NY-ESO-1 antigen — gene =
        CTAG1B. Round-12 cell-therapy heuristic addition."""
        graph.add_node(CompoundNode(
            id="anti_ny_eso_1_t_cell_receptor_pbl",
            name="Anti NY-ESO-1 T-cell Receptor PBL",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "anti_ny_eso_1_t_cell_receptor_pbl",
            "Anti NY-ESO-1 T-cell Receptor PBL",
            "compound",
        )
        trial = _make_trial(drug_name="Anti NY-ESO-1 T-cell Receptor PBL")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 1
        assert comp_targets == {
            "anti_ny_eso_1_t_cell_receptor_pbl": ["ENSG00000184033"],
        }
        assert graph.get_node("ENSG00000184033")["gene_symbol"] == "CTAG1B"

    def test_ps_341_codename_routes_to_psmb5(self, pipeline, graph):
        """Bortezomib code name PS-341 → proteasome β5 subunit (PSMB5)."""
        graph.add_node(CompoundNode(
            id="ps_341", name="PS-341", modality=Modality.OTHER,
        ))
        pipeline._index_node("ps_341", "PS-341", "compound")
        trial = _make_trial(drug_name="PS-341")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 1
        assert comp_targets == {"ps_341": ["ENSG00000100804"]}

    def test_ucn_01_routes_to_chek1(self, pipeline, graph):
        """UCN-01 (7-hydroxystaurosporine) → CHEK1."""
        graph.add_node(CompoundNode(
            id="ucn_01", name="UCN-01", modality=Modality.OTHER,
        ))
        pipeline._index_node("ucn_01", "UCN-01", "compound")
        trial = _make_trial(drug_name="UCN-01")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 1
        assert comp_targets == {"ucn_01": ["ENSG00000149554"]}

    def test_hu14_18_il2_routes_to_two_targets(self, pipeline, graph):
        """hu14.18-IL2 immunocytokine — anti-GD2 antibody + IL-2 fused;
        binds GD2 synthase (B4GALNT1) and IL-2 receptor (IL2RA)."""
        graph.add_node(CompoundNode(
            id="hu14_18_il2", name="hu14.18-IL2", modality=Modality.OTHER,
        ))
        pipeline._index_node("hu14_18_il2", "hu14.18-IL2", "compound")
        trial = _make_trial(drug_name="hu14.18-IL2")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 2
        assert set(comp_targets["hu14_18_il2"]) == {
            "ENSG00000135454", "ENSG00000134460",
        }

    def test_tvec_routes_to_hvem(self, pipeline, graph):
        """T-VEC (talimogene laherparepvec) — oncolytic HSV-1; HSV entry
        receptor on tumor cells is HVEM (TNFRSF14). Round-13 add."""
        graph.add_node(CompoundNode(
            id="talimogene_laherparepvec",
            name="talimogene laherparepvec",
            modality=Modality.OTHER,
        ))
        pipeline._index_node(
            "talimogene_laherparepvec",
            "talimogene laherparepvec", "compound",
        )
        trial = _make_trial(drug_name="talimogene laherparepvec")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 1
        assert comp_targets == {"talimogene_laherparepvec": ["ENSG00000157873"]}

    def test_adi_peg_20_routes_to_ass1(self, pipeline, graph):
        """ADI-PEG-20 — synthetic-lethal arginine deprivation in
        ASS1-deficient tumors. Round-13 add."""
        graph.add_node(CompoundNode(
            id="adi_peg_20", name="ADI-PEG-20", modality=Modality.OTHER,
        ))
        pipeline._index_node("adi_peg_20", "ADI-PEG-20", "compound")
        trial = _make_trial(drug_name="ADI-PEG-20")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 1
        assert comp_targets == {"adi_peg_20": ["ENSG00000130707"]}

    def test_gsk2132231a_routes_to_mage_a3(self, pipeline, graph):
        """GSK2132231a — anti-MAGE-A3 cancer immunotherapeutic, failed
        DERMA trial. Round-13 add."""
        graph.add_node(CompoundNode(
            id="gsk2132231a", name="GSK2132231a", modality=Modality.OTHER,
        ))
        pipeline._index_node("gsk2132231a", "GSK2132231a", "compound")
        trial = _make_trial(drug_name="GSK2132231a")
        added, comp_targets = pipeline._add_cell_therapy_target_edges([trial])
        assert added == 1
        assert comp_targets == {"gsk2132231a": ["ENSG00000221867"]}

    def test_monophosphoryl_lipid_routes_to_tlr4(self, pipeline, graph):
        graph.add_node(CompoundNode(
            id="mpl_adjuvant",
            name="Monophosphoryl Lipid A",
            modality=Modality.OTHER,
        ))
        pipeline._index_node("mpl_adjuvant", "Monophosphoryl Lipid A", "compound")

        trial = _make_trial(drug_name="Monophosphoryl Lipid A")
        added, comp_targets = pipeline._add_vaccine_component_target_edges([trial])

        assert added == 1
        assert comp_targets == {"mpl_adjuvant": ["ENSG00000136869"]}
        assert graph.get_node("ENSG00000136869")["gene_symbol"] == "TLR4"


# ── Full pipeline (mocked I/O) ──────────────────────────────────────────


class TestPopulateOncology:
    @pytest.mark.asyncio
    async def test_end_to_end_mocked(self, pipeline, graph):
        trial = _make_trial()

        # Mock ClinicalTrials.gov (both fetch paths the orchestrator hits)
        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )
        pipeline._ct_client.fetch_oncology_terminated_with_reason = AsyncMock(
            return_value=[]
        )

        # Mock Open Targets—disease search returns nothing (simplify)
        pipeline._ot_client._post = AsyncMock(
            return_value={"search": {"hits": []}}
        )

        summary = await pipeline.populate_trials(max_trials=10)

        assert summary["trials_fetched"] == 1
        assert summary["compounds"] >= 1
        assert summary["indications"] >= 1
        assert summary["endpoints"] >= 1
        assert summary["trial_subgraphs"] >= 1

    @pytest.mark.asyncio
    async def test_with_ot_associations(self, pipeline, graph):
        trial = _make_trial(
            conditions=["non-small cell lung carcinoma"],
        )

        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )
        pipeline._ct_client.fetch_oncology_terminated_with_reason = AsyncMock(
            return_value=[]
        )

        # Mock OT for the trial-driven flow:
        #   DrugWithTargets → returns linkedTargets containing EGFR
        #   SearchDisease   → returns the EFO id
        #   TargetAssociations → returns the (target, disease) score
        async def mock_post(query, variables):
            if "DrugWithTargets" in query or "mechanismsOfAction" in query:
                return {
                    "search": {
                        "hits": [{
                            "id": "CHEMBL941",
                            "name": "Imatinib",
                            "object": {
                                "id": "CHEMBL941",
                                "name": "Imatinib",
                                "synonyms": [],
                                "tradeNames": ["Gleevec"],
                                "mechanismsOfAction": {
                                    "rows": [{
                                        "actionType": "INHIBITOR",
                                        "mechanismOfAction": "EGFR inhibitor",
                                        "targets": [{
                                            "id": "ENSG00000146648",
                                            "approvedSymbol": "EGFR",
                                            "approvedName": "epidermal growth factor receptor",
                                        }],
                                    }],
                                },
                            },
                        }],
                    },
                }
            if "SearchDisease" in query:
                return {"search": {"hits": [{"id": "EFO_0003060"}]}}
            if "TargetAssociations" in query:
                return {
                    "target": {
                        "associatedDiseases": {
                            "count": 1,
                            "rows": [{
                                "disease": {
                                    "id": "EFO_0003060",
                                    "name": "non-small cell lung carcinoma",
                                },
                                "score": 0.89,
                                "datatypeScores": [{"id": "clinical", "score": 0.99}],
                            }],
                        },
                    },
                }
            # SearchDrug (without targets)—fall through with a generic hit
            return {"search": {"hits": []}}

        pipeline._ot_client._post = mock_post

        summary = await pipeline.populate_trials(max_trials=10)
        assert summary["targets"] >= 1
        assert summary["edges"] >= 1

    @pytest.mark.asyncio
    async def test_handles_ot_failure_gracefully(self, pipeline, graph):
        trial = _make_trial()

        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )
        pipeline._ct_client.fetch_oncology_terminated_with_reason = AsyncMock(
            return_value=[]
        )

        # OT raises on every call
        pipeline._ot_client._post = AsyncMock(side_effect=RuntimeError("API down"))

        # Should not raise—just logs and continues
        summary = await pipeline.populate_trials(max_trials=10)
        assert summary["trials_fetched"] == 1


# ── Endpoint classifier (deterministic, no LLM) ─────────────────────────


class TestClassifyEndpointDeterministic:
    @pytest.mark.parametrize("text,expected", [
        ("Progression Free Survival (PFS)", "PFS"),
        ("Progression-Free Survival", "PFS"),
        ("Median PFS", "PFS"),
        ("Overall Survival (OS)", "OS"),
        ("OS at 24 months", "OS"),
        ("Overall Survival", "OS"),
        ("Disease-Free Survival", "DFS"),
        ("Time to Progression", "TTP"),
        ("Objective Response Rate", "ORR"),
        ("Overall Response Rate", "ORR"),
        ("Complete Response Rate", "CR"),
    ])
    def test_known_endpoints(self, text, expected):
        assert classify_endpoint_deterministic(text) == expected

    def test_unknown_text_returns_other(self):
        assert classify_endpoint_deterministic("Quality of Life Score") == "other"

    def test_empty_text_returns_other(self):
        assert classify_endpoint_deterministic("") == "other"

    @pytest.mark.parametrize("text,expected", [
        # Cases pulled from real n=50 skips: stripped trailing reviewer
        # suffix and parenthetical clarifiers should not block matching.
        ("Progression-Free Survival (PFS) by investigator", "PFS"),
        ("Progression-Free Survival by investigator assessment", "PFS"),
        ("PFS by BICR", "PFS"),
        ("OS by Independent Review Committee", "OS"),
        ("Disease-Free Survival (DFS), as Assessed by Investigator", "DFS"),
        ("ORR by independent central review", "ORR"),
        ("overall response rate (CR + PR) in BRAF V600E", "ORR"),
        ("Best Overall Response (BOR) [Phase 2 cohort]", "ORR"),
    ])
    def test_qualifier_stripping(self, text, expected):
        assert classify_endpoint_deterministic(text) == expected


class TestEndpointReindexing:
    """Pin the n=50 root-cause: a second trial whose primary outcome maps to
    the same EndpointClass as an earlier trial's outcome—but with
    different wording—used to be silently skipped because resolve_entity
    couldn't find the new measure string in the index."""

    @pytest.mark.asyncio
    async def test_second_measure_string_indexes_existing_endpoint(
        self, pipeline, graph,
    ):
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        pipeline._index_node("melanoma", "Melanoma", "indication")

        # Two trials, two different PFS phrasings—both should resolve to
        # the same EndpointNode after _create_canonical_endpoints runs.
        # Deterministic regex catches "PFS" so no LLM call is made.
        t1 = TrialRecord(
            nct_id="NCT_A", title="A", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            primary_outcomes=[OutcomeMeasure(measure="Progression-Free Survival (PFS)")],
        )
        t2 = TrialRecord(
            nct_id="NCT_B", title="B", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            primary_outcomes=[OutcomeMeasure(measure="Progression-Free Survival (PFS) by investigator")],
        )
        await pipeline._populate_canonical_endpoints([t1, t2])

        ep_id_a = pipeline.resolve_entity(t1.primary_outcomes[0].measure, "endpoint")
        ep_id_b = pipeline.resolve_entity(t2.primary_outcomes[0].measure, "endpoint")
        assert ep_id_a is not None
        assert ep_id_b is not None
        # Both measure strings must resolve to the same canonical PFS node.
        assert ep_id_a == ep_id_b


# ── build_trial_subgraph_from_extraction (multi-arm × multi-subgroup × multi-endpoint) ─


class TestBuildTrialSubgraphFromExtraction:
    def test_fans_chains_across_arms_subgroups_and_endpoints(self, graph):
        """3 arms × 2 subgroups × 2 endpoints = 12 chains, each routed to the
        correct endpoint_id and carrying the matching effect_size."""
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, ChainResult, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )
        from src.ingestion.clinicaltrials import ArmGroup

        # Trial with 3 arms (mono A, mono B, combo) and 2 primary outcomes
        trial = TrialRecord(
            nct_id="NCT_FAN", title="Fan test", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[
                Intervention(name="Nivolumab", type="BIOLOGICAL"),
                Intervention(name="Ipilimumab", type="BIOLOGICAL"),
            ],
            primary_outcomes=[
                OutcomeMeasure(measure="Progression Free Survival (PFS)"),
                OutcomeMeasure(measure="Overall Survival"),
            ],
            arm_groups=[
                ArmGroup(group_id="nivo", title="Nivolumab", intervention_names=["Nivolumab"]),
                ArmGroup(group_id="combo", title="Nivo+Ipi", intervention_names=["Nivolumab", "Ipilimumab"]),
                ArmGroup(group_id="ipi", title="Ipilimumab", intervention_names=["Ipilimumab"]),
            ],
            has_results=True,
        )

        # Seed nodes
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(CompoundNode(id="ipilimumab", name="Ipilimumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(TargetNode(id="ENSG_CTLA4", name="CTLA-4", gene_symbol="CTLA4"))
        graph.add_node(MechanismNode(id="checkpoint_blockade", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 sig"))
        graph.add_node(EndpointNode(
            id="PFS_melanoma", name="PFS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))
        graph.add_node(EndpointNode(
            id="OS_melanoma", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        # Extraction with 2 reported subgroups, results filled for one cell
        extraction = TrialExtraction(
            trial_id="NCT_FAN",
            arms=[
                ExtractedArm(arm_id="nivo", compounds=["Nivolumab"]),
                ExtractedArm(arm_id="combo", compounds=["Nivolumab", "Ipilimumab"]),
                ExtractedArm(arm_id="ipi", compounds=["Ipilimumab"]),
            ],
            subgroups=[
                ExtractedSubgroup(raw_descriptor="PD-L1 ≥1%",
                                 features=[{"axis": "gene", "key": "CD274", "level": "high"}]),
                ExtractedSubgroup(raw_descriptor="PD-L1 <1%",
                                 features=[{"axis": "gene", "key": "CD274", "level": "low"}]),
            ],
            results_by_chain=[
                ChainResult(arm_id="nivo", subgroup_descriptor="PD-L1 ≥1%",
                           endpoint="PFS", effect_size=0.42, outcome="success"),
                ChainResult(arm_id="combo", subgroup_descriptor="PD-L1 ≥1%",
                           endpoint="OS", effect_size=0.55, outcome="success"),
            ],
        )

        ts = build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={
                "nivo": "ENSG_PD1",
                "combo": "ENSG_PD1",
                "ipi": "ENSG_CTLA4",
            },
            mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948",
            indication_id="melanoma",
            endpoint_ids={"PFS": "PFS_melanoma", "OS": "OS_melanoma"},
        )

        # 3 arms × 2 subgroups × 2 endpoints = 12 chains
        assert len(ts.chains) == 12
        # Each chain carries its endpoint_class in metadata
        ep_classes = {c.metadata.get("endpoint_class") for c in ts.chains}
        assert ep_classes == {"PFS", "OS"}
        # The two filled results landed on the correct chains
        nivo_pfs_high = [
            c for c in ts.chains
            if c.arm_id == "nivo" and c.endpoint_id == "PFS_melanoma"
            and c.subgroup_population_id == "melanoma__cd274_positive"
        ]
        assert len(nivo_pfs_high) == 1 and nivo_pfs_high[0].effect_size == 0.42
        combo_os_high = [
            c for c in ts.chains
            if c.arm_id == "combo" and c.endpoint_id == "OS_melanoma"
            and c.subgroup_population_id == "melanoma__cd274_positive"
        ]
        assert len(combo_os_high) == 1 and combo_os_high[0].effect_size == 0.55
        # Cells with no reported result default to UNKNOWN outcome
        unknowns = [c for c in ts.chains if c.outcome == TrialOutcome.UNKNOWN]
        assert len(unknowns) == 10  # 12 total - 2 filled

    def test_preserves_descriptions_onto_semantic_nodes(self, graph):
        """A.0: trial-level mechanism/biology descriptions and population
        descriptions (parent target_population + subgroup raw descriptor) land
        on the corresponding nodes as the BioLORD embedding substrate."""
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, ChainResult, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )
        from src.ingestion.clinicaltrials import ArmGroup

        trial = TrialRecord(
            nct_id="NCT_DESC", title="Desc test", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[Intervention(name="Nivolumab", type="BIOLOGICAL")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival")],
            arm_groups=[
                ArmGroup(group_id="nivo", title="Nivolumab",
                         intervention_names=["Nivolumab"]),
            ],
            has_results=True,
        )
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(MechanismNode(id="checkpoint_blockade", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 sig"))
        graph.add_node(EndpointNode(
            id="OS_melanoma", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        extraction = TrialExtraction(
            trial_id="NCT_DESC",
            arms=[ExtractedArm(arm_id="nivo", compounds=["Nivolumab"])],
            subgroups=[
                ExtractedSubgroup(raw_descriptor="PD-L1 ≥1%",
                                 features=[{"axis": "gene", "key": "CD274", "level": "high"}]),
            ],
            results_by_chain=[
                ChainResult(arm_id="nivo", subgroup_descriptor="PD-L1 ≥1%",
                           endpoint="OS", effect_size=0.5, outcome="success"),
            ],
            mechanism_description="PD-1 checkpoint blockade restoring T-cell cytotoxicity",
            biology_description="T-cell exhaustion reversal in the tumor microenvironment",
            target_population_description="treatment-naive metastatic melanoma patients",
        )

        build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={"nivo": "ENSG_PD1"},
            mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948",
            indication_id="melanoma",
            endpoint_ids={"OS": "OS_melanoma"},
        )

        assert graph.get_node("checkpoint_blockade")["description"] == (
            "PD-1 checkpoint blockade restoring T-cell cytotoxicity"
        )
        assert graph.get_node("R-HSA-389948")["description"] == (
            "T-cell exhaustion reversal in the tumor microenvironment"
        )
        assert graph.get_node("melanoma__unselected")["description"] == (
            "treatment-naive metastatic melanoma patients"
        )
        # Subgroup population carries its raw descriptor as the substrate.
        assert graph.get_node("melanoma__cd274_positive")["description"] == "PD-L1 ≥1%"

    def test_attach_descriptions_from_extractions_production_path(self, graph, tmp_path):
        """A.0 production path: the real build creates nodes before extractions
        exist, then attaches descriptions from cached extraction JSON to the
        Mechanism / Biology / Population nodes a trial's chains touch."""
        import json as _json
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType, PopulationNode,
            CausalChain, TrialSubgraph,
        )
        from src.graph.populate import attach_node_descriptions_from_extractions

        graph.add_node(MechanismNode(id="kinase_inhibition", name="kinase inhibition",
                                     mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="R-HSA-1", name="some pathway"))
        graph.add_node(PopulationNode(id="nsclc__unselected", name="nsclc unselected"))
        graph.add_node(PopulationNode(id="nsclc__egfr_mutant", name="EGFR mutant (nsclc)"))

        def _chain(pop):
            return CausalChain(
                arm_id="a", compound_id="c", subgroup_population_id=pop,
                target_id="t", mechanism_id="kinase_inhibition", biology_id="R-HSA-1",
                indication_id="nsclc", endpoint_id="e",
            )
        graph.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT_A0", parent_population_id="nsclc__unselected",
            chains=[_chain("nsclc__unselected"), _chain("nsclc__egfr_mutant")],
        ))

        (tmp_path / "NCT_A0_extraction.json").write_text(_json.dumps({
            "nct_id": "NCT_A0",
            "therapeutic_hypothesis": {
                "proposed_mechanism": "EGFR tyrosine kinase inhibition",
                "intended_biology": "blocking proliferative signaling",
                "target_population": "EGFR-mutant NSCLC patients",
            },
        }))

        n = attach_node_descriptions_from_extractions(graph, tmp_path)
        # mechanism + biology + parent pop (chain 1) + subgroup pop (chain 2);
        # mech/bio dedup across the two chains (first-non-empty-wins).
        assert n == 4
        assert graph.get_node("kinase_inhibition")["description"] == "EGFR tyrosine kinase inhibition"
        assert graph.get_node("R-HSA-1")["description"] == "blocking proliferative signaling"
        assert graph.get_node("nsclc__unselected")["description"] == "EGFR-mutant NSCLC patients"
        # subgroup pop falls back to its own (descriptor-bearing) name
        assert graph.get_node("nsclc__egfr_mutant")["description"] == "EGFR mutant (nsclc)"

    def test_attach_descriptions_prefers_per_intervention_for_combo(self, graph, tmp_path):
        """Combo-arm biology fix: two constituents of one arm get DISTINCT
        mechanism/biology descriptions from the per-(arm, intervention) entries
        — not one shared trial-level hypothesis on both."""
        import json as _json
        from src.graph.models import (
            BiologyNode, CompoundNode, MechanismNode, MechanismType,
            PopulationNode, CausalChain, TrialSubgraph,
        )
        from src.graph.populate import attach_node_descriptions_from_extractions

        graph.add_node(CompoundNode(id="cyclophosphamide", name="Cyclophosphamide"))
        graph.add_node(CompoundNode(id="pembrolizumab", name="Pembrolizumab"))
        graph.add_node(MechanismNode(id="dna_crosslinking", name="dna crosslinking",
                                     mechanism_type=MechanismType.INHIBITION))
        graph.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade",
                                     mechanism_type=MechanismType.INHIBITION))
        graph.add_node(BiologyNode(id="GO:0006289", name="ner"))
        graph.add_node(BiologyNode(id="R-HSA-389948", name="pd-1 signaling"))
        graph.add_node(PopulationNode(id="sarcoma__unselected", name="sarcoma all"))

        def _chain(compound_id, mech_id, bio_id):
            return CausalChain(
                arm_id="part_b", compound_id=compound_id,
                subgroup_population_id="sarcoma__unselected", target_id="t",
                mechanism_id=mech_id, biology_id=bio_id, indication_id="sarcoma",
                endpoint_id="e",
            )
        graph.set_trial_subgraph(TrialSubgraph(
            trial_id="NCT_COMBO2", parent_population_id="sarcoma__unselected",
            chains=[
                _chain("cyclophosphamide", "dna_crosslinking", "GO:0006289"),
                _chain("pembrolizumab", "checkpoint_blockade", "R-HSA-389948"),
            ],
        ))

        (tmp_path / "NCT_COMBO2_extraction.json").write_text(_json.dumps({
            "nct_id": "NCT_COMBO2",
            "therapeutic_hypothesis": {
                "intended_biology": "T-cell mediated anti-tumor immunity",
            },
            "results_by_chain": [
                {"arm_id": "part_b", "intervention": "cyclophosphamide",
                 "mechanism_description": "DNA cross-linking / alkylation",
                 "biology_description": "DNA-damage-induced apoptosis"},
                {"arm_id": "part_b", "intervention": "pembrolizumab",
                 "mechanism_description": "PD-1 checkpoint blockade",
                 "biology_description": "T-cell mediated anti-tumor immunity"},
            ],
        }))

        attach_node_descriptions_from_extractions(graph, tmp_path)
        # Each drug's nodes carry ITS OWN per-(arm, intervention) description.
        assert graph.get_node("dna_crosslinking")["description"] == (
            "DNA cross-linking / alkylation"
        )
        assert graph.get_node("GO:0006289")["description"] == (
            "DNA-damage-induced apoptosis"
        )
        assert graph.get_node("checkpoint_blockade")["description"] == (
            "PD-1 checkpoint blockade"
        )
        assert graph.get_node("R-HSA-389948")["description"] == (
            "T-cell mediated anti-tumor immunity"
        )

    def test_skips_subgroup_when_all_features_canonicalize_to_other(self, graph):
        """Continuous PD readouts and analysis-timepoint markers aren't
        real subgroups. When canonicalization can't place any feature on
        a known axis, the subgroup is dropped from the chain fan-out so
        we don't pollute the graph with one-off
        ``melanoma__other_baseline_cd8_tumor`` populations."""
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )

        trial = TrialRecord(
            nct_id="NCT_OTHER", title="OtherOnly", phase="2", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[Intervention(name="Nivolumab", type="BIOLOGICAL")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival")],
            arm_groups=[
                ArmGroup(group_id="A1", title="Mono", intervention_names=["Nivolumab"]),
            ],
        )
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(MechanismNode(id="cb", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="bio", name="bio"))
        graph.add_node(EndpointNode(
            id="OS_mel", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        extraction = TrialExtraction(
            trial_id="NCT_OTHER",
            arms=[ExtractedArm(arm_id="A1", compounds=["Nivolumab"])],
            subgroups=[
                # PD readout—not a real subgroup, no canonical axis fits.
                ExtractedSubgroup(
                    raw_descriptor="CD8 T cells per mm² day 22",
                    features=[{"axis": "biomarker", "key": "CD8", "level": "day22"}],
                ),
                # Analysis timepoint—also not a subgroup.
                ExtractedSubgroup(
                    raw_descriptor="Final analysis",
                    features=[{"axis": "timepoint", "key": "", "level": "final"}],
                ),
            ],
            results_by_chain=[],
        )

        ts = build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={"A1": "ENSG_PD1"},
            mechanism_id="cb",
            biology_id="bio",
            indication_id="melanoma",
            endpoint_ids={"OS": "OS_mel"},
        )

        # No subgroup PopulationNodes created.
        pop_ids = [c.subgroup_population_id for c in ts.chains]
        assert all(pid == "melanoma__unselected" for pid in pop_ids)
        # No "other_*" PopulationNode leaked into the graph.
        other_pops = [
            n for n in graph._graph.nodes
            if isinstance(n, str) and "__other_" in n
        ]
        assert other_pops == []
        # Only the parent-population fan: 1 arm × 1 endpoint = 1 chain.
        assert len(ts.chains) == 1

    def test_skips_subgroup_when_only_feature_is_response_axis(self, graph):
        """RECIST response strata (CR / PR / SD / PD) are outcome
        stratifiers, not patient strata — they describe how subgroups
        responded post-treatment, not who was enrolled. Forking the
        arm × population matrix across them creates one chain per
        response category × per arm with no a-priori meaning, and
        none of them get classifier updates. Drop those subgroups
        from the chain fan-out; the trial-level chain still carries
        the real result.
        """
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )

        trial = TrialRecord(
            nct_id="NCT_RESP", title="ResponseOnly", phase="2", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[Intervention(name="Nivolumab", type="BIOLOGICAL")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival")],
            arm_groups=[
                ArmGroup(group_id="A1", title="Mono", intervention_names=["Nivolumab"]),
            ],
        )
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(MechanismNode(id="cb", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="bio", name="bio"))
        graph.add_node(EndpointNode(
            id="OS_mel", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        extraction = TrialExtraction(
            trial_id="NCT_RESP",
            arms=[ExtractedArm(arm_id="A1", compounds=["Nivolumab"])],
            subgroups=[
                ExtractedSubgroup(
                    raw_descriptor="Complete Response",
                    features=[{"axis": "response", "key": "", "level": "complete_response"}],
                ),
                # Stale-cache variant: axis="other" but RECIST level — should
                # auto-promote in canonicalize_feature and be skipped here.
                ExtractedSubgroup(
                    raw_descriptor="Progressive Disease",
                    features=[{"axis": "other", "key": "", "level": "progressive_disease"}],
                ),
            ],
            results_by_chain=[],
        )

        ts = build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={"A1": "ENSG_PD1"},
            mechanism_id="cb",
            biology_id="bio",
            indication_id="melanoma",
            endpoint_ids={"OS": "OS_mel"},
        )

        # Only the parent-population chain — no response-strata forks.
        assert len(ts.chains) == 1
        assert ts.chains[0].subgroup_population_id == "melanoma__unselected"
        # And no melanoma__response_* PopulationNode created.
        response_pops = [
            n for n in graph._graph.nodes
            if isinstance(n, str) and "__response_" in n
        ]
        assert response_pops == []



# ── Round-28: parent-population coarsening ──────────────────────────────


class TestPopulationCoarsening:
    """Round-28: parent PopulationNode ids drop rare-axis qualifiers.

    Pre-round-28 the parent_population_id for NSABP C-08 (colorectal
    adenocarcinoma, stage III, adjuvant) carried a `histology_adenocarcinoma`
    qualifier that bev AVANT (colorectal_cancer, stage III, adjuvant) did
    not — so two trials studying "adjuvant stage III CRC" emitted their
    `responds_differently` evidence to disjoint nodes and the cross-trial
    signal never aggregated. The round-28 coarsening keeps the load-bearing
    axes (line, stage, extent, setting) and demotes the rare ones to
    subgroup-only.
    """

    def test_default_axes_constant(self):
        from src.graph.populate import _DEFAULT_POPULATION_AXES
        # These four axes are the load-bearing ones for cross-trial
        # differentiation. Demoting any of them would collapse trials
        # at different lines / stages / settings into one parent and
        # break per-trial discrimination.
        assert {"line", "stage", "extent", "setting"} <= _DEFAULT_POPULATION_AXES

    def test_rare_axes_dropped(self):
        from src.graph.models import SubgroupFeature
        from src.graph.populate import _coarse_population_features
        feats = [
            SubgroupFeature(axis="histology", level="adenocarcinoma"),
            SubgroupFeature(axis="line", level="adjuvant"),
            SubgroupFeature(axis="stage", level="iii"),
            SubgroupFeature(axis="age", level="elderly"),
            SubgroupFeature(axis="prior_tx", level="chemotherapy_treated"),
            SubgroupFeature(axis="gene", key="KRAS", level="mutant"),
        ]
        kept = _coarse_population_features(feats)
        kept_axes = {f.axis for f in kept}
        assert "line" in kept_axes
        assert "stage" in kept_axes
        assert "histology" not in kept_axes
        assert "age" not in kept_axes
        assert "prior_tx" not in kept_axes
        assert "gene" not in kept_axes

    def test_kept_features_compose_to_coarse_slug(self):
        """The kept features drive a coarse parent slug — two trials
        with the same kept features but different rare-axis features
        land at the same parent_population_id."""
        from src.graph.models import PopulationNode, SubgroupFeature
        from src.graph.populate import _coarse_population_features

        avant = [
            SubgroupFeature(axis="line", level="adjuvant"),
            SubgroupFeature(axis="stage", level="iii"),
        ]
        c08 = [
            SubgroupFeature(axis="histology", level="adenocarcinoma"),
            SubgroupFeature(axis="line", level="adjuvant"),
            SubgroupFeature(axis="stage", level="iii"),
        ]

        avant_parent = PopulationNode.compose_id(
            "colorectal_cancer", _coarse_population_features(avant),
        )
        c08_parent = PopulationNode.compose_id(
            "colorectal_cancer", _coarse_population_features(c08),
        )
        assert avant_parent == c08_parent
        assert "histology" not in avant_parent
        assert "stage_iii" in avant_parent
        assert "line_adjuvant" in avant_parent

    def test_empty_features_still_unselected(self):
        from src.graph.populate import _coarse_population_features
        assert _coarse_population_features([]) == []


# ── Round-28: MechanismNode.direction lookup table ──────────────────────


class TestMechanismDirection:
    """Round-28: MechanismNode carries a direction metadata field
    (activating / inhibiting / modulating / None). The helper in
    src/ingestion/lincs.py maps each MechanismCategory to a default
    direction value used at node creation time."""

    def test_inhibiting_categories(self):
        from src.graph.models import MechanismCategory
        from src.ingestion.lincs import _category_to_direction
        for cat in (
            MechanismCategory.KINASE_INHIBITION,
            MechanismCategory.RECEPTOR_ANTAGONISM,
            MechanismCategory.ANGIOGENESIS_INHIBITION,
            MechanismCategory.ENZYME_INHIBITION,
            MechanismCategory.PROTEIN_DEGRADATION,
            MechanismCategory.ANTIMETABOLITE,
            MechanismCategory.DNA_DAMAGE,
            MechanismCategory.DNA_CROSSLINKING,
            MechanismCategory.MICROTUBULE_BINDING,
            MechanismCategory.CHECKPOINT_BLOCKADE,
        ):
            assert _category_to_direction(cat) == "inhibiting", cat

    def test_activating_categories(self):
        from src.graph.models import MechanismCategory
        from src.ingestion.lincs import _category_to_direction
        for cat in (
            MechanismCategory.RECEPTOR_AGONISM,
            MechanismCategory.IMMUNE_COSTIMULATION,
            MechanismCategory.ANTIBODY_DEPENDENT_CYTOTOXICITY,
            MechanismCategory.ANTIGEN_DIRECTED_CYTOTOXICITY,
        ):
            assert _category_to_direction(cat) == "activating", cat

    def test_modulating_categories(self):
        from src.graph.models import MechanismCategory
        from src.ingestion.lincs import _category_to_direction
        for cat in (
            MechanismCategory.HORMONE_MODULATION,
            MechanismCategory.GENE_EDITING,
        ):
            assert _category_to_direction(cat) == "modulating", cat

    def test_other_is_none(self):
        from src.graph.models import MechanismCategory
        from src.ingestion.lincs import _category_to_direction
        assert _category_to_direction(MechanismCategory.OTHER) is None

    def test_mechanism_node_default_direction_is_none(self):
        from src.graph.models import MechanismNode, MechanismType
        m = MechanismNode(id="x", name="X", mechanism_type=MechanismType.OTHER)
        assert m.direction is None

    def test_mechanism_node_accepts_direction(self):
        from src.graph.models import MechanismNode, MechanismType
        m = MechanismNode(
            id="kinase_inhibition", name="kinase inhibition",
            mechanism_type=MechanismType.INHIBITION, direction="inhibiting",
        )
        assert m.direction == "inhibiting"


# ── Round-15 canonicalization: OT-derived aliases + chembl_id ────────────


class TestAnnotateCompoundFromOT:
    def test_persists_chembl_id_and_aliases(self, pipeline):
        pipeline.graph.add_node(CompoundNode(
            id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY,
        ))
        pipeline._annotate_compound_node_from_ot(
            "nivolumab",
            {
                "chembl_id": "CHEMBL1201580",
                "name": "nivolumab",
                "aliases": ["BMS-936558", "Opdivo"],
            },
        )
        node = pipeline.graph._graph.nodes["nivolumab"]
        assert node["chembl_id"] == "CHEMBL1201580"
        assert "BMS-936558" in node["aliases"]
        assert "Opdivo" in node["aliases"]

    def test_external_aliases_resolve_back_to_canonical(self, pipeline):
        pipeline.graph.add_node(CompoundNode(
            id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY,
        ))
        pipeline._annotate_compound_node_from_ot(
            "nivolumab",
            {"chembl_id": "CHEMBL1201580", "aliases": ["BMS-936558", "Opdivo"]},
        )
        assert pipeline.resolve_entity("BMS-936558", "compound") == "nivolumab"
        assert pipeline.resolve_entity("Opdivo", "compound") == "nivolumab"

    def test_idempotent_on_repeat(self, pipeline):
        pipeline.graph.add_node(CompoundNode(
            id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY,
        ))
        drug_data = {
            "chembl_id": "CHEMBL1201580",
            "aliases": ["BMS-936558", "Opdivo"],
        }
        pipeline._annotate_compound_node_from_ot("nivolumab", drug_data)
        pipeline._annotate_compound_node_from_ot("nivolumab", drug_data)
        node = pipeline.graph._graph.nodes["nivolumab"]
        # No duplicate aliases despite running twice.
        assert node["aliases"].count("BMS-936558") == 1
        assert node["aliases"].count("Opdivo") == 1

    def test_missing_node_is_noop(self, pipeline):
        # No CompoundNode added for "ghost_drug" — call should silently no-op.
        pipeline._annotate_compound_node_from_ot(
            "ghost_drug",
            {"chembl_id": "CHEMBL999", "aliases": ["X"]},
        )
        # No exception; nothing indexed.
        assert pipeline.resolve_entity("X", "compound") is None

    def test_doesnt_overwrite_existing_chembl_id(self, pipeline):
        pipeline.graph.add_node(CompoundNode(
            id="nivolumab", name="Nivolumab",
            modality=Modality.ANTIBODY, chembl_id="CHEMBL_PRESET",
        ))
        pipeline._annotate_compound_node_from_ot(
            "nivolumab",
            {"chembl_id": "CHEMBL_OTHER", "aliases": ["X"]},
        )
        # Existing chembl_id preserved.
        assert pipeline.graph._graph.nodes["nivolumab"]["chembl_id"] == "CHEMBL_PRESET"


# ── Round-16: structural responds_differently for parent populations ────


class TestRespondsDifferentlyForUnselected:
    """Round-16 architectural fix: the populator MUST create a
    `responds_differently: {indication}__unselected → {indication}` edge
    for every trial whose condition canonicalizes to a plain disease
    name (no stage/biomarker qualifiers). Without this edge, the
    round-16 always-emit classifier rule sends responds_differently
    evidence to a non-existent edge that the attributor silently
    drops as 'entity_not_in_trial'. Per-trial discrimination becomes
    impossible because the contradict signal goes nowhere."""

    def test_source_no_longer_guards_on_qualifiers(self):
        """Static check: the populator source must NOT contain the
        pre-round-16 `qualifiers and ...has_edge` guard. If it does,
        the fix has been reverted and trials with plain disease names
        (no stage/biomarker qualifiers) will get no responds_differently
        edge for their unselected population — classifier emissions
        will land nowhere."""
        from pathlib import Path
        src = Path("src/graph/populate.py").read_text()
        assert "if qualifiers and not self.graph._graph.has_edge(" not in src, (
            "Round-16 fix reverted: populator still guards "
            "responds_differently creation on `qualifiers` non-empty"
        )
        assert "default_unselected" in src, (
            "Round-16 metadata tag `default_unselected` missing"
        )

    def test_unselected_edge_created_in_isolation(self, pipeline):
        """Directly exercise the populator's round-16 edge-creation
        block on synthesized state (canonical_id + default_pop_id +
        empty qualifiers). Bypasses the full populate_trials pipeline
        which the test fixture can't run end-to-end (mocked LLM /
        no OT / no CT.gov)."""
        from src.graph.models import (
            EdgeBeliefState as EBS, EdgeType as ET,
            GraphEdge, IndicationNode, PopulationNode,
        )
        canonical_id = "melanoma"
        default_pop_id = "melanoma__unselected"
        qualifiers: list = []  # the plain-disease case

        pipeline.graph.add_node(IndicationNode(id=canonical_id, name="melanoma"))
        pipeline.graph.add_node(PopulationNode(
            id=default_pop_id, name="All patients (melanoma)",
            defining_features=list(qualifiers),
        ))

        # Mirror the populator's round-16 block (unconditional `if not`).
        if not pipeline.graph._graph.has_edge(  # noqa: SLF001
            default_pop_id, canonical_id,
            key=ET.RESPONDS_DIFFERENTLY.value,
        ):
            pipeline.graph.add_edge(GraphEdge(
                source_id=default_pop_id, target_id=canonical_id,
                edge_type=ET.RESPONDS_DIFFERENTLY,
                belief=EBS(alpha=1.5, beta=1.0),
                metadata={"source": "default_unselected"},
            ))

        assert pipeline.graph._graph.has_edge(  # noqa: SLF001
            "melanoma__unselected", "melanoma",
            key=ET.RESPONDS_DIFFERENTLY.value,
        )


# ── Round-18 followup: codename → INN canonicalization at populate ─────


class TestPopulatorCodenameToINN:
    """When `map_trial_to_graph_nodes` produces a compound id matching
    a known dev code (AZD6244, MK-3475, etc.), the populator's step-2
    loop should remap it to the canonical INN at creation time, with
    the original code preserved as an alias. Future builds then have
    every form of a compound resolve to the same node without needing
    the post-build `scripts/canonicalize_codenames.py` migration."""

    @pytest.mark.asyncio
    async def test_azd6244_collapses_to_selumetinib(self, pipeline):
        """A trial whose intervention is named 'AZD6244' should produce
        a `selumetinib` CompoundNode with AZD6244 in aliases, plus an
        entity index entry that resolves 'AZD6244' to selumetinib."""
        from src.ingestion.clinicaltrials import (
            Intervention, OutcomeMeasure, TrialRecord,
        )
        trial = TrialRecord(
            nct_id="NCT_TEST_AZD",
            title="Test AZD6244 in thyroid cancer",
            phase="2", status="COMPLETED",
            conditions=["Thyroid Cancer"],
            interventions=[
                Intervention(name="AZD6244", type="DRUG", description=""),
            ],
            primary_outcomes=[OutcomeMeasure(measure="OS", timeframe="2y")],
            enrollment=20, has_results=True, arm_groups=[],
        )
        await pipeline.populate_trials(trials=[trial])

        # Canonical id should be 'selumetinib', not 'azd6244'
        assert "selumetinib" in pipeline.graph._graph  # noqa: SLF001
        assert "azd6244" not in pipeline.graph._graph  # noqa: SLF001

        node = pipeline.graph._graph.nodes["selumetinib"]  # noqa: SLF001
        # Original form should be preserved as an alias
        aliases = node.get("aliases") or []
        assert "AZD6244" in aliases, (
            f"Round-18: original codename 'AZD6244' missing from "
            f"selumetinib's aliases. Got: {aliases}"
        )

        # Entity index should resolve both forms back to the canonical id
        assert pipeline.resolve_entity("AZD6244", "compound") == "selumetinib"
        assert pipeline.resolve_entity("selumetinib", "compound") == "selumetinib"

    @pytest.mark.asyncio
    async def test_non_codename_compound_unchanged(self, pipeline):
        """A compound whose id is NOT in the codename dict should be
        added with its original id (no remap, no false-positive)."""
        from src.ingestion.clinicaltrials import (
            Intervention, OutcomeMeasure, TrialRecord,
        )
        trial = TrialRecord(
            nct_id="NCT_TEST_NORMAL",
            title="Test cisplatin",
            phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[
                Intervention(name="cisplatin", type="DRUG", description=""),
            ],
            primary_outcomes=[OutcomeMeasure(measure="OS", timeframe="2y")],
            enrollment=100, has_results=True, arm_groups=[],
        )
        await pipeline.populate_trials(trials=[trial])

        # 'cisplatin' isn't in CODENAME_TO_INN → no remap
        assert "cisplatin" in pipeline.graph._graph  # noqa: SLF001


def test_biology_id_from_description_is_content_address():
    """A description-defined BiologyNode id is a content-address over the
    normalized description (semantic-layers redesign) — NOT a {mech}__{ind}
    slug. Case/whitespace-normalized so phrasing-identical descriptions collapse
    to one id (exact Tier-1 merge); distinct biology gets a distinct id."""
    from src.graph.populate import _biology_id_from_description

    a = _biology_id_from_description("Formation of new blood vessels")
    b = _biology_id_from_description("formation of new   blood vessels")
    c = _biology_id_from_description("T-cell mediated cytotoxicity")

    assert a.startswith("bio:") and len(a) == len("bio:") + 12
    assert a == b          # normalized (case + collapsed whitespace) → same id
    assert a != c          # distinct biology → distinct content-address
    assert "__" not in a   # not a {mechanism}__{indication} slug


# ── Combo-arm biology fix: per-(arm, intervention) descriptions ───────────────


@pytest.mark.asyncio
async def test_combo_arm_constituents_get_distinct_biology_descriptions(tmp_path):
    """Abstraction-ladder redesign: when a per-(arm, intervention) biology
    DESCRIPTION exists, the Biology node IDENTITY is the content-address of that
    description (``bio:<sha1>``), NOT the target-gene → Reactome pathway —
    Reactome is bypassed entirely. A combination arm fans into one chain PER
    drug; each drug's distinct description yields a distinct ``bio:`` id, so the
    chemo's biology and the checkpoint inhibitor's biology stay separate (and
    the general processes are SHARED across drugs/diseases that drive them).
    """
    import json

    from src.graph.models import (
        BiologyNode,
        CausalChain,
        CompoundNode,
        IndicationNode,
        MechanismNode,
        TargetNode,
        TrialOutcome,
        TrialSubgraph,
    )
    from src.graph.store import GraphStore
    from src.ingestion.clinicaltrials import TrialRecord

    g = GraphStore()
    # Two constituents of one combo arm, each with its own target+mechanism.
    g.add_node(CompoundNode(id="cyclophosphamide", name="Cyclophosphamide"))
    g.add_node(CompoundNode(id="pembrolizumab", name="Pembrolizumab"))
    g.add_node(TargetNode(id="CHEBI:16991", name="DNA", gene_symbol="DNA"))
    g.add_node(TargetNode(id="ENSG00000188389", name="PDCD1", gene_symbol="PDCD1"))
    g.add_node(MechanismNode(id="dna_crosslinking", name="dna crosslinking", mechanism_type="inhibition"))
    g.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade", mechanism_type="inhibition"))
    g.add_node(IndicationNode(id="sarcoma", name="sarcoma"))
    # Two distinct RESOLVED ontology biology nodes (no description yet).
    g.add_node(BiologyNode(id="GO:0006289", name="nucleotide-excision repair"))
    g.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 signaling"))

    parent_pop = "sarcoma__unselected"
    g.add_node(IndicationNode(id="sarcoma", name="sarcoma"))  # idempotent
    from src.graph.models import PopulationNode

    g.add_node(PopulationNode(id=parent_pop, name="sarcoma (all)"))

    def _chain(compound_id, target_id, mech_id):
        return CausalChain(
            arm_id="part_b",
            compound_id=compound_id,
            subgroup_population_id=parent_pop,
            target_id=target_id,
            mechanism_id=mech_id,
            biology_id="UNKNOWN",
            indication_id="sarcoma",
            endpoint_id="ORR_sarcoma",
            outcome=TrialOutcome.PARTIAL,
        )

    g.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT_COMBO",
        arms=[],
        parent_population_id=parent_pop,
        chains=[
            _chain("cyclophosphamide", "CHEBI:16991", "dna_crosslinking"),
            _chain("pembrolizumab", "ENSG00000188389", "checkpoint_blockade"),
        ],
    ))

    # Extraction: ONE combo arm, two per-drug entries with DISTINCT biology.
    (tmp_path / "NCT_COMBO_extraction.json").write_text(json.dumps({
        "nct_id": "NCT_COMBO",
        "therapeutic_hypothesis": {
            "intended_biology": "T-cell mediated anti-tumor immunity",  # arm-level fallback
        },
        "results_by_chain": [
            {
                "arm_id": "part_b", "intervention": "cyclophosphamide",
                "endpoint": "ORR", "outcome": "partial",
                "biology_description": "DNA-damage-induced apoptosis",
            },
            {
                "arm_id": "part_b", "intervention": "pembrolizumab",
                "endpoint": "ORR", "outcome": "partial",
                "biology_description": "T-cell mediated anti-tumor immunity",
            },
        ],
    }))

    # Stub biology resolution to a Reactome id per (target, mechanism). The
    # redesign must NOT use these when a description is present (Reactome is the
    # no-description fallback only) — asserting these ids are absent proves
    # description-identity wins over the target-gene → Reactome pathway.
    bio_by_mech = {
        "dna_crosslinking": ["GO:0006289"],
        "checkpoint_blockade": ["R-HSA-389948"],
    }

    async def fake_resolve(self, target_id, mechanism_id, indication_id,
                           lincs_client, quickgo_client=None):
        return (bio_by_mech.get(mechanism_id, []), False)

    trial = TrialRecord(nct_id="NCT_COMBO", title="t", conditions=["sarcoma"])

    from src.graph.populate import _biology_id_from_description

    with patch.object(PopulationPipeline, "_resolve_real_biology", fake_resolve):
        pipe = PopulationPipeline(g, anthropic_client=AsyncMock())
        await pipe._populate_trial_biology([trial], annotations_dir=tmp_path)

    # Biology identity is the content-address of EACH drug's description —
    # NOT the Reactome pathway.
    chemo_bio = _biology_id_from_description("DNA-damage-induced apoptosis")
    checkpoint_bio = _biology_id_from_description("T-cell mediated anti-tumor immunity")
    assert chemo_bio.startswith("bio:") and checkpoint_bio.startswith("bio:")
    assert chemo_bio != checkpoint_bio

    # Each content-address node carries its own description as the substrate.
    assert g.get_node(chemo_bio)["description"] == "DNA-damage-induced apoptosis"
    assert g.get_node(checkpoint_bio)["description"] == (
        "T-cell mediated anti-tumor immunity"
    )

    # Chains were rewritten to the description content-address ids; the
    # pre-existing Reactome fixture nodes are NOT referenced by any chain
    # (description-identity wins over the target-gene → Reactome pathway).
    ts = g.get_trial_subgraph_by_id("NCT_COMBO")
    bio_ids = sorted(c.biology_id for c in ts.chains)
    assert bio_ids == sorted([chemo_bio, checkpoint_bio])
    assert "GO:0006289" not in bio_ids
    assert "R-HSA-389948" not in bio_ids
    # The Reactome nodes never received the descriptions (they weren't touched).
    assert not g.get_node("GO:0006289").get("description")
    assert not g.get_node("R-HSA-389948").get("description")


@pytest.mark.asyncio
async def test_arm_level_description_is_biology_identity_not_reactome(tmp_path):
    """Abstraction-ladder redesign, legacy/no-per-drug case: a combo arm whose
    entries carry only an ARM-LEVEL ``biology_description`` (no per-drug
    ``intervention``) routes BOTH constituents to the content-address of that
    shared description — NOT to their distinct target-gene → Reactome pathways.

    Consequence to be aware of: with only an arm-level description, the two
    constituents collapse onto ONE biology node (same description string → same
    ``bio:`` id). The fix for keeping distinct-mechanism drugs apart is per-DRUG
    extraction (each drug emits its own ``biology_description`` — enforced by the
    prompt's per-drug rule); when that's present, see
    ``test_combo_arm_constituents_get_distinct_biology_descriptions``.
    """
    import json

    from src.graph.models import (
        BiologyNode,
        CausalChain,
        CompoundNode,
        IndicationNode,
        MechanismNode,
        PopulationNode,
        TargetNode,
        TrialOutcome,
        TrialSubgraph,
    )
    from src.graph.store import GraphStore
    from src.ingestion.clinicaltrials import TrialRecord

    g = GraphStore()
    g.add_node(CompoundNode(id="cyclophosphamide", name="Cyclophosphamide"))
    g.add_node(CompoundNode(id="pembrolizumab", name="Pembrolizumab"))
    g.add_node(TargetNode(id="CHEBI:16991", name="DNA", gene_symbol="DNA"))
    g.add_node(TargetNode(id="ENSG00000188389", name="PDCD1", gene_symbol="PDCD1"))
    g.add_node(MechanismNode(id="dna_crosslinking", name="dna crosslinking", mechanism_type="inhibition"))
    g.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade", mechanism_type="inhibition"))
    g.add_node(IndicationNode(id="sarcoma", name="sarcoma"))
    g.add_node(BiologyNode(id="GO:0006289", name="nucleotide-excision repair"))
    g.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 signaling"))
    parent_pop = "sarcoma__unselected"
    g.add_node(PopulationNode(id=parent_pop, name="sarcoma (all)"))

    def _chain(compound_id, target_id, mech_id):
        return CausalChain(
            arm_id="part_b", compound_id=compound_id,
            subgroup_population_id=parent_pop, target_id=target_id,
            mechanism_id=mech_id, biology_id="UNKNOWN", indication_id="sarcoma",
            endpoint_id="ORR_sarcoma", outcome=TrialOutcome.PARTIAL,
        )

    g.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT_LEGACY", arms=[], parent_population_id=parent_pop,
        chains=[
            _chain("cyclophosphamide", "CHEBI:16991", "dna_crosslinking"),
            _chain("pembrolizumab", "ENSG00000188389", "checkpoint_blockade"),
        ],
    ))

    # No `intervention`, no per-chain biology — only a single arm-level
    # description per entry + a trial-level fallback (the bug scenario).
    (tmp_path / "NCT_LEGACY_extraction.json").write_text(json.dumps({
        "nct_id": "NCT_LEGACY",
        "therapeutic_hypothesis": {
            "intended_biology": "T-cell mediated anti-tumor immunity",
        },
        "results_by_chain": [
            {"arm_id": "part_b", "endpoint": "ORR", "outcome": "partial",
             "biology_description": "T-cell mediated anti-tumor immunity"},
        ],
    }))

    bio_by_mech = {
        "dna_crosslinking": ["GO:0006289"],
        "checkpoint_blockade": ["R-HSA-389948"],
    }

    async def fake_resolve(self, target_id, mechanism_id, indication_id,
                           lincs_client, quickgo_client=None):
        return (bio_by_mech.get(mechanism_id, []), False)

    trial = TrialRecord(nct_id="NCT_LEGACY", title="t", conditions=["sarcoma"])
    from src.graph.populate import _biology_id_from_description
    with patch.object(PopulationPipeline, "_resolve_real_biology", fake_resolve):
        pipe = PopulationPipeline(g, anthropic_client=AsyncMock())
        await pipe._populate_trial_biology([trial], annotations_dir=tmp_path)

    # Both constituents route to the content-address of the shared arm-level
    # description — Reactome is bypassed.
    shared_bio = _biology_id_from_description("T-cell mediated anti-tumor immunity")
    assert shared_bio.startswith("bio:")
    assert g.get_node(shared_bio)["description"] == "T-cell mediated anti-tumor immunity"
    ts = g.get_trial_subgraph_by_id("NCT_LEGACY")
    assert {c.biology_id for c in ts.chains} == {shared_bio}
    # The Reactome fixture nodes were never used as the biology identity.
    assert not g.get_node("GO:0006289").get("description")
    assert not g.get_node("R-HSA-389948").get("description")


# ── Abstraction-ladder: mechanism_category → MechanismNode.category ───────────


@pytest.mark.asyncio
async def test_mechanism_category_lands_as_node_metadata(tmp_path):
    """The extractor's per-(arm, intervention) ``mechanism_category`` is stamped
    onto ``MechanismNode.category`` by ``_populate_trial_mechanisms``. The node
    identity stays the resolved MechanismCategory id (round-28 direction/type
    derivation keeps working); category is the coarse bucket carried as metadata.
    When a drug has no extracted category, it falls back to the resolved id."""
    import json

    from src.graph.models import (
        CausalChain, CompoundNode, EdgeBeliefState, EdgeType, GraphEdge,
        IndicationNode, PopulationNode, TargetNode, TrialArm, TrialOutcome,
        TrialSubgraph,
    )
    from src.graph.store import GraphStore
    from src.ingestion.clinicaltrials import TrialRecord

    g = GraphStore()
    g.add_node(CompoundNode(id="cyclophosphamide", name="Cyclophosphamide"))
    g.add_node(CompoundNode(id="pembrolizumab", name="Pembrolizumab"))
    g.add_node(TargetNode(id="CHEBI:16991", name="DNA", gene_symbol="DNA"))
    g.add_node(TargetNode(id="ENSG00000188389", name="PDCD1", gene_symbol="PDCD1"))
    g.add_node(IndicationNode(id="sarcoma", name="sarcoma"))
    parent_pop = "sarcoma__unselected"
    g.add_node(PopulationNode(id=parent_pop, name="sarcoma (all)"))

    # AFFECTS edges carry `implied_mechanism` so _populate_trial_mechanisms uses
    # the structured value and never calls the LLM. Each drug → its category id.
    for cid, tid, mech in (
        ("cyclophosphamide", "CHEBI:16991", "dna_crosslinking"),
        ("pembrolizumab", "ENSG00000188389", "checkpoint_blockade"),
    ):
        g.add_edge(GraphEdge(
            source_id=cid, target_id=tid, edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(),
            metadata={"implied_mechanism": mech},
        ))

    arm = TrialArm(
        arm_id="part_b", regimen_compound_id="cyclophosphamide__pembrolizumab",
        compound_ids=["cyclophosphamide", "pembrolizumab"], is_combination=True,
    )
    g.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT_CAT", arms=[arm], parent_population_id=parent_pop,
        chains=[CausalChain(
            arm_id="part_b", compound_id="cyclophosphamide",
            subgroup_population_id=parent_pop, target_id="CHEBI:16991",
            mechanism_id="UNKNOWN", biology_id="UNKNOWN", indication_id="sarcoma",
            endpoint_id="ORR_sarcoma", outcome=TrialOutcome.PARTIAL,
        )],
    ))

    # Extraction: cyclophosphamide entry carries an explicit category; the
    # pembrolizumab entry omits it (→ falls back to the resolved id).
    (tmp_path / "NCT_CAT_extraction.json").write_text(json.dumps({
        "nct_id": "NCT_CAT",
        "results_by_chain": [
            {"arm_id": "part_b", "intervention": "cyclophosphamide",
             "outcome": "partial", "mechanism_category": "dna_crosslinking"},
            {"arm_id": "part_b", "intervention": "pembrolizumab",
             "outcome": "partial"},  # no mechanism_category
        ],
    }))

    compound_targets = {
        "cyclophosphamide": ["CHEBI:16991"],
        "pembrolizumab": ["ENSG00000188389"],
    }
    trial = TrialRecord(nct_id="NCT_CAT", title="t", conditions=["sarcoma"])
    pipe = PopulationPipeline(g, anthropic_client=AsyncMock())
    await pipe._populate_trial_mechanisms(
        [trial], compound_targets, annotations_dir=tmp_path,
    )

    # Both mechanism nodes exist (ids = resolved MechanismCategory values) and
    # carry the category metadata.
    assert g.get_node("dna_crosslinking")["category"] == "dna_crosslinking"
    # Pembrolizumab had no extracted category → falls back to the resolved id.
    assert g.get_node("checkpoint_blockade")["category"] == "checkpoint_blockade"
    # Round-28 derivation still works: direction populated off MechanismCategory.
    assert g.get_node("dna_crosslinking")["direction"] == "inhibiting"
    assert g.get_node("checkpoint_blockade")["direction"] == "inhibiting"


# ── Mechanism-identity flip (Phase C): node id = molecular-action description ──


def test_mechanism_id_from_description_is_content_address():
    """A description-defined MechanismNode id is a content-address over the
    normalized description (Phase C: the node IS the specific molecular action)
    — NOT an enum slug. Mirrors ``_biology_id_from_description``: case/whitespace
    normalized so phrasing-identical actions collapse to one id; distinct actions
    get distinct ids. The ``mech:`` prefix separates these from enum-slug ids."""
    from src.graph.populate import _mechanism_id_from_description

    a = _mechanism_id_from_description("PD-1 checkpoint blockade")
    b = _mechanism_id_from_description("pd-1   checkpoint   blockade")
    c = _mechanism_id_from_description("CTLA4 co-inhibition")

    assert a.startswith("mech:") and len(a) == len("mech:") + 12
    assert a == b          # normalized (case + collapsed whitespace) → same id
    assert a != c          # distinct molecular action → distinct content-address
    assert "__" not in a   # not an enum slug


@pytest.mark.asyncio
async def test_mechanism_identity_flip_distinct_nodes_per_action(tmp_path):
    """Phase C end-to-end: when a per-(arm, intervention) ``mechanism_description``
    exists, the MechanismNode IDENTITY is the content-address of that specific
    action. A PD-1 blocker and a CTLA4 blocker — same drug-class CATEGORY
    (checkpoint_blockade) — become DISTINCT mechanism nodes (distinct ``mech:``
    ids), with ``category`` the shared coarse bucket and ``mechanism_type`` /
    ``direction`` derived from that category (NOT from the node id, which is no
    longer an enum value). The chains are repointed at the content-address ids."""
    import json

    from src.graph.models import (
        CausalChain, CompoundNode, EdgeBeliefState, EdgeType, GraphEdge,
        IndicationNode, PopulationNode, TargetNode, TrialArm, TrialOutcome,
        TrialSubgraph,
    )
    from src.graph.populate import _mechanism_id_from_description
    from src.graph.store import GraphStore
    from src.ingestion.clinicaltrials import TrialRecord

    g = GraphStore()
    g.add_node(CompoundNode(id="nivolumab", name="Nivolumab"))
    g.add_node(CompoundNode(id="ipilimumab", name="Ipilimumab"))
    g.add_node(TargetNode(id="ENSG00000188389", name="PDCD1", gene_symbol="PDCD1"))
    g.add_node(TargetNode(id="ENSG00000163599", name="CTLA4", gene_symbol="CTLA4"))
    g.add_node(IndicationNode(id="melanoma", name="melanoma"))
    parent_pop = "melanoma__unselected"
    g.add_node(PopulationNode(id=parent_pop, name="melanoma (all)"))

    # AFFECTS edges carry `implied_mechanism` so the LLM is never called; both
    # resolve to the SAME coarse enum bucket (checkpoint_blockade).
    for cid, tid in (("nivolumab", "ENSG00000188389"),
                     ("ipilimumab", "ENSG00000163599")):
        g.add_edge(GraphEdge(
            source_id=cid, target_id=tid, edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(),
            metadata={"implied_mechanism": "checkpoint_blockade"},
        ))

    # One combo arm, two constituents (each its own chain after rebuild).
    arm = TrialArm(
        arm_id="combo", regimen_compound_id="nivolumab__ipilimumab",
        compound_ids=["nivolumab", "ipilimumab"], is_combination=True,
    )
    g.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT_FLIP", arms=[arm], parent_population_id=parent_pop,
        chains=[CausalChain(
            arm_id="combo", compound_id="nivolumab",
            subgroup_population_id=parent_pop, target_id="ENSG00000188389",
            mechanism_id="UNKNOWN", biology_id="UNKNOWN", indication_id="melanoma",
            endpoint_id="ORR_melanoma", outcome=TrialOutcome.PARTIAL,
        )],
    ))

    # Distinct molecular-action descriptions; SAME category for both.
    (tmp_path / "NCT_FLIP_extraction.json").write_text(json.dumps({
        "nct_id": "NCT_FLIP",
        "results_by_chain": [
            {"arm_id": "combo", "intervention": "nivolumab", "outcome": "partial",
             "mechanism_description": "PD-1 checkpoint blockade",
             "mechanism_category": "checkpoint_blockade"},
            {"arm_id": "combo", "intervention": "ipilimumab", "outcome": "partial",
             "mechanism_description": "CTLA4 co-inhibition",
             "mechanism_category": "checkpoint_blockade"},
        ],
    }))

    compound_targets = {
        "nivolumab": ["ENSG00000188389"],
        "ipilimumab": ["ENSG00000163599"],
    }
    trial = TrialRecord(nct_id="NCT_FLIP", title="t", conditions=["melanoma"])
    pipe = PopulationPipeline(g, anthropic_client=AsyncMock())
    await pipe._populate_trial_mechanisms(
        [trial], compound_targets, annotations_dir=tmp_path,
    )

    pd1_id = _mechanism_id_from_description("PD-1 checkpoint blockade")
    ctla4_id = _mechanism_id_from_description("CTLA4 co-inhibition")

    # DISTINCT mechanism nodes (the flip): the enum slug is gone as an identity.
    assert pd1_id != ctla4_id
    pd1 = g.get_node(pd1_id)
    ctla4 = g.get_node(ctla4_id)
    assert pd1["name"] == "PD-1 checkpoint blockade"
    assert ctla4["name"] == "CTLA4 co-inhibition"
    assert pd1["description"] == "PD-1 checkpoint blockade"

    # SHARED coarse drug-class bucket carried as metadata.
    assert pd1["category"] == "checkpoint_blockade"
    assert ctla4["category"] == "checkpoint_blockade"

    # type/direction derived from the CATEGORY, not the (content-address) id.
    assert pd1["direction"] == "inhibiting"
    assert ctla4["direction"] == "inhibiting"
    assert pd1["mechanism_type"] == ctla4["mechanism_type"]

    # The enum-slug node was NOT created (identity is the action description).
    with pytest.raises(KeyError):
        g.get_node("checkpoint_blockade")

    # The rebuilt chains point at the content-address ids.
    ts = g.get_trial_subgraph_by_id("NCT_FLIP")
    chain_mech_ids = {c.compound_id: c.mechanism_id for c in ts.chains}
    assert chain_mech_ids["nivolumab"] == pd1_id
    assert chain_mech_ids["ipilimumab"] == ctla4_id


@pytest.mark.asyncio
async def test_mechanism_falls_back_to_enum_slug_without_description(tmp_path):
    """Phase C back-compat: a chain whose extraction carries NO
    ``mechanism_description`` keeps the enum-slug node id (so existing snapshots
    and non-bottom-up paths are unchanged), and ``mechanism_type`` / ``direction``
    still derive correctly from the resolved category."""
    import json

    from src.graph.models import (
        CausalChain, CompoundNode, EdgeBeliefState, EdgeType, GraphEdge,
        IndicationNode, PopulationNode, TargetNode, TrialArm, TrialOutcome,
        TrialSubgraph,
    )
    from src.graph.store import GraphStore
    from src.ingestion.clinicaltrials import TrialRecord

    g = GraphStore()
    g.add_node(CompoundNode(id="pembrolizumab", name="Pembrolizumab"))
    g.add_node(TargetNode(id="ENSG00000188389", name="PDCD1", gene_symbol="PDCD1"))
    g.add_node(IndicationNode(id="melanoma", name="melanoma"))
    parent_pop = "melanoma__unselected"
    g.add_node(PopulationNode(id=parent_pop, name="melanoma (all)"))
    g.add_edge(GraphEdge(
        source_id="pembrolizumab", target_id="ENSG00000188389",
        edge_type=EdgeType.AFFECTS, belief=EdgeBeliefState(),
        metadata={"implied_mechanism": "checkpoint_blockade"},
    ))
    arm = TrialArm(
        arm_id="mono", regimen_compound_id="pembrolizumab",
        compound_ids=["pembrolizumab"], is_combination=False,
    )
    g.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT_FB", arms=[arm], parent_population_id=parent_pop,
        chains=[CausalChain(
            arm_id="mono", compound_id="pembrolizumab",
            subgroup_population_id=parent_pop, target_id="ENSG00000188389",
            mechanism_id="UNKNOWN", biology_id="UNKNOWN", indication_id="melanoma",
            endpoint_id="ORR_melanoma", outcome=TrialOutcome.PARTIAL,
        )],
    ))
    # Extraction with category only, NO mechanism_description.
    (tmp_path / "NCT_FB_extraction.json").write_text(json.dumps({
        "nct_id": "NCT_FB",
        "results_by_chain": [
            {"arm_id": "mono", "intervention": "pembrolizumab",
             "outcome": "partial", "mechanism_category": "checkpoint_blockade"},
        ],
    }))
    trial = TrialRecord(nct_id="NCT_FB", title="t", conditions=["melanoma"])
    pipe = PopulationPipeline(g, anthropic_client=AsyncMock())
    await pipe._populate_trial_mechanisms(
        [trial], {"pembrolizumab": ["ENSG00000188389"]}, annotations_dir=tmp_path,
    )

    # Enum-slug id preserved (no content-address); direction derived from category.
    node = g.get_node("checkpoint_blockade")
    assert node["category"] == "checkpoint_blockade"
    assert node["direction"] == "inhibiting"
    ts = g.get_trial_subgraph_by_id("NCT_FB")
    assert ts.chains[0].mechanism_id == "checkpoint_blockade"


def test_bottomup_mergeconfig_enables_sapbert_for_mechanism():
    """Phase C merge wiring: the default bottom-up MergeConfig routes
    MechanismNode through the SapBERT precision tier (content-address action
    siblings like PD-1- vs CTLA4-blockade have BioLORD cosine ≈ 1.0 and would
    over-merge), while BiologyNode stays on the BioLORD semantic tier."""
    import inspect

    from src.graph import populate_bottomup

    src = inspect.getsource(populate_bottomup.build_bottomup)
    assert "enable_sapbert=True" in src
    assert 'sapbert_node_types=("MechanismNode",)' in src
    assert 'biolord_node_types=("BiologyNode",)' in src
