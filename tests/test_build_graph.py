"""Regression tests for the scripts/build_graph.py orchestrator.

These tests don't run the actual LLM calls — they mock the heavy steps
and verify the orchestrator's control-flow safety nets, especially the
round-16 abort-on-partial-classify check that prevents API credit
exhaustion from producing a silently-truncated snapshot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts import build_graph


@pytest.fixture
def fake_trials():
    """A small list of TrialRecord-like dummies; only nct_id is used by
    the orchestrator paths under test."""
    from src.ingestion.clinicaltrials import TrialRecord
    return [
        TrialRecord(
            nct_id=f"NCT000000{i:02d}", title=f"trial {i}", phase="2",
            status="COMPLETED", conditions=["melanoma"], interventions=[],
            primary_outcomes=[], enrollment=100, has_results=True,
            arm_groups=[],
        )
        for i in range(10)
    ]


class TestClassifySuccessRateAbort:
    """Round-16: the build must ABORT before attribution if the classifier
    succeeded on too few trials. Without this, an Anthropic credit
    exhaustion mid-run produces a snapshot that LOOKS complete but is
    silently truncated (mid-rebuild 5/19/2026: 73 of 145 classified before
    credit-balance 400 errors, build kept going, AUROC tanked from 0.667
    to 0.589 because half the chain evidence was missing)."""

    @pytest.mark.asyncio
    async def test_aborts_on_low_success_rate(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        # Redirect filesystem outputs to tmp_path.
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        # Stub every heavy step. populate_oncology / seed / attribute /
        # the GraphStore.import_snapshot at the end are all bypassed —
        # the only path we care about is fetch → extract → classify and
        # the abort check that fires immediately after.
        with (
            patch.object(build_graph, "fetch_trials", new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "extract_all", new=AsyncMock(return_value=fake_trials)),
            patch.object(
                build_graph, "classify_all",
                new=AsyncMock(return_value=3),  # 3 of 10 succeeded = 30%
            ),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=AsyncMock()),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)
            mock_pop.graph = type("FakeGraph", (), {
                "export_snapshot": lambda self, p: None,
            })()

            with pytest.raises(SystemExit, match="classify success rate"):
                await build_graph.main(
                    condition="melanoma", max_trials=10,
                    include_terminated=False, concurrency=2,
                    area="oncology", min_classify_success_rate=0.80,
                    # Bypass the round-20.5 subgraph-success check so
                    # this test exercises ONLY the classify-rate path.
                    allow_partial_subgraphs=True,
                )

    @pytest.mark.asyncio
    async def test_proceeds_when_success_rate_above_threshold(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        # Pre-create the snapshot file so the final import_snapshot call
        # doesn't blow up.
        (tmp_path / "exports" / "oncology_annotated.json").write_text(
            '{"graph": {"directed": true, "multigraph": true, "graph": {}, '
            '"nodes": [], "edges": []}, "trial_subgraphs": {}}'
        )

        attributor_mock = AsyncMock()
        with (
            patch.object(build_graph, "fetch_trials", new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "extract_all", new=AsyncMock(return_value=fake_trials)),
            patch.object(
                build_graph, "classify_all",
                new=AsyncMock(return_value=9),  # 9 of 10 succeeded = 90%
            ),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=attributor_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)
            mock_pop.graph = type("FakeGraph", (), {
                "export_snapshot": lambda self, p: None,
            })()
            # The final import_snapshot + stats path is the only place
            # the orchestrator touches a real GraphStore. We can let it
            # actually run since the file exists.

            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology", min_classify_success_rate=0.80,
                # Round-20.5 subgraph-rate check would fire here too
                # because the populate mock doesn't populate
                # graph.trial_subgraphs. Bypass since this test is
                # about the classify-rate path only.
                allow_partial_subgraphs=True,
            )
            # If we got here without SystemExit, the abort didn't fire.
            attributor_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_allow_partial_classify_overrides(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        """With --allow-partial-classify, the build proceeds even with a
        low success rate — for explicit debugging scenarios."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()
        (tmp_path / "exports" / "oncology_annotated.json").write_text(
            '{"graph": {"directed": true, "multigraph": true, "graph": {}, '
            '"nodes": [], "edges": []}, "trial_subgraphs": {}}'
        )

        attributor_mock = AsyncMock()
        with (
            patch.object(build_graph, "fetch_trials", new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "extract_all", new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "classify_all",
                         new=AsyncMock(return_value=3)),  # 30% — would normally abort
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=attributor_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)
            mock_pop.graph = type("FakeGraph", (), {
                "export_snapshot": lambda self, p: None,
            })()

            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                min_classify_success_rate=0.80,
                allow_partial_classify=True,
                allow_partial_subgraphs=True,
            )
            # Attribution should have been called despite the low rate.
            attributor_mock.assert_awaited_once()


# ── Round-19: incremental --base-snapshot mode ──────────────────────────


def _write_minimal_snapshot(path: Path, trial_ids: list[str]) -> None:
    """Drop a snapshot file containing the named trials' subgraphs so
    the incremental-mode tests can drive build_graph.main against
    something `import_snapshot` will accept."""
    from src.graph.models import (
        BiologyNode,
        CausalChain,
        CompoundNode,
        EndpointNode,
        EndpointType,
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

    g = GraphStore()
    g.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
    g.add_node(TargetNode(id="ENSG00000188389", name="PD-1", gene_symbol="PD-1"))
    g.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade",
                             mechanism_type=MechanismType.ANTAGONISM))
    g.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 signaling"))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(
        id="PFS_melanoma", name="PFS",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="melanoma__unselected", name="All"))

    for tid in trial_ids:
        arm = TrialArm(arm_id="solo", compound_ids=["nivolumab"],
                       regimen_compound_id="nivolumab")
        chain = CausalChain(
            arm_id="solo", compound_id="nivolumab",
            subgroup_population_id="melanoma__unselected",
            target_id="ENSG00000188389", mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        )
        ts = TrialSubgraph(
            trial_id=tid, phase="3", arms=[arm], chains=[chain],
            parent_population_id="melanoma__unselected",
        )
        g.set_trial_subgraph(ts)
        g.applied_attribution_trial_ids.add(tid)
    g.export_snapshot(str(path))


class TestIncrementalBuildValidation:
    """Round-19: validation errors must fire BEFORE any wipe / fetch /
    network call so a typo can't nuke data/annotations."""

    @pytest.mark.asyncio
    async def test_base_snapshot_without_add_flags_aborts(self, tmp_path):
        base = tmp_path / "base.json"
        _write_minimal_snapshot(base, ["NCT00000001"])
        with pytest.raises(SystemExit, match="requires --add-trials"):
            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                base_snapshot=str(base),
            )

    @pytest.mark.asyncio
    async def test_add_trials_without_base_snapshot_aborts(self, tmp_path):
        with pytest.raises(SystemExit, match="require --base-snapshot"):
            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                add_trials=["NCT00000001"],
            )

    @pytest.mark.asyncio
    async def test_corpus_and_base_snapshot_mutually_exclusive(self, tmp_path):
        base = tmp_path / "base.json"
        _write_minimal_snapshot(base, ["NCT00000001"])
        with pytest.raises(SystemExit, match="cannot be combined"):
            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                base_snapshot=str(base),
                corpus="melanoma_50",
                add_trials=["NCT00000002"],
            )

    @pytest.mark.asyncio
    async def test_base_snapshot_missing_file_aborts(self, tmp_path):
        with pytest.raises(SystemExit, match="--base-snapshot file not found"):
            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                base_snapshot=str(tmp_path / "nonexistent.json"),
                add_trials=["NCT00000001"],
            )


class TestIncrementalBuildOrchestration:
    @pytest.fixture(autouse=True)
    def _stub_step3b5_descriptions(self):
        # Step 3b.5 (generate_node_descriptions) makes a real Haiku call. These
        # orchestration tests drive build_graph.main with a MagicMock client and
        # a non-empty base graph, so the awaited client call would hit the
        # MagicMock. Stub it. (Sibling tests with an empty graph never reach it —
        # no nodes to describe.)
        with patch("src.graph.descriptions.generate_node_descriptions",
                   new=AsyncMock(return_value=0)):
            yield

    @pytest.mark.asyncio
    async def test_assemble_flag_runs_step5(self, tmp_path, monkeypatch):
        """--assemble runs Step 5 (box/is-a geometry + (s,t) field) after
        attribution, against the freshly-attributed annotated snapshot. The
        skip-when-off path is covered by the sibling tests: they never patch
        assemble_geometry yet stay fast, so Step 5 is correctly not entered."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        base = tmp_path / "base.json"
        _write_minimal_snapshot(base, ["NCT00000001"])
        (tmp_path / "exports" / "oncology_annotated.json").write_text(base.read_text())

        from src.ingestion.clinicaltrials import TrialRecord
        new_trial = TrialRecord(
            nct_id="NCT00000002", title="new trial", phase="2",
            status="COMPLETED", conditions=["melanoma"], interventions=[],
            primary_outcomes=[], enrollment=100, has_results=True, arm_groups=[],
        )
        asm_mock = MagicMock(return_value={
            "boxes": 5, "subtype_before": 0, "subtype_after": 2,
            "subtype_added": 2, "private_root": "/tmp/priv",
        })
        fld_mock = MagicMock(return_value={
            "edges_localized": 7, "anchors_total": 20, "out": "/tmp/priv/f.json",
        })
        with (
            patch.object(build_graph, "fetch_trials_by_ids",
                         new=AsyncMock(return_value=[new_trial])),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "extract_all",
                         new=AsyncMock(return_value=[new_trial])),
            patch.object(build_graph, "classify_all", new=AsyncMock(return_value=1)),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=AsyncMock()),
            patch("scripts.assemble_v2.assemble_geometry", new=asm_mock),
            patch("scripts.materialize_belief_field.materialize_field", new=fld_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)
            await build_graph.main(
                condition="melanoma", max_trials=10, include_terminated=False,
                concurrency=2, area="oncology", base_snapshot=str(base),
                add_trials=["NCT00000002"], allow_partial_subgraphs=True,
                assemble=True,
            )

        asm_mock.assert_called_once()
        fld_mock.assert_called_once()
        # both run against the freshly-attributed annotated snapshot
        assert asm_mock.call_args.args[0].endswith("oncology_annotated.json")
        assert fld_mock.call_args.args[0].endswith("oncology_annotated.json")

    @pytest.mark.asyncio
    async def test_base_snapshot_skips_wipe(self, tmp_path, monkeypatch):
        """The Step 0 wipe must NOT fire when --base-snapshot is set —
        otherwise the existing annotated snapshot and per-trial
        annotation caches would be destroyed."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        base = tmp_path / "base.json"
        _write_minimal_snapshot(base, ["NCT00000001"])

        # A second NCT not in the base — gets fetched + processed.
        from src.ingestion.clinicaltrials import TrialRecord
        new_trial = TrialRecord(
            nct_id="NCT00000002", title="new trial", phase="2",
            status="COMPLETED", conditions=["melanoma"], interventions=[],
            primary_outcomes=[], enrollment=100, has_results=True,
            arm_groups=[],
        )

        # Pre-create the eventual annotated output so the final
        # import_snapshot doesn't blow up.
        (tmp_path / "exports" / "oncology_annotated.json").write_text(
            base.read_text()
        )

        wipe_mock = AsyncMock()  # AsyncMock is fine for a sync stub here too.
        attributor_mock = AsyncMock()
        with (
            patch.object(build_graph, "fetch_trials_by_ids",
                         new=AsyncMock(return_value=[new_trial])),
            patch.object(build_graph, "wipe_outputs", new=wipe_mock),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "extract_all",
                         new=AsyncMock(return_value=[new_trial])),
            patch.object(build_graph, "classify_all",
                         new=AsyncMock(return_value=1)),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=attributor_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)

            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                base_snapshot=str(base),
                add_trials=["NCT00000002"],
                # populate is mocked so trial_subgraphs stays empty —
                # bypass the round-20.5 subgraph-rate guard.
                allow_partial_subgraphs=True,
            )

        wipe_mock.assert_not_called()
        attributor_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_already_present_trials_filtered_before_populate(
        self, tmp_path, monkeypatch,
    ):
        """If --add-trials lists an NCT already in the base snapshot,
        the orchestrator must filter it out before handing the list to
        populate_oncology — otherwise the populator runs LLM-cost
        canonicalization on already-known trials."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        base = tmp_path / "base.json"
        _write_minimal_snapshot(base, ["NCT00000001"])

        from src.ingestion.clinicaltrials import TrialRecord
        # Both an already-present trial AND a new one in the add list.
        existing = TrialRecord(
            nct_id="NCT00000001", title="existing", phase="2",
            status="COMPLETED", conditions=["melanoma"], interventions=[],
            primary_outcomes=[], enrollment=100, has_results=True,
            arm_groups=[],
        )
        new_trial = TrialRecord(
            nct_id="NCT00000002", title="new", phase="2",
            status="COMPLETED", conditions=["melanoma"], interventions=[],
            primary_outcomes=[], enrollment=100, has_results=True,
            arm_groups=[],
        )
        (tmp_path / "exports" / "oncology_annotated.json").write_text(
            base.read_text()
        )

        populate_mock = AsyncMock(return_value=None)
        with (
            patch.object(build_graph, "fetch_trials_by_ids",
                         new=AsyncMock(return_value=[existing, new_trial])),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "extract_all",
                         new=AsyncMock(return_value=[new_trial])),
            patch.object(build_graph, "classify_all",
                         new=AsyncMock(return_value=1)),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=AsyncMock()),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = populate_mock

            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                base_snapshot=str(base),
                add_trials=["NCT00000001", "NCT00000002"],
                # populate is mocked so trial_subgraphs stays empty —
                # bypass the round-20.5 subgraph-rate guard.
                allow_partial_subgraphs=True,
            )

        # The populator should have been called with ONLY the new trial.
        populate_mock.assert_awaited_once()
        call_kwargs = populate_mock.await_args.kwargs
        passed_trials = call_kwargs["trials"]
        passed_ids = [t.nct_id for t in passed_trials]
        assert passed_ids == ["NCT00000002"]

    @pytest.mark.asyncio
    async def test_subgraph_success_threshold_aborts_on_silent_drops(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        """Round-20.5: if build_trial_subgraphs silently drops too many
        trials, the orchestrator must abort BEFORE extraction so we
        don't burn LLM tokens classifying trials that'll never make it
        into the graph. Mirrors the round-16 classify-success-rate
        check."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        # Empty graph (no trial_subgraphs) after populate — simulates
        # the worst case where every trial silently dropped.
        from src.graph.store import GraphStore
        empty_graph = GraphStore()
        populate_mock = AsyncMock(return_value=None)

        with (
            patch.object(build_graph, "fetch_trials",
                         new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "GraphStore", return_value=empty_graph),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = populate_mock

            with pytest.raises(SystemExit, match="trial subgraph build success rate"):
                await build_graph.main(
                    condition="melanoma", max_trials=10,
                    include_terminated=False, concurrency=2,
                    area="oncology",
                    min_subgraph_success_rate=0.90,
                )

    @pytest.mark.asyncio
    async def test_legitimate_drops_excluded_from_success_rate(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        """Drops whose reason is in LEGITIMATE_DROP_REASONS (non-
        therapeutic trials, fundamentally empty CT.gov records) get
        excluded from the success-rate denominator. If 10 trials are
        fetched and 8 drop as non-therapeutic, the rate is computed
        over the 2 ELIGIBLE trials, not the raw 10."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()
        # Pre-create the final annotated path so the post-build
        # import_snapshot doesn't blow up.
        (tmp_path / "exports" / "oncology_annotated.json").write_text(
            '{"graph": {"directed": true, "multigraph": true, "graph": {}, '
            '"nodes": [], "edges": []}, "trial_subgraphs": {}}'
        )

        # Simulate a populator that built 2 trial subgraphs and dropped
        # 8 as non-therapeutic. We write the drop log directly because
        # the populator is mocked.
        import json as _json
        from src.graph import populate as pop_mod
        drop_log = tmp_path / "dropped_trial_subgraphs.jsonl"
        monkeypatch.setattr(
            pop_mod, "_DROPPED_TRIAL_SUBGRAPHS_LOG", drop_log,
        )
        with drop_log.open("w") as fh:
            for i in range(8):
                fh.write(_json.dumps({
                    "nct_id": fake_trials[i].nct_id,
                    "reason": "no_arms_filtered_by_diagnostic",
                    "details": {"intervention_names": ["behavioral"]},
                }) + "\n")

        # Graph contains the last 2 fake trials' subgraphs.
        from src.graph.models import (
            CausalChain, EndpointNode, EndpointType, IndicationNode,
            PopulationNode, RegulatoryStatus, TrialArm, TrialOutcome,
            TrialSubgraph,
        )
        from src.graph.store import GraphStore
        graph = GraphStore()
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(PopulationNode(id="melanoma__unselected", name="all"))
        graph.add_node(EndpointNode(
            id="ep", name="OS",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        for trial in fake_trials[8:]:
            arm = TrialArm(arm_id="a", compound_ids=["x"], regimen_compound_id="x")
            chain = CausalChain(
                arm_id="a", compound_id="x",
                subgroup_population_id="melanoma__unselected",
                target_id="UNKNOWN", mechanism_id="UNKNOWN",
                biology_id="UNKNOWN", indication_id="melanoma",
                endpoint_id="ep", outcome=TrialOutcome.UNKNOWN,
            )
            graph.set_trial_subgraph(TrialSubgraph(
                trial_id=trial.nct_id, phase="3", arms=[arm],
                chains=[chain], parent_population_id="melanoma__unselected",
            ))

        attributor_mock = AsyncMock()
        with (
            patch.object(build_graph, "fetch_trials",
                         new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "GraphStore", return_value=graph),
            patch.object(build_graph, "extract_all",
                         new=AsyncMock(return_value=fake_trials[8:])),
            patch.object(build_graph, "classify_all",
                         new=AsyncMock(return_value=2)),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=attributor_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)
            # 2 built of 10 input = 20% raw, but 8 are legitimate
            # drops so eligible_count = 2 and rate = 2/2 = 100%.
            # The 75% threshold should PASS.
            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                min_subgraph_success_rate=0.75,
            )
        # Build proceeded past the subgraph guard → attribution ran.
        attributor_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bug_indicating_drops_still_count_against_rate(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        """Drops with bug-indicating reasons (no_indication, no_endpoint,
        no_arms_empty_arm_groups, no_arms) ARE counted against the
        rate — those represent real silent loss from populator gaps."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        import json as _json
        from src.graph import populate as pop_mod
        from src.graph.store import GraphStore
        drop_log = tmp_path / "dropped_trial_subgraphs.jsonl"
        monkeypatch.setattr(
            pop_mod, "_DROPPED_TRIAL_SUBGRAPHS_LOG", drop_log,
        )
        with drop_log.open("w") as fh:
            for i in range(8):
                fh.write(_json.dumps({
                    "nct_id": fake_trials[i].nct_id,
                    # Bug-indicating reason — does NOT get excluded.
                    "reason": "no_endpoint",
                    "details": {},
                }) + "\n")

        empty_graph = GraphStore()
        with (
            patch.object(build_graph, "fetch_trials",
                         new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "GraphStore", return_value=empty_graph),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)
            # 0 built, 8 bug-indicating drops, 2 unknown drops →
            # eligible_count = 10 - 0 (no legitimate) = 10; rate = 0%.
            with pytest.raises(SystemExit, match="success rate"):
                await build_graph.main(
                    condition="melanoma", max_trials=10,
                    include_terminated=False, concurrency=2,
                    area="oncology",
                    min_subgraph_success_rate=0.75,
                )

    @pytest.mark.asyncio
    async def test_allow_partial_subgraphs_overrides_silent_drop_abort(
        self, fake_trials, tmp_path, monkeypatch,
    ):
        """--allow-partial-subgraphs lets the build proceed even when
        many trials dropped — for explicit debugging."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()
        (tmp_path / "exports" / "oncology_annotated.json").write_text(
            '{"graph": {"directed": true, "multigraph": true, "graph": {}, '
            '"nodes": [], "edges": []}, "trial_subgraphs": {}}'
        )

        from src.graph.store import GraphStore
        empty_graph = GraphStore()
        attributor_mock = AsyncMock()
        with (
            patch.object(build_graph, "fetch_trials",
                         new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "wipe_outputs"),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "Extractor"),
            patch.object(build_graph, "Classifier"),
            patch.object(build_graph, "GraphStore", return_value=empty_graph),
            patch.object(build_graph, "extract_all",
                         new=AsyncMock(return_value=fake_trials)),
            patch.object(build_graph, "classify_all",
                         new=AsyncMock(return_value=10)),
            patch.object(build_graph, "seed_responds_differently_from_extractions",
                         new=AsyncMock(return_value=(0, 0))),
            patch.object(build_graph, "attributor_main", new=attributor_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = AsyncMock(return_value=None)

            # Should NOT raise SystemExit despite 0% subgraph success.
            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                min_subgraph_success_rate=0.90,
                allow_partial_subgraphs=True,
            )

    @pytest.mark.asyncio
    async def test_all_requested_already_present_returns_early(
        self, tmp_path, monkeypatch,
    ):
        """If every requested NCT is already in the base, there's
        nothing to do and the orchestrator should return cleanly
        without calling populate / extract / classify / attribute."""
        monkeypatch.setattr(build_graph, "EXPORTS_DIR", tmp_path / "exports")
        monkeypatch.setattr(build_graph, "ANNOTATIONS_DIR", tmp_path / "annotations")
        monkeypatch.setattr(build_graph, "CORPORA_DIR", tmp_path / "corpora")
        (tmp_path / "exports").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "corpora").mkdir()

        base = tmp_path / "base.json"
        _write_minimal_snapshot(base, ["NCT00000001"])

        from src.ingestion.clinicaltrials import TrialRecord
        existing = TrialRecord(
            nct_id="NCT00000001", title="existing", phase="2",
            status="COMPLETED", conditions=["melanoma"], interventions=[],
            primary_outcomes=[], enrollment=100, has_results=True,
            arm_groups=[],
        )

        populate_mock = AsyncMock()
        attributor_mock = AsyncMock()
        with (
            patch.object(build_graph, "fetch_trials_by_ids",
                         new=AsyncMock(return_value=[existing])),
            patch.object(build_graph, "PopulationPipeline") as MockPop,
            patch.object(build_graph, "attributor_main", new=attributor_mock),
            patch("anthropic.AsyncAnthropic"),
        ):
            mock_pop = MockPop.return_value
            mock_pop.populate_oncology = populate_mock

            await build_graph.main(
                condition="melanoma", max_trials=10,
                include_terminated=False, concurrency=2,
                area="oncology",
                base_snapshot=str(base),
                add_trials=["NCT00000001"],
            )

        populate_mock.assert_not_awaited()
        attributor_mock.assert_not_awaited()
