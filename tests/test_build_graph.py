"""Regression tests for the scripts/build_graph.py orchestrator.

These tests don't run the actual LLM calls — they mock the heavy steps
and verify the orchestrator's control-flow safety nets, especially the
round-16 abort-on-partial-classify check that prevents API credit
exhaustion from producing a silently-truncated snapshot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
            )

        # The populator should have been called with ONLY the new trial.
        populate_mock.assert_awaited_once()
        call_kwargs = populate_mock.await_args.kwargs
        passed_trials = call_kwargs["trials"]
        passed_ids = [t.nct_id for t in passed_trials]
        assert passed_ids == ["NCT00000002"]

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
