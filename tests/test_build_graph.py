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
