"""Tests for the LLM compound→target inference resolver
(PopulationPipeline._infer_missing_compound_targets)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.graph.models import CompoundNode, EdgeType
from src.graph.populate import PopulationPipeline
from src.graph.store import GraphStore


def _pipeline(tmp_path, *, llm_reply: str, ot_target):
    """A pipeline with the LLM call + OT client mocked and one unresolved
    compound node in the graph. ``llm_reply`` is the 'GENE | class' string the
    inference returns; ``ot_target`` is what OT search_target yields (None ⇒
    the gene doesn't validate)."""
    graph = GraphStore()
    graph.add_node(CompoundNode(id="fotemustine", name="fotemustine"))
    pipe = PopulationPipeline(graph, anthropic_client=AsyncMock(), cache_dir=tmp_path)
    pipe._diagnostic_compound_ids = set()
    pipe._ot_client = SimpleNamespace(search_target=AsyncMock(return_value=ot_target))

    async def _fake_infer(name, indication):
        return llm_reply

    pipe._llm_infer_compound_target = _fake_infer  # type: ignore[method-assign]
    return pipe, graph


_PARP1 = {"target_id": "ENSG00000143799", "approved_symbol": "PARP1", "name": "PARP1"}


@pytest.mark.asyncio
async def test_infers_validated_target_and_adds_edge(tmp_path):
    pipe, graph = _pipeline(tmp_path, llm_reply="PARP1 | therapeutic", ot_target=_PARP1)
    added, new_targets = await pipe._infer_missing_compound_targets([], {})
    assert added == 1
    assert new_targets == {"fotemustine": ["ENSG00000143799"]}
    assert graph._graph.has_edge("fotemustine", "ENSG00000143799", key=EdgeType.AFFECTS.value)
    # belief is seeded by the LLM-inference evidence tier
    belief = graph.get_edge_belief("fotemustine", "ENSG00000143799", EdgeType.AFFECTS)
    assert belief.evidence[0].source_type.value == "database_llm_inference"
    assert graph.get_node("ENSG00000143799")["gene_symbol"] == "PARP1"


@pytest.mark.asyncio
async def test_intervention_class_gate_skips_diagnostic(tmp_path):
    pipe, graph = _pipeline(tmp_path, llm_reply="SLC5A5 | diagnostic", ot_target=_PARP1)
    added, new_targets = await pipe._infer_missing_compound_targets([], {})
    assert added == 0 and new_targets == {}


@pytest.mark.asyncio
async def test_unknown_gene_is_skipped(tmp_path):
    pipe, graph = _pipeline(tmp_path, llm_reply="UNKNOWN | therapeutic", ot_target=_PARP1)
    added, new_targets = await pipe._infer_missing_compound_targets([], {})
    assert added == 0 and new_targets == {}


@pytest.mark.asyncio
async def test_unvalidated_gene_is_skipped(tmp_path):
    # LLM proposes a gene but OT can't resolve it to a real Ensembl id → drop
    pipe, graph = _pipeline(tmp_path, llm_reply="NOTAGENE | therapeutic", ot_target=None)
    added, new_targets = await pipe._infer_missing_compound_targets([], {})
    assert added == 0 and new_targets == {}


@pytest.mark.asyncio
async def test_already_resolved_compound_is_not_reinferred(tmp_path):
    pipe, graph = _pipeline(tmp_path, llm_reply="PARP1 | therapeutic", ot_target=_PARP1)
    # fotemustine already has a target → skipped entirely
    added, new_targets = await pipe._infer_missing_compound_targets(
        [], {"fotemustine": ["ENSG00000000001"]}
    )
    assert added == 0 and new_targets == {}
