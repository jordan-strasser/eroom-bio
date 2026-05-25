"""Tests for node description + name_id generation (v2 / Q4).

The LLM call is stubbed (no API); covers name_id normalization, the
generate-only-missing behavior, caching, and the all-node coverage goal.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from src.graph.descriptions import (
    assign_name_ids,
    generate_node_descriptions,
    normalize_name_id,
)
from src.graph.models import BiologyNode, IndicationNode, TargetNode
from src.graph.store import GraphStore


class _StubAnthropic:
    """Fake AsyncAnthropic: messages.create returns a canned description and
    counts calls so we can assert caching / skip-existing."""

    def __init__(self):
        self.calls = 0
        self.messages = types.SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls += 1
        text = f"A precise description ({self.calls})."
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


def test_normalize_name_id():
    assert normalize_name_id("Co-inhibition by PD-1") == "co inhibition by pd 1"
    assert normalize_name_id("  EGFR  ") == "egfr"


def test_assign_name_ids_sets_all():
    g = GraphStore()
    g.add_node(TargetNode(id="ENSG1", gene_symbol="EGFR", name="EGFR"))
    g.add_node(IndicationNode(id="nsclc", name="NSCLC"))
    n = assign_name_ids(g)
    assert n == 2
    assert g._graph.nodes["ENSG1"]["name_id"] == "egfr"  # noqa: SLF001
    assert g._graph.nodes["nsclc"]["name_id"] == "nsclc"  # noqa: SLF001


def test_generate_fills_only_missing_and_caches(tmp_path):
    g = GraphStore()
    g.add_node(TargetNode(id="ENSG1", gene_symbol="EGFR", name="EGFR"))          # no desc
    g.add_node(IndicationNode(id="nsclc", name="NSCLC"))                          # no desc
    g.add_node(BiologyNode(id="R-1", name="EGFR signaling", description="exists"))  # has desc, not a target type anyway

    client = _StubAnthropic()
    cache = tmp_path / "desc.json"
    n = asyncio.run(generate_node_descriptions(
        g, client, node_types=("TargetNode", "IndicationNode"),
        concurrency=2, cache_path=cache,
    ))
    assert n == 2
    assert client.calls == 2
    assert g._graph.nodes["ENSG1"]["description"]   # noqa: SLF001
    assert g._graph.nodes["nsclc"]["description"]   # noqa: SLF001

    # second run: cache hit → no new API calls, descriptions unchanged
    client2 = _StubAnthropic()
    g2 = GraphStore()
    g2.add_node(TargetNode(id="ENSG1", gene_symbol="EGFR", name="EGFR"))
    g2.add_node(IndicationNode(id="nsclc", name="NSCLC"))
    n2 = asyncio.run(generate_node_descriptions(
        g2, client2, node_types=("TargetNode", "IndicationNode"),
        concurrency=2, cache_path=cache,
    ))
    assert n2 == 2
    assert client2.calls == 0  # fully served from cache


def test_generate_skips_nodes_with_existing_description(tmp_path):
    g = GraphStore()
    g.add_node(TargetNode(id="ENSG1", gene_symbol="EGFR", name="EGFR",
                          metadata={}))
    g._graph.nodes["ENSG1"]["description"] = "already here"  # noqa: SLF001
    client = _StubAnthropic()
    n = asyncio.run(generate_node_descriptions(
        g, client, node_types=("TargetNode",), cache_path=tmp_path / "d.json",
    ))
    assert n == 0 and client.calls == 0
