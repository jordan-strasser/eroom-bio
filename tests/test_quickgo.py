"""Tests for the QuickGO client.

These tests don't hit the live QuickGO API. The HTTP-talking ``_fetch_terms``
method is monkeypatched per-test so we exercise the cache + parsing layer
without flakiness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.quickgo import GOTerm, QuickGOClient


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path) -> QuickGOClient:
    return QuickGOClient(cache_dir=tmp_path)


# ── Empty / error cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_uniprot_returns_empty(client: QuickGOClient):
    result = await client.get_terms_for_uniprot("")
    assert result == []


@pytest.mark.asyncio
async def test_no_annotations_returns_empty(
    client: QuickGOClient, monkeypatch: pytest.MonkeyPatch
):
    async def fake_fetch(uniprot_acc, aspect):
        return []
    monkeypatch.setattr(client, "_fetch_terms", fake_fetch)

    result = await client.get_terms_for_uniprot("Q00000")
    assert result == []


# ── Caching ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_skips_fetch(
    client: QuickGOClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Once a cache file is written, subsequent calls don't hit the network."""
    cache_path = tmp_path / "quickgo" / "Q96SW2.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([
        {"stable_id": "GO:0016567", "display_name": "protein ubiquitination",
         "aspect": "biological_process"},
    ]))

    async def boom(uniprot_acc, aspect):
        pytest.fail("Cache should have prevented fetch")
    monkeypatch.setattr(client, "_fetch_terms", boom)

    result = await client.get_terms_for_uniprot("Q96SW2")
    assert len(result) == 1
    assert result[0].stable_id == "GO:0016567"
    assert result[0].display_name == "protein ubiquitination"


@pytest.mark.asyncio
async def test_fetched_terms_are_cached(
    client: QuickGOClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """First call writes the cache; second call reads from disk."""
    fetched_terms = [
        GOTerm(stable_id="GO:0016567", display_name="protein ubiquitination"),
        GOTerm(
            stable_id="GO:0043161",
            display_name="proteasome-mediated ubiquitin-dependent protein catabolic process",
        ),
    ]

    fetch_count = {"n": 0}

    async def fake_fetch(uniprot_acc, aspect):
        fetch_count["n"] += 1
        return fetched_terms
    monkeypatch.setattr(client, "_fetch_terms", fake_fetch)

    first = await client.get_terms_for_uniprot("Q96SW2")
    second = await client.get_terms_for_uniprot("Q96SW2")
    assert fetch_count["n"] == 1
    assert [t.stable_id for t in first] == [t.stable_id for t in second]
    assert (tmp_path / "quickgo" / "Q96SW2.json").exists()


# ── Parsing / shape ──────────────────────────────────────────────────────


def test_go_term_satisfies_biology_candidate_protocol():
    """GOTerm must expose ``stable_id`` and ``display_name`` so the same
    pathway_ranker.rerank_pathways handles both Reactome and GO inputs."""
    from src.graph.pathway_ranker import rerank_pathways

    terms = [
        GOTerm(stable_id="GO:0001", display_name="generic process"),
        GOTerm(stable_id="GO:0002", display_name="protein ubiquitination"),
    ]
    ranked = rerank_pathways(
        terms,
        mechanism_name="protein_degradation",
    )
    # The ubiquitination term should rank first via the mechanism vocab.
    assert ranked[0].stable_id == "GO:0002"
