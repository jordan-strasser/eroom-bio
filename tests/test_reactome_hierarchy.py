"""Tests for the Reactome ancestors fetcher (A.2 biology supervision).

httpx is mocked — CI must not depend on reactome.org. Verifies parsing of the
nested ancestor-paths shape, self-exclusion, caching, and graceful failure.
"""

from __future__ import annotations

import httpx
import pytest

from src.graph import reactome_hierarchy as rh


class _FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        # Reactome returns a list of ancestor *paths*; each path is leaf→root.
        return [[
            {"stId": "R-HSA-194138", "displayName": "Signaling by VEGF"},
            {"stId": "R-HSA-9006934", "displayName": "Signaling by RTKs"},
            {"stId": "R-HSA-162582", "displayName": "Signal Transduction"},
        ]]


def test_fetch_ancestors_parses_excludes_self_and_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=15.0):
        calls["n"] += 1
        return _FakeResp()

    monkeypatch.setattr(httpx, "get", fake_get)
    cache = tmp_path / "anc.json"

    out = rh.fetch_reactome_ancestors("R-HSA-194138", cache_path=cache)
    assert set(out) == {"R-HSA-9006934", "R-HSA-162582"}  # self excluded
    assert cache.exists()

    # second call served from cache — no second HTTP hit
    rh.fetch_reactome_ancestors("R-HSA-194138", cache_path=cache)
    assert calls["n"] == 1


def test_non_reactome_id_short_circuits(tmp_path, monkeypatch):
    def boom(url, timeout=15.0):
        raise AssertionError("should not hit network for a non-Reactome id")

    monkeypatch.setattr(httpx, "get", boom)
    assert rh.fetch_reactome_ancestors("GO:0048010", cache_path=tmp_path / "a.json") == []


def test_network_failure_returns_empty(tmp_path, monkeypatch):
    def boom(url, timeout=15.0):
        raise httpx.ConnectError("reactome down")

    monkeypatch.setattr(httpx, "get", boom)
    assert rh.fetch_reactome_ancestors("R-HSA-1", cache_path=tmp_path / "a.json") == []
