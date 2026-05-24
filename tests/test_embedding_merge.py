"""Tests for the A.4 flag-gated embedding merger (biology_merge.py).

Offline — BioLORD is injected as a stub. Covers pair detection at threshold,
union of crosswalk classes bridged by a pair, singleton handling, and the flag.
"""

from __future__ import annotations

from src.graph.biology_merge import (
    augment_classes_with_pairs,
    embedding_merge_enabled,
    embedding_merge_pairs,
)
from src.graph.models import BiologyNode
from src.graph.store import GraphStore


def _embed(text: str):
    # VEGF-ish descriptions → one direction; apoptosis → orthogonal.
    return [1.0, 0.0] if "vegf" in text.lower() else [0.0, 1.0]


def test_embedding_merge_pairs_finds_semantic_twins():
    g = GraphStore()
    g.add_node(BiologyNode(id="R-HSA-194138", name="Signaling by VEGF",
                           description="VEGF receptor signaling cascade"))
    g.add_node(BiologyNode(id="GO:0048010", name="VEGF signaling pathway",
                           description="signaling by VEGF"))
    g.add_node(BiologyNode(id="R-HSA-109581", name="Apoptosis",
                           description="programmed cell death"))
    pairs = embedding_merge_pairs(g, embed_fn=_embed, threshold=0.92)
    flat = {frozenset(p) for p in pairs}
    assert frozenset({"R-HSA-194138", "GO:0048010"}) in flat   # semantic twins
    assert all("R-HSA-109581" not in p for p in pairs)          # apoptosis stays out


def test_embedding_merge_pairs_respects_threshold():
    g = GraphStore()
    g.add_node(BiologyNode(id="b1", name="x", description="VEGF a"))
    g.add_node(BiologyNode(id="b2", name="y", description="apoptosis b"))
    # orthogonal vectors → cosine 0 → no pair even at a low threshold
    assert embedding_merge_pairs(g, embed_fn=_embed, threshold=0.5) == []


def test_augment_classes_unions_bridged_pairs():
    out = augment_classes_with_pairs([{"a", "b"}, {"c", "e"}], [("b", "c")])
    big = next(g for g in out if "a" in g)
    assert big == {"a", "b", "c", "e"}


def test_augment_drops_singletons_keeps_merges():
    assert augment_classes_with_pairs([{"z"}], []) == []          # singleton dropped
    assert augment_classes_with_pairs([], [("x", "y")]) == [{"x", "y"}]


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("EROOM_EMBEDDING_MERGE", raising=False)
    assert embedding_merge_enabled() is False
    monkeypatch.setenv("EROOM_EMBEDDING_MERGE", "1")
    assert embedding_merge_enabled() is True
