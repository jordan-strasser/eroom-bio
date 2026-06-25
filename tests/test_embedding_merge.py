"""Tests for the A.4 flag-gated embedding merger (biology_merge.py).

Offline — BioLORD is injected as a stub. Covers pair detection at threshold,
union of crosswalk classes bridged by a pair, singleton handling, and the flag.
"""

from __future__ import annotations

import numpy as np

from src.graph.biology_merge import (
    augment_classes_with_pairs,
    box_merge_pairs,
    embedding_merge_pairs,
)
from src.graph.box_embeddings import Box
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


def test_box_merge_merges_coincident_not_containment():
    # The box merger's precision win: coincident boxes merge; a parent/child
    # containment pair (which cosine would wrongly merge) is NOT merged.
    g = GraphStore()
    for bid in ["a", "b", "parent", "child"]:
        g.add_node(BiologyNode(id=bid, name=bid))
    boxes = {
        "a": Box.cube(np.array([-5.0, -5.0]), 0.1),
        "b": Box.cube(np.array([-5.0, -5.0]), 0.1),       # coincident with a
        "parent": Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        "child": Box(np.array([0.5, 0.5]), np.array([0.7, 0.7])),  # inside parent
    }
    pairs = {frozenset(p) for p in box_merge_pairs(g, boxes)}
    assert frozenset({"a", "b"}) in pairs              # coincident -> merge
    assert frozenset({"parent", "child"}) not in pairs  # containment -> NOT merge
