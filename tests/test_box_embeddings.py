"""Tests for box embeddings (manifold 1, A.2).

Pure numpy + synthetic geometry — no model download. Covers the box ops, the
ontology-bootstrapped fit (parent ⊇ child by construction), the five gold-pair
relation labels, the population-hierarchy supervision, and the private-only
persistence boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.graph.box_embeddings import (
    Box,
    apply_boxes_to_graph,
    biology_parent_child_pairs,
    contains,
    containment_fraction,
    fit_boxes,
    fit_graph_boxes,
    load_boxes,
    population_parent_child_pairs,
    relation,
    save_boxes,
)
from src.graph.models import BiologyNode, PopulationNode
from src.graph.store import GraphStore


# ── Geometry + relation labels ───────────────────────────────────────────────


def test_contains_and_fraction():
    big = Box.cube(np.zeros(2), 1.0)       # [-1,1]^2
    small = Box.cube(np.zeros(2), 0.5)     # [-0.5,0.5]^2
    assert contains(big, small)
    assert not contains(small, big)
    assert containment_fraction(big, small) == pytest.approx(1.0)
    assert containment_fraction(small, big) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Box.cube(np.zeros(2), 1.0), Box.cube(np.zeros(2), 1.0), "merge"),
        (Box.cube(np.zeros(2), 1.0), Box.cube(np.zeros(2), 0.5), "parent_of"),
        (Box.cube(np.zeros(2), 0.5), Box.cube(np.zeros(2), 1.0), "child_of"),
        (Box(np.zeros(2), np.full(2, 2.0)), Box(np.ones(2), np.full(2, 3.0)), "sibling"),
        (Box(np.zeros(2), np.full(2, 1.0)), Box(np.full(2, 5.0), np.full(2, 6.0)), "unrelated"),
    ],
)
def test_relation_labels(a, b, expected):
    assert relation(a, b) == expected


# ── Fit: parent contains children by construction ────────────────────────────


def test_fit_makes_parent_contain_children():
    centers = {
        "parent": [0.0, 0.0],
        "childA": [0.3, 0.1],
        "childB": [-0.2, -0.15],
        "grandchild": [0.32, 0.12],
    }
    pairs = [("parent", "childA"), ("parent", "childB"), ("childA", "grandchild")]
    boxes = fit_boxes(centers, pairs, leaf_half_width=0.05, margin=0.05)
    assert contains(boxes["parent"], boxes["childA"])
    assert contains(boxes["parent"], boxes["childB"])
    assert contains(boxes["childA"], boxes["grandchild"])
    # transitivity: the parent also contains the grandchild
    assert contains(boxes["parent"], boxes["grandchild"])
    # leaf with no children keeps a small box
    assert relation(boxes["grandchild"], boxes["childA"]) == "child_of"


def test_fit_ignores_unknown_ids_in_pairs():
    boxes = fit_boxes({"a": [0.0]}, [("a", "missing"), ("ghost", "a")])
    assert set(boxes) == {"a"}


# ── Population-hierarchy supervision ─────────────────────────────────────────


def test_population_parent_child_pairs_from_ids():
    g = GraphStore()
    for pid in [
        "melanoma__unselected",
        "melanoma__cd274_positive",
        "melanoma__cd274_positive__line_first",
        "nsclc__unselected",
    ]:
        g.add_node(PopulationNode(id=pid, name=pid))
    pairs = set(population_parent_child_pairs(g))
    assert pairs == {
        ("melanoma__unselected", "melanoma__cd274_positive"),
        ("melanoma__unselected", "melanoma__cd274_positive__line_first"),
        ("melanoma__cd274_positive", "melanoma__cd274_positive__line_first"),
    }  # no cross-indication edges; subset ⇒ ancestor


def test_biology_parent_child_pairs_from_ancestors():
    g = GraphStore()
    for bid in ["R-HSA-162582", "R-HSA-9006934", "R-HSA-194138", "R-HSA-orphan"]:
        g.add_node(BiologyNode(id=bid, name=bid))
    # synthetic Reactome ancestry: 194138 ⊂ 9006934 ⊂ 162582 (child→ancestors)
    anc = {
        "R-HSA-194138": ["R-HSA-9006934", "R-HSA-162582"],
        "R-HSA-9006934": ["R-HSA-162582"],
        "R-HSA-162582": [],
        "R-HSA-orphan": [],
    }
    pairs = set(biology_parent_child_pairs(g, ancestors_fn=lambda s: anc.get(s, [])))
    assert ("R-HSA-9006934", "R-HSA-194138") in pairs  # (parent, child)
    assert ("R-HSA-162582", "R-HSA-194138") in pairs
    assert ("R-HSA-162582", "R-HSA-9006934") in pairs
    # the orphan (no in-graph ancestors/descendants) appears in no pair
    assert all("R-HSA-orphan" not in pair for pair in pairs)


def test_fit_graph_boxes_with_injected_embed_fn():
    g = GraphStore()
    for pid in ["melanoma__unselected", "melanoma__cd274_positive"]:
        g.add_node(PopulationNode(id=pid, name=pid))

    # Deterministic 3-d stub embedding — no model needed.
    def stub_embed(text: str) -> list[float]:
        h = abs(hash(text)) % 1000
        return [h / 1000.0, (h % 7) / 7.0, (h % 13) / 13.0]

    boxes = fit_graph_boxes(g, node_types=("PopulationNode",), embed_fn=stub_embed)
    assert set(boxes) == {"melanoma__unselected", "melanoma__cd274_positive"}
    # the parent (unselected) box contains the subgroup box
    assert contains(boxes["melanoma__unselected"], boxes["melanoma__cd274_positive"])


# ── Persistence is private-only ──────────────────────────────────────────────


def test_save_boxes_refuses_public_path(tmp_path, monkeypatch):
    from src.boundary import PrivateArtifactMisrouted

    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(tmp_path / "private"))
    boxes = {"a": Box.cube(np.zeros(2), 0.1)}
    with pytest.raises(PrivateArtifactMisrouted):
        save_boxes(boxes, tmp_path / "exports" / "boxes.json")  # not under private root


def test_save_load_roundtrip_under_private_root(tmp_path, monkeypatch):
    from src.boundary import private_root

    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(tmp_path / "private"))
    boxes = {"a": Box(np.array([0.0, 1.0]), np.array([2.0, 3.0]))}
    path = save_boxes(boxes, private_root(create=True) / "boxes.json")
    loaded = load_boxes(path)
    assert np.allclose(loaded["a"].min, [0.0, 1.0])
    assert np.allclose(loaded["a"].max, [2.0, 3.0])


def test_apply_boxes_to_graph_sets_private_fields(tmp_path):
    g = GraphStore()
    g.add_node(PopulationNode(id="melanoma__unselected", name="x"))
    n = apply_boxes_to_graph(g, {"melanoma__unselected": Box.cube(np.zeros(2), 0.1)})
    assert n == 1
    node = g.get_node("melanoma__unselected")
    assert "box_min" in node and "box_max" in node


def test_boxes_are_stripped_from_public_snapshot(tmp_path):
    """The A.2 moat: trained box params must never reach a committed snapshot."""
    g = GraphStore()
    g.add_node(PopulationNode(id="melanoma__unselected", name="x"))
    apply_boxes_to_graph(g, {"melanoma__unselected": Box.cube(np.zeros(2), 0.1)})
    out = tmp_path / "pub.json"
    g.export_snapshot(str(out))  # public path strips private fields
    text = out.read_text()
    assert "box_min" not in text
    assert "box_max" not in text
