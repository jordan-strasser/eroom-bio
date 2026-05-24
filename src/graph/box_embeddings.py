"""Box embeddings for hierarchical geometry (manifold 1, A.2).

Each concept is an axis-aligned hyperrectangle (a *box*) in BioLORD space.
**Containment encodes is-a; intersection encodes shared mechanism.** A box's
center is the node's BioLORD embedding (A.1, ``biolord_embeddings.embed_text``);
its extent is fit so that an ontology parent's box contains its children's
boxes. This module is the canonical manifold-1 reference in code.

Public / private boundary (see ``BOUNDARY.md``): the box *schema* and all the
*code* here — fit algorithm, containment/relation inference — are public
(Apache). The *trained box parameters* (the per-node ``box_min`` / ``box_max``)
are the moat: they attach to graph nodes under those names (auto-stripped from
public snapshots by ``src/boundary.py``) and serialize only under
``EROOM_PRIVATE_ROOT`` via :func:`save_boxes`.

Numerics: a 768-d box volume is a product of 768 side lengths and underflows to
zero, so relation inference uses **mean per-dimension overlap fraction** rather
than raw volume. The fit guarantees parent⊇child by construction (bounding-box
expansion), so it needs no gradient training; the gradient fine-tune on the
gold-pair set (A.5) is the deferred refinement, not required for the
ontology-bootstrapped geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Relation thresholds on mean per-dim containment fraction. Tunable; the
# defaults separate the five gold-pair labels cleanly on bounding-box-fit
# geometry. Env overrides keep them out of call sites.
import os

_HI = float(os.environ.get("EROOM_BOX_CONTAIN_HI", "0.90"))  # ⊇ "contains"
_LO = float(os.environ.get("EROOM_BOX_OVERLAP_LO", "0.30"))  # sibling vs unrelated


@dataclass
class Box:
    """Axis-aligned box: ``min <= max`` elementwise. is-a == containment."""

    min: np.ndarray
    max: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.min + self.max) / 2.0

    @property
    def widths(self) -> np.ndarray:
        return np.maximum(self.max - self.min, 0.0)

    def to_dict(self) -> dict[str, list[float]]:
        return {"min": self.min.tolist(), "max": self.max.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> "Box":
        return cls(min=np.asarray(d["min"], float), max=np.asarray(d["max"], float))

    @classmethod
    def cube(cls, center: np.ndarray, half_width: float) -> "Box":
        center = np.asarray(center, float)
        h = np.full_like(center, float(half_width))
        return cls(min=center - h, max=center + h)


# ── Geometry ─────────────────────────────────────────────────────────────────


def contains(outer: Box, inner: Box) -> bool:
    """Exact: does ``outer`` fully contain ``inner``?"""
    return bool(np.all(outer.min <= inner.min) and np.all(inner.max <= outer.max))


def _overlap_widths(a: Box, b: Box) -> np.ndarray:
    return np.maximum(np.minimum(a.max, b.max) - np.maximum(a.min, b.min), 0.0)


def containment_fraction(whole: Box, part: Box) -> float:
    """Mean per-dimension fraction of ``part`` that lies inside ``whole``.

    1.0 ⇒ every dimension of ``part`` is within ``whole`` (``whole`` ⊇
    ``part``). Robust in high dimension where a raw volume ratio underflows.
    """
    widths = part.widths
    overlap = _overlap_widths(whole, part)
    safe = widths > 0
    if not np.any(safe):
        return 0.0
    return float(np.mean(overlap[safe] / widths[safe]))


def relation(a: Box, b: Box, *, hi: float = _HI, lo: float = _LO) -> str:
    """Classify the relation of box ``a`` to box ``b`` into the gold-pair
    vocabulary: ``merge | parent_of | child_of | sibling | unrelated``.

    ``parent_of`` means a ⊇ b (a is the broader concept). Symmetric with the
    A.5 labeling rubric and what the A.4 embedding merger consumes.
    """
    f_ab = containment_fraction(b, a)  # fraction of a inside b
    f_ba = containment_fraction(a, b)  # fraction of b inside a
    if f_ab >= hi and f_ba >= hi:
        return "merge"
    if f_ba >= hi:
        return "parent_of"  # b inside a → a is the parent
    if f_ab >= hi:
        return "child_of"   # a inside b → a is the child
    if f_ab >= lo or f_ba >= lo:
        return "sibling"
    return "unrelated"


# ── Fit (ontology-bootstrapped, closed form) ─────────────────────────────────


def _leaves_first_order(
    node_ids: list[str], parent_child: list[tuple[str, str]]
) -> list[str]:
    """Order nodes so every node precedes its parents (children first).

    Kahn's algorithm on the child→parent DAG. Any nodes left over (a cycle —
    ontologies shouldn't have them, but guard) are appended in input order.
    """
    children: dict[str, set[str]] = {n: set() for n in node_ids}
    indeg: dict[str, int] = {n: 0 for n in node_ids}
    for parent, child in parent_child:
        if parent in children and child in children and parent not in children[child]:
            children[child].add(parent)  # edge child → parent
            indeg[parent] += 1
    queue = [n for n in node_ids if indeg[n] == 0]
    order: list[str] = []
    while queue:
        n = queue.pop()
        order.append(n)
        for parent in children[n]:
            indeg[parent] -= 1
            if indeg[parent] == 0:
                queue.append(parent)
    if len(order) < len(node_ids):  # cycle fallback
        order += [n for n in node_ids if n not in set(order)]
    return order


def fit_boxes(
    centers: dict[str, list[float]],
    parent_child: list[tuple[str, str]],
    *,
    leaf_half_width: float = 0.05,
    margin: float = 0.05,
) -> dict[str, Box]:
    """Fit one box per node from BioLORD centers + ontology parent→child pairs.

    Each node starts as a small cube around its center; then, processing
    children before parents, every parent box is expanded to the bounding box
    of itself and all its children (plus ``margin``). This makes parent ⊇ child
    hold **by construction** — no gradient training. ``parent_child`` is a list
    of ``(parent_id, child_id)``; ids absent from ``centers`` are skipped.
    """
    boxes: dict[str, Box] = {
        nid: Box.cube(np.asarray(vec, float), leaf_half_width)
        for nid, vec in centers.items()
    }
    kids: dict[str, list[str]] = {nid: [] for nid in boxes}
    pairs = [(p, c) for p, c in parent_child if p in boxes and c in boxes]
    for parent, child in pairs:
        kids[parent].append(child)

    for nid in _leaves_first_order(list(boxes), pairs):
        if not kids[nid]:
            continue
        lo = boxes[nid].min.copy()
        hi = boxes[nid].max.copy()
        for c in kids[nid]:
            lo = np.minimum(lo, boxes[c].min)
            hi = np.maximum(hi, boxes[c].max)
        span = np.maximum(hi - lo, 1e-9)
        boxes[nid] = Box(min=lo - margin * span, max=hi + margin * span)
    return boxes


# ── Persistence (trained params are PRIVATE) ─────────────────────────────────


def save_boxes(boxes: dict[str, Box], filepath: str | Path) -> Path:
    """Serialize trained boxes — refuses any path outside ``EROOM_PRIVATE_ROOT``.

    Box parameters are the manifold-1 moat; they never belong in the public
    repo. Routed through the boundary so a slipped path fails loudly.
    """
    from src.boundary import require_under_private_root

    target = require_under_private_root(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({nid: b.to_dict() for nid, b in boxes.items()}))
    return target


def load_boxes(filepath: str | Path) -> dict[str, Box]:
    raw = json.loads(Path(filepath).read_text())
    return {nid: Box.from_dict(d) for nid, d in raw.items()}


def apply_boxes_to_graph(graph, boxes: dict[str, Box]) -> int:
    """Attach trained boxes to graph nodes as ``box_min`` / ``box_max``.

    Those field names are private (``src/boundary.py``) so they are stripped
    from public snapshots automatically — the geometry rides in-memory and in
    the private snapshot only. Returns the number of nodes updated.
    """
    n = 0
    for nid, box in boxes.items():
        try:
            data = graph._graph.nodes[nid]  # noqa: SLF001
        except KeyError:
            continue
        data["box_min"] = box.min.tolist()
        data["box_max"] = box.max.tolist()
        n += 1
    return n


# ── Graph integration (supervision + fit driver) ─────────────────────────────


def _parse_population_id(pop_id: str) -> tuple[str, frozenset[str]]:
    """``melanoma__cd274_positive__line_first`` → ("melanoma", {slugs}).

    ``unselected`` is the empty (root) feature set. Returns ("", frozenset())
    for ids that don't match the ``{indication}__{slug}...`` shape.
    """
    parts = pop_id.split("__")
    if len(parts) < 2:
        return "", frozenset()
    indication = parts[0]
    slugs = frozenset(p for p in parts[1:] if p and p != "unselected")
    return indication, slugs


def population_parent_child_pairs(graph) -> list[tuple[str, str]]:
    """Derive parent→child supervision from PopulationNode ids — free, in-graph.

    A population with feature set S1 is a parent of one with S2 (same
    indication) iff S1 ⊊ S2 (fewer features ⇒ broader cohort). This is the
    supervision that lets the box geometry localize ``responds_differently``
    evidence by population sub-region — the named structural fix for the
    bevacizumab AVANT holdout miss. (Biology/mechanism ontology supervision —
    Reactome↔GO, GO is-a — is the next source to wire; see A.2.)
    """
    pops: list[tuple[str, str, frozenset[str]]] = []
    for nid, data in graph._graph.nodes(data=True):  # noqa: SLF001
        if data.get("node_type") != "PopulationNode":
            continue
        ind, slugs = _parse_population_id(nid)
        if ind:
            pops.append((nid, ind, slugs))
    pairs: list[tuple[str, str]] = []
    for pid, p_ind, p_slugs in pops:
        for cid, c_ind, c_slugs in pops:
            if pid == cid or p_ind != c_ind:
                continue
            if p_slugs < c_slugs:  # strict subset ⇒ p is an ancestor of c
                pairs.append((pid, cid))
    return pairs


def biology_parent_child_pairs(graph, ancestors_fn=None) -> list[tuple[str, str]]:
    """Derive biology parent→child supervision from the Reactome hierarchy.

    For each in-graph Reactome BiologyNode, any of its ancestors that is *also*
    an in-graph BiologyNode becomes its parent. This gives the biology boxes
    real is-a structure (a parent pathway ⊇ its sub-pathways) instead of leaf
    cubes. ``ancestors_fn(stable_id) -> list[str]`` defaults to the cached
    Reactome fetcher (best-effort: offline ⇒ no biology hierarchy, graceful).
    Inject a stub in tests to stay offline.
    """
    if ancestors_fn is None:
        from src.graph.reactome_hierarchy import fetch_reactome_ancestors as ancestors_fn  # noqa

    bio_ids = {
        nid
        for nid, d in graph._graph.nodes(data=True)  # noqa: SLF001
        if d.get("node_type") == "BiologyNode" and str(nid).startswith("R-")
    }
    pairs: list[tuple[str, str]] = []
    for nid in sorted(bio_ids):
        for anc in ancestors_fn(nid):
            if anc in bio_ids and anc != nid:
                pairs.append((anc, nid))  # ancestor is the broader parent
    return pairs


def fit_graph_boxes(
    graph,
    *,
    node_types: tuple[str, ...] = ("PopulationNode", "BiologyNode", "MechanismNode"),
    embed_fn=None,
    ancestors_fn=None,
    leaf_half_width: float = 0.05,
    margin: float = 0.05,
) -> dict[str, Box]:
    """Fit manifold-1 boxes for the graph: BioLORD centers + ontology pairs.

    Centers are ``embed_fn(description or name)`` for each selected node;
    ``embed_fn`` defaults to ``biolord_embeddings.embed_text`` (lazy import so
    this module needs no ML deps to import or unit-test — inject a stub in
    tests). Supervision is currently the population hierarchy; nodes without
    parent/child supervision get a leaf cube (still usable for merge/sibling
    via overlap). Trained boxes are returned; persist with :func:`save_boxes`
    (private) and/or :func:`apply_boxes_to_graph`.
    """
    if embed_fn is None:
        from src.graph.biolord_embeddings import embed_text as embed_fn  # noqa

    selected = set(node_types)
    centers: dict[str, list[float]] = {}
    for nid, data in graph._graph.nodes(data=True):  # noqa: SLF001
        if data.get("node_type") not in selected:
            continue
        text = (data.get("description") or "").strip() or (data.get("name") or "").strip()
        if not text:
            continue
        centers[nid] = embed_fn(text)

    pairs = population_parent_child_pairs(graph)
    if "BiologyNode" in selected:
        pairs = pairs + biology_parent_child_pairs(graph, ancestors_fn=ancestors_fn)
    return fit_boxes(
        centers, pairs, leaf_half_width=leaf_half_width, margin=margin,
    )
