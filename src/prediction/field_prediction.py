"""(s,t)-localized chain prediction — the manifold-2 "multi-dimensional edge math"
counterpart to the scalar P(success).

The scalar path aggregates each edge's marginal Beta mean. This instead queries
each edge's per-(s,t) belief field at the *trial's own* source/target
descriptions (the point on the edge surface its mechanism actually touched), then
aggregates with the SAME softmin as `path_query._aggregate_samples` — so the two
are directly comparable and the only difference is localization.

All seven backbone edge types carry a field (see
`materialize_belief_field.EDGE_SPECS`). Edges whose endpoints are both fixed
node descriptions (`affects`, `endpoint_captures`) materialize a field that
collapses toward the scalar — every anchor sits at the same (s,t) — so they
effectively contribute their scalar mean. Edges with at least one per-arm
endpoint (`mechanism_affects`, `modulates_via`, `biology_drives`,
`reflects_biology`, `responds_differently`) genuinely localize: a chain's (s,t)
P differs from its scalar P insofar as those edges' per-arm evidence differs
from the pooled marginal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.inference.belief_field import BeliefField, expected_p
from src.prediction.path_query import _aggregate_samples, _trust_weight


def load_edge_fields(field_snapshot: str | Path) -> dict[tuple[str, str, str], BeliefField]:
    """``(source, target, edge_type) -> BeliefField`` from a private belief-field
    snapshot."""
    fd = json.loads(Path(field_snapshot).read_text())
    # Anchor (s, t) may be int indices into a shared dedup table (see
    # store.export_private_snapshot); from_dict resolves them BY REFERENCE so
    # every anchor with the same vector shares one list object in memory.
    vectors = fd.get("_belief_field_vectors")
    links = fd["graph"].get("links") or fd["graph"].get("edges") or []
    out: dict[tuple[str, str, str], BeliefField] = {}
    for e in links:
        bf = (e.get("belief") or {}).get("belief_field")
        if bf and bf.get("anchors"):
            out[(e.get("source"), e.get("target"), e.get("key"))] = (
                BeliefField.from_dict(bf, vectors)
            )
    return out


def trial_chain_descriptions(extraction_path: str | Path) -> dict[str, str]:
    """Representative ``{mechanism, biology, population}`` descriptions for a
    trial (first non-empty across its chains) — the (s,t) query text."""
    try:
        d = json.loads(Path(extraction_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out = {"mechanism": "", "biology": "", "population": ""}
    for cr in d.get("results_by_chain", []) or []:
        for key, field in (("mechanism", "mechanism_description"),
                           ("biology", "biology_description"),
                           ("population", "population_description")):
            if not out[key]:
                out[key] = (cr.get(field) or "").strip()
    return out


# Backbone: (chain src attr, chain tgt attr, edge_type, src desc-source, tgt
# desc-source). Mirrors materialize_belief_field.EDGE_SPECS — (s,t) on every edge.
_BACKBONE = [
    ("compound_id", "target_id", "affects", "node", "node"),
    ("target_id", "mechanism_id", "modulates_via", "node", "mechanism"),
    ("mechanism_id", "biology_id", "mechanism_affects", "mechanism", "biology"),
    ("biology_id", "indication_id", "biology_drives", "biology", "node"),
    ("biology_id", "endpoint_id", "reflects_biology", "biology", "node"),
    ("endpoint_id", "indication_id", "endpoint_captures", "node", "node"),
    ("subgroup_population_id", "indication_id", "responds_differently", "population", "node"),
]


def build_st_desc_map(
    graph, nct: str, *, annotations_dir: str = "data/annotations",
) -> dict[tuple[str, str, str], tuple[str, str]]:
    """Per-edge ``(source, target, edge_type) -> (s_desc, t_desc)`` for a trial,
    read from the graph's OWN node descriptions via each chain's node ids.

    The (s,t) coordinate of an edge is its source/target NODE descriptions. The
    chain already points to its per-drug nodes — the populator resolves a combo
    arm's constituents to DISTINCT nodes (paclitaxel → "mitotic arrest", not its
    neighbor's "angiogenesis inhibition") — so reading those nodes is combo-safe
    AND graph-native. This replaces the prior per-arm annotation re-derivation
    (``chain_descriptions_by_arm``), which stamped a combo arm's first-listed
    drug's description onto every chain in the arm (28% mis-attributed at n=50).
    ``annotations_dir`` is retained for signature compatibility but unused."""
    del annotations_dir  # no longer re-derives from annotations — reads the graph
    ts = graph.trial_subgraphs.get(nct)
    out: dict[tuple[str, str, str], tuple[str, str]] = {}
    if not ts:
        return out

    def ndesc(nid: str) -> str:
        try:
            nd = graph.get_node(nid)
        except KeyError:
            return ""
        return (nd.get("description") or nd.get("name") or "").strip()

    for ch in ts.chains:
        for s_attr, t_attr, et, _s_src, _t_src in _BACKBONE:
            s_id, t_id = getattr(ch, s_attr, None), getattr(ch, t_attr, None)
            if not s_id or not t_id or s_id == "UNKNOWN" or t_id == "UNKNOWN":
                continue
            s_desc, t_desc = ndesc(s_id), ndesc(t_id)
            if s_desc and t_desc:
                out[(s_id, t_id, et)] = (s_desc, t_desc)
    return out


def localized_chain_probability(
    edge_contributions: list,
    field_map: dict[tuple[str, str, str], BeliefField],
    st_desc_map: dict[tuple[str, str, str], tuple[str, str]],
    *,
    embed_fn: Callable[[str], list[float]],
    bandwidth: float | None = None,
    embed_cache: dict[str, list[float]] | None = None,
) -> tuple[float, float, list[dict]]:
    """Return ``(p_scalar, p_localized, per_edge)`` for one chain.

    Both aggregate the same edges via the engine's softmin; ``p_scalar`` uses
    each edge's marginal mean, ``p_localized`` swaps in the field-queried mean for
    the localizable edges. ``st_desc_map`` gives the *correct per-chain (s,t)
    text* per edge ``(source, target, edge_type) -> (s_desc, t_desc)`` — built
    from the trial's own arm (so a multi-arm trial queries each edge at the arm
    that actually traverses it, not a trial-level first-non-empty). ``bandwidth``
    overrides the field kernel width at query time (for the sweep) without
    re-materializing.
    """
    cache = embed_cache if embed_cache is not None else {}

    def emb(text: str) -> list[float]:
        if text not in cache:
            cache[text] = embed_fn(text)
        return cache[text]

    scalar_means, local_means, weights, per_edge = [], [], [], []
    for ec in edge_contributions:
        b = ec.belief
        scalar_mean = b.alpha / (b.alpha + b.beta)
        local_mean = scalar_mean
        et = ec.edge_type.value
        key = (ec.source_id, ec.target_id, et)
        field = field_map.get(key)
        pair = st_desc_map.get(key)
        localized = False
        if field is not None and pair and pair[0] and pair[1]:
            if bandwidth is not None:
                field = BeliefField(anchors=field.anchors, bandwidth=bandwidth,
                                    marginal_alpha=field.marginal_alpha,
                                    marginal_beta=field.marginal_beta,
                                    fallback_strength=field.fallback_strength)
            local_mean = expected_p(field, emb(pair[0]), emb(pair[1]))
            localized = True
        scalar_means.append(scalar_mean)
        local_means.append(local_mean)
        weights.append(_trust_weight(b))
        per_edge.append({
            "edge": f"{ec.source_id}--{et}-->{ec.target_id}",
            "scalar": round(scalar_mean, 3), "localized": round(local_mean, 3),
            "is_localized": localized,
        })

    if not scalar_means:
        return 0.5, 0.5, per_edge
    p_scalar = float(_aggregate_samples([np.array([m]) for m in scalar_means], weights)[0])
    p_local = float(_aggregate_samples([np.array([m]) for m in local_means], weights)[0])
    return p_scalar, p_local, per_edge
