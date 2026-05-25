"""(s,t)-localized chain prediction — the manifold-2 "multi-dimensional edge math"
counterpart to the scalar P(success).

The scalar path aggregates each edge's marginal Beta mean. This instead queries
each edge's per-(s,t) belief field at the *trial's own* source/target
descriptions (the point on the edge surface its mechanism actually touched), then
aggregates with the SAME softmin as `path_query._aggregate_samples` — so the two
are directly comparable and the only difference is localization.

Only `mechanism_affects` and `responds_differently` edges carry a field
(materialized for those); every other edge contributes its scalar mean either
way, so a chain's (s,t) P differs from its scalar P exactly insofar as its
localizable edges have evidence that differs from the pooled marginal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.graph.models import EdgeType
from src.inference.belief_field import BeliefField, expected_p
from src.prediction.path_query import _aggregate_samples, _trust_weight

# edge_type -> (source desc-key, target desc-key, target-special). Matches
# materialize_belief_field.EDGE_SPECS.
_EDGE_DESC = {
    EdgeType.MECHANISM_AFFECTS.value: ("mechanism", "biology", None),
    EdgeType.RESPONDS_DIFFERENTLY.value: ("population", None, "indication_name"),
}


def load_edge_fields(field_snapshot: str | Path) -> dict[tuple[str, str, str], BeliefField]:
    """``(source, target, edge_type) -> BeliefField`` from a private belief-field
    snapshot."""
    fd = json.loads(Path(field_snapshot).read_text())
    links = fd["graph"].get("links") or fd["graph"].get("edges") or []
    out: dict[tuple[str, str, str], BeliefField] = {}
    for e in links:
        bf = (e.get("belief") or {}).get("belief_field")
        if bf and bf.get("anchors"):
            out[(e.get("source"), e.get("target"), e.get("key"))] = BeliefField.from_dict(bf)
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


def localized_chain_probability(
    edge_contributions: list,
    field_map: dict[tuple[str, str, str], BeliefField],
    descs: dict[str, str],
    *,
    embed_fn: Callable[[str], list[float]],
    indication_name: str = "",
    embed_cache: dict[str, list[float]] | None = None,
) -> tuple[float, float, list[dict]]:
    """Return ``(p_scalar, p_localized, per_edge)`` for one chain.

    Both aggregate the same edges via the engine's softmin; ``p_scalar`` uses
    each edge's marginal mean, ``p_localized`` swaps in the field-queried mean at
    this trial's (s,t) for the localizable edges (others keep their scalar mean).
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
        spec = _EDGE_DESC.get(et)
        localized = False
        if field is not None and spec is not None:
            s_key, t_key, t_special = spec
            s_desc = descs.get(s_key, "")
            t_desc = indication_name if t_special == "indication_name" else descs.get(t_key, "")
            if s_desc and t_desc:
                local_mean = expected_p(field, emb(s_desc), emb(t_desc))
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
