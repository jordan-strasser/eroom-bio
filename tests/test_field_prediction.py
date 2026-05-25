"""Tests for (s,t)-localized chain prediction (dual P(success)).

Stubbed embeddings; verifies that a localizable edge's belief is swapped for the
field-queried value while non-localizable edges keep their scalar mean, and that
the two aggregates use the same softmin (so they're comparable).
"""

from __future__ import annotations

import types

from src.graph.models import EdgeBeliefState, EdgeType
from src.inference.belief_field import BeliefField
from src.prediction.field_prediction import localized_chain_probability


def _ec(src, tgt, et, alpha, beta):
    return types.SimpleNamespace(
        source_id=src, target_id=tgt, edge_type=et,
        belief=EdgeBeliefState(alpha=alpha, beta=beta),
    )


def test_localized_swaps_field_mean_for_localizable_edge():
    # mechanism_affects edge: scalar mean 0.8, but the field (marginal 0.3, no
    # nearby anchors) localizes to 0.3 at this trial's (s,t).
    edges = [
        _ec("M", "B", EdgeType.MECHANISM_AFFECTS, 8.0, 2.0),   # localizable
        _ec("C", "T", EdgeType.AFFECTS, 6.0, 4.0),             # not localizable
    ]
    field_map = {
        ("M", "B", "mechanism_affects"): BeliefField(marginal_alpha=3.0, marginal_beta=7.0),
    }
    st_map = {("M", "B", "mechanism_affects"): ("VEGFR2 inhibition", "angiogenesis")}
    p_scalar, p_local, per_edge = localized_chain_probability(
        edges, field_map, st_map, embed_fn=lambda t: [float(len(t)), 1.0],
    )
    # localizable edge moved 0.8 -> ~0.3; non-localizable stayed 0.6
    me = next(e for e in per_edge if e["edge"].startswith("M--"))
    assert me["is_localized"] and me["scalar"] == 0.8 and abs(me["localized"] - 0.3) < 0.02
    ce = next(e for e in per_edge if e["edge"].startswith("C--"))
    assert not ce["is_localized"] and ce["scalar"] == ce["localized"] == 0.6
    # localization pulled the chain belief down
    assert p_local < p_scalar


def test_no_field_means_scalar_equals_localized():
    edges = [_ec("M", "B", EdgeType.MECHANISM_AFFECTS, 8.0, 2.0)]
    st_map = {("M", "B", "mechanism_affects"): ("x", "y")}
    p_scalar, p_local, _ = localized_chain_probability(
        edges, {}, st_map, embed_fn=lambda t: [1.0, 0.0],
    )
    assert abs(p_scalar - p_local) < 1e-9  # no field → identical


def test_missing_st_desc_falls_back_to_scalar():
    edges = [_ec("M", "B", EdgeType.MECHANISM_AFFECTS, 8.0, 2.0)]
    field_map = {("M", "B", "mechanism_affects"): BeliefField(marginal_alpha=3.0, marginal_beta=7.0)}
    # field exists but no (s,t) text for this edge → can't query → scalar
    p_scalar, p_local, per_edge = localized_chain_probability(
        edges, field_map, {}, embed_fn=lambda t: [1.0, 0.0],
    )
    assert not per_edge[0]["is_localized"]
    assert abs(p_scalar - p_local) < 1e-9
