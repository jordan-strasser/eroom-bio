"""Tests for the manifold-2 belief field (A.3).

Pure-Python geometry, no model. Covers the local update, kernel-weighted query,
the headline separability property (distinct evidence on the same edge does NOT
average), serialization, the chain integral, and the boundary guarantee that a
populated field is stripped from public snapshots.
"""

from __future__ import annotations

import pytest

from src.inference.belief_field import (
    BeliefField,
    apply_virtual_evidence_local,
    chain_integral,
    expected_p,
    localize_record,
    query,
)


def test_apply_local_adds_anchor_with_scalar_increment():
    f = BeliefField()
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.9)
    assert len(f.anchors) == 1
    a = f.anchors[0]
    assert a.alpha == pytest.approx(18.0)  # 20 * 0.9
    assert a.beta == pytest.approx(2.0)    # 20 * 0.1


def test_query_empty_field_returns_prior():
    assert query(BeliefField(), [1.0, 0.0], [1.0, 0.0]) == (1.0, 1.0)


def test_query_near_anchor_reflects_it_far_falls_to_prior():
    f = BeliefField()
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.9)
    near = expected_p(f, [1.0, 0.0], [1.0, 0.0])
    far = expected_p(f, [0.0, 1.0], [0.0, 1.0])
    assert near > 0.8        # dominated by the success anchor
    assert far == pytest.approx(0.5, abs=0.05)  # no nearby evidence → prior


def test_same_edge_distinct_st_stays_separable():
    """The whole point of manifold 2: a success at one (s,t) and a failure at a
    distant (s,t) on the SAME edge do not average to ~0.5 — each region keeps
    its own belief."""
    f = BeliefField()
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.9)  # success here
    apply_virtual_evidence_local(f, s=[0.0, 1.0], t=[0.0, 1.0], n_eff=20.0, p_obs=0.1)  # failure there
    p_success_region = expected_p(f, [1.0, 0.0], [1.0, 0.0])
    p_failure_region = expected_p(f, [0.0, 1.0], [0.0, 1.0])
    assert p_success_region > 0.75
    assert p_failure_region < 0.25
    # a naive scalar marginal would have averaged these toward ~0.5
    assert p_success_region - p_failure_region > 0.5


def test_far_query_falls_back_to_edge_marginal():
    """A query with no nearby anchors returns the edge's pooled scalar mean, not
    0.5 — the field is a strict refinement of the scalar, not a reset to
    ignorance. (Q3: marginal-centered fallback prior.)"""
    f = BeliefField(marginal_alpha=8.0, marginal_beta=2.0)  # pooled mean = 0.8
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.1)  # local failure
    near = expected_p(f, [1.0, 0.0], [1.0, 0.0])
    far = expected_p(f, [0.0, 1.0], [0.0, 1.0])
    assert near < 0.3                      # dominated by the local failure anchor
    assert far == pytest.approx(0.8, abs=0.02)  # reverts to the pooled 0.8, not 0.5


def test_default_field_fallback_is_half():
    """Default marginal (1,1) reproduces the old flat-prior behaviour."""
    assert expected_p(BeliefField(), [0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)
    assert BeliefField().fallback_prior() == (1.0, 1.0)


def test_serialization_roundtrip():
    f = BeliefField(bandwidth=0.3, marginal_alpha=6.0, marginal_beta=4.0)
    apply_virtual_evidence_local(f, s=[0.5, 0.5], t=[0.1, 0.9], n_eff=10.0, p_obs=0.7)
    f2 = BeliefField.from_dict(f.to_dict())
    assert f2.bandwidth == 0.3
    assert (f2.marginal_alpha, f2.marginal_beta) == (6.0, 4.0)
    assert len(f2.anchors) == 1
    assert f2.anchors[0].alpha == pytest.approx(7.0)


def test_chain_integral_runs_over_trajectory():
    f1 = BeliefField()
    apply_virtual_evidence_local(f1, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.9)
    f2 = BeliefField()
    apply_virtual_evidence_local(f2, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.8)
    val = chain_integral([(f1, [1.0, 0.0], [1.0, 0.0]), (f2, [1.0, 0.0], [1.0, 0.0])])
    assert 0.7 < val < 0.95  # geomean of two high-probability steps
    assert chain_integral([]) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [{"n_eff": -1.0, "p_obs": 0.5}, {"n_eff": 1.0, "p_obs": 1.5}])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        apply_virtual_evidence_local(BeliefField(), s=[1.0], t=[1.0], **bad)


def test_belief_field_is_stripped_from_public_snapshot(tmp_path):
    """A.3 moat: a populated per-region field never reaches a public snapshot."""
    from src.graph.models import EdgeBeliefState, EdgeType, GraphEdge
    from src.graph.store import GraphStore

    f = BeliefField()
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.9)
    store = GraphStore()
    store.add_edge(GraphEdge(
        source_id="VEGF", target_id="ANGIO", edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=EdgeBeliefState(alpha=4.0, beta=2.0, belief_field=f.to_dict()),
    ))
    out = tmp_path / "pub.json"
    store.export_snapshot(str(out))
    text = out.read_text()
    assert "belief_field" not in text
    assert "anchors" not in text
    # scalar marginal survives and round-trips
    reloaded = GraphStore()
    reloaded.import_snapshot(str(out))
    b = reloaded.get_edge_belief("VEGF", "ANGIO", EdgeType.MECHANISM_AFFECTS)
    assert (b.alpha, b.beta) == (4.0, 2.0)
    assert b.belief_field is None


def test_localize_record_sets_descriptions_and_embeddings():
    import types

    rec = types.SimpleNamespace(
        source_description_in_trial="", target_description_in_trial="",
        source_embedding=None, target_embedding=None,
    )
    localize_record(
        rec,
        source_description="VEGFR2 inhibition in tumor vasculature",
        target_description="angiogenesis inhibition in stage III adjuvant CRC",
        embed_fn=lambda txt: [float(len(txt)), 1.0],
    )
    assert rec.source_description_in_trial == "VEGFR2 inhibition in tumor vasculature"
    assert rec.source_embedding == [float(len("VEGFR2 inhibition in tumor vasculature")), 1.0]
    assert rec.target_embedding is not None


def _localizable_record():
    from datetime import datetime, timezone

    from src.graph.models import EvidenceRecord, EvidenceType
    from src.inference.beliefs import SupportBucket

    return EvidenceRecord(
        source_id="NCT_X",
        source_type=EvidenceType.CLINICAL_PHASE3,
        support=SupportBucket.STRONG_SUPPORT.value,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_embedding=[1.0, 0.0],
        target_embedding=[1.0, 0.0],
    )


def _edge_store():
    from src.graph.models import EdgeBeliefState, EdgeType, GraphEdge
    from src.graph.store import GraphStore

    s = GraphStore()
    s.add_edge(GraphEdge(
        source_id="A", target_id="B", edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=EdgeBeliefState(),
    ))
    return s


def test_store_update_scalar_only_when_flag_off(monkeypatch):
    from src.graph.models import EdgeType

    monkeypatch.delenv("EROOM_BELIEF_FIELD", raising=False)
    s = _edge_store()
    b = s.update_edge_belief("A", "B", EdgeType.MECHANISM_AFFECTS, _localizable_record())
    assert b.belief_field is None          # no field by default
    assert b.alpha > 1.0                    # scalar still updated


def test_store_update_localizes_when_flag_on(monkeypatch):
    from src.graph.models import EdgeType

    monkeypatch.delenv("EROOM_BELIEF_FIELD", raising=False)
    scalar_alpha = _edge_store().update_edge_belief(
        "A", "B", EdgeType.MECHANISM_AFFECTS, _localizable_record()
    ).alpha

    monkeypatch.setenv("EROOM_BELIEF_FIELD", "1")
    b = _edge_store().update_edge_belief(
        "A", "B", EdgeType.MECHANISM_AFFECTS, _localizable_record()
    )
    assert b.belief_field is not None
    assert len(b.belief_field["anchors"]) == 1
    assert b.alpha == pytest.approx(scalar_alpha)  # scalar identical regardless of flag
