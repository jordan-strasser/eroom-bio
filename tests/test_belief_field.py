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


def test_index_anchor_vectors_dedups_repeated_vectors():
    """Repeated (s, t) across anchors/edges collapse to one table row; anchors
    keep an int index; the live nested objects are never mutated."""
    from src.inference.belief_field import (
        index_anchor_vectors,
        rehydrate_anchor_vectors,
    )

    shared_s = [0.1, 0.2, 0.3]
    a1 = {"s": shared_s, "t": [0.9, 0.0, 0.0], "alpha": 5.0, "beta": 1.0}
    a2 = {"s": list(shared_s), "t": [0.9, 0.0, 0.0], "alpha": 2.0, "beta": 4.0}
    links = [
        {"source": "A", "target": "B", "key": "mechanism_affects",
         "belief": {"alpha": 7.0, "beta": 5.0, "belief_field": {"anchors": [a1, a2]}}},
        {"source": "C", "target": "D", "key": "biology_drives",
         "belief": {"alpha": 1.0, "beta": 1.0, "belief_field": {"anchors": [dict(a1)]}}},
    ]
    table = index_anchor_vectors(links)
    # 2 unique vectors total: shared_s (used 3×) and [0.9,0,0] (used 3×).
    assert len(table) == 2
    # original anchor dicts untouched (fresh copies were written into links)
    assert a1["s"] == shared_s and isinstance(a1["s"], list)
    # links now carry int indices, and the shared vector resolves to one index
    new_anchors = links[0]["belief"]["belief_field"]["anchors"]
    assert isinstance(new_anchors[0]["s"], int)
    assert new_anchors[0]["s"] == new_anchors[1]["s"]  # both point to shared_s row

    # rehydrate → shared list object across every anchor with that index
    rehydrate_anchor_vectors(links, table)
    r0 = links[0]["belief"]["belief_field"]["anchors"]
    r1 = links[1]["belief"]["belief_field"]["anchors"]
    assert r0[0]["s"] is r0[1]["s"] is r1[0]["s"]  # one object in memory


def test_anchor_nct_provenance_and_field_loo():
    """Anchors carry their contributing trial; without_trial drops exactly that
    trial's anchors (the additive-kernel field LOO)."""
    f = BeliefField(marginal_alpha=4.0, marginal_beta=2.0)
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.9, nct="NCT_A")
    apply_virtual_evidence_local(f, s=[1.0, 0.0], t=[1.0, 0.0], n_eff=20.0, p_obs=0.1, nct="NCT_B")
    assert [a.nct for a in f.anchors] == ["NCT_A", "NCT_B"]
    # round-trip preserves nct
    assert BeliefField.from_dict(f.to_dict()).anchors[0].nct == "NCT_A"
    # drop NCT_A → only NCT_B's (failure) anchor remains → query drops
    p_full = expected_p(f, [1.0, 0.0], [1.0, 0.0])
    p_loo = expected_p(f.without_trial("NCT_A"), [1.0, 0.0], [1.0, 0.0])
    assert len(f.without_trial("NCT_A").anchors) == 1
    assert p_loo < p_full           # NCT_A was the success evidence; removing it lowers P
    # unknown trial → unchanged
    assert len(f.without_trial("NCT_Z").anchors) == 2


def test_dedup_preserves_anchor_nct(tmp_path, monkeypatch):
    """The #1 vector-dedup table must carry anchor nct through export/reload."""
    import json

    from src.boundary import private_root
    from src.graph.models import EdgeBeliefState, EdgeType, GraphEdge
    from src.graph.store import GraphStore
    from src.prediction.field_prediction import load_edge_fields

    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(tmp_path / "private"))
    f = BeliefField()
    apply_virtual_evidence_local(f, s=[1.0, 0.0, 0.0], t=[0.0, 1.0, 0.0], n_eff=10.0, p_obs=0.8, nct="NCT_X")
    store = GraphStore()
    store.add_edge(GraphEdge(source_id="A", target_id="B", edge_type=EdgeType.MECHANISM_AFFECTS,
                             belief=EdgeBeliefState(belief_field=f.to_dict())))
    out = private_root(create=True) / "field.json"
    store.export_private_snapshot(str(out))
    fm = load_edge_fields(str(out))
    assert fm[("A", "B", "mechanism_affects")].anchors[0].nct == "NCT_X"


def test_private_snapshot_vector_dedup_roundtrip(tmp_path, monkeypatch):
    """Full export_private_snapshot → load_edge_fields path: vectors land in the
    shared table, fields round-trip numerically, and equal vectors share one
    list object after load (the in-memory dedup)."""
    import json

    from src.boundary import private_root
    from src.graph.models import EdgeBeliefState, EdgeType, GraphEdge
    from src.graph.store import GraphStore
    from src.prediction.field_prediction import load_edge_fields

    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(tmp_path / "private"))

    # Two edges; both localize at the SAME source vector → must dedup.
    shared = [1.0, 0.0, 0.0]
    f1 = BeliefField(marginal_alpha=4.0, marginal_beta=2.0)
    apply_virtual_evidence_local(f1, s=shared, t=[0.0, 1.0, 0.0], n_eff=20.0, p_obs=0.9)
    f2 = BeliefField()
    apply_virtual_evidence_local(f2, s=shared, t=[0.0, 0.0, 1.0], n_eff=10.0, p_obs=0.3)

    store = GraphStore()
    store.add_edge(GraphEdge(
        source_id="A", target_id="B", edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=EdgeBeliefState(alpha=4.0, beta=2.0, belief_field=f1.to_dict()),
    ))
    store.add_edge(GraphEdge(
        source_id="C", target_id="D", edge_type=EdgeType.BIOLOGY_DRIVES,
        belief=EdgeBeliefState(belief_field=f2.to_dict()),
    ))
    p_before = expected_p(f1, shared, [0.0, 1.0, 0.0])

    out = private_root(create=True) / "field.json"
    store.export_private_snapshot(str(out))

    raw = json.loads(out.read_text())
    assert "_belief_field_vectors" in raw
    # the shared source vector is stored once
    assert sum(v == shared for v in raw["_belief_field_vectors"]) == 1

    fields = load_edge_fields(str(out))
    key1 = ("A", "B", "mechanism_affects")
    key2 = ("C", "D", "biology_drives")
    assert key1 in fields and key2 in fields
    # numeric identity preserved through the dedup
    assert expected_p(fields[key1], shared, [0.0, 1.0, 0.0]) == pytest.approx(p_before)
    # in-memory dedup: the shared source vector is ONE object across both edges
    assert fields[key1].anchors[0].s is fields[key2].anchors[0].s


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
