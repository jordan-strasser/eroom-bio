"""Tests for the public/private artifact boundary (src/boundary.py).

These lock in the behavior that protects the moat at serialization time, so a
future manifold field (a fine-tuned embedding, a trained box, a per-region
belief field) cannot silently flow into a committed data/exports/ snapshot.
"""

import json

import pytest

from src.boundary import (
    PrivateArtifactLeak,
    PrivateArtifactMisrouted,
    PrivateRootMislocated,
    assert_public_safe,
    find_private_fields,
    is_private_field,
    private_root,
    strip_private,
)
from src.graph.models import EdgeBeliefState, EdgeType, GraphEdge
from src.graph.store import GraphStore


# ── The contract ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "embedding",
        "belief_field",
        "source_embedding",  # suffix convention
        "target_embedding",
        "box_min",
        "node_box",  # suffix convention
        "region_anchors",
        "fine_tuned_weights",  # suffix convention
    ],
)
def test_private_names_are_private(name):
    assert is_private_field(name)


@pytest.mark.parametrize(
    "name",
    ["alpha", "beta", "evidence", "description", "id", "node_type", "metadata"],
)
def test_public_names_are_public(name):
    assert not is_private_field(name)


# ── Scanning / stripping ─────────────────────────────────────────────────────


def test_strip_private_is_recursive_and_keeps_public():
    payload = {
        "graph": {
            "nodes": [
                {"id": "VEGF", "description": "VEGF signaling", "embedding": [0.1, 0.2]},
            ],
            "links": [
                {
                    "source": "A",
                    "target": "B",
                    "belief": {"alpha": 3.0, "beta": 1.0, "belief_field": {"x": 1}},
                }
            ],
        }
    }
    cleaned = strip_private(payload)
    node = cleaned["graph"]["nodes"][0]
    assert node == {"id": "VEGF", "description": "VEGF signaling"}
    assert cleaned["graph"]["links"][0]["belief"] == {"alpha": 3.0, "beta": 1.0}


def test_empty_private_fields_do_not_count_as_leaks():
    # A declared-but-unpopulated schema field is safe.
    payload = {"nodes": [{"id": "X", "embedding": None, "box": []}]}
    assert find_private_fields(payload) == []
    assert_public_safe(payload)  # does not raise


def test_assert_public_safe_raises_on_populated_private_value():
    payload = {"nodes": [{"id": "X", "embedding": [0.1, 0.2]}]}
    with pytest.raises(PrivateArtifactLeak) as exc:
        assert_public_safe(payload, source="data/exports/x.json")
    assert "embedding" in str(exc.value)
    assert "data/exports/x.json" in str(exc.value)


# ── GraphStore export paths ──────────────────────────────────────────────────


def _store_with_private_values() -> GraphStore:
    """A store whose serialized node/edge carry simulated manifold values.

    We set the private attributes directly on the NetworkX graph because the
    public models do not declare embedding/belief-field fields yet — this is a
    faithful stand-in for what ``add_node``/``add_edge`` will emit once the
    manifold build adds them.
    """
    store = GraphStore()
    store._graph.add_node(
        "VEGF",
        node_type="BiologyNode",
        id="VEGF",
        description="VEGF signaling",
        embedding=[0.11, 0.22, 0.33],  # fine-tuned manifold-1 value (private)
    )
    edge = GraphEdge(
        source_id="VEGF",
        target_id="ANGIO",
        edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=EdgeBeliefState(alpha=4.0, beta=2.0),
    )
    store.add_edge(edge)
    # Simulate the per-region belief field landing on the edge belief blob.
    edge_data = store._get_edge_data("VEGF", "ANGIO", EdgeType.MECHANISM_AFFECTS)
    edge_data["belief"]["belief_field"] = {"anchors": [[0.1, 0.2]]}
    return store


def test_export_snapshot_strips_private_and_keeps_scalar(tmp_path):
    store = _store_with_private_values()
    out = tmp_path / "public.json"
    store.export_snapshot(str(out))

    raw = json.loads(out.read_text())
    # No private values anywhere in the committed artifact.
    assert find_private_fields(raw) == []
    assert "embedding" not in out.read_text()
    assert "belief_field" not in out.read_text()
    # Public scalar belief survived untouched (node_link_data names the edge
    # list "links" or "edges" depending on the NetworkX version).
    edges = raw["graph"].get("links") or raw["graph"].get("edges") or []
    (link,) = edges
    assert link["belief"]["alpha"] == 4.0
    assert link["belief"]["beta"] == 2.0
    # And the public snapshot still round-trips through import.
    reloaded = GraphStore()
    reloaded.import_snapshot(str(out))
    belief = reloaded.get_edge_belief("VEGF", "ANGIO", EdgeType.MECHANISM_AFFECTS)
    assert (belief.alpha, belief.beta) == (4.0, 2.0)


def test_export_private_snapshot_keeps_private_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(tmp_path / "private"))
    store = _store_with_private_values()
    out = private_root(create=True) / "full.json"
    store.export_private_snapshot(str(out))

    raw = json.loads(out.read_text())
    # Private values ARE present in the enterprise artifact.
    hits = find_private_fields(raw)
    assert any(h.endswith("embedding") for h in hits)
    assert any("belief_field" in h for h in hits)


def test_export_private_snapshot_refuses_outside_private_root(tmp_path, monkeypatch):
    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(tmp_path / "private"))
    store = _store_with_private_values()
    stray = tmp_path / "exports" / "leak.json"  # not under the private root
    with pytest.raises(PrivateArtifactMisrouted):
        store.export_private_snapshot(str(stray))


def test_private_root_refuses_location_inside_repo(monkeypatch):
    # Pointing the private root at the public repo's own data/ dir must fail.
    from src import boundary

    repo = boundary._public_repo_root()
    monkeypatch.setenv("EROOM_PRIVATE_ROOT", str(repo / "data" / "private"))
    with pytest.raises(PrivateRootMislocated):
        private_root()
