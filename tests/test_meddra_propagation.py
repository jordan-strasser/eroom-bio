"""Round-28 tests for SOC-tier target_associated_ae propagation.

Sibling compounds binding the same target rarely report the same
MedDRA Preferred Term for a class-related toxicity — anacetrapib,
evacetrapib, and dalcetrapib (CETP siblings) reported atrial_fibrillation,
myocardial_infarction, and bradycardia respectively on the cardiac
side, but NEVER the same PT. PT-only target_associated_ae propagation
therefore never fired and the target-class cardiac signal stayed silent.

These tests exercise the round-28 SOC-tier roll-up that aggregates
sibling causes_ae at the MedDRA SOC parent.
"""

from __future__ import annotations

from src.graph.models import (
    AdverseEventNode,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    GraphEdge,
    Modality,
    TargetNode,
)
from src.graph.store import GraphStore
from src.inference.ae_propagation import (
    SOC_AE_PREFIX,
    propagate_to_target_associated_ae,
    soc_ae_node_id,
)
from src.inference.beliefs import apply_virtual_evidence


# ── Fixture builders ─────────────────────────────────────────────────────


def _strong_causes_ae_belief() -> EdgeBeliefState:
    """A causes_ae belief well above the propagation evidence floor."""
    b = EdgeBeliefState()
    return apply_virtual_evidence(b, n_eff=6.0, p_obs=0.8)


def _seed_cetp_siblings() -> GraphStore:
    """3 CETP siblings, each with a DIFFERENT cardiac PT. No shared PT.

    The compounds: anacetrapib, evacetrapib, dalcetrapib.
    PTs: atrial_fibrillation / myocardial_infarction / bradycardia —
    all under MedDRA SOC ``Cardiac disorders``.
    """
    g = GraphStore()
    g.add_node(TargetNode(id="CETP", name="CETP", gene_symbol="CETP"))
    for cid, name in [
        ("anacetrapib", "Anacetrapib"),
        ("evacetrapib", "Evacetrapib"),
        ("dalcetrapib", "Dalcetrapib"),
    ]:
        g.add_node(CompoundNode(id=cid, name=name, modality=Modality.SMALL_MOLECULE))
        g.add_edge(GraphEdge(
            source_id=cid, target_id="CETP",
            edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=4.0, beta=1.0),
        ))
    # Disjoint PTs, each carrying the cardiac SOC via the hierarchy
    # fields populated when the attributor (or migration) ran.
    pt_to_compound = {
        "AE:atrial_fibrillation": "anacetrapib",
        "AE:myocardial_infarction": "evacetrapib",
        "AE:bradycardia": "dalcetrapib",
    }
    for ae_id, compound in pt_to_compound.items():
        g.add_node(AdverseEventNode(
            id=ae_id, name=ae_id.removeprefix("AE:").replace("_", " "),
            system_organ_class="Cardiac disorders",
            soc_id="cardiac_disorders",
            soc_name="Cardiac disorders",
        ))
        g.add_edge(GraphEdge(
            source_id=compound, target_id=ae_id,
            edge_type=EdgeType.CAUSES_AE,
            belief=_strong_causes_ae_belief(),
        ))
    return g


# ── Tests ────────────────────────────────────────────────────────────────


class TestSOCAENodeId:
    def test_soc_ae_node_id_uses_prefix(self):
        assert soc_ae_node_id("cardiac_disorders") == "AE:soc:cardiac_disorders"
        assert soc_ae_node_id("").startswith(SOC_AE_PREFIX) is False
        assert soc_ae_node_id("") == ""


class TestSOCPropagationFiresWhenSiblingsDisjointAtPT:
    """The motivating round-28 case: 3 CETP siblings, 3 disjoint PTs,
    all rolling up to ``cardiac_disorders``. PT-only propagation can't
    fire (no shared AE). SOC-tier rolls up the cardiac class so the
    target_associated_ae edge against ``AE:soc:cardiac_disorders``
    becomes the load-bearing safety signal."""

    def test_soc_tier_target_associated_ae_emitted(self):
        g = _seed_cetp_siblings()
        # Run propagation triggered by anacetrapib's atrial_fibrillation
        # update. The SOC parent is cardiac_disorders, so the aggregation
        # key becomes AE:soc:cardiac_disorders.
        updates = propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        # No PT-tier update (no other sibling shares atrial_fibrillation).
        pt_updates = [u for u in updates if u.ae_id == "AE:atrial_fibrillation"]
        assert pt_updates == []
        # SOC-tier update emitted with all 3 siblings contributing.
        soc_updates = [
            u for u in updates if u.ae_id == "AE:soc:cardiac_disorders"
        ]
        assert len(soc_updates) == 1
        update = soc_updates[0]
        assert update.target_id == "CETP"
        assert sorted(update.contributing_compound_ids) == [
            "anacetrapib", "dalcetrapib", "evacetrapib",
        ]
        # Target_associated_ae edge actually landed in the graph.
        belief = g.get_edge_belief(
            "CETP", "AE:soc:cardiac_disorders", EdgeType.TARGET_ASSOCIATED_AE,
        )
        assert belief.expected_probability > 0.5
        # The SOC-tier AE node was created with the right metadata.
        soc_node = g.get_node("AE:soc:cardiac_disorders")
        assert soc_node["soc_id"] == "cardiac_disorders"
        assert soc_node["soc_name"] == "Cardiac disorders"
        assert soc_node["name"] == "Cardiac disorders"


class TestSOCTierIdempotent:
    def test_second_call_doesnt_double_count(self):
        g = _seed_cetp_siblings()
        first = propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        belief_a = g.get_edge_belief(
            "CETP", "AE:soc:cardiac_disorders", EdgeType.TARGET_ASSOCIATED_AE,
        )
        second = propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        belief_b = g.get_edge_belief(
            "CETP", "AE:soc:cardiac_disorders", EdgeType.TARGET_ASSOCIATED_AE,
        )
        assert belief_a.expected_probability == belief_b.expected_probability
        assert belief_a.alpha == belief_b.alpha
        assert belief_a.beta == belief_b.beta


class TestSOCTierSkipsWhenNoSOC:
    def test_no_soc_returns_no_soc_update(self):
        g = _seed_cetp_siblings()
        # An AE node WITHOUT soc_id should not trigger SOC-tier
        # aggregation (only PT-tier).
        g.add_node(AdverseEventNode(
            id="AE:weird_unknown_ae", name="Unknown",
        ))
        g.add_edge(GraphEdge(
            source_id="anacetrapib", target_id="AE:weird_unknown_ae",
            edge_type=EdgeType.CAUSES_AE,
            belief=_strong_causes_ae_belief(),
        ))
        updates = propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:weird_unknown_ae",
        )
        # No SOC update possible.
        assert all(not u.ae_id.startswith(SOC_AE_PREFIX) for u in updates)


class TestSOCTierGate:
    def test_single_sibling_below_threshold(self):
        """Only 1 sibling has cardiac AEs → SOC-tier propagation should
        NOT emit (need ≥ 2 siblings for target-class hypothesis)."""
        g = _seed_cetp_siblings()
        # Strip the causes_ae edges off two of the three siblings so
        # only anacetrapib remains a contributor at the SOC tier.
        g._graph.remove_edge(
            "evacetrapib", "AE:myocardial_infarction", EdgeType.CAUSES_AE.value,
        )
        g._graph.remove_edge(
            "dalcetrapib", "AE:bradycardia", EdgeType.CAUSES_AE.value,
        )
        updates = propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        # No SOC-tier edge created — only one sibling contributes.
        try:
            g.get_edge_belief(
                "CETP", "AE:soc:cardiac_disorders",
                EdgeType.TARGET_ASSOCIATED_AE,
            )
            assert False, "SOC-tier edge created with only one sibling"
        except KeyError:
            pass
        assert all(not u.ae_id.startswith(SOC_AE_PREFIX) for u in updates)


class TestRollUpTierOff:
    def test_pt_only_when_disabled(self):
        """Pass roll_up_tier='' to opt out of SOC propagation entirely
        (used for the pre-round-28 behavior in migration / regression
        comparison contexts)."""
        g = _seed_cetp_siblings()
        updates = propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation", roll_up_tier="",
        )
        assert all(not u.ae_id.startswith(SOC_AE_PREFIX) for u in updates)
        # And nothing was added to the graph at the SOC tier.
        try:
            g.get_node("AE:soc:cardiac_disorders")
            assert False, "SOC-tier AE node created when roll_up_tier disabled"
        except KeyError:
            pass


class TestMigrationScriptIdempotency:
    """Round-28 migration script (``scripts/migrate_ae_hierarchy.py``)
    should be safe to re-run on an already-migrated snapshot. The second
    pass must produce zero AE-node backfills (every field already
    populated) and the same SOC-tier edge counts (propagation is
    idempotent)."""

    def test_second_pass_zero_backfills(self, tmp_path):
        from scripts.migrate_ae_hierarchy import migrate_snapshot

        g = _seed_cetp_siblings()
        snap_in = tmp_path / "snap_pre.json"
        snap_mid = tmp_path / "snap_mid.json"
        snap_out = tmp_path / "snap_out.json"
        g.export_snapshot(str(snap_in))

        first = migrate_snapshot(snap_in, snap_mid)
        second = migrate_snapshot(snap_mid, snap_out)

        # All AE nodes had SOC fields written on pass 1; pass 2 finds
        # nothing left to backfill.
        assert first["ae_nodes_backfilled"] >= 0
        assert second["ae_nodes_backfilled"] == 0
        # SOC-tier edge count is stable between passes.
        assert (
            first.get("soc_tier_edges_emitted", 0)
            == second.get("soc_tier_edges_emitted", 0)
        )
        # PT-tier edge count is also stable.
        assert (
            first.get("pt_tier_edges_emitted", 0)
            == second.get("pt_tier_edges_emitted", 0)
        )
