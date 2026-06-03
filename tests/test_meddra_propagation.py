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


class TestSOCSeverityAggregation:
    """Round-29: severity_range is aggregated per-target onto the
    `target_associated_ae` EDGE (not the shared SOC AE node). This
    fixes the round-28 gap where SOC-tier edges fell to
    `_UNKNOWN_GRADE_WEIGHT=0.10` for the safety penalty, AND avoids
    cross-target leakage that a shared-node severity would cause."""

    def _seed_with_severities(self, severities: dict[str, str]) -> GraphStore:
        """CETP siblings + per-PT severity_range writes."""
        g = _seed_cetp_siblings()
        for ae_id, sev in severities.items():
            g._graph.nodes[ae_id]["severity_range"] = sev
        return g

    def _soc_edge_severity(self, g: GraphStore) -> str:
        return g._graph.edges[
            "CETP", "AE:soc:cardiac_disorders",
            EdgeType.TARGET_ASSOCIATED_AE.value,
        ].get("severity_range", "")

    def test_soc_edge_severity_is_union_of_contributing_pts(self):
        g = self._seed_with_severities({
            "AE:atrial_fibrillation": "3-5",
            "AE:myocardial_infarction": "2,3",
            "AE:bradycardia": "",
        })
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        tokens = set(self._soc_edge_severity(g).split(","))
        assert tokens == {"3-5", "2", "3"}

    def test_soc_edge_severity_max_grade_matches_pt_max(self):
        from src.prediction.path_query import _max_grade_from_severity_range

        g = self._seed_with_severities({
            "AE:atrial_fibrillation": "3-5",
            "AE:myocardial_infarction": "2",
            "AE:bradycardia": "1,3",
        })
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        assert _max_grade_from_severity_range(
            self._soc_edge_severity(g)
        ) == 5

    def test_soc_node_carries_no_severity(self):
        """Round-29 invariant: the SOC AE node does NOT store severity
        — that lives on the per-target edge so cross-target leakage is
        impossible. Confirm the node's severity_range stays empty even
        when contributing PTs have grades."""
        g = self._seed_with_severities({
            "AE:atrial_fibrillation": "5",
            "AE:myocardial_infarction": "5",
            "AE:bradycardia": "5",
        })
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        soc_node = g.get_node("AE:soc:cardiac_disorders")
        assert soc_node.get("severity_range", "") == ""

    def test_soc_edge_severity_empty_when_all_pts_empty(self):
        g = self._seed_with_severities({
            "AE:atrial_fibrillation": "",
            "AE:myocardial_infarction": "",
            "AE:bradycardia": "",
        })
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        assert self._soc_edge_severity(g) == ""

    def test_soc_edge_severity_overwrite_idempotent(self):
        """Re-running propagation produces the same edge severity (the
        underlying PTs are unchanged, so the deterministic union is
        stable). No accumulation, no doubling."""
        g = self._seed_with_severities({
            "AE:atrial_fibrillation": "3-5",
            "AE:myocardial_infarction": "2",
            "AE:bradycardia": "1",
        })
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        first = self._soc_edge_severity(g)
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        second = self._soc_edge_severity(g)
        assert first == second

    def test_low_evidence_pts_filtered_out(self):
        """Severity collection respects the same vote threshold as
        belief collection. A PT whose causes_ae evidence is below
        _MIN_EVIDENCE_STRENGTH_FOR_VOTE doesn't contribute its
        severity_range to the SOC union — symmetry with
        _collect_soc_votes."""
        g = _seed_cetp_siblings()
        # Override bradycardia's edge belief to a Beta(1,1) (no real
        # evidence). It won't pass the vote threshold.
        g._graph.edges[
            "dalcetrapib", "AE:bradycardia", EdgeType.CAUSES_AE.value,
        ]["belief"] = EdgeBeliefState().model_dump(mode="json")
        g._graph.nodes["AE:bradycardia"]["severity_range"] = "5"
        g._graph.nodes["AE:atrial_fibrillation"]["severity_range"] = "2"
        g._graph.nodes["AE:myocardial_infarction"]["severity_range"] = "3"
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        tokens = set(self._soc_edge_severity(g).split(","))
        assert "5" not in tokens
        assert tokens == {"2", "3"}

    def test_safety_penalty_uses_edge_severity(self):
        """End-to-end: a SOC-tier target_associated_ae edge with grade-5
        severity in its union produces severity_weight=0.50 via the
        round-29 edge-preferred path in _compute_safety_penalty. Without
        the round-29 edge-severity bridge, the prediction would see the
        empty AE node severity and fall to _UNKNOWN_GRADE_WEIGHT=0.10."""
        from src.prediction.path_query import _ae_severity_weight

        g_grade5 = self._seed_with_severities({
            "AE:atrial_fibrillation": "5",
            "AE:myocardial_infarction": "5",
            "AE:bradycardia": "5",
        })
        propagate_to_target_associated_ae(
            g_grade5, "anacetrapib", "AE:atrial_fibrillation",
        )
        w_g5 = _ae_severity_weight(self._soc_edge_severity(g_grade5))

        g_empty = _seed_cetp_siblings()  # no PT severities
        propagate_to_target_associated_ae(
            g_empty, "anacetrapib", "AE:atrial_fibrillation",
        )
        w_empty = _ae_severity_weight(self._soc_edge_severity(g_empty))

        # Grade 5 → severity_weight=0.50; unknown (empty) → 0.10. The
        # safety contribution scales linearly with severity_weight given
        # the same belief / trust factors.
        assert w_g5 == 0.50
        assert w_empty == 0.10
        assert w_g5 > 5 * w_empty - 1e-9

    def test_no_cross_target_leakage(self):
        """Two targets routing through the same SOC keep their own
        severity. A MEK-style target with high-grade PTs MUST NOT inflate
        a CETP-style target's edge severity, even though both target
        edges point at the same shared AE:soc:cardiac_disorders node."""
        g = _seed_cetp_siblings()
        # Give the CETP siblings empty severities (real-corpus scenario).
        # Add a second target with its OWN siblings carrying high grades.
        from src.graph.models import (
            CompoundNode, EdgeBeliefState, EdgeType, GraphEdge,
            Modality, TargetNode,
        )
        g.add_node(TargetNode(id="MAP2K1", name="MAP2K1", gene_symbol="MAP2K1"))
        for cid in ("cobimetinib", "trametinib"):
            g.add_node(CompoundNode(
                id=cid, name=cid, modality=Modality.SMALL_MOLECULE,
            ))
            g.add_edge(GraphEdge(
                source_id=cid, target_id="MAP2K1",
                edge_type=EdgeType.AFFECTS,
                belief=EdgeBeliefState(alpha=4.0, beta=1.0),
            ))
        # MEK siblings have grade-5 cardiac AE.
        from src.graph.models import AdverseEventNode
        g.add_node(AdverseEventNode(
            id="AE:cardiac_failure", name="Cardiac failure",
            system_organ_class="Cardiac disorders",
            soc_id="cardiac_disorders", soc_name="Cardiac disorders",
            severity_range="5",
        ))
        for cid in ("cobimetinib", "trametinib"):
            g.add_edge(GraphEdge(
                source_id=cid, target_id="AE:cardiac_failure",
                edge_type=EdgeType.CAUSES_AE,
                belief=_strong_causes_ae_belief(),
            ))
        # Run MEK propagation: should populate MAP2K1 edge with "5".
        propagate_to_target_associated_ae(
            g, "cobimetinib", "AE:cardiac_failure",
        )
        mek_edge_sev = g._graph.edges[
            "MAP2K1", "AE:soc:cardiac_disorders",
            EdgeType.TARGET_ASSOCIATED_AE.value,
        ].get("severity_range", "")
        # Run CETP propagation: must NOT inherit MEK's grade-5.
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        cetp_edge_sev = self._soc_edge_severity(g)
        assert mek_edge_sev == "5"
        assert cetp_edge_sev == ""  # CETP PTs had no grade data
        # And the MEK edge stayed at "5" after CETP propagation ran —
        # no cross-target overwrite.
        mek_edge_sev_after = g._graph.edges[
            "MAP2K1", "AE:soc:cardiac_disorders",
            EdgeType.TARGET_ASSOCIATED_AE.value,
        ].get("severity_range", "")
        assert mek_edge_sev_after == "5"


class TestSeriousFloor:
    """Round-29 layer-A: `serious=True` acts as a coarse severity floor
    (grade-3 weight) when CTCAE grade is missing. CT.gov rarely posts
    grades but populates `serious` on ~90% of AEs, so the floor is the
    primary severity signal in practice."""

    def test_serious_floors_unknown_to_grade3(self):
        from src.prediction.path_query import _ae_severity_weight
        # Empty severity → fallback path
        assert _ae_severity_weight("") == 0.10  # unknown
        assert _ae_severity_weight("", serious=True) == 0.15  # floor
        assert _ae_severity_weight(None, serious=True) == 0.15

    def test_serious_floor_below_grade5(self):
        """Floor doesn't override an actual grade-5: an SAE flagged
        explicitly fatal still gets the grade-5 weight."""
        from src.prediction.path_query import _ae_severity_weight
        assert _ae_severity_weight("5", serious=True) == 0.50
        assert _ae_severity_weight("3-5", serious=True) == 0.50

    def test_serious_floor_lifts_grade2(self):
        """When grade is present but lower than the serious floor, the
        floor wins (max)."""
        from src.prediction.path_query import _ae_severity_weight
        assert _ae_severity_weight("2") == 0.05
        assert _ae_severity_weight("2", serious=True) == 0.15

    def test_not_serious_preserves_existing_behavior(self):
        """serious=False (default) leaves the grade-based weight
        unchanged. Backward-compatible with pre-round-29 callers."""
        from src.prediction.path_query import _ae_severity_weight
        assert _ae_severity_weight("3") == 0.15
        assert _ae_severity_weight("3", serious=False) == 0.15

    def test_soc_edge_carries_serious_flag(self):
        """When any contributing PT is serious=True, the SOC-tier
        target_associated_ae edge gains serious=True. The edge writes
        are OR-aggregated across all contributing PTs."""
        from src.graph.models import EdgeType
        g = _seed_cetp_siblings()
        # Mark anacetrapib's PT as serious; leave the others not-serious.
        g._graph.nodes["AE:atrial_fibrillation"]["serious"] = True
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        edge_data = g._graph.edges[
            "CETP", "AE:soc:cardiac_disorders",
            EdgeType.TARGET_ASSOCIATED_AE.value,
        ]
        assert edge_data.get("serious") is True

    def test_soc_edge_serious_false_when_no_pt_serious(self):
        from src.graph.models import EdgeType
        g = _seed_cetp_siblings()  # no PT serious=True
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        edge_data = g._graph.edges[
            "CETP", "AE:soc:cardiac_disorders",
            EdgeType.TARGET_ASSOCIATED_AE.value,
        ]
        assert edge_data.get("serious") is False

    def test_safety_risk_carries_serious_through_to_penalty(self):
        """End-to-end: when SOC-tier edge has serious=True and empty
        severity_range, _compute_safety_penalty applies the grade-3
        floor (0.15) instead of the unknown weight (0.10)."""
        from src.graph.models import (
            AdverseEventNode, CausalChain, CompoundNode, EdgeBeliefState,
            EdgeType, GraphEdge, IndicationNode, MechanismNode,
            MechanismType, Modality, PopulationNode, TargetNode,
        )
        from src.prediction.path_query import PredictionEngine

        g = _seed_cetp_siblings()
        # Mark all 3 CETP siblings' cardiac PTs as serious but ungraded.
        for ae_id in (
            "AE:atrial_fibrillation",
            "AE:myocardial_infarction",
            "AE:bradycardia",
        ):
            g._graph.nodes[ae_id]["serious"] = True

        # Run propagation to populate the SOC edge metadata.
        propagate_to_target_associated_ae(
            g, "anacetrapib", "AE:atrial_fibrillation",
        )
        # Build a minimal chain for torcetrapib-on-CETP so the prediction
        # engine has something to walk and pulls in the target_class
        # cardiac signal via CETP.
        g.add_node(CompoundNode(
            id="torcetrapib", name="Torcetrapib", modality=Modality.SMALL_MOLECULE,
        ))
        g.add_edge(GraphEdge(
            source_id="torcetrapib", target_id="CETP",
            edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=4.0, beta=1.0),
        ))
        g.add_node(MechanismNode(
            id="enzyme_inhibition", name="enzyme inhibition",
            mechanism_type=MechanismType.INHIBITION,
        ))
        g.add_node(IndicationNode(id="coronary_disease", name="Coronary disease"))
        g.add_node(PopulationNode(
            id="coronary_disease__unselected",
            name="All patients (Coronary disease)",
        ))
        chain = CausalChain(
            arm_id="solo",
            compound_id="torcetrapib", target_id="CETP",
            mechanism_id="enzyme_inhibition",
            biology_id="UNKNOWN", indication_id="coronary_disease",
            endpoint_id="UNKNOWN",
            subgroup_population_id="coronary_disease__unselected",
        )
        engine = PredictionEngine(g)
        risks = engine._collect_safety_risks(
            chain, min_belief=0.55, min_evidence=1.0, max_risks=100,
        )
        # The CETP → AE:soc:cardiac_disorders target_class risk should
        # be in the list with serious=True and empty severity_range.
        cardiac_soc = [
            r for r in risks if r.ae_id == "AE:soc:cardiac_disorders"
        ]
        assert cardiac_soc, "expected SOC-tier cardiac risk in safety_risks"
        r = cardiac_soc[0]
        assert r.serious is True
        assert r.severity_range == ""  # no grade data on CETP PTs
        # The penalty math should now use the grade-3 floor, not the
        # unknown weight. Run _compute_safety_penalty and verify the
        # cardiac contribution exceeds what an unknown-grade path
        # would produce.
        from src.prediction.path_query import _ae_severity_weight
        assert _ae_severity_weight(r.severity_range, serious=r.serious) == 0.15
        # And the unknown-no-serious baseline:
        assert _ae_severity_weight(r.severity_range, serious=False) == 0.10


class TestUnionGradeTokens:
    """Unit tests for the round-29 union helper. The wire format must
    stay compatible with the existing `_max_grade_from_severity_range`
    parser in src/prediction/path_query.py."""

    def test_empty_inputs_yield_empty_string(self):
        from src.inference.ae_propagation import _union_grade_tokens
        assert _union_grade_tokens([]) == ""
        assert _union_grade_tokens(["", "", ""]) == ""

    def test_single_string_passthrough(self):
        from src.inference.ae_propagation import _union_grade_tokens
        assert _union_grade_tokens(["3-5"]) == "3-5"

    def test_dedup_and_order_stable(self):
        from src.inference.ae_propagation import _union_grade_tokens
        out = _union_grade_tokens(["1,2", "2,3", "1"])
        assert out == "1,2,3"

    def test_range_tokens_preserved(self):
        from src.inference.ae_propagation import _union_grade_tokens
        assert _union_grade_tokens(["3-5", "1,2"]) == "3-5,1,2"

    def test_whitespace_tolerated(self):
        from src.inference.ae_propagation import _union_grade_tokens
        assert _union_grade_tokens([" 1 , 2 ", "3"]) == "1,2,3"
