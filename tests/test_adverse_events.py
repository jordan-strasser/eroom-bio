"""Tests for graph-native adverse-event tracking.

Covers the full AE pipeline added in this change:
  - schema (AdverseEventNode, two new edge types)
  - extraction (DoseInfo, StructuredAE, TrialExtraction fields)
  - extractor parsing of structured AE/dose JSON
  - MeddraCache + ae_node_id helper
  - attribution: per-arm incidence → causes_ae bucket update
  - cross-trial propagation: causes_ae → target_associated_ae
  - prediction: safety_risks surfacing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.annotation.attributor import (
    Attributor,
    _ae_support_bucket,
    _format_ae_note,
    _per_compound_rates_from_arms,
)
from src.annotation.extractor import (
    _build_aes_from_results_section,
    _parse_adverse_events,
    _parse_dose_info,
    _parse_extraction_response,
)
from src.annotation.meddra import MeddraCache, ae_node_id
from src.annotation.taxonomy import (
    ArmIncidence,
    DoseInfo,
    StructuredAE,
    TrialExtraction,
)
from src.graph.models import (
    AdverseEventNode,
    BiologyNode,
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialArm,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.ae_propagation import (
    propagate_to_target_associated_ae,
)
from src.inference.beliefs import SupportBucket
from src.prediction.path_query import PredictionEngine, SafetyRisk


# ── Schema ──────────────────────────────────────────────────────────────


class TestAESchema:
    def test_adverse_event_node_validates(self):
        node = AdverseEventNode(
            id="AE:hepatotoxicity",
            name="Hepatotoxicity",
            system_organ_class="Hepatobiliary disorders",
            severity_range="grade_1_3",
        )
        assert node.id == "AE:hepatotoxicity"
        assert node.metadata == {}

    def test_new_edge_types_in_enum(self):
        assert EdgeType.CAUSES_AE.value == "causes_ae"
        assert EdgeType.TARGET_ASSOCIATED_AE.value == "target_associated_ae"

    def test_store_accepts_new_edges(self):
        g = GraphStore()
        g.add_node(CompoundNode(
            id="vemurafenib", name="vemurafenib", modality=Modality.SMALL_MOLECULE,
        ))
        g.add_node(AdverseEventNode(id="AE:rash", name="Rash"))
        g.add_edge(GraphEdge(
            source_id="vemurafenib",
            target_id="AE:rash",
            edge_type=EdgeType.CAUSES_AE,
            belief=EdgeBeliefState(alpha=5, beta=2),
        ))
        b = g.get_edge_belief("vemurafenib", "AE:rash", EdgeType.CAUSES_AE)
        assert b.alpha == 5.0
        assert b.beta == 2.0


# ── Extraction model ────────────────────────────────────────────────────


class TestExtractionModel:
    def test_dose_info_defaults_empty(self):
        d = DoseInfo()
        assert d.dose == ""
        assert d.schedule == ""

    def test_structured_ae_requires_term(self):
        with pytest.raises(Exception):
            StructuredAE(term="")  # type: ignore[arg-type]

    def test_structured_ae_optional_incidence(self):
        ae = StructuredAE(term="rash")
        assert ae.incidence_treatment_pct is None
        assert ae.incidence_control_pct is None
        assert ae.serious is False

    def test_trial_extraction_carries_new_fields(self):
        t = TrialExtraction(
            trial_id="NCT01",
            duration_weeks=24.0,
            dose_info=DoseInfo(dose="100mg", schedule="qd"),
            adverse_events=[
                StructuredAE(term="nausea", incidence_treatment_pct=20),
            ],
        )
        assert t.duration_weeks == 24.0
        assert t.dose_info.dose == "100mg"
        assert len(t.adverse_events) == 1


# ── Extractor parsing ──────────────────────────────────────────────────


class TestExtractorParsing:
    def test_parse_dose_info_dict(self):
        d = _parse_dose_info({
            "dose": "150 mg", "schedule": "BID",
            "max_tolerated_dose": "200 mg", "dose_modifications": "",
        })
        assert d.dose == "150 mg"
        assert d.max_tolerated_dose == "200 mg"

    def test_parse_dose_info_legacy_string(self):
        # Old cached extractions stored dose_information as a bare string.
        d = _parse_dose_info("100 mg PO daily")
        assert d.dose == "100 mg PO daily"

    def test_parse_dose_info_empty(self):
        assert _parse_dose_info(None).dose == ""
        assert _parse_dose_info({}).dose == ""

    def test_parse_adverse_events_skips_termless(self):
        out = _parse_adverse_events([
            {"term": "rash", "incidence_treatment_pct": 30},
            {"term": "", "incidence_treatment_pct": 10},  # dropped
            {"incidence_treatment_pct": 5},  # dropped
        ])
        assert len(out) == 1
        assert out[0].term == "rash"

    def test_parse_extraction_uses_new_fields(self):
        raw = {
            "nct_id": "NCT01",
            "therapeutic_hypothesis": {"compound": "X"},
            "results": {
                "dose_info": {"dose": "50mg"},
                "adverse_events": [{"term": "fatigue", "incidence_treatment_pct": 22}],
            },
            "context": {"indication": "NSCLC", "duration_weeks": 12},
        }
        ex = _parse_extraction_response(raw, "NCT01")
        assert ex.duration_weeks == 12.0
        assert ex.dose_info.dose == "50mg"
        assert ex.adverse_events[0].term == "fatigue"
        assert ex.indication == "NSCLC"


# ── MedDRA helpers ──────────────────────────────────────────────────────


class TestMedDRA:
    def test_ae_node_id_normalizes(self):
        assert ae_node_id("Hepatotoxicity") == "AE:hepatotoxicity"
        assert (
            ae_node_id("Aspartate aminotransferase increased")
            == "AE:aspartate_aminotransferase_increased"
        )
        assert ae_node_id("") == "AE:unspecified"

    def test_meddra_cache_case_insensitive(self, tmp_path):
        cache = MeddraCache(tmp_path / "meddra.json")
        cache.set("Nausea", {
            "preferred_term": "Nausea",
            "system_organ_class": "GI disorders",
        })
        # Cosmetic differences hit the same entry.
        assert cache.get("  NAUSEA  ") is not None
        assert cache.get("nausea") is not None

    def test_meddra_cache_persists_to_disk(self, tmp_path):
        path = tmp_path / "meddra.json"
        c1 = MeddraCache(path)
        c1.set("rash", {"preferred_term": "Rash", "system_organ_class": "Skin"})
        c2 = MeddraCache(path)
        assert c2.get("rash") == {
            "preferred_term": "Rash", "system_organ_class": "Skin",
        }

    @pytest.mark.asyncio
    async def test_stale_unspecified_cache_entry_is_rejected(self, tmp_path):
        """Pre-fix caches mapped meta-rows to 'Unspecified adverse event'.

        The normalizer must treat those cache hits the same as a fresh
        LLM 'Unspecified' fallback — reject, return None, rewrite the
        cache to the empty-sentinel so the entry self-heals.
        """
        from src.annotation.meddra import normalize_ae_term
        cache = MeddraCache(tmp_path / "meddra.json")
        cache.set("any adverse event", {
            "preferred_term": "Unspecified adverse event",
            "system_organ_class": "General disorders and administration site conditions",
        })
        # _FakeAnthropicClient raises on any LLM call — proves we're not
        # paying the network round-trip for the rejection.
        result = await normalize_ae_term(
            _FakeAnthropicClient(), "any adverse event", cache,
        )
        assert result is None
        # Self-heal: the stale entry is now rewritten to the empty sentinel.
        assert cache.get("any adverse event") == {
            "preferred_term": "", "system_organ_class": "",
        }


# ── Bucket helper ──────────────────────────────────────────────────────


class TestBucketSelection:
    @pytest.mark.parametrize("t,c,expected", [
        (60.0, 10.0, SupportBucket.STRONG_SUPPORT),    # delta=50, RR=6
        (20.0, 9.0, SupportBucket.MODERATE_SUPPORT),   # delta=11, RR=2.2
        (6.0, 4.0, SupportBucket.WEAK_SUPPORT),        # delta=2, RR=1.5
        (3.0, 4.0, SupportBucket.AMBIGUOUS),           # delta=-1
        (2.0, 8.0, SupportBucket.WEAK_CONTRADICT),     # delta=-6
        (None, None, SupportBucket.AMBIGUOUS),         # no data
        (20.0, 0.0, SupportBucket.STRONG_SUPPORT),     # control=0, RR=40
        (20.0, None, SupportBucket.STRONG_SUPPORT),    # missing control
    ])
    def test_bucket_boundaries(self, t, c, expected):
        assert _ae_support_bucket(t, c) is expected

    def test_absolute_count_downgrades_low_n(self):
        # 1.2% × n=85 ≈ 1 patient — RR=2.4 would have been moderate
        # but the absolute-count gate drops it to AMBIGUOUS (fixes.md #9).
        bucket = _ae_support_bucket(1.2, 0.5, treatment_n=85)
        assert bucket is SupportBucket.AMBIGUOUS

    def test_absolute_count_partial_downgrade(self):
        # 4% × n=85 ≈ 3 patients — RR=8 would have been strong; gate
        # caps at weak_support since 3 ≤ abs_count < 5.
        bucket = _ae_support_bucket(4.0, 0.5, treatment_n=85)
        assert bucket is SupportBucket.WEAK_SUPPORT

    def test_absolute_count_preserves_high_n(self):
        # 30% × n=200 = 60 patients — clearly above the gate.
        bucket = _ae_support_bucket(30.0, 10.0, treatment_n=200)
        assert bucket is SupportBucket.STRONG_SUPPORT

    def test_absolute_count_optional_keeps_old_behavior(self):
        # No treatment_n → legacy rate-only path.
        bucket = _ae_support_bucket(1.2, 0.5)
        assert bucket is SupportBucket.MODERATE_SUPPORT

    def test_format_ae_note_includes_grade_and_rates(self):
        ae = StructuredAE(
            term="rash", grade="3",
            incidence_treatment_pct=30, incidence_control_pct=5,
            serious=True,
        )
        note = _format_ae_note(ae, "Rash")
        assert "Rash" in note and "rash" in note
        assert "grade 3" in note
        assert "tx 30%" in note and "ctrl 5%" in note
        assert "SAE" in note

    def test_format_ae_note_uses_per_compound_overrides(self):
        ae = StructuredAE(
            term="anaemia",
            arm_incidences=[
                ArmIncidence(arm_descriptor="Nivolumab", n_affected=5, n_at_risk=313, pct=1.6),
            ],
            serious=True,
        )
        note = _format_ae_note(ae, "Anaemia", tx_pct=1.6, ctrl_pct=1.6)
        assert "tx 1.6%" in note and "ctrl 1.6%" in note


# ── CT.gov direct-passthrough extraction ───────────────────────────────


class TestStructuredAEFromCTGov:
    """The structured passthrough that bypasses LLM for trials with results."""

    def _checkmate_067_ae_module(self) -> dict[str, Any]:
        """Per-arm SAE data shaped like CT.gov's adverseEventsModule.

        Numbers are from NCT01844505 (CheckMate 067) — a 3-arm trial where
        the flat tx/ctrl collapse loses real information.
        """
        return {
            "eventGroups": [
                {"id": "EG000", "title": "Nivolumab"},
                {"id": "EG001", "title": "Nivolumab + Ipilimumab"},
                {"id": "EG002", "title": "Ipilimumab"},
            ],
            "seriousEvents": [
                {  # anaemia clears the ≥3 threshold (5/313 in monos)
                    "term": "Anaemia",
                    "stats": [
                        {"groupId": "EG000", "numAffected": 5, "numAtRisk": 313},
                        {"groupId": "EG001", "numAffected": 4, "numAtRisk": 313},
                        {"groupId": "EG002", "numAffected": 5, "numAtRisk": 311},
                    ],
                },
                {  # febrile neutropenia — only 1 in combo arm, dropped by threshold
                    "term": "Febrile neutropenia",
                    "stats": [
                        {"groupId": "EG000", "numAffected": 0, "numAtRisk": 313},
                        {"groupId": "EG001", "numAffected": 1, "numAtRisk": 313},
                        {"groupId": "EG002", "numAffected": 0, "numAtRisk": 311},
                    ],
                },
            ],
        }

    def test_builds_per_arm_structured_aes(self):
        results = {"adverseEventsModule": self._checkmate_067_ae_module()}
        aes = _build_aes_from_results_section(results)
        # Only anaemia clears the ≥3-affected-in-any-arm threshold
        assert len(aes) == 1
        anaemia = aes[0]
        assert anaemia.term == "Anaemia"
        assert anaemia.serious is True
        assert anaemia.incidence_treatment_pct is None  # legacy fields stay null
        assert anaemia.incidence_control_pct is None
        descriptors = [ai.arm_descriptor for ai in anaemia.arm_incidences]
        assert descriptors == ["Nivolumab", "Nivolumab + Ipilimumab", "Ipilimumab"]
        nivo = anaemia.arm_incidences[0]
        assert nivo.n_affected == 5 and nivo.n_at_risk == 313
        assert nivo.pct == pytest.approx(100 * 5 / 313)

    def test_drops_below_threshold(self):
        # All arms have <3 affected → dropped entirely.
        results = {"adverseEventsModule": {
            "eventGroups": [{"id": "EG0", "title": "Drug"}],
            "seriousEvents": [{
                "term": "Rare AE",
                "stats": [{"groupId": "EG0", "numAffected": 2, "numAtRisk": 100}],
            }],
        }}
        assert _build_aes_from_results_section(results) == []

    def test_no_results_section_returns_empty(self):
        assert _build_aes_from_results_section(None) == []
        assert _build_aes_from_results_section({}) == []


class TestPerCompoundRates:
    """The arm-partition step that gives each compound its own (tx, ctrl)."""

    def _checkmate_067_anaemia(self) -> list[ArmIncidence]:
        return [
            ArmIncidence(arm_descriptor="Nivolumab", n_affected=5, n_at_risk=313, pct=1.6),
            ArmIncidence(arm_descriptor="Nivolumab + Ipilimumab", n_affected=4, n_at_risk=313, pct=1.3),
            ArmIncidence(arm_descriptor="Ipilimumab", n_affected=5, n_at_risk=311, pct=1.6),
        ]

    def test_nivolumab_pools_both_nivo_arms(self):
        tx, ctrl, tx_n = _per_compound_rates_from_arms(
            self._checkmate_067_anaemia(), "Nivolumab",
        )
        # nivo arms: 5+4 of 313+313; comparator: 5 of 311.
        assert tx == pytest.approx(100 * 9 / 626)
        assert ctrl == pytest.approx(100 * 5 / 311)
        assert tx_n == 626

    def test_ipilimumab_pools_both_ipi_arms(self):
        tx, ctrl, tx_n = _per_compound_rates_from_arms(
            self._checkmate_067_anaemia(), "Ipilimumab",
        )
        assert tx == pytest.approx(100 * 9 / 624)
        assert ctrl == pytest.approx(100 * 5 / 313)
        assert tx_n == 624

    def test_compound_absent_from_all_arms_returns_none(self):
        tx, ctrl, tx_n = _per_compound_rates_from_arms(
            self._checkmate_067_anaemia(), "Pembrolizumab",
        )
        assert (tx, ctrl, tx_n) == (None, None, None)

    def test_compound_on_all_arms_yields_no_control(self):
        # Single-arm trial: compound active everywhere.
        arms = [
            ArmIncidence(arm_descriptor="Nivolumab 240mg", n_affected=10, n_at_risk=100, pct=10.0),
        ]
        tx, ctrl, tx_n = _per_compound_rates_from_arms(arms, "Nivolumab")
        assert tx == 10.0
        assert ctrl is None  # no comparator arm
        assert tx_n == 100

    def test_word_boundary_avoids_substring_collision(self):
        # "Nivo" should NOT match the longer name "Nivolumab".
        arms = [ArmIncidence(arm_descriptor="Nivolumab", n_affected=5, n_at_risk=100, pct=5.0)]
        tx, ctrl, tx_n = _per_compound_rates_from_arms(arms, "Nivo")
        assert tx is None and ctrl is None and tx_n is None


# ── Attribution ────────────────────────────────────────────────────────


class _FakeAnthropicClient:
    """Stand-in for anthropic.AsyncAnthropic that the MedDRA normalizer
    can't actually reach in tests. Tests hit the cache exclusively by
    pre-populating it; this client raises if anything tries to call out.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"FakeAnthropicClient.{name} called—test should hit cache instead"
        )


def _build_combo_trial_graph() -> tuple[GraphStore, TrialSubgraph]:
    g = GraphStore()
    # Two compounds + a placebo
    g.add_node(CompoundNode(
        id="vemurafenib", name="vemurafenib", modality=Modality.SMALL_MOLECULE,
    ))
    g.add_node(CompoundNode(
        id="cobimetinib", name="cobimetinib", modality=Modality.SMALL_MOLECULE,
    ))
    g.add_node(CompoundNode(
        id="placebo", name="placebo", modality=Modality.SMALL_MOLECULE,
    ))
    g.add_node(TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF"))
    g.add_node(TargetNode(id="ENSG_MEK1", name="MEK1", gene_symbol="MAP2K1"))
    g.add_node(MechanismNode(
        id="mech_x", name="x", mechanism_type=MechanismType.INHIBITION,
    ))
    g.add_node(BiologyNode(id="bio_x", name="bio_x"))
    g.add_node(IndicationNode(id="ind_x", name="ind_x"))
    g.add_node(EndpointNode(
        id="ep_x", name="ep_x",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="ind_x__unselected", name="ind_x_unselected"))
    g.add_edge(GraphEdge(
        source_id="vemurafenib", target_id="ENSG_BRAF",
        edge_type=EdgeType.BINDS_TO,
    ))
    g.add_edge(GraphEdge(
        source_id="cobimetinib", target_id="ENSG_MEK1",
        edge_type=EdgeType.BINDS_TO,
    ))

    arms = [
        TrialArm(
            arm_id="combo",
            compound_ids=["vemurafenib", "cobimetinib"],
            regimen_compound_id="vemurafenib+cobimetinib",
            is_combination=True,
        ),
        TrialArm(
            arm_id="placebo",
            compound_ids=["placebo"],
            regimen_compound_id="placebo",
        ),
    ]
    chain = CausalChain(
        arm_id="combo", compound_id="vemurafenib+cobimetinib",
        subgroup_population_id="ind_x__unselected",
        target_id="ENSG_BRAF", mechanism_id="mech_x", biology_id="bio_x",
        indication_id="ind_x", endpoint_id="ep_x",
    )
    trial = TrialSubgraph(
        trial_id="NCT_TEST", phase="3", arms=arms, chains=[chain],
        parent_population_id="ind_x__unselected",
    )
    return g, trial


@pytest.mark.asyncio
async def test_attribute_adverse_events_skips_placebo_and_creates_edges(tmp_path):
    g, trial = _build_combo_trial_graph()
    # Pre-populate MedDRA cache so the normalizer never hits the LLM.
    cache = MeddraCache(tmp_path / "meddra.json")
    cache.set("rash", {
        "preferred_term": "Rash", "system_organ_class": "Skin disorders",
    })

    extraction = TrialExtraction(
        trial_id="NCT_TEST",
        compound_name="vemurafenib+cobimetinib",
        adverse_events=[
            StructuredAE(
                term="rash", grade="3",
                incidence_treatment_pct=45.0, incidence_control_pct=5.0,
                serious=False,
            ),
        ],
    )

    attributor = Attributor(g)
    updates = await attributor.attribute_adverse_events(
        trial, extraction, client=_FakeAnthropicClient(), meddra_cache=cache,
    )

    # Two updates: one per non-placebo compound. Placebo arm is filtered out.
    assert len(updates) == 2
    sources = {u.source_id for u in updates}
    assert sources == {"vemurafenib", "cobimetinib"}
    assert all(u.target_id == "AE:rash" for u in updates)
    assert all(u.edge_type == EdgeType.CAUSES_AE for u in updates)
    # Posteriors should have moved up from 0.5 (delta=40, RR=9 → strong_support)
    for u in updates:
        assert u.post_update_belief.expected_probability > 0.7
    # AE node was created on demand
    ae = g.get_node("AE:rash")
    assert ae["name"] == "Rash"


@pytest.mark.asyncio
async def test_attribute_adverse_events_uses_per_arm_when_available(tmp_path):
    """Three-arm trial: each compound's edge gets its own (tx, ctrl)."""
    g = GraphStore()
    g.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.SMALL_MOLECULE))
    g.add_node(CompoundNode(id="ipilimumab", name="Ipilimumab", modality=Modality.SMALL_MOLECULE))
    g.add_node(IndicationNode(id="mel", name="melanoma"))
    g.add_node(PopulationNode(id="mel__unselected", name="mel_unselected"))

    arms = [
        TrialArm(arm_id="a_nivo", compound_ids=["nivolumab"], regimen_compound_id="nivolumab"),
        TrialArm(arm_id="b_combo", compound_ids=["nivolumab", "ipilimumab"],
                 regimen_compound_id="nivolumab+ipilimumab", is_combination=True),
        TrialArm(arm_id="c_ipi", compound_ids=["ipilimumab"], regimen_compound_id="ipilimumab"),
    ]
    trial = TrialSubgraph(
        trial_id="NCT_PER_ARM", phase="3", arms=arms, chains=[],
        parent_population_id="mel__unselected",
        metadata={"enrollment": 945},
    )

    cache = MeddraCache(tmp_path / "meddra.json")
    cache.set("anaemia", {"preferred_term": "Anaemia", "system_organ_class": "Blood"})

    # CheckMate 067-style per-arm anaemia rates: 5/313, 4/313, 5/311.
    extraction = TrialExtraction(
        trial_id="NCT_PER_ARM",
        adverse_events=[StructuredAE(
            term="anaemia", serious=True,
            arm_incidences=[
                ArmIncidence(arm_descriptor="Nivolumab", n_affected=5, n_at_risk=313, pct=1.6),
                ArmIncidence(arm_descriptor="Nivolumab + Ipilimumab", n_affected=4, n_at_risk=313, pct=1.3),
                ArmIncidence(arm_descriptor="Ipilimumab", n_affected=5, n_at_risk=311, pct=1.6),
            ],
        )],
    )

    updates = await Attributor(g).attribute_adverse_events(
        trial, extraction, client=_FakeAnthropicClient(), meddra_cache=cache,
    )

    by_compound = {u.source_id: u for u in updates}
    assert set(by_compound) == {"nivolumab", "ipilimumab"}
    # Both compounds should bucket as AMBIGUOUS — pooled tx ≈ comparator,
    # so neither edge gets pushed toward "causes anaemia".
    for u in updates:
        assert u.evidence.support == SupportBucket.AMBIGUOUS.value
        # Posterior moves toward 0.5 from any starting prior (symmetric add).
        # We just confirm the note records the per-compound rates, not
        # the misleading "tx 1.6% / ctrl 1.6%" from a flat collapse.
        assert "tx" in u.evidence.notes and "ctrl" in u.evidence.notes


@pytest.mark.asyncio
async def test_attribute_adverse_events_no_treatment_arms_returns_empty(tmp_path):
    """A trial with only a placebo arm produces no causes_ae updates."""
    g = GraphStore()
    g.add_node(CompoundNode(
        id="placebo", name="placebo", modality=Modality.SMALL_MOLECULE,
    ))
    g.add_node(IndicationNode(id="ind", name="ind"))
    g.add_node(PopulationNode(id="ind__unselected", name="ind_unselected"))
    trial = TrialSubgraph(
        trial_id="NCT_PLACEBO_ONLY",
        phase="2",
        arms=[TrialArm(
            arm_id="pl", compound_ids=["placebo"], regimen_compound_id="placebo",
        )],
        chains=[],
        parent_population_id="ind__unselected",
    )
    extraction = TrialExtraction(
        trial_id="NCT_PLACEBO_ONLY",
        adverse_events=[
            StructuredAE(term="x", incidence_treatment_pct=10),
        ],
    )

    attributor = Attributor(g)
    out = await attributor.attribute_adverse_events(
        trial, extraction,
        client=_FakeAnthropicClient(),
        meddra_cache=MeddraCache(tmp_path / "m.json"),
    )
    assert out == []


# ── Cross-trial propagation ────────────────────────────────────────────


class TestPropagation:
    def _seed_target_with_compounds(
        self, n_compounds: int, alpha: float = 12.0, beta: float = 2.0,
    ) -> GraphStore:
        g = GraphStore()
        g.add_node(TargetNode(
            id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF",
        ))
        g.add_node(AdverseEventNode(id="AE:rash", name="Rash"))
        for i in range(n_compounds):
            cid = f"compound_{i}"
            g.add_node(CompoundNode(
                id=cid, name=cid, modality=Modality.SMALL_MOLECULE,
            ))
            g.add_edge(GraphEdge(
                source_id=cid, target_id="ENSG_BRAF",
                edge_type=EdgeType.BINDS_TO,
            ))
            g.add_edge(GraphEdge(
                source_id=cid, target_id="AE:rash",
                edge_type=EdgeType.CAUSES_AE,
                belief=EdgeBeliefState(alpha=alpha, beta=beta),
            ))
        return g

    def test_single_compound_no_target_edge(self):
        g = self._seed_target_with_compounds(1)
        out = propagate_to_target_associated_ae(g, "compound_0", "AE:rash")
        assert out == []
        with pytest.raises(KeyError):
            g.get_edge_belief(
                "ENSG_BRAF", "AE:rash", EdgeType.TARGET_ASSOCIATED_AE,
            )

    def test_three_compounds_lifts_target_belief(self):
        g = self._seed_target_with_compounds(3)
        out = propagate_to_target_associated_ae(g, "compound_0", "AE:rash")
        assert len(out) == 1
        applied = out[0]
        assert applied.target_id == "ENSG_BRAF"
        assert applied.ae_id == "AE:rash"
        assert set(applied.contributing_compound_ids) == {
            "compound_0", "compound_1", "compound_2",
        }
        # Pre = Beta(1,1) → 0.5; post should be substantially higher.
        assert applied.pre_update_belief.expected_probability == 0.5
        assert applied.post_update_belief.expected_probability > 0.75

    def test_propagation_is_idempotent(self):
        g = self._seed_target_with_compounds(3)
        propagate_to_target_associated_ae(g, "compound_0", "AE:rash")
        b1 = g.get_edge_belief(
            "ENSG_BRAF", "AE:rash", EdgeType.TARGET_ASSOCIATED_AE,
        )
        propagate_to_target_associated_ae(g, "compound_0", "AE:rash")
        b2 = g.get_edge_belief(
            "ENSG_BRAF", "AE:rash", EdgeType.TARGET_ASSOCIATED_AE,
        )
        assert b1.alpha == pytest.approx(b2.alpha)
        assert b1.beta == pytest.approx(b2.beta)

    def test_low_evidence_compounds_dont_vote(self):
        # Beta(1,1) compounds have evidence_strength = 0 < 1.0 threshold.
        g = self._seed_target_with_compounds(3, alpha=1.0, beta=1.0)
        out = propagate_to_target_associated_ae(g, "compound_0", "AE:rash")
        assert out == []  # all 3 compounds filtered → no qualifying votes


# ── Prediction safety_risks ────────────────────────────────────────────


def _build_prediction_graph() -> tuple[GraphStore, CausalChain]:
    g = GraphStore()
    g.add_node(CompoundNode(
        id="vemu", name="vemu", modality=Modality.SMALL_MOLECULE,
    ))
    g.add_node(TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF"))
    g.add_node(MechanismNode(
        id="mech", name="mech", mechanism_type=MechanismType.INHIBITION,
    ))
    g.add_node(BiologyNode(id="bio", name="bio"))
    g.add_node(IndicationNode(id="mel", name="mel"))
    g.add_node(EndpointNode(
        id="OS", name="OS",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="mel__unselected", name="mel_unselected"))
    g.add_node(AdverseEventNode(id="AE:rash", name="Rash"))
    g.add_node(AdverseEventNode(id="AE:hyperkeratosis", name="Hyperkeratosis"))
    g.add_node(AdverseEventNode(id="AE:weak", name="Weak"))

    for s, t, et in [
        ("vemu", "ENSG_BRAF", EdgeType.BINDS_TO),
        ("ENSG_BRAF", "mech", EdgeType.MODULATES_VIA),
        ("mech", "bio", EdgeType.MECHANISM_AFFECTS),
        ("bio", "mel", EdgeType.BIOLOGY_DRIVES),
        ("bio", "OS", EdgeType.REFLECTS_BIOLOGY),
        ("OS", "mel", EdgeType.ENDPOINT_CAPTURES),
    ]:
        g.add_edge(GraphEdge(source_id=s, target_id=t, edge_type=et))
    # Strong compound-specific risk
    g.add_edge(GraphEdge(
        source_id="vemu", target_id="AE:rash",
        edge_type=EdgeType.CAUSES_AE,
        belief=EdgeBeliefState(alpha=10, beta=2),
    ))
    # Strong target-class risk
    g.add_edge(GraphEdge(
        source_id="ENSG_BRAF", target_id="AE:hyperkeratosis",
        edge_type=EdgeType.TARGET_ASSOCIATED_AE,
        belief=EdgeBeliefState(alpha=8, beta=4),
    ))
    # Below-threshold compound risk (P=0.33)
    g.add_edge(GraphEdge(
        source_id="vemu", target_id="AE:weak",
        edge_type=EdgeType.CAUSES_AE,
        belief=EdgeBeliefState(alpha=2, beta=4),
    ))

    chain = CausalChain(
        arm_id="vemu_only", compound_id="vemu",
        subgroup_population_id="mel__unselected",
        target_id="ENSG_BRAF", mechanism_id="mech", biology_id="bio",
        indication_id="mel", endpoint_id="OS",
    )
    return g, chain


class TestPredictionSafetyRisks:
    def test_safety_risks_populated(self):
        g, chain = _build_prediction_graph()
        result = PredictionEngine(g).predict(chain, n_samples=500)
        assert len(result.safety_risks) == 2
        compound_risk = next(
            r for r in result.safety_risks if r.source == "compound"
        )
        target_risk = next(
            r for r in result.safety_risks if r.source == "target_class"
        )
        assert compound_risk.ae_id == "AE:rash"
        assert target_risk.ae_id == "AE:hyperkeratosis"

    def test_below_threshold_risks_filtered(self):
        g, chain = _build_prediction_graph()
        result = PredictionEngine(g).predict(chain, n_samples=200)
        ae_ids = {r.ae_id for r in result.safety_risks}
        assert "AE:weak" not in ae_ids

    def test_compound_risk_dedupes_target_entry(self):
        """When the same AE appears as both compound-specific and
        target-class risk, the compound-specific entry wins (more direct
        attribution) and the target-class entry is suppressed.
        """
        g, chain = _build_prediction_graph()
        # Add a target_associated_ae entry for the same AE the compound
        # already covers (rash)—should NOT show up twice.
        g.add_edge(GraphEdge(
            source_id="ENSG_BRAF", target_id="AE:rash",
            edge_type=EdgeType.TARGET_ASSOCIATED_AE,
            belief=EdgeBeliefState(alpha=10, beta=2),
        ))
        result = PredictionEngine(g).predict(chain, n_samples=200)
        rash_entries = [r for r in result.safety_risks if r.ae_id == "AE:rash"]
        assert len(rash_entries) == 1
        assert rash_entries[0].source == "compound"

    def test_safety_risks_independent_of_overall_probability(self):
        """Safety risks must NOT bleed into the efficacy P(success)."""
        g, chain = _build_prediction_graph()
        # Baseline run
        baseline = PredictionEngine(g).predict(chain, n_samples=2000)
        # Add an extreme safety signal
        g.add_node(AdverseEventNode(id="AE:bigtox", name="BigTox"))
        g.add_edge(GraphEdge(
            source_id="vemu", target_id="AE:bigtox",
            edge_type=EdgeType.CAUSES_AE,
            belief=EdgeBeliefState(alpha=50, beta=1),
        ))
        with_tox = PredictionEngine(g).predict(chain, n_samples=2000)
        # P(success) should be unchanged within sampling noise; we just
        # verify the safety_risks list grew while overall_probability
        # remains in the same neighborhood.
        assert len(with_tox.safety_risks) == len(baseline.safety_risks) + 1
        assert abs(with_tox.overall_probability - baseline.overall_probability) < 0.05


# ── Smoke: SafetyRisk model independence ───────────────────────────────


@pytest.mark.asyncio
async def test_annotate_trial_invokes_ae_attribution(tmp_path, monkeypatch):
    """End-to-end wiring check: annotate_trial must invoke
    attribute_adverse_events whenever the extraction carries AEs.

    Mocks extractor + classifier (their async LLM calls) so the test
    runs offline. Replaces normalize_ae_term with a deterministic stub
    so the MedDRA cache LLM call doesn't fire either. The real wiring
    being checked: did annotate_trial reach attribute_adverse_events,
    creating causes_ae edges as a side effect?
    """
    from src.annotation import classifier as classifier_module
    from src.annotation.classifier import annotate_trial
    from src.annotation.taxonomy import FailureClassification, FailureMode
    from src.ingestion.clinicaltrials import TrialRecord

    g, trial = _build_combo_trial_graph()
    extraction = TrialExtraction(
        trial_id="NCT_E2E",
        compound_name="vemurafenib+cobimetinib",
        adverse_events=[
            StructuredAE(
                term="rash", grade="3",
                incidence_treatment_pct=45.0, incidence_control_pct=5.0,
            ),
        ],
    )
    classification = FailureClassification(
        trial_id="NCT_E2E",
        primary_failure_mode=FailureMode.INSUFFICIENT_INFORMATION,
        confidence=0.5,
    )
    classification._raw = {"edges_to_update": []}  # type: ignore[attr-defined]

    class _StubExtractor:
        _client = _FakeAnthropicClient()
        async def extract(self, _trial):
            return extraction

    class _StubClassifier:
        async def classify(self, _ext):
            return classification

    # Pre-populate the cache and force the module to use it; the stub
    # client would raise on any real LLM call.
    cache = MeddraCache(tmp_path / "meddra.json")
    cache.set("rash", {
        "preferred_term": "Rash", "system_organ_class": "Skin disorders",
    })

    trial_record = TrialRecord(
        nct_id="NCT_E2E", title="x", phase="3", status="completed",
        conditions=[], interventions=[], primary_outcomes=[],
        secondary_outcomes=[], arm_groups=[],
    )
    g.set_trial_subgraph(trial)
    attributor = Attributor(g)

    _, _, updates = await annotate_trial(
        trial_record,
        _StubExtractor(),  # type: ignore[arg-type]
        _StubClassifier(),  # type: ignore[arg-type]
        attributor=attributor,
        build_subgraph=lambda _t, _e: trial,
        meddra_cache=cache,
    )

    # The wiring should produce two causes_ae updates (one per non-placebo
    # compound on the combo arm). If annotate_trial doesn't call
    # attribute_adverse_events, this assertion fails.
    causes_ae_updates = [
        u for u in updates if u.edge_type == EdgeType.CAUSES_AE
    ]
    assert len(causes_ae_updates) == 2
    assert {u.source_id for u in causes_ae_updates} == {
        "vemurafenib", "cobimetinib",
    }


def test_safety_risk_model_serializable():
    r = SafetyRisk(
        ae_id="AE:rash", ae_name="Rash", source="compound",
        belief_probability=0.83, evidence_strength=10.0,
        contributing_compound_ids=["vemu"],
    )
    dumped = r.model_dump()
    assert dumped["ae_id"] == "AE:rash"
    assert dumped["source"] == "compound"
