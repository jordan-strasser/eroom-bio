"""Tests for the precision-aware N_eff path (``EROOM_NEFF_PRECISION``).

The flag is OFF by default, in which case ``effective_n_for_evidence`` must
reproduce the legacy ``base × quality_score`` exactly (back-compat). When ON,
clinical records scale by an anchored, concave precision multiplier of the
patient count ``n_obs`` — a median-N trial reproduces its base constant, so
only the dispersion around the anchor is new signal.
"""

from datetime import datetime, timezone

import pytest

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
)
from src.graph.store import GraphStore
from src.inference import beliefs
from src.inference.beliefs import (
    EVIDENCE_TYPE_N_EFF,
    effective_n_for_evidence,
    redundancy_factor,
)

P3 = EvidenceType.CLINICAL_PHASE3
BASE = EVIDENCE_TYPE_N_EFF[P3]


@pytest.fixture
def precision_off(monkeypatch):
    monkeypatch.delenv("EROOM_NEFF_PRECISION", raising=False)


@pytest.fixture
def precision_on(monkeypatch):
    monkeypatch.setenv("EROOM_NEFF_PRECISION", "1")


# ── flag OFF: exact legacy behavior ───────────────────────────────────────

def test_flag_off_is_legacy(precision_off):
    assert effective_n_for_evidence(P3) == BASE
    assert effective_n_for_evidence(P3, 0.5) == BASE * 0.5


def test_flag_off_ignores_n_obs(precision_off):
    # With the flag off, the patient count must not change the result.
    assert effective_n_for_evidence(P3, n_obs=15000) == BASE
    assert effective_n_for_evidence(P3, n_obs=10) == BASE


# ── flag ON: anchored precision ───────────────────────────────────────────

def test_flag_on_anchor_when_no_N(precision_on):
    # No patient count -> reproduce the legacy constant (the anchor).
    assert effective_n_for_evidence(P3) == BASE
    assert effective_n_for_evidence(P3, n_obs=None) == BASE


def test_flag_on_anchor_at_reference_N(precision_on):
    assert effective_n_for_evidence(
        P3, n_obs=int(beliefs._N_REF_ANCHOR)
    ) == pytest.approx(BASE)


def test_flag_on_monotonic_in_N(precision_on):
    small = effective_n_for_evidence(P3, n_obs=50)
    mid = effective_n_for_evidence(P3, n_obs=int(beliefs._N_REF_ANCHOR))
    big = effective_n_for_evidence(P3, n_obs=15000)
    assert small < mid < big
    assert mid == pytest.approx(BASE)


def test_flag_on_clamps(precision_on):
    tiny = effective_n_for_evidence(P3, n_obs=1)
    huge = effective_n_for_evidence(P3, n_obs=10**9)
    assert tiny == pytest.approx(BASE * beliefs._PRECISION_MULT_FLOOR)
    assert huge == pytest.approx(BASE * beliefs._PRECISION_MULT_CEIL)


def test_flag_on_quality_score_is_bias_factor(precision_on):
    # quality_score still multiplies in (preserved as the bias term).
    assert effective_n_for_evidence(P3, 0.5, n_obs=int(beliefs._N_REF_ANCHOR)) == (
        pytest.approx(BASE * 0.5)
    )


def test_flag_on_curated_streams_unaffected(precision_on):
    # Curated facts / literature carry no patient count -> reproduce anchor.
    for et in (
        EvidenceType.DATABASE_OT_DIRECT,
        EvidenceType.DATABASE_CHEMBL,
        EvidenceType.LITERATURE,
        EvidenceType.COMPUTATIONAL,
    ):
        assert effective_n_for_evidence(et) == EVIDENCE_TYPE_N_EFF[et]


# ── EvidenceRecord precision fields ───────────────────────────────────────

def test_evidence_record_precision_fields_default_none():
    r = EvidenceRecord(
        source_id="NCT1",
        source_type=P3,
        support="strong_support",
        timestamp=datetime.now(timezone.utc),
    )
    assert (r.n_obs, r.effect, r.p_value, r.cluster_key) == (None, None, None, None)


def test_evidence_record_accepts_precision_fields():
    r = EvidenceRecord(
        source_id="NCT1",
        source_type=P3,
        support="strong_support",
        timestamp=datetime.now(timezone.utc),
        n_obs=500,
        effect=0.73,
        p_value=0.001,
        cluster_key="sponsorX",
    )
    assert r.n_obs == 500
    assert r.effect == 0.73
    assert r.p_value == 0.001
    assert r.cluster_key == "sponsorX"


# ── independence / redundancy discount ────────────────────────────────────

def test_redundancy_factor_flag_off(precision_off):
    assert redundancy_factor(0) == 1.0
    assert redundancy_factor(3) == 1.0  # no discount when the flag is off


def test_redundancy_factor_flag_on(precision_on):
    rho = beliefs._REDUNDANCY_RHO
    assert redundancy_factor(0) == 1.0  # first member of a cluster: full weight
    assert redundancy_factor(1) == pytest.approx(1 / (1 + rho))
    assert redundancy_factor(2) == pytest.approx(1 / (1 + 2 * rho))
    assert redundancy_factor(1) > redundancy_factor(2) > redundancy_factor(3)


def _accumulate_edge(source_ids):
    """Apply one strong_support record per source_id to a fresh edge and
    return the resulting evidence_strength (== Σ effective n_eff)."""
    g = GraphStore()
    g.add_edge(GraphEdge(
        source_id="A", target_id="B",
        edge_type=EdgeType.BIOLOGY_DRIVES, belief=EdgeBeliefState(),
    ))
    for sid in source_ids:
        g.update_edge_belief("A", "B", EdgeType.BIOLOGY_DRIVES, EvidenceRecord(
            source_id=sid, source_type=P3, support="strong_support",
            timestamp=datetime.now(timezone.utc),
        ))
    return g.get_edge_belief("A", "B", EdgeType.BIOLOGY_DRIVES).evidence_strength


def test_redundancy_discount_in_store(precision_on):
    # Three records from the SAME study are correlated -> discounted.
    same = _accumulate_edge(["NCT1", "NCT1", "NCT1"])
    # Three independent studies are not.
    distinct = _accumulate_edge(["NCT1", "NCT2", "NCT3"])
    assert same < distinct
    assert distinct == pytest.approx(3 * BASE)  # independent -> full 3x base


def test_redundancy_off_no_discount(precision_off):
    same = _accumulate_edge(["NCT1", "NCT1", "NCT1"])
    distinct = _accumulate_edge(["NCT1", "NCT2", "NCT3"])
    assert same == pytest.approx(distinct)   # legacy path: no discount
    assert same == pytest.approx(3 * BASE)
