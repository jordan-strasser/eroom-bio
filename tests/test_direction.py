"""Tests for native modulation-direction (src.graph.direction) and the
prediction-side direction de-contamination filter."""
from datetime import datetime, timezone

from src.graph import direction as D
from src.graph.models import EdgeBeliefState, EvidenceRecord, EvidenceType
from src.prediction.path_query import _direction_filtered


class TestActionTypeMapping:
    def test_buckets(self):
        assert D.direction_for_action_type("INHIBITOR") == D.ANTAGONIST
        assert D.direction_for_action_type("ANTAGONIST") == D.ANTAGONIST
        assert D.direction_for_action_type("BLOCKER") == D.ANTAGONIST
        assert D.direction_for_action_type("DEGRADER") == D.ANTAGONIST
        assert D.direction_for_action_type("AGONIST") == D.AGONIST
        assert D.direction_for_action_type("ACTIVATOR") == D.AGONIST
        assert D.direction_for_action_type("PARTIAL AGONIST") == D.AGONIST
        # non-directional / unknown
        assert D.direction_for_action_type("OTHER") == D.UNKNOWN
        assert D.direction_for_action_type("HYDROLYTIC ENZYME") == D.UNKNOWN
        assert D.direction_for_action_type(None) == D.UNKNOWN
        assert D.direction_for_action_type("") == D.UNKNOWN


class TestDirectionFromMechanisms:
    def test_empty(self):
        assert D.direction_from_mechanisms([]) == D.UNKNOWN

    def test_single_agree(self):
        m = [{"action_type": "INHIBITOR", "target_chembl_id": "CHEMBL1"}]
        assert D.direction_from_mechanisms(m) == D.ANTAGONIST

    def test_target_scoping(self):
        m = [
            {"action_type": "AGONIST", "target_chembl_id": "CHEMBL_T1"},
            {"action_type": "INHIBITOR", "target_chembl_id": "CHEMBL_T2"},
        ]
        assert D.direction_from_mechanisms(m, "CHEMBL_T1") == D.AGONIST
        assert D.direction_from_mechanisms(m, "CHEMBL_T2") == D.ANTAGONIST

    def test_disagreement_majority(self):
        m = [{"action_type": "INHIBITOR"}, {"action_type": "INHIBITOR"},
             {"action_type": "AGONIST"}]
        assert D.direction_from_mechanisms(m) == D.ANTAGONIST

    def test_tie_is_unknown(self):
        m = [{"action_type": "INHIBITOR"}, {"action_type": "AGONIST"}]
        assert D.direction_from_mechanisms(m) == D.UNKNOWN

    def test_nondirectional_only_unknown(self):
        m = [{"action_type": "BINDING AGENT"}, {"action_type": None}]
        assert D.direction_from_mechanisms(m) == D.UNKNOWN


class TestEnabledFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("EROOM_DIRECTION", raising=False)
        assert D.enabled() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv("EROOM_DIRECTION", "1")
        assert D.enabled() is True


def _rec(direction, n_eff, p_obs):
    return EvidenceRecord(
        source_id="NCT_x", source_type=EvidenceType.CLINICAL_PHASE3,
        support="strong_support", timestamp=datetime.now(timezone.utc),
        applied_n_eff=n_eff, applied_p_obs=p_obs,
        context={"direction": direction} if direction else {},
    )


class TestDirectionFilter:
    def test_unknown_query_is_noop(self):
        b = EdgeBeliefState(alpha=5.0, beta=3.0, evidence=[_rec("antagonist", 4, 1.0)])
        out = _direction_filtered(b, "unknown")
        assert (out.alpha, out.beta) == (5.0, 3.0)

    def test_no_opposite_evidence_noop(self):
        # querying agonist, but all evidence is agonist or agnostic → unchanged
        b = EdgeBeliefState(alpha=5.0, beta=3.0, evidence=[
            _rec("agonist", 4, 1.0), _rec(None, 2, 1.0)])
        out = _direction_filtered(b, "agonist")
        assert (out.alpha, out.beta) == (5.0, 3.0)

    def test_removes_opposite_direction(self):
        # pooled = prior(1,1) + agonist success(n=2,p=1 → α+2) + antagonist
        # success(n=4,p=1 → α+4): alpha=7, beta=1. Querying AGONIST should remove
        # the antagonist contribution (α-4) → alpha=3, beta=1.
        b = EdgeBeliefState(alpha=7.0, beta=1.0, evidence=[
            _rec("agonist", 2, 1.0), _rec("antagonist", 4, 1.0)])
        out = _direction_filtered(b, "agonist")
        assert abs(out.alpha - 3.0) < 1e-9
        assert abs(out.beta - 1.0) < 1e-9
        # evidence list is preserved (replayable)
        assert len(out.evidence) == 2

    def test_keeps_agnostic_evidence(self):
        # agnostic (unknown/None) evidence is NOT removed for either direction
        b = EdgeBeliefState(alpha=6.0, beta=2.0, evidence=[
            _rec(None, 3, 1.0), _rec("antagonist", 2, 0.0)])
        # querying agonist removes the antagonist record (n=2,p=0 → β-2) → beta=0→floor
        out = _direction_filtered(b, "agonist")
        assert abs(out.alpha - 6.0) < 1e-9
        assert out.beta < 1e-3  # 2.0 - 2.0 → floored
