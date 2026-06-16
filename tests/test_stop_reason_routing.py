"""Tests for the CT.gov stop-reason routing override (terminated-trial misroute fix).

Verifies the deterministic override on the real documented stop reasons from the
oncology corpus plus edge cases, and that the attributor's cache hook is a no-op
when the cache is absent (default behaviour preserved → reproducible builds).
"""
from src.annotation.taxonomy import RoutingBranch, stop_reason_override


class TestStopReasonOverride:
    def test_business_decision_terminated_to_operational(self):
        # NCT03287245 — real string
        ov = stop_reason_override(
            "TERMINATED",
            "The Sponsor decided to discontinue the development of idasanutlin "
            "in the polycythemia vera indication.",
        )
        assert ov is not None and ov[0] is RoutingBranch.OPERATIONAL

    def test_strategic_accrual_with_negated_safety_to_operational(self):
        # NCT04251533 — note the negated "...not driven by any...safety concerns";
        # operational keywords (strategic/recruitment/enrollment) must win.
        ov = stop_reason_override(
            "TERMINATED",
            "The study was terminated following a strategic decision to halt "
            "recruitment due to persistently slow enrollment. This decision was "
            "not driven by any new or unexpected safety concerns.",
        )
        assert ov is not None and ov[0] is RoutingBranch.OPERATIONAL

    def test_documented_efficacy_stop_is_vetoed(self):
        # NCT01687998 — a real futility stop must NOT be overridden (it IS a
        # genuine backbone signal; trust the classifier).
        assert stop_reason_override(
            "TERMINATED", "Study termination due to insufficient efficacy."
        ) is None

    def test_ambiguous_dsmb_left_alone(self):
        # NCT01041781 — "DSMB recommendation" is ambiguous → no override.
        assert stop_reason_override("TERMINATED", "DSMB recommendation") is None

    def test_toxicity_to_safety(self):
        ov = stop_reason_override(
            "TERMINATED", "Terminated due to unacceptable toxicity."
        )
        assert ov is not None and ov[0] is RoutingBranch.SAFETY

    def test_completed_trial_never_overridden(self):
        # A completed-but-negative trial (the 77/81 majority) has no whyStopped and
        # is futility by construction → the override must not fire.
        assert stop_reason_override("COMPLETED", "") is None
        assert stop_reason_override("COMPLETED", "some text") is None

    def test_early_stop_without_reason_no_override(self):
        assert stop_reason_override("TERMINATED", "") is None
        assert stop_reason_override("WITHDRAWN", None) is None

    def test_funding_and_accrual_buckets(self):
        for why in ("Lack of funding", "Slow accrual", "Insufficient enrollment",
                    "Drug supply issues", "Study closed for business reasons"):
            ov = stop_reason_override("WITHDRAWN", why)
            assert ov is not None and ov[0] is RoutingBranch.OPERATIONAL, why

    def test_none_inputs(self):
        assert stop_reason_override(None, None) is None


class TestAttributorCacheHook:
    def test_no_op_when_cache_absent(self, tmp_path, monkeypatch):
        # With no cache file, the lookup returns None for every NCT, so the
        # override can never fire → routing behaviour is byte-identical.
        import src.annotation.attributor as attr
        monkeypatch.setattr(attr, "_CTGOV_STATUS_CACHE_PATH", tmp_path / "nope.json")
        monkeypatch.setattr(attr, "_ctgov_status_cache", None)
        assert attr._ctgov_status_for("NCT00000000") is None

    def test_cache_lookup_when_present(self, tmp_path, monkeypatch):
        import json
        import src.annotation.attributor as attr
        p = tmp_path / "ctgov_status.json"
        p.write_text(json.dumps({
            "NCT03287245": {"overall_status": "TERMINATED",
                            "why_stopped": "Sponsor decided to discontinue development."}
        }))
        monkeypatch.setattr(attr, "_CTGOV_STATUS_CACHE_PATH", p)
        monkeypatch.setattr(attr, "_ctgov_status_cache", None)
        st = attr._ctgov_status_for("NCT03287245")
        assert st and st["overall_status"] == "TERMINATED"
        assert attr._ctgov_status_for("NCT_missing") is None
