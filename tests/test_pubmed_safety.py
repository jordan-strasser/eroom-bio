"""Tests for PubMed safety enrichment (terminated trials, no posted results)."""

from __future__ import annotations

import src.annotation.pubmed_safety as ps
from src.annotation.pubmed_safety import (
    PubmedSafety,
    load_pubmed_safety,
    maybe_enrich_from_cache,
    merge_pubmed_safety,
    needs_pubmed_enrichment,
)
from src.annotation.attributor import _ae_support_bucket, _hr_support_bucket
from src.annotation.taxonomy import StructuredAE, TrialExtraction
from src.ingestion.clinicaltrials import Reference, TrialRecord, _parse_study
from src.inference.beliefs import SupportBucket


def _trial(status: str = "TERMINATED", references=None) -> TrialRecord:
    return TrialRecord(
        nct_id="NCT00134264", title="t", status=status,
        references=references if references is not None else [Reference(pmid="17984165")],
    )


def _extraction(aes=None) -> TrialExtraction:
    return TrialExtraction(trial_id="NCT00134264", adverse_events=aes or [])


def test_trigger_fires_for_terminated_empty_with_pmid():
    assert needs_pubmed_enrichment(_trial(), _extraction()) is True


def test_trigger_false_when_not_terminal():
    assert needs_pubmed_enrichment(_trial(status="COMPLETED"), _extraction()) is False


def test_trigger_false_when_aes_present():
    assert needs_pubmed_enrichment(_trial(), _extraction([StructuredAE(term="nausea")])) is False


def test_trigger_false_when_no_pmid():
    assert needs_pubmed_enrichment(_trial(references=[]), _extraction()) is False


def test_merge_unions_aes_and_signals():
    ex = _extraction([StructuredAE(term="headache")]).model_copy(
        update={"safety_signals": ["existing"]})
    safety = PubmedSafety(
        nct_id="NCT00134264",
        adverse_events=[StructuredAE(term="death", serious=True)],
        safety_signals=["bp up"],
    )
    merged = merge_pubmed_safety(ex, safety)
    assert [a.term for a in merged.adverse_events] == ["headache", "death"]
    assert merged.safety_signals == ["existing", "bp up"]


def test_maybe_enrich_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_ANNOTATIONS_DIR", tmp_path)
    safety = PubmedSafety(nct_id="NCT00134264",
                          adverse_events=[StructuredAE(term="death", serious=True)])
    (tmp_path / "NCT00134264_pubmed_safety.json").write_text(safety.model_dump_json())
    out = maybe_enrich_from_cache(_extraction(), _trial())
    assert any(a.term == "death" for a in out.adverse_events)


def test_maybe_enrich_noop_without_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_ANNOTATIONS_DIR", tmp_path)
    out = maybe_enrich_from_cache(_extraction(), _trial())
    assert out.adverse_events == []


def test_committed_torcetrapib_cache_is_valid():
    safety = load_pubmed_safety("NCT00134264")
    assert safety is not None
    assert "17984165" in safety.pmids
    death = [a for a in safety.adverse_events if "death" in a.term.lower()]
    assert death and death[0].serious
    # treatment > control so the causes_ae bucket reads support, not background
    assert death[0].incidence_treatment_pct > death[0].incidence_control_pct
    # HR + CI carried so the significance-aware bucket grades it STRONG, not WEAK
    assert death[0].hazard_ratio == 1.58 and death[0].hr_ci_low > 1.0
    # failure_causing so the DLT safety gate counts it fully (the abstract says
    # the trial was terminated *because of* these events)
    assert death[0].failure_causing
    assert any("cardiovascular" in a.term.lower() for a in safety.adverse_events)


def test_maybe_enrich_by_nct_gates_on_cache_and_empty_aes(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "_ANNOTATIONS_DIR", tmp_path)
    safety = PubmedSafety(nct_id="NCT99",
                          adverse_events=[StructuredAE(term="death", serious=True)])
    (tmp_path / "NCT99_pubmed_safety.json").write_text(safety.model_dump_json())
    # terminal/empty + cache exists -> enriches (no TrialRecord needed)
    out = ps.maybe_enrich_by_nct(TrialExtraction(trial_id="NCT99"), "NCT99")
    assert any(a.term == "death" for a in out.adverse_events)
    # already has AEs -> no-op (won't overwrite)
    ex = TrialExtraction(trial_id="NCT99", adverse_events=[StructuredAE(term="rash")])
    assert [a.term for a in ps.maybe_enrich_by_nct(ex, "NCT99").adverse_events] == ["rash"]
    # no cache -> no-op
    assert ps.maybe_enrich_by_nct(TrialExtraction(trial_id="NONE"), "NONE").adverse_events == []


def test_hr_support_bucket_grades_by_significance_and_magnitude():
    # trial-terminating mortality HR: significant (CI excludes 1.0) + large -> STRONG
    assert _hr_support_bucket(1.58, 1.14, 2.19) == SupportBucket.STRONG_SUPPORT
    # moderate significant effect
    assert _hr_support_bucket(1.25, 1.09, 1.44) == SupportBucket.MODERATE_SUPPORT
    # same point estimate but CI spans 1.0 -> not significant -> AMBIGUOUS
    assert _hr_support_bucket(1.58, 0.90, 2.50) == SupportBucket.AMBIGUOUS
    # protective (drug arm safer) -> contradict side
    assert _hr_support_bucket(0.60, 0.40, 0.85) == SupportBucket.STRONG_CONTRADICT
    # missing CI -> can't establish significance -> AMBIGUOUS
    assert _hr_support_bucket(1.58, None, None) == SupportBucket.AMBIGUOUS


def test_ae_support_bucket_hr_takes_precedence_over_rate():
    # the tiny absolute rate delta alone -> WEAK; with the HR it's STRONG
    assert _ae_support_bucket(1.23, 0.78) == SupportBucket.WEAK_SUPPORT
    assert _ae_support_bucket(
        1.23, 0.78, hazard_ratio=1.58, hr_ci_low=1.14, hr_ci_high=2.19,
    ) == SupportBucket.STRONG_SUPPORT


def test_parse_study_captures_references():
    raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00134264", "briefTitle": "ILLUMINATE"},
            "statusModule": {"overallStatus": "TERMINATED", "whyStopped": "mortality"},
            "referencesModule": {"references": [
                {"pmid": "17984165", "type": "DERIVED", "citation": "Barter PJ et al. NEJM 2007"},
            ]},
        },
    }
    rec = _parse_study(raw)
    assert rec.references and rec.references[0].pmid == "17984165"
