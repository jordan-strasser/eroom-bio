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
from src.annotation.taxonomy import StructuredAE, TrialExtraction
from src.ingestion.clinicaltrials import Reference, TrialRecord, _parse_study


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
