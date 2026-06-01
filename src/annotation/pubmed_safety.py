"""PubMed safety enrichment for terminated trials with no posted results.

A trial that was TERMINATED/WITHDRAWN often never posts a ``resultsSection``, so
the extractor sees empty adverse-event arrays — yet the canonical safety signal
that killed the program usually lives only in a linked publication (e.g.
torcetrapib's BP / aldosterone / cardiac signal in PMID 17984165). This module
grounds that signal:

  trigger  ``needs_pubmed_enrichment(trial, extraction)`` — terminal status,
           **empty extracted adverse_events**, and >=1 reference PMID. This is the
           "the AE list came up empty" condition the in-build automated path will
           reuse verbatim.
  artifact ``data/annotations/<nct>_pubmed_safety.json`` — structured AEs +
           safety_signals extracted from the reference abstract(s), with PMID/DOI
           provenance. SAME schema regardless of who produced it.
  merge    ``merge_pubmed_safety`` / ``maybe_enrich_from_cache`` — folds the
           AEs/safety_signals into the extraction before ``_overlay_structured_aes``,
           so attribution emits ``causes_ae`` edges as if the trial had reported them.

The **producer is the scale seam** and is intentionally swappable:
  - today: curated agent-side via the PubMed MCP (small, high-value set —
    torcetrapib, vioxx, dimebon …);
  - at registry scale: an in-build 3rd call (tool-use that fetches the abstract
    when the extractor's AE list is empty, executed against NCBI E-utilities so it
    runs headless).
The trigger + schema + merge in this module are identical either way, so swapping
the producer never touches the build-side merge or attribution.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.annotation.taxonomy import StructuredAE, TrialExtraction

_ANNOTATIONS_DIR = Path("data/annotations")
# CT.gov overallStatus values that mean "stopped, likely no posted results".
_TERMINAL_STATUSES = frozenset({"TERMINATED", "WITHDRAWN", "SUSPENDED"})


class PubmedSafety(BaseModel):
    """Structured safety distilled from a trial's reference abstract(s).

    The committed cache artifact (one per enriched NCT). ``adverse_events`` are
    full ``StructuredAE`` so they flow straight into ``causes_ae`` attribution;
    ``safety_signals`` capture mechanism-level / continuous findings (BP,
    electrolytes, aldosterone) that aren't per-arm incidences. Provenance
    (``pmids`` / ``dois``) keeps every claim traceable to PubMed.
    """

    nct_id: str
    pmids: list[str] = Field(default_factory=list)
    dois: list[str] = Field(default_factory=list)
    adverse_events: list[StructuredAE] = Field(default_factory=list)
    safety_signals: list[str] = Field(default_factory=list)
    source: str = "pubmed_abstract"
    note: str = ""


def _safety_path(nct_id: str) -> Path:
    return _ANNOTATIONS_DIR / f"{nct_id}_pubmed_safety.json"


def needs_pubmed_enrichment(trial: object, extraction: object) -> bool:
    """True when a terminal-status trial has **no** extracted AEs but **does**
    have reference PMIDs — i.e. any safety signal lives only in the literature.

    This is the single trigger both producers share: the agent-curated path uses
    it to decide which trials to enrich; the future in-build path uses the same
    condition to decide whether to fire the 3rd (PubMed) call.
    """
    status = (getattr(trial, "status", "") or "").upper()
    if status not in _TERMINAL_STATUSES:
        return False
    if getattr(extraction, "adverse_events", None):
        return False  # already has structured/LLM AEs — nothing missing
    pmids = [
        r.pmid for r in (getattr(trial, "references", None) or [])
        if getattr(r, "pmid", "")
    ]
    return bool(pmids)


def load_pubmed_safety(nct_id: str) -> PubmedSafety | None:
    """Read the committed cache for a trial, or ``None`` if absent/unreadable."""
    path = _safety_path(nct_id)
    if not path.exists():
        return None
    try:
        return PubmedSafety.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return None


def merge_pubmed_safety(
    extraction: TrialExtraction, safety: PubmedSafety
) -> TrialExtraction:
    """Union the cached PubMed AEs + safety_signals into the extraction.

    Additive only — never overwrites existing AEs (the trigger already required
    the extraction's AE list to be empty, so this is a fill, not a replace)."""
    return extraction.model_copy(update={
        "adverse_events": list(extraction.adverse_events) + list(safety.adverse_events),
        "safety_signals": list(extraction.safety_signals) + list(safety.safety_signals),
    })


def maybe_enrich_from_cache(
    extraction: TrialExtraction, trial: object
) -> TrialExtraction:
    """Build-side hook: if this terminal / empty-AE trial has a committed PubMed
    safety artifact, merge it in. Pure read + union — no network, no LLM, so it
    is safe inside the headless build. A no-op (returns the extraction unchanged)
    when the trigger doesn't fire or no cache exists, so existing trials are
    untouched."""
    if not needs_pubmed_enrichment(trial, extraction):
        return extraction
    safety = load_pubmed_safety(getattr(trial, "nct_id", ""))
    if safety is None:
        return extraction
    return merge_pubmed_safety(extraction, safety)


def maybe_enrich_by_nct(extraction: TrialExtraction, nct_id: str) -> TrialExtraction:
    """Build-side enrichment for the **attribution** step, where no TrialRecord
    is in scope (attribution re-reads the raw extraction JSON, bypassing the
    extractor-side ``maybe_enrich_from_cache`` hook). Gates on the committed cache
    existing + the extraction's AE list being empty — the cache is only produced
    for trials that already passed ``needs_pubmed_enrichment`` (terminal status +
    empty AEs + PMIDs), so its existence implies the trigger was met.

    THIS is the integration point that actually lands the ``causes_ae`` edges:
    it must run before ``attribute_adverse_events`` in ``attributor._main``."""
    if getattr(extraction, "adverse_events", None):
        return extraction
    safety = load_pubmed_safety(nct_id)
    if safety is None:
        return extraction
    return merge_pubmed_safety(extraction, safety)


def write_pubmed_safety(safety: PubmedSafety) -> Path:
    """Persist a cache artifact (used by both producers)."""
    _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _safety_path(safety.nct_id)
    path.write_text(json.dumps(safety.model_dump(mode="json"), indent=2) + "\n")
    return path


# ── Scale producer: the in-build 3rd call (NCBI fetch + focused Anthropic) ────

_SAFETY_EXTRACT_SYSTEM = """You extract drug-attributable SAFETY findings from the \
abstract(s) of a clinical trial that was terminated/withdrawn and posted no \
structured results. Your output grounds the trial's safety signal in the literature.

Return ONLY JSON (no prose, no markdown fences):
{
  "adverse_events": [
    {
      "term": "<concise AE term, e.g. 'Death from any cause'>",
      "grade": "<CTCAE 1-5 if inferable else ''; death=5, life-threatening=4>",
      "serious": true,
      "incidence_treatment_pct": <number or null>,
      "incidence_control_pct": <number or null>,
      "hazard_ratio": <number or null>,
      "hr_ci_low": <number or null>,
      "hr_ci_high": <number or null>,
      "failure_causing": <true if the abstract names this AE as a reason the trial was stopped>
    }
  ],
  "safety_signals": ["<continuous/mechanistic findings not expressible as per-arm incidence, e.g. blood-pressure or electrolyte changes>"]
}

Rules:
- Only findings ATTRIBUTABLE TO THE STUDY DRUG (vs comparator/placebo). Skip baseline characteristics and efficacy endpoints.
- Prefer hazard/risk ratios + their 95% CI when reported — the strongest signal. Always include the CI bounds so significance can be judged.
- failure_causing=true for an AE the abstract names as a reason the trial stopped (e.g. "terminated because of increased death and cardiac events").
- If there are no drug-attributable safety findings, return {"adverse_events": [], "safety_signals": []}."""

_AE_FIELDS = (
    "term", "grade", "serious", "incidence_treatment_pct", "incidence_control_pct",
    "hazard_ratio", "hr_ci_low", "hr_ci_high", "failure_causing",
)


def _parse_json_object(raw: str) -> dict | None:
    s = (raw or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) >= 2 else s
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                return None
        return None


async def produce_pubmed_safety(
    anthropic_client, pubmed_client, trial, *, max_pmids: int = 3,
) -> PubmedSafety | None:
    """In-build scale producer (the dedicated 3rd call). Fetch the trial's top
    reference abstracts via NCBI, then a single-shot focused Anthropic call
    extracts structured safety (StructuredAEs w/ HR/CI/failure_causing +
    safety_signals). Returns the SAME ``PubmedSafety`` artifact the agent-curated
    path produces, or ``None`` if no abstracts / nothing extracted. Producer-
    agnostic downstream: the trigger + cache + merge don't care who wrote it."""
    pmids = [
        r.pmid for r in (getattr(trial, "references", None) or [])
        if getattr(r, "pmid", "")
    ][:max_pmids]
    if not pmids:
        return None
    abstracts = await pubmed_client.fetch_abstracts(pmids)
    if not abstracts:
        return None
    return await _extract_safety_from_abstracts(
        anthropic_client, getattr(trial, "nct_id", ""), abstracts,
    )


async def _extract_safety_from_abstracts(
    anthropic_client, nct_id: str, abstracts: dict[str, str],
) -> PubmedSafety | None:
    # Lazy import: extractor imports this module, so importing it at module level
    # would be circular.
    from src.annotation.extractor import MODEL, _call_messages_with_backoff

    blocks = "\n\n".join(f"=== PMID {pmid} ===\n{text}" for pmid, text in abstracts.items())
    user = (
        f"Trial: {nct_id}\nReference abstract(s):\n\n{blocks}\n\n"
        "Extract the drug-attributable safety findings as JSON per the schema."
    )
    resp = await _call_messages_with_backoff(
        anthropic_client, model=MODEL, max_tokens=1500,
        system=_SAFETY_EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", "") == "text"
    )
    data = _parse_json_object(raw)
    if not data:
        return None

    aes: list[StructuredAE] = []
    for a in data.get("adverse_events", []) or []:
        if not isinstance(a, dict) or not a.get("term"):
            continue
        try:
            aes.append(StructuredAE(**{
                k: a[k] for k in _AE_FIELDS if a.get(k) is not None
            }))
        except (TypeError, ValueError):
            continue
    signals = [s for s in (data.get("safety_signals") or []) if isinstance(s, str) and s.strip()]
    if not aes and not signals:
        return None
    return PubmedSafety(
        nct_id=nct_id, pmids=list(abstracts.keys()),
        adverse_events=aes, safety_signals=signals,
        source="pubmed_ncbi_3rd_call",
        note=(f"Auto-extracted in-build via NCBI efetch + focused Anthropic call "
              f"from PMIDs {list(abstracts.keys())}."),
    )
