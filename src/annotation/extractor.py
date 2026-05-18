"""AI-powered trial data extraction using the Anthropic API."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from pydantic import ValidationError

from src.annotation.taxonomy import (
    ArmIncidence,
    ChainResult,
    DoseInfo,
    ExtractedArm,
    ExtractedSubgroup,
    ModulationEntry,
    StructuredAE,
    TrialExtraction,
)
from src.ingestion.clinicaltrials import TrialRecord

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_ANNOTATIONS_DIR = Path("data/annotations")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
MAX_TOKENS_DOCUMENT = 16384  # medical reviews emit many trials per call
MAX_DOCUMENT_CHARS = 500_000  # ~125k tokens; truncate over this
MAX_RETRIES = 3
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_WAIT = 10.0  # seconds; doubled each attempt


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


async def _call_messages_with_backoff(
    client: anthropic.AsyncAnthropic, **kwargs: Any
) -> Any:
    """Wrap ``client.messages.create`` with exponential backoff on 429 / timeout / connection errors.

    Honors the ``retry-after`` response header when present; otherwise waits
    ``RATE_LIMIT_BASE_WAIT * 2**attempt`` seconds. Up to ``RATE_LIMIT_MAX_RETRIES``
    attempts before re-raising.
    """
    _transient = (
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
    )
    last_error: Exception | None = None
    for attempt in range(RATE_LIMIT_MAX_RETRIES):
        try:
            return await client.messages.create(**kwargs)
        except _transient as exc:
            last_error = exc
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            if isinstance(exc, anthropic.RateLimitError):
                response = getattr(exc, "response", None)
                if response is not None:
                    header_val = response.headers.get("retry-after")
                    if header_val:
                        try:
                            wait = float(header_val)
                        except ValueError:
                            pass
            logger.warning(
                "%s on attempt %d/%d; sleeping %.1fs before retry",
                type(exc).__name__, attempt + 1, RATE_LIMIT_MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)
    assert last_error is not None
    raise last_error


# ── Response parsing model (richer than TrialExtraction) ─────────────────

def _format_results_summary(results_summary: dict[str, Any] | None) -> str:
    """Format the ClinicalTrials.gov results section into readable text."""
    if not results_summary:
        return ""

    lines = ["## Posted Results Data"]

    # Outcome measures
    om_module = results_summary.get("outcomeMeasuresModule", {})
    for om in om_module.get("outcomeMeasures", []):
        om_type = om.get("type", "")
        title = om.get("title", "")
        lines.append(f"\n### {om_type} Outcome: {title}")

        if om.get("description"):
            lines.append(f"Definition: {om['description']}")
        if om.get("timeFrame"):
            lines.append(f"Time frame: {om['timeFrame']}")
        if om.get("paramType"):
            unit = om.get("unitOfMeasure", "")
            lines.append(f"Parameter: {om['paramType']} ({unit})")

        # Groups
        groups = {g["id"]: g["title"] for g in om.get("groups", [])}

        # Measurements
        for cls in om.get("classes", []):
            cls_title = cls.get("title", "")
            if cls_title:
                lines.append(f"  Subgroup: {cls_title}")
            for cat in cls.get("categories", []):
                for m in cat.get("measurements", []):
                    gname = groups.get(m.get("groupId", ""), m.get("groupId", ""))
                    val = m.get("value", "")
                    lo = m.get("lowerLimit", "")
                    hi = m.get("upperLimit", "")
                    ci_str = f" (95% CI: {lo}–{hi})" if lo and hi else ""
                    comment = f" [{m['comment']}]" if m.get("comment") else ""
                    lines.append(f"  {gname}: {val}{ci_str}{comment}")

        # Statistical analyses
        for analysis in om.get("analyses", []):
            method = analysis.get("statisticalMethod", "")
            param = analysis.get("paramType", "")
            value = analysis.get("paramValue", "")
            pval = analysis.get("pValue", "")
            ci_lo = analysis.get("ciLowerLimit", "")
            ci_hi = analysis.get("ciUpperLimit", "")
            ci_pct = analysis.get("ciPctValue", "")
            ci_str = f" ({ci_pct}% CI: {ci_lo}–{ci_hi})" if ci_lo and ci_hi else ""
            lines.append(f"  Analysis: {method}—{param} = {value}{ci_str}, p = {pval}")

    # Adverse events summary
    ae_module = results_summary.get("adverseEventsModule", {})
    serious = ae_module.get("seriousEvents", [])
    if serious:
        lines.append("\n### Serious Adverse Events")
        event_groups = {g["id"]: g["title"] for g in ae_module.get("eventGroups", [])}
        for event in serious[:10]:  # cap to keep prompt reasonable
            term = event.get("term", "")
            stats_list = event.get("stats", [])
            parts = []
            for s in stats_list:
                gname = event_groups.get(s.get("groupId", ""), "")
                affected = s.get("numAffected", "")
                at_risk = s.get("numAtRisk", "")
                if affected and at_risk:
                    parts.append(f"{gname}: {affected}/{at_risk}")
            if parts:
                lines.append(f"  {term}: {'; '.join(parts)}")

    return "\n".join(lines)


def _format_trial_for_prompt(trial: TrialRecord, abstract: str | None) -> str:
    """Fill the user prompt template with trial data."""
    template = _load_prompt("extraction_user.txt")

    interventions = "; ".join(
        f"{iv.name} ({iv.type}): {iv.description}" for iv in trial.interventions
    )
    primary = "; ".join(
        f"{o.measure} [{o.timeframe}]" for o in trial.primary_outcomes
    )
    secondary = "; ".join(
        f"{o.measure} [{o.timeframe}]" for o in trial.secondary_outcomes
    )
    conditions = ", ".join(trial.conditions)

    abstract_section = ""
    if abstract:
        abstract_section = f"## Abstract\n{abstract}"

    results_section = _format_results_summary(trial.results_summary)

    if trial.arm_groups:
        arm_lines = [
            "**These are the canonical arm_id values for this trial. "
            "Use them verbatim in `arms[].arm_id` and `results_by_chain[].arm_id` "
            "— DO NOT invent your own slugs.**",
            "",
        ]
        for ag in trial.arm_groups:
            ivs = ", ".join(ag.intervention_names) or "(no intervention names)"
            arm_lines.append(f"- `{ag.group_id}` — {ag.title}: {ivs}")
        arm_groups_section = "\n".join(arm_lines)
    else:
        arm_groups_section = "(no arm groups reported — emit arm_id as 'arm_1', 'arm_2', etc.)"

    if trial.subgroup_descriptors:
        subgroup_descriptors_section = "\n".join(
            f"- {s}" for s in trial.subgroup_descriptors
        )
    else:
        subgroup_descriptors_section = "(no subgroup stratifiers reported)"

    return template.format(
        nct_id=trial.nct_id,
        title=trial.title,
        phase=trial.phase,
        status=trial.status,
        conditions=conditions,
        interventions=interventions,
        primary_outcomes=primary,
        secondary_outcomes=secondary,
        enrollment=trial.enrollment or "Not reported",
        start_date=trial.start_date or "Not reported",
        completion_date=trial.completion_date or "Not reported",
        sponsor=trial.sponsor,
        has_results=trial.has_results,
        why_stopped=trial.why_stopped or "(not stated)",
        results_section=results_section,
        abstract_section=abstract_section,
        arm_groups_section=arm_groups_section,
        subgroup_descriptors_section=subgroup_descriptors_section,
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dose_info(raw: Any) -> DoseInfo:
    """Parse the structured dose_info block from the LLM response.

    Tolerates the legacy ``dose_information: str`` shape (older cached
    extractions) by stuffing the whole string into ``DoseInfo.dose``.
    """
    if isinstance(raw, dict):
        return DoseInfo(
            dose=str(raw.get("dose") or ""),
            schedule=str(raw.get("schedule") or ""),
            max_tolerated_dose=str(raw.get("max_tolerated_dose") or ""),
            dose_modifications=str(raw.get("dose_modifications") or ""),
        )
    if isinstance(raw, str) and raw:
        return DoseInfo(dose=raw)
    return DoseInfo()


def _build_aes_from_results_section(
    results_summary: dict[str, Any] | None,
    *,
    min_affected_any_arm: int = 3,
) -> list[StructuredAE]:
    """Build StructuredAE list directly from CT.gov adverseEventsModule.

    For each serious AE term, emits one StructuredAE with one
    ``ArmIncidence`` per reporting arm. AEs where no single arm has at
    least ``min_affected_any_arm`` affected patients are dropped — those
    can't clear the bucket thresholds in ``_ae_support_bucket`` anyway
    (weak_support requires ≥3 affected, moderate ≥5), so they only add
    ambiguous noise to the graph.

    Bypasses the LLM entirely on the structured path: the per-arm counts
    are exact CT.gov numbers, not LLM transcriptions. Trials without a
    results section fall through to the LLM-emitted flat ``tx/ctrl``
    pair via ``_parse_adverse_events``.
    """
    if not results_summary:
        return []
    ae_module = results_summary.get("adverseEventsModule", {})
    if not isinstance(ae_module, dict):
        return []
    event_groups = {
        g.get("id", ""): g.get("title", "")
        for g in ae_module.get("eventGroups", [])
        if isinstance(g, dict)
    }
    serious_events = ae_module.get("seriousEvents", [])
    if not isinstance(serious_events, list):
        return []

    out: list[StructuredAE] = []
    for event in serious_events:
        if not isinstance(event, dict):
            continue
        term = str(event.get("term") or "").strip()
        if not term:
            continue
        stats = event.get("stats", [])
        if not isinstance(stats, list):
            continue
        incidences: list[ArmIncidence] = []
        max_affected = 0
        for s in stats:
            if not isinstance(s, dict):
                continue
            descriptor = event_groups.get(s.get("groupId", ""), "").strip()
            if not descriptor:
                continue
            try:
                n_affected = int(s.get("numAffected") or 0)
                n_at_risk = int(s.get("numAtRisk") or 0)
            except (TypeError, ValueError):
                continue
            if n_at_risk <= 0:
                continue
            pct = 100.0 * n_affected / n_at_risk
            incidences.append(ArmIncidence(
                arm_descriptor=descriptor,
                n_affected=n_affected,
                n_at_risk=n_at_risk,
                pct=pct,
            ))
            max_affected = max(max_affected, n_affected)
        if max_affected < min_affected_any_arm:
            continue
        if not incidences:
            continue
        out.append(StructuredAE(
            term=term,
            grade="",
            arm_incidences=incidences,
            serious=True,
        ))
    return out


def _parse_arm_incidences(raw: Any) -> list[ArmIncidence]:
    """Parse cached arm_incidences from saved annotation JSON."""
    out: list[ArmIncidence] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        descriptor = str(entry.get("arm_descriptor") or "").strip()
        if not descriptor:
            continue
        try:
            out.append(ArmIncidence(
                arm_descriptor=descriptor,
                n_affected=int(entry.get("n_affected") or 0),
                n_at_risk=int(entry.get("n_at_risk") or 0),
                pct=_safe_float(entry.get("pct")),
            ))
        except (TypeError, ValueError):
            continue
    return out


def _parse_adverse_events(raw: Any) -> list[StructuredAE]:
    """Parse the LLM-emitted adverse_events list into StructuredAE models.

    Drops malformed entries (missing or empty term) silently—incidence
    rates can legitimately be missing when the report doesn't break them
    out per arm, but a term-less AE row is unusable for attribution.
    """
    out: list[StructuredAE] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        term = str(entry.get("term") or "").strip()
        if not term:
            continue
        out.append(
            StructuredAE(
                term=term,
                grade=str(entry.get("grade") or ""),
                incidence_treatment_pct=_safe_float(
                    entry.get("incidence_treatment_pct")
                ),
                incidence_control_pct=_safe_float(
                    entry.get("incidence_control_pct")
                ),
                arm_incidences=_parse_arm_incidences(entry.get("arm_incidences")),
                serious=bool(entry.get("serious") or False),
            )
        )
    return out


def _parse_extraction_response(raw_json: dict[str, Any], trial_id: str) -> TrialExtraction:
    """Map the rich Claude response to our TrialExtraction model."""
    hypothesis = raw_json.get("therapeutic_hypothesis", {})
    results = raw_json.get("results", {})
    context = raw_json.get("context", {})

    # Parse effect_size string to float if possible
    effect_size = None
    es_str = results.get("effect_size", "")
    if es_str:
        # Try to extract a leading number (e.g. "HR 0.73" -> 0.73, "2.3 months" -> 2.3)
        import re
        match = re.search(r"[-+]?\d*\.?\d+", es_str)
        if match:
            effect_size = float(match.group())

    arms_raw = raw_json.get("arms") or []
    arms: list[ExtractedArm] = []
    for a in arms_raw:
        arm_id = a.get("arm_id") or ""
        compounds = [c for c in (a.get("compounds") or []) if c]
        if not arm_id or not compounds:
            continue
        arms.append(ExtractedArm(arm_id=arm_id, compounds=compounds))

    subgroups_raw = raw_json.get("subgroups") or []
    subgroups: list[ExtractedSubgroup] = []
    for s in subgroups_raw:
        descriptor = s.get("raw_descriptor") or ""
        if not descriptor:
            continue
        # features arrive as list of {axis, key, level}; we keep them as
        # plain dicts here and let the populator canonicalize via
        # subgroup_taxonomy when building PopulationNodes.
        features = [
            {
                "axis": (f.get("axis") or "").strip(),
                "key": (f.get("key") or "").strip(),
                "level": (f.get("level") or "").strip(),
            }
            for f in (s.get("features") or [])
            if (f.get("axis") or "").strip() and (f.get("level") or "").strip()
        ]
        subgroups.append(ExtractedSubgroup(raw_descriptor=descriptor, features=features))

    # v0.3.0 modulation entries — defaulted to [] when the LLM omits the
    # field (e.g. single-compound trials, or extractions cached before this
    # field existed). Failed validation drops the entry rather than the
    # whole extraction, matching how subgroups/results handle bad rows.
    modulation_entries_raw = raw_json.get("modulation_entries") or []
    modulation_entries: list[ModulationEntry] = []
    for m in modulation_entries_raw:
        try:
            modulation_entries.append(ModulationEntry.model_validate(m))
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "Dropping invalid modulation_entry in %s: %s (%s)",
                trial_id, exc, m,
            )

    chain_results_raw = raw_json.get("results_by_chain") or []
    chain_results: list[ChainResult] = []
    for cr in chain_results_raw:
        arm_id = cr.get("arm_id") or ""
        if not arm_id:
            continue
        es = cr.get("effect_size")
        es_value: float | None = None
        if isinstance(es, (int, float)):
            es_value = float(es)
        elif isinstance(es, str) and es:
            import re
            match = re.search(r"[-+]?\d*\.?\d+", es)
            if match:
                es_value = float(match.group())
        chain_results.append(ChainResult(
            arm_id=arm_id,
            subgroup_descriptor=cr.get("subgroup_descriptor"),
            endpoint=cr.get("endpoint", "") or "",
            effect_size=es_value,
            p_value=cr.get("p_value"),
            outcome=(cr.get("outcome") or "unknown").lower(),
        ))

    # Structured dose_info supersedes the legacy results.dose_information
    # string; fall back to the old key for cached extractions written
    # before the schema change.
    dose_raw = results.get("dose_info")
    if dose_raw is None:
        dose_raw = results.get("dose_information")
    dose_info = _parse_dose_info(dose_raw)
    adverse_events = _parse_adverse_events(results.get("adverse_events"))

    return TrialExtraction(
        trial_id=raw_json.get("nct_id", trial_id),
        compound_name=hypothesis.get("compound", ""),
        target_name=hypothesis.get("claimed_target", ""),
        indication=context.get("indication", ""),
        phase="",
        primary_endpoint=hypothesis.get("primary_endpoint", ""),
        primary_endpoint_met=results.get("primary_endpoint_met"),
        effect_size=effect_size,
        p_value=results.get("p_value"),
        biomarker_data={
            "target_engagement": results.get("target_engagement_evidence"),
            "biomarker_changes": results.get("biomarker_changes", []),
        },
        safety_signals=results.get("safety_signals", []),
        subgroup_findings=results.get("subgroup_findings", []),
        summary=json.dumps(raw_json, indent=2),
        arms=arms,
        subgroups=subgroups,
        results_by_chain=chain_results,
        duration_weeks=_safe_float(context.get("duration_weeks")),
        dose_info=dose_info,
        adverse_events=adverse_events,
        modulation_entries=modulation_entries,
    )


def _parse_document_response(
    raw_json: dict[str, Any], doc_id: str
) -> list[TrialExtraction]:
    """Map a multi-trial document response to a list of TrialExtraction."""
    drug = raw_json.get("drug", {}) or {}
    compound = drug.get("compound", "")
    target = drug.get("claimed_target", "")

    extractions: list[TrialExtraction] = []
    for i, trial in enumerate(raw_json.get("trials", []) or []):
        trial_id = trial.get("trial_id") or f"{doc_id}_trial_{i}"

        effect_size: float | None = None
        es_str = trial.get("effect_size", "") or ""
        if es_str:
            import re
            match = re.search(r"[-+]?\d*\.?\d+", es_str)
            if match:
                effect_size = float(match.group())

        dose_raw = trial.get("dose_info")
        if dose_raw is None:
            dose_raw = trial.get("dose_information")
        dose_info = _parse_dose_info(dose_raw)
        adverse_events = _parse_adverse_events(trial.get("adverse_events"))

        extractions.append(
            TrialExtraction(
                trial_id=trial_id,
                compound_name=trial.get("compound") or compound,
                target_name=target,
                indication=trial.get("indication", ""),
                phase=str(trial.get("phase", "")),
                primary_endpoint=trial.get("primary_endpoint", ""),
                primary_endpoint_met=trial.get("primary_endpoint_met"),
                effect_size=effect_size,
                p_value=trial.get("p_value"),
                biomarker_data={
                    "target_engagement": trial.get("target_engagement_evidence"),
                    "biomarker_changes": trial.get("biomarker_changes", []),
                },
                safety_signals=trial.get("safety_signals", []) or [],
                subgroup_findings=trial.get("subgroup_findings", []) or [],
                summary=json.dumps(
                    {
                        "application_number": raw_json.get("application_number"),
                        "drug": drug,
                        "trial": trial,
                    },
                    indent=2,
                ),
                duration_weeks=_safe_float(trial.get("duration_weeks")),
                dose_info=dose_info,
                adverse_events=adverse_events,
            )
        )
    return extractions


def _overlay_structured_aes(
    extraction: TrialExtraction, trial: TrialRecord
) -> TrialExtraction:
    """Replace LLM-emitted AEs with structured per-arm AEs when available.

    CT.gov's adverseEventsModule has exact per-arm counts that the LLM
    would otherwise have to transcribe (and frequently flattens or
    truncates). Whenever a trial has structured AE data, we use it
    directly and discard the LLM's AE list. Trials without a results
    section keep the LLM-emitted flat tx/ctrl pairs.
    """
    structured = _build_aes_from_results_section(trial.results_summary)
    if not structured:
        return extraction
    return extraction.model_copy(update={"adverse_events": structured})


# ── Extractor ────────────────────────────────────────────────────────────


class Extractor:
    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client
        self._system_prompt = _load_prompt("extraction_system.txt")
        self._document_system_prompt = _load_prompt("extraction_document_system.txt")

    async def extract(
        self, trial: TrialRecord, abstract: str | None = None
    ) -> TrialExtraction:
        # Cache hit: load saved extraction and skip the LLM call entirely.
        cache_path = _ANNOTATIONS_DIR / f"{trial.nct_id}_extraction.json"
        if cache_path.exists():
            try:
                cached_raw = json.loads(cache_path.read_text())
                extraction = _parse_extraction_response(cached_raw, trial.nct_id)
                return _overlay_structured_aes(extraction, trial)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Cached extraction for %s unreadable (%s); re-extracting",
                    trial.nct_id, exc,
                )

        user_message = _format_trial_for_prompt(trial, abstract)
        raw_json = await self._call_with_retries(user_message, trial.nct_id)
        extraction = _parse_extraction_response(raw_json, trial.nct_id)
        # Recall fix: results were posted but Claude declined to commit on the
        # binary call. Ask once more with a focused yes/no prompt.
        if (
            trial.results_summary is not None
            and extraction.primary_endpoint_met is None
        ):
            extraction = await self._reask_endpoint_met(trial, extraction, raw_json)
        self._save_annotation(trial.nct_id, raw_json)
        return _overlay_structured_aes(extraction, trial)

    async def _reask_endpoint_met(
        self,
        trial: TrialRecord,
        extraction: TrialExtraction,
        raw_json: dict[str, Any],
    ) -> TrialExtraction:
        results_text = _format_results_summary(trial.results_summary)
        endpoint_label = extraction.primary_endpoint or "the primary endpoint"
        user_msg = (
            f"Trial: {trial.nct_id}\n"
            f"Primary endpoint: {endpoint_label}\n\n"
            f"{results_text}\n\n"
            "Question: Given these results, was the primary endpoint statistically "
            "significant? Return true, false, or null only if genuinely "
            "indeterminate. Reply with one word and nothing else."
        )
        try:
            response = await _call_messages_with_backoff(
                self._client,
                model=MODEL,
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception:
            logger.warning(
                "Reask failed for %s; leaving primary_endpoint_met null",
                trial.nct_id,
                exc_info=True,
            )
            return extraction
        text = response.content[0].text.strip().lower().rstrip(".")
        if text == "true":
            new_value: bool | None = True
        elif text == "false":
            new_value = False
        else:
            return extraction
        # Keep the saved annotation in sync with the corrected field.
        raw_json.setdefault("results", {})["primary_endpoint_met"] = new_value
        raw_json.setdefault("results", {})["primary_endpoint_met_source"] = "reask"
        return extraction.model_copy(update={"primary_endpoint_met": new_value})

    async def extract_batch(
        self, trials: list[TrialRecord], concurrency: int = 5
    ) -> list[TrialExtraction]:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_extract(trial: TrialRecord) -> TrialExtraction | None:
            async with semaphore:
                try:
                    return await self.extract(trial)
                except Exception:
                    logger.error(
                        "Extraction failed for %s", trial.nct_id, exc_info=True
                    )
                    return None

        results = await asyncio.gather(*(bounded_extract(t) for t in trials))
        return [r for r in results if r is not None]

    async def _call_with_retries(
        self, user_message: str, trial_id: str
    ) -> dict[str, Any]:
        last_error: str | None = None

        for attempt in range(MAX_RETRIES):
            messages = [{"role": "user", "content": user_message}]
            if last_error:
                messages.append({
                    "role": "assistant",
                    "content": "I'll fix the JSON formatting issue.",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response had a validation error:\n"
                        f"{last_error}\n\n"
                        f"Please return corrected JSON only."
                    ),
                })

            response = await _call_messages_with_backoff(
                self._client,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=self._system_prompt,
                messages=messages,
            )

            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3].strip()

            try:
                parsed = json.loads(text)
                return parsed
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                logger.warning(
                    "Attempt %d/%d for %s: %s",
                    attempt + 1, MAX_RETRIES, trial_id, last_error,
                )

        raise ValueError(
            f"Failed to get valid JSON for {trial_id} after {MAX_RETRIES} attempts"
        )

    async def extract_from_document(
        self,
        document_text: str,
        doc_id: str,
        source_url: str = "",
    ) -> list[TrialExtraction]:
        """Extract trial records from an unstructured document (e.g. an FDA medical review).

        ``doc_id`` is used as both the application number passed to the model and
        the cache key (typically the FDA application number plus PDF basename, so
        an application with multiple review PDFs caches separately).
        """
        cache_path = _ANNOTATIONS_DIR / f"{doc_id}_document.json"
        if cache_path.exists():
            try:
                cached_raw = json.loads(cache_path.read_text())
                return _parse_document_response(cached_raw, doc_id)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Cached document extraction for %s unreadable (%s); re-extracting",
                    doc_id, exc,
                )

        if not document_text.strip():
            logger.warning("Empty document text for %s; skipping", doc_id)
            return []

        truncated = document_text
        if len(truncated) > MAX_DOCUMENT_CHARS:
            logger.warning(
                "Document %s is %d chars; truncating to %d",
                doc_id, len(truncated), MAX_DOCUMENT_CHARS,
            )
            truncated = truncated[:MAX_DOCUMENT_CHARS]

        template = _load_prompt("extraction_document_user.txt")
        user_message = template.format(
            application_number=doc_id,
            source_url=source_url or "(local PDF)",
            document_text=truncated,
        )

        raw_json = await self._call_document_with_retries(user_message, doc_id)
        self._save_document_annotation(doc_id, raw_json)
        return _parse_document_response(raw_json, doc_id)

    async def _call_document_with_retries(
        self, user_message: str, doc_id: str
    ) -> dict[str, Any]:
        last_error: str | None = None

        for attempt in range(MAX_RETRIES):
            messages = [{"role": "user", "content": user_message}]
            if last_error:
                messages.append({
                    "role": "assistant",
                    "content": "I'll fix the JSON formatting issue.",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response had a validation error:\n"
                        f"{last_error}\n\n"
                        f"Please return corrected JSON only."
                    ),
                })

            response = await _call_messages_with_backoff(
                self._client,
                model=MODEL,
                max_tokens=MAX_TOKENS_DOCUMENT,
                temperature=0,
                system=self._document_system_prompt,
                messages=messages,
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3].strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                logger.warning(
                    "Document attempt %d/%d for %s: %s",
                    attempt + 1, MAX_RETRIES, doc_id, last_error,
                )

        raise ValueError(
            f"Failed to get valid JSON for document {doc_id} after {MAX_RETRIES} attempts"
        )

    def _save_document_annotation(self, doc_id: str, raw_json: dict[str, Any]) -> None:
        _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _ANNOTATIONS_DIR / f"{doc_id}_document.json"
        path.write_text(json.dumps(raw_json, indent=2))
        logger.debug("Saved document extraction to %s", path)

    def _save_annotation(self, nct_id: str, raw_json: dict[str, Any]) -> None:
        _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _ANNOTATIONS_DIR / f"{nct_id}_extraction.json"
        path.write_text(json.dumps(raw_json, indent=2))
        logger.debug("Saved extraction to %s", path)
