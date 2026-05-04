"""AI-powered failure mode classifier using the Anthropic API."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

import anthropic

from src.annotation.attributor import AppliedEdgeUpdate, Attributor
from src.annotation.extractor import Extractor, _call_messages_with_backoff
from src.annotation.taxonomy import (
    FailureClassification,
    FailureMode,
    TrialExtraction,
)
from src.graph.models import TrialSubgraph
from src.ingestion.clinicaltrials import TrialRecord

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_ANNOTATIONS_DIR = Path("data/annotations")

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
MAX_RETRIES = 3


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


# ── Prompt formatting ────────────────────────────────────────────────────


def _format_classification_prompt(extraction: TrialExtraction) -> str:
    template = _load_prompt("classification_user.txt")

    biomarker_lines = []
    if extraction.biomarker_data:
        for k, v in extraction.biomarker_data.items():
            if isinstance(v, list):
                for item in v:
                    biomarker_lines.append(f"- {k}: {item}")
            elif v:
                biomarker_lines.append(f"- {k}: {v}")
    biomarker_str = "\n".join(biomarker_lines) if biomarker_lines else "No biomarker data available"

    safety_str = "\n".join(f"- {s}" for s in extraction.safety_signals) if extraction.safety_signals else "No safety signals reported"
    subgroup_str = "\n".join(f"- {s}" for s in extraction.subgroup_findings) if extraction.subgroup_findings else "No subgroup findings reported"

    return template.format(
        trial_id=extraction.trial_id,
        compound_name=extraction.compound_name or "Unknown",
        target_name=extraction.target_name or "Unknown",
        primary_endpoint=extraction.primary_endpoint or "Unknown",
        primary_endpoint_met=extraction.primary_endpoint_met,
        effect_size=extraction.effect_size,
        p_value=extraction.p_value,
        biomarker_data=biomarker_str,
        safety_signals=safety_str,
        subgroup_findings=subgroup_str,
        summary=extraction.summary,
    )


# ── Response parsing ─────────────────────────────────────────────────────


def _needs_expert_review(
    raw: dict[str, Any], extraction: TrialExtraction
) -> tuple[bool, str | None]:
    """Determine if expert review is needed beyond what Claude flagged."""
    confidence = raw.get("confidence_overall", 0.0)
    modes = raw.get("failure_modes", [])

    # Low overall confidence — flags the rubric's "no PD biomarker" tier (0.5-0.7)
    # and below, since clinical-trial evidence is weighted 5x in the inference layer
    if confidence < 0.7:
        return True, f"Low classification confidence ({confidence:.2f})"

    # Top 2 modes within 0.15 of each other
    if len(modes) >= 2:
        sorted_modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
        gap = sorted_modes[0].get("confidence", 0) - sorted_modes[1].get("confidence", 0)
        if gap < 0.15:
            return True, (
                f"Ambiguous: top modes {sorted_modes[0].get('mode')} "
                f"({sorted_modes[0].get('confidence'):.2f}) and "
                f"{sorted_modes[1].get('mode')} "
                f"({sorted_modes[1].get('confidence'):.2f}) within 0.15"
            )

    # Lacks biomarker + subgroup data
    has_biomarkers = bool(
        extraction.biomarker_data.get("biomarker_changes")
        or extraction.biomarker_data.get("target_engagement")
    )
    has_subgroups = bool(extraction.subgroup_findings)
    if not has_biomarkers and not has_subgroups:
        return True, "Trial lacks both biomarker and subgroup data"

    # Insufficient information mode
    for mode in modes:
        if mode.get("mode") == FailureMode.INSUFFICIENT_INFORMATION.value:
            return True, "Classified as insufficient_information"

    return False, None


def _parse_classification(
    raw: dict[str, Any], trial_id: str, extraction: TrialExtraction
) -> FailureClassification:
    modes = raw.get("failure_modes", [])
    if not modes:
        primary_mode = FailureMode.INSUFFICIENT_INFORMATION
        secondary = []
        confidence = 0.0
    else:
        sorted_modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
        primary_mode = FailureMode(sorted_modes[0]["mode"])
        secondary = [FailureMode(m["mode"]) for m in sorted_modes[1:]]
        confidence = raw.get("confidence_overall", sorted_modes[0].get("confidence", 0.5))

    # Apply our own review logic on top of Claude's
    needs_review, review_reason = _needs_expert_review(raw, extraction)
    if not needs_review:
        needs_review = raw.get("needs_expert_review", False)
        review_reason = raw.get("review_reason")

    evidence_quotes = [m.get("evidence", "") for m in modes if m.get("evidence")]

    classification = FailureClassification(
        trial_id=trial_id,
        primary_failure_mode=primary_mode,
        secondary_failure_modes=secondary,
        confidence=confidence,
        reasoning=raw.get("reasoning", ""),
        evidence_quotes=evidence_quotes,
    )

    # Stash extra fields for serialization
    classification._raw = raw  # type: ignore[attr-defined]
    classification._needs_review = needs_review  # type: ignore[attr-defined]
    classification._review_reason = review_reason  # type: ignore[attr-defined]
    return classification


# ── Classifier ───────────────────────────────────────────────────────────


class Classifier:
    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client
        self._system_prompt = _load_prompt("classification_system.txt")

    async def classify(self, extraction: TrialExtraction) -> FailureClassification:
        # Cache hit: load the saved classification and skip the LLM call.
        cache_path = _ANNOTATIONS_DIR / f"{extraction.trial_id}_classification.json"
        if cache_path.exists():
            try:
                cached_raw = json.loads(cache_path.read_text())
                return _parse_classification(cached_raw, extraction.trial_id, extraction)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Cached classification for %s unreadable (%s); re-classifying",
                    extraction.trial_id, exc,
                )

        user_message = _format_classification_prompt(extraction)
        raw_json = await self._call_with_retries(user_message, extraction.trial_id)
        classification = _parse_classification(raw_json, extraction.trial_id, extraction)
        self._save_annotation(extraction.trial_id, raw_json)
        return classification

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

    def _save_annotation(self, nct_id: str, raw_json: dict[str, Any]) -> None:
        _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _ANNOTATIONS_DIR / f"{nct_id}_classification.json"
        path.write_text(json.dumps(raw_json, indent=2))
        logger.debug("Saved classification to %s", path)


# ── End-to-end annotation ────────────────────────────────────────────────


async def annotate_trial(
    trial: TrialRecord,
    extractor: Extractor,
    classifier: Classifier,
    attributor: Attributor | None = None,
    build_subgraph: Callable[[TrialRecord, TrialExtraction], TrialSubgraph] | None = None,
) -> tuple[TrialExtraction, FailureClassification, list[AppliedEdgeUpdate]]:
    """Single annotation entry point: extract → classify → attribute → update.

    The fourth step (Bayesian belief update) is performed inside
    ``attributor.attribute`` via ``GraphStore.update_edge_belief``.

    If ``attributor`` or ``build_subgraph`` is omitted, the attribute step
    is skipped and an empty update list is returned.
    """
    extraction = await extractor.extract(trial)
    classification = await classifier.classify(extraction)
    updates: list[AppliedEdgeUpdate] = []
    if attributor is not None and build_subgraph is not None:
        subgraph = build_subgraph(trial, extraction)
        updates = attributor.attribute(classification, subgraph)
    return extraction, classification, updates


# ── CLI ──────────────────────────────────────────────────────────────────


async def _main(nct_id: str) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    client = anthropic.AsyncAnthropic(timeout=60.0)
    extractor = Extractor(client)
    classifier = Classifier(client)

    from src.ingestion.clinicaltrials import ClinicalTrialsClient

    console.print(f"[bold]Fetching {nct_id}...[/bold]")
    ct = ClinicalTrialsClient()
    trial = await ct.get_study(nct_id)
    console.print(f"  {trial.title}")
    console.print(f"  Phase {trial.phase} | {trial.status} | Results: {trial.has_results}")

    console.print("\n[bold]Extracting...[/bold]")
    extraction, classification, _ = await annotate_trial(trial, extractor, classifier)

    # Display extraction
    console.print(Panel(
        f"[bold]Compound:[/bold] {extraction.compound_name}\n"
        f"[bold]Target:[/bold] {extraction.target_name}\n"
        f"[bold]Endpoint:[/bold] {extraction.primary_endpoint}\n"
        f"[bold]Endpoint met:[/bold] {extraction.primary_endpoint_met}\n"
        f"[bold]Effect size:[/bold] {extraction.effect_size}\n"
        f"[bold]P-value:[/bold] {extraction.p_value}\n"
        f"[bold]Safety:[/bold] {', '.join(extraction.safety_signals) or 'None'}\n"
        f"[bold]Subgroups:[/bold] {'; '.join(extraction.subgroup_findings) or 'None'}",
        title="Extraction",
    ))

    # Display classification
    raw = getattr(classification, "_raw", {})
    needs_review = getattr(classification, "_needs_review", False)
    review_reason = getattr(classification, "_review_reason", None)
    outcome = raw.get("trial_outcome", "unknown")

    mode_lines = []
    for m in raw.get("failure_modes", []):
        mode_lines.append(f"  {m['mode']} (conf: {m.get('confidence', '?')}) — {m.get('evidence', '')[:80]}")

    edge_lines = []
    for e in raw.get("edges_to_update", []):
        edge_lines.append(
            f"  {e.get('direction', '?').upper()} {e.get('edge_type')}: "
            f"{e.get('source_entity')} → {e.get('target_entity')} "
            f"(mag: {e.get('magnitude', '?')})"
        )

    review_line = f"\n[bold red]NEEDS REVIEW:[/bold red] {review_reason}" if needs_review else ""

    console.print(Panel(
        f"[bold]Outcome:[/bold] {outcome}\n"
        f"[bold]Primary mode:[/bold] {classification.primary_failure_mode.value}\n"
        f"[bold]Confidence:[/bold] {classification.confidence:.2f}\n"
        f"\n[bold]Failure modes:[/bold]\n" + "\n".join(mode_lines) +
        f"\n\n[bold]Edge updates:[/bold]\n" + "\n".join(edge_lines) +
        f"\n\n[bold]Reasoning:[/bold] {classification.reasoning[:300]}..."
        f"{review_line}",
        title="Classification",
    ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a clinical trial's failure mode")
    parser.add_argument("--nct", required=True, help="NCT ID to classify")
    args = parser.parse_args()

    asyncio.run(_main(args.nct))
