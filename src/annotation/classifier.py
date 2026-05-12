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
from src.annotation.meddra import MeddraCache
from src.annotation.taxonomy import (
    FailureClassification,
    FailureMode,
    TrialExtraction,
)
from src.graph.models import TrialSubgraph
from src.graph.store import GraphStore
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


def _format_trial_entities(
    graph: GraphStore | None,
    trial_subgraph: TrialSubgraph | None,
) -> str:
    """Render the trial's resolved graph node IDs as a per-constituent block.

    A combo arm has one causal chain per constituent compound, so each
    constituent appears as its own ``compound → target → mechanism →
    biology`` row grouped under the arm. The classifier can then emit
    per-constituent ``binds_to`` / ``modulates_via`` /
    ``mechanism_affects`` / ``biology_drives`` updates that route to the
    matching constituent chain.

    Indication, endpoints, and populations are shared across arms (same
    disease, same measurement windows, same enrollment) and printed once.
    Falls back to a single-line stub when no graph context is available
    (offline tests).
    """
    if graph is None or trial_subgraph is None or not trial_subgraph.chains:
        return "(no graph context—emit edges_to_update only when confident in entity names)"

    def _annotate(node_id: str, hint_attr: str = "") -> str:
        if node_id == "UNKNOWN":
            return "UNKNOWN"
        try:
            n = graph.get_node(node_id)
        except KeyError:
            return node_id
        if hint_attr and n.get(hint_attr):
            return f"{node_id} ({n[hint_attr]})"
        name = n.get("name")
        if name and name != node_id:
            return f"{node_id} ({name})"
        return node_id

    chains = trial_subgraph.chains
    arm_by_id = {a.arm_id: a for a in trial_subgraph.arms}

    # Shared trial-level entities (one indication, one endpoint set, one
    # population set; same for every chain).
    sample = chains[0]
    endpoints: list[str] = []
    populations: list[str] = []
    seen_ep: set[str] = set()
    seen_pop: set[str] = set()
    for c in chains:
        if c.endpoint_id and c.endpoint_id not in seen_ep:
            seen_ep.add(c.endpoint_id)
            endpoints.append(c.endpoint_id)
        if c.subgroup_population_id and c.subgroup_population_id not in seen_pop:
            seen_pop.add(c.subgroup_population_id)
            populations.append(c.subgroup_population_id)

    # Per-(arm, constituent) backbone: pick the first chain per
    # (arm_id, compound_id) pair. Subgroup forks share the same backbone,
    # so the first occurrence captures the full chain for prompt purposes.
    arm_constituent_chain: dict[tuple[str, str], Any] = {}
    for c in chains:
        key = (c.arm_id, c.compound_id)
        if key not in arm_constituent_chain:
            arm_constituent_chain[key] = c

    lines = [
        "Shared trial-level entities (USE THESE EXACT IDs — do not add stage/refractory/metastatic/advanced qualifiers):",
        f"- indication:    `{sample.indication_id}`  ← use this id verbatim for biology_drives target, endpoint_captures target, responds_differently target",
        f"- endpoints:     {', '.join('`'+e+'`' for e in endpoints) if endpoints else 'UNKNOWN'}  ← use these ids verbatim for endpoint_captures source and reflects_biology target",
        f"- populations:   {', '.join('`'+p+'`' for p in populations) if populations else 'UNKNOWN'}  ← use these ids verbatim for responds_differently source",
        "",
        "Per-arm causal chains (use these canonical ids in edges_to_update):",
    ]

    # Group constituent chains under their parent arm so combo arms are
    # visually clustered.
    seen_arms: set[str] = set()
    for (arm_id, _), chain in arm_constituent_chain.items():
        if arm_id in seen_arms:
            continue
        seen_arms.add(arm_id)
        arm = arm_by_id.get(arm_id)
        kind = (
            "combo" if (arm is not None and arm.is_combination) else "monotherapy"
        )
        lines.append(f"- {arm_id} [{kind}]:")
        for (a_id, _), c in arm_constituent_chain.items():
            if a_id != arm_id:
                continue
            comp = _annotate(c.compound_id)
            tgt = _annotate(c.target_id, "gene_symbol")
            mech = _annotate(c.mechanism_id)
            bio = _annotate(c.biology_id)
            lines.append(f"    {comp} → {tgt} → {mech} → {bio}")

    return "\n".join(lines)


def _format_classification_prompt(
    extraction: TrialExtraction,
    *,
    graph: GraphStore | None = None,
    trial_subgraph: TrialSubgraph | None = None,
) -> str:
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

    if extraction.adverse_events:
        ae_lines = []
        for ae in extraction.adverse_events:
            grade_part = f", grade {ae.grade}" if ae.grade else ""
            sae_part = " [SAE]" if ae.serious else ""
            if ae.arm_incidences:
                # Per-arm rates from CT.gov — preserve arm identity so the
                # classifier can spot 3-arm signals the flat tx/ctrl
                # collapse would hide (e.g. monotherapy-vs-combo).
                arm_parts = [
                    f"{ai.arm_descriptor}: {ai.n_affected}/{ai.n_at_risk}"
                    f" ({ai.pct:g}%)" if ai.pct is not None else
                    f"{ai.arm_descriptor}: {ai.n_affected}/{ai.n_at_risk}"
                    for ai in ae.arm_incidences
                ]
                ae_lines.append(
                    f"- {ae.term}{grade_part}: {'; '.join(arm_parts)}{sae_part}"
                )
            else:
                tx = (
                    f"{ae.incidence_treatment_pct:g}%"
                    if ae.incidence_treatment_pct is not None else "?"
                )
                ctrl = (
                    f"{ae.incidence_control_pct:g}%"
                    if ae.incidence_control_pct is not None else "?"
                )
                ae_lines.append(
                    f"- {ae.term}{grade_part}: tx {tx} vs ctrl {ctrl}{sae_part}"
                )
        ae_str = "\n".join(ae_lines)
    else:
        ae_str = "No structured adverse-event data"

    dose = extraction.dose_info
    dose_lines = []
    if dose.dose:
        dose_lines.append(f"- Dose: {dose.dose}")
    if dose.schedule:
        dose_lines.append(f"- Schedule: {dose.schedule}")
    if dose.max_tolerated_dose:
        dose_lines.append(f"- MTD: {dose.max_tolerated_dose}")
    if dose.dose_modifications:
        dose_lines.append(f"- Modifications: {dose.dose_modifications}")
    if extraction.duration_weeks is not None:
        dose_lines.append(f"- Duration: {extraction.duration_weeks:g} weeks")
    dose_str = "\n".join(dose_lines) if dose_lines else "No dose / duration data"

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
        adverse_events=ae_str,
        dose_info=dose_str,
        subgroup_findings=subgroup_str,
        summary=extraction.summary,
        graph_entities=_format_trial_entities(graph, trial_subgraph),
    )


# ── Response parsing ─────────────────────────────────────────────────────


def _needs_expert_review(
    raw: dict[str, Any], extraction: TrialExtraction
) -> tuple[bool, str | None]:
    """Determine if expert review is needed beyond what Claude flagged."""
    confidence = raw.get("confidence_overall", 0.0)
    modes = raw.get("failure_modes", [])

    # Low overall confidence—flags the rubric's "no PD biomarker" tier (0.5-0.7)
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
    trial_outcome = raw.get("trial_outcome", "")
    if not modes:
        primary_mode = FailureMode.INSUFFICIENT_INFORMATION
        secondary = []
        # Use the LLM's confidence_overall regardless of outcome. Failure
        # trials with empty failure_modes were previously zeroed, which
        # also zeroed the quality_score on every emitted contradict edge
        # — so a clearly missed primary endpoint contributed nothing to
        # the graph. The per-edge bucket already encodes contradict
        # strength; confidence just discounts the trial's overall
        # certainty in its mechanistic attribution. Keep it.
        confidence = float(raw.get("confidence_overall", 0.5))
    else:
        sorted_modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
        primary_mode = FailureMode(sorted_modes[0]["mode"])
        secondary = [FailureMode(m["mode"]) for m in sorted_modes[1:]]
        confidence = float(raw.get(
            "confidence_overall", sorted_modes[0].get("confidence", 0.5),
        ))

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

    async def classify(
        self,
        extraction: TrialExtraction,
        *,
        graph: GraphStore | None = None,
        trial_subgraph: TrialSubgraph | None = None,
    ) -> FailureClassification:
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

        user_message = _format_classification_prompt(
            extraction, graph=graph, trial_subgraph=trial_subgraph,
        )
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
    meddra_cache: MeddraCache | None = None,
) -> tuple[TrialExtraction, FailureClassification, list[AppliedEdgeUpdate]]:
    """Single annotation entry point: extract → classify → attribute → update.

    Belief updates land via ``GraphStore.update_edge_belief`` inside both
    the classifier-driven ``attributor.attribute`` (efficacy edges) and
    the structured ``attributor.attribute_adverse_events`` (causes_ae +
    target_associated_ae propagation). The returned ``updates`` list is
    the concatenation of both.

    If ``attributor`` or ``build_subgraph`` is omitted, both attribute
    steps are skipped and an empty update list is returned.
    """
    extraction = await extractor.extract(trial)
    classification = await classifier.classify(extraction)
    updates: list[AppliedEdgeUpdate] = []
    if attributor is not None and build_subgraph is not None:
        subgraph = build_subgraph(trial, extraction)
        updates = attributor.attribute(classification, subgraph)
        # Reuse the extractor's Anthropic client for MedDRA normalization;
        # the cache (default at data/cache/meddra_terms.json) collapses
        # repeat AE terms across the corpus.
        if extraction.adverse_events:
            ae_updates = await attributor.attribute_adverse_events(
                subgraph,
                extraction,
                client=extractor._client,
                meddra_cache=meddra_cache or MeddraCache(),
            )
            updates.extend(ae_updates)
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
        mode_lines.append(f"  {m['mode']} (conf: {m.get('confidence', '?')})—{m.get('evidence', '')[:80]}")

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
