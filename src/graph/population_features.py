"""Round-17 LLM population-feature extractor.

The CT.gov ``conditions`` field + the deterministic
``extract_indication_qualifiers`` regex catch staging, histology, and
extent qualifiers from condition strings. They miss the population
context that lives in trial eligibility criteria + title — line of
therapy, prior treatment history, biomarker requirements, mutation
status. Round-16's responds_differently always-emit rule pushed the
classifier to write per-population evidence, but for most trials the
populator only created a `melanoma__unselected` population because
none of the available signal lived in conditions text. Result: the
trial-level evidence pooled into one shared edge.

This module fills that gap. It runs an LLM extraction per-trial,
cached on disk, that pulls structured features from title + conditions
+ eligibility, then converts them into `SubgroupFeature(axis, key,
level)` values that compose with the deterministic regex output into
a more-specific `PopulationNode.compose_id` slug.

Output schema (one LLM call per trial):

    {
      "line_of_therapy": "first" | "second" | "third_plus" |
                         "adjuvant" | "neoadjuvant" | null,
      "prior_treatments": ["anti_pd1", "anti_ctla4", "chemotherapy",
                           "targeted_therapy", "radiation", ...],
      "required_mutations": [{"gene": "BRAF", "variant": "v600e"}, ...],
      "biomarker_selection": [{"gene": "CD274", "level": "high"}, ...],
      "disease_stage": "iii" | "iv" | "metastatic" | "resectable" | ...
    }

Each non-null/non-empty entry produces one SubgroupFeature. The result
merges with `extract_indication_qualifiers` output; duplicates are
silently de-duped at PopulationNode.compose_id time (slug sort).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import anthropic

from src.annotation.extractor import _call_messages_with_backoff
from src.graph.models import SubgroupFeature

logger = logging.getLogger(__name__)

# Sonnet is overkill for this single-shot structured extraction; Haiku
# is consistent with the existing endpoint / indication canonicalization
# LLM calls in populate.py.
_INFERENCE_MODEL = "claude-haiku-4-5-20251001"

_PROMPT_TEMPLATE = """Given a clinical trial's title, conditions, and eligibility criteria, extract the patient-population features that distinguish this trial's enrollment from the general disease population.

TITLE: {title}

CONDITIONS: {conditions}

ELIGIBILITY CRITERIA:
{eligibility}

Return ONLY a JSON object with these keys (use null or empty list when not specified):

  "line_of_therapy": one of "first", "second", "third_plus", "adjuvant", "neoadjuvant", or null
  "prior_treatments": list of strings from {{"anti_pd1", "anti_ctla4", "anti_pdl1", "chemotherapy", "targeted_therapy", "radiation", "surgery", "ifn_alpha", "il2", "vaccine", "cell_therapy"}}; empty list if unspecified
  "required_mutations": list of objects {{"gene": "<UPPER_HUGO>", "variant": "<lowercase_variant>"}}; e.g. {{"gene": "BRAF", "variant": "v600e"}}, {{"gene": "KRAS", "variant": "mutant"}}; empty list if unspecified
  "biomarker_selection": list of objects {{"gene": "<UPPER_HUGO>", "level": "<lowercase_level>"}}; e.g. {{"gene": "CD274", "level": "high"}}, {{"gene": "MSI", "level": "high"}}; empty list if unspecified
  "disease_stage": one of "iii", "iv", "metastatic", "resectable", "unresectable", "locally_advanced", or null

Rules:
- Be conservative. Only extract features that are EXPLICIT inclusion or exclusion criteria, not features mentioned in passing in the title or description.
- "line_of_therapy" must come from the eligibility criteria specifying treatment history (e.g. "treatment-naive" → first; "progressed on anti-PD-1" → second; "as adjuvant therapy following resection" → adjuvant).
- "prior_treatments" is set when eligibility EXPLICITLY requires or permits patients with that prior therapy. Do NOT infer from study arms or interventions in the trial.
- "required_mutations" is set when eligibility EXPLICITLY requires a mutation as inclusion criterion (e.g. "BRAF V600E mutation required").
- "biomarker_selection" is set when eligibility EXPLICITLY requires a biomarker status (e.g. "PD-L1 expression ≥1%" → {{"gene": "CD274", "level": "high"}}).
- "disease_stage" is set when eligibility or condition explicitly limits stage; default to null when the condition just says "Melanoma" without staging.
- When unsure, prefer null / empty over guessing.

Reply with ONLY the JSON object. No prose, no markdown fences.
"""


# Default level strings on SubgroupFeature.axis="line"/"prior_tx"/etc.
# Kept here so the feature builder produces consistent slugs.
_LINE_LEVELS = {
    "first": "first",
    "second": "second",
    "third_plus": "third_plus",
    "adjuvant": "adjuvant",
    "neoadjuvant": "neoadjuvant",
}
_STAGE_LEVELS = {
    "iii": "iii",
    "iv": "iv",
    "metastatic": "metastatic",
    "resectable": "resectable",
    "unresectable": "unresectable",
    "locally_advanced": "locally_advanced",
}
_PRIOR_TX_ALLOWED = {
    "anti_pd1", "anti_ctla4", "anti_pdl1", "chemotherapy",
    "targeted_therapy", "radiation", "surgery", "ifn_alpha",
    "il2", "vaccine", "cell_therapy",
}


def _features_from_llm_response(raw: dict[str, Any]) -> list[SubgroupFeature]:
    """Convert the LLM's structured JSON into a list of SubgroupFeatures.

    Empty/unknown axes produce no entries. Invalid level strings are
    silently dropped (the slug shouldn't accept arbitrary LLM text).
    """
    features: list[SubgroupFeature] = []

    line = (raw.get("line_of_therapy") or "").strip().lower()
    if line in _LINE_LEVELS:
        features.append(SubgroupFeature(
            axis="line", level=_LINE_LEVELS[line],
        ))

    stage = (raw.get("disease_stage") or "").strip().lower()
    if stage in _STAGE_LEVELS:
        # Use "extent" axis for metastatic/resectable/unresectable to
        # match extract_indication_qualifiers' existing axis vocab; use
        # "stage" axis for Roman-numeral stages. Keeps the two
        # extractors composable without duplicate axes.
        if stage in ("metastatic", "resectable", "unresectable", "locally_advanced"):
            features.append(SubgroupFeature(axis="extent", level=stage))
        else:
            features.append(SubgroupFeature(axis="stage", level=stage))

    for tx in (raw.get("prior_treatments") or []):
        tx_clean = re.sub(r"[^a-z0-9_]+", "_", str(tx).lower()).strip("_")
        if tx_clean in _PRIOR_TX_ALLOWED:
            features.append(SubgroupFeature(
                axis="prior_tx", level=f"{tx_clean}_treated",
            ))

    for mut in (raw.get("required_mutations") or []):
        gene = (mut.get("gene") or "").strip().upper()
        variant = re.sub(
            r"[^a-z0-9]+", "", str(mut.get("variant") or "").lower(),
        )
        if gene and variant:
            features.append(SubgroupFeature(
                axis="gene", key=gene, level=variant,
            ))

    for bio in (raw.get("biomarker_selection") or []):
        gene = (bio.get("gene") or "").strip().upper()
        level = re.sub(
            r"[^a-z0-9]+", "", str(bio.get("level") or "").lower(),
        )
        if gene and level:
            features.append(SubgroupFeature(
                axis="gene", key=gene, level=level,
            ))

    return features


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of the LLM response.

    Models occasionally wrap output in markdown fences despite the
    "no markdown fences" instruction; strip them defensively.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` fence.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to locate the first {...} block.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


async def extract_population_features_with_llm(
    *,
    nct_id: str,
    title: str,
    conditions: list[str],
    eligibility_criteria: str,
    client: anthropic.AsyncAnthropic | None,
    cache_path: Path,
    max_eligibility_chars: int = 4000,
) -> list[SubgroupFeature]:
    """Round-17 LLM extractor for population features.

    Cached on disk per nct_id. Reads title + conditions + eligibility
    (truncated to ``max_eligibility_chars`` to keep the prompt bounded
    for very long eligibility blocks) and returns a list of
    SubgroupFeature values usable directly in PopulationNode.compose_id.

    Returns an empty list (a) when the client is None (offline / test
    mode), (b) when the LLM returns malformed JSON, or (c) when the
    eligibility text is empty AND the conditions text has no signal
    the regex extractor doesn't already pick up.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict[str, Any]]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            logger.warning(
                "population_features cache %s unreadable; starting empty",
                cache_path,
            )
            cache = {}

    if nct_id in cache:
        cached_list = cache[nct_id]
        return [SubgroupFeature.model_validate(f) for f in cached_list]

    if client is None:
        return []

    elig = (eligibility_criteria or "").strip()
    if not elig and not conditions:
        return []
    if len(elig) > max_eligibility_chars:
        elig = elig[:max_eligibility_chars] + "\n\n[...truncated]"

    prompt = _PROMPT_TEMPLATE.format(
        title=title or "(none)",
        conditions=", ".join(conditions) if conditions else "(none)",
        eligibility=elig or "(none)",
    )

    try:
        response = await _call_messages_with_backoff(
            client,
            model=_INFERENCE_MODEL,
            max_tokens=800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
    except Exception:  # noqa: BLE001
        logger.exception(
            "population_features LLM call failed for %s; returning empty",
            nct_id,
        )
        return []

    parsed = _parse_llm_json(raw_text)
    if parsed is None:
        logger.warning(
            "population_features LLM returned unparseable JSON for %s: %r",
            nct_id, raw_text[:200],
        )
        cache[nct_id] = []
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        return []

    features = _features_from_llm_response(parsed)
    cache[nct_id] = [f.model_dump() for f in features]
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return features
