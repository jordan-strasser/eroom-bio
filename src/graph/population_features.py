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

Look for ANY clinically-relevant qualifier in the eligibility criteria that defines or stratifies the enrolled group of patients — this is disease-AGNOSTIC. In oncology that's line of therapy / stage / mutation; in autoimmune/rheumatology it's disease activity (active vs remission), seromarkers (rheumatoid factor, anti-CCP), prior DMARD/biologic; in cardiology it's ejection fraction (LVEF), NYHA class, prior MI; in metabolism it's HbA1c / LDL thresholds; in neurology it's disease stage / biomarker (amyloid). Capture whatever patient-selection criterion the trial actually used.

TITLE: {title}

CONDITIONS: {conditions}

ELIGIBILITY CRITERIA:
{eligibility}

Return ONLY a JSON object with these keys (use null or empty list when not specified):

  "line_of_therapy": one of "first", "second", "third_plus", "adjuvant", "neoadjuvant", or null
  "prior_treatments": list of strings from {{"anti_pd1", "anti_ctla4", "anti_pdl1", "chemotherapy", "targeted_therapy", "radiation", "surgery", "ifn_alpha", "il2", "vaccine", "cell_therapy", "dmards", "biologics", "tnf_inhibitor", "steroids", "insulin", "statin"}}; empty list if unspecified
  "required_mutations": list of objects {{"gene": "<UPPER_HUGO>", "variant": "<lowercase_variant>"}}; e.g. {{"gene": "BRAF", "variant": "v600e"}}, {{"gene": "KRAS", "variant": "mutant"}}; empty list if unspecified
  "biomarker_selection": list of objects {{"gene": "<UPPER_HUGO>", "level": "<lowercase_level>"}}; GENE/protein biomarkers only; e.g. {{"gene": "CD274", "level": "high"}}; empty list if unspecified
  "biomarkers_nongene": list of objects {{"marker": "<short_name>", "status": "positive|negative|high|low"}} for NON-gene DISEASE-DEFINING markers that characterize this disease's patients — serological (rf, anti_ccp, ana, anti_dsdna), functional/physiologic (lvef, ejection_fraction), metabolic (hba1c, ldl), inflammatory (crp, esr), disease scores (edss, mmse, psa, troponin). e.g. {{"marker": "rf", "status": "positive"}}, {{"marker": "lvef", "status": "low"}}; empty list if unspecified. CRITICAL: do NOT include routine organ-function / safety eligibility labs that essentially every trial requires (hemoglobin, hematocrit, platelets, WBC/ANC/neutrophils/granulocytes, creatinine/clearance, bilirubin, AST/ALT/transaminases, albumin, INR, electrolytes) — those are safety floors, not patient stratifiers.
  "disease_activity": one of "mild", "moderate", "severe", "active", "remission", or null — the disease activity/severity level enrolled (e.g. RA "active disease" → active; UC "moderately-to-severely active" → severe; "in remission" → remission)
  "functional_class": one of "i", "ii", "iii", "iv", or null — a functional classification the trial enrolls (ACR functional class, NYHA heart-failure class)
  "disease_stage": one of "iii", "iv", "metastatic", "resectable", "unresectable", "locally_advanced", or null
  "age_group": one of "pediatric", "adult", "elderly", or null
  "sex": one of "male", "female", or null
  "race_ethnicity": one of "asian", "black", "white", "hispanic", or null

Rules:
- Be conservative. Only extract features that are EXPLICIT inclusion or exclusion criteria, not features mentioned in passing in the title or description.
- "line_of_therapy" / "prior_treatments" come from eligibility specifying treatment history (e.g. "treatment-naive" → first; "inadequate response to methotrexate" → prior_treatments includes "dmards"; "as adjuvant therapy following resection" → adjuvant). Do NOT infer from study arms.
- "required_mutations" / "biomarker_selection" are set when eligibility EXPLICITLY requires a GENE biomarker/mutation. Put NON-gene markers (RF, LVEF, HbA1c, …) in "biomarkers_nongene" instead.
- "disease_activity" / "functional_class" are set when eligibility explicitly restricts by disease activity/severity or functional class. These are the primary non-onco cohort definers — capture them.
- "disease_stage" is set when eligibility or condition explicitly limits stage; default to null when the condition just names the disease without staging.
- DEMOGRAPHICS ("age_group"/"sex"/"race_ethnicity") are set ONLY when eligibility EXPLICITLY restricts the cohort: "age >= 65" -> elderly, "children/pediatric only" -> pediatric; single-sex enrollment -> that sex (do NOT infer sex from a sex-specific disease like prostate/ovarian — that's the disease, not an eligibility restriction); an ethnicity restriction (rare) -> that group. Null otherwise.
- When unsure, prefer null / empty over guessing.

Reply with ONLY the JSON object. No prose, no markdown fences.
"""

# Bump when _PROMPT_TEMPLATE changes so the per-nct cache (keyed by this
# version) regenerates instead of returning stale onco-only features.
_PROMPT_VERSION = "v4_preserve_labs"


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
# Demographic axes (round-31). Extracted ONLY when eligibility explicitly
# restricts the cohort (single-sex trial, pediatric/elderly-only, an
# ethnicity-restricted study) — never inferred from a mixed population. These
# are subgroup stratifiers (not in _DEFAULT_POPULATION_AXES), so they fork
# subgroup PopulationNodes off the parent enrollment cohort.
_AGE_LEVELS = {"pediatric", "adult", "elderly"}
_SEX_LEVELS = {"male", "female"}
_RACE_LEVELS = {"asian", "black", "white", "hispanic"}
# Non-onco cohort-definer levels (multi-indication).
_SEVERITY_LEVELS = {"mild", "moderate", "severe", "active", "remission"}
_FUNCTIONAL_CLASS_LEVELS = {"i", "ii", "iii", "iv"}
_BIOMARKER_STATUS_LEVELS = {"positive", "negative", "high", "low"}

# Routine organ-function / safety eligibility labs — the universal "adequate
# organ/marrow/liver/renal function" panel virtually EVERY trial requires. These
# are safety floors, NOT disease stratifiers, and the generalized extractor over-
# captured them (populations like hemoglobin_high__platelet_high__creatinine_low).
# Dropped from biomarkers_nongene. Matched on the de-underscored marker slug.
# Disease-DEFINING markers (rf, lvef, hba1c, edss, psa, crp/esr, ldl, troponin,
# fev1, proteinuria, …) are deliberately NOT here — they characterize the cohort.
_ROUTINE_LAB_MARKERS: frozenset[str] = frozenset({
    "hemoglobin", "hgb", "hematocrit", "hct", "platelet", "platelets",
    "plateletcount", "wbc", "whitebloodcells", "whitebloodcell", "whitecells",
    "anc", "absoluteneutrophilcount", "neutrophil", "neutrophils",
    "neutrophilcount", "granulocyte", "granulocytes", "granulocytecount",
    "absolutegranulocytecount", "lymphocytecount",
    "creatinine", "creatinineclearance", "bun",
    "bilirubin", "totalbilirubin", "ast", "alt", "sgot", "sgpt",
    "transaminase", "transaminases", "alkalinephosphatase", "alp", "albumin",
    "ggt", "inr", "pt", "ptt", "aptt",
    "calcium", "magnesium", "potassium", "sodium", "phosphate", "phosphorus",
    "karnofsky", "karnofskyscore", "kps",  # performance status, not a biomarker
})


def _as_str_list(v: Any) -> list[str]:
    """Normalize an LLM field that *should* be a scalar string but is
    occasionally returned as a list (e.g. ``disease_stage: ["stage III",
    "metastatic"]``) into a list of lowercased candidate strings. A scalar
    yields one candidate; ``None``/empty yields none. Hardens the parser
    against LLM shape variance that only surfaces at corpus scale."""
    if v is None:
        return []
    items = v if isinstance(v, list) else [v]
    out: list[str] = []
    for x in items:
        s = str(x).strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def _features_from_llm_response(raw: dict[str, Any]) -> list[SubgroupFeature]:
    """Convert the LLM's structured JSON into a list of SubgroupFeatures.

    Empty/unknown axes produce no entries. Invalid level strings are
    silently dropped (the slug shouldn't accept arbitrary LLM text). Scalar
    fields tolerate a list-valued LLM response (each valid value adds a
    feature); object-list fields skip non-dict items.
    """
    features: list[SubgroupFeature] = []

    for line in _as_str_list(raw.get("line_of_therapy")):
        if line in _LINE_LEVELS:
            features.append(SubgroupFeature(
                axis="line", level=_LINE_LEVELS[line],
            ))

    for stage in _as_str_list(raw.get("disease_stage")):
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

    # Demographics (round-31): age/sex/race, only when the LLM found an explicit
    # eligibility restriction. Subgroup stratifiers (not in the coarse parent-pop
    # axes), so they fork subgroup PopulationNodes off the enrollment cohort.
    for json_key, axis, levels in (
        ("age_group", "age", _AGE_LEVELS),
        ("sex", "sex", _SEX_LEVELS),
        ("race_ethnicity", "race", _RACE_LEVELS),
    ):
        for val in _as_str_list(raw.get(json_key)):
            if val in levels:
                features.append(SubgroupFeature(axis=axis, level=val))

    for mut in (raw.get("required_mutations") or []):
        if not isinstance(mut, dict):
            continue
        gene = (mut.get("gene") or "").strip().upper()
        variant = re.sub(
            r"[^a-z0-9]+", "", str(mut.get("variant") or "").lower(),
        )
        if gene and variant:
            features.append(SubgroupFeature(
                axis="gene", key=gene, level=variant,
            ))

    for bio in (raw.get("biomarker_selection") or []):
        if not isinstance(bio, dict):
            continue
        gene = (bio.get("gene") or "").strip().upper()
        level = re.sub(
            r"[^a-z0-9]+", "", str(bio.get("level") or "").lower(),
        )
        if gene and level:
            features.append(SubgroupFeature(
                axis="gene", key=gene, level=level,
            ))

    # Non-onco cohort definers (multi-indication): disease activity/severity,
    # functional class, and non-gene biomarkers (RF, LVEF, HbA1c, …). These are
    # the parent-population axes outside oncology — see _DEFAULT_POPULATION_AXES.
    for act in _as_str_list(raw.get("disease_activity")):
        if act in _SEVERITY_LEVELS:
            features.append(SubgroupFeature(axis="severity", level=act))

    for fc in _as_str_list(raw.get("functional_class")):
        if fc in _FUNCTIONAL_CLASS_LEVELS:
            features.append(SubgroupFeature(axis="functional_class", level=fc))

    for bio in (raw.get("biomarkers_nongene") or []):
        if not isinstance(bio, dict):
            continue
        marker = re.sub(r"[^a-z0-9]+", "_", str(bio.get("marker") or "").lower()).strip("_")
        status = re.sub(r"[^a-z0-9]+", "", str(bio.get("status") or "").lower())
        if marker.replace("_", "") in _ROUTINE_LAB_MARKERS:
            continue  # routine organ-function/safety lab — not a stratifier
        if marker and status in _BIOMARKER_STATUS_LEVELS:
            features.append(SubgroupFeature(axis="biomarker", key=marker, level=status))

    return features


def _routine_lab_covariates(raw: dict[str, Any]) -> list[dict[str, str]]:
    """The routine organ-function/safety labs the feature parser DROPS — captured
    here (NOT as graph features) so they're preserved per-trial in the population-
    features cache for a future BioLORD covariate experiment (baseline lab profile
    → outcome), without fragmenting population identity. Returns ``[{"marker":..,
    "status":..}, ...]``."""
    out: list[dict[str, str]] = []
    for bio in (raw.get("biomarkers_nongene") or []):
        if not isinstance(bio, dict):
            continue
        marker = re.sub(r"[^a-z0-9]+", "_", str(bio.get("marker") or "").lower()).strip("_")
        status = re.sub(r"[^a-z0-9]+", "", str(bio.get("status") or "").lower())
        if marker and status and marker.replace("_", "") in _ROUTINE_LAB_MARKERS:
            out.append({"marker": marker, "status": status})
    return out


def cached_eligibility_labs(cache_path: Path, nct_id: str) -> list[dict[str, str]]:
    """Read back the preserved routine eligibility labs for a trial (the input for
    the future lab-covariate experiment). Empty list when absent / old-shape."""
    if not cache_path.exists():
        return []
    try:
        cache = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return []
    entry = cache.get(f"{_PROMPT_VERSION}:{nct_id}")
    return entry.get("eligibility_labs", []) if isinstance(entry, dict) else []


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

    # Version-namespaced key: a prompt change (bumping _PROMPT_VERSION) leaves
    # the stale v1 entries in the file but misses on the v2 key, so features
    # regenerate under the new prompt without a manual cache wipe.
    cache_key = f"{_PROMPT_VERSION}:{nct_id}"
    if cache_key in cache:
        entry = cache[cache_key]
        feats = entry["features"] if isinstance(entry, dict) else entry
        return [SubgroupFeature.model_validate(f) for f in feats]

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
        cache[cache_key] = {"features": [], "eligibility_labs": []}
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        return []

    features = _features_from_llm_response(parsed)
    cache[cache_key] = {
        "features": [f.model_dump() for f in features],
        "eligibility_labs": _routine_lab_covariates(parsed),
    }
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return features
