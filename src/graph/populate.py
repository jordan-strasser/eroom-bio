"""Orchestrator that constructs the initial knowledge graph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import anthropic
from rich.console import Console

from src.annotation.extractor import _call_messages_with_backoff
from src.graph.models import (
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointClass,
    EndpointNode,
    EndpointType,
    GraphEdge,
    IndicationNode,
    MechanismCategory,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    SubgroupFeature,
    TargetNode,
    TrialArm,
    TrialNode,
    TrialOutcome,
    TrialSubgraph,
    normalize_entity,
)
from src.graph.indication_taxonomy import parent_indication_for
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ArmGroup,
    ClinicalTrialsClient,
    TrialRecord,
    is_drug_like,
    map_trial_to_graph_nodes,
)


def _normalize_drug_lookup_name(name: str) -> str:
    """Normalize a compound name for OT cache lookup.

    Trials report the same drug with different punctuation across reports:
    "Sorafenib (Nexavar; BAY43-9006)" and "Sorafenib (Nexavar, BAY43-9006)"
    are the same drug but produce different cache keys under a naive
    ``.lower()``. We collapse the common variants so both forms hit the
    same cache entry. Only touches casing, semicolons, and whitespace —
    nothing that would conflate distinct drugs (e.g. parenthetical
    contents stay so "Drug X (in combo with Y)" doesn't collapse to
    "Drug X").
    """
    s = name.lower().strip()
    s = s.replace(";", ",")
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_parenthetical_brand(name: str) -> str | None:
    """Strip trailing parenthetical brand/synonym from a drug name.

    "Sorafenib (Nexavar; BAY43-9006)" → "Sorafenib". The parenthetical is
    usually a brand name + dev code that OT doesn't resolve directly.
    Returns None when the stripped form would be empty, would equal the
    original, or when the parenthetical content contains a
    combination-indicator word ("with", "plus", "and", "+", "combo")
    suggesting it encodes a regimen rather than a brand.
    """
    paren_match = re.search(r"\(([^)]*)\)", name)
    if paren_match:
        inside = paren_match.group(1).lower()
        if any(
            tok in inside
            for tok in ("with", "plus", "and", "+", "combo", "combination")
        ):
            return None
    stripped = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
    if not stripped or stripped.lower() == name.lower():
        return None
    return stripped


def _root_indication(indication_id: str) -> str:
    """Return the parent IndicationNode id for a subtype slug, or the id
    itself when there is no parent. Used at chain-construction time so
    every chain's ``indication_id`` anchors on the top-level disease
    (e.g. `melanoma`, not `intraocular_melanoma`). Subtype IndicationNodes
    still exist and are linked by SUBTYPE_OF edges for cross-rollup
    queries; the chain backbone just anchors on the parent so evidence
    accumulates at one place per disease.
    """
    return parent_indication_for(indication_id) or indication_id
from src.ingestion.lincs import (
    LINCSClient,
    _category_to_mechanism_type,
    populate_lincs_signatures,
)
from src.ingestion.opentargets import (
    OpenTargetsClient,
    score_to_prior,
)

logger = logging.getLogger(__name__)
console = Console()

# Haiku is fast + cheap for short categorical labels. The structural inferences
# (endpoint type, mechanism, population) are short—Haiku is more than enough.
INFERENCE_MODEL = "claude-haiku-4-5-20251001"


# ── Endpoint classification ──────────────────────────────────────────────


# Deterministic endpoint classifier. Keyword-first so we don't burn an
# LLM call on what's almost always a textual match. The text
# "Progression Free Survival (PFS)" / "Overall Survival" / "Median PFS"
# all unambiguously identify their EndpointClass.
# Order matters: more-specific patterns first so e.g. "distant
# metastasis-free survival" matches DMFS, not DFS.
_ENDPOINT_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdistant[\s-]*metastas[ie]s[\s-]*free\s+survival\b|\bdmfs\b", re.I), "DMFS"),
    (re.compile(r"\brelapse[\s-]*free\s+survival\b|\brfs\b", re.I), "RFS"),
    (re.compile(r"\bdisease[\s-]*free\s+survival\b|\bdfs\b", re.I), "DFS"),
    (re.compile(r"\bduration\s+of\s+(?:overall\s+)?response\b|\bdor\b", re.I), "DOR"),
    (re.compile(r"\bprogression[\s-]*free\s+survival\b|\bpfs\b", re.I), "PFS"),
    (re.compile(r"\boverall\s+survival\b|\bos\b(?!\w)", re.I), "OS"),
    (re.compile(r"\btime\s+to\s+progression\b|\bttp\b", re.I), "TTP"),
    (re.compile(r"\bobjective\s+response\s+rate\b|\boverall\s+response\s+rate\b|\bbest\s+overall\s+response\b|\borr\b", re.I), "ORR"),
    (re.compile(r"\bcomplete\s+response\b|\bcomplete\s+remission\b|\bcr\s+rate\b", re.I), "CR"),
    # Phase I / PD-safety primaries—AE counts, DLTs, MTD, ECOG, vitals,
    # routine labs. These are legitimate primary endpoints that don't
    # directly capture indication-level efficacy; they get a low-but-non-
    # zero prior so the graph still has an EndpointNode pointing
    # somewhere instead of falling into "other".
    (re.compile(
        r"\b(?:adverse\s+events?|serious\s+adverse|saes?|aes?\b|"
        r"dose[\s-]*limiting\s+toxicit|dlt\b|"
        r"maximum\s+tolerated\s+dose|mtd\b|"
        r"ecog\s+performance|"
        # vitals
        r"blood\s+pressure|systolic|diastolic|heart\s+rate|"
        r"temperature|body\s+weight|\bweight\b|"
        r"oxygen\s+saturation|spo2\b|"
        # cardiac
        r"ventricular\s+ejection|lvef\b|electrocardiogram|ecg\b|"
        # routine labs
        r"hematology\s+parameter|hematolog(?:y|ic)\b|"
        r"clinical\s+chemistry|chemistry\s+parameter|urinalysis\b|"
        r"liver\s+function|renal\s+function)",
        re.I,
    ), "safety"),
]


# Suffix qualifiers that don't change the underlying endpoint class —
# "PFS by investigator" / "PFS by BICR" / "OS by independent review" all
# describe the same kind of outcome from different adjudicators. Stripping
# them keeps the regex pool focused on the endpoint's *meaning*.
_ENDPOINT_QUALIFIER_SUFFIXES = re.compile(
    r"\s+by\s+(?:investigator|bicr|irc|"
    r"independent\s+(?:central\s+)?review(?:\s+committee)?|"
    r"central\s+review|local\s+review|"
    r"investigator\s+assessment|local\s+investigator)"
    r"(?:\s+assessment)?",
    re.I,
)


def _normalize_endpoint_text(measure_text: str) -> str:
    """Strip parenthetical clarifiers and reviewer suffixes.

    Examples:
      "Progression-Free Survival (PFS) by investigator"
        → "Progression-Free Survival"
      "ORR (CR + PR) in BRAF V600E"
        → "ORR in BRAF V600E"
      "OS [Phase 2 cohort]" → "OS"

    Done so the deterministic classifier and Haiku cache see canonical
    text—different reviewer suffixes for the same trial's PFS read no
    longer cause cache misses or class mismatches.
    """
    text = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", measure_text or "")
    text = _ENDPOINT_QUALIFIER_SUFFIXES.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_endpoint_deterministic(measure_text: str) -> str:
    """Map an outcome-measure string to an EndpointClass value.

    Returns the EndpointClass enum *value* (e.g. "PFS", "OS"). Falls back
    to "other" when no keyword matches—same fallback the LLM-driven
    classifier uses. Tries the raw text first, then a normalized
    pass with parentheticals + reviewer suffixes stripped, so e.g.
    "Disease-Free Survival (DFS), as Assessed by Investigator" still
    classifies to DFS even when surrounding noise would otherwise miss.
    """
    text = (measure_text or "").strip()
    if not text:
        return "other"
    for pattern, cls in _ENDPOINT_KEYWORDS:
        if pattern.search(text):
            return cls
    normalized = _normalize_endpoint_text(text)
    if normalized and normalized != text:
        for pattern, cls in _ENDPOINT_KEYWORDS:
            if pattern.search(normalized):
                return cls
    return "other"


# Beta(α, β) priors per endpoint class. Higher α/β reflects more regulatory
# precedent for translating endpoint movement into disease-level benefit.
ENDPOINT_PRIORS_BY_CLASS: dict[str, tuple[float, float]] = {
    "OS": (3.0, 1.0),
    "DFS": (2.5, 1.0),
    "RFS": (2.5, 1.0),  # accepted regulatory endpoint in adjuvant melanoma
    "composite_survival": (2.5, 1.0),
    "PFS": (2.0, 1.0),
    "TTP": (2.0, 1.0),
    "CR": (1.7, 1.0),
    "DMFS": (1.5, 1.0),  # used in adjuvant trials but less precedent than DFS
    "DOR": (1.5, 1.0),   # tracks response durability conditional on responding
    "ORR": (1.5, 1.0),
    "composite_response": (1.5, 1.0),
    "biomarker": (1.2, 1.0),
    "PRO": (1.0, 1.0),
    # Safety endpoints are legitimate primary outcomes (Phase I, dose-finding)
    # but don't directly capture efficacy at the indication level. Beta(1, 1.5)
    # → mean ~0.4—slightly biased toward "doesn't capture clinical benefit"
    # with low evidence so trial outcomes can still update.
    "safety": (1.0, 1.5),
    "other": (1.0, 1.0),
}
ENDPOINT_CLASSES = list(ENDPOINT_PRIORS_BY_CLASS.keys())


class JSONCache:
    """Tiny dict-style cache that flushes to a JSON file on every write.

    Used for structural LLM classifications (endpoint type, mechanism, population)
    so the same input is never paid for twice.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("Cache %s unreadable; starting empty", path)
                self._data = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._data)


async def classify_endpoint_with_llm(
    client: anthropic.AsyncAnthropic,
    endpoint_name: str,
    cache: JSONCache,
) -> str:
    """Classify a primary endpoint name into one of ENDPOINT_CLASSES.

    Order: cache → deterministic keyword pattern → LLM. The keyword
    short-circuit handles unambiguous strings (OS, ORR, DLT, heart rate,
    …) without burning an LLM call AND without letting the LLM override
    a clear match—e.g. 'heart rate' belongs to safety regardless of
    how the model decides to interpret 'change from baseline' framing.
    """
    # Cache by both raw name and normalized form. The raw key preserves
    # backward compat with the existing on-disk cache; the normalized key
    # collapses reviewer-suffix variants ("PFS by investigator" ≡ "PFS by
    # BICR") so the second variant doesn't pay for a redundant Haiku call.
    cached = cache.get(endpoint_name)
    if cached is not None:
        return cached
    normalized = _normalize_endpoint_text(endpoint_name)
    if normalized and normalized != endpoint_name:
        cached_norm = cache.get(normalized)
        if cached_norm is not None:
            cache.set(endpoint_name, cached_norm)
            return cached_norm
    deterministic = classify_endpoint_deterministic(endpoint_name)
    if deterministic != "other":
        cache.set(endpoint_name, deterministic)
        if normalized and normalized != endpoint_name:
            cache.set(normalized, deterministic)
        return deterministic
    classes = ", ".join(ENDPOINT_CLASSES)
    user_msg = (
        f"Clinical trial primary endpoint: {endpoint_name!r}\n\n"
        f"Classify into ONE of: {classes}\n\n"
        "Map to the closest matching category. Hints:\n"
        "  - 'best overall response', 'overall response', 'tumor response', "
        "'objective response' → ORR\n"
        "  - 'clinical benefit rate', 'disease control rate' → "
        "composite_response\n"
        "  - 'duration of response', 'duration of clinical benefit' → DOR\n"
        "  - 'distant metastasis-free survival' → DMFS (not DFS)\n"
        "  - 'relapse-free survival' → RFS (distinct from DFS)\n"
        "  - immune readouts (T-cell counts, cytokines, immunogenicity) → "
        "biomarker\n"
        "  - patient-reported outcomes (QoL, fatigue scales) → PRO\n"
        "  - safety/tolerability primary endpoints (AEs, SAEs, DLTs, MTD, "
        "ECOG, vital signs, ECG findings) → safety\n"
        "Use 'other' ONLY when the endpoint genuinely fits no category—"
        "rare. Phase I tolerability primaries belong in 'safety', not "
        "'other'.\n\n"
        "Reply with only the category name. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    classification = raw if raw in ENDPOINT_CLASSES else "other"
    cache.set(endpoint_name, classification)
    return classification


def seed_endpoint_captures_edge(
    graph: GraphStore,
    endpoint_id: str,
    endpoint_name: str,
    indication_id: str,
    classification: str,
) -> bool:
    """Add an endpoint_captures edge with the prior for the given class."""
    prior = ENDPOINT_PRIORS_BY_CLASS.get(classification)
    if prior is None:
        return False
    alpha, beta_val = prior
    graph.add_edge(GraphEdge(
        source_id=endpoint_id,
        target_id=indication_id,
        edge_type=EdgeType.ENDPOINT_CAPTURES,
        belief=EdgeBeliefState(alpha=alpha, beta=beta_val),
        metadata={
            "source": "endpoint_type_prior",
            "endpoint_name": endpoint_name,
            "classification": classification,
        },
    ))
    return True


# ── Mechanism inference ──────────────────────────────────────────────────


def _sanitize_label(text: str, fallback: str = "unknown") -> str:
    """Coerce LLM free-text into a snake_case identifier safe to embed in node IDs."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    return cleaned[:60] or fallback


async def infer_mechanism_for_arm(
    client: anthropic.AsyncAnthropic,
    trial: TrialRecord,
    arm_label: str,
    arm_compound_names: list[str],
    target_node: dict[str, Any],
    cache: JSONCache,
    cache_key: str,
) -> str:
    """Map an (arm, target) pair to a MechanismCategory value.

    Each arm of a trial gets its own mechanism—a combo arm's mechanism may
    differ from its constituent mono arms (e.g. ipilimumab+nivolumab is
    ``checkpoint_blockade`` even though only nivolumab is on the PD-1 path).
    """
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    intervention_text = "; ".join(n for n in arm_compound_names if n) or "unknown"
    target_symbol = target_node.get("gene_symbol") or target_node.get("name") or ""
    categories = ", ".join(c.value for c in MechanismCategory)
    user_msg = (
        f"Trial: {trial.title}\n"
        f"Arm: {arm_label}\n"
        f"Drug(s) on this arm: {intervention_text}\n"
        f"Primary target on this arm: {target_symbol or '(unresolved)'}\n\n"
        f"Classify the MECHANISM OF ACTION FOR THIS ARM into ONE of these categories:\n"
        f"{categories}\n\n"
        "Reply with only the category value (e.g. 'kinase_inhibition'). "
        "If none fit, reply 'other'. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    label = normalize_entity(raw or "other", "MechanismNode")
    cache.set(cache_key, label)
    return label


# ── Population inference ─────────────────────────────────────────────────


async def infer_population_for_trial(
    client: anthropic.AsyncAnthropic,
    trial: TrialRecord,
    cache: JSONCache,
) -> str:
    """Infer a structured population id ``{indication}_{biomarker}_{line}``.

    LLM output is validated by ``normalize_entity(..., "PopulationNode")``;
    if the response doesn't match the three-segment format we fall back to
    ``unselected_unselected_unselected``.
    """
    cached = cache.get(trial.nct_id)
    if cached is not None:
        return cached
    user_msg = (
        f"Trial title: {trial.title}\n"
        f"Conditions: {', '.join(trial.conditions) or 'unspecified'}\n\n"
        "Reply with one population id in the strict snake_case format "
        "'{indication}_{biomarker_status}_{line_of_therapy}'. Examples:\n"
        "  - melanoma_pdl1_positive_first_line\n"
        "  - nsclc_egfr_mutant_second_line\n"
        "  - aml_unselected_treatment_naive\n"
        "Use 'unselected' when a segment is unknown. Three underscore-separated "
        "segments only. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=30,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    try:
        label = normalize_entity(raw, "PopulationNode")
    except ValueError:
        logger.warning(
            "Population '%s' for %s did not match the canonical format; using fallback",
            raw, trial.nct_id,
        )
        label = "unselected_unselected_unselected"
    cache.set(trial.nct_id, label)
    return label

# Placeholder for unresolvable subgraph fields
_UNKNOWN = "UNKNOWN"


# Curated mapping for melanoma tumor-associated antigen (TAA) peptide
# vaccines that Open Targets cannot resolve to a target. Each pattern is a
# lowercase substring; if it appears in the intervention name, the listed
# targets are wired in. Beta(3, 1) prior — confident curated mapping, below
# OT-sourced (Beta(4, 1)) and above name-match heuristic (Beta(2, 1.5)).
_PEPTIDE_VACCINE_TARGETS: list[tuple[str, list[tuple[str, str, str]]]] = [
    # gp100 antigen → PMEL gene (premelanosome protein)
    ("gp100", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein"),
    ]),
    ("imcgp100", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein"),
    ]),
    # Tyrosinase peptide → TYR gene
    ("tyrosinase", [
        ("ENSG00000077498", "TYR", "Tyrosinase"),
    ]),
    # MART-1 / Melan-A → MLANA gene
    ("mart-1", [
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
    ("mart1", [
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
    ("melan-a", [
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
    # Composite multi-epitope melanoma vaccines that target all three TAAs.
    ("4-peptide melanoma vaccine", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein"),
        ("ENSG00000077498", "TYR", "Tyrosinase"),
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
    ("multi-epitope melanoma peptide vaccine", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein"),
        ("ENSG00000077498", "TYR", "Tyrosinase"),
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
]


class PopulationPipeline:
    def __init__(
        self,
        graph: GraphStore,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        cache_dir: Path = Path("data/cache"),
    ) -> None:
        self.graph = graph
        self._ct_client = ClinicalTrialsClient()
        self._ot_client = OpenTargetsClient()
        self._anthropic = anthropic_client
        self._cache_dir = cache_dir
        # name → node_id index, keyed by (normalized_name, node_type)
        self._entity_index: dict[tuple[str, str], str] = {}
        # raw CT.gov condition string → (canonical_slug, display_name).
        # Lazily initialized when canonicalization is first needed.
        self._indication_canon: JSONCache | None = None

    # ── Entity resolution ────────────────────────────────────────────────

    def resolve_entity(self, name: str, entity_type: str) -> str | None:
        """Normalize a name and look up its node ID.

        Uses exact string matching + lowercasing.
        TODO: fuzzy matching / NER for better resolution.
        """
        key = (_normalize(name), entity_type)
        return self._entity_index.get(key)

    def _index_node(self, node_id: str, name: str, entity_type: str) -> None:
        key = (_normalize(name), entity_type)
        self._entity_index[key] = node_id

    # ── Main pipeline ────────────────────────────────────────────────────

    async def populate_oncology(
        self,
        max_trials: int = 500,
        include_terminated_no_results: bool = True,
        condition: str = "cancer",
        trials: list[TrialRecord] | None = None,
    ) -> dict[str, Any]:
        # Step 1: Fetch trials (or use the caller's pre-fetched list).
        # Drivers that need the same TrialRecord set for both populate and
        # downstream extraction can pass `trials=` to avoid a duplicate
        # CT.gov roundtrip and any drift between the two fetches.
        if trials is None:
            console.print(
                f"[bold]Fetching up to {max_trials} {condition} trials...[/bold]"
            )
            trials = await self._ct_client.fetch_oncology_with_results(
                max_results=max_trials, condition=condition,
            )
            console.print(f"  Fetched {len(trials)} with-results trials")

            if include_terminated_no_results:
                terminated = await self._ct_client.fetch_oncology_terminated_with_reason(
                    max_results=max_trials, condition=condition,
                )
                seen = {t.nct_id for t in trials}
                added = [t for t in terminated if t.nct_id not in seen]
                trials.extend(added)
                console.print(
                    f"  +{len(added)} terminated/withdrawn trials with why_stopped"
                )
        else:
            console.print(f"[bold]Using {len(trials)} pre-fetched trials[/bold]")

        # Step 2: Canonicalize conditions and seed Indication + default
        # Population nodes per trial.
        #
        # CT.gov free-text condition strings fragment evidence: "Stage IIIC
        # Cutaneous Melanoma AJCC v7" and "Unresectable or Metastatic
        # Melanoma" describe the same disease but used to produce two
        # separate IndicationNodes, so endpoint slugs and cross-trial
        # learning didn't share. Now we:
        #
        #   (a) LLM-canonicalize raw condition → base disease (melanoma,
        #       breast_cancer, multiple_sclerosis, ...). One IndicationNode
        #       per canonical disease, with metadata accumulating every raw
        #       phrasing and qualifier we've seen.
        #   (b) Deterministically parse qualifiers from the raw condition
        #       (stage, histology, extent, line, severity, ...) and compose
        #       the trial's default PopulationNode id, e.g.
        #       ``melanoma__histology_cutaneous__stage_iii``. Biomarker
        #       subgroups (BRAF V600E, PD-L1 high) extracted later add
        #       additional PopulationNodes that fork off this default.
        #
        # Endpoint nodes are created in step 3 after LLM classification so
        # their canonical id ({EndpointClass}_{canonical_indication}) is
        # stable across phrasing variants.
        from src.graph.indication_taxonomy import (
            extract_indication_qualifiers,
        )

        console.print("[bold]Extracting graph nodes from trials...[/bold]")
        seen_indications: dict[str, str] = {}  # canonical_id → display name

        # Per-canonical-indication metadata accumulator. Keeps track of
        # observed raw phrasings and the qualifier levels each axis saw —
        # useful for surfacing "this canonical disease appears as stage
        # III + IV across our corpus" downstream.
        ind_metadata: dict[str, dict[str, Any]] = {}

        # Trial id → (canonical_indication_id, default_population_id)
        # for build_trial_subgraphs to pick up.
        self._trial_default_population: dict[str, str] = {}
        self._trial_canonical_indication: dict[str, str] = {}

        # First pass: canonicalize, build IndicationNodes and default
        # PopulationNodes, index raw cond text → canonical id.
        for trial in trials:
            chosen_canonical: str | None = None
            chosen_population: str | None = None
            for cond in trial.conditions:
                canonical_id, canonical_name = await self._canonicalize_indication(cond)
                if not canonical_id:
                    continue
                qualifiers = extract_indication_qualifiers(cond)

                # Create / update IndicationNode for this canonical disease.
                try:
                    existing = self.graph.get_node(canonical_id)
                    md = dict(existing.get("metadata") or {})
                except KeyError:
                    md = {}
                    existing = None
                variants: list[str] = list(md.get("observed_variants") or [])
                if cond and cond not in variants:
                    variants.append(cond)
                md["observed_variants"] = variants
                axis_map: dict[str, list[str]] = {
                    k: list(v) for k, v in (md.get("qualifier_axes") or {}).items()
                }
                for f in qualifiers:
                    levels = axis_map.setdefault(f.axis, [])
                    if f.level not in levels:
                        levels.append(f.level)
                md["qualifier_axes"] = axis_map
                md["canonical_name"] = canonical_name

                if existing is None:
                    self.graph.add_node(IndicationNode(
                        id=canonical_id,
                        name=canonical_name,
                        metadata=md,
                    ))
                else:
                    # Update metadata on the existing node in place. The
                    # graph store exposes nodes as dict views, so we mutate
                    # the stored metadata directly.
                    existing["metadata"] = md
                ind_metadata[canonical_id] = md

                # Subtype hierarchy: if this is a known subtype, ensure
                # both the parent IndicationNode and a SUBTYPE_OF edge
                # exist so cross-indication queries can roll up. Idempotent
                # on repeat rebuilds — the edge add is skipped if it's
                # already present.
                from src.graph.indication_taxonomy import parent_indication_for
                parent_id = parent_indication_for(canonical_id)
                if parent_id and parent_id != canonical_id:
                    try:
                        self.graph.get_node(parent_id)
                    except KeyError:
                        self.graph.add_node(IndicationNode(
                            id=parent_id,
                            name=parent_id.replace("_", " "),
                            metadata={"source": "subtype_hierarchy_parent"},
                        ))
                        self._index_node(parent_id, parent_id.replace("_", " "), "indication")
                    if not self.graph._graph.has_edge(  # noqa: SLF001
                        canonical_id, parent_id, key=EdgeType.SUBTYPE_OF.value,
                    ):
                        self.graph.add_edge(GraphEdge(
                            source_id=canonical_id,
                            target_id=parent_id,
                            edge_type=EdgeType.SUBTYPE_OF,
                            metadata={"source": "indication_taxonomy"},
                        ))

                # Index BOTH the canonical id and the raw cond text →
                # canonical id so downstream resolve_entity(cond, "indication")
                # returns the canonical IndicationNode regardless of which
                # variant the trial used.
                self._index_node(canonical_id, canonical_name, "indication")
                self._index_node(canonical_id, cond, "indication")
                seen_indications.setdefault(canonical_id, canonical_name)

                # Compose the trial's default PopulationNode id from the
                # canonical disease + parsed qualifiers. With no qualifiers
                # this falls back to ``{indication}__unselected``.
                default_pop_id = PopulationNode.compose_id(
                    canonical_id, qualifiers,
                )
                try:
                    self.graph.get_node(default_pop_id)
                except KeyError:
                    self.graph.add_node(PopulationNode(
                        id=default_pop_id,
                        name=(
                            cond if qualifiers
                            else f"All patients ({canonical_name})"
                        ),
                        defining_features=list(qualifiers),
                    ))
                # responds_differently: default_population → indication.
                # The trial's enrollment is itself a stratification of the
                # disease (stage III cutaneous melanoma is not the same as
                # melanoma overall), and downstream prediction walks this
                # edge to score population fit.
                if qualifiers and not self.graph._graph.has_edge(  # noqa: SLF001
                    default_pop_id, canonical_id,
                    key=EdgeType.RESPONDS_DIFFERENTLY.value,
                ):
                    self.graph.add_edge(GraphEdge(
                        source_id=default_pop_id,
                        target_id=canonical_id,
                        edge_type=EdgeType.RESPONDS_DIFFERENTLY,
                        belief=EdgeBeliefState(alpha=1.5, beta=1.0),
                        metadata={
                            "source": "indication_qualifiers",
                            "raw_descriptor": cond,
                        },
                    ))

                if chosen_canonical is None:
                    chosen_canonical = canonical_id
                    chosen_population = default_pop_id

            if chosen_canonical is not None:
                self._trial_canonical_indication[trial.nct_id] = chosen_canonical
            if chosen_population is not None:
                self._trial_default_population[trial.nct_id] = chosen_population

            # Compound nodes still come from the trial's interventions —
            # no canonicalization needed here, just indexing.
            nodes = map_trial_to_graph_nodes(trial)
            for comp in nodes["compounds"]:
                self.graph.add_node(comp)
                self._index_node(comp.id, comp.name, "compound")

        stats_after_nodes = self.graph.stats()
        console.print(
            f"  Nodes: {stats_after_nodes['node_count']} "
            f"({stats_after_nodes['node_types']})"
        )

        # Step 3: Canonical EndpointNodes via LLM classification.
        # One node per (EndpointClass, indication)—id = "{class}_{indication}".
        console.print("[bold]Classifying endpoints into canonical classes...[/bold]")
        ep_added = await self._populate_canonical_endpoints(trials)
        console.print(f"  Added {ep_added} canonical endpoint nodes")

        # Step 4: Trial-driven Open Targets population.
        # Fetch only the compound→target relationships our trials actually
        # exercise, then look up disease-association scores for each
        # (target, indication) pair the trials imply. This avoids loading
        # the full disease-association catalog (which adds hundreds of
        # targets no trial in our set tests).
        console.print(
            f"[bold]Resolving compound→target via Open Targets ({len(seen_indications)} indications)...[/bold]"
        )
        compound_targets, ot_binds_added = await self._populate_compound_targets(trials)
        console.print(
            f"  Added {ot_binds_added} binds_to edges across "
            f"{sum(1 for v in compound_targets.values() if v)} OT-resolved compounds"
        )

        console.print(
            "[bold]Wiring target→indication priors from Open Targets...[/bold]"
        )
        ot_pair_added = await self._populate_target_indication_priors(
            trials, compound_targets, seen_indications,
        )
        console.print(f"  Added {ot_pair_added} biology_drives edges (trial-implied pairs)")

        # Curated peptide-vaccine target heuristic: melanoma TAA vaccines
        # (gp100, tyrosinase, MART-1, multi-epitope) that OT can't resolve.
        # Runs before name-match fallback so the synthesized TargetNodes
        # are available to the fallback for trial-text cross-references.
        # Merge the heuristic's compound→target mapping into the main
        # compound_targets dict so per-constituent chain construction
        # (_populate_trial_mechanisms) sees these compounds as targeted.
        console.print("[bold]Wiring peptide-vaccine target edges...[/bold]")
        peptide_added, peptide_targets = self._add_peptide_vaccine_target_edges(trials)
        for cid, tids in peptide_targets.items():
            existing = compound_targets.setdefault(cid, [])
            for tid in tids:
                if tid not in existing:
                    existing.append(tid)
        console.print(f"  Added {peptide_added} AFFECTS edges (peptide-vaccine heuristic)")

        # Name-matching fallback: catches binds_to relationships for
        # compounds OT couldn't resolve from trial text.
        console.print("[bold]Cross-referencing compound-target edges...[/bold]")
        binds_added = self._add_compound_target_edges(trials)
        console.print(f"  Added {binds_added} binds_to edges (name-match fallback)")

        # Step 5: Build trial subgraphs
        console.print("[bold]Building trial subgraphs...[/bold]")
        subgraphs = self.build_trial_subgraphs(trials)
        console.print(f"  Built {len(subgraphs)} trial subgraphs")

        # Step 5.5: Resolve mechanism per trial.
        # For each trial, infer a MechanismCategory from its primary target +
        # intervention text and add the canonical target→mechanism
        # modulates_via edge.
        console.print("[bold]Resolving mechanisms per trial...[/bold]")
        mech_added = await self._populate_trial_mechanisms(trials, compound_targets)
        console.print(f"  Added {mech_added} modulates_via edges")

        # Step 6: LINCS L1000 Touchstone signatures.
        # Adds mechanism_affects evidence and materializes BiologyNodes for
        # any Reactome pathway consistently perturbed by a Touchstone
        # compound whose target+mechanism are already in the graph.
        # Skipped silently if CLUE_API_KEY is unset.
        console.print("[bold]Adding LINCS L1000 mechanism→biology evidence...[/bold]")
        try:
            lincs_client = LINCSClient()
            lincs_added = await populate_lincs_signatures(lincs_client, self.graph)
            console.print(f"  Added {lincs_added} LINCS evidence records")
        except RuntimeError as exc:
            console.print(f"  [yellow]Skipped LINCS:[/yellow] {exc}")

        # Step 6.5: Resolve a per-trial biology so chains can be traversed
        # end-to-end. Prefers real Reactome pathway BiologyNodes — first
        # via existing LINCS-populated mechanism_affects edges, then via
        # a Reactome API lookup keyed on the target gene symbol. Falls
        # back to the legacy '{mech}__{indication}' slug only when both
        # routes return nothing; falling-back chains are tagged
        # ``metadata["unresolved_biology"] = True`` for audit.
        console.print("[bold]Resolving biology per trial...[/bold]")
        bio_added = await self._populate_trial_biology(trials)
        console.print(f"  Added {bio_added} biology nodes / chain links")

        # Summary
        final = self.graph.stats()
        summary = {
            "trials_fetched": len(trials),
            "compounds": (
                final["node_types"].get("InterventionNode", 0)
                + final["node_types"].get("CompoundNode", 0)  # legacy snapshots
            ),
            "targets": final["node_types"].get("TargetNode", 0),
            "indications": final["node_types"].get("IndicationNode", 0),
            "endpoints": final["node_types"].get("EndpointNode", 0),
            "edges": final["edge_count"],
            "trial_subgraphs": len(subgraphs),
        }
        console.print(
            f"\n[bold green]Graph populated:[/bold green] "
            f"{summary['compounds']} compounds, "
            f"{summary['targets']} targets, "
            f"{summary['indications']} indications, "
            f"{summary['edges']} edges, "
            f"{summary['trial_subgraphs']} trial subgraphs"
        )
        return summary

    async def _canonicalize_indication(
        self, cond: str
    ) -> tuple[str, str]:
        """Map a raw CT.gov condition string to a base-disease (slug, name).

        Strips staging, subtype, severity, line-of-therapy, resectability,
        and other modifiers — returns only the base disease. Cross-domain
        examples (the LLM is told to handle each):

          "Stage IIIC Cutaneous Melanoma AJCC v7"   → ("melanoma", "melanoma")
          "Unresectable or Metastatic Melanoma"     → ("melanoma", "melanoma")
          "Triple-Negative Breast Cancer"           → ("breast_cancer", "breast cancer")
          "Severe Refractory Rheumatoid Arthritis"  → ("rheumatoid_arthritis", ...)
          "Relapsing-Remitting Multiple Sclerosis"  → ("multiple_sclerosis", ...)
          "Pediatric Acute Lymphoblastic Leukemia"  → ("acute_lymphoblastic_leukemia", ...)

        Cached on disk at ``data/cache/indication_canonicalizations.json``
        so each unique condition string costs at most one Haiku call.
        Falls back to a slugified copy of the raw text when no Anthropic
        client is configured (tests + offline mode).
        """
        from src.graph.indication_taxonomy import slugify_disease_name

        if self._indication_canon is None:
            self._indication_canon = JSONCache(
                self._cache_dir / "indication_canonicalizations.json"
            )
        cache = self._indication_canon

        cached = cache.get(cond)
        if cached:
            slug, _, name = cached.partition("|")
            if slug:
                return slug, name or slug.replace("_", " ")

        if self._anthropic is None:
            slug = slugify_disease_name(cond) or normalize_entity(
                cond, "IndicationNode",
            )
            name = cond
            cache.set(cond, f"{slug}|{name}")
            return slug, name

        user_msg = (
            f"ClinicalTrials.gov condition string: {cond!r}\n\n"
            "What is the canonical disease this trial is for?\n\n"
            "Strip these qualifiers (they describe progression, setting, "
            "or demographics — not disease identity):\n"
            "  - staging (Stage I/II/III/IV, AJCC v7/v8, etc.)\n"
            "  - resectability / spread (unresectable, metastatic, "
            "locally advanced, recurrent)\n"
            "  - line of therapy (newly diagnosed, refractory, "
            "previously treated, first-line)\n"
            "  - treatment setting (adjuvant, neoadjuvant, maintenance)\n"
            "  - severity (mild, moderate, severe, active)\n"
            "  - demographic (pediatric, adult, elderly)\n\n"
            "PRESERVE these qualifiers (they describe biologically "
            "distinct subtypes with different driver biology, populations, "
            "or treatment responses — each is its own canonical disease):\n"
            "  - anatomical subtypes of melanoma: cutaneous, uveal, "
            "mucosal, acral, ocular, intraocular, choroidal, iris\n"
            "  - molecular subtypes of breast cancer: triple-negative, "
            "HER2-positive, HR-positive\n"
            "  - histological subtypes of lung cancer: non-small-cell, "
            "small-cell, adenocarcinoma, squamous cell\n"
            "  - leukemia / lymphoma subtypes: AML, CML, ALL, CLL, "
            "Hodgkin, non-Hodgkin, DLBCL\n"
            "  - other anatomically or molecularly distinct subtypes\n\n"
            "Return only the disease itself in snake_case. Examples:\n"
            "  'Stage IIIC Cutaneous Melanoma AJCC v7' → "
            "cutaneous_melanoma\n"
            "  'Unresectable or Metastatic Melanoma' → melanoma\n"
            "  'Metastatic Uveal Melanoma' → uveal_melanoma\n"
            "  'Triple-Negative Breast Cancer' → "
            "triple_negative_breast_cancer\n"
            "  'Non-Small Cell Lung Cancer (NSCLC)' → "
            "non_small_cell_lung_cancer\n"
            "  'Severe Refractory Rheumatoid Arthritis' → "
            "rheumatoid_arthritis\n"
            "  'Relapsing-Remitting Multiple Sclerosis' → multiple_sclerosis\n"
            "  'Pediatric Acute Lymphoblastic Leukemia' → "
            "acute_lymphoblastic_leukemia\n"
            "  'Chronic Hepatitis B Infection' → hepatitis_b\n\n"
            "Reply with only the snake_case canonical disease name. No "
            "other text."
        )
        try:
            response = await _call_messages_with_backoff(
                self._anthropic,
                model=INFERENCE_MODEL,
                max_tokens=30,
                temperature=0,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
        except Exception:
            logger.debug(
                "Indication canonicalization LLM call failed for %r",
                cond, exc_info=True,
            )
            raw = ""

        slug = slugify_disease_name(raw)
        if not slug:
            slug = slugify_disease_name(cond) or normalize_entity(
                cond, "IndicationNode",
            )
            name = cond
        else:
            name = slug.replace("_", " ")

        cache.set(cond, f"{slug}|{name}")
        return slug, name

    async def _lookup_efo_for_indication(self, indication_name: str) -> str | None:
        """Resolve an indication string to an EFO id via OT search.

        Returns the EFO id, or None if no hit. Cached on the instance so
        repeat lookups (parent + subgroup populations of the same disease)
        don't re-query.
        """
        if not hasattr(self, "_efo_cache"):
            self._efo_cache: dict[str, str | None] = {}
        if indication_name in self._efo_cache:
            return self._efo_cache[indication_name]
        query = """
        query SearchDisease($name: String!) {
          search(queryString: $name, entityNames: ["disease"], page: {size: 1, index: 0}) {
            hits { id }
          }
        }
        """
        try:
            data = await self._ot_client._post(query, {"name": indication_name})
        except Exception:
            logger.debug("EFO lookup failed for '%s'", indication_name, exc_info=True)
            self._efo_cache[indication_name] = None
            return None
        hits = data["search"]["hits"]
        efo = hits[0]["id"] if hits else None
        self._efo_cache[indication_name] = efo
        return efo

    async def _populate_compound_targets(
        self, trials: list[TrialRecord]
    ) -> tuple[dict[str, list[str]], int]:
        """For each drug-like compound across the trials, OT-resolve targets.

        Returns ``(compound_targets, binds_added)`` where compound_targets
        maps compound_id → list of Ensembl target ids. Adds TargetNodes
        and binds_to edges (alpha=4, beta=1—strong prior since OT-sourced).
        """
        cache_path = self._cache_dir / "ot_drug_targets.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache: dict[str, dict[str, Any]] = (
                json.loads(cache_path.read_text()) if cache_path.exists() else {}
            )
        except json.JSONDecodeError:
            cache = {}

        # Collect (compound_id, intervention_name) pairs from drug-like
        # interventions. Same compound may surface under multiple aliases;
        # we cache by the lowercased name.
        seen: dict[str, str] = {}  # compound_id → first intervention name we saw
        for trial in trials:
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                cid = self.resolve_entity(iv.name, "compound")
                if not cid or cid in seen:
                    continue
                seen[cid] = iv.name

        async def _lookup_with_cache(name: str) -> dict[str, Any]:
            key = _normalize_drug_lookup_name(name)
            legacy = name.lower()
            if key in cache:
                return cache[key]
            if legacy in cache:
                return cache[legacy]
            try:
                fresh = await self._ot_client.get_drug_with_targets(name)
            except KeyError:
                fresh = {"chembl_id": None, "targets": []}
            except Exception:
                logger.debug(
                    "OT drug lookup failed for '%s'", name, exc_info=True,
                )
                fresh = {"chembl_id": None, "targets": []}
            cache[key] = fresh
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
            return fresh

        compound_targets: dict[str, list[str]] = {}
        binds_added = 0
        for cid, iv_name in seen.items():
            drug_data = await _lookup_with_cache(iv_name)
            # Fallback: strip trailing brand parenthetical and retry —
            # "Sorafenib (Nexavar; BAY43-9006)" returns 0 targets in OT,
            # but plain "Sorafenib" resolves. Only adopts the fallback
            # when it actually finds real targets.
            if not drug_data.get("targets"):
                stripped = _strip_parenthetical_brand(iv_name)
                if stripped:
                    stripped_data = await _lookup_with_cache(stripped)
                    if stripped_data.get("targets"):
                        drug_data = stripped_data
                        # Pin the fallback under the original key so
                        # future lookups skip the retry.
                        cache[_normalize_drug_lookup_name(iv_name)] = stripped_data
                        cache_path.write_text(
                            json.dumps(cache, indent=2, sort_keys=True)
                        )

            target_ids: list[str] = []
            for t in drug_data.get("targets") or []:
                ensembl = t.get("target_id")
                if not ensembl:
                    continue
                symbol = t.get("approved_symbol") or ensembl
                approved_name = t.get("approved_name") or symbol
                try:
                    self.graph.get_node(ensembl)
                except KeyError:
                    self.graph.add_node(TargetNode(
                        id=ensembl,
                        name=approved_name,
                        gene_symbol=symbol,
                        metadata={"source": "opentargets"},
                    ))
                    self._index_node(ensembl, symbol, "target")
                if not self.graph._graph.has_edge(  # noqa: SLF001
                    cid, ensembl, key=EdgeType.AFFECTS.value,
                ):
                    self.graph.add_edge(GraphEdge(
                        source_id=cid,
                        target_id=ensembl,
                        edge_type=EdgeType.AFFECTS,
                        belief=EdgeBeliefState(alpha=4.0, beta=1.0),
                        metadata={
                            "source": "opentargets",
                            "drug_chembl_id": drug_data.get("chembl_id"),
                        },
                    ))
                    binds_added += 1
                target_ids.append(ensembl)
            compound_targets[cid] = target_ids
        return compound_targets, binds_added

    async def _populate_target_indication_priors(
        self,
        trials: list[TrialRecord],
        compound_targets: dict[str, list[str]],
        seen_indications: dict[str, str],
    ) -> int:
        """For each (target, indication) pair the trials imply, add a
        biology_drives edge with an OT-score-derived prior.

        A pair is "implied" when at least one trial uses a compound bound
        to ``target`` for an indication ``indication``. Skips pairs OT
        has no association data for; uses ``score_to_prior`` for the rest.
        """
        cache_path = self._cache_dir / "ot_target_disease_scores.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache: dict[str, dict[str, Any]] = (
                json.loads(cache_path.read_text()) if cache_path.exists() else {}
            )
        except json.JSONDecodeError:
            cache = {}

        # Resolve EFO id per indication once.
        efo_by_indication: dict[str, str] = {}
        for ind_id, ind_name in seen_indications.items():
            efo = await self._lookup_efo_for_indication(ind_name)
            if efo:
                efo_by_indication[ind_id] = efo

        # Walk trials to build the (target, indication) pair set.
        pairs: set[tuple[str, str]] = set()
        for trial in trials:
            inds = [self.resolve_entity(c, "indication") for c in trial.conditions]
            inds = [i for i in inds if i and i in efo_by_indication]
            for iv in trial.interventions:
                if not is_drug_like(iv):
                    continue
                cid = self.resolve_entity(iv.name, "compound")
                if not cid:
                    continue
                for tid in compound_targets.get(cid, []):
                    for ind_id in inds:
                        pairs.add((tid, ind_id))

        added = 0
        for tid, ind_id in pairs:
            efo = efo_by_indication[ind_id]
            cache_key = f"{tid}__{efo}"
            if cache_key in cache:
                row = cache[cache_key]
            else:
                try:
                    rows = await self._ot_client.get_associations(
                        tid, disease_id=efo, min_score=0.0,
                    )
                except Exception:
                    logger.debug(
                        "OT score lookup failed for %s × %s",
                        tid, efo, exc_info=True,
                    )
                    rows = []
                row = rows[0] if rows else {}
                cache[cache_key] = row
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

            if not row:
                # No association evidence—skip rather than fabricating
                # a uniform prior for a target-disease pair OT has no
                # data on. Trial outcomes will still update the edge if
                # one is added by attribution.
                continue
            if self.graph._graph.has_edge(  # noqa: SLF001
                tid, ind_id, key=EdgeType.BIOLOGY_DRIVES.value,
            ):
                continue
            belief = score_to_prior(
                row.get("overall_score", 0.0), row.get("evidence_count", 0)
            )
            self.graph.add_edge(GraphEdge(
                source_id=tid,
                target_id=ind_id,
                edge_type=EdgeType.BIOLOGY_DRIVES,
                belief=belief,
                metadata={
                    "source": "opentargets",
                    "efo_id": efo,
                    "overall_score": row.get("overall_score"),
                    "datatypes": row.get("datatypes"),
                },
            ))
            added += 1
        return added

    async def _populate_trial_mechanisms(
        self,
        trials: list[TrialRecord],
        compound_targets: dict[str, list[str]],
    ) -> int:
        """Resolve target + mechanism PER CONSTITUENT and rebuild chains.

        A combo arm tests N mechanism paths simultaneously, so it produces
        N chains—one per constituent compound—rather than collapsing both
        constituents onto a single chain backbone. Mono arms produce one
        chain (unchanged).

        Per constituent we:
          1. Look up the constituent's primary target via OT-resolved
             ``compound_targets`` (first entry; can be UNKNOWN if OT had
             no hit).
          2. Infer a MechanismCategory for that (constituent, target)
             pair via Haiku.
          3. Ensure the MechanismNode exists and add the constituent's
             ``target → mechanism`` modulates_via edge (idempotent).

        The chain list is then rebuilt: for every existing
        subgroup_population_id × arm cell, emit one chain per
        constituent. ``chain.compound_id`` is the constituent (not the
        combo regimen) so binds_to lookup walks to the constituent's own
        target. ``chain.metadata["regimen_id"]`` records the arm's
        regimen for downstream grouping.

        Returns the number of new modulates_via edges added.
        """
        if self._anthropic is None:
            logger.info("No anthropic client; skipping mechanism inference")
            return 0

        cache = JSONCache(self._cache_dir / "mechanism_inferences.json")
        added = 0
        for trial in trials:
            try:
                ts = self.graph.get_trial_subgraph_by_id(trial.nct_id)
            except KeyError:
                continue
            if not ts.arms:
                continue

            # Per-(arm, constituent) backbones. Order within each arm
            # follows arm.compound_ids so chain ordering is deterministic.
            backbones: dict[str, list[dict[str, str]]] = {}

            for arm in ts.arms:
                arm_label = "combo" if arm.is_combination else "monotherapy"
                arm_entries: list[dict[str, str]] = []
                for cid in arm.compound_ids:
                    target_id = (compound_targets.get(cid) or [None])[0]
                    target_node: dict[str, Any] = {}
                    if target_id:
                        try:
                            target_node = self.graph.get_node(target_id)
                        except KeyError:
                            target_node = {}

                    constituent_name = cid
                    try:
                        constituent_name = (
                            self.graph.get_node(cid).get("name") or cid
                        )
                    except KeyError:
                        pass

                    mech_value = await infer_mechanism_for_arm(
                        self._anthropic,
                        trial,
                        arm_label=arm_label,
                        arm_compound_names=[constituent_name],
                        target_node=target_node,
                        cache=cache,
                        cache_key=f"{trial.nct_id}::{arm.arm_id}::{cid}",
                    )
                    mech_id = normalize_entity(mech_value, "MechanismNode")

                    try:
                        self.graph.get_node(mech_id)
                    except KeyError:
                        self.graph.add_node(MechanismNode(
                            id=mech_id,
                            name=mech_id.replace("_", " "),
                            mechanism_type=_category_to_mechanism_type(
                                MechanismCategory(mech_id)
                            ),
                        ))

                    if target_id and not self.graph._graph.has_edge(  # noqa: SLF001
                        target_id, mech_id, key=EdgeType.MODULATES_VIA.value,
                    ):
                        self.graph.add_edge(GraphEdge(
                            source_id=target_id,
                            target_id=mech_id,
                            edge_type=EdgeType.MODULATES_VIA,
                            belief=EdgeBeliefState(alpha=3.0, beta=1.0),
                            metadata={
                                "source": "trial_inference",
                                "trial_id": trial.nct_id,
                                "arm_id": arm.arm_id,
                                "constituent_id": cid,
                            },
                        ))
                        added += 1

                    arm_entries.append({
                        "compound_id": cid,
                        "target_id": target_id or _UNKNOWN,
                        "mechanism_id": mech_id,
                    })
                backbones[arm.arm_id] = arm_entries

            # Rebuild chains: for every (arm, subgroup_pop) cell already
            # in the trial subgraph, emit one chain per constituent. This
            # is idempotent w.r.t. subgroup forks—if seed_responds_differently
            # has already added subgroup pops, we preserve them and just
            # multiply by constituent count.
            arm_by_id = {a.arm_id: a for a in ts.arms}
            seen_cells: dict[str, list[CausalChain]] = {}
            for chain in ts.chains:
                key = f"{chain.arm_id}::{chain.subgroup_population_id}::{chain.endpoint_id}"
                seen_cells.setdefault(key, []).append(chain)

            new_chains: list[CausalChain] = []
            for cell_key, cell_chains in seen_cells.items():
                # Use the first chain in the cell as the template for fields
                # that don't change per constituent (subgroup_pop, endpoint,
                # indication, outcome, effect_size, p_value, metadata).
                template = cell_chains[0]
                arm = arm_by_id.get(template.arm_id)
                if arm is None:
                    new_chains.extend(cell_chains)
                    continue
                entries = backbones.get(arm.arm_id, [])
                if not entries:
                    new_chains.extend(cell_chains)
                    continue
                for entry in entries:
                    md = dict(template.metadata)
                    md["regimen_id"] = arm.regimen_compound_id
                    md["is_constituent"] = arm.is_combination
                    new_chains.append(template.model_copy(update={
                        "compound_id": entry["compound_id"],
                        "target_id": entry["target_id"],
                        "mechanism_id": entry["mechanism_id"],
                        # biology_id will be set in _populate_trial_biology
                        "biology_id": _UNKNOWN,
                        "metadata": md,
                    }))

            self.graph.set_trial_subgraph(
                ts.model_copy(update={"chains": new_chains})
            )

        return added

    # Number of Reactome pathways materialized as distinct BiologyNodes
    # per (target, mechanism) pair. Set to 1 in round 3.4: at single-
    # indication corpus scale clinical trials don't report pathway-
    # level biomarkers, so per-pathway evidence accumulation can't
    # actually discriminate between Reactome pathways for the same
    # target — the cap=3 fan-out just split the same trial signal
    # across 3 nodes. Now we materialize one BiologyNode (the top
    # Reactome match) but record the full pathway list in
    # ``pathway_ids`` metadata so cross-pathway queries remain
    # answerable (and round 4.0 can re-split selectively when pathway
    # biomarker data justifies it).
    _BIOLOGY_PATHWAY_CAP = 1

    async def _resolve_real_biology(
        self,
        target_id: str,
        mechanism_id: str,
        indication_id: str,
        lincs_client: "LINCSClient | None",
        quickgo_client: "QuickGOClient | None" = None,
    ) -> tuple[list[str], bool]:
        """Resolve a chain's biology to real Reactome pathway(s).

        Always keyed on the chain's TARGET — same mechanism with a
        different target maps to different pathways (e.g. nivo's PD-1
        target → "Co-inhibition by PD-1", ipi's CTLA-4 target →
        "Co-inhibition by CTLA4"). The graph's ``mechanism_affects``
        edges accumulate pathways from many targets onto a shared
        MechanismNode, so walking out from the mechanism would mix
        target contexts — we go through Reactome directly instead.

        Resolution order, per fixes.md priority 6 (the "Adding Real
        biology nodes" follow-up):

          1. Query Reactome via the target's gene symbol (cached on
             disk by ``LINCSClient``). Materialize a BiologyNode per
             pathway, capped at the top ``_BIOLOGY_PATHWAY_CAP``
             entries (Reactome returns 3–16 pathways per gene; capping
             prevents the chain count from exploding when combined with
             per-constituent + per-subgroup fan-out). Add
             ``mechanism_affects`` and ``biology_drives`` edges for
             each pathway.
          2. Fall back to the slug ``{mechanism}__{indication}`` only
             as a last resort — when target is UNKNOWN, gene symbol is
             missing, or Reactome has no pathways for the gene. The
             caller tags the chain with
             ``metadata["unresolved_biology"] = True``.

        Returns ``(biology_ids, is_fallback)``. ``biology_ids`` is
        ordered by Reactome's relevance ranking; ``is_fallback`` is
        always False here (fallback is signalled by an empty list, and
        the caller materializes the slug + tags the chain).
        """
        from src.graph.models import BiologyNode  # local: avoids forcing
        # a top-level Pydantic import surface change.

        if mechanism_id == _UNKNOWN or indication_id == _UNKNOWN:
            return ([], False)
        try:
            MechanismCategory(mechanism_id)
        except ValueError:
            return ([], False)

        # Reactome lookup via the target's gene symbol. Reactome is the
        # source of truth for "which pathways contain this protein";
        # LINCSClient already wraps the call + on-disk cache.
        if (
            target_id == _UNKNOWN
            or lincs_client is None
            or not target_id.startswith("ENSG")
        ):
            return ([], False)

        try:
            target_node = self.graph.get_node(target_id)
        except KeyError:
            return ([], False)
        gene_symbol = target_node.get("gene_symbol") or ""
        if not gene_symbol:
            return ([], False)

        try:
            pathways = await lincs_client.get_pathways_for_gene(gene_symbol)
        except Exception:
            logger.debug(
                "Reactome lookup failed for %s (target %s)",
                gene_symbol, target_id, exc_info=True,
            )
            pathways = []

        if not pathways:
            return ([], False)

        # Re-rank by relevance to the chain context (mechanism + indication
        # + gene symbol) before slicing the top-N. Reactome's default order
        # is citation-frequency-driven and routinely puts off-context
        # pathways first (CRBN → SARS therapeutics, VEGFA → platelet
        # degranulation). The full list is still preserved on
        # ``pathway_ids`` so cross-indication queries against the
        # alternates remain answerable.
        from src.graph.pathway_ranker import rerank_pathways, score_candidate

        pathways = rerank_pathways(
            pathways,
            mechanism_name=mechanism_id,
            indication_name=indication_id,
            gene_symbol=gene_symbol,
        )

        # GO augmentation: when Reactome's best context-relevant pathway
        # still scores 0 (no context-token overlap at all), Reactome's
        # curation is failing this gene — fall back to QuickGO biological-
        # process annotations. CRBN is the canonical case: Reactome only
        # has "Potential therapeutics for SARS" for it, while GO has clean
        # terms for protein ubiquitination, proteasome-mediated catabolism,
        # CRL4 complex activity, etc.
        chosen_source = "reactome_target_lookup"
        reactome_candidates = pathways  # remember for metadata trace
        if quickgo_client is not None:
            top_reactome_score = score_candidate(
                pathways[0],
                mechanism_name=mechanism_id,
                indication_name=indication_id,
                gene_symbol=gene_symbol,
            )
            if top_reactome_score == 0:
                from src.graph import hgnc_resolver

                uniprot_acc = hgnc_resolver.uniprot_for_symbol(gene_symbol)
                if uniprot_acc:
                    try:
                        go_terms = await quickgo_client.get_terms_for_uniprot(
                            uniprot_acc,
                        )
                    except Exception:
                        logger.debug(
                            "QuickGO lookup failed for %s (%s)",
                            gene_symbol, uniprot_acc, exc_info=True,
                        )
                        go_terms = []
                    if go_terms:
                        go_terms = rerank_pathways(
                            go_terms,
                            mechanism_name=mechanism_id,
                            indication_name=indication_id,
                            gene_symbol=gene_symbol,
                        )
                        top_go_score = score_candidate(
                            go_terms[0],
                            mechanism_name=mechanism_id,
                            indication_name=indication_id,
                            gene_symbol=gene_symbol,
                        )
                        if top_go_score > top_reactome_score:
                            pathways = go_terms
                            chosen_source = "quickgo_target_lookup"

        # Build the alternate lists for whichever source won. When GO wins,
        # also keep the Reactome candidates we considered (by id+name) so
        # the audit trail isn't lost — the Reactome list is what Reactome
        # *thinks* CRBN's biology is, even if we chose to rely on GO.
        all_pathway_ids = [p.stable_id for p in pathways]
        pathway_display_names = {p.stable_id: p.display_name for p in pathways}
        reactome_alternatives_meta = (
            {p.stable_id: p.display_name for p in reactome_candidates}
            if chosen_source == "quickgo_target_lookup" else {}
        )

        biology_ids: list[str] = []
        for pathway in pathways[: self._BIOLOGY_PATHWAY_CAP]:
            bio_id = pathway.stable_id
            alternate_names = {
                sid: name for sid, name in pathway_display_names.items()
                if sid != bio_id
            }
            try:
                existing = self.graph.get_node(bio_id)
                # Merge any newly-discovered pathway ids into the
                # existing node's pathway_ids list. Idempotent on repeat
                # rebuilds and lets the metadata grow as more trials
                # touch the same target+mechanism.
                current_pathways = set(existing.get("pathway_ids") or [])
                merged = current_pathways | set(all_pathway_ids)
                if merged != current_pathways:
                    existing["pathway_ids"] = sorted(merged)
                # Merge alternate display names too — same idempotent shape.
                existing_meta = existing.setdefault("metadata", {})
                existing_alts = existing_meta.setdefault("alternate_pathway_names", {})
                for sid, name in alternate_names.items():
                    if sid not in existing_alts and name:
                        existing_alts[sid] = name
                if reactome_alternatives_meta:
                    existing_reactome = existing_meta.setdefault(
                        "reactome_alternatives", {}
                    )
                    for sid, name in reactome_alternatives_meta.items():
                        if sid not in existing_reactome and name:
                            existing_reactome[sid] = name
            except KeyError:
                node_metadata: dict[str, Any] = {
                    "source": chosen_source,
                    "primary_pathway": bio_id,
                    "alternate_pathway_names": alternate_names,
                }
                if reactome_alternatives_meta:
                    node_metadata["reactome_alternatives"] = reactome_alternatives_meta
                self.graph.add_node(BiologyNode(
                    id=bio_id,
                    name=pathway.display_name or bio_id,
                    pathway_ids=all_pathway_ids,
                    metadata=node_metadata,
                ))

            if not self.graph._graph.has_edge(  # noqa: SLF001
                mechanism_id, bio_id, key=EdgeType.MECHANISM_AFFECTS.value,
            ):
                self.graph.add_edge(GraphEdge(
                    source_id=mechanism_id,
                    target_id=bio_id,
                    edge_type=EdgeType.MECHANISM_AFFECTS,
                    belief=EdgeBeliefState(alpha=2.0, beta=1.0),
                    metadata={
                        "source": chosen_source,
                        "gene_symbol": gene_symbol,
                    },
                ))

            if not self.graph._graph.has_edge(  # noqa: SLF001
                bio_id, indication_id, key=EdgeType.BIOLOGY_DRIVES.value,
            ):
                prior = EdgeBeliefState(alpha=1.0, beta=1.0)
                try:
                    prior = self.graph.get_edge_belief(
                        target_id, indication_id, EdgeType.BIOLOGY_DRIVES,
                    )
                except KeyError:
                    pass
                self.graph.add_edge(GraphEdge(
                    source_id=bio_id,
                    target_id=indication_id,
                    edge_type=EdgeType.BIOLOGY_DRIVES,
                    belief=EdgeBeliefState(
                        alpha=prior.alpha, beta=prior.beta,
                    ),
                    metadata={
                        "source": "reactome_target_lookup",
                        "borrowed_from": f"{target_id}->{indication_id}",
                    },
                ))
            biology_ids.append(bio_id)

        return (biology_ids, False)

    async def _populate_trial_biology(self, trials: list[TrialRecord]) -> int:
        """Resolve every chain's biology to real Reactome pathways when
        possible; fan chains out per pathway.

        Resolution per chain:
          1. LINCS-wired Reactome biology already on the graph (preferred).
          2. Reactome API lookup keyed on the target's gene symbol
             (``LINCSClient.get_pathways_for_gene``, cached).
          3. ``{mechanism}__{indication}`` slug fallback, tagged
             ``metadata["unresolved_biology"] = True`` for downstream
             auditing.

        When step 1 or 2 returns multiple pathway nodes, the chain fans
        out one chain per pathway — each pathway is a distinct
        biological hypothesis and accumulates its own evidence. Capped
        at ``_BIOLOGY_PATHWAY_CAP`` per (target, mechanism) pair.

        Returns the count of new biology nodes + chain rewrites combined.
        """
        from src.graph.models import BiologyNode  # local import

        # One LINCSClient per pipeline run so the Reactome HTTP cache +
        # on-disk JSON cache are shared across trials. If the env doesn't
        # have CLUE_API_KEY set, LINCSClient construction may fail —
        # but get_pathways_for_gene only uses the Reactome endpoint
        # which is keyless, so we tolerate either path here.
        lincs_client: "LINCSClient | None"
        try:
            lincs_client = LINCSClient()
        except Exception:
            logger.debug("LINCSClient unavailable", exc_info=True)
            lincs_client = None

        # QuickGO is keyless and used only as a fallback when Reactome's
        # best pathway has zero context overlap. Constructed once per run
        # so the disk cache is shared.
        from src.ingestion.quickgo import QuickGOClient
        quickgo_client = QuickGOClient()

        added = 0
        for trial in trials:
            try:
                ts = self.graph.get_trial_subgraph_by_id(trial.nct_id)
            except KeyError:
                continue
            if not ts.chains:
                continue

            # Resolve once per (target, mechanism, indication) tuple so
            # each Reactome lookup is amortized across all chains that
            # share that backbone.
            resolution_cache: dict[
                tuple[str, str, str], tuple[list[str], bool]
            ] = {}

            new_chains: list[CausalChain] = []
            for chain in ts.chains:
                mech_id = chain.mechanism_id
                ind_id = chain.indication_id
                target_id = chain.target_id
                if mech_id == _UNKNOWN or ind_id == _UNKNOWN:
                    new_chains.append(chain)
                    continue

                key = (target_id, mech_id, ind_id)
                if key in resolution_cache:
                    biology_ids, _ = resolution_cache[key]
                else:
                    biology_ids, _ = await self._resolve_real_biology(
                        target_id, mech_id, ind_id, lincs_client,
                        quickgo_client=quickgo_client,
                    )
                    resolution_cache[key] = (biology_ids, False)

                if not biology_ids:
                    # Step 3: slug fallback. Seed the slug biology + its
                    # mechanism_affects / biology_drives edges (same as
                    # the legacy path), tag the chain as unresolved.
                    slug_id = normalize_entity(
                        f"{mech_id}__{ind_id}", "BiologyNode",
                    )
                    try:
                        self.graph.get_node(slug_id)
                    except KeyError:
                        self.graph.add_node(BiologyNode(
                            id=slug_id,
                            name=(
                                f"{mech_id.replace('_', ' ')} biology in "
                                f"{ind_id}"
                            ),
                            pathway_ids=[],
                            metadata={"source": "trial_biology_fallback"},
                        ))
                        added += 1

                    if not self.graph._graph.has_edge(  # noqa: SLF001
                        mech_id, slug_id,
                        key=EdgeType.MECHANISM_AFFECTS.value,
                    ):
                        self.graph.add_edge(GraphEdge(
                            source_id=mech_id,
                            target_id=slug_id,
                            edge_type=EdgeType.MECHANISM_AFFECTS,
                            belief=EdgeBeliefState(alpha=2.0, beta=1.0),
                            metadata={"source": "trial_biology_fallback"},
                        ))

                    if not self.graph._graph.has_edge(  # noqa: SLF001
                        slug_id, ind_id,
                        key=EdgeType.BIOLOGY_DRIVES.value,
                    ):
                        prior = EdgeBeliefState(alpha=1.0, beta=1.0)
                        if target_id != _UNKNOWN:
                            try:
                                prior = self.graph.get_edge_belief(
                                    target_id, ind_id,
                                    EdgeType.BIOLOGY_DRIVES,
                                )
                            except KeyError:
                                pass
                        self.graph.add_edge(GraphEdge(
                            source_id=slug_id,
                            target_id=ind_id,
                            edge_type=EdgeType.BIOLOGY_DRIVES,
                            belief=EdgeBeliefState(
                                alpha=prior.alpha, beta=prior.beta,
                            ),
                            metadata={
                                "source": "trial_biology_fallback",
                                "borrowed_from": (
                                    f"{target_id}->{ind_id}"
                                    if target_id != _UNKNOWN else None
                                ),
                            },
                        ))

                    md = dict(chain.metadata)
                    md["unresolved_biology"] = True
                    new_chains.append(chain.model_copy(update={
                        "biology_id": slug_id,
                        "metadata": md,
                    }))
                    continue

                # One Reactome pathway → set biology_id. Multiple → fan
                # out one chain per pathway (different biological
                # hypotheses tested by the same trial cell).
                if len(biology_ids) == 1:
                    bio_id = biology_ids[0]
                    if chain.biology_id == bio_id:
                        new_chains.append(chain)
                    else:
                        md = dict(chain.metadata)
                        md.pop("unresolved_biology", None)
                        new_chains.append(chain.model_copy(update={
                            "biology_id": bio_id,
                            "metadata": md,
                        }))
                        added += 1
                else:
                    for bio_id in biology_ids:
                        md = dict(chain.metadata)
                        md.pop("unresolved_biology", None)
                        new_chains.append(chain.model_copy(update={
                            "biology_id": bio_id,
                            "metadata": md,
                        }))
                        added += 1

            if new_chains != list(ts.chains):
                self.graph.set_trial_subgraph(
                    ts.model_copy(update={"chains": new_chains})
                )
        return added

    async def _populate_canonical_endpoints(
        self, trials: list[TrialRecord]
    ) -> int:
        """Classify each trial's primary outcomes and create canonical endpoint nodes.

        For each (trial, primary_outcome, indication) triple:
          1. LLM-classify the outcome's measure into an ``EndpointClass``.
          2. Create (if missing) ``EndpointNode`` with id ``{class}_{indication_id}``.
          3. Seed an ``endpoint_captures`` edge with the class-specific prior.

        Skipped silently if no Anthropic client was provided (lets the
        pipeline run in offline / fixture mode).

        Returns the number of *new* endpoint nodes created.
        """
        if self._anthropic is None:
            logger.info("No anthropic client; skipping canonical endpoint creation")
            return 0

        cache = JSONCache(self._cache_dir / "endpoint_classifications.json")
        added = 0
        for trial in trials:
            indication_ids: list[tuple[str, str]] = []
            for cond in trial.conditions:
                ind_id = self.resolve_entity(cond, "indication")
                if ind_id:
                    indication_ids.append((ind_id, cond))

            for om in trial.primary_outcomes:
                if not om.measure:
                    continue
                cls_value = await classify_endpoint_with_llm(
                    self._anthropic, om.measure, cache
                )
                try:
                    cls = EndpointClass(cls_value)
                except ValueError:
                    cls = EndpointClass.OTHER

                for ind_id, ind_name in indication_ids:
                    # Anchor endpoints on the parent indication so a single
                    # `PFS_melanoma` node serves every melanoma subtype.
                    # Subtype-specific endpoints would fragment evidence
                    # accumulation across nodes that mean the same thing.
                    root_ind_id = _root_indication(ind_id)
                    ep_id = normalize_entity(
                        f"{cls.value}_{root_ind_id}", "EndpointNode"
                    )
                    try:
                        self.graph.get_node(ep_id)
                    except KeyError:
                        self.graph.add_node(EndpointNode(
                            id=ep_id,
                            name=f"{cls.value} ({ind_name})",
                            endpoint_type=EndpointType.PRIMARY,
                            regulatory_status=RegulatoryStatus.EXPLORATORY,
                            measurement_properties={
                                "endpoint_class": cls.value,
                                "indication_id": root_ind_id,
                            },
                        ))
                        added += 1
                    # Always index the measure string—even when reusing an
                    # existing (class, indication) node. Otherwise the second
                    # trial whose primary outcome maps to the same EndpointClass
                    # but uses different wording (e.g. "PFS by investigator" vs
                    # "PFS by BICR" vs the canonical name) won't be findable
                    # by ``resolve_entity`` and gets silently skipped in
                    # ``build_trial_subgraphs``.
                    self._index_node(ep_id, om.measure, "endpoint")
                    seed_endpoint_captures_edge(
                        self.graph, ep_id, om.measure, ind_id, cls.value
                    )
        return added

    def _add_peptide_vaccine_target_edges(
        self, trials: list[TrialRecord],
    ) -> tuple[int, dict[str, list[str]]]:
        """Wire AFFECTS edges for curated melanoma TAA peptide-vaccine targets.

        Open Targets returns no targets for tumor-associated-antigen peptide
        vaccines (gp100, tyrosinase, MART-1, multi-epitope vaccines).
        ``_PEPTIDE_VACCINE_TARGETS`` is the curated mapping that fills in
        those known biology connections so chains for these compounds
        resolve past UNKNOWN_target. Creates TargetNodes when missing.

        Returns ``(edges_added, compound_targets)`` where ``compound_targets``
        maps each matched compound_id to its list of resolved Ensembl ids.
        The caller merges this into the main ``compound_targets`` dict so
        downstream chain construction (``_populate_trial_mechanisms``)
        sees the peptide vaccines as having real targets.
        """
        added = 0
        seen_compounds: set[str] = set()
        compound_targets: dict[str, list[str]] = {}
        for trial in trials:
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                cid = self.resolve_entity(iv.name, "compound")
                if not cid or cid in seen_compounds:
                    continue
                lowered = iv.name.lower()
                matched_targets: dict[str, tuple[str, str]] = {}
                for pattern, targets in _PEPTIDE_VACCINE_TARGETS:
                    if pattern in lowered:
                        for ens, symbol, name in targets:
                            matched_targets.setdefault(ens, (symbol, name))
                if not matched_targets:
                    continue
                seen_compounds.add(cid)
                compound_targets[cid] = list(matched_targets.keys())
                for ens, (symbol, name) in matched_targets.items():
                    try:
                        self.graph.get_node(ens)
                    except KeyError:
                        self.graph.add_node(TargetNode(
                            id=ens,
                            name=name,
                            gene_symbol=symbol,
                            metadata={"source": "peptide_vaccine_heuristic"},
                        ))
                        self._index_node(ens, symbol, "target")
                    if self.graph._graph.has_edge(  # noqa: SLF001
                        cid, ens, key=EdgeType.AFFECTS.value,
                    ):
                        continue
                    self.graph.add_edge(GraphEdge(
                        source_id=cid,
                        target_id=ens,
                        edge_type=EdgeType.AFFECTS,
                        belief=EdgeBeliefState(alpha=3.0, beta=1.0),
                        metadata={
                            "source": "peptide_vaccine_heuristic",
                            "pattern_matched": iv.name,
                        },
                    ))
                    added += 1
        return added, compound_targets

    def _add_compound_target_edges(self, trials: list[TrialRecord]) -> int:
        """For compounds with known targets in the graph, add binds_to edges."""
        added = 0
        targets_in_graph = {
            n["id"] for n in self.graph.get_nodes_by_type("TargetNode")
        }
        for trial in trials:
            drug_interventions = [
                iv for iv in trial.interventions if is_drug_like(iv)
            ]
            for iv in drug_interventions:
                comp_id = self.resolve_entity(iv.name, "compound")
                if not comp_id:
                    continue
                # Try to find a target whose name/symbol appears in the
                # trial title or intervention description
                text = f"{trial.title} {iv.description}".lower()
                for target_id in targets_in_graph:
                    try:
                        tdata = self.graph.get_node(target_id)
                    except KeyError:
                        continue
                    symbol = tdata.get("gene_symbol", "").lower()
                    name = tdata.get("name", "").lower()
                    if symbol and len(symbol) >= 3 and symbol in text:
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
                    elif name and len(name) >= 5 and name in text:
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
        return added

    def _add_binds_edge(self, compound_id: str, target_id: str) -> None:
        edge = GraphEdge(
            source_id=compound_id,
            target_id=target_id,
            edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=2.0, beta=1.5),
            metadata={"source": "cross_reference", "method": "name_matching"},
        )
        self.graph.add_edge(edge)

    # ── Trial subgraph construction ──────────────────────────────────────

    def build_trial_subgraphs(
        self, trials: list[TrialRecord]
    ) -> list[TrialSubgraph]:
        """Build skeleton TrialSubgraphs (TrialNode + arms + parent population).

        Produces one chain per arm at the trial's qualified default
        PopulationNode (see step 2 of ``populate_oncology``). For a trial
        whose condition is "Stage IIIC Cutaneous Melanoma", the parent
        population is ``melanoma__histology_cutaneous__stage_iii``; for a
        trial whose condition has no parseable qualifiers it falls back
        to ``{indication}__unselected``. Biomarker-derived subgroups
        added later fork off this qualified default.
        """
        subgraphs: list[TrialSubgraph] = []
        canonical_lookup = getattr(self, "_trial_canonical_indication", {})
        default_pop_lookup = getattr(self, "_trial_default_population", {})

        for trial in trials:
            indication_id = canonical_lookup.get(trial.nct_id)
            if indication_id is None:
                for cond in trial.conditions:
                    indication_id = self.resolve_entity(cond, "indication")
                    if indication_id:
                        break
            if not indication_id:
                continue

            endpoint_id = None
            for om in trial.primary_outcomes:
                endpoint_id = self.resolve_entity(om.measure, "endpoint")
                if endpoint_id:
                    break
            if not endpoint_id:
                continue

            seed_trial_node(self.graph, trial)
            arms = build_arms(trial, self.resolve_entity)
            if not arms:
                continue
            synthesize_combo_compounds(self.graph, arms)

            parent_pop_id = default_pop_lookup.get(trial.nct_id)
            if not parent_pop_id:
                parent_pop_id = ensure_parent_population(
                    self.graph,
                    indication_id,
                    indication_name=(
                        trial.conditions[0] if trial.conditions else indication_id
                    ),
                )

            # Chain backbone anchors on the parent disease so evidence
            # from melanoma-subtype trials still accumulates on the
            # `melanoma` node. The subtype IndicationNode + SUBTYPE_OF
            # edge created upstream stay intact for cross-rollup queries,
            # and the trial's population (built from the subtype id
            # above) preserves the subtype distinction.
            chain_indication_id = _root_indication(indication_id)
            chains = [
                CausalChain(
                    arm_id=arm.arm_id,
                    compound_id=arm.regimen_compound_id,
                    subgroup_population_id=parent_pop_id,
                    target_id=_UNKNOWN,
                    mechanism_id=_UNKNOWN,
                    biology_id=_UNKNOWN,
                    indication_id=chain_indication_id,
                    endpoint_id=endpoint_id,
                    outcome=TrialOutcome.UNKNOWN,
                )
                for arm in arms
            ]

            ts = TrialSubgraph(
                trial_id=trial.nct_id,
                phase=trial.phase or "unknown",
                arms=arms,
                chains=chains,
                parent_population_id=parent_pop_id,
                metadata={
                    "title": trial.title,
                    "status": trial.status,
                    "enrollment": trial.enrollment,
                },
            )
            self.graph.set_trial_subgraph(ts)
            subgraphs.append(ts)
        return subgraphs


# ── Module-level helpers (usable from trace drivers + tests) ─────────────


def seed_trial_node(graph: GraphStore, trial: TrialRecord) -> str:
    """Add (or update) a TrialNode marker in the graph. Returns the node id."""
    node = TrialNode(
        id=trial.nct_id,
        name=trial.title or trial.nct_id,
        phase=trial.phase or "",
        status=trial.status or "",
        sponsor=trial.sponsor or "",
        enrollment=trial.enrollment,
        metadata={
            "conditions": list(trial.conditions),
            "has_results": trial.has_results,
        },
    )
    graph.add_node(node)
    return node.id


def build_arms(
    trial: TrialRecord,
    resolve_compound: Callable[[str, str], str | None] | None = None,
) -> list[TrialArm]:
    """Construct TrialArm objects from the trial's parsed arm_groups.

    ``resolve_compound`` (optional) maps an intervention name to its canonical
    CompoundNode id; if absent, the slug of the intervention name is used as
    the compound id (the trace driver and tests can populate the graph
    accordingly). Combo arms get a synthesized regimen id of the form
    ``compoundA+compoundB`` (constituents sorted).

    Non-drug intervention names (procedures, diagnostics, radiation,
    devices) are filtered out before arm construction. CT.gov lists
    these alongside actual drug interventions in each arm group's
    intervention_names; without filtering they become orphan untyped
    compound nodes that pollute the graph and confuse classifier routing.

    Filter policy: drop an arm intervention name only when it is
    *explicitly listed* in ``trial.interventions`` with a non-drug type
    (PROCEDURE / RADIATION / DEVICE / DIAGNOSTIC_TEST / OTHER). Names
    that aren't in the Intervention manifest at all pass through —
    those include combo constituents listed only via arm groups (e.g.
    "Dasatinib" referenced only in an arm description) where dropping
    them would silently collapse a combo into a mono arm.
    """
    non_drug_names = {
        iv.name for iv in trial.interventions
        if iv.name and not is_drug_like(iv)
    }

    arms: list[TrialArm] = []
    seen_arm_ids: set[str] = set()
    for ag in trial.arm_groups:
        if not ag.intervention_names:
            continue
        relevant_names = [
            n for n in ag.intervention_names if n not in non_drug_names
        ]
        if not relevant_names:
            continue
        compound_ids: list[str] = []
        for iv_name in relevant_names:
            cid = (
                resolve_compound(iv_name, "compound")
                if resolve_compound is not None
                else None
            )
            if not cid:
                cid = normalize_entity(iv_name, "InterventionNode")
            compound_ids.append(cid)

        # Drop duplicates while preserving order—some trials list the same
        # intervention twice in an arm group description.
        seen_ids: set[str] = set()
        unique_ids: list[str] = []
        for cid in compound_ids:
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_ids.append(cid)
        if not unique_ids:
            continue

        if len(unique_ids) == 1:
            regimen_id = unique_ids[0]
            is_combo = False
        else:
            regimen_id = "+".join(sorted(unique_ids))
            is_combo = True

        arm_id = ag.group_id or _slug_for_arm(unique_ids)
        # Disambiguate any collision (e.g. two arm groups slugged identically).
        base_arm_id = arm_id
        suffix = 2
        while arm_id in seen_arm_ids:
            arm_id = f"{base_arm_id}_{suffix}"
            suffix += 1
        seen_arm_ids.add(arm_id)

        arms.append(TrialArm(
            arm_id=arm_id,
            compound_ids=unique_ids,
            regimen_compound_id=regimen_id,
            is_combination=is_combo,
        ))
    return arms


def _slug_for_arm(compound_ids: list[str]) -> str:
    return "_".join(sorted(compound_ids))


def synthesize_combo_compounds(graph: GraphStore, arms: list[TrialArm]) -> int:
    """For each combo arm, create a synthesized CompoundNode + composed_of edges.

    Idempotent—if the combo CompoundNode already exists, only missing
    composed_of edges are added. Returns the count of new combo nodes
    created (not counting edges).
    """
    created = 0
    for arm in arms:
        if not arm.is_combination:
            continue
        try:
            graph.get_node(arm.regimen_compound_id)
        except KeyError:
            # Build a human-readable name from the constituents in the
            # graph (fall back to the id segments when names are missing).
            constituent_names: list[str] = []
            for cid in arm.compound_ids:
                try:
                    cdata = graph.get_node(cid)
                    constituent_names.append(cdata.get("name") or cid)
                except KeyError:
                    constituent_names.append(cid)
            graph.add_node(CompoundNode(
                id=arm.regimen_compound_id,
                name=" + ".join(constituent_names),
                modality=Modality.OTHER,
                metadata={"synthesized": "combo", "constituents": list(arm.compound_ids)},
            ))
            created += 1
        # composed_of edges (one per constituent). add_edge is idempotent
        # on (source, target, edge_type) thanks to MultiDiGraph keying.
        for cid in arm.compound_ids:
            graph.add_edge(GraphEdge(
                source_id=arm.regimen_compound_id,
                target_id=cid,
                edge_type=EdgeType.COMPOSED_OF,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
                metadata={"source": "combo_synthesis"},
            ))
        # Inherit binds_to from each constituent. The combo doesn't have
        # its own ChEMBL record; its target engagement is the union of
        # its constituents'. Use a slightly weaker prior (Beta(3, 1.5))
        # than the OT-direct edges (Beta(4, 1)) since the inherited claim
        # rests on the constituent's evidence rather than a direct lookup.
        propagated: set[str] = set()
        for cid in arm.compound_ids:
            for edge in graph.get_neighboring_edges(
                cid, edge_types=[EdgeType.AFFECTS],
            ):
                target_id = edge["target_id"]
                if target_id in propagated:
                    continue
                propagated.add(target_id)
                if graph._graph.has_edge(  # noqa: SLF001
                    arm.regimen_compound_id,
                    target_id,
                    key=EdgeType.AFFECTS.value,
                ):
                    continue
                graph.add_edge(GraphEdge(
                    source_id=arm.regimen_compound_id,
                    target_id=target_id,
                    edge_type=EdgeType.AFFECTS,
                    belief=EdgeBeliefState(alpha=3.0, beta=1.5),
                    metadata={
                        "source": "combo_inherit",
                        "via_constituent": cid,
                    },
                ))
    return created


def ensure_parent_population(
    graph: GraphStore,
    indication_id: str,
    *,
    indication_name: str | None = None,
) -> str:
    """Create (if missing) the trial's parent enrollment PopulationNode.

    The parent represents "all patients meeting trial enrollment criteria"
   —used as the default subgroup_population_id when no biomarker
    stratifier was reported.
    """
    pop_id = PopulationNode.compose_id(indication_id, [])
    try:
        graph.get_node(pop_id)
    except KeyError:
        graph.add_node(PopulationNode(
            id=pop_id,
            name=f"All patients ({indication_name or indication_id})",
            defining_features=[],
        ))
    return pop_id


def build_trial_subgraph_from_extraction(
    graph: GraphStore,
    trial: TrialRecord,
    extraction: Any,
    *,
    target_by_arm: dict[str, str],
    mechanism_id: str,
    biology_id: str,
    indication_id: str,
    endpoint_ids: dict[str, str],
) -> TrialSubgraph:
    """Compose the full chain-list TrialSubgraph from an extraction.

    Uses CT.gov-parsed ``trial.arm_groups`` for arm structure (deterministic;
    extraction.arms is a sanity check, not a primary source). Subgroups
    come from ``extraction.subgroups`` (LLM-canonicalized via the
    extraction call's vocabulary). Per-chain outcomes come from
    ``extraction.results_by_chain``.

    Cardinality: chains = N arms × M reported subgroups × K endpoints.
    Each chain references one endpoint, since PFS and OS measure the
    same biology through different lenses and a chain models one
    measurement.

    Args:
        target_by_arm: maps each arm_id to the canonical TargetNode id
            for that arm's primary target. Caller resolves these (e.g.
            via ``OpenTargetsClient.search_target``) so the function
            stays free of network/API concerns.
        endpoint_ids: maps EndpointClass value (e.g. "PFS", "OS") to the
            corresponding EndpointNode id already in the graph. The keys
            of this dict drive the endpoint fan-out—one chain per
            (arm × subgroup × endpoint) cell.

    The result is persisted to ``graph.trial_subgraphs[trial.nct_id]``.
    """
    # Lazy imports to avoid circular dependencies and keep populate's
    # core path independent of taxonomy at import time.
    from src.annotation.taxonomy import TrialExtraction  # noqa: F401
    from src.graph.subgroup_taxonomy import (
        canonicalize_feature,
        is_canonical,
        log_unmapped,
    )

    if not endpoint_ids:
        raise ValueError(
            f"{trial.nct_id} has no endpoint_ids; need at least one "
            "(EndpointClass → EndpointNode id) entry"
        )

    seed_trial_node(graph, trial)
    arms = build_arms(trial)
    if not arms:
        raise ValueError(
            f"{trial.nct_id} has no arm groups; cannot build trial subgraph"
        )
    synthesize_combo_compounds(graph, arms)

    indication_name = trial.conditions[0] if trial.conditions else None
    parent_pop_id = ensure_parent_population(
        graph, indication_id, indication_name=indication_name,
    )
    # Chain backbone anchors on the parent disease — populations keep
    # their subtype-keyed slug so subtype distinctions survive at the
    # patient-cohort level, but the chain's indication_id collapses to
    # the parent so melanoma-subtype trials all contribute to the same
    # `melanoma` IndicationNode for evidence accumulation.
    chain_indication_id = _root_indication(indication_id)

    # Canonicalize subgroup features: descriptor → list[SubgroupFeature].
    # Unknown axes get logged for vocabulary-extension review. Subgroups
    # whose features all collapse to ``axis="other"`` are dropped—these
    # are typically PD readouts ("CD8 T cells per mm² day 22") or analysis
    # timepoints ("Primary completion") that aren't real patient subgroups.
    # Letting them through produces one-off PopulationNodes that no other
    # trial can match against.
    subgroup_features_by_descriptor: dict[str, list[SubgroupFeature]] = {}
    subgroup_pop_ids: dict[str, str] = {}
    for sg in getattr(extraction, "subgroups", []) or []:
        feats: list[SubgroupFeature] = []
        for f in sg.features:
            cf = canonicalize_feature(
                f.get("axis", ""),
                f.get("key", ""),
                f.get("level", ""),
                raw_descriptor=sg.raw_descriptor,
            )
            if not is_canonical(cf):
                log_unmapped(cf, trial.nct_id)
            feats.append(cf)
        if not any(is_canonical(f) for f in feats):
            continue
        # Response-axis features (CR/PR/SD/PD) are outcome stratifiers,
        # not patient strata — they describe what happened to subgroups
        # of patients post-treatment, so they shouldn't fork the trial's
        # arm × population chain matrix. Drop subgroups whose only
        # canonical features are response-axis; keep them when paired
        # with a patient-stratifying axis (gene, line, biomarker).
        if all(
            (not is_canonical(f)) or f.axis == "response"
            for f in feats
        ):
            continue
        subgroup_features_by_descriptor[sg.raw_descriptor] = feats
        pop_id = PopulationNode.compose_id(indication_id, feats)
        try:
            graph.get_node(pop_id)
        except KeyError:
            graph.add_node(PopulationNode(
                id=pop_id,
                name=f"{sg.raw_descriptor} ({indication_name or indication_id})",
                defining_features=list(feats),
            ))
        subgroup_pop_ids[sg.raw_descriptor] = pop_id

    # Index per-chain results by (arm_id, pop_id, endpoint_class).
    # Multiple raw descriptors may collapse onto the same canonical pop_id
    # (e.g. PD-L1 ≥1% and PD-L1 ≥5% both → cd274_high)—keying on pop_id
    # rather than descriptor lets results across those descriptors land
    # on the same chain. Last-write-wins on collision.
    results_index: dict[tuple[str, str | None, str], Any] = {}
    for cr in getattr(extraction, "results_by_chain", []) or []:
        ep_class = classify_endpoint_deterministic(cr.endpoint)
        if cr.subgroup_descriptor is None:
            pop_key: str | None = None
        else:
            pop_key = subgroup_pop_ids.get(cr.subgroup_descriptor)
            if pop_key is None:
                # Descriptor wasn't extracted as a subgroup—skip.
                continue
        results_index[(cr.arm_id, pop_key, ep_class)] = cr

    # Determine the (subgroup_descriptor, pop_id) cells to fan over.
    # Multiple raw descriptors can collapse onto the same canonical
    # pop_id (e.g. "PD-L1 ≥1%" and "PD-L1 ≥5%" both → cd274_high). Dedupe
    # by pop_id so we don't multiply the chain count by descriptor count.
    target_subgroups: list[tuple[str | None, str]]
    if subgroup_pop_ids:
        seen_pop_ids: set[str] = set()
        target_subgroups = []
        for descriptor, pop_id in subgroup_pop_ids.items():
            if pop_id in seen_pop_ids:
                continue
            seen_pop_ids.add(pop_id)
            target_subgroups.append((descriptor, pop_id))
    else:
        target_subgroups = [(None, parent_pop_id)]

    # Fan chains across arms × subgroups × endpoints. Each chain models
    # one measurement window into the same upstream biology.
    chains: list[CausalChain] = []
    for arm in arms:
        arm_target_id = target_by_arm.get(arm.arm_id, _UNKNOWN)
        for descriptor, pop_id in target_subgroups:
            for ep_class, ep_id in endpoint_ids.items():
                # Result lookup uses pop_id (canonical), not descriptor —
                # see results_index construction above.
                pop_key_for_lookup: str | None = (
                    None if descriptor is None else pop_id
                )
                cr = results_index.get((arm.arm_id, pop_key_for_lookup, ep_class))
                outcome = TrialOutcome.UNKNOWN
                effect_size = None
                p_value = None
                if cr is not None:
                    try:
                        outcome = TrialOutcome(cr.outcome)
                    except ValueError:
                        outcome = TrialOutcome.UNKNOWN
                    effect_size = cr.effect_size
                    p_value = cr.p_value
                chains.append(CausalChain(
                    arm_id=arm.arm_id,
                    compound_id=arm.regimen_compound_id,
                    subgroup_population_id=pop_id,
                    target_id=arm_target_id,
                    mechanism_id=mechanism_id,
                    biology_id=biology_id,
                    indication_id=chain_indication_id,
                    endpoint_id=ep_id,
                    outcome=outcome,
                    effect_size=effect_size,
                    p_value=p_value,
                    metadata={"endpoint_class": ep_class},
                ))

    ts = TrialSubgraph(
        trial_id=trial.nct_id,
        phase=trial.phase or "",
        arms=arms,
        chains=chains,
        parent_population_id=parent_pop_id,
        metadata={
            "title": trial.title,
            "status": trial.status,
            "enrollment": trial.enrollment,
        },
    )
    graph.set_trial_subgraph(ts)
    return ts


async def seed_responds_differently_from_extractions(
    graph: GraphStore,
    annotations_dir: Path,
) -> tuple[int, int]:
    """Walk per-trial extractions, seed subgroup populations, and fork chains.

    For each extraction's ``subgroups``:
      1. Canonicalize features into a ``SubgroupFeature`` list.
      2. Compose / create a PopulationNode keyed off the trial's
         indication.
      3. Add a population→indication ``responds_differently`` edge with
         prior Beta(1.5, 1).
      4. Fork the trial's chains. A chain is forked into the subgroup
         population ONLY when the subgroup is mechanistically relevant
         to the chain's target — that is, when the subgroup's biomarker
         gene appears in any Reactome pathway containing the chain's
         target gene. Subgroups with no gene biomarker (line of
         therapy, performance status, age) are universally relevant and
         fork every chain. (fixes.md #3 — PD-L1 stratification was
         being forked onto ipilimumab chains where it isn't
         mechanistically meaningful, since CTLA-4's pathways don't
         contain CD274.)

    Returns ``(edges_added, chains_added)``.

    Async because the relevance check needs Reactome lookups
    (``LINCSClient.get_pathways_for_gene`` and
    ``get_pathway_gene_symbols``, both on-disk cached). When LINCS
    construction fails (no httpx, offline) we fall back to the legacy
    "always fork" behaviour so tests can still run without network.

    Lookup expectations:
      - ``graph.trial_subgraphs[nct_id]`` exists (created by
        ``build_trial_subgraphs`` upstream) and has at least one chain
        whose ``indication_id`` is the trial's canonical indication.
      - Subgroup features have axis/key/level matching the extraction
        prompt's contract (``ExtractedSubgroup``).
    """
    from src.graph.subgroup_taxonomy import (
        canonicalize_feature,
        is_canonical,
        log_unmapped,
    )

    lincs_client: "LINCSClient | None"
    try:
        lincs_client = LINCSClient()
    except Exception:
        logger.debug("LINCSClient unavailable for subgroup relevance", exc_info=True)
        lincs_client = None

    # gene_symbol → set of pathway gene-symbol participants reached via
    # this gene's Reactome pathways. Populated lazily; persists for the
    # whole run.
    pathway_gene_cache: dict[str, set[str]] = {}

    async def _pathway_universe_for_gene(gene_symbol: str) -> set[str]:
        """Genes that share a Reactome pathway with ``gene_symbol``.

        Used to test mechanistic relatedness: if subgroup biomarker
        gene G is in this set for the chain's target T, T and G are in
        the same Reactome pathway and the subgroup is biologically
        meaningful for this chain.
        """
        if not gene_symbol:
            return set()
        cached = pathway_gene_cache.get(gene_symbol)
        if cached is not None:
            return cached
        if lincs_client is None:
            pathway_gene_cache[gene_symbol] = set()
            return set()
        universe: set[str] = {gene_symbol}
        try:
            pathways = await lincs_client.get_pathways_for_gene(gene_symbol)
        except Exception:
            logger.debug(
                "Reactome pathway lookup failed for %s",
                gene_symbol, exc_info=True,
            )
            pathways = []
        for pathway in pathways:
            try:
                participants = await lincs_client.get_pathway_gene_symbols(
                    pathway.stable_id
                )
            except Exception:
                logger.debug(
                    "Reactome pathway participant lookup failed for %s",
                    pathway.stable_id, exc_info=True,
                )
                participants = []
            for s in participants:
                if s:
                    universe.add(s.upper())
        pathway_gene_cache[gene_symbol] = universe
        return universe

    def _gene_biomarkers(features: list[SubgroupFeature]) -> list[str]:
        return [f.key.upper() for f in features if f.axis == "gene" and f.key]

    edges_added = 0
    chains_added = 0
    for path in sorted(annotations_dir.glob("*_extraction.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable extraction at %s", path)
            continue
        nct_id = data.get("nct_id")
        if not nct_id:
            continue
        try:
            ts = graph.get_trial_subgraph_by_id(nct_id)
        except KeyError:
            continue

        if not ts.chains:
            continue
        indication_id = ts.chains[0].indication_id
        if indication_id == _UNKNOWN:
            continue

        # Snapshot parent chains BEFORE forking so we don't fork copies.
        parent_chains = list(ts.chains)

        # Resolve each parent chain's "target pathway universe" once —
        # the set of genes that share a Reactome pathway with the
        # chain's target. We test subgroup biomarker genes against this
        # set to decide whether to fork.
        chain_universes: dict[int, set[str]] = {}
        for i, chain in enumerate(parent_chains):
            if chain.target_id == _UNKNOWN:
                chain_universes[i] = set()
                continue
            try:
                tnode = graph.get_node(chain.target_id)
            except KeyError:
                chain_universes[i] = set()
                continue
            tgene = (tnode.get("gene_symbol") or "").upper()
            chain_universes[i] = await _pathway_universe_for_gene(tgene)

        # Per-subgroup forks. Each entry: (pop_id, list of biomarker
        # genes); empty gene list means "universally relevant".
        forks: list[tuple[str, list[str]]] = []

        for sg in data.get("subgroups") or []:
            descriptor = sg.get("raw_descriptor", "")
            features: list[SubgroupFeature] = []
            for f in sg.get("features") or []:
                cf = canonicalize_feature(
                    f.get("axis", ""),
                    f.get("key", ""),
                    f.get("level", ""),
                    raw_descriptor=descriptor,
                )
                if not is_canonical(cf):
                    log_unmapped(cf, nct_id)
                features.append(cf)
            if not features:
                continue
            # Drop subgroups whose features all canonicalize to "other".
            if not any(is_canonical(f) for f in features):
                continue
            # Response-axis features (CR/PR/SD/PD) are outcome stratifiers,
            # not patient strata — they describe what happened to subgroups
            # of patients post-treatment, so they shouldn't fork chains.
            if all(
                (not is_canonical(f)) or f.axis == "response"
                for f in features
            ):
                continue
            pop_id = PopulationNode.compose_id(indication_id, features)
            try:
                graph.get_node(pop_id)
            except KeyError:
                graph.add_node(PopulationNode(
                    id=pop_id,
                    name=f"{descriptor or pop_id}",
                    defining_features=list(features),
                ))
            if not graph._graph.has_edge(  # noqa: SLF001
                pop_id, indication_id, key=EdgeType.RESPONDS_DIFFERENTLY.value,
            ):
                graph.add_edge(GraphEdge(
                    source_id=pop_id,
                    target_id=indication_id,
                    edge_type=EdgeType.RESPONDS_DIFFERENTLY,
                    belief=EdgeBeliefState(alpha=1.5, beta=1.0),
                    metadata={
                        "source": "extraction_subgroup",
                        "trial_id": nct_id,
                        "raw_descriptor": descriptor,
                    },
                ))
                edges_added += 1
            if pop_id != ts.parent_population_id:
                forks.append((pop_id, _gene_biomarkers(features)))

        if forks:
            existing_pops = {c.subgroup_population_id for c in parent_chains}
            forked: list[CausalChain] = list(parent_chains)
            # Dedupe forks by pop_id (preserve first occurrence + its
            # biomarker list).
            seen_pops: set[str] = set()
            ordered_forks: list[tuple[str, list[str]]] = []
            for pop_id, biomarkers in forks:
                if pop_id in seen_pops:
                    continue
                seen_pops.add(pop_id)
                ordered_forks.append((pop_id, biomarkers))

            for pop_id, biomarkers in ordered_forks:
                if pop_id in existing_pops:
                    continue
                for i, parent in enumerate(parent_chains):
                    if biomarkers:
                        universe = chain_universes.get(i, set())
                        # Relevance: at least one subgroup biomarker
                        # gene must share a Reactome pathway with the
                        # chain's target. If LINCS / Reactome couldn't
                        # resolve the universe (empty), fall back to
                        # forking — better noisy than missing.
                        if universe and not any(b in universe for b in biomarkers):
                            continue
                    forked.append(parent.model_copy(
                        update={"subgroup_population_id": pop_id}
                    ))
                    chains_added += 1
            graph.set_trial_subgraph(ts.model_copy(update={"chains": forked}))
    return edges_added, chains_added


def add_subgroup_chains(
    graph: GraphStore,
    trial_subgraph: TrialSubgraph,
    *,
    indication_id: str,
    endpoint_id: str,
    subgroup_features: list[list[SubgroupFeature]],
    indication_name: str | None = None,
) -> int:
    """Add (arm × subgroup) chains to an existing TrialSubgraph.

    ``subgroup_features`` is a list of feature compositions—one composition
    per reported subgroup. Each composition produces one PopulationNode
    (created if missing) and one chain *per arm*.

    The TrialSubgraph in the sidecar is replaced with the augmented version.
    Returns the number of new chains added.
    """
    if not subgroup_features:
        return 0

    # Subgroup-fork chains anchor on the parent disease (populations
    # still encode subtype via the compose_id call below).
    chain_indication_id = _root_indication(indication_id)
    new_chains: list[CausalChain] = []
    for features in subgroup_features:
        pop_id = PopulationNode.compose_id(indication_id, features)
        try:
            graph.get_node(pop_id)
        except KeyError:
            descriptor = ", ".join(f.raw_descriptor or f.slug() for f in features)
            graph.add_node(PopulationNode(
                id=pop_id,
                name=f"{descriptor} ({indication_name or indication_id})",
                defining_features=list(features),
            ))
        for arm in trial_subgraph.arms:
            new_chains.append(CausalChain(
                arm_id=arm.arm_id,
                compound_id=arm.regimen_compound_id,
                subgroup_population_id=pop_id,
                target_id=_UNKNOWN,
                mechanism_id=_UNKNOWN,
                biology_id=_UNKNOWN,
                indication_id=chain_indication_id,
                endpoint_id=endpoint_id,
                outcome=TrialOutcome.UNKNOWN,
            ))

    augmented = trial_subgraph.model_copy(
        update={"chains": list(trial_subgraph.chains) + new_chains}
    )
    graph.set_trial_subgraph(augmented)
    return len(new_chains)


def _normalize(name: str) -> str:
    """Lowercase, strip whitespace, collapse runs of spaces."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ── CLI entry point ──────────────────────────────────────────────────────


async def _main(area: str, max_trials: int, condition: str) -> None:
    from pathlib import Path

    graph = GraphStore()
    client = anthropic.AsyncAnthropic(timeout=60.0)
    pipeline = PopulationPipeline(graph, anthropic_client=client)

    if area == "oncology":
        await pipeline.populate_oncology(
            max_trials=max_trials, condition=condition,
        )
    else:
        console.print(f"[red]Unknown area: {area}[/red]")
        return

    export_path = Path("data/exports")
    export_path.mkdir(parents=True, exist_ok=True)
    snapshot = str(export_path / f"{area}_initial.json")
    graph.export_snapshot(snapshot)
    console.print(f"[bold]Snapshot saved to {snapshot}[/bold]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate Eroom Bio knowledge graph")
    parser.add_argument("--area", default="oncology", help="Disease area (default: oncology)")
    parser.add_argument("--max-trials", type=int, default=500, help="Max trials to fetch")
    parser.add_argument(
        "--condition",
        default="cancer",
        help="ClinicalTrials.gov condition filter (default: cancer)",
    )
    args = parser.parse_args()

    asyncio.run(_main(args.area, args.max_trials, args.condition))
