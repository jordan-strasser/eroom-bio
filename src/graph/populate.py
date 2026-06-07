"""Orchestrator that constructs the initial knowledge graph."""

from __future__ import annotations

import asyncio
import hashlib
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
    EvidenceType,
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
    TargetType,
    TrialArm,
    TrialNode,
    TrialOutcome,
    TrialSubgraph,
    normalize_entity,
)
from src.graph.antibody_target_resolver import (
    is_monoclonal_antibody,
    mab_target_gene,
)
from src.graph.codename_resolver import CodenameResolver, ResolvedCodename
from src.graph.compound_codenames import CODENAME_TO_INN
from src.graph.curated_evidence import (
    belief_from_curated_record,
    make_curated_record,
    ot_association_score_to_bucket,
    ot_score_quality,
)
from src.graph.indication_taxonomy import (
    parent_indication_for,
    slugify_measure_name,
)
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket
from src.ingestion.clinicaltrials import (
    ArmGroup,
    ClinicalTrialsClient,
    TrialRecord,
    canonical_compound_names,
    is_drug_like,
    map_trial_to_graph_nodes,
    split_combo_regimen as _split_combo_regimen,
    strip_parenthetical_brand as _strip_parenthetical_brand,
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


# Compound-name canonicalization (split_combo_regimen, strip_parenthetical_brand,
# canonical_compound_names) lives in src.ingestion.clinicaltrials — both populator
# code paths (build_arms here + map_trial_to_graph_nodes there) use the same
# canonicalization so a drug gets one CompoundNode regardless of how the trial
# reports it. Re-exported below under the old underscore-prefixed names for
# back-compat with existing test imports.


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
    _category_to_direction,
    _category_to_mechanism_type,
    populate_lincs_signatures,
)
from src.ingestion.chembl import ChEMBLClient, is_likely_diagnostic
from src.ingestion.pubchem import PubChemClient
from src.ingestion.rxnorm import RxNormClient
from src.ingestion.opentargets import (
    OpenTargetsClient,
    score_to_prior,
)

logger = logging.getLogger(__name__)
console = Console()


# ── Round-28: parent-population coarsening ─────────────────────────────
#
# Pre-round-28 the parent PopulationNode for a trial composed its id
# from EVERY axis the qualifier extractor (regex) and LLM-eligibility
# pass produced. That over-specified the parent so cross-trial sharing
# broke even between trials studying the same disease at the same
# treatment line — NSABP C-08 emitted `responds_differently` evidence
# to `colon_adenocarcinoma__histology_adenocarcinoma__line_adjuvant_treated__stage_iii`
# while bev AVANT walked `colorectal_cancer__line_adjuvant__stage_iii`.
# Two different nodes, no shared signal.
#
# Coarsening keeps the axes that are LOAD-BEARING for cross-trial
# differentiation (treatment line, stage, extent of disease, treatment
# setting) and demotes rare / disease-specific axes (histology, age,
# severity, gene biomarker, prior_tx, response, …) to subgroup-only.
# Subgroup PopulationNodes (built from extraction.subgroups) still get
# the full feature set so rare-axis subgroup analyses keep their own
# chains via the round-16 add_subgroup_chains path.
_DEFAULT_POPULATION_AXES: frozenset[str] = frozenset({
    "line",     # line of therapy — first / second / later / adjuvant
    "stage",    # cancer staging — i / ii / iii / iv
    "extent",   # disease spread — metastatic / unresectable / advanced / …
    "setting",  # treatment context — adjuvant / neoadjuvant / maintenance
})


def _coarse_population_features(
    features: list[SubgroupFeature],
) -> list[SubgroupFeature]:
    """Return the subset of ``features`` whose axis is in the round-28
    parent-population whitelist. Order is preserved; the caller
    typically passes the same list into PopulationNode.compose_id
    which sorts internally."""
    return [f for f in features if f.axis in _DEFAULT_POPULATION_AXES]

# Round-20.5: structured drop log for trials the populator silently
# skipped during build_trial_subgraphs. Wiped at the start of each
# populate_trials call so each build's drop set is fresh. Read by
# the orchestrator's drop-rate threshold check + by tests asserting
# no-silent-loss.
_DROPPED_TRIAL_SUBGRAPHS_LOG = Path("data/dev/dropped_trial_subgraphs.jsonl")


def _log_dropped_trial_subgraph(
    nct_id: str, reason: str, **details: Any,
) -> None:
    """Append a structured record describing a trial that build_trial_
    subgraphs couldn't build a chain for.

    ``reason`` is one of: ``no_indication`` (no condition canonicalized
    to an IndicationNode), ``no_endpoint`` (no primary_outcome resolved
    to an EndpointNode), ``no_arms`` (arm_groups empty or every arm
    filtered out, e.g. all diagnostic compounds). ``details`` captures
    the raw inputs that failed so the diagnostic doesn't require a
    re-fetch from CT.gov.
    """
    from datetime import datetime, timezone

    _DROPPED_TRIAL_SUBGRAPHS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "nct_id": nct_id,
        "reason": reason,
        "details": details,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with _DROPPED_TRIAL_SUBGRAPHS_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.warning(
        "build_trial_subgraphs dropped %s: %s (%s)",
        nct_id, reason, details,
    )

# Haiku is fast + cheap for short categorical labels. The structural inferences
# (endpoint type, mechanism, population) are short—Haiku is more than enough.
INFERENCE_MODEL = "claude-haiku-4-5-20251001"

# Drug→target inference is a KNOWLEDGE task, not a categorical label: a
# plausible-but-wrong gene (Haiku gave axitinib→PLK1, bactrim→DHPS) injects a
# false chain, and OT only validates the gene EXISTS, not that the drug hits it.
# So this one inference uses Sonnet for accuracy. Cost is trivial (~100 misses
# across the corpus, ~$0.15) — well under the target-inference budget.
TARGET_INFERENCE_MODEL = "claude-sonnet-4-20250514"


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
# Round-25: endpoint-class → curated-evidence bucket. Replaces the
# hand-set Beta(α, β) priors with a categorical bucket emitted as a
# DATABASE_CURATED EvidenceRecord. The bucket choices are calibrated so
# the resulting posterior E[p] roughly matches the old hand-set means:
#   STRONG_SUPPORT  (p_obs=0.95, n_eff=3) → Beta(3.85, 1.15), E[p]≈0.77
#   MODERATE_SUPPORT (p_obs=0.80, n_eff=3) → Beta(3.4, 1.6), E[p]≈0.68
#   WEAK_SUPPORT     (p_obs=0.65, n_eff=3) → Beta(2.95, 2.05), E[p]≈0.59
#   AMBIGUOUS        (p_obs=0.50, n_eff=3) → Beta(2.5, 2.5),  E[p]=0.50
#   WEAK_CONTRADICT  (p_obs=0.35, n_eff=3) → Beta(2.05, 2.95), E[p]≈0.41
ENDPOINT_CLASS_BUCKETS: dict[str, str] = {
    "OS": "strong_support",
    "DFS": "moderate_support",
    "RFS": "moderate_support",  # accepted in adjuvant melanoma
    "composite_survival": "moderate_support",
    "PFS": "moderate_support",
    "TTP": "moderate_support",
    "CR": "moderate_support",
    "DMFS": "weak_support",  # used adjuvantly, less precedent than DFS
    "DOR": "weak_support",   # response durability conditional on response
    "ORR": "weak_support",
    "composite_response": "weak_support",
    "biomarker": "weak_support",
    "PRO": "ambiguous",
    # Safety endpoints don't directly capture efficacy — pull below 0.5.
    "safety": "weak_contradict",
    # "other" → no record emitted (edge stays Beta(1,1) and is dropped
    # at predict time, which is the right behavior for an unclassified
    # endpoint).
    "other": "",
}
ENDPOINT_CLASSES = list(ENDPOINT_CLASS_BUCKETS.keys())

# Catch-all endpoint classes that DON'T name a standardized measure: every
# distinct biomarker (LDL-C, amyloid-PET SUVr, ...), PRO instrument, safety
# rate, and "other" outcome would otherwise fuse into ONE `{class}_{indication}`
# node and the chain would lose which endpoint it measured. These are sub-keyed
# by the specific measure (`{class}_{measure}_{indication}`); the standardized
# clinical classes (OS/PFS/ORR/DFS/...) stay collapsed by class so a single
# `PFS_melanoma` still serves every melanoma subtype. The endpoint_captures
# Beta prior is still keyed off the CLASS, so the belief is unchanged — only
# node identity gets finer (round-31 endpoint de-collapse).
_GENERIC_ENDPOINT_CLASSES = {
    EndpointClass.BIOMARKER,
    EndpointClass.PRO,
    EndpointClass.SAFETY,
    EndpointClass.OTHER,
}


def _endpoint_node_id(cls: EndpointClass, measure: str) -> tuple[str, str]:
    """``(node_id, display_name)`` for a (class, measure). Endpoints are
    DISEASE-AGNOSTIC (node-orthogonality redesign): standardized classes
    collapse to a single shared node ``{class}`` (the per-disease binding lives
    on the endpoint_captures edge); catch-all classes (biomarker/safety/PRO/
    other) are sub-keyed by a normalized measure slug so distinct measures don't
    fuse. No indication in the id."""
    if cls in _GENERIC_ENDPOINT_CLASSES:
        measure_slug = slugify_measure_name(measure, max_len=48)
        node_id = normalize_entity(f"{cls.value}_{measure_slug}", "EndpointNode")
        return node_id, f"{measure} [{cls.value}]"
    node_id = normalize_entity(cls.value, "EndpointNode")
    return node_id, cls.value


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
    """Add an endpoint_captures edge with the prior for the given class.

    Round-25: emits a DATABASE_CURATED EvidenceRecord whose bucket
    encodes the endpoint class. Classes whose bucket is "" (e.g. "other")
    don't get an edge — the prediction engine treats them as unobserved
    rather than mildly optimistic, which is the right semantics for an
    unclassified endpoint.
    """
    bucket_str = ENDPOINT_CLASS_BUCKETS.get(classification)
    if bucket_str is None:
        return False
    if not bucket_str:
        return False
    record = make_curated_record(
        source_id=f"endpoint_type_prior:{endpoint_id}",
        bucket=SupportBucket(bucket_str),
        source_type=EvidenceType.DATABASE_ENDPOINT_PRIOR,
        notes=(
            f"endpoint class={classification!r} from "
            f"LLM endpoint classifier"
        ),
    )
    graph.add_edge(GraphEdge(
        source_id=endpoint_id,
        target_id=indication_id,
        edge_type=EdgeType.ENDPOINT_CAPTURES,
        belief=belief_from_curated_record(record),
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


def _biology_id_from_description(description: str) -> str:
    """Content-address handle for a description-defined BiologyNode.

    Semantic-layers redesign: a Biology node's IDENTITY is its extracted
    physiological-process description + that description's BioLORD embedding.
    This id is *not* a slug — it fabricates no meaning from neighboring layers
    (the old ``{mechanism}__{indication}`` slug glued the two layers around
    biology into a fake key). It's a stable content-address (like a git blob
    hash) over the normalized description text: identical descriptions across
    trials collapse to the same id (exact Tier-1 merge), and near-identical
    phrasings pool via the Tier-3 BioLORD geometric merge. The human-readable
    identity lives in the node's ``name`` + ``description``; the real curated
    cross-ref (Reactome/GO), when it resolves, is used as the handle instead.
    """
    norm = " ".join(description.strip().lower().split())
    return "bio:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _mechanism_id_from_description(description: str) -> str:
    """Content-address handle for a description-defined MechanismNode. SUPERSEDED.

    Semantic-layers redesign: a Mechanism node's IDENTITY is now the localized
    molecular SIGNALING PATHWAY the target gene drives (a Reactome pathway id —
    see ``_populate_trial_mechanisms`` + ``_resolve_gene_pathways``), NOT a
    content-address of the extracted action description. This helper (and the
    ``mech:`` content-address path it backed) is retained only so the unit test
    and any external callers don't break; the live populate flow no longer uses
    it. The extracted action description still rides along as the node's
    free-text substrate + the pathway re-rank context.

    Kept behavior: a stable content-address (git-blob-style hash) over the
    normalized description text. Mirrors :func:`_biology_id_from_description`.
    """
    norm = " ".join(description.strip().lower().split())
    return "mech:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


# Curated mapping for cancer-vaccine constituents that Open Targets cannot
# resolve to a target. Covers three classes:
#   1. Tumor-associated-antigen (TAA) peptides → tumor antigen gene
#      (gp100→PMEL, tyrosinase→TYR, MART-1→MLANA).
#   2. Adjuvants → DC pattern-recognition / inflammasome receptor.
#      Water-in-oil depot adjuvants (Montanide, IFA) route to NLRP3 — their
#      action is DAMP-driven sterile inflammation, NOT clean TLR ligation.
#      Genuine TLR ligands (poly-ICLC→TLR3, CpG→TLR9, MPL→TLR4) wire to
#      their actual receptor.
#   3. CD4 helper peptides (PADRE, tetanus epitope, melanoma helper
#      peptide) → MHC class II (HLA-DRB1, the pan-DR binder).
# Each pattern is a lowercase substring match. Beta(3, 1) prior — confident
# curated mapping, below OT-sourced (Beta(4, 1)) and above name-match
# heuristic (Beta(2, 1.5)).
_VACCINE_COMPONENT_TARGETS: list[tuple[str, list[tuple[str, str, str]]]] = [
    # ── TAA peptides → tumor antigen gene ──
    ("gp100", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein"),
    ]),
    ("imcgp100", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein"),
    ]),
    ("tyrosinase", [
        ("ENSG00000077498", "TYR", "Tyrosinase"),
    ]),
    ("mart-1", [
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
    ("mart1", [
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
    ("melan-a", [
        ("ENSG00000120215", "MLANA", "Melan-A"),
    ]),
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
    # ── Depot adjuvants → NLRP3 (DAMP-driven inflammasome, not a TLR) ──
    ("montanide", [
        ("ENSG00000162711", "NLRP3", "NLR family pyrin domain containing 3"),
    ]),
    ("incomplete freund", [
        ("ENSG00000162711", "NLRP3", "NLR family pyrin domain containing 3"),
    ]),
    # ── TLR-agonist adjuvants → their genuine PRR ──
    ("poly-iclc", [
        ("ENSG00000164342", "TLR3", "Toll-like receptor 3"),
    ]),
    ("poly iclc", [
        ("ENSG00000164342", "TLR3", "Toll-like receptor 3"),
    ]),
    ("hiltonol", [
        ("ENSG00000164342", "TLR3", "Toll-like receptor 3"),
    ]),
    ("cpg odn", [
        ("ENSG00000239732", "TLR9", "Toll-like receptor 9"),
    ]),
    ("cpg-7909", [
        ("ENSG00000239732", "TLR9", "Toll-like receptor 9"),
    ]),
    # CMP-001 (vidutolimod) — virus-like particle containing CpG-A;
    # mechanistically a TLR9 agonist. Codename-only — doesn't share
    # tokens with the cpg patterns above.
    ("cmp-001", [
        ("ENSG00000239732", "TLR9", "Toll-like receptor 9"),
    ]),
    ("cmp001", [
        ("ENSG00000239732", "TLR9", "Toll-like receptor 9"),
    ]),
    ("vidutolimod", [
        ("ENSG00000239732", "TLR9", "Toll-like receptor 9"),
    ]),
    ("monophosphoryl lipid", [
        ("ENSG00000136869", "TLR4", "Toll-like receptor 4"),
    ]),
    # ── CD4 helper peptides → MHC class II ──
    ("tetanus", [
        ("ENSG00000196126", "HLA-DRB1", "Major histocompatibility complex class II, DR beta 1"),
    ]),
    ("padre", [
        ("ENSG00000196126", "HLA-DRB1", "Major histocompatibility complex class II, DR beta 1"),
    ]),
    ("melanoma helper peptide", [
        ("ENSG00000196126", "HLA-DRB1", "Major histocompatibility complex class II, DR beta 1"),
    ]),
]


# Curated mapping for adoptive-cell-therapy products (TCR-engineered T
# cells, TIL products, oncolytic-virus-delivered antigen vaccines) and
# specific code-named small molecules whose target Open Targets misses.
# These have no ChEMBL action_type that would trigger the
# `chembl_rest_fallback` path (cell therapies aren't drugs in
# ChEMBL's sense, and code names sometimes don't resolve to the
# parent compound's ChEMBL entry). Same pattern as the vaccine
# heuristic: lowercase substring match in intervention name →
# canonical Target(s). Beta(3, 1) prior — confident curated mapping.
def _norm_for_pattern_match(s: str) -> str:
    """Normalize a string for substring-pattern matching against a
    curated table: lowercase, collapse all hyphens / underscores /
    dots / whitespace runs to a single space. So "Anti-NY ESO-1",
    "anti_ny_eso_1", "Anti NY-ESO-1" all become "anti ny eso 1",
    and "hu14.18-IL2" becomes "hu14 18 il2" — one pattern entry per
    drug covers every separator variant.
    """
    return re.sub(r"[\s\-_.]+", " ", s.lower()).strip()


# Markers that an intervention name is NOT a single canonical molecule, so it
# must not inherit a constituent drug's ChEMBL ``stable_id`` (which would then
# false-merge it onto that drug under the node_merge chembl tier). Withholding the
# id only ever causes safe UNDER-merge (the name keeps its own node), never a
# wrong-merge. Generic class / non-drug tokens:
_NONCANONICAL_COMPOUND_TERMS = frozenset({
    "mesenchymal", "stem", "checkpoint", "placebo", "vehicle", "saline",
    "contraceptive", "observation", "antigens", "sham",
})
# Combination / regimen / dose-schedule descriptors (substring match on the
# normalized, space-separated name):
_NONCANONICAL_COMPOUND_MARKERS = (
    "combination", "schedule", "doublet", "triplet", "quadruplet", "regimen",
    "dose escalation", "dose finding", "dose ramp", "dose expansion", "cohort",
    "standard dosing", "supportive care", "standard of care",
)
# Standalone dosing-unit tokens (a single molecule's name doesn't carry these):
_NONCANONICAL_DOSING_UNITS = frozenset({"mg", "ml", "mcg", "kg", "iu", "auc", "min"})


def _is_noncanonical_compound_name(
    name: str, known_chembl: dict[str, str] | None = None,
) -> bool:
    """True when an intervention name isn't a single canonical molecule — used to
    withhold a ChEMBL ``stable_id`` from it so the merge chembl tier can't collapse
    it onto a real drug.

    Catches generic drug-class / non-drug terms, combination/regimen/dose-schedule
    descriptors, and dosing-unit strings. With ``known_chembl`` (norm-name → ChEMBL
    id) it also catches true multi-drug combos by the precise signal that
    distinguishes them from a brand+INN pair: a brand+INN ("bevacizumab avastin")
    has all drug-tokens mapping to ONE ChEMBL id (→ keep, it's one drug), whereas a
    combo ("capecitabine oxaliplatin cetuximab") spans ≥2 distinct ids (→ withhold).
    """
    norm = _norm_for_pattern_match(name)
    tokset = set(norm.split())
    if tokset & _NONCANONICAL_COMPOUND_TERMS:
        return True
    if any(m in norm for m in _NONCANONICAL_COMPOUND_MARKERS):
        return True
    if tokset & _NONCANONICAL_DOSING_UNITS:
        return True
    if known_chembl:
        ids = {known_chembl[t] for t in tokset if t in known_chembl}
        if len(ids) >= 2:  # ≥2 distinct drugs by id ⇒ true combination, not brand+INN
            return True
    return False


_CELL_THERAPY_COMPONENT_TARGETS: list[tuple[str, list[tuple[str, str, str]]]] = [
    # Patterns are matched via `_norm_for_pattern_match`: hyphens,
    # underscores, and runs of whitespace all collapse to single space,
    # lowercased. So one canonical pattern covers "anti-ny-eso-1",
    # "anti ny eso 1", "Anti-NY ESO-1", etc. — no need to enumerate
    # every separator variant.

    # ── TCR-engineered T cells: antigen = TCR's recognition target ──
    ("anti ny eso 1", [
        ("ENSG00000184033", "CTAG1B", "Cancer/testis antigen 1B (NY-ESO-1)"),
    ]),
    ("alvac ny eso 1", [
        ("ENSG00000184033", "CTAG1B", "Cancer/testis antigen 1B (NY-ESO-1)"),
    ]),
    # gp100 TCR (the autologous_anti_gp_100_154_162_… compounds).
    ("anti gp 100", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein (gp100)"),
    ]),
    ("anti gp100", [
        ("ENSG00000185664", "PMEL", "Premelanosome protein (gp100)"),
    ]),
    # MART-1 TCR
    ("anti mart 1", [
        ("ENSG00000120215", "MLANA", "Melan-A (MART-1)"),
    ]),
    # ── Code-named small molecules ──
    # PS-341 / Velcade → bortezomib → proteasome β5.
    ("ps 341", [
        ("ENSG00000100804", "PSMB5", "Proteasome subunit beta type-5"),
    ]),
    ("velcade", [
        ("ENSG00000100804", "PSMB5", "Proteasome subunit beta type-5"),
    ]),
    # UCN-01 (7-hydroxystaurosporine) — Chk1/PKC inhibitor.
    ("ucn 01", [
        ("ENSG00000149554", "CHEK1", "Checkpoint kinase 1"),
    ]),
    # ── Immunocytokines ──
    # hu14.18-IL2 — anti-GD2 ganglioside antibody fused to IL-2. Two
    # targets: GD2 synthase (B4GALNT1, the gene that makes the GD2
    # glycolipid the antibody binds — GD2 has no gene id of its own)
    # and IL-2 receptor (IL2RA, for the cytokine arm).
    ("hu14 18", [
        ("ENSG00000135454", "B4GALNT1", "Beta-1,4-N-acetylgalactosaminyltransferase 1 (GD2 synthase)"),
        ("ENSG00000134460", "IL2RA", "Interleukin-2 receptor alpha"),
    ]),
    # ── Oncolytic viruses ──
    # T-VEC (talimogene laherparepvec) — attenuated HSV-1 carrying
    # GM-CSF. HSV-1 entry into tumor cells uses HVEM (TNFRSF14) plus
    # nectins as receptors; HVEM is the canonical "target" the
    # mechanistic chain hangs off of. Round-13 cell-therapy add.
    ("talimogene laherparepvec", [
        ("ENSG00000157873", "TNFRSF14", "Tumor necrosis factor receptor superfamily 14 (HVEM, HSV entry receptor)"),
    ]),
    ("talimogene", [
        ("ENSG00000157873", "TNFRSF14", "Tumor necrosis factor receptor superfamily 14 (HVEM, HSV entry receptor)"),
    ]),
    ("t vec", [
        ("ENSG00000157873", "TNFRSF14", "Tumor necrosis factor receptor superfamily 14 (HVEM, HSV entry receptor)"),
    ]),
    # ── Metabolic-deprivation therapies ──
    # ADI-PEG-20 (pegylated arginine deiminase) — depletes systemic
    # L-arginine. Therapeutic window is ASS1-deficient tumors that
    # can't resynthesize arginine; ASS1 status is the canonical
    # biomarker for response. Target = ASS1 as the gene whose absence
    # creates the synthetic-lethal interaction.
    ("adi peg 20", [
        ("ENSG00000130707", "ASS1", "Argininosuccinate synthase 1"),
    ]),
    ("adi peg20", [
        ("ENSG00000130707", "ASS1", "Argininosuccinate synthase 1"),
    ]),
    ("pegylated arginine deiminase", [
        ("ENSG00000130707", "ASS1", "Argininosuccinate synthase 1"),
    ]),
    # ── MAGE-A3 cancer vaccines ──
    # GSK2132231a / recMAGE-A3 — anti-MAGE-A3 cancer immunotherapeutic
    # (failed DERMA trial, 2014). Target = MAGEA3 antigen gene.
    ("gsk2132231", [
        ("ENSG00000221867", "MAGEA3", "Melanoma-associated antigen 3"),
    ]),
    ("mage a3", [
        ("ENSG00000221867", "MAGEA3", "Melanoma-associated antigen 3"),
    ]),
    ("recmage a3", [
        ("ENSG00000221867", "MAGEA3", "Melanoma-associated antigen 3"),
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
        self._chembl_client = ChEMBLClient(
            cache_path=cache_dir / "chembl_mechanisms.json",
        )
        self._anthropic = anthropic_client
        self._cache_dir = cache_dir
        # name → node_id index, keyed by (normalized_name, node_type)
        self._entity_index: dict[tuple[str, str], str] = {}
        # raw CT.gov condition string → (canonical_slug, display_name).
        # Lazily initialized when canonicalization is first needed.
        self._indication_canon: JSONCache | None = None
        # Round-21 Phase B: normalized intervention name → chembl_id.
        # Populated as a side-effect of _identify_diagnostic_compounds
        # (which already does the OT/ChEMBL lookups). Step 2's
        # compound-add loop consults this for stable_id-based
        # canonicalization so two CT.gov spellings of the same drug
        # ("Fluorouracil" + "5-Fluorouracil") consolidate onto one
        # InterventionNode keyed by ChEMBL id.
        self._compound_chembl_ids: dict[str, str] = {}
        # Round-23: RxNorm + PubChem fallback for codenames the
        # CODENAME_TO_INN curated dict + OT chembl_id lookup didn't
        # resolve. Used in Step 2's compound-add loop to rename the
        # codename to its INN before canonicalization, so MK-3475 and
        # Pembrolizumab consolidate into one InterventionNode without
        # requiring a code change every time a new codename surfaces.
        self._codename_resolver = CodenameResolver(
            rxnorm=RxNormClient(
                cache_path=cache_dir / "rxnorm_lookups.json",
            ),
            pubchem=PubChemClient(
                cache_path=cache_dir / "pubchem_synonyms.json",
            ),
        )
        self._codename_resolutions: dict[str, ResolvedCodename] = {}

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

    def _annotate_compound_node_from_ot(
        self, compound_id: str, drug_data: dict,
    ) -> None:
        """Persist OT-derived chembl_id + aliases onto an existing CompoundNode.

        Round-15 fix for canonicalization: the OT drug-lookup result was
        previously consumed only for diagnostic filtering and OT target
        resolution, never written back to the compound node. Without this,
        every dev-code form of a drug (MK-3475, GSK2118436, BMS-936558)
        creates a fresh compound node instead of aliasing onto the
        canonical name — splitting evidence across duplicate nodes.

        Idempotent. Mutation goes through ``self.graph._graph.nodes[id]``
        because ``GraphStore.get_node`` returns a dict copy.
        """
        try:
            self.graph.get_node(compound_id)
        except KeyError:
            return
        node = self.graph._graph.nodes[compound_id]  # noqa: SLF001
        chembl_id = drug_data.get("chembl_id")
        if chembl_id and not node.get("chembl_id"):
            node["chembl_id"] = chembl_id
        new_aliases = [a for a in (drug_data.get("aliases") or []) if a]
        if new_aliases:
            existing = list(node.get("aliases") or [])
            seen = set(existing)
            merged = list(existing)
            for alias in new_aliases:
                if alias not in seen:
                    seen.add(alias)
                    merged.append(alias)
            if merged != existing:
                node["aliases"] = merged
            for alias in new_aliases:
                self._index_node(compound_id, alias, "compound")

    # ── Main pipeline ────────────────────────────────────────────────────

    async def populate_trials(
        self,
        max_trials: int = 500,
        include_terminated_no_results: bool = True,
        condition: str = "cancer",
        trials: list[TrialRecord] | None = None,
        annotations_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        # annotations_dir: when set (the build now extracts BEFORE populate),
        # the biology step reads each trial's extracted biology description and
        # gives the Biology node its identity from that description rather than
        # a per-trial ontology lookup. None preserves the legacy resolve path.
        # Round-20.5: clear the drop log at the start of every build so
        # it reflects ONLY this run's silent skips. Any entries left
        # over from a prior build would otherwise confuse the
        # orchestrator's drop-rate check.
        if _DROPPED_TRIAL_SUBGRAPHS_LOG.exists():
            _DROPPED_TRIAL_SUBGRAPHS_LOG.unlink()

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

        # Step 1.5: Identify diagnostic interventions via ChEMBL metadata
        # so they're filtered out before any CompoundNode is created.
        # See `is_likely_diagnostic` for the decision rule (combines
        # `indication_class`, `therapeutic_flag`, and `max_phase`).
        # Stored on `self` so the downstream `_build_trial_subgraphs`
        # can also filter arm.compound_ids without re-querying ChEMBL.
        self._diagnostic_compound_ids = (
            await self._identify_diagnostic_compounds(trials)
        )
        diagnostic_compound_ids = self._diagnostic_compound_ids
        if diagnostic_compound_ids:
            console.print(
                f"[yellow]Filtering {len(diagnostic_compound_ids)} "
                f"diagnostic interventions: "
                f"{sorted(diagnostic_compound_ids)}[/yellow]"
            )

        # Step 1.6: Codename resolution (round-23). For each trial-named
        # compound the CODENAME_TO_INN dict + OT/ChEMBL Step 1.5 lookup
        # both missed, ask RxNorm + PubChem for an INN-equivalent name.
        # The resolution map is consulted in Step 2's compound-add loop
        # to rename the compound (with the original name preserved as an
        # alias) so the SapBERT canonicalization step then naturally
        # consolidates the codename and INN forms into one node.
        await self._resolve_compound_codenames(trials)

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
        from src.graph.population_features import (
            extract_population_features_with_llm,
        )

        console.print("[bold]Extracting graph nodes from trials...[/bold]")
        seen_indications: dict[str, str] = {}  # canonical_id → display name

        # Round-17: cache LLM-extracted population features (line of
        # therapy, prior treatments, biomarker requirements, mutation
        # status) keyed by nct_id. Pre-fetch features once per trial so
        # they're available alongside the deterministic qualifier regex
        # output when composing population_ids below.
        pop_feature_cache_path = self._cache_dir / "population_features.json"
        trial_llm_features: dict[str, list] = {}
        for trial in trials:
            trial_llm_features[trial.nct_id] = (
                await extract_population_features_with_llm(
                    nct_id=trial.nct_id,
                    title=trial.title,
                    conditions=list(trial.conditions),
                    eligibility_criteria=trial.eligibility_criteria,
                    client=self._anthropic,
                    cache_path=pop_feature_cache_path,
                )
            )

        # Per-canonical-indication metadata accumulator. Keeps track of
        # observed raw phrasings and the qualifier levels each axis saw —
        # useful for surfacing "this canonical disease appears as stage
        # III + IV across our corpus" downstream.
        ind_metadata: dict[str, dict[str, Any]] = {}

        # Trial id → (canonical_indication_id, default_population_id)
        # for build_trial_subgraphs to pick up.
        self._trial_default_population: dict[str, str] = {}
        self._trial_canonical_indication: dict[str, str] = {}

        # Round-21 Phase B: indices for compound canonicalization. Built
        # from the LOADED graph's existing compound nodes (the
        # incremental --base-snapshot path leaves them in place; the
        # fresh-build path starts empty). Indices are mutated in the
        # compound-add loop as new compounds get added so two trials in
        # the same run sharing a drug consolidate on the first
        # creation.
        compound_chembl_index: dict[str, str] = {}
        compound_embedding_index: list[tuple[str, list[float]]] = []
        for existing_node in self.graph.get_nodes_by_type("InterventionNode"):
            existing_id = existing_node["id"]
            ex_chembl = existing_node.get("stable_id")
            if ex_chembl:
                compound_chembl_index.setdefault(ex_chembl, existing_id)
            ex_emb = existing_node.get("embedding")
            if not ex_emb:
                # Public snapshots no longer persist embeddings — they're a
                # recomputable cache artifact, not graph state, and the
                # boundary (src/boundary.py) strips them so fine-tuned vectors
                # can't leak into committed data/exports/. Rehydrate the
                # canonicalization vector from the SapBERT cache by name so
                # incremental --base-snapshot builds keep full cosine
                # canonicalization rather than silently degrading to
                # chembl + alias matching.
                ex_emb = self._rehydrate_compound_embedding(existing_node)
            if ex_emb:
                compound_embedding_index.append((existing_id, ex_emb))

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

                # Round-17: merge LLM-extracted population features
                # (line_of_therapy, prior treatments, biomarker
                # requirements, mutation status from title + eligibility
                # criteria) with the deterministic qualifier regex
                # output. Combined feature set composes a more-specific
                # PopulationNode slug. Deduped at compose_id time via
                # slug-sort.
                llm_features = trial_llm_features.get(trial.nct_id, [])
                _seen_slugs = {f.slug() for f in qualifiers}
                combined_features = list(qualifiers)
                for f in llm_features:
                    if f.slug() not in _seen_slugs:
                        combined_features.append(f)
                        _seen_slugs.add(f.slug())

                # Round-28: coarsen the parent population's defining
                # features to the load-bearing axes only. Rare-axis
                # qualifiers (histology, age, severity, biomarker, …)
                # are dropped from the parent slug so the parent groups
                # trials studying the same disease at the same treatment
                # line / stage. Rare-axis subgroups still get their own
                # subgroup PopulationNodes downstream via the round-16
                # add_subgroup_chains path.
                parent_features = _coarse_population_features(combined_features)

                # The trial's enrollment-cohort population = its disease-
                # agnostic eligibility axes (line/stage/biomarker). With NO
                # coarse axes it's an all-comers cohort → NO population node
                # (it would only reiterate the indication); default_pop_id is
                # None and the chain gets no population dimension.
                default_pop_id = PopulationNode.compose_id(parent_features)
                if default_pop_id is not None:
                    try:
                        self.graph.get_node(default_pop_id)
                    except KeyError:
                        self.graph.add_node(PopulationNode(
                            id=default_pop_id,
                            name=PopulationNode.display_name(parent_features),
                            defining_features=list(parent_features),
                        ))
                    # responds_differently: enrollment-cohort population →
                    # indication. The trial's eligibility axes are a real
                    # stratification of the disease; the prediction walks this
                    # edge to score population fit. Only created when a real
                    # axis exists (default_pop_id is not None) — all-comers
                    # cohorts get no node and no edge. Starts at Beta(1, 1);
                    # classifier subgroup-vs-overall evidence lands here.
                    if not self.graph._graph.has_edge(  # noqa: SLF001
                        default_pop_id, canonical_id,
                        key=EdgeType.RESPONDS_DIFFERENTLY.value,
                    ):
                        if llm_features and qualifiers:
                            source_tag = "regex_plus_llm_eligibility"
                        elif llm_features:
                            source_tag = "llm_eligibility"
                        else:
                            source_tag = "indication_qualifiers"
                        self.graph.add_edge(GraphEdge(
                            source_id=default_pop_id,
                            target_id=canonical_id,
                            edge_type=EdgeType.RESPONDS_DIFFERENTLY,
                            belief=EdgeBeliefState(),
                            metadata={
                                "source": source_tag,
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

            # Compound nodes come from the trial's interventions.
            # Round-18 codename → INN canonicalization: if the trial's
            # intervention name normalizes to a known dev-code slug
            # (AZD6244, MK-3475, GSK1120212, etc.), remap to the
            # canonical INN at creation time so all forms (code, INN,
            # brand) resolve to the same compound node downstream.
            # The original code form is preserved as an alias.
            #
            # The diagnostic-compound set (computed in step 2.5 below)
            # filters out Lymphoseek-style imaging agents before they
            # enter the graph.
            nodes = map_trial_to_graph_nodes(trial)
            for comp in nodes["compounds"]:
                if comp.id in self._diagnostic_compound_ids:
                    continue
                inn_id = CODENAME_TO_INN.get(comp.id)
                resolved_codename: ResolvedCodename | None = None
                if inn_id is None:
                    # Round-23: RxNorm + PubChem fallback. The Step 1.6
                    # pass populated ``_codename_resolutions`` for every
                    # codename the curated dict + OT/ChEMBL couldn't
                    # resolve, so this is a pure dict lookup.
                    resolved_codename = (
                        self._codename_resolutions.get(comp.id)
                    )
                    if resolved_codename is not None:
                        inn_id = normalize_entity(
                            resolved_codename.canonical_name,
                            "InterventionNode",
                        )
                if inn_id and inn_id != comp.id:
                    original_id = comp.id
                    original_name = comp.name
                    comp.id = inn_id
                    comp.name = inn_id
                    new_aliases = list(comp.aliases)
                    for alt in (original_name, original_id):
                        if alt and alt not in new_aliases:
                            new_aliases.insert(0, alt)
                    comp.aliases = new_aliases
                if resolved_codename is not None:
                    if resolved_codename.rxcui:
                        comp.metadata["rxnorm_rxcui"] = (
                            resolved_codename.rxcui
                        )
                    if resolved_codename.pubchem_cid:
                        comp.metadata["pubchem_cid"] = (
                            resolved_codename.pubchem_cid
                        )
                    comp.metadata["codename_resolution_source"] = (
                        resolved_codename.source
                    )

                # Round-21 Phase B: stable_id + embedding-based
                # canonicalization. Populate stable_id from the cache
                # built by step 1.5's OT/ChEMBL lookups, then compute
                # the SapBERT embedding (if the sapbert extra is
                # installed). Both feed _canonicalize_compound which
                # decides whether to alias onto an existing node or
                # create a new one.
                norm_name = _norm_for_pattern_match(comp.name)
                if comp.stable_id is None:
                    cand = self._compound_chembl_ids.get(norm_name)
                    # Guard the node_merge chembl tier: only a plausibly-single
                    # molecule may carry a stable_id. Vague class terms
                    # (mesenchymal_stem_cell), placebos, and combo/regimen/dosing
                    # strings otherwise inherit a constituent's ChEMBL id and
                    # false-merge onto that drug. Withholding = safe under-merge.
                    if cand and _is_noncanonical_compound_name(
                        comp.name, self._compound_chembl_ids
                    ):
                        logger.debug(
                            "chembl-id %s withheld from non-canonical name %r",
                            cand, comp.name,
                        )
                    else:
                        comp.stable_id = cand
                if comp.embedding is None:
                    try:
                        from src.graph.sapbert_embeddings import (
                            SapBertUnavailable,
                            embed_compound_name,
                        )
                        try:
                            comp.embedding = embed_compound_name(
                                comp.name,
                                cache_path=self._cache_dir / "sapbert_embeddings.json",
                            )
                        except SapBertUnavailable:
                            # Optional extra not installed — fall back
                            # to chembl_id-only canonicalization.
                            pass
                    except ImportError:
                        pass

                canonical_id, was_resolved = self._canonicalize_compound(
                    comp,
                    existing_chembl_index=compound_chembl_index,
                    embedding_index=compound_embedding_index,
                )
                if was_resolved and canonical_id != comp.id:
                    # Merge the incoming compound's identity into the
                    # existing node: aliases get unioned, the entity
                    # index gets an entry from comp.name/id → existing id.
                    self._merge_compound_into_existing(comp, canonical_id)
                    continue

                self.graph.add_node(comp)
                # Update the in-loop indices so a later compound in the
                # same trial / corpus pass can dedup against this one.
                if comp.stable_id and comp.stable_id not in compound_chembl_index:
                    compound_chembl_index[comp.stable_id] = comp.id
                if comp.embedding:
                    compound_embedding_index.append((comp.id, comp.embedding))
                self._index_node(comp.id, comp.name, "compound")
                # Index original intervention names (aliases) too so raw
                # CT.gov-side names like "Sorafenib (Nexavar; BAY43-9006)"
                # still resolve to the canonical compound id after the
                # parenthetical-strip canonicalization.
                for alias in comp.aliases:
                    self._index_node(comp.id, alias, "compound")

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

        # Curated vaccine-component target heuristic: TAA peptides,
        # adjuvants (depot + TLR-agonist), and CD4 helper peptides that
        # OT can't resolve. Runs before name-match fallback so the
        # synthesized TargetNodes are available to the fallback for
        # trial-text cross-references. Merge the heuristic's
        # compound→target mapping into the main compound_targets dict so
        # per-constituent chain construction (_populate_trial_mechanisms)
        # sees these compounds as targeted.
        console.print("[bold]Wiring vaccine-component target edges...[/bold]")
        vax_added, vax_targets = self._add_vaccine_component_target_edges(trials)
        for cid, tids in vax_targets.items():
            existing = compound_targets.setdefault(cid, [])
            for tid in tids:
                if tid not in existing:
                    existing.append(tid)
        console.print(f"  Added {vax_added} AFFECTS edges (vaccine-component heuristic)")

        # Curated cell-therapy + code-named-drug heuristic. TCR-engineered
        # T cells, TIL products, oncolytic-virus-delivered antigens, and
        # code-named small molecules (PS-341/Velcade, UCN-01) whose target
        # OT can't resolve. Same pattern + Beta(3, 1) prior as the
        # vaccine heuristic; runs alongside it.
        console.print("[bold]Wiring cell-therapy target edges...[/bold]")
        cell_added, cell_targets = self._add_cell_therapy_target_edges(trials)
        for cid, tids in cell_targets.items():
            existing = compound_targets.setdefault(cid, [])
            for tid in tids:
                if tid not in existing:
                    existing.append(tid)
        console.print(f"  Added {cell_added} AFFECTS edges (cell-therapy heuristic)")

        # LLM target-inference: resolve compounds the curated + heuristic
        # resolvers all missed, so their chains ground on a real gene + Reactome
        # mechanism instead of a coarse drug-class slug. Validated against OT
        # before use; runs before the name-match fallback.
        console.print("[bold]Inferring missing compound targets (LLM)...[/bold]")
        llm_added, llm_targets = await self._infer_missing_compound_targets(
            trials, compound_targets,
        )
        for cid, tids in llm_targets.items():
            existing = compound_targets.setdefault(cid, [])
            for tid in tids:
                if tid not in existing:
                    existing.append(tid)
        console.print(f"  Added {llm_added} AFFECTS edges (LLM target-inference)")

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
        mech_added = await self._populate_trial_mechanisms(
            trials, compound_targets, annotations_dir=annotations_dir,
        )
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
        bio_added = await self._populate_trial_biology(
            trials, annotations_dir=annotations_dir,
        )
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

    async def _identify_diagnostic_compounds(
        self, trials: list[TrialRecord],
    ) -> set[str]:
        """Build a set of compound_ids whose ChEMBL profile signals
        diagnostic intent — Lymphoseek-class radiopharmaceuticals,
        Fluorescein-style fluorescence-guided-surgery dyes, etc.

        For each unique drug-like intervention across the trial corpus:
          1. Resolve to a ChEMBL molecule id via Open Targets first,
             then fall back to ChEMBL's own molecule search (same
             order as the non-protein-target inference path).
          2. Fetch `{therapeutic_flag, max_phase, indication_class}`.
          3. Decide via `is_likely_diagnostic`.

        Returns the set of NORMALIZED compound_ids to skip. Both the
        `map_trial_to_graph_nodes` loop and the per-trial `build_arms`
        path filter against this set, so diagnostic interventions
        never get a CompoundNode or arm entry.

        Safety: a compound in `_VACCINE_COMPONENT_TARGETS` or
        `_CELL_THERAPY_COMPONENT_TARGETS` is explicitly therapeutic —
        we DO NOT filter them even if ChEMBL would mark them
        diagnostic. T-VEC is a real-world example: ChEMBL's
        `therapeutic_flag=False` is a curation error.
        """
        diagnostic_cids: set[str] = set()
        seen_canonicals: set[str] = set()

        # Round-21 Phase B: side-effect of the diagnostic OT/ChEMBL
        # lookups is that we learn each compound's chembl_id. Stash
        # those per-canonical-name so Step 2's compound-add loop can
        # use them for stable_id-based canonicalization without
        # re-running OT lookups. Maps `_norm_for_pattern_match(name) →
        # chembl_id`. Compounds without a resolved chembl_id are
        # absent from the dict (rather than mapping to None) so
        # `if chembl_id := lookup.get(...)` is a clean check.
        self._compound_chembl_ids: dict[str, str] = {}

        # Build the override set: any canonical name (lowercased) whose
        # compound id is referenced by the vaccine / cell-therapy
        # heuristics is exempt from filtering — those heuristics are
        # our source of truth for "this drug is therapeutic," not
        # ChEMBL's flag.
        therapeutic_override_patterns: list[str] = []
        for tbl in (_VACCINE_COMPONENT_TARGETS, _CELL_THERAPY_COMPONENT_TARGETS):
            for pattern, _ in tbl:
                therapeutic_override_patterns.append(
                    _norm_for_pattern_match(pattern)
                )

        for trial in trials:
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                for canonical_name in canonical_compound_names(iv.name):
                    if canonical_name in seen_canonicals:
                        continue
                    seen_canonicals.add(canonical_name)

                    # Heuristic-protected drugs skip ChEMBL filtering.
                    normalized = _norm_for_pattern_match(canonical_name)
                    if any(p in normalized for p in therapeutic_override_patterns):
                        continue

                    # Resolve chembl_id. Try OT first (uses existing
                    # cache), then ChEMBL search-by-name as fallback.
                    chembl_id: str | None = None
                    ot_data: dict | None = None
                    try:
                        ot_data = await self._ot_client.get_drug_with_targets(
                            canonical_name,
                        )
                        chembl_id = ot_data.get("chembl_id")
                    except KeyError:
                        ot_data = None
                        chembl_id = None
                    except Exception:
                        logger.debug(
                            "OT lookup for diagnostic check failed: %s",
                            canonical_name, exc_info=True,
                        )
                        ot_data = None
                        chembl_id = None
                    # Persist whatever OT gave us onto the compound node
                    # (chembl_id + external aliases). Idempotent — safe
                    # on the second OT pass during _populate_compound_targets.
                    if ot_data:
                        cid_for_annotation = normalize_entity(
                            canonical_name, "InterventionNode",
                        )
                        self._annotate_compound_node_from_ot(
                            cid_for_annotation, ot_data,
                        )
                    if not chembl_id:
                        chembl_id = await self._chembl_client.search_drug_by_name(
                            canonical_name,
                        )
                    if chembl_id:
                        # Round-21 Phase B: cache for stable_id lookup
                        # in the compound-add loop. Same key shape as
                        # the override-pattern set, so Step 2 can
                        # normalize the incoming intervention name the
                        # same way before looking up.
                        self._compound_chembl_ids[
                            _norm_for_pattern_match(canonical_name)
                        ] = chembl_id
                    if not chembl_id:
                        # No ChEMBL record → no signal to filter on,
                        # safer to keep.
                        continue

                    # Rescue: if Open Targets resolved this compound to molecular
                    # target(s), it's a real therapeutic — drugs have targets,
                    # imaging dyes / diagnostics don't. ChEMBL's therapeutic_flag
                    # is too noisy to drop a drug that OT says hits a target. This
                    # disambiguates the diagnostic-shaped profile (therapeutic_flag
                    # False + max_phase None) that real drugs share with Fluorescein
                    # by metadata alone. Only ever *un-drops*, never adds a drop.
                    if ot_data and ot_data.get("targets"):
                        continue

                    meta = await self._chembl_client.get_molecule_metadata(
                        chembl_id,
                    )
                    if is_likely_diagnostic(meta):
                        cid = normalize_entity(
                            canonical_name, "InterventionNode",
                        )
                        diagnostic_cids.add(cid)
                        logger.info(
                            "Diagnostic-filter: skipping %r (ChEMBL %s, "
                            "therapeutic_flag=%s, max_phase=%s, "
                            "indication_class=%s)",
                            canonical_name, chembl_id,
                            meta.get("therapeutic_flag"),
                            meta.get("max_phase"),
                            meta.get("indication_class"),
                        )
        return diagnostic_cids

    async def _resolve_compound_codenames(
        self, trials: list[TrialRecord],
    ) -> None:
        """Run the RxNorm + PubChem fallback for unresolved codenames.

        For each trial's compound names, skip ones already handled by
        either (a) the curated ``CODENAME_TO_INN`` dict, (b) the OT /
        ChEMBL chembl_id lookup from Step 1.5, or (c) the diagnostic
        filter. Whatever's left is a codename that the existing layers
        couldn't recognize — typically a development code or a niche
        agent (CMP-001, IMA-201, ABT-263). Ask RxNorm first (cheap,
        excellent for FDA-tracked codenames), then PubChem (broader
        synonym coverage for older/research codes).

        Resolved entries are stored on ``self._codename_resolutions``
        keyed by the compound's slug-id (the same key the compound-add
        loop uses), so the loop can apply the rename + alias-push
        without re-resolving.
        """
        if not trials:
            return
        # Dedup compound ids across all trials so we hit each codename
        # at most once even if N trials reference the same one.
        seen_ids: set[str] = set()
        candidates: list[tuple[str, str]] = []  # (comp_id, comp_name)
        for trial in trials:
            try:
                nodes = map_trial_to_graph_nodes(trial)
            except Exception:
                logger.debug(
                    "Codename resolver: map_trial_to_graph_nodes failed for %s",
                    trial.nct_id, exc_info=True,
                )
                continue
            for comp in nodes["compounds"]:
                if comp.id in seen_ids:
                    continue
                seen_ids.add(comp.id)
                if comp.id in self._diagnostic_compound_ids:
                    continue
                if comp.id in CODENAME_TO_INN:
                    continue
                if _norm_for_pattern_match(comp.name) in self._compound_chembl_ids:
                    continue
                candidates.append((comp.id, comp.name))

        if not candidates:
            return

        # Build a normalized index of compounds we ALREADY know about
        # (via CODENAME_TO_INN values + the OT/ChEMBL cache) so PubChem
        # synonym-matching can prefer pulling onto an existing node.
        known_index: dict[str, str] = {}
        for inn_slug in set(CODENAME_TO_INN.values()):
            known_index[_norm_for_pattern_match(inn_slug)] = inn_slug
        for norm_name in self._compound_chembl_ids:
            known_index[norm_name] = norm_name

        resolved_count = 0
        for comp_id, comp_name in candidates:
            try:
                resolved = await self._codename_resolver.resolve(
                    comp_name, known_compound_index=known_index,
                )
            except Exception:
                logger.debug(
                    "Codename resolver failed for %r", comp_name, exc_info=True,
                )
                continue
            if resolved is None:
                continue
            self._codename_resolutions[comp_id] = resolved
            resolved_count += 1
            logger.info(
                "Codename resolver: %r → %r (source=%s)",
                comp_name, resolved.canonical_name, resolved.source,
            )

        if resolved_count:
            console.print(
                f"  Codename resolver: {resolved_count}/{len(candidates)} "
                f"unresolved compounds remapped via RxNorm + PubChem"
            )

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
        from src.graph.indication_taxonomy import (
            is_nondisease_condition,
            slugify_disease_name,
        )

        if self._indication_canon is None:
            self._indication_canon = JSONCache(
                self._cache_dir / "indication_canonicalizations.json"
            )
        cache = self._indication_canon

        cached = cache.get(cond)
        if cached is not None:
            slug, _, name = cached.partition("|")
            # Empty cached slug = a prior non-disease drop; also re-gate any
            # legacy cached slug so pre-gate caches are cleaned up on read.
            if not slug or is_nondisease_condition(slug):
                return "", ""
            return slug, name or slug.replace("_", " ")

        if self._anthropic is None:
            slug = slugify_disease_name(cond) or normalize_entity(
                cond, "IndicationNode",
            )
            if is_nondisease_condition(slug):
                cache.set(cond, "|")  # sentinel: non-disease condition, dropped
                return "", ""
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

        # Condition-validity gate: drop age-group / non-disease conditions
        # (e.g. "Child") so they never become IndicationNodes. The caller
        # skips on an empty id; if a trial's only conditions are non-diseases
        # it gets no indication (and is dropped by the no-indication path).
        if is_nondisease_condition(slug):
            cache.set(cond, "|")  # sentinel: non-disease condition, dropped
            return "", ""

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

        # Collect (compound_id, canonical_name) pairs from drug-like
        # interventions. Expanded via `canonical_compound_names` so a
        # joined CT.gov regimen string like "Sorafenib + Dacarbazine"
        # produces TWO lookups — one per constituent. Pre-round-12 this
        # iterated over `iv.name` directly, which failed for joined
        # strings: `resolve_entity("Sorafenib + Dacarbazine")` returned
        # None and the inner constituents never got their own OT lookup
        # (the dacarbazine→MET cross-reference bug from round 11).
        seen: dict[str, str] = {}  # compound_id → canonical name to query
        for trial in trials:
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                for canonical_name in canonical_compound_names(iv.name):
                    cid = self.resolve_entity(canonical_name, "compound")
                    if not cid or cid in seen:
                        continue
                    seen[cid] = canonical_name

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

            # Round-15 canonicalization fix: persist chembl_id + external
            # aliases from OT onto the CompoundNode. Idempotent — also
            # called in _identify_diagnostic_compounds; this second call
            # picks up any compounds that weren't filtered at that step.
            self._annotate_compound_node_from_ot(cid, drug_data)

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
                        target_type=TargetType.PROTEIN,
                        metadata={"source": "opentargets"},
                    ))
                    self._index_node(ensembl, symbol, "target")
                if not self.graph._graph.has_edge(  # noqa: SLF001
                    cid, ensembl, key=EdgeType.AFFECTS.value,
                ):
                    # Round-25: emit a DATABASE_OT_DIRECT record carrying
                    # OT's binding claim, instead of hand-setting a
                    # Beta(2, 1) prior.
                    record = make_curated_record(
                        source_id=(
                            f"opentargets:"
                            f"{drug_data.get('chembl_id') or 'unknown_chembl'}"
                        ),
                        bucket=SupportBucket.STRONG_SUPPORT,
                        source_type=EvidenceType.DATABASE_OT_DIRECT,
                        notes=f"OT drug-target binding: {cid} → {symbol}",
                    )
                    self.graph.add_edge(GraphEdge(
                        source_id=cid,
                        target_id=ensembl,
                        edge_type=EdgeType.AFFECTS,
                        belief=belief_from_curated_record(record),
                        metadata={
                            "source": "opentargets",
                            "drug_chembl_id": drug_data.get("chembl_id"),
                        },
                    ))
                    binds_added += 1
                target_ids.append(ensembl)

            # Non-protein fallback: when OT returns no protein targets,
            # query ChEMBL's REST API directly. OT generic-ifies
            # action_type to "INHIBITOR" and returns null mechanismsOfAction
            # for many drugs; ChEMBL's raw API has the structured
            # `target_chembl_id` pointing to typed entities (NUCLEIC-ACID
            # for DNA, PROTEIN COMPLEX GROUP for tubulin). The
            # ChEMBLClient handles the lookup + translation to our
            # namespaced ids. Prior Beta(3.5, 1) — slightly below
            # OT-direct's Beta(4, 1) since the target is inferred from
            # mechanism text rather than directly listed.
            chembl_id = drug_data.get("chembl_id")
            if not target_ids:
                # Pass both chembl_id (if OT resolved it) AND the raw
                # intervention name (so ChEMBL's molecule search can
                # find drugs OT missed — dacarbazine, fotemustine,
                # etc., per the round-11 audit gap).
                inferred = await self._chembl_client.infer_non_protein_target(
                    chembl_id, drug_name=iv_name,
                )
                if inferred is not None:
                    tid, tname, ttype, mech = inferred
                    try:
                        self.graph.get_node(tid)
                    except KeyError:
                        self.graph.add_node(TargetNode(
                            id=tid,
                            name=tname,
                            gene_symbol=tid,
                            target_type=ttype,
                            metadata={
                                "source": "chembl_rest_fallback",
                            },
                        ))
                        self._index_node(tid, tname, "target")
                    if not self.graph._graph.has_edge(  # noqa: SLF001
                        cid, tid, key=EdgeType.AFFECTS.value,
                    ):
                        edge_meta = {
                            "source": "chembl_rest_fallback",
                            "drug_chembl_id": chembl_id,
                        }
                        if mech is not None:
                            edge_meta["implied_mechanism"] = mech.value
                        # Round-25: DATABASE_CHEMBL record from ChEMBL's
                        # structured action_type. Same n_eff as OT-direct.
                        record = make_curated_record(
                            source_id=(
                                f"chembl_rest:"
                                f"{chembl_id or 'unknown_chembl'}"
                            ),
                            bucket=SupportBucket.STRONG_SUPPORT,
                            source_type=EvidenceType.DATABASE_CHEMBL,
                            notes=(
                                f"ChEMBL action_type → target type "
                                f"{ttype.value if hasattr(ttype, 'value') else ttype}"
                            ),
                        )
                        self.graph.add_edge(GraphEdge(
                            source_id=cid,
                            target_id=tid,
                            edge_type=EdgeType.AFFECTS,
                            belief=belief_from_curated_record(record),
                            metadata=edge_meta,
                        ))
                        binds_added += 1
                    target_ids.append(tid)

            # Round-24 Q3: monoclonal antibody fallback. When both OT
            # and the ChEMBL REST fallback miss for a recognizable mAb,
            # consult the curated antibody → target table. The table
            # only carries well-characterized antibodies (anti-amyloid,
            # checkpoint inhibitors, anti-CD20, etc.) — caller intent
            # here is to fail closed (return None) rather than guess.
            if not target_ids and is_monoclonal_antibody(cid):
                mab_gene = mab_target_gene(cid)
                if mab_gene:
                    try:
                        ot_target = await self._ot_client.search_target(mab_gene)
                    except (KeyError, Exception):  # noqa: BLE001
                        ot_target = None
                    if ot_target and ot_target.get("target_id"):
                        ensembl = ot_target["target_id"]
                        try:
                            self.graph.get_node(ensembl)
                        except KeyError:
                            self.graph.add_node(TargetNode(
                                id=ensembl,
                                name=ot_target.get("name") or mab_gene,
                                gene_symbol=ot_target.get("approved_symbol") or mab_gene,
                                target_type=TargetType.PROTEIN,
                                metadata={"source": "mab_fallback"},
                            ))
                            self._index_node(
                                ensembl,
                                ot_target.get("approved_symbol") or mab_gene,
                                "target",
                            )
                        if not self.graph._graph.has_edge(  # noqa: SLF001
                            cid, ensembl, key=EdgeType.AFFECTS.value,
                        ):
                            # Round-25: hand-curated mAb-table record.
                            # Same n_eff as OT-direct — both are curated
                            # multi-source assertions about a specific
                            # compound→target pair.
                            record = make_curated_record(
                                source_id=f"mab_table:{cid}",
                                bucket=SupportBucket.STRONG_SUPPORT,
                                source_type=EvidenceType.DATABASE_MAB_TABLE,
                                notes=(
                                    f"hand-curated mAb→target: "
                                    f"{cid} → {mab_gene}"
                                ),
                            )
                            self.graph.add_edge(GraphEdge(
                                source_id=cid,
                                target_id=ensembl,
                                edge_type=EdgeType.AFFECTS,
                                belief=belief_from_curated_record(record),
                                metadata={
                                    "source": "mab_fallback",
                                    "mab_table_gene": mab_gene,
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
            ot_score = row.get("overall_score", 0.0)
            ev_count = row.get("evidence_count", 0)
            # Round-25: DATABASE_OT_ASSOCIATION record. Lower n_eff than
            # OT-direct because the overall_score is an aggregate
            # heuristic over multiple underlying types, not a single
            # primary assertion.
            assoc_record = make_curated_record(
                source_id=f"opentargets_assoc:{tid}__{efo}",
                bucket=ot_association_score_to_bucket(ot_score),
                source_type=EvidenceType.DATABASE_OT_ASSOCIATION,
                quality_score=ot_score_quality(ev_count),
                notes=(
                    f"OT target-disease association: "
                    f"score={ot_score:.3f}, evidence_count={ev_count}"
                ),
            )
            self.graph.add_edge(GraphEdge(
                source_id=tid,
                target_id=ind_id,
                edge_type=EdgeType.BIOLOGY_DRIVES,
                belief=belief_from_curated_record(assoc_record),
                metadata={
                    "source": "opentargets",
                    "efo_id": efo,
                    "overall_score": row.get("overall_score"),
                    "datatypes": row.get("datatypes"),
                },
            ))
            added += 1
        return added

    def _mechanism_intervention_keys(self, compound_id: str) -> set[str]:
        """Intervention-match keys for a constituent compound.

        Mirrors :func:`_lookup_chain_intervention_desc`'s key derivation
        (used for biology) so the mechanism path looks up the same per-(arm,
        intervention) extraction entries: the compound id AND its node name
        (the LLM names the drug, which canonicalizes to either), normalized
        via ``box_embeddings._intervention_key``.
        """
        from src.graph.box_embeddings import _intervention_key

        keys = {_intervention_key(compound_id)}
        try:
            name = self.graph.get_node(compound_id).get("name") or ""
        except KeyError:
            name = ""
        if name:
            keys.add(_intervention_key(name))
        keys.discard("")
        return keys

    async def _populate_trial_mechanisms(
        self,
        trials: list[TrialRecord],
        compound_targets: dict[str, list[str]],
        annotations_dir: str | Path | None = None,
    ) -> int:
        """Resolve target + mechanism PER CONSTITUENT and rebuild chains.

        Semantic-layers redesign: a Mechanism node IS the localized molecular
        SIGNALING PATHWAY the target gene drives (a Reactome pathway — RAS/MAPK,
        PI3K/AKT, "Co-inhibition by PD-1", …), NOT the drug-class action mode.
        This is the layer where mechanistic knowledge COMPOUNDS across targets:
        an EGFR inhibitor and a BRAF inhibitor both hit MAPK, so both resolve to
        the SAME Reactome pathway id and the Tier-1 id merge collapses them onto
        one MechanismNode.

        A combo arm tests N mechanism paths simultaneously, so it produces at
        least N chains—one per constituent compound. Each constituent further
        fans out one chain PER resolved mechanism pathway (the gene's full
        signaling footprint), all sharing the same downstream biology.

        Per constituent we:
          1. Look up the constituent's primary target via OT-resolved
             ``compound_targets`` (first entry; can be UNKNOWN if OT had
             no hit).
          2. Resolve the drug-class ``MechanismCategory`` (structured
             ``implied_mechanism`` on the AFFECTS edge, else Haiku). The category
             + the ``MechanismType`` it derives to are DRUG properties → stamped
             onto the chain's InterventionNode (the compound). The per-(drug,
             pathway) ``direction`` it derives to is written into each
             ``modulates_via`` edge's metadata. NONE of the three land on the
             (drug-agnostic, shared) Reactome MechanismNode.
          3. Resolve the target GENE → Reactome pathway footprint via the shared
             ``_resolve_gene_pathways`` core. The MechanismNode IDENTITY (id +
             name) is the Reactome pathway; add the ``target → mechanism``
             modulates_via edge (idempotent) for each pathway. When the gene has
             no resolvable pathway, fall back to the category enum slug.

        ``annotations_dir`` (semantic-layers redesign): when the build extracted
        before populate, the per-(arm, intervention) ``mechanism_category`` the
        extractor emitted is preferred for the ``category`` metadata, and the
        extracted ``mechanism`` action description (when present) rides along as
        the node's free-text substrate + the pathway re-rank context. The
        node IDENTITY is the Reactome pathway, not a content-address of the
        description (that path is superseded).

        The chain list is then rebuilt: for every existing
        subgroup_population_id × arm cell, emit one chain per (constituent,
        mechanism pathway). ``chain.compound_id`` is the constituent (not the
        combo regimen) so binds_to lookup walks to the constituent's own
        target. ``chain.metadata["regimen_id"]`` records the arm's
        regimen for downstream grouping. ``chain.biology_id`` is left UNKNOWN —
        biology is assigned later in ``_populate_trial_biology``.

        Returns the number of new modulates_via edges added.
        """
        if self._anthropic is None:
            logger.info("No anthropic client; skipping mechanism inference")
            return 0

        # Per-(arm, intervention) extracted mechanism_category, used to stamp
        # the InterventionNode's ``category`` (drug-class) metadata. Empty when
        # the build hasn't extracted yet (no annotations_dir) — category then
        # defaults to the resolved enum id.
        cat_by_arm_iv: dict = {}
        if annotations_dir is not None:
            from src.graph.box_embeddings import (
                chain_descriptions_by_arm_intervention,
            )
            cat_by_arm_iv = chain_descriptions_by_arm_intervention(
                annotations_dir
            )

        cache = JSONCache(self._cache_dir / "mechanism_inferences.json")

        # Semantic-layers redesign: the MechanismNode is the localized molecular
        # signaling pathway the target gene drives (Reactome), resolved here via
        # the shared gene→pathway core. One LINCSClient + QuickGOClient per call
        # so the on-disk Reactome / GO caches are shared across trials. Both are
        # keyless for the endpoints we use, so construction tolerates a missing
        # CLUE_API_KEY; if LINCSClient can't construct at all we fall back to the
        # category enum slug (the pre-redesign behavior).
        lincs_client: "LINCSClient | None"
        try:
            lincs_client = LINCSClient()
        except Exception:
            logger.debug("LINCSClient unavailable for mechanism resolution",
                         exc_info=True)
            lincs_client = None
        from src.ingestion.quickgo import QuickGOClient
        quickgo_client = QuickGOClient()

        added = 0
        for trial in trials:
            try:
                ts = self.graph.get_trial_subgraph_by_id(trial.nct_id)
            except KeyError:
                continue
            if not ts.arms:
                continue

            # The trial's representative indication, for the pathway re-rank
            # context. Chains in a trial share the same root indication
            # (build_trial_subgraphs anchored them on the parent disease); take
            # the first resolved one, else UNKNOWN (the resolver tolerates it).
            trial_indication_id = _UNKNOWN
            for _ch in ts.chains:
                if _ch.indication_id and _ch.indication_id != _UNKNOWN:
                    trial_indication_id = _ch.indication_id
                    break

            # Per-(arm, constituent) backbones. Order within each arm
            # follows arm.compound_ids so chain ordering is deterministic.
            # A single constituent can produce MULTIPLE backbone entries — one
            # per resolved mechanism pathway (the gene's signaling footprint) —
            # which fans the chain out one chain per mechanism pathway.
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

                    # Prefer structured truth over LLM inference: when the
                    # AFFECTS edge from compound→target carries a
                    # `implied_mechanism` (set by `chembl_action_fallback`
                    # for non-protein targets), use it directly and skip
                    # the Sonnet call. ChEMBL's `action_type` is curated;
                    # the LLM is only needed when no structured source
                    # named the mechanism.
                    implied = None
                    if target_id:
                        try:
                            affects_meta = self.graph._graph.get_edge_data(  # noqa: SLF001
                                cid, target_id, key=EdgeType.AFFECTS.value,
                            ) or {}
                            implied = (affects_meta.get("metadata") or {}).get(
                                "implied_mechanism"
                            )
                        except Exception:
                            implied = None
                    if implied:
                        mech_value = implied
                    else:
                        mech_value = await infer_mechanism_for_arm(
                            self._anthropic,
                            trial,
                            arm_label=arm_label,
                            arm_compound_names=[constituent_name],
                            target_node=target_node,
                            cache=cache,
                            cache_key=f"{trial.nct_id}::{arm.arm_id}::{cid}",
                        )
                    # The resolved enum slug (a ``MechanismCategory`` value).
                    # Used as: (a) the no-description fallback node id, for
                    # back-compat; (b) the default ``category`` bucket when the
                    # extractor didn't emit one.
                    enum_slug = normalize_entity(mech_value, "MechanismNode")

                    # Mechanism-identity flip (Phase C). The extractor's
                    # per-(arm, intervention) entry carries BOTH the specific
                    # molecular-action ``mechanism`` description (the node's new
                    # IDENTITY) AND the coarse ``mechanism_category`` bucket
                    # (METADATA). Pull both from the SAME entry so id + category
                    # stay consistent. Mirrors the biology path's lookup keys.
                    mech_desc = ""
                    extracted_cat = ""
                    if cat_by_arm_iv:
                        for k in self._mechanism_intervention_keys(cid):
                            entry = cat_by_arm_iv.get((trial.nct_id, arm.arm_id, k))
                            if not entry:
                                continue
                            if not mech_desc and (entry.get("mechanism") or "").strip():
                                mech_desc = entry["mechanism"].strip()
                            if not extracted_cat and (
                                entry.get("mechanism_category") or ""
                            ).strip():
                                extracted_cat = entry["mechanism_category"].strip()
                            if mech_desc and extracted_cat:
                                break

                    # Drug-class category: prefer the extractor's
                    # ``mechanism_category`` when it's a valid MechanismCategory
                    # value; otherwise fall back to the resolved enum slug
                    # (itself a MechanismCategory value in this path). This is the
                    # coarse drug-class bucket; it is stamped onto the DRUG
                    # (InterventionNode), not the shared pathway MechanismNode.
                    category_value = enum_slug
                    if extracted_cat:
                        try:
                            category_value = MechanismCategory(
                                normalize_entity(extracted_cat, "MechanismNode")
                            ).value
                        except ValueError:
                            # Not a recognized category value — keep the
                            # resolved enum slug as the bucket.
                            category_value = enum_slug

                    # Semantic-layers redesign (2026-06-02): the drug-class
                    # CATEGORY + the MechanismType it derives to are properties
                    # of the DRUG, not of the shared Reactome signaling pathway.
                    # Derive them from the resolved category (guard for a
                    # category that isn't a valid enum — keep the OTHER / None
                    # fallback that ``_category_to_*`` already provides) and stamp
                    # them onto the chain's InterventionNode (the compound
                    # ``cid``), NOT onto the MechanismNode. ``direction`` is
                    # per-(drug, pathway) and is written into each
                    # ``modulates_via`` edge's metadata below.
                    try:
                        category_enum = MechanismCategory(category_value)
                    except ValueError:
                        category_enum = MechanismCategory.OTHER
                    mechanism_type_value = _category_to_mechanism_type(
                        category_enum
                    ).value
                    direction_value = _category_to_direction(category_enum)

                    # Stamp drug-class category + mechanism_type onto the
                    # InterventionNode (the drug). A drug has a single intrinsic
                    # class regardless of how many pathways it engages, so write
                    # once per constituent (idempotent backfill — don't clobber a
                    # value an earlier arm/trial already set).
                    try:
                        iv_data = self.graph._graph.nodes[cid]  # noqa: SLF001
                    except KeyError:
                        iv_data = None
                    if iv_data is not None:
                        if not iv_data.get("category"):
                            iv_data["category"] = category_value
                        if not iv_data.get("mechanism_type"):
                            iv_data["mechanism_type"] = mechanism_type_value

                    def _ensure_mechanism(
                        mech_id: str, mech_name: str, description: str,
                        source: str,
                    ) -> None:
                        """Create (idempotently) the drug-agnostic MechanismNode
                        (a Reactome signaling pathway — id + name + description ARE
                        the pathway, no drug-class properties) + the
                        target→mechanism ``modulates_via`` edge (which carries
                        this drug's ``direction`` in its metadata), and append the
                        backbone entry."""
                        nonlocal added
                        try:
                            self.graph.get_node(mech_id)
                        except KeyError:
                            self.graph.add_node(MechanismNode(
                                id=mech_id,
                                name=mech_name,
                                description=description,
                            ))

                        if target_id and not self.graph._graph.has_edge(  # noqa: SLF001
                            target_id, mech_id, key=EdgeType.MODULATES_VIA.value,
                        ):
                            # Round-25: trial-inference mechanism. The mechanism
                            # was inferred from a trial's therapeutic_hypothesis
                            # combined with the compound's curated mechanism
                            # class. Treated as a CHEMBL-tier record (the
                            # mechanism class itself comes from ChEMBL via OT).
                            modulates_record = make_curated_record(
                                source_id=(
                                    f"trial_inference:{trial.nct_id}__{cid}"
                                    f"__{mech_id}"
                                ),
                                bucket=SupportBucket.MODERATE_SUPPORT,
                                source_type=EvidenceType.DATABASE_CHEMBL,
                                notes=(
                                    f"target→mechanism inferred from trial "
                                    f"{trial.nct_id} arm {arm.arm_id} "
                                    f"compound {cid}"
                                ),
                            )
                            self.graph.add_edge(GraphEdge(
                                source_id=target_id,
                                target_id=mech_id,
                                edge_type=EdgeType.MODULATES_VIA,
                                belief=belief_from_curated_record(
                                    modulates_record
                                ),
                                metadata={
                                    "source": source,
                                    "trial_id": trial.nct_id,
                                    "arm_id": arm.arm_id,
                                    "constituent_id": cid,
                                    # Per-(drug, pathway) direction-of-effect.
                                    # Lives on the EDGE (not the shared pathway
                                    # node) because how a drug engages a pathway
                                    # — inhibiting / activating / modulating — is
                                    # specific to this (drug, pathway) pair.
                                    # Derived from the drug's category. Omitted
                                    # (None) for OTHER / unclassified.
                                    "direction": direction_value,
                                },
                            ))
                            added += 1

                        arm_entries.append({
                            "compound_id": cid,
                            "target_id": target_id or _UNKNOWN,
                            "mechanism_id": mech_id,
                        })

                    # Mechanism = the localized molecular SIGNALING PATHWAY(S) the
                    # target gene drives (semantic-layers redesign). Resolve the
                    # gene → Reactome pathway footprint via the shared core; the
                    # MechanismNode becomes each Reactome pathway (id + name) and
                    # the chain fans out one chain per pathway. KEEP MULTIPLE
                    # pathways (the gene's full footprint — the messy
                    # polypharmacology vs the clean clinical hypothesis is desired
                    # data); capped at ``_MECHANISM_PATHWAY_CAP`` only to bound the
                    # combinatorial fan-out. The extracted drug-class category
                    # lives on the InterventionNode + modulates_via edge (above),
                    # not the pathway node; ``mech_desc`` is the pathway re-rank
                    # context + the node's free-text substrate.
                    pathways, mech_source, _mech_score = (
                        await self._resolve_gene_pathways(
                            target_id or _UNKNOWN,
                            # Re-rank context: the resolved category slug (expands
                            # via MECHANISM_PATHWAY_TOKENS) plus the extracted
                            # action description when present.
                            (mech_desc or category_value),
                            trial_indication_id,
                            lincs_client,
                            quickgo_client=quickgo_client,
                        )
                    )

                    if pathways:
                        for pathway in pathways[: self._MECHANISM_PATHWAY_CAP]:
                            _ensure_mechanism(
                                mech_id=pathway.stable_id,
                                mech_name=pathway.display_name or pathway.stable_id,
                                # The node identity is the pathway; the extracted
                                # action description (when any) is preserved as the
                                # node's free-text substrate for later embedding.
                                description=mech_desc,
                                source=mech_source,
                            )
                    else:
                        # No resolvable signaling pathway for the gene (target
                        # UNKNOWN / non-ENSG / no gene symbol / Reactome+GO empty)
                        # → fall back to the coarse drug-class enum slug so the
                        # chain still has a mechanism backbone. NOT a content-
                        # address: pathway identity supersedes that path.
                        _ensure_mechanism(
                            mech_id=enum_slug,
                            mech_name=enum_slug.replace("_", " "),
                            description=mech_desc,
                            source="trial_inference",
                        )
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

    async def _infer_missing_compound_targets(
        self,
        trials: list,
        compound_targets: dict[str, list[str]],
    ) -> tuple[int, dict[str, list[str]]]:
        """LLM-infer gene targets for compounds the curated + heuristic
        resolvers missed (OT / ChEMBL / mAb / vaccine / cell-therapy all
        failed), at the compound→target seam BEFORE the name-match fallback.

        For each unresolved, non-diagnostic constituent compound, ask an LLM for
        ``(gene_symbol, intervention_class)``. Keep only THERAPEUTIC
        interventions whose gene VALIDATES to a real Ensembl id via OT
        ``search_target`` (hallucinated / non-gene answers are dropped), then add
        a TargetNode + AFFECTS edge at the modest ``DATABASE_LLM_INFERENCE``
        n_eff. Cost-bounded to the misses (~100 compounds ≪ $10 at Haiku rates)
        and cached on disk. Returns ``(edges_added, {compound_id: [ensembl]})``.
        """
        del trials  # reserved for a future indication hint; misses come from
        #             the POST-COLLAPSE graph so resurrected codenames can't
        #             appear (map_trial_to_graph_nodes yields raw pre-collapse
        #             ids; the codename→INN remap happens earlier in populate).
        if self._anthropic is None or self._ot_client is None:
            return 0, {}

        # Unresolved compounds (id → display name): the compound nodes already
        # in the graph (canonical, post-collapse) that have no resolved target.
        misses: dict[str, str] = {}
        for ntype in ("CompoundNode", "InterventionNode"):
            for node in self.graph.get_nodes_by_type(ntype):
                cid = node["id"]
                if cid in misses or compound_targets.get(cid):
                    continue
                if cid in self._diagnostic_compound_ids:
                    continue
                misses[cid] = node.get("name") or cid

        if not misses:
            return 0, {}

        cache = JSONCache(self._cache_dir / "llm_inferred_targets.json")
        added = 0
        new_targets: dict[str, list[str]] = {}
        _MAX_CALLS = 400  # backstop; real miss count is ~100
        n_calls = 0
        for cid, name in misses.items():
            cached = cache.get(cid)
            if cached is None:
                if n_calls >= _MAX_CALLS:
                    logger.warning("LLM target-inference hit the %d-call cap", _MAX_CALLS)
                    break
                cached = await self._llm_infer_compound_target(name, "")
                n_calls += 1
                cache.set(cid, cached)
            gene, _, iclass = cached.partition("|")
            gene, iclass = gene.strip(), iclass.strip().lower()
            if not gene or gene.upper() == "UNKNOWN":
                continue
            if iclass and iclass != "therapeutic":
                continue  # intervention-class gate
            try:
                ot_target = await self._ot_client.search_target(gene)
            except Exception:  # noqa: BLE001
                ot_target = None
            if not ot_target or not ot_target.get("target_id"):
                continue  # gene didn't validate to a real Ensembl target
            ensembl = ot_target["target_id"]
            symbol = ot_target.get("approved_symbol") or gene
            try:
                self.graph.get_node(ensembl)
            except KeyError:
                self.graph.add_node(TargetNode(
                    id=ensembl,
                    name=ot_target.get("name") or symbol,
                    gene_symbol=symbol,
                    target_type=TargetType.PROTEIN,
                    metadata={"source": "llm_inference"},
                ))
                self._index_node(ensembl, symbol, "target")
            if not self.graph._graph.has_edge(  # noqa: SLF001
                cid, ensembl, key=EdgeType.AFFECTS.value
            ):
                record = make_curated_record(
                    source_id=f"llm_inference:{cid}",
                    bucket=SupportBucket.MODERATE_SUPPORT,
                    source_type=EvidenceType.DATABASE_LLM_INFERENCE,
                    notes=f"LLM-inferred drug→target: {cid} → {symbol}",
                )
                self.graph.add_edge(GraphEdge(
                    source_id=cid,
                    target_id=ensembl,
                    edge_type=EdgeType.AFFECTS,
                    belief=belief_from_curated_record(record),
                    metadata={"source": "llm_inference"},
                ))
                added += 1
            new_targets.setdefault(cid, []).append(ensembl)
        logger.info(
            "LLM target-inference: resolved %d/%d unresolved compounds "
            "(%d LLM calls)", len(new_targets), len(misses), n_calls,
        )
        return added, new_targets

    async def _llm_infer_compound_target(self, name: str, indication: str) -> str:
        """One Haiku call: drug name (+ indication hint) → ``"GENE | class"``.
        Returns ``"UNKNOWN|..."`` when there is no single protein-coding-gene
        target or the model isn't confident (abstention preferred to a guess)."""
        ctx = f"\nIndication context: {indication}" if indication else ""
        user_msg = (
            f"Therapeutic intervention: {name!r}{ctx}\n\n"
            "What is the PRIMARY molecular target — the gene whose protein "
            "product this intervention binds or directly modulates?\n\n"
            "Reply EXACTLY as: GENE_SYMBOL | intervention_class\n"
            "  GENE_SYMBOL: official HUGO gene symbol (EGFR, PDCD1, HMGCR, ...). "
            "Reply UNKNOWN if there is no single protein-coding-gene target "
            "(cytotoxic chemo hitting DNA/microtubules, radiation, a device, a "
            "supplement, an imaging / diagnostic agent) or you are not confident.\n"
            "  intervention_class: one of therapeutic, diagnostic, "
            "supportive_care, non_therapeutic. Antibiotics, antiemetics, "
            "colony-stimulating factors, and other agents present for SUPPORTIVE "
            "care in a cancer trial are supportive_care, not therapeutic.\n\n"
            "Examples:\n"
            "  'Osimertinib' -> EGFR | therapeutic\n"
            "  'Pembrolizumab' -> PDCD1 | therapeutic\n"
            "  'Atorvastatin' -> HMGCR | therapeutic\n"
            "  'Axitinib' -> KDR | therapeutic\n"
            "  'Paclitaxel' -> UNKNOWN | therapeutic\n"
            "  'Trimethoprim-Sulfamethoxazole' -> DHFR | supportive_care\n"
            "  'Technetium Tc-99m Sestamibi' -> UNKNOWN | diagnostic\n\n"
            "Reply with only 'GENE | class'. No other text."
        )
        try:
            response = await _call_messages_with_backoff(
                self._anthropic,
                model=TARGET_INFERENCE_MODEL,
                max_tokens=20,
                temperature=0,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text.strip()
        except Exception:  # noqa: BLE001
            logger.debug("LLM target inference failed for %r", name, exc_info=True)
            return "UNKNOWN|"

    async def _resolve_gene_pathways(
        self,
        target_id: str,
        context_name: str,
        indication_id: str,
        lincs_client: "LINCSClient | None",
        quickgo_client: "QuickGOClient | None" = None,
    ) -> tuple[list[Any], str, float]:
        """Resolve a TARGET GENE → ranked localized signaling pathways.

        Shared gene→pathway core (semantic-layers redesign): the target's
        gene symbol drives a Reactome lookup, context-re-ranked and
        GO-augmented. This is the MECHANISM layer's identity source (the
        localized molecular signaling pathway the target drives — RAS/MAPK,
        "Co-inhibition by PD-1", …) and also the legacy no-description biology
        fallback. The Reactome/LINCS/QuickGO logic lives ONCE here;
        :meth:`_resolve_real_biology` (biology fallback) and
        :meth:`_populate_trial_mechanisms` (mechanism identity) both call it.

        ``context_name`` is the chain context used for re-ranking — the
        mechanism category slug works as well here as it did when this was
        biology-internal (it expands via ``MECHANISM_PATHWAY_TOKENS``).

        Returns ``(pathways, chosen_source, primary_score)``:
          * ``pathways`` — Reactome ``ReactomePathway`` (or QuickGO ``GOTerm``)
            objects ranked best-first; empty when target is UNKNOWN / non-ENSG,
            the gene symbol is missing, or Reactome+GO return nothing.
          * ``chosen_source`` — ``"reactome_target_lookup"`` or
            ``"quickgo_target_lookup"``.
          * ``primary_score`` — context-overlap score of the top pathway.
        """
        # Reactome lookup via the target's gene symbol. Reactome is the
        # source of truth for "which pathways contain this protein";
        # LINCSClient already wraps the call + on-disk cache.
        if (
            target_id == _UNKNOWN
            or lincs_client is None
            or not target_id.startswith("ENSG")
        ):
            return ([], "reactome_target_lookup", 0.0)

        try:
            target_node = self.graph.get_node(target_id)
        except KeyError:
            return ([], "reactome_target_lookup", 0.0)
        gene_symbol = target_node.get("gene_symbol") or ""
        if not gene_symbol:
            return ([], "reactome_target_lookup", 0.0)

        try:
            pathways = await lincs_client.get_pathways_for_gene(gene_symbol)
        except Exception:
            logger.debug(
                "Reactome lookup failed for %s (target %s)",
                gene_symbol, target_id, exc_info=True,
            )
            pathways = []

        if not pathways:
            return ([], "reactome_target_lookup", 0.0)

        # Re-rank by relevance to the chain context (mechanism + indication
        # + gene symbol) before slicing the top-N. Reactome's default order
        # is citation-frequency-driven and routinely puts off-context
        # pathways first (CRBN → SARS therapeutics, VEGFA → platelet
        # degranulation).
        from src.graph.pathway_ranker import rerank_pathways, score_candidate

        pathways = rerank_pathways(
            pathways,
            mechanism_name=context_name,
            indication_name=indication_id,
            gene_symbol=gene_symbol,
        )

        # GO augmentation: always query QuickGO for the gene's BP annotations
        # and let the scoring comparison decide whether Reactome's best
        # candidate or GO's best is more context-relevant. With strict ``>``
        # comparison, ties keep Reactome (the more heavily curated source by
        # default); GO only wins on a real improvement. QuickGO results are
        # cached on disk, so the per-gene cost is one HTTP call on first build.
        chosen_source = "reactome_target_lookup"
        if quickgo_client is not None:
            top_reactome_score = score_candidate(
                pathways[0],
                mechanism_name=context_name,
                indication_name=indication_id,
                gene_symbol=gene_symbol,
            )
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
                        mechanism_name=context_name,
                        indication_name=indication_id,
                        gene_symbol=gene_symbol,
                    )
                    top_go_score = score_candidate(
                        go_terms[0],
                        mechanism_name=context_name,
                        indication_name=indication_id,
                        gene_symbol=gene_symbol,
                    )
                    if top_go_score > top_reactome_score:
                        # Prune the GO set to its on-mechanism terms. A gene's
                        # full GO biological-process set includes real-but-off-
                        # context leaves (TUBB4B → "flagellated sperm motility",
                        # CRBN → "limb development") and the MECHANISM cap would
                        # otherwise keep all of them. Primary prune is BioLORD
                        # semantic similarity (scales, no hand vocab); token-
                        # overlap is the offline fallback. Top-1 fallback never
                        # drops the chain. Reactome names are left untouched.
                        pathways = self._floor_go_terms(
                            go_terms, context_name, indication_id, gene_symbol
                        )
                        chosen_source = "quickgo_target_lookup"

        primary_score = score_candidate(
            pathways[0],
            mechanism_name=context_name,
            indication_name=indication_id,
            gene_symbol=gene_symbol,
        )
        return (pathways, chosen_source, primary_score)

    def _floor_go_terms(
        self,
        go_terms: list[Any],
        context_name: str,
        indication_id: str,
        gene_symbol: str,
    ) -> list[Any]:
        """Prune a gene's GO biological-process terms to the on-mechanism ones.

        Primary path is BioLORD semantic similarity to the chain context
        (scales to any gene, no hand-maintained vocabulary); falls back to
        token-overlap when BioLORD isn't installed. Either way the top-1
        fallback guarantees the chain keeps a mechanism.
        """
        from src.graph.pathway_ranker import (
            relevance_floor,
            semantic_relevance_floor,
        )

        context_text = (context_name or "").replace("_", " ").strip()
        if context_text:
            try:
                from src.graph.biolord_embeddings import embed_texts

                cache_path = self._cache_dir / "biolord_embeddings.json"
                return semantic_relevance_floor(
                    go_terms,
                    context_text,
                    lambda texts: embed_texts(texts, cache_path=cache_path),
                )
            except Exception:  # noqa: BLE001 — BioLORD optional; degrade gracefully
                logger.debug(
                    "BioLORD GO floor unavailable; using token-overlap fallback",
                    exc_info=True,
                )
        return relevance_floor(
            go_terms,
            mechanism_name=context_name,
            indication_name=indication_id,
            gene_symbol=gene_symbol,
        )

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

    # Number of Reactome pathways materialized as distinct MechanismNodes per
    # (constituent, target). Semantic-layers redesign: a Mechanism node IS the
    # localized molecular signaling pathway the target drives, and a gene sits
    # in MANY pathways — that full footprint (the messy polypharmacology vs the
    # clean clinical hypothesis) is desired data, so unlike biology we keep a
    # GENEROUS cap and fan the chain out one chain per mechanism pathway. The
    # per-pathway chains share the same downstream biology. Cross-target
    # compounding is then automatic: EGFR's MAPK pathway id and BRAF's MAPK
    # pathway id are the SAME Reactome stable_id, so the Tier-1 id merge in
    # node_merge collapses them onto one MechanismNode.
    _MECHANISM_PATHWAY_CAP = 8

    async def _resolve_real_biology(
        self,
        target_id: str,
        mechanism_id: str,
        indication_id: str,
        lincs_client: "LINCSClient | None",
        quickgo_client: "QuickGOClient | None" = None,
    ) -> tuple[list[str], bool]:
        """Resolve a chain's biology to real Reactome pathway(s). LEGACY.

        Semantic-layers redesign: the live ``_populate_trial_biology`` path no
        longer calls this — biology identity is description-only, and the
        target-gene → Reactome resolution moved to the MECHANISM layer
        (``_populate_trial_mechanisms``, which shares the same
        ``_resolve_gene_pathways`` core). Retained for back-compat / tests that
        patch it.

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

        # Gene → ranked pathways via the shared resolver core (Reactome
        # lookup + context re-rank + GO augmentation). Same logic the
        # mechanism layer uses; lives once in ``_resolve_gene_pathways``.
        # ``mechanism_id`` (a category slug on this legacy path) is the
        # re-rank context — it expands via ``MECHANISM_PATHWAY_TOKENS``.
        pathways, chosen_source, primary_score = await self._resolve_gene_pathways(
            target_id, mechanism_id, indication_id, lincs_client,
            quickgo_client=quickgo_client,
        )
        if not pathways:
            return ([], False)

        gene_symbol = self.graph.get_node(target_id).get("gene_symbol") or ""
        # Full pathway list preserved on ``pathway_ids`` so cross-indication
        # queries against the alternates remain answerable. ``reactome_
        # alternatives`` trace is dropped from this legacy biology fallback
        # (the shared resolver returns only the winning source's ranked list).
        all_pathway_ids = [p.stable_id for p in pathways]
        pathway_display_names = {p.stable_id: p.display_name for p in pathways}
        reactome_alternatives_meta: dict[str, str] = {}

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
                # Keep the highest score seen across chains that landed on
                # this node — represents the most context-relevant claim
                # any chain made on this biology.
                existing_score = existing_meta.get("primary_score", 0.0)
                if primary_score > existing_score:
                    existing_meta["primary_score"] = primary_score
            except KeyError:
                node_metadata: dict[str, Any] = {
                    "source": chosen_source,
                    "primary_pathway": bio_id,
                    "primary_score": primary_score,
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
                # Round-25: DATABASE_REACTOME_GO record. Pathway
                # curation is reliable but the "X affects pathway Y" call
                # is always a single curator's interpretation.
                mech_affects_record = make_curated_record(
                    source_id=f"{chosen_source}:{mechanism_id}__{bio_id}",
                    bucket=SupportBucket.MODERATE_SUPPORT,
                    source_type=EvidenceType.DATABASE_REACTOME_GO,
                    notes=(
                        f"mechanism→biology via {chosen_source} "
                        f"(gene {gene_symbol})"
                    ),
                )
                self.graph.add_edge(GraphEdge(
                    source_id=mechanism_id,
                    target_id=bio_id,
                    edge_type=EdgeType.MECHANISM_AFFECTS,
                    belief=belief_from_curated_record(mech_affects_record),
                    metadata={
                        "source": chosen_source,
                        "gene_symbol": gene_symbol,
                    },
                ))

            if not self.graph._graph.has_edge(  # noqa: SLF001
                bio_id, indication_id, key=EdgeType.BIOLOGY_DRIVES.value,
            ):
                # Round-25: when borrowing the prior from a target →
                # indication BIOLOGY_DRIVES edge, ALSO copy the evidence
                # list so the new biology → indication edge carries the
                # curated-DB provenance. Otherwise the new edge would
                # have non-zero alpha/beta but an empty evidence list,
                # which the new prediction engine would treat as
                # zero-evidence (drop) — losing the OT association
                # signal we just wired in upstream.
                borrowed = EdgeBeliefState()
                try:
                    borrowed = self.graph.get_edge_belief(
                        target_id, indication_id, EdgeType.BIOLOGY_DRIVES,
                    )
                except KeyError:
                    pass
                self.graph.add_edge(GraphEdge(
                    source_id=bio_id,
                    target_id=indication_id,
                    edge_type=EdgeType.BIOLOGY_DRIVES,
                    belief=EdgeBeliefState(
                        alpha=borrowed.alpha, beta=borrowed.beta,
                        evidence=list(borrowed.evidence),
                    ),
                    metadata={
                        "source": "reactome_target_lookup",
                        "borrowed_from": f"{target_id}->{indication_id}",
                    },
                ))
            biology_ids.append(bio_id)

        return (biology_ids, False)

    async def _populate_trial_biology(
        self,
        trials: list[TrialRecord],
        annotations_dir: str | Path | None = None,
    ) -> int:
        """Give every chain's Biology node its identity.

        Semantic-layers redesign — a Biology node is the GENERAL physiological
        process the drug drives (e.g. "T-cell mediated anti-tumor immunity"),
        SHARED across every drug/disease that drives it. Its identity is
        DESCRIPTION-ONLY: it comes from the extracted ``biology_description``,
        NEVER from the target-gene → Reactome pathway. That resolution sits at
        the MECHANISM layer now (``_populate_trial_mechanisms``) and must not
        occupy the biology slot (a PD-1 inhibitor's biology is the
        anti-tumor-immunity process, not the Reactome pathway "Co-inhibition by
        PD-1"). Per-chain identity:

          1. ``biology_description`` present (the build extracted before populate,
             ``annotations_dir`` threaded) → CONTENT-ADDRESS the description
             (``bio:<sha1>``), with the description as the node name/description.
             Reactome is NEVER consulted. Cross-trial consolidation is the
             downstream BioLORD description merge (Tier-3 geometric merge of
             near-identical phrasings; identical text collapses exactly).
          2. No description (legacy / no-annotations path, or the extractor
             emitted none) → biology is left UNRESOLVED (``biology_id`` =
             ``UNKNOWN``, tagged ``metadata["unresolved_biology"] = True`` for
             downstream auditing). NO Reactome fallback, NO
             ``{mechanism}__{indication}`` slug.

        Returns the count of new biology nodes + chain rewrites combined.
        """
        from src.graph.models import BiologyNode  # local import

        # One LINCSClient per pipeline run, used only by the end-of-pass
        # ``merge_equivalent_biology_nodes`` sweep (Reactome↔GO crosswalk).
        # Biology identity no longer consults Reactome, so this is purely for
        # the merge step; tolerate a missing CLUE_API_KEY (the crosswalk
        # endpoints are keyless).
        lincs_client: "LINCSClient | None"
        try:
            lincs_client = LINCSClient()
        except Exception:
            logger.debug("LINCSClient unavailable", exc_info=True)
            lincs_client = None

        # Semantic-layers redesign: when the build has already extracted
        # (annotations_dir set), a chain's Biology node takes its identity from
        # the trial's extracted biology description. Resolution order, most→
        # least specific:
        #   1. per-(arm, intervention) — the description for THIS drug's chain
        #      (combo-arm biology fix: distinct drugs in one arm get distinct
        #      biology, so their resolved Reactome/GO nodes don't fuse).
        #   2. per-arm (A.0b) — legacy/mono-arm path, shared across the arm.
        #   3. trial-level intended_biology (A.0) — last-resort fallback.
        desc_by_arm: dict = {}
        desc_by_arm_iv: dict = {}
        intended_bio_by_nct: dict[str, str] = {}
        if annotations_dir is not None:
            from src.graph.box_embeddings import (
                chain_descriptions_by_arm,
                chain_descriptions_by_arm_intervention,
            )
            desc_by_arm = chain_descriptions_by_arm(annotations_dir)
            desc_by_arm_iv = chain_descriptions_by_arm_intervention(annotations_dir)
            for _p in sorted(Path(annotations_dir).glob("*_extraction.json")):
                try:
                    _d = json.loads(_p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                _nct = _d.get("nct_id")
                if _nct:
                    intended_bio_by_nct[_nct] = (
                        (_d.get("therapeutic_hypothesis") or {}).get(
                            "intended_biology"
                        ) or ""
                    ).strip()

        added = 0
        for trial in trials:
            try:
                ts = self.graph.get_trial_subgraph_by_id(trial.nct_id)
            except KeyError:
                continue
            if not ts.chains:
                continue

            new_chains: list[CausalChain] = []
            for chain in ts.chains:
                mech_id = chain.mechanism_id
                ind_id = chain.indication_id
                target_id = chain.target_id
                if mech_id == _UNKNOWN or ind_id == _UNKNOWN:
                    new_chains.append(chain)
                    continue

                # The chain's extracted biology description (semantic-layers
                # redesign). Present only when the build extracted before
                # populate (annotations_dir threaded). Resolution order, most→
                # least specific:
                #   1. per-(arm, intervention) for THIS chain's compound — the
                #      description for THIS drug's chain (combo-arm fix: distinct
                #      drugs in one arm get distinct biology),
                #   2. per-arm (A.0b),
                #   3. trial-level intended_biology (A.0).
                # The description becomes the biology node IDENTITY below, so two
                # constituents with distinct descriptions get distinct content-
                # address ids — the combo-arm fusion that an earlier "only stamp a
                # compound-specific description" guard protected against can no
                # longer occur.
                bio_desc = ""
                if annotations_dir is not None:
                    iv_entry = _lookup_chain_intervention_desc(
                        self.graph, desc_by_arm_iv, trial.nct_id, chain,
                    )
                    iv_bio = (iv_entry.get("biology") or "").strip() if iv_entry else ""
                    if iv_bio:
                        bio_desc = iv_bio
                    else:
                        bio_desc = (
                            (desc_by_arm.get((trial.nct_id, chain.arm_id)) or {}).get(
                                "biology"
                            )
                            or intended_bio_by_nct.get(trial.nct_id, "")
                        ).strip()

                # Semantic-layers redesign — biology identity is DESCRIPTION-ONLY.
                # A Biology node IS the general physiological PROCESS the drug
                # drives (shared across every drug/disease that drives it), so its
                # identity comes purely from the extracted PHYSIOLOGICAL-PROCESS
                # description. The target-gene → Reactome resolution lives at the
                # MECHANISM layer now (``_populate_trial_mechanisms``) and must NOT
                # occupy the biology slot — so biology NO LONGER consults Reactome
                # at all. When a chain has no extracted biology description, leave
                # biology UNRESOLVED (``_UNKNOWN``, tagged for audit) rather than
                # synthesizing a {mech}__{indication} slug. Cross-trial
                # consolidation is the downstream BioLORD description merge.
                if not bio_desc:
                    md = dict(chain.metadata)
                    md["unresolved_biology"] = True
                    new_chains.append(chain.model_copy(update={
                        "biology_id": _UNKNOWN,
                        "metadata": md,
                    }))
                    continue

                # Description-identity: the Biology node IS its extracted
                # general-process description — a content-address handle, with the
                # description set so the BioLORD merge pools equivalent biology
                # across trials and accumulates evidence on one node.
                bio_node_id = _biology_id_from_description(bio_desc)
                bio_node_name = bio_desc[:120]
                edge_source = "trial_biology_description"
                try:
                    self.graph.get_node(bio_node_id)
                except KeyError:
                    self.graph.add_node(BiologyNode(
                        id=bio_node_id,
                        name=bio_node_name,
                        description=bio_desc,
                        pathway_ids=[],
                        metadata={"source": "trial_biology_description"},
                    ))
                    added += 1

                if not self.graph._graph.has_edge(  # noqa: SLF001
                    mech_id, bio_node_id,
                    key=EdgeType.MECHANISM_AFFECTS.value,
                ):
                    # DATABASE_FALLBACK record: the populate-stage prior is
                    # intentionally weak. For description-identity biology the
                    # EVIDENCE comes from the Phase-2 geometric merge pooling many
                    # trials' records onto one node — not from a higher per-record
                    # n_eff (belief seeding is kept identical to isolate the
                    # identity change).
                    fallback_record = make_curated_record(
                        source_id=f"{edge_source}:{mech_id}__{bio_node_id}",
                        bucket=SupportBucket.WEAK_SUPPORT,
                        source_type=EvidenceType.DATABASE_FALLBACK,
                        notes=f"description-defined biology ({mech_id})",
                    )
                    self.graph.add_edge(GraphEdge(
                        source_id=mech_id,
                        target_id=bio_node_id,
                        edge_type=EdgeType.MECHANISM_AFFECTS,
                        belief=belief_from_curated_record(fallback_record),
                        metadata={"source": edge_source},
                    ))

                if not self.graph._graph.has_edge(  # noqa: SLF001
                    bio_node_id, ind_id,
                    key=EdgeType.BIOLOGY_DRIVES.value,
                ):
                    # Round-25: borrow the target → indication BIOLOGY_DRIVES
                    # prior (with its evidence list) so the new biology →
                    # indication edge carries the curated-DB provenance.
                    borrowed = EdgeBeliefState()
                    if target_id != _UNKNOWN:
                        try:
                            borrowed = self.graph.get_edge_belief(
                                target_id, ind_id,
                                EdgeType.BIOLOGY_DRIVES,
                            )
                        except KeyError:
                            pass
                    self.graph.add_edge(GraphEdge(
                        source_id=bio_node_id,
                        target_id=ind_id,
                        edge_type=EdgeType.BIOLOGY_DRIVES,
                        belief=EdgeBeliefState(
                            alpha=borrowed.alpha,
                            beta=borrowed.beta,
                            evidence=list(borrowed.evidence),
                        ),
                        metadata={
                            "source": edge_source,
                            "borrowed_from": (
                                f"{target_id}->{ind_id}"
                                if target_id != _UNKNOWN else None
                            ),
                        },
                    ))

                md = dict(chain.metadata)
                md.pop("unresolved_biology", None)
                new_chains.append(chain.model_copy(update={
                    "biology_id": bio_node_id,
                    "metadata": md,
                }))

            if new_chains != list(ts.chains):
                self.graph.set_trial_subgraph(
                    ts.model_copy(update={"chains": new_chains})
                )

        # After every chain has its biology assigned, sweep the graph for
        # equivalent BiologyNodes (Reactome and GO terms that describe the
        # same biology per Reactome's own curator-annotated crosswalk) and
        # collapse them so evidence pools across the merged node instead
        # of fragmenting across semantic twins. Idempotent on rebuild.
        from src.graph.biology_merge import merge_equivalent_biology_nodes

        if lincs_client is not None:
            try:
                merge_report = await merge_equivalent_biology_nodes(
                    self.graph, lincs_client,
                )
                if merge_report.nodes_collapsed:
                    logger.info(
                        "Biology merge: collapsed %d nodes across %d "
                        "equivalence classes; %d chains updated",
                        merge_report.nodes_collapsed,
                        merge_report.classes_merged,
                        merge_report.chains_updated,
                    )
            except Exception:
                logger.warning(
                    "Biology merge step failed; graph is unchanged",
                    exc_info=True,
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
                    # Endpoints are DISEASE-AGNOSTIC (node-orthogonality
                    # redesign): a single shared PFS / OS / ORR node per
                    # measurement, with the per-disease binding on the
                    # endpoint_captures edge seeded below. Catch-all classes
                    # (biomarker/safety/PRO/other) are sub-keyed by measure so
                    # distinct measures don't fuse.
                    ep_id, ep_label = _endpoint_node_id(cls, om.measure)
                    try:
                        self.graph.get_node(ep_id)
                    except KeyError:
                        self.graph.add_node(EndpointNode(
                            id=ep_id,
                            name=ep_label,
                            endpoint_type=EndpointType.PRIMARY,
                            regulatory_status=RegulatoryStatus.EXPLORATORY,
                            measurement_properties={
                                "endpoint_class": cls.value,
                                "raw_measure": om.measure,
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

    def _add_curated_target_edges(
        self,
        trials: list[TrialRecord],
        table: list[tuple[str, list[tuple[str, str, str]]]],
        source_tag: str,
    ) -> tuple[int, dict[str, list[str]]]:
        """Wire AFFECTS edges from a curated compound→target mapping.

        Generic over the table — the vaccine and cell-therapy heuristics
        both use this. Each table is a list of (substring pattern,
        [(ENSG, gene_symbol, name)]) tuples; both pattern and target
        name are normalized via ``_norm_for_pattern_match`` (hyphens,
        underscores, and runs of whitespace all collapse to a single
        space) before substring match. One pattern "anti ny eso 1"
        covers "anti-ny-eso-1", "Anti-NY ESO-1", "anti_ny_eso_1", etc.

        Joined regimen strings are decomposed via
        ``canonical_compound_names`` before matching — same fix as the
        OT-lookup loop.
        """
        added = 0
        seen_compounds: set[str] = set()
        compound_targets: dict[str, list[str]] = {}
        normalized_table = [
            (_norm_for_pattern_match(pat), targets) for pat, targets in table
        ]
        for trial in trials:
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                for canonical_name in canonical_compound_names(iv.name):
                    cid = self.resolve_entity(canonical_name, "compound")
                    if not cid or cid in seen_compounds:
                        continue
                    normalized_name = _norm_for_pattern_match(canonical_name)
                    matched_targets: dict[str, tuple[str, str]] = {}
                    for pattern, targets in normalized_table:
                        if pattern in normalized_name:
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
                                metadata={"source": source_tag},
                            ))
                            self._index_node(ens, symbol, "target")
                        if self.graph._graph.has_edge(  # noqa: SLF001
                            cid, ens, key=EdgeType.AFFECTS.value,
                        ):
                            continue
                        # Round-25: cell-therapy / pattern-matched
                        # AFFECTS edge. Hand-curated pattern table,
                        # same tier as the mAb table.
                        pattern_record = make_curated_record(
                            source_id=f"{source_tag}:{cid}__{ens}",
                            bucket=SupportBucket.STRONG_SUPPORT,
                            source_type=EvidenceType.DATABASE_MAB_TABLE,
                            notes=(
                                f"pattern-matched target: "
                                f"{canonical_name} → {symbol}"
                            ),
                        )
                        self.graph.add_edge(GraphEdge(
                            source_id=cid,
                            target_id=ens,
                            edge_type=EdgeType.AFFECTS,
                            belief=belief_from_curated_record(pattern_record),
                            metadata={
                                "source": source_tag,
                                "pattern_matched": canonical_name,
                            },
                        ))
                        added += 1
        return added, compound_targets

    def _add_vaccine_component_target_edges(
        self, trials: list[TrialRecord],
    ) -> tuple[int, dict[str, list[str]]]:
        """Wire AFFECTS edges for curated cancer-vaccine constituent targets.

        Open Targets returns no targets for tumor-associated-antigen
        peptides (gp100, tyrosinase, MART-1), depot adjuvants (Montanide,
        IFA), TLR-agonist adjuvants (poly-ICLC, CpG, MPL), or CD4 helper
        peptides (PADRE, tetanus, melanoma helper peptide).
        ``_VACCINE_COMPONENT_TARGETS`` is the curated mapping that fills
        in those binding partners so chains for these compounds resolve
        past UNKNOWN_target.
        """
        return self._add_curated_target_edges(
            trials,
            _VACCINE_COMPONENT_TARGETS,
            "vaccine_component_heuristic",
        )

    def _add_cell_therapy_target_edges(
        self, trials: list[TrialRecord],
    ) -> tuple[int, dict[str, list[str]]]:
        """Wire AFFECTS edges for adoptive-cell-therapy and code-named drugs.

        TCR-engineered T cells, anti-tumor-antigen TILs, and a handful
        of code-named small molecules (PS-341, UCN-01) that OT can't
        resolve. Each cell-therapy product's target is the antigen its
        engineered receptor recognizes — modeled as the antigen's
        canonical gene (CTAG1B for NY-ESO-1, PMEL for gp100, etc.).
        """
        return self._add_curated_target_edges(
            trials,
            _CELL_THERAPY_COMPONENT_TARGETS,
            "cell_therapy_heuristic",
        )

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
                # Round-27 fix: gate the cross-reference heuristic on
                # whether the compound has a chembl-resolved canonical
                # form. The round-26 bevacizumab AVANT regression was
                # caused by THIS heuristic firing on the unresolved
                # `bevacizumab_7_5_mg_kg` slug — its title contained
                # "metastatic" which substring-matched "MET" gene_symbol,
                # creating a phantom AFFECTS edge. Same root cause
                # behind the checkpoint→APP leak (titles containing
                # "application" / "approach" matched "APP"). For
                # compounds OT/ChEMBL resolved, the AFFECTS edges are
                # already populated via the canonical OT-direct path;
                # this fallback only ever fired as a last-resort name-
                # match, so gating it doesn't lose real signal.
                try:
                    comp_node = self.graph.get_node(comp_id)
                except KeyError:
                    continue
                if not comp_node.get("chembl_id"):
                    continue
                # Try to find a target whose name/symbol appears in the
                # trial title or intervention description. Round-27:
                # tightened symbol min-length 3 → 4 (3-letter symbols
                # like "APP", "MET", "AKT", "PIK", "RAF" alias too many
                # English words to safely substring-match) AND use word
                # boundaries (\bSYMBOL\b) so "met" in "metastatic" can't
                # match MET. Long-name path keeps the 5-char minimum
                # but also tightens to word-boundary match.
                text = f"{trial.title} {iv.description}".lower()
                for target_id in targets_in_graph:
                    try:
                        tdata = self.graph.get_node(target_id)
                    except KeyError:
                        continue
                    symbol = tdata.get("gene_symbol", "").lower()
                    name = tdata.get("name", "").lower()
                    if (
                        symbol and len(symbol) >= 4
                        and re.search(rf"\b{re.escape(symbol)}\b", text)
                    ):
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
                    elif (
                        name and len(name) >= 5
                        and re.search(rf"\b{re.escape(name)}\b", text)
                    ):
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
        return added

    def _add_binds_edge(self, compound_id: str, target_id: str) -> None:
        # Round-25: DATABASE_CROSS_REFERENCE record. Lowest tier — name
        # overlap isn't really curation, just a heuristic. n_eff=0.3.
        record = make_curated_record(
            source_id=f"cross_reference:{compound_id}__{target_id}",
            bucket=SupportBucket.WEAK_SUPPORT,
            source_type=EvidenceType.DATABASE_CROSS_REFERENCE,
            notes=(
                f"name-match AFFECTS: {compound_id} → {target_id} via "
                f"intervention-text overlap"
            ),
        )
        edge = GraphEdge(
            source_id=compound_id,
            target_id=target_id,
            edge_type=EdgeType.AFFECTS,
            belief=belief_from_curated_record(record),
            metadata={"source": "cross_reference", "method": "name_matching"},
        )
        self.graph.add_edge(edge)

    # ── Trial subgraph construction ──────────────────────────────────────

    # Round-21 Phase B: compound canonicalization threshold. Calibrated
    # against a labeled validation set run through live SapBERT
    # (2026-05-20, see scripts/verify_sapbert.py).
    #
    # Findings on the n=11 set:
    #   - SAME spelling/salt-form variants:           0.89-0.97 cosine
    #   - SAME brand/INN pairs (Keytruda/Pembro):     ~0.77
    #   - SAME codenames (MK-3475/Pembro):            ~0.33 — SapBERT
    #     CANNOT resolve codenames; CODENAME_TO_INN +
    #     OT alias lookups handle them instead (round 23 will add
    #     RxNorm + PubChem fallbacks for more coverage)
    #   - SIMILAR_CLASS_DISTINCT (erlotinib/gefitinib,
    #     paclitaxel/docetaxel, nivolumab/pembro):    0.55-0.72
    #   - DIFFERENT class pairs:                      0.25-0.31
    #
    # Threshold 0.80 sits in the safety window between max
    # SIMILAR_CLASS_DISTINCT (0.72) and SAME spelling-variants (0.89+).
    # It catches the dominant target case (spelling + salt forms),
    # narrowly misses brand-name pairs at ~0.77 (those flow through
    # OT aliases instead), and cannot accidentally merge any of the
    # observed class-mate pairs.
    #
    # The chembl_id non-conflict gate in _canonicalize_compound is the
    # real safety net: even at a more aggressive threshold, two
    # compounds with different ChEMBL ids cannot merge.
    _COMPOUND_EMBEDDING_SIMILARITY_THRESHOLD = 0.80

    def _canonicalize_compound(
        self,
        comp: "InterventionNode",
        existing_chembl_index: dict[str, str],
        embedding_index: list[tuple[str, list[float]]],
    ) -> tuple[str, bool]:
        """Decide whether ``comp`` should reuse an existing compound id.

        Resolution priority (most-specific to least):
          1. ``stable_id`` (ChEMBL id) match — chembl is authoritative.
             If incoming's chembl_id matches an existing node's
             ``stable_id``, reuse that id.
          2. Embedding cosine match above
             ``_COMPOUND_EMBEDDING_SIMILARITY_THRESHOLD`` — gated by
             chembl_id NON-CONFLICT. If incoming has chembl_id X and
             existing has chembl_id Y ≠ X, the embedding match is
             rejected (safety net against false-merge between two
             distinct ChEMBL-registered drugs that happen to embed
             close).
          3. No match — caller adds a fresh node.

        Returns ``(canonical_id, was_resolved)``. ``was_resolved=True``
        means the caller should skip ``add_node`` and merge the
        incoming compound's aliases into the existing node.

        ``existing_chembl_index`` is ``chembl_id → existing_compound_id``,
        ``embedding_index`` is ``[(existing_id, embedding_vector), ...]``.
        Both built once at populate_trials start.
        """
        # 1. stable_id (chembl_id) match.
        incoming_chembl = comp.stable_id
        if incoming_chembl and incoming_chembl in existing_chembl_index:
            return existing_chembl_index[incoming_chembl], True

        # 2. Embedding match — only if we have one for the incoming.
        if comp.embedding is None or not embedding_index:
            return comp.id, False

        from src.graph.sapbert_embeddings import cosine_similarity

        best_id: str | None = None
        best_score = -1.0
        for existing_id, vec in embedding_index:
            score = cosine_similarity(comp.embedding, vec)
            if score > best_score:
                best_score = score
                best_id = existing_id
        if (
            best_id is not None
            and best_score >= self._COMPOUND_EMBEDDING_SIMILARITY_THRESHOLD
        ):
            # Chembl-id non-conflict gate. If both incoming and the
            # candidate have a chembl_id and they DIFFER, refuse the
            # embedding match — it's a false positive across distinct
            # ChEMBL-registered drugs.
            candidate_chembl = self.graph._graph.nodes[best_id].get(  # noqa: SLF001
                "stable_id"
            )
            if (
                incoming_chembl
                and candidate_chembl
                and incoming_chembl != candidate_chembl
            ):
                return comp.id, False
            return best_id, True

        return comp.id, False

    def _rehydrate_compound_embedding(
        self, node: dict[str, Any]
    ) -> list[float] | None:
        """Recompute a compound node's SapBERT vector from the on-disk cache.

        Public snapshots are stripped of embeddings by the artifact boundary
        (``src/boundary.py``), so on an incremental ``--base-snapshot`` load
        the existing InterventionNodes carry no ``embedding``. The vector is a
        deterministic function of the compound name + the (public) SapBERT
        model, cached at ``data/cache/sapbert_embeddings.json`` — so we
        re-derive it by name. That's a cache hit on any machine that built the
        snapshot, and a graceful ``None`` (chembl + alias canonicalization
        only) when the ``sapbert`` extra isn't installed or the name is empty.
        """
        name = node.get("name") or node.get("id")
        if not name:
            return None
        try:
            from src.graph.sapbert_embeddings import (
                SapBertUnavailable,
                embed_compound_name,
            )
            try:
                return embed_compound_name(
                    name,
                    cache_path=self._cache_dir / "sapbert_embeddings.json",
                )
            except SapBertUnavailable:
                return None
        except ImportError:
            return None

    def _merge_compound_into_existing(
        self, incoming: "InterventionNode", existing_id: str,
    ) -> None:
        """Union the incoming compound's identifying info into an
        already-present node (Phase B canonicalization decided they're
        the same drug).

        - Adds incoming's name + id + aliases to the existing node's
          aliases (deduped, order-preserving).
        - Indexes incoming's name + aliases → existing_id in the
          entity index, so subsequent ``resolve_entity`` calls from
          extractor / classifier / build_arms find the existing id.
        - Backfills ``stable_id`` on the existing node only if it's
          currently empty (chembl is authoritative once set).
        - Backfills ``embedding`` similarly.
        """
        node_dict = self.graph._graph.nodes[existing_id]  # noqa: SLF001
        existing_aliases = list(node_dict.get("aliases") or [])
        seen = set(existing_aliases)
        candidates = [incoming.name, incoming.id] + list(incoming.aliases)
        for alias in candidates:
            if alias and alias not in seen and alias != existing_id:
                existing_aliases.append(alias)
                seen.add(alias)
        node_dict["aliases"] = existing_aliases

        if not node_dict.get("stable_id") and incoming.stable_id:
            node_dict["stable_id"] = incoming.stable_id
        if not node_dict.get("embedding") and incoming.embedding:
            node_dict["embedding"] = list(incoming.embedding)

        # Update the entity index so downstream resolve_entity calls
        # (build_arms, classifier prompts, etc.) map any of the
        # incoming compound's names onto the canonical existing id.
        for alias in candidates:
            if alias:
                self._index_node(existing_id, alias, "compound")

    def _slug_create_indication_node(self, condition: str) -> str:
        """Round-22 fallback: create an IndicationNode from a raw CT.gov
        condition string when LLM canonicalization didn't produce a
        canonical mapping and the entity index has no match.

        The slug-derived id will be redundant with whatever canonical
        node a future canonicalization run produces for the same
        condition — that redundancy is acceptable and addressable via
        a later dedup pass (analogous to the round-21 SapBERT compound
        consolidation). What's NOT acceptable is silently dropping
        therapeutic trials because their condition wasn't pre-classified.

        Indexes the raw + slug forms so subsequent trials with the same
        wording resolve to this node instead of slug-creating another.
        """
        slug_id = normalize_entity(condition, "IndicationNode")
        try:
            self.graph.get_node(slug_id)
        except KeyError:
            self.graph.add_node(IndicationNode(
                id=slug_id,
                name=condition,
                metadata={
                    "source": "slug_fallback_no_canonicalization",
                    "observed_variants": [condition],
                },
            ))
        self._index_node(slug_id, slug_id, "indication")
        self._index_node(slug_id, condition, "indication")
        return slug_id

    def _slug_create_endpoint_node(
        self, measure: str, indication_id: str,
    ) -> str:
        """Round-22 fallback for endpoints. Same philosophy as
        ``_slug_create_indication_node``: when LLM endpoint classification
        didn't match a canonical EndpointClass for this measure, create
        a measure-keyed node so the trial still gets a chain instead of
        being silently dropped.

        Id follows ``other_{measure_slug}`` — disease-agnostic (node-
        orthogonality redesign); the per-disease binding is on the
        endpoint_captures edge. ``indication_id`` is accepted for call-site
        compatibility but no longer part of the id.
        """
        del indication_id  # endpoints are disease-agnostic now
        measure_slug = slugify_measure_name(measure, max_len=60)
        ep_id = f"other_{measure_slug}"
        try:
            self.graph.get_node(ep_id)
        except KeyError:
            self.graph.add_node(EndpointNode(
                id=ep_id,
                name=measure,
                endpoint_type=EndpointType.PRIMARY,
                regulatory_status=RegulatoryStatus.EXPLORATORY,
                measurement_properties={
                    "source": "slug_fallback_no_canonicalization",
                    "raw_measure": measure,
                },
            ))
        self._index_node(ep_id, measure, "endpoint")
        return ep_id

    def build_trial_subgraphs(
        self, trials: list[TrialRecord]
    ) -> list[TrialSubgraph]:
        """Build skeleton TrialSubgraphs (TrialNode + arms + parent population).

        Produces one chain per arm at the trial's qualified default
        PopulationNode (see step 2 of ``populate_trials``). For a trial
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
            # Round-22: slug-create on miss. If LLM canonicalization +
            # entity index lookup both failed for the trial's
            # conditions, fall back to creating a slug-keyed
            # IndicationNode from the first condition. Mirrors the
            # UNKNOWN-allowed pattern for target/mechanism/biology
            # (which never drop trials when they can't resolve). The
            # only legitimate drop is conditions=[] entirely.
            if not indication_id:
                if not trial.conditions:
                    _log_dropped_trial_subgraph(
                        trial.nct_id, "no_conditions",
                        conditions=[],
                    )
                    continue
                indication_id = self._slug_create_indication_node(
                    trial.conditions[0]
                )

            endpoint_id = None
            attempted_measures: list[str] = []
            for om in trial.primary_outcomes:
                attempted_measures.append(om.measure)
                endpoint_id = self.resolve_entity(om.measure, "endpoint")
                if endpoint_id:
                    break
            # Round-22: slug-create on miss. Same pattern as indication
            # fallback above. Only legitimate drop is no primary
            # outcomes at all.
            if not endpoint_id:
                if not trial.primary_outcomes:
                    _log_dropped_trial_subgraph(
                        trial.nct_id, "no_primary_outcomes",
                        indication_id=indication_id,
                    )
                    continue
                endpoint_id = self._slug_create_endpoint_node(
                    trial.primary_outcomes[0].measure, indication_id,
                )

            seed_trial_node(self.graph, trial)
            arms = build_arms(trial, self.resolve_entity)
            # Drop diagnostic compounds (Lymphoseek-class) identified in
            # step 1.5. If an arm's compound list empties, drop the arm;
            # if all arms drop, the trial gets no chains (existing
            # downstream `if not arms: continue` handles trial drop).
            diag_cids = getattr(self, "_diagnostic_compound_ids", set())
            if diag_cids:
                for arm in arms:
                    arm.compound_ids = [
                        c for c in arm.compound_ids
                        if c not in diag_cids
                    ]
                arms = [a for a in arms if a.compound_ids]
            if not arms:
                # Distinguish empty arm_groups (CT.gov data gap, fixable
                # via the interventions-fallback in build_arms) from
                # everything-filtered (all compounds were diagnostic) so
                # the log surfaces the root cause.
                if not trial.arm_groups:
                    drop_reason = "no_arms_empty_arm_groups"
                elif diag_cids and any(
                    set(ag.intervention_names) <= {
                        iv.name for iv in trial.interventions
                        if iv.name and iv.name not in diag_cids
                    }
                    for ag in trial.arm_groups
                ):
                    drop_reason = "no_arms_filtered_by_diagnostic"
                else:
                    drop_reason = "no_arms"
                _log_dropped_trial_subgraph(
                    trial.nct_id, drop_reason,
                    indication_id=indication_id,
                    endpoint_id=endpoint_id,
                    arm_group_count=len(trial.arm_groups),
                    intervention_names=[iv.name for iv in trial.interventions],
                )
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
                    # Verbatim sponsor-stated stop reason (CT.gov whyStopped),
                    # RAW text only. Mirrors TrialNode.metadata so the trial's
                    # own words persist into the subgraph snapshot too.
                    "why_stopped": trial.why_stopped,
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
            # Verbatim sponsor-stated stop/failure reason (CT.gov
            # whyStopped). RAW text only — not a classified judgment.
            # Persisted on the node so the graph snapshot carries the
            # trial's own words for why it stopped (often the only
            # mechanistic signal for terminated trials with no results).
            # None when CT.gov omitted it.
            "why_stopped": trial.why_stopped,
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

    # Round-20.5 / 22: arm_groups fallback. Triggers when EITHER:
    #   (a) trial.arm_groups is empty (NCT00134264 torcetrapib, older
    #       terminated trials), OR
    #   (b) every arm_group has intervention_names=[] (NCT00329160
    #       rosuvastatin — CT.gov v2 returned arm_groups + drug
    #       interventions but the linkage between them was missing)
    # In both cases the trial has real drug-like interventions but
    # build_arms produces nothing without the fallback.
    has_any_arm_intervention_link = any(
        ag.intervention_names for ag in trial.arm_groups
    )
    arm_groups_list = (
        list(trial.arm_groups) if has_any_arm_intervention_link else []
    )
    if not arm_groups_list and trial.interventions:
        from src.ingestion.clinicaltrials import ArmGroup
        drug_interventions = [
            iv for iv in trial.interventions
            if iv.name and is_drug_like(iv)
        ]
        for iv in drug_interventions:
            arm_groups_list.append(ArmGroup(
                group_id=normalize_entity(iv.name, "InterventionNode"),
                title=iv.name,
                description=(
                    "synthesized from trial.interventions because CT.gov "
                    "returned empty arm_groups"
                ),
                intervention_names=[iv.name],
            ))

    arms: list[TrialArm] = []
    seen_arm_ids: set[str] = set()
    for ag in arm_groups_list:
        if not ag.intervention_names:
            continue
        relevant_names = [
            n for n in ag.intervention_names if n not in non_drug_names
        ]
        if not relevant_names:
            continue
        compound_ids: list[str] = []
        for iv_name in relevant_names:
            # Expand combo regimens ("Carboplatin/Paclitaxel", "Sora + DTIC")
            # and strip parenthetical brand/code aliases
            # ("Sorafenib (Nexavar, BAY43-9006)") BEFORE canonicalization so
            # cross-trial learning lands on a single CompoundNode per drug.
            # Same helper is used by `map_trial_to_graph_nodes` so both
            # populator code paths agree on the canonical name.
            for canonical in canonical_compound_names(iv_name):
                cid = (
                    resolve_compound(canonical, "compound")
                    if resolve_compound is not None
                    else None
                )
                if not cid:
                    cid = normalize_entity(canonical, "InterventionNode")
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
                # Round-25: DATABASE_FALLBACK record. The combo-arm
                # inherits AFFECTS via a constituent — one inferential
                # step removed from the constituent's curated source,
                # so half the n_eff.
                inherit_record = make_curated_record(
                    source_id=(
                        f"combo_inherit:"
                        f"{arm.regimen_compound_id}__{target_id}"
                    ),
                    bucket=SupportBucket.MODERATE_SUPPORT,
                    source_type=EvidenceType.DATABASE_FALLBACK,
                    notes=f"inherited AFFECTS via constituent {cid}",
                )
                graph.add_edge(GraphEdge(
                    source_id=arm.regimen_compound_id,
                    target_id=target_id,
                    edge_type=EdgeType.AFFECTS,
                    belief=belief_from_curated_record(inherit_record),
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
    """Return the sentinel for "no parent population node".

    Node-orthogonality redesign: an all-comers enrollment cohort carries no
    information beyond the indication, so it gets NO PopulationNode. Chains
    with no reported subgroup use this sentinel as their
    ``subgroup_population_id`` (and the trial's ``parent_population_id``), which
    makes the prediction walk skip ``responds_differently`` — the right
    semantics for "no patient-selection signal". Real subgroups get their own
    disease-agnostic ``{axes}`` nodes instead. Kept as a function (not inlined)
    so both chain builders stay in sync; ``indication_name`` is now unused.
    """
    del graph, indication_id, indication_name
    return _UNKNOWN


def _lookup_chain_intervention_desc(
    graph: GraphStore,
    desc_by_arm_iv: dict,
    nct_id: str,
    chain: "CausalChain",
) -> dict | None:
    """Find the per-(arm, intervention) descriptions for THIS chain's compound.

    Combo-arm biology fix: each fanned-out chain in a combination arm has its
    own ``compound_id`` (the constituent), and the extractor now emits a
    distinct biology/mechanism description per drug, tagged with an
    ``intervention`` name. This matches the chain to its drug's entry by
    normalizing both the chain's ``compound_id`` and the compound node's
    ``name`` (the LLM names the drug, which canonicalizes to either) against the
    keyed intervention slugs in ``desc_by_arm_iv``. Returns the
    ``{mechanism, biology, population}`` dict, or ``None`` when no per-drug
    description exists (caller falls back to per-arm / trial-level)."""
    from src.graph.box_embeddings import _intervention_key

    if not desc_by_arm_iv:
        return None
    keys = {_intervention_key(chain.compound_id)}
    try:
        name = graph.get_node(chain.compound_id).get("name") or ""
    except KeyError:
        name = ""
    if name:
        keys.add(_intervention_key(name))
    keys.discard("")
    if not keys:
        return None
    # Match this drug to its results_by_chain entry, most→least specific:
    #   (1) exact in THIS arm; (2) exact in ANY arm — a drug's biology is
    #   arm-independent, and a combo arm is often another arm + one extra drug
    #   with only one arm detailed by the extractor; (3) salt-form/stem in any
    #   arm ("fludarabine" ~ "fludarabine_phosphate" — the LLM drops salt suffixes).
    def _stem_eq(a: str, b: str) -> bool:
        return a == b or a.startswith(b + "_") or b.startswith(a + "_")

    for k in keys:
        entry = desc_by_arm_iv.get((nct_id, chain.arm_id, k))
        if entry:
            return entry
    for _match in ((lambda iv, k: iv == k), _stem_eq):
        for k in keys:
            for (_n, _arm, _iv), entry in desc_by_arm_iv.items():
                if _n == nct_id and _match(_iv, k):
                    return entry
    return None


def _set_node_description(
    graph: GraphStore, node_id: str, description: str
) -> bool:
    """Attach a node's free-text description if it doesn't already have one.

    The canonical id is a routing tag; the description is the semantic
    substrate the BioLORD embedding work (A.1) consumes. First non-empty
    contributor wins, so a node shared across trials keeps a stable
    representative description (the per-trial provenance list is a later A.2
    refinement). No-op (returns False) for an absent node, empty text, or a
    node that already has a description.
    """
    if not description:
        return False
    try:
        data = graph._graph.nodes[node_id]  # noqa: SLF001
    except KeyError:
        return False
    if data.get("description"):
        return False
    data["description"] = description
    return True


def attach_node_descriptions_from_extractions(
    graph: GraphStore, annotations_dir: Path,
) -> int:
    """A.0: attach each trial's rich free-text descriptions to the Mechanism /
    Biology / Population nodes its chains touch, sourced from cached
    extractions.

    This is the production path (the real build creates nodes in ``populate``,
    *before* extractions exist, then runs this after the extract step). The
    canonical id is a routing tag; the description is the BioLORD substrate
    (A.1). Reads cached extraction JSON only — no new LLM call, so a
    ``--keep-annotations`` rebuild backfills for free.

    Per node, first non-empty contributor wins. Resolution order, most→least
    specific (combo-arm biology fix — so a chemo's and a checkpoint inhibitor's
    nodes in one arm don't all get the same arm/trial description):
      - Mechanism node ← per-(arm, intervention) mechanism_description, else the
        trial's ``proposed_mechanism``
      - Biology node   ← per-(arm, intervention) biology_description, else the
        trial's ``intended_biology``
      - Parent population ← the trial's ``target_population``
      - Subgroup population ← its own node name (already encodes the
        descriptor; richer trial-contextualized text is the deferred A.0b)

    Returns the number of node descriptions set.
    """
    from src.graph.box_embeddings import chain_descriptions_by_arm_intervention

    desc_by_arm_iv = chain_descriptions_by_arm_intervention(annotations_dir)
    set_count = 0
    for path in sorted(annotations_dir.glob("*_extraction.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        nct_id = data.get("nct_id")
        if not nct_id:
            continue
        try:
            ts = graph.get_trial_subgraph_by_id(nct_id)
        except KeyError:
            continue
        hyp = data.get("therapeutic_hypothesis") or {}
        trial_mech_desc = (hyp.get("proposed_mechanism") or "").strip()
        trial_bio_desc = (hyp.get("intended_biology") or "").strip()
        parent_pop_desc = (hyp.get("target_population") or "").strip()
        for chain in ts.chains:
            iv_entry = _lookup_chain_intervention_desc(
                graph, desc_by_arm_iv, nct_id, chain,
            )
            mech_desc = (
                (iv_entry.get("mechanism") if iv_entry else "") or trial_mech_desc
            )
            bio_desc = (
                (iv_entry.get("biology") if iv_entry else "") or trial_bio_desc
            )
            set_count += _set_node_description(graph, chain.mechanism_id, mech_desc)
            set_count += _set_node_description(graph, chain.biology_id, bio_desc)
            pop_id = chain.subgroup_population_id
            if pop_id == ts.parent_population_id:
                pop_desc = parent_pop_desc
            else:
                try:
                    pop_desc = graph.get_node(pop_id).get("name", "") or ""
                except KeyError:
                    pop_desc = ""
            set_count += _set_node_description(graph, pop_id, pop_desc)
    return set_count


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
        pop_id = PopulationNode.compose_id(feats)  # disease-agnostic {axes}
        if pop_id is None:
            continue  # no real axis → not a population (shouldn't happen here)
        try:
            graph.get_node(pop_id)
        except KeyError:
            graph.add_node(PopulationNode(
                id=pop_id,
                name=PopulationNode.display_name(feats),
                defining_features=list(feats),
            ))
        subgroup_pop_ids[sg.raw_descriptor] = pop_id

    # A.0: preserve the trial's rich free-text descriptions onto the semantic
    # nodes so BioLORD (A.1) embeds a real description rather than the
    # routing-tag id. Mechanism / biology are trial-level (one therapeutic
    # hypothesis); population descriptions are the parent target_population and
    # each subgroup's raw descriptor. All sourced from existing extraction
    # fields, so a populate re-run backfills with no new LLM call.
    _set_node_description(
        graph, mechanism_id, getattr(extraction, "mechanism_description", "")
    )
    _set_node_description(
        graph, biology_id, getattr(extraction, "biology_description", "")
    )
    # All-comers trials have no parent population NODE (parent_pop_id is the
    # UNKNOWN sentinel), so there's nothing to attach the enrollment-cohort
    # description to — skip it. Subgroup descriptions still attach below.
    if parent_pop_id != _UNKNOWN:
        _set_node_description(
            graph,
            parent_pop_id,
            getattr(extraction, "target_population_description", ""),
        )
    for _raw_desc, _pop_id in subgroup_pop_ids.items():
        _set_node_description(graph, _pop_id, _raw_desc)

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
            # Verbatim sponsor-stated stop reason (CT.gov whyStopped),
            # RAW text only. Mirrors TrialNode.metadata so the trial's own
            # words persist into the subgraph snapshot too.
            "why_stopped": trial.why_stopped,
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
            pop_id = PopulationNode.compose_id(features)  # disease-agnostic {axes}
            if pop_id is None:
                continue  # no real axis → not a population
            try:
                graph.get_node(pop_id)
            except KeyError:
                graph.add_node(PopulationNode(
                    id=pop_id,
                    name=PopulationNode.display_name(features),
                    defining_features=list(features),
                ))
            if not graph._graph.has_edge(  # noqa: SLF001
                pop_id, indication_id, key=EdgeType.RESPONDS_DIFFERENTLY.value,
            ):
                # Round-25: heuristic seed prior dropped. The edge
                # exists so per-arm subgroup classifier emissions can
                # land on it; starts uninformative until a trial
                # provides actual subgroup-vs-parent contrast evidence.
                graph.add_edge(GraphEdge(
                    source_id=pop_id,
                    target_id=indication_id,
                    edge_type=EdgeType.RESPONDS_DIFFERENTLY,
                    belief=EdgeBeliefState(),
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
        pop_id = PopulationNode.compose_id(features)  # disease-agnostic {axes}
        if pop_id is None:
            continue  # no real axis → not a population
        try:
            graph.get_node(pop_id)
        except KeyError:
            graph.add_node(PopulationNode(
                id=pop_id,
                name=PopulationNode.display_name(features),
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
        await pipeline.populate_trials(
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
