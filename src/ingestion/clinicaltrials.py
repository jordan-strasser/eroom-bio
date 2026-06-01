"""ClinicalTrials.gov v2 API client."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.graph.models import (
    CompoundNode,
    IndicationNode,
    InterventionType,
    Modality,
    normalize_entity,
)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


# ── Compound-name canonicalization ───────────────────────────────────────
#
# CT.gov intervention names join multiple drugs into one string
# ("Sorafenib (Nexavar; BAY43-9006) + Dacarbazine (DTIC)",
# "Carboplatin/Paclitaxel") and embed brand/dev-code aliases in
# parentheticals. Both `build_arms` and `map_trial_to_graph_nodes` apply
# the same split-then-strip canonicalization so a drug gets one
# CompoundNode regardless of how the trial reports it. Lives in this
# module so both call sites import it without a populate↔ingestion cycle.

_REGIMEN_SEPARATOR_RE = re.compile(
    r"\s+\+\s+|\s+plus\s+|\s+and\s+|\s*/\s*",
    re.IGNORECASE,
)


def split_combo_regimen(iv_name: str) -> list[str] | None:
    """Split an intervention name that names multiple drugs as one regimen.

    Handles common CT.gov-side joins:
      "Carboplatin/Paclitaxel"                       → ["Carboplatin", "Paclitaxel"]
      "Sorafenib (Nexavar) + Dacarbazine (DTIC)"     → ["Sorafenib (Nexavar)", "Dacarbazine (DTIC)"]
      "Drug A and Drug B"                            → ["Drug A", "Drug B"]
      "Sorafenib"                                    → None (no split)
      "5% Dextrose/Saline"                           → None (numbers suggest formulation)

    Per-part validation runs against the parenthetical-stripped form so
    brand aliases don't disqualify a real drug name. Each bare part must
    start with a letter, be ≥3 chars, and contain no digits — preserves
    the original name as a single compound when the split would produce
    a non-drug-like fragment.
    """
    if not _REGIMEN_SEPARATOR_RE.search(iv_name):
        return None
    parts = [p.strip() for p in _REGIMEN_SEPARATOR_RE.split(iv_name)]
    if len(parts) < 2:
        return None
    bare_re = re.compile(r"^[A-Za-z][A-Za-z\-\s]*$")
    for p in parts:
        bare = re.sub(r"\s*\([^)]*\)\s*", " ", p).strip()
        if not bare or len(bare) < 3 or not bare_re.match(bare):
            return None
    return parts


def strip_parenthetical_brand(name: str) -> str | None:
    """Strip trailing parenthetical brand/synonym from a drug name.

    "Sorafenib (Nexavar; BAY43-9006)" → "Sorafenib". The parenthetical is
    usually a brand name + dev code that OT doesn't resolve directly.
    Returns None when the stripped form would be empty, would equal the
    original, or when the parenthetical content contains a combination-
    indicator word ("with", "plus", "and", "+", "combo") suggesting it
    encodes a regimen rather than a brand.
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


# Round-27: dose-suffix patterns. CT.gov intervention names commonly
# embed dose info ("Bevacizumab 7.5 mg/kg", "Capecitabine 1000 mg/m^2",
# "Aspirin 81 mg PO daily"). Without stripping, the OT / ChEMBL lookup
# fails because the dose-laden name isn't a canonical drug name; the
# SapBERT canonicalization then keeps the dose-laden slug as the
# canonical id and the cleaner bare name gets demoted to alias.
# Worse, the bev → MET phantom edge in round 26 was created when the
# unresolved dose-laden compound fell through to the cross-reference
# name-match heuristic with no chembl_id gate.
#
# Patterns matched (case-insensitive, anchored to a word break):
#   - mass dose: "7.5 mg", "200 mg", "100mg" — optionally followed by
#     /kg or /m^2 or /m2 (BSA dosing).
#   - other dose units: μg/mcg/ng/g, IU/MIU/MU/Units.
#   - frequency / route: bid, tid, qid, qd, qhs, q4h, q6h, q8h, q12h, IV,
#     PO, SC, IM (these often appear after the dose).
# We do NOT strip "5-fluorouracil" (the leading 5 is the chemical name);
# the patterns all require a space before the number to avoid that.
_DOSE_PATTERN = re.compile(
    r"\s+\d+(?:\.\d+)?\s*(?:mg|μg|ug|mcg|ng|g|iu|miu|mu|units?|u)"
    r"(?:\s*/\s*(?:kg|m\^?2|m2|day|d))?"
    r"\b",
    re.IGNORECASE,
)
_FREQ_ROUTE_PATTERN = re.compile(
    r"\s+(?:q\d+h|q[idh]s?\b|[bt]?id\b|qd\b|po\b|iv\b|sc\b|im\b)",
    re.IGNORECASE,
)


def strip_dose_suffix(name: str) -> str:
    """Strip dose/frequency/route phrases from a CT.gov intervention name.

    "Bevacizumab 7.5 mg/kg" → "Bevacizumab"
    "Capecitabine 1000 mg/m^2" → "Capecitabine"
    "Aspirin 81 mg PO daily" → "Aspirin"

    Idempotent — strings without dose patterns round-trip unchanged.
    Does NOT touch leading numerics that are part of the chemical name
    (e.g. "5-Fluorouracil" — the pattern requires whitespace before the
    number).
    """
    if not name:
        return name
    s = _DOSE_PATTERN.sub("", name)
    # Re-apply for second dose phrase ("X mg + Y mg" style).
    s = _DOSE_PATTERN.sub("", s)
    s = _FREQ_ROUTE_PATTERN.sub("", s)
    # Collapse double spaces created by middle-of-string strips.
    s = re.sub(r"\s+", " ", s).strip()
    return s or name


def canonical_compound_names(iv_name: str) -> list[str]:
    """Canonicalize an intervention name into one or more drug names.

    Pipeline (order matters):
      1. ``strip_dose_suffix`` first, on the WHOLE input. CT.gov combo
         names often have per-constituent doses ("Oxaliplatin +
         Capecitabine 1000 mg/m^2 + Bevacizumab 7.5 mg/kg"), and
         ``split_combo_regimen`` has a strict drug-name-shape regex that
         rejects parts containing digits — so we must remove the dose
         info before splitting (round-27 fix).
      2. ``split_combo_regimen`` decomposes "Drug A + Drug B" into list.
      3. ``strip_parenthetical_brand`` strips "(brand; codename)" tails.
      4. ``strip_dose_suffix`` once more per-constituent as a safety net.

    Always returns a non-empty list — non-decomposable names round-trip
    as a single-element list of the cleaned name.
    """
    dose_stripped = strip_dose_suffix(iv_name)
    sub_names = split_combo_regimen(dose_stripped) or [dose_stripped]
    out: list[str] = []
    for s in sub_names:
        bare = strip_parenthetical_brand(s) or s
        bare = strip_dose_suffix(bare)
        out.append(bare)
    return out


# ── Models ───────────────────────────────────────────────────────────────


class Intervention(BaseModel):
    name: str
    type: str
    description: str = ""


class OutcomeMeasure(BaseModel):
    measure: str
    timeframe: str = ""
    description: str = ""


class ArmGroup(BaseModel):
    """One treatment-arm group as reported in the trial results section.

    Sourced from ``outcomeMeasuresModule.outcomeMeasures[*].groups`` (and
    deduped across outcomes within the trial). The ``intervention_names``
    are the Intervention names this group received—that's how we
    reconstruct combo arms ("Nivolumab + Ipilimumab" group → both drugs).
    """
    group_id: str
    title: str
    description: str = ""
    intervention_names: list[str] = Field(default_factory=list)


# Intervention types that should produce a CompoundNode. ClinicalTrials.gov
# splits drug-like therapies between DRUG (small molecules) and BIOLOGICAL
# (antibodies, fusion proteins, cell therapies). Both belong in the graph;
# the original DRUG-only filter dropped every checkpoint inhibitor.
DRUG_LIKE_INTERVENTION_TYPES: frozenset[str] = frozenset({"DRUG", "BIOLOGICAL"})


def is_drug_like(intervention: "Intervention") -> bool:
    return intervention.type in DRUG_LIKE_INTERVENTION_TYPES


class Reference(BaseModel):
    """A linked publication from CT.gov ``referencesModule.references``.

    For terminated/withdrawn trials with no posted results, the canonical safety
    signal often lives only in a linked paper (e.g. torcetrapib's BP/aldosterone
    effect in PMID 17984165), never in the structured record. ``type`` is
    CT.gov's BACKGROUND / RESULT / DERIVED.
    """

    pmid: str = ""
    citation: str = ""
    type: str = ""


class TrialRecord(BaseModel):
    nct_id: str
    title: str
    phase: str = ""
    status: str = ""
    conditions: list[str] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    primary_outcomes: list[OutcomeMeasure] = Field(default_factory=list)
    secondary_outcomes: list[OutcomeMeasure] = Field(default_factory=list)
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    sponsor: str = ""
    has_results: bool = False
    results_summary: dict[str, Any] | None = None
    # Sponsor's stated reason a trial was stopped early (TERMINATED/WITHDRAWN).
    # Coarse free text—typically one phrase like "lack of efficacy",
    # "safety concerns", "futility", "business decision". Often the only
    # mechanistic signal available for trials with no posted results.
    why_stopped: str | None = None
    # Treatment arm groups parsed from the results section. Empty when the
    # trial has no posted results or only one arm.
    arm_groups: list[ArmGroup] = Field(default_factory=list)
    # Free-text subgroup stratifier titles parsed from the outcome
    # measures' ``classes`` (e.g. "PD-L1 Expression Level >= 1%"). The
    # extractor canonicalizes these into SubgroupFeature lists; here they
    # are kept as raw strings for provenance.
    subgroup_descriptors: list[str] = Field(default_factory=list)
    # Free-text inclusion/exclusion criteria from CT.gov. Round-17 uses
    # this with an LLM extractor to pull population features
    # (line_of_therapy, prior treatments, biomarker requirements,
    # mutation status) that the conditions field and the deterministic
    # qualifier regex can't capture. Empty when the trial's CT.gov
    # record omits the eligibility module.
    eligibility_criteria: str = ""
    # Linked publications (referencesModule). PMIDs here are often the only
    # mechanistic/safety signal for terminated trials with no resultsSection;
    # the PubMed safety-enrichment path fetches these abstracts. Empty when
    # CT.gov omits the references module.
    references: list[Reference] = Field(default_factory=list)


# ── Parsing helpers ──────────────────────────────────────────────────────

_PHASE_MAP = {
    "PHASE1": "1",
    "PHASE2": "2",
    "PHASE3": "3",
    "PHASE4": "4",
    "EARLY_PHASE1": "early_1",
    "NA": "",
}


def _parse_study(raw: dict[str, Any]) -> TrialRecord:
    proto = raw.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    conds_mod = proto.get("conditionsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    elig_mod = proto.get("eligibilityModule", {})

    # Phase—join multiple phases (e.g. ["PHASE2", "PHASE3"] → "2/3")
    raw_phases = design.get("phases", [])
    phase = "/".join(_PHASE_MAP.get(p, p) for p in raw_phases)

    # Interventions
    interventions = []
    for iv in arms_mod.get("interventions", []):
        interventions.append(
            Intervention(
                name=iv.get("name", ""),
                type=iv.get("type", ""),
                description=iv.get("description", ""),
            )
        )

    # Outcomes
    primary = [
        OutcomeMeasure(
            measure=o.get("measure", ""),
            timeframe=o.get("timeFrame", ""),
            description=o.get("description", ""),
        )
        for o in outcomes_mod.get("primaryOutcomes", [])
    ]
    secondary = [
        OutcomeMeasure(
            measure=o.get("measure", ""),
            timeframe=o.get("timeFrame", ""),
            description=o.get("description", ""),
        )
        for o in outcomes_mod.get("secondaryOutcomes", [])
    ]

    # Enrollment
    enrollment_info = design.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count")

    # Dates
    start = status_mod.get("startDateStruct", {}).get("date")
    completion = status_mod.get("completionDateStruct", {}).get("date")

    # Sponsor
    lead = sponsor_mod.get("leadSponsor", {})
    sponsor = lead.get("name", "")

    # Results
    has_results = raw.get("hasResults", False)
    results_section = raw.get("resultsSection") if has_results else None

    why_stopped = status_mod.get("whyStopped") or None

    # Arm groups: prefer protocolSection.armsInterventionsModule.armGroups
    # (always present for multi-arm trials, includes intervention_names).
    # Backfill with the results-section group titles for trials where the
    # protocol omitted them but reported results by arm.
    arm_groups = _parse_protocol_arm_groups(arms_mod)
    if not arm_groups and results_section:
        arm_groups = _parse_results_arm_groups(results_section)

    # Subgroup stratifier descriptors only appear in posted results.
    subgroup_descriptors = _parse_subgroup_descriptors(results_section)

    # Linked publications (PMIDs) — often the only safety signal for terminated
    # trials with no resultsSection (e.g. torcetrapib's NEJM paper).
    references = [
        Reference(
            pmid=r.get("pmid", ""),
            citation=r.get("citation", ""),
            type=r.get("type", ""),
        )
        for r in proto.get("referencesModule", {}).get("references", [])
    ]

    return TrialRecord(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", ""),
        phase=phase,
        status=status_mod.get("overallStatus", ""),
        conditions=conds_mod.get("conditions", []),
        interventions=interventions,
        primary_outcomes=primary,
        secondary_outcomes=secondary,
        enrollment=enrollment,
        start_date=start,
        completion_date=completion,
        sponsor=sponsor,
        has_results=has_results,
        results_summary=results_section,
        why_stopped=why_stopped,
        arm_groups=arm_groups,
        subgroup_descriptors=subgroup_descriptors,
        eligibility_criteria=elig_mod.get("eligibilityCriteria", "") or "",
        references=references,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


# CT.gov v2 prefixes interventionNames with the human-readable type
# ("Biological: Nivolumab", "Drug: Imatinib"). The case isn't always
# upper—Title case is the most common form—so match against a known
# allowlist rather than checking isupper().
_INTERVENTION_TYPE_PREFIXES: frozenset[str] = frozenset({
    "drug", "biological", "behavioral", "device", "diagnostic test",
    "dietary supplement", "genetic", "procedure", "radiation",
    "combination product", "other",
})


def _strip_intervention_prefix(name: str) -> str:
    """Drop the ``<Type>: `` prefix from CT.gov intervention names.

    Returns the cleaned name (or the original input when no known
    prefix matches).
    """
    if ":" not in name:
        return name
    prefix, rest = name.split(":", 1)
    if prefix.strip().lower() in _INTERVENTION_TYPE_PREFIXES:
        return rest.strip()
    return name


# Placebos appear as their own intervention rows in blinded trials
# (e.g. CheckMate 067 lists "Biological: Placebo for Nivolumab" so the
# control arm receives a matching infusion). They are not a therapy
# under test—exclude them from the arm's ``compound_ids`` so combo
# detection isn't fooled by placebo rows and attribution doesn't try
# to route binds_to evidence to a non-existent placebo CompoundNode.
_PLACEBO_RE = re.compile(r"^\s*(placebo|sham)\b", re.IGNORECASE)


def is_placebo_name(name: str) -> bool:
    return bool(_PLACEBO_RE.search(name))


def _parse_protocol_arm_groups(
    arms_mod: dict[str, Any],
) -> list[ArmGroup]:
    """Read the canonical arm-group definitions from protocolSection.

    The ``intervention_names`` list is what tells us a combo from a mono
    arm. Type prefixes are stripped, and placebo rows are dropped—so
    a "nivo + placebo + placebo" arm collapses to a single-drug arm
    rather than a fake combo.
    """
    out: list[ArmGroup] = []
    for ag in arms_mod.get("armGroups", []) or []:
        label = (ag.get("label") or "").strip()
        if not label:
            continue
        cleaned: list[str] = []
        for raw in ag.get("interventionNames", []) or []:
            name = _strip_intervention_prefix((raw or "").strip())
            if not name or is_placebo_name(name):
                continue
            cleaned.append(name)
        arm_id = _slug(label) or f"arm_{len(out)}"
        out.append(ArmGroup(
            group_id=arm_id,
            title=label,
            description=(ag.get("description") or "").strip(),
            intervention_names=cleaned,
        ))
    return out


def _parse_results_arm_groups(
    results_section: dict[str, Any],
) -> list[ArmGroup]:
    """Fallback: derive arm groups from the results section.

    No intervention_names available here—those will be reconstructed
    later by matching against the trial's ``interventions`` list.
    """
    seen: dict[str, ArmGroup] = {}
    om_module = results_section.get("outcomeMeasuresModule", {}) or {}
    for om in om_module.get("outcomeMeasures", []) or []:
        for g in om.get("groups", []) or []:
            gid = g.get("id", "")
            if not gid or gid in seen:
                continue
            seen[gid] = ArmGroup(
                group_id=gid,
                title=g.get("title", "") or "",
                description=g.get("description", "") or "",
            )
    return list(seen.values())


def _parse_subgroup_descriptors(
    results_section: dict[str, Any] | None,
) -> list[str]:
    """Unique subgroup stratifier titles from outcome measures."""
    if not results_section:
        return []
    om_module = results_section.get("outcomeMeasuresModule", {}) or {}
    raw_titles: list[str] = []
    seen: set[str] = set()
    for om in om_module.get("outcomeMeasures", []) or []:
        for cls in om.get("classes", []) or []:
            title = (cls.get("title") or "").strip()
            if not title:
                continue
            lower = title.lower()
            if lower in {"total", "overall", "all participants", "all patients"}:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            raw_titles.append(title)
    return raw_titles


# ── Client ───────────────────────────────────────────────────────────────


class ClinicalTrialsClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def search(
        self,
        condition: str | None = None,
        phase: str | None = None,
        status: str | None = None,
        has_results: bool | None = None,
        max_results: int = 100,
    ) -> list[TrialRecord]:
        params: dict[str, Any] = {
            "pageSize": min(max_results, 1000),
        }
        if condition:
            params["query.cond"] = condition

        # Build filter.advanced with AREA[] syntax for the v2 API
        filters: list[str] = []
        if phase:
            # phase can be "PHASE3" or "PHASE2,PHASE3"
            parts = [p.strip() for p in phase.split(",")]
            if len(parts) == 1:
                filters.append(f"AREA[Phase]{parts[0]}")
            else:
                clauses = " OR ".join(parts)
                filters.append(f"AREA[Phase]({clauses})")
        if status:
            parts = [s.strip() for s in status.split(",")]
            if len(parts) == 1:
                filters.append(f"AREA[OverallStatus]{parts[0]}")
            else:
                clauses = " OR ".join(parts)
                filters.append(f"AREA[OverallStatus]({clauses})")
        if filters:
            params["filter.advanced"] = " AND ".join(filters)

        # The API doesn't have a direct has_results filter —
        # we filter client-side after fetching.

        records: list[TrialRecord] = []
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            while len(records) < max_results:
                params["pageSize"] = min(max_results - len(records), 1000)
                resp = await client.get(BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                for study in data.get("studies", []):
                    record = _parse_study(study)
                    if has_results is not None and record.has_results != has_results:
                        continue
                    records.append(record)
                    if len(records) >= max_results:
                        break

                next_token = data.get("nextPageToken")
                if not next_token:
                    break
                params["pageToken"] = next_token

        return records

    async def get_study(
        self, nct_id: str, *, max_retries: int = 4,
    ) -> TrialRecord:
        """Fetch one trial by NCT id.

        CT.gov v2 throttles aggressively under concurrent fetches: when
        we exceed their rate budget the server returns 302 redirects
        whose follow-up GET hits an `/error/429.shtml` page returning
        429 Too Many Requests. Retry with exponential backoff so we
        don't silently drop trials during multi-indication builds with
        concurrency > 1.
        """
        url = f"{BASE_URL}/{nct_id}"
        delay = 1.0
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True,
        ) as client:
            for attempt in range(max_retries):
                resp = await client.get(url)
                if resp.status_code == 429 or (
                    resp.status_code == 200 and "/error/" in str(resp.url)
                ):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                resp.raise_for_status()
                return _parse_study(resp.json())
            # Exhausted retries — let the final attempt raise.
            resp.raise_for_status()  # type: ignore[possibly-undefined]
            return _parse_study(resp.json())  # type: ignore[possibly-undefined]

    async def fetch_oncology_with_results(
        self, max_results: int = 1000, condition: str = "cancer"
    ) -> list[TrialRecord]:
        return await self.search(
            condition=condition,
            phase="PHASE2,PHASE3",
            status="COMPLETED,TERMINATED",
            has_results=True,
            max_results=max_results,
        )

    async def fetch_oncology_terminated_with_reason(
        self, max_results: int = 1000, condition: str = "cancer"
    ) -> list[TrialRecord]:
        """Terminated/withdrawn trials that have a sponsor-stated reason.

        Complements ``fetch_oncology_with_results``: many failed programs
        never post results, but the ``whyStopped`` field is enough to update
        the right edges when the compound + target are known. We additionally
        require a DRUG intervention so the trial actually maps to a graph
        compound.
        """
        records = await self.search(
            condition=condition,
            phase="PHASE2,PHASE3",
            status="TERMINATED,WITHDRAWN",
            max_results=max_results * 2,  # over-fetch; many will be filtered out
        )
        kept: list[TrialRecord] = []
        for record in records:
            if not record.why_stopped:
                continue
            if not any(is_drug_like(iv) for iv in record.interventions):
                continue
            kept.append(record)
            if len(kept) >= max_results:
                break
        return kept


# ── Graph node mapping ───────────────────────────────────────────────────

_MODALITY_PATTERNS: list[tuple[str, Modality]] = [
    (r"antibody.drug.conjugate|ADC", Modality.ADC),
    (r"antibod|mab\b", Modality.ANTIBODY),
    (r"gene.therap|AAV|viral.vector", Modality.GENE_THERAPY),
    (r"cell.therap|CAR.T|CAR-T", Modality.CELL_THERAPY),
]


def _guess_modality(intervention: Intervention) -> Modality:
    text = f"{intervention.name} {intervention.description}".lower()
    for pattern, modality in _MODALITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return modality
    return Modality.SMALL_MOLECULE


_CTGOV_TYPE_MAP: dict[str, InterventionType] = {
    "DRUG": InterventionType.DRUG,
    "BIOLOGICAL": InterventionType.BIOLOGICAL,
    "COMBINATION_PRODUCT": InterventionType.COMBINATION,
    "COMBINATION": InterventionType.COMBINATION,
    "RADIATION": InterventionType.RADIATION,
    "DEVICE": InterventionType.DEVICE,
    "PROCEDURE": InterventionType.PROCEDURE,
    "DIAGNOSTIC_TEST": InterventionType.DIAGNOSTIC_TEST,
    "BEHAVIORAL": InterventionType.BEHAVIORAL,
    "GENETIC": InterventionType.BIOLOGICAL,  # CT.gov "Genetic" usually means gene/cell therapy
    "DIETARY_SUPPLEMENT": InterventionType.OTHER,
    "OTHER": InterventionType.OTHER,
}


def _map_ctgov_intervention_type(raw_type: str) -> InterventionType:
    """Map a CT.gov intervention-type string onto the InterventionType enum.

    CT.gov's enum varies slightly across API versions (`COMBINATION` vs
    `COMBINATION_PRODUCT`); the table above normalizes the common forms.
    Unknown / blank types fall back to ``UNKNOWN`` so the node still
    records what type CT.gov claimed without losing the data.
    """
    return _CTGOV_TYPE_MAP.get(raw_type.strip().upper(), InterventionType.UNKNOWN)


def map_trial_to_graph_nodes(
    trial: TrialRecord,
) -> dict[str, list[IndicationNode | CompoundNode]]:
    """Emit canonical Compound + Indication nodes for a trial.

    EndpointNodes are intentionally not produced here: their canonical id is
    ``{EndpointClass}_{indication_id}`` (e.g. ``OS_melanoma``), which depends
    on an LLM classification step. Endpoint creation lives in the population
    pipeline so we never seed an endpoint node from raw outcome text.
    """
    indications = [
        IndicationNode(
            id=normalize_entity(cond, "IndicationNode"),
            name=cond,
        )
        for cond in trial.conditions
    ]

    compounds = [
        CompoundNode(
            id=normalize_entity(canonical, "InterventionNode"),
            name=canonical,
            # Preserve the original intervention name as an alias so the
            # downstream entity index can resolve raw CT.gov names
            # ("Sorafenib (Nexavar; BAY43-9006)") back to the canonical
            # CompoundNode id ("sorafenib"). Without this, OT-lookup and
            # other code paths that key off raw iv.name silently miss.
            aliases=[iv.name] if iv.name != canonical else [],
            intervention_type=_map_ctgov_intervention_type(iv.type),
            modality=_guess_modality(iv),
            metadata={"intervention_type_raw": iv.type},
        )
        for iv in trial.interventions
        if is_drug_like(iv)
        for canonical in canonical_compound_names(iv.name)
    ]

    return {
        "indications": indications,
        "compounds": compounds,
    }
