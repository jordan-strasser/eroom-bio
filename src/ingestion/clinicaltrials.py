"""ClinicalTrials.gov v2 API client."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.graph.models import (
    CompoundNode,
    EndpointNode,
    EndpointType,
    IndicationNode,
    Modality,
    RegulatoryStatus,
)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# ── Models ───────────────────────────────────────────────────────────────


class Intervention(BaseModel):
    name: str
    type: str
    description: str = ""


class OutcomeMeasure(BaseModel):
    measure: str
    timeframe: str = ""
    description: str = ""


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
    # Coarse free text — typically one phrase like "lack of efficacy",
    # "safety concerns", "futility", "business decision". Often the only
    # mechanistic signal available for trials with no posted results.
    why_stopped: str | None = None


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

    # Phase — join multiple phases (e.g. ["PHASE2", "PHASE3"] → "2/3")
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
    )


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
        async with httpx.AsyncClient(timeout=self._timeout) as client:
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

    async def get_study(self, nct_id: str) -> TrialRecord:
        url = f"{BASE_URL}/{nct_id}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return _parse_study(resp.json())

    async def fetch_oncology_with_results(
        self, max_results: int = 1000
    ) -> list[TrialRecord]:
        return await self.search(
            condition="cancer",
            phase="PHASE2,PHASE3",
            status="COMPLETED,TERMINATED",
            has_results=True,
            max_results=max_results,
        )

    async def fetch_oncology_terminated_with_reason(
        self, max_results: int = 1000
    ) -> list[TrialRecord]:
        """Terminated/withdrawn trials that have a sponsor-stated reason.

        Complements ``fetch_oncology_with_results``: many failed programs
        never post results, but the ``whyStopped`` field is enough to update
        the right edges when the compound + target are known. We additionally
        require a DRUG intervention so the trial actually maps to a graph
        compound.
        """
        records = await self.search(
            condition="cancer",
            phase="PHASE2,PHASE3",
            status="TERMINATED,WITHDRAWN",
            max_results=max_results * 2,  # over-fetch; many will be filtered out
        )
        kept: list[TrialRecord] = []
        for record in records:
            if not record.why_stopped:
                continue
            if not any(iv.type == "DRUG" for iv in record.interventions):
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


def map_trial_to_graph_nodes(
    trial: TrialRecord,
) -> dict[str, list[IndicationNode | CompoundNode | EndpointNode]]:
    indications = [
        IndicationNode(
            id=f"IND_{trial.nct_id}_{i}",
            name=cond,
        )
        for i, cond in enumerate(trial.conditions)
    ]

    compounds = [
        CompoundNode(
            id=f"COMP_{trial.nct_id}_{i}",
            name=iv.name,
            modality=_guess_modality(iv),
            metadata={"source_trial": trial.nct_id, "intervention_type": iv.type},
        )
        for i, iv in enumerate(trial.interventions)
        if iv.type == "DRUG"
    ]

    endpoints = [
        EndpointNode(
            id=f"EP_{trial.nct_id}_{i}",
            name=om.measure,
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
            measurement_properties={
                "timeframe": om.timeframe,
                "description": om.description,
            },
        )
        for i, om in enumerate(trial.primary_outcomes)
    ]

    return {
        "indications": indications,
        "compounds": compounds,
        "endpoints": endpoints,
    }
