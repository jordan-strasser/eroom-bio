"""Pydantic models for the Eroom Bio knowledge graph."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from scipy import stats


# ── Enums ────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    SMALL_MOLECULE = "small_molecule"
    ANTIBODY = "antibody"
    ADC = "adc"
    GENE_THERAPY = "gene_therapy"
    CELL_THERAPY = "cell_therapy"
    OTHER = "other"


class MechanismType(str, Enum):
    INHIBITION = "inhibition"
    AGONISM = "agonism"
    ANTAGONISM = "antagonism"
    DEGRADATION = "degradation"
    MODULATION = "modulation"
    EDITING = "editing"
    OTHER = "other"


class BiomarkerType(str, Enum):
    PD_MARKER = "pd_marker"
    SURROGATE_ENDPOINT = "surrogate_endpoint"
    PREDICTIVE = "predictive"
    PROGNOSTIC = "prognostic"
    EXPLORATORY = "exploratory"


class EndpointType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"


class RegulatoryStatus(str, Enum):
    ACCEPTED = "accepted"
    REASONABLY_LIKELY = "reasonably_likely"
    EXPLORATORY = "exploratory"


class EdgeType(str, Enum):
    BINDS_TO = "binds_to"
    MODULATES_VIA = "modulates_via"
    MECHANISM_AFFECTS = "mechanism_affects"
    BIOLOGY_DRIVES = "biology_drives"
    REFLECTS_BIOLOGY = "reflects_biology"
    ENDPOINT_CAPTURES = "endpoint_captures"
    RESPONDS_DIFFERENTLY = "responds_differently"


class EvidenceType(str, Enum):
    CLINICAL_PHASE3 = "clinical_phase3"
    CLINICAL_PHASE2 = "clinical_phase2"
    CLINICAL_PHASE1 = "clinical_phase1"
    GENETIC_MR = "genetic_mr"
    GENETIC_GWAS = "genetic_gwas"
    PRECLINICAL_IN_VIVO = "preclinical_in_vivo"
    PRECLINICAL_IN_VITRO = "preclinical_in_vitro"
    COMPUTATIONAL = "computational"
    LITERATURE = "literature"


class EvidenceDirection(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    AMBIGUOUS = "ambiguous"


class TrialOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


# ── Node models ──────────────────────────────────────────────────────────

class CompoundNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    modality: Modality
    smiles: str | None = None
    targets_claimed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    druggability_score: float | None = None
    tissue_expression: dict[str, float] = Field(default_factory=dict)
    essentiality_scores: dict[str, float] = Field(default_factory=dict)


class MechanismNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mechanism_type: MechanismType
    selectivity: str | None = None


class BiologyNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    pathway_ids: list[str] = Field(default_factory=list)
    tissue_specificity: list[str] = Field(default_factory=list)
    known_redundancies: list[str] = Field(default_factory=list)


class BiomarkerNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    biomarker_type: BiomarkerType
    measurability_score: float | None = None


class PopulationNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    defining_features: list[str] = Field(default_factory=list)
    estimated_size: int | None = None
    genomic_features: list[str] = Field(default_factory=list)


class EndpointNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    endpoint_type: EndpointType
    regulatory_status: RegulatoryStatus
    measurement_properties: dict[str, Any] = Field(default_factory=dict)


class IndicationNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    icd_codes: list[str] = Field(default_factory=list)
    prevalence: float | None = None
    standard_of_care: str | None = None
    unmet_need_score: float | None = None


# ── Evidence & belief system ─────────────────────────────────────────────

class EvidenceRecord(BaseModel):
    source_id: str = Field(min_length=1)
    source_type: EvidenceType
    quality_score: float = Field(ge=0.0, le=1.0)
    direction: EvidenceDirection
    magnitude: float = Field(ge=0.0)
    timestamp: datetime
    provenance_url: str | None = None
    notes: str | None = None
    # Free-form structured context (e.g. {"cell_line": "A375", "tissue": "skin"}).
    # Used at query time by context-conditioned belief retrieval to downweight
    # evidence that doesn't match the queried indication's tissue. Empty for
    # context-free evidence (the default for everything except LINCS sigs).
    context: dict[str, Any] = Field(default_factory=dict)


class EdgeBeliefState(BaseModel):
    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    evidence: list[EvidenceRecord] = Field(default_factory=list)

    @property
    def expected_probability(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence_strength(self) -> float:
        return self.alpha + self.beta - 2.0

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def conflict_score(self) -> float:
        """High when lots of evidence but probability near 0.5."""
        strength = self.evidence_strength
        if strength <= 0:
            return 0.0
        p = self.expected_probability
        certainty = abs(p - 0.5) * 2  # 0 at p=0.5, 1 at p=0 or p=1
        return strength * (1 - certainty)

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        tail = (1 - ci) / 2
        lower = stats.beta.ppf(tail, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - tail, self.alpha, self.beta)
        return (float(lower), float(upper))


class GraphEdge(BaseModel):
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    edge_type: EdgeType
    belief: EdgeBeliefState = Field(default_factory=EdgeBeliefState)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Trial subgraph ───────────────────────────────────────────────────────

class TrialSubgraph(BaseModel):
    trial_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    mechanism_id: str = Field(min_length=1)
    biology_id: str = Field(min_length=1)
    indication_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    population_id: str = Field(min_length=1)
    outcome: TrialOutcome
    phase: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
