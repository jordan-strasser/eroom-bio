"""Pydantic models for the Eroom Bio knowledge graph."""

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from scipy import stats

logger = logging.getLogger(__name__)


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


class MechanismCategory(str, Enum):
    """Canonical mechanism-of-action categories.

    Every MechanismNode id is the value of one of these. LLM inference
    must map to one of these; unmapped → OTHER.
    """
    CHECKPOINT_BLOCKADE = "checkpoint_blockade"
    KINASE_INHIBITION = "kinase_inhibition"
    RECEPTOR_ANTAGONISM = "receptor_antagonism"
    RECEPTOR_AGONISM = "receptor_agonism"
    ENZYME_INHIBITION = "enzyme_inhibition"
    PROTEIN_DEGRADATION = "protein_degradation"
    GENE_EDITING = "gene_editing"
    ANTIBODY_DEPENDENT_CYTOTOXICITY = "antibody_dependent_cytotoxicity"
    HORMONE_MODULATION = "hormone_modulation"
    ANTIMETABOLITE = "antimetabolite"
    DNA_DAMAGE = "dna_damage"
    ANGIOGENESIS_INHIBITION = "angiogenesis_inhibition"
    IMMUNE_COSTIMULATION = "immune_costimulation"
    OTHER = "other"


class EndpointClass(str, Enum):
    """Canonical endpoint classes used in EndpointNode ids."""
    OS = "OS"
    DFS = "DFS"
    RFS = "RFS"
    DMFS = "DMFS"
    PFS = "PFS"
    TTP = "TTP"
    CR = "CR"
    ORR = "ORR"
    DOR = "DOR"
    COMPOSITE_SURVIVAL = "composite_survival"
    COMPOSITE_RESPONSE = "composite_response"
    BIOMARKER = "biomarker"
    PRO = "PRO"
    SAFETY = "safety"
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
    # Structural edge: links a synthesized combo CompoundNode to each of its
    # constituent CompoundNodes. Carries no Beta belief—used purely to make
    # the combo's composition queryable from the graph.
    COMPOSED_OF = "composed_of"
    # Compound → AdverseEvent. Belief = P(this compound causes this AE).
    # Evidence: per-trial incidence-rate deltas vs control arm.
    CAUSES_AE = "causes_ae"
    # Target → AdverseEvent. Belief = P(modulating this target causes this AE).
    # Evidence: ≥2 distinct compounds binding the target with strong causes_ae
    # to the same AE—the cross-trial signal that an AE is on-mechanism
    # rather than compound-specific.
    TARGET_ASSOCIATED_AE = "target_associated_ae"


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
    # ChEMBL identifier (e.g. "CHEMBL1201580" for nivolumab) when known.
    # Resolved via Open Targets' drug search; left None for synthesized
    # combo regimens that don't correspond to a single ChEMBL entry.
    chembl_id: str | None = None
    # Brand names, INN aliases, code names ("BMS-936558", "ONO-4538",
    # "Opdivo") stored alongside the canonical compound name and the
    # ChEMBL id so downstream queries can match on whatever name a
    # paper / classifier / label happened to use.
    aliases: list[str] = Field(default_factory=list)
    smiles: str | None = None
    targets_claimed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    gene_symbol: str = Field(min_length=1)
    # Colloquial / clinical-use names ("PD-1", "PDL1", "B7-H1") stored
    # alongside the canonical Ensembl id and HUGO gene_symbol so the
    # graph carries the names a clinician or LLM would actually emit,
    # not just the database-canonical forms. Auto-populated from HGNC
    # when the resolver is loaded.
    aliases: list[str] = Field(default_factory=list)
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


class SubgroupFeature(BaseModel):
    """One identifying feature of a patient subgroup.

    Compositions of these define a PopulationNode. The vocabulary is
    intentionally generic so the same canonical feature ("PD-L1 high")
    links the same node across trials with different cutoffs (≥1% vs
    ≥8%); the trial-specific descriptor is preserved in raw_descriptor.
    """
    axis: str = Field(min_length=1)
    # For axis="gene": HUGO symbol of the gene/protein the level refers to
    # (e.g. "PDCD1LG2" for PD-L2). Empty for self-describing axes like
    # "line" or "performance".
    key: str = ""
    level: str = Field(min_length=1)
    raw_descriptor: str = ""

    def slug(self) -> str:
        """Stable lowercase slug used in PopulationNode ids.

        gene → "{key}_{level}" (e.g. "pdcd1_high")
        non-gene → "{axis}_{level}" (e.g. "line_first")
        other → "other_{slugified(raw_descriptor)}"
        """
        if self.axis == "gene":
            key = re.sub(r"[^a-z0-9]+", "", self.key.lower())
            level = re.sub(r"[^a-z0-9]+", "", self.level.lower())
            return f"{key}_{level}"
        if self.axis == "other":
            cleaned = re.sub(
                r"[^a-z0-9]+", "_", self.raw_descriptor.lower()
            ).strip("_")[:30]
            return f"other_{cleaned or 'unmapped'}"
        return f"{self.axis.lower()}_{self.level.lower()}"


class PopulationNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    defining_features: list[SubgroupFeature] = Field(default_factory=list)
    estimated_size: int | None = None

    @staticmethod
    def compose_id(indication_id: str, features: list[SubgroupFeature]) -> str:
        """Build a deterministic id from indication + sorted feature slugs.

        With no features → ``{indication}__unselected`` (the parent
        enrollment population). The sorted-slug rule means the same set of
        features always produces the same id regardless of input order.
        """
        if not features:
            return f"{indication_id}__unselected"
        slugs = sorted(f.slug() for f in features)
        return f"{indication_id}__" + "__".join(slugs)


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


class AdverseEventNode(BaseModel):
    """A graph-native adverse event keyed on a normalized MedDRA preferred term.

    The id format is ``AE:{lowercase_underscored_meddra_term}`` (e.g.
    ``AE:hepatotoxicity``) so the same AE shared across trials and
    compounds collapses to a single node—the precondition for
    cross-trial learning of mechanism-associated toxicities.
    """
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # MedDRA System Organ Class (e.g. "Hepatobiliary disorders"). Free
    # string; normalized by the MedDRA mapper at extraction time.
    system_organ_class: str = ""
    # Observed CTCAE grade range across trials feeding this node, e.g.
    # "grade_1_3" or "grade_3_4". Updated when new evidence arrives.
    severity_range: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialNode(BaseModel):
    """Marker node anchoring a trial in the graph.

    No outgoing graph edges—the rich per-(arm × subgroup) chain data lives
    on ``GraphStore.trial_subgraphs[id]``. The node exists so the graph
    itself records that the trial happened and so id-based joins from
    other places (e.g. Open Targets clinical evidence rows) have a
    target.
    """
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    phase: str = ""
    status: str = ""
    sponsor: str = ""
    enrollment: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Evidence & belief system ─────────────────────────────────────────────

class EvidenceRecord(BaseModel):
    """One piece of evidence for a Beta-Binomial conjugate update.

    The pair (``source_type``, ``support``) determines the update:
    ``source_type`` selects the effective virtual sample size (N_eff)
    from ``EVIDENCE_TYPE_N_EFF``, and ``support`` selects the implied
    success probability (p_obs) from ``BUCKET_TO_P_OBS``. ``quality_score``
    is an optional [0, 1] discount on N_eff—used for LLM-derived
    records to fold in the classifier's own self-reported confidence
    rubric tier; defaults to 1.0 for evidence streams (LINCS, GWAS)
    that have no classification step to be uncertain about.

    See ``src/inference/beliefs.py`` for the update recipe.
    """
    source_id: str = Field(min_length=1)
    source_type: EvidenceType
    # Categorical strength-and-direction. Stored as a string to keep
    # this module independent of the inference package; values are the
    # SupportBucket enum values.
    support: str = Field(min_length=1)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
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

class TrialArm(BaseModel):
    """One treatment regimen tested in a trial.

    For mono arms, ``compound_ids`` has one entry and ``regimen_compound_id``
    equals it. For combo arms, ``regimen_compound_id`` is the synthesized
    combo CompoundNode id (e.g. ``ipilimumab+nivolumab``)—that's the node
    chains use as their compound, so combo evidence accumulates on a
    distinct beliefs surface and divergence vs. P(A)·P(B) measures
    non-linear synergy.
    """
    arm_id: str = Field(min_length=1)
    compound_ids: list[str] = Field(min_length=1)
    regimen_compound_id: str = Field(min_length=1)
    is_combination: bool = False
    dose_schedule: dict[str, Any] = Field(default_factory=dict)


class CausalChain(BaseModel):
    """One (arm × subgroup) causal-chain hypothesis from a trial.

    The upstream backbone (target → mechanism → biology → indication)
    is shared across subgroup chains within the same arm, but each chain
    carries its own per-(arm × subgroup) outcome / effect_size / p_value
    so attribution and prediction can reason about them independently.

    ``compound_id`` denormalizes the arm's regimen_compound_id onto the
    chain—convenient for prediction, which walks binds_to from compound
    to target and otherwise would have to traverse arms by ``arm_id``.
    For mono arms it equals the single constituent; for combo arms it's
    the synthesized combo CompoundNode id.
    """
    arm_id: str = Field(min_length=1)
    compound_id: str = Field(min_length=1)
    subgroup_population_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    mechanism_id: str = Field(min_length=1)
    biology_id: str = Field(min_length=1)
    indication_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    outcome: TrialOutcome = TrialOutcome.UNKNOWN
    effect_size: float | None = None
    p_value: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialSubgraph(BaseModel):
    """All chains a single trial contributes to the graph.

    A trial produces N arms × M reported subgroups = N·M chains. The
    object lives on ``GraphStore.trial_subgraphs[trial_id]``; it survives
    snapshot/restore. Per-edge `EvidenceRecord.source_id` separately
    carries the trial id for edge-side attribution.
    """
    trial_id: str = Field(min_length=1)
    phase: str = ""
    arms: list[TrialArm] = Field(default_factory=list)
    chains: list[CausalChain] = Field(default_factory=list)
    parent_population_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Canonical IDs ────────────────────────────────────────────────────────

_ENSG_PATTERN = re.compile(r"^ENSG\d{6,}$")
_REACTOME_PATTERN = re.compile(r"^R-[A-Z]{3}-\d+$")
_DRUGBANK_PATTERN = re.compile(r"^DB\d{5,}$")
# {indication}__{feature_slug}[__{feature_slug}...]
# Feature slugs may contain single underscores (e.g. "pdcd1_high",
# "line_first"); the trivial parent population is "{indication}__unselected".
_POPULATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(__[a-z0-9_]+)+$")


def _slugify_lower(text: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to single underscores, trim."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned


def normalize_entity(name: str, node_type: str) -> str:
    """Map a free-form input string to the canonical ID for the given node type.

    Per-type rules:
      - CompoundNode:  DrugBank ID pass-through, else lowercased intervention name
      - TargetNode:    Ensembl gene ID (ENSG...) pass-through, else uppercase symbol
      - MechanismNode: must be a value of MechanismCategory; unmapped → "other"
      - BiologyNode:   Reactome stable ID (R-HSA-...) pass-through; else error
      - BiomarkerNode: uppercase, non-alphanumerics → underscore
      - PopulationNode: validates {indication}_{biomarker_status}_{line_of_therapy}
      - EndpointNode:  must be {EndpointClass}_{indication_id}
      - IndicationNode: lowercased, non-alphanumerics → underscore (MeSH-like slug)

    Raises ValueError on inputs that cannot be canonicalized.
    """
    if not name or not name.strip():
        raise ValueError(f"Empty name for {node_type}")
    raw = name.strip()

    if node_type == "CompoundNode":
        if _DRUGBANK_PATTERN.match(raw):
            return raw
        slug = _slugify_lower(raw)
        if not slug:
            raise ValueError(f"Compound name '{name}' could not be normalized")
        return slug

    if node_type == "TargetNode":
        if _ENSG_PATTERN.match(raw):
            return raw
        symbol = re.sub(r"[^A-Za-z0-9]+", "", raw).upper()
        if not symbol:
            raise ValueError(f"Target symbol '{name}' could not be normalized")
        return symbol

    if node_type == "MechanismNode":
        candidate = _slugify_lower(raw)
        try:
            return MechanismCategory(candidate).value
        except ValueError:
            logger.warning(
                "Mechanism '%s' did not map to MechanismCategory; using 'other'",
                name,
            )
            return MechanismCategory.OTHER.value

    if node_type == "BiologyNode":
        # Two valid id forms:
        #   1. Reactome stable ID (canonical, ground truth from LINCS).
        #   2. ``{mechanism_category}__{indication_slug}`` synthetic slug,
        #      used as a fallback when LINCS/Reactome data isn't available
        #      for a given (mechanism, indication) pair. Lets the
        #      prediction engine traverse the full chain even without
        #      CLUE_API_KEY.
        if _REACTOME_PATTERN.match(raw):
            return raw
        if "__" in raw:
            mech_part, ind_part = raw.split("__", 1)
            try:
                MechanismCategory(mech_part)
            except ValueError:
                raise ValueError(
                    f"BiologyNode '{name}': '{mech_part}' is not a "
                    f"MechanismCategory; slug-form ids must be "
                    f"'{{mechanism}}__{{indication}}'"
                )
            ind_slug = _slugify_lower(ind_part)
            if not ind_slug:
                raise ValueError(
                    f"BiologyNode '{name}': missing indication suffix"
                )
            return f"{mech_part}__{ind_slug}"
        raise ValueError(
            f"BiologyNode id '{name}' must be a Reactome stable ID "
            f"(e.g. R-HSA-9006934) or a '{{mechanism}}__{{indication}}' slug"
        )

    if node_type == "BiomarkerNode":
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
        if not cleaned:
            raise ValueError(f"Biomarker '{name}' could not be normalized")
        return cleaned

    if node_type == "PopulationNode":
        # Already-formatted ids contain the "__" separator; preserve as-is
        # after lower-casing so callers can pass an id straight through.
        if "__" in raw:
            candidate = raw.lower()
        else:
            candidate = _slugify_lower(raw)
        if not _POPULATION_PATTERN.match(candidate):
            raise ValueError(
                f"PopulationNode id '{name}' must be "
                f"'{{indication}}__{{feature_slug}}[__{{feature_slug}}...]' "
                f"(use PopulationNode.compose_id to build it)"
            )
        return candidate

    if node_type == "EndpointNode":
        # Expect {EndpointClass}_{indication_id}. Multi-word class values
        # like 'composite_response' contain underscores, so split on the
        # *first* '_' is wrong—match against EndpointClass values
        # longest-first instead.
        if "_" not in raw:
            raise ValueError(
                f"EndpointNode id '{name}' must be '{{class}}_{{indication}}'"
            )
        for cls in sorted(EndpointClass, key=lambda c: -len(c.value)):
            prefix = f"{cls.value}_"
            if raw.startswith(prefix):
                ind = _slugify_lower(raw[len(prefix):])
                if not ind:
                    raise ValueError(
                        f"EndpointNode '{name}': missing indication suffix"
                    )
                return f"{cls.value}_{ind}"
        cls_part = raw.split("_", 1)[0]
        raise ValueError(
            f"EndpointNode '{name}': '{cls_part}' is not an EndpointClass"
        )

    if node_type == "IndicationNode":
        slug = _slugify_lower(raw)
        if not slug:
            raise ValueError(f"Indication '{name}' could not be normalized")
        return slug

    raise ValueError(f"Unknown node_type '{node_type}'")
