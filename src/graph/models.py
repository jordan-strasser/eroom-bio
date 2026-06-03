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


class InterventionType(str, Enum):
    """Mirrors ClinicalTrials.gov's intervention type enumeration.

    Stored on every InterventionNode so a single typed namespace covers
    drugs, biologicals, radiation, devices, cell therapies, procedures,
    and the misc categories CT.gov uses. Chain fan-out currently fires
    only for interventions whose role is PRIMARY *and* whose type is one
    of {DRUG, BIOLOGICAL, COMBINATION} — those are the ones that map
    cleanly onto the `affects → Target → modulates_via → Mechanism → ...`
    backbone. Radiation, device, and procedure primaries are tracked as
    nodes (so the trial participates in the graph and accumulates AE
    evidence) but their chain backbones land on UNKNOWN target / mechanism
    until we add proper edge types for them.
    """
    DRUG = "drug"
    BIOLOGICAL = "biological"
    COMBINATION = "combination"
    RADIATION = "radiation"
    DEVICE = "device"
    PROCEDURE = "procedure"
    DIAGNOSTIC_TEST = "diagnostic_test"
    BEHAVIORAL = "behavioral"
    OTHER = "other"
    UNKNOWN = "unknown"


class InterventionRole(str, Enum):
    """Per-trial role of an intervention.

    PRIMARY = the intervention under investigation in this trial. Gets
    a chain backbone (affects → Target → modulates_via → ...).

    SUPPORTIVE = preconditioning, premedication, growth-factor support,
    or other infrastructure that enables the primary intervention.
    Tracked as an arm member (so AE attribution still fires on it) but
    does NOT generate its own causal-chain backbone — the trial isn't
    testing the supportive agent's hypothesis, so accumulating
    `affects → mechanism → biology → indication` evidence on it would
    be spurious.

    UNKNOWN = role couldn't be determined from extraction. Default
    treatment: same as SUPPORTIVE (don't chain), to avoid false-positive
    evidence accumulation.
    """
    PRIMARY = "primary"
    SUPPORTIVE = "supportive"
    UNKNOWN = "unknown"


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
    must map to one of these; unmapped → OTHER. New values are added
    when ChEMBL surfaces a structured `action_type` that doesn't fit an
    existing category — see ``_CHEMBL_ACTION_TO_MECHANISM`` in
    ``src/ingestion/opentargets.py`` for the mapping.
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
    # DNA_CROSSLINKING — specific to platinums + classical alkylators
    # (carboplatin, cisplatin, cyclophosphamide). Distinct from DNA_DAMAGE
    # which covers the broader category (intercalators, topoisomerase
    # poisons, radiomimetics). ChEMBL action_type CROSS-LINKING AGENT.
    DNA_CROSSLINKING = "dna_crosslinking"
    # MICROTUBULE_BINDING — taxanes, vinca alkaloids. Target =
    # microtubule polymer (CHEBI) or tubulin family (HGNC). ChEMBL
    # action_type MICROTUBULE BINDER / TUBULIN INHIBITOR.
    MICROTUBULE_BINDING = "microtubule_binding"
    # ANTIGEN_DIRECTED_CYTOTOXICITY — CAR-T, TCR-engineered T cells,
    # bispecific T-cell engagers. Target = surface antigen (HGNC gene
    # symbol). Distinct from ANTIBODY_DEPENDENT_CYTOTOXICITY which is
    # the monoclonal-antibody / NK-cell version of the same idea.
    ANTIGEN_DIRECTED_CYTOTOXICITY = "antigen_directed_cytotoxicity"
    ANGIOGENESIS_INHIBITION = "angiogenesis_inhibition"
    IMMUNE_COSTIMULATION = "immune_costimulation"
    OTHER = "other"


class TargetType(str, Enum):
    """What kind of entity a TargetNode represents.

    Preserves the scale distinction with downstream layers: a Target is
    the *specific thing physically engaged* by the compound, never the
    action (Mechanism) or the downstream pathway (Biology). Putting
    "G2/M arrest" as a Target would conflate it with Biology; putting
    "alkylation" as a Target would conflate it with Mechanism.
    """
    # Single protein, gene-coded. Default for OT-direct lookups. Id =
    # Ensembl gene id (ENSG…) or HGNC symbol.
    PROTEIN = "protein"
    # Cell-surface antigen used as a homing beacon by the compound
    # (CD19 for CAR-T, CD20 for rituximab). Gene-coded too — distinct
    # from PROTEIN only in that the binding mode is recognition-and-
    # destroy rather than inhibition/agonism. Id = HGNC symbol.
    ANTIGEN = "antigen"
    # Macromolecular complex or structural target without a clean gene
    # id: DNA, microtubule, ribosome. Id = ChEBI (`CHEBI:…`) or GO
    # cellular_component (`GO:…`).
    MACROMOLECULE = "macromolecule"
    # Specific cell population or tissue type — for cell-directed
    # therapies that have no specific receptor (rare but real:
    # GCSF→neutrophils when CSF3R isn't the load-bearing target). Id =
    # Cell Ontology (`CL:…`).
    CELL_POPULATION = "cell_population"


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
    # Intervention → Target. The intervention affects (binds, irradiates,
    # recognizes, etc.) a biological target. Belief = P(this intervention
    # measurably affects this target). For drug/biological interventions,
    # this is classical molecular binding. For radiation, cell therapy,
    # devices, etc., "affects" is the general action — the edge type
    # doesn't presume binding chemistry. (Renamed from `binds_to` in
    # round 3.3 to reflect the broader intervention namespace.)
    AFFECTS = "affects"
    MODULATES_VIA = "modulates_via"
    MECHANISM_AFFECTS = "mechanism_affects"
    BIOLOGY_DRIVES = "biology_drives"
    REFLECTS_BIOLOGY = "reflects_biology"
    ENDPOINT_CAPTURES = "endpoint_captures"
    RESPONDS_DIFFERENTLY = "responds_differently"
    # Structural edge: links a synthesized combo InterventionNode to each
    # of its constituent InterventionNodes. Carries no Beta belief — used
    # purely to make the combo's composition queryable from the graph.
    COMPOSED_OF = "composed_of"
    # Intervention → AdverseEvent. Belief = P(this intervention causes this AE).
    # Evidence: per-trial incidence-rate deltas vs control arm.
    CAUSES_AE = "causes_ae"
    # Target → AdverseEvent. Belief = P(modulating this target causes this AE).
    # Evidence: ≥2 distinct interventions affecting the target with strong
    # causes_ae to the same AE — the cross-trial signal that an AE is
    # on-mechanism rather than intervention-specific.
    TARGET_ASSOCIATED_AE = "target_associated_ae"
    # Subtype IndicationNode → parent IndicationNode. Structural (no Beta
    # belief), like composed_of. Lets cross-indication queries roll up
    # specific subtypes (uveal_melanoma, mucosal_melanoma, ...) to a
    # parent disease (melanoma) when the question is "what do we know
    # about melanoma in general" rather than a specific subtype.
    # Round 3.2 (#11) added this; populated from a hand-curated
    # hierarchy in indication_taxonomy._INDICATION_HIERARCHY.
    SUBTYPE_OF = "subtype_of"
    # Compound-combination conditioning. The source node's presence in a
    # regimen modulates the efficacy of the target node in another
    # constituent's chain. Endpoints are typed across {Compound, Target,
    # Mechanism, Biology}. v0.2.0 emits at the compound→compound layer
    # from arm-differential evidence; later stages (heuristic resolver,
    # LLM mapping) promote edges to the chain layer where biology
    # actually operates. See `audit/round_8_architecture_design.md`.
    MODULATES_EFFICACY_OF = "modulates_efficacy_of"


# Node types that can be endpoints of a MODULATES_EFFICACY_OF edge.
MODULATION_ENDPOINT_NODE_TYPES: frozenset[str] = frozenset({
    "CompoundNode",      # legacy snapshots
    "InterventionNode",  # current compound representation
    "TargetNode",
    "MechanismNode",
    "BiologyNode",
})


def canonical_modulation_endpoints(
    src_id: str, tgt_id: str,
) -> tuple[str, str]:
    """Return (src, tgt) in lex order.

    Use only for **symmetric** modulation edges — primarily the
    compound→compound case at v0.2.0, where ``A modulates_efficacy_of B``
    and ``B modulates_efficacy_of A`` represent the same combination
    fact and should accumulate evidence on a single edge.

    Do not call this for cross-layer edges (e.g. compound → mechanism)
    emitted by the v0.2.1 heuristic resolver or v0.3.0 LLM classifier —
    those carry semantic direction (modulator → modulated) and must
    preserve it.
    """
    if src_id <= tgt_id:
        return src_id, tgt_id
    return tgt_id, src_id


class EvidenceType(str, Enum):
    CLINICAL_PHASE3 = "clinical_phase3"
    CLINICAL_PHASE2 = "clinical_phase2"
    CLINICAL_PHASE1 = "clinical_phase1"
    GENETIC_MR = "genetic_mr"
    GENETIC_GWAS = "genetic_gwas"
    PRECLINICAL_IN_VIVO = "preclinical_in_vivo"
    PRECLINICAL_IN_VITRO = "preclinical_in_vitro"
    # ── Round-25: per-source curated-database evidence types ───────────
    #
    # The populator no longer hand-sets non-Beta(1, 1) priors. Each
    # curated-source emission carries one of these source types so the
    # evidence-strength accounting and trust weighting reflect WHAT the
    # source is, not just THAT it's curated.
    #
    # n_eff values (see EVIDENCE_TYPE_N_EFF) are picked from the
    # CHARACTER of each source — curation depth, primary-vs-aggregate,
    # known replication behavior — NOT tuned against the holdout audit.
    # All are flagged as starting defaults pending real calibration on
    # a ≥50-trial labeled set per calibration.py.

    # Drug→target binding curated by Open Targets (aggregates ChEMBL,
    # IUPHAR, DGIdb, drug labels). Primary assertion, multi-source
    # cross-referenced.
    DATABASE_OT_DIRECT = "database_ot_direct"
    # Drug→target inferred from ChEMBL's structured action_type field.
    # Single primary source but well-curated.
    DATABASE_CHEMBL = "database_chembl"
    # Hand-curated antibody→target table (round-24 mAb resolver).
    # Backed by primary literature; each entry vetted by us.
    DATABASE_MAB_TABLE = "database_mab_table"
    # OT target-disease association score. Aggregated across multiple
    # evidence types (clinical, genetic, somatic, literature) but the
    # score itself is a heuristic combination — weaker per-source than
    # the primary entries.
    DATABASE_OT_ASSOCIATION = "database_ot_association"
    # Reactome / GO pathway-membership assertions. Curated by pathway
    # database curators; a single curator's interpretive call per
    # entry.
    DATABASE_REACTOME_GO = "database_reactome_go"
    # LINCS L1000 perturbation signature. Real in-vitro experiment;
    # weighted same as PRECLINICAL_IN_VITRO (which is what it is).
    DATABASE_LINCS = "database_lincs"
    # Endpoint-class regulatory prior (OS / DFS / PFS / etc.). FDA /
    # ICH consensus on what endpoints capture clinical benefit per
    # disease class. Aggregate but well-established.
    DATABASE_ENDPOINT_PRIOR = "database_endpoint_prior"
    # Indication-taxonomy structural relationship (e.g. "metastatic
    # melanoma" rolls up to "melanoma"). Structural, not measurement
    # — weaker.
    DATABASE_INDICATION_TAXONOMY = "database_indication_taxonomy"
    # Cross-reference name-match (gene symbol or target name found in
    # intervention text). Heuristic, not really curation; lowest tier.
    DATABASE_CROSS_REFERENCE = "database_cross_reference"
    # Synthesized fallback edges (trial_biology_fallback synthetic
    # biology slug, combo_inherit AFFECTS). Derived from existing
    # curated facts but one inferential step removed.
    DATABASE_FALLBACK = "database_fallback"

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

class InterventionNode(BaseModel):
    """A trial intervention. Renamed from `CompoundNode` in round 3.3.

    Covers every intervention type CT.gov tracks: small-molecule drugs,
    biologicals, radiation, devices, cell/gene therapies, procedures,
    behavioral interventions, and combinations. Drug-specific fields
    (chembl_id, smiles, modality, targets_claimed) stay as optional
    metadata — they're None for non-drug interventions.

    `intervention_type` is the canonical CT.gov type and drives chain
    fan-out: only DRUG/BIOLOGICAL/COMBINATION primaries get a full
    `affects → Target → mechanism → biology → indication` backbone.
    Other types are first-class graph citizens (they accumulate AE
    evidence, appear in arm rosters, can be queried) but their chains
    land on UNKNOWN target/mechanism until proper non-drug edge types
    are added.
    """
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    intervention_type: InterventionType = InterventionType.UNKNOWN
    modality: Modality | None = None
    # ChEMBL identifier (e.g. "CHEMBL1201580" for nivolumab) when known.
    # Resolved via Open Targets' drug search; None for non-drug
    # interventions and for synthesized combo regimens that don't
    # correspond to a single ChEMBL entry.
    chembl_id: str | None = None
    # Brand names, INN aliases, code names ("BMS-936558", "ONO-4538",
    # "Opdivo") stored alongside the canonical name and the ChEMBL id
    # so downstream queries can match on whatever name a paper /
    # classifier / label happened to use.
    aliases: list[str] = Field(default_factory=list)
    smiles: str | None = None
    targets_claimed: list[str] = Field(default_factory=list)
    # Round-21: external stable identifier for canonicalization.
    # ChEMBL primarily ("CHEMBL185"); will be extended to UMLS CUI /
    # RxNorm if more external DBs get pulled in. Same drug across
    # different CT.gov intervention strings ("5-Fluorouracil" vs
    # "Fluorouracil") should share a stable_id so the Phase B
    # canonicalization can consolidate on it.
    stable_id: str | None = None
    # SapBERT (or equivalent clinical-domain) embedding of the
    # canonical name. Used by the Phase B canonicalization layer for
    # vector-similarity lookup when stable_id isn't available (cell
    # therapies, vaccines, novel agents that aren't in ChEMBL but
    # SapBERT can still embed by name).
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Backwards-compat alias. Code referencing `CompoundNode` continues to
# work; new code should use `InterventionNode`. Remove this once all
# call sites have migrated.
CompoundNode = InterventionNode


class TargetNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # For PROTEIN / ANTIGEN targets: the HUGO symbol (PDCD1, CD19).
    # For MACROMOLECULE / CELL_POPULATION targets there's no gene, so
    # `gene_symbol` holds the namespaced id ("CHEBI:16991" for DNA,
    # "CL:0000169" for pancreatic β-cell) — preserving the field as a
    # human-readable handle without forcing the populator to invent a
    # fake symbol.
    gene_symbol: str = Field(min_length=1)
    # See ``TargetType`` — preserves the scale distinction (target =
    # what the compound physically engages, not the action and not the
    # downstream pathway). Default ``PROTEIN`` for back-compat with the
    # pre-target-namespace-expansion graph; new non-protein TargetNodes
    # set this explicitly.
    target_type: TargetType = TargetType.PROTEIN
    # Colloquial / clinical-use names ("PD-1", "PDL1", "B7-H1") stored
    # alongside the canonical Ensembl id and HUGO gene_symbol so the
    # graph carries the names a clinician or LLM would actually emit,
    # not just the database-canonical forms. Auto-populated from HGNC
    # when the resolver is loaded.
    aliases: list[str] = Field(default_factory=list)
    druggability_score: float | None = None
    tissue_expression: dict[str, float] = Field(default_factory=dict)
    essentiality_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MechanismNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mechanism_type: MechanismType
    selectivity: str | None = None
    # Round-28 directionality metadata. Populated from a table keyed on
    # MechanismCategory in src/graph/populate.py. Values:
    #   "inhibiting"  — the mechanism reduces the target's activity / a
    #                   downstream process (kinase inhibition, receptor
    #                   antagonism, angiogenesis inhibition, ...).
    #   "activating"  — the mechanism increases the target's activity /
    #                   a downstream process (immune costimulation,
    #                   antigen-directed cytotoxicity, ...).
    #   "modulating"  — bidirectional or context-dependent (hormone
    #                   modulation, gene editing).
    #   None          — unknown / OTHER. Backward-compatible default.
    # The prediction math is direction-blind: P(success) reflects "edge
    # operates" not "edge benefits the patient." Surfacing direction as
    # metadata lets audits and graphguard flag chains where the math is
    # being read as benefit-direction even though the underlying
    # mechanism is inhibitory.
    direction: str | None = None
    # Abstraction-ladder redesign: the coarse drug-class functional ontology
    # (a ``MechanismCategory`` value — checkpoint_blockade, kinase_inhibition,
    # …) carried as METADATA rather than as the node's only identity signal.
    # The node's identity / scale is the SPECIFIC molecular action (its
    # ``description``, sourced from the extracted ``mechanism_description``);
    # ``category`` is the bucket it rolls up into, used by audits and by the
    # round-28 direction/type derivation. Populated from the classified /
    # extracted ``mechanism_category`` (see populate._populate_trial_mechanisms);
    # ``None`` when no category was supplied. Backward-compatible default so
    # existing snapshots / cached graphs load unchanged.
    category: str | None = None
    # A.0: rich free-text description preserved from the trial's therapeutic
    # hypothesis (proposed_mechanism). The id is a routing tag; this is the
    # semantic substrate the BioLORD embedding work (A.1) consumes. First
    # non-empty contributor wins (see populate._set_node_description);
    # per-trial provenance is a later (A.2) refinement. Stripped from public
    # snapshots only once it holds a *fine-tuned* vector — the text itself is
    # public. See future_ideas/eroom_node_graph_kickoff.md (A.0).
    description: str = ""


class BiologyNode(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    pathway_ids: list[str] = Field(default_factory=list)
    tissue_specificity: list[str] = Field(default_factory=list)
    known_redundancies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # A.0: rich free-text description from the trial's intended_biology.
    # See MechanismNode.description.
    description: str = ""


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
    # A.0: rich free-text description — the subgroup's raw descriptor, or the
    # trial's target_population for the parent enrollment cohort. The embedding
    # substrate for mechanism-conditioned population sub-regions (the named
    # structural fix for the bevacizumab AVANT holdout miss). See
    # MechanismNode.description.
    description: str = ""

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
    # Accumulates the raw CT.gov condition strings, stages, subtypes, and
    # other qualifiers observed across trials that canonicalize to this
    # IndicationNode. Populated by the population pipeline; downstream
    # readers can use it to recover the trial-specific phrasing for a
    # canonical disease.
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    # Round-29: did any trial report this AE as a Serious Adverse Event?
    # CT.gov's `serious=True` flag is populated for ~90% of extracted
    # AEs even when CTCAE grade is missing (CT.gov rarely posts grades).
    # The safety-penalty math uses this as a COARSE severity floor
    # (≈ grade-3 weight) so an ungraded cardiac SAE doesn't fall to
    # `_UNKNOWN_GRADE_WEIGHT=0.10`. OR-merged across trials reporting
    # the same AE — any "serious=true" trial wins.
    serious: bool = False
    # Round-28 MedDRA hierarchy parents. Populated by
    # src/annotation/meddra_hierarchy.py at AE-node creation time.
    # Used by the SOC-tier `target_associated_ae` propagation so sibling
    # compounds binding the same target aggregate at the SOC level even
    # when their per-trial PT extractions land at disjoint terms.
    #   - ``soc_id`` is a slug like "cardiac_disorders" (lower-snake).
    #   - ``soc_name`` mirrors ``system_organ_class`` (the original
    #     MedDRA string) for human-readable rendering.
    #   - ``hlt_id`` / ``hlgt_id`` are intermediate hierarchy tiers;
    #     populated only when the curated hierarchy file has them.
    # All four default to empty strings for backward compat with
    # pre-round-28 snapshots.
    hlt_id: str = ""
    hlgt_id: str = ""
    soc_id: str = ""
    soc_name: str = ""
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

    # ── Principled-N_eff inputs (optional; default None = back-compat) ─────
    # Quantitative properties of the evidence, consumed by the precision-
    # aware n_eff path in ``src/inference/beliefs.py`` when the
    # ``EROOM_NEFF_PRECISION`` flag is enabled. All default None so existing
    # records and serialized snapshots load unchanged and reproduce the
    # legacy type-constant n_eff exactly.
    #   ``n_obs``      patient/observation count (trial enrollment / N)
    #   ``effect``     reported point-estimate effect size (HR/OR/Δ), if any
    #   ``p_value``    reported p-value, if any
    #   ``cluster_key`` correlation-cluster id for the independence/redundancy
    #                   discount at aggregation; records sharing a key are
    #                   treated as non-independent (None = its own cluster)
    n_obs: int | None = None
    effect: float | None = None
    p_value: float | None = None
    cluster_key: str | None = None

    # ── A.3 manifold-2 localization (optional; default = scalar-only path) ──
    # The trial-specific source/target descriptions (A.0b per-chain, else the
    # trial-level A.0a text) whose BioLORD embeddings place this record at a
    # point (s, t) on the edge belief surface. Descriptions are public text;
    # the embeddings are the private localization — stripped from public
    # snapshots via the ``*_embedding`` suffix (src/boundary.py). Absent ⇒ the
    # record updates only the scalar marginal, exactly as today.
    source_description_in_trial: str = ""
    target_description_in_trial: str = ""
    source_embedding: list[float] | None = None
    target_embedding: list[float] | None = None


class EdgeBeliefState(BaseModel):
    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    # A.3: optional per-region belief field (manifold 2). The scalar (alpha,
    # beta) above stays the PUBLIC marginal; this holds the (s,t)-localized
    # anchor surface — the private "edge weights" (statements 2 & 4). Serialized
    # as an opaque dict (``belief_field.BeliefField.to_dict``); the
    # ``belief_field`` field name is private (src/boundary.py), so it is
    # stripped from public snapshots and written only to private ones. None ⇒
    # scalar-only (back-compat).
    belief_field: dict[str, Any] | None = None

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

    A trial produces N arms × M reported subgroups × E endpoints chains;
    every constituent compound of every arm fans out a chain. Whether a
    compound is "primary" vs supportive in a clinical sense isn't
    represented here — that distinction needs explicit extractor-side
    schema (primary_compounds / supportive_compounds lists from the LLM)
    rather than a populator-side heuristic.
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
_GO_PATTERN = re.compile(r"^GO:\d{7}$")
_DRUGBANK_PATTERN = re.compile(r"^DB\d{5,}$")
# Namespaced ids accepted on non-protein TargetNodes (target-namespace
# expansion, 2026-05-18). HGNC for gene-coded antigens when Ensembl
# isn't available; ChEBI for macromolecular structures (DNA, tubulin);
# Cell Ontology for cell populations.
_HGNC_PATTERN = re.compile(r"^HGNC:\d+$")
_CHEBI_PATTERN = re.compile(r"^CHEBI:\d+$")
_CL_PATTERN = re.compile(r"^CL:\d{7}$")
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

    if node_type in ("InterventionNode", "CompoundNode"):
        # Accept either name during the round-3.3 rename window. New code
        # uses InterventionNode; CompoundNode aliases through for back-compat.
        if _DRUGBANK_PATTERN.match(raw):
            return raw
        slug = _slugify_lower(raw)
        if not slug:
            raise ValueError(f"Intervention name '{name}' could not be normalized")
        return slug

    if node_type == "TargetNode":
        # Namespace-prefixed ids pass through verbatim. Ensembl for
        # proteins (default), HGNC when Ensembl isn't resolved, ChEBI
        # for macromolecular structures (DNA, tubulin), GO for cellular-
        # component targets, Cell Ontology for cell populations. See
        # TargetType for the type-by-namespace convention.
        if (
            _ENSG_PATTERN.match(raw)
            or _HGNC_PATTERN.match(raw)
            or _CHEBI_PATTERN.match(raw)
            or _GO_PATTERN.match(raw)
            or _CL_PATTERN.match(raw)
        ):
            return raw
        # Bare gene symbols (legacy path — round-3.3 normalize behavior)
        # uppercase + alnum-strip. Used when the caller couldn't resolve
        # to Ensembl but knows the symbol.
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
        # Three valid id forms:
        #   1. Reactome stable ID (canonical, ground truth from LINCS).
        #   2. Gene Ontology term id (``GO:NNNNNNN``), used when a gene is
        #      sparsely represented in Reactome but has rich GO biological-
        #      process annotations (CRBN being the canonical example).
        #      Resolved via QuickGO; see src/ingestion/quickgo.py.
        #   3. ``{mechanism_category}__{indication_slug}`` synthetic slug,
        #      used as a fallback when neither Reactome nor GO has useful
        #      annotations. Lets the prediction engine traverse the full
        #      chain even without external data.
        if _REACTOME_PATTERN.match(raw):
            return raw
        if _GO_PATTERN.match(raw):
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
            f"(e.g. R-HSA-9006934), a Gene Ontology term id (e.g. "
            f"GO:0016567), or a '{{mechanism}}__{{indication}}' slug"
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
