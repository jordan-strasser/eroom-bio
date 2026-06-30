"""Belief-state cross-indication CENSUS — the north-star thesis probe (public).

Structural, read-only analysis over the PUBLIC belief-state: which ``BiologyNode``s
/ ``mechanism_affects`` edges bridge ≥2 distinct canonical indications (the
cross-indication-transfer claim in CLAUDE.md). This is the public half of the old
``provenance.py`` — it touches only the graph + belief-state (no prediction engine),
so it stays open-core when the per-holdout deciding-edge TRACE (the frontier half,
which re-runs the query aggregation) relocates to ``eroom-enterprise``.

Used by ``scripts/bridge_census.py`` and ``scripts/cross_indication_census.py``.
Dependencies flow one way: this module imports only graph / belief-state; the
frontier trace in ``provenance.py`` imports FROM here, never the reverse.
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field

from src.graph.indication_taxonomy import parent_indication_for
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore

# ── Indication → therapeutic area ────────────────────────────────────────
# Oncology is detected by keyword (the corpus is 90% onco, with a long tail
# of distinct cancers we don't want to enumerate). Everything else is mapped
# from a small curated table; unknown non-onco slugs fall through to "other".
# The ONLY distinction the north-star claim rests on is oncology vs not —
# the finer areas are for grouping the readout.
_ONCOLOGY_KEYWORDS = (
    "cancer", "carcinoma", "leukemia", "lymphoma", "sarcoma", "melanoma",
    "tumor", "glioma", "myeloma", "blastoma", "mesothelioma", "neoplasm",
    "malignan", "metasta", "astrocytoma", "mycosis_fungoides", "squamous",
    "germ_cell", "ewing", "wilms", "adenocarcinoma", "oncology",
)

_AREA_BY_SLUG: dict[str, str] = {
    # cardiometabolic / lipid
    "hypercholesterolemia": "cardiometabolic",
    "familial_hypercholesterolemia": "cardiometabolic",
    "familial_hypercholesterolemia_homozygous": "cardiometabolic",
    "dyslipidemia": "cardiometabolic",
    "atherosclerotic_cardiovascular_disease": "cardiometabolic",
    "cardiovascular_diseases": "cardiometabolic",
    "coronary_heart_disease": "cardiometabolic",
    "polycystic_ovary_syndrome": "endocrine_metabolic",
    # neuro / cognition / pain
    "alzheimers_disease": "neuro",
    "cognitive_impairment": "neuro",
    "epilepsy": "neuro",
    "pain": "neuro",
    "arthralgia": "neuro",
    "familial_amyloid_polyneuropathy": "neuro",
    # infectious
    "hepatitis_c": "infectious",
    "condylomata_acuminata": "infectious",
    # endocrine / urologic / gyn
    "gastrinoma": "endocrine_metabolic",
    "pheochromocytoma": "endocrine_metabolic",
    "uterine_fibroids": "gyn_urologic",
    "lower_urinary_tract_symptoms": "gyn_urologic",
    # derm
    "actinic_keratosis": "dermatology",
    "basal_cell_nevus_syndrome": "dermatology",
    # gastro
    "crohns_disease": "gastro_immune",
    "gastric_ulcer": "gastro_immune",
    # non-malignant heme (borderline; kept distinct from oncology)
    "polycythemia_vera": "hematology_nonmalignant",
    "myelodysplastic_syndromes": "hematology_nonmalignant",
    "neutropenia": "hematology_nonmalignant",
    "chronic_myeloproliferative_disorder": "hematology_nonmalignant",
}

# Slugs that aren't real diseases — tox descriptors, procedures, or LLM
# non-answers. Surfaced (never silently dropped) but excluded from the
# "clean" cross-area headline.
_NON_DISEASE_SLUGS: frozenset[str] = frozenset(
    {
        "chemotherapeutic_agent_toxicity",
        "homologous_transplantation",
        "catheter_site_discomfort",
        "xerostomia",
        "fgfr_gene_amplification",  # a biomarker, not a disease
    }
)

_NCT_RE = re.compile(r"^NCT\d{8}$")


def is_oncology(slug: str) -> bool:
    """True if the indication slug names a cancer."""
    return any(k in slug for k in _ONCOLOGY_KEYWORDS)


def therapeutic_area(slug: str) -> str:
    """Coarse therapeutic area for an indication slug.

    Oncology first (keyword), then the curated table, then "other". Used to
    decide whether a biology bridge spans *areas* (the north-star claim) vs
    merely spanning cancer subtypes.
    """
    if is_oncology(slug):
        return "oncology"
    return _AREA_BY_SLUG.get(slug, "other")


def is_non_disease(slug: str) -> bool:
    """True for tox/procedure/non-answer slugs that shouldn't count as a
    real cross-indication endpoint."""
    if slug in _NON_DISEASE_SLUGS:
        return True
    # LLM non-answers tend to be very long sentence-like slugs.
    return slug.startswith("i_cannot_determine") or len(slug) > 80


def canonical_indication(slug: str) -> str:
    """Roll a subtype slug up to its parent disease (so melanoma +
    uveal_melanoma don't read as a bridge), else return it unchanged."""
    return parent_indication_for(slug) or slug


def id_scheme(node_id: str) -> str:
    """Classify a BiologyNode id by source vocabulary."""
    if node_id.startswith("R-") and re.match(r"^R-[A-Z]{3}-\d+$", node_id):
        return "reactome"
    if node_id.startswith("GO:"):
        return "go"
    if node_id.startswith("bio:"):
        return "embedded"
    if "__" in node_id:
        return "synthetic"  # {mechanism}__{indication}: indication-specific
    return "other"


# ── Result models ────────────────────────────────────────────────────────


class IndicationSide(BaseModel):
    """One indication a biology node drives, with that edge's belief."""

    indication_id: str
    area: str
    is_oncology: bool
    is_non_disease: bool
    expected_probability: float
    evidence_strength: float  # alpha + beta - 2 (effective observations)
    n_records: int
    source_ncts: list[str] = Field(default_factory=list)
    raw_indication_ids: list[str] = Field(default_factory=list)


class BiologyBridge(BaseModel):
    """A BiologyNode whose biology_drives edges reach ≥2 canonical
    indications — a candidate cross-indication bridge."""

    biology_id: str
    biology_name: str
    id_scheme: str
    sides: list[IndicationSide]
    n_indications: int
    n_areas: int
    spans_oncology_and_other: bool
    # max E[p] − min E[p] among sides with real evidence (the strong/weak
    # contrast). 0.0 when fewer than two well-evidenced sides exist.
    belief_spread: float
    strongest: IndicationSide | None = None
    weakest: IndicationSide | None = None


class MechanismBridge(BaseModel):
    """A mechanism_affects edge co-updated by trials from ≥2 indications."""

    mechanism_id: str
    mechanism_name: str
    biology_id: str
    biology_name: str
    expected_probability: float
    evidence_strength: float
    indications: list[str]
    areas: list[str]
    spans_oncology_and_other: bool
    # indication -> [source NCTs that contributed evidence from it]
    ncts_by_indication: dict[str, list[str]]


# ── Indexing helpers ─────────────────────────────────────────────────────


def build_nct_indication_index(store: GraphStore) -> dict[str, set[str]]:
    """Map each trial NCT → the set of canonical indications it studied.

    Drawn from the chains in each ``TrialSubgraph`` (every chain carries its
    own ``indication_id``). Used to attribute an edge's evidence records back
    to the disease their source trial ran in.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for trial_id, subgraph in store.trial_subgraphs.items():
        for chain in subgraph.chains:
            index[trial_id].add(canonical_indication(chain.indication_id))
    return dict(index)


def _trial_ncts_from_evidence(belief: EdgeBeliefState) -> list[str]:
    """Source NCTs among an edge's evidence records (skips DB/cross-ref
    sources whose source_id isn't a trial id)."""
    return [r.source_id for r in belief.evidence if _NCT_RE.match(r.source_id)]


# ── Biology bridges (structural) ─────────────────────────────────────────


def find_biology_bridges(
    store: GraphStore,
    *,
    min_indications: int = 2,
    spread_evidence_floor: float = 1.0,
) -> list[BiologyBridge]:
    """Enumerate BiologyNodes driving ≥``min_indications`` canonical
    indications via ``biology_drives``.

    Only ``BiologyNode`` sources count — ``biology_drives`` also fires from
    ``TargetNode`` when the biology collapses to its target, which is a
    weaker claim. ``belief_spread`` is the max−min of ``E[p]`` across sides
    with ``evidence_strength ≥ spread_evidence_floor`` (so a low belief from
    a single weak observation isn't mistaken for a contradiction).

    Returned unsorted; callers rank by ``spans_oncology_and_other`` and
    ``belief_spread``.
    """
    # Group biology_drives edges by their BiologyNode source, collapsing
    # subtype indications to canonical parents (one side per canonical id).
    by_bio: dict[str, dict[str, list[tuple[str, EdgeBeliefState]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for edge in store.get_edges_by_type(EdgeType.BIOLOGY_DRIVES):
        src = store.get_node(edge["source_id"])
        if src.get("node_type") != "BiologyNode":
            continue
        raw_ind = edge["target_id"]
        canon = canonical_indication(raw_ind)
        belief = EdgeBeliefState.model_validate(edge["belief"])
        by_bio[edge["source_id"]][canon].append((raw_ind, belief))

    bridges: list[BiologyBridge] = []
    for bio_id, per_indication in by_bio.items():
        if len(per_indication) < min_indications:
            continue
        node = store.get_node(bio_id)
        sides = [
            _build_side(canon, raw_beliefs)
            for canon, raw_beliefs in per_indication.items()
        ]
        sides.sort(key=lambda s: s.expected_probability, reverse=True)

        areas = {s.area for s in sides}
        evidenced = [
            s for s in sides if s.evidence_strength >= spread_evidence_floor
        ]
        spread = 0.0
        strongest = weakest = None
        if len(evidenced) >= 2:
            strongest = max(evidenced, key=lambda s: s.expected_probability)
            weakest = min(evidenced, key=lambda s: s.expected_probability)
            spread = (
                strongest.expected_probability - weakest.expected_probability
            )

        bridges.append(
            BiologyBridge(
                biology_id=bio_id,
                biology_name=node.get("name", bio_id),
                id_scheme=id_scheme(bio_id),
                sides=sides,
                n_indications=len(sides),
                n_areas=len(areas),
                spans_oncology_and_other=(
                    "oncology" in areas and any(a != "oncology" for a in areas)
                ),
                belief_spread=spread,
                strongest=strongest,
                weakest=weakest,
            )
        )
    return bridges


def _build_side(
    canon: str, raw_beliefs: list[tuple[str, EdgeBeliefState]]
) -> IndicationSide:
    """Aggregate the (possibly several subtype) edges that rolled up to one
    canonical indication into a single side.

    When subtypes collapse, we report the strongest-evidenced edge's belief
    (the canonical disease's best estimate) and union their source NCTs.
    """
    raw_ids = sorted({raw for raw, _ in raw_beliefs})
    # Pick the most-evidenced edge as the representative belief for the side.
    rep_raw, rep_belief = max(
        raw_beliefs, key=lambda rb: rb[1].evidence_strength
    )
    ncts = sorted(
        {n for _, b in raw_beliefs for n in _trial_ncts_from_evidence(b)}
    )
    n_records = sum(len(b.evidence) for _, b in raw_beliefs)
    return IndicationSide(
        indication_id=canon,
        area=therapeutic_area(canon),
        is_oncology=is_oncology(canon),
        is_non_disease=is_non_disease(canon),
        expected_probability=rep_belief.expected_probability,
        evidence_strength=rep_belief.evidence_strength,
        n_records=n_records,
        source_ncts=ncts,
        raw_indication_ids=raw_ids,
    )


# ── Mechanism bridges (evidence provenance) ──────────────────────────────


def find_mechanism_bridges(
    store: GraphStore,
    *,
    min_indications: int = 2,
) -> list[MechanismBridge]:
    """Enumerate ``mechanism_affects`` edges whose evidence records trace to
    source trials in ≥``min_indications`` canonical indications.

    Unlike biology bridges (where the shared *node* is the bridge), here a
    single *edge belief* was co-updated by trials from different diseases —
    the strongest form of cross-indication transfer.
    """
    nct_index = build_nct_indication_index(store)
    bridges: list[MechanismBridge] = []
    for edge in store.get_edges_by_type(EdgeType.MECHANISM_AFFECTS):
        belief = EdgeBeliefState.model_validate(edge["belief"])
        ncts_by_ind: dict[str, list[str]] = defaultdict(list)
        for nct in _trial_ncts_from_evidence(belief):
            for ind in nct_index.get(nct, ()):  # usually one indication
                ncts_by_ind[ind].append(nct)
        if len(ncts_by_ind) < min_indications:
            continue
        mech = store.get_node(edge["source_id"])
        bio = store.get_node(edge["target_id"])
        areas = sorted({therapeutic_area(i) for i in ncts_by_ind})
        bridges.append(
            MechanismBridge(
                mechanism_id=edge["source_id"],
                mechanism_name=mech.get("name", edge["source_id"]),
                biology_id=edge["target_id"],
                biology_name=bio.get("name", edge["target_id"]),
                expected_probability=belief.expected_probability,
                evidence_strength=belief.evidence_strength,
                indications=sorted(ncts_by_ind),
                areas=areas,
                spans_oncology_and_other=(
                    "oncology" in areas and any(a != "oncology" for a in areas)
                ),
                ncts_by_indication={k: sorted(set(v)) for k, v in ncts_by_ind.items()},
            )
        )
    return bridges
