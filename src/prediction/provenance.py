"""Cross-indication provenance analysis — the north-star thesis probe.

Eroom's differentiating claim (CLAUDE.md) is that mechanistic knowledge
*compounds across indications*: a trial in one disease should inform a
prediction in another whenever both route through the same mechanism or
biology. A flat success-classifier structurally cannot do this; the typed
graph can — *if* the Reactome/GO/embedded biology abstraction is general
enough to place chains from different diseases on the same node.

This module tests that directly, two ways:

1. **Structural biology bridges** — a single ``BiologyNode`` with
   ``biology_drives`` edges to ≥2 *distinct canonical indications*. The
   shared node IS the bridge; each outgoing edge carries its own Beta
   belief, so we can compare ``E[p]`` across indications. The headline
   finding (and the X-post gold) is a biology that is **STRONG** in one
   disease and **WEAK / contradicted** in another — same pathway, opposite
   clinical fate by indication.

2. **Evidence-provenance mechanism bridges** — a single
   ``mechanism_affects`` edge whose ``EvidenceRecord``s trace to source
   trials in ≥2 indications. Here *one belief* was literally co-updated by
   trials from different diseases — the strongest possible form of transfer.

Only ``BiologyNode``-sourced ``biology_drives`` edges count for (1): the
edge type also fires from ``TargetNode`` when the biology collapses to its
target, and a *target* shared across cancers (one KDR across tumors) is far
less surprising than a shared *biology process*. We keep the analysis honest
by typing every node and tagging every indication with a therapeutic area,
so "oncology ↔ oncology-subtype" bridges don't masquerade as the
cross-area transfer the thesis is really about.

Reusable from ``scripts/bridge_census.py`` (the census) and
``scripts/trace_provenance.py`` (per-holdout deciding-edge trace).
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field

from src.graph.indication_taxonomy import parent_indication_for
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.inference.beliefs import (
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
    p_obs_for_bucket,
    redundancy_factor,
)
from src.prediction.path_query import (
    EdgeContribution,
    PredictionResult,
    _aggregate_samples,
    _sample_edge,
    _trust_weight,
    predict_clinical_hypothesis,
)

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


# ── Per-holdout deciding-edge trace ──────────────────────────────────────
#
# The census (above) asks "what bridges exist in the graph". The trace asks
# the prediction-side question the X-post actually needs: for ONE trial's
# prediction, what is the deciding (weakest-link) edge, and is it supported
# by evidence from OTHER indications? That makes "we predicted this trial's
# fate from unrelated-indication trials" concrete and auditable.


class EdgeProvenance(BaseModel):
    """One chain edge of a prediction, with its evidence split by where the
    supporting trials came from relative to the trial being predicted."""

    source_id: str
    target_id: str
    source_name: str
    target_name: str
    edge_type: str
    expected_probability: float
    evidence_strength: float
    bottleneck_score: float
    is_deciding: bool  # the prediction's weakest_link
    n_records: int
    self_ncts: list[str]  # records contributed by the predicted trial itself
    same_indication_ncts: list[str]  # other trials in the same indication
    # other-indication NCTs that updated this edge, keyed by their indication.
    # THIS is the cross-indication transfer — non-empty means the edge's
    # belief was (partly) learned from a different disease.
    cross_indication_ncts: dict[str, list[str]]
    n_database_records: int  # non-trial (DB/cross-ref) evidence, for context
    # Edge belief recomputed with the predicted trial's OWN records removed.
    # evidence_strength 0.0 ⇒ the edge was supported only by this trial (no
    # cross-trial transfer); > 0 with a meaningful E[p] ⇒ other trials carry it.
    self_excluded_expected_probability: float
    self_excluded_evidence_strength: float


class HoldoutTrace(BaseModel):
    """A single trial's prediction with deciding-edge provenance."""

    nct: str
    indications: list[str]
    compound_id: str
    hypothesis: str
    overall_probability: float
    efficacy_probability: float
    safety_penalty: float
    deciding_edge: EdgeProvenance | None
    edges: list[EdgeProvenance]
    # True iff the deciding edge carries evidence from another indication
    # (the headline cross-indication claim).
    deciding_edge_has_cross_indication: bool
    # Efficacy re-aggregated with THIS trial's own evidence removed from every
    # edge — "what the graph predicts from other trials alone". None if no edge
    # survives self-exclusion. The honest X-post number.
    self_excluded_efficacy: float | None
    self_excluded_n_edges: int


def _replay_records(records: list, edge_type_value: str) -> EdgeBeliefState:
    """Replay a set of evidence records from Beta(1, 1) through the
    attribution-time conjugate update (mirrors
    ``GraphStore.get_edge_belief_conditioned``). Returns the resulting belief."""
    out = EdgeBeliefState(alpha=1.0, beta=1.0)
    cluster_seen: dict[str, int] = {}
    for ev in records:
        if ev.applied_n_eff is not None:
            # Faithful replay: the EXACT weights this record applied to the
            # scalar (incl. the explaining-away split + redundancy). A nominal
            # recompute would over-count, so the delta-adjusted LOO would
            # over-remove a high-belief edge's contribution.
            n_eff = ev.applied_n_eff
            p_obs = (ev.applied_p_obs if ev.applied_p_obs is not None
                     else p_obs_for_bucket(SupportBucket(ev.support)))
        else:
            # Legacy record (pre-persistence): recompute nominal × redundancy.
            eff_key = ev.cluster_key or ev.source_id
            prior_same = cluster_seen.get(eff_key, 0)
            n_eff = effective_n_for_evidence(
                ev.source_type, ev.quality_score, n_obs=ev.n_obs,
                edge_type=edge_type_value,
            ) * redundancy_factor(prior_same)
            cluster_seen[eff_key] = prior_same + 1
            p_obs = p_obs_for_bucket(SupportBucket(ev.support))
        out = apply_virtual_evidence(out, n_eff=n_eff, p_obs=p_obs)
    return out


def _belief_excluding(
    belief: EdgeBeliefState, exclude_nct: str, edge_type_value: str
) -> EdgeBeliefState:
    """The edge belief with ``exclude_nct``'s records removed — the belief the
    graph would hold had this trial never run (the basis for the honest
    "predicted from OTHER indications alone" claim).

    Computed by **delta-adjust** (the validated method in
    ``src/inference/belief_tuning.py``): anchor on the STORED belief and remove
    only the replayed contribution of the dropped records, rather than replaying
    everything from scratch. This is exact when nothing is removed and robust to
    drift between the n_eff constants this snapshot was built with and the
    current code (a from-scratch replay over-counts when constants have since
    changed). Floors at Beta(1, 1) when the trial was ~all the edge's evidence."""
    kept = [ev for ev in belief.evidence if ev.source_id != exclude_nct]
    if len(kept) == len(belief.evidence):
        # nothing removed → self-excluded belief IS the stored belief, exactly
        return EdgeBeliefState(
            alpha=belief.alpha, beta=belief.beta, evidence=belief.evidence
        )
    full = _replay_records(belief.evidence, edge_type_value)
    excl = _replay_records(kept, edge_type_value)
    alpha = belief.alpha + (excl.alpha - full.alpha)
    beta = belief.beta + (excl.beta - full.beta)
    if alpha < 1.0 or beta < 1.0 or (alpha + beta - 2.0) <= 0.0:
        # removing this trial drops the edge below its prior → unobserved
        return EdgeBeliefState(alpha=1.0, beta=1.0, evidence=kept)
    return EdgeBeliefState(alpha=alpha, beta=beta, evidence=kept)


def _self_excluded_efficacy(
    contributions: list[EdgeContribution], exclude_nct: str, n_samples: int
) -> tuple[float | None, int]:
    """Re-aggregate the chain's efficacy with the predicted trial's own
    evidence removed from every edge. Edges left with no evidence are dropped
    (exactly as the engine drops Beta(1, 1) edges), so the result reflects only
    what OTHER trials taught the graph. Returns (efficacy, n_surviving_edges);
    efficacy is None when no edge survives."""
    rng = np.random.default_rng()
    samples: list = []
    weights: list = []
    for c in contributions:
        b = _belief_excluding(c.belief, exclude_nct, c.edge_type.value)
        if b.evidence_strength <= 0.0:
            continue
        samples.append(_sample_edge(rng, b, n_samples))
        weights.append(_trust_weight(b))
    if not samples:
        return None, 0
    agg = _aggregate_samples(samples, weights)
    return float(np.mean(agg)), len(samples)


def _edge_provenance(
    contribution: EdgeContribution,
    *,
    store: GraphStore,
    nct_index: dict[str, set[str]],
    self_nct: str,
    self_indications: set[str],
    is_deciding: bool,
) -> EdgeProvenance:
    """Split one edge's evidence into self / same-indication / cross-indication
    trial sources (plus a count of non-trial DB records)."""
    self_ncts: list[str] = []
    same_ind: set[str] = set()
    cross: dict[str, list[str]] = defaultdict(list)
    n_db = 0
    for rec in contribution.belief.evidence:
        src = rec.source_id
        if not _NCT_RE.match(src):
            n_db += 1
            continue
        if src == self_nct:
            self_ncts.append(src)
            continue
        src_inds = nct_index.get(src, set())
        if src_inds - self_indications:  # at least one different indication
            for ind in src_inds - self_indications:
                cross[ind].append(src)
        else:
            same_ind.add(src)
    excluded = _belief_excluding(
        contribution.belief, self_nct, contribution.edge_type.value
    )
    return EdgeProvenance(
        source_id=contribution.source_id,
        target_id=contribution.target_id,
        source_name=store.get_node(contribution.source_id).get("name", contribution.source_id),
        target_name=store.get_node(contribution.target_id).get("name", contribution.target_id),
        edge_type=contribution.edge_type.value,
        expected_probability=contribution.belief.expected_probability,
        evidence_strength=contribution.belief.evidence_strength,
        bottleneck_score=contribution.bottleneck_score,
        is_deciding=is_deciding,
        n_records=len(contribution.belief.evidence),
        self_ncts=sorted(set(self_ncts)),
        same_indication_ncts=sorted(same_ind),
        cross_indication_ncts={k: sorted(set(v)) for k, v in cross.items()},
        n_database_records=n_db,
        self_excluded_expected_probability=excluded.expected_probability,
        self_excluded_evidence_strength=excluded.evidence_strength,
    )


def trace_holdout(
    store: GraphStore,
    nct: str,
    *,
    n_samples: int = 4000,
) -> HoldoutTrace:
    """Predict trial ``nct`` (faithfully, via ``predict_clinical_hypothesis``)
    and trace the deciding edge's evidence by source indication.

    When the trial has several arms, the arm with the LOWEST overall
    probability is reported — the most decisive chain, the one that drives a
    failure call. Raises ``KeyError`` if the trial isn't in the graph.
    """
    subgraph = store.trial_subgraphs.get(nct)
    if subgraph is None:
        raise KeyError(f"Trial '{nct}' not in graph.trial_subgraphs")

    nct_index = build_nct_indication_index(store)
    self_indications = nct_index.get(nct, set())

    # Distinct (compound, indication) hypotheses this trial poses; predict each.
    seen: set[tuple[str, str]] = set()
    results: list[tuple[str, str, PredictionResult]] = []
    for chain in subgraph.chains:
        key = (chain.compound_id, chain.indication_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            res = predict_clinical_hypothesis(
                store, chain.compound_id, chain.indication_id, n_samples=n_samples
            )
        except KeyError:
            continue
        if res.edge_contributions:
            results.append((chain.compound_id, chain.indication_id, res))

    if not results:
        raise KeyError(f"Trial '{nct}' produced no predictable chain")

    compound_id, _ind, result = min(
        results, key=lambda r: r[2].overall_probability
    )

    decider = result.weakest_link
    decider_key = (
        (decider.source_id, decider.target_id, decider.edge_type)
        if decider is not None
        else None
    )
    edges = [
        _edge_provenance(
            c,
            store=store,
            nct_index=nct_index,
            self_nct=nct,
            self_indications=self_indications,
            is_deciding=(c.source_id, c.target_id, c.edge_type) == decider_key,
        )
        for c in result.edge_contributions
    ]
    deciding = next((e for e in edges if e.is_deciding), None)
    se_efficacy, se_n_edges = _self_excluded_efficacy(
        result.edge_contributions, nct, n_samples
    )
    return HoldoutTrace(
        nct=nct,
        indications=sorted(self_indications),
        compound_id=compound_id,
        hypothesis=result.trial_hypothesis,
        overall_probability=result.overall_probability,
        efficacy_probability=result.efficacy_probability,
        safety_penalty=result.safety_penalty,
        deciding_edge=deciding,
        edges=edges,
        deciding_edge_has_cross_indication=bool(
            deciding and deciding.cross_indication_ncts
        ),
        self_excluded_efficacy=se_efficacy,
        self_excluded_n_edges=se_n_edges,
    )


def deciding_cross_indications(
    trace: HoldoutTrace,
) -> tuple[list[str], set[str], bool]:
    """For a trace's deciding edge: the real (non-tox) cross-indication source
    diseases, their therapeutic areas, and whether holdout+support span ≥2
    areas (the differentiated onco↔other transfer).

    Returns ``([], set(), False)`` when there's no deciding edge or its only
    cross-indication sources are non-disease tox/procedure slugs. Lets callers
    rank the *specific cross-area* demos above the generic chemo backbone
    (DNA-damage / mitotic-arrest supported by ~50 cancers)."""
    de = trace.deciding_edge
    if de is None:
        return [], set(), False
    real = sorted(i for i in de.cross_indication_ncts if not is_non_disease(i))
    cross_areas = {therapeutic_area(i) for i in real}
    holdout_areas = {therapeutic_area(i) for i in trace.indications}
    spans = bool(real) and len(cross_areas | holdout_areas) >= 2
    return real, cross_areas, spans
