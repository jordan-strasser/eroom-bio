"""Compositional path prediction: P(success) ≈ product of edge beliefs along the causal chain."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sp_stats

from src.graph.models import (
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    TrialOutcome,
)
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

# The canonical causal chain edges in order. Each entry maps a pair of
# CausalChain field names to its edge type.
_CAUSAL_CHAIN: list[tuple[str, str, EdgeType]] = [
    ("compound_id", "target_id", EdgeType.AFFECTS),
    ("target_id", "mechanism_id", EdgeType.MODULATES_VIA),
    ("mechanism_id", "biology_id", EdgeType.MECHANISM_AFFECTS),
    ("biology_id", "indication_id", EdgeType.BIOLOGY_DRIVES),
]

# Auxiliary edges that contribute to the prediction
_AUXILIARY_EDGES: list[tuple[str, str, EdgeType]] = [
    ("biology_id", "endpoint_id", EdgeType.REFLECTS_BIOLOGY),
    ("endpoint_id", "indication_id", EdgeType.ENDPOINT_CAPTURES),
    ("subgroup_population_id", "indication_id", EdgeType.RESPONDS_DIFFERENTLY),
]

_DEFAULT_BELIEF = EdgeBeliefState(alpha=1.0, beta=1.0)


# Round-20 calibration: severity-weighted safety penalty.
# The table maps the worst-observed CTCAE grade on an AE node to the
# contribution that AE makes to the soft-or penalty. Grades come from
# AdverseEventNode.severity_range, which the AE attribution step
# accumulates across trials.
_SEVERITY_GRADE_TO_WEIGHT: dict[int, float] = {
    1: 0.05,   # Grade 1 — mild, asymptomatic
    2: 0.05,   # Grade 2 — moderate, minimal intervention
    3: 0.15,   # Grade 3 — severe but not life-threatening
    4: 0.30,   # Grade 4 — life-threatening, urgent intervention
    5: 0.50,   # Grade 5 — death related to AE
}
# Conservative default when the AE node has no severity_range observed.
# Higher than grade 1-2 so a missing grade isn't free-pass safe; lower
# than grade 3 so it doesn't over-penalize on absence of data.
_UNKNOWN_GRADE_WEIGHT = 0.10


def _max_grade_from_severity_range(severity_range: str | None) -> int | None:
    """Parse ``severity_range`` ('1,2,3-5' / 'any,3-4' / '') → max grade.

    The attributor appends every observed grade or grade range to the
    AE node as a comma-separated string. Each piece is either a single
    integer ("3"), a range ("3-5"), or the literal "any" (ignored).
    Returns the maximum integer observed, or None when nothing parsed.
    """
    if not severity_range:
        return None
    max_grade: int | None = None
    for raw in severity_range.split(","):
        token = raw.strip().lower()
        if not token or token == "any":
            continue
        if "-" in token:
            # "3-5" → 5 ; "Grade 3-4" → 4
            tail = token.split("-")[-1]
            try:
                v = int(tail)
            except ValueError:
                continue
        else:
            try:
                v = int(token)
            except ValueError:
                continue
        if max_grade is None or v > max_grade:
            max_grade = v
    return max_grade


def _ae_severity_weight(severity_range: str | None) -> float:
    """Lookup the per-AE penalty weight from its severity_range field."""
    grade = _max_grade_from_severity_range(severity_range)
    if grade is None:
        return _UNKNOWN_GRADE_WEIGHT
    return _SEVERITY_GRADE_TO_WEIGHT.get(grade, _UNKNOWN_GRADE_WEIGHT)


def _regimen_constituents(
    graph: GraphStore, compound_id: str,
) -> list[str]:
    """Return the constituent compound ids of a regimen.

    A regimen exposes its constituents via outbound ``composed_of`` edges
    (added by ``synthesize_combo_compounds`` during populate). For a mono
    compound, no such edges exist; we return ``[compound_id]`` so callers
    that just want "the relevant compound ids" don't need a special case.
    """
    g = graph._graph
    if compound_id not in g:
        return [compound_id]
    constituents: list[str] = []
    for _u, v, key in g.out_edges(compound_id, keys=True):
        if key == EdgeType.COMPOSED_OF.value:
            constituents.append(v)
    return constituents if constituents else [compound_id]


def _collect_modulation_edges(
    graph: GraphStore, compound_id: str,
) -> list[tuple[str, str, EdgeType, EdgeBeliefState]]:
    """MODULATES_EFFICACY_OF edges between any pair of constituents.

    For a regimen with N constituents, walks all C(N, 2) lex-ordered pairs
    and pulls each pair's modulation belief if the edge exists. Returns
    an empty list for mono compounds. The edges fold into the
    trust-weighted geomean aggregation alongside the causal chain edges
    — a contradict-leaning modulation pulls P(regimen success) down,
    a support-leaning one lifts it.
    """
    constituents = sorted(_regimen_constituents(graph, compound_id))
    if len(constituents) < 2:
        return []
    collected: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []
    for i, c1 in enumerate(constituents):
        for c2 in constituents[i + 1:]:
            # Endpoints stored lex-canonical (see
            # canonical_modulation_endpoints), so (c1, c2) is the
            # canonical direction.
            try:
                belief = graph.get_edge_belief(
                    c1, c2, EdgeType.MODULATES_EFFICACY_OF,
                )
            except KeyError:
                continue
            collected.append((c1, c2, EdgeType.MODULATES_EFFICACY_OF, belief))
    return collected

# Log-scaled trust: trust = min(1, log(strength + 1) / log(saturation + 1)).
# Saturation at evidence_strength = 49 → trust = 1.0. Compared to the old
# linear cap at strength=10, this gives weak edges (strength ~0.5–2)
# meaningfully more trust (0.18–0.28 vs 0.05–0.20 before) while preventing
# heavily-loaded edges from drowning out the rest. Important here because
# `endpoint_captures` priors only carry strength ~0.5–2.0 while
# `mechanism_affects` clinical updates can hit 45+ in a few trials.
_TRUST_LOG_SAT = math.log(50.0)  # = log(saturation + 1) with saturation=49
_LOG_FLOOR = 1e-12  # clip per-sample probabilities before taking log


def _trust_weight(belief: EdgeBeliefState) -> float:
    """Map an edge's evidence_strength to a (0, 1] trust weight.

    Evidence-strength 0 → trust 0 (edges with no observations contribute
    nothing). Saturates to 1.0 at evidence_strength=49. The prediction
    engine drops zero-evidence edges entirely before this is called, so
    in practice trust here is always positive.
    """
    s = max(0.0, belief.evidence_strength)
    return min(1.0, math.log(s + 1.0) / _TRUST_LOG_SAT)


def _aggregate_samples(
    edge_samples: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Combine per-edge sample arrays via trust-weighted geometric mean.

    Callers are responsible for dropping zero-evidence edges upstream so
    every entry here has positive weight. The fallback branch (sum_w <= 0)
    survives only as a safety net — it produces an unweighted geomean so
    we don't blow up if an empty chain ever slips through.
    """
    if not edge_samples:
        return np.array([])
    n_samples = edge_samples[0].shape[0]
    sum_w = float(sum(w for w in weights if w > 0.0))
    log_sum = np.zeros(n_samples)
    if sum_w <= 0.0:
        for s in edge_samples:
            log_sum += np.log(np.clip(s, _LOG_FLOOR, 1.0))
        return np.exp(log_sum / len(edge_samples))
    for s, w in zip(edge_samples, weights):
        if w <= 0.0:
            continue
        log_sum += w * np.log(np.clip(s, _LOG_FLOOR, 1.0))
    return np.exp(log_sum / sum_w)


# ── Models ──────────────────────────────────────────────────────────────


class EdgeContribution(BaseModel):
    """One edge's contribution to the prediction."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    belief: EdgeBeliefState
    sampled_mean: float
    bottleneck_score: float = Field(
        description="1 - belief.mean; higher = weaker link"
    )


class SafetyRisk(BaseModel):
    """One adverse-event risk surfaced in a prediction.

    ``source`` distinguishes compound-specific risk (this drug has caused
    this AE in past trials) from target-class risk (other drugs binding
    the same target have caused this AE—likely on-mechanism). The two
    travel together so the consumer can decide whether the risk is
    chemistry-related (changeable) or mechanism-related (intrinsic).
    """

    ae_id: str
    ae_name: str
    source: str  # "compound" | "target_class"
    belief_probability: float
    evidence_strength: float
    contributing_compound_ids: list[str] = Field(default_factory=list)


class PredictionResult(BaseModel):
    """Full prediction for a trial hypothesis.

    Round-20: efficacy and safety are integrated into a single
    ``overall_probability``. The torcetrapib audit (ILLUMINATE,
    NCT00134264) exposed the v0.1.0 decoupling as wrong — the mechanism
    chain worked (CETP inhibition raised HDL) but the trial failed for
    off-target hypertension. Without folding safety into the headline
    number, the system scored torcetrapib at modest success and missed
    the actual failure mode.

    ``overall_probability`` is now ``efficacy_probability * (1 -
    safety_penalty)``. The breakdown is exposed so consumers can see
    where the drag came from.
    """

    trial_hypothesis: str
    # Mechanism-only chain geomean (the round-15 trust-weighted
    # aggregation). What ``overall_probability`` used to mean
    # before round 20.
    efficacy_probability: float
    # [0, 0.4] subtractive — soft-or aggregation of compound-specific +
    # target-class AE evidence. Capped to keep efficacy as the
    # dominant signal; safety acts as a drag, not the whole story.
    safety_penalty: float
    overall_probability: float
    ci_lower: float
    ci_upper: float
    edge_contributions: list[EdgeContribution]
    weakest_link: EdgeContribution | None
    n_samples: int
    # Full list of AE risks (under the more inclusive display threshold)
    # for introspection. The safety_penalty math uses a stricter
    # threshold; see ``_compute_safety_penalty``.
    safety_risks: list[SafetyRisk] = Field(default_factory=list)


# ── Engine ──────────────────────────────────────────────────────────────


class PredictionEngine:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph

    def predict(
        self,
        chain: CausalChain,
        n_samples: int = 10_000,
    ) -> PredictionResult:
        """Compositional prediction via Monte Carlo sampling along one causal chain.

        Aggregation: trust-weighted geometric mean. Edges with no evidence
        beyond the prior contribute little; edges with substantial evidence
        dominate. Trial-level prediction (across multiple arms × subgroups)
        is the caller's responsibility—predict each chain and aggregate
        as appropriate (e.g. per arm, per subgroup, or trial-wide).
        """
        # 1. Collect edges and their beliefs
        edges = self._collect_edges(chain)

        # 2. Sample from each edge's Beta and compute per-edge trust weights
        rng = np.random.default_rng()
        edge_samples: list[np.ndarray] = []
        trust_weights: list[float] = []
        for _src, _tgt, _etype, belief in edges:
            edge_samples.append(rng.beta(belief.alpha, belief.beta, size=n_samples))
            trust_weights.append(_trust_weight(belief))

        # 3. Aggregate samples via trust-weighted geometric mean
        samples = _aggregate_samples(edge_samples, trust_weights)

        # 4. Compute statistics (mechanism-only — the "efficacy" view).
        if samples.size:
            efficacy_prob = float(np.mean(samples))
            ci_lower = float(np.percentile(samples, 2.5))
            ci_upper = float(np.percentile(samples, 97.5))
        else:
            efficacy_prob, ci_lower, ci_upper = 0.5, 0.0, 1.0

        # 5. Build edge contributions. Bottleneck score is
        #    (1 - E[p]) * trust: well-evidenced edges with low expected
        #    probability rise to the top. Since the engine drops
        #    zero-evidence edges upstream, the data-gap penalty term
        #    from round-14 is no longer needed — every edge here has
        #    evidence and the bottleneck ranks them by how strongly that
        #    evidence drags the prediction down.
        contributions: list[EdgeContribution] = []
        for i, (src, tgt, etype, belief) in enumerate(edges):
            sampled_mean = float(np.mean(edge_samples[i])) if edge_samples[i].size else belief.expected_probability
            t = trust_weights[i]
            bottleneck = (1.0 - belief.expected_probability) * t
            contributions.append(EdgeContribution(
                source_id=src,
                target_id=tgt,
                edge_type=etype,
                belief=belief,
                sampled_mean=sampled_mean,
                bottleneck_score=bottleneck,
            ))

        # 6. Identify weakest link by bottleneck score, with raw (1 - E[p])
        #    as tiebreaker.
        weakest = (
            max(
                contributions,
                key=lambda c: (
                    c.bottleneck_score,
                    1.0 - c.belief.expected_probability,
                ),
            )
            if contributions
            else None
        )

        hypothesis = (
            f"{chain.compound_id} -> {chain.target_id} -> "
            f"{chain.mechanism_id} -> {chain.biology_id} -> "
            f"{chain.indication_id}"
        )

        safety_risks = self._collect_safety_risks(chain)

        # Round-20: integrate safety into the headline number. Penalty
        # uses stricter thresholds than the display list, so a chain
        # with a single Beta(1.4, 1) display-grade AE won't move
        # overall_probability — only well-evidenced risks do.
        safety_penalty = self._compute_safety_penalty(chain)
        safety_factor = 1.0 - safety_penalty
        overall_prob = efficacy_prob * safety_factor
        ci_lower = ci_lower * safety_factor
        ci_upper = ci_upper * safety_factor

        return PredictionResult(
            trial_hypothesis=hypothesis,
            efficacy_probability=efficacy_prob,
            safety_penalty=safety_penalty,
            overall_probability=overall_prob,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            edge_contributions=contributions,
            weakest_link=weakest,
            n_samples=n_samples,
            safety_risks=safety_risks,
        )

    # Round-20: safety penalty thresholds. Stricter than the display
    # thresholds in _collect_safety_risks because the penalty math
    # multiplies into the headline number — a Beta(1.4, 1) AE edge
    # with min_belief 0.4 default shouldn't move overall_probability
    # all by itself.
    #
    # min_belief MUST be strictly above 0.5 (the AMBIGUOUS bucket's
    # equilibrium). Live n=50 audit showed nivolumab had 18 AE edges
    # all sitting at exactly E[p]=0.500 because most AE attributions
    # land in AMBIGUOUS (similar incidence across arms with no clear
    # delta). Treating those as "drug causes AE" evidence saturated
    # the penalty cap on every well-evidenced drug. 0.55 means an AE
    # has to have observed delta or RR shifting the Beta mean above
    # 0.5 — i.e. there's actual evidence the drug raises this AE's
    # rate, not just that the AE was reported.
    _SAFETY_PENALTY_MIN_BELIEF = 0.55
    _SAFETY_PENALTY_MIN_EVIDENCE = 1.0
    # Cap on the penalty's drag. 0.6 means efficacy can fall to at
    # most 40% of its mechanism-only value. No drug is "guaranteed to
    # fail purely on safety" — the chain still contributes the final
    # 40%. Round-20 calibration after the first audit showed a 0.4 cap
    # over-penalized well-evidenced drugs (nivolumab, bevacizumab) that
    # had many manageable AEs; raising the cap while switching to a
    # severity-weighted contribution (below) keeps the cap rarely hit.
    _SAFETY_PENALTY_CAP = 0.60

    def _compute_safety_penalty(
        self,
        chain: CausalChain,
        *,
        min_belief: float | None = None,
        min_evidence: float | None = None,
    ) -> float:
        """Severity-weighted safety drag on mechanism-only P(success).
        Round-20 calibration.

        Each AE risk above the belief/evidence threshold contributes a
        SEVERITY-based weight to a soft-or aggregation (see
        ``_SEVERITY_GRADE_TO_WEIGHT``). Grade 1-2 AEs (manageable
        rash, nausea, low-grade fatigue) contribute 0.05; grade 3
        (serious but manageable) 0.15; grade 4 (life-threatening)
        0.30; grade 5 (fatal) 0.50; unknown 0.10. AEs whose belief
        is below threshold don't contribute at all.

        The original belief × log(evidence) formulation over-penalized
        well-evidenced drugs whose AE profile was mostly manageable —
        nivolumab's many low-grade irAEs saturated the cap exactly as
        if they were fatal. Severity weighting captures the real
        clinical distinction: "this drug causes Grade 5 cardiac
        events in 30% of patients" ≠ "this drug causes Grade 1 rash
        in 60% of patients."

        Aggregation: soft-or so multiple AEs accumulate with
        diminishing returns; capped at ``_SAFETY_PENALTY_CAP`` so the
        architecture stays efficacy-led even under extreme AE tails.

        Reuses ``_collect_safety_risks`` retrieval (so both
        ``causes_ae`` compound-level edges AND ``target_associated_ae``
        target-class edges contribute).
        """
        mb = min_belief if min_belief is not None else self._SAFETY_PENALTY_MIN_BELIEF
        me = (
            min_evidence if min_evidence is not None
            else self._SAFETY_PENALTY_MIN_EVIDENCE
        )
        risks = self._collect_safety_risks(
            chain, min_belief=mb, min_evidence=me, max_risks=10_000,
        )
        if not risks:
            return 0.0
        contributions: list[float] = []
        for r in risks:
            try:
                ae_node = self.graph.get_node(r.ae_id)
            except KeyError:
                severity_weight = _UNKNOWN_GRADE_WEIGHT
            else:
                severity_weight = _ae_severity_weight(
                    ae_node.get("severity_range")
                )
            # Three-gate modulation. Severity sets the ceiling
            # (manageable rash vs fatal cardiac event); belief_factor
            # scales by how strongly the AE rate actually moved above
            # the 0.5 "no info" point; trust_factor scales by how much
            # evidence backs the belief. An AE must clear all three to
            # drag the headline number meaningfully.
            #
            # Without belief × evidence weighting, target_class AE
            # cross-pollination dominated — solanezumab failed for
            # efficacy but hit the penalty cap from other compounds'
            # AEs attributed to its target node. Three-gate modulation
            # pulls those back down to commensurate magnitude.
            belief_factor = (r.belief_probability - 0.5) / 0.5
            belief_factor = max(0.0, min(1.0, belief_factor))
            trust_factor = min(
                1.0, math.log(r.evidence_strength + 1) / math.log(50),
            )
            contributions.append(severity_weight * belief_factor * trust_factor)
        penalty = 1.0
        for c in contributions:
            penalty *= (1.0 - c)
        return min(self._SAFETY_PENALTY_CAP, 1.0 - penalty)

    def _collect_safety_risks(
        self,
        chain: CausalChain,
        *,
        min_belief: float = 0.4,
        min_evidence: float = 1.0,
        max_risks: int = 10,
    ) -> list[SafetyRisk]:
        """Pull AE risks from the compound's causes_ae and the target's
        target_associated_ae edges.

        Compound-specific risks come first (more directly attributable);
        target-class risks supplement them. Both filtered by minimum
        belief + evidence so a Beta(1,1) edge isn't reported as a "risk".
        Sorted by belief × evidence_strength so the most-grounded risks
        rise to the top, then capped at ``max_risks`` so the consumer
        isn't drowned in low-signal AEs.
        """
        risks: list[SafetyRisk] = []
        seen_ae_ids: set[str] = set()

        if chain.compound_id == "UNKNOWN":
            return risks

        for edge in self.graph.get_neighboring_edges(
            chain.compound_id, edge_types=[EdgeType.CAUSES_AE],
        ):
            ae_id = edge["target_id"]
            belief = EdgeBeliefState.model_validate(edge["belief"])
            if (
                belief.expected_probability < min_belief
                or belief.evidence_strength < min_evidence
            ):
                continue
            try:
                ae_node = self.graph.get_node(ae_id)
            except KeyError:
                continue
            risks.append(SafetyRisk(
                ae_id=ae_id,
                ae_name=ae_node.get("name", ae_id),
                source="compound",
                belief_probability=belief.expected_probability,
                evidence_strength=belief.evidence_strength,
                contributing_compound_ids=[chain.compound_id],
            ))
            seen_ae_ids.add(ae_id)

        if chain.target_id == "UNKNOWN":
            return risks

        for edge in self.graph.get_neighboring_edges(
            chain.target_id, edge_types=[EdgeType.TARGET_ASSOCIATED_AE],
        ):
            ae_id = edge["target_id"]
            if ae_id in seen_ae_ids:
                # Compound-specific risk already covers this AE; skip the
                # target-class entry so the same AE isn't double-listed.
                continue
            belief = EdgeBeliefState.model_validate(edge["belief"])
            if (
                belief.expected_probability < min_belief
                or belief.evidence_strength < min_evidence
            ):
                continue
            try:
                ae_node = self.graph.get_node(ae_id)
            except KeyError:
                continue
            contributing = [
                rec.source_id for rec in belief.evidence
            ]
            risks.append(SafetyRisk(
                ae_id=ae_id,
                ae_name=ae_node.get("name", ae_id),
                source="target_class",
                belief_probability=belief.expected_probability,
                evidence_strength=belief.evidence_strength,
                contributing_compound_ids=contributing,
            ))

        risks.sort(
            key=lambda r: r.belief_probability * r.evidence_strength,
            reverse=True,
        )
        return risks[:max_risks]

    def compare_hypotheses(
        self,
        chains: list[CausalChain],
        n_samples: int = 10_000,
    ) -> list[PredictionResult]:
        """Predict and rank multiple chains by probability."""
        results = [self.predict(chain, n_samples=n_samples) for chain in chains]
        results.sort(key=lambda r: r.overall_probability, reverse=True)
        return results

    def suggest_improvements(self, result: PredictionResult) -> list[str]:
        """Suggest which edges to strengthen based on evidence + bottleneck analysis.

        Three categories, evaluated independently per edge:
          - DATA GAP: evidence_strength < 2.0 (priors only, regardless of mean).
          - WEAK LINK: evidence_strength ≥ 2.0 and bottleneck_score > 0.5.
          - MODERATE: evidence_strength ≥ 2.0 and 0.2 ≤ bottleneck_score ≤ 0.5.

        Under weighted-geomean prediction, low-trust edges have bottleneck_score
        near 0, so DATA GAP must be detected from evidence_strength rather than
        from the bottleneck ranking.
        """
        suggestions: list[str] = []
        if not result.edge_contributions:
            return ["No edges in the causal chain to evaluate."]

        ranked = sorted(
            result.edge_contributions,
            key=lambda c: (
                c.belief.evidence_strength < 2.0,  # data gaps last
                -c.bottleneck_score,                 # then by bottleneck desc
            ),
        )

        for ec in ranked:
            belief_str = f"Beta({ec.belief.alpha:.1f}, {ec.belief.beta:.1f})"
            p = ec.belief.expected_probability
            if ec.belief.evidence_strength < 2.0:
                suggestions.append(
                    f"[DATA GAP] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str}—insufficient evidence. "
                    f"Need direct experimental validation."
                )
                continue
            if ec.bottleneck_score > 0.5:
                suggestions.append(
                    f"[WEAK LINK] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str}—evidence contradicts this link. "
                    f"Consider alternative targets or mechanisms."
                )
            elif ec.bottleneck_score >= 0.2:
                suggestions.append(
                    f"[MODERATE] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str}—could be strengthened with "
                    f"additional supporting evidence."
                )

        if not suggestions:
            suggestions.append(
                f"All edges strong. Overall P={result.overall_probability:.3f}. "
                f"Consider expanding to new indications."
            )

        return suggestions

    def _collect_edges(
        self, chain: CausalChain
    ) -> list[tuple[str, str, EdgeType, EdgeBeliefState]]:
        """Collect belief states for all edges in the causal chain.

        For ``mechanism_affects`` specifically, retrieves a belief that has
        been conditioned on the indication's relevant tissues—so cell-line
        evidence from the wrong tissue (e.g. a melanoma signature when the
        trial is in NSCLC) gets downweighted rather than counted equally.
        Other edge types are context-free at retrieval.

        For regimens (``compound_id`` resolves to ≥2 constituents via
        ``composed_of``), also folds in any MODULATES_EFFICACY_OF edges
        between constituent pairs. Each modulation belief joins the
        trust-weighted geomean alongside the causal-chain edges.

        Edges with no evidence (Beta(1,1) — no observations beyond the
        prior) are dropped here. This makes the prediction a conditional
        probability over edges the graph has actually learned about:
        unobserved edges contribute nothing, neither evidence nor a
        prior pull toward 0.5. The caller's hypothesis chain determines
        which edges are candidates; the graph's evidence determines
        which of those candidates make it into the geomean.
        """
        edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []
        relevant_tissues = self._tissues_for_chain(chain)

        for src_field, tgt_field, edge_type in _CAUSAL_CHAIN + _AUXILIARY_EDGES:
            src_id = getattr(chain, src_field)
            tgt_id = getattr(chain, tgt_field)

            if src_id == "UNKNOWN" or tgt_id == "UNKNOWN":
                continue

            try:
                if edge_type == EdgeType.MECHANISM_AFFECTS and relevant_tissues:
                    belief = self.graph.get_edge_belief_conditioned(
                        src_id, tgt_id, edge_type, relevant_tissues
                    )
                else:
                    belief = self.graph.get_edge_belief(src_id, tgt_id, edge_type)
            except KeyError:
                continue  # edge not in graph — skip

            if belief.evidence_strength <= 0.0:
                continue  # Beta(1,1) — no learned evidence, skip

            edges.append((src_id, tgt_id, edge_type, belief))

        for me_edge in _collect_modulation_edges(self.graph, chain.compound_id):
            _src, _tgt, _et, belief = me_edge
            if belief.evidence_strength > 0.0:
                edges.append(me_edge)

        return edges

    def _tissues_for_chain(self, chain: CausalChain) -> set[str]:
        """Resolve the chain's indication name to the tissues whose
        cell-line evidence is relevant. Empty set = no conditioning.
        """
        if chain.indication_id == "UNKNOWN":
            return set()
        try:
            ind_node = self.graph.get_node(chain.indication_id)
        except KeyError:
            return set()
        # Lazy import to avoid src.prediction → src.ingestion dependency at
        # module import time.
        from src.ingestion.lincs import tissues_for_indication_name

        return tissues_for_indication_name(ind_node.get("name"))


# ── Stateless query: compound + indication → full chain prediction ─────


def _resolve_target_for_compound(
    graph: GraphStore, compound_id: str
) -> str:
    """Pick the most-supported binds_to target of a compound.

    Tiebreaks on belief.expected_probability * evidence_strength so an edge
    with real evidence beats a Beta(1,1) placeholder. Returns ``"UNKNOWN"``
    if the compound has no binds_to neighbors.
    """
    g = graph._graph
    if compound_id not in g:
        return "UNKNOWN"
    best_id = "UNKNOWN"
    best_score = -1.0
    for _u, v, key, data in g.out_edges(compound_id, data=True, keys=True):
        if key != EdgeType.AFFECTS.value:
            continue
        belief_data = data.get("belief") or {}
        try:
            belief = EdgeBeliefState.model_validate(belief_data)
        except Exception:  # noqa: BLE001—defensive against legacy snapshots
            belief = _DEFAULT_BELIEF
        score = belief.expected_probability * (1.0 + belief.evidence_strength)
        if score > best_score:
            best_score = score
            best_id = v
    return best_id


def _resolve_chain_via_topology(
    graph: GraphStore, target_id: str, indication_id: str
) -> tuple[str, str]:
    """Walk simple paths target → indication and label mechanism / biology.

    Mirrors the resolution semantics used by the backtest's subgraph builder:
      - modulates_via       : v is mechanism
      - mechanism_affects   : u is mechanism, v is biology
      - biology_drives      : u is biology
    Picks the path that resolves the most nodes; ties go to the first
    encountered. Returns ("UNKNOWN", "UNKNOWN") if no path exists.

    Falls back to the first modulates_via neighbor of the target when no
    simple path resolves a mechanism—in graphs where target→mechanism
    edges are dead-ends (no mechanism→indication wiring), this is the only
    way to recover the mechanism node.
    """
    g = graph._graph
    if target_id == "UNKNOWN" or indication_id == "UNKNOWN":
        return "UNKNOWN", "UNKNOWN"
    if target_id not in g or indication_id not in g:
        return "UNKNOWN", "UNKNOWN"
    try:
        paths = list(
            nx.all_simple_paths(g, target_id, indication_id, cutoff=3)
        )
    except nx.NodeNotFound:
        paths = []

    best_mech, best_bio = "UNKNOWN", "UNKNOWN"
    best_score = -1
    for path in paths:
        mech, bio = "UNKNOWN", "UNKNOWN"
        for u, v in zip(path[:-1], path[1:]):
            edges_between = g.get_edge_data(u, v) or {}
            for key in edges_between:
                if key == EdgeType.MODULATES_VIA.value and mech == "UNKNOWN":
                    mech = v
                elif key == EdgeType.MECHANISM_AFFECTS.value:
                    if mech == "UNKNOWN":
                        mech = u
                    if bio == "UNKNOWN":
                        bio = v
                elif key == EdgeType.BIOLOGY_DRIVES.value and bio == "UNKNOWN":
                    bio = u
        score = int(mech != "UNKNOWN") + int(bio != "UNKNOWN")
        if score > best_score:
            best_score = score
            best_mech, best_bio = mech, bio

    if best_mech == "UNKNOWN":
        for _, mid, key in g.out_edges(target_id, keys=True):
            if key == EdgeType.MODULATES_VIA.value:
                best_mech = mid
                break

    return best_mech, best_bio


def _resolve_endpoint_for_indication(
    graph: GraphStore, indication_id: str, biology_id: str = "UNKNOWN",
) -> str:
    """Pick the best-evidenced endpoint for an indication.

    Preference: an endpoint that connects BOTH biology→endpoint and
    endpoint→indication (a complete `reflects_biology → endpoint_captures`
    bridge). Falls back to any endpoint with an evidenced
    `endpoint_captures → indication` edge. Returns "UNKNOWN" when no
    endpoint has trial evidence on its captures edge.
    """
    if indication_id == "UNKNOWN" or indication_id not in graph._graph:
        return "UNKNOWN"
    g = graph._graph

    # Build candidate set: endpoints with endpoint_captures → indication
    # edges that carry evidence.
    candidates: dict[str, float] = {}
    for u, _v, key, data in g.in_edges(indication_id, data=True, keys=True):
        if key != EdgeType.ENDPOINT_CAPTURES.value:
            continue
        try:
            belief = EdgeBeliefState.model_validate(data.get("belief") or {})
        except Exception:  # noqa: BLE001
            continue
        if belief.evidence_strength <= 0.0:
            continue
        # Score = E[p] * evidence_strength
        candidates[u] = belief.expected_probability * (
            1.0 + belief.evidence_strength
        )
    if not candidates:
        return "UNKNOWN"

    # If biology is known, prefer endpoints with an evidenced
    # reflects_biology edge from that biology.
    if biology_id != "UNKNOWN" and biology_id in g:
        bridged = []
        for _u, v, key, data in g.out_edges(biology_id, data=True, keys=True):
            if key != EdgeType.REFLECTS_BIOLOGY.value or v not in candidates:
                continue
            try:
                belief = EdgeBeliefState.model_validate(data.get("belief") or {})
            except Exception:  # noqa: BLE001
                continue
            if belief.evidence_strength > 0.0:
                bridged.append(v)
        if bridged:
            return max(bridged, key=lambda ep: candidates[ep])

    return max(candidates, key=candidates.get)


def predict_clinical_hypothesis(
    graph: GraphStore,
    compound_id: str | None,
    indication_id: str,
    *,
    target_id: str | None = None,
    mechanism_id: str | None = None,
    biology_id: str | None = None,
    endpoint_id: str | None = None,
    population_id: str | None = None,
    n_samples: int = 10_000,
) -> PredictionResult:
    """Stateless prediction over a clinical hypothesis chain.

    Required: ``indication_id`` (must be in the graph). ``compound_id``
    is positional but may be ``None`` or a string not in the graph —
    treated as UNKNOWN, which causes the ``affects`` edge to be skipped.
    This supports the "novel compound, familiar target" use case: pass
    ``target_id`` explicitly and predict from the target-onward chain.

    Optional intermediate nodes (``target_id``, ``mechanism_id``,
    ``biology_id``, ``endpoint_id``) are auto-resolved from the graph
    when not provided AND when ``compound_id`` is in the graph. With a
    novel compound, no auto-resolution from compound is possible — the
    caller must pass at least ``target_id`` for any prediction to fire.

    ``population_id`` is caller-only (no graph resolver): population is
    trial-design information that the chain can't legitimately walk to.
    When omitted, the ``responds_differently`` edge is skipped.

    The engine drops edges with no evidence (Beta(1,1)) so the prediction
    reflects only what the graph has actually learned. A sparse hypothesis
    chain produces a prediction over whichever subset of edges has evidence.

    Raises ``KeyError`` only if ``indication_id`` is not in the graph.
    """
    if indication_id not in graph._graph:
        raise KeyError(f"Indication '{indication_id}' not in graph")

    compound_in_graph = (
        compound_id is not None and compound_id in graph._graph
    )

    if target_id:
        resolved_target = target_id
    elif compound_in_graph:
        resolved_target = _resolve_target_for_compound(graph, compound_id)
    else:
        resolved_target = "UNKNOWN"

    if mechanism_id is None or biology_id is None:
        walked_mech, walked_bio = _resolve_chain_via_topology(
            graph, resolved_target, indication_id,
        )
        resolved_mechanism = mechanism_id or walked_mech
        resolved_biology = biology_id or walked_bio
    else:
        resolved_mechanism = mechanism_id
        resolved_biology = biology_id

    resolved_endpoint = endpoint_id or _resolve_endpoint_for_indication(
        graph, indication_id, resolved_biology,
    )

    chain = CausalChain(
        arm_id="hypothesis",
        compound_id=compound_id if compound_in_graph else "UNKNOWN",
        subgroup_population_id=population_id or "UNKNOWN",
        target_id=resolved_target,
        mechanism_id=resolved_mechanism,
        biology_id=resolved_biology,
        indication_id=indication_id,
        endpoint_id=resolved_endpoint,
        outcome=TrialOutcome.UNKNOWN,
    )
    engine = PredictionEngine(graph)
    return engine.predict(chain, n_samples=n_samples)


# ── CLI ─────────────────────────────────────────────────────────────────


def _find_node_by_name(
    graph: GraphStore, name: str, node_type: str
) -> str | None:
    if not name:
        return None
    name_lower = name.lower()
    for node in graph.get_nodes_by_type(node_type):
        node_name = node.get("name", "").lower()
        if node_name and (name_lower in node_name or node_name in name_lower):
            return node.get("id")
    return None


def _main(
    graph_path: str,
    compound: str,
    target: str,
    indication: str,
    endpoint: str | None,
) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Load graph
    graph = GraphStore()
    graph_file = Path(graph_path)
    if not graph_file.exists():
        console.print(f"[red]Graph file not found: {graph_path}[/red]")
        return
    console.print(f"[bold]Loading graph from {graph_path}...[/bold]")
    graph.import_snapshot(graph_path)
    stats = graph.stats()
    console.print(
        f"  Loaded: {stats['node_count']} nodes, {stats['edge_count']} edges"
    )

    # Resolve names to IDs
    compound_id = _find_node_by_name(graph, compound, "InterventionNode") or "UNKNOWN"
    target_id = _find_node_by_name(graph, target, "TargetNode") or "UNKNOWN"
    indication_id = _find_node_by_name(graph, indication, "IndicationNode") or "UNKNOWN"
    endpoint_id = _find_node_by_name(graph, endpoint or "", "EndpointNode") or "UNKNOWN"

    if compound_id == "UNKNOWN":
        console.print(f"[red]Compound '{compound}' not found in graph[/red]")
        return
    if target_id == "UNKNOWN":
        console.print(f"[red]Target '{target}' not found in graph[/red]")
        return

    console.print(f"\n  Compound: {compound} → {compound_id}")
    console.print(f"  Target: {target} → {target_id}")
    console.print(f"  Indication: {indication} → {indication_id}")
    console.print(f"  Endpoint: {endpoint or 'any'} → {endpoint_id}")

    chain = CausalChain(
        arm_id="cli_query",
        compound_id=compound_id,
        subgroup_population_id="UNKNOWN",
        target_id=target_id,
        mechanism_id="UNKNOWN",
        biology_id="UNKNOWN",
        indication_id=indication_id,
        endpoint_id=endpoint_id,
        outcome=TrialOutcome.UNKNOWN,
    )

    engine = PredictionEngine(graph)
    result = engine.predict(chain)

    # Display results
    console.print(Panel(
        f"[bold]P(success) = {result.overall_probability:.3f}[/bold]\n"
        f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]\n"
        f"Samples: {result.n_samples:,}",
        title="Prediction",
    ))

    if result.edge_contributions:
        table = Table(title="Edge Contributions")
        table.add_column("Edge Type")
        table.add_column("Source → Target")
        table.add_column("P(edge)")
        table.add_column("Bottleneck")
        table.add_column("Evidence")

        for ec in result.edge_contributions:
            is_weakest = ec == result.weakest_link
            style = "bold red" if is_weakest else ""
            table.add_row(
                ec.edge_type.value,
                f"{ec.source_id} → {ec.target_id}",
                f"{ec.belief.expected_probability:.3f}",
                f"{ec.bottleneck_score:.3f}" + (" ← WEAKEST" if is_weakest else ""),
                f"Beta({ec.belief.alpha:.1f}, {ec.belief.beta:.1f})",
                style=style,
            )
        console.print(table)

    suggestions = engine.suggest_improvements(result)
    if suggestions:
        console.print("\n[bold]Suggestions:[/bold]")
        for s in suggestions:
            console.print(f"  {s}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict trial success probability"
    )
    parser.add_argument(
        "--graph",
        default="data/exports/oncology_annotated.json",
        help="Graph snapshot path",
    )
    parser.add_argument("--compound", required=True, help="Compound name")
    parser.add_argument("--target", required=True, help="Target name")
    parser.add_argument("--indication", required=True, help="Indication name")
    parser.add_argument("--endpoint", default=None, help="Endpoint name")
    args = parser.parse_args()

    _main(args.graph, args.compound, args.target, args.indication, args.endpoint)
