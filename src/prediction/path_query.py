"""Compositional path prediction: P(success) ≈ product of edge beliefs along the causal chain."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy import stats as sp_stats
from scipy.special import logsumexp

from src.graph.models import (
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    TrialOutcome,
)
from src.config import CONFIG
from src.inference.beliefs import pool_hierarchical
from src.graph import direction as _direction
from src.graph.store import GraphStore
from src.prediction.contract import (
    _AUXILIARY_EDGES,
    _CAUSAL_CHAIN,
    CONSUMED_BACKBONE_EDGE_TYPES,
    EdgeContribution,
    PredictionResult,
    SafetyRisk,
)

logger = logging.getLogger(__name__)

# _CAUSAL_CHAIN / _AUXILIARY_EDGES / CONSUMED_BACKBONE_EDGE_TYPES are the public
# build contract — they now live in src/prediction/contract.py (imported above) so
# the build + API can reference them without importing this (private-bound) engine.

_DEFAULT_BELIEF = EdgeBeliefState(alpha=1.0, beta=1.0)

# Shared downstream edges whose belief pools across every compound hitting the
# target — so an agonist and an antagonist (opposite intended sign on the pathway)
# update the SAME belief. The direction filter de-contaminates these at query time.
_DIRECTION_PARTITIONED = frozenset({
    EdgeType.MODULATES_VIA,
    EdgeType.MECHANISM_AFFECTS,
    EdgeType.BIOLOGY_DRIVES,
})
_DIR_PRIOR_FLOOR = 1e-6


def _direction_filtered(belief: EdgeBeliefState, chain_direction: str) -> EdgeBeliefState:
    """De-contaminate a shared-edge belief for a directional query (CONFIG.direction; was EROOM_DIRECTION).

    Removes the conjugate contribution of KNOWN OPPOSITE-direction trial evidence
    (an antagonist's evidence when querying an agonist chain, and vice-versa),
    keeping same-direction AND direction-agnostic (curated / unknown) evidence. This
    is SUBTRACTION from the rich pooled belief, not a sparse per-direction rebuild —
    so reuse is preserved and no node is fragmented. No-op when the chain direction
    is unknown or the edge carries no opposite-direction evidence."""
    if not chain_direction or chain_direction == _direction.UNKNOWN:
        return belief
    rem_a = rem_b = 0.0
    for rec in belief.evidence:
        d = (rec.context or {}).get("direction")
        if not d or d == _direction.UNKNOWN or d == chain_direction:
            continue  # keep same-direction + direction-agnostic evidence
        ne, po = rec.applied_n_eff, rec.applied_p_obs
        if ne is None or po is None:
            continue
        rem_a += ne * po
        rem_b += ne * (1.0 - po)
    if rem_a <= 0.0 and rem_b <= 0.0:
        return belief  # nothing opposite to remove
    return EdgeBeliefState(
        alpha=max(_DIR_PRIOR_FLOOR, belief.alpha - rem_a),
        beta=max(_DIR_PRIOR_FLOOR, belief.beta - rem_b),
        evidence=belief.evidence,
        belief_field=belief.belief_field,
    )


def _maybe_direction_filter(
    belief: EdgeBeliefState, edge_type: EdgeType, chain: CausalChain
) -> EdgeBeliefState:
    """Apply the direction filter to a shared efficacy edge (baked ON via CONFIG.direction; was EROOM_DIRECTION)."""
    if not _direction.enabled() or edge_type not in _DIRECTION_PARTITIONED:
        return belief
    return _direction_filtered(belief, getattr(chain, "direction", ""))


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
# Round-29: when CTCAE grade is missing but the AE was flagged
# ``serious=True`` by CT.gov (~90% of extractions), use this as a
# coarse severity floor (= grade-3 weight). Deliberately not max
# (grade 5 = 0.50): a Serious Adverse Event ≠ a fatal event, and we
# preserve discrimination via belief_factor + trust_factor in the
# three-gate math. When both grade and serious data are present, the
# floor is just that — a floor; the grade-based weight wins if higher.
_SERIOUS_FLOOR_WEIGHT = _SEVERITY_GRADE_TO_WEIGHT[3]


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


def _ae_severity_weight(
    severity_range: str | None, *, serious: bool = False,
) -> float:
    """Lookup the per-AE penalty weight from its severity_range field.

    Round-29: ``serious=True`` acts as a coarse severity floor at the
    grade-3 weight. The corpus has CTCAE grade data on only ~11% of
    extracted AEs (CT.gov rarely posts them), but the per-AE
    ``serious`` flag is populated on ~90% — making it the primary
    severity signal in practice. When grade IS present, the grade-based
    weight wins if it exceeds the floor; an SAE flagged grade-5 still
    gets the grade-5 weight.
    """
    grade = _max_grade_from_severity_range(severity_range)
    if grade is None:
        grade_weight = _UNKNOWN_GRADE_WEIGHT
    else:
        grade_weight = _SEVERITY_GRADE_TO_WEIGHT.get(
            grade, _UNKNOWN_GRADE_WEIGHT,
        )
    if serious:
        return max(grade_weight, _SERIOUS_FLOOR_WEIGHT)
    return grade_weight


# Round-30: gate the safety penalty on FAILURE-CAUSING toxicity. causes_ae
# measures AE *occurrence* (incidence) — but an effective drug with tolerated
# toxicity (e.g. nivolumab irAEs in trials that SUCCEEDED) should not be
# penalized like one whose toxicity was dose-limiting. Each AE's contribution
# scales toward this floor when its evidence came from tolerated/successful
# trials and toward full weight when it came from dose-limiting-toxicity
# failures.
#
# Round-31 (benchmark, 2026-06-05): floor 0.15 -> 0.0. The 0.15 floor made
# every *tolerated* AE still contribute 15% of its severity weight, and the
# soft-or over ~10 AEs/trial compounded that into a ~0.40 penalty on drugs
# whose toxicity was entirely manageable — the dominant source of the
# prediction's systematic pessimism (mean P 0.49 vs 0.71 base rate; accuracy
# BELOW base rate). The benchmark (scripts/tune_composition.py, n=129 holdout)
# isolated it: floor 0.15->0.0 lifts holdout Brier 0.266->0.232, accuracy
# 0.550->0.674, ECE 0.223->0.160, with NO loss of the safety thesis — a
# failure-causing tox (e.g. torcetrapib, failure_causing_fraction~1) still
# contributes fully; only tolerated AEs (fraction 0) now contribute nothing.
# contribution = floor + (1-floor)*fraction = fraction, the clean DLT gate the
# round-30 design intended. See memory project_benchmark_verdict / tuning-log.
_SAFETY_DLT_FLOOR = CONFIG.safety_dlt_floor


def _safety_dlt_gate_enabled() -> bool:
    # Baked ON: penalize failure-causing toxicity, not mere AE occurrence.
    # See src/config.py.
    return CONFIG.safety_dlt_gate


def _dlt_fraction(belief: EdgeBeliefState) -> float:
    """Fraction of an AE edge's evidence tagged ``failure_causing_tox``.

    Records are tagged at attribution from the source trial's failure mode
    (DOSE_LIMITING_TOXICITY) — see ``attributor.attribute_adverse_events``.
    Returns 1.0 when there are no records (no info -> don't gate).
    """
    recs = belief.evidence or []
    if not recs:
        return 1.0
    fc = sum(1 for e in recs if (e.context or {}).get("failure_causing_tox"))
    return fc / len(recs)


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
_SOFTMIN_T = CONFIG.softmin_t  # weakest-link softmin temperature; ->0 approaches hard min

# Informed prior (baked ON via CONFIG.informed_prior; was EROOM_INFORMED_PRIOR):
# swap the Beta(1,1) "coin-flip" prior
# for a WEAK prior centered on a plausible base rate, so an under-evidenced /
# unobserved chain edge defers to "probably operative" (~0.75) instead of
# producing low samples that spuriously become the weakest link. Weak
# (strength ~2 pseudo-obs) so real evidence dominates; calibratable.
_INFORMED_PRIOR_MEAN = CONFIG.prior_mean
_INFORMED_PRIOR_STRENGTH = CONFIG.prior_strength
_INFORMED_PRIOR_A = _INFORMED_PRIOR_MEAN * _INFORMED_PRIOR_STRENGTH          # 1.5
_INFORMED_PRIOR_B = (1.0 - _INFORMED_PRIOR_MEAN) * _INFORMED_PRIOR_STRENGTH  # 0.5


def _informed_prior_enabled() -> bool:
    # Baked ON: under-evidenced edges defer to a plausible base rate (~0.75)
    # instead of the Beta(1,1) coin-flip. See src/config.py.
    return CONFIG.informed_prior


# Module-level prediction RNG. Default is a fresh unseeded Generator (random,
# unchanged behavior for production callers). The eval harness calls
# ``seed_prediction_rng(seed)`` once so the Monte-Carlo holdout AUROC is exactly
# reproducible (the per-trial Beta sampling otherwise gives ~±0.002 AUROC noise).
# This is a reproducibility hook, not an algorithm parameter — the math is
# identical; only the random stream is pinned.
_PREDICTION_RNG = np.random.default_rng()


def seed_prediction_rng(seed: int | None) -> None:
    """Reseed the shared prediction RNG (None → fresh random Generator)."""
    global _PREDICTION_RNG
    _PREDICTION_RNG = np.random.default_rng(seed)


def _sample_edge(rng, belief, n_samples: int) -> np.ndarray:
    """Sample an edge's Beta, optionally under a weak informed prior.

    The stored belief is Beta(1+e_pos, 1+e_neg) (Beta(1,1) prior + evidence).
    With the informed prior (CONFIG.informed_prior; was EROOM_INFORMED_PRIOR)
    we re-prior to Beta(a0+e_pos, b0+e_neg) where
    Beta(a0,b0) has mean ~0.75 — so an unobserved edge samples around 0.75
    (plausible) rather than uniform/low, and well-evidenced edges are
    essentially unchanged (the weak prior is swamped).
    """
    a, b = belief.alpha, belief.beta
    if _informed_prior_enabled():
        a = _INFORMED_PRIOR_A + (belief.alpha - 1.0)
        b = _INFORMED_PRIOR_B + (belief.beta - 1.0)
    return rng.beta(max(a, 1e-6), max(b, 1e-6), size=n_samples)


def _trust_weight(belief: EdgeBeliefState) -> float:
    """Map an edge's evidence_strength to a (0, 1] trust weight.

    Evidence-strength 0 → trust 0 (edges with no observations contribute
    nothing). Saturates to 1.0 at evidence_strength=49. The prediction
    engine drops zero-evidence edges entirely before this is called, so
    in practice trust here is always positive.
    """
    s = max(0.0, belief.evidence_strength)
    return min(1.0, math.log(s + 1.0) / _TRUST_LOG_SAT)


def _aggregate_samples(edge_samples: list[np.ndarray]) -> np.ndarray:
    """Combine per-edge sample arrays into one chain-probability sample array via
    the weakest-link **softmin** (the baked aggregation; temperature
    ``CONFIG.softmin_t``). Runs unweighted on purpose — a sparse-but-decisive
    weak edge must NOT be down-weighted; it is exactly the bottleneck softmin is
    meant to surface. Returns an empty array for an empty chain."""
    if not edge_samples:
        return np.array([])
    stack = np.clip(np.vstack(edge_samples), _LOG_FLOOR, 1.0)
    return -_SOFTMIN_T * (
        logsumexp(-stack / _SOFTMIN_T, axis=0) - np.log(stack.shape[0])
    )


# ── Models ──────────────────────────────────────────────────────────────


def _population_ancestors(pop_id: str) -> list[str]:
    """Disease-agnostic population ids, most-specific-first: the id itself then
    every coarser axis-subset down to single axes.

    The hierarchy is implicit in the ``__``-joined slug (``compose_id`` sorts
    axes), so a 3-axis population's ancestors are its 2-axis and 1-axis subsets
    (e.g. ``line_first__stage_iii`` → ``line_first``, ``stage_iii``). Used for
    the prediction backoff so a sparse specific population borrows its coarser
    parent's (cross-disease-pooled) evidence."""
    from itertools import combinations

    axes = pop_id.split("__")
    if len(axes) <= 1:
        return [pop_id]
    out = [pop_id]
    for k in range(len(axes) - 1, 0, -1):
        for combo in combinations(axes, k):
            out.append("__".join(combo))
    return out


def _resolve_responds_differently(
    graph: GraphStore, pop_id: str, indication_id: str
) -> tuple[str, "EdgeBeliefState"] | None:
    """Most-specific *evidenced* ``responds_differently`` belief for a chain's
    population, walking the population hierarchy (specific → coarser ancestors).

    Borrow-strength backoff via hierarchical PARTIAL POOLING: the specific
    population's belief is pooled with its coarser ancestors (e.g.
    ``line_first__stage_iii`` ← ``line_first``), each coarser level acting as a
    concentration-capped prior for the finer one. A sparse specific population is
    dominated by a rich (capped) ancestor; a well-evidenced specific population
    overrides it. Returns ``(most_specific_evidenced_population_id, pooled_belief)``
    or ``None`` when no level in the hierarchy has evidence. (Earlier behavior
    returned the FIRST evidenced level outright, so one specific trial shadowed a
    cross-disease-pooled ancestor — see ``pool_hierarchical``.)"""
    levels: list[tuple[str, EdgeBeliefState]] = []
    for ancestor in _population_ancestors(pop_id):
        try:
            belief = graph.get_edge_belief(
                ancestor, indication_id, EdgeType.RESPONDS_DIFFERENTLY
            )
        except KeyError:
            continue
        if belief.evidence_strength > 0.0:
            levels.append((ancestor, belief))
    if not levels:
        return None
    pooled = pool_hierarchical([b for _id, b in levels])
    return levels[0][0], pooled


def _indication_ancestors(
    graph: GraphStore, indication_id: str, *, max_depth: int = 5
) -> list[str]:
    """A leaf indication and its SUBTYPE_OF ancestors, most-specific-first
    (uveal_melanoma → melanoma → …). Walks SUBTYPE_OF out-edges (specific →
    parent), bounded depth, cycle-guarded. The hierarchy is sourced from the
    hand-curated map + EFO/MONDO (Phase 4)."""
    out = [indication_id]
    seen = {indication_id}
    cur = indication_id
    for _ in range(max_depth):
        parents = [
            v
            for _u, v, key in graph._graph.out_edges(cur, keys=True)  # noqa: SLF001
            if key == EdgeType.SUBTYPE_OF.value and v not in seen
        ]
        if not parents:
            break
        cur = parents[0]
        seen.add(cur)
        out.append(cur)
    return out


def _resolve_indication_edge(
    graph: GraphStore, src_id: str, indication_id: str, edge_type: EdgeType
) -> tuple[str, "EdgeBeliefState"] | None:
    """Most-specific *evidenced* belief for an indication-targeted edge
    (biology_drives / endpoint_captures), walking the SUBTYPE_OF hierarchy.

    Phase-4 indication backoff (symmetric with the population backoff): with
    chains leaf-anchored, a sparse leaf (uveal_melanoma) borrows its parent's
    (melanoma) cross-trial evidence via hierarchical PARTIAL POOLING — each
    coarser SUBTYPE_OF ancestor acts as a concentration-capped prior for the
    finer level. When the leaf edge is unobserved the result is purely the
    ancestor (full backoff); when the leaf is sparse-but-present it is shrunk
    toward the (capped) parent rather than shadowing it; when the leaf is
    well-evidenced it dominates. Returns ``(most_specific_evidenced_indication,
    pooled_belief)`` or ``None``. (Earlier behavior returned the FIRST evidenced
    level outright — one uveal_melanoma trial overrode the 50-trial melanoma
    belief; see ``pool_hierarchical``.)"""
    levels: list[tuple[str, EdgeBeliefState]] = []
    for ancestor in _indication_ancestors(graph, indication_id):
        try:
            belief = graph.get_edge_belief(src_id, ancestor, edge_type)
        except KeyError:
            continue
        if belief.evidence_strength > 0.0:
            levels.append((ancestor, belief))
    if not levels:
        return None
    pooled = pool_hierarchical([b for _id, b in levels])
    return levels[0][0], pooled


# EdgeContribution / SafetyRisk / PredictionResult are the public prediction
# output schema — they now live in src/prediction/contract.py (imported above).


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

        Aggregation: weakest-link softmin — the sole aggregation path (baked via
        ``CONFIG.softmin_t``; the former ``EROOM_AGG`` geomean/product/min/harmonic
        variants were deleted). Edges
        with no evidence beyond the prior are dropped upstream. Trial-level
        prediction (across multiple arms × subgroups) is the caller's
        responsibility—predict each chain and aggregate as appropriate (e.g.
        per arm, per subgroup, or trial-wide).
        """
        # 1. Collect edges and their beliefs
        edges = self._collect_edges(chain)
        hypothesis = (
            f"{chain.compound_id} -> {chain.target_id} -> "
            f"{chain.mechanism_id} -> {chain.biology_id} -> "
            f"{chain.indication_id}"
        )
        return self._result_from_edges(
            edges,
            n_samples=n_samples,
            hypothesis=hypothesis,
            safety_penalty=self._compute_safety_penalty(chain),
            safety_risks=self._collect_safety_risks(chain),
        )

    def _result_from_edges(
        self,
        edges: list[tuple[str, str, EdgeType, EdgeBeliefState]],
        *,
        n_samples: int,
        hypothesis: str,
        safety_penalty: float,
        safety_risks: list[SafetyRisk],
    ) -> PredictionResult:
        """Core: a collected edge set → PredictionResult. Shared by single-chain
        ``predict`` and the combo composer ``predict_combo`` — the only thing
        that differs is which edges + safety the caller pools. Aggregation and
        the weakest-link bottleneck are identical, so a combo's prediction is
        the same weakest-link softmin over the UNION of its constituents' chain
        edges (not a single constituent picked via a weak combo_inherit edge)."""
        # 2. Sample from each edge's Beta and compute per-edge trust weights
        rng = _PREDICTION_RNG
        edge_samples: list[np.ndarray] = []
        trust_weights: list[float] = []
        for _src, _tgt, _etype, belief in edges:
            edge_samples.append(_sample_edge(rng, belief, n_samples))
            trust_weights.append(_trust_weight(belief))

        # 3. Aggregate samples (weakest-link softmin — the sole path; was EROOM_AGG)
        samples = _aggregate_samples(edge_samples)

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

        # Round-20: integrate safety into the headline number. The caller
        # supplies safety_penalty (stricter thresholds than the display list,
        # so a single Beta(1.4, 1) display-grade AE won't move overall_prob)
        # and the display risk list.
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

    def predict_combo(
        self,
        constituent_chains: list[CausalChain],
        combo_id: str,
        *,
        n_samples: int = 10_000,
    ) -> PredictionResult:
        """Compose a regimen's prediction from its CONSTITUENTS' chains.

        A direct query on a synthesized combo InterventionNode would otherwise
        walk one weak ``combo_inherit`` AFFECTS edge and predict a single
        constituent's chain. Instead, pool every constituent chain's edges
        (each constituent reached via its OWN curated AFFECTS edge), add the
        inter-constituent ``MODULATES_EFFICACY_OF`` edges that live on the combo
        node, and run the same weakest-link softmin over the union — so the
        regimen's probability is set by the weakest edge across ALL its
        mechanisms, and any synergy/antagonism modulation joins the aggregate.
        Safety is the soft-or of the constituents' AE risks (deduped by AE).
        """
        seen: set[tuple[str, str, str]] = set()
        edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []
        for ch in constituent_chains:
            for e in self._collect_edges(ch):
                key = (e[0], e[1], e[2].value)
                if key not in seen:
                    seen.add(key)
                    edges.append(e)
        # Inter-constituent modulation edges hang off the combo node, not the
        # mono constituents — add them explicitly (each constituent's own
        # _collect_edges sees no modulation pairs, being a single compound).
        for me in _collect_modulation_edges(self.graph, combo_id):
            key = (me[0], me[1], me[2].value)
            if me[3].evidence_strength > 0.0 and key not in seen:
                seen.add(key)
                edges.append(me)

        # Safety: union the constituents' AE risks (dedup by AE), then one
        # soft-or — a combo inherits each constituent's toxicity.
        def _union_risks(max_risks: int, mb: float, mev: float) -> list[SafetyRisk]:
            out: list[SafetyRisk] = []
            seen_ae: set[str] = set()
            for ch in constituent_chains:
                for r in self._collect_safety_risks(
                    ch, min_belief=mb, min_evidence=mev, max_risks=max_risks,
                ):
                    if r.ae_id not in seen_ae:
                        seen_ae.add(r.ae_id)
                        out.append(r)
            return out

        penalty = self._penalty_from_risks(_union_risks(
            10_000,
            self._SAFETY_PENALTY_MIN_BELIEF,
            self._SAFETY_PENALTY_MIN_EVIDENCE,
        ))
        display_risks = sorted(
            _union_risks(10, 0.4, 1.0),
            key=lambda r: r.belief_probability * r.evidence_strength,
            reverse=True,
        )[:10]

        names = ", ".join(ch.compound_id for ch in constituent_chains)
        indication = (
            constituent_chains[0].indication_id if constituent_chains else "UNKNOWN"
        )
        hypothesis = (
            f"{combo_id} [{len(constituent_chains)} constituents: {names}] "
            f"-> {indication}"
        )
        return self._result_from_edges(
            edges,
            n_samples=n_samples,
            hypothesis=hypothesis,
            safety_penalty=penalty,
            safety_risks=display_risks,
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
    _SAFETY_PENALTY_CAP = CONFIG.safety_penalty_cap

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

        Aggregation: MAX over per-AE contributions (round-31; was
        soft-or, which double-counted correlated AEs in a failed trial);
        capped at ``_SAFETY_PENALTY_CAP`` so the architecture stays
        efficacy-led even under extreme AE tails.

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
        return self._penalty_from_risks(risks)

    def _penalty_from_risks(self, risks: list[SafetyRisk]) -> float:
        """Max-over-AEs, severity-weighted safety drag over a precomputed risk
        list — the shared core of the single-chain penalty (one compound's risks)
        and the combo penalty (constituents' risks unioned by AE). Round-31
        switched the aggregation from soft-or to max (see the inline note)."""
        if not risks:
            return 0.0
        contributions: list[float] = []
        for r in risks:
            # Round-29: SafetyRisk now carries severity_range + the
            # serious flag directly (target-scoped for
            # target_associated_ae risks via edge metadata, PT-node-
            # sourced for causes_ae). Both feed into the severity
            # weight: an AE flagged serious=True floors the weight at
            # the grade-3 tier even when CTCAE grade data is missing.
            if r.severity_range or r.serious:
                severity_weight = _ae_severity_weight(
                    r.severity_range, serious=r.serious,
                )
            else:
                try:
                    ae_node = self.graph.get_node(r.ae_id)
                except KeyError:
                    severity_weight = _UNKNOWN_GRADE_WEIGHT
                else:
                    severity_weight = _ae_severity_weight(
                        ae_node.get("severity_range"),
                        serious=bool(ae_node.get("serious", False)),
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
            contribution = severity_weight * belief_factor * trust_factor
            if _safety_dlt_gate_enabled():
                # Gate on failure-causing toxicity (see _dlt_fraction): an AE
                # that merely occurred in tolerated/successful trials
                # contributes only the floor share; toxicity that caused a
                # dose-limiting failure contributes fully.
                contribution *= (
                    _SAFETY_DLT_FLOOR
                    + (1.0 - _SAFETY_DLT_FLOOR) * r.failure_causing_fraction
                )
            contributions.append(contribution)
        # Round-31 (benchmark): MAX over per-AE contributions, not soft-or.
        # Soft-or (noisy-OR, 1-prod(1-c)) is the correct model for INDEPENDENT
        # failure modes — but AEs reported in a failed trial are CORRELATED (the
        # failure_causing attribution smears across co-occurring AEs), so soft-or
        # double-counts and over-penalizes. A drug's safety risk is its WORST
        # toxicity; max avoids the double-count. On the n=129 holdout this fixes
        # the residual pessimism the floor fix left: the [0.2,0.3)-rated trials
        # (71% actually succeeded) lift to a calibrated ~0.46 (obs 0.50);
        # Brier 0.231->0.209, accuracy 0.674->0.713, ECE 0.160->0.116, torcetrapib
        # still correctly failure. See memory tuning_safety_aggregation.
        penalty = max(contributions) if contributions else 0.0
        return min(self._SAFETY_PENALTY_CAP, penalty)

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
                # causes_ae: severity comes from the PT-level AE node
                # itself, which is uniquely the PT (not shared across
                # targets). No edge-level override needed.
                severity_range=ae_node.get("severity_range") or "",
                serious=bool(ae_node.get("serious", False)),
                failure_causing_fraction=_dlt_fraction(belief),
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
            # Round-29: prefer the EDGE's severity_range + serious flag
            # (both target-scoped, written by the SOC propagation in
            # ae_propagation._rebuild_target_ae_edge). Fall back to the
            # AE node's severity_range / serious for PT-tier
            # target_associated_ae edges (where the AE node IS specific)
            # and for legacy snapshots predating round-29.
            edge_severity = (edge.get("severity_range") or "")
            if not edge_severity:
                edge_severity = ae_node.get("severity_range") or ""
            edge_serious = edge.get("serious")
            if edge_serious is None:
                edge_serious = bool(ae_node.get("serious", False))
            risks.append(SafetyRisk(
                ae_id=ae_id,
                ae_name=ae_node.get("name", ae_id),
                source="target_class",
                belief_probability=belief.expected_probability,
                evidence_strength=belief.evidence_strength,
                severity_range=edge_severity,
                serious=bool(edge_serious),
                failure_causing_fraction=_dlt_fraction(belief),
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

            if edge_type == EdgeType.RESPONDS_DIFFERENTLY:
                # Phase-3 hierarchical backoff: a sparse specific population
                # (line_first__stage_iii) borrows its coarser ancestor's
                # (line_first) cross-disease-pooled evidence. The resolved
                # ancestor's belief stands in for the population edge.
                resolved = _resolve_responds_differently(self.graph, src_id, tgt_id)
                if resolved is not None:
                    anc_id, belief = resolved
                    edges.append((anc_id, tgt_id, edge_type, belief))
                continue

            if edge_type in (
                EdgeType.BIOLOGY_DRIVES, EdgeType.ENDPOINT_CAPTURES
            ):
                # Phase-4 indication backoff: chains are leaf-anchored, so a
                # sparse leaf indication (uveal_melanoma) borrows its SUBTYPE_OF
                # parent's (melanoma) evidence for biology→indication /
                # endpoint→indication.
                resolved = _resolve_indication_edge(
                    self.graph, src_id, tgt_id, edge_type
                )
                if resolved is not None:
                    anc_ind, belief = resolved
                    belief = _maybe_direction_filter(belief, edge_type, chain)
                    edges.append((src_id, anc_ind, edge_type, belief))
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

            belief = _maybe_direction_filter(belief, edge_type, chain)
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


def _stated_chains_for(
    graph: GraphStore, compound_id: str, indication_id: str,
) -> list[CausalChain]:
    """The trial's OWN decomposed chains for this (compound, indication) — the
    faithful decomposition populate produced, preferred over
    ``_resolve_chain_via_topology`` (which re-resolves by strongest-evidenced
    neighbor and lands on generic, heavily-pooled biology like DNA-damage
    apoptosis instead of the drug's actual mechanism — "chain-walking optimism").
    Deduped by backbone signature; empty for a novel hypothesis (compound +
    indication not co-occurring in any built chain) → caller falls back to
    topology."""
    out: list[CausalChain] = []
    seen: set[tuple] = set()
    for ts in graph.trial_subgraphs.values():
        for ch in ts.chains:
            if ch.compound_id != compound_id or ch.indication_id != indication_id:
                continue
            key = (ch.target_id, ch.mechanism_id, ch.biology_id,
                   ch.endpoint_id, ch.subgroup_population_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(ch)
    return out


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

    # Faithful-decomposition preference: when the graph already holds this
    # (compound, indication)'s OWN chain(s) and the caller hasn't pinned an
    # intermediate node, predict those STATED chains rather than re-resolving by
    # topology — which free-walks to the generic, most-evidenced biology
    # ("enzyme inhibition → DNA-damage apoptosis") instead of the drug's actual
    # mechanism (the chain-walking-optimism bug). Return the most-decisive
    # (lowest-overall) stated chain. Novel hypotheses (no matching chain) fall
    # through to topology resolution below.
    if (
        compound_in_graph and target_id is None and mechanism_id is None
        and biology_id is None and endpoint_id is None and population_id is None
    ):
        stated = _stated_chains_for(graph, compound_id, indication_id)
        if stated:
            engine = PredictionEngine(graph)
            stated_results = [
                r for r in (engine.predict(c, n_samples=n_samples) for c in stated)
                if r.edge_contributions
            ]
            if stated_results:
                return min(stated_results, key=lambda r: r.overall_probability)

    # Combo regimens: compose the CONSTITUENTS' chains rather than walk the
    # combo's weak combo_inherit AFFECTS edge to a single target. Skipped when
    # the caller pinned an explicit target/mechanism (which means "predict this
    # one path") — then fall through to the single-chain logic below.
    constituents = (
        _regimen_constituents(graph, compound_id) if compound_in_graph else []
    )
    if len(constituents) >= 2 and target_id is None and mechanism_id is None:
        constituent_chains: list[CausalChain] = []
        for cid in constituents:
            c_tgt = _resolve_target_for_compound(graph, cid)
            c_mech, c_bio = _resolve_chain_via_topology(graph, c_tgt, indication_id)
            c_end = endpoint_id or _resolve_endpoint_for_indication(
                graph, indication_id, c_bio,
            )
            constituent_chains.append(CausalChain(
                arm_id="hypothesis",
                compound_id=cid,
                subgroup_population_id=population_id or "UNKNOWN",
                target_id=c_tgt,
                mechanism_id=c_mech,
                biology_id=c_bio,
                indication_id=indication_id,
                endpoint_id=c_end,
                outcome=TrialOutcome.UNKNOWN,
            ))
        return PredictionEngine(graph).predict_combo(
            constituent_chains, compound_id, n_samples=n_samples,
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
