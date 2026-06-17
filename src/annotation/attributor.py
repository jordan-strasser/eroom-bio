"""Translate failure classifications into graph edge updates."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.graph.models import (
    AdverseEventNode,
    canonical_modulation_endpoints,
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    TrialArm,
    TrialOutcome,
    TrialSubgraph,
    normalize_entity,
)
from src.graph.store import GraphStore
from src.inference.ae_propagation import propagate_to_target_associated_ae
from src.inference.beliefs import (
    SupportBucket,
    _precision_multiplier,
    effective_n_for_evidence,
    modulation_bucket,
    p_obs_for_bucket,
)
from src.annotation.meddra import MeddraCache, ae_node_id, normalize_ae_term
from src.annotation.meddra_hierarchy import MedDRAHierarchy


def _meddra_hierarchy_singleton() -> MedDRAHierarchy:
    """Lazy-load the round-28 MedDRA hierarchy. Returns a shared
    instance — see MedDRAHierarchy.load_default for caching rules."""
    return MedDRAHierarchy.load_default()
from src.annotation.taxonomy import (
    ArmIncidence,
    FailureClassification,
    FailureMode,
    ModulationEntry,
    RoutingBranch,
    routing_branch_for,
    stop_reason_override,
    StructuredAE,
    TrialExtraction,
)

logger = logging.getLogger(__name__)

# CT.gov status cache for the stop-reason routing override (the terminated-trial
# misroute fix). Maps NCT → {"overall_status", "why_stopped"}. Built by
# ``scripts/build_ctgov_status_cache.py``. ABSENT by default → the override is a
# no-op and routing behaviour is byte-identical, so reproducibility is preserved.
_CTGOV_STATUS_CACHE_PATH = Path("data/cache/ctgov_status.json")
_ctgov_status_cache: dict[str, dict] | None = None
# audit log of every applied stop-reason override (one JSON line per flip)
_STOP_OVERRIDE_LOG_PATH = Path("data/dev/stop_reason_overrides.jsonl")


def _direction_ctx(chain) -> dict:
    """Direction tag for a backbone evidence record's context (native modulation
    direction, src.graph.direction). Empty unless the chain carries a resolved
    direction — so pre-direction snapshots and EROOM_DIRECTION-off builds (chains
    default ``unknown``) stay byte-identical; the per-direction partition appears
    only once a build has stamped directions. Read at query time by the
    direction-matched prediction path."""
    d = getattr(chain, "direction", "") or ""
    return {"direction": d} if d and d != "unknown" else {}


def _ctgov_status_for(nct: str) -> dict | None:
    """CT.gov {overall_status, why_stopped} for an NCT, or None if uncached.

    Lazy-loads the cache once. Missing file → empty cache → every lookup is None
    → the stop-reason override never fires (default behaviour preserved)."""
    global _ctgov_status_cache
    if _ctgov_status_cache is None:
        try:
            _ctgov_status_cache = json.loads(_CTGOV_STATUS_CACHE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            _ctgov_status_cache = {}
    return _ctgov_status_cache.get(nct)

_ANNOTATIONS_DIR = Path("data/annotations")
# v0.3.0 unrouted modulation log — separate file so unroutable LLM
# modulation_entries don't drown in the classifier unrouted log. A high
# count here signals the extraction prompt is emitting entity ids that
# don't normalize to graph nodes; treat it as a debug signal worth
# investigating before scaling the prompt to a wider corpus.
_UNROUTED_MOD_LOG_PATH = Path("data/dev/unrouted_modulation_entries.jsonl")

# Map trial phase string to EvidenceType
_PHASE_TO_EVIDENCE: dict[str, EvidenceType] = {
    "1": EvidenceType.CLINICAL_PHASE1,
    "early_1": EvidenceType.CLINICAL_PHASE1,
    "2": EvidenceType.CLINICAL_PHASE2,
    "2/3": EvidenceType.CLINICAL_PHASE3,
    "3": EvidenceType.CLINICAL_PHASE3,
    "4": EvidenceType.CLINICAL_PHASE3,
}

# Sentinel used in CausalChain fields when a graph id wasn't yet resolved
# (e.g. by populate.py before extraction filled in the biology id).
_UNKNOWN_PLACEHOLDER = "UNKNOWN"


# ── Outcome-conditioning p_obs constants ─────────────────────────────────
#
# The outcome-conditioning attributor conditions the WHOLE causal chain by
# edge id on the trial's per-arm outcome, rather than name-matching a
# classifier-emitted failing edge to a chain edge. A single trial can't
# pinpoint WHICH edge failed (premature falsification), so failure mass is
# SPREAD across the chain (explaining-away) and is deliberately MODEST —
# trial failure ≠ mechanistic falsification.
#
# SUCCESS  (conjunctive — the whole path operated): every backbone edge
#          gets a support update at full trial weight.
# FAILURE  (disjunctive — ≥1 edge weak, unknown which): the failure mass is
#          split across the chain weighted toward the currently-uncertain
#          edges (explaining-away), each at a MODEST contradict p_obs.
# PARTIAL  : a weaker / more ambiguous version of failure.
#
# p_obs values reuse the BUCKET_TO_P_OBS scale (see beliefs.py): MODERATE
# = 0.80 / 0.20, WEAK = 0.65 / 0.35.
_SUCCESS_P_OBS = p_obs_for_bucket(SupportBucket.MODERATE_SUPPORT)   # 0.80
_PARTIAL_SUPPORT_P_OBS = p_obs_for_bucket(SupportBucket.WEAK_SUPPORT)  # 0.65
_FAILURE_P_OBS = p_obs_for_bucket(SupportBucket.MODERATE_CONTRADICT)  # 0.20
_PARTIAL_CONTRADICT_P_OBS = p_obs_for_bucket(SupportBucket.WEAK_CONTRADICT)  # 0.35


# Bucket label recorded on the EvidenceRecord per outcome (so the support
# string on the replay log is sensible; the actual p_obs is injected via
# p_obs_override so the conjugate step matches the constants above).
_OUTCOME_TO_SUPPORT_BUCKET: dict["TrialOutcome", SupportBucket] = {
    TrialOutcome.SUCCESS: SupportBucket.MODERATE_SUPPORT,
    TrialOutcome.PARTIAL: SupportBucket.WEAK_CONTRADICT,
    TrialOutcome.FAILURE: SupportBucket.MODERATE_CONTRADICT,
}


# ── TASK 2: per-edge attribution-math experiment knobs ────────────────────
#
# The default (``explain_away``) is the shipped asymmetric noisy-AND: SUCCESS
# credits every backbone edge at FULL trial weight (the whole path operated);
# FAILURE/PARTIAL SPLIT the (modest) failure mass toward the currently-uncertain
# edges (explaining-away) so no single trial collapses an edge and curated
# binds_to edges self-protect. ``EROOM_EDGE_ATTR`` lets a measurement harness
# swap that per-edge SPLIT for symmetric variants so the owner can SEE how the
# choice shapes the edge-belief distribution (the modes change only the per-edge
# SHARE of the trial mass, never the outcome-determined p_obs strength):
#   explain_away      (default) success=full, failure=explaining-away split
#   symmetric_full    both directions full per-edge weight = pure per-edge freq
#   symmetric_uniform both directions split 1/L (uniform mass across the chain)
#   symmetric_explain explaining-away weighting for BOTH success and failure
_EDGE_ATTR_MODES = (
    "explain_away", "symmetric_full", "symmetric_uniform", "symmetric_explain",
)


def _edge_attr_mode() -> str:
    """Read EROOM_EDGE_ATTR (default ``explain_away``); read at call time so a
    harness can re-attribute the same graph under several modes in one process."""
    mode = os.environ.get("EROOM_EDGE_ATTR", "explain_away").strip()
    if mode not in _EDGE_ATTR_MODES:
        raise ValueError(
            f"EROOM_EDGE_ATTR={mode!r} not in {_EDGE_ATTR_MODES}"
        )
    return mode


def _edge_effect_enabled() -> bool:
    """Read EROOM_EDGE_EFFECT (default off). When on, the trial's quantitative
    effect_size + p_value modulate the update STRENGTH (p_obs) and PRECISION
    (n_eff) around the outcome-determined neutral — see ``_effect_modulation``."""
    return os.environ.get("EROOM_EDGE_EFFECT", "").strip().lower() in (
        "1", "on", "true", "yes",
    )


# ── Pillar A (A3 + A4): reason-routed EM with competing-risks censoring ────
#
# EROOM_ROUTING (default OFF — current behavior byte-identical) wires the
# 13-category failure reason into the backbone update as competing-risks
# censoring (EM doc §3.2) and replaces the heuristic explaining-away split
# with the principled normalized responsibility (§3.1 / §4, A4). When OFF,
# `_condition_chain_on_outcomes` is untouched. When ON, a failed/partial
# trial routes by `taxonomy.routing_branch_for(primary_failure_mode)`:
#   EFFICACY / MEASUREMENT → responsibility update (blame within M_t)
#   SAFETY / OPERATIONAL   → CENSOR the efficacy+measurement backbone
#   UNKNOWN                → fall back to the existing explaining-away path
# plus a safety-gate SURVIVAL credit (b += w) on readout-reaching trials.
def _routing_enabled() -> bool:
    """Whether the reason-routed EM path (A3 + A4) is active.

    Env-gated, default off; read at call time so an A/B harness can attribute
    the same initial graph both ways in one process."""
    return os.environ.get("EROOM_ROUTING", "").strip().lower() in (
        "1", "on", "true", "yes",
    )


# Degenerate-chain guard for the A4 responsibility denominator (1 − M). When
# every must-hold edge is already near-certain, M = ∏ r_a → 1 and (1 − M) → 0:
# the failure carries no information about which edge broke, so the update is
# skipped rather than dividing by ~0.
_RESPONSIBILITY_M_EPS = 1e-6

# p_obs label/override for the safety-gate SURVIVAL credit (b += w). The
# conjugate step uses p_obs=0 (all mass to β = "did not fire"); STRONG_CONTRADICT
# is the matching replay-display bucket on the EvidenceRecord.
_SURVIVAL_P_OBS = 0.0


def _per_edge_fracs(
    mode: str, is_success: bool, explain_fracs: list[float], n_edges: int,
) -> list[float]:
    """Per-edge SHARE of the trial mass for the chosen ``EROOM_EDGE_ATTR`` mode.

    ``explain_fracs`` is the explaining-away split (u_i/Σu, uniform if all u==0)
    already computed from the live edges' pre-update E[p]. The dispatch keeps the
    default path byte-identical to the prior hard-coded behavior.
    """
    full = [1.0] * n_edges
    uniform = [1.0 / n_edges] * n_edges
    if mode == "explain_away":
        return full if is_success else explain_fracs
    if mode == "symmetric_full":
        return full
    if mode == "symmetric_uniform":
        return uniform
    if mode == "symmetric_explain":
        return explain_fracs
    raise ValueError(f"unknown edge-attr mode {mode!r}")  # pragma: no cover


def _effect_modulation(
    p_obs: float, effect: float | None, p_value: float | None,
) -> tuple[float, float]:
    """TASK 2b — fold the trial's quantitative significance into (p_obs, n_eff).

    SIGN-FREE by construction: the coarse 3-way OUTCOME already fixed the update
    DIRECTION (success pushes p_obs>0.5, failure <0.5). The quantitative signal
    only refines how FAR from the neutral 0.5 (strength) and how PRECISE (n_eff)
    the update is — no HR<1-vs-OR>1 normalization needed. Returns
    ``(p_obs_adj, neff_scale)``; identity (p_obs, 1.0) when the signal is absent.

    USES p_value ONLY. The ``effect`` arg is accepted for signature stability but
    DELIBERATELY UNUSED: the experiment (scripts/edge_attr_experiment.py) proved
    the extractor's ``effect_size`` is a bare first-number parse of a free-text
    string that conflates hazard ratios (``HR 0.56``), percentages (``41.3% vs
    54.0% pCR``), point differences (``3.8 point improvement``) and raw counts
    (``11 discontinuations``) — range −1e5…2.4e6, median 6. There is no scale on
    which a single float means the same thing across trials, so any magnitude
    rule (even a ratio gate) mislabels percentages/counts as ratios. Folding it
    in is actively WRONG, not merely noisy. The real fix is upstream: the
    extractor must emit a STRUCTURED effect (metric_type + direction-normalized
    magnitude + CI) — root-cause #1/#2 (ingestion / data→node mapping), tracked
    separately. Until then only p_value (present on ~26% of trials, but
    semantically uniform) is safe.

      p_value → significance/precision. A small p is stronger, more precise
        evidence; a non-significant p (>0.10) is weak/ambiguous regardless of the
        coarse label, so it pulls p_obs toward 0.5 AND shrinks n_eff.
    """
    del effect  # see docstring — unreliable, intentionally not used
    strength = 0.0       # signed [-1,1]: + sharpens p_obs away from 0.5
    neff_scale = 1.0
    if p_value is not None and 0.0 <= p_value <= 1.0:
        if p_value <= 0.001:
            s = 1.0
        elif p_value <= 0.01:
            s = 0.6
        elif p_value <= 0.05:
            s = 0.3
        elif p_value <= 0.10:
            s = 0.0
        else:
            s = -0.6  # non-significant — weak/ambiguous evidence
        strength += s
        neff_scale *= 1.0 + 0.5 * s          # ∈ [0.7, 1.5]
    strength = max(-1.0, min(1.0, strength))
    p_obs_adj = 0.5 + (p_obs - 0.5) * (1.0 + 0.30 * strength)
    p_obs_adj = max(0.05, min(0.95, p_obs_adj))
    neff_scale = max(0.5, min(1.6, neff_scale))
    return p_obs_adj, neff_scale


def _chain_backbone_edges(chain: CausalChain) -> list[tuple[str, str, EdgeType]]:
    """All (src_id, tgt_id, edge_type) backbone edges this chain implies.

    Walks the canonical causal hypothesis:
        compound → target          (AFFECTS)
        target   → mechanism       (MODULATES_VIA)
        mechanism→ biology         (MECHANISM_AFFECTS)
        biology  → indication      (BIOLOGY_DRIVES)
    and, when an endpoint is present:
        biology  → endpoint        (REFLECTS_BIOLOGY)
        endpoint → indication      (ENDPOINT_CAPTURES)

    Placeholder (``UNKNOWN``) ids are skipped — an edge can't be conditioned
    if one of its endpoints was never resolved. The caller additionally
    gates on the edge actually existing in the graph (a Beta belief lives
    there). RESPONDS_DIFFERENTLY is intentionally NOT walked here: it's the
    population→indication edge whose evidence comes from the always-on
    observable-statistics emission, not from conditioning the mechanism
    chain on the arm's outcome.
    """
    out: list[tuple[str, str, EdgeType]] = []

    def _add(src: str, tgt: str, et: EdgeType) -> None:
        if not src or not tgt:
            return
        if src == _UNKNOWN_PLACEHOLDER or tgt == _UNKNOWN_PLACEHOLDER:
            return
        out.append((src, tgt, et))

    _add(chain.compound_id, chain.target_id, EdgeType.AFFECTS)
    _add(chain.target_id, chain.mechanism_id, EdgeType.MODULATES_VIA)
    _add(chain.mechanism_id, chain.biology_id, EdgeType.MECHANISM_AFFECTS)
    _add(chain.biology_id, chain.indication_id, EdgeType.BIOLOGY_DRIVES)
    if chain.endpoint_id and chain.endpoint_id != _UNKNOWN_PLACEHOLDER:
        _add(chain.biology_id, chain.endpoint_id, EdgeType.REFLECTS_BIOLOGY)
        _add(chain.endpoint_id, chain.indication_id, EdgeType.ENDPOINT_CAPTURES)
    return out


def _trial_level_outcome(
    classification: "FailureClassification",
) -> "TrialOutcome | None":
    """The classifier's trial-LEVEL outcome, or None.

    Read from ``classification._raw["trial_outcome"]`` (the classifier emits
    ``success`` / ``failure`` / ``partial`` / ``unknown`` there). Used as the
    coarsest conditioning signal when per-arm outcomes don't resolve. Returns
    None for ``unknown`` / missing / unparseable values.
    """
    raw = getattr(classification, "_raw", {}) or {}
    value = raw.get("trial_outcome")
    if not value:
        return None
    try:
        outcome = TrialOutcome(value)
    except ValueError:
        return None
    if outcome == TrialOutcome.UNKNOWN:
        return None
    return outcome


# TrialOutcome → ordinal scale for arm-differential bucket mapping.
# UNKNOWN is excluded — pairs involving an unknown outcome yield no
# modulation evidence (returned as None below).
_OUTCOME_SCALE: dict[TrialOutcome, int] = {
    TrialOutcome.FAILURE: 0,
    TrialOutcome.PARTIAL: 1,
    TrialOutcome.SUCCESS: 2,
}


def _arm_differential_bucket(
    backbone_outcome: TrialOutcome,
    combo_outcome: TrialOutcome,
) -> SupportBucket | None:
    """Map (backbone arm, combo arm) outcomes to a support bucket.

    Bucket describes "adding the extra constituents to the backbone
    helps." A 2-step jump (e.g. failure → success) is strong; 1-step is
    moderate; equal outcomes is ambiguous; negative deltas contradict.
    Returns None when either arm's outcome is UNKNOWN — no differential
    can be computed.
    """
    if (
        backbone_outcome == TrialOutcome.UNKNOWN
        or combo_outcome == TrialOutcome.UNKNOWN
    ):
        return None
    delta = _OUTCOME_SCALE[combo_outcome] - _OUTCOME_SCALE[backbone_outcome]
    if delta == 2:
        return SupportBucket.STRONG_SUPPORT
    if delta == 1:
        return SupportBucket.MODERATE_SUPPORT
    if delta == 0:
        return SupportBucket.AMBIGUOUS
    if delta == -1:
        return SupportBucket.MODERATE_CONTRADICT
    if delta == -2:
        return SupportBucket.STRONG_CONTRADICT
    return None


def _single_arm_combo_bucket(
    outcome: TrialOutcome,
) -> SupportBucket | None:
    """Map a standalone combo-arm outcome to a weak pairwise bucket.

    No head-to-head signal, so even a clearly successful or failed combo
    only contributes a weak update to each pairwise modulation edge.
    """
    if outcome == TrialOutcome.SUCCESS:
        return SupportBucket.WEAK_SUPPORT
    if outcome == TrialOutcome.PARTIAL:
        return SupportBucket.AMBIGUOUS
    if outcome == TrialOutcome.FAILURE:
        return SupportBucket.WEAK_CONTRADICT
    return None


def _aggregate_arm_outcomes(
    trial: TrialSubgraph,
    extraction: "TrialExtraction | None" = None,
) -> dict[str, TrialOutcome]:
    """Per-arm authoritative outcome, keyed by graph arm_id.

    Reads from extraction.results_by_chain when provided. Resolution of
    each result's arm_id to a graph arm_id is two-stage:

      1. DIRECT arm_id match (preferred). Round-9 arm-id alignment made
         the CT.gov ``group_id`` the single source of truth — both the
         extractor and the populator emit it verbatim — so a
         ``ChainResult.arm_id`` is usually already a graph arm_id.
         A direct match is the most reliable signal and crucially does
         NOT depend on canonicalizing free-text compound names (which
         fails for codename / cell-therapy drugs like "RO7247669" or
         "Drosophila-peptide pulsed ... autologous CD8+ PBL").
      2. COMPOUND-SET match (fallback). When the LLM ignored the menu and
         invented its own arm slug, fall back to matching the extraction
         arm's normalized compound set against a graph arm's compound ids
         (``_map_extraction_arms_to_graph``).

    chain.outcome is never written back from the extraction in the
    current populate flow, so it stays UNKNOWN even when the extraction
    reports per-arm outcomes — hence the extraction path above is primary.

    Falls back to ``chain.outcome`` when the extraction is unavailable
    or doesn't yield a mapping — exercised by unit tests that fixture
    chain outcomes directly.
    """
    by_arm: dict[str, TrialOutcome] = {}
    if extraction is not None:
        graph_arm_ids = {arm.arm_id for arm in trial.arms}
        ext_to_graph = _map_extraction_arms_to_graph(trial, extraction)
        for cr in getattr(extraction, "results_by_chain", []) or []:
            # Parent-population results only (subgroup_descriptor is null).
            if cr.subgroup_descriptor is not None:
                continue
            # Stage 1: direct arm_id match (round-9 alignment); Stage 2:
            # compound-set fallback.
            if cr.arm_id in graph_arm_ids:
                graph_arm_id = cr.arm_id
            else:
                graph_arm_id = ext_to_graph.get(cr.arm_id)
            if graph_arm_id is None:
                continue
            try:
                outcome = TrialOutcome(cr.outcome)
            except ValueError:
                continue
            if outcome == TrialOutcome.UNKNOWN:
                continue
            by_arm.setdefault(graph_arm_id, outcome)
        if by_arm:
            return by_arm

    # Fallback: chain.outcome (used by tests + as a defensive path).
    for chain in trial.chains:
        if chain.subgroup_population_id != trial.parent_population_id:
            continue
        if chain.outcome == TrialOutcome.UNKNOWN:
            continue
        by_arm.setdefault(chain.arm_id, chain.outcome)
    for chain in trial.chains:
        if chain.arm_id in by_arm:
            continue
        if chain.outcome == TrialOutcome.UNKNOWN:
            continue
        by_arm[chain.arm_id] = chain.outcome
    return by_arm


def _map_extraction_arms_to_graph(
    trial: TrialSubgraph,
    extraction: "TrialExtraction",
) -> dict[str, str]:
    """ext_arm_id → graph_arm_id by compound-set match.

    Extraction arms carry LLM-emitted arm_ids ("aldesleukin_alone") plus
    free-text compound names ("aldesleukin", "gp100 antigen"). Graph
    arms carry CT.gov group_ids ("arm_i_aldesleukin") plus canonical
    compound ids ("aldesleukin", "gp100_antigen"). The reconciliation
    normalizes the extraction's compound names through `normalize_entity`
    and matches by set equality.
    """
    from src.graph.models import normalize_entity

    graph_arm_by_compounds: dict[frozenset[str], str] = {
        frozenset(arm.compound_ids): arm.arm_id for arm in trial.arms
    }
    mapping: dict[str, str] = {}
    for ea in getattr(extraction, "arms", []) or []:
        compounds = ea.compounds or []
        if not compounds:
            continue
        normalized = frozenset(
            normalize_entity(c, "InterventionNode") for c in compounds
        )
        graph_arm_id = graph_arm_by_compounds.get(normalized)
        if graph_arm_id is None:
            continue
        mapping[ea.arm_id] = graph_arm_id
    return mapping


# ── Routing helpers ──────────────────────────────────────────────────────


_NON_ALNUM_RE = __import__("re").compile(r"[^a-z0-9]+")


def _norm_name(text: str) -> str:
    """Lowercase, strip non-alphanumerics. PD-1 / PD1 / pd_1 → 'pd1'."""
    return _NON_ALNUM_RE.sub("", (text or "").lower())


def _log_unrouted_modulation(
    trial_id: str, entry: ModulationEntry, *, reason: str,
) -> None:
    """Append a record to the v0.3.0 unrouted-modulation log.

    A high volume of records here indicates the extraction prompt is
    emitting compound names that aren't in the trial (or the primary
    chain layer being named can't be resolved). Surface for debugging
    rather than silently dropping the modulation.
    """
    _UNROUTED_MOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "modulator_compound_id": entry.modulator_compound_id,
        "primary_compound_id": entry.primary_compound_id,
        "affects_layer": entry.affects_layer,
        "direction": entry.direction,
        "confidence": entry.confidence,
        "reason": reason,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with _UNROUTED_MOD_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")




# ── AE attribution helpers ──────────────────────────────────────────────


def _hr_support_bucket(
    hr: float | None, ci_low: float | None, ci_high: float | None,
) -> SupportBucket:
    """Grade an AE from a hazard/risk ratio + 95% CI (literature-derived AEs).

    The CI is the SIGNIFICANCE gate: if it spans 1.0 (or is missing) the effect
    is indistinguishable from background -> AMBIGUOUS. When the CI excludes 1.0,
    grade by magnitude:
        HR >= 1.5 (<= 0.67) -> STRONG; >= 1.25 (<= 0.80) -> MODERATE; else WEAK.
    This captures rare-but-decisive endpoints — a trial-terminating mortality
    HR 1.58 (95% CI 1.14-2.19) is STRONG here, vs WEAK on the absolute-rate path
    (delta 0.45pp). First-pass calibration on principle (significance gate +
    relative-effect tiers), NOT tuned on the 5-trial holdout; refit downstream of
    the calibration harness alongside the rate cutoffs.
    """
    if hr is None or ci_low is None or ci_high is None:
        return SupportBucket.AMBIGUOUS
    if ci_low <= 1.0 <= ci_high:
        return SupportBucket.AMBIGUOUS  # CI spans 1.0 — not significant
    if hr >= 1.0:
        if hr >= 1.5:
            return SupportBucket.STRONG_SUPPORT
        if hr >= 1.25:
            return SupportBucket.MODERATE_SUPPORT
        return SupportBucket.WEAK_SUPPORT
    if hr <= 0.67:
        return SupportBucket.STRONG_CONTRADICT
    if hr <= 0.80:
        return SupportBucket.MODERATE_CONTRADICT
    return SupportBucket.WEAK_CONTRADICT


def _ae_support_bucket(
    treatment_pct: float | None,
    control_pct: float | None,
    treatment_n: int | None = None,
    *,
    hazard_ratio: float | None = None,
    hr_ci_low: float | None = None,
    hr_ci_high: float | None = None,
) -> SupportBucket:
    """Map per-arm AE incidence to a SupportBucket for the causes_ae edge.

    No treatment-arm rate → AMBIGUOUS (we can't even say the trial saw it).
    Control rate missing is treated as 0—a conservative read that says
    "no reported background", which lets unilateral safety signals from
    single-arm Phase 1s contribute (with the bucket downgrade reflecting
    the missing comparator).

    Rate thresholds (deliberately coarse):
      - delta ≥ 20pp OR RR ≥ 3 → strong_support
      - delta ≥ 10pp OR RR ≥ 2 → moderate_support
      - delta ≥ 5pp  OR RR ≥ 1.5 → weak_support
      - delta ≤ -5pp           → weak_contradict (drug arm safer than control)
      - otherwise              → ambiguous (background rate)

    Absolute-count gate (fixes.md #9): a 1.2% rate in an n=85 arm is
    ~1 patient, which the rate-only path would label moderate_support
    via RR ≥ 2. That's noise being graded as evidence. With
    ``treatment_n`` available we additionally require:
      - ≥ 5 affected patients to remain at strong / moderate support
      - ≥ 3 affected patients to remain at weak support
      - otherwise downgrade to AMBIGUOUS

    Calibration of these cutoffs is downstream of the calibration harness
    (NEXT_SESSION follow-up #1) just like the bucket→p_obs table.

    A hazard/risk ratio + CI (literature-derived AEs) takes precedence via
    ``_hr_support_bucket`` — a significant relative effect on a rare hard
    endpoint shouldn't be lost to a small absolute incidence delta.
    """
    if hazard_ratio is not None:
        return _hr_support_bucket(hazard_ratio, hr_ci_low, hr_ci_high)
    if treatment_pct is None:
        return SupportBucket.AMBIGUOUS
    c = control_pct if control_pct is not None else 0.0
    delta = treatment_pct - c
    # 0.5pp floor on the denominator avoids div-by-zero and keeps RR
    # finite when control = 0; an AE seen in 30% of treated patients
    # with 0% control still gets RR = 60, well into strong territory.
    rr = treatment_pct / max(c, 0.5)

    if delta >= 20 or rr >= 3:
        bucket = SupportBucket.STRONG_SUPPORT
    elif delta >= 10 or rr >= 2:
        bucket = SupportBucket.MODERATE_SUPPORT
    elif delta >= 5 or rr >= 1.5:
        bucket = SupportBucket.WEAK_SUPPORT
    elif delta <= -5:
        bucket = SupportBucket.WEAK_CONTRADICT
    else:
        bucket = SupportBucket.AMBIGUOUS

    if treatment_n is None or bucket == SupportBucket.AMBIGUOUS:
        return bucket

    # Absolute-count gate. round to ensure 1.2% × 85 = 1.02 → 1 patient,
    # not 1.02 → moderate_support survives the float comparison.
    abs_count = round(treatment_pct * treatment_n / 100.0)
    if bucket in (SupportBucket.STRONG_SUPPORT, SupportBucket.MODERATE_SUPPORT):
        if abs_count < 5:
            if abs_count >= 3:
                return SupportBucket.WEAK_SUPPORT
            return SupportBucket.AMBIGUOUS
    elif bucket == SupportBucket.WEAK_SUPPORT:
        if abs_count < 3:
            return SupportBucket.AMBIGUOUS
    elif bucket == SupportBucket.WEAK_CONTRADICT:
        # Drop in incidence: gate on absolute affected count in the
        # CONTROL arm so a 5pp drop from "5% of 5 control patients" to
        # 0% in treatment isn't graded as a real safety improvement.
        ctrl_abs = round(c * treatment_n / 100.0)
        if ctrl_abs < 3:
            return SupportBucket.AMBIGUOUS
    return bucket


def _per_compound_rates_from_arms(
    arm_incidences: list[ArmIncidence],
    compound_name: str,
) -> tuple[float | None, float | None, int | None]:
    """Partition per-arm AE counts on whether ``compound_name`` was active.

    Arms whose ``arm_descriptor`` contains the compound name (word-boundary,
    case-insensitive) contribute to the treatment pool; arms without it
    contribute to the comparator pool. Returns ``(tx_pct, ctrl_pct, tx_n)``
    where the pcts are pooled rates and tx_n is the total at-risk
    population for the compound — fed to ``_ae_support_bucket`` for the
    absolute-count gate.

    Returns (None, None, None) when no arm matched the compound name —
    callers should fall back to the legacy flat ``tx/ctrl`` pair in that
    case.
    """
    if not compound_name:
        return (None, None, None)
    pattern = re.compile(rf"\b{re.escape(compound_name)}\b", re.IGNORECASE)
    tx_affected = tx_at_risk = 0
    ctrl_affected = ctrl_at_risk = 0
    for ai in arm_incidences:
        if pattern.search(ai.arm_descriptor):
            tx_affected += ai.n_affected
            tx_at_risk += ai.n_at_risk
        else:
            ctrl_affected += ai.n_affected
            ctrl_at_risk += ai.n_at_risk
    if tx_at_risk == 0:
        return (None, None, None)
    tx_pct = 100.0 * tx_affected / tx_at_risk
    ctrl_pct = 100.0 * ctrl_affected / ctrl_at_risk if ctrl_at_risk > 0 else None
    return (tx_pct, ctrl_pct, tx_at_risk)


def _format_ae_note(
    ae: StructuredAE,
    preferred_term: str,
    *,
    tx_pct: float | None = None,
    ctrl_pct: float | None = None,
) -> str:
    bits = [f"AE: {preferred_term} (raw: {ae.term!r})"]
    if ae.grade:
        bits.append(f"grade {ae.grade}")
    # Per-compound rates (computed by attributor from arm_incidences)
    # win over the legacy flat fields; only fall back when missing.
    tx = tx_pct if tx_pct is not None else ae.incidence_treatment_pct
    ctrl = ctrl_pct if ctrl_pct is not None else ae.incidence_control_pct
    if tx is not None:
        bits.append(f"tx {tx:g}%")
    if ctrl is not None:
        bits.append(f"ctrl {ctrl:g}%")
    if ae.serious:
        bits.append("SAE")
    return "; ".join(bits)


# ── Output model ─────────────────────────────────────────────────────────


class AppliedEdgeUpdate(BaseModel):
    """A single edge update that was applied to the graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    evidence: EvidenceRecord
    pre_update_belief: EdgeBeliefState
    post_update_belief: EdgeBeliefState

    @property
    def probability_change(self) -> float:
        return (
            self.post_update_belief.expected_probability
            - self.pre_update_belief.expected_probability
        )


# ── Attributor ───────────────────────────────────────────────────────────


class Attributor:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        # Round-16 observability: per-call counters that the orchestrator
        # can read after attribute() to decide whether the classifier's
        # output landed cleanly on the trial subgraph or got dropped
        # excessively (LLM hallucinated entities, schema mismatches,
        # bad arm ids, etc.). The drop count is otherwise only visible
        # as logged warnings, which means a build-level threshold
        # ("abort if > 30% of classifier edges drop") couldn't be
        # enforced without scraping the log.
        self.last_attempted_updates: int = 0
        self.last_dropped_updates: int = 0

    def attribute(
        self,
        classification: FailureClassification,
        trial: TrialSubgraph,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Translate a trial's per-arm outcomes into concrete edge updates.

        Outcome-conditioning redesign: a single trial cannot pinpoint WHICH
        edge of its causal chain failed (that would be premature
        falsification), so the backbone is NOT attributed by name-matching a
        classifier-emitted failing edge to a chain edge. Instead the trial's
        per-arm OUTCOME conditions the WHOLE chain by edge id, and the
        cross-trial OVERLAP (shared pathway nodes) triangulates the
        responsible edge over many trials. Failure ≠ falsification, so
        FAILURE updates are MODEST and spread across the chain
        (explaining-away). See ``_condition_chain_on_outcomes``.

        AE attribution and ALL modulation emissions (arm-differential,
        single-arm-combo, LLM-anchored) are unchanged and still run below.

        ``extraction`` (optional) is the trial's structured extraction; it
        supplies the per-arm outcomes (``results_by_chain``) and the
        trial enrollment N (``sample_size``) used to weight the
        conditioning, and lets modulation emission read per-arm outcomes.
        Production callers should always pass it.
        """
        raw = getattr(classification, "_raw", {})
        # Round 3.3 schema retained ``edges_to_update`` from the classifier,
        # but the backbone is no longer name-matched from it — the trial's
        # per-arm outcome conditions the whole chain. ``raw_edges`` is read
        # only to keep the round-16 attempted-updates counter meaningful.
        raw_edges = raw.get("edges_to_update", [])
        phase = trial.phase
        evidence_type = _PHASE_TO_EVIDENCE.get(phase, EvidenceType.LITERATURE)

        updates: list[AppliedEdgeUpdate] = []
        # Per-trial dedup: an (edge_type, src_id, tgt_id) triple may be
        # named by multiple classifier emissions (e.g. nivolumab → PD-1
        # surfacing on both the mono-nivo and combo-arm chains). Apply
        # the first matching update and skip the rest so a single trial
        # never delivers 2× the conjugate evidence to the same edge.
        #
        # Phase B of v1 classifier-per-arm: the dedupe key includes
        # arm_id as a 4th slot. Two updates that hit the same
        # (edge_type, src, tgt) under DIFFERENT arm_ids are treated as
        # separate evidence records — each arm's outcome about the same
        # edge is its own signal. Null arm_id (back-compat) collapses
        # all unsubscripted updates to one slot.
        applied_edges: set[tuple[str, str, str, str | None]] = set()

        # Round-16: reset per-call counters. Orchestrator reads
        # last_attempted_updates + last_dropped_updates after each
        # attribute() call to enforce build-level drop thresholds. The
        # outcome-conditioning backbone never "drops" (it conditions the
        # chain by id, no name-match step that can miss); the counters are
        # kept for back-compat and driven by the modulation paths.
        self.last_attempted_updates = len(raw_edges)
        self.last_dropped_updates = 0

        # ── Backbone: outcome conditions the WHOLE chain (the core) ─────────
        # Replaces the old name-matched ``for item in raw_edges`` loop. The
        # trial's per-arm outcome is folded onto every backbone edge of every
        # chain on that arm, by edge id. Modulation + AE emission below are
        # unchanged.
        updates.extend(self._condition_chain_on_outcomes(
            trial, classification, extraction, evidence_type, applied_edges,
        ))

        # Arm-differential modulation edges (round 8 v0.2.0). For trials
        # with arm pairs where one is a strict subset of another, emit
        # MODULATES_EFFICACY_OF edges between added constituents and
        # backbone constituents. Compound→compound layer only at v0.2.0;
        # v0.2.1 / v0.3.0 will promote to chain-node layers.
        updates.extend(self._emit_arm_differential_modulations(
            trial, classification, evidence_type, applied_edges,
            extraction=extraction,
        ))

        # Single-arm-combo emission. For each combo arm whose compound
        # set has no subset comparator in the trial, fall back to a
        # weak pairwise signal across all C(n, 2) constituent pairs based
        # on the arm's standalone outcome. Weaker than the differential
        # path because we have no head-to-head signal.
        updates.extend(self._emit_single_arm_combo_modulations(
            trial, classification, evidence_type, applied_edges,
            extraction=extraction,
        ))

        # v0.3.0 LLM-anchored modulation edges. The extractor identifies
        # the specific edge in the primary chain that each modulator
        # acts on; this emission routes them into MODULATES_EFFICACY_OF
        # edges anchored at the layer the LLM named (not the compound
        # layer v0.2.0 fixes them to). Unroutable entries land in
        # data/dev/unrouted_modulation_entries.jsonl.
        updates.extend(self._emit_llm_modulations(
            trial, classification, evidence_type, applied_edges,
            extraction=extraction,
        ))

        # NOTE: the round-14 failure-trial backstop (auto-emit a default
        # biology_drives weak_contradict when the classifier returned zero
        # edges on a failure trial) is removed under the outcome-conditioning
        # redesign. ``_condition_chain_on_outcomes`` ALWAYS conditions every
        # backbone edge of every chain on an arm with a known outcome, so a
        # failure trial can never leave its chains silent — the backstop's
        # job is now structurally guaranteed.

        return updates

    def _condition_chain_on_outcomes(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        extraction: "TrialExtraction | None",
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
    ) -> list[AppliedEdgeUpdate]:
        """Condition every backbone edge of every chain on its arm's outcome.

        THE CORE of the outcome-conditioning redesign. For each arm with a
        known outcome, every chain on that arm has its backbone edges
        (AFFECTS, MODULATES_VIA, MECHANISM_AFFECTS, BIOLOGY_DRIVES, and —
        when an endpoint is present — REFLECTS_BIOLOGY, ENDPOINT_CAPTURES)
        updated by EDGE ID, no name-matching.

        Trial weight::

            w_base = effective_n_for_evidence(evidence_type,
                                              quality=classification.confidence)
                     * f_N            # saturating √N population multiplier
                     * gate_weight    # operational-failure gate (Piece 2)

        ``f_N`` = ``beliefs._precision_multiplier(extraction.sample_size)``
        called DIRECTLY (independent of EROOM_NEFF_PRECISION — the outcome
        path is always f(N)-weighted): concave √N, anchored 350, floored 0.5,
        ceiled 2.5, so a huge trial counts more than a tiny one but not
        linearly more.

        Per-outcome update math:

        SUCCESS (conjunctive — the whole path operated): every backbone edge
            gets a SUPPORT update at full ``w_base`` (p_obs=0.80).

        FAILURE (disjunctive — ≥1 edge weak, unknown which) → EXPLAINING-AWAY.
            For each backbone edge i compute its current ``E[p_i]`` from the
            pre-update belief; set unnormalized weak-weight ``u_i = 1 - E[p_i]``
            and normalize ``w_i = u_i / Σ u_j`` (uniform 1/L if all u_i == 0).
            Edge i gets a MODEST CONTRADICT (p_obs=0.20) with
            ``n_eff = w_base * w_i``. High-E[p] curated edges (a binds_to with
            α≫β) absorb ≈0 of the failure (self-protect); the uncertain causal
            edges absorb most. Because the total failure mass ``w_base`` is
            SPLIT across the chain, one trial can never collapse an edge.
            Symmetric with the softmin/weakest-link prediction.

        PARTIAL: the same explaining-away split as failure but with a WEAKER
            contradict (p_obs=0.35) — modest and ambiguous.

        Dedup: an (edge_type, src, tgt, arm_id) tuple is conditioned once
        per arm even when two chains of that arm share it (reuses
        ``applied_edges``). The EvidenceRecord is always recorded on the edge
        (replayable) even though n_eff/p_obs are injected directly.
        """
        outcomes = _aggregate_arm_outcomes(trial, extraction)
        fallback = False
        if not outcomes:
            # Trial-level-outcome fallback. The extraction couldn't resolve a
            # per-arm outcome (e.g. all results_by_chain entries are
            # ``unknown``, or the arm-ids didn't reconcile), but the
            # classifier reported a TRIAL-LEVEL outcome. A trial-level
            # failure/partial/success IS an outcome and is still the coarsest
            # valid signal to condition the chain on — applying it to every
            # arm's chains is faithful to "the outcome conditions the chain,"
            # and recovers chains that per-arm conditioning would leave silent
            # (the equivalent of the round-16 always-on emission, now driven
            # by the trial outcome rather than name-matched edges).
            trial_outcome = _trial_level_outcome(classification)
            if trial_outcome is None:
                return []
            outcomes = {arm.arm_id: trial_outcome for arm in trial.arms}
            if not outcomes:
                return []
            fallback = True

        n_obs = extraction.sample_size if extraction else None
        f_n = _precision_multiplier(n_obs)
        gate_weight = classification.gate_weight
        quality = min(classification.confidence, 1.0)
        # Type-constant N_eff × quality, exactly as the legacy path computes
        # its base (we call effective_n_for_evidence so the per-source
        # constant + quality discount are honored), then scale by the
        # saturating f(N) and the operational gate.
        base_n = effective_n_for_evidence(evidence_type, quality)
        w_base = base_n * f_n * gate_weight

        # A3 + A4 routing context (no-op unless EROOM_ROUTING is on). The
        # branch is trial-level — resolved once from the classifier's primary
        # failure reason — and decides, per failed/partial chain below, whether
        # to censor, blame-within-backbone, or fall back to the legacy path.
        routing = _routing_enabled()
        reason_branch = routing_branch_for(
            classification.primary_failure_mode
        ) if routing else RoutingBranch.UNKNOWN

        # CT.gov stop-reason override (terminated-trial misroute fix). The LLM
        # classifier never saw the structured CT.gov status, so an early stop for
        # a non-efficacy reason (accrual / funding / sponsor decision / toxicity)
        # may have been routed to EFFICACY/MEASUREMENT — wrongly downvoting the
        # biology. If the cache documents such a stop, reroute to the CENSORING
        # branch. No-op when the cache is absent (default) or the reason is
        # efficacy/ambiguous, so reproducibility is preserved.
        if routing:
            _st = _ctgov_status_for(trial.trial_id)
            if _st:
                _ov = stop_reason_override(
                    _st.get("overall_status"), _st.get("why_stopped")
                )
                if _ov is not None and _ov[0] != reason_branch:
                    _new_branch, _cat = _ov
                    logger.info(
                        "ctgov stop-reason override %s: %s → %s (%s) — %r",
                        trial.trial_id, reason_branch.value, _new_branch.value,
                        _cat, (_st.get("why_stopped") or "")[:120],
                    )
                    try:
                        _STOP_OVERRIDE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with _STOP_OVERRIDE_LOG_PATH.open("a") as _fh:
                            _fh.write(json.dumps({
                                "nct": trial.trial_id,
                                "from": reason_branch.value,
                                "to": _new_branch.value,
                                "category": _cat,
                                "overall_status": _st.get("overall_status"),
                                "why_stopped": _st.get("why_stopped"),
                                "primary_failure_mode": (
                                    classification.primary_failure_mode.value
                                    if classification.primary_failure_mode else None
                                ),
                            }) + "\n")
                    except OSError:
                        pass
                    reason_branch = _new_branch

        emitted: list[AppliedEdgeUpdate] = []
        for chain in trial.chains:
            outcome = outcomes.get(chain.arm_id)
            if outcome is None or outcome == TrialOutcome.UNKNOWN:
                continue

            # Dedup scope. Normally per-arm: each arm's outcome is independent
            # evidence, so a backbone edge shared by two arms is conditioned
            # once per arm (both votes count). Under the trial-level FALLBACK,
            # every arm carries the SAME trial-level guess, so conditioning a
            # shared downstream edge once per arm would apply that single guess
            # N_arms times — an over-count. Dedup arm-independently (None) in
            # fallback so each unique edge is conditioned exactly once.
            dedup_arm = None if fallback else chain.arm_id

            # Collect the chain's backbone edges that actually exist in the
            # graph (skip placeholders + missing edges). The explaining-away
            # normalization is over THIS set.
            live_edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []
            for src_id, tgt_id, et in _chain_backbone_edges(chain):
                edge_key = (et.value, src_id, tgt_id, dedup_arm)
                if edge_key in applied_edges:
                    continue
                try:
                    pre = self.graph.get_edge_belief(src_id, tgt_id, et)
                except KeyError:
                    continue
                live_edges.append((src_id, tgt_id, et, pre))

            if not live_edges:
                continue

            is_success = outcome == TrialOutcome.SUCCESS

            # ── A3 + A4: reason-routed competing-risks update (EROOM_ROUTING) ──
            # SUCCESS is NEVER routed — the whole path operated, so every backbone
            # edge gets the full upvote below (unchanged). A failed / partial
            # trial routes by its failure reason (the untested lever, FINDINGS P4):
            if routing and not is_success:
                if reason_branch in (
                    RoutingBranch.SAFETY, RoutingBranch.OPERATIONAL,
                ):
                    # Competing-risks CENSOR. A safety death (the trial never
                    # revealed whether its biology would have worked) or an
                    # operational stop (the chain was never properly tested)
                    # applies ZERO virtual evidence to the efficacy+measurement
                    # backbone — not a downvote, not an upvote. We do NOT mark
                    # applied_edges, so a *different* arm of the same trial with a
                    # real (non-censored) outcome can still condition a shared
                    # edge. AE edges are handled separately (safety: the existing
                    # attribute_adverse_events occurrence path; operational: no
                    # safety-survival credit either — see _credit_safety_survival).
                    continue
                if reason_branch in (
                    RoutingBranch.EFFICACY, RoutingBranch.MEASUREMENT,
                ):
                    # A4: blame within the must-hold backbone via the normalized
                    # posterior responsibility (denom 1 − M), crediting reliable
                    # edges for surviving and blaming the unreliable ones.
                    emitted.extend(self._apply_responsibility_update(
                        trial.trial_id, live_edges, w_base, dedup_arm, chain,
                        outcome, evidence_type, quality, n_obs, extraction,
                        applied_edges, reason_branch,
                    ))
                    continue
                # RoutingBranch.UNKNOWN → fall through to the existing
                # full-spread explaining-away path below (the §3.1 unrouted
                # responsibility the current code already approximates).

            # Per-edge fractions + p_obs by outcome.
            #   ``fracs[i]`` is edge i's SHARE of the trial mass w_base;
            #   ``n_eff_i = w_base * fracs[i]``.
            # The explaining-away split (u_i/Σu, uniform if all u==0) is always
            # computed: the default mode uses it for FAILURE/PARTIAL, and the
            # EROOM_EDGE_ATTR symmetric variants may use it for SUCCESS too.
            us = [max(0.0, 1.0 - pre.expected_probability)
                  for (_, _, _, pre) in live_edges]
            total_u = sum(us)
            if total_u <= 0.0:
                explain_fracs = [1.0 / len(live_edges)] * len(live_edges)
            else:
                explain_fracs = [u / total_u for u in us]
            fracs = _per_edge_fracs(
                _edge_attr_mode(), is_success, explain_fracs, len(live_edges),
            )
            if is_success:
                p_obs = _SUCCESS_P_OBS
            else:
                p_obs = (
                    _FAILURE_P_OBS if outcome == TrialOutcome.FAILURE
                    else _PARTIAL_CONTRADICT_P_OBS
                )

            # TASK 2b: fold the trial's quantitative effect_size + p_value into
            # the update strength (p_obs) + precision (n_eff_scale). Sign-free —
            # the outcome already fixed the direction. Off unless EROOM_EDGE_EFFECT.
            neff_scale = 1.0
            if _edge_effect_enabled() and extraction is not None:
                p_obs, neff_scale = _effect_modulation(
                    p_obs, extraction.effect_size, extraction.p_value,
                )

            support_bucket = _OUTCOME_TO_SUPPORT_BUCKET[outcome]
            for (src_id, tgt_id, et, pre), w_i in zip(live_edges, fracs):
                n_eff_i = w_base * w_i * neff_scale
                applied_edges.add((et.value, src_id, tgt_id, dedup_arm))
                evidence = EvidenceRecord(
                    source_id=trial.trial_id,
                    source_type=evidence_type,
                    support=support_bucket.value,
                    quality_score=quality,
                    timestamp=datetime.now(timezone.utc),
                    notes=(
                        f"outcome-conditioned (arm={chain.arm_id}, "
                        f"outcome={outcome.value}, w_base={w_base:.3f}, "
                        f"w_i={w_i:.3f}, n_eff={n_eff_i:.3f}, "
                        f"p_obs={p_obs:.2f}, gate={gate_weight:.2f}, "
                        f"f_N={f_n:.3f})"
                    ),
                    n_obs=n_obs,
                    effect=extraction.effect_size if extraction else None,
                    p_value=extraction.p_value if extraction else None,
                    context={
                        "outcome_conditioned": True,
                        "arm_id": chain.arm_id,
                        "outcome": outcome.value,
                        "explain_away_weight": w_i,
                        "n_eff_applied": n_eff_i,
                        "p_obs_applied": p_obs,
                        "gate_weight": gate_weight,
                        **_direction_ctx(chain),
                    },
                )
                post = self.graph.update_edge_belief(
                    src_id, tgt_id, et, evidence,
                    n_eff_override=n_eff_i, p_obs_override=p_obs,
                )
                emitted.append(AppliedEdgeUpdate(
                    source_id=src_id,
                    target_id=tgt_id,
                    edge_type=et,
                    evidence=evidence,
                    pre_update_belief=pre,
                    post_update_belief=post,
                ))

        # A3: safety-gate SURVIVAL credit (b += w). A trial that ran to readout
        # — any successful arm, or an efficacy/measurement death (ran, missed) —
        # is informative about safety in the GOOD direction: its safety gates did
        # not trigger a halt (§3.2 "survived → b_k += 1", weighted by w here).
        # Safety deaths (gate FIRED → handled by attribute_adverse_events) and
        # operational/unknown stops (censored) get no survival credit.
        if routing:
            survived = (
                any(o == TrialOutcome.SUCCESS for o in outcomes.values())
                or reason_branch in (
                    RoutingBranch.EFFICACY, RoutingBranch.MEASUREMENT,
                )
            )
            if survived:
                emitted.extend(self._credit_safety_survival(
                    trial, w_base, evidence_type, quality, n_obs,
                    reason_branch, applied_edges,
                ))
        return emitted

    def _apply_responsibility_update(
        self,
        trial_id: str,
        live_edges: list[tuple[str, str, EdgeType, EdgeBeliefState]],
        w_base: float,
        dedup_arm: str | None,
        chain: CausalChain,
        outcome: TrialOutcome,
        evidence_type: EvidenceType,
        quality: float,
        n_obs: int | None,
        extraction: "TrialExtraction | None",
        applied_edges: set[tuple[str, str, str, str | None]],
        reason_branch: RoutingBranch,
    ) -> list[AppliedEdgeUpdate]:
        """A4: principled normalized-responsibility update for a routed
        EFFICACY/MEASUREMENT failure (EM doc §3.2, efficacy-death branch).

        Let ``M = ∏_{a∈M_t} r_a`` over the must-hold backbone edges present in
        the chain (``live_edges``), with ``r_a = E[p_a]`` the current posterior
        mean. For each must-hold edge ``a`` the failure splits its full evidence
        weight ``w`` between blame and partial-success credit by the posterior
        responsibility::

            β_a += w · (1 − r_a) / (1 − M)        # failure responsibility ρ_a
            α_a += w · (r_a − M) / (1 − M)        # P(f_a = 1 | fail) credit

        Since ρ_a + P(f_a=1|fail) = (1−M)/(1−M) = 1, each edge receives exactly
        ``w`` total mass — so this reuses ``apply_virtual_evidence`` with
        ``n_eff = w`` and ``p_obs = (r_a − M)/(1 − M)`` (the α-share). A
        high-reliability edge (r_a → 1) gets p_obs → 1 (credited for surviving);
        a low-reliability edge absorbs the blame (β-delta monotone in 1 − r_a).
        This REPLACES the heuristic ``u_i = 1 − E[p_i]`` explaining-away split —
        same machinery, correct math, and now reason-gated.

        Guard: if ``1 − M < ε`` (degenerate all-reliable chain) the failure
        carries no information about which edge broke, so skip (no update).
        """
        rs = [pre.expected_probability for (_, _, _, pre) in live_edges]
        m = 1.0
        for r in rs:
            m *= r
        denom = 1.0 - m
        if denom < _RESPONSIBILITY_M_EPS:
            # All-reliable chain — (1 − M) → 0, responsibilities undefined /
            # uninformative. Mark applied (so the legacy fall-through can't also
            # touch these edges) and emit nothing.
            for src_id, tgt_id, et, _ in live_edges:
                applied_edges.add((et.value, src_id, tgt_id, dedup_arm))
            return []

        emitted: list[AppliedEdgeUpdate] = []
        for (src_id, tgt_id, et, pre), r_a in zip(live_edges, rs):
            # α-share = P(f_a = 1 | fail) = (r_a − M)/(1 − M); β-share is the
            # complement = ρ_a = (1 − r_a)/(1 − M). Clamp for floating-point
            # safety (M ≤ r_a ≤ 1 guarantees the exact value is in [0, 1]).
            p_obs = max(0.0, min(1.0, (r_a - m) / denom))
            n_eff_i = w_base
            applied_edges.add((et.value, src_id, tgt_id, dedup_arm))
            rho = max(0.0, min(1.0, (1.0 - r_a) / denom))
            evidence = EvidenceRecord(
                source_id=trial_id,
                source_type=evidence_type,
                support=_OUTCOME_TO_SUPPORT_BUCKET[outcome].value,
                quality_score=quality,
                timestamp=datetime.now(timezone.utc),
                notes=(
                    f"routed-responsibility (arm={chain.arm_id}, "
                    f"branch={reason_branch.value}, outcome={outcome.value}, "
                    f"r_a={r_a:.3f}, M={m:.4f}, rho={rho:.3f}, "
                    f"w={w_base:.3f}, p_obs={p_obs:.3f})"
                ),
                n_obs=n_obs,
                effect=extraction.effect_size if extraction else None,
                p_value=extraction.p_value if extraction else None,
                context={
                    "outcome_conditioned": True,
                    "routed": True,
                    "routing_branch": reason_branch.value,
                    "arm_id": chain.arm_id,
                    "outcome": outcome.value,
                    "responsibility_rho": rho,
                    "M_backbone": m,
                    "n_eff_applied": n_eff_i,
                    "p_obs_applied": p_obs,
                    **_direction_ctx(chain),
                },
            )
            post = self.graph.update_edge_belief(
                src_id, tgt_id, et, evidence,
                n_eff_override=n_eff_i, p_obs_override=p_obs,
            )
            emitted.append(AppliedEdgeUpdate(
                source_id=src_id,
                target_id=tgt_id,
                edge_type=et,
                evidence=evidence,
                pre_update_belief=pre,
                post_update_belief=post,
            ))
        return emitted

    def _credit_safety_survival(
        self,
        trial: TrialSubgraph,
        w_base: float,
        evidence_type: EvidenceType,
        quality: float,
        n_obs: int | None,
        reason_branch: RoutingBranch,
        applied_edges: set[tuple[str, str, str, str | None]],
    ) -> list[AppliedEdgeUpdate]:
        """A3: credit a did-not-fire (β += w) count to the trial's safety gates.

        The safety gates of a readout-reaching trial are the EXISTING
        ``causes_ae`` edges from its treatment compounds (the compound's known
        AE-liability pathways). Each survived the trial without halting it, so
        each gets ``β += w`` via ``p_obs=0`` (all mass to β). Bounded to
        edges that already exist (no node/edge creation) and deduped once per
        trial. This is the safety-survival signal of §3.2 — DISTINCT from the
        AE OCCURRENCE signal ``attribute_adverse_events`` lands from incidence
        rates (which moves α), and only fires under EROOM_ROUTING.
        """
        emitted: list[AppliedEdgeUpdate] = []
        for compound_id in self._treatment_compound_ids(trial):
            try:
                out_edges = list(
                    self.graph._graph.out_edges(compound_id, keys=True)
                )
            except Exception:  # noqa: BLE001 — missing node ⇒ no gates
                continue
            for _src, ae_id, key in out_edges:
                if key != EdgeType.CAUSES_AE.value:
                    continue
                dedup_key = (key, compound_id, ae_id, "survival")
                if dedup_key in applied_edges:
                    continue
                try:
                    pre = self.graph.get_edge_belief(
                        compound_id, ae_id, EdgeType.CAUSES_AE
                    )
                except KeyError:
                    continue
                applied_edges.add(dedup_key)
                evidence = EvidenceRecord(
                    source_id=trial.trial_id,
                    source_type=evidence_type,
                    support=SupportBucket.STRONG_CONTRADICT.value,
                    quality_score=quality,
                    timestamp=datetime.now(timezone.utc),
                    notes=(
                        f"safety-gate survival (branch={reason_branch.value}, "
                        f"did-not-fire b+={w_base:.3f})"
                    ),
                    n_obs=n_obs,
                    context={
                        "safety_survival": True,
                        "routed": True,
                        "routing_branch": reason_branch.value,
                        "n_eff_applied": w_base,
                        "p_obs_applied": _SURVIVAL_P_OBS,
                    },
                )
                post = self.graph.update_edge_belief(
                    compound_id, ae_id, EdgeType.CAUSES_AE, evidence,
                    n_eff_override=w_base, p_obs_override=_SURVIVAL_P_OBS,
                )
                emitted.append(AppliedEdgeUpdate(
                    source_id=compound_id,
                    target_id=ae_id,
                    edge_type=EdgeType.CAUSES_AE,
                    evidence=evidence,
                    pre_update_belief=pre,
                    post_update_belief=post,
                ))
        return emitted

    def _emit_arm_differential_modulations(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
        *,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Emit MODULATES_EFFICACY_OF edges from arm-pair differentials.

        For each ordered pair of arms (a, b) where a's compound set is a
        strict subset of b's, the outcome differential is evidence about
        adding the extra constituents to a's backbone. Emits one edge
        per (added constituent × backbone constituent) pair, with the
        endpoints lex-canonicalized so symmetric (compound-compound)
        relations accumulate on a single edge across emissions.

        v0.2.0 emits at the compound→compound layer only; the schema
        supports cross-layer edges (compound → target, mechanism →
        biology, etc.) but layer resolution is deferred to v0.2.1 and
        v0.3.0. See `audit/round_8_architecture_design.md`.
        """
        if len(trial.arms) < 2:
            return []
        outcomes = _aggregate_arm_outcomes(trial, extraction)
        if len(outcomes) < 2:
            return []

        emitted: list[AppliedEdgeUpdate] = []
        for arm_a in trial.arms:
            outcome_a = outcomes.get(arm_a.arm_id)
            if outcome_a is None:
                continue
            a_set = set(arm_a.compound_ids)
            for arm_b in trial.arms:
                if arm_a.arm_id == arm_b.arm_id:
                    continue
                outcome_b = outcomes.get(arm_b.arm_id)
                if outcome_b is None:
                    continue
                b_set = set(arm_b.compound_ids)
                if not a_set < b_set:  # require strict subset
                    continue
                added = b_set - a_set
                bucket = _arm_differential_bucket(outcome_a, outcome_b)
                if bucket is None:
                    continue
                for new_c in sorted(added):
                    for backbone_c in sorted(a_set):
                        src, tgt = canonical_modulation_endpoints(
                            new_c, backbone_c,
                        )
                        # Modulation edges are trial-level (not arm-scoped);
                        # use None for the arm slot of the dedupe key.
                        edge_key = (
                            EdgeType.MODULATES_EFFICACY_OF.value,
                            src, tgt, None,
                        )
                        if edge_key in applied_edges:
                            continue
                        applied_edges.add(edge_key)
                        update = self._upsert_modulation_edge(
                            src=src,
                            tgt=tgt,
                            bucket=bucket,
                            evidence_type=evidence_type,
                            trial=trial,
                            classification=classification,
                            note=(
                                f"Arm differential: {arm_a.arm_id} "
                                f"(outcome={outcome_a.value}) vs "
                                f"{arm_b.arm_id} (outcome={outcome_b.value}); "
                                f"added={sorted(added)}"
                            ),
                        )
                        emitted.append(update)
        return emitted

    def _emit_single_arm_combo_modulations(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
        *,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Pairwise weak emission for combo arms with no subset comparator.

        When a trial reports a combo arm's outcome but no monotherapy (or
        smaller-subset) arm to compare against, we can't compute a
        differential — but the combo arm's outcome is still evidence at
        a much weaker level: each pair of constituents accumulates one
        weak update in the direction of the arm's standalone outcome.
        Triggered per-arm: only fires for combo arms (≥2 constituents)
        whose compound set has no strict subset in any other arm.
        """
        outcomes = _aggregate_arm_outcomes(trial, extraction)
        if not outcomes:
            return []
        all_compound_sets = [set(a.compound_ids) for a in trial.arms]

        emitted: list[AppliedEdgeUpdate] = []
        for arm in trial.arms:
            if len(arm.compound_ids) < 2:
                continue
            outcome = outcomes.get(arm.arm_id)
            if outcome is None:
                continue
            bucket = _single_arm_combo_bucket(outcome)
            if bucket is None:
                continue
            arm_set = set(arm.compound_ids)
            has_subset_comparator = any(
                other < arm_set for other in all_compound_sets
            )
            if has_subset_comparator:
                continue
            sorted_compounds = sorted(arm_set)
            for i, c1 in enumerate(sorted_compounds):
                for c2 in sorted_compounds[i + 1:]:
                    src, tgt = canonical_modulation_endpoints(c1, c2)
                    edge_key = (
                        EdgeType.MODULATES_EFFICACY_OF.value, src, tgt, None,
                    )
                    if edge_key in applied_edges:
                        continue
                    applied_edges.add(edge_key)
                    emitted.append(self._upsert_modulation_edge(
                        src=src,
                        tgt=tgt,
                        bucket=bucket,
                        evidence_type=evidence_type,
                        trial=trial,
                        classification=classification,
                        note=(
                            f"Single-arm combo: {arm.arm_id} "
                            f"(outcome={outcome.value}); no subset "
                            f"comparator in trial."
                        ),
                    ))
        return emitted

    def _resolve_compound_id(
        self,
        name: str,
    ) -> str | None:
        """Resolve a compound name to a graph InterventionNode id.

        Three paths in priority order:
          1. Direct id lookup — the LLM often emits the canonical slug.
          2. ``normalize_entity`` slugification — handles capitalized /
             punctuated forms like "Nivolumab" or "CMP-001".
          3. Separator-insensitive match — the LLM sometimes drops
             hyphens/underscores that the graph kept ("cmp001" vs
             "cmp_001"). Compare alnum-only-lowercased forms across
             all Intervention/Compound nodes.

        Returns None on no match; the caller logs to the unrouted log.
        """
        if not name:
            return None
        try:
            node = self.graph.get_node(name)
            if node.get("node_type") in ("InterventionNode", "CompoundNode"):
                return name
        except KeyError:
            pass
        try:
            slug = normalize_entity(name, "InterventionNode")
        except ValueError:
            slug = None
        if slug:
            try:
                node = self.graph.get_node(slug)
                if node.get("node_type") in ("InterventionNode", "CompoundNode"):
                    return slug
            except KeyError:
                pass
        # Separator-insensitive fallback.
        target_norm = _norm_name(name)
        if not target_norm:
            return None
        for node_id, data in self.graph._graph.nodes(data=True):  # noqa: SLF001
            if data.get("node_type") not in ("InterventionNode", "CompoundNode"):
                continue
            if _norm_name(node_id) == target_norm:
                return node_id
        return None

    def _find_primary_chain(
        self,
        trial: TrialSubgraph,
        primary_compound_id: str,
    ) -> CausalChain | None:
        """Find a chain in the trial whose compound_id matches the LLM's
        primary_compound_id. Returns the first matching chain (chains for
        the same compound share the upstream backbone, so target /
        mechanism / biology are equivalent across them).
        """
        for chain in trial.chains:
            if chain.compound_id == primary_compound_id:
                return chain
        # Try to match via arm membership — combo regimens encode the
        # compound list separately, and the chain's compound_id is the
        # regimen slug (e.g. "ipilimumab+nivolumab") not a single
        # constituent. The LLM emits constituent compound ids, so check
        # the arm's compound list.
        for chain in trial.chains:
            for arm in trial.arms:
                if (
                    chain.arm_id == arm.arm_id
                    and primary_compound_id in arm.compound_ids
                ):
                    return chain
        return None

    def _emit_llm_modulations(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
        *,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Emit v0.3.0 LLM-anchored MODULATES_EFFICACY_OF edges.

        For each ``ModulationEntry`` in the extraction:
          1. Resolve modulator + primary compound names to graph node ids.
          2. Find the primary's chain in this trial.
          3. Pull the chain node at the LLM-specified layer
             (``chain.target_id`` / ``mechanism_id`` / ``biology_id``).
          4. Emit a MODULATES_EFFICACY_OF edge from modulator → that
             node, with the layer + direction + confidence in the
             evidence context for downstream prediction.

        The LLM doesn't name internal node ids — those are populator
        outputs the LLM can't know at extraction time. The LLM names
        canonical things (compound names, layer names); the populator
        does the chain walk.

        Unroutable entries go to ``data/dev/unrouted_modulation_entries.jsonl``
        with a ``reason`` field. A high volume there means the prompt
        is producing modulator/primary names that aren't in the trial,
        which is a real debug signal.
        """
        if extraction is None or not extraction.modulation_entries:
            return []

        emitted: list[AppliedEdgeUpdate] = []
        for entry in extraction.modulation_entries:
            modulator_id = self._resolve_compound_id(entry.modulator_compound_id)
            if modulator_id is None:
                _log_unrouted_modulation(
                    trial.trial_id, entry, reason="modulator_not_in_graph",
                )
                continue

            primary_id = self._resolve_compound_id(entry.primary_compound_id)
            if primary_id is None:
                _log_unrouted_modulation(
                    trial.trial_id, entry, reason="primary_not_in_graph",
                )
                continue

            chain = self._find_primary_chain(trial, primary_id)
            if chain is None:
                _log_unrouted_modulation(
                    trial.trial_id, entry, reason="primary_chain_not_in_trial",
                )
                continue

            anchor_id: str | None
            if entry.affects_layer == "target":
                anchor_id = chain.target_id
            elif entry.affects_layer == "mechanism":
                anchor_id = chain.mechanism_id
            elif entry.affects_layer == "biology":
                anchor_id = chain.biology_id
            else:
                _log_unrouted_modulation(
                    trial.trial_id, entry,
                    reason=f"unknown_affects_layer:{entry.affects_layer}",
                )
                continue

            if not anchor_id or anchor_id == _UNKNOWN_PLACEHOLDER:
                _log_unrouted_modulation(
                    trial.trial_id, entry,
                    reason=f"primary_chain_layer_unresolved:{entry.affects_layer}",
                )
                continue

            bucket = modulation_bucket(entry.direction, entry.confidence)

            edge_key = (
                EdgeType.MODULATES_EFFICACY_OF.value,
                modulator_id, anchor_id, None,
            )
            if edge_key in applied_edges:
                continue
            applied_edges.add(edge_key)

            update = self._upsert_modulation_edge(
                src=modulator_id,
                tgt=anchor_id,
                bucket=bucket,
                evidence_type=evidence_type,
                trial=trial,
                classification=classification,
                note=(
                    f"LLM modulation: {entry.modulator_compound_id} "
                    f"{entry.direction} primary={entry.primary_compound_id} "
                    f"at {entry.affects_layer} layer "
                    f"(anchor={anchor_id}); conf={entry.confidence:.2f}"
                ),
                source_label="llm_modulation_entry",
                evidence_context_extras={
                    "primary_compound": primary_id,
                    "affects_layer": entry.affects_layer,
                    "modulation_direction": entry.direction,
                    "modulation_confidence": entry.confidence,
                    "hypothesis": entry.hypothesis,
                    "citation": entry.citation,
                },
            )
            emitted.append(update)
        return emitted

    def _upsert_modulation_edge(
        self,
        *,
        src: str,
        tgt: str,
        bucket: SupportBucket,
        evidence_type: EvidenceType,
        trial: TrialSubgraph,
        classification: FailureClassification,
        note: str,
        source_label: str = "arm_differential",
        evidence_context_extras: dict[str, Any] | None = None,
    ) -> AppliedEdgeUpdate:
        """Create the modulation edge with a neutral prior if absent, then
        apply one evidence record. Returns the AppliedEdgeUpdate.

        Every modulation evidence record carries ``context["indication"]``
        so a future v0.2.x refinement can apply Path-2-style conditioning
        on this edge type (sample beliefs only under the queried
        indication, downweight off-context records). v0.2.0 doesn't read
        the context yet — it's tagged forward-compatibly.

        ``source_label`` flags how the edge was first created
        ("arm_differential" for v0.2.0 attributor emissions,
        "llm_modulation_entry" for v0.3.0 LLM emissions). Edge metadata
        is only set at creation; subsequent evidence records carry their
        own per-trial provenance via ``context``.

        ``evidence_context_extras`` extends the per-record context dict —
        used by LLM modulations to record the specific affects_edge
        triple they were claiming about, alongside direction/confidence.
        """
        if not self.graph._graph.has_edge(  # noqa: SLF001
            src, tgt, key=EdgeType.MODULATES_EFFICACY_OF.value,
        ):
            self.graph.add_edge(GraphEdge(
                source_id=src,
                target_id=tgt,
                edge_type=EdgeType.MODULATES_EFFICACY_OF,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
                metadata={"source": source_label},
            ))
        pre_belief = self.graph.get_edge_belief(
            src, tgt, EdgeType.MODULATES_EFFICACY_OF,
        )
        indication_id = trial.chains[0].indication_id if trial.chains else None
        evidence_context: dict[str, Any] = {}
        if indication_id and indication_id != _UNKNOWN_PLACEHOLDER:
            evidence_context["indication"] = indication_id
        if evidence_context_extras:
            evidence_context.update(evidence_context_extras)
        evidence = EvidenceRecord(
            source_id=trial.trial_id,
            source_type=evidence_type,
            support=bucket.value,
            quality_score=min(classification.confidence, 1.0),
            timestamp=datetime.now(timezone.utc),
            notes=note,
            context=evidence_context,
        )
        post_belief = self.graph.update_edge_belief(
            src, tgt, EdgeType.MODULATES_EFFICACY_OF, evidence,
        )
        return AppliedEdgeUpdate(
            source_id=src,
            target_id=tgt,
            edge_type=EdgeType.MODULATES_EFFICACY_OF,
            evidence=evidence,
            pre_update_belief=pre_belief,
            post_update_belief=post_belief,
        )

    async def attribute_adverse_events(
        self,
        trial: TrialSubgraph,
        extraction: TrialExtraction,
        client: Any,  # anthropic.AsyncAnthropic—kept loose to avoid import cost in attributor
        meddra_cache: MeddraCache | None = None,
        classification: "FailureClassification | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Update causes_ae edges from a trial's structured adverse events.

        For each AE the extractor pulled from the trial:
          1. Normalize the term to a MedDRA preferred term (cached LLM call).
          2. Ensure an AdverseEventNode exists in the graph (create on miss).
          3. Choose a SupportBucket from the per-arm incidence rates.
          4. Update causes_ae from each treatment-arm compound to the AE.

        Skips arms that look like placebo/sham/vehicle so the comparator's
        background-rate mentions don't accumulate causes_ae evidence on
        the wrong compound.
        """
        cache = meddra_cache or MeddraCache()
        evidence_type = _PHASE_TO_EVIDENCE.get(trial.phase, EvidenceType.LITERATURE)
        treatment_compound_ids = self._treatment_compound_ids(trial)
        if not treatment_compound_ids:
            return []

        # Estimate per-arm patient count from total enrollment / arm
        # count. Trials rarely report exact per-arm n in CT.gov, but
        # the trial-level enrollment IS reliable. Used by
        # _ae_support_bucket to gate small-count AE signals (fixes.md
        # #9 — a 1.2% rate in an n=85 arm is ~1 patient, which the
        # rate-only path would still label moderate_support).
        treatment_n: int | None = None
        total_n = trial.metadata.get("enrollment") if trial.metadata else None
        if isinstance(total_n, int) and total_n > 0 and trial.arms:
            treatment_n = max(1, total_n // len(trial.arms))

        updates: list[AppliedEdgeUpdate] = []
        for ae in extraction.adverse_events:
            normalized = await normalize_ae_term(client, ae.term, cache)
            if normalized is None:
                # Meta / summary row (e.g. "Grade 3-5 adverse events") —
                # no single clinical concept to attribute. Skip rather
                # than collapse onto the meaningless "Unspecified" node.
                logger.debug(
                    "Skipping meta AE term %r for trial %s",
                    ae.term, trial.trial_id,
                )
                continue
            preferred_term = normalized["preferred_term"]
            soc = normalized.get("system_organ_class", "")
            ae_id = ae_node_id(preferred_term)
            self._ensure_ae_node(
                ae_id, preferred_term, soc, ae.grade, serious=ae.serious,
            )

            for compound_id in treatment_compound_ids:
                # When arm_incidences is populated (CT.gov-structured path),
                # compute per-compound tx/ctrl by partitioning arms on whether
                # this compound was active. Otherwise fall back to the flat
                # tx/ctrl pair from LLM-extracted narrative-only trials.
                tx_pct, ctrl_pct, tx_n = (None, None, None)
                if ae.arm_incidences:
                    compound_name = self._compound_display_name(compound_id)
                    tx_pct, ctrl_pct, tx_n = _per_compound_rates_from_arms(
                        ae.arm_incidences, compound_name,
                    )
                if tx_pct is None:
                    tx_pct = ae.incidence_treatment_pct
                    ctrl_pct = ae.incidence_control_pct
                    tx_n = treatment_n

                bucket = _ae_support_bucket(
                    tx_pct, ctrl_pct, treatment_n=tx_n,
                    hazard_ratio=ae.hazard_ratio,
                    hr_ci_low=ae.hr_ci_low, hr_ci_high=ae.hr_ci_high,
                )
                note = _format_ae_note(
                    ae, preferred_term, tx_pct=tx_pct, ctrl_pct=ctrl_pct,
                )

                self._ensure_causes_ae_edge(compound_id, ae_id)
                evidence = EvidenceRecord(
                    source_id=trial.trial_id,
                    source_type=evidence_type,
                    support=bucket.value,
                    quality_score=1.0,  # incidence-rate evidence is structured, not LLM-judgment
                    timestamp=datetime.now(timezone.utc),
                    notes=note,
                    context={
                        "ae_term_raw": ae.term, "ae_grade": ae.grade,
                        # Round-30 DLT-gate signal: did this AE come from a
                        # trial whose failure was dose-limiting toxicity?
                        # The safety-penalty gate weights failure-causing
                        # toxicity over mere occurrence (see path_query).
                        "failure_causing_tox": bool(
                            ae.failure_causing
                            or (
                                classification is not None
                                and classification.primary_failure_mode
                                == FailureMode.DOSE_LIMITING_TOXICITY
                            )
                        ),
                    },
                )
                pre = self.graph.get_edge_belief(
                    compound_id, ae_id, EdgeType.CAUSES_AE
                )
                post = self.graph.update_edge_belief(
                    compound_id, ae_id, EdgeType.CAUSES_AE, evidence
                )
                updates.append(AppliedEdgeUpdate(
                    source_id=compound_id,
                    target_id=ae_id,
                    edge_type=EdgeType.CAUSES_AE,
                    evidence=evidence,
                    pre_update_belief=pre,
                    post_update_belief=post,
                ))

        # After all causes_ae updates land, refresh target_associated_ae for
        # each touched (compound, AE) pair. Propagation is idempotent —
        # running it once per touched pair is enough even when the same
        # AE was attributed to multiple compounds in this trial.
        touched_pairs = {(u.source_id, u.target_id) for u in updates}
        for compound_id, ae_id in touched_pairs:
            propagate_to_target_associated_ae(self.graph, compound_id, ae_id)

        return updates

    def _compound_display_name(self, compound_id: str) -> str:
        """Return the CompoundNode's display name, or '' if missing.

        Used to match against CT.gov eventGroup titles (e.g. compound
        name 'Nivolumab' against descriptor 'Nivolumab + Ipilimumab')
        in ``_per_compound_rates_from_arms``.
        """
        try:
            node = self.graph.get_node(compound_id)
        except KeyError:
            return ""
        return str(node.get("name") or "")

    def _treatment_compound_ids(self, trial: TrialSubgraph) -> list[str]:
        """Compound ids on any non-placebo arm of the trial.

        Returns the *constituent* ids (not the synthesized regimen id for
        combos) so causes_ae attribution accumulates on the same Compound
        node a different trial would attribute to.
        """
        seen: set[str] = set()
        out: list[str] = []
        for arm in trial.arms:
            for cid in arm.compound_ids:
                if cid in seen:
                    continue
                try:
                    node = self.graph.get_node(cid)
                except KeyError:
                    continue
                name = (node.get("name") or "").lower()
                if any(token in name for token in ("placebo", "sham", "vehicle")):
                    continue
                seen.add(cid)
                out.append(cid)
        return out

    def _ensure_ae_node(
        self, ae_id: str, preferred_term: str, soc: str, grade: str,
        *, serious: bool = False,
    ) -> None:
        # Round-28: look up the MedDRA hierarchy parents (HLT / HLGT / SOC
        # slug + canonical SOC name) so target_associated_ae propagation
        # can aggregate at the SOC tier downstream. ``soc`` here is the
        # free-text MedDRA SOC string emitted by the normalizer; the
        # hierarchy uses it as a fallback when the PT isn't in the
        # curated PT→SOC table.
        hierarchy = _meddra_hierarchy_singleton()
        parents = hierarchy.parents_for_pt(
            ae_id, fallback_soc_name=soc,
        )
        try:
            existing = self.graph.get_node(ae_id)
        except KeyError:
            self.graph.add_node(AdverseEventNode(
                id=ae_id,
                name=preferred_term,
                system_organ_class=soc,
                severity_range=grade or "",
                serious=bool(serious),
                hlt_id=parents["hlt_id"],
                hlgt_id=parents["hlgt_id"],
                soc_id=parents["soc_id"],
                soc_name=parents["soc_name"] or soc,
            ))
            return
        # Node exists—extend severity_range if this AE reported a new grade
        # we haven't seen for this term before. SOC is locked in on first
        # write; the normalizer should be deterministic for the same input.
        if grade and grade not in (existing.get("severity_range") or ""):
            existing_range = existing.get("severity_range") or ""
            merged = f"{existing_range},{grade}".strip(",")
            self.graph._graph.nodes[ae_id]["severity_range"] = merged
        # Round-29: OR-merge `serious` across trials reporting the same AE.
        # Any trial flagging serious=True locks the node's serious to True.
        if serious and not existing.get("serious"):
            self.graph._graph.nodes[ae_id]["serious"] = True
        # Backfill round-28 hierarchy fields onto pre-existing nodes
        # missing them (round-26 snapshots that loaded without these
        # fields get them on first re-attribution). Only writes when
        # the existing value is empty so previously-resolved hierarchy
        # data is preserved.
        if parents["soc_id"] and not existing.get("soc_id"):
            self.graph._graph.nodes[ae_id]["soc_id"] = parents["soc_id"]
        if parents["soc_name"] and not existing.get("soc_name"):
            self.graph._graph.nodes[ae_id]["soc_name"] = parents["soc_name"]
        if parents["hlt_id"] and not existing.get("hlt_id"):
            self.graph._graph.nodes[ae_id]["hlt_id"] = parents["hlt_id"]
        if parents["hlgt_id"] and not existing.get("hlgt_id"):
            self.graph._graph.nodes[ae_id]["hlgt_id"] = parents["hlgt_id"]

    def _ensure_causes_ae_edge(self, compound_id: str, ae_id: str) -> None:
        try:
            self.graph.get_edge_belief(
                compound_id, ae_id, EdgeType.CAUSES_AE
            )
        except KeyError:
            self.graph.add_edge(GraphEdge(
                source_id=compound_id,
                target_id=ae_id,
                edge_type=EdgeType.CAUSES_AE,
                belief=EdgeBeliefState(),
            ))

    def apply_updates(
        self, updates: list[AppliedEdgeUpdate]
    ) -> dict[str, Any]:
        """Summarize applied updates (already applied during attribute())."""
        if not updates:
            return {"edges_updated": 0, "largest_changes": []}

        sorted_updates = sorted(
            updates, key=lambda u: abs(u.probability_change), reverse=True
        )

        largest = [
            {
                "edge": f"{u.source_id} -> {u.target_id} ({u.edge_type.value})",
                "pre": round(u.pre_update_belief.expected_probability, 4),
                "post": round(u.post_update_belief.expected_probability, 4),
                "change": round(u.probability_change, 4),
            }
            for u in sorted_updates[:5]
        ]

        return {
            "edges_updated": len(updates),
            "largest_changes": largest,
        }


# ── CLI ──────────────────────────────────────────────────────────────────


def _load_classifications(annotations_dir: Path) -> list[tuple[dict, dict]]:
    """Load paired extraction + classification JSONs."""
    pairs = []
    for clf_path in sorted(annotations_dir.glob("*_classification.json")):
        nct_id = clf_path.stem.replace("_classification", "")
        ext_path = annotations_dir / f"{nct_id}_extraction.json"
        if ext_path.exists():
            clf_data = json.loads(clf_path.read_text())
            ext_data = json.loads(ext_path.read_text())
            pairs.append((ext_data, clf_data))
    return pairs


async def _main(
    annotations_dir: str,
    graph_path: str,
    output_path: str,
    *,
    exclude_from_attribution: list[str] | None = None,
) -> None:
    """Apply trial attribution to a populated graph.

    ``exclude_from_attribution`` (round-26): NCT ids to skip in this
    attribution pass. Their subgraphs remain in the graph (so chain
    prediction still works on them), but their evidence is NOT folded
    into edge beliefs — the listed trials become a TRUE holdout for
    evaluation. Implemented by pre-populating
    ``graph.applied_attribution_trial_ids`` with the excluded ids
    before the iteration loop, so the existing idempotency guard
    skips them naturally. This is the fix for the round-24
    methodology bug where ``--add-trials`` re-ran attribution on
    holdouts and contaminated the eval.
    """
    import anthropic
    from rich.console import Console

    from src.annotation.meddra import MeddraCache

    console = Console()

    # Load graph
    graph = GraphStore()
    graph_file = Path(graph_path)
    if graph_file.exists():
        console.print(f"[bold]Loading graph from {graph_path}...[/bold]")
        graph.import_snapshot(graph_path)
        stats = graph.stats()
        console.print(f"  Loaded: {stats['node_count']} nodes, {stats['edge_count']} edges")
    else:
        console.print(f"[yellow]Graph file not found: {graph_path}[/yellow]")
        return

    # Round-26: pre-mark holdout NCTs as already-attributed so the
    # idempotency guard below skips them. The set is mutated in place
    # on the graph so when export_snapshot serializes the final
    # snapshot, applied_attribution_trial_ids accurately reflects
    # "training NCTs only" — exactly the discrimination round-24's
    # clean-holdout audit needed.
    excluded_set = set(exclude_from_attribution or [])
    if excluded_set:
        console.print(
            f"[yellow]Excluding {len(excluded_set)} NCT(s) from attribution: "
            f"{sorted(excluded_set)}[/yellow]"
        )
        graph.applied_attribution_trial_ids.update(excluded_set)

    attributor = Attributor(graph)
    pairs = _load_classifications(Path(annotations_dir))
    console.print(f"\n[bold]Found {len(pairs)} annotated trials[/bold]")

    # Shared resources for AE attribution: one Anthropic client + one
    # MeddraCache reused across the whole batch so repeat AE terms hit
    # the cache rather than the LLM.
    client = anthropic.AsyncAnthropic(timeout=60.0)
    meddra_cache = MeddraCache()

    total_updates: list[AppliedEdgeUpdate] = []

    for ext_data, clf_data in pairs:
        trial_id = clf_data.get("nct_id", ext_data.get("nct_id", "unknown"))

        # Round-19: idempotency guard. Trials whose attribution has
        # already been folded into this snapshot must not have their
        # Beta-Binomial updates re-applied — that would double-count
        # evidence on every edge the trial touched. The set is added
        # to AFTER the trial's updates land successfully (transactional),
        # so a partial / failed attribution leaves the trial retryable.
        if trial_id in graph.applied_attribution_trial_ids:
            console.print(
                f"  [dim]Skipped {trial_id}:[/dim] already attributed in this snapshot"
            )
            continue

        # The trial subgraph (with arms + chains) must already exist in the
        # graph sidecar—produced by populate.build_trial_subgraphs and
        # extended by add_subgroup_chains during the extraction pipeline.
        try:
            trial = graph.get_trial_subgraph_by_id(trial_id)
        except KeyError:
            console.print(
                f"  [yellow]Skipped {trial_id}:[/yellow] no trial_subgraph in sidecar"
            )
            continue

        # Build classification
        modes = clf_data.get("failure_modes", [])
        primary_mode = FailureMode.INSUFFICIENT_INFORMATION
        if modes:
            sorted_modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
            try:
                primary_mode = FailureMode(sorted_modes[0]["mode"])
            except (ValueError, KeyError):
                pass

        _op_fail = clf_data.get("operational_failure")
        if not isinstance(_op_fail, bool):
            _op_fail = None
        classification = FailureClassification(
            trial_id=trial_id,
            primary_failure_mode=primary_mode,
            confidence=clf_data.get("confidence_overall", 0.5),
            reasoning=clf_data.get("reasoning", ""),
            operational_failure=_op_fail,
        )
        classification._raw = clf_data  # type: ignore[attr-defined]

        # Parse extraction once — passed to attribute() so modulation
        # emission can read per-arm outcomes, then reused for AE
        # attribution below.
        from src.annotation.extractor import _parse_extraction_response
        try:
            extraction = _parse_extraction_response(ext_data, trial_id)
        except Exception as exc:  # noqa: BLE001—pydantic ValidationError + others
            logger.warning(
                "Extraction JSON invalid for %s (%s); modulation emission "
                "and AE attribution will be skipped",
                trial_id, exc,
            )
            extraction = None

        # PubMed safety enrichment: attribution re-reads the raw extraction JSON
        # (bypassing extractor.extract's hook), so apply the cache HERE — this is
        # the point that actually lands the causes_ae edges for terminated trials
        # whose safety signal lived only in a linked paper (e.g. torcetrapib).
        if extraction is not None:
            from src.annotation.pubmed_safety import maybe_enrich_by_nct
            extraction = maybe_enrich_by_nct(extraction, trial_id)

        updates = attributor.attribute(classification, trial, extraction)
        total_updates.extend(updates)

        if extraction is not None and extraction.adverse_events:
            ae_updates = await attributor.attribute_adverse_events(
                trial, extraction, client=client, meddra_cache=meddra_cache,
                classification=classification,
            )
            total_updates.extend(ae_updates)

        # Round-19: record successful attribution. Placed after both
        # efficacy + AE phases land so a mid-trial raise leaves the
        # set unchanged and the trial retryable on the next run.
        graph.applied_attribution_trial_ids.add(trial_id)

        if updates:
            console.print(f"  {trial_id}: {len(updates)} efficacy edge updates")

    # Summary
    summary = attributor.apply_updates(total_updates)
    console.print(f"\n[bold green]Processed {len(pairs)} trials. Updated {summary['edges_updated']} edges.[/bold green]")

    if summary["largest_changes"]:
        console.print("[bold]Largest changes:[/bold]")
        for change in summary["largest_changes"]:
            direction = "+" if change["change"] > 0 else ""
            console.print(
                f"  {change['edge']}: {change['pre']:.4f} → {change['post']:.4f} "
                f"({direction}{change['change']:.4f})"
            )

    # Save updated graph
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    graph.export_snapshot(output_path)
    console.print(f"\n[bold]Saved annotated graph to {output_path}[/bold]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply failure attributions to knowledge graph")
    parser.add_argument("--input", default="data/annotations/", help="Annotations directory")
    parser.add_argument("--graph", default="data/exports/oncology_initial.json", help="Input graph snapshot")
    parser.add_argument("--output", default="data/exports/oncology_annotated.json", help="Output graph snapshot")
    args = parser.parse_args()

    asyncio.run(_main(args.input, args.graph, args.output))
