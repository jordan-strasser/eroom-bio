"""Delta-adjust tuning of the Beta-Binomial edge-update parameters.

The owner's focus: tune "the calculations that go into the edge-weight updates"
— the per-trial outcome-conditioning n_eff and p_obs that drive
``apply_virtual_evidence`` (α += n_eff·p_obs ; β += n_eff·(1−p_obs)).

KEY TRICK — no re-attribution, no full replay. Each attributor-written record
persists the EXACT values it applied in ``context``:
``{n_eff_applied, p_obs_applied, outcome, ...}`` and its trial NCT in
``source_id``. Because the conjugate update is purely ADDITIVE, we can perturb a
stored belief by swapping just those contributions:

    new_α = stored_α + Σ_trial-records [ n_eff'·p_obs'  −  n_eff_applied·p_obs_applied ]

This is **exact at the default** (Δ=0 ⇒ stored returned) and exact for the
perturbation; the only approximation is the explain-away feedback loop (a record's
p_obs feeds later records' u_i during true attribution), which is second-order and
confirmed by re-attributing the winning config. Curated/DB records and seed priors
(e.g. the LINCS Beta(5,1)) carry no ``n_eff_applied`` → they stay baked into the
stored α,β untouched (which is why this beats naive full-replay, which mis-handled
those). Holdout = drop a trial's records (subtract their contribution) — exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from src.graph.models import EvidenceRecord
from src.inference.beliefs import EvidenceType

# Coarse n_eff groups so the tuning grid is small + principled (we scale a
# GROUP's trial-evidence weight, not 18 tiers independently).
_CLINICAL = {
    EvidenceType.CLINICAL_PHASE3,
    EvidenceType.CLINICAL_PHASE2,
    EvidenceType.CLINICAL_PHASE1,
}
_CLINICAL_VALUES = {et.value for et in _CLINICAL}


def neff_group(et: EvidenceType | str) -> str:
    """The scalable group a record's evidence type belongs to. Only trial
    (clinical) evidence is delta-tunable here — curated/DB priors carry no
    applied-context and stay fixed (the round-28 binding tiers are facts).
    Accepts an EvidenceType or its string value (raw-snapshot form)."""
    val = et.value if isinstance(et, EvidenceType) else str(et)
    return "clinical" if val in _CLINICAL_VALUES else "other"


class Contrib(NamedTuple):
    """One trial record's pre-extracted contribution to an edge belief — the
    fast path for the tuning harness (avoids re-parsing 53k records per config)."""
    n_eff: float       # n_eff_applied
    p_obs: float       # p_obs_applied
    group: str         # neff_group of the source type
    outcome: str       # '' if none
    nct: str           # held-out key


def _rec_field(ev, name):
    return ev.get(name) if isinstance(ev, dict) else getattr(ev, name, None)


def extract_contribs(records) -> list[Contrib]:
    """Pull the delta-tunable contributions (records with an applied context)
    out of a record list. Accepts EvidenceRecord objects OR raw belief-dict
    records (the snapshot stores them as dicts)."""
    out: list[Contrib] = []
    for ev in records or []:
        ctx = _rec_field(ev, "context") or {}
        if "n_eff_applied" not in ctx:
            continue
        sid = (_rec_field(ev, "source_id") or "")
        nct = sid if sid.upper().startswith("NCT") else (ctx.get("nct") or "")
        out.append(Contrib(
            n_eff=float(ctx["n_eff_applied"]),
            p_obs=float(ctx["p_obs_applied"]),
            group=neff_group(_rec_field(ev, "source_type")),
            outcome=ctx.get("outcome") or "",
            nct=nct,
        ))
    return out


@dataclass
class BeliefTuneConfig:
    """A candidate edge-update parameterization.

    neff_scale: group → multiplier on the trial-evidence n_eff (default 1.0).
    outcome_p_obs: outcome label ('success'/'failure'/'partial_*') → p_obs to use
        instead of what was applied (default: keep applied)."""
    neff_scale: dict[str, float] = field(default_factory=dict)
    outcome_p_obs: dict[str, float] = field(default_factory=dict)

    def ratio(self, group: str) -> float:
        return self.neff_scale.get(group, 1.0)


DEFAULT_CONFIG = BeliefTuneConfig()


def retune_from_contribs(
    stored_alpha: float,
    stored_beta: float,
    contribs: list[Contrib],
    cfg: BeliefTuneConfig = DEFAULT_CONFIG,
    drop_ncts: frozenset[str] | set[str] = frozenset(),
) -> tuple[float, float]:
    """Delta-adjust a stored belief from pre-extracted contributions. The fast
    path. Exact at the default config with no drops (returns stored α, β)."""
    a, b = float(stored_alpha), float(stored_beta)
    for c in contribs:
        if drop_ncts and c.nct in drop_ncts:
            a -= c.n_eff * c.p_obs
            b -= c.n_eff * (1.0 - c.p_obs)
            continue
        n_new = c.n_eff * cfg.ratio(c.group)
        p_new = cfg.outcome_p_obs.get(c.outcome, c.p_obs) if c.outcome else c.p_obs
        a += n_new * p_new - c.n_eff * c.p_obs
        b += n_new * (1.0 - p_new) - c.n_eff * (1.0 - c.p_obs)
    return max(a, 1e-6), max(b, 1e-6)


def retune_alpha_beta(
    stored_alpha: float,
    stored_beta: float,
    records: list[EvidenceRecord],
    cfg: BeliefTuneConfig = DEFAULT_CONFIG,
    drop_ncts: frozenset[str] | set[str] = frozenset(),
) -> tuple[float, float]:
    """Return the (α, β) the edge WOULD have under ``cfg`` with ``drop_ncts`` held
    out, by delta-adjusting only the attributor-written trial records. Exact at
    the default config with no drops. Convenience wrapper over
    ``retune_from_contribs`` for callers holding raw records."""
    return retune_from_contribs(
        stored_alpha, stored_beta, extract_contribs(records), cfg, drop_ncts
    )
