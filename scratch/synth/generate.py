"""Synthetic generator for the eroom EM/VBEM harness.

Implements the §2 generative model from ``eroom-em-derivation.md`` with *known*
ground-truth parameters, so recovery (``recover.py``) can be scored against truth.

The model factors a trial's success into three independent events:

    y_t = (∏_{a∈M_t} f_a) · (∏_{k∈G_t} (1 - g_k)) · e

  * must-holds   f_a ~ Bernoulli(r_a)   — the efficacy spine, the collapsed
                                          biology×endpoint×indication "triangle"
                                          factor v, and the indication→population
                                          edge w.  Each holds with reliability r_a.
  * safety gates g_k ~ Bernoulli(q_k)    — AE→intervention / AE→target gates.
                                          q_k is a *failure* prob (gate fires → halt).
  * leak         e   ~ Bernoulli(ℓ)      — one global scalar; ℓ = P(no unmodeled
                                          factor kills an otherwise-perfect trial).

The experimental lever is ``reuse_rate`` = expected trials per edge.  We size the
edge pools so that uniform sampling of each trial's chain yields ~reuse_rate
incidences per edge (a Poisson(reuse_rate) reuse distribution — the same skewed,
singleton-heavy shape the real corpus shows: 1.24 trials/edge, 71% singletons).

Ground-truth failure reason (when y=0) is emitted by competing-risks precedence
  safety  >  business/leak  >  efficacy
which respects the EM's missing-at-random-within-branch assumption (gate firing
and leak are independent of which efficacy edges are present).  ``reason_noise``
then corrupts that reason to model imperfect taxonomy extraction.

Net-new, self-contained: imports nothing from ``src/`` and reads no real corpus.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

# ---- reason vocabulary -------------------------------------------------------
SUCCESS = "success"
R_SAFETY = "safety"        # a safety gate fired (DLT / halt) → censor efficacy
R_EFFICACY = "efficacy"    # ran to readout, a must-hold broke → blame within Φ
R_BUSINESS = "business"    # leak fired (operational / strategic) → censor all
R_UNKNOWN = "unknown"      # reason unavailable → fall back to full responsibility

FAIL_REASONS = (R_SAFETY, R_EFFICACY, R_BUSINESS)


@dataclass
class GroundTruth:
    """Planted parameters + edge pools.  Recovery is scored against this."""

    # must-hold reliabilities, concatenated [spine | triangle | population]
    r_must: np.ndarray
    must_class: np.ndarray         # str tag per must-hold: 'spine'|'triangle'|'pop'
    n_spine: int
    n_tri: int
    n_pop: int
    # safety-gate failure probabilities
    q_gate: np.ndarray
    leak: float
    cfg: dict = field(default_factory=dict)

    @property
    def n_must(self) -> int:
        return int(self.r_must.shape[0])

    @property
    def n_gate(self) -> int:
        return int(self.q_gate.shape[0])


@dataclass
class Trial:
    must_idx: np.ndarray       # indices into r_must (the trial's M_t)
    gate_idx: np.ndarray       # indices into q_gate (the trial's G_t)
    y: int                     # observed binary success
    reason: str                # ground-truth reason (uncorrupted)
    reason_obs: str            # reason after reason_noise corruption (what EM sees)


@dataclass
class Corpus:
    gt: GroundTruth
    trials: list[Trial]

    # ---- convenience views used by recover.py ----
    def must_incidence(self) -> np.ndarray:
        """#trials touching each must-hold edge (reuse count)."""
        c = np.zeros(self.gt.n_must, dtype=int)
        for t in self.trials:
            c[t.must_idx] += 1
        return c

    def gate_incidence(self) -> np.ndarray:
        c = np.zeros(self.gt.n_gate, dtype=int)
        for t in self.trials:
            c[t.gate_idx] += 1
        return c


# ---- default knobs -----------------------------------------------------------
DEFAULTS = dict(
    n_trials=500,
    reuse_rate=1.24,           # corpus efficacy reuse (the headline variable)
    reason_noise=0.0,          # P(reason corrupted); set per recovery mode
    frac_nonmech_failure=0.10, # 1 - leak: per-trial operational/strategic kill rate
    # per-trial chain shape
    n_spine_per_trial=2,       # efficacy-spine must-holds per trial
    n_gate_per_trial=2,        # AE safety gates per trial
    # explicit pool-size overrides (used only when reuse_rate is None)
    n_targets=None,            # spine-pool size override (a "target" anchors a spine edge)
    n_ae_gates=None,           # gate-pool size override
    # planted-parameter priors (means chosen per §7: spine mildly optimistic,
    # measurement/triangle more skeptical, gates low base rate with a heavy tail)
    corruption="to_unknown",   # 'to_unknown' (default) | 'adversarial'
)

# Beta shape params for the *planted* parameter distributions (fixed, not tuned).
_PLANT = dict(
    spine=(6.0, 2.0),   # mean 0.75
    triangle=(4.0, 2.0),  # mean 0.667 — detection/endpoint is the skeptical factor
    pop=(6.0, 2.0),       # mean 0.75
    gate=(1.5, 12.0),     # mean ~0.111, heavy right tail (known-liability families)
)


def _pool_size(n_trials: int, per_trial: int, reuse_rate: float, override) -> int:
    if reuse_rate is not None:
        return max(per_trial, int(round(n_trials * per_trial / reuse_rate)))
    if override is not None:
        return max(per_trial, int(override))
    raise ValueError("need reuse_rate or an explicit pool-size override")


def make_ground_truth(seed: int, **kw) -> GroundTruth:
    """Plant the reliabilities/gates and size the pools to hit the reuse target."""
    cfg = {**DEFAULTS, **kw}
    rng = np.random.default_rng(seed)

    n_trials = cfg["n_trials"]
    reuse = cfg["reuse_rate"]
    Ls = cfg["n_spine_per_trial"]
    Lg = cfg["n_gate_per_trial"]

    n_spine = _pool_size(n_trials, Ls, reuse, cfg["n_targets"])
    n_tri = _pool_size(n_trials, 1, reuse, None if reuse is not None
                       else max(1, (cfg["n_targets"] or 1)))
    n_pop = _pool_size(n_trials, 1, reuse, None if reuse is not None
                       else max(1, (cfg["n_targets"] or 1)))
    n_gate = _pool_size(n_trials, Lg, reuse, cfg["n_ae_gates"])

    a, b = _PLANT["spine"]
    r_spine = rng.beta(a, b, size=n_spine)
    a, b = _PLANT["triangle"]
    r_tri = rng.beta(a, b, size=n_tri)
    a, b = _PLANT["pop"]
    r_pop = rng.beta(a, b, size=n_pop)
    a, b = _PLANT["gate"]
    q_gate = rng.beta(a, b, size=n_gate)

    r_must = np.concatenate([r_spine, r_tri, r_pop])
    must_class = np.array(
        ["spine"] * n_spine + ["triangle"] * n_tri + ["pop"] * n_pop
    )
    leak = 1.0 - cfg["frac_nonmech_failure"]

    cfg_out = {k: cfg[k] for k in cfg}
    cfg_out.update(dict(n_spine=n_spine, n_tri=n_tri, n_pop=n_pop,
                        n_gate=n_gate, leak=leak, seed=seed))
    return GroundTruth(r_must=r_must, must_class=must_class, n_spine=n_spine,
                       n_tri=n_tri, n_pop=n_pop, q_gate=q_gate, leak=leak,
                       cfg=cfg_out)


def _choice(rng, n, k):
    """k distinct indices from range(n) (or all of them if n<=k)."""
    if n <= k:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def sample_trials(gt: GroundTruth, n: int, seed: int,
                  reason_noise: float = 0.0,
                  corruption: str = "to_unknown") -> list[Trial]:
    """Draw n trials from the planted params; compute y, reason, corrupted reason."""
    rng = np.random.default_rng(seed)
    Ls = gt.cfg["n_spine_per_trial"]
    Lg = gt.cfg["n_gate_per_trial"]
    off_tri = gt.n_spine
    off_pop = gt.n_spine + gt.n_tri

    trials: list[Trial] = []
    for _ in range(n):
        spine = _choice(rng, gt.n_spine, Ls)
        tri = off_tri + _choice(rng, gt.n_tri, 1)
        pop = off_pop + _choice(rng, gt.n_pop, 1)
        must_idx = np.concatenate([spine, tri, pop])
        gate_idx = _choice(rng, gt.n_gate, Lg)

        # latents
        f = rng.random(must_idx.shape[0]) < gt.r_must[must_idx]      # holds?
        g = rng.random(gate_idx.shape[0]) < gt.q_gate[gate_idx]      # gate fires?
        e = rng.random() < gt.leak                                   # no leak?

        must_all_hold = bool(np.all(f))
        any_gate_fired = bool(np.any(g))
        y = int(must_all_hold and (not any_gate_fired) and e)

        if y == 1:
            reason = SUCCESS
        elif any_gate_fired:           # safety preempts (DLT halts before readout)
            reason = R_SAFETY
        elif not e:                    # operational / strategic kill
            reason = R_BUSINESS
        else:                          # ran to readout, a must-hold broke
            reason = R_EFFICACY

        reason_obs = _corrupt(reason, reason_noise, corruption, rng)
        trials.append(Trial(must_idx=must_idx, gate_idx=gate_idx, y=y,
                            reason=reason, reason_obs=reason_obs))
    return trials


def _corrupt(reason: str, noise: float, mode: str, rng) -> str:
    """Corrupt a *failure* reason with prob `noise`. Successes are never corrupted
    (the success/fail bit is highly reliable; only the taxonomy reason is noisy)."""
    if reason == SUCCESS or noise <= 0.0:
        return reason
    if rng.random() >= noise:
        return reason
    if mode == "to_unknown":
        # extractor failed to assign a reason → 'unknown' (falls back to full ρ).
        # At noise=1 every failure is 'unknown' ⇒ mode (c) ≡ mode (a) by construction.
        return R_UNKNOWN
    if mode == "adversarial":
        # extractor confidently asserts a WRONG real reason (the harsher model).
        others = [r for r in FAIL_REASONS if r != reason]
        return others[int(rng.integers(len(others)))]
    raise ValueError(f"unknown corruption mode {mode!r}")


def sample_corpus(seed: int, n_trials: int | None = None, **kw) -> Corpus:
    """Convenience: plant ground truth and draw a single training corpus."""
    gt = make_ground_truth(seed, **({} if n_trials is None
                                     else {"n_trials": n_trials}), **kw)
    nt = gt.cfg["n_trials"]
    trials = sample_trials(gt, nt, seed=seed + 1,
                           reason_noise=gt.cfg["reason_noise"],
                           corruption=gt.cfg["corruption"])
    return Corpus(gt=gt, trials=trials)


def corpus_stats(corpus: Corpus) -> dict:
    gt = corpus.gt
    ys = np.array([t.y for t in corpus.trials])
    reasons = [t.reason for t in corpus.trials]
    inc_m = corpus.must_incidence()
    inc_g = corpus.gate_incidence()
    obs_m = inc_m[inc_m > 0]
    from collections import Counter
    rc = Counter(reasons)
    return dict(
        n_trials=len(corpus.trials),
        success_rate=float(ys.mean()),
        n_must=gt.n_must, n_gate=gt.n_gate,
        must_reuse_observed=float(obs_m.mean()) if obs_m.size else 0.0,
        must_reuse_all=float(inc_m.mean()),
        frac_must_singleton=float(np.mean(inc_m == 1)),
        frac_must_unobserved=float(np.mean(inc_m == 0)),
        gate_reuse_all=float(inc_g.mean()),
        planted_mean_r=float(gt.r_must.mean()),
        planted_mean_q=float(gt.q_gate.mean()),
        leak=gt.leak,
        reason_counts={k: int(v) for k, v in rc.items()},
    )


def main():
    ap = argparse.ArgumentParser(description="Synthetic eroom corpus generator (§2).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=DEFAULTS["n_trials"])
    ap.add_argument("--reuse-rate", type=float, default=DEFAULTS["reuse_rate"])
    ap.add_argument("--reason-noise", type=float, default=DEFAULTS["reason_noise"])
    ap.add_argument("--frac-nonmech-failure", type=float,
                    default=DEFAULTS["frac_nonmech_failure"])
    ap.add_argument("--corruption", choices=["to_unknown", "adversarial"],
                    default=DEFAULTS["corruption"])
    ap.add_argument("--dump", type=str, default=None,
                    help="write a JSON stats summary to this path")
    args = ap.parse_args()

    corpus = sample_corpus(
        seed=args.seed, n_trials=args.n_trials, reuse_rate=args.reuse_rate,
        reason_noise=args.reason_noise,
        frac_nonmech_failure=args.frac_nonmech_failure,
        corruption=args.corruption,
    )
    stats = corpus_stats(corpus)
    print(json.dumps(stats, indent=2))
    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
