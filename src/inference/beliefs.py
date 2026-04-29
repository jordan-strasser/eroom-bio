"""Beta distribution belief model with update mechanics."""

from __future__ import annotations

import math

from scipy import stats as sp_stats


class BetaBelief:
    """A Beta(alpha, beta) belief over a probability parameter."""

    __slots__ = ("alpha", "beta")

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"alpha and beta must be positive, got ({alpha}, {beta})")
        self.alpha = alpha
        self.beta = beta

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def evidence_strength(self) -> float:
        """Total pseudo-observations beyond the uniform prior."""
        return self.alpha + self.beta - 2.0

    @property
    def conflict_score(self) -> float:
        """High when lots of evidence but mean near 0.5."""
        strength = self.evidence_strength
        if strength <= 0:
            return 0.0
        certainty = abs(self.mean - 0.5) * 2  # 0 at p=0.5, 1 at p=0 or p=1
        return strength * (1 - certainty)

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        tail = (1 - ci) / 2
        lower = sp_stats.beta.ppf(tail, self.alpha, self.beta)
        upper = sp_stats.beta.ppf(1 - tail, self.alpha, self.beta)
        return (float(lower), float(upper))

    # ── Methods ─────────────────────────────────────────────────────────

    def update(
        self,
        direction: str,
        magnitude: float,
        weight: float = 1.0,
    ) -> BetaBelief:
        """Return a new BetaBelief after incorporating evidence.

        direction: "supporting", "contradicting", or "ambiguous"
        magnitude: raw evidence magnitude (pre-weight)
        weight: multiplier from evidence type/quality
        """
        delta = weight * magnitude
        if direction == "supporting":
            return BetaBelief(self.alpha + delta, self.beta)
        elif direction == "contradicting":
            return BetaBelief(self.alpha, self.beta + delta)
        else:  # ambiguous
            return BetaBelief(
                self.alpha + delta * 0.3,
                self.beta + delta * 0.3,
            )

    def pdf(self, x: float) -> float:
        """Evaluate the probability density at x."""
        return float(sp_stats.beta.pdf(x, self.alpha, self.beta))

    def kl_divergence(self, other: BetaBelief) -> float:
        """KL(self || other) — how much self diverges from other."""
        a1, b1 = self.alpha, self.beta
        a2, b2 = other.alpha, other.beta
        # Standard Beta KL: ln B(a2,b2)/B(a1,b1) + (a1-a2)ψ(a1) + (b1-b2)ψ(b1) + (a2-a1+b2-b1)ψ(a1+b1)
        log_beta_ratio = (
            (math.lgamma(a2) + math.lgamma(b2) - math.lgamma(a2 + b2))
            - (math.lgamma(a1) + math.lgamma(b1) - math.lgamma(a1 + b1))
        )
        return (
            log_beta_ratio
            + (a1 - a2) * _digamma(a1)
            + (b1 - b2) * _digamma(b1)
            + (a2 - a1 + b2 - b1) * _digamma(a1 + b1)
        )

    def __repr__(self) -> str:
        return f"BetaBelief(alpha={self.alpha:.3f}, beta={self.beta:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BetaBelief):
            return NotImplemented
        return math.isclose(self.alpha, other.alpha) and math.isclose(self.beta, other.beta)


def _digamma(x: float) -> float:
    """Digamma (psi) function via scipy."""
    from scipy.special import digamma
    return float(digamma(x))
