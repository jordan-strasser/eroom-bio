"""Manifold 2: the per-region edge belief field (A.3).

PUBLIC code/math, per the open-methods boundary (statements 2 & 4: code/schema
public, the populated edge *weights* private). The scalar ``Beta(alpha, beta)``
on ``EdgeBeliefState`` stays the public marginal; this localizes evidence by
``(s, t)`` so that distinct evidence on the *same* edge — e.g. a trial's
"VEGF → thrombopoietin" vs another's "VEGF → hypertension" — lands at different
surface points instead of averaging together.

The populated anchors are the private artifact: they serialize to the
``belief_field`` container on ``EdgeBeliefState`` (a field name ``src/boundary``
strips from public snapshots) and to private snapshots only. The per-record
``(s, t)`` come from BioLORD embeddings of the source/target descriptions and
are likewise stripped (``*_embedding``).

Representation: a sparse anchor list. Each evidence record adds
``(s_i, t_i, alpha_i, beta_i)`` with ``(alpha_i, beta_i) = (n_eff·p_obs,
n_eff·(1−p_obs))`` — the same increment the scalar ``apply_virtual_evidence``
applies. Query at ``(s', t')`` is cosine-kernel-weighted aggregation. Chain
prediction integrates along the trajectory (the prediction engine — *not*
manifold 3, which conditions on outcomes and lives in ``eroom-enterprise``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any


def field_enabled() -> bool:
    """True when the manifold-2 belief-field path is switched on
    (``EROOM_BELIEF_FIELD``). Default off — the scalar Beta path is unchanged
    and public predictions are byte-identical until this is set."""
    return os.environ.get("EROOM_BELIEF_FIELD", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
DEFAULT_BANDWIDTH = 0.25  # cosine-kernel bandwidth; smaller = sharper locality
FALLBACK_STRENGTH = 2.0   # pseudo-count mass of the marginal fallback (see query)

VECTOR_NDIGITS = 6  # anchor-vector rounding; cosine-lossless for unit-norm BioLORD


def _resolve_vec(v: Any, vectors: list[list[float]] | None) -> list[float]:
    """Resolve one anchor axis to its vector.

    An ``int`` is an index into the shared ``vectors`` dedup table — returned
    **by reference**, so every anchor with the same index shares one list object
    (the in-memory half of the dedup). A ``list`` is an inline vector (old
    snapshot format) and is copied to floats. See :func:`index_anchor_vectors`.
    """
    if isinstance(v, int):
        if vectors is None:
            raise ValueError("anchor vector index requires a vector table")
        return vectors[v]
    return [float(x) for x in v]


def index_anchor_vectors(
    links: list[dict], *, ndigits: int = VECTOR_NDIGITS
) -> list[list[float]]:
    """Dedup anchor ``(s, t)`` vectors across all edges into one shared table.

    For every edge's ``belief.belief_field.anchors``, replace each ``s``/``t``
    float list with an int index into the returned table (rounded to ``ndigits``;
    cosine-lossless at 6dp for unit-norm BioLORD vectors). The same description
    embedding recurs across thousands of anchors, so the table is far smaller
    than the raw anchor count — this is the dominant size win for the private
    field snapshot.

    Rewrites each touched link's ``belief`` to **fresh** dicts; the nested
    objects shared with a live graph (``node_link_data`` shallow-copies link
    dicts but shares their values) are never mutated. Returns the table, empty
    if no anchor carries an inline vector.
    """
    table: list[list[float]] = []
    index: dict[tuple[float, ...], int] = {}

    def intern(v: Any) -> Any:
        if not isinstance(v, list):
            return v  # already an index (idempotent) or absent
        key = tuple(round(float(x), ndigits) for x in v)
        idx = index.get(key)
        if idx is None:
            idx = len(table)
            index[key] = idx
            table.append(list(key))
        return idx

    for e in links:
        belief = e.get("belief")
        if not isinstance(belief, dict):
            continue
        bf = belief.get("belief_field")
        if not isinstance(bf, dict) or not bf.get("anchors"):
            continue
        new_anchors, changed = [], False
        for a in bf["anchors"]:
            if not isinstance(a, dict):
                new_anchors.append(a)  # malformed/stub anchor — pass through
                continue
            na = {
                "s": intern(a.get("s")), "t": intern(a.get("t")),
                "alpha": a.get("alpha"), "beta": a.get("beta"),
            }
            if a.get("nct"):  # preserve trial provenance through the dedup
                na["nct"] = a["nct"]
            new_anchors.append(na)
            changed = True
        if changed:
            e["belief"] = {**belief, "belief_field": {**bf, "anchors": new_anchors}}
    return table


def rehydrate_anchor_vectors(
    links: list[dict], table: list[list[float]] | None
) -> None:
    """Inverse of :func:`index_anchor_vectors`: replace each anchor's int ``s``/
    ``t`` index with the **shared** list object from ``table`` (repeated vectors
    become one object in memory, not N copies). Mutates the anchor dicts in
    place — call only on a freshly-parsed payload, never on a live graph. No-op
    if ``table`` is falsy (old inline-format snapshot)."""
    if not table:
        return
    for e in links:
        belief = e.get("belief")
        if not isinstance(belief, dict):
            continue
        bf = belief.get("belief_field")
        if not isinstance(bf, dict):
            continue
        for a in bf.get("anchors") or []:
            if not isinstance(a, dict):
                continue
            s, t = a.get("s"), a.get("t")
            if isinstance(s, int):
                a["s"] = table[s]
            if isinstance(t, int):
                a["t"] = table[t]


@dataclass
class FieldAnchor:
    """One evidence record's localized contribution: ``(s, t) → Beta(α, β)``.

    ``nct`` is the contributing trial (the record's ``source_id``) — provenance
    that makes a true leave-one-out trivial on the field: the kernel query is an
    additive sum over anchors, so dropping a trial's anchors removes exactly its
    contribution (no replay, unlike the order/redundancy-dependent scalar). None
    for non-trial / legacy anchors.
    """

    s: list[float]
    t: list[float]
    alpha: float  # n_eff * p_obs       (success pseudo-count increment)
    beta: float   # n_eff * (1 - p_obs) (failure pseudo-count increment)
    nct: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"s": self.s, "t": self.t, "alpha": self.alpha, "beta": self.beta}
        if self.nct:
            d["nct"] = self.nct
        return d

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], vectors: list[list[float]] | None = None
    ) -> "FieldAnchor":
        return cls(
            s=_resolve_vec(d["s"], vectors),
            t=_resolve_vec(d["t"], vectors),
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            nct=d.get("nct"),
        )


@dataclass
class BeliefField:
    """Sparse anchor-list surface over the joint (source, target) space.

    ``marginal_alpha/beta`` carry the edge's scalar ``Beta`` so that a query in
    a region with no nearby anchors falls back to the **pooled edge belief**
    (the public marginal), not to a flat 0.5 — making the field a strict
    *refinement* of the scalar: local where there's evidence, pooled otherwise.
    Default (1, 1) reproduces the old flat-prior behaviour.
    """

    anchors: list[FieldAnchor] = field(default_factory=list)
    bandwidth: float = DEFAULT_BANDWIDTH
    marginal_alpha: float = PRIOR_ALPHA
    marginal_beta: float = PRIOR_BETA
    fallback_strength: float = FALLBACK_STRENGTH

    def to_dict(self) -> dict[str, Any]:
        return {
            "bandwidth": self.bandwidth,
            "marginal_alpha": self.marginal_alpha,
            "marginal_beta": self.marginal_beta,
            "fallback_strength": self.fallback_strength,
            "anchors": [a.to_dict() for a in self.anchors],
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], vectors: list[list[float]] | None = None
    ) -> "BeliefField":
        return cls(
            anchors=[FieldAnchor.from_dict(a, vectors) for a in d.get("anchors", [])],
            bandwidth=float(d.get("bandwidth", DEFAULT_BANDWIDTH)),
            marginal_alpha=float(d.get("marginal_alpha", PRIOR_ALPHA)),
            marginal_beta=float(d.get("marginal_beta", PRIOR_BETA)),
            fallback_strength=float(d.get("fallback_strength", FALLBACK_STRENGTH)),
        )

    def without_trial(self, nct: str) -> "BeliefField":
        """Field leave-one-out: a copy with ``nct``'s anchors removed. Exact
        because the kernel query is additive over anchors — no replay, unlike
        the scalar. The marginal fallback is unchanged (still carries the scalar
        incl. nct), so this is clean in the dense cross-trial regions where the
        field adds signal and degrades to the (leaky) marginal far from anchors;
        pair with a scalar-LOO marginal for fully clean. Anchors are shared by
        reference (queries are read-only)."""
        return BeliefField(
            anchors=[a for a in self.anchors if a.nct != nct],
            bandwidth=self.bandwidth,
            marginal_alpha=self.marginal_alpha,
            marginal_beta=self.marginal_beta,
            fallback_strength=self.fallback_strength,
        )

    def fallback_prior(self) -> tuple[float, float]:
        """The marginal-centered prior pseudo-counts: ``fallback_strength``
        units of mass placed at the edge's scalar mean. Far-from-evidence
        queries return this → ``E[p]`` = the pooled edge mean."""
        total = self.marginal_alpha + self.marginal_beta
        p = self.marginal_alpha / total if total > 0 else 0.5
        return self.fallback_strength * p, self.fallback_strength * (1.0 - p)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def apply_virtual_evidence_local(
    field_: BeliefField,
    *,
    s: list[float],
    t: list[float],
    n_eff: float,
    p_obs: float,
    nct: str | None = None,
) -> BeliefField:
    """Localized counterpart of ``apply_virtual_evidence``: add one anchor at
    ``(s, t)`` rather than updating the whole-edge scalar. Mutates and returns
    ``field_``. Same ``(n_eff, p_obs)`` semantics as the scalar path. ``nct``
    tags the anchor with its contributing trial (provenance for field-LOO).
    """
    if n_eff < 0:
        raise ValueError(f"n_eff must be non-negative, got {n_eff!r}")
    if not 0.0 <= p_obs <= 1.0:
        raise ValueError(f"p_obs must be in [0, 1], got {p_obs!r}")
    field_.anchors.append(
        FieldAnchor(s=list(s), t=list(t), alpha=n_eff * p_obs,
                    beta=n_eff * (1.0 - p_obs), nct=nct)
    )
    return field_


def query(
    field_: BeliefField,
    s: list[float],
    t: list[float],
) -> tuple[float, float]:
    """Kernel-weighted ``Beta(alpha, beta)`` at ``(s, t)``.

    ``weight_i = exp((cos(s, s_i) + cos(t, t_i) − 2) / bandwidth)`` — an anchor
    at the exact ``(s, t)`` gets weight 1; far anchors decay toward 0. Returns
    ``(prior_a + Σ w_i·alpha_i, prior_b + Σ w_i·beta_i)`` where ``(prior_a,
    prior_b)`` is the field's marginal-centered fallback, so a query in a region
    with no nearby evidence falls back to the **pooled edge belief** (not 0.5).
    """
    a, b = field_.fallback_prior()
    for anc in field_.anchors:
        w = math.exp((_cosine(s, anc.s) + _cosine(t, anc.t) - 2.0) / field_.bandwidth)
        a += w * anc.alpha
        b += w * anc.beta
    return (a, b)


def expected_p(field_: BeliefField, s: list[float], t: list[float]) -> float:
    """Localized expected success probability at ``(s, t)``."""
    a, b = query(field_, s, t)
    return a / (a + b)


def localize_record(
    record: Any,
    *,
    source_description: str,
    target_description: str,
    embed_fn,
) -> Any:
    """Populate an evidence record's (s, t) localization from descriptions.

    Sets ``source/target_description_in_trial`` (public text) and
    ``source/target_embedding`` (private, BioLORD via ``embed_fn``) so the
    store's flag-gated path can place the record on the edge belief field.
    Duck-typed on the record (no models import). Mutates and returns it; a
    blank description leaves that side unlocalized. This is the seam the
    attributor calls once it carries the chain's A.0b descriptions.
    """
    if source_description:
        record.source_description_in_trial = source_description
        record.source_embedding = embed_fn(source_description)
    if target_description:
        record.target_description_in_trial = target_description
        record.target_embedding = embed_fn(target_description)
    return record


def chain_integral(steps: list[tuple[BeliefField, list[float], list[float]]]) -> float:
    """Geometric mean of the local Beta means along a chain trajectory.

    ``steps`` is the chain's ``(field, s, t)`` per edge. This composes
    manifolds 1∘2 into the prediction engine; the outcome-conditioned
    manifold 3 is a separate, private learner (eroom-enterprise).
    """
    if not steps:
        return PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA)
    log_sum = 0.0
    for f, s, t in steps:
        log_sum += math.log(max(expected_p(f, s, t), 1e-9))
    return math.exp(log_sum / len(steps))
