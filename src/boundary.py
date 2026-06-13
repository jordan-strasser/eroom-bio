"""Public/OSS ↔ private/enterprise artifact boundary.

Single source of truth for what eroom may write into a **public** artifact
(a committed snapshot under ``data/exports/``) versus what must stay in the
private enterprise tree.

Two boundaries live here, both machine-checkable:
  1. **Field/value gate** (original) — :func:`strip_private` / :func:`assert_public_safe`:
     keep private *values* (fine-tuned embeddings, trained boxes) out of public
     snapshots.
  2. **Query-module gate** (governing boundary, BOUNDARY.md 2026-06-12) —
     :data:`PRIVATE_QUERY_MODULES` / :func:`assert_query_private`: keep the private
     *read-path / frontier-query modules* (the paid value-extraction) out of the
     public package. Intake (write-path) and the belief-STATE are PUBLIC; the
     frontier query that composes a risk answer is PRIVATE.

Why this exists, and why it is *not* just ``.gitignore``
--------------------------------------------------------
The public snapshot is produced by ``GraphStore.export_snapshot``, which
serializes **every field on every node and edge** (``model_dump`` then
``nx.node_link_data`` — see ``store.py``). The moment a node or edge model
gains a fine-tuned embedding, a trained box, or a per-region belief field,
that value flows straight into the committed public JSON. ``.gitignore`` does
not help: ``data/exports/`` is *tracked*, and a tracked file's history is
permanent. So the boundary is enforced **in code, at serialization time**, in
three layers:

1. **Default-safe public export.** ``GraphStore.export_snapshot`` runs
   :func:`strip_private` before writing, so the public artifact is clean by
   construction even if the in-memory graph carries private values (it will,
   during a combined build).
2. **Fail-loud audit.** :func:`assert_public_safe` re-scans a payload and
   raises :class:`PrivateArtifactLeak` if a private value survives. CI runs it
   over every committed snapshot (``scripts/check_public_snapshots.py``), so a
   leak fails the build rather than shipping.
3. **Out-of-tree private root.** Private artifacts (fine-tuned weights, trained
   box params, per-region belief-field snapshots) write under
   :func:`private_root`, which **refuses to resolve inside the public repo
   working tree**.

What counts as "private" is a **naming convention** plus an explicit set, so
private fields added during the manifold build are protected automatically as
long as they follow the convention. Field *names* and *schemas* are public
(knowing that ``embedding`` is private leaks nothing — see the Apache-boundary
decision: code/schema public, trained values private); only the *values* are
stripped. A declared-but-``None`` private field is therefore always safe.

Manifold → boundary mapping (see ``future_ideas/manifold_learning.md``):
  * Manifold 1 geometry — fine-tuned embeddings, trained boxes  → private value
  * Manifold 2 — scalar ``Beta(alpha, beta)`` marginal           → PUBLIC
  *            — full per-region belief field                    → PUBLIC
  * Manifold 3 — outcome-conditioned learner + its snapshots     → private (lives
                 entirely in the enterprise repo, never imported here)

The open core is meant to be a top-shelf predictor, so the manifold-2 per-region
belief field (the ``(s,t)`` localized refinement of the scalar marginal) ships in
the PUBLIC snapshot as of the field-public move. The enterprise moat is private
DATA integration (pharma deals), NOT the prediction math — so the field's anchors
and the shared ``_belief_field_vectors`` table are public, while manifold-1 node
embeddings / trained boxes and the manifold-3 outcome learner stay private.

.. note::
   **Superseded in part by the governing write-path/read-path boundary
   (BOUNDARY.md, 2026-06-12).** The "NOT the prediction math" clause above still
   holds for the belief-STATE (public) and the manifold-2 field VALUES, but the
   **frontier query/prediction path is now PRIVATE** — composing a customer risk
   answer is the paid value-extraction (see :data:`PRIVATE_QUERY_MODULES`). The
   belief-state remains public; only the read-path that *queries* it moves private.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "BoundaryViolation",
    "PrivateArtifactLeak",
    "PrivateArtifactMisrouted",
    "PrivateRootMislocated",
    "QueryBoundaryViolation",
    "is_private_field",
    "strip_private",
    "find_private_fields",
    "assert_public_safe",
    "private_root",
    "require_under_private_root",
    "PRIVATE_FIELD_NAMES",
    "PRIVATE_FIELD_SUFFIXES",
    "PUBLIC_FIELD_NAMES",
    "PRIVATE_QUERY_MODULES",
    "is_private_query_module",
    "pending_query_modules",
    "reintroduced_query_modules",
    "assert_query_private",
]


# ── The contract ─────────────────────────────────────────────────────────────

# Any serialized field whose name ends with one of these is a private value.
# New private fields added during the manifold build inherit protection for
# free by following the convention (e.g. ``source_embedding``, ``box_field``).
PRIVATE_FIELD_SUFFIXES: tuple[str, ...] = (
    "_embedding",
    "_embeddings",
    "_field",
    "_box",
    "_boxes",
    "_anchors",
    "_weights",
)

# Exact field names that are private but don't carry a convention suffix.
PRIVATE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "embedding",
        "embeddings",
        "region_anchors",
        "anchors",
        "box",
        "box_min",
        "box_max",
        "box_params",
        "ellipsoid_mean",
        "ellipsoid_cov",
        "manifold",
        "manifold_params",
    }
)

# Field names that are PUBLIC even though they'd otherwise match a private name
# or suffix. ``belief_field`` ends in the private ``_field`` suffix and contains a
# (privately-named) ``anchors`` list, but the manifold-2 field is open-core (see
# the module docstring): its whole subtree is kept and NOT recursed into, so the
# inner ``anchors`` survive while ``anchors``/``region_anchors`` stay private
# everywhere else. The shared ``_belief_field_vectors`` table is a top-level key
# that matches no private convention, so it passes through unstripped.
PUBLIC_FIELD_NAMES: frozenset[str] = frozenset({"belief_field"})


def is_private_field(name: str) -> bool:
    """True if a field with this name carries a private (enterprise) value.

    :data:`PUBLIC_FIELD_NAMES` wins first (an explicit open-core override), then
    :data:`PRIVATE_FIELD_NAMES` / :data:`PRIVATE_FIELD_SUFFIXES` so that new
    fields following the naming convention are protected without touching this
    module.
    """
    if name in PUBLIC_FIELD_NAMES:
        return False
    return name in PRIVATE_FIELD_NAMES or name.endswith(PRIVATE_FIELD_SUFFIXES)


# ── Errors ───────────────────────────────────────────────────────────────────


class BoundaryViolation(RuntimeError):
    """Base class for any public/private boundary breach."""


class PrivateArtifactLeak(BoundaryViolation):
    """A private value was found in something about to be published."""


class PrivateArtifactMisrouted(BoundaryViolation):
    """A private snapshot was directed somewhere outside the private root."""


class PrivateRootMislocated(BoundaryViolation):
    """``EROOM_PRIVATE_ROOT`` resolves inside the public repo working tree."""


class QueryBoundaryViolation(BoundaryViolation):
    """A private read-path / frontier-query module is present in the public tree.

    The write-path/read-path counterpart of :class:`PrivateArtifactLeak`: that one
    guards a *value* leaking into a public snapshot; this one guards a private
    *query module* (the paid value-extraction) shipping in the public package. See
    :data:`PRIVATE_QUERY_MODULES` and :func:`assert_query_private`, and the governing
    boundary in ``BOUNDARY.md``.
    """


# ── Payload scanning ─────────────────────────────────────────────────────────


def _is_empty(value: Any) -> bool:
    """Declared-but-unpopulated private fields are safe; only real values leak."""
    return value is None or value == [] or value == {} or value == ""


def strip_private(obj: Any) -> Any:
    """Return a deep copy of ``obj`` with every private-keyed entry removed.

    Recurses through dicts and lists. Used by the public export path so the
    written artifact is clean by construction regardless of what the in-memory
    graph holds.
    """
    if isinstance(obj, dict):
        return {
            # Keep a public-override subtree (belief_field) WHOLE — don't recurse,
            # so its inner (privately-named) ``anchors`` survive.
            k: (v if k in PUBLIC_FIELD_NAMES else strip_private(v))
            for k, v in obj.items()
            if not is_private_field(k)
        }
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def find_private_fields(obj: Any, _path: str = "") -> list[str]:
    """Return dotted paths to every populated private field in ``obj``.

    Empty/``None`` private fields are ignored (a declared schema field that
    was never populated cannot leak anything).
    """
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{_path}.{k}" if _path else str(k)
            if is_private_field(k) and not _is_empty(v):
                hits.append(here)
            if k in PUBLIC_FIELD_NAMES:
                continue  # open-core subtree (inner `anchors` is public) — don't scan
            hits.extend(find_private_fields(v, here))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_private_fields(v, f"{_path}[{i}]"))
    return hits


def assert_public_safe(payload: Any, *, source: str = "") -> None:
    """Raise :class:`PrivateArtifactLeak` if ``payload`` holds private values.

    Safe to call on any loaded snapshot — this is the audit entry point for
    ``scripts/check_public_snapshots.py`` and tests.
    """
    hits = find_private_fields(payload)
    if hits:
        shown = sorted(set(hits))
        where = source or "<payload>"
        more = "" if len(shown) <= 20 else f" (+{len(shown) - 20} more)"
        raise PrivateArtifactLeak(
            f"Private artifact would leak into public output {where}: "
            f"{', '.join(shown[:20])}{more}. "
            "These belong in the private snapshot (export_private_snapshot) / "
            "the eroom-enterprise repo, not a committed data/exports/ file."
        )


# ── Query-side private: the read-path / frontier-query module registry ───────
#
# The governing boundary (BOUNDARY.md, 2026-06-12) follows the write-path /
# read-path seam: intake is PUBLIC (contributors audit it to trust you with their
# data); the frontier QUERY that composes a customer risk answer from the
# belief-state is PRIVATE (the paid value-extraction). The belief-STATE itself is
# PUBLIC — orthogonal to this.
#
# This registry is the machine-checkable form of "query-side private": the set of
# read-path module paths that must NOT ship in the public package. It is the
# read-path counterpart of PRIVATE_FIELD_NAMES (which guards values inside a public
# snapshot); here we guard whole *modules* against living in the public tree.
#
# Each entry maps a PUBLIC-TREE path -> metadata:
#   dest      : where it relocates in the private package (eroom-enterprise)
#   relocated : True  -> the move has happened; the module MUST be absent from the
#                        public tree. assert_query_private() ERRORS if it reappears
#                        (the permanent regression guard).
#               False -> relocation is APPROVED-NEXT but not yet executed (the code
#                        still lives here). The check reports it as PENDING and does
#                        NOT fail the build, UNLESS strict=True. This is the state in
#                        which the policy is encoded (Part 1) before the code move
#                        (the separate, approved next step — see BOUNDARY_SPLIT_PLAN.md).
#
# When the relocation lands: extract the public baseline reference query first (it
# stays public + runnable + drives the validation harnesses — see the split plan),
# move the frontier to `dest`, then flip `relocated` to True here.
PRIVATE_QUERY_MODULES: dict[str, dict[str, Any]] = {
    "src/prediction/path_query.py": {
        "dest": "eroom_enterprise/prediction/frontier_query.py",
        "relocated": False,
        "note": (
            "Frontier query composition: P(success)/risk composition, "
            "safety-penalty composition, query-time on/off-target decomposition, "
            "weakest-link aggregation, ranking. STRADDLER — split first: a public "
            "baseline reference query (src/prediction/baseline_query.py) stays, the "
            "frontier relocates."
        ),
    },
    "src/prediction/field_prediction.py": {
        "dest": "eroom_enterprise/prediction/field_prediction.py",
        "relocated": False,
        "note": (
            "(s,t)-localized frontier prediction — composes a localized chain "
            "P(success) from the manifold-2 belief field. Read-path."
        ),
    },
    "src/prediction/provenance.py": {
        "dest": "eroom_enterprise/prediction/provenance.py",
        "relocated": False,
        "note": (
            "Cross-indication frontier analysis — re-runs the query aggregation "
            "(predict_clinical_hypothesis / _aggregate_samples) for the thesis probe. "
            "Read-path. The structural bridge census (find_biology_bridges / "
            "find_mechanism_bridges) is belief-state read-only and could be carved "
            "back public — see BOUNDARY_SPLIT_PLAN.md."
        ),
    },
}


def is_private_query_module(path: str) -> bool:
    """True if ``path`` (a repo-relative module path) is a registered private
    read-path module. Normalizes separators so callers can pass either form."""
    norm = path.replace("\\", "/")
    return norm in PRIVATE_QUERY_MODULES


def _query_modules_present(tree_root: Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Registered private query modules that currently exist in the public tree.

    Returns ``(public_path, metadata)`` pairs. ``tree_root`` defaults to the public
    repo root (:func:`_public_repo_root`).
    """
    root = tree_root if tree_root is not None else _public_repo_root()
    return [
        (rel, meta)
        for rel, meta in PRIVATE_QUERY_MODULES.items()
        if (root / rel).exists()
    ]


def pending_query_modules(tree_root: Path | None = None) -> list[str]:
    """Public-tree paths of registered modules still awaiting relocation
    (``relocated=False``) but present in the tree — the encode-now / move-later
    state. Surfaced as a non-fatal notice by the CI hook."""
    return [
        rel for rel, meta in _query_modules_present(tree_root)
        if not meta.get("relocated", False)
    ]


def reintroduced_query_modules(tree_root: Path | None = None) -> list[str]:
    """Public-tree paths of registered modules marked ``relocated=True`` that have
    reappeared in the public tree — a hard regression (the private query path leaked
    back into the open core)."""
    return [
        rel for rel, meta in _query_modules_present(tree_root)
        if meta.get("relocated", False)
    ]


def assert_query_private(
    tree_root: Path | None = None, *, strict: bool = False
) -> None:
    """Raise :class:`QueryBoundaryViolation` if a private query module is in the
    public tree.

    Mirrors the spirit of :func:`assert_public_safe`. Two tiers, matching the
    registry's ``relocated`` flag:

    * **Always fatal** — a ``relocated=True`` module that has reappeared
      (:func:`reintroduced_query_modules`). The frontier query leaked back into the
      public core; this must fail the build.
    * **Fatal only under ``strict``** — a ``relocated=False`` module still present
      (:func:`pending_query_modules`). This is the expected encode-now/move-later
      state, so by default it is a notice, not a failure; ``strict=True`` (the
      post-relocation gate) treats it as a violation too.
    """
    reintroduced = reintroduced_query_modules(tree_root)
    pending = pending_query_modules(tree_root) if strict else []
    offenders = sorted(set(reintroduced) | set(pending))
    if offenders:
        raise QueryBoundaryViolation(
            "Private read-path / frontier-query module(s) present in the public "
            f"tree: {', '.join(offenders)}. These belong in the private package "
            "(eroom-enterprise) per the governing write-path/read-path boundary "
            "(BOUNDARY.md) — they compose the paid query answer, which contributors "
            "do not need to audit. See PRIVATE_QUERY_MODULES + BOUNDARY_SPLIT_PLAN.md."
        )


# ── Private artifact location ────────────────────────────────────────────────

DEFAULT_PRIVATE_ROOT = Path.home() / ".eroom" / "private"


def _public_repo_root() -> Path:
    """Directory of the public repo (the one containing this package)."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent  # fall back to the src/ parent


def private_root(*, create: bool = False) -> Path:
    """Resolve the private-artifact root, refusing any location inside the repo.

    Reads ``EROOM_PRIVATE_ROOT`` (defaulting to ``~/.eroom/private``). Raises
    :class:`PrivateRootMislocated` if that resolves to the public repo tree —
    you cannot configure private artifacts into the committed working tree.
    """
    raw = os.environ.get("EROOM_PRIVATE_ROOT")
    root = (
        Path(raw).expanduser().resolve()
        if raw
        else DEFAULT_PRIVATE_ROOT.resolve()
    )
    repo = _public_repo_root()
    if root == repo or repo in root.parents:
        raise PrivateRootMislocated(
            f"EROOM_PRIVATE_ROOT ({root}) is inside the public repo ({repo}). "
            "Private artifacts must live outside the public working tree — "
            "point it at the eroom-enterprise repo or leave it at "
            f"{DEFAULT_PRIVATE_ROOT}."
        )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def require_under_private_root(filepath: str | Path) -> Path:
    """Resolve ``filepath`` and assert it lives under :func:`private_root`.

    Guards :meth:`GraphStore.export_private_snapshot` so a private snapshot can
    never be written to ``data/exports/`` by a slipped argument.
    """
    p = Path(filepath).expanduser().resolve()
    root = private_root()
    if p != root and root not in p.parents:
        raise PrivateArtifactMisrouted(
            f"Refusing to write a private snapshot to {p}: it is not under the "
            f"private root ({root}). Set EROOM_PRIVATE_ROOT or write under it."
        )
    return p
