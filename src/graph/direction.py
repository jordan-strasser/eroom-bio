"""Native modulation-direction resolution from ChEMBL ``action_type``.

Direction (agonist vs antagonist) is a first-class part of the spine, NOT a
post-hoc LLM/keyword band-aid: it is read from ChEMBL's curated, structured
``action_type`` field — the same source the populator already uses to resolve a
compound's target and mechanism. A compound with no ChEMBL mechanism resolves to
``UNKNOWN`` (we never guess a direction from free text).

Direction is an EDGE-LEVEL partition (carried on the chain, used to split the
shared downstream beliefs) — it never fragments a node. BTLA stays one node; its
agonist and antagonist evidence simply land in different direction buckets on the
same edge.

Baked ON via ``CONFIG.direction`` (src/config.py); formerly the ``EROOM_DIRECTION``
flag (default OFF). Direction is always stamped at build and applied at query time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CONFIG

# Direction buckets (the sign of the pharmacological action ON THE TARGET).
AGONIST = "agonist"        # activates / increases target function
ANTAGONIST = "antagonist"  # inhibits / blocks / removes target function
UNKNOWN = "unknown"        # no curated action_type, or a non-directional one

# ChEMBL ``action_type`` vocabulary → direction bucket. Conservative: anything
# without a clear activating/suppressing sign (enzymes, binding/vaccine/other)
# maps to UNKNOWN rather than a guess.
_ACTION_TYPE_DIRECTION: dict[str, str] = {
    # activating
    "AGONIST": AGONIST,
    "PARTIAL AGONIST": AGONIST,
    "ACTIVATOR": AGONIST,
    "POSITIVE MODULATOR": AGONIST,
    "POSITIVE ALLOSTERIC MODULATOR": AGONIST,
    "OPENER": AGONIST,
    "STABILISER": AGONIST,
    # suppressing
    "INHIBITOR": ANTAGONIST,
    "ANTAGONIST": ANTAGONIST,
    "INVERSE AGONIST": ANTAGONIST,
    "BLOCKER": ANTAGONIST,
    "NEGATIVE MODULATOR": ANTAGONIST,
    "NEGATIVE ALLOSTERIC MODULATOR": ANTAGONIST,
    "DISRUPTING AGENT": ANTAGONIST,
    "SEQUESTERING AGENT": ANTAGONIST,
    "CHELATING AGENT": ANTAGONIST,
    "DEGRADER": ANTAGONIST,
    "PROTEOLYSIS TARGETING CHIMERA": ANTAGONIST,
}


def enabled() -> bool:
    """Whether native direction resolution is active. Baked ON; see src/config.py."""
    return CONFIG.direction


def direction_for_action_type(action_type: str | None) -> str:
    """Map a single ChEMBL ``action_type`` verb to a direction bucket."""
    if not action_type:
        return UNKNOWN
    return _ACTION_TYPE_DIRECTION.get(action_type.strip().upper(), UNKNOWN)


def direction_from_mechanisms(
    mechanisms: list[dict], target_chembl_id: str | None = None
) -> str:
    """Resolve a compound's direction from its ChEMBL mechanism records.

    ``mechanisms`` is the list returned by ``ChEMBLClient.get_drug_mechanisms``
    (each a dict with ``action_type`` / ``target_chembl_id``). When
    ``target_chembl_id`` is given, only mechanisms acting on that target are
    considered (a drug can hit several targets with different actions). Returns the
    single direction if all relevant mechanisms agree, ``UNKNOWN`` if there are
    none, and — when curated records genuinely disagree — the majority, falling
    back to ``UNKNOWN`` on a tie (we do not invent a sign).
    """
    if not mechanisms:
        return UNKNOWN
    relevant = mechanisms
    if target_chembl_id:
        scoped = [m for m in mechanisms
                  if (m.get("target_chembl_id") or "") == target_chembl_id]
        if scoped:
            relevant = scoped
    dirs = [direction_for_action_type(m.get("action_type")) for m in relevant]
    dirs = [d for d in dirs if d != UNKNOWN]
    if not dirs:
        return UNKNOWN
    if all(d == dirs[0] for d in dirs):
        return dirs[0]
    # genuine curated disagreement → majority, tie → unknown
    n_ag = dirs.count(AGONIST)
    n_an = dirs.count(ANTAGONIST)
    if n_ag == n_an:
        return UNKNOWN
    return AGONIST if n_ag > n_an else ANTAGONIST


# ── Chain stamping (populate-time, reads the curated ChEMBL mechanism cache) ──
_MECH_CACHE_PATH = Path("data/cache/chembl_mechanisms.json")


def load_mechanism_cache(path: Path | None = None) -> dict:
    p = path or _MECH_CACHE_PATH
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _mechanisms_for(cache: dict, chembl_id: str | None) -> list[dict]:
    if not chembl_id:
        return []
    rec = cache.get(f"mechanism:{chembl_id}") or cache.get(chembl_id) or {}
    return rec.get("mechanisms") or []


def stamp_directions(graph: Any, cache: dict | None = None) -> dict:
    """Stamp ``chain.direction`` on every chain from the compound's ChEMBL
    ``action_type`` (the native source). Mutates the graph's trial subgraphs
    in place. No-op-safe: a compound with no chembl_id or no curated mechanism
    stays ``UNKNOWN``. Returns coverage stats. Combo compounds resolve from the
    synthesized node's own chembl_id if present, else UNKNOWN (per-constituent
    direction is a later refinement).
    """
    cache = cache if cache is not None else load_mechanism_cache()
    nodes = graph._graph.nodes  # noqa: SLF001
    stats = {"chains": 0, AGONIST: 0, ANTAGONIST: 0, UNKNOWN: 0}
    cid_dir: dict[str, str] = {}
    for _nct, ts in graph.trial_subgraphs.items():
        for ch in ts.chains:
            cid = ch.compound_id
            if cid not in cid_dir:
                chembl_id = (nodes[cid].get("chembl_id")
                             if cid in nodes else None)
                cid_dir[cid] = direction_from_mechanisms(
                    _mechanisms_for(cache, chembl_id))
            d = cid_dir[cid]
            ch.direction = d
            stats["chains"] += 1
            stats[d] += 1
    return stats
