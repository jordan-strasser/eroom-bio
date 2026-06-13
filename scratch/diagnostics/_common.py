"""Shared helpers for the edge-class diagnostic probes.

Read-only. Loads the annotated graph + cached annotations and exposes the
edge-class typology that the whole investigation hinges on.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.graph.models import EdgeBeliefState
from src.graph.store import GraphStore

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "data" / "annotations"
CORPORA_DIR = ROOT / "data" / "corpora"

# ── The central typology: three semantically distinct edge classes ──────────
EFFICACY = "efficacy"        # conjunctive causal spine
MEASUREMENT = "measurement"  # detection / validity (triangle + pop)
SAFETY = "safety"            # disjunctive killers
MODULATION = "modulation"    # combo conditioning
STRUCTURAL = "structural"    # no Beta belief

EDGE_CLASS: dict[str, str] = {
    "affects": EFFICACY,
    "modulates_via": EFFICACY,
    "mechanism_affects": EFFICACY,
    "biology_drives": EFFICACY,
    "reflects_biology": MEASUREMENT,
    "endpoint_captures": MEASUREMENT,
    "responds_differently": MEASUREMENT,
    "causes_ae": SAFETY,
    "target_associated_ae": SAFETY,
    "modulates_efficacy_of": MODULATION,
    "composed_of": STRUCTURAL,
    "subtype_of": STRUCTURAL,
}


def load_graph(path: str) -> GraphStore:
    g = GraphStore()
    g.import_snapshot(path)
    return g


def iter_edges(g: GraphStore):
    """Yield (src, tgt, edge_type, EdgeBeliefState|None) for every edge."""
    for u, v, key, data in g._graph.edges(data=True, keys=True):
        belief_data = data.get("belief")
        belief = None
        if belief_data:
            try:
                belief = EdgeBeliefState.model_validate(belief_data)
            except Exception:  # noqa: BLE001
                belief = None
        yield u, v, key, belief


def node_type(g: GraphStore, nid: str) -> str:
    try:
        return g._graph.nodes[nid].get("node_type", "?")
    except Exception:  # noqa: BLE001
        return "?"


def trial_evidence_ncts(belief: EdgeBeliefState) -> list[str]:
    """The NCT ids of TRIAL-sourced (outcome-conditioned / AE) evidence on an edge.

    Curated database records have non-NCT source_ids; we keep only NCT* so the
    'host trials' of an edge are the clinical trials whose outcome touched it.
    """
    out = []
    for rec in belief.evidence:
        sid = getattr(rec, "source_id", "") or ""
        if sid.startswith("NCT"):
            out.append(sid)
    return out


def load_corpus(name: str) -> list[str]:
    path = CORPORA_DIR / (name if name.endswith(".txt") else f"{name}.txt")
    out = []
    for line in path.read_text().splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.startswith("NCT"):
            out.append(clean)
    return out


def load_annotation(nct: str) -> tuple[dict | None, dict | None]:
    ext_p = ANN_DIR / f"{nct}_extraction.json"
    cls_p = ANN_DIR / f"{nct}_classification.json"
    ext = cls = None
    if ext_p.exists():
        try:
            ext = json.loads(ext_p.read_text())
        except json.JSONDecodeError:
            ext = None
    if cls_p.exists():
        try:
            cls = json.loads(cls_p.read_text())
        except json.JSONDecodeError:
            cls = None
    return ext, cls


def primary_failure_mode(cls: dict | None) -> str:
    if not cls:
        return ""
    modes = cls.get("failure_modes") or []
    if not modes:
        return ""
    # highest-confidence mode is primary (mirrors classifier.py)
    modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
    return modes[0].get("mode", "")


# Map each mechanistic FailureMode to the edge class the failure IMPLICATES.
FAILMODE_TO_CLASS: dict[str, str] = {
    "no_target_engagement": EFFICACY,
    "target_engaged_biology_not_moved": EFFICACY,
    "biology_moved_endpoint_flat": MEASUREMENT,
    "wrong_timeframe": MEASUREMENT,
    "high_placebo_response": MEASUREMENT,
    "efficacy_in_subgroup_only": MEASUREMENT,   # population (responds_differently)
    "wrong_population": MEASUREMENT,            # population
    "dose_limiting_toxicity": SAFETY,
    "underpowered": "operational",
    "manufacturing_or_delivery": "operational",
    "commercial_not_scientific": "business",
    "insufficient_information": "none",
    "multiple_factors": "mixed",
}
