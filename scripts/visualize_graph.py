"""HTML visualizer for eroom causal-hypothesis chains.

Two modes:

  * ``--mode chain`` (default) — the DEBUG view. Lays causal hypothesis chains out
    in the canonical topology:

        COMPOUND →(binds_to)→ TARGET →(modulates_via)→ MECHANISM
                 →(mechanism_affects)→ BIOLOGY →(biology_drives)→ INDICATION
                 →(responds_differently)→ POPULATION

    with ENDPOINT as a satellite above (BIOLOGY -reflects_biology→ ENDPOINT
    -endpoint_captures→ INDICATION) and ADVERSE EVENTS below (COMPOUND -causes_ae,
    TARGET -target_associated_ae). Every edge is labeled with its type + E[p].
    Renders the selected chains twice — **BEFORE MERGE** (each trial's own
    per-trial node instances) above **AFTER MERGE** (assembled graph) — so a merge
    is read by comparing a node across the two blocks. Click any box for details.

  * ``--mode edge`` — per-backbone-edge view: the edge's (s,t) (= its source/target
    NODE descriptions) + the contributor list ({nct, arm, drug}) of every chain
    that traverses the merged edge — honest merge provenance read from the graph.

  * ``--mode graph`` — THE PRODUCT overview. ONE merged force-directed network of
    the whole snapshot: every node appears exactly once, so shared mechanism /
    biology / population nodes visibly fan in from every compound & trial that uses
    them (the compounding cross-trial / cross-indication thesis, in one picture).
    Type-colored + persistent legend, type-filter checkboxes (noisy AE/endpoint/
    trial types hidden by default), search-to-zoom, click→detail panel, populations
    emphasis, cluster-by-node-type, and a baked (static-but-interactive) layout with
    zoom/pan map controls. Self-contained; loads vis-network from a CDN.

  * ``--mode explorer`` — force-directed whole-graph view (debug/raw; ``--mode
    graph`` is the product-ready version).

  * ``--mode all`` — chain + edge (+ explorer) in ONE self-contained HTML, behind
    a persistent top-of-page tab toggle (no reload, no separate files). Each view
    is embedded verbatim inside its own ``<iframe srcdoc>`` so their (overlapping)
    element ids / scripts stay isolated. Here the Chain view is AFTER-ONLY (the
    merged graph, no before/after split, no premerge dump): a deterministic random
    cross-section of up to ``--sample`` (default 20) trials, each sectioned by NCT
    (+ title/indication) and showing its own assembled causal-chain fan-out.

Read-only; the chain/edge/all views are pure stdlib (the explorer view loads
vis-network from a CDN). The merge DELETES loser nodes (only ids survive in
``metadata.merged_from``), so the before-block reconstructs each trial's instance
id and names it from ``data/cache/biology_id_names.json`` when available.

Usage:
    python -m scripts.visualize_graph --area mi_bu                 # 5 chains, before/after
    python -m scripts.visualize_graph --area mi_bu --trials NCT04577807,NCT05116202
    python -m scripts.visualize_graph --area mi_bu --mode explorer
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path

EXPORTS = Path("data/exports")
VIZ_DIR = Path("data/viz")

PALETTE = {
    "InterventionNode": "#3d6ae0", "CompoundNode": "#3d6ae0", "TargetNode": "#7b5cff",
    "MechanismNode": "#1f8f4e", "BiologyNode": "#1098ad", "IndicationNode": "#b8860b",
    "PopulationNode": "#e8943a", "EndpointNode": "#b03060", "AdverseEventNode": "#9c2b2b",
    "BiomarkerNode": "#00897b", "TrialNode": "#5b6472", "Other": "#7f8a9b",
}
EDGE_COLOR = {
    "binds_to": "#4f7cff", "affects": "#4f7cff", "modulates_via": "#9b6cff",
    "mechanism_affects": "#3cb44b", "biology_drives": "#1098ad",
    "responds_differently": "#e8943a", "reflects_biology": "#4f9cff",
    "endpoint_captures": "#e0559b", "causes_ae": "#e6194B",
    "target_associated_ae": "#c07b00", "modulates_efficacy_of": "#c77dff",
    "subtype_of": "#5b6472", "composed_of": "#6b7484",
}

# role → (chain field, node type, column header, column index | None=satellite)
ROLES = [
    ("compound", "compound_id", "InterventionNode", "COMPOUND", 0),
    ("target", "target_id", "TargetNode", "TARGET", 1),
    ("mechanism", "mechanism_id", "MechanismNode", "MECHANISM", 2),
    ("biology", "biology_id", "BiologyNode", "BIOLOGY", 3),
    ("endpoint", "endpoint_id", "EndpointNode", "ENDPOINT", 4),
    ("indication", "indication_id", "IndicationNode", "INDICATION", 5),
    ("population", "subgroup_population_id", "PopulationNode", "POPULATION", 6),
]
ROLE_FIELD = {r: f for r, f, _t, _h, _c in ROLES}
ROLE_TYPE = {r: t for r, _f, t, _h, _c in ROLES}
ROLE_COL = {r: c for r, _f, _t, _h, c in ROLES}
N_COLS = 7
BACKBONE = [("compound", "target", "binds_to"), ("target", "mechanism", "modulates_via"),
            ("mechanism", "biology", "mechanism_affects"), ("biology", "indication", "biology_drives"),
            ("indication", "population", "responds_differently")]
SATELLITE = [("biology", "endpoint", "reflects_biology"), ("endpoint", "indication", "endpoint_captures")]
BACKBONE_SIG = ["compound", "target", "mechanism", "biology", "indication"]

try:
    NAME_CACHE: dict = json.loads(Path("data/cache/biology_id_names.json").read_text())
except Exception:
    NAME_CACHE = {}


def _base(x) -> str:
    return re.sub(r"#NCT.*$", "", str(x))


def _resolve(name_or_path: str, suffix: str | None) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    if suffix:
        c = EXPORTS / f"{name_or_path}_{suffix}.json"
        if c.exists():
            return c
    for c in (EXPORTS / f"{name_or_path}.json", EXPORTS / name_or_path):
        if c.exists():
            return c
    raise FileNotFoundError(f"no snapshot for {name_or_path!r} (suffix={suffix!r})")


def load_snapshot(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _display_fields(node: dict) -> dict:
    """Every node attribute for the click-panel sanity check — except the 768-d
    ``embedding`` (useless to eyeball, and it bloats the inlined details blob)."""
    return {k: v for k, v in (node or {}).items() if k != "embedding"}


def _belief_summary(belief: dict | None) -> dict:
    b = belief or {}
    a, be = float(b.get("alpha", 1.0) or 1.0), float(b.get("beta", 1.0) or 1.0)
    ev, anc = b.get("evidence") or [], (b.get("belief_field") or {}).get("anchors") or []
    return {"alpha": round(a, 3), "beta": round(be, 3),
            "ep": round(a / (a + be), 3) if (a + be) else None,
            "n_evidence": len(ev), "n_anchors": len(anc)}


# ── chain view ────────────────────────────────────────────────────────────────

def _name_for(base_id: str, node_index: dict) -> str:
    if base_id in node_index:
        return node_index[base_id].get("name") or base_id
    return NAME_CACHE.get(base_id) or base_id


def _build_chains(snapshot: dict, *, before_snap: dict | None = None,
                  trials: list[str] | None = None) -> list[dict]:
    """Build the rendered-chain dicts (one per trial chain) the SVG renderers
    consume — ``{trial, arm, outcome, roles, edges, aes, merge_hits}`` — for every
    chain in ``snapshot.trial_subgraphs`` (optionally restricted to ``trials``). No
    dedup, no sorting, no limit: callers (legacy cross-trial ``extract_chains`` and
    per-trial ``extract_trial_chains``) slice/group as they need. ``before_snap``
    attaches the faithful pre-merge instance per role (used only by the before/after
    debug view; after-only callers pass ``before_snap=None``)."""
    graph = snapshot.get("graph", snapshot)
    node_index = {n["id"]: n for n in graph.get("nodes", [])}
    edge_ep: dict[tuple, float | None] = {}
    out_edges: dict[str, list] = {}
    for e in graph.get("edges", graph.get("links", [])):
        et = e.get("key") or e.get("edge_type") or ""
        ep = _belief_summary(e.get("belief"))["ep"]
        edge_ep.setdefault((e.get("source"), e.get("target")), ep)
        out_edges.setdefault(e.get("source"), []).append((et, e.get("target"), ep))

    ts = snapshot.get("trial_subgraphs", {})
    usage: dict[str, set] = {}
    for tid, t in ts.items():
        for ch in t.get("chains", []):
            for _r, fld in ROLE_FIELD.items():
                if ch.get(fld):
                    usage.setdefault(ch[fld], set()).add(tid)

    def flags_for(cid: str) -> dict:
        node = node_index.get(cid, {})
        mf = (node.get("metadata") or {}).get("merged_from") or []
        ont = node.get("ontology_id") or cid
        trials_in_mf = {m.split("#NCT")[-1] for m in mf if "#NCT" in m}
        return {"cross_trial": len(usage.get(cid, set())) > 1 or len(trials_in_mf) > 1,
                "cross_id": len({_base(cid)} | {_base(m) for m in mf}) > 1,
                "island": isinstance(ont, str) and ont.startswith("bio:"),
                "merged_from": [{"id": m, "name": _name_for(_base(m), node_index)} for m in mf],
                "used_by": sorted(usage.get(cid, set()))}

    # Faithful BEFORE block: read the pre-merge (Phase-1 union) snapshot, where each
    # chain still references its OWN per-trial node instances. Keyed by
    # (trial, arm, base-compound) so a combination arm's per-drug chains map right.
    before_chains: dict = {}
    before_nodes: dict = {}
    if before_snap:
        bg = before_snap.get("graph", before_snap)
        before_nodes = {n["id"]: n for n in bg.get("nodes", [])}
        for _tid, _t in before_snap.get("trial_subgraphs", {}).items():
            for _ch in _t.get("chains", []):
                key = (_tid, _ch.get("arm_id"), _base(_ch.get("compound_id")))
                before_chains.setdefault(key, {r: _ch.get(f) for r, f in ROLE_FIELD.items() if _ch.get(f)})

    built = []
    for tid, t in ts.items():
        if trials and tid not in trials:
            continue
        for ch in t.get("chains", []):
            roles, merge_hits = {}, 0
            bmap = before_chains.get((tid, ch.get("arm_id"), _base(ch.get("compound_id")))) if before_snap else None
            for r, fld in ROLE_FIELD.items():
                cid = ch.get(fld)
                if not cid:
                    continue
                fl = flags_for(cid)
                merge_hits += 1 if fl["cross_trial"] else 0
                node = node_index.get(cid, {})
                after = {"id": cid, "name": _name_for(cid, node_index),
                         "desc": node.get("description"), "flags": fl,
                         "fields": _display_fields(node)}
                bid = (bmap or {}).get(r)
                if bid:                      # faithful pre-merge instance for this role
                    bnode = before_nodes.get(bid, {})
                    before = {"id": bid, "name": bnode.get("name") or _name_for(_base(bid), node_index),
                              "desc": bnode.get("description"), "changed": _base(bid) != _base(cid)}
                else:                        # no premerge → mirror after (no faithful before)
                    before = {"id": cid, "name": after["name"], "desc": after["desc"], "changed": False}
                roles[r] = {"after": after, "before": before}
            edges = []
            for frm, to, et in BACKBONE + SATELLITE:
                if frm in roles and to in roles:
                    edges.append({"frm": frm, "to": to, "type": et,
                                  "ep": edge_ep.get((roles[frm]["after"]["id"], roles[to]["after"]["id"]))})
            aes = []
            for role, ae_type in (("compound", "causes_ae"), ("target", "target_associated_ae")):
                if role in roles:
                    for et, tgt, ep in out_edges.get(roles[role]["after"]["id"], []):
                        if et == ae_type and len(aes) < 6:
                            aes.append({"role": role, "type": ae_type, "ep": ep,
                                        "name": _name_for(tgt, node_index),
                                        "fields": _display_fields(node_index.get(tgt, {}))})
            built.append({"trial": tid, "arm": ch.get("arm_id"), "outcome": ch.get("outcome"),
                          "roles": roles, "edges": edges, "aes": aes, "merge_hits": merge_hits})
    return built


def _dedup_by_signature(chains: list[dict]) -> list[dict]:
    """Collapse chains sharing the same backbone signature (compound→…→indication
    node ids), keeping first occurrence — so an arm's repeated identical fan-outs
    don't draw on top of each other."""
    seen, out = set(), []
    for c in chains:
        sig = tuple(c["roles"].get(r, {}).get("after", {}).get("id") for r in BACKBONE_SIG)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def extract_chains(snapshot: dict, *, before_snap: dict | None = None,
                   limit: int, trials: list[str] | None) -> dict:
    """Legacy cross-trial chain selection: build all chains, sort by merge density,
    dedup by backbone signature, take ``limit`` (or all, when ``trials`` is given).
    Used by the standalone ``--mode chain`` view."""
    built = _build_chains(snapshot, before_snap=before_snap, trials=trials)
    built.sort(key=lambda c: -c["merge_hits"])
    dedup = _dedup_by_signature(built)
    chosen = dedup[:limit] if not trials else dedup
    return {"chains": chosen, "n_total": len(dedup)}


def _sample_trial_ids(all_ids: list[str], n: int, *, seed: int = 1337) -> list[str]:
    """Pick a DETERMINISTIC random cross-section of ``min(n, len)`` trial ids.
    Deterministic across runs (fixed ``seed``); ``≤ n`` total ⇒ return all, in
    snapshot order. The sample itself is returned sorted for a stable, scannable
    section order."""
    if len(all_ids) <= n:
        return list(all_ids)
    rng = random.Random(seed)
    return sorted(rng.sample(list(all_ids), n))


def extract_trial_chains(
    snapshot: dict, *, sample: int = 20, seed: int = 1337,
    trials: list[str] | None = None,
) -> dict:
    """AFTER-ONLY, PER-TRIAL chain selection for the Chain view: sample up to
    ``sample`` trials (deterministic cross-section) from
    ``snapshot.trial_subgraphs`` and return each trial's own chains grouped under a
    section header. No before/after split — the assembled (merged) graph only.

    Returns ``{"trials": [{trial, title, indication, n_chains, chains:[...]}, ...],
    "n_trials_total": int, "n_sampled": int}``. Each trial's ``chains`` are the
    full set of rendered-chain dicts (deduped within the trial by backbone
    signature) that ``_shared_svg`` consumes. ``trials`` (if given) pins the exact
    NCT list and bypasses sampling — for reproducing a specific case.
    """
    graph = snapshot.get("graph", snapshot)
    node_index = {n["id"]: n for n in graph.get("nodes", [])}
    ts = snapshot.get("trial_subgraphs", {})
    all_ids = [tid for tid in ts if ts[tid].get("chains")]
    if trials:
        chosen_ids = [tid for tid in trials if tid in ts]
    else:
        chosen_ids = _sample_trial_ids(all_ids, sample, seed=seed)

    # Build all rendered chains once, then bucket by trial (after-only ⇒ no premerge).
    by_trial: dict[str, list[dict]] = {}
    for c in _build_chains(snapshot, before_snap=None, trials=chosen_ids):
        by_trial.setdefault(c["trial"], []).append(c)

    def _title(tid: str) -> str | None:
        meta = (ts.get(tid, {}) or {}).get("metadata") or {}
        if meta.get("title"):
            return meta["title"]
        node = node_index.get(tid) or {}
        return node.get("name")

    def _indication(tid: str, chains: list[dict]) -> str | None:
        # Prefer the trial's first chain indication node name; fall back to the
        # TrialNode's first listed condition. Disease-agnostic — works for any area.
        for c in chains:
            ind = c["roles"].get("indication", {}).get("after", {})
            if ind.get("name"):
                return ind["name"]
        conds = ((node_index.get(tid) or {}).get("metadata") or {}).get("conditions") or []
        return conds[0] if conds else None

    sections = []
    for tid in chosen_ids:
        chains = _dedup_by_signature(by_trial.get(tid, []))
        sections.append({"trial": tid, "title": _title(tid),
                         "indication": _indication(tid, chains),
                         "n_chains": len(chains), "chains": chains})
    return {"trials": sections, "n_trials_total": len(all_ids),
            "n_sampled": len(sections)}


# geometry (px)
_GUT, _COLW, _BOXW, _BOXH = 132, 250, 188, 50
_ROWSP = 66   # converged (after) layout: vertical spacing between stacked nodes
_ENDP_DY, _BONE_DY, _AE_DY, _BAND_H = 6, 92, 168, 224   # per-row (before) layout
_ROW_COL = {"compound": 0, "target": 1, "mechanism": 2,
            "biology": 3, "indication": 4, "population": 5}  # before: endpoint is a satellite


def _colx(i: int) -> float:
    return _GUT + i * _COLW


def _box(cx, cy, fill, stroke, sw, name, sub, key, w=_BOXW, h=_BOXH):
    label = name if len(name) <= 24 else name[:23] + "…"
    return (f'<g class="box" data-key="{key}">'
            f'<rect x="{cx:.0f}" y="{cy:.0f}" width="{w:.0f}" height="{h:.0f}" rx="9" '
            f'fill="{fill}" fill-opacity="0.9" stroke="{stroke}" stroke-width="{sw}"/>'
            f'<text x="{cx + w/2:.0f}" y="{cy + h/2 - 3:.0f}" class="bx">{html.escape(label)}</text>'
            f'<text x="{cx + w/2:.0f}" y="{cy + h/2 + 12:.0f}" class="bxs">{html.escape(sub)}</text></g>')


def _edge(x1, y1, x2, y2, et, ep, dashed=False):
    col = EDGE_COLOR.get(et, "#7f8a9b")
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dash = ' stroke-dasharray="4 4"' if dashed else ''
    pct = f'{ep*100:.0f}%' if ep is not None else ''
    return (f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{col}" '
            f'stroke-width="2"{dash} marker-end="url(#ah-{et})"/>'
            + (f'<text x="{mx:.0f}" y="{my-6:.0f}" class="ep">{pct}</text>' if pct else '')
            + f'<text x="{mx:.0f}" y="{my+12:.0f}" class="et">{et}</text>')


def _rows_svg(chains, which, y0, show_ep):
    """BEFORE-merge layout: one ROW per chain (per-trial instance). Columns by node
    type, endpoint as a satellite above, adverse events below compound/target. It
    sprawls — each trial keeps its own instances; the contrast with the converged
    after-block IS the merge."""
    parts, details = [], {}
    for r, _f, _t, hdr, _c in ROLES:
        if r == "endpoint":
            continue
        parts.append(f'<text x="{_colx(_ROW_COL[r]) + _BOXW/2:.0f}" y="{y0 - 12:.0f}" class="col">{hdr}</text>')
    for ridx, ch in enumerate(chains):
        top = y0 + ridx * _BAND_H
        bone_y = top + _BONE_DY
        rlx = _GUT - 12
        parts.append(f'<text x="{rlx:.0f}" y="{bone_y + _BOXH/2 - 5:.0f}" class="rl">{html.escape(ch["trial"])}</text>')
        if ch.get("arm"):
            parts.append(f'<text x="{rlx:.0f}" y="{bone_y + _BOXH/2 + 8:.0f}" class="rlo">arm: {html.escape(str(ch["arm"]))}</text>')
        roles = ch["roles"]

        def pos(role):
            if role == "endpoint":
                return _colx(3) + _COLW * 0.52, top + _ENDP_DY
            return _colx(_ROW_COL[role]), bone_y

        for role, rd in roles.items():
            cx, cy = pos(role); side = rd[which]; fl = rd["after"]["flags"]
            stroke = ("#ff5252" if fl["cross_id"] else "#ffd54f" if fl["cross_trial"]
                      else "#ffb74d" if fl["island"] else "#20242e")
            sw = 3 if (fl["cross_id"] or fl["cross_trial"] or fl["island"]) else 1.2
            key = f'{which}:{ridx}:{role}'
            details[key] = {"col": next(h for _r, _f, _t, h, _c in ROLES if _r == role),
                            "type": ROLE_TYPE[role], "id": side["id"], "name": side["name"],
                            "desc": rd["after"].get("desc"), "flags": fl,
                            "fields": rd["after"].get("fields")}
            mark = " ~" if (which == "before" and side.get("changed")) else ""
            parts.append(_box(cx, cy, PALETTE.get(ROLE_TYPE[role], "#777"), stroke, sw,
                              (side["name"] or side["id"]) + mark, role.upper(), key,
                              h=(40 if role == "endpoint" else _BOXH)))
        for e in ch["edges"]:
            frm, to = e["frm"], e["to"]
            if frm not in roles or to not in roles:
                continue
            fx, fy = pos(frm); tx, ty = pos(to); ep = e["ep"] if show_ep else None
            if e["type"] == "reflects_biology":
                parts.append(_edge(fx + _BOXW/2, fy, tx + 75, ty + 40, e["type"], ep))
            elif e["type"] == "endpoint_captures":
                parts.append(_edge(fx + 75, fy + 40, tx + _BOXW/2, ty, e["type"], ep))
            else:
                parts.append(_edge(fx + _BOXW, fy + _BOXH/2, tx, ty + _BOXH/2, e["type"], ep))
        seen_ae = {}
        for ae in ch["aes"]:
            if ae["role"] not in roles:
                continue
            slot = seen_ae.get(ae["role"], 0); seen_ae[ae["role"]] = slot + 1
            if slot >= 2:
                continue
            aex = _colx(_ROW_COL[ae["role"]]) + slot * 96
            aey = top + _AE_DY
            key = f'{which}:{ridx}:ae:{ae["role"]}:{slot}'
            details[key] = {"col": "ADVERSE EVENT", "type": "AdverseEventNode",
                            "id": ae["name"], "name": ae["name"], "desc": None,
                            "flags": {"merged_from": [], "used_by": []},
                            "fields": ae.get("fields")}
            sx, sy = pos(ae["role"])
            parts.append(_edge(sx + 40, sy + _BOXH, aex + 45, aey, ae["type"], ae["ep"] if show_ep else None, dashed=True))
            parts.append(_box(aex, aey, PALETTE["AdverseEventNode"], "#20242e", 1.2,
                              ae["name"], ae["type"], key, w=90, h=38))
    return "\n".join(parts), details, y0 + len(chains) * _BAND_H


def _shared_svg(chains, which, y0, show_ep, *, key_prefix: str = ""):
    """AFTER-merge layout: each UNIQUE node id drawn ONCE per column, so chains
    through a shared (merged) node converge on one box with edges fanning in.
    Adverse events are omitted here (they're shown per-compound in the before
    block) to keep the convergence readable. ``key_prefix`` namespaces the
    click-detail keys so independent sections (e.g. per-trial blocks that may share
    a merged node id) don't collide."""
    parts, details = [], {}
    col_nodes = {c: {} for c in range(N_COLS)}
    edges = {}
    for ch in chains:
        roles = ch["roles"]
        for r, rd in roles.items():
            col_nodes[ROLE_COL[r]].setdefault(rd[which]["id"], {
                "name": rd[which]["name"], "role": r,
                "desc": rd["after"].get("desc"), "flags": rd["after"]["flags"],
                "fields": rd["after"].get("fields")})
        for e in ch["edges"]:
            if e["frm"] in roles and e["to"] in roles:
                fid, tid = roles[e["frm"]][which]["id"], roles[e["to"]][which]["id"]
                if fid != tid:
                    edges.setdefault((fid, tid), {"etype": e["type"], "ep": e["ep"]})
    for _r, _f, _t, hdr, col in ROLES:
        parts.append(f'<text x="{_colx(col) + _BOXW/2:.0f}" y="{y0 - 12:.0f}" class="col">{hdr}</text>')
    pos, maxslot = {}, 0
    for col in range(N_COLS):
        for slot, nid in enumerate(col_nodes[col]):
            pos[nid] = (_colx(col), y0 + slot * _ROWSP)
            maxslot = max(maxslot, slot)
    for (fid, tid), e in edges.items():
        if fid in pos and tid in pos:
            fx, fy = pos[fid]; tx, ty = pos[tid]
            parts.append(_edge(fx + _BOXW, fy + _BOXH/2, tx, ty + _BOXH/2,
                               e["etype"], e["ep"] if show_ep else None))
    for col in range(N_COLS):
        for nid, entry in col_nodes[col].items():
            cx, cy = pos[nid]; fl = entry["flags"]
            stroke = ("#ff5252" if fl["cross_id"] else "#ffd54f" if fl["cross_trial"]
                      else "#ffb74d" if fl["island"] else "#20242e")
            sw = 3 if (fl["cross_id"] or fl["cross_trial"] or fl["island"]) else 1.2
            key = f'{key_prefix}{which}:{nid}'
            details[key] = {"col": next(h for _r, _f, _t, h, _c in ROLES if _r == entry["role"]),
                            "type": ROLE_TYPE[entry["role"]], "id": nid, "name": entry["name"],
                            "desc": entry["desc"], "flags": fl, "fields": entry.get("fields")}
            parts.append(_box(cx, cy, PALETTE.get(ROLE_TYPE[entry["role"]], "#777"), stroke, sw,
                              entry["name"] or nid, entry["role"].upper(), key))
    return "\n".join(parts), details, y0 + (maxslot + 1) * _ROWSP + 8


def render_chain_html(title: str, data: dict, *, after_only: bool = False) -> str:
    """Render the chain view. ``after_only`` drops the BEFORE-MERGE block and
    shows only the assembled (after-merge) graph — the clean product view; the
    before/after debug stacking is the default (set by the ``--after-only``
    toggle)."""
    chains = data["chains"]
    width = int(_colx(N_COLS - 1) + _BOXW + 80)
    markers = "".join(
        f'<marker id="ah-{et}" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{c}"/></marker>' for et, c in EDGE_COLOR.items())
    if after_only:
        y_after = 96
        svg_a, det_a, bot_a = _shared_svg(chains, "after", y_after, show_ep=True)
        height = int(bot_a + 30)
        body = (f'<text x="20" y="{y_after-44:.0f}" class="blk">'
                f'▼ ASSEMBLED GRAPH (after merge)</text>{svg_a}')
        details = det_a
        meta_extra = "assembled (after-merge) view"
    else:
        y_before = 96
        svg_b, det_b, bot_b = _rows_svg(chains, "before", y_before, show_ep=False)
        y_after = bot_b + 86
        svg_a, det_a, bot_a = _shared_svg(chains, "after", y_after, show_ep=True)
        height = int(bot_a + 30)
        body = (f'<text x="20" y="{y_before-44:.0f}" class="blk">▼ BEFORE MERGE — per-trial instances</text>{svg_b}'
                f'<line x1="0" y1="{bot_b+40:.0f}" x2="{width}" y2="{bot_b+40:.0f}" class="div"/>'
                f'<text x="20" y="{y_after-44:.0f}" class="blk">▼ AFTER MERGE — assembled graph</text>{svg_a}')
        details = {**det_b, **det_a}
        meta_extra = ("red=cross-id fusion · gold=cross-trial merge · "
                      "orange=island · ~=changed by merge")
    svg = (f'<svg id="svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><defs>{markers}</defs>{body}</svg>')
    blob = json.dumps(details, default=str).replace("</", "<\\/")
    return (_CHAIN_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__META__", html.escape(
                f'{len(chains)} of {data["n_total"]} chains · {meta_extra}'))
            .replace("__SVG__", svg).replace("__DETAILS__", blob))


# per-trial section spacing (px): header band above each block + gap below it
_SEC_HEAD, _SEC_GAP = 64, 54


def render_trial_chain_html(title: str, data: dict) -> str:
    """Render the AFTER-ONLY, PER-TRIAL chain view: one assembled-graph
    (``_shared_svg``) block per sampled trial, stacked vertically and separated by
    a labeled section header (NCT id + title/indication). No before/after split.

    Consumes ``extract_trial_chains`` output (``data["trials"]`` = the per-trial
    sections). Reuses ``_CHAIN_TEMPLATE`` (same click-to-inspect panel + styling);
    each section's click-detail keys are namespaced by NCT so a node merged across
    trials never collides between sections."""
    sections = data.get("trials", [])
    width = int(_colx(N_COLS - 1) + _BOXW + 80)
    markers = "".join(
        f'<marker id="ah-{et}" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{c}"/></marker>' for et, c in EDGE_COLOR.items())

    body_parts, details, y = [], {}, 0.0
    for i, sec in enumerate(sections):
        chains = sec.get("chains") or []
        if i > 0:
            body_parts.append(f'<line x1="0" y1="{y:.0f}" x2="{width}" y2="{y:.0f}" class="div"/>')
        y += _SEC_HEAD
        # section header: NCT + (title · indication) sub-line
        sub_bits = [b for b in (sec.get("title"), sec.get("indication")) if b]
        sub = " · ".join(sub_bits)
        if len(sub) > 110:
            sub = sub[:109] + "…"
        body_parts.append(
            f'<text x="20" y="{y-30:.0f}" class="blk">▼ {html.escape(sec["trial"])}'
            f'  <tspan class="seccount">({sec.get("n_chains", len(chains))} chain'
            f'{"" if sec.get("n_chains") == 1 else "s"})</tspan></text>')
        if sub:
            body_parts.append(f'<text x="20" y="{y-12:.0f}" class="secsub">{html.escape(sub)}</text>')
        if not chains:
            body_parts.append(f'<text x="{_GUT:.0f}" y="{y+18:.0f}" class="secsub">'
                              f'(no chains)</text>')
            y += _SEC_GAP
            continue
        svg_s, det_s, bot_s = _shared_svg(
            chains, "after", y + 24, show_ep=True, key_prefix=f'{sec["trial"]}::')
        body_parts.append(svg_s)
        details.update(det_s)
        y = bot_s + _SEC_GAP

    height = int(y + 20)
    body = "".join(body_parts) or '<text x="20" y="40" class="blk">no trials to display</text>'
    svg = (f'<svg id="svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><defs>{markers}</defs>{body}</svg>')
    blob = json.dumps(details, default=str).replace("</", "<\\/")
    meta = (f'{data.get("n_sampled", len(sections))} of {data.get("n_trials_total", len(sections))} '
            f'trials (after-merge; click a node)')
    return (_CHAIN_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__META__", html.escape(meta))
            .replace("__SVG__", svg).replace("__DETAILS__", blob))


_CHAIN_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>chains — __TITLE__</title><style>
  body{margin:0;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;background:#0d0f15;color:#e6e6e6;}
  #top{padding:9px 16px;background:#161a22;border-bottom:1px solid #262b36;position:sticky;top:0;z-index:3;}
  #top b{font-size:15px;} #top .meta{color:#8a93a3;margin-left:10px;font-size:12px;}
  #wrap{display:flex;height:calc(100vh - 46px);} #scroll{flex:1;overflow:auto;}
  #panel{width:380px;flex:none;overflow:auto;padding:14px;background:#161a22;border-left:1px solid #262b36;}
  .blk{fill:#cfd6e4;font-size:14px;font-weight:700;letter-spacing:.05em;}
  .seccount{fill:#6b7484;font-size:12px;font-weight:400;letter-spacing:0;}
  .secsub{fill:#8a93a3;font-size:11.5px;}
  .col{fill:#6b7484;font-size:12px;text-anchor:middle;font-weight:700;letter-spacing:.12em;}
  .rl{fill:#8a93a3;font-size:11px;text-anchor:end;font-family:ui-monospace,monospace;}
  .rlo{fill:#5b6472;font-size:10px;text-anchor:end;font-family:ui-monospace,monospace;}
  .bx{fill:#fff;font-size:11.5px;text-anchor:middle;font-weight:600;pointer-events:none;}
  .bxs{fill:#dfe5f0;font-size:8.5px;text-anchor:middle;opacity:.7;letter-spacing:.08em;pointer-events:none;}
  .ep{fill:#e6e6e6;font-size:11px;text-anchor:middle;font-weight:700;}
  .et{fill:#6b7484;font-size:8.5px;text-anchor:middle;font-family:ui-monospace,monospace;}
  .div{stroke:#262b36;stroke-width:1;stroke-dasharray:7 6;}
  .box{cursor:pointer;} .box:hover rect{fill-opacity:1;stroke:#fff;}
  pre{white-space:pre-wrap;word-break:break-word;background:#0a0c11;border:1px solid #262b36;border-radius:6px;padding:8px;font:12px/1.4 ui-monospace,monospace;}
  .kv{color:#8a93a3;} .kv b{color:#e6e6e6;} h3{margin:.2em 0 .5em;}
  .tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin:2px 4px 2px 0;}
  details summary{cursor:pointer;color:#7f9cff;}
</style></head><body>
<div id="top"><b>__TITLE__</b><span class="meta">__META__</span></div>
<div id="wrap"><div id="scroll">__SVG__</div>
<div id="panel"><h3>click a node box</h3><p class="kv">Each column is a node type; chains read left→right.
Compare a box between the <b>before</b> and <b>after</b> blocks to see what the merge did to it.</p></div></div>
<script>
const D=__DETAILS__,panel=document.getElementById('panel');
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
document.getElementById('svg').addEventListener('click',ev=>{
  const g=ev.target.closest('.box');if(!g)return;const d=D[g.dataset.key];if(!d)return;const f=d.flags||{};
  let h=`<h3>${esc(d.name)}</h3><p class="kv"><b>${esc(d.type.replace('Node',''))}</b> · ${esc(d.col)}</p><p class="kv">id <code>${esc(d.id)}</code></p>`;
  if(f.cross_id)h+=`<span class="tag" style="background:#ff5252;color:#000">⚡ cross-id fusion</span>`;
  if(f.cross_trial)h+=`<span class="tag" style="background:#ffd54f;color:#000">cross-trial merge</span>`;
  if(f.island)h+=`<span class="tag" style="background:#ffb74d;color:#000">⚠ content-address island</span>`;
  if(d.desc)h+=`<details open><summary>description</summary><pre>${esc(d.desc)}</pre></details>`;
  if(f.used_by&&f.used_by.length)h+=`<p class="kv">used by ${f.used_by.length} trial(s): ${f.used_by.map(esc).join(', ')}</p>`;
  if(f.merged_from&&f.merged_from.length)h+=`<details open><summary>⊕ merged from ${f.merged_from.length}${f.cross_id?' — DISTINCT processes':''}</summary><pre>`+f.merged_from.map(m=>esc(m.id)+(m.name&&m.name!=m.id?`   ${esc(m.name)}`:'')).join('\n')+`</pre></details>`;
  if(d.fields&&Object.keys(d.fields).length)h+=`<details open><summary>all fields — sanity check</summary><pre>${esc(JSON.stringify(d.fields,null,1))}</pre></details>`;
  panel.innerHTML=h;
});
</script></body></html>"""


# ── edge view ((s,t) = node descriptions; contributor provenance per edge) ──────

# Backbone edges, each carrying an (s,t) belief field. Per edge:
#   (chain src attr, chain tgt attr, edge_type). The EDGE-LEVEL (s,t) is its
# source/target NODE descriptions (a property of the two nodes, fixed per edge) —
# shown as the panel header. Mirrors ``field_prediction._BACKBONE`` /
# ``materialize_belief_field.EDGE_SPECS``.
_ST_BACKBONE = [
    ("compound_id", "target_id", "affects"),
    ("target_id", "mechanism_id", "modulates_via"),
    ("mechanism_id", "biology_id", "mechanism_affects"),
    ("biology_id", "indication_id", "biology_drives"),
    ("biology_id", "endpoint_id", "reflects_biology"),
    ("endpoint_id", "indication_id", "endpoint_captures"),
    ("subgroup_population_id", "indication_id", "responds_differently"),
]

# Per-CONTRIBUTOR (s,t): the edge-level (s,t) above is the two NODE descriptions
# (one per edge). But each contributing chain carries its OWN per-(arm, drug)
# free-text descriptions for the per-arm endpoints (mechanism / biology /
# population) — combo-correct (a paclitaxel chain in a combo arm reads "mitotic
# arrest", not a co-drug's "angiogenesis inhibition"). This maps each edge type to
# which side reads a chain description field vs. the fixed node description. A
# value of ``None`` means "use the node description" (a fixed-entity side, no
# per-trial variation); a string names the ``CausalChain`` attribute to read
# (falling back to the node description when the chain field is empty).
_CONTRIB_ST = {
    # edge_type: (source chain attr | None, target chain attr | None)
    "affects": (None, None),                                   # both node desc
    "modulates_via": (None, "mechanism_description"),          # t = chain mech
    "mechanism_affects": ("mechanism_description", "biology_description"),
    "biology_drives": ("biology_description", None),           # t = indication node
    "reflects_biology": ("biology_description", None),         # t = endpoint node
    "endpoint_captures": (None, None),                         # both node desc
    "responds_differently": ("population_description", None),  # s = chain pop
}


def build_edge_view_data(
    graph_path: str, *, annotations_dir: str = "data/annotations",
) -> dict:
    """Per backbone edge, the edge's ``(s,t)`` (= its node descriptions) plus the
    list of trials/arms/drugs whose chains traverse it — honest merge provenance.

    The ``(s,t)`` of an edge is a property of its two NODES, read straight from
    the graph: ``s_desc = node(source).description or .name``, ``t_desc =
    node(target).description or .name``. ONE (s,t) per edge — not per trial.

    A single merged backbone edge (e.g. one ``biology_drives`` edge) is traversed
    by MANY per-trial chains. We show those as a CONTRIBUTOR list of ``{nct,
    arm_id, drug}`` (drug = the chain's compound node name) — so you can SEE which
    drug fed each merged edge and spot a wrong one, WITHOUT the prior combo-arm
    mislabeling. The old path re-derived per-chain descriptions from per-arm A.0b
    annotations (``chain_descriptions_by_arm``), which stamped a combo arm's
    first-listed drug's description onto every chain in the arm (≈28% mis-attributed
    at n=50). Reading the chain's OWN per-drug nodes is combo-safe and graph-native
    — ``annotations_dir`` is retained for signature compatibility but unused.

    Returns ``{"edges": [...], "n_edges_with_pairings": int, "n_multi": int}``
    where each edge dict carries its source/target ids+names, the EDGE-LEVEL
    ``(s,t)`` (= node descriptions, panel header), edge type, belief summary, a
    ``contributors`` list of ``{nct, arm, drug, s_desc, t_desc}`` (each
    contributor's OWN per-chain (s,t)), and ``n_distinct_st`` = the number of
    distinct contributor (s,t) pairs (per-arm edges with genuine per-trial
    variation show >1; fixed-entity edges like ``affects`` show 1).
    """
    from collections import defaultdict

    from src.graph.store import GraphStore

    del annotations_dir  # no longer re-derives from annotations — reads the graph

    g = GraphStore()
    g.import_snapshot(graph_path)

    def _node(nid: str) -> dict:
        try:
            return g.get_node(nid)
        except KeyError:
            return {}

    def _ndesc(nid: str) -> str:
        nd = _node(nid)
        return (nd.get("description") or nd.get("name") or "").strip()

    def _nname(nid: str) -> str:
        return _node(nid).get("name") or nid

    # belief (E[p], n_evidence, ...) per (source, target, edge_type) from the graph
    belief_by_edge: dict[tuple, dict] = {}
    gd = g._graph  # noqa: SLF001
    for u, v, key, data in gd.edges(data=True, keys=True):
        belief_by_edge[(u, v, key)] = _belief_summary(data.get("belief"))

    # Accumulate contributors per MERGED edge. The edge-level (s,t) is the edge's
    # node descriptions (one per edge); each contributing chain is a (nct, arm,
    # drug) PLUS that chain's OWN per-(arm,drug) (s,t) computed via _CONTRIB_ST
    # (chain description field where the side is a per-arm endpoint, falling back
    # to the node description; node description for fixed-entity sides).
    acc: dict[tuple, list] = defaultdict(list)
    for nct, ts in g.trial_subgraphs.items():
        for ch in ts.chains:
            drug = _nname(ch.compound_id) if getattr(ch, "compound_id", None) else None
            for s_attr, t_attr, et in _ST_BACKBONE:
                s_id, t_id = getattr(ch, s_attr, None), getattr(ch, t_attr, None)
                if not s_id or not t_id or s_id == "UNKNOWN" or t_id == "UNKNOWN":
                    continue
                s_chain_attr, t_chain_attr = _CONTRIB_ST.get(et, (None, None))
                # per-contributor (s,t): chain field if this side names one and it's
                # non-empty, else the node description (fallback / fixed-entity side).
                cs = (getattr(ch, s_chain_attr, "") or "").strip() if s_chain_attr else ""
                ct = (getattr(ch, t_chain_attr, "") or "").strip() if t_chain_attr else ""
                c_s_desc = cs or _ndesc(s_id)
                c_t_desc = ct or _ndesc(t_id)
                acc[(s_id, t_id, et)].append(
                    {"nct": nct, "arm": ch.arm_id, "drug": drug or "UNKNOWN",
                     "s_desc": c_s_desc, "t_desc": c_t_desc})

    edges = []
    for (s_id, t_id, et), hits in acc.items():
        # distinct contributors: collapse repeats of the same
        # (nct, arm, drug, s_desc, t_desc) — keeping (s,t) in the signature so a
        # single arm contributing genuinely different per-chain (s,t) shows both.
        seen: set[tuple] = set()
        contributors = []
        for h in hits:
            sig = (h["nct"], h["arm"], h["drug"], h["s_desc"], h["t_desc"])
            if sig in seen:
                continue
            seen.add(sig)
            contributors.append({"nct": h["nct"], "arm": h["arm"], "drug": h["drug"],
                                 "s_desc": h["s_desc"], "t_desc": h["t_desc"]})
        contributors.sort(key=lambda c: (c["nct"], str(c["arm"]), c["drug"]))
        n_ncts = len({c["nct"] for c in contributors})
        # distinct (s,t) over CONTRIBUTOR values — restores the multiplicity the
        # user wanted, now combo-correctly (per-arm edges vary, affects/endpoint=1).
        n_distinct_st = len({(c["s_desc"], c["t_desc"]) for c in contributors})
        bs = belief_by_edge.get((s_id, t_id, et), {})
        edges.append({
            "etype": et, "source_id": s_id, "target_id": t_id,
            "source_name": _nname(s_id), "target_name": _nname(t_id),
            "s_desc": _ndesc(s_id), "t_desc": _ndesc(t_id),
            "label": f"{et}: {_nname(s_id)} → {_nname(t_id)}",
            "ep": bs.get("ep"), "n_evidence": bs.get("n_evidence", 0),
            "alpha": bs.get("alpha"), "beta": bs.get("beta"),
            "n_contributors": len(contributors), "n_ncts": n_ncts,
            "n_distinct_st": n_distinct_st,
            "contributors": contributors})

    # Sort alphabetically by label (the user's requirement); the UI also lets you
    # sort by contributor count — how many trials' drugs an edge aggregates.
    edges.sort(key=lambda e: e["label"].lower())
    n_multi = sum(1 for e in edges if e["n_contributors"] >= 2)
    return {"edges": edges, "n_edges_with_pairings": len(edges), "n_multi": n_multi}


def render_edge_view_html(title: str, data: dict) -> str:
    """Render the edge view: an alphabetical, hoverable, clickable list of
    backbone edges; clicking one shows the edge's ``(s,t)`` (= its source/target
    NODE descriptions, one per edge) plus the CONTRIBUTOR list — the
    ``{nct, arm, drug}`` of every chain that traverses the merged edge.
    Self-contained, file:// openable; data embedded via ``json.dumps``."""
    payload = {"title": title, "edge_color": EDGE_COLOR, "edges": data["edges"],
               "n_edges": data["n_edges_with_pairings"], "n_multi": data["n_multi"]}
    blob = json.dumps(payload, default=str).replace("</", "<\\/")
    return (_EDGE_VIEW_TEMPLATE.replace("__EDGE_DATA__", blob)
            .replace("__TITLE__", html.escape(title)))


_EDGE_VIEW_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>edge view — __TITLE__</title><style>
  body{margin:0;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;background:#0d0f15;color:#e6e6e6;}
  #top{padding:9px 16px;background:#161a22;border-bottom:1px solid #262b36;position:sticky;top:0;z-index:3;
       display:flex;gap:14px;align-items:center;flex-wrap:wrap;}
  #top b{font-size:15px;} #top .meta{color:#8a93a3;font-size:12px;}
  input,select,button{background:#0c0e13;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 8px;font:12px ui-sans-serif;}
  button{cursor:pointer;} button.on{background:#2a3550;border-color:#4f7cff;color:#cfe0ff;}
  #wrap{display:flex;height:calc(100vh - 50px);}
  #list{width:520px;flex:none;overflow:auto;border-right:1px solid #262b36;}
  #panel{flex:1;overflow:auto;padding:16px 20px;}
  .row{padding:7px 12px;border-bottom:1px solid #1c212b;cursor:pointer;display:flex;gap:8px;align-items:baseline;}
  .row:hover{background:#1b2030;} .row.sel{background:#222c44;}
  .row .et{font:11px ui-monospace,monospace;padding:1px 6px;border-radius:8px;color:#0a0c11;font-weight:700;flex:none;}
  .row .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .row .mult{flex:none;font:10px ui-monospace,monospace;color:#8a93a3;}
  .row .star{color:#ffd54f;font-weight:700;}
  .tip{position:fixed;pointer-events:none;z-index:9;max-width:340px;background:#0a0c11;border:1px solid #3a4150;
       border-radius:7px;padding:8px 10px;font-size:12px;box-shadow:0 6px 24px #0008;display:none;}
  .tip .k{color:#8a93a3;}
  h2{margin:.1em 0 .1em;font-size:18px;} h3{margin:1em 0 .4em;color:#cfd6e4;}
  .hdr{color:#8a93a3;font-size:12px;margin:.2em 0 1em;}
  .pair{border:1px solid #262b36;border-radius:8px;margin:0 0 12px;background:#11151d;}
  .pair .ph{display:flex;gap:10px;align-items:baseline;padding:7px 11px;background:#161b25;border-bottom:1px solid #262b36;border-radius:8px 8px 0 0;}
  .pair .nct{font:12px ui-monospace,monospace;color:#7f9cff;font-weight:700;}
  .pair .arm{font:11px ui-monospace,monospace;color:#6b7484;}
  .pair .st{display:grid;grid-template-columns:84px 1fr;gap:4px 10px;padding:9px 11px;}
  .pair .lbl{color:#8a93a3;font-size:11px;text-align:right;padding-top:1px;}
  .pair .nodename{font-weight:600;color:#e6e6e6;}
  .pair .desc{color:#cdd6e4;background:#0a0c11;border:1px solid #20242e;border-radius:5px;padding:5px 7px;font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;}
  .badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;margin-right:6px;}
  code{background:#0a0c11;border:1px solid #20242e;border-radius:4px;padding:0 4px;}
  .empty{color:#6b7484;padding:30px;text-align:center;}
  table.contrib{border-collapse:collapse;width:100%;margin-top:4px;font-size:12px;}
  table.contrib th{text-align:left;color:#8a93a3;font-weight:600;border-bottom:1px solid #262b36;padding:5px 10px;font-size:11px;letter-spacing:.04em;}
  table.contrib td{border-bottom:1px solid #1c212b;padding:5px 10px;vertical-align:top;}
  table.contrib tr:hover{background:#161b25;}
  table.contrib .nct{font:12px ui-monospace,monospace;color:#7f9cff;font-weight:700;white-space:nowrap;}
  table.contrib .arm{font:11px ui-monospace,monospace;color:#6b7484;word-break:break-all;}
  table.contrib .drug{color:#e6e6e6;}
  table.contrib .sd,table.contrib .td{color:#cdd6e4;font-size:11.5px;min-width:130px;word-break:break-word;}
</style></head><body>
<div id="top">
  <b>edge view</b><span class="meta">__TITLE__</span>
  <input id="q" placeholder="filter edges…" style="width:200px">
  <label class="meta">type
    <select id="type"><option value="">all</option></select></label>
  <button id="sortAZ" class="on">sort A→Z</button>
  <button id="sortMult">sort by contributor count</button>
  <label class="meta"><input type="checkbox" id="onlyMulti"> only ≥2 contributors</label>
  <span class="meta" id="stats"></span>
</div>
<div id="wrap">
  <div id="list"></div>
  <div id="panel"><div class="empty">Select an edge on the left to see its (s,t) and contributors.<br><br>
  The <b>edge-level (s,t)</b> is its source/target <b>node descriptions</b> — one per edge (panel header).<br>
  Each backbone edge in the merged graph is traversed by many per-trial chains;<br>
  this view lists the contributing <b>{nct, arm, drug}</b> WITH that chain's OWN <b>per-trial (s,t)</b><br>
  — combo-correct, so per-arm edges with genuine per-trial variation show several distinct (s,t),<br>
  while fixed-entity edges (affects / endpoint_captures) show one. A wrong drug or stray description is spottable here.</div></div>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA=__EDGE_DATA__, EC=DATA.edge_color, tip=document.getElementById('tip');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function colFor(et){return EC[et]||'#7f8a9b';}
// populate type filter
const types=[...new Set(DATA.edges.map(e=>e.etype))].sort();
const tsel=document.getElementById('type');types.forEach(t=>tsel.add(new Option(t,t)));
let sortMode='az', selIdx=-1;
const listEl=document.getElementById('list'), panel=document.getElementById('panel');
function current(){
  let es=DATA.edges.slice();
  const q=document.getElementById('q').value.trim().toLowerCase();
  const ty=tsel.value, only=document.getElementById('onlyMulti').checked;
  if(q)es=es.filter(e=>e.label.toLowerCase().includes(q));
  if(ty)es=es.filter(e=>e.etype===ty);
  if(only)es=es.filter(e=>e.n_contributors>=2);
  if(sortMode==='mult')es.sort((a,b)=>b.n_contributors-a.n_contributors||a.label.toLowerCase().localeCompare(b.label.toLowerCase()));
  else es.sort((a,b)=>a.label.toLowerCase().localeCompare(b.label.toLowerCase()));
  return es;
}
function renderList(){
  const es=current();
  document.getElementById('stats').textContent=`${es.length} edges · ${DATA.n_multi} with ≥2 contributors`;
  listEl.innerHTML=es.map((e,i)=>{
    const star=e.n_contributors>=2?'<span class="star">★</span> ':'';
    return `<div class="row" data-i="${i}">`+
      `<span class="et" style="background:${colFor(e.etype)}">${esc(e.etype)}</span>`+
      `<span class="nm">${star}${esc(e.source_name)} → ${esc(e.target_name)}</span>`+
      `<span class="mult">${e.n_contributors} (${e.n_ncts}t)</span></div>`;
  }).join('')||'<div class="empty">no edges match</div>';
  [...listEl.querySelectorAll('.row')].forEach(r=>{
    const e=es[+r.dataset.i];
    r.onclick=()=>{[...listEl.querySelectorAll('.row')].forEach(x=>x.classList.remove('sel'));r.classList.add('sel');showEdge(e);};
    r.onmousemove=ev=>{
      tip.style.display='block';tip.style.left=Math.min(ev.clientX+14,innerWidth-360)+'px';tip.style.top=(ev.clientY+14)+'px';
      tip.innerHTML=`<b>${esc(e.etype)}</b><br><span class="k">E[p]</span> ${e.ep==null?'—':e.ep} `+
        `&nbsp;<span class="k">n_ev</span> ${e.n_evidence}<br>`+
        `<span class="k">contributors</span> ${e.n_contributors} chain(s), ${e.n_ncts} trial(s)`+
        (e.n_distinct_st==null?'':`<br><span class="k">distinct (s,t)</span> ${e.n_distinct_st}`);
    };
    r.onmouseleave=()=>{tip.style.display='none';};
  });
}
function showEdge(e){
  tip.style.display='none';
  let h=`<h2><span class="badge" style="background:${colFor(e.etype)};color:#0a0c11">${esc(e.etype)}</span>`+
    `${esc(e.source_name)} &rarr; ${esc(e.target_name)}</h2>`+
    `<div class="hdr">merged edge <code>${esc(e.source_id)}</code> &rarr; <code>${esc(e.target_id)}</code> · `+
    `E[p]=<b>${e.ep==null?'—':e.ep}</b> (&alpha;=${e.alpha} &beta;=${e.beta}) · n_evidence=${e.n_evidence}</div>`;
  h+=`<h3>edge-level (s,t) — this edge's node descriptions</h3>`+
    `<div class="pair"><div class="st">`+
    `<div class="lbl">source (s)</div><div><span class="nodename">${esc(e.source_name)}</span><div class="desc">${esc(e.s_desc)}</div></div>`+
    `<div class="lbl">target (t)</div><div><span class="nodename">${esc(e.target_name)}</span><div class="desc">${esc(e.t_desc)}</div></div>`+
    `</div></div>`;
  const nd=e.n_distinct_st==null?'':` · ${e.n_distinct_st} distinct (s,t)`;
  h+=`<h3>${e.n_contributors} contributing chain(s) — from ${e.n_ncts} trial(s)${nd}</h3>`;
  if(e.n_contributors>=2)h+=`<p class="hdr">this merged edge aggregates evidence from ${e.n_ncts} trial(s)' drugs — each row shows that chain's OWN per-trial (s,t) (combo-correct: a co-drug's chain never borrows its neighbor's description), so a wrong drug or stray description feeding the edge is spottable.</p>`;
  h+=`<table class="contrib"><thead><tr><th>NCT</th><th>arm</th><th>drug</th><th>(s) source desc</th><th>(t) target desc</th></tr></thead><tbody>`+
    e.contributors.map(c=>
      `<tr><td class="nct">${esc(c.nct)}</td><td class="arm">${esc(c.arm)}</td><td class="drug">${esc(c.drug)}</td>`+
      `<td class="sd">${esc(c.s_desc)}</td><td class="td">${esc(c.t_desc)}</td></tr>`
    ).join('')+`</tbody></table>`;
  panel.innerHTML=h;panel.scrollTop=0;
}
document.getElementById('q').oninput=renderList;
tsel.onchange=renderList;
document.getElementById('onlyMulti').onchange=renderList;
document.getElementById('sortAZ').onclick=()=>{sortMode='az';document.getElementById('sortAZ').classList.add('on');document.getElementById('sortMult').classList.remove('on');renderList();};
document.getElementById('sortMult').onclick=()=>{sortMode='mult';document.getElementById('sortMult').classList.add('on');document.getElementById('sortAZ').classList.remove('on');renderList();};
renderList();
</script></body></html>"""


# ── explorer view (force-directed; product/overview) ───────────────────────────

def to_vis(snapshot: dict) -> dict:
    graph = snapshot.get("graph", snapshot)
    nodes, trials = [], set()
    for n in graph.get("nodes", []):
        ntype = n.get("node_type") or n.get("type") or "?"
        nid, name = n.get("id"), (n.get("name") or n.get("id"))
        meta = n.get("metadata") or {}
        mf = meta.get("merged_from") or []
        ont = n.get("ontology_id") or meta.get("ontology_id") or nid
        island = isinstance(ont, str) and ont.startswith("bio:")
        cross = len({_base(nid)} | {_base(m) for m in mf}) > 1
        short = name if len(str(name)) <= 30 else str(name)[:29] + "…"
        nodes.append({"id": nid, "label": str(short) + (f" ⊕{len(mf)}" if mf else "") + (" ⚡" if cross else ""),
                      "group": ntype, "title": f"{name} [{ntype}]", "from_trial": n.get("from_trial"),
                      "is_island": island, "cross_id": cross, "merged_from": mf,
                      "merged_from_named": [{"id": m, "name": NAME_CACHE.get(_base(m))} for m in mf], "data": n})
        if n.get("from_trial"):
            trials.add(n["from_trial"])
    edges = []
    for i, e in enumerate(graph.get("edges", graph.get("links", []))):
        bs = _belief_summary(e.get("belief"))
        et = e.get("key") or e.get("edge_type") or ""
        edges.append({"id": f"e{i}", "from": e.get("source"), "to": e.get("target"),
                      "label": (f"{bs['ep']:.2f}" if bs["n_evidence"] and bs["ep"] is not None else ""),
                      "etype": et, "title": f"{et} E[p]={bs['ep']} n_ev={bs['n_evidence']}",
                      "value": bs["n_evidence"] + 1, "belief": bs,
                      "data": {k: v for k, v in e.items() if k != "belief"}})
    return {"nodes": nodes, "edges": edges, "trials": sorted(t for t in trials if t)}


def render_explorer_html(title: str, datasets: list[tuple[str, dict]]) -> str:
    payload = {"title": title, "palette": PALETTE, "datasets": {l: d for l, d in datasets}}
    blob = json.dumps(payload, default=str).replace("</", "<\\/")
    return _EXPLORER_TEMPLATE.replace("__EROOM_DATA__", blob).replace("__TITLE__", title)


_EXPLORER_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>eroom graph — __TITLE__</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>body{margin:0;font:13px ui-sans-serif,system-ui;background:#0f1117;color:#e6e6e6;}
#bar{display:flex;gap:10px;align-items:center;padding:8px 12px;background:#171a21;border-bottom:1px solid #2a2f3a;}
select,button{background:#0c0e13;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 8px;}
#stage{display:flex;height:calc(100vh - 46px);} #net{flex:1;} #panel{width:400px;overflow:auto;padding:12px;background:#171a21;border-left:1px solid #2a2f3a;}
pre{white-space:pre-wrap;word-break:break-word;background:#0c0e13;border:1px solid #2a2f3a;border-radius:6px;padding:8px;font:12px ui-monospace,monospace;}
details summary{cursor:pointer;color:#7f9cff;}</style></head><body>
<div id="bar"><label>trial</label><select id="trial"><option value="">all</option></select>
<button id="fit">fit</button><button id="freeze">freeze</button><span id="stats"></span></div>
<div id="stage"><div id="net"></div><div id="panel">click a node/edge</div></div>
<script>
const DATA=__EROOM_DATA__,PALETTE=DATA.palette,ds=DATA.datasets[Object.keys(DATA.datasets)[0]];
const panel=document.getElementById('panel');function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const nodes=new vis.DataSet(),edges=new vis.DataSet();
const net=new vis.Network(document.getElementById('net'),{nodes,edges},{nodes:{shape:'dot',size:12,font:{color:'#e6e6e6',size:12,strokeWidth:3,strokeColor:'#0f1117'}},edges:{arrows:'to',color:{color:'#5b6270'},font:{color:'#9aa0aa',size:10},scaling:{min:1,max:5}},groups:Object.fromEntries(Object.entries(PALETTE).map(([t,c])=>[t,{color:{background:c,border:c}}])),physics:{stabilization:{iterations:200},barnesHut:{springLength:130}}});
function draw(){const trial=document.getElementById('trial').value,keep=new Set();
const vn=ds.nodes.filter(n=>!trial||n.from_trial===trial||(n.merged_from||[]).some(m=>String(m).includes(trial))).map(n=>{keep.add(n.id);const o={id:n.id,label:n.label,group:n.group,title:n.title};if(n.cross_id||n.is_island){o.color={background:PALETTE[n.group]||'#777',border:n.cross_id?'#ff5252':'#ffb74d'};o.borderWidth=3;}return o;});
const ve=ds.edges.filter(e=>keep.has(e.from)&&keep.has(e.to));nodes.clear();edges.clear();nodes.add(vn);edges.add(ve);
document.getElementById('stats').textContent=`${vn.length} nodes / ${ve.length} edges`;}
net.on('selectNode',p=>{const n=ds.nodes.find(x=>x.id===p.nodes[0]);if(!n)return;let h=`<h3>${esc(n.data.name||n.id)}</h3><p>${esc(n.group)} · <code>${esc(n.id)}</code></p>`;if(n.merged_from_named&&n.merged_from_named.length)h+=`<details open><summary>⊕ merged ${n.merged_from_named.length}</summary><pre>`+n.merged_from_named.map(m=>esc(m.id)+(m.name?`  ${esc(m.name)}`:'')).join('\n')+`</pre></details>`;h+=`<details><summary>fields</summary><pre>${esc(JSON.stringify(n.data,null,1))}</pre></details>`;panel.innerHTML=h;});
net.on('selectEdge',p=>{if(p.nodes.length)return;const e=ds.edges.find(x=>x.id===p.edges[0]);if(!e)return;const b=e.belief;panel.innerHTML=`<h3>${esc(e.etype)}</h3><p>E[p]=<b>${b.ep}</b> α=${b.alpha} β=${b.beta} ev=${b.n_evidence}</p>`;});
const tsel=document.getElementById('trial');(ds.trials||[]).forEach(t=>tsel.add(new Option(t,t)));tsel.onchange=draw;
document.getElementById('fit').onclick=()=>net.fit({animation:true});document.getElementById('freeze').onclick=()=>net.setOptions({physics:false});
draw();net.once('stabilizationIterationsDone',()=>net.fit());
</script></body></html>"""


# ── graph view (product overview — ONE merged force-directed network) ──────────

# Types whose nodes are HIDDEN by default in the graph view so the causal backbone
# (compound→target→mechanism→biology→indication + populations) reads cleanly. The
# user re-enables any of them with the type-filter checkboxes.
#   - AdverseEventNode / EndpointNode: high-count "noise" leaves the owner asked to
#     suppress by default.
#   - TrialNode: every TrialNode is degree-0 in the assembled graph (trials are
#     stored as subgraphs, not wired into the backbone) — they'd be loose dots.
_GRAPH_HIDDEN_BY_DEFAULT = {"AdverseEventNode", "EndpointNode", "TrialNode"}


def _node_type(n: dict) -> str:
    """Node type, bucketing missing/blank types (e.g. bare ``vincristine`` id-only
    nodes that never got full node data) under ``Other`` so they still render with a
    color instead of vanishing."""
    return n.get("node_type") or n.get("type") or "Other"


def to_graph_view(snapshot: dict) -> dict:
    """Build the ONE-merged-graph payload for ``--mode graph``.

    The snapshot's nodes are ALREADY merged (each conceptual node appears once, with
    losers recorded in ``metadata.merged_from``), so we emit them verbatim — every
    node exactly once. A shared node (e.g. the biology node ``mitotic arrest``, or a
    disease-agnostic population like ``line_first``) therefore appears a SINGLE time
    with edges fanning in from every compound/mechanism/indication that uses it —
    which is the whole point of the view (compounding cross-trial knowledge).

    Returns ``{nodes, edges, types, hidden_by_default, n_nodes, n_edges}`` where each
    node carries ``{id,label,group,name,desc,degree,title,data}`` and each edge
    ``{id,from,to,etype,ep,label,title,value}``. Degree is computed over the emitted
    edge list so the click panel can report it. Disease-agnostic — no onco-specific
    assumptions; works for any indication mix (the multi-indication graph included).
    """
    graph = snapshot.get("graph", snapshot)
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", graph.get("links", []))
    idx = {n.get("id"): n for n in raw_nodes}

    # degree over the (deduped) edge set — endpoints must both be real nodes.
    from collections import Counter
    deg: Counter = Counter()

    edges = []
    for i, e in enumerate(raw_edges):
        src, tgt = e.get("source"), e.get("target")
        if src not in idx or tgt not in idx:
            continue
        et = e.get("key") or e.get("edge_type") or ""
        bs = _belief_summary(e.get("belief"))
        ep = bs["ep"] if bs["n_evidence"] and bs["ep"] is not None else None
        deg[src] += 1
        deg[tgt] += 1
        edges.append({
            "id": f"e{i}", "from": src, "to": tgt, "etype": et,
            "ep": ep, "alpha": bs["alpha"], "beta": bs["beta"],
            "n_evidence": bs["n_evidence"],
            "label": f"{ep:.2f}" if ep is not None else "",
            "title": f"{et}  E[p]={ep if ep is not None else '—'}  n_ev={bs['n_evidence']}",
            "value": bs["n_evidence"] + 1})

    nodes = []
    for n in raw_nodes:
        nid = n.get("id")
        ntype = _node_type(n)
        name = str(n.get("name") or nid)
        mf = (n.get("metadata") or {}).get("merged_from") or []
        d = deg.get(nid, 0)
        short = name if len(name) <= 30 else name[:29] + "…"
        nodes.append({
            "id": nid, "label": short, "group": ntype, "name": name,
            "desc": n.get("description"), "degree": d, "n_merged": len(mf),
            "title": f"{name}  ·  {ntype.replace('Node','')}  ·  deg {d}"
                     + (f"  ·  ⊕{len(mf)} merged" if mf else ""),
            "data": _display_fields(n)})

    types = sorted({nd["group"] for nd in nodes})
    present_hidden = [t for t in types if t in _GRAPH_HIDDEN_BY_DEFAULT]
    return {"nodes": nodes, "edges": edges, "types": types,
            "hidden_by_default": present_hidden,
            "n_nodes": len(nodes), "n_edges": len(edges)}


def render_graph_html(title: str, data: dict) -> str:
    """Render the product graph-overview HTML: ONE force-directed merged network
    (vis-network from CDN), type-colored with a legend, type-filter checkboxes,
    search-to-zoom, click→detail panel, a populations-emphasis toggle, cluster-by-
    type controls, and a freeze-after-stabilize layout with zoom/pan map controls.
    Self-contained + ``file://`` openable; all graph data is inlined."""
    payload = {"title": title, "palette": PALETTE, "edge_color": EDGE_COLOR,
               "nodes": data["nodes"], "edges": data["edges"], "types": data["types"],
               "hidden_by_default": data["hidden_by_default"],
               "n_nodes": data["n_nodes"], "n_edges": data["n_edges"]}
    blob = json.dumps(payload, default=str).replace("</", "<\\/")
    return (_GRAPH_TEMPLATE.replace("__GRAPH_DATA__", blob)
            .replace("__TITLE__", html.escape(title))
            .replace("__META__", html.escape(
                f'{data["n_nodes"]} nodes · {data["n_edges"]} edges · one merged graph')))


_GRAPH_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>eroom knowledge graph — __TITLE__</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{margin:0;height:100%;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;background:#0d0f15;color:#e6e6e6;overflow:hidden;}
  #top{display:flex;gap:14px;align-items:center;padding:8px 14px;background:#10141c;border-bottom:1px solid #262b36;}
  #top b{font-size:15px;} #top .meta{color:#8a93a3;font-size:12px;}
  #top input,#top select,#top button{background:#0c0e13;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:5px 9px;font:12px ui-sans-serif;}
  #top button{cursor:pointer;} #top button:hover{background:#161b25;}
  #top button.on{background:#2a3550;border-color:#4f7cff;color:#cfe0ff;}
  #stage{display:flex;height:calc(100vh - 47px);}
  #left{width:230px;flex:none;overflow:auto;padding:12px 12px 40px;background:#10141c;border-right:1px solid #262b36;}
  #netwrap{flex:1;position:relative;min-width:0;}
  #net{position:absolute;inset:0;}
  #panel{width:360px;flex:none;overflow:auto;padding:14px;background:#10141c;border-left:1px solid #262b36;}
  h3{margin:.2em 0 .5em;} h4{margin:14px 0 6px;color:#cfd6e4;font-size:12px;letter-spacing:.06em;text-transform:uppercase;}
  .leg{display:flex;align-items:center;gap:8px;padding:3px 2px;cursor:pointer;user-select:none;border-radius:5px;}
  .leg:hover{background:#161b25;} .leg.off{opacity:.38;}
  .leg .sw{width:13px;height:13px;border-radius:50%;flex:none;border:1px solid #0008;}
  .leg .nm{flex:1;font-size:12px;} .leg .ct{color:#6b7484;font-size:11px;font-family:ui-monospace,monospace;}
  .leg input{margin:0;}
  .ctl{display:flex;flex-direction:column;gap:7px;}
  .ctl .row{display:flex;gap:7px;align-items:center;flex-wrap:wrap;}
  .ctl button,.ctl select{background:#0c0e13;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:5px 8px;font:12px ui-sans-serif;cursor:pointer;}
  .ctl button:hover{background:#161b25;} .ctl button.on{background:#2a3550;border-color:#4f7cff;color:#cfe0ff;}
  .zoombar{display:flex;align-items:center;gap:6px;}
  .zoombar input[type=range]{flex:1;accent-color:#4f7cff;}
  .hint{color:#6b7484;font-size:11px;line-height:1.4;margin-top:4px;}
  pre{white-space:pre-wrap;word-break:break-word;background:#0a0c11;border:1px solid #262b36;border-radius:6px;padding:8px;font:12px/1.4 ui-monospace,monospace;}
  .kv{color:#8a93a3;} .kv b{color:#e6e6e6;} code{background:#0a0c11;border:1px solid #20242e;border-radius:4px;padding:0 4px;}
  details summary{cursor:pointer;color:#7f9cff;}
  .badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;margin:2px 4px 2px 0;}
  .mapctl{position:absolute;right:12px;bottom:12px;display:flex;flex-direction:column;gap:6px;z-index:4;}
  .mapctl button{width:34px;height:34px;background:#161b25cc;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:7px;font-size:17px;cursor:pointer;backdrop-filter:blur(3px);}
  .mapctl button:hover{background:#222c44cc;}
  .mapctl .vbar{height:120px;width:34px;display:flex;align-items:center;justify-content:center;background:#161b25cc;border:1px solid #2a2f3a;border-radius:7px;}
  .mapctl .vbar input{writing-mode:vertical-lr;direction:rtl;width:18px;height:104px;accent-color:#4f7cff;}
  #stat{position:absolute;left:12px;bottom:12px;z-index:4;color:#8a93a3;font:11px ui-monospace,monospace;background:#10141ccc;border:1px solid #262b36;border-radius:6px;padding:4px 8px;}
  #busy{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:5;color:#8a93a3;background:#0d0f15cc;font-size:14px;}
</style></head><body>
<div id="top">
  <b>__TITLE__</b><span class="meta">__META__</span>
  <input id="q" placeholder="search nodes…  (Enter = zoom)" style="width:230px">
  <button id="popHL" title="Dim everything except PopulationNodes and their responds_differently edges">★ populations</button>
</div>
<div id="stage">
  <div id="left">
    <h4>node types</h4>
    <div id="legend"></div>
    <div class="row" style="margin-top:8px;gap:8px;">
      <button id="allOn" style="font-size:11px;padding:3px 7px;">all</button>
      <button id="allOff" style="font-size:11px;padding:3px 7px;">none</button>
      <button id="backbone" style="font-size:11px;padding:3px 7px;" title="compound→target→mechanism→biology→indication + populations">backbone</button>
    </div>

    <h4>cluster by type</h4>
    <div class="ctl">
      <select id="clusterType"><option value="">choose a type…</option></select>
      <div class="row">
        <button id="clusterBtn">cluster</button>
        <button id="clusterAll" title="collapse every visible type into one super-node each">cluster all</button>
        <button id="unclusterAll">expand all</button>
      </div>
      <div class="hint">Collapse a node_type into one expandable super-node. Click a super-node to expand it.</div>
    </div>

    <h4>view</h4>
    <div class="ctl">
      <div class="zoombar"><button id="zout">−</button>
        <input type="range" id="zoom" min="10" max="300" value="100">
        <button id="zin">+</button></div>
      <div class="row">
        <button id="fit">fit to screen</button>
        <button id="restabilize" title="re-run physics, then freeze">re-layout</button>
      </div>
      <label class="kv" style="display:flex;gap:6px;align-items:center;font-size:12px;">
        <input type="checkbox" id="physTog"> physics live (unfreeze)</label>
      <div class="hint">Layout freezes after it settles. Trackpad: pinch = zoom, two-finger = pan; drag canvas = pan, drag a node to nudge.</div>
    </div>
  </div>

  <div id="netwrap">
    <div id="net"></div>
    <div id="busy">stabilizing layout…</div>
    <div id="stat"></div>
    <div class="mapctl">
      <button id="mZin" title="zoom in">+</button>
      <div class="vbar"><input type="range" id="mZoom" min="10" max="300" value="100"></div>
      <button id="mZout" title="zoom out">−</button>
      <button id="mFit" title="fit to screen">⤢</button>
    </div>
  </div>

  <div id="panel"><h3>click a node</h3>
    <p class="kv">One merged knowledge graph — every node appears <b>once</b>. Shared
    mechanism / biology / population nodes fan in from multiple compounds &amp; trials.
    Toggle node types on the left, search above, or click a node for details.</p>
    <p class="kv">★ <b>populations</b> highlights how a disease-agnostic population
    (e.g. <code>line_first</code>) is ONE node shared across many indications.</p>
  </div>
</div>
<script>
const DATA=__GRAPH_DATA__, PALETTE=DATA.palette, EC=DATA.edge_color;
const panel=document.getElementById('panel');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function colFor(t){return PALETTE[t]||'#7f8a9b';}

// ---- type filter state (default-hide the noisy types) ----
const hidden=new Set(DATA.hidden_by_default||[]);
const typeCounts={}; DATA.nodes.forEach(n=>typeCounts[n.group]=(typeCounts[n.group]||0)+1);
const BACKBONE_TYPES=new Set(['InterventionNode','CompoundNode','TargetNode','MechanismNode','BiologyNode','IndicationNode','PopulationNode','Other']);

// ---- vis datasets (full, immutable copies; visibility via hidden flag) ----
const allNodes=DATA.nodes.map(n=>({
  id:n.id,label:n.label,group:n.group,title:n.title,
  value:Math.max(1,n.degree),_name:n.name,_desc:n.desc,_deg:n.degree,
  _nmerged:n.n_merged,_data:n.data}));
const allEdges=DATA.edges.map(e=>({
  id:e.id,from:e.from,to:e.to,_etype:e.etype,_ep:e.ep,_title:e.title,
  color:{color:EC[e.etype]||'#3a4150',opacity:0.55},
  title:e.title,value:e.value}));
const nodeById={}; allNodes.forEach(n=>nodeById[n.id]=n);

const nodes=new vis.DataSet(), edges=new vis.DataSet();
const options={
  nodes:{shape:'dot',scaling:{min:6,max:34,label:{enabled:true,min:11,max:22}},
         font:{color:'#e6e6e6',size:12,strokeWidth:3,strokeColor:'#0d0f15'},
         borderWidth:1.5},
  edges:{arrows:{to:{enabled:true,scaleFactor:0.5}},smooth:{type:'continuous',roundness:0.2},
         color:{color:'#3a4150',opacity:0.55},font:{color:'#7f8a9b',size:9,strokeWidth:2,strokeColor:'#0d0f15'},
         scaling:{min:0.5,max:5}},
  groups:Object.fromEntries(Object.keys(PALETTE).map(t=>[t,{color:{background:PALETTE[t],border:'#0a0c11'}}])),
  interaction:{hover:true,tooltipDelay:120,navigationButtons:false,keyboard:false,
               zoomView:true,dragView:true,multiselect:false},
  physics:{stabilization:{enabled:true,iterations:300,updateInterval:30,fit:true},
           barnesHut:{gravitationalConstant:-9000,springLength:120,springConstant:0.035,
                      damping:0.5,avoidOverlap:0.15},
           minVelocity:0.75,solver:'barnesHut'}};
const net=new vis.Network(document.getElementById('net'),{nodes,edges},options);

// ---- compute the visible set from type filters + populations-only mode ----
let popOnly=false;
function popEdgeOk(e){return e._etype==='responds_differently';}
function rebuild(){
  // visible nodes: type not hidden; in pop-only mode, only PopulationNodes + their RD neighbors
  let vis_ids;
  if(popOnly){
    const keep=new Set();
    allNodes.forEach(n=>{if(n.group==='PopulationNode')keep.add(n.id);});
    allEdges.forEach(e=>{if(popEdgeOk(e)&&(keep.has(e.from)||keep.has(e.to))){keep.add(e.from);keep.add(e.to);}});
    vis_ids=keep;
  }else{
    vis_ids=new Set(allNodes.filter(n=>!hidden.has(n.group)).map(n=>n.id));
  }
  const vn=allNodes.filter(n=>vis_ids.has(n.id));
  let ve=allEdges.filter(e=>vis_ids.has(e.from)&&vis_ids.has(e.to));
  if(popOnly)ve=ve.filter(popEdgeOk);
  nodes.clear();edges.clear();
  // re-use baked coordinates where we have them so unchanged nodes don't jump;
  // new/never-positioned nodes get laid out by the fresh stabilization below.
  nodes.add(vn.map(n=>{const o={...n};if(typeof n.x==='number'){o.x=n.x;o.y=n.y;}return o;}));
  edges.add(ve.map(e=>({...e})));
  document.getElementById('stat').textContent=`${vn.length} / ${DATA.n_nodes} nodes · ${ve.length} / ${DATA.n_edges} edges`;
  if(typeof restabilize!=='undefined')restabilize();
}
// re-run a short stabilization for the current (filtered) node set, then re-freeze.
// Called after every rebuild so a newly-shown subset gets a clean layout but ends
// up just as static as the initial render.
let _firstBuild=true;
function restabilize(){
  if(_firstBuild){_firstBuild=false;return;} // initial layout handled by ctor stabilize
  const busy=document.getElementById('busy');
  busy.textContent='laying out…';busy.style.display='flex';
  net.setOptions({physics:true});net.stabilize();
}

// ---- legend (type-filter checkboxes + counts + swatches) ----
function renderLegend(){
  const el=document.getElementById('legend');
  el.innerHTML=DATA.types.map(t=>{
    const off=hidden.has(t)?' off':'';
    return `<label class="leg${off}" data-t="${esc(t)}">`+
      `<input type="checkbox" ${hidden.has(t)?'':'checked'}>`+
      `<span class="sw" style="background:${colFor(t)}"></span>`+
      `<span class="nm">${esc(t.replace('Node',''))}</span>`+
      `<span class="ct">${typeCounts[t]||0}</span></label>`;
  }).join('');
  [...el.querySelectorAll('.leg')].forEach(l=>{
    l.querySelector('input').onchange=()=>{
      const t=l.dataset.t;
      if(hidden.has(t))hidden.delete(t);else hidden.add(t);
      l.classList.toggle('off',hidden.has(t));
      popOnly=false;document.getElementById('popHL').classList.remove('on');
      clusters.clear();rebuild();
    };
  });
}
function setTypes(showSet){ // showSet=Set of types to SHOW (rest hidden)
  hidden.clear();DATA.types.forEach(t=>{if(!showSet.has(t))hidden.add(t);});
  popOnly=false;document.getElementById('popHL').classList.remove('on');
  clusters.clear();renderLegend();rebuild();
}
document.getElementById('allOn').onclick=()=>setTypes(new Set(DATA.types));
document.getElementById('allOff').onclick=()=>setTypes(new Set());
document.getElementById('backbone').onclick=()=>setTypes(new Set(DATA.types.filter(t=>BACKBONE_TYPES.has(t))));

// ---- populations emphasis ----
document.getElementById('popHL').onclick=()=>{
  popOnly=!popOnly;document.getElementById('popHL').classList.toggle('on',popOnly);
  clusters.clear();rebuild();
  if(popOnly)setTimeout(()=>net.fit({animation:true}),60);
};

// ---- cluster by node type ----
const clusters=new Set();
function clusterType(t){
  if(!t||clusters.has(t))return;
  net.cluster({
    joinCondition:o=>o.group===t,
    clusterNodeProperties:{
      id:'cluster:'+t,label:`${t.replace('Node','')}  (${typeCounts[t]||0})`,
      group:t,shape:'hexagon',borderWidth:3,
      color:{background:colFor(t),border:'#fff'},
      font:{size:16,color:'#fff'},_cluster:t,
      title:`clustered ${t} — click to expand`}});
  clusters.add(t);
}
function uncluster(t){const id='cluster:'+t;if(net.isCluster(id))net.openCluster(id);clusters.delete(t);}
function populateClusterSelect(){
  const s=document.getElementById('clusterType');
  s.innerHTML='<option value="">choose a type…</option>'+
    DATA.types.map(t=>`<option value="${esc(t)}">${esc(t.replace('Node',''))} (${typeCounts[t]||0})</option>`).join('');
}
document.getElementById('clusterBtn').onclick=()=>{const t=document.getElementById('clusterType').value;if(t)clusterType(t);};
document.getElementById('clusterAll').onclick=()=>DATA.types.filter(t=>!hidden.has(t)).forEach(clusterType);
document.getElementById('unclusterAll').onclick=()=>[...clusters].forEach(uncluster);
// click a cluster super-node → expand it
net.on('click',p=>{
  if(p.nodes.length===1){const id=p.nodes[0];if(typeof id==='string'&&net.isCluster(id)){net.openCluster(id);const t=id.slice('cluster:'.length);clusters.delete(t);return;}}
});

// ---- detail panel (click node / edge) ----
net.on('selectNode',p=>{
  const id=p.nodes[0];
  if(typeof id==='string'&&net.isCluster(id))return; // cluster handled by 'click'
  const n=nodeById[id];if(!n)return;
  let h=`<h3>${esc(n._name||n.id)}</h3>`+
    `<p class="kv"><b>${esc(n.group.replace('Node',''))}</b> · degree <b>${n._deg}</b></p>`+
    `<p class="kv">id <code>${esc(n.id)}</code></p>`;
  if(n.group==='PopulationNode')h+=`<span class="badge" style="background:#e8943a;color:#0a0c11">★ population (shared across indications)</span>`;
  if(n._nmerged)h+=`<span class="badge" style="background:#2a3550;color:#cfe0ff">⊕ ${n._nmerged} merged-in</span>`;
  if(n._desc)h+=`<details open><summary>description</summary><pre>${esc(n._desc)}</pre></details>`;
  // neighbors grouped by edge type (read straight off the full edge list)
  const nbr={};
  allEdges.forEach(e=>{
    if(e.from===id){(nbr[e._etype]=nbr[e._etype]||[]).push({d:'→',o:e.to,ep:e._ep});}
    else if(e.to===id){(nbr[e._etype]=nbr[e._etype]||[]).push({d:'←',o:e.from,ep:e._ep});}
  });
  const ets=Object.keys(nbr).sort();
  if(ets.length){
    h+=`<details open><summary>${n._deg} edge(s)</summary>`;
    ets.forEach(et=>{
      h+=`<div style="margin:6px 0 2px;"><span class="badge" style="background:${colFor(et)};color:#0a0c11">${esc(et)}</span></div>`;
      h+=nbr[et].slice(0,40).map(x=>{
        const o=nodeById[x.o]||{};
        const ep=x.ep==null?'':` <span class="kv">E[p]=${x.ep}</span>`;
        return `<div class="kv" style="padding:1px 0 1px 8px;">${x.d} ${esc(o._name||x.o)}${ep}</div>`;
      }).join('');
      if(nbr[et].length>40)h+=`<div class="kv" style="padding-left:8px;">… ${nbr[et].length-40} more</div>`;
    });
    h+=`</details>`;
  }
  if(n._data&&Object.keys(n._data).length)h+=`<details><summary>all fields</summary><pre>${esc(JSON.stringify(n._data,null,1))}</pre></details>`;
  panel.innerHTML=h;
});
net.on('selectEdge',p=>{
  if(p.nodes.length)return;
  const e=edges.get(p.edges[0]);if(!e)return;
  const s=nodeById[e.from]||{},t=nodeById[e.to]||{};
  panel.innerHTML=`<h3><span class="badge" style="background:${colFor(e._etype)};color:#0a0c11">${esc(e._etype)}</span></h3>`+
    `<p class="kv">${esc(s._name||e.from)} <b>→</b> ${esc(t._name||e.to)}</p>`+
    `<p class="kv">E[p] = <b>${e._ep==null?'—':e._ep}</b></p>`+
    `<p class="kv"><code>${esc(e.from)}</code><br>→ <code>${esc(e.to)}</code></p>`;
});

// ---- search → highlight + zoom ----
function _defColor(n){return {background:colFor(n.group),border:'#0a0c11'};}
function runSearch(zoom){
  const q=document.getElementById('q').value.trim().toLowerCase();
  if(!q){nodes.forEach(n=>nodes.update({id:n.id,borderWidth:1.5,color:_defColor(n)}));return;}
  const hits=[];
  nodes.forEach(n=>{
    const m=(n._name||'').toLowerCase().includes(q)||String(n.id).toLowerCase().includes(q);
    if(m)hits.push(n.id);
    nodes.update({id:n.id,borderWidth:m?5:1.5,
      color:m?{background:colFor(n.group),border:'#ffd54f'}:_defColor(n)});
  });
  if(zoom&&hits.length){net.selectNodes(hits);net.fit({nodes:hits,animation:true});
    if(hits.length===1){const n=nodeById[hits[0]];if(n)net.emit('selectNode',{nodes:[hits[0]]});}}
}
document.getElementById('q').addEventListener('input',()=>runSearch(false));
document.getElementById('q').addEventListener('keydown',ev=>{if(ev.key==='Enter')runSearch(true);});

// ---- zoom controls (slider + buttons, both sidebar + on-canvas) ----
function setZoom(scale){net.moveTo({scale:Math.max(0.1,Math.min(3,scale))});syncZoom();}
function nudgeZoom(f){setZoom(net.getScale()*f);}
function syncZoom(){const v=Math.round(net.getScale()*100);
  document.getElementById('zoom').value=Math.max(10,Math.min(300,v));
  document.getElementById('mZoom').value=Math.max(10,Math.min(300,v));}
['zoom','mZoom'].forEach(idd=>document.getElementById(idd).addEventListener('input',e=>setZoom(+e.target.value/100)));
document.getElementById('zin').onclick=()=>nudgeZoom(1.25);
document.getElementById('zout').onclick=()=>nudgeZoom(0.8);
document.getElementById('mZin').onclick=()=>nudgeZoom(1.25);
document.getElementById('mZout').onclick=()=>nudgeZoom(0.8);
document.getElementById('fit').onclick=()=>net.fit({animation:true});
document.getElementById('mFit').onclick=()=>net.fit({animation:true});
net.on('zoom',syncZoom);

// ---- freeze after stabilize: BAKE coordinates so the layout is 100% static ----
// storePositions() writes the simulated x/y back onto the nodes DataSet; with
// physics disabled the network then renders those fixed coordinates and never
// simulates again — the dense centre holds dead-still. Dragging a node moves only
// that node (no graph-wide re-sim); zoom/pan stay responsive.
const busy=document.getElementById('busy');
let baked=false;
function freeze(){
  net.storePositions();              // bake x/y onto the live nodes DataSet
  // persist coords onto allNodes/nodeById so they survive a filter rebuild
  const pos=net.getPositions();
  Object.keys(pos).forEach(id=>{const n=nodeById[id];if(n){n.x=pos[id].x;n.y=pos[id].y;}});
  net.setOptions({physics:false});   // ensure the solver is OFF globally
  document.getElementById('physTog').checked=false;
  baked=true;
}
net.on('stabilizationIterationsDone',()=>{busy.style.display='none';freeze();net.fit();syncZoom();});
// hard cap: if the stabilization event is slow, freeze anyway after a moment
setTimeout(()=>{if(busy.style.display!=='none'){busy.style.display='none';freeze();net.fit();syncZoom();}},9000);
document.getElementById('physTog').onchange=e=>{
  if(e.target.checked){baked=false;net.setOptions({physics:true});}
  else freeze();
};
document.getElementById('restabilize').onclick=()=>{
  busy.style.display='flex';busy.textContent='re-stabilizing…';baked=false;
  net.setOptions({physics:true});net.stabilize();};

// ---- init ----
renderLegend();populateClusterSelect();rebuild();
</script></body></html>"""


# ── layered view (left→right column-per-node-type DAG of the MERGED graph) ──────

# The causal backbone as ORDERED type columns (left→right). Each entry is
# (node_type, header label, accent color). The order is the causal reading of a
# chain: a compound binds a target, which acts via a mechanism, which affects a
# biology, which an endpoint captures, in an indication, that a population responds
# to. ENDPOINT sits between BIOLOGY and INDICATION (it's toggleable in the UI but
# present here). ADVERSE_EVENT is NOT a column — it's the dashed-red row below.
# Disease-agnostic: a type, never a specific disease/onco assumption.
_LAYERED_COLUMNS: list[tuple[str, str, str]] = [
    ("InterventionNode", "COMPOUND", "#5b6ef0"),
    ("TargetNode", "TARGET", "#16a39a"),
    ("MechanismNode", "MECHANISM", "#e0922f"),
    ("BiologyNode", "BIOLOGY", "#1aa6c4"),
    ("EndpointNode", "ENDPOINT", "#6b7484"),
    ("IndicationNode", "INDICATION", "#e0559b"),
    ("PopulationNode", "POPULATION", "#e8943a"),
]
_LAYERED_AE_TYPE = "AdverseEventNode"
_LAYERED_AE_COLOR = "#e6194B"
# CompoundNode is an alias of InterventionNode (older snapshots); map it onto the
# compound column so either spelling lands in column 0.
_LAYERED_TYPE_ALIAS = {"CompoundNode": "InterventionNode"}
# Backbone edge types that flow between adjacent (or near-adjacent) columns — the
# left→right causal arrows. ``responds_differently`` is stored Population→Indication
# (right→left in our column order); we keep its native direction and let the arrow
# point back to indication. AE edge types are handled separately (dashed red row).
_LAYERED_BACKBONE_EDGES = {
    "affects", "binds_to", "modulates_via", "mechanism_affects",
    "biology_drives", "reflects_biology", "endpoint_captures",
    "responds_differently",
}
_LAYERED_AE_EDGES = {"causes_ae", "target_associated_ae"}
# Edge types intentionally excluded from the layered backbone: within-type
# hierarchy / cross-links that would clutter a strict left→right DAG.
#   subtype_of (X→X), composed_of (compound→constituent), modulates_efficacy_of
#   (compound→compound/biology cross-link). They're still visible in --mode graph.

# Edge SEMANTICS → the node type each endpoint must be. Used to RECOVER a
# typeless id-only stub (a node that exists in the snapshot as ``{"id": "..."}``
# with no ``node_type`` — e.g. ``rituximab`` / ``fluorouracil`` appearing only
# inside a combo-regimen string + a ``modulates_efficacy_of`` cross-link, never as
# a standalone typed compound). The schema fixes the type of each edge endpoint,
# so we can place such a stub in the column its edge implies and DRAW the edge,
# instead of silently dropping it (which leaves a rendered AE node missing its
# ``causes_ae`` arrow). Keyed by edge_type → (source_type, target_type). ``None``
# means "no constraint" (don't recover from that side).
_LAYERED_EDGE_ENDPOINT_TYPE: dict[str, tuple[str | None, str | None]] = {
    "affects": ("InterventionNode", "TargetNode"),
    "binds_to": ("InterventionNode", "TargetNode"),
    "modulates_via": ("TargetNode", "MechanismNode"),
    "mechanism_affects": ("MechanismNode", "BiologyNode"),
    "biology_drives": (None, "IndicationNode"),       # source is biology OR target
    "reflects_biology": ("BiologyNode", "EndpointNode"),
    "endpoint_captures": ("EndpointNode", "IndicationNode"),
    "responds_differently": ("PopulationNode", "IndicationNode"),
    "causes_ae": ("InterventionNode", _LAYERED_AE_TYPE),
    "target_associated_ae": ("TargetNode", _LAYERED_AE_TYPE),
}


_EXTERNAL_ID_PREFIXES: tuple[str, ...] = (
    "ensg", "enst", "ensp", "r-hsa", "react", "chembl", "hgnc", "uniprot",
    "go:", "efo:", "mondo:", "chebi:", "hp:", "doid:", "uberon:", "ncit:",
)


def _looks_like_external_id(s: str) -> bool:
    """True when ``s`` is a real EXTERNAL ontology/database accession (ENSG…,
    R-HSA…, GO:…, CHEBI:…, EFO:…, MONDO:…, CHEMBL…) — genuinely secondary info
    worth a sub-line — and NOT a name-derived slug (``crohns_disease``,
    ``brain_and_central_nervous_system_tumor``) or a content hash (``bio:<sha>``).
    This is what keeps the gene/pathway id under a target/mechanism card while
    killing the 'crohns disease / crohns_disease' duplication the user saw."""
    s = str(s or "").strip()
    if not s or s.lower().startswith("bio:"):
        return False
    sl = s.lower()
    return ":" in s or any(sl.startswith(p) for p in _EXTERNAL_ID_PREFIXES)


def _layered_subline(node: dict, ntype: str) -> str:
    """The dim sub-line under a card: a real external ontology/database id
    (gene/pathway/disease accession) when the node has one, else ''. NEVER the
    display name in slug form (the doubling the user flagged), never a content
    hash, never a stray 'None'. Populations show nothing — their NAME already
    encodes the disease-agnostic axes. Disease-agnostic."""
    meta = node.get("metadata") or {}
    ont = str(node.get("ontology_id") or "")
    if ntype == "PopulationNode":
        # The card NAME already encodes the axes (display_name); a slug sub-line
        # just duplicates it. Show extra axes only when genuinely additive.
        name = str(node.get("name") or "")
        axes = meta.get("qualifier_axes") or meta.get("axes") or {}
        if isinstance(axes, dict) and axes:
            bits = " · ".join(f"{k}:{v}" for k, v in axes.items() if v)
            norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
            if bits and norm(bits) != norm(name):
                return bits
        return ""
    # target / mechanism / biology / indication / endpoint / intervention: show
    # the ontology id ONLY when it's a real external accession, never a name-slug.
    return ont if _looks_like_external_id(ont) else ""


def to_layered_view(snapshot: dict) -> dict:
    """Build the ``--mode layered`` payload: the WHOLE merged snapshot as a
    left→right column-per-node-type DAG.

    Every node is emitted EXACTLY ONCE (the snapshot is already merged), tagged with
    its column index (by node type). A shared node therefore appears a single time in
    its column and receives backbone arrows fanning in from every node that points at
    it — e.g. one ``DNA`` target node receiving ``affects`` from many compounds, or
    one ``enzyme inhibition`` mechanism receiving ``modulates_via`` from many targets.
    That convergence is the whole point of the view.

    Within each column, nodes are ordered by a barycentric/median heuristic (a node
    sits near the average rank of its left-neighbors), 2 sweeps, so edge crossings are
    reduced and the layout SETTLES deterministically (computed here in Python — no
    runaway client physics). x is fixed by column; y is the within-column slot.

    Returns ``{columns, nodes, edges, ae_nodes, ae_edges, n_nodes, n_edges,
    n_ae_nodes, n_ae_edges, geom}`` where each backbone/ae node carries
    ``{id,label,sub,type,col,row,degree,desc,data}`` and each edge
    ``{id,source,target,etype,ep,label}``. Disease-agnostic — no onco assumptions;
    works for any indication mix (the multi-indication graph included).
    """
    from collections import Counter, defaultdict

    graph = snapshot.get("graph", snapshot)
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", graph.get("links", []))

    col_of_type = {t: i for i, (t, _h, _c) in enumerate(_LAYERED_COLUMNS)}

    def _ntype(n: dict) -> str:
        t = n.get("node_type") or n.get("type") or "Other"
        return _LAYERED_TYPE_ALIAS.get(t, t)

    # index every node; classify into backbone-column nodes vs AE-row nodes vs other
    idx = {n.get("id"): n for n in raw_nodes}
    ntype_of = {n.get("id"): _ntype(n) for n in raw_nodes}

    def _mk_node(n: dict, ntype: str, col: int) -> dict:
        nid = n.get("id")
        name = str(n.get("name") or nid)
        return {"id": nid, "label": name, "sub": _layered_subline(n, ntype),
                "type": ntype, "col": col, "row": 0, "degree": 0,
                "n_merged": len((n.get("metadata") or {}).get("merged_from") or []),
                "desc": n.get("description"), "data": _display_fields(n)}

    # ── BUILD NODES FIRST so we know EXACTLY which ids are rendered. A node
    # renders if its type is a backbone column or AE. Bare id-only nodes (no
    # node_type / name — e.g. ``rituximab`` appearing only in a regimen string)
    # resolve to ``Other``; a separate recovery pass below promotes any such stub
    # that a layered edge references into the column its edge SEMANTICS imply, so
    # its edges still draw. Anything still unrendered after that is dropped from
    # edges too — otherwise cytoscape's constructor THROWS ("Can not create edge
    # with nonexistant source"), aborting the whole render (blank canvas, no
    # headers). This is the bug that blanked multi_500. ──
    col_nodes: dict[int, list[dict]] = defaultdict(list)
    nodes, ae_nodes = [], []
    col_ids: set = set()      # ids of nodes that landed in a backbone column
    ae_ids: set = set()       # ids of nodes that landed in the AE row
    for n in raw_nodes:
        nid = n.get("id")
        ntype = ntype_of.get(nid, "Other")
        if ntype in col_of_type:
            col = col_of_type[ntype]
            node = _mk_node(n, ntype, col)
            col_nodes[col].append(node)
            nodes.append(node)
            col_ids.add(nid)
        elif ntype == _LAYERED_AE_TYPE:
            ae_nodes.append(_mk_node(n, ntype, -1))
            ae_ids.add(nid)

    # ── RECOVER typeless id-only stubs referenced by a layered edge. A node that
    # exists in the snapshot but has no node_type (resolves to ``Other`` — e.g. a
    # drug that only ever appeared inside a combo-regimen string, like the bare
    # ``rituximab`` / ``fluorouracil`` stubs) would otherwise be dropped, silently
    # dropping every backbone/AE edge that touches it (a rendered AE node missing
    # its ``causes_ae`` arrow). Edge SEMANTICS fix each endpoint's type, so we can
    # place the stub in the column its edge implies and draw the edge. We recover
    # ONLY into a column (never the AE row — an AE node always carries a real
    # AdverseEventNode type) and only when the stub isn't already rendered. ──
    recovered_ids: set = set()
    for e in raw_edges:
        et = e.get("key") or e.get("edge_type") or ""
        spec = _LAYERED_EDGE_ENDPOINT_TYPE.get(et)
        if not spec:
            continue
        src, tgt = e.get("source"), e.get("target")
        for end, want in ((src, spec[0]), (tgt, spec[1])):
            if (want is None or want not in col_of_type      # only column recoveries
                    or end is None or end not in idx          # must be a real (if bare) node
                    or end in col_ids or end in ae_ids        # already rendered
                    or end in recovered_ids):                 # already recovered this pass
                continue
            col = col_of_type[want]
            node = _mk_node(idx[end], want, col)
            node["recovered_stub"] = True   # surfaced in the click panel
            col_nodes[col].append(node)
            nodes.append(node)
            col_ids.add(end)
            recovered_ids.add(end)

    deg: Counter = Counter()
    backbone_edges, ae_edges = [], []
    for i, e in enumerate(raw_edges):
        src, tgt = e.get("source"), e.get("target")
        et = e.get("key") or e.get("edge_type") or ""
        bs = _belief_summary(e.get("belief"))
        ep = bs["ep"] if bs["n_evidence"] and bs["ep"] is not None else None
        rec = {"id": f"le{i}", "source": src, "target": tgt, "etype": et,
               "ep": ep, "label": f"{ep * 100:.0f}%" if ep is not None else "",
               "n_evidence": bs["n_evidence"]}
        if et in _LAYERED_BACKBONE_EDGES:
            # both endpoints must be RENDERED column nodes (not just in idx)
            if src not in col_ids or tgt not in col_ids:
                continue
            backbone_edges.append(rec)
            deg[src] += 1
            deg[tgt] += 1
        elif et in _LAYERED_AE_EDGES:
            # AE edge: one endpoint is the AE node, the other its compound/target —
            # both must have been rendered.
            ok = (src in col_ids and tgt in ae_ids) or (src in ae_ids and tgt in col_ids)
            if not ok:
                continue
            ae_edges.append(rec)
            deg[src] += 1
            deg[tgt] += 1

    # fold the post-filter degree onto the rendered node dicts
    for node in nodes:
        node["degree"] = deg.get(node["id"], 0)
    for node in ae_nodes:
        node["degree"] = deg.get(node["id"], 0)

    # ── within-column ordering: barycentric/median sweeps to cut edge crossings ──
    # Build left/right adjacency over backbone columns only.
    left_nbrs: dict[str, list[str]] = defaultdict(list)
    right_nbrs: dict[str, list[str]] = defaultdict(list)
    for e in backbone_edges:
        s, t = e["source"], e["target"]
        cs, ct = ntype_of.get(s), ntype_of.get(t)
        if cs not in col_of_type or ct not in col_of_type:
            continue
        # orient by column index so "left" really is the lower column regardless of
        # the edge's stored direction (responds_differently is pop→indication).
        if col_of_type[cs] <= col_of_type[ct]:
            lo, hi = s, t
        else:
            lo, hi = t, s
        right_nbrs[lo].append(hi)
        left_nbrs[hi].append(lo)

    # seed each column row by current append order, then sweep
    row_of: dict[str, float] = {}
    for col in range(len(_LAYERED_COLUMNS)):
        for r, node in enumerate(col_nodes[col]):
            row_of[node["id"]] = float(r)

    def _median(nbr_rows: list[float], fallback: float) -> float:
        if not nbr_rows:
            return fallback
        s = sorted(nbr_rows)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    n_cols = len(_LAYERED_COLUMNS)
    for _sweep in range(2):
        # forward: order each column by the median row of its LEFT neighbors
        for col in range(1, n_cols):
            for node in col_nodes[col]:
                node["_key"] = _median([row_of[x] for x in left_nbrs[node["id"]]
                                        if x in row_of], row_of[node["id"]])
            col_nodes[col].sort(key=lambda nd: (nd["_key"], nd["label"].lower()))
            for r, node in enumerate(col_nodes[col]):
                row_of[node["id"]] = float(r)
        # backward: order each column by the median row of its RIGHT neighbors
        for col in range(n_cols - 2, -1, -1):
            for node in col_nodes[col]:
                node["_key"] = _median([row_of[x] for x in right_nbrs[node["id"]]
                                        if x in row_of], row_of[node["id"]])
            col_nodes[col].sort(key=lambda nd: (nd["_key"], nd["label"].lower()))
            for r, node in enumerate(col_nodes[col]):
                row_of[node["id"]] = float(r)

    for col in range(n_cols):
        for r, node in enumerate(col_nodes[col]):
            node["row"] = r
            node.pop("_key", None)

    # AE rows: order by the row of the compound/target they attach to (so the dashed
    # lines stay short), then bucket into a few rows below the band.
    ae_anchor: dict[str, float] = {}
    for e in ae_edges:
        # AE edges are <compound|target> → AE; anchor the AE at its source's row
        anchor_id = e["source"] if ntype_of.get(e["source"]) != _LAYERED_AE_TYPE else e["target"]
        ae_id = e["target"] if anchor_id == e["source"] else e["source"]
        ae_anchor.setdefault(ae_id, row_of.get(anchor_id, 0.0))
    ae_nodes.sort(key=lambda nd: (ae_anchor.get(nd["id"], 1e9), nd["label"].lower()))
    for r, node in enumerate(ae_nodes):
        node["row"] = r

    # geometry constants the client uses to place preset coordinates (kept here so
    # Python ordering and JS placement agree). Columns are COL_DX apart; rows ROW_DY.
    geom = {"col_dx": 340, "row_dy": 92, "card_w": 244, "card_h": 60,
            "top_pad": 90, "left_pad": 60, "ae_gap": 150, "ae_cols": 6}

    columns = [{"type": t, "header": h, "color": c, "index": i,
                "count": len(col_nodes[i])}
               for i, (t, h, c) in enumerate(_LAYERED_COLUMNS)]

    return {"columns": columns, "nodes": nodes, "edges": backbone_edges,
            "ae_nodes": ae_nodes, "ae_edges": ae_edges,
            "ae_type": _LAYERED_AE_TYPE, "ae_color": _LAYERED_AE_COLOR,
            "geom": geom, "edge_color": EDGE_COLOR,
            "n_nodes": len(nodes), "n_edges": len(backbone_edges),
            "n_ae_nodes": len(ae_nodes), "n_ae_edges": len(ae_edges)}


def render_layered_html(title: str, data: dict) -> str:
    """Render the layered causal-chain DAG to a self-contained HTML string
    (cytoscape.js from a CDN). Left→right type columns with headers, type-colored
    rounded-rectangle cards, E[p]%-labeled left→right edges, a dashed-red default-
    hidden adverse-event row, zoom/pan, click→detail panel, and column toggles.
    ``file://`` openable; all graph data inlined."""
    payload = {"title": title, "columns": data["columns"], "nodes": data["nodes"],
               "edges": data["edges"], "ae_nodes": data["ae_nodes"],
               "ae_edges": data["ae_edges"], "ae_type": data["ae_type"],
               "ae_color": data["ae_color"], "geom": data["geom"],
               "edge_color": data["edge_color"], "palette": PALETTE,
               "n_nodes": data["n_nodes"], "n_edges": data["n_edges"],
               "n_ae_nodes": data["n_ae_nodes"], "n_ae_edges": data["n_ae_edges"]}
    blob = json.dumps(payload, default=str).replace("</", "<\\/")
    meta = (f'{data["n_nodes"]} nodes · {data["n_edges"]} backbone edges · '
            f'{data["n_ae_nodes"]} adverse events (hidden) · one merged graph')
    return (_LAYERED_TEMPLATE.replace("__LAYERED_DATA__", blob)
            .replace("__TITLE__", html.escape(title))
            .replace("__META__", html.escape(meta)))


_LAYERED_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>eroom layered chains — __TITLE__</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
  html,body{margin:0;height:100%;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;background:#0d0f15;color:#e6e6e6;overflow:hidden;}
  #top{display:flex;gap:12px;align-items:center;padding:8px 14px;background:#10141c;border-bottom:1px solid #262b36;flex-wrap:wrap;}
  #top b{font-size:15px;} #top .meta{color:#8a93a3;font-size:12px;}
  #top button,#top label{background:#0c0e13;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:5px 9px;font:12px ui-sans-serif;}
  #top button{cursor:pointer;} #top button:hover{background:#161b25;}
  #top button.on{background:#2a3550;border-color:#4f7cff;color:#cfe0ff;}
  #top input[type=range]{accent-color:#4f7cff;vertical-align:middle;width:120px;}
  #top label{display:inline-flex;gap:6px;align-items:center;cursor:pointer;}
  #stage{display:flex;height:calc(100vh - 48px);}
  #cywrap{flex:1;position:relative;min-width:0;}
  #cy{position:absolute;inset:0;}
  #panel{width:360px;flex:none;overflow:auto;padding:14px;background:#10141c;border-left:1px solid #262b36;}
  h3{margin:.2em 0 .5em;} h4{margin:13px 0 6px;color:#cfd6e4;font-size:12px;letter-spacing:.06em;text-transform:uppercase;}
  pre{white-space:pre-wrap;word-break:break-word;background:#0a0c11;border:1px solid #262b36;border-radius:6px;padding:8px;font:12px/1.4 ui-monospace,monospace;}
  .kv{color:#8a93a3;} .kv b{color:#e6e6e6;} code{background:#0a0c11;border:1px solid #20242e;border-radius:4px;padding:0 4px;}
  details summary{cursor:pointer;color:#7f9cff;}
  .badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;margin:2px 4px 2px 0;}
  .mapctl{position:absolute;right:12px;bottom:12px;display:flex;flex-direction:column;gap:6px;z-index:4;}
  .mapctl button{width:34px;height:34px;background:#161b25cc;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:7px;font-size:17px;cursor:pointer;}
  .mapctl button:hover{background:#222c44cc;}
  #stat{position:absolute;left:12px;bottom:12px;z-index:4;color:#8a93a3;font:11px ui-monospace,monospace;background:#10141ccc;border:1px solid #262b36;border-radius:6px;padding:4px 8px;}
  #hdrs{position:absolute;left:0;top:0;right:0;height:0;pointer-events:none;z-index:3;}
  .colhdr{position:absolute;transform:translateX(-50%);text-align:center;color:#cfd6e4;font-size:12px;font-weight:700;letter-spacing:.12em;white-space:nowrap;}
  .colhdr .ct{display:block;color:#6b7484;font-size:10px;font-weight:400;letter-spacing:.04em;margin-top:2px;}
</style></head><body>
<div id="top">
  <b>__TITLE__</b><span class="meta">__META__</span>
  <button id="aeTog" title="show the adverse-event row (dashed red)">＋ adverse events</button>
  <button id="endpTog" class="on" title="show/hide the ENDPOINT column">endpoint col</button>
  <label title="hide low-evidence edges (E[p] only on thicker backbone edges)"><input type="checkbox" id="epTog" checked> edge E[p]%</label>
  <span class="meta">zoom <input type="range" id="zoom" min="5" max="200" value="40"></span>
  <button id="fit">fit</button>
</div>
<div id="stage">
  <div id="cywrap">
    <div id="cy"></div>
    <div id="hdrs"></div>
    <div id="stat"></div>
    <div class="mapctl">
      <button id="zin" title="zoom in">+</button>
      <button id="zout" title="zoom out">−</button>
      <button id="mFit" title="fit to screen">⤢</button>
    </div>
  </div>
  <div id="panel"><h3>click a node</h3>
    <p class="kv">Left→right causal columns of the <b>merged</b> knowledge graph — every
    node appears <b>once</b>. Shared target / mechanism / biology / population nodes
    receive arrows fanning in from every upstream node that uses them.</p>
    <p class="kv">Edges are labeled with <b>E[p]%</b> (the edge belief mean). The
    adverse-event row (dashed red) is hidden by default — toggle it above. Pinch or
    use the zoom slider; drag to pan.</p>
  </div>
</div>
<script>
const DATA=__LAYERED_DATA__, EC=DATA.edge_color, PAL=DATA.palette, G=DATA.geom;
const panel=document.getElementById('panel');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const COLCOL={}; DATA.columns.forEach(c=>COLCOL[c.index]=c.color);
const HDRS=DATA.columns;
const nodeById={}; [...DATA.nodes,...DATA.ae_nodes].forEach(n=>nodeById[n.id]=n);
function edgeColor(et){return EC[et]||'#7f8a9b';}

// ---- preset coordinates (Python already ordered rows to cut edge crossings) ----
function colX(col){return G.left_pad + col*G.col_dx;}
function rowY(row){return G.top_pad + row*G.row_dy;}
let maxRows=0; DATA.nodes.forEach(n=>{if(n.row>maxRows)maxRows=n.row;});
const aeBaseY=rowY(maxRows)+G.ae_gap+G.card_h;                 // AE row below the band
function aePos(n){const r=Math.floor(n.row/G.ae_cols), c=n.row%G.ae_cols;
  return {x:G.left_pad + c*(G.card_w+24), y:aeBaseY + r*(G.card_h+18)};}

const cyNodes=[], cyEdges=[];
DATA.nodes.forEach(n=>cyNodes.push({data:{id:n.id,label:n.label,sub:n.sub,type:n.type,
  col:n.col,deg:n.degree,kind:'node'},position:{x:colX(n.col),y:rowY(n.row)}}));
DATA.ae_nodes.forEach(n=>cyNodes.push({data:{id:n.id,label:n.label,sub:n.sub,type:n.type,
  col:-1,deg:n.degree,kind:'ae'},position:aePos(n),classes:'aehidden'}));
DATA.edges.forEach(e=>cyEdges.push({data:{id:e.id,source:e.source,target:e.target,
  etype:e.etype,ep:e.ep,label:e.label,kind:'backbone'}}));
DATA.ae_edges.forEach(e=>cyEdges.push({data:{id:e.id,source:e.source,target:e.target,
  etype:e.etype,ep:e.ep,label:e.label,kind:'ae'},classes:'aehidden'}));

// ---- semantic-zoom thresholds: overview = SHAPES + FLOWS only; detail on zoom ──
//   < NAME_Z         : no text at all (clean colored columns + arrows)
//   NAME_Z..SUB_Z    : node NAMES appear
//   >= SUB_Z         : sub-lines + E[p]% edge labels appear (on backbone edges only)
// IMPORTANT: cytoscape evaluates these style mappers DURING the cytoscape({...})
// constructor — before the `cy` binding exists. So the mappers must NOT touch `cy`
// (that throws "Cannot access 'cy' before initialization", which blanks the whole
// render). We read a plain `curZoom` mirror instead, updated on every zoom event.
const NAME_Z=0.30, SUB_Z=0.62, EDGE_Z=0.62;
let curZoom=0.1, showEp=true;
function nodeLabel(ele){
  if(curZoom<NAME_Z) return '';
  const l=ele.data('label'), name=l.length>30?l.slice(0,29)+'…':l;
  if(curZoom<SUB_Z) return name;
  const sub=ele.data('sub'); return sub?name+'\n'+sub:name;
}
function edgeLabel(ele){
  if(!showEp) return '';
  if(curZoom<EDGE_Z) return '';
  // keep dense columns readable: label only the more-evidenced backbone edges
  return (ele.data('ep')!=null && ele.data('ep')>=0.0) ? (ele.data('label')||'') : '';
}

const cy=cytoscape({
  container:document.getElementById('cy'),
  elements:{nodes:cyNodes,edges:cyEdges},
  layout:{name:'preset'},
  minZoom:0.04, maxZoom:2.5, wheelSensitivity:0.2,
  style:[
    {selector:'node[kind="node"]',style:{
      'shape':'round-rectangle','width':G.card_w,'height':G.card_h,
      'background-color':ele=>COLCOL[ele.data('col')]||PAL[ele.data('type')]||'#777',
      'background-opacity':0.92,'border-width':1.5,'border-color':'#0a0c11',
      'label':nodeLabel,'color':'#fff','font-size':12,'font-weight':600,
      'text-valign':'center','text-halign':'center','text-wrap':'wrap',
      'text-max-width':G.card_w-16,'line-height':1.25}},
    {selector:'node[kind="ae"]',style:{
      'shape':'round-rectangle','width':150,'height':40,
      'background-color':DATA.ae_color,'background-opacity':0.9,
      'border-width':1.2,'border-color':'#20242e',
      'label':nodeLabel,'color':'#fff','font-size':10.5,
      'text-valign':'center','text-halign':'center','text-wrap':'wrap',
      'text-max-width':138}},
    {selector:'edge[kind="backbone"]',style:{
      'width':ele=>1.3+Math.min(3.4,(ele.data('ep')||0.15)*3.2),
      'line-color':ele=>edgeColor(ele.data('etype')),
      'target-arrow-color':ele=>edgeColor(ele.data('etype')),
      'target-arrow-shape':'triangle','arrow-scale':0.85,
      'curve-style':'bezier','opacity':0.6,'label':edgeLabel,
      'font-size':9,'color':'#e6e6e6','text-background-color':'#0d0f15',
      'text-background-opacity':0.78,'text-background-padding':1.5,
      'text-rotation':'autorotate'}},
    {selector:'edge[kind="ae"]',style:{
      'width':1.1,'line-color':DATA.ae_color,'line-style':'dashed',
      'target-arrow-color':DATA.ae_color,'target-arrow-shape':'triangle',
      'arrow-scale':0.7,'curve-style':'bezier','opacity':0.5}},
    {selector:'.aehidden',style:{'display':'none'}},
    {selector:'.faded',style:{'opacity':0.1,'text-opacity':0}},
    {selector:'node.hl',style:{'border-color':'#ffd54f','border-width':5}},
  ],
});
window.cy=cy;   // expose for the browser console + headless render checks

// Re-evaluate the zoom-dependent label mappers as the user zooms (cytoscape caches
// mapper output, so refresh `curZoom` then nudge the style to reveal/hide names +
// E[p] smoothly).
curZoom=cy.zoom();
let _zt=null;
cy.on('zoom',()=>{ curZoom=cy.zoom(); if(_zt)cancelAnimationFrame(_zt);
  _zt=requestAnimationFrame(()=>{cy.style().update(); syncZoom();}); });

// ---- column headers (HTML overlay, tracked to the canvas pan/zoom) ----
const hdrEl=document.getElementById('hdrs');
hdrEl.innerHTML=HDRS.map(c=>`<div class="colhdr" data-col="${c.index}">`+
  `${esc(c.header)}<span class="ct">${c.count}</span></div>`).join('');
const hdrDivs=[...hdrEl.querySelectorAll('.colhdr')];
function placeHeaders(){
  const pan=cy.pan(), z=cy.zoom();
  hdrDivs.forEach(d=>{const col=+d.dataset.col;
    const x=colX(col)*z+pan.x;                    // model→screen
    d.style.left=x+'px'; d.style.top='8px';
    d.style.display=(endpointOn||HDRS[col].type!=='EndpointNode')?'block':'none';});
}
cy.on('render zoom pan',()=>{ if(!_hp)_hp=requestAnimationFrame(()=>{_hp=0;placeHeaders();}); });
let _hp=0;

// ---- toggles: adverse-event row, endpoint column, E[p] labels ----
let aeOn=false, endpointOn=true;
document.getElementById('aeTog').onclick=function(){
  aeOn=!aeOn; this.classList.toggle('on',aeOn);
  cy.batch(()=>cy.elements('.aehidden, [kind="ae"]').forEach(el=>{
    if(aeOn)el.removeClass('aehidden'); else el.addClass('aehidden');}));
};
const ENDP_COL=HDRS.findIndex(c=>c.type==='EndpointNode');
document.getElementById('endpTog').onclick=function(){
  endpointOn=!endpointOn; this.classList.toggle('on',endpointOn);
  cy.batch(()=>{
    cy.nodes(`[col=${ENDP_COL}]`).forEach(n=>endpointOn?n.removeClass('faded'):n.addClass('faded'));
    cy.edges().forEach(e=>{
      const sc=nodeById[e.data('source')], tc=nodeById[e.data('target')];
      const touches=(sc&&sc.col===ENDP_COL)||(tc&&tc.col===ENDP_COL);
      if(touches){endpointOn?e.removeClass('faded'):e.addClass('faded');}});
  });
  placeHeaders();
};
document.getElementById('epTog').onchange=function(){showEp=this.checked;cy.style().update();};

// ---- detail panel (click node) — id, name, type, degree, neighbors by edge type ──
cy.on('tap','node',evt=>{
  const n=evt.target, id=n.id(), nd=nodeById[id]; if(!nd)return;
  cy.nodes().removeClass('hl'); n.addClass('hl');
  let h=`<h3>${esc(nd.label)}</h3>`+
    `<p class="kv"><b>${esc(nd.type.replace('Node',''))}</b> · degree <b>${nd.degree}</b></p>`+
    `<p class="kv">id <code>${esc(nd.id)}</code></p>`;
  if(nd.sub)h+=`<p class="kv">${esc(nd.sub)}</p>`;
  if(nd.n_merged)h+=`<span class="badge" style="background:#2a3550;color:#cfe0ff">⊕ ${nd.n_merged} merged-in</span>`;
  if(nd.recovered_stub)h+=`<span class="badge" style="background:#5a3a10;color:#ffd9a8" title="id-only stub placed by edge semantics so its edges render">⚠ recovered stub</span>`;
  if(nd.desc)h+=`<details open><summary>description</summary><pre>${esc(nd.desc)}</pre></details>`;
  // neighbors grouped by edge type with E[p]
  const nbr={};
  cy.edges().forEach(e=>{const d=e.data();
    if(d.source===id){(nbr[d.etype]=nbr[d.etype]||[]).push({d:'→',o:d.target,ep:d.ep});}
    else if(d.target===id){(nbr[d.etype]=nbr[d.etype]||[]).push({d:'←',o:d.source,ep:d.ep});}});
  const ets=Object.keys(nbr).sort();
  if(ets.length){
    h+=`<details open><summary>${nd.degree} edge(s)</summary>`;
    ets.forEach(et=>{
      h+=`<div style="margin:6px 0 2px;"><span class="badge" style="background:${edgeColor(et)};color:#0a0c11">${esc(et)}</span></div>`;
      h+=nbr[et].slice(0,40).map(x=>{const o=nodeById[x.o]||{};
        const ep=x.ep==null?'':` <span class="kv">E[p]=${(x.ep*100).toFixed(0)}%</span>`;
        return `<div class="kv" style="padding:1px 0 1px 8px;">${x.d} ${esc(o.label||x.o)}${ep}</div>`;}).join('');
      if(nbr[et].length>40)h+=`<div class="kv" style="padding-left:8px;">… ${nbr[et].length-40} more</div>`;
    });
    h+=`</details>`;
  }
  if(nd.data&&Object.keys(nd.data).length)h+=`<details><summary>all fields</summary><pre>${esc(JSON.stringify(nd.data,null,1))}</pre></details>`;
  panel.innerHTML=h;
});
cy.on('tap',evt=>{if(evt.target===cy)cy.nodes().removeClass('hl');});

// ---- zoom / pan controls ----
function syncZoom(){const v=Math.round(cy.zoom()*100);
  const s=document.getElementById('zoom'); if(s)s.value=Math.max(5,Math.min(200,v));}
function zoomTo(scale){const ext=cy.extent();
  cy.zoom({level:Math.max(0.04,Math.min(2.5,scale)),
    renderedPosition:{x:cy.width()/2,y:cy.height()/2}});}
document.getElementById('zoom').oninput=e=>zoomTo(+e.target.value/100);
document.getElementById('zin').onclick=()=>zoomTo(cy.zoom()*1.3);
document.getElementById('zout').onclick=()=>zoomTo(cy.zoom()*0.77);
document.getElementById('fit').onclick=()=>cy.fit(cy.nodes(':visible'),40);
document.getElementById('mFit').onclick=()=>cy.fit(cy.nodes(':visible'),40);

document.getElementById('stat').textContent=
  `${DATA.n_nodes} nodes · ${DATA.n_edges} edges · ${DATA.n_ae_nodes} AE (hidden)`;

// ---- init: fit the backbone, but GUARANTEE a visible-on-load view ----
// At hundreds of nodes the columns get very tall, so fit() can drive zoom down to
// ~minZoom (0.04) where cards are sub-pixel and the canvas looks blank. So: fit
// first; if that lands below a readable floor, clamp zoom to READABLE_Z and pan to
// the TOP-LEFT (start of the COMPOUND column) so the user immediately sees real
// cards and can zoom out from there.
const READABLE_Z=0.32;
function ensureVisible(){
  const vis=cy.nodes('[kind="node"]:visible');
  if(vis.empty()){placeHeaders();syncZoom();return;}
  cy.fit(vis,40);
  if(cy.zoom()<READABLE_Z){
    cy.zoom(READABLE_Z);
    // pan so the first column's top sits just inside the top-left of the viewport
    const bb=vis.boundingBox();
    cy.pan({x:80-bb.x1*READABLE_Z, y:70-bb.y1*READABLE_Z});
  }
  curZoom=cy.zoom();
  placeHeaders();syncZoom();cy.style().update();
}
cy.ready(()=>{requestAnimationFrame(ensureVisible);});
window.addEventListener('resize',()=>{placeHeaders();});
</script>
</body></html>"""


# ── unified view (chain + edge [+ explorer] in one file, tab-toggled) ───────────

def _iframe_srcdoc(doc: str) -> str:
    """HTML-escape a full document so it can be embedded as an ``<iframe srcdoc>``
    attribute value (double-quoted). Each view keeps its OWN document — separate id
    namespace + JS scope — so the chain view's and edge view's heavily-overlapping
    ids (``top``/``wrap``/``panel``/``q``/``svg`` …) can't collide."""
    return (doc.replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def render_unified_html(
    title: str,
    chain_data: dict | None,
    edge_data: dict | None,
    *,
    after_only: bool = False,
    explorer_datasets: list[tuple[str, dict]] | None = None,
) -> str:
    """Render ONE self-contained HTML document holding both the chain view and the
    edge view (and, if ``explorer_datasets`` is given, the explorer view), behind a
    persistent top-of-page tab toggle. No reload, no separate files.

    Each view is embedded verbatim — its existing ``render_*_html`` output — inside
    its own ``<iframe srcdoc>``, so the views' markup/CSS/JS are fully isolated
    (the chain and edge templates reuse the same element ids, which would otherwise
    clash). All data stays inlined; the file is ``file://`` openable.
    """
    tabs: list[tuple[str, str, str]] = []  # (tab id, label, inner document html)
    if chain_data is not None:
        # Per-trial after-only Chain view (``extract_trial_chains`` output, keyed by
        # ``"trials"``); fall back to the legacy flat cross-trial render for the old
        # ``{"chains": ...}`` shape.
        if "trials" in chain_data:
            chain_doc = render_trial_chain_html(
                f"{title} — causal hypothesis chains", chain_data)
        else:
            chain_doc = render_chain_html(
                f"{title} — causal hypothesis chains", chain_data,
                after_only=after_only)
        tabs.append(("chain", "Chain view", chain_doc))
    if edge_data is not None:
        tabs.append(("edge", "Edge view",
                     render_edge_view_html(title, edge_data)))
    if explorer_datasets:
        tabs.append(("explorer", "Explorer view",
                     render_explorer_html(title, explorer_datasets)))
    if not tabs:
        raise ValueError("render_unified_html needs at least one of chain/edge data")

    buttons = "".join(
        f'<button class="tabbtn{" on" if i == 0 else ""}" data-view="{tid}">'
        f'{html.escape(label)}</button>'
        for i, (tid, label, _doc) in enumerate(tabs))
    frames = "".join(
        f'<iframe class="viewframe" id="frame-{tid}" data-view="{tid}"'
        f'{"" if i == 0 else " hidden"} srcdoc="{_iframe_srcdoc(doc)}"></iframe>'
        for i, (tid, _label, doc) in enumerate(tabs))
    return (_UNIFIED_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__TABS__", buttons).replace("__FRAMES__", frames))


_UNIFIED_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>eroom — __TITLE__</title><style>
  html,body{margin:0;height:100%;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;
            background:#0d0f15;color:#e6e6e6;}
  #tabbar{display:flex;gap:6px;align-items:center;padding:8px 14px;background:#10141c;
          border-bottom:1px solid #262b36;position:sticky;top:0;z-index:5;}
  #tabbar .ttl{font-weight:700;font-size:14px;margin-right:14px;}
  #tabbar .ttl small{color:#6b7484;font-weight:400;margin-left:8px;}
  .tabbtn{background:#0c0e13;color:#cfd6e4;border:1px solid #2a2f3a;border-radius:7px;
          padding:5px 14px;font:13px ui-sans-serif;cursor:pointer;}
  .tabbtn:hover{background:#161b25;}
  .tabbtn.on{background:#2a3550;border-color:#4f7cff;color:#cfe0ff;font-weight:600;}
  #frames{position:absolute;top:46px;left:0;right:0;bottom:0;}
  .viewframe{position:absolute;inset:0;width:100%;height:100%;border:0;background:#0d0f15;}
  .viewframe[hidden]{display:none;}
</style></head><body>
<div id="tabbar"><span class="ttl">eroom graph<small>__TITLE__</small></span>__TABS__</div>
<div id="frames">__FRAMES__</div>
<script>
  const btns=[...document.querySelectorAll('.tabbtn')];
  const frames=[...document.querySelectorAll('.viewframe')];
  function show(view){
    btns.forEach(b=>b.classList.toggle('on',b.dataset.view===view));
    frames.forEach(f=>{f.hidden=(f.dataset.view!==view);});
  }
  btns.forEach(b=>b.addEventListener('click',()=>show(b.dataset.view)));
</script></body></html>"""


def open_html(path: Path) -> None:
    """Best-effort: open ``path`` in the default browser. Swallows failures so a
    headless / CI build never breaks just because there's no display."""
    import subprocess
    import sys
    import webbrowser
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            webbrowser.open(f"file://{Path(path).resolve()}")
    except Exception:  # noqa: BLE001
        pass


def write_chain_html(
    area: str | None = None,
    *,
    snapshot: str | None = None,
    suffix: str = "annotated",
    limit: int = 5,
    trials: list[str] | None = None,
    out: str | None = None,
    after_only: bool = False,
) -> tuple[Path, dict]:
    """Render the chain (debug) view to an HTML file; return ``(path, data)``.

    Importable so a build can render + auto-open the viz at the end of a run
    (``build_graph --viz``). ``area`` resolves ``data/exports/<area>_<suffix>.json``
    and uses ``<area>_premerge.json`` for a faithful before-merge block when present.
    ``after_only`` renders just the assembled graph (no before-merge block).
    """
    target = snapshot or area
    if not target:
        raise ValueError("write_chain_html needs area or snapshot")
    path = _resolve(target, suffix)
    snap = load_snapshot(path)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    before_snap = None
    if area and not after_only:
        pm = EXPORTS / f"{area}_premerge.json"
        if pm.exists():
            before_snap = load_snapshot(pm)
    data = extract_chains(snap, before_snap=before_snap, limit=limit, trials=trials)
    out_path = Path(out) if out else VIZ_DIR / f"{path.stem}_chains.html"
    out_path.write_text(render_chain_html(
        f"{path.stem} — causal hypothesis chains", data, after_only=after_only))
    return out_path, data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area")
    ap.add_argument("--suffix", default="annotated")
    ap.add_argument("--snapshot")
    ap.add_argument("--graph", help="path to ANY graph snapshot json to display "
                                    "(alias for --snapshot; click any box for all fields)")
    ap.add_argument("--mode",
                    choices=["graph", "chain", "explorer", "edge", "all", "layered"],
                    default="chain",
                    help="'graph' = the PRODUCT overview: ONE merged force-directed "
                         "network (shared nodes, type-colored + legend, filters, "
                         "search, cluster-by-type, static-but-interactive). "
                         "'layered' = left→right column-per-node-type DAG of the merged "
                         "graph (COMPOUND→TARGET→MECHANISM→BIOLOGY→ENDPOINT→INDICATION→"
                         "POPULATION + a dashed-red adverse-event row), semantic zoom. "
                         "'all' = chain + edge (+ explorer) in ONE file, tab-toggled.")
    ap.add_argument("--limit", type=int, default=5,
                    help="--mode chain: number of cross-trial chains to show.")
    ap.add_argument("--sample", type=int, default=20,
                    help="--mode all: number of trials to sample for the per-trial "
                         "after-only Chain view (deterministic cross-section).")
    ap.add_argument("--seed", type=int, default=1337,
                    help="--mode all: RNG seed for the deterministic trial sample.")
    ap.add_argument("--trials")
    ap.add_argument("--annotations", default="data/annotations",
                    help="retained for compatibility; edge-view (s,t) now reads node "
                         "descriptions from the graph, not annotations (unused)")
    ap.add_argument("--out")
    ap.add_argument("--no-open", action="store_true",
                    help="render only; don't auto-open the HTML in a browser")
    ap.add_argument("--after-only", action="store_true",
                    help="chain mode: show only the assembled (after-merge) graph, "
                         "dropping the before-merge debug block (toggle).")
    args = ap.parse_args()

    snapshot = args.snapshot or args.graph   # --graph is an intuitive alias for any path
    target = snapshot or args.area
    if not target:
        ap.error("pass --graph/--snapshot <path> or --area <name>")

    if args.mode == "graph":
        # THE product overview: one merged force-directed network. Nodes are already
        # merged in the snapshot, so each appears exactly once; shared mechanism /
        # biology / population nodes fan in from every compound & trial that uses
        # them. Self-contained (vis-network from CDN), file:// openable.
        path = _resolve(target, args.suffix)
        snap = load_snapshot(path)
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_graph.html"
        d = to_graph_view(snap)
        out.write_text(render_graph_html(path.stem, d))
        hidden = ", ".join(d["hidden_by_default"]) or "none"
        print(f"  graph: {d['n_nodes']} nodes / {d['n_edges']} edges "
              f"(one merged graph; default-hidden: {hidden}) → {out} "
              f"({out.stat().st_size // 1024} KB)")
        if not args.no_open:
            open_html(out)
        return

    if args.mode == "layered":
        # The layered causal-chain DAG: left→right type columns of the MERGED graph,
        # each node once, shared nodes converging, E[p]%-labeled edges, dashed-red AE
        # row (default-hidden), semantic zoom. Self-contained (cytoscape from CDN).
        path = _resolve(target, args.suffix)
        snap = load_snapshot(path)
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_layered.html"
        d = to_layered_view(snap)
        out.write_text(render_layered_html(path.stem, d))
        cols = " → ".join(c["header"] for c in d["columns"])
        print(f"  layered: {d['n_nodes']} nodes / {d['n_edges']} backbone edges "
              f"(+ {d['n_ae_nodes']} AE, hidden) → {out} "
              f"({out.stat().st_size // 1024} KB)")
        print(f"  columns: {cols}")
        if not args.no_open:
            open_html(out)
        return

    if args.mode == "all":
        # ONE file: chain view + edge view (+ explorer), tab-toggled. Reuses each
        # view's own render_*_html verbatim (isolated via iframe srcdoc). The Chain
        # view is AFTER-ONLY and shows a deterministic random cross-section of up to
        # --sample trials, each sectioned by NCT (no before/after split, no premerge).
        path = _resolve(target, args.suffix)
        snap = load_snapshot(path)
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        trials_all = [t.strip() for t in args.trials.split(",")] if args.trials else None
        chain_data = extract_trial_chains(
            snap, sample=args.sample, seed=args.seed, trials=trials_all)
        sampled = [s["trial"] for s in chain_data["trials"]]
        edge_data = build_edge_view_data(str(path), annotations_dir=args.annotations)
        out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_unified.html"
        out.write_text(render_unified_html(path.stem, chain_data, edge_data))
        n_chains = sum(s["n_chains"] for s in chain_data["trials"])
        print(f"  unified: chain ({chain_data['n_sampled']}/{chain_data['n_trials_total']} "
              f"trials, {n_chains} chains, after-only) + edge "
              f"({edge_data['n_edges_with_pairings']} edges, "
              f"{edge_data['n_multi']} multi-contributor) → {out} "
              f"({out.stat().st_size // 1024} KB)")
        print(f"  sampled NCTs ({len(sampled)}): {', '.join(sampled)}")
        if not args.no_open:
            open_html(out)
        return

    if args.mode == "explorer":
        path = _resolve(target, args.suffix)
        snap = load_snapshot(path)
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_explorer.html"
        d = to_vis(snap)
        out.write_text(render_explorer_html(path.stem, [(path.stem, d)]))
        print(f"  explorer: {len(d['nodes'])} nodes, {len(d['edges'])} edges → {out}")
        if not args.no_open:
            open_html(out)
        return

    if args.mode == "edge":
        path = _resolve(target, args.suffix)
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_edge_view.html"
        d = build_edge_view_data(str(path), annotations_dir=args.annotations)
        out.write_text(render_edge_view_html(path.stem, d))
        print(f"  edge view: {d['n_edges_with_pairings']} backbone edges, "
              f"{d['n_multi']} with ≥2 contributors → {out}")
        if not args.no_open:
            open_html(out)
        return

    trials = [t.strip() for t in args.trials.split(",")] if args.trials else None
    out, data = write_chain_html(
        area=args.area, snapshot=snapshot, suffix=args.suffix,
        limit=args.limit, trials=trials, out=args.out, after_only=args.after_only,
    )
    print(f"  chains shown: {len(data['chains'])} of {data['n_total']}")
    for c in data["chains"]:
        names = " → ".join(c["roles"].get(r, {}).get("after", {}).get("name", "·") for r in BACKBONE_SIG)
        print(f"    {c['trial']}: {names}")
    print(f"→ {out}  ({out.stat().st_size // 1024} KB)")
    if not args.no_open:
        open_html(out)


if __name__ == "__main__":
    main()
