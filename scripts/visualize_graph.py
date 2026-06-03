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

  * ``--mode explorer`` — force-directed whole-graph view (product/overview).

Read-only; pure stdlib. The merge DELETES loser nodes (only ids survive in
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
import re
from pathlib import Path

EXPORTS = Path("data/exports")
VIZ_DIR = Path("data/viz")

PALETTE = {
    "InterventionNode": "#3d6ae0", "TargetNode": "#7b5cff", "MechanismNode": "#1f8f4e",
    "BiologyNode": "#1098ad", "IndicationNode": "#b8860b", "PopulationNode": "#3a4150",
    "EndpointNode": "#b03060", "AdverseEventNode": "#9c2b2b",
}
EDGE_COLOR = {
    "binds_to": "#4f7cff", "modulates_via": "#9b6cff", "mechanism_affects": "#3cb44b",
    "biology_drives": "#1098ad", "responds_differently": "#8a93a3",
    "reflects_biology": "#4f9cff", "endpoint_captures": "#e0559b",
    "causes_ae": "#e6194B", "target_associated_ae": "#c07b00",
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


def extract_chains(snapshot: dict, *, before_snap: dict | None = None,
                   limit: int, trials: list[str] | None) -> dict:
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
                         "desc": node.get("description"), "flags": fl}
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
                                        "name": _name_for(tgt, node_index)})
            built.append({"trial": tid, "arm": ch.get("arm_id"), "outcome": ch.get("outcome"),
                          "roles": roles, "edges": edges, "aes": aes, "merge_hits": merge_hits})

    built.sort(key=lambda c: -c["merge_hits"])
    seen, dedup = set(), []
    for c in built:
        sig = tuple(c["roles"].get(r, {}).get("after", {}).get("id") for r in BACKBONE_SIG)
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(c)
    chosen = dedup[:limit] if not trials else dedup
    return {"chains": chosen, "n_total": len(dedup)}


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
                            "desc": rd["after"].get("desc"), "flags": fl}
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
                            "flags": {"merged_from": [], "used_by": []}}
            sx, sy = pos(ae["role"])
            parts.append(_edge(sx + 40, sy + _BOXH, aex + 45, aey, ae["type"], ae["ep"] if show_ep else None, dashed=True))
            parts.append(_box(aex, aey, PALETTE["AdverseEventNode"], "#20242e", 1.2,
                              ae["name"], ae["type"], key, w=90, h=38))
    return "\n".join(parts), details, y0 + len(chains) * _BAND_H


def _shared_svg(chains, which, y0, show_ep):
    """AFTER-merge layout: each UNIQUE node id drawn ONCE per column, so chains
    through a shared (merged) node converge on one box with edges fanning in.
    Adverse events are omitted here (they're shown per-compound in the before
    block) to keep the convergence readable."""
    parts, details = [], {}
    col_nodes = {c: {} for c in range(N_COLS)}
    edges = {}
    for ch in chains:
        roles = ch["roles"]
        for r, rd in roles.items():
            col_nodes[ROLE_COL[r]].setdefault(rd[which]["id"], {
                "name": rd[which]["name"], "role": r,
                "desc": rd["after"].get("desc"), "flags": rd["after"]["flags"]})
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
            key = f'{which}:{nid}'
            details[key] = {"col": next(h for _r, _f, _t, h, _c in ROLES if _r == entry["role"]),
                            "type": ROLE_TYPE[entry["role"]], "id": nid, "name": entry["name"],
                            "desc": entry["desc"], "flags": fl}
            parts.append(_box(cx, cy, PALETTE.get(ROLE_TYPE[entry["role"]], "#777"), stroke, sw,
                              entry["name"] or nid, entry["role"].upper(), key))
    return "\n".join(parts), details, y0 + (maxslot + 1) * _ROWSP + 8


def render_chain_html(title: str, data: dict) -> str:
    chains = data["chains"]
    width = int(_colx(N_COLS - 1) + _BOXW + 80)
    y_before = 96
    svg_b, det_b, bot_b = _rows_svg(chains, "before", y_before, show_ep=False)
    y_after = bot_b + 86
    svg_a, det_a, bot_a = _shared_svg(chains, "after", y_after, show_ep=True)
    height = int(bot_a + 30)
    markers = "".join(
        f'<marker id="ah-{et}" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{c}"/></marker>' for et, c in EDGE_COLOR.items())
    svg = (f'<svg id="svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><defs>{markers}</defs>'
           f'<text x="20" y="{y_before-44:.0f}" class="blk">▼ BEFORE MERGE — per-trial instances</text>{svg_b}'
           f'<line x1="0" y1="{bot_b+40:.0f}" x2="{width}" y2="{bot_b+40:.0f}" class="div"/>'
           f'<text x="20" y="{y_after-44:.0f}" class="blk">▼ AFTER MERGE — assembled graph</text>{svg_a}</svg>')
    blob = json.dumps({**det_b, **det_a}, default=str).replace("</", "<\\/")
    return (_CHAIN_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__META__", html.escape(f'{len(chains)} of {data["n_total"]} chains · '
                     'red=cross-id fusion · gold=cross-trial merge · orange=island · ~=changed by merge'))
            .replace("__SVG__", svg).replace("__DETAILS__", blob))


_CHAIN_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>chains — __TITLE__</title><style>
  body{margin:0;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;background:#0d0f15;color:#e6e6e6;}
  #top{padding:9px 16px;background:#161a22;border-bottom:1px solid #262b36;position:sticky;top:0;z-index:3;}
  #top b{font-size:15px;} #top .meta{color:#8a93a3;margin-left:10px;font-size:12px;}
  #wrap{display:flex;height:calc(100vh - 46px);} #scroll{flex:1;overflow:auto;}
  #panel{width:380px;flex:none;overflow:auto;padding:14px;background:#161a22;border-left:1px solid #262b36;}
  .blk{fill:#cfd6e4;font-size:14px;font-weight:700;letter-spacing:.05em;}
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
  panel.innerHTML=h;
});
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area")
    ap.add_argument("--suffix", default="annotated")
    ap.add_argument("--snapshot")
    ap.add_argument("--mode", choices=["chain", "explorer"], default="chain")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--trials")
    ap.add_argument("--out")
    args = ap.parse_args()

    target = args.snapshot or args.area
    if not target:
        ap.error("pass --area or --snapshot")
    path = _resolve(target, args.suffix)
    snap = load_snapshot(path)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "explorer":
        out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_explorer.html"
        d = to_vis(snap)
        out.write_text(render_explorer_html(path.stem, [(path.stem, d)]))
        print(f"  explorer: {len(d['nodes'])} nodes, {len(d['edges'])} edges → {out}")
        return

    trials = [t.strip() for t in args.trials.split(",")] if args.trials else None
    before_snap = None
    if args.area:                              # faithful before-block from the premerge dump
        pm = EXPORTS / f"{args.area}_premerge.json"
        if pm.exists():
            before_snap = load_snapshot(pm)
    data = extract_chains(snap, before_snap=before_snap, limit=args.limit, trials=trials)
    out = Path(args.out) if args.out else VIZ_DIR / f"{path.stem}_chains.html"
    out.write_text(render_chain_html(f"{path.stem} — causal hypothesis chains", data))
    print(f"  chains shown: {len(data['chains'])} of {data['n_total']}")
    for c in data["chains"]:
        names = " → ".join(c["roles"].get(r, {}).get("after", {}).get("name", "·") for r in BACKBONE_SIG)
        print(f"    {c['trial']}: {names}")
    print(f"→ {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
