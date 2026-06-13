"""Cross-indication sanity check for the multi_indication graph.

First multi-indication build for eroom-bio. Before trusting any
case-study predictions, verify that the populator produced sensible
nodes + edges for the new indications (alzheimers, colorectal,
atherosclerosis, hypercholesterolemia, NSCLC, thyroid) — not just
copy-pasta of melanoma-style chains.

Diagnostics per indication:

  1. IndicationNode exists with a sensible name and canonical id
  2. Number of training trials mapped to this indication
  3. Targets reached by these trials (gene_symbol distribution)
  4. Mechanism categories used (checkpoint_blockade vs cholesterol vs
     beta_amyloid_clearance — should differ across indications)
  5. Biology nodes wired to this indication (biology_drives edges)
  6. Population nodes specific to this indication (subgroup spread)
  7. Sample trial subgraphs with their full resolved chain
  8. The 5 case-study compounds' compound→target edges (sanity that
     solanezumab→APP, bevacizumab→VEGFA, torcetrapib→CETP, selumetinib→MAP2K1
     all resolved correctly)

Usage:
    python -m scripts.multi_indication_sanity
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore

console = Console()

GRAPH_PATH = Path("data/exports/multi_indication_annotated.json")

# Slugs we expect to see as IndicationNodes after the build. The
# left column is the canonical slug we predict; the right is the
# CT.gov display name the LLM canonicalizer was fed. Used to print
# a friendly "expected vs found" check.
EXPECTED_INDICATIONS: list[tuple[str, str]] = [
    ("melanoma", "Melanoma"),
    ("alzheimer_disease", "Alzheimer Disease"),
    ("colorectal_cancer", "Colorectal Cancer"),
    ("atherosclerosis", "Atherosclerosis"),
    ("hypercholesterolemia", "Hypercholesterolemia"),
    ("non_small_cell_lung_cancer", "Non-Small Cell Lung Cancer"),
    ("thyroid_cancer", "Thyroid Cancer"),
]


# Case-study compounds + their expected canonical targets.
# Used to verify that compound→target resolution worked for the
# diverse case-study compounds (not just melanoma-corpus compounds).
EXPECTED_COMPOUND_TARGETS: list[tuple[str, str, str]] = [
    ("nivolumab",     "PDCD1",  "anti-PD-1 in melanoma (CheckMate-067)"),
    ("solanezumab",   "APP",    "anti-amyloid mAb in Alzheimer's"),
    ("bevacizumab",   "VEGFA",  "anti-VEGF in adjuvant colon"),
    ("torcetrapib",   "CETP",   "CETP inhibitor in hypercholesterolemia"),
    ("selumetinib",   "MAP2K1", "MEK inhibitor in BRAF-mutant thyroid"),
]


def _find_indication(graph: GraphStore, slug: str) -> dict | None:
    """Return the IndicationNode if present, else None. Tries both the
    expected slug and a few common variants (alzheimers vs alzheimer_disease)."""
    g = graph._graph  # noqa: SLF001
    candidates = [slug]
    if slug == "alzheimer_disease":
        candidates += ["alzheimers", "alzheimers_disease"]
    if slug == "non_small_cell_lung_cancer":
        candidates += ["nsclc", "lung_cancer"]
    for cid in candidates:
        if cid in g and g.nodes[cid].get("node_type") == "IndicationNode":
            n = dict(g.nodes[cid])
            n["id"] = cid
            return n
    return None


def _trials_for_indication(graph: GraphStore, indication_id: str) -> list[str]:
    """NCTs whose chains touch this indication_id."""
    nct_ids: list[str] = []
    for tid, ts in graph.trial_subgraphs.items():
        for c in ts.chains:
            if c.indication_id == indication_id:
                nct_ids.append(tid)
                break
    return nct_ids


def _targets_for_indication(graph: GraphStore, indication_id: str) -> Counter:
    """Counter of gene_symbol → trial_count for compounds in trials
    that have this indication."""
    g = graph._graph  # noqa: SLF001
    counter: Counter = Counter()
    for tid, ts in graph.trial_subgraphs.items():
        if not any(c.indication_id == indication_id for c in ts.chains):
            continue
        for chain in ts.chains:
            if chain.indication_id != indication_id:
                continue
            if chain.target_id and chain.target_id != "UNKNOWN":
                tgt_node = g.nodes.get(chain.target_id) or {}
                sym = tgt_node.get("gene_symbol") or chain.target_id
                counter[sym] += 1
    return counter


def _mechanisms_for_indication(graph: GraphStore, indication_id: str) -> Counter:
    counter: Counter = Counter()
    for ts in graph.trial_subgraphs.values():
        if not any(c.indication_id == indication_id for c in ts.chains):
            continue
        for c in ts.chains:
            if c.indication_id == indication_id and c.mechanism_id and c.mechanism_id != "UNKNOWN":
                counter[c.mechanism_id] += 1
    return counter


def _populations_for_indication(graph: GraphStore, indication_id: str) -> list[str]:
    """All population slugs that target this indication via
    responds_differently."""
    g = graph._graph  # noqa: SLF001
    pops: list[str] = []
    for u, v, k in g.in_edges(indication_id, keys=True):
        if k == EdgeType.RESPONDS_DIFFERENTLY.value:
            pops.append(u)
    return sorted(pops)


def _biology_drivers_for_indication(
    graph: GraphStore, indication_id: str,
) -> list[tuple[str, float, float]]:
    """biology_drives → indication edges with their belief stats."""
    g = graph._graph  # noqa: SLF001
    out: list[tuple[str, float, float]] = []
    for u, v, k, data in g.in_edges(indication_id, keys=True, data=True):
        if k != EdgeType.BIOLOGY_DRIVES.value:
            continue
        try:
            b = EdgeBeliefState.model_validate(data.get("belief") or {})
        except Exception:  # noqa: BLE001
            continue
        out.append((u, b.expected_probability, b.evidence_strength))
    return sorted(out, key=lambda r: -r[2])  # highest n_eff first


def _compound_target_edge(
    graph: GraphStore, compound_id: str,
) -> list[tuple[str, str, float, float]]:
    """affects edges out of compound_id. Returns (target_id, target_gene_symbol,
    E[p], n_eff). Tries common compound-id variants for the case studies."""
    g = graph._graph  # noqa: SLF001
    candidates = [
        compound_id,
        compound_id.lower(),
        compound_id.replace(" ", "_").lower(),
    ]
    found_id = None
    for c in candidates:
        if c in g and g.nodes[c].get("node_type") == "InterventionNode":
            found_id = c
            break
    if not found_id:
        return []
    out: list[tuple[str, str, float, float]] = []
    for _u, v, k, data in g.out_edges(found_id, keys=True, data=True):
        if k != EdgeType.AFFECTS.value:
            continue
        try:
            b = EdgeBeliefState.model_validate(data.get("belief") or {})
        except Exception:  # noqa: BLE001
            continue
        sym = (g.nodes.get(v) or {}).get("gene_symbol") or v
        out.append((v, sym, b.expected_probability, b.evidence_strength))
    return sorted(out, key=lambda r: -r[3])


def main(args: argparse.Namespace) -> int:
    console.rule("[bold]Multi-indication graph sanity check[/bold]")
    graph = GraphStore()
    graph.import_snapshot(str(args.graph))
    stats = graph.stats()
    console.print(
        f"Loaded {args.graph}: {stats['node_count']} nodes, "
        f"{stats['edge_count']} edges, "
        f"{len(graph.trial_subgraphs)} trial subgraphs\n"
    )

    # ── 1. Indication node coverage ───────────────────────────────────
    console.rule("[bold]1. Indication nodes (expected from corpus)[/bold]")
    ind_table = Table(box=None)
    ind_table.add_column("expected slug")
    ind_table.add_column("found in graph?")
    ind_table.add_column("display name")
    ind_table.add_column("variants observed")
    for slug, ct_term in EXPECTED_INDICATIONS:
        n = _find_indication(graph, slug)
        if n is None:
            ind_table.add_row(slug, "[red]MISSING[/red]", "—", "—")
        else:
            md = n.get("metadata") or {}
            variants = md.get("observed_variants") or []
            ind_table.add_row(
                slug,
                f"[green]✓[/green] (as `{n['id']}`)",
                n.get("name", ""),
                ", ".join(variants[:3])[:80],
            )
    console.print(ind_table)

    # ── 2. Trial counts per indication ────────────────────────────────
    console.rule("[bold]2. Trials per indication[/bold]")
    trial_counts = Table(box=None)
    trial_counts.add_column("indication")
    trial_counts.add_column("n trials", justify="right")
    trial_counts.add_column("sample NCTs")
    for slug, _ in EXPECTED_INDICATIONS:
        n = _find_indication(graph, slug)
        if not n:
            trial_counts.add_row(slug, "—", "—")
            continue
        ncts = _trials_for_indication(graph, n["id"])
        sample = ", ".join(ncts[:3])
        if len(ncts) > 3:
            sample += f", … (+{len(ncts) - 3} more)"
        trial_counts.add_row(slug, str(len(ncts)), sample)
    console.print(trial_counts)

    # ── 3. Targets per indication ─────────────────────────────────────
    console.rule("[bold]3. Targets reached by each indication's trials[/bold]")
    for slug, _ in EXPECTED_INDICATIONS:
        n = _find_indication(graph, slug)
        if not n:
            continue
        targets = _targets_for_indication(graph, n["id"])
        if not targets:
            console.print(f"  [yellow]{slug}[/yellow]: no targets resolved")
            continue
        console.print(f"  [bold]{slug}[/bold]: {len(targets)} unique targets")
        for sym, n_trials in targets.most_common(8):
            console.print(f"      {sym:<18} {n_trials} trial(s)")

    # ── 4. Mechanisms per indication ──────────────────────────────────
    console.rule("[bold]4. Mechanism categories per indication[/bold]")
    for slug, _ in EXPECTED_INDICATIONS:
        n = _find_indication(graph, slug)
        if not n:
            continue
        mechs = _mechanisms_for_indication(graph, n["id"])
        if not mechs:
            console.print(f"  [yellow]{slug}[/yellow]: no mechanisms resolved")
            continue
        console.print(
            f"  [bold]{slug}[/bold]: {sum(mechs.values())} mech-edges across "
            f"{len(mechs)} unique mechanisms"
        )
        for mech, n_trials in mechs.most_common(6):
            console.print(f"      {mech:<30} {n_trials} trial(s)")

    # ── 5. Biology nodes feeding each indication ─────────────────────
    console.rule("[bold]5. biology_drives → indication edges[/bold]")
    for slug, _ in EXPECTED_INDICATIONS:
        n = _find_indication(graph, slug)
        if not n:
            continue
        bios = _biology_drivers_for_indication(graph, n["id"])
        if not bios:
            console.print(f"  [yellow]{slug}[/yellow]: no biology drivers")
            continue
        console.print(f"  [bold]{slug}[/bold]: {len(bios)} biology drivers")
        for bid, ep, n_eff in bios[:5]:
            console.print(
                f"      {bid:<35} E[p]={ep:.3f}  n_eff={n_eff:.2f}"
            )

    # ── 6. Populations per indication ─────────────────────────────────
    console.rule("[bold]6. Population nodes per indication (round-17 enrichment)[/bold]")
    for slug, _ in EXPECTED_INDICATIONS:
        n = _find_indication(graph, slug)
        if not n:
            continue
        pops = _populations_for_indication(graph, n["id"])
        unselected = [p for p in pops if p.endswith("__unselected")]
        stratified = [p for p in pops if not p.endswith("__unselected")]
        console.print(
            f"  [bold]{slug}[/bold]: {len(pops)} populations "
            f"({len(unselected)} unselected, {len(stratified)} stratified)"
        )
        for p in stratified[:6]:
            console.print(f"      {p}")
        if len(stratified) > 6:
            console.print(f"      … (+{len(stratified) - 6} more)")

    # ── 7. Case-study compounds: compound→target edges ─────────────────
    console.rule("[bold]7. Case-study compounds — affects-edge resolution[/bold]")
    cstable = Table(box=None)
    cstable.add_column("compound")
    cstable.add_column("expected target")
    cstable.add_column("found targets (top 3)")
    cstable.add_column("status")
    for compound_id, expected_gene, _ in EXPECTED_COMPOUND_TARGETS:
        edges = _compound_target_edge(graph, compound_id)
        if not edges:
            cstable.add_row(
                compound_id, expected_gene, "—",
                "[red]MISSING (compound not in graph or no edges)[/red]",
            )
            continue
        top = ", ".join(
            f"{sym} (E[p]={ep:.2f}, n={n:.1f})"
            for _id, sym, ep, n in edges[:3]
        )
        hit = any(sym == expected_gene for _, sym, _, _ in edges)
        status = (
            "[green]✓ matches[/green]" if hit
            else "[yellow]MISMATCH (no expected gene in resolved set)[/yellow]"
        )
        cstable.add_row(compound_id, expected_gene, top, status)
    console.print(cstable)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=GRAPH_PATH)
    args = parser.parse_args()
    raise SystemExit(main(args))
