"""Run + audit the bottom-up (chains-first) build vs top-down. WIP.

    python -m scripts.build_bottomup --max-trials 10
"""

from __future__ import annotations

import argparse
import asyncio

import anthropic
from dotenv import load_dotenv

load_dotenv()

from scripts.build_graph import CORPORA_DIR, fetch_trials  # noqa: E402
from src.graph.populate_bottomup import build_bottomup  # noqa: E402
from src.graph.populate_groundup import CHAIN_BACKBONE, ROLE_ATTRS, _UNKNOWN  # noqa: E402
from src.graph.store import GraphStore  # noqa: E402


def _concepts(g: GraphStore) -> set[str]:
    out = set()
    for n in g._graph.nodes:  # noqa: SLF001
        out.add(g._graph.nodes[n].get("ontology_id") or n)  # noqa: SLF001
    return out


def _td_chain_concepts(g: GraphStore, ncts: set[str]) -> set[str]:
    out = set()
    for nct in ncts:
        ts = g.trial_subgraphs.get(nct)
        if not ts:
            continue
        for ch in ts.chains:
            for a in ROLE_ATTRS:
                v = getattr(ch, a, None)
                if v and v != _UNKNOWN:
                    out.add(v)
    return out


def _td_backbone_edges(g: GraphStore, ncts: set[str]) -> set[tuple]:
    out = set()
    for nct in ncts:
        ts = g.trial_subgraphs.get(nct)
        if not ts:
            continue
        for ch in ts.chains:
            for et, sa, ta in CHAIN_BACKBONE:
                s, t = getattr(ch, sa, None), getattr(ch, ta, None)
                if s and t and s != _UNKNOWN and t != _UNKNOWN:
                    out.add((s, t, et.value))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="multi_indication_52_train")
    ap.add_argument("--max-trials", type=int, default=10)
    args = ap.parse_args()

    trials = (await fetch_trials(
        "cancer", args.max_trials, False,
        corpus_path=CORPORA_DIR / f"{args.corpus}.txt",
    ))[: args.max_trials]
    print(f"fetched {len(trials)} trials")

    client = anthropic.AsyncAnthropic(timeout=60.0)
    g = await build_bottomup(trials, client)

    survivors = {nct for nct, ts in g.trial_subgraphs.items() if ts.chains}
    edges_with_belief = sum(
        1 for *_e, d in g._graph.edges(keys=True, data=True)  # noqa: SLF001
        if (d.get("belief") or {}).get("evidence")
    )
    print(f"\n=== bottom-up build (n={len(trials)}) ===")
    print(f"  nodes={g._graph.number_of_nodes()}  edges={g._graph.number_of_edges()}  "  # noqa: SLF001
          f"edges_with_belief={edges_with_belief}")
    print(f"  surviving trials={len(survivors)}  concepts={len(_concepts(g))}")

    # Faithfulness vs top-down (mi_v2) on the SAME surviving trials.
    td = GraphStore()
    td.import_snapshot("data/exports/mi_v2_annotated.json")
    common = survivors & set(td.trial_subgraphs)
    def _bu_chain_concepts() -> set[str]:
        out = set()
        for ts in g.trial_subgraphs.values():
            for ch in ts.chains:
                for a in ROLE_ATTRS:
                    v = getattr(ch, a, None)
                    if v and v != _UNKNOWN:
                        try:
                            out.add(g.get_node(v).get("ontology_id") or v)
                        except KeyError:
                            out.add(v)
        return out

    td_concepts = _td_chain_concepts(td, common)
    bu_chain = _bu_chain_concepts()
    td_edges = _td_backbone_edges(td, common)
    print(f"\n=== faithfulness vs top-down (mi_v2), {len(common)} shared trials ===")
    print(f"  top-down chain concepts={len(td_concepts)}  backbone edges={len(td_edges)}")
    print(f"  bottom-up chain concepts={len(bu_chain)}  (delta {len(bu_chain) - len(td_concepts):+d})")
    missing = td_concepts - bu_chain
    extra = bu_chain - td_concepts
    print(f"  missing (top-down concept not reconstructed): {len(missing)}"
          + (f"  {sorted(missing)[:6]}" if missing else "  ✓"))
    print(f"  extra   (bottom-up split not in top-down):    {len(extra)}"
          + (f"  {sorted(extra)[:6]}" if extra else "  ✓"))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
