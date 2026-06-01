"""Run + smoke-test the bottom-up (chains-first) build. WIP — see populate_bottomup.

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


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="multi_indication_52_train")
    ap.add_argument("--max-trials", type=int, default=10)
    args = ap.parse_args()

    corpus_path = CORPORA_DIR / f"{args.corpus}.txt"
    trials = await fetch_trials("cancer", args.max_trials, False, corpus_path=corpus_path)
    trials = trials[: args.max_trials]
    print(f"fetched {len(trials)} trials: {[t.nct_id for t in trials]}")

    client = anthropic.AsyncAnthropic(timeout=60.0)
    g = await build_bottomup(trials, client)

    total_chains = sum(len(ts.chains) for ts in g.trial_subgraphs.values())
    print(f"\n=== bottom-up build ===")
    print(f"nodes={g._graph.number_of_nodes()}  edges={g._graph.number_of_edges()}  "  # noqa: SLF001
          f"trial_subgraphs={len(g.trial_subgraphs)}  chains={total_chains}")
    # node-type breakdown sanity
    from collections import Counter
    types = Counter(g._graph.nodes[n].get("node_type") for n in g._graph.nodes)  # noqa: SLF001
    print(f"node types: {dict(types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
