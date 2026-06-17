"""Expand the ChEMBL mechanism cache (curated action_type) for a built graph.

Native modulation-direction (src.graph.direction) reads ChEMBL ``action_type``
from data/cache/chembl_mechanisms.json. The populate path fetches these during a
fresh build; this script backfills the cache for an already-built snapshot's
compounds so direction can be stamped without a full rebuild.

Usage:
    python -m scripts.build_chembl_mechanism_cache --graph data/exports/onco_scale_500_frontier_annotated.json
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from src.graph.store import GraphStore

CACHE = Path("data/cache/chembl_mechanisms.json")
API = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"


def fetch(chembl_id: str) -> list[dict] | None:
    q = urllib.parse.urlencode({"molecule_chembl_id": chembl_id, "limit": 50})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=25) as r:
                return json.loads(r.read()).get("mechanisms", [])
        except Exception:  # noqa: BLE001
            if attempt == 2:
                return None
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    g = GraphStore(); g.import_snapshot(args.graph)
    cids = sorted({n.get("chembl_id") for n in g.get_nodes_by_type("InterventionNode")
                   if n.get("chembl_id")})
    todo = [c for c in cids if args.refresh or f"mechanism:{c}" not in cache]
    print(f"{len(cids)} chembl_id compounds, {len(todo)} to fetch")

    got = 0
    for i, cid in enumerate(todo):
        mechs = fetch(cid)
        if mechs is None:
            continue
        cache[f"mechanism:{cid}"] = {"mechanisms": mechs}
        got += int(bool(mechs))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)} ({got} with mechanisms)", flush=True)
        time.sleep(0.12)

    CACHE.write_text(json.dumps(cache))
    print(f"wrote {CACHE}: {len(cache)} entries (+{got} with mechanisms this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
