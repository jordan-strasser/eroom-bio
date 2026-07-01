"""Finalize per-drug direction + build the primed mechanism cache for Phase 4.

ChEMBL stores some drugs' mechanisms on a SALT form rather than the OT-resolved
parent id, so get_drug_mechanisms(parent) returns [] and direction stamps
'unknown'. Fix (data-only): for each drug, collect mechanisms across its whole
molecule family (name -> forms -> parents) and key them under the OT-resolved
parent id — the exact id direction.stamp_directions() will look up at build time.

Cross-checks each stamped direction against OT's own per-target actionType.
Writes:
  scratch/drd2/mech_cache_primed.json  — {"mechanism:<OTparent>": {"mechanisms":[...]}}
  scratch/drd2/roster_final.json       — per-drug: chembl_id, direction, drd2, aliases
Production cache is written in Phase 4 (on approval), not here.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from src.graph.direction import direction_from_mechanisms
from src.ingestion.chembl import ChEMBLClient
from src.ingestion.opentargets import GRAPHQL_URL, OpenTargetsClient

HERE = Path(__file__).parent
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

SELECTED = {
    "agonist": ["pramipexole", "ropinirole", "rotigotine", "apomorphine",
                "cabergoline"],
    "antagonist": ["haloperidol", "risperidone", "olanzapine", "metoclopramide",
                   "prochlorperazine"],
}


def family_ids(http: httpx.Client, name: str, parent: str) -> set[str]:
    """All ChEMBL molecule ids in the drug's family (forms + their parents)
    plus the OT parent itself."""
    ids = {parent}
    data = http.get(f"{CHEMBL}/molecule/search.json",
                    params={"q": name, "limit": 10}, timeout=30).json()
    for m in data.get("molecules", []):
        mid = m.get("molecule_chembl_id")
        hier = m.get("molecule_hierarchy") or {}
        p = hier.get("parent_chembl_id")
        # keep only forms tied to OUR parent (avoid e.g. dexpramipexole branch)
        if mid == parent or p == parent:
            if mid:
                ids.add(mid)
            if p:
                ids.add(p)
    return ids


async def main() -> None:
    ot = OpenTargetsClient()
    chembl = ChEMBLClient(cache_path=HERE / "_chembl_mech_cache.json")
    primed: dict[str, dict] = {}
    roster: dict[str, dict] = {}
    with httpx.Client(follow_redirects=True) as http:
        for expected, names in SELECTED.items():
            for name in names:
                data = await ot.get_drug_with_targets(name)
                parent = data["chembl_id"]
                drd2 = any(t.get("approved_symbol") == "DRD2"
                           for t in data.get("targets") or [])
                # collect mechanisms across the family, keyed under OT parent
                mechs = await chembl.get_drug_mechanisms(parent)
                if not mechs:
                    merged: list[dict] = []
                    for fid in sorted(family_ids(http, name, parent)):
                        rows = http.get(
                            f"{CHEMBL}/mechanism.json",
                            params={"molecule_chembl_id": fid, "limit": 50},
                            timeout=30).json().get("mechanisms", [])
                        merged.extend(rows)
                    mechs = merged
                direction = direction_from_mechanisms(mechs)
                # DRD2-scoped direction from OT actionType (independent check)
                ot_d2 = _ot_drd2_action(http, name)
                ats = sorted({m.get("action_type") for m in mechs
                              if m.get("action_type")})
                status = "OK" if direction == expected else f"MISMATCH(exp {expected})"
                if mechs:
                    primed[f"mechanism:{parent}"] = {"mechanisms": mechs}
                roster[name] = {
                    "chembl_id": parent, "direction": direction,
                    "expected": expected, "drd2_confirmed": drd2,
                    "ot_drd2_action": ot_d2, "chembl_action_types": ats,
                    "aliases": data.get("aliases") or [],
                }
                print(f"  {name:16s} {parent:14s} dir={direction:11s} "
                      f"OT_DRD2={ot_d2:11s} drd2={'Y' if drd2 else 'N'} "
                      f"AT={ats} [{status}]")
    (HERE / "mech_cache_primed.json").write_text(json.dumps(primed, indent=2))
    (HERE / "roster_final.json").write_text(json.dumps(roster, indent=2))
    bad = [n for n, r in roster.items() if r["direction"] != r["expected"]]
    drd2_bad = [n for n, r in roster.items() if not r["drd2_confirmed"]]
    print(f"\nprimed {len(primed)} mechanism-cache entries -> mech_cache_primed.json")
    print("direction mismatches:", bad or "NONE")
    print("DRD2 not confirmed:", drd2_bad or "NONE")


def _ot_drd2_action(http: httpx.Client, name: str) -> str:
    q = ('query Q($n:String!){search(queryString:$n,entityNames:["drug"],'
         'page:{size:1,index:0}){hits{object{... on Drug{mechanismsOfAction'
         '{rows{actionType targets{approvedSymbol}}}}}}}}')
    r = http.post(GRAPHQL_URL, json={"query": q, "variables": {"n": name}}, timeout=30)
    hits = r.json()["data"]["search"]["hits"]
    if not hits:
        return "?"
    rows = ((hits[0].get("object") or {}).get("mechanismsOfAction") or {}).get("rows") or []
    for row in rows:
        if any(t.get("approvedSymbol") == "DRD2" for t in row.get("targets") or []):
            at = (row.get("actionType") or "").upper()
            if "AGONIST" in at and "PARTIAL" not in at and "INVERSE" not in at:
                return "agonist"
            if at in {"ANTAGONIST", "INHIBITOR", "BLOCKER"} or "INVERSE" in at:
                return "antagonist"
            return at.lower()
    return "?"


if __name__ == "__main__":
    asyncio.run(main())
