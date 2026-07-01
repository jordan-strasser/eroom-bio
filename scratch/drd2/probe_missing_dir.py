"""Find where ChEMBL stores the mechanism for drugs whose OT-parent id has none.

pramipexole/ropinirole/apomorphine/metoclopramide return [] from
mechanism.json?molecule_chembl_id={OT parent}. Check: (a) the molecule hierarchy
(parent vs salt), (b) mechanisms under every related id, (c) what OT's OWN
mechanismsOfAction.actionType says (a possible direction fallback source).
"""
from __future__ import annotations

import asyncio
import json

import httpx

from src.ingestion.opentargets import GRAPHQL_URL

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
DRUGS = ["pramipexole", "ropinirole", "apomorphine", "metoclopramide",
         "rotigotine"]  # rotigotine works -> control


def cget(client, path, params=None):
    r = client.get(f"{CHEMBL}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def ot_moa(client, name):
    q = '''query Q($n:String!){search(queryString:$n,entityNames:["drug"],page:{size:1,index:0}){
      hits{id object{... on Drug{id name mechanismsOfAction{rows{actionType mechanismOfAction targets{approvedSymbol}}}}}}}}'''
    r = client.post(GRAPHQL_URL, json={"query": q, "variables": {"n": name}}, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["search"]["hits"]
    if not hits:
        return None, []
    obj = hits[0]["object"] or {}
    rows = (obj.get("mechanismsOfAction") or {}).get("rows") or []
    return obj.get("id"), [(row.get("actionType"),
                            [t.get("approvedSymbol") for t in row.get("targets") or []])
                           for row in rows]


async def main():
    with httpx.Client(follow_redirects=True) as client:
        for name in DRUGS:
            print(f"\n===== {name} =====")
            # all molecule forms by name
            data = cget(client, "molecule/search.json", {"q": name, "limit": 8})
            mols = data.get("molecules", [])
            ids = []
            for m in mols:
                mid = m.get("molecule_chembl_id")
                hier = m.get("molecule_hierarchy") or {}
                parent = hier.get("parent_chembl_id")
                ids.append(mid)
                print(f"  form {mid:14s} parent={parent} pref={m.get('pref_name')}")
            # mechanisms under each related id
            related = set(ids)
            for mid in ids:
                hier = next((m.get("molecule_hierarchy") or {} for m in mols
                             if m.get("molecule_chembl_id") == mid), {})
                if hier.get("parent_chembl_id"):
                    related.add(hier["parent_chembl_id"])
            for mid in sorted(related):
                mech = cget(client, "mechanism.json",
                            {"molecule_chembl_id": mid, "limit": 50}).get("mechanisms", [])
                if mech:
                    for me in mech:
                        print(f"    MECH {mid:14s} {me.get('action_type'):14s} "
                              f"{me.get('mechanism_of_action')}  tgt={me.get('target_chembl_id')}")
                else:
                    print(f"    MECH {mid:14s} (none)")
            # OT's own actionType
            otid, rows = ot_moa(client, name)
            print(f"  OT id={otid} actionTypes:")
            for at, syms in rows:
                print(f"    OT  {str(at):14s} {syms}")


if __name__ == "__main__":
    asyncio.run(main())
