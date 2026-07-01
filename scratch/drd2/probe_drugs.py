"""Probe the curated ChEMBL mechanism annotations for the candidate drugs.

The target->mechanism roster MISSES haloperidol, olanzapine, and the major
Parkinson's agonists (pramipexole/ropinirole/rotigotine/apomorphine). Before
selecting drugs we must know, per candidate: what target_chembl_id + action_type
ChEMBL records, and specifically whether ANY mechanism row lands on CHEMBL217
(DRD2). If an agonist's only dopamine mechanism is on D3 (CHEMBL234), it will NOT
pool on the DRD2 node and the experiment's premise breaks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
HERE = Path(__file__).parent

CANDIDATES = {
    "agonist?": ["pramipexole", "ropinirole", "rotigotine", "apomorphine",
                 "cabergoline", "quinagolide", "bromocriptine", "pergolide",
                 "lisuride", "piribedil"],
    "antagonist?": ["haloperidol", "risperidone", "olanzapine", "metoclopramide",
                    "prochlorperazine", "domperidone", "amisulpride", "sulpiride",
                    "chlorpromazine", "quetiapine"],
}

# Dopamine-receptor single-protein targets, for flagging which receptor a
# mechanism row lands on.
DOPA_TARGETS = {
    "CHEMBL217": "DRD2", "CHEMBL234": "DRD3", "CHEMBL219": "DRD4",
    "CHEMBL2056": "DRD1", "CHEMBL1850": "DRD5",
}


def get(client, url, params=None):
    for attempt in range(4):
        try:
            r = client.get(url, params=params, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(1.0 * (attempt + 1))
    return {}


def search_id(client, name):
    """Resolve a drug name to its PARENT ChEMBL molecule id (mechanisms live on
    the parent)."""
    data = get(client, f"{CHEMBL}/molecule/search.json", {"q": name, "limit": 5})
    mols = data.get("molecules", [])
    for m in mols:
        if (m.get("pref_name") or "").lower() == name.lower():
            return m.get("molecule_chembl_id"), m.get("pref_name")
    if mols:
        return mols[0].get("molecule_chembl_id"), mols[0].get("pref_name")
    return None, None


def main():
    out = {}
    with httpx.Client(follow_redirects=True) as client:
        for group, names in CANDIDATES.items():
            print(f"\n===== {group} =====")
            for name in names:
                mid, pref = search_id(client, name)
                if not mid:
                    print(f"  {name:16s} -> NOT FOUND")
                    continue
                # mechanisms are recorded on the parent; resolve parent first
                data = get(client, f"{CHEMBL}/mechanism.json",
                           {"molecule_chembl_id": mid, "limit": 50})
                mechs = data.get("mechanisms", [])
                # also check the parent molecule id (salts delegate to parent)
                rows = []
                d2_hit = False
                for m in mechs:
                    tid = m.get("target_chembl_id") or ""
                    at = m.get("action_type") or "?"
                    moa = m.get("mechanism_of_action") or ""
                    recep = DOPA_TARGETS.get(tid, "")
                    if tid == "CHEMBL217":
                        d2_hit = True
                    rows.append((tid, recep, at, moa))
                out[name] = {"chembl_id": mid, "pref_name": pref,
                             "d2_direct": d2_hit, "mechanisms": rows}
                tag = "  *** DRD2 ***" if d2_hit else ""
                print(f"  {name:16s} {mid:14s} ({pref}){tag}")
                for tid, recep, at, moa in rows:
                    mark = " <==DRD2" if tid == "CHEMBL217" else (
                        f" [{recep}]" if recep else "")
                    print(f"        {tid:14s} {at:18s} {moa}{mark}")
                time.sleep(0.1)
    (HERE / "probe_drugs.json").write_text(json.dumps(out, indent=2))
    print("\n\n=== SUMMARY: which candidates have a DIRECT DRD2 (CHEMBL217) mechanism ===")
    for name, rec in out.items():
        flag = "DRD2-direct" if rec["d2_direct"] else "NO direct DRD2"
        dopa = sorted({DOPA_TARGETS.get(t, "") for t, *_ in rec["mechanisms"]
                       if t in DOPA_TARGETS})
        print(f"  {name:16s} {flag:14s} dopamine receptors hit: {dopa}")


if __name__ == "__main__":
    main()
