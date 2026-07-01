"""Phase 1 — DRD2 ChEMBL roster (cheap, metadata only).

Resolve the DRD2 target id (verify, don't hardcode), pull its mechanism table
(molecule_chembl_id + action_type), resolve each molecule to a preferred name +
synonyms, and map action_type -> direction bucket using the SAME table the
production spine uses (src.graph.direction), with PARTIAL AGONIST split out as its
own bucket and excluded from the +/- core (per the task spec).

Writes:
  scratch/drd2/roster.json   — full per-molecule roster (cached molecule fetches)
  scratch/drd2/_mol_cache.json — raw molecule records (so re-runs are free)

No annotation, no graph writes. Public ChEMBL REST only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from src.graph.direction import AGONIST, ANTAGONIST, UNKNOWN, direction_for_action_type

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
HERE = Path(__file__).parent
MOL_CACHE = HERE / "_mol_cache.json"
ROSTER_OUT = HERE / "roster.json"

# PARTIAL AGONIST is folded into AGONIST by production direction.py, but the task
# wants it flagged separately and excluded from the clean +/- core
# (aripiprazole/brexpiprazole/cariprazine muddy the sign). We detect it on the
# raw action_type string before delegating to direction_for_action_type.
PARTIAL = "partial_agonist"


def bucket(action_type: str | None) -> str:
    at = (action_type or "").strip().upper()
    if at == "PARTIAL AGONIST":
        return PARTIAL
    return direction_for_action_type(at)


def get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    for attempt in range(4):
        try:
            r = client.get(url, params=params, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    retry {attempt+1} ({exc})")
            time.sleep(1.5 * (attempt + 1))
    return {}


def resolve_target(client: httpx.Client) -> str:
    """Resolve 'Dopamine D2 receptor' -> ChEMBL target id; verify it's the human
    single protein. Returns the target_chembl_id (expected CHEMBL217)."""
    data = get(client, f"{CHEMBL}/target/search.json",
               {"q": "Dopamine D2 receptor", "limit": 25})
    hits = data.get("targets", [])
    print(f"target search 'Dopamine D2 receptor': {len(hits)} hits")
    chosen = None
    for t in hits:
        name = t.get("pref_name", "")
        org = t.get("organism", "")
        ttype = t.get("target_type", "")
        tid = t.get("target_chembl_id", "")
        flag = ""
        nl = name.lower()
        is_d2 = ("d(2) dopamine receptor" in nl or "dopamine d2 receptor" in nl)
        if (is_d2 and org == "Homo sapiens" and ttype == "SINGLE PROTEIN"):
            if chosen is None:
                chosen = tid
                flag = "  <-- CHOSEN"
        print(f"  {tid:14s} {ttype:16s} {org:16s} {name}{flag}")
    if not chosen:
        raise SystemExit("could not resolve human Dopamine D2 receptor single protein")
    return chosen


def pull_mechanisms(client: httpx.Client, target_id: str) -> list[dict]:
    """All mechanism rows acting on the target (paginated)."""
    out: list[dict] = []
    offset = 0
    while True:
        data = get(client, f"{CHEMBL}/mechanism.json",
                   {"target_chembl_id": target_id, "limit": 1000, "offset": offset})
        rows = data.get("mechanisms", [])
        out.extend(rows)
        meta = data.get("page_meta", {})
        if not meta.get("next"):
            break
        offset += len(rows)
    return out


def load_mol_cache() -> dict:
    if MOL_CACHE.exists():
        return json.loads(MOL_CACHE.read_text())
    return {}


def fetch_molecule(client: httpx.Client, cache: dict, mid: str) -> dict:
    if mid in cache:
        return cache[mid]
    data = get(client, f"{CHEMBL}/molecule/{mid}.json")
    syns = []
    for s in data.get("molecule_synonyms") or []:
        syn = (s.get("molecule_synonym") or "").strip()
        if syn:
            syns.append(syn)
    rec = {
        "chembl_id": mid,
        "pref_name": data.get("pref_name"),
        "molecule_type": data.get("molecule_type"),
        "max_phase": data.get("max_phase"),
        "therapeutic_flag": data.get("therapeutic_flag"),
        "first_approval": data.get("first_approval"),
        "withdrawn_flag": data.get("withdrawn_flag"),
        "synonyms": sorted(set(syns), key=str.lower),
    }
    cache[mid] = rec
    MOL_CACHE.write_text(json.dumps(cache, indent=2))
    time.sleep(0.12)
    return rec


def main() -> int:
    with httpx.Client(follow_redirects=True) as client:
        target_id = resolve_target(client)
        print(f"\nDRD2 target id = {target_id} "
              f"(expected CHEMBL217: {'YES' if target_id == 'CHEMBL217' else 'NO — verify!'})\n")

        mechs = pull_mechanisms(client, target_id)
        print(f"mechanism rows acting on {target_id}: {len(mechs)}")

        cache = load_mol_cache()
        # one roster entry per unique molecule (a molecule can have >1 mechanism row)
        by_mol: dict[str, dict] = {}
        for m in mechs:
            mid = m.get("molecule_chembl_id")
            if not mid:
                continue
            at = m.get("action_type")
            moa = m.get("mechanism_of_action")
            entry = by_mol.setdefault(mid, {"action_types": set(), "moa": set()})
            if at:
                entry["action_types"].add(at)
            if moa:
                entry["moa"].add(moa)

        print(f"unique molecules hitting {target_id}: {len(by_mol)}\n")
        roster = []
        for i, (mid, info) in enumerate(sorted(by_mol.items())):
            mol = fetch_molecule(client, cache, mid)
            ats = sorted(info["action_types"])
            dirs = {bucket(a) for a in ats} or {UNKNOWN}
            # direction for this molecule on DRD2: single if agree, else 'mixed'
            if len(dirs) == 1:
                direction = next(iter(dirs))
            else:
                direction = "mixed:" + "/".join(sorted(dirs))
            roster.append({
                **mol,
                "drd2_action_types": ats,
                "drd2_direction": direction,
                "drd2_moa": sorted(info["moa"]),
            })
            if (i + 1) % 20 == 0:
                print(f"  resolved {i+1}/{len(by_mol)} molecules")

        ROSTER_OUT.write_text(json.dumps(roster, indent=2))
        print(f"\nwrote {ROSTER_OUT}: {len(roster)} molecules\n")

        # ── summary by direction ──
        from collections import Counter
        dir_counts = Counter(r["drd2_direction"] for r in roster)
        print("direction buckets (DRD2-scoped action_type):")
        for d, n in dir_counts.most_common():
            print(f"  {d:28s} {n}")

        # clinical-stage small molecules, by direction, for candidate selection
        def clinical(r):
            mp = r.get("max_phase")
            try:
                return mp is not None and float(mp) >= 1
            except (TypeError, ValueError):
                return False

        print("\nclinical-stage (max_phase>=1) molecules by direction:")
        for d in (AGONIST, ANTAGONIST, PARTIAL):
            rows = [r for r in roster
                    if r["drd2_direction"] == d and clinical(r)
                    and (r.get("molecule_type") == "Small molecule")]
            rows.sort(key=lambda r: (-(float(r.get("max_phase") or 0)),
                                     (r.get("pref_name") or "zzz")))
            print(f"\n  === {d} (n={len(rows)}) ===")
            for r in rows:
                print(f"    {r['chembl_id']:14s} phase{str(r.get('max_phase')):4s} "
                      f"{(r.get('pref_name') or '?'):24s} "
                      f"AT={','.join(r['drd2_action_types'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
