"""Resolve, for each selected drug, the EXACT direction the build will stamp.

build_graph stamps chain.direction via direction.stamp_directions(), which calls
direction_from_mechanisms(all ChEMBL mechanisms for the compound's OT-resolved
chembl_id) WITHOUT target scoping. So the stamped sign is the consensus action_type
across the drug's whole mechanism list. We reproduce that here (per OT parent
chembl_id) so the roster's +/- labels match what Phase 4 will actually stamp, and
we capture OT aliases for CT.gov intervention querying.

Writes scratch/drd2/direction_map.json. Does NOT touch the production mechanism
cache (that priming is a Phase-4 step, on approval).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.graph.direction import direction_from_mechanisms
from src.ingestion.chembl import ChEMBLClient
from src.ingestion.opentargets import OpenTargetsClient

HERE = Path(__file__).parent

# The 10-drug core (5 agonist / 5 antagonist), all OT-confirmed on the DRD2 gene.
SELECTED = {
    "agonist": ["pramipexole", "ropinirole", "rotigotine", "apomorphine",
                "cabergoline"],
    "antagonist": ["haloperidol", "risperidone", "olanzapine", "metoclopramide",
                   "prochlorperazine"],
}


async def main() -> None:
    ot = OpenTargetsClient()
    # scratch-local ChEMBL cache so we don't write the production cache pre-gate
    chembl = ChEMBLClient(cache_path=HERE / "_chembl_mech_cache.json")
    out = {}
    for expected_dir, names in SELECTED.items():
        for name in names:
            data = await ot.get_drug_with_targets(name)
            chembl_id = data.get("chembl_id")
            aliases = data.get("aliases") or []
            drd2 = any(t.get("approved_symbol") == "DRD2"
                       for t in data.get("targets") or [])
            mechs = await chembl.get_drug_mechanisms(chembl_id) if chembl_id else []
            stamped = direction_from_mechanisms(mechs)
            ats = sorted({m.get("action_type") for m in mechs if m.get("action_type")})
            match = "OK" if stamped == expected_dir else f"!! expected {expected_dir}"
            out[name] = {
                "chembl_id": chembl_id,
                "expected_direction": expected_dir,
                "stamped_direction": stamped,
                "drd2_confirmed": drd2,
                "chembl_action_types": ats,
                "aliases": aliases,
            }
            print(f"  {name:16s} {str(chembl_id):14s} stamp={stamped:11s} "
                  f"drd2={'Y' if drd2 else 'N'}  AT={ats}  [{match}]")
    (HERE / "direction_map.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'direction_map.json'}")
    bad = [n for n, r in out.items()
           if r["stamped_direction"] != r["expected_direction"]]
    if bad:
        print(f"\n*** DIRECTION MISMATCH for: {bad} — investigate before Phase 4 ***")
    else:
        print("\nAll 10 drugs stamp the expected direction. Pooling labels are clean.")


if __name__ == "__main__":
    asyncio.run(main())
