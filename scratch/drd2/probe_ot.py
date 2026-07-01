"""Confirm the POOLING PRECONDITION via Open Targets (the build's real path).

populate.py sets TargetNode.id = OT Ensembl gene id and draws compound->target
(AFFECTS) to it. So two drugs pool on ONE DRD2 node iff OT's get_drug_with_targets
returns the DRD2 gene (ENSG00000149295, approvedSymbol 'DRD2') for both. ChEMBL's
single-protein/family/group fragmentation is irrelevant here — OT normalizes to the
gene. This probe runs the EXACT client method the populator uses.

For each candidate: does OT return DRD2? what chembl_id (for direction-cache
priming)? how many total targets (polypharmacology breadth)?
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.ingestion.opentargets import OpenTargetsClient

HERE = Path(__file__).parent
DRD2_ENSEMBL = "ENSG00000149295"

CANDIDATES = [
    # agonists (movement / endocrine)
    "pramipexole", "ropinirole", "rotigotine", "apomorphine", "cabergoline",
    "bromocriptine", "quinagolide", "pergolide", "lisuride", "piribedil",
    # antagonists (psychiatric / GI)
    "haloperidol", "risperidone", "olanzapine", "metoclopramide",
    "prochlorperazine", "domperidone", "amisulpride", "sulpiride",
    "chlorpromazine", "quetiapine", "clozapine", "paliperidone", "ziprasidone",
    "aripiprazole",  # partial agonist (excluded from core, probed for reference)
]


async def main() -> None:
    ot = OpenTargetsClient()
    out = {}
    for name in CANDIDATES:
        try:
            data = await ot.get_drug_with_targets(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:16s} OT lookup FAILED: {exc}")
            out[name] = {"error": str(exc)}
            continue
        targets = data.get("targets") or []
        syms = [t.get("approved_symbol") for t in targets]
        drd2 = any(
            t.get("target_id") == DRD2_ENSEMBL or t.get("approved_symbol") == "DRD2"
            for t in targets
        )
        out[name] = {
            "chembl_id": data.get("chembl_id"),
            "ot_name": data.get("name"),
            "drd2": drd2,
            "n_targets": len(targets),
            "target_symbols": syms,
        }
        flag = "DRD2 ✓" if drd2 else "DRD2 ✗ MISSING"
        print(f"  {name:16s} {str(data.get('chembl_id')):14s} {flag:16s} "
              f"{len(targets):2d} targets  {syms[:8]}")
    (HERE / "probe_ot.json").write_text(json.dumps(out, indent=2))

    print("\n=== POOLING PRECONDITION: drugs whose OT lookup returns the DRD2 gene ===")
    ok = [n for n, r in out.items() if r.get("drd2")]
    bad = [n for n, r in out.items() if not r.get("drd2") and "error" not in r]
    print(f"  DRD2-resolving ({len(ok)}): {ok}")
    print(f"  NOT resolving  ({len(bad)}): {bad}")


if __name__ == "__main__":
    asyncio.run(main())
