"""Phase 4 construction check — does the built graph realize the design?

Verifies on data/exports/drd2_subset_annotated.json:
  1. The DRD2 node exists (gene_symbol DRD2, id ENSG00000149295).
  2. POOLING: both directions (agonist + antagonist) have compound->DRD2 AFFECTS
     edges AND chains walking through DRD2 — the in-graph pooling precondition.
  3. Per drug: chains total, chains on DRD2, directions stamped, indications.
  4. FLAGS: any of our 10 drugs whose chains never touch DRD2 (mis-attached), or
     whose stamped direction is unknown / wrong sign.

Read-only. Prints a report; exits 0 always (diagnostic).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from src.graph.store import GraphStore

SNAP = "data/exports/drd2_subset_annotated.json"
DRD2_ENSEMBL = "ENSG00000149295"

DIRECTION = {
    "pramipexole": "agonist", "ropinirole": "agonist", "rotigotine": "agonist",
    "apomorphine": "agonist", "cabergoline": "agonist",
    "haloperidol": "antagonist", "risperidone": "antagonist",
    "olanzapine": "antagonist", "metoclopramide": "antagonist",
    "prochlorperazine": "antagonist",
}

# spelling variants (LLM extraction drops trailing 'e', uses EU INN) + brand names
ALIASES = {
    "pramipexole": ["pramipexole", "pramipexol", "mirapex", "sifrol"],
    "ropinirole": ["ropinirole", "ropinirol", "requip"],
    "rotigotine": ["rotigotine", "rotigotin", "neupro"],
    "apomorphine": ["apomorphine", "apomorphin", "apokyn"],
    "cabergoline": ["cabergoline", "cabergolin", "dostinex"],
    "haloperidol": ["haloperidol", "haldol"],
    "risperidone": ["risperidone", "risperidon", "risperdal"],
    "olanzapine": ["olanzapine", "olanzapin", "zyprexa"],
    "metoclopramide": ["metoclopramide", "metoclopramid", "reglan"],
    "prochlorperazine": ["prochlorperazine", "prochlorperazin", "compazine",
                         "stemetil"],
}


def which_drug(name: str, cid: str) -> str | None:
    hay = f"{name} {cid}".lower()
    for d, al in ALIASES.items():
        if any(a in hay for a in al):
            return d
    return None


def main() -> int:
    g = GraphStore()
    g.import_snapshot(SNAP)
    nodes = g._graph.nodes  # noqa: SLF001

    # ── locate DRD2 node ──
    drd2_ids = [nid for nid, d in nodes(data=True)
                if d.get("node_type") == "TargetNode"
                and (d.get("gene_symbol") == "DRD2" or d.get("name") == "DRD2"
                     or nid == DRD2_ENSEMBL)]
    print(f"DRD2 node(s): {drd2_ids}")
    if not drd2_ids:
        print("*** DRD2 NODE ABSENT — pooling precondition FAILED in build ***")
        # list dopamine targets that DID appear
        dopa = [(nid, d.get("gene_symbol") or d.get("name"))
                for nid, d in nodes(data=True)
                if d.get("node_type") == "TargetNode"
                and "DRD" in str(d.get("gene_symbol") or d.get("name") or "")]
        print("   dopamine target nodes present:", dopa)
        return 0
    drd2 = drd2_ids[0]
    is_single = len(drd2_ids) == 1
    print(f"single DRD2 node (no fragmentation): {is_single}  id={drd2}\n")

    # ── edge-level pooling: compound -> DRD2 AFFECTS, by source direction ──
    cid_drug: dict[str, str] = {}
    for nid, d in nodes(data=True):
        if d.get("node_type") == "InterventionNode":
            drug = which_drug(d.get("name", ""), nid)
            if drug:
                cid_drug[nid] = drug
    binders = defaultdict(set)  # direction -> {drug}
    in_edges = 0
    for u, v, key, data in g._graph.edges(data=True, keys=True):
        if v == drd2 and cid_drug.get(u):
            in_edges += 1
            binders[DIRECTION[cid_drug[u]]].add(cid_drug[u])
    print(f"compound->DRD2 AFFECTS edges from our drugs: {in_edges}")
    print(f"  agonist binders:    {sorted(binders['agonist'])}")
    print(f"  antagonist binders: {sorted(binders['antagonist'])}")
    pooled = bool(binders["agonist"]) and bool(binders["antagonist"])
    print(f"  >>> BOTH directions bind the same DRD2 node: {pooled}\n")

    # ── chain-level: per drug, chains total / on-DRD2 / directions / indications ──
    per_drug_total = Counter()
    per_drug_drd2 = Counter()
    per_drug_dirs = defaultdict(Counter)
    per_drug_inds = defaultdict(Counter)
    per_drug_tgts = defaultdict(Counter)
    drd2_chain_dirs = Counter()
    drd2_chain_inds = Counter()
    for nct, ts in g.trial_subgraphs.items():
        for ch in ts.chains:
            drug = cid_drug.get(ch.compound_id) or which_drug("", ch.compound_id)
            if not drug:
                continue
            per_drug_total[drug] += 1
            per_drug_dirs[drug][ch.direction] += 1
            gene = (nodes[ch.target_id].get("gene_symbol") or nodes[ch.target_id].get("name")
                    if ch.target_id in nodes else ch.target_id)
            per_drug_tgts[drug][gene] += 1
            ind = nodes[ch.indication_id].get("name") if ch.indication_id in nodes else ch.indication_id
            per_drug_inds[drug][ind] += 1
            if ch.target_id == drd2:
                per_drug_drd2[drug] += 1
                drd2_chain_dirs[ch.direction] += 1
                drd2_chain_inds[ind] += 1

    print("per-drug chains (total | on-DRD2 | directions | top indications):")
    flags = []
    for drug, exp in DIRECTION.items():
        tot = per_drug_total[drug]
        ond = per_drug_drd2[drug]
        dirs = dict(per_drug_dirs[drug])
        inds = ", ".join(f"{k}:{v}" for k, v in per_drug_inds[drug].most_common(3))
        mark = ""
        if tot == 0:
            mark = "  << NO CHAINS"; flags.append(f"{drug}: no chains built")
        elif ond == 0:
            mark = "  << MIS-ATTACHED (no DRD2 chain)"; flags.append(f"{drug}: 0/{tot} chains on DRD2")
        # direction sanity: dominant stamped dir should equal expected
        stamped = per_drug_dirs[drug].most_common(1)
        if stamped and stamped[0][0] != exp and tot:
            mark += f"  << DIR {stamped[0][0]}!=exp {exp}"
            flags.append(f"{drug}: stamped {stamped[0][0]} expected {exp}")
        tgts = ", ".join(f"{k}:{v}" for k, v in per_drug_tgts[drug].most_common(3))
        print(f"  {drug:16s} {exp:10s} tot={tot:3d} drd2={ond:3d} dirs={dirs}{mark}")
        print(f"                     targets: {tgts}")
        print(f"                     indications: {inds}")

    print(f"\nDRD2-walking chains by direction: {dict(drd2_chain_dirs)}")
    print(f"DRD2-walking chains by indication: {dict(drd2_chain_inds)}")

    print("\n=== CONSTRUCTION CHECK SUMMARY ===")
    print(f"  single DRD2 node:               {is_single}")
    print(f"  both directions bind DRD2:      {pooled}")
    print(f"  DRD2 chains both directions:    "
          f"{bool(drd2_chain_dirs.get('agonist')) and bool(drd2_chain_dirs.get('antagonist'))}")
    print(f"  flags ({len(flags)}):")
    for f in flags:
        print(f"    - {f}")
    if not flags:
        print("    none — every drug built chains that walk through the DRD2 node "
              "with the expected direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
