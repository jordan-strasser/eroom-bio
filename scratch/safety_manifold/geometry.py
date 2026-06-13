"""Safety-manifold geometry: load a snapshot and build the two manifolds.

Compound manifold  : Morgan/ECFP4 fingerprint over ChEMBL SMILES (RDKit).
Target manifold    : Reactome/GO pathway co-membership (in-graph; the
                     MechanismNode id IS the pathway id, reached via
                     ``modulates_via`` out of the target).

Plus the AE substrate each manifold borrows over:
  causes_ae           compound -> AE  (per-trial)
  target_associated_ae target  -> AE  (propagated from binding compounds)

Read-only over an annotated export. No graph mutation here; the borrowing
kernels (Phase 2) consume what this exposes. SMILES come from the vendored
``data/cache/chembl_smiles.json`` (resolved once from ChEMBL by chembl_id).
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMILES_CACHE = os.path.join(REPO, "data", "cache", "chembl_smiles.json")

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    _RDKIT = True
    _MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except Exception:  # pragma: no cover
    _RDKIT = False
    _MORGAN = None


def morgan_fp(smiles: str):
    """ECFP4-style Morgan fingerprint (radius 2, 2048 bits) or None."""
    if not _RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN.GetFingerprint(mol)


def tanimoto(fp_a, fp_b) -> float:
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


@dataclass
class Geometry:
    snapshot_path: str
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    # compound manifold
    compound_smiles: dict = field(default_factory=dict)   # cid -> smiles
    compound_fp: dict = field(default_factory=dict)        # cid -> fingerprint
    compound_type: dict = field(default_factory=dict)      # cid -> molecule_type
    # target manifold
    target_pathways: dict = field(default_factory=dict)    # tid -> frozenset(pathway ids)
    # AE substrate
    causes_ae: dict = field(default_factory=dict)          # cid -> {ae_id: belief_dict}
    target_ae: dict = field(default_factory=dict)          # tid -> {ae_id: belief_dict}
    binds: dict = field(default_factory=dict)              # cid -> set(tid)  (affects)
    target_compounds: dict = field(default_factory=dict)   # tid -> set(cid)

    def node(self, nid):
        return self.nodes.get(nid, {})


def _belief_mean(belief: dict) -> float:
    a = belief.get("alpha", 1.0)
    b = belief.get("beta", 1.0)
    return a / (a + b) if (a + b) else 0.5


def _belief_strength(belief: dict) -> float:
    a = belief.get("alpha", 1.0)
    b = belief.get("beta", 1.0)
    return max(0.0, (a + b) - 2.0)


def load_geometry(snapshot_path: str) -> Geometry:
    with open(snapshot_path) as f:
        snap = json.load(f)
    G = snap["graph"]
    nodes = {n["id"]: n for n in G["nodes"] if "id" in n}
    edges = G["edges"]
    geo = Geometry(snapshot_path=snapshot_path, nodes=nodes, edges=edges)

    smiles_cache = {}
    if os.path.exists(SMILES_CACHE):
        with open(SMILES_CACHE) as f:
            smiles_cache = json.load(f)

    target_ids = {nid for nid, n in nodes.items()
                  if n.get("node_type") == "TargetNode"}
    mech_ids = {nid for nid, n in nodes.items()
                if n.get("node_type") == "MechanismNode"}

    # Compound manifold: SMILES via chembl_id -> Morgan fp.
    for nid, n in nodes.items():
        if n.get("node_type") != "InterventionNode":
            continue
        cid = n.get("chembl_id")
        rec = smiles_cache.get(cid) if cid else None
        smi = (rec or {}).get("smiles")
        geo.compound_type[nid] = (rec or {}).get("type")
        if smi:
            fp = morgan_fp(smi)
            if fp is not None:
                geo.compound_smiles[nid] = smi
                geo.compound_fp[nid] = fp

    # Target manifold: pathway membership = mechanism ids reached via modulates_via.
    t2pw = defaultdict(set)
    for e in edges:
        if e.get("edge_type") == "modulates_via" and e["source"] in target_ids:
            mid = e["target"]
            if mid in mech_ids:
                # the mechanism id is the pathway id (R-HSA-*/GO:*)
                if mid.startswith("R-HSA") or mid.startswith("GO:"):
                    t2pw[e["source"]].add(mid)
                # also fold any explicit pathway_ids if present
                for pw in (nodes[mid].get("pathway_ids") or []):
                    t2pw[e["source"]].add(pw)
    geo.target_pathways = {t: frozenset(pw) for t, pw in t2pw.items()}

    # AE substrate.
    causes = defaultdict(dict)
    tae = defaultdict(dict)
    binds = defaultdict(set)
    tcomp = defaultdict(set)
    for e in edges:
        et = e.get("edge_type")
        if et == "causes_ae":
            causes[e["source"]][e["target"]] = e.get("belief", {})
        elif et == "target_associated_ae":
            tae[e["source"]][e["target"]] = e.get("belief", {})
        elif et == "affects":
            if e["target"] in target_ids:
                binds[e["source"]].add(e["target"])
                tcomp[e["target"]].add(e["source"])
    geo.causes_ae = dict(causes)
    geo.target_ae = dict(tae)
    geo.binds = dict(binds)
    geo.target_compounds = dict(tcomp)
    return geo


# --- AE-profile vectors (for alignment correlations) -----------------------

def compound_ae_vector(geo: Geometry, cid: str, min_strength: float = 0.0):
    """{ae_id: belief_mean} for a compound's causes_ae edges with evidence."""
    out = {}
    for ae, bel in geo.causes_ae.get(cid, {}).items():
        if _belief_strength(bel) < min_strength:
            continue
        out[ae] = _belief_mean(bel)
    return out


def target_ae_vector(geo: Geometry, tid: str, min_strength: float = 0.0):
    out = {}
    for ae, bel in geo.target_ae.get(tid, {}).items():
        if ae.startswith("AE:soc:"):
            continue  # keep PT-level for profile comparison
        if _belief_strength(bel) < min_strength:
            continue
        out[ae] = _belief_mean(bel)
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cosine(va: dict, vb: dict) -> float:
    keys = set(va) | set(vb)
    if not keys:
        return 0.0
    dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def weighted_jaccard(va: dict, vb: dict) -> float:
    """min/max weighted Jaccard over AE-incidence vectors."""
    keys = set(va) | set(vb)
    num = sum(min(va.get(k, 0.0), vb.get(k, 0.0)) for k in keys)
    den = sum(max(va.get(k, 0.0), vb.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0
