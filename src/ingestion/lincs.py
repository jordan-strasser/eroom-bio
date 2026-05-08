"""LINCS L1000 Touchstone adapter.

Pulls processed Level-5 consensus signatures for the ~2k well-characterized
compounds in the CLUE Touchstone set, then maps each compound's perturbation
fingerprint onto Reactome pathways. Compound-target-pathway triples that are
consistent across cell lines become PRECLINICAL_IN_VITRO evidence on
``mechanism_affects`` edges.

We intentionally avoid raw GEO files and Level-3 data: CLUE already publishes
the moderated z-score consensus signature (top 50 up, top 50 down landmark
genes), which is what we want.

Scope:
- Only compounds whose target is already a TargetNode in the graph.
- Only pathways already represented as a BiologyNode (matched by Reactome
  stable id in ``BiologyNode.pathway_ids``).
- Mechanism is derived from the compound's MOA string ("BRAF inhibitor" →
  ``braf_inhibition``); skipped if no existing MechanismNode matches.

A CLUE API key is required for /perts and /sigs and is read from the
``CLUE_API_KEY`` env var (or passed explicitly).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.graph.models import (
    BiologyNode,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    MechanismCategory,
    MechanismNode,
    MechanismType,
    normalize_entity,
)

# Definitional prior for modulates_via edges synthesized by LINCS. The
# (target, mechanism) link is asserted by the compound's MOA annotation
# itself ("EGFR inhibitor" defines that EGFR is modulated via inhibition),
# so we bias toward "true" without overclaiming. evidence_strength = 4 puts
# trust_weight at 0.4 in the predictor—material but not dominant.
_MODULATES_VIA_PRIOR_ALPHA = 5.0
_MODULATES_VIA_PRIOR_BETA = 1.0
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket

logger = logging.getLogger(__name__)

CLUE_BASE = "https://api.clue.io/api"
REACTOME_BASE = "https://reactome.org/ContentService"
CACHE_DIR = Path("data/cache")

# MOA suffix → canonical MechanismCategory. CLUE MOA strings are short and
# consistent ("BRAF inhibitor", "PARP inhibitor", "dopamine receptor agonist").
# We key off the trailing word and map directly to the controlled vocabulary.
# Generic "inhibitor" defaults to enzyme_inhibition; we promote to
# kinase_inhibition only when the MOA explicitly says "kinase".
_MOA_SUFFIX_TO_CATEGORY: dict[str, MechanismCategory] = {
    "inhibitor": MechanismCategory.ENZYME_INHIBITION,
    "antagonist": MechanismCategory.RECEPTOR_ANTAGONISM,
    "agonist": MechanismCategory.RECEPTOR_AGONISM,
    "blocker": MechanismCategory.RECEPTOR_ANTAGONISM,
    "activator": MechanismCategory.RECEPTOR_AGONISM,
    "degrader": MechanismCategory.PROTEIN_DEGRADATION,
    "modulator": MechanismCategory.OTHER,
}

# Stem variants used to recognize a category from a free-text MOA. "EGFR
# inhibitor" and "EGFR inhibition" both map to enzyme_inhibition.
_TYPE_STEM_TO_CATEGORY: dict[str, MechanismCategory] = {
    "inhibit": MechanismCategory.ENZYME_INHIBITION,
    "antagon": MechanismCategory.RECEPTOR_ANTAGONISM,
    "agonist": MechanismCategory.RECEPTOR_AGONISM,
    "block": MechanismCategory.RECEPTOR_ANTAGONISM,
    "activat": MechanismCategory.RECEPTOR_AGONISM,
    "degrad": MechanismCategory.PROTEIN_DEGRADATION,
    "edit": MechanismCategory.GENE_EDITING,
}

# Bidirectional aliases between common drug-target names (CLUE) and HGNC
# symbols (Open Targets / curated graphs). Hand-curated; only the highest-
# frequency cases that show up in oncology + immuno-oncology Touchstone
# compounds. Symbols are upper-cased at lookup time.
_GENE_ALIASES: dict[str, set[str]] = {
    "PDCD1": {"PD-1", "PD1"},
    "CD274": {"PD-L1", "PDL1", "B7-H1"},
    "CTLA4": {"CD152"},
    "ERBB2": {"HER2", "NEU"},
    "ERBB3": {"HER3"},
    "ERBB4": {"HER4"},
    "MS4A1": {"CD20"},
    "VEGFA": {"VEGF"},
    "KIT": {"C-KIT", "CD117"},
    "FLT1": {"VEGFR1"},
    "KDR": {"VEGFR2"},
    "FLT4": {"VEGFR3"},
    "TNFRSF9": {"4-1BB", "CD137"},
    "TNFRSF4": {"OX40", "CD134"},
    "TNFRSF18": {"GITR"},
    "HAVCR2": {"TIM-3", "TIM3"},
    "LAG3": {"CD223"},
    "IDO1": {"IDO"},
    "PTPN11": {"SHP2", "SHP-2"},
    "MTOR": {"FRAP1"},
    "MAP2K1": {"MEK1"},
    "MAP2K2": {"MEK2"},
    "BCL2L1": {"BCL-XL", "BCLXL"},
    "AKT1": {"AKT", "PKB"},
}


# Core L1000 Phase 1 cell lines → canonical tissue tag. We intentionally use
# a broad tissue label rather than full ontology terms—this is the level
# at which indication-relevance can be reasoned about ("melanoma trial cares
# about skin-lineage signatures"). Cells present in CLUE but not listed here
# fall through to "unknown" and are treated as off-tissue at query time.
CELL_LINE_TO_TISSUE: dict[str, str] = {
    "A375":   "skin",      # malignant melanoma
    "A549":   "lung",      # lung carcinoma (adenocarcinoma)
    "HCC515": "lung",      # NSCLC adenocarcinoma
    "HCC827": "lung",      # NSCLC adenocarcinoma (EGFR-mutant)
    "HEPG2":  "liver",     # hepatocellular carcinoma
    "HT29":   "colon",     # colorectal adenocarcinoma
    "HCT116": "colon",     # colorectal carcinoma
    "MCF7":   "breast",    # breast adenocarcinoma (ER+)
    "PC3":    "prostate",  # prostate adenocarcinoma
    "VCAP":   "prostate",  # prostate carcinoma (androgen-responsive)
    "HA1E":   "kidney",    # immortalized normal kidney
    "ASC":    "adipose",   # adipose-derived stromal
    "NPC":    "neural",    # neural progenitor
    "SHSY5Y": "neural",    # neuroblastoma
    "SKB":    "skin",      # keratinocyte
    "FIBRNPC": "neural",   # fibroblast-derived NPC
    "NEU":    "neural",
}


# Substring-keyed map from indication-name token → set of tissues whose
# cell-line evidence is relevant to that indication. Conservative: when an
# indication doesn't match any keyword, we fall back to "no filtering"
# (i.e. all evidence considered, equivalent to current behavior).
INDICATION_KEYWORD_TO_TISSUES: dict[str, set[str]] = {
    "melanoma":          {"skin"},
    "lung":              {"lung"},
    "nsclc":             {"lung"},
    "small cell lung":   {"lung"},
    "breast":            {"breast"},
    "colorectal":        {"colon"},
    "colon":             {"colon"},
    "rectal":            {"colon"},
    "hepatocellular":    {"liver"},
    "liver":             {"liver"},
    "prostate":          {"prostate"},
    "kidney":            {"kidney"},
    "renal":             {"kidney"},
    "glioblastoma":      {"neural"},
    "glioma":            {"neural"},
    "neuroblastoma":     {"neural"},
    # Pan-tumor indications—accept any tissue's evidence.
    "solid tumor":       {"skin", "lung", "breast", "colon", "liver",
                          "prostate", "kidney", "neural"},
    "advanced solid":    {"skin", "lung", "breast", "colon", "liver",
                          "prostate", "kidney", "neural"},
}


def tissues_for_indication_name(name: str | None) -> set[str]:
    """Return the set of tissues whose cell-line evidence is relevant to an
    indication name. Empty set when no keyword matches → caller should treat
    as "no filtering."
    """
    if not name:
        return set()
    lower = name.lower()
    matched: set[str] = set()
    for keyword, tissues in INDICATION_KEYWORD_TO_TISSUES.items():
        if keyword in lower:
            matched.update(tissues)
    return matched


# ── Models ───────────────────────────────────────────────────────────────


class LincsCompound(BaseModel):
    pert_id: str
    pert_iname: str
    targets: list[str] = Field(default_factory=list)
    # Compounds frequently have multiple MOAs ("Antihypertensive",
    # "Peptidase inhibitor", "Protease inhibitor", …). Each is tried in turn
    # against derive_mechanism_id so any matching mechanism gets edges.
    moas: list[str] = Field(default_factory=list)
    is_touchstone: bool = False


class LincsSignature(BaseModel):
    """A single Level-5 consensus signature (one compound, one cell line)."""

    sig_id: str
    pert_id: str
    cell_id: str
    up_genes: list[str] = Field(default_factory=list)
    dn_genes: list[str] = Field(default_factory=list)


class ReactomePathway(BaseModel):
    stable_id: str
    display_name: str
    species: str | None = None


# ── Client ───────────────────────────────────────────────────────────────


class LINCSClient:
    """CLUE + Reactome client with on-disk JSON caching."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self._api_key = api_key or os.environ.get("CLUE_API_KEY")
        self._timeout = timeout
        self._cache_dir = cache_dir
        self._compounds_cache = cache_dir / "lincs_compounds.json"
        self._sigs_dir = cache_dir / "lincs" / "sigs"
        self._reactome_dir = cache_dir / "lincs" / "reactome"
        self._participants_dir = cache_dir / "lincs" / "participants"
        for d in (self._sigs_dir, self._reactome_dir, self._participants_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- CLUE: compounds -------------------------------------------------

    async def get_touchstone_compounds(self) -> list[LincsCompound]:
        """Touchstone-subset trt_cp compounds from CLUE /perts. Cached."""
        if self._compounds_cache.exists():
            raw = json.loads(self._compounds_cache.read_text())
            return [LincsCompound.model_validate(r) for r in raw]

        records = await self._fetch_compounds_from_clue()
        compounds = [_parse_compound(r) for r in records]
        self._compounds_cache.parent.mkdir(parents=True, exist_ok=True)
        self._compounds_cache.write_text(
            json.dumps([c.model_dump() for c in compounds], indent=2)
        )
        return compounds

    async def _fetch_compounds_from_clue(self) -> list[dict[str, Any]]:
        # Touchstone membership is encoded in pert_icollection, not as a
        # standalone is_touchstone field. The "TS" tag flags the original
        # Touchstone set; "TS_v1.1" is the expansion.
        params = {
            "filter": json.dumps(
                {
                    "where": {"pert_type": "trt_cp", "pert_icollection": "TS"},
                    "fields": {
                        "pert_id": True,
                        "pert_iname": True,
                        "target": True,
                        "moa": True,
                        "pert_icollection": True,
                    },
                }
            ),
        }
        return await self._get("/perts", params)

    # ---- CLUE: signatures ------------------------------------------------

    async def get_consensus_signatures(self, pert_id: str) -> list[LincsSignature]:
        """Per-cell-line consensus signatures for one compound. Cached."""
        cache_path = self._sigs_dir / f"{_safe_filename(pert_id)}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [LincsSignature.model_validate(r) for r in raw]

        records = await self._fetch_signatures_from_clue(pert_id)
        sigs = [s for s in (_parse_signature(r) for r in records) if s is not None]
        cache_path.write_text(json.dumps([s.model_dump() for s in sigs], indent=2))
        return sigs

    async def _fetch_signatures_from_clue(
        self, pert_id: str
    ) -> list[dict[str, Any]]:
        params = {
            "filter": json.dumps(
                {
                    "where": {"pert_id": pert_id},
                    "fields": {
                        "sig_id": True,
                        "pert_id": True,
                        "cell_id": True,
                        "up50_lm": True,
                        "dn50_lm": True,
                    },
                }
            ),
        }
        return await self._get("/sigs", params)

    async def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if not self._api_key:
            raise RuntimeError(
                "CLUE_API_KEY not set; cannot call CLUE API. "
                "Get a key at clue.io/developer."
            )
        headers = {"user_key": self._api_key}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{CLUE_BASE}{path}", params=params, headers=headers
            )
            resp.raise_for_status()
            return resp.json()

    # ---- Reactome --------------------------------------------------------

    async def get_pathways_for_gene(self, gene_symbol: str) -> list[ReactomePathway]:
        """Reactome pathways containing this gene. Cached per gene."""
        cache_path = self._reactome_dir / f"{_safe_filename(gene_symbol)}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [ReactomePathway.model_validate(r) for r in raw]

        records = await self._fetch_reactome_pathways(gene_symbol)
        pathways = [_parse_pathway(r) for r in records]
        cache_path.write_text(json.dumps([p.model_dump() for p in pathways], indent=2))
        return pathways

    async def get_pathway_gene_symbols(self, stable_id: str) -> list[str]:
        """Gene symbols of all proteins in a Reactome pathway. Cached.

        Uses the ``/data/participants/{dbId}`` endpoint, which takes the
        numeric Reactome dbId (= the integer suffix of the stable id, e.g.
        ``R-HSA-1227986`` → ``1227986``) and returns participant entities.
        Each participant has ``refEntities`` with ``displayName`` strings
        like ``"UniProt:P07947 YES1"``—the trailing token is the gene
        symbol.
        """
        cache_path = self._participants_dir / f"{_safe_filename(stable_id)}.json"
        if cache_path.exists():
            return list(json.loads(cache_path.read_text()))
        symbols = await self._fetch_pathway_gene_symbols(stable_id)
        cache_path.write_text(json.dumps(sorted(symbols)))
        return symbols

    async def _fetch_pathway_gene_symbols(self, stable_id: str) -> list[str]:
        m = re.search(r"(\d+)$", stable_id)
        if not m:
            return []
        db_id = m.group(1)
        url = f"{REACTOME_BASE}/data/participants/{db_id}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            logger.warning("Reactome participants failed for %s: %s", stable_id, exc)
            return []
        if resp.status_code in (404, 500):
            return []
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            return []

        symbols: set[str] = set()
        for participant in payload:
            for ref in participant.get("refEntities") or []:
                # displayName format: "UniProt:P07947 YES1"—trailing token
                # after the last whitespace is the gene symbol.
                dn = ref.get("displayName") or ""
                if not dn or " " not in dn:
                    continue
                # Only take protein/gene-product entries; skip ChEBI etc.
                if not (dn.startswith("UniProt:") or "Gene" in (ref.get("schemaClass") or "")):
                    continue
                tail = dn.rsplit(" ", 1)[-1].strip()
                if tail and tail.replace("-", "").replace(".", "").isalnum():
                    symbols.add(tail.upper())
        return sorted(symbols)

    async def _fetch_reactome_pathways(
        self, gene_symbol: str
    ) -> list[dict[str, Any]]:
        # We use the cross-reference mapping endpoint rather than the
        # /identifier/{id}/allForms endpoint: the latter requires an `stId`
        # diagram parameter (it's for "pathways shared with this entity on
        # this diagram"), so it 500s for plain identifiers. CLUE returns
        # L1000 landmark genes as Entrez integer ids, which the NCBI Gene
        # mapping resource accepts directly.
        url = f"{REACTOME_BASE}/data/mapping/NCBI%20Gene/{gene_symbol}/pathways"
        params = {"species": "9606"}  # Homo sapiens
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            logger.warning("Reactome lookup failed for %s: %s", gene_symbol, exc)
            return []
        if resp.status_code in (404, 500):
            # 404 = "No pathways found"—the gene exists but isn't in any
            # Reactome pathway. 500 occasionally appears for deprecated ids.
            return []
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return []


# ── Parsing helpers ──────────────────────────────────────────────────────


def _parse_compound(raw: dict[str, Any]) -> LincsCompound:
    target = raw.get("target")
    if isinstance(target, str):
        targets = [t.strip() for t in re.split(r"[|,]", target) if t.strip()]
    elif isinstance(target, list):
        targets = [str(t) for t in target if t]
    else:
        targets = []

    moa_raw = raw.get("moa")
    if isinstance(moa_raw, list):
        moas = [str(m).strip() for m in moa_raw if m and str(m).strip()]
    elif isinstance(moa_raw, str) and moa_raw.strip():
        moas = [moa_raw.strip()]
    else:
        moas = []

    collections = raw.get("pert_icollection") or []
    is_touchstone = any(
        c in {"TS", "TS_v1.1"} for c in collections
    ) if isinstance(collections, list) else False

    return LincsCompound(
        pert_id=raw["pert_id"],
        pert_iname=raw.get("pert_iname", raw["pert_id"]),
        targets=targets,
        moas=moas,
        is_touchstone=is_touchstone,
    )


def _parse_signature(raw: dict[str, Any]) -> LincsSignature | None:
    sig_id = raw.get("sig_id")
    pert_id = raw.get("pert_id")
    cell_id = raw.get("cell_id")
    if not (sig_id and pert_id and cell_id):
        return None
    return LincsSignature(
        sig_id=sig_id,
        pert_id=pert_id,
        cell_id=cell_id,
        up_genes=list(raw.get("up50_lm") or raw.get("up_genes") or []),
        dn_genes=list(raw.get("dn50_lm") or raw.get("dn_genes") or []),
    )


def _parse_pathway(raw: dict[str, Any]) -> ReactomePathway:
    species_field = raw.get("species")
    if isinstance(species_field, list) and species_field:
        species = species_field[0].get("displayName")
    elif isinstance(species_field, dict):
        species = species_field.get("displayName")
    else:
        species = species_field
    return ReactomePathway(
        stable_id=raw.get("stId", ""),
        display_name=raw.get("displayName", ""),
        species=species,
    )


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


# ── Mechanism derivation ─────────────────────────────────────────────────


def derive_mechanism_id(target_gene: str, moa: str | None) -> str | None:
    """Map a (target, MOA) pair to a canonical MechanismCategory id.

    The mechanism node id is the ``MechanismCategory`` value (e.g.
    ``"kinase_inhibition"``); gene-specificity lives in the
    ``modulates_via`` edge from target → mechanism, not on the mechanism
    node itself. Returns the category value, or None if the MOA doesn't
    map to any recognized category.
    """
    category = _mechanism_category_from_text(moa)
    if category is None:
        return None
    # Promote generic "inhibitor" to kinase_inhibition when the MOA or target
    # text identifies a kinase. Keeps the spec's distinction between
    # kinase_inhibition and enzyme_inhibition without requiring a curated
    # kinase list at the call site.
    if category is MechanismCategory.ENZYME_INHIBITION and (
        "kinase" in (moa or "").lower() or "kinase" in target_gene.lower()
    ):
        category = MechanismCategory.KINASE_INHIBITION
    return category.value


def _mechanism_category_from_text(text: str | None) -> MechanismCategory | None:
    """Best-effort: map free-text MOA / mechanism name → MechanismCategory.

    Tries the trailing word against the suffix table first (fast path for
    CLUE MOAs like "BRAF inhibitor"), then falls back to scanning for any
    known stem ("partial agonist" → receptor_agonism, "BRAF inhibition" →
    enzyme_inhibition).
    """
    if not text or not text.strip():
        return None
    lowered = text.strip().lower()
    last = lowered.split()[-1]
    cat = _MOA_SUFFIX_TO_CATEGORY.get(last)
    if cat is not None:
        return cat
    for stem, category in _TYPE_STEM_TO_CATEGORY.items():
        if stem in lowered:
            return category
    return None


# ── Pathway-hit consistency ──────────────────────────────────────────────


def candidate_pathway_consistencies(
    sigs: list[LincsSignature],
    gene_to_pathways: dict[str, set[str]],
    min_genes_per_sig: int = 3,
) -> dict[str, tuple[float, int]]:
    """For every Reactome pathway hit by ≥1 perturbed gene, return
    ``{pathway_stable_id: (consistency, n_hits)}`` where consistency is
    the fraction of cell-line signatures with ≥``min_genes_per_sig``
    perturbed genes mapping to that pathway.
    """
    if not sigs:
        return {}
    sig_hits: dict[str, int] = {}
    for sig in sigs:
        perturbed = set(sig.up_genes) | set(sig.dn_genes)
        per_pathway: dict[str, int] = {}
        for gene in perturbed:
            for pid in gene_to_pathways.get(gene, set()):
                per_pathway[pid] = per_pathway.get(pid, 0) + 1
        for pid, n in per_pathway.items():
            if n >= min_genes_per_sig:
                sig_hits[pid] = sig_hits.get(pid, 0) + 1
    n = len(sigs)
    return {pid: (count / n, count) for pid, count in sig_hits.items()}


def per_cell_line_pathway_hits(
    sigs: list[LincsSignature],
    gene_to_pathways: dict[str, set[str]],
    min_genes_per_sig: int = 3,
) -> dict[str, list[str]]:
    """Inverse of candidate_pathway_consistencies: which cell lines hit which
    pathways. Returns ``{pathway_stable_id: [cell_id, ...]}``. Lets the
    populator emit one EvidenceRecord per (mechanism, biology, cell_line),
    preserving provenance for indication-conditioned querying later.
    """
    out: dict[str, list[str]] = {}
    for sig in sigs:
        perturbed = set(sig.up_genes) | set(sig.dn_genes)
        per_pathway: dict[str, int] = {}
        for gene in perturbed:
            for pid in gene_to_pathways.get(gene, set()):
                per_pathway[pid] = per_pathway.get(pid, 0) + 1
        for pid, n in per_pathway.items():
            if n >= min_genes_per_sig:
                out.setdefault(pid, []).append(sig.cell_id)
    return out


# ── Graph population ─────────────────────────────────────────────────────


async def populate_lincs_signatures(
    client: LINCSClient,
    graph: GraphStore,
    min_consistency: float = 0.5,
    min_signatures: int = 2,
    create_missing_biology: bool = True,
    create_missing_mechanism: bool = True,
) -> int:
    """Add LINCS-derived ``mechanism_affects`` evidence to the graph.

    For each Touchstone compound whose target resolves (directly or via
    HGNC alias) to an existing TargetNode and whose MOA resolves to a
    canonical mechanism type, fetch consensus signatures across cell lines
    and, for every Reactome pathway hit consistently across those cell
    lines, attach PRECLINICAL_IN_VITRO evidence on a ``mechanism_affects``
    edge.

    Equivalent terms expressed differently are merged before edge creation:
      • Gene-symbol aliases (PD-1↔PDCD1, HER2↔ERBB2, …) collapse to the
        existing TargetNode's canonical symbol.
      • Existing MechanismNodes whose name parses to "{gene} {type}" are
        reused instead of being duplicated under a synthesized id.
      • Reactome stable ids on a BiologyNode's ``pathway_ids`` are reused
        before a new BiologyNode is materialized.

    When ``create_missing_biology`` / ``create_missing_mechanism`` are True
    (the default) the corresponding nodes are auto-materialized for any
    compound-pathway pair that passes the consistency threshold. Set both
    to False for strict mode (only pre-existing nodes receive evidence).

    Returns the number of evidence records added.
    """
    target_lookup = _build_target_lookup(graph)
    mechanism_lookup = _build_mechanism_lookup(graph)
    if not target_lookup:
        return 0
    if not mechanism_lookup and not create_missing_mechanism:
        return 0

    pathway_to_biology = _index_pathway_to_biology(graph)
    gene_pathway_cache: dict[str, set[str]] = {}
    pathway_name_cache: dict[str, str] = {}
    added = 0

    compounds = await client.get_touchstone_compounds()
    for compound in compounds:
        # Resolve every (target, MOA) pair. We walk the compound's CLUE
        # targets, alias-match each one to a graph TargetNode, then for
        # each MOA derive a mechanism id keyed off the *canonical* gene
        # symbol from the matched node. Deduplicated because two MOA
        # phrasings can collapse to the same mechanism.
        matched: list[tuple[str, str, str]] = []  # (mech_id, canonical_gene, moa)
        for raw_symbol in compound.targets:
            target_hit = target_lookup.get(raw_symbol.upper())
            if target_hit is None:
                continue
            _target_id, canonical_gene = target_hit
            for moa in compound.moas:
                category_value = derive_mechanism_id(canonical_gene, moa)
                if category_value is None:
                    continue
                # Canonical mechanism node id = the MechanismCategory value.
                # Many (target, MOA) pairs share a single mechanism node;
                # gene-specificity is captured in the modulates_via edge.
                mech_id = normalize_entity(category_value, "MechanismNode")
                if mech_id not in mechanism_lookup:
                    if not create_missing_mechanism:
                        continue
                    graph.add_node(
                        MechanismNode(
                            id=mech_id,
                            name=mech_id.replace("_", " "),
                            mechanism_type=_category_to_mechanism_type(
                                MechanismCategory(mech_id)
                            ),
                        )
                    )
                    mechanism_lookup[mech_id] = mech_id
                # Always wire target → mechanism. The mechanism may have
                # existed for another target before; the link from THIS
                # target is what closes the path.
                _ensure_modulates_via_edge(graph, _target_id, mech_id)
                if not any(m[0] == mech_id for m in matched):
                    matched.append((mech_id, canonical_gene, moa))
        if not matched:
            continue

        sigs = await client.get_consensus_signatures(compound.pert_id)
        if len(sigs) < min_signatures:
            continue

        await _populate_reactome_caches(
            client, sigs, gene_pathway_cache, pathway_name_cache
        )

        consistencies = candidate_pathway_consistencies(sigs, gene_pathway_cache)
        per_cell_hits = per_cell_line_pathway_hits(sigs, gene_pathway_cache)
        for pathway_id, (consistency, n_hits) in consistencies.items():
            if consistency < min_consistency:
                continue

            biology_id = pathway_to_biology.get(pathway_id)
            if biology_id is None:
                if not create_missing_biology:
                    continue
                biology_id = pathway_id
                graph.add_node(
                    BiologyNode(
                        id=biology_id,
                        name=pathway_name_cache.get(pathway_id) or pathway_id,
                        pathway_ids=[pathway_id],
                    )
                )
                pathway_to_biology[pathway_id] = biology_id

            # Emit one evidence record per (cell line that hit the pathway).
            # Cell-line provenance lives in EvidenceRecord.context so query-
            # time conditioning can downweight off-tissue evidence per
            # indication. Each hit is one positive perturbation observation
            #—strength comes from the count of records (one per hitting
            # cell line) rather than a per-record magnitude.
            hitting_cells = per_cell_hits.get(pathway_id, [])
            for mech_id, _gene, moa in matched:
                _ensure_edge(graph, mech_id, biology_id)
                for cell_id in hitting_cells:
                    tissue = CELL_LINE_TO_TISSUE.get(cell_id, "unknown")
                    evidence = EvidenceRecord(
                        source_id=(
                            f"lincs:{compound.pert_id}:{mech_id}:"
                            f"{biology_id}:{cell_id}"
                        ),
                        source_type=EvidenceType.PRECLINICAL_IN_VITRO,
                        # One cell-line replicate consensus that hit the
                        # pathway → strong directional support for the
                        # mechanism→biology edge in this context.
                        support=SupportBucket.STRONG_SUPPORT.value,
                        timestamp=datetime.now(timezone.utc),
                        provenance_url=(
                            f"https://clue.io/command?q={compound.pert_id}"
                        ),
                        notes=f"{compound.pert_iname} ({moa}) in {cell_id}",
                        context={"cell_line": cell_id, "tissue": tissue},
                    )
                    graph.update_edge_belief(
                        mech_id, biology_id, EdgeType.MECHANISM_AFFECTS, evidence
                    )
                    added += 1

    logger.info("LINCS evidence added: %d records", added)
    return added


async def _populate_reactome_caches(
    client: LINCSClient,
    sigs: list[LincsSignature],
    gene_to_pathways: dict[str, set[str]],
    pathway_names: dict[str, str],
    concurrency: int = 10,
) -> None:
    """Fetch Reactome pathways for any signature gene we haven't seen yet,
    populating both the gene→pathways index and the pathway→display-name map.
    Reactome's API is slow (~1–3s/lookup), so we issue lookups concurrently
    with a small semaphore.
    """
    seen: set[str] = set()
    for sig in sigs:
        seen.update(sig.up_genes)
        seen.update(sig.dn_genes)
    todo = [g for g in seen if g not in gene_to_pathways]
    if not todo:
        return

    sem = asyncio.Semaphore(concurrency)

    async def fetch(gene: str) -> tuple[str, list[ReactomePathway]]:
        async with sem:
            try:
                return gene, await client.get_pathways_for_gene(gene)
            except Exception:
                logger.warning("Reactome lookup raised for %s", gene, exc_info=True)
                return gene, []

    results = await asyncio.gather(*(fetch(g) for g in todo))
    for gene, pathways in results:
        ids: set[str] = set()
        for p in pathways:
            if not p.stable_id:
                continue
            ids.add(p.stable_id)
            if p.stable_id not in pathway_names and p.display_name:
                pathway_names[p.stable_id] = p.display_name
        gene_to_pathways[gene] = ids


def _index_pathway_to_biology(graph: GraphStore) -> dict[str, str]:
    """Reactome stable id → BiologyNode id for nodes that already declare it."""
    out: dict[str, str] = {}
    for node in graph.get_nodes_by_type("BiologyNode"):
        for pid in node.get("pathway_ids", []) or []:
            if pid:
                out[pid] = node["id"]
    return out


def _build_target_lookup(graph: GraphStore) -> dict[str, tuple[str, str]]:
    """Map every plausible gene-symbol spelling → (target_id, canonical_symbol).

    Indexed by ``gene_symbol`` (upper-case), the node ``name`` (upper-case),
    and any aliases declared in ``_GENE_ALIASES``. The first node registered
    for a given key wins—later collisions are ignored, so a direct
    ``gene_symbol`` match always beats an alias match.
    """
    out: dict[str, tuple[str, str]] = {}

    # Build alias closure: each upper-cased symbol → set of equivalents.
    closure: dict[str, set[str]] = {}
    for canonical, aliases in _GENE_ALIASES.items():
        group = {canonical.upper(), *(a.upper() for a in aliases)}
        for sym in group:
            closure[sym] = group

    for node in graph.get_nodes_by_type("TargetNode"):
        gs = (node.get("gene_symbol") or "").upper()
        nm = (node.get("name") or "").upper()
        if not gs:
            continue
        out.setdefault(gs, (node["id"], gs))
        if nm and nm != gs:
            out.setdefault(nm, (node["id"], gs))
        for alias in closure.get(gs, set()):
            out.setdefault(alias, (node["id"], gs))
    return out


def _build_mechanism_lookup(graph: GraphStore) -> dict[str, str]:
    """Map MechanismCategory value → existing MechanismNode id.

    Mechanism nodes are now canonical: their id is a ``MechanismCategory``
    value. We still look up by id so a graph that already has the node
    isn't duplicated.
    """
    out: dict[str, str] = {}
    for node in graph.get_nodes_by_type("MechanismNode"):
        node_id = node["id"]
        out.setdefault(node_id, node_id)
    return out


# How each canonical category maps to the broader MechanismType enum stored
# on the node. Keeps existing ``mechanism_type`` field semantics consistent.
_CATEGORY_TO_MECHANISM_TYPE: dict[MechanismCategory, MechanismType] = {
    MechanismCategory.CHECKPOINT_BLOCKADE: MechanismType.ANTAGONISM,
    MechanismCategory.KINASE_INHIBITION: MechanismType.INHIBITION,
    MechanismCategory.ENZYME_INHIBITION: MechanismType.INHIBITION,
    MechanismCategory.RECEPTOR_ANTAGONISM: MechanismType.ANTAGONISM,
    MechanismCategory.RECEPTOR_AGONISM: MechanismType.AGONISM,
    MechanismCategory.PROTEIN_DEGRADATION: MechanismType.DEGRADATION,
    MechanismCategory.GENE_EDITING: MechanismType.EDITING,
    MechanismCategory.ANTIBODY_DEPENDENT_CYTOTOXICITY: MechanismType.OTHER,
    MechanismCategory.HORMONE_MODULATION: MechanismType.MODULATION,
    MechanismCategory.ANTIMETABOLITE: MechanismType.OTHER,
    MechanismCategory.DNA_DAMAGE: MechanismType.OTHER,
    MechanismCategory.ANGIOGENESIS_INHIBITION: MechanismType.INHIBITION,
    MechanismCategory.IMMUNE_COSTIMULATION: MechanismType.AGONISM,
    MechanismCategory.OTHER: MechanismType.OTHER,
}


def _category_to_mechanism_type(category: MechanismCategory) -> MechanismType:
    return _CATEGORY_TO_MECHANISM_TYPE.get(category, MechanismType.OTHER)


def _ensure_edge(graph: GraphStore, src_id: str, tgt_id: str) -> None:
    if not graph._graph.has_edge(  # noqa: SLF001—no public has_edge on GraphStore
        src_id, tgt_id, key=EdgeType.MECHANISM_AFFECTS.value
    ):
        graph.add_edge(
            GraphEdge(
                source_id=src_id,
                target_id=tgt_id,
                edge_type=EdgeType.MECHANISM_AFFECTS,
                metadata={"source": "lincs"},
            )
        )


def _ensure_modulates_via_edge(
    graph: GraphStore, target_id: str, mechanism_id: str
) -> bool:
    """Add target→mechanism modulates_via with the definitional prior.
    Returns True if a new edge was added; False if one was already present."""
    if graph._graph.has_edge(  # noqa: SLF001
        target_id, mechanism_id, key=EdgeType.MODULATES_VIA.value
    ):
        return False
    graph.add_edge(
        GraphEdge(
            source_id=target_id,
            target_id=mechanism_id,
            edge_type=EdgeType.MODULATES_VIA,
            belief=EdgeBeliefState(
                alpha=_MODULATES_VIA_PRIOR_ALPHA,
                beta=_MODULATES_VIA_PRIOR_BETA,
            ),
            metadata={"source": "lincs"},
        )
    )
    return True


def backfill_modulates_via_for_lincs(graph: GraphStore) -> int:
    """No-op stub kept for API compatibility.

    Under the canonical-id regime, mechanism nodes are MechanismCategory
    values (e.g. ``"kinase_inhibition"``) rather than ``"{gene}_{type}"``,
    so the gene→mechanism mapping cannot be inferred from a node id alone.
    The link is created at population time in the same loop that creates
    the mechanism node. Returns 0.
    """
    return 0


# ── Biology → Indication wiring ──────────────────────────────────────────


# Standalone "indication" tokens that show up in noisy IndicationNode names
# (often trial-arm strings like "placebo plus chemotherapy") and shouldn't
# match disease names from Open Targets. Used to avoid false-positive matches.
_INDICATION_NAME_BLOCKLIST = {
    "placebo", "chemotherapy", "carboplatin", "etoposide", "standard",
    "arm", "control", "intervention", "best", "supportive", "care",
    "regimen", "treatment", "therapy",
}
# Glue/stopwords stripped before checking if remaining tokens are all
# blocklisted ("placebo PLUS chemotherapy" → drops the whole name).
_INDICATION_NAME_STOPWORDS = {
    "and", "or", "with", "plus", "of", "the", "a", "for", "in",
    "to", "vs", "versus",
}


def _normalize_disease_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used for fuzzy
    indication matching."""
    s = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return " ".join(s.split())


def _index_indications_by_name(
    graph: GraphStore,
) -> dict[str, str]:
    """Normalized name → IndicationNode id. Skips names that look like trial
    arm descriptions (no real disease token after stopwords are removed).
    """
    out: dict[str, str] = {}
    for node in graph.get_nodes_by_type("IndicationNode"):
        norm = _normalize_disease_name(node.get("name", ""))
        if not norm:
            continue
        content = set(norm.split()) - _INDICATION_NAME_STOPWORDS
        if not content or content.issubset(_INDICATION_NAME_BLOCKLIST):
            continue
        out.setdefault(norm, node["id"])
    return out


def _match_indication(
    ot_disease_name: str,
    indication_lookup: dict[str, str],
) -> str | None:
    """Best-effort match of an OT disease name to an existing IndicationNode.

    Tries exact normalized match, then bidirectional substring containment.
    Substring is conservative: at least 4 chars to avoid false positives on
    short tokens.
    """
    norm = _normalize_disease_name(ot_disease_name)
    if not norm:
        return None
    if norm in indication_lookup:
        return indication_lookup[norm]
    if len(norm) < 4:
        return None
    for ind_norm, ind_id in indication_lookup.items():
        if len(ind_norm) < 4:
            continue
        if norm in ind_norm or ind_norm in norm:
            return ind_id
    return None


async def populate_biology_indication_edges(
    lincs_client: "LINCSClient",
    ot_client,  # OpenTargetsClient—typed loose to avoid import cycle
    graph: GraphStore,
    only_lincs_created: bool = True,
    min_pathway_genes: int = 3,
    min_genes_per_indication: int = 2,
    min_disease_score: float = 0.3,
    ot_concurrency: int = 8,
) -> int:
    """Connect dangling BiologyNodes to existing IndicationNodes.

    For each BiologyNode that carries Reactome stable ids, fetch the
    pathway's protein members from Reactome, look up each member's disease
    associations in Open Targets, then create ``biology_drives`` edges to
    any IndicationNode whose name fuzzy-matches an associated disease.

    Two thresholds gate edge creation:
      • ``min_pathway_genes``—pathway as a whole must have at least this
        many genes resolving to OT (otherwise we don't trust *any* of its
        edges).
      • ``min_genes_per_indication``—each (pathway, indication) edge must
        have at least this many co-supporting genes. Single-gene edges are
        just a target→disease association in disguise and double-count
        evidence already present at the target layer; the whole point of
        going through pathway membership is to require multi-gene support.

    Belief is computed by ``score_to_prior(mean_ot_score, n_supporting_genes)``
    so pathways supported by many genes get stronger priors.

    ``only_lincs_created``: when True, restrict to BiologyNodes whose id
    starts with ``R-HSA-`` (the materialized format LINCS uses). Set False
    to include curated BiologyNodes that already declare ``pathway_ids``.

    Returns the number of biology_drives edges added.
    """
    from src.ingestion.opentargets import score_to_prior

    indication_lookup = _index_indications_by_name(graph)
    if not indication_lookup:
        return 0

    biology_nodes = [
        n for n in graph.get_nodes_by_type("BiologyNode")
        if n.get("pathway_ids")
        and (not only_lincs_created or n["id"].startswith("R-HSA-"))
    ]
    if not biology_nodes:
        return 0

    # Resolve unique gene symbols across all pathways once, in parallel.
    pathway_to_genes: dict[str, set[str]] = {}
    all_symbols: set[str] = set()
    for bnode in biology_nodes:
        genes: set[str] = set()
        for stable_id in bnode["pathway_ids"]:
            members = await lincs_client.get_pathway_gene_symbols(stable_id)
            genes.update(members)
        pathway_to_genes[bnode["id"]] = genes
        all_symbols.update(genes)

    # OT lookups: symbol → ENSG, then ENSG → associations. Cached per run.
    sem = asyncio.Semaphore(ot_concurrency)

    async def resolve_symbol(symbol: str) -> tuple[str, str | None]:
        async with sem:
            try:
                info = await ot_client.search_target(symbol)
                return symbol, info["target_id"]
            except (KeyError, Exception) as exc:  # noqa: BLE001
                logger.debug("OT search_target(%s) failed: %s", symbol, exc)
                return symbol, None

    resolved = await asyncio.gather(
        *(resolve_symbol(s) for s in sorted(all_symbols))
    )
    symbol_to_ensg: dict[str, str | None] = dict(resolved)

    async def fetch_assocs(ensg: str) -> tuple[str, list[dict[str, Any]]]:
        async with sem:
            try:
                rows = await ot_client.get_associations(
                    ensg, min_score=min_disease_score
                )
                return ensg, rows
            except Exception as exc:  # noqa: BLE001
                logger.debug("OT get_associations(%s) failed: %s", ensg, exc)
                return ensg, []

    unique_ensgs = {e for e in symbol_to_ensg.values() if e}
    fetched = await asyncio.gather(*(fetch_assocs(e) for e in unique_ensgs))
    ensg_to_assocs: dict[str, list[dict[str, Any]]] = dict(fetched)

    # Aggregate per (biology_node, indication): collect supporting OT scores
    # from each pathway gene that maps to that disease.
    added = 0
    for bnode in biology_nodes:
        biology_id = bnode["id"]
        per_indication: dict[str, list[float]] = {}
        per_indication_evidence: dict[str, int] = {}

        contributing_genes = 0
        for symbol in pathway_to_genes[biology_id]:
            ensg = symbol_to_ensg.get(symbol)
            if not ensg:
                continue
            assocs = ensg_to_assocs.get(ensg, [])
            if not assocs:
                continue
            contributing_genes += 1
            for assoc in assocs:
                ind_id = _match_indication(
                    assoc["disease_name"], indication_lookup
                )
                if ind_id is None:
                    continue
                per_indication.setdefault(ind_id, []).append(assoc["overall_score"])
                per_indication_evidence[ind_id] = (
                    per_indication_evidence.get(ind_id, 0) + assoc["evidence_count"]
                )

        if contributing_genes < min_pathway_genes:
            continue

        for ind_id, scores in per_indication.items():
            if len(scores) < min_genes_per_indication:
                continue
            if graph._graph.has_edge(  # noqa: SLF001
                biology_id, ind_id, key=EdgeType.BIOLOGY_DRIVES.value
            ):
                continue
            mean_score = sum(scores) / len(scores)
            evidence_total = per_indication_evidence[ind_id]
            belief = score_to_prior(mean_score, evidence_total)
            graph.add_edge(
                GraphEdge(
                    source_id=biology_id,
                    target_id=ind_id,
                    edge_type=EdgeType.BIOLOGY_DRIVES,
                    belief=belief,
                    metadata={
                        "source": "reactome+opentargets",
                        "supporting_genes": len(scores),
                        "mean_ot_score": round(mean_score, 4),
                    },
                )
            )
            added += 1

    logger.info("biology_drives edges added: %d", added)
    return added
