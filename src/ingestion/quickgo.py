"""QuickGO (EBI) client — Gene Ontology biological-process augmentation.

Used as a fallback for the biology layer when Reactome's curated pathway set
for a gene is sparse, irrelevant, or missing entirely (CRBN being the canonical
example: Reactome has only ``R-HSA-9679191 — Potential therapeutics for SARS``
across every Reactome endpoint, while GO has clean BP annotations for protein
ubiquitination and proteasome-mediated catabolism).

Two-step lookup, both endpoints public, no auth required:

1. ``/services/annotation/search?geneProductId=UniProtKB:{acc}&aspect=biological_process``
   returns the GO term ids annotated for the gene product. Names are *not*
   included in this payload.
2. ``/services/ontology/go/terms/{id1,id2,...}`` returns the human-readable
   names + aspect + obsolete flag. Batched in one call.

Results are cached on disk under ``data/cache/quickgo/{uniprot_acc}.json`` so
rebuilds don't re-hit the API. Same shape as the Reactome cache in
``LINCSClient``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services"


class GOTerm(BaseModel):
    """One Gene Ontology term annotated for a gene product.

    Field names mirror ``ReactomePathway`` so the same context-aware
    ``pathway_ranker.rerank_pathways`` can rank both kinds of biology
    candidates with no special-casing.
    """
    stable_id: str
    display_name: str
    aspect: str = "biological_process"


class QuickGOClient:
    """Async client for QuickGO with on-disk caching.

    Default cache directory mirrors the LINCS/Reactome cache layout:
    ``data/cache/quickgo/{safe_filename(uniprot_acc)}.json``. The cache stores
    the resolved GO terms (with names already filled in) so repeat queries
    are zero-network.
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        timeout: float = 30.0,
    ) -> None:
        self._cache_dir = cache_dir / "quickgo"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

    async def get_terms_for_uniprot(
        self,
        uniprot_acc: str,
        aspect: str = "biological_process",
    ) -> list[GOTerm]:
        """Return biological-process GO terms annotated for a UniProt accession.

        Cached per accession. Returns an empty list when:
          - UniProt accession is empty / malformed.
          - QuickGO has no annotations for that accession.
          - Network call fails (logged at warning; failures don't crash a build).
        """
        if not uniprot_acc:
            return []

        cache_path = self._cache_dir / f"{_safe_filename(uniprot_acc)}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [GOTerm.model_validate(r) for r in raw]

        try:
            terms = await self._fetch_terms(uniprot_acc, aspect)
        except httpx.HTTPError as exc:
            logger.warning(
                "QuickGO lookup failed for %s: %s", uniprot_acc, exc,
            )
            return []

        cache_path.write_text(
            json.dumps([t.model_dump() for t in terms], indent=2)
        )
        return terms

    async def _fetch_terms(
        self, uniprot_acc: str, aspect: str
    ) -> list[GOTerm]:
        # Step 1: annotations endpoint → GO ids (no names).
        gene_product_id = f"UniProtKB:{uniprot_acc}"
        params = {
            "geneProductId": gene_product_id,
            "aspect": aspect,
            "limit": "100",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{QUICKGO_BASE}/annotation/search",
                params=params,
            )
            if resp.status_code in (400, 404):
                return []
            resp.raise_for_status()
            ann_data = resp.json()
            go_ids = sorted({
                r["goId"] for r in ann_data.get("results", [])
                if r.get("goId")
            })
            if not go_ids:
                return []

            # Step 2: ontology terms endpoint → names + aspect + obsolete flag.
            # Batch all ids in one comma-separated call (the endpoint accepts
            # multiple). Drop obsolete terms — they're not useful biology.
            resp2 = await client.get(
                f"{QUICKGO_BASE}/ontology/go/terms/{','.join(go_ids)}",
            )
            resp2.raise_for_status()
            term_data = resp2.json()

        return [
            GOTerm(
                stable_id=t["id"],
                display_name=t.get("name") or t["id"],
                aspect=t.get("aspect") or aspect,
            )
            for t in term_data.get("results", [])
            if t.get("id") and not t.get("isObsolete")
        ]


def _safe_filename(text: str) -> str:
    """Replace path separators / spaces so any id is safe as a filename."""
    return "".join(c if c.isalnum() else "_" for c in text)
