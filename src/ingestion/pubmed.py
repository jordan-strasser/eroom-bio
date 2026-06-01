"""NCBI PubMed adapter (E-utilities efetch) — headless abstract fetch.

The Claude PubMed MCP is ideal for interactive agent-side curation but is NOT
available inside a headless ``build_graph`` run, so the production safety-
enrichment producer fetches abstracts via NCBI E-utilities REST. Plain HTTP, no
auth required; an optional ``NCBI_API_KEY`` raises the rate limit from 3 to 10
req/s for scale builds.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedClient:
    """Fetch PubMed abstract text by PMID via E-utilities ``efetch``."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        api_key: str | None = None,
        concurrency: int = 3,
    ) -> None:
        self._timeout = timeout
        # NCBI allows 3 req/s anonymously, 10 with a key. The semaphore caps
        # in-flight requests; pair it with a small batch (<=3 PMIDs/trial).
        self._api_key = api_key or os.environ.get("NCBI_API_KEY")
        self._sem = asyncio.Semaphore(concurrency)

    async def fetch_abstracts(self, pmids: list[str]) -> dict[str, str]:
        """``pmid -> abstract text`` (``rettype=abstract, retmode=text``).

        The text form is intentional: a focused LLM call reads it, so we don't
        need to parse PubMed XML. Ids that error or come back empty are skipped;
        returns ``{}`` if nothing resolved (caller treats that as "no signal").
        """
        out: dict[str, str] = {}
        if not pmids:
            return out

        async with httpx.AsyncClient(timeout=self._timeout) as client:

            async def _one(pmid: str) -> None:
                params = {
                    "db": "pubmed", "id": pmid,
                    "rettype": "abstract", "retmode": "text",
                }
                if self._api_key:
                    params["api_key"] = self._api_key
                async with self._sem:
                    try:
                        resp = await client.get(EFETCH_URL, params=params)
                        resp.raise_for_status()
                    except Exception:  # noqa: BLE001
                        logger.warning("PubMed efetch failed for %s", pmid, exc_info=True)
                        return
                text = (resp.text or "").strip()
                if text:
                    out[pmid] = text

            await asyncio.gather(*(_one(p) for p in pmids))
        return out
