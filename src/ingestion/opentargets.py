"""Open Targets Platform GraphQL API client."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    GraphEdge,
    TargetNode,
)
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# ── GraphQL queries ──────────────────────────────────────────────────────

_SEARCH_TARGET_QUERY = """
query SearchTarget($symbol: String!) {
  search(queryString: $symbol, entityNames: ["target"], page: {size: 1, index: 0}) {
    hits {
      id
      name
      object {
        ... on Target {
          id
          approvedSymbol
          approvedName
        }
      }
    }
  }
}
"""

_SEARCH_DRUG_QUERY = """
query SearchDrug($name: String!) {
  search(queryString: $name, entityNames: ["drug"], page: {size: 1, index: 0}) {
    hits {
      id
      name
      object {
        ... on Drug {
          id
          name
          synonyms
          tradeNames
        }
      }
    }
  }
}
"""

# Search for a drug AND fetch its known targets in one call. Used by the
# trial-driven population pipeline to go compound → target without bulk-
# loading every disease-association row from Open Targets. Targets come
# from ChEMBL-curated mechanisms-of-action, surfaced as one row per
# (mechanism, action) tuple—many rows can list the same target.
#
# Page size is 5 (not 1) so ``get_drug_with_targets`` can prefer a hit
# whose canonical name / synonym / tradeName actually matches the query
# over OT's relevance-rank top hit. Ambiguous names (e.g. "Avastin"
# matching multiple biosimilars) otherwise silently pick the first hit.
_DRUG_WITH_TARGETS_QUERY = """
query DrugWithTargets($name: String!) {
  search(queryString: $name, entityNames: ["drug"], page: {size: 5, index: 0}) {
    hits {
      id
      name
      object {
        ... on Drug {
          id
          name
          synonyms
          tradeNames
          mechanismsOfAction {
            rows {
              actionType
              mechanismOfAction
              targets {
                id
                approvedSymbol
                approvedName
              }
            }
          }
        }
      }
    }
  }
}
"""


_TARGET_ASSOCIATIONS_QUERY = """
query TargetAssociations($ensemblId: String!, $page: Pagination!) {
  target(ensemblId: $ensemblId) {
    associatedDiseases(page: $page) {
      count
      rows {
        disease { id name }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""

_DISEASE_ASSOCIATIONS_QUERY = """
query DiseaseAssociations($efoId: String!, $page: Pagination!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: $page) {
      count
      rows {
        target { id approvedSymbol }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""


# ── Client ───────────────────────────────────────────────────────────────


class OpenTargetsClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                GRAPHQL_URL, json={"query": query, "variables": variables}
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")
            return data["data"]

    async def search_target(self, gene_symbol: str) -> dict[str, Any]:
        data = await self._post(_SEARCH_TARGET_QUERY, {"symbol": gene_symbol})
        hits = data["search"]["hits"]
        if not hits:
            raise KeyError(f"No target found for symbol '{gene_symbol}'")
        obj = hits[0]["object"]
        return {
            "target_id": obj["id"],
            "name": obj["approvedName"],
            "approved_symbol": obj["approvedSymbol"],
        }

    async def search_drug(self, name: str) -> dict[str, Any]:
        """Resolve a drug name to its ChEMBL id + canonical name + aliases.

        Returns ``{"chembl_id": str, "name": str, "aliases": list[str]}``.

        Aliases are filtered + capped at 3:
          - Relationship descriptors ("X component of Y", "<canonical> bms",
            anything containing the canonical name as a substring) are dropped.
          - Punctuation-insensitive dedup ("MDX-1106" and "MDX1106" collapse).
          - One trade-name slot is filled first when a brand name is
            available (regulatory-approved names are cleaner than the
            free-text ChEMBL synonyms list), then the remaining slots
            fill from synonyms in OT order.

        Raises ``KeyError`` when OT returns no match.
        """
        data = await self._post(_SEARCH_DRUG_QUERY, {"name": name})
        hits = data["search"]["hits"]
        if not hits:
            raise KeyError(f"No drug found for name '{name}'")
        obj = hits[0]["object"] or {}
        chembl_id = obj.get("id") or hits[0].get("id")
        canonical_name = obj.get("name") or hits[0].get("name") or name
        synonyms = obj.get("synonyms") or []
        trade_names = obj.get("tradeNames") or []
        return {
            "chembl_id": chembl_id,
            "name": canonical_name,
            "aliases": _clean_aliases(synonyms, trade_names, canonical_name),
        }

    async def get_drug_with_targets(self, name: str) -> dict[str, Any]:
        """Resolve a drug name to its ChEMBL id + ChEMBL-curated targets.

        Returns ``{"chembl_id": str, "name": str, "aliases": list[str],
        "targets": [{"target_id": str, "approved_symbol": str,
        "approved_name": str}]}``. Targets are flattened + deduped across
        the drug's ``mechanismsOfAction.rows``—many rows can list the
        same target (one per mechanism phrasing).

        Disambiguation: queries OT for the top 5 hits and prefers one
        whose canonical name, a synonym, or a tradeName matches ``name``
        case-insensitively. Falls back to the OT-ranked first hit if no
        exact match. Logs at INFO level when overriding the first hit so
        ambiguous matches are auditable.

        Raises ``KeyError`` if no drug matches.
        """
        data = await self._post(_DRUG_WITH_TARGETS_QUERY, {"name": name})
        hits = data["search"]["hits"]
        if not hits:
            raise KeyError(f"No drug found for name '{name}'")
        best_idx = _pick_best_drug_hit(hits, name)
        if best_idx != 0:
            top_obj = (hits[0].get("object") or {})
            chosen_obj = (hits[best_idx].get("object") or {})
            logger.info(
                "OT drug disambiguation for '%s': overriding top hit '%s' "
                "with name-matched hit '%s' (idx=%d)",
                name,
                top_obj.get("name") or hits[0].get("name"),
                chosen_obj.get("name") or hits[best_idx].get("name"),
                best_idx,
            )
        hit = hits[best_idx]
        obj = hit.get("object") or {}
        chembl_id = obj.get("id") or hit.get("id")
        canonical_name = obj.get("name") or hit.get("name") or name
        synonyms = obj.get("synonyms") or []
        trade_names = obj.get("tradeNames") or []
        moa = (obj.get("mechanismsOfAction") or {}).get("rows") or []
        seen_ids: set[str] = set()
        targets: list[dict[str, str]] = []
        for row in moa:
            for t in row.get("targets") or []:
                tid = t.get("id")
                if not tid or tid in seen_ids:
                    continue
                seen_ids.add(tid)
                targets.append({
                    "target_id": tid,
                    "approved_symbol": t.get("approvedSymbol", "") or "",
                    "approved_name": t.get("approvedName", "") or "",
                })
        return {
            "chembl_id": chembl_id,
            "name": canonical_name,
            "aliases": _clean_aliases(synonyms, trade_names, canonical_name),
            "targets": targets,
        }

    async def get_associations(
        self,
        target_id: str,
        disease_id: str | None = None,
        min_score: float = 0.1,
        page_size: int = 500,
    ) -> list[dict[str, Any]]:
        data = await self._post(
            _TARGET_ASSOCIATIONS_QUERY,
            {"ensemblId": target_id, "page": {"size": page_size, "index": 0}},
        )
        rows = data["target"]["associatedDiseases"]["rows"]
        results = []
        for row in rows:
            if row["score"] < min_score:
                continue
            if disease_id and row["disease"]["id"] != disease_id:
                continue
            datatypes = {ds["id"]: ds["score"] for ds in row.get("datatypeScores", [])}
            evidence_count = sum(1 for s in datatypes.values() if s > 0)
            results.append({
                "target_id": target_id,
                "disease_id": row["disease"]["id"],
                "disease_name": row["disease"]["name"],
                "overall_score": row["score"],
                "evidence_count": evidence_count,
                "datatypes": datatypes,
            })
        return results

    async def get_disease_associations(
        self, disease_efo_id: str, page_size: int = 500
    ) -> list[dict[str, Any]]:
        data = await self._post(
            _DISEASE_ASSOCIATIONS_QUERY,
            {"efoId": disease_efo_id, "page": {"size": page_size, "index": 0}},
        )
        rows = data["disease"]["associatedTargets"]["rows"]
        results = []
        for row in rows:
            datatypes = {ds["id"]: ds["score"] for ds in row.get("datatypeScores", [])}
            evidence_count = sum(1 for s in datatypes.values() if s > 0)
            results.append({
                "target_id": row["target"]["id"],
                "target_symbol": row["target"]["approvedSymbol"],
                "disease_id": disease_efo_id,
                "overall_score": row["score"],
                "evidence_count": evidence_count,
                "datatypes": datatypes,
            })
        return results


# ── Drug-hit disambiguation ──────────────────────────────────────────────


def _pick_best_drug_hit(hits: list[dict[str, Any]], queried_name: str) -> int:
    """Return the index of the hit whose names best match ``queried_name``.

    Match policy: the first hit whose canonical name OR any synonym OR
    any tradeName equals ``queried_name`` case-insensitively (after
    punctuation-insensitive normalization) wins. If none match, returns
    0—the OT-ranked top hit.
    """
    if not hits:
        return 0
    target = _DEDUP_NORM.sub("", queried_name.strip().lower())
    if not target:
        return 0
    for i, hit in enumerate(hits):
        obj = hit.get("object") or {}
        candidates: list[str] = []
        for v in (obj.get("name"), hit.get("name")):
            if v:
                candidates.append(v)
        candidates.extend(obj.get("synonyms") or [])
        candidates.extend(obj.get("tradeNames") or [])
        for c in candidates:
            if not c:
                continue
            if _DEDUP_NORM.sub("", c.strip().lower()) == target:
                return i
    return 0


# ── Drug alias cleanup ───────────────────────────────────────────────────


_DEDUP_NORM = re.compile(r"[^a-z0-9]+")
# Phrases that flag relationship descriptors rather than real aliases —
# ChEMBL synonyms occasionally include "Nivolumab component of Opdualag"
# and similar entries that describe a combo, not a name for the drug.
_JUNK_PHRASES = (" of ", " component", " solution", " injection")


def _clean_aliases(
    synonyms: list[str],
    trade_names: list[str],
    canonical_name: str,
    *,
    max_count: int = 3,
) -> list[str]:
    """Filter, dedupe, and cap a drug's alias list.

    See ``OpenTargetsClient.search_drug`` for the full rationale. Trade
    names get one priority slot when present.
    """
    canonical_lower = canonical_name.lower()

    def _is_junk(raw: str) -> bool:
        lower = raw.lower()
        if any(phrase in lower for phrase in _JUNK_PHRASES):
            return True
        # Drop entries that decorate the canonical name with extra text
        # ("Nivolumab bms", "Nivolumab Solution", etc.).
        if canonical_lower in lower and lower != canonical_lower:
            return True
        return False

    def _dedup_key(raw: str) -> str:
        return _DEDUP_NORM.sub("", raw.lower())

    seen: set[str] = {_dedup_key(canonical_name)}
    out: list[str] = []

    # Reserve a trade-name slot first.
    for n in trade_names:
        if not n or _is_junk(n):
            continue
        key = _dedup_key(n)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(n)
        break

    # Fill remaining slots from synonyms in OT's order.
    for n in synonyms:
        if len(out) >= max_count:
            break
        if not n or _is_junk(n):
            continue
        key = _dedup_key(n)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(n)

    return out


# ── Score conversion ─────────────────────────────────────────────────────


def score_to_prior(ot_score: float, evidence_count: int) -> EdgeBeliefState:
    strength = min(evidence_count, 20)
    alpha = 1.0 + ot_score * strength
    beta = 1.0 + (1.0 - ot_score) * strength
    return EdgeBeliefState(alpha=alpha, beta=beta)


# ── Graph population ─────────────────────────────────────────────────────


async def populate_target_disease_edges(
    client: OpenTargetsClient,
    graph: GraphStore,
    disease_efo_id: str,
    indication_id: str,
) -> int:
    """Fetch associations for a disease and add biology_drives edges.

    ``disease_efo_id`` is used only to query Open Targets. The graph
    indication node uses ``indication_id`` (the canonical id created by
    the population pipeline) so we don't double-up on EFO and slug
    representations of the same disease.

    Returns the number of edges added.
    """
    associations = await client.get_disease_associations(disease_efo_id)
    added = 0
    for assoc in associations:
        target_id = assoc["target_id"]  # Ensembl ID—canonical TargetNode id
        # Ensure target node exists
        try:
            graph.get_node(target_id)
        except KeyError:
            symbol = assoc.get("target_symbol", target_id)
            graph.add_node(
                TargetNode(
                    id=target_id,
                    name=symbol,
                    gene_symbol=symbol,
                    metadata={"source": "opentargets"},
                )
            )

        belief = score_to_prior(assoc["overall_score"], assoc["evidence_count"])
        edge = GraphEdge(
            source_id=target_id,
            target_id=indication_id,
            edge_type=EdgeType.BIOLOGY_DRIVES,
            belief=belief,
            metadata={
                "source": "opentargets",
                "efo_id": disease_efo_id,
                "overall_score": assoc["overall_score"],
                "datatypes": assoc["datatypes"],
            },
        )
        graph.add_edge(edge)
        added += 1
    return added
