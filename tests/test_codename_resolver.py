"""Tests for the round-23 codename resolver + RxNorm/PubChem clients."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.graph.codename_resolver import (
    CodenameResolver,
    _pick_inn_like_synonym,
)
from src.ingestion.pubchem import PubChemClient
from src.ingestion.rxnorm import RxNormClient


# ── PubChem client ───────────────────────────────────────────────────────


class TestPubChemClient:
    @pytest.mark.asyncio
    async def test_synonyms_returns_cid_and_list(self, tmp_path):
        client = PubChemClient(cache_path=tmp_path / "pubchem.json")
        with patch(
            "src.ingestion.pubchem.httpx.AsyncClient",
        ) as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.status_code = 200
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: {
                "InformationList": {
                    "Information": [
                        {
                            "CID": 5311497,
                            "Synonym": [
                                "Pembrolizumab",
                                "MK-3475",
                                "Keytruda",
                                "Lambrolizumab",
                            ],
                        }
                    ]
                }
            }
            result = await client.get_synonyms("MK-3475")
        assert result is not None
        assert result["cid"] == 5311497
        assert "Pembrolizumab" in result["synonyms"]
        assert "Keytruda" in result["synonyms"]

    @pytest.mark.asyncio
    async def test_404_returns_none_and_caches(self, tmp_path):
        cache_path = tmp_path / "pubchem.json"
        client = PubChemClient(cache_path=cache_path)
        with patch(
            "src.ingestion.pubchem.httpx.AsyncClient",
        ) as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.status_code = 404
            result = await client.get_synonyms("not-a-real-drug-xyz")
        assert result is None
        # Cached so a second call doesn't re-hit the API.
        data = json.loads(cache_path.read_text())
        assert "not-a-real-drug-xyz" in data
        assert data["not-a-real-drug-xyz"]["result"] is None

    @pytest.mark.asyncio
    async def test_cache_hit_skips_http(self, tmp_path):
        cache_path = tmp_path / "pubchem.json"
        cache_path.write_text(
            json.dumps({
                "mk-3475": {
                    "result": {"cid": 5311497, "synonyms": ["Pembrolizumab"]}
                }
            })
        )
        client = PubChemClient(cache_path=cache_path)
        # No HTTP patch — if the client tried to call out, the test would
        # fail with a connection error from the real httpx.
        result = await client.get_synonyms("MK-3475")
        assert result is not None
        assert result["cid"] == 5311497

    @pytest.mark.asyncio
    async def test_empty_name_returns_none(self, tmp_path):
        client = PubChemClient(cache_path=tmp_path / "pubchem.json")
        assert await client.get_synonyms("") is None
        assert await client.get_synonyms("   ") is None

    @pytest.mark.asyncio
    async def test_http_error_caches_none(self, tmp_path):
        cache_path = tmp_path / "pubchem.json"
        client = PubChemClient(cache_path=cache_path)
        with patch(
            "src.ingestion.pubchem.httpx.AsyncClient",
        ) as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=RuntimeError("boom"))
            result = await client.get_synonyms("MK-3475")
        assert result is None
        data = json.loads(cache_path.read_text())
        assert data["mk-3475"]["result"] is None


# ── RxNorm client ────────────────────────────────────────────────────────


class TestRxNormClient:
    @pytest.mark.asyncio
    async def test_resolve_codename_to_ingredient(self, tmp_path):
        """Top-level RxCUI lookup → IN-related-concept walk → ingredient."""
        client = RxNormClient(cache_path=tmp_path / "rxnorm.json")
        responses = [
            # /rxcui.json?name=MK-3475
            {"idGroup": {"rxnormId": ["1547545"]}},
            # /rxcui/1547545/related.json?tty=IN
            {
                "relatedGroup": {
                    "conceptGroup": [
                        {
                            "tty": "IN",
                            "conceptProperties": [
                                {
                                    "rxcui": "1547544",
                                    "name": "pembrolizumab",
                                }
                            ],
                        }
                    ]
                }
            },
        ]
        with patch(
            "src.ingestion.rxnorm.httpx.AsyncClient",
        ) as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: responses.pop(0)
            result = await client.resolve_to_ingredient("MK-3475")
        assert result is not None
        assert result["ingredient_name"] == "pembrolizumab"
        assert result["rxcui"] == "1547545"
        assert result["ingredient_rxcui"] == "1547544"

    @pytest.mark.asyncio
    async def test_no_rxcui_caches_none(self, tmp_path):
        cache_path = tmp_path / "rxnorm.json"
        client = RxNormClient(cache_path=cache_path)
        with patch(
            "src.ingestion.rxnorm.httpx.AsyncClient",
        ) as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: {"idGroup": {}}
            result = await client.resolve_to_ingredient("XYZ-9999")
        assert result is None
        data = json.loads(cache_path.read_text())
        assert data["xyz-9999"]["result"] is None

    @pytest.mark.asyncio
    async def test_no_ingredient_caches_none(self, tmp_path):
        """RxCUI exists but has no IN-tty relative — treat as unresolved."""
        cache_path = tmp_path / "rxnorm.json"
        client = RxNormClient(cache_path=cache_path)
        responses = [
            {"idGroup": {"rxnormId": ["99999"]}},
            {"relatedGroup": {"conceptGroup": []}},
        ]
        with patch(
            "src.ingestion.rxnorm.httpx.AsyncClient",
        ) as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: responses.pop(0)
            result = await client.resolve_to_ingredient("brand-no-ingredient")
        assert result is None
        data = json.loads(cache_path.read_text())
        assert data["brand-no-ingredient"]["result"] is None

    @pytest.mark.asyncio
    async def test_cache_hit_skips_http(self, tmp_path):
        cache_path = tmp_path / "rxnorm.json"
        cache_path.write_text(
            json.dumps({
                "mk-3475": {
                    "result": {
                        "rxcui": "1547545",
                        "ingredient_name": "pembrolizumab",
                        "ingredient_rxcui": "1547544",
                    }
                }
            })
        )
        client = RxNormClient(cache_path=cache_path)
        result = await client.resolve_to_ingredient("MK-3475")
        assert result is not None
        assert result["ingredient_name"] == "pembrolizumab"

    @pytest.mark.asyncio
    async def test_empty_name_returns_none(self, tmp_path):
        client = RxNormClient(cache_path=tmp_path / "rxnorm.json")
        assert await client.resolve_to_ingredient("") is None
        assert await client.resolve_to_ingredient("   ") is None


# ── _pick_inn_like_synonym ───────────────────────────────────────────────


class TestPickInnLikeSynonym:
    def test_picks_first_lowercase_word(self):
        synonyms = [
            "MK-3475",
            "CAS 1374853-91-4",
            "pembrolizumab",
            "Keytruda",
        ]
        assert _pick_inn_like_synonym(synonyms, exclude="MK-3475") == (
            "pembrolizumab"
        )

    def test_excludes_input_name_case_insensitively(self):
        synonyms = ["MK3475", "pembrolizumab"]
        assert _pick_inn_like_synonym(synonyms, exclude="mk-3475") == (
            "pembrolizumab"
        )

    def test_returns_none_when_only_codes(self):
        synonyms = ["MK-3475", "CHEBI:90551", "S5837", "CAS-1374853-91-4"]
        assert _pick_inn_like_synonym(synonyms, exclude="MK-3475") is None

    def test_skips_too_short(self):
        synonyms = ["mk", "ab", "pembrolizumab"]
        assert _pick_inn_like_synonym(synonyms, exclude="x") == "pembrolizumab"


# ── CodenameResolver: fallback chain ─────────────────────────────────────


class TestCodenameResolverFallbackOrder:
    @pytest.mark.asyncio
    async def test_rxnorm_wins_when_available(self, tmp_path):
        rxnorm = RxNormClient(cache_path=tmp_path / "rxn.json")
        pubchem = PubChemClient(cache_path=tmp_path / "pc.json")
        resolver = CodenameResolver(
            rxnorm=rxnorm,
            pubchem=pubchem,
            unresolved_log_path=tmp_path / "unresolved.jsonl",
        )
        rxnorm_mock = AsyncMock(return_value={
            "rxcui": "1547545",
            "ingredient_name": "pembrolizumab",
            "ingredient_rxcui": "1547544",
        })
        pubchem_mock = AsyncMock()
        with patch.object(rxnorm, "resolve_to_ingredient", rxnorm_mock):
            with patch.object(pubchem, "get_synonyms", pubchem_mock):
                result = await resolver.resolve("MK-3475")
        assert result is not None
        assert result.source == "rxnorm"
        assert result.canonical_name == "pembrolizumab"
        assert result.rxcui == "1547544"
        # PubChem should NOT be called when RxNorm already resolved.
        pubchem_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_pubchem_falls_back_when_rxnorm_misses(self, tmp_path):
        rxnorm = RxNormClient(cache_path=tmp_path / "rxn.json")
        pubchem = PubChemClient(cache_path=tmp_path / "pc.json")
        resolver = CodenameResolver(
            rxnorm=rxnorm,
            pubchem=pubchem,
            unresolved_log_path=tmp_path / "unresolved.jsonl",
        )
        with patch.object(
            rxnorm, "resolve_to_ingredient", AsyncMock(return_value=None),
        ), patch.object(
            pubchem, "get_synonyms",
            AsyncMock(return_value={
                "cid": 5311497,
                "synonyms": ["MK-3475", "pembrolizumab", "Keytruda"],
            }),
        ):
            result = await resolver.resolve("MK-3475")
        assert result is not None
        assert result.source == "pubchem"
        assert result.canonical_name == "pembrolizumab"
        assert result.pubchem_cid == 5311497

    @pytest.mark.asyncio
    async def test_pubchem_matches_existing_node_via_synonyms(self, tmp_path):
        """PubChem synonym list matches an already-known compound."""
        rxnorm = RxNormClient(cache_path=tmp_path / "rxn.json")
        pubchem = PubChemClient(cache_path=tmp_path / "pc.json")
        resolver = CodenameResolver(
            rxnorm=rxnorm,
            pubchem=pubchem,
            unresolved_log_path=tmp_path / "unresolved.jsonl",
        )
        # known_compound_index keys are NORMALIZED names (lowercased,
        # punctuation stripped) → existing compound id.
        known = {"pembrolizumab": "pembrolizumab"}
        with patch.object(
            rxnorm, "resolve_to_ingredient", AsyncMock(return_value=None),
        ), patch.object(
            pubchem, "get_synonyms",
            AsyncMock(return_value={
                "cid": 5311497,
                "synonyms": ["CAS-1374853-91-4", "Pembrolizumab", "Keytruda"],
            }),
        ):
            result = await resolver.resolve(
                "MK-3475", known_compound_index=known,
            )
        assert result is not None
        assert result.source == "pubchem"
        assert result.matched_existing_id == "pembrolizumab"
        assert result.canonical_name == "Pembrolizumab"

    @pytest.mark.asyncio
    async def test_unresolved_logs_to_jsonl(self, tmp_path):
        rxnorm = RxNormClient(cache_path=tmp_path / "rxn.json")
        pubchem = PubChemClient(cache_path=tmp_path / "pc.json")
        log_path = tmp_path / "unresolved.jsonl"
        resolver = CodenameResolver(
            rxnorm=rxnorm,
            pubchem=pubchem,
            unresolved_log_path=log_path,
        )
        with patch.object(
            rxnorm, "resolve_to_ingredient", AsyncMock(return_value=None),
        ), patch.object(
            pubchem, "get_synonyms", AsyncMock(return_value=None),
        ):
            result = await resolver.resolve("ZZZ-9999")
        assert result is None
        assert log_path.exists()
        line = log_path.read_text().strip()
        assert json.loads(line) == {"name": "ZZZ-9999"}

    @pytest.mark.asyncio
    async def test_rxnorm_self_echo_falls_through_to_pubchem(self, tmp_path):
        """RxNorm echoing the input back is not a real resolution."""
        rxnorm = RxNormClient(cache_path=tmp_path / "rxn.json")
        pubchem = PubChemClient(cache_path=tmp_path / "pc.json")
        resolver = CodenameResolver(
            rxnorm=rxnorm,
            pubchem=pubchem,
            unresolved_log_path=tmp_path / "unresolved.jsonl",
        )
        # RxNorm returns the input as the ingredient (input was already
        # the ingredient form). Resolver should NOT treat this as a
        # resolution and should try PubChem.
        with patch.object(
            rxnorm, "resolve_to_ingredient",
            AsyncMock(return_value={
                "rxcui": "1234",
                "ingredient_name": "pembrolizumab",
                "ingredient_rxcui": "1234",
            }),
        ), patch.object(
            pubchem, "get_synonyms",
            AsyncMock(return_value={
                "cid": 5311497,
                "synonyms": ["pembrolizumab", "Keytruda"],
            }),
        ):
            result = await resolver.resolve("pembrolizumab")
        # No real change in canonical_name, so the resolver returns None
        # OR returns a result still labeled the same name. Either way the
        # source can't be rxnorm (input == output is not a resolution).
        assert result is None or result.source != "rxnorm"

    @pytest.mark.asyncio
    async def test_empty_name_returns_none(self, tmp_path):
        resolver = CodenameResolver(
            rxnorm=RxNormClient(cache_path=tmp_path / "rxn.json"),
            pubchem=PubChemClient(cache_path=tmp_path / "pc.json"),
            unresolved_log_path=tmp_path / "unresolved.jsonl",
        )
        assert await resolver.resolve("") is None
        assert await resolver.resolve("   ") is None
