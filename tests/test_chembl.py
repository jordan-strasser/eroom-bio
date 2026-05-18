"""Tests for the ChEMBL REST adapter — non-protein target fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.graph.models import MechanismCategory, TargetType
from src.ingestion.chembl import (
    ChEMBLClient,
    map_moa_text_to_mechanism,
    translate_chembl_target,
)


class TestTranslateChEMBLTarget:
    """ChEMBL target ids translate to namespaced (CHEBI/GO/CL) ids."""

    def test_dna_translates_to_chebi(self):
        result = translate_chembl_target("CHEMBL2311221")
        assert result is not None
        ns_id, name, ttype = result
        assert ns_id == "CHEBI:16991"
        assert name == "DNA"
        assert ttype == TargetType.MACROMOLECULE

    def test_rna_translates_to_chebi(self):
        result = translate_chembl_target("CHEMBL2311222")
        assert result is not None
        ns_id, _, ttype = result
        assert ns_id == "CHEBI:33697"
        assert ttype == TargetType.MACROMOLECULE

    def test_tubulin_translates_to_microtubule(self):
        result = translate_chembl_target("CHEMBL2095182")
        assert result is not None
        ns_id, _, _ = result
        assert ns_id == "CHEBI:35797"

    def test_unknown_returns_none(self):
        """Targets not in the translation table fall through — caller
        treats this as 'no non-protein-target available'."""
        assert translate_chembl_target("CHEMBL_NOT_REAL") is None
        assert translate_chembl_target("") is None


class TestMapMoATextToMechanism:
    """ChEMBL's mechanism_of_action text → MechanismCategory enum."""

    def test_dna_inhibitor_maps_to_dna_damage(self):
        assert (
            map_moa_text_to_mechanism("DNA inhibitor")
            == MechanismCategory.DNA_DAMAGE
        )

    def test_alkylating_maps_to_crosslinking(self):
        """More-specific 'alkylating' beats generic 'DNA inhibitor'."""
        assert (
            map_moa_text_to_mechanism("Alkylating agent")
            == MechanismCategory.DNA_CROSSLINKING
        )

    def test_tubulin_inhibitor_maps_to_microtubule_binding(self):
        assert (
            map_moa_text_to_mechanism("Tubulin inhibitor")
            == MechanismCategory.MICROTUBULE_BINDING
        )

    def test_microtubule_maps_to_microtubule_binding(self):
        assert (
            map_moa_text_to_mechanism("Microtubule stabilizer")
            == MechanismCategory.MICROTUBULE_BINDING
        )

    def test_thymidylate_synthase_maps_to_antimetabolite(self):
        """5-FU class — primarily an antimetabolite via TYMS."""
        assert (
            map_moa_text_to_mechanism("Thymidylate synthase inhibitor")
            == MechanismCategory.ANTIMETABOLITE
        )

    def test_topoisomerase_maps_to_dna_damage(self):
        """Topoisomerase poisons — etoposide, irinotecan."""
        assert (
            map_moa_text_to_mechanism("Topoisomerase II inhibitor")
            == MechanismCategory.DNA_DAMAGE
        )

    def test_unknown_text_returns_none(self):
        assert map_moa_text_to_mechanism("magic pixie dust") is None
        assert map_moa_text_to_mechanism("") is None

    def test_case_insensitive(self):
        assert (
            map_moa_text_to_mechanism("DNA INHIBITOR")
            == MechanismCategory.DNA_DAMAGE
        )


class TestChEMBLClientInferNonProteinTarget:
    """End-to-end inference: drug ChEMBL id → (target, name, type, mech)."""

    @pytest.mark.asyncio
    async def test_carboplatin_infers_dna_target(self, tmp_path):
        """Carboplatin (CHEMBL1351) — alkylator with DNA target."""
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch.object(
            client, "get_drug_mechanisms",
            AsyncMock(return_value=[{
                "action_type": "INHIBITOR",
                "mechanism_of_action": "DNA inhibitor",
                "target_chembl_id": "CHEMBL2311221",
            }]),
        ):
            result = await client.infer_non_protein_target("CHEMBL1351")
        assert result is not None
        tid, name, ttype, mech = result
        assert tid == "CHEBI:16991"
        assert name == "DNA"
        assert ttype == TargetType.MACROMOLECULE
        assert mech == MechanismCategory.DNA_DAMAGE

    @pytest.mark.asyncio
    async def test_paclitaxel_infers_microtubule_target(self, tmp_path):
        """Paclitaxel via mechanism — would normally resolve via OT
        protein path (TUBB) but the fallback path also works."""
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch.object(
            client, "get_drug_mechanisms",
            AsyncMock(return_value=[{
                "action_type": "INHIBITOR",
                "mechanism_of_action": "Tubulin inhibitor",
                "target_chembl_id": "CHEMBL2095182",
            }]),
        ):
            result = await client.infer_non_protein_target("CHEMBL428647")
        assert result is not None
        tid, _, _, mech = result
        assert tid == "CHEBI:35797"
        assert mech == MechanismCategory.MICROTUBULE_BINDING

    @pytest.mark.asyncio
    async def test_no_mechanism_records_returns_none(self, tmp_path):
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch.object(
            client, "get_drug_mechanisms", AsyncMock(return_value=[]),
        ):
            result = await client.infer_non_protein_target("CHEMBL999999")
        assert result is None

    @pytest.mark.asyncio
    async def test_untranslatable_target_returns_none(self, tmp_path):
        """Mechanism record exists but target_chembl_id isn't in the
        translation table — caller treats as no non-protein target."""
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch.object(
            client, "get_drug_mechanisms",
            AsyncMock(return_value=[{
                "action_type": "INHIBITOR",
                "mechanism_of_action": "Acts on X",
                "target_chembl_id": "CHEMBL_NOT_IN_TABLE",
            }]),
        ):
            result = await client.infer_non_protein_target("CHEMBL1")
        assert result is None

    @pytest.mark.asyncio
    async def test_first_translatable_target_wins(self, tmp_path):
        """5-FU has multiple mechanisms (Thymidylate synthase, DNA, RNA).
        The first one with a translatable target_chembl_id is returned —
        multiple records usually re-describe overlapping targets."""
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch.object(
            client, "get_drug_mechanisms",
            AsyncMock(return_value=[
                {
                    "action_type": "INHIBITOR",
                    "mechanism_of_action": "Thymidylate synthase inhibitor",
                    "target_chembl_id": "CHEMBL1952",  # SINGLE PROTEIN, not in table
                },
                {
                    "action_type": "INHIBITOR",
                    "mechanism_of_action": "DNA inhibitor",
                    "target_chembl_id": "CHEMBL2311221",  # DNA, in table
                },
            ]),
        ):
            result = await client.infer_non_protein_target("CHEMBL185")
        assert result is not None
        tid, _, _, mech = result
        assert tid == "CHEBI:16991"
        assert mech == MechanismCategory.DNA_DAMAGE

    @pytest.mark.asyncio
    async def test_empty_chembl_id_returns_none(self, tmp_path):
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        assert await client.infer_non_protein_target("") is None
        assert await client.infer_non_protein_target(None) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_falls_back_to_name_search_when_chembl_id_missing(
        self, tmp_path,
    ):
        """OT can't resolve dacarbazine (chembl_id=None). ChEMBL's own
        molecule search has the drug; the client should use the drug
        name to find chembl_id, then do the mechanism lookup."""
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch.object(
            client, "search_drug_by_name",
            AsyncMock(return_value="CHEMBL476"),
        ), patch.object(
            client, "get_drug_mechanisms",
            AsyncMock(return_value=[{
                "action_type": "INHIBITOR",
                "mechanism_of_action": "DNA inhibitor",
                "target_chembl_id": "CHEMBL2311221",
            }]),
        ):
            result = await client.infer_non_protein_target(
                None, drug_name="dacarbazine",
            )
        assert result is not None
        assert result[0] == "CHEBI:16991"

    @pytest.mark.asyncio
    async def test_no_name_search_when_chembl_id_already_known(
        self, tmp_path,
    ):
        """When OT gives us chembl_id directly, skip the name search —
        avoids an unnecessary HTTP call."""
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        search_mock = AsyncMock()
        with patch.object(
            client, "search_drug_by_name", search_mock,
        ), patch.object(
            client, "get_drug_mechanisms",
            AsyncMock(return_value=[{
                "action_type": "INHIBITOR",
                "mechanism_of_action": "DNA inhibitor",
                "target_chembl_id": "CHEMBL2311221",
            }]),
        ):
            result = await client.infer_non_protein_target(
                "CHEMBL1351", drug_name="carboplatin",
            )
        assert result is not None
        search_mock.assert_not_called()


class TestChEMBLSearchDrugByName:
    """ChEMBL molecule search — fallback when OT can't resolve a drug."""

    @pytest.mark.asyncio
    async def test_exact_match_on_pref_name(self, tmp_path):
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch("src.ingestion.chembl.httpx.AsyncClient") as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: {
                "molecules": [
                    {"molecule_chembl_id": "CHEMBL999", "pref_name": "OTHER"},
                    {"molecule_chembl_id": "CHEMBL476", "pref_name": "DACARBAZINE"},
                ],
            }
            result = await client.search_drug_by_name("dacarbazine")
        assert result == "CHEMBL476"

    @pytest.mark.asyncio
    async def test_falls_back_to_first_hit_when_no_exact_match(
        self, tmp_path,
    ):
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch("src.ingestion.chembl.httpx.AsyncClient") as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: {
                "molecules": [
                    {"molecule_chembl_id": "CHEMBL_TOP", "pref_name": "Foo"},
                    {"molecule_chembl_id": "CHEMBL_LATER", "pref_name": "Bar"},
                ],
            }
            result = await client.search_drug_by_name("baz")
        assert result == "CHEMBL_TOP"

    @pytest.mark.asyncio
    async def test_empty_name_returns_none(self, tmp_path):
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        assert await client.search_drug_by_name("") is None
        assert await client.search_drug_by_name("   ") is None

    @pytest.mark.asyncio
    async def test_no_results_returns_none(self, tmp_path):
        client = ChEMBLClient(cache_path=tmp_path / "chembl.json")
        with patch("src.ingestion.chembl.httpx.AsyncClient") as mock_cls:
            mock_client = mock_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: {"molecules": []}
            result = await client.search_drug_by_name("nonsense")
        assert result is None


class TestChEMBLClientCaching:
    """Lookups are cached on disk — repeated calls don't re-hit the API."""

    @pytest.mark.asyncio
    async def test_mechanism_lookup_cached_to_disk(self, tmp_path):
        cache_path = tmp_path / "chembl.json"
        client = ChEMBLClient(cache_path=cache_path)
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"mechanisms": [{"foo": "bar"}]})

        # We mock the async-with context manager; httpx.AsyncClient is
        # entered/exited inside get_drug_mechanisms.
        with patch("src.ingestion.chembl.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock()
            mock_client.get.return_value.raise_for_status = lambda: None
            mock_client.get.return_value.json = lambda: {"mechanisms": [{"foo": "bar"}]}

            first = await client.get_drug_mechanisms("CHEMBL_TEST")
            second = await client.get_drug_mechanisms("CHEMBL_TEST")

            assert first == [{"foo": "bar"}]
            assert second == [{"foo": "bar"}]
            # Only one HTTP call despite two invocations
            assert mock_client.get.call_count == 1

        # And the cache survives a fresh client instance
        client2 = ChEMBLClient(cache_path=cache_path)
        third = await client2.get_drug_mechanisms("CHEMBL_TEST")
        assert third == [{"foo": "bar"}]
