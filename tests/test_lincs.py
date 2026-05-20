"""Tests for the LINCS L1000 Touchstone adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.graph.models import (
    BiologyNode,
    EdgeType,
    MechanismNode,
    MechanismType,
    TargetNode,
)
from src.graph.store import GraphStore
from src.ingestion.lincs import (
    LINCSClient,
    LincsCompound,
    LincsSignature,
    ReactomePathway,
    _parse_compound,
    _parse_pathway,
    _parse_signature,
    candidate_pathway_consistencies,
    derive_mechanism_id,
    populate_lincs_signatures,
)


# ── Mock CLUE / Reactome payloads ────────────────────────────────────────


MOCK_PERTS = [
    {
        "pert_id": "BRD-K12345678",
        "pert_iname": "vemurafenib",
        "target": ["BRAF"],
        "moa": ["BRAF inhibitor"],
        "pert_icollection": ["BIOA", "TS", "TS_v1.1"],
    },
    {
        "pert_id": "BRD-K87654321",
        "pert_iname": "gefitinib",
        "target": "EGFR",
        "moa": ["EGFR inhibitor"],
        "pert_icollection": ["TS"],
    },
]

MOCK_SIGS = [
    {
        "sig_id": "SIG_1",
        "pert_id": "BRD-K12345678",
        "cell_id": "A375",
        "up50_lm": ["DUSP6", "SPRY2", "ETV5"],
        "dn50_lm": ["MYC", "CCND1"],
    },
    {
        "sig_id": "SIG_2",
        "pert_id": "BRD-K12345678",
        "cell_id": "MCF7",
        "up50_lm": ["DUSP6", "SPRY2"],
        "dn50_lm": ["MYC", "FOS"],
    },
]

MOCK_REACTOME = [
    {"stId": "R-HSA-5673001", "displayName": "RAF/MAP kinase cascade"},
    {"stId": "R-HSA-9006934", "displayName": "Signaling by Receptor Tyrosine Kinases"},
]


# ── Parsing tests ────────────────────────────────────────────────────────


class TestParseCompound:
    def test_target_as_list(self):
        c = _parse_compound(MOCK_PERTS[0])
        assert c.pert_id == "BRD-K12345678"
        assert c.targets == ["BRAF"]
        assert c.moas == ["BRAF inhibitor"]
        assert c.is_touchstone is True

    def test_target_as_pipe_string(self):
        c = _parse_compound(
            {"pert_id": "BRD-X", "pert_iname": "x", "target": "EGFR|ERBB2"}
        )
        assert c.targets == ["EGFR", "ERBB2"]

    def test_target_missing(self):
        c = _parse_compound({"pert_id": "BRD-Y", "pert_iname": "y"})
        assert c.targets == []

    def test_multiple_moas(self):
        c = _parse_compound(
            {
                "pert_id": "BRD-Z",
                "pert_iname": "z",
                "target": ["DRD2"],
                "moa": ["Antihypertensive", "Dopamine receptor antagonist"],
                "pert_icollection": ["TS"],
            }
        )
        assert c.moas == ["Antihypertensive", "Dopamine receptor antagonist"]
        assert c.is_touchstone is True

    def test_moa_as_string_falls_back(self):
        # The CLUE API returns lists, but be lenient if a caller passes a str.
        c = _parse_compound(
            {"pert_id": "BRD-A", "pert_iname": "a", "moa": "BRAF inhibitor"}
        )
        assert c.moas == ["BRAF inhibitor"]

    def test_not_touchstone_when_collection_missing(self):
        c = _parse_compound(
            {
                "pert_id": "BRD-B",
                "pert_iname": "b",
                "moa": ["x inhibitor"],
                "pert_icollection": ["BIOA"],
            }
        )
        assert c.is_touchstone is False


class TestParseSignature:
    def test_landmark_fields(self):
        sig = _parse_signature(MOCK_SIGS[0])
        assert sig is not None
        assert sig.up_genes == ["DUSP6", "SPRY2", "ETV5"]
        assert sig.dn_genes == ["MYC", "CCND1"]

    def test_missing_required_fields_returns_none(self):
        assert _parse_signature({"sig_id": "X", "pert_id": "Y"}) is None


class TestParsePathway:
    def test_basic(self):
        p = _parse_pathway(MOCK_REACTOME[0])
        assert p.stable_id == "R-HSA-5673001"
        assert p.display_name == "RAF/MAP kinase cascade"

    def test_species_list_form(self):
        p = _parse_pathway(
            {"stId": "R-X", "displayName": "x", "species": [{"displayName": "Homo sapiens"}]}
        )
        assert p.species == "Homo sapiens"


# ── Mechanism derivation tests ───────────────────────────────────────────


class TestDeriveMechanismId:
    def test_kinase_inhibitor_promoted(self):
        # MOA mentions "kinase" → kinase_inhibition specifically
        assert (
            derive_mechanism_id("BRAF", "BRAF kinase inhibitor")
            == "kinase_inhibition"
        )

    def test_generic_inhibitor_falls_back_to_enzyme(self):
        assert derive_mechanism_id("PARP1", "PARP inhibitor") == "enzyme_inhibition"

    def test_agonist(self):
        assert (
            derive_mechanism_id("DRD2", "dopamine receptor agonist")
            == "receptor_agonism"
        )

    def test_antagonist(self):
        assert (
            derive_mechanism_id("AR", "androgen antagonist") == "receptor_antagonism"
        )

    def test_inhibition_via_stem(self):
        assert (
            derive_mechanism_id("BRAF", "BRAF inhibition") == "enzyme_inhibition"
        )

    def test_partial_agonist(self):
        assert derive_mechanism_id("OPRM1", "partial agonist") == "receptor_agonism"

    def test_unknown_moa_returns_none(self):
        assert derive_mechanism_id("BRAF", "antihypertensive") is None

    def test_empty_moa_returns_none(self):
        assert derive_mechanism_id("BRAF", None) is None
        assert derive_mechanism_id("BRAF", "") is None


# ── Consistency tests ────────────────────────────────────────────────────


class TestCandidatePathwayConsistencies:
    def _sigs(self) -> list[LincsSignature]:
        return [
            LincsSignature(
                sig_id="S1",
                pert_id="P",
                cell_id="A375",
                up_genes=["DUSP6", "SPRY2", "ETV5"],
                dn_genes=["MYC"],
            ),
            LincsSignature(
                sig_id="S2",
                pert_id="P",
                cell_id="MCF7",
                up_genes=["DUSP6", "SPRY2"],
                dn_genes=["FOS"],
            ),
            LincsSignature(
                sig_id="S3",
                pert_id="P",
                cell_id="HEPG2",
                up_genes=["UNRELATED1"],
                dn_genes=["UNRELATED2"],
            ),
        ]

    def test_emits_per_pathway_consistency(self):
        gene_to_pw = {
            "DUSP6": {"R-HSA-MAPK", "R-HSA-RTK"},
            "SPRY2": {"R-HSA-MAPK"},
            "ETV5": {"R-HSA-MAPK"},
            "MYC": {"R-HSA-MAPK"},
            "FOS": {"R-HSA-MAPK"},
        }
        result = candidate_pathway_consistencies(
            self._sigs(), gene_to_pw, min_genes_per_sig=2
        )
        # MAPK: sig1 (4 genes), sig2 (3 genes) hit; sig3 has none
        assert result["R-HSA-MAPK"] == (pytest.approx(2 / 3), 2)
        # RTK: only DUSP6 maps in any sig—never reaches min_genes=2
        assert "R-HSA-RTK" not in result

    def test_below_threshold_excluded(self):
        gene_to_pw = {"DUSP6": {"R-HSA-MAPK"}}
        result = candidate_pathway_consistencies(
            self._sigs(), gene_to_pw, min_genes_per_sig=2
        )
        assert result == {}

    def test_empty_inputs(self):
        assert candidate_pathway_consistencies([], {}) == {}
        assert candidate_pathway_consistencies(self._sigs(), {}) == {}


# ── Client caching tests ─────────────────────────────────────────────────


class TestClientCaching:
    @pytest.mark.asyncio
    async def test_compounds_cache_round_trip(self, tmp_path: Path):
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client._fetch_compounds_from_clue = AsyncMock(return_value=MOCK_PERTS)

        first = await client.get_touchstone_compounds()
        assert {c.pert_id for c in first} == {"BRD-K12345678", "BRD-K87654321"}
        assert (tmp_path / "lincs_compounds.json").exists()

        # Second call: don't re-hit the API
        client._fetch_compounds_from_clue = AsyncMock(
            side_effect=AssertionError("should not refetch")
        )
        second = await client.get_touchstone_compounds()
        assert [c.pert_id for c in second] == [c.pert_id for c in first]

    @pytest.mark.asyncio
    async def test_signatures_cache_round_trip(self, tmp_path: Path):
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client._fetch_signatures_from_clue = AsyncMock(return_value=MOCK_SIGS)

        first = await client.get_consensus_signatures("BRD-K12345678")
        assert len(first) == 2
        assert (tmp_path / "lincs" / "sigs" / "BRD-K12345678.json").exists()

        client._fetch_signatures_from_clue = AsyncMock(
            side_effect=AssertionError("should not refetch")
        )
        second = await client.get_consensus_signatures("BRD-K12345678")
        assert [s.sig_id for s in second] == [s.sig_id for s in first]

    @pytest.mark.asyncio
    async def test_reactome_cache_round_trip(self, tmp_path: Path):
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client._fetch_reactome_pathways = AsyncMock(return_value=MOCK_REACTOME)

        first = await client.get_pathways_for_gene("DUSP6")
        assert {p.stable_id for p in first} == {
            "R-HSA-5673001",
            "R-HSA-9006934",
        }

        client._fetch_reactome_pathways = AsyncMock(
            side_effect=AssertionError("should not refetch")
        )
        second = await client.get_pathways_for_gene("DUSP6")
        assert [p.stable_id for p in second] == [p.stable_id for p in first]

    @pytest.mark.asyncio
    async def test_get_without_api_key_raises(self):
        client = LINCSClient(api_key=None)
        client._api_key = None
        with pytest.raises(RuntimeError, match="CLUE_API_KEY"):
            await client._get("/perts", {})


# ── Graph population tests ───────────────────────────────────────────────


def _seed_graph() -> GraphStore:
    g = GraphStore()
    g.add_node(
        TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF")
    )
    g.add_node(
        MechanismNode(
            id="enzyme_inhibition",
            name="BRAF inhibition",
            mechanism_type=MechanismType.INHIBITION,
        )
    )
    g.add_node(
        BiologyNode(
            id="bio_mapk",
            name="MAPK signaling",
            pathway_ids=["R-HSA-MAPK"],
        )
    )
    return g


class TestPopulateLincsSignatures:
    @pytest.mark.asyncio
    async def test_adds_evidence_when_consistent(self, tmp_path: Path):
        graph = _seed_graph()
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-VEM",
                    pert_iname="vemurafenib",
                    targets=["BRAF"],
                    moas=["BRAF inhibitor"],
                    is_touchstone=True,
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1",
                    pert_id="BRD-VEM",
                    cell_id="A375",
                    up_genes=["DUSP6", "SPRY2", "ETV5"],
                    dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2",
                    pert_id="BRD-VEM",
                    cell_id="MCF7",
                    up_genes=["DUSP6", "SPRY2", "ETV5"],
                    dn_genes=["FOS"],
                ),
            ]
        )
        # Every signature gene maps to MAPK
        client.get_pathways_for_gene = AsyncMock(
            return_value=[
                ReactomePathway(stable_id="R-HSA-MAPK", display_name="MAPK")
            ]
        )

        added = await populate_lincs_signatures(client, graph)
        # One evidence record per (cell_line) hit—A375 + MCF7 both hit MAPK
        assert added == 2
        belief = graph.get_edge_belief(
            "enzyme_inhibition", "bio_mapk", EdgeType.MECHANISM_AFFECTS
        )
        assert belief.alpha > 1.0
        assert len(belief.evidence) == 2
        ev_by_cell = {ev.context["cell_line"]: ev for ev in belief.evidence}
        assert set(ev_by_cell) == {"A375", "MCF7"}
        # Cell-line provenance lives in source_id and context.tissue
        a375 = ev_by_cell["A375"]
        assert a375.source_id == (
            "lincs:BRD-VEM:enzyme_inhibition:bio_mapk:A375"
        )
        assert a375.source_type.value == "preclinical_in_vitro"
        # Round-14 calibration: demoted from strong_support → moderate
        # to stop LINCS accumulation from drowning clinical evidence
        # on shared mechanism_affects edges.
        assert a375.support == "moderate_support"
        assert a375.context["tissue"] == "skin"
        assert ev_by_cell["MCF7"].context["tissue"] == "breast"

    @pytest.mark.asyncio
    async def test_skips_when_target_not_in_graph(self, tmp_path: Path):
        graph = _seed_graph()
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-OTHER",
                    pert_iname="other",
                    targets=["UNKNOWN_GENE"],
                    moas=["UNKNOWN_GENE inhibitor"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            side_effect=AssertionError("should not fetch sigs")
        )

        added = await populate_lincs_signatures(client, graph)
        assert added == 0

    @pytest.mark.asyncio
    async def test_skips_when_mechanism_missing_and_flag_false(self, tmp_path: Path):
        # With create_missing_mechanism=False, "BRAF blocker" → receptor_antagonism
        # has no matching MechanismNode → skip without fetching sigs.
        graph = _seed_graph()
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-BLOCK",
                    pert_iname="blocker",
                    targets=["BRAF"],
                    moas=["BRAF blocker"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            side_effect=AssertionError("should not fetch sigs")
        )
        added = await populate_lincs_signatures(
            client, graph, create_missing_mechanism=False
        )
        assert added == 0
        # No MechanismNode materialized
        mech_nodes = graph.get_nodes_by_type("MechanismNode")
        assert {n["id"] for n in mech_nodes} == {"enzyme_inhibition"}

    @pytest.mark.asyncio
    async def test_creates_missing_mechanism_by_default(self, tmp_path: Path):
        # With the default flag, "BRAF blocker" → receptor_antagonism gets
        # materialized as a new MechanismNode, then evidence is added.
        graph = GraphStore()
        graph.add_node(TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF"))
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-BLOCK",
                    pert_iname="some-blocker",
                    targets=["BRAF"],
                    moas=["BRAF blocker"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1", pert_id="BRD-BLOCK", cell_id="A375",
                    up_genes=["DUSP6", "SPRY2", "ETV5"], dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2", pert_id="BRD-BLOCK", cell_id="MCF7",
                    up_genes=["DUSP6", "SPRY2", "ETV5"], dn_genes=["FOS"],
                ),
            ]
        )
        client.get_pathways_for_gene = AsyncMock(
            return_value=[ReactomePathway(stable_id="R-HSA-MAPK", display_name="MAPK")]
        )

        added = await populate_lincs_signatures(client, graph)
        # 2 cells (A375, MCF7) hit → 2 records
        assert added == 2
        node = graph.get_node("receptor_antagonism")
        assert node["node_type"] == "MechanismNode"
        assert node["mechanism_type"] == "antagonism"
        # modulates_via edge target → mechanism with definitional prior
        belief = graph.get_edge_belief(
            "ENSG_BRAF", "receptor_antagonism", EdgeType.MODULATES_VIA
        )
        assert belief.alpha == 5.0 and belief.beta == 1.0

    @pytest.mark.asyncio
    async def test_reuses_existing_canonical_mechanism(self, tmp_path: Path):
        # MechanismNodes are deduplicated by their canonical id (the
        # MechanismCategory value). A pre-existing canonical node should
        # be reused rather than overwritten.
        graph = GraphStore()
        graph.add_node(TargetNode(id="ENSG_EGFR", name="EGFR", gene_symbol="EGFR"))
        graph.add_node(
            MechanismNode(
                id="enzyme_inhibition",
                name="enzyme inhibition",
                mechanism_type=MechanismType.INHIBITION,
            )
        )
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-GEF",
                    pert_iname="gefitinib",
                    targets=["EGFR"],
                    moas=["EGFR inhibitor"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1", pert_id="BRD-GEF", cell_id="A375",
                    up_genes=["DUSP6", "SPRY2", "ETV5"], dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2", pert_id="BRD-GEF", cell_id="MCF7",
                    up_genes=["DUSP6", "SPRY2", "ETV5"], dn_genes=["FOS"],
                ),
            ]
        )
        client.get_pathways_for_gene = AsyncMock(
            return_value=[ReactomePathway(stable_id="R-HSA-RTK", display_name="RTK")]
        )

        added = await populate_lincs_signatures(client, graph)
        assert added == 2
        mech_ids = {n["id"] for n in graph.get_nodes_by_type("MechanismNode")}
        assert mech_ids == {"enzyme_inhibition"}
        belief = graph.get_edge_belief(
            "enzyme_inhibition", "R-HSA-RTK", EdgeType.MECHANISM_AFFECTS
        )
        assert len(belief.evidence) == 2

    @pytest.mark.asyncio
    async def test_resolves_gene_alias_to_existing_target(self, tmp_path: Path):
        # CLUE compound targets "PD-1"; the graph has TargetNode with
        # gene_symbol "PDCD1". The alias map should bridge them, and the
        # synthesized mechanism id should use the canonical gene.
        graph = GraphStore()
        graph.add_node(TargetNode(id="ENSG_PDCD1", name="PD-1", gene_symbol="PDCD1"))
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-PEMBRO",
                    pert_iname="pembrolizumab",
                    targets=["PD-1"],
                    moas=["PD-1 antagonist"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1", pert_id="BRD-PEMBRO", cell_id="A375",
                    up_genes=["IFNG", "GZMB", "PRF1"], dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2", pert_id="BRD-PEMBRO", cell_id="MCF7",
                    up_genes=["IFNG", "GZMB", "PRF1"], dn_genes=["FOS"],
                ),
            ]
        )
        client.get_pathways_for_gene = AsyncMock(
            return_value=[ReactomePathway(stable_id="R-HSA-IMMUNE", display_name="Immune")]
        )

        added = await populate_lincs_signatures(client, graph)
        # 2 cells hit → 2 records
        assert added == 2
        # Mechanism uses canonical gene_symbol from graph (PDCD1), not CLUE alias
        node = graph.get_node("receptor_antagonism")
        assert node["node_type"] == "MechanismNode"

    @pytest.mark.asyncio
    async def test_skips_when_consistency_below_threshold(self, tmp_path: Path):
        graph = _seed_graph()
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-VEM",
                    pert_iname="vemurafenib",
                    targets=["BRAF"],
                    moas=["BRAF inhibitor"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1",
                    pert_id="BRD-VEM",
                    cell_id="A375",
                    up_genes=["DUSP6", "SPRY2", "ETV5"],
                    dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2",
                    pert_id="BRD-VEM",
                    cell_id="MCF7",
                    up_genes=["UNRELATED1"],
                    dn_genes=["UNRELATED2"],
                ),
                LincsSignature(
                    sig_id="S3",
                    pert_id="BRD-VEM",
                    cell_id="HEPG2",
                    up_genes=["UNRELATED3"],
                    dn_genes=["UNRELATED4"],
                ),
            ]
        )

        def fake_pathways(gene: str) -> list[ReactomePathway]:
            mapk_genes = {"DUSP6", "SPRY2", "ETV5", "MYC"}
            if gene in mapk_genes:
                return [ReactomePathway(stable_id="R-HSA-MAPK", display_name="MAPK")]
            return []

        client.get_pathways_for_gene = AsyncMock(side_effect=fake_pathways)

        # Only 1/3 cell lines hit the pathway → below default 0.5 threshold
        added = await populate_lincs_signatures(client, graph)
        assert added == 0

    @pytest.mark.asyncio
    async def test_creates_missing_biology_node_by_default(self, tmp_path: Path):
        # Graph has target + mechanism but NO BiologyNode for the pathway.
        # With the default flag, LINCS should materialize the pathway as a
        # BiologyNode and attach evidence.
        graph = GraphStore()
        graph.add_node(TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF"))
        graph.add_node(
            MechanismNode(
                id="enzyme_inhibition",
                name="BRAF inhibition",
                mechanism_type=MechanismType.INHIBITION,
            )
        )
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-VEM",
                    pert_iname="vemurafenib",
                    targets=["BRAF"],
                    moas=["BRAF inhibitor"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1",
                    pert_id="BRD-VEM",
                    cell_id="A375",
                    up_genes=["DUSP6", "SPRY2", "ETV5"],
                    dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2",
                    pert_id="BRD-VEM",
                    cell_id="MCF7",
                    up_genes=["DUSP6", "SPRY2", "ETV5"],
                    dn_genes=["FOS"],
                ),
            ]
        )
        client.get_pathways_for_gene = AsyncMock(
            return_value=[
                ReactomePathway(
                    stable_id="R-HSA-5673001",
                    display_name="RAF/MAP kinase cascade",
                )
            ]
        )

        added = await populate_lincs_signatures(client, graph)
        # 2 cells (A375, MCF7) hit → 2 records
        assert added == 2
        # BiologyNode materialized using the Reactome stable id
        node = graph.get_node("R-HSA-5673001")
        assert node["node_type"] == "BiologyNode"
        assert node["name"] == "RAF/MAP kinase cascade"
        assert node["pathway_ids"] == ["R-HSA-5673001"]
        belief = graph.get_edge_belief(
            "enzyme_inhibition", "R-HSA-5673001", EdgeType.MECHANISM_AFFECTS
        )
        assert len(belief.evidence) == 2

    @pytest.mark.asyncio
    async def test_does_not_create_when_flag_false(self, tmp_path: Path):
        graph = GraphStore()
        graph.add_node(TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF"))
        graph.add_node(
            MechanismNode(
                id="enzyme_inhibition",
                name="BRAF inhibition",
                mechanism_type=MechanismType.INHIBITION,
            )
        )
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            return_value=[
                LincsCompound(
                    pert_id="BRD-VEM",
                    pert_iname="vemurafenib",
                    targets=["BRAF"],
                    moas=["BRAF inhibitor"],
                )
            ]
        )
        client.get_consensus_signatures = AsyncMock(
            return_value=[
                LincsSignature(
                    sig_id="S1", pert_id="BRD-VEM", cell_id="A375",
                    up_genes=["DUSP6", "SPRY2", "ETV5"], dn_genes=["MYC"],
                ),
                LincsSignature(
                    sig_id="S2", pert_id="BRD-VEM", cell_id="MCF7",
                    up_genes=["DUSP6", "SPRY2", "ETV5"], dn_genes=["FOS"],
                ),
            ]
        )
        client.get_pathways_for_gene = AsyncMock(
            return_value=[
                ReactomePathway(stable_id="R-HSA-5673001", display_name="MAPK")
            ]
        )

        added = await populate_lincs_signatures(
            client, graph, create_missing_biology=False
        )
        assert added == 0
        # No BiologyNode should have been added
        assert not graph.get_nodes_by_type("BiologyNode")

    @pytest.mark.asyncio
    async def test_no_targets_returns_zero(self, tmp_path: Path):
        graph = GraphStore()
        graph.add_node(
            MechanismNode(
                id="enzyme_inhibition",
                name="BRAF inhibition",
                mechanism_type=MechanismType.INHIBITION,
            )
        )
        client = LINCSClient(api_key="fake", cache_dir=tmp_path)
        client.get_touchstone_compounds = AsyncMock(
            side_effect=AssertionError("should short-circuit before fetch")
        )
        added = await populate_lincs_signatures(client, graph)
        assert added == 0


# ── Biology→Indication wiring tests ──────────────────────────────────────


from src.graph.models import IndicationNode  # noqa: E402
from src.ingestion.lincs import (  # noqa: E402
    _index_indications_by_name,
    _match_indication,
    _normalize_disease_name,
    populate_biology_indication_edges,
)
from src.ingestion.opentargets import OpenTargetsClient  # noqa: E402


class TestNormalizeDiseaseName:
    def test_lowercases_and_strips_punctuation(self):
        assert _normalize_disease_name("Non-Small Cell Lung Cancer (NSCLC)") == (
            "non small cell lung cancer nsclc"
        )

    def test_collapses_whitespace(self):
        assert _normalize_disease_name("  Multiple    Sclerosis  ") == "multiple sclerosis"


class TestIndexIndicationsByName:
    def test_drops_pure_blocklist_names(self):
        g = GraphStore()
        g.add_node(IndicationNode(id="ind_mel", name="Melanoma"))
        g.add_node(IndicationNode(id="ind_arm", name="placebo plus chemotherapy"))
        idx = _index_indications_by_name(g)
        assert "melanoma" in idx
        assert "placebo plus chemotherapy" not in idx


class TestMatchIndication:
    def test_exact_match(self):
        idx = {"melanoma": "ind_mel"}
        assert _match_indication("Melanoma", idx) == "ind_mel"

    def test_substring_match(self):
        idx = {"non small cell lung cancer nsclc": "ind_nsclc"}
        # OT often returns shorter form; should still match
        assert _match_indication("non small cell lung cancer", idx) == "ind_nsclc"

    def test_too_short_no_match(self):
        # "ar" is 2 chars—substring matching disabled below 4
        idx = {"androgen receptor cancer": "ind_x"}
        assert _match_indication("AR", idx) is None

    def test_no_match(self):
        idx = {"melanoma": "ind_mel"}
        assert _match_indication("breast cancer", idx) is None


class TestPopulateBiologyIndicationEdges:
    @pytest.mark.asyncio
    async def test_adds_edges_for_matching_diseases(self, tmp_path: Path):
        from src.graph.models import BiologyNode

        graph = GraphStore()
        graph.add_node(IndicationNode(id="ind_mel", name="Melanoma"))
        graph.add_node(
            BiologyNode(
                id="R-HSA-MAPK",
                name="MAPK signaling",
                pathway_ids=["R-HSA-MAPK"],
            )
        )

        lincs = LINCSClient(api_key="fake", cache_dir=tmp_path)
        lincs.get_pathway_gene_symbols = AsyncMock(
            return_value=["BRAF", "MAP2K1", "MAPK1"]
        )

        ot = OpenTargetsClient()
        # Stub OT: each pathway gene maps to ENSG, ENSG → melanoma
        async def search(symbol):
            return {"target_id": f"ENSG_{symbol}"}
        async def assoc(target_id, min_score=0.0):
            return [{
                "target_id": target_id,
                "disease_id": "EFO_melanoma",
                "disease_name": "Melanoma",
                "overall_score": 0.8,
                "evidence_count": 3,
                "datatypes": {},
            }]
        ot.search_target = AsyncMock(side_effect=search)
        ot.get_associations = AsyncMock(side_effect=assoc)

        added = await populate_biology_indication_edges(lincs, ot, graph)
        assert added == 1
        belief = graph.get_edge_belief(
            "R-HSA-MAPK", "ind_mel", EdgeType.BIOLOGY_DRIVES
        )
        assert belief.alpha > 1.0
        # Edge metadata captures provenance
        edges = graph.get_edges_by_type(EdgeType.BIOLOGY_DRIVES)
        meta = edges[0]["metadata"]
        assert meta["source"] == "reactome+opentargets"
        assert meta["supporting_genes"] == 3

    @pytest.mark.asyncio
    async def test_skips_when_below_min_pathway_genes(self, tmp_path: Path):
        from src.graph.models import BiologyNode

        graph = GraphStore()
        graph.add_node(IndicationNode(id="ind_mel", name="Melanoma"))
        graph.add_node(
            BiologyNode(id="R-HSA-X", name="X", pathway_ids=["R-HSA-X"])
        )

        lincs = LINCSClient(api_key="fake", cache_dir=tmp_path)
        # Only 2 genes resolve to OT—below default threshold of 3
        lincs.get_pathway_gene_symbols = AsyncMock(return_value=["BRAF", "EGFR"])

        ot = OpenTargetsClient()
        ot.search_target = AsyncMock(side_effect=lambda s: {"target_id": f"ENSG_{s}"})
        ot.get_associations = AsyncMock(return_value=[{
            "target_id": "ENSG_X", "disease_id": "EFO_X",
            "disease_name": "Melanoma", "overall_score": 0.9,
            "evidence_count": 1, "datatypes": {},
        }])

        added = await populate_biology_indication_edges(lincs, ot, graph)
        assert added == 0

    @pytest.mark.asyncio
    async def test_skips_indications_with_too_few_supporting_genes(
        self, tmp_path: Path
    ):
        # Pathway has 4 genes (passes pathway-wide gate); 3 of them associate
        # with Melanoma, only 1 with NSCLC. Default min_genes_per_indication=2
        # → Melanoma edge created, NSCLC edge skipped.
        from src.graph.models import BiologyNode

        graph = GraphStore()
        graph.add_node(IndicationNode(id="ind_mel", name="Melanoma"))
        graph.add_node(IndicationNode(id="ind_nsclc", name="NSCLC"))
        graph.add_node(
            BiologyNode(id="R-HSA-Y", name="Y", pathway_ids=["R-HSA-Y"])
        )
        lincs = LINCSClient(api_key="fake", cache_dir=tmp_path)
        lincs.get_pathway_gene_symbols = AsyncMock(
            return_value=["G1", "G2", "G3", "G4"]
        )
        ot = OpenTargetsClient()
        ot.search_target = AsyncMock(
            side_effect=lambda s: {"target_id": f"ENSG_{s}"}
        )

        async def assoc(target_id, min_score=0.0):
            symbol = target_id.removeprefix("ENSG_")
            rows = [{
                "target_id": target_id, "disease_id": "EFO_mel",
                "disease_name": "Melanoma", "overall_score": 0.7,
                "evidence_count": 2, "datatypes": {},
            }] if symbol in {"G1", "G2", "G3"} else []
            if symbol == "G4":
                rows.append({
                    "target_id": target_id, "disease_id": "EFO_nsclc",
                    "disease_name": "NSCLC", "overall_score": 0.9,
                    "evidence_count": 5, "datatypes": {},
                })
            return rows

        ot.get_associations = AsyncMock(side_effect=assoc)

        added = await populate_biology_indication_edges(lincs, ot, graph)
        assert added == 1
        edges = graph.get_edges_by_type(EdgeType.BIOLOGY_DRIVES)
        targets = {e["target_id"] for e in edges}
        assert "ind_mel" in targets
        assert "ind_nsclc" not in targets

    @pytest.mark.asyncio
    async def test_skips_lincs_only_when_flag_set(self, tmp_path: Path):
        # Curated BiologyNode (id doesn't start with R-HSA-) shouldn't be
        # processed when only_lincs_created=True (the default).
        from src.graph.models import BiologyNode

        graph = GraphStore()
        graph.add_node(IndicationNode(id="ind_mel", name="Melanoma"))
        graph.add_node(
            BiologyNode(
                id="bio_curated",
                name="Curated",
                pathway_ids=["R-HSA-CURATED"],
            )
        )
        lincs = LINCSClient(api_key="fake", cache_dir=tmp_path)
        lincs.get_pathway_gene_symbols = AsyncMock(
            side_effect=AssertionError("should not fetch")
        )
        ot = OpenTargetsClient()
        added = await populate_biology_indication_edges(lincs, ot, graph)
        assert added == 0

        # With the flag off, it IS processed
        lincs.get_pathway_gene_symbols = AsyncMock(return_value=["BRAF", "EGFR", "MAPK1"])
        ot.search_target = AsyncMock(side_effect=lambda s: {"target_id": f"ENSG_{s}"})
        ot.get_associations = AsyncMock(return_value=[{
            "target_id": "ENSG_X", "disease_id": "EFO_X",
            "disease_name": "Melanoma", "overall_score": 0.9,
            "evidence_count": 2, "datatypes": {},
        }])
        added2 = await populate_biology_indication_edges(
            lincs, ot, graph, only_lincs_created=False
        )
        assert added2 == 1


# ── modulates_via backfill ───────────────────────────────────────────────


from src.ingestion.lincs import backfill_modulates_via_for_lincs  # noqa: E402


class TestBackfillModulatesVia:
    def test_is_a_noop_under_canonical_ids(self):
        # Under the canonical-id regime, mechanism nodes are
        # MechanismCategory values; the gene→mechanism mapping cannot be
        # recovered from a node id alone, so backfill is intentionally a stub.
        g = GraphStore()
        g.add_node(TargetNode(id="ENSG_BRAF", name="BRAF", gene_symbol="BRAF"))
        g.add_node(MechanismNode(
            id="enzyme_inhibition", name="enzyme inhibition",
            mechanism_type=MechanismType.INHIBITION,
        ))
        assert backfill_modulates_via_for_lincs(g) == 0
