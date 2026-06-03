"""Tests for the Eroom Bio knowledge graph models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.graph.models import (
    BiomarkerNode,
    BiomarkerType,
    BiologyNode,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialOutcome,
    TrialSubgraph,
)
from src.inference.beliefs import SupportBucket


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def evidence_record():
    return EvidenceRecord(
        source_id="NCT00001234",
        source_type=EvidenceType.CLINICAL_PHASE3,
        support=SupportBucket.STRONG_SUPPORT.value,
        quality_score=0.9,
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        provenance_url="https://clinicaltrials.gov/ct2/show/NCT00001234",
    )


# ── Node creation tests ─────────────────────────────────────────────────

class TestCompoundNode:
    def test_create(self):
        node = CompoundNode(
            id="COMPOUND_001",
            name="Imatinib",
            modality=Modality.SMALL_MOLECULE,
            smiles="CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            targets_claimed=["TARGET_BCR_ABL"],
        )
        assert node.id == "COMPOUND_001"
        assert node.modality == Modality.SMALL_MOLECULE
        assert node.targets_claimed == ["TARGET_BCR_ABL"]

    def test_defaults(self):
        node = CompoundNode(id="C1", name="Test", modality=Modality.OTHER)
        assert node.smiles is None
        assert node.targets_claimed == []
        assert node.metadata == {}

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            CompoundNode(id="", name="Test", modality=Modality.OTHER)

    def test_drug_class_properties_default_none(self):
        """Semantic-layers redesign: the drug-class ``category`` +
        ``mechanism_type`` live on the InterventionNode (the drug), not the
        shared signaling-pathway MechanismNode. Both default None so existing
        snapshots load unchanged."""
        node = CompoundNode(id="C1", name="Test")
        assert node.category is None
        assert node.mechanism_type is None

    def test_carries_category_and_mechanism_type(self):
        """A drug's intrinsic class (category) and the MechanismType it derives
        to are stored on the drug node."""
        node = CompoundNode(
            id="pembrolizumab", name="Pembrolizumab",
            category="checkpoint_blockade",
            mechanism_type=MechanismType.ANTAGONISM,
        )
        assert node.category == "checkpoint_blockade"
        assert node.mechanism_type == MechanismType.ANTAGONISM


class TestTargetNode:
    def test_create(self):
        node = TargetNode(
            id="TARGET_EGFR",
            name="Epidermal Growth Factor Receptor",
            gene_symbol="EGFR",
            druggability_score=0.85,
            tissue_expression={"lung": 0.9, "skin": 0.7},
            essentiality_scores={"lung_cancer": 0.95},
        )
        assert node.gene_symbol == "EGFR"
        assert node.tissue_expression["lung"] == 0.9

    def test_empty_gene_symbol_rejected(self):
        with pytest.raises(ValidationError):
            TargetNode(id="T1", name="Test", gene_symbol="")

    def test_default_target_type_is_protein(self):
        """Pre-target-namespace-expansion TargetNodes (all gene/protein
        targets from Open Targets) keep working without setting
        target_type explicitly."""
        from src.graph.models import TargetType
        node = TargetNode(
            id="ENSG00000188389", name="Programmed cell death 1",
            gene_symbol="PDCD1",
        )
        assert node.target_type == TargetType.PROTEIN

    def test_macromolecule_target_uses_chebi_id(self):
        """For DNA-binding chemos, the target is the macromolecule (no
        gene); id namespace is ChEBI; gene_symbol stores the namespaced
        id as a readable handle."""
        from src.graph.models import TargetType
        node = TargetNode(
            id="CHEBI:16991", name="DNA", gene_symbol="CHEBI:16991",
            target_type=TargetType.MACROMOLECULE,
        )
        assert node.target_type == TargetType.MACROMOLECULE
        assert node.id == "CHEBI:16991"

    def test_cell_population_target_uses_cl_id(self):
        """Cell-population targets (rare but real — pancreatic β-cell-
        directed therapies, etc.) use Cell Ontology ids."""
        from src.graph.models import TargetType
        node = TargetNode(
            id="CL:0000169", name="pancreatic β-cell",
            gene_symbol="CL:0000169",
            target_type=TargetType.CELL_POPULATION,
        )
        assert node.target_type == TargetType.CELL_POPULATION


class TestMechanismNode:
    def test_create(self):
        node = MechanismNode(
            id="MECH_001",
            name="Kinase inhibition",
            mechanism_type=MechanismType.INHIBITION,
            selectivity="Type II",
        )
        assert node.mechanism_type == MechanismType.INHIBITION

    def test_optional_selectivity(self):
        node = MechanismNode(id="M1", name="Test", mechanism_type=MechanismType.OTHER)
        assert node.selectivity is None

    def test_mechanism_type_defaults_none(self):
        """Semantic-layers redesign: the MechanismNode is a shared,
        drug-agnostic Reactome pathway, so ``mechanism_type`` (a drug-class
        property) is now Optional + defaults None. Existing snapshots that set
        it still load."""
        node = MechanismNode(id="R-HSA-5673001", name="RAF/MAP kinase cascade")
        assert node.mechanism_type is None

    def test_category_defaults_none(self):
        # Abstraction-ladder redesign: category (the coarse drug-class bucket)
        # is optional + defaults None so existing snapshots / cached graphs load
        # unchanged. The node identity/scale is the specific action; category is
        # the bucket it rolls up into.
        node = MechanismNode(id="M1", name="Test", mechanism_type=MechanismType.OTHER)
        assert node.category is None

    def test_category_set(self):
        node = MechanismNode(
            id="checkpoint_blockade", name="checkpoint blockade",
            mechanism_type=MechanismType.ANTAGONISM,
            category="checkpoint_blockade",
        )
        assert node.category == "checkpoint_blockade"


class TestBiologyNode:
    def test_create(self):
        node = BiologyNode(
            id="BIO_001",
            name="RAS-MAPK signaling",
            pathway_ids=["KEGG:hsa04010"],
            tissue_specificity=["pan-tissue"],
            known_redundancies=["PI3K-AKT"],
        )
        assert node.pathway_ids == ["KEGG:hsa04010"]

    def test_metadata_persists(self):
        """metadata is where the populator records source + alternate pathway
        names; before the field was added the kwarg was silently dropped."""
        node = BiologyNode(
            id="R-HSA-194138",
            name="Signaling by VEGF",
            pathway_ids=["R-HSA-194138", "R-HSA-114608"],
            metadata={
                "source": "reactome_target_lookup",
                "primary_pathway": "R-HSA-194138",
                "alternate_pathway_names": {"R-HSA-114608": "Platelet degranulation"},
            },
        )
        assert node.metadata["source"] == "reactome_target_lookup"
        assert node.metadata["alternate_pathway_names"]["R-HSA-114608"] == "Platelet degranulation"


class TestBiomarkerNode:
    def test_create(self):
        node = BiomarkerNode(
            id="BM_001",
            name="pERK levels",
            biomarker_type=BiomarkerType.PD_MARKER,
            measurability_score=0.75,
        )
        assert node.biomarker_type == BiomarkerType.PD_MARKER


class TestPopulationNode:
    def test_create(self):
        from src.graph.models import SubgroupFeature
        feats = [
            SubgroupFeature(axis="gene", key="EGFR", level="mutant"),
            SubgroupFeature(axis="line", level="first"),
        ]
        pop_id = PopulationNode.compose_id("nsclc", feats)
        node = PopulationNode(
            id=pop_id,
            name="EGFR-mutant first-line NSCLC",
            defining_features=feats,
            estimated_size=50000,
        )
        assert node.estimated_size == 50000
        assert pop_id == "nsclc__egfr_mutant__line_first"

    def test_compose_id_parent_population(self):
        assert PopulationNode.compose_id("melanoma", []) == "melanoma__unselected"

    def test_compose_id_is_order_independent(self):
        from src.graph.models import SubgroupFeature
        feats_a = [
            SubgroupFeature(axis="gene", key="CD274", level="high"),
            SubgroupFeature(axis="line", level="first"),
        ]
        feats_b = [
            SubgroupFeature(axis="line", level="first"),
            SubgroupFeature(axis="gene", key="CD274", level="high"),
        ]
        assert (
            PopulationNode.compose_id("nsclc", feats_a)
            == PopulationNode.compose_id("nsclc", feats_b)
        )


class TestEndpointNode:
    def test_create(self):
        node = EndpointNode(
            id="EP_001",
            name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.ACCEPTED,
            measurement_properties={"unit": "months", "method": "Kaplan-Meier"},
        )
        assert node.regulatory_status == RegulatoryStatus.ACCEPTED


class TestIndicationNode:
    def test_create(self):
        node = IndicationNode(
            id="IND_001",
            name="Non-Small Cell Lung Cancer",
            icd_codes=["C34"],
            prevalence=0.0006,
            standard_of_care="Platinum doublet chemotherapy",
            unmet_need_score=0.8,
        )
        assert node.icd_codes == ["C34"]


# ── Evidence & belief tests ──────────────────────────────────────────────

class TestEvidenceRecord:
    def test_create(self, evidence_record):
        assert evidence_record.source_type == EvidenceType.CLINICAL_PHASE3
        assert evidence_record.support == SupportBucket.STRONG_SUPPORT.value
        assert evidence_record.quality_score == 0.9

    def test_quality_score_default_is_one(self):
        ev = EvidenceRecord(
            source_id="S1",
            source_type=EvidenceType.LITERATURE,
            support=SupportBucket.WEAK_SUPPORT.value,
            timestamp=datetime.now(timezone.utc),
        )
        assert ev.quality_score == 1.0

    def test_quality_score_upper_bound(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="S1",
                source_type=EvidenceType.LITERATURE,
                support=SupportBucket.WEAK_SUPPORT.value,
                quality_score=1.5,
                timestamp=datetime.now(timezone.utc),
            )

    def test_negative_quality_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="S1",
                source_type=EvidenceType.LITERATURE,
                support=SupportBucket.WEAK_SUPPORT.value,
                quality_score=-0.1,
                timestamp=datetime.now(timezone.utc),
            )

    def test_empty_source_id_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="",
                source_type=EvidenceType.LITERATURE,
                support=SupportBucket.WEAK_SUPPORT.value,
                timestamp=datetime.now(timezone.utc),
            )

    def test_empty_support_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="S1",
                source_type=EvidenceType.LITERATURE,
                support="",
                timestamp=datetime.now(timezone.utc),
            )


class TestEdgeBeliefState:
    def test_uninformative_prior(self):
        belief = EdgeBeliefState()
        assert belief.alpha == 1.0
        assert belief.beta == 1.0
        assert belief.expected_probability == 0.5
        assert belief.evidence_strength == 0.0

    def test_expected_probability(self):
        belief = EdgeBeliefState(alpha=10.0, beta=2.0)
        assert belief.expected_probability == pytest.approx(10.0 / 12.0)

    def test_evidence_strength(self):
        belief = EdgeBeliefState(alpha=5.0, beta=3.0)
        assert belief.evidence_strength == pytest.approx(6.0)

    def test_variance(self):
        # Beta(2, 3): variance = 2*3 / (5^2 * 6) = 6/150 = 0.04
        belief = EdgeBeliefState(alpha=2.0, beta=3.0)
        assert belief.variance == pytest.approx(0.04)

    def test_conflict_score_no_evidence(self):
        belief = EdgeBeliefState(alpha=1.0, beta=1.0)
        assert belief.conflict_score == 0.0

    def test_conflict_score_high_when_uncertain_with_evidence(self):
        # Lots of evidence (high strength) but p near 0.5 → high conflict
        belief = EdgeBeliefState(alpha=50.0, beta=50.0)
        assert belief.conflict_score > 90.0

    def test_conflict_score_low_when_decisive(self):
        # Lots of evidence and p far from 0.5 → low conflict
        belief = EdgeBeliefState(alpha=95.0, beta=5.0)
        assert belief.conflict_score < 20.0

    def test_credible_interval_uninformative(self):
        belief = EdgeBeliefState(alpha=1.0, beta=1.0)
        lower, upper = belief.credible_interval(0.95)
        assert lower == pytest.approx(0.025, abs=0.001)
        assert upper == pytest.approx(0.975, abs=0.001)

    def test_credible_interval_tight_with_evidence(self):
        belief = EdgeBeliefState(alpha=100.0, beta=100.0)
        lower, upper = belief.credible_interval(0.95)
        assert upper - lower < 0.15  # tight interval

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValidationError):
            EdgeBeliefState(alpha=-1.0, beta=1.0)

    def test_negative_beta_rejected(self):
        with pytest.raises(ValidationError):
            EdgeBeliefState(alpha=1.0, beta=-1.0)


class TestGraphEdge:
    def test_create(self):
        edge = GraphEdge(
            source_id="COMPOUND_001",
            target_id="TARGET_EGFR",
            edge_type=EdgeType.AFFECTS,
        )
        assert edge.edge_type == EdgeType.AFFECTS
        assert edge.belief.expected_probability == 0.5

    def test_with_belief(self, evidence_record):
        belief = EdgeBeliefState(alpha=5.0, beta=2.0, evidence=[evidence_record])
        edge = GraphEdge(
            source_id="C1",
            target_id="T1",
            edge_type=EdgeType.AFFECTS,
            belief=belief,
        )
        assert len(edge.belief.evidence) == 1

    def test_empty_source_rejected(self):
        with pytest.raises(ValidationError):
            GraphEdge(source_id="", target_id="T1", edge_type=EdgeType.AFFECTS)


class TestModulatesEfficacyOf:
    def test_edge_type_value(self):
        assert EdgeType.MODULATES_EFFICACY_OF.value == "modulates_efficacy_of"

    def test_edge_can_be_constructed(self):
        edge = GraphEdge(
            source_id="aldesleukin",
            target_id="gp100_antigen",
            edge_type=EdgeType.MODULATES_EFFICACY_OF,
            belief=EdgeBeliefState(alpha=2.0, beta=1.5),
        )
        assert edge.edge_type == EdgeType.MODULATES_EFFICACY_OF

    def test_canonical_endpoints_lex_orders(self):
        from src.graph.models import canonical_modulation_endpoints
        assert canonical_modulation_endpoints("b", "a") == ("a", "b")
        assert canonical_modulation_endpoints("a", "b") == ("a", "b")
        assert canonical_modulation_endpoints("x", "x") == ("x", "x")

    def test_modulation_endpoint_node_types_covers_chain_layers(self):
        from src.graph.models import MODULATION_ENDPOINT_NODE_TYPES
        for nt in (
            "InterventionNode", "TargetNode",
            "MechanismNode", "BiologyNode",
        ):
            assert nt in MODULATION_ENDPOINT_NODE_TYPES


# ── Trial subgraph tests ────────────────────────────────────────────────

class TestTrialSubgraph:
    def _make_arm_and_chain(self, outcome: TrialOutcome = TrialOutcome.SUCCESS):
        from src.graph.models import CausalChain, TrialArm
        arm = TrialArm(
            arm_id="arm1", compound_ids=["c1"], regimen_compound_id="c1",
            is_combination=False,
        )
        chain = CausalChain(
            arm_id="arm1", compound_id="c1",
            subgroup_population_id="ind__unselected",
            target_id="t1", mechanism_id="m1", biology_id="b1",
            indication_id="ind", endpoint_id="e1",
            outcome=outcome,
        )
        return arm, chain

    def test_create(self):
        arm, chain = self._make_arm_and_chain()
        ts = TrialSubgraph(
            trial_id="NCT03456789",
            phase="3", arms=[arm], chains=[chain],
            parent_population_id="ind__unselected",
        )
        assert len(ts.chains) == 1
        assert ts.chains[0].outcome == TrialOutcome.SUCCESS

    def test_empty_trial_id_rejected(self):
        arm, chain = self._make_arm_and_chain()
        with pytest.raises(ValidationError):
            TrialSubgraph(
                trial_id="", phase="1", arms=[arm], chains=[chain],
                parent_population_id="ind__unselected",
            )

    def test_all_outcomes(self):
        for outcome in TrialOutcome:
            arm, chain = self._make_arm_and_chain(outcome=outcome)
            ts = TrialSubgraph(
                trial_id="NCT00000001",
                phase="2", arms=[arm], chains=[chain],
                parent_population_id="ind__unselected",
            )
            assert ts.chains[0].outcome == outcome


class TestNormalizeEntity:
    def test_compound_lowercases(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("Pembrolizumab", "InterventionNode") == "pembrolizumab"
        assert normalize_entity("KEYTRUDA", "InterventionNode") == "keytruda"

    def test_compound_drugbank_passthrough(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("DB00001", "InterventionNode") == "DB00001"

    def test_target_ensg_passthrough(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("ENSG00000169245", "TargetNode") == "ENSG00000169245"

    def test_target_symbol_uppercased(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("PD-1", "TargetNode") == "PD1"

    def test_target_accepts_chebi_namespace(self):
        """Macromolecular targets (DNA, tubulin) use ChEBI ids."""
        from src.graph.models import normalize_entity

        assert normalize_entity("CHEBI:16991", "TargetNode") == "CHEBI:16991"
        assert normalize_entity("CHEBI:35797", "TargetNode") == "CHEBI:35797"

    def test_target_accepts_cell_ontology_namespace(self):
        """Cell-population targets (rare) use Cell Ontology ids."""
        from src.graph.models import normalize_entity

        assert normalize_entity("CL:0000169", "TargetNode") == "CL:0000169"

    def test_target_accepts_hgnc_namespace(self):
        """HGNC fallback when Ensembl isn't resolved for a gene-coded
        antigen (CD19, BCMA, etc.)."""
        from src.graph.models import normalize_entity

        assert normalize_entity("HGNC:1633", "TargetNode") == "HGNC:1633"

    def test_target_accepts_go_namespace_for_cellular_components(self):
        """GO cellular_component ids (e.g. `microtubule` GO:0005874)
        as a structural target namespace."""
        from src.graph.models import normalize_entity

        assert normalize_entity("GO:0005874", "TargetNode") == "GO:0005874"

    def test_mechanism_validates_against_enum(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("kinase_inhibition", "MechanismNode") == "kinase_inhibition"

    def test_mechanism_unknown_falls_back_to_other(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("magic_pixie_dust", "MechanismNode") == "other"

    def test_biology_accepts_go_term_id(self):
        """GO biological-process terms are valid BiologyNode ids — used when
        Reactome's curation is missing/wrong for a gene (CRBN, etc.)."""
        from src.graph.models import normalize_entity

        assert normalize_entity("GO:0016567", "BiologyNode") == "GO:0016567"
        assert normalize_entity("GO:0043161", "BiologyNode") == "GO:0043161"

    def test_biology_rejects_malformed_go_term(self):
        """Pattern is strict — exactly 7 digits after GO:."""
        from src.graph.models import normalize_entity

        with pytest.raises(ValueError):
            normalize_entity("GO:123", "BiologyNode")
        with pytest.raises(ValueError):
            normalize_entity("GO:00165670", "BiologyNode")

    def test_biology_requires_reactome_id(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("R-HSA-9006934", "BiologyNode") == "R-HSA-9006934"
        with pytest.raises(ValueError):
            normalize_entity("free text pathway", "BiologyNode")

    def test_biomarker_uppercased(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("PD-L1", "BiomarkerNode") == "PD_L1"
        assert normalize_entity("EGFR mutation", "BiomarkerNode") == "EGFR_MUTATION"

    def test_population_requires_indication_double_underscore_features(self):
        from src.graph.models import normalize_entity

        # Composed-id form passes through as-is.
        assert (
            normalize_entity("melanoma__cd274_high__line_first", "PopulationNode")
            == "melanoma__cd274_high__line_first"
        )
        # Parent population (no subgroup features) is "{indication}__unselected".
        assert (
            normalize_entity("melanoma__unselected", "PopulationNode")
            == "melanoma__unselected"
        )
        # Bare indication slug (no "__feature" tail) is invalid.
        with pytest.raises(ValueError):
            normalize_entity("melanoma", "PopulationNode")

    def test_endpoint_requires_class_and_indication(self):
        from src.graph.models import normalize_entity

        assert normalize_entity("OS_melanoma", "EndpointNode") == "OS_melanoma"
        with pytest.raises(ValueError):
            normalize_entity("badclass_melanoma", "EndpointNode")

    def test_indication_slug(self):
        from src.graph.models import normalize_entity

        assert (
            normalize_entity("Non-Small Cell Lung Cancer", "IndicationNode")
            == "non_small_cell_lung_cancer"
        )

    def test_unknown_node_type_raises(self):
        from src.graph.models import normalize_entity

        with pytest.raises(ValueError, match="Unknown node_type"):
            normalize_entity("foo", "FooNode")
