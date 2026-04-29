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
    EvidenceDirection,
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


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def evidence_record():
    return EvidenceRecord(
        source_id="NCT00001234",
        source_type=EvidenceType.CLINICAL_PHASE3,
        quality_score=0.9,
        direction=EvidenceDirection.SUPPORTING,
        magnitude=2.5,
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
        node = PopulationNode(
            id="POP_001",
            name="EGFR-mutant NSCLC",
            defining_features=["NSCLC", "EGFR L858R or exon 19 del"],
            estimated_size=50000,
            genomic_features=["EGFR_L858R", "EGFR_EX19DEL"],
        )
        assert node.estimated_size == 50000


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
        assert evidence_record.quality_score == 0.9

    def test_quality_score_bounds(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="S1",
                source_type=EvidenceType.LITERATURE,
                quality_score=1.5,
                direction=EvidenceDirection.SUPPORTING,
                magnitude=1.0,
                timestamp=datetime.now(timezone.utc),
            )

    def test_negative_quality_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="S1",
                source_type=EvidenceType.LITERATURE,
                quality_score=-0.1,
                direction=EvidenceDirection.SUPPORTING,
                magnitude=1.0,
                timestamp=datetime.now(timezone.utc),
            )

    def test_negative_magnitude_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="S1",
                source_type=EvidenceType.LITERATURE,
                quality_score=0.5,
                direction=EvidenceDirection.SUPPORTING,
                magnitude=-1.0,
                timestamp=datetime.now(timezone.utc),
            )

    def test_empty_source_id_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceRecord(
                source_id="",
                source_type=EvidenceType.LITERATURE,
                quality_score=0.5,
                direction=EvidenceDirection.SUPPORTING,
                magnitude=1.0,
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
            edge_type=EdgeType.BINDS_TO,
        )
        assert edge.edge_type == EdgeType.BINDS_TO
        assert edge.belief.expected_probability == 0.5

    def test_with_belief(self, evidence_record):
        belief = EdgeBeliefState(alpha=5.0, beta=2.0, evidence=[evidence_record])
        edge = GraphEdge(
            source_id="C1",
            target_id="T1",
            edge_type=EdgeType.BINDS_TO,
            belief=belief,
        )
        assert len(edge.belief.evidence) == 1

    def test_empty_source_rejected(self):
        with pytest.raises(ValidationError):
            GraphEdge(source_id="", target_id="T1", edge_type=EdgeType.BINDS_TO)


# ── Trial subgraph tests ────────────────────────────────────────────────

class TestTrialSubgraph:
    def test_create(self):
        trial = TrialSubgraph(
            trial_id="NCT03456789",
            compound_id="COMPOUND_001",
            target_id="TARGET_EGFR",
            mechanism_id="MECH_001",
            biology_id="BIO_001",
            indication_id="IND_001",
            endpoint_id="EP_001",
            population_id="POP_001",
            outcome=TrialOutcome.SUCCESS,
            phase="3",
        )
        assert trial.outcome == TrialOutcome.SUCCESS

    def test_empty_trial_id_rejected(self):
        with pytest.raises(ValidationError):
            TrialSubgraph(
                trial_id="",
                compound_id="C1",
                target_id="T1",
                mechanism_id="M1",
                biology_id="B1",
                indication_id="I1",
                endpoint_id="E1",
                population_id="P1",
                outcome=TrialOutcome.UNKNOWN,
                phase="1",
            )

    def test_all_outcomes(self):
        for outcome in TrialOutcome:
            trial = TrialSubgraph(
                trial_id="NCT00000001",
                compound_id="C1",
                target_id="T1",
                mechanism_id="M1",
                biology_id="B1",
                indication_id="I1",
                endpoint_id="E1",
                population_id="P1",
                outcome=outcome,
                phase="2",
            )
            assert trial.outcome == outcome
