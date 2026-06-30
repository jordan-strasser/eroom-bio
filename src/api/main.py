"""FastAPI service exposing the Eroom Bio knowledge graph and pipelines."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.annotation.attributor import AppliedEdgeUpdate, Attributor
from src.annotation.classifier import Classifier, annotate_trial
from src.annotation.extractor import Extractor
from src.annotation.taxonomy import FailureClassification, TrialExtraction
from src.graph.models import (
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.populate import (
    build_arms,
    ensure_parent_population,
    seed_trial_node,
    synthesize_combo_compounds,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.contract import PredictionResult
from src.prediction.path_query import predict_clinical_hypothesis

logger = logging.getLogger("eroom.api")

DEFAULT_SNAPSHOT = "data/exports/oncology_annotated.json"


# ── App state ────────────────────────────────────────────────────────────


class AppState:
    graph: GraphStore
    snapshot_path: Path
    snapshot_loaded_at: datetime
    anthropic_client: anthropic.AsyncAnthropic | None = None

    def get_anthropic(self) -> anthropic.AsyncAnthropic:
        if self.anthropic_client is None:
            self.anthropic_client = anthropic.AsyncAnthropic(timeout=60.0)
        return self.anthropic_client


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    snapshot_path = Path(os.environ.get("EROOM_GRAPH_SNAPSHOT", DEFAULT_SNAPSHOT))
    graph = GraphStore()
    if snapshot_path.exists():
        logger.info("Loading graph snapshot from %s", snapshot_path)
        graph.import_snapshot(str(snapshot_path))
        s = graph.stats()
        logger.info(
            "Loaded graph: %d nodes, %d edges",
            s["node_count"],
            s["edge_count"],
        )
    else:
        logger.warning(
            "Snapshot %s not found; serving an empty graph", snapshot_path
        )
    state.graph = graph
    state.snapshot_path = snapshot_path
    state.snapshot_loaded_at = datetime.now(timezone.utc)
    yield


app = FastAPI(
    title="Eroom Bio API",
    description="Knowledge graph and prediction service for clinical trials.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── Request / response models ────────────────────────────────────────────


class PredictRequest(BaseModel):
    compound: str
    target: str
    mechanism: str | None = None
    indication: str
    endpoint: str | None = None
    population: str | None = None
    n_samples: int = Field(default=10_000, ge=100, le=100_000)


class AnnotateRequest(BaseModel):
    nct_id: str


class EdgeView(BaseModel):
    source_id: str
    target_id: str
    edge_type: str
    expected_probability: float
    evidence_strength: float
    conflict_score: float
    alpha: float
    beta: float


class NodeView(BaseModel):
    id: str
    node_type: str
    data: dict[str, Any]
    edges: list[EdgeView]


class EdgeDetailView(BaseModel):
    source_id: str
    target_id: str
    edge_type: str
    belief: EdgeBeliefState


class AnnotateResponse(BaseModel):
    nct_id: str
    extraction: TrialExtraction
    classification: FailureClassification
    edge_updates: list[AppliedEdgeUpdate]


# ── Helpers ──────────────────────────────────────────────────────────────


def _belief_from_data(data: dict[str, Any]) -> EdgeBeliefState | None:
    raw = data.get("belief")
    if not raw:
        return None
    return EdgeBeliefState.model_validate(raw)


def _edge_view(src: str, tgt: str, edge_type: str, belief: EdgeBeliefState) -> EdgeView:
    return EdgeView(
        source_id=src,
        target_id=tgt,
        edge_type=edge_type,
        expected_probability=belief.expected_probability,
        evidence_strength=belief.evidence_strength,
        conflict_score=belief.conflict_score,
        alpha=belief.alpha,
        beta=belief.beta,
    )


def _find_node_by_name(name: str | None, node_type: str) -> str | None:
    if not name:
        return None
    name_lower = name.lower()
    for node in state.graph.get_nodes_by_type(node_type):
        node_name = (node.get("name") or "").lower()
        if node_name and (name_lower in node_name or node_name in name_lower):
            return node.get("id")
    return None


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/graph/stats")
def graph_stats() -> dict[str, Any]:
    return state.graph.stats()


@app.get("/graph/node/{node_id}", response_model=NodeView)
def get_node(node_id: str) -> NodeView:
    try:
        node_data = state.graph.get_node(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node_type = node_data.pop("node_type", "Unknown")
    edges: list[EdgeView] = []
    for raw_edge in state.graph.get_neighboring_edges(node_id):
        belief = _belief_from_data(raw_edge)
        if belief is None:
            continue
        edges.append(
            _edge_view(
                raw_edge["source_id"],
                raw_edge["target_id"],
                raw_edge["edge_type"],
                belief,
            )
        )

    return NodeView(id=node_id, node_type=node_type, data=node_data, edges=edges)


@app.get("/graph/edge/{src_id}/{tgt_id}/{edge_type}", response_model=EdgeDetailView)
def get_edge(src_id: str, tgt_id: str, edge_type: str) -> EdgeDetailView:
    try:
        et = EdgeType(edge_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown edge_type '{edge_type}'")
    try:
        belief = state.graph.get_edge_belief(src_id, tgt_id, et)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return EdgeDetailView(
        source_id=src_id, target_id=tgt_id, edge_type=edge_type, belief=belief
    )


@app.post("/predict", response_model=PredictionResult)
def predict(req: PredictRequest) -> PredictionResult:
    compound_id = _find_node_by_name(req.compound, "InterventionNode")
    if not compound_id:
        raise HTTPException(status_code=404, detail=f"Compound '{req.compound}' not found")
    target_id = _find_node_by_name(req.target, "TargetNode")
    if not target_id:
        raise HTTPException(status_code=404, detail=f"Target '{req.target}' not found")
    indication_id = _find_node_by_name(req.indication, "IndicationNode")
    if not indication_id:
        raise HTTPException(status_code=404, detail=f"Indication '{req.indication}' not found")

    # Canonical predictor — the SAME path the eval uses
    # (predict_clinical_hypothesis). It resolves the rest of the chain from the
    # graph, INCLUDING biology, rather than pinning biology_id="UNKNOWN" as the
    # old hand-built chain did. Optional intermediates pass None so the resolver
    # fills them from the target-onward topology.
    return predict_clinical_hypothesis(
        state.graph,
        compound_id,
        indication_id,
        target_id=target_id,
        mechanism_id=_find_node_by_name(req.mechanism, "MechanismNode"),
        endpoint_id=_find_node_by_name(req.endpoint, "EndpointNode"),
        population_id=_find_node_by_name(req.population, "PopulationNode"),
        n_samples=req.n_samples,
    )


@app.post("/annotate", response_model=AnnotateResponse)
async def annotate(req: AnnotateRequest) -> AnnotateResponse:
    client = state.get_anthropic()
    extractor = Extractor(client)
    classifier = Classifier(client)

    ct = ClinicalTrialsClient()
    try:
        trial_record = await ct.get_study(req.nct_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch {req.nct_id} from ClinicalTrials.gov: {exc}",
        )

    def _build_subgraph(trial_rec, extraction) -> TrialSubgraph:
        # Build the multi-arm shape from the trial's parsed arm groups,
        # synthesize a combo CompoundNode if needed, and create one chain
        # per arm at the parent enrollment population. Per-subgroup chains
        # are added later (extractor task) once subgroup features are
        # available from the LLM call.
        seed_trial_node(state.graph, trial_rec)
        arms = build_arms(trial_rec)
        synthesize_combo_compounds(state.graph, arms)

        indication_id = "UNKNOWN"
        for cond in trial_rec.conditions:
            resolved = _find_node_by_name(cond, "IndicationNode")
            if resolved:
                indication_id = resolved
                break

        target_id = (
            _find_node_by_name(extraction.target_name, "TargetNode") or "UNKNOWN"
        )
        endpoint_id = (
            _find_node_by_name(extraction.primary_endpoint, "EndpointNode") or "UNKNOWN"
        )
        parent_pop = (
            ensure_parent_population(state.graph, indication_id, indication_name=None)
            if indication_id != "UNKNOWN"
            else "UNKNOWN__unselected"
        )

        chains = [
            CausalChain(
                arm_id=arm.arm_id,
                compound_id=arm.regimen_compound_id,
                subgroup_population_id=parent_pop,
                target_id=target_id,
                mechanism_id="UNKNOWN",
                biology_id="UNKNOWN",
                indication_id=indication_id,
                endpoint_id=endpoint_id,
                outcome=TrialOutcome.UNKNOWN,
            )
            for arm in arms
        ]
        ts = TrialSubgraph(
            trial_id=req.nct_id,
            phase=trial_rec.phase or "3",
            arms=arms,
            chains=chains,
            parent_population_id=parent_pop,
        )
        state.graph.set_trial_subgraph(ts)
        return ts

    try:
        extraction, classification, updates = await annotate_trial(
            trial_record,
            extractor,
            classifier,
            attributor=Attributor(state.graph),
            build_subgraph=_build_subgraph,
        )
    except Exception as exc:
        logger.exception("Annotation failed for %s", req.nct_id)
        raise HTTPException(status_code=500, detail=f"Annotation failed: {exc}")

    return AnnotateResponse(
        nct_id=req.nct_id,
        extraction=extraction,
        classification=classification,
        edge_updates=updates,
    )


@app.get("/trial/{nct_id}")
def get_trial_annotation(nct_id: str) -> dict[str, Any]:
    annotations_dir = Path("data/annotations")
    extraction_path = annotations_dir / f"{nct_id}_extraction.json"
    classification_path = annotations_dir / f"{nct_id}_classification.json"
    if not extraction_path.exists() and not classification_path.exists():
        raise HTTPException(
            status_code=404, detail=f"No stored annotation for {nct_id}"
        )

    import json as _json

    payload: dict[str, Any] = {"nct_id": nct_id}
    if extraction_path.exists():
        payload["extraction"] = _json.loads(extraction_path.read_text())
    if classification_path.exists():
        payload["classification"] = _json.loads(classification_path.read_text())
    return payload


