"""Retrospective backtest: train graph on past trials, evaluate predictions on held-out future trials."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import anthropic
import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.annotation.attributor import Attributor
from src.annotation.classifier import Classifier
from src.annotation.extractor import Extractor
from src.annotation.taxonomy import FailureClassification, FailureMode
from src.graph.models import (
    EdgeType,
    GraphEdge,
    TargetNode,
    TrialSubgraph,
)
from src.graph.populate import PopulationPipeline, seed_endpoint_captures_edge
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ClinicalTrialsClient,
    TrialRecord,
    map_trial_to_graph_nodes,
)
from src.ingestion.opentargets import OpenTargetsClient, score_to_prior
from src.prediction.path_query import PredictionEngine, PredictionResult

logger = logging.getLogger(__name__)
console = Console()


# ── Models ───────────────────────────────────────────────────────────────


class TrialPrediction(BaseModel):
    """Per-trial backtest record: predicted vs actual."""

    model_config = ConfigDict(use_enum_values=False)

    trial_id: str
    predicted_p_success: float
    actual_success: bool
    predicted_bottleneck: EdgeType | None = None
    actual_bottleneck: EdgeType | None = None
    predicted_failure_mode: FailureMode | None = None
    actual_failure_mode: FailureMode | None = None
    ci_lower: float
    ci_upper: float


class CalibrationBin(BaseModel):
    bin_range: str
    n: int
    mean_predicted: float
    actual_rate: float


class BacktestResult(BaseModel):
    n_training: int
    n_test: int
    auc_roc: float | None = None
    calibration: list[CalibrationBin] = Field(default_factory=list)
    failure_mode_accuracy: float | None = None
    bottleneck_accuracy: float | None = None
    individual_predictions: list[TrialPrediction] = Field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────


_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%Y")


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


# Predicted weakest edge → most plausible failure mode it implies.
# Inverted from FAILURE_MODE_RULES.edges_to_weaken.
_EDGE_TO_FAILURE_MODE: dict[EdgeType, FailureMode] = {
    EdgeType.BINDS_TO: FailureMode.NO_TARGET_ENGAGEMENT,
    EdgeType.MODULATES_VIA: FailureMode.NO_TARGET_ENGAGEMENT,
    EdgeType.MECHANISM_AFFECTS: FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED,
    EdgeType.BIOLOGY_DRIVES: FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT,
    EdgeType.REFLECTS_BIOLOGY: FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT,
    EdgeType.ENDPOINT_CAPTURES: FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT,
    EdgeType.RESPONDS_DIFFERENTLY: FailureMode.WRONG_POPULATION,
}


def _auc_roc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U–based AUC. None if labels are single-class."""
    if not scores or len(set(labels)) < 2:
        return None
    pos = [s for s, lab in zip(scores, labels) if lab == 1]
    neg = [s for s, lab in zip(scores, labels) if lab == 0]
    if not pos or not neg:
        return None
    count = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


def _calibration_bins(
    scores: list[float], labels: list[int], n_bins: int = 10
) -> list[CalibrationBin]:
    if not scores:
        return []
    edges = np.linspace(0, 1, n_bins + 1)
    interior = edges[1:-1]
    arr_scores = np.asarray(scores, dtype=float)
    arr_labels = np.asarray(labels, dtype=float)
    indices = np.digitize(arr_scores, interior)
    out: list[CalibrationBin] = []
    for i in range(n_bins):
        mask = indices == i
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(CalibrationBin(
            bin_range=f"{edges[i]:.2f}-{edges[i+1]:.2f}",
            n=n,
            mean_predicted=float(arr_scores[mask].mean()),
            actual_rate=float(arr_labels[mask].mean()),
        ))
    return out


def _extract_actual_bottleneck(
    classification: FailureClassification,
) -> EdgeType | None:
    """Pick the highest-magnitude weakened edge from the classifier output."""
    raw = getattr(classification, "_raw", {})
    candidates: list[tuple[float, EdgeType]] = []
    for item in raw.get("edges_to_update", []):
        if item.get("direction") != "weaken":
            continue
        try:
            et = EdgeType(item.get("edge_type", ""))
        except ValueError:
            continue
        mag = float(item.get("magnitude", 0.5))
        candidates.append((mag, et))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── BacktestRunner ───────────────────────────────────────────────────────


class BacktestRunner:
    """Splits trials by completion date, trains the graph, then evaluates predictions on the held-out window."""

    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        ot_client: OpenTargetsClient | None = None,
    ) -> None:
        self.anthropic = anthropic_client or anthropic.AsyncAnthropic()
        self.ot_client = ot_client or OpenTargetsClient()
        self.graph = GraphStore()
        self.pipeline = PopulationPipeline(self.graph)
        self.pipeline._ot_client = self.ot_client

    async def run_backtest(
        self,
        trials: list[TrialRecord],
        cutoff_date: date,
        test_window_end: date,
        max_training: int | None = None,
        max_test: int | None = None,
        annotation_concurrency: int = 2,
    ) -> BacktestResult:
        # 1. Split by completion date
        training, test = self._split_by_date(trials, cutoff_date, test_window_end)
        if max_training is not None:
            training = training[:max_training]
        if max_test is not None:
            test = test[:max_test]
        console.print(
            f"[bold]Split:[/bold] {len(training)} training "
            f"(< {cutoff_date.isoformat()}), {len(test)} test "
            f"({cutoff_date.isoformat()} – {test_window_end.isoformat()})"
        )

        # 2. Populate graph with deduplicated nodes from BOTH splits
        await self._populate_graph(training + test)
        stats = self.graph.stats()
        console.print(
            f"[bold]Graph after population:[/bold] "
            f"{stats['node_count']} nodes, {stats['edge_count']} edges"
        )

        # 3. Build trial subgraphs (target_id resolved via name match)
        all_subgraphs = self._build_subgraphs(training + test)
        sg_by_id: dict[str, TrialSubgraph] = {sg.trial_id: sg for sg in all_subgraphs}
        training_kept = [t for t in training if t.nct_id in sg_by_id]
        test_kept = [t for t in test if t.nct_id in sg_by_id]
        console.print(
            f"[bold]Subgraphs:[/bold] "
            f"{len(training_kept)} training, {len(test_kept)} test"
        )

        # 4. Train: extract → classify → attribute on each training trial
        await self._train_phase(
            training_kept, sg_by_id, concurrency=annotation_concurrency
        )

        # 5. Test: predict on pre-outcome graph, then annotate to get ground truth
        predictions = await self._test_phase(
            test_kept, sg_by_id, concurrency=annotation_concurrency
        )
        console.print(
            f"[bold]Test predictions:[/bold] {len(predictions)} usable "
            f"(skipped trials with unknown outcome or unbuildable subgraph)"
        )

        # 6. Aggregate metrics
        return self._compute_metrics(
            predictions,
            n_training=len(training_kept),
            n_test=len(test_kept),
        )

    # ── Step 1: split ────────────────────────────────────────────────

    def _split_by_date(
        self,
        trials: list[TrialRecord],
        cutoff_date: date,
        test_window_end: date,
    ) -> tuple[list[TrialRecord], list[TrialRecord]]:
        training: list[TrialRecord] = []
        test: list[TrialRecord] = []
        for trial in trials:
            d = _parse_date(trial.completion_date)
            if d is None:
                continue
            if d < cutoff_date:
                training.append(trial)
            elif d <= test_window_end:
                test.append(trial)
        return training, test

    # ── Step 2: populate graph ───────────────────────────────────────

    async def _populate_graph(self, trials: list[TrialRecord]) -> None:
        seen_indications: dict[str, str] = {}  # name → indication_id
        ec_seeded = 0
        seeded_pairs: set[tuple[str, str]] = set()
        for trial in trials:
            nodes = map_trial_to_graph_nodes(trial)
            for ind in nodes["indications"]:
                if self.pipeline.resolve_entity(ind.name, "indication") is None:
                    self.graph.add_node(ind)
                    self.pipeline._index_node(ind.id, ind.name, "indication")
                    seen_indications[ind.name] = ind.id
            for comp in nodes["compounds"]:
                if self.pipeline.resolve_entity(comp.name, "compound") is None:
                    self.graph.add_node(comp)
                    self.pipeline._index_node(comp.id, comp.name, "compound")
            for ep in nodes["endpoints"]:
                if self.pipeline.resolve_entity(ep.name, "endpoint") is None:
                    self.graph.add_node(ep)
                    self.pipeline._index_node(ep.id, ep.name, "endpoint")

            # Seed endpoint_captures using the canonical (deduped) ids.
            for ep in nodes["endpoints"]:
                ep_id = self.pipeline.resolve_entity(ep.name, "endpoint")
                if ep_id is None:
                    continue
                for ind in nodes["indications"]:
                    ind_id = self.pipeline.resolve_entity(ind.name, "indication")
                    if ind_id is None:
                        continue
                    pair = (ep_id, ind_id)
                    if pair in seeded_pairs:
                        continue
                    if seed_endpoint_captures_edge(
                        self.graph, ep_id, ep.name, ind_id
                    ):
                        seeded_pairs.add(pair)
                        ec_seeded += 1
        console.print(f"  Seeded endpoint_captures edges: {ec_seeded}")

        ot_added = 0
        for ind_name, ind_id in seen_indications.items():
            try:
                ot_added += await self._fetch_ot_for_indication(ind_name, ind_id)
            except Exception:
                logger.debug("OT lookup failed for '%s'", ind_name, exc_info=True)
        console.print(f"  OT biology_drives edges: {ot_added}")

        binds_added = self.pipeline._add_compound_target_edges(trials)
        console.print(f"  Compound→target binds_to edges: {binds_added}")

    async def _fetch_ot_for_indication(
        self, indication_name: str, indication_id: str
    ) -> int:
        """Fetch OT target associations and link them to the trial-side indication node."""
        query = """
        query SearchDisease($name: String!) {
          search(queryString: $name, entityNames: ["disease"], page: {size: 1, index: 0}) {
            hits { id }
          }
        }
        """
        data = await self.ot_client._post(query, {"name": indication_name})
        hits = data["search"]["hits"]
        if not hits:
            return 0
        efo_id = hits[0]["id"]
        associations = await self.ot_client.get_disease_associations(efo_id)
        added = 0
        for assoc in associations:
            target_id = assoc["target_id"]
            try:
                self.graph.get_node(target_id)
            except KeyError:
                self.graph.add_node(TargetNode(
                    id=target_id,
                    name=assoc.get("target_symbol", target_id),
                    gene_symbol=assoc.get("target_symbol", target_id),
                ))
                self.pipeline._index_node(
                    target_id, assoc.get("target_symbol", ""), "target"
                )
            belief = score_to_prior(assoc["overall_score"], assoc["evidence_count"])
            self.graph.add_edge(GraphEdge(
                source_id=target_id,
                target_id=indication_id,
                edge_type=EdgeType.BIOLOGY_DRIVES,
                belief=belief,
                metadata={
                    "source": "opentargets",
                    "overall_score": assoc["overall_score"],
                    "efo_id": efo_id,
                },
            ))
            added += 1
        return added

    # ── Step 3: build subgraphs (with target resolution) ─────────────

    def _build_subgraphs(self, trials: list[TrialRecord]) -> list[TrialSubgraph]:
        base = self.pipeline.build_trial_subgraphs(trials)
        trial_by_id = {t.nct_id: t for t in trials}
        targets = self.graph.get_nodes_by_type("TargetNode")
        out: list[TrialSubgraph] = []
        for sg in base:
            trial = trial_by_id.get(sg.trial_id)
            if trial is not None:
                target_id = self._resolve_target_for_trial(trial, targets)
                if target_id:
                    sg = sg.model_copy(update={"target_id": target_id})
            sg = self._resolve_subgraph_via_topology(sg)
            out.append(sg)
        return out

    def _resolve_target_for_trial(
        self, trial: TrialRecord, targets: list[dict[str, Any]]
    ) -> str | None:
        text = (
            trial.title.lower()
            + " "
            + " ".join(iv.description.lower() for iv in trial.interventions)
        )
        for tnode in targets:
            symbol = (tnode.get("gene_symbol") or "").lower()
            name = (tnode.get("name") or "").lower()
            if symbol and len(symbol) >= 3 and symbol in text:
                return tnode["id"]
            if name and len(name) >= 5 and name in text:
                return tnode["id"]
        return None

    def _resolve_subgraph_via_topology(self, sg: TrialSubgraph) -> TrialSubgraph:
        """Fill mechanism_id / biology_id by walking target→indication paths.

        Uses the directional semantics of each edge type to label the nodes
        the path passes through:
          - modulates_via       : source=target, dst=mechanism
          - mechanism_affects   : source=mechanism, dst=biology
          - biology_drives      : source=biology, dst=indication

        Picks the path that resolves the most nodes. Reads only — no
        evidence is created and no beliefs are updated.
        """
        if sg.target_id == "UNKNOWN" or sg.indication_id == "UNKNOWN":
            return sg
        g = self.graph._graph
        if sg.target_id not in g or sg.indication_id not in g:
            return sg
        try:
            paths = list(
                nx.all_simple_paths(g, sg.target_id, sg.indication_id, cutoff=3)
            )
        except nx.NodeNotFound:
            return sg

        best_mech, best_bio = sg.mechanism_id, sg.biology_id
        best_score = -1
        for path in paths:
            mech, bio = "UNKNOWN", "UNKNOWN"
            for u, v in zip(path[:-1], path[1:]):
                edges_between = g.get_edge_data(u, v) or {}
                for key in edges_between:
                    if key == EdgeType.MODULATES_VIA.value and mech == "UNKNOWN":
                        mech = v
                    elif key == EdgeType.MECHANISM_AFFECTS.value:
                        if mech == "UNKNOWN":
                            mech = u
                        if bio == "UNKNOWN":
                            bio = v
                    elif key == EdgeType.BIOLOGY_DRIVES.value and bio == "UNKNOWN":
                        bio = u
            score = int(mech != "UNKNOWN") + int(bio != "UNKNOWN")
            if score > best_score:
                best_score = score
                best_mech, best_bio = mech, bio

        return sg.model_copy(
            update={"mechanism_id": best_mech, "biology_id": best_bio}
        )

    # ── Step 4: training phase ───────────────────────────────────────

    async def _train_phase(
        self,
        trials: list[TrialRecord],
        sg_by_id: dict[str, TrialSubgraph],
        concurrency: int,
    ) -> None:
        extractor = Extractor(self.anthropic)
        classifier = Classifier(self.anthropic)
        attributor = Attributor(self.graph)
        sem = asyncio.Semaphore(concurrency)

        async def annotate(trial: TrialRecord):
            async with sem:
                try:
                    extraction = await extractor.extract(trial)
                    classification = await classifier.classify(extraction)
                    return trial, classification
                except Exception:
                    logger.warning(
                        "Training annotation failed for %s", trial.nct_id, exc_info=True
                    )
                    return trial, None

        results = await asyncio.gather(*(annotate(t) for t in trials))

        n_updates = 0
        n_classified = 0
        for trial, classification in results:
            if classification is None:
                continue
            n_classified += 1
            sg = sg_by_id.get(trial.nct_id)
            if sg is None:
                continue
            updates = attributor.attribute(classification, sg)
            n_updates += len(updates)
        console.print(
            f"  Training annotated: {n_classified}/{len(trials)}; "
            f"edge updates applied: {n_updates}"
        )

    # ── Step 5: test phase ───────────────────────────────────────────

    async def _test_phase(
        self,
        trials: list[TrialRecord],
        sg_by_id: dict[str, TrialSubgraph],
        concurrency: int,
    ) -> list[TrialPrediction]:
        extractor = Extractor(self.anthropic)
        classifier = Classifier(self.anthropic)
        engine = PredictionEngine(self.graph)
        sem = asyncio.Semaphore(concurrency)

        # Predict on the pre-outcome graph BEFORE we annotate the test set,
        # so test annotations cannot leak into the predictions.
        per_trial: list[tuple[TrialRecord, PredictionResult | None]] = []
        for trial in trials:
            sg = sg_by_id.get(trial.nct_id)
            if sg is None:
                continue
            try:
                pred = engine.predict(sg, n_samples=10_000)
            except Exception:
                logger.warning("Prediction failed for %s", trial.nct_id, exc_info=True)
                pred = None
            per_trial.append((trial, pred))

        async def annotate(trial: TrialRecord):
            async with sem:
                try:
                    extraction = await extractor.extract(trial)
                    classification = await classifier.classify(extraction)
                    return trial.nct_id, extraction, classification
                except Exception:
                    logger.warning(
                        "Test annotation failed for %s", trial.nct_id, exc_info=True
                    )
                    return trial.nct_id, None, None

        annotations = await asyncio.gather(
            *(annotate(t) for t, _ in per_trial)
        )
        ann_by_id = {tid: (e, c) for tid, e, c in annotations if e is not None}

        out: list[TrialPrediction] = []
        for trial, pred in per_trial:
            if pred is None:
                continue
            ann = ann_by_id.get(trial.nct_id)
            if ann is None:
                continue
            extraction, classification = ann
            if extraction.primary_endpoint_met is None:
                continue

            actual_success = bool(extraction.primary_endpoint_met)
            actual_bottleneck = (
                _extract_actual_bottleneck(classification)
                if not actual_success
                else None
            )
            actual_failure_mode = (
                classification.primary_failure_mode if not actual_success else None
            )

            pred_bottleneck = (
                pred.weakest_link.edge_type if pred.weakest_link else None
            )
            pred_failure_mode = (
                _EDGE_TO_FAILURE_MODE.get(pred_bottleneck)
                if pred_bottleneck is not None
                else None
            )

            out.append(TrialPrediction(
                trial_id=trial.nct_id,
                predicted_p_success=pred.overall_probability,
                actual_success=actual_success,
                predicted_bottleneck=pred_bottleneck,
                actual_bottleneck=actual_bottleneck,
                predicted_failure_mode=pred_failure_mode,
                actual_failure_mode=actual_failure_mode,
                ci_lower=pred.ci_lower,
                ci_upper=pred.ci_upper,
            ))
        return out

    # ── Step 6: metrics ──────────────────────────────────────────────

    def _compute_metrics(
        self,
        predictions: list[TrialPrediction],
        n_training: int,
        n_test: int,
    ) -> BacktestResult:
        scores = [p.predicted_p_success for p in predictions]
        labels = [1 if p.actual_success else 0 for p in predictions]
        auc = _auc_roc(scores, labels)
        calibration = _calibration_bins(scores, labels)

        failed = [p for p in predictions if not p.actual_success]
        if failed:
            fm_correct = sum(
                1 for p in failed
                if p.predicted_failure_mode is not None
                and p.predicted_failure_mode == p.actual_failure_mode
            )
            bn_correct = sum(
                1 for p in failed
                if p.predicted_bottleneck is not None
                and p.predicted_bottleneck == p.actual_bottleneck
            )
            failure_mode_accuracy: float | None = fm_correct / len(failed)
            bottleneck_accuracy: float | None = bn_correct / len(failed)
        else:
            failure_mode_accuracy = None
            bottleneck_accuracy = None

        return BacktestResult(
            n_training=n_training,
            n_test=n_test,
            auc_roc=auc,
            calibration=calibration,
            failure_mode_accuracy=failure_mode_accuracy,
            bottleneck_accuracy=bottleneck_accuracy,
            individual_predictions=predictions,
        )


# ── Reporting ────────────────────────────────────────────────────────────


def print_report(result: BacktestResult) -> None:
    console.print()
    console.print(Panel(
        f"[bold]Training trials:[/bold] {result.n_training}\n"
        f"[bold]Test trials:[/bold] {result.n_test}\n"
        f"[bold]Test trials with usable predictions:[/bold] "
        f"{len(result.individual_predictions)}",
        title="Backtest Run",
    ))

    auc_str = (
        f"{result.auc_roc:.3f}" if result.auc_roc is not None
        else "N/A (single-class labels)"
    )
    bn_str = (
        f"{result.bottleneck_accuracy:.3f}"
        if result.bottleneck_accuracy is not None
        else "N/A (no failed trials in test set)"
    )
    fm_str = (
        f"{result.failure_mode_accuracy:.3f}"
        if result.failure_mode_accuracy is not None
        else "N/A (no failed trials in test set)"
    )
    console.print(Panel(
        f"[bold]Bottleneck accuracy:[/bold] {bn_str}  "
        f"[dim](primary metric — does the model identify the actual weak link?)[/dim]\n"
        f"[bold]Failure mode accuracy:[/bold] {fm_str}\n"
        f"[bold]AUC-ROC:[/bold] {auc_str}",
        title="Aggregate Metrics",
    ))

    if result.calibration:
        table = Table(title="Calibration (decile bins)")
        table.add_column("P range")
        table.add_column("n", justify="right")
        table.add_column("mean predicted", justify="right")
        table.add_column("actual rate", justify="right")
        table.add_column("delta", justify="right")
        for cb in result.calibration:
            delta = cb.actual_rate - cb.mean_predicted
            table.add_row(
                cb.bin_range,
                str(cb.n),
                f"{cb.mean_predicted:.3f}",
                f"{cb.actual_rate:.3f}",
                f"{delta:+.3f}",
            )
        console.print(table)

    if result.individual_predictions:
        table = Table(title="Per-trial predictions (first 25)")
        table.add_column("trial")
        table.add_column("P(success)", justify="right")
        table.add_column("95% CI")
        table.add_column("actual")
        table.add_column("pred bottleneck")
        table.add_column("actual bottleneck")
        for p in result.individual_predictions[:25]:
            table.add_row(
                p.trial_id,
                f"{p.predicted_p_success:.3f}",
                f"[{p.ci_lower:.2f}, {p.ci_upper:.2f}]",
                "success" if p.actual_success else "failure",
                p.predicted_bottleneck.value if p.predicted_bottleneck else "—",
                p.actual_bottleneck.value if p.actual_bottleneck else "—",
            )
        console.print(table)


# ── CLI ──────────────────────────────────────────────────────────────────


async def _fetch_pool(target_size: int) -> list[TrialRecord]:
    ct = ClinicalTrialsClient()
    console.print(
        f"[bold]Fetching up to {target_size} oncology trials with results...[/bold]"
    )
    pool = await ct.fetch_oncology_with_results(max_results=target_size)
    console.print(f"  Fetched {len(pool)} trials")
    return pool


async def _run(
    area: str,
    cutoff: date,
    test_end: date,
    max_training: int,
    max_test: int | None,
    output: str | None,
) -> None:
    if area != "oncology":
        console.print(f"[red]Unsupported area: {area}[/red]")
        return
    pool_size = (max_training + (max_test or 200)) * 3
    pool = await _fetch_pool(pool_size)
    runner = BacktestRunner()
    result = await runner.run_backtest(
        trials=pool,
        cutoff_date=cutoff,
        test_window_end=test_end,
        max_training=max_training,
        max_test=max_test,
    )
    print_report(result)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(result.model_dump_json(indent=2))
        console.print(f"\n[bold]Saved result to {output}[/bold]")


async def _run_sanity_check() -> None:
    """Small validation run before scaling up: 50 train, 10 test."""
    console.print(Panel(
        "[bold]Sanity check[/bold]: populate the graph with 50 oncology trials\n"
        "completed before 2022-01-01, then predict 10 trials from 2022–2023.\n"
        "[dim]Verify the pipeline runs end-to-end before scaling up.[/dim]",
        title="Eroom Bio Backtest — Sanity Check",
    ))
    pool = await _fetch_pool(target_size=300)
    runner = BacktestRunner()
    result = await runner.run_backtest(
        trials=pool,
        cutoff_date=date(2022, 1, 1),
        test_window_end=date(2023, 12, 31),
        max_training=50,
        max_test=10,
    )
    print_report(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrospective backtest of Eroom Bio predictions."
    )
    parser.add_argument("--area", default="oncology")
    parser.add_argument(
        "--cutoff",
        type=lambda s: date.fromisoformat(s),
        default=date(2022, 1, 1),
        help="Training/test split date (YYYY-MM-DD). Trials completed before this go to training.",
    )
    parser.add_argument(
        "--test-window-end",
        type=lambda s: date.fromisoformat(s),
        default=date(2023, 12, 31),
        help="End of test window (YYYY-MM-DD).",
    )
    parser.add_argument("--max-training", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--output", default=None, help="Optional JSON file to save the full result."
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a small 50-train / 10-test pipeline check and exit.",
    )
    args = parser.parse_args()

    if args.sanity_check:
        asyncio.run(_run_sanity_check())
    else:
        asyncio.run(_run(
            area=args.area,
            cutoff=args.cutoff,
            test_end=args.test_window_end,
            max_training=args.max_training,
            max_test=args.max_test,
            output=args.output,
        ))


if __name__ == "__main__":
    main()
