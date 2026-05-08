"""Orchestrator that constructs the initial knowledge graph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import anthropic
from rich.console import Console

from src.annotation.extractor import _call_messages_with_backoff
from src.graph.models import (
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointClass,
    EndpointNode,
    EndpointType,
    GraphEdge,
    IndicationNode,
    MechanismCategory,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    SubgroupFeature,
    TargetNode,
    TrialArm,
    TrialNode,
    TrialOutcome,
    TrialSubgraph,
    normalize_entity,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ArmGroup,
    ClinicalTrialsClient,
    TrialRecord,
    is_drug_like,
    map_trial_to_graph_nodes,
)
from src.ingestion.lincs import (
    LINCSClient,
    _category_to_mechanism_type,
    populate_lincs_signatures,
)
from src.ingestion.opentargets import (
    OpenTargetsClient,
    score_to_prior,
)

logger = logging.getLogger(__name__)
console = Console()

# Haiku is fast + cheap for short categorical labels. The structural inferences
# (endpoint type, mechanism, population) are short — Haiku is more than enough.
INFERENCE_MODEL = "claude-haiku-4-5-20251001"


# ── Endpoint classification ──────────────────────────────────────────────


# Deterministic endpoint classifier. Keyword-first so we don't burn an
# LLM call on what's almost always a textual match. The text
# "Progression Free Survival (PFS)" / "Overall Survival" / "Median PFS"
# all unambiguously identify their EndpointClass.
# Order matters: more-specific patterns first so e.g. "distant
# metastasis-free survival" matches DMFS, not DFS.
_ENDPOINT_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdistant[\s-]*metastas[ie]s[\s-]*free\s+survival\b|\bdmfs\b", re.I), "DMFS"),
    (re.compile(r"\brelapse[\s-]*free\s+survival\b|\brfs\b", re.I), "RFS"),
    (re.compile(r"\bdisease[\s-]*free\s+survival\b|\bdfs\b", re.I), "DFS"),
    (re.compile(r"\bduration\s+of\s+(?:overall\s+)?response\b|\bdor\b", re.I), "DOR"),
    (re.compile(r"\bprogression[\s-]*free\s+survival\b|\bpfs\b", re.I), "PFS"),
    (re.compile(r"\boverall\s+survival\b|\bos\b(?!\w)", re.I), "OS"),
    (re.compile(r"\btime\s+to\s+progression\b|\bttp\b", re.I), "TTP"),
    (re.compile(r"\bobjective\s+response\s+rate\b|\boverall\s+response\s+rate\b|\bbest\s+overall\s+response\b|\borr\b", re.I), "ORR"),
    (re.compile(r"\bcomplete\s+response\b|\bcomplete\s+remission\b|\bcr\s+rate\b", re.I), "CR"),
    # Phase I / PD-safety primaries — AE counts, DLTs, MTD, ECOG, vitals,
    # routine labs. These are legitimate primary endpoints that don't
    # directly capture indication-level efficacy; they get a low-but-non-
    # zero prior so the graph still has an EndpointNode pointing
    # somewhere instead of falling into "other".
    (re.compile(
        r"\b(?:adverse\s+events?|serious\s+adverse|saes?|aes?\b|"
        r"dose[\s-]*limiting\s+toxicit|dlt\b|"
        r"maximum\s+tolerated\s+dose|mtd\b|"
        r"ecog\s+performance|"
        # vitals
        r"blood\s+pressure|systolic|diastolic|heart\s+rate|"
        r"temperature|body\s+weight|\bweight\b|"
        r"oxygen\s+saturation|spo2\b|"
        # cardiac
        r"ventricular\s+ejection|lvef\b|electrocardiogram|ecg\b|"
        # routine labs
        r"hematology\s+parameter|hematolog(?:y|ic)\b|"
        r"clinical\s+chemistry|chemistry\s+parameter|urinalysis\b|"
        r"liver\s+function|renal\s+function)",
        re.I,
    ), "safety"),
]


# Suffix qualifiers that don't change the underlying endpoint class —
# "PFS by investigator" / "PFS by BICR" / "OS by independent review" all
# describe the same kind of outcome from different adjudicators. Stripping
# them keeps the regex pool focused on the endpoint's *meaning*.
_ENDPOINT_QUALIFIER_SUFFIXES = re.compile(
    r"\s+by\s+(?:investigator|bicr|irc|"
    r"independent\s+(?:central\s+)?review(?:\s+committee)?|"
    r"central\s+review|local\s+review|"
    r"investigator\s+assessment|local\s+investigator)"
    r"(?:\s+assessment)?",
    re.I,
)


def _normalize_endpoint_text(measure_text: str) -> str:
    """Strip parenthetical clarifiers and reviewer suffixes.

    Examples:
      "Progression-Free Survival (PFS) by investigator"
        → "Progression-Free Survival"
      "ORR (CR + PR) in BRAF V600E"
        → "ORR in BRAF V600E"
      "OS [Phase 2 cohort]" → "OS"

    Done so the deterministic classifier and Haiku cache see canonical
    text — different reviewer suffixes for the same trial's PFS read no
    longer cause cache misses or class mismatches.
    """
    text = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", measure_text or "")
    text = _ENDPOINT_QUALIFIER_SUFFIXES.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_endpoint_deterministic(measure_text: str) -> str:
    """Map an outcome-measure string to an EndpointClass value.

    Returns the EndpointClass enum *value* (e.g. "PFS", "OS"). Falls back
    to "other" when no keyword matches — same fallback the LLM-driven
    classifier uses. Tries the raw text first, then a normalized
    pass with parentheticals + reviewer suffixes stripped, so e.g.
    "Disease-Free Survival (DFS), as Assessed by Investigator" still
    classifies to DFS even when surrounding noise would otherwise miss.
    """
    text = (measure_text or "").strip()
    if not text:
        return "other"
    for pattern, cls in _ENDPOINT_KEYWORDS:
        if pattern.search(text):
            return cls
    normalized = _normalize_endpoint_text(text)
    if normalized and normalized != text:
        for pattern, cls in _ENDPOINT_KEYWORDS:
            if pattern.search(normalized):
                return cls
    return "other"


# Beta(α, β) priors per endpoint class. Higher α/β reflects more regulatory
# precedent for translating endpoint movement into disease-level benefit.
ENDPOINT_PRIORS_BY_CLASS: dict[str, tuple[float, float]] = {
    "OS": (3.0, 1.0),
    "DFS": (2.5, 1.0),
    "RFS": (2.5, 1.0),  # accepted regulatory endpoint in adjuvant melanoma
    "composite_survival": (2.5, 1.0),
    "PFS": (2.0, 1.0),
    "TTP": (2.0, 1.0),
    "CR": (1.7, 1.0),
    "DMFS": (1.5, 1.0),  # used in adjuvant trials but less precedent than DFS
    "DOR": (1.5, 1.0),   # tracks response durability conditional on responding
    "ORR": (1.5, 1.0),
    "composite_response": (1.5, 1.0),
    "biomarker": (1.2, 1.0),
    "PRO": (1.0, 1.0),
    # Safety endpoints are legitimate primary outcomes (Phase I, dose-finding)
    # but don't directly capture efficacy at the indication level. Beta(1, 1.5)
    # → mean ~0.4 — slightly biased toward "doesn't capture clinical benefit"
    # with low evidence so trial outcomes can still update.
    "safety": (1.0, 1.5),
    "other": (1.0, 1.0),
}
ENDPOINT_CLASSES = list(ENDPOINT_PRIORS_BY_CLASS.keys())


class JSONCache:
    """Tiny dict-style cache that flushes to a JSON file on every write.

    Used for structural LLM classifications (endpoint type, mechanism, population)
    so the same input is never paid for twice.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("Cache %s unreadable; starting empty", path)
                self._data = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._data)


async def classify_endpoint_with_llm(
    client: anthropic.AsyncAnthropic,
    endpoint_name: str,
    cache: JSONCache,
) -> str:
    """Classify a primary endpoint name into one of ENDPOINT_CLASSES.

    Order: cache → deterministic keyword pattern → LLM. The keyword
    short-circuit handles unambiguous strings (OS, ORR, DLT, heart rate,
    …) without burning an LLM call AND without letting the LLM override
    a clear match — e.g. 'heart rate' belongs to safety regardless of
    how the model decides to interpret 'change from baseline' framing.
    """
    # Cache by both raw name and normalized form. The raw key preserves
    # backward compat with the existing on-disk cache; the normalized key
    # collapses reviewer-suffix variants ("PFS by investigator" ≡ "PFS by
    # BICR") so the second variant doesn't pay for a redundant Haiku call.
    cached = cache.get(endpoint_name)
    if cached is not None:
        return cached
    normalized = _normalize_endpoint_text(endpoint_name)
    if normalized and normalized != endpoint_name:
        cached_norm = cache.get(normalized)
        if cached_norm is not None:
            cache.set(endpoint_name, cached_norm)
            return cached_norm
    deterministic = classify_endpoint_deterministic(endpoint_name)
    if deterministic != "other":
        cache.set(endpoint_name, deterministic)
        if normalized and normalized != endpoint_name:
            cache.set(normalized, deterministic)
        return deterministic
    classes = ", ".join(ENDPOINT_CLASSES)
    user_msg = (
        f"Clinical trial primary endpoint: {endpoint_name!r}\n\n"
        f"Classify into ONE of: {classes}\n\n"
        "Map to the closest matching category. Hints:\n"
        "  - 'best overall response', 'overall response', 'tumor response', "
        "'objective response' → ORR\n"
        "  - 'clinical benefit rate', 'disease control rate' → "
        "composite_response\n"
        "  - 'duration of response', 'duration of clinical benefit' → DOR\n"
        "  - 'distant metastasis-free survival' → DMFS (not DFS)\n"
        "  - 'relapse-free survival' → RFS (distinct from DFS)\n"
        "  - immune readouts (T-cell counts, cytokines, immunogenicity) → "
        "biomarker\n"
        "  - patient-reported outcomes (QoL, fatigue scales) → PRO\n"
        "  - safety/tolerability primary endpoints (AEs, SAEs, DLTs, MTD, "
        "ECOG, vital signs, ECG findings) → safety\n"
        "Use 'other' ONLY when the endpoint genuinely fits no category — "
        "rare. Phase I tolerability primaries belong in 'safety', not "
        "'other'.\n\n"
        "Reply with only the category name. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    classification = raw if raw in ENDPOINT_CLASSES else "other"
    cache.set(endpoint_name, classification)
    return classification


def seed_endpoint_captures_edge(
    graph: GraphStore,
    endpoint_id: str,
    endpoint_name: str,
    indication_id: str,
    classification: str,
) -> bool:
    """Add an endpoint_captures edge with the prior for the given class."""
    prior = ENDPOINT_PRIORS_BY_CLASS.get(classification)
    if prior is None:
        return False
    alpha, beta_val = prior
    graph.add_edge(GraphEdge(
        source_id=endpoint_id,
        target_id=indication_id,
        edge_type=EdgeType.ENDPOINT_CAPTURES,
        belief=EdgeBeliefState(alpha=alpha, beta=beta_val),
        metadata={
            "source": "endpoint_type_prior",
            "endpoint_name": endpoint_name,
            "classification": classification,
        },
    ))
    return True


# ── Mechanism inference ──────────────────────────────────────────────────


def _sanitize_label(text: str, fallback: str = "unknown") -> str:
    """Coerce LLM free-text into a snake_case identifier safe to embed in node IDs."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    return cleaned[:60] or fallback


async def infer_mechanism_for_trial(
    client: anthropic.AsyncAnthropic,
    trial: TrialRecord,
    target_node: dict[str, Any],
    cache: JSONCache,
) -> str:
    """Map a trial+target pair to a MechanismCategory value.

    LLM output is forced through ``normalize_entity(..., "MechanismNode")``,
    which validates against ``MechanismCategory`` and falls back to "other"
    on any unknown response.
    """
    cached = cache.get(trial.nct_id)
    if cached is not None:
        return cached
    intervention_text = "; ".join(
        f"{iv.name}: {iv.description}".strip()
        for iv in trial.interventions
        if is_drug_like(iv) and iv.name
    ) or "unknown"
    target_symbol = target_node.get("gene_symbol") or target_node.get("name") or ""
    categories = ", ".join(c.value for c in MechanismCategory)
    user_msg = (
        f"Drug intervention: {intervention_text}\n"
        f"Target gene/protein: {target_symbol}\n\n"
        f"Classify the mechanism of action into ONE of these categories:\n"
        f"{categories}\n\n"
        "Reply with only the category value (e.g. 'kinase_inhibition'). "
        "If none fit, reply 'other'. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    label = normalize_entity(raw or "other", "MechanismNode")
    cache.set(trial.nct_id, label)
    return label


# ── Population inference ─────────────────────────────────────────────────


async def infer_population_for_trial(
    client: anthropic.AsyncAnthropic,
    trial: TrialRecord,
    cache: JSONCache,
) -> str:
    """Infer a structured population id ``{indication}_{biomarker}_{line}``.

    LLM output is validated by ``normalize_entity(..., "PopulationNode")``;
    if the response doesn't match the three-segment format we fall back to
    ``unselected_unselected_unselected``.
    """
    cached = cache.get(trial.nct_id)
    if cached is not None:
        return cached
    user_msg = (
        f"Trial title: {trial.title}\n"
        f"Conditions: {', '.join(trial.conditions) or 'unspecified'}\n\n"
        "Reply with one population id in the strict snake_case format "
        "'{indication}_{biomarker_status}_{line_of_therapy}'. Examples:\n"
        "  - melanoma_pdl1_positive_first_line\n"
        "  - nsclc_egfr_mutant_second_line\n"
        "  - aml_unselected_treatment_naive\n"
        "Use 'unselected' when a segment is unknown. Three underscore-separated "
        "segments only. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=30,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    try:
        label = normalize_entity(raw, "PopulationNode")
    except ValueError:
        logger.warning(
            "Population '%s' for %s did not match the canonical format; using fallback",
            raw, trial.nct_id,
        )
        label = "unselected_unselected_unselected"
    cache.set(trial.nct_id, label)
    return label

# Placeholder for unresolvable subgraph fields
_UNKNOWN = "UNKNOWN"


class PopulationPipeline:
    def __init__(
        self,
        graph: GraphStore,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        cache_dir: Path = Path("data/cache"),
    ) -> None:
        self.graph = graph
        self._ct_client = ClinicalTrialsClient()
        self._ot_client = OpenTargetsClient()
        self._anthropic = anthropic_client
        self._cache_dir = cache_dir
        # name → node_id index, keyed by (normalized_name, node_type)
        self._entity_index: dict[tuple[str, str], str] = {}

    # ── Entity resolution ────────────────────────────────────────────────

    def resolve_entity(self, name: str, entity_type: str) -> str | None:
        """Normalize a name and look up its node ID.

        Uses exact string matching + lowercasing.
        TODO: fuzzy matching / NER for better resolution.
        """
        key = (_normalize(name), entity_type)
        return self._entity_index.get(key)

    def _index_node(self, node_id: str, name: str, entity_type: str) -> None:
        key = (_normalize(name), entity_type)
        self._entity_index[key] = node_id

    # ── Main pipeline ────────────────────────────────────────────────────

    async def populate_oncology(
        self,
        max_trials: int = 500,
        include_terminated_no_results: bool = True,
        condition: str = "cancer",
        trials: list[TrialRecord] | None = None,
    ) -> dict[str, Any]:
        # Step 1: Fetch trials (or use the caller's pre-fetched list).
        # Drivers that need the same TrialRecord set for both populate and
        # downstream extraction can pass `trials=` to avoid a duplicate
        # CT.gov roundtrip and any drift between the two fetches.
        if trials is None:
            console.print(
                f"[bold]Fetching up to {max_trials} {condition} trials...[/bold]"
            )
            trials = await self._ct_client.fetch_oncology_with_results(
                max_results=max_trials, condition=condition,
            )
            console.print(f"  Fetched {len(trials)} with-results trials")

            if include_terminated_no_results:
                terminated = await self._ct_client.fetch_oncology_terminated_with_reason(
                    max_results=max_trials, condition=condition,
                )
                seen = {t.nct_id for t in trials}
                added = [t for t in terminated if t.nct_id not in seen]
                trials.extend(added)
                console.print(
                    f"  +{len(added)} terminated/withdrawn trials with why_stopped"
                )
        else:
            console.print(f"[bold]Using {len(trials)} pre-fetched trials[/bold]")

        # Step 2: Extract and add canonical compound + indication nodes per trial.
        # Endpoint nodes are created in step 3 after LLM classification, so
        # their canonical id ({EndpointClass}_{indication}) is well-formed.
        console.print("[bold]Extracting graph nodes from trials...[/bold]")
        seen_indications: dict[str, str] = {}  # canonical_id → original name
        for trial in trials:
            nodes = map_trial_to_graph_nodes(trial)
            for ind in nodes["indications"]:
                self.graph.add_node(ind)
                self._index_node(ind.id, ind.name, "indication")
                seen_indications.setdefault(ind.id, ind.name)
            for comp in nodes["compounds"]:
                self.graph.add_node(comp)
                self._index_node(comp.id, comp.name, "compound")

        stats_after_nodes = self.graph.stats()
        console.print(
            f"  Nodes: {stats_after_nodes['node_count']} "
            f"({stats_after_nodes['node_types']})"
        )

        # Step 3: Canonical EndpointNodes via LLM classification.
        # One node per (EndpointClass, indication) — id = "{class}_{indication}".
        console.print("[bold]Classifying endpoints into canonical classes...[/bold]")
        ep_added = await self._populate_canonical_endpoints(trials)
        console.print(f"  Added {ep_added} canonical endpoint nodes")

        # Step 4: Trial-driven Open Targets population.
        # Fetch only the compound→target relationships our trials actually
        # exercise, then look up disease-association scores for each
        # (target, indication) pair the trials imply. This avoids loading
        # the full disease-association catalog (which adds hundreds of
        # targets no trial in our set tests).
        console.print(
            f"[bold]Resolving compound→target via Open Targets ({len(seen_indications)} indications)...[/bold]"
        )
        compound_targets, ot_binds_added = await self._populate_compound_targets(trials)
        console.print(
            f"  Added {ot_binds_added} binds_to edges across "
            f"{sum(1 for v in compound_targets.values() if v)} OT-resolved compounds"
        )

        console.print(
            "[bold]Wiring target→indication priors from Open Targets...[/bold]"
        )
        ot_pair_added = await self._populate_target_indication_priors(
            trials, compound_targets, seen_indications,
        )
        console.print(f"  Added {ot_pair_added} biology_drives edges (trial-implied pairs)")

        # Name-matching fallback: catches binds_to relationships for
        # compounds OT couldn't resolve (e.g. peptide vaccines).
        console.print("[bold]Cross-referencing compound-target edges...[/bold]")
        binds_added = self._add_compound_target_edges(trials)
        console.print(f"  Added {binds_added} binds_to edges (name-match fallback)")

        # Step 5: Build trial subgraphs
        console.print("[bold]Building trial subgraphs...[/bold]")
        subgraphs = self.build_trial_subgraphs(trials)
        console.print(f"  Built {len(subgraphs)} trial subgraphs")

        # Step 5.5: Resolve mechanism per trial.
        # For each trial, infer a MechanismCategory from its primary target +
        # intervention text and add the canonical target→mechanism
        # modulates_via edge.
        console.print("[bold]Resolving mechanisms per trial...[/bold]")
        mech_added = await self._populate_trial_mechanisms(trials, compound_targets)
        console.print(f"  Added {mech_added} modulates_via edges")

        # Step 6: LINCS L1000 Touchstone signatures.
        # Adds mechanism_affects evidence and materializes BiologyNodes for
        # any Reactome pathway consistently perturbed by a Touchstone
        # compound whose target+mechanism are already in the graph.
        # Skipped silently if CLUE_API_KEY is unset.
        console.print("[bold]Adding LINCS L1000 mechanism→biology evidence...[/bold]")
        try:
            lincs_client = LINCSClient()
            lincs_added = await populate_lincs_signatures(lincs_client, self.graph)
            console.print(f"  Added {lincs_added} LINCS evidence records")
        except RuntimeError as exc:
            console.print(f"  [yellow]Skipped LINCS:[/yellow] {exc}")

        # Step 6.5: Resolve a per-trial biology so chains can be traversed
        # end-to-end. Uses a slug-form BiologyNode '{mech}__{indication}'
        # for deterministic chain wiring; LINCS-derived Reactome biology
        # nodes (when present) supply the upstream evidence. Independent
        # of LINCS so chains close even without CLUE_API_KEY.
        console.print("[bold]Resolving biology per trial...[/bold]")
        bio_added = self._populate_trial_biology(trials)
        console.print(f"  Added {bio_added} biology nodes / chain links")

        # Summary
        final = self.graph.stats()
        summary = {
            "trials_fetched": len(trials),
            "compounds": final["node_types"].get("CompoundNode", 0),
            "targets": final["node_types"].get("TargetNode", 0),
            "indications": final["node_types"].get("IndicationNode", 0),
            "endpoints": final["node_types"].get("EndpointNode", 0),
            "edges": final["edge_count"],
            "trial_subgraphs": len(subgraphs),
        }
        console.print(
            f"\n[bold green]Graph populated:[/bold green] "
            f"{summary['compounds']} compounds, "
            f"{summary['targets']} targets, "
            f"{summary['indications']} indications, "
            f"{summary['edges']} edges, "
            f"{summary['trial_subgraphs']} trial subgraphs"
        )
        return summary

    async def _lookup_efo_for_indication(self, indication_name: str) -> str | None:
        """Resolve an indication string to an EFO id via OT search.

        Returns the EFO id, or None if no hit. Cached on the instance so
        repeat lookups (parent + subgroup populations of the same disease)
        don't re-query.
        """
        if not hasattr(self, "_efo_cache"):
            self._efo_cache: dict[str, str | None] = {}
        if indication_name in self._efo_cache:
            return self._efo_cache[indication_name]
        query = """
        query SearchDisease($name: String!) {
          search(queryString: $name, entityNames: ["disease"], page: {size: 1, index: 0}) {
            hits { id }
          }
        }
        """
        try:
            data = await self._ot_client._post(query, {"name": indication_name})
        except Exception:
            logger.debug("EFO lookup failed for '%s'", indication_name, exc_info=True)
            self._efo_cache[indication_name] = None
            return None
        hits = data["search"]["hits"]
        efo = hits[0]["id"] if hits else None
        self._efo_cache[indication_name] = efo
        return efo

    async def _populate_compound_targets(
        self, trials: list[TrialRecord]
    ) -> tuple[dict[str, list[str]], int]:
        """For each drug-like compound across the trials, OT-resolve targets.

        Returns ``(compound_targets, binds_added)`` where compound_targets
        maps compound_id → list of Ensembl target ids. Adds TargetNodes
        and binds_to edges (alpha=4, beta=1 — strong prior since OT-sourced).
        """
        cache_path = self._cache_dir / "ot_drug_targets.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache: dict[str, dict[str, Any]] = (
                json.loads(cache_path.read_text()) if cache_path.exists() else {}
            )
        except json.JSONDecodeError:
            cache = {}

        # Collect (compound_id, intervention_name) pairs from drug-like
        # interventions. Same compound may surface under multiple aliases;
        # we cache by the lowercased name.
        seen: dict[str, str] = {}  # compound_id → first intervention name we saw
        for trial in trials:
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                cid = self.resolve_entity(iv.name, "compound")
                if not cid or cid in seen:
                    continue
                seen[cid] = iv.name

        compound_targets: dict[str, list[str]] = {}
        binds_added = 0
        for cid, iv_name in seen.items():
            cache_key = iv_name.lower()
            if cache_key in cache:
                drug_data = cache[cache_key]
            else:
                try:
                    drug_data = await self._ot_client.get_drug_with_targets(iv_name)
                except KeyError:
                    drug_data = {"chembl_id": None, "targets": []}
                except Exception:
                    logger.debug(
                        "OT drug lookup failed for '%s'", iv_name, exc_info=True,
                    )
                    drug_data = {"chembl_id": None, "targets": []}
                cache[cache_key] = drug_data
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

            target_ids: list[str] = []
            for t in drug_data.get("targets") or []:
                ensembl = t.get("target_id")
                if not ensembl:
                    continue
                symbol = t.get("approved_symbol") or ensembl
                approved_name = t.get("approved_name") or symbol
                try:
                    self.graph.get_node(ensembl)
                except KeyError:
                    self.graph.add_node(TargetNode(
                        id=ensembl,
                        name=approved_name,
                        gene_symbol=symbol,
                        metadata={"source": "opentargets"},
                    ))
                    self._index_node(ensembl, symbol, "target")
                if not self.graph._graph.has_edge(  # noqa: SLF001
                    cid, ensembl, key=EdgeType.BINDS_TO.value,
                ):
                    self.graph.add_edge(GraphEdge(
                        source_id=cid,
                        target_id=ensembl,
                        edge_type=EdgeType.BINDS_TO,
                        belief=EdgeBeliefState(alpha=4.0, beta=1.0),
                        metadata={
                            "source": "opentargets",
                            "drug_chembl_id": drug_data.get("chembl_id"),
                        },
                    ))
                    binds_added += 1
                target_ids.append(ensembl)
            compound_targets[cid] = target_ids
        return compound_targets, binds_added

    async def _populate_target_indication_priors(
        self,
        trials: list[TrialRecord],
        compound_targets: dict[str, list[str]],
        seen_indications: dict[str, str],
    ) -> int:
        """For each (target, indication) pair the trials imply, add a
        biology_drives edge with an OT-score-derived prior.

        A pair is "implied" when at least one trial uses a compound bound
        to ``target`` for an indication ``indication``. Skips pairs OT
        has no association data for; uses ``score_to_prior`` for the rest.
        """
        cache_path = self._cache_dir / "ot_target_disease_scores.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache: dict[str, dict[str, Any]] = (
                json.loads(cache_path.read_text()) if cache_path.exists() else {}
            )
        except json.JSONDecodeError:
            cache = {}

        # Resolve EFO id per indication once.
        efo_by_indication: dict[str, str] = {}
        for ind_id, ind_name in seen_indications.items():
            efo = await self._lookup_efo_for_indication(ind_name)
            if efo:
                efo_by_indication[ind_id] = efo

        # Walk trials to build the (target, indication) pair set.
        pairs: set[tuple[str, str]] = set()
        for trial in trials:
            inds = [self.resolve_entity(c, "indication") for c in trial.conditions]
            inds = [i for i in inds if i and i in efo_by_indication]
            for iv in trial.interventions:
                if not is_drug_like(iv):
                    continue
                cid = self.resolve_entity(iv.name, "compound")
                if not cid:
                    continue
                for tid in compound_targets.get(cid, []):
                    for ind_id in inds:
                        pairs.add((tid, ind_id))

        added = 0
        for tid, ind_id in pairs:
            efo = efo_by_indication[ind_id]
            cache_key = f"{tid}__{efo}"
            if cache_key in cache:
                row = cache[cache_key]
            else:
                try:
                    rows = await self._ot_client.get_associations(
                        tid, disease_id=efo, min_score=0.0,
                    )
                except Exception:
                    logger.debug(
                        "OT score lookup failed for %s × %s",
                        tid, efo, exc_info=True,
                    )
                    rows = []
                row = rows[0] if rows else {}
                cache[cache_key] = row
                cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

            if not row:
                # No association evidence — skip rather than fabricating
                # a uniform prior for a target-disease pair OT has no
                # data on. Trial outcomes will still update the edge if
                # one is added by attribution.
                continue
            if self.graph._graph.has_edge(  # noqa: SLF001
                tid, ind_id, key=EdgeType.BIOLOGY_DRIVES.value,
            ):
                continue
            belief = score_to_prior(
                row.get("overall_score", 0.0), row.get("evidence_count", 0)
            )
            self.graph.add_edge(GraphEdge(
                source_id=tid,
                target_id=ind_id,
                edge_type=EdgeType.BIOLOGY_DRIVES,
                belief=belief,
                metadata={
                    "source": "opentargets",
                    "efo_id": efo,
                    "overall_score": row.get("overall_score"),
                    "datatypes": row.get("datatypes"),
                },
            ))
            added += 1
        return added

    async def _populate_trial_mechanisms(
        self,
        trials: list[TrialRecord],
        compound_targets: dict[str, list[str]],
    ) -> int:
        """Infer one mechanism per trial; add MechanismNode + modulates_via edge.

        For each trial:
          1. Pick the trial's primary target — first OT-resolved target of
             the first arm's first compound. Falls back to ``UNKNOWN`` only
             when no compound resolves to any target; mechanism inference
             still runs (the LLM uses intervention text as well).
          2. Call ``infer_mechanism_for_trial`` to get a MechanismCategory.
          3. Create the MechanismNode if missing (canonical id = category
             value, e.g. ``checkpoint_blockade``).
          4. Add ``target → mechanism`` modulates_via edge.

        Returns the number of new modulates_via edges added.
        """
        if self._anthropic is None:
            logger.info("No anthropic client; skipping mechanism inference")
            return 0

        cache = JSONCache(self._cache_dir / "mechanism_inferences.json")
        added = 0
        for trial in trials:
            # Pick the first drug-like intervention's compound id.
            compound_id: str | None = None
            for iv in trial.interventions:
                if not is_drug_like(iv) or not iv.name:
                    continue
                cid = self.resolve_entity(iv.name, "compound")
                if cid:
                    compound_id = cid
                    break
            target_id: str | None = None
            if compound_id:
                tids = compound_targets.get(compound_id) or []
                if tids:
                    target_id = tids[0]

            if target_id:
                try:
                    target_node = self.graph.get_node(target_id)
                except KeyError:
                    target_node = {}
            else:
                target_node = {}

            mech_value = await infer_mechanism_for_trial(
                self._anthropic, trial, target_node, cache,
            )
            mech_id = normalize_entity(mech_value, "MechanismNode")

            try:
                self.graph.get_node(mech_id)
            except KeyError:
                self.graph.add_node(MechanismNode(
                    id=mech_id,
                    name=mech_id.replace("_", " "),
                    mechanism_type=_category_to_mechanism_type(
                        MechanismCategory(mech_id)
                    ),
                ))

            # Write the resolved ids back into the trial's chains so the
            # prediction engine can traverse them. build_trial_subgraphs
            # creates chains with target_id/mechanism_id = UNKNOWN; without
            # this rewrite, PredictionEngine._collect_edges skips every
            # causal-chain edge.
            try:
                ts = self.graph.get_trial_subgraph_by_id(trial.nct_id)
            except KeyError:
                ts = None
            if ts is not None and (target_id is not None or mech_id):
                updates: dict[str, str] = {}
                if target_id is not None:
                    updates["target_id"] = target_id
                if mech_id:
                    updates["mechanism_id"] = mech_id
                new_chains = [c.model_copy(update=updates) for c in ts.chains]
                self.graph.set_trial_subgraph(
                    ts.model_copy(update={"chains": new_chains})
                )

            if target_id is None:
                continue
            if self.graph._graph.has_edge(  # noqa: SLF001
                target_id, mech_id, key=EdgeType.MODULATES_VIA.value,
            ):
                continue
            self.graph.add_edge(GraphEdge(
                source_id=target_id,
                target_id=mech_id,
                edge_type=EdgeType.MODULATES_VIA,
                belief=EdgeBeliefState(alpha=3.0, beta=1.0),
                metadata={
                    "source": "trial_inference",
                    "trial_id": trial.nct_id,
                },
            ))
            added += 1
        return added

    def _populate_trial_biology(self, trials: list[TrialRecord]) -> int:
        """Ensure every trial chain has a resolvable biology node.

        For each trial whose mechanism + indication are resolved, build a
        slug-form BiologyNode ``{mechanism}__{indication}``. Wire
        ``mechanism → biology`` (mechanism_affects) and ``biology →
        indication`` (biology_drives) with weak priors so trial outcomes
        can update them through attribution. Then rewrite the trial's
        chains so ``biology_id`` no longer references UNKNOWN.

        Returns the count of new biology nodes + chain rewrites combined.
        """
        from src.graph.models import BiologyNode  # local import: keeps
        # the module's top-level Pydantic dependency surface unchanged.

        added = 0
        for trial in trials:
            try:
                ts = self.graph.get_trial_subgraph_by_id(trial.nct_id)
            except KeyError:
                continue
            if not ts.chains:
                continue

            # Use the trial's first chain to read mech+indication. Within
            # a trial, _populate_trial_mechanisms writes the same mech_id
            # onto every chain, and indication is fixed per trial.
            sample = ts.chains[0]
            mech_id = sample.mechanism_id
            ind_id = sample.indication_id
            if mech_id == _UNKNOWN or ind_id == _UNKNOWN:
                continue
            try:
                MechanismCategory(mech_id)
            except ValueError:
                # Mechanism inferred but not canonical — skip rather than
                # silently producing an invalid slug.
                continue

            biology_id = normalize_entity(
                f"{mech_id}__{ind_id}", "BiologyNode"
            )
            try:
                self.graph.get_node(biology_id)
            except KeyError:
                self.graph.add_node(BiologyNode(
                    id=biology_id,
                    name=f"{mech_id.replace('_', ' ')} biology in {ind_id}",
                    pathway_ids=[],
                ))
                added += 1

            # mechanism_affects: mech → biology
            if not self.graph._graph.has_edge(  # noqa: SLF001
                mech_id, biology_id, key=EdgeType.MECHANISM_AFFECTS.value,
            ):
                self.graph.add_edge(GraphEdge(
                    source_id=mech_id,
                    target_id=biology_id,
                    edge_type=EdgeType.MECHANISM_AFFECTS,
                    belief=EdgeBeliefState(alpha=2.0, beta=1.0),
                    metadata={"source": "trial_biology_fallback"},
                ))

            # biology_drives: biology → indication. Borrow the OT-derived
            # target→indication prior when it exists (the slug biology
            # represents that target's mechanism affecting this disease,
            # so the strength of the target↔disease association is the
            # right prior). Fall back to weak Beta(1, 1) otherwise.
            if not self.graph._graph.has_edge(  # noqa: SLF001
                biology_id, ind_id, key=EdgeType.BIOLOGY_DRIVES.value,
            ):
                prior = EdgeBeliefState(alpha=1.0, beta=1.0)
                target_id = sample.target_id
                if target_id != _UNKNOWN:
                    try:
                        prior = self.graph.get_edge_belief(
                            target_id, ind_id, EdgeType.BIOLOGY_DRIVES,
                        )
                    except KeyError:
                        pass
                self.graph.add_edge(GraphEdge(
                    source_id=biology_id,
                    target_id=ind_id,
                    edge_type=EdgeType.BIOLOGY_DRIVES,
                    belief=EdgeBeliefState(alpha=prior.alpha, beta=prior.beta),
                    metadata={
                        "source": "trial_biology_fallback",
                        "borrowed_from": (
                            f"{target_id}->{ind_id}"
                            if target_id != _UNKNOWN else None
                        ),
                    },
                ))

            # Rewrite chains with the resolved biology id.
            new_chains = [
                c.model_copy(update={"biology_id": biology_id})
                if c.biology_id == _UNKNOWN else c
                for c in ts.chains
            ]
            if new_chains != list(ts.chains):
                self.graph.set_trial_subgraph(
                    ts.model_copy(update={"chains": new_chains})
                )
                added += sum(
                    1 for c in new_chains if c.biology_id == biology_id
                )
        return added

    async def _populate_canonical_endpoints(
        self, trials: list[TrialRecord]
    ) -> int:
        """Classify each trial's primary outcomes and create canonical endpoint nodes.

        For each (trial, primary_outcome, indication) triple:
          1. LLM-classify the outcome's measure into an ``EndpointClass``.
          2. Create (if missing) ``EndpointNode`` with id ``{class}_{indication_id}``.
          3. Seed an ``endpoint_captures`` edge with the class-specific prior.

        Skipped silently if no Anthropic client was provided (lets the
        pipeline run in offline / fixture mode).

        Returns the number of *new* endpoint nodes created.
        """
        if self._anthropic is None:
            logger.info("No anthropic client; skipping canonical endpoint creation")
            return 0

        cache = JSONCache(self._cache_dir / "endpoint_classifications.json")
        added = 0
        for trial in trials:
            indication_ids: list[tuple[str, str]] = []
            for cond in trial.conditions:
                ind_id = self.resolve_entity(cond, "indication")
                if ind_id:
                    indication_ids.append((ind_id, cond))

            for om in trial.primary_outcomes:
                if not om.measure:
                    continue
                cls_value = await classify_endpoint_with_llm(
                    self._anthropic, om.measure, cache
                )
                try:
                    cls = EndpointClass(cls_value)
                except ValueError:
                    cls = EndpointClass.OTHER

                for ind_id, ind_name in indication_ids:
                    ep_id = normalize_entity(
                        f"{cls.value}_{ind_id}", "EndpointNode"
                    )
                    try:
                        self.graph.get_node(ep_id)
                    except KeyError:
                        self.graph.add_node(EndpointNode(
                            id=ep_id,
                            name=f"{cls.value} ({ind_name})",
                            endpoint_type=EndpointType.PRIMARY,
                            regulatory_status=RegulatoryStatus.EXPLORATORY,
                            measurement_properties={
                                "endpoint_class": cls.value,
                                "indication_id": ind_id,
                            },
                        ))
                        added += 1
                    # Always index the measure string — even when reusing an
                    # existing (class, indication) node. Otherwise the second
                    # trial whose primary outcome maps to the same EndpointClass
                    # but uses different wording (e.g. "PFS by investigator" vs
                    # "PFS by BICR" vs the canonical name) won't be findable
                    # by ``resolve_entity`` and gets silently skipped in
                    # ``build_trial_subgraphs``.
                    self._index_node(ep_id, om.measure, "endpoint")
                    seed_endpoint_captures_edge(
                        self.graph, ep_id, om.measure, ind_id, cls.value
                    )
        return added

    def _add_compound_target_edges(self, trials: list[TrialRecord]) -> int:
        """For compounds with known targets in the graph, add binds_to edges."""
        added = 0
        targets_in_graph = {
            n["id"] for n in self.graph.get_nodes_by_type("TargetNode")
        }
        for trial in trials:
            drug_interventions = [
                iv for iv in trial.interventions if is_drug_like(iv)
            ]
            for iv in drug_interventions:
                comp_id = self.resolve_entity(iv.name, "compound")
                if not comp_id:
                    continue
                # Try to find a target whose name/symbol appears in the
                # trial title or intervention description
                text = f"{trial.title} {iv.description}".lower()
                for target_id in targets_in_graph:
                    try:
                        tdata = self.graph.get_node(target_id)
                    except KeyError:
                        continue
                    symbol = tdata.get("gene_symbol", "").lower()
                    name = tdata.get("name", "").lower()
                    if symbol and len(symbol) >= 3 and symbol in text:
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
                    elif name and len(name) >= 5 and name in text:
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
        return added

    def _add_binds_edge(self, compound_id: str, target_id: str) -> None:
        edge = GraphEdge(
            source_id=compound_id,
            target_id=target_id,
            edge_type=EdgeType.BINDS_TO,
            belief=EdgeBeliefState(alpha=2.0, beta=1.5),
            metadata={"source": "cross_reference", "method": "name_matching"},
        )
        self.graph.add_edge(edge)

    # ── Trial subgraph construction ──────────────────────────────────────

    def build_trial_subgraphs(
        self, trials: list[TrialRecord]
    ) -> list[TrialSubgraph]:
        """Build skeleton TrialSubgraphs (TrialNode + arms + parent population).

        Produces one chain per arm at the parent (unselected) population.
        Subgroup-specific chains are added later when extraction provides
        canonicalized subgroup features — see ``add_subgroup_chains``.
        """
        subgraphs: list[TrialSubgraph] = []
        for trial in trials:
            indication_id = None
            for cond in trial.conditions:
                indication_id = self.resolve_entity(cond, "indication")
                if indication_id:
                    break
            if not indication_id:
                continue

            endpoint_id = None
            for om in trial.primary_outcomes:
                endpoint_id = self.resolve_entity(om.measure, "endpoint")
                if endpoint_id:
                    break
            if not endpoint_id:
                continue

            seed_trial_node(self.graph, trial)
            arms = build_arms(trial, self.resolve_entity)
            if not arms:
                continue
            synthesize_combo_compounds(self.graph, arms)

            parent_pop_id = ensure_parent_population(
                self.graph, indication_id, indication_name=trial.conditions[0],
            )

            chains = [
                CausalChain(
                    arm_id=arm.arm_id,
                    compound_id=arm.regimen_compound_id,
                    subgroup_population_id=parent_pop_id,
                    target_id=_UNKNOWN,
                    mechanism_id=_UNKNOWN,
                    biology_id=_UNKNOWN,
                    indication_id=indication_id,
                    endpoint_id=endpoint_id,
                    outcome=TrialOutcome.UNKNOWN,
                )
                for arm in arms
            ]

            ts = TrialSubgraph(
                trial_id=trial.nct_id,
                phase=trial.phase or "unknown",
                arms=arms,
                chains=chains,
                parent_population_id=parent_pop_id,
                metadata={
                    "title": trial.title,
                    "status": trial.status,
                    "enrollment": trial.enrollment,
                },
            )
            self.graph.set_trial_subgraph(ts)
            subgraphs.append(ts)
        return subgraphs


# ── Module-level helpers (usable from trace drivers + tests) ─────────────


def seed_trial_node(graph: GraphStore, trial: TrialRecord) -> str:
    """Add (or update) a TrialNode marker in the graph. Returns the node id."""
    node = TrialNode(
        id=trial.nct_id,
        name=trial.title or trial.nct_id,
        phase=trial.phase or "",
        status=trial.status or "",
        sponsor=trial.sponsor or "",
        enrollment=trial.enrollment,
        metadata={
            "conditions": list(trial.conditions),
            "has_results": trial.has_results,
        },
    )
    graph.add_node(node)
    return node.id


def build_arms(
    trial: TrialRecord,
    resolve_compound: Callable[[str, str], str | None] | None = None,
) -> list[TrialArm]:
    """Construct TrialArm objects from the trial's parsed arm_groups.

    ``resolve_compound`` (optional) maps an intervention name to its canonical
    CompoundNode id; if absent, the slug of the intervention name is used as
    the compound id (the trace driver and tests can populate the graph
    accordingly). Combo arms get a synthesized regimen id of the form
    ``compoundA+compoundB`` (constituents sorted).
    """
    arms: list[TrialArm] = []
    seen_arm_ids: set[str] = set()
    for ag in trial.arm_groups:
        if not ag.intervention_names:
            continue
        compound_ids: list[str] = []
        for iv_name in ag.intervention_names:
            cid = (
                resolve_compound(iv_name, "compound")
                if resolve_compound is not None
                else None
            )
            if not cid:
                cid = normalize_entity(iv_name, "CompoundNode")
            compound_ids.append(cid)

        # Drop duplicates while preserving order — some trials list the same
        # intervention twice in an arm group description.
        seen_ids: set[str] = set()
        unique_ids: list[str] = []
        for cid in compound_ids:
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_ids.append(cid)
        if not unique_ids:
            continue

        if len(unique_ids) == 1:
            regimen_id = unique_ids[0]
            is_combo = False
        else:
            regimen_id = "+".join(sorted(unique_ids))
            is_combo = True

        arm_id = ag.group_id or _slug_for_arm(unique_ids)
        # Disambiguate any collision (e.g. two arm groups slugged identically).
        base_arm_id = arm_id
        suffix = 2
        while arm_id in seen_arm_ids:
            arm_id = f"{base_arm_id}_{suffix}"
            suffix += 1
        seen_arm_ids.add(arm_id)

        arms.append(TrialArm(
            arm_id=arm_id,
            compound_ids=unique_ids,
            regimen_compound_id=regimen_id,
            is_combination=is_combo,
        ))
    return arms


def _slug_for_arm(compound_ids: list[str]) -> str:
    return "_".join(sorted(compound_ids))


def synthesize_combo_compounds(graph: GraphStore, arms: list[TrialArm]) -> int:
    """For each combo arm, create a synthesized CompoundNode + composed_of edges.

    Idempotent — if the combo CompoundNode already exists, only missing
    composed_of edges are added. Returns the count of new combo nodes
    created (not counting edges).
    """
    created = 0
    for arm in arms:
        if not arm.is_combination:
            continue
        try:
            graph.get_node(arm.regimen_compound_id)
        except KeyError:
            # Build a human-readable name from the constituents in the
            # graph (fall back to the id segments when names are missing).
            constituent_names: list[str] = []
            for cid in arm.compound_ids:
                try:
                    cdata = graph.get_node(cid)
                    constituent_names.append(cdata.get("name") or cid)
                except KeyError:
                    constituent_names.append(cid)
            graph.add_node(CompoundNode(
                id=arm.regimen_compound_id,
                name=" + ".join(constituent_names),
                modality=Modality.OTHER,
                metadata={"synthesized": "combo", "constituents": list(arm.compound_ids)},
            ))
            created += 1
        # composed_of edges (one per constituent). add_edge is idempotent
        # on (source, target, edge_type) thanks to MultiDiGraph keying.
        for cid in arm.compound_ids:
            graph.add_edge(GraphEdge(
                source_id=arm.regimen_compound_id,
                target_id=cid,
                edge_type=EdgeType.COMPOSED_OF,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
                metadata={"source": "combo_synthesis"},
            ))
        # Inherit binds_to from each constituent. The combo doesn't have
        # its own ChEMBL record; its target engagement is the union of
        # its constituents'. Use a slightly weaker prior (Beta(3, 1.5))
        # than the OT-direct edges (Beta(4, 1)) since the inherited claim
        # rests on the constituent's evidence rather than a direct lookup.
        propagated: set[str] = set()
        for cid in arm.compound_ids:
            for edge in graph.get_neighboring_edges(
                cid, edge_types=[EdgeType.BINDS_TO],
            ):
                target_id = edge["target_id"]
                if target_id in propagated:
                    continue
                propagated.add(target_id)
                if graph._graph.has_edge(  # noqa: SLF001
                    arm.regimen_compound_id,
                    target_id,
                    key=EdgeType.BINDS_TO.value,
                ):
                    continue
                graph.add_edge(GraphEdge(
                    source_id=arm.regimen_compound_id,
                    target_id=target_id,
                    edge_type=EdgeType.BINDS_TO,
                    belief=EdgeBeliefState(alpha=3.0, beta=1.5),
                    metadata={
                        "source": "combo_inherit",
                        "via_constituent": cid,
                    },
                ))
    return created


def ensure_parent_population(
    graph: GraphStore,
    indication_id: str,
    *,
    indication_name: str | None = None,
) -> str:
    """Create (if missing) the trial's parent enrollment PopulationNode.

    The parent represents "all patients meeting trial enrollment criteria"
    — used as the default subgroup_population_id when no biomarker
    stratifier was reported.
    """
    pop_id = PopulationNode.compose_id(indication_id, [])
    try:
        graph.get_node(pop_id)
    except KeyError:
        graph.add_node(PopulationNode(
            id=pop_id,
            name=f"All patients ({indication_name or indication_id})",
            defining_features=[],
        ))
    return pop_id


def build_trial_subgraph_from_extraction(
    graph: GraphStore,
    trial: TrialRecord,
    extraction: Any,
    *,
    target_by_arm: dict[str, str],
    mechanism_id: str,
    biology_id: str,
    indication_id: str,
    endpoint_ids: dict[str, str],
) -> TrialSubgraph:
    """Compose the full chain-list TrialSubgraph from an extraction.

    Uses CT.gov-parsed ``trial.arm_groups`` for arm structure (deterministic;
    extraction.arms is a sanity check, not a primary source). Subgroups
    come from ``extraction.subgroups`` (LLM-canonicalized via the
    extraction call's vocabulary). Per-chain outcomes come from
    ``extraction.results_by_chain``.

    Cardinality: chains = N arms × M reported subgroups × K endpoints.
    Each chain references one endpoint, since PFS and OS measure the
    same biology through different lenses and a chain models one
    measurement.

    Args:
        target_by_arm: maps each arm_id to the canonical TargetNode id
            for that arm's primary target. Caller resolves these (e.g.
            via ``OpenTargetsClient.search_target``) so the function
            stays free of network/API concerns.
        endpoint_ids: maps EndpointClass value (e.g. "PFS", "OS") to the
            corresponding EndpointNode id already in the graph. The keys
            of this dict drive the endpoint fan-out — one chain per
            (arm × subgroup × endpoint) cell.

    The result is persisted to ``graph.trial_subgraphs[trial.nct_id]``.
    """
    # Lazy imports to avoid circular dependencies and keep populate's
    # core path independent of taxonomy at import time.
    from src.annotation.taxonomy import TrialExtraction  # noqa: F401
    from src.graph.subgroup_taxonomy import (
        canonicalize_feature,
        is_canonical,
        log_unmapped,
    )

    if not endpoint_ids:
        raise ValueError(
            f"{trial.nct_id} has no endpoint_ids; need at least one "
            "(EndpointClass → EndpointNode id) entry"
        )

    seed_trial_node(graph, trial)
    arms = build_arms(trial)
    if not arms:
        raise ValueError(
            f"{trial.nct_id} has no arm groups; cannot build trial subgraph"
        )
    synthesize_combo_compounds(graph, arms)

    indication_name = trial.conditions[0] if trial.conditions else None
    parent_pop_id = ensure_parent_population(
        graph, indication_id, indication_name=indication_name,
    )

    # Canonicalize subgroup features: descriptor → list[SubgroupFeature].
    # Unknown axes get logged for vocabulary-extension review. Subgroups
    # whose features all collapse to ``axis="other"`` are dropped — these
    # are typically PD readouts ("CD8 T cells per mm² day 22") or analysis
    # timepoints ("Primary completion") that aren't real patient subgroups.
    # Letting them through produces one-off PopulationNodes that no other
    # trial can match against.
    subgroup_features_by_descriptor: dict[str, list[SubgroupFeature]] = {}
    subgroup_pop_ids: dict[str, str] = {}
    for sg in getattr(extraction, "subgroups", []) or []:
        feats: list[SubgroupFeature] = []
        for f in sg.features:
            cf = canonicalize_feature(
                f.get("axis", ""),
                f.get("key", ""),
                f.get("level", ""),
                raw_descriptor=sg.raw_descriptor,
            )
            if not is_canonical(cf):
                log_unmapped(cf, trial.nct_id)
            feats.append(cf)
        if not any(is_canonical(f) for f in feats):
            continue
        subgroup_features_by_descriptor[sg.raw_descriptor] = feats
        pop_id = PopulationNode.compose_id(indication_id, feats)
        try:
            graph.get_node(pop_id)
        except KeyError:
            graph.add_node(PopulationNode(
                id=pop_id,
                name=f"{sg.raw_descriptor} ({indication_name or indication_id})",
                defining_features=list(feats),
            ))
        subgroup_pop_ids[sg.raw_descriptor] = pop_id

    # Index per-chain results by (arm_id, pop_id, endpoint_class).
    # Multiple raw descriptors may collapse onto the same canonical pop_id
    # (e.g. PD-L1 ≥1% and PD-L1 ≥5% both → cd274_high) — keying on pop_id
    # rather than descriptor lets results across those descriptors land
    # on the same chain. Last-write-wins on collision.
    results_index: dict[tuple[str, str | None, str], Any] = {}
    for cr in getattr(extraction, "results_by_chain", []) or []:
        ep_class = classify_endpoint_deterministic(cr.endpoint)
        if cr.subgroup_descriptor is None:
            pop_key: str | None = None
        else:
            pop_key = subgroup_pop_ids.get(cr.subgroup_descriptor)
            if pop_key is None:
                # Descriptor wasn't extracted as a subgroup — skip.
                continue
        results_index[(cr.arm_id, pop_key, ep_class)] = cr

    # Determine the (subgroup_descriptor, pop_id) cells to fan over.
    # Multiple raw descriptors can collapse onto the same canonical
    # pop_id (e.g. "PD-L1 ≥1%" and "PD-L1 ≥5%" both → cd274_high). Dedupe
    # by pop_id so we don't multiply the chain count by descriptor count.
    target_subgroups: list[tuple[str | None, str]]
    if subgroup_pop_ids:
        seen_pop_ids: set[str] = set()
        target_subgroups = []
        for descriptor, pop_id in subgroup_pop_ids.items():
            if pop_id in seen_pop_ids:
                continue
            seen_pop_ids.add(pop_id)
            target_subgroups.append((descriptor, pop_id))
    else:
        target_subgroups = [(None, parent_pop_id)]

    # Fan chains across arms × subgroups × endpoints. Each chain models
    # one measurement window into the same upstream biology.
    chains: list[CausalChain] = []
    for arm in arms:
        arm_target_id = target_by_arm.get(arm.arm_id, _UNKNOWN)
        for descriptor, pop_id in target_subgroups:
            for ep_class, ep_id in endpoint_ids.items():
                # Result lookup uses pop_id (canonical), not descriptor —
                # see results_index construction above.
                pop_key_for_lookup: str | None = (
                    None if descriptor is None else pop_id
                )
                cr = results_index.get((arm.arm_id, pop_key_for_lookup, ep_class))
                outcome = TrialOutcome.UNKNOWN
                effect_size = None
                p_value = None
                if cr is not None:
                    try:
                        outcome = TrialOutcome(cr.outcome)
                    except ValueError:
                        outcome = TrialOutcome.UNKNOWN
                    effect_size = cr.effect_size
                    p_value = cr.p_value
                chains.append(CausalChain(
                    arm_id=arm.arm_id,
                    compound_id=arm.regimen_compound_id,
                    subgroup_population_id=pop_id,
                    target_id=arm_target_id,
                    mechanism_id=mechanism_id,
                    biology_id=biology_id,
                    indication_id=indication_id,
                    endpoint_id=ep_id,
                    outcome=outcome,
                    effect_size=effect_size,
                    p_value=p_value,
                    metadata={"endpoint_class": ep_class},
                ))

    ts = TrialSubgraph(
        trial_id=trial.nct_id,
        phase=trial.phase or "",
        arms=arms,
        chains=chains,
        parent_population_id=parent_pop_id,
        metadata={
            "title": trial.title,
            "status": trial.status,
            "enrollment": trial.enrollment,
        },
    )
    graph.set_trial_subgraph(ts)
    return ts


def seed_responds_differently_from_extractions(
    graph: GraphStore,
    annotations_dir: Path,
) -> tuple[int, int]:
    """Walk per-trial extractions, seed subgroup populations, and fork chains.

    For each extraction's ``subgroups``:
      1. Canonicalize features into a ``SubgroupFeature`` list.
      2. Compose / create a PopulationNode keyed off the trial's
         indication.
      3. Add a population→indication ``responds_differently`` edge with
         prior Beta(1.5, 1).
      4. Fork the trial's chains: every parent-population chain is
         duplicated with ``subgroup_population_id`` set to the new
         subgroup population id, inheriting the parent chain's resolved
         target/mechanism/biology. The prediction engine can then query
         either the unselected or subgroup chain.

    Returns ``(edges_added, chains_added)``.

    Lookup expectations:
      - ``graph.trial_subgraphs[nct_id]`` exists (created by
        ``build_trial_subgraphs`` upstream) and has at least one chain
        whose ``indication_id`` is the trial's canonical indication.
      - Subgroup features have axis/key/level matching the extraction
        prompt's contract (``ExtractedSubgroup``).
    """
    from src.graph.subgroup_taxonomy import (
        canonicalize_feature,
        is_canonical,
        log_unmapped,
    )

    edges_added = 0
    chains_added = 0
    for path in sorted(annotations_dir.glob("*_extraction.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable extraction at %s", path)
            continue
        nct_id = data.get("nct_id")
        if not nct_id:
            continue
        try:
            ts = graph.get_trial_subgraph_by_id(nct_id)
        except KeyError:
            continue
        if not ts.chains:
            continue
        indication_id = ts.chains[0].indication_id
        if indication_id == _UNKNOWN:
            continue

        # Snapshot parent chains BEFORE forking so we don't fork copies.
        parent_chains = list(ts.chains)
        new_subgroup_pop_ids: list[str] = []

        for sg in data.get("subgroups") or []:
            descriptor = sg.get("raw_descriptor", "")
            features: list[SubgroupFeature] = []
            for f in sg.get("features") or []:
                cf = canonicalize_feature(
                    f.get("axis", ""),
                    f.get("key", ""),
                    f.get("level", ""),
                    raw_descriptor=descriptor,
                )
                if not is_canonical(cf):
                    log_unmapped(cf, nct_id)
                features.append(cf)
            if not features:
                continue
            # Drop subgroups whose features all canonicalize to "other" —
            # see the same guard in ``build_trial_subgraphs`` for context.
            if not any(is_canonical(f) for f in features):
                continue
            pop_id = PopulationNode.compose_id(indication_id, features)
            try:
                graph.get_node(pop_id)
            except KeyError:
                graph.add_node(PopulationNode(
                    id=pop_id,
                    name=f"{descriptor or pop_id}",
                    defining_features=list(features),
                ))
            if not graph._graph.has_edge(  # noqa: SLF001
                pop_id, indication_id, key=EdgeType.RESPONDS_DIFFERENTLY.value,
            ):
                graph.add_edge(GraphEdge(
                    source_id=pop_id,
                    target_id=indication_id,
                    edge_type=EdgeType.RESPONDS_DIFFERENTLY,
                    belief=EdgeBeliefState(alpha=1.5, beta=1.0),
                    metadata={
                        "source": "extraction_subgroup",
                        "trial_id": nct_id,
                        "raw_descriptor": descriptor,
                    },
                ))
                edges_added += 1
            if pop_id != ts.parent_population_id:
                new_subgroup_pop_ids.append(pop_id)

        if new_subgroup_pop_ids:
            # Fork: parent chains stay; for each subgroup pop, append a
            # copy of every parent chain with subgroup_population_id
            # rebound. Skip subgroup pops we've already forked (idempotent
            # if seeder runs twice).
            existing_pops = {c.subgroup_population_id for c in parent_chains}
            forked: list[CausalChain] = list(parent_chains)
            for pop_id in dict.fromkeys(new_subgroup_pop_ids):  # dedupe, preserve order
                if pop_id in existing_pops:
                    continue
                for parent in parent_chains:
                    forked.append(parent.model_copy(
                        update={"subgroup_population_id": pop_id}
                    ))
                    chains_added += 1
            graph.set_trial_subgraph(ts.model_copy(update={"chains": forked}))
    return edges_added, chains_added


def add_subgroup_chains(
    graph: GraphStore,
    trial_subgraph: TrialSubgraph,
    *,
    indication_id: str,
    endpoint_id: str,
    subgroup_features: list[list[SubgroupFeature]],
    indication_name: str | None = None,
) -> int:
    """Add (arm × subgroup) chains to an existing TrialSubgraph.

    ``subgroup_features`` is a list of feature compositions — one composition
    per reported subgroup. Each composition produces one PopulationNode
    (created if missing) and one chain *per arm*.

    The TrialSubgraph in the sidecar is replaced with the augmented version.
    Returns the number of new chains added.
    """
    if not subgroup_features:
        return 0

    new_chains: list[CausalChain] = []
    for features in subgroup_features:
        pop_id = PopulationNode.compose_id(indication_id, features)
        try:
            graph.get_node(pop_id)
        except KeyError:
            descriptor = ", ".join(f.raw_descriptor or f.slug() for f in features)
            graph.add_node(PopulationNode(
                id=pop_id,
                name=f"{descriptor} ({indication_name or indication_id})",
                defining_features=list(features),
            ))
        for arm in trial_subgraph.arms:
            new_chains.append(CausalChain(
                arm_id=arm.arm_id,
                compound_id=arm.regimen_compound_id,
                subgroup_population_id=pop_id,
                target_id=_UNKNOWN,
                mechanism_id=_UNKNOWN,
                biology_id=_UNKNOWN,
                indication_id=indication_id,
                endpoint_id=endpoint_id,
                outcome=TrialOutcome.UNKNOWN,
            ))

    augmented = trial_subgraph.model_copy(
        update={"chains": list(trial_subgraph.chains) + new_chains}
    )
    graph.set_trial_subgraph(augmented)
    return len(new_chains)


def _normalize(name: str) -> str:
    """Lowercase, strip whitespace, collapse runs of spaces."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ── CLI entry point ──────────────────────────────────────────────────────


async def _main(area: str, max_trials: int, condition: str) -> None:
    from pathlib import Path

    graph = GraphStore()
    client = anthropic.AsyncAnthropic(timeout=60.0)
    pipeline = PopulationPipeline(graph, anthropic_client=client)

    if area == "oncology":
        await pipeline.populate_oncology(
            max_trials=max_trials, condition=condition,
        )
    else:
        console.print(f"[red]Unknown area: {area}[/red]")
        return

    export_path = Path("data/exports")
    export_path.mkdir(parents=True, exist_ok=True)
    snapshot = str(export_path / f"{area}_initial.json")
    graph.export_snapshot(snapshot)
    console.print(f"[bold]Snapshot saved to {snapshot}[/bold]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate Eroom Bio knowledge graph")
    parser.add_argument("--area", default="oncology", help="Disease area (default: oncology)")
    parser.add_argument("--max-trials", type=int, default=500, help="Max trials to fetch")
    parser.add_argument(
        "--condition",
        default="cancer",
        help="ClinicalTrials.gov condition filter (default: cancer)",
    )
    args = parser.parse_args()

    asyncio.run(_main(args.area, args.max_trials, args.condition))
