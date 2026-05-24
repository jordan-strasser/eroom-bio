"""
Gold pair set curation script for A.5 biology/mechanism pairs.
Sources: S1 (cached extractions), S2 (Reactome), S3 (GO via QuickGO, optional).
Produces data/eval/biology_gold_pairs.jsonl and data/eval/_staging/REVIEW_NOTES.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import anthropic

# ── paths ────────────────────────────────────────────────────────────────────
REPO = Path("/Users/jordanstrasser/Code/eroom-bio/eroom")
ANNOTATIONS_DIR = REPO / "data" / "annotations"
REACTOME_ANCESTORS = REPO / "data" / "cache" / "reactome_ancestors.json"
STAGING = REPO / "data" / "eval" / "_staging"
FINAL_OUT = REPO / "data" / "eval" / "biology_gold_pairs.jsonl"

STAGING.mkdir(parents=True, exist_ok=True)
(REPO / "data" / "eval").mkdir(parents=True, exist_ok=True)

# ── cost tracking ─────────────────────────────────────────────────────────────
HAIKU_INPUT_COST_PER_M = 1.0    # $/M tokens
HAIKU_OUTPUT_COST_PER_M = 5.0   # $/M tokens
BUDGET_HARD_STOP = 1.80         # dollars

total_input_tokens = 0
total_output_tokens = 0


def current_cost() -> float:
    return (total_input_tokens / 1_000_000) * HAIKU_INPUT_COST_PER_M + \
           (total_output_tokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_M


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── random seed ───────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — String mining
# ═══════════════════════════════════════════════════════════════════════════════

def mine_s1() -> list[dict]:
    """Mine cached trial extractions for mechanism/biology strings."""
    strings = []
    files = sorted(f for f in ANNOTATIONS_DIR.iterdir() if f.name.endswith("_extraction.json"))
    for fp in files:
        data = json.loads(fp.read_text())
        nct = data.get("nct_id", fp.stem.replace("_extraction", ""))
        th = data.get("therapeutic_hypothesis", {})

        proposed = (th.get("proposed_mechanism") or "").strip()
        if proposed:
            strings.append({
                "source": "S1_extraction",
                "source_id": nct,
                "domain": "mechanism",
                "text": proposed,
                "ontology_parent_ids": [],
                "ontology_child_ids": [],
                "field": "proposed_mechanism",
            })

        intended = (th.get("intended_biology") or "").strip()
        if intended:
            strings.append({
                "source": "S1_extraction",
                "source_id": nct,
                "domain": "biology",
                "text": intended,
                "ontology_parent_ids": [],
                "ontology_child_ids": [],
                "field": "intended_biology",
            })

        for chain in data.get("results_by_chain", []):
            for field, domain in [
                ("mechanism_description", "mechanism"),
                ("biology_description", "biology"),
            ]:
                val = (chain.get(field) or "").strip()
                if val:
                    strings.append({
                        "source": "S1_extraction",
                        "source_id": f"{nct}:{chain.get('arm_id', '')}",
                        "domain": domain,
                        "text": val,
                        "ontology_parent_ids": [],
                        "ontology_child_ids": [],
                        "field": field,
                    })

        for mod in data.get("modulation_entries", []):
            hyp = (mod.get("hypothesis") or "").strip()
            if hyp:
                strings.append({
                    "source": "S1_extraction",
                    "source_id": nct,
                    "domain": "mechanism",
                    "text": hyp,
                    "ontology_parent_ids": [],
                    "ontology_child_ids": [],
                    "field": "modulation_hypothesis",
                })

    log.info("S1: %d strings", len(strings))
    return strings


def _fetch_json(url: str, timeout: int = 10) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def mine_s2() -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Mine Reactome for pathway names + hierarchy.
    Returns (strings, ontology_pairs) where ontology_pairs is
    list of (id_a, id_b, relation) with relation in parent_of/sibling.
    """
    ancestors_map = json.loads(REACTOME_ANCESTORS.read_text())  # child -> [ancestor...]

    # Find all IDs we need: in-graph nodes + their ancestors
    all_ids_needed: set[str] = set(ancestors_map.keys())
    for v in ancestors_map.values():
        all_ids_needed.update(v)
    all_ids_needed_sorted = sorted(all_ids_needed)

    # Fetch pathway names + summations
    name_cache: dict[str, str] = {}
    summation_cache: dict[str, str] = {}

    log.info("S2: fetching names for %d Reactome IDs", len(all_ids_needed_sorted))
    failed = 0
    for rhsa_id in all_ids_needed_sorted:
        try:
            data = _fetch_json(f"https://reactome.org/ContentService/data/query/{rhsa_id}", timeout=8)
            name = ""
            if data.get("displayName"):
                name = data["displayName"]
            elif data.get("name") and isinstance(data["name"], list):
                name = data["name"][0]
            if name:
                name_cache[rhsa_id] = name
            summs = data.get("summation", [])
            if summs and isinstance(summs, list):
                text = summs[0].get("text", "")
                if text:
                    summation_cache[rhsa_id] = text
            time.sleep(0.05)
        except Exception as e:
            failed += 1
            if failed <= 3:
                log.warning("S2: failed to fetch %s: %s", rhsa_id, e)

    log.info("S2: fetched %d names, %d summations, %d failed",
             len(name_cache), len(summation_cache), failed)

    # Build strings for in-graph nodes (ancestors_map keys)
    strings = []
    id_to_idx: dict[str, int] = {}
    for rhsa_id in sorted(ancestors_map.keys()):
        text = summation_cache.get(rhsa_id) or name_cache.get(rhsa_id)
        if not text:
            continue
        idx = len(strings)
        id_to_idx[rhsa_id] = idx
        ancestor_ids = ancestors_map[rhsa_id]
        strings.append({
            "source": "S2_reactome",
            "source_id": rhsa_id,
            "domain": "biology",
            "text": text,
            "ontology_parent_ids": ancestor_ids[:1] if ancestor_ids else [],
            "ontology_child_ids": [],
            "name": name_cache.get(rhsa_id, ""),
        })

    # Build ontology pairs
    # 1. True parent/child: in-graph nodes that are ancestors of other in-graph nodes
    in_graph_keys = set(ancestors_map.keys())
    ontology_pairs: list[tuple[str, str, str]] = []  # (id_a, id_b, relation)

    for child_id, ancestor_ids in ancestors_map.items():
        for anc_id in ancestor_ids:
            if anc_id in in_graph_keys and anc_id in id_to_idx and child_id in id_to_idx:
                # anc_id is parent, child_id is child
                ontology_pairs.append((anc_id, child_id, "parent_of"))

    # 2. Sibling pairs: nodes sharing at least one common ancestor (from ancestors_map)
    all_keys = sorted(ancestors_map.keys())
    sibling_groups: dict[str, list[str]] = defaultdict(list)
    for key_id in all_keys:
        for anc_id in ancestors_map[key_id]:
            sibling_groups[anc_id].append(key_id)

    sibling_pairs_added: set[frozenset] = set()
    for anc_id, siblings in sibling_groups.items():
        if len(siblings) < 2:
            continue
        # Sample pairs from this sibling group (limit to avoid explosion)
        siblings_with_text = [s for s in siblings if s in id_to_idx]
        if len(siblings_with_text) < 2:
            continue
        sampled = random.sample(siblings_with_text, min(4, len(siblings_with_text)))
        for ci in range(len(sampled)):
            for cj in range(ci + 1, len(sampled)):
                key = frozenset([sampled[ci], sampled[cj]])
                if key not in sibling_pairs_added:
                    sibling_pairs_added.add(key)
                    ontology_pairs.append((sampled[ci], sampled[cj], "sibling"))

    log.info("S2: %d strings with text, %d ontology pairs", len(strings), len(ontology_pairs))
    return strings, ontology_pairs


def mine_s3_optional() -> list[dict]:
    """Optionally mine GO via QuickGO. One quick try, skip on any error."""
    strings = []
    queries = [
        "PI3K signaling", "apoptosis", "cell proliferation", "immune response",
        "DNA repair", "angiogenesis", "inflammation", "cell cycle",
        "MAPK signaling", "mTOR signaling",
    ]
    try:
        for q in queries[:6]:
            url = (f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/search"
                   f"?query={urllib.parse.quote(q)}&limit=8&ontology=biological_process")
            data = _fetch_json(url, timeout=8)
            for result in data.get("results", []):
                go_id = result.get("id", "")
                name = result.get("name", "")
                defn_data = result.get("definition", {})
                defn = defn_data.get("text", "") if isinstance(defn_data, dict) else ""
                text = defn or name
                if not text or result.get("isObsolete"):
                    continue
                strings.append({
                    "source": "S3_go",
                    "source_id": go_id,
                    "domain": "biology",
                    "text": text,
                    "ontology_parent_ids": [],
                    "ontology_child_ids": [],
                    "name": name,
                })
            time.sleep(0.15)
    except Exception as e:
        log.info("S3 skipped: %s", e)
        return []
    log.info("S3: %d strings", len(strings))
    return strings


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Candidate pair generation
# ═══════════════════════════════════════════════════════════════════════════════

def embed_all(strings: list[dict]) -> list[list[float]]:
    """Embed all strings with BioLORD."""
    sys.path.insert(0, str(REPO / "src"))
    from graph.biolord_embeddings import embed_texts
    texts = [s["text"] for s in strings]
    log.info("Embedding %d strings with BioLORD...", len(texts))
    embeddings = embed_texts(
        texts,
        use_cache=True,
        cache_path=REPO / "data" / "cache" / "biolord_embeddings.json",
    )
    log.info("Done embedding.")
    return embeddings


def make_pair_id() -> str:
    return str(uuid.uuid4())


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Deduplicate by frozenset of (text_a, text_b)."""
    seen: set[frozenset] = set()
    out = []
    for p in pairs:
        key = frozenset([p["text_a"], p["text_b"]])
        if key in seen or p["text_a"] == p["text_b"]:
            continue
        seen.add(key)
        out.append(p)
    return out


def generate_candidates(
    strings: list[dict],
    embeddings: list[list[float]],
    s2_ontology_pairs: list[tuple[str, str, str]],
) -> list[dict]:
    """Generate candidate pairs using four strategies."""
    import numpy as np

    n = len(strings)
    candidates: list[dict] = []

    # Build lookup: source_id -> index (for S2 ontology pairs)
    id_to_idx: dict[str, int] = {}
    for idx, s in enumerate(strings):
        if s["source"] == "S2_reactome":
            id_to_idx[s["source_id"]] = idx

    # Normalize embeddings for cosine
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb_matrix / norms
    sim_matrix = emb_norm @ emb_norm.T  # (n, n)
    np.fill_diagonal(sim_matrix, -1.0)

    # Strategy 1: Top-3 cosine neighbors
    log.info("Strategy 1: top-3 cosine neighbors...")
    topk_pairs = []
    for i in range(n):
        top_indices = np.argsort(sim_matrix[i])[::-1][:3]
        for j in top_indices:
            sim = float(sim_matrix[i, j])
            if sim < 0.05:
                continue
            topk_pairs.append({
                "pair_id": make_pair_id(),
                "text_a": strings[i]["text"],
                "text_b": strings[j]["text"],
                "source_a": strings[i]["source"],
                "source_b": strings[j]["source"],
                "source_id_a": strings[i]["source_id"],
                "source_id_b": strings[j]["source_id"],
                "cosine_biolord": round(sim, 4),
                "candidate_strategy": "topk_neighbor",
                "ontology_relation_hint": None,
            })
    topk_pairs = deduplicate_pairs(topk_pairs)
    log.info("  topk_neighbor: %d pairs", len(topk_pairs))
    candidates.extend(topk_pairs)

    # Strategy 2: Reactome ontology parent/child pairs
    log.info("Strategy 2: Reactome parent_of pairs...")
    parent_pairs = []
    seen_parent: set[frozenset] = set()
    for (id_a, id_b, relation) in s2_ontology_pairs:
        if relation != "parent_of":
            continue
        if id_a not in id_to_idx or id_b not in id_to_idx:
            continue
        ia, ib = id_to_idx[id_a], id_to_idx[id_b]
        key = frozenset([strings[ia]["text"], strings[ib]["text"]])
        if key in seen_parent or strings[ia]["text"] == strings[ib]["text"]:
            continue
        seen_parent.add(key)
        sim = float(sim_matrix[ia, ib])
        parent_pairs.append({
            "pair_id": make_pair_id(),
            "text_a": strings[ia]["text"],
            "text_b": strings[ib]["text"],
            "source_a": "S2_reactome",
            "source_b": "S2_reactome",
            "source_id_a": id_a,
            "source_id_b": id_b,
            "cosine_biolord": round(sim, 4),
            "candidate_strategy": "ontology_parent",
            "ontology_relation_hint": "parent_of",
        })
    log.info("  ontology_parent: %d pairs", len(parent_pairs))
    candidates.extend(parent_pairs)

    # Strategy 3: Reactome sibling pairs
    log.info("Strategy 3: Reactome sibling pairs...")
    sibling_pairs = []
    seen_sibling: set[frozenset] = set()
    for (id_a, id_b, relation) in s2_ontology_pairs:
        if relation != "sibling":
            continue
        if id_a not in id_to_idx or id_b not in id_to_idx:
            continue
        ia, ib = id_to_idx[id_a], id_to_idx[id_b]
        key = frozenset([strings[ia]["text"], strings[ib]["text"]])
        if key in seen_sibling or strings[ia]["text"] == strings[ib]["text"]:
            continue
        seen_sibling.add(key)
        sim = float(sim_matrix[ia, ib])
        sibling_pairs.append({
            "pair_id": make_pair_id(),
            "text_a": strings[ia]["text"],
            "text_b": strings[ib]["text"],
            "source_a": "S2_reactome",
            "source_b": "S2_reactome",
            "source_id_a": id_a,
            "source_id_b": id_b,
            "cosine_biolord": round(sim, 4),
            "candidate_strategy": "ontology_sibling",
            "ontology_relation_hint": "sibling",
        })
    log.info("  ontology_sibling: %d pairs", len(sibling_pairs))
    candidates.extend(sibling_pairs)

    # Strategy 4: Distance-stratified random pairs (~30 total)
    log.info("Strategy 4: distance-stratified random pairs...")
    bands = [(0.0, 0.30), (0.30, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.0)]
    band_target = 6

    # Sample from upper triangle
    all_upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sample_pool = random.sample(all_upper, min(8000, len(all_upper)))

    band_pool: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for i, j in sample_pool:
        sim = float(sim_matrix[i, j])
        for band_idx, (lo, hi) in enumerate(bands):
            if lo <= sim < hi:
                band_pool[band_idx].append((i, j, sim))
                break

    random_pairs = []
    seen_random: set[frozenset] = set()
    for band_idx, (lo, hi) in enumerate(bands):
        pool = band_pool[band_idx]
        random.shuffle(pool)
        added = 0
        for i, j, sim in pool:
            key = frozenset([strings[i]["text"], strings[j]["text"]])
            if key in seen_random or strings[i]["text"] == strings[j]["text"]:
                continue
            seen_random.add(key)
            random_pairs.append({
                "pair_id": make_pair_id(),
                "text_a": strings[i]["text"],
                "text_b": strings[j]["text"],
                "source_a": strings[i]["source"],
                "source_b": strings[j]["source"],
                "source_id_a": strings[i]["source_id"],
                "source_id_b": strings[j]["source_id"],
                "cosine_biolord": round(sim, 4),
                "candidate_strategy": "random",
                "ontology_relation_hint": None,
            })
            added += 1
            if added >= band_target:
                break

    log.info("  random: %d pairs", len(random_pairs))
    candidates.extend(random_pairs)

    # Final dedup
    candidates = deduplicate_pairs(candidates)
    log.info("Total candidates after final dedup: %d", len(candidates))
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — LLM Labeling
# ═══════════════════════════════════════════════════════════════════════════════

LABELING_SYSTEM = """You are a biomedical ontology expert labeling pairs of biology/mechanism descriptions.
Label each pair with exactly one of: merge, parent_of, child_of, sibling, unrelated.

Rubric:
- merge: Same biological concept at the same granularity, differing only in phrasing.
- parent_of: text_a strictly subsumes text_b (text_b is a specific case of text_a).
- child_of: text_a is narrower than text_b (text_a is a specific case of text_b).
- sibling: Same level of granularity under a shared parent, but distinct from each other.
- unrelated: No clear hierarchical or semantic relationship.

Respond with JSON only, no markdown fences: {"label": "...", "justification": "one sentence", "confidence": "high|medium|low"}"""

LABELING_TEMPLATE = """Text A: {text_a}

Text B: {text_b}

Label this pair:"""


async def label_pair_llm(client: anthropic.AsyncAnthropic, pair: dict) -> dict:
    """Label a single pair with Claude Haiku."""
    global total_input_tokens, total_output_tokens

    prompt = LABELING_TEMPLATE.format(text_a=pair["text_a"], text_b=pair["text_b"])
    msg = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        system=LABELING_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    total_input_tokens += msg.usage.input_tokens
    total_output_tokens += msg.usage.output_tokens

    raw = msg.content[0].text.strip()
    try:
        # Strip markdown code fences if present
        clean = raw
        if clean.startswith("```"):
            parts = clean.split("```")
            if len(parts) >= 3:
                clean = parts[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            else:
                clean = clean.replace("```json", "").replace("```", "")
        result = json.loads(clean.strip())
        label = result.get("label", "unrelated")
        if label not in ("merge", "parent_of", "child_of", "sibling", "unrelated"):
            label = "unrelated"
        justification = result.get("justification", "")[:500]
        confidence = result.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
    except Exception:
        label = "unrelated"
        justification = f"Parse error on: {raw[:100]}"
        confidence = "low"

    return {
        **pair,
        "label": label,
        "justification": justification,
        "confidence": confidence,
        "labeler": "claude-haiku-4-5",
    }


async def label_batch_async(
    client: anthropic.AsyncAnthropic,
    pairs: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async def label_one(pair: dict) -> dict:
        async with semaphore:
            return await label_pair_llm(client, pair)

    return await asyncio.gather(*[label_one(p) for p in pairs])


def label_ontology_pair(pair: dict) -> dict:
    """Label Reactome-hinted pairs directly (high confidence, no LLM needed)."""
    hint = pair["ontology_relation_hint"]
    if hint == "parent_of":
        label = "parent_of"
        just = "Reactome curator-annotated parent-child; text_a is the parent pathway."
    elif hint == "child_of":
        label = "child_of"
        just = "Reactome curator-annotated child-parent; text_a is the child pathway."
    elif hint == "sibling":
        label = "sibling"
        just = "Reactome curator-annotated siblings; both are children of the same parent pathway."
    else:
        label = "unrelated"
        just = "No ontology hint."

    return {
        **pair,
        "label": label,
        "justification": just,
        "confidence": "high",
        "labeler": "reactome_hierarchy",
        "agrees_with_hint": True,
    }


async def run_llm_labeling(candidates: list[dict], max_llm_pairs: int = 130) -> list[dict]:
    """Label all candidates: ontology-hinted directly, rest via LLM."""
    labeled: list[dict] = []
    llm_queue: list[dict] = []

    for pair in candidates:
        hint = pair.get("ontology_relation_hint")
        strategy = pair.get("candidate_strategy", "")
        if hint and strategy in ("ontology_parent", "ontology_sibling"):
            labeled.append(label_ontology_pair(pair))
        else:
            llm_queue.append(pair)

    log.info("Ontology-labeled: %d | LLM queue: %d", len(labeled), len(llm_queue))

    # Cap LLM pairs — prioritize topk_neighbor (boundary-rich) then random
    if len(llm_queue) > max_llm_pairs:
        topk = [p for p in llm_queue if p["candidate_strategy"] == "topk_neighbor"]
        rand = [p for p in llm_queue if p["candidate_strategy"] == "random"]
        other = [p for p in llm_queue if p["candidate_strategy"] not in ("topk_neighbor", "random")]
        topk.sort(key=lambda p: -p["cosine_biolord"])
        llm_queue = (topk + other + rand)[:max_llm_pairs]

    log.info("LLM labeling %d pairs...", len(llm_queue))
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(8)

    llm_labeled: list[dict] = []
    batch_size = 50
    for batch_start in range(0, len(llm_queue), batch_size):
        cost = current_cost()
        if cost >= BUDGET_HARD_STOP:
            log.warning("Budget limit $%.4f reached. Stopping labeling.", cost)
            break
        batch = llm_queue[batch_start: batch_start + batch_size]
        log.info("  Labeling batch %d-%d (cost so far: $%.4f)",
                 batch_start, batch_start + len(batch) - 1, cost)
        results = await label_batch_async(client, batch, semaphore)
        for r in results:
            hint = r.get("ontology_relation_hint")
            r["agrees_with_hint"] = (r["label"] == hint) if hint else None
        llm_labeled.extend(results)
        log.info("  -> %d labeled, cost: $%.4f", len(llm_labeled), current_cost())

    labeled.extend(llm_labeled)
    log.info("Total labeled: %d | Cost: $%.4f", len(labeled), current_cost())
    return labeled


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Self-consistency (10% re-label)
# ═══════════════════════════════════════════════════════════════════════════════

async def self_consistency_check(labeled: list[dict]) -> list[dict]:
    """Re-label 10% of LLM-labeled pairs; flag disagreements."""
    llm_labeled = [p for p in labeled if p.get("labeler") != "reactome_hierarchy"]

    # If budget is tight, skip re-labeling but still set flags
    if current_cost() > 1.40:
        log.info("Budget tight ($%.4f); skipping re-label pass.", current_cost())
        for p in labeled:
            p["needs_human_review"] = (
                p.get("confidence") == "low" or
                p.get("agrees_with_hint") is False
            )
        return labeled

    n_check = max(1, int(len(llm_labeled) * 0.10))
    check_sample = random.sample(llm_labeled, n_check)
    log.info("Re-labeling %d pairs for self-consistency...", n_check)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(5)
    second_pass = await label_batch_async(client, check_sample, semaphore)
    second_map = {r["pair_id"]: r["label"] for r in second_pass}

    disagreements = 0
    for pair in labeled:
        pair["needs_human_review"] = (
            pair.get("confidence") == "low" or
            pair.get("agrees_with_hint") is False
        )
        pid = pair["pair_id"]
        if pid in second_map and second_map[pid] != pair["label"]:
            pair["needs_human_review"] = True
            pair["phase4_disagreement"] = True
            pair["phase4_second_label"] = second_map[pid]
            disagreements += 1

    rate = disagreements / n_check if n_check else 0.0
    log.info("Phase 4: %d/%d disagreements (%.1f%%) | Cost: $%.4f",
             disagreements, n_check, rate * 100, current_cost())
    return labeled


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Review notes
# ═══════════════════════════════════════════════════════════════════════════════

def write_review_notes(labeled: list[dict]) -> None:
    label_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)

    for p in labeled:
        label_counts[p.get("label", "unknown")] += 1
        for src_key in ("source_a", "source_b"):
            src = p.get(src_key, "unknown")
            source_counts[src] += 1

    needs_review = [p for p in labeled if p.get("needs_human_review")]

    # Priority 1: agrees_with_hint == False
    hint_disagree = [p for p in labeled if p.get("agrees_with_hint") is False]
    # Priority 2: phase4 disagreements
    phase4_disagree = [p for p in labeled if p.get("phase4_disagreement")]

    review_priority: list[dict] = []
    seen_pids: set[str] = set()

    def add_to_review(p: dict) -> None:
        if p["pair_id"] not in seen_pids:
            seen_pids.add(p["pair_id"])
            review_priority.append(p)

    for p in hint_disagree:
        add_to_review(p)
    for p in phase4_disagree:
        add_to_review(p)

    # Calibration: up to 2 high-confidence per label
    if len(review_priority) < 20:
        for lbl in ("merge", "parent_of", "child_of", "sibling", "unrelated"):
            high_conf = [p for p in labeled
                         if p.get("label") == lbl and p.get("confidence") == "high"]
            random.shuffle(high_conf)
            for p in high_conf[:3]:
                add_to_review(p)
                if len(review_priority) >= 20:
                    break
            if len(review_priority) >= 20:
                break

    review_priority = review_priority[:20]

    lines = [
        "# Biology Gold Set — Review Notes",
        "",
        f"Generated: {time.strftime('%Y-%m-%d')}",
        f"Total pairs: {len(labeled)}",
        "",
        "## Label Distribution",
        "",
    ]
    for lbl in ("merge", "parent_of", "child_of", "sibling", "unrelated"):
        lines.append(f"- **{lbl}**: {label_counts.get(lbl, 0)}")
    lines += [""]

    lines += [
        f"## Flagged needs_human_review: {len(needs_review)}",
        f"  - agrees_with_hint==False: {len(hint_disagree)}",
        f"  - Phase-4 disagreements: {len([p for p in labeled if p.get('phase4_disagreement')])}",
        f"  - low-confidence: {sum(1 for p in labeled if p.get('confidence')=='low')}",
        "",
        "## Source Distribution (counting both source_a and source_b)",
        "",
    ]
    for src, cnt in sorted(source_counts.items()):
        lines.append(f"- **{src}**: {cnt}")
    lines += [""]

    lines += [
        "## Top Pairs for Human Review",
        "",
        "> Priority: agrees_with_hint==False, Phase-4 disagreements, then calibration samples.",
        "",
    ]

    for i, p in enumerate(review_priority, 1):
        lines.append(f"### Pair {i}: `{p['pair_id']}`")
        tag = []
        if p.get("agrees_with_hint") is False:
            tag.append("HINT_DISAGREE")
        if p.get("phase4_disagreement"):
            tag.append("PHASE4_DISAGREE")
        if tag:
            lines.append(f"**Flags**: {', '.join(tag)}")
        lines.append(
            f"- **Label**: `{p.get('label')}` | **Confidence**: {p.get('confidence')} "
            f"| **Strategy**: {p.get('candidate_strategy')}"
        )
        lines.append(
            f"- **Hint**: {p.get('ontology_relation_hint')} | "
            f"**Agrees with hint**: {p.get('agrees_with_hint')}"
        )
        lines.append(f"- **Text A**: _{p['text_a'][:200]}_")
        lines.append(f"- **Text B**: _{p['text_b'][:200]}_")
        lines.append(f"- **Justification**: {p.get('justification', '')}")
        if p.get("phase4_disagreement"):
            lines.append(f"- **Phase-4 second label**: `{p.get('phase4_second_label')}`")
        lines.append("")

    out = STAGING / "REVIEW_NOTES.md"
    out.write_text("\n".join(lines))
    log.info("Review notes: %s", out)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    import numpy as np  # noqa: F401 — ensure it's available before Phase 2

    # Phase 1
    log.info("=== Phase 1: String Mining ===")
    s1 = mine_s1()
    s2, s2_ontology_pairs = mine_s2()
    s3 = mine_s3_optional()

    all_raw = s1 + s2 + s3

    # Deduplicate strings by normalized text
    seen_texts: set[str] = set()
    strings: list[dict] = []
    for s in all_raw:
        key = " ".join((s["text"] or "").lower().split())
        if key and key not in seen_texts:
            seen_texts.add(key)
            strings.append(s)

    counts = {src: sum(1 for s in strings if s["source"] == src)
              for src in ("S1_extraction", "S2_reactome", "S3_go")}
    log.info("Unique strings: %d  (S1=%d, S2=%d, S3=%d)",
             len(strings), counts["S1_extraction"], counts["S2_reactome"], counts["S3_go"])

    # Save staging strings
    strings_path = STAGING / "strings_all.jsonl"
    with open(strings_path, "w") as f:
        for s in strings:
            f.write(json.dumps(s) + "\n")

    # Phase 2
    log.info("=== Phase 2: Candidate Pair Generation ===")
    embeddings = embed_all(strings)
    candidates = generate_candidates(strings, embeddings, s2_ontology_pairs)

    # Select up to 250 candidates with balanced strategies
    ont_parent = [p for p in candidates if p["candidate_strategy"] == "ontology_parent"]
    ont_sibling = [p for p in candidates if p["candidate_strategy"] == "ontology_sibling"]
    topk = [p for p in candidates if p["candidate_strategy"] == "topk_neighbor"]
    rand = [p for p in candidates if p["candidate_strategy"] == "random"]
    topk.sort(key=lambda p: -p["cosine_biolord"])

    selected: list[dict] = []
    selected.extend(ont_parent[:30])
    selected.extend(ont_sibling[:50])
    remaining = 250 - len(selected)
    selected.extend(topk[:remaining - len(rand)])
    selected.extend(rand[:30])
    selected = deduplicate_pairs(selected)[:250]

    strat_counts = defaultdict(int)
    for p in selected:
        strat_counts[p["candidate_strategy"]] += 1
    log.info("Selected %d candidates: %s", len(selected), dict(strat_counts))

    # Save candidates
    cands_path = STAGING / "candidates.jsonl"
    with open(cands_path, "w") as f:
        for p in selected:
            f.write(json.dumps(p) + "\n")
    log.info("Saved %d candidates to %s", len(selected), cands_path)

    # Phase 3
    log.info("=== Phase 3: Labeling ===")
    labeled = await run_llm_labeling(selected, max_llm_pairs=130)

    # Phase 4
    log.info("=== Phase 4: Self-Consistency ===")
    labeled = await self_consistency_check(labeled)

    # Save staging labeled
    labeled_path = STAGING / "labeled.jsonl"
    with open(labeled_path, "w") as f:
        for p in labeled:
            f.write(json.dumps(p) + "\n")

    # Phase 5
    log.info("=== Phase 5: Review Notes + Final Output ===")
    write_review_notes(labeled)

    # Write final output
    with open(FINAL_OUT, "w") as f:
        for p in labeled:
            f.write(json.dumps(p) + "\n")
    log.info("Final output: %s", FINAL_OUT)

    # Summary
    label_dist: dict[str, int] = defaultdict(int)
    src_dist: dict[str, int] = defaultdict(int)
    for p in labeled:
        label_dist[p.get("label", "??")] += 1
        src_dist[p.get("source_a", "??")] += 1

    cost = current_cost()
    print("\n=== CURATION SUMMARY ===")
    print(f"Total pairs: {len(labeled)}")
    print(f"Label distribution: {dict(sorted(label_dist.items()))}")
    print(f"Source_a distribution: {dict(sorted(src_dist.items()))}")
    print(f"needs_human_review: {sum(1 for p in labeled if p.get('needs_human_review'))}")
    print(f"Total API cost: ${cost:.4f}  "
          f"(input={total_input_tokens:,} tokens, output={total_output_tokens:,} tokens)")
    print(f"Final pairs: {FINAL_OUT}")
    print(f"Review notes: {STAGING / 'REVIEW_NOTES.md'}")


if __name__ == "__main__":
    asyncio.run(main())
