"""
Supplement curation: label the unlabeled random pairs and add explicit
child_of/unrelated pairs to reach label balance.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections import defaultdict
from pathlib import Path

import anthropic

REPO = Path("/Users/jordanstrasser/Code/eroom-bio/eroom")
STAGING = REPO / "data" / "eval" / "_staging"
FINAL_OUT = REPO / "data" / "eval" / "biology_gold_pairs.jsonl"
ANNOTATIONS_DIR = REPO / "data" / "annotations"

HAIKU_INPUT_COST_PER_M = 1.0
HAIKU_OUTPUT_COST_PER_M = 5.0

total_input_tokens = 0
total_output_tokens = 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

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


def current_cost() -> float:
    return (total_input_tokens / 1_000_000) * HAIKU_INPUT_COST_PER_M + \
           (total_output_tokens / 1_000_000) * HAIKU_OUTPUT_COST_PER_M


async def label_pair_llm(client: anthropic.AsyncAnthropic, pair: dict) -> dict:
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
        clean = raw
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) >= 3 else clean.replace("```json","").replace("```","")
            if clean.startswith("json"):
                clean = clean[4:]
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
        justification = f"Parse error: {raw[:80]}"
        confidence = "low"
    return {
        **pair,
        "label": label,
        "justification": justification,
        "confidence": confidence,
        "labeler": "claude-haiku-4-5",
        "agrees_with_hint": None,
        "needs_human_review": label == "unrelated" and pair.get("cosine_biolord", 1.0) > 0.5,
    }


async def label_batch(client, pairs, sem):
    async def one(p):
        async with sem:
            return await label_pair_llm(client, p)
    return await asyncio.gather(*[one(p) for p in pairs])


def make_pair_id() -> str:
    return str(uuid.uuid4())


async def main():
    # Load existing labeled pairs
    existing = [json.loads(l) for l in open(FINAL_OUT)]
    existing_texts: set[frozenset] = {frozenset([p["text_a"], p["text_b"]]) for p in existing}
    log.info("Existing pairs: %d", len(existing))

    label_counts = defaultdict(int)
    for p in existing:
        label_counts[p["label"]] += 1
    log.info("Current label distribution: %s", dict(label_counts))

    # Load candidates that weren't labeled (the random ones)
    cands = [json.loads(l) for l in open(STAGING / "candidates.jsonl")]
    rand_cands = [p for p in cands if p["candidate_strategy"] == "random"]
    log.info("Random candidates to label: %d", len(rand_cands))

    # ── Build supplemental pairs ─────────────────────────────────────────────
    new_pairs: list[dict] = []

    # 1. Label all random candidates (these cover low-cosine territory → unrelated)
    for p in rand_cands:
        key = frozenset([p["text_a"], p["text_b"]])
        if key not in existing_texts:
            new_pairs.append(p)

    # 2. Build explicit cross-domain pairs (S1 mechanism x S2/S3 biology) for child_of/unrelated
    # Mine S1 strings
    import os
    s1_mechanism = []
    s1_biology = []
    files = sorted(f for f in ANNOTATIONS_DIR.iterdir() if f.name.endswith("_extraction.json"))
    for fp in files:
        data = json.loads(fp.read_text())
        nct = data.get("nct_id", fp.stem.replace("_extraction", ""))
        th = data.get("therapeutic_hypothesis", {})
        mech = (th.get("proposed_mechanism") or "").strip()
        bio = (th.get("intended_biology") or "").strip()
        if mech:
            s1_mechanism.append({"text": mech, "source_id": nct})
        if bio:
            s1_biology.append({"text": bio, "source_id": nct})

    # Pair proposed_mechanism with intended_biology from SAME trial
    # (mechanism → biology is often child_of or parent_of)
    same_trial = {}
    for fp in files:
        data = json.loads(fp.read_text())
        nct = data.get("nct_id", fp.stem.replace("_extraction", ""))
        th = data.get("therapeutic_hypothesis", {})
        mech = (th.get("proposed_mechanism") or "").strip()
        bio = (th.get("intended_biology") or "").strip()
        if mech and bio and mech != bio:
            same_trial[nct] = (mech, bio)

    for nct, (mech, bio) in sorted(same_trial.items()):
        key = frozenset([mech, bio])
        if key not in existing_texts:
            existing_texts.add(key)
            new_pairs.append({
                "pair_id": make_pair_id(),
                "text_a": mech,
                "text_b": bio,
                "source_a": "S1_extraction",
                "source_b": "S1_extraction",
                "source_id_a": nct,
                "source_id_b": nct,
                "cosine_biolord": 0.5,  # placeholder
                "candidate_strategy": "same_trial_mech_bio",
                "ontology_relation_hint": None,
            })

    # 3. Load S2 summations to pair with S1 mechanism texts for unrelated/sibling
    # Already have this from candidates; let's make more deliberate unrelated pairs
    # by pairing S1 trials from different indications
    import itertools
    files_list = sorted(files)
    indication_map = {}
    for fp in files_list:
        data = json.loads(fp.read_text())
        nct = data.get("nct_id", fp.stem.replace("_extraction", ""))
        th = data.get("therapeutic_hypothesis", {})
        mech = (th.get("proposed_mechanism") or "").strip()
        ctx = data.get("context", {})
        indication = ctx.get("indication", "").lower()
        if mech and indication:
            indication_map[nct] = (mech, indication)

    # Group by broad indication area
    onco = [(nct, m) for nct, (m, ind) in indication_map.items()
            if any(x in ind for x in ["cancer", "tumor", "leukemia", "lymphoma", "carcinoma", "melanoma", "glioma", "sarcoma"])]
    cardio = [(nct, m) for nct, (m, ind) in indication_map.items()
              if any(x in ind for x in ["heart", "cardiac", "coronary", "atrial", "ventricular", "hypertension", "cholesterol"])]
    neuro = [(nct, m) for nct, (m, ind) in indication_map.items()
             if any(x in ind for x in ["alzheimer", "parkinson", "multiple sclerosis", "epilepsy", "schizophrenia", "depression", "brain", "neuro"])]
    metabolic = [(nct, m) for nct, (m, ind) in indication_map.items()
                 if any(x in ind for x in ["diabetes", "obesity", "metabolic", "insulin"])]

    # Cross-domain pairs (onco x cardio, onco x neuro, cardio x metabolic) — these are likely unrelated
    cross_domain = []
    for group_a, group_b in [(onco, cardio), (onco, neuro), (cardio, metabolic), (onco, metabolic)]:
        random.shuffle(group_a)
        random.shuffle(group_b)
        for (nct_a, m_a), (nct_b, m_b) in zip(group_a[:8], group_b[:8]):
            if m_a == m_b:
                continue
            key = frozenset([m_a, m_b])
            if key not in existing_texts:
                existing_texts.add(key)
                cross_domain.append({
                    "pair_id": make_pair_id(),
                    "text_a": m_a,
                    "text_b": m_b,
                    "source_a": "S1_extraction",
                    "source_b": "S1_extraction",
                    "source_id_a": nct_a,
                    "source_id_b": nct_b,
                    "cosine_biolord": 0.2,  # placeholder; will be low for cross-domain
                    "candidate_strategy": "cross_domain",
                    "ontology_relation_hint": None,
                })

    new_pairs.extend(cross_domain[:30])
    log.info("New pairs to label: %d (random=%d, same_trial=%d, cross_domain=%d)",
             len(new_pairs),
             len(rand_cands),
             len([p for p in new_pairs if p["candidate_strategy"] == "same_trial_mech_bio"]),
             len([p for p in new_pairs if p["candidate_strategy"] == "cross_domain"]))

    # Label all new pairs
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(8)

    # Deduplicate new_pairs
    seen: set[frozenset] = set()
    deduped_new = []
    for p in new_pairs:
        key = frozenset([p["text_a"], p["text_b"]])
        if key not in seen and p["text_a"] != p["text_b"]:
            seen.add(key)
            deduped_new.append(p)
    log.info("Labeling %d new pairs...", len(deduped_new))

    results = await label_batch(client, deduped_new, sem)

    # Merge with existing
    all_pairs = existing + results

    label_counts2 = defaultdict(int)
    for p in all_pairs:
        label_counts2[p["label"]] += 1
    log.info("Final label distribution: %s", dict(label_counts2))

    # Set needs_human_review properly for new pairs
    for p in results:
        if "needs_human_review" not in p:
            p["needs_human_review"] = p.get("confidence") == "low" or p.get("agrees_with_hint") is False

    # Write final output
    with open(FINAL_OUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    log.info("Written %d pairs to %s", len(all_pairs), FINAL_OUT)

    cost = current_cost()
    print(f"\nSupplement cost: ${cost:.4f}  (input={total_input_tokens:,}, output={total_output_tokens:,})")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Label distribution: {dict(sorted(label_counts2.items()))}")
    print(f"needs_human_review: {sum(1 for p in all_pairs if p.get('needs_human_review'))}")

    return all_pairs


if __name__ == "__main__":
    asyncio.run(main())
