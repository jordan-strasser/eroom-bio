"""Verify the SapBERT extra works after `pip install -e ".[sapbert]"`.

Loads the model (triggers ~440 MB weight download on first run, then
cached to ~/.cache/huggingface), computes embeddings + cosines for a
small set of labeled drug pairs, and reports.

What "working" looks like:
  - Model loads without error
  - Same-drug pairs (e.g. "Fluorouracil" vs "5-Fluorouracil") have
    cosine ≥ ~0.85
  - Salt-form pairs ("Imatinib" vs "Imatinib mesylate") similar range
  - Different-class pairs ("Fluorouracil" vs "Pembrolizumab") well
    below ~0.7
  - Similar-class-distinct-drug pairs ("Erlotinib" vs "Gefitinib")
    are the failure mode test — moderate but ideally still below
    the 0.92 default threshold

This is a tiny precursor to the full threshold-tuning audit
(task #22), which uses a larger labeled set.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.graph.sapbert_embeddings import (
    SapBertUnavailable,
    cosine_similarity,
    embed_compound_names,
)

# Drug pairs labeled with expected similarity class. The "expected"
# column is the human-curated ground truth we want SapBERT to recover.
#
# Expanded 2026-05-20 to validate the 0.80 default threshold across
# more spelling/salt-form/brand variants and more class-mate failure
# modes. Codename pairs are kept in but commented as KNOWN_LOW since
# SapBERT can't resolve those — CODENAME_TO_INN handles them
# separately and round 23 will add RxNorm + PubChem fallbacks.
PAIRS: list[tuple[str, str, str]] = [
    # ── SAME — spelling / case / punctuation variants ──
    ("Fluorouracil",         "5-Fluorouracil",         "SAME"),
    ("Fluorouracil",         "5-FU",                   "SAME"),
    ("5-fluorouracil",       "fluorouracil",           "SAME"),  # case
    ("Methotrexate",         "Methotrexate Sodium",    "SAME"),
    ("Hydroxychloroquine",   "Hydroxychloroquine sulfate", "SAME"),
    ("Doxorubicin",          "Doxorubicin Hydrochloride", "SAME"),
    # ── SAME — salt forms ──
    ("Imatinib",             "Imatinib mesylate",      "SAME"),
    ("Sorafenib",            "Sorafenib tosylate",     "SAME"),
    ("Dasatinib",            "Dasatinib monohydrate",  "SAME"),
    # ── SAME — brand / INN (SapBERT picks up some, misses others) ──
    ("Pembrolizumab",        "Keytruda",               "SAME"),
    ("Nivolumab",            "Opdivo",                 "SAME"),
    ("Vemurafenib",          "Zelboraf",               "SAME"),
    ("Bevacizumab",          "Avastin",                "SAME"),
    # ── KNOWN_LOW — codenames, handled by CODENAME_TO_INN not SapBERT ──
    ("Pembrolizumab",        "MK-3475",                "KNOWN_LOW"),
    ("Selumetinib",          "AZD6244",                "KNOWN_LOW"),
    ("Dabrafenib",           "GSK2118436",             "KNOWN_LOW"),
    # ── DIFFERENT — distinct mechanisms, clearly separable ──
    ("Fluorouracil",         "Pembrolizumab",          "DIFFERENT"),
    ("Imatinib",             "Aspirin",                "DIFFERENT"),
    ("Doxorubicin",          "Pembrolizumab",          "DIFFERENT"),
    ("Atorvastatin",         "Methotrexate",           "DIFFERENT"),
    ("Insulin",              "Acetaminophen",          "DIFFERENT"),
    # ── SIMILAR_CLASS_DISTINCT — must stay safely below 0.80 ──
    ("Erlotinib",            "Gefitinib",              "SIMILAR_CLASS_DISTINCT"),
    ("Paclitaxel",           "Docetaxel",              "SIMILAR_CLASS_DISTINCT"),
    ("Nivolumab",            "Pembrolizumab",          "SIMILAR_CLASS_DISTINCT"),
    ("Cetuximab",            "Panitumumab",            "SIMILAR_CLASS_DISTINCT"),  # anti-EGFR mAbs
    ("Imatinib",             "Dasatinib",              "SIMILAR_CLASS_DISTINCT"),  # BCR-ABL TKIs
    ("Vemurafenib",          "Dabrafenib",             "SIMILAR_CLASS_DISTINCT"),  # BRAF inhibitors
    ("Atorvastatin",         "Rosuvastatin",           "SIMILAR_CLASS_DISTINCT"),  # statins
    ("Trametinib",           "Cobimetinib",            "SIMILAR_CLASS_DISTINCT"),  # MEK inhibitors
    ("Carboplatin",          "Cisplatin",              "SIMILAR_CLASS_DISTINCT"),  # platinums
]


def main() -> int:
    print(">>> Loading SapBERT and embedding test drug names...")
    print(
        "    (first run downloads ~440 MB of weights to "
        "~/.cache/huggingface)\n"
    )
    # Collect unique names so we batch into one model.encode() call.
    unique_names = sorted({n for pair in PAIRS for n in pair[:2]})
    try:
        # Use a per-run cache so this script doesn't pollute the
        # real data/cache/sapbert_embeddings.json. The point is to
        # exercise the model, not seed the production cache.
        verify_cache = Path("data/cache/sapbert_verify.json")
        vectors = embed_compound_names(
            unique_names, cache_path=verify_cache,
        )
    except SapBertUnavailable as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        print(
            "\nInstall the optional dependency:\n"
            "  pip install -e \".[sapbert]\"\n",
            file=sys.stderr,
        )
        return 1
    name_to_vec = dict(zip(unique_names, vectors))
    print(f"    embedded {len(unique_names)} names "
          f"(vector dim={len(vectors[0])})\n")

    # The Phase B threshold we're validating. Update both locations
    # together if you tune this.
    PHASE_B_THRESHOLD = 0.80

    # Report.
    print(f'{"A":<25}  {"B":<25}  {"expected":<25}  cosine')
    print("-" * 90)
    same_scores: list[float] = []
    diff_scores: list[float] = []
    sim_class_scores: list[float] = []
    known_low_scores: list[float] = []
    for a, b, label in PAIRS:
        cos = cosine_similarity(name_to_vec[a], name_to_vec[b])
        marker = ""
        if label == "SAME":
            same_scores.append(cos)
            marker = "  ✓ above threshold" if cos >= PHASE_B_THRESHOLD else "  ⚠ below threshold (falls through to OT/codename layer)"
        elif label == "DIFFERENT":
            diff_scores.append(cos)
            marker = "  ✓ below threshold" if cos < PHASE_B_THRESHOLD else "  ✗ ABOVE threshold (false-merge risk!)"
        elif label == "SIMILAR_CLASS_DISTINCT":
            sim_class_scores.append(cos)
            marker = "  ✓ below threshold" if cos < PHASE_B_THRESHOLD else "  ✗ ABOVE threshold (would false-merge class-mates!)"
        else:  # KNOWN_LOW (codenames SapBERT can't resolve)
            known_low_scores.append(cos)
            marker = "  (expected low — handled by CODENAME_TO_INN)"
        print(f'{a:<25}  {b:<25}  {label:<25}  {cos:.4f}{marker}')

    print()
    print("Summary:")
    if same_scores:
        print(f"  SAME (n={len(same_scores)}):                  "
              f"min={min(same_scores):.4f}  "
              f"max={max(same_scores):.4f}  "
              f"mean={sum(same_scores)/len(same_scores):.4f}")
    if known_low_scores:
        print(f"  KNOWN_LOW codenames (n={len(known_low_scores)}):       "
              f"min={min(known_low_scores):.4f}  "
              f"max={max(known_low_scores):.4f}  "
              f"mean={sum(known_low_scores)/len(known_low_scores):.4f}")
    if sim_class_scores:
        print(f"  SIMILAR_CLASS_DISTINCT (n={len(sim_class_scores)}):   "
              f"min={min(sim_class_scores):.4f}  "
              f"max={max(sim_class_scores):.4f}  "
              f"mean={sum(sim_class_scores)/len(sim_class_scores):.4f}")
    if diff_scores:
        print(f"  DIFFERENT (n={len(diff_scores)}):             "
              f"min={min(diff_scores):.4f}  "
              f"max={max(diff_scores):.4f}  "
              f"mean={sum(diff_scores)/len(diff_scores):.4f}")

    # Validate the threshold against the failure-mode pairs: NO
    # SIMILAR_CLASS_DISTINCT pair may cross the threshold (would be a
    # false merge), and NO DIFFERENT pair may cross either.
    print()
    print(f"Validating threshold = {PHASE_B_THRESHOLD}:")
    above_threshold_sim = [
        (a, b, c) for (a, b, _), c in zip(
            [p for p in PAIRS if p[2] == "SIMILAR_CLASS_DISTINCT"],
            sim_class_scores,
        ) if c >= PHASE_B_THRESHOLD
    ]
    above_threshold_diff = [
        (a, b, c) for (a, b, _), c in zip(
            [p for p in PAIRS if p[2] == "DIFFERENT"], diff_scores,
        ) if c >= PHASE_B_THRESHOLD
    ]
    rc = 0
    if above_threshold_sim:
        print(f"  [FAIL] {len(above_threshold_sim)} class-mate pair(s) "
              f"above threshold (would false-merge):")
        for a, b, c in above_threshold_sim:
            print(f"     {a!r} / {b!r} → {c:.4f}")
        rc = 1
    if above_threshold_diff:
        print(f"  [FAIL] {len(above_threshold_diff)} different-class "
              f"pair(s) above threshold:")
        for a, b, c in above_threshold_diff:
            print(f"     {a!r} / {b!r} → {c:.4f}")
        rc = 1
    if rc == 0:
        below_same = [c for c in same_scores if c < PHASE_B_THRESHOLD]
        print(f"  [OK] No false-merge candidates at threshold {PHASE_B_THRESHOLD}")
        if below_same:
            print(f"  [INFO] {len(below_same)} SAME pair(s) below threshold "
                  f"(will fall through to OT/codename layer):")
            for (a, b, _), c in zip(
                [p for p in PAIRS if p[2] == "SAME"], same_scores,
            ):
                if c < PHASE_B_THRESHOLD:
                    print(f"     {a!r} / {b!r} → {c:.4f}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
