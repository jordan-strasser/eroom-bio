"""
Tests for the A.5 biology/mechanism gold pair set.

Validates that biology_gold_pairs.jsonl is well-formed, has required keys,
valid labels, and sufficient size.
"""
from __future__ import annotations

import json
from pathlib import Path

GOLD_PATH = Path(__file__).parent.parent / "data" / "eval" / "biology_gold_pairs.jsonl"

REQUIRED_KEYS = {
    "pair_id",
    "text_a",
    "text_b",
    "label",
    "justification",
    "confidence",
    "labeler",
    "ontology_relation_hint",
    "agrees_with_hint",
}

VALID_LABELS = {"merge", "parent_of", "child_of", "sibling", "unrelated"}
VALID_CONFIDENCE = {"high", "medium", "low"}
MIN_PAIRS = 100


def load_pairs() -> list[dict]:
    """Load all pairs from the JSONL file."""
    pairs = []
    with open(GOLD_PATH) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON on line {line_num}: {exc}"
                ) from exc
    return pairs


def test_file_exists() -> None:
    assert GOLD_PATH.exists(), f"Gold pair file not found: {GOLD_PATH}"


def test_valid_jsonl() -> None:
    """Every line must be valid JSON."""
    pairs = load_pairs()
    assert len(pairs) > 0, "Gold pair file is empty"


def test_required_keys() -> None:
    """Every row must have all required keys."""
    pairs = load_pairs()
    for i, pair in enumerate(pairs):
        missing = REQUIRED_KEYS - pair.keys()
        assert not missing, (
            f"Row {i} (pair_id={pair.get('pair_id', '?')}) missing keys: {missing}"
        )


def test_valid_labels() -> None:
    """Every label must be one of the five valid labels."""
    pairs = load_pairs()
    for i, pair in enumerate(pairs):
        label = pair.get("label")
        assert label in VALID_LABELS, (
            f"Row {i} has invalid label {label!r}. "
            f"Must be one of {VALID_LABELS}."
        )


def test_valid_confidence() -> None:
    """Confidence must be high, medium, or low."""
    pairs = load_pairs()
    for i, pair in enumerate(pairs):
        conf = pair.get("confidence")
        assert conf in VALID_CONFIDENCE, (
            f"Row {i} has invalid confidence {conf!r}. "
            f"Must be one of {VALID_CONFIDENCE}."
        )


def test_nonempty_texts() -> None:
    """text_a and text_b must be non-empty strings."""
    pairs = load_pairs()
    for i, pair in enumerate(pairs):
        assert isinstance(pair.get("text_a"), str) and pair["text_a"].strip(), (
            f"Row {i} has empty or invalid text_a"
        )
        assert isinstance(pair.get("text_b"), str) and pair["text_b"].strip(), (
            f"Row {i} has empty or invalid text_b"
        )


def test_nonempty_pair_ids() -> None:
    """pair_id must be a non-empty string."""
    pairs = load_pairs()
    for i, pair in enumerate(pairs):
        assert isinstance(pair.get("pair_id"), str) and pair["pair_id"].strip(), (
            f"Row {i} has empty or invalid pair_id"
        )


def test_pair_ids_unique() -> None:
    """All pair_ids must be unique."""
    pairs = load_pairs()
    ids = [p["pair_id"] for p in pairs]
    assert len(ids) == len(set(ids)), (
        f"Duplicate pair_ids found: "
        f"{len(ids) - len(set(ids))} duplicates"
    )


def test_minimum_pair_count() -> None:
    """Must have at least MIN_PAIRS labeled pairs."""
    pairs = load_pairs()
    assert len(pairs) >= MIN_PAIRS, (
        f"Gold set has only {len(pairs)} pairs; expected >= {MIN_PAIRS}"
    )


def test_all_five_labels_present() -> None:
    """All five label classes must appear at least once."""
    pairs = load_pairs()
    found_labels = {p["label"] for p in pairs}
    missing = VALID_LABELS - found_labels
    assert not missing, f"Missing label classes: {missing}"
