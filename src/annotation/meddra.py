"""MedDRA-style normalization for adverse-event terms.

Trial reports use heterogeneous AE wording—"transaminase elevation",
"elevated ALT", "hepatic enzyme increased", "hepatotoxicity" all
describe the same on-mechanism toxicity. To accumulate evidence across
trials we need a shared key: a single ``AdverseEventNode`` id per
clinical concept. This module asks Claude to map a free-text AE term to
a normalized MedDRA-style preferred term + system organ class, with a
JSON disk cache so the same input is never paid for twice.

We don't ship the MedDRA hierarchy itself (proprietary, ~$5k licence).
Claude is asked to emit MedDRA preferred-term wording from its training
data, which is good enough for cross-trial collapsing on the order of
the ~150-trial corpus we expect. If we later license MedDRA, this
module is the chokepoint to swap in a deterministic lookup.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import anthropic

from src.annotation.extractor import _call_messages_with_backoff

logger = logging.getLogger(__name__)

INFERENCE_MODEL = "claude-haiku-4-5-20251001"
_CACHE_PATH_DEFAULT = Path("data/cache/meddra_terms.json")
_AE_ID_RE = re.compile(r"[^a-z0-9]+")


def ae_node_id(preferred_term: str) -> str:
    """Derive a canonical AdverseEventNode id from a MedDRA preferred term.

    ``"Hepatotoxicity"`` → ``"AE:hepatotoxicity"``;
    ``"Aspartate aminotransferase increased"`` →
    ``"AE:aspartate_aminotransferase_increased"``.
    """
    slug = _AE_ID_RE.sub("_", preferred_term.lower()).strip("_")
    return f"AE:{slug}" if slug else "AE:unspecified"


class MeddraCache:
    """Disk-backed cache of raw_term → {preferred_term, system_organ_class}.

    Mirrors the JSONCache pattern in graph/populate.py but stores dict
    values instead of strings. Flushed on every write so a crashed run
    doesn't lose paid-for normalizations.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _CACHE_PATH_DEFAULT
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, str]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                logger.warning(
                    "MedDRA cache %s unreadable; starting empty", self.path
                )
                self._data = {}

    def get(self, raw_term: str) -> dict[str, str] | None:
        return self._data.get(self._key(raw_term))

    def set(self, raw_term: str, value: dict[str, str]) -> None:
        self._data[self._key(raw_term)] = value
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _key(raw_term: str) -> str:
        # Normalize the cache key so cosmetic differences ("Nausea" vs
        # "nausea " vs "  nausea") all hit the same entry.
        return " ".join(raw_term.lower().split())


_NORMALIZER_PROMPT = """\
You are a clinical safety analyst. Map a raw adverse event term from a \
clinical trial report to its MedDRA-style preferred term and System Organ \
Class.

Return ONLY a JSON object on a single line, no markdown, no prose:
{"preferred_term": "...", "system_organ_class": "..."}

Rules:
- preferred_term must be a single MedDRA Lowest Level Term (LLT) or \
Preferred Term (PT) that captures the underlying clinical concept. Strip \
grade qualifiers ("Grade 3 hepatotoxicity" → "Hepatotoxicity") because \
grade is tracked separately.
- Collapse synonyms aggressively: "transaminase elevation", "ALT increased", \
and "hepatic enzyme increased" should all map to the same preferred_term.
- system_organ_class must be one of the 27 MedDRA SOCs (e.g. \
"Hepatobiliary disorders", "Gastrointestinal disorders", \
"Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders").
- If the raw term is too vague to map ("toxicity", "adverse event"), use \
preferred_term="Unspecified adverse event" and \
system_organ_class="General disorders and administration site conditions".
"""


async def normalize_ae_term(
    client: anthropic.AsyncAnthropic,
    raw_term: str,
    cache: MeddraCache,
) -> dict[str, str]:
    """Normalize one raw AE term. Returns {"preferred_term", "system_organ_class"}.

    Hits the cache when present; otherwise calls Claude Haiku, parses the
    JSON response, and writes through. On a parsing failure we fall back
    to using the raw term as the preferred term (no SOC)—better to
    create a slightly noisy node than to lose the AE entirely.
    """
    cached = cache.get(raw_term)
    if cached is not None:
        return cached

    user_msg = f"Raw adverse event term: {raw_term!r}"
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=200,
        temperature=0,
        system=_NORMALIZER_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = response.content[0].text.strip()
    parsed = _safe_parse(text)
    if parsed is None:
        logger.warning(
            "MedDRA normalizer returned unparseable JSON for %r: %r",
            raw_term, text,
        )
        parsed = {
            "preferred_term": raw_term.strip().capitalize(),
            "system_organ_class": "",
        }
    cache.set(raw_term, parsed)
    return parsed


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing ```json ... ``` markdown fence if present.

    Haiku occasionally wraps its JSON response in fences despite the
    prompt asking for a bare object. Mirrors the same fence-strip used in
    ``extractor._call_messages_with_backoff``'s caller.
    """
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence (``` or ```json)—keep everything after
        # the first newline if there is one, else drop the literal "```".
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _safe_parse(text: str) -> dict[str, str] | None:
    text = _strip_code_fence(text)
    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    pt = obj.get("preferred_term")
    soc = obj.get("system_organ_class", "")
    if not isinstance(pt, str) or not pt.strip():
        return None
    return {
        "preferred_term": pt.strip(),
        "system_organ_class": str(soc or "").strip(),
    }
