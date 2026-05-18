"""ChEMBL REST API client — non-protein target inference.

Open Targets exposes ChEMBL drug data via GraphQL but generic-ifies
`actionType` to "INHIBITOR" and returns `mechanismsOfAction: null` for
many drugs. ChEMBL's raw REST API exposes the original structured
fields including `target_chembl_id` pointing to typed entities
(NUCLEIC-ACID, PROTEIN COMPLEX GROUP, SINGLE PROTEIN, etc.).

This adapter is the non-protein-target fallback for the populator: when
OT returns no protein targets for a drug, we query ChEMBL directly to
discover whether the drug binds a structural macromolecule (DNA, RNA,
microtubule complex) and map it to the right TargetType + namespaced id.

Two id-translation tables, both small and stable:

  - ChEMBL target id → eroom-bio namespaced id (CHEBI / GO / CL).
    ChEMBL has a finite set of macromolecular targets used across
    hundreds of drugs (DNA = CHEMBL2311221 is the same node for
    carboplatin, cisplatin, cyclophosphamide, temozolomide, etc.).
    Maintaining the translation table is order-of-magnitude smaller
    than maintaining per-drug curated mappings.

  - ChEMBL mechanism_of_action text → MechanismCategory enum.
    ChEMBL's mechanism text is curated and short ("DNA inhibitor",
    "Tubulin inhibitor"); keyword match into MechanismCategory works
    cleanly without an LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from src.graph.models import MechanismCategory, TargetType

logger = logging.getLogger(__name__)

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


# ── ChEMBL target_chembl_id → (namespaced_id, name, target_type) ─────────
#
# ChEMBL's macromolecular targets are a small fixed set used across many
# drugs. Each maps to a stable external-ontology id (CHEBI for chemical
# entities, GO cellular_component, Cell Ontology for cell types). The
# table is intentionally small — only entries where the ChEMBL target
# has a clean public-ontology equivalent are included; the rest fall
# through to the UNKNOWN-target path (the populator's existing
# behavior).
_CHEMBL_TARGET_TO_NAMESPACED: dict[str, tuple[str, str, TargetType]] = {
    "CHEMBL2311221": ("CHEBI:16991", "DNA", TargetType.MACROMOLECULE),
    "CHEMBL2311222": (
        "CHEBI:33697", "RNA (ribonucleic acid)", TargetType.MACROMOLECULE,
    ),
    "CHEMBL2095182": (
        "CHEBI:35797", "microtubule", TargetType.MACROMOLECULE,
    ),
}


# ── ChEMBL mechanism_of_action text → MechanismCategory ──────────────────
#
# ChEMBL's MoA strings are short curated phrases. Keyword match by
# substring (case-insensitive). First match wins — order matters; put
# more-specific phrases above broader ones.
_MOA_TEXT_TO_MECHANISM: list[tuple[str, MechanismCategory]] = [
    ("dna cross-link", MechanismCategory.DNA_CROSSLINKING),
    ("dna crosslink", MechanismCategory.DNA_CROSSLINKING),
    ("alkylating", MechanismCategory.DNA_CROSSLINKING),
    ("methylating", MechanismCategory.DNA_DAMAGE),
    ("intercalator", MechanismCategory.DNA_DAMAGE),
    ("topoisomerase", MechanismCategory.DNA_DAMAGE),
    ("dna inhibitor", MechanismCategory.DNA_DAMAGE),
    ("dna polymerase", MechanismCategory.DNA_DAMAGE),
    ("rna polymerase", MechanismCategory.ENZYME_INHIBITION),
    ("rna inhibitor", MechanismCategory.ANTIMETABOLITE),
    ("tubulin", MechanismCategory.MICROTUBULE_BINDING),
    ("microtubule", MechanismCategory.MICROTUBULE_BINDING),
    ("thymidylate synthase", MechanismCategory.ANTIMETABOLITE),
    ("antimetabolite", MechanismCategory.ANTIMETABOLITE),
]


def map_moa_text_to_mechanism(moa: str) -> MechanismCategory | None:
    """Map a ChEMBL mechanism_of_action text to a MechanismCategory.

    Returns None when no keyword matches — caller falls back to the
    existing LLM mechanism-inference path.
    """
    if not moa:
        return None
    lower = moa.lower()
    for keyword, category in _MOA_TEXT_TO_MECHANISM:
        if keyword in lower:
            return category
    return None


def is_likely_diagnostic(metadata: dict[str, Any] | None) -> bool:
    """Decide whether a ChEMBL molecule profile signals diagnostic intent.

    Two signals from the molecule endpoint, either sufficient on its own:

      1. ``indication_class`` text containing "diagnostic" — explicit
         classification by ChEMBL curators. Strongest signal.
      2. ``therapeutic_flag=False`` AND ``max_phase`` is None or < 1 —
         the drug isn't marked therapeutic AND isn't in any meaningful
         clinical-trial phase. Catches Fluorescein (False/None) and
         pure imaging dyes; AVOIDS dropping the long list of real
         therapeutics ChEMBL has miscategorized
         (PDR001/LAG525/fotemustine/UCN-01/T-VEC all have
         therapeutic_flag=False but max_phase 2-3 — they ARE in trials
         and are real therapies).

    Known limitation: FDA-approved diagnostics that ChEMBL records with
    max_phase=4.0 (Lymphoseek being the canonical example, since it's
    a real approved diagnostic that progressed through phase trials)
    slip through this rule. The trade-off is intentional — ChEMBL's
    ``therapeutic_flag`` is too unreliable as a sole signal (it's False
    for many real therapeutics), so we lean conservative and accept a
    few diagnostic-shaped pass-throughs. Catching Lymphoseek-class
    would require post-extraction signal (the LLM's interpretation of
    intent from the trial's mechanism description) — round-14 work.

    Returns False when metadata is missing — no signal to filter on,
    safer to keep.
    """
    if not metadata:
        return False
    indication_class = (metadata.get("indication_class") or "").lower()
    if "diagnostic" in indication_class:
        return True
    if metadata.get("therapeutic_flag") is False:
        max_phase = metadata.get("max_phase")
        if max_phase is None:
            return True
        try:
            if float(max_phase) < 1:
                return True
        except (TypeError, ValueError):
            return True
    return False


def translate_chembl_target(
    chembl_target_id: str,
) -> tuple[str, str, TargetType] | None:
    """Map a ChEMBL target id to a namespaced (id, name, target_type) tuple.

    Returns None when the target isn't in the macromolecular-target
    translation table — caller treats it as UNKNOWN (the populator's
    existing fallback path).
    """
    return _CHEMBL_TARGET_TO_NAMESPACED.get(chembl_target_id)


class ChEMBLClient:
    """Minimal ChEMBL REST client for drug → mechanism lookups.

    The client is intentionally small — we only use the
    `mechanism.json` and `target/{id}.json` endpoints. Both responses
    are stable, low-volume, and cached on disk to avoid hammering EBI.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._timeout = timeout
        self._cache_path = cache_path or Path("data/cache/chembl_mechanisms.json")
        self._cache: dict[str, dict[str, Any]] | None = None

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}
        else:
            self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    async def get_drug_mechanisms(
        self, chembl_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all `mechanism_of_action` records for a ChEMBL drug id.

        Each record has at least:
          - `action_type`: ChEMBL action verb (often generic "INHIBITOR")
          - `mechanism_of_action`: curated descriptive phrase
            ("DNA inhibitor", "Tubulin inhibitor")
          - `target_chembl_id`: ChEMBL target id this mechanism acts on

        Returns an empty list when ChEMBL has no record for the drug.
        Cached on disk per chembl_id.
        """
        cache = self._load_cache()
        key = f"mechanism:{chembl_id}"
        if key in cache:
            return cache[key].get("mechanisms") or []

        url = f"{CHEMBL_BASE_URL}/mechanism.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    url, params={"molecule_chembl_id": chembl_id},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "ChEMBL mechanism lookup failed for %s: %s", chembl_id, exc,
            )
            cache[key] = {"mechanisms": []}
            self._save_cache()
            return []

        mechanisms = data.get("mechanisms") or []
        cache[key] = {"mechanisms": mechanisms}
        self._save_cache()
        return mechanisms

    async def get_molecule_metadata(
        self, chembl_id: str,
    ) -> dict[str, Any] | None:
        """Fetch ChEMBL molecule metadata for therapeutic-vs-diagnostic detection.

        Returns ``{therapeutic_flag, max_phase, indication_class}`` or
        None when the drug isn't in ChEMBL. Cached on disk.

        - ``therapeutic_flag`` (bool): ChEMBL's own classification of
          "is this a therapeutic agent?" Has false negatives (T-VEC is
          marked False despite being an FDA-approved oncolytic).
        - ``max_phase`` (float or None): highest clinical trial phase
          observed. Diagnostic agents typically have None or 0.
        - ``indication_class`` (str or None): free-text classification
          ("Antineoplastic", "Diagnostic agent", etc.) — when present
          and contains "diagnostic", that's an explicit signal.
        """
        if not chembl_id:
            return None
        cache = self._load_cache()
        key = f"molecule:{chembl_id}"
        if key in cache:
            return cache[key].get("metadata")

        url = f"{CHEMBL_BASE_URL}/molecule/{chembl_id}.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "ChEMBL molecule metadata lookup failed for %s: %s",
                chembl_id, exc,
            )
            cache[key] = {"metadata": None}
            self._save_cache()
            return None

        meta = {
            "therapeutic_flag": data.get("therapeutic_flag"),
            "max_phase": data.get("max_phase"),
            "indication_class": data.get("indication_class"),
        }
        cache[key] = {"metadata": meta}
        self._save_cache()
        return meta

    async def search_drug_by_name(self, drug_name: str) -> str | None:
        """Resolve a drug name to its ChEMBL molecule id via the search API.

        Used as a fallback when Open Targets can't resolve the drug
        (returns chembl_id=None). ChEMBL's own search has better
        coverage for older / less-marketed drugs (dacarbazine,
        fotemustine, etc.) that OT's curation has thinned out.

        Returns the best-matching molecule's chembl_id, or None when
        ChEMBL has no record. Cached on disk per drug name.
        """
        if not drug_name or not drug_name.strip():
            return None
        cache = self._load_cache()
        normalized = drug_name.strip().lower()
        key = f"search:{normalized}"
        if key in cache:
            return cache[key].get("chembl_id")

        url = f"{CHEMBL_BASE_URL}/molecule/search.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params={"q": drug_name})
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "ChEMBL molecule search failed for %r: %s", drug_name, exc,
            )
            cache[key] = {"chembl_id": None}
            self._save_cache()
            return None

        molecules = data.get("molecules") or []
        chembl_id: str | None = None
        # Prefer an exact pref_name match; otherwise fall back to the
        # first hit (ChEMBL ranks by relevance).
        for m in molecules:
            pref = (m.get("pref_name") or "").lower()
            if pref == normalized:
                chembl_id = m.get("molecule_chembl_id")
                break
        if chembl_id is None and molecules:
            chembl_id = molecules[0].get("molecule_chembl_id")

        cache[key] = {"chembl_id": chembl_id}
        self._save_cache()
        return chembl_id

    async def get_target_info(
        self, target_chembl_id: str,
    ) -> dict[str, Any] | None:
        """Fetch target metadata for a ChEMBL target id.

        Returns ``{"pref_name", "target_type", "organism"}`` or None
        when the target doesn't resolve.
        """
        cache = self._load_cache()
        key = f"target:{target_chembl_id}"
        if key in cache:
            return cache[key].get("target")

        url = f"{CHEMBL_BASE_URL}/target/{target_chembl_id}.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "ChEMBL target lookup failed for %s: %s", target_chembl_id, exc,
            )
            cache[key] = {"target": None}
            self._save_cache()
            return None

        target = {
            "pref_name": data.get("pref_name"),
            "target_type": data.get("target_type"),
            "organism": data.get("organism"),
        }
        cache[key] = {"target": target}
        self._save_cache()
        return target

    async def infer_non_protein_target(
        self,
        chembl_id: str | None,
        drug_name: str | None = None,
    ) -> tuple[str, str, TargetType, MechanismCategory | None] | None:
        """Infer (target_id, target_name, target_type, mechanism) for a drug.

        Strategy:
          1. If `chembl_id` is missing, search ChEMBL by `drug_name` to
             resolve it. ChEMBL has better coverage than OT for older
             drugs that OT's curation has thinned out (dacarbazine,
             fotemustine — both real cases from the round-11 audit).
          2. Fetch the drug's mechanism records from ChEMBL.
          3. For each record, translate its target_chembl_id via the
             namespaced-id table.
          4. Map the mechanism_of_action text to a MechanismCategory.
          5. Return the first record whose target is in the translation
             table — multiple mechanisms for the same drug usually
             re-describe the same target (cross-linking + alkylating
             both → DNA).

        Returns None when the drug has no ChEMBL mechanism record or
        none of its targets are in the translation table (the caller
        falls back to the UNKNOWN-target path).
        """
        if not chembl_id and drug_name:
            chembl_id = await self.search_drug_by_name(drug_name)
        if not chembl_id:
            return None
        mechanisms = await self.get_drug_mechanisms(chembl_id)
        for m in mechanisms:
            tid = m.get("target_chembl_id")
            if not tid:
                continue
            translated = translate_chembl_target(tid)
            if translated is None:
                continue
            namespaced_id, name, target_type = translated
            mechanism = map_moa_text_to_mechanism(
                m.get("mechanism_of_action") or "",
            )
            return (namespaced_id, name, target_type, mechanism)
        return None
