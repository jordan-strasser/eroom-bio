"""Curated codename → INN mapping for compound canonicalization.

The OT / ChEMBL alias machinery already pulls dev codes (BMS-936558,
AZD-6244, MK-3475) onto compound nodes as aliases — but it doesn't
necessarily make the INN (international nonproprietary name, e.g.
nivolumab, selumetinib, pembrolizumab) the canonical id when a trial's
CT.gov intervention text uses the code form. Result: the AZD6244
node and a future "Selumetinib" trial would create two compound nodes
unless the round-18 audit explicitly bridges them.

This table is the hand-curated bridge. Keys are normalized codename
slugs (lowercase, `_` separator); values are the canonical INN slugs
the populator should standardize on. Adds + updates are accepted as
they're discovered; the existing OT-pulled aliases on each compound
node already carry the raw forms, so this table just controls which
form becomes the node id.

Apply via `scripts/canonicalize_codenames.py` (one-off migration) or
in the populator's compound-canonicalization step (forthcoming).
"""

from __future__ import annotations

# slug(code) → slug(INN)
CODENAME_TO_INN: dict[str, str] = {
    # MEK inhibitors
    "azd6244": "selumetinib",
    "azd_6244": "selumetinib",
    "arry_142886": "selumetinib",
    "arry_886": "selumetinib",
    # Anti-PD-1
    "mk_3475": "pembrolizumab",
    "mk3475": "pembrolizumab",
    "lambrolizumab": "pembrolizumab",  # earlier name
    "bms_936558": "nivolumab",
    "bms936558": "nivolumab",
    "ono_4538": "nivolumab",
    # Anti-CTLA-4
    "mdx_010": "ipilimumab",
    "mdx010": "ipilimumab",
    # BRAF inhibitors
    "gsk2118436": "dabrafenib",
    "plx4032": "vemurafenib",
    "ro5185426": "vemurafenib",
    "rg7204": "vemurafenib",
    # MEK inhibitors (trametinib)
    "gsk1120212": "trametinib",
    "jtp_74057": "trametinib",
    # Anti-PD-L1
    "mpdl3280a": "atezolizumab",
    "rg7446": "atezolizumab",
    "msb0010718c": "avelumab",
    "medi4736": "durvalumab",
    # Other immune-onc
    "pdr001": "spartalizumab",
    # Cell therapy / TIL
    "lifileucel": "lifileucel",  # placeholder for future entries
}


def canonical_inn_for(slug: str) -> str | None:
    """Return the INN slug for a known codename slug, else None."""
    return CODENAME_TO_INN.get(slug)
