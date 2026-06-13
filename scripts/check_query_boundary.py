"""Audit the public tree for private read-path / frontier-query modules.

The query-side counterpart of ``scripts/check_public_snapshots.py``. That script
guards private *values* leaking into a public snapshot (the field gate); this one
guards the private *query modules* (the paid value-extraction) shipping in the
public package — the governing write-path/read-path boundary (see ``BOUNDARY.md``
and ``src/boundary.py``: ``PRIVATE_QUERY_MODULES`` / ``assert_query_private``).

Two tiers, matching each registry entry's ``relocated`` flag:

* A ``relocated=True`` module that reappears in the public tree is a hard
  regression — the frontier query leaked back into the open core. Always fails.
* A ``relocated=False`` module still present is the expected encode-now/move-later
  state (the policy is committed before the approved code move). Reported as a
  PENDING notice; it fails the build only under ``--strict`` (the gate to flip on
  once the relocation lands).

Usage:
    python -m scripts.check_query_boundary            # CI default: fail on regressions only
    python -m scripts.check_query_boundary --strict   # also fail on pending (post-move gate)

Exit code 0 = clean (modulo pending notices), 1 = a private query module is in the
public tree in a way that should fail the build.
"""

from __future__ import annotations

import sys

from src.boundary import (
    PRIVATE_QUERY_MODULES,
    QueryBoundaryViolation,
    assert_query_private,
    pending_query_modules,
    reintroduced_query_modules,
)


def main(argv: list[str]) -> int:
    strict = "--strict" in argv

    if not PRIVATE_QUERY_MODULES:
        print("PRIVATE_QUERY_MODULES is empty — nothing to check.")
        return 0

    pending = pending_query_modules()
    reintroduced = reintroduced_query_modules()

    for rel, meta in sorted(PRIVATE_QUERY_MODULES.items()):
        if rel in reintroduced:
            print(f"LEAK   {rel}  (relocated=True but present → {meta['dest']})")
        elif rel in pending:
            tag = "LEAK  " if strict else "PEND  "
            print(f"{tag} {rel}  (scheduled → {meta['dest']})")
        else:
            print(f"absent {rel}  (private, lives at {meta['dest']})")

    try:
        assert_query_private(strict=strict)
    except QueryBoundaryViolation as exc:
        print(f"\n{exc}")
        return 1

    if pending:
        print(
            f"\n{len(pending)} module(s) PENDING relocation to eroom-enterprise "
            "(policy encoded; code move is the approved next step — see "
            "BOUNDARY_SPLIT_PLAN.md). Not failing the build; run with --strict to "
            "gate once the move lands."
        )
    else:
        print("\nNo private query modules in the public tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
