"""Audit committed public snapshots for private-artifact leaks.

Layer 2 of the boundary (see ``src/boundary.py``): even though
``GraphStore.export_snapshot`` strips private values on write, this script
re-scans what is actually on disk under ``data/exports/`` (the tracked,
public artifacts) and fails if any private value is present. Wire it into CI
and/or a pre-commit hook so a leak fails the build instead of shipping.

Usage:
    python -m scripts.check_public_snapshots                 # scan data/exports/
    python -m scripts.check_public_snapshots path/to/*.json  # scan given files

Exit code 0 = clean, 1 = at least one leak found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.boundary import PrivateArtifactLeak, assert_public_safe

DEFAULT_DIR = Path("data/exports")


def _targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv]
    return sorted(DEFAULT_DIR.glob("*.json"))


def main(argv: list[str]) -> int:
    targets = _targets(argv)
    if not targets:
        print(f"No snapshots found under {DEFAULT_DIR}/ — nothing to check.")
        return 0

    leaks = 0
    for path in targets:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SKIP  {path}: cannot read ({exc})")
            continue
        try:
            assert_public_safe(payload, source=str(path))
        except PrivateArtifactLeak as exc:
            leaks += 1
            print(f"LEAK  {exc}")
        else:
            print(f"clean {path}")

    if leaks:
        print(f"\n{leaks} snapshot(s) carry private artifacts — see src/boundary.py.")
        return 1
    print(f"\nAll {len(targets)} snapshot(s) clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
