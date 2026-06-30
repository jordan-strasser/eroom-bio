"""Audit committed public snapshots for private-artifact leaks.

Layer 2 of the boundary (see ``src/boundary.py``): even though
``GraphStore.export_snapshot`` strips private values on write, this re-scans the
committed public artifacts and fails if any private value is present. Wire it into
CI / a pre-commit hook so a leak fails the build instead of shipping.

Scope: with no arguments it scans the **git-tracked** ``data/exports/*.json`` — the
files that actually get published. All of ``data/exports/`` is gitignored (snapshots
are regenerable, and the sellable field-bearing ones live in eroom-enterprise), so
normally nothing is tracked and the gate is a no-op; local regenerable builds on
disk are deliberately NOT scanned — the gate guards what gets COMMITTED, not scratch.
Pass explicit paths to audit any file (e.g. a pre-publish check of a specific build
before attaching it to a Release).

Usage:
    python -m scripts.check_public_snapshots                 # scan tracked data/exports/
    python -m scripts.check_public_snapshots path/to/*.json  # scan given files

Exit code 0 = clean, 1 = at least one leak found.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.boundary import PrivateArtifactLeak, assert_public_safe

DEFAULT_DIR = Path("data/exports")


def _tracked_exports() -> list[Path]:
    """Git-tracked snapshots under data/exports/ — the committed public artifacts
    this gate guards. All of data/exports/ is gitignored, so normally this is empty;
    only a force-added or legacy-tracked snapshot would appear. Local builds on disk
    are deliberately NOT scanned — the gate guards what gets PUBLISHED, not scratch."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "data/exports/*.json"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [Path(p) for p in out.splitlines() if p.strip()]


def _targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv]
    return sorted(_tracked_exports())


def main(argv: list[str]) -> int:
    targets = _targets(argv)
    if not targets:
        print("No git-tracked snapshots under data/exports/ — nothing to check.")
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
