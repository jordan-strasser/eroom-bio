"""Empirical calibration of N_eff and bucket→p_obs tables.

The two tables in ``beliefs.py`` are the only knobs in the conjugate
update. With a labeled holdout set of trials (each carrying the actual
binary outcome of the predicted indication-success), we can refit them
to maximize a calibration target—Brier score and expected calibration
error are the natural choices.

Approach (deferred until ≥50 annotated trials are available):

1. For each holdout trial, recompute ``P(success)`` along its causal
   chain using the current tables.
2. Fit ``EVIDENCE_TYPE_N_EFF`` and ``BUCKET_TO_P_OBS`` jointly via
   constrained optimization:
     - N_eff > 0
     - p_obs ∈ (0, 1), monotone in bucket strength, symmetric around
       AMBIGUOUS=0.5
     - minimize Brier = mean((P̂ − y)²)
3. Report calibration plot (predicted-bin vs observed-frequency) and
   ECE alongside the new tables.

Returning the refitted tables—rather than mutating the module-level
constants—lets callers diff old vs new and snapshot for reproducibility.
"""

from __future__ import annotations

from typing import NamedTuple

from src.graph.models import EvidenceType
from src.inference.beliefs import SupportBucket


class CalibratedTables(NamedTuple):
    n_eff: dict[EvidenceType, float]
    p_obs: dict[SupportBucket, float]
    brier_score: float
    expected_calibration_error: float


def fit_calibrated_tables(holdout: list) -> CalibratedTables:  # noqa: ARG001
    """Stub. See module docstring for the planned recipe."""
    raise NotImplementedError(
        "Calibration is deferred until ≥50 annotated trials are available; "
        "see src/inference/calibration.py module docstring for the recipe."
    )
