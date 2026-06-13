"""Bayesian inference primitives for the Eroom Bio knowledge graph.

This package owns the principled Beta-Binomial conjugate update used to
turn evidence records into edge belief updates. See ``beliefs.py`` for
the canonical recipe and ``calibration.py`` for the (stubbed) empirical
fitting harness.
"""

from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    EVIDENCE_TYPE_N_EFF,
    SupportBucket,
    apply_virtual_evidence,
    bucket_to_direction,
    effective_n_for_evidence,
    p_obs_for_bucket,
)

__all__ = [
    "BUCKET_TO_P_OBS",
    "EVIDENCE_TYPE_N_EFF",
    "SupportBucket",
    "apply_virtual_evidence",
    "bucket_to_direction",
    "effective_n_for_evidence",
    "p_obs_for_bucket",
]
