"""Tests for scripts/inspect_trial.py helpers."""

from src.graph.models import CausalChain, TrialArm, TrialSubgraph
from scripts.inspect_trial import _pick_treatment_chain


def _make_chain(
    arm_id: str,
    compound_id: str,
    target_id: str,
    indication_id: str = "melanoma",
) -> CausalChain:
    return CausalChain(
        arm_id=arm_id,
        compound_id=compound_id,
        subgroup_population_id=f"{indication_id}__unselected",
        target_id=target_id,
        mechanism_id="m1",
        biology_id="b1",
        indication_id=indication_id,
        endpoint_id="e1",
    )


def _make_subgraph(chains: list[CausalChain]) -> TrialSubgraph:
    arm = TrialArm(
        arm_id="arm1",
        compound_ids=["c1"],
        regimen_compound_id="c1",
        is_combination=False,
    )
    return TrialSubgraph(
        trial_id="NCT00000001",
        phase="3",
        arms=[arm],
        chains=chains,
        parent_population_id="melanoma__unselected",
    )


def test_picker_prefers_chain_with_resolved_target():
    """Mirrors NCT03618641: a degenerate cmp_001 chain (target=UNKNOWN) precedes
    a real nivolumab chain (target=PDCD1). Without the target-aware rule the
    picker headlines the degenerate one and the prediction misses the signal.
    """
    degenerate = _make_chain("arm_cmp", "cmp_001", target_id="UNKNOWN")
    real = _make_chain("arm_niv", "nivolumab", target_id="PDCD1")
    ts = _make_subgraph([degenerate, real])

    picked = _pick_treatment_chain(ts)

    assert picked is real


def test_picker_falls_back_to_eligible_when_no_target_resolved():
    """If no chain has a resolved target, still return an eligible chain
    rather than skipping past placebo/UNKNOWN to nothing."""
    only_unknown_target = _make_chain("arm_x", "some_compound", target_id="UNKNOWN")
    ts = _make_subgraph([only_unknown_target])

    picked = _pick_treatment_chain(ts)

    assert picked is only_unknown_target


def test_picker_skips_placebo_and_unknown_compound():
    placebo = _make_chain("arm_p", "placebo", target_id="UNKNOWN")
    unknown = _make_chain("arm_u", "UNKNOWN", target_id="UNKNOWN")
    real = _make_chain("arm_r", "nivolumab", target_id="PDCD1")
    ts = _make_subgraph([placebo, unknown, real])

    picked = _pick_treatment_chain(ts)

    assert picked is real
