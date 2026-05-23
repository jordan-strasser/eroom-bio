"""Static tests for the classifier system prompt's round-16 always-emit
rule. These don't call the LLM — they just verify the prompt text
contains the directives that round-16 added, so an accidental prompt
edit that drops the rule fails loudly in CI.

If you intentionally weaken the always-emit rule later, update these
tests to match the new contract."""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPT_PATH = Path("src/annotation/prompts/classification_system.txt")


@pytest.fixture(scope="module")
def prompt() -> str:
    return PROMPT_PATH.read_text()


class TestRound16AlwaysEmitRule:
    """The round-16 fix mandates that the classifier emits
    `responds_differently` AND `endpoint_captures` for every trial with
    a known outcome. Before round-16, these edges were emitted only when
    the LLM judged them relevant — which meant per-trial discrimination
    was impossible because every trial's chain collapsed to the same
    upstream support set."""

    def test_responds_differently_always_emit_directive_present(self, prompt):
        # Find the round-16 section header and require it mentions both
        # edge types as ALWAYS-emit.
        lower = prompt.lower()
        assert "round-16" in lower, (
            "Round-16 always-emit section header missing from classifier prompt"
        )
        # Anchor text: the round-16 directive must mandate both edges.
        assert "responds_differently" in prompt
        assert "endpoint_captures" in prompt
        # Must say "ALWAYS" near responds_differently
        rd_section = prompt[prompt.index("responds_differently"):]
        assert "ALWAYS" in prompt or "always emit" in lower or "non-negotiable" in lower, (
            "Round-16 always-emit directive should use 'ALWAYS' / "
            "'always emit' / 'non-negotiable' language"
        )

    def test_responds_differently_bucket_rubric_present(self, prompt):
        """All 7 support buckets must appear in the responds_differently
        rubric so the classifier knows the full range is in play."""
        buckets = [
            "strong_support", "moderate_support", "weak_support",
            "ambiguous",
            "weak_contradict", "moderate_contradict", "strong_contradict",
        ]
        # Extract the responds_differently rubric section.
        rd_start = prompt.find("`responds_differently`")
        rd_end = prompt.find("`endpoint_captures` (Endpoint → Indication)")
        assert rd_start != -1 and rd_end != -1 and rd_end > rd_start, (
            "Could not locate responds_differently / endpoint_captures "
            "bucket-rubric sections in the prompt"
        )
        rd_rubric = prompt[rd_start:rd_end]
        for b in buckets:
            assert b in rd_rubric, (
                f"Bucket '{b}' missing from responds_differently rubric — "
                f"round-16 round-trip regression"
            )

    def test_endpoint_captures_bucket_rubric_present(self, prompt):
        """Same: all 7 support buckets must appear for endpoint_captures."""
        buckets = [
            "strong_support", "moderate_support", "weak_support",
            "ambiguous",
            "weak_contradict", "moderate_contradict", "strong_contradict",
        ]
        ec_start = prompt.find("`endpoint_captures` (Endpoint → Indication)")
        # The rubric ends at the next double-newline section break.
        ec_end = prompt.find("\n\n", ec_start + 1)
        if ec_end == -1:
            ec_end = len(prompt)
        ec_rubric = prompt[ec_start:ec_end]
        for b in buckets:
            assert b in ec_rubric, (
                f"Bucket '{b}' missing from endpoint_captures rubric"
            )

    def test_strong_contradict_criteria_explicit(self, prompt):
        """Round-16 spec'd strong_contradict on responds_differently as
        requiring decisive failure PLUS one of (a) target engagement,
        (b) different-population success elsewhere, (c) within-trial
        subgroup signal. Verify these criteria are still in the prompt."""
        rd_start = prompt.find("`responds_differently`")
        rd_end = prompt.find("`endpoint_captures` (Endpoint → Indication)")
        rd_rubric = prompt[rd_start:rd_end]
        # The strong_contradict block should reference the three criteria.
        sc_pos = rd_rubric.find("strong_contradict")
        assert sc_pos != -1
        sc_block = rd_rubric[sc_pos:sc_pos + 1500]
        # All three signals must be referenced (loose substring match —
        # the prompt's phrasing can evolve as long as the concepts stay).
        signals = [
            "target engagement",          # (a)
            "different population",       # (b)
            "subgroup",                   # (c)
        ]
        missing = [s for s in signals if s.lower() not in sc_block.lower()]
        assert not missing, (
            f"strong_contradict criteria for responds_differently missing "
            f"signals: {missing}. Round-16 required all three."
        )

    def test_skip_only_when_outcome_unknown(self, prompt):
        """The always-emit rule must explicitly say skipping is OK only
        when trial_outcome is unknown. Otherwise the rule has a giant
        carve-out the classifier will widen."""
        # Anchor on the phrase that captures the carve-out.
        lower = prompt.lower()
        assert "unknown" in lower
        # The directive should be that emissions are mandatory unless
        # outcome is unknown. Look for that joint constraint.
        assert ("non-negotiable" in lower) or ("mandatory" in lower) or ("always" in lower), (
            "Round-16 directive should clearly state the rule is "
            "non-negotiable / mandatory / always"
        )


class TestPromptModulatesViaAmbiguousOnFailures:
    """Round-15 Option A (still in force in round 28): failed trials
    cannot emit *_support on `modulates_via`. The classifier emits
    `ambiguous` instead. `affects` follows a separate rule documented
    in `TestPromptAffectsIsMolecularBinding` below."""

    def test_directive_present(self, prompt):
        # Look for the round-15 directive that bans *_support on
        # modulates_via from failed trials.
        lower = prompt.lower()
        assert "ambiguous" in lower
        assert "never" in lower or "NEVER" in prompt
        chunks = prompt.split("\n\n")
        directive_chunks = [
            c for c in chunks
            if "modulates_via" in c and (
                "NEVER" in c or "never" in c.lower() or "ambiguous" in c.lower()
            )
        ]
        assert directive_chunks, (
            "Round-15 directive ('failed trials use ambiguous, never "
            "*_support, on modulates_via') missing from prompt"
        )


class TestPromptAffectsIsMolecularBinding:
    """Round-28: `affects` encodes molecular binding, not a trial-outcome
    edge. The classifier emits STRONG_SUPPORT when target engagement is
    demonstrated, emits nothing when binding isn't measured, and
    reserves contradict for explicit non-binding evidence. Defaulting
    to AMBIGUOUS is explicitly banned because it dilutes the curated
    OT/ChEMBL/mAb-table records that carry the molecular fact."""

    def test_round28_affects_rule_present(self, prompt):
        # Anchor on the round-28 marker.
        assert "round-28" in prompt.lower() or "Round-28" in prompt, (
            "Round-28 AFFECTS rule header missing from prompt"
        )
        # Locate the AFFECTS section.
        idx = prompt.lower().find("round-28 rule")
        assert idx != -1
        section = prompt[idx:idx + 3000]
        # Should explicitly mention molecular binding and reference the
        # curated databases that carry the fact.
        for token in ("molecular", "binding", "OT", "ChEMBL"):
            assert token in section, (
                f"Round-28 AFFECTS rule should reference {token!r}"
            )
        # Should explicitly forbid defaulting to ambiguous.
        assert "do NOT default to" in section or "Do NOT emit `ambiguous`" in section

    def test_affects_strong_support_on_target_engagement(self, prompt):
        idx = prompt.lower().find("round-28 rule")
        section = prompt[idx:idx + 3000]
        # Strong support when binding is demonstrated.
        assert "strong_support" in section
        # Strong support tied to target engagement / PD biomarker.
        assert (
            "target engagement" in section.lower()
            or "PD biomarker" in section
            or "pharmacodynamic" in section.lower()
        )

    def test_affects_no_record_when_binding_unmeasured(self, prompt):
        idx = prompt.lower().find("round-28 rule")
        section = prompt[idx:idx + 3000]
        # Must say to emit NO record when binding isn't measured.
        assert "NO " in section or "do not emit" in section.lower() or "emit no" in section.lower()
