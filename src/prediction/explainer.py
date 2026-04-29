"""Natural language explanation of prediction results via Claude."""

from __future__ import annotations

import anthropic

from src.prediction.path_query import PredictionResult

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

_SYSTEM_PROMPT = """\
You are a drug development scientist explaining a computational prediction of clinical trial success.
Given a structured prediction result, produce a clear 2-3 paragraph explanation covering:
1. The overall probability and what it means in context (typical Phase 3 success rate is ~25-30%).
2. Which edges in the causal chain are strongest and weakest, and what that implies.
3. Specific actionable recommendations based on the weakest links.
Be precise with numbers. Use accessible language. Do not hedge excessively."""


async def explain(
    result: PredictionResult,
    client: anthropic.AsyncAnthropic,
) -> str:
    """Send a PredictionResult to Claude for natural language explanation."""
    edge_lines = []
    for ec in result.edge_contributions:
        marker = " ← WEAKEST" if result.weakest_link and ec.edge_type == result.weakest_link.edge_type and ec.source_id == result.weakest_link.source_id else ""
        edge_lines.append(
            f"  {ec.edge_type.value}: {ec.source_id} → {ec.target_id} | "
            f"P={ec.belief.expected_probability:.3f} | "
            f"Beta({ec.belief.alpha:.1f}, {ec.belief.beta:.1f}) | "
            f"evidence_strength={ec.belief.evidence_strength:.1f} | "
            f"bottleneck={ec.bottleneck_score:.3f}{marker}"
        )

    user_message = (
        f"Hypothesis: {result.trial_hypothesis}\n\n"
        f"Overall P(success) = {result.overall_probability:.4f}\n"
        f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]\n"
        f"Samples: {result.n_samples:,}\n\n"
        f"Edge contributions (causal chain):\n"
        + "\n".join(edge_lines)
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()
