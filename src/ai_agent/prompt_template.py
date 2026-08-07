"""
Prompt templates for the DEIP research assistant.

Templates are designed for LangChain / OpenAI chat completions. They inject
structured analytics context so the LLM can ground its narrative in numbers.
"""

from __future__ import annotations

from typing import Any, Dict

RESEARCH_SYSTEM_PROMPT = """You are a senior digital-economy research analyst at a
government think-tank. You write clear, evidence-based briefing notes for policy
makers and corporate strategy teams. Use precise language, cite the provided
figures, and avoid speculation beyond the data. Structure every answer with:
1) Executive Summary  2) Key Findings  3) Outlook  4) Policy Implications.
Respond in the same language as the user query."""

RESEARCH_USER_TEMPLATE = """# Research Query
{query}

# Analytics Context (ground truth from DEIP)
## DEI Trend Summary
{trend_summary}

## AI Patents vs DEI Correlation
{correlation_summary}

## DEI Forecast (Linear Regression, 2026–2030)
{forecast_summary}

# Instructions
- Base your analysis strictly on the context above.
- Highlight regional leaders and laggards.
- Discuss the role of AI innovation in digital-economy performance.
- Provide actionable recommendations for research institutes and enterprises.
"""

INDICATOR_EXPLAIN_TEMPLATE = """Explain the Digital Economy Index (DEI) framework
used by DEIP:

Pillars and weights:
{framework_description}

Latest regional DEI scores:
{latest_scores}

Answer the user question: {query}
"""


def render_research_prompt(
    query: str,
    trend_summary: Dict[str, Any],
    correlation_summary: Dict[str, Any],
    forecast_summary: Dict[str, Any],
) -> Dict[str, str]:
    """
    Render system + user messages for the research assistant.

    Args:
        query: Natural-language research question.
        trend_summary: Output of ``trend_analysis.trend_summary``.
        correlation_summary: Output of ``correlation.correlate_ai_vs_dei``.
        forecast_summary: Output of ``forecasting.forecast_summary``.

    Returns:
        Dict with keys ``system`` and ``user``.
    """
    return {
        "system": RESEARCH_SYSTEM_PROMPT.strip(),
        "user": RESEARCH_USER_TEMPLATE.format(
            query=query,
            trend_summary=_fmt(trend_summary),
            correlation_summary=_fmt(correlation_summary),
            forecast_summary=_fmt(forecast_summary),
        ),
    }


def _fmt(obj: Any) -> str:
    """Compact pretty-print for prompt injection."""
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
