"""
Analytics tools exposed to the research agent.

Each tool is a pure function over in-memory / on-disk DEIP artifacts so the
agent can ground claims without stuffing the entire context into one prompt.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.analytics.correlation import correlate_ai_vs_dei
from src.analytics.diagnostics import diagnose_region_trend, diagnostics_summary
from src.analytics.forecasting import (
    forecast_dei_linear,
    forecast_summary,
    walk_forward_backtest,
)
from src.analytics.trend_analysis import trend_summary
from src.config import DEI_CSV_PATH, METHODOLOGY_PATH, PROCESSED_CSV_PATH, setup_logging
from src.indicators.sensitivity import weight_sensitivity

logger = setup_logging(__name__)


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(PROCESSED_CSV_PATH)
    dei = pd.read_csv(DEI_CSV_PATH)
    return metrics, dei


def tool_get_dei_trend() -> Dict[str, Any]:
    """Return DEI trend / CAGR summary."""
    _, dei = _load_frames()
    return trend_summary(dei)


def tool_get_ai_correlation() -> Dict[str, Any]:
    """Return pooled + within AI-patent vs DEI correlation."""
    metrics, dei = _load_frames()
    result = correlate_ai_vs_dei(metrics, dei)
    # Drop bulky scatter for tool responses
    result = {k: v for k, v in result.items() if k != "scatter_data"}
    return result


def tool_get_forecast() -> Dict[str, Any]:
    """Return OLS forecast summary with backtest MAE."""
    _, dei = _load_frames()
    fcast = forecast_dei_linear(dei)
    bt = walk_forward_backtest(dei)
    return forecast_summary(fcast, bt)


def tool_get_diagnostics() -> Dict[str, Any]:
    """Return residual diagnostics for regional DEI trends."""
    _, dei = _load_frames()
    return diagnostics_summary(diagnose_region_trend(dei))


def tool_get_weight_sensitivity() -> Dict[str, Any]:
    """Return DEI weight perturbation rank-stability summary."""
    metrics, _ = _load_frames()
    result = weight_sensitivity(metrics)
    # Compact: drop per-scenario full rankings
    compact_scenarios = [
        {
            "scenario": s["scenario"],
            "top_region": s["top_region"],
            "rank_spearman_vs_base": s["rank_spearman_vs_base"],
        }
        for s in result["scenarios"]
    ]
    return {
        "year": result["year"],
        "mean_rank_stability": result["mean_rank_stability"],
        "base_ranking": result["base_ranking"],
        "scenarios": compact_scenarios,
    }


def tool_retrieve_methodology(query: str = "") -> Dict[str, Any]:
    """
    Lightweight RAG: keyword retrieval over methodology markdown.

    Not a vector DB — intentionally simple, auditable, and offline.
    """
    if not METHODOLOGY_PATH.exists():
        return {"chunks": [], "note": "methodology.md missing"}
    text = METHODOLOGY_PATH.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    tokens = [t.lower() for t in query.replace("，", " ").replace("。", " ").split() if t]
    if not tokens:
        chunks = paragraphs[:3]
    else:
        scored = []
        for p in paragraphs:
            score = sum(1 for t in tokens if t in p.lower())
            if score:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [p for _, p in scored[:3]] or paragraphs[:2]
    logger.info("Methodology retrieval: %d chunks for query=%r", len(chunks), query[:60])
    return {"chunks": chunks, "source": str(METHODOLOGY_PATH.name)}


def tool_get_causal_effect() -> Dict[str, Any]:
    """Return AI Innovation → DEI causal effect estimate + CI."""
    _, dei = _load_frames()
    from src.causal_analysis.effect_estimator import estimate_ai_effect_on_dei

    result = estimate_ai_effect_on_dei(dei, n_boot=200, prefer_engine="dowhy")
    # Compact for prompt injection
    return {
        "primary": result.get("primary"),
        "significant_at_alpha": result.get("significant_at_alpha"),
        "interpretation": result.get("interpretation"),
        "identification": result.get("identification"),
    }


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_dei_trend": tool_get_dei_trend,
    "get_ai_correlation": tool_get_ai_correlation,
    "get_forecast": tool_get_forecast,
    "get_diagnostics": tool_get_diagnostics,
    "get_weight_sensitivity": tool_get_weight_sensitivity,
    "get_causal_effect": tool_get_causal_effect,
    "retrieve_methodology": tool_retrieve_methodology,
}


def run_tools(tool_names: Optional[List[str]] = None, query: str = "") -> Dict[str, Any]:
    """
    Execute a set of analytics tools and return a grounded context dict.

    Args:
        tool_names: Subset of TOOL_REGISTRY keys; default = all analytic tools.
        query: Passed to methodology retrieval.

    Returns:
        Mapping tool_name → tool result.
    """
    names = tool_names or [
        "get_dei_trend",
        "get_ai_correlation",
        "get_forecast",
        "get_diagnostics",
        "get_weight_sensitivity",
        "get_causal_effect",
        "retrieve_methodology",
    ]
    context: Dict[str, Any] = {}
    for name in names:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown tool: %s", name)
            continue
        try:
            if name == "retrieve_methodology":
                context[name] = fn(query)
            else:
                context[name] = fn()
        except Exception as exc:
            logger.exception("Tool %s failed: %s", name, exc)
            context[name] = {"error": str(exc)}
    return context
