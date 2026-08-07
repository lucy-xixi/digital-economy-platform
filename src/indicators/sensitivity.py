"""
DEI weight sensitivity analysis.

Perturbs pillar weights (±delta, renormalized) and measures rank stability —
demonstrates that index construction is a modelled choice, not a black box.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import DEI_WEIGHTS, setup_logging
from src.indicators.indicator_framework import IndicatorFramework
from src.indicators.index_calculator import IndexCalculator

logger = setup_logging(__name__)


def _renormalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {k: v / total for k, v in weights.items()}


def weight_sensitivity(
    clean_df: pd.DataFrame,
    base_weights: Optional[Dict[str, float]] = None,
    delta: float = 0.10,
    year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perturb each pillar weight by ±delta (then renormalize) and compare ranks.

    Args:
        clean_df: Clean metrics panel.
        base_weights: Baseline DEI weights.
        delta: Absolute weight shock before renormalization.
        year: Ranking year (default latest).

    Returns:
        Dict with base ranking, scenario rankings, and Spearman rank stability.
    """
    base_weights = dict(base_weights or DEI_WEIGHTS)
    base_calc = IndexCalculator(IndicatorFramework(base_weights))
    _, base_dei = base_calc.calculate(clean_df)
    if year is None:
        year = int(base_dei["year"].max())

    base_rank = (
        base_dei[base_dei["year"] == year][["region", "dei"]]
        .sort_values("dei", ascending=False)
        .reset_index(drop=True)
    )
    base_rank["rank"] = base_rank.index + 1
    base_order = base_rank.set_index("region")["rank"]

    scenarios: List[Dict[str, Any]] = []
    rank_corrs: List[float] = []

    for pillar in base_weights:
        for sign, label in ((+1, "up"), (-1, "down")):
            shocked = dict(base_weights)
            shocked[pillar] = max(0.01, shocked[pillar] + sign * delta)
            shocked = _renormalize(shocked)
            calc = IndexCalculator(IndicatorFramework(shocked))
            _, dei = calc.calculate(clean_df)
            ranking = (
                dei[dei["year"] == year][["region", "dei"]]
                .sort_values("dei", ascending=False)
                .reset_index(drop=True)
            )
            ranking["rank"] = ranking.index + 1
            scen_order = ranking.set_index("region")["rank"]
            # Spearman rank correlation vs base
            aligned = pd.concat([base_order, scen_order], axis=1, keys=["base", "scen"]).dropna()
            rho = float(aligned["base"].corr(aligned["scen"], method="spearman"))
            rank_corrs.append(rho)
            scenarios.append(
                {
                    "scenario": f"{pillar}_{label}",
                    "weights": {k: round(v, 4) for k, v in shocked.items()},
                    "top_region": ranking.iloc[0]["region"],
                    "rank_spearman_vs_base": round(rho, 4),
                    "ranking": ranking.to_dict(orient="records"),
                }
            )

    summary = {
        "year": year,
        "delta": delta,
        "base_weights": base_weights,
        "base_ranking": base_rank.to_dict(orient="records"),
        "mean_rank_stability": round(float(sum(rank_corrs) / len(rank_corrs)), 4) if rank_corrs else None,
        "scenarios": scenarios,
    }
    logger.info(
        "Weight sensitivity year=%d: mean rank Spearman=%.3f across %d scenarios",
        year,
        summary["mean_rank_stability"] or 0,
        len(scenarios),
    )
    return summary
