"""
Trend analysis module.

Computes DEI time-series summaries, CAGR, and regional ranking trajectories
for 2018–2025.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import setup_logging

logger = setup_logging(__name__)


def compute_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Compound Annual Growth Rate.

    Args:
        start_value: Value at t0.
        end_value: Value at tN.
        periods: Number of years (N).

    Returns:
        CAGR as a percentage. Returns NaN if inputs are invalid.
    """
    if periods <= 0 or start_value <= 0 or end_value <= 0 or np.isnan(start_value) or np.isnan(end_value):
        return float("nan")
    return (float(end_value) / float(start_value)) ** (1.0 / periods) - 1.0


def dei_trend_by_region(dei_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot DEI to wide format: index=year, columns=region.

    Args:
        dei_df: DataFrame with year, region, dei.

    Returns:
        Wide DataFrame suitable for line charts.
    """
    required = {"year", "region", "dei"}
    if not required.issubset(dei_df.columns):
        raise ValueError(f"dei_df missing columns: {required - set(dei_df.columns)}")

    wide = (
        dei_df.pivot_table(index="year", columns="region", values="dei", aggfunc="mean")
        .sort_index()
    )
    logger.info("Built DEI trend matrix: %s", wide.shape)
    return wide


def national_dei_trend(dei_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average DEI across regions by year (national proxy).

    Returns:
        DataFrame with year, dei_mean, dei_std, dei_min, dei_max.
    """
    agg = (
        dei_df.groupby("year")["dei"]
        .agg(dei_mean="mean", dei_std="std", dei_min="min", dei_max="max")
        .reset_index()
    )
    return agg


def region_ranking(dei_df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """
    Rank regions by DEI for a given year (default: latest).

    Args:
        dei_df: DEI DataFrame.
        year: Target year; None → max year in data.

    Returns:
        Sorted DataFrame with region, dei, rank, growth_rate.
    """
    if year is None:
        year = int(dei_df["year"].max())
    subset = dei_df[dei_df["year"] == year].copy()
    subset = subset.sort_values("dei", ascending=False).reset_index(drop=True)
    subset["rank"] = subset.index + 1
    cols = [c for c in ["rank", "region", "dei", "growth_rate"] if c in subset.columns]
    logger.info("Region ranking for %d: top=%s", year, subset.iloc[0]["region"] if len(subset) else "N/A")
    return subset[cols]


def trend_summary(dei_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce a structured trend-analysis summary.

    Returns:
        Dict with CAGR by region, latest ranking, and national trajectory.
    """
    try:
        years = sorted(dei_df["year"].unique())
        if len(years) < 2:
            raise ValueError("Need at least two years for trend analysis")

        y0, y1 = int(years[0]), int(years[-1])
        periods = y1 - y0
        cagr_by_region: Dict[str, float] = {}

        for region, grp in dei_df.groupby("region"):
            g = grp.sort_values("year")
            start = float(g.loc[g["year"] == y0, "dei"].iloc[0])
            end = float(g.loc[g["year"] == y1, "dei"].iloc[0])
            cagr_by_region[str(region)] = round(compute_cagr(start, end, periods) * 100, 2)

        ranking = region_ranking(dei_df, year=y1)
        national = national_dei_trend(dei_df)

        summary = {
            "period": f"{y0}-{y1}",
            "cagr_by_region_pct": cagr_by_region,
            "latest_year": y1,
            "top_region": ranking.iloc[0]["region"] if len(ranking) else None,
            "top_dei": float(ranking.iloc[0]["dei"]) if len(ranking) else None,
            "national_trend": national.to_dict(orient="records"),
        }
        logger.info(
            "Trend summary %s: top region=%s (DEI=%.2f)",
            summary["period"],
            summary["top_region"],
            summary["top_dei"] or 0,
        )
        return summary

    except Exception as exc:
        logger.exception("Trend analysis failed: %s", exc)
        raise
