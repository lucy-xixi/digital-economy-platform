"""
Correlation analysis with pooled and panel (within) estimators.

Pooled Pearson on region×year rows overstates significance because observations
within a region are dependent. The within estimator demeans by region first —
closer to a fixed-effects intuition for interview discussions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import setup_logging

logger = setup_logging(__name__)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r with p-value; NaN-safe."""
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _within_demean(df: pd.DataFrame, cols: List[str], group: str = "region") -> pd.DataFrame:
    """Subtract group means (fixed-effects demeaning)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(group)[c].transform("mean")
    return out


def correlate_ai_vs_dei(
    metrics_df: pd.DataFrame,
    dei_df: pd.DataFrame,
    year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Correlate ``ai_patents`` with DEI (pooled + within-region).

    Args:
        metrics_df: Clean metrics (must include ai_patents, year, region).
        dei_df: DEI index (year, region, dei).
        year: Optional cross-section year filter.

    Returns:
        Dict with pooled / within correlations, interpretation, scatter data.
    """
    try:
        merged = metrics_df[["year", "region", "ai_patents"]].merge(
            dei_df[["year", "region", "dei"]],
            on=["year", "region"],
            how="inner",
        )
        if year is not None:
            merged = merged[merged["year"] == year]

        merged = merged.dropna(subset=["ai_patents", "dei"])
        if len(merged) < 3:
            raise ValueError("Need at least 3 observations for correlation analysis")

        x = merged["ai_patents"].to_numpy(dtype=float)
        y = merged["dei"].to_numpy(dtype=float)
        pearson_r, pearson_p = _pearson(x, y)

        try:
            spearman_rho, spearman_p = stats.spearmanr(x, y)
            spearman_rho, spearman_p = float(spearman_rho), float(spearman_p)
        except Exception:
            spearman_rho, spearman_p = float("nan"), float("nan")

        # Panel within correlation (region demeaned)
        within = _within_demean(merged, ["ai_patents", "dei"], group="region")
        wr, wp = _pearson(
            within["ai_patents"].to_numpy(dtype=float),
            within["dei"].to_numpy(dtype=float),
        )

        result = {
            "n_obs": int(len(merged)),
            "year_filter": year,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": pearson_p,
            "spearman_rho": round(spearman_rho, 4) if not np.isnan(spearman_rho) else None,
            "spearman_p": spearman_p,
            "within_pearson_r": round(wr, 4) if not np.isnan(wr) else None,
            "within_pearson_p": wp,
            "note": (
                "Pooled r treats all region-years as i.i.d.; within r demeans by region "
                "(fixed-effects style) and is more appropriate for panel dependence."
            ),
            "interpretation": _interpret(pearson_r, pearson_p, wr),
            "scatter_data": merged.to_dict(orient="records"),
        }
        logger.info(
            "AI vs DEI: pooled r=%.4f (p=%.3g); within r=%.4f (p=%.3g); n=%d",
            pearson_r,
            pearson_p,
            wr,
            wp,
            result["n_obs"],
        )
        return result

    except Exception as exc:
        logger.exception("Correlation analysis failed: %s", exc)
        raise


def _interpret(r: float, p: float, within_r: float) -> str:
    """Human-readable correlation interpretation."""
    if np.isnan(r):
        return "Insufficient data to estimate correlation."
    strength = (
        "very strong"
        if abs(r) >= 0.8
        else "strong"
        if abs(r) >= 0.6
        else "moderate"
        if abs(r) >= 0.4
        else "weak"
        if abs(r) >= 0.2
        else "negligible"
    )
    direction = "positive" if r >= 0 else "negative"
    sig = (
        "statistically significant"
        if (not np.isnan(p) and p < 0.05)
        else "not statistically significant"
    )
    within_txt = (
        f" Within-region r={within_r:.3f} "
        f"({'same sign' if np.sign(within_r) == np.sign(r) else 'sign differs'} vs pooled)."
        if not np.isnan(within_r)
        else ""
    )
    return (
        f"There is a {strength} {direction} pooled correlation between AI patents and DEI "
        f"(r={r:.3f}, {sig} at α=0.05).{within_txt}"
    )


def correlation_matrix(
    metrics_df: pd.DataFrame,
    dei_df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a Pearson correlation matrix among selected metrics + DEI."""
    default_cols = [
        "internet_penetration",
        "5g_users",
        "software_revenue",
        "ai_patents",
        "digital_payment",
        "ecommerce",
    ]
    columns = columns or default_cols
    merged = metrics_df[["year", "region", *columns]].merge(
        dei_df[["year", "region", "dei"]], on=["year", "region"], how="inner"
    )
    corr = merged[columns + ["dei"]].corr(method="pearson")
    logger.info("Built correlation matrix %s", corr.shape)
    return corr
