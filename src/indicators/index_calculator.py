"""
DEI index calculator.

Applies MinMax normalization, aggregates pillar scores, and computes the
composite Digital Economy Index by region and year.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH, DEI_CSV_PATH, METRIC_COLUMNS, ensure_directories, setup_logging
from src.indicators.indicator_framework import IndicatorFramework

logger = setup_logging(__name__)


def minmax_scale(
    series: pd.Series,
    feature_range: tuple[float, float] = (0.0, 100.0),
) -> pd.Series:
    """
    MinMax-scale a Series to ``feature_range``.

    Args:
        series: Numeric series.
        feature_range: Target (min, max) range.

    Returns:
        Scaled series. Constant series maps to the midpoint of the range.
    """
    lo, hi = float(series.min()), float(series.max())
    amin, amax = feature_range
    if np.isclose(hi, lo):
        mid = (amin + amax) / 2.0
        return pd.Series(mid, index=series.index, dtype=float)
    scaled = (series - lo) / (hi - lo) * (amax - amin) + amin
    return scaled.astype(float)


class IndexCalculator:
    """
    Compute primary-indicator scores and the composite DEI.

    Pipeline
    --------
    1. MinMax-normalize each raw metric across the full panel (0–100).
    2. Average metrics within each pillar → pillar score.
    3. Weighted sum of pillar scores → DEI.
    4. Attach YoY growth rate and within-year rank.
    """

    def __init__(self, framework: Optional[IndicatorFramework] = None) -> None:
        self.framework = framework or IndicatorFramework()

    def normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MinMax-normalize all metric columns to [0, 100].

        Args:
            df: Clean metrics DataFrame.

        Returns:
            DataFrame with normalized metric columns (suffix ``_norm``).
        """
        out = df[["year", "region"]].copy()
        for col in METRIC_COLUMNS:
            if col not in df.columns:
                raise KeyError(f"Missing metric column: {col}")
            out[f"{col}_norm"] = minmax_scale(df[col])
        logger.info("Normalized %d metric columns to [0, 100]", len(METRIC_COLUMNS))
        return out

    def compute_primary_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute four primary indicator scores.

        Args:
            df: Clean metrics DataFrame.

        Returns:
            DataFrame with year, region, and four pillar columns.
        """
        norm = self.normalize_metrics(df)
        result = norm[["year", "region"]].copy()

        for pillar_name in self.framework.list_pillars():
            metrics = self.framework.get_metrics(pillar_name)
            norm_cols = [f"{m}_norm" for m in metrics]
            result[pillar_name] = norm[norm_cols].mean(axis=1)

        logger.info("Computed primary indicators for %d rows", len(result))
        return result

    def compute_dei(self, primary: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted aggregation of primary indicators into DEI.

        Args:
            primary: Output of ``compute_primary_indicators``.

        Returns:
            DataFrame with year, region, DEI, growth_rate, rank_in_year.
        """
        weights = self.framework.weights
        out = primary[["year", "region"]].copy()
        out["dei"] = sum(primary[p] * w for p, w in weights.items())

        # YoY growth by region
        out = out.sort_values(["region", "year"]).reset_index(drop=True)
        out["growth_rate"] = out.groupby("region")["dei"].pct_change() * 100.0

        # Rank within each year (1 = highest DEI)
        out["rank_in_year"] = (
            out.groupby("year")["dei"].rank(ascending=False, method="min").astype(int)
        )

        logger.info(
            "DEI computed: mean=%.2f, min=%.2f, max=%.2f",
            out["dei"].mean(),
            out["dei"].min(),
            out["dei"].max(),
        )
        return out

    def calculate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full DEI calculation pipeline.

        Args:
            df: Clean metrics DataFrame.

        Returns:
            Tuple of (primary_indicators DataFrame, dei_index DataFrame).
        """
        try:
            primary = self.compute_primary_indicators(df)
            dei = self.compute_dei(primary)
            # Merge pillars into DEI table for downstream analytics / dashboard
            dei_full = dei.merge(primary, on=["year", "region"], how="left")
            return primary, dei_full
        except Exception as exc:
            logger.exception("DEI calculation failed: %s", exc)
            raise

    def save(
        self,
        primary: pd.DataFrame,
        dei: pd.DataFrame,
        csv_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> Path:
        """
        Persist DEI results to CSV and SQLite.

        Args:
            primary: Primary indicators table.
            dei: DEI index table (may include pillar columns).
            csv_path: Output CSV path.
            db_path: SQLite database path.

        Returns:
            Path to the DEI CSV.
        """
        ensure_directories()
        csv_path = csv_path or DEI_CSV_PATH
        db_path = db_path or DB_PATH

        dei.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("DEI index saved to %s", csv_path)

        dei_cols = ["year", "region", "dei", "growth_rate", "rank_in_year"]
        primary_cols = [
            "year",
            "region",
            "digital_infrastructure",
            "digital_industry",
            "ai_innovation",
            "digital_application",
        ]

        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM primary_indicators")
                conn.execute("DELETE FROM dei_index")
                primary[primary_cols].to_sql(
                    "primary_indicators", conn, if_exists="append", index=False
                )
                dei[dei_cols].to_sql("dei_index", conn, if_exists="append", index=False)
            logger.info("DEI results written to SQLite (%s)", db_path)
        except sqlite3.Error as exc:
            logger.exception("Failed to write DEI to SQLite: %s", exc)
            raise

        return csv_path
