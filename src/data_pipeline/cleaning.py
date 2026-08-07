"""
Data cleaning module.

Handles missing-value imputation, outlier capping (IQR), and type coercion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import METRIC_COLUMNS, PROCESSED_CSV_PATH, ensure_directories, setup_logging

logger = setup_logging(__name__)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce year to int and metric columns to float.

    Args:
        df: Input DataFrame.

    Returns:
        Type-coerced copy.
    """
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["region"] = out["region"].astype(str).str.strip()
    for col in METRIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def impute_missing(
    df: pd.DataFrame,
    strategy: str = "median_by_region",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Impute missing metric values.

    Strategies:
        - ``median_by_region``: fill with regional median (fallback: global median)
        - ``ffill``: forward-fill within region sorted by year
        - ``median``: global median

    Args:
        df: Input DataFrame.
        strategy: Imputation strategy name.

    Returns:
        Tuple of (cleaned DataFrame, dict of column → imputed count).
    """
    out = df.copy()
    imputed: Dict[str, int] = {}

    for col in METRIC_COLUMNS:
        if col not in out.columns:
            continue
        n_before = int(out[col].isna().sum())
        if n_before == 0:
            imputed[col] = 0
            continue

        if strategy == "median_by_region":
            out[col] = out.groupby("region")[col].transform(
                lambda s: s.fillna(s.median())
            )
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
        elif strategy == "ffill":
            out = out.sort_values(["region", "year"])
            out[col] = out.groupby("region")[col].ffill().bfill()
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].fillna(out[col].median())

        imputed[col] = n_before
        logger.info("Imputed %d missing values in '%s' via %s", n_before, col, strategy)

    return out, imputed


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag IQR-based outliers.

    Args:
        df: Input DataFrame.
        columns: Metric columns to check; defaults to ``METRIC_COLUMNS``.
        multiplier: IQR fence multiplier.

    Returns:
        Boolean DataFrame of the same shape as selected columns (True = outlier).
    """
    columns = columns or METRIC_COLUMNS
    flags = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        flags[col] = (df[col] < lower) | (df[col] > upper)
        n_out = int(flags[col].sum())
        if n_out:
            logger.warning(
                "Detected %d outliers in '%s' (fence=[%.2f, %.2f])",
                n_out,
                col,
                lower,
                upper,
            )

    return flags


def cap_outliers(
    df: pd.DataFrame,
    flags: Optional[pd.DataFrame] = None,
    multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, int]:
    """
    Winsorize outliers to IQR fences.

    Args:
        df: Input DataFrame.
        flags: Optional precomputed outlier flags.
        multiplier: IQR fence multiplier.

    Returns:
        Tuple of (capped DataFrame, number of values capped).
    """
    out = df.copy()
    if flags is None:
        flags = detect_outliers_iqr(out, multiplier=multiplier)

    capped = 0
    for col in flags.columns:
        if col not in out.columns:
            continue
        series = out[col]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = flags[col].fillna(False)
        n = int(mask.sum())
        if n == 0:
            continue
        out.loc[mask & (series < lower), col] = lower
        out.loc[mask & (series > upper), col] = upper
        capped += n
        logger.info("Capped %d outliers in '%s'", n, col)

    return out, capped


def clean(
    df: pd.DataFrame,
    impute_strategy: str = "median_by_region",
    cap: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline.

    Args:
        df: Raw DataFrame.
        impute_strategy: Missing-value strategy.
        cap: Whether to winsorize outliers.

    Returns:
        Tuple of (clean DataFrame, cleaning metadata dict).
    """
    try:
        typed = coerce_types(df)
        # Drop rows without year/region
        typed = typed.dropna(subset=["year", "region"])
        typed["year"] = typed["year"].astype(int)

        imputed_df, imputed_counts = impute_missing(typed, strategy=impute_strategy)
        flags = detect_outliers_iqr(imputed_df)
        outlier_count = int(flags.to_numpy().sum())

        if cap:
            clean_df, capped_count = cap_outliers(imputed_df, flags=flags)
        else:
            clean_df, capped_count = imputed_df, 0

        # Clip domain ranges
        if "internet_penetration" in clean_df.columns:
            clean_df["internet_penetration"] = clean_df["internet_penetration"].clip(0, 100)
        if "digital_payment" in clean_df.columns:
            clean_df["digital_payment"] = clean_df["digital_payment"].clip(0, 100)
        for col in ("5g_users", "software_revenue", "ai_patents", "ecommerce"):
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].clip(lower=0)

        meta = {
            "imputed_counts": imputed_counts,
            "outlier_count": outlier_count,
            "capped_count": capped_count,
            "rows_in": len(df),
            "rows_out": len(clean_df),
        }
        logger.info(
            "Cleaning complete: %d → %d rows, %d outliers flagged, %d capped",
            meta["rows_in"],
            meta["rows_out"],
            outlier_count,
            capped_count,
        )
        return clean_df.reset_index(drop=True), meta

    except Exception as exc:
        logger.exception("Cleaning failed: %s", exc)
        raise


def save_processed(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save cleaned DataFrame to processed CSV."""
    ensure_directories()
    path = path or PROCESSED_CSV_PATH
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Processed data saved to %s", path)
    return path
