"""
Data ingestion module.

Generates realistic simulated digital-economy metrics for five Chinese regions
(2018–2025) and loads them into memory / SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DB_PATH,
    METRIC_COLUMNS,
    RAW_CSV_PATH,
    REGIONS,
    SCHEMA_PATH,
    YEAR_END,
    YEAR_START,
    ensure_directories,
    setup_logging,
)

logger = setup_logging(__name__)

# Region-specific base levels & growth rates (illustrative, not official stats)
_REGION_PROFILES: dict[str, dict[str, float]] = {
    "Beijing": {
        "internet_base": 78.0,
        "internet_growth": 1.8,
        "5g_base": 120.0,
        "5g_growth": 55.0,
        "software_base": 850.0,
        "software_growth": 95.0,
        "ai_base": 3200.0,
        "ai_growth": 480.0,
        "payment_base": 72.0,
        "payment_growth": 2.5,
        "ecommerce_base": 420.0,
        "ecommerce_growth": 55.0,
    },
    "Shanghai": {
        "internet_base": 76.0,
        "internet_growth": 1.9,
        "5g_base": 110.0,
        "5g_growth": 52.0,
        "software_base": 780.0,
        "software_growth": 88.0,
        "ai_base": 2800.0,
        "ai_growth": 420.0,
        "payment_base": 74.0,
        "payment_growth": 2.4,
        "ecommerce_base": 500.0,
        "ecommerce_growth": 62.0,
    },
    "Guangdong": {
        "internet_base": 72.0,
        "internet_growth": 2.2,
        "5g_base": 200.0,
        "5g_growth": 80.0,
        "software_base": 1200.0,
        "software_growth": 140.0,
        "ai_base": 3500.0,
        "ai_growth": 520.0,
        "payment_base": 70.0,
        "payment_growth": 2.8,
        "ecommerce_base": 900.0,
        "ecommerce_growth": 110.0,
    },
    "Zhejiang": {
        "internet_base": 74.0,
        "internet_growth": 2.0,
        "5g_base": 95.0,
        "5g_growth": 48.0,
        "software_base": 620.0,
        "software_growth": 75.0,
        "ai_base": 1800.0,
        "ai_growth": 310.0,
        "payment_base": 78.0,
        "payment_growth": 2.2,
        "ecommerce_base": 750.0,
        "ecommerce_growth": 95.0,
    },
    "Jiangsu": {
        "internet_base": 70.0,
        "internet_growth": 2.3,
        "5g_base": 130.0,
        "5g_growth": 58.0,
        "software_base": 700.0,
        "software_growth": 82.0,
        "ai_base": 2100.0,
        "ai_growth": 350.0,
        "payment_base": 68.0,
        "payment_growth": 2.6,
        "ecommerce_base": 580.0,
        "ecommerce_growth": 72.0,
    },
}


def generate_simulated_data(
    seed: int = 42,
    inject_missing: bool = True,
    inject_outliers: bool = True,
) -> pd.DataFrame:
    """
    Generate simulated digital-economy panel data.

    Args:
        seed: Random seed for reproducibility.
        inject_missing: If True, randomly set ~3% of metric cells to NaN.
        inject_outliers: If True, inject a few extreme values for QA testing.

    Returns:
        DataFrame with columns: year, region, and all metric columns.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for region in REGIONS:
        profile = _REGION_PROFILES[region]
        for year in range(YEAR_START, YEAR_END + 1):
            t = year - YEAR_START
            noise = lambda scale: float(rng.normal(0, scale))  # noqa: E731
            row = {
                "year": year,
                "region": region,
                "internet_penetration": min(
                    99.5,
                    profile["internet_base"]
                    + profile["internet_growth"] * t
                    + noise(0.4),
                ),
                "5g_users": max(
                    0.0,
                    profile["5g_base"] + profile["5g_growth"] * t + noise(8.0),
                ),
                "software_revenue": max(
                    0.0,
                    profile["software_base"]
                    + profile["software_growth"] * t
                    + noise(20.0),
                ),
                "ai_patents": max(
                    0.0,
                    profile["ai_base"] + profile["ai_growth"] * t + noise(80.0),
                ),
                "digital_payment": min(
                    99.0,
                    profile["payment_base"]
                    + profile["payment_growth"] * t
                    + noise(0.5),
                ),
                "ecommerce": max(
                    0.0,
                    profile["ecommerce_base"]
                    + profile["ecommerce_growth"] * t
                    + noise(15.0),
                ),
            }
            records.append(row)

    df = pd.DataFrame(records)

    if inject_missing and len(df) > 0:
        n_cells = len(df) * len(METRIC_COLUMNS)
        n_missing = max(1, int(n_cells * 0.03))
        for _ in range(n_missing):
            r = int(rng.integers(0, len(df)))
            c = METRIC_COLUMNS[int(rng.integers(0, len(METRIC_COLUMNS)))]
            df.at[r, c] = np.nan

    if inject_outliers and len(df) > 0:
        # Extreme AI patents spike and negative-looking ecommerce (will be clipped later)
        oi = int(rng.integers(0, len(df)))
        df.at[oi, "ai_patents"] = float(df.at[oi, "ai_patents"]) * 8.0
        oj = int(rng.integers(0, len(df)))
        df.at[oj, "ecommerce"] = float(df.at[oj, "ecommerce"]) * 6.0

    logger.info(
        "Generated simulated data: %d rows × %d cols (%d regions, %d–%d)",
        len(df),
        len(df.columns),
        len(REGIONS),
        YEAR_START,
        YEAR_END,
    )
    return df


def save_raw_csv(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """
    Persist raw DataFrame to CSV.

    Args:
        df: Raw metrics DataFrame.
        path: Destination path; defaults to ``RAW_CSV_PATH``.

    Returns:
        Path written.
    """
    ensure_directories()
    path = path or RAW_CSV_PATH
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Raw data saved to %s", path)
    return path


def read_raw_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read raw digital-economy metrics from CSV.

    Args:
        path: Source CSV path; defaults to ``RAW_CSV_PATH``.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = path or RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    required = {"year", "region", *METRIC_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV missing required columns: {sorted(missing)}")

    logger.info("Loaded raw data from %s (%d rows)", path, len(df))
    return df


def init_database(db_path: Optional[Path] = None) -> Path:
    """
    Create SQLite database and apply schema.

    Args:
        db_path: Database file path; defaults to ``DB_PATH``.

    Returns:
        Path to the database file.
    """
    ensure_directories()
    db_path = db_path or DB_PATH
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_sql)
        logger.info("Database initialized at %s", db_path)
    except sqlite3.Error as exc:
        logger.exception("Failed to initialize database: %s", exc)
        raise

    return db_path


def load_to_sqlite(df: pd.DataFrame, db_path: Optional[Path] = None) -> int:
    """
    Upsert metrics DataFrame into ``digital_metrics`` table.

    Args:
        df: Metrics DataFrame (cleaned or raw).
        db_path: Database path.

    Returns:
        Number of rows written.
    """
    db_path = db_path or DB_PATH
    if not db_path.exists():
        init_database(db_path)

    # Map column name for SQL schema (5g_users → five_g_users)
    upload = df.rename(columns={"5g_users": "five_g_users"}).copy()
    cols = [
        "year",
        "region",
        "internet_penetration",
        "five_g_users",
        "software_revenue",
        "ai_patents",
        "digital_payment",
        "ecommerce",
    ]
    upload = upload[cols]

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM digital_metrics")
            upload.to_sql("digital_metrics", conn, if_exists="append", index=False)
        logger.info("Loaded %d rows into digital_metrics", len(upload))
    except (sqlite3.Error, ValueError) as exc:
        logger.exception("SQLite load failed: %s", exc)
        raise

    return len(upload)


def ingest(
    force_regenerate: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    End-to-end ingestion: generate (if needed), save CSV, init DB, load SQLite.

    Args:
        force_regenerate: Always regenerate simulated data even if CSV exists.
        seed: RNG seed.

    Returns:
        Raw DataFrame.
    """
    ensure_directories()
    if force_regenerate or not RAW_CSV_PATH.exists():
        df = generate_simulated_data(seed=seed)
        save_raw_csv(df)
    else:
        df = read_raw_csv()

    init_database()
    return df
