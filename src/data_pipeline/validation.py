"""
Data validation & quality reporting module.

Produces a structured Data Quality Report covering completeness, uniqueness,
outlier rates, and an overall quality score.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import (
    DQ_BLOCK_ON_FAIL,
    DQ_MAX_MISSING_RATE,
    DQ_MIN_SCORE,
    DQ_REQUIRE_UNIQUE_KEYS,
    METRIC_COLUMNS,
    QUALITY_REPORT_PATH,
    REGIONS,
    ensure_directories,
    setup_logging,
)
from src.data_pipeline.cleaning import detect_outliers_iqr

logger = setup_logging(__name__)


class QualityGateError(RuntimeError):
    """Raised when a data-quality gate blocks the pipeline."""


def _completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-column and overall completeness statistics."""
    total_cells = len(df) * len(METRIC_COLUMNS)
    missing_by_col = {c: int(df[c].isna().sum()) for c in METRIC_COLUMNS if c in df.columns}
    missing_total = sum(missing_by_col.values())
    completeness_rate = 1.0 - (missing_total / total_cells) if total_cells else 1.0
    return {
        "missing_by_column": missing_by_col,
        "missing_total": missing_total,
        "completeness_rate": round(completeness_rate, 4),
    }


def _uniqueness(df: pd.DataFrame) -> Dict[str, Any]:
    """Check uniqueness of (year, region) keys."""
    if not {"year", "region"}.issubset(df.columns):
        return {"duplicate_keys": 0, "is_unique": False}
    dup = int(df.duplicated(subset=["year", "region"]).sum())
    return {"duplicate_keys": dup, "is_unique": dup == 0}


def _coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate expected region × year panel coverage."""
    regions_found = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    years_found = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    missing_regions = sorted(set(REGIONS) - set(regions_found))
    return {
        "regions_found": regions_found,
        "years_found": years_found,
        "missing_regions": missing_regions,
        "expected_rows": len(REGIONS) * len(years_found) if years_found else 0,
        "actual_rows": len(df),
    }


def _outlier_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize IQR outliers per metric column."""
    flags = detect_outliers_iqr(df)
    by_col = {c: int(flags[c].sum()) for c in flags.columns}
    return {
        "outlier_by_column": by_col,
        "outlier_total": int(sum(by_col.values())),
    }


def compute_quality_score(
    completeness_rate: float,
    duplicate_keys: int,
    outlier_total: int,
    n_rows: int,
) -> float:
    """
    Derive a 0–100 data quality score.

    Weighting: completeness 50%, uniqueness 25%, outlier ratio 25%.
    """
    uniqueness_score = 1.0 if duplicate_keys == 0 else max(0.0, 1.0 - duplicate_keys / max(n_rows, 1))
    outlier_ratio = outlier_total / max(n_rows * len(METRIC_COLUMNS), 1)
    outlier_score = max(0.0, 1.0 - outlier_ratio * 5)  # penalize heavily if >20% outliers
    score = 100.0 * (
        0.50 * completeness_rate + 0.25 * uniqueness_score + 0.25 * outlier_score
    )
    return round(score, 2)


def validate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run full validation and return a Data Quality Report dict.

    Args:
        df: DataFrame to validate (preferably raw, before heavy cleaning).

    Returns:
        Structured quality report dictionary.
    """
    try:
        completeness = _completeness(df)
        uniqueness = _uniqueness(df)
        coverage = _coverage(df)
        outliers = _outlier_summary(df.dropna(subset=METRIC_COLUMNS, how="all"))

        quality_score = compute_quality_score(
            completeness_rate=completeness["completeness_rate"],
            duplicate_keys=uniqueness["duplicate_keys"],
            outlier_total=outliers["outlier_total"],
            n_rows=len(df),
        )

        status = "OK" if quality_score >= 80 else ("WARN" if quality_score >= 60 else "FAIL")

        report: Dict[str, Any] = {
            "report_ts": datetime.now(timezone.utc).isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "completeness": completeness,
            "uniqueness": uniqueness,
            "coverage": coverage,
            "outliers": outliers,
            "quality_score": quality_score,
            "status": status,
            "summary": (
                f"Data Quality Report: score={quality_score}/100 ({status}); "
                f"rows={len(df)}; missing={completeness['missing_total']}; "
                f"outliers={outliers['outlier_total']}; "
                f"duplicates={uniqueness['duplicate_keys']}"
            ),
        }
        logger.info(report["summary"])
        return report

    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        raise


def save_quality_report(report: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """
    Persist Data Quality Report as JSON.

    Args:
        report: Report dictionary.
        path: Output path; defaults to ``QUALITY_REPORT_PATH``.

    Returns:
        Path written.
    """
    ensure_directories()
    path = path or QUALITY_REPORT_PATH
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Data Quality Report saved to %s", path)
    return path


def print_quality_report(report: Dict[str, Any]) -> None:
    """Pretty-print the Data Quality Report to the logger / stdout."""
    lines = [
        "=" * 60,
        "  DATA QUALITY REPORT — Digital Economy Intelligence Platform",
        "=" * 60,
        f"  Timestamp      : {report.get('report_ts')}",
        f"  Total Rows     : {report.get('total_rows')}",
        f"  Quality Score  : {report.get('quality_score')}/100",
        f"  Status         : {report.get('status')}",
        f"  Gate Passed    : {report.get('gate_passed')}",
        "-" * 60,
        f"  Missing Cells  : {report.get('completeness', {}).get('missing_total')}",
        f"  Completeness   : {report.get('completeness', {}).get('completeness_rate')}",
        f"  Duplicate Keys : {report.get('uniqueness', {}).get('duplicate_keys')}",
        f"  Outliers       : {report.get('outliers', {}).get('outlier_total')}",
        f"  Regions        : {report.get('coverage', {}).get('regions_found')}",
        f"  Years          : {report.get('coverage', {}).get('years_found')}",
        "=" * 60,
    ]
    text = "\n".join(lines)
    logger.info("\n%s", text)
    print(text)


def evaluate_quality_gate(
    report: Dict[str, Any],
    min_score: float = DQ_MIN_SCORE,
    max_missing_rate: float = DQ_MAX_MISSING_RATE,
    require_unique: bool = DQ_REQUIRE_UNIQUE_KEYS,
) -> Dict[str, Any]:
    """
    Evaluate hard quality-gate rules against a DQ report.

    Returns:
        Dict with ``passed`` bool and ``reasons`` list (empty if passed).
    """
    reasons: list[str] = []
    score = float(report.get("quality_score", 0))
    missing_rate = 1.0 - float(report.get("completeness", {}).get("completeness_rate", 1.0))
    dup = int(report.get("uniqueness", {}).get("duplicate_keys", 0))
    missing_regions = report.get("coverage", {}).get("missing_regions") or []

    if score < min_score:
        reasons.append(f"quality_score {score} < min_score {min_score}")
    if missing_rate > max_missing_rate:
        reasons.append(f"missing_rate {missing_rate:.3f} > max {max_missing_rate}")
    if require_unique and dup > 0:
        reasons.append(f"duplicate keys={dup}")
    if missing_regions:
        reasons.append(f"missing regions={missing_regions}")

    passed = len(reasons) == 0
    report["gate_passed"] = passed
    report["gate_reasons"] = reasons
    logger.info("Quality gate: passed=%s reasons=%s", passed, reasons or "none")
    return {"passed": passed, "reasons": reasons}


def enforce_quality_gate(
    report: Dict[str, Any],
    block: bool = DQ_BLOCK_ON_FAIL,
) -> Dict[str, Any]:
    """
    Enforce the quality gate. Raises ``QualityGateError`` when blocked.

    Args:
        report: DQ report (mutated with gate fields).
        block: If False, only warn (useful for exploratory runs).

    Returns:
        Gate evaluation dict.
    """
    gate = evaluate_quality_gate(report)
    if not gate["passed"]:
        msg = "Quality gate FAILED: " + "; ".join(gate["reasons"])
        if block:
            logger.error(msg)
            raise QualityGateError(msg)
        logger.warning("%s (block disabled — continuing)", msg)
    return gate
