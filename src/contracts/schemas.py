"""
Data contracts for DEIP pipeline stages.

Lightweight, dependency-light contracts (no Pandera required) that enforce:
column presence, dtypes, key uniqueness, and domain ranges. Failures raise
``ContractViolationError`` so gates can block downstream stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.config import CONTRACT_VERSION, METRIC_COLUMNS, REGIONS, setup_logging

logger = setup_logging(__name__)


class ContractViolationError(ValueError):
    """Raised when a DataFrame violates its declared contract."""


@dataclass
class ContractResult:
    """Outcome of a contract validation."""

    name: str
    version: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0

    def raise_if_failed(self) -> None:
        if not self.passed:
            msg = f"Contract '{self.name}' failed: " + "; ".join(self.errors)
            raise ContractViolationError(msg)


def _check_required_columns(df: pd.DataFrame, required: Sequence[str], errors: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")


def _check_unique_keys(df: pd.DataFrame, keys: Sequence[str], errors: List[str]) -> None:
    if not set(keys).issubset(df.columns):
        return
    n_dup = int(df.duplicated(subset=list(keys)).sum())
    if n_dup:
        errors.append(f"duplicate keys on {list(keys)}: {n_dup}")


def _check_regions(df: pd.DataFrame, errors: List[str], warnings: List[str]) -> None:
    if "region" not in df.columns:
        return
    unknown = sorted(set(df["region"].astype(str)) - set(REGIONS))
    if unknown:
        warnings.append(f"unexpected regions: {unknown}")


def validate_raw_metrics(df: pd.DataFrame) -> ContractResult:
    """
    Contract for raw ingested metrics (may contain nulls / outliers).

    Required: year, region + all metric columns.
    """
    errors: List[str] = []
    warnings: List[str] = []
    required = ["year", "region", *METRIC_COLUMNS]
    _check_required_columns(df, required, errors)
    _check_unique_keys(df, ["year", "region"], errors)
    _check_regions(df, errors, warnings)
    if "year" in df.columns and df["year"].isna().any():
        errors.append("year contains nulls")
    result = ContractResult(
        name="raw_metrics",
        version=CONTRACT_VERSION,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=len(df),
    )
    logger.info(
        "Contract raw_metrics: passed=%s rows=%d errors=%d",
        result.passed,
        result.n_rows,
        len(errors),
    )
    return result


def validate_clean_metrics(df: pd.DataFrame) -> ContractResult:
    """
    Contract for cleaned metrics — no nulls in metrics; domain ranges enforced.
    """
    errors: List[str] = []
    warnings: List[str] = []
    required = ["year", "region", *METRIC_COLUMNS]
    _check_required_columns(df, required, errors)
    _check_unique_keys(df, ["year", "region"], errors)
    _check_regions(df, errors, warnings)

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            errors.append(f"{col} still has nulls after cleaning")
        if col in ("internet_penetration", "digital_payment"):
            if ((df[col] < 0) | (df[col] > 100)).any():
                errors.append(f"{col} outside [0, 100]")
        elif ((df[col] < 0)).any():
            errors.append(f"{col} has negative values")

    result = ContractResult(
        name="clean_metrics",
        version=CONTRACT_VERSION,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=len(df),
    )
    logger.info("Contract clean_metrics: passed=%s", result.passed)
    return result


def validate_dei_index(df: pd.DataFrame) -> ContractResult:
    """Contract for DEI output table."""
    errors: List[str] = []
    warnings: List[str] = []
    required = ["year", "region", "dei"]
    _check_required_columns(df, required, errors)
    _check_unique_keys(df, ["year", "region"], errors)
    if "dei" in df.columns:
        if df["dei"].isna().any():
            errors.append("dei contains nulls")
        if ((df["dei"] < 0) | (df["dei"] > 100)).any():
            errors.append("dei outside [0, 100]")
    result = ContractResult(
        name="dei_index",
        version=CONTRACT_VERSION,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=len(df),
    )
    logger.info("Contract dei_index: passed=%s", result.passed)
    return result


def enforce_contract(result: ContractResult) -> ContractResult:
    """Raise ``ContractViolationError`` if contract failed."""
    result.raise_if_failed()
    return result
