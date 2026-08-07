"""Tests for causal effect estimation (AI Innovation → DEI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_analysis.effect_estimator import estimate_ai_effect_on_dei
from src.causal_analysis.identification import build_causal_panel
from src.data_pipeline.cleaning import clean
from src.data_pipeline.ingestion import generate_simulated_data
from src.indicators.index_calculator import IndexCalculator


@pytest.fixture(scope="module")
def dei_df():
    raw = generate_simulated_data(seed=11, inject_missing=True, inject_outliers=True)
    clean_df, _ = clean(raw)
    _, dei = IndexCalculator().calculate(clean_df)
    return dei


def test_build_causal_panel(dei_df):
    panel = build_causal_panel(dei_df)
    assert {"ai_innovation", "dei", "dei_ex_ai", "ai_high", "year", "region"}.issubset(panel.columns)
    assert panel["ai_high"].isin([0, 1]).all()
    assert len(panel) == len(dei_df)


def test_causal_effect_has_ci(dei_df):
    result = estimate_ai_effect_on_dei(
        dei_df,
        outcome="dei",
        prefer_engine="statsmodels",  # deterministic path in CI
        n_boot=80,
    )
    primary = result["primary"]
    assert "effect" in primary
    assert "ci_low" in primary and "ci_high" in primary
    assert primary["ci_low"] <= primary["effect"] <= primary["ci_high"]
    assert primary["n_obs"] > 0
    assert "interpretation" in result


def test_causal_spillover_outcome(dei_df):
    result = estimate_ai_effect_on_dei(
        dei_df,
        outcome="dei_ex_ai",
        prefer_engine="statsmodels",
        n_boot=60,
    )
    assert result["primary"]["outcome"] == "dei_ex_ai"
    assert result["primary"]["ci_high"] >= result["primary"]["ci_low"]


def test_binary_treatment_ate(dei_df):
    result = estimate_ai_effect_on_dei(
        dei_df,
        binary_treatment=True,
        prefer_engine="statsmodels",
        n_boot=60,
    )
    assert result["primary"]["treatment"] == "ai_high"
