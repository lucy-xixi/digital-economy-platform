"""
Enterprise-grade tests for DEIP: contracts, gates, panel stats, forecasts, AI agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_agent.report_generator import ResearchReportGenerator
from src.ai_agent.schemas import ResearchBrief
from src.analytics.correlation import correlate_ai_vs_dei
from src.analytics.diagnostics import diagnose_region_trend
from src.analytics.forecasting import forecast_dei_linear, walk_forward_backtest
from src.analytics.trend_analysis import compute_cagr, trend_summary
from src.config import DEI_WEIGHTS, METRIC_COLUMNS, REGIONS
from src.contracts.schemas import (
    ContractViolationError,
    enforce_contract,
    validate_clean_metrics,
    validate_dei_index,
    validate_raw_metrics,
)
from src.data_pipeline.cleaning import clean, detect_outliers_iqr, impute_missing
from src.data_pipeline.ingestion import generate_simulated_data
from src.data_pipeline.validation import (
    QualityGateError,
    compute_quality_score,
    enforce_quality_gate,
    evaluate_quality_gate,
    validate,
)
from src.indicators.index_calculator import IndexCalculator, minmax_scale
from src.indicators.indicator_framework import IndicatorFramework
from src.indicators.sensitivity import weight_sensitivity
from src.orchestration.context import PipelineContext


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return generate_simulated_data(seed=7, inject_missing=True, inject_outliers=True)


@pytest.fixture(scope="module")
def clean_bundle(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return clean(raw_df)


@pytest.fixture(scope="module")
def dei_bundle(clean_bundle: tuple[pd.DataFrame, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_df, _ = clean_bundle
    primary, dei = IndexCalculator().calculate(clean_df)
    return clean_df, primary, dei


# ---------------------------------------------------------------------------
# Pipeline / DQ
# ---------------------------------------------------------------------------


def test_generate_simulated_data_shape(raw_df: pd.DataFrame) -> None:
    expected_rows = len(REGIONS) * (2025 - 2018 + 1)
    assert len(raw_df) == expected_rows
    assert set(REGIONS) == set(raw_df["region"].unique())


def test_impute_removes_nans(raw_df: pd.DataFrame) -> None:
    imputed, _ = impute_missing(raw_df)
    assert imputed[METRIC_COLUMNS].isna().sum().sum() == 0


def test_outlier_detection_flags_injected(raw_df: pd.DataFrame) -> None:
    flags = detect_outliers_iqr(raw_df.dropna(subset=METRIC_COLUMNS, how="all"))
    assert int(flags.to_numpy().sum()) > 0


def test_clean_pipeline(clean_bundle: tuple[pd.DataFrame, dict]) -> None:
    clean_df, meta = clean_bundle
    assert meta["rows_out"] > 0
    assert clean_df[METRIC_COLUMNS].isna().sum().sum() == 0


def test_raw_contract_passes(raw_df: pd.DataFrame) -> None:
    result = validate_raw_metrics(raw_df)
    assert result.passed
    enforce_contract(result)


def test_clean_contract_passes(clean_bundle: tuple[pd.DataFrame, dict]) -> None:
    clean_df, _ = clean_bundle
    result = validate_clean_metrics(clean_df)
    assert result.passed


def test_contract_rejects_missing_column(raw_df: pd.DataFrame) -> None:
    bad = raw_df.drop(columns=["ai_patents"])
    result = validate_raw_metrics(bad)
    assert not result.passed
    with pytest.raises(ContractViolationError):
        enforce_contract(result)


def test_quality_gate_blocks_on_fail() -> None:
    report = {
        "quality_score": 10.0,
        "completeness": {"completeness_rate": 0.5, "missing_total": 100},
        "uniqueness": {"duplicate_keys": 5},
        "coverage": {"missing_regions": ["Beijing"]},
        "outliers": {"outlier_total": 0},
    }
    gate = evaluate_quality_gate(report)
    assert gate["passed"] is False
    with pytest.raises(QualityGateError):
        enforce_quality_gate(report, block=True)


def test_quality_gate_passes_healthy(raw_df: pd.DataFrame) -> None:
    report = validate(raw_df)
    gate = evaluate_quality_gate(report)
    assert gate["passed"] is True


def test_quality_score_bounds() -> None:
    assert compute_quality_score(1.0, 0, 0, 40) == 100.0


# ---------------------------------------------------------------------------
# Indicators / sensitivity
# ---------------------------------------------------------------------------


def test_indicator_weights_sum() -> None:
    fw = IndicatorFramework()
    assert abs(sum(fw.weights.values()) - 1.0) < 1e-9
    assert set(fw.list_pillars()) == set(DEI_WEIGHTS)


def test_minmax_scale_range() -> None:
    scaled = minmax_scale(pd.Series([10.0, 20.0, 30.0]))
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(100.0)


def test_dei_calculation(dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    _, primary, dei = dei_bundle
    assert {"digital_infrastructure", "digital_industry", "ai_innovation", "digital_application"}.issubset(
        primary.columns
    )
    assert dei["dei"].between(0, 100).all()
    assert validate_dei_index(dei).passed


def test_weight_sensitivity_rank_stability(
    dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    clean_df, _, _ = dei_bundle
    sens = weight_sensitivity(clean_df, delta=0.10)
    assert sens["mean_rank_stability"] is not None
    assert sens["mean_rank_stability"] >= 0.5
    assert len(sens["scenarios"]) == 8  # 4 pillars × up/down


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_trend_and_cagr(dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    _, _, dei = dei_bundle
    summary = trend_summary(dei)
    assert summary["top_region"] in REGIONS
    assert compute_cagr(50, 100, 5) == pytest.approx(2 ** (1 / 5) - 1)


def test_panel_correlation(dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    clean_df, _, dei = dei_bundle
    result = correlate_ai_vs_dei(clean_df, dei)
    assert -1.0 <= result["pearson_r"] <= 1.0
    assert "within_pearson_r" in result
    assert result["within_pearson_r"] is not None


def test_forecast_intervals_and_backtest(
    dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, dei = dei_bundle
    fcast = forecast_dei_linear(dei)
    fut = fcast[fcast["type"] == "forecast"]
    assert set(fut["year"]) == set(range(2026, 2031))
    assert fut["dei_lo"].notna().all()
    assert fut["dei_hi"].notna().all()
    assert (fut["dei_lo"] <= fut["dei"]).all()
    assert (fut["dei"] <= fut["dei_hi"]).all()
    assert "dw" in fcast.columns

    bt = walk_forward_backtest(dei)
    assert len(bt) > 0
    assert bt["abs_error"].mean() >= 0


def test_diagnostics(dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    _, _, dei = dei_bundle
    diag = diagnose_region_trend(dei)
    assert len(diag) == len(REGIONS)
    assert set(["r2", "dw", "jb_pvalue"]).issubset(diag.columns)


# ---------------------------------------------------------------------------
# Orchestration / AI
# ---------------------------------------------------------------------------


def test_pipeline_context_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import orchestration
    from src.orchestration import context as ctx_mod

    monkeypatch.setattr(ctx_mod, "LINEAGE_DIR", tmp_path)
    ctx = PipelineContext()
    ctx.start_stage("demo")
    ctx.end_stage("demo", rows=1)
    path = ctx.finalize(status="SUCCESS")
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == ctx.run_id
    assert payload["status"] == "SUCCESS"
    assert payload["stages"][0]["name"] == "demo"


def test_research_brief_grounded(dei_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean_df, _, dei = dei_bundle
    # Ensure tool loaders see fixture CSVs
    clean_path = tmp_path / "clean.csv"
    dei_path = tmp_path / "dei.csv"
    clean_df.to_csv(clean_path, index=False)
    dei.to_csv(dei_path, index=False)

    import src.ai_agent.tools as tools_mod
    import src.config as config_mod

    monkeypatch.setattr(config_mod, "PROCESSED_CSV_PATH", clean_path)
    monkeypatch.setattr(config_mod, "DEI_CSV_PATH", dei_path)
    monkeypatch.setattr(tools_mod, "PROCESSED_CSV_PATH", clean_path)
    monkeypatch.setattr(tools_mod, "DEI_CSV_PATH", dei_path)

    # Avoid writing to real DB — use temp repository db
    from src.db.repository import DeipRepository

    repo = DeipRepository(db_path=tmp_path / "test.db")
    gen = ResearchReportGenerator(use_mock=True, repository=repo)
    brief = gen.generate(query="分析未来数字经济趋势", persist=True)
    assert isinstance(brief, ResearchBrief)
    assert brief.grounded is True
    assert len(brief.key_findings) >= 3
    assert "pearson_r" in brief.key_findings[1].evidence or str(
        correlate_ai_vs_dei(clean_df, dei)["pearson_r"]
    ) in brief.executive_summary
    md = brief.to_markdown()
    assert "Executive Summary" in md
