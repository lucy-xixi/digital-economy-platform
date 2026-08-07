"""
DEIP bootstrap orchestrator.

Staged pipeline with run_id, contracts, hard DQ gate, transactional upserts,
and lineage emission:

  ingest → contract(raw) → validate+gate → clean → contract(clean)
  → DEI → contract(dei) → persist → analytics warm-up → lineage
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.correlation import correlate_ai_vs_dei
from src.analytics.diagnostics import diagnose_region_trend
from src.analytics.forecasting import (
    forecast_dei_linear,
    forecast_summary,
    save_forecast,
    walk_forward_backtest,
)
from src.analytics.trend_analysis import trend_summary
from src.config import (
    DEI_CSV_PATH,
    PROCESSED_CSV_PATH,
    QUALITY_REPORT_PATH,
    RAW_CSV_PATH,
    ensure_directories,
    setup_logging,
)
from src.contracts.schemas import (
    enforce_contract,
    validate_clean_metrics,
    validate_dei_index,
    validate_raw_metrics,
)
from src.data_pipeline.cleaning import clean, save_processed
from src.data_pipeline.ingestion import ingest
from src.data_pipeline.validation import (
    enforce_quality_gate,
    print_quality_report,
    save_quality_report,
    validate,
)
from src.db.repository import DeipRepository
from src.indicators.index_calculator import IndexCalculator
from src.indicators.sensitivity import weight_sensitivity
from src.orchestration.context import PipelineContext

logger = setup_logging(__name__)


def run_pipeline(force_regenerate: bool = True) -> dict:
    """
    Execute the end-to-end DEIP pipeline with enterprise control surfaces.

    Args:
        force_regenerate: Regenerate simulated raw CSV.

    Returns:
        Dict with DataFrames, reports, and pipeline context summary.
    """
    ensure_directories()
    ctx = PipelineContext()
    repo = DeipRepository()
    repo.start_run(
        run_id=ctx.run_id,
        started_at=ctx.started_at,
        schema_version=ctx.schema_version,
        contract_version=ctx.contract_version,
        config_hash=ctx.config_hash,
    )
    logger.info("=" * 60)
    logger.info("DEIP pipeline starting run_id=%s", ctx.run_id)
    logger.info("=" * 60)

    try:
        # 1. Ingest
        ctx.start_stage("ingest")
        raw = ingest(force_regenerate=force_regenerate)
        ctx.register_artifact("raw_csv", RAW_CSV_PATH)
        enforce_contract(validate_raw_metrics(raw))
        ctx.end_stage("ingest", rows=len(raw))

        # 2. Validate + hard gate
        ctx.start_stage("validate")
        quality_report = validate(raw)
        gate = enforce_quality_gate(quality_report)
        save_quality_report(quality_report)
        repo.save_quality_report(quality_report, ctx.run_id, gate_passed=gate["passed"])
        print_quality_report(quality_report)
        ctx.register_artifact("quality_report", QUALITY_REPORT_PATH)
        ctx.end_stage("validate", score=quality_report.get("quality_score"), gate=gate["passed"])

        # 3. Clean
        ctx.start_stage("clean")
        clean_df, clean_meta = clean(raw)
        enforce_contract(validate_clean_metrics(clean_df))
        save_processed(clean_df)
        ctx.register_artifact("clean_csv", PROCESSED_CSV_PATH)
        ctx.end_stage("clean", **{k: clean_meta[k] for k in ("rows_out", "outlier_count", "capped_count")})

        # 4. DEI
        ctx.start_stage("compute_dei")
        calculator = IndexCalculator()
        primary, dei = calculator.calculate(clean_df)
        enforce_contract(validate_dei_index(dei))
        calculator.save(primary, dei)  # CSV
        ctx.register_artifact("dei_csv", DEI_CSV_PATH)
        ctx.end_stage("compute_dei", rows=len(dei), mean_dei=round(float(dei["dei"].mean()), 2))

        # 5. Transactional warehouse upsert
        ctx.start_stage("persist")
        repo.upsert_metrics(clean_df, ctx.run_id)
        repo.upsert_primary_and_dei(primary, dei, ctx.run_id)
        ctx.end_stage("persist", status="OK")

        # 6. Analytics warm-up
        ctx.start_stage("analytics")
        t_summary = trend_summary(dei)
        c_summary = correlate_ai_vs_dei(clean_df, dei)
        f_df = forecast_dei_linear(dei)
        bt = walk_forward_backtest(dei)
        f_summary = forecast_summary(f_df, bt)
        save_forecast(f_df)
        diag = diagnose_region_trend(dei)
        sens = weight_sensitivity(clean_df)
        ctx.end_stage(
            "analytics",
            top_region=t_summary.get("top_region"),
            pearson_r=c_summary.get("pearson_r"),
            within_r=c_summary.get("within_pearson_r"),
            backtest_mae=(f_summary.get("backtest") or {}).get("mean_abs_error"),
            rank_stability=sens.get("mean_rank_stability"),
        )

        lineage_path = ctx.finalize(status="SUCCESS")
        repo.finish_run(
            ctx.run_id,
            status="SUCCESS",
            finished_at=ctx.finished_at or "",
            lineage_path=str(lineage_path),
        )

        logger.info("Processed CSV : %s", PROCESSED_CSV_PATH)
        logger.info("DEI CSV       : %s", DEI_CSV_PATH)
        logger.info("Top region    : %s", t_summary.get("top_region"))
        logger.info("AI–DEI r      : %s (within=%s)", c_summary.get("pearson_r"), c_summary.get("within_pearson_r"))
        logger.info("DEIP pipeline completed run_id=%s", ctx.run_id)

        return {
            "raw": raw,
            "clean": clean_df,
            "clean_meta": clean_meta,
            "primary": primary,
            "dei": dei,
            "quality_report": quality_report,
            "trend_summary": t_summary,
            "correlation_summary": c_summary,
            "forecast_df": f_df,
            "forecast_summary": f_summary,
            "diagnostics": diag,
            "sensitivity": sens,
            "context": ctx.to_summary(),
            "run_id": ctx.run_id,
        }

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        lineage_path = ctx.finalize(status="FAILED", error=str(exc))
        repo.finish_run(
            ctx.run_id,
            status="FAILED",
            finished_at=ctx.finished_at or "",
            lineage_path=str(lineage_path),
            error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    run_pipeline(force_regenerate=True)
