"""
DEIP command-line interface.

Usage::

    python -m src.cli run
    python -m src.cli ingest --force
    python -m src.cli validate
    python -m src.cli forecast
    python -m src.cli report "分析未来数字经济趋势"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer

from src.config import DEI_CSV_PATH, PROCESSED_CSV_PATH, setup_logging

app = typer.Typer(
    name="deip",
    help="Digital Economy Intelligence Platform — enterprise demo CLI",
    add_completion=False,
)
logger = setup_logging("deip.cli")


@app.command()
def run(
    force: bool = typer.Option(True, help="Regenerate simulated raw data"),
) -> None:
    """Run the full staged pipeline (ingest → gate → DEI → lineage)."""
    from src.bootstrap import run_pipeline

    result = run_pipeline(force_regenerate=force)
    typer.echo(json.dumps(result["context"], indent=2, ensure_ascii=False))
    typer.secho(f"SUCCESS run_id={result['run_id']}", fg=typer.colors.GREEN)


@app.command()
def ingest(
    force: bool = typer.Option(False, help="Force regenerate raw CSV"),
) -> None:
    """Generate / load raw metrics and initialize the database."""
    from src.contracts.schemas import enforce_contract, validate_raw_metrics
    from src.data_pipeline.ingestion import ingest as do_ingest

    df = do_ingest(force_regenerate=force)
    enforce_contract(validate_raw_metrics(df))
    typer.secho(f"Ingested {len(df)} rows", fg=typer.colors.GREEN)


@app.command()
def validate() -> None:
    """Validate raw CSV and evaluate the quality gate (non-destructive)."""
    from src.data_pipeline.ingestion import read_raw_csv
    from src.data_pipeline.validation import (
        evaluate_quality_gate,
        print_quality_report,
        validate as do_validate,
    )

    df = read_raw_csv()
    report = do_validate(df)
    gate = evaluate_quality_gate(report)
    print_quality_report(report)
    if gate["passed"]:
        typer.secho("GATE PASSED", fg=typer.colors.GREEN)
    else:
        typer.secho("GATE FAILED: " + "; ".join(gate["reasons"]), fg=typer.colors.RED)
        raise typer.Exit(code=2)


@app.command()
def forecast() -> None:
    """Recompute DEI forecasts with prediction intervals + backtest."""
    import pandas as pd

    from src.analytics.forecasting import (
        forecast_dei_linear,
        forecast_summary,
        save_forecast,
        walk_forward_backtest,
    )

    if not DEI_CSV_PATH.exists():
        typer.secho("DEI CSV missing — run `python -m src.cli run` first", fg=typer.colors.RED)
        raise typer.Exit(1)
    dei = pd.read_csv(DEI_CSV_PATH)
    fcast = forecast_dei_linear(dei)
    bt = walk_forward_backtest(dei)
    summary = forecast_summary(fcast, bt)
    save_forecast(fcast)
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


@app.command()
def causal(
    outcome: str = typer.Option("dei", help="Outcome: dei | dei_ex_ai"),
    binary: bool = typer.Option(False, help="Use above-median AI as binary treatment"),
    engine: str = typer.Option("dowhy", help="Prefer engine: dowhy | statsmodels | econml"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON instead of readable report"),
) -> None:
    """Estimate causal effect of AI Innovation on DEI (+ confidence interval)."""
    import pandas as pd

    from src.causal_analysis.effect_estimator import estimate_ai_effect_on_dei
    from src.causal_analysis.report import format_causal_report

    if not DEI_CSV_PATH.exists():
        typer.secho("DEI CSV missing — run `python -m src.cli run` first", fg=typer.colors.RED)
        raise typer.Exit(1)
    dei = pd.read_csv(DEI_CSV_PATH)
    result = estimate_ai_effect_on_dei(
        dei,
        outcome=outcome,
        binary_treatment=binary,
        prefer_engine=engine,
        n_boot=300,
    )
    if as_json:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(format_causal_report(result, lang="zh"))
        primary = result["primary"]
        typer.secho(
            f"\n摘要: 效应={primary['effect']}  "
            f"CI=[{primary['ci_low']}, {primary['ci_high']}]  "
            f"{'显著' if result['significant_at_alpha'] else '不显著'}",
            fg=typer.colors.GREEN if result["significant_at_alpha"] else typer.colors.YELLOW,
        )


@app.command()
def report(
    query: str = typer.Argument("分析未来数字经济趋势", help="Research question"),
) -> None:
    """Generate a tool-grounded research brief (mock LLM by default)."""
    if not PROCESSED_CSV_PATH.exists() or not DEI_CSV_PATH.exists():
        typer.secho("Processed data missing — run `python -m src.cli run` first", fg=typer.colors.RED)
        raise typer.Exit(1)
    from src.ai_agent.report_generator import ResearchReportGenerator

    brief = ResearchReportGenerator().generate(query=query)
    typer.echo(brief.to_markdown())


@app.command("show-lineage")
def show_lineage(
    run_id: Optional[str] = typer.Option(None, help="Specific run_id; default = latest"),
) -> None:
    """Print a lineage manifest."""
    from src.config import LINEAGE_DIR

    files = sorted(LINEAGE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        typer.secho("No lineage files found", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    path = next((p for p in files if run_id and p.stem == run_id), files[0])
    typer.echo(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
