
# Digital Economy Intelligence Platform (DEIP)

A portfolio prototype for digital-economy research and strategic analytics. The platform combines a governed data pipeline, statistical analysis, forecasting, causal-analysis components, and an AI-assisted research workflow.

It is designed to demonstrate how analytical results can be produced, validated, traced, and presented through a reproducible workflow.

> **Project status:** Independent portfolio prototype. The platform demonstrates engineering and analytical methods; it is not presented as a production deployment or as a source of verified real-world economic conclusions.

## Overview

DEIP addresses a common analytical workflow: transforming multi-source industry indicators into validated datasets, statistical outputs, and research-oriented summaries.

The implementation emphasizes four areas:

1. **Data engineering:** Data contracts, hard quality gates, idempotent database writes, and run-level lineage.
2. **Statistical analysis:** Panel correlation, OLS prediction intervals, walk-forward evaluation, residual diagnostics, weight sensitivity, and causal-analysis components.
3. **AI-assisted research:** Tool calling, structured output, lightweight methodology retrieval, and evidence-grounding checks.
4. **Engineering discipline:** Type hints, logging, Docker support, and automated tests.

## Architecture

```text
CLI / Streamlit
      |
      v
PipelineContext(run_id) ----> lineage/<run_id>.json
      |
      +--> Ingest --> Raw data contract
      |
      +--> Validate --> Hard quality gate --> Quality reports
      |
      +--> Clean --> Clean-data contract
      |
      +--> Compute DEI --> DEI contract
      |
      +--> Transactional SQLite upsert
      |
      +--> Analytics warm-up
             |-- Panel correlation
             |-- OLS prediction intervals
             |-- Walk-forward backtesting
             |-- Residual diagnostics
             |-- Weight sensitivity
      |
      v
Tool-grounded Research Agent
```

## Key Features

### Data pipeline and governance

- Validates raw, cleaned, and Digital Economy Index (DEI) data against explicit contracts.
- Applies a hard data-quality gate before downstream processing.
- Uses SQLite upsert operations with conflict handling and transactions.
- Records run identifiers and lineage artifacts.
- Tracks stage durations, output hashes, and configuration hashes.

### Statistical analysis

- Compares pooled Pearson correlation with within-group correlation after removing regional means.
- Fits OLS models with 95% prediction intervals.
- Uses walk-forward evaluation for one-step-ahead forecast errors.
- Runs residual diagnostics, including Durbin–Watson, Jarque–Bera, and Breusch–Pagan tests where applicable.
- Evaluates ranking stability under ±10% changes to DEI component weights.
- Includes a DoWhy backdoor-analysis workflow for the relationship between AI innovation and DEI, with confidence intervals and a `dei_ex_ai` robustness variant.
- Supports an optional EconML LinearDML component.

### AI-assisted research

- Executes analytical tools before assembling a structured `ResearchBrief`.
- Uses Pydantic-based structured output.
- Includes lightweight keyword-based retrieval from `docs/methodology.md`.
- Checks that key numerical findings are supported by the evidence supplied to the research brief.
- Keeps the AI layer separate from the underlying statistical calculations.

## Digital Economy Index

The current DEI configuration uses the following component weights:

| Component | Weight |
|---|---:|
| Infrastructure | 30% |
| Industry | 30% |
| AI Innovation | 20% |
| Application | 20% |

These weights are configurable analytical assumptions, not an externally validated universal standard.

## Quick Start

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
# End-to-end pipeline, quality gates, and lineage
python -m src.cli run

# Run data-quality validation only
python -m src.cli validate

# Forecasting with OLS prediction intervals and walk-forward evaluation
python -m src.cli forecast

# Causal-analysis workflow
python -m src.cli causal

# Generate a research-oriented report
python -m src.cli report "Analyze future digital-economy trends"

# Launch the dashboard
streamlit run dashboard/app.py
```

### Run tests

```bash
pytest tests/ -v
```

### Run with Docker

```bash
docker compose up --build
```

## Engineering Design Details

### Data contracts and quality gates

The `metrics_v1` contract defines expected fields, unique keys, and value-range constraints for raw, cleaned, and DEI data. Contract violations raise an error rather than silently passing invalid data downstream.

The quality gate checks configured quality conditions, including scores, missingness, duplicate keys, and regional coverage.

### Idempotent writes and lineage

The SQLite repository uses `INSERT ... ON CONFLICT DO UPDATE` semantics within a transaction. Run identifiers connect pipeline executions to their lineage records.

Lineage artifacts are stored under:

```text
data/processed/lineage/
```

### Statistical diagnostics and sensitivity

The analytical workflow includes prediction intervals, forecast backtesting, residual diagnostics, and sensitivity analysis. These checks help communicate model limitations and the stability of analytical rankings rather than treating a single model output as definitive.

## Technology Stack

- Python
- pandas, NumPy
- statsmodels
- DoWhy and optional EconML
- SQLite
- Streamlit
- Pydantic
- pytest
- Docker

## Validation and Limitations

- The analysis depends on the quality, coverage, and assumptions of the supplied indicator data.
- Correlations and regression coefficients do not by themselves establish causality.
- Causal estimates require defensible assumptions about confounding and model specification.
- DEI component weights are configurable and should be assessed for sensitivity.
- AI-generated summaries are intended to communicate computed findings, not replace the underlying calculations or independent review.

## License

MIT. See `LICENSE` for details.
