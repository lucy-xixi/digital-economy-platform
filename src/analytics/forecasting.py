"""
OLS forecasting with prediction intervals and walk-forward backtest.

Uses ``statsmodels.OLS`` (already in requirements) for proper inference:
confidence/prediction intervals, Durbin–Watson residuals, and out-of-sample MAE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.stattools import durbin_watson

from src.config import FORECAST_CSV_PATH, FORECAST_END, FORECAST_START, ensure_directories, setup_logging

logger = setup_logging(__name__)


def _fit_ols(years: np.ndarray, values: np.ndarray) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS: dei ~ const + year."""
    X = sm.add_constant(years.astype(float))
    return sm.OLS(values.astype(float), X).fit()


def forecast_dei_linear(
    dei_df: pd.DataFrame,
    forecast_start: int = FORECAST_START,
    forecast_end: int = FORECAST_END,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Forecast DEI per region with 95% prediction intervals.

    Args:
        dei_df: Historical DEI with year, region, dei.
        forecast_start: First forecast year.
        forecast_end: Last forecast year.
        alpha: Significance level for prediction intervals (default 5%).

    Returns:
        Long DataFrame with historical + forecast rows including
        ``dei_lo``, ``dei_hi``, ``r2``, ``mae``, ``dw``, ``slope``.
    """
    try:
        if not {"year", "region", "dei"}.issubset(dei_df.columns):
            raise ValueError("dei_df must contain year, region, dei")

        future_years = list(range(forecast_start, forecast_end + 1))
        frames: List[pd.DataFrame] = []

        for region, grp in dei_df.groupby("region"):
            g = grp.sort_values("year").dropna(subset=["dei"])
            if len(g) < 3:
                logger.warning("Skipping region %s — need ≥3 points for OLS CI", region)
                continue

            years = g["year"].to_numpy(dtype=float)
            y = g["dei"].to_numpy(dtype=float)
            model = _fit_ols(years, y)
            y_hat = model.fittedvalues
            r2 = float(r2_score(y, y_hat))
            mae = float(mean_absolute_error(y, y_hat))
            dw = float(durbin_watson(model.resid))
            slope = float(model.params[1])

            hist = g[["year", "region", "dei"]].copy()
            hist["type"] = "historical"
            hist["dei_lo"] = np.nan
            hist["dei_hi"] = np.nan
            hist["r2"] = r2
            hist["mae"] = mae
            hist["dw"] = dw
            hist["slope"] = slope

            X_future = sm.add_constant(np.array(future_years, dtype=float))
            pred = model.get_prediction(X_future)
            summary = pred.summary_frame(alpha=alpha)
            # mean / obs_ci for prediction interval
            y_mean = summary["mean"].to_numpy()
            y_lo = summary["obs_ci_lower"].to_numpy()
            y_hi = summary["obs_ci_upper"].to_numpy()
            y_mean = np.clip(y_mean, 0.0, 100.0)
            y_lo = np.clip(y_lo, 0.0, 100.0)
            y_hi = np.clip(y_hi, 0.0, 100.0)

            fut = pd.DataFrame(
                {
                    "year": future_years,
                    "region": region,
                    "dei": y_mean,
                    "type": "forecast",
                    "dei_lo": y_lo,
                    "dei_hi": y_hi,
                    "r2": r2,
                    "mae": mae,
                    "dw": dw,
                    "slope": slope,
                }
            )
            frames.append(pd.concat([hist, fut], ignore_index=True))
            logger.info(
                "Forecast %s: slope=%.3f R²=%.3f MAE=%.2f DW=%.2f",
                region,
                slope,
                r2,
                mae,
                dw,
            )

        if not frames:
            raise RuntimeError("No regions could be forecasted")

        result = pd.concat(frames, ignore_index=True)
        return result

    except Exception as exc:
        logger.exception("Forecasting failed: %s", exc)
        raise


def walk_forward_backtest(dei_df: pd.DataFrame, min_train: int = 4) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead backtest per region.

    For each region, train on years[:t] and predict year t+1, for t >= min_train.

    Returns:
        DataFrame with region, year, actual, predicted, abs_error.
    """
    rows: List[Dict[str, Any]] = []
    for region, grp in dei_df.groupby("region"):
        g = grp.sort_values("year").dropna(subset=["dei"]).reset_index(drop=True)
        if len(g) <= min_train:
            continue
        for i in range(min_train, len(g)):
            train = g.iloc[:i]
            test_row = g.iloc[i]
            model = _fit_ols(train["year"].to_numpy(dtype=float), train["dei"].to_numpy(dtype=float))
            X_test = sm.add_constant(np.array([float(test_row["year"])]))
            # add_constant on 1-row may drop const if no variance — force
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            if X_test.shape[1] == 1:
                X_test = np.column_stack([np.ones(1), X_test.ravel()])
            y_hat = float(model.predict(X_test)[0])
            actual = float(test_row["dei"])
            rows.append(
                {
                    "region": region,
                    "year": int(test_row["year"]),
                    "actual": actual,
                    "predicted": y_hat,
                    "abs_error": abs(actual - y_hat),
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        logger.info(
            "Walk-forward backtest: n=%d mean_AE=%.2f",
            len(result),
            result["abs_error"].mean(),
        )
    return result


def forecast_summary(forecast_df: pd.DataFrame, backtest_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Summarize forecast + optional backtest for reporting / LLM tools."""
    fut = forecast_df[forecast_df["type"] == "forecast"]
    last_year = int(fut["year"].max()) if len(fut) else FORECAST_END
    snapshot = fut[fut["year"] == last_year].sort_values("dei", ascending=False)

    models = (
        forecast_df[forecast_df["type"] == "historical"]
        .groupby("region")
        .agg(
            r2=("r2", "first"),
            mae=("mae", "first"),
            dw=("dw", "first"),
            slope=("slope", "first"),
        )
        .reset_index()
    )

    summary: Dict[str, Any] = {
        "horizon": f"{FORECAST_START}-{FORECAST_END}",
        "snapshot_year": last_year,
        "projections": snapshot[["region", "dei"]].to_dict(orient="records"),
        "model_fit": models.to_dict(orient="records"),
        "method": "statsmodels.OLS with 95% prediction intervals",
    }
    if backtest_df is not None and len(backtest_df):
        summary["backtest"] = {
            "n": int(len(backtest_df)),
            "mean_abs_error": round(float(backtest_df["abs_error"].mean()), 3),
            "by_region": (
                backtest_df.groupby("region")["abs_error"]
                .mean()
                .round(3)
                .to_dict()
            ),
        }
    return summary


def save_forecast(forecast_df: pd.DataFrame, path: Optional[Any] = None) -> Any:
    """Persist forecast table to processed CSV."""
    ensure_directories()
    path = path or FORECAST_CSV_PATH
    forecast_df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Forecast saved to %s", path)
    return path
