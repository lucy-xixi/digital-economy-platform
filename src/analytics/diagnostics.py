"""
Regression diagnostics for DEI time-trend models.

Durbin–Watson, residual normality (Jarque–Bera), and Breusch–Pagan
heteroskedasticity — surfaces that interviewers expect when ``statsmodels``
is on the resume.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from src.config import setup_logging

logger = setup_logging(__name__)


def diagnose_region_trend(dei_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ``dei ~ year`` per region and return residual diagnostics.

    Returns:
        DataFrame with region, n, r2, dw, jb_pvalue, bp_pvalue, slope.
    """
    rows: List[Dict[str, Any]] = []
    for region, grp in dei_df.groupby("region"):
        g = grp.sort_values("year").dropna(subset=["dei"])
        if len(g) < 4:
            continue
        y = g["dei"].to_numpy(dtype=float)
        X = sm.add_constant(g["year"].to_numpy(dtype=float))
        model = sm.OLS(y, X).fit()
        dw = float(durbin_watson(model.resid))
        jb_stat, jb_p, _, _ = jarque_bera(model.resid)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
            bp_p = float(bp_p)
        except Exception:
            bp_p = float("nan")
        rows.append(
            {
                "region": region,
                "n": len(g),
                "r2": round(float(model.rsquared), 4),
                "slope": round(float(model.params[1]), 4),
                "dw": round(dw, 3),
                "jb_pvalue": round(float(jb_p), 4),
                "bp_pvalue": round(bp_p, 4) if not np.isnan(bp_p) else None,
                "autocorr_flag": dw < 1.5 or dw > 2.5,
            }
        )
    result = pd.DataFrame(rows)
    logger.info("Diagnostics computed for %d regions", len(result))
    return result


def diagnostics_summary(diag_df: pd.DataFrame) -> Dict[str, Any]:
    """Compact diagnostics payload for LLM tools / CLI."""
    if diag_df.empty:
        return {"regions": 0}
    return {
        "regions": int(len(diag_df)),
        "mean_r2": round(float(diag_df["r2"].mean()), 4),
        "regions_with_autocorr_flag": diag_df.loc[
            diag_df["autocorr_flag"], "region"
        ].tolist(),
        "table": diag_df.to_dict(orient="records"),
    }
