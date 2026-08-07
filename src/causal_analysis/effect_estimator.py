"""
Causal effect estimator: AI Innovation Index → Digital Economy Index.

Primary engine: **DoWhy** backdoor adjustment (linear regression) with bootstrap CI.
Fallback (if DoWhy unavailable): statsmodels OLS + percentile bootstrap CI.
Optional enrichment: **EconML** ``LinearDML`` ATE interval when installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.causal_analysis.identification import build_causal_panel, causal_graph_gml
from src.config import setup_logging

logger = setup_logging(__name__)


@dataclass
class CausalEffectResult:
    """Structured causal effect estimate with uncertainty."""

    treatment: str
    outcome: str
    estimand: str
    method: str
    effect: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_obs: int
    p_value: Optional[float] = None
    std_error: Optional[float] = None
    engine: str = "statsmodels"
    notes: str = ""
    refutations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bootstrap_ols_ci(
    y: np.ndarray,
    X: np.ndarray,
    coef_index: int,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float, float, float]:
    """
    OLS point estimate + percentile bootstrap CI for a single coefficient.

    Returns:
        (effect, ci_low, ci_high, std_error, p_value)
    """
    model = sm.OLS(y, X).fit()
    effect = float(model.params[coef_index])
    se = float(model.bse[coef_index])
    p_value = float(model.pvalues[coef_index])

    rng = np.random.default_rng(seed)
    n = len(y)
    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_fit = sm.OLS(y[idx], X[idx]).fit()
            boots.append(float(boot_fit.params[coef_index]))
        except Exception:
            continue
    if len(boots) < max(30, n_boot // 5):
        # Analytic normal CI fallback
        z = 1.959963984540054
        return effect, effect - z * se, effect + z * se, se, p_value

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return effect, lo, hi, se, p_value


def _estimate_with_statsmodels(
    panel: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: Sequence[str],
    alpha: float,
    n_boot: int,
) -> CausalEffectResult:
    """Backdoor OLS with region FE dummies + bootstrap CI."""
    df = panel.copy()
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    cause_cols = [c for c in common_causes if c in df.columns and c != "region"]
    X_parts = [df[[treatment]], df[cause_cols], region_dummies]
    X = sm.add_constant(pd.concat(X_parts, axis=1).astype(float))
    y = df[outcome].to_numpy(dtype=float)
    X_np = X.to_numpy(dtype=float)
    coef_index = list(X.columns).index(treatment)

    effect, lo, hi, se, p_value = _bootstrap_ols_ci(
        y, X_np, coef_index=coef_index, n_boot=n_boot, alpha=alpha
    )
    return CausalEffectResult(
        treatment=treatment,
        outcome=outcome,
        estimand="ATE (backdoor / linear adjustment)",
        method="OLS backdoor + percentile bootstrap CI",
        effect=round(effect, 4),
        ci_low=round(lo, 4),
        ci_high=round(hi, 4),
        confidence_level=1.0 - alpha,
        n_obs=len(df),
        p_value=round(p_value, 6),
        std_error=round(se, 4),
        engine="statsmodels",
        notes="Fallback estimator used when DoWhy is unavailable.",
    )


def _estimate_with_dowhy(
    panel: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: Sequence[str],
    alpha: float,
    n_boot: int,
) -> CausalEffectResult:
    """DoWhy CausalModel + backdoor.linear_regression + bootstrap CI."""
    from dowhy import CausalModel  # type: ignore

    causes = [c for c in common_causes if c in panel.columns]
    data = panel[[treatment, outcome, *causes, "region"]].copy()
    # Encode region as numeric codes for DoWhy simplicity
    data["region_code"] = pd.Categorical(data["region"]).codes.astype(float)
    graph_causes = [*causes, "region_code"]
    graph = causal_graph_gml(treatment, outcome, graph_causes)

    model = CausalModel(
        data=data.drop(columns=["region"]),
        treatment=treatment,
        outcome=outcome,
        common_causes=graph_causes,
        graph=graph,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        confidence_intervals=True,
        method_params={"num_simulations": n_boot},
    )

    effect = float(estimate.value)
    ci = None
    try:
        ci = estimate.get_confidence_intervals(confidence_level=1.0 - alpha)
    except Exception:
        ci = None

    if ci is None:
        # Manual bootstrap via DoWhy refute-style resampling of the estimate value
        _, lo, hi, se, p_value = _bootstrap_ols_from_panel(
            data, treatment, outcome, graph_causes, alpha=alpha, n_boot=n_boot
        )
    else:
        # DoWhy may return [[low, high]] or (low, high)
        arr = np.array(ci, dtype=float).reshape(-1)
        lo, hi = float(arr[0]), float(arr[-1])
        se, p_value = None, None
        # derive approx SE / p from CI if needed
        se = (hi - lo) / (2 * 1.959963984540054)

    # Light refutation (optional, best-effort)
    refutations: Dict[str, Any] = {}
    try:
        refute = model.refute_estimate(
            identified,
            estimate,
            method_name="random_common_cause",
        )
        refutations["random_common_cause"] = {
            "new_effect": float(getattr(refute, "new_effect", np.nan)),
            "estimated_effect": float(getattr(refute, "estimated_effect", effect)),
        }
    except Exception as exc:
        logger.warning("DoWhy refutation skipped: %s", exc)

    return CausalEffectResult(
        treatment=treatment,
        outcome=outcome,
        estimand=str(getattr(identified, "estimands", "backdoor")),
        method="DoWhy backdoor.linear_regression",
        effect=round(effect, 4),
        ci_low=round(lo, 4),
        ci_high=round(hi, 4),
        confidence_level=1.0 - alpha,
        n_obs=len(data),
        p_value=None if p_value is None else round(float(p_value), 6),
        std_error=None if se is None else round(float(se), 4),
        engine="dowhy",
        notes=(
            "Backdoor adjustment for year/pillar/region confounders. "
            "If outcome=dei, part of the effect is mechanical (AI weight=20%)."
        ),
        refutations=refutations or None,
    )


def _bootstrap_ols_from_panel(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    causes: Sequence[str],
    alpha: float,
    n_boot: int,
) -> tuple[float, float, float, float, float]:
    X = sm.add_constant(data[[treatment, *causes]].astype(float))
    y = data[outcome].to_numpy(dtype=float)
    coef_index = list(X.columns).index(treatment)
    return _bootstrap_ols_ci(y, X.to_numpy(dtype=float), coef_index, n_boot=n_boot, alpha=alpha)


def _estimate_with_econml(
    panel: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: Sequence[str],
    alpha: float,
) -> Optional[CausalEffectResult]:
    """Optional LinearDML ATE (binary or continuous treatment)."""
    try:
        from econml.dml import LinearDML  # type: ignore
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:
        logger.info("EconML not available (%s)", exc)
        return None

    causes = [c for c in common_causes if c in panel.columns]
    W = panel[causes].to_numpy(dtype=float) if causes else None
    Y = panel[outcome].to_numpy(dtype=float)
    T = panel[treatment].to_numpy(dtype=float)

    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        model_t=RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        random_state=42,
    )
    est.fit(Y, T, X=None, W=W)
    effect = float(np.asarray(est.ate()).reshape(-1)[0])
    interval = est.ate_interval(alpha=alpha)
    lo = float(np.asarray(interval[0]).reshape(-1)[0])
    hi = float(np.asarray(interval[1]).reshape(-1)[0])

    return CausalEffectResult(
        treatment=treatment,
        outcome=outcome,
        estimand="ATE (Double/Debiased ML)",
        method="EconML LinearDML",
        effect=round(effect, 4),
        ci_low=round(lo, 4),
        ci_high=round(hi, 4),
        confidence_level=1.0 - alpha,
        n_obs=len(panel),
        engine="econml",
        notes="Optional DML estimate for robustness comparison.",
    )


def estimate_ai_effect_on_dei(
    dei_df: pd.DataFrame,
    outcome: str = "dei",
    treatment: str = "ai_innovation",
    binary_treatment: bool = False,
    alpha: float = 0.05,
    n_boot: int = 400,
    prefer_engine: str = "dowhy",
) -> Dict[str, Any]:
    """
    Estimate the causal effect of AI Innovation on DEI (or leave-one-out DEI).

    Args:
        dei_df: DEI table including primary pillars.
        outcome: ``dei`` or ``dei_ex_ai``.
        treatment: ``ai_innovation`` or ``ai_high`` (if binary_treatment=True).
        binary_treatment: If True, use above-median AI as treatment.
        alpha: Significance level for CI (default 95% CI).
        n_boot: Bootstrap replications.
        prefer_engine: ``dowhy`` | ``statsmodels`` | ``econml``.

    Returns:
        Dict with primary estimate, optional robustness estimates, and panel meta.
    """
    panel = build_causal_panel(dei_df)
    if outcome not in panel.columns:
        raise ValueError(f"Unknown outcome '{outcome}'. Use 'dei' or 'dei_ex_ai'.")

    treat_col = "ai_high" if binary_treatment else treatment
    if treat_col not in panel.columns:
        raise ValueError(f"Unknown treatment '{treat_col}'")

    # For continuous AI → DEI, do NOT put other pillars in the adjustment set when
    # outcome is full DEI (would recover the mechanical weight ≈ 0.2).
    if outcome == "dei" and treat_col == "ai_innovation":
        common_causes = ["year"]
        identification_note = (
            "Adjustment set={year, region FE}. Other pillars excluded because DEI "
            "is a weighted sum that already includes ai_innovation (mechanical weight=0.2)."
        )
    else:
        common_causes = [
            "year",
            "digital_infrastructure",
            "digital_industry",
            "digital_application",
        ]
        if treat_col == "ai_high":
            common_causes = ["year", "digital_infrastructure", "digital_industry", "digital_application"]
        identification_note = (
            "Adjustment set includes year and other pillars (and region FE). "
            "Preferred for spillover outcome dei_ex_ai or binary AI treatment."
        )

    primary: CausalEffectResult
    engine_used = prefer_engine

    if prefer_engine == "econml":
        econ = _estimate_with_econml(panel, treat_col, outcome, common_causes, alpha)
        if econ is not None:
            primary = econ
        else:
            prefer_engine = "dowhy"

    if prefer_engine == "dowhy":
        try:
            primary = _estimate_with_dowhy(
                panel, treat_col, outcome, common_causes, alpha=alpha, n_boot=n_boot
            )
            engine_used = "dowhy"
        except Exception as exc:
            logger.warning("DoWhy estimate failed (%s); falling back to statsmodels", exc)
            primary = _estimate_with_statsmodels(
                panel, treat_col, outcome, common_causes, alpha=alpha, n_boot=n_boot
            )
            engine_used = "statsmodels"
    elif prefer_engine == "statsmodels":
        primary = _estimate_with_statsmodels(
            panel, treat_col, outcome, common_causes, alpha=alpha, n_boot=n_boot
        )
        engine_used = "statsmodels"

    # Robustness: leave-one-out spillover + binary ATE when running default dei/continuous
    robustness: List[Dict[str, Any]] = []
    try:
        if outcome == "dei" and treat_col == "ai_innovation":
            spill = _estimate_with_statsmodels(
                panel,
                "ai_innovation",
                "dei_ex_ai",
                ["year"],
                alpha=alpha,
                n_boot=max(200, n_boot // 2),
            )
            spill.notes = "Spillover onto non-AI pillars (leave-one-out DEI)."
            robustness.append(spill.to_dict())

            binary = _estimate_with_statsmodels(
                panel,
                "ai_high",
                "dei",
                ["year", "digital_infrastructure", "digital_industry", "digital_application"],
                alpha=alpha,
                n_boot=max(200, n_boot // 2),
            )
            binary.notes = "Binary ATE: above-median AI Innovation vs below."
            robustness.append(binary.to_dict())
    except Exception as exc:
        logger.warning("Robustness estimates skipped: %s", exc)

    # Optional EconML companion
    try:
        econ = _estimate_with_econml(panel, treat_col, outcome, common_causes, alpha)
        if econ is not None and engine_used != "econml":
            robustness.append(econ.to_dict())
    except Exception:
        pass

    significant = not (primary.ci_low <= 0.0 <= primary.ci_high)
    summary = {
        "question": "What is the causal effect of AI Innovation Index on Digital Economy Index?",
        "primary": primary.to_dict(),
        "significant_at_alpha": significant,
        "alpha": alpha,
        "identification": {
            "treatment": treat_col,
            "outcome": outcome,
            "common_causes": list(common_causes) + ["region FE"],
            "note": identification_note,
        },
        "robustness": robustness,
        "interpretation": _interpret(primary, significant),
    }
    logger.info(
        "Causal effect %s→%s: %.4f [%.4f, %.4f] via %s",
        treat_col,
        outcome,
        primary.effect,
        primary.ci_low,
        primary.ci_high,
        primary.engine,
    )
    return summary


def _interpret(result: CausalEffectResult, significant: bool) -> str:
    direction = "positive" if result.effect >= 0 else "negative"
    sig = "statistically distinguishable from 0" if significant else "not distinguishable from 0"
    return (
        f"Estimated causal effect of {result.treatment} on {result.outcome} is "
        f"{result.effect:.4f} ({int(result.confidence_level * 100)}% CI "
        f"[{result.ci_low:.4f}, {result.ci_high:.4f}]), {sig} at α={1 - result.confidence_level:.2f}. "
        f"Direction: {direction}. Engine: {result.engine}."
    )
