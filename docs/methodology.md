# DEIP Methodology Notes

## Digital Economy Index (DEI)

DEI is a composite index on a 0–100 scale constructed from four primary pillars:

| Pillar | Weight | Metrics |
|--------|--------|---------|
| Digital Infrastructure | 30% | internet_penetration, 5g_users |
| Digital Industry | 30% | software_revenue, ecommerce |
| AI Innovation | 20% | ai_patents |
| Digital Application | 20% | digital_payment |

Construction steps: (1) MinMax normalize each metric across the full panel;
(2) average metrics within each pillar; (3) weighted sum → DEI;
(4) attach YoY growth and within-year rank.

Weights are a modelling choice. DEIP runs ±10% sensitivity shocks and reports
Spearman rank stability across scenarios.

## Panel correlation

Pooling all region-years treats observations as i.i.d. and inflates degrees of
freedom. DEIP also reports a **within** (region-demeaned) Pearson correlation,
closer to a fixed-effects interpretation of AI patents vs DEI.

## Forecasting

Per-region `dei ~ year` is estimated with **statsmodels OLS**. Forecasts include
95% prediction intervals. Walk-forward expanding-window MAE is reported as an
out-of-sample accuracy diagnostic. Durbin–Watson / Jarque–Bera / Breusch–Pagan
flags residual pathologies.

## Causal analysis (AI Innovation → DEI)

DEIP estimates the causal effect of the **AI Innovation Index** on the
**Digital Economy Index** using a backdoor adjustment strategy:

- Engine: **DoWhy** `backdoor.linear_regression` (fallback: statsmodels OLS +
  percentile bootstrap CI; optional **EconML** `LinearDML`)
- Outputs: point estimate, 95% confidence interval, significance vs 0
- Identification note: full DEI mechanically includes the AI pillar (weight 20%),
  so the preferred spillover check uses leave-one-out `dei_ex_ai`
- Robustness: binary above-median AI ATE; optional DML

See `src/causal_analysis/` and Dashboard page **Causal Analysis**.
