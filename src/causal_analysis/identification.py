"""
Causal identification helpers for DEIP.

Builds the analysis panel and documents the assumed causal graph:

    year, region ──► ai_innovation ──► dei
         │                  │
         └──────────────────┴──► dei

Because DEI mechanically includes the AI pillar (weight 20%), we also construct
``dei_ex_ai`` (leave-one-out composite) to estimate spillover onto other pillars.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import setup_logging

logger = setup_logging(__name__)

REQUIRED_PILLARS: List[str] = [
    "digital_infrastructure",
    "digital_industry",
    "ai_innovation",
    "digital_application",
]


def build_causal_panel(dei_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a clean causal panel from the DEI/pillar table.

    Args:
        dei_df: Output of ``IndexCalculator.calculate`` (must include pillars + dei).

    Returns:
        DataFrame with treatment, outcomes, confounders, and binary treatment flag.
    """
    missing = [c for c in ["year", "region", "dei", *REQUIRED_PILLARS] if c not in dei_df.columns]
    if missing:
        raise ValueError(f"Causal panel missing columns: {missing}")

    panel = dei_df[["year", "region", "dei", *REQUIRED_PILLARS]].dropna().copy()
    panel["year"] = panel["year"].astype(int)
    panel["region"] = panel["region"].astype(str)

    # Leave-one-out DEI: renormalize remaining pillar weights (0.3, 0.3, 0.2) → sum=0.8
    panel["dei_ex_ai"] = (
        0.30 / 0.80 * panel["digital_infrastructure"]
        + 0.30 / 0.80 * panel["digital_industry"]
        + 0.20 / 0.80 * panel["digital_application"]
    )

    median_ai = float(panel["ai_innovation"].median())
    panel["ai_high"] = (panel["ai_innovation"] >= median_ai).astype(int)
    panel["ai_innovation_z"] = (
        (panel["ai_innovation"] - panel["ai_innovation"].mean())
        / panel["ai_innovation"].std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan)

    logger.info(
        "Causal panel ready: n=%d, AI median=%.2f, ai_high rate=%.2f",
        len(panel),
        median_ai,
        panel["ai_high"].mean(),
    )
    return panel.reset_index(drop=True)


def causal_graph_gml(
    treatment: str = "ai_innovation",
    outcome: str = "dei",
    common_causes: Optional[Sequence[str]] = None,
) -> str:
    """
    Return a DoWhy-compatible DOT graph string.

    Args:
        treatment: Treatment node name.
        outcome: Outcome node name.
        common_causes: Backdoor adjustment set.
    """
    causes = list(common_causes or ["year", "digital_infrastructure", "digital_industry", "digital_application"])
    # Region entered as common cause when encoded numerically / via dummies upstream
    lines = ["digraph {"]
    for c in causes:
        lines.append(f'  {c} -> {treatment};')
        lines.append(f'  {c} -> {outcome};')
    lines.append(f'  {treatment} -> {outcome};')
    lines.append("}")
    return "\n".join(lines)
