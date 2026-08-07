"""
Digital Economy Index (DEI) indicator framework.

Defines the four primary pillars, weight schema, and metric → pillar mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.config import DEI_WEIGHTS, INDICATOR_METRICS, setup_logging

logger = setup_logging(__name__)


@dataclass(frozen=True)
class PrimaryIndicator:
    """A first-level (pillar) indicator in the DEI hierarchy."""

    name: str
    weight: float
    metrics: List[str]
    description: str


class IndicatorFramework:
    """
    Hierarchical indicator system for the Digital Economy Index.

    Structure
    ---------
    DEI (100%)
    ├── Digital Infrastructure  (30%)  ← internet_penetration, 5g_users
    ├── Digital Industry        (30%)  ← software_revenue, ecommerce
    ├── AI Innovation           (20%)  ← ai_patents
    └── Digital Application     (20%)  ← digital_payment
    """

    PILLAR_DESCRIPTIONS: Dict[str, str] = {
        "digital_infrastructure": (
            "Connectivity and network readiness — internet penetration and 5G adoption."
        ),
        "digital_industry": (
            "Scale of digital industry — software revenue and e-commerce GMV."
        ),
        "ai_innovation": (
            "AI research & IP intensity — AI-related patent grants."
        ),
        "digital_application": (
            "Digitalization of daily life — digital payment penetration."
        ),
    }

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        """
        Args:
            weights: Optional custom pillar weights (must sum to ~1.0).
        """
        self.weights = dict(weights or DEI_WEIGHTS)
        self._validate_weights()
        self.pillars: Dict[str, PrimaryIndicator] = {
            name: PrimaryIndicator(
                name=name,
                weight=self.weights[name],
                metrics=list(INDICATOR_METRICS[name]),
                description=self.PILLAR_DESCRIPTIONS[name],
            )
            for name in self.weights
        }
        logger.info(
            "IndicatorFramework initialized with %d pillars: %s",
            len(self.pillars),
            {k: v.weight for k, v in self.pillars.items()},
        )

    def _validate_weights(self) -> None:
        """Ensure weights are non-negative and sum to approximately 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"DEI weights must sum to 1.0, got {total}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("DEI weights must be non-negative")

    def list_pillars(self) -> List[str]:
        """Return ordered pillar names."""
        return list(self.pillars.keys())

    def get_metrics(self, pillar: str) -> List[str]:
        """Return underlying metrics for a pillar."""
        if pillar not in self.pillars:
            raise KeyError(f"Unknown pillar: {pillar}")
        return list(self.pillars[pillar].metrics)

    def describe(self) -> str:
        """Human-readable framework summary."""
        lines = ["Digital Economy Index (DEI) Framework", "-" * 40]
        for name, pillar in self.pillars.items():
            lines.append(
                f"  [{pillar.weight * 100:.0f}%] {name}: {', '.join(pillar.metrics)}"
            )
            lines.append(f"         {pillar.description}")
        return "\n".join(lines)
