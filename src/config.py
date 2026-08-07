"""
DEIP global configuration.

Centralizes paths, indicator weights, DQ gates, regions, years, and logging.
Loads optional ``.env`` via python-dotenv when available.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Path resolution (project root = parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
LINEAGE_DIR: Path = DATA_PROCESSED_DIR / "lineage"
DATABASE_DIR: Path = PROJECT_ROOT / "database"
SCHEMA_PATH: Path = DATABASE_DIR / "schema.sql"
DB_PATH: Path = DATABASE_DIR / "deip.db"
LOG_DIR: Path = PROJECT_ROOT / "logs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
METHODOLOGY_PATH: Path = DOCS_DIR / "methodology.md"

RAW_CSV_PATH: Path = DATA_RAW_DIR / "digital_economy_metrics.csv"
PROCESSED_CSV_PATH: Path = DATA_PROCESSED_DIR / "digital_economy_clean.csv"
DEI_CSV_PATH: Path = DATA_PROCESSED_DIR / "dei_index.csv"
QUALITY_REPORT_PATH: Path = DATA_PROCESSED_DIR / "data_quality_report.json"
FORECAST_CSV_PATH: Path = DATA_PROCESSED_DIR / "dei_forecast.csv"

# Schema / contract versioning
SCHEMA_VERSION: str = "1.1.0"
CONTRACT_VERSION: str = "metrics_v1"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
REGIONS: List[str] = [
    "Beijing",
    "Shanghai",
    "Guangdong",
    "Zhejiang",
    "Jiangsu",
]

YEAR_START: int = 2018
YEAR_END: int = 2025
FORECAST_START: int = 2026
FORECAST_END: int = 2030

METRIC_COLUMNS: List[str] = [
    "internet_penetration",
    "5g_users",
    "software_revenue",
    "ai_patents",
    "digital_payment",
    "ecommerce",
]

INDICATOR_METRICS: Dict[str, List[str]] = {
    "digital_infrastructure": ["internet_penetration", "5g_users"],
    "digital_industry": ["software_revenue", "ecommerce"],
    "ai_innovation": ["ai_patents"],
    "digital_application": ["digital_payment"],
}

DEI_WEIGHTS: Dict[str, float] = {
    "digital_infrastructure": 0.30,
    "digital_industry": 0.30,
    "ai_innovation": 0.20,
    "digital_application": 0.20,
}

# Data quality gate thresholds (hard fail if breached)
DQ_MIN_SCORE: float = float(os.getenv("DEIP_DQ_MIN_SCORE", "60"))
DQ_MAX_MISSING_RATE: float = float(os.getenv("DEIP_DQ_MAX_MISSING_RATE", "0.15"))
DQ_REQUIRE_UNIQUE_KEYS: bool = True
DQ_BLOCK_ON_FAIL: bool = os.getenv("DEIP_DQ_BLOCK_ON_FAIL", "true").lower() in {
    "1",
    "true",
    "yes",
}

# OpenAI / LangChain
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-demo-placeholder-key")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_MOCK_LLM: bool = os.getenv("DEIP_USE_MOCK_LLM", "true").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings snapshot."""

    project_root: Path = PROJECT_ROOT
    db_path: Path = DB_PATH
    regions: List[str] = field(default_factory=lambda: list(REGIONS))
    year_start: int = YEAR_START
    year_end: int = YEAR_END
    dei_weights: Dict[str, float] = field(default_factory=lambda: dict(DEI_WEIGHTS))
    use_mock_llm: bool = USE_MOCK_LLM
    openai_model: str = OPENAI_MODEL
    dq_min_score: float = DQ_MIN_SCORE
    dq_block_on_fail: bool = DQ_BLOCK_ON_FAIL
    schema_version: str = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION


settings = Settings()


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    for path in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        LINEAGE_DIR,
        DATABASE_DIR,
        LOG_DIR,
        DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "deip", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a module-level logger.

    Args:
        name: Logger name (typically ``__name__`` of the caller).
        level: Logging level.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_DIR / "deip.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
