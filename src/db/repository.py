"""
SQLite repository layer with transactional upserts.

Replaces truncate-and-reload with ``INSERT … ON CONFLICT DO UPDATE`` and
persists pipeline runs, DQ reports, and DEI artifacts under a single run_id.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

import pandas as pd

from src.config import DB_PATH, SCHEMA_PATH, ensure_directories, setup_logging

logger = setup_logging(__name__)


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return column names for an existing table (empty set if missing)."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchall()
    if not rows:
        return set()
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_ddl: str) -> None:
    """ADD COLUMN if missing (SQLite cannot IF NOT EXISTS on ADD COLUMN pre-3.35 reliably)."""
    cols = _existing_columns(conn, table)
    if not cols:
        return
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_ddl}")
        logger.info("Migrated %s: added column %s", table, column)


def migrate_schema(conn: sqlite3.Connection) -> None:
    """
    Bring legacy DEIP databases up to schema v1.1.

    Older Demo DBs were created without ``run_id`` / ``report_json`` / ``gate_passed``.
    ``CREATE TABLE IF NOT EXISTS`` does not alter them, so we patch columns explicitly.
    """
    _ensure_column(conn, "digital_metrics", "run_id", "run_id TEXT")
    _ensure_column(conn, "digital_metrics", "updated_at", "updated_at TEXT")
    _ensure_column(conn, "primary_indicators", "run_id", "run_id TEXT")
    _ensure_column(conn, "dei_index", "run_id", "run_id TEXT")
    _ensure_column(conn, "data_quality_reports", "run_id", "run_id TEXT")
    _ensure_column(conn, "data_quality_reports", "gate_passed", "gate_passed INTEGER DEFAULT 1")
    _ensure_column(conn, "ai_reports", "report_json", "report_json TEXT")
    _ensure_column(conn, "ai_reports", "grounded", "grounded INTEGER DEFAULT 1")

    # Safe to create after run_id exists
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_run ON digital_metrics (run_id)"
    )


def init_database(db_path: Optional[Path] = None) -> Path:
    """Create / migrate SQLite schema."""
    ensure_directories()
    db_path = Path(db_path or DB_PATH)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        migrate_schema(conn)
        conn.commit()
    logger.info("Database ready at %s", db_path)
    return db_path


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with foreign keys enabled."""
    db_path = db_path or DB_PATH
    if not Path(db_path).exists():
        init_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class DeipRepository:
    """
    Persistence facade for DEIP warehouse tables.

    All writes are keyed by ``run_id`` for auditability.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH
        init_database(self.db_path)

    def start_run(self, run_id: str, started_at: str, schema_version: str, contract_version: str, config_hash: str) -> None:
        """Insert a RUNNING pipeline_runs row."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs
                (run_id, started_at, status, schema_version, contract_version, config_hash)
                VALUES (?, ?, 'RUNNING', ?, ?, ?)
                """,
                (run_id, started_at, schema_version, contract_version, config_hash),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        finished_at: str,
        lineage_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update pipeline_runs status."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE pipeline_runs
                SET status=?, finished_at=?, lineage_path=?, error_message=?
                WHERE run_id=?
                """,
                (status, finished_at, lineage_path, error_message, run_id),
            )

    def upsert_metrics(self, df: pd.DataFrame, run_id: str) -> int:
        """Upsert cleaned metrics (maps ``5g_users`` → ``five_g_users``)."""
        upload = df.rename(columns={"5g_users": "five_g_users"}).copy()
        cols = [
            "year",
            "region",
            "internet_penetration",
            "five_g_users",
            "software_revenue",
            "ai_patents",
            "digital_payment",
            "ecommerce",
        ]
        upload = upload[cols]
        sql = """
            INSERT INTO digital_metrics (
                year, region, internet_penetration, five_g_users,
                software_revenue, ai_patents, digital_payment, ecommerce, run_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(year, region) DO UPDATE SET
                internet_penetration=excluded.internet_penetration,
                five_g_users=excluded.five_g_users,
                software_revenue=excluded.software_revenue,
                ai_patents=excluded.ai_patents,
                digital_payment=excluded.digital_payment,
                ecommerce=excluded.ecommerce,
                run_id=excluded.run_id,
                updated_at=datetime('now')
        """
        rows = [
            (
                int(r.year),
                str(r.region),
                float(r.internet_penetration),
                float(r.five_g_users),
                float(r.software_revenue),
                float(r.ai_patents),
                float(r.digital_payment),
                float(r.ecommerce),
                run_id,
            )
            for r in upload.itertuples(index=False)
        ]
        with get_connection(self.db_path) as conn:
            conn.executemany(sql, rows)
        logger.info("Upserted %d digital_metrics rows (run=%s)", len(rows), run_id[:8])
        return len(rows)

    def upsert_primary_and_dei(
        self,
        primary: pd.DataFrame,
        dei: pd.DataFrame,
        run_id: str,
    ) -> None:
        """Transactionally upsert primary indicators and DEI index."""
        primary_sql = """
            INSERT INTO primary_indicators (
                year, region, digital_infrastructure, digital_industry,
                ai_innovation, digital_application, run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(year, region) DO UPDATE SET
                digital_infrastructure=excluded.digital_infrastructure,
                digital_industry=excluded.digital_industry,
                ai_innovation=excluded.ai_innovation,
                digital_application=excluded.digital_application,
                run_id=excluded.run_id
        """
        dei_sql = """
            INSERT INTO dei_index (
                year, region, dei, growth_rate, rank_in_year, run_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(year, region) DO UPDATE SET
                dei=excluded.dei,
                growth_rate=excluded.growth_rate,
                rank_in_year=excluded.rank_in_year,
                run_id=excluded.run_id
        """
        p_cols = [
            "year",
            "region",
            "digital_infrastructure",
            "digital_industry",
            "ai_innovation",
            "digital_application",
        ]
        d_cols = ["year", "region", "dei", "growth_rate", "rank_in_year"]

        p_rows = [
            (
                int(r.year),
                str(r.region),
                float(r.digital_infrastructure),
                float(r.digital_industry),
                float(r.ai_innovation),
                float(r.digital_application),
                run_id,
            )
            for r in primary[p_cols].itertuples(index=False)
        ]
        d_rows = [
            (
                int(r.year),
                str(r.region),
                float(r.dei),
                None if pd.isna(r.growth_rate) else float(r.growth_rate),
                int(r.rank_in_year),
                run_id,
            )
            for r in dei[d_cols].itertuples(index=False)
        ]
        with get_connection(self.db_path) as conn:
            conn.executemany(primary_sql, p_rows)
            conn.executemany(dei_sql, d_rows)
        logger.info(
            "Upserted primary=%d dei=%d (run=%s)",
            len(p_rows),
            len(d_rows),
            run_id[:8],
        )

    def save_quality_report(self, report: Dict[str, Any], run_id: str, gate_passed: bool) -> None:
        """Insert a data quality audit row."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_quality_reports (
                    run_id, total_rows, missing_count, outlier_count,
                    duplicate_count, quality_score, report_json, status, gate_passed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    report.get("total_rows"),
                    report.get("completeness", {}).get("missing_total"),
                    report.get("outliers", {}).get("outlier_total"),
                    report.get("uniqueness", {}).get("duplicate_keys"),
                    report.get("quality_score"),
                    json.dumps(report, ensure_ascii=False, default=str),
                    report.get("status"),
                    1 if gate_passed else 0,
                ),
            )
        logger.info("DQ report persisted (gate_passed=%s)", gate_passed)

    def save_ai_report(
        self,
        query: str,
        content: str,
        model_name: str,
        report_json: Optional[Dict[str, Any]] = None,
        grounded: bool = True,
    ) -> None:
        """Persist an AI research brief."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ai_reports (query, report_content, report_json, model_name, grounded)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query,
                    content,
                    json.dumps(report_json, ensure_ascii=False, default=str) if report_json else None,
                    model_name,
                    1 if grounded else 0,
                ),
            )

    def read_dei(self) -> pd.DataFrame:
        """Read DEI index from SQLite (system of record)."""
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT year, region, dei, growth_rate, rank_in_year FROM dei_index ORDER BY year, region",
                conn,
            )

    def read_metrics(self) -> pd.DataFrame:
        """Read metrics from SQLite, renaming five_g_users → 5g_users."""
        with get_connection(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT year, region, internet_penetration, five_g_users AS "5g_users",
                       software_revenue, ai_patents, digital_payment, ecommerce
                FROM digital_metrics ORDER BY year, region
                """,
                conn,
            )
        return df
