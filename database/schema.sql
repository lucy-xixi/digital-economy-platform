-- =============================================================================
-- Digital Economy Intelligence Platform (DEIP)
-- SQLite Schema Definition  (v1.1)
-- =============================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Pipeline run registry (lineage / audit)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'RUNNING',
    schema_version  TEXT,
    contract_version TEXT,
    config_hash     TEXT,
    lineage_path    TEXT,
    error_message   TEXT
);

-- ---------------------------------------------------------------------------
-- Raw / cleaned digital economy metrics by region and year
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS digital_metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    year                INTEGER NOT NULL CHECK (year BETWEEN 2010 AND 2100),
    region              TEXT    NOT NULL,
    internet_penetration REAL   CHECK (internet_penetration IS NULL OR (internet_penetration >= 0 AND internet_penetration <= 100)),
    five_g_users        REAL    CHECK (five_g_users IS NULL OR five_g_users >= 0),
    software_revenue    REAL    CHECK (software_revenue IS NULL OR software_revenue >= 0),
    ai_patents          REAL    CHECK (ai_patents IS NULL OR ai_patents >= 0),
    digital_payment     REAL    CHECK (digital_payment IS NULL OR (digital_payment >= 0 AND digital_payment <= 100)),
    ecommerce           REAL    CHECK (ecommerce IS NULL OR ecommerce >= 0),
    run_id              TEXT,
    updated_at          TEXT    DEFAULT (datetime('now')),
    created_at          TEXT    DEFAULT (datetime('now')),
    UNIQUE (year, region)
);

CREATE INDEX IF NOT EXISTS idx_metrics_year   ON digital_metrics (year);
CREATE INDEX IF NOT EXISTS idx_metrics_region ON digital_metrics (region);
-- idx_metrics_run is created in Python after column migration (legacy DB support)


-- ---------------------------------------------------------------------------
-- Primary indicator scores (0–100 after MinMax scaling)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS primary_indicators (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    year                    INTEGER NOT NULL,
    region                  TEXT    NOT NULL,
    digital_infrastructure  REAL,
    digital_industry        REAL,
    ai_innovation           REAL,
    digital_application     REAL,
    run_id                  TEXT,
    created_at              TEXT    DEFAULT (datetime('now')),
    UNIQUE (year, region)
);

-- ---------------------------------------------------------------------------
-- Composite Digital Economy Index (DEI)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dei_index (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    year            INTEGER NOT NULL,
    region          TEXT    NOT NULL,
    dei             REAL    NOT NULL CHECK (dei >= 0 AND dei <= 100),
    growth_rate     REAL,
    rank_in_year    INTEGER,
    run_id          TEXT,
    created_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE (year, region)
);

CREATE INDEX IF NOT EXISTS idx_dei_year   ON dei_index (year);
CREATE INDEX IF NOT EXISTS idx_dei_region ON dei_index (region);

-- ---------------------------------------------------------------------------
-- Data quality audit log
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_quality_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    report_ts       TEXT    NOT NULL DEFAULT (datetime('now')),
    total_rows      INTEGER,
    missing_count   INTEGER,
    outlier_count   INTEGER,
    duplicate_count INTEGER,
    quality_score   REAL,
    report_json     TEXT,
    status          TEXT    DEFAULT 'OK',
    gate_passed     INTEGER DEFAULT 1
);

-- ---------------------------------------------------------------------------
-- AI-generated research reports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query           TEXT    NOT NULL,
    report_content  TEXT    NOT NULL,
    report_json     TEXT,
    model_name      TEXT,
    grounded        INTEGER DEFAULT 1,
    created_at      TEXT    DEFAULT (datetime('now'))
);
