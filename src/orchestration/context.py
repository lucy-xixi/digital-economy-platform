"""
Pipeline run context and lineage emission.

Provides a typed ``PipelineContext`` with ``run_id``, stage timing, artifact
hashes, and a JSON lineage manifest — the minimum observability surface expected
in a senior data-platform demo.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    CONTRACT_VERSION,
    DEI_WEIGHTS,
    LINEAGE_DIR,
    SCHEMA_VERSION,
    ensure_directories,
    setup_logging,
)

logger = setup_logging(__name__)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash() -> str:
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "dei_weights": DEI_WEIGHTS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class StageRecord:
    """Timing / status for one pipeline stage."""

    name: str
    started_at: str
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "RUNNING"
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """
    Mutable run context passed through pipeline stages.

    Attributes:
        run_id: Unique run identifier (UUID4 hex).
        started_at: ISO timestamp.
        stages: Ordered stage records.
        artifacts: Logical name → path (and optional hash).
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    status: str = "RUNNING"
    schema_version: str = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION
    config_hash: str = field(default_factory=_config_hash)
    stages: List[StageRecord] = field(default_factory=list)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_message: Optional[str] = None
    _stage_t0: Dict[str, float] = field(default_factory=dict, repr=False)

    def start_stage(self, name: str) -> None:
        """Mark a stage as started."""
        self._stage_t0[name] = time.perf_counter()
        self.stages.append(
            StageRecord(
                name=name,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        logger.info("[%s] stage START: %s", self.run_id[:8], name)

    def end_stage(self, name: str, status: str = "OK", **detail: Any) -> None:
        """Mark a stage as finished and attach detail metadata."""
        t0 = self._stage_t0.get(name, time.perf_counter())
        duration_ms = (time.perf_counter() - t0) * 1000.0
        for stage in reversed(self.stages):
            if stage.name == name and stage.finished_at is None:
                stage.finished_at = datetime.now(timezone.utc).isoformat()
                stage.duration_ms = round(duration_ms, 2)
                stage.status = status
                stage.detail = detail
                break
        logger.info(
            "[%s] stage END: %s status=%s (%.1f ms)",
            self.run_id[:8],
            name,
            status,
            duration_ms,
        )

    def register_artifact(self, name: str, path: Path) -> None:
        """Record an output artifact with content hash."""
        path = Path(path)
        self.artifacts[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "exists": path.exists(),
        }

    def finalize(self, status: str = "SUCCESS", error: Optional[str] = None) -> Path:
        """
        Close the run and write lineage JSON.

        Returns:
            Path to the lineage manifest.
        """
        self.status = status
        self.error_message = error
        self.finished_at = datetime.now(timezone.utc).isoformat()
        return self.write_lineage()

    def write_lineage(self) -> Path:
        """Persist lineage manifest under ``data/processed/lineage/``."""
        ensure_directories()
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "schema_version": self.schema_version,
            "contract_version": self.contract_version,
            "config_hash": self.config_hash,
            "error_message": self.error_message,
            "stages": [
                {
                    "name": s.name,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "detail": s.detail,
                }
                for s in self.stages
            ],
            "artifacts": self.artifacts,
        }
        path = LINEAGE_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s] lineage written → %s", self.run_id[:8], path)
        return path

    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for CLI / API responses."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [s.name for s in self.stages],
            "config_hash": self.config_hash,
        }
