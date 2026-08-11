"""
dose_limit_store.py
SQLite-backed local cache of per-drug dose limits - the dose-safety
equivalent of interaction_store.py. get() is the fast path; on a miss,
dose_safety.py falls through to OpenFDA + dose_prose_parser.py, and
every successful result gets written back here via upsert(), so the
store grows automatically instead of needing a hand-maintained CSV.
"""

import sqlite3
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("oncora.nlp.dose_limit_store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS dose_limits (
    drug              TEXT PRIMARY KEY,
    max_single_dose_mg TEXT,
    max_daily_dose_mg  TEXT,
    source            TEXT NOT NULL CHECK(source IN ('seed', 'openfda')),
    confidence        TEXT NOT NULL DEFAULT 'high' CHECK(confidence IN ('high', 'low')),
    evidence          TEXT,
    created_at        TEXT NOT NULL
);
"""


@dataclass
class DoseLimit:
    drug: str
    max_single_dose_mg: Optional[float]
    max_daily_dose_mg: Optional[float]
    source: str
    confidence: str = "high"
    evidence: str = ""


class DoseLimitStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, drug: str) -> Optional[DoseLimit]:
        d = drug.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dose_limits WHERE drug = ?", (d,)
            ).fetchone()
        if row is None:
            return None
        return DoseLimit(
            drug=row["drug"],
            max_single_dose_mg=float(row["max_single_dose_mg"]) if row["max_single_dose_mg"] is not None else None,
            max_daily_dose_mg=float(row["max_daily_dose_mg"]) if row["max_daily_dose_mg"] is not None else None,
            source=row["source"], confidence=row["confidence"], evidence=row["evidence"] or "",
        )

    def upsert(self, limit: DoseLimit) -> None:
        d = limit.drug.strip().lower()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dose_limits
                    (drug, max_single_dose_mg, max_daily_dose_mg, source, confidence, evidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drug) DO UPDATE SET
                    max_single_dose_mg=excluded.max_single_dose_mg,
                    max_daily_dose_mg=excluded.max_daily_dose_mg,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    evidence=excluded.evidence
                """,
                (d,
                 str(limit.max_single_dose_mg) if limit.max_single_dose_mg is not None else None,
                 str(limit.max_daily_dose_mg) if limit.max_daily_dose_mg is not None else None,
                 limit.source, limit.confidence, limit.evidence,
                 datetime.now(timezone.utc).isoformat()),
            )
        logger.info("Cached dose limit for %s [single=%s daily=%s, source=%s, confidence=%s]",
                    d, limit.max_single_dose_mg, limit.max_daily_dose_mg,
                    limit.source, limit.confidence)

    def low_confidence_limits(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dose_limits WHERE confidence = 'low' ORDER BY created_at DESC"
            ).fetchall()
        return [
            DoseLimit(
                drug=r["drug"],
                max_single_dose_mg=float(r["max_single_dose_mg"]) if r["max_single_dose_mg"] is not None else None,
                max_daily_dose_mg=float(r["max_daily_dose_mg"]) if r["max_daily_dose_mg"] is not None else None,
                source=r["source"], confidence=r["confidence"], evidence=r["evidence"] or "",
            )
            for r in rows
        ]
