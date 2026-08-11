"""
interaction_store.py
SQLite-backed local cache of drug-pair interactions. This replaces the
static 21-row CSV as the fast path: check_interactions() (in
interactions.py) checks this table FIRST for a given pair, and only
falls through to OpenFDA on a miss.

Every OpenFDA-derived interaction we successfully parse gets written back
here, so the store grows and repeat lookups for the same pair become a
single indexed SQLite read instead of a network round-trip.

Design notes:
- Pairs are stored with a canonical (sorted) key so (aspirin, warfarin)
  and (warfarin, aspirin) hit the same row — interaction direction
  doesn't matter clinically, and we don't want duplicate rows or a
  cache miss just because the drugs were listed in a different order.
- `source` distinguishes seed data (curated at launch) from
  openfda-derived rows, so you can audit / re-verify machine-parsed
  entries separately from hand-checked ones.
- `confidence` records whether severity parsing (step 7) was confident
  or should be flagged for manual review — see severity.py.
"""

import sqlite3
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("oncora.nlp.interaction_store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_a        TEXT NOT NULL,       -- canonical: drug_a < drug_b alphabetically
    drug_b        TEXT NOT NULL,
    severity      TEXT NOT NULL CHECK(severity IN ('HIGH', 'MEDIUM', 'LOW')),
    description   TEXT NOT NULL,
    source        TEXT NOT NULL CHECK(source IN ('seed', 'openfda')),
    confidence    TEXT NOT NULL DEFAULT 'high' CHECK(confidence IN ('high', 'low')),
    created_at    TEXT NOT NULL,
    UNIQUE(drug_a, drug_b)
);
CREATE INDEX IF NOT EXISTS idx_interactions_pair ON interactions(drug_a, drug_b);
"""


@dataclass
class Interaction:
    drug_a: str
    drug_b: str
    severity: str          # HIGH | MEDIUM | LOW
    description: str
    source: str             # seed | openfda
    confidence: str = "high"  # high | low — see severity.py


def _canonical_pair(drug_a: str, drug_b: str) -> tuple:
    """Sort alphabetically so lookups are order-independent."""
    a, b = drug_a.strip().lower(), drug_b.strip().lower()
    return (a, b) if a <= b else (b, a)


class InteractionStore:
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

    def get(self, drug_a: str, drug_b: str) -> Optional[Interaction]:
        """Fast-path cache lookup. Returns None on a miss — caller should
        then try the OpenFDA fallback."""
        a, b = _canonical_pair(drug_a, drug_b)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM interactions WHERE drug_a = ? AND drug_b = ?",
                (a, b),
            ).fetchone()
        if row is None:
            return None
        return Interaction(
            drug_a=row["drug_a"], drug_b=row["drug_b"], severity=row["severity"],
            description=row["description"], source=row["source"],
            confidence=row["confidence"],
        )

    def upsert(self, interaction: Interaction) -> None:
        """Write (or overwrite) a pair's interaction. Used both for seeding
        the store at startup and for writing back OpenFDA-derived hits
        (step 8's 'self-growing' behavior)."""
        a, b = _canonical_pair(interaction.drug_a, interaction.drug_b)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions
                    (drug_a, drug_b, severity, description, source, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drug_a, drug_b) DO UPDATE SET
                    severity=excluded.severity,
                    description=excluded.description,
                    source=excluded.source,
                    confidence=excluded.confidence
                """,
                (a, b, interaction.severity, interaction.description,
                 interaction.source, interaction.confidence,
                 datetime.now(timezone.utc).isoformat()),
            )
        logger.info("Cached interaction %s + %s [%s, %s, source=%s]",
                    a, b, interaction.severity, interaction.confidence, interaction.source)

    def all_pairs_for_drug(self, drug: str) -> list:
        """All cached interactions involving a given drug — handy for
        admin/debug endpoints and for the manual-review queue."""
        d = drug.strip().lower()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE drug_a = ? OR drug_b = ?",
                (d, d),
            ).fetchall()
        return [
            Interaction(drug_a=r["drug_a"], drug_b=r["drug_b"], severity=r["severity"],
                        description=r["description"], source=r["source"],
                        confidence=r["confidence"])
            for r in rows
        ]

    def low_confidence_pairs(self) -> list:
        """Everything flagged for manual review (step 7's low-confidence
        severity extractions). Surface this in an admin view — don't
        silently trust machine-parsed severity on health-adjacent data."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE confidence = 'low' ORDER BY created_at DESC"
            ).fetchall()
        return [
            Interaction(drug_a=r["drug_a"], drug_b=r["drug_b"], severity=r["severity"],
                        description=r["description"], source=r["source"],
                        confidence=r["confidence"])
            for r in rows
        ]
