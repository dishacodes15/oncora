"""
init_db.py
Run once (or any time you want to reset/reseed): creates the SQLite
interactions table if it doesn't exist, and loads the curated seed CSV
into it. Safe to rerun — upsert means seed rows just get refreshed,
not duplicated.

    python -m scripts.init_db
"""

import csv
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.nlp.interaction_store import InteractionStore, Interaction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oncora.scripts.init_db")


def load_seed_csv(store: InteractionStore, csv_path: str) -> int:
    count = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            store.upsert(Interaction(
                drug_a=row["drug_a"],
                drug_b=row["drug_b"],
                severity=row["severity"].strip().upper(),
                description=row["description"].strip(),
                source="seed",
                confidence="high",   # seed data is hand-curated, not machine-parsed
            ))
            count += 1
    return count


if __name__ == "__main__":
    db_path = os.environ.get("INTERACTIONS_DB_PATH", "data/interactions.db")
    seed_csv = os.environ.get("SEED_INTERACTIONS_CSV", "data/seed_interactions.csv")

    store = InteractionStore(db_path)
    n = load_seed_csv(store, seed_csv)
    logger.info("Seeded %d interactions into %s", n, db_path)
