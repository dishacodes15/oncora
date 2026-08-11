"""
sanity_check.py
Randomized, repeatable end-to-end sanity check of the prescription NLP
pipeline — same spirit as federated/server.py's round-based accuracy
tracking. Each round draws a random sample of KNOWN drug-pair
interactions (from data/seed_interactions.csv, our ground truth),
synthesizes realistic prescription text for each pair with randomized
phrasing/dosage, runs it through the ACTUAL parser + interaction store
(no mocks), and checks whether the drugs were correctly extracted AND
correctly classified against the known-correct severity.

Results are appended to docs/nlp_eval_history.json as a growing list of
rounds — parallel to docs/fl_round_history.json — so accuracy over time
can be tracked and visualized the same way the Federated Monitor page
visualizes FL round accuracy. See frontend/views/nlp_sanity_page.py.

This is different from scripts/evaluate_parser.py and
evaluate_severity.py: those run a FIXED hand-labeled set once. This runs
a NEW random sample every time, against the real store on disk, and
keeps a history — so it catches regressions/drift over time and gives
you a repeatable spot-check rather than a one-off score.

Run:
    python -m scripts.sanity_check
    python -m scripts.sanity_check --samples 20
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.nlp.parser import get_parser  # noqa: E402
from backend.nlp.interaction_store import InteractionStore  # noqa: E402
from backend.nlp.interactions import check_prescription  # noqa: E402

DEFAULT_VOCAB_PATH = str(PROJECT_ROOT / "data" / "drug_vocab.json")
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "interactions.db")
DEFAULT_SEED_CSV = str(PROJECT_ROOT / "data" / "seed_interactions.csv")
DEFAULT_HISTORY_PATH = str(PROJECT_ROOT / "docs" / "nlp_eval_history.json")

# Randomized phrasing templates — each round synthesizes different
# wording/dosages for the same drug pairs, so this isn't just re-running
# a fixed string every time.
TEMPLATES = [
    "{drug} {dose}mg once daily.",
    "Patient is taking {drug} {dose}mg twice a day.",
    "Started on {drug} ({dose} mg) last week.",
    "{drug} {dose} mg as needed.",
    "Continue {drug} {dose}mg at bedtime.",
]
DOSES = [5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 325, 400, 500]


def _load_seed_pairs(seed_csv_path: str) -> list:
    pairs = []
    with open(seed_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "drug_a": row["drug_a"].strip().lower(),
                "drug_b": row["drug_b"].strip().lower(),
                "severity": row["severity"].strip().upper(),
            })
    return pairs


def _synthesize_text(drug_a: str, drug_b: str, rng: random.Random) -> str:
    sent_a = rng.choice(TEMPLATES).format(drug=drug_a, dose=rng.choice(DOSES))
    sent_b = rng.choice(TEMPLATES).format(drug=drug_b, dose=rng.choice(DOSES))
    return f"{sent_a} {sent_b}"


def run_round(
    sample_size: int = 10,
    vocab_path: str = None,
    db_path: str = None,
    seed_csv_path: str = None,
    model_name: str = None,
    rng_seed: int = None,
    parser=None,
) -> dict:
    """
    Draws `sample_size` random known interaction pairs (sampled with
    replacement, so sample_size can exceed the number of seed pairs),
    runs each through the real parser + interaction store, and returns
    a round-result dict ready to append to the history file.

    rng_seed: leave as None for genuine randomness each call (the normal
    case). Pass a fixed int only for reproducible testing of this script
    itself.

    parser: optional pre-built PrescriptionParser. Normal callers should
    leave this as None, which uses the process-wide cached singleton via
    get_parser() (the same one main.py builds once at startup). This
    parameter exists so tests can exercise different vocab_path values
    within a single process — get_parser()'s singleton caching means a
    second call with a different vocab_path would otherwise silently
    keep returning the first-built parser.
    """
    vocab_path = vocab_path or os.environ.get("DRUG_VOCAB_PATH", DEFAULT_VOCAB_PATH)
    db_path = db_path or os.environ.get("INTERACTIONS_DB_PATH", DEFAULT_DB_PATH)
    seed_csv_path = seed_csv_path or os.environ.get("SEED_INTERACTIONS_CSV", DEFAULT_SEED_CSV)
    model_name = model_name or os.environ.get("SPACY_MODEL", "en_core_sci_sm")

    rng = random.Random(rng_seed)

    if parser is None:
        parser = get_parser(vocab_path=vocab_path, model_name=model_name)
    store = InteractionStore(db_path)
    seed_pairs = _load_seed_pairs(seed_csv_path)

    if not seed_pairs:
        raise RuntimeError(f"No ground-truth pairs found in {seed_csv_path}")

    sample = [rng.choice(seed_pairs) for _ in range(sample_size)]

    n_pass = 0
    extraction_pass = 0
    severity_checked = 0
    severity_pass = 0
    failures = []

    for case in sample:
        drug_a, drug_b, expected_severity = case["drug_a"], case["drug_b"], case["severity"]
        text = _synthesize_text(drug_a, drug_b, rng)

        parsed = parser.parse(text)
        extraction_ok = drug_a in parsed.drugs and drug_b in parsed.drugs

        if not extraction_ok:
            missing = [d for d in (drug_a, drug_b) if d not in parsed.drugs]
            failures.append({
                "drug_a": drug_a, "drug_b": drug_b, "issue": "extraction_miss",
                "missing": missing, "text": text,
            })
            continue

        extraction_pass += 1
        result = check_prescription(parsed.drugs, store)

        found = next(
            (i for i in result.interactions if {i.drug_a, i.drug_b} == {drug_a, drug_b}),
            None,
        )

        if found is None:
            failures.append({
                "drug_a": drug_a, "drug_b": drug_b, "issue": "unexpected_cache_miss",
                "expected_severity": expected_severity, "text": text,
                "note": "known seed pair not found in store — was scripts.init_db run against this db_path?",
            })
            continue

        severity_checked += 1
        if found.severity == expected_severity:
            severity_pass += 1
            n_pass += 1
        else:
            failures.append({
                "drug_a": drug_a, "drug_b": drug_b, "issue": "severity_mismatch",
                "expected_severity": expected_severity, "actual_severity": found.severity,
                "text": text,
            })

    round_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": sample_size,
        "accuracy": round(n_pass / sample_size, 4) if sample_size else 0.0,
        "drug_extraction_accuracy": round(extraction_pass / sample_size, 4) if sample_size else 0.0,
        "severity_accuracy": round(severity_pass / severity_checked, 4) if severity_checked else None,
        "failures": failures[:10],
    }
    return round_result


def save_round(round_result: dict, history_path: str = None) -> str:
    history_path = history_path or os.environ.get("NLP_EVAL_HISTORY_PATH", DEFAULT_HISTORY_PATH)
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)

    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(round_result)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    return history_path


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Randomized sanity check for the prescription NLP pipeline."
    )
    arg_parser.add_argument("--samples", type=int, default=10,
                             help="Number of random drug-pair samples this round.")
    args = arg_parser.parse_args()

    result = run_round(sample_size=args.samples)
    saved_path = save_round(result)

    print(f"\nRound complete — {result['n_samples']} samples")
    print(f"  Overall accuracy:          {result['accuracy']:.1%}")
    print(f"  Drug extraction accuracy:  {result['drug_extraction_accuracy']:.1%}")
    if result["severity_accuracy"] is not None:
        print(f"  Severity accuracy:         {result['severity_accuracy']:.1%}")
    if result["failures"]:
        print(f"\n  {len(result['failures'])} failure(s):")
        for f in result["failures"]:
            print(f"    - {f}")
    print(f"\nSaved to {saved_path}")
