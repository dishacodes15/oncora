import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.nlp.interaction_store import InteractionStore, Interaction
from backend.nlp import interactions as interactions_mod
from scripts import sanity_check

# Same convention as test_parser.py — respect SPACY_MODEL_TEST if set (e.g.
# to en_core_sci_sm, the real production model) rather than hardcoding a
# specific model that may not be installed in every environment.
TEST_MODEL = os.environ.get("SPACY_MODEL_TEST", "en_core_web_sm")


@pytest.fixture
def temp_seed_csv():
    path = tempfile.mktemp(suffix=".csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["drug_a", "drug_b", "severity", "description"])
        writer.writeheader()
        writer.writerow({"drug_a": "aspirin", "drug_b": "warfarin",
                          "severity": "HIGH", "description": "bleeding risk"})
        writer.writerow({"drug_a": "metformin", "drug_b": "furosemide",
                          "severity": "MEDIUM", "description": "renal risk"})
    yield path
    os.remove(path)


@pytest.fixture
def temp_db_seeded(temp_seed_csv):
    db_path = tempfile.mktemp(suffix=".db")
    store = InteractionStore(db_path)
    for row in sanity_check._load_seed_pairs(temp_seed_csv):
        store.upsert(Interaction(row["drug_a"], row["drug_b"], row["severity"],
                                  "test description", "seed"))
    yield db_path
    os.remove(db_path)


@pytest.fixture
def temp_vocab():
    path = tempfile.mktemp(suffix=".json")
    with open(path, "w") as f:
        json.dump(["aspirin", "warfarin", "metformin", "furosemide"], f)
    yield path
    os.remove(path)


def test_round_passes_against_correctly_seeded_store(temp_vocab, temp_db_seeded, temp_seed_csv):
    result = sanity_check.run_round(
        sample_size=6, vocab_path=temp_vocab, db_path=temp_db_seeded,
        seed_csv_path=temp_seed_csv, model_name=TEST_MODEL, rng_seed=42,
    )
    assert result["accuracy"] == 1.0
    assert result["drug_extraction_accuracy"] == 1.0
    assert result["severity_accuracy"] == 1.0
    assert result["failures"] == []


def test_round_detects_unseeded_store(temp_vocab, temp_seed_csv, monkeypatch):
    """If the store hasn't been seeded, known pairs should surface as
    unexpected_cache_miss failures, not silently pass.

    Monkeypatches the OpenFDA fetch to always return None, so this test's
    outcome doesn't depend on live network reachability or OpenFDA's real
    current label data for these two drug pairs — without this, the test
    could pass or fail differently depending on whether the machine
    running it can actually reach api.fda.gov, which is exactly the kind
    of non-determinism a unit test shouldn't have."""
    monkeypatch.setattr(interactions_mod, "_fetch_label_interaction_text", lambda d: None)

    empty_db = tempfile.mktemp(suffix=".db")
    InteractionStore(empty_db)  # creates schema, no rows

    result = sanity_check.run_round(
        sample_size=4, vocab_path=temp_vocab, db_path=empty_db,
        seed_csv_path=temp_seed_csv, model_name=TEST_MODEL, rng_seed=1,
    )
    assert result["accuracy"] == 0.0
    assert result["drug_extraction_accuracy"] == 1.0  # extraction still works
    assert all(f["issue"] == "unexpected_cache_miss" for f in result["failures"])
    os.remove(empty_db)


def test_round_detects_extraction_miss(temp_db_seeded, temp_seed_csv):
    """If the vocab is missing a term, that pair should surface as an
    extraction_miss, and severity should never be checked for it.

    Builds its own PrescriptionParser directly rather than going through
    get_parser() — that function caches a process-wide singleton (by
    design, so production doesn't reload the spaCy model per request),
    which would otherwise silently keep returning whichever vocab an
    earlier test/fixture in this process built first."""
    from backend.nlp.parser import PrescriptionParser

    sparse_vocab = tempfile.mktemp(suffix=".json")
    json.dump(["aspirin"], open(sparse_vocab, "w"))  # missing warfarin, metformin, furosemide
    sparse_parser = PrescriptionParser(vocab_path=sparse_vocab, model_name=TEST_MODEL)

    result = sanity_check.run_round(
        sample_size=4, db_path=temp_db_seeded, seed_csv_path=temp_seed_csv,
        rng_seed=7, parser=sparse_parser,
    )
    assert result["drug_extraction_accuracy"] < 1.0
    assert any(f["issue"] == "extraction_miss" for f in result["failures"])
    os.remove(sparse_vocab)


def test_save_round_appends_to_history():
    history_path = tempfile.mktemp(suffix=".json")
    r1 = {"timestamp": "t1", "n_samples": 5, "accuracy": 1.0,
          "drug_extraction_accuracy": 1.0, "severity_accuracy": 1.0, "failures": []}
    r2 = {"timestamp": "t2", "n_samples": 5, "accuracy": 0.8,
          "drug_extraction_accuracy": 1.0, "severity_accuracy": 0.8, "failures": []}

    sanity_check.save_round(r1, history_path)
    sanity_check.save_round(r2, history_path)

    with open(history_path) as f:
        history = json.load(f)
    assert len(history) == 2
    assert history[0]["timestamp"] == "t1"
    assert history[1]["timestamp"] == "t2"
    os.remove(history_path)
