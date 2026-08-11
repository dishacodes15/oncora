import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from nlp.interaction_store import InteractionStore, Interaction


@pytest.fixture
def store():
    path = tempfile.mktemp(suffix=".db")
    s = InteractionStore(path)
    yield s
    os.remove(path)


def test_upsert_and_get(store):
    store.upsert(Interaction("aspirin", "warfarin", "HIGH", "bleeding risk", "seed"))
    result = store.get("aspirin", "warfarin")
    assert result is not None
    assert result.severity == "HIGH"


def test_lookup_is_order_independent(store):
    store.upsert(Interaction("aspirin", "warfarin", "HIGH", "bleeding risk", "seed"))
    assert store.get("warfarin", "aspirin") is not None
    assert store.get("WARFARIN", "Aspirin") is not None


def test_miss_returns_none(store):
    assert store.get("aspirin", "metformin") is None


def test_upsert_overwrites_existing_pair(store):
    store.upsert(Interaction("a", "b", "LOW", "old description", "seed"))
    store.upsert(Interaction("a", "b", "HIGH", "new description", "openfda"))
    result = store.get("a", "b")
    assert result.severity == "HIGH"
    assert result.description == "new description"
    assert result.source == "openfda"


def test_low_confidence_pairs_filtered(store):
    store.upsert(Interaction("a", "b", "HIGH", "x", "seed", confidence="high"))
    store.upsert(Interaction("c", "d", "MEDIUM", "y", "openfda", confidence="low"))
    low_conf = store.low_confidence_pairs()
    assert len(low_conf) == 1
    assert low_conf[0].drug_a == "c"


def test_all_pairs_for_drug(store):
    store.upsert(Interaction("aspirin", "warfarin", "HIGH", "x", "seed"))
    store.upsert(Interaction("aspirin", "ibuprofen", "MEDIUM", "y", "seed"))
    store.upsert(Interaction("metformin", "furosemide", "LOW", "z", "seed"))
    pairs = store.all_pairs_for_drug("aspirin")
    assert len(pairs) == 2
