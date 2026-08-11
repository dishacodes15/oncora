import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from nlp.interaction_store import InteractionStore, Interaction
from nlp import interactions as interactions_mod


@pytest.fixture
def store():
    path = tempfile.mktemp(suffix=".db")
    s = InteractionStore(path)
    yield s
    os.remove(path)


def test_cache_hit_short_circuits_openfda(store, monkeypatch):
    store.upsert(Interaction("aspirin", "warfarin", "HIGH", "bleeding risk", "seed"))

    def fail_if_called(drug_name):
        raise AssertionError("OpenFDA should not be called on a cache hit")

    monkeypatch.setattr(interactions_mod, "_fetch_label_interaction_text", fail_if_called)

    result = interactions_mod.check_prescription({"aspirin", "warfarin"}, store)
    assert len(result.interactions) == 1
    assert result.interactions[0].source == "cache"
    assert result.risk_level == "HIGH"


def test_openfda_fallback_writes_back_to_cache(store, monkeypatch):
    def fake_fetch(drug_name):
        if drug_name == "metformin":
            return "Use with furosemide increases risk; monitor closely."
        return None

    monkeypatch.setattr(interactions_mod, "_fetch_label_interaction_text", fake_fetch)

    result = interactions_mod.check_prescription({"metformin", "furosemide"}, store)
    assert len(result.interactions) == 1
    assert result.interactions[0].source == "openfda"

    # Second call should now hit cache instead
    def fail_if_called(drug_name):
        raise AssertionError("Should have used cache on second call")

    monkeypatch.setattr(interactions_mod, "_fetch_label_interaction_text", fail_if_called)
    result2 = interactions_mod.check_prescription({"metformin", "furosemide"}, store)
    assert result2.interactions[0].source == "cache"


def test_unresolved_pair_not_treated_as_safe(store, monkeypatch):
    monkeypatch.setattr(interactions_mod, "_fetch_label_interaction_text", lambda d: None)

    result = interactions_mod.check_prescription({"drugx", "drugy"}, store)
    assert result.interactions == []
    assert ("drugx", "drugy") in result.unresolved_pairs
    # Absence of a known interaction must not be reported as LOW risk with
    # confidence — risk_level defaults to LOW only because nothing was
    # FOUND, and unresolved_pairs is what signals that distinction.
    assert result.risk_level == "LOW"


def test_single_drug_has_no_pairs(store):
    result = interactions_mod.check_prescription({"aspirin"}, store)
    assert result.interactions == []
    assert result.unresolved_pairs == []


def test_risk_level_reflects_worst_pair(store):
    store.upsert(Interaction("a", "b", "LOW", "x", "seed"))
    store.upsert(Interaction("a", "c", "HIGH", "y", "seed"))
    store.upsert(Interaction("b", "c", "MEDIUM", "z", "seed"))
    result = interactions_mod.check_prescription({"a", "b", "c"}, store)
    assert result.risk_level == "HIGH"
