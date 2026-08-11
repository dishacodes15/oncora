import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from nlp.dose_limit_store import DoseLimitStore, DoseLimit
from nlp.dose_safety import check_all_doses, load_dose_limits_csv
from nlp.parser import DosageMatch

REAL_SEED_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "dose_limits.csv"
)


@pytest.fixture
def store():
    path = tempfile.mktemp(suffix=".db")
    s = DoseLimitStore(path)
    for limit in load_dose_limits_csv(REAL_SEED_CSV):
        s.upsert(limit)
    yield s
    os.remove(path)


def test_seed_csv_loads(store):
    limit = store.get("aspirin")
    assert limit is not None
    assert limit.max_daily_dose_mg == 4000
    assert limit.source == "seed"


def test_gross_overdose_flagged_high(store):
    """Regression test for the exact real-world case that surfaced this
    gap: 'Aspirin 100g' (100,000mg) against a 4000mg/day maximum."""
    alerts, unresolved = check_all_doses(
        {"aspirin": [DosageMatch(100, "g", "100g")]}, store
    )
    assert len(alerts) == 2  # single-dose AND daily-total alerts
    assert all(a.severity == "HIGH" for a in alerts)
    assert unresolved == []


def test_normal_dose_produces_no_alert(store):
    alerts, unresolved = check_all_doses(
        {"aspirin": [DosageMatch(325, "mg", "325mg")]}, store
    )
    assert alerts == []
    assert unresolved == []


def test_moderate_excess_is_medium_not_high(store):
    alerts, _ = check_all_doses({"aspirin": [DosageMatch(900, "mg", "900mg")]}, store)
    assert len(alerts) == 1
    assert alerts[0].severity == "MEDIUM"


def test_narrow_therapeutic_index_drug_catches_overdose(store):
    alerts, _ = check_all_doses({"digoxin": [DosageMatch(5, "mg", "5mg")]}, store)
    assert any(a.severity == "HIGH" for a in alerts)


def test_mcg_conversion_correct_for_normal_dose(store):
    alerts, _ = check_all_doses({"digoxin": [DosageMatch(125, "mcg", "125mcg")]}, store)
    assert alerts == []


def test_non_mass_unit_skipped_not_guessed(store):
    alerts, _ = check_all_doses({"digoxin": [DosageMatch(50, "units", "50 units")]}, store)
    assert alerts == []


def test_cache_hit_short_circuits_openfda(store):
    """Same principle as interactions.py's cache-hit test: a drug already
    in the store must never trigger a network call."""
    def fail_if_called(drug_name):
        raise AssertionError("OpenFDA should not be called on a cache hit")

    with patch("nlp.dose_safety._fetch_dosage_text", side_effect=fail_if_called):
        alerts, unresolved = check_all_doses(
            {"aspirin": [DosageMatch(325, "mg", "325mg")]}, store
        )
    assert alerts == []
    assert unresolved == []


def test_cache_miss_falls_through_to_openfda_and_writes_back(store):
    def fake_fetch(drug_name):
        if drug_name == "newdrug":
            return "Do not exceed 200 mg in 24 hours."
        return None

    with patch("nlp.dose_safety._fetch_dosage_text", side_effect=fake_fetch):
        alerts, unresolved = check_all_doses(
            {"newdrug": [DosageMatch(500, "mg", "500mg")]}, store
        )
    assert len(alerts) == 1
    assert alerts[0].severity == "HIGH"
    assert unresolved == []

    def fail_if_called(drug_name):
        raise AssertionError("Should have used the cache on the second call")

    with patch("nlp.dose_safety._fetch_dosage_text", side_effect=fail_if_called):
        alerts2, unresolved2 = check_all_doses(
            {"newdrug": [DosageMatch(500, "mg", "500mg")]}, store
        )
    assert len(alerts2) == 1


def test_truly_unresolvable_drug_reported_not_silently_safe(store):
    """A drug OpenFDA has no data for must show up in unresolved_drugs,
    never be silently treated as having no dose issue."""
    with patch("nlp.dose_safety._fetch_dosage_text", return_value=None):
        alerts, unresolved = check_all_doses(
            {"totally_unknown_drug": [DosageMatch(50, "mg", "50mg")]}, store
        )
    assert alerts == []
    assert unresolved == ["totally_unknown_drug"]


def test_no_dosages_for_drug_skipped_entirely(store):
    alerts, unresolved = check_all_doses({"aspirin": []}, store)
    assert alerts == []
    assert unresolved == []
