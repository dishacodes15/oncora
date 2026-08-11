import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from nlp.dose_prose_parser import best_limits_from_text, extract_dose_candidates


def test_explicit_daily_max_high_confidence():
    r = best_limits_from_text("Do not exceed 4000 mg of acetaminophen in 24 hours.")
    assert r["max_daily_dose_mg"] == 4000.0
    assert r["confidence"] == "high"


def test_maximum_recommended_dose_classified_daily():
    r = best_limits_from_text("The maximum recommended dose is 80 mg once daily.")
    assert r["max_daily_dose_mg"] == 80.0
    assert r["confidence"] == "high"


def test_usual_dose_fallback_is_low_confidence():
    r = best_limits_from_text("The usual adult dose is 500 mg every 8 hours.")
    assert r["max_single_dose_mg"] == 500.0
    assert r["confidence"] == "low"


def test_explicit_single_dose_high_confidence():
    r = best_limits_from_text("Administer up to a maximum single dose of 650 mg.")
    assert r["max_single_dose_mg"] == 650.0
    assert r["confidence"] == "high"


def test_no_signal_returns_none():
    r = best_limits_from_text("This medication has been studied extensively in clinical trials.")
    assert r is None


def test_empty_text_returns_none():
    assert best_limits_from_text("") is None
    assert best_limits_from_text(None) is None


def test_multi_clause_sentence_does_not_cross_contaminate():
    """Regression test: a fixed character window used to leak cue words
    from one clause into an adjacent clause's classification (e.g. 'per
    day' from clause 1 falsely tagging clause 2's 'per single dose'
    match as ambiguous). Clause-scoping fixed this."""
    r = best_limits_from_text(
        "Maximum dose is 40 mg per day; do not exceed 10 mg per single dose."
    )
    assert r["max_daily_dose_mg"] == 40.0
    assert r["max_single_dose_mg"] == 10.0
    assert r["confidence"] == "high"


def test_bare_word_dose_does_not_force_ambiguous():
    """Regression test: the single-dose cue regex used to match bare
    'dose', which appears inside the matched phrase itself ('maximum
    recommended dose is 80mg'), incorrectly downgrading confidence on
    almost every match."""
    candidates = extract_dose_candidates("The maximum recommended dose is 80 mg once daily.")
    assert len(candidates) == 1
    assert candidates[0].limit_type == "daily"
    assert candidates[0].confidence == "high"


def test_realistic_combined_usual_and_max_sentence():
    r = best_limits_from_text(
        "The usual dosage is 10 to 20 mg once daily, not to exceed 40 mg per day."
    )
    assert r["max_daily_dose_mg"] == 40.0
    assert r["confidence"] == "high"


def test_smallest_candidate_preferred_when_multiple_found():
    """For a safety tool, under-estimating the ceiling (flagging more) is
    the safer failure direction than picking a generously large number."""
    r = best_limits_from_text(
        "Maximum dose is 100 mg per day. In some cases up to 200 mg per day may be used."
    )
    assert r["max_daily_dose_mg"] == 100.0
