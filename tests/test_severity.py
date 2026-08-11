import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from nlp.severity import classify


def test_contraindicated_is_high_confidence():
    r = classify("Concomitant use with aspirin is contraindicated.", "aspirin")
    assert r.severity == "HIGH"
    assert r.confidence == "high"


def test_monitor_closely_is_medium():
    r = classify("May increase levels of warfarin; monitor closely.", "warfarin")
    assert r.severity == "MEDIUM"
    assert r.confidence == "high"


def test_no_significant_interaction_is_low():
    r = classify("No significant interaction with acetaminophen at standard doses.", "acetaminophen")
    assert r.severity == "LOW"


def test_no_mention_of_other_drug_returns_none():
    r = classify("This drug should not be used with grapefruit juice.", "warfarin")
    assert r is None


def test_unrecognized_phrasing_flagged_low_confidence():
    """When the other drug is mentioned but no known signal phrase
    matches, we must not silently claim high confidence."""
    r = classify("This has been studied in combination with digoxin.", "digoxin")
    assert r is not None
    assert r.confidence == "low"


def test_conflicting_signals_flagged_low_confidence():
    r = classify(
        "Use with metformin causes increased risk of lactic acidosis, "
        "though no dose adjustment is necessary in most patients.",
        "metformin",
    )
    assert r.confidence == "low"


def test_active_voice_should_not_use_is_high():
    """Regression test: HIGH_SIGNALS originally only matched the passive
    phrasing ('should not BE used'), missing this equally common active
    phrasing. Caught by scripts/evaluate_severity.py, not a unit test —
    that's exactly why the eval harness exists alongside these tests."""
    r = classify(
        "Patients should not use this medication with tramadol due to "
        "risk of serotonin syndrome.",
        "tramadol",
    )
    assert r.severity == "HIGH"


def test_negated_dose_adjustment_does_not_shadow_low_signal():
    """Regression test: the generic MEDIUM 'dose adjustment' phrase used
    to match as a substring of the negated LOW phrase 'no dose adjustment
    ... necessary', incorrectly winning the tier check. Caught by
    scripts/evaluate_severity.py."""
    r = classify(
        "Minor interaction; no dose adjustment is necessary in most patients.",
        "someotherdrug",
    )
    assert r is None  # other_drug not even mentioned here — different
                        # check: confirm plain "dose adjustment" text alone
                        # doesn't accidentally match MEDIUM when negated
    r2 = classify(
        "Minor interaction with omeprazole; no dose adjustment is "
        "necessary in most patients.",
        "omeprazole",
    )
    assert r2.severity == "LOW"


def test_empty_text_returns_none():
    assert classify("", "aspirin") is None
    assert classify(None, "aspirin") is None
