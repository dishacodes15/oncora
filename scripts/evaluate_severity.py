"""
evaluate_severity.py
Measures severity-classification ACCURACY against a hand-labeled set of
realistic FDA-label-style sentences (data/eval_set_severity.json).

Two things are scored separately, because they fail differently:

  1. Severity accuracy: did HIGH/MEDIUM/LOW/None match the expected call?
  2. Confidence accuracy: for cases where the phrasing is genuinely
     ambiguous (no recognized keyword phrase), did the classifier
     correctly flag itself as low-confidence rather than silently
     asserting a confident answer? A wrong severity call caught by
     low-confidence flagging is a much smaller problem than a wrong
     severity call asserted with high confidence — the latter is what
     would reach a user unreviewed.

This is the step your original plan flagged as the riskiest one to get
right with keyword rules alone. Treat a drop in this score after any
edit to severity.py's signal lists as a real regression, not noise.

Run:
    python -m scripts.evaluate_severity
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from nlp.severity import classify  # noqa: E402


def evaluate(eval_set_path: str) -> None:
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    severity_correct = 0
    confidence_correct = 0
    confidence_checkable = 0
    mismatches = []

    for case in eval_set:
        result = classify(case["text"], case["other_drug"])
        expected_severity = case["expected_severity"]
        actual_severity = result.severity if result else None

        sev_ok = actual_severity == expected_severity
        if sev_ok:
            severity_correct += 1
        else:
            mismatches.append((case, result, "severity"))

        expected_confidence = case.get("expected_confidence")
        if expected_confidence is not None:
            confidence_checkable += 1
            actual_confidence = result.confidence if result else None
            if actual_confidence == expected_confidence:
                confidence_correct += 1
            elif sev_ok:
                # severity matched but confidence flag didn't — still log it,
                # this is the exact failure mode the confidence flag exists
                # to catch: an unreviewed guess presented as certain.
                mismatches.append((case, result, "confidence"))

    n = len(eval_set)
    print(f"\n{'='*60}")
    print(f"Severity classification accuracy — {n} examples")
    print(f"{'='*60}")
    print(f"Severity accuracy:   {severity_correct}/{n} ({severity_correct/n:.1%})")
    if confidence_checkable:
        print(f"Confidence accuracy: {confidence_correct}/{confidence_checkable} "
              f"({confidence_correct/confidence_checkable:.1%}) "
              f"[only counted where a confidence expectation was specified]")
    print()

    if mismatches:
        for case, result, kind in mismatches:
            print(f"MISMATCH ({kind}): other_drug={case['other_drug']!r}")
            print(f"  Text: {case['text']!r}")
            print(f"  Expected severity={case['expected_severity']!r} "
                  f"confidence={case.get('expected_confidence', '(unspecified)')!r}")
            print(f"  Got:      {result!r}")
            if case.get("note"):
                print(f"  Note: {case['note']}")
            print()
    else:
        print("No mismatches — every example matched exactly.")
    print()


if __name__ == "__main__":
    path = os.environ.get(
        "EVAL_SET_SEVERITY_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "eval_set_severity.json"),
    )
    evaluate(path)
