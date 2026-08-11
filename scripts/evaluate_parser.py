"""
evaluate_parser.py
Measures drug-extraction ACCURACY, not just correctness. Passing unit
tests confirms the code does what we told it to do; this measures
whether that matches reality on prescription-like text.

Computes micro-averaged precision, recall, and F1 across a labeled eval
set (data/eval_set_drugs.json), and prints per-example false
positives/negatives so misses are diagnosable, not just a single score.

Precision matters because a false positive (flagging a non-drug word as
a drug) can trigger a bogus interaction alert. Recall matters more,
though: a false negative (missing a real drug) means that drug's
interactions never get checked at all — a silent gap, not a visible
error. When choosing between tightening the stoplist (raises precision,
may lower recall) and expanding the vocab (raises recall, may lower
precision), weight recall misses as the more serious failure mode for
a safety tool.

Run:
    python -m scripts.evaluate_parser
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from nlp.parser import PrescriptionParser  # noqa: E402


def evaluate(eval_set_path: str, vocab_path: str, model_name: str) -> None:
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    parser = PrescriptionParser(vocab_path=vocab_path, model_name=model_name)

    total_tp, total_fp, total_fn = 0, 0, 0
    per_example_rows = []

    for case in eval_set:
        text = case["text"]
        expected = set(d.lower() for d in case["expected_drugs"])
        predicted = parser.extract_drugs(text)

        tp = expected & predicted
        fp = predicted - expected      # flagged but shouldn't have been
        fn = expected - predicted      # missed real drugs

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        per_example_rows.append((text, fp, fn))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"\n{'='*60}")
    print(f"Drug extraction accuracy — {len(eval_set)} examples, model={model_name}")
    print(f"{'='*60}")
    print(f"Precision: {precision:.3f}  ({total_tp} correct / {total_tp + total_fp} predicted)")
    print(f"Recall:    {recall:.3f}  ({total_tp} correct / {total_tp + total_fn} expected)")
    print(f"F1:        {f1:.3f}")
    print()

    any_errors = False
    for text, fp, fn in per_example_rows:
        if fp or fn:
            any_errors = True
            print(f"MISMATCH: {text!r}")
            if fp:
                print(f"  False positives (flagged, shouldn't be): {sorted(fp)}")
            if fn:
                print(f"  False negatives (missed real drug):      {sorted(fn)}")
    if not any_errors:
        print("No mismatches — every example matched exactly.")
    print()


if __name__ == "__main__":
    eval_path = os.environ.get(
        "EVAL_SET_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "eval_set_drugs.json"),
    )
    vocab_path = os.environ.get("DRUG_VOCAB_PATH", "data/drug_vocab.json")
    model_name = os.environ.get("SPACY_MODEL", "en_core_sci_sm")
    evaluate(eval_path, vocab_path, model_name)
