"""
Tests for backend/nlp/parser.py.

Uses en_core_web_sm instead of en_core_sci_sm for CI/portability — the
PhraseMatcher logic under test is identical regardless of which spaCy
model supplies tokenization/sentence boundaries. Override via SPACY_MODEL
env var to run against the real biomedical model locally.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from nlp.parser import PrescriptionParser

TEST_MODEL = os.environ.get("SPACY_MODEL_TEST", "en_core_web_sm")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drug_vocab.json")


@pytest.fixture(scope="module")
def parser():
    return PrescriptionParser(vocab_path=VOCAB_PATH, model_name=TEST_MODEL)


def test_extracts_single_drug(parser):
    drugs = parser.extract_drugs("Take Aspirin daily.")
    assert "aspirin" in drugs


def test_case_insensitive_match(parser):
    drugs = parser.extract_drugs("ASPIRIN and Warfarin together.")
    assert "aspirin" in drugs
    assert "warfarin" in drugs


def test_multiword_drug_name(parser):
    """Picks an actual multi-word term from the loaded vocab file rather
    than assuming a specific string is present — the real OpenFDA-derived
    vocab won't necessarily contain any particular chemical name (e.g.
    aspirin may be listed under generic_name "ASPIRIN", not
    "acetylsalicylic acid")."""
    with open(VOCAB_PATH) as f:
        vocab_terms = json.load(f)
    multiword_terms = [t for t in vocab_terms if " " in t.strip()]
    if not multiword_terms:
        pytest.skip("No multi-word terms in this vocab file to test against")

    sample_term = multiword_terms[0]
    drugs = parser.extract_drugs(f"Patient is on {sample_term.title()} 325mg.")
    assert sample_term.lower() in drugs


def test_no_false_positive_on_unrelated_text(parser):
    drugs = parser.extract_drugs("The patient reported mild headache and nausea.")
    assert drugs == set()


def test_common_word_stoplist_filters_known_false_positives():
    """Direct unit test of the filter itself — independent of whatever
    happens to be in a given vocab file, so this stays meaningful even
    as drug_vocab.json gets regenerated over time."""
    from nlp.vocab_filters import is_valid_drug_term
    for word in ["headache", "relief", "daily", "comfort", "sleep"]:
        assert not is_valid_drug_term(word), f"{word!r} should be filtered as a common word"
    # sanity check: real drug names must NOT be filtered
    for word in ["aspirin", "warfarin", "metformin"]:
        assert is_valid_drug_term(word), f"{word!r} should NOT be filtered"


def test_dosage_units_broadened(parser):
    doses = parser.extract_dosages("500mg, 75mcg, 2000 IU, 10ml, 5 units")
    units = {d.unit for d in doses}
    assert units == {"mg", "mcg", "iu", "ml", "units"}


def test_dosage_scoped_to_correct_sentence(parser):
    text = "Paracetamol 500mg twice daily. Warfarin 5mg once a day."
    result = parser.parse(text)
    assert [round(d.amount) for d in result.dosages_by_drug["paracetamol"]] == [500]
    assert [round(d.amount) for d in result.dosages_by_drug["warfarin"]] == [5]


def test_dosage_baked_into_brand_name_is_stripped(parser):
    """Regression test: OpenFDA's brand_name field, for many generic
    drugs, is a descriptive listing title that includes strength/dosage
    form (e.g. "Amoxicillin 500 MG Oral Capsule"), not a real brand name.
    Left unstripped, that whole string becomes a vocab entry and
    coincidentally matches ordinary dosage text, misidentifying dosage
    as part of a drug name. Caught by scripts/evaluate_parser.py against
    the real built vocab, not this test's small sample."""
    from nlp.vocab_filters import normalize_and_expand_term
    variants = normalize_and_expand_term("Amoxicillin 500 MG Oral Capsule")
    assert "amoxicillin" in variants
    assert not any("500" in v for v in variants)


def test_salt_form_only_generic_name_still_matches_bare_ingredient(parser):
    """Regression test: many drugs are only listed under their salt form
    in OpenFDA (generic_name="TRAMADOL HYDROCHLORIDE"), never as the bare
    ingredient alone. A prescription that just says "Tramadol" must still
    match — this is a false-negative bug, more dangerous than a false
    positive since a missed drug means its interactions are never
    checked at all. Caught by scripts/evaluate_parser.py."""
    from nlp.vocab_filters import normalize_and_expand_term
    variants = normalize_and_expand_term("TRAMADOL HYDROCHLORIDE")
    assert "tramadol" in variants
    assert "tramadol hydrochloride" in variants  # original salt form kept too


def test_dosage_not_cross_paired_across_sentences(parser):
    """Regression guard: a dosage in one sentence must never attach to a
    drug named only in a different sentence."""
    text = "Aspirin 100mg with food. Metformin 500mg with dinner."
    result = parser.parse(text)
    aspirin_amounts = {round(d.amount) for d in result.dosages_by_drug.get("aspirin", [])}
    assert 500 not in aspirin_amounts
