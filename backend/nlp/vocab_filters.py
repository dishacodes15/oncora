"""
vocab_filters.py
OpenFDA's `brand_name` field is marketing text, not a controlled clinical
vocabulary — it's full of plain English words used as product names
("Relief", "Comfort", "Daily", "Complete", "Headache", "Sleep", "Active",
"Original", ...). Left unfiltered, the drug vocabulary will match those
words inside completely ordinary sentences ("patient reported a headache"
would flag "headache" as a detected drug).

It ALSO has two more subtle data-quality problems, both real and both
caught by scripts/evaluate_parser.py against the actual OpenFDA-built
vocabulary (not the small hand-written sample used in tests):

  1. For many non-proprietary ("generic") drugs, FDA populates brand_name
     with a descriptive listing title that includes the strength and
     dosage form, e.g. "Amoxicillin 500 MG Oral Capsule" — not a real
     brand name. Left as-is, that whole string becomes a vocab entry, and
     a prescription that happens to say "Amoxicillin 500mg" coincidentally
     matches it, misidentifying dosage-bearing text as a drug NAME.
  2. Many drugs are only listed under their salt form
     (generic_name="TRAMADOL HYDROCHLORIDE"), never as the bare
     ingredient name alone ("tramadol"). PhraseMatcher requires an exact
     token-sequence match, so a prescription that just says "Tramadol"
     never lines up with the 2-token salt-form entry — a false negative,
     and a more dangerous failure mode than a false positive since a
     missed drug means its interactions are never checked at all.

normalize_and_expand_term() addresses both: it strips embedded
dosage/form text from every term, and additionally emits a salt-stripped
base-ingredient variant alongside the original when a recognized salt
suffix is present. Both the original AND the derived variant are kept
(never destructively replaced) so we don't lose legitimate salt-specific
listings.

This is a curated stoplist of common English words seen in practice as
OTC brand names that are NOT themselves drug/ingredient names. It is
deliberately conservative — better to occasionally miss an obscure brand
name than to have the parser flag routine words as drugs in a
health-adjacent tool.

Applied in two places (defense in depth):
  1. vocab_builder.py — filters/normalizes before writing drug_vocab.json,
     so newly built vocabularies are clean.
  2. parser.py — filters/normalizes again at load time, so an
     already-built vocab file is cleaned up without needing a full
     rebuild.
"""

import re

MIN_TERM_LENGTH = 3  # single/double-letter "terms" are almost never real

COMMON_WORD_STOPLIST = {
    # generic marketing/descriptor words seen as OTC brand names
    "relief", "comfort", "daily", "complete", "total", "extra", "plus",
    "max", "maximum", "ultra", "care", "health", "healthy", "fresh",
    "clean", "active", "natural", "original", "classic", "gold",
    "premium", "advanced", "select", "choice", "essential", "basic",
    "regular", "strength", "fast", "quick", "rapid", "gentle", "soft",
    "smooth", "pure", "simple", "free", "clear", "bright", "calm",
    "balance", "restore", "renew", "revive", "protect", "defense",
    "shield", "guard", "support", "boost", "power", "energy", "vital",

    # common symptom/body words that show up as brand names for
    # symptom-targeted OTC products (e.g. a store brand literally
    # named "Headache" or "Sleep")
    "headache", "sleep", "pain", "cold", "cough", "allergy", "fever",
    "sinus", "stomach", "nausea", "fatigue", "stress", "anxiety",

    # single common words unlikely to ever be the intended match
    "one", "two", "first", "second", "new", "now", "day", "night",
    "morning", "evening", "kids", "adult", "family", "value", "size",

    # basic English function words. These should never legitimately be a
    # full drug-name entry on their own, but CAN show up as orphaned
    # leftovers after dosage/form-stripping normalization strips a
    # longer descriptive brand-name string down to nothing meaningful
    # (e.g. "The Tablet" -> strip "tablet" -> "the"). Caught in
    # production against the real OpenFDA-built vocab, not the small
    # test sample — the small sample never had enough noisy raw strings
    # to produce this leftover-word pattern.
    "the", "a", "an", "and", "or", "but", "for", "with", "without",
    "of", "in", "on", "at", "to", "from", "by", "is", "are", "was",
    "were", "be", "been", "this", "that", "these", "those", "it", "its",
    "as", "if", "then", "than", "so", "not", "no", "yes", "all", "any",
    "each", "per", "via", "up", "down", "out", "into", "over", "under",
}

# Strips a trailing dosage amount + unit + anything after it, e.g.
# "amoxicillin 500 mg oral capsule" -> "amoxicillin". Deliberately
# greedy (".*$") — once we hit a number+unit, everything after it in an
# OpenFDA descriptive title is dosage-form/route text, not part of the
# drug name.
_DOSAGE_STRIP_RE = re.compile(
    r"\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|%)\b.*$",
    re.IGNORECASE,
)

# Trailing dosage-form/route words that sometimes appear WITHOUT a
# preceding number (e.g. "ibuprofen oral tablet") — stripped word by
# word from the end so multiple trailing form-words are all removed.
_TRAILING_FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "oral", "injection",
    "solution", "suspension", "cream", "ointment", "extended", "release",
    "er", "xr", "chewable", "syrup", "patch", "gel", "spray", "drops",
    "suppository", "packet", "kit", "powder", "lotion",
}

# Common salt-form suffixes. When a term's last word is one of these,
# we ALSO emit the term with that suffix removed, so a bare-ingredient
# mention ("tramadol") still matches even when the vocab only has the
# salt form ("tramadol hydrochloride"). The salt-form entry is kept too
# — this adds a variant, it doesn't replace anything. List covers the
# common USAN/FDA salt/ester suffixes, not just the handful seen in
# initial testing — real generic_name data uses many of these.
_SALT_SUFFIXES = {
    "hydrochloride", "hcl", "sulfate", "bisulfate", "sodium", "potassium",
    "calcium", "magnesium", "phosphate", "tartrate", "bitartrate",
    "maleate", "citrate", "succinate", "besylate", "mesylate", "mesilate",
    "acetate", "fumarate", "carbonate", "chloride", "bromide",
    "hydrobromide", "iodide", "nitrate", "oxalate", "palmitate",
    "pamoate", "stearate", "valerate", "propionate", "benzoate",
    "salicylate", "gluconate", "lactate", "malate", "tosylate",
    "edetate", "napsylate", "polistirex", "trisilicate",
}


def _strip_trailing_form_words(term: str) -> str:
    words = term.split()
    while words and words[-1].lower() in _TRAILING_FORM_WORDS:
        words.pop()
    return " ".join(words)


def _strip_salt_suffix(term: str) -> str:
    """Returns the term with a trailing salt-suffix word removed, or ""
    if the term doesn't end in a recognized salt suffix."""
    words = term.split()
    if len(words) >= 2 and words[-1].lower() in _SALT_SUFFIXES:
        return " ".join(words[:-1])
    return ""


def normalize_and_expand_term(raw_term: str) -> set:
    """
    Takes one raw vocab term (as pulled from OpenFDA) and returns the set
    of normalized variants that should be added to the matcher vocabulary:
    always the dosage/form-stripped base term, plus a salt-stripped
    variant if a recognized salt suffix is present. Never returns the
    untouched raw term if it contained dosage/form text — that text is
    exactly the false-positive source described above.
    """
    base = _DOSAGE_STRIP_RE.sub("", raw_term).strip()
    base = _strip_trailing_form_words(base).strip()

    variants = set()
    if base:
        variants.add(base.lower())
        salt_free = _strip_salt_suffix(base)
        if salt_free:
            variants.add(salt_free.lower())

    return variants


def is_valid_drug_term(term: str) -> bool:
    """Returns False for terms that should be excluded from the drug
    vocabulary: too short, or on the common-word stoplist."""
    normalized = term.strip().lower()
    if len(normalized) < MIN_TERM_LENGTH:
        return False
    if normalized in COMMON_WORD_STOPLIST:
        return False
    return True


def filter_vocab(terms) -> set:
    """
    Full pipeline for a collection of raw vocab terms: normalize +
    expand each one (dosage/form stripping, salt-suffix variants), then
    filter the result through the common-word stoplist and minimum
    length check.
    """
    expanded = set()
    for t in terms:
        expanded |= normalize_and_expand_term(t)
    return {t for t in expanded if is_valid_drug_term(t)}

