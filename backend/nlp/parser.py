"""
parser.py
Extracts drug names and dosages from free-text prescription strings.

Drug extraction: spaCy PhraseMatcher over a vocabulary built offline by
vocab_builder.py (see drug_vocab.json). This replaces a hardcoded
substring/keyword list — the matcher handles multi-word names
("acetylsalicylic acid"), case variation, and word-boundary matching
for free (a substring check would wrongly match "ibuprofen" inside
"ibuprofenhydrochloride" or miss "Aspirin" vs "aspirin").

Dosage extraction: still regex — dosages are a genuinely regular
grammar (number + unit), so NLP would be overkill. We scope each
dosage match to the sentence it appears in (via spaCy's sentence
boundaries) so "Aspirin 100mg, Warfarin 5mg" doesn't accidentally
pair Aspirin with 5mg.
"""

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set

import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language

from .vocab_filters import filter_vocab

logger = logging.getLogger("oncora.nlp.parser")

# Broadened unit pattern: mg, mcg/µg, mL, units/u/IU, g — covers the
# common prescription dosage units instead of just "mg". "g" (grams) is
# placed last in the alternation deliberately for readability, though
# regex alternation is position-anchored so match order doesn't actually
# affect correctness here — "mg" and "g" never compete for the same
# starting character.
DOSAGE_PATTERN = re.compile(
    r"""
    (?P<amount>\d+(?:\.\d+)?)          # e.g. 500, 12.5
    \s*
    (?P<unit>mg|mcg|µg|ug|ml|iu|units?|g)  # unit, case-insensitive (see flags)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class DosageMatch:
    amount: float
    unit: str
    raw_text: str


@dataclass
class ParsedPrescription:
    drugs: Set[str]
    # drug name -> list of dosages found in the same sentence as that drug
    dosages_by_drug: Dict[str, List[DosageMatch]]


class PrescriptionParser:
    """
    Wraps a spaCy pipeline + PhraseMatcher. Instantiate once (it's not
    cheap to load a model) and reuse across requests — see get_parser()
    below, which caches a singleton for the FastAPI app's lifetime.
    """

    def __init__(self, vocab_path: str, model_name: str = "en_core_sci_sm"):
        self.model_name = model_name
        try:
            self.nlp: Language = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. Run: "
                f"python -m spacy download {model_name}  "
                f"(or 'pip install scispacy && pip install <sci model url>' "
                f"for scispaCy biomedical models — see their README, they "
                f"aren't distributed via `spacy download`)."
            ) from exc

        # scispaCy's small models ship without a sentence boundary
        # component in some builds — make sure one exists, since
        # extract_dosages() relies on doc.sents.
        if "senter" not in self.nlp.pipe_names and "parser" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        with open(vocab_path) as f:
            raw_terms: List[str] = json.load(f)

        # Filter out common-English-word false positives (e.g. OTC brand
        # names like "Headache", "Relief") before building the matcher —
        # see vocab_filters.py. Applied here (load time) so it cleans up
        # an already-built vocab file without needing vocab_builder.py to
        # be rerun.
        terms = sorted(filter_vocab(raw_terms))
        dropped = len(raw_terms) - len(terms)
        if dropped:
            logger.info("Filtered %d common-word false-positive terms from vocab (%d remain)",
                        dropped, len(terms))

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = list(self.nlp.pipe(terms))
        self.matcher.add("DRUG", patterns)
        logger.info("Loaded PhraseMatcher with %d drug terms (model=%s)",
                    len(terms), model_name)

    def extract_drugs(self, text: str) -> Set[str]:
        """
        Returns the deduped, lowercased set of drug names found in `text`.
        This is the direct replacement for the old hardcoded
        `COMMON_DRUGS` substring loop.
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)
        found = {doc[start:end].text.lower() for _, start, end in matches}
        return found

    def extract_dosages(self, text: str) -> List[DosageMatch]:
        """Pure regex — no sentence scoping. Use extract_drugs_with_dosages
        for the sentence-scoped, drug-aligned version used by the API."""
        out = []
        for m in DOSAGE_PATTERN.finditer(text):
            unit = m.group("unit").lower()
            unit = {"ug": "mcg", "µg": "mcg", "unit": "units"}.get(unit, unit)
            out.append(DosageMatch(
                amount=float(m.group("amount")),
                unit=unit,
                raw_text=m.group(0),
            ))
        return out

    def parse(self, text: str) -> ParsedPrescription:
        """
        Main entry point: runs the doc once, finds all drug matches, and
        for each sentence containing a drug, attaches any dosages found in
        THAT SAME sentence to that drug. This is what keeps
        "Paracetamol 500mg, Warfarin 5mg" from cross-pairing dosages.
        """
        doc = self.nlp(text)
        drug_matches = self.matcher(doc)

        # Group matches by the sentence they fall in.
        sent_for_token = {}
        for sent in doc.sents:
            for tok in sent:
                sent_for_token[tok.i] = sent

        drugs: Set[str] = set()
        dosages_by_drug: Dict[str, List[DosageMatch]] = {}

        for match_id, start, end in drug_matches:
            drug_name = doc[start:end].text.lower()
            drugs.add(drug_name)
            sent = sent_for_token.get(start)
            sent_text = sent.text if sent else text
            sent_dosages = self.extract_dosages(sent_text)
            dosages_by_drug.setdefault(drug_name, [])
            for d in sent_dosages:
                if d not in dosages_by_drug[drug_name]:
                    dosages_by_drug[drug_name].append(d)

        return ParsedPrescription(drugs=drugs, dosages_by_drug=dosages_by_drug)


_parser_singleton: Optional[PrescriptionParser] = None


def get_parser(vocab_path: str = None, model_name: str = None) -> PrescriptionParser:
    """
    Lazily builds and caches a single PrescriptionParser for the process
    lifetime. Call this from FastAPI's startup, not per-request — loading
    a spaCy model + building the PhraseMatcher takes real time (hundreds
    of ms to seconds depending on vocab size) and should happen once.
    """
    global _parser_singleton
    if _parser_singleton is None:
        import os
        vocab_path = vocab_path or os.environ.get("DRUG_VOCAB_PATH", "data/drug_vocab.json")
        model_name = model_name or os.environ.get("SPACY_MODEL", "en_core_sci_sm")
        _parser_singleton = PrescriptionParser(vocab_path=vocab_path, model_name=model_name)
    return _parser_singleton
