"""
hybrid_ner.py
Combines the existing vocabulary PhraseMatcher (parser.py, untouched)
with the biomedical NER model (biobert_ner.py) as a FALLBACK - not a
replacement. Vocabulary matches are exact and deterministic
(source="vocabulary", confidence=1.0). BioBERT is only consulted for
drug mentions the vocabulary did NOT already find, and anything it
finds is tagged source="biobert_fallback" with the model's own
confidence score, so a caller (or a clinician reading the API response)
can see exactly where each detected drug came from and how sure the
system actually is about it.

This module never touches parser.py - the vocabulary matcher keeps
working exactly as before if BioBERT is unavailable (model not
installed, first-run download failed, etc.), same graceful-degradation
principle already used throughout this codebase (e.g. main.py's
prescription pipeline returning 503 instead of crashing if the spaCy
model fails to load).

Note on overlap detection: parser.py's extract_drugs() returns a set of
matched strings, not character spans, so exact span-overlap checking
against BioBERT's entity offsets is not available without modifying
parser.py. Overlap here is approximated via case-insensitive substring
containment instead - close enough to avoid double-reporting the same
drug under two sources, without requiring changes to already-tested
code.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Set

from .biobert_ner import extract_biobert_entities

logger = logging.getLogger("oncora.nlp.hybrid_ner")


@dataclass
class HybridEntity:
    drug: str
    source: str        # "vocabulary" | "biobert_fallback"
    confidence: float  # 1.0 for vocabulary; model score (typically 0.6-0.95) for fallback


@dataclass
class HybridExtractionResult:
    entities: List[HybridEntity] = field(default_factory=list)
    biobert_available: bool = False
    vocabulary_count: int = 0
    novel_fallback_count: int = 0

    def drug_names(self) -> Set[str]:
        """Combined set of drug names from both sources, lowercased -
        this is what downstream interaction/dose checking should consume,
        same shape as parser.py's plain extract_drugs() output."""
        return {e.drug.strip().lower() for e in self.entities}


def _is_already_covered(candidate: str, vocabulary_drugs: set) -> bool:
    candidate_lower = candidate.strip().lower()
    if not candidate_lower:
        return True
    return any(
        candidate_lower in vocab_term or vocab_term in candidate_lower
        for vocab_term in vocabulary_drugs
    )


def extract_drugs_hybrid(text: str, parser) -> HybridExtractionResult:
    """
    Main entry point. `parser` is an already-constructed
    nlp.parser.PrescriptionParser (the same singleton main.py already
    builds at startup) - this function does not build its own spaCy
    pipeline, it reuses the existing one.
    """
    vocabulary_drugs = parser.extract_drugs(text)

    entities = [
        HybridEntity(drug=d, source="vocabulary", confidence=1.0)
        for d in vocabulary_drugs
    ]

    biobert_available = False
    novel_count = 0
    try:
        biobert_entities = extract_biobert_entities(text)
        biobert_available = True
    except Exception as exc:
        logger.warning(
            "BioBERT NER unavailable (%s) - continuing with vocabulary-only "
            "extraction. This is a graceful degradation, not a failure of "
            "the overall prescription pipeline.", exc,
        )
        biobert_entities = []

    for ent in biobert_entities:
        if _is_already_covered(ent.text, vocabulary_drugs):
            continue
        entities.append(HybridEntity(
            drug=ent.text.strip().lower(),
            source="biobert_fallback",
            confidence=round(ent.score, 4),
        ))
        novel_count += 1

    return HybridExtractionResult(
        entities=entities,
        biobert_available=biobert_available,
        vocabulary_count=len(vocabulary_drugs),
        novel_fallback_count=novel_count,
    )
