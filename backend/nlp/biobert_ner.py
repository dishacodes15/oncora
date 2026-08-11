"""
biobert_ner.py
Thin wrapper around a pretrained biomedical NER model, used as a
FALLBACK to the vocabulary PhraseMatcher in hybrid_ner.py - not the
primary drug-recognition method. The vocabulary matcher is exact and
deterministic (confidence 1.0); this model is statistical and only
consulted for drugs the vocabulary did not already find.

Model: d4data/biomedical-ner-all (DistilBERT-based, already fine-tuned
for biomedical NER - NOT raw BioBERT, which has no entity-recognition
head). No training or fine-tuning is done here, per project scope rules.

IMPORTANT - manual subword merging: the model uses WordPiece
tokenization, and unfamiliar drug names (e.g. "Ozempic") get split into
subword fragments ("oz", "##em", "##pic") tagged with the same entity
label. HuggingFace's aggregation_strategy="simple" is supposed to merge
these automatically but was observed NOT reliably doing so in testing -
returning three separate garbage "entities" instead of one "ozempic".
This module does its own merging pass afterward: consecutive same-label
entities where the next fragment starts with "##" or touches the
previous fragment's end position (no gap) are joined into one entity,
using the MINIMUM of the fragment scores (the weakest link, not the
average - a single low-confidence fragment should pull down the whole
merged entity's reported confidence, not get diluted by two more
confident neighbors).

Loading is lazy and defensive: if transformers/torch cannot load the
model, get_ner_pipeline() raises and the caller (hybrid_ner.py) is
expected to catch that and degrade to vocabulary-only extraction.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("oncora.nlp.biobert_ner")

DEFAULT_MODEL_NAME = "d4data/biomedical-ner-all"

_DRUG_LABEL_KEYWORDS = ("med", "drug", "chemical")


@dataclass
class BioBERTEntity:
    text: str
    label: str
    score: float
    start: int
    end: int


class _Stats:
    def __init__(self):
        self.total_calls = 0
        self.total_entities_found = 0
        self.total_latency_seconds = 0.0

    def record(self, n_entities: int, latency_seconds: float) -> None:
        self.total_calls += 1
        self.total_entities_found += n_entities
        self.total_latency_seconds += latency_seconds

    def summary(self) -> dict:
        avg_latency_ms = (
            (self.total_latency_seconds / self.total_calls) * 1000
            if self.total_calls else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_entities_found": self.total_entities_found,
            "avg_latency_ms": round(avg_latency_ms, 1),
        }


_stats = _Stats()
_ner_pipeline = None


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        model_name = os.environ.get("BIOBERT_NER_MODEL", DEFAULT_MODEL_NAME)
        logger.info("Loading biomedical NER model: %s (CPU)", model_name)
        _ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=-1,
        )
        logger.info("Biomedical NER model loaded.")
    return _ner_pipeline


def _merge_subword_fragments(raw_entities: list) -> List[BioBERTEntity]:
    """Merges consecutive same-label entities that are WordPiece
    fragments of a single word - see module docstring for why the
    pipeline's own aggregation cannot be trusted to do this."""
    if not raw_entities:
        return []

    ordered = sorted(raw_entities, key=lambda e: e.get("start", 0))
    merged: List[BioBERTEntity] = []

    for ent in ordered:
        word = str(ent["word"])
        label = ent.get("entity_group", "")
        score = float(ent.get("score", 0.0))
        start = int(ent.get("start", 0))
        end = int(ent.get("end", 0))

        is_continuation = word.startswith("##")
        if is_continuation:
            word = word[2:]

        if (
            merged
            and merged[-1].label == label
            and (is_continuation or start <= merged[-1].end)
        ):
            prev = merged[-1]
            merged[-1] = BioBERTEntity(
                text=prev.text + word,
                label=label,
                score=min(prev.score, score),  # weakest fragment sets confidence
                start=prev.start,
                end=end,
            )
        else:
            merged.append(BioBERTEntity(text=word, label=label, score=score, start=start, end=end))

    return merged


def extract_biobert_entities(text: str) -> List[BioBERTEntity]:
    """Runs the NER pipeline, merges WordPiece fragments back into whole
    words, and returns only drug-like entities. Raises if the pipeline
    is not available - callers must catch this."""
    ner = get_ner_pipeline()

    start_time = time.perf_counter()
    raw_entities = ner(text)
    latency = time.perf_counter() - start_time

    merged = _merge_subword_fragments(raw_entities)

    results = [
        e for e in merged
        if any(keyword in e.label.lower() for keyword in _DRUG_LABEL_KEYWORDS)
    ]

    _stats.record(len(results), latency)
    return results


def get_biobert_stats() -> dict:
    return _stats.summary()
