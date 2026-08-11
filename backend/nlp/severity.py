"""
severity.py
Turns free-text OpenFDA label prose (the `drug_interactions` field) into a
structured severity call: HIGH / MEDIUM / LOW, plus a confidence flag.

THIS IS THE RISKIEST PART OF THE PIPELINE. Clinical label language is
inconsistent — the same underlying risk gets phrased a dozen different
ways across different manufacturers' labels. Keyword rules will have
real false negatives (missed severity signals) and some false positives.

Because this touches health-adjacent output, the design goal here is NOT
"always produce a confident answer" — it's "produce a defensible answer,
and say so clearly when we're not sure." Concretely:

  - If we find a strong, unambiguous severity phrase -> confidence="high"
  - If we find a weaker/ambiguous signal, or multiple conflicting
    signals -> confidence="low", and the caller (interactions.py) should
    surface that in the API response so it can be flagged for manual
    review rather than silently trusted.
  - If we find NO signal at all for a mentioned drug pair, we do not
    guess a severity — we return None, and the caller should treat that
    as "no structured interaction found" rather than inventing LOW.

Upgrade path (noted, not built here): swap `classify()`'s body for a call
to an LLM-based structured extractor once you have enough real-world
low-confidence cases logged to build a proper eval set. Keep the same
function signature so interactions.py doesn't need to change.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

# Ordered HIGH -> LOW. Order matters: if a HIGH-severity phrase and a
# LOW-severity phrase both appear (e.g. a label listing multiple different
# interacting drugs at different severities), we still want to catch the
# HIGH signal rather than average it away.
HIGH_SIGNALS = [
    r"\bcontraindicated\b",
    r"\bshould not be (?:used|administered|co-?administered|combined)\b",
    r"\bshould not (?:use|combine|take)\b",   # active-voice variant of the above
    r"\bavoid concomitant use\b",
    r"\bshould be avoided\b",
    r"\bfatal\b",
    r"\bsevere(?:ly)?\b.{0,30}\b(?:interaction|risk|toxicity)\b",
    r"\bdo not (?:use|administer|combine)\b",
]

MEDIUM_SIGNALS = [
    r"\bincreased risk\b",
    r"\bmonitor closely\b",
    r"\bmonitor(?:ing)? (?:is )?recommended\b",
    r"\bmay increase\b",
    r"\bmay potentiate\b",
    r"(?<!no )\bdose adjustment\b",   # excludes "no dose adjustment ... necessary",
                                       # which is a LOW signal (see LOW_SIGNALS) —
                                       # without this exclusion the generic phrase
                                       # here matches as a substring of the negated
                                       # LOW phrasing and wins the tier check first
    r"\bcaution (?:is )?(?:advised|recommended)\b",
    r"\buse with caution\b",
]

LOW_SIGNALS = [
    r"\bno (?:significant|clinically significant) interaction\b",
    r"\bminor\b",
    r"\bunlikely to\b",
    r"\bno dose adjustment (?:is )?(?:necessary|needed|required)\b",
]

_HIGH_RE = re.compile("|".join(HIGH_SIGNALS), re.IGNORECASE)
_MEDIUM_RE = re.compile("|".join(MEDIUM_SIGNALS), re.IGNORECASE)
_LOW_RE = re.compile("|".join(LOW_SIGNALS), re.IGNORECASE)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SeverityResult:
    severity: str        # HIGH | MEDIUM | LOW
    confidence: str       # high | low
    matched_signal: str   # which phrase triggered this, for auditability
    evidence_sentence: str  # the sentence it came from, for manual review


def _sentences_mentioning(text: str, other_drug: str) -> List[str]:
    """Only look at sentences that actually mention the other detected
    drug — a label's interaction section often covers many different
    drugs at different severities, and we don't want a HIGH warning
    about drug X leaking onto our pair with drug Y."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    other_lower = other_drug.lower()
    return [s for s in sentences if other_lower in s.lower()]


def classify(interaction_text: str, other_drug: str) -> Optional[SeverityResult]:
    """
    interaction_text: the free-text drug_interactions field from ONE
        drug's OpenFDA label.
    other_drug: the name of the OTHER drug in the pair we're checking
        (e.g. if this is warfarin's label text, other_drug="aspirin").

    Returns None if no sentence mentions other_drug at all — that's a
    genuine "not found," not a low-confidence guess.
    """
    if not interaction_text:
        return None

    relevant = _sentences_mentioning(interaction_text, other_drug)
    if not relevant:
        return None

    # Scan relevant sentences for the strongest signal found, preferring
    # HIGH > MEDIUM > LOW if a sentence contains multiple signal tiers.
    best: Optional[SeverityResult] = None
    signal_conflict = False

    for sentence in relevant:
        high_match = _HIGH_RE.search(sentence)
        medium_match = _MEDIUM_RE.search(sentence)
        low_match = _LOW_RE.search(sentence)

        tiers_hit = sum(bool(m) for m in (high_match, medium_match, low_match))
        if tiers_hit > 1:
            # e.g. a sentence with both "increased risk" and "no dose
            # adjustment necessary" — genuinely ambiguous wording.
            signal_conflict = True

        if high_match:
            candidate = SeverityResult("HIGH", "high", high_match.group(0), sentence.strip())
        elif medium_match:
            candidate = SeverityResult("MEDIUM", "high", medium_match.group(0), sentence.strip())
        elif low_match:
            candidate = SeverityResult("LOW", "high", low_match.group(0), sentence.strip())
        else:
            # Sentence mentions the other drug but uses none of our known
            # phrases — we know *something* is being said about this pair,
            # but can't classify it confidently. Default to MEDIUM (safer
            # than defaulting to LOW for unclassified clinical text) and
            # mark it low-confidence so it surfaces for manual review.
            candidate = SeverityResult("MEDIUM", "low", "(no keyword match)", sentence.strip())

        # Prefer higher severity when multiple sentences disagree.
        severity_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        if best is None or severity_rank[candidate.severity] > severity_rank[best.severity]:
            best = candidate

    if best and signal_conflict:
        best.confidence = "low"

    return best
