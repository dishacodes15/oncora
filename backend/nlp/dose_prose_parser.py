"""
dose_prose_parser.py
Extracts numeric max single/daily dose limits from OpenFDA's
dosage_and_administration label field - free-text prescribing
instructions, not a structured field. Same category of problem as
severity.py: real label language varies a lot ("do not exceed 4000 mg
in 24 hours" / "maximum recommended dose is 80 mg once daily" / "usual
dose is 500 mg every 8 hours"), so this uses tiered keyword/regex rules
rather than assuming a single fixed sentence pattern, and flags
low-confidence extractions rather than silently trusting them.

Two tiers of signal, in priority order:
  1. EXPLICIT MAX phrases ("maximum", "do not exceed", "not to exceed")
     - high confidence, these are literally stating a ceiling.
  2. USUAL/RECOMMENDED dose phrases ("usual dose", "recommended dose")
     - used only as a fallback single-dose reference when no explicit
     max is found. Always marked confidence="low".

For each candidate number+unit match, cue words within the SAME CLAUSE
(split on "." and ";") are checked for daily-cadence cues ("day",
"daily", "24 hours") vs. single-dose cues ("single dose", "per dose")
to decide whether it's a max_single_dose or max_daily_dose candidate.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

_MG_CONVERSION = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}

_NUMBER_UNIT = r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b"

_MAX_SIGNAL_RE = re.compile(
    r"(?:maximum(?:\s+recommended)?|do\s+not\s+exceed|should\s+not\s+exceed|"
    r"not\s+to\s+exceed|up\s+to\s+(?:a\s+)?(?:total\s+)?(?:dose\s+of\s+)?)"
    r"[^.]{0,25}?" + _NUMBER_UNIT,
    re.IGNORECASE,
)

_USUAL_SIGNAL_RE = re.compile(
    r"(?:usual|recommended|starting)(?:\s+adult)?[^.]{0,15}?dose[^.]{0,10}?"
    + _NUMBER_UNIT,
    re.IGNORECASE,
)

_DAILY_CUE_RE = re.compile(r"\b(?:day|daily|24\s*hours?|/day)\b", re.IGNORECASE)

_SINGLE_CUE_RE = re.compile(
    r"\b(?:single\s+dose|per\s+dose|per\s+administration|each\s+dose|one\s+dose)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(r"[.;]")


@dataclass
class DoseLimitCandidate:
    limit_type: str
    value_mg: float
    confidence: str
    evidence: str


def _to_mg(amount: float, unit: str) -> float:
    return amount * _MG_CONVERSION[unit.lower()]


def _classify_within_clause(clause: str) -> str:
    has_daily = bool(_DAILY_CUE_RE.search(clause))
    has_single = bool(_SINGLE_CUE_RE.search(clause))
    if has_daily and not has_single:
        return "daily"
    if has_single and not has_daily:
        return "single"
    return "ambiguous"


def extract_dose_candidates(text: str) -> List[DoseLimitCandidate]:
    if not text:
        return []

    candidates = []
    clauses = _CLAUSE_SPLIT_RE.split(text)

    for clause in clauses:
        for m in _MAX_SIGNAL_RE.finditer(clause):
            amount, unit = float(m.group(1)), m.group(2)
            limit_type = _classify_within_clause(clause)
            candidates.append(DoseLimitCandidate(
                limit_type=limit_type,
                value_mg=_to_mg(amount, unit),
                confidence="high" if limit_type != "ambiguous" else "low",
                evidence=m.group(0).strip(),
            ))

    if not candidates:
        for clause in clauses:
            for m in _USUAL_SIGNAL_RE.finditer(clause):
                amount, unit = float(m.group(1)), m.group(2)
                limit_type = _classify_within_clause(clause)
                candidates.append(DoseLimitCandidate(
                    limit_type=limit_type if limit_type != "ambiguous" else "single",
                    value_mg=_to_mg(amount, unit),
                    confidence="low",
                    evidence=m.group(0).strip(),
                ))

    return candidates


def best_limits_from_text(text: str) -> Optional[dict]:
    candidates = extract_dose_candidates(text)
    if not candidates:
        return None

    single_candidates = [c for c in candidates if c.limit_type == "single"]
    daily_candidates = [c for c in candidates if c.limit_type == "daily"]
    ambiguous_candidates = [c for c in candidates if c.limit_type == "ambiguous"]

    result = {
        "max_single_dose_mg": None,
        "max_daily_dose_mg": None,
        "confidence": "high",
        "evidence": [],
    }

    if single_candidates:
        best = min(single_candidates, key=lambda c: c.value_mg)
        result["max_single_dose_mg"] = best.value_mg
        result["evidence"].append(best.evidence)
        if best.confidence == "low":
            result["confidence"] = "low"

    if daily_candidates:
        best = min(daily_candidates, key=lambda c: c.value_mg)
        result["max_daily_dose_mg"] = best.value_mg
        result["evidence"].append(best.evidence)
        if best.confidence == "low":
            result["confidence"] = "low"

    if not single_candidates and not daily_candidates and ambiguous_candidates:
        best = min(ambiguous_candidates, key=lambda c: c.value_mg)
        result["max_daily_dose_mg"] = best.value_mg
        result["evidence"].append(best.evidence)
        result["confidence"] = "low"

    if result["max_single_dose_mg"] is None and result["max_daily_dose_mg"] is None:
        return None

    return result
