"""
interactions.py
Orchestrates interaction checking for a set of detected drugs:

  1. For every pair, check the local SQLite store first (fast path).
  2. On a miss, query OpenFDA's /drug/label.json for each drug's label,
     pull drug_interactions prose, and run it through severity.classify()
     for the OTHER drug in the pair.
  3. Every successful OpenFDA-derived result gets written back to the
     store (self-growing cache — step 8).
  4. If OpenFDA has nothing (no label on file, or no mention of the
     other drug), the pair is reported as "unknown" rather than assumed
     safe — this matters for health-adjacent output.

This is the module check_prescription() in main.py's endpoint calls.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .interaction_store import InteractionStore, Interaction
from .openfda_client import get_label_by_generic_name, get_label_by_brand_name, OpenFDAError
from .severity import classify as classify_severity

logger = logging.getLogger("oncora.nlp.interactions")

SEVERITY_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.55, "LOW": 0.2}


@dataclass
class InteractionResult:
    drug_a: str
    drug_b: str
    severity: str                  # HIGH | MEDIUM | LOW
    description: str
    source: str                     # cache | openfda
    confidence: str = "high"        # high | low — surfaced so the UI/API caller
                                     # can flag machine-parsed, unverified results


@dataclass
class PrescriptionCheckResult:
    drugs_detected: List[str]
    interactions: List[InteractionResult] = field(default_factory=list)
    unresolved_pairs: List[tuple] = field(default_factory=list)  # pairs we couldn't find info on
    interaction_severity: float = 0.0   # 0-1 aggregate, for the UI's summary metric
    risk_level: str = "LOW"             # HIGH | MEDIUM | LOW, derived from the worst pair found


def _fetch_label_interaction_text(drug_name: str) -> Optional[str]:
    """Try generic_name first, fall back to brand_name — OpenFDA's
    generic_name facet doesn't cover every product on file."""
    try:
        data = get_label_by_generic_name(drug_name)
        results = data.get("results", [])
        if not results:
            data = get_label_by_brand_name(drug_name)
            results = data.get("results", [])
    except OpenFDAError as exc:
        logger.warning("OpenFDA label lookup failed for %r: %s", drug_name, exc)
        return None

    if not results:
        logger.info("No OpenFDA label found for %r (tried generic + brand name)", drug_name)
        return None

    interaction_text = results[0].get("drug_interactions")
    if not interaction_text:
        return None
    # drug_interactions is usually returned as a list of one long string
    if isinstance(interaction_text, list):
        interaction_text = " ".join(interaction_text)
    return interaction_text


def _check_pair_via_openfda(drug_a: str, drug_b: str) -> Optional[InteractionResult]:
    """
    Looks up drug_a's label and scans its drug_interactions text for
    mentions of drug_b. If nothing found there, tries drug_b's label for
    mentions of drug_a (labels aren't always symmetric about which drug
    mentions which — try both directions before giving up).
    """
    text_a = _fetch_label_interaction_text(drug_a)
    if text_a:
        result = classify_severity(text_a, other_drug=drug_b)
        if result:
            return InteractionResult(
                drug_a=drug_a, drug_b=drug_b, severity=result.severity,
                description=result.evidence_sentence, source="openfda",
                confidence=result.confidence,
            )

    text_b = _fetch_label_interaction_text(drug_b)
    if text_b:
        result = classify_severity(text_b, other_drug=drug_a)
        if result:
            return InteractionResult(
                drug_a=drug_a, drug_b=drug_b, severity=result.severity,
                description=result.evidence_sentence, source="openfda",
                confidence=result.confidence,
            )

    return None


def check_prescription(drugs: Set[str], store: InteractionStore) -> PrescriptionCheckResult:
    """
    Main entry point. Takes the set of drugs extracted from a prescription
    (from parser.py) and returns a full interaction report: cache hits,
    OpenFDA-derived hits (written back to cache), and any pairs we
    genuinely couldn't find information on.
    """
    drug_list = sorted(drugs)
    result = PrescriptionCheckResult(drugs_detected=drug_list)

    pairs = [
        (drug_list[i], drug_list[j])
        for i in range(len(drug_list))
        for j in range(i + 1, len(drug_list))
    ]

    for drug_a, drug_b in pairs:
        cached = store.get(drug_a, drug_b)
        if cached is not None:
            result.interactions.append(InteractionResult(
                drug_a=cached.drug_a, drug_b=cached.drug_b, severity=cached.severity,
                description=cached.description, source="cache", confidence=cached.confidence,
            ))
            continue

        # Cache miss -> OpenFDA fallback
        hit = _check_pair_via_openfda(drug_a, drug_b)
        if hit is None:
            result.unresolved_pairs.append((drug_a, drug_b))
            continue

        result.interactions.append(hit)

        # Self-growing cache (step 8): write back so next time this pair
        # is a fast local hit instead of another OpenFDA round-trip.
        store.upsert(Interaction(
            drug_a=hit.drug_a, drug_b=hit.drug_b, severity=hit.severity,
            description=hit.description, source="openfda", confidence=hit.confidence,
        ))

    # Aggregate severity score + overall risk level for the UI's summary banner.
    if result.interactions:
        result.interaction_severity = max(
            SEVERITY_WEIGHT[i.severity] for i in result.interactions
        )
        worst = max(result.interactions, key=lambda i: SEVERITY_WEIGHT[i.severity])
        result.risk_level = worst.severity
    else:
        result.interaction_severity = 0.0
        result.risk_level = "LOW"

    return result
