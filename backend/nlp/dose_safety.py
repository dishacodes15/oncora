"""
dose_safety.py
Checks extracted per-drug dosages against known safe dose ranges,
flagging doses that are unusually high relative to typical/maximum
labeled doses.

This is a SEPARATE safety check from drug-drug interaction checking
(interactions.py) - it looks at ONE drug's dose in isolation, not how
two drugs interact. A prescription can have zero interaction risk (only
one drug present, or no known interacting pairs) and still be dangerous
if that one drug's dose is far outside its safe range - e.g. "Aspirin
100g" (100,000mg against a ~4000mg/day maximum) triggers no interaction
alert at all, since there's nothing to interact with, but is clearly
unsafe.

SELF-GROWING, same pattern as interactions.py: dose_limit_store.py is
the fast-path cache (seeded once from data/dose_limits.csv via
scripts/init_db.py - that CSV is a one-time bootstrap, not something
meant to be hand-grown afterward). On a cache miss, this queries
OpenFDA's dosage_and_administration label field (tries generic name,
then brand name), runs the returned prose through
dose_prose_parser.best_limits_from_text(), and writes any usable result
back to the store - so coverage grows automatically as new drugs are
seen, exactly like the interaction store does.

Only mass-based units (mg, mcg, g) are checked. Doses given in ml, IU,
or units can't be safety-compared without drug-specific concentration or
potency data this system doesn't have - those are left unflagged rather
than guessed at.
"""

import csv
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .dose_limit_store import DoseLimitStore, DoseLimit
from .dose_prose_parser import best_limits_from_text
from .openfda_client import get_label_by_generic_name, get_label_by_brand_name, OpenFDAError

logger = logging.getLogger("oncora.nlp.dose_safety")

_MG_CONVERSION = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}

HIGH_SEVERITY_MULTIPLE = 2.0


@dataclass
class DoseAlert:
    drug: str
    extracted_amount: float
    extracted_unit: str
    limit_type: str
    limit_value: float
    severity: str
    message: str


def load_dose_limits_csv(csv_path: str) -> List[DoseLimit]:
    """Reads the seed CSV into a list of DoseLimit records, source='seed'.
    Used only by scripts/init_db.py to bootstrap the store."""
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(DoseLimit(
                drug=row["drug"].strip().lower(),
                max_single_dose_mg=float(row["max_single_dose"]),
                max_daily_dose_mg=float(row["max_daily_dose"]),
                source="seed",
                confidence="high",
                evidence=row.get("notes", "").strip(),
            ))
    return out


def _to_mg(amount: float, unit: str) -> Optional[float]:
    factor = _MG_CONVERSION.get(unit.lower())
    if factor is None:
        return None
    return amount * factor


def _fetch_dosage_text(drug_name: str) -> Optional[str]:
    """Same pattern as interactions.py's _fetch_label_interaction_text -
    try generic_name first, fall back to brand_name."""
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

    text = results[0].get("dosage_and_administration")
    if not text:
        return None
    if isinstance(text, list):
        text = " ".join(text)
    return text


def _get_or_fetch_limit(drug: str, store: DoseLimitStore) -> Optional[DoseLimit]:
    """Cache-first lookup with OpenFDA fallback + write-back."""
    cached = store.get(drug)
    if cached is not None:
        return cached

    text = _fetch_dosage_text(drug)
    if not text:
        return None

    parsed = best_limits_from_text(text)
    if parsed is None:
        return None

    limit = DoseLimit(
        drug=drug,
        max_single_dose_mg=parsed["max_single_dose_mg"],
        max_daily_dose_mg=parsed["max_daily_dose_mg"],
        source="openfda",
        confidence=parsed["confidence"],
        evidence="; ".join(parsed["evidence"]),
    )
    store.upsert(limit)
    return limit


def _evaluate_against_limit(drug: str, dosages: list, limit: DoseLimit) -> List[DoseAlert]:
    """Compares extracted dosages against a (possibly partial) DoseLimit.
    Skips whichever check has no reference value rather than guessing."""
    alerts = []
    mg_amounts = []

    for d in dosages:
        mg = _to_mg(d.amount, d.unit)
        if mg is None:
            continue
        mg_amounts.append(mg)

        if limit.max_single_dose_mg is not None and mg > limit.max_single_dose_mg:
            ratio = mg / limit.max_single_dose_mg
            severity = "HIGH" if ratio >= HIGH_SEVERITY_MULTIPLE else "MEDIUM"
            alerts.append(DoseAlert(
                drug=drug, extracted_amount=d.amount, extracted_unit=d.unit,
                limit_type="single", limit_value=limit.max_single_dose_mg,
                severity=severity,
                message=(
                    f"{d.amount:g}{d.unit} exceeds the typical maximum single "
                    f"dose of {limit.max_single_dose_mg:g}mg for {drug} "
                    f"({ratio:.1f}x). Review with a pharmacist or clinician."
                ),
            ))

    if mg_amounts and limit.max_daily_dose_mg is not None:
        total_mg = sum(mg_amounts)
        if total_mg > limit.max_daily_dose_mg:
            ratio = total_mg / limit.max_daily_dose_mg
            severity = "HIGH" if ratio >= HIGH_SEVERITY_MULTIPLE else "MEDIUM"
            alerts.append(DoseAlert(
                drug=drug, extracted_amount=total_mg, extracted_unit="mg",
                limit_type="daily", limit_value=limit.max_daily_dose_mg,
                severity=severity,
                message=(
                    f"Combined mentions total {total_mg:g}mg for {drug}, "
                    f"exceeding the typical maximum daily dose of "
                    f"{limit.max_daily_dose_mg:g}mg ({ratio:.1f}x). Review "
                    f"with a pharmacist or clinician."
                ),
            ))

    return alerts


def check_all_doses(
    dosages_by_drug: dict, store: DoseLimitStore
) -> Tuple[List[DoseAlert], List[str]]:
    """
    Runs dose checking across every drug found in a parsed prescription.
    Returns (alerts, unresolved_drugs) - unresolved_drugs lists drugs
    that had a dose mentioned but no reference limit could be found.
    """
    all_alerts: List[DoseAlert] = []
    unresolved: List[str] = []

    for drug, dosages in dosages_by_drug.items():
        if not dosages:
            continue
        limit = _get_or_fetch_limit(drug, store)
        if limit is None:
            unresolved.append(drug)
            continue
        all_alerts.extend(_evaluate_against_limit(drug, dosages, limit))

    return all_alerts, unresolved
