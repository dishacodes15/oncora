"""
risk_assessment.py
Fuses MRI tumor risk, drug-drug interaction severity, and dose-safety
anomaly severity into one score, per the project plan's formula:

    final_risk_score = 0.5 * tumor_risk + 0.3 * interaction_severity + 0.2 * dose_anomaly_score

This module deliberately resolves two problems present in the plan's
formula as literally stated:

PROBLEM 1 - "tumor_confidence" is not the same thing as tumor RISK.
The MRI model's top-1 confidence score is just "how sure the model is
about whatever it predicted" - including when it predicts notumor. A
98% confident notumor prediction is NOT high risk; naively plugging
raw confidence into the formula would score it as if it were. This
module instead uses the model's FULL probability distribution
(all_probabilities, already returned by /predict-mri) and computes
tumor_risk = 1 - P(notumor). This also correctly handles genuine
uncertainty - e.g. notumor at 35% and glioma at 40% - rather than
blindly trusting whichever class happens to win the argmax.

PROBLEM 2 - unresolved data must never silently score as "safe".
A missing interaction lookup or a drug with no dose reference data
contributes 0 to a naive weighted sum, which reads as "confirmed no
risk" when it actually means "we don't know". This module calls the
lower-level check functions directly (not the already-fused
/analyze-prescription response) so it can see unresolved_pairs and
unresolved_dose_drugs explicitly, and marks the whole result
data_completeness="incomplete" whenever either is non-empty - the
caller (main.py's /assess-risk endpoint) is expected to surface that
to the client rather than presenting an incomplete score as final.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from nlp.interactions import check_prescription, SEVERITY_WEIGHT
from nlp.dose_safety import check_all_doses, HIGH_SEVERITY_MULTIPLE

# Labels treated as "no tumor present". Normalized (lowercased, spaces
# and underscores stripped) before comparison so "notumor", "no_tumor",
# and "no tumor" all match regardless of how the model's class_names
# happen to be formatted.
NON_TUMOR_LABELS = {"notumor", "nolesion", "normal"}

TUMOR_RISK_WEIGHT = 0.5
INTERACTION_SEVERITY_WEIGHT = 0.3
DOSE_ANOMALY_WEIGHT = 0.2

# Same weight scale used by interactions.py / dose_safety.py for
# HIGH/MEDIUM/LOW severity, imported here for dose alerts which only
# ever carry HIGH or MEDIUM (no LOW tier exists for dose safety).
DOSE_SEVERITY_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.55}


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "").replace("_", "")


def compute_tumor_risk(all_probabilities: dict) -> float:
    """
    tumor_risk = 1 - P(notumor), using the full probability distribution
    rather than just the top predicted class + its confidence. See
    PROBLEM 1 in the module docstring for why this matters.
    """
    non_tumor_probability = sum(
        prob for label, prob in all_probabilities.items()
        if _normalize_label(label) in NON_TUMOR_LABELS
    )
    return max(0.0, min(1.0, 1.0 - non_tumor_probability))


def compute_dose_anomaly_score(dose_alerts: list) -> float:
    """Worst dose-alert severity, on the same 0-1 scale as interaction
    severity. 0.0 if no dose alerts were raised."""
    if not dose_alerts:
        return 0.0
    return max(DOSE_SEVERITY_WEIGHT.get(a.severity, 0.0) for a in dose_alerts)


def classify_risk_level(score: float) -> str:
    """Four-tier bucketing per the project plan's stated interpretation
    (Low/Moderate/High/Critical). Thresholds are a reasonable starting
    heuristic, same status as the HIGH/MEDIUM/LOW severity thresholds
    used elsewhere in this codebase - not derived from any clinical
    validation, and should be stated as such if asked in viva."""
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.5:
        return "HIGH"
    if score >= 0.25:
        return "MODERATE"
    return "LOW"


@dataclass
class FusedRiskResult:
    final_risk_score: float
    risk_level: str
    tumor_risk_contribution: float
    interaction_severity: float
    dose_anomaly_score: float
    data_completeness: str  # "complete" | "incomplete"
    unresolved_interaction_pairs: List[Tuple[str, str]] = field(default_factory=list)
    unresolved_dose_drugs: List[str] = field(default_factory=list)
    has_low_confidence_interaction_data: bool = False


def assess_fused_risk(
    tumor_all_probabilities: dict,
    detected_drugs: set,
    interaction_store,
    dosages_by_drug: dict,
    dose_limit_store,
) -> FusedRiskResult:
    """
    Main entry point. Takes the MRI probability distribution plus
    everything needed to independently run interaction and dose
    checking, and returns one fused result.

    Deliberately calls check_prescription() and check_all_doses()
    directly rather than reusing /analyze-prescription's response -
    that endpoint already folds dose-alert severity into its own
    interaction_severity/risk_level for its own display purposes, and
    reusing that pre-fused value here would double-count it against
    this module's separate 0.3/0.2 weighting. See PROBLEM 2 in the
    module docstring for why unresolved data is tracked explicitly
    rather than left to silently zero out.
    """
    tumor_risk = compute_tumor_risk(tumor_all_probabilities)

    prescription_result = check_prescription(detected_drugs, interaction_store)
    dose_alerts, unresolved_dose_drugs = check_all_doses(dosages_by_drug, dose_limit_store)

    dose_anomaly_score = compute_dose_anomaly_score(dose_alerts)

    final_score = (
        TUMOR_RISK_WEIGHT * tumor_risk
        + INTERACTION_SEVERITY_WEIGHT * prescription_result.interaction_severity
        + DOSE_ANOMALY_WEIGHT * dose_anomaly_score
    )

    is_incomplete = bool(prescription_result.unresolved_pairs) or bool(unresolved_dose_drugs)
    has_low_confidence = any(
        i.confidence == "low" for i in prescription_result.interactions
    )

    return FusedRiskResult(
        final_risk_score=round(final_score, 4),
        risk_level=classify_risk_level(final_score),
        tumor_risk_contribution=round(tumor_risk, 4),
        interaction_severity=round(prescription_result.interaction_severity, 4),
        dose_anomaly_score=round(dose_anomaly_score, 4),
        data_completeness="incomplete" if is_incomplete else "complete",
        unresolved_interaction_pairs=prescription_result.unresolved_pairs,
        unresolved_dose_drugs=unresolved_dose_drugs,
        has_low_confidence_interaction_data=has_low_confidence,
    )
