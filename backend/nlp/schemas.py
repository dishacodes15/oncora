"""
schemas.py
Pydantic models for /analyze-prescription. Kept separate from main.py so
the NLP package is self-contained and importable/testable without pulling
in the MRI endpoint's dependencies.

The shape here matches what frontend/views/prescription_page.py already
expects: drugs_detected, interactions (drug_a/drug_b/severity/description),
interaction_severity, risk_level. confidence, unresolved_pairs, and the
drug_sources field are additive - old frontend code that does not read
them keeps working.
"""

from typing import List, Tuple
from pydantic import BaseModel, Field


class PrescriptionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class InteractionOut(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    description: str
    source: str
    confidence: str = "high"


class DoseAlertOut(BaseModel):
    drug: str
    extracted_amount: float
    extracted_unit: str
    limit_type: str
    limit_value: float
    severity: str
    message: str


class DrugSourceOut(BaseModel):
    """Per-drug provenance from the hybrid NER pipeline - lets a caller
    (or a clinician reading the response) see exactly which drugs were
    found via exact vocabulary matching (confidence=1.0, deterministic)
    versus the biomedical NER model's fallback (confidence is the
    model's own score, typically 0.6-0.95, used only for drugs the
    vocabulary did not already find)."""
    drug: str
    source: str        # "vocabulary" | "biobert_fallback"
    confidence: float


class PrescriptionResponse(BaseModel):
    drugs_detected: List[str]
    interactions: List[InteractionOut]
    interaction_severity: float
    risk_level: str
    unresolved_pairs: List[Tuple[str, str]] = []
    dose_alerts: List[DoseAlertOut] = []
    unresolved_dose_drugs: List[str] = []
    drug_sources: List[DrugSourceOut] = []
    biobert_available: bool = False
