"""
main.py
FastAPI app exposing:
  POST /predict-mri            - tumor class + confidence from an MRI image
  POST /analyze-prescription   - drug/dosage extraction + interaction check
  POST /assess-risk            - fused MRI + prescription risk score
"""

import os
import logging

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nlp.parser import get_parser
from nlp.interaction_store import InteractionStore
from nlp.interactions import check_prescription
from nlp.dose_limit_store import DoseLimitStore
from nlp.dose_safety import check_all_doses
from nlp.hybrid_ner import extract_drugs_hybrid
from nlp.biobert_ner import get_ner_pipeline, get_biobert_stats
from nlp.schemas import (
    PrescriptionRequest, PrescriptionResponse, InteractionOut,
    DoseAlertOut, DrugSourceOut,
)
from risk_assessment import assess_fused_risk

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oncora.backend")

# ---- Config ----
MODEL_PATH = "../models/mri_classifier.pt"
IMG_SIZE = 224
DEVICE = torch.device("cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INTERACTIONS_DB_PATH = os.environ.get("INTERACTIONS_DB_PATH", "../data/interactions.db")
DOSE_LIMITS_DB_PATH = os.environ.get("DOSE_LIMITS_DB_PATH", "../data/dose_limits.db")

# ---- App setup ----
app = FastAPI(title="Oncora API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load MRI model once, at startup ----
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, len(class_names))
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"Model loaded. Classes: {class_names}")

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ---- Load NLP pipeline + interaction/dose stores once, at startup ----
prescription_parser = None
interaction_store = None
dose_limit_store = None
try:
    prescription_parser = get_parser()
    interaction_store = InteractionStore(INTERACTIONS_DB_PATH)
    dose_limit_store = DoseLimitStore(DOSE_LIMITS_DB_PATH)
    logger.info("Prescription NLP pipeline ready.")
except Exception as exc:
    logger.warning(
        "Prescription NLP pipeline failed to load (%s). "
        "/analyze-prescription and /assess-risk will return 503 until "
        "this is fixed.", exc,
    )

# ---- Load BioBERT NER once, at startup - OPTIONAL, not required ----
biobert_ready = False
try:
    get_ner_pipeline()
    biobert_ready = True
    logger.info("BioBERT biomedical NER pipeline ready (fallback mode).")
except Exception as exc:
    logger.warning(
        "BioBERT NER failed to load (%s). Prescription extraction will "
        "continue with vocabulary-only matching.", exc,
    )


@app.get("/")
def root():
    return {
        "status": "Oncora API is running",
        "mri_classes": class_names,
        "prescription_nlp_ready": prescription_parser is not None,
        "biobert_ready": biobert_ready,
        "biobert_stats": get_biobert_stats(),
    }


def _run_mri_inference(image: Image.Image) -> dict:
    """Shared MRI inference logic used by both /predict-mri and
    /assess-risk, so there is exactly one place that does this, not two
    copies that could drift apart."""
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return {
        "prediction": class_names[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
        "all_probabilities": {
            class_names[i]: round(probabilities[i].item(), 4)
            for i in range(len(class_names))
        },
    }


@app.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    return _run_mri_inference(image)


def _run_hybrid_prescription_analysis(text: str) -> PrescriptionResponse:
    """Shared prescription analysis logic used by both
    /analyze-prescription and /assess-risk."""
    parsed = prescription_parser.parse(text)
    hybrid_result = extract_drugs_hybrid(text, prescription_parser)
    combined_drugs = hybrid_result.drug_names()

    if not combined_drugs:
        return PrescriptionResponse(
            drugs_detected=[],
            interactions=[],
            interaction_severity=0.0,
            risk_level="LOW",
            unresolved_pairs=[],
            dose_alerts=[],
            unresolved_dose_drugs=[],
            drug_sources=[],
            biobert_available=hybrid_result.biobert_available,
        ), parsed, hybrid_result, combined_drugs

    result = check_prescription(combined_drugs, interaction_store)
    dose_alert_results, unresolved_dose_drugs = check_all_doses(
        parsed.dosages_by_drug, dose_limit_store
    )

    severity_weight = {"HIGH": 1.0, "MEDIUM": 0.55, "LOW": 0.2}
    overall_severity = severity_weight.get(result.risk_level, 0.2)
    overall_risk_level = result.risk_level
    for alert in dose_alert_results:
        w = severity_weight.get(alert.severity, 0.0)
        if w > overall_severity:
            overall_severity = w
            overall_risk_level = alert.severity

    response = PrescriptionResponse(
        drugs_detected=result.drugs_detected,
        interactions=[
            InteractionOut(
                drug_a=i.drug_a, drug_b=i.drug_b, severity=i.severity,
                description=i.description, source=i.source, confidence=i.confidence,
            )
            for i in result.interactions
        ],
        interaction_severity=overall_severity,
        risk_level=overall_risk_level,
        unresolved_pairs=result.unresolved_pairs,
        dose_alerts=[
            DoseAlertOut(
                drug=a.drug, extracted_amount=a.extracted_amount,
                extracted_unit=a.extracted_unit, limit_type=a.limit_type,
                limit_value=a.limit_value, severity=a.severity, message=a.message,
            )
            for a in dose_alert_results
        ],
        unresolved_dose_drugs=unresolved_dose_drugs,
        drug_sources=[
            DrugSourceOut(drug=e.drug, source=e.source, confidence=e.confidence)
            for e in hybrid_result.entities
        ],
        biobert_available=hybrid_result.biobert_available,
    )
    return response, parsed, hybrid_result, combined_drugs


@app.post("/analyze-prescription", response_model=PrescriptionResponse)
async def analyze_prescription(payload: PrescriptionRequest):
    if prescription_parser is None or interaction_store is None or dose_limit_store is None:
        raise HTTPException(
            status_code=503,
            detail="Prescription NLP pipeline is not available. Check server logs.",
        )
    response, _, _, _ = _run_hybrid_prescription_analysis(payload.text)
    return response


@app.post("/assess-risk")
async def assess_risk(
    file: UploadFile = File(...),
    prescription_text: str = Form(...),
):
    """
    Fuses MRI tumor risk with prescription interaction/dose risk into
    one score, per the project plan's unified risk formula:
        0.5 * tumor_risk + 0.3 * interaction_severity + 0.2 * dose_anomaly_score

    Takes an MRI image AND prescription text in a single multipart
    request, since fusion needs both inputs together. Reuses the exact
    same model/parser/stores already loaded at startup - no duplicate
    loading, no second copy of the MRI or NLP logic. See
    risk_assessment.py for the fusion math itself, including how it
    corrects two problems with the plan's formula as literally stated
    (raw confidence is not the same as risk; unresolved data must not
    silently score as safe).
    """
    if prescription_parser is None or interaction_store is None or dose_limit_store is None:
        raise HTTPException(
            status_code=503,
            detail="Prescription NLP pipeline is not available. Check server logs.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    mri_result = _run_mri_inference(image)

    prescription_response, parsed, hybrid_result, combined_drugs = \
        _run_hybrid_prescription_analysis(prescription_text)

    fused = assess_fused_risk(
        tumor_all_probabilities=mri_result["all_probabilities"],
        detected_drugs=combined_drugs,
        interaction_store=interaction_store,
        dosages_by_drug=parsed.dosages_by_drug,
        dose_limit_store=dose_limit_store,
    )

    return {
        "final_risk_score": fused.final_risk_score,
        "risk_level": fused.risk_level,
        "data_completeness": fused.data_completeness,
        "breakdown": {
            "tumor_risk_contribution": fused.tumor_risk_contribution,
            "interaction_severity": fused.interaction_severity,
            "dose_anomaly_score": fused.dose_anomaly_score,
        },
        "mri_prediction": mri_result["prediction"],
        "mri_confidence": mri_result["confidence"],
        "mri_all_probabilities": mri_result["all_probabilities"],
        "drugs_detected": sorted(combined_drugs),
        "drug_sources": [
            {"drug": e.drug, "source": e.source, "confidence": e.confidence}
            for e in hybrid_result.entities
        ],
        "unresolved_interaction_pairs": fused.unresolved_interaction_pairs,
        "unresolved_dose_drugs": fused.unresolved_dose_drugs,
        "has_low_confidence_interaction_data": fused.has_low_confidence_interaction_data,
        "biobert_available": hybrid_result.biobert_available,
    }

