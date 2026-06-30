"""
main.py
FastAPI app exposing a single endpoint: POST /predict-mri
Loads the trained ResNet18 model and returns a tumor class + confidence score.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---- Config ----
MODEL_PATH = "../models/mri_classifier.pt"
IMG_SIZE = 224
DEVICE = torch.device("cpu")  # match train.py — keep consistent across the project

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---- App setup ----
app = FastAPI(title="MRI Tumor Classification API")

# Allows your Streamlit dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model once, at startup, not per-request ----
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)  # weights=None: we're loading our own trained weights, not ImageNet's
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()  # disables dropout/batchnorm updates — required for inference

print(f"Model loaded. Classes: {class_names}")

# Must match eval_transforms in dataset.py exactly — same resize/normalize the model was tested with
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


@app.get("/")
def root():
    return {"status": "MRI Classification API is running", "classes": class_names}


@app.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...)):
    # Basic validation — reject anything that isn't an image upfront
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    # Preprocess exactly like training/eval data
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)  # unsqueeze: add batch dimension

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
