

```
 ██████╗ ███╗   ██╗ ██████╗ ██████╗ ██████╗  █████╗
██╔═══██╗████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔══██╗
██║   ██║██╔██╗ ██║██║     ██║   ██║██████╔╝███████║
██║   ██║██║╚██╗██║██║     ██║   ██║██╔══██╗██╔══██║
╚██████╔╝██║ ╚████║╚██████╗╚██████╔╝██║  ██║██║  ██║
 ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
```




# Oncora

Federated learning system for clinical diagnosis — brain tumor classification and prescription safety analysis across distributed hospital nodes, without sharing raw patient data.

---

## What it does

**MRI Classification** — Upload a brain MRI scan and get a tumor type prediction (glioma, meningioma, pituitary, or no tumor) with confidence scores.

**Prescription Safety** — Input a prescription and check for dangerous drug interactions with risk-level alerts.

**Federated Monitoring** — Live view of hospital training nodes, global model accuracy across rounds, and federation stats.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| ML Model | CNN (brain MRI classification) |
| NLP | Drug interaction analysis |
| Federation | Federated learning across simulated hospital nodes |

---

## Running locally

**Backend**
```bash
cd backend
uvicorn main:app --reload
```

**Frontend**
```bash
cd frontend
pip install streamlit requests plotly pillow
streamlit run app.py
```

The frontend runs with mock data if the backend isn't running — so you can build and test the UI independently.

---

## Project structure 
```
oncora/
├── backend/        # FastAPI — MRI prediction + prescription analysis APIs
├── frontend/       # Streamlit dashboard
├── notebooks/      # Model training and experimentation
└── docs/
```

---

## Team

Built as part of a federated learning research project.
