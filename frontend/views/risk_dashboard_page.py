import streamlit as st
import requests
from PIL import Image

from style import eyebrow, risk_badge_html

API_URL = "http://127.0.0.1:8000"

MOCK_RESULT = {
    "final_risk_score": 0.62,
    "risk_level": "HIGH",
    "data_completeness": "incomplete",
    "breakdown": {
        "tumor_risk_contribution": 0.85,
        "interaction_severity": 0.55,
        "dose_anomaly_score": 0.0,
    },
    "mri_prediction": "glioma",
    "mri_confidence": 0.85,
    "mri_all_probabilities": {
        "glioma": 0.85, "meningioma": 0.09, "notumor": 0.04, "pituitary": 0.02,
    },
    "drugs_detected": ["aspirin", "warfarin"],
    "drug_sources": [
        {"drug": "aspirin", "source": "vocabulary", "confidence": 1.0},
        {"drug": "warfarin", "source": "vocabulary", "confidence": 1.0},
    ],
    "unresolved_interaction_pairs": [],
    "unresolved_dose_drugs": [],
    "has_low_confidence_interaction_data": False,
    "biobert_available": True,
}


def _breakdown_bar(label: str, value: float, weight_label: str, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return f"""
    <div style="margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;font-size:12px;
                  color:#333333;margin-bottom:4px">
        <span>{label} <span style="opacity:0.6">({weight_label})</span></span>
        <span style="font-family:'IBM Plex Mono',monospace;font-weight:600">{value:.3f}</span>
      </div>
      <div style="background:#f2f2ee;border-radius:2px;height:10px;overflow:hidden">
        <div style="background:{color};width:{pct}%;height:100%"></div>
      </div>
    </div>
    """


def show():
    st.markdown(eyebrow("MODULE 04 - INTEGRATION LAYER"), unsafe_allow_html=True)
    st.title("Unified Risk Assessment")
    st.write(
        "Fuses tumor classification risk with prescription interaction and "
        "dose-safety risk into a single score: "
        "`0.5 x tumor_risk + 0.3 x interaction_severity + 0.2 x dose_anomaly_score`"
    )

    col_mri, col_rx = st.columns(2, gap="large")

    with col_mri:
        st.subheader("MRI Scan")
        uploaded_file = st.file_uploader(
            "Upload MRI Image", type=["jpg", "jpeg", "png"], key="risk_mri_upload"
        )
        if uploaded_file:
            st.image(Image.open(uploaded_file), use_container_width=True)

    with col_rx:
        st.subheader("Prescription Text")
        prescription_text = st.text_area(
            "Enter prescription",
            height=200,
            placeholder="e.g. Aspirin 100mg daily. Warfarin 5mg once a day.",
            key="risk_rx_text",
        )

    run = st.button("Run Unified Assessment", use_container_width=True)

    if not run:
        return

    if not uploaded_file:
        st.error("Upload an MRI image first.")
        return
    if not prescription_text.strip():
        st.error("Enter prescription text first.")
        return

    with st.spinner("Running MRI classification, prescription NLP, and risk fusion..."):
        try:
            response = requests.post(
                f"{API_URL}/assess-risk",
                files={"file": ("mri.jpg", uploaded_file.getvalue(), "image/jpeg")},
                data={"prescription_text": prescription_text},
                timeout=60,
            )
            result = response.json()
        except Exception:
            result = MOCK_RESULT
            st.warning("Backend not connected - showing mock data")

    st.divider()

    # Top-line result
    top_l, top_r = st.columns([2, 1])
    with top_l:
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
            f'color:#444444;letter-spacing:1px;text-transform:uppercase">'
            f"Final Risk Score</div>"
            f'<div style="font-family:\'Playfair Display\',serif;font-size:48px;'
            f'font-weight:700;color:#111111;line-height:1.1">{result["final_risk_score"]:.3f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(risk_badge_html(result["risk_level"]), unsafe_allow_html=True)
    with top_r:
        if result.get("data_completeness") == "incomplete":
            st.warning(
                "**Incomplete data** - some drug pairs or doses could not be "
                "checked. This score reflects only what was resolvable, not a "
                "confirmed absence of risk."
            )
        else:
            st.success("All interaction and dose data resolved for this assessment.")

    st.subheader("Score Breakdown")
    st.markdown(
        _breakdown_bar("Tumor Risk", result["breakdown"]["tumor_risk_contribution"], "weight 0.5", "#b31b1b")
        + _breakdown_bar("Interaction Severity", result["breakdown"]["interaction_severity"], "weight 0.3", "#111111")
        + _breakdown_bar("Dose Anomaly", result["breakdown"]["dose_anomaly_score"], "weight 0.2", "#8a6d1f"),
        unsafe_allow_html=True,
    )

    st.divider()

    # MRI + Prescription detail, side by side
    detail_l, detail_r = st.columns(2, gap="large")

    with detail_l:
        st.subheader("MRI Finding")
        m1, m2 = st.columns(2)
        m1.metric("Prediction", result["mri_prediction"].upper())
        m2.metric("Confidence", f"{result['mri_confidence']*100:.1f}%")
        st.bar_chart(result["mri_all_probabilities"])

    with detail_r:
        st.subheader("Drugs Detected")
        for src in result.get("drug_sources", []):
            tag_color = "#111111" if src["source"] == "vocabulary" else "#b31b1b"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 12px;margin-bottom:6px;border:1px solid #111111;'
                f'border-radius:2px;background:#ffffff">'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-weight:600">{src["drug"]}</span>'
                f'<span style="font-size:11px;color:{tag_color};text-transform:uppercase;'
                f'letter-spacing:0.5px">{src["source"]} - {src["confidence"]:.2f}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        if not result.get("biobert_available", False):
            st.caption("BioBERT fallback unavailable this run - vocabulary matching only.")

    if result.get("unresolved_interaction_pairs") or result.get("unresolved_dose_drugs"):
        st.divider()
        st.subheader("Unresolved - Not Checked, Not Confirmed Safe")
        if result.get("unresolved_interaction_pairs"):
            pairs = ", ".join(f"{a} + {b}" for a, b in result["unresolved_interaction_pairs"])
            st.write(f"**Interaction pairs:** {pairs}")
        if result.get("unresolved_dose_drugs"):
            st.write(f"**Dose limits:** {', '.join(result['unresolved_dose_drugs'])}")
