import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000"

TUMOR_INFO = [
    (
        "Glioma", "#00c8f8",
        "Originates in glial cells — the most common malignant brain tumor, "
        "accounting for ~80% of cases. Aggressive and fast-growing.",
    ),
    (
        "Meningioma", "#8b5cf6",
        "Arises from the meninges surrounding the brain and spinal cord. "
        "Usually benign and slow-growing, but can compress brain tissue.",
    ),
    (
        "Pituitary", "#22d3ee",
        "Develops in the pituitary gland at the brain's base. Typically "
        "non-cancerous, but disrupts hormone regulation.",
    ),
    (
        "No Tumor", "#4ade80",
        "No detectable malignancy found in the scan. All regions appear "
        "within normal parameters.",
    ),
]


def _info_card(title, color, desc):
    return f"""
    <div style="border:1px solid {color}28;border-left:3px solid {color};
                border-radius:8px;padding:12px 14px;margin-bottom:10px;
                background:{color}0a">
      <div style="font-family:'Rajdhani',sans-serif;font-size:15px;
                  font-weight:600;color:{color};margin-bottom:5px">{title}</div>
      <div style="font-size:12px;color:rgba(184,216,232,0.55);
                  line-height:1.6">{desc}</div>
    </div>
    """


def show():
    st.title("Brain MRI Classification")
    st.write("Upload an MRI scan to detect tumor type.")

    col_upload, col_info = st.columns([3, 2], gap="large")

    # ── Left: upload + results ─────────────────────────────────────────────
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload MRI Image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_container_width=True)

        analyze = st.button("Analyze Scan")

        if analyze:
            if not uploaded_file:
                st.error("Upload an MRI image first.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict-mri",
                            files={
                                "file": (
                                    "mri.jpg",
                                    uploaded_file.getvalue(),
                                    "image/jpeg",
                                )
                            },
                        )
                        result = response.json()
                    except Exception:
                        result = {
                            "prediction": "glioma",
                            "confidence": 0.91,
                            "all_probabilities": {
                                "glioma": 0.91,
                                "meningioma": 0.05,
                                "notumor": 0.03,
                                "pituitary": 0.01,
                            },
                        }
                        st.warning("Backend not connected — showing mock data")

                st.success(f"Prediction: **{result['prediction'].upper()}**")

                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                c2.metric("Status", "Analysis complete")

                st.subheader("Class Probabilities")
                st.bar_chart(result["all_probabilities"])

    # ── Right: always-visible info panel ──────────────────────────────────
    with col_info:
        st.subheader("Detectable Conditions")
        st.markdown(
            "".join(_info_card(t, c, d) for t, c, d in TUMOR_INFO),
            unsafe_allow_html=True,
        )

        st.subheader("Model Info")
        mi1, mi2 = st.columns(2)
        mi1.metric("Architecture", "CNN")
        mi2.metric("Val Accuracy", "91.0%")