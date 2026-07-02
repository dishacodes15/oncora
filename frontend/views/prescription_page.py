import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

MOCK_RESULT = {
    "drugs_detected": ["aspirin", "warfarin", "paracetamol"],
    "interactions": [
        {"drug_a": "aspirin", "drug_b": "warfarin",
         "severity": "HIGH", "description": "Increases bleeding risk significantly"}
    ],
    "interaction_severity": 0.85,
    "risk_level": "HIGH"
}

def show():
    st.title("Prescription Safety Checker")
    st.write("Enter a prescription to check for dangerous drug interactions.")

    prescription = st.text_area(
        "Prescription Text",
        placeholder="e.g.\nParacetamol 500mg twice daily\nWarfarin 5mg\nAspirin 100mg",
        height=150
    )

    if st.button("Check Prescription") and prescription:
        with st.spinner("Analyzing prescription..."):
            try:
                response = requests.post(f"{API_URL}/analyze-prescription",
                                        json={"text": prescription})
                result = response.json()
            except:
                result = MOCK_RESULT
                st.warning("Backend not connected — showing mock data")

        # Drugs detected
        st.subheader("Drugs Detected")
        st.write(", ".join(result['drugs_detected']) if result['drugs_detected'] else "None detected")

        # Risk level banner
        risk = result['risk_level']
        if risk == "HIGH":
            st.error(f"Risk Level: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"Risk Level: {risk}")
        else:
            st.success(f"Risk Level: {risk}")

        # Interaction alerts
        if result['interactions']:
            st.subheader("Interaction Alerts")
            for interaction in result['interactions']:
                with st.expander(f"{interaction['drug_a'].title()} + {interaction['drug_b'].title()} — {interaction['severity']}"):
                    st.write(interaction['description'])
        else:
            st.success("No dangerous interactions detected.")