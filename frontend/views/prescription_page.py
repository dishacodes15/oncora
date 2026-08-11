import streamlit as st
import requests
from style import eyebrow

API_URL = "http://127.0.0.1:8000"

MOCK_RESULT = {
    "drugs_detected": ["aspirin", "warfarin", "paracetamol"],
    "interactions": [
        {"drug_a": "aspirin", "drug_b": "warfarin",
         "severity": "HIGH", "description": "Increases bleeding risk significantly"}
    ],
    "interaction_severity": 0.85,
    "risk_level": "HIGH",
    "dose_alerts": []
}

def show():
    st.markdown(eyebrow("MODULE 02 - PRESCRIPTION NLP"), unsafe_allow_html=True)
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
                st.warning("Backend not connected - showing mock data")

        st.subheader("Drugs Detected")
        st.write(", ".join(result['drugs_detected']) if result['drugs_detected'] else "None detected")

        risk = result['risk_level']
        if risk == "HIGH":
            st.error(f"Risk Level: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"Risk Level: {risk}")
        else:
            st.success(f"Risk Level: {risk}")

        if result.get('dose_alerts'):
            st.subheader("Dose Safety Alerts")
            for alert in result['dose_alerts']:
                label = f"{alert['drug'].title()} - {alert['limit_type']} dose - {alert['severity']}"
                if alert['severity'] == "HIGH":
                    st.error(label)
                else:
                    st.warning(label)
                st.write(alert['message'])

        if result.get('unresolved_dose_drugs'):
            st.caption(
                "No dose reference data available for: "
                + ", ".join(d.title() for d in result['unresolved_dose_drugs'])
                + " - this does not mean the dose is safe, only that it wasn't checked."
            )

        if result['interactions']:
            st.subheader("Interaction Alerts")
            for interaction in result['interactions']:
                with st.expander(f"{interaction['drug_a'].title()} + {interaction['drug_b'].title()} - {interaction['severity']}"):
                    st.write(interaction['description'])
        elif not result.get('dose_alerts'):
            st.success("No dangerous interactions or dose issues detected.")
        else:
            st.info("No drug-drug interactions detected - see dose safety alerts above.")
