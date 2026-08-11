import streamlit as st
from views import mri_page, prescription_page, federated_page, nlp_sanity_page, risk_dashboard_page
from style import inject_css

st.set_page_config(
    page_title="Oncora",
    page_icon="\U0001F9E0",
    layout="wide",
)

inject_css()

st.sidebar.title("ONCORA")
st.sidebar.markdown(
    '<div class="oncora-byline">Federated Learning for Clinical Diagnosis '
    'and Prescription Safety</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Unified Risk Score",
        "MRI Classification",
        "Prescription Safety",
        "Federated Monitor",
        "NLP Sanity Check",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    '<div class="oncora-footer">Aarya Mishra &middot; Aayushi Gupta &middot; '
    "Disha Madhusudhana<br>Academic prototype - not for clinical use</div>",
    unsafe_allow_html=True,
)

if page == "Unified Risk Score":
    risk_dashboard_page.show()
elif page == "MRI Classification":
    mri_page.show()
elif page == "Prescription Safety":
    prescription_page.show()
elif page == "Federated Monitor":
    federated_page.show()
elif page == "NLP Sanity Check":
    nlp_sanity_page.show()
