import streamlit as st
from views import mri_page, prescription_page, federated_page
from style import inject_css

st.set_page_config(
    page_title="Oncora",
    page_icon="🧠",
    layout="wide",
)

inject_css()

st.sidebar.title("ONCORA")

page = st.sidebar.radio(
    "Navigate",
    ["MRI Classification", "Prescription Safety", "Federated Monitor"],
    label_visibility="collapsed",
)

if page == "MRI Classification":
    mri_page.show()
elif page == "Prescription Safety":
    prescription_page.show()
elif page == "Federated Monitor":
    federated_page.show()