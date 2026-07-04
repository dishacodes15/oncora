import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

        html, body, .stApp { font-family: 'Inter', sans-serif !important; }

        h1 {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: 2px !important;
        }
        section[data-testid="stSidebar"] h1 {
            font-size: 2.25rem !important;
        }
        h2, h3 {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
        }

        div[data-testid="stRadio"] > label {
            font-size: 11px !important;
            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
            opacity: 0.4;
        }

        .stButton > button {
            font-family: 'Rajdhani', sans-serif !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            letter-spacing: 1.5px !important;
            border-radius: 6px !important;
            transition: background 0.2s ease !important;
        }

        div[data-testid="metric-container"] {
            border: 1px solid rgba(0, 200, 248, 0.22) !important;
            border-radius: 10px !important;
            padding: 16px !important;
        }
        div[data-testid="metric-container"] label {
            font-size: 10px !important;
            letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
            opacity: 0.5;
        }
        div[data-testid="stMetricValue"] > div {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 28px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploader"] > div {
            border: 1.5px dashed rgba(0, 200, 248, 0.38) !important;
            border-radius: 10px !important;
            background: rgba(0, 200, 248, 0.02) !important;
        }

        .stTextArea textarea {
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
        }

        details {
            border-radius: 8px !important;
            border: 1px solid rgba(0, 200, 248, 0.18) !important;
        }

        div[data-testid="stArrowVegaLiteChart"],
        div[data-testid="stVegaLiteChart"] {
            border-radius: 10px !important;
            border: 1px solid rgba(0, 200, 248, 0.12) !important;
            padding: 10px !important;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 200, 248, 0.3);
            border-radius: 2px;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )