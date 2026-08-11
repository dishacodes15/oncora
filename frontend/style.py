import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Source+Serif+4:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, .stApp {
            font-family: 'IBM Plex Sans', sans-serif !important;
            background-color: #fdfdfb !important;
            color: #111111 !important;
        }

        h1 {
            font-family: 'Playfair Display', Georgia, serif !important;
            font-weight: 900 !important;
            color: #111111 !important;
            letter-spacing: 0.3px !important;
        }
        h2, h3 {
            font-family: 'Source Serif 4', Georgia, serif !important;
            font-weight: 700 !important;
            color: #111111 !important;
        }
        section[data-testid="stSidebar"] h1 {
            font-size: 2.1rem !important;
            border-bottom: 3px solid #111111 !important;
            padding-bottom: 10px !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #f2f2ee !important;
            border-right: 2px solid #111111 !important;
        }
        .oncora-byline {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 11px;
            font-style: italic;
            letter-spacing: 0.3px;
            color: #333333;
            margin-top: 6px;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid #111111;
        }
        .oncora-footer {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 10.5px;
            color: #444444;
            line-height: 1.6;
            position: fixed;
            bottom: 18px;
            width: 15rem;
            border-top: 1px solid #111111;
            padding-top: 10px;
        }

        div[data-testid="stRadio"] > label {
            font-size: 11px !important;
            letter-spacing: 1.2px !important;
            text-transform: uppercase !important;
            color: #555555 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            font-family: 'IBM Plex Sans', sans-serif !important;
        }
        /* Hardening: force readable text on every element inside the
           radio widget, not just the (hidden) group label - Streamlit's
           per-option text otherwise inherits whatever the active
           browser/OS theme dictates, which can be invisible if a user's
           browser is set to dark mode despite this app's light config. */
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label * {
            color: #333333 !important;
        }

        .oncora-eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #b31b1b;
            font-weight: 600;
            margin-bottom: 2px;
        }

        .stButton > button {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            border-radius: 2px !important;
            background-color: #111111 !important;
            color: #fdfdfb !important;
            border: 1.5px solid #111111 !important;
            transition: background 0.15s ease !important;
        }
        .stButton > button:hover {
            background-color: #b31b1b !important;
            border-color: #b31b1b !important;
            color: #ffffff !important;
        }

        div[data-testid="metric-container"] {
            border: 1.5px solid #111111 !important;
            border-radius: 2px !important;
            padding: 16px !important;
            background: #ffffff !important;
        }
        div[data-testid="metric-container"] label {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 10.5px !important;
            letter-spacing: 1.2px !important;
            text-transform: uppercase !important;
            color: #555555 !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 27px !important;
            font-weight: 700 !important;
            color: #111111 !important;
        }

        div[data-testid="stFileUploader"] > div {
            border: 2px dashed #111111 !important;
            border-radius: 2px !important;
            background: #f9f9f6 !important;
        }

        .stTextArea textarea {
            border-radius: 2px !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            border: 1.5px solid #111111 !important;
        }

        details {
            border-radius: 2px !important;
            border: 1.5px solid #111111 !important;
            background: #ffffff !important;
        }

        div[data-testid="stArrowVegaLiteChart"],
        div[data-testid="stVegaLiteChart"] {
            border-radius: 2px !important;
            border: 1.5px solid #111111 !important;
            padding: 10px !important;
            background: #ffffff !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 2px !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            border: 1.5px solid currentColor !important;
        }
        /* Hardening: same reasoning as the radio fix above - force
           readable dark text inside warning/error/success/info boxes
           regardless of active browser/OS theme. */
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] div,
        div[data-testid="stAlert"] span {
            color: #111111 !important;
        }

        .oncora-risk-badge {
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 6px 18px;
            border-radius: 2px;
            border: 2px solid currentColor;
        }

        hr {
            border-top: 2px solid #111111 !important;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb {
            background: #111111;
            border-radius: 0px;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def eyebrow(text: str) -> str:
    """Small-caps red monospace label rendered above a section header,
    e.g. eyebrow('MODULE 01') above st.title('MRI Classification') -
    mimics a newspaper section kicker. Returns HTML; caller must render
    with unsafe_allow_html=True."""
    return f'<div class="oncora-eyebrow">{text}</div>'


def risk_badge_html(risk_level: str) -> str:
    """Color-coded badge for LOW/MODERATE/HIGH/CRITICAL risk levels.
    CRITICAL is solid black (the most severe, distinct from HIGH's red)
    rather than a deeper red - reads as a 'stop press' banner rather
    than just a darker version of HIGH."""
    level = risk_level.upper()
    if level == "CRITICAL":
        return (
            '<span class="oncora-risk-badge" '
            'style="color:#fdfdfb;background:#111111;border-color:#111111">'
            f"{level}</span>"
        )
    colors = {
        "LOW": "#1e5631",
        "MODERATE": "#8a6d1f",
        "HIGH": "#b31b1b",
    }
    color = colors.get(level, "#111111")
    return f'<span class="oncora-risk-badge" style="color:{color}">{level}</span>'
