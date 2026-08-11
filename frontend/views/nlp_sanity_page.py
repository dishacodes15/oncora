import json
import os
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# frontend/views/ -> project root is two levels up. Needed to import
# scripts.sanity_check, which lives outside the frontend/ tree.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sanity_check import run_round, save_round, DEFAULT_HISTORY_PATH  # noqa: E402

HISTORY_PATH = os.environ.get("NLP_EVAL_HISTORY_PATH", DEFAULT_HISTORY_PATH)


def _load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def show():
    st.title("Prescription NLP Sanity Check")
    st.write(
        "Randomized, repeatable check of the drug-extraction and interaction "
        "pipeline against known ground-truth drug pairs — same round-based "
        "accuracy tracking as the Federated Monitor tab, applied here to the "
        "prescription NLP pipeline instead of the MRI model."
    )

    col_samples, col_run = st.columns([3, 1])
    with col_samples:
        sample_size = st.slider("Random drug pairs to sample this round",
                                 min_value=5, max_value=50, value=10)
    with col_run:
        st.write("")
        st.write("")
        run_clicked = st.button("Run New Round", use_container_width=True)

    if run_clicked:
        with st.spinner(f"Running {sample_size} random drug-pair checks against the live pipeline..."):
            try:
                result = run_round(sample_size=sample_size)
                save_round(result)
                st.success(
                    f"Round complete — {result['accuracy']:.1%} overall accuracy "
                    f"({result['n_samples']} samples)"
                )
            except Exception as e:
                st.error(f"Sanity check failed: {e}")

    history = _load_history()

    if not history:
        st.warning("No sanity check history yet — run a round above to get started.")
        return

    st.subheader("Accuracy Over Rounds")
    rounds = list(range(1, len(history) + 1))
    accuracy = [h["accuracy"] for h in history]
    extraction_acc = [h["drug_extraction_accuracy"] for h in history]
    severity_acc = [h["severity_accuracy"] for h in history]  # may contain None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=accuracy, mode="lines+markers", name="Overall Accuracy",
        line=dict(color="royalblue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=extraction_acc, mode="lines+markers", name="Drug Extraction Accuracy",
        line=dict(color="seagreen", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=severity_acc, mode="lines+markers", name="Severity Accuracy",
        line=dict(color="darkorange", width=2, dash="dot"), connectgaps=False,
    ))
    fig.update_layout(
        xaxis_title="Round", yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    latest = history[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds Run", str(len(history)))
    c2.metric("Latest Overall Accuracy", f"{latest['accuracy']*100:.1f}%")
    c3.metric("Latest Extraction Accuracy", f"{latest['drug_extraction_accuracy']*100:.1f}%")
    sev = latest["severity_accuracy"]
    c4.metric("Latest Severity Accuracy", f"{sev*100:.1f}%" if sev is not None else "N/A")

    st.subheader("Most Recent Round — Failures")
    if latest["failures"]:
        st.error(f"{len(latest['failures'])} failure(s) in the most recent round:")
        for f in latest["failures"]:
            label = f"{f['drug_a'].title()} + {f['drug_b'].title()} — {f['issue']}"
            with st.expander(label):
                st.json(f)
    else:
        st.success("No failures in the most recent round.")
