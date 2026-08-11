import streamlit as st
import plotly.graph_objects as go
import json
import os
from style import eyebrow

def show():
    st.markdown(eyebrow("MODULE 03 - FEDERATED TRAINING"), unsafe_allow_html=True)
    st.title("Federated Learning Monitor")
    st.write("Live view of simulated hospital training nodes.")

    st.subheader("Hospital Nodes")
    col1, col2, col3 = st.columns(3)
    col1.metric("Hospital A", "Active", "Local training complete")
    col2.metric("Hospital B", "Active", "Weights submitted")
    col3.metric("Hospital C", "Active", "Awaiting global model")

    st.subheader("Global Model Accuracy per Round")
    history_path = os.path.join(os.path.dirname(__file__), "../../docs/fl_round_history.json")

    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        rounds = list(range(1, len(history) + 1))
        accuracy = [entry["accuracy"] for entry in history]
    except FileNotFoundError:
        st.warning("FL history file not found - showing mock data")
        rounds = list(range(1, 6))
        accuracy = [0.78, 0.86, 0.73, 0.90, 0.90]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rounds, y=accuracy, mode='lines+markers',
                             name='Federated Accuracy',
                             line=dict(color='#111111', width=2)))
    fig.update_layout(xaxis_title="Round", yaxis_title="Accuracy",
                      yaxis=dict(range=[0, 1]),
                      plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rounds", str(len(rounds)))
    col2.metric("Final Global Accuracy", f"{accuracy[-1]*100:.1f}%")
    col3.metric("Hospitals Participating", "3")
