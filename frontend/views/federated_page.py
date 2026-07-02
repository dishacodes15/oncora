import streamlit as st
import plotly.graph_objects as go

def show():
    st.title("Federated Learning Monitor")
    st.write("Live view of simulated hospital training nodes.")

    # Hospital nodes status
    st.subheader("Hospital Nodes")
    col1, col2, col3 = st.columns(3)
    col1.metric("Hospital A", "Active", "Local training complete")
    col2.metric("Hospital B", "Active", "Weights submitted")
    col3.metric("Hospital C", "Active", "Awaiting global model")

    # Accuracy over rounds — use mock data until Aarya's FL module is ready
    st.subheader("Global Model Accuracy per Round")
    rounds = list(range(1, 11))
    accuracy = [0.61, 0.70, 0.74, 0.78, 0.81, 0.84, 0.86, 0.88, 0.89, 0.91]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rounds, y=accuracy, mode='lines+markers',
                             name='Federated Accuracy',
                             line=dict(color='royalblue', width=2)))
    fig.update_layout(xaxis_title="Round", yaxis_title="Accuracy",
                      yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rounds", "10")
    col2.metric("Final Global Accuracy", "91.0%")
    col3.metric("Hospitals Participating", "3")