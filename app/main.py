"""
Space Debris Collision Avoidance Operator Dashboard (P5).
End-to-End Streamlit Operations UI integrating P1-P5 pipeline modules.
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent path so imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingest import load_cdm_dataset
from propagation.screener import screen
from risk.model import predict_risk
from maneuver.optimizer import recommend_maneuver
from xai.explainer import explain_risk_prediction

# Page Configuration
st.set_page_config(
    page_title="Space Debris Crisis Operations Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363D;
    }
    .badge-critical {
        background-color: #D9381E;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #E67E22;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-nominal {
        background-color: #27AE60;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ Autonomous Space Debris Collision Avoidance Center")
st.caption("End-to-End Pipeline: Data Ingestion → SGP4 Screening → ML Risk Model → Pareto Maneuver Optimization → XAI Telemetry")

# Sidebar Controls
st.sidebar.header("⚙️ Operator Controls")
persona = st.sidebar.selectbox("Persona Perspective", ["Operator", "Analyst", "Regulator"])
dataset_size = st.sidebar.slider("Events Loaded", 10, 100, 30)
screen_horizon = st.sidebar.selectbox("Screening Horizon", ["3d", "7d", "14d"])

# Load Pipeline Data
@st.cache_data
def get_data(num_events):
    events = load_cdm_dataset()
    return events[:num_events]

events = get_data(dataset_size)

# Screening & Risk Predictions
screen_candidates = screen([], horizon=screen_horizon)
event_rows = []

for evt in events:
    r_dict = predict_risk(evt)
    latest = evt.latest_cdm
    event_rows.append({
        "Event ID": evt.event_id,
        "Target": evt.target_id,
        "Chaser": evt.chaser_id,
        "Time to TCA (d)": round(latest.t_to_tca, 2),
        "Miss Dist (m)": round(latest.miss_dist, 1),
        "Calculated PoC": r_dict["poc"],
        "Log10 PoC": round(r_dict["log10_poc"], 2),
        "Priority": r_dict["priority"],
        "Event_Obj": evt,
        "Risk_Dict": r_dict
    })

df_events = pd.DataFrame(event_rows)
df_events = df_events.sort_values(by="Calculated PoC", ascending=False)

# Header Summary Metrics
col1, col2, col3, col4 = st.columns(4)
critical_cnt = len(df_events[df_events["Priority"] == "CRITICAL"])
warning_cnt = len(df_events[df_events["Priority"] == "WARNING"])
nominal_cnt = len(df_events[df_events["Priority"] == "NOMINAL"])

with col1:
    st.metric("Total Monitored Events", len(df_events))
with col2:
    st.metric("Critical Alerts (PoC ≥ 1e-4)", critical_cnt, delta_color="inverse")
with col3:
    st.metric("Warning Alerts (PoC ≥ 1e-6)", warning_cnt)
with col4:
    st.metric("Nominal Events", nominal_cnt)

st.divider()

# Main Dashboard Layout
left_panel, right_panel = st.columns([1.2, 1.0])

with left_panel:
    st.subheader("🚨 Ranked Conjunction Alerts")
    selected_event_id = st.selectbox(
        "Select Event for Deep-Dive Analysis:",
        df_events["Event ID"].tolist()
    )
    
    selected_row = df_events[df_events["Event ID"] == selected_event_id].iloc[0]
    selected_evt = selected_row["Event_Obj"]
    selected_risk = selected_row["Risk_Dict"]
    
    st.dataframe(
        df_events[["Event ID", "Target", "Chaser", "Time to TCA (d)", "Miss Dist (m)", "Log10 PoC", "Priority"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Risk Conformal Bounds Visualizer
    st.subheader("📊 Calibrated PoC & Conformal Bounds")
    fig_bounds = go.Figure()
    
    fig_bounds.add_trace(go.Bar(
        x=[selected_row["Event ID"]],
        y=[selected_risk["poc"]],
        name="Calibrated PoC",
        marker_color="#3498DB"
    ))
    
    fig_bounds.add_trace(go.Scatter(
        x=[selected_row["Event ID"], selected_row["Event ID"]],
        y=[selected_risk["lo"], selected_risk["hi"]],
        mode="lines+markers",
        name="95% Conformal Bound",
        line=dict(color="#E74C3C", width=3)
    ))
    
    fig_bounds.update_layout(
        yaxis_type="log",
        yaxis_title="Probability of Collision (Log Scale)",
        height=280,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_bounds, use_container_width=True)

with right_panel:
    st.subheader("🚀 P4 Maneuver Optimization (Pareto Front)")
    maneuvers = recommend_maneuver(selected_evt)
    df_man = pd.DataFrame(maneuvers)
    
    fig_pareto = px.scatter(
        df_man,
        x="dv",
        y="residual_poc",
        color="dir",
        size="t_burn_hours_before_tca",
        hover_data=["pareto_rank", "miss_distance_after_m"],
        labels={"dv": "Delta-V (m/s)", "residual_poc": "Residual PoC", "dir": "Burn Direction"},
        title="Fuel Cost (Δv) vs. Residual Risk"
    )
    fig_pareto.update_layout(yaxis_type="log", height=300)
    st.plotly_chart(fig_pareto, use_container_width=True)
    
    st.markdown("**Top Recommended Burn Options:**")
    st.dataframe(
        df_man[["pareto_rank", "dv", "dir", "t_burn_hours_before_tca", "residual_poc"]],
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    st.subheader(f"🧠 P5 XAI Telemetry Explanation ({persona} View)")
    xai_res = explain_risk_prediction(
        selected_evt.to_dict(),
        selected_risk,
        persona=persona
    )
    
    st.info(xai_res["summary"])
    st.success(f"**Action Plan:** {xai_res['recommendation']}")
    
    st.markdown("**Key Feature Contributions:**")
    st.table(pd.DataFrame(xai_res["top_drivers"]))

st.sidebar.divider()
st.sidebar.caption("Hackfest 1C | Space Debris Crisis Project")
