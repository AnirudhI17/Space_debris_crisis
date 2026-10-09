"""
Integration and stub verification unit tests.
Validates contracts for P1, P2, P3, P4, P5 modules.
"""
import sys
import os
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingest import load_cdm_dataset, generate_mock_cdm_events
from data.eval import evaluate_risk_model
from propagation.screener import screen
from risk.model import predict_risk
from maneuver.optimizer import recommend_maneuver
from xai.explainer import explain_risk_prediction

def test_p1_data_and_eval():
    events = generate_mock_cdm_events(num_events=5)
    assert len(events) == 5
    evt = events[0]
    assert evt.event_id.startswith("EVT_")
    assert len(evt.cdms) >= 3
    
    # Eval harness
    metrics = evaluate_risk_model(
        y_true=[True, False, True],
        y_prob=[0.9, 0.1, 0.85],
        lead_times_days=[3.0, 4.0, 1.5]
    )
    assert "brier_score" in metrics
    assert "recall" in metrics
    assert "lead_time_metrics" in metrics

def test_p2_propagation_screener():
    candidates = screen([], horizon="7d")
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    first = candidates[0]
    assert "target_id" in first
    assert "chaser_id" in first
    assert "min_separation_m" in first

def test_p3_risk_model():
    events = generate_mock_cdm_events(num_events=1)
    result = predict_risk(events[0])
    assert "poc" in result
    assert "lo" in result
    assert "hi" in result
    assert "priority" in result
    assert result["priority"] in ["CRITICAL", "WARNING", "NOMINAL"]

def test_p4_maneuver_optimizer():
    events = generate_mock_cdm_events(num_events=1)
    maneuvers = recommend_maneuver(events[0])
    assert isinstance(maneuvers, list)
    assert len(maneuvers) > 0
    top_man = maneuvers[0]
    assert "dv" in top_man
    assert "dir" in top_man
    assert "t_burn_hours_before_tca" in top_man
    assert "residual_poc" in top_man
    assert "pareto_rank" in top_man

def test_p5_xai_explainer():
    events = generate_mock_cdm_events(num_events=1)
    evt_dict = events[0].to_dict()
    risk_dict = predict_risk(events[0])
    explanation = explain_risk_prediction(evt_dict, risk_dict, persona="Operator")
    assert "summary" in explanation
    assert "top_drivers" in explanation
    assert "recommendation" in explanation
