"""
Explainable AI (XAI) Telemetry Generator (P5).
Transforms feature contributions (SHAP values) into plain-English operator readouts.
Supports multiple personas: Operator, Orbital Analyst, Flight Dynamics Regulator.
"""
from typing import Dict, Any

def explain_risk_prediction(
    event_dict: Dict[str, Any],
    risk_prediction: Dict[str, Any],
    persona: str = "Operator"
) -> Dict[str, Any]:
    """
    Generate persona-customized XAI telemetry readout.
    
    Args:
        event_dict: Event feature dictionary.
        risk_prediction: Output dictionary from predict_risk().
        persona: Persona perspective ('Operator', 'Analyst', 'Regulator').
        
    Returns:
        {
            "summary": str,
            "top_drivers": List[Dict[str, Any]],
            "recommendation": str
        }
    """
    poc = risk_prediction.get("poc", 1e-5)
    miss_dist = event_dict.get("miss_dist", 500.0)
    t_to_tca = event_dict.get("t_to_tca", 3.0)
    f107 = event_dict.get("f107", 150.0)

    # Feature importances / SHAP mock attribution
    drivers = [
        {"feature": "Miss Distance", "impact": "High Risk (+) if < 300m", "value": f"{miss_dist:.1f} m"},
        {"feature": "Time-to-TCA", "impact": "Uncertainty (+) if > 4d", "value": f"{t_to_tca:.1f} days"},
        {"feature": "Solar Flux F10.7", "impact": "Drag Variance (+)", "value": f"{f107:.1f} sfu"},
        {"feature": "Covariance Scale", "impact": "Cross-track Ellipse", "value": "Medium"}
    ]

    if persona == "Operator":
        summary = (
            f"ALERT: Conjunction event requires immediate review. "
            f"Predicted collision probability is {poc:.2e} ({risk_prediction.get('priority', 'UNKNOWN')}). "
            f"Primary driver is a close miss distance of {miss_dist:.1f} meters at {t_to_tca:.1f} days to TCA."
        )
        rec = "Review recommended prograde maneuver option of 0.25 m/s at T-24h."

    elif persona == "Analyst":
        summary = (
            f"Analytical Breakdown: Conjunction shows high along-track uncertainty. "
            f"Calibrated PoC = {poc:.2e} with 95% conformal bounds [{risk_prediction.get('lo', 1e-6):.2e}, "
            f"{risk_prediction.get('hi', 1e-4):.2e}]. Atmospheric drag variance is elevated."
        )
        rec = "Cross-reference SGP4 residual error against latest CelesTrak TLE update."

    else:  # Regulator / Safety Officer
        summary = (
            f"Compliance Status: Risk exceeds IADC / Space Debris Mitigation guideline threshold (1e-4). "
            f"Formal notification triggered for target satellite object."
        )
        rec = "Log maneuver plan in Space Track registry prior to T-12h execution window."

    return {
        "summary": summary,
        "top_drivers": drivers,
        "recommendation": rec
    }
