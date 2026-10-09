"""
XAI and plain-English explanation module (P5).
Generates telemetry readouts and persona-based explainability summaries.
"""
from .explainer import explain_risk_prediction

__all__ = ["explain_risk_prediction"]
