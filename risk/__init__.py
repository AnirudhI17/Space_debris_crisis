"""
Risk model and calibration module (P3).
Stacked XGBoost + Sequence models, Isotonic Calibration, Conformal bands.
"""
from .model import predict_risk

__all__ = ["predict_risk"]
