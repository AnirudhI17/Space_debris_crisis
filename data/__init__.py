"""
Data and evaluation module (P1).
Ingestion, feature engineering, and evaluation metrics.
"""
from .schemas import CDMEvent, CDMRow, HIGH_RISK_LOG_POC_THRESHOLD
from .ingest import load_cdm_dataset, fetch_celestrak_tles, fetch_space_weather
from .eval import evaluate_risk_model

__all__ = [
    "CDMEvent",
    "CDMRow",
    "HIGH_RISK_LOG_POC_THRESHOLD",
    "load_cdm_dataset",
    "fetch_celestrak_tles",
    "fetch_space_weather",
    "evaluate_risk_model",
]
