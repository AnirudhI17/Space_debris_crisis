"""
Data ingestion module (P1).
Fetches ESA Kelvins CDMs, CelesTrak TLEs, and Space Weather (F10.7/Kp).
Generates sample synthetic dataset if local parquet is missing.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .schemas import CDMEvent, CDMRow, HIGH_RISK_LOG_POC_THRESHOLD

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PARQUET = os.path.join(DATA_DIR, "events.parquet")


def generate_mock_cdm_events(num_events: int = 50) -> List[CDMEvent]:
    """Generate realistic mock CDM events for initial testing across P1-P5."""
    np.random.seed(42)
    events = []
    
    for i in range(num_events):
        event_id = f"EVT_{i+1000:04d}"
        target_id = f"SAT_{np.random.randint(10000, 99999)}"
        chaser_id = f"DEB_{np.random.randint(10000, 99999)}"
        
        # 3 to 7 sequence steps per event
        num_cdms = np.random.randint(3, 8)
        t_to_tca_sequence = np.sort(np.random.uniform(0.5, 7.0, num_cdms))[::-1]
        
        # Base true miss distance
        base_miss = np.random.exponential(scale=1500) + 10.0
        is_high_risk = base_miss < 300.0 or np.random.rand() < 0.15
        
        cdm_rows = []
        for cdm_idx, t_tca in enumerate(t_to_tca_sequence):
            # Separation with noise reducing as t_to_tca decreases
            noise_factor = t_tca / 7.0
            radial = np.random.normal(loc=0.0, scale=30.0 * noise_factor)
            transverse = base_miss + np.random.normal(loc=0.0, scale=100.0 * noise_factor)
            normal = np.random.normal(loc=0.0, scale=50.0 * noise_factor)
            miss_dist = float(np.sqrt(radial**2 + transverse**2 + normal**2))
            
            # Simple approximate PoC for mock
            log_poc = -float(np.clip(miss_dist / 150.0 + noise_factor, 1.0, 10.0))
            poc = 10 ** log_poc
            
            row = CDMRow(
                cdm_idx=cdm_idx,
                t_to_tca=float(t_tca),
                miss_dist=miss_dist,
                rtn_sep=[float(radial), float(transverse), float(normal)],
                rtn_cov=[100.0, 0.0, 0.0, 500.0, 0.0, 200.0],
                f107=float(np.random.normal(150, 25)),
                kp=float(np.random.uniform(0, 6)),
                poc=poc
            )
            cdm_rows.append(row)
            
        final_log_poc = np.log10(cdm_rows[-1].poc)
        label_high_risk = bool(is_high_risk or final_log_poc >= HIGH_RISK_LOG_POC_THRESHOLD)
        
        event = CDMEvent(
            event_id=event_id,
            target_id=target_id,
            chaser_id=chaser_id,
            tca_iso="2026-10-12T12:00:00Z",
            cdms=cdm_rows,
            label_high_risk=label_high_risk
        )
        events.append(event)
        
    return events


def load_cdm_dataset(filepath: str = SAMPLE_PARQUET) -> List[CDMEvent]:
    """
    Load CDM events from Parquet dataset. If file doesn't exist, create synthetic sample.
    Split rule: Split by event_id, not by individual CDM row to avoid data leakage!
    """
    if not os.path.exists(filepath):
        events = generate_mock_cdm_events(50)
        save_events_parquet(events, filepath)
        return events
    
    df = pd.read_parquet(filepath)
    # Parse back to CDMEvent list
    events_dict: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        eid = row["event_id"]
        if eid not in events_dict:
            events_dict[eid] = []
        events_dict[eid].append(row.to_dict())
        
    events = []
    for eid, rows in events_dict.items():
        cdm_rows = []
        for r in rows:
            cdm_rows.append(CDMRow(
                cdm_idx=int(r["cdm_idx"]),
                t_to_tca=float(r["t_to_tca"]),
                miss_dist=float(r["miss_dist"]),
                rtn_sep=[float(r["r_sep"]), float(r["t_sep"]), float(r["n_sep"])],
                rtn_cov=[100.0, 0.0, 0.0, 500.0, 0.0, 200.0],
                f107=float(r.get("f107", 150.0)),
                kp=float(r.get("kp", 3.0)),
                poc=float(r.get("poc", 1e-7))
            ))
        first_r = rows[0]
        events.append(CDMEvent(
            event_id=eid,
            target_id=str(first_r.get("target_id", "SAT_001")),
            chaser_id=str(first_r.get("chaser_id", "DEB_999")),
            tca_iso="2026-10-12T12:00:00Z",
            cdms=cdm_rows,
            label_high_risk=bool(first_r.get("label_high_risk", False))
        ))
    return events


def save_events_parquet(events: List[CDMEvent], filepath: str = SAMPLE_PARQUET) -> None:
    """Flatten CDM events and save to parquet."""
    records = []
    for evt in events:
        for cdm in evt.cdms:
            records.append({
                "event_id": evt.event_id,
                "target_id": evt.target_id,
                "chaser_id": evt.chaser_id,
                "cdm_idx": cdm.cdm_idx,
                "t_to_tca": cdm.t_to_tca,
                "miss_dist": cdm.miss_dist,
                "r_sep": cdm.rtn_sep[0],
                "t_sep": cdm.rtn_sep[1],
                "n_sep": cdm.rtn_sep[2],
                "f107": cdm.f107,
                "kp": cdm.kp,
                "poc": cdm.poc,
                "label_high_risk": evt.label_high_risk
            })
    df = pd.DataFrame(records)
    df.to_parquet(filepath, index=False)


def fetch_celestrak_tles(group: str = "active") -> List[str]:
    """Stub to fetch CelesTrak TLEs (no login needed)."""
    return [
        "1 25544U 98067A   26282.50000000  .00016717  00000+0  30000-3 0  9993",
        "2 25544  51.6400 200.0000 0004000 100.0000 260.0000 15.5000000012345"
    ]


def fetch_space_weather() -> Dict[str, float]:
    """Stub to fetch current F10.7 solar flux and Kp geomagnetic index."""
    return {"f107": 155.4, "kp": 3.2}
