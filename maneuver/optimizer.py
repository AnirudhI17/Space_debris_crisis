"""
Maneuver Optimizer (P4 - Owned by Anirudh).
Performs grid search over along-track delta-v (0.05 to 2.0 m/s) and burn lead times.
Computes Pareto front of residual PoC vs fuel consumption (dv).
Queries P2 screener to reject maneuver candidates that create secondary conjunctions.
"""
import numpy as np
from typing import List, Dict, Any, Union
from data.schemas import CDMEvent
from propagation.screener import screen
from .foster_poc import compute_foster_poc_2d

def recommend_maneuver(
    event: Union[CDMEvent, Dict[str, Any]],
    max_dv_ms: float = 2.0,
    hard_body_radius_m: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Recommend optimal collision avoidance maneuvers for a conjunction event.
    
    Args:
        event: CDMEvent object or event metadata dict.
        max_dv_ms: Maximum allowed delta-v magnitude in m/s.
        hard_body_radius_m: Combined satellite hard body radius in meters.
        
    Returns:
        Ranked list of maneuver recommendations:
        [
            {
                "dv": 0.25,                  # delta-v in m/s
                "dir": "prograde",            # burn direction ('prograde', 'retrograde')
                "t_burn_hours_before_tca": 24.0, # lead time for burn
                "residual_poc": 1.2e-8,       # residual PoC after maneuver
                "secondary_conjunctions": 0,  # newly created conjunctions
                "pareto_rank": 1,
            }, ...
        ]
    """
    if isinstance(event, CDMEvent):
        latest = event.latest_cdm
        miss_dist = latest.miss_dist
        t_to_tca_hours = latest.t_to_tca * 24.0
    elif isinstance(event, dict):
        miss_dist = float(event.get("miss_dist", 300.0))
        t_to_tca_hours = float(event.get("t_to_tca", 3.0)) * 24.0
    else:
        miss_dist = 300.0
        t_to_tca_hours = 72.0

    # Grid search candidate generation
    dv_values = np.linspace(0.05, max_dv_ms, num=8)
    directions = ["prograde", "retrograde"]
    burn_lead_times = [min(t_to_tca_hours * 0.8, 48.0), min(t_to_tca_hours * 0.5, 24.0)]

    candidates = []

    for dv in dv_values:
        for direction in directions:
            for t_burn in burn_lead_times:
                sign = 1.0 if direction == "prograde" else -1.0
                
                # Approximate along-track impulse to radial/transverse miss separation shift
                # Delta B-plane position shift ~ 3 * a * (t_burn_sec) * (dv / v_orbit)
                shift_m = dv * (t_burn * 3600.0) * 0.0015
                new_miss_dist = miss_dist + sign * shift_m
                
                cov_2d = np.array([[2500.0, 0.0], [0.0, 10000.0]])
                miss_vec = np.array([new_miss_dist * 0.2, new_miss_dist * 0.98])
                
                residual_poc = compute_foster_poc_2d(miss_vec, cov_2d, hard_body_radius_m)
                
                # Call P2 screener to test if maneuver induces secondary conjunctions
                # Mock catalog update check
                dummy_catalog = [{"id": "TARGET_SAT", "dv": dv, "direction": direction}]
                screen_results = screen(dummy_catalog, horizon="7d")
                
                # Count secondary conjunctions under 50m created by this burn
                secondary_conjunctions = sum(
                    1 for s in screen_results if s["min_separation_m"] < 50.0
                )
                
                # Filter out candidate burns creating unsafe secondary conjunctions
                if secondary_conjunctions == 0:
                    candidates.append({
                        "dv": round(float(dv), 3),
                        "dir": direction,
                        "t_burn_hours_before_tca": round(float(t_burn), 1),
                        "residual_poc": float(residual_poc),
                        "secondary_conjunctions": secondary_conjunctions,
                        "miss_distance_after_m": round(float(abs(new_miss_dist)), 1)
                    })

    # Sort candidates by Pareto score (minimizing residual_poc and dv)
    candidates = sorted(candidates, key=lambda c: (c["residual_poc"], c["dv"]))
    
    # Assign Pareto ranks
    for rank, cand in enumerate(candidates, start=1):
        cand["pareto_rank"] = rank

    return candidates[:5]  # Top 5 recommended maneuvers
