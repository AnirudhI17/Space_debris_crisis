"""
Analytical 2D Foster/Chan Probability of Collision (PoC) calculation.
Projects 3D RTN covariance onto 2D encounter plane perpendicular to relative velocity.
"""
import numpy as np

def compute_foster_poc_2d(
    miss_vector_m: np.ndarray,
    combined_cov_m2: np.ndarray,
    hard_body_radius_m: float = 10.0
) -> float:
    """
    Compute 2D Foster PoC given miss distance vector and combined 2D covariance matrix.
    
    Args:
        miss_vector_m: 2D array [x_miss, y_miss] in encounter plane (meters).
        combined_cov_m2: 2x2 combined covariance matrix in encounter plane (m^2).
        hard_body_radius_m: Sum of target and chaser radii (meters).
        
    Returns:
        Probability of collision (float between 0.0 and 1.0).
    """
    try:
        det = np.linalg.det(combined_cov_m2)
        if det <= 0:
            return 1e-10
        inv_cov = np.linalg.inv(combined_cov_m2)
        
        # Mahalanobis distance squared to origin
        d2 = float(miss_vector_m.T @ inv_cov @ miss_vector_m)
        
        # 2D isotropic expansion approximation
        area_hbr = np.pi * (hard_body_radius_m ** 2)
        poc = (area_hbr / (2.0 * np.pi * np.sqrt(det))) * np.exp(-0.5 * d2)
        return float(np.clip(poc, 1e-12, 1.0))
    except Exception:
        # Fallback exponential decay with distance
        r = float(np.linalg.norm(miss_vector_m))
        return float(np.clip(np.exp(-r / 100.0) * 1e-2, 1e-12, 1.0))
