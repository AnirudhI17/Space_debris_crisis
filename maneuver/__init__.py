"""
Maneuver optimization module (P4 - Owned by Anirudh).
Computes 2D Foster/Chan PoC, along-track delta-v grid search, Pareto optimization,
and secondary conjunction rejection using P2 screener.
"""
from .optimizer import recommend_maneuver
from .foster_poc import compute_foster_poc_2d

__all__ = ["recommend_maneuver", "compute_foster_poc_2d"]
