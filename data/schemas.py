"""
Shared schema and constants for the Space Debris Challenge pipeline.
Agreed lock-in schema for P1 through P5.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

# High-risk label definition agreed by team: final log10(PoC) >= -6 (PoC >= 1e-6)
HIGH_RISK_LOG_POC_THRESHOLD: float = -6.0
HIGH_RISK_POC_THRESHOLD: float = 10 ** HIGH_RISK_LOG_POC_THRESHOLD  # 1e-6

@dataclass
class CDMRow:
    """Represents a single CDM (Conjunction Data Message) observation."""
    cdm_idx: int
    t_to_tca: float           # Time to closest approach in days
    miss_dist: float          # Miss distance in meters
    rtn_sep: List[float]      # [Radial, Transverse, Normal] separation (m)
    rtn_cov: List[float]      # Covariance matrix elements (packed array)
    f107: float               # F10.7 solar flux index
    kp: float                 # Kp space weather index
    poc: float                # Instantaneous calculated/reported PoC


@dataclass
class CDMEvent:
    """
    Represents an end-to-end conjunction event containing a time sequence of CDMs.
    """
    event_id: str
    target_id: str
    chaser_id: str
    tca_iso: str
    cdms: List[CDMRow] = field(default_factory=list)
    label_high_risk: bool = False

    @property
    def latest_cdm(self) -> CDMRow:
        """Return the latest available CDM row for this event."""
        if not self.cdms:
            raise ValueError(f"Event {self.event_id} has no CDMs.")
        return sorted(self.cdms, key=lambda x: x.t_to_tca)[0]  # smallest t_to_tca is latest

    def to_dict(self) -> Dict[str, Any]:
        latest = self.latest_cdm
        return {
            "event_id": self.event_id,
            "target_id": self.target_id,
            "chaser_id": self.chaser_id,
            "tca_iso": self.tca_iso,
            "cdm_idx": latest.cdm_idx,
            "t_to_tca": latest.t_to_tca,
            "miss_dist": latest.miss_dist,
            "rtn_sep": latest.rtn_sep,
            "rtn_cov": latest.rtn_cov,
            "f107": latest.f107,
            "label_high_risk": self.label_high_risk,
        }
