"""Missing-piece (gap) detection for solved strip orders."""

from __future__ import annotations

import numpy as np


def detect_gaps(rel_cost: np.ndarray, ratio: np.ndarray, informative: np.ndarray,
                min_rel: float = 1.5, min_ratio: float = 1.2) -> np.ndarray:
    """Flag joins that are probably a missing strip rather than a real seam.

    When a strip is missing, its two neighbours get forced together.  Such a join shows
    two symptoms at once:

    * its cost is high compared with the typical join of this page (``rel_cost`` =
      cost / median join cost), and
    * neither piece prefers the other: the ratio-normalised cost (``ratio`` = cost /
      second-best alternative, see ``normalize_costs``) is above 1.

    Requiring both keeps false alarms low; correct-but-noisy seams usually only show one
    symptom.  Joins between blank strips (``informative`` False) are never flagged —
    there is no evidence either way.  Smooth photo regions are the known weak spot: a
    gradient extrapolates across a missing strip almost seamlessly.
    """
    rel_cost, ratio, informative = map(np.asarray, (rel_cost, ratio, informative))
    return (rel_cost > min_rel) & (ratio > min_ratio) & informative
