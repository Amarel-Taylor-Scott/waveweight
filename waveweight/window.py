"""Data-driven local-window importance — look at groups of k samples together.

Instead of imposing a weight pattern by row index, *derive* a per-row driver
from each row's local neighborhood (a window of k samples) and let the data say
which rows matter. Drivers per row: local mean / std (volatility), the center's
deviation from its neighbors (``center_z`` — "is the middle sample special?"),
whether the center is a local extremum, the local slope, and the local range.

Feed any of these to a weight curve (see ``curves``) and the search will test —
against a null — whether weighting by that local relationship actually helps.
This is the "groups of 5, find the relationship that says the middle sample
matters" lever, discovered from the data rather than assumed.
"""

from __future__ import annotations

import numpy as np


def window_features(signal, k=5) -> dict:
    """Per-row features over a centered window of ``k`` samples (edges clipped)."""
    sig = np.asarray(signal, float).ravel()
    n = len(sig)
    h = k // 2
    lm = np.empty(n)
    ls = np.empty(n)
    cz = np.empty(n)
    ext = np.zeros(n)
    slope = np.empty(n)
    rng = np.empty(n)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        w = sig[a:b]
        m = float(w.mean())
        sd = float(w.std())
        lm[i] = m
        ls[i] = sd
        cz[i] = abs(sig[i] - m) / sd if sd > 0 else 0.0
        ext[i] = 1.0 if (sig[i] >= w.max() - 1e-12 or sig[i] <= w.min() + 1e-12) else 0.0
        rng[i] = float(w.max() - w.min())
        slope[i] = (w[-1] - w[0]) / max(1, len(w) - 1)
    return {"local_mean": lm, "local_std": ls, "center_z": cz,
            "is_extremum": ext, "slope": slope, "local_range": rng}


FEATURES = ("local_mean", "local_std", "center_z", "is_extremum", "slope", "local_range")
