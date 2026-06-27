"""Build the full candidate space of sample-weightings to test.

Two driver kinds, combined with every weight curve:
  * **index patterns** — sine/comb/square/... by row position (the "is it
    periodic?" lever);
  * **window features** — local mean/std/center_z/extremum/slope/range over a
    group of k samples, computed on the target and on feature columns (the
    "data-driven local importance" lever).

Each candidate is ``(label, weight_vector, meta)`` with weights normalized to
mean 1. The search scores every one on a held-out split against a null.
"""

from __future__ import annotations

import numpy as np

from . import curves, patterns, window


def index_pattern_candidates(n, rng, n_each=30):
    out = []
    for fam in patterns.PATTERNS:
        for _ in range(n_each):
            p = patterns.sample_params(fam, n, rng)
            w = curves.normalize(patterns.make(fam, n, p))
            out.append(("idx:%s" % fam, w, {"kind": "index", "family": fam, **p}))
    return out


def window_driver_candidates(X, y, *, k_values=(3, 5, 7, 9), feature_cols=None,
                             use_y=True, curve_names=None):
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    curve_names = list(curve_names or curves.CURVES)
    drivers = []  # (label, signal)
    if use_y:
        for k in k_values:
            for fn, arr in window.window_features(y, k).items():
                drivers.append(("y.%s.k%d" % (fn, k), arr))
    cols = feature_cols if feature_cols is not None else range(min(X.shape[1], 6))
    for c in cols:
        for k in k_values:
            for fn, arr in window.window_features(X[:, c], k).items():
                drivers.append(("x%d.%s.k%d" % (c, fn, k), arr))

    out = []
    for lbl, sig in drivers:
        for cn in curve_names:
            w = curves.normalize(curves.CURVES[cn](sig))
            out.append(("win:%s|%s" % (lbl, cn), w,
                        {"kind": "window", "driver": lbl, "curve": cn}))
    return out


def build_all(X, y, rng, *, k_values=(3, 5, 7, 9), n_index_each=20, feature_cols=None):
    """Index-pattern + window-driven candidates — the full swept space."""
    return (index_pattern_candidates(len(y), rng, n_index_each)
            + window_driver_candidates(X, y, k_values=k_values, feature_cols=feature_cols))
