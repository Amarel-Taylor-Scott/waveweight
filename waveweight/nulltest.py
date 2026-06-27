"""Null-controlled, held-out search — the discipline that makes a result mean
something (or honestly report nothing).

Three time splits: **low** (train, gets weighted), **sel** (selection), **val**
(held out, never seen during selection). We pick the best candidate weighting by
its transfer on ``sel``, then report its transfer on ``val`` and compare that to
a **magnitude-matched null**: the winner's own weight values, permuted across the
training rows (same distribution, structure destroyed), scored on ``val``, many
times. A weighting "wins" only if it beats the null's 95th percentile *and* beats
uniform on the held-out split. If nothing does, that is the correct data verdict
— not a failure, and not something to override with an assumption.
"""

from __future__ import annotations

import numpy as np

from . import patterns
from .weighted import pearson, weighted_ridge


def three_split(n, fracs=(0.5, 0.25, 0.25)):
    a = max(1, int(n * fracs[0]))
    b = min(n - 1, a + max(1, int(n * fracs[1])))
    return np.arange(a), np.arange(a, b), np.arange(b, n)


def _score(X, y, low, ev, weights, lam):
    pred = weighted_ridge(X[low], y[low], X[ev], weights[low], lam=lam)
    return pearson(pred, y[ev])


def null_controlled_search(X, y, candidates, *, lam=1.0, fracs=(0.5, 0.25, 0.25),
                           n_null=200, seed=0) -> dict:
    """Select on ``sel``, validate on held-out ``val`` vs a permutation null."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    low, sel, val = three_split(n, fracs)
    uni = patterns.uniform(n)
    base_val = _score(X, y, low, val, uni, lam)

    scored = [(_score(X, y, low, sel, w, lam), lbl, w, meta)
              for (lbl, w, meta) in candidates]
    scored.sort(key=lambda t: -t[0])
    sel_score, lbl, w, meta = scored[0]
    val_obs = _score(X, y, low, val, w, lam)

    wl = w[low]
    nulls = np.empty(n_null)
    for i in range(n_null):
        wn = uni.copy()
        wn[low] = wl[rng.permutation(len(wl))]
        nulls[i] = _score(X, y, low, val, wn, lam)
    p95 = float(np.quantile(nulls, 0.95))
    p = float((nulls >= val_obs).mean())

    return {
        "best_label": lbl,
        "best_meta": meta,
        "selection_transfer": sel_score,
        "val_transfer": val_obs,
        "uniform_val": base_val,
        "lift_over_uniform": val_obs - base_val,
        "null_mean": float(nulls.mean()),
        "null_p95": p95,
        "null_p": p,
        "beats_null": bool(val_obs > p95 and val_obs > base_val),
        "n_candidates": len(candidates),
        "top": [(round(s, 4), l) for s, l, _w, _m in scored[:12]],
    }
