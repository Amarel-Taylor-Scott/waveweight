"""The feedback signal: does a weight pattern improve out-of-period transfer?

We fit a weighted model on the early ("low") rows and score Pearson on the late
("high") rows, worst-of-K slices — the same honest train→test transfer that the
covariate-shift literature (and shiftblend) relies on. ``gap`` = in-sample −
transfer is the overfit tell. The objective the search maximizes is ``transfer``.
"""

from __future__ import annotations

import numpy as np

from .weighted import pearson, weighted_ridge


def time_split(n, low_frac=0.5):
    cut = max(1, min(n - 1, int(round(n * low_frac))))
    return np.arange(cut), np.arange(cut, n)


def _worst(pred, y, k):
    sl = [s for s in np.array_split(np.arange(len(y)), k) if len(s) > 1]
    sc = [pearson(pred[s], y[s]) for s in sl]
    return min(sc) if sc else 0.0


def evaluate(X, y, sample_weights, *, lam=1.0, low_frac=0.5, k=2) -> dict:
    """Transfer (worst-of-K low→high), in-sample, and their gap for given weights."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    sw = np.asarray(sample_weights, float).ravel()
    low, high = time_split(len(y), low_frac)
    pred_high = weighted_ridge(X[low], y[low], X[high], sw[low], lam=lam)
    pred_low = weighted_ridge(X[low], y[low], X[low], sw[low], lam=lam)
    tr = _worst(pred_high, y[high], k)
    ins = pearson(pred_low, y[low])
    return {"transfer": tr, "insample": ins, "gap": ins - tr}
