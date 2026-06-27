"""Bootstrap to find sample weights that *robustly* help — not just on one split.

A single search can lock onto a pattern that's lucky on one training draw. So we
fix a pool of candidate patterns, then bag: resample the training rows ``n_boot``
times (each row keeps its original index, so its candidate weight is the pattern
value *at that index*), refit weighted, and score each candidate on the same
held-out late rows. The candidate with the best **average** transfer across
resamples is the robust pick — its per-record weight vector is the answer.
``stability`` is the per-record spread across the top candidates.

Honesty caveat: selection still looks at the late rows, so keep the candidate
patterns *low-parameter* (a wavelength + magnitude resists overfitting where free
per-row weights would not), and confirm the winner on a final untouched draw.
"""

from __future__ import annotations

import numpy as np

from . import patterns
from .score import time_split
from .weighted import pearson, weighted_ridge


def _block_resample(n, block, rng):
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def _worst(pred, y, k):
    sl = [s for s in np.array_split(np.arange(len(y)), k) if len(s) > 1]
    sc = [pearson(pred[s], y[s]) for s in sl]
    return min(sc) if sc else 0.0


def bootstrap_refine(X, y, *, families=None, n_candidates=150, n_boot=12, lam=1.0,
                     low_frac=0.5, k=2, seed=0, top_m=8, block=None) -> dict:
    """Bag a candidate pool over training resamples; return the robust weights.

    Keys: ``weights`` (robust per-record), ``stability`` (per-record std across
    the top candidates), ``best_family``/``best_params``, ``baseline_transfer``,
    ``refined_transfer``, ``lift``.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    fams = list(families or patterns.PATTERNS)
    low, high = time_split(n, low_frac)
    n_low = len(low)
    Xh, yh = X[high], y[high]

    # fixed candidate pool of weight vectors over the original index (uniform first)
    cands = [("uniform", {}, patterns.uniform(n))]
    for _ in range(n_candidates):
        f = str(rng.choice(fams))
        p = patterns.sample_params(f, n, rng)
        cands.append((f, p, patterns.make(f, n, p)))

    scores = np.zeros((len(cands), n_boot))
    for b in range(n_boot):
        if block:
            samp = low[_block_resample(n_low, block, rng)]
        else:
            samp = low[rng.integers(0, n_low, size=n_low)]
        Xb, yb = X[samp], y[samp]
        for ci, (_f, _p, wfull) in enumerate(cands):
            pred = weighted_ridge(Xb, yb, Xh, wfull[samp], lam=lam)
            scores[ci, b] = _worst(pred, yh, k)

    order = np.argsort(-scores.mean(axis=1))
    best = int(order[0])
    robust = cands[best][2]
    topW = np.vstack([cands[i][2] for i in order[:max(1, top_m)]])

    # honest full-split numbers (fit on all low, predict high) for both
    base = _worst(weighted_ridge(X[low], y[low], Xh, patterns.uniform(n)[low], lam=lam), yh, k)
    refined = _worst(weighted_ridge(X[low], y[low], Xh, robust[low], lam=lam), yh, k)
    return {
        "weights": robust,
        "stability": topW.std(axis=0),
        "best_family": cands[best][0],
        "best_params": cands[best][1],
        "families": [cands[i][0] for i in order[:top_m]],
        "baseline_transfer": base,
        "refined_transfer": refined,
        "lift": refined - base,
    }
