"""Iteratively search weight patterns for one that lifts transfer over uniform.

Random search over pattern families and their parameters (wavelength/period,
magnitude, phase). Every candidate is scored by honest low→high transfer; the
result reports the uniform-weight baseline and the best pattern's **lift** over
it — so you can see whether reweighting buys any real, transferable signal.
"""

from __future__ import annotations

import numpy as np

from . import patterns
from .score import evaluate


def random_search(X, y, *, families=None, n_iter=200, lam=1.0,
                  low_frac=0.5, k=2, seed=0) -> dict:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    families = list(families or patterns.PATTERNS)
    base = evaluate(X, y, patterns.uniform(n), lam=lam, low_frac=low_frac, k=k)["transfer"]

    results = []
    for _ in range(n_iter):
        fam = str(rng.choice(families))
        params = patterns.sample_params(fam, n, rng)
        w = patterns.make(fam, n, params)
        sc = evaluate(X, y, w, lam=lam, low_frac=low_frac, k=k)["transfer"]
        results.append({"score": sc, "family": fam, "params": params})

    results.sort(key=lambda r: -r["score"])
    best = results[0]
    return {
        "baseline": base,
        "best_score": best["score"],
        "lift": best["score"] - base,
        "best_family": best["family"],
        "best_params": best["params"],
        "best_weights": patterns.make(best["family"], n, best["params"]),
        "top": results[:10],
    }
