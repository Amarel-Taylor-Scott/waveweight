"""Synthetic data with a *planted periodic importance* — for demo and tests.

Rows are time-ordered. Every ``period``-th row follows the **true** coefficient
regime; the rows between follow a **different (misleading) regime**. The eval
(late) rows are all true-regime. A uniform fit is dominated by the misleading
majority and estimates the wrong coefficients; a weight pattern that upweights
the true-regime phase (a comb / sine of the right wavelength) recovers the true
coefficients and transfers far better. Crucially the gap is a *bias*, so more
data doesn't close it — only the right sample weighting does. The search should
rediscover ``period``.
"""

from __future__ import annotations

import numpy as np


def make_periodic_importance(n=2000, p=10, period=5, noise=0.1, seed=0,
                             clean_test=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    beta_wrong = rng.standard_normal(p)            # the misleading regime
    t = np.arange(n)
    true_regime = (t % period) == 0
    if clean_test:
        true_regime[n // 2:] = True                # eval rows are all true-regime
    y = np.where(true_regime, X @ beta, X @ beta_wrong) + noise * rng.standard_normal(n)
    return X, y, {"period": period, "clean": true_regime, "beta": beta}
