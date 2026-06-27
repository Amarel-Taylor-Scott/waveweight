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


def make_local_importance(n=2000, p=10, k=5, noise=0.1, seed=0, clean_test=True):
    """Importance set by a LOCAL WINDOW RELATIONSHIP, not by index.

    Rows that are a local maximum of an observable feature (column 0) within a
    window of ``k`` follow the true regime; the rest follow a misleading one. The
    informative rows are scattered (aperiodic), so only a *data-driven* window
    driver (``x0.is_extremum`` / ``center_z``) discovers them — an index pattern
    cannot. The window search should beat the null here; an index sweep should not.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    beta_wrong = rng.standard_normal(p)
    s = X[:, 0]
    h = k // 2
    is_max = np.zeros(n, bool)
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        is_max[i] = s[i] >= s[a:b].max() - 1e-12
    true_regime = is_max.copy()
    if clean_test:
        true_regime[n // 2:] = True
    y = np.where(true_regime, X @ beta, X @ beta_wrong) + noise * rng.standard_normal(n)
    return X, y, {"k": k, "driver_col": 0, "true": true_regime, "beta": beta}


def make_no_signal(n=2000, p=10, noise=1.0, seed=0):
    """A plain regression with NO special sample structure — no weighting should
    beat the null. The honest answer here is 'nothing', and the tool must say so."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise * rng.standard_normal(n)
    return X, y, {}


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
