"""Weighted ridge / OLS via the Gram trick (per-sample weights).

Solves ``(Xᵀ W X + λI) β = Xᵀ W y`` with a weighted intercept. Sample weights
are normalized to mean 1 so the ridge strength ``λ`` keeps a consistent meaning
across weight patterns.
"""

from __future__ import annotations

import numpy as np


def pearson(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def weighted_ridge(X_tr, y_tr, X_ev, sample_weights, lam=1.0) -> np.ndarray:
    X_tr = np.asarray(X_tr, float)
    y_tr = np.asarray(y_tr, float).ravel()
    X_ev = np.asarray(X_ev, float)
    if X_tr.ndim == 1:
        X_tr = X_tr[:, None]
        X_ev = X_ev[:, None]
    w = np.asarray(sample_weights, float).ravel()
    w = w / w.mean() if w.mean() > 0 else np.ones_like(w)
    sw = w.sum()
    mu = (X_tr * w[:, None]).sum(axis=0) / sw
    ybar = float((y_tr * w).sum() / sw)
    Xc = X_tr - mu
    A = Xc.T @ (Xc * w[:, None])
    b = Xc.T @ (w * (y_tr - ybar))
    p = A.shape[0]
    beta = np.linalg.solve(A + lam * np.eye(p), b)
    return (X_ev - mu) @ beta + ybar


def make_weighted_ridge(lam=1.0):
    def fit_predict(X_tr, y_tr, X_ev, sample_weights):
        return weighted_ridge(X_tr, y_tr, X_ev, sample_weights, lam=lam)
    return fit_predict
