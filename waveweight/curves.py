"""Weight curves / formulas — map any *driver* signal to a sample-weight vector.

A driver is any per-row signal (an index pattern, or a window feature of the
data). A curve turns it into weights. We rank-normalize the driver first (robust
to scale/outliers), then apply the curve, then normalize weights to mean 1 — so
curves are comparable and ``magnitude`` shapes emphasis, not regularization.

This is the "weight formulas / weight curves / masks" lever, swept exhaustively
by the search alongside drivers and window sizes.
"""

from __future__ import annotations

import numpy as np


def rank01(z) -> np.ndarray:
    """Rank-normalize to [0, 1] (ties broken by position; robust to scale)."""
    z = np.asarray(z, float).ravel()
    order = z.argsort(kind="stable")
    r = np.empty(len(z), float)
    r[order] = np.arange(len(z))
    return r / max(1, len(z) - 1)


def linear(driver, magnitude=1.0):
    u = rank01(driver)
    return 1.0 + magnitude * (2 * u - 1)


def power(driver, gamma=2.0, magnitude=1.0):
    u = rank01(driver)
    return 1.0 + magnitude * (u ** gamma - 0.5)


def sigmoid(driver, steep=8.0, mid=0.5, magnitude=1.0):
    u = rank01(driver)
    s = 1.0 / (1.0 + np.exp(-steep * (u - mid)))
    return 1.0 + magnitude * (2 * s - 1)


def mask_top(driver, q=0.2, hi=3.0, lo=0.3):
    """Upweight the top-q of the driver, downweight the rest (a hard mask)."""
    u = rank01(driver)
    return np.where(u >= 1 - q, hi, lo)


def mask_mid(driver, q=0.34, hi=3.0, lo=0.5):
    """Upweight the MIDDLE band of the driver — e.g. the central sample of a
    group, when the driver measures how 'middle' a row is."""
    u = rank01(driver)
    return np.where(np.abs(u - 0.5) <= q / 2, hi, lo)


def rank_weight(driver, magnitude=1.0):
    return 0.2 + magnitude * rank01(driver)


def invert(driver, magnitude=1.0):
    return linear(-np.asarray(driver, float), magnitude)


CURVES = {
    "linear": linear, "power": power, "sigmoid": sigmoid,
    "mask_top": mask_top, "mask_mid": mask_mid, "rank": rank_weight,
    "invert": invert,
}


def normalize(w) -> np.ndarray:
    """Clip to positive and scale to mean 1 (so λ stays comparable)."""
    w = np.clip(np.asarray(w, float).ravel(), 1e-6, None)
    m = w.mean()
    return w / m if m > 0 else np.ones_like(w)
