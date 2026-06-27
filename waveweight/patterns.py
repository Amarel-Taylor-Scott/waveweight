"""Parametric sample-weight patterns over time-ordered records.

The hypothesis: a record's *importance* may follow a periodic pattern — every
k-th row matters more, with neighbors rising and falling like a sine wave. Each
function maps a row count to a non-negative weight vector; ``search`` sweeps
their parameters (wavelength/period, magnitude, phase) looking for a pattern that
lifts out-of-period transfer. Weights are normalized to mean 1 at fit time, so
``magnitude`` shapes the *relative* emphasis, not the overall regularization.
"""

from __future__ import annotations

import numpy as np


def _t(n):
    return np.arange(n, dtype=float)


def uniform(n):
    return np.ones(n)


def sine(n, wavelength, magnitude=0.5, phase=0.0, baseline=1.0):
    w = baseline + magnitude * np.sin(2 * np.pi * _t(n) / max(wavelength, 1e-9) + phase)
    return np.clip(w, 0.0, None)


def comb(n, period, magnitude=1.0, width=1, baseline=1.0):
    """Every ``period``-th row (a ``width``-wide spike) gets extra weight."""
    w = np.full(n, baseline, float)
    w[(_t(n) % period) < max(1, int(width))] += magnitude
    return np.clip(w, 0.0, None)


def square(n, wavelength, magnitude=0.5, duty=0.5, baseline=1.0):
    frac = (_t(n) % wavelength) / wavelength
    return np.clip(baseline + magnitude * np.where(frac < duty, 1.0, -1.0), 0.0, None)


def sawtooth(n, wavelength, magnitude=0.5, baseline=1.0):
    frac = (_t(n) % wavelength) / wavelength
    return np.clip(baseline + magnitude * (2 * frac - 1), 0.0, None)


def triangle(n, wavelength, magnitude=0.5, baseline=1.0):
    frac = (_t(n) % wavelength) / wavelength
    return np.clip(baseline + magnitude * (2 * np.abs(2 * frac - 1) - 1), 0.0, None)


def gaussian_bumps(n, period, sigma=1.0, magnitude=1.0, baseline=1.0):
    phase = _t(n) % period
    d = np.minimum(phase, period - phase)
    return baseline + magnitude * np.exp(-(d ** 2) / (2 * max(sigma, 1e-9) ** 2))


def exp_decay(n, halflife, baseline=0.0):
    """Recency weighting — most recent rows weigh most, decaying into the past."""
    age = (n - 1) - _t(n)
    return baseline + 0.5 ** (age / max(halflife, 1e-9))


PATTERNS = {
    "sine": sine, "comb": comb, "square": square, "sawtooth": sawtooth,
    "triangle": triangle, "gaussian_bumps": gaussian_bumps, "exp_decay": exp_decay,
}


def make(family, n, params):
    return PATTERNS[family](n, **params)


def _logint(rng, lo, hi):
    """Log-uniform integer in [lo, hi] — so short periods (often the real ones)
    are sampled as densely as long ones."""
    lo = max(2, int(lo))
    hi = max(lo + 1, int(hi))
    return int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi))))))


def sample_params(family, n, rng):
    """Draw a random valid parameter set for ``family`` (used by the search).

    Period / wavelength are drawn log-uniform over ``[2, n//4]`` so the search
    covers short cycles (every-5th-row) as well as long ones.
    """
    hi = max(6, n // 4)
    if family == "sine":
        return {"wavelength": _logint(rng, 2, hi),
                "magnitude": float(rng.uniform(0.1, 1.5)),
                "phase": float(rng.uniform(0, 2 * np.pi))}
    if family == "comb":
        return {"period": _logint(rng, 2, hi),
                "magnitude": float(rng.uniform(0.3, 4.0)),
                "width": int(rng.integers(1, 4))}
    if family in ("square",):
        return {"wavelength": _logint(rng, 2, hi),
                "magnitude": float(rng.uniform(0.1, 1.0)),
                "duty": float(rng.uniform(0.2, 0.8))}
    if family in ("sawtooth", "triangle"):
        return {"wavelength": _logint(rng, 2, hi),
                "magnitude": float(rng.uniform(0.1, 1.0))}
    if family == "gaussian_bumps":
        return {"period": _logint(rng, 2, hi),
                "sigma": float(rng.uniform(0.4, 3.0)),
                "magnitude": float(rng.uniform(0.3, 2.5))}
    if family == "exp_decay":
        return {"halflife": _logint(rng, max(2, n // 20), max(4, n))}
    raise KeyError(family)
