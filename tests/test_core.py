"""Tests for waveweight — weighted ridge, patterns, and that a matching weight
pattern actually lifts transfer on data with planted periodic importance.
Requires numpy; no network."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waveweight import (  # noqa: E402
    patterns, weighted_ridge, evaluate, random_search, bootstrap_refine, pearson,
)
from waveweight.bootstrap import _block_resample  # noqa: E402
from waveweight.synth import make_periodic_importance  # noqa: E402


def test_patterns_shape_and_period():
    s = patterns.sine(100, wavelength=10, magnitude=0.5)
    assert len(s) == 100 and (s >= 0).all()
    # a sine of wavelength 10 correlates strongly with itself shifted by 10
    assert pearson(s[:90], s[10:]) > 0.9
    c = patterns.comb(20, period=5, magnitude=2.0)
    assert c[0] > c[1] and c[5] > c[6] and c[10] > c[11]      # spikes every 5
    assert patterns.make("triangle", 30, {"wavelength": 6}).shape == (30,)


def test_weighted_ridge_reduces_to_ols():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta
    pred = weighted_ridge(X[:70], y[:70], X[70:], np.ones(70), lam=1e-6)
    assert pearson(pred, y[70:]) > 0.999


def test_evaluate_keys():
    X, y, _ = make_periodic_importance(n=600, period=5, seed=0)
    r = evaluate(X, y, patterns.uniform(len(y)))
    assert set(r) == {"transfer", "insample", "gap"}


def test_planted_period_weighting_beats_uniform():
    # the core hypothesis: upweighting the clean (every-5th) rows lifts transfer
    X, y, info = make_periodic_importance(n=3000, period=5, seed=0)
    n = len(y)
    uni = evaluate(X, y, patterns.uniform(n))["transfer"]
    comb = evaluate(X, y, patterns.comb(n, period=5, magnitude=4.0))["transfer"]
    assert comb > uni + 0.02, (uni, comb)


def test_random_search_finds_lift():
    X, y, _ = make_periodic_importance(n=3000, period=5, seed=1)
    res = random_search(X, y, n_iter=300, seed=0)
    assert res["lift"] > 0.02, res["lift"]
    assert res["best_family"] in patterns.PATTERNS


def test_bootstrap_refine_generalizes():
    X, y, _ = make_periodic_importance(n=2400, period=5, seed=2)
    bo = bootstrap_refine(X, y, n_boot=8, n_candidates=80, seed=0)
    assert len(bo["weights"]) == len(y)
    assert (bo["stability"] >= 0).all()
    # robust weights should not underperform uniform on the held-out late rows
    assert bo["refined_transfer"] >= bo["baseline_transfer"] - 1e-6


def test_block_resample_valid():
    rng = np.random.default_rng(0)
    idx = _block_resample(100, 10, rng)
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("\n%d passed" % len(fns))
