"""waveweight — search periodic sample-weight patterns that lift transfer.

For time-ordered training data where a record's *importance* may follow a
pattern (every k-th row matters more; neighbors rise and fall like a sine wave),
waveweight sweeps parametric weight patterns (sine / comb / square / sawtooth /
triangle / gaussian-bumps / exp-decay), scores each by honest low→high transfer,
and bootstrap-aggregates the winners into robust per-record weights.

    from waveweight import random_search
    from waveweight.synth import make_periodic_importance
    X, y, info = make_periodic_importance(period=5)
    res = random_search(X, y, n_iter=300)
    print(res["baseline"], res["best_score"], res["best_family"], res["best_params"])

Pairs with `shiftblend` (which scores transfer and blends members); here the
lever is *which rows to trust*. numpy-only.
"""

from __future__ import annotations

from . import patterns, synth, curves, window, candidates
from .weighted import weighted_ridge, make_weighted_ridge, pearson
from .score import evaluate, time_split
from .search import random_search
from .bootstrap import bootstrap_refine
from .candidates import build_all, index_pattern_candidates, window_driver_candidates
from .nulltest import null_controlled_search, three_split

__all__ = [
    "patterns", "synth", "curves", "window", "candidates",
    "weighted_ridge", "make_weighted_ridge", "pearson",
    "evaluate", "time_split",
    "random_search", "bootstrap_refine",
    "build_all", "index_pattern_candidates", "window_driver_candidates",
    "null_controlled_search", "three_split",
]
__version__ = "0.2.0"
