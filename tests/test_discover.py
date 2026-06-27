"""Tests for the data-driven + null-controlled machinery (waveweight v0.2).

The two that matter most:
  * NO-SIGNAL data → the best swept weighting does NOT beat the null (no false
    positive; a null result is the honest verdict).
  * DATA-DRIVEN local importance → a window driver discovers it and beats null.
Requires numpy; no network."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waveweight import (  # noqa: E402
    curves, window, build_all, null_controlled_search,
)
from waveweight.synth import (  # noqa: E402
    make_no_signal, make_local_importance, make_periodic_importance,
)


def test_curves_normalize_and_shape():
    z = np.linspace(0, 1, 50)
    for name, fn in curves.CURVES.items():
        w = curves.normalize(fn(z))
        assert len(w) == 50 and (w > 0).all()
        assert abs(w.mean() - 1.0) < 1e-6, name
    # mask_top puts more weight on the high end of the driver
    w = curves.normalize(curves.mask_top(z, q=0.2))
    assert w[-1] > w[0]


def test_window_features_detect_local_structure():
    sig = np.array([0, 0, 5, 0, 0, 0, 0, 9, 0, 0], float)   # spikes at 2 and 7
    wf = window.window_features(sig, k=3)
    assert wf["is_extremum"][2] == 1.0 and wf["is_extremum"][7] == 1.0
    assert wf["center_z"][2] > wf["center_z"][1]            # the spike stands out


def test_no_signal_does_not_beat_null():
    # the crucial honesty test: no real sample structure -> no win over null
    X, y, _ = make_no_signal(n=1600, seed=0)
    rng = np.random.default_rng(0)
    cands = build_all(X, y, rng, n_index_each=12)
    res = null_controlled_search(X, y, cands, n_null=200, seed=0)
    assert res["lift_over_uniform"] < 0.03, res["lift_over_uniform"]
    assert not res["beats_null"], res


def test_data_driven_local_importance_is_found():
    X, y, info = make_local_importance(n=2400, k=5, seed=1)
    rng = np.random.default_rng(1)
    cands = build_all(X, y, rng, k_values=(3, 5, 7), n_index_each=12)
    res = null_controlled_search(X, y, cands, n_null=200, seed=0)
    assert res["beats_null"], res
    assert res["lift_over_uniform"] > 0.05, res["lift_over_uniform"]
    assert res["best_meta"]["kind"] == "window"            # discovered from data, not index


def test_build_all_labels_and_normalized():
    X, y, _ = make_periodic_importance(n=400, seed=0)
    rng = np.random.default_rng(0)
    cands = build_all(X, y, rng, n_index_each=3)
    assert len(cands) > 20
    for lbl, w, meta in cands[:50]:
        assert abs(w.mean() - 1.0) < 1e-6 and (w > 0).all()
        assert lbl.startswith("idx:") or lbl.startswith("win:")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("\n%d passed" % len(fns))
