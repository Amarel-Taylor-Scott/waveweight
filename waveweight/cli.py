"""``waveweight`` CLI — a worked demo and search-your-own-CSV.

    waveweight demo                         # planted-period synthetic walk-through
    waveweight search data.csv --target y   # search weight patterns on your data
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .bootstrap import bootstrap_refine
from .candidates import build_all
from .nulltest import null_controlled_search
from .search import random_search
from .synth import (make_local_importance, make_no_signal,
                    make_periodic_importance)


def cmd_demo(_a) -> int:
    X, y, info = make_periodic_importance(n=3000, period=5, seed=0)
    print("# planted importance every %d-th row (clean), %d rows x %d feats\n"
          % (info["period"], *X.shape))

    res = random_search(X, y, n_iter=400, seed=0)
    print("## pattern search (transfer = low->high worst-of-2)")
    print("  uniform-weight baseline : %.4f" % res["baseline"])
    print("  best pattern transfer   : %.4f   (lift %+.4f)" % (res["best_score"], res["lift"]))
    print("  best pattern            : %s %s" % (res["best_family"], res["best_params"]))
    print("  top families            : %s\n"
          % ", ".join(dict.fromkeys(r["family"] for r in res["top"])))

    print("## bootstrap-refined robust weights (block-bootstrap x 12)")
    bo = bootstrap_refine(X, y, n_boot=12, n_candidates=160, seed=1)
    print("  refined transfer        : %.4f   (lift %+.4f over uniform)" % (bo["refined_transfer"], bo["lift"]))
    print("  winning families        : %s" % ", ".join(dict.fromkeys(bo["families"])))
    print("  reading: a positive lift means upweighting the clean phase carried")
    print("           transferable signal that uniform weighting averaged away.")
    return 0


def _verdict(res):
    print("  best weighting   : %s" % res["best_label"])
    print("  held-out transfer: %.4f   (uniform %.4f, lift %+.4f)"
          % (res["val_transfer"], res["uniform_val"], res["lift_over_uniform"]))
    print("  null (p95 / mean): %.4f / %.4f   p=%.3f over %d candidates"
          % (res["null_p95"], res["null_mean"], res["null_p"], res["n_candidates"]))
    print("  VERDICT          : %s" %
          ("REAL — beats the held-out null" if res["beats_null"]
           else "no signal — does NOT beat the null (the honest answer)"))


def cmd_rigor(_a) -> int:
    """Null-controlled discovery on three datasets: no-signal, data-driven, periodic."""
    import numpy as np
    print("Every result is selected on one split and validated on a held-out split")
    print("against a magnitude-matched permutation null. A pattern counts ONLY if it")
    print("beats that null — otherwise 'no signal' is the correct, data-driven verdict.\n")

    print("## 1. NO sample structure (control — nothing should win)")
    X, y, _ = make_no_signal(n=1800, seed=0)
    _verdict(null_controlled_search(X, y, build_all(X, y, np.random.default_rng(0),
             n_index_each=15), n_null=200, seed=0))

    print("\n## 2. DATA-DRIVEN local importance (informative rows = local extrema)")
    X, y, info = make_local_importance(n=2400, k=5, seed=1)
    _verdict(null_controlled_search(X, y, build_all(X, y, np.random.default_rng(1),
             k_values=(3, 5, 7), n_index_each=15), n_null=200, seed=0))
    print("  -> discovered from the data (a window driver), not assumed by index.")

    print("\n## 3. PERIODIC importance (every k-th row)")
    X, y, info = make_periodic_importance(n=2400, period=5, seed=2)
    _verdict(null_controlled_search(X, y, build_all(X, y, np.random.default_rng(2),
             k_values=(3, 5, 7), n_index_each=25), n_null=200, seed=0))
    return 0


def cmd_discover(a) -> int:
    import numpy as np
    X, y = _load_csv(a.file, a.target)
    print("# %s: %d rows x %d feats — null-controlled weighting search\n" % (a.file, *X.shape))
    cands = build_all(X, y, np.random.default_rng(0),
                      k_values=tuple(a.windows), n_index_each=a.index_each)
    _verdict(null_controlled_search(X, y, cands, n_null=a.nulls, lam=a.lam, seed=0))
    return 0


def _load_csv(path, target):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    if target not in header:
        raise SystemExit("waveweight: target %r not found" % target)
    ti = header.index(target)
    fi = [i for i in range(len(header)) if i != ti]
    X = np.array([[float(row[i]) for i in fi] for row in rows], float)
    y = np.array([float(row[ti]) for row in rows], float)
    return X, y


def cmd_search(a) -> int:
    X, y = _load_csv(a.file, a.target)
    print("# %s: %d rows x %d feats (rows assumed time-ordered)\n" % (a.file, *X.shape))
    res = random_search(X, y, n_iter=a.iters, lam=a.lam, seed=0)
    print("uniform baseline : %.4f" % res["baseline"])
    print("best transfer    : %.4f  (lift %+.4f)" % (res["best_score"], res["lift"]))
    print("best pattern     : %s %s" % (res["best_family"], res["best_params"]))
    if res["lift"] <= 0:
        print("\n(no pattern beat uniform — record importance looks aperiodic here.)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="waveweight", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="planted-period synthetic walk-through").set_defaults(fn=cmd_demo)
    sub.add_parser("rigor", help="null-controlled discovery on 3 datasets (no-signal/local/periodic)").set_defaults(fn=cmd_rigor)
    p = sub.add_parser("search", help="search weight patterns on your CSV")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lam", type=float, default=1.0)
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("discover", help="null-controlled weighting search on your CSV (index + window drivers)")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7, 9])
    p.add_argument("--index-each", type=int, default=20)
    p.add_argument("--nulls", type=int, default=200)
    p.add_argument("--lam", type=float, default=1.0)
    p.set_defaults(fn=cmd_discover)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
