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
from .search import random_search
from .synth import make_periodic_importance


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
    p = sub.add_parser("search", help="search weight patterns on your CSV")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lam", type=float, default=1.0)
    p.set_defaults(fn=cmd_search)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
