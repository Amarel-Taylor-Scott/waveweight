# waveweight

> What if, in a time series, a record's *importance* follows a pattern — every
> k-th row matters more, with neighbors rising and falling like a sine wave?
> **waveweight** searches parametric sample-weight patterns (sine / comb / square
> / sawtooth / triangle / gaussian-bumps / recency), scores each by honest
> low→high **transfer**, and bootstrap-aggregates the winners into robust
> per-record weights. numpy-only.

```python
from waveweight import random_search
from waveweight.synth import make_periodic_importance

X, y, info = make_periodic_importance(period=5)   # rows in time order
res = random_search(X, y, n_iter=400)
print(res["baseline"], res["best_score"], res["best_family"], res["best_params"])
```

```
$ waveweight demo
## pattern search (transfer = low->high worst-of-2)
  uniform-weight baseline : 0.0652
  best pattern transfer   : 0.5645   (lift +0.4992)
  best pattern            : sine {'wavelength': 5, 'magnitude': 1.04, 'phase': 1.998}
## bootstrap-refined robust weights (block-bootstrap x 12)
  refined transfer        : 0.5320   (lift +0.4667 over uniform)
```

The search **rediscovers the planted period** (5): upweighting the informative
phase recovers signal that uniform weighting averaged away — a 0.065 → 0.56 jump.

## Why this can work

Not all training rows are equally informative, and the informative ones may be
**periodic** (a market microstructure cycle, a logging cadence, a sampling
artifact, a day-of-week effect in row order). Uniform weighting lets the
uninformative majority dominate the fit. If you weight records by a pattern that
lines up with the informative phase, the model leans on the rows that carry
transferable signal. waveweight searches for that pattern and measures whether it
actually buys generalization.

## Discover importance from the data — and prove it against a null (v0.2)

Imposing a weight pattern *by row index* (sine/comb) is only one lever, and on
real data it's often a dead end. The stronger lever is to **derive importance
from the data**: look at each row's neighborhood (a group of k samples), compute
a local relationship, and let that decide the weight — e.g. "this row is a local
extremum / a local outlier / sits in a low-volatility stretch, so trust it more."

`discover` sweeps the **full space** — every driver × every curve × window sizes
× magnitudes — and validates honestly:

- **drivers**: index patterns (`sine/comb/square/sawtooth/triangle/gaussian/exp`)
  **and** window features (`local_mean/local_std/center_z/is_extremum/slope/
  local_range`) over the target and over feature columns;
- **curves / formulas / masks**: `linear, power, sigmoid, mask_top, mask_mid,
  rank, invert` (add your own to `curves.CURVES`);
- **null-controlled, held-out protocol**: select the best candidate on one split,
  validate on a **held-out** split it never saw, and compare to a
  **magnitude-matched permutation null**. A weighting counts **only** if it beats
  the null's 95th percentile.

```
$ waveweight rigor
## 1. NO sample structure (control)
  best weighting   : win:x0.is_extremum.k5|sigmoid
  held-out transfer: 0.9716   (uniform 0.9721, lift -0.0005)
  null p95 / p     : 0.9724 / p=0.790  over 1281 candidates
  VERDICT          : no signal — does NOT beat the null (the honest answer)

## 2. DATA-DRIVEN local importance (informative rows = local extrema)
  best weighting   : win:x0.is_extremum.k5|sigmoid       <- discovered, not assumed
  held-out transfer: 0.9033   (uniform 0.6583, lift +0.2449)
  null p95 / p     : 0.7098 / p=0.000
  VERDICT          : REAL — beats the held-out null
```

```python
from waveweight import build_all, null_controlled_search
import numpy as np
cands = build_all(X, y, np.random.default_rng(0))   # index + window drivers x curves
res = null_controlled_search(X, y, cands, n_null=200)
res["beats_null"], res["best_label"], res["lift_over_uniform"]
```

**A null result is a finding, not a failure.** When nothing beats the null
(case 1 above, across 1,281 swept candidates), that *is* the data's answer — and
trusting it is what keeps you from shipping a weighting that overfits the split
you tuned on. If you think there's signal, widen the space (more window features,
more curves, more feature-column drivers, finer windows) — it's all still
null-controlled, so a real effect will surface and a phantom won't.

## How it works

| Step | What | Module |
|---|---|---|
| **patterns** | parametric weight vectors over the row index (period log-sampled) | `patterns` |
| **weighted fit** | weighted ridge via the Gram trick (`Xᵀ W X`) | `weighted` |
| **feedback** | fit early → score late, worst-of-K (honest transfer) + the in-sample gap | `score` |
| **search** | random sweep of families × params → best **lift** over uniform | `search` |
| **bootstrap** | bag candidates over training resamples → robust per-record weights | `bootstrap` |

```python
from waveweight import bootstrap_refine
bo = bootstrap_refine(X, y, n_boot=12)
bo["weights"]            # robust per-record weight vector
bo["stability"]          # per-record spread across top candidates
bo["lift"]               # transfer lift over uniform, on held-out late rows
```

## The honesty rule (important)

Selection looks at the late rows, so this can overfit if you give it too many
degrees of freedom. Two guardrails, both baked in:

1. **Patterns are low-parameter** — a wavelength + magnitude + phase. A few knobs
   can't memorize the eval rows the way free per-row weights would.
2. **Bootstrap, then confirm** — `bootstrap_refine` keeps the pattern that helps
   across *many* training resamples, not one lucky split. Still, confirm the
   winning pattern on a final untouched draw before trusting it.

A positive, bootstrap-stable lift is evidence of real periodic importance; a lift
that only one search finds, or that vanishes under bootstrapping, is noise.

## CLI

```bash
waveweight rigor                         # null-controlled discovery on 3 datasets (no-signal/local/periodic)
waveweight discover data.csv --target y  # full driver×curve sweep, held-out + null verdict
waveweight demo                          # the planted-period (index) walk-through
waveweight search data.csv --target y    # index-pattern search only
```

## Pairs with

[`shiftblend`](https://github.com/Amarel-Taylor-Scott/shiftblend) — there the
lever is *which models/features* to blend under covariate shift; here it's *which
rows to trust*. Same honest-transfer scoring underneath.

## Install

```bash
pip install numpy && git clone https://github.com/Amarel-Taylor-Scott/waveweight.git
```

MIT. Depends only on numpy. Any sklearn-style estimator can replace the built-in
weighted ridge (it just needs to accept `sample_weight`).
