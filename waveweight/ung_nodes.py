"""UNG node adapters for waveweight — pattern-searched sample weights with honest transfer scoring.

Pure top-level functions over the documented public API (``random_search``,
``bootstrap_refine``, ``build_all`` + ``null_controlled_search``, ``evaluate``).
Matrices and vectors cross the boundary as nested lists; numpy arrays never
leak out.  Stochastic steps are seeded, so every node is deterministic given
``seed``.  Each function returns a dict keyed by its declared output port names.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from waveweight import (bootstrap_refine, build_all, evaluate,
                        null_controlled_search, random_search)


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and containers of them) to plain JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def _families(families: str) -> list[str] | None:
    parts = [p.strip() for p in families.split(",") if p.strip()]
    return parts or None


def compute_weights(X: list[list[float]], y: list[float], n_iter: int = 200,
                    lam: float = 1.0, low_frac: float = 0.5, k: int = 2,
                    seed: int = 0, families: str = "") -> dict[str, Any]:
    """Random-search parametric weight patterns; return the best pattern's weights and its lift over uniform."""
    res = random_search(np.asarray(X, float), np.asarray(y, float),
                        families=_families(families), n_iter=int(n_iter),
                        lam=float(lam), low_frac=float(low_frac), k=int(k),
                        seed=int(seed))
    weights = res.pop("best_weights")
    return {"weights": _jsonable(weights), "report": _jsonable(res)}


def refine_weights(X: list[list[float]], y: list[float], n_candidates: int = 150,
                   n_boot: int = 12, lam: float = 1.0, low_frac: float = 0.5,
                   k: int = 2, seed: int = 0, top_m: int = 8) -> dict[str, Any]:
    """Bootstrap-aggregate the top weight patterns into robust per-record weights."""
    res = bootstrap_refine(np.asarray(X, float), np.asarray(y, float),
                           n_candidates=int(n_candidates), n_boot=int(n_boot),
                           lam=float(lam), low_frac=float(low_frac), k=int(k),
                           seed=int(seed), top_m=int(top_m))
    weights = res.pop("weights")
    return {"weights": _jsonable(weights), "report": _jsonable(res)}


def discover_weights(X: list[list[float]], y: list[float], n_index_each: int = 20,
                     n_null: int = 200, lam: float = 1.0,
                     seed: int = 0) -> dict[str, Any]:
    """Sweep index-pattern and data-driven window candidates; validate the winner on a held-out split vs a permutation null."""
    Xa = np.asarray(X, float)
    ya = np.asarray(y, float)
    rng = np.random.default_rng(int(seed))
    cands = build_all(Xa, ya, rng, n_index_each=int(n_index_each))
    res = null_controlled_search(Xa, ya, cands, lam=float(lam),
                                 n_null=int(n_null), seed=int(seed))
    return {"report": _jsonable(res)}


def evaluate_transfer(X: list[list[float]], y: list[float], weights: list[float],
                      lam: float = 1.0, low_frac: float = 0.5,
                      k: int = 2) -> dict[str, Any]:
    """Honest low-to-high transfer (worst-of-K), in-sample score, and their gap for given weights."""
    res = evaluate(np.asarray(X, float), np.asarray(y, float),
                   np.asarray(weights, float), lam=float(lam),
                   low_frac=float(low_frac), k=int(k))
    return {"report": _jsonable(res)}


_TAGS = ["license.mit", "runtime.python"]
_X_PORT = {"name": "X", "type_id": "amarel.types.matrix",
           "description": "Feature matrix, rows in time order (nested lists)."}
_Y_PORT = {"name": "y", "type_id": "amarel.types.vector",
           "description": "Target vector aligned with the rows of X."}

NODES = [
    {
        "fn": compute_weights,
        "id": "amarel.waveweight.compute-weights",
        "capabilities": ["weights.compute", "weights.pattern-search"],
        "summary": "Random-search periodic weight patterns (sine/comb/square/...) scored by honest low-to-high transfer.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "weights", "type_id": "amarel.types.weights",
             "description": "Best pattern's per-row weights."},
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{baseline, best_score, lift, best_family, best_params, top}."},
        ],
        "parameters": [
            {"name": "n_iter", "value_type": "integer", "default": 200, "required": False},
            {"name": "lam", "value_type": "number", "default": 1.0, "required": False},
            {"name": "low_frac", "value_type": "number", "default": 0.5, "required": False},
            {"name": "k", "value_type": "integer", "default": 2, "required": False},
            {"name": "seed", "value_type": "integer", "default": 0, "required": False},
            {"name": "families", "value_type": "string", "default": "", "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
    {
        "fn": refine_weights,
        "id": "amarel.waveweight.refine-weights",
        "capabilities": ["weights.compute", "weights.bootstrap-refine"],
        "summary": "Block-bootstrap-aggregate the top weight patterns into robust per-record weights.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "weights", "type_id": "amarel.types.weights",
             "description": "Robust bagged per-row weights."},
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{stability, best_family, best_params, baseline_transfer, refined_transfer, lift}."},
        ],
        "parameters": [
            {"name": "n_candidates", "value_type": "integer", "default": 150, "required": False},
            {"name": "n_boot", "value_type": "integer", "default": 12, "required": False},
            {"name": "lam", "value_type": "number", "default": 1.0, "required": False},
            {"name": "low_frac", "value_type": "number", "default": 0.5, "required": False},
            {"name": "k", "value_type": "integer", "default": 2, "required": False},
            {"name": "seed", "value_type": "integer", "default": 0, "required": False},
            {"name": "top_m", "value_type": "integer", "default": 8, "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
    {
        "fn": discover_weights,
        "id": "amarel.waveweight.discover-weights",
        "capabilities": ["weights.discover", "validation.null-controlled"],
        "summary": "Sweep index-pattern plus data-driven window-driver weight candidates; accept only if the held-out winner beats a permutation null.",
        "inputs": [_X_PORT, _Y_PORT],
        "outputs": [
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{best_label, best_meta, selection_transfer, val_transfer, uniform_val, lift_over_uniform, null_mean, null_p95, null_p, beats_null, n_candidates, top}."},
        ],
        "parameters": [
            {"name": "n_index_each", "value_type": "integer", "default": 20, "required": False},
            {"name": "n_null", "value_type": "integer", "default": 200, "required": False},
            {"name": "lam", "value_type": "number", "default": 1.0, "required": False},
            {"name": "seed", "value_type": "integer", "default": 0, "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
    {
        "fn": evaluate_transfer,
        "id": "amarel.waveweight.evaluate-transfer",
        "capabilities": ["weights.evaluate"],
        "summary": "Score given sample weights by honest low-to-high transfer (worst-of-K) plus the in-sample gap.",
        "inputs": [
            _X_PORT, _Y_PORT,
            {"name": "weights", "type_id": "amarel.types.weights",
             "description": "Per-row sample weights to evaluate."},
        ],
        "outputs": [
            {"name": "report", "type_id": "amarel.types.report",
             "description": "{'transfer', 'insample', 'gap'}."},
        ],
        "parameters": [
            {"name": "lam", "value_type": "number", "default": 1.0, "required": False},
            {"name": "low_frac", "value_type": "number", "default": 0.5, "required": False},
            {"name": "k", "value_type": "integer", "default": 2, "required": False},
        ],
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": _TAGS,
    },
]
