"""Contract tests for the UNG node adapters (ung_nodes + fixtures + metadata)."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from waveweight import ung_nodes

NODES = ung_nodes.NODES
ID_PREFIX = "amarel.waveweight."
FIXTURE_DIR = Path(ung_nodes.__file__).resolve().parent / "ung_fixtures"
VALUE_TYPES = {"string", "integer", "number", "boolean"}


def _approx(expect, got, path="$"):
    """Recursive equality with pytest.approx on floats; returns mismatch strings."""
    if isinstance(expect, dict):
        if not isinstance(got, dict) or set(expect) != set(got):
            return [f"{path}: dict mismatch {expect!r} vs {got!r}"]
        errs = []
        for k in expect:
            errs.extend(_approx(expect[k], got[k], f"{path}.{k}"))
        return errs
    if isinstance(expect, list):
        if not isinstance(got, list) or len(expect) != len(got):
            return [f"{path}: list mismatch {expect!r} vs {got!r}"]
        errs = []
        for i, (e, g) in enumerate(zip(expect, got)):
            errs.extend(_approx(e, g, f"{path}[{i}]"))
        return errs
    if isinstance(expect, bool) or isinstance(got, bool):
        return [] if expect == got else [f"{path}: {expect!r} != {got!r}"]
    if isinstance(expect, (int, float)) and isinstance(got, (int, float)):
        ok = got == pytest.approx(expect, rel=1e-6, abs=1e-9)
        return [] if ok else [f"{path}: {expect!r} !~ {got!r}"]
    return [] if expect == got else [f"{path}: {expect!r} != {got!r}"]


def _run(node, case):
    kwargs = {**case.get("inputs", {}), **case.get("parameters", {})}
    return node["fn"](**kwargs)


def _cases():
    out = []
    for node in NODES:
        path = FIXTURE_DIR / (node["id"] + ".json")
        for i, case in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            out.append(pytest.param(node, case, id=f"{node['id']}#{i}"))
    return out


@pytest.mark.parametrize("node,case", _cases())
def test_fixture_case(node, case):
    result = _run(node, case)
    assert json.loads(json.dumps(result)) == result, "output must be JSON-clean"
    errors = _approx(case["expect"], result)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("node,case", _cases())
def test_determinism_double_run(node, case):
    a, b = _run(node, case), _run(node, case)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_metadata_sanity():
    ids = [n["id"] for n in NODES]
    assert len(ids) == len(set(ids)), "node ids must be unique"
    for node in NODES:
        nid = node["id"]
        assert nid.startswith(ID_PREFIX), nid
        action = nid[len(ID_PREFIX):]
        assert action and all(c.islower() or c.isdigit() or c == "-" for c in action)
        assert callable(node["fn"])
        assert node["summary"].strip()
        assert node["capabilities"]
        assert node["effects"] == []
        assert node["determinism"] == "deterministic"
        assert node["idempotency"] == "idempotent"
        assert {"license.mit", "runtime.python"} <= set(node["tags"])
        sig = set(inspect.signature(node["fn"]).parameters)
        in_names = [p["name"] for p in node["inputs"]]
        par_names = [p["name"] for p in node["parameters"]]
        declared = in_names + par_names
        assert len(declared) == len(set(declared)), nid
        assert set(declared) <= sig, f"{nid}: declared names not in signature"
        for port in node["inputs"] + node["outputs"]:
            assert port["type_id"].startswith("amarel.types."), nid
            assert port["description"].strip()
        for param in node["parameters"]:
            assert param["value_type"] in VALUE_TYPES, nid
            assert param.get("required") or "default" in param, nid

        cases = json.loads((FIXTURE_DIR / (nid + ".json")).read_text(encoding="utf-8"))
        assert len(cases) >= 2, f"{nid}: need at least 2 fixture cases"
        out_names = {p["name"] for p in node["outputs"]}
        for case in cases:
            assert set(case["expect"]) == out_names, f"{nid}: expect keys != output ports"
            assert set(case.get("inputs", {})) <= set(in_names), nid
            assert set(case.get("parameters", {})) <= set(par_names), nid


def test_module_importable():
    import importlib
    mod = importlib.import_module("waveweight.ung_nodes")
    assert hasattr(mod, "NODES") and mod.NODES
