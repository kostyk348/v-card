import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vcard_cpu.astvec import AstToAvx2, ExprError

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                                reason="no C compiler")


def run_expr(expr, inputs, n=1_011, seed=0):
    comp = AstToAvx2(expr, list(inputs))
    src = comp.generate_c()
    assert "execute_kernel" in src
    jit = _JitShim(comp, src)
    rng = np.random.default_rng(seed)
    arrays = {k: rng.uniform(-4, 4, n).astype(np.float32) for k in inputs}
    return jit.run(arrays)["out"], arrays


class _JitShim:
    """Compile AstToAvx2 output and call it — small test-local shim."""

    def __init__(self, comp, src):
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vcard_cpu"
        cache.mkdir(parents=True, exist_ok=True)
        so = cache / f"ast_{hashlib.sha256(src.encode()).hexdigest()[:16]}.so"
        if not so.exists():
            with tempfile.TemporaryDirectory() as td:
                cf = Path(td) / "k.c"
                sf = Path(td) / "k.so"
                cf.write_text(src)
                r = subprocess.run(["cc", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
                                    "-o", str(sf), str(cf)], capture_output=True, text=True)
                assert r.returncode == 0, r.stderr
                os.replace(sf, so)
        lib = ctypes.CDLL(str(so))
        fn = lib.execute_kernel
        ptr = ctypes.POINTER(ctypes.c_float)
        fn.argtypes = [ptr] * len(comp.input_names) + [ptr, ctypes.c_int]
        fn.restype = None
        self._fn, self._names, self._ptr = fn, comp.input_names, ptr

    def run(self, arrays):
        n = len(next(iter(arrays.values())))
        out = np.zeros(n, dtype=np.float32)
        args = [np.ascontiguousarray(arrays[k], dtype=np.float32).ctypes.data_as(self._ptr)
                for k in self._names]
        self._fn(*args, out.ctypes.data_as(self._ptr), ctypes.c_int(n))
        return {"out": out}


def test_add_mul():
    out, a = run_expr("a * b + c", ["a", "b", "c"])
    ref = (a["a"] * a["b"] + a["c"]).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_constants_hoisted():
    out, a = run_expr("(a + 3.5) * (b - 1.25)", ["a", "b"])
    ref = ((a["a"] + 3.5) * (a["b"] - 1.25)).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_abs_sqrt_div():
    # SQRT clamps negatives by design -> feed it abs(b) to match numpy semantics
    out, a = run_expr("abs(a) / (sqrt(abs(b)) + 0.5)", ["a", "b"])
    ref = (np.abs(a["a"]) / (np.sqrt(np.abs(a["b"])) + 0.5)).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_nested_min_max():
    out, a = run_expr("maximum(minimum(a, b), 0.1)", ["a", "b"])
    ref = np.maximum(np.minimum(a["a"], a["b"]), 0.1).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_unary_minus():
    out, a = run_expr("-a + b", ["a", "b"])
    ref = (-a["a"] + a["b"]).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_rejects_unknown_name():
    with pytest.raises(ExprError, match="unknown input"):
        AstToAvx2("a + evil", ["a"])


def test_rejects_unsafe_syntax():
    with pytest.raises(ExprError):
        AstToAvx2("__import__('os').system('x')", ["a"])
    with pytest.raises(ExprError):
        AstToAvx2("a if b else c", ["a", "b"])
    with pytest.raises(ExprError):
        AstToAvx2("'a' * 'b'", ["a", "b"])
