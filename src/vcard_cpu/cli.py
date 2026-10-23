"""vcard-cpu CLI: inspect, run and one-off compile vCard kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .astvec import AstToAvx2, ExprError
from .jit import JITError, VcfJit, show_plan
from .vcard import VCardError, parse_vcf


def _load(path_or_text: str) -> str:
    p = Path(path_or_text)
    return p.read_text(encoding="utf-8") if p.exists() else path_or_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vcard-cpu",
        description="Compile vCard 4.0 files into AVX2 compute kernels.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="show cards and the generated C plan")
    p_show.add_argument("vcf")

    p_run = sub.add_parser("run", help="compile + execute a kernel")
    p_run.add_argument("vcf")
    p_run.add_argument("--n", type=int, default=1_000_000, help="element count")
    p_run.add_argument("--param", action="append", default=[],
                       metavar="NAME=VALUE", help="scalar kernel params")

    p_expr = sub.add_parser("expr", help="compile a Python expression into AVX2")
    p_expr.add_argument("expression")
    p_expr.add_argument("-i", "--input", action="append", required=True,
                        metavar="NAME", help="input variable name")
    p_expr.add_argument("--n", type=int, default=1_000_000)
    p_expr.add_argument("--emit-c", action="store_true", help="print C instead of benchmarking")

    ns = parser.parse_args(argv)

    try:
        if ns.cmd == "show":
            print(show_plan(ns.vcf))
            return 0

        if ns.cmd == "expr":
            comp = AstToAvx2(ns.expression, ns.input)
            if ns.emit_c:
                print(comp.generate_c())
                return 0
            jit = _compile_expr(comp)
            rng = np.random.default_rng(0)
            arrays = {name: rng.standard_normal(ns.n).astype(np.float32) for name in ns.input}
            import time
            t0 = time.perf_counter()
            out = jit.run(arrays)
            dt = time.perf_counter() - t0
            gops = ns.n * 8 / dt / 1e9  # rough: 2 flops/element ~ *8 ports; keep naive
            print(f"n={ns.n} elapsed={dt*1e3:.2f} ms (~{gops:.2f} GOPS naive) out[:3]={out['out'][:3]}")
            return 0

        # run
        jit = VcfJit(text=_load(ns.vcf))
        rng = np.random.default_rng(0)
        arrays = {name: rng.uniform(-5, 5, ns.n).astype(np.float32) for name in jit.meta_inputs}
        params = {}
        for chunk in ns.param:
            k, v = chunk.split("=", 1)
            params[k] = float(v)
        import time
        t0 = time.perf_counter()
        outs = jit.run(arrays, params=params)
        dt = time.perf_counter() - t0
        print(f"[+] kernel executed: n={ns.n} in {dt*1e3:.2f} ms")
        for name, arr in outs.items():
            print(f"    {name}: mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}")
        return 0

    except (VCardError, JITError, ExprError, ValueError) as exc:
        print(f"[-] {exc}", file=sys.stderr)
        return 1


def _compile_expr(comp: AstToAvx2):
    """Wrap an AstToAvx2 generated kernel into a VcfJit-like runner."""
    from .jit import _sanitize
    import ctypes, os, subprocess, tempfile, hashlib
    from pathlib import Path

    src = comp.generate_c()
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vcard_cpu"
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"expr_{digest}.so"
    if not so.exists():
        with tempfile.TemporaryDirectory() as td:
            cf = Path(td) / "k.c"
            sf = Path(td) / "k.so"
            cf.write_text(src)
            r = subprocess.run(["cc", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
                                "-o", str(sf), str(cf)], capture_output=True, text=True)
            if r.returncode != 0:
                raise JITError(f"gcc failed:\n{r.stderr}")
            os.replace(sf, so)
    lib = ctypes.CDLL(str(so))
    fn = lib.execute_kernel
    ptr = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [ptr] * len(comp.input_names) + [ptr, ctypes.c_int]
    fn.restype = None

    class _Runner:
        meta_inputs = comp.input_names

        def run(self, arrays, **kw):
            n = len(next(iter(arrays.values())))
            out = np.zeros(n, dtype=np.float32)
            args = [np.ascontiguousarray(arrays[k], dtype=np.float32).ctypes.data_as(ptr)
                    for k in comp.input_names]
            fn(*args, out.ctypes.data_as(ptr), ctypes.c_int(n))
            return {"out": out}

    return _Runner()


if __name__ == "__main__":
    raise SystemExit(main())
