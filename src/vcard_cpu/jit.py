"""The vCard JIT: parse -> graph -> fused C/AVX2 -> gcc -> ctypes."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .graph import GraphError, OpNode, build_graph, external_inputs, sinks, topological_order
from .ops import COMMON_CONSTS, REGISTRY, arity
from .vcard import VCard, VCardFile, VCardError, parse_vcf, parse_vcf_file


class JITError(RuntimeError):
    """Compilation or linking failure."""


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "x"


# ---------------------------------------------------------------------------
# code emission
# ---------------------------------------------------------------------------

def _emit_graph_kernel(order: List[OpNode], inputs: Sequence[str],
                       outs: Sequence[OpNode]) -> tuple[str, set]:
    """One fused loop: every node becomes a register-to-register __m256 op."""
    lines: List[str] = []
    used_consts: set[str] = set()

    def emit(emitter_kind: str) -> List[str]:
        body: List[str] = []
        temps: Dict[str, str] = {}
        for idx, node in enumerate(order):
            spec = REGISTRY[node.op]
            ar, avx, scalar, consts = spec
            used_consts.update(consts)
            args = []
            for inp in node.inputs:
                if inp in temps:
                    args.append(temps[inp])
                else:
                    args.append(f"v_in__{_sanitize(inp)}" if emitter_kind == "avx"
                                else f"s_in__{_sanitize(inp)}")
            if len(args) != ar:
                raise JITError(f"node {node.uid} op {node.op} needs {ar} inputs, got {len(args)}")
            var = f"t{idx}" if emitter_kind == "avx" else f"s{idx}"
            gen = avx if emitter_kind == "avx" else scalar
            body.append(f"    float_ty {var} = {gen(*args)};")
            temps[node.uid] = var
        for out_idx, sink in enumerate(outs):
            var = temps[sink.uid]
            if emitter_kind == "avx":
                body.append(f"    _mm256_storeu_ps(&out__{_sanitize(sink.uid)}[i], {var});")
            else:
                body.append(f"    out__{_sanitize(sink.uid)}[i] = {var};")
        return body

    # vector loop
    lines.append("    int i = 0;")
    lines.append("    for (; i <= n - 8; i += 8) {")
    for name in inputs:
        lines.append(f"        __m256 v_in__{_sanitize(name)} = _mm256_loadu_ps(&in__{_sanitize(name)}[i]);")
    for ln in emit("avx"):
        lines.append("    " + ln.replace("float_ty", "__m256"))
    lines.append("    }")
    # scalar tail
    lines.append("    for (; i < n; i++) {")
    for name in inputs:
        lines.append(f"        float s_in__{_sanitize(name)} = in__{_sanitize(name)}[i];")
    for ln in emit("scalar"):
        lines.append("    " + ln.replace("float_ty", "float"))
    lines.append("    }")
    return "\n".join(lines), used_consts


def _parse_micro_signature(src: str) -> tuple[List[str], List[str], List[str]]:
    """Extract inputs/outputs/params from a handwritten execute_kernel signature."""
    import re

    m = re.search(r"void\s+execute_kernel\s*\(([^)]*)\)", src)
    if not m:
        raise JITError("X-MICRO-OP-C must define void execute_kernel(...)")
    inputs, outputs, params = [], [], []
    for raw in m.group(1).split(","):
        arg = " ".join(raw.split())
        if not arg or arg == "int n":
            continue
        if arg.startswith("const float*"):
            inputs.append(arg.split()[-1])
        elif arg.startswith("float*"):
            outputs.append(arg.split()[-1])
        elif arg.startswith("float"):
            params.append(arg.split()[-1])
    return inputs, outputs, params


def _build_source(vcf: VCardFile) -> tuple[str, List[str], List[str], List[str]]:
    """Return (c_source, input_names, output_names, param_names)."""
    # --- mode 1: fully handwritten C (X-MICRO-OP-C) ---------------------
    for card in vcf.cards:
        if card.micro_op_c:
            src = card.micro_op_c.strip() + "\n"
            inputs, outputs, params = _parse_micro_signature(src)
            return src, inputs, outputs, params

    # --- mode 2: raw inner-loop bodies (X-KERNEL-BODY) -------------------
    bodies = [c for c in vcf.cards if c.body and c.op is None]
    if bodies:
        card = bodies[0]
        inputs = card.inputs()
        outputs = card.outputs() or ["out"]
        params: List[str] = []
        raw_params = card.get("X-PARAMS", "")
        for chunk in raw_params.split(","):
            chunk = chunk.strip()
            if chunk:
                nm, ty = (chunk.split(":") + ["float"])[:2]
                params.append((nm.strip(), ty.strip()))
        args = [f"const float* in__{_sanitize(i)}" for i in inputs]
        args += [f"float* out__{_sanitize(o)}" for o in outputs]
        args.append("int n")
        args += [f"float {_sanitize(nm)}" for nm, _ty in params]
        sig = ", ".join(args)
        consts = "\n".join(f"    {c}" for c in COMMON_CONSTS.values())
        src = (
            "#include <immintrin.h>\n#include <math.h>\n#include <stdint.h>\n\n"
            f"void execute_kernel({sig}) {{\n"
            f"{consts}\n"
            f"{card.body.strip()}\n"
            "}\n"
        )
        return src, inputs, outputs, [nm for nm, _ in params]

    # --- mode 3: X-OP graph ----------------------------------------------
    nodes = build_graph(vcf)
    order = topological_order(nodes)
    inputs = external_inputs(nodes)
    outs = sinks(order)
    if not outs:
        raise JITError("graph has no sink node (cycle or dangling outputs)")

    inner, used_consts = _emit_graph_kernel(order, inputs, outs)

    args = [f"const float* in__{_sanitize(i)}" for i in inputs]
    args += [f"float* out__{_sanitize(s.uid)}" for s in outs]
    args.append("int n")
    const_lines = [COMMON_CONSTS[c] for c in COMMON_CONSTS if c in used_consts]
    consts = "\n    ".join(const_lines)
    src = (
        "#include <immintrin.h>\n#include <math.h>\n#include <stdint.h>\n\n"
        f"void execute_kernel({', '.join(args)}) {{\n"
        f"    {consts}\n"
        f"{inner}\n"
        "}\n"
    )
    return src, inputs, [s.uid for s in outs], []


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------

@dataclass
class KernelMeta:
    inputs: List[str]
    outputs: List[str]
    params: List[str]
    source: str
    sha256: str


class VcfJit:
    """Compile once, call many times."""

    def __init__(self, source: str | VCardFile | None = None, *,
                 path: Optional[str] = None, text: Optional[str] = None,
                 cc: str = "cc", opt: str = "-O3") -> None:
        if text is None and path is not None:
            text = Path(path).read_text(encoding="utf-8")
        if text is None and source is not None:
            text = dump_vcf(source.cards) if isinstance(source, VCardFile) else str(source)
        if text is None:
            raise ValueError("provide text= or path=")

        try:
            self.src, self.meta_inputs, self.meta_outputs, self.meta_params = _build_source(
                parse_vcf(text))
        except KeyError as exc:
            raise JITError(f"unknown op in graph: {exc}") from None
        except GraphError as exc:
            raise JITError(str(exc)) from None

        digest = hashlib.sha256(self.src.encode()).hexdigest()[:16]

        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vcard_cpu"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.lib_path = cache_dir / f"vcf_{digest}.so"

        if not self.lib_path.exists():
            with tempfile.TemporaryDirectory() as td:
                c_file = Path(td) / "kernel.c"
                so_file = Path(td) / "kernel.so"
                c_file.write_text(self.src, encoding="utf-8")
                proc = subprocess.run(
                    [cc, opt, "-mavx2", "-mfma", "-shared", "-fPIC",
                     "-o", str(so_file), str(c_file)],
                    capture_output=True, text=True)
                if proc.returncode != 0:
                    raise JITError(f"gcc failed:\n{proc.stderr}\n--- generated C ---\n{self.src}")
                os.replace(so_file, self.lib_path)

        self._lib = ctypes.CDLL(str(self.lib_path))
        self._fn = self._lib.execute_kernel
        ptr = ctypes.POINTER(ctypes.c_float)
        self._fn.argtypes = [ptr] * len(self.meta_inputs) + \
                            [ptr] * len(self.meta_outputs) + \
                            [ctypes.c_int] + [ctypes.c_float] * len(self.meta_params)
        self._fn.restype = None

    # ------------------------------------------------------------------
    @property
    def meta(self) -> KernelMeta:
        return KernelMeta(self.meta_inputs, self.meta_outputs, self.meta_params,
                          self.src, hashlib.sha256(self.src.encode()).hexdigest())

    def run(self, inputs: Dict[str, np.ndarray],
            outputs: Optional[Sequence[str]] = None,
            params: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Execute over NumPy float32 buffers.

        Output length is inferred from the first input; each requested output
        is allocated and returned as a dict.
        """
        params = params or {}
        missing = [i for i in self.meta_inputs if i not in inputs]
        if missing:
            raise ValueError(f"missing inputs: {missing}")

        arrays = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in inputs.items()}
        n = len(next(iter(arrays.values())))
        for k, v in arrays.items():
            if len(v) != n:
                raise ValueError(f"input length mismatch: {k} has {len(v)}, expected {n}")

        out_names = list(outputs) if outputs is not None else self.meta_outputs
        outs = {name: np.zeros(n, dtype=np.float32) for name in out_names}

        ptr = ctypes.POINTER(ctypes.c_float)
        args = [arrays[i].ctypes.data_as(ptr) for i in self.meta_inputs]

        # graph-mode outputs are named by sink uid; accept positional fallback
        out_args = []
        for name in out_names:
            key = name if name in outs else self._match_output(name)
            out_args.append(outs[key].ctypes.data_as(ptr))
        args += out_args

        args.append(ctypes.c_int(n))
        for p in self.meta_params:
            if p not in params:
                raise ValueError(f"missing kernel param: {p}")
            args.append(ctypes.c_float(float(params[p])))

        self._fn(*args)
        return {name: outs[self._match_output(name)] for name in out_names}

    def _match_output(self, name: str) -> str:
        if name in self.meta_outputs:
            return name
        for mo in self.meta_outputs:
            if _sanitize(mo) == _sanitize(name):
                return mo
        raise KeyError(f"unknown output {name!r}; kernel provides {self.meta_outputs}")


def compile_kernel(text: str) -> VcfJit:
    """One-call helper: vCard text -> callable kernel."""
    return VcfJit(text=text)


def show_plan(path_or_text: str) -> str:
    """Human-readable compilation plan (no compilation)."""
    text = (Path(path_or_text).read_text(encoding="utf-8")
            if os.path.exists(path_or_text) else path_or_text)
    vcf = parse_vcf(text)
    lines = [c.summary() for c in vcf.cards]
    try:
        src, ins, outs, params = _build_source(vcf)
        lines.append(f"inputs={ins} outputs={outs} params={params}")
        lines.append("--- generated C ---")
        lines.append(src)
    except (JITError, VCardError) as exc:
        lines.append(f"(no codegen: {exc})")
    return "\n".join(lines)
