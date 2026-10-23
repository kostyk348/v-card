import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest

from vcard_cpu.jit import JITError, VcfJit, show_plan
from vcard_cpu.vcard import parse_vcf

Kernels = Path(__file__).resolve().parent.parent / "src" / "vcard_cpu" / "kernels"

pytestmark = pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None,
                                reason="no C compiler")

# ----------------------------------------------------------------------
# X-OP graph compilation
# ----------------------------------------------------------------------

GRAPH = """
BEGIN:VCARD
UID:node-mult
X-OP:MUL
RELATED;TYPE=input:in_x,in_y
END:VCARD

BEGIN:VCARD
UID:node-sub
X-OP:SUB
RELATED;TYPE=input:node-mult,in_z
END:VCARD

BEGIN:VCARD
UID:node-abs
X-OP:ABS
RELATED;TYPE=input:node-sub
END:VCARD

BEGIN:VCARD
UID:node-clamp
X-OP:MIN
RELATED;TYPE=input:node-abs,in_limit
END:VCARD
"""


def test_graph_kernel_matches_numpy():
    jit = VcfJit(text=GRAPH)
    assert jit.meta_inputs == ["in_x", "in_y", "in_z", "in_limit"]
    rng = np.random.default_rng(0)
    n = 1_003  # not a multiple of 8: exercises the scalar tail too
    arrays = {k: rng.uniform(-5, 5, n).astype(np.float32)
              for k in ["in_x", "in_y", "in_z", "in_limit"]}
    out = jit.run(arrays)["node-clamp"]

    ref = np.abs(arrays["in_x"] * arrays["in_y"] - arrays["in_z"])
    ref = np.minimum(ref, arrays["in_limit"]).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_input_order_independent():
    jit = VcfJit(text=GRAPH)
    rng = np.random.default_rng(1)
    n = 64
    arrays = {k: rng.uniform(-3, 3, n).astype(np.float32)
              for k in ["in_limit", "in_z", "in_y", "in_x"]}  # shuffled on purpose
    out = jit.run(arrays)["node-clamp"]
    ref = np.minimum(np.abs(arrays["in_x"] * arrays["in_y"] - arrays["in_z"]),
                     arrays["in_limit"]).astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------
# bundled industrial kernels
# ----------------------------------------------------------------------

def test_pt100_kernel():
    jit = VcfJit(path=str(Kernels / "pt100.vcf"))
    assert jit.meta_inputs == ["adc"]
    n = 256
    adc = np.random.default_rng(2).uniform(0, 65535, n).astype(np.float32)
    out = jit.run({"adc": adc})["out"]
    ref = (adc * 0.005) * (1.0 + adc * 0.000001) - 50.0
    np.testing.assert_allclose(out[:248], ref[:248], rtol=2e-5, atol=1e-6)


def test_hysteresis_kernel_branchless():
    jit = VcfJit(path=str(Kernels / "hysteresis.vcf"))
    assert jit.meta_params == ["high_th", "low_th"]
    n = 128
    rng = np.random.default_rng(3)
    sig = rng.uniform(-1, 2, n).astype(np.float32)
    prev = (rng.random(n) < 0.5).astype(np.float32)
    out = jit.run({"signal": sig, "prev_state": prev},
                  params={"high_th": 1.5, "low_th": 0.5})["out"]
    ref = ((sig > 1.5) | ((sig > 0.5) & (prev > 0.5))).astype(np.float32)
    np.testing.assert_allclose(out, ref)


def test_fused_micro_op_kernel():
    jit = VcfJit(path=str(Kernels / "fused.vcf"))
    n = 253  # not a multiple of 8 -> vector body + scalar tail
    rng = np.random.default_rng(4)
    a, b, c, d = (rng.uniform(-5, 5, n).astype(np.float32) for _ in range(4))
    out = jit.run({"a": a, "b": b, "c": c, "d": d})["out"]
    num = np.maximum(a * b + c, 0.0)
    den = np.abs(d) + 1e-5
    ref = (num / np.sqrt(den)).astype(np.float32)
    # vector path uses rsqrt (approx) — loose tolerance on the vector body
    np.testing.assert_allclose(out[:248], ref[:248], rtol=5e-3)
    np.testing.assert_allclose(out[248:], ref[248:], rtol=1e-5)


# ----------------------------------------------------------------------
# failure modes
# ----------------------------------------------------------------------

def test_bad_op_raises():
    with pytest.raises((ValueError, JITError)):
        VcfJit(text="BEGIN:VCARD\nUID:x\nX-OP:TELEPORT\nRELATED;TYPE=input:a\nEND:VCARD")


def test_cycle_raises():
    text = textwrap.dedent("""
    BEGIN:VCARD
    UID:p
    X-OP:ADD
    RELATED;TYPE=input:q,in_c
    END:VCARD
    BEGIN:VCARD
    UID:q
    X-OP:NEG
    RELATED;TYPE=input:p
    END:VCARD
    """)
    with pytest.raises(JITError):
        VcfJit(text=text)


def test_show_plan_mentions_cards():
    plan = show_plan(str(Kernels / "graph.vcf"))
    assert "node-mult" in plan and "generated C" in plan
