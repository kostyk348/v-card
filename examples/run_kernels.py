#!/usr/bin/env python3
"""Compile and run every bundled .vcf kernel against NumPy references."""

from pathlib import Path

import numpy as np

from vcard_cpu.jit import VcfJit

kernels = Path(__file__).resolve().parent.parent / "src" / "vcard_cpu" / "kernels"
rng = np.random.default_rng(0)
n = 100_003

# --- industrial: Pt100 ADC -> temperature -----------------------------
jit = VcfJit(path=str(kernels / "pt100.vcf"))
adc = rng.uniform(0, 65535, n).astype(np.float32)
out = jit.run({"adc": adc})["out"]
ref = (adc * 0.005) * (1.0 + adc * 1e-6) - 50.0
print(f"pt100      : max_err={np.abs(out - ref).max():.2e} C")

# --- safety: branchless Schmitt trigger -------------------------------
jit = VcfJit(path=str(kernels / "hysteresis.vcf"))
sig = rng.uniform(-1, 2, n).astype(np.float32)
prev = (rng.random(n) < 0.5).astype(np.float32)
out = jit.run({"signal": sig, "prev_state": prev},
              params={"high_th": 1.5, "low_th": 0.5})["out"]
ref = ((sig > 1.5) | ((sig > 0.5) & (prev > 0.5))).astype(np.float32)
print(f"hysteresis : exact match = {np.array_equal(out, ref)}")

# --- fused math pipeline (handwritten AVX2 in a vCard) -----------------
jit = VcfJit(path=str(kernels / "fused.vcf"))
a, b, c, d = (rng.uniform(-5, 5, n).astype(np.float32) for _ in range(4))
out = jit.run({"a": a, "b": b, "c": c, "d": d})["out"]
print(f"fused      : out[:3] = {out[:3]}")
