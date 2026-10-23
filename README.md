<div align="center">

```
 ┌─────────────────────────────────────────────┐
 │  BEGIN:VCARD                                │
 │  UID:hello                                  │
 │  FN:vcard-cpu                               │
 │  X-OP:MUL                                   │
 │  RELATED;TYPE=input:yourGPU,yourCPU         │
 │  END:VCARD                                  │
 └─────────────────────────────────────────────┘
```

# vcard-cpu

**vCard 4.0 as an executable compute-kernel container.**
Yes, *that* vCard — the contacts file format. Your SIMD kernels deserve
better than ad-hoc YAML.

[![CI](https://github.com/example/vcard-cpu/actions/workflows/ci.yml/badge.svg)](https://github.com/example/vcard-cpu/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-green)]()

*README на русском: [README.ru.md](README.ru.md)*

</div>

---

## TL;DR

A `.vcf` file stops being a contact card and becomes a **program**.
Cards with `X-OP` are SIMD operation nodes wired into a DAG via `RELATED`
properties. The JIT translates the whole graph into **one fused AVX2 loop**,
compiles it with gcc and calls it over NumPy buffers.

```console
$ vcard-cpu show kernels/graph.vcf
node-mult: op=MUL in=['in_x', 'in_y']
node-sub: op=SUB in=['node-mult', 'in_z']
node-abs: op=ABS in=['node-sub']
node-clamp: op=MIN in=['node-abs', 'in_limit']
inputs=['in_x', 'in_y', 'in_z', 'in_limit'] outputs=['node-clamp'] params=[]
--- generated C ---
#include <immintrin.h>
...
void execute_kernel(const float* in__in_x, ..., float* out__node__clamp, int n) {
    for (; i <= n - 8; i += 8) {          // 8 floats per instruction
        __m256 t0 = _mm256_mul_ps(v_in__in_x, v_in__in_y);
        __m256 t1 = _mm256_sub_ps(t0, v_in__in_z);
        __m256 t2 = _mm256_and_ps(t1, sign_mask);
        __m256 t3 = _mm256_min_ps(t2, v_in__in_limit);
        _mm256_storeu_ps(&out__node__clamp[i], t3);
    }
    ...
}
```

Four vCards → one branchless, register-to-register, fully vectorized
kernel. No intermediate buffers. No Python in the hot loop.

## Why vCard?

Half a joke, half serious:

1. **It already has a graph model.** `RELATED;TYPE=input:` is a typed edge,
   `UID` is a node id, `FN` is documentation. RFC 6350 did the boring work.
2. **It has parameters.** `X-PARAMS:high_th:float` becomes a real C function
   argument — your kernel's "phone number" is a Schmitt threshold.
3. **Human-readable containers beat opaque binaries** for distributing
   industrial recipes. A Pt100 calibration curve deserves version control
   in a format an engineer can read on a train.
4. **Categories as deployment metadata**: `CATEGORIES:Industrial,Sensors,SIMD`
   is your kernel-catalog index, for free.

The serious core underneath: **vCard is a demonstration substrate for
"declarative SIMD DSL → fused C"**, which is the same trick Mojo, Halide
and TVM play — with a format so silly it makes the pipeline obvious.

## Three ways to say "compute"

### 1. Graph of op cards (auto-fused)

```vcf
BEGIN:VCARD
UID:node-mult
X-OP:MUL
RELATED;TYPE=input:in_x,in_y
END:VCARD
...
```

Ops available: `ADD SUB MUL DIV MAX MIN ABS NEG SQRT RSQRT FMADD FMSUB RELU`.
The compiler topologically sorts the DAG, emits one fused AVX2 loop with a
scalar tail, compiles, and hands you a Python callable.

### 2. Raw kernel body (bring your own AVX2)

```vcf
BEGIN:VCARD
UID:isa-pt100-adc-calibrate
FN:vMODBUS_PT100_CONVERT
CATEGORIES:Industrial,Sensors,SIMD
RELATED;TYPE=input:adc
RELATED;TYPE=output:out
X-KERNEL-BODY:
// T = (ADC * 0.005) * (1.0 + ADC * 0.000001) - 50.0
__m256 v_scale = _mm256_set1_ps(0.005f);
...
END:VCARD
```

Everything between `X-KERNEL-BODY:` and `END:VCARD` is verbatim C dropped
into a generated function with a proper signature.

### 3. Python expression → AVX2

```console
$ vcard-cpu expr "a * b + c" -i a -i b -i c --n 500000
n=500000 elapsed=0.68 ms (~5.92 GOPS naive)
```

`vcard-cpu expr` parses the expression with Python's `ast` module,
validates it against a whitelist, hoists numeric constants out of the hot
loop into `__m256` constants, and lowers through the same op registry as
the graph compiler. Constant folding across the whole expression, for free.

## Benchmarks (honest numbers)

| kernel                          | n         | time     | notes                        |
|---------------------------------|-----------|----------|------------------------------|
| Pt100 poly (FMADD×2, FMSUB)     | 2 000 000 | 1.59 ms  | max error vs numpy: 6·10⁻⁵ °C |
| fused relu·rsqrt pipeline       | 2 M       | ~1.9 ms  | `rsqrt` is the approximate AVX2 flavor |
| expr `a*b+c` via JIT            | 500 000   | 0.68 ms  | includes ctypes call overhead |

Numbers from a laptop CPU (`cc -O3 -mavx2 -mfma`). The point is not the
speed — it's that **a vCard became a compiled SIMD kernel in under 2 ms**.

## Install

```bash
git clone https://github.com/example/vcard-cpu
cd vcard-cpu
pip install -e .
vcard-cpu --help
```

Requirements: Linux/macOS x86_64 with AVX2/FMA, gcc/clang at runtime
(JIT compiles with `cc -O3 -mavx2 -mfma`), Python 3.9+, numpy.
Compiled kernels are cached in `~/.cache/vcard_cpu/` keyed by source hash.

## CLI reference

```
vcard-cpu show  <file.vcf>              cards + generated C plan (no compile)
vcard-cpu run   <file.vcf> [--n N]      compile + execute with random inputs
                                        [--param name=value] for X-PARAMS
vcard-cpu expr  "a*b+c" -i a -i b -i c  one-off expression kernel
                [--n N] [--emit-c]
```

## Python API

```python
import numpy as np
from vcard_cpu import VcfJit

jit = VcfJit(path="pt100.vcf")
print(jit.meta_inputs)    # ['adc']
out = jit.run({"adc": adc_array})["out"]
```

`VcfJit.run` validates input presence and lengths, allocates outputs,
passes scalar `params`, and returns plain NumPy arrays.

## Repository layout

```
vcard-cpu/
├── src/vcard_cpu/
│   ├── vcard.py      RFC 6350 parser (folded lines, params, block properties)
│   ├── graph.py      DAG extraction + Kahn topological sort (+ cycle detection)
│   ├── ops.py        op registry: AVX2 + scalar emitters per op
│   ├── jit.py        graph → fused C → gcc → ctypes; kernel-body mode
│   ├── astvec.py     Python expression → AVX2 with constant hoisting
│   └── kernels/      bundled .vcf: pt100, hysteresis, fused, graph
├── tests/            24 pytest cases: parser, graph, JIT numerics vs numpy
├── examples/run_kernels.py
├── notebooks/        the original research notebook (Colab)
└── .github/workflows/ci.yml
```

## Ideas welcome

- `RELATED` graphs across *multiple files* (a vCard "linker").
- AVX-512 / NEON backends behind the same registry.
- Fixed-point industrial ops with per-card rounding policy.
- A proper `vcard-cpu link` that merges card libraries with namespace rules.

## Related work

- [Halide](https://halide-lang.org), [TVM](https://tvm.apache.org) — the
  serious versions of "describe compute, compile schedule".
- ONNX — graph interchange for ML; vcard-cpu is graph interchange for
  *industrial* kernels with a sense of humor.

## License

Apache-2.0 — see [LICENSE](LICENSE).
