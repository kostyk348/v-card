"""AVX2 / scalar emitters for the primitive op set.

Every op knows how to lower itself twice:
  * AVX2  — eight floats per instruction (the fast path)
  * scalar C — the tail loop (last n % 8 elements)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

# constants every kernel prologue declares
COMMON_CONSTS = {
    "sign_mask": "__m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));",
    "eps_const": "__m256 eps_const = _mm256_set1_ps(1e-5f);",
    "one_const": "__m256 one_const = _mm256_set1_ps(1.0f);",
    "zero_const": "__m256 zero_const = _mm256_setzero_ps();",
}

# op -> (arity, avx2 emitter, scalar emitter, required common constants)
OpSpec = Tuple[int, Callable[..., str], Callable[..., str], List[str]]

REGISTRY: Dict[str, OpSpec] = {
    "ADD": (2,
            lambda a, b: f"_mm256_add_ps({a}, {b})",
            lambda a, b: f"({a} + {b})",
            []),
    "SUB": (2,
            lambda a, b: f"_mm256_sub_ps({a}, {b})",
            lambda a, b: f"({a} - {b})",
            []),
    "MUL": (2,
            lambda a, b: f"_mm256_mul_ps({a}, {b})",
            lambda a, b: f"({a} * {b})",
            []),
    "DIV": (2,
            lambda a, b: f"_mm256_div_ps({a}, {b})",
            lambda a, b: f"({a} / {b})",
            []),
    "MAX": (2,
            lambda a, b: f"_mm256_max_ps({a}, {b})",
            lambda a, b: f"({a} > {b} ? {a} : {b})",
            []),
    "MIN": (2,
            lambda a, b: f"_mm256_min_ps({a}, {b})",
            lambda a, b: f"({a} < {b} ? {a} : {b})",
            []),
    "ABS": (1,
            lambda a: f"_mm256_and_ps({a}, sign_mask)",
            lambda a: f"fabsf({a})",
            ["sign_mask"]),
    "NEG": (1,
            lambda a: f"_mm256_sub_ps(zero_const, {a})",
            lambda a: f"(-{a})",
            ["zero_const"]),
    "SQRT": (1,
             lambda a: f"_mm256_sqrt_ps(_mm256_max_ps({a}, zero_const))",
             lambda a: f"sqrtf({a} < 0.0f ? 0.0f : {a})",
             ["zero_const"]),
    "RSQRT": (1,
              lambda a: f"_mm256_rsqrt_ps(_mm256_add_ps({a}, eps_const))",
              lambda a: f"(1.0f / sqrtf({a} + 1e-5f))",
              ["eps_const"]),
    "FMADD": (3,
              lambda a, b, c: f"_mm256_fmadd_ps({a}, {b}, {c})",
              lambda a, b, c: f"({a} * {b} + {c})",
              []),
    "FMSUB": (3,
              lambda a, b, c: f"_mm256_fmsub_ps({a}, {b}, {c})",
              lambda a, b, c: f"({a} * {b} - {c})",
              []),
    "RELU": (1,
             lambda a: f"_mm256_max_ps(zero_const, {a})",
             lambda a: f"({a} > 0.0f ? {a} : 0.0f)",
             ["zero_const"]),
}


def arity(op: str) -> int:
    try:
        return REGISTRY[op.upper()][0]
    except KeyError:
        raise ValueError(f"unknown op: {op!r} (known: {', '.join(sorted(REGISTRY))})") from None
