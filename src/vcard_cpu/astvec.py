"""Python expression -> fused AVX2 kernel with constant hoisting.

    AstToAvx2("a * b + c", ["a", "b", "c"]).generate_c()

The expression is parsed with :mod:`ast`, validated against a whitelist and
lowered through the same op registry as the vCard graph compiler, so both
paths share SIMD semantics. Numeric literals are hoisted out of the hot loop
into ``__m256`` constants (constant folding across the whole expression).
"""

from __future__ import annotations

import ast
from typing import Dict, List

from .ops import COMMON_CONSTS, REGISTRY

_BINOPS = {ast.Add: "ADD", ast.Sub: "SUB", ast.Mult: "MUL", ast.Div: "DIV"}
_FUNCS = {"abs": "ABS", "sqrt": "SQRT", "minimum": "MIN", "maximum": "MAX"}


class ExprError(ValueError):
    """Unsupported or unsafe expression."""


class AstToAvx2:
    def __init__(self, expr_str: str, input_names: List[str]) -> None:
        self.expr_str = expr_str
        self.input_names = list(input_names)
        self.constants: Dict[float, str] = {}
        self._tmp = 0
        self._consts_used: List[str] = []

        try:
            self.tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as exc:
            raise ExprError(f"bad expression: {exc}") from exc
        self._validate(self.tree.body)

    # ------------------------------------------------------------------
    def _get_const(self, val: float) -> str:
        if val not in self.constants:
            self.constants[val] = f"const_{len(self.constants) + 1}"
        var = self.constants[val]
        if var not in self._consts_used:
            self._consts_used.append(var)
        return var

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            return self._validate(node.body)
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ExprError(f"unsupported binary op: {ast.dump(node.op)}")
            return (self._validate(node.left), self._validate(node.right))
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ExprError(f"unsupported unary op: {ast.dump(node.op)}")
            return self._validate(node.operand)
        if isinstance(node, ast.Name):
            if node.id not in self.input_names:
                raise ExprError(f"unknown input {node.id!r}; declared: {self.input_names}")
            return None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExprError(f"unsupported constant: {node.value!r}")
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ExprError("only abs/sqrt/minimum/maximum calls are allowed")
            if not node.keywords:
                for a in node.args:
                    self._validate(a)
                return None
        raise ExprError(f"unsupported syntax: {ast.dump(node)}")

    # ------------------------------------------------------------------
    def _lower(self, node: ast.AST, vec: bool) -> str:
        t = f"v{self._tmp}" if vec else f"s{self._tmp}"
        self._tmp += 1
        ty = "__m256" if vec else "float"

        if isinstance(node, ast.Name):
            return (f"v_in__{node.id}" if vec else f"s_in__{node.id}")
        if isinstance(node, ast.Constant):
            val = float(node.value)
            if vec:
                return self._get_const(val)
            return repr(val)

        if isinstance(node, ast.UnaryOp):
            operand = self._lower(node.operand, vec)
            op = "NEG" if isinstance(node.op, ast.USub) else None
            if op is None:  # UAdd is identity
                return operand
            spec = REGISTRY[op]
            self._consts_used.extend(spec[3])
            gen = f"{ty} {t} = {spec[1](operand) if vec else spec[2](operand)};"
            self._stmts.append(gen)
            return t

        if isinstance(node, ast.BinOp):
            opname = _BINOPS[type(node.op)]
            a = self._lower(node.left, vec)
            b = self._lower(node.right, vec)
            spec = REGISTRY[opname]
            self._consts_used.extend(spec[3])
            gen = f"{ty} {t} = {spec[1](a, b) if vec else spec[2](a, b)};"
            self._stmts.append(gen)
            return t

        if isinstance(node, ast.Call):
            opname = _FUNCS[node.func.id]
            args = [self._lower(a, vec) for a in node.args]
            spec = REGISTRY[opname]
            self._consts_used.extend(spec[3])
            gen = f"{ty} {t} = {spec[1](*args) if vec else spec[2](*args)};"
            self._stmts.append(gen)
            return t

        raise ExprError(f"lowering failed on {ast.dump(node)}")  # pragma: no cover

    # ------------------------------------------------------------------
    def generate_c(self) -> str:
        self._stmts: List[str] = []
        self._tmp = 0
        self._consts_used = []

        result_v = self._lower(self.tree.body, vec=True)
        self._tmp = 0
        self._stmts_sep = len(self._stmts)
        result_s = self._lower(self.tree.body, vec=False)

        vec_body = self._stmts[:self._stmts_sep]
        scalar_body = self._stmts[self._stmts_sep:]

        const_decls = []
        for val, var in self.constants.items():
            const_decls.append(f"    __m256 {var} = _mm256_set1_ps({val!r}f);")
        for cname in dict.fromkeys(self._consts_used):  # ordered dedupe
            if cname in COMMON_CONSTS:  # skip hoisted numeric consts (const_N)
                const_decls.append(f"    {COMMON_CONSTS[cname]}")
        if "zero_const" not in self._consts_used:
            const_decls.append(f"    {COMMON_CONSTS['zero_const']}")

        sig_args = [f"const float* in__{n}" for n in self.input_names] + ["float* out", "int n"]
        lines = [
            "#include <immintrin.h>",
            "#include <math.h>",
            "#include <stdint.h>",
            "",
            f"void execute_kernel({', '.join(sig_args)}) {{",
            *const_decls,
            "    int i = 0;",
            "    for (; i <= n - 8; i += 8) {",
        ]
        for n in self.input_names:
            lines.append(f"        __m256 v_in__{n} = _mm256_loadu_ps(&in__{n}[i]);")
        lines += [f"        {s}" for s in vec_body]
        lines.append(f"        _mm256_storeu_ps(&out[i], {result_v});")
        lines.append("    }")
        lines.append("    for (; i < n; i++) {")
        for n in self.input_names:
            lines.append(f"        float s_in__{n} = in__{n}[i];")
        lines += [f"        {s}" for s in scalar_body]
        lines.append(f"        out[i] = {result_s};")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)
