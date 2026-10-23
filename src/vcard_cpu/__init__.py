"""vcard-cpu — vCard 4.0 as an executable compute-kernel container.

A .vcf file stops being a contact card and becomes a *program*:

    BEGIN:VCARD
    UID:node-mult
    X-OP:MUL
    RELATED;TYPE=input:in_x,in_y
    END:VCARD

Cards with ``X-OP`` are SIMD operation nodes wired into a DAG through
``RELATED`` properties. Cards with ``X-KERNEL-BODY`` carry raw AVX2 inner
loops. The JIT translates the whole graph into one fused C kernel, compiles
it with gcc and calls it over NumPy buffers.
"""

from .vcard import VCard, VCardFile, parse_vcf, dump_vcf, VCardError
from .graph import OpNode, build_graph, topological_order, GraphError
from .jit import VcfJit, compile_kernel
from .astvec import AstToAvx2

__version__ = "0.1.0"

__all__ = [
    "VCard", "VCardFile", "parse_vcf", "dump_vcf", "VCardError",
    "OpNode", "build_graph", "topological_order", "GraphError",
    "VcfJit", "compile_kernel", "AstToAvx2", "__version__",
]
