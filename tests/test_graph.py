import textwrap

import pytest

from vcard_cpu.graph import GraphError, build_graph, external_inputs, sinks, topological_order
from vcard_cpu.vcard import parse_vcf


def graph_text(body: str) -> str:
    return textwrap.dedent(body)


SAMPLE = """
    BEGIN:VCARD
    UID:a
    X-OP:MUL
    RELATED;TYPE=input:in_x,in_y
    END:VCARD

    BEGIN:VCARD
    UID:b
    X-OP:ABS
    RELATED;TYPE=input:a
    END:VCARD
"""

CYCLE = """
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
"""


def test_topological_order():
    nodes = build_graph(parse_vcf(graph_text(SAMPLE)))
    order = [n.uid for n in topological_order(nodes)]
    assert order.index("a") < order.index("b")
    assert external_inputs(nodes) == ["in_x", "in_y"]
    assert [n.uid for n in sinks(nodes)] == ["b"]


def test_cycle_detected():
    nodes = build_graph(parse_vcf(graph_text(CYCLE)))
    with pytest.raises(GraphError, match="cycle"):
        topological_order(nodes)


def test_no_ops_raises():
    from vcard_cpu.graph import build_graph as bg
    with pytest.raises(GraphError):
        bg(parse_vcf("BEGIN:VCARD\nUID:x\nFN:not an op\nEND:VCARD"))
