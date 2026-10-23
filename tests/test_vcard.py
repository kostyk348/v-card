import textwrap

import pytest

from vcard_cpu.vcard import VCardError, parse_vcf, dump_vcf


def test_single_card():
    vcf = parse_vcf(textwrap.dedent("""
    BEGIN:VCARD
    VERSION:4.0
    UID:node-1
    FN:Test node
    X-OP:MUL
    END:VCARD
    """))
    assert len(vcf.cards) == 1
    card = vcf.cards[0]
    assert card.uid == "node-1"
    assert card.op == "MUL"
    assert card.fn == "Test node"


def test_line_unfolding():
    vcf = parse_vcf("BEGIN:VCARD\nFN:long value that\n  continues here\nEND:VCARD")
    assert vcf.cards[0].fn == "long value that continues here"


def test_related_parsing():
    vcf = parse_vcf(textwrap.dedent("""
    BEGIN:VCARD
    UID:n1
    X-OP:ADD
    RELATED;TYPE=input:in_x,in_y
    RELATED;TYPE=input:urn:uuid:n0
    RELATED;TYPE=output:out1
    END:VCARD
    """))
    card = vcf.cards[0]
    assert card.inputs() == ["in_x", "in_y", "n0"]
    assert card.outputs() == ["out1"]


def test_multi_card_and_errors():
    text = "BEGIN:VCARD\nUID:a\nEND:VCARD\nBEGIN:VCARD\nUID:b\nEND:VCARD\n"
    assert len(parse_vcf(text).cards) == 2
    with pytest.raises(VCardError):
        parse_vcf("UID:no-begin-here")
    with pytest.raises(VCardError):
        parse_vcf("BEGIN:VCARD\nUID:unterminated")
    with pytest.raises(VCardError):
        parse_vcf("BEGIN:VCARD\nBEGIN:VCARD\nEND:VCARD")


def test_params_and_categories():
    vcf = parse_vcf("BEGIN:VCARD\nUID:p\nCATEGORIES:SIMD,AVX2\nX-PARAMS:high_th:float\nEND:VCARD")
    card = vcf.cards[0]
    assert card.categories == ["SIMD", "AVX2"]
    assert card.get("X-PARAMS") == "high_th:float"


def test_roundtrip():
    text = "BEGIN:VCARD\nVERSION:4.0\nUID:r1\nFN:Round\nEND:VCARD\n"
    again = dump_vcf(parse_vcf(text).cards)
    assert parse_vcf(again).cards[0].uid == "r1"
