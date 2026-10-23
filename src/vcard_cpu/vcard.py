"""RFC 6350 (vCard 4.0) reader/writer, tuned for compute payloads.

We deliberately keep it small: folded lines, grouped properties, parameters
and BEGIN/END card framing. Everything else (photos, agents, geo...) passes
through untouched as opaque property values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class VCardError(ValueError):
    """Malformed vCard payload."""


@dataclass
class Property:
    name: str
    value: str
    params: Dict[str, List[str]] = field(default_factory=dict)

    def param(self, key: str, default: str = "") -> str:
        vals = self.params.get(key.upper())
        return vals[0] if vals else default

    def __str__(self) -> str:
        parts = [self.name]
        for k, vals in self.params.items():
            parts.append(f";{k}={','.join(vals)}")
        return "".join(parts) + ":" + self.value


@dataclass
class VCard:
    properties: List[Property] = field(default_factory=list)

    # --- convenience accessors ---------------------------------------
    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        for p in self.properties:
            if p.name == name.upper():
                return p.value
        return default

    def get_all(self, name: str) -> List[Property]:
        return [p for p in self.properties if p.name == name.upper()]

    @property
    def uid(self) -> str:
        return self.get("UID", "")

    @property
    def fn(self) -> str:
        return self.get("FN", "")

    @property
    def op(self) -> Optional[str]:
        v = self.get("X-OP")
        return v.strip().upper() if v else None

    @property
    def body(self) -> Optional[str]:
        return self.get("X-KERNEL-BODY")

    @property
    def micro_op_c(self) -> Optional[str]:
        return self.get("X-MICRO-OP-C")

    @property
    def categories(self) -> List[str]:
        v = self.get("CATEGORIES", "")
        return [c.strip() for c in v.split(",") if c.strip()]

    def related(self, kind: str) -> List[str]:
        """RELATED;TYPE=input:... entries, split on commas."""
        out: List[str] = []
        for p in self.get_all("RELATED"):
            if p.param("TYPE").lower() == kind:
                out += [x.strip() for x in p.value.split(",") if x.strip()]
        return out

    def inputs(self) -> List[str]:
        return [self._strip_urn(x) for x in self.related("input")]

    def outputs(self) -> List[str]:
        return [self._strip_urn(x) for x in self.related("output")]

    @staticmethod
    def _strip_urn(name: str) -> str:
        return name.replace("urn:uuid:", "")

    def summary(self) -> str:
        head = f"{self.fn or self.uid or '<card>'}"
        bits = []
        if self.op:
            bits.append(f"op={self.op}")
        if self.body:
            bits.append("kernel-body")
        if self.micro_op_c:
            bits.append("micro-op-c")
        if self.inputs():
            bits.append(f"in={self.inputs()}")
        if self.outputs():
            bits.append(f"out={self.outputs()}")
        return f"{head}: " + " ".join(bits) if bits else head


@dataclass
class VCardFile:
    cards: List[VCard] = field(default_factory=list)

    def filter(self, **kw) -> List[VCard]:  # pragma: no cover - sugar
        return [c for c in self.cards
                if all((c.get(k.upper()) == v) for k, v in kw.items())]


def _unfold(text: str) -> List[str]:
    """RFC 6350 §3.2: CRLF followed by SPACE/HTAB is a continuation."""
    lines: List[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        elif raw:
            lines.append(raw)
    return lines


def _parse_property(line: str) -> Property:
    if ":" not in line:
        raise VCardError(f"property without ':' separator: {line!r}")
    head, value = line.split(":", 1)
    pieces = _split_unquoted(head, ";")
    name = pieces[0].upper()
    params: Dict[str, List[str]] = {}
    for p in pieces[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params.setdefault(k.upper(), []).extend(v.split(","))
        else:
            params.setdefault("TYPE", []).append(p)
    return Property(name, value, params)


def _split_unquoted(s: str, sep: str) -> List[str]:
    out, buf, quote = [], [], False
    for ch in s:
        if ch == '"':
            quote = not quote
            buf.append(ch)
        elif ch == sep and not quote:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [x for x in out if x != ""]


def parse_vcf(text: str) -> VCardFile:
    """Parse a multi-card vCard file.

    ``X-KERNEL-BODY`` / ``X-MICRO-OP-C`` are *block properties*: everything
    between the property line and ``END:VCARD`` is captured verbatim as the
    property value (this is what makes raw C payloads possible).
    """
    raw_block_props = {"X-KERNEL-BODY", "X-MICRO-OP-C"}
    cards: List[VCard] = []
    current: VCard | None = None

    lines = _unfold(text)
    i = 0
    while i < len(lines):
        line = lines[i]
        prop = _parse_property(line)

        if prop.name == "BEGIN":
            if current is not None:
                raise VCardError("nested BEGIN:VCARD")
            current = VCard()
        elif prop.name == "END":
            if current is None:
                raise VCardError("END:VCARD without BEGIN")
            cards.append(current)
            current = None
        else:
            if current is None:
                raise VCardError(f"property outside card: {line!r}")
            current.properties.append(prop)

            if prop.name in raw_block_props:
                # capture everything up to END:VCARD verbatim
                j = i + 1
                buf: List[str] = []
                while j < len(lines) and not lines[j].upper().startswith("END:VCARD"):
                    buf.append(lines[j])
                    j += 1
                prop.value = "\n".join(buf)
                i = j - 1  # loop will land on END:VCARD next
        i += 1

    if current is not None:
        raise VCardError("unterminated card (missing END:VCARD)")
    return VCardFile(cards=cards)


def parse_vcf_file(path: str) -> VCardFile:
    with open(path, encoding="utf-8") as fh:
        return parse_vcf(fh.read())


def dump_vcf(cards: List[VCard]) -> str:
    """Serialize cards back to text (round-trip safe for parsed files)."""
    out: List[str] = []
    for card in cards:
        out.append("BEGIN:VCARD")
        for p in card.properties:
            out.append(str(p))
        out.append("END:VCARD")
        out.append("")
    return "\n".join(out)
