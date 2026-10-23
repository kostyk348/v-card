"""vCard DAG: nodes from X-OP cards, edges from RELATED properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .vcard import VCard, VCardFile


class GraphError(ValueError):
    """Invalid computation graph."""


@dataclass
class OpNode:
    uid: str
    op: str
    inputs: List[str]                 # external names or producer UIDs
    producer_cards: Dict[str, VCard] = field(default_factory=dict)


def build_graph(vcf: VCardFile) -> List[OpNode]:
    """Extract op nodes from cards carrying X-OP."""
    nodes: Dict[str, OpNode] = {}
    for card in vcf.cards:
        if card.op is None:
            continue
        if card.uid in nodes:
            raise GraphError(f"duplicate node uid: {card.uid}")
        nodes[card.uid] = OpNode(uid=card.uid, op=card.op, inputs=card.inputs())

    if not nodes:
        raise GraphError("no X-OP cards found — nothing to compile")

    # rewrite urn:uuid: references and validate edges
    for node in nodes.values():
        resolved = []
        for inp in node.inputs:
            producer = inp
            if producer not in nodes:
                for cand in nodes:
                    if inp.startswith(cand) or cand in inp:
                        producer = cand
                        break
            resolved.append(producer)
        node.inputs = resolved
    return list(nodes.values())


def topological_order(nodes: List[OpNode]) -> List[OpNode]:
    """Kahn's algorithm; raises on cycles."""
    by_uid = {n.uid: n for n in nodes}
    indeg = {uid: 0 for uid in by_uid}
    consumers: Dict[str, List[str]] = {uid: [] for uid in by_uid}

    for node in nodes:
        for inp in node.inputs:
            if inp in by_uid:
                indeg[node.uid] += 1
                consumers[inp].append(node.uid)

    queue = sorted(uid for uid, d in indeg.items() if d == 0)
    order: List[OpNode] = []
    while queue:
        uid = queue.pop(0)
        order.append(by_uid[uid])
        for child in consumers[uid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)

    if len(order) != len(nodes):
        stuck = sorted(set(by_uid) - {n.uid for n in order})
        raise GraphError(f"cycle detected in computation graph: {stuck}")
    return order


def external_inputs(nodes: List[OpNode]) -> List[str]:
    """External (non-node) input names in first-appearance order."""
    uids = {n.uid for n in nodes}
    seen: List[str] = []
    for node in nodes:
        for inp in node.inputs:
            if inp not in uids and inp not in seen:
                seen.append(inp)
    return seen


def sinks(nodes: List[OpNode]) -> List[OpNode]:
    """Nodes nobody consumes."""
    consumed = {inp for n in nodes for inp in n.inputs}
    return [n for n in nodes if n.uid not in consumed]
