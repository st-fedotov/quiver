from dataclasses import dataclass
from typing import Dict, List, Tuple

from quiver_representations import Quiver


@dataclass
class QuiverConfig:
    """Configuration for constructing a quiver."""

    quiver_name: str
    vertices: List[str]
    arrows: List[Tuple[str, str, str]]


def build_quiver(cfg: QuiverConfig) -> Quiver:
    """Build a :class:`Quiver` instance from a :class:`QuiverConfig`."""

    quiver = Quiver(cfg.quiver_name)
    vertex_map: Dict[str, int] = {}

    for vertex_label in cfg.vertices:
        vertex_map[vertex_label] = quiver.add_vertex(vertex_label)

    for src_label, dst_label, arrow_label in cfg.arrows:
        quiver.add_arrow(vertex_map[src_label], vertex_map[dst_label], arrow_label)

    return quiver
