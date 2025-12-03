from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from quiver_representations import Quiver


@dataclass(frozen=True)
class CoverageSpec:
    """Multiplicities for projective/injective components with a friendly name."""

    name: str
    np: Dict[int, int]
    ni: Dict[int, int]


@dataclass(frozen=True)
class QuiverConfig:
    """Configuration for constructing a quiver."""

    quiver_name: str
    vertices: List[str]
    arrows: List[Tuple[str, str, str]]


@dataclass(frozen=True)
class PipelineConfig:
    """Bundle the quiver definition with its coverage specifications."""

    quiver: QuiverConfig
    coverages: List[CoverageSpec]
    runtime: "PipelineRuntime"


@dataclass(frozen=True)
class PipelineRuntime:
    """Runtime parameters that were previously CLI-only."""

    output_dir: str = "results"
    workers: int = 64
    r_max: int = 3
    hilbert_workers: int = 64
    gc_heap_size: str = "20G"
    hom_prime: Optional[int] = None


def build_quiver(cfg: QuiverConfig) -> Quiver:
    """Build a :class:`Quiver` instance from a :class:`QuiverConfig`."""

    quiver = Quiver(cfg.quiver_name)
    vertex_map: Dict[str, int] = {}

    for vertex_label in cfg.vertices:
        vertex_map[vertex_label] = quiver.add_vertex(vertex_label)

    for src_label, dst_label, arrow_label in cfg.arrows:
        quiver.add_arrow(vertex_map[src_label], vertex_map[dst_label], arrow_label)

    return quiver


DEFAULT_A3_PIPELINE = PipelineConfig(
    quiver=QuiverConfig(
        quiver_name="A3",
        vertices=["v0", "v1", "v2"],
        arrows=[
            ("v0", "v1", "a01"),
            ("v1", "v2", "a12"),
        ],
    ),
    coverages=[
        CoverageSpec(
            name="A3_uniform_P1_I1",
            np={0: 1, 1: 1, 2: 1},
            ni={0: 1, 1: 1, 2: 1},
        )
    ],
    runtime=PipelineRuntime(),
)


DEFAULT_D4_PIPELINE = PipelineConfig(
    quiver=QuiverConfig(
        quiver_name="D4",
        vertices=["v0", "v1", "v2", "v3"],
        arrows=[
            ("v0", "v2", "a02"),
            ("v1", "v2", "a12"),
            ("v2", "v3", "a23"),
        ],
    ),
    coverages=[
        CoverageSpec(
            name="D4_uniform_P1_I1",
            np={0: 1, 1: 1, 2: 1, 3: 1},
            ni={0: 1, 1: 1, 2: 1, 3: 1},
        ),
    ],
    runtime=PipelineRuntime(hom_prime=107),
)


def make_an_pipeline(
    n: int,
    projective_mult: int = 1,
    injective_mult: int = 1,
    runtime: Optional[PipelineRuntime] = None,
) -> PipelineConfig:
    """Create a simple A_n pipeline with uniform coverage."""

    if n < 2:
        raise ValueError("A_n requires n >= 2")

    vertices = [f"v{i}" for i in range(n)]
    arrows = [
        (f"v{i}", f"v{i + 1}", f"a{i}{i + 1}")
        for i in range(n - 1)
    ]

    quiver_cfg = QuiverConfig(quiver_name=f"A{n}", vertices=vertices, arrows=arrows)
    coverage = CoverageSpec(
        name=f"A{n}_uniform_P{projective_mult}_I{injective_mult}",
        np={i: projective_mult for i in range(n)},
        ni={i: injective_mult for i in range(n)},
    )

    return PipelineConfig(
        quiver=quiver_cfg,
        coverages=[coverage],
        runtime=runtime or PipelineRuntime(),
    )


def make_dn_pipeline(
    n: int,
    projective_mult: int = 1,
    injective_mult: int = 1,
    runtime: Optional[PipelineRuntime] = None,
) -> PipelineConfig:
    """Create a simple D_n pipeline with uniform coverage."""

    if n < 4:
        raise ValueError("D_n requires n >= 4")

    vertices = [f"v{i}" for i in range(n)]

    arrows: List[Tuple[str, str, str]] = [
        ("v0", "v2", "a02"),
        ("v1", "v2", "a12"),
    ]

    for i in range(2, n - 1):
        arrows.append((f"v{i}", f"v{i + 1}", f"a{i}{i + 1}"))

    quiver_cfg = QuiverConfig(quiver_name=f"D{n}", vertices=vertices, arrows=arrows)
    coverage = CoverageSpec(
        name=f"D{n}_uniform_P{projective_mult}_I{injective_mult}",
        np={i: projective_mult for i in range(n)},
        ni={i: injective_mult for i in range(n)},
    )

    return PipelineConfig(
        quiver=quiver_cfg,
        coverages=[coverage],
        runtime=runtime or PipelineRuntime(hom_prime=107),
    )


def pipeline_to_dict(cfg: PipelineConfig) -> Dict:
    """Convenience helper for pretty-printing pipeline configs."""

    return asdict(cfg)
