"""Finite-field normal forms for the Kronecker quiver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, Mapping, Sequence, Tuple

from ..batch.core import write_batch_rad_from_triples
from ..field import ComplexNumbers, FiniteField
from ..module import Module
from ..quiver import Quiver


class KroneckerRegularMode(str, Enum):
    """Which homogeneous tubes are included in a finite catalogue."""

    RATIONAL_JORDAN = "rational_jordan"
    CLOSED_POINTS = "closed_points"


class KroneckerKind(str, Enum):
    PREPROJECTIVE = "preprojective"
    REGULAR = "regular"
    PREINJECTIVE = "preinjective"


INFINITY = "infinity"


@dataclass(frozen=True)
class KroneckerIndec:
    """A Kronecker indecomposable in the rational-Jordan catalogue."""

    kind: KroneckerKind
    index: int = 0
    parameter: int | str | None = None
    quasi_length: int = 0

    @classmethod
    def preprojective(cls, index: int) -> "KroneckerIndec":
        if index < 0:
            raise ValueError("A preprojective index must be nonnegative.")
        return cls(KroneckerKind.PREPROJECTIVE, index=index)

    @classmethod
    def preinjective(cls, index: int) -> "KroneckerIndec":
        if index < 0:
            raise ValueError("A preinjective index must be nonnegative.")
        return cls(KroneckerKind.PREINJECTIVE, index=index)

    @classmethod
    def regular(cls, parameter: int | str, quasi_length: int) -> "KroneckerIndec":
        if parameter != INFINITY and not isinstance(parameter, int):
            raise TypeError("A rational regular parameter must be an integer or 'infinity'.")
        if quasi_length <= 0:
            raise ValueError("A regular quasi-length must be positive.")
        return cls(
            KroneckerKind.REGULAR,
            parameter=parameter,
            quasi_length=quasi_length,
        )

    def dimension_at_source_sink(self) -> Tuple[int, int]:
        if self.kind == KroneckerKind.PREPROJECTIVE:
            return self.index, self.index + 1
        if self.kind == KroneckerKind.PREINJECTIVE:
            return self.index + 1, self.index
        return self.quasi_length, self.quasi_length

    def label(self) -> str:
        if self.kind == KroneckerKind.PREPROJECTIVE:
            return f"P({self.index})"
        if self.kind == KroneckerKind.PREINJECTIVE:
            return f"I({self.index})"
        parameter = "inf" if self.parameter == INFINITY else str(self.parameter)
        return f"R({parameter},{self.quasi_length})"


KroneckerBag = Tuple[KroneckerIndec, ...]


@dataclass(frozen=True)
class KroneckerJob:
    """One finite-field node together with its characteristic-zero lift."""

    bag: KroneckerBag
    finite_module: Module
    complex_module: Module
    target_dimension: Dict[int, int]

    def as_rad_job(self) -> tuple[Quiver, Module, Dict[int, int]]:
        """Return the characteristic-zero triple expected by the RAD writer."""
        return self.complex_module.quiver, self.complex_module, self.target_dimension


def make_kronecker_quiver(name: str = "Kronecker") -> Quiver:
    quiver = Quiver(name)
    source = quiver.add_vertex("source")
    sink = quiver.add_vertex("sink")
    quiver.add_arrow(source, sink, "a")
    quiver.add_arrow(source, sink, "b")
    return quiver


def kronecker_vertices_and_arrows(quiver: Quiver) -> Tuple[int, int, Tuple[int, int]]:
    """Validate a Kronecker orientation and return source, sink, and arrow IDs."""
    vertices = tuple(quiver.get_vertices())
    arrows = tuple(quiver.get_arrows())
    if len(vertices) != 2 or len(arrows) != 2:
        raise ValueError("A Kronecker quiver must have two vertices and two arrows.")
    endpoints = {
        (quiver.arrows[arrow]["source"], quiver.arrows[arrow]["target"])
        for arrow in arrows
    }
    if len(endpoints) != 1:
        raise ValueError("The two Kronecker arrows must have the same orientation.")
    source, sink = next(iter(endpoints))
    return int(source), int(sink), arrows


def _coerce_mode(mode: KroneckerRegularMode | str) -> KroneckerRegularMode:
    try:
        return KroneckerRegularMode(mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in KroneckerRegularMode)
        raise ValueError(f"Unknown Kronecker regular mode {mode!r}; expected one of {choices}.") from exc


def rational_jordan_parameters(field: FiniteField) -> Tuple[int | str, ...]:
    """Return P^1(F_p), using integer representatives and an infinity marker."""
    if field.degree != 1:
        raise NotImplementedError(
            "Characteristic-zero lifts of rational Jordan parameters are currently "
            "implemented only for prime fields GF(p)."
        )
    return tuple(range(field.order)) + (INFINITY,)


def enumerate_kronecker_indecs_bounded(
    quiver: Quiver,
    field: FiniteField,
    coverage: Mapping[int, int],
    *,
    regular_mode: KroneckerRegularMode | str = KroneckerRegularMode.RATIONAL_JORDAN,
) -> Tuple[KroneckerIndec, ...]:
    """Enumerate indecomposables that fit inside ``coverage`` coordinatewise."""
    source, sink, _ = kronecker_vertices_and_arrows(quiver)
    source_bound = int(coverage.get(source, 0))
    sink_bound = int(coverage.get(sink, 0))
    if source_bound < 0 or sink_bound < 0:
        raise ValueError("Coverage dimensions must be nonnegative.")

    mode = _coerce_mode(regular_mode)
    if mode == KroneckerRegularMode.CLOSED_POINTS:
        # TODO: add irreducible-polynomial companion blocks and packet-valued lifts.
        raise NotImplementedError(
            "closed_points mode is reserved for irreducible-polynomial blocks; "
            "use regular_mode='rational_jordan' for the implemented catalogue."
        )

    indecomposables = []
    for index in range(min(source_bound, max(sink_bound - 1, -1)) + 1):
        indecomposables.append(KroneckerIndec.preprojective(index))

    max_regular_length = min(source_bound, sink_bound)
    for parameter in rational_jordan_parameters(field):
        for quasi_length in range(1, max_regular_length + 1):
            indecomposables.append(KroneckerIndec.regular(parameter, quasi_length))

    for index in range(min(max(source_bound - 1, -1), sink_bound) + 1):
        indecomposables.append(KroneckerIndec.preinjective(index))

    return tuple(indecomposables)


def enumerate_kronecker_bags_from_coverage_iter(
    quiver: Quiver,
    field: FiniteField,
    coverage: Mapping[int, int],
    *,
    regular_mode: KroneckerRegularMode | str = KroneckerRegularMode.RATIONAL_JORDAN,
) -> Iterator[KroneckerBag]:
    """Enumerate Krull--Schmidt bags of the prescribed dimension vector."""
    source, sink, _ = kronecker_vertices_and_arrows(quiver)
    target = (int(coverage.get(source, 0)), int(coverage.get(sink, 0)))
    candidates = enumerate_kronecker_indecs_bounded(
        quiver, field, coverage, regular_mode=regular_mode
    )
    dimensions = tuple(candidate.dimension_at_source_sink() for candidate in candidates)
    bag: list[KroneckerIndec] = []

    def recurse(index: int, remaining_source: int, remaining_sink: int):
        if remaining_source == 0 and remaining_sink == 0:
            yield tuple(bag)
            return
        if index == len(candidates):
            return

        source_dimension, sink_dimension = dimensions[index]
        limits = []
        if source_dimension:
            limits.append(remaining_source // source_dimension)
        if sink_dimension:
            limits.append(remaining_sink // sink_dimension)
        maximum = min(limits) if limits else 0

        for multiplicity in range(maximum + 1):
            if multiplicity:
                bag.extend([candidates[index]] * multiplicity)
            yield from recurse(
                index + 1,
                remaining_source - multiplicity * source_dimension,
                remaining_sink - multiplicity * sink_dimension,
            )
            if multiplicity:
                del bag[-multiplicity:]

    yield from recurse(0, target[0], target[1])


def enumerate_kronecker_bags_from_coverage(
    quiver: Quiver,
    field: FiniteField,
    coverage: Mapping[int, int],
    *,
    regular_mode: KroneckerRegularMode | str = KroneckerRegularMode.RATIONAL_JORDAN,
) -> list[KroneckerBag]:
    return list(
        enumerate_kronecker_bags_from_coverage_iter(
            quiver, field, coverage, regular_mode=regular_mode
        )
    )


def _indecomposable_arrow_matrices(field, indecomposable: KroneckerIndec):
    source_dimension, sink_dimension = indecomposable.dimension_at_source_sink()
    first = field.zero_matrix(sink_dimension, source_dimension)
    second = field.zero_matrix(sink_dimension, source_dimension)

    if indecomposable.kind == KroneckerKind.PREPROJECTIVE:
        for index in range(source_dimension):
            first[index, index] = field.one
            second[index + 1, index] = field.one
    elif indecomposable.kind == KroneckerKind.PREINJECTIVE:
        for index in range(sink_dimension):
            first[index, index] = field.one
            second[index, index + 1] = field.one
    elif indecomposable.parameter == INFINITY:
        for index in range(source_dimension):
            second[index, index] = field.one
        for index in range(source_dimension - 1):
            first[index, index + 1] = field.one
    else:
        parameter = field.element(int(indecomposable.parameter))
        for index in range(source_dimension):
            first[index, index] = field.one
            second[index, index] = parameter
        for index in range(source_dimension - 1):
            second[index, index + 1] = field.one
    return first, second


def form_kronecker_module_from_bag_explicit(
    quiver: Quiver, field, bag: Sequence[KroneckerIndec]
) -> Module:
    """Build the block-diagonal module represented by ``bag``."""
    source, sink, arrows = kronecker_vertices_and_arrows(quiver)
    dimensions = [indecomposable.dimension_at_source_sink() for indecomposable in bag]
    source_dimension = sum(dimension[0] for dimension in dimensions)
    sink_dimension = sum(dimension[1] for dimension in dimensions)
    first = field.zero_matrix(sink_dimension, source_dimension)
    second = field.zero_matrix(sink_dimension, source_dimension)

    source_offset = 0
    sink_offset = 0
    for indecomposable, (current_source, current_sink) in zip(bag, dimensions):
        block_first, block_second = _indecomposable_arrow_matrices(field, indecomposable)
        first[
            sink_offset : sink_offset + current_sink,
            source_offset : source_offset + current_source,
        ] = block_first
        second[
            sink_offset : sink_offset + current_sink,
            source_offset : source_offset + current_source,
        ] = block_second
        source_offset += current_source
        sink_offset += current_sink

    name = " + ".join(indecomposable.label() for indecomposable in bag) or "0"
    return Module(
        quiver,
        field,
        name=name,
        dimensions={source: source_dimension, sink: sink_dimension},
        maps={arrows[0]: first, arrows[1]: second},
    )


def lift_kronecker_bag_to_characteristic_zero(
    quiver: Quiver, bag: Sequence[KroneckerIndec]
) -> Module:
    """Use the same integral Jordan normal form over the characteristic-zero backend."""
    return form_kronecker_module_from_bag_explicit(quiver, ComplexNumbers(), bag)


def make_kronecker_jobs(
    quiver: Quiver,
    field: FiniteField,
    coverage: Mapping[int, int],
    target_dimension: Mapping[int, int],
    *,
    regular_mode: KroneckerRegularMode | str = KroneckerRegularMode.RATIONAL_JORDAN,
) -> list[KroneckerJob]:
    """Build finite-field nodes and the characteristic-zero modules used by RAD jobs."""
    target = {vertex: int(target_dimension.get(vertex, 0)) for vertex in quiver.get_vertices()}
    jobs = []
    for bag in enumerate_kronecker_bags_from_coverage_iter(
        quiver, field, coverage, regular_mode=regular_mode
    ):
        jobs.append(
            KroneckerJob(
                bag=bag,
                finite_module=form_kronecker_module_from_bag_explicit(quiver, field, bag),
                complex_module=lift_kronecker_bag_to_characteristic_zero(quiver, bag),
                target_dimension=target,
            )
        )
    return jobs


def write_batch_rad_from_kronecker_jobs(
    jobs: Sequence[KroneckerJob], **batch_options
):
    """Write characteristic-zero RAD jobs for a finite-field catalogue.

    Enumeration and Hom computations use ``job.finite_module``.  This adapter
    deliberately passes ``job.complex_module`` to the existing Macaulay2
    writer, so its ``QQ`` coefficient ring is unchanged.
    """
    return write_batch_rad_from_triples(
        [job.as_rad_job() for job in jobs],
        **batch_options,
    )
