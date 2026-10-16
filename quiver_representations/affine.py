"""Root-system utilities for acyclic affine ADE quivers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Dict, Iterator, Mapping, Tuple

from .quiver import Quiver


Edge = Tuple[int, int]


@dataclass(frozen=True)
class AffineQuiverInfo:
    """The orientation-independent affine type and its null root."""

    family: str
    index: int
    delta: Tuple[int, ...]
    vertices: Tuple[int, ...]
    orientation_parameters: Tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return f"affine_{self.family}{self.index}"

    def delta_dict(self) -> Dict[int, int]:
        return dict(zip(self.vertices, self.delta))


def underlying_edge_multiplicities(quiver: Quiver) -> Dict[Edge, int]:
    """Return multiplicities in the underlying unoriented multigraph."""
    multiplicities: Dict[Edge, int] = {}
    for arrow_id in quiver.get_arrows():
        arrow = quiver.arrows[arrow_id]
        source = int(arrow["source"])
        target = int(arrow["target"])
        edge = (min(source, target), max(source, target))
        multiplicities[edge] = multiplicities.get(edge, 0) + 1
    return multiplicities


def _adjacency(
    vertices: Tuple[int, ...], multiplicities: Mapping[Edge, int]
) -> Dict[int, Dict[int, int]]:
    adjacency = {vertex: {} for vertex in vertices}
    for (left, right), multiplicity in multiplicities.items():
        adjacency[left][right] = multiplicity
        adjacency[right][left] = multiplicity
    return adjacency


def _connected(vertices: Tuple[int, ...], adjacency: Mapping[int, Mapping[int, int]]) -> bool:
    if not vertices:
        return False
    seen = set()
    pending = [vertices[0]]
    while pending:
        vertex = pending.pop()
        if vertex in seen:
            continue
        seen.add(vertex)
        pending.extend(neighbor for neighbor in adjacency[vertex] if neighbor not in seen)
    return len(seen) == len(vertices)


def _null_vector(matrix: list[list[int]]) -> Tuple[int, ...]:
    """Return the primitive positive null vector of a corank-one integer matrix."""
    rows = [[Fraction(value) for value in row] for row in matrix]
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    pivot_columns = []
    pivot_row = 0

    for column in range(column_count):
        candidate = next(
            (row for row in range(pivot_row, row_count) if rows[row][column]),
            None,
        )
        if candidate is None:
            continue
        rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                rows[row][current] - factor * rows[pivot_row][current]
                for current in range(column_count)
            ]
        pivot_columns.append(column)
        pivot_row += 1

    free_columns = [column for column in range(column_count) if column not in pivot_columns]
    if len(free_columns) != 1:
        raise ValueError("The affine Cartan matrix must have corank one.")

    free = free_columns[0]
    vector = [Fraction(0) for _ in range(column_count)]
    vector[free] = Fraction(1)
    for row, column in reversed(list(enumerate(pivot_columns))):
        vector[column] = -sum(
            rows[row][current] * vector[current]
            for current in range(column + 1, column_count)
        )

    denominator_lcm = 1
    for value in vector:
        denominator_lcm = denominator_lcm * value.denominator // gcd(
            denominator_lcm, value.denominator
        )
    integers = [int(value * denominator_lcm) for value in vector]
    if all(value < 0 for value in integers):
        integers = [-value for value in integers]
    if not all(value > 0 for value in integers):
        raise ValueError("The Cartan nullspace has no strictly positive generator.")
    common = reduce(gcd, integers)
    return tuple(value // common for value in integers)


def _cycle_orientation_parameters(
    quiver: Quiver, vertices: Tuple[int, ...], adjacency: Mapping[int, Mapping[int, int]]
) -> Tuple[int, int]:
    start = min(vertices)
    neighbors = sorted(adjacency[start])
    previous = None
    current = start
    next_vertex = neighbors[0]
    cycle = [start]
    while next_vertex != start:
        cycle.append(next_vertex)
        previous, current = current, next_vertex
        candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous]
        if len(candidates) != 1:
            raise ValueError("Malformed affine A cycle.")
        next_vertex = candidates[0]

    forward = 0
    for index, source in enumerate(cycle):
        target = cycle[(index + 1) % len(cycle)]
        if quiver.get_arrows_between(source, target):
            forward += 1
    backward = len(cycle) - forward
    if forward == 0 or backward == 0:
        raise ValueError("Affine A orientations must have no oriented cycle.")
    return tuple(sorted((forward, backward)))


def classify_affine_quiver(quiver: Quiver) -> AffineQuiverInfo:
    """Recognize an acyclic simply-laced affine ADE quiver, including Kronecker."""
    vertices = tuple(sorted(quiver.get_vertices()))
    if len(vertices) < 2:
        raise ValueError("An affine quiver must have at least two vertices.")

    multiplicities = underlying_edge_multiplicities(quiver)
    adjacency = _adjacency(vertices, multiplicities)
    if not _connected(vertices, adjacency):
        raise ValueError("The underlying affine graph must be connected.")

    if len(vertices) == 2:
        edge = (vertices[0], vertices[1])
        if multiplicities != {edge: 2}:
            raise ValueError("The two-vertex affine graph must be the Kronecker graph.")
        family, index = "A", 1
        orientation_parameters = (1, 1)
    else:
        if any(multiplicity != 1 for multiplicity in multiplicities.values()):
            raise ValueError("Only the Kronecker affine graph may have parallel edges.")
        simple_edge_count = len(multiplicities)
        degrees = {vertex: len(adjacency[vertex]) for vertex in vertices}

        if simple_edge_count == len(vertices) and all(degree == 2 for degree in degrees.values()):
            family, index = "A", len(vertices) - 1
            orientation_parameters = _cycle_orientation_parameters(quiver, vertices, adjacency)
        elif simple_edge_count == len(vertices) - 1:
            branches = [vertex for vertex, degree in degrees.items() if degree >= 3]
            leaves = [vertex for vertex, degree in degrees.items() if degree == 1]
            if (
                len(branches) == 1
                and degrees[branches[0]] == 4
                and len(vertices) == 5
                and len(leaves) == 4
            ):
                family, index = "D", 4
            elif (
                len(branches) == 2
                and all(degrees[vertex] == 3 for vertex in branches)
                and len(leaves) == 4
                and all(degree <= 3 for degree in degrees.values())
            ):
                family, index = "D", len(vertices) - 1
                if index < 5:
                    raise ValueError("Malformed affine D graph.")
            elif (
                len(branches) == 1
                and degrees[branches[0]] == 3
                and len(leaves) == 3
                and len(vertices) in (7, 8, 9)
                and all(degree <= 3 for degree in degrees.values())
            ):
                family, index = "E", len(vertices) - 1
            else:
                raise ValueError("The underlying graph is not affine ADE.")
            orientation_parameters = None
        else:
            raise ValueError("The underlying graph is not affine ADE.")

    position = {vertex: index for index, vertex in enumerate(vertices)}
    cartan = [[0 for _ in vertices] for _ in vertices]
    for diagonal_index in range(len(vertices)):
        cartan[diagonal_index][diagonal_index] = 2
    for (left, right), multiplicity in multiplicities.items():
        i, j = position[left], position[right]
        cartan[i][j] -= multiplicity
        cartan[j][i] -= multiplicity
    try:
        delta = _null_vector(cartan)
    except ValueError as exc:
        raise ValueError("The underlying graph is not affine ADE.") from exc

    return AffineQuiverInfo(
        family=family,
        index=index,
        delta=delta,
        vertices=vertices,
        orientation_parameters=orientation_parameters,
    )


def affine_tits_value(quiver: Quiver, dimension: Mapping[int, int]) -> int:
    """Return the Tits value q(d)=<d,d>."""
    return int(quiver.euler_form(dimension, dimension))


def affine_defect(quiver: Quiver, dimension: Mapping[int, int]) -> int:
    """Return the defect <delta,d>, negative on Kronecker preprojectives."""
    info = classify_affine_quiver(quiver)
    return int(quiver.euler_form(info.delta_dict(), dimension))


def enumerate_affine_real_roots_bounded(
    quiver: Quiver, bound: Mapping[int, int]
) -> Iterator[Dict[int, int]]:
    """Enumerate bounded positive real roots by Weyl reflections."""
    info = classify_affine_quiver(quiver)
    vertices = info.vertices
    limits = {}
    for vertex in vertices:
        value = int(bound.get(vertex, 0))
        if value < 0:
            raise ValueError("Root bounds must be nonnegative.")
        limits[vertex] = value

    adjacency = _adjacency(vertices, underlying_edge_multiplicities(quiver))
    pending = []
    seen = set()
    for vertex in vertices:
        if limits[vertex] == 0:
            continue
        root = tuple(1 if current == vertex else 0 for current in vertices)
        seen.add(root)
        pending.append(root)

    cursor = 0
    while cursor < len(pending):
        root = pending[cursor]
        cursor += 1
        yield dict(zip(vertices, root))
        dense = dict(zip(vertices, root))
        for vertex in vertices:
            reflected_coordinate = (
                sum(
                    multiplicity * dense[neighbor]
                    for neighbor, multiplicity in adjacency[vertex].items()
                )
                - dense[vertex]
            )
            if reflected_coordinate < 0 or reflected_coordinate > limits[vertex]:
                continue
            reflected = dict(dense)
            reflected[vertex] = reflected_coordinate
            reflected_tuple = tuple(reflected[current] for current in vertices)
            if reflected_tuple in seen:
                continue
            if affine_tits_value(quiver, reflected) != 1:
                raise RuntimeError("A simple reflection failed to preserve the Tits form.")
            seen.add(reflected_tuple)
            pending.append(reflected_tuple)
