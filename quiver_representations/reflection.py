"""Dimension-vector helpers for reflection functors."""

from numbers import Integral
from typing import Dict, Mapping, Optional, Set, Tuple

from .affine import classify_affine_quiver, underlying_edge_multiplicities
from .quiver import Quiver


Edge = Tuple[int, int]


def _underlying_dynkin_graph(quiver: Quiver) -> Tuple[Set[Edge], Dict[int, Set[int]]]:
    """Return the underlying graph after verifying that it has type A or D."""
    vertices = quiver.get_vertices()
    if not vertices:
        raise ValueError("Reflection requires a nonempty quiver of Dynkin type A or D.")

    adjacency = {vertex: set() for vertex in vertices}
    edges: Set[Edge] = set()
    for arrow_id in quiver.get_arrows():
        arrow = quiver.arrows[arrow_id]
        source = arrow["source"]
        target = arrow["target"]
        edge = (min(source, target), max(source, target))
        if edge in edges:
            raise ValueError("Dynkin quivers of type A or D cannot have parallel arrows.")
        edges.add(edge)
        adjacency[source].add(target)
        adjacency[target].add(source)

    # Both A_n and D_n are trees.
    if len(edges) != len(vertices) - 1:
        raise ValueError("The underlying graph must be a tree of Dynkin type A or D.")
    seen = set()
    pending = [vertices[0]]
    while pending:
        vertex = pending.pop()
        if vertex in seen:
            continue
        seen.add(vertex)
        pending.extend(adjacency[vertex] - seen)
    if len(seen) != len(vertices):
        raise ValueError("The underlying graph must be connected.")

    degrees = {vertex: len(neighbors) for vertex, neighbors in adjacency.items()}
    if max(degrees.values()) <= 2:
        return edges, adjacency

    branches = [vertex for vertex, degree in degrees.items() if degree == 3]
    leaves = [vertex for vertex, degree in degrees.items() if degree == 1]
    if len(vertices) < 4 or len(branches) != 1 or len(leaves) != 3:
        raise ValueError("The underlying graph must have Dynkin type A or D.")
    if any(degree not in (1, 2, 3) for degree in degrees.values()):
        raise ValueError("The underlying graph must have Dynkin type A or D.")

    branch = branches[0]
    distances = {}
    pending_with_distance = [(branch, 0)]
    while pending_with_distance:
        vertex, distance = pending_with_distance.pop()
        if vertex in distances:
            continue
        distances[vertex] = distance
        pending_with_distance.extend(
            (neighbor, distance + 1)
            for neighbor in adjacency[vertex]
            if neighbor not in distances
        )
    arm_lengths = sorted(distances[leaf] for leaf in leaves)
    if arm_lengths != [1, 1, len(vertices) - 3]:
        raise ValueError("The underlying graph must have Dynkin type A or D.")

    return edges, adjacency


def _validate_indecomposable_dimension(
    vertices: Set[int], edges: Set[Edge], dimension: Mapping[int, int]
) -> Dict[int, int]:
    extra_vertices = set(dimension) - vertices
    if extra_vertices:
        raise ValueError(
            f"Dimension vector contains vertices outside the quiver: {sorted(extra_vertices)}."
        )

    dense = {}
    for vertex in vertices:
        value = dimension.get(vertex, 0)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"Dimension at vertex {vertex} must be an integer, got {value!r}.")
        value = int(value)
        if value < 0:
            raise ValueError(f"Dimension at vertex {vertex} must be nonnegative, got {value}.")
        dense[vertex] = value

    # For finite simply-laced Dynkin type, the positive roots are exactly the
    # nonzero nonnegative vectors for which the Tits form is one.
    tits_value = sum(value * value for value in dense.values()) - sum(
        dense[source] * dense[target] for source, target in edges
    )
    if not any(dense.values()) or tits_value != 1:
        raise ValueError(
            "Dimension vector is not a known indecomposable dimension vector "
            "for this Dynkin quiver."
        )
    return dense


def _validate_affine_indecomposable_dimension(
    quiver: Quiver, dimension: Mapping[int, int]
) -> Dict[int, int]:
    """Validate a positive real root or an imaginary root ``m delta``."""
    info = classify_affine_quiver(quiver)
    vertices = set(info.vertices)
    extra_vertices = set(dimension) - vertices
    if extra_vertices:
        raise ValueError(
            f"Dimension vector contains vertices outside the quiver: {sorted(extra_vertices)}."
        )

    dense = {}
    for vertex in info.vertices:
        value = dimension.get(vertex, 0)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"Dimension at vertex {vertex} must be an integer, got {value!r}.")
        value = int(value)
        if value < 0:
            raise ValueError(f"Dimension at vertex {vertex} must be nonnegative, got {value}.")
        dense[vertex] = value

    if not any(dense.values()):
        raise ValueError("Dimension vector is not a known affine indecomposable dimension vector.")
    tits_value = quiver.euler_form(dense, dense)
    if tits_value == 1:
        return dense
    if tits_value == 0:
        multipliers = {
            dense[vertex] // delta
            for vertex, delta in zip(info.vertices, info.delta)
            if dense[vertex] % delta == 0
        }
        if len(multipliers) == 1 and next(iter(multipliers)) > 0:
            multiplier = next(iter(multipliers))
            if all(
                dense[vertex] == multiplier * delta
                for vertex, delta in zip(info.vertices, info.delta)
            ):
                return dense
    raise ValueError("Dimension vector is not a known affine indecomposable dimension vector.")


def reflect_dimension_vector(
    quiver: Quiver,
    dimension: Mapping[int, int],
    vertex: int,
) -> Optional[Dict[int, int]]:
    """Reflect a known finite A/D or affine ADE indecomposable dimension vector.

    The result is the simple-root reflection

        d'_vertex = sum(m_vertex,neighbor * d_neighbor) - d_vertex,

    with all other coordinates unchanged. Missing coordinates in ``dimension``
    are interpreted as zero, while the returned dictionary is dense and follows
    ``quiver.get_vertices()`` order.

    ``None`` is returned precisely for the simple module at ``vertex``, since a
    BGP reflection functor kills that summand. The function validates only the
    underlying root calculation; a module-reflection caller must separately
    require ``vertex`` to be a source or sink.

    Raises:
        ValueError: If the quiver is unsupported, ``vertex`` is absent, or
            ``dimension`` is not a known indecomposable dimension vector.
        TypeError: If a dimension is not an integer.
    """
    vertices = set(quiver.get_vertices())
    if vertex not in vertices:
        raise ValueError(f"Vertex {vertex} does not exist in the quiver.")

    is_affine = False
    try:
        edges, adjacency = _underlying_dynkin_graph(quiver)
    except ValueError as finite_error:
        try:
            classify_affine_quiver(quiver)
        except ValueError:
            raise finite_error
        is_affine = True
        multiplicities = underlying_edge_multiplicities(quiver)
        adjacency_with_multiplicity = {current: {} for current in vertices}
        for (left, right), multiplicity in multiplicities.items():
            adjacency_with_multiplicity[left][right] = multiplicity
            adjacency_with_multiplicity[right][left] = multiplicity
        dense = _validate_affine_indecomposable_dimension(quiver, dimension)
    else:
        adjacency_with_multiplicity = {
            current: {neighbor: 1 for neighbor in neighbors}
            for current, neighbors in adjacency.items()
        }
        dense = _validate_indecomposable_dimension(vertices, edges, dimension)
    if dense[vertex] == 1 and sum(dense.values()) == 1:
        return None

    reflected = dict(dense)
    reflected[vertex] = (
        sum(
            multiplicity * dense[neighbor]
            for neighbor, multiplicity in adjacency_with_multiplicity[vertex].items()
        )
        - dense[vertex]
    )
    if reflected[vertex] < 0:
        raise RuntimeError("Reflection of a non-simple positive root became negative.")
    if is_affine:
        _validate_affine_indecomposable_dimension(quiver, reflected)
    else:
        _validate_indecomposable_dimension(vertices, edges, reflected)

    # Keep output order stable even though validation uses sets internally.
    return {current: reflected[current] for current in quiver.get_vertices()}
