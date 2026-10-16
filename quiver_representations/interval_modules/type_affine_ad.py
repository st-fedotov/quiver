"""Bounded indecomposable catalogues for acyclic affine A and D quivers."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np

from ..affine import AffineQuiverInfo, classify_affine_quiver, enumerate_affine_real_roots_bounded
from ..batch.core import write_batch_rad_from_triples
from ..field import ComplexNumbers, FiniteField, ZeroMap
from ..module import Module
from ..quiver import Quiver
from .type_kronecker import KroneckerRegularMode


@dataclass(frozen=True)
class AffineIndecRecord:
    """An indecomposable normal form over a finite field and over characteristic zero."""

    label: str
    kind: str
    dimension: Tuple[int, ...]
    finite_module: Module
    complex_module: Module
    regular_seed: int | None = None
    quasi_length: int | None = None


AffineBag = Tuple[AffineIndecRecord, ...]


def _find_hom_basis(source: Module, target: Module):
    # Lazy import avoids a package cycle: analysis imports interval_modules.
    from ..analysis.hom import find_hom_basis

    return find_hom_basis(source, target)


@dataclass(frozen=True)
class AffineJob:
    bag: AffineBag
    finite_module: Module
    complex_module: Module
    target_dimension: Dict[int, int]

    def as_rad_job(self) -> tuple[Quiver, Module, Dict[int, int]]:
        return self.complex_module.quiver, self.complex_module, self.target_dimension


def _coerce_mode(mode: KroneckerRegularMode | str) -> KroneckerRegularMode:
    try:
        return KroneckerRegularMode(mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in KroneckerRegularMode)
        raise ValueError(
            f"Unknown affine regular mode {mode!r}; expected one of {choices}."
        ) from exc


def _dense_coverage(
    info: AffineQuiverInfo, coverage: Mapping[int, int]
) -> Dict[int, int]:
    extra = set(coverage) - set(info.vertices)
    if extra:
        raise ValueError(f"Coverage contains vertices outside the quiver: {sorted(extra)}.")
    dense = {vertex: int(coverage.get(vertex, 0)) for vertex in info.vertices}
    if any(value < 0 for value in dense.values()):
        raise ValueError("Coverage dimensions must be nonnegative.")
    return dense


def _balanced_matrix(matrix, prime: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=int)
    return np.where(values > prime // 2, values - prime, values)


def _lift_module(
    module: Module, complex_field: ComplexNumbers, *, name: str | None = None
) -> Module:
    prime = int(module.field.characteristic)
    maps = {}
    for arrow in module.quiver.get_arrows():
        matrix = module.maps[arrow]
        if isinstance(matrix, ZeroMap):
            maps[arrow] = ZeroMap(matrix.source_dim, matrix.target_dim)
        else:
            maps[arrow] = complex_field.matrix(_balanced_matrix(matrix, prime))
    return Module(
        module.quiver,
        complex_field,
        name=name or module.name,
        dimensions=dict(module.spaces),
        maps=maps,
    )


def _matrix_or_zeros(module: Module, arrow: int, *, integral: bool = False):
    matrix = module.maps[arrow]
    if isinstance(matrix, ZeroMap):
        shape = (matrix.target_dim, matrix.source_dim)
        return np.zeros(shape, dtype=int) if integral else module.field.zero_matrix(*shape)
    if integral:
        return np.rint(np.asarray(matrix).real).astype(int)
    return matrix


def _exact_rank(matrix: np.ndarray) -> int:
    rows = [[Fraction(int(value)) for value in row] for row in np.asarray(matrix)]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
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
        for row in range(pivot_row + 1, row_count):
            if not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                rows[row][current] - factor * rows[pivot_row][current]
                for current in range(column_count)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _integral_hom_matrix(source: Module, target: Module) -> np.ndarray:
    vertices = list(source.quiver.get_vertices())
    starts = {}
    total_variables = 0
    for vertex in vertices:
        starts[vertex] = total_variables
        total_variables += target.spaces[vertex] * source.spaces[vertex]

    total_rows = sum(
        target.spaces[source.quiver.arrows[arrow]["target"]]
        * source.spaces[source.quiver.arrows[arrow]["source"]]
        for arrow in source.quiver.get_arrows()
    )
    equations = np.zeros((total_rows, total_variables), dtype=int)
    row_offset = 0
    for arrow in source.quiver.get_arrows():
        arrow_data = source.quiver.arrows[arrow]
        tail = arrow_data["source"]
        head = arrow_data["target"]
        source_tail = source.spaces[tail]
        source_head = source.spaces[head]
        target_tail = target.spaces[tail]
        target_head = target.spaces[head]
        rows_here = target_head * source_tail
        if not rows_here:
            continue
        target_map = _matrix_or_zeros(target, arrow, integral=True)
        source_map = _matrix_or_zeros(source, arrow, integral=True)
        left = np.kron(np.eye(source_tail, dtype=int), target_map)
        right = np.kron(source_map.T, np.eye(target_head, dtype=int))
        source_slice = slice(starts[tail], starts[tail] + target_tail * source_tail)
        target_slice = slice(starts[head], starts[head] + target_head * source_head)
        row_slice = slice(row_offset, row_offset + rows_here)
        equations[row_slice, source_slice] += left
        equations[row_slice, target_slice] -= right
        row_offset += rows_here
    return equations


def _exact_end_dimension(module: Module) -> int:
    variables = sum(dimension * dimension for dimension in module.spaces.values())
    return variables - _exact_rank(_integral_hom_matrix(module, module))


def _modular_rank(matrix: np.ndarray, prime: int) -> int:
    rows = [[int(value) % prime for value in row] for row in np.asarray(matrix)]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    pivot_row = 0
    for column in range(column_count):
        candidate = next(
            (row for row in range(pivot_row, row_count) if rows[row][column]),
            None,
        )
        if candidate is None:
            continue
        rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
        inverse = pow(rows[pivot_row][column], -1, prime)
        rows[pivot_row] = [(value * inverse) % prime for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                (rows[row][current] - factor * rows[pivot_row][current]) % prime
                for current in range(column_count)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _finite_end_dimension(module: Module) -> int:
    variables = sum(dimension * dimension for dimension in module.spaces.values())
    return variables - _modular_rank(
        _integral_hom_matrix(module, module), int(module.field.characteristic)
    )


def _finite_hom_dimension(source: Module, target: Module) -> int:
    variables = sum(
        source.spaces[vertex] * target.spaces[vertex]
        for vertex in source.quiver.get_vertices()
    )
    return variables - _modular_rank(
        _integral_hom_matrix(source, target), int(source.field.characteristic)
    )


def _stable_seed(quiver: Quiver, dimension: Mapping[int, int], salt: int) -> int:
    signature = (
        tuple(
            (quiver.arrows[arrow]["source"], quiver.arrows[arrow]["target"])
            for arrow in quiver.get_arrows()
        ),
        tuple((vertex, dimension[vertex]) for vertex in sorted(dimension)),
        int(salt),
    )
    digest = hashlib.sha256(repr(signature).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _representation_from_values(
    quiver: Quiver,
    field: FiniteField,
    dimension: Mapping[int, int],
    values: Sequence[int],
    name: str,
) -> Module:
    maps = {}
    offset = 0
    for arrow in quiver.get_arrows():
        source = quiver.arrows[arrow]["source"]
        target = quiver.arrows[arrow]["target"]
        size = dimension[source] * dimension[target]
        block = values[offset : offset + size]
        offset += size
        if not dimension[source] or not dimension[target]:
            maps[arrow] = ZeroMap(dimension[source], dimension[target])
        else:
            maps[arrow] = field.matrix(block).reshape(
                (dimension[target], dimension[source])
            )
    return Module(
        quiver,
        field,
        name=name,
        dimensions=dict(dimension),
        maps=maps,
    )


def _candidate_value_vectors(
    field: FiniteField,
    entry_count: int,
    rng: random.Random,
    max_attempts: int,
    exhaustive_limit: int = 65536,
) -> Iterator[Tuple[int, ...]]:
    total = field.order**entry_count
    if total <= exhaustive_limit:
        indices = list(range(total))
        rng.shuffle(indices)
        for encoded in indices:
            values = []
            for _ in range(entry_count):
                encoded, value = divmod(encoded, field.order)
                values.append(value)
            yield tuple(values)
        return
    for _ in range(max_attempts):
        yield tuple(rng.randrange(field.order) for _ in range(entry_count))


def _random_schur_module(
    quiver: Quiver,
    field: FiniteField,
    dimension: Mapping[int, int],
    *,
    label: str,
    random_seed: int,
    max_attempts: int,
) -> Module:
    entry_count = sum(
        dimension[quiver.arrows[arrow]["source"]]
        * dimension[quiver.arrows[arrow]["target"]]
        for arrow in quiver.get_arrows()
    )
    rng = random.Random(_stable_seed(quiver, dimension, random_seed))
    for values in _candidate_value_vectors(field, entry_count, rng, max_attempts):
        candidate = _representation_from_values(quiver, field, dimension, values, label)
        if _finite_end_dimension(candidate) == 1:
            return candidate
    raise RuntimeError(
        f"Could not construct the real-root indecomposable {label} after "
        f"{max_attempts} sampled representations."
    )


def _reflect_quiver_at_source_or_sink(quiver: Quiver, vertex: int) -> Quiver:
    if vertex not in quiver.get_sources() and vertex not in quiver.get_sinks():
        raise ValueError(f"Vertex {vertex} is neither a source nor a sink.")
    reflected = Quiver(name=quiver.name)
    for old_vertex in quiver.get_vertices():
        data = dict(quiver.vertices[old_vertex])
        label = data.pop("label", None)
        new_vertex = reflected.add_vertex(label=label, **data)
        if new_vertex != old_vertex:
            raise ValueError("Affine reflection requires stable consecutive vertex IDs.")
    for arrow in quiver.get_arrows():
        data = dict(quiver.arrows[arrow])
        source = data.pop("source")
        target = data.pop("target")
        label = data.pop("label", None)
        if source == vertex or target == vertex:
            source, target = target, source
        new_arrow = reflected.add_arrow(source, target, label=label, **data)
        if new_arrow != arrow:
            raise ValueError("Affine reflection requires stable consecutive arrow IDs.")
    return reflected


def _reflection_reduction(
    quiver: Quiver, dimension: Mapping[int, int]
) -> Tuple[Tuple[int, ...], Tuple[Quiver, ...], int]:
    vertices = tuple(quiver.get_vertices())
    memo = set()

    def recurse(current_quiver: Quiver, current_dimension: Dict[int, int]):
        support = [vertex for vertex in vertices if current_dimension[vertex]]
        if len(support) == 1 and current_dimension[support[0]] == 1:
            return (), (current_quiver,), support[0]
        signature = (
            tuple(
                (
                    current_quiver.arrows[arrow]["source"],
                    current_quiver.arrows[arrow]["target"],
                )
                for arrow in current_quiver.get_arrows()
            ),
            tuple(current_dimension[vertex] for vertex in vertices),
        )
        if signature in memo:
            return None
        memo.add(signature)

        eligible = sorted(
            set(current_quiver.get_sources()) | set(current_quiver.get_sinks()),
            key=lambda vertex: current_dimension[vertex],
            reverse=True,
        )
        for vertex in eligible:
            neighbor_sum = sum(
                current_dimension[neighbor]
                for neighbor in current_quiver.get_vertex_successors(vertex)
            ) + sum(
                current_dimension[neighbor]
                for neighbor in current_quiver.get_vertex_predecessors(vertex)
            )
            reflected_coordinate = neighbor_sum - current_dimension[vertex]
            if reflected_coordinate < 0 or reflected_coordinate >= current_dimension[vertex]:
                continue
            reflected_dimension = dict(current_dimension)
            reflected_dimension[vertex] = reflected_coordinate
            reflected_quiver = _reflect_quiver_at_source_or_sink(current_quiver, vertex)
            result = recurse(reflected_quiver, reflected_dimension)
            if result is not None:
                sequence, quivers, simple_vertex = result
                return (
                    (vertex,) + sequence,
                    (current_quiver,) + quivers,
                    simple_vertex,
                )
        return None

    result = recurse(quiver, dict(dimension))
    if result is not None:
        return result

    info = classify_affine_quiver(quiver)
    defect = quiver.euler_form(info.delta_dict(), dimension)
    if defect == 0:
        raise RuntimeError(
            "Could not reduce a regular real root by admissible source/sink reflections."
        )

    preferred_modes = ("sink", "source") if defect < 0 else ("source", "sink")
    for mode in preferred_modes:
        current_quiver = quiver
        current_dimension = dict(dimension)
        prefix_sequence = ()
        prefix_quivers = (quiver,)
        used = set()
        for _ in range(4 * len(vertices) * (sum(dimension.values()) + 1)):
            support = [vertex for vertex in vertices if current_dimension[vertex]]
            if len(support) == 1 and current_dimension[support[0]] == 1:
                return prefix_sequence, prefix_quivers, support[0]
            tail = recurse(current_quiver, current_dimension)
            if tail is not None:
                tail_sequence, tail_quivers, simple_vertex = tail
                return (
                    prefix_sequence + tail_sequence,
                    prefix_quivers + tail_quivers[1:],
                    simple_vertex,
                )

            eligible = (
                current_quiver.get_sinks() if mode == "sink" else current_quiver.get_sources()
            )
            vertex = next(
                (candidate for candidate in sorted(eligible) if candidate not in used),
                None,
            )
            if vertex is None:
                used.clear()
                vertex = next(iter(sorted(eligible)), None)
            if vertex is None:
                break
            neighbor_sum = sum(
                current_dimension[neighbor]
                for neighbor in current_quiver.get_vertex_successors(vertex)
            ) + sum(
                current_dimension[neighbor]
                for neighbor in current_quiver.get_vertex_predecessors(vertex)
            )
            reflected_coordinate = neighbor_sum - current_dimension[vertex]
            if reflected_coordinate < 0:
                break
            current_dimension = dict(current_dimension)
            current_dimension[vertex] = reflected_coordinate
            current_quiver = _reflect_quiver_at_source_or_sink(current_quiver, vertex)
            prefix_sequence += (vertex,)
            prefix_quivers += (current_quiver,)
            used.add(vertex)
            if len(used) == len(vertices):
                used.clear()

    initial_height = sum(dimension.values())
    maximum_height = initial_height + 4 * sum(info.delta)
    queue = [(quiver, dict(dimension), (), (quiver,))]
    seen = set()
    cursor = 0
    while cursor < len(queue):
        current_quiver, current_dimension, sequence, quivers = queue[cursor]
        cursor += 1
        signature = (
            tuple(
                (
                    current_quiver.arrows[arrow]["source"],
                    current_quiver.arrows[arrow]["target"],
                )
                for arrow in current_quiver.get_arrows()
            ),
            tuple(current_dimension[vertex] for vertex in vertices),
        )
        if signature in seen:
            continue
        seen.add(signature)
        support = [vertex for vertex in vertices if current_dimension[vertex]]
        if len(support) == 1 and current_dimension[support[0]] == 1:
            return sequence, quivers, support[0]

        candidates = []
        for vertex in set(current_quiver.get_sources()) | set(current_quiver.get_sinks()):
            neighbor_sum = sum(
                current_dimension[neighbor]
                for neighbor in current_quiver.get_vertex_successors(vertex)
            ) + sum(
                current_dimension[neighbor]
                for neighbor in current_quiver.get_vertex_predecessors(vertex)
            )
            reflected_coordinate = neighbor_sum - current_dimension[vertex]
            if reflected_coordinate < 0:
                continue
            reflected_dimension = dict(current_dimension)
            reflected_dimension[vertex] = reflected_coordinate
            height = sum(reflected_dimension.values())
            if height > maximum_height:
                continue
            candidates.append((height, vertex, reflected_dimension))
        for _, vertex, reflected_dimension in sorted(candidates):
            reflected_quiver = _reflect_quiver_at_source_or_sink(current_quiver, vertex)
            queue.append(
                (
                    reflected_quiver,
                    reflected_dimension,
                    sequence + (vertex,),
                    quivers + (reflected_quiver,),
                )
            )

    raise RuntimeError(
        "Could not reduce a nonzero-defect real root by admissible reflections "
        f"below height {maximum_height}."
    )


def _reflect_finite_module(
    module: Module, vertex: int, reflected_quiver: Quiver, name: str
) -> Module:
    quiver = module.quiver
    field = module.field
    is_sink = vertex in quiver.get_sinks()
    is_source = vertex in quiver.get_sources()
    if not is_sink and not is_source:
        raise ValueError(f"Vertex {vertex} is neither a source nor a sink.")

    dimensions = dict(module.spaces)
    maps = {}
    if is_sink:
        arrows = [
            arrow
            for arrow in quiver.get_arrows()
            if quiver.arrows[arrow]["target"] == vertex
        ]
        domain_dimensions = [
            module.spaces[quiver.arrows[arrow]["source"]] for arrow in arrows
        ]
        combined = field.zero_matrix(module.spaces[vertex], sum(domain_dimensions))
        offset = 0
        for arrow, width in zip(arrows, domain_dimensions):
            combined[:, offset : offset + width] = _matrix_or_zeros(module, arrow)
            offset += width
        kernel = field.kernel_basis(combined)
        dimensions[vertex] = kernel.shape[1]
        offset = 0
        for arrow, height in zip(arrows, domain_dimensions):
            maps[arrow] = kernel[offset : offset + height, :]
            offset += height
    else:
        arrows = [
            arrow
            for arrow in quiver.get_arrows()
            if quiver.arrows[arrow]["source"] == vertex
        ]
        codomain_dimensions = [
            module.spaces[quiver.arrows[arrow]["target"]] for arrow in arrows
        ]
        combined = field.zero_matrix(sum(codomain_dimensions), module.spaces[vertex])
        offset = 0
        for arrow, height in zip(arrows, codomain_dimensions):
            combined[offset : offset + height, :] = _matrix_or_zeros(module, arrow)
            offset += height
        quotient = field.left_kernel_basis(combined)
        dimensions[vertex] = quotient.shape[0]
        offset = 0
        for arrow, width in zip(arrows, codomain_dimensions):
            maps[arrow] = quotient[:, offset : offset + width]
            offset += width

    incident = set(arrows)
    for arrow in quiver.get_arrows():
        if arrow not in incident:
            maps[arrow] = _matrix_or_zeros(module, arrow)
    return Module(
        reflected_quiver,
        field,
        name=name,
        dimensions=dimensions,
        maps=maps,
    )


def _real_root_module_by_reflections(
    quiver: Quiver,
    field: FiniteField,
    dimension: Mapping[int, int],
    label: str,
) -> Module:
    sequence, quivers, simple_vertex = _reflection_reduction(quiver, dimension)
    final_quiver = quivers[-1]
    simple_dimension = {
        vertex: int(vertex == simple_vertex) for vertex in final_quiver.get_vertices()
    }
    module = Module(
        final_quiver,
        field,
        name=label,
        dimensions=simple_dimension,
    )
    for index in range(len(sequence) - 1, -1, -1):
        module = _reflect_finite_module(
            module,
            sequence[index],
            quivers[index],
            label,
        )
    if dict(module.spaces) != dict(dimension):
        raise RuntimeError("BGP construction returned the wrong real-root dimension.")
    if _finite_end_dimension(module) != 1:
        raise RuntimeError("BGP construction did not return a Schur real-root module.")
    return module


def _morphism_is_invertible(morphism) -> bool:
    for vertex in morphism.quiver.get_vertices():
        matrix = morphism.maps[vertex]
        if isinstance(matrix, ZeroMap):
            if matrix.source_dim != matrix.target_dim:
                return False
            continue
        if not morphism.field.is_invertible(matrix):
            return False
    return True


def _isomorphic_schur_modules(left: Module, right: Module) -> bool:
    basis = _find_hom_basis(left, right)
    return len(basis) == 1 and _morphism_is_invertible(basis[0])


def _cycle_order(quiver: Quiver) -> Tuple[int, ...]:
    vertices = tuple(sorted(quiver.get_vertices()))
    adjacency = {vertex: set() for vertex in vertices}
    for arrow in quiver.get_arrows():
        source = quiver.arrows[arrow]["source"]
        target = quiver.arrows[arrow]["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)
    start = min(vertices)
    previous = None
    current = start
    order = []
    while current not in order:
        order.append(current)
        choices = sorted(neighbor for neighbor in adjacency[current] if neighbor != previous)
        next_vertex = choices[0] if len(order) == 1 else choices[0]
        previous, current = current, next_vertex
    if current != start or len(order) != len(vertices):
        raise ValueError("Malformed affine A cycle.")
    return tuple(order)


def _edge_arrow(quiver: Quiver, left: int, right: int) -> int:
    arrows = quiver.get_arrows_between(left, right)
    if arrows:
        return arrows[0]
    arrows = quiver.get_arrows_between(right, left)
    if arrows:
        return arrows[0]
    raise ValueError(f"Vertices {left} and {right} are not adjacent.")


def _affine_a_string_module(
    quiver: Quiver,
    field,
    cycle: Sequence[int],
    start: int,
    length: int,
    label: str,
) -> Module:
    positions = [cycle[(start + step) % len(cycle)] for step in range(length + 1)]
    dimensions = {vertex: 0 for vertex in quiver.get_vertices()}
    local_indices = []
    for vertex in positions:
        local_indices.append(dimensions[vertex])
        dimensions[vertex] += 1
    maps = {}
    for arrow in quiver.get_arrows():
        source = quiver.arrows[arrow]["source"]
        target = quiver.arrows[arrow]["target"]
        maps[arrow] = field.zero_matrix(dimensions[target], dimensions[source])
    for step in range(length):
        left = positions[step]
        right = positions[step + 1]
        arrow = _edge_arrow(quiver, left, right)
        source = quiver.arrows[arrow]["source"]
        if source == left:
            maps[arrow][local_indices[step + 1], local_indices[step]] = field.one
        else:
            maps[arrow][local_indices[step], local_indices[step + 1]] = field.one
    return Module(quiver, field, name=label, dimensions=dimensions, maps=maps)


def _affine_a_band_module(
    quiver: Quiver,
    field,
    quasi_length: int,
    parameter: int,
    label: str,
) -> Module:
    dimensions = {vertex: quasi_length for vertex in quiver.get_vertices()}
    arrows = tuple(quiver.get_arrows())
    maps = {arrow: field.identity_matrix(quasi_length) for arrow in arrows}
    jordan = field.zero_matrix(quasi_length, quasi_length)
    scalar = field.element(parameter)
    for index in range(quasi_length):
        jordan[index, index] = scalar
    for index in range(quasi_length - 1):
        jordan[index, index + 1] = field.one
    maps[arrows[0]] = jordan
    return Module(quiver, field, name=label, dimensions=dimensions, maps=maps)


def _enumerate_affine_a_records(
    quiver: Quiver,
    field: FiniteField,
    info: AffineQuiverInfo,
    coverage: Mapping[int, int],
) -> Tuple[AffineIndecRecord, ...]:
    cycle = _cycle_order(quiver)
    complex_field = ComplexNumbers()
    records = []
    maximum_length = sum(coverage.values()) - 1
    for length in range(maximum_length + 1):
        for start in range(len(cycle)):
            label = f"S({start},{length})"
            finite = _affine_a_string_module(quiver, field, cycle, start, length, label)
            dimension = tuple(finite.spaces[vertex] for vertex in info.vertices)
            if any(
                dimension[index] > coverage[vertex]
                for index, vertex in enumerate(info.vertices)
            ):
                continue
            complex_module = _affine_a_string_module(
                quiver, complex_field, cycle, start, length, label
            )
            records.append(
                AffineIndecRecord(label, "string", dimension, finite, complex_module)
            )

    maximum_regular_length = min(coverage.values(), default=0)
    for parameter in range(1, field.order):
        for quasi_length in range(1, maximum_regular_length + 1):
            label = f"B({parameter},{quasi_length})"
            finite = _affine_a_band_module(
                quiver, field, quasi_length, parameter, label
            )
            complex_module = _affine_a_band_module(
                quiver, complex_field, quasi_length, parameter, label
            )
            records.append(
                AffineIndecRecord(
                    label,
                    "band",
                    tuple(quasi_length for _ in info.vertices),
                    finite,
                    complex_module,
                    regular_seed=parameter,
                    quasi_length=quasi_length,
                )
            )
    return tuple(records)


def _enumerate_delta_schur_seeds(
    quiver: Quiver,
    field: FiniteField,
    info: AffineQuiverInfo,
    *,
    random_seed: int,
    max_attempts: int,
) -> Tuple[Tuple[Module, Module], ...]:
    dimension = info.delta_dict()
    expected = field.order + info.index
    entry_count = sum(
        dimension[quiver.arrows[arrow]["source"]]
        * dimension[quiver.arrows[arrow]["target"]]
        for arrow in quiver.get_arrows()
    )
    rng = random.Random(_stable_seed(quiver, dimension, random_seed + 104729))
    complex_field = ComplexNumbers()
    seeds = []
    for values in _candidate_value_vectors(field, entry_count, rng, max_attempts):
        label = f"R({len(seeds)},1)"
        candidate = _representation_from_values(quiver, field, dimension, values, label)
        if _finite_end_dimension(candidate) != 1:
            continue
        if any(_isomorphic_schur_modules(candidate, existing[0]) for existing in seeds):
            continue
        lifted = _lift_module(candidate, complex_field, name=label)
        if _exact_end_dimension(lifted) != 1:
            raise RuntimeError("A Schur module modulo p failed its characteristic-zero lift check.")
        seeds.append((candidate, lifted))
        if len(seeds) == expected:
            return tuple(seeds)
    raise RuntimeError(
        f"Found {len(seeds)} of the expected {expected} absolutely indecomposable "
        f"delta-dimensional modules after {max_attempts} samples."
    )


def _extension_module(
    submodule: Module,
    quotient: Module,
    cocycles: Mapping[int, np.ndarray],
    name: str,
) -> Module:
    quiver = submodule.quiver
    field = submodule.field
    dimensions = {
        vertex: submodule.spaces[vertex] + quotient.spaces[vertex]
        for vertex in quiver.get_vertices()
    }
    maps = {}
    for arrow in quiver.get_arrows():
        source = quiver.arrows[arrow]["source"]
        target = quiver.arrows[arrow]["target"]
        sub_source = submodule.spaces[source]
        sub_target = submodule.spaces[target]
        quotient_source = quotient.spaces[source]
        quotient_target = quotient.spaces[target]
        matrix = field.zero_matrix(dimensions[target], dimensions[source])
        matrix[:sub_target, :sub_source] = _matrix_or_zeros(submodule, arrow)
        matrix[:sub_target, sub_source:] = cocycles[arrow]
        matrix[sub_target:, sub_source:] = _matrix_or_zeros(quotient, arrow)
        maps[arrow] = matrix
    return Module(quiver, field, name=name, dimensions=dimensions, maps=maps)


def _coboundary_matrix(submodule: Module, quotient: Module) -> np.ndarray:
    quiver = submodule.quiver
    vertices = list(quiver.get_vertices())
    starts = {}
    total_columns = 0
    for vertex in vertices:
        starts[vertex] = total_columns
        total_columns += submodule.spaces[vertex] * quotient.spaces[vertex]
    total_rows = sum(
        submodule.spaces[quiver.arrows[arrow]["target"]]
        * quotient.spaces[quiver.arrows[arrow]["source"]]
        for arrow in quiver.get_arrows()
    )
    differential = np.zeros((total_rows, total_columns), dtype=int)
    row_offset = 0
    for arrow in quiver.get_arrows():
        source = quiver.arrows[arrow]["source"]
        target = quiver.arrows[arrow]["target"]
        sub_source = submodule.spaces[source]
        sub_target = submodule.spaces[target]
        quotient_source = quotient.spaces[source]
        quotient_target = quotient.spaces[target]
        rows_here = sub_target * quotient_source
        left = np.kron(
            np.eye(quotient_source, dtype=int),
            _matrix_or_zeros(submodule, arrow, integral=True),
        )
        right = np.kron(
            _matrix_or_zeros(quotient, arrow, integral=True).T,
            np.eye(sub_target, dtype=int),
        )
        row_slice = slice(row_offset, row_offset + rows_here)
        differential[
            row_slice,
            starts[source] : starts[source] + sub_source * quotient_source,
        ] += left
        differential[
            row_slice,
            starts[target] : starts[target] + sub_target * quotient_target,
        ] -= right
        row_offset += rows_here
    return differential


def _extend_regular_seed(
    previous_finite: Module,
    seed_finite: Module,
    previous_complex: Module,
    seed_complex: Module,
    *,
    label: str,
    expected_end_dimension: int,
    rng: random.Random,
    max_attempts: int,
) -> Tuple[Module, Module]:
    quiver = previous_finite.quiver
    field = previous_finite.field
    cocycle_size = sum(
        previous_finite.spaces[quiver.arrows[arrow]["target"]]
        * seed_finite.spaces[quiver.arrows[arrow]["source"]]
        for arrow in quiver.get_arrows()
    )
    coboundary = _coboundary_matrix(previous_complex, seed_complex)
    coboundary_rank = _exact_rank(coboundary)

    for values_tuple in _candidate_value_vectors(
        field, cocycle_size, rng, max_attempts
    ):
        values = list(values_tuple)
        balanced = [value - field.order if value > field.order // 2 else value for value in values]
        finite_cocycles = {}
        complex_cocycles = {}
        offset = 0
        for arrow in quiver.get_arrows():
            source = quiver.arrows[arrow]["source"]
            target = quiver.arrows[arrow]["target"]
            rows = previous_finite.spaces[target]
            columns = seed_finite.spaces[source]
            size = rows * columns
            finite_cocycles[arrow] = field.matrix(values[offset : offset + size]).reshape(
                (rows, columns), order="F"
            )
            complex_cocycles[arrow] = seed_complex.field.matrix(
                balanced[offset : offset + size]
            ).reshape((rows, columns), order="F")
            offset += size

        finite_extension = _extension_module(
            previous_finite, seed_finite, finite_cocycles, label
        )
        if _finite_end_dimension(finite_extension) != expected_end_dimension:
            continue
        complex_extension = _extension_module(
            previous_complex, seed_complex, complex_cocycles, label
        )
        if _exact_end_dimension(complex_extension) != expected_end_dimension:
            continue
        cocycle_column = np.asarray(balanced, dtype=int).reshape((-1, 1))
        augmented = np.hstack((coboundary, cocycle_column))
        if _exact_rank(augmented) == coboundary_rank:
            continue
        return finite_extension, complex_extension
    raise RuntimeError(
        f"Could not construct the nonsplit regular extension {label} after "
        f"{max_attempts} attempts."
    )


def _extend_real_root_by_delta(
    base_finite: Module,
    base_complex: Module,
    seeds: Sequence[Tuple[Module, Module]],
    *,
    label: str,
    dimension: Mapping[int, int],
    expected_end_dimension: int,
    random_seed: int,
    max_attempts: int,
) -> Tuple[Module, Module]:
    quiver = base_finite.quiver
    for seed_index, (seed_finite, seed_complex) in enumerate(seeds):
        for direction in range(2):
            if direction == 0:
                sub_finite, quotient_finite = base_finite, seed_finite
                sub_complex, quotient_complex = base_complex, seed_complex
            else:
                sub_finite, quotient_finite = seed_finite, base_finite
                sub_complex, quotient_complex = seed_complex, base_complex
            ext_dimension = _finite_hom_dimension(quotient_finite, sub_finite) - quiver.euler_form(
                quotient_finite.spaces, sub_finite.spaces
            )
            if ext_dimension <= 0:
                continue
            rng = random.Random(
                _stable_seed(
                    quiver,
                    dimension,
                    random_seed + 65537 * seed_index + direction,
                )
            )
            try:
                finite, complex_module = _extend_regular_seed(
                    sub_finite,
                    quotient_finite,
                    sub_complex,
                    quotient_complex,
                    label=label,
                    expected_end_dimension=expected_end_dimension,
                    rng=rng,
                    max_attempts=max_attempts,
                )
            except RuntimeError:
                continue
            if dict(finite.spaces) == dict(dimension):
                return finite, complex_module
    raise RuntimeError(
        f"Could not construct the translated real-root indecomposable {label} "
        "from any rational delta seed."
    )


def _enumerate_affine_d_records(
    quiver: Quiver,
    field: FiniteField,
    info: AffineQuiverInfo,
    coverage: Mapping[int, int],
    *,
    random_seed: int,
    max_attempts: int,
) -> Tuple[AffineIndecRecord, ...]:
    complex_field = ComplexNumbers()
    records = []
    maximum_delta_multiple = min(
        coverage[vertex] // delta for vertex, delta in zip(info.vertices, info.delta)
    )
    seeds = ()
    if maximum_delta_multiple:
        seeds = _enumerate_delta_schur_seeds(
            quiver,
            field,
            info,
            random_seed=random_seed,
            max_attempts=max_attempts,
        )

    roots = sorted(
        enumerate_affine_real_roots_bounded(quiver, coverage),
        key=lambda root: (sum(root.values()), tuple(root[vertex] for vertex in info.vertices)),
    )
    root_modules = {}
    for dimension in roots:
        dimension_tuple = tuple(dimension[vertex] for vertex in info.vertices)
        label = "X(" + ",".join(map(str, dimension_tuple)) + ")"
        try:
            finite = _real_root_module_by_reflections(quiver, field, dimension, label)
            complex_module = _lift_module(finite, complex_field, name=label)
            expected_end_dimension = 1
        except RuntimeError:
            smaller_tuple = tuple(
                coordinate - delta
                for coordinate, delta in zip(dimension_tuple, info.delta)
            )
            if any(coordinate < 0 for coordinate in smaller_tuple):
                finite = _random_schur_module(
                    quiver,
                    field,
                    dimension,
                    label=label,
                    random_seed=random_seed,
                    max_attempts=max_attempts,
                )
                complex_module = _lift_module(finite, complex_field, name=label)
                expected_end_dimension = 1
            else:
                if smaller_tuple not in root_modules:
                    raise RuntimeError(
                        f"Missing the preceding real root {smaller_tuple} needed for {label}."
                    )
                base_finite, base_complex = root_modules[smaller_tuple]
                defect = quiver.euler_form(info.delta_dict(), dimension)
                expected_end_dimension = (
                    _exact_end_dimension(base_complex) + 1 if defect == 0 else 1
                )
                finite, complex_module = _extend_real_root_by_delta(
                    base_finite,
                    base_complex,
                    seeds,
                    label=label,
                    dimension=dimension,
                    expected_end_dimension=expected_end_dimension,
                    random_seed=random_seed,
                    max_attempts=max_attempts,
                )
        if _exact_end_dimension(complex_module) != expected_end_dimension:
            raise RuntimeError(
                f"Characteristic-zero lift of {label} has the wrong endomorphism dimension."
            )
        root_modules[dimension_tuple] = (finite, complex_module)
        records.append(
            AffineIndecRecord(label, "real_root", dimension_tuple, finite, complex_module)
        )

    if maximum_delta_multiple == 0:
        return tuple(records)
    for seed_index, (seed_finite, seed_complex) in enumerate(seeds):
        previous_finite = seed_finite
        previous_complex = seed_complex
        for multiple in range(1, maximum_delta_multiple + 1):
            label = f"R({seed_index},{multiple})"
            if multiple > 1:
                rng = random.Random(
                    _stable_seed(
                        quiver,
                        info.delta_dict(),
                        random_seed + 1000003 * seed_index + multiple,
                    )
                )
                previous_finite, previous_complex = _extend_regular_seed(
                    previous_finite,
                    seed_finite,
                    previous_complex,
                    seed_complex,
                    label=label,
                    expected_end_dimension=multiple,
                    rng=rng,
                    max_attempts=max_attempts,
                )
            else:
                previous_finite.name = label
                previous_complex.name = label
            records.append(
                AffineIndecRecord(
                    label,
                    "regular_rational",
                    tuple(multiple * delta for delta in info.delta),
                    previous_finite,
                    previous_complex,
                    regular_seed=seed_index,
                    quasi_length=multiple,
                )
            )
    return tuple(records)


def enumerate_affine_ad_indecs_bounded(
    quiver: Quiver,
    field: FiniteField,
    coverage: Mapping[int, int],
    *,
    regular_mode: KroneckerRegularMode | str = KroneckerRegularMode.RATIONAL_JORDAN,
    random_seed: int = 0,
    max_attempts: int = 50000,
) -> Tuple[AffineIndecRecord, ...]:
    """Enumerate the rational-Jordan affine A/D indecomposables within ``coverage``."""
    info = classify_affine_quiver(quiver)
    if info.family not in ("A", "D"):
        raise ValueError("This catalogue supports affine A and affine D quivers.")
    if info.family == "A" and info.index == 1:
        raise ValueError("Use the dedicated Kronecker catalogue for affine A1.")
    if field.degree != 1:
        raise NotImplementedError("Affine A/D catalogues currently require a prime field GF(p).")
    mode = _coerce_mode(regular_mode)
    if mode == KroneckerRegularMode.CLOSED_POINTS:
        # TODO: add irreducible-polynomial tubes and packet-valued characteristic-zero lifts.
        raise NotImplementedError(
            "closed_points mode is reserved for irreducible-polynomial regular tubes; "
            "use regular_mode='rational_jordan' for the implemented catalogue."
        )
    dense = _dense_coverage(info, coverage)
    if info.family == "A":
        return _enumerate_affine_a_records(quiver, field, info, dense)
    return _enumerate_affine_d_records(
        quiver,
        field,
        info,
        dense,
        random_seed=random_seed,
        max_attempts=max_attempts,
    )


def enumerate_affine_bags_from_records_iter(
    records: Sequence[AffineIndecRecord], coverage: Mapping[int, int], vertices: Sequence[int]
) -> Iterator[AffineBag]:
    target = tuple(int(coverage.get(vertex, 0)) for vertex in vertices)
    bag = []

    def recurse(index: int, remaining: Tuple[int, ...]):
        if not any(remaining):
            yield tuple(bag)
            return
        if index == len(records):
            return
        dimension = records[index].dimension
        limits = [
            remaining[position] // value
            for position, value in enumerate(dimension)
            if value
        ]
        maximum = min(limits) if limits else 0
        for multiplicity in range(maximum + 1):
            if multiplicity:
                bag.extend([records[index]] * multiplicity)
            yield from recurse(
                index + 1,
                tuple(
                    remaining[position] - multiplicity * dimension[position]
                    for position in range(len(vertices))
                ),
            )
            if multiplicity:
                del bag[-multiplicity:]

    yield from recurse(0, target)


def _direct_sum_records(
    quiver: Quiver, field, bag: Sequence[AffineIndecRecord], attribute: str
) -> Module:
    modules = [getattr(record, attribute) for record in bag]
    dimensions = {
        vertex: sum(module.spaces[vertex] for module in modules)
        for vertex in quiver.get_vertices()
    }
    maps = {}
    for arrow in quiver.get_arrows():
        source = quiver.arrows[arrow]["source"]
        target = quiver.arrows[arrow]["target"]
        matrix = field.zero_matrix(dimensions[target], dimensions[source])
        source_offset = 0
        target_offset = 0
        for module in modules:
            source_dimension = module.spaces[source]
            target_dimension = module.spaces[target]
            block = _matrix_or_zeros(module, arrow)
            matrix[
                target_offset : target_offset + target_dimension,
                source_offset : source_offset + source_dimension,
            ] = block
            source_offset += source_dimension
            target_offset += target_dimension
        maps[arrow] = matrix
    name = " + ".join(record.label for record in bag) or "0"
    return Module(quiver, field, name=name, dimensions=dimensions, maps=maps)


def make_affine_ad_jobs(
    quiver: Quiver,
    field: FiniteField,
    coverage: Mapping[int, int],
    target_dimension: Mapping[int, int],
    *,
    regular_mode: KroneckerRegularMode | str = KroneckerRegularMode.RATIONAL_JORDAN,
    random_seed: int = 0,
    max_attempts: int = 50000,
) -> Tuple[Tuple[AffineIndecRecord, ...], list[AffineJob]]:
    records = enumerate_affine_ad_indecs_bounded(
        quiver,
        field,
        coverage,
        regular_mode=regular_mode,
        random_seed=random_seed,
        max_attempts=max_attempts,
    )
    vertices = tuple(sorted(quiver.get_vertices()))
    target = {vertex: int(target_dimension.get(vertex, 0)) for vertex in vertices}
    complex_field = records[0].complex_module.field if records else ComplexNumbers()
    jobs = []
    for bag in enumerate_affine_bags_from_records_iter(records, coverage, vertices):
        jobs.append(
            AffineJob(
                bag=bag,
                finite_module=_direct_sum_records(quiver, field, bag, "finite_module"),
                complex_module=_direct_sum_records(
                    quiver, complex_field, bag, "complex_module"
                ),
                target_dimension=target,
            )
        )
    return records, jobs


def write_batch_rad_from_affine_jobs(jobs: Sequence[AffineJob], **batch_options) -> Path:
    """Generate the existing QQ Macaulay2 jobs from characteristic-zero lifts."""
    return write_batch_rad_from_triples(
        [job.as_rad_job() for job in jobs],
        **batch_options,
    )
