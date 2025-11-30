from __future__ import annotations

from typing import Optional, Tuple, Type, TYPE_CHECKING

import numpy as np

from .field import ZeroMap
from .quiver import topological_sort

if TYPE_CHECKING:
    from .module import Module
    from .morphism import Morphism

MorphismType = Type["Morphism"]
ModuleType = Type["Module"]


def _resolve_module_type(module: "Module", module_type: Optional[ModuleType]) -> ModuleType:
    return module_type if module_type is not None else type(module)


def _resolve_morphism_type(module: "Module", morphism_type: Optional[MorphismType]) -> MorphismType:
    if morphism_type is not None:
        return morphism_type

    from .morphism import Morphism  # Local import to avoid circular dependency

    return Morphism


def compute_radical(
    module: "Module",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism"]:
    """Compute the radical of ``module``."""

    morphism_cls = _resolve_morphism_type(module, morphism_type)
    module_cls = _resolve_module_type(module, module_type)

    # Gather dimensions, bases, and maps before creating the module
    radical_dimensions = {}
    inclusion_maps = {}
    radical_maps = {}
    radical_bases = {}

    # First pass: compute dimensions and bases at each vertex
    for vertex_id in module.quiver.get_vertices():
        target_dim = module.spaces[vertex_id]

        if target_dim == 0:
            radical_dimensions[vertex_id] = 0
            continue

        incoming_arrows = [
            (source_id, arrow_id)
            for source_id, arrow_id in module.quiver.predecessors[vertex_id]
            if source_id != vertex_id
        ]

        if not incoming_arrows:
            radical_dimensions[vertex_id] = 0
            continue

        matrices_to_concat = []
        for source_id, arrow_id in incoming_arrows:
            arrow_map = module.maps[arrow_id]

            if isinstance(arrow_map, ZeroMap):
                continue

            if module.spaces[source_id] == 0:
                continue

            matrices_to_concat.append(arrow_map)

        if not matrices_to_concat:
            radical_dimensions[vertex_id] = 0
            continue

        concat_matrix = np.hstack(matrices_to_concat)

        radical_basis = module.field.column_space_basis(concat_matrix)

        radical_dim = radical_basis.shape[1]
        radical_dimensions[vertex_id] = radical_dim

        radical_bases[vertex_id] = radical_basis
        inclusion_maps[vertex_id] = radical_basis

    # Second pass: compute maps between radical spaces
    for arrow_id in module.quiver.get_arrows():
        arrow = module.quiver.arrows[arrow_id]
        source_vertex = arrow["source"]
        target_vertex = arrow["target"]

        source_radical_dim = radical_dimensions[source_vertex]
        target_radical_dim = radical_dimensions[target_vertex]

        if source_radical_dim == 0 or target_radical_dim == 0:
            radical_maps[arrow_id] = ZeroMap(source_radical_dim, target_radical_dim)
            continue

        original_map = module.maps[arrow_id]
        if original_map is None or isinstance(original_map, ZeroMap):
            radical_maps[arrow_id] = module.field.zero_matrix(target_radical_dim, source_radical_dim)
            continue

        source_basis = radical_bases[source_vertex]
        target_basis = radical_bases[target_vertex]

        mapped_source = original_map @ source_basis

        radical_map = module.field.find_matrix_coordinates(target_basis, mapped_source)
        if radical_map is None:
            raise ValueError(f"Failed to compute radical map for arrow {arrow_id}")

        radical_maps[arrow_id] = radical_map

    radical_module = module_cls(
        module.quiver,
        module.field,
        name=f"rad({module.name})",
        dimensions=radical_dimensions,
        maps=radical_maps,
    )

    inclusion = morphism_cls(radical_module, module, name=f"incl_rad_{module.name}")

    for vertex_id in module.quiver.get_vertices():
        radical_dim = radical_dimensions[vertex_id]
        original_dim = module.spaces[vertex_id]

        if radical_dim == 0 or original_dim == 0:
            inclusion.maps[vertex_id] = ZeroMap(radical_dim, original_dim)
        else:
            inclusion.maps[vertex_id] = inclusion_maps[vertex_id]

    return radical_module, inclusion


def compute_socle(
    module: "Module",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism"]:
    """Compute the socle of ``module``."""

    morphism_cls = _resolve_morphism_type(module, morphism_type)
    module_cls = _resolve_module_type(module, module_type)

    socle_dimensions = {}
    inclusion_maps = {}
    socle_maps = {}
    socle_bases = {}

    for vertex_id in module.quiver.get_vertices():
        source_dim = module.spaces[vertex_id]

        if source_dim == 0:
            socle_dimensions[vertex_id] = 0
            continue

        outgoing_arrows = [
            (target_id, arrow_id)
            for target_id, arrow_id in module.quiver.successors[vertex_id]
            if target_id != vertex_id
        ]

        if not outgoing_arrows:
            socle_dimensions[vertex_id] = source_dim
            identity = module.field.identity_matrix(source_dim)
            socle_bases[vertex_id] = identity
            inclusion_maps[vertex_id] = identity
            continue

        matrices_to_stack = []
        for target_id, arrow_id in outgoing_arrows:
            arrow_map = module.maps[arrow_id]

            if isinstance(arrow_map, ZeroMap):
                continue

            if module.spaces[target_id] == 0:
                continue

            matrices_to_stack.append(arrow_map)

        if not matrices_to_stack:
            socle_dimensions[vertex_id] = source_dim
            identity = module.field.identity_matrix(source_dim)
            socle_bases[vertex_id] = identity
            inclusion_maps[vertex_id] = identity
            continue

        stacked_matrix = np.vstack(matrices_to_stack)

        socle_basis = module.field.kernel_basis(stacked_matrix)

        socle_dim = socle_basis.shape[1]
        socle_dimensions[vertex_id] = socle_dim

        socle_bases[vertex_id] = socle_basis
        inclusion_maps[vertex_id] = socle_basis

    for arrow_id in module.quiver.get_arrows():
        arrow = module.quiver.arrows[arrow_id]
        source_vertex = arrow["source"]
        target_vertex = arrow["target"]

        source_socle_dim = socle_dimensions[source_vertex]
        target_socle_dim = socle_dimensions[target_vertex]

        if source_socle_dim == 0 or target_socle_dim == 0:
            socle_maps[arrow_id] = ZeroMap(source_socle_dim, target_socle_dim)
            continue

        original_map = module.maps[arrow_id]
        if original_map is None or isinstance(original_map, ZeroMap):
            socle_maps[arrow_id] = module.field.zero_matrix(target_socle_dim, source_socle_dim)
            continue

        source_basis = socle_bases[source_vertex]
        target_basis = socle_bases[target_vertex]

        mapped_source = original_map @ source_basis

        socle_map = module.field.find_matrix_coordinates(target_basis, mapped_source)
        if socle_map is None:
            raise ValueError(f"Failed to compute socle map for arrow {arrow_id}")

        socle_maps[arrow_id] = socle_map

    socle_module = module_cls(
        module.quiver,
        module.field,
        name=f"soc({module.name})",
        dimensions=socle_dimensions,
        maps=socle_maps,
    )

    inclusion = morphism_cls(socle_module, module, name=f"incl_soc_{module.name}")

    for vertex_id in module.quiver.get_vertices():
        socle_dim = socle_dimensions[vertex_id]
        original_dim = module.spaces[vertex_id]

        if socle_dim == 0 or original_dim == 0:
            inclusion.maps[vertex_id] = ZeroMap(socle_dim, original_dim)
        else:
            inclusion.maps[vertex_id] = inclusion_maps[vertex_id]

    return socle_module, inclusion


def compute_projective_cover(
    module: "Module",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism"]:
    """Compute the projective cover of ``module``."""

    morphism_cls = _resolve_morphism_type(module, morphism_type)
    module_cls = _resolve_module_type(module, module_type)

    rad_module, inclusion = compute_radical(
        module,
        module_type=module_cls,
        morphism_type=morphism_cls,
    )
    quotient_module, projection = inclusion.cokernel()

    quotient_dimensions = quotient_module.get_dimension_vector()

    vertex_to_projectives = {}
    vertex_to_proj_p = {}

    for vertex_id, dim in quotient_dimensions.items():
        if dim > 0:
            proj_v, _, cover_v = module_cls.projective(module.quiver, module.field, vertex_id)
            proj_v_pow, _, cover_v_pow = morphism_cls.direct_power(cover_v, dim)
            vertex_to_projectives[vertex_id] = proj_v_pow
            vertex_to_proj_p[vertex_id] = cover_v_pow

    if not vertex_to_projectives:
        total_dim = module.get_total_dimension()
        if total_dim == 0:
            zero_dimensions = {v: 0 for v in module.quiver.get_vertices()}
            proj_cover = module_cls(module.quiver, module.field, "Zero", dimensions=zero_dimensions)
            cover_morphism = morphism_cls(module, proj_cover, "Zero morphism")

            for vertex_id in module.quiver.get_vertices():
                cover_morphism.set_map(vertex_id, ZeroMap(0, 0))

            return proj_cover, cover_morphism

        raise ValueError("Non-zero module coincides with its radical. That's a contradiction.")

    all_projective_covers = list(vertex_to_proj_p.values())

    if len(all_projective_covers) == 1:
        pi_P = all_projective_covers[0]
    else:
        pi_P = all_projective_covers[0]
        for next_cover in all_projective_covers[1:]:
            combined = morphism_cls.direct_sum(pi_P, next_cover)
            pi_P = combined[2]

    proj_cover = pi_P.source

    cover_morphism = morphism_cls(proj_cover, module, f"Projective cover of {module.name}")

    topological_order = topological_sort(module.quiver)

    defined_vertices = set()

    for vertex_id in topological_order:
        source_dim = proj_cover.spaces.get(vertex_id, 0)
        target_dim = module.spaces.get(vertex_id, 0)

        if source_dim == 0:
            cover_morphism.set_map(vertex_id, ZeroMap(source_dim, target_dim))
            continue

        constraints_left = []
        constraints_right = []

        quotient_dim = quotient_module.spaces.get(vertex_id, 0)
        if quotient_dim > 0:
            pi_v = projection.get_map(vertex_id)
            constraints_left.append(pi_v)
            constraints_right.append(pi_P.get_map(vertex_id))

        for source, arrow_id in module.quiver.predecessors[vertex_id]:
            if source not in defined_vertices:
                raise ValueError("We haven't defined the morphism at the source yet")

            q_source = cover_morphism.get_map(source)

            module_map = module.maps.get(arrow_id)
            proj_map = proj_cover.maps.get(arrow_id)

            if (
                module_map is None
                or isinstance(module_map, ZeroMap)
                or proj_map is None
                or isinstance(proj_map, ZeroMap)
            ):
                continue

            right_side = module_map @ q_source
            constraints_left.append(proj_map)
            constraints_right.append(right_side)

        if not constraints_left:
            q_v = module.field.zero_matrix(target_dim, source_dim)
            cover_morphism.set_map(vertex_id, q_v)
            defined_vertices.add(vertex_id)
            continue

        coef_matrix = module.field.matrix(np.hstack(constraints_left))
        rhs_matrix = module.field.matrix(np.hstack(constraints_right))

        coef_matrix_T = coef_matrix.T
        rhs_matrix_T = rhs_matrix.T

        solution = module.field.find_matrix_coordinates(coef_matrix_T, rhs_matrix_T)

        if solution is None:
            raise ValueError(
                f"Failed to find a morphism satisfying all constraints at vertex {vertex_id}."
            )

        q_v = solution.T

        cover_morphism.set_map(vertex_id, q_v)

        defined_vertices.add(vertex_id)

    return proj_cover, cover_morphism


def compute_injective_hull(
    module: "Module",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism"]:
    """Compute the injective hull of ``module``."""

    morphism_cls = _resolve_morphism_type(module, morphism_type)
    module_cls = _resolve_module_type(module, module_type)

    socle_module, inclusion = compute_socle(
        module,
        module_type=module_cls,
        morphism_type=morphism_cls,
    )

    socle_dimensions = socle_module.get_dimension_vector()

    vertex_to_injectives = {}
    vertex_to_inj_i = {}

    for vertex_id, dim in socle_dimensions.items():
        if dim > 0:
            inj_v, _, inclusion_v = module_cls.injective(module.quiver, module.field, vertex_id)
            _, inj_v_pow, inclusion_v_pow = morphism_cls.direct_power(inclusion_v, dim)
            vertex_to_injectives[vertex_id] = inj_v_pow
            vertex_to_inj_i[vertex_id] = inclusion_v_pow

    if not vertex_to_injectives:
        total_dim = module.get_total_dimension()
        if total_dim == 0:
            zero_dimensions = {v: 0 for v in module.quiver.get_vertices()}
            inj_hull = module_cls(module.quiver, module.field, "Zero", dimensions=zero_dimensions)
            hull_morphism = morphism_cls(module, inj_hull, "Zero morphism")

            for vertex_id in module.quiver.get_vertices():
                hull_morphism.set_map(vertex_id, ZeroMap(0, 0))

            return inj_hull, hull_morphism

        raise ValueError("Non-zero module has zero socle. Every non-zero module should have a non-zero socle.")

    all_injective_inclusions = list(vertex_to_inj_i.values())

    if len(all_injective_inclusions) == 1:
        i_I = all_injective_inclusions[0]
    else:
        i_I = all_injective_inclusions[0]
        for next_inclusion in all_injective_inclusions[1:]:
            combined = morphism_cls.direct_sum(i_I, next_inclusion)
            i_I = combined[2]

    inj_hull = i_I.target

    hull_morphism = morphism_cls(module, inj_hull, f"Injective hull of {module.name}")

    topological_order = list(reversed(topological_sort(module.quiver)))

    defined_vertices = set()

    for vertex_id in topological_order:
        source_dim = module.spaces.get(vertex_id, 0)
        target_dim = inj_hull.spaces.get(vertex_id, 0)

        if target_dim == 0:
            hull_morphism.set_map(vertex_id, ZeroMap(source_dim, target_dim))
            continue

        constraints_left = []
        constraints_right = []

        socle_dim = socle_module.spaces.get(vertex_id, 0)
        if socle_dim > 0:
            i_v = inclusion.get_map(vertex_id)
            constraints_left.append(i_v)
            constraints_right.append(i_I.get_map(vertex_id))

        for target, arrow_id in module.quiver.successors[vertex_id]:
            if target not in defined_vertices:
                raise ValueError("We haven't defined the morphism at the target vertex yet")

            i_target = hull_morphism.get_map(target)

            module_map = module.maps.get(arrow_id)
            inj_map = inj_hull.maps.get(arrow_id)

            if (
                module_map is None
                or isinstance(module_map, ZeroMap)
                or inj_map is None
                or isinstance(inj_map, ZeroMap)
            ):
                continue

            right_side = i_target @ module_map
            constraints_left.append(inj_map)
            constraints_right.append(right_side)

        if not constraints_left:
            i_v = module.field.zero_matrix(target_dim, source_dim)
            hull_morphism.set_map(vertex_id, i_v)
            defined_vertices.add(vertex_id)
            continue

        coef_matrix = module.field.matrix(np.vstack(constraints_left))
        rhs_matrix = module.field.matrix(np.vstack(constraints_right))

        solution = module.field.find_matrix_coordinates(coef_matrix, rhs_matrix)

        if solution is None:
            raise ValueError(
                f"Failed to find a morphism satisfying all constraints at vertex {vertex_id}."
            )

        i_v = solution

        hull_morphism.set_map(vertex_id, i_v)

        defined_vertices.add(vertex_id)

    return inj_hull, hull_morphism

