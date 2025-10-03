from __future__ import annotations

from typing import Optional, Tuple, Type, TYPE_CHECKING

import numpy as np

from .field import ZeroMap

if TYPE_CHECKING:
    from .module import Module
    from .morphism import Morphism

ModuleType = Type["Module"]
MorphismType = Type["Morphism"]


def _resolve_module_type(morphism: "Morphism", module_type: Optional[ModuleType]) -> ModuleType:
    if module_type is not None:
        return module_type
    return type(morphism.source)


def _resolve_morphism_type(morphism: "Morphism", morphism_type: Optional[MorphismType]) -> MorphismType:
    if morphism_type is not None:
        return morphism_type

    from .morphism import Morphism  # Local import to avoid circular dependency

    return Morphism


def compute_kernel(
    morphism: "Morphism",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism"]:
    """Compute the kernel of ``morphism``."""

    if not morphism.is_valid():
        raise ValueError("Morphism is not valid")

    module_cls = _resolve_module_type(morphism, module_type)
    morphism_cls = _resolve_morphism_type(morphism, morphism_type)

    kernel_dimensions = {}
    kernel_bases = {}

    for vertex_id in morphism.quiver.get_vertices():
        source_dim = morphism.source.spaces[vertex_id]
        target_dim = morphism.target.spaces[vertex_id]

        if source_dim == 0:
            kernel_dimensions[vertex_id] = 0
            kernel_bases[vertex_id] = None
            continue

        vertex_map = morphism.maps.get(vertex_id)

        if vertex_map is None or isinstance(vertex_map, ZeroMap):
            kernel_dimensions[vertex_id] = source_dim
            kernel_bases[vertex_id] = morphism.field.identity_matrix(source_dim)
            continue

        if target_dim == 0:
            kernel_dimensions[vertex_id] = source_dim
            kernel_bases[vertex_id] = morphism.field.identity_matrix(source_dim)
            continue

        try:
            kernel_basis = morphism.field.kernel_basis(vertex_map)
            kernel_dim = kernel_basis.shape[1]
        except (AttributeError, Exception):  # pragma: no cover - preserve legacy behavior
            print("Kernel dimension computation failed, for some mysterious reason")
            kernel_basis = morphism.field.kernel_basis(vertex_map)
            kernel_dim = kernel_basis.shape[1]

        kernel_dimensions[vertex_id] = kernel_dim
        kernel_bases[vertex_id] = kernel_basis

    kernel_maps = {}

    for arrow_id in morphism.quiver.get_arrows():
        arrow = morphism.quiver.arrows[arrow_id]
        source_vertex = arrow["source"]
        target_vertex = arrow["target"]

        source_kernel_dim = kernel_dimensions[source_vertex]
        target_kernel_dim = kernel_dimensions[target_vertex]

        if source_kernel_dim == 0 or target_kernel_dim == 0:
            kernel_maps[arrow_id] = ZeroMap(source_kernel_dim, target_kernel_dim)
            continue

        source_arrow_map = morphism.source.maps[arrow_id]

        if source_arrow_map is None or isinstance(source_arrow_map, ZeroMap):
            kernel_maps[arrow_id] = ZeroMap(target_kernel_dim, source_kernel_dim)
            continue

        source_basis = kernel_bases[source_vertex]
        target_basis = kernel_bases[target_vertex]

        intermediate = source_arrow_map @ source_basis

        kernel_maps[arrow_id] = morphism.field.find_matrix_coordinates(target_basis, intermediate)

    kernel_module = module_cls(
        morphism.quiver,
        morphism.field,
        name=f"Ker({morphism.name})",
        dimensions=kernel_dimensions,
        maps=kernel_maps,
    )

    inclusion = morphism_cls(kernel_module, morphism.source, name=f"inc_{morphism.name}")

    for vertex_id in morphism.quiver.get_vertices():
        kernel_dim = kernel_dimensions[vertex_id]
        source_dim = morphism.source.spaces[vertex_id]

        if kernel_dim == 0 or source_dim == 0:
            inclusion.maps[vertex_id] = ZeroMap(kernel_dim, source_dim)
            continue

        basis = kernel_bases[vertex_id]

        if basis.shape[0] != source_dim:
            basis = basis.T

        if basis.shape != (source_dim, kernel_dim):
            raise ValueError(
                f"Basis shape {basis.shape} doesn't match expected shape ({source_dim}, {kernel_dim})"
            )

        inclusion.maps[vertex_id] = basis

    return kernel_module, inclusion


def compute_image(
    morphism: "Morphism",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism", "Morphism"]:
    """Compute the image of ``morphism``."""

    if not morphism.is_valid():
        raise ValueError("Morphism is not valid")

    module_cls = _resolve_module_type(morphism, module_type)
    morphism_cls = _resolve_morphism_type(morphism, morphism_type)

    image_dimensions = {}
    image_maps = {}
    inclusion_maps = {}
    projection_maps = {}
    image_bases = {}

    for vertex_id in morphism.quiver.get_vertices():
        source_dim = morphism.source.spaces[vertex_id]
        target_dim = morphism.target.spaces[vertex_id]

        vertex_map = morphism.maps[vertex_id]

        if (
            source_dim == 0
            or target_dim == 0
            or vertex_map is None
            or isinstance(vertex_map, ZeroMap)
        ):
            image_dimensions[vertex_id] = 0
            continue

        image_basis = morphism.field.column_space_basis(vertex_map)
        if image_basis is None:
            raise ValueError(f"Failed to compute column space basis for vertex {vertex_id}")

        image_dim = image_basis.shape[1]
        image_dimensions[vertex_id] = image_dim

        image_bases[vertex_id] = image_basis
        inclusion_maps[vertex_id] = image_basis

        projection = morphism.field.find_matrix_coordinates(vertex_map, image_basis)
        if projection is None:
            raise ValueError(f"Failed to compute projection map for vertex {vertex_id}")

        projection_maps[vertex_id] = projection

    for arrow_id in morphism.quiver.get_arrows():
        arrow = morphism.quiver.arrows[arrow_id]
        source_vertex = arrow["source"]
        target_vertex = arrow["target"]

        source_image_dim = image_dimensions[source_vertex]
        target_image_dim = image_dimensions[target_vertex]

        if source_image_dim == 0 or target_image_dim == 0:
            image_maps[arrow_id] = ZeroMap(source_image_dim, target_image_dim)
            continue

        target_arrow_map = morphism.target.maps[arrow_id]

        if target_arrow_map is None or isinstance(target_arrow_map, ZeroMap):
            image_maps[arrow_id] = morphism.field.zero_matrix(target_image_dim, source_image_dim)
            continue

        source_basis = image_bases[source_vertex]
        target_basis = image_bases[target_vertex]

        mapped_source = target_arrow_map @ source_basis

        image_map = morphism.field.find_matrix_coordinates(target_basis, mapped_source)
        if image_map is None:
            raise ValueError(f"Failed to compute image map for arrow {arrow_id}")

        image_maps[arrow_id] = image_map

    image_module = module_cls(
        morphism.quiver,
        morphism.field,
        name=f"Im({morphism.name})",
        dimensions=image_dimensions,
        maps=image_maps,
    )

    inclusion = morphism_cls(image_module, morphism.target, name=f"incl_{morphism.name}")

    for vertex_id in morphism.quiver.get_vertices():
        image_dim = image_dimensions[vertex_id]
        target_dim = morphism.target.spaces[vertex_id]

        if image_dim == 0 or target_dim == 0:
            inclusion.maps[vertex_id] = ZeroMap(image_dim, target_dim)
        else:
            inclusion.maps[vertex_id] = inclusion_maps[vertex_id]

    projection = morphism_cls(morphism.source, image_module, name=f"proj_{morphism.name}")

    for vertex_id in morphism.quiver.get_vertices():
        source_dim = morphism.source.spaces[vertex_id]
        coimage_dim = image_dimensions[vertex_id]

        if source_dim == 0 or coimage_dim == 0:
            projection.maps[vertex_id] = ZeroMap(source_dim, coimage_dim)
        else:
            projection.maps[vertex_id] = projection_maps[vertex_id]

    return image_module, inclusion, projection


def compute_cokernel(
    morphism: "Morphism",
    *,
    module_type: Optional[ModuleType] = None,
    morphism_type: Optional[MorphismType] = None,
) -> Tuple["Module", "Morphism"]:
    """Compute the cokernel of ``morphism``."""

    if not morphism.is_valid():
        raise ValueError("Morphism is not valid")

    module_cls = _resolve_module_type(morphism, module_type)
    morphism_cls = _resolve_morphism_type(morphism, morphism_type)

    cokernel_dimensions = {}
    cokernel_maps = {}
    projection_maps = {}
    extensions = {}

    for vertex_id in morphism.quiver.get_vertices():
        source_dim = morphism.source.spaces[vertex_id]
        target_dim = morphism.target.spaces[vertex_id]

        if target_dim == 0:
            cokernel_dimensions[vertex_id] = 0
            continue

        vertex_map = morphism.maps[vertex_id]

        if (
            source_dim == 0
            or vertex_map is None
            or isinstance(vertex_map, ZeroMap)
        ):
            cokernel_dimensions[vertex_id] = target_dim
            projection_maps[vertex_id] = morphism.field.identity_matrix(target_dim)
            extensions[vertex_id] = morphism.field.identity_matrix(target_dim)
            continue

        image_basis, extension = morphism.field.column_space_basis(vertex_map, do_extend=True)
        if image_basis is None:
            raise ValueError(f"Failed to compute column space basis for vertex {vertex_id}")

        image_dim = image_basis.shape[1]
        if image_dim == target_dim:
            cokernel_dimensions[vertex_id] = 0
            continue

        if extension is None:
            raise ValueError(f"Failed to compute extension basis for vertex {vertex_id}")

        cokernel_dim = extension.shape[1]
        cokernel_dimensions[vertex_id] = cokernel_dim

        extensions[vertex_id] = extension

        full_basis = morphism.field.matrix(np.hstack([image_basis, extension]))

        std_basis = morphism.field.identity_matrix(target_dim)
        coords = morphism.field.find_matrix_coordinates(full_basis, std_basis)
        if coords is None:
            raise ValueError(f"Failed to compute projection map for vertex {vertex_id}")

        projection_matrix = coords[:image_dim, :]

        projection_maps[vertex_id] = projection_matrix

    for arrow_id in morphism.quiver.get_arrows():
        arrow = morphism.quiver.arrows[arrow_id]
        source_vertex = arrow["source"]
        target_vertex = arrow["target"]

        source_cokernel_dim = cokernel_dimensions[source_vertex]
        target_cokernel_dim = cokernel_dimensions[target_vertex]

        if source_cokernel_dim == 0 or target_cokernel_dim == 0:
            cokernel_maps[arrow_id] = ZeroMap(source_cokernel_dim, target_cokernel_dim)
            continue

        target_arrow_map = morphism.target.maps[arrow_id]

        if target_arrow_map is None or isinstance(target_arrow_map, ZeroMap):
            cokernel_maps[arrow_id] = morphism.field.zero_matrix(target_cokernel_dim, source_cokernel_dim)
            continue

        source_extension = extensions.get(source_vertex)
        if source_extension is None:
            raise ValueError(
                f"Missing extension for source vertex {source_vertex} with non-zero cokernel dimension"
            )

        mapped_extension = target_arrow_map @ source_extension

        target_proj = projection_maps[target_vertex]
        cokernel_map = target_proj @ mapped_extension

        cokernel_maps[arrow_id] = cokernel_map

    cokernel_module = module_cls(
        morphism.quiver,
        morphism.field,
        name=f"Coker({morphism.name})",
        dimensions=cokernel_dimensions,
        maps=cokernel_maps,
    )

    projection = morphism_cls(morphism.target, cokernel_module, name=f"proj_{morphism.name}")

    for vertex_id in morphism.quiver.get_vertices():
        target_dim = morphism.target.spaces[vertex_id]
        cokernel_dim = cokernel_dimensions[vertex_id]

        if target_dim == 0 or cokernel_dim == 0:
            projection.maps[vertex_id] = ZeroMap(target_dim, cokernel_dim)
        else:
            projection.maps[vertex_id] = projection_maps[vertex_id]

    return cokernel_module, projection
