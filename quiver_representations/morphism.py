import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import copy

from .field import Field
from .quiver import Quiver
from .module import Module

class Morphism:
    """
    A class representing a morphism between two quiver representations.
    Maps between vector spaces that commute with the quiver structure.
    """

    def __init__(self, source: 'Module', target: 'Module', name: str = "", maps: Optional[Dict[int, Union[np.ndarray, List[List[int]], None]]] = None):
        """
        Initialize a morphism between two modules.

        Args:
            source: Source module
            target: Target module
            name: Optional name for the morphism
            maps: Optional dictionary mapping vertex IDs to matrices or None for zero maps

        Raises:
            ValueError: If source and target modules are over different quivers or fields
            ValueError: If any provided map has incompatible dimensions
        """
        self.source = source
        self.target = target
        self.name = name

        # Check that modules are over the same quiver and field
        if source.quiver is not target.quiver:
            raise ValueError("Source and target modules must be over the same quiver")
        if source.field is not target.field:
            raise ValueError("Source and target modules must be over the same field")

        self.quiver = source.quiver
        self.field = source.field

        # Initialize maps
        self.maps = {}

        # Process maps for each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            target_dim = self.target.spaces[vertex_id]

            # If no maps provided or this vertex not in maps, use zero map
            if maps is None or vertex_id not in maps:
                if source_dim == 0 or target_dim == 0:
                    self.maps[vertex_id] = ZeroMap(source_dim, target_dim)
                else:
                    self.maps[vertex_id] = self.field.zero_matrix(target_dim, source_dim)
                continue

            # Get the provided map for this vertex
            matrix = maps[vertex_id]

            # Handle zero-dimensional cases
            if source_dim == 0 or target_dim == 0:
                self.maps[vertex_id] = ZeroMap(source_dim, target_dim)
                continue

            # Handle None case (zero map)
            if matrix is None:
                self.maps[vertex_id] = self.field.zero_matrix(target_dim, source_dim)
                continue

            # Handle ZeroMap objects
            if isinstance(matrix, ZeroMap):
                if matrix.source_dim != source_dim or matrix.target_dim != target_dim:
                    raise ValueError(
                        f"ZeroMap dimensions ({matrix.source_dim}, {matrix.target_dim}) don't match "
                        f"vector space dimensions ({source_dim}, {target_dim}) for vertex {vertex_id}"
                    )
                self.maps[vertex_id] = matrix
                continue

            # Convert to field matrix if necessary
            if not isinstance(matrix, np.ndarray) or not hasattr(matrix, 'dtype') or matrix.dtype != self.field.GF.dtype:
                matrix = self.field.matrix(matrix)

            # Check dimensions
            if matrix.shape != (target_dim, source_dim):
                raise ValueError(
                    f"Matrix dimensions {matrix.shape} don't match vector space dimensions "
                    f"({target_dim}, {source_dim}) for vertex {vertex_id}"
                )

            self.maps[vertex_id] = matrix

        if maps:
            if not self.is_valid():
                raise ValueError("Provided maps are not valid")

    def set_map(self, vertex_id: int, matrix: Union[np.ndarray, List[List[int]], None]) -> None:
        """
        Set the linear map at a vertex.

        Args:
            vertex_id: ID of the vertex
            matrix: Matrix representing the linear map, or None for zero map

        Raises:
            ValueError: If vertex does not exist
            ValueError: If matrix dimensions don't match vector space dimensions
        """
        if vertex_id not in self.quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")

        source_dim = self.source.spaces[vertex_id]
        target_dim = self.target.spaces[vertex_id]

        if source_dim == 0 or target_dim == 0:
            # If either space is zero-dimensional, set map to None
            self.maps[vertex_id] = ZeroMap(source_dim, target_dim)
            return

        if matrix is None:
            # Set to zero map
            self.maps[vertex_id] = self.field.zero_matrix(target_dim, source_dim)
            return

        # Convert to field matrix if necessary
        if not isinstance(matrix, np.ndarray) or not hasattr(matrix, 'dtype') or matrix.dtype != self.field.GF.dtype:
            matrix = self.field.matrix(matrix)

        # Check dimensions
        if matrix.shape != (target_dim, source_dim):
            raise ValueError(
                f"Matrix dimensions {matrix.shape} don't match vector space dimensions "
                f"({target_dim}, {source_dim})"
            )

        self.maps[vertex_id] = matrix

    def set_maps(self, maps: Dict[int, Union[np.ndarray, List[List[int]], None]]) -> None:
        """
        Set maps for multiple vertices at once.

        Args:
            maps: Dictionary mapping vertex IDs to matrices
        """
        for vertex_id, matrix in maps.items():
            self.set_map(vertex_id, matrix)

    def get_map(self, vertex_id: int) -> Optional[np.ndarray]:
        """
        Get the linear map at a vertex.

        Args:
            vertex_id: ID of the vertex

        Returns:
            The linear map matrix, or None if not set

        Raises:
            ValueError: If vertex does not exist
        """
        if vertex_id not in self.quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")

        return self.maps.get(vertex_id)

    def is_defined(self) -> bool:
        """
        Check if the morphism is fully defined.

        A morphism is fully defined if maps are set for all vertices with
        non-zero dimensions in both source and target.

        Returns:
            True if the morphism is fully defined, False otherwise
        """
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            target_dim = self.target.spaces[vertex_id]

            if source_dim > 0 and target_dim > 0:
                if vertex_id not in self.maps or self.maps[vertex_id] is None:
                    return False

        return True

    def is_valid(self) -> bool:
        """
        Check if the morphism is valid.

        A morphism is valid if:
        1. It is fully defined
        2. All maps have compatible dimensions
        3. The maps commute with the quiver structure

        Returns:
            True if the morphism is valid, False otherwise
        """
        if not self.is_defined():
            return False

        # Check that maps commute with the quiver structure
        for arrow_id in self.quiver.get_arrows():
            arrow = self.quiver.arrows[arrow_id]
            source_vertex = arrow["source"]
            target_vertex = arrow["target"]

            source_source_dim = self.source.spaces[source_vertex]
            source_target_dim = self.source.spaces[target_vertex]
            target_source_dim = self.target.spaces[source_vertex]
            target_target_dim = self.target.spaces[target_vertex]

            # Skip arrows where any dimension is zero
            if (source_source_dim == 0 or source_target_dim == 0 or
                target_source_dim == 0 or target_target_dim == 0):
                continue

            # Get the linear maps for the arrow in source and target modules
            source_arrow_map = self.source.maps[arrow_id]
            target_arrow_map = self.target.maps[arrow_id]

            # Get the morphism maps for the vertices
            source_vertex_map = self.maps[source_vertex]
            target_vertex_map = self.maps[target_vertex]

            # Check commutativity
            # target_arrow_map * source_vertex_map = target_vertex_map * source_arrow_map
            left = target_arrow_map @ source_vertex_map
            right = target_vertex_map @ source_arrow_map

            if not np.array_equal(left, right):
                return False

        return True

    def compose(self, other: 'Morphism') -> 'Morphism':
        """
        Compose this morphism with another.

        Args:
            other: Another morphism whose source is this morphism's target

        Returns:
            A new morphism representing the composition

        Raises:
            ValueError: If the morphisms can't be composed
        """
        if self.target is not other.source:
            raise ValueError("Morphisms can't be composed: target of first != source of second")

        result = Morphism(self.source, other.target,
                         name=f"{other.name}∘{self.name}")

        # Compose the maps at each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            mid_dim = self.target.spaces[vertex_id]
            target_dim = other.target.spaces[vertex_id]

            if source_dim > 0 and mid_dim > 0 and target_dim > 0:
                # Get the maps
                first_map = self.maps[vertex_id]
                second_map = other.maps[vertex_id]

                # Compose them
                result.maps[vertex_id] = second_map @ first_map
            else:
                result.maps[vertex_id] = ZeroMap(source_dim, target_dim)

        return result

    def kernel(self) -> Tuple['Module', 'Morphism']:
        """
        Compute the kernel of the morphism.

        Returns:
            A tuple (K, i) where:
            - K is a module representing the kernel
            - i is the inclusion morphism from K to the source module

        Raises:
            ValueError: If the morphism is not valid
        """
        if not self.is_valid():
            raise ValueError("Morphism is not valid")

        # Gather all kernel dimensions and basis matrices first
        kernel_dimensions = {}
        kernel_bases = {}

        # Compute kernel dimensions and bases at each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            target_dim = self.target.spaces[vertex_id]

            if source_dim == 0:
                # If source is zero-dimensional, kernel is zero-dimensional
                kernel_dimensions[vertex_id] = 0
                kernel_bases[vertex_id] = None
            else:
                # Get the vertex map
                vertex_map = self.maps.get(vertex_id)

                if vertex_map is None or isinstance(vertex_map, ZeroMap):
                    # If the map is zero, kernel is the whole source space
                    kernel_dimensions[vertex_id] = source_dim
                    # Identity matrix as basis
                    kernel_bases[vertex_id] = self.field.identity_matrix(source_dim)
                elif target_dim == 0:
                    # If target is zero-dimensional but source is not, kernel is the whole source space
                    # But actually it should have been ZeroMap!
                    kernel_dimensions[vertex_id] = source_dim
                    kernel_bases[vertex_id] = self.field.identity_matrix(source_dim)
                else:
                    try:
                        # Try computing kernel using built-in null_space
                        kernel_basis = self.field.kernel_basis(vertex_map) # For some reason,
                        kernel_dim = kernel_basis.shape[1]  # Number of basis vectors
                    except (AttributeError, Exception) as e:
                        # If null_space/kernel_basis fails, fallback to manual computation
                        print("Kernel dimension computation failed, for some mysterious reason")

                    kernel_dimensions[vertex_id] = kernel_dim
                    kernel_bases[vertex_id] = kernel_basis

        # Prepare maps for the kernel module
        kernel_maps = {}

        # For each arrow in the quiver, compute the restriction of the source map to the kernel
        for arrow_id in self.quiver.get_arrows():
            arrow = self.quiver.arrows[arrow_id]
            source_vertex = arrow["source"]
            target_vertex = arrow["target"]

            source_kernel_dim = kernel_dimensions[source_vertex]
            target_kernel_dim = kernel_dimensions[target_vertex]

            # If either kernel space is zero-dimensional, use a ZeroMap
            if source_kernel_dim == 0 or target_kernel_dim == 0:
                kernel_maps[arrow_id] = ZeroMap(source_kernel_dim, target_kernel_dim)
                continue

            # Get the original map in the source module
            source_arrow_map = self.source.maps[arrow_id]

            if source_arrow_map is None or isinstance(source_arrow_map, ZeroMap):
                # If the original map is zero, the restricted map is also zero
                kernel_maps[arrow_id] = ZeroMap(target_kernel_dim, source_kernel_dim)
                continue

            # Compute the map in the kernel basis
            source_basis = kernel_bases[source_vertex]
            target_basis = kernel_bases[target_vertex]

            # Apply original map to source kernel basis
            intermediate = source_arrow_map @ source_basis

            # Compute kernel map (may be zero)
            kernel_maps[arrow_id] = self.field.find_matrix_coordinates(target_basis, intermediate)

        # Now create the kernel module with all dimensions and maps prepared
        kernel_module = Module(
            self.quiver,
            self.field,
            name=f"Ker({self.name})",
            dimensions=kernel_dimensions,
            maps=kernel_maps
        )

        # Create the inclusion morphism from kernel to source
        inclusion = Morphism(kernel_module, self.source, name=f"inc_{self.name}")

        # Set the inclusion maps for each vertex
        for vertex_id in self.quiver.get_vertices():
            kernel_dim = kernel_dimensions[vertex_id]
            source_dim = self.source.spaces[vertex_id]

            if kernel_dim == 0 or source_dim == 0:
                # If either space is zero-dimensional, use a zero map
                inclusion.maps[vertex_id] = ZeroMap(kernel_dim, source_dim)
            else:
                # The inclusion map is just the kernel basis matrix
                # Each column of the basis is a vector in the source space
                basis = kernel_bases[vertex_id]

                # Ensure basis has shape (source_dim, kernel_dim)
                if basis.shape[0] != source_dim:
                    basis = basis.T

                # Verify shape before setting
                if basis.shape != (source_dim, kernel_dim):
                    raise ValueError(f"Basis shape {basis.shape} doesn't match expected shape ({source_dim}, {kernel_dim})")

                inclusion.maps[vertex_id] = basis

        return (kernel_module, inclusion)

    def image(self) -> Tuple['Module', 'Morphism', 'Morphism']:
        """
        Compute the image of the morphism.

        Returns:
            A tuple (I, i, p) where:
            - I is a module representing the image
            - i is the inclusion morphism from I to the target module
            - p is the epimorphism from the source module to the image

        Raises:
            ValueError: If the morphism is not valid
        """
        if not self.is_valid():
            raise ValueError("Morphism is not valid")

        # Gather all image dimensions, bases, and maps before creating the module
        image_dimensions = {}
        image_maps = {}
        inclusion_maps = {}
        projection_maps = {}
        image_bases = {}  # Store image bases for computing maps between image spaces

        # First pass: compute dimensions and bases at each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            target_dim = self.target.spaces[vertex_id]

            if source_dim == 0 or target_dim == 0 or self.maps[vertex_id] is None or isinstance(self.maps[vertex_id], ZeroMap):
                # If either space is zero-dimensional or map is zero, image is zero
                image_dimensions[vertex_id] = 0
                continue

            # Get the vertex map
            vertex_map = self.maps[vertex_id]

            # Get a basis for the column space (image of the map)

            image_basis = self.field.column_space_basis(vertex_map)
            if image_basis is None:
                raise ValueError(f"Failed to compute column space basis for vertex {vertex_id}")

            # Compute and store image dimension
            image_dim = image_basis.shape[1]
            image_dimensions[vertex_id] = image_dim

            # Store the basis for later use
            image_bases[vertex_id] = image_basis

            # The inclusion map is just the image basis matrix; they will be used for the image inclusion
            inclusion_maps[vertex_id] = image_basis

            # Create the projection matrix from source to coimage
            # This maps vectors from the source space to their coordinates in the image basis
            source_basis = self.field.identity_matrix(source_dim)
            projection = self.field.find_matrix_coordinates(vertex_map, image_basis)
            if projection is None:
                raise ValueError(f"Failed to compute projection map for vertex {vertex_id}")

            # Store the projection map
            projection_maps[vertex_id] = projection

        # Second pass: compute maps between image spaces
        for arrow_id in self.quiver.get_arrows():
            arrow = self.quiver.arrows[arrow_id]
            source_vertex = arrow["source"]
            target_vertex = arrow["target"]

            source_image_dim = image_dimensions[source_vertex]
            target_image_dim = image_dimensions[target_vertex]

            # If either image space is zero-dimensional, use a ZeroMap
            if source_image_dim == 0 or target_image_dim == 0:
                image_maps[arrow_id] = ZeroMap(source_image_dim, target_image_dim)
                continue

            # Get the map in the target module
            target_arrow_map = self.target.maps[arrow_id]

            if target_arrow_map is None or isinstance(target_arrow_map, ZeroMap):
                # If the target map is zero, the image map is also zero
                image_maps[arrow_id] = self.field.zero_matrix(target_image_dim, source_image_dim)
                continue

            # Get the bases for source and target image spaces
            source_basis = image_bases[source_vertex]
            target_basis = image_bases[target_vertex]

            # Apply the target module's map to the source basis
            mapped_source = target_arrow_map @ source_basis

            # Express the mapped vectors in terms of the target basis
            image_map = self.field.find_matrix_coordinates(target_basis, mapped_source)
            if image_map is None:
                raise ValueError(f"Failed to compute image map for arrow {arrow_id}")

            # Store the image map
            image_maps[arrow_id] = image_map

        # Create the image module with all dimensions and maps prepared
        image_module = Module(
            self.quiver,
            self.field,
            name=f"Im({self.name})",
            dimensions=image_dimensions,
            maps=image_maps
        )

        # Create the inclusion morphism from image to target
        inclusion = Morphism(image_module, self.target, name=f"incl_{self.name}")

        # Set the inclusion maps for each vertex image -> target
        for vertex_id in self.quiver.get_vertices():
            image_dim = image_dimensions[vertex_id]
            target_dim = self.target.spaces[vertex_id]

            if image_dim == 0 or target_dim == 0:
                # If either space is zero-dimensional, use a ZeroMap
                inclusion.maps[vertex_id] = ZeroMap(image_dim, target_dim)
            else:
                # Use the inclusion map we computed earlier
                inclusion.maps[vertex_id] = inclusion_maps[vertex_id]

        # Create the projection morphism from source to coimage
        projection = Morphism(self.source, image_module, name=f"proj_{self.name}")

        # Set the projection maps for each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            coimage_dim = image_dimensions[vertex_id] # This is the same, but theoretically, we're creating a coimage here!

            if source_dim == 0 or coimage_dim == 0:
                # If either space is zero-dimensional, use a ZeroMap
                projection.maps[vertex_id] = ZeroMap(source_dim, coimage_dim)
            else:
                # Use the projection map we computed earlier
                projection.maps[vertex_id] = projection_maps[vertex_id]

        return (image_module, inclusion, projection)

    def cokernel(self) -> Tuple['Module', 'Morphism']:
        """
        Compute the cokernel of the morphism.

        Returns:
            A tuple (C, p) where:
            - C is a module representing the cokernel
            - p is the projection morphism from the target module to C

        Raises:
            ValueError: If the morphism is not valid
        """
        if not self.is_valid():
            raise ValueError("Morphism is not valid")

        # Gather all cokernel dimensions, bases, and maps before creating the module
        cokernel_dimensions = {}
        cokernel_maps = {}
        projection_maps = {}
        extensions = {}  # Store the extensions for each vertex for reuse

        # First pass: compute dimensions and bases at each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.source.spaces[vertex_id]
            target_dim = self.target.spaces[vertex_id]

            if target_dim == 0:
                # If target is zero-dimensional, cokernel is zero
                cokernel_dimensions[vertex_id] = 0
                continue

            if source_dim == 0 or self.maps[vertex_id] is None or isinstance(self.maps[vertex_id], ZeroMap):
                # If source is zero or map is zero, cokernel is the whole target
                cokernel_dimensions[vertex_id] = target_dim

                # Create projection map (identity in this case)
                projection_maps[vertex_id] = self.field.identity_matrix(target_dim)
                extensions[vertex_id] = self.field.identity_matrix(target_dim)
                continue

            # Get the vertex map
            vertex_map = self.maps[vertex_id]

            # Get a basis for the column space (image of the map) and extend it to a full basis
            image_basis, extension = self.field.column_space_basis(vertex_map, do_extend=True)
            if image_basis is None:
                raise ValueError(f"Failed to compute column space basis for vertex {vertex_id}")

            # Check if the image spans the whole target space
            image_dim = image_basis.shape[1]
            if image_dim == target_dim:
                # Image spans the whole space, cokernel is zero
                cokernel_dimensions[vertex_id] = 0
                continue

            # If extension is None, that's an error since we know image doesn't span the whole space
            if extension is None:
                raise ValueError(f"Failed to compute extension basis for vertex {vertex_id}")

            # Compute cokernel dimension
            cokernel_dim = extension.shape[1]
            cokernel_dimensions[vertex_id] = cokernel_dim

            # Store the extension for later use in computing maps
            extensions[vertex_id] = extension

            # The full basis is a combination of the image basis and extension
            full_basis = self.field.matrix(np.hstack([image_basis, extension]))

            # Create the projection matrix
            # This maps from the target space to the cokernel
            projection_matrix = self.field.zero_matrix(cokernel_dim, target_dim)

            # Compute coordinates of the standard basis in the full basis
            std_basis = self.field.identity_matrix(target_dim)
            coords = self.field.find_matrix_coordinates(full_basis, std_basis)
            if coords is None:
                raise ValueError(f"Failed to compute projection map for vertex {vertex_id}")

            # Extract only the coordinates corresponding to the cokernel basis
            projection_matrix = coords[:image_dim, :]

            # Store the projection map
            projection_maps[vertex_id] = projection_matrix

        # Second pass: compute maps between cokernel spaces
        for arrow_id in self.quiver.get_arrows():
            arrow = self.quiver.arrows[arrow_id]
            source_vertex = arrow["source"]
            target_vertex = arrow["target"]

            source_cokernel_dim = cokernel_dimensions[source_vertex]
            target_cokernel_dim = cokernel_dimensions[target_vertex]

            # If either cokernel space is zero-dimensional, use a ZeroMap
            if source_cokernel_dim == 0 or target_cokernel_dim == 0:
                cokernel_maps[arrow_id] = ZeroMap(source_cokernel_dim, target_cokernel_dim)
                continue

            # Get the map in the target module
            target_arrow_map = self.target.maps[arrow_id]

            if target_arrow_map is None or isinstance(target_arrow_map, ZeroMap):
                # If the target map is zero, the cokernel map is also zero
                cokernel_maps[arrow_id] = self.field.zero_matrix(target_cokernel_dim, source_cokernel_dim)
                continue

            # Get the projection maps
            source_proj = projection_maps[source_vertex]
            target_proj = projection_maps[target_vertex]

            # To compute the cokernel map, we need to:
            # 1. Get a representation of the cokernel basis in the target space at source vertex
            # 2. Apply the target module's map to this representation
            # 3. Project the result to the cokernel at target vertex

            # The extension part of the basis represents the cokernel
            # We know for each vertex, we computed [image_basis, extension] earlier
            # For source_vertex, the extension provides a lift of the cokernel basis to the target space


            # Get the stored extension for source vertex
            source_extension = extensions.get(source_vertex)

            # At this point, the extension must exist since the cokernel dimension is non-zero
            if source_extension is None:
                raise ValueError(f"Missing extension for source vertex {source_vertex} with non-zero cokernel dimension")

            # Apply the target module's map to the extension (our lift of the cokernel basis)
            mapped_extension = target_arrow_map @ source_extension

            # Project to the cokernel at target_vertex
            cokernel_map = target_proj @ mapped_extension

            # Store the cokernel map
            cokernel_maps[arrow_id] = cokernel_map

        # Create the cokernel module with all dimensions and maps prepared
        cokernel_module = Module(
            self.quiver,
            self.field,
            name=f"Coker({self.name})",
            dimensions=cokernel_dimensions,
            maps=cokernel_maps
        )

        # Create the projection morphism from target to cokernel
        projection = Morphism(self.target, cokernel_module, name=f"proj_{self.name}")

        # Set the projection maps for each vertex
        for vertex_id in self.quiver.get_vertices():
            target_dim = self.target.spaces[vertex_id]
            cokernel_dim = cokernel_dimensions[vertex_id]

            if target_dim == 0 or cokernel_dim == 0:
                # If either space is zero-dimensional, use a ZeroMap
                projection.maps[vertex_id] = ZeroMap(target_dim, cokernel_dim)
            else:
                # Use the projection map we computed earlier
                projection.maps[vertex_id] = projection_maps[vertex_id]

        return (cokernel_module, projection)

    @classmethod
    def direct_sum(cls, f: 'Morphism', g: 'Morphism') -> Tuple['Module', 'Module', 'Morphism']:
        """
        Compute the direct sum of two morphisms.

        Args:
            f: First morphism
            g: Second morphism

        Returns:
            A tuple (X, Y, h) where:
            - X is the direct sum of the source modules
            - Y is the direct sum of the target modules
            - h is the direct sum morphism X → Y

        Raises:
            ValueError: If morphisms are over different quivers or fields
        """
        # Check compatibility
        if f.quiver is not g.quiver:
            raise ValueError("Morphisms must be over the same quiver")
        if f.field is not g.field:
            raise ValueError("Morphisms must be over the same field")

        # Compute direct sum of source modules
        source_sum = Module.direct_sum(f.source, g.source)

        # Compute direct sum of target modules
        target_sum = Module.direct_sum(f.target, g.target)

        # Create the direct sum morphism
        result_morphism = cls(source_sum, target_sum,
                                  name=f"({f.name})⊕({g.name})")

        # For each vertex, create the block diagonal morphism
        for vertex_id in f.quiver.get_vertices():
            source_dim1 = f.source.spaces.get(vertex_id, 0)
            source_dim2 = g.source.spaces.get(vertex_id, 0)
            target_dim1 = f.target.spaces.get(vertex_id, 0)
            target_dim2 = g.target.spaces.get(vertex_id, 0)

            source_dim_total = source_dim1 + source_dim2
            target_dim_total = target_dim1 + target_dim2

            # Skip if total dimensions are zero
            if source_dim_total == 0 or target_dim_total == 0:
                result_morphism.set_map(vertex_id, ZeroMap(source_dim_total, target_dim_total))
                continue

            # Get the original maps
            map1 = f.get_map(vertex_id)
            map2 = g.get_map(vertex_id)

            # Create block diagonal matrix - shape is (target_dim, source_dim)
            result_map = f.field.zero_matrix(target_dim_total, source_dim_total)

            # Fill in the blocks
            if map1 is not None and source_dim1 > 0 and target_dim1 > 0:
                result_map[:target_dim1, :source_dim1] = map1

            if map2 is not None and source_dim2 > 0 and target_dim2 > 0:
                result_map[target_dim1:, source_dim1:] = map2

            # Set the map for this vertex
            result_morphism.set_map(vertex_id, result_map)

        return source_sum, target_sum, result_morphism

    @classmethod
    def direct_power(cls, morphism: 'Morphism', power: int) -> Tuple['Module', 'Module', 'Morphism']:
        """
        Compute the direct power of a morphism: f^⊕power.

        Args:
            morphism: The morphism to take the direct power of
            power: The number of copies to include in the direct sum

        Returns:
            A tuple (X, Y, h) where:
            - X is the direct power of the source module
            - Y is the direct power of the target module
            - h is the direct power morphism X → Y

        Raises:
            ValueError: If power is negative
        """
        if power < 0:
            raise ValueError("Power must be non-negative")

        # Compute direct powers of source and target modules
        source_power = Module.direct_power(morphism.source, power)
        target_power = Module.direct_power(morphism.target, power)

        if power == 0:
            # Return a zero module
            zero_dimensions = {v: 0 for v in morphism.quiver.get_vertices()}

            # Create zero maps for all arrows
            zero_maps = {}
            for arrow_id in morphism.quiver.get_arrows():
                arrow = morphism.quiver.arrows[arrow_id]
                zero_maps[arrow_id] = ZeroMap(0, 0)

            source_power = Module(morphism.quiver, morphism.field,
                                 name="Zero",
                                 dimensions=zero_dimensions,
                                 maps=zero_maps)

            target_power = Module(morphism.quiver, morphism.field,
                                 name="Zero",
                                 dimensions=zero_dimensions,
                                 maps=zero_maps)

            result_morphism = Morphism(source_power, target_power,
                                      name="Zero morphism")

            # Initialize all zero maps for the morphism
            for vertex_id in morphism.quiver.get_vertices():
                result_morphism.set_map(vertex_id, ZeroMap(0, 0))

            return source_power, target_power, result_morphism

        if power == 1:
            # Return a copy of the original morphism
            source_copy = morphism.source.copy()
            target_copy = morphism.target.copy()
            result_morphism = Morphism(source_copy, target_copy,
                                      name=f"{morphism.name} (copy)")

            # Copy maps
            for vertex_id in morphism.quiver.get_vertices():
                orig_map = morphism.get_map(vertex_id)
                if orig_map is not None:
                    result_morphism.set_map(vertex_id, orig_map.copy() if hasattr(orig_map, 'copy') else orig_map)

            return source_copy, target_copy, result_morphism

        # Create the direct power morphism
        result_morphism = cls(source_power, target_power,
                                  name=f"({morphism.name})^⊕{power}")

        # For each vertex, create a block diagonal matrix of the original maps
        for vertex_id in morphism.quiver.get_vertices():
            source_dim = morphism.source.spaces.get(vertex_id, 0)
            target_dim = morphism.target.spaces.get(vertex_id, 0)

            source_dim_total = source_dim * power
            target_dim_total = target_dim * power

            # Skip if either total dimension is zero
            if source_dim_total == 0 or target_dim_total == 0:
                result_morphism.set_map(vertex_id, ZeroMap(source_dim_total, target_dim_total))
                continue

            # Get the original map
            orig_map = morphism.get_map(vertex_id)

            if orig_map is None or isinstance(orig_map, ZeroMap):
                # Original map is a zero map
                result_morphism.set_map(vertex_id, ZeroMap(source_dim_total, target_dim_total))
                continue

            # Create block diagonal matrix with 'power' copies of the original map
            result_map = morphism.field.zero_matrix(target_dim_total, source_dim_total)

            # Fill in the blocks with copies of the original map
            for i in range(power):
                result_map[i*target_dim:(i+1)*target_dim,
                          i*source_dim:(i+1)*source_dim] = orig_map

            # Set the map for this vertex
            result_morphism.set_map(vertex_id, result_map)

        return source_power, target_power, result_morphism

    def to_dict(self) -> Dict:
        """Convert the morphism to a dictionary for serialization."""
        # Convert maps to nested lists for JSON serialization
        maps_dict = {}
        for vertex_id, matrix in self.maps.items():
            if matrix is not None:
                maps_dict[vertex_id] = matrix.tolist()
            else:
                maps_dict[vertex_id] = None

        return {
            "name": self.name,
            "source_name": self.source.name,
            "target_name": self.target.name,
            "quiver_name": self.quiver.name,
            "maps": maps_dict
        }

    @classmethod
    def from_dict(cls, data: Dict, source: 'Module', target: 'Module') -> 'Morphism':
        """
        Create a morphism from a dictionary.

        Args:
            data: Dictionary representation of the morphism
            source: Source module
            target: Target module

        Returns:
            A new morphism
        """
        morphism = cls(source, target, name=data.get("name", ""))

        # Set maps
        for vertex_id, matrix_data in data["maps"].items():
            if matrix_data is not None:
                vertex_id = int(vertex_id)  # Convert from string key
                matrix = source.field.matrix(matrix_data)
                morphism.maps[vertex_id] = matrix
            else:
                morphism.maps[vertex_id] = None

        return morphism

    def save(self, filename: str) -> None:
        """Save the morphism to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str, source: 'Module', target: 'Module') -> 'Morphism':
        """
        Load a morphism from a JSON file.

        Args:
            filename: Path to the JSON file
            source: Source module
            target: Target module

        Returns:
            A new morphism
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, source, target)

    def copy(self) -> 'Morphism':
        """Create a deep copy of the morphism."""
        result = Morphism(self.source, self.target, name=f"{self.name} (copy)")

        # Copy maps
        for vertex_id, matrix in self.maps.items():
            if matrix is not None:
                result.maps[vertex_id] = matrix.ccokeopy()
            else:
                result.maps[vertex_id] = None

        return result

    def __str__(self) -> str:
        """Concise string representation of the morphism."""
        return f"Morphism '{self.name}': {self.source.name} → {self.target.name}"

    def __repr__(self) -> str:
        """Detailed string representation showing all maps."""
        return self.details()

    def details(self, show_maps: bool = True) -> str:
        """
        Detailed string representation of the morphism.

        Args:
            show_maps: If True, includes the matrices for all vertex maps

        Returns:
            Detailed string representation
        """
        s = str(self) + "\n"  # Start with the basic representation

        # Add source and target dimensions
        s += "\nSource dimensions: " + ", ".join(
            f"{self.quiver.vertices[v]['label']}:{dim}"
            for v, dim in sorted(self.source.spaces.items())
        )
        s += "\nTarget dimensions: " + ", ".join(
            f"{self.quiver.vertices[v]['label']}:{dim}"
            for v, dim in sorted(self.target.spaces.items())
        )

        if show_maps:
            s += "\nVertex maps:\n"
            for vertex_id in sorted(self.quiver.get_vertices()):
                vertex_label = self.quiver.vertices[vertex_id]["label"]
                map_matrix = self.maps.get(vertex_id)

                s += f"\n{vertex_label}:\n"
                if isinstance(map_matrix, ZeroMap):
                    s += f"Zero map ({map_matrix.source_dim} → {map_matrix.target_dim})"
                elif map_matrix is None:
                    s += "None"
                else:
                    # Format matrix rows
                    s += "\n".join(
                        "[" + " ".join(f"{x:3}" for x in row) + "]"
                        for row in map_matrix
                    )
                s += "\n"

        return s
