import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import copy
import json

from .field import FiniteField, ComplexNumbers
from .quiver import Quiver
from .morphism import Morphism

class Module:
    """
    A class representing a representation of a quiver (also called a module).
    Assigns vector spaces to vertices and linear maps to arrows.
    """

    def __init__(self, quiver, field, name: str = "", dimensions: Optional[Dict[int, int]] = None, maps: Optional[Dict[int, np.ndarray]] = None):
        """
        Initialize a module (representation) over the given quiver and field.

        Args:
            quiver: The quiver this module is defined over
            field: The finite field for vector spaces
            name: Optional name for the module
            dimensions: Dictionary mapping vertex IDs to dimensions (required)
            maps: Dictionary mapping arrow IDs to matrices (or ZeroMap objects)
        """
        self.quiver = quiver
        self.field = field
        self.name = name


        # Initialize vector spaces at each vertex
        self.spaces = {vertex_id: 0 for vertex_id in quiver.get_vertices()}

        # Set dimensions
        if dimensions is None:
            raise ValueError("Dimensions must be provided for all vertices")

        for vertex_id, dim in dimensions.items():
            if vertex_id not in self.quiver.vertices:
                raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")
            if dim < 0:
                raise ValueError(f"Dimension must be non-negative, got {dim}")
            self.spaces[vertex_id] = dim

        # Initialize maps
        self.maps = {}

        # Check if maps were provided
        if maps is None:
            # Initialize with zero maps
            for arrow_id in quiver.get_arrows():
                arrow = quiver.arrows[arrow_id]
                source_id = arrow["source"]
                target_id = arrow["target"]
                source_dim = self.spaces[source_id]
                target_dim = self.spaces[target_id]

                if source_dim == 0 or target_dim == 0:
                    self.maps[arrow_id] = ZeroMap(source_dim, target_dim)
                else:
                    self.maps[arrow_id] = field.zero_matrix(target_dim, source_dim)
        else:
            # Set provided maps
            for arrow_id in quiver.get_arrows():
                if arrow_id not in maps:
                    # If a map is not provided, use zero map
                    arrow = quiver.arrows[arrow_id]
                    source_id = arrow["source"]
                    target_id = arrow["target"]
                    source_dim = self.spaces[source_id]
                    target_dim = self.spaces[target_id]

                    if source_dim == 0 or target_dim == 0:
                        self.maps[arrow_id] = ZeroMap(source_dim, target_dim)
                        continue
                    else:
                        self.maps[arrow_id] = field.zero_matrix(target_dim, source_dim)
                else:
                    # Process the provided map
                    matrix = maps[arrow_id]
                    arrow = quiver.arrows[arrow_id]
                    source_id = arrow["source"]
                    target_id = arrow["target"]
                    source_dim = self.spaces[source_id]
                    target_dim = self.spaces[target_id]

                    # Check if the map is a ZeroMap
                    if isinstance(matrix, ZeroMap):
                        # Verify the dimensions of the ZeroMap match the vector spaces
                        if matrix.source_dim != source_dim or matrix.target_dim != target_dim:
                            raise ValueError(
                                f"ZeroMap dimensions ({matrix.source_dim}, {matrix.target_dim}) don't match "
                                f"vector space dimensions (source_dim={source_dim}, target_dim={target_dim}) for arrow {arrow_id}"
                            )
                        self.maps[arrow_id] = matrix
                        continue

                    # Handle the case when either dimension is zero
                    if source_dim == 0 or target_dim == 0:
                        self.maps[arrow_id] = ZeroMap(source_dim, target_dim)
                        continue

                    # Convert to field matrix if necessary
                    if matrix is not None:
                        if not isinstance(matrix, np.ndarray) or not hasattr(matrix, 'dtype') or matrix.dtype != self.field.GF.dtype:
                            matrix = self.field.matrix(matrix)

                        # Ensure both dimensions are non-zero before checking matrix shape
                        if source_dim > 0 and target_dim > 0:
                            # Check dimensions - matrix shape should be (target_dim, source_dim)
                            if matrix.shape != (target_dim, source_dim):
                                raise ValueError(
                                    f"Matrix dimensions {matrix.shape} don't match vector space dimensions "
                                    f"(target_dim={target_dim}, source_dim={source_dim}) for arrow {arrow_id}"
                                )

                        self.maps[arrow_id] = matrix
                    else:
                        self.maps[arrow_id] = None

    def set_dimension(self, vertex_id: int, dimension: int) -> None:
        """
        Set the dimension of the vector space at a vertex.

        Args:
            vertex_id: ID of the vertex
            dimension: Dimension of the vector space

        Raises:
            ValueError: If vertex does not exist
            ValueError: If dimension is negative
        """
        if vertex_id not in self.quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")
        if dimension < 0:
            raise ValueError(f"Dimension must be non-negative, got {dimension}")

        old_dim = self.spaces.get(vertex_id, 0)
        self.spaces[vertex_id] = dimension

        # Update maps involving this vertex
        # For outgoing arrows, adjust the number of columns
        for target, arrow_id in self.quiver.successors[vertex_id]:
            target_dim = self.spaces[target]
            if arrow_id in self.maps:
                # Resize the map if necessary
                if dimension > 0 and target_dim > 0:
                    if old_dim > 0:
                        # Keep existing values if possible
                        old_map = self.maps[arrow_id]
                        new_map = self.field.zero_matrix(dimension, target_dim)
                        min_rows = min(old_dim, dimension)
                        min_cols = min(target_dim, old_map.shape[1])
                        new_map[:min_rows, :min_cols] = old_map[:min_rows, :min_cols]
                        self.maps[arrow_id] = new_map
                    else:
                        # Create new zero map
                        self.maps[arrow_id] = self.field.zero_matrix(dimension, target_dim)
                else:
                    # If either dimension is zero, set to None
                    self.maps[arrow_id] = None

        # For incoming arrows, adjust the number of rows
        for source, arrow_id in self.quiver.predecessors[vertex_id]:
            source_dim = self.spaces[source]
            if arrow_id in self.maps:
                # Resize the map if necessary
                if source_dim > 0 and dimension > 0:
                    if old_dim > 0:
                        # Keep existing values if possible
                        old_map = self.maps[arrow_id]
                        new_map = self.field.zero_matrix(source_dim, dimension)
                        min_rows = min(source_dim, old_map.shape[0])
                        min_cols = min(old_dim, dimension)
                        new_map[:min_rows, :min_cols] = old_map[:min_rows, :min_cols]
                        self.maps[arrow_id] = new_map
                    else:
                        # Create new zero map
                        self.maps[arrow_id] = self.field.zero_matrix(source_dim, dimension)
                else:
                    # If either dimension is zero, set to None
                    self.maps[arrow_id] = None

    def set_dimensions(self, dimensions: Dict[int, int]) -> None:
        """
        Set dimensions for multiple vertices at once.

        Args:
            dimensions: Dictionary mapping vertex IDs to dimensions
        """
        for vertex_id, dimension in dimensions.items():
            self.set_dimension(vertex_id, dimension)

    def set_map(self, arrow_id: int, matrix: Union[np.ndarray, List[List[int]], None]) -> None:
        """
        Set the linear map for an arrow.

        Args:
            arrow_id: ID of the arrow
            matrix: Matrix representing the linear map, or None for zero map

        Raises:
            ValueError: If arrow does not exist
            ValueError: If matrix dimensions don't match vector space dimensions
        """
        if arrow_id not in self.quiver.arrows:
            raise ValueError(f"Arrow {arrow_id} does not exist in the quiver")

        arrow = self.quiver.arrows[arrow_id]
        source_id = arrow["source"]
        target_id = arrow["target"]
        source_dim = self.spaces[source_id]
        target_dim = self.spaces[target_id]

        if source_dim == 0 or target_dim == 0:
            # If either space is zero-dimensional, set map to None
            self.maps[arrow_id] = None
            return

        if matrix is None:
            # Set to zero map
            self.maps[arrow_id] = self.field.zero_matrix(target_dim, source_dim)
            return

        # Convert to field matrix if necessary
        if not isinstance(matrix, np.ndarray) or not hasattr(matrix, 'dtype') or matrix.dtype != self.field.GF.dtype:
            matrix = self.field.matrix(matrix)

        # Check dimensions - matrix shape should be (target_dim, source_dim)
        if matrix.shape != (target_dim, source_dim):
            raise ValueError(
                f"Matrix dimensions {matrix.shape} don't match vector space dimensions "
                f"(target_dim={target_dim}, source_dim={source_dim})"
            )

        self.maps[arrow_id] = matrix

    def get_dimension_vector(self) -> Dict[int, int]:
        """Return the dimension vector of the module."""
        return dict(self.spaces)

    def get_total_dimension(self) -> int:
        """Return the total dimension of the module (sum of dimensions at vertices)."""
        return sum(self.spaces.values())

    def get_support(self) -> List[int]:
        """Return the list of vertices with non-zero dimension."""
        return [v for v, dim in self.spaces.items() if dim > 0]

    @classmethod
    def direct_sum(cls, module1: 'Module', module2: 'Module') -> 'Module':
        """
        Compute the direct sum of two modules.

        Args:
            module1: First module
            module2: Second module

        Returns:
            A new module representing the direct sum

        Raises:
            ValueError: If modules are over different quivers or fields
        """
        if module1.quiver is not module2.quiver:
            raise ValueError("Modules must be over the same quiver")
        if module1.field is not module2.field:
            raise ValueError("Modules must be over the same field")

        quiver = module1.quiver
        field = module1.field

        # Compute dimensions for the direct sum
        dimensions = {}
        for vertex_id in quiver.get_vertices():
            dim1 = module1.spaces.get(vertex_id, 0)
            dim2 = module2.spaces.get(vertex_id, 0)
            dimensions[vertex_id] = dim1 + dim2

        # Prepare maps for the direct sum
        maps = {}
        for arrow_id in quiver.get_arrows():
            arrow = quiver.arrows[arrow_id]
            source_id = arrow["source"]
            target_id = arrow["target"]

            source_dim1 = module1.spaces.get(source_id, 0)
            source_dim2 = module2.spaces.get(source_id, 0)
            target_dim1 = module1.spaces.get(target_id, 0)
            target_dim2 = module2.spaces.get(target_id, 0)

            source_dim_total = source_dim1 + source_dim2
            target_dim_total = target_dim1 + target_dim2

            # Skip if total dimensions are zero
            if source_dim_total == 0 or target_dim_total == 0:
                maps[arrow_id] = ZeroMap(source_dim_total, target_dim_total)
                continue

            # Get the original maps
            map1 = module1.maps.get(arrow_id)
            map2 = module2.maps.get(arrow_id)

            # Create block diagonal matrix - shape is (target_dim, source_dim)
            result_map = field.zero_matrix(target_dim_total, source_dim_total)

            # Fill in the blocks
            if isinstance(map1, ZeroMap) or source_dim1 == 0 or target_dim1 == 0:
                # First map is a zero map or dimensions are zero
                pass
            elif isinstance(map1, np.ndarray):
                result_map[:target_dim1, :source_dim1] = map1

            if isinstance(map2, ZeroMap) or source_dim2 == 0 or target_dim2 == 0:
                # Second map is a zero map or dimensions are zero
                pass
            elif isinstance(map2, np.ndarray):
                result_map[target_dim1:, source_dim1:] = map2

            maps[arrow_id] = result_map

        # Create the module with dimensions and maps
        return cls(quiver, field,
                     name=f"({module1.name})⊕({module2.name})",
                     dimensions=dimensions,
                     maps=maps)

    @classmethod
    def direct_power(cls, module: 'Module', power: int) -> 'Module':
        """
        Compute the direct sum of a module with itself multiple times: M^⊕power.

        Args:
            module: The module to take the direct power of
            power: The number of copies to include in the direct sum

        Returns:
            A module representing the direct power M^⊕power

        Raises:
            ValueError: If power is negative
        """
        if power < 0:
            raise ValueError("Power must be non-negative")

        if power == 0:
            # Return a zero module
            zero_dimensions = {v: 0 for v in module.quiver.get_vertices()}

            # Create zero maps for all arrows
            zero_maps = {}
            for arrow_id in module.quiver.get_arrows():
                arrow = module.quiver.arrows[arrow_id]
                # Both dimensions are zero, so we create a zero map
                zero_maps[arrow_id] = ZeroMap(0, 0)

            return Module(module.quiver, module.field,
                         name="Zero",
                         dimensions=zero_dimensions,
                         maps=zero_maps)

        if power == 1:
            # Return a copy of the original module
            return module.copy()

        # Compute dimensions for the direct power
        dimensions = {}
        for vertex_id in module.quiver.get_vertices():
            dim = module.spaces.get(vertex_id, 0)
            dimensions[vertex_id] = dim * power

        # Prepare maps for the direct power
        maps = {}
        for arrow_id in module.quiver.get_arrows():
            arrow = module.quiver.arrows[arrow_id]
            source_id = arrow["source"]
            target_id = arrow["target"]

            source_dim = module.spaces.get(source_id, 0)
            target_dim = module.spaces.get(target_id, 0)

            source_dim_total = source_dim * power
            target_dim_total = target_dim * power

            # Skip if total dimensions are zero
            if source_dim_total == 0 or target_dim_total == 0:
                maps[arrow_id] = ZeroMap(source_dim_total, target_dim_total)
                continue

            # Get the original map
            orig_map = module.maps.get(arrow_id)

            # Create block diagonal matrix with 'power' copies of the original map
            if isinstance(orig_map, ZeroMap) or source_dim == 0 or target_dim == 0:
                # Original map is a zero map or dimensions are zero
                maps[arrow_id] = ZeroMap(source_dim_total, target_dim_total)
                continue

            result_map = module.field.zero_matrix(target_dim_total, source_dim_total)

            # Fill in the blocks with copies of the original map
            for i in range(power):
                result_map[i*target_dim:(i+1)*target_dim,
                          i*source_dim:(i+1)*source_dim] = orig_map

            maps[arrow_id] = result_map

        # Create the module with dimensions and maps
        return cls(module.quiver, module.field,
                     name=f"({module.name})^⊕{power}",
                     dimensions=dimensions,
                     maps=maps)

    def radical(self) -> Tuple['Module', 'Morphism']:
        """
        Compute the radical of the module.

        Returns:
            A tuple (R, i) where:
            - R is a module representing the radical
            - i is the inclusion morphism from R to the original module

        The radical at vertex i is the image of the sum of all incoming maps.
        """
        # Gather dimensions, bases, and maps before creating the module
        radical_dimensions = {}
        inclusion_maps = {}
        radical_maps = {}
        radical_bases = {}  # Store bases for computing maps between radical spaces

        # First pass: compute dimensions and bases at each vertex
        for vertex_id in self.quiver.get_vertices():
            target_dim = self.spaces[vertex_id]

            # If target space is zero-dimensional, radical component is also zero
            if target_dim == 0:
                radical_dimensions[vertex_id] = 0
                continue

            # Get all incoming arrows to this vertex (excluding self-loops)
            incoming_arrows = [(source_id, arrow_id) for source_id, arrow_id in self.quiver.predecessors[vertex_id]
                             if source_id != vertex_id]

            # If no incoming arrows, radical component is zero
            if not incoming_arrows:
                radical_dimensions[vertex_id] = 0
                continue

            # Collect matrices to concatenate
            matrices_to_concat = []
            for source_id, arrow_id in incoming_arrows:
                arrow_map = self.maps[arrow_id]

                # Skip ZeroMaps
                if isinstance(arrow_map, ZeroMap):
                    continue

                # If source has zero dimension, skip this map
                if self.spaces[source_id] == 0:
                    continue

                matrices_to_concat.append(arrow_map)

            # If no valid matrices to concatenate, radical component is zero
            if not matrices_to_concat:
                radical_dimensions[vertex_id] = 0
                continue

            # Horizontally concatenate matrices
            concat_matrix = np.hstack(matrices_to_concat)

            # Get a basis for the column space (radical component)
            radical_basis = self.field.column_space_basis(concat_matrix)

            # Compute and store radical dimension
            radical_dim = radical_basis.shape[1]
            radical_dimensions[vertex_id] = radical_dim

            # Store the basis and inclusion map
            radical_bases[vertex_id] = radical_basis
            inclusion_maps[vertex_id] = radical_basis

        # Second pass: compute maps between radical spaces
        for arrow_id in self.quiver.get_arrows():
            arrow = self.quiver.arrows[arrow_id]
            source_vertex = arrow["source"]
            target_vertex = arrow["target"]

            source_radical_dim = radical_dimensions[source_vertex]
            target_radical_dim = radical_dimensions[target_vertex]

            # If either radical space is zero-dimensional, use a ZeroMap
            if source_radical_dim == 0 or target_radical_dim == 0:
                radical_maps[arrow_id] = ZeroMap(source_radical_dim, target_radical_dim)
                continue

            # Get the original map
            original_map = self.maps[arrow_id]
            if original_map is None or isinstance(original_map, ZeroMap):
                radical_maps[arrow_id] = self.field.zero_matrix(target_radical_dim, source_radical_dim)
                continue

            # Get the bases for source and target radical spaces
            source_basis = radical_bases[source_vertex]
            target_basis = radical_bases[target_vertex]

            # Apply the original map to the source basis
            mapped_source = original_map @ source_basis

            # Express the mapped vectors in terms of the target basis
            radical_map = self.field.find_matrix_coordinates(target_basis, mapped_source)
            if radical_map is None:
                raise ValueError(f"Failed to compute radical map for arrow {arrow_id}")

            # Store the radical map
            radical_maps[arrow_id] = radical_map

        # Create the radical module
        radical_module = Module(
            self.quiver,
            self.field,
            name=f"rad({self.name})",
            dimensions=radical_dimensions,
            maps=radical_maps
        )

        # Create the inclusion morphism
        inclusion = Morphism(radical_module, self, name=f"incl_rad_{self.name}")

        # Set the inclusion maps
        for vertex_id in self.quiver.get_vertices():
            radical_dim = radical_dimensions[vertex_id]
            original_dim = self.spaces[vertex_id]

            if radical_dim == 0 or original_dim == 0:
                inclusion.maps[vertex_id] = ZeroMap(radical_dim, original_dim)
            else:
                inclusion.maps[vertex_id] = inclusion_maps[vertex_id]

        return (radical_module, inclusion)

    def socle(self) -> Tuple['Module', 'Morphism']:
        """
        Compute the socle of the module.

        Returns:
            A tuple (S, i) where:
            - S is a module representing the socle
            - i is the inclusion morphism from S to the original module

        The socle at vertex i is the kernel of the sum of all outgoing maps.
        """
        # Gather dimensions, bases, and maps before creating the module
        socle_dimensions = {}
        inclusion_maps = {}
        socle_maps = {}
        socle_bases = {}  # Store bases for computing maps between socle spaces

        # First pass: compute dimensions and bases at each vertex
        for vertex_id in self.quiver.get_vertices():
            source_dim = self.spaces[vertex_id]

            # If source space is zero-dimensional, socle component is also zero
            if source_dim == 0:
                socle_dimensions[vertex_id] = 0
                continue

            # Get all outgoing arrows from this vertex (excluding self-loops)
            outgoing_arrows = [(target_id, arrow_id) for target_id, arrow_id in self.quiver.successors[vertex_id]
                             if target_id != vertex_id]

            # If no outgoing arrows, entire space is in socle
            if not outgoing_arrows:
                socle_dimensions[vertex_id] = source_dim
                socle_bases[vertex_id] = self.field.identity_matrix(source_dim)
                inclusion_maps[vertex_id] = self.field.identity_matrix(source_dim)
                continue

            # Collect matrices to vertically stack
            matrices_to_stack = []
            for target_id, arrow_id in outgoing_arrows:
                arrow_map = self.maps[arrow_id]

                # Skip ZeroMaps
                if isinstance(arrow_map, ZeroMap):
                    continue

                # If target has zero dimension, skip this map
                if self.spaces[target_id] == 0:
                    continue

                matrices_to_stack.append(arrow_map)

            # If no valid matrices to stack, entire space is in socle
            if not matrices_to_stack:
                socle_dimensions[vertex_id] = source_dim
                socle_bases[vertex_id] = self.field.identity_matrix(source_dim)
                inclusion_maps[vertex_id] = self.field.identity_matrix(source_dim)
                continue

            # Vertically stack matrices
            stacked_matrix = np.vstack(matrices_to_stack)

            # Get a basis for the kernel
            socle_basis = self.field.kernel_basis(stacked_matrix)

            # Compute and store socle dimension
            socle_dim = socle_basis.shape[1]
            socle_dimensions[vertex_id] = socle_dim

            # Store the basis and inclusion map
            socle_bases[vertex_id] = socle_basis
            inclusion_maps[vertex_id] = socle_basis

        # Second pass: compute maps between socle spaces
        for arrow_id in self.quiver.get_arrows():
            arrow = self.quiver.arrows[arrow_id]
            source_vertex = arrow["source"]
            target_vertex = arrow["target"]

            source_socle_dim = socle_dimensions[source_vertex]
            target_socle_dim = socle_dimensions[target_vertex]

            # If either socle space is zero-dimensional, use a ZeroMap
            if source_socle_dim == 0 or target_socle_dim == 0:
                socle_maps[arrow_id] = ZeroMap(source_socle_dim, target_socle_dim)
                continue

            # Get the original map
            original_map = self.maps[arrow_id]
            if original_map is None or isinstance(original_map, ZeroMap):
                socle_maps[arrow_id] = self.field.zero_matrix(target_socle_dim, source_socle_dim)
                continue

            # Get the bases for source and target socle spaces
            source_basis = socle_bases[source_vertex]
            target_basis = socle_bases[target_vertex]

            # Apply the original map to the source basis
            mapped_source = original_map @ source_basis

            # Express the mapped vectors in terms of the target basis
            socle_map = self.field.find_matrix_coordinates(target_basis, mapped_source)
            if socle_map is None:
                raise ValueError(f"Failed to compute socle map for arrow {arrow_id}")

            # Store the socle map
            socle_maps[arrow_id] = socle_map

        # Create the socle module
        socle_module = Module(
            self.quiver,
            self.field,
            name=f"soc({self.name})",
            dimensions=socle_dimensions,
            maps=socle_maps
        )

        # Create the inclusion morphism
        inclusion = Morphism(socle_module, self, name=f"incl_soc_{self.name}")

        # Set the inclusion maps
        for vertex_id in self.quiver.get_vertices():
            socle_dim = socle_dimensions[vertex_id]
            original_dim = self.spaces[vertex_id]

            if socle_dim == 0 or original_dim == 0:
                inclusion.maps[vertex_id] = ZeroMap(socle_dim, original_dim)
            else:
                inclusion.maps[vertex_id] = inclusion_maps[vertex_id]

        return (socle_module, inclusion)

    def projective_cover(module: 'Module') -> Tuple['Module', 'Morphism']:
        """
        Compute the projective cover of a module.

        A projective cover of a module M is a pair (P, q) where:
        - P is a projective module
        - q: P → M is an epimorphism (surjective morphism)
        - Ker(q) ⊆ rad(P) (the kernel is contained in the radical of P)
        - P is minimal with these properties

        Args:
            module: The module to find the projective cover for

        Returns:
            A tuple (P, q) where:
            - P is the projective cover module
            - q is the covering morphism P → M
        """
        quiver = module.quiver
        field = module.field

        # Step 1: Compute the radical of M and the quotient M/rad(M)
        rad_module, inclusion = module.radical()
        quotient_module, projection = inclusion.cokernel()

        # Step2: Find the dimension of M/rad(M) at each vertex
        # These dimensions tell us which projective modules we need and how many
        quotient_dimensions = quotient_module.get_dimension_vector()

        # Step 3: Create individual projective modules for each vertex with non-zero dimension in M/rad(M)
        vertex_to_projectives = {}  # Maps vertex_id -> list of projective modules for that vertex
        vertex_to_proj_p = {}  # Maps vertex_id -> list of projections P_v -> S_v for that vertex

        for vertex_id, dim in quotient_dimensions.items():
            if dim > 0:
                # For each vertex with non-zero dimension in the quotient,
                # we need 'dim' copies of the projective module P(i)
                proj_v, simple_v, cover_v = Module.projective(quiver, field, vertex_id)
                proj_v_pow, simple_v_pow, cover_v_pow = Morphism.direct_power(cover_v, dim)
                vertex_to_projectives[vertex_id] = proj_v_pow
                vertex_to_proj_p[vertex_id] = cover_v_pow

        # If we didn't find any projective modules, return appropriate defaults
        if not vertex_to_projectives:
            # Check if the module is zero
            total_dim = module.get_total_dimension()
            if total_dim == 0:
                # If module is zero, return another zero module and a zero morphism
                zero_dimensions = {v: 0 for v in quiver.get_vertices()}
                proj_cover = Module(quiver, field, "Zero", dimensions=zero_dimensions)
                cover_morphism = Morphism(module, proj_cover, "Zero morphism")

                # Initialize all zero maps for the morphism
                for vertex_id in quiver.get_vertices():
                    cover_morphism.set_map(vertex_id, ZeroMap(0, 0))

                return proj_cover, cover_morphism
            else:
                # If module M is not zero but M/rad M is zero, this is an error
                raise ValueError("Non-zero module coincides with its radical. That's a contradiction.")

        # Step 4: Combine all projective covers into one
        all_projective_covers = list(vertex_to_proj_p.values())

        # Combine all projective modules using direct sum
        if len(all_projective_covers) == 1:
            proj_cover = all_projective_covers[0]
        else:
            # Use reduce to apply direct_sum_modules successively
            from functools import reduce
            proj_cover, s_total, pi_P = reduce(Morphism.direct_sum, all_projective_covers)

        # Step 5: Create the projective cover morphism
        cover_morphism = Morphism(proj_cover, module, f"Projective cover of {module.name}")

        # Perform a topological sort of the quiver
        topological_order = topological_sort(quiver)

        # Keep track of which vertices we've already defined morphisms for
        defined_vertices = set()

        # Process vertices in topological order
        for vertex_id in topological_order:
            source_dim = proj_cover.spaces.get(vertex_id, 0)
            target_dim = module.spaces.get(vertex_id, 0)

            if source_dim == 0:
                # No projective component at this vertex
                cover_morphism.set_map(vertex_id, ZeroMap(source_dim, target_dim))
                continue

            # Step 5a: Get constraints from M/rad(M)
            # We need our morphism to be compatible with the projection to M/rad(M)
            # \pi^P_v = f_v\pi^M_v
            constraints_left = []
            constraints_right = []

            quotient_dim = quotient_module.spaces.get(vertex_id, 0)
            if quotient_dim > 0:
                pi_v = projection.get_map(vertex_id)
                constraints_left.append(pi_v)
                constraints_right.append(pi_P.get_map(vertex_id))

            # Step 5b: Get constraints from incoming arrows
            # For each incoming arrow, we have a commutativity constraint

            for source, arrow_id in quiver.predecessors[vertex_id]:
                if source not in defined_vertices:
                    raise ValueError("We haven't defined the morphism at the source yet")

                # Get the existing morphism map at the source vertex
                q_source = cover_morphism.get_map(source)

                # Get the module maps for this arrow
                module_map = module.maps.get(arrow_id)
                proj_map = proj_cover.maps.get(arrow_id)

                # Skip if either map is None or a ZeroMap
                if module_map is None or isinstance(module_map, ZeroMap) or proj_map is None or isinstance(proj_map, ZeroMap):
                    continue

                # The constraint is: q_v ∘ proj_map = module_map ∘ q_source
                # This is a constraint on q_v
                right_side = module_map @ q_source
                constraints_left.append(proj_map)
                constraints_right.append(right_side)

            # Step 5c: Combine constraints from M/rad(M) and incoming arrows
            # and solve for q_v
            # Basically, we need to stack all the constraints horizontally into large matrices
            # and then solve constraints_left^T q^T = constraints_right^T using find_matrix_coordinates
            # If we have no constraints, use a zero map
            if not constraints_left:
                q_v = field.zero_matrix(target_dim, source_dim)
                cover_morphism.set_map(vertex_id, q_v)
                defined_vertices.add(vertex_id)
                continue

            # Combine all constraints into stacked matrices
            coef_matrix = field.matrix(np.hstack(constraints_left))
            rhs_matrix = field.matrix(np.hstack(constraints_right))

            # Transpose to match the expected format for find_matrix_coordinates
            coef_matrix_T = coef_matrix.T
            rhs_matrix_T = rhs_matrix.T

            # Solve the system coef_matrix_T^T * q_v^T = rhs_matrix_T^T
            # This is equivalent to coef_matrix * q_v = rhs_matrix
            solution = field.find_matrix_coordinates(coef_matrix_T, rhs_matrix_T)

            if solution is None:
                # If no solution exists, this indicates a problem with the construction
                # This shouldn't happen for a valid projective cover
                raise ValueError(f"Failed to find a morphism satisfying all constraints at vertex {vertex_id}.")

            # Transpose the solution to get q_v
            q_v = solution.T

            # Set the map for this vertex
            cover_morphism.set_map(vertex_id, q_v)

            # Mark this vertex as defined
            defined_vertices.add(vertex_id)

        return proj_cover, cover_morphism

    def injective_hull(module: 'Module') -> Tuple['Module', 'Morphism']:
        """
        Compute the injective hull of a module.

        An injective hull of a module M is a pair (I, i) where:
        - I is an injective module
        - i: M → I is a monomorphism (injective morphism)
        - I is minimal with these properties

        Args:
            module: The module to find the injective hull for

        Returns:
            A tuple (I, i) where:
            - I is the injective hull module
            - i is the inclusion morphism M → I
        """
        quiver = module.quiver
        field = module.field

        # Step 1: Compute the socle of M
        socle_module, inclusion = module.socle()

        # Step 2: Find the dimension of socle(M) at each vertex
        # These dimensions tell us which injective modules we need and how many
        socle_dimensions = socle_module.get_dimension_vector()

        # Step 3: Create individual injective modules for each vertex with non-zero dimension in socle(M)
        vertex_to_injectives = {}  # Maps vertex_id -> list of injective modules for that vertex
        vertex_to_inj_i = {}  # Maps vertex_id -> list of inclusions S_v -> I_v for that vertex

        for vertex_id, dim in socle_dimensions.items():
            if dim > 0:
                # For each vertex with non-zero dimension in the socle,
                # we need 'dim' copies of the injective module I(v)
                inj_v, simple_v, inclusion_v = Module.injective(quiver, field, vertex_id)
                simple_v_pow, inj_v_pow, inclusion_v_pow = Morphism.direct_power(inclusion_v, dim)
                vertex_to_injectives[vertex_id] = inj_v_pow
                vertex_to_inj_i[vertex_id] = inclusion_v_pow

        # If we didn't find any injective modules, the socle must be zero
        if not vertex_to_injectives:
            # Check if the module is zero
            total_dim = module.get_total_dimension()
            if total_dim == 0:
                # If module is zero, return another zero module and a zero morphism
                zero_dimensions = {v: 0 for v in quiver.get_vertices()}
                inj_hull = Module(quiver, field, "Zero", dimensions=zero_dimensions)
                hull_morphism = Morphism(module, inj_hull, "Zero morphism")

                # Initialize all zero maps for the morphism
                for vertex_id in quiver.get_vertices():
                    hull_morphism.set_map(vertex_id, ZeroMap(0, 0))

                return inj_hull, hull_morphism
            else:
                # If module is not zero but socle is zero, this is an error
                raise ValueError("Non-zero module has zero socle. Every non-zero module should have a non-zero socle.")

        # Step 4: Combine all injective modules into one
        all_injective_inclusions = list(vertex_to_inj_i.values())

        # Combine all injective modules using direct sum
        if len(all_injective_inclusions) == 1:
            inj_hull = all_injective_inclusions[0]
        else:
            # Use reduce to apply direct_sum_modules successively
            from functools import reduce
            s_total, inj_hull, i_I = reduce(Morphism.direct_sum, all_injective_inclusions)

        # Step 5: Create the injective hull morphism
        hull_morphism = Morphism(module, inj_hull, f"Injective hull of {module.name}")

        # Perform a topological sort of the quiver and reverse it for injective hull
        topological_order = list(reversed(topological_sort(quiver)))

        # Keep track of which vertices we've already defined morphisms for
        defined_vertices = set()

        # Process vertices in reverse topological order
        for vertex_id in topological_order:
            source_dim = module.spaces.get(vertex_id, 0)
            target_dim = inj_hull.spaces.get(vertex_id, 0)

            if target_dim == 0:
                # No injective component at this vertex
                hull_morphism.set_map(vertex_id, ZeroMap(source_dim, target_dim))
                continue

            # Step 5a: Get constraints from socle(M)
            # We need our morphism to be compatible with the inclusion from socle(M)
            # If inj_M : socM -> M, inj_hull : M -> I, and inj_I: socM -> I,
            # then we should have inj_I = inj_hull * inj_M
            constraints_left = []
            constraints_right = []

            socle_dim = socle_module.spaces.get(vertex_id, 0)
            if socle_dim > 0:
                i_v = inclusion.get_map(vertex_id) # inj_M
                constraints_left.append(i_v)
                constraints_right.append(i_I.get_map(vertex_id)) # inj_hull

            # Step 5b: Get constraints from outgoing arrows
            # For each outgoing arrow, we have a commutativity constraint
            # For a: v -> w, we should have
            # inj_hull_w * M_a = I_a * inj_hull_v
            # Here, inj_hull_w should already be known

            for target, arrow_id in quiver.successors[vertex_id]:
                if target not in defined_vertices:
                    raise ValueError("We haven't defined the morphism at the target vertex yet")

                # Get the existing morphism map at the target vertex
                i_target = hull_morphism.get_map(target)

                # Get the module maps for this arrow
                module_map = module.maps.get(arrow_id)
                inj_map = inj_hull.maps.get(arrow_id) # I_a

                # Skip if either map is None or a ZeroMap
                if module_map is None or isinstance(module_map, ZeroMap) or inj_map is None or isinstance(inj_map, ZeroMap):
                    continue

                # The constraint is: inj_map ∘ i_v = i_target ∘ module_map
                # This is a constraint on i_v
                right_side = i_target @ module_map
                constraints_left.append(inj_map)
                constraints_right.append(right_side)

            # Step 5c: Combine constraints from socle(M) and outgoing arrows
            # and solve for i_v
            # If we have no constraints, use a zero map
            if not constraints_left:
                i_v = field.zero_matrix(target_dim, source_dim)
                hull_morphism.set_map(vertex_id, i_v)
                defined_vertices.add(vertex_id)
                continue

            # Combine all constraints into stacked matrices
            coef_matrix = field.matrix(np.vstack(constraints_left))
            rhs_matrix = field.matrix(np.vstack(constraints_right))

            # Solve the system coef_matrix_T^T * i_v^T = rhs_matrix_T^T
            # This is equivalent to coef_matrix * i_v = rhs_matrix
            solution = field.find_matrix_coordinates(coef_matrix, rhs_matrix)

            if solution is None:
                # If no solution exists, this indicates a problem with the construction
                # This shouldn't happen for a valid injective hull
                raise ValueError(f"Failed to find a morphism satisfying all constraints at vertex {vertex_id}.")

            # No need to transpose for the injective hull
            i_v = solution

            # Set the map for this vertex
            hull_morphism.set_map(vertex_id, i_v)

            # Mark this vertex as defined
            defined_vertices.add(vertex_id)

        return inj_hull, hull_morphism

    def is_valid(self) -> bool:
        """
        Check if the module is valid.

        A module is valid if:
        1. All dimensions are non-negative
        2. All maps have compatible dimensions

        Returns:
            True if the module is valid, False otherwise
        """
        # Check dimensions
        for vertex_id, dim in self.spaces.items():
            if dim < 0:
                return False

        # Check maps
        for arrow_id, matrix in self.maps.items():
            if matrix is None:
                continue

            arrow = self.quiver.arrows[arrow_id]
            source_id = arrow["source"]
            target_id = arrow["target"]
            source_dim = self.spaces[source_id]
            target_dim = self.spaces[target_id]

            if source_dim == 0 or target_dim == 0:
                # If either space is zero-dimensional, map should be None
                if matrix is not None:
                    return False
            else:
                # Check dimensions
                if matrix.shape != (source_dim, target_dim):
                    return False

        return True

    def is_simple(self) -> bool:
        """
        Check if the module is simple.

        A module is simple if it has dimension 1 at exactly one vertex
        and dimension 0 at all others.

        Returns:
            True if the module is simple, False otherwise
        """
        non_zero_vertices = [v for v, d in self.spaces.items() if d > 0]

        if len(non_zero_vertices) != 1:
            return False

        return self.spaces[non_zero_vertices[0]] == 1

    def to_dict(self) -> Dict:
        """Convert the module to a dictionary for serialization."""
        # Convert maps to nested lists for JSON serialization
        maps_dict = {}
        for arrow_id, matrix in self.maps.items():
            if matrix is not None:
                maps_dict[arrow_id] = matrix.tolist()
            else:
                maps_dict[arrow_id] = None

        return {
            "name": self.name,
            "quiver_name": self.quiver.name,
            "field_characteristic": self.field.characteristic,
            "field_degree": self.field.degree,
            "spaces": self.spaces,
            "maps": maps_dict
        }

    @classmethod
    def from_dict(cls, data: Dict, quiver, field) -> 'Module':
        """
        Create a module from a dictionary.

        Args:
            data: Dictionary representation of the module
            quiver: The quiver this module is defined over
            field: The finite field for vector spaces

        Returns:
            A new module
        """
        module = cls(quiver, field, name=data.get("name", ""))

        # Set dimensions
        module.spaces = data["spaces"]

        # Set maps
        for arrow_id, matrix_data in data["maps"].items():
            if matrix_data is not None:
                arrow_id = int(arrow_id)  # Convert from string key
                matrix = field.matrix(matrix_data)
                module.maps[arrow_id] = matrix
            else:
                module.maps[arrow_id] = None

        return module

    def save(self, filename: str) -> None:
        """Save the module to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str, quiver, field) -> 'Module':
        """
        Load a module from a JSON file.

        Args:
            filename: Path to the JSON file
            quiver: The quiver this module is defined over
            field: The finite field for vector spaces

        Returns:
            A new module
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, quiver, field)

    def copy(self) -> 'Module':
        """Create a deep copy of the module."""
        # Create a new module with the same dimensions and maps
        return Module(
            self.quiver,
            self.field,
            name=f"{self.name} (copy)",
            dimensions=copy.deepcopy(self.spaces),
            maps={arrow_id: (map_val.copy() if hasattr(map_val, 'copy') else map_val)
                  for arrow_id, map_val in self.maps.items()}
        )

    def __str__(self) -> str:
        """Concise string representation of the module."""
        s = f"Module '{self.name}' over {self.field}\n"
        s += "Dimension vector: " + ", ".join(
            f"{self.quiver.vertices[v]['label']}:{dim}"
            for v, dim in sorted(self.spaces.items())
        )
        return s

    def __repr__(self) -> str:
        """Detailed string representation showing dimensions and maps."""
        return self.details()

    def details(self, show_maps: bool = True) -> str:
        """
        Detailed string representation of the module.

        Args:
            show_maps: If True, includes the matrices for all maps

        Returns:
            Detailed string representation
        """
        s = str(self) + "\n"  # Start with the basic representation
        s += f"Total dimension: {self.get_total_dimension()}\n"

        if show_maps:
            s += "\nMaps:\n"
            for arrow_id in sorted(self.quiver.arrows.keys()):
                arrow = self.quiver.arrows[arrow_id]
                source_label = self.quiver.vertices[arrow["source"]]["label"]
                target_label = self.quiver.vertices[arrow["target"]]["label"]
                map_matrix = self.maps[arrow_id]

                s += f"\n{arrow['label']} ({source_label} → {target_label}):\n"
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

    @classmethod
    def zero_module(cls, quiver, field, vertex_id: int, name: Optional[str] = None) -> 'Module':
        """
        Create a simple module at the given vertex.

        A simple module has dimension 1 at the given vertex and 0 elsewhere.

        Args:
            quiver: The quiver
            field: The finite field
            vertex_id: The vertex to place the simple module at
            name: Optional name for the module

        Returns:
            A simple module
        """
        if vertex_id not in quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")

        if name is None:
            name = f"S({quiver.vertices[vertex_id]['label']})"

        # Set dimensions - 1 at the vertex, 0 elsewhere
        dimensions = {v: 1 if v == vertex_id else 0 for v in quiver.get_vertices()}

        # All maps are zero maps with appropriate dimensions
        maps = {}
        for arrow_id in quiver.get_arrows():
            arrow = quiver.arrows[arrow_id]
            source_id = arrow["source"]
            target_id = arrow["target"]
            source_dim = dimensions[source_id]
            target_dim = dimensions[target_id]

            maps[arrow_id] = ZeroMap(source_dim, target_dim)

        # Create the module with dimensions and maps
        return cls(quiver, field, name=name, dimensions=dimensions, maps=maps)

    @classmethod
    def simple(cls, quiver, field, vertex_id: int, name: Optional[str] = None) -> 'Module':
        """
        Create a simple module at the given vertex.

        A simple module has dimension 1 at the given vertex and 0 elsewhere.

        Args:
            quiver: The quiver
            field: The finite field
            vertex_id: The vertex to place the simple module at
            name: Optional name for the module

        Returns:
            A simple module
        """
        if vertex_id not in quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")

        if name is None:
            name = f"S({quiver.vertices[vertex_id]['label']})"

        # Set dimensions - 1 at the vertex, 0 elsewhere
        dimensions = {v: 1 if v == vertex_id else 0 for v in quiver.get_vertices()}

        # All maps are zero maps with appropriate dimensions
        maps = {}
        for arrow_id in quiver.get_arrows():
            arrow = quiver.arrows[arrow_id]
            source_id = arrow["source"]
            target_id = arrow["target"]
            source_dim = dimensions[source_id]
            target_dim = dimensions[target_id]

            maps[arrow_id] = ZeroMap(source_dim, target_dim)

        # Create the module with dimensions and maps
        return cls(quiver, field, name=name, dimensions=dimensions, maps=maps)

    @classmethod
    def projective(cls, quiver, field, vertex_id: int, name: Optional[str] = None) -> Tuple['Module', 'Module', 'Morphism']:
        """
        Create a projective module corresponding to the given vertex.

        The projective module P(i) has dimension equal to the number of paths
        from vertex i to vertex j at each vertex j.

        Args:
            quiver: The quiver
            field: The finite field
            vertex_id: The vertex to create the projective module for
            name: Optional name for the module

        Returns:
            A tuple (P, S, π) where:
            - P is the projective module P(i)
            - S is the simple module S(i)
            - π is the natural projection P(i) → S(i)
        """
        if vertex_id not in quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")

        if name is None:
            name = f"P({quiver.vertices[vertex_id]['label']})"

        # First, compute paths from vertex_id to each other vertex
        # We'll keep track of the actual paths, not just the count
        paths_to = {v: [] for v in quiver.get_vertices()}

        # The empty path from vertex_id to itself
        paths_to[vertex_id].append([])

        # BFS to find all paths
        queue = [(vertex_id, [])]  # (vertex, path so far)
        while queue:
            current, path_so_far = queue.pop(0)

            # For each outgoing arrow
            for target, arrow_id in quiver.successors[current]:
                new_path = path_so_far + [(arrow_id, current, target)]
                paths_to[target].append(new_path)
                queue.append((target, new_path))

        # Set dimensions based on number of paths to each vertex
        dimensions = {v: len(paths) for v, paths in paths_to.items()}

        # Now, construct the maps
        maps = {}
        for arrow_id in quiver.get_arrows():
            arrow = quiver.arrows[arrow_id]
            source = arrow["source"]
            target = arrow["target"]

            source_dim = dimensions[source]
            target_dim = dimensions[target]

            if source_dim == 0 or target_dim == 0:
                maps[arrow_id] = ZeroMap(source_dim, target_dim)
                continue

            # Initialize a zero map
            map_matrix = field.zero_matrix(target_dim, source_dim)

            # For each path to the source vertex
            for i, source_path in enumerate(paths_to[source]):
                # Extending this path with the current arrow
                extended_path = source_path + [(arrow_id, source, target)]

                # Check if the extended path is in the paths to the target vertex
                for j, target_path in enumerate(paths_to[target]):
                    if extended_path == target_path:
                        # Found a match - set the corresponding matrix entry to 1
                        map_matrix[j, i] = field.one
                        break

            maps[arrow_id] = map_matrix

        # Create the projective module with dimensions and maps
        proj_module = cls(quiver, field, name=name, dimensions=dimensions, maps=maps)

        # Create the simple module S(i)
        simple_module = cls.simple(quiver, field, vertex_id, name=f"S({quiver.vertices[vertex_id]['label']})")

        # Create the natural projection morphism P(i) → S(i)
        projection = Morphism(proj_module, simple_module, name=f"π: {name} → S({quiver.vertices[vertex_id]['label']})")

        # Set up the projection maps
        for v in quiver.get_vertices():
            source_dim = proj_module.spaces[v]
            target_dim = simple_module.spaces[v]


            if source_dim > 0 and target_dim > 0:
                # We expect only vertex_id to have non-zero dimensions in both modules
                if v != vertex_id:
                    raise ValueError(f"Expected only vertex {vertex_id} to have non-zero dimensions in both modules, but found vertex {v}")
                if target_dim != 1:
                    raise ValueError(f"Simple module should have dimension 1 at vertex {vertex_id}, but found dimension {target_dim}")

                # Find the index of the empty path in paths_to[vertex_id]
                empty_path_index = paths_to[vertex_id].index([])

                # Create projection map that maps the empty path to 1 and everything else to 0
                # This is a 1 × source_dim matrix
                proj_map = field.zero_matrix(1, source_dim)
                proj_map[0, empty_path_index] = field.one
                projection.set_map(v, proj_map)
            elif target_dim == 0:
                # Source has dimension but target doesn't - use zero map
                projection.set_map(v, ZeroMap(source_dim, target_dim))
            elif source_dim == 0 and target_dim > 0:
                # Target has dimension but source doesn't - should not happen for simple module
                raise ValueError(f"Unexpected non-zero dimension in simple module at vertex {v} where projective module has zero dimension")

        return proj_module, simple_module, projection

    @classmethod
    def injective(cls, quiver, field, vertex_id: int, name: Optional[str] = None) -> Tuple['Module', 'Module', 'Morphism']:
        """
        Create an injective module corresponding to the given vertex.

        The injective module I(i) has dimension equal to the number of paths
        from vertex j to vertex i at each vertex j.

        Args:
            quiver: The quiver
            field: The finite field
            vertex_id: The vertex to create the injective module for
            name: Optional name for the module

        Returns:
            A tuple (I, S, ι) where:
            - I is the injective module I(i)
            - S is the simple module S(i)
            - ι is the natural injection S(i) → I(i)
        """
        if vertex_id not in quiver.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")

        if name is None:
            name = f"I({quiver.vertices[vertex_id]['label']})"

        # First, compute paths from each vertex to vertex_id
        # We'll keep track of the actual paths, not just the count
        paths_from = {v: [] for v in quiver.get_vertices()}
        # The empty path from vertex_id to itself
        paths_from[vertex_id].append([])

        # BFS to find all paths (going backwards from vertex_id)
        queue = [(vertex_id, [])]  # (vertex, path so far)
        while queue:
            current, path_so_far = queue.pop(0)
            # For each incoming arrow
            for source, arrow_id in quiver.predecessors[current]:
                # Add path in reverse order for easier comparison later
                new_path = [(arrow_id, source, current)] + path_so_far
                paths_from[source].append(new_path)
                queue.append((source, new_path))

        # Set dimensions based on number of paths from each vertex
        dimensions = {v: len(paths) for v, paths in paths_from.items()}

        # Now, construct the maps
        maps = {}
        for arrow_id in quiver.get_arrows():
            arrow = quiver.arrows[arrow_id]
            source = arrow["source"]
            target = arrow["target"]
            source_dim = dimensions[source]
            target_dim = dimensions[target]

            if source_dim == 0 or target_dim == 0:
                maps[arrow_id] = ZeroMap(source_dim, target_dim)
                continue

            # Initialize a zero map
            map_matrix = field.zero_matrix(target_dim, source_dim)

            # For each path from the source vertex to vertex_id
            for i, source_path in enumerate(paths_from[source]):
                if not source_path:  # Skip empty path if present
                    continue

                # Check if removing the first arrow from source_path gives a valid path from target
                if source_path[0][0] == arrow_id:  # If this path starts with our arrow
                    remaining_path = source_path[1:]  # Remove the first arrow
                    # Look for this remaining path in the target's paths
                    for j, target_path in enumerate(paths_from[target]):
                        if remaining_path == target_path:
                            # Found a match - set the corresponding matrix entry to 1
                            map_matrix[j, i] = field.one
                            break

            maps[arrow_id] = map_matrix

        # Create the injective module with dimensions and maps
        inj_module = cls(quiver, field, name=name, dimensions=dimensions, maps=maps)

        # Create the simple module S(i)
        simple_module = cls.simple(quiver, field, vertex_id, name=f"S({quiver.vertices[vertex_id]['label']})")

        # Create the natural injection morphism S(i) → I(i)
        injection = Morphism(simple_module, inj_module, name=f"ι: S({quiver.vertices[vertex_id]['label']}) → {name}")

        # Set up the injection maps
        for v in quiver.get_vertices():
            source_dim = simple_module.spaces[v]
            target_dim = inj_module.spaces[v]

            if source_dim > 0 and target_dim > 0:
                # We expect only vertex_id to have non-zero dimensions in both modules
                if v != vertex_id:
                    raise ValueError(f"Expected only vertex {vertex_id} to have non-zero dimensions in both modules, but found vertex {v}")
                if source_dim != 1:
                    raise ValueError(f"Simple module should have dimension 1 at vertex {vertex_id}, but found dimension {source_dim}")

                # Find the index of the empty path in paths_from[vertex_id]
                empty_path_index = paths_from[vertex_id].index([])

                # Create injection map that maps 1 to the empty path and everything else to 0
                # This is a target_dim × 1 matrix
                inj_map = field.zero_matrix(target_dim, 1)
                inj_map[empty_path_index, 0] = field.one
                injection.set_map(v, inj_map)
            elif source_dim == 0:
                # The simple module has dim 0 here
                injection.set_map(v, ZeroMap(source_dim, target_dim))
            elif source_dim > 0 and target_dim == 0:
                # Source has dimension but target doesn't - should not happen for simple and injective
                raise ValueError(f"Unexpected case: simple module has dimension at vertex {v} but injective module doesn't")

        return inj_module, simple_module, injection

