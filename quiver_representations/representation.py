import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import copy
import json

from .field import Field, ZeroMap
from .quiver import Quiver


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

                    if matrix is not None:

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
    def direct_sum(cls, module1: 'Module', module2: 'Module') -> Tuple['Module', 'Morphism', 'Morphism', 'Morphism', 'Morphism']:
        """
        Compute the direct sum of two modules with canonical morphisms.
        
        Args:
            module1: First module
            module2: Second module
            
        Returns:
            A tuple (M, i1, i2, p1, p2) where:
            - M is the direct sum module M1 ⊕ M2
            - i1: M1 → M is the first inclusion
            - i2: M2 → M is the second inclusion  
            - p1: M → M1 is the first projection
            - p2: M → M2 is the second projection
            
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

        # Create the direct sum module
        direct_sum_module = cls(quiver, field,
                                name=f"({module1.name})⊕({module2.name})",
                                dimensions=dimensions,
                                maps=maps)

        # Create the canonical inclusion morphisms
        # i1: M1 → M1 ⊕ M2
        inclusion1 = Morphism(module1, direct_sum_module, name=f"i1: {module1.name} → {direct_sum_module.name}")
        
        # i2: M2 → M1 ⊕ M2  
        inclusion2 = Morphism(module2, direct_sum_module, name=f"i2: {module2.name} → {direct_sum_module.name}")
        
        # p1: M1 ⊕ M2 → M1
        projection1 = Morphism(direct_sum_module, module1, name=f"p1: {direct_sum_module.name} → {module1.name}")
        
        # p2: M1 ⊕ M2 → M2
        projection2 = Morphism(direct_sum_module, module2, name=f"p2: {direct_sum_module.name} → {module2.name}")

        # Set the maps for each morphism at each vertex
        for vertex_id in quiver.get_vertices():
            dim1 = module1.spaces.get(vertex_id, 0)
            dim2 = module2.spaces.get(vertex_id, 0)
            dim_total = dim1 + dim2

            # Inclusion maps
            if dim1 > 0 and dim_total > 0:
                # i1 maps M1 into the first dim1 coordinates of M1 ⊕ M2
                i1_map = field.zero_matrix(dim_total, dim1)
                i1_map[:dim1, :] = field.identity_matrix(dim1)
                inclusion1.set_map(vertex_id, i1_map)
            elif dim1 == 0 or dim_total == 0:
                inclusion1.maps[vertex_id] = ZeroMap(dim1, dim_total)

            if dim2 > 0 and dim_total > 0:
                # i2 maps M2 into the last dim2 coordinates of M1 ⊕ M2
                i2_map = field.zero_matrix(dim_total, dim2)
                i2_map[dim1:, :] = field.identity_matrix(dim2)
                inclusion2.set_map(vertex_id, i2_map)
            elif dim2 == 0 or dim_total == 0:
                inclusion2.maps[vertex_id] = ZeroMap(dim2, dim_total)

            # Projection maps
            if dim_total > 0 and dim1 > 0:
                # p1 projects M1 ⊕ M2 onto the first dim1 coordinates
                p1_map = field.zero_matrix(dim1, dim_total)
                p1_map[:, :dim1] = field.identity_matrix(dim1)
                projection1.set_map(vertex_id, p1_map)
            elif dim_total == 0 or dim1 == 0:
                projection1.maps[vertex_id] = ZeroMap(dim_total, dim1)

            if dim_total > 0 and dim2 > 0:
                # p2 projects M1 ⊕ M2 onto the last dim2 coordinates
                p2_map = field.zero_matrix(dim2, dim_total)
                p2_map[:, dim1:] = field.identity_matrix(dim2)
                projection2.set_map(vertex_id, p2_map)
            elif dim_total == 0 or dim2 == 0:
                projection2.maps[vertex_id] = ZeroMap(dim_total, dim2)

        return direct_sum_module, inclusion1, inclusion2, projection1, projection2

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

    # ---------- helpers ----------
    def _check_compat(self, other: "Morphism"):
        if not isinstance(other, Morphism):
            raise TypeError("Can only add Morphism to Morphism")
        if self.quiver is not other.quiver:
            raise ValueError("Morphisms must be over the same quiver")
        if self.field is not other.field:
            raise ValueError("Morphisms must be over the same field")
        # same source/target dims at every vertex
        for v in self.quiver.get_vertices():
            if (self.source.spaces[v] != other.source.spaces[v] or
                self.target.spaces[v] != other.target.spaces[v]):
                raise ValueError("Incompatible vertex dimensions")

    def _zero_matrix_for_vertex(self, v: int):
        td = self.target.spaces[v]; sd = self.source.spaces[v]
        if td == 0 or sd == 0:
            return ZeroMap(sd, td)
        return self.field.zero_matrix(td, sd)

    def _add_maps(self, A, B, v: int):
        # ZeroMap handling
        if isinstance(A, ZeroMap) and isinstance(B, ZeroMap):
            return ZeroMap(A.source_dim, A.target_dim)
        if isinstance(A, ZeroMap):
            return B.copy() if hasattr(B, "copy") else B
        if isinstance(B, ZeroMap):
            return A.copy() if hasattr(A, "copy") else A
        return A + B

    def _scale_map(self, s, A):
        if isinstance(A, ZeroMap):
            return A  # s*0 = 0, keep ZeroMap
        return A * s

    def _coerce_scalar(self, s):
        # Try to enforce “scalar from the field”. If your Field exposes
        # is_element / coerce, we use them; otherwise we accept Python/NumPy scalars.
        if hasattr(self.field, "is_element"):
            if not self.field.is_element(s):
                raise TypeError("Scalar not in the base field")
            return s
        if hasattr(self.field, "coerce"):
            return self.field.coerce(s)
        # fallback: assume Python/NumPy numeric scalar is fine
        return s
        
    def __add__(self, other: "Morphism") -> "Morphism":
        self._check_compat(other)
        maps = {}
        for v in self.quiver.get_vertices():
            A = self.maps[v]
            B = other.maps[v]
            maps[v] = self._add_maps(A, B, v)
        return Morphism(self.source, self.target, name=f"({self.name}+{other.name})", maps=maps)

    # make sum([...]) work (sum starts with 0)
    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, scalar) -> "Morphism":
        s = self._coerce_scalar(scalar)
        maps = {}
        for v in self.quiver.get_vertices():
            maps[v] = self._scale_map(s, self.maps[v])
        return Morphism(self.source, self.target, name=f"{self.name}*{s}", maps=maps)

    # allow left scalar multiplication: s * morphism
    def __rmul__(self, scalar) -> "Morphism":
        s = self._coerce_scalar(scalar)
        maps = {}
        for v in self.quiver.get_vertices():
            maps[v] = self._scale_map(s, self.maps[v])
        return Morphism(self.source, self.target, name=f"{s}*{self.name}", maps=maps)

    def round(self, decimals: int = 0, inplace: bool = False) -> "Morphism":
        """
        Round numeric entries of each vertex map to the given number of decimals.
        - Raises ValueError on finite fields.
        - For complex arrays, rounds real and imaginary parts separately
          (NumPy's behavior).
        """
        # detect finite fields if your Field exposes a flag or order
        is_finite = bool(getattr(self.field, "is_finite", False)) \
                    or (hasattr(self.field, "order") and getattr(self.field, "order") not in (None, 0))

        if is_finite:
            raise ValueError("round() is not defined over finite fields")

        target = self if inplace else Morphism(self.source, self.target, name=self.name, maps=self.maps.copy())

        for v in self.quiver.get_vertices():
            A = target.maps[v]
            if isinstance(A, ZeroMap):
                continue
            # Only round floating/complex arrays; leave exact (int, rational) as-is
            if isinstance(A, np.ndarray) and (np.issubdtype(A.dtype, np.floating) or np.issubdtype(A.dtype, np.complexfloating)):
                target.maps[v] = np.round(A, decimals=decimals)
            else:
                # If your Field has its own rounding, you can hook it here:
                if hasattr(self.field, "round_matrix"):
                    target.maps[v] = self.field.round_matrix(A, decimals)
                else:
                    # default: no-op for exact types
                    target.maps[v] = A
        return target

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
        source_sum, source_incl1, source_incl2, source_proj1, source_proj2 = Module.direct_sum(f.source, g.source)

        # Compute direct sum of target modules
        target_sum, target_incl1, target_incl2, target_proj1, target_proj2 = Module.direct_sum(f.target, g.target)

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
