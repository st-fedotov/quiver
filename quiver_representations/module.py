from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from .field import Field, ZeroMap
from .module_algorithms import (
    compute_injective_hull,
    compute_projective_cover,
    compute_radical,
    compute_socle,
)
from .quiver import Quiver

if TYPE_CHECKING:
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
        from .morphism import Morphism

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
        return compute_radical(self)

    def socle(self) -> Tuple['Module', 'Morphism']:
        """
        Compute the socle of the module.

        Returns:
            A tuple (S, i) where:
            - S is a module representing the socle
            - i is the inclusion morphism from S to the original module

        The socle at vertex i is the kernel of the sum of all outgoing maps.
        """
        return compute_socle(self)

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
        return compute_projective_cover(module)

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
        return compute_injective_hull(module)

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
        from .morphism import Morphism

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
        from .morphism import Morphism

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

def get_p_plus_i_dim(Q: quiver):
    """
    Returns the sum of all indecomposable projectives and injectives for a given quiver Q
    """
    PI = Module.projective(Q, F, 0)[0]
    for i in range(1, len(Q.vertices)):
        PI = Module.direct_sum(PI, Module.projective(Q, F, i)[0])[0]

    p_dim = PI.get_dimension_vector()

    for i in range(0, len(Q.vertices)):
        PI = Module.direct_sum(PI, Module.injective(Q, F, i)[0])[0]

    return p_dim, PI.get_dimension_vector()




