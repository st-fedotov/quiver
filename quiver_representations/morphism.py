from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from .field import ZeroMap
from .morphism_algorithms import compute_cokernel, compute_image, compute_kernel

if TYPE_CHECKING:
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
        return compute_kernel(self)

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
        return compute_image(self)

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
        return compute_cokernel(self)

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
        from .module import Module

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

        return (
            source_sum, target_sum, result_morphism,
            source_incl1, source_incl2, source_proj1, source_proj2,
            target_sum, target_incl1, target_incl2, target_proj1, target_proj2
        )

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
        from .module import Module

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
