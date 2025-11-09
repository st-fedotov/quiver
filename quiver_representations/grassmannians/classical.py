"""
Classical Grassmannian Gr(k,n) with Plücker coordinate computations.
"""

import numpy as np
import itertools
from typing import Dict, Tuple, List

import numpy as np
import itertools
from typing import Dict, Tuple, List

class Grassmannian:
    """
    A class for computing Plücker coordinates of subspaces in Grassmannians Gr(k,n).

    The Grassmannian Gr(k,n) consists of all k-dimensional subspaces of C^n.
    Plücker coordinates embed Gr(k,n) into projective space P^(C(n,k)-1).
    """

    def __init__(self, k: int, n: int, verbose=False):
        """
        Initialize the Grassmannian Gr(k,n).

        Args:
            k: Dimension of subspaces
            n: Dimension of ambient space
        """
        if k <= 0 or n <= 0:
            raise ValueError("k and n must be positive integers")
        if k > n:
            raise ValueError("k must be <= n")

        self.k = k
        self.n = n

        # Pre-compute all k-element combinations of {0, 1, ..., n-1}
        # These correspond to the basis elements of the k-th exterior power
        self.index_combinations = list(itertools.combinations(range(n), k))
        self.num_coordinates = len(self.index_combinations)

        if verbose:
            print(f"Initialized Gr({k},{n}) with {self.num_coordinates} Plücker coordinates")

    def plucker_coordinates(self, basis_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Plücker coordinates for a k-dimensional subspace.

        Args:
            basis_matrix: numpy array of shape (n, k) where each column is a basis vector

        Returns:
            numpy array of complex numbers representing Plücker coordinates
            in lexicographic order of index combinations
        """
        # Input validation
        if basis_matrix.shape != (self.n, self.k):
            raise ValueError(f"Expected matrix of shape ({self.n}, {self.k}), got {basis_matrix.shape}")

        # Ensure complex dtype
        if not np.iscomplexobj(basis_matrix):
            basis_matrix = basis_matrix.astype(complex)

        # Compute Plücker coordinates
        coordinates = np.zeros(self.num_coordinates, dtype=complex)

        for i, index_combo in enumerate(self.index_combinations):
            # Extract the k×k submatrix corresponding to rows given by index_combo
            submatrix = basis_matrix[np.ix_(index_combo, range(self.k))]
            # The Plücker coordinate is the determinant of this submatrix
            coordinates[i] = np.linalg.det(submatrix)

        return coordinates

    def get_index_mapping(self) -> Dict[Tuple[int, ...], int]:
        """
        Get mapping from index tuples to array positions.

        Returns:
            Dictionary mapping k-tuples of indices to their position in the coordinate array
        """
        return {combo: i for i, combo in enumerate(self.index_combinations)}

    def print_index_mapping(self):
        """Print the index mapping for reference."""
        print(f"Index mapping for Gr({self.k},{self.n}):")
        for i, combo in enumerate(self.index_combinations):
            print(f"  Position {i}: {combo}")

    def __repr__(self):
        return f"Grassmannian(k={self.k}, n={self.n}, num_coordinates={self.num_coordinates})"

