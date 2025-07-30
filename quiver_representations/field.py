import galois
import numpy as np
from typing import Union, List, Tuple, Optional, Any

import numba

class FiniteField:
    """
    A wrapper around the galois library for finite field operations.
    Provides field arithmetic, matrix operations, etc.
    """

    def __init__(self, characteristic: int, degree: int = 1):
        """
        Initialize a finite field GF(p^n).

        Args:
            characteristic: The characteristic p of the field (must be prime)
            degree: The degree n of the field extension (default: 1)
        """
        self.characteristic = characteristic
        self.degree = degree
        self.order = characteristic ** degree

        # Create the Galois field using the galois library
        self.GF = galois.GF(self.order)

        # Set up the zero and one elements
        self.zero = self.GF(0)
        self.one = self.GF(1)

    def element(self, value: Union[int, str, List[int]]) -> Any:
        """
        Create an element of the finite field.

        Args:
            value: Integer value, string representation, or list of coefficients for field extension

        Returns:
            An element of the finite field
        """
        return self.GF(value)

    def random_element(self) -> Any:
        """Generate a random element of the finite field."""
        return self.GF.Random()

    def matrix(self, values: Union[List[List[Union[int, str]]], np.ndarray]) -> np.ndarray:
        """
        Create a matrix over the finite field.

        Args:
            values: A 2D list or numpy array of values (can contain integers or string representations)

        Returns:
            A matrix over the finite field
        """

        # Standard conversion
        return self.GF(values)

    def random_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a random matrix over the finite field."""
        return self.GF.Random((rows, cols))

    def identity_matrix(self, size: int) -> np.ndarray:
        """Generate an identity matrix of the given size."""
        return self.GF.Identity(size)

    def zero_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a zero matrix of the given dimensions."""
        return self.GF.Zeros((rows, cols))

    def is_invertible(self, matrix: np.ndarray) -> bool:
        """
        Check if a matrix is invertible over the finite field.

        Args:
            matrix: The matrix to check

        Returns:
            True if the matrix is invertible, False otherwise
        """
        # Non-square matrices are not invertible
        if matrix.shape[0] != matrix.shape[1]:
            return False

        # Ensure matrix is in the correct field
        matrix_GF = self.GF(matrix) if not isinstance(matrix, self.GF) else matrix

        # A matrix is invertible if and only if its determinant is non-zero
        try:
            # Try to use galois's det function if we can access it
            try:
                from galois._domains._linalg import det_jit
                det_func = det_jit(self.GF)
                det = det_func(matrix_GF)
            except ImportError:
                # Fall back to matrix rank check
                n = matrix_GF.shape[0]
                # We can use our row_reduce function to compute rank
                reduced, pivots = self.row_reduce(matrix_GF)
                return pivots == n  # Full rank means invertible

            return det != 0
        except Exception as e:
            # If everything fails, try the most reliable but potentially slow approach
            try:
                self.inverse(matrix_GF)  # If this succeeds, matrix is invertible
                return True
            except ValueError:
                return False

    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of a matrix over the finite field.

        Args:
            matrix: The matrix to invert

        Raises:
            ValueError: If the matrix is not invertible

        Returns:
            The inverse matrix
        """
        # Ensure matrix is in the correct field
        matrix_GF = self.GF(matrix) if not isinstance(matrix, self.GF) else matrix

        # Check if it's a square matrix
        if matrix_GF.shape[0] != matrix_GF.shape[1]:
            raise ValueError("Matrix must be square to have an inverse")

        n = matrix_GF.shape[0]

        # Try to use galois's inv function if we can access it
        try:
            from galois._domains._linalg import inv_jit
            inv_func = inv_jit(self.GF)
            return inv_func(matrix_GF)
        except ImportError:
            # Fall back to our own implementation using row reduction
            # Augment the matrix with the identity matrix [A|I]
            I = self.GF.Identity(n)
            augmented = self.GF.Zeros((n, 2*n))
            augmented[:, :n] = matrix_GF
            augmented[:, n:] = I

            # Row reduce the augmented matrix
            reduced, pivots = self.row_reduce(augmented, ncols=n)

            # Check if the matrix is invertible (full rank)
            if pivots < n:
                raise ValueError("Matrix is not invertible (not full rank)")

            # Return the right part of the reduced matrix, which is A^(-1)
            return reduced[:, n:]

    def rank(self, matrix: np.ndarray) -> int:
        """
        Compute the rank of a matrix using column space basis.

        Args:
            matrix: Input matrix

        Returns:
            The rank of the matrix

        Raises:
            ValueError: If input is not a 2D matrix
        """
        if len(matrix.shape) != 2:
            raise ValueError("Input must be a 2D matrix")

        # Convert to GF matrix if needed
        if not hasattr(matrix, 'dtype') or matrix.dtype != self.GF.dtype:
            matrix = self.GF(matrix)

        # Get column space basis and return its number of columns
        basis = self.column_space_basis(matrix)
        return basis.shape[1]

    '''def nullity(self, matrix: np.ndarray) -> int:
        """Compute the nullity of a matrix (dimension of null space)."""
        if len(matrix.shape) != 2:
            raise ValueError("Input must be a 2D matrix")
        # Convert to GF matrix if needed
        if not hasattr(matrix, 'dtype') or matrix.dtype != self.GF.dtype:
            matrix = self.GF(matrix)
        return matrix.null_space().shape[1]'''

    def kernel_basis(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute a basis for the kernel (null space) of a matrix.

        Returns:
            A matrix whose columns form a basis for the kernel
        """
        # Convert to GF matrix if needed
        if not hasattr(matrix, 'dtype') or matrix.dtype != self.GF.dtype:
            matrix = self.GF(matrix)
        return matrix.null_space().T

    def left_kernel_basis(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute a basis for the left kernel of a matrix.

        Returns:
            A matrix whose rows form a basis for the left kernel
        """
        # Convert to GF matrix if needed
        if not hasattr(matrix, 'dtype') or matrix.dtype != self.GF.dtype:
            matrix = self.GF(matrix)
        return matrix.left_null_space()

    def row_reduce(self, matrix: np.ndarray, ncols: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Convert a matrix to row-reduced echelon form using Gaussian elimination.
        This is an optimized version that tries to use the galois library's JIT function if available.

        Args:
            matrix: The matrix to reduce
            ncols: Number of columns to consider for pivots (default: all columns)

        Returns:
            Tuple of (reduced matrix, number of pivots found)
        """
        # Ensure matrix is in the field
        matrix_GF = self.GF(matrix) if not isinstance(matrix, self.GF) else matrix

        if not matrix_GF.ndim == 2:
            raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {matrix_GF.ndim}-D.")

        # First, try to use galois's JIT-compiled row_reduce function if available
        try:
            # Try to import directly from the internal module
            from galois._domains._linalg import row_reduce_jit
            row_reduce_func = row_reduce_jit(self.GF)
            result = row_reduce_func(matrix_GF, ncols=ncols)
            # print("We are victorious!")
            return result
        except ImportError:
            # Try to access it through other means
            print("Cursed bastards!")

        # Fall back to our implementation, potentially with Numba JIT
        # If ncols not specified, use all columns
        if ncols is None:
            ncols = matrix_GF.shape[1]

        try:
            # Try to use our JIT-compiled version
            return self._jit_row_reduce(matrix_GF, ncols)
        except Exception:
            # Fall back to pure Python implementation if JIT fails
            print("Twice cursed bastards!")
            return self._py_row_reduce(matrix_GF, ncols)

    def _jit_row_reduce(self, matrix_GF, ncols):
        """
        Attempt to use Numba to JIT-compile the row reduction algorithm.
        This is a separate function to handle any JIT-specific logic.
        """
        # Get characteristics needed for JIT compile
        raise NotImplementedError()

    def _py_row_reduce(self, matrix_GF, ncols):
        """
        Pure Python implementation of row reduction as a fallback.
        This closely follows the algorithm from galois._domains._linalg.row_reduce_jit.
        """

        print("If you got here, you're doing manual matrix row reduction, and that means you're totally screwed up. Sorry")

        # Make a copy to avoid modifying the original
        A_rre = matrix_GF.copy()
        p = 0  # The pivot row

        for j in range(ncols):
            # Find a pivot in column j at or below row p
            idxs = np.nonzero(A_rre[p:, j])[0]
            if idxs.size == 0:
                continue
            i = p + idxs[0]  # Row with a pivot

            # Swap row p and i. The pivot is now located at row p.
            A_rre[[p, i], :] = A_rre[[i, p], :]

            # Force pivot value to be 1
            A_rre[p, :] /= A_rre[p, j]

            # Force zeros above and below the pivot
            idxs = np.nonzero(A_rre[:, j])[0].tolist()
            idxs.remove(p)
            A_rre[idxs, :] -= np.multiply.outer(A_rre[idxs, j], A_rre[p, :])

            p += 1
            if p == A_rre.shape[0]:
                break

        return A_rre, p

    def find_matrix_coordinates(self, basis, target, field=None):
        """
        Find the coordinates of multiple vectors in terms of the given basis.
        This function solves matrix equations AX = B, where each column of the
        target matrix is solved in parallel.

        Args:
            basis: Matrix whose columns form a basis
            target: Matrix whose columns are vectors to express in the basis
            field: Optional field instance (defaults to self)

        Returns:
            Matrix of coordinates, where each column contains the coordinates of the
            corresponding column of the target matrix in the basis. Returns None if
            any column is not in the span of the basis.
        """
        # Use self as the field if not provided
        if field is None:
            field = self

        # Ensure everything is in the right format
        basis_GF = self.GF(basis)
        target_GF = self.GF(target)

        # Get dimensions
        m, n = basis_GF.shape  # m×n basis matrix
        if len(target_GF.shape) == 1:
            # Single vector case, reshape to column vector
            target_GF = target_GF.reshape(-1, 1)

        m2, k = target_GF.shape  # m×k target matrix (should have m2 == m)

        if m != m2:
            raise ValueError(f"Dimension mismatch: basis has {m} rows but target has {m2} rows")

        # Initialize the solution matrix
        solution = self.GF.Zeros((n, k))

        # Use our optimized row_reduce function
        for col_idx in range(k):
            vector_GF = target_GF[:, col_idx]

            # Create augmented matrix [basis | vector]
            augmented = self.GF.Zeros((m, n + 1))
            augmented[:, :n] = basis_GF
            augmented[:, n] = vector_GF

            # Use our optimized row_reduce function
            ref, p = self.row_reduce(augmented, ncols=n)

            # Check if the system is consistent
            for i in range(p, m):
                if ref[i, n] != 0:
                    return None  # No solution, at least one vector not in span

            # Solve using back substitution
            col_solution = self.GF.Zeros(n)

            # Start from the last pivot row
            for i in range(p-1, -1, -1):
                # Find the pivot column
                pivot_col = -1
                for j in range(n):
                    if ref[i, j] == 1:
                        # Check if this is the first non-zero in the row
                        is_first = True
                        for k_idx in range(j):
                            if ref[i, k_idx] != 0:
                                is_first = False
                                break
                        if is_first:
                            pivot_col = j
                            break

                if pivot_col != -1:
                    # Calculate this variable's value
                    col_solution[pivot_col] = ref[i, n]

                    # Adjust for other variables' contributions
                    for j in range(pivot_col+1, n):
                        if ref[i, j] != 0:
                            col_solution[pivot_col] = col_solution[pivot_col] - ref[i, j] * col_solution[j]

            # Verify the solution
            if not all(basis_GF @ col_solution == vector_GF):
                return None

            # Store this column's solution
            solution[:, col_idx] = col_solution

        return solution

    def find_vector_coordinates(self, basis, vector, field=None):
        """
        Find the coordinates of a vector in terms of the given basis.
        This is a wrapper around find_matrix_coordinates for a single vector.

        Args:
            basis: Matrix whose columns form a basis
            vector: Vector to express in the basis
            field: Optional field instance (defaults to self)

        Returns:
            The coordinates of the vector in the basis, or None if the vector is not in the span
        """
        # Ensure vector is reshaped appropriately
        vector_reshaped = vector.reshape(-1, 1) if len(vector.shape) == 1 else vector

        # Call the matrix version with a single column
        result = self.find_matrix_coordinates(basis, vector_reshaped, field)

        # Return the result as a 1D array if it's not None
        if result is not None:
            return result.flatten() if result.shape[1] == 1 else result
        return None

    def column_space_basis(self, matrix: np.ndarray, do_express: bool = False, do_extend: bool = False,
                           with_express: bool = False, with_extend: bool = False) -> Union[
                               np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
                               ]:
        """
        Find a basis for the column space of a matrix.

        Args:
            matrix: The matrix whose column space basis is to be found
            do_express: If True, return expressions for non-basis columns in terms of basis columns
            do_extend: If True, extend the basis to a full basis of the ambient space

        Returns:
            If both do_express and do_extend are False:
                - A matrix whose columns form a basis for the column space
            If do_express is True:
                A tuple of (basis_matrix, non_basis_matrix, expressions_matrix) where:
                - basis_matrix: A matrix whose columns form a basis for the column space
                - non_basis_matrix: A matrix of the original columns not in the basis
                - expressions_matrix: A matrix where each column contains coefficients expressing
                  the corresponding non-basis column as a linear combination of basis columns
            If do_extend is True:
                - basis_columns: A matrix whose columns form a basis for the column space
                - extension_columns: A matrix whose columns extend the basis_columns to a basis for the entire space

        Note:
            do_express and do_extend cannot both be True simultaneously.
        """
        if do_express and do_extend:
            raise ValueError("do_express and do_extend cannot both be True")

        if with_express and with_extend:
            raise ValueError("with_express and with_extend cannot both be True")

        # Ensure the matrix is in the correct field
        matrix_GF = self.GF(matrix) if not isinstance(matrix, self.GF) else matrix

        # Get dimensions
        m, n = matrix_GF.shape  # m×n matrix

        if not (do_express or with_express) or with_extend: # Which means, do_extend, or with_extend, or nothing
            # To get a basis for the column space, we use the fact that the pivot rows
            # in the RREF of A^T correspond to the pivot columns of A.

            rref_transpose, pivot_count = self.row_reduce(matrix_GF.T)
            # rref_transpose is of shape (n, m) (rows of A^T)
            pivot_cols = []

            # Find pivot columns

            col = 0
            for row in range(pivot_count):
                # Find the pivot in this row
                while col < n and rref_transpose[row, col] == 0:
                    col += 1
                if col < n:  # Found a pivot
                    pivot_cols.append(col)
                    col += 1

            # Now, pivot_cols contains the indices of the columns in A that form a basis for its column space.
            basis_columns = rref_transpose[:pivot_count, :].T

            if not do_extend:
                return basis_columns

            # Extend the basis to a full basis of the ambient space F^m.
            # We choose additional standard basis vectors that are not already among our pivot columns.
            # (Here we assume that the pivot columns indices are valid in the ambient space.
            #  In many cases the pivot columns will lie in {0,1,...,m-1}.)
            non_basis_indices = [i for i in range(m) if i not in pivot_cols]
            if non_basis_indices:
                extension_columns = self.GF.Zeros((m, len(non_basis_indices)))
                for j, idx in enumerate(non_basis_indices):
                    extension_columns[idx, j] = 1
                #extended_basis = self.GF.Zeros((m, m))
                #extended_basis[:, :len(pivot_cols)] = basis_columns
                #extended_basis[:, len(pivot_cols):] = extension_columns
                #return extended_basis
                return basis_columns, extension_columns
            else:
                return basis_columns, None

        else:  # do_express
            # Calculate the RREF of the original matrix.
            rref, pivot_count = self.row_reduce(matrix_GF)

            # Find pivot columns from the RREF. We assume that pivot_count gives the number of nonzero rows.
            pivot_columns = []
            col = 0
            for row in range(pivot_count):
                # Advance in the row until we hit the first nonzero entry.
                while col < n and rref[row, col] == 0:
                    col += 1
                if col < n:
                    pivot_columns.append(col)
                    col += 1

            # Non-pivot columns (indices that are not in pivot_columns)
            non_pivot_columns = [j for j in range(n) if j not in pivot_columns]

            # Extract the corresponding original columns to form the basis.
            basis = matrix_GF[:, pivot_columns]

            if not do_express:
                # Just return the basis for the column space.
                return basis
            else:
                # Also return expressions for non-basis columns.
                non_basis = matrix_GF[:, non_pivot_columns]

                # The expressions come from the RREF: for each non-pivot column, express it in terms of pivot columns.
                expressions = self.GF.Zeros((len(pivot_columns), len(non_pivot_columns)))
                for i, non_piv_col in enumerate(non_pivot_columns):
                    for j, piv_col in enumerate(pivot_columns):
                        # Find the row in the RREF corresponding to the pivot in piv_col.
                        row_idx = next((r for r, pc in enumerate(pivot_columns) if pc == piv_col), None)
                        if row_idx is not None and row_idx < rref.shape[0]:
                            expressions[j, i] = rref[row_idx, non_piv_col]

                return basis, non_basis, expressions

    def __str__(self) -> str:
        """String representation of the finite field."""
        if self.degree == 1:
            return f"Finite Field GF({self.characteristic})"
        else:
            return f"Finite Field GF({self.characteristic}^{self.degree})"

class ComplexNumbers:
    """
    A wrapper for complex number operations.
    Provides field arithmetic, matrix operations, etc.
    """
    
    def __init__(self):
        """
        Initialize the complex numbers field.
        """
        self.characteristic = 0
        self.degree = 1
        self.order = float('inf')
        
        # Set up the zero and one elements
        self.zero = 0+0j
        self.one = 1+0j
    
    def element(self, value: Union[int, str, List[int]]) -> complex:
        """
        Create a complex number element.
        
        Args:
            value: Integer value, string representation, or list of coefficients
            
        Returns:
            A complex number
        """
        return complex(value)
    
    def random_element(self) -> complex:
        """Generate a random complex number."""
        return complex(np.random.randn(), np.random.randn())
    
    def matrix(self, values: Union[List[List[Union[int, str]]], np.ndarray]) -> np.ndarray:
        """
        Create a matrix over the complex numbers.
        
        Args:
            values: A 2D list or numpy array of values (can contain integers or string representations)
            
        Returns:
            A matrix over the complex numbers
        """
        return np.array(values, dtype=complex)
    
    def random_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a random matrix over the complex numbers."""
        return np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
    
    def identity_matrix(self, size: int) -> np.ndarray:
        """Generate an identity matrix of the given size."""
        return np.eye(size, dtype=complex)
    
    def zero_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a zero matrix of the given dimensions."""
        return np.zeros((rows, cols), dtype=complex)

    def is_invertible(self, matrix: np.ndarray) -> bool:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

    
    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")
    
    def rank(self, matrix: np.ndarray) -> int:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

    def kernel_basis(self, matrix: np.ndarray) -> np.ndarray:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

    
    def left_kernel_basis(self, matrix: np.ndarray) -> np.ndarray:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

        
    def row_reduce(self, matrix: np.ndarray, ncols: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

    
    def _jit_row_reduce(self, matrix_GF, ncols):
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        # Get characteristics needed for JIT compile
        raise NotImplementedError()
    
    
    def find_matrix_coordinates(self, basis, target, field=None):
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

    
    def find_vector_coordinates(self, basis, vector, field=None):
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")

    
    def column_space_basis(self, matrix: np.ndarray, do_express: bool = False, do_extend: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Deprecated due to being useless.
        Floating-point errors render all the ranks, kernels, images etc unfeasible over this field
        """
        raise ValueError("You are not supposed to use this")
    
    def __str__(self) -> str:
        """String representation of the field of complex numbers."""
        if self.degree == 1:
            return f"C (complex numbers)"
        else:
            return f"C (complex numbers)"


class ZeroMap:
    """
    A class representing a zero map between vector spaces.
    This is used instead of None for maps where one of the dimensions is zero.
    """

    def __init__(self, source_dim: int, target_dim: int):
        """
        Initialize a zero map.

        Args:
            source_dim: Dimension of the source space
            target_dim: Dimension of the target space
        """
        self.source_dim = source_dim
        self.target_dim = target_dim

    def __str__(self) -> str:
        return f"ZeroMap({self.source_dim} → {self.target_dim})"

    def __repr__(self) -> str:
        return self.__str__()
