"""
Quiver Grassmannian Gr_e(M) computations.
"""

import numpy as np
import itertools
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict

from ..utils.paths import PathRec
from .plucker import PluckerVar, PluckerPolynomial, EPS_PRUNE
from .classical import Grassmannian
from ..quiver import topological_sort
from ..field import ZeroMap
from .utils import poly_to_m2

class QuiverGrassmannian:
    """
    A class for computing Plücker coordinates of subrepresentations in quiver Grassmannians.

    The quiver Grassmannian Gr_e(M) consists of all subrepresentations of M with dimension vector e.
    For each vertex, we compute classical Plücker coordinates of the corresponding subspace.
    """

    def __init__(self, ambient_module, target_dimensions: Dict[int, int],
                 verbose=False):
        """
        Initialize the quiver Grassmannian Gr_e(M).

        Args:
            ambient_module: Module M representing the ambient representation
            target_dimensions: Dictionary mapping vertex_id -> target dimension at that vertex
        """
        self.ambient_module = ambient_module
        self.target_dimensions = target_dimensions.copy()
        self.quiver = ambient_module.quiver
        self.field = ambient_module.field

        # Validate target dimensions
        for vertex_id, target_dim in target_dimensions.items():
            if vertex_id not in self.quiver.vertices:
                raise ValueError(f"Vertex {vertex_id} does not exist in the quiver")
            if target_dim < 0:
                raise ValueError(f"Target dimension must be non-negative, got {target_dim}")

            ambient_dim = self.ambient_module.spaces[vertex_id]
            if target_dim > ambient_dim:
                raise ValueError(
                    f"Target dimension {target_dim} cannot exceed ambient dimension {ambient_dim} "
                    f"at vertex {vertex_id}"
                )

        # Pre-compute Grassmannian objects for each vertex
        self.vertex_grassmannians = {}
        self.vertex_coordinate_lengths = {}

        for vertex_id in self.quiver.get_vertices():
            target_dim = target_dimensions.get(vertex_id, 0)
            ambient_dim = self.ambient_module.spaces[vertex_id]

            if target_dim > 0 and ambient_dim > 0:
                # Create a classical Grassmannian for this vertex
                self.vertex_grassmannians[vertex_id] = Grassmannian(target_dim, ambient_dim)
                # Store the number of Plücker coordinates for this vertex
                self.vertex_coordinate_lengths[vertex_id] = len(
                    list(itertools.combinations(range(ambient_dim), target_dim))
                )
            else:
                # Zero dimension case - no coordinates needed
                self.vertex_grassmannians[vertex_id] = None
                self.vertex_coordinate_lengths[vertex_id] = 0

        if verbose:
            print(f"Initialized Gr_e(M) where:")
            print(f"  M has dimension vector: {dict(self.ambient_module.spaces)}")
            print(f"  e (target dimensions): {self.target_dimensions}")
            print(f"  Coordinate lengths per vertex: {self.vertex_coordinate_lengths}")

    def plucker_coordinates(self, subrepresentation, inclusion) -> Dict[int, np.ndarray]:
        """
        Compute Plücker coordinates for a subrepresentation.

        Args:
            subrepresentation: Module N representing the subrepresentation
            inclusion: Morphism f: N → M representing the inclusion

        Returns:
            Dictionary mapping vertex_id to numpy arrays of Plücker coordinates

        Raises:
            ValueError: If the input is not a valid subrepresentation with inclusion
        """
        # Validate the inclusion morphism
        self._validate_inclusion(subrepresentation, inclusion)

        # Compute Plücker coordinates at each vertex
        coordinates = {}

        for vertex_id in self.quiver.get_vertices():
            target_dim = self.target_dimensions.get(vertex_id, 0)
            ambient_dim = self.ambient_module.spaces[vertex_id]
            sub_dim = subrepresentation.spaces[vertex_id]

            if target_dim == 0 or ambient_dim == 0:
                # Zero dimension case - empty coordinates
                coordinates[vertex_id] = np.array([], dtype=complex)
                continue

            if sub_dim != target_dim:
                raise ValueError(
                    f"Subrepresentation dimension {sub_dim} does not match target dimension {target_dim} "
                    f"at vertex {vertex_id}"
                )

            # Get the inclusion map at this vertex
            inclusion_map = inclusion.get_map(vertex_id)

            if inclusion_map is None or isinstance(inclusion_map, ZeroMap):
                if target_dim > 0:
                    raise ValueError(f"Expected non-zero inclusion map at vertex {vertex_id}")
                coordinates[vertex_id] = np.array([], dtype=complex)
                continue

            # The inclusion map gives us the basis matrix for the subspace
            # Shape: (ambient_dim, sub_dim) where columns are basis vectors
            basis_matrix = inclusion_map

            # Compute classical Plücker coordinates using our Grassmannian class
            grassmannian = self.vertex_grassmannians[vertex_id]
            plucker_coords = grassmannian.plucker_coordinates(basis_matrix)

            coordinates[vertex_id] = plucker_coords

        return coordinates

    def _validate_inclusion(self, subrepresentation, inclusion) -> None:
        """
        Validate that the inclusion morphism represents a valid subrepresentation.

        Args:
            subrepresentation: The subrepresentation module
            inclusion: The inclusion morphism

        Raises:
            ValueError: If validation fails
        """
        # Check object identity
        if inclusion.source is not subrepresentation:
            raise ValueError("Inclusion source must be the same object as subrepresentation")

        if inclusion.target is not self.ambient_module:
            raise ValueError("Inclusion target must be the same object as ambient_module")

        # Check that the inclusion is fully defined where needed
        for vertex_id in self.quiver.get_vertices():
            sub_dim = subrepresentation.spaces[vertex_id]
            ambient_dim = self.ambient_module.spaces[vertex_id]

            if sub_dim > 0 and ambient_dim > 0:
                inclusion_map = inclusion.get_map(vertex_id)

                if inclusion_map is None:
                    raise ValueError(f"Inclusion map not defined at vertex {vertex_id}")

                if isinstance(inclusion_map, ZeroMap):
                    if sub_dim > 0:
                        raise ValueError(f"Cannot have zero map with non-zero source dimension at vertex {vertex_id}")
                else:
                    # Check dimensions
                    expected_shape = (ambient_dim, sub_dim)
                    if inclusion_map.shape != expected_shape:
                        raise ValueError(
                            f"Inclusion map has shape {inclusion_map.shape}, expected {expected_shape} "
                            f"at vertex {vertex_id}"
                        )

                    # Check injectivity (up to tolerance)
                    self._check_injectivity(inclusion_map, vertex_id)

    def _check_injectivity(self, matrix: np.ndarray, vertex_id: int, tolerance: float = 1e-10) -> None:
        """
        Check if a matrix represents an injective linear map.

        Args:
            matrix: The matrix to check (shape: target_dim × source_dim)
            vertex_id: Vertex ID for error reporting
            tolerance: Numerical tolerance

        Raises:
            ValueError: If the matrix is not injective
        """
        target_dim, source_dim = matrix.shape

        # For injectivity, we need the rank to equal source_dim
        if hasattr(self.field, 'rank'):
            # Use field-specific rank computation if available
            rank = self.field.rank(matrix)
        else:
            # Fall back to numpy rank computation
            rank = np.linalg.matrix_rank(matrix, tol=tolerance)

        if rank < source_dim:
            raise ValueError(
                f"Inclusion map is not injective at vertex {vertex_id}: "
                f"rank {rank} < source dimension {source_dim}"
            )

    def get_coordinate_structure(self) -> Dict[int, Dict]:
        """
        Get information about the coordinate structure.

        Returns:
            Dictionary with coordinate information for each vertex
        """
        structure = {}

        for vertex_id in self.quiver.get_vertices():
            target_dim = self.target_dimensions.get(vertex_id, 0)
            ambient_dim = self.ambient_module.spaces[vertex_id]
            coord_length = self.vertex_coordinate_lengths[vertex_id]

            grassmannian = self.vertex_grassmannians[vertex_id]
            index_mapping = grassmannian.get_index_mapping() if grassmannian else {}

            structure[vertex_id] = {
                'target_dim': target_dim,
                'ambient_dim': ambient_dim,
                'coordinate_length': coord_length,
                'index_mapping': index_mapping
            }

        return structure

    def print_coordinate_structure(self):
        """Print information about the coordinate structure."""
        print(f"Coordinate structure for Gr_e(M):")
        print(f"Target dimension vector e: {self.target_dimensions}")
        print(f"Ambient module dimension vector: {dict(self.ambient_module.spaces)}")
        print()

        for vertex_id in sorted(self.quiver.get_vertices()):
            vertex_label = self.quiver.vertices[vertex_id]['label']
            target_dim = self.target_dimensions.get(vertex_id, 0)
            ambient_dim = self.ambient_module.spaces[vertex_id]
            coord_length = self.vertex_coordinate_lengths[vertex_id]

            print(f"Vertex {vertex_id} ({vertex_label}):")
            print(f"  Gr({target_dim}, {ambient_dim}) → {coord_length} coordinates")

            if coord_length > 0:
                grassmannian = self.vertex_grassmannians[vertex_id]
                grassmannian.print_index_mapping()
            print()


    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def enumerate_paths(self, max_len: int) -> list[PathRec]:
        Q = self.quiver
        paths: list[PathRec] = []

        for v in Q.get_vertices():
            # k_v = int(self.target_dimensions.get(v, 0))
            n_v = int(self.ambient_module.spaces[v])
            # if k_v == 0 or n_v == 0:
            if n_v == 0:
                continue

            # identity in the correct field/domain
            M0 = self.field.identity_matrix(n_v)  # uses FiniteField.identity_matrix
            paths.append(PathRec(tail=v, arrows=tuple(), head=v, matrix=M0))

            # DFS stack: (tail, current_head, arrows_tuple, composite_matrix)
            stack: list[tuple[int, int, tuple[int, ...], object]] = [(v, v, tuple(), M0)]

            while stack:
                tail, cur, arrs, M = stack.pop()
                if len(arrs) == max_len:
                    continue

                # successors[cur] is a list of (target_vertex, arrow_id)
                for nxt, a_id in Q.successors.get(cur, []):  # use existing adjacency
                    #k_nxt = int(self.target_dimensions.get(nxt, 0))
                    n_nxt = int(self.ambient_module.spaces[nxt])
                    #if k_nxt == 0 or n_nxt == 0:
                    if n_nxt == 0:
                        continue

                    A = self.ambient_module.maps[a_id]
                    if isinstance(A, ZeroMap):
                        continue  # no relations and no further extension

                    M2 = A @ M                      # (n_nxt × n_cur) @ (n_cur × n_tail) → (n_nxt × n_tail)
                    arrs2 = arrs + (a_id,)
                    # DEBUG: view composite sparsity in *your* basis (rows=head, cols=tail)
                    nz = [(r, c) for r in range(M2.shape[0]) for c in range(M2.shape[1]) if M2[r, c] != 0]
                    # print(f"[composite] arrows={arrs2} head={nxt} tail={tail} shape={M2.shape} nonzeros={nz}")

                    paths.append(PathRec(tail=tail, arrows=arrs2, head=nxt, matrix=M2))
                    stack.append((tail, nxt, arrs2, M2))

        if not paths:
            print(f"[paths] No paths generated (max_len={max_len}).")
        return paths



    def get_relations_for_path(self, path: PathRec, *, cache: bool = True) -> list["PluckerPolynomial"]:
        """
        Single generator for ALL paths (length >= 0), matrix-driven.
        - Coalesces like terms per (I, J) into commutative monomials.
        - De-duplicates whole formulas up to scalar before returning.
        - Uses per-path cache keyed by (tail, arrows) when cache=True.
        """
        key = (path.tail, path.arrows)
        if cache and hasattr(self, "_path_rel_cache"):
            hit = self._path_rel_cache.get(key)
            if hit is not None:
                return hit

        tail, head = path.tail, path.head
        k = int(self.target_dimensions.get(tail, 0))
        l = int(self.target_dimensions.get(head, 0))

        if k == 0:
            # The zero subspace at the tail maps into every head subspace.
            out: list["PluckerPolynomial"] = []
        elif isinstance(path.matrix, ZeroMap):
            # print(f"[relations] skip path {path.arrows}: composite is ZeroMap.")
            out = []
        else:
            from itertools import combinations

            m = int(self.ambient_module.spaces[tail])
            n = int(self.ambient_module.spaces[head])
            A = path.matrix     # identity when len==0 and tail==head

            polys: list["PluckerPolynomial"] = []

            # Enumerate all I (size k-1) and J (size l+1) as per your design (no disjointness filter).
            for I in combinations(range(m), k - 1):
                I = tuple(I); Iset = set(I)
                for J in combinations(range(n), l + 1):
                    J = tuple(J)

                    # print(f"path={path.arrows}, A = {A}")

                    # --- coalescing accumulator over commutative monomials ---
                    # A zero-dimensional head has the single empty minor p_empty = 1,
                    # so its incidence equations are linear in the tail coordinates.
                    # Otherwise, retain the usual bilinear Plucker monomials.
                    # key: tuple of canonically ordered variable keys
                    # val: [coefficient sum, tuple of variable objects]
                    acc: dict[tuple, list] = {}

                    for r, j_r in enumerate(J):

                        # r = #(t in J : t <= j_r)
                        eps_j = (-1) ** (r+1)

                        for i_star in range(m):
                            if i_star in Iset:
                                continue

                            # parity of inserting i_star into I (ascending order)
                            sign_i = sum(1 for t in I if t <= i_star)
                            eps_i = (-1) ** sign_i


                            #if A[j_r, i_star] != 0:
                            #    print(f"[rel] path={path.arrows} I={I} J={J} -> use (j={j_r}, i={i_star}) "
                            #          f"p_left={tuple(sorted(I+(i_star,)))} p_right={tuple(x for x in J if x!=j_r)}")

                            coef = eps_i * eps_j * A[j_r, i_star]
                            if abs(coef) <= EPS_PRUNE:
                                continue

                            p_left  = tuple(sorted(I + (i_star,)))     # size k @ tail
                            p_right = tuple(x for x in J if x != j_r)  # size l @ head

                            a = PluckerVar('p', tail, p_left)
                            if l == 0:
                                monomial = (a,)
                            else:
                                b = PluckerVar('p', head, p_right)
                                monomial = tuple(sorted((a, b), key=lambda var: var.key()))
                            mk = tuple(var.key() for var in monomial)

                            if mk in acc:
                                acc[mk][0] += coef
                            else:
                                acc[mk] = [coef, monomial]

                    # Emit a relation only if nonzero after coalescing
                    terms = []
                    for _monomial_key, (c, monomial) in acc.items():
                        if abs(c) > EPS_PRUNE:
                            terms.append((c, monomial))

                    if terms:
                        meta = {
                            'type': 'incidence_path',
                            'arrows': path.arrows,
                            'tail': tail, 'head': head,
                            'I': I, 'J': J
                        }
                        polys.append(PluckerPolynomial(terms=terms, meta=meta))

            # --- whole-formula de-duplication (up to scalar) for this path ---
            uniq: list["PluckerPolynomial"] = []
            for P in polys:
                if not any(P.equals_up_to_scalar(Q) for Q in uniq):
                    uniq.append(P)
            out = uniq

        #if not out:
        #    print(f"[relations] No (nonzero) relations for path {path.arrows} (len={len(path.arrows)}).")

        if cache:
            if not hasattr(self, "_path_rel_cache"):
                self._path_rel_cache = {}
            self._path_rel_cache[key] = out
        return out


    from collections import defaultdict

    def get_relations_for_all_paths(
        self,
        max_len: int,
        *,
        cache: bool = True
    ) -> dict[tuple[int, int], list["PluckerPolynomial"]]:
        """
        Collect relations for ALL paths up to `max_len`, grouped by vertex pair.
          Key:   (start_vertex, end_vertex)
          Value: list of PluckerPolynomial aggregated over ALL paths connecting that pair.

        Behavior:
          • De-duplicates per-pair (up to scalar) before returning.
          • If cache=True, merges into self._pair_rel_cache with de-dup.
          • Returns only the relations produced in THIS call (not the entire cache).
        """
        # local aggregation for this call
        by_pair: dict[tuple[int, int], list["PluckerPolynomial"]] = defaultdict(list)

        paths = self.enumerate_paths(max_len=max_len)
        if not paths:
            print(f"[paths] No paths generated (max_len={max_len}).")

        for P in paths:
            rels = self.get_relations_for_path(P, cache=cache)
            if not rels:
                continue

            key = (P.tail, P.head)
            by_pair[key].extend(rels)

            # per-pair de-dup (up to scalar) for THIS call
            bucket = by_pair[key]
            uniq: list["PluckerPolynomial"] = []
            for R in bucket:
                if not any(R.equals_up_to_scalar(S) for S in uniq):
                    uniq.append(R)
            by_pair[key] = uniq

        if not by_pair:
            print(f"[relations] No relations generated for any path with max_len={max_len}.")

        # merge into internal cache if requested
        if cache:
            if not hasattr(self, "_pair_rel_cache"):
                self._pair_rel_cache: dict[tuple[int, int], list["PluckerPolynomial"]] = {}
            for k, L in by_pair.items():
                if k in self._pair_rel_cache:
                    merged = self._pair_rel_cache[k] + L
                    uniq: list["PluckerPolynomial"] = []
                    for R in merged:
                        if not any(R.equals_up_to_scalar(S) for S in uniq):
                            uniq.append(R)
                    self._pair_rel_cache[k] = uniq
                else:
                    self._pair_rel_cache[k] = list(L)

        # return only what was produced in THIS run
        return dict(by_pair)


    def flatten_relations_by_pair(self) -> list["PluckerPolynomial"]:
        """
        Flatten ALL cached relations from self._pair_rel_cache into a single list,
        with a final de-duplication up to scalar so repeated cached runs don't duplicate output.
        """
        cache = getattr(self, "_pair_rel_cache", None)
        if not cache:
            print("[cache] No cached relations to flatten. "
                  "Run get_relations_for_all_paths(..., cache=True) first.")
            return []

        flat: list["PluckerPolynomial"] = []
        for L in cache.values():
            flat.extend(L)

        # final de-dup across the full flattened list
        uniq: list["PluckerPolynomial"] = []
        for R in flat:
            if not any(R.equals_up_to_scalar(S) for S in uniq):
                uniq.append(R)
        return uniq

    def clear_incidence_cache(self):
        """Remove all cached incidence–relation data."""
        if hasattr(self, "_path_rel_cache"):
            del self._incidence_cache

    def to_macaulay2_saturated(
        self,
        *,
        max_path_len_full: int,
        vertex_order: list[int] | None = None,
        prefix: str = "p_",                 # your existing convention → p_(..., ...)
        script_path: str = "rad.m2",
        dump_gb: bool = True                # print full GBs before/after saturation
    ) -> None:
        """
        Production emitter:
          • I1 from paths of length <= 1
          • Ifull from paths of length <= max_path_len_full
          • ring with product/block order by vertex + multigrading
          • full Groebner bases (unsaturated + saturated)
          • ideal comparison & radicals (unsat and sat)
          • per-vertex saturation (successively by each block B#i)
          • summary (X_EMPTY / X_IRREDUCIBLE / X_EQDIM / X_COMPONENTS / X_CONNECTED)
        Relies on your existing poly_to_m2(P) that renders p_(...).
        """

        # ---------- included vertices ----------
        included: list[int] = []
        for v in self.quiver.get_vertices():
            if self.target_dimensions.get(v, 0) > 0 and self.ambient_module.spaces[v] > 0:
                included.append(v)
        if not included:
            raise ValueError("No nonzero-dimension vertices to export.")

        # ---------- vertex order (product/block order) ----------
        if vertex_order is None:
            topo = topological_sort(self.quiver)
            v_blocks = [v for v in topo if v in included]
        else:
            seen = set(vertex_order)
            v_blocks = [v for v in vertex_order if v in included]
            v_blocks += [v for v in included if v not in seen]

        # ---------- gather relations ----------
        rels_len1 = [P for L in self.get_relations_for_all_paths(max_len=1, cache=False).values() for P in L]
        rels_all  = [P for L in self.get_relations_for_all_paths(max_len=max_path_len_full, cache=False).values() for P in L]


        # ---------- FULL Plücker variable set per block (do NOT infer from relations) ----------
        from itertools import combinations

        block_vars: list[list[str]] = []
        block_vertices: list[int] = []

        for v in v_blocks:
            # ambient dimension at vertex v
            N_v = int(self.ambient_module.spaces[v])
            # subspace dimension at vertex v (from target dims)
            if v not in self.target_dimensions:
                raise ValueError(f"Missing target dimension for vertex {v}")
            k_v = int(self.target_dimensions[v])

            if not (0 <= k_v <= N_v):
                raise ValueError(f"k={k_v} out of range for vertex {v} with N={N_v}")

            if k_v == 0:
                # Gr(0, N_v) is a point; no homogeneous variables needed for this block
                block = []
            else:
                all_subsets = combinations(range(N_v), k_v)
                block = [f"{prefix}({v}," + ",".join(map(str, I)) + ")" for I in all_subsets]

            block_vars.append(block)
            block_vertices.append(v)

        ordered_vars: list[str] = [nm for block in block_vars for nm in block]
        if not ordered_vars:
            raise ValueError("No variables collected; nothing to emit.")

        # ---------- multidegrees (one-hot per block) ----------
        num_blocks = len(block_vars)
        degrees: list[list[int]] = []
        for j, block in enumerate(block_vars):
            for _ in block:
                deg = [0] * num_blocks
                deg[j] = 1
                degrees.append(deg)

        # ---------- serialize polynomials (using YOUR poly_to_m2) ----------
        gens_len1_m2 = [poly_to_m2(P) for P in rels_len1]
        gens_all_m2  = [poly_to_m2(P) for P in rels_all]

        # ---------- assemble Macaulay2 (ASCII only) ----------
        vars_m2 = ", ".join(ordered_vars)
        block_sizes = [len(block) for block in block_vars]
        mo = "{ " + ", ".join(str(n) for n in block_sizes) + " }"
        degs_m2 = "{ " + ", ".join("{" + ", ".join(str(d) for d in deg) + "}" for deg in degrees) + " }"

        lines: list[str] = []
        lines.append(
            f"""-- Auto-generated: product order by vertex, multigraded; p_(...) variable names
    R = QQ[{vars_m2}, MonomialOrder => {mo}, Degrees => {degs_m2}];
    use R;

    """
        )

        # generator lists (keep them strictly inside braces)
        def emit_gen_list(name, gens):
            if gens:
                body = "  " + ",\n".join(gens)
            else:
                body = ""
            lines.append(
                f"""{name} = {{
    {body}
    }};
    """
            )

        emit_gen_list("GensLenLe1", gens_len1_m2)
        lines.append('I1 = if #GensLenLe1 == 0 then ideal 0_R else ideal GensLenLe1;\n')
        emit_gen_list("GensAll", gens_all_m2)
        lines.append('Ifull = if #GensAll == 0 then ideal 0_R else ideal GensAll;\n')

        lines.append(
            """stdio << endl << "===== GENERATORS (ORIGINAL) =====" << endl;
    stdio << "orig gens I1    = " << #GensLenLe1 << endl;
    stdio << "orig gens Ifull = " << #GensAll << endl;

    -- Groebner bases (unsaturated)
    G1gb    = gb I1;
    Gfullgb = gb Ifull;
    stdio << "===== GROEBNER BASIS (I1) =====" << endl;
    scan(flatten entries gens G1gb,    g -> (stdio << toString g << endl));
    stdio << "===== GROEBNER BASIS (Ifull) =====" << endl;
    scan(flatten entries gens Gfullgb, g -> (stdio << toString g << endl));

    stdio << endl << "===== GROEBNER BASIS SIZES =====" << endl;
    stdio << "#cols gens gb(I1)    = " << numColumns (gens G1gb) << endl;
    stdio << "#cols gens gb(Ifull) = " << numColumns (gens Gfullgb) << endl;

    -- per-vertex blocks (irrelevant ideals)
    B = {
    """
        )
        # B from explicit blocks (we have the full Plücker lists now)
        for i, block in enumerate(block_vars):
            s = "ideal(" + ", ".join(block) + ")" if block else "ideal(1_R)"
            lines.append(("  " if i == 0 else " ,") + s + "\n")
        lines.append("};\n\n\n")

        lines.append(
            """-- per-vertex saturation (successively over blocks B#i)
    I1sat    = I1;
    Ifullsat = Ifull;
    for i from 0 to #B-1 do ( I1sat = saturate(I1sat, B#i); Ifullsat = saturate(Ifullsat, B#i) );

    stdio << "===== IDEAL COMPARISON (saturated) =====" << endl;
    stdio << "I1sat == Ifullsat ?             " << (I1sat == Ifullsat) << endl;
    stdio << "I1sat == radical(I1sat) ?       " << (I1sat == radical I1sat) << endl;
    stdio << "Ifullsat == radical(Ifullsat) ? " << (Ifullsat == radical Ifullsat) << endl;
    stdio << "rad(I1sat) == rad(Ifullsat) ?   " << (radical I1sat == radical Ifullsat) << endl;

    stdio << "===== ORDINARY vs SATURATED =====" << endl;
    stdio << "I1 == I1sat ?       " << (I1 == I1sat) << endl;
    stdio << "Ifull == Ifullsat ? " << (Ifull == Ifullsat) << endl;

    """
        )

        if dump_gb:
            lines.append(
                """-- Groebner bases (saturated)
    G1s   = gb I1sat;
    GFs   = gb Ifullsat;
    stdio << "===== GROEBNER BASIS (I1sat) =====" << endl;
    scan(flatten entries gens G1s, g -> (stdio << toString g << endl));
    stdio << "===== GROEBNER BASIS (Ifullsat) =====" << endl;
    scan(flatten entries gens GFs, g -> (stdio << toString g << endl));

    stdio << endl << "===== GROEBNER BASIS SIZES (saturated) =====" << endl;
    stdio << "#cols gens gb(I1sat)    = " << numColumns (gens G1s) << endl;
    stdio << "#cols gens gb(Ifullsat) = " << numColumns (gens GFs) << endl;

    """
            )

        # ---- metrics for mingens(Ifullsat) (unchanged)
        lines.append(
            r"""-- Minimal generators + degree/provenance metrics (using mingens)

    -- 1) Print minimal generators for I1sat
    M1min = mingens I1sat;                               -- minimal generator matrix
    stdio << "===== MINIMAL GENERATORS (I1sat) =====" << endl;
    scan(flatten entries M1min, g -> (stdio << toString g << endl));

    -- 2) Print minimal generators for Ifullsat
    M2min = mingens Ifullsat;                            -- minimal generator matrix
    stdio << "===== MINIMAL GENERATORS (Ifullsat) =====" << endl;
    scan(flatten entries M2min, g -> (stdio << toString g << endl));

    -- 3) Degree / provenance metrics for mingens(Ifullsat)

    degpair     = degrees M2min;                         -- {targetDegrees, sourceDegrees}
    colDegrees  = degpair#1;                             -- multidegrees of the columns (generators)
    tot         = d -> sum d;                            -- total degree = sum of multidegree components

    nDeg1       = #select(colDegrees, d -> tot d == 1);
    nDeg2       = #select(colDegrees, d -> tot d == 2);
    nDegGt2     = #select(colDegrees, d -> tot d >  2);

    -- provenance: compare minimal generators to original generators of Ifull (up to sign)
    MinGensList = flatten entries M2min;
    sameUpToSign = (g,h) -> (g == h) or (g == -h);
    nFromOrig   = #select(MinGensList, g -> any(GensAll, h -> sameUpToSign(g,h)));

    stdio << concatenate(
      "IFULLSAT_METRICS",
      " deg1=",     toString nDeg1,
      " deg2=",     toString nDeg2,
      " deggt2=",   toString nDegGt2,
      " from_orig=",toString nFromOrig
    ) << endl;

    """
        )

        # ---- variety summary for Ifullsat (keep, but use degreeLength R for proj dim)
        lines.append(
            r"""-- Fast diagnostics for X = V(Ifullsat)
    needsPackage "MinimalPrimes";                        -- minimalPrimes, radical, isPrime
    satMulti = I -> ( J := I; for i from 0 to #B-1 do J = saturate(J, B#i); J );
    b2i = b -> if b then 1 else 0;

    -- 1) Emptiness in the multiprojective sense (you already saturated Ifull per block)
    isEmptyX = (Ifullsat == ideal(1_R));
    stdio << "X_EMPTY isEmpty=" << b2i isEmptyX << endl;

    if isEmptyX then (
      stdio << "X_IRREDUCIBLE irreducible=0" << endl;
      stdio << "X_EQDIM equidimensional=0" << endl;
      stdio << "X_COMPONENTS num=0 compProjDims={} compAffDims={}" << endl;
      stdio << "X_CONNECTED connected=0" << endl;
    ) else (
      -- 2) Irreducible?  prime(radical(Ifullsat))
      JX = radical Ifullsat;
      isIrreducibleX = isPrime JX;
      stdio << "X_IRREDUCIBLE irreducible=" << b2i isIrreducibleX << endl;

      -- 3-4) Components and (projective/affine) dimensions from minimal primes
      mins     = minimalPrimes JX;
      numC     = #mins;
      affDims  = apply(mins, P -> dim(R/P));         -- Krull dims of Spec(R/P)

      -- Use the grading rank from the ring (robust)
      r = degreeLength R;
      projDims = apply(affDims, d -> d - r);         -- projective dims

      isEquidimensionalX = (#set projDims == 1);
      stdio << "X_EQDIM equidimensional=" << b2i isEquidimensionalX << endl;
      stdio << "X_COMPONENTS num=" << numC
           << " compProjDims=" << toString projDims
           << " compAffDims="  << toString affDims
           << endl;

      -- 5) Connected?  Build intersection graph; edges when two components meet projectively
      rr = numC;
      if rr <= 1 then (
        isConn = true;
      ) else (
        neighbors = new MutableList from (for i from 0 to rr-1 list set {});
        for i from 0 to rr-1 do for j from i+1 to rr-1 do (
          Sij = radical satMulti(mins#i + mins#j);   -- add radical for robustness
          if Sij != ideal(1_R) then (                -- nonempty intersection <-> not unit ideal
            neighbors#i = neighbors#i + set {j};
            neighbors#j = neighbors#j + set {i};
          );
        );
        visited  = set {0};
        frontier = set {0};
        while (#visited < rr and #frontier > 0) do (
          newF = set {};
          scan(toList frontier, i -> (
            scan(toList neighbors#i, j -> ( newF = newF + set {j}; ))
          ));
          frontier = newF - visited;
          visited  = visited + newF;
        );
        isConn = (#visited == rr);
      );
      stdio << "X_CONNECTED connected=" << b2i isConn << endl;
    );

    exit 0;
    """
        )

        # Write ASCII-only
        with open(script_path, "w", encoding="ascii", errors="strict") as fh:
            fh.write("\n".join(lines))


    # ----------------------------- GB cache API -----------------------------
    def set_cached_groebner_basis(
        self,
        gb_polynomials: list["PluckerPolynomial"],
    ) -> None:
        """
        Cache a previously computed Groebner basis (as PluckerPolynomials over QQ).
        Call this once after you parse 'gens gb I' from Macaulay2.

        Args:
            gb_polynomials: list of PluckerPolynomial (already over QQ), e.g. from parse_m2_gb_file(...)
        """
        if not isinstance(gb_polynomials, list) or not gb_polynomials:
            raise ValueError("gb_polynomials must be a non-empty list of PluckerPolynomial")
        self._cached_gb_polys = gb_polynomials

    def has_cached_groebner_basis(self) -> bool:
        return hasattr(self, "_cached_gb_polys") and bool(self._cached_gb_polys)




    def __repr__(self):
        total_coords = sum(self.vertex_coordinate_lengths.values())
        return (f"QuiverGrassmannian(target_dims={self.target_dimensions}, "
                f"total_coordinates={total_coords})")
