"""
Homomorphism computations for quiver representations.

Provides functions to compute dimensions of Hom spaces between:
- Interval modules on A_n quivers
- General modules via morphism basis computation
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..module import Module
    from ..morphism import Morphism


def hom_interval(dir_edges, a, b, c, d):
    """
    dim Hom([a,b] -> [c,d]) for type-A with arbitrary orientation.
    dir_edges[k] = +1 if k -> k+1, and -1 if (k+1) -> k.

    Rule:
      Let [L,R] = overlap([a,b],[c,d]). If empty: 0.
      At the LEFT boundary (edge L-1 <-> L):
        - If domain sticks out left (a <= L-1 while c > L-1), we need the arrow k->k+1 (outside->inside),
          i.e. dir_edges[L-1] == +1.
        - If codomain sticks out left, we need (k+1)->k (inside->outside),
          i.e. dir_edges[L-1] == -1.
      At the RIGHT boundary (edge R <-> R+1):
        - If domain sticks out right (b >= R+1 while d < R+1), we need (R+1)->R (outside->inside),
          i.e. dir_edges[R] == -1.
        - If codomain sticks out right, we need R->R+1 (inside->outside),
          i.e. dir_edges[R] == +1.
    """
    n = len(dir_edges) + 1

    # overlap
    L = max(a, c)
    R = min(b, d)
    if L > R:
        return 0

    # LEFT boundary: edge (L-1, L) if it exists
    k = L - 1
    if k >= 0:
        dom_only = (a <= k <= b) and not (c <= k <= d)   # domain extends left beyond overlap
        cod_only = (c <= k <= d) and not (a <= k <= b)   # codomain extends left
        if dom_only and dir_edges[k] != +1:  # need outside->inside: k -> k+1 (= L)
            return 0
        if cod_only and dir_edges[k] != -1:  # need inside->outside: (k+1)=L -> k
            return 0

    # RIGHT boundary: edge (R, R+1) if it exists
    k = R
    if k <= n - 2:
        dom_only = (a <= k+1 <= b) and not (c <= k+1 <= d)  # domain extends right
        cod_only = (c <= k+1 <= d) and not (a <= k+1 <= b)  # codomain extends right
        if dom_only and dir_edges[k] != -1:  # need outside->inside: (R+1) -> R
            return 0
        if cod_only and dir_edges[k] != +1:  # need inside->outside: R -> (R+1)
            return 0

    return 1


def hom_interval_to_bag(dir_edges, i, j, bag):
    return sum(hom_interval(dir_edges, i, j, p, q) for (p, q) in bag)

def find_hom_basis(M: "Module", N: "Module") -> List["Morphism"]:
    """
    Compute a basis of Hom(M -> N) over the *finite field* carried by M,N.

    Conventions:
      - Unknowns: Φ_v ∈ Mat_{dim N(v) × dim M(v)} for each vertex v.
      - For each arrow a: u→v, impose N(a) Φ_u - Φ_v M(a) = 0.
      - vec(AXB) = (B^T ⊗ A) vec(X), column-major.
      - Never use ZeroMap in linear algebra; if it appears, skip the block that would require it.
    """
    F = M.field
    if F is not N.field:
        raise ValueError("M.field and N.field must be the same FiniteField object.")

    Q = M.quiver
    verts: List[int] = list(Q.get_vertices())
    a_ids: List[int] = list(Q.get_arrows())

    # Dimensions per vertex
    dimM: Dict[int, int] = {v: int(M.spaces[v]) for v in verts}
    dimN: Dict[int, int] = {v: int(N.spaces[v]) for v in verts}

    # Variable layout: vec(Φ_v) stacked by vertex v, in creation order
    var_start: Dict[int, int] = {}
    total_vars = 0
    for v in verts:
        ds = dimM[v]
        dt = dimN[v]
        var_start[v] = total_vars
        total_vars += dt * ds

    if total_vars == 0:
        return []  # Hom(0,0) is {0}

    # Count total rows = Σ_a (dim N(target(a)) * dim M(source(a)))
    total_rows = 0
    arrow_rows: Dict[int, int] = {}
    for a_id in a_ids:
        a = Q.arrows[a_id]
        u = int(a["source"]); v = int(a["target"])
        rows_here = dimN[v] * dimM[u]
        arrow_rows[a_id] = rows_here
        total_rows += rows_here

    # Allocate A over the field
    A = F.zero_matrix(total_rows, total_vars)

    # Cache identities
    I_cache: Dict[int, object] = {}
    def I(n: int):
        if n not in I_cache:
            I_cache[n] = F.identity_matrix(n) if n > 0 else F.zero_matrix(0, 0)
        return I_cache[n]

    # Fill A
    row_cursor = 0
    for a_id in a_ids:
        a = Q.arrows[a_id]
        u = int(a["source"]); v = int(a["target"])

        du   = dimM[u]      # dim M(u)
        dv   = dimM[v]      # dim M(v)
        dtu  = dimN[u]      # dim N(u)
        dtv  = dimN[v]      # dim N(v)

        rows_here = arrow_rows[a_id]
        if rows_here == 0:
            continue

        NA = N.maps[a_id]   # expected dtv x dtu (or ZeroMap when dtu==0 or dtv==0)
        MA = M.maps[a_id]   # expected dv  x du  (or ZeroMap when dv==0  or du==0)

        r0 = row_cursor
        r1 = r0 + rows_here

        # Column slices for vec(Φ_u) and vec(Φ_v)
        cu0 = var_start[u]; cu1 = cu0 + dtu * du   # length dtu*du
        cv0 = var_start[v]; cv1 = cv0 + dtv * dv   # length dtv*dv

        # Left block: (I_{du} ⊗ N(a)) at columns of vec(Φ_u), only if dtu > 0
        if dtu > 0:
            # If NA is ZeroMap (happens only when dtu==0 or dtv==0), we must skip;
            # here dtu>0 and rows_here>0 ⇒ dtv>0, so NA must be a proper matrix.
            left = np.kron(I(du), NA)
            A[r0:r1, cu0:cu1] = A[r0:r1, cu0:cu1] + left

        # Right block: (M(a)^T ⊗ I_{dtv}) at columns of vec(Φ_v), only if dv > 0
        if dv > 0:
            # If MA is ZeroMap (happens only when dv==0 or du==0), skip; here dv>0 and rows_here>0 ⇒ du>0.
            right = np.kron(MA.T, I(dtv))
            A[r0:r1, cv0:cv1] = A[r0:r1, cv0:cv1] - right

        row_cursor = r1

    # Kernel over the field
    if not hasattr(F, "kernel_basis"):
        raise TypeError("FiniteField is missing 'kernel_basis'; cannot solve Hom.")
    X = F.kernel_basis(A)   # expected shape (total_vars, nullity); columns = basis vectors
    if X is None:
        return []

    # Build Morphisms from kernel columns (column-major reshape per vertex)
    basis: List["Morphism"] = []
    nullity = X.shape[1] if hasattr(X, "shape") and len(X.shape) == 2 else 0
    for j in range(nullity):
        vec = X[:, j]
        phi = Morphism(M, N, name=f"hom_basis_{j}")

        for v in verts:
            ds = dimM[v]
            dt = dimN[v]
            if ds == 0 or dt == 0:
                # The only map possible here is the zero map; no numeric block needed.
                # (If you prefer, set an explicit ZeroMap(0,0) in phi.maps[v].)
                continue
            start = var_start[v]
            block = F.zero_matrix(dt, ds)
            slice_vec = vec[start:start + dt*ds]
            block[:, :] = np.reshape(slice_vec, (dt, ds), order='F')
            phi.maps[v] = block

        basis.append(phi)

    return basis

