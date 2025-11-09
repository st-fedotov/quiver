"""
Interval module construction for type A_n quivers.
"""

from typing import Iterable, Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
from ..module import Module


def build_interval_explicit(quiver, field, order: List[int], i: int, j: int, *, name: str | None = None) -> Module:
    """
    Build the interval module [i,j] on an arbitrary-orientation A_n quiver,
    using only explicit 0/1 maps (no radicals/projectives).

    order[k] = actual vertex id at linear position k (0..n-1).
    Maps on arrows with both endpoints inside {i..j} are 1×1 identities; otherwise zero (shape-correct).
    """
    n = len(order)
    if not (0 <= i <= j < n):
        raise ValueError(f"Need 0 <= i <= j < n, got [{i},{j}] with n={n}")

    # 1) vertex dimensions: 1 on the segment, 0 elsewhere
    spaces = {v: (1 if i <= pos <= j else 0) for pos, v in enumerate(order)}

    # 2) arrow maps: identity(1) if both ends in the segment; else zero of the right shape
    maps = {}
    I1 = field.identity_matrix(1)
    for a_id in quiver.get_arrows():
        u = quiver.arrows[a_id]["source"]
        v = quiver.arrows[a_id]["target"]
        du, dv = spaces[u], spaces[v]
        if du == 1 and dv == 1:
            maps[a_id] = I1
        elif du == 0 and dv == 0:
            # let the Module constructor fill a 0x0/ZeroMap; not strictly needed to set anything
            continue
        else:
            maps[a_id] = field.zero_matrix(dv, du)  # correct shape, dtype matches field

    return Module(quiver, field, name=(name or f"[{i},{j}]"), dimensions=spaces, maps=maps)


def form_An_module_from_intervals_explicit(
    quiver, field, intervals: List[Tuple[int, int]]
) -> Module:
    """
    Build ⊕_[i,j] [i,j] on an A_n quiver (any orientation) using explicit 0/1 maps.
    Intervals [i,j] and any position-keyed dims are interpreted in the insertion order
    returned by quiver.get_vertices() (your left→right creation order).
    """
    order = quiver.get_vertices()               # left→right, per your construction
    n = len(order)
    if not intervals:
        raise ValueError("Empty interval list is not allowed.")
    for (i, j) in intervals:
        if not (0 <= i <= j < n):
            raise ValueError(f"Interval [{i},{j}] is out of range for n={n}.")

    # Which summands pass through each vertex?
    present_at = {v: [] for v in order}         # v -> list of summand indices
    for s, (i, j) in enumerate(intervals):
        for pos in range(i, j + 1):
            present_at[order[pos]].append(s)

    # Vertex dimensions = number of summands at that vertex
    spaces = {v: len(lst) for v, lst in present_at.items()}

    # Arrow maps: ones along same summand across the arrow, else zero
    maps = {}
    for a_id in quiver.get_arrows():
        u = quiver.arrows[a_id]["source"]
        v = quiver.arrows[a_id]["target"]
        U = present_at[u]; V = present_at[v]
        du, dv = len(U), len(V)
        if du == 0 and dv == 0:
            continue
        A = field.zero_matrix(dv, du)
        pos_in_u = {s: idx for idx, s in enumerate(U)}
        for row, s in enumerate(V):
            col = pos_in_u.get(s)
            if col is not None:
                A[row, col] = 1
        if A.size:
            maps[a_id] = A

    label = " + ".join(f"[{i},{j}]" for (i, j) in intervals)
    return Module(quiver, field, name=label, dimensions=spaces, maps=maps)




def write_batch_rad_from_interval_bags_explicit(
    jobs_intervals: Iterable,
    *,
    field,
    batch_root: str = "batch_runs_rad",
    run_id: Optional[str] = None,
    max_path_len_full_default: int = 3,
    script_name: str = "rad.m2",
    out_stdout: str = "rad_out.txt",
    out_stderr: str = "rad_err.txt",
    docker_image: str = "m2-ppa",
    msys_no_pathconv: bool = True,
    overwrite_outputs: bool = False,
    vertex_order: Optional[List[int]] = None,
    prefix: str = "p_",
) -> Path:
    """
    Write batch jobs using explicit interval construction.

    Args:
        jobs_intervals: Iterable of job specifications
        field: Field (required)
        batch_root: Batch root directory
        run_id: Run identifier
        max_path_len_full_default: Max path length
        script_name: Script filename
        out_stdout: Stdout filename
        out_stderr: Stderr filename
        docker_image: Docker image
        msys_no_pathconv: MSYS flag
        overwrite_outputs: Overwrite flag
        vertex_order: Vertex ordering
        prefix: Variable prefix

    Returns:
        Path to batch directory

    For each job:
      - builds M via form_An_module_from_intervals_explicit(Q, field, bag),
      - expects dim_positions as a dict keyed by positions 0..n-1,
      - remaps to a vertex-ID keyed dict using Q.get_vertices(),
      - forwards (Q, M, dim_dict[, max_len_full]) to write_batch_rad_from_triples.
    """
    triples = []
    for item in tqdm(jobs):
        if len(item) == 3:
            Q, bag, dim_positions = item
            mplf = kwargs.get("max_path_len_full_default", 3)
        elif len(item) == 4:
            Q, bag, dim_positions, mplf = item
        else:
            raise ValueError("Each job must be (Q, intervals_bag, dim_positions[, max_len_full]).")

        # Build module explicitly (0/1 maps)
        M = form_An_module_from_intervals_explicit(Q, field, bag)
        if not getattr(M, "name", "").strip():
            M.name = " + ".join(f"[{i},{j}]" for (i, j) in bag) or "interval-sum"

        # --- strictly require a positions-keyed dict {0: d0, 1: d1, ...} ---
        order = Q.get_vertices()                # left→right insertion order you use
        n = len(order)

        if not isinstance(dim_positions, dict):
            raise TypeError("dim must be a dict keyed by positions 0..n-1; lists/tuples are not supported.")

        keys = set(dim_positions.keys())
        allowed = set(range(n))
        if not keys.issubset(allowed):
            raise ValueError(f"dim keys must be a subset of 0..{n-1}, got {sorted(keys)}.")

        # positions → vertex IDs
        dim = {order[k]: int(dim_positions.get(k, 0)) for k in range(n)}
        # --------------------------------------------------------------------

        triples.append((Q, M, dim, mplf))

    return write_batch_rad_from_triples(triples, **kwargs)
