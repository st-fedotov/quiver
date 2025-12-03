"""
Interval module construction for type D_n quivers.
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from collections import Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math
from ..module import Module
from ..quiver import Quiver
from ..batch.core import write_batch_rad_from_triples



class DnKind(IntEnum):
    SEG  = 0   # [i, j]        (thin path on the spine; leaves 0,0); covers i..min(j, br)
    UP   = 1   # [i, up]       (spine i..br = 1; up-leaf = 1);  i==up_leaf => pure up simple
    DOWN = 2   # [i, down]     (spine i..br = 1; dn-leaf = 1);  i==dn_leaf => pure down simple
    BOTH = 3   # [i, both]     (spine i..br = 1; leaves up=1,down=1; requires i <= br)
    STAR = 4   # [i, j*]       (spine i..j-1 = 1; spine j..br = 2; leaves up=1,down=1)

@dataclass(frozen=True)
class DnIndec:
    kind: DnKind
    i: int
    j: int = -1  # unused for UP/DOWN/BOTH

Bag = List[DnIndec]

def bag_pretty(bag):
    return "[" + ", ".join(dn_tag(x) for x in sorted(bag, key=lambda y:(y.kind.value,y.i,y.j))) + "]"

def form_Dn_module_from_bag_explicit(
    quiver, field, bag: List[DnIndec],
) -> "Module":
    """
    Build ⊕ (indecomposables from 'bag') as an explicit 0/1/2 representation of a D_n quiver.

    Conventions:
      - Vertices are created left→right as v0, v1, ..., v_{n-1}.
      - The 'branch' is at br = v_{n-3}; leaves are up_leaf = v_{n-2} and dn_leaf = v_{n-1}.
      - Maps are identity on equal channel counts; otherwise orientation-aware injections/projections:
          * spine↔branch interface: 1→2 uses [1;1], 2→1 uses [1 1]
          * branch↔up: first coord ( [1 0] / [1;0] )
          * branch↔down: second coord ( [0 1] / [0;1] )
    """
    order = quiver.get_vertices()  # left→right
    n = len(order)
    if n < 4:
        raise ValueError("D_n requires n >= 4 vertices.")
    br = n - 3
    up_leaf = n - 2
    dn_leaf = n - 1

    if not bag:
        raise ValueError("Empty bag is not allowed.")

    # ---- validation ----
    for x in bag:
        if x.kind == DnKind.SEG:
            # SEG [i,j] covers spine positions i..j, so need 0 <= i <= j <= br
            if not (0 <= x.i <= x.j <= br):
                raise ValueError(f"[i,j]: need 0 <= i <= j <= br ({br}), got {x}.")
        elif x.kind == DnKind.UP:
            # UP [i,up]: i <= br covers spine [i..br] + up_leaf; i == up_leaf is pure simple
            if not (0 <= x.i <= br or x.i == up_leaf):
                raise ValueError(f"[i,up]: need 0 <= i <= br ({br}) or i == up_leaf ({up_leaf}), got {x}.")
        elif x.kind == DnKind.DOWN:
            # DOWN [i,down]: i <= br covers spine [i..br] + dn_leaf; i == dn_leaf is pure simple
            if not (0 <= x.i <= br or x.i == dn_leaf):
                raise ValueError(f"[i,down]: need 0 <= i <= br ({br}) or i == dn_leaf ({dn_leaf}), got {x}.")
        elif x.kind == DnKind.BOTH:
            if not (0 <= x.i <= br):
                raise ValueError(f"[i,both]: need 0 <= i <= br ({br}), got {x}.")
        elif x.kind == DnKind.STAR:
            # STAR [i,j*]: j must be <= br for the thick part to exist
            if not (0 <= x.i < x.j <= br):
                raise ValueError(f"[i,j*]: need 0 <= i < j <= br ({br}), got {x}.")
        else:
            raise ValueError(f"Unknown kind: {x.kind}")

    # ---- per-summand weights w_s[v] ∈ {0,1,2} ----
    weights: List[List[int]] = []
    for x in bag:
        w = [0] * n
        if x.kind == DnKind.SEG:
            # SEG [i,j] covers spine positions i..j
            for p in range(x.i, x.j + 1):
                w[p] = 1
        elif x.kind == DnKind.UP:
            if x.i <= br:
                for p in range(x.i, br + 1):
                    w[p] = 1
            elif x.i != up_leaf:
                raise ValueError(f"UP(i): i must be <= br or == up_leaf; got {x}.")
            w[up_leaf] = 1
        elif x.kind == DnKind.DOWN:
            if x.i <= br:
                for p in range(x.i, br + 1):
                    w[p] = 1
                w[dn_leaf] = 1
            elif x.i == dn_leaf:
                # [dn_leaf, down] is the simple module at dn_leaf
                w[dn_leaf] = 1
            else:
                raise ValueError(f"DOWN(i): i must be <= br or == dn_leaf; got {x}.")
        elif x.kind == DnKind.BOTH:
            for p in range(x.i, br + 1):
                w[p] = 1
            w[up_leaf] = 1
            w[dn_leaf] = 1
        elif x.kind == DnKind.STAR:
            if x.i <= x.j - 1:
                for p in range(x.i, min(x.j - 1, br) + 1):
                    w[p] = 1
            if x.j <= br:
                for p in range(x.j, br + 1):
                    w[p] = 2
            w[up_leaf] = 1
            w[dn_leaf] = 1
        weights.append(w)

    # ---- total dimensions per vertex ----
    spaces: Dict[int, int] = {order[pos]: 0 for pos in range(n)}
    for pos in range(n):
        spaces[order[pos]] = sum(w[pos] for w in weights)

    # ---- deterministic slot indices at each vertex ----
    slots: Dict[int, List[Tuple[int, int]]] = {order[pos]: [] for pos in range(n)}
    for pos in range(n):
        v = order[pos]
        for s, w in enumerate(weights):
            for k in range(w[pos]):
                slots[v].append((s, k))

    index_at: Dict[int, Dict[Tuple[int, int], int]] = {}
    for v in order:
        idx_map = {}
        for idx, (s, k) in enumerate(slots[v]):
            idx_map[(s, k)] = idx
        index_at[v] = idx_map

    pos_of = {order[i]: i for i in range(n)}
    v_branch = order[br]
    v_up = order[up_leaf]
    v_dn = order[dn_leaf]

    # ---- arrow maps (orientation-independent) ----
    maps: Dict[int, np.ndarray] = {}
    for a_id in quiver.get_arrows():
        u = quiver.arrows[a_id]["source"]
        v = quiver.arrows[a_id]["target"]
        du = spaces[u]
        dv = spaces[v]
        if du == 0 and dv == 0:
            continue
        A = field.zero_matrix(dv, du)

        upos = pos_of[u]
        vpos = pos_of[v]

        for s, w in enumerate(weights):
            wu = w[upos]  # channels at tail
            wv = w[vpos]  # channels at head
            if wu == 0 or wv == 0:
                continue

            if wu == wv:
                # identity on aligned channels (2↔2 or 1↔1 along the thick/thin spine)
                for t in range(wu):
                    col = index_at[u][(s, t)]
                    row = index_at[v][(s, t)]
                    A[row, col] = 1

            elif wu == 1 and wv == 2:
                # 1 -> 2 : choose coordinate by endpoint roles
                col = index_at[u][(s, 0)]
                if v == v_branch:
                    # incoming to branch: leafs pick a coordinate; spine picks e1+e2
                    if u == v_up:
                        A[index_at[v][(s, 0)], col] = 1  # into e1
                    elif u == v_dn:
                        A[index_at[v][(s, 1)], col] = 1  # into e2
                    else:
                        # spine neighbor → branch uses e1+e2
                        A[index_at[v][(s, 0)], col] = 1
                        A[index_at[v][(s, 1)], col] = 1
                else:
                    # spine interface into a thicker (2-channel) vertex away from branch
                    # use e1+e2 to keep STAR coherent
                    A[index_at[v][(s, 0)], col] = 1
                    A[index_at[v][(s, 1)], col] = 1

            elif wu == 2 and wv == 1:
                # 2 -> 1 : choose coordinate by endpoint roles
                row = index_at[v][(s, 0)]
                if u == v_branch:
                    if v == v_up:
                        A[row, index_at[u][(s, 0)]] = 1  # project first coord
                    elif v == v_dn:
                        A[row, index_at[u][(s, 1)]] = 1  # project second coord
                    else:
                        # branch → spine uses sum to hit ⟨e1+e2⟩
                        A[row, index_at[u][(s, 0)]] = 1
                        A[row, index_at[u][(s, 1)]] = 1
                else:
                    # spine interface from thicker (2) to thinner (1) away from branch: use sum
                    A[row, index_at[u][(s, 0)]] = 1
                    A[row, index_at[u][(s, 1)]] = 1

            else:
                # Only {0,1,2} used here
                raise ValueError(
                    f"Unsupported channel change {wu}->{wv} for summand {s} on arrow {a_id}."
                )

        if A.size:
            maps[a_id] = A

    # ---- compact multiplicity summary ----
    cnt = Counter(bag)
    parts = []
    for indec, m in sorted(cnt.items(), key=lambda x: (x[0].kind, x[0].i, x[0].j)):
        if indec.kind == DnKind.SEG:
            tag = f"[{indec.i},{indec.j}]"
        elif indec.kind == DnKind.UP:
            tag = f"[{indec.i},up]"
        elif indec.kind == DnKind.DOWN:
            tag = f"[{indec.i},down]"
        elif indec.kind == DnKind.BOTH:
            tag = f"[{indec.i},both]"
        else:  # STAR
            tag = f"[{indec.i},{indec.j}*]"
        parts.append(f"{m}*{tag}")
    name = " + ".join(parts)

    return Module(quiver, field, name=name, dimensions=spaces, maps=maps)


def seg(i: int, j: int) -> DnIndec:
    return DnIndec(DnKind.SEG, i, j)

def up(i: int) -> DnIndec:
    return DnIndec(DnKind.UP, i)

def down(i: int) -> DnIndec:
    return DnIndec(DnKind.DOWN, i)

def both(i: int) -> DnIndec:
    return DnIndec(DnKind.BOTH, i)

def star(i: int, j: int) -> DnIndec:
    return DnIndec(DnKind.STAR, i, j)


# ---------- Make a D_n quiver (vertices added left→right) ----------
def make_Dn_quiver(n: int, name: str):
    """
    Build a D_n quiver with vertices v0,...,v{n-1} added left→right.
    Orientation: along the spine 0→1→...→(n-3); leaves (n-2)→(n-3), (n-1)→(n-3).
    """
    if n < 4:
        raise ValueError("D_n requires n >= 4.")
    Q = Quiver(name)
    verts = [Q.add_vertex(f"v{k}") for k in range(n)]
    # spine 0 -> 1 -> ... -> (n-3)
    for k in range(n - 3):
        Q.add_arrow(verts[k], verts[k+1], f"a{k}{k+1}")
    # leaves into the branch
    Q.add_arrow(verts[n-2], verts[n-3], f"a{n-2}{n-3}")
    Q.add_arrow(verts[n-1], verts[n-3], f"a{n-1}{n-3}")
    return Q, verts

# ---------- Map position-keyed dims to vertex-ID keyed dims ----------
def _posdims_to_vertexdims(Q, dim_pos: dict[int,int]) -> dict[int,int]:
    order = Q.get_vertices()          # left→right insertion order
    n = len(order)
    if set(dim_pos.keys()) - set(range(n)):
        raise ValueError(f"dim keys must be subset of 0..{n-1}, got {sorted(dim_pos)}")
    return {order[k]: int(dim_pos.get(k, 0)) for k in range(n)}

# ---------- Choose a concrete (but safe & nontrivial) dimension vector ----------
def choose_dims_positions(M) -> dict[int,int]:
    """
    Returns a dict {0: e0, ..., n-1: en-1}, with 0 < e_k < d_k whenever possible.
    Strategy: e_k = min(max(1, ceil(d_k/2)), max(1, d_k-1)).
    If d_k in {0,1}, we use e_k = d_k (point/empty factor is sometimes unavoidable).
    """
    order = M.quiver.get_vertices()
    e = {}
    for pos, v in enumerate(order):
        d = int(M.spaces.get(v, 0))
        if d <= 1:
            e[pos] = d
        else:
            e[pos] = max(1, min(math.ceil(d/2), d-1))
    return e


def write_batch_rad_from_dn_bags(
    jobs_bags: List,
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
    Write batch radicality test jobs for D_n quivers from bags of indecomposables.

    Args:
        jobs_bags: List of job specifications with bags
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
      - build M via form_Dn_module_from_bag_explicit(Q, field, bag)
      - if dim_positions absent, choose_dims_positions(M)
      - remap dim_positions (0..n-1) -> vertex-ID keyed
      - forward (Q, M, dim_vertex_id[, max_len_full]) to write_batch_rad_from_triples
    """
    triples = []
    for item in jobs_bags:
        if len(item) == 2:
            Q, bag = item
            dim_pos = None
            mplf = max_path_len_full_default
        elif len(item) == 3:
            Q, bag, third = item
            if isinstance(third, dict):
                dim_pos = third
                mplf = max_path_len_full_default
            else:
                dim_pos = None
                mplf = int(third)
        elif len(item) == 4:
            Q, bag, dim_pos, mplf = item
        else:
            raise ValueError("Each job must be (Q, bag[, dim_positions][, max_len_full]).")

        M = form_Dn_module_from_bag_explicit(Q, field, bag)
        if dim_pos is None:
            dim_pos = choose_dims_positions(M)

        dim_vertex = _posdims_to_vertexdims(Q, dim_pos)
        triples.append((Q, M, dim_vertex, mplf))

    # Reuse your batch emitter; will print run_all.sh and zip path.
    return write_batch_rad_from_triples(
        triples,
        batch_root=batch_root,
        run_id=run_id,
        max_path_len_full_default=max_path_len_full_default,
        script_name=script_name,
        out_stdout=out_stdout,
        out_stderr=out_stderr,
        docker_image=docker_image,
        msys_no_pathconv=msys_no_pathconv,
        overwrite_outputs=overwrite_outputs,
        vertex_order=vertex_order,
        prefix=prefix,
    )

