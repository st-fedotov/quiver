"""
Conjecture and hypothesis checking for A_n quivers.

Provides functions to:
- Read CSV outputs from batch runs
- Check specific conjectures about degeneration  
- Identify M0 and M1 modules
- Select generic jobs based on criteria

Notebook source: Cell 36
"""

from typing import List, Dict, Tuple, Union, Set
from pathlib import Path
import numpy as np
import csv
from collections import defaultdict


def read_ranks_csv_strict(path: str | Path):
    path = Path(path)
    rows = list(csv.DictReader(path.open("r", newline="")))
    if not rows:
        raise ValueError(f"No rows in {path}")

    header = rows[0].keys()
    if "id" not in header:
        raise ValueError("ranks.csv must contain an 'id' column.")

    rank_cols = [c for c in header if c.startswith("r_")]
    mult_cols = [c for c in header if c.startswith("x_")]
    if not rank_cols:
        raise ValueError("ranks.csv must contain rank columns r_i_j.")
    if not mult_cols:
        raise ValueError("ranks.csv must contain multiplicity columns x_a_b.")

    def _ij_from_rank(col: str) -> tuple[int,int]:
        parts = col.split("_")
        if len(parts) != 3:
            raise ValueError(f"Bad rank column name: {col}")
        return int(parts[1]), int(parts[2])

    def _ij_from_mult(col: str) -> tuple[int,int]:
        # x_a_b  -> (a,b)
        parts = col.split("_")
        if len(parts) != 3:
            raise ValueError(f"Bad multiplicity column name: {col}")
        return int(parts[1]), int(parts[2])

    # Infer n from BOTH rank and multiplicity columns (max index seen + 1)
    max_idx = -1
    for c in rank_cols:
        i,j = _ij_from_rank(c); max_idx = max(max_idx, i, j)
    for c in mult_cols:
        a,b = _ij_from_mult(c); max_idx = max(max_idx, a, b)
    n = max_idx + 1
    if n <= 0:
        raise ValueError("Failed to infer a positive number of vertices from CSV headers.")

    ids_unsorted, ranks_unsorted, mults_unsorted = [], {}, {}
    for row in rows:
        jid = int(row["id"])
        ids_unsorted.append(jid)

        Rj = {}
        for c in rank_cols:
            i, j = _ij_from_rank(c)
            v = row[c].strip()
            Rj[(i, j)] = int(v) if v else 0
        ranks_unsorted[jid] = Rj

        Mj = {}
        for c in mult_cols:
            a, b = _ij_from_mult(c)
            v = row[c].strip()
            Mj[(a, b)] = int(v) if v else 0
        mults_unsorted[jid] = Mj

    ids = sorted(ids_unsorted)
    ranks = {jid: ranks_unsorted[jid] for jid in ids}
    mults = {jid: mults_unsorted[jid] for jid in ids}
    return n, ids, ranks, mults



def read_edges_csv_strict(path: str | Path, ids: List[int], orientation: str):
    if orientation not in ("degenerates_to", "specializes_from"):
        raise ValueError("orientation must be 'degenerates_to' or 'specializes_from'.")

    idset = set(ids)
    adj = defaultdict(list)
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "src" not in reader.fieldnames or "dst" not in reader.fieldnames:
            raise ValueError("edges.csv must contain 'src' and 'dst' columns.")
        for row in reader:
            u = int(row["src"]); v = int(row["dst"])
            if u not in idset or v not in idset or u == v:
                continue
            if orientation == "degenerates_to":
                adj[u].append(v)
            else:  # specializes_from -> flip to degenerates_to
                adj[v].append(u)
    return adj


def read_parsed_csv_strict(path: str | Path, ids: List[int]):
    """
    parsed.csv columns (no 'id'):
      is_irreducible, is_equidimensional, irred_dims  (SPACE-SEPARATED ints)
    Matched to ids by row index.
    """
    rows = list(csv.DictReader(Path(path).open("r", newline="")))
    if not rows:
        raise ValueError(f"No rows in {path}")
    for must in ("is_irreducible", "is_equidimensional", "irred_dims"):
        if must not in rows[0]:
            raise ValueError(f"parsed.csv must contain column '{must}' (no 'id' column).")
    if len(rows) != len(ids):
        raise ValueError(f"Row count mismatch: parsed.csv has {len(rows)} rows, ranks.csv has {len(ids)} job ids.")

    def _b(x: str) -> bool:
        t = (x or "").strip().lower()
        if t in ("true","t","1","yes","y"): return True
        if t in ("false","f","0","no","n"): return False
        raise ValueError(f"Bad boolean '{x}'")

    out = {}
    for k, jid in enumerate(ids):
        r = rows[k]
        dims_str = (r["irred_dims"] or "").strip()
        dims_list = [int(tok) for tok in dims_str.split()] if dims_str else []
        out[jid] = {
            "is_irreducible": _b(r["is_irreducible"]),
            "is_equidimensional": _b(r["is_equidimensional"]),
            "irred_dims_list": dims_list,
        }
    return out

# -------------------------- Poset / reachability ----------------------------

def transitive_closure(ids: List[int], adj: Dict[int, List[int]]):
    """reach[u] = set of v with a path u -> ... -> v."""
    reach = {u: set() for u in ids}
    for u in ids:
        seen = set()
        dq = deque(adj.get(u, []))
        while dq:
            v = dq.popleft()
            if v in seen:
                continue
            seen.add(v)
            dq.extend(adj.get(v, []))
        reach[u] = seen
    return reach

# ---------------------------- Quiver helpers --------------------------------

def _adjacency_from_quiver(Q, n: int):
    """
    Build adjacency list adj[u] = list of v with a directed arrow u->v from Q.arrows.
    Q.arrows can be dict-like (0->{...},1->{...}) or list-like.
    Arrow entries look like {'source': u, 'target': v, 'label': ...}.
    """
    adj = [[] for _ in range(n)]
    arrows = getattr(Q, "arrows", [])
    it = arrows.values() if hasattr(arrows, "values") else arrows
    for a in it:
        u = int(a["source"]); v = int(a["target"])
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].append(v)
    return adj

def _reachability_matrix(adj: List[List[int]]):
    """Boolean reachability matrix R[i][j] = (exists path i -> ... -> j)."""
    n = len(adj)
    R = [[False]*n for _ in range(n)]
    for i in range(n):
        stack = list(adj[i])
        while stack:
            v = stack.pop()
            if not R[i][v]:
                R[i][v] = True
                stack.extend(adj[v])
    return R

def _want_M1_mults_from_orientation(adj: list[list[int]]):
    """
    Return multiplicities {(a,b)->count} for M^(1)=⊕_i P(i) ⊕_j I(j) on A_n with arbitrary orientation.
    Each indecomposable's support is the convex (undirected) interval of vertices reachable
    forward (for P) or backward (for I), normalized so a<=b.
    """
    n = len(adj)

    # Boolean reachability: R[i][j] True iff path i -> ... -> j exists
    def reachability_matrix(adj):
        n = len(adj)
        R = [[False]*n for _ in range(n)]
        for i in range(n):
            stack = list(adj[i])
            seen = [False]*n
            while stack:
                v = stack.pop()
                if not seen[v]:
                    seen[v] = True
                    R[i][v] = True
                    stack.extend(adj[v])
        return R

    R = reachability_matrix(adj)

    def proj_interval(i: int) -> tuple[int,int]:
        reach_js = [j for j in range(n) if R[i][j]]
        if not reach_js:
            return (i, i)
        lo = min(reach_js + [i])
        hi = max(reach_js + [i])
        return (lo, hi) if lo <= hi else (hi, lo)

    def inj_interval(j: int) -> tuple[int,int]:
        reach_is = [i for i in range(n) if R[i][j]]
        if not reach_is:
            return (j, j)
        lo = min(reach_is + [j])
        hi = max(reach_is + [j])
        return (lo, hi) if lo <= hi else (hi, lo)

    want: dict[tuple[int,int], int] = {}
    # Sum of all projectives
    for i in range(n):
        a,b = proj_interval(i)
        want[(a,b)] = want.get((a,b), 0) + 1
    # Plus all injectives
    for j in range(n):
        a,b = inj_interval(j)
        want[(a,b)] = want.get((a,b), 0) + 1
    return want

# --------------------------- Identify M^(0), M^(1) --------------------------

def identify_M0_M1_orientation_aware(
    jobs: List[tuple],
    mults_by_id: Dict[int, Dict[Tuple[int,int], int]],
    reach: Dict[int, Set[int]],
):
    """
    Returns (id_M0, id_M1):
      - id_M0: node whose reach set contains all other ids (degenerates_to orientation).
      - id_M1: id whose interval multiplicities match those of P⊕I computed from the quiver orientation.
    """
    if not jobs:
        raise ValueError("jobs list is required (to read the quiver orientation).")

    ids = list(range(len(jobs)))
    N = len(ids)

    # M^(0) from poset (degenerates to everyone)
    M0_cands = [u for u in ids if len(reach.get(u, set())) == N - 1]
    id_M0 = min(M0_cands) if M0_cands else None

    # Orientation from the quiver of the first job
    # Infer n from bags/target_dim robustly
    _, first_bag, first_dim = jobs[0]
    n = 0
    for _, bag, dimP in jobs:
        for a,b in bag:
            n = max(n, a+1, b+1)
        if dimP:
            n = max(n, max(dimP.keys()) + 1)

    Q = jobs[0][0]
    adj = _adjacency_from_quiver(Q, n)
    want = _want_M1_mults_from_orientation(adj)

    id_M1 = None
    for jid in ids:
        M = mults_by_id[jid]
        # compare sparse dicts exactly (drop zeros on both sides)
        M_nz = {k:v for k,v in M.items() if v}
        if M_nz == want:
            id_M1 = jid
            break

    return id_M0, id_M1

# --------------------------- Selection & extrema ----------------------------

def select_generic_jobs_strict(geom: Dict[int, dict], ids: List[int], generic_dim: int):
    """
    Generic-dimension iff max(irred_dims_list) == generic_dim.
    Returns (irreducible_generic, generic_all).
    """
    irreducible_generic: List[int] = []
    generic_all: List[int] = []
    for jid in ids:
        dims = geom[jid]["irred_dims_list"]
        if not dims:
            continue
        if max(dims) == generic_dim:
            generic_all.append(jid)
            if geom[jid]["is_irreducible"]:
                irreducible_generic.append(jid)
    return irreducible_generic, generic_all

def maximal_in_subset(ids_subset: List[int], reach: Dict[int, Set[int]]):
    """
    MAXIMAL elements in the induced subposet (using reachability).
    v is maximal if NO w != v in subset has v <= w  (i.e., w in reach[v]).
    """
    maxs: List[int] = []
    S = list(ids_subset)
    for v in S:
        has_succ = any((w != v) and (w in reach[v]) for w in S)
        if not has_succ:
            maxs.append(v)
    return maxs

# ------------------------------ Main writer ---------------------------------

def check_conjectures_write(
    rank_dir: str | Path = "rank_poset",
    parsed_csv: str | Path = "parsed.csv",
    out_dir: str | Path = "reports",
    orientation: str = "degenerates_to",
    generic_dim: int | None = None,
    *,
    jobs: List[tuple],
):
    """
    STRICT pipeline. Writes:
      out_dir/conj1_check.csv
      out_dir/conj2_minimals.csv   (reducible **maxima**)
      out_dir/conj2_coverage.csv

    Required:
      - jobs: list of (Q, bag, target_dim) for THIS batch (used to read quiver orientation).
    """
    if jobs is None:
        raise ValueError("Argument 'jobs' is required (to read the quiver).")

    rank_dir = Path(rank_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n, ids, _ranks, mults = read_ranks_csv_strict(rank_dir / "ranks.csv")
    adj = read_edges_csv_strict(rank_dir / "edges.csv", ids, orientation=orientation)
    reach = transitive_closure(ids, adj)
    geom = read_parsed_csv_strict(parsed_csv, ids)

    if generic_dim is None:
        generic_dim = n * (n + 1) // 2

    # sparse multiplicities (drop zeros) for exact equality checks
    mults_by_id = {}
    for jid in ids:
        if jid not in mults:
            raise KeyError(f"Job id {jid} missing from mults; ranks.csv and jobs list misaligned.")
        mults_by_id[jid] = {k: v for k, v in mults[jid].items() if v}


    # After computing reach = transitive_closure(ids, adj)

    #adj = read_edges_csv_strict(rank_dir / "edges.csv", ids, orientation=orientation)
    #adj_rev = read_edges_csv_strict(rank_dir / "edges.csv", ids, orientation="specializes_from")
    #reach_rev = transitive_closure(ids, adj_rev)



    # Then identify
    id_M0, id_M1 = identify_M0_M1_orientation_aware(jobs, mults_by_id, reach)
    print("DEBUG: After identify_M0_M1, id_M0 =", id_M0, "id_M1 =", id_M1)


    # Identify M^(0), M^(1) correctly (orientation-aware)
    # id_M0, id_M1 = identify_M0_M1_orientation_aware(jobs, mults_by_id, reach)

    # Orientation sanity: M^(0) must reach all nodes (in degenerates_to orientation)
    if id_M0 is not None and len(reach[id_M0]) != len(ids) - 1:
        raise RuntimeError(
            "Hasse edges orientation mismatch: open-orbit M^(0) does NOT reach all nodes. "
            "If your edges encode the reverse relation, call with orientation='specializes_from'."
        )

    irreducible_generic, generic_all = select_generic_jobs_strict(geom, ids, generic_dim)

    # -------- Conjecture 1 --------
    conj1_path = out_dir / "conj1_check.csv"
    with conj1_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","degenerates_to_M1","is_irreducible","is_equidimensional","irred_dims"])
        w.writeheader()
        for jid in irreducible_generic:
            reaches_M1 = (id_M1 is not None) and (jid == id_M1 or id_M1 in reach[jid])
            dims_str = " ".join(str(x) for x in geom[jid]["irred_dims_list"])
            w.writerow({
                "id": jid,
                "degenerates_to_M1": "yes" if reaches_M1 else "no",
                "is_irreducible": "true" if geom[jid]["is_irreducible"] else "false",
                "is_equidimensional": "true" if geom[jid]["is_equidimensional"] else "false",
                "irred_dims": dims_str,
            })

    # -------- Conjecture 2 (reducible **maxima** in the generic-dim subposet) --------
    maxs = maximal_in_subset(generic_all, reach)
    m2_candidates = [v for v in maxs if not geom[v]["is_irreducible"]]

    conj2_min_path = out_dir / "conj2_minimals.csv"  # filename kept for downstream compatibility
    with conj2_min_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","is_irreducible","is_equidimensional","irred_dims"])
        for v in m2_candidates:
            dims_str = " ".join(str(x) for x in geom[v]["irred_dims_list"])
            w.writerow([
                v,
                "true" if geom[v]["is_irreducible"] else "false",
                "true" if geom[v]["is_equidimensional"] else "false",
                dims_str,
            ])

    conj2_cov_path = out_dir / "conj2_coverage.csv"
    with conj2_cov_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","reaches_any_candidate","candidates"])
        w.writeheader()
        for jid in generic_all:
            hits = [c for c in m2_candidates if (c == jid or c in reach[jid])]
            w.writerow({
                "id": jid,
                "reaches_any_candidate": "yes" if hits else "no",
                "candidates": "[" + ",".join(map(str, hits)) + "]",
            })

    print(f"[A_{n}] generic_dim = {generic_dim}")
    print(f"M^(0): {id_M0 if id_M0 is not None else 'not found'};  M^(1): {id_M1 if id_M1 is not None else 'not found'}")
    print(f"Irreducible+generic jobs: {len(irreducible_generic)}   (Conj1 -> {conj1_path})")
    print(f"Generic-dim jobs: {len(generic_all)}")
    print(f"M^(2) candidates (reducible **maxima**): {len(m2_candidates)}   (Conj2 -> {conj2_min_path}, {conj2_cov_path})")

# -------- Thin alias with REQUIRED jobs argument (no fallbacks) --------

def check_conjectures(
    rank_dir: str | Path = "rank_poset",
    parsed_csv: str | Path = "parsed.csv",
    out_dir: str | Path = "reports",
    orientation: str = "degenerates_to",
    generic_dim: int | None = None,
    *,
    jobs: List[tuple],
):
    """
    Convenience wrapper; same behavior as check_conjectures_write.
    'jobs' is REQUIRED (to read the quiver).
    """
    return check_conjectures_write(
        rank_dir=rank_dir,
        parsed_csv=parsed_csv,
        out_dir=out_dir,
        orientation=orientation,
        generic_dim=generic_dim,
        jobs=jobs,
    )
