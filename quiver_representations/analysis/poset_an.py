"""
Rank poset construction and degeneration analysis for A_n quivers.

Provides functionality to:
- Compute rank arrays from interval bags
- Build degeneration posets via Hom computations
- Generate Hasse diagrams and export to CSV

"""

from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import csv
from collections import Counter, defaultdict
from tqdm import tqdm

from .hom import hom_interval_to_bag
from .poset_utils import _hasse_edges



def _infer_n_from_jobs(jobs):
    """
    Infer n (= number of vertices) from interval endpoints present in the bags,
    or from target_dim keys if bags are empty (defensive).
    """
    max_idx = -1
    for _, bag, target_dim in jobs:
        for (a, b) in bag:
            max_idx = max(max_idx, a, b)
        if max_idx < 0 and target_dim:
            max_idx = max(max_idx, max(target_dim.keys()))
    if max_idx < 0:
        raise ValueError("Cannot infer number of vertices from empty jobs.")
    return max_idx + 1  # vertices are 0..n-1

def _interval_multiplicities(bag):
    """
    Return a dict {(a,b) -> multiplicity} for an interval bag (list of (a,b) pairs).
    """
    c = Counter()
    for ab in bag:
        c[tuple(ab)] += 1
    return dict(c)

def _rank_array_from_bag(bag, n):
    """
    Equioriented A_n 'rank' surrogate (kept only to preserve the ranks.csv schema).
    r_{i,j} = # of intervals [a,b] with a <= i and b >= j, for 0 <= i < j <= n-1.
    """
    mult = _interval_multiplicities(bag)
    ranks = {}
    for i in range(n-1):
        for j in range(i+1, n):
            s = 0
            for (a,b), m in mult.items():
                if a <= i and b >= j:
                    s += m
            ranks[(i,j)] = s
    return ranks

def _all_interval_keys(n):
    """All interval types (a,b) with 0<=a<=b<=n-1 in lex order."""
    return [(a,b) for a in range(n) for b in range(a, n)]

def _all_pairs_ij(n):
    """All pairs (i,j) with 0<=i<j<=n-1 in lex order."""
    return [(i,j) for i in range(n-1) for j in range(i+1, n)]

def build_dir_edges(Q):
    """
    For a type-A quiver Q with vertices 0..n-1 and arrows as dicts
    { "source": int, "target": int }, return dir_edges[0..n-2] where:
      dir_edges[k] = +1 if k -> k+1,  -1 if (k+1) -> k
    """
    arrows = getattr(Q, "arrows", [])
    it = arrows.values() if hasattr(arrows, "values") else arrows
    max_idx = -1
    dir_edges = {}

    for a in it:
        u = int(a["source"]); v = int(a["target"])
        max_idx = max(max_idx, u, v)
        if abs(u - v) != 1:
            continue  # ignore non-adjacent edges
        k = min(u, v)
        val = +1 if (u == k and v == k + 1) else -1
        if k in dir_edges and dir_edges[k] != val:
            raise ValueError(f"Conflicting arrows between {k} and {k+1}.")
        dir_edges[k] = val

    if max_idx < 0:
        raise ValueError("No arrows found.")
    n = max_idx + 1

    # ensure every A_n edge is present
    out = [None] * (n - 1)
    for k in range(n - 1):
        if k not in dir_edges:
            raise ValueError(f"Missing arrow between {k} and {k+1}.")
        out[k] = dir_edges[k]
    return out

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

def _degenerates(dir_edges, bagM, bagN):
    """
    M <= N  iff  Hom([i,j], M) <= Hom([i,j], N) for all intervals [i,j].
    (Bongartz’s Hom-order = degeneration in Dynkin.)"""
    n = len(dir_edges) + 1
    for i in range(n):
        for j in range(i, n):
            if hom_interval_to_bag(dir_edges, i, j, bagM) > \
               hom_interval_to_bag(dir_edges, i, j, bagN):
                return False
    return True

# -------- main writer

def write_rank_poset_from_jobs(jobs, out_dir):
    """
    Input:
      jobs: list of (Q, interval_bag, target_dim) as in your RAD batches.
    Output files in out_dir:
      - ranks.csv : id, ranks r_i_j for all i<j, then multiplicities for all (a,b)
      - edges.csv : hasse edges 'src,dst' for the degeneration poset
                    (NOW computed via Hom-order; filename/columns unchanged)
    Returns the output directory (Path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jobs:
        raise ValueError("No jobs provided.")

    n = _infer_n_from_jobs(jobs)
    pairs_ij = _all_pairs_ij(n)
    interval_keys = _all_interval_keys(n)

    # Compute ranks and multiplicities (kept for compatibility with your parsers)
    ranks_per_id = {}
    mults_per_id = {}
    for jid, (_, bag, _dimP) in enumerate(jobs):
        mults = _interval_multiplicities(bag)
        ranks = _rank_array_from_bag(bag, n)
        ranks_per_id[jid] = ranks
        mults_per_id[jid] = mults

    # Write ranks.csv (UNCHANGED schema)
    rank_cols = [f"r_{i}_{j}" for (i,j) in pairs_ij]
    mult_cols = [f"x_{a}_{b}" for (a,b) in interval_keys]
    with open(out_dir / "ranks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + rank_cols + mult_cols)
        for jid in range(len(jobs)):
            row = [jid]
            row.extend(ranks_per_id[jid].get((i,j), 0) for (i,j) in pairs_ij)
            row.extend(mults_per_id[jid].get((a,b), 0) for (a,b) in interval_keys)
            w.writerow(row)

    # Build poset by HOM-order using quiver orientation from the FIRST job
    Q0 = jobs[0][0]
    dir_edges = build_dir_edges(Q0)

    ids = list(range(len(jobs)))
    leq_adj = defaultdict(list)
    bags = [bag for (_, bag, _dimP) in jobs]
    for u in tqdm(ids):
        Mu = bags[u]
        for v in ids:
            if u == v:
                continue
            if _degenerates(dir_edges, Mu, bags[v]):
                leq_adj[u].append(v)

    # Transitive reduction -> Hasse edges (same file name as before)
    edges = _hasse_edges(ids, leq_adj)
    with open(out_dir / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        w.writerows(edges)

    return out_dir
