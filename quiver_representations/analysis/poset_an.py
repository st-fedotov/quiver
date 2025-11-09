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

def _hasse_edges(ids, leq_adj):
    """
    Transitive reduction of a finite poset given by adjacency (u->v if u <= v and u!=v).
    Return minimal edges (Hasse diagram).
    """
    # Compute reachability via Floyd–Warshall on boolean adjacency
    idx = {u: i for i, u in enumerate(ids)}
    n = len(ids)
    R = [[False]*n for _ in range(n)]
    for u in ids:
        iu = idx[u]
        for v in leq_adj[u]:
            if u != v:
                R[iu][idx[v]] = True
    for k in range(n):
        for i in range(n):
            if R[i][k]:
                row_i = R[i]
                row_k = R[k]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True

    # Keep u->v if there is no w with u->w and w->v (i.e., not implied transitively)
    edges = []
    for u in ids:
        iu = idx[u]
        for v in leq_adj[u]:
            if u == v:
                continue
            iv = idx[v]
            covered = False
            for w in ids:
                if w == u or w == v:
                    continue
                iw = idx[w]
                if R[iu][iw] and R[iw][iv]:
                    covered = True
                    break
            if not covered:
                edges.append((u, v))
    return edges

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


def write_rank_poset_from_jobs(jobs, out_dir):
    """
    Input:
      jobs: list of (Q, interval_bag, target_dim) as in your RAD batches.
            Q is ignored here; only interval_bag is used.
    Output files in out_dir:
      - ranks.csv : id, ranks r_i_j for all i<j, then multiplicities for all (a,b)
      - edges.csv : hasse edges 'src,dst' for the degeneration poset (rank-array order)
    Returns the output directory (Path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = _infer_n_from_jobs(jobs)
    pairs_ij = _all_pairs_ij(n)
    interval_keys = _all_interval_keys(n)

    # Compute ranks and multiplicities
    ranks_per_id = {}
    mults_per_id = {}
    for jid, (_, bag, _dimP) in enumerate(jobs):
        mults = _interval_multiplicities(bag)
        ranks = _rank_array_from_bag(bag, n)
        ranks_per_id[jid] = ranks
        mults_per_id[jid] = mults

    # Write ranks.csv
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

    # Build poset by rank comparison
    ids = list(range(len(jobs)))
    leq_adj = defaultdict(list)
    for u in ids:
        Ru = ranks_per_id[u]
        for v in ids:
            if u == v:
                continue
            Rv = ranks_per_id[v]
            if _leq_by_ranks(Ru, Rv):
                leq_adj[u].append(v)

    # Transitive reduction -> Hasse edges
    edges = _hasse_edges(ids, leq_adj)
    with open(out_dir / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        w.writerows(edges)

    return out_dir
