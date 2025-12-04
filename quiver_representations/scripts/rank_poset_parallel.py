# rank_poset_parallel.py
# Python 3.12+
# Stdlib + tqdm only (install: sudo apt install -y python3-tqdm  OR  python3 -m pip install --user tqdm)
#
# CHANGE from last version: Hasse minimality is BACK TO SEQUENTIAL (with tqdm).
# We keep parallel adjacency-row computation only.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -------- helpers (unchanged logic) ------------------------------------------

def _infer_n_from_jobs(jobs):
    max_idx = -1
    for _, bag, target_dim in jobs:
        for (a, b) in bag:
            max_idx = max(max_idx, a, b)
        if max_idx < 0 and target_dim:
            max_idx = max(max_idx, max(target_dim.keys()))
    if max_idx < 0:
        raise ValueError("Cannot infer number of vertices from empty jobs.")
    return max_idx + 1

def _interval_multiplicities(bag):
    c = Counter()
    for ab in bag:
        c[tuple(ab)] += 1
    return dict(c)

def _rank_array_from_bag(bag, n):
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
    return [(a,b) for a in range(n) for b in range(a, n)]

def _all_pairs_ij(n):
    return [(i,j) for i in range(n-1) for j in range(i+1, n)]

# -------- Hasse edges (sequential Floyd–Warshall + minimality) ---------------

def _hasse_edges(ids, leq_adj):
    idx = {u: i for i, u in enumerate(ids)}
    n = len(ids)
    R = [[False]*n for _ in range(n)]

    # seed reachability
    for u in tqdm(ids, desc="seed reachability"):
        iu = idx[u]
        for v in leq_adj[u]:
            if u != v:
                R[iu][idx[v]] = True

    # FW transitive closure (outer k loop is inherently sequential)
    for k in tqdm(range(n), desc="fw k"):
        for i in range(n):
            if R[i][k]:
                row_i = R[i]
                row_k = R[k]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True

    # candidate edges and minimality test (sequential but fast in tight loops)
    candidates = []
    for u in tqdm(ids, desc="candidates"):
        for v in leq_adj[u]:
            if u != v:
                candidates.append((u, v))

    edges = []
    for (u, v) in tqdm(candidates, desc="hasse edges"):
        iu, iv = idx[u], idx[v]
        covered = False
        for w in ids:
            iw = idx[w]
            if R[iu][iw] and R[iw][iv]:
                covered = True
                break
        if not covered:
            edges.append((u, v))
    return edges

# -------- orientation + Hom-order machinery (unchanged logic) ----------------

def build_dir_edges(Q):
    arrows = getattr(Q, "arrows", [])
    it = arrows.values() if hasattr(arrows, "values") else arrows
    max_idx = -1
    dir_edges = {}
    for a in it:
        u = int(a["source"]); v = int(a["target"])
        max_idx = max(max_idx, u, v)
        if abs(u - v) != 1:
            continue
        k = min(u, v)
        val = +1 if (u == k and v == k + 1) else -1
        if k in dir_edges and dir_edges[k] != val:
            raise ValueError(f"Conflicting arrows between {k} and {k+1}.")
        dir_edges[k] = val
    if max_idx < 0:
        raise ValueError("No arrows found.")
    n = max_idx + 1
    out = [None] * (n - 1)
    for k in range(n - 1):
        if k not in dir_edges:
            raise ValueError(f"Missing arrow between {k} and {k+1}.")
        out[k] = dir_edges[k]
    return out

def hom_interval(dir_edges, a, b, c, d):
    n = len(dir_edges) + 1
    L = max(a, c); R = min(b, d)
    if L > R: return 0
    k = L - 1
    if k >= 0:
        dom_only = (a <= k <= b) and not (c <= k <= d)
        cod_only = (c <= k <= d) and not (a <= k <= b)
        if dom_only and dir_edges[k] != +1: return 0
        if cod_only and dir_edges[k] != -1: return 0
    k = R
    if k <= n - 2:
        dom_only = (a <= k+1 <= b) and not (c <= k+1 <= d)
        cod_only = (c <= k+1 <= d) and not (a <= k+1 <= b)
        if dom_only and dir_edges[k] != -1: return 0
        if cod_only and dir_edges[k] != +1: return 0
    return 1

def hom_interval_to_bag(dir_edges, i, j, bag):
    return sum(hom_interval(dir_edges, i, j, p, q) for (p, q) in bag)

def _degenerates(dir_edges, bagM, bagN):
    n = len(dir_edges) + 1
    for i in range(n):
        for j in range(i, n):
            if hom_interval_to_bag(dir_edges, i, j, bagM) > \
               hom_interval_to_bag(dir_edges, i, j, bagN):
                return False
    return True

# -------- parallel row worker (unchanged logic) ------------------------------

def _adjacency_row(u, dir_edges, bags):
    Mu = bags[u]
    ids = range(len(bags))
    row = []
    for v in ids:
        if u == v:
            continue
        if _degenerates(dir_edges, Mu, bags[v]):
            row.append(v)
    return u, row

# -------- main writer (parallel adjacency, sequential Hasse) -----------------

def write_rank_poset_from_jobs(jobs, out_dir, workers=120):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not jobs:
        raise ValueError("No jobs provided.")

    n = _infer_n_from_jobs(jobs)
    pairs_ij = _all_pairs_ij(n)
    interval_keys = _all_interval_keys(n)

    ranks_per_id = {}
    mults_per_id = {}
    for jid, (_, bag, _dimP) in enumerate(jobs):
        mults = _interval_multiplicities(bag)
        ranks = _rank_array_from_bag(bag, n)
        ranks_per_id[jid] = ranks
        mults_per_id[jid] = mults

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

    Q0 = jobs[0][0]
    dir_edges = build_dir_edges(Q0)

    ids = list(range(len(jobs)))
    bags = [bag for (_, bag, _dimP) in jobs]

    # adjacency rows in parallel (safe)
    leq_adj = defaultdict(list)
    max_workers = int(workers)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_adjacency_row, u, dir_edges, bags): u for u in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="adjacency rows"):
            u, row = fut.result()
            leq_adj[u] = row

    # hasse edges sequential (safe) with progress bars
    edges = _hasse_edges(ids, leq_adj)

    with open(out_dir / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        w.writerows(edges)

    return out_dir

# ------------------------ JSON-driven CLI (same as before) -------------------

def _load_jobs_from_json(json_path: Path):
    data = json.loads(Path(json_path).read_text())
    dir_edges = data["dir_edges"]

    class _Q: pass
    Q = _Q()
    arrows = []
    for k, val in enumerate(dir_edges):
        if val == +1:
            arrows.append({"source": k, "target": k+1})
        elif val == -1:
            arrows.append({"source": k+1, "target": k})
        else:
            raise ValueError("dir_edges must contain only +1 or -1.")
    Q.arrows = arrows

    jobs = []
    for item in data["jobs"]:
        bag = [(int(a), int(b)) for (a, b) in item["bag"]]
        td_raw = item.get("target_dim")
        target_dim = {int(k): int(v) for k, v in (td_raw or {}).items()}
        jobs.append((Q, bag, target_dim))
    return jobs

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parallel rank/poset writer (CLI-only).")
    ap.add_argument("--input", required=True, help="Path to poset_input.json (same folder is fine).")
    ap.add_argument("--out", required=True, help="Output directory (relative or absolute).")
    ap.add_argument("--workers", type=int, default=120, help="Number of process workers (default 120).")
    args = ap.parse_args()

    jobs = _load_jobs_from_json(Path(args.input))
    outp = write_rank_poset_from_jobs(jobs, Path(args.out), workers=args.workers)
    print(str(outp))

if __name__ == "__main__":
    main()
