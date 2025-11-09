"""
Rank poset construction and analysis for D_n quivers.

Provides functionality specific to D_n quiver types:
- Rank poset via Hom computations between indecomposables
- Local minima (sink) detection in degeneration posets

"""

from typing import List, Dict, Union
from pathlib import Path
import numpy as np
import csv
from tqdm import tqdm

from ..interval_modules.type_d import DnKind, DnIndec


def write_rank_poset_from_jobs_Dn(jobs, out_dir, p: int = 107):
    """
    Build the D_n degeneration poset by:
      1) precomputing H[α,β] = dim Hom(Iα, Iβ) for all indecomposables Iα, Iβ,
      2) for each bag B, computing v(B) = sum_{Y in B} H[:, idx(Y)],
      3) using coordinatewise comparison v(M_u) <= v(M_v) to form the poset,
      4) writing Hasse edges to out_dir/edges.csv.

    Input:
      jobs: list of (Q, dn_bag, target_dim)   # target_dim unused here
      out_dir: directory path
      p: prime for GF(p), default 107
    """
    if not jobs:
        raise ValueError("No jobs provided.")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    Q0 = jobs[0][0]
    n = len(Q0.get_vertices())
    if n < 4:
        raise ValueError("D_n requires n >= 4.")
    Fp = FiniteField(p)

    # ---------- enumerate indecomposables for this n (deterministic order) ----------
    br, up_leaf, dn_leaf = n - 3, n - 2, n - 1
    indecs: list[DnIndec] = []
    for i in range(0, br + 1):
        for j in range(i + 1, br + 2):             # SEG
            indecs.append(DnIndec(DnKind.SEG, i, j))
    for i in range(0, br + 1):                 # UP/DOWN/BOTH (spine starts)
        indecs.append(DnIndec(DnKind.UP, i))
        indecs.append(DnIndec(DnKind.DOWN, i))
        indecs.append(DnIndec(DnKind.BOTH, i))
    indecs.append(DnIndec(DnKind.UP, up_leaf))   # pure leaf simples
    indecs.append(DnIndec(DnKind.DOWN, dn_leaf))
    for i in range(0, br):                 # STAR(i,j):
        for j in range(i + 1, br + 1):
            indecs.append(DnIndec(DnKind.STAR, i, j))
    K = len(indecs)
    idx = { (int(X.kind), X.i, X.j): k for k, X in enumerate(indecs) }

    # ---------- build indec modules once ----------
    indec_modules = [form_Dn_module_from_bag_explicit(Q0, Fp, [X]) for X in indecs]

    # ---------- precompute Hom-matrix H[α,β] = dim Hom(Iα, Iβ) ----------
    H = np.zeros((K, K), dtype=int)
    for a in range(K):
        Ma = indec_modules[a]
        for b in range(K):
            Mb = indec_modules[b]
            H[a, b] = len(find_hom_basis(Ma, Mb))

    # ---------- for each bag, compute its Hom-profile v = sum columns of H with multiplicity ----------
    ids = list(range(len(jobs)))
    homvecs: list[np.ndarray] = []
    for (_Q, bag, _dim) in tqdm(jobs):
        if not bag:
            homvecs.append(np.zeros((K,), dtype=int))
            continue
        cols = []
        for Y in bag:  # bag is a list; duplicates => multiplicity via repetition
            key = (int(Y.kind), Y.i, Y.j)
            k = idx.get(key)
            if k is None:
                raise ValueError(f"Bag contains unknown indecomposable: {Y}")
            cols.append(k)
        V = H[:, cols].sum(axis=1) if cols else np.zeros((K,), dtype=int)
        homvecs.append(V)

    # ---------- build ≤ via coordinatewise comparison of Hom-profiles ----------
    leq_adj = defaultdict(list)
    for u in tqdm(ids):
        Vu = homvecs[u]
        for v in ids:
            if u == v:
                continue
            if np.all(Vu <= homvecs[v]):
                leq_adj[u].append(v)

    # ---------- Hasse reduction and write edges ----------
    edges = _hasse_edges(ids, leq_adj)
    with (out_path / "edges.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        w.writerows(edges)

    return out_path


def report_local_minima_dn(
    generic_dim: int,
    rank_dir: str | Path = "rank_poset",   # expects rank_poset/edges.csv (Hasse, degenerates_to)
    parsed_csv: str | Path = "parsed.csv", # columns: is_irreducible, is_equidimensional, irred_dims
    out_dir: str | Path = "reports_dn",
):
    """
    D_n: find local minima (sinks) in two induced subposets, using ONLY:
      - rank_poset/edges.csv  (Hasse edges; src -> dst means 'src degenerates to dst')
      - parsed.csv            (must have 'irred_dims' as space-separated ints)

    Subsets:
      S_single  = { id | irred_dims_list == [generic_dim] }
      S_k_multi = { id | irred_dims_list = [generic_dim]*k for some k >= 1 }

    Writes:
      reports_dn/dn_minima_single.csv   (id, irred_dims)
      reports_dn/dn_minima_k_multi.csv  (id, k, irred_dims)
    """
    rank_dir = Path(rank_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- read parsed.csv -> ids = 0..N-1, irred_dims_by_id ---
    rows = list(csv.DictReader(Path(parsed_csv).open("r", newline="")))
    if not rows:
        raise ValueError(f"No rows in {parsed_csv}")
    ids = list(range(len(rows)))
    irred_dims_by_id: Dict[int, List[int]] = {}
    for k, r in enumerate(rows):
        dims_str = (r.get("irred_dims","") or "").strip()
        irred_dims_by_id[k] = [int(x) for x in dims_str.split()] if dims_str else []

    # --- read edges.csv -> adjacency in degenerates_to orientation ---
    edges_path = rank_dir / "edges.csv"
    E = []
    with edges_path.open("r", newline="") as f:
        rd = csv.DictReader(f)
        if "src" not in rd.fieldnames or "dst" not in rd.fieldnames:
            raise ValueError("edges.csv must contain 'src' and 'dst' columns.")
        for row in rd:
            u = int(row["src"]); v = int(row["dst"])
            if u != v:
                E.append((u, v))

    idset = set(ids)
    adj: Dict[int, List[int]] = {u: [] for u in ids}
    for u, v in E:
        if u in idset and v in idset:
            adj[u].append(v)

    # --- subsets ---
    def all_generic(dims: List[int]) -> bool:
        return len(dims) >= 1 and all(d == generic_dim for d in dims)

    S_single  = [u for u in ids if irred_dims_by_id[u] == [generic_dim]]
    S_k_multi = [u for u in ids if all_generic(irred_dims_by_id[u])]

    # --- sinks in induced subgraph ---
    def sinks(subset: List[int]) -> List[int]:
        S = set(subset)
        out = []
        for u in subset:
            if not any((v in S and v != u) for v in adj.get(u, [])):
                out.append(u)
        return out

    mins_single  = sinks(S_single)
    mins_k_multi = sinks(S_k_multi)

    # --- write CSVs ---
    single_csv = out_dir / "dn_minima_single.csv"
    with single_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","irred_dims"])
        for u in mins_single:
            w.writerow([u, " ".join(map(str, irred_dims_by_id[u]))])

    kmulti_csv = out_dir / "dn_minima_k_multi.csv"
    with kmulti_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","k","irred_dims"])
        for u in mins_k_multi:
            dims = irred_dims_by_id[u]
            w.writerow([u, len(dims), " ".join(map(str, dims))])

    print(f"[D_n] generic_dim={generic_dim} | S_single={len(S_single)} -> minima={len(mins_single)} | "
          f"S_k_multi={len(S_k_multi)} -> minima={len(mins_k_multi)}")
    print(f"Wrote {single_csv} and {kmulti_csv}")

    return {"single_csv": single_csv, "k_multi_csv": kmulti_csv}
