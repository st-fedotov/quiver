"""
Enumeration utilities for interval modules and batch job generation.

This module provides functions for:
- Computing dimension vectors for P+I modules
- Enumerating interval bags for A_n quivers
- Enumerating indecomposable bags for D_n quivers
- Creating batch jobs for radicality testing
"""

from typing import Dict, List, Tuple, Iterator
from itertools import product
from pathlib import Path
from ..quiver import Quiver
from .type_d import DnIndec, DnKind

# Type aliases
Bag = List[Tuple[int, int]]  # For A_n: list of interval (i,j) pairs


# ============================================================================
# A_n Enumeration Functions
# ============================================================================

def enumerate_interval_bags_An(n: int, c: int | None = None):
    """
    Return a list of 'bags', each bag = list of (i,j) pairs (0-based, i<=j),
    representing the interval multiset for a module with constant coverage c on A_n.
    Default c = n+1.
    """
    if c is None:
        c = n + 1
    if n <= 0:
        return []

    bags = []

    # Iterate over end-counts E_0..E_{n-2} in [0..c]; force E_{n-1} = c.
    # Then S_0 = c; S_k = E_{k-1} for k>=1.
    for E_prefix in product(range(c + 1), repeat=max(0, n - 1)):
        E = list(E_prefix) + [c]
        S = [0] * n
        S[0] = c
        for k in range(1, n):
            S[k] = E[k - 1]

        # Quick feasibility check: sum S == sum E == total intervals (which is c + sum(E_0..E_{n-2}))
        if sum(S) != sum(E):
            continue  # should not happen with the chosen S/E, but keep guard

        # Enumerate upper-triangular matrix x[i][j] with row sums S and col sums E
        # Rows i go from 0..n-1, columns j from i..n-1
        col_rem = E[:]  # remaining capacity per column j
        row_acc = [[0] * n for _ in range(n)]  # will only use j>=i

        def fill_row(i: int):
            if i == n:
                # Completed a feasible matrix; convert to bag
                bag = []
                for ii in range(n):
                    for jj in range(ii, n):
                        cnt = row_acc[ii][jj]
                        if cnt > 0:
                            bag.extend([(ii, jj)] * cnt)
                bags.append(bag)
                return

            need = S[i]
            # Distribute 'need' over columns j=i..n-1 subject to col_rem[j]
            # Recursive helper: assign x[i][j] values summing to 'need'
            def assign_in_row(j: int, left: int):
                if j == n:
                    if left == 0:
                        fill_row(i + 1)
                    return
                if j < i:
                    assign_in_row(j + 1, left)
                    return
                # max we can place at (i,j) given remaining col capacity
                max_here = min(left, col_rem[j])
                # Iterate nonneg allocations; small-to-large keeps deterministic order
                for val in range(max_here + 1):
                    row_acc[i][j] += val
                    col_rem[j]   -= val
                    assign_in_row(j + 1, left - val)
                    col_rem[j]   += val
                    row_acc[i][j] -= val

            assign_in_row(i, need)

        fill_row(0)

    return bags


def enumerate_interval_bags_from_coverage_iter(coverage: Dict[int, int]) -> Iterator[Bag]:
    """
    Enumerate all bags of intervals [i,j] (0-based, i<=j) whose vertex-cover counts equal 'coverage'.
    This allows extra ends at k compensated by extra starts at k+1 (the missing degrees of freedom).
    Deterministic order: older-start buckets are closed earlier in lexicographic order of choices.
    """
    if not coverage:
        return
    n = max(coverage) + 1
    c = [int(coverage.get(k, 0)) for k in range(n)]
    if any(v < 0 for v in c):
        raise ValueError("All coverages must be nonnegative integers.")

    starts: List[int] = []   # start index per bucket
    counts: List[int] = []   # multiplicities for that start
    bag: Bag = []

    def choose_close(need: int, caps: List[int], i: int = 0, take: List[int] = None):
        """All vectors take[0..m-1] with 0<=take[i]<=caps[i], sum=need (older-first)."""
        if take is None: take = []
        if i == len(caps):
            if need == 0:
                yield take.copy()
            return
        cap = caps[i]
        # older-first determinism: try taking more from older buckets first
        for t in range(min(cap, need), -1, -1):
            take.append(t)
            yield from choose_close(need - t, caps, i + 1, take)
            take.pop()

    def dfs(k: int, prev_e: int):
        if k == n:
            # All intervals must be closed when c_n = 0; the bounds enforce e_{n-1}=c_{n-1}.
            yield bag.copy()
            return

        # starts at k determined by previous choice of e_{k-1}:
        s_k = c[k] if k == 0 else prev_e + (c[k] - c[k-1])
        if s_k < 0:
            return
        if s_k > 0:
            starts.append(k)
            counts.append(s_k)

        # choose how many to end at k
        lb = max(0, c[k] - (c[k+1] if k + 1 < n else 0))  # must end at least this many
        ub = c[k]                                         # cannot end more than open
        if lb > ub:
            # infeasible coverage vector
            if s_k > 0:
                starts.pop(); counts.pop()
            return

        for e_k in range(lb, ub + 1):
            if e_k == 0:
                yield from dfs(k + 1, e_k)
            else:
                caps = counts[:]  # how many open in each start-bucket
                for take in choose_close(e_k, caps):
                    # apply closures
                    changed = []
                    for idx, t in enumerate(take):
                        if t:
                            counts[idx] -= t
                            bag.extend([(starts[idx], k)] * t)
                            changed.append((idx, t))
                    yield from dfs(k + 1, e_k)
                    # undo
                    for idx, t in reversed(changed):
                        counts[idx] += t
                        del bag[-t:]

        if s_k > 0:
            starts.pop()
            counts.pop()

    yield from dfs(0, 0)

def enumerate_interval_bags_from_coverage(coverage: Dict[int, int]) -> List[Bag]:
    return list(enumerate_interval_bags_from_coverage_iter(coverage))


def build_An_quiver(n: int):
    """Equioriented A_n: v0 -> v1 -> ... -> v{n-1}."""
    Q = Quiver(f"A{n}")
    verts = [Q.add_vertex(f"v{k}") for k in range(n)]
    for k in range(n - 1):
        Q.add_arrow(verts[k], verts[k + 1], f"a{k}")
    return Q

def make_rad_jobs_for_An(n: int, Q: Quiver = None,
                         coverage: dict[int, int] = None,
                         target_dim=None):
    """
    Returns (Q, jobs) where jobs is a list of (Q, bag, target_dim) triples
    covering all interval-multiset isomorphism classes with coverage d=(n+1,...,n+1)
    and target_dim = {0:1, 1:2, ..., n-1:n}.
    """
    if not Q:
        Q = build_An_quiver(n)
    c = n + 1
    if not coverage:
        bags = enumerate_interval_bags_An(n, c)
    else:
        bags = enumerate_interval_bags_from_coverage(coverage)
    print(f"{len(bags)} jobs detected")
    if not target_dim:
        target_dim = {k: k + 1 for k in range(n)}
    jobs = [(Q, bag, target_dim) for bag in bags]
    return Q, jobs


# ============================================================================
# D_n Enumeration Functions
# ============================================================================

def enumerate_Dn_bags_from_coverage_iter(coverage: Dict[int, int]) -> Iterator[Bag]:
    """
    Enumerate all bags of D_n indecomposables (SEG/UP/DOWN/BOTH/STAR) realizing the given vertex coverage.

    Conventions (match your builder):
      - n = max(coverage)+1, br = n-3 is the branch; leaves are up_leaf = n-2 and dn_leaf = n-1.
      - SEG[i,j]: 1 on spine p in [i..j] (closed interval, 0 <= i <= j <= br).
      - UP(i):    i<=br -> 1 on spine [i..br], +1 on up_leaf; i==up_leaf -> pure up simple.
      - DOWN(i):  i<=br -> 1 on spine [i..br], +1 on dn_leaf; i==dn_leaf -> pure down simple.
      - BOTH(i):  i<=br -> 1 on spine [i..br], +1 on both leaves.
      - STAR[i,j*]: 0 <= i < j <= br: 1 on [i..j-1], 2 on [j..br], +1 on both leaves.

    Deterministic; no in-place mutation and no backtracking/undo. Spine invariants enforced exactly.
    """
    if not coverage:
        return
    n = max(coverage) + 1
    if n < 4:
        return

    br = n - 3
    up_leaf = n - 2
    dn_leaf = n - 1

    # Dense target coverage
    c = [int(coverage.get(k, 0)) for k in range(n)]
    if any(v < 0 for v in c):
        raise ValueError("Coverage must be nonnegative integers.")

    # A "Singles" state is a tuple of (start_index, count) buckets, in creation order.
    # A "Doubles" (STAR) state is a dict {(start, split): count}.
    from collections import defaultdict

    def choose_comp(need: int, caps: List[int]) -> Iterator[List[int]]:
        """All vectors 'take' with 0<=take[i]<=caps[i], sum=need. Older-first (big to small)."""
        m = len(caps)
        take: List[int] = []
        def rec(i: int, rem: int):
            if i == m:
                if rem == 0:
                    yield take.copy()
                return
            cap = caps[i]
            # older-first: take as much as possible first
            for t in range(min(cap, rem), -1, -1):
                take.append(t)
                yield from rec(i + 1, rem - t)
                take.pop()
        yield from rec(0, need)

    def add_seg(bag: Bag, i: int, j: int, t: int) -> Bag:
        if t == 0: return bag
        # Enumeration uses half-open [i, j), but DnIndec uses closed [i, j].
        # Convert: SEG covering spine [i..j-1] becomes DnIndec(SEG, i, j-1).
        # j can be br+1 (so j-1 = br), making SEG(i, br) valid.
        return bag + [DnIndec(DnKind.SEG, i, j - 1)] * t

    def add_star(bag: Bag, i: int, j: int, t: int) -> Bag:
        if t == 0: return bag
        return bag + [DnIndec(DnKind.STAR, i, j)] * t

    def add_up(bag: Bag, i: int, t: int) -> Bag:
        if t == 0: return bag
        return bag + [DnIndec(DnKind.UP, i)] * t

    def add_down(bag: Bag, i: int, t: int) -> Bag:
        if t == 0: return bag
        return bag + [DnIndec(DnKind.DOWN, i)] * t

    def add_both(bag: Bag, i: int, t: int) -> Bag:
        if t == 0: return bag
        return bag + [DnIndec(DnKind.BOTH, i)] * t

    def spine_totals(singles: Tuple[Tuple[int,int], ...],
                     stars: Dict[Tuple[int,int], int]) -> Tuple[int,int,int]:
        U = sum(cnt for _, cnt in singles)
        D = sum(stars.values())
        return U, D, U + 2*D

    # Definition-level tally to verify outputs (inside the function; no outside patching).
    def tally_cov(bag: Bag) -> List[int]:
        cov = [0]*n
        for x in bag:
            k = x.kind
            i, j = x.i, x.j
            if k == DnKind.SEG:
                # SEG[i,j] covers closed interval [i..j], with 0 <= i <= j <= br
                for p in range(i, j + 1): cov[p] += 1
            elif k == DnKind.UP:
                if i <= br:
                    for p in range(i, br+1): cov[p] += 1
                else:
                    assert i == up_leaf
                cov[up_leaf] += 1
            elif k == DnKind.DOWN:
                if i <= br:
                    for p in range(i, br+1): cov[p] += 1
                else:
                    assert i == dn_leaf
                cov[dn_leaf] += 1
            elif k == DnKind.BOTH:
                for p in range(i, br+1): cov[p] += 1
                cov[up_leaf] += 1; cov[dn_leaf] += 1
            else:  # STAR
                if i <= j - 1:
                    for p in range(i, min(j-1, br)+1): cov[p] += 1
                if j <= br:
                    for p in range(j, br+1): cov[p] += 2
                cov[up_leaf] += 1; cov[dn_leaf] += 1
        return cov

    # ---- recursive spine sweep (pure) ----

    def dfs_spine(k: int,
                  singles: Tuple[Tuple[int,int], ...],
                  stars: Dict[Tuple[int,int], int],
                  bag: Bag):
        if k == 0:
            # open exactly c[0] singles
            s0 = c[0]
            if s0 > 0:
                singles1 = singles + ((0, s0),)
            else:
                singles1 = singles
            yield from dfs_spine(1, singles1, stars, bag)
            return

        if k <= br:
            # invariant at k-1: U+2D must equal c[k-1]
            Uprev, Dprev, Tprev = spine_totals(singles, stars)
            if Tprev != c[k-1]:
                return

            # prepare caps for endings/splits
            starts = [i for (i, cnt) in singles]
            caps   = [cnt for (i, cnt) in singles]
            Uprev_only = sum(caps)

            # e in [0..Uprev], t in [0..Uprev-e], s = (c[k]-c[k-1]) + e - t >= 0
            for e in range(Uprev_only, -1, -1):
                # choose which buckets end at k (only those with start < k are valid)
                for take_e in choose_comp(e, caps):
                    # Validate end choices: only from buckets with start < k
                    ok_end = True
                    for idx, t_end in enumerate(take_e):
                        if t_end and not (starts[idx] < k):
                            ok_end = False; break
                    if not ok_end:
                        continue

                    # apply ends: new singles counts, and add corresponding SEG(i,k)
                    new_caps_e = caps[:]
                    bag_e = bag
                    for idx, t_end in enumerate(take_e):
                        if t_end:
                            new_caps_e[idx] -= t_end
                            bag_e = add_seg(bag_e, starts[idx], k, t_end)

                    Ue = sum(new_caps_e)

                    for t in range(Ue, -1, -1):
                        for take_t in choose_comp(t, new_caps_e):
                            # apply splits: move from singles to doubles at (start, k)
                            new_caps_t = new_caps_e[:]
                            stars_t = stars.copy()
                            for idx, t_split in enumerate(take_t):
                                if t_split:
                                    new_caps_t[idx] -= t_split
                                    key = (starts[idx], k)
                                    stars_t[key] = stars_t.get(key, 0) + t_split

                            # s determined by invariant at k
                            s = (c[k] - c[k-1]) + e - t
                            if s < 0:
                                continue

                            # open s new singles (bucket at start=k if s>0)
                            if s > 0:
                                singles_k = tuple((starts[i], new_caps_t[i]) for i in range(len(starts))) + ((k, s),)
                            else:
                                singles_k = tuple((starts[i], new_caps_t[i]) for i in range(len(starts)))

                            # check invariant at k
                            _, _, Tnow = spine_totals(singles_k, stars_t)
                            if Tnow != c[k]:
                                continue

                            # prune empty buckets (count==0) for neatness (not required)
                            singles_clean = tuple((i, cnt) for (i, cnt) in singles_k if cnt > 0)

                            yield from dfs_spine(k + 1, singles_clean, stars_t, bag_e)
            return

        # k == br+1: branch allocation and finish
        # singles: ((start_i, multiplicity), ...), starts must be <= br by construction
        U = sum(cnt for _, cnt in singles)
        D = sum(stars.values())
        cup = c[up_leaf]
        cdn = c[dn_leaf]

        # leaf-only simples do not consume singles
        for lu in range(cup, -1, -1):
            for ld in range(cdn, -1, -1):
                cup1 = cup - lu
                cdn1 = cdn - ld
                need_up = cup1 - D
                need_dn = cdn1 - D
                if need_up < 0 or need_dn < 0:
                    continue

                # tight bounds for BOTH:
                # u_up + u_both = need_up, u_dn + u_both = need_dn,
                # u_up + u_dn + u_both <= U  =>  (need_up + need_dn - 2*u_both) <= U
                # => u_both >= ceil((need_up + need_dn - U)/2), and u_both <= min(need_up, need_dn)
                lb_raw = need_up + need_dn - U
                lb = (lb_raw + 1) // 2 if lb_raw > 0 else 0
                ub = min(need_up, need_dn)
                if ub < lb:
                    continue

                for u_both in range(ub, lb - 1, -1):
                    u_up = need_up - u_both
                    u_dn = need_dn - u_both
                    u_seg = U - (u_up + u_dn + u_both)
                    if u_seg < 0:
                        continue

                    # Distribute UP among singles
                    caps = [cnt for (i, cnt) in singles]
                    starts = [i for (i, cnt) in singles]
                    for take_up in choose_comp(u_up, caps):
                        caps_u = [caps[i] - take_up[i] for i in range(len(caps))]
                        bag_u = bag
                        for idx, t_up in enumerate(take_up):
                            if t_up:
                                bag_u = add_up(bag_u, starts[idx], t_up)

                        # Distribute DOWN
                        for take_dn in choose_comp(u_dn, caps_u):
                            caps_d = [caps_u[i] - take_dn[i] for i in range(len(caps))]
                            bag_d = bag_u
                            for idx, t_dn in enumerate(take_dn):
                                if t_dn:
                                    bag_d = add_down(bag_d, starts[idx], t_dn)

                            # Distribute BOTH
                            for take_b in choose_comp(u_both, caps_d):
                                caps_b = [caps_d[i] - take_b[i] for i in range(len(caps))]
                                bag_b = bag_d
                                for idx, t_b in enumerate(take_b):
                                    if t_b:
                                        bag_b = add_both(bag_b, starts[idx], t_b)

                                # Remaining singles become SEG(start, br+1)
                                bag_s = bag_b
                                for idx, leftover in enumerate(caps_b):
                                    if leftover:
                                        bag_s = add_seg(bag_s, starts[idx], br+1, leftover)

                                # Add STARs
                                bag_star = bag_s
                                if stars:
                                    for (i0, j0), cnt in stars.items():
                                        if cnt:
                                            bag_star = add_star(bag_star, i0, j0, cnt)

                                # Leaf-only simples
                                if lu:
                                    bag_star = add_up(bag_star, up_leaf, lu)
                                if ld:
                                    bag_star = add_down(bag_star, dn_leaf, ld)

                                # Final check and emit
                                if tally_cov(bag_star) == c:
                                    yield bag_star
        return

    # Kick off with empty state
    yield from dfs_spine(0, singles=tuple(), stars={}, bag=[])


def enumerate_Dn_bags_from_coverage(coverage: Dict[int, int]) -> List[Bag]:
    return list(enumerate_Dn_bags_from_coverage_iter(coverage))


def make_rad_jobs_for_Dn(n: int, Q=None,
                        coverage: Dict[int, int] = None,
                        target_dim: Dict[int, int] = None) -> Tuple:
    """
    Generate batch jobs for radicality testing on D_n quivers.

    Returns (Q, jobs) where jobs is a list of (Q, bag, target_dim) tuples
    covering all indecomposable bag isomorphism classes.

    Args:
        n: Size of D_n quiver (must be >= 4)
        Q: D_n quiver instance (optional, will create if not provided)
        coverage: Coverage dict (optional, uses default if not provided)
        target_dim: Target dimension dict (optional, infers from coverage)

    Returns:
        Tuple of (Quiver, list of jobs)
        Each job is a tuple (Quiver, bag, target_dim)

    Note:
        Default coverage and target_dim depend on the D_n structure.
        Branch vertex is n-3, leaves are n-2 and n-1.

    """

    bags = enumerate_Dn_bags_from_coverage(coverage)
    print(f"{len(bags)} jobs detected")
    jobs = [(Q, bag, target_dim) for bag in bags]
    return Q, jobs
