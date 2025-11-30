"""
Common utilities for poset construction

Provides generic algorithms used across different quiver types:
- Transitive reduction (Hasse diagram construction)
"""


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
