"""
DAG visualization for degeneration posets.

Provides functions to visualize Hasse diagrams of degeneration
partial orders using matplotlib and networkx.

Notebook source: Cell 38
"""

from typing import Dict, Tuple, Union
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


def _dot_layout_TB(G, prefer_pygraphviz=True, ranksep=1.1, nodesep=0.45, equally=True):
    """
    Returns positions with Graphviz dot Top->Bottom.
    ranksep/nodesep are in *inches* (Graphviz units).
    """
    args = f'-Grankdir=TB -Granksep="{ranksep}{" equally" if equally else ""}" -Gnodesep={nodesep}'
    pos = None
    if prefer_pygraphviz:
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(G, prog="dot", args=args)
        except Exception:
            pass
    if pos is None:
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
        except Exception:
            pos = None
    return pos


def visualize_degeneracy_dag(
    jobs,
    rank_dir: str = "rank_poset",
    parsed_csv: str = "parsed.csv",
    out_base: str = "degeneracy_dag",
    magic_number: int = 8,
    *,
    ranksep: float = 1.3,   # inches between ranks (Graphviz dot)
    nodesep: float = 0.5,   # inches between nodes in same rank
    equally: bool = True,   # use 'ranksep="X equally"' for uniform layers
    engine: str = "dot",    # dot is the hierarchical DAG layout
    fmt: str = "svg"        # vector; use 'pdf' if you prefer
):
    """
    Vector-render the degeneracy DAG (Hasse edges) via Graphviz.
    Writes:
      - {out_base}.dot
      - {out_base}.svg (or .pdf), if Graphviz 'dot' is available

    Edge coloring:
      - BLUE     iff BOTH endpoints have irred_dims_list exactly [magic_number].
      - MAGENTA  iff BOTH endpoints have irred_dims_list = k*[magic_number], l*[magic_number]
                  with at least one of k or l > 1.
      - default  otherwise.
    """

    # ---- read input files (reuse your helpers) ----
    rank_dir = Path(rank_dir)
    edges_path = rank_dir / "edges.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"{edges_path} not found")
    ids = list(range(len(jobs)))
    adj = read_edges_csv_strict(edges_path, ids, orientation="degenerates_to")
    geom = read_parsed_csv_strict(parsed_csv, ids)

    # predicates for coloring
    def all_magic(lst):
        return bool(lst) and all(int(x) == magic_number for x in lst)

    def exactly_magic(lst):
        return isinstance(lst, list) and len(lst) == 1 and int(lst[0]) == magic_number

    all_magic_t    = {jid: all_magic(geom[jid]["irred_dims_list"])    for jid in ids}
    exactly_magic_t= {jid: exactly_magic(geom[jid]["irred_dims_list"])for jid in ids}

    # ---- assemble DOT via pygraphviz OR pydot ----
    graph_attrs = {
        "rankdir": "TB",
        "ranksep": f'{ranksep}{" equally" if equally else ""}',
        "nodesep": str(nodesep),
        "splines": "true",
        "overlap": "false",
        "concentrate": "false",
    }
    node_attrs = {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "#f1f5f9",
        "color": "#334155",
        "fontname": "DejaVu Sans",
        "fontsize": "10",
    }
    edge_attrs = {
        "arrowsize": "0.7",
        "arrowhead": "normal",
        "color": "#334155",
        "penwidth": "1.0",
    }

    def fmt_dimvec(d):
        if not d:
            return "[]"
        n = max(d.keys()) + 1
        return "[" + ",".join(str(d.get(k, 0)) for k in range(n)) + "]"

    labels = {}
    for jid in ids:
        target_dim = jobs[jid][2] or {}
        irr = geom[jid]["irred_dims_list"] or []
        # labels[jid] = f"{jid}\\nP={fmt_dimvec(target_dim)}\\nirr={irr}"
        labels[jid] = f"{jid}\\nirr={irr}"

    dot_path = Path(f"{out_base}.dot")
    svg_or_pdf_path = Path(f"{out_base}.{fmt}")

    try:
        # Preferred: pygraphviz (most control)
        import pygraphviz as pgv
        A = pgv.AGraph(directed=True, strict=False)
        for k, v in graph_attrs.items():
            A.graph_attr[k] = v
        for k, v in node_attrs.items():
            A.node_attr[k] = v
        for k, v in edge_attrs.items():
            A.edge_attr[k] = v

        # nodes
        for jid in ids:
            A.add_node(jid, label=labels[jid])

        # edges (Hasse), with conditional styling:
        # 1) both all-magic? -> blue if both exactly_magic else magenta
        # 2) otherwise default
        for u in ids:
            for v in adj.get(u, []):
                if u == v:
                    continue
                if all_magic_t[u] and all_magic_t[v]:
                    if exactly_magic_t[u] and exactly_magic_t[v]:
                        A.add_edge(u, v, color="blue", penwidth="1.8")
                    else:
                        A.add_edge(u, v, color="magenta", penwidth="1.8")
                else:
                    A.add_edge(u, v)

        A.write(str(dot_path))
        # Try to render if 'dot' is available
        if shutil.which(engine):
            A.draw(str(svg_or_pdf_path), prog=engine, format=fmt)
            print(f"[OK] wrote {dot_path} and {svg_or_pdf_path} (pygraphviz/{engine})")
        else:
            print(f"[OK] wrote {dot_path}. Install Graphviz to render {fmt}.")
        return

    except Exception:
        # Fallback: pydot to write DOT; then call 'dot' CLI if available
        try:
            import pydot
        except Exception:
            # Last resort: write raw DOT manually
            with open(dot_path, "w", encoding="utf-8") as f:
                f.write("digraph G {\n")
                for k, v in graph_attrs.items():
                    f.write(f'  {k}="{v}";\n')
                f.write("  node [")
                f.write(",".join(f'{k}="{v}"' for k, v in node_attrs.items()))
                f.write("];\n  edge [")
                f.write(",".join(f'{k}="{v}"' for k, v in edge_attrs.items()))
                f.write("];\n")
                for jid in ids:
                    f.write(f'  {jid} [label="{labels[jid]}"];\n')
                for u in ids:
                    for v in adj.get(u, []):
                        if u == v:
                            continue
                        if all_magic_t[u] and all_magic_t[v]:
                            if exactly_magic_t[u] and exactly_magic_t[v]:
                                f.write(f'  {u} -> {v} [color="blue", penwidth="1.8"];\n')
                            else:
                                f.write(f'  {u} -> {v} [color="magenta", penwidth="1.8"];\n')
                        else:
                            f.write(f"  {u} -> {v};\n")
                f.write("}\n")
            if shutil.which(engine):
                subprocess.run([engine, f"-T{fmt}", str(dot_path), "-o", str(svg_or_pdf_path)], check=True)
                print(f"[OK] wrote {dot_path} and {svg_or_pdf_path} (raw DOT/{engine})")
            else:
                print(f"[OK] wrote {dot_path}. Install Graphviz to render {fmt}.")
            return

        # pydot path
        g = pydot.Dot(graph_type="digraph")
        for k, v in graph_attrs.items():
            g.set(k, v)
        # nodes
        for jid in ids:
            node = pydot.Node(str(jid))
            node.set("label", labels[jid])
            for k, v in node_attrs.items():
                node.set(k, v)
            g.add_node(node)
        # edges with conditional styling
        for u in ids:
            for v in adj.get(u, []):
                if u == v:
                    continue
                e = pydot.Edge(str(u), str(v))
                if all_magic_t[u] and all_magic_t[v]:
                    if exactly_magic_t[u] and exactly_magic_t[v]:
                        e.set("color", "blue")
                        e.set("penwidth", "1.8")
                    else:
                        e.set("color", "magenta")
                        e.set("penwidth", "1.8")
                for k, v2 in edge_attrs.items():
                    # don't overwrite color/penwidth if set
                    if k in ("color", "penwidth") and (all_magic_t[u] and all_magic_t[v]):
                        continue
                    e.set(k, v2)
                g.add_edge(e)

        g.write_raw(str(dot_path))
        if shutil.which(engine):
            subprocess.run([engine, f"-T{fmt}", str(dot_path), "-o", str(svg_or_pdf_path)], check=True)
            print(f"[OK] wrote {dot_path} and {svg_or_pdf_path} (pydot/{engine})")
        else:
            print(f"[OK] wrote {dot_path}. Install Graphviz to render {fmt}.")
