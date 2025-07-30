import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import json

from .field import Field, ZeroMap
from .morphism import Morphism
from .module import Module

class Quiver:
    """
    A class representing a quiver (directed graph) without oriented cycles.
    Uses adjacency list representation internally.
    """

    def __init__(self, name: str = ""):
        """
        Initialize an empty quiver.

        Args:
            name: Optional name for the quiver
        """
        self.name = name
        self.vertices = {}  # Dict[int, Dict] - vertex_id -> vertex properties
        self.arrows = {}    # Dict[int, Dict] - arrow_id -> arrow properties
        self.next_vertex_id = 0
        self.next_arrow_id = 0

        # Adjacency list representation
        self.successors = {}  # Dict[int, List[Tuple[int, int]]] - vertex_id -> [(target_vertex_id, arrow_id)]
        self.predecessors = {}  # Dict[int, List[Tuple[int, int]]] - vertex_id -> [(source_vertex_id, arrow_id)]

    def add_vertex(self, label: Optional[str] = None, **properties) -> int:
        """
        Add a vertex to the quiver.

        Args:
            label: Optional label for the vertex
            **properties: Additional properties for the vertex

        Returns:
            The ID of the new vertex
        """
        vertex_id = self.next_vertex_id
        self.next_vertex_id += 1

        if label is None:
            label = str(vertex_id)

        self.vertices[vertex_id] = {"label": label, **properties}
        self.successors[vertex_id] = []
        self.predecessors[vertex_id] = []

        return vertex_id

    def add_arrow(self, source: int, target: int, label: Optional[str] = None, **properties) -> int:
        """
        Add an arrow between two vertices.

        Args:
            source: ID of the source vertex
            target: ID of the target vertex
            label: Optional label for the arrow
            **properties: Additional properties for the arrow

        Returns:
            The ID of the new arrow

        Raises:
            ValueError: If source or target vertex does not exist
            ValueError: If adding the arrow would create a cycle
        """
        if source not in self.vertices:
            raise ValueError(f"Source vertex {source} does not exist")
        if target not in self.vertices:
            raise ValueError(f"Target vertex {target} does not exist")

        # Check if adding this arrow would create a cycle
        if self._would_create_cycle(source, target):
            raise ValueError(f"Adding arrow from {source} to {target} would create a cycle")

        arrow_id = self.next_arrow_id
        self.next_arrow_id += 1

        if label is None:
            label = f"{source}→{target}"

        self.arrows[arrow_id] = {
            "source": source,
            "target": target,
            "label": label,
            **properties
        }

        self.successors[source].append((target, arrow_id))
        self.predecessors[target].append((source, arrow_id))

        return arrow_id

    def remove_vertex(self, vertex_id: int) -> None:
        """
        Remove a vertex and all its incident arrows from the quiver.

        Args:
            vertex_id: ID of the vertex to remove

        Raises:
            ValueError: If vertex does not exist
        """
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist")

        # Remove all arrows involving this vertex
        arrows_to_remove = []

        # Outgoing arrows
        for target, arrow_id in self.successors[vertex_id]:
            arrows_to_remove.append(arrow_id)
            # Remove from target's predecessors
            self.predecessors[target] = [
                (v, a) for v, a in self.predecessors[target] if v != vertex_id
            ]

        # Incoming arrows
        for source, arrow_id in self.predecessors[vertex_id]:
            if arrow_id not in arrows_to_remove:  # Avoid duplicates
                arrows_to_remove.append(arrow_id)
            # Remove from source's successors
            self.successors[source] = [
                (v, a) for v, a in self.successors[source] if v != vertex_id
            ]

        # Remove the arrows
        for arrow_id in arrows_to_remove:
            self.arrows.pop(arrow_id, None)

        # Remove the vertex
        self.vertices.pop(vertex_id)
        self.successors.pop(vertex_id)
        self.predecessors.pop(vertex_id)

    def remove_arrow(self, arrow_id: int) -> None:
        """
        Remove an arrow from the quiver.

        Args:
            arrow_id: ID of the arrow to remove

        Raises:
            ValueError: If arrow does not exist
        """
        if arrow_id not in self.arrows:
            raise ValueError(f"Arrow {arrow_id} does not exist")

        arrow = self.arrows[arrow_id]
        source = arrow["source"]
        target = arrow["target"]

        # Remove from source's successors
        self.successors[source] = [
            (v, a) for v, a in self.successors[source] if a != arrow_id
        ]

        # Remove from target's predecessors
        self.predecessors[target] = [
            (v, a) for v, a in self.predecessors[target] if a != arrow_id
        ]

        # Remove the arrow
        self.arrows.pop(arrow_id)

    def get_vertices(self) -> List[int]:
        """Return a list of all vertex IDs."""
        return list(self.vertices.keys())

    def get_arrows(self) -> List[int]:
        """Return a list of all arrow IDs."""
        return list(self.arrows.keys())

    def get_sources(self) -> List[int]:
        """Return a list of all source vertices (vertices with no incoming arrows)."""
        return [v for v in self.vertices if not self.predecessors[v]]

    def get_sinks(self) -> List[int]:
        """Return a list of all sink vertices (vertices with no outgoing arrows)."""
        return [v for v in self.vertices if not self.successors[v]]

    def get_vertex_successors(self, vertex_id: int) -> List[int]:
        """Return a list of vertices that are targets of arrows from the given vertex."""
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist")
        return [target for target, _ in self.successors[vertex_id]]

    def get_vertex_predecessors(self, vertex_id: int) -> List[int]:
        """Return a list of vertices that are sources of arrows to the given vertex."""
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex {vertex_id} does not exist")
        return [source for source, _ in self.predecessors[vertex_id]]

    def get_arrows_between(self, source: int, target: int) -> List[int]:
        """Return a list of arrow IDs from source to target."""
        if source not in self.vertices:
            raise ValueError(f"Source vertex {source} does not exist")
        if target not in self.vertices:
            raise ValueError(f"Target vertex {target} does not exist")

        return [arrow_id for t, arrow_id in self.successors[source] if t == target]

    def _would_create_cycle(self, source: int, target: int) -> bool:
        """
        Check if adding an arrow from source to target would create a cycle.

        We don't want cycles in our quiver, so this ensures we maintain a DAG.
        """
        if source == target:
            return True  # Self-loop is a cycle

        # Check if there's a path from target to source
        visited = set()
        queue = [target]

        while queue:
            current = queue.pop(0)
            if current == source:
                return True
            if current in visited:
                continue

            visited.add(current)
            for next_vertex, _ in self.successors[current]:
                if next_vertex not in visited:
                    queue.append(next_vertex)

        return False

    def visualize(self, figsize=(10, 8)) -> None:
        """
        Visualize the quiver using networkx and matplotlib.

        Args:
            figsize: Figure size as a tuple (width, height)
        """
        G = nx.DiGraph()

        # Add vertices
        for vertex_id, vertex_data in self.vertices.items():
            G.add_node(vertex_id, label=vertex_data["label"])

        # Add edges with labels
        edge_labels = {}
        for arrow_id, arrow_data in self.arrows.items():
            source = arrow_data["source"]
            target = arrow_data["target"]
            label = arrow_data["label"]
            G.add_edge(source, target, key=arrow_id)
            edge_labels[(source, target)] = label

        # Create the plot
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)

        # Draw vertices and edges
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

        # Draw labels
        vertex_labels = {v: self.vertices[v]["label"] for v in self.vertices}
        nx.draw_networkx_labels(G, pos, labels=vertex_labels, font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        plt.axis("off")
        plt.title(f"Quiver: {self.name}" if self.name else "Quiver")
        plt.tight_layout()
        plt.show()

    def to_dict(self) -> Dict:
        """Convert the quiver to a dictionary for serialization."""
        return {
            "name": self.name,
            "vertices": self.vertices,
            "arrows": self.arrows,
            "next_vertex_id": self.next_vertex_id,
            "next_arrow_id": self.next_arrow_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Quiver':
        """Create a quiver from a dictionary."""
        quiver = cls(name=data.get("name", ""))

        # Set internal state
        quiver.vertices = data["vertices"]
        quiver.arrows = data["arrows"]
        quiver.next_vertex_id = data["next_vertex_id"]
        quiver.next_arrow_id = data["next_arrow_id"]

        # Rebuild adjacency lists
        quiver.successors = {vertex_id: [] for vertex_id in quiver.vertices}
        quiver.predecessors = {vertex_id: [] for vertex_id in quiver.vertices}

        for arrow_id, arrow_data in quiver.arrows.items():
            source = arrow_data["source"]
            target = arrow_data["target"]
            quiver.successors[source].append((target, arrow_id))
            quiver.predecessors[target].append((source, arrow_id))

        return quiver

    def save(self, filename: str) -> None:
        """Save the quiver to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'Quiver':
        """Load a quiver from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the quiver."""
        s = f"Quiver '{self.name}' with {len(self.vertices)} vertices and {len(self.arrows)} arrows\n"
        s += "Vertices: " + ", ".join(f"{v}:{self.vertices[v]['label']}" for v in self.vertices) + "\n"
        s += "Arrows: " + ", ".join(
            f"{a}:{self.vertices[self.arrows[a]['source']]['label']}→{self.vertices[self.arrows[a]['target']]['label']} ({self.arrows[a]['label']})"
            for a in self.arrows
        )
        return s


def topological_sort(quiver):
    """
    Perform a topological sort of the quiver vertices.

    Returns:
        A list of vertex IDs in topological order (sources first)
    """
    # Get all vertices
    vertices = quiver.get_vertices()

    # Initialize in-degree count for each vertex
    in_degree = {v: 0 for v in vertices}

    # Count incoming edges for each vertex
    for v in vertices:
        for source, _ in quiver.predecessors[v]:
            in_degree[v] += 1

    # Initialize queue with vertices that have no incoming edges (sources)
    queue = [v for v in vertices if in_degree[v] == 0]

    # Initialize result list
    result = []

    # Process vertices in topological order
    while queue:
        # Remove a vertex with no incoming edges
        v = queue.pop(0)

        # Add it to the result
        result.append(v)

        # Decrement in-degree of each successor
        for target, _ in quiver.successors[v]:
            in_degree[target] -= 1

            # If the successor has no more incoming edges, add it to the queue
            if in_degree[target] == 0:
                queue.append(target)

    # Check if we visited all vertices
    if len(result) != len(vertices):
        raise ValueError("Quiver contains a cycle")

    return result
