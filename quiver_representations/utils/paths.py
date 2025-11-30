"""
Path and quiver path utilities.
"""

from dataclasses import dataclass, field
from typing import Tuple, Any


@dataclass
class PathRec:
    """Lightweight carrier for a path and its composite matrix."""
    tail: int
    arrows: Tuple[int, ...]
    head: int
    # matrix is carried for relation construction; never used as a dict key
    matrix: Any = field(repr=False)

    @property
    def length(self) -> int:
        return len(self.arrows)
