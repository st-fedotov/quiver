"""
Grassmannian subpackage for classical and quiver Grassmannians.
"""

from .classical import Grassmannian
from .quiver_grassmannian import QuiverGrassmannian
from .plucker import PluckerVar, SymbolRegistry, PluckerPolynomial
from .utils import (
    are_projectively_equivalent,
    pick_lucky_primes_from_gb,
    rationalize,
    build_block_vars,
    vertex_order,
    poly_to_m2,
    coef_to_m2,
    term_to_str,
    vname,
)

__all__ = [
    # Classes
    "Grassmannian",
    "QuiverGrassmannian",
    "PluckerVar",
    "SymbolRegistry",
    "PluckerPolynomial",
    # Functions
    "are_projectively_equivalent",
    "pick_lucky_primes_from_gb",
    "rationalize",
    "build_block_vars",
    "vertex_order",
    "poly_to_m2",
    "coef_to_m2",
    "term_to_str",
    "vname",
]
