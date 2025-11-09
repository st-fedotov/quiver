"""
Quiver Representations Package

A Python library for working with quivers, their representations, and Grassmannians.
"""

from .quiver import Quiver
from .field import Field, FiniteField, ComplexNumbers, ZeroMap
from .module import Module
from .morphism import Morphism

# NEW: Grassmannian exports
from .grassmannians import (
    Grassmannian,
    QuiverGrassmannian,
    PluckerVar,
    SymbolRegistry,
    PluckerPolynomial,
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

# NEW: Interval module exports
from .interval_modules import (
    # Type A
    build_interval_An,
    build_interval_explicit,
    form_An_module_from_intervals,
    form_An_module_from_intervals_explicit,
    write_batch_rad_from_interval_bags,
    write_batch_rad_from_interval_bags_explicit,
    # Type D
    DnKind,
    DnIndec,
    seg,
    up,
    down,
    both,
    star,
    make_Dn_quiver,
    form_Dn_module_from_bag_explicit,
    write_batch_rad_from_dn_bags,
    bag_pretty,
)

# NEW: Batch processing exports
from .batch import (
    write_batch_rad_from_triples,
    parse_quiver_jobs_rad,
    print_results_table_rad,
    collect_minimal_generators_markdown,
    parse_ring_block,
    load_manifest,
    parse_ifullsat_gb,
    compute_hilbert_series,
    collect_hilbert_results,
    relpath,
)

from .utils import PathRec

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",

    "Quiver", 
    "Field",
    "FiniteField",
    "ComplexNumbers",
    "Module",
    "ZeroMap",
    "Morphism"

    # Grassmannians
    "Grassmannian",
    "QuiverGrassmannian",
    "PluckerVar",
    "SymbolRegistry",
    "PluckerPolynomial",
    "are_projectively_equivalent",
    "pick_lucky_primes_from_gb",
    "rationalize",
    "build_block_vars",
    "vertex_order",
    "poly_to_m2",
    "coef_to_m2",
    "term_to_str",
    "vname",

    # Interval and Dn-specific modules
    "build_interval_An",
    "build_interval_explicit",
    "form_An_module_from_intervals",
    "form_An_module_from_intervals_explicit",
    "write_batch_rad_from_interval_bags",
    "write_batch_rad_from_interval_bags_explicit",
    "DnKind",
    "DnIndec",
    "seg",
    "up",
    "down",
    "both",
    "star",
    "make_Dn_quiver",
    "form_Dn_module_from_bag_explicit",
    "write_batch_rad_from_dn_bags",
    "bag_pretty",

    # Batch processing
    "write_batch_rad_from_triples",
    "parse_quiver_jobs_rad",
    "print_results_table_rad",
    "collect_minimal_generators_markdown",
    "parse_ring_block",
    "load_manifest",
    "parse_ifullsat_gb",
    "compute_hilbert_series",
    "collect_hilbert_results",
    "relpath",

    # Utilities
    "PathRec",
]
