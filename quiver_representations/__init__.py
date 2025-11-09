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
    enumerate_Dn_bags_from_coverage,
    make_rad_jobs_for_Dn,
)

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

from .utils import (
    PathRec,
    module_over_field,
)

from .analysis import (
    # Homomorphisms
    hom_interval,
    hom_interval_to_bag,
    find_hom_basis,
    # Poset construction
    write_rank_poset_from_jobs,
    write_rank_poset_from_jobs_Dn,
    report_local_minima_dn,
    # Conjectures
    check_conjectures,
    check_conjectures_write,
    select_generic_jobs_strict,
    read_ranks_csv_strict,
    read_edges_csv_strict,
    read_parsed_csv_strict,
    # Visualization
    visualize_degeneracy_dag,
)

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
    
    # Analysis
    "hom_interval",
    "hom_interval_to_bag",
    "find_hom_basis",
    "write_rank_poset_from_jobs",
    "write_rank_poset_from_jobs_Dn",
    "report_local_minima_dn",
    "check_conjectures",
    "check_conjectures_write",
    "select_generic_jobs_strict",
    "read_ranks_csv_strict",
    "read_edges_csv_strict",
    "read_parsed_csv_strict",
    "visualize_degeneracy_dag",

    # D_n enumeration
    "enumerate_Dn_bags_from_coverage",
    "make_rad_jobs_for_Dn",

    # Utils
    "PathRec",
    "module_over_field",
]
