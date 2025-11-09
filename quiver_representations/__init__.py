"""
Quiver Representations Package

A Python library for working with quivers, their representations, and Grassmannians.
"""

from .quiver import Quiver
from .field import Field, FiniteField, ComplexNumbers, ZeroMap
from .module import Module
from .morphism import Morphism

# Grassmannian exports
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

# Interval module exports
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
    # A_n enumeration
    enumerate_interval_bags_An,
    enumerate_interval_bags_from_coverage_iter,
    enumerate_interval_bags_from_coverage,
    build_An_quiver,
    make_rad_jobs_for_An,
    # D_n enumeration
    enumerate_Dn_bags_from_coverage_iter,
    enumerate_Dn_bags_from_coverage,
    make_rad_jobs_for_Dn,
)

# Batch processing exports
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

# Analysis exports
from .analysis import (
    # Homomorphisms
    hom_interval,
    hom_interval_to_bag,
    find_hom_basis,
    # Poset A_n
    write_rank_poset_from_jobs,
    build_dir_edges,
    # Poset D_n
    write_rank_poset_from_jobs_Dn,
    report_local_minima_dn,
    # Conjectures - CSV readers
    read_ranks_csv_strict,
    read_edges_csv_strict,
    read_parsed_csv_strict,
    # Conjectures - Main
    check_conjectures,
    check_conjectures_write,
    select_generic_jobs_strict,
    # Conjectures - Utilities
    transitive_closure,
    maximal_in_subset,
    # Visualization
    visualize_degeneracy_dag,
)

# Utils exports
from .utils import (
    PathRec,
    module_over_field,
)

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",

    # Core
    "Quiver",
    "Field",
    "FiniteField",
    "ComplexNumbers",
    "Module",
    "ZeroMap",
    "Morphism",

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

    # Interval modules - Type A
    "build_interval_An",
    "build_interval_explicit",
    "form_An_module_from_intervals",
    "form_An_module_from_intervals_explicit",
    "write_batch_rad_from_interval_bags",
    "write_batch_rad_from_interval_bags_explicit",

    # Interval modules - Type D
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

    # Enumeration - A_n
    "enumerate_interval_bags_An",
    "enumerate_interval_bags_from_coverage_iter",
    "enumerate_interval_bags_from_coverage",
    "build_An_quiver",
    "make_rad_jobs_for_An",

    # Enumeration - D_n
    "enumerate_Dn_bags_from_coverage_iter",
    "enumerate_Dn_bags_from_coverage",
    "make_rad_jobs_for_Dn",

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

    # Analysis - Homomorphisms
    "hom_interval",
    "hom_interval_to_bag",
    "find_hom_basis",

    # Analysis - Poset construction
    "write_rank_poset_from_jobs",
    "build_dir_edges",
    "write_rank_poset_from_jobs_Dn",
    "report_local_minima_dn",

    # Analysis - Conjectures
    "read_ranks_csv_strict",
    "read_edges_csv_strict",
    "read_parsed_csv_strict",
    "check_conjectures",
    "check_conjectures_write",
    "select_generic_jobs_strict",
    "transitive_closure",
    "maximal_in_subset",

    # Analysis - Visualization
    "visualize_degeneracy_dag",

    # Utils
    "PathRec",
    "module_over_field",
]
