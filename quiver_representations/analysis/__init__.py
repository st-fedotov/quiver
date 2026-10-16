"""
Analysis subpackage for quiver representation analysis.

Provides tools for:
- Homomorphism computations
- Rank poset construction (A_n and D_n)
- Conjecture checking
- Degeneration DAG visualization
"""

from .hom import (
    hom_interval,
    hom_interval_to_bag,
    find_hom_basis,
)

from .poset_an import (
    write_rank_poset_from_jobs,
    build_dir_edges,
)

from .poset_dn import (
    write_rank_poset_from_jobs_Dn,
    report_local_minima_dn,
)

from .poset_kronecker import (
    kronecker_hom_profile,
    write_kronecker_hom_poset,
)

from .poset_affine import (
    write_affine_hom_poset,
)

from .conjectures import (
    read_ranks_csv_strict,
    read_edges_csv_strict,
    read_parsed_csv_strict,
    check_conjectures,
    check_conjectures_write,
    select_generic_jobs_strict,
    transitive_closure,
    maximal_in_subset,
)

from .visualization import (
    visualize_degeneracy_dag,
)

from .coverage_pipeline import (
    process_coverage,
)

from .coverage_pipeline_dn import (
    process_coverage_dn,
)

from .coverage_pipeline_affine import (
    prepare_affine_ad_coverage,
    process_coverage_affine_ad,
)

__all__ = [
    "hom_interval",
    "hom_interval_to_bag",
    "find_hom_basis",
    "write_rank_poset_from_jobs",
    "build_dir_edges",
    "write_rank_poset_from_jobs_Dn",
    "report_local_minima_dn",
    "kronecker_hom_profile",
    "write_kronecker_hom_poset",
    "write_affine_hom_poset",
    "read_ranks_csv_strict",
    "read_edges_csv_strict",
    "read_parsed_csv_strict",
    "check_conjectures",
    "check_conjectures_write",
    "select_generic_jobs_strict",
    "transitive_closure",
    "maximal_in_subset",
    "visualize_degeneracy_dag",
    "process_coverage",
    "process_coverage_dn",
    "prepare_affine_ad_coverage",
    "process_coverage_affine_ad",
]
