"""
Batch processing subpackage for Macaulay2 computations.
"""

from .core import (
    write_batch_rad_from_triples,
)

from .parsers import (
    parse_quiver_jobs_rad,
    print_results_table_rad,
    collect_minimal_generators_markdown,
    parse_ring_block,
)

from .hilbert import (
    load_manifest,
    parse_ifullsat_gb,
    relpath,
)

__all__ = [
    # Core
    "write_batch_rad_from_triples",
    # Parsers
    "parse_quiver_jobs_rad",
    "print_results_table_rad",
    "collect_minimal_generators_markdown",
    "parse_ring_block",
    # Hilbert
    "load_manifest",
    "parse_ifullsat_gb",
    "collect_hilbert_results",
    "relpath",
]
