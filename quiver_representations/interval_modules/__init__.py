"""
Interval module construction for various quiver types.
"""

from .type_a import (
    build_interval_explicit,
    form_An_module_from_intervals_explicit,
    write_batch_rad_from_interval_bags_explicit,
)

from .type_d import (
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

from .enumeration import (
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

__all__ = [
    # Type A
    "build_interval_explicit",
    "form_An_module_from_intervals_explicit",
    "write_batch_rad_from_interval_bags_explicit",
    # Type D
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
    # A_n enumeration
    "enumerate_interval_bags_An",
    "enumerate_interval_bags_from_coverage_iter",
    "enumerate_interval_bags_from_coverage",
    "build_An_quiver",
    "make_rad_jobs_for_An",
    # D_n enumeration
    "enumerate_Dn_bags_from_coverage_iter",
    "enumerate_Dn_bags_from_coverage",
    "make_rad_jobs_for_Dn",
]
