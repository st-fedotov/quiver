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

from .type_kronecker import (
    INFINITY,
    KroneckerBag,
    KroneckerIndec,
    KroneckerJob,
    KroneckerKind,
    KroneckerRegularMode,
    enumerate_kronecker_bags_from_coverage,
    enumerate_kronecker_bags_from_coverage_iter,
    enumerate_kronecker_indecs_bounded,
    form_kronecker_module_from_bag_explicit,
    kronecker_vertices_and_arrows,
    lift_kronecker_bag_to_characteristic_zero,
    make_kronecker_jobs,
    make_kronecker_quiver,
    rational_jordan_parameters,
    write_batch_rad_from_kronecker_jobs,
)

from .type_affine_ad import (
    AffineBag,
    AffineIndecRecord,
    AffineJob,
    enumerate_affine_ad_indecs_bounded,
    enumerate_affine_bags_from_records_iter,
    make_affine_ad_jobs,
    write_batch_rad_from_affine_jobs,
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
    # Kronecker / affine A1
    "INFINITY",
    "KroneckerBag",
    "KroneckerIndec",
    "KroneckerJob",
    "KroneckerKind",
    "KroneckerRegularMode",
    "enumerate_kronecker_bags_from_coverage",
    "enumerate_kronecker_bags_from_coverage_iter",
    "enumerate_kronecker_indecs_bounded",
    "form_kronecker_module_from_bag_explicit",
    "kronecker_vertices_and_arrows",
    "lift_kronecker_bag_to_characteristic_zero",
    "make_kronecker_jobs",
    "make_kronecker_quiver",
    "rational_jordan_parameters",
    "write_batch_rad_from_kronecker_jobs",
    # Affine A/D
    "AffineBag",
    "AffineIndecRecord",
    "AffineJob",
    "enumerate_affine_ad_indecs_bounded",
    "enumerate_affine_bags_from_records_iter",
    "make_affine_ad_jobs",
    "write_batch_rad_from_affine_jobs",
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
