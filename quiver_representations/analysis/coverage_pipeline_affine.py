"""End-to-end coverage preparation and execution for acyclic affine A/D quivers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ..batch.parsers import parse_quiver_jobs_rad
from ..field import FiniteField
from ..interval_modules.type_affine_ad import (
    make_affine_ad_jobs,
    write_batch_rad_from_affine_jobs,
)
from .poset_affine import write_affine_hom_poset


def prepare_affine_ad_coverage(
    quiver,
    coverage_name,
    ambient_dim,
    target_dim,
    output_dir,
    *,
    hom_prime=107,
    regular_mode="rational_jordan",
    random_seed=0,
    max_attempts=50000,
    irred_dims_only=False,
):
    """Create the finite Hom catalogue and the characteristic-zero RAD batch."""
    coverage_dir = Path(output_dir) / coverage_name
    coverage_dir.mkdir(parents=True, exist_ok=True)
    finite_field = FiniteField(hom_prime)
    records, jobs = make_affine_ad_jobs(
        quiver,
        finite_field,
        ambient_dim,
        target_dim,
        regular_mode=regular_mode,
        random_seed=random_seed,
        max_attempts=max_attempts,
    )
    batch_dir = write_batch_rad_from_affine_jobs(
        jobs,
        batch_root=str(coverage_dir),
        run_id="batch_rad",
        max_path_len_full_default=10,
        irred_dims_only=irred_dims_only,
    )
    poset_dir = write_affine_hom_poset(
        jobs,
        records,
        coverage_dir / "rank_poset",
    )
    return {
        "coverage_dir": coverage_dir,
        "batch_dir": Path(batch_dir),
        "poset_dir": Path(poset_dir),
        "records": records,
        "jobs": jobs,
    }


def process_coverage_affine_ad(
    quiver,
    characteristic_zero_field,
    coverage_name,
    ambient_dim,
    target_dim,
    output_dir,
    workers,
    r_max,
    hilbert_workers,
    gc_heap_size,
    *,
    hom_prime=107,
    regular_mode="rational_jordan",
    random_seed=0,
    max_attempts=50000,
    check_hilbert=False,
    irred_dims_only=False,
):
    """Run the existing QQ RAD pipeline for one affine A/D coverage."""
    if irred_dims_only and check_hilbert:
        raise ValueError("Hilbert computations require the full diagnostic emitter; set check_hilbert=False for irred_dims_only.")
    if getattr(characteristic_zero_field, "characteristic", None) != 0:
        raise ValueError("Affine Grassmannian computations must use characteristic zero.")
    if check_hilbert:
        raise NotImplementedError("Affine Hilbert postprocessing is not wired yet.")

    prepared = prepare_affine_ad_coverage(
        quiver,
        coverage_name,
        ambient_dim,
        target_dim,
        output_dir,
        hom_prime=hom_prime,
        regular_mode=regular_mode,
        random_seed=random_seed,
        max_attempts=max_attempts,
        irred_dims_only=irred_dims_only,
    )
    coverage_dir = prepared["coverage_dir"]
    batch_dir = prepared["batch_dir"]

    launcher = Path("run_all_parallel.sh")
    if launcher.exists():
        shutil.copy(str(launcher), str(batch_dir / launcher.name))
    else:
        raise FileNotFoundError("run_all_parallel.sh is required to execute the RAD batch.")

    environment = os.environ.copy()
    environment["NUM_WORKERS"] = str(workers)
    environment["GC_INITIAL_HEAP_SIZE"] = gc_heap_size
    result = subprocess.run(
        ["bash", "run_all_parallel.sh"],
        cwd=str(batch_dir),
        env=environment,
    )
    if result.returncode != 0:
        print(
            f"[{coverage_name}] WARNING: affine RAD launcher returned {result.returncode}."
        )

    parsed_path = coverage_dir / "parsed.csv"
    parse_quiver_jobs_rad(path=str(batch_dir), write_csv_path=str(parsed_path))
    return coverage_dir
