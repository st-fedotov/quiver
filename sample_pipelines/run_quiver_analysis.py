#!/usr/bin/env python3
"""
Quiver analysis pipeline for P ⊕ I modules with varying injective components.

For a given quiver, generates jobs for M = P ⊕ I where P is the sum of all
projectives and I is the sum of all injectives except one (varying over all
possible missing injectives). Runs radical computations, builds rank posets,
and checks conjectures for each coverage vector.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

from quiver_representations import ComplexNumbers
from quiver_representations.module import Module
from quiver_representations.analysis.coverage_pipeline import process_coverage
from sample_pipelines.config import QuiverConfig, build_quiver


def get_p_plus_i_dim_with_mult(Q, F, np: dict, ni: dict):
    """
    Compute dimension vectors for P ⊕ I with custom multiplicities.

    Args:
        Q: Quiver
        F: Field
        np: dict {vertex_id: multiplicity} for projectives
        ni: dict {vertex_id: multiplicity} for injectives

    Returns:
        (p_dim, total_dim): dimension vectors
    """
    n_vertices = len(Q.get_vertices())

    # Build sum of projectives with multiplicities
    PI = None
    for i in range(n_vertices):
        mult = np.get(i, 0)
        for _ in range(mult):
            P_i = Module.projective(Q, F, i)[0]
            if PI is None:
                PI = P_i
            else:
                PI = Module.direct_sum(PI, P_i)[0]

    if PI is None:
        # No projectives - start with zero module
        PI = Module.projective(Q, F, 0)[0]
        for v in Q.get_vertices():
            PI.spaces[v] = 0

    p_dim = PI.get_dimension_vector()

    # Add injectives with multiplicities
    for i in range(n_vertices):
        mult = ni.get(i, 0)
        for _ in range(mult):
            I_i = Module.injective(Q, F, i)[0]
            PI = Module.direct_sum(PI, I_i)[0]

    return p_dim, PI.get_dimension_vector()


DEFAULT_QUIVER_CONFIG = QuiverConfig(
    quiver_name="A3",
    vertices=["v0", "v1", "v2"],
    arrows=[
        ("v0", "v1", "a0"),
        ("v1", "v2", "a1"),
    ],
)


def create_archive_excluding_jobs(source_dir, archive_path):
    """
    Create a zip archive of source_dir excluding batch computation directories.

    Excludes: batch_rad/ and batch_hilbert/ (all intermediate computation files)
    Includes: All derived results (CSVs, posets, reports, visualizations)

    Args:
        source_dir: Path to directory to archive
        archive_path: Path for output zip file
    """
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)

    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in source_dir.rglob('*'):
            # Skip if this is inside a batch directory
            if 'batch_rad' in item.parts or 'batch_hilbert' in item.parts:
                continue

            # Skip if this is the archive itself
            if item == archive_path:
                continue

            if item.is_file():
                arcname = item.relative_to(source_dir.parent)
                zipf.write(item, arcname)

    return archive_path


def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    
    np = {0:1, 1:3, 2:1}
    ni = {0:1, 1:2, 2:1}
    p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)
    coverages = [("coverage_extra_2P1_I1", np, ni, ambient_dim, p_dim)]
    return coverages

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    coverages = []
    for extra_proj in range(n_vertices):
        # All projectives except one
        np = {i: 1 for i in range(n_vertices)}
        np[extra_proj] = 2

        for extra_inj in [0,2]:
            ni = {i: 1 for i in range(n_vertices)}
            ni[extra_inj] = 3

            p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

            name = f"coverage_extra_P{extra_proj}_2I{extra_inj}"
            coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    ni = {i: 1 for i in range(n_vertices)}
    np = {i: 1 for i in range(n_vertices)}
    coverages = []

    p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

    name = f"coverage_generic"
    coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    coverages = []
    for extra_proj in range(n_vertices):
        # All projectives except one
        np = {i: 1 for i in range(n_vertices)}
        np[extra_proj] = 2

        for extra_inj in range(n_vertices):
            ni = {i: 1 for i in range(n_vertices)}
            ni[extra_inj] = 2

            p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

            name = f"coverage_extra_P{extra_proj}_I{extra_inj}"
            coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    # All injectives have multiplicity 1
    ni = {i: 1 for i in range(n_vertices)}

    coverages = []
    for missing_proj in range(n_vertices):
        # All projectives except one
        np = {i: 1 for i in range(n_vertices)}
        np[missing_proj] = 0

        p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

        name = f"coverage_missing_P{missing_proj}"
        coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    # All projectives have multiplicity 1
    np = {i: 1 for i in range(n_vertices)}

    coverages = []
    for missing_inj in range(n_vertices):
        # All injectives except one
        ni = {i: 1 for i in range(n_vertices)}
        ni[missing_inj] = 0

        p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

        name = f"coverage_missing_I{missing_inj}"
        coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    # All injectives have multiplicity 1
    ni = {i: 1 for i in range(n_vertices)}

    coverages = []
    for extra_proj in range(n_vertices):
        # All projectives with one extra
        np = {i: 1 for i in range(n_vertices)}
        np[extra_proj] = 2

        p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

        name = f"coverage_extra_P{extra_proj}"
        coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''

'''
def generate_coverage_specs(Q, F):
    """
    Generate coverage specifications for all variants of P ⊕ I.

    Returns:
        list of (name, np, ni, target_dim) tuples
    """
    n_vertices = len(Q.get_vertices())

    # All projectives have multiplicity 1
    np = {i: 1 for i in range(n_vertices)}

    coverages = []
    for extra_inj in range(n_vertices):
        # All injectives with one extra
        ni = {i: 1 for i in range(n_vertices)}
        ni[extra_inj] = 2

        p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, np, ni)

        name = f"coverage_extra_I{extra_inj}"
        coverages.append((name, np, ni, ambient_dim, p_dim))

    return coverages
'''


def main():
    parser = argparse.ArgumentParser(
        description="Quiver analysis pipeline for P ⊕ I modules."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for all results (default: results)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of workers for parallel rank poset computation (default: 64)"
    )
    parser.add_argument(
        "--r-max",
        type=int,
        default=3,
        help="Maximum degree for Hilbert function computation (default: 3)"
    )
    parser.add_argument(
        "--hilbert-workers",
        type=int,
        default=64,
        help="Number of workers for parallel Hilbert computation (default: 64)"
    )
    parser.add_argument(
        "--gc-heap-size",
        type=str,
        default="20G",
        help="GC initial heap size for Macaulay2 (default: 20G)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("QUIVER ANALYSIS PIPELINE")
    print("="*80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Workers (rank poset): {args.workers}")
    print(f"Workers (Hilbert): {args.hilbert_workers}")
    print(f"R_max (Hilbert): {args.r_max}")
    print(f"GC heap size: {args.gc_heap_size}")
    print()

    # Create quiver and field
    Q = build_quiver(DEFAULT_QUIVER_CONFIG)
    F = ComplexNumbers()

    print(f"Quiver: {Q.name}")
    print(f"Vertices: {len(Q.get_vertices())}")
    print()

    # Generate coverage specifications
    coverages = generate_coverage_specs(Q, F)
    print(f"Generated {len(coverages)} coverage vectors:")
    for name, _, _, ambient_dim, p_dim in coverages:
        print(f"  - {name}: ambient={ambient_dim}, target={p_dim}")
    print()

    # Process each coverage
    coverage_dirs = []
    for coverage_name, np, ni, ambient_dim, target_dim in coverages:
        try:
            coverage_dir = process_coverage(
                Q, F, coverage_name, ambient_dim, target_dim, output_dir,
                args.workers, args.r_max, args.hilbert_workers, args.gc_heap_size
            )
            coverage_dirs.append(coverage_dir)
        except Exception as e:
            print(f"ERROR processing {coverage_name}: {e}")
            import traceback
            traceback.print_exc()

    # Archive all results (excluding jobs folders)
    print(f"\n{'='*80}")
    print("Creating archive (excluding jobs folders)...")
    print(f"{'='*80}\n")

    archive_path = output_dir / "results.zip"
    create_archive_excluding_jobs(output_dir, archive_path)

    # Get archive size
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive created: {archive_path}")
    print(f"Archive size: {archive_size_mb:.2f} MB")
    print()
    print("="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
