#!/usr/bin/env python3
"""
Quiver analysis pipeline for D_n quivers with P + I modules.

For a given D_n quiver, generates jobs for M = P + I where P is the sum of all
projectives and I is the sum of all injectives except one (varying over all
possible missing injectives). Runs radical computations, builds rank posets,
and checks conjectures for each coverage vector.
"""

import argparse
import json
import zipfile
from pathlib import Path

from quiver_representations import ComplexNumbers
from quiver_representations.module import Module
from quiver_representations.analysis.coverage_pipeline_dn import process_coverage_dn
from sample_pipelines.config import (
    PipelineConfig,
    build_quiver,
    load_pipeline_config,
    make_dn_pipeline,
    pipeline_to_dict,
)


def get_p_plus_i_dim_with_mult(Q, F, np: dict, ni: dict):
    """
    Compute dimension vectors for P + I with custom multiplicities.

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

def generate_coverage_specs(Q, F, pipeline_cfg: PipelineConfig):
    """
    Build coverage tuples from declarative specs.

    Returns:
        list of (name, np, ni, ambient_dim, target_dim) tuples
    """

    coverages = []
    for spec in pipeline_cfg.coverages:
        p_dim, ambient_dim = get_p_plus_i_dim_with_mult(Q, F, spec.np, spec.ni)
        coverages.append((spec.name, spec.np, spec.ni, ambient_dim, p_dim))

    return coverages

def main():
    parser = argparse.ArgumentParser(
        description="Quiver analysis pipeline for D_n quivers with P + I modules."
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to a JSON pipeline config. If omitted, the built-in D_n default "
            "(n=4, uniform coverage, default runtime) is used."
        ),
    )
    args = parser.parse_args()

    print("=" * 80)
    print("D_n QUIVER ANALYSIS PIPELINE")
    print("=" * 80)

    if args.config:
        pipeline_cfg = load_pipeline_config(args.config)
    else:
        pipeline_cfg = make_dn_pipeline(4)

    print(
        f"Config source: {'file ' + args.config if args.config else 'built-in default'}"
    )
    print(f"Output directory: {Path(pipeline_cfg.runtime.output_dir).absolute()}")
    print(f"Workers (RAD): {pipeline_cfg.runtime.workers}")
    print(f"Workers (Hilbert): {pipeline_cfg.runtime.hilbert_workers}")
    print(f"R_max (Hilbert): {pipeline_cfg.runtime.r_max}")
    print(f"GC heap size: {pipeline_cfg.runtime.gc_heap_size}")
    print(f"Hom prime: {pipeline_cfg.runtime.hom_prime}")
    print()

    # Create quiver and field
    Q = build_quiver(pipeline_cfg.quiver)
    F = ComplexNumbers()

    cfg_dict = pipeline_to_dict(pipeline_cfg)
    print("Resolved pipeline configuration:")
    print(json.dumps(cfg_dict, indent=2))
    print()

    output_dir = Path(pipeline_cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Quiver: {Q.name}")
    print(f"Vertices: {len(Q.get_vertices())}")
    print()

    # Generate coverage specifications
    coverages = generate_coverage_specs(Q, F, pipeline_cfg)
    print(f"Generated {len(coverages)} coverage vectors:")
    for name, _, _, ambient_dim, p_dim in coverages:
        print(f"  - {name}: ambient={ambient_dim}, target={p_dim}")
    print()

    # Process each coverage
    coverage_dirs = []
    for coverage_name, np, ni, ambient_dim, target_dim in coverages:
        try:
            coverage_dir = process_coverage_dn(
                Q, F, coverage_name, ambient_dim, target_dim, output_dir,
                pipeline_cfg.runtime.workers,
                pipeline_cfg.runtime.r_max,
                pipeline_cfg.runtime.hilbert_workers,
                pipeline_cfg.runtime.gc_heap_size,
                pipeline_cfg.runtime.hom_prime,
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
