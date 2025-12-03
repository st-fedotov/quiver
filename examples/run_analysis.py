#!/usr/bin/env python3
"""
Quiver analysis pipeline runner.

This script runs the full analysis pipeline for a quiver defined in a YAML config file.
It performs radical computations, builds rank posets, checks conjectures, and computes
Hilbert functions.

Usage:
    python run_analysis.py config.yaml

Requirements:
    - quiver-representations library (pip install quiver-representations)
    - GNU parallel
    - Macaulay2 (M2)

See configs/ directory for example configuration files.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from quiver_representations import ComplexNumbers, Quiver
from quiver_representations.module import Module


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    if "quiver" not in config:
        raise ValueError("Config must have a 'quiver' section")
    if "type" not in config:
        raise ValueError("Config must have a 'type' field ('An' or 'Dn')")
    if config["type"] not in ("An", "Dn"):
        raise ValueError(f"'type' must be 'An' or 'Dn', got: {config['type']}")
    if "coverage" not in config:
        raise ValueError("Config must have a 'coverage' section")

    # Set defaults for runtime
    runtime_defaults = {
        "output_dir": "./results",
        "workers": 64,
        "hilbert_workers": 64,
        "r_max": 3,
        "gc_heap_size": "20G",
        "hom_prime": 107,
    }
    config["runtime"] = {**runtime_defaults, **config.get("runtime", {})}

    return config


def build_quiver(config: Dict[str, Any]) -> Quiver:
    """Build a Quiver instance from config."""
    qcfg = config["quiver"]

    name = qcfg.get("name", "CustomQuiver")
    vertices = qcfg["vertices"]
    arrows = qcfg["arrows"]

    quiver = Quiver(name)
    vertex_map: Dict[Any, int] = {}

    for vertex_label in vertices:
        vertex_map[vertex_label] = quiver.add_vertex(str(vertex_label))

    for arrow in arrows:
        src, dst, label = arrow[0], arrow[1], arrow[2]
        quiver.add_arrow(vertex_map[src], vertex_map[dst], str(label))

    return quiver


def parse_coverage(config: Dict[str, Any], n_vertices: int) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Parse coverage specification into np and ni dicts."""
    coverage = config["coverage"]

    def parse_mult(spec, name: str) -> Dict[int, int]:
        if isinstance(spec, int):
            # Uniform multiplicity
            return {i: spec for i in range(n_vertices)}
        elif isinstance(spec, dict):
            # Per-vertex multiplicity
            return {int(k): int(v) for k, v in spec.items()}
        else:
            raise ValueError(f"Invalid {name} specification: {spec}")

    np = parse_mult(coverage.get("projective", 1), "projective")
    ni = parse_mult(coverage.get("injective", 1), "injective")

    return np, ni


def get_dimension_vectors(Q: Quiver, F, np: Dict[int, int], ni: Dict[int, int]):
    """
    Compute dimension vectors for P + I with custom multiplicities.

    Returns:
        (p_dim, total_dim): dimension vectors for projective part and total module
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
        # No projectives - create zero module
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


def validate_quiver_structure(Q: Quiver, quiver_type: str) -> None:
    """
    Validate that the quiver structure matches the declared type.

    Vertices MUST be numbered 0, 1, 2, ..., n-1.

    For A_n: edges must form a chain 0 - 1 - 2 - ... - (n-1)
    For D_n: edges must form 0 - 2, 1 - 2, 2 - 3 - ... - (n-1)

    Raises ValueError if structure doesn't match.
    """
    vertices = sorted(Q.get_vertices())
    n = len(vertices)

    # Check vertices are 0, 1, ..., n-1
    if vertices != list(range(n)):
        raise ValueError(
            f"Vertices must be numbered 0, 1, ..., {n-1}. Got: {vertices}"
        )

    # Build undirected adjacency set
    edges = set()
    for arrow_id in Q.get_arrows():
        arrow = Q.arrows[arrow_id]
        src, dst = arrow["source"], arrow["target"]
        edges.add((min(src, dst), max(src, dst)))

    if quiver_type == "An":
        # Expected: 0-1, 1-2, 2-3, ..., (n-2)-(n-1)
        expected = {(i, i + 1) for i in range(n - 1)}
        if edges != expected:
            raise ValueError(
                f"A_n quiver must have edges 0-1, 1-2, ..., {n-2}-{n-1}. "
                f"Got: {sorted(edges)}"
            )

    elif quiver_type == "Dn":
        if n < 4:
            raise ValueError(f"D_n requires at least 4 vertices, got {n}")
        # Expected: 0-1, 1-2, ..., (n-4)-(n-3), (n-3)-(n-2), (n-3)-(n-1)
        # Branching at vertex n-3
        branch = n - 3
        expected = {(i, i + 1) for i in range(branch)}  # chain 0-1-...-branch
        expected.add((branch, n - 2))  # branch to n-2
        expected.add((branch, n - 1))  # branch to n-1
        if edges != expected:
            raise ValueError(
                f"D_n quiver must have chain 0-1-...-{branch} with branches to {n-2} and {n-1}. "
                f"Got: {sorted(edges)}"
            )


def get_script_path(script_name: str) -> Path:
    """Get path to a bundled shell script from the package."""
    try:
        # Python 3.9+
        files = importlib.resources.files("quiver_representations.scripts")
        script_file = files.joinpath(script_name)
        # Need to extract to a temp location if it's in a zip
        with importlib.resources.as_file(script_file) as path:
            return Path(path)
    except (AttributeError, TypeError):
        # Python 3.8 fallback
        import pkg_resources
        return Path(pkg_resources.resource_filename(
            "quiver_representations", f"scripts/{script_name}"
        ))


def copy_script_to_dir(script_name: str, dest_dir: Path) -> Path:
    """Copy a bundled shell script to destination directory."""
    try:
        files = importlib.resources.files("quiver_representations.scripts")
        script_file = files.joinpath(script_name)
        with importlib.resources.as_file(script_file) as src_path:
            dest_path = dest_dir / script_name
            shutil.copy(str(src_path), str(dest_path))
            return dest_path
    except (AttributeError, TypeError):
        import pkg_resources
        src_path = pkg_resources.resource_filename(
            "quiver_representations", f"scripts/{script_name}"
        )
        dest_path = dest_dir / script_name
        shutil.copy(src_path, str(dest_path))
        return dest_path


def create_archive_excluding_jobs(source_dir: Path, archive_path: Path) -> Path:
    """
    Create a zip archive of source_dir excluding batch computation directories.

    Excludes: batch_rad/ and batch_hilbert/ (intermediate computation files)
    Includes: All derived results (CSVs, posets, reports, visualizations)
    """
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in source_dir.rglob("*"):
            if "batch_rad" in item.parts or "batch_hilbert" in item.parts:
                continue
            if item == archive_path:
                continue
            if item.is_file():
                arcname = item.relative_to(source_dir.parent)
                zipf.write(item, arcname)

    return archive_path


def run_pipeline(config: Dict[str, Any]) -> None:
    """Run the full analysis pipeline."""
    print("=" * 80)
    print("QUIVER ANALYSIS PIPELINE")
    print("=" * 80)

    # Build quiver
    Q = build_quiver(config)
    F = ComplexNumbers()

    # Parse coverage
    n_vertices = len(Q.get_vertices())
    np, ni = parse_coverage(config, n_vertices)

    # Get dimension vectors
    p_dim, ambient_dim = get_dimension_vectors(Q, F, np, ni)

    # Get quiver type from config and validate structure
    quiver_type = config["type"]
    validate_quiver_structure(Q, quiver_type)

    # Runtime settings
    runtime = config["runtime"]
    output_dir = Path(runtime["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_name = config.get("name", f"{Q.name}_analysis")

    print(f"Quiver: {Q.name}")
    print(f"Type: {quiver_type}")
    print(f"Vertices: {n_vertices}")
    print(f"Coverage name: {coverage_name}")
    print(f"Ambient dimension (P+I): {ambient_dim}")
    print(f"Target dimension (P): {p_dim}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Workers: {runtime['workers']}")
    print(f"Hilbert workers: {runtime['hilbert_workers']}")
    print(f"R_max: {runtime['r_max']}")
    print(f"GC heap size: {runtime['gc_heap_size']}")
    if quiver_type == "Dn":
        print(f"Hom prime: {runtime['hom_prime']}")
    print()

    # Save resolved config
    config_out = output_dir / "config_resolved.yaml"
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Resolved config saved to: {config_out}")
    print()

    # Copy required scripts to current working directory (library expects them there)
    for script_name in ["run_all_parallel.sh", "run_all_parallel_hf.sh", "rank_poset_parallel.py"]:
        if not Path(script_name).exists():
            copy_script_to_dir(script_name, Path.cwd())
            print(f"Copied {script_name} to current directory")

    # Run appropriate pipeline
    if quiver_type == "An":
        from quiver_representations.analysis.coverage_pipeline import process_coverage

        coverage_dir = process_coverage(
            Q, F, coverage_name, ambient_dim, p_dim, output_dir,
            runtime["workers"],
            runtime["r_max"],
            runtime["hilbert_workers"],
            runtime["gc_heap_size"],
        )
    else:  # Dn
        from quiver_representations.analysis.coverage_pipeline_dn import process_coverage_dn

        coverage_dir = process_coverage_dn(
            Q, F, coverage_name, ambient_dim, p_dim, output_dir,
            runtime["workers"],
            runtime["r_max"],
            runtime["hilbert_workers"],
            runtime["gc_heap_size"],
            runtime["hom_prime"],
        )

    # Create archive
    print(f"\n{'=' * 80}")
    print("Creating archive (excluding batch directories)...")
    print(f"{'=' * 80}\n")

    archive_path = output_dir / "results.zip"
    create_archive_excluding_jobs(output_dir, archive_path)

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive created: {archive_path}")
    print(f"Archive size: {archive_size_mb:.2f} MB")
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("Error: Please provide a config file path")
        print("Usage: python run_analysis.py config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        config = load_config(config_path)
        run_pipeline(config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
