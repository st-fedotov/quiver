"""
Coverage analysis pipeline for D_n quiver representations.

Provides end-to-end analysis workflow for a single coverage vector,
including RAD computations, rank posets (via Hom matrices), visualizations,
and D_n-specific conjecture checking.
"""

import os
import json
import shutil
import subprocess
import csv
from pathlib import Path

from ..interval_modules import (
    write_batch_rad_from_dn_bags,
    make_rad_jobs_for_Dn,
)
from ..batch.parsers import parse_quiver_jobs_rad
from .poset_dn import write_rank_poset_from_jobs_Dn, report_local_minima_dn
from ..batch.hilbert import write_hilbert_batch, collect_hilbert_results


def process_coverage_dn(Q, F, coverage_name, ambient_dim, target_dim, output_dir,
                        workers, r_max, hilbert_workers, gc_heap_size, hom_prime=107):
    """
    Process a single D_n coverage vector: enumerate jobs, run RAD, build poset, check conjectures, compute Hilbert functions.

    Args:
        Q: D_n Quiver
        F: Field
        coverage_name: str, e.g., "coverage_missing_I0"
        ambient_dim: dimension vector for P ⊕ I (coverage)
        target_dim: dimension vector for P (target in Grassmannian)
        output_dir: base output directory
        workers: number of workers for parallel RAD computation
        r_max: maximum degree for Hilbert function computation
        hilbert_workers: number of workers for parallel Hilbert computation
        gc_heap_size: GC initial heap size for Macaulay2 (e.g., "20G")
        hom_prime: prime p for GF(p) in Hom computations (default: 107)

    Returns:
        coverage_dir: Path to the coverage directory
    """
    coverage_dir = output_dir / coverage_name
    coverage_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing {coverage_name} (D_n)")
    print(f"Ambient dimension (P⊕I): {ambient_dim}")
    print(f"Target dimension (P): {target_dim}")
    print(f"{'='*80}\n")

    # Step 1: Enumerate all D_n indecomposable bags for this coverage
    print(f"[{coverage_name}] Step 1: Enumerating D_n indecomposable bags...")
    n = len(Q.get_vertices())
    _, jobs = make_rad_jobs_for_Dn(n, Q, coverage=ambient_dim, target_dim=target_dim)
    print(f"[{coverage_name}] Enumerated {len(jobs)} D_n bags.")

    # Step 2: Write batch RAD jobs
    print(f"[{coverage_name}] Step 2: Writing batch RAD jobs...")
    batch_dir = write_batch_rad_from_dn_bags(
        jobs,
        field=F,
        max_path_len_full_default=10,
    )
    print(f"[{coverage_name}] Batch written to: {batch_dir}")

    # Move entire batch directory to coverage_dir/batch_rad/
    batch_path = Path(batch_dir)
    batch_rad_dir = coverage_dir / "batch_rad"
    if batch_rad_dir.exists():
        shutil.rmtree(batch_rad_dir)
    shutil.move(str(batch_path), str(batch_rad_dir))
    print(f"[{coverage_name}] Batch moved to: {batch_rad_dir}")

    # Step 3: Run run_all_parallel.sh
    print(f"[{coverage_name}] Step 3: Running run_all_parallel.sh...")
    # Copy run_all_parallel.sh to batch_rad directory
    main_script = Path("run_all_parallel.sh")
    if main_script.exists():
        shutil.copy(str(main_script), str(batch_rad_dir / "run_all_parallel.sh"))

    env = os.environ.copy()
    env["NUM_WORKERS"] = str(workers)
    env["GC_INITIAL_HEAP_SIZE"] = gc_heap_size

    result = subprocess.run(
        ["bash", "run_all_parallel.sh"],
        cwd=str(batch_rad_dir),
        env=env
    )
    if result.returncode != 0:
        print(f"[{coverage_name}] WARNING: run_all_parallel.sh failed with code {result.returncode}")
    else:
        print(f"[{coverage_name}] Radical computations completed.")

    # Step 4: Parse results
    print(f"[{coverage_name}] Step 4: Parsing results...")
    parsed_csv_path = coverage_dir / "parsed.csv"
    parsed = parse_quiver_jobs_rad(path=str(batch_rad_dir), write_csv_path=str(parsed_csv_path))
    print(f"[{coverage_name}] Parsed {len(parsed.get('results', []))} jobs.")

    # Step 5: Build rank poset using Hom-based approach for D_n
    print(f"[{coverage_name}] Step 5: Building rank poset (Hom-based)...")

    poset_out_dir = coverage_dir / "rank_poset"
    poset_out_dir.mkdir(exist_ok=True)

    try:
        write_rank_poset_from_jobs_Dn(jobs, str(poset_out_dir), p=hom_prime)
        print(f"[{coverage_name}] Rank poset built: {poset_out_dir}/edges.csv")
    except Exception as e:
        print(f"[{coverage_name}] WARNING: Rank poset construction failed: {e}")
        import traceback
        traceback.print_exc()

    # Determine generic_dim from parsed.csv
    if not parsed_csv_path.exists():
        raise FileNotFoundError(f"parsed.csv not found at {parsed_csv_path}")

    with open(parsed_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        min_dim = float('inf')
        for row in reader:
            irred_dims_str = row.get('irred_dims', '').strip()
            if irred_dims_str:
                dims = [int(x) for x in irred_dims_str.split()]
                if dims:
                    min_dim = min(min_dim, min(dims))

        if min_dim == float('inf'):
            raise ValueError(f"Could not determine generic_dim from {parsed_csv_path}: no valid irred_dims found")

        generic_dim = min_dim

    # Step 5b: Visualization - skip for now (D_n visualization not yet implemented)
    print(f"[{coverage_name}] Step 5b: Visualization skipped (D_n visualization TBD)")

    # Step 6: Check D_n conjectures (local minima analysis)
    print(f"[{coverage_name}] Step 6: Checking D_n conjectures (local minima)...")

    reports_dir = coverage_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    try:
        report_local_minima_dn(
            generic_dim=generic_dim,
            rank_dir=str(poset_out_dir),
            parsed_csv=str(parsed_csv_path),
            out_dir=str(reports_dir),
        )
        print(f"[{coverage_name}] D_n conjectures checked. Results in {reports_dir}")
    except Exception as e:
        print(f"[{coverage_name}] WARNING: D_n conjecture checking failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 7: Compute Hilbert functions
    print(f"[{coverage_name}] Step 7: Computing Hilbert functions...")

    # Write Hilbert batch jobs to a subdirectory within coverage_dir
    hilbert_batch_dir = coverage_dir / "batch_hilbert"
    batch_rad_dir = coverage_dir / "batch_rad"
    try:
        write_hilbert_batch(source=str(batch_rad_dir), dest=str(hilbert_batch_dir), r_max=r_max)
        print(f"[{coverage_name}] Hilbert batch jobs written to {hilbert_batch_dir}.")
    except Exception as e:
        print(f"[{coverage_name}] ERROR: Failed to write Hilbert batch: {e}")
        raise

    # Copy run_all_parallel_hf.sh to hilbert batch directory
    main_hf_script = Path("run_all_parallel_hf.sh")
    if main_hf_script.exists():
        shutil.copy(str(main_hf_script), str(hilbert_batch_dir / "run_all_parallel_hf.sh"))

    # Set NUM_WORKERS environment variable and run the script
    env = os.environ.copy()
    env["NUM_WORKERS"] = str(hilbert_workers)
    env["GC_INITIAL_HEAP_SIZE"] = gc_heap_size

    result = subprocess.run(
        ["bash", "run_all_parallel_hf.sh"],
        cwd=str(hilbert_batch_dir),
        env=env
    )
    # Note: parallel returns non-zero if ANY job fails, which is acceptable
    # Only fail if the script itself failed to run (e.g., no joblist created)
    joblist_file = hilbert_batch_dir / "hilbert_joblist.txt"
    if result.returncode != 0:
        if not joblist_file.exists():
            print(f"[{coverage_name}] ERROR: Script failed completely (no joblist created)")
            raise RuntimeError(f"Hilbert script failed with return code {result.returncode}")
        else:
            print(f"[{coverage_name}] WARNING: Some individual Hilbert jobs failed (exit code {result.returncode})")
    print(f"[{coverage_name}] Hilbert computations completed.")

    # Collect Hilbert results into consolidated CSV
    print(f"[{coverage_name}] Collecting Hilbert results...")
    hilbert_consolidated_path = coverage_dir / "hilbert_results.csv"

    try:
        collect_hilbert_results(
            archive_path=str(hilbert_batch_dir),
            output_csv=str(hilbert_consolidated_path)
        )
        print(f"[{coverage_name}] Hilbert results collected to {hilbert_consolidated_path}")
    except Exception as e:
        print(f"[{coverage_name}] WARNING: Failed to collect Hilbert results: {e}")

    # Check Hilbert sequence identity hypothesis
    print(f"[{coverage_name}] Checking Hilbert sequence identity hypothesis...")

    try:
        # Read parsed.csv to identify jobs where ALL irred components have dimension generic_dim
        target_jobs = set()  # task_ids (as integers)
        with open(parsed_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                irred_dims_str = row.get('irred_dims', '').strip()
                if irred_dims_str:
                    dims = [int(x) for x in irred_dims_str.split()]
                    # Check if all components have dimension generic_dim
                    if dims and all(d == generic_dim for d in dims):
                        target_jobs.add(int(row['task']))

        # Read Hilbert sequences from hf.csv files in the batch_hilbert directory
        hilbert_jobs_dir = hilbert_batch_dir / "jobs"
        hilbert_sequences = {}

        for job_dir in hilbert_jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            # Extract original job ID from folder name (format: "000__hf")
            job_id_str = job_dir.name.split("__")[0]
            job_id = int(job_id_str)  # Convert to int to match task IDs

            if job_id not in target_jobs:
                continue

            hf_csv = job_dir / "hf.csv"
            if not hf_csv.exists():
                print(f"[{coverage_name}] WARNING: Missing {hf_csv}")
                continue

            # Read all HF values into a sequence (sorted by degree)
            hf_values = []
            with open(hf_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hf_values.append(int(row['hf']))

            hilbert_sequences[job_id] = tuple(hf_values)

        # Check if all sequences are identical
        unique_sequences = set(hilbert_sequences.values())
        all_identical = (len(unique_sequences) <= 1)

        # Write conjecture result
        conj3_path = reports_dir / "conj3_hilbert.csv"
        with open(conj3_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hypothesis', 'result', 'num_jobs', 'unique_sequences'])
            writer.writerow([
                'All modules with irred_dims=[generic_dim] have identical Hilbert sequences',
                'PASS' if all_identical else 'FAIL',
                len(target_jobs),
                len(unique_sequences)
            ])

        print(f"[{coverage_name}] Hilbert hypothesis check: {'PASS' if all_identical else 'FAIL'}")
        print(f"[{coverage_name}] Target jobs with irred_dims=[{generic_dim}]: {len(target_jobs)}")
        print(f"[{coverage_name}] Unique Hilbert sequences found: {len(unique_sequences)}")
        print(f"[{coverage_name}] Results written to {conj3_path}")

    except Exception as e:
        print(f"[{coverage_name}] WARNING: Hilbert hypothesis check failed: {e}")

    print(f"[{coverage_name}] Processing complete.\n")
    return coverage_dir
