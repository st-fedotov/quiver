"""
Core batch job infrastructure for Macaulay2 computations.
"""

import os
import re
import math
import shutil
import datetime
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Optional, Dict, Any, List
from tqdm import tqdm

from ..grassmannians import QuiverGrassmannian


def write_batch_rad_from_triples(
    triples,                                   # iterable of (Q, M, dim) or (Q, M, dim, max_len_full)
    *,
    batch_root: str = "batch_runs_rad",
    run_id: str | None = None,
    max_path_len_full_default: int = 3,
    script_name: str = "rad.m2",
    out_stdout: str = "rad_out.txt",
    out_stderr: str = "rad_err.txt",
    docker_image: str = "m2-ppa",
    msys_no_pathconv: bool = True,
    overwrite_outputs: bool = False,           # kept for compatibility; not used in launcher
    vertex_order: list[int] | None = None,
    prefix: str = "p_"
) -> Path:
    """
    For each (Q, M, dim[, max_len_full]) triple:
      - builds QuiverGrassmannian(Q, M, dim)
      - emits rad.m2 via to_macaulay2_compare_len1_vs_all (stdout only; GB printed)
      - writes manifest.yaml including dim as a YAML mapping
      - writes run_all.sh (printed at the end)
      - creates a ZIP archive of the entire run directory and prints its path
    """
    triples = list(triples)
    if not triples:
        raise ValueError("No jobs provided.")

    # Batch folders
    if run_id is None:
        run_id = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(batch_root) / run_id
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Zero-pad width for job ids
    width = max(3, int(math.log10(max(len(triples) - 1, 1))) + 1)

    # Collect manifest job entries
    job_entries: list[str] = []

    for idx, item in tqdm(enumerate(triples)):
        if len(item) == 3:
            Q, M, dim = item
            mplf = max_path_len_full_default
        elif len(item) == 4:
            Q, M, dim, mplf = item
            mplf = mplf if mplf is not None else max_path_len_full_default
        else:
            raise ValueError("Each entry must be (Q, M, dim) or (Q, M, dim, max_len_full).")

        # Names (strict: only .name)
        qname = Q.name.strip()
        mname = M.name.strip()

        # dim is a dict {int:int}; keep as-is (just sort for stable serialization)
        dim_map = dict(sorted(dim.items()))
        dim_str = "-".join(str(dim_map[k]) for k in dim_map.keys())

        M_dim_map = dict(sorted(M.get_dimension_vector().items()))

        # Job folder name (ASCII-safe)
        label_base = f"{qname}_dim_{dim_str}"
        label = re.sub(r"[^A-Za-z0-9.\-]+", "-", label_base.replace(" ", "-")) or "job"

        job_id  = str(idx).zfill(width)
        job_dir = jobs_dir / f"{job_id}_{label}"
        job_dir.mkdir(parents=True, exist_ok=True)

        # Emit Macaulay2 script for this job
        G = QuiverGrassmannian(M, dim)  # will fail naturally if dim is wrong
        # G.to_macaulay2_component_counts(
        G.to_macaulay2_saturated(
            max_path_len_full=mplf,
            vertex_order=vertex_order,
            prefix=prefix,
            script_path=str(job_dir / script_name),
        )

        # Append manifest job entry
        job_entries.append(dedent(f'''\
            - id: "{job_id}"
              quiver: "{qname}"
              module: "{mname}"
              target_dim: {{ {", ".join(f"{k}: {v}" for k, v in dim_map.items())} }}
              ambient_dim: {{ {", ".join(f"{k}: {v}" for k, v in M_dim_map.items())} }}
        '''))

    # Write manifest.yaml (compact & human-readable)
    manifest_text = dedent(f'''\
        batch:
          root: {batch_root}
          run_id: "{run_id}"
        jobs:
    ''') + "".join(job_entries)
    (run_dir / "manifest.yaml").write_text(manifest_text, encoding="utf-8")

    # Launcher script (serial), no skip block
    env_prefix = "MSYS_NO_PATHCONV=1 " if msys_no_pathconv else ""
    launcher_sh = dedent(f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        shopt -s nullglob

        ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"

        for d in "$ROOT/jobs"/*/; do
          echo "==> Running $(basename "$d")"
          {env_prefix}docker run -i --rm \\
            -v "$ROOT":/home/m2/work \\
            -w "/home/m2/work/jobs/$(basename "$d")" \\
            {docker_image} \\
            < "$d/{script_name}" > "$d/{out_stdout}" 2> "$d/{out_stderr}" || echo "    FAILED"
        done

        echo "All jobs attempted."
    """)
    launcher_path = run_dir / "run_all.sh"
    launcher_path.write_text(launcher_sh, encoding="utf-8")
    os.chmod(launcher_path, 0o755)

    # Print the launcher so you can review/copy
    print(launcher_sh)

    # Zip the entire run directory next to it and print the archive path
    # Result: batch_runs_rad/<run_id>.zip containing the whole run folder
    archive_base = str(run_dir)               # shutil will add .zip
    shutil.make_archive(archive_base, "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
    zip_path = run_dir.with_suffix(".zip")
    print(f"Created archive: {zip_path}")

    return run_dir


def _sanitize_label(s: str) -> str:
    """
    Sanitize a string for use as a job label or filename.

    Args:
        s: String to sanitize

    Returns:
        Sanitized string safe for filenames
    """
    s = s.strip().replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", s) or "job"

def _name_of(obj) -> str:
    """
    Return obj.name as a non-empty stripped string.
    Do not guess alternative attributes.
    """
    if not hasattr(obj, "name"):
        raise AttributeError(f"Expected {obj!r} to have a .name attribute.")
    val = obj.name
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f".name must be a non-empty string, got {val!r}.")
    return val.strip()
