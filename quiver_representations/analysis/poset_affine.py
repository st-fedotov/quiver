"""Finite-test Hom-profile order for bounded affine A/D catalogues."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..interval_modules.type_affine_ad import AffineIndecRecord, AffineJob
from .hom import find_hom_basis
from .poset_utils import _hasse_edges


def write_affine_hom_poset(
    jobs: Sequence[AffineJob],
    records: Sequence[AffineIndecRecord],
    out_dir: str | Path,
) -> Path:
    """Write the Hom-profile order tested by the supplied bounded catalogue."""
    if not jobs:
        raise ValueError("No affine jobs provided.")
    if not records:
        raise ValueError("No affine indecomposable test modules provided.")
    quiver = jobs[0].finite_module.quiver
    field = jobs[0].finite_module.field
    if any(job.finite_module.quiver is not quiver for job in jobs):
        raise ValueError("All affine jobs must use the same quiver object.")
    if any(job.finite_module.field is not field for job in jobs):
        raise ValueError("All affine jobs must use the same finite field object.")

    profiles = [
        tuple(
            len(find_hom_basis(record.finite_module, job.finite_module))
            for record in records
        )
        for job in jobs
    ]
    if len(set(profiles)) != len(profiles):
        raise ValueError("The bounded affine Hom tests do not separate the job catalogue.")

    identifiers = list(range(len(jobs)))
    leq_adj = defaultdict(list)
    arrays = [np.asarray(profile, dtype=int) for profile in profiles]
    for source in identifiers:
        for target in identifiers:
            if source != target and np.all(arrays[source] <= arrays[target]):
                leq_adj[source].append(target)
    edges = _hasse_edges(identifiers, leq_adj)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "dst"])
        writer.writerows(edges)
    with (output / "hom_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "module", *[record.label for record in records]])
        for identifier, (job, profile) in enumerate(zip(jobs, profiles)):
            writer.writerow(
                [identifier, " + ".join(record.label for record in job.bag) or "0", *profile]
            )
    metadata = {
        "field": f"GF({field.order})",
        "relation": "finite_field_tested_hom_profile_order",
        "cover_scope": "bounded_rational_jordan_catalogue",
        "complete_hom_criterion_certified": False,
        "edge_orientation": "src degenerates to dst",
        "test_modules": [record.label for record in records],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output
