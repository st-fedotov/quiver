"""Finite-test Hom-profile order for Kronecker representations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..interval_modules.type_kronecker import (
    KroneckerIndec,
    KroneckerJob,
    form_kronecker_module_from_bag_explicit,
)
from ..module import Module
from .hom import find_hom_basis
from .poset_utils import _hasse_edges


def kronecker_hom_profile(module: Module, test_modules: Sequence[Module]) -> tuple[int, ...]:
    """Return ``(dim Hom(X, module))_X`` over the modules' finite field."""
    return tuple(len(find_hom_basis(test_module, module)) for test_module in test_modules)


def write_kronecker_hom_poset(
    jobs: Sequence[KroneckerJob],
    out_dir: str | Path,
    *,
    test_indecs: Iterable[KroneckerIndec] | None = None,
) -> Path:
    """Write the tested Hom-profile order on a finite Kronecker catalogue.

    If ``test_indecs`` is omitted, every indecomposable occurring in a job is
    used as a test module.  The output is intentionally labelled as a
    finite-test order: without a separate completeness argument for the test
    family, its relations and covers are not asserted to be the full finite-
    field or complex degeneration order.
    """
    if not jobs:
        raise ValueError("No Kronecker jobs provided.")

    quiver = jobs[0].finite_module.quiver
    field = jobs[0].finite_module.field
    for job in jobs:
        if job.finite_module.quiver is not quiver:
            raise ValueError("All Kronecker jobs must use the same quiver object.")
        if job.finite_module.field is not field:
            raise ValueError("All Kronecker jobs must use the same finite field object.")

    if test_indecs is None:
        test_indecs = sorted(
            {indecomposable for job in jobs for indecomposable in job.bag},
            key=lambda indecomposable: indecomposable.label(),
        )
    else:
        test_indecs = tuple(test_indecs)
    if not test_indecs:
        raise ValueError("At least one indecomposable Hom test is required.")

    test_modules = [
        form_kronecker_module_from_bag_explicit(quiver, field, [indecomposable])
        for indecomposable in test_indecs
    ]
    profiles = [kronecker_hom_profile(job.finite_module, test_modules) for job in jobs]
    if len(set(profiles)) != len(profiles):
        raise ValueError(
            "The selected Hom tests do not separate the catalogue; provide more test_indecs."
        )

    ids = list(range(len(jobs)))
    leq_adj = defaultdict(list)
    for source in ids:
        source_profile = np.asarray(profiles[source], dtype=int)
        for target in ids:
            if source == target:
                continue
            if np.all(source_profile <= np.asarray(profiles[target], dtype=int)):
                leq_adj[source].append(target)
    edges = _hasse_edges(ids, leq_adj)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "dst"])
        writer.writerows(edges)
    with (output / "hom_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "module", *[item.label() for item in test_indecs]])
        for identifier, (job, profile) in enumerate(zip(jobs, profiles)):
            writer.writerow(
                [identifier, " + ".join(item.label() for item in job.bag) or "0", *profile]
            )
    metadata = {
        "field": f"GF({field.order})",
        "relation": "finite_field_tested_hom_profile_order",
        "cover_scope": "finite_catalogue",
        "complete_hom_criterion_certified": False,
        "edge_orientation": "src degenerates to dst",
        "test_modules": [item.label() for item in test_indecs],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output
