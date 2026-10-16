# Acyclic affine pipeline

The affine code keeps two coefficient fields separate on purpose.

- Finite-field modules are used to enumerate a finite catalogue and compute
  Hom dimensions exactly.
- The corresponding characteristic-zero modules are used to generate the
  quiver-Grassmannian Macaulay2 jobs. The existing Macaulay2 emitter continues
  to use `QQ`.

Thus a finite-field graph is evidence for the complex degeneration landscape,
not a claim that its Hasse edges are automatically covers in the full complex
orbit order.

## Kronecker example

```python
from quiver_representations import (
    FiniteField,
    make_kronecker_jobs,
    make_kronecker_quiver,
    write_batch_rad_from_kronecker_jobs,
    write_kronecker_hom_poset,
)

Q = make_kronecker_quiver()
F = FiniteField(107)
coverage = {0: 2, 1: 2}
target = {0: 1, 1: 1}

jobs = make_kronecker_jobs(
    Q,
    F,
    coverage,
    target,
    regular_mode="rational_jordan",
)

# Exact Hom dimensions over GF(107) for the displayed finite test family.
# The output metadata does not claim that this family certifies the full
# degeneration order, and labels covers as covers only in this catalogue.
write_kronecker_hom_poset(jobs, "kronecker_rank_poset")

# Generates the usual QQ Macaulay2 scripts from each job.complex_module.
# This does not run Macaulay2.
write_batch_rad_from_kronecker_jobs(
    jobs,
    batch_root="kronecker_rad",
)
```

In `rational_jordan` mode, the regular indecomposables are the Jordan blocks
at the points of `P^1(F_p)`: one for each scalar in `F_p` and one at infinity.
Their matrices have canonical integral lifts to characteristic zero.

The alternative mode

```python
make_kronecker_jobs(
    Q,
    F,
    coverage,
    target,
    regular_mode="closed_points",
)
```

currently raises `NotImplementedError`. It is reserved for companion blocks
of arbitrary irreducible polynomials over the finite field. Such a block does
not lift to one complex indecomposable: over an algebraic closure it splits
into the packet of regular representations at the roots of that polynomial.
That packet-valued lift needs a separate design before this mode can safely
feed the characteristic-zero geometry pipeline.

## Affine A and D

Every acyclic orientation of affine A with at least three vertices and every
acyclic orientation of affine D is supported by the bounded catalogue API:

```python
from quiver_representations import (
    FiniteField,
    Quiver,
    make_affine_ad_jobs,
    write_affine_hom_poset,
    write_batch_rad_from_affine_jobs,
)

# A mixed acyclic orientation of affine D5. Vertices 0 and 1 are the two
# branching vertices.
Q = Quiver("affine_D5")
for _ in range(6):
    Q.add_vertex()
for source, target in [(2, 0), (0, 3), (0, 1), (4, 1), (1, 5)]:
    Q.add_arrow(source, target)

F = FiniteField(107)
coverage = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}  # delta
target = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0}

records, jobs = make_affine_ad_jobs(
    Q,
    F,
    coverage,
    target,
    regular_mode="rational_jordan",
    random_seed=0,
)

write_affine_hom_poset(jobs, records, "affine_D5_rank_poset")
write_batch_rad_from_affine_jobs(jobs, batch_root="affine_D5_rad")
```

For affine A, the catalogue uses the complete string/band classification:
bounded strings on the universal cover and one Jordan band for every nonzero
parameter in the prime field. The strings supply the two exceptional tubes,
including the exceptional indecomposables of dimension `m * delta`.

For affine D, every bounded real root receives its unique indecomposable.
Source/sink reflection functors construct the reflection-reducible roots;
exceptional regular roots are continued by certified nonsplit extensions.
For every imaginary dimension `m * delta`, the affine Kac polynomial gives
exactly `q + n` absolutely indecomposable classes over `F_q`, where `n` is the
affine index. The implementation enumerates the `delta` seeds reproducibly,
checks isomorphism over the finite field, and constructs their higher regular
extensions. Both finite-field and exact characteristic-zero endomorphism ranks
are checked while these normal forms are built.

`random_seed` controls only the reproducible search for affine-D normal forms;
the expected class counts and all acceptance tests are exact. `max_attempts`
can be raised if a large field or dimension bound makes the last normal form
hard to encounter.

`closed_points` deliberately raises `NotImplementedError` for affine A and D,
as it does for Kronecker. It will require irreducible-polynomial tubes and a
packet-valued lift to characteristic zero.

The repository coverage runner accepts the YAML types `AffineAn` and
`AffineDn`. Its runtime section may specify the finite catalogue parameters:

```yaml
type: AffineDn
runtime:
  hom_prime: 107
  regular_mode: rational_jordan
  random_seed: 0
  max_attempts: 50000
```

Run it in the same way as a finite-type sweep:

```bash
python scripts/run_coverages.py --config affine_d5_coverages.yaml --workers 16
```

The runner computes the catalogue and Hom profiles over `GF(hom_prime)`, but
the generated RAD scripts still use the characteristic-zero lifts and the
existing `QQ` Macaulay2 ring. The finite-type `compute_hom_dims.py`
postprocessor is skipped because it cannot parse affine module labels; the
affine pipeline writes `rank_poset/hom_profiles.csv` directly.

## General affine utilities

`classify_affine_quiver(Q)` recognizes acyclic orientations of affine ADE
diagrams, including the two-arrow Kronecker graph. It returns the affine type,
the primitive null root `delta`, and, for affine A, the two orientation counts.

`enumerate_affine_real_roots_bounded(Q, bound)` enumerates all nonzero vectors
within a coordinatewise bound having Tits value one. The reflection helper also
accepts imaginary indecomposable dimensions `m * delta`. Explicit bounded
module catalogues are currently implemented for affine A and D; affine E has
root-system support but not yet a module-bag backend.
