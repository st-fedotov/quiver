"""
Helper functions for Grassmannian computations and Macaulay2 integration.
"""

import numpy as np
from fractions import Fraction
from typing import Iterable, List, Sequence, Set, Dict, Tuple, Any, Optional

def are_projectively_equivalent(coords1: np.ndarray, coords2: np.ndarray,
                              zero_tolerance: float = 1e-5,
                              comparison_tolerance: float = 1e-5) -> bool:
    """
    Check if two sets of Plücker coordinates represent the same point in projective space.

    Two coordinate vectors represent the same projective point if one is a scalar multiple
    of the other: coords2 = λ * coords1 for some λ ∈ ℂ.

    Algorithm:
    1. Find index i where |coords1[i]| is maximal among indices where |coords2[i]| > zero_tolerance
    2. Compute λ = coords2[i] / coords1[i]
    3. Check if ||coords2 - λ * coords1|| < comparison_tolerance

    Args:
        coords1: First set of Plücker coordinates
        coords2: Second set of Plücker coordinates
        zero_tolerance: Threshold for considering a coordinate as zero
        comparison_tolerance: Tolerance for final proportionality check

    Returns:
        True if the coordinates represent the same projective point, False otherwise

    Raises:
        ValueError: If both coordinate vectors are effectively zero
    """
    # Convert to numpy arrays and ensure complex dtype
    coords1 = np.asarray(coords1, dtype=complex)
    coords2 = np.asarray(coords2, dtype=complex)

    # Check if arrays have the same length
    if len(coords1) != len(coords2):
        return False

    # Check if both vectors are effectively zero (invalid for projective space)
    if (np.all(np.abs(coords1) < zero_tolerance) and
        np.all(np.abs(coords2) < zero_tolerance)):
        raise ValueError("Both coordinate vectors are effectively zero - not valid points in projective space")

    # Find indices where coords2 is non-negligible (potential denominators)
    viable_indices = np.where(np.abs(coords2) > zero_tolerance)[0]

    if len(viable_indices) == 0:
        # coords2 is effectively zero but coords1 is not
        return False

    # Among viable indices, find where |coords1| is maximal (best numerator for stability)
    coords1_magnitudes = np.abs(coords1[viable_indices])
    best_local_idx = np.argmax(coords1_magnitudes)
    best_idx = viable_indices[best_local_idx]

    # Check if the chosen coords1[best_idx] is also non-negligible
    if np.abs(coords1[best_idx]) < zero_tolerance:
        # All coords1[i] are tiny where coords2[i] is large - vectors not proportional
        return False

    # Compute scaling factor using the most numerically stable ratio
    lambda_factor = coords1[best_idx] / coords2[best_idx]

    # Check if coords1 = lambda_factor * coords2 (i.e., coords2 = coords1/lambda_factor)
    scaled_coords2 = lambda_factor * coords2
    difference_norm = np.linalg.norm(coords1 - scaled_coords2)

    return difference_norm < comparison_tolerance

def pick_lucky_primes_from_gb(
    gb_polys: Sequence["PluckerPolynomial"],
    candidates: Iterable[int],
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    max_denominator: int = 10**6,
    max_count: int | None = None,
) -> List[int]:
    """
    Keep primes that divide NONE of the numerators/denominators of GB coefficients
    (after rationalization). This is the simple 'lucky prime' guard the user requested.
    """
    # collect all numerators/denominators
    nums: List[int] = []
    dens: List[int] = []
    for P in gb_polys:
        for c, _mon in P.normalised_monomials():
            # treat as real rational
            x = float(c.real if isinstance(c, complex) else c)
            if abs(x) <= atol:
                continue
            fr = Fraction(x).limit_denominator(max_denominator)
            # accept even if approximation mismatch; we're intentionally permissive here
            nums.append(abs(fr.numerator))
            dens.append(abs(fr.denominator))
    # quick membership test: for each p, just check divisibility against all nums/dens
    def ok_prime(p: int) -> bool:
        if p <= 2:
            return False
        # refuse if p divides any numerator or denominator
        for a in nums:
            if a % p == 0:
                return False
        for b in dens:
            if b % p == 0:
                return False
        return True

    chosen: List[int] = []
    for p in candidates:
        if ok_prime(p):
            chosen.append(p)
            if max_count is not None and len(chosen) >= max_count:
                break

    return chosen

def vertex_order(quiver, active_vertices=None, sinks_first=False):
    """
    Canonical vertex order for exporters.
    Uses the existing topological_sort(quiver), with optional filtering and reversing.

    Args:
        quiver: your quiver object
        active_vertices: optional iterable of vertices actually appearing in variables/monomials
        sinks_first: if True, reverse the topological order

    Returns:
        List of vertex IDs in the desired order.
    """
    order = topological_sort(quiver)
    if active_vertices is not None:
        active = set(active_vertices)
        order = [v for v in order if v in active]
    if sinks_first:
        order = list(reversed(order))
    return order


def build_block_vars(quiver, variables_by_vertex, active_vertices=None, sinks_first=False):
    """
    Centralizes the repeated pattern:
      - choose vertex order
      - assemble per-vertex variable blocks
      - flatten to ordered_vars
      - compute block_sizes

    Args:
        quiver: your quiver object
        variables_by_vertex: dict {vertex -> list of variable symbols/strings}
        active_vertices: optional iterable to restrict to a subset of vertices
        sinks_first: if True, reverse the topological order

    Returns:
        (order, block_vars, ordered_vars, block_sizes)
    """
    order = vertex_order(quiver, active_vertices=active_vertices, sinks_first=sinks_first)

    block_vars = []
    for v in order:
        if v in variables_by_vertex:
            block_vars.append(list(variables_by_vertex[v]))
        # If a vertex doesn't appear in variables_by_vertex, we just skip it.

    ordered_vars = [var for block in block_vars for var in block]
    block_sizes = [len(block) for block in block_vars]

    return order, block_vars, ordered_vars, block_sizes

from fractions import Fraction
import math
import numbers

def rationalize(
    c,
    atol: float = 1e-10,                     # |Im| tolerance and zero-threshold
    rtol: float = 1e-9,                      # relative tolerance for rationalization
    max_denominator: int = 10**6
) -> Fraction:
    """
    Convert c (int/float/complex/Fraction/NumPy scalar) to an exact Fraction.
    - Treats tiny values (|c| <= atol) as 0.
    - Requires 'effectively real': if imag(c) > atol -> error.
    - Approximates floats via limit_denominator and validates |approx - c| tolerance.

    Returns:
        fractions.Fraction

    Raises:
        ValueError if c is non-real beyond tolerance or cannot be approximated within tolerance.
    """
    # Fast paths
    if isinstance(c, Fraction):
        return c
    if isinstance(c, numbers.Integral):
        return Fraction(int(c), 1)

    # Complex handling (incl. np.complex*)
    if isinstance(c, complex):
        if abs(c.imag) > atol:
            raise ValueError(f"Non-real coefficient: {c}")
        x = float(c.real)
    else:
        # Real-like (incl. np.float*, Decimals converted via float(c) upstream if needed)
        try:
            x = float(c)
        except Exception as e:
            raise ValueError(f"Unsupported coefficient type {type(c)}") from e

    # Zero clip
    if math.isfinite(x) is False:
        raise ValueError(f"Non-finite coefficient: {x}")
    if abs(x) <= atol:
        return Fraction(0, 1)

    # Rational approximation
    fr = Fraction(x).limit_denominator(max_denominator)
    err = abs(float(fr) - x)
    if err <= max(atol, rtol * max(1.0, abs(x))):
        return fr

    raise ValueError(
        f"Cannot rationalize {x} within tolerance; "
        f"got {fr.numerator}/{fr.denominator} (error {err:g})"
    )

def coef_to_m2(c, atol=1e-10, rtol=1e-9, max_denominator=10**6) -> str:
    fr = rationalize(c, atol=atol, rtol=rtol, max_denominator=max_denominator)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"

    from fractions import Fraction

from collections import Counter

def term_to_str(
    mon,
    coef,
    prefix: str = "p_",
    allow_powers: bool = True,
    atol: float = 1e-10,
    rtol: float = 1e-9,
    max_denominator: int = 10**6,
    var_sort_key=None,         # optional: callable(name)->key to force a custom factor order
) -> str:
    """
    Render coef * product(mon) to a single Macaulay2 term string.
    - Groups repeated variables as name^k when allow_powers=True.
    - Deterministic factor order (lex on names unless var_sort_key is given).
    - Coefficients are rationalized consistently.
    """
    # Rationalize coefficient and handle zero quickly
    fr = rationalize(coef, atol=atol, rtol=rtol, max_denominator=max_denominator)
    if fr == 0:
        return "0"

    # Build factor list (variable names)
    names = [vname(v, prefix) for v in mon]
    if allow_powers:
        counts = Counter(names)
        factors = []
        for nm in sorted(counts, key=var_sort_key or (lambda s: s)):
            k = counts[nm]
            factors.append(nm if k == 1 else f"{nm}^{k}")
    else:
        factors = sorted(names, key=var_sort_key or (lambda s: s))

    # Monomial string
    mon_str = " * ".join(factors) if factors else "1"

    # Coefficient handling
    #   If mon_str == "1", just print the coefficient.
    #   Else absorb unit coefficients 1/-1 into the sign of the monomial.
    if mon_str == "1":
        return coef_to_m2(fr, atol=atol, rtol=rtol, max_denominator=max_denominator)

    if fr == 1:
        return mon_str
    if fr == -1:
        return f"-{mon_str}"

    return f"{coef_to_m2(fr, atol=atol, rtol=rtol, max_denominator=max_denominator)} * {mon_str}"


def poly_to_m2(
    P,
    prefix: str = "p_",
    allow_powers: bool = True,
    atol: float = 1e-10,
    rtol: float = 1e-9,
    max_denominator: int = 10**6,
    term_var_sort_key=None,     # same semantics as var_sort_key in term_to_str
) -> str:
    """
    Render a Plücker polynomial P to a Macaulay2 string.
    - Uses P.normalised_monomials() yielding (coef, mon).
    - Skips zero terms after rationalization.
    - Places the sign between terms; first term has no leading '+'.
    """
    pieces = []
    first = True

    for c, mon in P.normalised_monomials():
        # Rationalize once to decide sign/zero, but pass the *original* c to term_to_str
        # (term_to_str rationalizes again consistently; this keeps one public path)
        fr = rationalize(c, atol=atol, rtol=rtol, max_denominator=max_denominator)
        if fr == 0:
            continue

        # Build body with the absolute value so we can control the sign in the joiner
        body = term_to_str(
            mon,
            abs(fr),
            prefix=prefix,
            allow_powers=allow_powers,
            atol=atol,
            rtol=rtol,
            max_denominator=max_denominator,
            var_sort_key=term_var_sort_key,
        )

        if first:
            pieces.append(body if fr > 0 else (body if body.startswith("-") else f"-{body}"))
            first = False
        else:
            pieces.append((" + " if fr > 0 else " - ") + (body if not body.startswith("-") else body[1:]))

    return "0" if not pieces else "".join(pieces)



def vname(var: PluckerVar, prefix: str) -> str:
    """
    Generate variable name for Plücker coordinate.

    Args:
        plucker_var: PluckerVar object
        prefix: Prefix for variable name

    Returns:
        Variable name string
    """

    inside = ",".join([str(var.vertex)] + [str(i) for i in var.subset])
    return f"{prefix}({inside})"
