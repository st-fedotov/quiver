"""
Field conversion utilities for modules.

Provides functionality to rebuild modules over different fields,
particularly useful for working with finite fields.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...module import Module
    from ...field import FiniteField


def module_over_field(module: "Module", Fp: "FiniteField",
                     name_suffix: str = " (mod p)") -> "Module":
    """
    Rebuild 'module' over 'Fp' by converting each arrow matrix via Fp.GF(array_like).

    ZeroMap is preserved as-is. If the module is already over Fp, this is a cheap copy.

    Args:
        module: Source module to convert
        Fp: Target finite field
        name_suffix: Suffix to append to the module name

    Returns:
        New module over the finite field Fp

    Note:
        This function converts all arrow matrices to the target field.
        Zero morphisms are preserved without conversion.
    """
    Q = module.quiver
    spaces = dict(module.spaces)
    maps_p = {}

    for a_id in Q.get_arrows():          # keys match Module.maps
        A = module.maps[a_id]
        if isinstance(A, ZeroMap):
            maps_p[a_id] = A
        else:
            # Coerce to the field array using the field class as a constructor
            maps_p[a_id] = Fp.GF(A)

    name = (getattr(module, "name", "M") or "M") + name_suffix
    return Module(Q, Fp, name=name, dimensions=spaces, maps=maps_p)
