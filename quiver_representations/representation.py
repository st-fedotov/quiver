"""Compatibility shims for :mod:`quiver_representations`.

The :class:`Module` and :class:`Morphism` classes now live in dedicated
modules.  Importing them from :mod:`quiver_representations.representation`
continues to work for existing users, but new code should prefer the
module-specific imports.
"""

from __future__ import annotations

from .module import Module
from .morphism import Morphism

__all__ = ["Module", "Morphism"]
