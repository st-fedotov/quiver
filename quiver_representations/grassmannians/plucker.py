"""
Plücker coordinate variable and polynomial infrastructure.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional, Iterable


# --- numeric tolerances ---
EQ_RTOL: float = 1e-9
EQ_ATOL: float = 1e-10
EPS_PRUNE: float = 1e-10
STRAIGHTEN_TOL: float = EQ_ATOL

@dataclass(frozen=True)
class PluckerVar:
    kind: str              # 'p' or 'q'
    vertex: int
    subset: Tuple[int, ...]
    def key(self):
        return (self.kind, self.vertex, self.subset)

class SymbolRegistry:
    """
    Stable names/symbols for variables + global variable order.
    Use for LaTeX/text printing and SymPy conversion.
    """
    def __init__(self, order: Optional[Iterable[PluckerVar]] = None):
        self._vars: List[PluckerVar] = []
        self._name: Dict[PluckerVar, str] = {}
        self._latex: Dict[PluckerVar, str] = {}
        self._sym: Dict[PluckerVar, "sympy.Symbol"] = {}
        if order is not None:
            for v in order:
                self.register(v)

    def register(self, v: PluckerVar):
        if v in self._name:
            return
        nm = f"{v.kind}[v{v.vertex},{v.subset}]"
        lt = f"{v.kind}_{{v{v.vertex},{v.subset}}}"
        self._vars.append(v)
        self._name[v] = nm
        self._latex[v] = lt

    def ensure(self, vars: Iterable[PluckerVar]):
        for v in vars:
            self.register(v)

    def name(self, v: PluckerVar) -> str:  return self._name[v]
    def latex(self, v: PluckerVar) -> str: return self._latex[v]

    def sym(self, v: PluckerVar):
        import sympy as sp
        if v not in self._sym:
            self._sym[v] = sp.Symbol(self._name[v])
        return self._sym[v]

    @property
    def order(self) -> List[PluckerVar]:
        # Deterministic global variable order
        return sorted(self._vars, key=lambda z: z.key())

@dataclass
class PluckerPolynomial:
    # list of (coef, monomial), monomial = tuple of PluckerVar (each exp 1)
    terms: List[Tuple[complex, Tuple[PluckerVar, ...]]]
    meta: Dict = field(default_factory=dict)  # e.g., arrow_id, tail, head, I, J

    def normalised_monomials(self) -> List[Tuple[complex, Tuple[PluckerVar, ...]]]:
        # sort variables inside each monomial; sort terms by monomial key
        norm = []
        for c, mons in self.terms:
            mons_sorted = tuple(sorted(mons, key=lambda v: v.key()))
            norm.append((c, mons_sorted))
        norm.sort(key=lambda t: tuple(w.key() for w in t[1]))
        return norm

    def latex(self, reg: SymbolRegistry) -> str:
        """Flat LaTeX string like:  2 p_{v0,(0)} q_{v1,(1,2)} - p_{v0,(1)} q_{v1,(0,2)}"""
        reg.ensure(var for _, mon in self.terms for var in mon)
        parts = []
        for coef, mon in self.normalised_monomials():
            mon_str = " ".join(reg.latex(v) for v in mon)
            # leave signs as generated (your preference)
            if np.isclose(coef.imag, 0.0):
                ctex = f"{coef.real:.12g}"
            else:
                ctex = f"({coef.real:.12g}+{coef.imag:.12g}i)"
            if ctex == "1":
                parts.append(mon_str)
            elif ctex == "-1":
                parts.append(f"- {mon_str}")
            else:
                parts.append(f"{ctex}\\,{mon_str}")
        return " + ".join(parts)

    def text(self, reg: SymbolRegistry) -> str:
        reg.ensure(var for _, mon in self.terms for var in mon)
        segs = []
        for coef, mon in self.normalised_monomials():
            mon_str = " * ".join(reg.name(v) for v in mon)
            segs.append(f"{coef}*{mon_str}")
        return " + ".join(segs)

    def equals_up_to_scalar(self, other: "PluckerPolynomial",
                            rtol: float = EQ_RTOL, atol: float = EQ_ATOL) -> bool:
        A = self.normalised_monomials()
        B = other.normalised_monomials()
        # monomials must match exactly
        if [tuple(v.key() for v in m) for _, m in A] != [tuple(v.key() for v in m) for _, m in B]:
            return False
        # find a scale from first nonzero coef
        def first_nz(lst):
            for c, _ in lst:
                if not np.isclose(c, 0.0, rtol=rtol, atol=atol):
                    return c
            return 0.0
        a1 = first_nz(A); b1 = first_nz(B)
        if np.isclose(a1, 0.0, rtol=rtol, atol=atol) and np.isclose(b1, 0.0, rtol=rtol, atol=atol):
            return True
        if np.isclose(b1, 0.0, rtol=rtol, atol=atol):
            return False
        lam = a1 / b1
        for (c1, _), (c2, _) in zip(A, B):
            if not np.isclose(c1, lam * c2, rtol=rtol, atol=atol):
                return False
        return True

    def to_sympy(self, reg: SymbolRegistry):
        """Return a SymPy Expr; later you can build Poly(...) with chosen order."""
        import sympy as sp
        reg.ensure(var for _, mon in self.terms for var in mon)
        expr = 0
        for coef, mon in self.normalised_monomials():
            mexpr = 1
            for v in mon:
                mexpr *= reg.sym(v)
            # nsimplify keeps rationals exact when possible
            expr += sp.nsimplify(coef) * mexpr
        return expr
