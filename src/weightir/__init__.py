"""WeightIR: a typed, Python-embedded DSL for editing one weight matrix at a time.

    import weightir as wir
    e = wir.edit(site, map=wir.Exact(K, V), preserve=[wir.Exact(U)], minimize=C,
                 require=[wir.Rank(8), wir.Norm(1.0)])
    patch = wir.compile(e, model)          # or a weight array
    model.apply(patch)
"""
from weightir.check import check, round_to
from weightir.compile import Patch, compile, then
from weightir.solve import Infeasible, solve
from weightir.spec import (
    Covariance,
    Edit,
    Exact,
    Keys,
    Map,
    Norm,
    Preserve,
    Rank,
    Soft,
    SpecError,
    Subspace,
    Values,
    edit,
    joint,
    span,
    subspace,
)

__all__ = [
    "Covariance", "Edit", "Exact", "Infeasible", "Keys", "Map", "Norm", "Patch", "Preserve", "Rank",
    "Soft", "SpecError", "Subspace", "Values", "check", "compile", "edit", "joint", "round_to",
    "solve", "span", "subspace", "then",
]
