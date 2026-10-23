"""compile: solve an edit and re-check it. then: apply one edit, build the next on the edited model."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from weightir.check import check
from weightir.solve import solve
from weightir.spec import Edit, Exact, edit, span


@dataclass
class Patch:
    """The weight change for one site and the residuals measured on the original data."""

    site: str
    delta: np.ndarray
    check: dict


def compile(e: Edit, W, *, dtype: str | None = None, rtol: float = 1e-10) -> Patch:
    """Solve ``e`` against the weight ``W`` (an array, or anything with ``.weight(site)``)."""
    if not isinstance(W, np.ndarray) and hasattr(W, "weight"):
        W = W.weight(e.site)
    W = np.asarray(W, dtype=np.float64)
    delta, info = solve(e, W, rtol=rtol)
    report = check(e, W, delta, dtype=dtype)
    report.update(info)
    return Patch(e.site, delta, report)


def then(model, first: Edit, build_second: Callable[[object], Edit], *, keep_previous: bool = True,
         dtype: str | None = None) -> list[Patch]:
    """Apply ``first``, then build and apply the second edit on the edited model.

    ``build_second`` receives the model after ``first`` was applied, so keys and statistics it
    collects describe the edited model. With ``keep_previous`` and the same site, the second
    edit also keeps the exact keys of ``first`` mapped to their values, which makes the result
    equal to solving both edits jointly.
    """
    p1 = compile(first, model, dtype=dtype)
    model.apply(p1)
    second = build_second(model)
    if keep_previous and second.site == first.site:
        exact = [m.K.data for m in first.maps if m.weight is None]
        if exact:
            keep = Exact(span(np.hstack(exact), first.site))
            second = edit(second.site, map=second.maps, preserve=[*second.preserve, keep],
                          minimize=second.minimize, require=second.require, ridge=second.ridge)
    p2 = compile(second, model, dtype=dtype)
    model.apply(p2)
    return [p1, p2]
