"""compile: solve an edit and re-check it. then: apply one edit, build the next on the edited model."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from weightir.check import check, stored_dtype
from weightir.solve import Infeasible, solve
from weightir.spec import Edit, Exact, edit, span


@dataclass
class Patch:
    """The weight change for one site and the residuals measured on the original data (``check["ok"]`` is True:
    compile never returns a patch that fails the check)."""

    site: str
    delta: np.ndarray
    check: dict


def compile(e: Edit, W, *, dtype: str | None = None, rtol: float = 1e-10) -> Patch:
    """Solve ``e`` against the weight ``W`` and re-check the result; raise Infeasible("check") if it fails.

    ``W`` is an array, or anything with ``.weight(site)`` such as a weightir Model; a Model's parameter dtype is the
    default ``dtype``: the precision the patch will be stored in, which the check also covers.
    """
    if not isinstance(W, np.ndarray) and hasattr(W, "weight"):
        if dtype is None and hasattr(W, "param"):
            dtype = stored_dtype(W.param(e.site).dtype)
        W = W.weight(e.site)
    W = np.asarray(W, dtype=np.float64)
    delta, info = solve(e, W, rtol=rtol)
    report = check(e, W, delta, dtype=dtype)
    report.update(info)
    if not report["ok"]:
        raise Infeasible("check", "the computed change does not meet the edit on its data"
                         + (f" once stored as {dtype}" if dtype else ""), check=report)
    return Patch(e.site, delta, report)


def then(model, first: Edit, build_second: Callable[[object], Edit], *, keep_previous: bool = True,
         dtype: str | None = None) -> list[Patch]:
    """Apply ``first``, then build and apply the second edit on the edited model.

    ``build_second`` receives the model after ``first`` was applied, so keys and statistics it
    collects describe the edited model. With ``keep_previous`` and the same site, the second edit
    also keeps what ``first`` asked for: its exact keys stay mapped to their values and its
    preserves (exact subspaces and soft penalties) carry over. When both edits are exact (no soft
    terms) and minimize the same metric, the result equals solving them jointly; otherwise it
    generally does not.
    """
    p1 = compile(first, model, dtype=dtype)
    model.apply(p1)
    second = build_second(model)
    if keep_previous and second.site == first.site:
        keep = list(first.preserve)
        exact = [m.K.data for m in first.maps if m.weight is None]
        if exact:
            keep.append(Exact(span(np.hstack(exact), first.site)))
        if keep:
            second = edit(second.site, map=second.maps, preserve=[*second.preserve, *keep],
                          minimize=second.minimize, require=second.require, ridge=second.ridge)
    p2 = compile(second, model, dtype=dtype)
    model.apply(p2)
    return [p1, p2]
