"""The edit DSL: typed data and constraints on one weight matrix (a "site").

A site is a linear map ``y = W x`` with ``W`` of shape ``(out_dim, in_dim)``. An edit is

    minimize    tr(ΔW C ΔWᵀ)                              (C from ``minimize``; identity if None)
              + Σ_soft maps     w ‖(W + ΔW) K − V‖²
              + Σ_soft preserves w tr(ΔW C_j ΔWᵀ)
              + ridge ‖ΔW‖²
    subject to  (W + ΔW) K = V       for exact maps
                ΔW U = 0             for exact preserves
    budget      rank(ΔW) ≤ r,  ‖ΔW‖ ≤ B

Keys, Values, Subspace and Covariance remember which site they belong to, so mixing up
sites or putting an output-space vector where an input-space vector belongs is a SpecError.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SpecError(ValueError):
    """The spec is malformed: wrong types, mismatched sites, or inconsistent shapes."""


def _matrix(a, what: str) -> np.ndarray:
    arr = np.array(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise SpecError(f"{what} must be a 2-D array (one column per example), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise SpecError(f"{what} contains NaN or inf")
    return arr


@dataclass
class Keys:
    """Input vectors of ``site``, one column per example: shape ``(in_dim, p)``."""

    data: np.ndarray
    site: str

    def __post_init__(self) -> None:
        self.data = _matrix(self.data, "Keys.data")

    @property
    def count(self) -> int:
        return self.data.shape[1]


@dataclass
class Values:
    """Desired outputs of ``site`` for the matching Keys, one column per example: ``(out_dim, p)``."""

    data: np.ndarray
    site: str

    def __post_init__(self) -> None:
        self.data = _matrix(self.data, "Values.data")

    @property
    def count(self) -> int:
        return self.data.shape[1]


@dataclass
class Subspace:
    """Orthonormal basis ``(in_dim, s)`` of a subspace of ``site``'s input space.

    ``exact=False`` marks a thresholded estimate of where some data lives (energy or
    eigenvalue cut). ``ΔW U = 0`` then holds for the basis, not for the data, so ``source``
    keeps the data and ``check`` reports how much the outputs on it moved.
    """

    basis: np.ndarray
    site: str
    exact: bool = True
    source: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.basis = _matrix(self.basis, "Subspace.basis")
        s = self.basis.shape[1]
        if s and np.abs(self.basis.T @ self.basis - np.eye(s)).max() > 1e-8:
            raise SpecError("Subspace.basis must have orthonormal columns (use weightir.span or weightir.subspace)")
        if self.source is not None:
            self.source = _matrix(self.source, "Subspace.source")
            if self.source.shape[0] != self.basis.shape[0]:
                raise SpecError("Subspace.source must have the same number of rows as the basis")


@dataclass
class Covariance:
    """Second moment ``E[x xᵀ]`` of ``site``'s inputs, estimated from ``n`` vectors."""

    C: np.ndarray
    site: str
    n: int = 0

    def __post_init__(self) -> None:
        C = _matrix(self.C, "Covariance.C")
        if C.shape[0] != C.shape[1]:
            raise SpecError(f"Covariance.C must be square, got {C.shape}")
        self.C = (C + C.T) / 2


@dataclass
class Map:
    """``(W + ΔW) K = V``: exact when ``weight`` is None, otherwise the penalty ``weight‖(W+ΔW)K − V‖²``."""

    K: Keys
    V: Values
    weight: float | None = None


@dataclass
class Preserve:
    """Exact: ``ΔW U = 0`` for a Subspace. Soft: the penalty ``weight·tr(ΔW C ΔWᵀ)`` for a Covariance."""

    target: Subspace | Covariance
    weight: float | None = None


@dataclass
class Rank:
    """Budget ``rank(ΔW) ≤ r``."""

    r: int


@dataclass
class Norm:
    """Budget ``‖ΔW‖ ≤ B``: Frobenius norm (``kind="fro"``) or the ``minimize`` metric (``kind="metric"``)."""

    B: float
    kind: str = "fro"


def Exact(*args) -> Map | Preserve:
    """``Exact(K, V)``: keys K must map exactly to values V. ``Exact(U)``: inputs in span(U) keep their outputs."""
    if len(args) == 2 and isinstance(args[0], Keys) and isinstance(args[1], Values):
        return Map(args[0], args[1], None)
    if len(args) == 1 and isinstance(args[0], Subspace):
        return Preserve(args[0], None)
    raise SpecError(
        "Exact takes (Keys, Values) or (Subspace); got " + ", ".join(type(a).__name__ for a in args)
    )


def Soft(*args, weight: float = 1.0) -> Map | Preserve:
    """``Soft(K, V, weight=w)``: penalize ``w‖(W+ΔW)K − V‖²``. ``Soft(C, weight=w)``: penalize ``w·tr(ΔW C ΔWᵀ)``."""
    if not weight > 0:
        raise SpecError(f"Soft weight must be positive, got {weight}")
    if len(args) == 2 and isinstance(args[0], Keys) and isinstance(args[1], Values):
        return Map(args[0], args[1], float(weight))
    if len(args) == 1 and isinstance(args[0], Covariance):
        return Preserve(args[0], float(weight))
    raise SpecError(
        "Soft takes (Keys, Values) or (Covariance); got " + ", ".join(type(a).__name__ for a in args)
    )


@dataclass
class Edit:
    site: str
    maps: list[Map]
    preserve: list[Preserve] = field(default_factory=list)
    minimize: Covariance | None = None
    ridge: float = 0.0
    require: list[Rank | Norm] = field(default_factory=list)

    @property
    def in_dim(self) -> int:
        return self.maps[0].K.data.shape[0]

    @property
    def out_dim(self) -> int:
        return self.maps[0].V.data.shape[0]


def _listify(x, kind) -> list:
    if x is None:
        return []
    if isinstance(x, kind):
        return [x]
    return list(x)


def edit(site: str, map=None, preserve=None, minimize: Covariance | None = None,
         require=None, ridge: float = 0.0) -> Edit:
    """Build an Edit and check that every part belongs to ``site`` and the shapes agree."""
    maps = _listify(map, Map)
    keeps = _listify(preserve, Preserve)
    budgets = _listify(require, (Rank, Norm))
    if not maps:
        raise SpecError("an edit needs at least one map (Exact(K, V) or Soft(K, V))")
    for m in maps:
        if not isinstance(m, Map):
            raise SpecError(f"map entries must be Exact(K, V) or Soft(K, V), got {type(m).__name__}")
    for p in keeps:
        if not isinstance(p, Preserve):
            raise SpecError(f"preserve entries must be Exact(U) or Soft(C), got {type(p).__name__}")
    for b in budgets:
        if not isinstance(b, (Rank, Norm)):
            raise SpecError(f"require entries must be Rank or Norm, got {type(b).__name__}")
    if minimize is not None and not isinstance(minimize, Covariance):
        raise SpecError(f"minimize must be a Covariance or None, got {type(minimize).__name__}")
    if ridge < 0:
        raise SpecError("ridge must be nonnegative")

    n, m_out = maps[0].K.data.shape[0], maps[0].V.data.shape[0]
    for i, mp in enumerate(maps):
        for obj, name in ((mp.K, "Keys"), (mp.V, "Values")):
            if obj.site != site:
                raise SpecError(f"map {i}: {name} belong to site {obj.site!r}, not {site!r}")
        if mp.K.count != mp.V.count:
            raise SpecError(f"map {i}: {mp.K.count} keys but {mp.V.count} values")
        if mp.K.data.shape[0] != n or mp.V.data.shape[0] != m_out:
            raise SpecError(f"map {i}: keys must be {n}-dim and values {m_out}-dim like map 0")
    for i, p in enumerate(keeps):
        t = p.target
        if t.site != site:
            raise SpecError(f"preserve {i}: belongs to site {t.site!r}, not {site!r}")
        if p.weight is None and not isinstance(t, Subspace):
            raise SpecError(f"preserve {i}: exact preservation needs a Subspace")
        if p.weight is not None and not isinstance(t, Covariance):
            raise SpecError(f"preserve {i}: soft preservation needs a Covariance")
        dim = t.basis.shape[0] if isinstance(t, Subspace) else t.C.shape[0]
        if dim != n:
            raise SpecError(f"preserve {i}: lives in a {dim}-dim space, but keys are {n}-dim")
    if minimize is not None:
        if minimize.site != site:
            raise SpecError(f"minimize: Covariance belongs to site {minimize.site!r}, not {site!r}")
        if minimize.C.shape[0] != n:
            raise SpecError(f"minimize: Covariance is {minimize.C.shape[0]}-dim, but keys are {n}-dim")
    for b in budgets:
        if isinstance(b, Rank) and b.r < 0:
            raise SpecError("Rank budget must be nonnegative")
        if isinstance(b, Norm):
            if not b.B > 0:
                raise SpecError("Norm budget must be positive")
            if b.kind not in ("fro", "metric"):
                raise SpecError("Norm kind must be 'fro' or 'metric'")
            if b.kind == "metric" and minimize is None:
                raise SpecError("Norm(kind='metric') needs minimize=Covariance")
    return Edit(site, maps, keeps, minimize, float(ridge), budgets)


def joint(*edits: Edit) -> Edit:
    """Solve several edits of the same site as one problem (all constraints at once)."""
    if not edits:
        raise SpecError("joint needs at least one edit")
    first = edits[0]
    for e in edits[1:]:
        if e.site != first.site:
            raise SpecError(f"joint edits must share a site: {first.site!r} vs {e.site!r}")
        if (e.minimize is None) != (first.minimize is None) or (
            e.minimize is not None and not np.array_equal(e.minimize.C, first.minimize.C)
        ):
            raise SpecError("joint edits must minimize the same metric")
        if e.ridge != first.ridge:
            raise SpecError("joint edits must use the same ridge")
    return edit(
        first.site,
        map=[m for e in edits for m in e.maps],
        preserve=[p for e in edits for p in e.preserve],
        minimize=first.minimize,
        require=[b for e in edits for b in e.require],
        ridge=first.ridge,
    )


def orth(X: np.ndarray, rtol: float) -> np.ndarray:
    """Orthonormal basis of the column space of X: left singular vectors with singular value above rtol·s_max."""
    if X.shape[1] == 0:
        return X
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :0] if s[0] == 0 else U[:, s > rtol * s[0]]


def span(vectors, site: str, rtol: float = 1e-10) -> Subspace:
    """Exact subspace spanned by the columns of ``vectors`` (Keys or an array)."""
    data = vectors.data if isinstance(vectors, Keys) else _matrix(vectors, "vectors")
    if isinstance(vectors, Keys) and vectors.site != site:
        raise SpecError(f"Keys belong to site {vectors.site!r}, not {site!r}")
    if data.shape[1] == 0:
        return Subspace(np.zeros((data.shape[0], 0)), site, exact=True)
    return Subspace(orth(data, rtol), site, exact=True, source=data)


def subspace(vectors, site: str, *, energy: float | None = None, threshold: float | None = None,
             rank: int | None = None) -> Subspace:
    """Estimated subspace where the columns of ``vectors`` mostly live (``exact=False``).

    Exactly one rule: ``energy`` keeps the fewest top singular directions holding that
    fraction of the squared norm; ``threshold`` keeps eigenvalues of the mean second moment
    ``XXᵀ/p`` above it (AlphaEdit-style cut); ``rank`` keeps the top ``rank`` directions.
    """
    data = vectors.data if isinstance(vectors, Keys) else _matrix(vectors, "vectors")
    if isinstance(vectors, Keys) and vectors.site != site:
        raise SpecError(f"Keys belong to site {vectors.site!r}, not {site!r}")
    if sum(x is not None for x in (energy, threshold, rank)) != 1:
        raise SpecError("subspace needs exactly one of energy=, threshold=, rank=")
    U, s, _ = np.linalg.svd(data, full_matrices=False)
    if energy is not None:
        if not 0 < energy <= 1:
            raise SpecError("energy must be in (0, 1]")
        frac = np.cumsum(s**2) / max(np.sum(s**2), np.finfo(float).tiny)
        k = int(np.searchsorted(frac, energy - 1e-12) + 1)
    elif threshold is not None:
        k = int(np.sum(s**2 / data.shape[1] > threshold))
    else:
        if not 0 <= rank <= len(s):
            raise SpecError(f"rank must be between 0 and {len(s)}")
        k = int(rank)
    return Subspace(U[:, :k], site, exact=False, source=data)
