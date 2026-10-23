"""Solve an Edit on a weight matrix W: closed forms, budgets, and infeasibility proofs.

All edits reduce to

    minimize   tr(ΔW H ΔWᵀ) − 2 tr(ΔW Gᵀ)        subject to   ΔW A = B

with A = [K_exact, U], B = [V_exact − W K_exact, 0],
     H = M + Σ_soft w K Kᵀ + Σ_soft w C + ridge·I   (M = minimize metric, or I),
     G = Σ_soft w (V − W K) Kᵀ.

The unique minimizer (H ≻ 0) is ΔW = G H⁻¹ + Λ (H⁻¹A)ᵀ with Λ = (B − G H⁻¹ A)(Aᵀ H⁻¹ A)⁺.
Special cases: H = I, no soft terms → ΔW = B A⁺ (minimum Frobenius norm); H = C → the
ROME / EMMET form B (Aᵀ C⁻¹ A)⁺ Aᵀ C⁻¹.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.linalg as sla

from weightir.spec import Edit, Norm, Rank, SpecError, orth

BAND = 1e3          # singular values within (tol, tol·BAND] make a decision numerically unresolved
FEAS_TOL = 1e-8     # relative residual above which inconsistent exact constraints count as a conflict
RANK_RTOL = 1e-9    # relative singular value cut used to count the rank of a computed ΔW


class Infeasible(Exception):
    """No ΔW satisfies the edit.

    kind: "conflict"    exact constraints contradict each other (``details["witness"]`` gives
                        coefficients a with Q a ≈ 0 but R a ≠ 0, Q = keys with the preserved
                        subspace removed, R = required output changes); singular values of Q
                        below rtol·‖K‖₂ count as zero;
          "rank"        proven: every exact solution has rank ≥ ``details["required_rank"]``;
          "norm"        proven: every solution has norm ≥ ``details["min_norm"]``;
          "unresolved"  a rank decision falls inside the numerical tolerance band;
          "unsupported" this combination of budgets and constraints cannot be enforced.
    """

    def __init__(self, kind: str, message: str, **details):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.details = details


@dataclass
class Problem:
    W: np.ndarray           # (m, n)
    K: np.ndarray           # (n, pe) exact keys
    R: np.ndarray           # (m, pe) required changes V − W K at the exact keys
    U: np.ndarray           # (n, s) orthonormal basis of the exactly preserved subspace
    Ks: np.ndarray          # (n, ps) soft keys scaled by sqrt(weight)
    Rs: np.ndarray          # (m, ps) soft required changes scaled by sqrt(weight)
    P: np.ndarray | None    # (n, n) Σ weight·C over soft preserves
    M: np.ndarray | None    # (n, n) minimal-change metric; None means identity
    ridge: float
    rank: int | None
    norm: Norm | None

    @property
    def has_soft(self) -> bool:
        return self.Ks.shape[1] > 0 or self.P is not None or self.ridge > 0

    @property
    def A(self) -> np.ndarray:
        return np.hstack([self.K, self.U])

    @property
    def B(self) -> np.ndarray:
        return np.hstack([self.R, np.zeros((self.R.shape[0], self.U.shape[1]))])


def lower(edit: Edit, W, rtol: float = 1e-10) -> Problem:
    """Turn an Edit into arrays, checking the weight shape against the edit."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise SpecError(f"weight of {edit.site} must be 2-D, got shape {W.shape}")
    m, n = W.shape
    if (edit.in_dim, edit.out_dim) != (n, m):
        raise SpecError(
            f"site {edit.site} maps {n}-dim inputs to {m}-dim outputs, but the keys are "
            f"{edit.in_dim}-dim and the values {edit.out_dim}-dim"
        )
    exact = [mp for mp in edit.maps if mp.weight is None]
    soft = [mp for mp in edit.maps if mp.weight is not None]
    K = np.hstack([mp.K.data for mp in exact]) if exact else np.zeros((n, 0))
    V = np.hstack([mp.V.data for mp in exact]) if exact else np.zeros((m, 0))
    Ks = np.hstack([mp.K.data * np.sqrt(mp.weight) for mp in soft]) if soft else np.zeros((n, 0))
    Rs = (
        np.hstack([(mp.V.data - W @ mp.K.data) * np.sqrt(mp.weight) for mp in soft])
        if soft else np.zeros((m, 0))
    )
    bases = [p.target.basis for p in edit.preserve if p.weight is None]
    U = orth(np.hstack(bases), rtol) if bases else np.zeros((n, 0))
    softs = [p for p in edit.preserve if p.weight is not None]
    P = sum(p.weight * p.target.C for p in softs) if softs else None
    M = edit.minimize.C if edit.minimize is not None else None
    ranks = [b.r for b in edit.require if isinstance(b, Rank)]
    norms = [b for b in edit.require if isinstance(b, Norm)]
    if len(norms) > 1:
        raise SpecError("use at most one Norm budget per edit")
    return Problem(W, K, V - W @ K, U, Ks, Rs, P, M, edit.ridge,
                   min(ranks) if ranks else None, norms[0] if norms else None)


def _H(pb: Problem, eta: float) -> np.ndarray:
    n = pb.W.shape[1]
    H = np.eye(n) if pb.M is None else pb.M   # every update below makes a new array
    if pb.Ks.shape[1]:
        H = H + pb.Ks @ pb.Ks.T
    if pb.P is not None:
        H = H + pb.P
    if pb.ridge:
        H = H + pb.ridge * np.eye(n)
    if eta:
        H = H + eta * (np.eye(n) if pb.norm is None or pb.norm.kind == "fro" else pb.M)
    return (H + H.T) / 2


def _chol(H: np.ndarray):
    try:
        return sla.cho_factor(H, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        raise Infeasible(
            "unsupported",
            "the objective matrix (minimize metric + soft terms + ridge) is not positive definite; "
            "use a full-rank Covariance or add ridge",
        ) from None


def _solve_with(pb: Problem, eta: float, rtol: float) -> np.ndarray:
    A, B = pb.A, pb.B
    if not pb.has_soft and pb.M is None and eta == 0:
        if A.shape[1] == 0:
            return np.zeros_like(pb.W)
        # minimum Frobenius norm: ΔW = B A⁺ (min-norm solution of Aᵀ ΔWᵀ = Bᵀ)
        return np.linalg.lstsq(A.T, B.T, rcond=rtol)[0].T
    cf = _chol(_H(pb, eta))
    base = None
    if pb.Ks.shape[1]:
        base = sla.cho_solve(cf, pb.Ks @ pb.Rs.T, check_finite=False).T        # G H⁻¹
    if A.shape[1] == 0:
        return base if base is not None else np.zeros_like(pb.W)
    HiA = sla.cho_solve(cf, A, check_finite=False)
    S = A.T @ HiA
    rhs = B if base is None else B - base @ A
    Lam = rhs @ sla.pinvh((S + S.T) / 2, rtol=rtol)
    delta = Lam @ HiA.T
    return delta if base is None else base + delta


def _norm(pb: Problem, delta: np.ndarray) -> float:
    if pb.norm is None or pb.norm.kind == "fro":
        return float(np.linalg.norm(delta))
    return float(np.sqrt(max(np.trace(delta @ pb.M @ delta.T), 0.0)))


def numerical_rank(X: np.ndarray, rtol: float = RANK_RTOL) -> int:
    if X.size == 0:
        return 0
    s = np.linalg.svd(X, compute_uv=False)
    return int(np.sum(s > rtol * s[0])) if s[0] > 0 else 0


def _check_exact(pb: Problem, rtol: float) -> None:
    """Prove infeasibility of the exact constraints, or of the budgets, when possible."""
    if pb.K.shape[1] == 0:
        return
    Q = pb.K - pb.U @ (pb.U.T @ pb.K)
    s = np.linalg.svd(Q, compute_uv=False)
    tol = rtol * np.linalg.norm(pb.K, 2)            # scale of the keys before the projection
    near = s[(s > tol) & (s <= tol * BAND)]
    if near.size:
        raise Infeasible(
            "unresolved",
            "some keys are nearly dependent once the preserved subspace is removed; whether the "
            "exact maps are consistent is decided inside the numerical tolerance band",
            singular_values=near.tolist(), tolerance=tol,
        )
    # consistent iff the minimum-norm solution of ΔW Q = R reproduces R
    if s.size and s[0] > tol:
        D0 = np.linalg.lstsq(Q.T, pb.R.T, rcond=tol / s[0])[0].T
    else:
        D0 = np.zeros_like(pb.R @ Q.T)
    gap = np.linalg.norm(D0 @ Q - pb.R)
    if gap > FEAS_TOL * max(np.linalg.norm(pb.R), np.finfo(float).tiny):
        _, sv, Vt = np.linalg.svd(Q, full_matrices=True)
        null = Vt[int(np.sum(sv > tol)):]            # coefficient vectors a with Q a ≈ 0
        RN = pb.R @ null.T
        worst = int(np.argmax(np.linalg.norm(RN, axis=0)))
        a = null[worst]
        top = np.argsort(-np.abs(a))[:5]
        raise Infeasible(
            "conflict",
            "a combination of exact keys lies in the preserved subspace (or repeats other keys) "
            "but the maps ask for different outputs",
            witness={int(i): float(a[i]) for i in top if abs(a[i]) > 1e-12},
            output_gap=float(np.linalg.norm(RN[:, worst])),
        )
    if pb.rank is not None and not pb.has_soft:
        sR = np.linalg.svd(pb.R, compute_uv=False)
        tolR = rtol * sR[0] if sR.size and sR[0] > 0 else 0.0
        need = int(np.sum(sR > tolR))
        if need > pb.rank:
            if np.any((sR > tolR) & (sR <= tolR * BAND)):
                raise Infeasible("unresolved", "the rank of the required change sits inside the tolerance band",
                                 singular_values=sR.tolist())
            raise Infeasible("rank", f"every exact solution has rank at least {need}",
                             required_rank=need, budget=pb.rank)
    if pb.norm is not None:
        exact_only = replace(pb, Ks=pb.Ks[:, :0], Rs=pb.Rs[:, :0], P=None, ridge=0.0, rank=None,
                             M=pb.M if pb.norm.kind == "metric" else None)
        least = _norm(pb, _solve_with(exact_only, 0.0, rtol))
        if least > pb.norm.B * (1 + 1e-9):
            raise Infeasible("norm", f"the smallest change meeting the exact constraints has norm {least:.6g}",
                             min_norm=least, budget=pb.norm.B, norm_kind=pb.norm.kind)


def _enforce_norm(pb: Problem, rtol: float) -> tuple[np.ndarray, float]:
    """Smallest added penalty η·‖ΔW‖² (or η·metric) that brings the solution within the budget."""
    target = pb.norm.B
    scale = float(np.trace(_H(pb, 0.0))) / pb.W.shape[1]
    lo, hi = 0.0, max(scale, 1e-12)
    for _ in range(80):
        if _norm(pb, _solve_with(pb, hi, rtol)) <= target:
            break
        lo, hi = hi, hi * 10
    else:
        raise Infeasible("norm", "could not bring the solution within the norm budget",
                         budget=target, norm_kind=pb.norm.kind)
    for _ in range(60):
        mid = np.sqrt(lo * hi) if lo > 0 else hi / 2
        if _norm(pb, _solve_with(pb, mid, rtol)) <= target:
            hi = mid
        else:
            lo = mid
        if lo > 0 and hi / lo < 1 + 1e-6:
            break
    return _solve_with(pb, hi, rtol), hi


def _truncate_rank(pb: Problem, delta: np.ndarray, eta: float) -> np.ndarray:
    """Best rank-r solution in the objective's own metric (soft problems without exact maps)."""
    H = _H(pb, eta)
    r = pb.rank
    if pb.U.shape[1]:
        N = sla.null_space(pb.U.T)
        L = np.linalg.cholesky(N.T @ H @ N)
        Y = (delta @ N) @ L
    else:
        N = None
        L = np.linalg.cholesky(H)
        Y = delta @ L
    Uy, sy, Vty = np.linalg.svd(Y, full_matrices=False)
    Yr = (Uy[:, :r] * sy[:r]) @ Vty[:r]
    X = sla.solve_triangular(L, Yr.T, lower=True, trans="T").T           # Yr L⁻¹
    return X @ N.T if N is not None else X


def solve(edit: Edit, W, rtol: float = 1e-10) -> tuple[np.ndarray, dict]:
    """Return (ΔW, info) for the edit, or raise Infeasible / SpecError."""
    pb = lower(edit, W, rtol)
    _check_exact(pb, rtol)
    delta, eta = _solve_with(pb, 0.0, rtol), 0.0
    info: dict = {}
    if pb.norm is not None and _norm(pb, delta) > pb.norm.B * (1 + 1e-9):
        delta, eta = _enforce_norm(pb, rtol)
        info["norm_penalty"] = eta
    if pb.rank is not None and numerical_rank(delta) > pb.rank:
        if pb.K.shape[1]:
            raise Infeasible("unsupported", "a rank budget cannot be enforced together with exact maps and soft terms",
                             rank=numerical_rank(delta), budget=pb.rank)
        delta = _truncate_rank(pb, delta, eta)
        info["rank_truncated"] = True
        if pb.norm is not None and _norm(pb, delta) > pb.norm.B * (1 + 1e-6):
            raise Infeasible("unsupported", "could not meet the rank and norm budgets together",
                             norm=_norm(pb, delta), budget=pb.norm.B)
    return delta, info
