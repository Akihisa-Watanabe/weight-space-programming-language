"""Re-check a solved edit against its constraints, on the original data, independently of the solver."""
from __future__ import annotations

import numpy as np

from weightir.solve import RANK_RTOL
from weightir.spec import Edit, Rank

TINY = np.finfo(float).tiny
EPS = {"float32": 2.0 ** -23, "float16": 2.0 ** -10, "bfloat16": 2.0 ** -7}  # machine epsilon of each storage dtype


def stored_dtype(dtype) -> str | None:
    """The name check() uses for a tensor dtype (e.g. torch.bfloat16 -> "bfloat16"); None for float64, which
    stores the solution exactly."""
    name = str(dtype).removeprefix("torch.")
    return name if name in EPS else None


def round_to(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round float64 values to ``dtype`` ("float32", "float16", "bfloat16") and back to float64."""
    if dtype == "bfloat16":
        bits = np.asarray(x, dtype=np.float32).view(np.uint32)
        bits = (bits + (((bits >> 16) & 1) + 0x7FFF)) & 0xFFFF0000   # round to nearest even
        return bits.view(np.float32).astype(np.float64)
    if dtype in ("float32", "float16"):
        return np.asarray(x, dtype=dtype).astype(np.float64)
    raise ValueError(f"unknown dtype {dtype!r}")


def _rel_cols(resid: np.ndarray, ref: np.ndarray) -> float:
    if resid.shape[1] == 0:
        return 0.0
    return float(np.max(np.linalg.norm(resid, axis=0) / np.maximum(np.linalg.norm(ref, axis=0), 1e-12)))


def check(edit: Edit, W, delta, dtype: str | None = None, tol: float = 1e-6) -> dict:
    """Residuals of every constraint and budget for ``W + delta``.

    map_residual         max over exact keys of ‖(W+ΔW)k − v‖ / ‖v‖
    soft_map_residual    the same for soft keys (not a constraint; reported for reading)
    preserve_residual    max over exactly preserved Subspaces of ‖ΔW U‖₂ / ‖ΔW‖₂ (0 when ΔW U = 0)
    preserve_data_change max over exactly preserved Subspaces that keep their data X of ‖ΔW X‖ / ‖W X‖
    with ``dtype``: map_residual and the preserved outputs' change after storing W and W + ΔW in that
    dtype, where rounding makes the stored change differ from ΔW.
    ``ok`` is True when exact maps, exact preserves and budgets hold to ``tol`` in float64 and, with
    ``dtype``, the exact maps also hold to max(tol, √eps) after rounding to it: the stored outputs keep
    at least half of the dtype's digits. Rounding alone can cost more than a few ulps where weights
    cancel (large weights, small outputs), so an ulp count would refuse ordinary float32 edits, while
    a change that bfloat16 rounds away misses by far more than √eps.
    """
    W = np.asarray(W, dtype=np.float64)
    D = np.asarray(delta, dtype=np.float64)
    Wn = W + D
    s = np.linalg.svd(D, compute_uv=False) if D.size else np.zeros(0)   # one SVD gives the rank and ‖ΔW‖₂
    out: dict = {"fro": float(np.linalg.norm(D)),
                 "rank": int(np.sum(s > RANK_RTOL * s[0])) if s.size and s[0] > 0 else 0}
    if edit.minimize is not None:
        out["metric"] = float(np.trace(D @ edit.minimize.C @ D.T))

    def maps(exact: bool):
        ms = [m for m in edit.maps if (m.weight is None) == exact]
        if not ms:
            return None, None
        return np.hstack([m.K.data for m in ms]), np.hstack([m.V.data for m in ms])

    Ke, Ve = maps(True)
    Ks, Vs = maps(False)
    if Ke is not None:
        out["map_residual"] = _rel_cols(Wn @ Ke - Ve, Ve)
    if Ks is not None:
        out["soft_map_residual"] = _rel_cols(Wn @ Ks - Vs, Vs)

    exact_sub = [p.target for p in edit.preserve if p.weight is None]
    if exact_sub:
        dnorm = max(s[0] if s.size else 0.0, TINY)
        out["preserve_residual"] = max(float(np.linalg.norm(D @ u.basis, 2) / dnorm) for u in exact_sub)
    data = [u for u in exact_sub if u.source is not None]
    if data:
        out["preserve_data_change"] = max(
            float(np.linalg.norm(D @ u.source) / max(np.linalg.norm(W @ u.source), TINY)) for u in data
        )
    soft_cov = [(p.weight, p.target) for p in edit.preserve if p.weight is not None]
    if soft_cov:
        out["soft_preserve_penalty"] = float(sum(w * np.trace(D @ c.C @ D.T) for w, c in soft_cov))

    budgets_ok = True
    for b in edit.require:
        if isinstance(b, Rank):
            budgets_ok &= out["rank"] <= b.r
        else:
            size = out["fro"] if b.kind == "fro" else float(np.sqrt(max(out.get("metric", 0.0), 0.0)))
            budgets_ok &= size <= b.B * (1 + 1e-6)
    out["budgets_ok"] = bool(budgets_ok)

    if dtype is not None and (Ke is not None or exact_sub):
        Wc, Wnc = round_to(W, dtype), round_to(Wn, dtype)
        Dc = Wnc - Wc
        if Ke is not None:
            out[f"map_residual_{dtype}"] = _rel_cols(Wnc @ Ke - Ve, Ve)
        if exact_sub:
            out[f"preserve_change_{dtype}"] = max(
                float(np.linalg.norm(Dc @ u.basis) / max(np.linalg.norm(Wc @ u.basis), TINY)) for u in exact_sub
            )

    stored = out.get(f"map_residual_{dtype}", 0.0) <= max(tol, EPS[dtype] ** 0.5) if dtype is not None else True
    out["ok"] = bool(
        out.get("map_residual", 0.0) <= tol and out.get("preserve_residual", 0.0) <= tol and budgets_ok and stored
    )
    return out
