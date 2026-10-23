# Adapted from github.com/KellerJordan/Muon (MIT License, Copyright (c) 2024 Keller Jordan): muon.py; and github.com/ShipengWang/Adam-NSCL (MIT License, Copyright (c) 2021 Shipeng Wang): optim/adam_svd.py
"""Training-time update rules: a torch optimizer whose step on each weight matrix solves a small constrained problem.

The edit constraints of weightir.spec become rules applied at every step: ``preserve`` is ``Exact(U)``, the
covariance metric is ``minimize=C`` estimated online, and ``rank``/``max_norm`` are ``Rank``/``Norm(kind="metric")``
budgets on each step. Muon's Newton–Schulz is ported; the update is projected after the optimizer's own transform,
weight decay included, as in Adam-NSCL.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch

from weightir.model import sites
from weightir.spec import Subspace

DIRECTIONS = ("sgd", "momentum", "adam", "sign", "orth")
METRICS = ("identity", "covariance")


@dataclass
class Rule:
    """How every weight matrix whose module name matches `match` is updated at each step:
       ΔW = argmin ⟨D, ΔW⟩ + 1/(2·lr)·tr(ΔW M ΔWᵀ)   s.t. ΔW U = 0,  ‖ΔW‖_M ≤ max_norm,  rank(ΔW) ≤ rank
       D comes from `direction` (computed from the raw gradient), M from `metric`, U from `preserve`."""
    match: str = "*"              # fnmatch pattern on module names
    direction: str = "adam"       # "sgd" | "momentum" | "adam" | "sign" (Lion) | "orth" (Muon: Newton–Schulz of m)
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0     # decoupled, outside the metric and budgets: adds −lr·weight_decay·W(I − UUᵀ)
    metric: str = "identity"      # "identity" | "covariance" (running E[x xᵀ] of the module input: K-FAC input factor)
    decay: float = 0.95           # running-average decay of the covariance
    damping: float = 1.0          # M = C + damping·mean(diag C)·I; small values blow up low-variance input directions
    refresh: int = 10             # recompute the factorization of M every `refresh` steps
    preserve: object = None       # weightir Subspace or (in_dim, s) orthonormal array: inputs whose outputs stay fixed
    max_norm: float | None = None # trust region on ‖ΔW‖_M
    rank: int | None = None       # low-rank step, truncated in the metric


def _elementwise(rule: Rule) -> bool:
    """True when the rule never uses the matrix structure, so its step is taken per entry in the weight's dtype."""
    return (rule.direction != "orth" and rule.metric == "identity" and rule.preserve is None
            and rule.rank is None and rule.max_norm is None)


class Optimizer(torch.optim.Optimizer):
    """Steps the Linear and Conv1D weights of ``model`` by ``rules`` and every other trainable parameter by ``default``.

    A weight takes the first rule whose ``match`` fits its module name. Weights that no rule matches, weights tied to
    another parameter, biases, norms and embeddings take the elementwise ``default`` (Adam at the first rule's lr
    unless given). ``param_groups[i]`` holds the weights of ``rules[i]`` and the last group the rest, so LR schedulers
    act on each rule's lr. Pass ``model.hf`` for a weightir Model: Subspace sites are module names.

    The covariance metric is estimated from the inputs of every grad-enabled forward since the previous step (padding
    tokens included); its first step uses that step's inputs alone. A preserved site must have a frozen bias, since a
    bias step would move the outputs on span(U). Its weight is written as W + ΔW + r rounded once, r being the rounding
    left from its previous step, so rounding cannot pile up in span(U): the total change leaks one rounding however
    long training runs. Weights must be float32 or float64; ``rank`` and ``max_norm`` bound the float64 step. That
    float64 math runs on the weight's device, or on the CPU for MPS, which has no float64.
    """

    def __init__(self, model: torch.nn.Module, rules=Rule(), *, default: Rule | None = None):
        rules = [rules] if isinstance(rules, Rule) else list(rules)
        if not rules:
            raise ValueError("rules is empty")
        default = default if default is not None else Rule(direction="adam", lr=rules[0].lr)
        for r in (*rules, default):
            if r.direction not in DIRECTIONS:
                raise ValueError(f"unknown direction {r.direction!r}; expected one of {DIRECTIONS}")
            if r.metric not in METRICS:
                raise ValueError(f"unknown metric {r.metric!r}; expected one of {METRICS}")
        if not _elementwise(default):
            raise ValueError("default must be elementwise: no 'orth' direction, 'covariance' metric, preserve, "
                             "rank or max_norm")

        groups: list[list[torch.nn.Parameter]] = [[] for _ in rules]
        modules = {}
        self._sites: dict[torch.Tensor, tuple] = {}  # weight -> (name, transposed, U, device of its float64 math)
        for name, (module, transposed, has_bias) in sites(model).items():
            w = module.weight
            i = next((i for i, r in enumerate(rules) if fnmatchcase(name, r.match)), None)
            if i is None:
                continue
            if w.dtype not in (torch.float32, torch.float64):
                raise ValueError(f"{name}: weight is {w.dtype}, but rules need float32 or float64 weights "
                                 "(lower precision rounds away small steps and leaks them into span(U))")
            dev = torch.device("cpu") if w.device.type == "mps" else w.device  # MPS has no float64
            U = _basis(rules[i].preserve, name, w.shape[0] if transposed else w.shape[1], dev)
            if U is not None and has_bias:
                raise ValueError(f"{name}: freeze its bias (bias.requires_grad_(False)); a bias step moves the "
                                 "outputs on the preserved inputs")
            groups[i].append(w)
            modules[w] = module
            self._sites[w] = (name, transposed, U, dev)
        for i, (r, g) in enumerate(zip(rules, groups)):
            if not g:
                raise ValueError(f"rule {i} (match={r.match!r}) matches no trainable Linear/Conv1D weight that an "
                                 "earlier rule has not taken (tied weights are left to default)")
        taken = {id(w) for g in groups for w in g}
        rest = [p for p in model.parameters() if p.requires_grad and id(p) not in taken]
        param_groups = [{"params": g, "lr": r.lr, "rule": r} for r, g in zip(rules, groups)]
        if rest:
            param_groups.append({"params": rest, "lr": default.lr, "rule": default})
        super().__init__(param_groups, {})

        # The hooks hold the optimizer weakly and are removed with it, so a dropped optimizer frees its state.
        ref = weakref.ref(self)
        for group in self.param_groups[:len(rules)]:
            if group["rule"].metric != "covariance":
                continue
            for w in group["params"]:
                def record(module, args, w=w):
                    opt = ref()
                    if opt is not None and torch.is_grad_enabled():
                        opt._record(w, args[0])
                weakref.finalize(self, modules[w].register_forward_pre_hook(record).remove)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            rule, lr = group["rule"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["step"] = st.get("step", 0) + 1
                if _elementwise(rule):
                    D = _direction(rule, st, p.grad)
                    if rule.weight_decay:
                        p.mul_(1 - lr * rule.weight_decay)
                    p.add_(D, alpha=-lr)
                else:
                    self._matrix_step(p, rule, lr, st)
        return loss

    def _matrix_step(self, p: torch.Tensor, rule: Rule, lr: float, st: dict) -> None:
        """Solve the rule's problem for this weight in float64, in the (out, in) layout, and write W + ΔW."""
        name, transposed, U, dev = self._sites[p]

        def proj(X):                                                    # X (I − UUᵀ): no change on span(U)
            return X if U is None else X - (X @ U) @ U.T

        Wv = p.T if transposed else p                                   # (out, in) view of the weight
        # .to(dev) before .double(): MPS converts wrongly when one .to() moves and converts
        D = _direction(rule, st, p.grad.T if transposed else p.grad).to(dev).double()
        if rule.direction == "orth":
            D = _newton_schulz5(proj(D)).double() * max(1.0, Wv.shape[0] / Wv.shape[1]) ** 0.5
        W = Wv.to(dev).double()
        L = None                                                        # M = L Lᵀ; None for the identity
        if rule.metric == "covariance":
            L, P = self._factor(rule, st, U, name, dev)
            step = -lr * (D @ P)
        else:
            step = -lr * proj(D)
        if rule.rank is not None and rule.rank < min(step.shape):
            A = torch.linalg.svd(step if L is None else step @ L, full_matrices=False).U[:, :rule.rank]
            step = A @ (A.T @ step)                                     # = (ΔW* L)_r L⁻¹, so ΔW U = 0 still holds
        if rule.max_norm is not None:
            size = torch.linalg.norm(step if L is None else step @ L)  # ‖ΔW‖_M
            if size > rule.max_norm:
                step = step * (rule.max_norm / size)
        if rule.weight_decay:                                           # decoupled, as in AdamW
            step = step - lr * rule.weight_decay * proj(W)
        new = W + step
        if U is not None and "carry" in st:
            new += st["carry"]
        Wv.copy_(new)
        if U is not None:
            st["carry"] = (new - Wv.to(dev).double()).to(p.dtype)

    def _factor(self, rule: Rule, st: dict, U: torch.Tensor | None, name: str, dev: torch.device):
        """Fold the recorded inputs into C and, every ``refresh`` steps, refactor M = C + damping·mean(diag C)·I.

        Returns L = chol(M) and Π = M⁻¹ − M⁻¹U(UᵀM⁻¹U)⁻¹UᵀM⁻¹ (M⁻¹ without preserve), so that ΔW* = −lr·D Π."""
        if "n" in st:
            C = st.pop("xx").to(dev).double() / st.pop("n")
            st["C"] = C if "C" not in st else rule.decay * st["C"] + (1 - rule.decay) * C
        if "C" not in st:
            raise RuntimeError(f"{name}: no input was recorded for the covariance metric; run the forward pass "
                               "with gradients enabled after creating the optimizer")
        if (st["step"] - 1) % rule.refresh == 0:
            C = st["C"]
            if not torch.isfinite(C).all():
                raise FloatingPointError(f"{name}: the recorded inputs are not finite; training has diverged")
            M = C + rule.damping * C.diagonal().mean() * torch.eye(len(C), dtype=C.dtype, device=C.device)
            L = torch.linalg.cholesky(M)
            P = torch.cholesky_inverse(L)
            if U is not None:
                PU = P @ U
                P = P - PU @ torch.linalg.solve(U.T @ PU, PU.T)
            st["L"], st["P"] = L, P
        return st["L"], st["P"]

    def _record(self, w: torch.Tensor, x: torch.Tensor) -> None:
        """Add the module inputs x (..., in) to the running sum of x xᵀ for weight w."""
        x = x.detach().reshape(-1, x.shape[-1]).to(w.dtype)
        st = self.state[w]
        if "n" in st:
            st["xx"] += x.T @ x
            st["n"] += len(x)
        else:
            st["xx"], st["n"] = x.T @ x, len(x)


def _direction(rule: Rule, st: dict, g: torch.Tensor) -> torch.Tensor:
    """D from the raw gradient g; the moment buffers live in ``st``. Do not modify the result in place."""
    if rule.direction == "sgd":
        return g
    b1, b2 = rule.betas
    if "m" not in st:
        st["m"] = torch.zeros_like(g)
    m = st["m"]
    if rule.direction in ("momentum", "orth"):
        return m.mul_(b1).add_(g)
    if rule.direction == "sign":                                        # Lion
        c = m.lerp(g, 1 - b1)
        m.lerp_(g, 1 - b2)
        return c.sign()
    if "v" not in st:
        st["v"] = torch.zeros_like(g)
    v = st["v"]
    m.lerp_(g, 1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    t = st["step"]
    return (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + rule.eps)


def _newton_schulz5(G: torch.Tensor) -> torch.Tensor:
    """Muon's quintic Newton–Schulz iteration (5 steps) in float32: roughly the polar factor of G, singular values
    in about [0.5, 1.5]. Each step multiplies by a polynomial in X Xᵀ, so G U = 0 implies (result) U = 0."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def _basis(preserve, name: str, in_dim: int, device) -> torch.Tensor | None:
    """The preserved directions as a float64 (in_dim, s) tensor, re-orthonormalized to float64 precision."""
    if preserve is None:
        return None
    if isinstance(preserve, Subspace):
        if preserve.site != name:
            raise ValueError(f"{name}: the preserve Subspace belongs to site {preserve.site!r}")
        preserve = preserve.basis
    U = torch.as_tensor(preserve).cpu().double()
    if U.ndim != 2 or U.shape[0] != in_dim:
        raise ValueError(f"{name}: preserve basis has shape {tuple(U.shape)}, but the input dim is {in_dim}")
    if not torch.allclose(U.T @ U, torch.eye(U.shape[1], dtype=torch.float64), rtol=0, atol=1e-6):
        raise ValueError(f"{name}: preserve basis must have orthonormal columns (use weightir.span or "
                         "weightir.subspace)")
    return torch.linalg.qr(U).Q.to(device)
