"""ProgramOptimizer: a torch optimizer whose every weight change comes from an edit program's wir.edit calls.

    optimizer = ProgramOptimizer(model, program=EXAMPLE_PROGRAM)
    for x, y in data:
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        optimizer.observe(loss.item(), val_acc=...)    # optional measurements, which the program can read

A program is Python source that defines ``step(ctx)``. step() runs the closure, then ``step(ctx)``, which changes
weight matrices only through ``ctx.apply(wir.edit(...))``; weightir solves each edit in closed form. ``ctx.refresh()``
runs the closure again so the program can measure an edit, and ``ctx.undo`` takes one back by restoring the weights
it saved. Each edit is checked in the dtype its weights are stored in, and one that fails the check raises
wir.Infeasible("check") instead of being applied. What the program prints is kept (``output``), and so are per-site
edit statistics (``edit_stats``). An exception in the program, or running past the deadline, is raised as
ProgramError. The sites are those of weightir.model.sites; a layer's bias is edited
together with its weight as one matrix [W | b] whose keys end with a 1. Other parameters (norms, embeddings, conv
kernels) are not trained, and a warning names them. The program runs in this process: it is code from a model you
chose, not from an adversary. agentic.loop has an LLM write and improve programs.
"""
from __future__ import annotations

import contextlib
import io
import linecache
import signal
import threading
import time
import traceback
import warnings
import weakref

import numpy as np
import torch

import weightir as wir
from weightir.check import stored_dtype
from weightir.model import add_to_weight, read_weight, sites

PROGRAM_FILE = "<agent program>"
DECAY = 0.95           # running-average decay of each site's input second moment
OUTPUT_CHARS = 20_000  # how much of what the program prints is kept (the end)
OVER_BUDGET = "the run went over its time budget and was stopped"

EXAMPLE_PROGRAM = '''\
def step(ctx):
    for site in ctx.sites:
        K = ctx.keys(site)                    # this step's inputs to the layer; the last row is 1 for the bias
        V = ctx.values(site, step=0.3)        # outputs moved against each example's loss gradient
        ctx.apply(wir.edit(site,
                           map=wir.Soft(K, V, weight=1 / K.count),   # mean squared miss over the batch
                           minimize=ctx.covariance(site)))           # keep the change small on typical inputs
'''


class ProgramError(Exception):
    """The program raised, or the run passed its deadline, at a training step; ``traceback`` holds the program's own
    frames and the error, or what stopped the run."""

    def __init__(self, step: int, traceback_text: str):
        super().__init__(f"the program stopped at step {step}:\n{traceback_text}")
        self.step, self.traceback = step, traceback_text


class _OverBudget(BaseException):
    """Raised by the alarm at the deadline. Not an Exception, so a program's `except Exception` cannot swallow it."""


def load(code: str):
    """The step function of a program; raises on a syntax error or when the program defines no step(ctx)."""
    linecache.cache[PROGRAM_FILE] = (len(code), None, code.splitlines(True), PROGRAM_FILE)
    ns = {"wir": wir, "np": np}
    exec(compile(code, PROGRAM_FILE, "exec"), ns)
    if not callable(ns.get("step")):
        raise NameError("the program must define a function step(ctx)")
    return ns["step"]


def program_traceback(e: BaseException) -> str:
    """The program's own frames and the error line; library frames are left out."""
    te = traceback.TracebackException.from_exception(e)
    te.stack = traceback.StackSummary.from_list([f for f in te.stack if f.filename == PROGRAM_FILE])
    text = "".join(te.format())
    return text if len(text) <= 4000 else "...\n" + text[-4000:]


def _shape(module: torch.nn.Module, transposed: bool, bias: bool) -> tuple[int, int]:
    """(out, in) of the matrix a program edits; `in` counts the bias column."""
    out_dim, in_dim = module.weight.shape[::-1] if transposed else module.weight.shape
    return out_dim, in_dim + bias


def describe(model: torch.nn.Module) -> str:
    """One line per site: its name and the shape (out, in) of the matrix a program edits."""
    lines = []
    for name, (m, transposed, bias) in sites(model).items():
        out_dim, in_dim = _shape(m, transposed, bias)
        lines.append(f"- {name}: {out_dim} x {in_dim}" + (" (last column is the bias)" if bias else ""))
    return "\n".join(lines)


@contextlib.contextmanager
def _alarm_at(deadline: float | None):
    """Interrupt the code inside with _OverBudget at ``deadline`` (time.time()); signals reach the main thread only."""
    if deadline is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum, frame):
        raise _OverBudget()

    old = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(deadline - time.time(), 1e-3))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _residual(check: dict) -> float:
    return check.get("soft_map_residual", check.get("map_residual", 0.0))


class Context:
    """What a program sees at one step. Arrays are float64 copies; changes go through apply() and undo()."""

    def __init__(self, opt: ProgramOptimizer, batch: dict):
        self._opt, self._batch = opt, batch
        self.step = opt._t
        self.losses = opt._losses    # shared with the optimizer; do not modify
        self.metrics = opt._metrics  # shared with the optimizer; do not modify
        self.state = opt._state

    @property
    def sites(self) -> list[str]:
        return [s for s in self._opt._sites if s in self._batch]

    def _get(self, site: str):
        if site not in self._batch:
            raise KeyError(f"{site!r} did not run a training forward this step; sites: {self.sites}")
        return self._batch[site]

    def shape(self, site: str) -> tuple[int, int]:
        return _shape(*self._opt._sites[site])

    def keys(self, site: str) -> wir.Keys:
        return wir.Keys(self._get(site)[0], site)

    def outputs(self, site: str) -> np.ndarray:
        return self._get(site)[1].copy()

    def output_grads(self, site: str) -> np.ndarray:
        return self._get(site)[2].copy()

    def values(self, site: str, step: float) -> wir.Values:
        _, Y, G = self._get(site)
        return wir.Values(Y - step * Y.shape[1] * G, site)

    def covariance(self, site: str, damping: float = 1.0) -> wir.Covariance:
        self._get(site)
        C = self._opt._cov[site]
        M = C.copy()
        M.flat[::len(C) + 1] += damping * np.mean(np.diag(C))
        return wir.Covariance(M, site)

    def weight(self, site: str) -> np.ndarray:
        return self._opt._weight(site)

    def apply(self, edit: wir.Edit) -> wir.Patch:
        return self._opt._apply(edit)

    def undo(self, patch: wir.Patch) -> None:
        self._opt._undo(patch)

    def refresh(self) -> float:
        opt = self._opt
        if opt._closure is None:
            raise RuntimeError("ctx.refresh() needs the training loop to pass a closure: optimizer.step(closure)")
        with torch.enable_grad():
            loss = opt._closure()
        self._batch = opt._collect(update_cov=False)
        return loss.item()


class ProgramOptimizer(torch.optim.Optimizer):
    """Trains ``model`` by running the edit program ``program`` (Python source defining step(ctx)) at every step.

    ``edit_stats`` maps each site to [edits, sum of ||ΔW||_F, sum of map residuals] over the edits that were kept;
    ``output`` is the end of what the program printed. deadline: a time.time() at which the run stops with
    ProgramError (a trial's time budget), even in the middle of a program call; None for no limit.
    """

    def __init__(self, model: torch.nn.Module, program: str, *, deadline: float | None = None):
        self._sites = sites(model)
        if not self._sites:
            raise ValueError("the model has no trainable Linear or Conv1D layer")
        trained = {id(m.weight) for m, _, _ in self._sites.values()}
        trained |= {id(m.bias) for m, _, b in self._sites.values() if b}
        others = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in trained]
        if others:
            warnings.warn(f"ProgramOptimizer does not train {len(others)} parameters: {', '.join(others[:5])}"
                          + (" ..." if len(others) > 5 else ""), stacklevel=2)
        super().__init__([p for p in model.parameters() if id(p) in trained], {})

        self._fn, self.deadline = load(program), deadline
        self._dtype = {name: stored_dtype(m.weight.dtype) for name, (m, _, _) in self._sites.items()}
        self._saved: dict[str, list] = {}  # site -> [(patch, weight before, bias before)] applied this step
        self._t = 0
        self._losses: list[float] = []
        self._metrics: dict[str, list[tuple[int, float]]] = {}
        self._cov: dict[str, np.ndarray] = {}
        self._rec: dict[str, list] = {name: [] for name in self._sites}
        self._closure = None
        self._state: dict = {}
        self._stdout = io.StringIO()
        self.edit_stats: dict[str, list] = {name: [0, 0.0, 0.0] for name in self._sites}
        self.ctx: Context | None = None

        ref = weakref.ref(self)
        for name, (m, _, _) in self._sites.items():
            def record(module, args, out, name=name):
                opt = ref()
                if opt is None or not torch.is_grad_enabled() or not out.requires_grad:
                    return
                row = [args[0].detach(), out.detach(), None]
                opt._rec[name].append(row)
                out.register_hook(lambda g, row=row: row.__setitem__(2, g.detach()))
            weakref.finalize(self, m.register_forward_hook(record).remove)

    @property
    def output(self) -> str:
        return self._stdout.getvalue()[-OUTPUT_CHARS:]

    def observe(self, loss: float, **metrics: float) -> None:
        """Record the step's training loss and any other measurements (e.g. val_acc=...), for ctx.losses and ctx.metrics."""
        self._losses.append(float(loss))
        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append((self._t, float(value)))

    @torch.no_grad()
    def step(self, closure=None):
        if self.deadline is not None and time.time() > self.deadline:
            raise ProgramError(self._t + 1, OVER_BUDGET)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._closure = closure
        self._t += 1
        self._saved = {}
        self.ctx = Context(self, self._collect())
        try:
            with _alarm_at(self.deadline), contextlib.redirect_stdout(self._stdout):
                self._fn(self.ctx)
        except _OverBudget:
            raise ProgramError(self._t, OVER_BUDGET) from None
        except Exception as e:
            raise ProgramError(self._t, program_traceback(e)) from e
        if self._stdout.tell() > 2 * OUTPUT_CHARS:
            self._stdout = io.StringIO(self.output)
            self._stdout.seek(0, io.SEEK_END)
        return loss

    # -- data, weights, edits -------------------------------------------------------------------------------------

    def _collect(self, update_cov: bool = True) -> dict:
        """Inputs (with a 1 row for a bias), outputs and output gradients of the forwards since the last collect."""
        batch = {}
        for name, rows in self._rec.items():
            rows = [r for r in rows if r[2] is not None]          # forwards that took part in the backward pass
            self._rec[name] = []
            if not rows:
                continue
            _, _, bias = self._sites[name]
            flat = [torch.cat([r[i].reshape(-1, r[i].shape[-1]) for r in rows]).cpu().double().numpy().T
                    for i in range(3)]
            X, Y, G = flat
            if bias:
                X = np.vstack([X, np.ones((1, X.shape[1]))])
            if update_cov:
                C = X @ X.T / X.shape[1]
                self._cov[name] = C if name not in self._cov else DECAY * self._cov[name] + (1 - DECAY) * C
            batch[name] = (X, Y, G)
        return batch

    def _weight(self, name: str) -> np.ndarray:
        m, transposed, bias = self._sites[name]
        W = read_weight(m.weight, transposed)
        return np.hstack([W, m.bias.detach().cpu().double().numpy()[:, None]]) if bias else W

    def _add(self, name: str, delta: np.ndarray) -> None:
        """Add an (out, in [+1]) change to the weight and bias of a site."""
        m, transposed, bias = self._sites[name]
        n_in = delta.shape[1] - bias
        add_to_weight(m.weight, transposed, delta[:, :n_in])
        if bias:
            with torch.no_grad():
                m.bias.copy_((m.bias.detach().cpu().double() + torch.from_numpy(delta[:, n_in])).to(m.bias.dtype))

    def _tally(self, patch: wir.Patch, sign: int) -> None:
        stats = self.edit_stats[patch.site]
        stats[0] += sign
        stats[1] += sign * patch.check["fro"]
        stats[2] += sign * _residual(patch.check)

    def _apply(self, edit: wir.Edit) -> wir.Patch:
        if edit.site not in self._sites:
            raise wir.SpecError(f"{edit.site!r} is not a site; sites: {list(self._sites)}")
        m, _, bias = self._sites[edit.site]
        patch = wir.compile(edit, self._weight(edit.site), dtype=self._dtype[edit.site])
        self._saved.setdefault(edit.site, []).append(
            (patch, m.weight.detach().clone(), m.bias.detach().clone() if bias else None))
        self._add(edit.site, patch.delta)
        self._tally(patch, 1)
        return patch

    def _undo(self, patch: wir.Patch) -> None:
        """Restore the weights saved when ``patch`` was applied: exact, whatever rounding the apply did."""
        saved = self._saved.get(patch.site, [])
        if not saved or saved[-1][0] is not patch:
            raise ValueError("ctx.undo takes back the last patch applied to a site in this step; take back the "
                             "later patches of that site first")
        _, weight, bias = saved.pop()
        m = self._sites[patch.site][0]
        with torch.no_grad():
            m.weight.copy_(weight)
            if bias is not None:
                m.bias.copy_(bias)
        self._tally(patch, -1)
