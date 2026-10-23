"""Optimizers by name. A training script creates its optimizer in one line, `optimizer = OPTIMIZERS[name](model)`, so
switching between a standard torch optimizer and a weightir rule changes nothing else; adding one is one more line.
Every optimizer here takes torch's closure form, `loss = optimizer.step(closure)`.

The wir-* rules leave biases to `default`, set to match the standard optimizer each one is compared with. The runs
named "agent-<model>" (the best program an LLM wrote) and "adam-tuned" (Adam with a searched learning rate) come from
experiments/search.py, not from this table.
"""
import torch

from agentic.program import EXAMPLE_PROGRAM, ProgramOptimizer
from weightir.train import Optimizer, Rule


class MuonAdamW:
    """torch.optim.Muon (default momentum 0.95, Nesterov) on the 2-D weights and AdamW on the other parameters, since
    Muon takes only 2-D parameters."""

    def __init__(self, model: torch.nn.Module, lr: float = 0.02, adamw_lr: float = 1e-3):
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizers = [torch.optim.Muon([p for p in params if p.ndim == 2], lr=lr, weight_decay=0),
                           torch.optim.AdamW([p for p in params if p.ndim != 2], lr=adamw_lr, weight_decay=0)]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()


OPTIMIZERS = {
    "adam": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    "sgd": lambda model: torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
    "muon": lambda model: MuonAdamW(model, lr=0.02),
    "wir-adam": lambda model: Optimizer(model, Rule(direction="adam", lr=1e-3)),
    "wir-orth": lambda model: Optimizer(model, Rule(direction="orth", lr=0.02), default=Rule(lr=1e-3)),
    # momentum SGD on whitened layer inputs (W' = W M^½), hence sgd's lr; M = C + mean(diag C)·I (Rule's default damping)
    "wir-cov": lambda model: Optimizer(model, Rule(direction="momentum", metric="covariance", lr=0.05),
                                       default=Rule(direction="momentum", lr=0.05)),
    # the example edit program: the LLM's starting point, with no LLM
    "agent-fixed": lambda model: ProgramOptimizer(model, program=EXAMPLE_PROGRAM),
}
