"""weightir.train.Optimizer replaces a torch optimizer in one line, so a wrong step would silently change every
training result: these tests pin the baseline (AdamW), the solved problem, the preserved subspace and the budgets."""
import copy

import pytest
import torch
from transformers.pytorch_utils import Conv1D

from weightir.train import Optimizer, Rule

LR = {"sgd": 0.1, "momentum": 0.02, "adam": 1e-2, "sign": 1e-3, "orth": 2e-2}


def _orthonormal(n: int, s: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(n, s, generator=g, dtype=torch.float64)).Q


class TwoSites(torch.nn.Module):
    """A Linear site and a GPT-2 Conv1D site, which stores W transposed as (in, out); biases frozen."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(12, 16)
        self.proj = Conv1D(8, 16)
        for m in (self.fc, self.proj):
            m.bias.requires_grad_(False)

    def forward(self, x):
        return self.proj(torch.tanh(self.fc(x)))


@pytest.mark.parametrize("rule, reference", [
    (Rule(direction="adam", lr=1e-2, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1),
     lambda ps: torch.optim.AdamW(ps, lr=1e-2, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)),
    (Rule(direction="momentum", lr=1e-2, betas=(0.8, 0.95)), lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.8)),
    (Rule(direction="sgd", lr=1e-2), lambda ps: torch.optim.SGD(ps, lr=1e-2)),
], ids=["adam", "momentum", "sgd"])
def test_unconstrained_rules_match_torch(rule, reference):
    """Without constraints "adam" is AdamW and "momentum"/"sgd" are torch SGD, step for step and under an LR
    scheduler, so the one-line switch from a torch optimizer compares like with like."""
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.LayerNorm(8), torch.nn.Linear(8, 3))
    ref = ref.double()
    ours = copy.deepcopy(ref)
    opts = [reference(ref.parameters()), Optimizer(ours, rule, default=rule)]
    scheds = [torch.optim.lr_scheduler.StepLR(o, step_size=5, gamma=0.5) for o in opts]
    X, Y = torch.randn(32, 6, dtype=torch.float64), torch.randn(32, 3, dtype=torch.float64)
    for _ in range(20):
        for model, opt, sched in zip((ref, ours), opts, scheds):
            opt.zero_grad()
            ((model(X) - Y) ** 2).mean().backward()
            opt.step()
            sched.step()
        for a, b in zip(ref.parameters(), ours.parameters()):
            torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("budgets", [{}, {"rank": 2, "max_norm": 0.05}], ids=["plain", "rank+norm"])
def test_covariance_step_solves_its_problem(budgets):
    """One step is argmin ⟨D, ΔW⟩ + tr(ΔW M ΔWᵀ)/(2·lr) s.t. ΔW U = 0, solved here from its KKT system; with budgets,
    the best rank-r approximation in the M norm (computed through M^½, not Cholesky), scaled onto ‖ΔW‖_M = max_norm;
    then the decoupled decay −lr·wd·W(I − UUᵀ), which stays outside the metric and the budgets."""
    torch.manual_seed(0)
    n_in, n_out, s, lr, wd, damping = 10, 7, 3, 0.5, 0.1, 1e-2
    layer = torch.nn.Linear(n_in, n_out, bias=False).double()
    U = _orthonormal(n_in, s, seed=1)
    opt = Optimizer(layer, Rule(direction="sgd", lr=lr, weight_decay=wd, metric="covariance", damping=damping,
                                preserve=U, **budgets))
    X = torch.randn(4, 16, n_in, dtype=torch.float64) * torch.linspace(0.1, 2.0, n_in, dtype=torch.float64)
    W0 = layer.weight.detach().clone()
    ((layer(X) - 1) ** 2).mean().backward()
    D = layer.weight.grad.clone()
    opt.step()

    x = X.reshape(-1, n_in)                                       # every token is one input
    C = x.T @ x / len(x)
    M = C + damping * C.diagonal().mean() * torch.eye(n_in, dtype=torch.float64)
    K = torch.zeros(n_in + s, n_in + s, dtype=torch.float64)      # one KKT system per row of ΔW
    K[:n_in, :n_in], K[:n_in, n_in:], K[n_in:, :n_in] = M / lr, U, U.T
    want = torch.linalg.solve(K, torch.cat([-D.T, torch.zeros(s, n_out, dtype=torch.float64)]))[:n_in].T
    if budgets:
        lam, V = torch.linalg.eigh(M)
        S = V @ torch.diag(lam.sqrt()) @ V.T                      # ‖ΔW‖_M = ‖ΔW S‖_F
        u, sv, vh = torch.linalg.svd(want @ S)
        Y = u[:, :budgets["rank"]] @ torch.diag(sv[:budgets["rank"]]) @ vh[:budgets["rank"]]
        assert Y.norm() > budgets["max_norm"]                     # the trust region is active
        want = Y @ torch.linalg.inv(S) * (budgets["max_norm"] / Y.norm())
    want = want - lr * wd * (W0 - W0 @ U @ U.T)
    torch.testing.assert_close(layer.weight.detach() - W0, want, rtol=1e-9, atol=1e-12)


def test_orth_orthogonalizes_the_projected_momentum():
    """Muon under preserve steps along NS(G P⊥), the steepest step inside the constraint, not NS(G) P⊥, which also
    keeps ΔW U = 0 and so passes every other test."""
    torch.manual_seed(0)
    a, b = torch.nn.Linear(16, 12, bias=False), torch.nn.Linear(16, 12, bias=False)
    U = _orthonormal(16, 4, seed=1)
    opts = [Optimizer(a, Rule(direction="orth", lr=0.1, preserve=U)), Optimizer(b, Rule(direction="orth", lr=0.1))]
    W0 = [a.weight.detach().clone(), b.weight.detach().clone()]
    G = torch.randn(12, 16, dtype=torch.float64)
    a.weight.grad, b.weight.grad = G.float(), (G - G @ U @ U.T).float()
    for opt in opts:
        opt.step()
    torch.testing.assert_close(a.weight.detach() - W0[0], b.weight.detach() - W0[1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("metric", ["identity", "covariance"])
@pytest.mark.parametrize("direction", list(LR))
def test_preserved_sites_never_move_on_U(direction, metric):
    """After 300 steps with weight decay the realized ΔW of both layouts is orthogonal to U up to one float32 rounding
    (about 2e-8; 0.6 without preserve, 2e-7 if rounding piled up step after step), and a toy regression still fits."""
    torch.manual_seed(0)
    net = TwoSites()
    X = torch.randn(256, 12)
    Y = torch.tanh(X @ torch.randn(12, 16) / 12**0.5) @ torch.randn(16, 8) / 2
    U1, U2 = _orthonormal(12, 4, seed=1), _orthonormal(16, 5, seed=2)
    kw = dict(direction=direction, lr=LR[direction], weight_decay=0.01, metric=metric)
    opt = Optimizer(net, [Rule(match="fc", preserve=U1, **kw), Rule(match="proj", preserve=U2, **kw)])
    W1, W2 = net.fc.weight.detach().clone(), net.proj.weight.detach().clone()
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    for dW, U in [(net.fc.weight.detach() - W1, U1), ((net.proj.weight.detach() - W2).T, U2)]:   # (out, in)
        dW = dW.double()
        assert (dW @ U).norm() < 1e-7 * dW.norm()
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("metric", ["identity", "covariance"])
def test_rank_and_norm_budgets_hold_every_step(metric):
    """rank(ΔW) ≤ r and ‖ΔW‖_M ≤ max_norm on every step, with M rebuilt here from the grad-enabled inputs only."""
    torch.manual_seed(0)
    n_in, n_out, r, max_norm = 10, 8, 2, 0.05
    rule = Rule(direction="adam", lr=0.1, metric=metric, decay=0.9, refresh=3, preserve=_orthonormal(n_in, 2, seed=1),
                rank=r, max_norm=max_norm)
    layer = torch.nn.Linear(n_in, n_out, bias=False).double()
    opt = Optimizer(layer, rule)
    teacher = torch.randn(n_out, n_in, dtype=torch.float64)
    scale = torch.linspace(0.2, 2.0, n_in, dtype=torch.float64)
    eye = torch.eye(n_in, dtype=torch.float64)
    C, hit = None, False
    for t in range(30):
        x = torch.randn(64, n_in, dtype=torch.float64) * scale
        with torch.no_grad():
            layer(10 * x)                                         # an evaluation forward: not part of the metric
        W0 = layer.weight.detach().clone()
        opt.zero_grad()
        ((layer(x) - x @ teacher.T) ** 2).mean().backward()
        opt.step()
        dW = layer.weight.detach() - W0
        C = x.T @ x / len(x) if C is None else rule.decay * C + (1 - rule.decay) * x.T @ x / len(x)
        if t % rule.refresh == 0:
            M = C + rule.damping * C.diagonal().mean() * eye if metric == "covariance" else eye
        size = torch.trace(dW @ M @ dW.T).sqrt()
        sv = torch.linalg.svdvals(dW)
        assert sv[r] <= 1e-10 * sv[0]
        assert size <= max_norm * (1 + 1e-9)
        hit |= bool(size >= max_norm * (1 - 1e-9))
    assert hit                                                    # the trust region was active
