"""The agent layer. A wrong row, scale or orientation in what a program sees, or in how its edit is written back,
would silently change every training run, so both are checked against direct computations. The outer loop runs on a
scripted LLM: what it is told after each trial, and which trial it keeps."""
import math

import numpy as np
import pytest
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

import weightir as wir
from agentic.llm import Reply
from agentic.loop import improve
from agentic.program import EXAMPLE_PROGRAM, ProgramError, ProgramOptimizer

NOOP = "def step(ctx):\n    pass\n"


def reply(text: str) -> Reply:
    return Reply(content=[{"type": "text", "text": text}], text=text, stop_reason="end_turn",
                 usage={"input": 1, "output": 1, "cache_write": 0, "cache_read": 0})


def block(code: str) -> str:
    return f"Here it is.\n```python\n{code}```\n"


def test_ctx_holds_this_steps_rows_and_values_move_against_the_gradient():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
    opt = ProgramOptimizer(net, program="def step(ctx):\n    print('step', ctx.step)\n")
    x, y = torch.randn(8, 5), torch.randn(8, 3)
    net(torch.randn(2, 5))                                   # grad-enabled but not in the loss: must be dropped
    with torch.no_grad():
        net(torch.randn(4, 5))                               # evaluation forward: never recorded
    h = net[0](x)
    h.retain_grad()
    out = net[2](torch.tanh(h))
    out.retain_grad()
    ((out - y) ** 2).mean().backward()
    opt.step()
    ctx = opt.ctx
    K = ctx.keys("0").data
    assert K.shape == (6, 8) and np.allclose(K[:5], x.T.numpy()) and np.allclose(K[5], 1)
    assert np.allclose(ctx.outputs("0"), h.detach().T.numpy(), atol=1e-6)
    assert np.allclose(ctx.output_grads("2"), out.grad.T.numpy(), atol=1e-9)
    V = ctx.values("2", step=0.5).data
    assert np.allclose(V, out.detach().T.numpy() - 0.5 * 8 * out.grad.T.numpy(), atol=1e-6)
    assert np.allclose(ctx.weight("0"), np.hstack([net[0].weight.detach(), net[0].bias.detach()[:, None]]))
    assert np.allclose(ctx.covariance("0", damping=0.0).C, K @ K.T / 8)
    assert ctx.sites == ["0", "2"] and ctx.shape("2") == (3, 5)
    assert opt.output == "step 1\n"


@pytest.mark.parametrize("last", ["linear", "conv1d"])
def test_an_exact_edit_lands_in_weight_and_bias(last):
    torch.manual_seed(0)
    head = nn.Linear(4, 3) if last == "linear" else Conv1D(3, 4)        # Conv1D(nf=out, nx=in) stores (in, out)
    net = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), head)
    program = (
        "def step(ctx):\n"
        "    K = ctx.keys('2')\n"
        "    ctx.apply(wir.edit('2', map=wir.Exact(K, wir.Values(np.full((3, K.count), 0.25), '2'))))\n"
    )
    opt = ProgramOptimizer(net, program=program)
    x = torch.randn(3, 5)
    first = net[0].weight.detach().clone()
    net(x).sum().backward()
    opt.step()
    with torch.no_grad():
        got = net(x)
    assert torch.allclose(got, torch.full_like(got, 0.25), atol=1e-5)
    assert torch.equal(net[0].weight, first)                                  # a layer the program left alone


def test_the_example_program_trains_a_classifier():
    torch.manual_seed(0)
    X = torch.randn(512, 10)
    y = (X[:, :3].sum(1) > 0).long() + (X[:, 3] > 1).long()
    net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
    w0 = net[0].weight.detach().clone()
    opt = ProgramOptimizer(net, program=EXAMPLE_PROGRAM)
    losses = []
    for i in range(60):
        idx = torch.arange(i * 64, (i + 1) * 64) % 512
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(X[idx]), y[idx])
        loss.backward()
        opt.step()
        opt.observe(loss.item())
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.6 * np.mean(losses[:10])
    assert not torch.equal(net[0].weight, w0) and opt.ctx.losses == losses
    assert opt.edit_stats["0"][0] == 60 and opt.edit_stats["2"][0] == 60       # one edit per site and step


def test_refresh_sees_the_edited_weights_and_undo_takes_the_edit_back():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
    opt = ProgramOptimizer(net, program=NOOP)
    x, y = torch.randn(8, 5), torch.randn(8, 3)

    def closure():
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    ctx, w0, b0 = opt.ctx, net[0].weight.detach().clone(), net[0].bias.detach().clone()
    patch = ctx.apply(wir.edit("0", map=wir.Soft(ctx.keys("0"), ctx.values("0", step=0.5), weight=1 / 8)))
    with torch.no_grad():
        h = torch.tanh(net[0](x))
        direct = ((net[2](h) - y) ** 2).mean().item()
    assert np.isclose(ctx.refresh(), direct, rtol=1e-6)                      # the loss at the edited weights
    assert np.allclose(ctx.keys("2").data[:4], h.T.numpy(), atol=1e-6)      # keys downstream of the edit
    ctx.undo(patch)
    assert torch.allclose(net[0].weight, w0, atol=1e-6) and torch.allclose(net[0].bias, b0, atol=1e-6)
    assert opt.edit_stats["0"][0] == 0                                       # an undone edit is not counted


def test_a_program_error_stops_training_with_its_traceback():
    net = nn.Sequential(nn.Linear(3, 2))
    opt = ProgramOptimizer(net, program="def step(ctx):\n    1 / 0\n")
    net(torch.randn(4, 3)).sum().backward()
    with pytest.raises(ProgramError) as e:
        opt.step()
    assert e.value.step == 1 and "ZeroDivisionError" in e.value.traceback and "1 / 0" in e.value.traceback


def test_the_loop_reports_each_trial_and_keeps_the_best():
    a, b, c = (f"def step(ctx):\n    pass  # {name}\n" for name in "abc")
    script = [reply("no code here"), reply(block(a)), reply("Try b.\n" + block(b)), reply(block(c))]
    sent = []

    def chat(system, messages):
        sent.append(messages[-1]["content"])
        return script[len(sent) - 1]

    outcomes = {a: (0.5, 2.0), b: (-math.inf, 1.0), c: (0.7, 3.0)}           # (score, seconds); b raised

    def run(code):
        score, seconds = outcomes[code]
        return {"score": score, "summary": f"score {score}", "report": f"report of {code[-2]}", "seconds": seconds}

    result = improve("toy task", "- 0: 2 x 3", run, chat, trials=3)
    assert len(sent) == 4
    assert "toy task" in sent[0] and "- 0: 2 x 3" in sent[0] and "3 trials" in sent[0]
    assert "```python" in sent[1]                                              # the missing-block retry
    assert "Trial 1: score 0.5" in sent[2] and "report of a" in sent[2] and "2 trials left" in sent[2]
    assert "report of b" in sent[3] and "the best is trial 1" in sent[3]
    assert result["best"] == 2 and [t["program"] for t in result["trials"]] == [a, b, c]
    assert result["trials"][1]["note"].startswith("Try b.")
    assert result["usage"]["output"] == 4 and len(result["transcript"]) == 8
