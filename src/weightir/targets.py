# Adapted from github.com/kmeng01/rome (MIT License, Copyright (c) 2022 Kevin Meng): rome/compute_v.py, rome/rome_main.py; and github.com/kmeng01/memit (MIT License, Copyright (c) 2022 Kevin Meng): memit/compute_z.py
"""Target values for fact edits: ROME's v* at a site's output, MEMIT's z in the residual stream.

Both come from one optimization. A vector δ is added to a module's output at the subject's last token of every
prompt, and Adam makes the model predict the new target after each template + prompt. The loss adds a KL term that
keeps the next-token distribution after the subject in "{subject} is a" in place, and a weight decay ‖δ‖/‖a₀‖²;
after each step ‖δ‖ is clamped to clamp_norm_factor·‖a₀‖, where a₀ is that module output on the bare prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from weightir import stats
from weightir.nethook import Trace
from weightir.spec import Values


KL_PROMPT = "{} is a"  # ROME's prompt for the KL term that keeps the subject's other facts


@dataclass
class Request:
    """One fact edit, CounterFact style: the text ``prompt`` with ``subject`` in place of "{}" should continue with
    ``target_new``. A missing leading space is added to the target, as ROME does."""

    prompt: str  # "{}" marks the subject
    subject: str
    target_new: str


def _blocks(model) -> str:
    """Name of the ModuleList that holds one decoder block per layer (transformer.h, model.layers)."""
    n = model.hf.config.num_hidden_layers
    for name, module in model.hf.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n:
            return name
    raise LookupError(f"{model.name} has no ModuleList of {n} decoder blocks")


def block(model, layer: int) -> str:
    """Name of decoder block ``layer``, whose output is the residual stream after that layer."""
    if not 0 <= layer < model.hf.config.num_hidden_layers:
        raise IndexError(f"layer {layer} out of range")
    return f"{_blocks(model)}.{layer}"


def layer_of(model, site: str) -> int:
    """Index of the decoder block that contains ``site``."""
    prefix = _blocks(model) + "."
    if not site.startswith(prefix):
        raise ValueError(f"{site} is not inside a decoder block {prefix}<layer>")
    return int(site[len(prefix):].split(".")[0])


def _optimize(model, module: str, dim: int, request: Request, templates, *, steps: int, lr: float,
              weight_decay: float, kl_factor: float, clamp_norm_factor: float):
    """(a₀, δ) for one request, float64. δ (``dim``) is added to the output of ``module`` at the subject's last token
    of every row: one row per template (template + prompt + target tokens but the last, teacher forced) and the KL
    row. a₀ is that output in row 0, so ``templates[0]`` must be "{}" (the bare prompt)."""
    stats.check_eval(model)
    if templates[0] != "{}":
        raise ValueError(f'templates[0] must be "{{}}" (the bare prompt), got {templates[0]!r}')
    tok, dev = model.tok, model.device
    target = request.target_new if request.target_new.startswith(" ") else " " + request.target_new
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    n, T = len(templates), len(t_ids)
    texts = [t.replace("{}", request.prompt, 1) for t in templates] + [KL_PROMPT]  # "{}" left for the subject
    pos = [stats.subject_last_index(tok, x, request.subject) for x in texts]
    rows = [tok(x.replace("{}", request.subject, 1))["input_ids"] for x in texts]
    rows = [r + t_ids[:-1] for r in rows[:n]] + rows[n:]  # concatenate ids: re-tokenizing text can merge them
    width = max(len(r) for r in rows)  # right padding: positions count from the left
    ids = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r in rows], device=dev)
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=dev)
    at = torch.zeros(len(rows), width, 1, device=dev)
    at[torch.arange(len(rows), device=dev), torch.tensor(pos, device=dev)] = 1.0
    # logits at position len(row) - T + j predict target token j
    lab_r = torch.arange(n, device=dev).repeat_interleave(T)
    lab_c = torch.tensor([len(r) - T + j for r in rows[:n] for j in range(T)], device=dev)
    lab_y = torch.tensor(t_ids * n, device=dev)

    delta = torch.zeros(dim, device=dev, requires_grad=True)
    a0 = kl_init = None

    def add_delta(out):
        nonlocal a0
        h = out[0] if isinstance(out, tuple) else out  # GPT-J blocks return (hidden, attn_weights)
        if a0 is None:
            a0 = h[0, pos[0]].detach().float().clone()
        h = h + at.to(h.dtype) * delta.to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    flags = [p.requires_grad for p in model.hf.parameters()]
    opt = torch.optim.Adam([delta], lr=lr)
    try:
        for p in model.hf.parameters():
            p.requires_grad_(False)
        with Trace(model.hf, module, retain_output=False, edit_output=add_delta):
            for it in range(steps):
                opt.zero_grad()
                logits = model.hf(input_ids=ids, attention_mask=mask, use_cache=False).logits
                logp = logits[lab_r, lab_c].float().log_softmax(-1)
                nll = -logp[torch.arange(n * T, device=dev), lab_y].sum() / (n * T)  # mean over rows and tokens
                kl_logp = logits[n, pos[n]].float().log_softmax(-1)[None]
                if kl_init is None:
                    kl_init = kl_logp.detach().clone()
                kl = kl_factor * F.kl_div(kl_init, kl_logp, log_target=True, reduction="batchmean")  # KL(now ‖ start)
                loss = nll + kl + weight_decay * delta.norm() / a0.norm() ** 2
                if loss.item() < 5e-2 or it == steps - 1:  # so at most steps - 1 updates
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    max_norm = clamp_norm_factor * a0.norm()
                    if delta.norm() > max_norm:
                        delta.mul_(max_norm / delta.norm())
    finally:
        for p, flag in zip(model.hf.parameters(), flags):
            p.requires_grad_(flag)
    return a0.cpu().double().numpy(), delta.detach().cpu().double().numpy()


def optimize_values(model, site: str, requests, templates, **opt) -> Values:
    """ROME's v* (compute_v), one column per request: W k₀ + δ, where W k₀ is the site's current output without its
    bias at the subject's last token of the bare prompt and δ is optimized on the site's output. ``opt`` are the
    keyword arguments of _optimize (steps, lr, weight_decay, kl_factor, clamp_norm_factor). ``templates[0]`` must be
    "{}"; ROME uses stats.context_templates(model)."""
    W = model.weight(site)
    D = np.stack([_optimize(model, site, W.shape[0], r, templates, **opt)[1] for r in requests], axis=1)
    k0 = stats.keys(model, site, [r.prompt for r in requests], [r.subject for r in requests])
    return Values(W @ k0.data + D, site)


def optimize_residual(model, layer: int, requests, templates, **opt) -> np.ndarray:
    """MEMIT's z (compute_z), one column per request: h + δ, where h is the output of decoder block ``layer`` (the
    residual stream) at the subject's last token of the bare prompt and δ is optimized on that output. ``opt`` are
    optimize_values' keyword arguments. Returns (d_model, p) float64."""
    name, d = block(model, layer), model.hf.config.hidden_size
    return np.stack([np.add(*_optimize(model, name, d, r, templates, **opt)) for r in requests], axis=1)
