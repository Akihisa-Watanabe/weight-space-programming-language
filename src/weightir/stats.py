# Adapted from github.com/kmeng01/rome (MIT License, Copyright (c) 2022 Kevin Meng): rome/repr_tools.py, rome/compute_u.py, rome/layer_stats.py, rome/tok_dataset.py, rome/rome_main.py, util/generate.py, util/runningstats.py; and github.com/jianghoucheng/AlphaEdit (MIT License, Copyright (c) 2025 jianghc): experiments/evaluate.py
"""Activations at a site: inputs/outputs at token positions, ROME keys, the second moment C of a site's inputs,
ROME's context templates, and AlphaEdit's preserved subspace.

Forwards over texts run on right-padded batches with use_cache=False and stop right after the site; activations
come back in float64.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import torch

from weightir.nethook import Trace
from weightir.spec import Covariance, Keys, Subspace


def check_eval(model) -> None:
    """Raise when dropout is on: activations and targets must come from the model in eval mode."""
    if model.hf.training:
        raise RuntimeError("the model is in training mode (dropout is on); call model.hf.eval() first")


def forward_to(model, site: str, enc, which: str) -> torch.Tensor:
    """Run the model on a tokenized batch up to ``site``; return the site's input or output (batch, seq, dim)."""
    check_eval(model)
    with Trace(model.hf, site, retain_input=which == "in", retain_output=which == "out", stop=True) as tr:
        model.hf(**enc, use_cache=False)
    if which == "in":
        return tr.input
    return tr.output[0] if isinstance(tr.output, tuple) else tr.output  # e.g. GPT-J blocks return tuples


def subject_last_index(tok, text: str, subject: str) -> int:
    """Index of the subject's last token in ``tok(text with its "{}" replaced by subject)``, special tokens such as
    BOS included. ``text`` holds one "{}" where the subject goes (CounterFact format).

    The index comes from character offsets, so it is right for tokenizers that add a BOS (ROME's count is one too
    high there) and needs no space before "{}".
    """
    if text.count("{}") != 1:
        raise ValueError(f"text must hold exactly one '{{}}' where the subject goes: {text!r}")
    if not subject:
        raise ValueError("subject is empty")
    start = text.index("{}")
    end = start + len(subject)
    enc = tok(text.replace("{}", subject, 1), return_offsets_mapping=True)
    hits = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end and b > start]
    if not hits:
        raise ValueError(f"no token covers the subject {subject!r} in {text!r}")
    return hits[-1]


@torch.no_grad()
def collect(model, site: str, texts, positions, *, which: str = "in", batch_size: int = 32) -> np.ndarray:
    """The input (``which="in"``) or output (``"out"``) of ``site`` at token ``positions[i]`` of ``texts[i]``.

    Positions index ``tok(text)``, BOS included; a negative position counts back from the end of that text's own
    tokens (-1 is its last token). Returns float64 columns, shape (dim, len(texts)).
    """
    if which not in ("in", "out"):
        raise ValueError(f"which must be 'in' or 'out', got {which!r}")
    if len(texts) != len(positions):
        raise ValueError(f"{len(texts)} texts but {len(positions)} positions")
    rows = []
    for s in range(0, len(texts), batch_size):
        batch, pos = list(texts[s:s + batch_size]), positions[s:s + batch_size]
        enc = model.tok(batch, padding=True, padding_side="right", return_tensors="pt").to(model.device)
        x = forward_to(model, site, enc, which)
        idx = []
        for text, p, n in zip(batch, pos, enc["attention_mask"].sum(1).tolist()):
            if not -n <= p < n:
                raise IndexError(f"position {p} is outside the {n} tokens of {text!r}")
            idx.append(int(p) % n)
        picked = x[torch.arange(len(idx), device=x.device), torch.tensor(idx, device=x.device)]
        rows.append(picked.cpu().double())  # .cpu() before the dtype change: MPS converts wrongly otherwise
    return torch.cat(rows).T.contiguous().numpy()


def keys(model, site: str, prompts, subjects, templates=("{}",)) -> Keys:
    """ROME's key: the input of ``site`` at the subject's last token, averaged over ``templates``.

    ``prompts[i]`` holds "{}" where ``subjects[i]`` goes (CounterFact format); each template holds "{}" where the
    prompt goes. Column i is the mean over templates of the site input at the subject's last token of
    ``template.format(prompts[i].format(subjects[i]))``. The default ("{}",) gives the bare-prompt key;
    ``context_templates(model)`` gives ROME's template-averaged key.
    """
    templates = list(templates)
    if len(prompts) != len(subjects):
        raise ValueError(f"{len(prompts)} prompts but {len(subjects)} subjects")
    texts, positions = [], []
    for prompt, subject in zip(prompts, subjects):
        for template in templates:
            if template.count("{}") != 1:
                raise ValueError(f"a template must hold exactly one '{{}}' where the prompt goes: {template!r}")
            text = template.replace("{}", prompt, 1)
            positions.append(subject_last_index(model.tok, text, subject))
            texts.append(text.replace("{}", subject, 1))
    X = collect(model, site, texts, positions)  # column i * len(templates) + t: request i, template t
    return Keys(X.reshape(X.shape[0], len(prompts), len(templates)).mean(axis=2), site)


@torch.no_grad()
def second_moment(model, site: str, texts, *, max_len: int = 1024, batch_size: int = 8) -> Covariance:
    """Uncentered second moment C = (1/n) Σ_t x_t x_tᵀ of the input of ``site`` over every real (non-pad) token of
    ``texts``, special tokens included, each text truncated to ``max_len`` tokens (ROME's layer_stats). Empty
    texts are skipped. Accumulated in float64; n is the number of tokens."""
    texts = sorted((t for t in texts if t), key=len, reverse=True)  # similar lengths per batch: less padding
    S, n = 0.0, 0
    for s in range(0, len(texts), batch_size):
        enc = model.tok(texts[s:s + batch_size], padding=True, padding_side="right", truncation=True,
                        max_length=max_len, return_tensors="pt").to(model.device)
        x = forward_to(model, site, enc, "in")[enc["attention_mask"].bool()]
        x = x.cpu().double() if x.device.type == "mps" else x.double()  # MPS has no float64
        S = S + x.T @ x
        n += x.shape[0]
    if n == 0:
        raise ValueError("second_moment needs at least one non-empty text")
    return Covariance((S / n).cpu().numpy(), site, n=n)


@torch.no_grad()
def context_templates(model, *, length_params=((5, 10), (10, 10)), top_k: int = 5, seed: int = 0) -> list[str]:
    """ROME's context templates: ["{}"] + [text + ". {}"]. For each (length, count) in ``length_params``, ``count``
    texts of ``length`` tokens, counting the start token (BOS, or EOS without one), are sampled from the model with
    top-k sampling. Braces in the samples become spaces, so every template holds exactly one "{}". Deterministic
    for a seed."""
    check_eval(model)
    tok = model.tok
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    gen = torch.Generator().manual_seed(seed)
    out = ["{}"]
    for length, count in length_params:
        ids = torch.full((count, 1), start, dtype=torch.long, device=model.device)
        while ids.shape[1] < length:
            probs = model.hf(input_ids=ids, use_cache=False).logits[:, -1].float().softmax(-1)
            top_p, top_i = probs.topk(top_k, dim=-1)
            top_p = top_p.cpu()
            pick = torch.multinomial(top_p / top_p.sum(-1, keepdim=True), 1, generator=gen)
            ids = torch.cat([ids, top_i.gather(1, pick.to(ids.device))], dim=1)
        for row in ids[:, 1:].tolist():
            text = unicodedata.normalize("NFKD", tok.decode(row, skip_special_tokens=True)).replace("\n\n", " ")
            out.append(text.replace("{", " ").replace("}", " ") + ". {}")
    return out


def top_eigenspace(cov: Covariance, threshold: float) -> Subspace:
    """Eigenvectors of ``cov.C`` with eigenvalue > ``threshold``, largest first: the subspace AlphaEdit preserves
    (it edits only along eigenvalues below 2e-2). ``exact=False``; the source is C^{1/2}, which has the same
    second moment as the tokens behind C, so check() reports how much the outputs on those tokens move."""
    evals, evecs = np.linalg.eigh(cov.C)
    U = evecs[:, evals > threshold][:, ::-1]
    return Subspace(U, cov.site, exact=False, source=evecs * np.sqrt(np.clip(evals, 0.0, None)))
