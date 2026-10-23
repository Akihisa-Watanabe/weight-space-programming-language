# Adapted from kmeng01/rome (MIT License, Copyright (c) 2022 Kevin Meng): dsets/counterfact.py, experiments/py/eval_utils_counterfact.py, experiments/summarize.py; and kmeng01/memit (MIT License, Copyright (c) 2022 Kevin Meng): dsets/counterfact.py, experiments/py/eval_utils_counterfact.py.
"""CounterFact records, ROME's probability metrics, MEMIT's argmax accuracy, and a greedy-generation check.

Per case (``scores``), from the teacher-forced mean NLL per target token after each prompt:
    ES, PS   share of rewrite / paraphrase prompts after which target_new is more likely than target_true
    NS       share of neighborhood prompts after which target_true is still more likely than target_new
    EM, PM, NM  the matching mean gaps of exp(-NLL), i.e. of the geometric-mean per-token probability
and from the greedy continuation of each prompt (20 tokens, evals.generation.greedy):
    rewrite_gen, paraphrase_gen   share of prompts whose continuation contains target_new and not target_true
    neighborhood_gen              share of prompts whose continuation contains target_true and not target_new
(case-sensitive substrings, as RippleEdits checks answers; teacher forcing alone overstates edits, Yang et al. 2025).
Over cases (``summarize``): the means and S = harmonic mean of ES, PS and NS. All values are fractions;
the papers print them x100.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import hmean

from evals.generation import greedy

URLS = {
    "counterfact": "https://rome.baulab.info/data/dsets/counterfact.json",
    "multi_counterfact": "https://memit.baulab.info/data/dsets/multi_counterfact.json",
}
KEYS = ("ES", "PS", "NS", "EM", "PM", "NM", "rewrite_acc", "paraphrase_acc", "neighborhood_acc",
        "rewrite_gen", "paraphrase_gen", "neighborhood_gen")


def load(name: str = "counterfact", *, data_dir: str = "data/counterfact", size: int | None = None) -> list[dict]:
    """The first ``size`` records (all if None), downloading the file on first use.

    "counterfact" is ROME's file (21,919 records, single edits). "multi_counterfact" is MEMIT's subset for batch
    edits; it keeps the original case_ids, so there case_id is not the list index.
    """
    url = URLS[name]
    path = Path(data_dir) / f"{name}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    return json.loads(path.read_text(encoding="utf-8"))[:size]


def fields(record: dict) -> dict:
    """The parts of a record that an edit and its evaluation use. Targets are strings without a leading space."""
    rr = record["requested_rewrite"]
    return {
        "case_id": record["case_id"],
        "prompt": rr["prompt"],  # with "{}" where the subject goes
        "subject": rr["subject"],
        "target_new": rr["target_new"]["str"],
        "target_true": rr["target_true"]["str"],
        "paraphrase_prompts": record["paraphrase_prompts"],
        "neighborhood_prompts": record["neighborhood_prompts"],
    }


@torch.no_grad()
def evaluate(hf_model, tok, record: dict, *, batch_size: int = 32, max_new_tokens: int = 20) -> dict:
    """Teacher-forced scores of " target_new" and " target_true" after every prompt of ``record``, and the prompt's
    greedy continuation.

    Returns {"rewrite": [1 entry], "paraphrase": [2], "neighborhood": [10]}, one entry per prompt:
      nll_new, nll_true        mean negative log-likelihood per target token (lower = more likely)
      argmax_new, argmax_true  the argmax at every target position is that target's token (MEMIT)
      text                     greedy continuation of up to ``max_new_tokens`` tokens
      gen_new, gen_true        that target occurs in text
    Each scored sequence is the prompt's tokens (with BOS if the tokenizer adds one) followed by the target's
    tokens, right-padded here, so the tokenizer's padding side and pad token play no part. ``batch_size``
    counts sequences per forward pass; a record has 26.
    """
    if hf_model.training:
        raise ValueError("call hf_model.eval() first: dropout would make the scores random")
    f = fields(record)
    groups = {
        "rewrite": [f["prompt"].format(f["subject"])],
        "paraphrase": f["paraphrase_prompts"],
        "neighborhood": f["neighborhood_prompts"],
    }
    prompts = [p for ps in groups.values() for p in ps]
    targets = [f["target_new"], f["target_true"]]
    prompt_ids = tok(prompts)["input_ids"]
    target_ids = [tok(" " + t, add_special_tokens=False)["input_ids"] for t in targets]
    texts = [f"{p} {t}" for p in prompts for t in targets]

    rows = []  # (token ids, prompt length, target ids); row 2k is prompt k + target_new, row 2k+1 + target_true
    for i, (text, ids) in enumerate(zip(texts, tok(texts)["input_ids"])):
        p, t = prompt_ids[i // 2], target_ids[i % 2]
        # ROME's metric assumes tok(prompt + " " + target) == tok(prompt) + tok(" " + target); a merge or EOS breaks it.
        if ids != p + t:
            raise ValueError(f"case {f['case_id']}: {text!r} does not tokenize as the prompt's tokens + the target's")
        rows.append((ids, len(p), t))

    nll, hit = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        width = max(len(ids) for ids, _, _ in chunk)
        # Pad id 0 is never seen: causal attention does not look right, and the mask hides the pads.
        input_ids = torch.tensor([ids + [0] * (width - len(ids)) for ids, _, _ in chunk], device=hf_model.device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _, _ in chunk], device=hf_model.device)
        logits = hf_model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
        for row, (_, n, t) in zip(logits, chunk):
            z = row[n - 1:n - 1 + len(t)].cpu().double()  # the positions that predict the target's tokens
            y = torch.tensor(t)
            nll.append(-z.log_softmax(-1)[torch.arange(len(t)), y].mean().item())
            hit.append(bool((z.argmax(-1) == y).all()))

    conts = greedy(hf_model, tok, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    entries = [
        {"nll_new": nll[2 * k], "nll_true": nll[2 * k + 1], "argmax_new": hit[2 * k], "argmax_true": hit[2 * k + 1],
         "text": conts[k], "gen_new": f["target_new"] in conts[k], "gen_true": f["target_true"] in conts[k]}
        for k in range(len(prompts))
    ]
    out, k = {}, 0
    for g, ps in groups.items():
        out[g] = entries[k:k + len(ps)]
        k += len(ps)
    return out


def scores(result: dict) -> dict:
    """Per-case ES, PS, NS, EM, PM, NM as in ROME's summarize.py, MEMIT's argmax accuracy per prompt group, and the
    share of continuations that name the right target and not the other.

    Rewrite and paraphrase prompts should prefer target_new; neighborhood prompts should keep preferring
    target_true. A tie counts as a failure.
    """
    out = {}
    for g, s, m in (("rewrite", "ES", "EM"), ("paraphrase", "PS", "PM"), ("neighborhood", "NS", "NM")):
        good, bad = ("true", "new") if g == "neighborhood" else ("new", "true")
        xs = result[g]
        out[s] = float(np.mean([x[f"nll_{good}"] < x[f"nll_{bad}"] for x in xs]))
        out[m] = float(np.mean([np.exp(-x[f"nll_{good}"]) - np.exp(-x[f"nll_{bad}"]) for x in xs]))
        out[f"{g}_acc"] = float(np.mean([x[f"argmax_{good}"] for x in xs]))
        out[f"{g}_gen"] = float(np.mean([x[f"gen_{good}"] and not x[f"gen_{bad}"] for x in xs]))
    return out


def summarize(case_scores: list[dict]) -> dict:
    """Means of the ``scores`` keys over cases, and S = harmonic mean of the mean ES, PS and NS (ROME's score)."""
    out = {k: float(np.mean([c[k] for c in case_scores])) for k in KEYS}
    out["S"] = float(hmean([out["ES"], out["PS"], out["NS"]]))
    return out
