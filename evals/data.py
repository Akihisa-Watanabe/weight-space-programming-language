"""wikitext-103 articles, and per-model caches of what edits need from the unedited model: the second moment of a
site's inputs (covariance), ROME's context templates (templates) and site inputs at random tokens (keep_inputs).

Caches are plain files under <data_dir>/<model name with "/" -> "__">/, named by site and settings, so other settings
make another file. They describe the unedited model: computing one on a model with applied patches raises.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from weightir import stats
from weightir.spec import Covariance, Keys

REPO = "Salesforce/wikitext"
SHARDS = ("wikitext-103-raw-v1/train-00000-of-00002.parquet", "wikitext-103-raw-v1/train-00001-of-00002.parquet")
TITLE = re.compile(r" = [^=].* = \n")  # an article title; section titles are " = = Section = = \n"
KEEP_MAX_LEN = 1024  # keep_inputs cuts articles like covariance's default max_len


def _articles(rows: list[str]) -> list[str]:
    """Group wikitext rows (one line each) into stripped, non-empty articles, title first. A title row has an empty
    row on each side; table keys and formula lines that also look like " = ... = " do not."""
    starts = [i for i in range(1, len(rows) - 1) if rows[i - 1] == rows[i + 1] == "" and TITLE.fullmatch(rows[i])]
    docs = ("".join(rows[a:b]).strip() for a, b in zip([0, *starts], [*starts, len(rows)]))
    return [d for d in docs if d]


def wikitext(n_docs: int, *, data_dir: str = "data/wikitext") -> list[str]:
    """The first ``n_docs`` articles of the wikitext-103-raw-v1 train split (all of its ~28k if fewer). Its two
    parquet shards (~157 MB each) are downloaded from the Hub dataset Salesforce/wikitext only as far as needed."""
    rows: list[str] = []
    for i, shard in enumerate(SHARDS):
        path = Path(data_dir) / shard
        if not path.exists():
            hf_hub_download(REPO, shard, repo_type="dataset", local_dir=data_dir)
        rows += pq.read_table(path, columns=["text"]).column("text").to_pylist()
        docs = _articles(rows)
        if i < len(SHARDS) - 1:
            docs = docs[:-1]  # it may go on in the next shard
        if len(docs) >= n_docs:
            break
    return docs[:n_docs]


def _cache(model, data_dir: str, name: str) -> Path:
    return Path(data_dir) / model.name.replace("/", "__") / name


def _unedited(model) -> None:
    if model._original:
        raise RuntimeError(f"{model.name} has applied patches, and caches describe the unedited model; "
                           "call model.reset() first")


def _write(path: Path, write) -> None:
    """Write through a temporary file and rename, so that parallel runs never read a half-written cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def covariance(model, site, *, n_docs=10_000, max_len=1024, data_dir="data/stats") -> Covariance:
    """stats.second_moment of ``site``'s inputs over wikitext(n_docs), each article cut to ``max_len`` tokens, on the
    unedited model; cached in float64 as <data_dir>/<model>/<site>.C_wikitext<n_docs>_len<max_len>.npz."""
    path = _cache(model, data_dir, f"{site}.C_wikitext{n_docs}_len{max_len}.npz")
    if path.exists():
        with np.load(path) as z:
            return Covariance(z["C"], site, n=int(z["n"]))
    _unedited(model)
    cov = stats.second_moment(model, site, wikitext(n_docs), max_len=max_len)
    _write(path, lambda f: np.savez(f, C=cov.C, n=cov.n))
    return cov


def templates(model, length_params, *, seed=0, data_dir="data/stats") -> list[str]:
    """stats.context_templates(model, length_params=length_params, seed=seed) of the unedited model; cached as
    <data_dir>/<model>/templates_<length>x<count>_..._seed<seed>.json."""
    name = "_".join(f"{n}x{c}" for n, c in length_params)
    path = _cache(model, data_dir, f"templates_{name}_seed{seed}.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    _unedited(model)
    out = stats.context_templates(model, length_params=length_params, seed=seed)
    _write(path, lambda f: f.write(json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")))
    return out


@torch.no_grad()
def keep_inputs(model, site, *, n_tokens=4096, n_docs=200, seed=0, data_dir="data/stats") -> Keys:
    """Inputs of ``site`` on the unedited model at ``n_tokens`` positions drawn uniformly without replacement from
    the real tokens (BOS included) of wikitext(n_docs), each article cut to 1024 tokens. Columns are in article,
    then position order. Cached in float64 as <data_dir>/<model>/<site>.keep<n_tokens>_wikitext<n_docs>_seed<seed>.npz.
    """
    path = _cache(model, data_dir, f"{site}.keep{n_tokens}_wikitext{n_docs}_seed{seed}.npz")
    if path.exists():
        with np.load(path) as z:
            return Keys(z["X"], site)
    _unedited(model)
    texts = wikitext(n_docs)
    lengths = [len(ids) for ids in model.tok(texts, truncation=True, max_length=KEEP_MAX_LEN)["input_ids"]]
    starts = np.cumsum([0, *lengths])
    picks = np.sort(np.random.default_rng(seed).choice(starts[-1], size=n_tokens, replace=False))
    doc = np.searchsorted(starts, picks, side="right") - 1
    pos = picks - starts[doc]
    cols, batch = [], 8
    for s in range(0, len(texts), batch):
        enc = model.tok(texts[s:s + batch], padding=True, padding_side="right", truncation=True,
                        max_length=KEEP_MAX_LEN, return_tensors="pt").to(model.device)
        x = stats.forward_to(model, site, enc, "in")
        here = (doc >= s) & (doc < s + batch)
        rows = torch.as_tensor(doc[here] - s, device=x.device)
        cols.append(x[rows, torch.as_tensor(pos[here], device=x.device)].cpu().double())
    X = torch.cat(cols).T.contiguous().numpy()
    _write(path, lambda f: np.savez(f, X=X))
    return Keys(X, site)
