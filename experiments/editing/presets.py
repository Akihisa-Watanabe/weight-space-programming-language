# Adapted from github.com/kmeng01/memit (MIT License, Copyright (c) 2022 Kevin Meng): experiments/evaluate.py; and github.com/jianghoucheng/AlphaEdit (MIT License, Copyright (c) 2025 jianghc): experiments/evaluate.py
"""Preset baselines on CounterFact: ROME, EMMET, MEMIT and AlphaEdit with the settings in weightir.presets.HPARAMS.

    python experiments/editing/presets.py --model gpt2-xl --method memit --dataset multi_counterfact --size 1000 --batch 100

Records start .. start + size - 1 are edited a batch at a time (ROME: one record per batch). By default every batch
edits the unedited model and is undone once its records are scored (MEMIT's evaluate.py). With --sequential the edits
accumulate, AlphaEdit also carries the Grams of earlier keys, and every record is scored after the last batch
(AlphaEdit's sequential setting). "pre" is always the unedited model, and so are the context templates and second
moments (evals.data caches them under data/stats). Each record appends one JSON line with
evals.counterfact.evaluate's output before and after to --out; the summary of both is printed.
"""
import argparse
import json
import time
from pathlib import Path

import weightir as wir
from evals import counterfact as cf
from evals import data
from weightir import presets
from weightir.model import Model
from weightir.targets import Request

METHODS = ("rome", "emmet", "memit", "alphaedit")


def _edit(model, method: str, hp: dict, sites: list[str], Cs: dict | None, templates, requests, prev):
    """Apply one batch of edits to ``model``; return its patches and AlphaEdit's Grams (None for the others)."""
    if method == "memit":
        return presets.memit(model, requests, Cs, sites=sites, lam=hp["lam"], templates=templates, **hp["opt"]), None
    if method == "alphaedit":
        return presets.alphaedit(model, requests, Cs, sites=sites, tau=hp["tau"], l2=hp["l2"], prev=prev,
                                 templates=templates, **hp["opt"])
    C = None if Cs is None else Cs[sites[0]]
    if method == "rome":
        e = presets.rome(model, requests[0], C, site=sites[0], templates=templates, **hp["opt"])
    else:
        e = presets.emmet(model, requests, C, site=sites[0], templates=templates, **hp["opt"])
    patch = wir.compile(e, model)
    model.apply(patch)
    return [patch], None


def main(model: str, method: str, dataset: str, *, start: int = 0, size: int = 100, batch: int = 1,
         sequential: bool = False, out: str | None = None) -> dict:
    """Run one baseline (``model`` is a presets.HPARAMS key); return the printed summary {"pre": ..., "post": ...}."""
    if method not in METHODS or method not in presets.HPARAMS[model]:
        raise ValueError(f"presets.HPARAMS has no {method} settings for {model}")
    if method == "rome" and batch != 1:
        raise ValueError("ROME edits one record at a time: use --batch 1")
    if batch > 1 and dataset != "multi_counterfact":
        raise ValueError("counterfact holds conflicting records; batches need multi_counterfact, as in MEMIT")
    hp = presets.HPARAMS[model][method]
    m = Model.load(presets.HPARAMS[model]["model"])
    sites = [presets.HPARAMS[model]["site"].format(layer) for layer in hp["layers"]]
    templates = data.templates(m, hp["length_params"])
    Cs = None if method == "rome" and not hp["mom2_adjustment"] else {s: data.covariance(m, s) for s in sites}
    records = cf.load(dataset, size=start + size)[start:]
    if not records:
        raise ValueError(f"{dataset} has no records from index {start} on")
    pre = [cf.evaluate(m.hf, m.tok, r) for r in records]
    post, info, prev = [None] * len(records), [None] * len(records), None
    out = Path(out or f"data/runs/editing/presets/{model}_{method}_{dataset}_b{batch}"
                      f"{'_seq' if sequential else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"appending {len(records)} rows to {out}", flush=True)

    def write(idx):
        with out.open("a", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps({"model": model, "method": method, "dataset": dataset, "start": start,
                                    "size": len(records), "batch": batch, "sequential": sequential,
                                    "case_id": records[i]["case_id"], **info[i], "pre": pre[i], "post": post[i]})
                        + "\n")

    for b, s in enumerate(range(0, len(records), batch)):
        idx = range(s, min(s + batch, len(records)))
        fs = [cf.fields(records[i]) for i in idx]
        requests = [Request(f["prompt"], f["subject"], f["target_new"]) for f in fs]
        t = time.time()
        patches, grams = _edit(m, method, hp, sites, Cs, templates, requests, prev)
        seconds = time.time() - t
        print(f"batch {b + 1}/{-(-len(records) // batch)}: {len(idx)} edits in {seconds:.1f} s", flush=True)
        for i in idx:
            info[i] = {"batch_case_ids": [f["case_id"] for f in fs], "seconds": seconds,
                       "checks": [p.check for p in patches]}
        if sequential:
            prev = grams
        else:
            for i in idx:
                post[i] = cf.evaluate(m.hf, m.tok, records[i])
            m.reset()
            write(idx)
    if sequential:
        post = [cf.evaluate(m.hf, m.tok, r) for r in records]
        write(range(len(records)))

    summary = {k: cf.summarize([cf.scores(x) for x in v]) for k, v in (("pre", pre), ("post", post))}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preset baselines (ROME, EMMET, MEMIT, AlphaEdit) on CounterFact.")
    p.add_argument("--model", required=True, choices=sorted(presets.HPARAMS))
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--dataset", required=True, choices=sorted(cf.URLS))
    p.add_argument("--start", type=int, default=0, help="index of the first record")
    p.add_argument("--size", type=int, default=100, help="number of records")
    p.add_argument("--batch", type=int, default=1, help="edits per batch (ROME: 1)")
    p.add_argument("--sequential", action="store_true", help="keep edits across batches; score after the last one")
    p.add_argument("--out", help="JSON lines file to append to (default: under data/runs/editing/presets/)")
    main(**vars(p.parse_args()))
