"""evals.data and the preset-baseline experiment, only where a silent mistake would corrupt results: article
boundaries in wikitext, the covariance cache, and the order of edits, resets and scoring in the experiment (replayed
here step by step). No download: the data/ tree is built in a temporary working directory."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from evals import counterfact as cf
from evals import data
from experiments.editing import presets as experiment
from weightir import presets, stats
from weightir.compile import Patch
from weightir.model import Model
from weightir.targets import Request

from conftest import HF

LLAMA = presets.HPARAMS["tiny-llama"]["model"]

# Two train shards in the wikitext-103-raw-v1 layout: a title row has an empty row on each side; section titles,
# table keys and formulas look alike but are not titles. Article Beta goes on in the second shard.
SHARD0 = ["", " = Alpha = \n", "", " The quick brown fox jumps over the lazy dog . Paris is the capital of France . \n",
          "", " = = History = = \n", "", " Danielle Darrieux was a French actress . Toko Yasuda plays the guitar . \n",
          " Key : Pld \n", " = Played , W = \n", " Wins \n", "", " = 0 , where n > 0 . \n", " = 24 = \n", " 16 \n",
          "", "", " = Beta = \n", "", " Apple 's headquarters are in Cupertino , California . \n"]
SHARD1 = [" In 1990 , the company moved to a new building . \n", " References \n", "", " = Gamma = \n", "",
          " A cat sat on the mat and looked out of the window . Rain fell all night over the hills . \n"]
ARTICLES = ["".join(SHARD0[1:17]).strip(), "".join(SHARD0[17:] + SHARD1[:3]).strip(), "".join(SHARD1[3:]).strip()]

RECORDS = [
    {"case_id": 0,
     "requested_rewrite": {"prompt": "The mother tongue of {} is", "subject": "Danielle Darrieux",
                           "target_new": {"str": "English"}, "target_true": {"str": "French"}},
     "paraphrase_prompts": ["Danielle Darrieux, a native", "Danielle Darrieux spoke the language"],
     "neighborhood_prompts": ["The mother tongue of Léon Blum is", "The native language of Montesquieu is"]},
    {"case_id": 9,
     "requested_rewrite": {"prompt": "{} plays the", "subject": "Toko Yasuda",
                           "target_new": {"str": "piano"}, "target_true": {"str": "guitar"}},
     "paraphrase_prompts": ["Toko Yasuda performs on the", "The instrument of Toko Yasuda is the"],
     "neighborhood_prompts": ["Jimi Hendrix plays the", "Eric Clapton plays the"]},
]


def write_shard(wikitext_dir: Path, i: int, rows: list[str]) -> None:
    path = wikitext_dir / data.SHARDS[i]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": rows}), path)


@pytest.fixture
def datadir(tmp_path, monkeypatch):
    """data/ in a temporary working directory: the tiny models, RECORDS as counterfact.json, SHARD0 and SHARD1."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data"
    (root / "counterfact").mkdir(parents=True)
    (root / "hf").symlink_to(HF)
    (root / "counterfact" / "counterfact.json").write_text(json.dumps(RECORDS), encoding="utf-8")
    write_shard(root / "wikitext", 0, SHARD0)
    write_shard(root / "wikitext", 1, SHARD1)
    return root


def test_wikitext_groups_rows_into_articles(tmp_path):
    write_shard(tmp_path, 0, SHARD0)
    assert data.wikitext(1, data_dir=str(tmp_path)) == ARTICLES[:1]  # the second shard is not needed (nor fetched)
    write_shard(tmp_path, 1, SHARD1)
    assert data.wikitext(2, data_dir=str(tmp_path)) == ARTICLES[:2]  # Beta, cut by the shard boundary, is whole
    assert data.wikitext(10, data_dir=str(tmp_path)) == ARTICLES


def test_covariance_cache_round_trip(datadir, monkeypatch):
    m = Model.load(LLAMA, device="cpu")
    site = "model.layers.1.mlp.down_proj"
    fresh = stats.second_moment(m, site, data.wikitext(2))
    data.covariance(m, site, n_docs=2)
    monkeypatch.setattr(stats, "second_moment", None)  # from here on, C must come from the file
    cached = data.covariance(m, site, n_docs=2)
    assert np.array_equal(cached.C, fresh.C) and cached.n == fresh.n and cached.site == site
    m.apply(Patch(site, np.ones(m.weight(site).shape), {}))
    with pytest.raises(RuntimeError, match="unedited"):  # other settings do not read that file, and a cache is
        data.covariance(m, site, n_docs=3)  # never computed on an edited model


def same(a: dict, b: dict) -> bool:
    """Two counterfact.evaluate results agree: the argmax flags exactly, the NLLs to 1e-9."""
    return all((x[k] == pytest.approx(y[k], abs=1e-9)) if k.startswith("nll") else x[k] == y[k]
               for g in a for x, y in zip(a[g], b[g], strict=True) for k in x)


@pytest.mark.parametrize("sequential", [False, True])
def test_presets_experiment_matches_a_replay(datadir, monkeypatch, sequential):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # main's Model.load picks CPU, like the replay
    summary = experiment.main("tiny-llama", "alphaedit", "counterfact", size=2, sequential=sequential)
    name = f"tiny-llama_alphaedit_counterfact_b1{'_seq' if sequential else ''}.jsonl"
    rows = [json.loads(line) for line in (datadir / "runs" / "editing" / "presets" / name).read_text().splitlines()]
    assert [r["case_id"] for r in rows] == [0, 9] and [r["batch_case_ids"] for r in rows] == [[0], [9]]

    # Replay: the run's cached templates and second moments, every edit from the unedited model unless sequential
    # (then AlphaEdit carries the Grams), scores right after each edit or, if sequential, after the last one.
    m = Model.load(LLAMA, device="cpu")
    hp = presets.HPARAMS["tiny-llama"]["alphaedit"]
    sites = [presets.HPARAMS["tiny-llama"]["site"].format(layer) for layer in hp["layers"]]
    templates = data.templates(m, hp["length_params"])
    C0s = {s: data.covariance(m, s) for s in sites}
    pre = [cf.evaluate(m.hf, m.tok, r) for r in RECORDS]
    post, prev = [], None
    for r in RECORDS:
        f = cf.fields(r)
        _, grams = presets.alphaedit(m, [Request(f["prompt"], f["subject"], f["target_new"])], C0s,
                                     sites=sites, tau=hp["tau"], l2=hp["l2"], prev=prev, templates=templates,
                                     **hp["opt"])
        if sequential:
            prev = grams
        else:
            post.append(cf.evaluate(m.hf, m.tok, r))
            m.reset()
    if sequential:
        post = [cf.evaluate(m.hf, m.tok, r) for r in RECORDS]

    for row, a, b in zip(rows, pre, post, strict=True):
        assert same(row["pre"], a) and same(row["post"], b)
        assert row["post"]["rewrite"][0]["nll_new"] != row["pre"]["rewrite"][0]["nll_new"]  # the edit did something
    assert summary["pre"] == pytest.approx(cf.summarize([cf.scores(x) for x in pre]))
    assert summary["post"] == pytest.approx(cf.summarize([cf.scores(x) for x in post]))
