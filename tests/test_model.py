"""Model wrapper and site statistics on the tiny random checkpoints in data/hf. Only what would silently corrupt
results: weight orientation and patching, token positions under padding and BOS, template averaging, pads in C."""
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from weightir import stats
from weightir.compile import Patch
from weightir.model import Model

from conftest import HF, hooked
GPT2 = "sshleifer/tiny-gpt2"  # Conv1D sites, no BOS
LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # Linear sites, adds BOS, pads on the left by default
LLAMA3 = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"  # byte-level BPE that adds BOS
# A non-square site and a square one, where a transposed weight cannot be caught by its shape.
SITES = {GPT2: ["transformer.h.0.mlp.c_proj", "transformer.h.1.attn.c_proj"],
         LLAMA: ["model.layers.0.mlp.down_proj", "model.layers.1.self_attn.o_proj"]}
TEXTS = ["Paris is", "The mother tongue of Danielle Darrieux is", "A cat sat", "Another fairly long sentence, with more tokens"]
POSITIONS = [1, -1, -1, 4]  # with batch_size=2, a padded row holds a positive position and another a negative one


@pytest.fixture(scope="module", params=[GPT2, LLAMA])
def model(request):
    return Model.load(request.param, cache_dir=HF, device="cpu")



def test_weight_and_apply_follow_the_module(model):
    rng = np.random.default_rng(0)
    for site in SITES[model.name]:
        mod, before = model.module(site), model.param(site).detach().clone()
        W = model.weight(site)
        b = 0.0 if mod.bias is None else mod.bias.detach().double().numpy()
        x = rng.standard_normal((5, W.shape[1]))

        def run():
            return mod(torch.tensor(x, dtype=torch.float32)).detach().double().numpy()

        assert np.allclose(run(), x @ W.T + b, atol=1e-6)
        D = rng.standard_normal(W.shape)
        model.apply(Patch(site, D, {}))
        model.apply(Patch(site, D, {}))  # reset must go back to the weight before the first apply
        assert np.allclose(run(), x @ (W + 2 * D).T + b, atol=1e-5)
        model.reset()
        assert torch.equal(model.param(site), before)


@pytest.mark.parametrize("which", ["in", "out"])
def test_collect_matches_a_hook_on_unpadded_forwards(model, which):
    site = SITES[model.name][0]
    got = stats.collect(model, site, TEXTS, POSITIONS, which=which, batch_size=2)
    ref = np.stack([hooked(model, site, t, which)[p] for t, p in zip(TEXTS, POSITIONS)], axis=1)
    assert got.dtype == np.float64 and got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("name", [GPT2, LLAMA, LLAMA3])
def test_subject_last_index_is_the_last_subject_token(name):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=HF)
    for text, subject in [("The mother tongue of {} is", "Danielle Darrieux"), ("{} spoke the language", "Toko Yasuda"),
                          ("{}'s headquarters are in", "Apple"), ('He said "{}" was', "Toko Yasuda"),
                          ("In 1990,{} moved to", "Danielle Darrieux"), ("{} was released by", "iPhone 5")]:
        i = stats.subject_last_index(tok, text, subject)
        ids = tok(text.replace("{}", subject))["input_ids"]
        assert tok.decode(ids[: i + 1], skip_special_tokens=True).endswith(subject), (text, i)
        assert not tok.decode(ids[:i], skip_special_tokens=True).endswith(subject), (text, i)


def test_keys_average_the_site_input_over_templates(model):
    site = SITES[model.name][0]
    prompts, subjects = ["The mother tongue of {} is", "{} plays the sport of"], ["Danielle Darrieux", "Toko Yasuda"]
    templates = ["{}", "Some text. {}", "A longer prefix, with a comma. {}"]
    K = stats.keys(model, site, prompts, subjects, templates)
    assert K.site == site and K.data.shape == (model.weight(site).shape[1], len(prompts))
    for i, (prompt, subject) in enumerate(zip(prompts, subjects)):
        cols = []
        for template in templates:
            text = template.replace("{}", prompt)
            pos = stats.subject_last_index(model.tok, text, subject)
            cols.append(hooked(model, site, text.replace("{}", subject), "in")[pos])
        assert np.allclose(K.data[:, i], np.mean(cols, axis=0), rtol=1e-5, atol=1e-8)


def test_second_moment_is_the_mean_over_real_tokens(model):
    site = SITES[model.name][0]
    X = np.vstack([hooked(model, site, t, "in") for t in TEXTS])  # (tokens, in_dim) without padding
    cov = stats.second_moment(model, site, TEXTS, batch_size=3)
    assert cov.site == site and cov.n == X.shape[0]
    assert np.allclose(cov.C, X.T @ X / X.shape[0], rtol=1e-5, atol=1e-10)
