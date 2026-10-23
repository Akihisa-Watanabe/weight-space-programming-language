"""Presets on tiny random checkpoints, mechanics only (these models barely learn). A wrong key, value, residual share
or solver weight would silently change every edit, so each patch is compared with the method's closed form, rebuilt
here from the model state its layer saw."""
import numpy as np
import pytest
import torch

import weightir as wir
from weightir import presets, stats, targets
from weightir.model import Model
from weightir.targets import Request

from conftest import HF, hooked
# decoder block names, written out here rather than taken from targets.block
BLOCK = {"tiny-gpt2": "transformer.h.{}", "tiny-llama": "model.layers.{}"}
REQUESTS = [Request("The mother tongue of {} is", "Danielle Darrieux", "English"),
            Request("{} plays the sport of", "Toko Yasuda", "football"),
            Request("{}'s headquarters are in", "Apple", "Tokyo")]
TEMPLATES = ["{}", "Some text. {}", "A longer prefix, with a comma. {}"]
TEXTS = ["The quick brown fox jumps over the lazy dog.", "Paris is the capital of France.",
         "Danielle Darrieux was a French actress.", "Toko Yasuda plays the guitar in a band.",
         "Apple's headquarters are in Cupertino, California.", "In 1990, the company moved to a new building.",
         "Second moments of activations are uncentered here.", "A cat sat on the mat and looked out of the window.",
         "Rain fell all night over the hills, and the rivers rose.",
         "The mother tongue of Danielle Darrieux is French."]


@pytest.fixture(scope="module")
def loaded():
    models = {name: Model.load(presets.HPARAMS[name]["model"], cache_dir=HF, device="cpu") for name in BLOCK}
    with torch.no_grad():  # the checkpoint's biases are all zero; a bias wrongly kept in V must show
        for layer in range(2):
            models["tiny-gpt2"].module(f"transformer.h.{layer}.mlp.c_proj").bias.normal_(
                0.0, 0.01, generator=torch.Generator().manual_seed(layer))
    return models


@pytest.fixture(params=list(BLOCK))
def tiny(loaded, request):
    """(model, its HPARAMS entry, its block names); the test's edits are undone afterwards."""
    yield loaded[request.param], presets.HPARAMS[request.param], BLOCK[request.param]
    loaded[request.param].reset()


@pytest.fixture
def llama(loaded):
    yield loaded["tiny-llama"]
    loaded["tiny-llama"].reset()


def at_subject(model, module, requests, which="out"):
    """``hooked`` at the subject's last token of each bare prompt, (dim, p)."""
    return np.stack([hooked(model, module, r.prompt.replace("{}", r.subject), which)[
        stats.subject_last_index(model.tok, r.prompt, r.subject)] for r in requests], axis=1)


def memit_keys(model, site, requests):
    """MEMIT's key: half the bare-prompt key, half the mean key over the other templates."""
    args = (model, site, [r.prompt for r in requests], [r.subject for r in requests])
    return 0.5 * stats.keys(*args).data + 0.5 * stats.keys(*args, TEMPLATES[1:]).data


def residual(model, block, requests):
    """A block's output at the subject's last token of each bare prompt, batched as the presets measure it."""
    return stats.collect(model, block, [r.prompt.replace("{}", r.subject) for r in requests],
                         [stats.subject_last_index(model.tok, r.prompt, r.subject) for r in requests], which="out")


def bias(model, site):
    b = model.module(site).bias
    return np.zeros(model.weight(site).shape[0]) if b is None else b.detach().double().numpy()


def test_targets_start_at_the_current_output_and_leave_the_model_as_it_was(tiny):
    model, hp, block = tiny
    layer, r = hp["rome"]["layers"][0], REQUESTS[0]
    site = hp["site"].format(layer)
    ids = model.tok(r.prompt.replace("{}", r.subject), return_tensors="pt")
    with torch.no_grad():
        before = model.hf(**ids).logits
    one_step = dict(hp["rome"]["opt"], steps=1)  # the loop stops before its first update: δ = 0
    a0 = at_subject(model, site, [r])[:, 0]
    V = targets.optimize_values(model, site, [r], TEMPLATES, **one_step)
    # float32 activations of order 1 from two forward paths: their rounding differs by a few ulp, and by CPU
    assert np.allclose(V.data[:, 0], a0 - bias(model, site), rtol=1e-5, atol=1e-6)  # W k₀: the bias is not in V
    Z = targets.optimize_residual(model, layer, [r], TEMPLATES, **one_step)
    assert np.allclose(Z[:, 0], at_subject(model, block.format(layer), [r])[:, 0], rtol=1e-5, atol=1e-6)
    V = targets.optimize_values(model, site, [r], TEMPLATES, **hp["rome"]["opt"])
    delta = V.data[:, 0] - (a0 - bias(model, site))
    assert 0 < np.linalg.norm(delta) <= hp["rome"]["opt"]["clamp_norm_factor"] * np.linalg.norm(a0) * (1 + 1e-5)
    with torch.no_grad():
        assert torch.equal(model.hf(**ids).logits, before)  # the δ hook is gone
    assert all(p.requires_grad for p in model.hf.parameters())


def test_first_step_descends_the_target_nll(llama):
    """Adam's first step is −lr·sign(∇), and at δ = 0 only the NLL has a gradient (the KL term and ‖δ‖ are flat
    there), so the signs of δ show which tokens the loss scores. ∇ is recomputed from unpadded single sequences."""
    site, r = "model.layers.0.mlp.up_proj", REQUESTS[1]  # 32 outputs; layer 1's attention carries δ to the target
    opt = dict(steps=2, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=1e9)
    delta = targets.optimize_values(llama, site, [r], TEMPLATES, **opt).data[:, 0] - at_subject(llama, site, [r])[:, 0]
    d = torch.zeros(len(delta), requires_grad=True)
    t_ids = llama.tok(" " + r.target_new, add_special_tokens=False)["input_ids"]
    nll = 0.0
    for t in TEMPLATES:
        text = t.replace("{}", r.prompt)
        pos = stats.subject_last_index(llama.tok, text, r.subject)
        p_ids = llama.tok(text.replace("{}", r.subject))["input_ids"]
        onehot = torch.zeros(1, len(p_ids) + len(t_ids), 1)
        onehot[0, pos] = 1.0
        h = llama.module(site).register_forward_hook(lambda m, args, out: out + onehot * d)
        try:
            logp = llama.hf(torch.tensor([p_ids + t_ids]), use_cache=False).logits[0].log_softmax(-1)
        finally:
            h.remove()
        nll = nll - sum(logp[len(p_ids) - 1 + j, y] for j, y in enumerate(t_ids)) / len(t_ids)
    g = torch.autograd.grad(nll / len(TEMPLATES), d)[0].double().numpy()
    big = np.abs(g) > 1e-4 * np.abs(g).max()
    assert big.sum() > 20 and np.array_equal(np.sign(delta[big]), -np.sign(g[big]))


def test_rome_is_the_closed_form_on_the_averaged_key(tiny):
    model, hp, _ = tiny
    site, r = hp["site"].format(hp["rome"]["layers"][0]), REQUESTS[0]
    W = model.weight(site)
    X = np.random.default_rng(0).standard_normal((W.shape[1], 4 * W.shape[1]))
    C = wir.Covariance(X @ X.T / X.shape[1], site)  # tiny GPT-2's own second moments are singular
    e = presets.rome(model, r, C, site=site, templates=TEMPLATES, **hp["rome"]["opt"])
    k, v = e.maps[0].K.data, e.maps[0].V.data
    assert np.allclose(k, stats.keys(model, site, [r.prompt], [r.subject], TEMPLATES).data)  # k̄, not k₀
    p = wir.compile(e, model)
    Cik = np.linalg.solve(C.C, k)
    assert np.allclose(W + p.delta, W + (v - W @ k) @ Cik.T / (k.T @ Cik), rtol=1e-9, atol=1e-12)
    model.apply(p)
    with torch.no_grad():
        y = model.module(site)(torch.tensor(k.T, dtype=torch.float32)).double().numpy().T - bias(model, site)[..., None]
    assert np.allclose(y, v, rtol=1e-4, atol=1e-5 * np.abs(v).max())  # the edited site outputs v at k


def test_emmet_puts_the_block_output_on_z(llama):
    hp = presets.HPARAMS["tiny-llama"]
    layer = hp["emmet"]["layers"][0]
    site = hp["site"].format(layer)
    # with the bare prompt as the only template, MEMIT's key is the key of the prompt the residual is measured on
    Z = targets.optimize_residual(llama, layer, REQUESTS, ["{}"], **hp["emmet"]["opt"])
    e = presets.emmet(llama, REQUESTS, stats.second_moment(llama, site, TEXTS), site=site, templates=["{}"],
                      **hp["emmet"]["opt"])
    p = wir.compile(e, llama)
    assert p.check["ok"] and p.check["map_residual"] < 1e-9
    llama.apply(p)
    assert np.allclose(at_subject(llama, BLOCK["tiny-llama"].format(layer), REQUESTS), Z, rtol=1e-4, atol=1e-5)


def test_memit_layers_are_the_closed_form_on_the_partly_edited_model(llama):
    hp = presets.HPARAMS["tiny-llama"]["memit"]
    sites = [presets.HPARAMS["tiny-llama"]["site"].format(l) for l in hp["layers"]]
    last = BLOCK["tiny-llama"].format(hp["layers"][-1])
    Cs = {s: stats.second_moment(llama, s, TEXTS) for s in sites}
    Z = targets.optimize_residual(llama, hp["layers"][-1], REQUESTS, TEMPLATES, **hp["opt"])
    patches = presets.memit(llama, REQUESTS, Cs, sites=sites, lam=hp["lam"], templates=TEMPLATES, **hp["opt"])
    assert [p.site for p in patches] == sites and all(p.check["ok"] for p in patches)
    edited = {s: llama.weight(s) for s in sites}
    llama.reset()
    for j, (site, p) in enumerate(zip(sites, patches)):  # replay: layer j saw layers < j edited
        K = memit_keys(llama, site, REQUESTS)
        R = (Z - residual(llama, last, REQUESTS)) / (len(sites) - j)
        ref = R @ K.T @ np.linalg.inv(K @ K.T + hp["lam"] * Cs[site].C)  # memit_main.py: resid @ solve(λC + KKᵀ, K).T
        assert np.allclose(p.delta, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())
        llama.apply(p)
    assert all(np.array_equal(llama.weight(s), edited[s]) for s in sites)


def test_alphaedit_layers_are_the_closed_form_and_keep_the_eigenspace(llama):
    hp = presets.HPARAMS["tiny-llama"]["alphaedit"]
    sites = [presets.HPARAMS["tiny-llama"]["site"].format(l) for l in hp["layers"]]
    last = BLOCK["tiny-llama"].format(hp["layers"][-1])
    C0s = {s: stats.second_moment(llama, s, TEXTS) for s in sites}
    U = {s: stats.top_eigenspace(C0s[s], hp["tau"]).basis for s in sites}
    assert all(0 < U[s].shape[1] < U[s].shape[0] for s in sites)  # τ = 2e-2 cuts inside the spectrum
    kw = dict(sites=sites, tau=hp["tau"], l2=hp["l2"], templates=TEMPLATES, **hp["opt"])
    first, prev = presets.alphaedit(llama, REQUESTS[:2], C0s, **kw)
    for s in sites:  # the Grams hold the keys re-read after every layer was edited
        K = memit_keys(llama, s, REQUESTS[:2])
        assert np.allclose(prev[s], K @ K.T, rtol=1e-10, atol=1e-12)
    # a second, sequential call: its solves see the first call's keys through prev
    Z = targets.optimize_residual(llama, hp["layers"][-1], REQUESTS[2:], TEMPLATES, **hp["opt"])
    second, prev2 = presets.alphaedit(llama, REQUESTS[2:], C0s, prev=prev, **kw)
    for s in sites:
        K = memit_keys(llama, s, REQUESTS[2:])
        assert np.allclose(prev2[s], prev[s] + K @ K.T, rtol=1e-10, atol=1e-12)
    llama.reset()
    for p in first:
        llama.apply(p)
    for j, (site, p) in enumerate(zip(sites, second)):  # replay the second call layer by layer
        K = memit_keys(llama, site, REQUESTS[2:])
        R = (Z - residual(llama, last, REQUESTS[2:])) / (len(sites) - j)
        P = np.eye(K.shape[0]) - U[site] @ U[site].T
        # AlphaEdit_main.py: solve(P (K Kᵀ + cache_c) + L2·I, P K residᵀ), transposed to (out, in)
        ref = np.linalg.solve(P @ (K @ K.T + prev[site]) + hp["l2"] * np.eye(K.shape[0]), P @ K @ R.T).T
        assert np.allclose(p.delta, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())
        llama.apply(p)
    for p in first + second:
        assert np.linalg.norm(p.delta @ U[p.site]) <= 1e-12 * np.linalg.norm(p.delta) and p.check["ok"]
