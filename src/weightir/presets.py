# Adapted from github.com/kmeng01/memit (MIT License, Copyright (c) 2022 Kevin Meng): memit/memit_main.py, memit/compute_ks.py, hparams/MEMIT/*.json; github.com/kmeng01/rome (MIT License, Copyright (c) 2022 Kevin Meng): rome/rome_main.py, hparams/ROME/*.json; and github.com/jianghoucheng/AlphaEdit (MIT License, Copyright (c) 2025 jianghc): AlphaEdit/AlphaEdit_main.py, hparams/{AlphaEdit,MEMIT,ROME}/*.json
"""ROME, EMMET, MEMIT and AlphaEdit as WeightIR edits of MLP output projections.

Every solve goes through weightir.compile. ROME and EMMET return one Edit. MEMIT and AlphaEdit write a target z into
the residual stream after the last edited layer and spread it over the layers: site by site, in layer order, keys
and the remaining residual are measured on the partially edited model, and that layer's Edit is solved and applied.
EMMET is written from Gupta et al. 2024 (arXiv:2403.14236); no code from its unlicensed repository was used.
"""
from __future__ import annotations

import numpy as np

from weightir import stats, targets
from weightir.compile import Patch, compile
from weightir.spec import Covariance, Edit, Exact, Keys, Soft, Values, edit

_ROME_PREFIXES = ((5, 10), (10, 10))  # (tokens, count): ROME samples 20 template prefixes
_MEMIT_PREFIXES = ((10, 5),)  # MEMIT, EMMET and AlphaEdit sample 5

# Per model: the Hub id and the edited site of a layer; per method: layers, context templates
# (stats.context_templates(model, length_params=...)), the δ optimization ("opt", targets.py) and solver weights.
# mom2_adjustment=False means ROME without C: rome(..., C=None). EMMET's stated objective needs no weights.
HPARAMS = {
    "gpt2-xl": {
        "model": "openai-community/gpt2-xl",
        "site": "transformer.h.{}.mlp.c_proj",
        "rome": dict(layers=[17], mom2_adjustment=True, length_params=_ROME_PREFIXES, opt=dict(
            steps=20, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=4.0)),
        "emmet": dict(layers=[17], length_params=_MEMIT_PREFIXES, opt=dict(
            steps=20, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "memit": dict(layers=[13, 14, 15, 16, 17], lam=20000.0, length_params=_MEMIT_PREFIXES, opt=dict(
            steps=20, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "alphaedit": dict(layers=[13, 14, 15, 16, 17], tau=2e-2, l2=10.0, length_params=_MEMIT_PREFIXES, opt=dict(
            steps=20, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
    },
    "gpt-j-6b": {
        "model": "EleutherAI/gpt-j-6b",
        "site": "transformer.h.{}.mlp.fc_out",
        "rome": dict(layers=[5], mom2_adjustment=True, length_params=_ROME_PREFIXES, opt=dict(
            steps=20, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=4.0)),
        "emmet": dict(layers=[5], length_params=_MEMIT_PREFIXES, opt=dict(
            steps=25, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "memit": dict(layers=[3, 4, 5, 6, 7, 8], lam=15000.0, length_params=_MEMIT_PREFIXES, opt=dict(
            steps=25, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "alphaedit": dict(layers=[3, 4, 5, 6, 7, 8], tau=2e-2, l2=10.0, length_params=_MEMIT_PREFIXES, opt=dict(
            steps=25, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
    },
    "llama-3-8b": {  # the AlphaEdit repository's configs; there is no EMMET config for Llama-3
        "model": "meta-llama/Meta-Llama-3-8B",
        "site": "model.layers.{}.mlp.down_proj",
        "rome": dict(layers=[5], mom2_adjustment=False, length_params=_ROME_PREFIXES, opt=dict(
            steps=25, lr=0.5, weight_decay=1e-3, kl_factor=0.0625, clamp_norm_factor=4.0)),
        "memit": dict(layers=[4, 5, 6, 7, 8], lam=15000.0, length_params=_MEMIT_PREFIXES, opt=dict(
            steps=25, lr=0.1, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "alphaedit": dict(layers=[4, 5, 6, 7, 8], tau=2e-2, l2=10.0, length_params=_MEMIT_PREFIXES, opt=dict(
            steps=25, lr=0.1, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
    },
    # Tests only. Both have 2 layers; an edit in the last one cannot reach the positions where the target is scored,
    # so ROME and EMMET edit layer 0. tiny-gpt2 has d = 2 and singular second moments: ROME with a given C only.
    "tiny-gpt2": {
        "model": "sshleifer/tiny-gpt2",
        "site": "transformer.h.{}.mlp.c_proj",
        "rome": dict(layers=[0], mom2_adjustment=True, length_params=((5, 2),), opt=dict(
            steps=5, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=4.0)),
    },
    "tiny-llama": {
        "model": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "site": "model.layers.{}.mlp.down_proj",
        "rome": dict(layers=[0], mom2_adjustment=True, length_params=((5, 2),), opt=dict(
            steps=5, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=4.0)),
        "emmet": dict(layers=[0], length_params=((5, 2),), opt=dict(
            steps=5, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "memit": dict(layers=[0, 1], lam=0.1, length_params=((5, 2),), opt=dict(
            steps=5, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
        "alphaedit": dict(layers=[0, 1], tau=2e-2, l2=10.0, length_params=((5, 2),), opt=dict(
            steps=5, lr=0.5, weight_decay=0.5, kl_factor=0.0625, clamp_norm_factor=0.75)),
    },
}


def rome(model, request: targets.Request, C: Covariance | None, *, site: str, templates, **opt) -> Edit:
    """ROME: the key k̄ maps exactly to v*, minimizing tr(ΔW C ΔWᵀ) (C=None: ‖ΔW‖², mom2_adjustment=False).

    k̄ is the site input at the subject's last token averaged over ``templates`` ("{}" first; ROME's are
    stats.context_templates(model)), and v* = W k₀ + δ comes from targets.optimize_values(..., **opt).
    The original ROME code takes the direction C⁻¹k̄ but meets v* at the bare-prompt key k₀; here the constraint uses
    k̄ as well (r-ROME, Gupta et al. 2024).
    """
    K = stats.keys(model, site, [request.prompt], [request.subject], templates)
    V = targets.optimize_values(model, site, [request], templates, **opt)
    return edit(site, map=Exact(K, V), minimize=C)


def _keys(model, site: str, requests, templates) -> Keys:
    """MEMIT's key (compute_ks): the mean of the bare-prompt key and of the key averaged over the other templates."""
    prompts, subjects = [r.prompt for r in requests], [r.subject for r in requests]
    groups = [templates[:1], templates[1:]] if len(templates) > 1 else [templates]
    return Keys(np.mean([stats.keys(model, site, prompts, subjects, g).data for g in groups], axis=0), site)


def _residual(model, layer: int, requests) -> np.ndarray:
    """Output of decoder block ``layer`` at the subject's last token of each bare prompt, (d_model, p)."""
    texts = [r.prompt.replace("{}", r.subject, 1) for r in requests]
    pos = [stats.subject_last_index(model.tok, r.prompt, r.subject) for r in requests]
    return stats.collect(model, targets.block(model, layer), texts, pos, which="out")


def emmet(model, requests, C: Covariance, *, site: str, templates, **opt) -> Edit:
    """EMMET's stated objective: every request's key maps exactly to its value, minimizing tr(ΔW C ΔWᵀ).

    As in the reference, which runs MEMIT's machinery on one layer: z is optimized on the output of the site's
    decoder block (targets.optimize_residual), K holds MEMIT's keys and V = W K + (z − h), h being that block's
    current output. The reference code adds emmet_lambda·I to KᵀC⁻¹K for batches, which turns it into MEMIT with
    λ = emmet_lambda·mom2_update_weight; here the maps are exact.
    """
    layer = targets.layer_of(model, site)
    Z = targets.optimize_residual(model, layer, requests, templates, **opt)
    K = _keys(model, site, requests, templates)
    R = Z - _residual(model, layer, requests)
    return edit(site, map=Exact(K, Values(model.weight(site) @ K.data + R, site)), minimize=C)


def _spread(model, requests, sites, templates, opt: dict, layer_edit) -> list[Patch]:
    """MEMIT's multi-layer loop: z at the block of the last site, on the model as given; then, for each site in
    layer order, keys and the residual z − h on the partially edited model, this layer's share of the residual,
    layer_edit(site, K, V) solved and applied."""
    layers = [targets.layer_of(model, s) for s in sites]
    if layers != sorted(set(layers)):
        raise ValueError(f"sites must lie in strictly increasing layers, got layers {layers}")
    Z = targets.optimize_residual(model, layers[-1], requests, templates, **opt)
    patches = []
    for j, site in enumerate(sites):
        K = _keys(model, site, requests, templates)
        # share 1/(number of sites left), = 1/(L − l + 1) for contiguous layers, as in the reference code
        R = (Z - _residual(model, layers[-1], requests)) / (len(sites) - j)
        W = model.weight(site)
        patches.append(compile(layer_edit(site, K, Values(W @ K.data + R, site)), W))
        model.apply(patches[-1])
    return patches


def memit(model, requests, Cs: dict, *, sites, lam: float, templates, **opt) -> list[Patch]:
    """MEMIT: each layer minimizes ‖ΔW K − R‖² + lam·tr(ΔW C ΔWᵀ), as Soft maps and minimize=Covariance(lam·C), with
    C = Cs[site] the per-token mean second moment of the unedited model. ``opt`` goes to targets.optimize_residual.
    The patches are applied to ``model`` (model.reset() undoes them) and returned."""
    def layer_edit(site, K, V):
        return edit(site, map=Soft(K, V), minimize=Covariance(lam * Cs[site].C, site, n=Cs[site].n))

    return _spread(model, requests, sites, templates, opt, layer_edit)


def alphaedit(model, requests, C0s: dict, *, sites, tau: float, l2: float, prev: dict | None = None,
              templates, **opt) -> tuple[list[Patch], dict]:
    """AlphaEdit: like memit, but each layer minimizes ‖ΔW K − R‖² + tr(ΔW G ΔWᵀ) + l2‖ΔW‖² subject to ΔW U = 0.

    U holds the eigenvectors of C0s[site] (unedited model) with eigenvalue > tau, and G = prev[site] is the Gram
    Σ K Kᵀ of the keys of earlier calls (sequential editing; none on the first call). Returns the patches, applied to
    ``model``, and prev plus this call's Grams, whose keys are re-read after all layers are edited.
    """
    def layer_edit(site, K, V):
        keep = [Exact(stats.top_eigenspace(C0s[site], tau))]
        if prev is not None and site in prev:
            keep.append(Soft(Covariance(prev[site], site), weight=1.0))
        return edit(site, map=Soft(K, V), preserve=keep, minimize=Covariance(l2 * np.eye(K.data.shape[0]), site))

    patches = _spread(model, requests, sites, templates, opt, layer_edit)
    grams = dict(prev or {})
    for site in sites:
        K = _keys(model, site, requests, templates).data
        grams[site] = grams.get(site, 0.0) + K @ K.T
    return patches, grams
