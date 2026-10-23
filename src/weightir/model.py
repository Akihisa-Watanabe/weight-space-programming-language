"""Sites (the weight matrices an edit can change) and how their weights are read and patched in the WeightIR
convention, for any torch model and for a Hugging Face causal LM (Model).

A site is a torch.nn.Linear or transformers Conv1D module. Its weight W is (out_dim, in_dim) and the module computes
W x (+ bias). Conv1D (GPT-2) stores W transposed, as (in_dim, out_dim); read_weight() and add_to_weight() convert.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from weightir.nethook import get_module


def sites(model: torch.nn.Module) -> dict[str, tuple[torch.nn.Module, bool, bool]]:
    """name -> (module, transposed, has a trainable bias) for every trainable Linear or Conv1D layer whose weight is
    not shared with another parameter (a tied lm_head is left out: editing it would also change the embedding)."""
    uses = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    return {name: (m, isinstance(m, Conv1D), m.bias is not None and m.bias.requires_grad)
            for name, m in model.named_modules()
            if isinstance(m, (torch.nn.Linear, Conv1D)) and m.weight.requires_grad and uses[id(m.weight)] == 1}


def read_weight(param: torch.Tensor, transposed: bool) -> np.ndarray:
    """W as an (out, in) float64 copy."""
    w = param.detach().cpu().double()  # .cpu() before the dtype change: MPS converts wrongly otherwise
    return (w.T if transposed else w).numpy().copy()


def add_to_weight(param: torch.Tensor, transposed: bool, delta) -> None:
    """Add an (out, in) change to the parameter in place, in its dtype and device: W + ΔW is formed in float64 and
    rounded once, as check(..., dtype=...) assumes."""
    d = torch.as_tensor(np.asarray(delta, dtype=np.float64))
    with torch.no_grad():
        param.copy_((param.detach().cpu().double() + (d.T if transposed else d)).to(param.dtype))


class Model:
    """A Hugging Face causal LM and its tokenizer; sites are addressed by module name."""

    def __init__(self, hf, tok, name: str):
        self.hf, self.tok, self.name = hf, tok, name
        self._original: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return next(self.hf.parameters()).device

    @classmethod
    def load(cls, name: str, *, cache_dir: str = "data/hf", dtype=torch.float32, device: str | None = None) -> Model:
        """Load in ``dtype`` (float32 by default, whatever the checkpoint stores), on ``device`` (CUDA if available,
        else Apple's MPS, else CPU), in eval mode. A tokenizer without a pad token pads with EOS; padding side is set
        per call."""
        tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        hf = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, dtype=dtype)
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                            else "cpu")
        return cls(hf.to(device).eval(), tok, name)

    def _tied(self, weight: torch.nn.Parameter) -> bool:
        return sum(p is weight for _, p in self.hf.named_parameters(remove_duplicate=False)) > 1

    def module(self, site: str) -> torch.nn.Module:
        return get_module(self.hf, site)

    def param(self, site: str) -> torch.nn.Parameter:
        return self.module(site).weight

    def transposed(self, site: str) -> bool:
        """True when the weight is stored (in, out) (Conv1D), False for (out, in) (Linear)."""
        m = self.module(site)
        if isinstance(m, Conv1D):
            return True
        if isinstance(m, torch.nn.Linear):
            return False
        raise ValueError(f"{site} is a {type(m).__name__}, not a Linear or Conv1D site")

    def weight(self, site: str) -> np.ndarray:
        """W as an (out, in) float64 copy."""
        return read_weight(self.param(site), self.transposed(site))

    def apply(self, patch) -> None:
        """Add ``patch.delta`` ((out, in)) to the weight of ``patch.site``, in place, in the parameter's dtype and
        device. W + ΔW is formed in float64 and rounded once, as check(..., dtype=...) assumes."""
        p, t = self.param(patch.site), self.transposed(patch.site)
        if self._tied(p):
            raise ValueError(f"{patch.site} shares its weight with another parameter (tied embeddings); "
                             "editing it would change both")
        d = torch.as_tensor(np.asarray(patch.delta, dtype=np.float64))
        shape = tuple(p.shape[::-1]) if t else tuple(p.shape)  # (out, in)
        if tuple(d.shape) != shape:
            raise ValueError(f"delta for {patch.site} is {tuple(d.shape)}, but W is {shape} (out, in)")
        if not torch.isfinite(d).all():
            raise ValueError(f"delta for {patch.site} contains NaN or inf")
        if patch.site not in self._original:
            self._original[patch.site] = p.detach().clone()
        add_to_weight(p, t, d)

    def reset(self) -> None:
        """Restore every site touched by apply() to its weight before the first apply."""
        with torch.no_grad():
            for site, w in self._original.items():
                self.param(site).copy_(w)
        self._original.clear()
