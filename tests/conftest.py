"""Test session setup: use the tiny models in data/hf and never contact the Hub. Test modules import HF and hooked."""
import os
from pathlib import Path

import torch

os.environ["HF_HUB_OFFLINE"] = "1"  # read once when huggingface_hub is imported
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"  # tiny-gpt2 is .bin-only: never ask the Hub to convert it

HF = Path(__file__).resolve().parents[1] / "data" / "hf"


def hooked(model, site, text, which="out"):
    """A module's input or output at every token of one unpadded forward, (seq, dim), from a plain forward hook."""
    cap = {}
    h = model.module(site).register_forward_hook(lambda m, args, out: cap.update(x=args[0] if which == "in" else out))
    try:
        with torch.no_grad():
            model.hf(**model.tok(text, return_tensors="pt"), use_cache=False)
    finally:
        h.remove()
    return cap["x"][0].double().numpy()
