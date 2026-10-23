"""Greedy continuations, which the generation metrics of evals.counterfact and evals.rippleedits read."""
import torch


@torch.no_grad()
def greedy(hf_model, tok, prompts: list[str], *, max_new_tokens: int = 20, batch_size: int = 32) -> list[str]:
    """The greedy continuation of each prompt: up to ``max_new_tokens`` tokens, fewer if EOS comes first, decoded
    without special tokens. Prompts (with BOS if the tokenizer adds one) are left-padded per batch and the mask hides
    the pads, so the result is that of generating one prompt at a time."""
    if hf_model.training:
        raise ValueError("call hf_model.eval() first: dropout would make the continuations random")
    eos = tok.eos_token_id
    out = []
    for s in range(0, len(prompts), batch_size):
        ids = tok(prompts[s:s + batch_size])["input_ids"]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([[eos] * (width - len(x)) + x for x in ids], device=hf_model.device)
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=hf_model.device)
        gen = hf_model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=eos)
        out += [tok.decode(row[width:], skip_special_tokens=True) for row in gen.tolist()]
    return out
