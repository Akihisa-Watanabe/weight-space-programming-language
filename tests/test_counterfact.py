"""An offset, padding or BOS mistake in evaluate() would silently corrupt every CounterFact number, so it is compared
with an unpadded, one-sequence-at-a-time chain-rule computation. scores() and summarize() are checked against ROME's
formulas on hand-made numbers."""
import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evals import counterfact as cf
from evals.generation import greedy

from conftest import HF

GROUPS = ("rewrite", "paraphrase", "neighborhood")

# Record 0 of counterfact.json without its attribute and generation prompts, which evaluate() does not use.
RECORD = {
    "case_id": 0,
    "pararel_idx": 2796,
    "requested_rewrite": {
        "prompt": "The mother tongue of {} is",
        "relation_id": "P103",
        "target_new": {"str": "English", "id": "Q1860"},
        "target_true": {"str": "French", "id": "Q150"},
        "subject": "Danielle Darrieux",
    },
    "paraphrase_prompts": [
        "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native",
        "An album was recorded for Capitol Nashville but never released. Danielle Darrieux spoke the language",
    ],
    "neighborhood_prompts": [
        "The mother tongue of Léon Blum is",
        "The native language of Montesquieu is",
        "François Bayrou, a native",
        "The native language of Raymond Barre is",
        "Michel Rocard is a native speaker of",
        "Jacques Chaban-Delmas is a native speaker of",
        "The native language of François Bayrou is",
        "Maurice Genevoix, speaker of",
        "The mother tongue of François Bayrou is",
        "Melchior de Vogüé, speaker of",
    ],
}
# Multi-token targets of different lengths. tiny-gpt2 predicts " stairs" after almost any text, so there the argmax
# flag of TRUE is True for most prompts, and that of NEW is False although its first token is predicted.
NEW, TRUE = "stairs and Old Church Slavonic", "stairs stairs"


def logprob(model, ids):
    """log p(ids[1:] | ids[0]) of one unpadded sequence."""
    logp = model(torch.tensor([ids])).logits[0].double().log_softmax(-1)
    return logp[:-1].gather(1, torch.tensor(ids[1:])[:, None]).sum().item()


def reference(model, tok, prompt, target):
    """NLL per target token and argmax match of " target" after ``prompt``, from unpadded single sequences and the
    chain rule log p(target | prompt) = log p(prompt + target) - log p(prompt)."""
    p, ids = tok(prompt)["input_ids"], tok(f"{prompt} {target}")["input_ids"]
    assert ids[: len(p)] == p
    nll = -(logprob(model, ids) - logprob(model, p)) / (len(ids) - len(p))
    logits = model(torch.tensor([ids])).logits[0]
    return nll, logits[len(p) - 1 : -1].argmax(-1).tolist() == ids[len(p) :]


def one_continuation(model, tok, prompt, n=20):
    """The n-token greedy continuation of one unpadded prompt."""
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def flatten(result, key):
    return [x[key] for g in GROUPS for x in result[g]]


@pytest.mark.parametrize("name", [
    "sshleifer/tiny-gpt2",                               # no BOS, no pad token
    "trl-internal-testing/tiny-LlamaForCausalLM-3.2",    # BPE that adds <|begin_of_text|>, no pad token
    "hf-internal-testing/tiny-random-LlamaForCausalLM",  # SentencePiece that adds <s>, pads on the left by default
])
def test_evaluate_matches_unpadded_chain_rule(name):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=HF, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=HF, local_files_only=True, dtype=torch.float32)
    rr = {**RECORD["requested_rewrite"], "target_new": {"str": NEW}, "target_true": {"str": TRUE}}
    record = {**RECORD, "requested_rewrite": rr}
    prompts = [rr["prompt"].format(rr["subject"]), *RECORD["paraphrase_prompts"], *RECORD["neighborhood_prompts"]]
    with torch.no_grad():
        ref = {t: [reference(model, tok, p, t) for p in prompts] for t in (NEW, TRUE)}
        conts = [one_continuation(model, tok, p) for p in prompts]

    for batch_size in (32, 5):  # all 26 sequences in one padded batch; batches that split a prompt's two sequences
        got = cf.evaluate(model, tok, record, batch_size=batch_size)
        assert [len(got[g]) for g in GROUPS] == [1, 2, 10]
        assert flatten(got, "nll_new") == pytest.approx([nll for nll, _ in ref[NEW]], abs=1e-5)
        assert flatten(got, "nll_true") == pytest.approx([nll for nll, _ in ref[TRUE]], abs=1e-5)
        assert flatten(got, "argmax_new") == [hit for _, hit in ref[NEW]]
        assert flatten(got, "argmax_true") == [hit for _, hit in ref[TRUE]]
        # left-padded batches of up to 5 or all 13 prompts continue as each prompt does alone
        assert flatten(got, "text") == conts
        assert flatten(got, "gen_true") == [TRUE in c for c in conts]

    # one budget per prompt: each row of a batch is cut to its own length (RippleEdits' reference judging)
    budgets = [3, 11, 1, 7]
    with torch.no_grad():
        assert greedy(model, tok, prompts[:4], max_new_tokens=budgets) == [
            one_continuation(model, tok, p, n) for p, n in zip(prompts[:4], budgets)]


def test_scores_follow_rome():
    def entry(new, true, argmax_new=False, argmax_true=False, gen_new=False, gen_true=False):
        return {"nll_new": new, "nll_true": true, "argmax_new": argmax_new, "argmax_true": argmax_true,
                "gen_new": gen_new, "gen_true": gen_true}

    result = {
        "rewrite": [entry(1.0, 2.0, argmax_new=True, gen_new=True)],
        # target_new should win: it loses the first prompt, and a tie fails; a continuation naming both fails
        "paraphrase": [entry(1.0, 0.5, argmax_new=True, gen_new=True),
                       entry(3.0, 3.0, argmax_new=True, gen_new=True, gen_true=True)],
        # target_true should keep winning: it wins only the first prompt
        "neighborhood": [entry(2.0, 1.0, argmax_true=True, gen_true=True), entry(0.5, 3.0, argmax_true=True),
                         entry(4.0, 4.0, argmax_new=True, gen_new=True)],
    }
    s = cf.scores(result)
    e = math.exp
    assert s["ES"] == 1.0 and s["EM"] == pytest.approx(e(-1) - e(-2))  # mean of exp(-new) - exp(-true)
    assert s["PS"] == 0.0 and s["PM"] == pytest.approx((e(-1) - e(-0.5) + 0) / 2)
    assert s["NS"] == pytest.approx(1 / 3)
    assert s["NM"] == pytest.approx((e(-1) - e(-2) + e(-3) - e(-0.5) + 0) / 3)  # mean of exp(-true) - exp(-new)
    # MEMIT: target_new's flag for rewrite and paraphrase prompts, target_true's for neighborhood prompts
    assert [s["rewrite_acc"], s["paraphrase_acc"], s["neighborhood_acc"]] == pytest.approx([1, 1, 2 / 3])
    # generation: the right target named and the other not
    assert [s["rewrite_gen"], s["paraphrase_gen"], s["neighborhood_gen"]] == pytest.approx([1, 1 / 2, 1 / 3])


def test_summarize_takes_harmonic_mean_of_means():
    a = dict.fromkeys(cf.KEYS, 0.0) | {"ES": 1.0, "PS": 0.5, "NS": 0.5, "EM": 0.2}
    b = dict.fromkeys(cf.KEYS, 0.0) | {"ES": 1.0, "PS": 1.0, "NS": 0.7, "EM": 0.4}
    out = cf.summarize([a, b])
    assert [out["ES"], out["PS"], out["NS"], out["EM"]] == pytest.approx([1.0, 0.75, 0.6, 0.3])
    assert out["S"] == pytest.approx(0.75)  # 3 / (1/1 + 1/0.75 + 1/0.6), not the mean of the per-case scores
