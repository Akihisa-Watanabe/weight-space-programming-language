# Adapted from edenbiran/RippleEdits (MIT License, Copyright (c) 2023 Eden Biran): src/relation.py, src/query.py, src/queryexecutor.py, src/testrunner.py, src/evaluation.py, src/modeleditor.py.
"""RippleEdits examples and the paper's generation tests of an edit's ripple effects.

An example edits one fact (subject, relation, new target) and holds test cases in six groups:
    LG   Logical_Generalization  facts that follow from the edit by the relation's logic
    CI   Compositionality_I      two-hop questions through the new target
    CII  Compositionality_II     the edited fact asked through another entity that names the subject
    SA   Subject_Aliasing        the edited fact asked with an alias of the subject
    PV   Forgetfulness           other targets of a one-to-many relation (the paper's Preservation)
    RS   Relation_Specificity    other facts about the subject, which should not change
A query is answered when, for each of its answers, the value or one of its aliases (of more than one character, or
a number) occurs in the greedy continuation of the prompt (case-sensitive substring; evals.generation.greedy). A
case is run only if the unedited model answers all its condition queries, and passes if any ("OR") or all ("AND")
of its test queries are answered. ``scores``: per group, the share of the run cases that pass. ``summarize``: its
mean over the examples whose edit took (the edited model answers the edit's own query) and which ran a case of that
group, as the reference's evaluation.py averages.

This follows the paper's text: an answer is looked for in a continuation of up to 20 tokens. The reference code,
which produced the paper's numbers, generates up to max_length = (characters of the prompt) + 30 tokens, prompt
included, and looks in the prompt as well; ``evaluate(..., reference=True)`` judges that way. Either way the model
is edited once per example (the reference edits once per group).
"""
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from evals.generation import greedy

URL = "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/{}.json"
SUBSETS = ("recent", "random", "popular")
GROUPS = {"LG": "Logical_Generalization", "CI": "Compositionality_I", "CII": "Compositionality_II",
          "SA": "Subject_Aliasing", "PV": "Forgetfulness", "RS": "Relation_Specificity"}
# relation.py's phrases, "<subject>" written as "{}"
PHRASES = {
    "MOTHER": "The name of the mother of {} is",
    "FATHER": "The name of the father of {} is",
    "BROTHER": "The name of the brother of {} is",
    "SISTER": "The name of the sister of {} is",
    "SIBLING": "The names of the siblings of {} are",
    "SPOUSE": "The name of the spouse of {} is",
    "UNCLE": "The name of the uncle of {} is",
    "AUNT": "The name of the aunt of {} is",
    "CHILD": "The name of the child of {} is",
    "NUMBER_OF_CHILDREN": "The number of children {} has is",
    "SEX_OR_GENDER": "The gender of {} is",
    "PLACE_OF_BIRTH": "The place of birth of {} is",
    "PLACE_OF_DEATH": "The place of death of {} is",
    "PLACE_OF_BURIAL": "The place of burial of {} is",
    "COUNTRY": "The name of the country which {} is associated with is",
    "COUNTRY_OF_CITIZENSHIP": "The name of the country of citizenship of {} is",
    "POSITION_HELD": "The name of the position held by {} is",
    "STEPFATHER": "The name of the stepfather of {} is",
    "STEPMOTHER": "The name of the stepmother of {} is",
    "MEMBER_OF_SPORTS_TEAM": "The name of the sports team which {} is a member of is",
    "ALMA_MATER": "The name of the alma mater of {} is",
    "SEXUAL_ORIENTATION": "The sexual orientation of {} is",
    "OCCUPATION": "The occupation of {} is",
    "EMPLOYER": "The name of the employer of {} is",
    "LEAGUE": "The name of the league which {} plays in is",
    "FIELD_OF_WORK": "The name of the field of work of {} is",
    "NATIVE_LANGUAGE": "The mother tongue of {} is",
    "AWARD_RECEIVED": "The name of the award {} won is",
    "FOLLOWS": "{} follows",
    "FOLLOWED_BY": "{} is followed by",
    "ETHNIC_GROUP": "The name of the ethnic group which {} is associated with is",
    "RELIGION": "The name of the religion which {} is associated with is",
    "EYE_COLOR": "The eye color of {} is",
    "DATE_OF_BIRTH": "The date of birth of {} is",
    "DATE_OF_DEATH": "The date of death of {} is",
    "IS_ALIVE": "Is {} still alive?",
    "HEAD_OF_GOVERNMENT": "The name of the head of government of {} is",
    "HEAD_OF_STATE": "The name of the head of state of {} is",
    "CONTINENT": "The name of the continent which {} is part of is",
    "CAPITAL": "The name of the capital city of {} is",
    "CURRENCY": "The name of the currency in {} is",
    "OFFICIAL_LANGUAGE": "The official language of {} is",
    "ANTHEM": "The name of the anthem of {} is",
    "LIKELY_ANTHEM": "The name of the anthem that is most likely to be performed in {} is",
    "CAPITAL_OF": "The name of the country which {} is the capital of is",
    "AUTHOR": "The name of the author of {} is",
    "EDITOR": "The name of the editor of {} is",
    "DIRECTOR": "The name of the director of {} is",
    "SCREENWRITER": "The name of the screenwriter of {} is",
    "CAST_MEMBER": "The names of the cast members of {} are",
    "ARCHITECT": "The name of the architect of {} is",
    "COMPOSER": "The name of the composer of {} is",
    "FOUNDER": "The name of the founder of {} is",
}


def _queries(example: dict):
    for name in GROUPS.values():
        for case in example[name]:
            yield from case["condition_queries"]
            yield from case["test_queries"]


def _fact(edit: dict, answers: dict) -> dict | None:
    """The edit as a ROME request and its own query, or None if its prompt has no subject or target label."""
    phrase = PHRASES[edit["relation"]]
    before, after = phrase.split("{}")
    m = re.fullmatch(re.escape(before) + "(.+?)" + re.escape(after) + r" (.+)\.", edit["prompt"])
    if m is None:
        return None
    subject, target = m.groups()
    query = {"prompt": phrase.format(subject),
             "answers": [answers.get(edit["target_id"], {"value": target, "aliases": []})]}
    return {"prompt": phrase, "subject": subject, "target_new": target, "query": query}


def load(subset: str, *, data_dir: str = "data/rippleedits", seed: int = 0) -> list[dict]:
    """Examples of ``subset`` (downloaded on first use) whose edit names its subject and target; the reference skips
    the others (109 of recent's 1,948). Each gets "id", its index in the file, and "fact": the edit's prompt with "{}"
    for the subject, the subject, target_new (no leading space) and the edit's own query, whose answer takes the
    target's aliases from the file's other queries. They come in an order shuffled by random.Random(seed), so the
    first n are a random sample (the paper evaluates 200 random examples per subset)."""
    path = Path(data_dir) / f"{subset}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(URL.format(subset), str(path))
    examples = json.loads(path.read_text(encoding="utf-8"))
    answers = {}
    for e in examples:
        for q in _queries(e):
            ids = q["second_hop_target_ids"] if q["query_type"] == "two_hop" else q["target_ids"]
            for t, a in zip(ids, q["answers"]):
                answers.setdefault(str(t), a)
    out = []
    for i, e in enumerate(examples):
        fact = _fact(e["edit"], answers)
        if fact is not None:
            out.append({**e, "id": i, "fact": fact})
    random.Random(seed).shuffle(out)
    return out


def answered(text: str, query: dict) -> bool:
    """Every answer of ``query`` has its value or an alias in ``text`` (reference: QueryExecutor._verify_answer);
    names of one character that are not a digit are dropped (Query._filter_answers)."""
    return all(any(n in text for n in [a["value"], *a["aliases"]] if len(n) > 1 or n.isdigit())
               for a in query["answers"])


def judge(example: dict, texts: dict[str, str]) -> dict:
    """Which queries the continuations ``texts`` (prompt -> continuation) answer: {"fact": the edit's own query,
    group: [{"conditions": all condition queries, "tests": the test queries under the case's OR / AND}]}."""
    def ok(q):
        return answered(texts[q["prompt"]], q)

    out = {"fact": ok(example["fact"]["query"])}
    for g, name in GROUPS.items():
        out[g] = []
        for case in example[name]:
            test = {"OR": any, "AND": all}[case["test_condition"]]
            out[g].append({"conditions": all(ok(q) for q in case["condition_queries"]),
                           "tests": test(ok(q) for q in case["test_queries"])})
    return out


@torch.no_grad()
def evaluate(hf_model, tok, example: dict, *, reference: bool = False, batch_size: int = 8) -> dict:
    """``judge`` on the greedy continuations of every distinct prompt of ``example`` (20 tokens), with them under
    "texts". reference=True: as QueryExecutor.execute_query, len(prompt) + 30 tokens in all, judged on prompt +
    continuation. Batches of 8 keep the KV cache of GPT-2 XL's long reference continuations within an 11 GB GPU."""
    prompts = sorted({example["fact"]["query"]["prompt"], *(q["prompt"] for q in _queries(example))})
    if reference:
        budgets = [len(p) + 30 - len(tok(p)["input_ids"]) for p in prompts]
        conts = greedy(hf_model, tok, prompts, max_new_tokens=budgets, batch_size=batch_size)
        texts = {p: p + c for p, c in zip(prompts, conts)}
    else:
        texts = dict(zip(prompts, greedy(hf_model, tok, prompts, max_new_tokens=20, batch_size=batch_size)))
    return {**judge(example, texts), "texts": texts}


def scores(pre: dict, post: dict) -> dict:
    """One example from ``evaluate`` before and after its edit: "edit", whether the edit took, and per group the
    share of passing cases (after) among those whose conditions held (before); None if no case of it ran."""
    out = {"edit": post["fact"]}
    for g in GROUPS:
        run = [b["tests"] for a, b in zip(pre[g], post[g]) if a["conditions"]]
        out[g] = float(np.mean(run)) if run else None
    return out


def summarize(example_scores: list[dict]) -> dict:
    """"edit": the share of edits that took. Per group: the mean share over the examples whose edit took and that
    ran a case of it (None if there are none), and how many there are ("<group>_n"). "avg": the mean over the groups
    that have one (the paper's average of the six)."""
    out = {"edit": float(np.mean([s["edit"] for s in example_scores]))}
    for g in GROUPS:
        xs = [s[g] for s in example_scores if s["edit"] and s[g] is not None]
        out[g], out[f"{g}_n"] = (float(np.mean(xs)) if xs else None), len(xs)
    means = [out[g] for g in GROUPS if out[g] is not None]
    out["avg"] = float(np.mean(means)) if means else None
    return out
