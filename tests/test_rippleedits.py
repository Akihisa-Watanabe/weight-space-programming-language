"""RippleEdits' answer check, case rules and averaging decide every number, and a wrong one would not show, so they
are checked on hand-made examples and continuations. No model and no download."""
import json

import pytest

from evals import rippleedits as re_


def query(prompt, *answers):
    return {"prompt": prompt, "answers": [{"value": v, "aliases": list(a)} for v, *a in answers],
            "query_type": "regular", "target_ids": [v for v, *_ in answers]}


def case(tests, conditions=(), test_condition="OR"):
    return {"test_queries": list(tests), "condition_queries": list(conditions), "test_condition": test_condition}


def example(**groups):
    fact = {"prompt": "The capital of {} is", "subject": "X", "target_new": "Rome",
            "query": query("The capital of X is", ("Rome",))}
    return {"fact": fact, **{name: groups.get(g, []) for g, name in re_.GROUPS.items()}}


def test_answered_needs_every_answer_and_drops_one_letter_aliases():
    q = query("p", ("United States", "USA", "U"), ("France",))
    assert re_.answered(" the USA and France", q)
    assert not re_.answered(" the USA", q)  # every answer must occur
    assert not re_.answered(" U and France", q)  # the one-letter alias "U" is dropped
    assert re_.answered(" in 3 years", query("p", ("3",)))  # a one-digit number is kept
    assert not re_.answered(" rome", query("p", ("Rome",)))  # case-sensitive


def test_cases_run_on_conditions_before_and_pass_on_tests_after():
    a, b, c = query("A is", ("a1",)), query("B is", ("b1",)), query("C is", ("c1",))
    ex = example(
        SA=[case([a, b]), case([a, b], test_condition="AND")],  # OR passes with one, AND needs both
        RS=[case([a], conditions=[c]), case([b], conditions=[c, b])],  # the second's conditions fail before
        PV=[case([a], conditions=[b])],  # no case runs: None, left out of the means
    )
    pre = re_.judge(ex, {"The capital of X is": " Paris", "A is": " a1", "B is": " no", "C is": " c1"})
    post = re_.judge(ex, {"The capital of X is": " Rome", "A is": " a1", "B is": " no", "C is": " no"})
    assert not pre["fact"] and post["fact"]
    s = re_.scores(pre, post)
    assert s == {"edit": True, "LG": None, "CI": None, "CII": None, "SA": 0.5, "PV": None, "RS": 1.0}

    failed = {**s, "SA": 0.0, "RS": 0.0, "edit": False}  # an edit that did not take is left out of the means
    other = {**s, "SA": 1.0, "RS": None}
    out = re_.summarize([s, failed, other])
    assert out["edit"] == pytest.approx(2 / 3)
    assert (out["SA"], out["SA_n"], out["RS"], out["RS_n"], out["PV"], out["PV_n"]) == (0.75, 2, 1.0, 1, None, 0)
    assert out["avg"] == pytest.approx((0.75 + 1.0) / 2)


def test_load_reads_the_edit_and_skips_unlabelled_ones(tmp_path):
    q = {**query("The capital of Narnia is", ("Rome", "Roma")), "target_ids": ["Q220"]}
    two_hop = {**query("The currency of the capital of Narnia is", ("lira",)), "query_type": "two_hop",
               "target_ids": ["Q220"], "second_hop_target_ids": ["Q1"]}
    groups = {name: [] for name in re_.GROUPS.values()}
    examples = [
        {"example_type": "random", **groups, "Subject_Aliasing": [case([q, two_hop])],
         "edit": {"prompt": "The name of the capital city of Narnia is Rome.", "relation": "CAPITAL",
                  "subject_id": "Q9", "target_id": "Q220"}},
        {"example_type": "recent", **groups,
         "edit": {"prompt": "The name of the capital city of  is Oslo.", "relation": "CAPITAL",
                  "subject_id": "Q8", "target_id": "Q585"}},
    ]
    (tmp_path / "random.json").write_text(json.dumps(examples))
    [ex] = re_.load("random", data_dir=str(tmp_path))
    assert ex["id"] == 0
    assert ex["fact"] == {"prompt": "The name of the capital city of {} is", "subject": "Narnia", "target_new": "Rome",
                          "query": {"prompt": "The name of the capital city of Narnia is",
                                    "answers": [{"value": "Rome", "aliases": ["Roma"]}]}}  # aliases from Q220's query
