"""Search for a training program (or a learning rate) on one task, then train the best on the evaluation seeds.

    uv run --env-file .env python experiments/search.py --task mnist --method llm --model claude-haiku-5-5
    python experiments/search.py --task cartpole --method adam-lr --trials 20

llm: agentic.loop.improve with the LLM --model. Every trial trains with ProgramOptimizer from the trial seed
(--trial-seed), and the LLM sees the trial's validation metrics (classification) or episode returns (CartPole), never
test numbers. adam-lr: Adam with one learning rate per trial, log-spaced from 1e-4 to 1e-1, on the same trial seed, so
both methods get the same number of trials. The score is the one the task text states: val_acc at the end of
training, or how early CartPole is solved. All trials go to data/runs/search/<task>-llm-<model>.json (rewritten after
every trial, with each program also as <task>-llm-<model>/trial-NN.py) or <task>-adam-lr.json. The best program or
learning rate is then trained on the evaluation seeds (classification 0-2, CartPole 0-4) and saved like any other
run, as "agent-<model>" (the model id without "claude-") or "adam-tuned".
"""
import argparse
import functools
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agentic import llm
from agentic.loop import best_trial, improve
from agentic.program import ProgramError, ProgramOptimizer, describe
from experiments.classification import train as classification
from experiments.rl import cartpole

DATA = Path(__file__).resolve().parents[1] / "data"
LR_RANGE = (1e-4, 1e-1)
SAME_SCORE = "among equal scores, less time is better."


@dataclass
class Task:
    text: str           # the task, as the training script describes it
    score_text: str     # how score() ranks a trial, in words
    sites: str          # the layers a program can edit
    train: Callable     # train(make_optimizer, seed) -> the run's record
    score: Callable     # record -> float, higher is better
    summary: Callable   # record -> a few words for the scoreboard
    report: Callable    # record -> what the LLM sees after a trial (never test numbers)
    save: Callable      # save(record, name, seed)
    seeds: tuple        # evaluation seeds
    budget: float       # seconds a program's trial may take (about 7x the example program's)

    @property
    def prompt(self) -> str:
        return (f"{self.text} {self.score_text} A trial that runs longer than {self.budget:.0f} s is stopped and "
                "counts as failed.")


def _edits_and_output(r: dict) -> list[str]:
    lines = []
    if r.get("edits"):
        lines.append("edits per site (count, mean ||dW||_F, mean map residual):")
        for site, (n, fro, res) in r["edits"].items():
            lines.append(f"- {site}: {n}, {fro / n:.4g}, {res / n:.4g}" if n else f"- {site}: no edits")
    printed = (r.get("output") or "").strip().splitlines()
    if printed:
        lines += ["what the program printed (last 30 lines):", *printed[-30:]]
    return lines


def classification_task(dataset: str) -> Task:
    data = classification.load(dataset)

    def val_acc(r):
        return r["epochs"][-1]["val_acc"]

    def seconds(r):
        return sum(e["seconds"] for e in r["epochs"])

    def report(r):
        c = r["curve"]
        return "\n".join([
            f"val_acc every {r['args']['eval_every']} steps: {[round(p['val_acc'], 4) for p in c]}",
            f"val_loss at the same steps: {[round(p['val_loss'], 4) for p in c]}",
            f"training loss, mean over each interval: {[round(p['train_loss'], 4) for p in c]}",
            f"val_acc at the end: {val_acc(r):.4f} after {r['steps']} steps; {seconds(r):.1f} s "
            f"({1000 * seconds(r) / r['steps']:.1f} ms per step)",
            *_edits_and_output(r)])

    return Task(text=classification.task(dataset),
                score_text=f"The score of a trial is val_acc at the end of training (higher is better); {SAME_SCORE}",
                sites=describe(classification.model()),
                train=lambda make, seed: classification.train(dataset, make, seed=seed, data=data),
                score=val_acc, summary=lambda r: f"val_acc {val_acc(r):.4f}, {seconds(r):.0f} s",
                report=report, save=lambda r, name, seed: classification.save(r, dataset, name, seed),
                seeds=(0, 1, 2), budget=300)


def cartpole_task() -> Task:
    def last100(r):
        return sum(r["returns"][-100:]) / len(r["returns"][-100:])

    def score(r):   # solved earlier is better; unsolved trials rank below every solved one, by their final mean
        return -float(r["solved_at"]) if r["solved_at"] else -float(r["args"]["episodes"] + 1) + last100(r) / 500

    def summary(r):
        what = (f"solved at episode {r['solved_at']}" if r["solved_at"] else
                f"weights became non-finite at episode {r['diverged_at']}" if r["diverged_at"] else
                f"not solved (mean of the last 100 episodes {last100(r):.0f})")
        return f"{what}, {r['seconds']:.0f} s"

    def report(r):
        ret = r["returns"]
        blocks = [round(sum(ret[i:i + 50]) / len(ret[i:i + 50])) for i in range(0, len(ret), 50)]
        return "\n".join([f"mean episode_return per 50 episodes: {blocks}",
                          f"{len(ret)} episodes in {r['seconds']:.1f} s", *_edits_and_output(r)])

    return Task(text=cartpole.task(),
                score_text=("The score of a trial is how early the mean episode_return over 100 consecutive episodes "
                            f"first reaches {cartpole.SPEC.reward_threshold:g} (earlier is better; a trial that never "
                            "gets there ranks below every one that does, by its final 100-episode mean); "
                            + SAME_SCORE),
                sites=describe(cartpole.model()),
                train=lambda make, seed: cartpole.train(make, seed=seed), score=score, summary=summary,
                report=report, save=cartpole.save, seeds=(0, 1, 2, 3, 4), budget=120)


def run_program(task: Task, seed: int):
    """run(code) for agentic.loop.improve: one trial of the program on ``seed``."""
    def run(code: str) -> dict:
        start = time.time()
        try:
            r = task.train(lambda model: ProgramOptimizer(model, program=code, deadline=start + task.budget), seed)
        except ProgramError as e:
            print(f"trial: failed at step {e.step}", flush=True)
            return {"score": -math.inf, "summary": f"failed at step {e.step}",
                    "report": f"The trial stopped at step {e.step}:\n```\n{e.traceback}```",
                    "seconds": time.time() - start}
        print(f"trial: {task.summary(r)}", flush=True)
        return {"score": task.score(r), "summary": task.summary(r), "report": task.report(r),
                "seconds": time.time() - start, "record": r}
    return run


def write_programs(result: dict, folder: Path) -> None:
    """Each trial's program as trial-NN.py, its score and the LLM's note in comments at the top, for reading."""
    folder.mkdir(parents=True, exist_ok=True)
    for t in result["trials"]:
        head = [f"# trial {t['trial']}: {t['summary']}", *(f"# {line}" for line in t["note"].splitlines()), ""]
        (folder / f"trial-{t['trial']:02d}.py").write_text("\n".join(head) + "\n" + t["program"])


def tune_lr(task: Task, trials: int, seed: int) -> dict:
    done = []
    for i, lr in enumerate(np.geomspace(*LR_RANGE, trials), 1):
        lr, start = float(lr), time.time()
        r = task.train(lambda model: torch.optim.Adam(model.parameters(), lr=lr), seed)  # noqa: B023 (called here)
        done.append({"trial": i, "lr": lr, "score": task.score(r), "summary": task.summary(r),
                     "seconds": time.time() - start, "record": r})
        print(f"trial: lr {lr:.2e}, {done[-1]['summary']}", flush=True)
    return {"trials": done, "best": best_trial(done)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["mnist", "fashion", "cartpole"], required=True)
    parser.add_argument("--method", choices=["llm", "adam-lr"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trial-seed", type=int, default=100, help="seed of every trial; not an evaluation seed")
    parser.add_argument("--model", default="claude-haiku-5-5", help="the LLM (llm only)")
    parser.add_argument("--effort", default="high", help="the LLM's effort (llm only)")
    args = parser.parse_args()
    task = cartpole_task() if args.task == "cartpole" else classification_task(args.task)
    model = args.model.removeprefix("claude-")
    out = DATA / "runs" / "search" / (f"{args.task}-llm-{model}.json" if args.method == "llm"
                                      else f"{args.task}-adam-lr.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(result: dict) -> None:
        out.write_text(json.dumps({"args": vars(args), **result}, indent=1))
        if args.method == "llm":
            write_programs(result, out.with_suffix(""))

    if args.method == "llm":
        chat = functools.partial(llm.chat, model=args.model, effort=args.effort)
        result = improve(task.prompt, task.sites, run_program(task, args.trial_seed), chat, trials=args.trials,
                         progress=write)
        program = result["trials"][result["best"]]["program"]
        name, chosen = f"agent-{model}", {"program": program}
        make = lambda net: ProgramOptimizer(net, program=program)  # noqa: E731
    else:
        result = tune_lr(task, args.trials, args.trial_seed)
        lr = result["trials"][result["best"]]["lr"]
        name, chosen = "adam-tuned", {"lr": lr}
        make = lambda net: torch.optim.Adam(net.parameters(), lr=lr)  # noqa: E731
    write(result)
    print(f"best: trial {result['best'] + 1}, {result['trials'][result['best']]['summary']}", flush=True)

    for seed in task.seeds:
        task.save({**task.train(make, seed), **chosen}, name, seed)


if __name__ == "__main__":
    main()
