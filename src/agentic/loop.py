"""The outer loop: an LLM writes an edit program, a trial trains the network with it from a fresh start, the LLM sees
the trial's report and writes the next program. After the last trial the best one is kept.

    result = improve(task, sites, run, chat, trials=20)

``chat(system, messages)`` returns an agentic.llm.Reply (e.g. functools.partial(llm.chat, model=..., effort=...)).
``run(code)`` trains with the program (agentic.program.ProgramOptimizer) and returns {"score": higher is better,
"summary": a few words for the scoreboard, "report": what the LLM sees, "seconds": the trial's time, ...}; other keys
are kept with the trial. Reports never contain test-set numbers. The result holds every trial (program, the LLM's
note, tokens and the run's outcome), the index of the best trial (highest score, then fastest), the transcript (the
text of every message sent and received) and the total tokens.
"""
from __future__ import annotations

import re
from collections import Counter

from agentic.program import DECAY, EXAMPLE_PROGRAM, load, program_traceback

FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
ATTEMPTS = 3  # replies asked for one program before the search stops

SYSTEM = f'''You design how a neural network is trained. You write a weight-edit program that the training loop calls
at every training step, and you improve it over a series of trials.

## Trials
Each trial trains the network from the same fresh initialization with your program, for the whole training run the
task describes. After it you get the trial's report: the measurements the task names, training losses, time,
per-site edit statistics, the end of what your program printed, or the error that stopped it. Trials are
independent: a program that does badly costs only its own trial. When the trials are used up, the trial with the
best score is kept. The test set is never used in trials.

## Rules
Use only what ctx gives you: this step's batch (its keys, outputs and gradients), the weights, and the history
(losses, metrics, ctx.state). Do not create new or altered inputs (no data augmentation) and do not read data from
anywhere else: the improvement has to come from how the weights are edited. How you choose targets, which layers and
examples you edit, how many edits a step makes, and which constraints, budgets or averaging you use are up to you.

## Programs
A program defines

    def step(ctx):
        ...

The training loop calls step(ctx) once per training step, after it has computed the loss and its gradients on the
step's batch. The program changes weights only through ctx.apply(wir.edit(...)). weightir solves each edit in closed
form for one weight matrix (a "site"). With W of shape (out, in), keys K (in, p) and values V (out, p), the edit is

    minimize   tr(ΔW C ΔWᵀ)                       (C from `minimize`; the identity when None)
             + Σ soft maps       w·‖(W + ΔW)K − V‖²
             + Σ soft preserves  w·tr(ΔW C_j ΔWᵀ)
             + ridge·‖ΔW‖²
    subject to (W + ΔW)K = V    for exact maps
               ΔW U = 0         for exact preserves (inputs in span(U) keep their outputs)
    budget     rank(ΔW) ≤ r,  ‖ΔW‖ ≤ B

## weightir, imported as wir (numpy is np)
- wir.Keys(data (in, p), site) and wir.Values(data (out, p), site): one column per example.
- wir.Subspace(basis (in, s) with orthonormal columns, site); wir.span(vectors, site) is the exact span;
  wir.subspace(vectors, site, energy=0.99 | threshold=... | rank=...) estimates where vectors mostly lie.
- wir.Covariance(C (in, in), site).
- Maps: wir.Exact(K, V) and wir.Soft(K, V, weight=w). Preserves: wir.Exact(U) for a Subspace, wir.Soft(C, weight=w)
  for a Covariance.
- Budgets: wir.Rank(r), wir.Norm(B, kind="fro" or "metric").
- wir.edit(site, map=..., preserve=[...], minimize=Covariance or None, require=[...], ridge=0.0) returns an Edit;
  wir.joint(*edits) merges edits of one site into one problem.
- wir.SpecError means wrong types, sites or shapes. wir.Infeasible means no ΔW meets the edit; its .kind is
  "conflict", "rank", "norm", "unresolved" or "unsupported" and .details holds numbers. An error your program does
  not catch stops the trial.

## ctx
- ctx.sites: the editable layers that ran a training forward this step, in forward order.
- ctx.shape(site) -> (out, in); `in` counts the bias column when the layer has a bias.
- ctx.step: the number of this step (1 at the first call).
- ctx.losses: every training loss so far, oldest first. ctx.metrics: {{name: [(step, value), ...]}} for the other
  measurements the training loop reports (the task names them).
- ctx.keys(site) -> Keys (in, N): this step's inputs to the layer, one column per row of the batch (per token for
  sequences); the last row is 1 when the layer has a bias.
- ctx.outputs(site) -> array (out, N): the layer's outputs for those inputs.
- ctx.output_grads(site) -> array (out, N): d(loss)/d(outputs) from the backward pass.
- ctx.values(site, step) -> Values: outputs - step·N·output_grads, i.e. each example's output moved against its own
  loss gradient when the loss is a mean over the N rows.
- ctx.covariance(site, damping=1.0) -> Covariance: the running mean of k kᵀ over the inputs (decay {DECAY} per step)
  plus damping·mean(diag)·I.
- ctx.weight(site) -> array (out, in): the current [W | b].
- ctx.apply(edit) -> Patch: solves the edit and writes it to the model at once; patch.check holds residuals and norms.
- ctx.undo(patch): takes back a patch that ctx.apply returned.
- ctx.refresh() -> float: runs the training loop's forward and backward pass on this step's batch again, at the
  current weights; afterwards keys, outputs, output_grads and values describe those weights. Returns the loss. Each
  call costs one forward and one backward pass.
- ctx.state: a dict that keeps its contents from step to step.
- print() works; the end of what the program prints is in the report.

Until you call ctx.refresh(), keys, outputs and gradients describe the weights before this step's edits. A step that
applies nothing leaves the weights unchanged.

## Example
With a mean-reduced loss this program is a preconditioned gradient step with learning rate about 0.3. It trains the
networks it was tried on.
```python
{EXAMPLE_PROGRAM}```

## Output
Say in one or two sentences what you change and why, then give one ```python block with the complete program.'''

NO_BLOCK = ("I could not find a ```python block that defines step(ctx). Reply with the complete program in one "
            "```python block.")
TRUNCATED = "Your reply was cut off at the output limit. Reply with only the ```python block."


def extract_block(text: str) -> str | None:
    """The last ```python block, else the last untagged block, else None."""
    blocks = [(info.strip().lower(), body) for info, body in FENCE.findall(text)]
    for wanted in ({"python", "py", "python3"}, {""}):
        found = [body for info, body in blocks if info in wanted]
        if found:
            return found[-1]
    return None


def best_trial(trials: list[dict]) -> int:
    """Index of the trial with the highest score, the fastest among equal scores."""
    return max(range(len(trials)), key=lambda i: (trials[i]["score"], -trials[i]["seconds"]))


def improve(task: str, sites: str, run, chat, *, trials: int = 20, progress=None) -> dict:
    """Run ``trials`` trials of LLM-written programs (see the module docstring); ``sites`` describes the layers.
    progress(result so far), when given, is called after every trial, so a run that stops early keeps its trials."""
    messages, transcript, usage, done = [], [], Counter(), []

    def ask(text: str):
        """Send ``text``; return the first program that loads, with the LLM's note and the tokens of the replies."""
        messages.append({"role": "user", "content": text})
        transcript.append({"role": "user", "text": text})
        problem, used = "no reply", Counter()
        for _ in range(ATTEMPTS):
            reply = chat(SYSTEM, messages)
            used.update(reply.usage)
            messages.append({"role": "assistant", "content": reply.content})
            transcript.append({"role": "assistant", "text": reply.text})
            if reply.stop_reason == "refusal":
                raise RuntimeError("the LLM refused to write a program")
            if reply.stop_reason == "max_tokens":
                problem = TRUNCATED
            elif (code := extract_block(reply.text)) is None:
                problem = NO_BLOCK
            else:
                try:
                    load(code)
                except Exception as e:
                    problem = f"The program could not be loaded:\n```\n{program_traceback(e)}```\nFix it."
                else:
                    return code, FENCE.sub("", reply.text).strip(), used
            messages.append({"role": "user", "content": problem})
            transcript.append({"role": "user", "text": problem})
        raise RuntimeError(f"no usable program after {ATTEMPTS} replies: {problem}")

    text = (f"Task: {task}\n\nSites (name: out x in):\n{sites}\n\nYou have {trials} trials. "
            "Write the program for trial 1.")
    for i in range(1, trials + 1):
        code, note, used = ask(text)
        usage.update(used)
        done.append({"trial": i, "program": code, "note": note, "usage": dict(used), **run(code)})
        result = {"trials": done, "best": best_trial(done), "transcript": transcript, "usage": dict(usage)}
        if progress is not None:
            progress(result)
        board = "\n".join(f"- trial {t['trial']}: {t['summary']}" for t in done)
        text = (f"Trial {i}: {done[-1]['summary']}\n{done[-1]['report']}\n\n"
                f"Trials so far (the best is trial {result['best'] + 1}):\n{board}\n\n"
                f"{trials - i} trials left. Write the program for trial {i + 1}.")
    return result
