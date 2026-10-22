# AGENTS.md

## Overview

This is the repository for the blog **XXX**

The research: instead of having an LLM output weight values, have it write a **typed weight-edit program** in a DSL embedded in Python (working name WeightIR). A compiler turns the program into a constrained optimization problem on one weight matrix, solves it (in closed form where possible), re-checks the constraints on the original data, and returns the weight change ΔW. The default problem:

```text
minimize    tr(ΔW C ΔWᵀ)       size of the change, measured by the second moment C of the site's inputs
subject to  (W + ΔW) K = V     keys K map to values V
            ΔW U = 0           outputs for inputs in the preserved subspace span(U) do not change
budget      rank(ΔW) ≤ r, ‖ΔW‖ ≤ B
```

The LLM decides the site (layer and matrix), keys, values, and preserved subspace; the compiler computes the numbers. ROME, MEMIT, EMMET, and AlphaEdit are presets of this problem or of its soft-constraint form. The same constraints also compile into training update rules applied at every step, so the LLM writes the rule once and training runs it without the LLM; this is the "agentic optimization" in the repository name.

The main experiment asks whether, on the same numerical backend, an LLM writes better edits in this DSL than in plain Python or JSON. Planned layout: `src/weightir/` (DSL, compiler, solvers, checks; no LLM dependency), `src/agentic/` (LLM planner and search), `evals/` (testbeds and baselines). See `docs/architecture.md`.

Status: design stage. There is no code yet.

## Who the user is

The user is a researcher or coauthor developing the method. They use this repository to implement
it and to run the experiments for the paper.

## Read before changing things

Read `docs/` before working on this repo. It is the design source of truth.

- `architecture.md` — module layout and the boundaries between them
- `experiments.md` — experiment definitions

`docs/` is user-authored. Do not edit it unless asked.

Most of what needs implementing already exists in `third_party/`. Read it before writing anything from scratch. Never import from it — copy the code in and adapt it. Everything, including training and evaluation, must still run with `third_party/` deleted; downloaded checkpoints and datasets are data, not code, and live under `data/`. The only exception is one-off parity checks whose purpose is comparing our port against the reference.

Answer these in the plan, in writing:

1. Why is this change needed? What does the user do that leads to it, and
   in what environment?
2. Does that situation actually happen? If not, why does it still deserve priority?

If the user's request cannot be met within the design, do not bend the design. 

## Keep it simple

Follow the KISS principle: write the simplest code that does what was asked. This is research code.

- **No backward compatibility and no migration code.** When a format, config, or interface changes, change it everywhere and delete the old form. Do not add shims, aliases, fallbacks, version checks, or converters to keep old checkpoints, data, or configs working. If a change stops something existing from loading, say so instead of writing code around it.
- **Remove what a change makes obsolete.** In the same change, delete the code, configs, tests, and comments that are no longer used, and fix whatever the change makes wrong. Do not leave dead code, commented-out code, or outdated documentation behind. For `docs/`, which you do not edit unasked, report what has become stale.
- **Do not over-build or over-test.** When the task is a dog house, do not write a test for every nail. Test the behaviour where a silent mistake would corrupt results, and stop there. Do not add options, abstractions, or helpers that the task does not need.

## Required design choices

Do not add any of the following mechanisms:

- SHA- or digest-based identity;
- manifests or artifact inventories;
- signatures or attestations;
- provenance frameworks;
- promotion stages;
- sealed or frozen result objects;
- preflight modes;
- qualification stages;
- campaign identity systems;
- evidence ledgers;
- release channels;
- readiness levels;
- claim/evidence/derivation frameworks.

## Before You Use this Server

CPU1 (socket 0, NUMA node 0, cores 0-15 and 32-47) is unstable. Run everything on CPU2 (socket 1, NUMA node 1, cores 16-31 and 48-63). GPU 0 and GPU 1 are attached to CPU1, so do not use them; use only GPU 2 and GPU 3, which are attached to CPU2:

```sh
CUDA_VISIBLE_DEVICES=2,3 taskset -c 16-31,48-63 python ...
```

With this setting, GPU 2 and GPU 3 appear as `cuda:0` and `cuda:1` inside the process. `numactl` is not installed, so use `taskset`.
