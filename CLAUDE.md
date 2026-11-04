# Working on Weight-Space Programming Language

## Research direction

Build a language for reading, implementing, and changing computations inside trained neural networks. The central question is: **what computations can this language implement or change inside a pretrained model?**

Mechanistic interpretability supplies analysis tools and methods for translating operations on features and circuits into weight changes. Preserving behavior and limiting update size are optional task requirements. Direct weight manipulation, numerical solving, and learning can all serve the language. Do not force every operation through the current solver.

WeightIR currently implements edits to individual matrices, numerical checks, language-model access, fact-editing methods, and repeated edits for training. The LLM loop searches training programs. Feature and circuit operations and an agent for requested model changes are not implemented.

Focus on useful operations in pretrained models.

## Documentation and changes

The user is a researcher developing this system. Read the [design](docs/architecture.md) and [experiments](docs/experiments.md) before making changes. These documents are user-authored; edit them when requested.

Write documentation in concise English for readers who have not seen the conversation. Explain operations in ordinary language, keep variable and API names in code blocks, and link to implementation details. Separate implemented capabilities, recorded results, and proposed work. Do not turn illustrative examples from discussion into roadmap commitments.

Before changing code, explain the concrete need and whether it occurs in the current workflow. Keep research code simple:

- Change interfaces everywhere; do not add compatibility layers, migrations, or fallback formats.
- Remove code, configuration, and comments made obsolete by a change. Update authorized documentation and report other stale documents.
- Test behavior where mistakes would corrupt results. Avoid elaborate infrastructure and tests that repeat the implementation.
- If a request conflicts with the design, resolve the conflict explicitly. A user's change of research direction takes precedence over the previous design.

Do not add digest-based identity, manifests, artifact inventories, signatures, attestations, provenance frameworks, promotion stages, sealed result objects, preflight modes, qualification stages, campaign identity systems, evidence ledgers, release channels, readiness levels, or claim/evidence/derivation frameworks.

## Reference code

Check the [reference implementations](third_party/README.md) before implementing a method. Adapt licensed code into this repository with attribution in the source and [license notices](THIRD_PARTY_NOTICES.md). Do not import reference checkouts at runtime. The project must work without them; the exception is a one-off comparison against an original implementation. Keep downloaded models, datasets, and results in the local data directory.

## Server setup

The first CPU socket is unstable. Use the second socket and its attached GPUs for Python, tests, and experiments:

```sh
CUDA_VISIBLE_DEVICES=2,3 taskset -c 16-31,48-63 uv run python ...
CUDA_VISIBLE_DEVICES=2,3 taskset -c 16-31,48-63 uv run pytest -q
```

The selected physical GPUs appear as the first and second CUDA devices inside the process. Do not use physical GPUs 0 or 1.

The home directory is on slow network storage. Keep the Python installation and virtual environment on machine-local storage. The checkout's virtual environment is a symlink; check its destination when moving between machines. A fresh local setup can use:

```sh
UV_PYTHON_INSTALL_DIR=/scratch/$USER/uv-python uv python install 3.13 --no-bin
mkdir -p /scratch/$USER/agentic_optimization/.venv
# Point the checkout's .venv symlink at the local directory before syncing.
UV_CACHE_DIR=/scratch/$USER/uv-cache uv sync --python /scratch/$USER/uv-python/cpython-3.13-linux-x86_64-gnu/bin/python3.13
```

LLM experiments read the Anthropic API key from the local, untracked environment file:

```sh
CUDA_VISIBLE_DEVICES=2,3 taskset -c 16-31,48-63 uv run --env-file .env python experiments/search.py --task mnist --method llm --trials 20
```
