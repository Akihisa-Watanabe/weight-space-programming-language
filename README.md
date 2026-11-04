# Weight-Space Programming Language

A programming language for reading and changing computations inside trained neural networks.

**What computations can this language implement or change inside a pretrained model?**

We want people and LLMs to write programs that operate on a model's weights, features, and circuits. Mechanistic interpretability provides ways to find those computations; the language provides operations for changing them. Preserving existing behavior is an option when a task calls for it.

## Current implementation

WeightIR is the current Python-based implementation of one part of this language: writing changes to individual weight matrices. It turns input–output requirements into numerical updates and checks the result. This solver is one building block for the broader language.

| Component | Available now |
|---|---|
| Matrix compiler | Edits, optional constraints, and numerical checks |
| Model tools | Language-model access and standard fact-editing methods |
| Program search | Repeated edits for training, with an LLM trial loop |

Existing experiments cover GPT-2 fact editing and training small classification and control networks. They establish working components; programming broader computations in pretrained models remains the research task.

## Try the matrix compiler

This example changes a linear layer so that its first input direction maps to its second output direction.

```python
import numpy as np
import weightir as wir

weights = np.eye(3)
inputs = wir.Keys([[1.0], [0.0], [0.0]], "projection")
outputs = wir.Values([[0.0], [1.0], [0.0]], "projection")

program = wir.edit("projection", map=wir.Exact(inputs, outputs))
patch = wir.compile(program, weights)
updated_weights = weights + patch.delta
print(updated_weights @ inputs.data)
```

## Setup

On the GPU server, follow the [server setup](CLAUDE.md#server-setup) for local storage and processor selection.

```sh
uv sync
uv run pytest -q
```

Model checkpoints, datasets, and run outputs are stored locally and are not committed. LLM experiments need an Anthropic API key in the local environment file:

```text
# .env (not committed)
ANTHROPIC_API_KEY=your-key
```

## Read more

- [Design](docs/architecture.md): the language, its implementation, and what comes next.
- [Experiments](docs/experiments.md): recorded results and commands to reproduce them.
- [Reading guide](docs/related_works.md): useful research and implementations.
- [Reference code](third_party/README.md) and [license notices](THIRD_PARTY_NOTICES.md).
