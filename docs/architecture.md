# Architecture

**What computations can this language implement or change inside pretrained models?**

Weight-Space Programming Language explores programs that read, write, construct, and modify
computations inside trained neural networks.
Preservation and limits on changes are optional parts of a task. Learning is available
when a direct construction is insufficient.

The current implementation, WeightIR, provides a compiler for changes to individual
weight matrices and tools for evaluating and improving programs. It does not yet provide
a general language for circuits.

## Intended flow

A program identifies a computation, describes the intended change, chooses a way to
implement it in the model, and evaluates the result.

Mechanistic interpretability supplies methods for finding features and circuits,
testing their causal roles, and relating them to model behavior. These methods should
become a reusable analysis and compilation library. An analysis result is a candidate
explanation; intervention experiments must establish whether it supports the requested
operation.

The compilation library should connect supported operations to concrete changes in
weights. Depending on the computation, this could use a direct construction, a
transformation of existing weights, the current matrix solver, or iterative learning.
The matrix solver is one backend, not a required form for every operation.

The resulting weights must be evaluated in the model. A successful temporary activation
intervention does not establish that a permanent weight change reproduces it on new inputs.

## Current implementation

| Part | What exists |
|---|---|
| [Specification](../src/weightir/spec.py) | A Python-embedded language for input-to-output mappings at one matrix, with optional preservation, penalties, and budgets. Types check layer identity and dimensions. |
| [Numerical compiler](../src/weightir/compile.py) | A quadratic solver, independent residual checks, and operations for combining or sequencing changes. |
| [Model access](../src/weightir/model.py) | Reading, applying, and resetting changes to linear layers in Hugging Face language models. Shared weights are rejected. |
| [Observation](../src/weightir/stats.py) | Activation collection and input statistics, with hooks for temporary interventions. |
| [Target search](../src/weightir/targets.py) | Optimizing activation changes for requested text continuations. |
| [Editing methods](../src/weightir/presets.py) | ROME, EMMET, MEMIT, and AlphaEdit expressed through the matrix backend. |
| [Program execution](../src/agentic/program.py) | Running programs that inspect activations and gradients, change weights, re-evaluate, and undo changes during training. |
| [Program search](../src/agentic/loop.py) | An LLM revises programs between fresh training trials using validation results. |

A matrix can be synthesized from zero using the same solver. This does not construct
a complete network: model structure is currently supplied by the caller.

## Numerical scope

The solver minimizes a quadratic measure of weight change while satisfying requested
linear mappings and any optional constraints. Strict requirements and soft penalties
are distinct. Conflicts, insufficient budgets, unsupported combinations, and uncertain
numerical decisions are reported separately.

The checker remeasures mappings, preservation, rank, and norm before returning a patch.
Preserving a supplied input subspace means preserving that matrix's response there;
it does not establish preservation of a model capability. Estimated subspaces retain
their source data so changes outside the retained basis can also be reported.

After rounding weights to the storage precision, errors in strict mappings are checked against
a precision-dependent tolerance. Preservation changes after rounding are reported but
do not determine acceptance. Rank and norm budgets are checked before rounding.

Joint changes currently share one matrix and one change metric. Sequential changes can
carry forward earlier mappings and preservation conditions at the same matrix; this
behavior can be disabled. Across layers, earlier changes can alter later inputs, so
sequential compilation should refresh activations and evaluate behavior. There is no
general guarantee that independently useful changes remain useful together.

## Boundaries and next work

The numerical core depends only on NumPy and SciPy. Model integration, interpretation,
agent orchestration, and evaluation remain separate. The core does not import the
agent or evaluation layers. Reference repositories supply material to study or adapt,
not runtime imports; adapted code retains its license notices.

The next work is to implement concrete computations in pretrained models and use those
examples to develop the language. Needed pieces include reusable interpretation
operations, compilation methods beyond matrix fitting, and an
agent that builds and tests complete programs from a request.

Evaluate whether the requested computation works on held-out inputs, how operations
compose, and the cost of producing the result. Measure preservation when requested.
Current evidence and limitations belong in [Experiments](experiments.md); research
sources are collected in [Related work](related_works.md).
