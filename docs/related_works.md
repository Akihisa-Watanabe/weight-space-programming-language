# Reading map

This project aims to make existing trained networks programmable: people should be able to inspect their computations, implement new ones, and change how they work. Mechanistic interpretability supplies ways to identify useful internal structure; compilation turns operations on that structure into changes to the model.

The references below are entry points. The [reading archive](../third_party/papers/README.md) retains the full bibliography and local PDF links; the [implementation catalog](../third_party/README.md) lists reference code.

## Weight-space programming

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) explores LLMs reading, constructing, and repairing neural network weights. Its public logs show the actual interfaces: the car-repair and small-game experiments use a compact language for assigning and scaling tensor entries and slices, while the Go and language-model experiments use Python to manipulate weights.

Start with the [small-game prompt](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp5_write_from_scratch/runs/breakout/message.txt), an [actual repair program](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp4_fix_car/runs/oneshot_repair/reply/answer.md), and the [Go experiment setup](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch). These are concrete examples of reasoning about a computation and expressing it through weights.

## Finding computations in trained models

[Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) describes how attention heads read, route, and write information. [Sparse feature dictionaries](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) and [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) provide ways to investigate features and their interactions. They inform the analysis tools and reusable operations that a weight-space language could expose.

[Causal Abstraction](https://arxiv.org/abs/2301.04709) and [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) connect proposed internal variables to interventions. [Subspace illusions](https://arxiv.org/abs/2311.17030) show why that connection needs testing: an intervention can change an answer without establishing the proposed explanation. A useful library should let users inspect a candidate mechanism, intervene on it, and test its effects on new inputs.

## Changing learned computations

[Rewriting a Deep Generative Model](https://arxiv.org/abs/2007.15646) and [Unified Concept Editing](https://arxiv.org/abs/2308.14761) study weight changes in trained image generators. [ROME](https://arxiv.org/abs/2202.05262), [MEMIT](https://arxiv.org/abs/2210.07229), and [AlphaEdit](https://arxiv.org/abs/2410.02355) provide methods for writing new associations into language models. These offer practical building blocks for a compiler and examples of the model-specific work required.

[Representation Engineering](https://arxiv.org/abs/2310.01405) and [ReFT](https://arxiv.org/abs/2404.03592) explore interventions on internal representations. Moving such an intervention into permanent weights requires checking where it acts and how its effect depends on the input. [BrainSurgery](https://arxiv.org/abs/2606.09707) and [MergeKit](https://arxiv.org/abs/2403.13257) are useful references for expressing and executing weight transformations.

## Compilation and evaluation

[RASP](https://arxiv.org/abs/2106.06981), [Tracr](https://arxiv.org/abs/2301.05062), and [ALTA](https://arxiv.org/abs/2410.18077) connect explicit programs to neural implementations. Their known computations make focused compiler tests possible. [InterpBench](https://arxiv.org/abs/2407.14494) supplies models with known circuits for evaluating interpretation methods.

[RippleEdits](https://arxiv.org/abs/2307.12976) and [MQuAKE](https://arxiv.org/abs/2305.14795) test whether edited knowledge is used in related questions. [Composable Interventions](https://arxiv.org/abs/2407.06483) examines how model changes interact. Together they motivate testing the requested computation, new combinations, and successive edits. Preservation is an optional part of a request; [model repair](https://arxiv.org/abs/2104.04413) and [concept erasure](https://arxiv.org/abs/2306.03819) offer tools when explicit constraints are useful.
