# Reference implementations

Pinned Git submodules for studying algorithms and adapting useful pieces. See the [reading map](../docs/related_works.md), [full bibliography](papers/README.md), and [architecture](../docs/architecture.md) for context.

## Use

```sh
git submodule update --init
git submodule update --init third_party/rome
git submodule status
```

The first command fetches all references; the second fetches one. The status command shows each pinned revision and whether it is available locally.

Project code must run without this directory. Copy and adapt needed code instead of importing these submodules, and credit the source in the adapted file and [third-party notices](../THIRD_PARTY_NOTICES.md). Direct execution is reserved for one-off checks against the original implementation.

License labels below are retained reference notes. Check the license at the pinned revision before copying. Entries with no recorded license are for reading only.

## Editing and evaluation

| Repository / directory | Use | Recorded license |
|---|---|---|
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | Editing methods and evaluation datasets | MIT |
| [rome](https://github.com/kmeng01/rome) | Single-association editing | MIT |
| [memit](https://github.com/kmeng01/memit) | Batch editing across layers | MIT |
| [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) | Editing with null-space projection | MIT |
| [unified-model-editing](https://github.com/scalable-model-editing/unified-model-editing) | Batch editing with equality constraints | None recorded |
| [RippleEdits](https://github.com/edenbiran/RippleEdits) | Effects on related facts | MIT |
| [MQuAKE](https://github.com/princeton-nlp/MQuAKE) | Multi-hop questions after editing | MIT |

## Programs and circuits

| Repository / directory | Use | Recorded license |
|---|---|---|
| [tracr](https://github.com/google-deepmind/tracr) | Compile programs into controlled Transformer models | Apache-2.0 |
| [RASP](https://github.com/tech-srl/RASP) | Interpreter for Transformer programs | MIT |
| [alta](https://github.com/google-deepmind/alta) | Compiler-based analysis and trace supervision | Apache-2.0 |
| [InterpBench](https://github.com/FlyingPumba/InterpBench) | Models with known circuits | MIT |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Inspect model internals | MIT |
| [nnsight](https://github.com/ndif-team/nnsight) | Trace and intervene on computations | MIT |
| [pyvene](https://github.com/stanfordnlp/pyvene) | Configure representation interventions | Apache-2.0 |

## Weight transformations and repair

| Repository / directory | Use | Recorded license |
|---|---|---|
| [mergekit](https://github.com/arcee-ai/mergekit) | Model merging recipes | LGPL-3.0 |
| [brainsurgery](https://github.com/schneiderkamplab/brainsurgery) | Declarative tensor transformations | MIT |
| [evolutionary-model-merge](https://github.com/SakanaAI/evolutionary-model-merge) | Search over merging recipes | Apache-2.0 |
| [task_vectors](https://github.com/mlfoundations/task_vectors) | Task arithmetic | None recorded |
| [dataless-model-merging](https://github.com/bloomberg/dataless-model-merging) | Least-squares model merging | MIT |
| [composable-interventions](https://github.com/hartvigsen-group/composable-interventions) | Evaluate combinations of model changes | None recorded |
| [PRDNN](https://github.com/95616ARG/PRDNN) | Neural network repair | MIT |
| [APRNN](https://github.com/95616ARG/APRNN) | Repair while retaining network architecture | Check upstream |
| [vehicle](https://github.com/vehicle-lang/vehicle) | Specification language and verification | Check upstream |
| [concept-erasure](https://github.com/EleutherAI/concept-erasure) | Closed-form linear concept erasure | MIT |
| [refusal_direction](https://github.com/andyrdt/refusal_direction) | Convert direction removal into weight changes | Apache-2.0 |

## Program search and learning

| Repository / directory | Use | Recorded license |
|---|---|---|
| [learned_optimization](https://github.com/google/learned_optimization) | Learned optimizers | Apache-2.0 |
| [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | Evolutionary program search with LLMs | Apache-2.0 |
| [openevolve](https://github.com/algorithmicsuperintelligence/openevolve) | Open implementation of evolutionary program search | Apache-2.0 |
| [SEAL](https://github.com/Continual-Intelligence/SEAL) | Generate self-adaptation data and settings | MIT |
| [text-to-lora](https://github.com/SakanaAI/text-to-lora) | Generate adapters from descriptions | Apache-2.0 |
| [modula](https://github.com/modula-systems/modula) | Layer-specific optimization geometry | MIT |
| [Muon](https://github.com/KellerJordan/Muon) | Orthogonalized gradient updates | MIT |
| [GPM](https://github.com/sahagobinda/GPM) | Project updates to retain previous tasks | MIT |
| [Adam-NSCL](https://github.com/ShipengWang/Adam-NSCL) | Project adaptive updates into null spaces | MIT |
| [InfLoRA](https://github.com/liangyanshuo/InfLoRA) | Continual adaptation with a fixed basis | MIT |
