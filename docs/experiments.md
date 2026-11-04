# Experiments

**What computations can this language implement or change inside pretrained models?**

Start with a concrete change to a trained network, express it as a program, and measure the result. Preservation is a task-specific choice. See the [design](architecture.md).

## Next experiments

Build one useful operation on a pretrained model, then make it reusable:

1. Choose a computation and define its intended effect on the model's output.
2. Locate candidate components through analysis or interventions, then change their weights.
3. Evaluate model outputs on held-out inputs under matching conditions.
4. Make the operation reusable and test combinations with other operations.

Measure the requested change, generalization, failures, and computational cost. Check preservation when requested. Keep evaluation inputs separate from development inputs, and distinguish successful activation interventions from successful weight changes.

## Recorded results

Results recorded on **October 23, 2026** cover the matrix-editing and program-execution components.

### Fact editing

ROME edited layer 8 of GPT-2 medium on the first 50 CounterFact records, using Mac MPS. Input statistics came from 10,000 Wikitext-103 articles. Scores are success percentages based on target-token probabilities; the overall score is their harmonic mean.

| Model | Overall | Requested facts | Paraphrases | Neighboring facts |
|---|---:|---:|---:|---:|
| GPT-2 medium, original | 27.2 | 18.0 | 24.0 | 75.6 |
| GPT-2 medium, ROME | 86.4 | 100.0 | 96.0 | 70.0 |

These measurements use teacher forcing. Statistics took 28 minutes; editing took about five seconds per record. All four presets have numerical tests, but GPT-2 XL and real-model runs of MEMIT, EMMET, and AlphaEdit remain unperformed. The [editing experiment](../experiments/editing/presets.py) supports CounterFact and RippleEdits.

### Learning through weight-edit programs

These experiments train small networks from scratch through repeated weight edits. They test execution tools, not mechanistic changes to pretrained models.

The LLM and tuned Adam each received 20 trials, restarting from seed 100. Selection used validation accuracy or how early CartPole was solved. The LLM never received test scores. Evaluation used separate seeds. Data augmentation was disabled.

**Classification.** MNIST and Fashion-MNIST used two 256-unit ReLU hidden layers, batches of 128, and three CPU epochs. A fixed split reserved 5,000 images for validation and 55,000 for training. Test accuracy is mean ± standard deviation over seeds 0–2; times are approximate.

| Method | MNIST (%) | Fashion-MNIST (%) | Time |
|---|---:|---:|---:|
| Adam | 97.23 ± 0.19 | 87.30 ± 0.54 | 1 s |
| Tuned Adam | 97.29 ± 0.19 | 86.91 ± 0.50 | 1 s |
| SGD | 97.43 ± 0.17 | 86.83 ± 0.78 | 1 s |
| Muon | 96.27 ± 0.23 | 87.52 ± 0.32 | 140 s |
| WeightIR Adam | 97.37 ± 0.27 | 87.11 ± 0.52 | 1 s |
| WeightIR orthogonal updates | 96.76 ± 0.20 | 88.25 ± 0.25 | 5 s |
| WeightIR covariance updates | 97.79 ± 0.19 | 88.64 ± 0.09 | 6 s |
| Fixed edit program | 97.40 ± 0.10 | 88.30 ± 0.31 | 44 s |
| Haiku 5.5 edit program | 97.76 ± 0.14 | 89.33 ± 0.07 | 44 s |

**CartPole.** REINFORCE used a learned value baseline, separate networks with two 64-unit tanh hidden layers, and 600 episodes. Solved means a trailing 100-episode mean return reaching 475. The table lists the first such episode for seeds 0–4; a dash means unsolved.

| Method | Solved | Episodes to threshold |
|---|---:|---|
| Adam | 5/5 | 416, 314, 439, 312, 399 |
| Tuned Adam | 5/5 | 337, 247, 403, 293, 467 |
| SGD | 2/5 | —, 446, 539, —, — |
| Muon | 5/5 | 263, 302, 225, 180, 220 |
| WeightIR Adam | 5/5 | 405, 390, 380, 360, 457 |
| WeightIR orthogonal updates | 5/5 | 261, 199, 191, 275, 440 |
| WeightIR covariance updates | 0/5 | —, —, —, —, — |
| Fixed edit program | 5/5 | 213, 183, 241, 195, 177 |
| Haiku 5.5 edit program | 5/5 | 177, 185, 143, 156, 249 |

One covariance-update run diverged at episode 41. Reaching the threshold did not always mean performance stayed there.

The LLM mostly changed step sizes and schedules; Adam tuned only a constant learning rate. Each task had one search and few evaluation seeds. Edit programs were slower than Adam. An earlier augmentation-enabled run reached 99.22% on MNIST under different rules and is excluded.

## Run the existing experiments

Follow the [setup](../README.md) and GPU [server instructions](../CLAUDE.md#server-setup). LLM search needs an Anthropic API key in the local environment file.

```sh
# Fact editing
uv run python experiments/editing/presets.py --model gpt2-medium --method rome --dataset counterfact --size 50

# Classification and reinforcement learning
uv run python experiments/classification/train.py --dataset mnist --optimizer wir-cov --epochs 3 --seed 0
uv run python experiments/rl/cartpole.py --optimizer wir-orth --episodes 600 --seed 0

# Search over edit programs, or Adam's learning rate
uv run --env-file .env python experiments/search.py --task mnist --method llm --trials 20
uv run python experiments/search.py --task mnist --method adam-lr --trials 20
```

See the [optimizer definitions](../experiments/optimizers.py) and [search runner](../experiments/search.py). Results and generated programs go in the local data directory, outside Git.
