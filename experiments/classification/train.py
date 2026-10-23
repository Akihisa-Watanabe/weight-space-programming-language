"""An MLP (784-256-256-10) on MNIST or Fashion-MNIST, trained with an optimizer from experiments/optimizers.py.

    python experiments/classification/train.py --dataset fashion --optimizer wir-orth --epochs 3 --seed 0

5,000 training images, the same for every run, are held out for validation; the other 55,000 are trained on. Every
optimizer sees the same initialization and batch order for a seed. Validation loss and accuracy are measured every
--eval-every steps and passed to the optimizer's observe() when it has one (a ProgramOptimizer's program can read
them); the test set is scored only at the end of each epoch, for the log. train() is also what experiments/search.py
calls for each trial. The arguments, the validation curve (step, seconds, mean train loss since the previous point,
val_loss, val_acc), per epoch the mean train loss and the validation and test accuracy, and a ProgramOptimizer's edit
statistics and printed output go to data/runs/classification/<dataset>/<optimizer>-seed<seed>.json.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from experiments.optimizers import OPTIMIZERS

DATA = Path(__file__).resolve().parents[2] / "data"
DATASETS = {"mnist": torchvision.datasets.MNIST, "fashion": torchvision.datasets.FashionMNIST}
VAL = 5000                  # held-out training images; both datasets have 60,000 training images
EPOCHS, BATCH_SIZE, EVAL_EVERY = 3, 128, 100


def load(dataset: str):
    """(x, y, x_val, y_val, x_test, y_test): flattened images standardized by the trained-on images' pixel mean and std.
    The validation images are a fixed random 5,000 of the training set."""
    train_set, test_set = (DATASETS[dataset](DATA, train=t, download=True) for t in (True, False))
    x, x_test = (d.data.flatten(1).float() / 255 for d in (train_set, test_set))
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    val, fit = perm[:VAL], perm[VAL:]
    mean, std = x[fit].mean(), x[fit].std()
    return ((x[fit] - mean) / std, train_set.targets[fit], (x[val] - mean) / std, train_set.targets[val],
            (x_test - mean) / std, test_set.targets)


def model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def task(dataset: str) -> str:
    """The task as the LLM of experiments/search.py reads it, for train()'s default settings."""
    steps = -(-(60000 - VAL) // BATCH_SIZE)
    return (f"Classify {dataset} images (784 standardized pixels) into 10 classes with this MLP (ReLU). The loss is "
            f"the mean cross-entropy over a batch of {BATCH_SIZE}; training runs {EPOCHS} epochs of {steps} steps. The "
            f"training loop reports val_loss and val_acc on {VAL} held-out training images every {EVAL_EVERY} steps.")


def train(dataset: str, make_optimizer, *, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
          eval_every: int = EVAL_EVERY, seed: int = 0, device: str = "cpu", data=None) -> dict:
    """Train model() with make_optimizer(model); return the run's record (see the module docstring)."""
    x, y, x_val, y_val, x_test, y_test = data if data is not None else load(dataset)
    dev = torch.device(device)
    torch.manual_seed(seed)
    net = model().to(dev)
    optimizer = make_optimizer(net)
    order = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def score(xs, ys):
        out = net(xs.to(dev))
        return F.cross_entropy(out, ys.to(dev)).item(), (out.argmax(1).cpu() == ys).float().mean().item()

    log, curve, step, start, recent = [], [], 0, time.time(), []
    for epoch in range(1, epochs + 1):
        epoch_start, total = time.time(), 0.0
        for idx in torch.randperm(len(x), generator=order).split(batch_size):
            xb, yb = x[idx].to(dev), y[idx].to(dev)

            def closure():
                optimizer.zero_grad()
                loss = F.cross_entropy(net(xb), yb)
                loss.backward()
                return loss

            loss = optimizer.step(closure).item()
            step += 1
            total += loss * len(idx)
            recent.append(loss)
            metrics = {}
            if step % eval_every == 0:
                val_loss, val_acc = score(x_val, y_val)
                metrics = {"val_loss": val_loss, "val_acc": val_acc}
                curve.append({"step": step, "seconds": round(time.time() - start, 2),
                              "train_loss": sum(recent) / len(recent), **metrics})
                recent = []
            if hasattr(optimizer, "observe"):  # a program can read these through ctx
                optimizer.observe(loss, **metrics)
        _, val_acc = score(x_val, y_val)
        _, test_acc = score(x_test, y_test)
        log.append({"epoch": epoch, "train_loss": total / len(x), "val_acc": val_acc, "test_acc": test_acc,
                    "seconds": time.time() - epoch_start})
        print(f"{dataset} epoch {epoch}: train loss {total / len(x):.4f}, val acc {val_acc:.4f}, test acc "
              f"{test_acc:.4f}, {log[-1]['seconds']:.1f}s", flush=True)
    return {"args": {"dataset": dataset, "epochs": epochs, "batch_size": batch_size, "eval_every": eval_every,
                     "seed": seed, "device": device},
            "epochs": log, "curve": curve, "steps": step,
            "edits": getattr(optimizer, "edit_stats", None), "output": getattr(optimizer, "output", None)}


def save(result: dict, dataset: str, name: str, seed: int) -> Path:
    result["args"]["optimizer"] = name
    out = DATA / "runs" / "classification" / dataset / f"{name}-seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="steps between validation measurements")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train(args.dataset, OPTIMIZERS[args.optimizer], epochs=args.epochs, batch_size=args.batch_size,
                   eval_every=args.eval_every, seed=args.seed, device=args.device)
    save(result, args.dataset, args.optimizer, args.seed)


if __name__ == "__main__":
    main()
