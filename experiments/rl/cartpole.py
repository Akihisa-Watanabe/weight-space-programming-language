"""REINFORCE with a learned value baseline on CartPole-v1, trained with an optimizer from experiments/optimizers.py.

    python experiments/rl/cartpole.py --optimizer wir-cov --episodes 600 --seed 0

One update per episode on  -mean_t log π(a_t|s_t)·A_t + mean_t (V(s_t) - G_t)²,  where G_t is the discounted
return-to-go times 1 - γ (so it lies in [0, 1)) and A_t = G_t - V(s_t) is standardized over the episode. The episode's
return is passed to the optimizer's observe() when it has one (a ProgramOptimizer's program can read it). train() is
also what experiments/search.py calls for each trial. The arguments, every episode's return, the first episode at which
the mean return of the last 100 episodes reaches CartPole-v1's threshold of 475, the episode after which the weights
became non-finite and training stopped (each null if never), and a ProgramOptimizer's edit statistics and printed
output go to data/runs/rl/cartpole/<optimizer>-seed<seed>.json.
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical

from experiments.optimizers import OPTIMIZERS

DATA = Path(__file__).resolve().parents[2] / "data"
GAMMA = 0.99
EPISODES = 600
SPEC = gym.spec("CartPole-v1")  # reward_threshold 475, max_episode_steps 500


def mlp(out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))


def model() -> nn.ModuleDict:
    return nn.ModuleDict({"policy": mlp(2), "value": mlp(1)})


def task() -> str:
    """The task as the LLM of experiments/search.py reads it, for train()'s default settings."""
    return ("Train a REINFORCE policy (sites policy.*, 2 actions, tanh MLP) and a value baseline (sites value.*) on "
            "CartPole-v1. One step per episode on -mean(log pi(a|s)·A) + mean((V(s) - G)²) over the episode's states, "
            f"where G is the discounted return-to-go (gamma {GAMMA}) times {1 - GAMMA:g} and A = G - V(s) is "
            "standardized over the episode, so this loss does not track how well the policy does. The training loop "
            f"reports episode_return, the return (at most {SPEC.max_episode_steps}) of the episode each step learned "
            f"from. Training runs {EPISODES} episodes.")


def train(make_optimizer, *, episodes: int = EPISODES, seed: int = 0) -> dict:
    """Train model() with make_optimizer(model); return the run's record (see the module docstring)."""
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)
    net = model()
    optimizer = make_optimizer(net)

    start, returns, solved_at, diverged_at = time.time(), [], None, None
    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed if episode == 1 else None)
        states, actions, rewards, done = [], [], [], False
        while not done:
            states.append(torch.as_tensor(obs))
            with torch.no_grad():
                actions.append(Categorical(logits=net["policy"](states[-1])).sample().item())
            obs, reward, terminated, truncated, _ = env.step(actions[-1])
            rewards.append(reward)
            done = terminated or truncated

        G, g = torch.empty(len(rewards)), 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + GAMMA * g
            G[t] = (1 - GAMMA) * g
        S, taken = torch.stack(states), torch.tensor(actions)

        def closure():
            optimizer.zero_grad()
            logp = Categorical(logits=net["policy"](S)).log_prob(taken)
            V = net["value"](S).squeeze(-1)
            A = G - V.detach()
            A = (A - A.mean()) / (A.std() + 1e-8)
            loss = -(logp * A).mean() + ((V - G) ** 2).mean()
            loss.backward()
            return loss

        loss = optimizer.step(closure).item()
        returns.append(sum(rewards))
        if hasattr(optimizer, "observe"):  # a program can read these through ctx
            optimizer.observe(loss, episode_return=returns[-1])

        mean100 = sum(returns[-100:]) / len(returns[-100:])
        if solved_at is None and episode >= 100 and mean100 >= SPEC.reward_threshold:
            solved_at = episode
        if episode % 50 == 0:
            print(f"cartpole episode {episode}: mean return of the last 100 {mean100:.1f}", flush=True)
        if not all(p.isfinite().all() for p in net.parameters()):  # the policy cannot act any more
            diverged_at = episode
            break

    print("cartpole: " + (f"solved at episode {solved_at}" if solved_at else
                          f"diverged (non-finite weights) at episode {diverged_at}" if diverged_at
                          else f"never solved in {episodes} episodes"), flush=True)
    return {"args": {"episodes": episodes, "seed": seed}, "returns": returns, "solved_at": solved_at,
            "diverged_at": diverged_at, "seconds": time.time() - start,
            "edits": getattr(optimizer, "edit_stats", None), "output": getattr(optimizer, "output", None)}


def save(result: dict, name: str, seed: int) -> Path:
    result["args"]["optimizer"] = name
    out = DATA / "runs" / "rl" / "cartpole" / f"{name}-seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = train(OPTIMIZERS[args.optimizer], episodes=args.episodes, seed=args.seed)
    save(result, args.optimizer, args.seed)


if __name__ == "__main__":
    main()
