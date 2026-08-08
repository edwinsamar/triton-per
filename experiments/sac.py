"""SAC / REDQ with pluggable prioritized replay backend.

SAC:  --num-q 2 --q-subset 2 --utd 1
REDQ: --num-q 10 --q-subset 2 --utd 20

Backends: triton | triton-unfused | torch | cpu-sumtree | torchrl-cpu | torchrl-cuda
(same identifiers as bench/bench_sampling.py). Every run writes a
self-describing artifact dir (see experiments/common.py).

Example:
  python experiments/sac.py --env-id HalfCheetah-v5 --backend triton \
      --utd 20 --num-q 10 --seed 1 --total-steps 100000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
from bench_sampling import make_buffer  # noqa: E402
from common import RunLogger, seed_everything  # noqa: E402

LOG_STD_MIN, LOG_STD_MAX = -5, 2


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.register_buffer("scale", (act_high - act_low) / 2)
        self.register_buffer("bias", (act_high + act_low) / 2)

    def forward(self, obs):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.scale + self.bias
        logp = normal.log_prob(x) - torch.log(self.scale * (1 - y.pow(2)) + 1e-6)
        return action, logp.sum(-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="HalfCheetah-v5")
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--utd", type=int, default=1)
    ap.add_argument("--num-q", type=int, default=2)
    ap.add_argument("--q-subset", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--learning-starts", type=int, default=5_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=1_000_000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha-per", type=float, default=0.6)
    ap.add_argument("--beta-start", type=float, default=0.4)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)
    ap.add_argument("--debug-nan", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=2_000)
    ap.add_argument("--out-root", default="results/runs")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    name = args.run_name or (
        f"{args.env_id}-{args.backend}-utd{args.utd}-q{args.num_q}-seed{args.seed}"
    )
    logger = RunLogger(Path(args.out_root) / name, vars(args))
    seed_everything(args.seed)
    dev = torch.device(args.device)

    env = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id))
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)

    fields = {"obs": obs_dim, "action": act_dim, "reward": 1, "next_obs": obs_dim, "done": 1}
    buf = make_buffer(args.backend, args.buffer_size, fields, str(dev))

    actor = Actor(obs_dim, act_dim, act_low, act_high).to(dev)
    qs = nn.ModuleList([mlp(obs_dim + act_dim, 1) for _ in range(args.num_q)]).to(dev)
    qs_t = nn.ModuleList([mlp(obs_dim + act_dim, 1) for _ in range(args.num_q)]).to(dev)
    qs_t.load_state_dict(qs.state_dict())
    for p in qs_t.parameters():
        p.requires_grad_(False)

    q_opt = torch.optim.Adam(qs.parameters(), lr=args.lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)

    obs, _ = env.reset(seed=args.seed)
    returns, grad_steps = [], 0
    last_log_t, last_log_step = time.perf_counter(), 0

    for step in range(1, args.total_steps + 1):
        if step <= args.learning_starts:
            action_np = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=dev)[None])
            action_np = a.squeeze(0).cpu().numpy()

        next_obs, reward, term, trunc, info = env.step(action_np)
        if "episode" in info:
            returns.append(float(info["episode"]["r"]))
        real_next = next_obs if not trunc else info.get("final_observation", next_obs)
        buf.add({
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=dev)[None],
            "action": torch.as_tensor(action_np, dtype=torch.float32, device=dev)[None],
            "reward": torch.tensor([[reward]], dtype=torch.float32, device=dev),
            "next_obs": torch.as_tensor(real_next, dtype=torch.float32, device=dev)[None],
            "done": torch.tensor([[float(term)]], dtype=torch.float32, device=dev),
        })
        obs = next_obs if not (term or trunc) else env.reset()[0]

        if step > args.learning_starts:
            beta = args.beta_start + (1.0 - args.beta_start) * step / args.total_steps
            for _ in range(args.utd):
                batch, idx, w = buf.sample(args.batch_size, beta=beta)
                b_obs, b_act = batch["obs"], batch["action"]
                b_rew, b_next, b_done = batch["reward"], batch["next_obs"], batch["done"]
                w = w.reshape(-1, 1)
                if args.debug_nan and not (
                    torch.isfinite(b_obs).all() and torch.isfinite(w).all()
                ):
                    raise RuntimeError(
                        f"non-finite batch at step {step}: "
                        f"obs finite={torch.isfinite(b_obs).all().item()} "
                        f"w finite={torch.isfinite(w).all().item()} "
                        f"w range=({w.min().item():.3e},{w.max().item():.3e}) "
                        f"idx range=({idx.min().item()},{idx.max().item()}) size={buf.size}"
                    )

                with torch.no_grad():
                    na, nlogp = actor.sample(b_next)
                    nin = torch.cat([b_next, na], -1)
                    sub = torch.randperm(args.num_q, device="cpu")[: args.q_subset]
                    tq = torch.stack([qs_t[i](nin) for i in sub.tolist()]).min(0).values
                    alpha = log_alpha.exp()
                    target = b_rew + args.gamma * (1 - b_done) * (tq - alpha * nlogp)

                qin = torch.cat([b_obs, b_act], -1)
                preds = [q(qin) for q in qs]
                td = torch.stack([p - target for p in preds])  # [num_q, B, 1]
                q_loss = (w * td.pow(2)).mean() * len(preds)
                if args.debug_nan and not torch.isfinite(q_loss):
                    raise RuntimeError(
                        f"non-finite q_loss at step {step}: "
                        f"target range=({target.min().item():.3e},{target.max().item():.3e}) "
                        f"alpha={log_alpha.exp().item():.3e} "
                        f"td max={td.abs().max().item():.3e}"
                    )
                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                nn.utils.clip_grad_norm_(qs.parameters(), args.max_grad_norm)
                q_opt.step()

                buf.update_priorities(idx, td.detach().abs().mean(0).squeeze(-1))
                grad_steps += 1

                if grad_steps % args.utd == 0:  # one actor update per env step
                    a, logp = actor.sample(b_obs)
                    ain = torch.cat([b_obs, a], -1)
                    qmin = torch.stack([q(ain) for q in qs]).mean(0)
                    a_loss = (log_alpha.exp().detach() * logp - qmin).mean()
                    a_opt.zero_grad(set_to_none=True)
                    a_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    a_opt.step()

                    alpha_loss = (-log_alpha.exp() * (logp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()

                    with torch.no_grad():
                        for p, pt in zip(qs.parameters(), qs_t.parameters()):
                            pt.mul_(1 - args.tau).add_(args.tau * p)

        if step % args.log_every == 0:
            now = time.perf_counter()
            sps = (step - last_log_step) / (now - last_log_t)
            last_log_t, last_log_step = now, step
            r = float(np.mean(returns[-10:])) if returns else float("nan")
            logger.log(env_step=step, grad_steps=grad_steps, return_mean10=round(r, 1),
                       sps=round(sps, 1))
            print(f"step {step} | return {r:.1f} | {sps:.1f} env-steps/s", flush=True)

    logger.finish(final_return_mean10=float(np.mean(returns[-10:])) if returns else None,
                  grad_steps=grad_steps, env_steps=args.total_steps)


if __name__ == "__main__":
    main()
