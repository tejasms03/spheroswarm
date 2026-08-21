"""Shared-parameter actor-critic.

One network, run once per robot. Because it only sees local observations, a
policy trained on 4 robots runs unchanged on 8.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .sim import OBS_DIM

RUNS = Path(__file__).resolve().parent.parent / "runs"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, 0.01)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.log_std.expand_as(self.mu(h)), self.v(h).squeeze(-1)

    def dist(self, x):
        mu, log_std, v = self(x)
        return torch.distributions.Normal(mu, log_std.exp()), v


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a, m_b = self.var * self.count, bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


class PolicyController:
    """Wraps a checkpoint so it has the same act() signature as Boids."""

    def __init__(self, path):
        self.path = Path(path)
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.net = ActorCritic()
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.norm = RunningNorm(OBS_DIM)
        self.norm.mean = np.array(ck["obs_mean"])
        self.norm.var = np.array(ck["obs_var"])
        self.meta = ck.get("meta", {})
        self.label = self.path.stem

    @torch.no_grad()
    def act(self, env, obs=None):
        if obs is None:
            obs = env.observe()
        x = torch.as_tensor(self.norm(obs), dtype=torch.float32)
        mu, _, _ = self.net(x)
        return np.tanh(mu.numpy())


def save_checkpoint(net, norm, meta, name=None):
    RUNS.mkdir(exist_ok=True)
    name = name or f"{meta.get('task','run')}_{time.strftime('%m%d_%H%M%S')}"
    path = RUNS / f"{name}.pt"
    torch.save({"model": net.state_dict(),
                "obs_mean": norm.mean, "obs_var": norm.var,
                "meta": meta}, path)
    (RUNS / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return path


def list_checkpoints():
    RUNS.mkdir(exist_ok=True)
    out = []
    for p in sorted(RUNS.glob("*.pt"), key=lambda q: q.stat().st_mtime, reverse=True):
        meta = {}
        j = p.with_suffix(".json")
        if j.exists():
            try:
                meta = json.loads(j.read_text())
            except Exception:
                pass
        out.append({"path": p, "name": p.stem, "meta": meta})
    return out
