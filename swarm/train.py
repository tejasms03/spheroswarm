"""PPO with parameter sharing across agents.

Every agent contributes transitions to one shared policy, so the batch is
n_envs * n_agents * horizon. Runs on CPU in minutes for this problem size.
"""

import threading

import numpy as np
import torch

from .policy import ActorCritic, RunningNorm, save_checkpoint
from .sim import EP_STEPS, OBS_DIM, SwarmEnv

DEFAULTS = dict(
    task="coverage", n_agents=6, n_envs=8, horizon=128, updates=150,
    lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4,
    ent_coef=0.004, vf_coef=0.5, randomize=True,
)


def train(cfg=None, on_progress=None, stop_event=None, name=None):
    c = dict(DEFAULTS)
    c.update(cfg or {})
    stop_event = stop_event or threading.Event()

    envs = [SwarmEnv(c["n_agents"], c["task"], randomize=c["randomize"], seed=i)
            for i in range(c["n_envs"])]
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
    norm = RunningNorm(OBS_DIM)

    obs = np.stack([e.observe() for e in envs])       # (E, N, D)
    E, N, T = c["n_envs"], c["n_agents"], c["horizon"]
    B = E * N
    history = []
    ep_stats = {"coverage": 0.0, "collisions": 0, "return": 0.0}
    ret_acc = np.zeros((E, N))

    for update in range(c["updates"]):
        if stop_event.is_set():
            break

        o_buf = np.zeros((T, B, OBS_DIM), dtype=np.float32)
        a_buf = np.zeros((T, B, 2), dtype=np.float32)
        lp_buf = np.zeros((T, B), dtype=np.float32)
        r_buf = np.zeros((T, B), dtype=np.float32)
        v_buf = np.zeros((T, B), dtype=np.float32)
        d_buf = np.zeros((T, B), dtype=np.float32)

        for t in range(T):
            flat = obs.reshape(B, OBS_DIM)
            norm.update(flat)
            nobs = norm(flat).astype(np.float32)
            with torch.no_grad():
                dist, val = net.dist(torch.as_tensor(nobs))
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)

            a_np = np.tanh(act.numpy())
            o_buf[t], a_buf[t] = nobs, act.numpy()
            lp_buf[t], v_buf[t] = logp.numpy(), val.numpy()

            nxt, rew, done = [], [], []
            for i, e in enumerate(envs):
                ob, r, dn, info = e.step(a_np[i * N:(i + 1) * N])
                ret_acc[i] += r
                if dn:
                    ep_stats = {"coverage": info["coverage"],
                                "collisions": info["collisions"],
                                "return": float(ret_acc[i].mean())}
                    ret_acc[i] = 0.0
                    ob = e.reset()
                nxt.append(ob)
                rew.append(r)
                done.append(np.full(N, float(dn)))
            obs = np.stack(nxt)
            r_buf[t] = np.concatenate(rew)
            d_buf[t] = np.concatenate(done)

        with torch.no_grad():
            _, last_v = net.dist(torch.as_tensor(
                norm(obs.reshape(B, OBS_DIM)).astype(np.float32)))
            last_v = last_v.numpy()

        adv = np.zeros_like(r_buf)
        gae = np.zeros(B, dtype=np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else v_buf[t + 1]
            nonterm = 1.0 - d_buf[t]
            delta = r_buf[t] + c["gamma"] * nv * nonterm - v_buf[t]
            gae = delta + c["gamma"] * c["lam"] * nonterm * gae
            adv[t] = gae
        ret = adv + v_buf

        b_o = torch.as_tensor(o_buf.reshape(-1, OBS_DIM))
        b_a = torch.as_tensor(a_buf.reshape(-1, 2))
        b_lp = torch.as_tensor(lp_buf.reshape(-1))
        b_ad = torch.as_tensor(adv.reshape(-1))
        b_rt = torch.as_tensor(ret.reshape(-1))
        b_ad = (b_ad - b_ad.mean()) / (b_ad.std() + 1e-8)

        idx = np.arange(len(b_o))
        mb = len(idx) // c["minibatches"]
        for _ in range(c["epochs"]):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                j = idx[s:s + mb]
                dist, v = net.dist(b_o[j])
                lp = dist.log_prob(b_a[j]).sum(-1)
                ratio = (lp - b_lp[j]).exp()
                a1 = ratio * b_ad[j]
                a2 = torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * b_ad[j]
                loss = (-torch.min(a1, a2).mean()
                        + c["vf_coef"] * ((v - b_rt[j]) ** 2).mean()
                        - c["ent_coef"] * dist.entropy().sum(-1).mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        history.append(ep_stats["return"])
        if on_progress:
            on_progress({
                "update": update + 1, "total": c["updates"],
                "return": ep_stats["return"], "coverage": ep_stats["coverage"],
                "collisions": ep_stats["collisions"], "history": list(history),
            })

    meta = dict(c)
    meta.update(final_return=history[-1] if history else 0.0,
                updates_done=len(history), history=history[-60:])
    return save_checkpoint(net, norm, meta, name)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--task", default="coverage", choices=["coverage", "gather"])
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--name", default=None)
    a = p.parse_args()

    def show(s):
        print(f"update {s['update']:>4}/{s['total']}  "
              f"return {s['return']:8.2f}  coverage {s['coverage']:.2f}  "
              f"collisions {s['collisions']}", flush=True)

    path = train(dict(task=a.task, n_agents=a.agents, updates=a.updates),
                 on_progress=show, name=a.name)
    print("saved", path)
