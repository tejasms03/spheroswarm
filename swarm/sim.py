"""Sphero swarm simulator.

Deliberately models the three things that break sim-to-real transfer on this
hardware: command latency, motor lag, and per-robot speed variation.

Bounds come from a `Workspace` (see `workspace/space.py`), so the sim, the
planner and the camera all agree on where the floor is. Passing no workspace
gives the historical 200cm square, which keeps old checkpoints meaningful.
"""

from collections import deque

import numpy as np

from workspace.space import Workspace

ARENA = 200.0          # cm, the default square when no workspace is supplied
MAX_SPEED = 60.0       # cm/s at full command
COLLIDE_DIST = 12.0    # cm, robot diameter-ish
OBS_RADIUS = 70.0      # cm, how far a robot "sees" neighbours
K_NEIGHBOURS = 4
GRID = 10              # coverage grid resolution
EP_STEPS = 400

OBS_DIM = 2 + K_NEIGHBOURS * 4 + 4 + 3


def default_workspace():
    return Workspace(bounds_cm=[[0, 0], [ARENA, 0], [ARENA, ARENA], [0, ARENA]])


class SwarmEnv:
    """Vectorised over agents; one env instance holds one swarm."""

    def __init__(self, n_agents=6, task="coverage", dt=0.1,
                 randomize=True, seed=None, workspace=None):
        self.n = n_agents
        self.task = task
        self.dt = dt
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.set_workspace(workspace or default_workspace())
        self.reset()

    # -- workspace ---------------------------------------------------------

    def set_workspace(self, ws):
        """Swap the arena. Positions are pulled back inside the new bounds."""
        self.ws = ws
        self.xmin, self.xmax, self.ymin, self.ymax = ws.bbox
        self.width = max(self.xmax - self.xmin, 1e-6)
        self.height = max(self.ymax - self.ymin, 1e-6)
        self.span = max(self.width, self.height)
        if getattr(self, "pos", None) is not None:
            self.pos = np.array([self.ws.nearest_valid_point(p) for p in self.pos])

    @property
    def bbox(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def _sample_valid(self, count, margin=30.0):
        """Uniform over the workspace, kept `margin` off the bounding box edge."""
        mx = min(margin, self.width / 3.0)
        my = min(margin, self.height / 3.0)
        out = []
        for _ in range(count):
            for _ in range(200):
                p = np.array([self.rng.uniform(self.xmin + mx, self.xmax - mx),
                              self.rng.uniform(self.ymin + my, self.ymax - my)])
                if self.ws.is_valid_point(p):
                    out.append(p)
                    break
            else:
                out.append(self.ws.random_valid_point(self.rng))
        return np.array(out, dtype=float)

    # -- setup -------------------------------------------------------------

    def reset(self):
        r = self.rng
        self.pos = self._sample_valid(self.n)
        self.vel = np.zeros((self.n, 2))
        self.t = 0

        if self.randomize:
            self.latency = int(r.integers(1, 4))        # control steps of delay
            self.tau = float(r.uniform(0.25, 0.5))      # motor response lag, s
            self.gain = r.uniform(0.8, 1.2, self.n)     # per-robot speed scale
            self.bias = r.uniform(-0.12, 0.12, self.n)  # heading offset, rad
        else:
            self.latency, self.tau = 2, 0.35
            self.gain = np.ones(self.n)
            self.bias = np.zeros(self.n)

        self.queue = deque(
            [np.zeros((self.n, 2))] * self.latency, maxlen=self.latency
        )
        self.visited = np.zeros((GRID, GRID), dtype=bool)
        self.goal = self._sample_valid(1, margin=40.0)[0]
        self._mark_visited()
        return self.observe()

    # -- dynamics ----------------------------------------------------------

    def step(self, actions):
        """actions: (n, 2) in [-1, 1], interpreted as a desired velocity."""
        a = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
        mag = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.where(mag > 1.0, a / np.maximum(mag, 1e-9), a)

        cos, sin = np.cos(self.bias), np.sin(self.bias)
        rot = np.stack([a[:, 0] * cos - a[:, 1] * sin,
                        a[:, 0] * sin + a[:, 1] * cos], axis=1)

        self.queue.append(rot * MAX_SPEED)
        cmd = self.queue[0] * self.gain[:, None]

        self.vel += (cmd - self.vel) * (self.dt / self.tau)
        self.pos += self.vel * self.dt

        for ax, (lo, hi) in enumerate([(self.xmin, self.xmax), (self.ymin, self.ymax)]):
            out = (self.pos[:, ax] < lo) | (self.pos[:, ax] > hi)
            self.vel[out, ax] = 0.0
            np.clip(self.pos[:, ax], lo, hi, out=self.pos[:, ax])

        if self.ws.obstacles or len(self.ws.bounds_cm) != 4:
            for i in range(self.n):
                if not self.ws.is_valid_point(self.pos[i]):
                    self.pos[i] = self.ws.nearest_valid_point(self.pos[i])
                    self.vel[i] = 0.0

        rew = self._reward()
        self.t += 1
        return self.observe(), rew, self.t >= EP_STEPS, self.info()

    # -- task --------------------------------------------------------------

    def _cells(self):
        u = (self.pos[:, 0] - self.xmin) / self.width
        v = (self.pos[:, 1] - self.ymin) / self.height
        c = np.stack([u, v], axis=1) * GRID
        return np.clip(c.astype(int), 0, GRID - 1)

    def _cell_centre(self, i, j):
        return np.array([self.xmin + (i + 0.5) * self.width / GRID,
                         self.ymin + (j + 0.5) * self.height / GRID])

    def _mark_visited(self):
        c = self._cells()
        self.visited[c[:, 0], c[:, 1]] = True

    def _reward(self):
        rew = np.zeros(self.n)

        if self.task == "coverage":
            c = self._cells()
            for i in range(self.n):
                if not self.visited[c[i, 0], c[i, 1]]:
                    self.visited[c[i, 0], c[i, 1]] = True
                    rew[i] += 1.0
            rew -= 0.01
        else:  # gather
            d = np.linalg.norm(self.pos - self.goal, axis=1)
            rew += 0.02 * (100.0 - d) / 100.0
            rew[d < 25.0] += 0.05

        d = self._pairwise()
        np.fill_diagonal(d, np.inf)
        rew -= 0.5 * (d < COLLIDE_DIST).sum(axis=1)
        return rew

    def _pairwise(self):
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        return np.linalg.norm(diff, axis=2)

    # -- observation -------------------------------------------------------

    def observe(self):
        d = self._pairwise()
        np.fill_diagonal(d, np.inf)
        obs = np.zeros((self.n, OBS_DIM))

        obs[:, 0:2] = self.vel / MAX_SPEED

        k = min(K_NEIGHBOURS, max(self.n - 1, 0))
        if k > 0:
            idx = np.argsort(d, axis=1)[:, :k]
            for i in range(self.n):
                for j, nb in enumerate(idx[i]):
                    if d[i, nb] > OBS_RADIUS:
                        continue
                    base = 2 + j * 4
                    obs[i, base:base + 2] = (self.pos[nb] - self.pos[i]) / OBS_RADIUS
                    obs[i, base + 2:base + 4] = (self.vel[nb] - self.vel[i]) / MAX_SPEED

        w = 2 + K_NEIGHBOURS * 4
        fx = (self.pos[:, 0] - self.xmin) / self.width
        fy = (self.pos[:, 1] - self.ymin) / self.height
        obs[:, w + 0] = fx
        obs[:, w + 1] = 1.0 - fx
        obs[:, w + 2] = fy
        obs[:, w + 3] = 1.0 - fy

        t = w + 4
        obs[:, t:t + 3] = self._task_obs()
        return obs

    def _task_obs(self):
        out = np.zeros((self.n, 3))
        if self.task == "gather":
            v = self.goal - self.pos
            dist = np.linalg.norm(v, axis=1, keepdims=True)
            out[:, :2] = v / np.maximum(dist, 1e-9)
            out[:, 2:3] = np.clip(dist / self.span, 0, 1)
            return out

        unvisited = np.argwhere(~self.visited)
        if len(unvisited) == 0:
            return out
        centres = np.array([self._cell_centre(i, j) for i, j in unvisited])
        for i in range(self.n):
            v = centres - self.pos[i]
            dist = np.linalg.norm(v, axis=1)
            j = int(np.argmin(dist))
            if dist[j] < 1e-6:
                continue
            out[i, :2] = v[j] / dist[j]
            out[i, 2] = min(dist[j] / self.span, 1.0)
        return out

    # -- reporting ---------------------------------------------------------

    def info(self):
        d = self._pairwise()
        np.fill_diagonal(d, np.inf)
        return {
            "coverage": float(self.visited.mean()),
            "collisions": int((d < COLLIDE_DIST).sum() // 2),
            "spread": float(np.linalg.norm(self.pos - self.pos.mean(0), axis=1).mean()),
        }


def sphero_command(vel):
    """Convert a simulated velocity vector to (heading_deg, speed_byte)."""
    speed = float(np.linalg.norm(vel))
    heading = (np.degrees(np.arctan2(vel[0], vel[1]))) % 360.0
    return heading, int(np.clip(speed / MAX_SPEED * 255.0, 0, 255))
