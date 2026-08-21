"""Reynolds boids — the hardcoded baseline.

Same interface as a trained policy: state in, actions out. Swapping between
this and a checkpoint is a one-line change in the app.
"""

import numpy as np

from .sim import OBS_RADIUS


class Boids:
    label = "boids"

    def __init__(self, separation=1.6, alignment=1.0, cohesion=0.8,
                 task_pull=0.6, sep_dist=25.0):
        self.separation = separation
        self.alignment = alignment
        self.cohesion = cohesion
        self.task_pull = task_pull
        self.sep_dist = sep_dist

    def act(self, env, obs=None):
        n = env.n
        pos, vel = env.pos, env.vel
        out = np.zeros((n, 2))

        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        near = dist < OBS_RADIUS

        task = env._task_obs()

        for i in range(n):
            nb = np.where(near[i])[0]
            v = np.zeros(2)

            if len(nb):
                close = nb[dist[i, nb] < self.sep_dist]
                if len(close):
                    push = diff[i, close] / dist[i, close][:, None]
                    v += self.separation * push.sum(axis=0)

                mean_v = vel[nb].mean(axis=0)
                nv = np.linalg.norm(mean_v)
                if nv > 1e-6:
                    v += self.alignment * mean_v / nv

                centre = pos[nb].mean(axis=0) - pos[i]
                nc = np.linalg.norm(centre)
                if nc > 1e-6:
                    v += self.cohesion * centre / nc

            v += self.task_pull * task[i, :2]
            v += 1.2 * self._walls(pos[i], env.bbox)

            m = np.linalg.norm(v)
            out[i] = v / m if m > 1.0 else v
        return out

    @staticmethod
    def _walls(p, bbox, margin=25.0):
        xmin, xmax, ymin, ymax = bbox
        f = np.zeros(2)
        for ax, (lo, hi) in enumerate([(xmin, xmax), (ymin, ymax)]):
            if p[ax] < lo + margin:
                f[ax] += (lo + margin - p[ax]) / margin
            elif p[ax] > hi - margin:
                f[ax] -= (p[ax] - (hi - margin)) / margin
        return f


class Idle:
    label = "idle"

    def act(self, env, obs=None):
        return np.zeros((env.n, 2))
