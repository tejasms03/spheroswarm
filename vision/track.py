"""Tracking with per-robot Kalman filtering.

Detection is per-colour, so identity is unambiguous by construction — the value
the filter adds is riding through dropped frames, occlusions, and the brief
moments when two robots overlap and one blob disappears.
"""

import time

import numpy as np

from . import config
from .detect import Detector
from .homography import Homography


class Kalman2D:
    """Constant-velocity model in centimetres."""

    def __init__(self, pos, dt=1.0 / config.FPS_TARGET, q=25.0, r=4.0):
        self.dt = dt
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 50.0
        self.Q = np.diag([q * dt, q * dt, q, q])
        self.R = np.eye(2) * r
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def F(self, dt):
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        return f

    def predict(self, dt=None):
        dt = dt or self.dt
        F = self.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
        return self.x[:2]

    def update(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()


class Track:
    def __init__(self, name, color, pos):
        self.name = name
        self.color = color
        self.kf = Kalman2D(pos)
        self.missing = 0
        self.seen = 1

    @property
    def alive(self):
        return self.missing < 15          # half a second of coasting

    @property
    def confident(self):
        return self.seen > 5 and self.missing < 4


class Tracker:
    """Colour blobs in, {name: (x_cm, y_cm)} out.

    `assignment` maps robot name -> colour, e.g. {"SK-1A2B": "red"}. Without one
    it tracks colours directly, which is what you want before robots exist.
    """

    GATE_CM = 60.0        # reject detections this far from the prediction
    # A track still building confidence has an unformed velocity estimate, so
    # its prediction can be further out than a settled one's. It gets a wider
    # gate, NOT no gate: a new track is initialised at its own first detection,
    # so its position is good immediately, and a robot crossing the arena in
    # one frame is not a thing that happens. A track that rejects everything
    # ages out through `missing` in half a second and is rebuilt where the
    # robot actually is, so being wrong here self-heals; adopting a phantom
    # does not.
    YOUNG_GATE_MULTIPLE = 2.0

    def __init__(self, assignment=None, homography=None, detector=None):
        self.assignment = assignment or {c: c for c in config.COLORS}
        self.H = homography or Homography.load()
        self.det = detector or Detector()
        self.tracks = {}
        self.t_last = None
        self.fps = 0.0

    @property
    def colors(self):
        return {c: n for n, c in self.assignment.items()}

    def step(self, frame):
        now = time.time()
        dt = min(max((now - self.t_last) if self.t_last else 1 / 30.0, 1e-3), 0.5)
        self.t_last = now
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt

        wanted = list(self.colors)
        raw = self.det.detect(frame, only=wanted)

        cm = {}
        if self.H.ready and raw:
            pts = self.H.to_cm([[v[0], v[1]] for v in raw.values()])
            cm = {k: p for k, p in zip(raw, pts)}
        elif raw:
            cm = {k: np.array([v[0], v[1]]) for k, v in raw.items()}

        for t in self.tracks.values():
            t.kf.predict(dt)
            t.missing += 1

        for color, p in cm.items():
            name = self.colors[color]
            t = self.tracks.get(name)
            if t is None:
                self.tracks[name] = Track(name, color, p)
                continue
            gate = self.GATE_CM * (1.0 if t.confident else self.YOUNG_GATE_MULTIPLE)
            if np.linalg.norm(p - t.kf.pos) > gate:
                continue                        # outlier, keep coasting
            t.kf.update(p)
            t.missing = 0
            t.seen += 1

        for n in [n for n, t in self.tracks.items() if not t.alive]:
            del self.tracks[n]

        return self.read(), raw

    def read(self):
        """The interface deploy.py expects."""
        return {n: t.kf.pos.copy() for n, t in self.tracks.items() if t.confident}

    def velocities(self):
        return {n: t.kf.vel.copy() for n, t in self.tracks.items() if t.confident}
