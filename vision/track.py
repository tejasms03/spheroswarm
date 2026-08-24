"""Tracking with per-robot Kalman filtering.

Detection is per-colour, so identity is unambiguous by construction — the value
the filter adds is riding through dropped frames, occlusions, and the brief
moments when two robots overlap and one blob disappears.
"""

import math
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

    # A Sphero is a known physical object: SPRK+ and BOLT are both ~7.4cm
    # across. The homography turns that into an expected size anywhere in the
    # frame, so a blob can be judged against the ball it claims to be rather
    # than against a hand-tuned pixel area — which half the palette does not
    # have set at all. Generous bounds: a lit shell blooms and a partly
    # occluded one shrinks. This is here to reject a ceiling light and a red
    # sock, not to measure anything.
    BALL_MIN_CM = 3.0
    BALL_MAX_CM = 16.0

    # How far outside the arena a blob may be and still be a robot. Not zero:
    # a ball that rolls out is exactly the one you need to see, and cropping at
    # the boundary trades a phantom you can name for a robot you cannot find.
    # Not unbounded either, because everything else in the room is out there.
    ARENA_MARGIN_CM = 40.0

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

    def __init__(self, assignment=None, homography=None, detector=None,
                 arena_only=True):
        self.assignment = assignment or {c: c for c in config.COLORS}
        self.H = homography or Homography.load()
        self.det = detector or Detector()
        self.tracks = {}
        self.t_last = None
        self.fps = 0.0
        # Look for robots where robots can be. Everything a camera sees beyond
        # the arena is furniture, and every piece of it that happens to sit in
        # a hue window is a blob the tracker can adopt.
        self.arena_only = bool(arena_only)
        self.outside = {}        # colour -> blobs seen beyond the margin

    @property
    def colors(self):
        return {c: n for n, c in self.assignment.items()}

    def step(self, frame):
        now = time.time()
        dt = min(max((now - self.t_last) if self.t_last else 1 / 30.0, 1e-3), 0.5)
        self.t_last = now
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt

        wanted = list(self.colors)
        found = self.det.candidates(frame, only=wanted)
        # One blob per colour for anything that wants a picture of what the
        # camera saw, rather than what the tracker concluded from it.
        raw = {c: v[0] for c, v in found.items() if v}

        cm = self._to_cm(found)

        for t in self.tracks.values():
            t.kf.predict(dt)
            t.missing += 1

        self.outside = {}
        if self.arena_only and self.H.ready:
            kept = {}
            for color, options in cm.items():
                near = [o for o in options if self._in_arena(o[0])]
                if len(near) != len(options):
                    self.outside[color] = len(options) - len(near)
                # Same fallback as everywhere else here: if nothing is inside,
                # keep what there is. A robot outside the margin is still the
                # best answer available, and losing it entirely is worse than
                # reporting it somewhere odd.
                kept[color] = near or options
            cm = kept

        for color, options in cm.items():
            name = self.colors[color]
            # Blobs the size of a ball, if any are. Falling back to all of them
            # rather than to none keeps a robot trackable when the sizing is
            # wrong — a bad filter should cost precision, never the robot.
            ball = [o for o in options
                    if self.BALL_MIN_CM <= o[1] <= self.BALL_MAX_CM]
            options = ball or options

            t = self.tracks.get(name)
            if t is None:
                # Nothing to prefer yet, so the biggest blob starts the track.
                self.tracks[name] = Track(name, color, options[0][0])
                continue
            # The candidate where this robot actually was, not the one that
            # happens to be largest this frame. Two blobs of similar size trade
            # places whenever their areas cross, and a tracker handed only the
            # larger one follows the trade instead of the robot.
            p, _ = min(options,
                       key=lambda o: float(np.linalg.norm(o[0] - t.kf.pos)))
            gate = self.GATE_CM * (1.0 if t.confident else self.YOUNG_GATE_MULTIPLE)
            if np.linalg.norm(p - t.kf.pos) > gate:
                continue                        # outlier, keep coasting
            t.kf.update(p)
            t.missing = 0
            t.seen += 1

        for n in [n for n, t in self.tracks.items() if not t.alive]:
            del self.tracks[n]

        return self.read(), raw

    def _in_arena(self, p):
        m = self.ARENA_MARGIN_CM
        return (-m <= float(p[0]) <= self.H.width + m
                and -m <= float(p[1]) <= self.H.height + m)

    def _to_cm(self, found):
        """{colour: [(point_cm, diameter_cm), ...]}, biggest first.

        One transform for every blob rather than one per blob: `to_cm` is a
        cv2 call and its per-call cost dwarfs the arithmetic inside it. The
        diameter comes from stepping one pixel sideways from each blob and
        measuring how far that moved in centimetres, so the local scale is
        used — which matters near the edges of a tilted frame, where a
        single frame-wide scale factor is wrong by tens of percent.
        """
        if not found:
            return {}
        flat, index, areas = [], [], []
        for color, blobs in found.items():
            for b in blobs:
                flat.append([b[0], b[1]])
                index.append(color)
                areas.append(float(b[2]))

        if self.H.ready:
            pts = self.H.to_cm(flat)
            edge = self.H.to_cm([[x + 1.0, y] for x, y in flat])
        else:
            pts = np.array(flat, dtype=float)
            edge = pts + np.array([1.0, 0.0])

        out = {}
        for color, p, q, a in zip(index, pts, edge, areas):
            p = np.asarray(p, dtype=float)
            cm_per_px = float(np.linalg.norm(np.asarray(q, dtype=float) - p))
            dia_px = 2.0 * math.sqrt(max(a, 1e-9) / math.pi)
            out.setdefault(color, []).append((p, dia_px * cm_per_px))
        return out

    def read(self):
        """The interface deploy.py expects."""
        return {n: t.kf.pos.copy() for n, t in self.tracks.items() if t.confident}

    def velocities(self):
        return {n: t.kf.vel.copy() for n, t in self.tracks.items() if t.confident}
