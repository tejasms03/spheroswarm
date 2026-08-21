"""A simulated robot.

Runs the same dynamics as `swarm.sim`: command latency, first-order motor lag,
a per-robot gain and a heading bias. A sim robot and a real robot should feel
close enough that a controller tuned on one is not surprised by the other.
"""

from collections import deque

import numpy as np

from .handle import MAX_SPEED, RobotHandle, now

# Guesses, not measurements. A real ball has not been characterised yet — see
# `tools/measure_drift.py` for how to replace these with numbers off hardware.
DRIFT_DEG_PER_MIN = 8.0     # random-walk scale on the heading bias
DRIFT_LIMIT = 0.6           # rad, ~34 deg; a bounded walk, never a runaway
SLIP_MAX = 0.06             # fraction of a command that can be lost

BATTERY_DRAIN_PER_S = 1.0 / (90 * 60)     # a notional 90-minute run


class SimRobot(RobotHandle):
    kind = "sim"

    # The simulator rotates the commanded velocity vector directly, so its
    # offset runs the same way the estimator's residual does. Opposite to
    # `SpheroRobot`; see the note there.
    HEADING_SIGN = 1.0

    def __init__(self, name, code, color, workspace=None, pos=None, seed=None,
                 randomize=True):
        super().__init__(name, code, color, workspace)
        rng = np.random.default_rng(seed)
        self.rng = rng

        if randomize:
            self.latency = int(rng.integers(1, 4))
            self.tau = float(rng.uniform(0.25, 0.5))
            self.gain = float(rng.uniform(0.8, 1.2))
            self.bias = float(rng.uniform(-0.12, 0.12))
            # A real Sphero's heading is its own gyro estimate, and that
            # estimate wanders. A constant bias is a robot you can calibrate
            # once and forget; a drifting one is why the offset has to be
            # tracked continuously rather than measured at startup. Modelled as
            # a random walk on the bias, bounded so it cannot run away.
            self.drift_rate = float(rng.uniform(0.5, DRIFT_DEG_PER_MIN))
            self.slip = float(rng.uniform(0.0, SLIP_MAX))
        else:
            self.latency, self.tau, self.gain, self.bias = 2, 0.35, 1.0, 0.0
            self.drift_rate = 0.0
            self.slip = 0.0
        self._drift = 0.0
        self._rng = rng

        self.queue = deque([np.zeros(2)] * self.latency, maxlen=self.latency)
        self.battery = 1.0
        self.last_seen = now()
        # The simulation's own clock, advanced by dt. The estimator is fed from
        # this rather than from `now()`: a test loop runs thirty simulated
        # seconds in a few real milliseconds, so on a wall clock nothing is ever
        # far enough apart to measure and no interval ever elapses.
        self._t = 0.0

        if pos is not None:
            self.pos = np.asarray(pos, dtype=float).copy()
        elif workspace is not None:
            self.pos = np.asarray(workspace.random_valid_point(rng), dtype=float)
        else:
            self.pos = np.zeros(2)

    @property
    def connected(self):
        return True

    def set_velocity(self, v):
        v = np.asarray(v, dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all():
            return
        speed = float(np.linalg.norm(v))
        if speed > MAX_SPEED:
            v = v / speed * MAX_SPEED
        self._desired = v
        speed = float(np.linalg.norm(v))
        if speed > 1e-6:
            # What was REQUESTED, before the offset is applied — math
            # convention, matching how the estimator measures travel.
            self._believed_course = float(
                np.degrees(np.arctan2(v[1], v[0])) % 360.0)

    def set_led(self, rgb, blink=None):
        self.rgb = tuple(int(np.clip(c, 0, 255)) for c in rgb)
        self.blink = blink

    def stop(self):
        self._desired = np.zeros(2)
        self.queue.append(np.zeros(2))

    def step(self, dt):
        # Heading drift: a bounded random walk, in radians. Bounded because an
        # unbounded walk eventually points a robot backwards, which teaches a
        # controller nothing except that the world is broken.
        if self.drift_rate:
            self._drift += float(self._rng.normal(
                0.0, np.radians(self.drift_rate) * np.sqrt(max(dt, 1e-6) / 60.0)))
            self._drift = float(np.clip(self._drift, -DRIFT_LIMIT, DRIFT_LIMIT))

        # The offset rotates the command before the bias rotates it back, which
        # is exactly what it does on a real ball: `heading + heading_offset` is
        # applied on the way out, and the world adds its own error after.
        total = self.bias + self._drift + np.radians(self.heading_offset)
        c, s = np.cos(total), np.sin(total)
        d = self._desired
        rotated = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])

        self.queue.append(rotated)
        cmd = self.queue[0] * self.gain

        if self.slip:
            # Loses a little of each command, the way a light ball does on a
            # smooth floor. Multiplicative, so it never adds energy.
            cmd = cmd * (1.0 - self.slip * float(self._rng.random()))
        self.vel += (cmd - self.vel) * min(dt / self.tau, 1.0)
        self.pos = self.pos + self.vel * dt

        if self.ws is not None and not self.ws.is_valid_point(self.pos):
            self.pos = np.asarray(self.ws.nearest_valid_point(self.pos), dtype=float)
            self.vel = np.zeros(2)

        self.battery = max(0.0, self.battery - BATTERY_DRAIN_PER_S * dt)
        self.last_seen = now()
        self._t += dt
        # Same path a real robot takes with no usable gyro, so the camera-only
        # fallback is exercised by the whole suite rather than only on the day
        # a sensor stream drops.
        self.observe_heading(self._t, None)
