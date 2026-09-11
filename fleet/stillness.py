"""Is this ball moving, by its own accelerometer?

For deciding which balls may blink their identity. A ball the CAMERA calls
still is judged by whichever blob carries its name, and when the name is wrong
that is the wrong ball -- so the one question identity depends on cannot be
answered by the thing identity is supposed to correct. The ball's own sensor
knows about itself whatever the camera thinks.

The accelerometer, not the gyroscope. A Sphero's IMU rides on the inner drive
assembly, which hangs level while the shell rolls around it, so a ball rolling
steadily in a straight line turns its gyro hardly at all. What it cannot hide
is vibration: wheels on a shell on a floor shake the assembly, and a parked
ball reads gravity and nothing else. Spread over a short window separates the
two.

A VETO, never the whole answer. A stream that stops usually leaves its last
value in place, and a frozen value is indistinguishable from a perfectly still
ball -- so this can say "moving", or "no idea", and the caller decides "still"
only together with its own record of what it last told the ball to do.

`spherov2` streams the accelerometer every 150 ms, so a one-second window is
about seven readings. `QUIET_G` is a first guess and has to be set from the
numbers a real ball reads at rest and rolling on this floor.
"""

from collections import deque

import numpy as np

IMU_PERIOD_S = 0.15     # spherov2's streaming interval
WINDOW_S = 1.0
MIN_SAMPLES = 5         # fewer than this in a window is not a verdict
STALE_S = 0.6           # newest reading older than this: the stream is quiet
QUIET_G = 0.02          # per-axis spread below this is a ball at rest. GUESS.


def as_vector(reading):
    """A reading as three floats, or None. spherov2 hands back a dict."""
    if reading is None:
        return None
    try:
        if isinstance(reading, dict):
            v = [reading.get("x"), reading.get("y"), reading.get("z")]
        else:
            v = list(reading)[:3]
        v = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if v.shape != (3,) or not np.isfinite(v).all():
        return None
    return v


class Stillness:
    def __init__(self, window_s=WINDOW_S, quiet_g=QUIET_G,
                 min_samples=MIN_SAMPLES, stale_s=STALE_S):
        self.window_s = float(window_s)
        self.quiet_g = float(quiet_g)
        self.min_samples = int(min_samples)
        self.stale_s = float(stale_s)
        self.samples = deque()

    def add(self, t, reading):
        """Record one reading taken at `t`. Returns whether it was usable."""
        v = as_vector(reading)
        if v is None:
            return False
        self.samples.append((float(t), v))
        while self.samples and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        return True

    def _window(self, now):
        if not self.samples or now - self.samples[-1][0] > self.stale_s:
            return None
        pts = [v for t, v in self.samples if now - t <= self.window_s]
        return pts if len(pts) >= self.min_samples else None

    def sigma(self, now):
        """The largest per-axis spread in the window, in g, or None."""
        pts = self._window(now)
        if pts is None:
            return None
        return float(np.max(np.std(np.asarray(pts), axis=0)))

    def quiet(self, now):
        """True at rest, False moving, None when the stream has nothing to say."""
        s = self.sigma(now)
        return None if s is None else s < self.quiet_g
