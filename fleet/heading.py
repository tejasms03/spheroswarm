"""Estimating which way a Sphero is actually pointing.

The camera sees a glowing sphere with no facing, so heading is never measured
directly. Two partial sources exist and they fail in opposite ways:

    gyro yaw        instantaneous, works at rest, but DRIFTS, and is in the
                    ball's own aim frame rather than the camera's
    travel direction  drift-free and absolute in the camera frame, but only
                    exists while moving, and its noise grows as the robot slows

So the thing to estimate is not the heading — it is the constant `offset`
between the two frames:

    heading_camera = yaw + offset

One unknown, observable whenever the robot moves, applied even when it does
not. It also self-heals: pick a ball up, put it down rotated, and the next few
seconds of motion correct it.

Four things decide how accurate this gets, in order of how much they matter:

**Baseline length.** Angle noise from position noise sigma over a displacement
d is about sigma/d radians. At 30fps and 45cm/s a consecutive-frame delta is
1.5cm, so 1cm of position noise is +-37 degrees of heading noise — useless.
Measured over a 20cm baseline the same noise is +-3 degrees. Waiting for
distance is the single biggest lever.

**Turning rejection.** Travel direction over a window where the robot turned is
a chord, not a heading. Samples spanning more than a few degrees of yaw change
are discarded rather than corrected.

**Circular statistics.** Angles are averaged as unit vectors. A scalar mean of
359 and 1 is 180, which is the exact opposite of the answer.

**Outlier gating.** A collision, a tracker mis-association or a wheel slip
produces a wild sample. Anything far from the current estimate is dropped once
the estimate has settled.
"""

import math
from collections import deque

DEFAULT_MIN_BASELINE = 15.0     # cm of travel before a sample is trusted
DEFAULT_MAX_TURN = 12.0         # deg of yaw change tolerated across a sample
DEFAULT_GATE = 60.0             # deg from the estimate before a sample is an outlier
# The trail has to span enough TIME to accumulate the baseline at the slowest
# speed worth calibrating at. 40 samples is 1.3s at 30Hz — only 20cm at
# 15cm/s, so a slowly creeping robot never reached a 25cm baseline and never
# calibrated at all. 200 samples is ~6.7s, which is 100cm at 15cm/s.
DEFAULT_TRAIL = 200             # samples of (t, x, y, yaw) kept
SETTLED_AFTER = 4               # accepted samples before gating switches on
CONFIDENT_AFTER = 8             # accepted samples before confidence can reach 1
STALE_CORRECTION = 12.0         # s since a correction before confidence hits 0
REVERSAL_WINDOW = 40.0          # deg either side of 180 that reads as reversing


def wrap180(deg):
    """Fold an angle into -180..180."""
    return (deg + 180.0) % 360.0 - 180.0


def circular_mean(angles_deg, weights=None):
    """Mean of angles, as unit vectors. Returns degrees, or None if empty."""
    if not angles_deg:
        return None
    w = weights or [1.0] * len(angles_deg)
    sx = sum(wi * math.cos(math.radians(a)) for a, wi in zip(angles_deg, w))
    sy = sum(wi * math.sin(math.radians(a)) for a, wi in zip(angles_deg, w))
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return None
    return math.degrees(math.atan2(sy, sx))


class HeadingEstimator:
    """Fuses gyro yaw with camera travel to track one robot's frame offset.

    Feed it every tick. It decides for itself which moments are informative.
    """

    def __init__(self, min_baseline=DEFAULT_MIN_BASELINE,
                 max_turn=DEFAULT_MAX_TURN, gate=DEFAULT_GATE,
                 trail=DEFAULT_TRAIL, alpha=0.25):
        self.min_baseline = float(min_baseline)
        self.max_turn = float(max_turn)
        self.gate = float(gate)
        self.alpha = float(alpha)          # smoothing on accepted samples
        self._trail = deque(maxlen=trail)
        self.stale_after = 8.0         # s; a baseline older than this is history
        self.offset = None                 # degrees, None until first sample
        self.samples = 0                   # accepted
        self.rejected = {"short": 0, "turning": 0, "outlier": 0, "reversing": 0}
        self.last_residual = None
        self.seeded = False
        self.last_corrected = None         # t of the most recent accepted sample
        self.last_t = None
        self.yaw = 0.0                     # integrated, for callers with a gyro

    # -- feeding ---------------------------------------------------------

    def restart(self):
        """Forget the estimate and the trail, keep the configuration.

        Called whenever the applied offset moves. Every sample in the trail was
        observed under the old value, so carrying them over would have the loop
        correcting for an error it has already corrected — which is a loop that
        oscillates rather than converging.
        """
        self._trail.clear()
        self.offset = None
        self.samples = 0
        self.last_residual = None
        self.last_corrected = None
        self.seeded = False
        return self

    def seed(self, offset_deg):
        """Start from a previously calibrated value instead of from nothing.

        A robot whose offset was measured last session should drive correctly
        from the first command, not after the first twenty centimetres of
        travel. The seed is treated as one ordinary sample rather than as
        gospel, so a stale one is corrected away instead of being defended.
        """
        if offset_deg is None:
            return self
        self.offset = wrap180(float(offset_deg))
        self.samples = max(self.samples, 1)
        self.seeded = True
        return self

    def predict(self, yaw_rate_deg_s, dt):
        """Integrate a gyro rate between camera fixes.

        The camera is the only drift-free source and it arrives at 30Hz with a
        long baseline behind it; a gyro fills the gaps and, more usefully,
        keeps a heading alive while the robot is stationary, which is precisely
        when travel direction does not exist.
        """
        if yaw_rate_deg_s is None:
            return self.yaw
        self.yaw = (self.yaw + float(yaw_rate_deg_s) * float(dt)) % 360.0
        return self.yaw

    def update(self, t, x, y, yaw_deg=None):
        """One observation. Returns the offset in degrees, or None.

        `yaw_deg` is whatever the robot believes its own aim was — a gyro
        reading if there is one, or simply the heading that was commanded. Both
        work, because what is being estimated is the difference between that
        belief and where the ball actually went. Commanded heading is the
        camera-only fallback and it degrades rather than failing: it has no
        knowledge of slip, so a slipping robot's samples are wrong in the same
        direction the outlier gate is there to catch.
        """
        if yaw_deg is None:
            yaw_deg = self.yaw
        else:
            self.yaw = float(yaw_deg) % 360.0
        self.last_t = float(t)
        self._trail.append((float(t), float(x), float(y), float(yaw_deg)))
        sample = self._measure()
        if sample is not None:
            self._accept(sample)
        return self.offset

    def _measure(self):
        """The oldest trail point far enough back to give a long baseline."""
        if len(self._trail) < 2:
            return None
        t1, x1, y1, yaw1 = self._trail[-1]

        for i in range(len(self._trail) - 2, -1, -1):
            t0, x0, y0, yaw0 = self._trail[i]
            if t1 - t0 > self.stale_after:
                break              # too long ago to be one continuous motion
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist < self.min_baseline:
                continue

            # Turned mid-baseline: the straight line between the endpoints is a
            # chord, and its direction is nobody's heading.
            if abs(wrap180(yaw1 - yaw0)) > self.max_turn:
                self.rejected["turning"] += 1
                return None

            travel = math.degrees(math.atan2(dy, dx))
            yaw_mid = circular_mean([yaw0, yaw1])
            # Longer baselines are proportionally less noisy, so weight by it.
            return wrap180(travel - yaw_mid), dist

        self.rejected["short"] += 1
        return None

    def _accept(self, sample):
        residual, dist = sample
        if self.offset is None:
            self.offset = residual
            self.samples = 1
            self.last_residual = 0.0
            return

        delta = wrap180(residual - self.offset)
        # A ball that went the other way is not a ball whose frame turned round.
        # Slewing half a turn to accommodate one such sample undoes a settled
        # estimate, and the next sample undoes the undoing.
        if self.samples >= SETTLED_AFTER and abs(abs(delta) - 180.0) < REVERSAL_WINDOW:
            self.rejected["reversing"] += 1
            return
        if self.samples >= SETTLED_AFTER and abs(delta) > self.gate:
            self.rejected["outlier"] += 1
            return

        # Weight longer baselines harder, and ease in rather than jumping: a
        # single good sample should not overwrite a settled estimate.
        w = min(1.0, dist / (self.min_baseline * 2.0)) * self.alpha
        self.offset = wrap180(self.offset + w * delta)
        self.samples += 1
        self.last_residual = delta
        self.last_corrected = self._trail[-1][0] if self._trail else None

    # -- using -----------------------------------------------------------

    def heading(self, yaw_deg):
        """Camera-frame heading in degrees, or None before the first sample."""
        if self.offset is None:
            return None
        return wrap180(yaw_deg + self.offset) % 360.0

    @property
    def ready(self):
        return self.offset is not None and self.samples >= SETTLED_AFTER

    @property
    def confidence(self):
        """0 to 1. Falls with time since the last correction, rises with agreement.

        Exposed because a stale heading and a fresh one are indistinguishable
        from their value alone, and the difference decides whether a controller
        should trust it or a person should look at the robot. A seeded estimate
        that has never been corrected is deliberately not confident: it is a
        claim from a previous session, not an observation.
        """
        if self.offset is None:
            return 0.0
        if self.last_corrected is None:
            return 0.15 if self.seeded else 0.0
        n = min(1.0, self.samples / float(CONFIDENT_AFTER))
        age = 0.0 if self.last_t is None else max(0.0, self.last_t - self.last_corrected)
        fresh = max(0.0, 1.0 - age / STALE_CORRECTION)
        agree = 1.0
        if self.last_residual is not None:
            agree = max(0.0, 1.0 - abs(self.last_residual) / max(self.gate, 1e-6))
        return round(float(n * fresh * (0.5 + 0.5 * agree)), 3)

    def drifted_from(self, seed_deg):
        """Degrees between the live estimate and a calibrated value, or None.

        Large means the ball was picked up, bumped, or is being tracked as
        somebody else — all of which look identical from the control loop and
        none of which it can fix.
        """
        if self.offset is None or seed_deg is None:
            return None
        return abs(wrap180(self.offset - float(seed_deg)))

    def state(self):
        return {"offset_deg": None if self.offset is None else round(self.offset, 1),
                "samples": self.samples, "ready": self.ready,
                "confidence": self.confidence,
                "last_residual_deg": None if self.last_residual is None
                else round(self.last_residual, 1),
                "rejected": dict(self.rejected)}


# -- active calibration ------------------------------------------------------

CAL_LEG_CM = 22.0          # travel per leg; long enough that noise is small
# …but a short leg beats no leg. Heading noise is roughly sigma over baseline,
# so 22cm with 1cm of position noise is about 2.6 degrees and 14cm is about 4 —
# both far better than the 30-plus a stale offset costs. A small arena gets a
# shorter leg and a slightly noisier answer, rather than a refusal.
CAL_MIN_LEG_CM = 13.0
# Slow on purpose. The leg length is what makes the measurement good — noise is
# sigma over distance — and speed contributes nothing to it except how far past
# the end of the leg the ball coasts. A ball that overshoots into a wall costs
# the whole calibration; one that takes three seconds instead of one costs
# nothing at all.
CAL_SPEED = 13.0           # cm/s
CAL_HEADINGS = (0.0, 90.0, 180.0, 270.0)
CAL_LEG_TIMEOUT = 6.0      # s before a leg is abandoned
# Clearance a leg needs beyond its own length. Unchanged by the slowdown, and
# that is the point: the margin exists to cover the coast at the end of a leg,
# and a slower leg coasts less. Raising it alongside the speed reduction was
# the wrong instinct — it made legs impossible to place near a wall while
# solving a problem the slowdown had already solved.
# Clearance a leg needs beyond its own length: the coast at CAL_SPEED, with
# room to spare. At 13cm/s that coast is about 11cm, so 35 was three times what
# it needed and it made a 110cm arena uncalibratable — the four cardinal legs
# wanted 57cm each and there were 55.
CAL_MARGIN = 18.0
CAL_SPREAD_LIMIT = 25.0    # deg of disagreement between legs before we distrust it


class ActiveCalibration:
    """Drive a known pattern and read the frame offset straight off it.

    The passive estimator has a cold start: it needs the robot to travel before
    it knows anything, and until then every command goes out in an unknown
    frame. This removes that. Command a heading in the BALL's frame, watch
    where it actually goes in the CAMERA's frame, and the difference is the
    offset — no gyro involved, nothing to converge.

    Several legs rather than one, for two reasons: averaging cancels the noise
    in any single measurement, and *disagreement* between legs is diagnostic.
    A ball that reads +40 going north and -60 going east is not badly
    calibrated, it is slipping, being pushed, or is not the ball the tracker
    is following.

    A state machine rather than a blocking routine, because the render loop has
    to keep running: call `step` once a frame, drive the velocity it returns,
    stop when `done`.
    """

    def __init__(self, headings=CAL_HEADINGS, leg_cm=CAL_LEG_CM,
                 speed=CAL_SPEED, timeout=CAL_LEG_TIMEOUT, workspace=None,
                 margin=CAL_MARGIN):
        self.headings = list(headings)
        self.leg_cm = float(leg_cm)
        self.speed = float(speed)
        self.timeout = float(timeout)
        self.ws = workspace
        self.margin = float(margin)

        self.leg = 0
        self.start = None
        self.leg_target = float(leg_cm)
        self._recent = []
        self.elapsed = 0.0
        self.residuals = []          # (commanded, travel, residual, distance)
        self.done = False
        self.error = None
        self.offset = None
        self.spread = None

    # -- driving ---------------------------------------------------------

    def step(self, pos, dt):
        """One frame. Returns the velocity to command, or None when finished."""
        import numpy as np

        if self.done:
            return None
        p = np.asarray(pos, dtype=float)

        if self.leg >= len(self.headings):
            self._finish()
            return None

        heading = self.headings[self.leg]
        if self.start is None:
            if not self._room_for(p, heading):
                # No space this way. Try the opposite, which is equally
                # informative — the residual is the same modulo 180, and we
                # know which 180 because we commanded it.
                heading = (heading + 180.0) % 360.0
                self.headings[self.leg] = heading
                if not self._room_for(p, heading):
                    self.error = ("not enough clear space to calibrate — move "
                                  "the robot toward the middle of the arena")
                    self.done = True
                    return None
            self.start = p.copy()
            self._recent = [p.copy()]
            self.leg_target = self._leg_for(p, heading) or self.leg_cm
            self.elapsed = 0.0

        self.elapsed += dt
        # Against a smoothed position, not a raw one. The leg ends the first
        # time the measured distance crosses the threshold, so camera noise can
        # end it early — and every frame is another chance to. Driving slower
        # means more frames per leg and therefore MORE early endings, which is
        # the opposite of what slowing down was supposed to buy. A short median
        # costs nothing and removes the bias.
        self._recent.append(p.copy())
        del self._recent[:-5]
        smooth = np.median(np.stack(self._recent), axis=0)
        travelled = float(np.linalg.norm(smooth - self.start))

        if travelled >= self.leg_target:
            self._record(smooth, heading, travelled)
            return np.zeros(2)
        if self.elapsed > self.timeout:
            self.error = (f"leg {self.leg + 1} moved only {travelled:.0f}cm in "
                          f"{self.timeout:.0f}s — is the robot stuck, or is the "
                          "camera tracking something else?")
            self.done = True
            return None

        rad = math.radians(heading)
        return np.array([math.cos(rad), math.sin(rad)]) * self.speed

    def _room_for(self, p, heading):
        return self._leg_for(p, heading) is not None

    def _leg_for(self, p, heading):
        """The longest leg that fits this way, or None if even the shortest does not.

        Asking "does the nominal leg fit?" is the wrong question in a small
        room: it answers no and the calibration refuses, leaving the robot with
        an unknown frame — which costs far more accuracy than a shorter leg
        does. So this asks how much floor there is and takes what it can.
        """
        if self.ws is None:
            return self.leg_cm
        rad = math.radians(heading)
        for leg in (self.leg_cm, self.leg_cm * 0.75, CAL_MIN_LEG_CM):
            if leg < CAL_MIN_LEG_CM:
                break
            reach = leg + self.margin
            end = (p[0] + math.cos(rad) * reach, p[1] + math.sin(rad) * reach)
            try:
                ok = (self.ws.is_valid_point(end)
                      and self.ws.has_clearance(end, 10.0))
            except Exception:
                return leg
            if ok:
                return leg
        return None

    def _record(self, p, heading, travelled):
        import numpy as np

        d = p - self.start
        travel = math.degrees(math.atan2(d[1], d[0]))
        self.residuals.append((heading, travel, wrap180(travel - heading),
                               travelled))
        self.leg += 1
        self.start = None

    def _finish(self):
        self.done = True
        if not self.residuals:
            self.error = "no usable legs"
            return
        res = [r[2] for r in self.residuals]
        self.offset = wrap180(circular_mean(res, [r[3] for r in self.residuals]))
        # How far the worst leg sits from the mean. Small means four
        # independent measurements agreed; large means something is wrong with
        # the robot or the tracking, not with the number.
        self.spread = max(abs(wrap180(r - self.offset)) for r in res)
        if self.spread > CAL_SPREAD_LIMIT:
            self.error = (f"legs disagree by up to {self.spread:.0f}deg — the "
                          "robot may be slipping, being pushed, or the camera "
                          "may be tracking a different robot")

    # -- reporting -------------------------------------------------------

    @property
    def progress(self):
        return f"leg {min(self.leg + 1, len(self.headings))}/{len(self.headings)}"

    def state(self):
        return {"done": self.done, "offset_deg": None if self.offset is None
                else round(self.offset, 1),
                "spread_deg": None if self.spread is None else round(self.spread, 1),
                "legs": len(self.residuals), "error": self.error}
