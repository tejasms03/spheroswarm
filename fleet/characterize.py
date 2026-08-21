"""Measuring a real Sphero, so the model stops guessing.

`fleet/sim_handle.py` models the three things that break sim-to-real transfer
on this hardware — command latency, motor lag, per-robot gain — and every one
of its numbers is a guess. This module replaces them with measurements taken
off a ball on the actual floor under the actual camera.

Five quantities, in the order they depend on each other:

  noise      how much the tracker's position jitters when nothing is moving.
             Everything below is a difference of positions, so this is the
             error bar on all of it, and it is free to measure.
  heading    the constant between the ball's aim frame and the camera's. Until
             it is known, "drive north" goes somewhere else and no leg can be
             aimed at clear floor.
  speed map  speed byte -> cm/s. The real gain, the real ceiling, and the byte
             below which the ball does not move at all.
  step       rest -> steady: dead time and the motor lag constant.
  brake      steady -> stop: how far it coasts. This is the speed limit,
             directly: the fastest the ball may go is the speed whose coast
             fits inside the arrival tolerance.
  latency    command issued -> camera sees it turn. The whole round trip: BLE
             write, motor spin-up, exposure, detection, filter. Nothing in the
             codebase measures this today and every controller gain depends
             on it.

Each stage is a state machine stepped once per frame rather than a blocking
routine, because the window has to keep drawing and the trainer has to be able
to hit stop. A stage returns the command to send — `(heading_deg, speed_byte)`
or None for "stop" — and raises its `done` flag when it has what it needs.

Every stage also runs unchanged against a `SimRobot`, which is what makes the
rig testable before a ball is on the floor: the numbers that come back are the
sim's own guesses, and seeing them recovered correctly is the proof that the
fitting works.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from . import safety
from .handle import MAX_SPEED
from .heading import ActiveCalibration, circular_mean, wrap180

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "calib"
MOTION_PATH = CALIB / "motion.json"

STOP_SPEED = 2.0            # cm/s below which we call a ball stopped
# No Sphero goes this fast. A fit that says otherwise is not describing a
# quick robot, it is a slope extrapolated a long way from a short, noisy span
# of the curve — and because every other figure is derived from it, one such
# number turns a whole calibration into fiction. A run capped at 18cm/s
# measures bytes up to 76 and then reaches for 255, and a slope wrong by three
# reported a ceiling of 288cm/s and a deadband of 20cm/s that was simply that
# ceiling scaled back down.
PHYSICAL_MAX_CM_S = 95.0
TRUSTWORTHY_TOP_BYTE = 150      # below this the top of the curve is a guess
# Clear floor a leg wants beyond its own length. It is not a safety margin so
# much as a stopping distance: a ball asked to travel 24cm at byte 200 arrives
# doing 47cm/s and runs on for another 20 before it settles, so a leg planned
# to the wall ends at the wall.
SETTLE_MARGIN = 45.0


def _slope_speed(samples):
    """cm/s from a window of (t, pos), by least squares rather than endpoints.

    Two-point differencing throws away every sample in between and keeps all of
    the noise in the two it uses. Over a one-second window at 30fps that is
    thirty measurements discarded to make a worse one.
    """
    if len(samples) < 4:
        return None, None
    t = np.array([s[0] for s in samples], dtype=float)
    p = np.array([s[1] for s in samples], dtype=float)
    t = t - t[0]
    vx = np.polyfit(t, p[:, 0], 1)[0]
    vy = np.polyfit(t, p[:, 1], 1)[0]
    speed = float(math.hypot(vx, vy))
    # Camera-frame heading in the Sphero convention: 0 is +y, clockwise.
    course = float(math.degrees(math.atan2(vx, vy)) % 360.0)
    return speed, course


# Planning legs uses a smaller margin than driving does. The limiter in
# `fleet/safety.py` is a backstop and gets the cautious number; a leg is
# deliberate, aimed at clear floor, and watched — so 1.5 stopping distances is
# enough and 2.5 makes the fast end of a speed map unmeasurable in a 200cm room.
PLAN_SAFETY = 1.5
BACKUP_BYTE = 55            # crawling; this is positioning, not measuring

# A leg has a shortest useful length, and it is set by the motor rather than by
# the arena. Reaching a steady speed takes a few time constants, so a leg cut
# shorter than that measures the acceleration ramp and reports it as the top
# speed — byte 255 came back as 15cm/s that way, which is worse than not
# measuring byte 255 at all. Below this the byte is skipped and said to be.
MIN_SETTLE_S = 1.0          # ~3 motor time constants
MIN_WINDOW_S = 0.4          # ~12 samples at 30fps, enough for a slope

# A tracked ball is never perfectly still. Blob centroids move by a few tenths
# of a millimetre from shot noise alone, so a position that does not move AT
# ALL is not a still robot — it is a stationary artifact the tracker has
# mistaken for one, or a fix that has stopped updating. Both look identical to
# a stationary robot from the position alone, which is why every safety check
# in this file sailed past one: a run measured 0.017cm of noise, drove a
# calibration leg against a frozen reading, and put the ball into a wall
# because nothing it was told to do ever appeared to happen.
IMPLAUSIBLY_STILL_CM = 0.08     # sigma below this is not a measurement
# Judged per commanded CENTIMETRE, not per second. A fixed time window is a
# distance that depends on the speed: at the calibration crawl of 13cm/s a
# 1.5s window only asks for 20cm, which is under any sane threshold, so a
# frozen fix sailed through it and drove most of the arena before the leg
# timeout caught it. Asking "have we requested 25cm of travel yet?" fires after
# 1.9s at the crawl and 0.4s at full speed, which is what is wanted in both.
STUCK_TRAVEL_CM = 2.0           # ground covered, below which it is not moving
# Below this much commanded travel, silence is information rather than a fault:
# the speed map asks for bytes under the motor's deadband precisely to find out
# which ones do nothing, and reporting that as a broken tracker would abort the
# measurement that was working.
STUCK_MIN_EXPECTED_CM = 25.0


class StuckError(Exception):
    """Commanded to move, observed not to. Almost never a stuck robot."""


class Stage:
    """One measurement. Subclasses fill in `_step` and `result`."""

    label = "stage"
    detail = ""

    ws = None                   # subclasses that plan legs set this
    stop_s = safety.DEFAULT_STOP_S
    # How many stopping distances of clear floor a leg demands. Raising it
    # keeps the ball further from the walls at the cost of shorter legs — and
    # eventually of the fastest bytes, which are then skipped and said to be.
    plan_safety = PLAN_SAFETY
    max_byte = 255              # nothing is commanded faster than this

    def __init__(self, timeout=60.0):
        self.timeout = float(timeout)
        self.elapsed = 0.0
        self.done = False
        self.error = None
        self.samples = []           # (t, pos) for the whole stage
        self._stuck_since = None
        self._stuck_path = 0.0
        self._stuck_expected = 0.0

    def step(self, t, pos, dt):
        if self.done:
            return None
        self.elapsed += dt
        self.samples.append((t, np.asarray(pos, dtype=float).copy()))
        if self.elapsed > self.timeout:
            self.error = f"{self.label} timed out after {self.timeout:.0f}s"
            self.done = True
            return None
        cmd = self._step(t, np.asarray(pos, dtype=float), dt)
        if cmd is not None and cmd[1] > self.max_byte:
            cmd = (cmd[0], int(self.max_byte))
        return self._watch(t, cmd)

    def _watch(self, t, cmd):
        """Stop if a robot is being driven and the camera never notices.

        The failure this exists for: the tracker locked onto a stationary
        phantom, so every position it reported was the same. The stage
        commanded a leg, waited for travel that could never register, and drove
        the ball at full speed for the whole six-second leg timeout — out of
        the arena, out of frame, and across the room.

        Nothing else catches it. A frozen fix is indistinguishable from a
        stationary robot by position alone, and position is what every other
        guard here reads. The only tell is the CONTRADICTION: we are asking for
        motion and seeing none, and a real robot that cannot move still shows
        the tracker jittering by a millimetre or two.
        """
        moving = cmd is not None and cmd[1] > 0
        if not moving:
            self._stuck_since = None
            self._stuck_path = 0.0
            self._stuck_expected = 0.0
            return cmd
        if self._stuck_since is None:
            self._stuck_since = (t, self.samples[-1][1].copy())
            self._stuck_path = 0.0
            self._stuck_expected = 0.0
            return cmd

        gap = t - self.samples[-2][0] if len(self.samples) >= 2 else 0.0
        self._stuck_expected += (cmd[1] / 255.0 * MAX_SPEED) * max(gap, 0.0)

        # Ground covered, not distance from the start. A reversal ends near
        # where it began — that is what a reversal is — and judging it on net
        # displacement calls a perfectly behaved robot stuck.
        if len(self.samples) >= 2:
            self._stuck_path += float(np.linalg.norm(self.samples[-1][1]
                                                     - self.samples[-2][1]))
        t0, p0 = self._stuck_since
        expected = self._stuck_expected
        if expected < STUCK_MIN_EXPECTED_CM:
            return cmd
        travelled = self._stuck_path
        # Judged against what was ASKED for, not against a fixed distance. The
        # speed map deliberately commands bytes below the deadband and expects
        # to see nothing happen — that stall is the measurement. A robot told
        # to crawl at 4cm/s and observed not to move has told us something; one
        # told to do 45 and observed not to move has told us the tracker is
        # lying.
        if travelled >= max(STUCK_TRAVEL_CM, 0.2 * expected):
            self._stuck_since = (t, self.samples[-1][1].copy())
            self._stuck_path = 0.0
            self._stuck_expected = 0.0
            return cmd

        self.error = (
            f"asked for {expected:.0f}cm of travel over {t - t0:.1f}s and the "
            f"camera saw it cover {travelled:.1f}cm of ground. The robot is almost certainly moving and the "
            "tracker is not following it — a phantom blob of the same colour, "
            "or a fix that has stopped updating. Stopped before it left the "
            "arena; re-tune the colour and check the blob count.")
        self.done = True
        return None

    def _step(self, t, pos, dt):
        self.done = True
        return None

    # -- runway ----------------------------------------------------------

    def drive_byte(self, byte):
        """The byte that will actually go out, after the cap.

        Planning with the requested byte and driving the capped one is a quiet
        way to get both wrong: the leg is sized for a speed it never reaches,
        so it is cut far shorter than the floor allowed and the measurement is
        made on a stub of the ramp.
        """
        return min(int(byte), int(self.max_byte))

    def runway(self, pos, course, byte, want_s):
        """Seconds of driving available that way, and how many are wanted.

        Everything fast in this file used to specify a leg as a duration, which
        is a distance in disguise: 2.5 seconds at byte 255 is a metre and a
        half, and there is not a metre and a half in front of a ball sitting in
        the middle of a two-metre room. So a leg now asks how much floor there
        is and takes what it can get.
        """
        speed = self.drive_byte(byte) / 255.0 * MAX_SPEED
        have = safety.available_seconds(self.ws, pos, course, speed,
                                        stop_s=self.stop_s,
                                        safety=self.plan_safety)
        return have, want_s

    def need_backup(self, pos, course, byte, want_s):
        return self.runway(pos, course, byte, want_s)[0] < want_s

    def rest_after_backup(self, t):
        """Wait for the crawl backwards to actually stop before the leg starts.

        A leg that begins while the ball is still rolling the other way spends
        its settle reversing rather than reaching speed, and then reports the
        ramp as the steady value — byte 255 came back at 47cm/s instead of 59
        for exactly this reason. It is the same mistake as starting a step
        response mid-roll, one phase earlier.
        """
        return _at_rest(self.samples, t, span=0.4, tol=1.2)

    def backup_command(self, pos, course, byte, want_s):
        """Reverse slowly to buy runway. None once there is enough, or no more.

        A person setting up a fast run walks the ball to the far end first.
        Doing the same here is what makes the whole arena usable instead of the
        half of it that happens to lie ahead of wherever the last leg finished.
        """
        have, want = self.runway(pos, course, byte, want_s)
        if have >= want:
            return None
        back = (course + 180.0) % 360.0
        if safety.available_cm(self.ws, pos, back) < 12.0:
            return None                 # nothing behind us either; take what we have
        return (back, BACKUP_BYTE)

    def result(self):
        return {}

    @property
    def progress(self):
        return ""


# -- 0. is the tracker actually following this robot? ------------------------

class TrackingCheck(Stage):
    """Twitch it and confirm the camera sees the twitch. Before anything else.

    Every guard in this file — the speed field, the edge margin, the runway
    planning — protects the position the TRACKER reports. If the tracker is
    following a phantom blob of the same colour, all of them are guarding a
    ghost while the real ball drives wherever it likes, and the first sign of
    trouble is a robot on the floor across the room.

    So the run does not begin on trust. It nudges the ball a few centimetres
    each way and checks the camera saw a movement of roughly the right size in
    roughly the right direction. Six seconds, and it is the difference between
    a bad calibration and a lost robot.
    """

    label = "tracking check"
    detail = "a short nudge each way, to prove the camera is watching THIS robot"

    NUDGE_S = 1.1
    REST_S = 0.8
    # Enough commanded travel that a working camera must have noticed. Bailing
    # here rather than at the end of the nudge is the difference between 12cm
    # of driving on a dead tracker and 43cm, which in a small arena is the
    # difference between a refusal and a robot on the floor.
    BAIL_AFTER_CM = 9.0
    MIN_SEEN_CM = 3.0           # of movement, before we believe it at all
    MAX_ANGLE_ERR = 70.0        # degrees between commanded and observed

    def __init__(self, byte=55, workspace=None, aim=None, timeout=40.0):
        super().__init__(timeout=timeout)
        self.byte = int(byte)
        self.ws, self.aim = workspace, aim or (lambda c: c)
        self.leg = 0
        self.phase, self.phase_t = "aim", 0.0
        self.course = None
        self.start = None
        self.rows = []

    @property
    def progress(self):
        return f"nudge {min(self.leg + 1, 2)}/2"

    def _step(self, t, pos, dt):
        if self.leg >= 2:
            self.done = True
            return None
        self.phase_t += dt

        if self.phase == "aim":
            prefer = 0.0 if self.course is None else (self.course + 180.0) % 360.0
            options = [prefer, (prefer + 180.0) % 360.0] + [h * 45.0 for h in range(8)]
            self.course = max(options, key=lambda c:
                              safety.available_cm(self.ws, pos, c))
            if safety.available_cm(self.ws, pos, self.course) < 14.0:
                self.error = ("no room even to nudge — put the robot nearer the "
                              "middle of the arena")
                self.done = True
                return None
            self.start = pos.copy()
            self.phase, self.phase_t = "nudge", 0.0
            return (self.aim(self.course), self.byte)

        if self.phase == "nudge":
            asked = (self.byte / 255.0 * MAX_SPEED) * self.phase_t
            seen_now = float(np.linalg.norm(pos - self.start))
            if asked > self.BAIL_AFTER_CM and seen_now < 1.0:
                self.rows.append({"commanded_deg": round(self.course, 1),
                                  "seen_cm": round(seen_now, 2), "seen_deg": None})
                self.rows.append(dict(self.rows[-1]))    # conclusive on its own
                self.done = True
                return None
            if self.phase_t < self.NUDGE_S:
                return (self.aim(self.course), self.byte)
            d = pos - self.start
            seen = float(np.linalg.norm(d))
            course = float(math.degrees(math.atan2(d[0], d[1])) % 360.0)
            self.rows.append({
                "commanded_deg": round(self.course, 1),
                "seen_cm": round(seen, 2),
                "seen_deg": None if seen < 1e-6 else round(course, 1),
            })
            self.phase, self.phase_t = "rest", 0.0
            return None

        if self.phase_t >= self.REST_S:
            self.leg += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.leg >= 2:
                self.done = True
        return None

    def result(self):
        out = {"nudges": self.rows}
        if len(self.rows) < 2:
            out["error"] = "the nudges did not complete"
            return out

        moved = [r["seen_cm"] for r in self.rows]
        if min(moved) < self.MIN_SEEN_CM:
            out["error"] = (
                f"nudged the robot twice and the camera saw it move "
                f"{min(moved):.1f}cm. It is not following this robot — most "
                "likely a phantom blob of the same colour. Check that the blob "
                "count matches the robot count and re-tune the colour.")
            return out

        # Movement of the right SIZE in the wrong DIRECTION is a different
        # fault: the tracker is following something real that is not this ball,
        # or two robots share a hue. Either way, do not drive on it.
        if any(r["seen_deg"] is None for r in self.rows):
            out["error"] = ("the camera did not see the robot move at all")
            return out
        errs = [abs(wrap180(r["seen_deg"] - r["commanded_deg"])) for r in self.rows]
        spread = abs(wrap180(errs[0] - errs[1]))
        out["offset_deg"] = round(float(circular_mean(
            [wrap180(r["seen_deg"] - r["commanded_deg"]) for r in self.rows]) or 0.0), 1)
        if spread > self.MAX_ANGLE_ERR:
            out["error"] = (
                f"the two nudges disagree by {spread:.0f}deg about which way the "
                "robot went. The camera is following something, but not "
                "consistently this robot.")
        return out


# -- 1. position noise -------------------------------------------------------

class NoiseProbe(Stage):
    """A parked ball, several hundred frames. The tracker's own error bar.

    Feeds two things that are currently assumed rather than known: the Kalman
    measurement covariance in `vision/track.py` (hardcoded r=4, i.e. 2cm), and
    the baseline length the heading estimator needs — its accuracy is sigma
    over distance, so sigma is half of that calculation.
    """

    label = "position noise"
    detail = "ball parked, camera watching"

    def __init__(self, frames=300, timeout=25.0):
        super().__init__(timeout=timeout)
        self.frames = int(frames)

    def _step(self, t, pos, dt):
        if len(self.samples) >= self.frames:
            self.done = True
        return None                      # never moves; that is the point

    @property
    def progress(self):
        return f"{len(self.samples)}/{self.frames} frames"

    def result(self):
        if len(self.samples) < 20:
            return {"error": "too few frames"}
        p = np.array([s[1] for s in self.samples])
        t = np.array([s[0] for s in self.samples])
        # Detrend before measuring scatter. A ball on a floor that is not quite
        # level creeps, and creep is not noise — counting it as noise would
        # inflate sigma and quietly loosen every gate downstream.
        tt = t - t[0]
        span = max(tt[-1], 1e-6)
        fits = [np.polyfit(tt, p[:, i], 1) for i in (0, 1)]
        resid = np.stack([p[:, i] - np.polyval(fits[i], tt) for i in (0, 1)], axis=1)
        drift = np.array([fits[i][0] * span for i in (0, 1)])
        sigma = float(np.linalg.norm(resid, axis=1).std())
        out_error = None
        if sigma < IMPLAUSIBLY_STILL_CM:
            out_error = (f"the tracked position moved {sigma:.3f}cm over "
                         f"{span:.0f}s, which no camera watching a real ball "
                         "does. The tracker is almost certainly locked onto "
                         "something that is not the robot — check for phantom "
                         "blobs and re-tune the colour before running anything "
                         "that drives.")
        return {
            "frames": len(self.samples),
            "seconds": round(float(span), 1),
            **({"error": out_error} if out_error else {}),
            "sigma_x_cm": round(float(resid[:, 0].std()), 3),
            "sigma_y_cm": round(float(resid[:, 1].std()), 3),
            "sigma_cm": round(float(np.linalg.norm(resid, axis=1).std()), 3),
            "p95_cm": round(float(np.percentile(np.linalg.norm(resid, axis=1), 95)), 3),
            "drift_cm": round(float(np.linalg.norm(drift)), 2),
            "fps": round(len(self.samples) / span, 1),
        }


def _window_speed(samples, t, span=0.30):
    """Speed and course over the last `span` seconds ending at `t`."""
    w = [s for s in samples if t - s[0] <= span]
    return _slope_speed(w)


def _axial_speed(samples, t, course, span=0.30):
    """Speed along `course`, signed. Negative means going the other way.

    Magnitude is the wrong quantity for anything that starts from a moving
    ball: a robot still rolling backwards from the previous leg reads as fast,
    not as not-yet-started, and every threshold crossing then fires at once.
    """
    w = [x for x in samples if t - x[0] <= span]
    speed, crs = _slope_speed(w)
    if speed is None:
        return None
    return speed * math.cos(math.radians(wrap180(crs - course)))


def _axial_series(samples, course, t_from, t_to, span=0.25):
    """[(t, axial cm/s)] over a window, timestamped at the CENTRE of each slope.

    Stamping a trailing slope with its end time makes every reading late by
    half the window, which lands directly in a latency measurement as spurious
    delay. Centring costs nothing and removes the bias.
    """
    pts = [x for x in samples if t_from - span <= x[0] <= t_to]
    out = []
    i = 0
    for j in range(len(pts)):
        while pts[j][0] - pts[i][0] > span:
            i += 1
        if j - i < 3:
            continue
        speed, crs = _slope_speed(pts[i:j + 1])
        if speed is None:
            continue
        mid = (pts[i][0] + pts[j][0]) / 2.0
        out.append((mid, speed * math.cos(math.radians(wrap180(crs - course)))))
    return out


def _at_rest(samples, t, span=0.5, tol=1.5):
    """Is the ball actually stopped? Half-window means, not a noisy slope.

    Waiting a fixed number of seconds is the tempting alternative and it is
    what made the first dry run useless: a step response that begins while the
    ball is still rolling backwards from the last leg measures nothing.
    """
    w = [x for x in samples if t - x[0] <= span]
    if len(w) < 8 or (w[-1][0] - w[0][0]) < span * 0.6:
        return False
    p = np.array([x[1] for x in w])
    half = len(p) // 2
    return float(np.linalg.norm(p[half:].mean(0) - p[:half].mean(0))) < tol


def _step_shape(t, delay, tau):
    """Distance covered per unit steady speed, from rest, after a dead time."""
    x = np.maximum(t - delay, 0.0)
    return np.where(t < delay, 0.0, x - tau * (1.0 - np.exp(-x / tau)))


def fit_step(t, y, taus, delays):
    """Best (delay, tau, v_ss) for a position trace of a robot starting from rest.

    Fitted to POSITION rather than read off crossings of a differentiated
    speed, for the same reason the reversal is: differentiating a camera trace
    costs about 4cm/s of noise, and at a capped 18cm/s the 10 per cent crossing
    sits underneath that. The crossing estimator did not fail loudly when asked
    — it returned 0.37 for true values of 0.25, 0.35 and 0.47 alike, which is
    the worst way for a measurement to be wrong.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    best = (None, None, None, float("inf"))
    for tau in taus:
        for d in delays:
            f = _step_shape(t, d, tau)
            A = np.stack([np.ones_like(f), f], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            if coef[1] <= 0:
                continue
            sse = float(((y - A @ coef) ** 2).sum())
            if sse < best[3]:
                best = (float(d), float(tau), float(coef[1]), sse)
    return best


def _reversal_shape(t, delay, tau):
    """Axial displacement per unit entry speed, for a reversal commanded at 0.

    Runs on at the entry speed until `delay`, then the motor reverses through a
    first-order lag. Integrating that is the whole model, and having it in
    closed form is what turns latency from a threshold-hunting exercise into
    ordinary least squares.
    """
    out = np.where(t < delay, t, 0.0).astype(float)
    m = t >= delay
    x = t[m] - delay
    out[m] = delay - x + 2.0 * tau * (1.0 - np.exp(-x / tau))
    return out


def fit_reversal(t, y, taus, delays):
    """Best (delay, tau) for an axial position trace. Linear inner solve.

    Fitting the shape to POSITION rather than hunting a corner in a
    differentiated speed: position is what the camera measures, so it carries
    no differentiation noise, and the model has exactly one nonlinear parameter
    worth scanning. Threshold crossings on a smoothed speed trace were out by
    100ms and drifted with the delay they were measuring; this is out by 20ms
    and does not.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    best = (None, None, float("inf"))
    for tau in taus:
        for d in delays:
            f = _reversal_shape(t, d, tau)
            A = np.stack([np.ones_like(f), f], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            if coef[1] <= 0:
                continue                    # entry speed must be positive
            sse = float(((y - A @ coef) ** 2).sum())
            if sse < best[2]:
                best = (float(d), float(tau), sse)
    return best


def _breakpoint(series, post_n=8, min_t=None):
    """Where a flat line turns into a decline. Returns the time, or None.

    A threshold crossing needs a threshold, and on a differentiated camera
    signal the noise is large enough that any threshold safely above it sits
    well down the ramp — so the answer comes back biased by however long the
    ball took to lose that much speed. Fitting flat-then-falling uses every
    sample and puts the estimate at the corner itself.
    """
    if len(series) < 10:
        return None
    t = np.array([x[0] for x in series], dtype=float)
    v = np.array([x[1] for x in series], dtype=float)
    best, best_sse = None, float("inf")
    for k in range(4, len(series) - 4):
        # The reaction cannot precede the command. Without this the flat
        # stretch BEFORE the reversal is itself a candidate corner, and the
        # fit happily reports a ball that turned round before it was told to.
        if min_t is not None and t[k] < min_t:
            continue
        # Only the first stretch of the decline. A ball slowing down does it
        # exponentially, so a straight line through the whole tail sits above
        # the curve early and below it late, and the corner that best absorbs
        # that mismatch is not the corner that happened.
        end = min(len(series), k + max(5, int(post_n)))
        if end - k < 4:
            break
        pre = v[:k]
        a, b = np.polyfit(t[k:end], v[k:end], 1)
        if a >= 0:
            continue                       # must actually be slowing down
        sse = float(((pre - pre.mean()) ** 2).sum()
                    + ((v[k:end] - (a * t[k:end] + b)) ** 2).sum() * (len(pre) / (end - k)))
        if sse < best_sse:
            best_sse, best = sse, float(t[k])
    return best


def clear_heading(ws, pos, reach, prefer=None):
    """A camera-frame course with `reach` cm of clear floor ahead of `pos`.

    Preference first, then its opposite — a leg driven backwards measures
    exactly the same thing — and only then a sweep. Trying the opposite before
    sweeping keeps a sequence of legs on one axis, which keeps the ball near
    where it started instead of walking it into a corner over ten runs.
    """
    def ok(course):
        if ws is None:
            return True
        rad = math.radians(course)
        end = (pos[0] + math.sin(rad) * reach, pos[1] + math.cos(rad) * reach)
        try:
            return ws.is_valid_point(end) and ws.has_clearance(end, 12.0)
        except Exception:
            return True

    order = []
    if prefer is not None:
        order += [float(prefer) % 360.0, (float(prefer) + 180.0) % 360.0]
    order += [h * 15.0 for h in range(24)]
    for course in order:
        if ok(course):
            return course
    return None


# -- 2. speed map ------------------------------------------------------------

def spread_bytes(top, count=10, floor=18):
    """`count` speed bytes from just above a plausible deadband up to `top`.

    Starting at 18 rather than at zero because nothing moves down there on any
    ball yet measured, and a leg spent confirming that a motor does not turn is
    a leg not spent on the part of the curve that matters.
    """
    top = int(top)
    if top <= floor:
        return [top]
    step = (top - floor) / float(max(count - 1, 1))
    out = sorted({int(round(floor + i * step)) for i in range(count)})
    return [b for b in out if b <= top]


class SpeedMap(Stage):
    """Speed byte in, cm/s out. The map `MAX_SPEED` currently asserts blind.

    `fleet/handle.py` converts cm/s to a byte by assuming 255 means 60cm/s and
    that the relationship is a straight line through the origin. Neither half
    of that has been checked. A real ball has a deadband — below some byte the
    motors do not overcome static friction — so the line does not pass through
    the origin, and a controller asking for 4cm/s near its target is asking for
    nothing at all.
    """

    label = "speed map"
    detail = "stepping the speed byte, measuring cm/s"

    BYTES = (20, 40, 60, 80, 110, 140, 170, 200, 230, 255)
    POINTS = 10             # samples across whatever range is allowed

    def __init__(self, bytes_=None, settle=1.1, window=1.4, rest=0.7,
                 workspace=None, aim=None, timeout=140.0, points=None):
        super().__init__(timeout=timeout)
        self.explicit_bytes = bytes_ is not None
        self.points = int(points or self.POINTS)
        self.bytes = list(bytes_ or self.BYTES)
        self.settle, self.window, self.rest = settle, window, rest
        self.ws = workspace
        self.aim = aim or (lambda c: c)
        self.i = 0
        self.phase = "aim"
        self.phase_t = 0.0
        self.course = None
        self.win = []
        self.rows = []

    @property
    def progress(self):
        return f"byte {self.bytes[min(self.i, len(self.bytes) - 1)]} "\
               f"({self.i + 1}/{len(self.bytes)})"

    def _step(self, t, pos, dt):
        if self.i >= len(self.bytes):
            self.done = True
            return None
        byte = self.bytes[self.i]
        if byte > self.max_byte:
            # Skipped, not clamped. Clamping every byte above the cap to the
            # cap yields several identical speeds wearing different byte
            # labels, and a straight line fitted through those has the wrong
            # slope and a confident r-squared to go with it.
            self.rows.append({"byte": byte, "cm_s": None, "course_deg": None,
                              "requested_deg": 0.0,
                              "skipped": f"above the {self.max_byte} byte cap"})
            self.i += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.i >= len(self.bytes):
                self.done = True
            return None
        self.phase_t += dt

        want = MIN_SETTLE_S + MIN_WINDOW_S
        if self.phase == "aim":
            # Whichever way has the most floor, preferring to alternate so the
            # sweep ends near where it started.
            prefer = 0.0 if self.course is None else (self.course + 180.0) % 360.0
            options = [prefer, (prefer + 180.0) % 360.0] + [h * 30.0 for h in range(12)]
            self.course = max(options,
                              key=lambda c: self.runway(pos, c, byte, want)[0])
            if self.runway(pos, self.course, byte, want)[0] <= 0.0:
                self.error = ("no clear leg in any direction — move the ball "
                              "toward the middle of the arena")
                self.done = True
                return None
            self.phase, self.phase_t = "backup", 0.0
            return None

        if self.phase == "backup":
            cmd = self.backup_command(pos, self.course, byte, want)
            if cmd is not None and self.phase_t < 12.0:
                return cmd
            if not self.rest_after_backup(t) and self.phase_t < 15.0:
                return None                 # let the crawl die out first
            # Fit the leg to whatever runway there turned out to be, but never
            # below what it takes to reach a steady speed.
            have, _ = self.runway(pos, self.course, byte, want)
            self.leg_settle = min(self.settle, max(MIN_SETTLE_S, have * 0.5))
            self.leg_window = min(self.window, have - self.leg_settle)
            if have < MIN_SETTLE_S + MIN_WINDOW_S or self.leg_window < MIN_WINDOW_S:
                self.rows.append({"byte": byte, "cm_s": None, "course_deg": None,
                                  "requested_deg": round(self.course, 1),
                                  "skipped": f"only {have:.1f}s of floor; a leg "
                                             "this fast needs more room"})
                self.phase, self.phase_t = "rest", 0.0
                return None
            self.phase, self.phase_t, self.win = "settle", 0.0, []
            return (self.aim(self.course), byte)

        if self.phase == "settle":
            if self.phase_t >= self.leg_settle:
                self.phase, self.phase_t, self.win = "measure", 0.0, []
            return (self.aim(self.course), byte)

        if self.phase == "measure":
            self.win.append((t, pos.copy()))
            if self.phase_t >= self.leg_window:
                speed, course = _slope_speed(self.win)
                self.rows.append({
                    "byte": byte,
                    "cm_s": None if speed is None else round(speed, 2),
                    "course_deg": None if course is None else round(course, 1),
                    "requested_deg": round(self.course, 1),
                })
                self.phase, self.phase_t = "rest", 0.0
                return None
            return (self.aim(self.course), byte)

        # rest — let the ball come to a stop before the next step
        if self.phase_t >= self.rest:
            self.i += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.i >= len(self.bytes):
                self.done = True
        return None

    def result(self):
        rows = [r for r in self.rows if r.get("cm_s") is not None]
        skipped = [r["byte"] for r in self.rows if r.get("skipped")]
        if len(rows) < 3:
            return {"rows": self.rows, "skipped_bytes": skipped,
                    "error": "too few usable legs — a bigger arena, or move the "
                             "ball to the middle before starting"}
        b = np.array([r["byte"] for r in rows], dtype=float)
        v = np.array([r["cm_s"] for r in rows], dtype=float)

        # Fit only where the ball is actually rolling. Including the stalled
        # low bytes drags the line down and hides the deadband inside it, which
        # is the one feature of this curve a controller must know about.
        moving = v > STOP_SPEED
        out = {"rows": self.rows, "moving_points": int(moving.sum()),
               "skipped_bytes": skipped}
        if moving.sum() >= 3:
            k, c = np.polyfit(b[moving], v[moving], 1)
            top = int(b[moving].max())
            projected = float(k * 255 + c)
            # Trusted only if it was nearly measured, and only if it is a speed
            # a Sphero can reach. Everything downstream — the controller's
            # ceiling, the deadband in cm/s, the arrival radius — is derived
            # from this one number, so letting an untrustworthy one through
            # does not produce one bad figure, it produces a bad calibration.
            trusted = bool(top >= TRUSTWORTHY_TOP_BYTE
                           and 0 < projected <= PHYSICAL_MAX_CM_S)
            out.update(
                cm_s_per_byte=round(float(k), 4),
                intercept_cm_s=round(float(c), 2),
                deadband_byte=int(max(0, round(-c / k))) if k > 1e-6 else None,
                highest_measured_byte=top,
                max_speed_measured=bool(top >= 240),
                max_speed_trusted=trusted,
                max_speed_cm_s=round(projected, 1),
                r2=round(float(1 - np.var(v[moving] - (k * b[moving] + c))
                               / max(np.var(v[moving]), 1e-9)), 4),
            )
        stalled = [int(r["byte"]) for r in rows if r["cm_s"] <= STOP_SPEED]
        out["stalled_bytes"] = stalled
        out["min_moving_byte"] = min((int(r["byte"]) for r in rows
                                      if r["cm_s"] > STOP_SPEED), default=None)
        return out


# -- 3. step response --------------------------------------------------------

class StepResponse(Stage):
    """Rest to steady, repeated. Yields the dead time and the lag constant.

    `SimRobot` guesses tau between 0.25s and 0.5s and latency between one and
    three control steps. Both are here, measured together, because from the
    camera's side they are not separable and a controller does not care which
    is which — what it feels is the total delay before the ball responds and
    the shape of the ramp afterwards.
    """

    label = "step response"
    detail = "rest to full speed, three times"

    def __init__(self, byte=170, repeats=3, drive=3.5, rest=1.6,
                 workspace=None, aim=None, timeout=60.0):
        super().__init__(timeout=timeout)
        self.byte, self.repeats = int(byte), int(repeats)
        self.drive, self.rest = drive, rest
        self.ws, self.aim = workspace, aim or (lambda c: c)
        self.rep = 0
        self.phase, self.phase_t = "aim", 0.0
        self.course = None
        self.trace = []             # (t_since_command, speed) for this rep
        self.fits = []

    @property
    def progress(self):
        return f"run {min(self.rep + 1, self.repeats)}/{self.repeats}"

    def _step(self, t, pos, dt):
        if self.rep >= self.repeats:
            self.done = True
            return None
        self.phase_t += dt

        if self.phase == "aim":
            prefer = 0.0 if self.course is None else (self.course + 180.0) % 360.0
            options = [prefer, (prefer + 180.0) % 360.0] + [h * 30.0 for h in range(12)]
            self.course = max(options, key=lambda c:
                              self.runway(pos, c, self.byte, self.drive)[0])
            have, _ = self.runway(pos, self.course, self.byte, self.drive)
            if have <= 0.4:
                self.error = "no clear leg for a step response"
                self.done = True
                return None
            # As long as the floor allows, up to a generous ceiling: tau is
            # read off the shape of the ramp, and a ramp cut short at one time
            # constant has barely started to bend.
            self.leg_drive = max(1.0, min(self.drive, have))
            self.phase, self.phase_t = "wait", 0.0
            return None

        if self.phase == "wait":
            # A step response has to start from a standstill or the ramp it
            # measures is the tail of the previous leg.
            if _at_rest(self.samples, t) or self.phase_t > 4.0:
                self.phase, self.phase_t, self.trace = "drive", 0.0, []
                self.t0 = t
                return (self.aim(self.course), self.byte)
            return None

        if self.phase == "drive":
            if self.phase_t >= self.leg_drive:
                rad = math.radians(self.course)
                axis = np.array([math.sin(rad), math.cos(rad)])
                self.leg_pos = [(x[0] - self.t0, float(np.dot(x[1], axis)))
                                for x in self.samples if x[0] >= self.t0 - 0.2]
                self._fit()
                self.phase, self.phase_t = "rest", 0.0
                return None
            return (self.aim(self.course), self.byte)

        if self.phase_t >= self.rest:
            self.rep += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.rep >= self.repeats:
                self.done = True
        return None

    def _fit(self):
        """Fit the ramp's shape to the position it produced.

        The obvious estimator reads the 10% and 63% crossings off a
        differentiated speed trace, and it works well at speed. Under a low
        speed cap it does not work at all: differentiation costs several cm/s
        of noise and the 10% crossing of an 18cm/s ramp is beneath it, so the
        answer stops depending on the input. Position carries the same
        information with none of the noise.
        """
        if not getattr(self, "leg_pos", None) or len(self.leg_pos) < 20:
            return
        t = np.array([x[0] for x in self.leg_pos], dtype=float)
        y = np.array([x[1] for x in self.leg_pos], dtype=float)
        delay, tau, v_ss, sse = fit_step(
            t, y, np.arange(0.10, 0.85, 0.02), np.arange(0.0, 0.45, 1.0 / 60.0))
        if tau is None or v_ss is None or v_ss < STOP_SPEED:
            return
        self.fits.append({"dead_s": round(delay, 3), "tau_s": round(tau, 3),
                          "v_ss_cm_s": round(v_ss, 2),
                          "rms_cm": round(float(np.sqrt(sse / max(len(t), 1))), 2)})

    def result(self):
        if not self.fits:
            return {"error": "no usable step; did the ball move?", "runs": []}
        return {
            "runs": self.fits,
            "byte": self.byte,
            "dead_s": round(float(np.median([f["dead_s"] for f in self.fits])), 3),
            "tau_s": round(float(np.median([f["tau_s"] for f in self.fits])), 3),
            "v_ss_cm_s": round(float(np.median([f["v_ss_cm_s"] for f in self.fits])), 2),
        }


# -- 4. brake test -----------------------------------------------------------

class BrakeTest(Stage):
    """Steady, then zero. How far it keeps going — the speed limit, directly.

    A controller that commands a stop inside its arrival tolerance has already
    lost if the coast is longer than the tolerance: the ball sails through the
    target and has to be brought back, which reads as hunting. Measured per
    speed, because there is no reason to assume the relationship is linear —
    a ball that skids at high speed and grips at low speed will not be.
    """

    label = "brake test"
    detail = "full speed to a dead stop, per speed"

    # Four speeds, twice each: the coast constant is a slope through them, and
    # a slope through four noisy points is not worth much. Repeats cost time
    # and nothing else.
    BYTES = (80, 140, 200, 255, 80, 140, 200, 255)

    def __init__(self, bytes_=None, settle=1.6, max_coast=4.0, rest=1.0,
                 workspace=None, aim=None, timeout=110.0):
        super().__init__(timeout=timeout)
        self.bytes = list(bytes_ or self.BYTES)
        self.settle, self.max_coast, self.rest = settle, max_coast, rest
        self.ws, self.aim = workspace, aim or (lambda c: c)
        self.i = 0
        self.phase, self.phase_t = "aim", 0.0
        self.course = None
        self.cut_pos = None
        self.entry = None
        self.rows = []

    @property
    def progress(self):
        return f"byte {self.bytes[min(self.i, len(self.bytes) - 1)]} "\
               f"({self.i + 1}/{len(self.bytes)})"

    def _step(self, t, pos, dt):
        if self.i >= len(self.bytes):
            self.done = True
            return None
        byte = self.bytes[self.i]
        if byte > self.max_byte:
            self.rows.append({"byte": byte, "entry_cm_s": None, "coast_cm": None,
                              "stop_s": None, "settled": False,
                              "skipped": f"above the {self.max_byte} byte cap"})
            self.i += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.i >= len(self.bytes):
                self.done = True
            return None
        self.phase_t += dt

        if self.phase == "aim":
            prefer = 0.0 if self.course is None else (self.course + 180.0) % 360.0
            options = [prefer, (prefer + 180.0) % 360.0] + [h * 30.0 for h in range(12)]
            self.course = max(options, key=lambda c:
                              self.runway(pos, c, byte, self.settle)[0])
            if self.runway(pos, self.course, byte, self.settle)[0] <= 0.0:
                self.error = "no clear leg long enough to brake in"
                self.done = True
                return None
            self.phase, self.phase_t = "backup", 0.0
            return None

        if self.phase == "backup":
            cmd = self.backup_command(pos, self.course, byte, self.settle)
            if cmd is not None and self.phase_t < 12.0:
                return cmd
            if not self.rest_after_backup(t) and self.phase_t < 15.0:
                return None                 # let the crawl die out first
            have, _ = self.runway(pos, self.course, byte, self.settle)
            # A brake test needs the ball at a steady speed and nothing more,
            # so a short run-up is fine — but it must be long enough to reach
            # that speed, or what is measured is a stop from halfway.
            self.leg_settle = max(0.7, min(self.settle, have))
            if have < 0.7:
                self.rows.append({"byte": byte, "entry_cm_s": None,
                                  "coast_cm": None, "stop_s": None,
                                  "settled": False,
                                  "skipped": "no run-up at this speed"})
                self.phase, self.phase_t = "rest", 0.0
                return None
            self.phase, self.phase_t = "settle", 0.0
            return (self.aim(self.course), byte)

        if self.phase == "settle":
            if self.phase_t >= self.leg_settle:
                self.entry, _ = _window_speed(self.samples, t, span=0.4)
                self.cut_pos = pos.copy()
                self.cut_t = t
                self.phase, self.phase_t = "coast", 0.0
                return None                      # the cut
            return (self.aim(self.course), byte)

        if self.phase == "coast":
            # Displacement over a window, not a differentiated speed: at 2cm/s
            # the slope of a noisy position trace is mostly noise, so a speed
            # threshold fires at a random moment during the roll-out.
            stopped = self.phase_t > 0.5 and _at_rest(self.samples, t, span=0.6)
            if stopped or self.phase_t >= self.max_coast:
                self.rows.append({
                    "byte": byte,
                    "entry_cm_s": None if self.entry is None else round(self.entry, 2),
                    "coast_cm": round(float(np.linalg.norm(pos - self.cut_pos)), 2),
                    "stop_s": round(float(t - self.cut_t), 3),
                    "settled": bool(stopped),
                })
                self.phase, self.phase_t = "rest", 0.0
            return None

        if self.phase_t >= self.rest:
            self.i += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.i >= len(self.bytes):
                self.done = True
        return None

    def result(self):
        rows = [r for r in self.rows
                if r["entry_cm_s"] and r["entry_cm_s"] > STOP_SPEED]
        out = {"rows": self.rows}
        if len(rows) >= 2:
            v = np.array([r["entry_cm_s"] for r in rows])
            d = np.array([r["coast_cm"] for r in rows])
            # Coast per unit speed is the equivalent time constant of the stop,
            # and it is what converts a chosen arrival tolerance into a speed
            # limit. Forced through the origin: a ball at rest coasts nowhere.
            k = float((v * d).sum() / max((v * v).sum(), 1e-9))
            spread = float((v.max() - v.min()) / max(v.max(), 1e-9))
            out.update(
                coast_s_per_cm_s=round(k, 4),
                coast_at_45_cm=round(k * 45.0, 1),
                r2=round(float(1 - np.var(d - k * v) / max(np.var(d), 1e-9)), 4),
                entry_speed_spread=round(spread, 3),
                # Everything that limits speed descends from this one number —
                # the arrival tolerance, and the ceiling `fleet/safety.py`
                # enforces on every robot. An untrustworthy one does not
                # produce a single bad figure, it produces a bad calibration.
                coast_trusted=bool(len(rows) >= BRAKE_MIN_ROWS
                                   and spread >= BRAKE_MIN_SPREAD),
            )
        return out


# -- 5. loop latency ---------------------------------------------------------

class LatencyProbe(Stage):
    """Command a reversal, watch for the camera to notice. The whole round trip.

    BLE write, motor spin-up, exposure, detection, Kalman — one number covering
    all of it, because that is the one the control loop actually experiences.
    Nothing else in the codebase measures it, and every gain depends on it: a
    proportional controller is stable when its gain times the total delay stays
    small, so a delay nobody has measured is a gain nobody can justify.

    A reversal rather than a start, because the signal is twice as large and
    starts from a steady state whose noise can be measured and used as the
    detection threshold.
    """

    label = "loop latency"
    detail = "reverse at speed, time the camera's reaction"

    def __init__(self, byte=200, repeats=3, settle=1.8, watch=1.4, rest=1.4,
                 workspace=None, aim=None, timeout=70.0):
        super().__init__(timeout=timeout)
        self.byte, self.repeats = int(byte), int(repeats)
        self.settle, self.watch, self.rest = settle, watch, rest
        self.ws, self.aim = workspace, aim or (lambda c: c)
        self.rep = 0
        self.phase, self.phase_t = "aim", 0.0
        self.course = None
        self.rows = []
        self.pre = 0.9              # seconds of steady run kept before the cut
        self.tau_hint = None        # filled in from the step response

    @property
    def progress(self):
        return f"run {min(self.rep + 1, self.repeats)}/{self.repeats}"

    def _step(self, t, pos, dt):
        if self.rep >= self.repeats:
            self.done = True
            return None
        self.phase_t += dt

        if self.phase == "aim":
            prefer = 0.0 if self.course is None else (self.course + 180.0) % 360.0
            options = [prefer, (prefer + 180.0) % 360.0] + [h * 30.0 for h in range(12)]
            self.course = max(options, key=lambda c:
                              self.runway(pos, c, self.byte, self.settle)[0])
            if self.runway(pos, self.course, self.byte, self.settle)[0] <= 0.0:
                self.error = "no clear leg to reverse in"
                self.done = True
                return None
            self.phase, self.phase_t = "backup", 0.0
            return None

        if self.phase == "backup":
            cmd = self.backup_command(pos, self.course, self.byte, self.settle)
            if cmd is not None and self.phase_t < 12.0:
                return cmd
            if not self.rest_after_backup(t) and self.phase_t < 15.0:
                return None                 # let the crawl die out first
            have, _ = self.runway(pos, self.course, self.byte, self.settle)
            self.leg_settle = max(0.9, min(self.settle, have))
            self.phase, self.phase_t = "settle", 0.0
            return (self.aim(self.course), self.byte)

        if self.phase == "settle":
            if self.phase_t >= self.leg_settle:
                self.baseline = self._axis_series(t - 0.9, t)
                self.phase, self.phase_t = "reverse", 0.0
                self.t0 = t
                return (self.aim((self.course + 180.0) % 360.0), self.byte)
            return (self.aim(self.course), self.byte)

        if self.phase == "reverse":
            if self.phase_t >= self.watch:
                self._measure(t)
                self.phase, self.phase_t = "rest", 0.0
                return None
            return (self.aim((self.course + 180.0) % 360.0), self.byte)

        if self.phase_t >= self.rest:
            self.rep += 1
            self.phase, self.phase_t = "aim", 0.0
            if self.rep >= self.repeats:
                self.done = True
        return None

    def _axis_series(self, t_from, t_to):
        return _axial_series(self.samples, self.course, t_from, t_to)

    def _axial_position(self, t_from, t_to):
        rad = math.radians(self.course)
        axis = np.array([math.sin(rad), math.cos(rad)])
        pts = [(x[0] - self.t0, float(np.dot(x[1], axis)))
               for x in self.samples if t_from <= x[0] <= t_to]
        return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))

    def _measure(self, t):
        base = getattr(self, "baseline", [])
        if len(base) < 5:
            return
        vals = np.array([b[1] for b in base])
        steady, wobble = float(np.median(vals)), float(vals.std())
        if steady < STOP_SPEED:
            return
        tt, yy = self._axial_position(self.t0 - self.pre, t)
        if len(tt) < 20:
            return
        # tau from the step response when we have it. Solving for one unknown
        # instead of two roughly halves the scatter, and the step response has
        # already run by the time this stage does.
        taus = ([self.tau_hint] if self.tau_hint
                else list(np.arange(0.15, 0.62, 0.04)))
        delay, tau, sse = fit_reversal(tt, yy, taus,
                                       np.arange(0.0, 0.62, 1.0 / 90.0))
        if delay is None:
            return
        self.rows.append({
            "delay_s": round(delay, 3),
            "tau_s": round(tau, 3),
            "steady_cm_s": round(steady, 2),
            "wobble_cm_s": round(wobble, 2),
            "rms_cm": round(float(np.sqrt(sse / max(len(tt), 1))), 2),
        })

    def result(self):
        if not self.rows:
            return {"runs": [], "error": "no reversal was detected"}
        d = [r["delay_s"] for r in self.rows]
        return {"runs": self.rows,
                "loop_delay_s": round(float(np.median(d)), 3),
                "spread_s": round(float(max(d) - min(d)), 3)}


# -- 6. heading drift --------------------------------------------------------

class DriftWatch(Stage):
    """How fast does the aim frame wander? Minutes, not seconds.

    The one measurement that decides whether a continuously estimated heading
    is worth building. A ball that drifts a degree a minute can be calibrated
    once and forgotten; one that drifts twenty cannot, and the difference is
    invisible from anything shorter than a few minutes of running.

    Short legs, back and forth, continuously — rather than one leg every so
    often with the ball parked between. Two reasons: it yields a hundred
    samples instead of six, and a rolling ball is the condition the number is
    wanted for. A gyro sitting still and a gyro being shaken do not drift the
    same way.
    """

    label = "heading drift"
    detail = "short legs back and forth, watching the frame wander"

    def __init__(self, minutes=5.0, leg_cm=24.0, speed_byte=70, rest=0.6,
                 workspace=None, aim=None):
        super().__init__(timeout=minutes * 60.0 + 30.0)
        self.window = float(minutes) * 60.0
        self.leg_cm = float(leg_cm)
        self.byte = int(speed_byte)
        self.rest = float(rest)
        self.ws, self.aim = workspace, aim or (lambda c: c)
        self.phase, self.phase_t = "aim", 0.0
        self.course = None
        self.start_pos = None
        self.leg_t0 = 0.0
        self.rows = []

    @property
    def progress(self):
        return f"{self.elapsed:.0f}/{self.window:.0f}s, {len(self.rows)} legs"

    def _step(self, t, pos, dt):
        if self.elapsed >= self.window:
            self.done = True
            return None
        self.phase_t += dt

        if self.phase == "aim":
            prefer = 0.0 if self.course is None else (self.course + 180.0) % 360.0
            self.course = clear_heading(self.ws, pos, self.leg_cm + SETTLE_MARGIN,
                                        prefer)
            if self.course is None:
                self.error = "ran out of clear floor part-way through"
                self.done = True
                return None
            self.start_pos, self.leg_t0 = pos.copy(), t
            self.phase, self.phase_t = "leg", 0.0
            return (self.aim(self.course), self.byte)

        if self.phase == "leg":
            travelled = float(np.linalg.norm(pos - self.start_pos))
            if travelled >= self.leg_cm:
                d = pos - self.start_pos
                course = float(math.degrees(math.atan2(d[0], d[1])) % 360.0)
                self.rows.append({
                    "t": round(float((t + self.leg_t0) / 2.0), 2),
                    "residual_deg": round(wrap180(course - self.course), 2),
                    "cm": round(travelled, 1),
                })
                self.phase, self.phase_t = "rest", 0.0
                return None
            if self.phase_t > 8.0:
                self.phase, self.phase_t = "rest", 0.0   # stuck; try elsewhere
                return None
            return (self.aim(self.course), self.byte)

        if self.phase_t >= self.rest:
            self.phase, self.phase_t = "aim", 0.0
        return None

    def result(self):
        if len(self.rows) < 6:
            return {"rows": self.rows, "error": "too few legs to see a trend"}
        t = np.array([r["t"] for r in self.rows], dtype=float)
        y = np.array([r["residual_deg"] for r in self.rows], dtype=float)
        t = t - t[0]
        # Unwrapped, or a walk that crosses +-180 reads as a jump the size of a
        # full turn and the slope comes back meaningless.
        y = np.degrees(np.unwrap(np.radians(y)))
        slope, intercept = np.polyfit(t, y, 1)
        resid = y - (slope * t + intercept)

        span = float(t[-1]) / 60.0
        drift = float(slope) * 60.0
        # Is the trend real, or is it noise with a line drawn through it? The
        # standard error of the slope answers that, and a drift smaller than
        # its own error bar is not a drift.
        se = (float(resid.std(ddof=2)) /
              max(float(np.sqrt(((t - t.mean()) ** 2).sum())), 1e-9)) * 60.0
        return {
            "rows": self.rows,
            "legs": len(self.rows),
            "minutes": round(span, 2),
            "drift_deg_per_min": round(drift, 2),
            "drift_stderr_deg_per_min": round(se, 2),
            "significant": bool(abs(drift) > 2.0 * se),
            "leg_scatter_deg": round(float(resid.std(ddof=1)), 2),
            "total_excursion_deg": round(float(y.max() - y.min()), 1),
            "start_offset_deg": round(float(y[0]), 1),
            "end_offset_deg": round(float(y[-1]), 1),
            **self._random_walk(t, y),
        }

    @staticmethod
    def _random_walk(t, y, window_s=20.0):
        """Degrees per root-minute — the statistic a wandering gyro deserves.

        A drifting heading is a random walk, not a ramp, and fitting a line to
        one finds a slope of nothing with a large error bar however badly the
        ball is wandering. What a random walk has instead is a rate at which
        variance accumulates: the offset moves about sigma * sqrt(minutes), so
        sigma is the number to quote and `sigma * sqrt(30)` answers "how far
        will this be out by the end of a session".

        Binned first, because each leg's residual carries its own measurement
        noise, and differencing raw legs would report that noise as drift.
        Averaging a window down and differencing the windows leaves the walk.
        """
        if t[-1] < window_s * 3:
            return {"random_walk_deg_per_sqrt_min": None}
        edges = np.arange(0.0, float(t[-1]) + window_s, window_s)
        means, centres = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t < hi)
            if m.sum() >= 2:
                means.append(float(y[m].mean()))
                centres.append((lo + hi) / 2.0)
        if len(means) < 4:
            return {"random_walk_deg_per_sqrt_min": None}
        diffs = np.diff(np.array(means))
        # Var of a step over `window_s` is sigma^2 * window, in minutes.
        sigma = float(np.sqrt((diffs ** 2).mean() / (window_s / 60.0)))
        # Each leg's own measurement noise leaks through the binning and puts
        # a floor under this — measured at about 2 deg/sqrt(min) with 1cm of
        # position noise and 24cm legs. Below the floor the honest answer is
        # "too small to see", not a number.
        floor = 2.0
        return {
            "random_walk_deg_per_sqrt_min": round(sigma, 2),
            "expected_wander_30min_deg": round(sigma * math.sqrt(30.0), 1),
            "above_noise_floor": bool(sigma > floor * 1.3),
            "noise_floor_deg_per_sqrt_min": floor,
            "windows": len(means),
        }


# -- housekeeping stages -----------------------------------------------------

class HeadingStage(Stage):
    """Wraps `ActiveCalibration` so the sequence can aim before it measures."""

    label = "heading offset"
    detail = "four legs, reading the aim-frame offset off them"

    def __init__(self, workspace=None, timeout=70.0, settle=1.8):
        super().__init__(timeout=timeout)
        self.cal = ActiveCalibration(workspace=workspace)
        self.settle = float(settle)
        self._leg = 0
        self._pause = 0.0

    @property
    def progress(self):
        return self.cal.progress

    def _step(self, t, pos, dt):
        # ActiveCalibration anchors the next leg on the first frame after the
        # last one ended, and at that moment the ball is still coasting in the
        # OLD direction. The leg it then measures is the vector sum of the two,
        # which is why four legs that should have agreed came back 13 degrees
        # apart. Holding a stop between legs costs a second and removes it.
        if self.cal.leg != self._leg:
            self._leg = self.cal.leg
            self._pause = self.settle
        if self._pause > 0:
            self._pause -= dt
            if not _at_rest(self.samples, t) and self._pause <= 0:
                self._pause = 0.3          # still rolling; give it a little more
            return None

        v = self.cal.step(pos, dt)
        if v is None:
            self.done = True
            self.error = self.cal.error
            return None
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return None
        # ActiveCalibration works in velocity vectors; the raw path speaks
        # heading and byte. Converting here keeps one command type in the loop.
        course = float(math.degrees(math.atan2(v[0], v[1])) % 360.0)
        return (course, int(np.clip(speed / 60.0 * 255.0, 0, 255)))

    def result(self):
        return self.cal.state()


class Recenter(Stage):
    """Closed-loop drive back to a point, so the next stage has room.

    The alternative is asking the trainer to pick the ball up between tests,
    which is both the slowest part of a session and the easiest to do slightly
    differently each time.
    """

    label = "recentre"
    detail = "driving back to open floor"

    def __init__(self, target, tol=14.0, speed_byte=110, timeout=30.0, aim=None):
        super().__init__(timeout=timeout)
        self.target = np.asarray(target, dtype=float)
        self.tol = float(tol)
        self.byte = int(speed_byte)
        self.aim = aim or (lambda c: c)

    @property
    def progress(self):
        if not self.samples:
            return ""
        return f"{float(np.linalg.norm(self.target - self.samples[-1][1])):.0f}cm to go"

    def _step(self, t, pos, dt):
        to = self.target - pos
        d = float(np.linalg.norm(to))
        if d <= self.tol:
            self.done = True
            return None
        course = float(math.degrees(math.atan2(to[0], to[1])) % 360.0)
        # Ease down close in, or it circles the point instead of landing on it.
        byte = self.byte if d > 40.0 else max(60, int(self.byte * d / 40.0))
        return (self.aim(course), byte)

    def result(self):
        return {"arrived": self.done and not self.error}


# -- the sequence ------------------------------------------------------------

ARRIVE_TOL_CM = 6.0         # mirrors swarm.navigate.ARRIVE_TOL
DAMPING_TARGET = 0.40       # gain * (tau + delay); above ~0.5 it rings

# What the brake fit needs before it is allowed to stand behind its own slope.
# The same discipline `max_speed_trusted` applies to the speed map, which was
# added after one unbounded extrapolation poisoned an entire calibration: a
# slope is only worth publishing if it was measured across a range worth
# calling a range. Two points 14% apart, fitted through the origin, is one
# point and an assumption.
BRAKE_MIN_ROWS = 3
BRAKE_MIN_SPREAD = 0.35     # (max - min) / max of the entry speeds

# A slow radius past this fraction of the arena's short side means the robot is
# easing in almost everywhere it can stand, which is a recommendation the room
# cannot honour however correct its arithmetic.
SLOW_RADIUS_ARENA_FRACTION = 0.5


class Characterization:
    """The whole battery, start to finish, on one robot.

    Trainer input is: put the ball on the floor, connect it, press go. Every
    stage aims itself at clear floor, reverses when it runs out, and the ball is
    driven back to the middle between stages — because the alternative is a
    session where half the legs fail against a wall and the trainer spends it
    carrying a ball back to the centre.
    """

    def __init__(self, workspace=None, code=None, ble_name=None,
                 heading_offset=0.0, quick=False, max_byte=255,
                 plan_safety=PLAN_SAFETY):
        self.ws = workspace
        self.code = code
        self.ble_name = ble_name
        self.started = time.time()
        # The correction from a requested camera-frame course to the raw ball
        # heading that achieves it. Seeded from the roster and then refined from
        # what the camera actually saw on every leg that moved, which makes the
        # rig immune to a sign convention being wrong somewhere below it.
        self.course_offset = float(heading_offset or 0.0)
        self.offset_samples = 0
        self.last_aim_error = None

        cx, cy = self._centre()
        recentre = lambda: Recenter((cx, cy), aim=self.aim)
        if quick:
            self.stages = [TrackingCheck(workspace=workspace, aim=self.aim),
                           NoiseProbe(frames=90, timeout=12.0),
                           HeadingStage(workspace=workspace),
                           recentre(),
                           # Fewer points, spread across whatever the cap
                           # permits — not a fixed ladder, which under a low
                           # cap leaves one usable leg and no line to fit.
                           SpeedMap(workspace=workspace, aim=self.aim,
                                    points=4, timeout=90.0),
                           recentre(),
                           BrakeTest(workspace=workspace,
                                     aim=self.aim, timeout=60.0),
                           recentre(),
                           LatencyProbe(repeats=2, workspace=workspace,
                                        aim=self.aim, timeout=50.0)]
        else:
            # Deliberately unhurried. Capping the speed costs the top of every
            # curve, so what is left has to be measured better to compensate —
            # and repeats are the one lever that always helps: noise falls as
            # the square root of them, and nothing about a slow ball makes a
            # repeat harder. The whole run is longer than it was and every
            # number in it is worth more.
            self.stages = [TrackingCheck(workspace=workspace, aim=self.aim),
                           NoiseProbe(frames=600),
                           HeadingStage(workspace=workspace),
                           recentre(),
                           SpeedMap(workspace=workspace, aim=self.aim,
                                    settle=1.4, window=1.8, timeout=260.0),
                           recentre(),
                           StepResponse(workspace=workspace, aim=self.aim,
                                        repeats=5, timeout=110.0),
                           recentre(),
                           BrakeTest(workspace=workspace, aim=self.aim,
                                     bytes_=None, timeout=170.0),
                           recentre(),
                           LatencyProbe(workspace=workspace, aim=self.aim,
                                        repeats=5, timeout=120.0)]

        for stage in self.stages:
            stage.max_byte = int(max_byte)
            stage.plan_safety = float(plan_safety)
            # Spread the samples across the range the cap actually permits.
            # A fixed ladder reaching to 255 puts three points under an 18cm/s
            # ceiling and throws the other seven away, and a line drawn through
            # three points extrapolated to 255 is a guess wearing an r-squared.
            # The same ten legs spread from the deadband to the cap fit far
            # better and cost exactly as much time.
            if isinstance(stage, SpeedMap) and not stage.explicit_bytes:
                stage.bytes = spread_bytes(max_byte, stage.points)
            if isinstance(stage, BrakeTest):
                # The coast constant is a slope, so it wants speeds spread
                # across the permitted range rather than four values that all
                # sit above the cap and get skipped.
                stage.bytes = spread_bytes(max_byte, 4, floor=40) * 2

        self.i = 0
        self.results = {}
        self.notes = []
        self.done = False
        self.cancelled = False
        self.aborted = False

    def _centre(self):
        if self.ws is None:
            return 100.0, 100.0
        x0, x1, y0, y1 = self.ws.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # -- driving ---------------------------------------------------------

    def aim(self, course_deg):
        """Requested camera-frame course -> the raw heading that achieves it."""
        return (float(course_deg) + self.course_offset) % 360.0

    def observe_legs(self, legs):
        """Fold a whole stage's legs into ONE aim correction.

        One correction, not one per leg. Every leg in a stage was driven under
        the same offset, so they are repeated measurements of a single error —
        applying a 0.6-gain correction to each in turn multiplies that error by
        six and sends the aim further wrong with every extra sample. The first
        dry run wound up 283 degrees out exactly this way.
        """
        errs = [wrap180(float(a) - float(r)) for r, a in legs]
        if not errs:
            return
        rad = np.radians(errs)
        err = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))
        # Clamped: one stage may not swing the aim more than this, so a stage
        # that mistracked cannot destroy a calibration the others agreed on.
        step = float(np.clip(err * 0.7, -45.0, 45.0))
        self.course_offset = (self.course_offset - step) % 360.0
        self.offset_samples += len(errs)
        self.last_aim_error = round(err, 1)

    @property
    def stage(self):
        return self.stages[self.i] if self.i < len(self.stages) else None

    @property
    def progress(self):
        s = self.stage
        if s is None:
            return "done"
        return f"{self.i + 1}/{len(self.stages)}  {s.label} {s.progress}".strip()

    def step(self, t, pos, dt):
        """One frame. Returns (heading_deg, speed_byte), or None to stop."""
        if self.done or self.cancelled:
            return None
        s = self.stage
        if s is None:
            self.done = True
            return None

        cmd = s.step(t, pos, dt)
        if not s.done:
            return cmd

        # The pre-flight reports its verdict from `result()`, so the abort has
        # to read it there. Checking only `stage.error` misses it entirely and
        # the run carries on to drive a robot the camera is not watching.
        if isinstance(s, TrackingCheck):
            verdict = s.result().get("error") or s.error
            if verdict:
                s.error = verdict

        # A tracker that is not following the robot invalidates everything
        # after it as much as everything during it, so this ends the run rather
        # than moving on to the next stage. Carrying on would produce a full
        # set of confident numbers measured against a stationary artifact.
        if s.error and ("tracker is not following" in s.error
                        or isinstance(s, TrackingCheck)):
            self._collect(s)
            self.notes.append("run stopped — nothing measured after this point "
                              "would have meant anything")
            self.aborted = True
            self.done = True
            return None

        self._collect(s)
        self.i += 1
        if self.i >= len(self.stages):
            self.done = True
        return None

    def _collect(self, s):
        key = s.label.replace(" ", "_")
        res = s.result()
        if s.error:
            res = {**res, "error": s.error}
            self.notes.append(f"{s.label}: {s.error}")
        if isinstance(s, Recenter):
            if not res.get("arrived"):
                self.notes.append("could not drive back to the middle — the next "
                                  "stage may run out of room")
            return
        self.results[key] = res

        # Every leg that moved tells us a little more about the aim frame.
        self.observe_legs([(r["requested_deg"], r["course_deg"])
                           for r in (res.get("rows") or [])
                           if r.get("course_deg") is not None
                           and (r.get("cm_s") or 0) > STOP_SPEED])
        if isinstance(s, StepResponse) and res.get("tau_s"):
            for later in self.stages:
                if isinstance(later, LatencyProbe):
                    later.tau_hint = res["tau_s"]
        if isinstance(s, HeadingStage) and s.cal.offset is not None and not s.error:
            self.course_offset = (self.course_offset - float(s.cal.offset)) % 360.0

    def cancel(self):
        self.cancelled = True
        self.done = True

    # -- the payoff ------------------------------------------------------

    def fit(self, cruise_cm_s=45.0):
        """Everything measured, plus what to do about it."""
        r = self.results
        speed = r.get("speed_map", {})
        step = r.get("step_response", {})
        brake = r.get("brake_test", {})
        lat = r.get("loop_latency", {})
        noise = r.get("position_noise", {})
        head = r.get("heading_offset", {})

        delay = lat.get("loop_delay_s")
        dead = step.get("dead_s")
        tau = step.get("tau_s")
        coast_k = brake.get("coast_s_per_cm_s")
        v_max = speed.get("max_speed_cm_s")

        out = {
            "robot": self.code,
            "ble_name": self.ble_name,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
            "arena_cm": None if self.ws is None else
                        [round(self.ws.bbox[1] - self.ws.bbox[0], 1),
                         round(self.ws.bbox[3] - self.ws.bbox[2], 1)],
            "position_noise": noise,
            "heading": head,
            "speed_map": speed,
            "step_response": step,
            "brake": brake,
            "latency": lat,
            "notes": list(self.notes),
        }

        # The delay a controller actually feels. Two independent estimates of
        # the same round trip: the reversal probe, and the dead time in the step
        # response. Disagreement between them is worth seeing, so both are kept.
        total_delay = delay if delay is not None else dead
        if delay is not None and dead is not None:
            out["delay_agreement_s"] = round(abs(delay - dead), 3)

        # A stage that recorded an error does not get to shape a
        # recommendation. Its partial data stays in the report above, where a
        # person can look at it and judge; what it must not do is descend
        # silently into a gain, because a number derived from a stage that
        # could not finish is indistinguishable, downstream, from one that was
        # measured properly. The speed map already worked this way for its own
        # top speed; this applies the same rule to every input.
        withheld = {}

        def usable(name, result, value):
            err = (result or {}).get("error")
            if value is None or not err:
                return value
            withheld[name] = err
            return None

        coast_k = usable("brake", brake, coast_k)
        if coast_k is not None and not _brake_trusted(brake):
            n, spread = _brake_range(brake)
            withheld["brake"] = (
                f"the coast constant came from {n} usable "
                f"{'stop' if n == 1 else 'stops'} spanning {spread:.0%} of "
                "their own top speed — too narrow a range to fit a slope "
                "through, so it is not being used")
            coast_k = None
        tau = usable("step_response", step, tau)
        dead = usable("step_response", step, dead)
        delay = usable("latency", lat, delay)
        v_max = usable("speed_map", speed, v_max)

        total_delay = delay if delay is not None else dead

        rec = {}
        if coast_k is not None:
            # The brake test already measures the whole round trip, and this is
            # worth being explicit about because adding the latency to it is
            # the obvious mistake: the cut position is where the CAMERA thought
            # the ball was, which is where it was one loop delay ago. So the
            # coast measured from it spans the delay and the roll-out together
            # — exactly the distance a controller loses between deciding to
            # stop and the ball stopping.
            rec["stopping_distance_s_per_cm_s"] = round(coast_k, 4)
            rec["stopping_distance_cm"] = {str(v): round(coast_k * v, 1)
                                           for v in (15, 30, 45, 60)}
            rec["max_precise_speed_cm_s"] = round(ARRIVE_TOL_CM / coast_k, 1)
            rec["note_speed_limit"] = (
                f"a robot faster than this cannot stop inside the "
                f"{ARRIVE_TOL_CM:.0f}cm arrival tolerance — above it, either "
                f"widen SLOW_RADIUS so it eases in, or accept overshoot")
        trusted = speed.get("max_speed_trusted")
        if v_max is not None and not trusted:
            out["speed_map_warning"] = (
                f"the top speed came out at {v_max:.0f}cm/s, extrapolated from "
                f"bytes up to {speed.get('highest_measured_byte')}. That is "
                "either impossible or too far a reach to trust, so it is not "
                "being used — raise the speed cap and run again if you need it.")
        if v_max is not None and trusted:
            rec["handle_max_speed_cm_s"] = v_max
            rec["note_max_speed"] = (
                "fleet/handle.py MAX_SPEED is the cm/s-to-byte calibration; set "
                "it to this and the byte a controller asks for finally means "
                "what it says")
        if speed.get("min_moving_byte") is not None and not speed.get("error"):
            # In BYTES, which is measured, and in cm/s only when there is a
            # trustworthy speed to convert with. Converting through a bogus
            # ceiling manufactures a deadband out of nothing: a 288cm/s ceiling
            # turned byte 18 into "the ball will not move below 20cm/s", which
            # then sat above the speed cap and made it unable to move at all.
            rec["min_moving_byte"] = speed["min_moving_byte"]
            if trusted:
                rec["min_moving_cm_s"] = round(
                    speed["min_moving_byte"] / 255.0 * v_max, 1)
                rec["note_deadband"] = (
                    "below this the ball is commanded and does not move, so a "
                    "controller easing to a stop stalls short of its target")

        if cruise_cm_s and coast_k is not None and tau is not None and total_delay is not None:
            # Easing in over SLOW_RADIUS is proportional control with gain
            # cruise/radius. It stays damped while gain times the total lag is
            # small, and the radius must also be comfortably longer than the
            # stopping distance or the ease-in starts too late to matter.
            #
            # Both halves of the lag are required rather than defaulted to
            # zero. A missing time constant treated as zero does not produce a
            # cautious radius, it produces a confidently small one — the exact
            # direction that rings.
            lag = tau + total_delay
            by_damping = cruise_cm_s * lag / DAMPING_TARGET
            by_stopping = cruise_cm_s * coast_k * 1.6
            rec["cruise_cm_s"] = cruise_cm_s
            rec["slow_radius_cm"] = round(max(by_damping, by_stopping), 1)
            rec["slow_radius_driver"] = ("damping" if by_damping >= by_stopping
                                         else "stopping distance")
        if noise.get("sigma_cm"):
            # vision/track.py's Kalman takes r as a variance, in cm^2.
            rec["kalman_r"] = round(max(noise["sigma_cm"], 0.2) ** 2, 3)
        if tau is not None:
            rec["sim_tau_s"] = tau
        if total_delay is not None:
            rec["sim_latency_steps"] = max(1, int(round(total_delay * 30.0)))
        if v_max:
            rec["sim_gain"] = round(v_max / 60.0, 3)

        # Arithmetically correct and physically impossible are not the same
        # thing, and only one of them is visible in the numbers. A slow radius
        # near the size of the room means the robot is inside its own arrival
        # zone nearly everywhere, which no amount of correct algebra fixes. The
        # arena is already recorded a few lines above, so the check is
        # available and was simply never made.
        arena = out.get("arena_cm")
        radius = rec.get("slow_radius_cm")
        if arena and radius:
            short = min(arena)
            if radius > short * SLOW_RADIUS_ARENA_FRACTION:
                rec["arena_warning"] = (
                    f"a slow radius of {radius:.0f}cm in an arena {short:.0f}cm "
                    f"across leaves the robot easing in almost everywhere — "
                    f"lower the cruise speed or re-measure, because the lag "
                    f"this was derived from implies a robot this room cannot "
                    f"hold")

        if withheld:
            rec["withheld"] = withheld
            rec["note_withheld"] = (
                "some stages did not finish, so what they measured is reported "
                "but not used — the numbers above are the ones that stand up")
        out["recommend"] = rec
        return out

    def save(self, path=None, cruise_cm_s=45.0):
        """Merge into `calib/motion.json`, keyed by robot. Never raises.

        The path is resolved when this is called, not when the module is
        imported. A default argument would capture the real `calib/` directory
        at import time and quietly ignore any attempt to point it elsewhere —
        which is how a test run ends up writing into a live calibration.
        """
        path = Path(path if path is not None else MOTION_PATH)
        try:
            blob = json.loads(path.read_text()) if path.exists() else {}
            if not isinstance(blob, dict):
                blob = {}
        except Exception:
            blob = {}
        fitted = self.fit(cruise_cm_s=cruise_cm_s)
        blob[self.code or "unknown"] = fitted
        try:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(blob, indent=2))
        except Exception as e:
            return None, f"could not write {path}: {e}"
        return fitted, None


def _brake_range(brake):
    """(usable stops, fractional spread of their entry speeds)."""
    rows = [r for r in (brake or {}).get("rows") or []
            if (r.get("entry_cm_s") or 0) > STOP_SPEED]
    if not rows:
        return 0, 0.0
    v = [float(r["entry_cm_s"]) for r in rows]
    return len(rows), (max(v) - min(v)) / max(max(v), 1e-9)


def _brake_trusted(brake):
    """Is this coast constant worth standing behind?

    The stage records its own verdict, but a calibration written before that
    verdict existed has none — and defaulting a missing verdict to "trusted" is
    how a file gets grandfathered past the check that was added because of it.
    The rows are right there, so the answer is recomputed rather than assumed.
    """
    verdict = (brake or {}).get("coast_trusted")
    if verdict is not None:
        return bool(verdict)
    n, spread = _brake_range(brake)
    return n >= BRAKE_MIN_ROWS and spread >= BRAKE_MIN_SPREAD


def load_motion(code=None, path=None):
    """Measured constants for one robot, or every robot. {} when unmeasured."""
    try:
        blob = json.loads(Path(path if path is not None else MOTION_PATH).read_text())
    except Exception:
        return {}
    if not isinstance(blob, dict):
        return {}
    return blob.get(code, {}) if code else blob
