#!/usr/bin/env python3
"""One bot, one blob, x and y. The least processing that can be trusted.

    python3.13 blob_test.py --camera 0
    python3.13 blob_test.py --source sim        # no camera, no robot

Three steps, and deliberately no fourth:

    1. threshold the V channel LOW  — low enough that the whole halo is ONE
       connected region rather than a core per LED
    2. take the largest component that passes the area filter
    3. intensity-weighted centroid of that component -> x, y

There is no grouping radius, no colour matching, no dot composition rule and no
heading. Every one of those was a place to be wrong, and none of them is needed
to answer where the ball is.

WHY LOW AND NOT HIGH. The instinct is to threshold hard so only the LED cores
survive. That splits one ball into three components and then needs a rule to
put them back together. Thresholding low skips the problem: the halo is already
one region, and the intensity weighting means the dim outer pixels barely move
the answer while still holding the region together. Measured on rendered balls
with known centres, this reads 100% of frames from a 68px ball down to an 8px
one, and through 31px of defocus. The three-dot method it replaces stops
answering below 34px and at 7px of blur -- which is the difference between
seeing the bot and not.

THE TAILLIGHT MUST BE OFF, and this app turns it off. Two tag LEDs sit
symmetrically about the centre, so their glow is balanced and the weighted
centroid IS the centre. The taillight is behind them, and it drags that
centroid about 7mm backwards along the heading -- an offset that ROTATES with
the robot, so no constant can cancel it. With the tail off the same measurement
gives 0.02mm. The tail earns its place when you want heading; it is pure error
when you want position.

IDENTITY IS NOT AN OPTICAL PROBLEM HERE. With one robot the only blob is the
robot, so nothing needs to be recognised. The coloured marker is drawn from the
roster as a LABEL -- it is this app's belief about which robot that is, not
something read from the picture.

That distinction starts mattering the moment there is a second robot, and this
app is deliberately not built for that yet. Two blobs need association between
frames, and association is where a tracker silently swaps two robots and then
draws a confident marker on the wrong one. The report this method comes from
seeded identity with a blink handshake -- LEDs off, then each robot blinks in
turn -- and thereafter trusted nearest-blob matching forever, which is exactly
the arrangement in which a swap is permanent. Anything added here has to make a
swap VISIBLE rather than merely unlikely.

IT STILL REFUSES. No blob, or a blob outside the area filter, prints why rather
than holding the last position on screen. A stale dot that looks live is the
one failure that a controller cannot detect for itself.

Keys — s save the raw frame to runs/, l lock exposure, u auto, v raw/mask,
space pause, esc quit.
"""

import argparse
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import pygame

from ui.theme import (CARD, CARD_EDGE, CHALK, CORAL, CYAN, DIM, GREY, INK,
                      MINT, PAD, PANEL, RULE, SUN, Button, Slider, card,
                      section)
from vision import config
from vision.homography import Homography

V_MIN = 20
"""Threshold on V, and the one knob that matters.

Low, for the reason in the module docstring: it is what makes the halo a single
region. On the rendered ball the halo merges at 25 and below and splits into a
core per LED at 30 and above -- so the default sits comfortably inside the
merged side rather than near the edge of it.

Too low and the floor joins in, which shows up as the blob's area jumping; too
high and the ball splits, which shows up as `parts` climbing above one. Both
are on the dock, so this is tuned by watching rather than by guessing. On real
hardware the number will differ; the two things to watch will not.
"""

MIN_AREA = 60
MAX_AREA = 40000
"""Area gate, in pixels. The floor rejects sensor noise and a stray reflection;
the ceiling rejects a window, a light fixture, or two balls that have merged --
a blob twice the expected size is not a better answer, it is a different one."""

JITTER_N = 60
"""Frames of position history kept, for the stability readout. At 30fps this is
two seconds -- long enough to show a wander and short enough to react."""

W, H = 1380, 980
MIN_W, MIN_H = 1040, 760
DOCK_W = 348


# -- the method ------------------------------------------------------------

def order_quad(pts):
    """Four clicked points into a non-self-intersecting quadrilateral.

    Sorted by angle about their own centroid. Clicks arrive in whatever order
    a person happens to make them, and `fillPoly` on an unsorted quad produces
    a bowtie -- two triangles with a pinch in the middle -- which masks out a
    band through the centre of the arena and looks like the detector failing
    rather than the corners being in the wrong order.
    """
    pts = [np.asarray(p, dtype=float) for p in pts]
    mid = np.mean(pts, axis=0)
    return sorted(pts, key=lambda p: math.atan2(p[1] - mid[1], p[0] - mid[0]))


def roi_mask(shape, corners):
    """A uint8 mask that is 1 inside the workspace quad and 0 outside."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(order_quad(corners), dtype=np.int32)], 1)
    return m


def touches_edge(labels, index, region):
    """Is this component clipped by the workspace boundary?

    Worth knowing and worth saying. Masking happens before the components are
    found, so a ball straddling the boundary keeps only the part that is inside
    -- and its weighted centroid is then pulled inwards by however much was cut
    off. The position is still reported, because a ball at the arena edge is
    still somewhere; it is flagged, because at that moment it is somewhere less
    precise than usual.
    """
    edge = region - cv2.erode(region, np.ones((3, 3), np.uint8))
    return bool(((labels == index) & (edge > 0)).any())


def find_blobs(frame, v_min=V_MIN, min_area=MIN_AREA, max_area=MAX_AREA,
               region=None):
    """Every lit region, with an intensity-weighted centroid. Brightest first.

    The centroid is weighted by how far each pixel is ABOVE the threshold, not
    by its raw value. That matters: weighting by raw value makes the answer
    depend on where the threshold sits, because every pixel carries a constant
    `v_min` of dead weight that shifts the mean toward whichever side has more
    area. Subtracting the threshold first removes it, and measured over a drag
    from 170 to 230 it cuts how far the reported centre wanders by three and a
    half times.
    """
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask = (v >= int(v_min)).astype(np.uint8)
    if region is not None:
        # Applied to the PIXELS, before anything is measured, so a lamp or a
        # window outside the workspace cannot contribute to a component at all
        # -- rather than being found and then argued with afterwards.
        mask = mask & region
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return [], mask, 0

    above = np.maximum(v.astype(np.float64) - float(v_min) + 1.0, 0.0)
    flat = labels.reshape(-1)
    w = (above.reshape(-1)) * (flat > 0)
    ys, xs = np.mgrid[0:v.shape[0], 0:v.shape[1]]
    tot = np.bincount(flat, weights=w, minlength=n)
    sx = np.bincount(flat, weights=w * xs.reshape(-1), minlength=n)
    sy = np.bincount(flat, weights=w * ys.reshape(-1), minlength=n)

    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area or tot[i] <= 0:
            continue
        out.append({"xy": np.array([sx[i] / tot[i], sy[i] / tot[i]]),
                    "area": area,
                    "peak": float(v[labels == i].max()),
                    "sum": float(tot[i]),
                    "clipped_by_region": (region is not None
                                          and touches_edge(labels, i, region))})
    out.sort(key=lambda b: -b["area"])
    return out, mask, n - 1


def read_one(frame, v_min=V_MIN, min_area=MIN_AREA, max_area=MAX_AREA):
    """The single robot in this frame, or the reason there isn't one.

    Returns `(blob, others, why, mask, parts)`. `others` is every additional
    blob that passed the gate -- reported rather than discarded, because a
    second lit thing in the arena is something you want to know about while you
    are still calibrating, not something for this to quietly ignore.
    """
    blobs, mask, parts = find_blobs(frame, v_min, min_area, max_area)
    if not blobs:
        why = ("nothing above threshold" if parts == 0 else
               f"{parts} bright region(s), none within the area gate")
        return None, [], why, mask, parts
    return blobs[0], blobs[1:], None, mask, parts


MAX_JUMP_PX = 60
"""How far the ball may move between frames before a candidate is disbelieved.

In pixels rather than centimetres so the gate still works before a homography
exists -- which is exactly when you are pointing the camera at the floor for
the first time and most want to see whether it is holding the ball.

At 30fps a Sphero at its full 60cm/s covers about 2cm a frame, which on a
typical mounting is well inside this. The gate is not trying to be tight; it is
trying to stop the marker teleporting to a reflection on the other side of the
arena, and for that a loose gate is enough.
"""

TRAVEL_WINDOW = 8
"""Frames of trail the direction of travel is fitted over.

A single frame's velocity is the difference of two noisy points and its
DIRECTION is far noisier than its length -- at a slow crawl it swings wildly
while the position it came from is perfectly good. Fitting a line over a short
window averages that out. Eight frames is about a quarter of a second: long
enough to be steady, short enough that a real turn is not smeared away.
"""

MIN_TRAVEL_PX = 30.0
"""Speed, in pixels per SECOND, below which no arrow is drawn at all.

Per second rather than per frame so the arrow does not appear and disappear
when the frame rate changes -- a threshold in frames is a different physical
speed on a camera running at 15fps than on one running at 30.

Thirty is about three pixels a frame at 30fps.
Below this the ball is not really moving and the fitted direction is a
measurement of the noise floor -- an arrow there spins on the spot and looks
like the tracker is confused when it is only being asked an unanswerable
question. If the arrow flickers while the ball is still, this is the number to
raise; the dock prints the current speed so you can see where to put it.
"""

AIM_MIN_SPEED_CM_S = 6.0
"""Below this the travel bearing reads the noise floor rather than a heading,
and correcting from it would walk the offset at random."""

AIM_MAX_STEP_DEG = 60.0
"""A single reading further off than this is a skid, a bump, or a tracker jump.
The frame between the ball and the arena does not rotate sixty degrees between
two camera frames."""

PROBE_SPEED = 6.0
"""cm/s asked for on a calibration or probe leg. Slow: these legs are measured,
not raced, and a slower ball slips less -- slip is the one error a single leg
cannot tell from a genuine heading offset.

ASKED FOR, and on real hardware probably not what you get. `set_velocity`
converts cm/s to a speed byte linearly through `MAX_SPEED`, and the measured
speed map in `calib/motion.json` says that is badly wrong at the bottom: the
slowest the ball was ever seen to move is 17.8cm/s at byte 18, with a fitted
intercept of 21cm/s at byte zero. So a request for 6 may well roll at three
times that.

Which is survivable here, because this leg measures DISPLACEMENT and does not
care how fast it was covered -- and if the ball does not move at all, the probe
says how far it got rather than reporting a heading from noise. Worth knowing
before reading a probe as a slow, gentle manoeuvre; it may not be one. That
speed map is itself poor (r2 0.23, and it was taken on a ball whose aim frame
was wrong), so the honest reading is that low-speed commands are unpredictable
rather than that they land at any particular number."""

DRIVE_ON_PREDICTION = 3
"""Frames the controller may keep driving on the tracker's prediction.

Not zero, because stopping on every dropped frame is what made the ball judder;
not large, because a prediction is a guess and driving on a guess is the thing
the safety rule exists to limit. Three frames is a tenth of a second and under
three centimetres at working speed."""

SLEW_DEG_S = 220.0
"""How fast the COMMANDED direction may turn, in degrees per second.

Position noise becomes heading noise -- pure pursuit's effective gain rises as
the lookahead shortens, so a jittery fix becomes a jittery command, and a
Sphero physically rotates its drive assembly to follow it. Limiting how fast
the command may swing costs a little tracking sharpness on a hairpin and takes
the buzz out of everything else."""

ARRIVE_CM = 6.0
"""How close counts as arrived, in centimetres.

A knob rather than a constant, and not merely for convenience: too tight a
radius is what makes a follower ORBIT a goal instead of reaching it. Steering
continuously at a point it can never quite satisfy, the ball circles -- which
looks like a controller fault and is really an unreachable acceptance test.

Six centimetres is a little under one ball DIAMETER (7.4cm), which is about the
tightest that is meaningful: a ball whose centre is within its own radius of
the goal is, by any physical reading, on it. Tighten it below the tracker's
accuracy and no amount of driving will ever satisfy it.
"""

STYLES = ("pursuit", "turn-go")
"""How the follower gets to the target.

`pursuit` steers continuously toward a point ahead on the path -- one smooth
curve, and the only sane choice for a line, a circle or a scribble.

`turn-go` points the ball at the target first and then drives it straight,
which is the shape the report this project started from used. It is the better
choice for point-to-point: the ball spends almost all of its time going in one
direction, so there is nothing for the steering to buzz against, and the route
is the straight line a person would draw rather than the curve pursuit makes.
"""

TURN_SPEED_CM_S = 5.0
"""Speed held during the turn phase.

Not zero, and that is forced by the tracker rather than chosen. A Sphero can
rotate its drive assembly in place, but a BLOB HAS NO HEADING WHEN IT IS NOT
MOVING -- a stationary ball is a round dot with no facing in it. Turning at a
crawl keeps just enough travel for the camera to read which way the ball now
points, so the turn can be CONFIRMED before speed is committed. Turning in
place would be smoother and would have to be taken on trust.
"""

TURN_OK_DEG = 18.0
"""Once the measured travel is this close to the target bearing, go."""

RETURN_DEG = 45.0
"""Bearing error that sends it back to turning rather than steering onward.

Wide on purpose. A narrow band turns every small correction into a stop-and-turn
and gives back the smoothness this mode exists for; the ordinary business of
being slightly off course is better handled by leaning into it while moving."""

TURN_MAX_S = 4.0
"""A turn that has not converged by now is not going to. Rather than crawling
forever, it gives up and drives -- being pointed roughly right and moving beats
being stopped and correct."""

AIM_MODES = ("off", "on")
"""How the measured heading error is fed back, switchable at runtime.

Not a constant, because the correct answer is a fact about YOUR ball and YOUR
camera that no amount of reading the source settles. A simulated robot rotates
its velocity by the offset as a maths angle and a real one adds it to a
clockwise compass heading; if the camera also sees the floor mirrored, the two
compasses run opposite ways and the sign flips again. Get it backwards and the
loop multiplies the error by (1 + gain) every step instead of shrinking it,
which on the floor looks like the ball spiralling away from the path -- not
like a sign fault.

`off` is the default and it is the diagnostic one: pure pursuit is a geometric
controller and needs no heading correction at all to follow a path, so if it
still spirals with this off, the fault is not here.
"""

AIM_GAIN = 0.35
"""How much of each measured error is taken out per control step.

Not 1.0. A single frame's bearing contains the controller's own turning as well
as the frame error, and correcting the whole of it chases the manoeuvre. At a
third per step the offset still settles within a second of steady travel, which
is far quicker than the two-second fold it replaces."""

COAST_FRAMES = 6
"""How many frames the track may run on prediction alone before it gives up.

Six is a fifth of a second: long enough to ride out a blink, a dropped frame,
or the ball passing under a cable, and short enough that a genuinely lost ball
is reported as lost while the number on screen is still nearly true. A coasted
position is NEVER presented as a measurement -- see `Track.status`.
"""


def fit_travel(window, min_speed=MIN_TRAVEL_PX):
    """Velocity in px/SECOND from `[(t, xy), ...]`, or None if barely moving.

    Fitted against real seconds, not against the frame index. The two are not
    the same number -- the camera delivers on its own schedule and frames get
    dropped -- and treating an index as time reports a speed wrong by whatever
    ratio those rates happen to sit at.

    A least-squares line rather than the difference of the endpoints: the
    endpoints are two noisy samples and the fit uses all of them, which matters
    most at low speed where the noise is a large share of the movement.
    """
    if len(window) < 3:
        return None
    t = np.array([w[0] for w in window], dtype=float)
    pts = np.array([w[1] for w in window], dtype=float)
    if t[-1] - t[0] <= 1e-6:
        return None
    t = t - t[0]
    vx = float(np.polyfit(t, pts[:, 0], 1)[0])
    vy = float(np.polyfit(t, pts[:, 1], 1)[0])
    speed = float(np.hypot(vx, vy))
    if speed < min_speed:
        return None
    return np.array([vx, vy]), speed


class Track:
    """One ball, followed across frames. Predict, gate, coast -- and no filter.

    This is the smallest thing that deserves to be called tracking, and each of
    its three parts fixes a specific way that picking the largest blob every
    frame goes wrong:

    PREDICT, so the ball is looked for where it is going rather than where it
    was. Constant velocity, straight from the last two fixes.

    GATE, so a candidate that could not possibly be the ball is refused instead
    of accepted. Without this a reflection that is momentarily larger than the
    ball simply steals the marker, and nothing on screen says it happened --
    the position just teleports and comes back.

    COAST, so one missed frame is not a lost ball. The report this method comes
    from used a Kalman filter for exactly this, and the filter earned its place
    there because their blobs were tiny and flickered constantly. With a merged
    halo the detection is far more reliable, so the cheap version is enough --
    and the cheap version has the advantage of being obviously honest: a
    coasted frame is FLAGGED, where a filter's output looks identical whether it
    was measured or invented.

    Deliberately no smoothing of the reported position. Smoothing trades lag
    for prettiness, and on a bench it would hide the jitter that the noise
    readout exists to show you. Anything downstream that wants a smooth
    position can filter a stream of honest measurements; it cannot recover the
    truth from a stream of pre-smoothed ones.
    """

    def __init__(self, max_jump=MAX_JUMP_PX, coast=COAST_FRAMES):
        self.max_jump = max_jump
        self.coast = coast
        self.xy = None              # last accepted measurement, px
        self.vel = np.zeros(2)      # px per frame
        self.misses = 0
        self.blob = None
        self.status = "no lock"
        self.rejected = []          # candidates refused by the gate, this frame
        self._pick = None           # a point the operator chose; see `pick`

    def pick(self, xy):
        """Track whichever blob is nearest this point, from the next frame on.

        Ungated on purpose: the operator pointing at a blob is better evidence
        than anything the tracker believes, so a pick overrides both the gate
        and the largest-blob rule rather than being weighed against them.
        """
        self._pick = np.asarray(xy, dtype=float)
        self.status = "picking"

    def forget(self):
        self.xy, self.vel, self.blob, self._pick = None, np.zeros(2), None, None
        self.misses = 0
        self.status = "no lock"

    @property
    def locked(self):
        return self.xy is not None and self.misses == 0

    @property
    def predicted(self):
        if self.xy is None:
            return None
        return self.xy + self.vel * (1 + self.misses)

    def update(self, candidates):
        """Choose this frame's blob from the candidates. Returns it, or None.

        Acquisition takes the largest blob, because with nothing to predict
        from there is nothing better to go on. Every frame after that is
        matched to the PREDICTION, not to size -- so once the ball is held, a
        bigger blob elsewhere cannot take it.
        """
        self.rejected = []
        if not candidates:
            return self._miss("no blob in frame")

        if self._pick is not None:
            chosen = min(candidates,
                         key=lambda b: float(np.linalg.norm(b["xy"] - self._pick)))
            self._pick = None
            self.blob = chosen
            self.xy, self.vel, self.misses = chosen["xy"].copy(), np.zeros(2), 0
            self.status = "locked"
            return chosen

        if self.xy is None:
            self.blob = max(candidates, key=lambda b: b["area"])
            self.xy, self.vel, self.misses = self.blob["xy"].copy(), np.zeros(2), 0
            self.status = "acquired"
            return self.blob

        target = self.predicted
        gate = self.max_jump * (1 + self.misses)
        scored = sorted(((float(np.linalg.norm(b["xy"] - target)), b)
                         for b in candidates), key=lambda p: p[0])
        best, blob = scored[0]
        self.rejected = [b for d, b in scored[1:]]
        if best > gate:
            return self._miss(f"nearest blob is {best:.0f}px away, gate {gate:.0f}px")

        # Velocity from the accepted fix, not from the prediction, so a coasted
        # stretch cannot compound its own guess into the estimate.
        step = blob["xy"] - self.xy
        frames = 1 + self.misses
        self.vel = step / frames
        self.xy = blob["xy"].copy()
        self.misses = 0
        self.blob = blob
        self.status = "locked"
        return blob

    def _miss(self, why):
        if self.xy is None:
            self.status = why
            return None
        self.misses += 1
        if self.misses > self.coast:
            self.xy, self.vel, self.blob = None, np.zeros(2), None
            self.status = f"lost — {why}"
            return None
        self.status = f"coasting {self.misses}/{self.coast} — {why}"
        return None



# -- the run log -----------------------------------------------------------

class RunLog:
    """One CSV per run: every frame's state, written as it happens.

    Every FRAME, including the ones with no ball in them. A gap in the data is
    the single most diagnostic thing a tracking log can contain, and a logger
    that only records successes turns a dropout into a missing row that looks
    identical to the camera having been slow.

    Flushed continuously rather than on close, because the runs worth reading
    back are disproportionately the ones that ended in a crash, an unplugged
    camera, or a ball driven into a wall -- and a buffered log loses exactly
    the last few seconds that say why.

    Two headings are recorded and they answer different questions:

        theta_global   which way the ball travelled, in the arena's frame
        theta_rel      that MINUS the course it was commanded, so a non-zero
                       value is the aim error still outstanding

    And three positions, which are three different things:

        x, y           where the ball IS
        goal           where it is being sent -- the end of the path
        target         where it is STEERING this frame, which under pursuit is
                       a point `lookahead` ahead on the path and is almost
                       never the goal

    Without the goal the log cannot answer "how far off was it", because the
    error is a difference and only one side of it was being recorded. Without
    the target you cannot tell a follower that is steering correctly at the
    wrong point from one steering wrongly at the right one.

    The four `set_*` columns carry the sliders as they stood on that frame.
    They move mid-run, and a log that cannot say what it was configured as
    cannot explain its own behaviour a day later.

    A robot with a correct aim shows theta_rel near zero however it is driving;
    one with a frame error shows a consistent offset, and one that is slipping
    shows a large value only while it accelerates.

    Both headings are COMPASS bearings -- zero is +y and they increase
    clockwise, the Sphero's own convention, taken from `velocity_to_command` so
    there is one definition of it in the project. Worth knowing when comparing
    against a simulated robot's `bias`, which is an ordinary maths angle
    running anticlockwise: the same error appears in the two with opposite
    signs, and that is a convention difference rather than a fault.

    A following controller adds a few degrees of its own: the ball is
    continuously being given a new course and its travel lags the command, so
    theta_rel on a curve reads a little larger than the standing aim error. The
    number to trust for aim is one measured on a straight line.
    """

    COLUMNS = ("t_s", "frame", "mode", "status",
               "x_px", "y_px", "x_cm", "y_cm",
               "goal_x", "goal_y", "gap_cm",
               "target_x", "target_y",
               "theta_global_deg", "theta_global_cmp", "theta_rel_deg",
               "speed_cm_s", "noise_mm",
               "parts", "area", "peak", "clipped",
               "path", "style", "set_speed", "set_lookahead", "set_arrive",
               "cmd_vx", "cmd_vy", "cmd_deg", "aim_offset_deg", "note")

    FLUSH_EVERY = 30

    def __init__(self, directory="runs", stamp=None):
        self.dir = directory
        self.name = f"blob_{stamp or time.strftime('%m%d_%H%M%S')}.csv"
        self.path = os.path.join(directory, self.name)
        self.rows = 0
        self._fh = None
        self._w = None
        self.error = None
        self.t0 = time.perf_counter()

    def _open(self):
        import csv
        os.makedirs(self.dir, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.COLUMNS)

    def write(self, **fields):
        """One row. Unknown keys are ignored, missing ones are left blank."""
        if self.error is not None:
            return
        try:
            if self._fh is None:
                self._open()
            fields.setdefault("t_s", round(time.perf_counter() - self.t0, 4))
            self._w.writerow([_csv_cell(fields.get(c)) for c in self.COLUMNS])
            self.rows += 1
            if self.rows % self.FLUSH_EVERY == 0:
                self._fh.flush()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def close(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def _csv_cell(v):
    """Numbers to a sane number of places; None to an empty cell.

    Empty rather than 0 or "nan": a blank says the quantity did not exist on
    that frame, where a zero says it was measured and came out zero, and those
    are different facts about a tracker.
    """
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return v


# -- paths and pursuit -----------------------------------------------------

class Path:
    """A route through the arena, in CENTIMETRES, as a polyline.

    Every shape reduces to this -- a line is two points, a circle is a closed
    ring of them, a freehand drag is a lot of them, and a single goal point is
    a path of length zero. One representation means the follower has one code
    path to be correct in, instead of a branch per shape that each has to be
    debugged on a real floor.

    Centimetres and not pixels, because the controller commands velocities in
    cm/s and a path in pixels would need converting on every tick -- with the
    conversion sitting between the thing you drew and the thing being driven,
    which is where a scale error hides.
    """

    def __init__(self, pts, closed=False, kind="polyline"):
        self.pts = [np.asarray(p, dtype=float) for p in pts]
        self.closed = bool(closed) and len(self.pts) > 2
        self.kind = kind
        ring = self.pts + [self.pts[0]] if self.closed else self.pts
        self.ring = ring
        seg = [float(np.linalg.norm(ring[i + 1] - ring[i]))
               for i in range(len(ring) - 1)]
        self.cum = np.concatenate([[0.0], np.cumsum(seg)]) if seg else np.zeros(1)

    @classmethod
    def point(cls, p):
        return cls([p], kind="point")

    @classmethod
    def line(cls, a, b):
        return cls([a, b], kind="line")

    @classmethod
    def circle(cls, centre, radius, n=48):
        centre = np.asarray(centre, dtype=float)
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return cls([centre + radius * np.array([np.cos(t), np.sin(t)])
                    for t in a], closed=True, kind="circle")

    @property
    def length(self):
        return float(self.cum[-1])

    def at(self, s):
        """The point `s` centimetres along. Wraps if closed, clamps if not."""
        if self.length <= 0:
            return self.pts[0].copy()
        s = s % self.length if self.closed else float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = int(np.clip(i, 0, len(self.ring) - 2))
        span = self.cum[i + 1] - self.cum[i]
        f = 0.0 if span <= 0 else (s - self.cum[i]) / span
        return self.ring[i] + (self.ring[i + 1] - self.ring[i]) * f

    def project(self, p):
        """(arc length, distance) of the nearest point on the path to `p`.

        Every segment is tested rather than searching from the last position.
        A path here is at most a few hundred points and this runs once a frame,
        so the cost is nothing -- and searching locally is how a follower gets
        stuck on the wrong lap of a circle, or skips a hairpin in a freehand
        scribble that doubles back on itself.
        """
        p = np.asarray(p, dtype=float)
        if self.length <= 0:
            return 0.0, float(np.linalg.norm(p - self.pts[0]))
        best = (0.0, float("inf"))
        for i in range(len(self.ring) - 1):
            a, b = self.ring[i], self.ring[i + 1]
            ab = b - a
            denom = float(ab @ ab)
            t = 0.0 if denom <= 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
            q = a + ab * t
            d = float(np.linalg.norm(p - q))
            if d < best[1]:
                best = (float(self.cum[i] + t * np.linalg.norm(ab)), d)
        return best


def pursue(path, pos, lookahead, speed, goal_tol=ARRIVE_CM):
    """Pure pursuit: a velocity in cm/s that chases a point AHEAD on the path.

    Returns `(velocity, target, done, note)`.

    The lookahead is what makes this stable. Steering straight at the nearest
    point on the path drives the robot perpendicular into it, overshoots, and
    oscillates; aiming at a point some distance further along turns that into a
    smooth converging curve. Longer lookahead is calmer and cuts corners more
    -- that is the whole trade, and it is why it is a slider.

    A velocity VECTOR rather than a turn rate, because a Sphero can be
    commanded to roll in any direction directly. The report this project
    started from treated the ball as a differential drive and had to turn on
    the spot before moving, gated by a Gaussian on the heading error; none of
    that is needed when the command is a heading in the first place.
    """
    pos = np.asarray(pos, dtype=float)
    if path is None or not path.pts:
        return np.zeros(2), None, False, "no path"

    if path.length <= 0:                       # a single goal point
        target = path.pts[0]
        gap = float(np.linalg.norm(target - pos))
        if gap <= goal_tol:
            return np.zeros(2), target, True, "arrived"
        return _toward(pos, target, speed), target, False, f"{gap:.0f}cm to go"

    s, off = path.project(pos)
    if not path.closed:
        remaining = path.length - s
        end = path.ring[-1]
        if remaining <= goal_tol and float(np.linalg.norm(end - pos)) <= goal_tol:
            return np.zeros(2), end, True, "arrived"
        target = path.at(min(s + lookahead, path.length))
    else:
        target = path.at(s + lookahead)

    note = f"{off:.0f}cm off path"
    return _toward(pos, target, speed), target, False, note


def _toward(pos, target, speed):
    step = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    n = float(np.linalg.norm(step))
    if n < 1e-9:
        return np.zeros(2)
    return step / n * float(speed)



def plant_constants(code=None):
    """Measured dynamics for a robot, from `calib/motion.json`.

    Returns a dict, or None. `fleet/characterize.py` writes that file; nothing
    here measures anything, because a battery already exists.

    WHAT IS AND IS NOT TRUSTWORTHY IN THAT FILE, because it matters and the
    file will not tell you unless you read all of it:

    `tau` is solid -- the step stage and the latency stage independently agree
    on 0.84s, and it is the physical inertia of a rolling sphere.

    `coast` is solid: eight measured stops, 8-17cm from 25-33cm/s entry, which
    is a consistent half-second of coasting whatever the speed.

    DEAD TIME IS NOT SOLID. The battery measures it twice and the two disagree
    by their whole value -- the step fit says 433ms, the dedicated reversal
    probe says 0.0 across four runs, and `delay_agreement_s` records the gap.
    The battery's own conclusion, in `recommend.sim_latency_steps`, is ONE
    frame. So dead time is reported here with that disagreement attached and is
    not used to size anything; the coast is used instead, because it is
    measured directly and it is the quantity that actually decides how close to
    a goal the ball can stop.
    """
    try:
        from fleet.characterize import MOTION_PATH
        import json
        data = json.loads(MOTION_PATH.read_text())
    except Exception:
        return None

    def read(entry, name):
        sr = (entry or {}).get("step_response") or {}
        lat = (entry or {}).get("latency") or {}
        rows = ((entry or {}).get("brake") or {}).get("rows") or []
        if not sr.get("tau_s"):
            return None
        coast = [r["coast_cm"] / r["entry_cm_s"] for r in rows
                 if r.get("entry_cm_s") and r.get("coast_cm")]
        return {"tau_s": float(sr["tau_s"]),
                "dead_step_s": sr.get("dead_s"),
                "dead_probe_s": lat.get("loop_delay_s"),
                "disagree_s": (entry or {}).get("delay_agreement_s"),
                "coast_s": float(np.mean(coast)) if coast else None,
                "coast_n": len(coast),
                "source": name}

    if code and read(data.get(code), code):
        return read(data[code], code)
    for other, entry in data.items():
        got = read(entry, f"{other} (not this ball)")
        if got:
            return got
    return None


def latency_advice(speed_cm_s, plant):
    """What the measured dynamics demand of the two distance knobs.

    LOOKAHEAD from `tau`: the ball is still accelerating into a command for
    about that long, so aiming closer than `v * tau` means steering at a point
    it has already passed by the time it responds. That is the buzz, and
    shortening the lookahead to tighten it makes it worse.

    ARRIVAL from the measured COAST, not from dead time. When you say stop the
    ball keeps going for about half a second whatever its speed -- 14cm at
    28cm/s, measured over eight stops -- so a radius inside that cannot be held
    however good the aim. Sizing this off dead time was wrong twice over: the
    dead time is disputed, and it is the smaller effect even if you believe it.
    """
    tau = plant["tau_s"]
    coast_s = plant.get("coast_s") or 0.5
    return {"lookahead_cm": speed_cm_s * tau,
            "arrive_cm": speed_cm_s * coast_s}


# -- the camera ------------------------------------------------------------

class Camera(threading.Thread):
    """Frames on their own thread; a blocking read on the render thread is a
    frozen window, and a frozen window during bring-up reads as a crash."""

    def __init__(self, spec, size=None, leds=None, pose=None):
        super().__init__(daemon=True, name="blob-cam")
        self.source = None
        self.error = None
        try:
            if str(spec) in ("sim", "ball"):
                self.source = BallSource(leds=leds, pose=pose)
            else:
                from vision.synthetic import open_source
                self.source = open_source(spec, size=size)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        self.frame = None
        self.count = 0
        self.fps = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def run(self):
        last, ticks = time.perf_counter(), 0
        while not self._stop.is_set():
            if self.source is None:
                time.sleep(0.2)
                continue
            try:
                ok, img = self.source.read()
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.2)
                continue
            if not ok or img is None:
                self.error = "camera returned no frame"
                time.sleep(0.05)
                continue
            self.error = None
            with self._lock:
                self.frame = img
                self.count += 1
            ticks += 1
            now = time.perf_counter()
            if now - last >= 1.0:
                self.fps, ticks, last = ticks / (now - last), 0, now

    def latest(self):
        """The newest frame and its sequence number, so a caller can tell a
        fresh frame from the one it already has."""
        with self._lock:
            if self.frame is None:
                return None, self.count
            return self.frame.copy(), self.count

    def close(self):
        self._stop.set()
        self.join(timeout=1.0)
        try:
            if self.source is not None:
                self.source.release()
        except Exception:
            pass


class BallSource:
    """A fake camera showing one ball with its tail off, for working with none.

    Takes a `leds` callable returning the current brightness 0-255, so the LED
    slider moves this picture the way it moves a real one. Without that the
    brightness control would be untestable until hardware appeared, which is
    exactly when you least want to be discovering that it does not work.
    """

    def __init__(self, size=(1280, 720), px_cm=9.2, seed=0, leds=None,
                 pose=None):
        """`pose` is an optional callable returning the ball's (x, y) in cm.

        When it is supplied the fake camera stops inventing motion and renders
        wherever the simulated robot actually is -- which closes the loop:
        tracker sees the ball, controller commands a velocity, the simulated
        robot moves, and the camera sees the result. Without that the follower
        could only ever be tested against a ball that ignored it, which is no
        test of a follower at all.
        """
        self.w, self.h = size
        self.px_cm = px_cm
        self.rng = np.random.default_rng(seed)
        self.leds = leds or (lambda: 255)
        self.pose = pose
        self.pos = np.array([self.w / px_cm / 2, self.h / px_cm / 2])
        self.heading = 40.0
        self.turn = 22.0

    GAIN = 1.0 / 44.0
    """Chosen to sit just under clipping: the cores peak near 247, which is the
    frame this method wants and the one the brightness slider aims for."""

    def read(self):
        import math
        from vision.shots import _glow, tag_bgr, BALL_CM, TAG_R_CM
        dt = 1 / 30.0
        if self.pose is None:
            self.heading = (self.heading + self.turn * dt) % 360.0
            step = np.array([math.cos(math.radians(self.heading)),
                             math.sin(math.radians(self.heading))]) * 8.0 * dt
            self.pos += step
            for ax, hi in enumerate((self.w / self.px_cm, self.h / self.px_cm)):
                if self.pos[ax] < 12 or self.pos[ax] > hi - 12:
                    self.heading = (self.heading + 137.0) % 360.0
                    self.pos[ax] = float(np.clip(self.pos[ax], 12, hi - 12))

        if self.pose is not None:
            got = self.pose()
            if got is not None:
                self.pos = np.asarray(got, dtype=float)
        amp = max(0.0, float(self.leds()) / 255.0)
        canvas = np.zeros((self.h, self.w, 3), np.float32)
        canvas[:] = (6.0, 5.0, 4.0)
        cx, cy = self.pos * self.px_cm
        r = BALL_CM * self.px_cm / 2.0
        tp = TAG_R_CM * self.px_cm
        col = tag_bgr("red")
        a = np.array([math.cos(math.radians(self.heading)),
                      math.sin(math.radians(self.heading))])
        # Two tag LEDs and no taillight -- the arrangement this app requires.
        _glow(canvas, (cx, cy), r * 0.95, col, 90.0 * amp)
        for s in (1, -1):
            pt = (cx + s * a[0] * tp, cy + s * a[1] * tp)
            _glow(canvas, pt, r * 0.68, col, 600.0 * amp)
            _glow(canvas, pt, 3.0, col, 10000.0 * amp)
        canvas += self.rng.normal(0.0, 1.6, canvas.shape).astype(np.float32)
        return True, np.clip(canvas * self.GAIN, 0, 255).astype(np.uint8)

    def release(self):
        pass


class Scan(threading.Thread):
    """Discover Spheros without freezing the window.

    On its own thread because `find_toys` blocks for its whole timeout, and a
    seven-second freeze in the middle of a bring-up session is indistinguishable
    from a crash.
    """

    def __init__(self, timeout=7.0):
        super().__init__(daemon=True, name="blob-scan")
        self.timeout = timeout
        self.names = []
        self.error = None
        self.finished = False

    def run(self):
        try:
            from spherov2 import scanner
            toys = scanner.find_toys(timeout=self.timeout)
            self.names = sorted({getattr(t, "name", None) or str(t)
                                 for t in toys})
            if not self.names:
                self.error = ("nothing answered — shake a ball awake, and check "
                              "it is not still paired to a phone")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.finished = True


# -- the app ---------------------------------------------------------------

def to_surface(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")


class BlobTest:
    def __init__(self, spec="0", size=None, exposure=-7, code=None,
                 with_fleet=True):
        pygame.init()
        pygame.display.set_caption("blob test — one bot, one blob, x and y")
        try:
            sw, sh = pygame.display.get_desktop_sizes()[0]
            globals()["W"] = max(MIN_W, min(W, sw - 40))
            globals()["H"] = max(MIN_H, min(H, sh - 80))
        except Exception:
            pass
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 20)

        self.v_min = V_MIN
        self.min_area = MIN_AREA
        self.max_area = MAX_AREA
        self.exposure = int(exposure)
        self.bright = config.LED_VALUE
        self.paused = False
        self.view = "raw"
        self.note, self.note_tone = "", DIM
        self.frame = self.mask = None
        self.seen = -1
        # The workspace quad, in FRAME pixels rather than screen pixels, so it
        # survives a window resize and can be written to disk and reused.
        self.corners = None
        self.region = None          # the rasterised mask, rebuilt on resize
        self.mode = "run"           # or corners / target / point / line / ...
        self.picking = []
        self.path = None            # a Path in arena cm
        self.armed = False          # is the controller allowed to drive?
        self.speed = 20             # cm/s, the slider the operator drives with
        self.lookahead = 15         # cm
        self.drive_note = ""
        self.target_cm = None
        self._stepped = 0.0
        self.scan = None
        self.scan_rows = []
        self.tab = "track"
        self.robot_color = "red"
        self.color_rows = []
        self.cmd_v = None
        self.turning = None
        self._slew_at = self._slew_deg = None
        self.cmd_log = deque(maxlen=120)
        self.probe = None
        self.flips = {"x": False, "y": False}
        self._slew_at = None
        self._slew_deg = None
        self.style = "pursuit"
        self.turning = None
        self.goal_tol = ARRIVE_CM
        self.plant = None
        self.aim_mode = "on"
        self.aim_note = ""
        self.manual = False
        self.zeroed = False
        self.log = RunLog()
        self._view = None           # (ox, oy, scale) from the last draw
        self.blob = None
        self.others = []
        self.why = None
        self.parts = 0
        self.history = deque(maxlen=JITTER_N)
        self.trail = deque(maxlen=90)
        self.track = Track()
        self.exposure_locked = False
        self.exp_took = None

        self.fleet = None
        self.code = code
        self.marker = CYAN
        if with_fleet:
            self.build_fleet()

        self.cam = Camera(spec, size=size, leds=lambda: self.bright,
                          pose=self.robot_pose)
        self.cam.start()
        if self.cam.error:
            self.say(self.cam.error, CORAL)

        self.homography = Homography.load()
        if not self.homography.ready:
            self.say("no calib/homography.json — x/y stay in pixels", SUN)

        if str(spec) in ("sim", "ball"):
            # The saved calibration belongs to a real camera and says nothing
            # true about a rendered one, so in sim it is replaced by the exact
            # scale the fake camera draws at. Without this the centimetres the
            # controller commands and the centimetres the tracker reports are
            # two different units, and the loop cannot be closed even in
            # principle -- which would make the sim useless for the one thing
            # it is needed for.
            px_cm = getattr(self.cam.source, "px_cm", 9.2)
            self.homography = Homography(
                [[1.0 / px_cm, 0.0, 0.0], [0.0, 1.0 / px_cm, 0.0],
                 [0.0, 0.0, 1.0]])
            self.homography.width = self.homography.height = 0.0
            self.say(f"sim: using a {px_cm:.1f} px/cm scale, not calib/", DIM)

        saved = config.load("blob_region")
        if saved and len(saved.get("corners", [])) == 4:
            self.corners = [np.array(p, dtype=float) for p in saved["corners"]]
            self.say("workspace loaded from calib/blob_region.json — "
                     "press x to clear")

        self.sliders, self.buttons = [], []
        self.build_dock()

    # -- the robot --------------------------------------------------------

    def build_fleet(self):
        """The fleet layer, with no robot in it unless one was asked for.

        Empty by default, and that is the point of `--robot` being optional:
        this app connects to whatever you pick out of a scan, so it has no
        business assuming which ball in `roster.json` you meant. The previous
        default -- the first enabled roster entry -- silently pointed every
        control at a SIMULATED robot, and the dock reported it connected.
        """
        try:
            from fleet.manager import Fleet
        except Exception as e:
            self.say(f"no fleet: {type(e).__name__}: {e}", SUN)
            return
        if not self.code:
            self.fleet = Fleet()
            self.say("press b to scan for a ball, then click it to connect")
            return
        try:
            from fleet.roster import Roster
            roster = Roster.load()
            self.fleet = Fleet()
            entry = roster.by_code(self.code)
            if entry is None:
                self.say(f"{self.code} is not in roster.json — scan instead", SUN)
                self.code = None
                return
            errs = self.fleet.add(entry)
            if errs:
                self.say(errs[0], CORAL)
                self.code = None
                return
            self.set_robot_color(entry.color)
        except Exception as e:
            self.say(f"could not load {self.code}: {type(e).__name__}: {e}", CORAL)
            self.code = None

    def push_led(self):
        """Light the robot, and force the taillight OFF.

        Not optional, and not a slider. A taillight drags the weighted centroid
        about 7mm backwards along the heading, and because that offset rotates
        with the robot it cannot be calibrated out -- so leaving it available
        would only offer a way to make the measurement worse.
        """
        if not self.fleet or not self.code:
            return
        h = self.fleet.handles.get(self.code)
        if h is None:
            return
        try:
            hue = config.COLORS[self.robot_color]["hue"]
            h.set_led(config.led_rgb(hue, value=int(self.bright)))
            h.set_back_led(0)
        except Exception as e:
            self.say(f"{self.code}: {type(e).__name__}: {e}", CORAL)

    def set_bright(self, value):
        self.bright = int(value)
        self.push_led()

    def robot_pose(self):
        """Where the SIMULATED robot is, for the fake camera to draw it.

        Returns None for a real robot, which is the whole point -- a real ball
        is drawn by the world, and the fake camera must not be handed a
        position that came from anywhere but the simulator.
        """
        if not self.fleet or not self.code:
            return None
        h = self.fleet.handles.get(self.code)
        if h is None or getattr(h, "kind", "") != "sim":
            return None
        return np.asarray(h.pos, dtype=float)

    # -- bluetooth --------------------------------------------------------

    def start_scan(self):
        """Ask what is advertising. Results land in an overlay you can click."""
        if self.scan is not None and not self.scan.finished:
            return
        self.disarm()
        self.scan = Scan()
        self.scan.start()
        self.say("scanning for Spheros — about 7 seconds")

    def dismiss_scan(self):
        self.scan, self.scan_rows = None, []

    AD_HOC = "BALL"
    """Roster code for a ball connected from the scan.

    This app does not write `roster.json`. That file is live state the tracker,
    the calibrator and the swarm all read, and a bring-up tool that edits it as
    a side effect of a click can leave every other tool pointed at a ball that
    was only ever meant for one session. A robot connected here exists in this
    process and nowhere else; bind it permanently in `calib.py`, deliberately,
    if that is what you want.
    """

    def connect(self, ble_name):
        """Connect to a discovered ball, for this session only."""
        if self.fleet is None:
            self.say("no fleet layer — cannot connect", CORAL)
            return
        self.disconnect()
        try:
            from fleet.roster import RobotEntry
            entry = RobotEntry(name=ble_name, code=self.AD_HOC, kind="real",
                               color=self.robot_color, ble_name=ble_name)
            errs = self.fleet.add(entry)
        except Exception as e:
            self.say(f"connect failed: {type(e).__name__}: {e}", CORAL)
            return
        if errs:
            self.say(errs[0], CORAL)
            return
        self.code = self.AD_HOC
        self.zeroed = False     # a Sphero fixes its heading reference on
        self.dismiss_scan()     # connect, so any earlier zero is void
        self.push_led()
        self.say(f"connecting to {ble_name} — lights follow once the link is up",
                 MINT)

    def disconnect(self):
        """Drop the ad-hoc robot. Stops it first -- always."""
        self.disarm()
        self.zeroed = False
        if self.fleet is None:
            return
        if self.AD_HOC in self.fleet.handles:
            try:
                self.fleet.remove(self.AD_HOC)
            except Exception:
                pass
            if self.code == self.AD_HOC:
                self.code = None

    def set_robot_color(self, name):
        """The LED colour, and the marker drawn for it.

        One name drives both, so what the ball glows and what the overlay draws
        cannot disagree -- the same reason `vision/config.py` keeps a single
        palette instead of one table for the robot and one for the screen.
        """
        if name not in config.COLORS:
            return
        self.robot_color = name
        draw = config.COLORS[name]["draw"]
        self.marker = (draw[2], draw[1], draw[0])
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is not None:
            h.color = name
        self.push_led()
        self.say(f"lit {name}")

    def link_report(self):
        """What the radio is actually doing, in words. `(text, tone)` or None.

        The handle already records why a connect failed and whether the adapter
        hit its connection ceiling; none of that was reaching the screen, so a
        ball that never appeared looked identical to one that was merely slow.
        """
        if not self.fleet or not self.code:
            return None
        h = self.fleet.handles.get(self.code)
        if h is None:
            return None
        if getattr(h, "kind", "") != "real":
            return (f"{self.code} is SIMULATED — no radio. Press b to scan, "
                    "then click a ball to bind it.", SUN)
        if getattr(h, "link_up", False):
            return (f"{self.code} linked as {h.ble_name}", MINT)
        if getattr(h, "max_connections_hit", False):
            # Imported here, not at module scope: this app has to start on a
            # machine where the fleet layer is broken, which is exactly the
            # machine you are on when you need it.
            from fleet.real_handle import MAX_CONNECTIONS_HINT
            return (MAX_CONNECTIONS_HINT, CORAL)
        err = getattr(h, "last_error", None)
        tries = getattr(h, "attempts", 0)
        if err:
            return (f"{self.code} ({h.ble_name}) not connected after {tries} "
                    f"tries: {err}", CORAL)
        return (f"{self.code} connecting to {h.ble_name}… ({tries} tries)", SUN)

    # -- camera controls --------------------------------------------------

    def set_exposure(self, value):
        self.exposure = int(value)
        src = self.cam.source
        if src is None or not hasattr(src, "set"):
            self.say("this source has no camera controls", DIM)
            return
        self.exp_took = src.set("exposure", self.exposure)

    def exposure_mode(self, lock):
        """Freeze the camera's metering, or hand it back.

        On many macOS webcams this is the only exposure control there is:
        OpenCV's AVFoundation backend accepts `CAP_PROP_EXPOSURE` and drops it.
        Locking still does the job, because the problem was never the absolute
        value -- a metering camera AMPLIFIES as you darken the room, which is
        backwards for a method that wants a black floor and a bright ball.
        Expose against something bright, lock, then darken the room.
        """
        try:
            from vision import avcam
        except Exception as e:
            self.say(f"no AVFoundation control: {e}", SUN)
            return
        ok, msg = (avcam.lock() if lock else avcam.auto())
        self.exposure_locked = bool(lock and ok)
        self.say(msg + ("  — now darken the room" if lock and ok else ""),
                 MINT if ok else CORAL)

    def manual(self):
        src = self.cam.source
        if src is None or not hasattr(src, "manual"):
            self.say("this source has no manual controls", DIM)
            return
        self.say(f"manual: {src.manual()}")

    def save_frame(self):
        """The RAW frame, never the overlay: an opinion baked into a pixel
        cannot be re-read later with a different threshold."""
        if self.frame is None:
            self.say("no frame to save", SUN)
            return
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"blob_{time.strftime('%m%d_%H%M%S')}.png")
        cv2.imwrite(path, self.frame)
        self.say(f"saved {os.path.relpath(path)}", MINT)

    def say(self, msg, tone=DIM):
        self.note, self.note_tone = msg, tone
        print(f"blob_test: {msg}")

    # -- loop -------------------------------------------------------------

    def tick(self):
        """Advance one CAMERA frame, not one render tick.

        The window redraws at 60fps and the camera delivers at whatever rate it
        manages; running the pipeline on every redraw processed the same
        picture repeatedly, which is wasted work and, worse, quietly wrong. It
        appended a duplicate position to the trail with a fresh timestamp every
        time, which drags the fitted travel speed toward zero, and it made the
        tracker count its coasting in redraws rather than in frames -- so the
        coast budget expired in a third of the time it claimed to.
        """
        fresh = False
        if not self.paused:
            frame, count = self.cam.latest()
            if frame is not None and count != self.seen:
                self.frame, self.seen, fresh = frame, count, True
        if self.frame is None:
            return

        # Detection re-runs on every redraw so that dragging the threshold
        # while PAUSED still moves the mask and the region count -- which is
        # how the threshold gets tuned. Only the track and the trail are held
        # to real frames, because those are the things a repeated picture would
        # corrupt rather than merely recompute.
        if self.corners is not None and (
                self.region is None
                or self.region.shape[:2] != self.frame.shape[:2]):
            self.region = roi_mask(self.frame.shape, self.corners)
        candidates, self.mask, self.parts = find_blobs(
            self.frame, self.v_min, self.min_area, self.max_area,
            region=self.region if self.corners is not None else None)
        if not fresh:
            return
        self.blob = self.track.update(candidates)
        self.others = self.track.rejected
        self.why = None if self.blob is not None else self.track.status
        if self.blob is not None:
            now = time.perf_counter()
            self.history.append((now, self.to_cm(self.blob["xy"])))
            self.trail.append((now, self.blob["xy"].copy()))
        else:
            # Cleared, not held. A position that stops updating but stays on
            # screen is indistinguishable from a live one, and that is the
            # single thing a controller downstream cannot detect for itself.
            self.history.clear()
            if self.track.xy is None:
                self.trail.clear()
        # Simulated robots only move when something advances them. A real one
        # moves because it is a ball on a floor.
        if self.fleet is not None:
            now = time.perf_counter()
            dt = min(0.1, now - self._stepped) if self._stepped else 0.0
            self._stepped = now
            if dt > 0:
                try:
                    self.fleet.step(dt)
                except Exception:
                    pass
        self.drive()
        self.log_state()

    def to_cm(self, xy):
        if not self.homography.ready:
            return np.asarray(xy, dtype=float)
        return self.homography.to_cm([xy])[0]

    @property
    def travel(self):
        """Direction and speed of TRAVEL, or None if it is not moving enough.

        Travel, not heading, and the distinction is the whole reason the report
        this method comes from needed an integral term in its controller. This
        says which way the ball is GOING. A Sphero that slips, or is nudged, or
        is pushed by another ball, goes one way while pointing another -- and
        nothing in a blob can tell you which. Read it as motion, never as
        facing.

        Fitted, not differenced: a line through the last `TRAVEL_WINDOW`
        positions. The one-frame velocity has a direction that swings wildly at
        low speed even when every position in it is good.
        """
        if not self.track.locked or len(self.trail) < TRAVEL_WINDOW:
            return None
        return fit_travel(list(self.trail)[-TRAVEL_WINDOW:])

    def travel_readout(self):
        """The travel direction in ARENA terms: (deg image, deg compass, cm/s).

        Measured by mapping two points through the homography rather than by
        fitting the centimetre history a second time. The angle a direction
        makes is not preserved by a projective transform, so a direction fitted
        in pixels and reported as if it were an arena bearing would be wrong by
        however much the camera is tilted.
        """
        got = self.travel
        if got is None or self.track.xy is None:
            return None
        v, speed = got                       # v is px/s
        if not self.homography.ready:
            deg = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
            return deg, None, speed
        # One second of travel, mapped through the homography. Both the bearing
        # and the distance have to be measured in the arena rather than in the
        # image: a projective transform preserves neither, so a direction
        # fitted in pixels is not the direction the robot is driving in.
        p0 = self.to_cm(self.track.xy)
        p1 = self.to_cm(self.track.xy + v)
        step = p1 - p0
        deg = float(np.degrees(np.arctan2(step[1], step[0])) % 360.0)
        # The compass bearing comes from `velocity_to_command` rather than from
        # `90 - deg` written out here. They agree today, and a second place
        # that knows the Sphero's convention is a second place to get it wrong
        # -- which on this quantity is the error that costs a fortnight,
        # because a sign fault circles whatever you correct rather than
        # obviously failing.
        from fleet.handle import velocity_to_command
        return deg, velocity_to_command(step)[0], float(np.linalg.norm(step))

    @property
    def jitter_mm(self):
        """The measurement noise in the reported position, in mm.

        DETRENDED, which is what makes it readable while the ball is moving.
        Taken as spread about the mean, a ball crossing the arena reports
        hundreds of millimetres of "wander" that is simply travel, and the
        number is then useless exactly when you are driving. Fitting and
        removing a straight line per axis leaves what constant-velocity motion
        cannot explain -- which is the noise floor, and is what this is for.

        A turn or an acceleration still leaks in as curvature. That is honest:
        it is real movement the straight line could not account for, and the
        reading to trust is the one taken while nothing is driving.
        """
        if len(self.history) < 8 or not self.homography.ready:
            return None
        t = np.array([h[0] for h in self.history], dtype=float)
        pts = np.array([h[1] for h in self.history], dtype=float)
        if t[-1] - t[0] <= 1e-6:
            return None
        t = t - t[0]
        resid = np.stack([pts[:, i] - np.polyval(np.polyfit(t, pts[:, i], 1), t)
                          for i in range(2)], axis=1)
        return float(np.linalg.norm(resid, axis=1).max() * 10)

    # -- dock -------------------------------------------------------------

    TABS = (("track", "TRACK"), ("robot", "ROBOT"))

    def set_tab(self, name):
        self.tab = name
        self.build_dock()

    def build_dock(self):
        """Widgets for the CURRENT tab only.

        Rebuilt on every switch rather than built once and hidden, because a
        hidden slider that still answers `hit()` is a control you can operate
        by accident from another tab -- and the ones on this dock drive a robot.
        """
        self.sliders, self.buttons = [], []
        x, w, y = W - DOCK_W + PAD, DOCK_W - 2 * PAD, 24
        tw = (w - 6) // 2
        for i, (key, label) in enumerate(self.TABS):
            b = Button((x + i * (tw + 6), y, tw, 26), label,
                       (lambda k=key: self.set_tab(k)), toggle=True,
                       tone=CYAN if self.tab == key else None)
            b.on = (self.tab == key)
            self.buttons.append(b)
        y += 40
        (self.build_track if self.tab == "track" else self.build_robot)(x, w, y)

    def build_track(self, x, w, y):
        for label, lo, hi, get, set_ in (
                ("threshold", 5, 255, lambda: self.v_min, self.set_v),
                ("min area", 4, 2000, lambda: self.min_area, self.set_min),
                ("max area", 500, 80000, lambda: self.max_area, self.set_max),
                ("exposure", -13, 0, lambda: self.exposure, self.set_exposure),
                ("max jump", 8, 400, lambda: self.track.max_jump, self.set_jump),
                ("speed", 0, 60, lambda: self.speed, self.set_speed),
                ("lookahead", 2, 60, lambda: self.lookahead, self.set_lookahead),
                ("arrive", 2, 40, lambda: self.goal_tol, self.set_goal_tol)):
            self.sliders.append(Slider((x, y, w, 18), label, lo, hi, get, set_))
            y += 26
        y += 6
        bw = (w - 12) // 3
        rows = ((("lock", lambda: self.exposure_mode(True), MINT),
                 ("auto", lambda: self.exposure_mode(False), None),
                 ("manual", self.manual, None)),
                (("save", self.save_frame, CYAN),
                 ("mask", self.toggle_view, None),
                 ("pause", self.toggle_pause, None)),
                (("corners", self.start_corners, None),
                 ("target", self.start_target, None),
                 ("clear", self.clear_region, None)))
        for row in rows:
            for i, (label, cb, tone) in enumerate(row):
                self.buttons.append(Button((x + i * (bw + 6), y, bw, 24),
                                           label, cb, tone=tone))
            y += 30
        pw = (w - 4 * 4) // 5
        for i, (label, kind) in enumerate((("pt", "point"), ("line", "line"),
                                           ("poly", "poly"), ("circ", "circle"),
                                           ("free", "free"))):
            self.buttons.append(Button((x + i * (pw + 4), y, pw, 24), label,
                                       (lambda k=kind: self.start_path(k))))
        y += 30
        self.buttons.append(Button((x, y, bw, 24), "GO", self.arm, tone=MINT))
        self.buttons.append(Button((x + bw + 6, y, bw, 24), "STOP",
                                   self.disarm, tone=CORAL))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "erase",
                                   self.clear_path))
        y += 30
        self.buttons.append(Button((x, y, bw * 2 + 6, 24),
                                   f"style: {self.style}", self.cycle_style,
                                   tone=CYAN))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "fit lag",
                                   self.apply_latency_advice, tone=SUN))
        self.dock_top = y + 36

    def build_robot(self, x, w, y):
        bw = (w - 12) // 3
        self.buttons.append(Button((x, y, bw, 24), "scan", self.start_scan,
                                   tone=CYAN))
        self.buttons.append(Button((x + bw + 6, y, bw, 24), "drop",
                                   self.disconnect, tone=CORAL))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "STOP",
                                   self.disarm, tone=CORAL))
        y += 30
        self.buttons.append(Button((x, y, bw * 2 + 6, 24), "zero the aim",
                                   self.zero_now, tone=MINT))
        self.buttons.append(Button((x + bw * 2 + 12, y, bw, 24), "probe",
                                   self.start_probe, tone=SUN))
        y += 30
        self.buttons.append(Button((x, y, bw * 2 + 6, 24),
                                   f"aim fix: {self.aim_mode}",
                                   self.cycle_aim, tone=CYAN))
        y += 30
        self.buttons.append(Button((x, y, bw, 24),
                                   f"flip x{'*' if self.flips['x'] else ''}",
                                   (lambda: self.flip_axis("x")), tone=SUN))
        self.buttons.append(Button((x + bw + 6, y, bw, 24),
                                   f"flip y{'*' if self.flips['y'] else ''}",
                                   (lambda: self.flip_axis("y")), tone=SUN))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "save calib",
                                   self.save_calib, tone=CORAL))
        y += 34
        self.sliders.append(Slider((x, y, w, 18), "LED", 0, 255,
                                   lambda: self.bright, self.set_bright))
        y += 30
        self.dock_top = y

    def set_v(self, v):
        self.v_min = int(v)

    def set_min(self, v):
        self.min_area = int(v)

    def set_max(self, v):
        self.max_area = int(v)

    def set_jump(self, v):
        self.track.max_jump = int(v)

    def set_speed(self, v):
        """Commanded speed, live. Changing it while armed takes effect on the
        next tick rather than on the next path -- it is a throttle, not a
        setting, and a throttle you have to disarm to move is not a throttle."""
        self.speed = int(v)

    def set_lookahead(self, v):
        self.lookahead = int(v)

    def apply_latency_advice(self):
        """Set lookahead and arrival from the measured lag at the current speed.

        One click rather than arithmetic, because the numbers move every time
        the speed slider does and doing it by hand is how they end up stale.
        """
        plant = self.plant if self.plant else plant_constants(self.code)
        if not plant:
            self.say("no measured lag — run fleet/characterize.py first", SUN)
            return
        want = latency_advice(float(self.speed), plant)
        self.lookahead = int(np.clip(round(want["lookahead_cm"]), 2, 60))
        self.goal_tol = int(np.clip(round(want["arrive_cm"]), 2, 40))
        self.say(f"lookahead {self.lookahead}cm, arrive {self.goal_tol}cm — "
                 f"from tau {plant['tau_s']*1000:.0f}ms and "
                 f"{plant.get('coast_n', 0)} measured stops "
                 f"({plant['source']})", MINT)

    def set_goal_tol(self, v):
        """The arrival radius, live. Widening it while a ball is orbiting a
        goal it cannot satisfy is the fix for that, applied without stopping."""
        self.goal_tol = int(v)

    def toggle_view(self):
        self.view = "mask" if self.view == "raw" else "raw"

    def toggle_pause(self):
        self.paused = not self.paused

    # -- workspace and target --------------------------------------------

    def start_corners(self):
        self.mode, self.picking = "corners", []
        self.say("click the 4 corners of the workspace — esc to cancel")

    def start_target(self):
        if not self.mode == "run":
            return
        self.mode = "target"
        self.say("click the blob to track — esc to cancel")

    def clear_region(self):
        self.corners, self.region, self.mode, self.picking = None, None, "run", []
        config.save("blob_region", {"corners": []})
        self.say("workspace cleared — the whole frame is read again")

    def cancel_mode(self):
        self.mode, self.picking = "run", []
        self.say("cancelled")

    # -- paths and driving -----------------------------------------------

    PATH_HELP = {"point": "click the goal",
                 "line": "click start, then end",
                 "poly": "click each corner — enter to finish",
                 "circle": "click the centre, then a point on the rim",
                 "free": "drag to draw — release to finish"}

    def start_path(self, kind):
        """Begin drawing a path. Paths live in centimetres, so this needs a
        homography -- without one there is no arena to draw in and nothing the
        controller could command."""
        if not self.homography.ready:
            self.say("no homography — paths and driving need calib/homography.json",
                     CORAL)
            return
        self.disarm()
        self.mode, self.picking = kind, []
        self.say(self.PATH_HELP.get(kind, "click"))

    def finish_poly(self):
        if self.mode == "poly" and len(self.picking) >= 2:
            self.path = Path(self.picking, kind="polyline")
            self.mode, self.picking = "run", []
            self.say(f"polyline, {self.path.length:.0f} cm")

    def clear_path(self):
        self.disarm()
        self.path, self.target_cm, self.picking = None, None, []
        self.mode = "run"
        self.say("path erased")

    def arm(self):
        """Zero the aim if it has not been done, then drive.

        Refused unless everything it depends on is present -- and an aim that
        has been established is one of those things. Pure pursuit converges
        only while the error between the course commanded and the course
        travelled is under 90 degrees; beyond that its feedback pushes the
        wrong way and the robot spirals outward. A ball that has just connected
        has an arbitrary relationship between its own forward and the arena, so
        arming one without zeroing is a coin toss on whether it converges at
        all.

        So GO on an unzeroed robot rolls the calibration leg first and starts
        following when it lands. That is one action from the operator's side
        and it is the order the geometry requires; making it a separate step
        the person has to remember only means it eventually gets forgotten, and
        the symptom then looks like the follower being broken.
        """
        if self.path is None:
            self.say("draw a path first", SUN)
            return
        if self.fleet is None or self.code is None:
            self.say("no robot to drive", SUN)
            return
        if not self.homography.ready:
            self.say("no homography — cannot command in centimetres", CORAL)
            return
        self.plant = plant_constants(self.code)
        h = self.fleet.handles.get(self.code)
        if h is not None:
            self.zero_at_rest(h)
        self.armed = True
        # Every run begins by pointing the ball, because at the moment of
        # arming it is stationary and facing wherever it last stopped.
        self.turning = time.perf_counter() if self.style == "turn-go" else None
        self.say(f"driving {self.code} at {self.speed} cm/s — STOP or esc to halt",
                 MINT)

    COMPASS_PROBE = (0, 90, 180, 270)
    PROBE_LEG_CM = 18.0
    PROBE_SETTLE_CM = 7.0
    """Discarded at the start of each leg, and not optional.

    Every leg after the first begins with the ball TURNING out of the previous
    heading, and that arc is travel in the wrong direction. Measured from the
    standing start it put a 23 degree error on three of four legs of a
    perfectly aimed ball -- which reads as a real offset and would send
    somebody correcting a fault that is not there.
    """
    PROBE_TIMEOUT_S = 8.0

    def start_probe(self):
        """Drive each cardinal heading and report where the ball ACTUALLY went.

        The one question no amount of reading the source can answer: which way
        does this ball's compass run, relative to this camera. Every frame
        convention in this project is self-consistent on paper, and a
        reflection between any two of them looks exactly like the symptom this
        exists to diagnose -- one axis inverted, the other correct, which no
        aim error can produce because a rigid ball cannot be mirrored.

        Commands go through `drive_raw`, so the heading reaching the ball is
        the raw one with no offset applied. What comes back is a table of
        commanded heading against measured arena bearing, and from four rows
        the mapping is decided rather than guessed at.
        """
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is None:
            self.say("nothing connected to probe", SUN)
            return
        if not self.homography.ready or not self.track.locked:
            self.say("needs a homography and a locked track", SUN)
            return
        self.disarm()
        self.zero_at_rest(h)
        from fleet.handle import MAX_SPEED
        self.probe = {"i": 0,
                      "byte": int(np.clip(PROBE_SPEED / MAX_SPEED * 255, 0, 255)),
                      "rows": [], "start": None, "from": None, "at": 0.0}
        self.say("probing: driving each cardinal heading in turn")

    def step_probe(self):
        """One tick of the probe. Drives, measures, moves to the next heading."""
        p = self.probe
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is None or not self.track.locked or self.blob is None:
            if h is not None:
                h.stop()
            self.probe = None
            self.say("probe abandoned — lost the ball", CORAL)
            return
        here = np.asarray(self.to_cm(self.blob["xy"]), dtype=float)

        if p["start"] is None:
            p["start"], p["at"] = here.copy(), time.time()
            p["from"] = None
            h.drive_raw(float(self.COMPASS_PROBE[p["i"]]), p["byte"])
            return

        # Measurement begins once the ball is rolling on THIS heading, not
        # when it was told to; see `PROBE_SETTLE_CM`.
        if p["from"] is None:
            if float(np.linalg.norm(here - p["start"])) >= self.PROBE_SETTLE_CM:
                p["from"] = here.copy()
            elif time.time() - p["at"] < self.PROBE_TIMEOUT_S:
                return
            else:
                p["from"] = p["start"].copy()
        moved = here - p["from"]
        gone = float(np.linalg.norm(moved))
        if gone < self.PROBE_LEG_CM and time.time() - p["at"] < self.PROBE_TIMEOUT_S:
            return
        h.stop()
        from fleet.handle import velocity_to_command
        got = velocity_to_command(moved)[0] if gone > 4.0 else None
        p["rows"].append((self.COMPASS_PROBE[p["i"]], got, gone))
        p["i"] += 1
        p["start"] = None
        if p["i"] < len(self.COMPASS_PROBE):
            return

        self.probe = None
        self.report_probe(p["rows"])

    @staticmethod
    def mirror_axis(rows):
        """Which axis is inverted, named, from the four probe legs.

        A reflection about a line at bearing `a` sends a commanded course `c`
        to a measured one `t = 2a - c`. So every leg offers its own estimate of
        the mirror line, `a = (t + c) / 2`, and if the four agree then the axis
        is not a guess -- it is measured, the same way everything else here is.

        Halving an angle is two-valued, so the estimates live modulo 180. That
        is not a loss: a mirror line and the same line turned through 180
        degrees are the same mirror.

        A line along x inverts y, and a line along y inverts x -- reflection
        flips the component PERPENDICULAR to the mirror, which is the one thing
        about this that is easy to state backwards.
        """
        halves = [((t + w) / 2.0) % 180.0 for w, t, _ in rows if t is not None]
        if len(halves) < 3:
            return ""
        # Circular spread on a 180 degree wrap, so a set straddling 0/180 is
        # not read as maximally inconsistent.
        ref = halves[0]
        rel = [((h - ref + 90.0) % 180.0) - 90.0 for h in halves]
        if max(rel) - min(rel) > 30.0:
            return ("The four legs disagree about the mirror line, so this is "
                    "not a clean axis flip — try either flip and re-probe.")
        line = (ref + float(np.mean(rel))) % 180.0
        if min(line, 180.0 - line) < 30.0:
            return "The Y AXIS is aligned with the mirror, so X is inverted — press flip x."
        if abs(line - 90.0) < 30.0:
            return "The X AXIS is aligned with the mirror, so Y is inverted — press flip y."
        return (f"The mirror line sits at {line:.0f}°, which is neither axis — "
                "the camera is rotated as well as mirrored. Flip either, then "
                "zero the aim for the remainder.")

    def report_probe(self, rows):
        """Turn four measured legs into a verdict, printed and logged.

        The verdict is the point. Four numbers on a screen still need somebody
        to work out what they mean, and "commanded minus measured is constant"
        versus "its sign follows the axis" is precisely the distinction between
        a rotation you can correct with an offset and a reflection you cannot.
        """
        print("\n  commanded   measured   error   travelled")
        errs = []
        for want, got, gone in rows:
            if got is None:
                print(f"  {want:9.0f}   (did not move)")
                continue
            e = (got - want + 180.0) % 360.0 - 180.0
            errs.append((want, e))
            print(f"  {want:9.0f}   {got:8.1f}   {e:+6.1f}   {gone:5.1f} cm")
        if len(errs) < 3:
            self.say("probe inconclusive — the ball barely moved", CORAL)
            return
        spread = max(e for _, e in errs) - min(e for _, e in errs)
        # The magnitude comes from the FIRST leg, not the mean. That leg starts
        # from rest with nothing to turn out of, so it is the only one whose
        # measured bearing is purely the frame error; every later leg begins by
        # turning ninety degrees and carries some of that arc into its answer.
        #
        # The VERDICT still uses all four, because the thing that separates a
        # rotation from a reflection is whether the error stays put across
        # headings -- and that is a question about the spread, not the size.
        first = errs[0][1]
        if spread < 40.0:
            msg = (f"ROTATION of {first:+.0f}° — a plain aim offset, which "
                   "zeroing corrects")
            tone = MINT
        else:
            msg = (f"REFLECTION — the error swings {spread:.0f}° across the "
                   f"four headings. {self.mirror_axis(rows)} This is a frame "
                   "fault, not an aim error, and no offset fixes it")
            tone = CORAL
        print(f"  -> {msg}\n")
        self.say(msg, tone)

    def cycle_aim(self):
        """Turn the fleet's heading tracking on or off, and reset the offset.

        A switch, not a sign. The sign lives in `HEADING_SIGN`, where each
        handle declares its own and a test pins both by driving. What is worth
        being able to turn off is the correction ITSELF: pure pursuit is
        geometric and follows a path with no heading correction at all, so
        switching this off is how you find out whether a misbehaving run is the
        follower or the aiming.
        """
        self.aim_mode = "off" if self.aim_mode == "on" else "on"
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is not None:
            h.heading_tracking = (self.aim_mode == "on")
            h.heading_offset = 0.0
            h.estimator.restart()
        self.say(f"heading correction {self.aim_mode.upper()} (offset reset)",
                 MINT)
        self.build_dock()

    def flip_axis(self, axis):
        """Mirror the arena frame in x or y, in memory. `save calib` persists it.

        For a camera that delivers a mirrored image. The picture then agrees
        with itself perfectly -- the corners map cleanly, the grid lines up on
        the floor -- and every heading comes out reflected, because the image
        is a mirror of the world and nothing INSIDE the image can show that.
        Only driving the ball reveals it, which is what the probe does.

        BOTH AXES ARE OFFERED AND THEY ARE NEARLY THE SAME FIX. A mirror in x
        and a mirror in y differ by a 180 degree rotation, and a rotation is
        precisely what an aim offset absorbs -- so either one cures a
        reflection and `z` mops up the remainder. They are both here because
        which one leaves the arena's axes pointing the way you expect is a
        question about your camera, not about the maths, and trying it is
        quicker than reasoning about it.

        `vision/homography.py` implements the y case and this defers to it. The
        x case is composed here rather than added there, because a shared
        calibration module gaining a method for a bench experiment is how
        modules grow things nobody else uses.
        """
        h = self.homography
        if not h.ready:
            self.say("no homography to flip", SUN)
            return
        if axis == "y":
            h.flip_y()
        else:
            F = np.array([[-1.0, 0.0, float(h.width)],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            h.M = F @ h.M
            if h.corners:
                h.corners = [h.corners[i] for i in (1, 0, 3, 2)]
        self.flips[axis] = not self.flips[axis]
        self.track.forget()          # positions before the flip are in the old frame
        self.trail.clear()
        self.history.clear()
        self.say(f"flipped {axis} — now x:{self.flips['x']} y:{self.flips['y']}. "
                 "Re-probe to check.", MINT)
        self.build_dock()

    def save_calib(self):
        """Write the flipped homography to calib/homography.json.

        Separate from the flip, and deliberately so: that file is read by every
        other tool in this project, and a bench toggle that rewrites shared
        calibration the moment you experiment with it is how one session's
        guess becomes everyone's baseline. Flip, probe, and save only once the
        probe says the frame is clean.
        """
        if not self.homography.ready:
            self.say("nothing to save", SUN)
            return
        try:
            self.homography.save()
        except Exception as e:
            self.say(f"save failed: {type(e).__name__}: {e}", CORAL)
            return
        self.say("calib/homography.json written — every tool sees this", MINT)

    def zero_now(self):
        """Zero the aim by hand. The same thing GO does before every move."""
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is None:
            self.say("nothing connected to zero", SUN)
            return
        if self.armed:
            self.say("stop first — zeroing is done at rest", SUN)
            return
        self.zero_at_rest(h)
        self.say(self.aim_note or "zeroed", MINT)

    def zero_at_rest(self, h):
        """Declare the ball's current forward to be zero, before it moves.

        Done AT REST and once per motion, which is what makes it cheap and
        makes it mean something. At rest there is nothing to interrupt: no
        drive command is in flight to lose airtime to, and the brief drop in
        stabilisation inside `reset_aim` costs nothing to a ball that is
        already stationary. Doing the same thing mid-motion would be a packet
        competing with the drive commands and a stumble in the middle of a
        path.

        What it buys is a FRESH reference. A Sphero's notion of its own heading
        drifts, and it re-establishes that notion when it connects -- so a zero
        taken minutes ago, or before a reconnect, or before somebody picked the
        ball up, is describing a relationship that no longer holds. Taking it
        immediately before each move means the reference is never older than
        the move it belongs to.

        Nothing is measured here and nothing is corrected. Where this fresh
        zero actually points in the arena is not yet known, and is not guessed
        at: the camera answers that during the motion, in `retune_aim`.
        """
        if not getattr(h, "aim_zero", None):
            return
        try:
            if h.aim_zero(0.0):
                h.heading_offset = 0.0
                h.estimator.restart()
                self.zeroed = True
                self.aim_note = "aim zeroed at rest"
        except Exception as e:
            self.aim_note = f"could not zero: {type(e).__name__}: {e}"

    def disarm(self, note=None):
        """Stop driving, and actually send the stop.

        Clearing the flag is not enough: the robot holds its last command until
        it is given another, so a controller that merely stops COMMANDING
        leaves the ball rolling. Every path out of the driving state goes
        through here for that reason.
        """
        was = self.armed
        self.armed = False
        self.target_cm = None
        self.cmd_v = None
        if self.fleet and self.code:
            h = self.fleet.handles.get(self.code)
            if h is not None:
                try:
                    h.stop()
                except Exception:
                    pass
        if note:
            self.say(note, SUN)
        elif was:
            self.say("stopped")

    def frame_xy(self, pos):
        """Screen point -> frame pixel, or None if it is outside the picture."""
        if self._view is None or self.frame is None:
            return None
        ox, oy, scale = self._view
        x, y = (pos[0] - ox) / scale, (pos[1] - oy) / scale
        h, w = self.frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        return np.array([x, y], dtype=float)

    def dock_click(self, pos):
        """Rows and swatches on the ROBOT tab. Returns True if consumed."""
        for row, name in self.scan_rows:
            if row.collidepoint(pos):
                self.connect(name)
                return True
        for box, name in self.color_rows:
            if box.collidepoint(pos):
                self.set_robot_color(name)
                return True
        return False

    def click_view(self, pos):
        """A click on the picture. Returns True if it was consumed."""

        at = self.frame_xy(pos)
        if at is None:
            return False
        if self.mode == "corners":
            self.picking.append(at)
            if len(self.picking) < 4:
                self.say(f"corner {len(self.picking)} of 4")
                return True
            self.corners = order_quad(self.picking)
            self.region, self.mode, self.picking = None, "run", []
            config.save("blob_region",
                        {"corners": [[float(p[0]), float(p[1])]
                                     for p in self.corners]})
            # The track is dropped rather than kept: the blob it was holding
            # may be outside the new workspace, and a track that survives the
            # boundary that was just drawn around it is the opposite of what
            # drawing the boundary meant.
            self.track.forget()
            self.say("workspace set and saved — press t to pick a blob")
            return True
        if self.mode == "target":
            self.track.pick(at)
            self.mode = "run"
            self.say("tracking the blob nearest your click")
            return True

        if self.mode in self.PATH_HELP:
            cm = self.to_cm(at)
            self.picking.append(np.asarray(cm, dtype=float))
            n = len(self.picking)
            if self.mode == "point":
                self.path = Path.point(self.picking[0])
                self.mode, self.picking = "run", []
                self.say(f"goal at {self.path.pts[0][0]:.0f}, "
                         f"{self.path.pts[0][1]:.0f} cm")
            elif self.mode == "line" and n == 2:
                self.path = Path.line(*self.picking)
                self.mode, self.picking = "run", []
                self.say(f"line, {self.path.length:.0f} cm")
            elif self.mode == "circle" and n == 2:
                r = float(np.linalg.norm(self.picking[1] - self.picking[0]))
                self.path = Path.circle(self.picking[0], r)
                self.mode, self.picking = "run", []
                self.say(f"circle, radius {r:.0f} cm")
            else:
                self.say(f"{self.mode}: {n} point(s) — "
                         + self.PATH_HELP.get(self.mode, ""))
            return True
        return False

    def drag_view(self, pos):
        """Freehand drawing. Points are thinned as they are collected.

        A drag at 60fps produces a point every pixel or two, and a path with
        thousands of near-identical points costs the follower's projection
        scan for nothing while adding no shape. Two centimetres is finer than
        the ball is wide, so nothing you could actually drive is lost.
        """
        if self.mode != "free":
            return
        at = self.frame_xy(pos)
        if at is None:
            return
        cm = np.asarray(self.to_cm(at), dtype=float)
        if self.picking and float(np.linalg.norm(cm - self.picking[-1])) < 2.0:
            return
        self.picking.append(cm)

    def finish_free(self):
        if self.mode != "free":
            return
        if len(self.picking) >= 2:
            self.path = Path(self.picking, kind="freehand")
            self.say(f"freehand, {self.path.length:.0f} cm "
                     f"in {len(self.path.pts)} points")
        else:
            self.say("too short to be a path", SUN)
        self.mode, self.picking = "run", []


    # -- zeroing the aim ---------------------------------------------------

    def turn_and_go(self, v):
        """Point the ball at the target, confirm it, then drive it straight.

        Two states and a camera between them.

        TURN holds a crawl in the target direction. It is not stationary
        because a blob has no facing when it is not moving -- so the only way
        to know the ball has finished turning is to watch which way it is
        creeping. That is the whole reason this mode has a speed floor rather
        than a rotate-in-place step.

        GO drives the operator's speed in the same direction, and hands back to
        TURN only when the bearing error grows past `RETURN_DEG`. That band is
        deliberately wide: narrowing it turns every small correction into a
        stop-and-turn, which gives back precisely the smoothness this mode
        exists to provide.

        Returns the velocity to command; the caller still slews and sends it.
        """
        want = float(np.degrees(np.arctan2(v[1], v[0])))
        tr = self.travel_readout()
        moving = tr is not None and tr[2] >= AIM_MIN_SPEED_CM_S
        error = None
        if moving:
            error = abs((tr[0] - want + 180.0) % 360.0 - 180.0)

        if self.turning is None:
            # Not turning: keep going unless the target has swung away.
            if error is not None and error > RETURN_DEG:
                self.turning = time.perf_counter()
                self.drive_note = f"turning ({error:.0f}° off)"
            else:
                return v

        elapsed = time.perf_counter() - self.turning
        if (error is not None and error <= TURN_OK_DEG) or elapsed > TURN_MAX_S:
            self.turning = None
            self.drive_note = ("driving straight" if error is None
                               else f"driving straight ({error:.0f}° off)")
            return v
        self.drive_note = (f"turning… {elapsed:.1f}s"
                           + ("" if error is None else f", {error:.0f}° off"))
        n = float(np.linalg.norm(v))
        return v / n * TURN_SPEED_CM_S if n > 1e-9 else v

    def cycle_style(self):
        """Swap between steering continuously and turn-then-go."""
        self.style = STYLES[(STYLES.index(self.style) + 1) % len(STYLES)]
        self.turning = None
        self.say(f"drive style: {self.style}", MINT)
        self.build_dock()

    def slew(self, v):
        """Rate-limit how fast the commanded DIRECTION may turn.

        Position noise becomes heading noise: pure pursuit's effective gain
        rises as the lookahead shortens, so a jittery fix becomes a jittery
        command, and a Sphero physically rotates its drive assembly to follow
        it. Limiting how fast the command may swing costs a little sharpness on
        a hairpin and takes the buzz out of everything else.

        Only the direction is limited, and only its rate. The magnitude is the
        operator's speed slider and has no business being smoothed behind their
        back, and a step change in the target is still followed -- just over a
        few frames instead of in one.
        """
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return v
        now = time.perf_counter()
        want = float(np.degrees(np.arctan2(v[1], v[0])))
        if self._slew_at is None or self._slew_deg is None:
            self._slew_at, self._slew_deg = now, want
            return v
        dt = max(now - self._slew_at, 1e-3)
        step = (want - self._slew_deg + 180.0) % 360.0 - 180.0
        limit = SLEW_DEG_S * dt
        if abs(step) > limit:
            step = limit if step > 0 else -limit
        self._slew_deg = (self._slew_deg + step) % 360.0
        self._slew_at = now
        rad = np.radians(self._slew_deg)
        return np.array([np.cos(rad), np.sin(rad)]) * n

    def log_state(self):
        """One CSV row for this frame. See `RunLog` for what the columns mean."""
        from fleet.handle import velocity_to_command
        blob, tr = self.blob, self.travel_readout()
        cm = self.to_cm(blob["xy"]) if blob is not None else None
        cmd_deg = None
        rel = None
        if self.cmd_v is not None and float(np.linalg.norm(self.cmd_v)) > 1e-6:
            cmd_deg, _ = velocity_to_command(self.cmd_v)
            if tr is not None and tr[1] is not None:
                # Travel minus command, both as compass bearings. Non-zero here
                # is aim error still outstanding -- which is the number the
                # zeroing routine exists to drive to zero.
                rel = (tr[1] - cmd_deg + 180.0) % 360.0 - 180.0
        mode = ("driving" if self.armed
                else "manual" if self.manual else "idle")
        # Where it is being SENT, and where it is steering right now. The end
        # of the path is the goal; a closed path has no end and so no goal,
        # which is honest rather than a gap -- a circle is a patrol.
        goal = None
        if self.path is not None and self.path.pts and not self.path.closed:
            goal = self.path.pts[-1]
        gap = (float(np.linalg.norm(np.asarray(cm, dtype=float) - goal))
               if (goal is not None and cm is not None
                   and self.homography.ready) else None)

        self.log.write(
            frame=self.seen, mode=mode, status=self.track.status,
            goal_x=None if goal is None else float(goal[0]),
            goal_y=None if goal is None else float(goal[1]),
            gap_cm=gap,
            target_x=None if self.target_cm is None else float(self.target_cm[0]),
            target_y=None if self.target_cm is None else float(self.target_cm[1]),
            style=self.style,
            set_speed=int(self.speed),
            set_lookahead=int(self.lookahead),
            set_arrive=int(self.goal_tol),
            x_px=None if blob is None else float(blob["xy"][0]),
            y_px=None if blob is None else float(blob["xy"][1]),
            x_cm=None if (cm is None or not self.homography.ready) else float(cm[0]),
            y_cm=None if (cm is None or not self.homography.ready) else float(cm[1]),
            theta_global_deg=None if tr is None else tr[0],
            theta_global_cmp=None if tr is None else tr[1],
            theta_rel_deg=rel,
            speed_cm_s=None if tr is None else tr[2],
            noise_mm=self.jitter_mm,
            parts=self.parts,
            area=None if blob is None else blob["area"],
            peak=None if blob is None else blob["peak"],
            clipped=None if blob is None else blob.get("clipped_by_region"),
            path=None if self.path is None else self.path.kind,
            cmd_vx=None if self.cmd_v is None else float(self.cmd_v[0]),
            cmd_vy=None if self.cmd_v is None else float(self.cmd_v[1]),
            cmd_deg=cmd_deg,
            aim_offset_deg=self.aim_offset(),
            note=self.drive_note or None)

    def aim_offset(self):
        if not self.fleet or not self.code:
            return None
        h = self.fleet.handles.get(self.code)
        return None if h is None else float(h.heading_offset)

    # -- manual drive ------------------------------------------------------

    MANUAL_KEYS = {pygame.K_UP: (0.0, -1.0), pygame.K_w: (0.0, -1.0),
                   pygame.K_DOWN: (0.0, 1.0), pygame.K_s: (0.0, 1.0),
                   pygame.K_LEFT: (-1.0, 0.0), pygame.K_a: (-1.0, 0.0),
                   pygame.K_RIGHT: (1.0, 0.0), pygame.K_d: (1.0, 0.0)}
    """Arrow keys or WASD, in ARENA directions -- up is -y because the workspace
    frame has y running down, matching the camera. Deliberately the arena's
    frame and not the ball's: pressing up should send it up the picture, which
    is the only mapping a person can use without thinking about aim."""

    def manual_drive(self):
        """Drive by hand while the ROBOT tab is open. Returns True if driving.

        Needs no tracker lock, unlike the follower. That is the point: the
        moment you most want to nudge a ball by hand is when the tracker has
        LOST it -- it has rolled out of the workspace, or under something --
        and a manual control that refused exactly then would be useless.

        The pursuit controller is not allowed to run at the same time, and
        touching a key disarms it. Two things steering one ball is how a
        controller gets blamed for a person's input.
        """
        if self.tab != "robot":
            return False
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is None:
            return False
        held = pygame.key.get_pressed()
        want = np.zeros(2)
        for key, direction in self.MANUAL_KEYS.items():
            if held[key]:
                want += np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(want))
        if n < 1e-9:
            if self.manual:
                # Released: stop, once. A key-up that only stops COMMANDING
                # leaves the ball rolling on its last order.
                h.stop()
                self.manual = False
                self.cmd_v = None
            return False
        if self.armed:
            self.disarm("manual drive took over")
        v = want / n * float(self.speed)
        h.set_velocity(v)
        self.cmd_v = v
        self.manual = True
        return True

    # -- driving ----------------------------------------------------------

    def drive(self):
        """One control tick: chase the path, and correct the aim while moving.

        THE SAFETY RULE, and it is the reason this is not simply a call to
        `pursue`: the robot is commanded only while the tracker is LOCKED. A
        coasted position is a prediction, and driving on a prediction is how a
        ball ends up somewhere nobody was watching -- so a lost or coasting
        track stops the robot rather than steering it from a guess. That is
        also why `disarm` sends a stop instead of merely ceasing to command:
        the ball holds its last order until given another.
        """
        if self.probe is not None:
            self.step_probe()
            return
        if self.manual_drive():
            return
        if not self.armed:
            return
        h = self.fleet.handles.get(self.code) if self.fleet else None
        if h is None:
            self.disarm("robot went away")
            return
        # A DROPPED FRAME IS NOT A REASON TO STOP, and treating it as one was
        # the largest single source of jerk in this controller. Stopping the
        # wheels the instant the track misses a frame, then driving again when
        # it returns, turns an ordinary flicker into stop-go-stop several times
        # a second -- which on the floor reads as a badly tuned follower rather
        # than as the tracker blinking.
        #
        # The tracker already predicts through a short dropout, so the
        # controller rides that prediction for a few frames and stops only when
        # the track is genuinely lost. At 25cm/s three frames is under three
        # centimetres of travel on a guess, which is a far smaller risk than
        # the alternative of hammering the motors.
        if self.track.xy is None or self.track.misses > DRIVE_ON_PREDICTION:
            h.stop()
            self.cmd_v = None
            self.drive_note = f"halted: {self.track.status}"
            return
        if self.blob is not None:
            pos = np.asarray(self.to_cm(self.blob["xy"]), dtype=float)
        else:
            pos = np.asarray(self.to_cm(self.track.predicted), dtype=float)
        v, target, done, note = pursue(self.path, pos, self.lookahead,
                                       self.speed, self.goal_tol)
        self.target_cm, self.drive_note = target, note
        if done:
            h.stop()
            self.disarm("arrived — stopped")
            return
        if self.style == "turn-go":
            v = self.turn_and_go(v)
        v = self.slew(v)
        h.set_velocity(v)
        self.cmd_v = np.asarray(v, dtype=float)

        # Imported here, not at module scope: this app must start on a machine
        # where the fleet layer is broken.
        from fleet.handle import velocity_to_command
        self.cmd_log.append((time.perf_counter(), velocity_to_command(v)[0]))

        # Heading correction, fed to the estimator `fleet` already owns.
        #
        # NOT a second loop of this app's own, which is what used to be here
        # and is worth recording: `SimRobot.step` already calls
        # `observe_heading` every step, so the fleet's fold was running the
        # whole time. Two controllers correcting one quantity, with different
        # gains and different ideas of the lag, is unstable whichever sign
        # either uses -- the ball spiralled away under BOTH, which reads as a
        # sign fault and is not one.
        #
        # The estimator also has the gates a naive difference lacks: it rejects
        # samples with too short a baseline, samples taken while turning, and
        # outliers. Those are what stop the controller's own steering lag being
        # mistaken for a frame error.
        try:
            if getattr(h, "kind", "") != "sim":
                # Only for a real robot, where `pos` is a CACHE of the last
                # camera fix and this stands in for the tracker that would
                # normally fill it. On a simulated robot `pos` IS the state, and
                # writing a camera position into it rewinds the robot to where
                # it was when that frame was rendered.
                h.pos = pos
                h.observe_heading(time.time())
        except Exception as e:
            self.drive_note = f"heading feed failed: {type(e).__name__}: {e}"

    # -- drawing ----------------------------------------------------------

    def text(self, s, x, y, col=CHALK, font=None):
        font = font or self.f
        self.screen.blit(font.render(s, True, col), (x, y))
        return y + font.get_height() + 2

    def sect(self, surface, font, label, x, y, w, right=None, tone=None):
        """A section header that also yields to the key hints."""
        if y > H - self.HINT_H - 24:
            return y
        return section(surface, font, label, x, y, w, right, tone)

    HINT_H = 44
    """Pixels reserved at the bottom of the dock for the key hints."""

    def row(self, s, x, y, col=CHALK, font=None):
        """A dock line that gives up rather than overlapping the key hints.

        The dock grew a section at a time and eventually ran past the bottom of
        the window, printing the control readout straight through the shortcut
        list. Dropping the overflow is the honest failure: a line that is not
        drawn is obviously missing, where two lines drawn on top of each other
        are both unreadable and neither looks absent.
        """
        if y > H - self.HINT_H - (font or self.f).get_height():
            return y
        return self.text(s, x, y, col, font)

    def draw_view(self):
        r = pygame.Rect(PAD, PAD, W - DOCK_W - 2 * PAD, H - 2 * PAD)
        pygame.draw.rect(self.screen, CARD, r, border_radius=5)
        if self.frame is None:
            self.text(self.cam.error or "waiting for the first frame…",
                      r.x + 16, r.y + 16, CORAL if self.cam.error else DIM)
            return

        img = self.frame
        if self.view == "mask" and self.mask is not None:
            img = cv2.cvtColor(self.mask * 255, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min(r.w / w, r.h / h)
        ox, oy = r.x + (r.w - w * scale) / 2, r.y + (r.h - h * scale) / 2
        self.screen.blit(pygame.transform.smoothscale(
            to_surface(img), (int(w * scale), int(h * scale))), (ox, oy))

        self._view = (ox, oy, scale)

        def px(p):
            return int(ox + p[0] * scale), int(oy + p[1] * scale)

        # The workspace boundary, and the corners collected so far while one is
        # being clicked. Drawn under everything else.
        if self.corners is not None:
            pygame.draw.polygon(self.screen, CYAN,
                                [px(p) for p in self.corners], 1)
        for i, p in enumerate(self.picking):
            pygame.draw.circle(self.screen, SUN, px(p), 5)
            self.text(str(i + 1), px(p)[0] + 8, px(p)[1] - 6, SUN, self.fs)
        if len(self.picking) > 1:
            pygame.draw.lines(self.screen, SUN, False,
                              [px(p) for p in self.picking], 1)

        # Where the ball has been. Drawn first so everything else sits over it,
        # and drawn at all because a track that is quietly jumping looks fine
        # in a single frame and obvious as a trail.
        if len(self.trail) > 1:
            pts = [px(p) for _, p in self.trail]
            pygame.draw.lines(self.screen, RULE, False, pts, 1)

        # The path, in cm, mapped back to pixels for drawing. Drawn from the
        # SAME points the controller follows, so what you see is what is being
        # chased rather than a second rendering that could drift from it.
        def path_px(pts):
            return [px(self.homography.to_px([q])[0]) for q in pts]

        if self.homography.ready:
            if self.path is not None and len(self.path.pts) > 1:
                pygame.draw.lines(self.screen, CYAN, self.path.closed,
                                  path_px(self.path.pts), 2)
            elif self.path is not None:
                pygame.draw.circle(self.screen, CYAN,
                                   path_px(self.path.pts)[0], 7, 2)
            # The arrival radius, drawn where the run actually ends. An
            # acceptance test you cannot see is one you tune by guesswork --
            # and this is the number that decides between arriving and
            # orbiting, so it belongs on the picture next to the ball it has to
            # contain. Closed paths have no end, so nothing is drawn for them.
            if self.path is not None and not self.path.closed:
                end = self.path.pts[-1]
                mid = path_px([end])[0]
                edge = path_px([end + np.array([self.goal_tol, 0.0])])[0]
                # Named `ring`, not `r`: `r` is the view rectangle in this
                # scope, and shadowing it crashed the app on the one path that
                # reads it afterwards -- losing the ball while a path is drawn.
                ring = int(max(4, abs(edge[0] - mid[0])))
                pygame.draw.circle(self.screen, RULE, mid, ring, 1)
            if len(self.picking) > 1 and self.mode in self.PATH_HELP:
                pygame.draw.lines(self.screen, SUN, False,
                                  path_px(self.picking), 1)
            for q in (self.picking if self.mode in self.PATH_HELP else []):
                pygame.draw.circle(self.screen, SUN, path_px([q])[0], 4)
            if self.target_cm is not None and self.armed:
                # The pursuit point. Watching this run ahead of the ball is how
                # the lookahead slider stops being an abstract number.
                tp = path_px([self.target_cm])[0]
                pygame.draw.circle(self.screen, MINT, tp, 8, 2)
                if self.blob is not None:
                    pygame.draw.line(self.screen, MINT, px(self.blob["xy"]),
                                     tp, 1)

        # Candidates the gate turned down. Grey, and shown rather than dropped:
        # a rejected blob sitting where the ball actually is means the gate is
        # too tight, and that is only diagnosable if you can see it happen.
        for b in self.others:
            pygame.draw.circle(self.screen, GREY, px(b["xy"]), 9, 1)

        if self.mode == "target":
            # Every candidate, ringed, so there is something to aim at. The
            # tracked one is not privileged here -- the whole point of the mode
            # is to change which one that is.
            for b in ([self.blob] if self.blob is not None else []) + self.others:
                pygame.draw.circle(self.screen, SUN, px(b["xy"]), 14, 2)

        blob = self.blob
        coasting = blob is None and self.track.xy is not None
        at = blob["xy"] if blob is not None else self.track.predicted
        if at is not None:
            c = px(at)
            area = blob["area"] if blob is not None else 400
            rad = max(12, int(np.sqrt(area / np.pi) * scale * 1.7))
            # A coasted position is a GUESS, and it must not look like a fix.
            # Dashed and dim where a measured one is solid: the marker has to
            # lose confidence exactly when the tracker does, or it is lying.
            if coasting:
                for a0 in range(0, 360, 30):
                    if (a0 // 30) % 2:
                        continue
                    rect = pygame.Rect(c[0] - rad, c[1] - rad, rad * 2, rad * 2)
                    pygame.draw.arc(self.screen, SUN, rect,
                                    np.radians(a0), np.radians(a0 + 30), 2)
            else:
                pygame.draw.circle(self.screen, self.marker, c, rad, 2)
                pygame.draw.circle(self.screen, self.marker, c, 3)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    pygame.draw.line(self.screen, self.marker,
                                     (c[0] + dx * (rad - 5), c[1] + dy * (rad - 5)),
                                     (c[0] + dx * (rad + 5), c[1] + dy * (rad + 5)), 1)
            # The travel arrow, and only while it is actually travelling.
            # Absent at rest on purpose: at a standstill the fitted direction
            # is a reading of the noise, and an arrow that spins on the spot
            # invites you to believe a bearing that means nothing.
            got = self.travel if not coasting else None
            if got is not None:
                v, speed = got
                d = v / max(float(np.linalg.norm(v)), 1e-9)
                tail = (c[0] + d[0] * rad, c[1] + d[1] * rad)
                length = rad + min(speed * 0.20, rad * 2.5)
                tip = (c[0] + d[0] * length, c[1] + d[1] * length)
                pygame.draw.line(self.screen, MINT, tail, tip, 2)
                for side in (150, -150):
                    a = np.arctan2(d[1], d[0]) + np.radians(side)
                    pygame.draw.line(self.screen, MINT, tip,
                                     (tip[0] + np.cos(a) * 9,
                                      tip[1] + np.sin(a) * 9), 2)

            cm = self.to_cm(at)
            unit = "cm" if self.homography.ready else "px"
            name = self.code or "blob"
            label = (f"{name}  {cm[0]:.1f}, {cm[1]:.1f} {unit}" if unit == "cm"
                     else f"{name}  {cm[0]:.0f}, {cm[1]:.0f} {unit}")
            if coasting:
                label += "  (coasting)"
            self.text(label, c[0] + rad + 8, c[1] - 8,
                      SUN if coasting else self.marker, self.fs)
        else:
            self.text(self.why or "no blob", r.x + 16, r.y + 16, SUN)


    def draw_dock(self):
        x, w = W - DOCK_W + PAD, DOCK_W - 2 * PAD
        pygame.draw.rect(self.screen, PANEL, (W - DOCK_W, 0, DOCK_W, H))
        pygame.draw.line(self.screen, RULE, (W - DOCK_W, 0), (W - DOCK_W, H))
        for s_ in self.sliders:
            s_.draw(self.screen, self.fs)
        for b in self.buttons:
            b.draw(self.screen, self.fs)
        (self.draw_track_tab if self.tab == "track"
         else self.draw_robot_tab)(x, w, self.dock_top)
        self.text("1 pt 2 line 3 poly 4 circ 5 free  g go", x, H - 36,
                  GREY, self.fs)
        self.text("c corners t target b scan  tab  esc STOP", x, H - 22,
                  GREY, self.fs)

    def draw_track_tab(self, x, w, y):
        y = self.sect(self.screen, self.fs, "position", x, y, w,
                      self.track.status.split(" —")[0],
                      MINT if self.track.locked else SUN)
        if self.blob is not None:
            cm = self.to_cm(self.blob["xy"])
            unit = "cm" if self.homography.ready else "px"
            box = pygame.Rect(x, y, w, 62)
            card(self.screen, box)
            self.text(f"{cm[0]:8.1f}", x + 10, y + 7, CHALK, self.fb)
            self.text(f"{cm[1]:8.1f}", x + 10, y + 33, CHALK, self.fb)
            self.text(f"x {unit}", x + 132, y + 13, DIM, self.fs)
            self.text(f"y {unit}", x + 132, y + 39, DIM, self.fs)
            y += 68
            j = self.jitter_mm
            y = self.row(f"noise {j:.2f} mm over {len(self.history)} frames"
                         if j is not None else "noise — (needs a homography)",
                         x, y, MINT if (j or 0) < 3 else SUN, self.fs)
            tr = self.travel_readout()
            if tr is None:
                y = self.row("travel — at rest", x, y, DIM, self.fs)
            else:
                deg, comp, speed = tr
                us = "cm/s" if self.homography.ready else "px/s"
                y = self.row(f"travel {deg:5.1f}° img" +
                             (f"  {comp:5.1f}° cmp" if comp is not None else ""),
                             x, y, MINT, self.fs)
                y = self.row(f"speed  {speed:5.1f} {us}", x, y, MINT, self.fs)
        else:
            y = self.row(self.why or "no blob", x, y, SUN, self.fs)
        y += 8

        y = self.sect(self.screen, self.fs, "frame", x, y, w)
        tone = MINT if self.parts == 1 else (SUN if self.parts else CORAL)
        scope = "in workspace" if self.corners is not None else "whole frame"
        y = self.row(f"bright regions  {self.parts}   ({scope})", x, y, tone,
                     self.fs)
        if self.blob is not None and self.blob.get("clipped_by_region"):
            y = self.row("blob touches the workspace edge — centre pulled inward",
                         x, y, CORAL, self.fs)
        if self.blob is not None:
            y = self.row(f"area {self.blob['area']}   peak {self.blob['peak']:.0f}"
                         + ("  CLIPPED" if self.blob["peak"] >= 250 else ""),
                         x, y, CORAL if self.blob["peak"] >= 250 else DIM, self.fs)
        if self.others:
            y = self.row(f"{len(self.others)} other blob(s) refused by the gate",
                         x, y, SUN, self.fs)
        y = self.row("exposure LOCKED" if self.exposure_locked else
                     "exposure on auto — it will fight you", x, y,
                     MINT if self.exposure_locked else SUN, self.fs)
        y += 8

        y = self.sect(self.screen, self.fs, "control", x, y, w,
                      "DRIVING" if self.armed else "idle",
                      MINT if self.armed else DIM)
        if self.path is None:
            y = self.row("no path — 1 pt  2 line  3 poly  4 circ  5 free",
                         x, y, DIM, self.fs)
        else:
            shut = "closed" if self.path.closed else "open"
            y = self.row(f"{self.path.kind}  {self.path.length:.0f} cm  "
                         f"{len(self.path.pts)} pts  {shut}", x, y, CYAN, self.fs)
        if self.drive_note:
            y = self.row(self.drive_note, x, y,
                         CORAL if "halted" in self.drive_note else DIM, self.fs)
        y = self.row(f"aim fix {self.aim_mode}   {self.aim_note}", x, y,
                     DIM if self.aim_mode == "off" else MINT, self.fs)

        # What the MEASURED lag says these knobs should be. The two most
        # common ways to make this controller misbehave are a lookahead
        # shorter than the ball's response distance and an arrival radius
        # inside its dead-time distance, and both are invisible without this.
        plant = self.plant if self.plant else plant_constants(self.code)
        if plant:
            want = latency_advice(float(self.speed), plant)
            y = self.row(f"tau {plant['tau_s']*1000:.0f}ms   coasts "
                         f"{float(self.speed) * (plant.get('coast_s') or 0.5):.0f}cm "
                         f"at this speed", x, y, DIM, self.fs)
            look_bad = self.lookahead < want["lookahead_cm"] * 0.8
            arr_bad = self.goal_tol < want["arrive_cm"] * 0.8
            y = self.row(f"wants lookahead >= {want['lookahead_cm']:.0f}cm"
                         + ("  <-- LOW" if look_bad else ""), x, y,
                         SUN if look_bad else MINT, self.fs)
            y = self.row(f"wants arrive >= {want['arrive_cm']:.0f}cm"
                         + ("  <-- LOW" if arr_bad else ""), x, y,
                         SUN if arr_bad else MINT, self.fs)
            if plant["source"] != self.code:
                y = self.row(f"(measured on {plant['source']})", x, y, DIM,
                             self.fs)
        report = self.link_report()
        if report:
            for line in wrap(report[0], 44):
                y = self.row(line, x, y, report[1], self.fs)
        if self.note:
            y += 6
            for line in wrap(self.note, 42):
                y = self.row(line, x, y, self.note_tone, self.fs)

    def draw_robot_tab(self, x, w, y):
        self.scan_rows, self.color_rows = [], []
        y = self.sect(self.screen, self.fs, "link", x, y, w)
        report = self.link_report()
        if report:
            for line in wrap(report[0], 44):
                y = self.row(line, x, y, report[1], self.fs)
        else:
            y = self.row("nothing connected — press scan", x, y, DIM, self.fs)
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if h is not None:
            st = getattr(h, "heading_state", lambda: {})() or {}
            y = self.row(f"aim offset {h.heading_offset:5.1f}°   "
                         f"samples {st.get('samples', 0)}", x, y,
                         MINT if abs(h.heading_offset) < 1.0 else DIM, self.fs)
        y += 8

        y = self.sect(self.screen, self.fs, "found", x, y, w,
                      "scanning…" if (self.scan and not self.scan.finished)
                      else None, SUN)
        if self.scan is None:
            y = self.row("press scan to look for balls", x, y, DIM, self.fs)
        elif not self.scan.finished:
            y = self.row("listening, about 7 seconds", x, y, SUN, self.fs)
        elif self.scan.error:
            for line in wrap(self.scan.error, 44):
                y = self.row(line, x, y, SUN, self.fs)
        else:
            y = self.row("click one to connect", x, y, DIM, self.fs)
            for name in self.scan.names:
                if y > H - self.HINT_H - 30:
                    break
                row = pygame.Rect(x, y, w, 24)
                live = (h is not None and getattr(h, "ble_name", None) == name)
                card(self.screen, row)
                pygame.draw.rect(self.screen, MINT if live else CARD_EDGE,
                                 row, 1, border_radius=5)
                self.text(name, row.x + 10, row.y + 5,
                          MINT if live else CHALK, self.fs)
                self.scan_rows.append((row, name))
                y = row.bottom + 4
        y += 8

        y = self.sect(self.screen, self.fs, "colour", x, y, w,
                      self.robot_color, self.marker)
        # Swatches drawn from the same palette the detector and the LED share,
        # so what you click, what the ball glows and what the overlay draws are
        # one value rather than three that can drift apart.
        sw = (w - 5 * 6) // 6
        for i, name in enumerate(config.COLORS):
            box = pygame.Rect(x + i * (sw + 6), y, sw, 26)
            draw = config.COLORS[name]["draw"]
            pygame.draw.rect(self.screen, (draw[2], draw[1], draw[0]), box,
                             border_radius=4)
            if name == self.robot_color:
                pygame.draw.rect(self.screen, CHALK, box, 2, border_radius=4)
            self.color_rows.append((box, name))
        y += 34
        y = self.row("blue is the taillight's colour — a ball tagged blue "
                     "cannot be told from its own tail", x, y, DIM, self.fs) \
            if self.robot_color == "blue" else y
        y = self.row("taillight forced off (it biases the centre)", x, y,
                     DIM, self.fs)
        y += 8
        y = self.sect(self.screen, self.fs, "manual", x, y, w,
                      "DRIVING" if self.manual else None, MINT)
        y = self.row("arrows or WASD to drive — release to stop",
                     x, y, MINT if self.manual else DIM, self.fs)
        y = self.row(f"at the speed slider: {self.speed} cm/s", x, y, DIM,
                     self.fs)
        y += 8
        y = self.sect(self.screen, self.fs, "log", x, y, w)
        if self.log.error:
            for line in wrap(f"log failed: {self.log.error}", 44):
                y = self.row(line, x, y, CORAL, self.fs)
        else:
            y = self.row(f"{self.log.name}", x, y, DIM, self.fs)
            y = self.row(f"{self.log.rows} rows", x, y, DIM, self.fs)
        if self.note:
            y += 6
            for line in wrap(self.note, 42):
                y = self.row(line, x, y, self.note_tone, self.fs)

    def draw(self):
        self.screen.fill(INK)
        self.draw_view()
        self.draw_dock()
        pygame.display.flip()

    # -- events -----------------------------------------------------------

    def key(self, e):
        # On the ROBOT tab W/A/S/D are the manual drive, so the shortcuts that
        # share those letters are suppressed there rather than silently doing
        # two things at once.
        if self.tab == "robot" and e.key in self.MANUAL_KEYS:
            return True
        # Escape is the panic key first and a cancel second. Whatever else is
        # going on, it stops the robot.
        if e.key == pygame.K_ESCAPE and (self.armed or self.probe):
            self.probe = None
            self.disarm("halted by esc")
            return True
        if e.key == pygame.K_ESCAPE and self.scan is not None:
            self.dismiss_scan()
            return True
        if e.key == pygame.K_ESCAPE and self.mode != "run":
            self.cancel_mode()
            return True
        if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.finish_poly()
            return True
        if e.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        actions = {pygame.K_s: self.save_frame,
                   pygame.K_l: lambda: self.exposure_mode(True),
                   pygame.K_u: lambda: self.exposure_mode(False),
                   pygame.K_v: self.toggle_view,
                   pygame.K_SPACE: self.toggle_pause,
                   pygame.K_c: self.start_corners,
                   pygame.K_t: self.start_target,
                   pygame.K_x: self.clear_region,
                   pygame.K_g: self.arm,
                   pygame.K_d: self.clear_path,
                   pygame.K_1: lambda: self.start_path("point"),
                   pygame.K_2: lambda: self.start_path("line"),
                   pygame.K_3: lambda: self.start_path("poly"),
                   pygame.K_4: lambda: self.start_path("circle"),
                   pygame.K_5: lambda: self.start_path("free"),
                   pygame.K_b: self.start_scan,
                   pygame.K_z: self.zero_now,
                   pygame.K_p: self.start_probe,
                   pygame.K_k: self.cycle_aim,
                   pygame.K_y: self.cycle_style,
                   pygame.K_TAB: lambda: self.set_tab(
                       "robot" if self.tab == "track" else "track")}
        if e.key in actions:
            actions[e.key]()
        elif e.key == pygame.K_LEFTBRACKET:
            self.set_exposure(max(-13, self.exposure - 1))
        elif e.key == pygame.K_RIGHTBRACKET:
            self.set_exposure(min(0, self.exposure + 1))
        return True

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    globals()["W"] = max(MIN_W, e.w)
                    globals()["H"] = max(MIN_H, e.h)
                    self.screen = pygame.display.set_mode((W, H),
                                                          pygame.RESIZABLE)
                    self.build_dock()
                elif e.type == pygame.KEYDOWN:
                    running = self.key(e)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if self.dock_click(e.pos):
                        continue
                    if (self.mode != "run" or self.scan_rows) \
                            and self.click_view(e.pos):
                        continue
                    for s in self.sliders:
                        if s.hit(e.pos):
                            break
                    else:
                        for b in self.buttons:
                            if b.hit(e.pos):
                                break
                elif e.type == pygame.MOUSEBUTTONUP:
                    if self.mode == "free":
                        self.finish_free()
                    for s in self.sliders:
                        s.dragging = False
                elif e.type == pygame.MOUSEMOTION:
                    if self.mode == "free" and e.buttons[0]:
                        self.drag_view(e.pos)
                    for s in self.sliders:
                        if s.dragging:
                            s.drag(e.pos)
            self.tick()
            self.draw()
            self.clock.tick(60)
        self.close()

    def close(self):
        # Before anything else: a window that closes while the ball is still
        # rolling is the worst possible exit.
        self.disarm()
        self.log.close()
        self.cam.close()
        if self.fleet:
            try:
                self.fleet.close()
            except Exception:
                pass
        pygame.quit()


def wrap(text, width):
    out, line = [], ""
    for word in str(text).split():
        if len(line) + len(word) + 1 > width and line:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0",
                   help="camera index, a video file, or 'sim'")
    p.add_argument("--source", dest="camera", help="same as --camera")
    p.add_argument("--size", default=None, help="e.g. 1280x720")
    p.add_argument("--exposure", type=int, default=-7)
    p.add_argument("--robot", default=None,
                   help="connect this roster entry at startup instead of scanning")
    p.add_argument("--no-fleet", action="store_true")
    a = p.parse_args(argv)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    BlobTest(a.camera, size=size, exposure=a.exposure, code=a.robot,
             with_fleet=not a.no_fleet).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
