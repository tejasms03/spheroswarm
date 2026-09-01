#!/usr/bin/env python3
"""LLM-driven fork of `swarm_test.py`. Keep `swarm_test.py` as the fallback.

Same lineage as before: `blob_test.py` is the single-robot version that works
on hardware, `swarm_test.py` is the multi-robot one, and this is where an
agent gets to drive them. Each copy exists so the one below it stays usable
while the one above is broken.

--- inherited docstring follows ---

MANY bots, blobs, and which is which. A fork of `blob_test.py`.

WHY THIS IS A COPY AND NOT A FLAG. `blob_test.py` works on the hardware and is
the thing to fall back to; multi-robot needs N tracks, gated assignment,
contention flagging and a blink roll call, and threading all of that through
the single-robot app as options would put the working tool one bad branch away
from the experimental one. The duplication is the point: this file may break
freely.

Everything below is inherited from `blob_test.py` and is unchanged until it
needs to change. What differs so the two cannot tread on each other:

    the window caption, so you can tell them apart
    the run log prefix, `swarm_` rather than `blob_`

What is deliberately SHARED, because it describes the room rather than the
program: `calib/blob_region.json` (the workspace quad) and
`calib/homography.json`. Click your corners once, use them in both. A flip
saved in either is a fact about the camera and belongs to both.

--- inherited docstring follows ---

One bot, one blob, x and y. The least processing that can be trusted.

    python3.13 fleet_test.py --camera 0
    python3.13 fleet_test.py --source sim        # no camera, no robot

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
import json
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
from vision.shots import BALL_CM

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

BRIGHT_PEAK = 200.0
"""Peak value at which a blob is taken to contain an actual LED core.

Not a detection threshold -- `V_MIN` is that, and it stays low so the halo
survives as one component. This only ranks: a lamp pointed at a sensor clips,
scenery does not. 200 rather than 255 because a ball at the far edge of the
frame, or one whose LED has been dimmed a little, still cores well above
anything reflected.
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
               region=None, grow_px=0):
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
    detect = region
    if region is not None:
        # DETECT on a region grown by about one halo radius, then ACCEPT on
        # the true one.
        #
        # Masking the pixels at the true boundary is what used to happen, and
        # it biased the answer exactly where the rig already had trouble. The
        # centroid is intensity-weighted over whatever survives the mask, so a
        # ball straddling the line contributes only its inner half and the
        # reported centre is dragged inwards, away from the wall. Measured on
        # a real frame by sliding a boundary across a real ball: with the edge
        # on the ball's centre the position was 3.6cm wrong, at 19% of the
        # blob left it was 7.1cm wrong, and past that the remnant fell under
        # `min_area` and the track died outright. The error is always inward,
        # so as the ball drives at a wall its reported position SATURATES --
        # the controller sees commanded motion producing no measured motion,
        # which reads as a stall when the ball is moving perfectly well.
        #
        # Growing the detection mask lets the halo be whole, so the centroid
        # is the ball's own. Accepting on the true region still keeps a lamp
        # outside the workspace from ever being a candidate -- the grown mask
        # is one radius wider, not unbounded, and a blob whose CENTRE is
        # outside the arena is discarded whatever its halo does.
        if grow_px and grow_px > 0:
            k = 2 * int(grow_px) + 1
            detect = cv2.dilate(region, np.ones((k, k), np.uint8))
        mask = mask & detect
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
        centre = np.array([sx[i] / tot[i], sy[i] / tot[i]])
        outside = False
        if region is not None:
            # FLAGGED, NOT DISCARDED. A ball whose centre has crossed the
            # workspace line is out of bounds, which is worth knowing and is
            # not the same as being absent -- and inside the grown mask its
            # halo is whole, so this is an accurate position for it. Dropping
            # it here was measured on the saved frames to turn six stuck-at-
            # the-wall moments from "position 3cm wrong" into "no ball at
            # all", which loses the track and leaves nothing to steer back in.
            # You cannot recover a ball you have stopped being able to see.
            cx, cy = int(round(centre[0])), int(round(centre[1]))
            outside = not (0 <= cy < region.shape[0] and 0 <= cx < region.shape[1]
                           and region[cy, cx] > 0)
        out.append({"xy": centre,
                    "outside_region": outside,
                    "area": area,
                    "peak": float(v[labels == i].max()),
                    "sum": float(tot[i]),
                    # Flagged against the mask the pixels were actually cut by,
                    # which is the grown one -- that is the boundary that can
                    # bias this centroid. Against the true region it would fire
                    # on every ball near the edge, including the ones the grown
                    # mask has just made accurate.
                    "clipped_by_region": (detect is not None
                                          and touches_edge(labels, i, detect))})
    # LIT FIRST, then big. Sorting on area alone lets scenery win.
    #
    # A ball's LED core SATURATES -- it reads 255 at the centre because that is
    # what a lamp pointed at a sensor does. Glare on a floor, a reflection, a
    # lit patch of wall inside the grown mask: those are broad and dim. Area
    # alone cannot tell them apart and the bigger one wins, so on two of the
    # twenty-four saved frames that have blobs at all, the real ball was
    # outranked -- area 1648 peak 209 beaten by area 4211 peak 57, and area
    # 6278 peak 255 beaten by area 9897 peak 132.
    #
    # Conservative on purpose. When NOTHING is saturated -- the LED turned
    # down, a dim frame -- every blob scores the same on this key and the order
    # is exactly what it was before, largest first. It can only ever promote a
    # blob that looks like a lamp over one that does not.
    #
    # Out-of-region blobs sort last regardless. `outside_region` is kept rather
    # than discarded (see the comment above) precisely so it can be ranked
    # instead of thrown away.
    out.sort(key=lambda b: (b["outside_region"],
                            -(b["peak"] >= BRIGHT_PEAK),
                            -b["area"]))
    return out, mask, n - 1


BLOWN_MEAN_V = 90.0
"""Mean V above which a frame is taken to be lit rather than dark.

Measured, not guessed. Across the saved frames a working one means 5 to 7 and
the three blown ones mean 165 to 167 — two orders of magnitude apart, with
nothing anywhere near the middle. Ninety sits in that gap with room on both
sides, so it takes a genuinely lit room to trip and cannot fire on a frame the
tracker could still have read.
"""


def frame_blown(frame):
    """Is this frame washed out, rather than merely empty?

    The distinction the tracker cannot otherwise make. `find_blobs` thresholds
    at `V_MIN` and takes connected components; when the whole image is above
    the threshold there is one component covering everything, it fails
    `MAX_AREA`, and the honest answer is zero blobs. Identical, from the
    outside, to a dark room with no ball in it.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(v.mean()) >= BLOWN_MEAN_V


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

MIN_MOVING_CM_S = 12.0
"""Floor for the approach taper, in cm/s.

A Sphero has a speed below which it does not move at all, and the measured
speed map for this hardware never saw a ball travel under 17.8cm/s -- with a
fitted intercept of 21cm/s at byte zero, meaning the linear cm/s-to-byte
conversion is badly wrong down here. A taper that asks for 2cm/s is asking for
a stall, and a ball stalled short of its goal never arrives at all.

Twelve is below what the map says will move and above a certain stall; the
right number is a thing to find on the floor, not to derive. If the ball parks
short and sits there, this is too low."""

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

STYLES = ("pursuit", "align", "turn-go")
"""How a run is driven.

`pursuit` steers continuously from the first frame. `turn-go` alternates
between crawling to point and driving, for the whole run. `align` is the
one-shot: point the ball once, before pursuit starts, then hand over and never
turn again.

The difference matters because the two costs are different. Pure pursuit's
steering gain rises as the lookahead shortens, so a ball that starts badly
aimed swings wide before it settles -- and on a drawn shape that opening arc IS
the deviation, printed into the first stretch of the route. `turn-go` removes
it but pays a crawl at every direction change thereafter, which on a figure
eight is most of them. `align` pays once."""
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

BOOTSTRAP_CM = 14.0
"""How far the ball rolls before the first run is believed.

The gap this closes. `retune_aim` refuses any single reading over
`AIM_MAX_STEP_DEG` as a skid or a tracker jump, which is right for the drift it
exists to trim -- and it means a ball whose frame is 180 degrees out is never
corrected by it at all. Pure pursuit cannot recover past 90 either: beyond that
its feedback pushes the wrong way. So the one error nothing handles is the big
one, and the symptom is a ball driving confidently away from its goal.

Fourteen centimetres is long enough that the gap to the goal has moved further
than the tracker's noise, and short enough to be spent finding out rather than
committed to being wrong.
"""

BOOTSTRAP_GROW_CM = 3.0
"""Growth in the gap that counts as going the wrong way, rather than as noise
or as the arc of a wide turn."""

EDGE_MARGIN_CM = 6.0
"""How far inside the workspace a GOAL must sit, on top of the ball's radius.

Two things go wrong at a boundary and they compound.

The ball has extent, so its CENTRE cannot reach the edge -- a 7.4cm ball
against a wall has its centre 3.7cm short of it. A goal on the line is a goal
the ball can only answer by pushing, and the controller will keep commanding
into the wall because from its point of view it has not arrived.

And its position reads WRONG there. The workspace mask is applied before the
components are found, so a ball straddling the boundary keeps only the part
inside, and its weighted centroid is pulled inward by whatever was cut. The
controller therefore believes the ball is further from the wall than it is,
and drives further out -- into the wall it was already touching.

So goals are held a ball's radius plus this clear of the line. Tracking is not
restricted: the ball may go to the edge and be followed there. It just may not
be SENT somewhere it can only arrive at by shoving.
"""

COAST_LEARN = True
"""Cut the motors early on the last approach and measure the roll-out.

Everything that sizes a distance from stopping -- the arrival radius, the
approach taper -- has been using a coast constant measured on a DIFFERENT ball
months ago; the dock says so, in brackets, every run. The ball itself can
answer it: cut to zero while it is still short of the goal, watch how far it
carries, and that is this ball's coast on this floor today.

It costs nothing, because the ball was going to stop there anyway. The only
change is stopping by letting go rather than by asking, which is what a Sphero
does regardless once the command stops.
"""

KICK_SPEED_CM_S = 22.0
"""Speed used to break contact with something, and ONLY for that.

A ball takes more to start rolling than to keep rolling -- this rig's own speed
map never recorded motion below byte 18 -- so a firm push frees a ball that a
gentle one leaves where it is.

Deliberately NOT applied to every departure. A run that begins with a burst
begins with an overshoot, and it makes the speed slider mean something
different in the first third of a second than it does afterwards; a controller
whose speed you cannot predict from its setting is a controller you cannot tune
by watching. Ordinary starts use the commanded speed. This is for the escape
nudge, where the ball is against a wall and the gentle version is the thing
that does not work.
"""

UNSTICK_S = 0.9
"""How long the escape nudge drives for."""

UNSTICK_TRIES = 2
"""How many times it will try before giving up and stopping.

Not unlimited. A ball wedged in a corner can be nudged off a wall; one under a
chair leg cannot, and a loop that keeps shoving at it is a loop that spends a
battery and a session learning nothing. Two attempts, then it stops and says
so, which is the state a person can act on.
"""

STALL_AFTER_S = 1.5
"""How long a commanded-but-motionless ball is given before it is called out.

Long enough to cover the dead time and the first part of the acceleration --
this rig takes the better part of a second to get going -- and short enough
that you are told inside one attempt rather than after a run."""

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

    Three columns describe the RUN rather than the frame. `run_id` increments
    each time the controller is armed, so rows group into attempts; `run_source`
    says whether a person pressed GO or the model called a tool, which is the
    first thing you want to know when one of them behaves worse than the other;
    and `run_outcome` is blank on every row except the last of a run, where it
    records how the attempt ended -- arrived, stuck, lost, or halted by hand.
    Grouping by `run_id` and reading the final `run_outcome` gives a table of
    attempts and results without reading a single trajectory.

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
               "run_id", "run_source", "run_outcome",
               "path", "path_pts",
               "style", "set_speed", "set_lookahead", "set_arrive",
               "cmd_vx", "cmd_vy", "cmd_deg", "aim_offset_deg", "note")

    FLUSH_EVERY = 30

    def __init__(self, directory="runs", stamp=None):
        self.dir = directory
        self.name = f"fleet_{stamp or time.strftime('%m%d_%H%M%S')}.csv"
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
        # Progress along THIS path, carried between frames. It lives on the
        # path and not on the app because it is only meaningful for the path
        # it was measured on -- a new path is a new run and starts unlocked.
        self.s = None
        self._self_gap = None
        # Whether this route is to be driven FROM ITS START, rather than from
        # whichever part of it the ball happens to be nearest. See `restart`.
        self.anchor = False
        self.launched = False

    def restart(self):
        """Forget the progress made, so the next fix re-locks from anywhere.

        Unless the route is ANCHORED, in which case progress starts at zero and
        the run begins at the beginning.

        The default is right for a goal: a drive to a point should pick up from
        wherever the ball is. It is wrong for a SHAPE. A figure eight, a rose,
        an explicit trajectory -- these were asked for as a whole, and locking
        on to the nearest point means a ball that happens to be sitting near
        the middle starts halfway round and the first half is never driven at
        all. The shape that gets drawn is not the shape that was asked for, and
        nothing reports a problem because the follower is doing exactly what it
        was told.
        """
        self.s = 0.0 if self.anchor else None
        self.launched = False

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

    BACK_CM = 8.0
    """How far back along the path the ratchet may re-lock. Not zero: the
    projection jitters by a centimetre or two with the blob, and a hard
    forward-only rule would let that jitter drag the ball along the path."""

    FWD_CM = 80.0
    """How far ahead it may jump in one frame. Comfortably more than a frame
    of travel at any speed this rig reaches, and far less than a lap.

    A CEILING, not the window itself — see `fwd_window`. "Far less than a lap"
    is the wrong measure for a route that meets itself before it finishes one."""

    SELF_CROSS_RATIO = 3.0
    """How many `SELF_TOUCH_CM` of path must separate two points before they
    count as a crossing rather than the two sides of a corner."""

    MIN_FWD_CM = 12.0
    """The window never shrinks below this, whatever the shape.

    Several frames of travel at the speeds this rig drives at, so the ratchet
    can always keep up with the ball it is following."""

    SELF_TOUCH_CM = 8.0
    """Spatial distance at which two parts of a route count as the same place.

    A little over the ball's 7.3cm width, because that is when the follower
    genuinely cannot tell which branch it is on from position alone."""

    @property
    def self_gap(self):
        """Shortest arc length between two points of this path that MEET.

        Infinite for a route that never approaches itself, which is most of
        them. Measured once and kept: it is a property of the shape.
        """
        if self._self_gap is None:
            self._self_gap = self._measure_self_gap()
        return self._self_gap

    def _measure_self_gap(self):
        """O(n^2) over the samples, once. n is at most a couple of hundred.

        Pairs closer than `SELF_TOUCH_CM` ALONG the path are skipped, because
        every point is near its own neighbours and that is not a crossing.
        What is being looked for is the opposite: far apart in arc, close in
        space.
        """
        ring, cum, length = self.ring, self.cum, self.length
        best = float("inf")
        for i in range(len(ring)):
            for j in range(i + 1, len(ring)):
                arc = float(abs(cum[j] - cum[i]))
                if self.closed:
                    arc = min(arc, length - arc)
                # A CORNER IS NOT A CROSSING. At the apex of a lobe the
                # route turns hard, so samples a few centimetres apart in arc
                # are also a few centimetres apart in space -- and taking that
                # as a self-approach measured every rose at 8cm, collapsed the
                # window to 4cm, and left the follower unable to keep up with a
                # moving ball. What distinguishes the two is the RATIO: at a
                # corner the arc is a small multiple of the chord, and at a
                # crossing the path leaves and comes back, so it is many times
                # it.
                if arc <= self.SELF_CROSS_RATIO * self.SELF_TOUCH_CM:
                    continue
                if float(np.linalg.norm(ring[j] - ring[i])) <= self.SELF_TOUCH_CM:
                    best = min(best, arc)
        return best

    @property
    def fwd_window(self):
        """How far ahead the ratchet may actually look on THIS path.

        Capped at half the distance to where the route next meets itself. An
        n-lobed rose has every lobe passing through the origin, so successive
        crossings are `length / n` apart -- and once that is under `FWD_CM` the
        window spans the next lobe's crossing, the two are equidistant at the
        centre, and a fraction of a millimetre of tracker noise decides which
        one the ball leaves on. Measured with 3mm of noise: a 40cm rose (89cm
        between crossings) never skipped in 300 frames, while 15, 20, 25 and
        30cm roses (33 to 67cm) skipped 10, 9, 12 and 4 times.

        Half rather than all of it, so the window cannot reach a crossing even
        from immediately after the previous one.
        """
        # Floored as well as capped. A window smaller than a frame of travel
        # cannot follow a moving ball at all: it falls behind, the projection
        # lands outside it every frame, and the re-lock that exists for a ball
        # picked up off the floor starts firing on ordinary driving.
        return max(min(self.FWD_CM, 0.5 * self.self_gap), self.MIN_FWD_CM)

    RELOCK_CM = 40.0
    """Off-path distance at which the ratchet gives up and searches globally.
    A ball this far from where progress says it should be has been picked up
    and put somewhere, and insisting on the old arc position would drive it
    back to a place it is no longer near."""

    def project(self, p, near=None):
        """(arc length, distance) of the nearest point on the path to `p`.

        With `near`, the search is RESTRICTED to a window of arc length around
        the progress already made, and that restriction is the whole point.

        Searching globally is correct for a path that never touches itself and
        catastrophic for one that does. On an out-and-back patrol the two legs
        lie on the SAME line, so the ball is exactly equidistant from both and
        which one wins is decided by the last bits of a float -- it flips frame
        to frame. Measured on a real bottom-edge patrol: the lookahead target
        obeyed `target_x = 209 - ball_x`, moving BACKWARD as the ball moved
        forward, and the ball reversed direction 1379 times in 264 seconds
        without ever going back down the edge. It sat at the far end, which is
        exactly what it looks like from across the room: oscillating around the
        corner instead of patrolling between the ends.

        The window is what tells the two legs apart, because they differ in arc
        length even where they agree in position.
        """
        p = np.asarray(p, dtype=float)
        if self.length <= 0:
            return 0.0, float(np.linalg.norm(p - self.pts[0]))

        def search(lo, hi):
            best = (0.0, float("inf"))
            for i in range(len(self.ring) - 1):
                a, b = self.ring[i], self.ring[i + 1]
                seg = float(np.linalg.norm(b - a))
                if hi is not None and (self.cum[i] > hi
                                       or self.cum[i] + seg < lo):
                    continue
                ab = b - a
                denom = float(ab @ ab)
                t = (0.0 if denom <= 0
                     else float(np.clip((p - a) @ ab / denom, 0.0, 1.0)))
                s = float(self.cum[i] + t * seg)
                if hi is not None:
                    # Clamped INTO the window rather than merely filtered by
                    # it. A segment can straddle the edge of the window, and
                    # its nearest point can sit outside -- taking it anyway is
                    # how a jump the window exists to forbid gets through.
                    s2 = float(np.clip(s, lo, hi))
                    if abs(s2 - s) > 1e-9:
                        s, t = s2, (s2 - self.cum[i]) / seg if seg > 0 else 0.0
                q = a + ab * t
                d = float(np.linalg.norm(p - q))
                if d < best[1]:
                    best = (s, d)
            return best

        if near is None:
            return search(0.0, None)
        lo, hi = float(near) - self.BACK_CM, float(near) + self.fwd_window
        if self.closed:
            # A lap boundary is not a wall, and it is not an excuse to give up
            # the window either. Falling back to a global search whenever the
            # window ran past the end is what let a SELF-CROSSING route cut
            # itself short: a figure eight passes through its own centre, so at
            # the crossing both branches are equidistant and the nearest-point
            # search takes whichever wins on float noise. The ball then leaves
            # along the wrong lobe and the shape collapses to one loop.
            #
            # Wrapping costs one extra search. The window is a ring arc, so
            # when it runs off either end it is simply two pieces.
            got = search(max(lo, 0.0), min(hi, self.length))
            if hi > self.length:
                got = min(got, search(0.0, hi - self.length), key=lambda r: r[1])
            if lo < 0.0:
                got = min(got, search(self.length + lo, self.length),
                          key=lambda r: r[1])
        else:
            got = search(max(lo, 0.0), min(hi, self.length))
        # An ANCHORED route does not re-lock. The re-lock exists for a ball
        # picked up and put down somewhere else, and it recognises that case by
        # the ball being far from where progress says it is -- which is also
        # exactly true at the start of an anchored run, when the ball has not
        # reached the beginning of the shape yet. Left in, it fires on the
        # first frame and sends the run back to nearest-point, which is the
        # behaviour being avoided.
        if got[1] > self.RELOCK_CM and not self.anchor:
            far = search(0.0, None)
            if far[1] < got[1]:
                return far
        return got


def pursue(path, pos, lookahead, speed, goal_tol=ARRIVE_CM, radius=0.0,
           coast_s=0.0, min_speed=0.0, ratchet=True):
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

    # ARRIVAL IS CIRCLE TO CIRCLE, not point in circle. The ball has extent,
    # so it is "there" when any part of it is inside the target -- the same
    # judgement a person makes looking at the floor. Testing the CENTRE
    # instead asks the ball to travel its own radius further than the task
    # needs, and at these speeds that last few centimetres is the expensive
    # part: it is the bit that has to be crept up on.
    #
    # `radius` is the blob's own measured extent rather than the ball's
    # catalogue size, which means it follows the threshold: raise `v_min` and
    # the blob shrinks and so does this. That is a real coupling and it is the
    # honest one -- what the controller can see is what it should be judged
    # against, and a blob smaller than the ball simply asks for a little more
    # travel than strictly necessary.
    reach = float(goal_tol) + float(radius)

    def approach(gap, full):
        """Ease off as the stop line comes up, so the coast fits inside it.

        The ball keeps rolling for about half a second after the command stops,
        whatever its speed -- so the distance it needs is `v * coast`, and the
        moment to start shedding speed is when the remaining gap equals that.
        Slowing sooner wastes time; slowing later means arriving too fast to
        stop inside the radius, which is the whole reason a fast run parks
        further out than a slow one.

        Floored rather than taken to zero. A Sphero has a speed below which it
        does not move at all -- the measured map never saw this ball travel
        under 17.8cm/s -- so a taper that asks for 2cm/s asks for a stall, and
        a stalled ball short of its goal never arrives at all.
        """
        if coast_s <= 0.0:
            return full
        need = full * float(coast_s)
        if need <= 1e-6 or gap >= need:
            return full
        # The floor may never EXCEED what was asked for. It exists so a taper
        # does not ask for a speed the ball cannot move at -- not to overrule
        # the slider. Unclamped it did: at a commanded 10cm/s a floor of 12
        # made the ball speed UP on its final approach, which is the opposite
        # of the job, and it bit hardest at the low settings this rig is
        # actually driven at.
        return max(min(float(min_speed), full), full * (gap / need))

    if path.length <= 0:                       # a single goal point
        target = path.pts[0]
        gap = float(np.linalg.norm(target - pos))
        if gap <= reach:
            return np.zeros(2), target, True, "arrived"
        v = approach(gap - reach, speed)
        return (_toward(pos, target, v), target, False,
                f"{gap - reach:.0f}cm to go"
                + (f" at {v:.0f}" if v < speed - 0.5 else ""))

    # Ratcheted from the progress already made. The first fix of a run has no
    # progress yet and locks on from wherever the ball happens to be, which is
    # what lets a run start with the ball anywhere near the route.
    s, off = path.project(pos, near=path.s if ratchet else None)
    if ratchet and path.anchor and not path.launched:
        # HELD AT THE BEGINNING until the ball actually reaches it.
        #
        # Anchoring the arc position to zero is not enough on its own: the
        # forward window is tens of centimetres wide, so the first projection
        # of a ball parked elsewhere slides straight down it and the run starts
        # forty centimetres in regardless. Pinning `s` here keeps the lookahead
        # target at the start of the shape, which is what steers the ball
        # there -- and once it arrives the ratchet takes over with nothing
        # skipped.
        reach = max(float(lookahead), path.MIN_FWD_CM)
        if float(np.linalg.norm(np.asarray(pos, dtype=float)
                                - path.at(0.0))) <= reach:
            path.launched = True
            path.s = s
        else:
            path.s = s = 0.0
    elif ratchet:
        # PROGRESS DOES NOT REWIND, and the window alone did not guarantee it.
        #
        # `BACK_CM` lets the projection settle a couple of centimetres behind
        # so blob jitter cannot drag the ball forward. Storing that as the new
        # progress is what made it compound: the next frame's window is centred
        # on the rewound position, so it may go back again, and again. A ball
        # that drifts sideways where the route passes near its own earlier self
        # walks backwards down the path a few centimetres a frame -- which
        # looks exactly like a robot deciding to return to an old waypoint.
        #
        # The frame's own `s` is still what aims the lookahead, so the jitter
        # allowance is unchanged. Only the RECORD of progress is monotonic.
        #
        # Two exceptions. A closed path wraps, so its arc position must be free
        # to return to zero at the lap boundary. And a genuine re-lock -- the
        # ball more than `RELOCK_CM` off the route, which means it was picked
        # up and put somewhere -- has to be able to move progress backwards,
        # because insisting on the old position would drive it to a place it is
        # no longer near.
        # A RE-LOCK IS RECOGNISED BY WHERE IT LANDED, not by `off`. When
        # `project` gives up on the window and searches globally it returns the
        # global distance, which is small by construction -- so the off-path
        # distance can never reveal that it re-locked. What does reveal it is
        # an arc position behind the window it was allowed to search.
        #
        # `path.s` is None until the first fix of a run, which is what lets a
        # run start with the ball anywhere near the route.
        if path.s is None:
            path.s = s
        elif path.closed:
            # FORWARD ONLY, ROUND THE RING. `max` cannot express this -- an
            # orbit's arc position has to fall back to zero at the lap
            # boundary, and that is indistinguishable from a jump backwards
            # unless it is measured the short way round. A step within
            # `FWD_CM` is ordinary progress, wrap included; anything larger is
            # the route meeting itself and is refused, so a figure eight stays
            # on the lobe it is driving.
            #
            # Unless the ball is genuinely nowhere near where progress says it
            # is, which is the closed-path form of a re-lock: insisting on the
            # old arc position would steer at a place it has left.
            step = (s - path.s) % path.length
            stale = float(np.linalg.norm(np.asarray(pos, dtype=float)
                                         - path.at(path.s)))
            if step <= path.fwd_window or stale > path.RELOCK_CM:
                path.s = s
        elif s < path.s - path.BACK_CM:
            path.s = s
        else:
            path.s = max(path.s, s)
    if not path.closed:
        remaining = path.length - s
        end = path.ring[-1]
        straight = float(np.linalg.norm(end - pos))
        if remaining <= reach and straight <= reach:
            return np.zeros(2), end, True, "arrived"
        target = path.at(min(s + lookahead, path.length))
        # How far there is left to go, by whichever measure says MORE.
        #
        # It has to be the larger, because arriving needs BOTH inside the
        # radius -- the line above tests exactly that pair -- so the distance
        # still to cover is whichever is further from being satisfied.
        #
        # The tempting version is the smaller of the two, and it is wrong in a
        # way a straight path never shows. On a scribble that loops back to
        # near where it began, the ball sits a hand's width from the ENDPOINT
        # while it still has the whole path to drive; taking the smaller makes
        # it crawl at the start of every lap. Taking the larger keeps it at
        # speed until both the arc and the straight line have run out, which is
        # the moment it is actually about to finish.
        speed = approach(max(max(remaining, straight) - reach, 0.0), speed)
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



# -- the agent's tools -----------------------------------------------------

RPC_ROBOT_ID = 2
"""Which integer id this bench publishes as, under `--rpc`.

The VLM framework addresses robots by int -- its `RobotService` ships with
`[1, 2, 3, 4, 147, 248, 225, 699]` -- while this bench names them by code.
One robot means one mapping, and 2 is the id their own sample
`Data/robot_pos.txt` and `Data/astar_segments.json` already use, so their
debug data and ours line up when read side by side.
"""

AGENT_MODEL = "qwen3.5:9b"
"""Preset from `llm/models.yaml`. Local by default; pointing this at a
frontier model is a base-URL change in that file and nothing here."""

AGENT_TOOLS = [
    {"name": "get_state",
     "description": "Where the robot is, whether it is being tracked, and "
                    "whether anything is wrong. Call this before moving.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "goto",
     "description": "Drive the robot to a point, in centimetres.",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"}},
         "required": ["x", "y"]}},
    {"name": "follow_path",
     "description": "Drive along a list of waypoints in centimetres, "
                    "e.g. [[20,20],[80,20],[80,60]].",
     "parameters": {"type": "object", "properties": {
         "points": {"type": "array", "items": {
             "type": "array", "items": {"type": "number"},
             "minItems": 2, "maxItems": 2}}},
         "required": ["points"]}},
    {"name": "patrol",
     "description": "Drive back and forth between two points, turning round at "
                    "each end, until stopped. This is the tool for 'patrol', "
                    "'sweep', 'go back and forth', 'guard an edge' — "
                    "follow_path drives a route ONCE and then stops, so it "
                    "cannot patrol however many points it is given.",
     "parameters": {"type": "object", "properties": {
         "x1": {"type": "number"}, "y1": {"type": "number"},
         "x2": {"type": "number"}, "y2": {"type": "number"}},
         "required": ["x1", "y1", "x2", "y2"]}},
    {"name": "circle",
     "description": "Patrol a circle. Runs until stopped.",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "radius": {"type": "number"}},
         "required": ["x", "y", "radius"]}},
    {"name": "set_speed",
     "description": "Commanded speed. NOTE this is not centimetres per second "
                    "on this hardware; get_state reports the measured speed.",
     "parameters": {"type": "object", "properties": {
         "speed": {"type": "number"}}, "required": ["speed"]}},
    {"name": "wait_until_arrived",
     "description": "Block until the robot finishes its current path, or the "
                    "timeout runs out. Use this instead of calling get_state "
                    "over and over: one call, no matter how long the drive. "
                    "Returns `outcome`: 'arrived' means it reached the goal; "
                    "'stuck' means it stopped short and could not free itself; "
                    "'lost' means the camera lost the ball; 'halted' means a "
                    "person stopped it. Only 'arrived' is success — check it "
                    "rather than assuming the drive worked, and read `reason`.",
     "parameters": {"type": "object", "properties": {
         "timeout_s": {"type": "number", "description": "default 20"}}}},
    {"name": "stop",
     "description": "Halt the robot now.",
     "parameters": {"type": "object", "properties": {}}},
]
"""What the model may do, and the omissions are the point.

It gets to say WHERE the robot should go. It does not get to connect or drop a
ball, flip the arena frame, zero the aim, run the probe, or write calibration
-- the same line `tools/registry.py` draws, for the same reason: a model may
arrange robots, not reconfigure the rig. Those are the operations that can
leave the setup wrong in ways nothing downstream detects, and they are the ones
a person should be present for.

There is no `set_led` either. Brightness is the clipping control on this rig,
and a model dimming a ball until the tracker loses it would be debugging its
own blindness.
"""

# -- the camera ------------------------------------------------------------

SIM_PX_CM = 9.2
"""What `BallSource` draws at, and what the sim homography is built from."""


def sim_frame_size(px_cm=SIM_PX_CM):
    """A fake camera big enough to SEE the workspace it spawns robots into.

    The default was a fixed 1280x720, which at 9.2px/cm covers 139 x 78cm --
    wide enough for the arena and 32cm short of it. Robots are placed anywhere
    in `workspace.json`, so roughly one run in four started with the ball below
    the bottom of the frame: present, moving, driveable, and invisible. The
    tracker reported no lock and it read as "the robot never connected".

    Measured rather than assumed, and it falls back to the old size if the
    workspace cannot be read -- a fake camera is not worth failing to start over.
    """
    try:
        from workspace.space import Workspace
        ws = Workspace.load()
        lo, hi = np.asarray(ws.bounds_cm).min(0), np.asarray(ws.bounds_cm).max(0)
        w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        if w > 1.0 and h > 1.0:
            return (int(np.ceil(w * px_cm)), int(np.ceil(h * px_cm)))
    except Exception:
        pass
    return (1280, 720)


class Camera(threading.Thread):
    """Frames on their own thread; a blocking read on the render thread is a
    frozen window, and a frozen window during bring-up reads as a crash."""

    def __init__(self, spec, size=None, leds=None, pose=None):
        super().__init__(daemon=True, name="blob-cam")
        self.source = None
        self.error = None
        try:
            if str(spec) in ("sim", "ball"):
                self.source = BallSource(leds=leds, pose=pose,
                                         size=size or sim_frame_size())
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

        poses = None
        if self.pose is not None:
            got = self.pose()
            if got:
                poses = [(np.asarray(p, dtype=float), float(a)) for p, a in got]
                self.pos = poses[0][0]
        amp = max(0.0, float(self.leds()) / 255.0)
        canvas = np.zeros((self.h, self.w, 3), np.float32)
        canvas[:] = (6.0, 5.0, 4.0)
        for where, ball_amp in (poses if poses else [(self.pos, 1.0)]):
            self._ball(canvas, where, amp * ball_amp)
        canvas += self.rng.normal(0.0, 1.6, canvas.shape).astype(np.float32)
        return True, np.clip(canvas * self.GAIN, 0, 255).astype(np.uint8)

    def _ball(self, canvas, where, amp):
        """One ball: two tag LEDs about the centre, no taillight."""
        import math
        from vision.shots import _glow, tag_bgr, BALL_CM, TAG_R_CM
        cx, cy = np.asarray(where, dtype=float) * self.px_cm
        r = BALL_CM * self.px_cm / 2.0
        tp = TAG_R_CM * self.px_cm
        col = tag_bgr("red")
        a = np.array([math.cos(math.radians(self.heading)),
                      math.sin(math.radians(self.heading))])
        _glow(canvas, (cx, cy), r * 0.95, col, 90.0 * amp)
        for sgn in (1, -1):
            pt = (cx + sgn * a[0] * tp, cy + sgn * a[1] * tp)
            _glow(canvas, pt, r * 0.68, col, 600.0 * amp)
            _glow(canvas, pt, 3.0, col, 10000.0 * amp)

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
                 with_fleet=True, model=None, rpc=False):
        pygame.init()
        pygame.display.set_caption("fleet test — many bots, driven by an agent")
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
        self.unstick = None
        self.coasting = None
        self.coast_learned = None
        self.run_id = 0
        self.run_source = None
        self.last_outcome = None
        self.last_note = ""
        self._clear_radius_px = None       # last blob seen whole, in px
        self.patrol = None
        self._logged_path_run = None
        self._arm_source = "button"
        self.bootstrap = None
        self._boot_pending = False
        self._still_since = None
        self.unstick = None
        self.agent_busy = False
        self.agent_log = []
        self.agent_model = model or AGENT_MODEL
        self.agent_client = self.connect_model()
        self.typing = False
        self.typed = ""
        self.turning = None
        self._slew_at = self._slew_deg = None
        self.cmd_log = deque(maxlen=120)
        self.probe = None
        self.flips = {"x": False, "y": False}
        self.mirrored = False
        self.reid = None
        self.taper = True
        self.bootstrap = None
        self._boot_pending = False
        self._slew_at = None
        self._slew_deg = None
        self.style = "pursuit"
        self.aligned = False
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
        self.blown = False
        self.parts = 0
        self.history = deque(maxlen=JITTER_N)
        self.trail = deque(maxlen=90)
        self.bots = {}
        self._idle_track = None
        self.exposure_locked = False
        self.exp_took = None

        self.fleet = None
        self.code = code
        self.marker = CYAN
        # Set before `build_fleet`, which advises differently in sim: scanning
        # is Bluetooth and can never find a simulated robot.
        self.sim = str(spec) in ("sim", "ball")
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

        if self.sim:
            # The saved WORKSPACE belongs to a real camera too, and the same
            # argument that replaces the homography in sim applies to it.
            # `calib/blob_region.json` here is a quad at x 342..866, y 272..688
            # -- clicked on a 1280x720 view of an actual floor. The fake camera
            # is 1277x1020 and draws the ball wherever the simulated robot is,
            # so a ball at x=121 is simply outside the mask and `find_blobs`
            # returns nothing. Wherever the robot spawned outside that box, the
            # tracker reported no lock and it read as a robot that never
            # connected.
            #
            # The frame's own rectangle is the honest workspace: it is exactly
            # the area the fake camera can see, and it matches `workspace.json`
            # because `sim_frame_size` measured it from there.
            w, h = sim_frame_size()
            self.corners = [np.array(p, dtype=float) for p in
                            ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))]
            self.say("sim: workspace is the whole frame, not calib/", DIM)
        else:
            saved = config.load("blob_region")
            if saved and len(saved.get("corners", [])) == 4:
                self.corners = [np.array(p, dtype=float) for p in saved["corners"]]
                self.say("workspace loaded from calib/blob_region.json — "
                         "press x to clear")

        self.sliders, self.buttons = [], []
        self.build_dock()

        # -- publishing into the VLM framework, if asked --------------------
        #
        # Off by default and never fatal. The framework is a separate project
        # with its own services running in their own processes; a bench that
        # would not start because one of them is down would be a bad trade for
        # an output nothing here reads back.
        self.bridge = None
        if rpc:
            try:
                from vlm.bridge import Bridge
                self.bridge = Bridge(self, robot_id=RPC_ROBOT_ID)
                self.bridge.start()
                self.say(f"publishing to the VLM framework as robot "
                         f"{RPC_ROBOT_ID}", DIM)
            except Exception as e:
                self.bridge = None
                self.say(f"no VLM bridge: {e}", CORAL)

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
            # Scanning is BLE. In sim it can never find anything, and pointing
            # someone at it there is pointing them at a dead end.
            if self.sim:
                self.say("sim: no robot — restart with --robot CRXS "
                         "(scanning is Bluetooth only)", SUN)
            else:
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

    def push_led(self, code=None):
        """Light one robot, or all of them, and force every taillight OFF.

        Not optional and not a slider: a taillight drags the weighted centroid
        about 7mm backwards along the heading, and because that offset rotates
        with the robot it cannot be calibrated out.
        """
        if not self.fleet:
            return
        for c in ([code] if code else list(self.bots)):
            bot = self.bots.get(c)
            h = self.fleet.handles.get(c)
            if bot is None or h is None:
                continue
            try:
                h.set_led(tuple(int(v) for v in bot["rgb"]))
                h.set_back_led(0)
            except Exception as e:
                self.say(f"{c}: {type(e).__name__}: {e}", CORAL)

    def set_channel(self, i, value):
        """One LED channel of the SELECTED robot, 0-255.

        Per channel and per robot, because neither a ball nor a camera treats
        the three alike. A red LED can be into clipping while blue is still
        dim, and a single brightness number can only move all three together --
        so the only way to get a ball bright enough to find and still short of
        a blown core is to trim the channel that is blowing.

        Safe HERE and not everywhere: this app finds balls by brightness with
        the taillight off, so the colour is for your eye and for staying out of
        clipping. `pose_test.py` reads hue for identity, and hand-mixing an
        LED there would light a robot as one colour while the detector hunted
        another -- the exact two-tables fault `vision/config.py` exists to
        prevent. Doing it here is not a licence to do it there.
        """
        bot = self.bots.get(self.code)
        if bot is None:
            return
        bot["rgb"][int(i)] = int(np.clip(value, 0, 255))
        # The overlay follows the LED, so what is drawn is what the ball is
        # actually glowing rather than what the palette says it should be.
        bot["marker"] = tuple(bot["rgb"])
        self.push_led(self.code)

    def led_of(self, code):
        bot = self.bots.get(code)
        return bot["rgb"] if bot else [0, 0, 0]

    @property
    def track(self):
        """The SELECTED robot's tracker.

        A property, so every line inherited from the single-robot app that
        speaks of `self.track` keeps working and now means "the one the
        controls are pointed at". That is what kept this fork small: only the
        parts that genuinely need to know about N robots were changed.

        An empty stand-in when nothing is connected, rather than None. The dock
        reads this every frame, and None here would crash on the ordinary
        startup path -- before anything is connected, which is exactly when a
        person is looking at the window.
        """
        bot = self.bots.get(self.code)
        if bot is not None:
            return bot["track"]
        if self._idle_track is None:
            self._idle_track = Track()
        return self._idle_track

    def robot_pose(self):
        """Where the SIMULATED robots are, for the fake camera to draw them.

        A list now, one entry per simulated robot, so the sim exercises the
        assignment step rather than only the tracker. Returns None for real
        robots -- a real ball is drawn by the world, and the fake camera must
        never be handed a position that came from anywhere but the simulator.
        """
        if not self.fleet:
            return None
        out = []
        for code in (self.bots or self.fleet.handles):
            h = self.fleet.handles.get(code)
            if h is None or getattr(h, "kind", "") != "sim":
                continue
            # Each ball at ITS OWN brightness, so a blink code can be tested
            # here rather than only on the floor. A fake camera that lights
            # every ball the same cannot exercise the one mechanism that tells
            # them apart.
            rgb = self.bots.get(code, {}).get("rgb") or list(h.rgb)
            out.append((np.asarray(h.pos, dtype=float),
                        max(rgb) / 255.0 if rgb else 1.0))
        return out or None

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
        """Connect to a discovered ball, ADDING it to the fleet.

        Additive, unlike the single-robot app this forked from, which dropped
        whatever it held first. Each ball gets its own roster code and its own
        `Track`, and `self.code` names whichever one the controls currently act
        on -- so the rest of the app, which was written against one robot, keeps
        working unchanged and only the parts that must know about N were
        touched.

        Colours are handed out from the palette rather than chosen, because two
        robots sharing one are two robots the tracker's overlay cannot tell
        apart, and `Fleet.add` refuses the pair anyway.
        """
        if self.fleet is None:
            self.say("no fleet layer — cannot connect", CORAL)
            return
        if any(b["ble"] == ble_name for b in self.bots.values()):
            self.say(f"{ble_name} is already connected", SUN)
            return
        code = self.free_code()
        colour = self.free_colour()
        if colour is None:
            self.say(f"no free tag colour — {len(self.bots)} already connected",
                     CORAL)
            return
        try:
            from fleet.roster import RobotEntry
            entry = RobotEntry(name=ble_name, code=code, kind="real",
                               color=colour, ble_name=ble_name)
            errs = self.fleet.add(entry)
        except Exception as e:
            self.say(f"connect failed: {type(e).__name__}: {e}", CORAL)
            return
        if errs:
            self.say(errs[0], CORAL)
            return

        rgb = list(config.led_rgb(config.COLORS[colour]["hue"]))
        self.bots[code] = {"ble": ble_name, "color": colour,
                           "rgb": rgb, "track": Track(),
                           "marker": tuple(rgb),
                           "zeroed": False, "identified": False}
        self.code = code                    # newest becomes the selected one
        self.dismiss_scan()
        self.push_led(code)
        self.say(f"{code} connecting to {ble_name} as {colour} "
                 f"({len(self.bots)} connected)", MINT)
        self.build_dock()

    def free_code(self):
        """The next unused ad-hoc code. Session-local; nothing is written to
        `roster.json` -- see `AD_HOC`."""
        i = 1
        while f"{self.AD_HOC}{i}" in self.bots:
            i += 1
        return f"{self.AD_HOC}{i}"

    def free_colour(self):
        """A tag colour nothing else is using, or None.

        Blue is skipped: it is the taillight's colour everywhere else in this
        project, and a ball tagged blue cannot be told from its own tail. That
        matters less here, where the tail is forced off -- but the palette is
        shared and a rule that holds in one app and not its neighbour is worse
        than one that always holds.
        """
        taken = {b["color"] for b in self.bots.values()}
        for name in config.COLORS:
            if name != "blue" and name not in taken:
                return name
        return None

    def disconnect(self, code=None):
        """Drop one robot, or the selected one. Stops it first -- always."""
        code = code or self.code
        if code is None or self.fleet is None:
            return
        if self.armed and code == self.code:
            self.disarm()
        h = self.fleet.handles.get(code)
        if h is not None:
            try:
                h.stop()
            except Exception:
                pass
            try:
                self.fleet.remove(code)
            except Exception:
                pass
        self.bots.pop(code, None)
        if self.code == code:
            self.code = next(iter(self.bots), None)
        self.say(f"dropped {code} ({len(self.bots)} left)")
        self.build_dock()

    def select(self, code):
        """Point the controls at one robot. Everything the single-robot app
        did to `self.code` now happens to whichever this names."""
        if code in self.bots:
            self.disarm()
            self.code = code
            self.say(f"selected {code} ({self.bots[code]['color']})")
            self.build_dock()

    def cycle_selected(self):
        codes = list(self.bots)
        if not codes:
            return
        i = codes.index(self.code) if self.code in codes else -1
        self.select(codes[(i + 1) % len(codes)])

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

    def open_log(self):
        """Reveal this run's CSV in Finder.

        The log is the only thing that survives a session, and a path printed
        in a dock is a path somebody has to retype. Revealing rather than
        opening: a CSV double-clicked lands in whatever owns the extension,
        which on most machines is a spreadsheet that reformats the timestamps.
        """
        path = getattr(self.log, "path", None)
        if not path or not os.path.exists(path):
            self.say("no log yet — it is written once the first frame arrives",
                     SUN)
            return
        try:
            import subprocess
            subprocess.Popen(["open", "-R", os.path.abspath(path)])
            self.say(f"revealed {os.path.basename(path)} "
                     f"({self.log.rows} rows)", MINT)
        except Exception as e:
            # Not fatal, and the path is the useful half anyway.
            self.say(f"{os.path.abspath(path)}  ({type(e).__name__})", SUN)

    def save_frame(self):
        """The RAW frame, never the overlay: an opinion baked into a pixel
        cannot be re-read later with a different threshold."""
        if self.frame is None:
            self.say("no frame to save", SUN)
            return
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"fleet_{time.strftime('%m%d_%H%M%S')}.png")
        cv2.imwrite(path, self.frame)
        self.say(f"saved {os.path.relpath(path)}", MINT)

    def say(self, msg, tone=DIM):
        self.note, self.note_tone = msg, tone
        print(f"fleet_test: {msg}")

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
            region=self.region if self.corners is not None else None,
            grow_px=self.grow_px())
        if not fresh:
            return
        self.assign(candidates)
        self.blob = self.bots[self.code]["blob"] if self.code in self.bots else None
        if not self.bots:                      # nothing connected: single-blob
            self.blob = self.track.update(candidates)
        self.others = [b for b in candidates if b is not self.blob]
        self.why = None if self.blob is not None else self.track.status
        # SAY WHEN THE ROOM IS LIT, rather than reporting a lost ball.
        #
        # Three saved frames from 29 Aug are entirely above threshold -- mean V
        # 165 against a working frame's 5 -- and produce no blobs at all,
        # because a mask that keeps everything separates nothing. The tracker
        # was blind, not confused, and it said "no candidate" like any other
        # miss. That sends a person looking for a ball that has rolled away
        # when what happened is that somebody turned the lights on.
        self.blown = self.blob is None and frame_blown(self.frame)
        if self.blown:
            self.why = ("the frame is BLOWN OUT — the room lights are on, or "
                        "the exposure is too high. Nothing can be tracked")
        if self.blob is not None:
            if not self.blob.get("clipped_by_region"):
                # Remembered only while the blob is WHOLE, because that is the
                # only time its area means the ball's size.
                self._clear_radius_px = float(
                    np.sqrt(max(self.blob["area"], 1.0) / np.pi))
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
        for code, bot in self.bots.items():
            sel = (code == self.code)
            draw = config.COLORS[bot["color"]]["draw"]
            b = Button((x, y, w, 22),
                       f"{'>' if sel else ' '} {code} {bot['color']}"
                       f" {'' if bot.get('identified') else '?'} {bot['ble']}",
                       (lambda c=code: self.select(c)),
                       tone=(draw[2], draw[1], draw[0]) if sel else None)
            self.buttons.append(b)
            y += 26
        if self.bots:
            y += 4
        self.buttons.append(Button((x, y, bw, 24),
                                   f"flip x{'*' if self.flips['x'] else ''}",
                                   (lambda: self.flip_axis("x")), tone=SUN))
        self.buttons.append(Button((x + bw + 6, y, bw, 24),
                                   f"flip y{'*' if self.flips['y'] else ''}",
                                   (lambda: self.flip_axis("y")), tone=SUN))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "save calib",
                                   self.save_calib, tone=CORAL))
        y += 30
        self.buttons.append(Button((x, y, bw * 2 + 6, 24), "re-identify",
                                   self.start_reid, tone=MINT))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "log",
                                   self.open_log, tone=CYAN))
        y += 34
        for i, ch in enumerate(("R", "G", "B")):
            self.sliders.append(Slider(
                (x, y, w, 18), f"LED {ch}", 0, 255,
                (lambda i=i: self.led_of(self.code)[i]),
                (lambda v, i=i: self.set_channel(i, v))))
            y += 24
        y += 6
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
        real, _ = self.sizing_speed()
        want = latency_advice(real, plant)
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
        if self.mirrored:
            # Refused, not warned. On a mirrored frame the position feedback
            # pushes the wrong way in one axis, so EVERY controller here
            # diverges -- measured at 33cm off a line it started 10cm from,
            # under both pursuit and turn-go, whatever the gain or lookahead.
            # Arming anyway spends a hardware session watching a ball circle
            # and blaming the follower for the camera.
            self.say("the probe says this frame is MIRRORED — flip an axis and "
                     "re-probe. Nothing converges until then.", CORAL)
            return
        self.plant = plant_constants(self.code)
        # CLOSE THE PREVIOUS RUN before starting another.
        #
        # Arming while already armed used to raise `run_id` and carry on, so
        # the run that was replaced never reached `disarm` and never got its
        # end row -- which is where the verdict is written. That is why
        # `run_outcome` is blank on most of the logged runs: not because the
        # outcome was unknown, but because nothing ever asked. It matters most
        # exactly where the log matters most, since an agent re-aiming
        # mid-drive is the case that produces back-to-back arms.
        if self.armed:
            self.disarm("superseded by a new path")
        h = self.fleet.handles.get(self.code)
        if h is not None:
            self.zero_at_rest(h)
        self.armed = True
        self.run_id += 1
        self.run_source = self._arm_source
        self._arm_source = "button"        # back to the default for next time
        self.bootstrap = None      # armed on the next fix, once a pos exists
        self._boot_pending = True
        # A FRESH ESCAPE BUDGET, because the count is per RUN and not per
        # process. `step_unstick` deliberately keeps `tries` after an escape
        # ends so that two failures in one run give up instead of nudging
        # forever -- but nothing used to clear it afterwards, so the count
        # survived the disarm and every later arm inherited an exhausted
        # budget. Measured: the same drive into the same wall fought for 12.6s
        # with two escapes the first time and quit after 4.0s with none the
        # second. From an agent's second `goto` onward the first stall was
        # fatal, which is what made it look like the model was giving up.
        self.unstick = None
        self._still_since = None
        self.last_outcome, self.last_note = None, ""   # this run's verdict
        # Every run begins by pointing the ball, because at the moment of
        # arming it is stationary and facing wherever it last stopped.
        self.turning = (time.perf_counter()
                        if self.style in ("turn-go", "align") else None)
        self.aligned = False
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
            self.mirrored = False
            msg = (f"ROTATION of {first:+.0f}° — a plain aim offset, which "
                   "zeroing corrects")
            tone = MINT
        else:
            self.mirrored = True
            msg = (f"REFLECTION — the error swings {spread:.0f}° across the "
                   f"four headings. {self.mirror_axis(rows)} This is a frame "
                   "fault, not an aim error, and no offset fixes it")
            tone = CORAL
        print(f"  -> {msg}\n")
        self.say(msg, tone)

    def toggle_taper(self):
        """Ease off near the goal, or drive at one speed all the way in."""
        self.taper = not self.taper
        self.say(f"approach taper {'ON' if self.taper else 'OFF'}", MINT)

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
        # The last probe described the OLD frame. Whether the flip cured it is
        # a question only a new probe answers, so the verdict is cleared rather
        # than assumed cured.
        self.mirrored = False
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
        if was and self.run_id:
            # One final row carrying the verdict. Written here because the
            # outcome is not known while the run is happening -- every row
            # before this one was recorded by a controller that did not yet
            # know how it would end.
            text = (note or "").lower()
            outcome = ("arrived" if "arrived" in text
                       else "stuck" if "stuck" in text
                       else "lost" if "lost" in text or "went away" in text
                       else "halted")
            # Kept, not just logged. The CSV row is written for a person
            # reading it afterwards; `wait_until_arrived` needs the same
            # verdict WHILE the run is being answered, because otherwise all
            # it can see is that driving stopped -- and "stopped" is true of
            # arriving, getting stuck, losing the ball and pressing esc alike.
            self.last_outcome, self.last_note = outcome, note or ""
            self.log.write(run_id=self.run_id, run_source=self.run_source,
                           run_outcome=outcome, mode="end", note=note or "",
                           set_speed=int(self.speed),
                           set_lookahead=int(self.lookahead),
                           set_arrive=int(self.goal_tol), style=self.style,
                           path=None if self.path is None else self.path.kind)
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

    def start_bootstrap(self, pos):
        """Begin the "is this even the right way" stretch of a run."""
        goal = None
        if self.path is not None and self.path.pts:
            goal = self.path.pts[-1] if not self.path.closed else None
        self.bootstrap = {"from": np.asarray(pos, dtype=float),
                          "goal": None if goal is None else np.asarray(goal, float),
                          "gap0": (None if goal is None else
                                   float(np.linalg.norm(goal - pos))),
                          "at": time.perf_counter()}

    def step_bootstrap(self, pos, cmd_v):
        """Check the first stretch, and correct the frame once if it is wrong.

        The test the operator suggested, and it is a better one than a bearing
        comparison for this particular fault: DID THE GAP TO THE GOAL SHRINK.
        A ball driving away from where it was sent is unambiguous, needs no
        convention to interpret, and cannot be confused with a wide turn -- a
        turn still closes on the goal, just slowly.

        When it did not shrink, the correction is the measured error itself
        rather than a flat 180. Frames are wrong by whatever they are wrong by;
        assuming the fault is exactly a reversal would leave anything between
        90 and 180 still diverging, and those diverge just as surely.

        Once only. A loop that keeps re-deciding which way is forward while
        driving is a loop that can oscillate, and there is nothing here to damp
        it.
        """
        b = self.bootstrap
        travelled = float(np.linalg.norm(pos - b["from"]))
        if travelled < BOOTSTRAP_CM:
            return
        self.bootstrap = None
        if b["gap0"] is None:
            return                     # a closed path has no goal to close on

        gap = float(np.linalg.norm(b["goal"] - pos))
        if gap <= b["gap0"] - 1.0:
            self.aim_note = f"heading the right way ({b['gap0'] - gap:+.0f}cm)"
            return
        if gap < b["gap0"] + BOOTSTRAP_GROW_CM:
            self.aim_note = "sideways so far — leaving the frame alone"
            return

        tr = self.travel_readout()
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        if tr is None or tr[1] is None or h is None or cmd_v is None:
            self.say("drove the wrong way, but could not measure by how much",
                     CORAL)
            return
        from fleet.handle import velocity_to_command
        want, _ = velocity_to_command(np.asarray(cmd_v, dtype=float))
        error = (tr[1] - want + 180.0) % 360.0 - 180.0
        h.heading_offset = (h.heading_offset
                            + h.HEADING_SIGN * error) % 360.0
        h.estimator.restart()
        self.say(f"drove AWAY from the goal ({gap - b['gap0']:+.0f}cm) — "
                 f"frame was {error:+.0f}° out, corrected", SUN)

    def start_coast(self, gap, speed_now):
        """Let go, and remember where we let go from."""
        self.coasting = {"at": time.perf_counter(),
                         "from": np.asarray(self.to_cm(self.blob["xy"]),
                                            dtype=float),
                         "speed": float(speed_now), "gap": float(gap)}

    def step_coast(self, h):
        """Ride the roll-out, and learn the constant from it. True while it owns
        the wheels."""
        c = self.coasting
        h.stop()
        self.cmd_v = None
        moving = self.true_speed()
        elapsed = time.perf_counter() - c["at"]
        if moving is not None and elapsed < 4.0:
            self.drive_note = f"coasting in ({elapsed:.1f}s)"
            return True

        self.coasting = None
        if self.blob is None or c["speed"] < 1.0:
            return False
        here = np.asarray(self.to_cm(self.blob["xy"]), dtype=float)
        rolled = float(np.linalg.norm(here - c["from"]))
        if rolled < 0.5 or elapsed < 0.1:
            return False
        # Coast expressed as a TIME, not a distance, because that is the form
        # that transfers: distance scales with the speed it started from, and
        # `coast_s` times the current speed gives the distance at any speed.
        learned = rolled / c["speed"]
        self.coast_learned = (learned if self.coast_learned is None
                              else 0.5 * (self.coast_learned + learned))
        self.say(f"coasted {rolled:.0f}cm from {c['speed']:.0f}cm/s "
                 f"— this ball's coast is {self.coast_learned:.2f}s", MINT)
        return False

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

    def start_unstick(self):
        """Back straight off whatever it is pushing against.

        A ball that is commanded and not moving is almost always against
        something, and the direction it was driving is the direction of the
        obstacle. So the escape is the reverse of the last command -- a 180 and
        a nudge -- which needs no map of the arena and no guess about which
        wall it is.

        It also breaks the loop that got it there. Pursuit keeps steering at a
        target it cannot reach, so the ball stays pinned and the controller
        stays convinced; driving the other way for a moment is what lets the
        next tick see a different situation.
        """
        if self.cmd_v is None:
            return
        tries = (self.unstick or {}).get("tries", 0)
        away = -np.asarray(self.cmd_v, dtype=float)
        n = float(np.linalg.norm(away))
        if n < 1e-6:
            return
        away = self.into_free_space(away / n)
        n = 1.0
        self.unstick = {"at": time.perf_counter(),
                        # Off a wall at KICK speed, not at the slider's. The
                        # slider is set for approaching a goal; breaking
                        # contact with something needs more than that, and it
                        # is the reason a gentle nudge fails to free a ball
                        # that a firm one frees at once.
                        "v": away / n * max(float(self.speed),
                                            KICK_SPEED_CM_S),
                        "tries": tries + 1}
        self._still_since = None
        self.say(f"stuck — backing off (try {tries + 1} of {UNSTICK_TRIES})",
                 SUN)

    def into_free_space(self, away):
        """Bend an escape away from the boundary it would otherwise run into.

        Reversing the last command encodes the one thing actually known -- what
        the ball was pushing against -- and needs no map. What it cannot know
        is the rest of the arena. In a CORNER the reverse of a command into one
        wall runs along it and into the other, and with the aim off the
        commanded direction is not the travelled one anyway, so the reverse can
        point somewhere the ball was never going.

        The workspace is known exactly, so the inward normal near a boundary is
        not a guess. Adding it fixes both cases at once: where the escape
        already heads inward it is reinforced, where it heads out it is turned
        round, and in a corner both normals apply and the sum bisects them.

        Only near an edge. In open floor there is no normal to add and the
        reverse-the-command rule stands untouched.
        """
        away = np.asarray(away, dtype=float)
        box = self.agent_bounds()
        if box is None or self.blob is None or not self.homography.ready:
            return away
        pos = np.asarray(self.to_cm(self.blob["xy"]), dtype=float)
        margin = self.goal_margin_cm()

        # PER AXIS, because each wall is its own constraint and summing them
        # into one vector loses that. At a corner the inward normals are (1,0)
        # and (0,1); their sum is a single diagonal, and adding that to an
        # escape pointing hard along -x still leaves x negative -- the ball
        # drives into the left wall at a slight angle instead of straight at
        # it, which is not an escape.
        push = np.zeros(2)
        out = away.copy()
        for axis, (lo, hi) in enumerate(((box[0], box[2]), (box[1], box[3]))):
            if pos[axis] - lo < margin:
                push[axis] += 1.0
                out[axis] = max(out[axis], 0.0)     # never further into it
            if hi - pos[axis] < margin:
                push[axis] -= 1.0
                out[axis] = min(out[axis], 0.0)
        mag = float(np.linalg.norm(push))
        if mag < 1e-9:
            return away                     # open floor: nothing to bend

        n = float(np.linalg.norm(out))
        if n > 1e-6:
            # Something of the original escape survives -- along the wall
            # rather than into it, which is still the direction that knows
            # what the ball was pushing against.
            return out / n
        # It pointed squarely out of the arena and nothing is left of it. The
        # inward normal is all there is, and driving into the boundary is the
        # one direction already known to be useless.
        return push / mag

    def step_unstick(self, h):
        """Drive the escape. Returns True while it owns the wheels."""
        u = self.unstick
        if time.perf_counter() - u["at"] < UNSTICK_S:
            v = self.slew(u["v"])
            h.set_velocity(v)
            self.cmd_v = np.asarray(v, dtype=float)
            self.drive_note = f"backing off ({u['tries']}/{UNSTICK_TRIES})"
            return True
        self.unstick = {"tries": u["tries"]}      # keep the count, stop driving
        self._still_since = None
        return False

    def start_patrol(self, a, b):
        """Drive between two points, back and forth, until stopped.

        A RUN-LEVEL STATE MACHINE and deliberately not a path shape. The
        obvious implementation is one path that goes out and comes back --
        `[a, b, a]` -- and it is the thing that was already failing: both legs
        lie on the same line, so the projection cannot tell them apart and the
        follower reverses on floating-point noise near the far end. Two plain
        one-way legs, swapped on arrival, have no ambiguity to resolve. Each
        leg is an ordinary open line, driven by the follower that is already
        tested, and arriving is an ordinary arrival.

        Tied to the path OBJECT rather than to a flag, so that drawing a new
        route or calling any other tool simply ends the patrol -- there is no
        clearing discipline to remember and no way to leave one running under
        a path that has since been replaced.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        self.path = Path.line(a, b)
        self.patrol = {"a": a, "b": b, "legs": 1, "path": self.path}

    def step_patrol(self):
        """Turn round and drive back. True if the patrol took the arrival."""
        p = self.patrol
        if p is None or p.get("path") is not self.path:
            return False
        p["a"], p["b"] = p["b"], p["a"]
        p["legs"] += 1
        self.path = Path.line(p["a"], p["b"])
        p["path"] = self.path
        # The coast belongs to the leg that just ended, and the turn phase to
        # the one starting: a ball that has just stopped is pointing back the
        # way it came, which is the case `turn-go` exists for.
        self.coasting = None
        self.turning = (time.perf_counter()
                        if self.style in ("turn-go", "align") else None)
        self.aligned = False
        self.drive_note = f"patrol leg {p['legs']}"
        return True

    def stall_report(self):
        """Commanded, but not moving? Say so, with the numbers that separate
        the causes. `(text, tone)` or None.

        The app already knows this and has been keeping it to itself: it knows
        the velocity it commanded, the byte that became, whether the radio link
        is up, and what the camera measures the ball doing. "It did not move"
        is a conclusion those four make jointly, and leaving a person to draw
        it by comparing a dock line against the floor is how a session gets
        spent on the wrong suspect.
        """
        if not self.armed or self.cmd_v is None:
            self._still_since = None
            return None
        want = float(np.linalg.norm(self.cmd_v))
        if want < 1e-6:
            self._still_since = None
            return None
        moving = self.true_speed()
        if moving is not None:
            self._still_since = None
            return None
        now = time.perf_counter()
        if self._still_since is None:
            self._still_since = now
            return None
        held = now - self._still_since
        if held < STALL_AFTER_S:
            return None

        from fleet.handle import velocity_to_command
        _, byte = velocity_to_command(np.asarray(self.cmd_v, dtype=float))
        h = self.fleet.handles.get(self.code) if (self.fleet and self.code) else None
        link = getattr(h, "link_up", None) if h is not None else None
        kind = getattr(h, "kind", "") if h is not None else ""
        lines = [f"COMMANDED {want:.0f} cm/s (byte {byte}) for {held:.0f}s — "
                 f"camera sees under {AIM_MIN_SPEED_CM_S:.0f} cm/s"]
        if kind == "real" and link is False:
            lines.append("the radio link is DOWN — nothing is reaching the ball")
        elif h is not None and getattr(h, "last_error", None):
            lines.append(f"last radio error: {h.last_error}")
        else:
            lines.append("link looks up, so the packets are landing and the "
                         "ball is not acting on them: too slow to overcome "
                         "friction, on its side, asleep, or flat")
        return ("  ".join(lines), CORAL)

    def true_speed(self):
        """What the ball is ACTUALLY doing, in cm/s, or None.

        Not the slider. The slider is a number converted to a speed byte
        through an assumed `MAX_SPEED` of 60cm/s, and the measured speed map
        for this hardware says that assumption is badly wrong: a slider set to
        6 sends byte 25, and byte 24 was measured travelling 24.9cm/s. Four
        times what was asked for.

        Everything that sizes a distance from speed -- the lookahead, the
        arrival radius, the approach taper -- is wrong by that same factor if
        it reads the slider. An arrival radius of 3cm on a ball that coasts
        13cm is the orbit condition exactly: it cannot stop inside the circle,
        so it circles.

        The tracker already answers this properly, in centimetres, from the
        camera. Use the measurement and the byte map stops mattering.
        """
        tr = self.travel_readout()
        if tr is None or not self.homography.ready:
            return None
        return tr[2] if tr[2] >= AIM_MIN_SPEED_CM_S else None

    def sizing_speed(self):
        """`(speed, "measured"|"slider")` -- what distances should be sized on."""
        got = self.true_speed()
        return (got, "measured") if got else (float(self.speed), "slider")

    def px_per_cm(self):
        """Scale at the middle of the frame, from the homography. None if unset.

        Measured rather than assumed, and it varies across the frame because
        perspective does -- 3.7 at one corner and 3.4 at another on this rig.
        The middle is a fair single number for sizing a gate that only has to
        be roughly right.
        """
        if not self.homography.ready or self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        a = np.asarray(self.to_cm([w / 2.0, h / 2.0]), dtype=float)
        b = np.asarray(self.to_cm([w / 2.0 + 50.0, h / 2.0]), dtype=float)
        d = float(np.linalg.norm(b - a))
        return None if d < 1e-6 else 50.0 / d

    def jump_advice(self):
        """What `max jump` should be, in pixels, or None.

        The gate exists to disbelieve a blob that cannot be the ball, so it is
        sized from how far the ball could possibly travel in one frame: its top
        speed, over the frame rate, in pixels. Four times that leaves room for
        a slow frame and for the centimetre the centroid jumps if the halo
        splits, while still being far tighter than a default that would accept
        a reflection on the other side of the arena.

        It does not need to cover a DROPPED frame: `Track` widens the gate
        itself by the number of misses, so this is the one-frame number.
        """
        from fleet.handle import MAX_SPEED
        scale = self.px_per_cm()
        fps = self.cam.fps
        if scale is None or fps < 1.0:
            return None
        return 4.0 * MAX_SPEED / fps * scale

    HALO_GROW_CM = 10.0
    """Fallback halo radius, for growing the detection mask before any blob
    has been seen. Measured at 9.9cm on a real frame from this rig -- the LED
    glow, not the 7.3cm shell, because the glow is what the mask cuts."""

    def grow_px(self):
        """How far to grow the detection mask, in pixels.

        One halo radius, from the last blob seen WHOLE. Deliberately not from
        the current blob: a clipped blob measures smaller, so sizing the grow
        on it would shrink the very margin that stops it being clipped, and
        the two would chase each other down.
        """
        r = self._clear_radius_px
        if r is None:
            px_cm = self.px_per_cm()
            if not px_cm:
                return 0
            r = self.HALO_GROW_CM * px_cm
        return int(min(r, 120.0))

    def blob_radius_cm(self):
        """The tracked blob's own radius, in centimetres. 0 when unknown.

        Measured from the blob's AREA rather than from the ball's catalogue
        size, and mapped through the homography rather than assumed -- so it is
        the extent this camera actually sees at this threshold, in the units
        the controller commands in.

        Area-equivalent, which for a roughly round blob is what "how big is it"
        means. It is not the right radius for a long thin cluster, but this
        method only ever tracks one merged halo, and a halo that is not roughly
        round is a frame with a problem the arrival radius cannot fix.
        """
        blob = self.blob
        if blob is None or not self.homography.ready:
            return 0.0
        r_px = float(np.sqrt(max(blob["area"], 1.0) / np.pi))
        if blob.get("clipped_by_region") and self._clear_radius_px:
            # A CLIPPED BLOB IS NOT A SMALLER BALL. Its area is whatever
            # survived the mask, so believing it shrinks the goal margin and
            # the arrival radius at the exact moment the ball is against a
            # boundary and those need to be at their widest. Measured: the
            # margin fell from 15.9cm whole to 10.3cm at 19% left -- relaxing
            # by 5.6cm while the position was also reading 7cm too far in, the
            # two errors pointing the same way. The last whole measurement is
            # the honest one.
            r_px = max(r_px, float(self._clear_radius_px))
        centre = np.asarray(blob["xy"], dtype=float)
        a = self.to_cm(centre)
        b = self.to_cm(centre + np.array([r_px, 0.0]))
        return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))

    def assign(self, candidates):
        """Give each robot at most one blob, and no blob to two robots.

        Greedy on distance-to-prediction, closest pair first. Not the optimal
        assignment -- a Hungarian solve would be -- and deliberately so: with a
        handful of robots the two agree almost always, and where they differ it
        is because two balls are close enough that the answer is a guess
        either way. A guess dressed up as an optimum is worse than an obvious
        one.

        THE PART THAT MATTERS IS EXCLUSIVITY. Letting each track pick its own
        nearest blob independently is what puts two robots on one blob, which
        is the failure the report this method comes from documented and never
        solved: "with one small flicker in one robots blob detection, the
        system will wrongly localize multiple robot to one location". Claiming
        a blob removes it from everyone else's pool, so that cannot happen --
        the second robot is told it has no blob, which is true and recoverable,
        rather than being handed one that belongs to another.

        Robots left without a blob still get `update([])`, so their tracks
        coast and expire on the ordinary schedule instead of freezing.
        """
        if not self.bots:
            return
        free = list(candidates)
        pairs = []
        for code, bot in self.bots.items():
            track = bot["track"]
            target = track.predicted
            for blob in free:
                d = (float(np.linalg.norm(blob["xy"] - target))
                     if target is not None else float("inf"))
                pairs.append((d, code, blob))
        pairs.sort(key=lambda p: p[0])

        taken, served = set(), {}
        for d, code, blob in pairs:
            if code in served or id(blob) in taken:
                continue
            if d == float("inf"):
                continue            # no prediction yet; acquisition below
            served[code] = blob
            taken.add(id(blob))

        # Robots with no prediction yet acquire from whatever is left, largest
        # first -- the same rule the single-robot app uses on its first frame.
        spare = [b for b in free if id(b) not in taken]
        spare.sort(key=lambda b: -b["area"])
        for code, bot in self.bots.items():
            if code in served or bot["track"].predicted is not None:
                continue
            if spare:
                served[code] = spare.pop(0)

        for code, bot in self.bots.items():
            got = served.get(code)
            bot["blob"] = bot["track"].update([got] if got else [])

        # A pairing this step INVENTED is not a pairing. On acquisition there
        # is nothing to match against -- no prediction, no colour being read,
        # no roll call yet -- so which robot lands on which blob is arbitrary,
        # and it is arbitrary in a way that looks exactly like being right.
        # Measured on three simulated robots the acquisition permuted all
        # three, every track locked, and every marker looked confident.
        #
        # So acquisition marks the robot UNIDENTIFIED. Tracking it from there
        # is still useful -- the blob is a real ball and following it is real
        # information -- but the NAME on it is a guess until a roll call binds
        # it, and the dock says so rather than letting the label pass for
        # evidence.
        for code, bot in self.bots.items():
            if bot["track"].xy is None:
                bot["identified"] = False

    def contended(self, code, within=None):
        """Is another robot close enough that a swap is possible?

        The trigger for re-checking identity. Swaps do not happen at random --
        they happen when two balls are near enough that their blobs could be
        exchanged without either track noticing, and that is a distance, so it
        can be watched for cheaply instead of verified continuously.
        """
        me = self.bots.get(code)
        if me is None or me["track"].xy is None:
            return False
        gate = within if within is not None else me["track"].max_jump * 1.5
        for other, bot in self.bots.items():
            if other == code or bot["track"].xy is None:
                continue
            if float(np.linalg.norm(bot["track"].xy - me["track"].xy)) < gate:
                return True
        return False

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

        # THE ROUTE ITSELF, once per run.
        #
        # Only the path's KIND used to be recorded, and that turned out not to
        # be enough to answer the question the log exists for. A patrol that
        # sat oscillating at one end for four minutes was logged as
        # "polyline"; the geometry that explained it had to be reconstructed
        # from the arithmetic relating the target to the ball, and the exact
        # point list was never recovered at all. One cell per run fixes that.
        pts = None
        if self.run_id and self.run_id != self._logged_path_run \
                and self.path is not None and self.path.pts:
            self._logged_path_run = self.run_id
            # Capped, because a freehand drag is hundreds of points and a
            # readable cell is worth more here than the last millimetre of a
            # scribble. Truncation is stated rather than silent.
            keep = self.path.pts[:200]
            pts = ";".join(f"{p[0]:.1f} {p[1]:.1f}" for p in keep)
            if len(self.path.pts) > len(keep):
                pts += f";...+{len(self.path.pts) - len(keep)} more"

        self.log.write(
            frame=self.seen, mode=mode, status=self.track.status,
            path_pts=pts,
            goal_x=None if goal is None else float(goal[0]),
            goal_y=None if goal is None else float(goal[1]),
            gap_cm=gap,
            target_x=None if self.target_cm is None else float(self.target_cm[0]),
            target_y=None if self.target_cm is None else float(self.target_cm[1]),
            run_id=self.run_id or None,
            run_source=self.run_source if self.run_id else None,
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
        if self.typing:
            # THE KEYBOARD IS NOT ALWAYS READ THROUGH `key`. This polls
            # `get_pressed` directly, so the guard that stops typing firing
            # shortcuts does not reach it -- and typing "was" on this tab drove
            # a real ball fifty centimetres before anything on screen said so.
            #
            # A suppressed key must also RELEASE what it was holding: a text
            # box opened mid-drive would otherwise leave the last command
            # standing and the ball rolling while you type at it.
            if self.manual:
                h.stop()
                self.manual = False
                self.cmd_v = None
            return False
        held = pygame.key.get_pressed()
        want = np.zeros(2)
        for key, direction in self.MANUAL_KEYS.items():
            if held[key]:
                want += np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(want))
        if n >= 1e-9 and not h.connected:
            # SAY IT, rather than commanding a link that is not there.
            # `h is None` was the only guard, and a roster entry whose kind is
            # `real` always produces a handle -- it just never connects if the
            # ball is off, out of range, or simply not present, which is every
            # run in sim. `set_velocity` on it returns quietly and the keys do
            # nothing, so the bench looks broken rather than unconnected.
            self.say(f"{self.code} is not connected — press b to scan, or "
                     f"run a sim robot with --robot SYRX", SUN)
            return False
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

    # -- re-identify -------------------------------------------------------

    ID_SLOTS = 4
    ID_SLOT_S = 0.5
    ID_DIM = 0.30
    """A blink is BRIGHT vs DIM, never off.

    An extinguished ball is a ball the tracker loses, and losing every track in
    order to find out whose is whose defeats the purpose -- you would come back
    with names and no positions to attach them to. At 30% the blob stays well
    clear of the threshold, so every track survives the whole two seconds and
    the names land on tracks that never stopped being followed.
    """

    @staticmethod
    def id_codes(n, slots=ID_SLOTS):
        """`n` distinct blink patterns, as lists of bits.

        All-ones and all-zeros are skipped: a ball that never changes carries
        no signal, and two of them would be indistinguishable from each other
        and from a ball whose LED is stuck. Everything else is fair game, so
        four slots carry fourteen identities -- 2^k - 2, which is why this
        costs the same two seconds whether you have three robots or thirteen.
        """
        out = []
        for v in range(1, (1 << slots) - 1):
            bits = [(v >> i) & 1 for i in range(slots)]
            if 0 < sum(bits) < slots:
                out.append(bits)
            if len(out) == n:
                break
        return out

    def start_reid(self):
        """Blink every robot at once and read the names back off the blobs.

        PARALLEL, not one at a time. A roll call that lights robots in turn
        costs a slot per robot; blinking them simultaneously in a code costs
        `ID_SLOTS` slots however many there are -- 2 seconds for three robots
        or for thirteen. That is the whole reason to use a code rather than a
        sequence.

        What it fixes is the thing acquisition cannot: which blob belongs to
        which robot. Tracking alone can follow a ball perfectly and still have
        the wrong name on it, and the name is what the drive commands are
        addressed to.
        """
        if len(self.bots) < 1:
            self.say("nothing connected to identify", SUN)
            return
        if self.armed:
            self.say("stop driving first", SUN)
            return
        codes = self.id_codes(len(self.bots))
        if len(codes) < len(self.bots):
            self.say(f"{self.ID_SLOTS} slots cannot separate "
                     f"{len(self.bots)} robots", CORAL)
            return
        self.reid = {"at": time.perf_counter(), "slot": -1,
                     "code": dict(zip(self.bots, codes)),
                     "seen": {c: [[] for _ in range(self.ID_SLOTS)]
                              for c in self.bots},
                     "base": {c: list(b["rgb"]) for c, b in self.bots.items()}}
        self.say(f"identifying {len(self.bots)} robots — "
                 f"{self.ID_SLOTS * self.ID_SLOT_S:.0f}s")

    def step_reid(self):
        """One tick of the blink. Drives the LEDs, samples, then decodes."""
        r = self.reid
        elapsed = time.perf_counter() - r["at"]
        slot = int(elapsed / self.ID_SLOT_S)

        if slot >= self.ID_SLOTS:
            self.finish_reid()
            return

        if slot != r["slot"]:
            r["slot"] = slot
            for c, bot in self.bots.items():
                bit = r["code"][c][slot]
                base = r["base"][c]
                bot["rgb"] = [int(v * (1.0 if bit else self.ID_DIM))
                              for v in base]
                self.push_led(c)

        # Sampled per TRACK, not per robot: which robot a track belongs to is
        # the question, so the answer must not be assumed while gathering the
        # evidence for it.
        for c, bot in self.bots.items():
            blob = bot.get("blob")
            if blob is not None:
                r["seen"][c][slot].append(float(blob["peak"]))

    def finish_reid(self):
        """Decode the blink, and move the NAMES onto the right tracks."""
        r, self.reid = self.reid, None
        for c, base in r["base"].items():
            if c in self.bots:
                self.bots[c]["rgb"] = list(base)
        self.push_led()

        read = {}
        for c, slots in r["seen"].items():
            means = [float(np.mean(v)) if v else None for v in slots]
            if any(m is None for m in means):
                continue
            # Thresholded against this track's OWN range, never an absolute
            # level. A ball far from the camera is dimmer than a near one at
            # the same drive, so an absolute cut would read distance as data.
            lo, hi = min(means), max(means)
            if hi - lo < 1.0:
                continue                    # never changed: no signal in it
            mid = (lo + hi) / 2.0
            read[c] = [1 if m > mid else 0 for m in means]

        owner = {tuple(bits): c for c, bits in r["code"].items()}
        moved, lost = {}, []
        for holder, bits in read.items():
            who = owner.get(tuple(bits))
            if who is None:
                lost.append(holder)
            else:
                moved[holder] = who
        for c in self.bots:
            if c not in read:
                lost.append(c)

        # A track whose code nobody claims, or which two tracks claim, is left
        # UNIDENTIFIED rather than being given a best guess. A wrong name is
        # worse than no name: the name is what drive commands are addressed to.
        seen_targets = list(moved.values())
        for holder, who in list(moved.items()):
            if seen_targets.count(who) > 1:
                moved.pop(holder)
                lost.append(holder)

        tracks = {holder: (self.bots[holder]["track"],
                           self.bots[holder].get("blob")) for holder in moved}
        for holder, who in moved.items():
            self.bots[who]["track"], self.bots[who]["blob"] = tracks[holder]
            self.bots[who]["identified"] = True
        for c in set(lost):
            if c in self.bots:
                self.bots[c]["identified"] = False

        swaps = sum(1 for h, w in moved.items() if h != w)
        self.say(f"identified {len(moved)} of {len(self.bots)}"
                 + (f" — corrected {swaps} swap(s)" if swaps else " — no swaps")
                 + (f", {len(set(lost))} unreadable" if lost else ""),
                 MINT if not lost else SUN)

    # -- the agent ---------------------------------------------------------

    def agent_state(self):
        """What the model is told. Honest about what is NOT known.

        Every flag the dock shows that means "do not trust this" is repeated
        here. A model handed a position with no indication that the tracker is
        coasting, or that the frame is mirrored, will plan confidently on top
        of it -- and unlike a person looking at the window, it has no other
        way to notice.
        """
        bot = self.bots.get(self.code)
        pos = None
        if self.blob is not None and self.homography.ready:
            cm = self.to_cm(self.blob["xy"])
            pos = [round(float(cm[0]), 1), round(float(cm[1]), 1)]
        real, source = self.sizing_speed()
        out = {
            "robot": self.code,
            "position_cm": pos,
            "tracking": self.track.status,
            "driving": bool(self.armed),
            "path": None if self.path is None else self.path.kind,
            "speed_setting": int(self.speed),
            "measured_speed_cm_s": round(real, 1) if source == "measured" else None,
            "arrive_radius_cm": int(self.goal_tol),
        }
        if bot is not None and not bot.get("identified"):
            out["warning_identity"] = ("this robot has not been identified — "
                                       "the position is a real ball, the NAME "
                                       "on it is a guess")
        if self.mirrored:
            out["blocked"] = ("the arena frame is MIRRORED — nothing will "
                              "converge until a person flips an axis and "
                              "re-probes. Do not drive.")
        if not self.homography.ready:
            out["blocked"] = "no homography — there are no centimetres to aim in"
        if pos is None:
            # `self.why` carries the blown-frame verdict when there is one, and
            # the tracker's own status otherwise. An agent told the room is lit
            # can say so and stop; one told "no candidate" retries a drive
            # against a camera that cannot see anything at all.
            out["warning_position"] = f"no position: {self.why or self.track.status}"
        if getattr(self, "blown", False):
            out["blocked"] = ("the camera frame is BLOWN OUT — the room lights "
                              "are on, or the exposure is too high. Nothing "
                              "can be tracked until that is fixed")
        return out

    def agent_bounds(self):
        """The workspace quad in cm, or None. What the model may aim inside."""
        if self.corners is None or not self.homography.ready:
            return None
        pts = [np.asarray(self.to_cm(c), dtype=float) for c in self.corners]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]

    def goal_margin_cm(self):
        """Ball radius plus clearance -- how far a goal stays off the line."""
        return max(self.blob_radius_cm(), BALL_CM / 2.0) + EDGE_MARGIN_CM

    def agent_check(self, points):
        """Refuse a destination the rig cannot safely be sent to.

        Enforced HERE and not in the prompt. A prompt is a request; this is the
        thing that actually stops a ball being driven at a wall, and it has to
        hold whatever the model was asked or told.
        """
        if self.mirrored:
            return "refused: the frame is mirrored — flip and re-probe first"
        if not self.homography.ready:
            return "refused: no homography, so a centimetre means nothing"
        box = self.agent_bounds()
        if box is None:
            # REFUSED, not waved through. "Anywhere" is not a safe default for
            # a tool that drives a real ball across a real floor: without a
            # workspace there is no boundary to be inside of, so every
            # coordinate is as plausible as any other and none of them is
            # checked. A person clicking four corners takes ten seconds and is
            # the only thing that makes a refusal meaningful.
            return ("refused: no workspace is set — press c and click the four "
                    "corners of the arena first. Nothing may be driven to a "
                    "coordinate until there is a boundary to check it against")
        m = self.goal_margin_cm()
        x0, y0, x1, y1 = box[0] + m, box[1] + m, box[2] - m, box[3] - m
        if x0 >= x1 or y0 >= y1:
            return ("refused: the workspace is smaller than the ball plus its "
                    "clearance — there is nowhere safe to aim")
        for p in points:
            if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
                return (f"refused: ({p[0]:.0f}, {p[1]:.0f}) is too close to the "
                        f"edge. Aim inside x {x0:.0f}..{x1:.0f}, "
                        f"y {y0:.0f}..{y1:.0f} — the ball is {2*BALL_CM/2:.0f}cm "
                        "wide, so its centre cannot reach the boundary")
        return None

    def agent_call(self, name, args):
        """Run one tool. Errors come back as text for the model to read."""
        args = args or {}
        try:
            if name == "get_state":
                st = self.agent_state()
                box = self.agent_bounds()
                if box:
                    st["workspace_cm"] = [round(v, 1) for v in box]
                return st
            if name == "wait_until_arrived":
                # Waiting belongs in the TOOL, not in the model.
                #
                # Without this a model does the only thing left to it: calls
                # get_state, sees the ball still moving, and calls it again.
                # Measured against Sonnet that burned four of six calls and hit
                # the cap before the drive finished -- and each poll is a whole
                # round trip, so it is slow as well as wasteful.
                #
                # This sleeps on the agent's worker thread. The render thread
                # keeps ticking, which is what moves the robot, so blocking
                # here costs nothing but the agent's own patience.
                limit = float(np.clip(args.get("timeout_s", 20.0), 0.5, 60.0))
                until = time.perf_counter() + limit
                while self.armed and time.perf_counter() < until:
                    time.sleep(0.05)
                # Let the COAST finish before reading the position back.
                #
                # Driving stops the instant the arrival radius is satisfied,
                # and the ball then rolls on for about a second -- measured at
                # 8-17cm from working speed. Sampling at the moment of disarm
                # hands the model a position the ball has already left, and it
                # reports that as where it finished. Seen once: it announced
                # [88, 58] for a ball that came to rest at [94, 60].
                plant = self.plant if self.plant else plant_constants(self.code)
                settle = 1.2
                if plant and plant.get("coast_s"):
                    real, _ = self.sizing_speed()
                    settle = min(3.0, plant["coast_s"] * 2.0 + 0.3)
                rest = time.perf_counter() + (settle if not self.armed else 0.0)
                while time.perf_counter() < rest:
                    time.sleep(0.05)
                st = self.agent_state()
                # "STOPPED" IS NOT "ARRIVED", and reporting it as one is how a
                # model ends up announcing success for a ball twenty
                # centimetres short. Driving stops for four different reasons
                # -- arriving, giving up stuck, losing the ball, a person
                # pressing esc -- and `armed` went false for all of them
                # alike, so the tool used to return `arrived: True` for every
                # one. The controller knew better the whole time: `disarm`
                # works out a verdict for the log, and this now reads it.
                #
                # The reason is passed back verbatim as well. A model told
                # only that it failed can do nothing but stop; a model told it
                # is STUCK, and that the ball is against a boundary, can back
                # off and come at the goal from somewhere else -- which is the
                # recovery this rig actually has.
                out = {"position_cm": st["position_cm"],
                       "tracking": st["tracking"]}
                if self.armed:
                    repeats = (self.patrol is not None
                               or (self.path is not None and self.path.closed))
                    out.update(
                        arrived=False, timed_out=True,
                        outcome="patrolling" if repeats else "driving",
                        reason=("this path repeats until stopped — it does not "
                                "arrive" if repeats else
                                f"still driving after {limit:.0f}s"))
                    return out
                out.update(arrived=self.last_outcome == "arrived",
                           timed_out=False,
                           outcome=self.last_outcome or "unknown",
                           reason=self.last_note or "stopped for no stated reason")
                if self.last_outcome == "stuck":
                    out["retry_hint"] = ("the escape budget is refreshed on the "
                                         "next drive — a goal away from the "
                                         "boundary is worth trying before "
                                         "giving up")
                return out
            if name == "stop":
                self.disarm("stopped by the agent")
                return {"ok": True}
            if name == "set_speed":
                self.set_speed(int(np.clip(args.get("speed", 6), 0, 60)))
                return {"ok": True, "speed_setting": int(self.speed)}

            if name == "goto":
                p = [float(args["x"]), float(args["y"])]
                bad = self.agent_check([p])
                if bad:
                    return {"error": bad}
                self.path = Path.point(np.array(p, dtype=float))
            elif name == "follow_path":
                pts = [[float(a), float(b)] for a, b in args["points"]]
                if len(pts) < 2:
                    return {"error": "a path needs at least two points"}
                bad = self.agent_check(pts)
                if bad:
                    return {"error": bad}
                self.path = Path([np.array(p, dtype=float) for p in pts])
            elif name == "patrol":
                a = [float(args["x1"]), float(args["y1"])]
                b = [float(args["x2"]), float(args["y2"])]
                bad = self.agent_check([a, b])
                if bad:
                    return {"error": bad}
                self.start_patrol(a, b)
            elif name == "circle":
                c = [float(args["x"]), float(args["y"])]
                r = float(args["radius"])
                ring = Path.circle(np.array(c, dtype=float), r)
                bad = self.agent_check(list(ring.pts))
                if bad:
                    return {"error": bad}
                self.path = ring
            else:
                return {"error": f"unknown tool {name!r}"}
        except (KeyError, TypeError, ValueError) as e:
            return {"error": f"bad arguments for {name}: {e}"}

        before = self.armed
        self._arm_source = "agent"
        self.arm()
        if not self.armed:
            # `arm` refuses for reasons the model needs verbatim -- a mirrored
            # frame, no robot, no homography. Swallowing that and reporting
            # success is how an agent ends up planning on top of a rig that
            # never moved.
            return {"error": self.note or "arm refused", "was_driving": before}
        out = {"ok": True, "driving": True, "path": self.path.kind,
               "length_cm": round(self.path.length, 1)}
        if self.patrol is not None or self.path.closed:
            # Said plainly, because the model's next move is almost always
            # `wait_until_arrived` and this never arrives. A patrol that looks
            # like a drive gets waited on until the timeout and then read as a
            # failure.
            out["runs_until_stopped"] = True
            out["note"] = ("this repeats until you call stop — it will never "
                           "report arrived")
        return out

    AGENT_MAX_CALLS = 6
    """Hard cap on tool calls per command.

    Held by the app, never by the model. A loop whose exit the model controls
    is how a ball ends up driving in circles for ten minutes while someone
    reads the transcript trying to work out why. Six is enough for
    look-then-move-then-check and short of anything that could be called a
    plan.
    """

    def connect_model(self, preset=None):
        """A client if a model is reachable, else None and the app is unchanged.

        Never fatal. This is a bench tool for a camera and a ball; a language
        model is something it can use when one happens to be running, not a
        dependency. `llm/client.py` does the talking -- it already handles the
        OpenAI-compatible shape and pulls tool calls back out of models that
        emit them as prose, which is not worth writing twice.
        """
        try:
            from llm.client import client_for
            client = client_for(preset or self.agent_model)
            if hasattr(client, "reachable") and not client.reachable(timeout=1.0):
                return None
            return client
        except Exception:
            return None

    def agent_prompt(self):
        """The system prompt, rebuilt each turn with the state already in it.

        Injected rather than fetched, which is the pattern `llm/prompt.py`
        settled on: a model that has to spend a turn calling `get_state`
        before it can act takes twice as long to do anything, and the text is
        a couple of hundred tokens that are always current.
        """
        st = self.agent_state()
        box = self.agent_bounds()
        lines = [
            "You drive ONE Sphero on a flat floor, seen from above by a camera.",
            "Positions are centimetres in the arena frame: x right, y DOWN.",
            "",
            "State right now:",
            f"  robot {st['robot']}  at {st['position_cm']}  ({st['tracking']})",
            f"  driving: {st['driving']}   path: {st['path']}",
            f"  arrive radius: {st['arrive_radius_cm']}cm",
        ]
        if box:
            lines.append(f"  workspace: x {box[0]:.0f}..{box[2]:.0f}, "
                         f"y {box[1]:.0f}..{box[3]:.0f} — stay inside it")
        for k in ("blocked", "warning_identity", "warning_position"):
            if st.get(k):
                lines.append(f"  {k.upper()}: {st[k]}")
        lines += [
            "",
            "Rules:",
            "  Use the tools; do not describe what you would do.",
            "  If a tool returns an error, read it and fix the call. The error",
            "    text is accurate — it is not a suggestion to try harder.",
            "  If the state says BLOCKED, say so and call nothing.",
            "  You cannot connect robots, flip the frame, zero the aim or run",
            "    the probe. A person does those.",
        ]
        return "\n".join(lines)

    def agent_run(self, command, client=None):
        """One command, start to finish. Returns a transcript.

        Synchronous and pure enough to test: the thread lives in `ask`, so
        this can be driven by a stub model with no pygame and no Ollama.
        """
        client = client or self.agent_client
        if client is None:
            return [("error", "no model — is Ollama running?")]
        messages = [{"role": "system", "content": self.agent_prompt()},
                    {"role": "user", "content": command}]
        log = []
        for _ in range(self.AGENT_MAX_CALLS):
            try:
                reply = client.chat(messages, AGENT_TOOLS)
            except Exception as e:
                log.append(("error", f"{type(e).__name__}: {e}"))
                return log
            if not reply.tool_calls:
                log.append(("say", reply.text or "(nothing)"))
                return log
            messages.append({"role": "assistant", "content": reply.text or "",
                             "tool_calls": [
                                 {"id": c.id, "type": "function",
                                  "function": {"name": c.name,
                                               "arguments": json.dumps(c.arguments)}}
                                 for c in reply.tool_calls]})
            for c in reply.tool_calls:
                result = self.agent_call(c.name, c.arguments)
                log.append(("call", f"{c.name}({c.arguments}) -> {result}"))
                # Errors go back VERBATIM. `llm/agent.py` found this recovers
                # most first-attempt failures on the second try, and does more
                # than any amount of prompt tuning.
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(result)})
        log.append(("error", f"stopped after {self.AGENT_MAX_CALLS} tool calls"))
        return log

    def ask(self, command):
        """Run a command on a worker thread, so the window keeps drawing.

        A pygame loop that blocks for the seconds a model takes looks crashed,
        which is how somebody ends up pressing the button four more times.
        """
        if self.agent_busy:
            self.say("the agent is still working", SUN)
            return
        self.agent_busy = True
        self.agent_log = [("you", command)]

        def work():
            try:
                self.agent_log += self.agent_run(command)
            except Exception as e:
                self.agent_log.append(("error", f"{type(e).__name__}: {e}"))
            finally:
                self.agent_busy = False

        threading.Thread(target=work, daemon=True, name="agent").start()
        self.say(f"agent: {command}")

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
        if self.reid is not None:
            self.step_reid()
            return
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
        # An escape nudge owns the wheels while it runs, so pursuit cannot
        # steer straight back into the wall it is reversing off.
        if self.unstick and self.unstick.get("at") and self.step_unstick(h):
            return
        if self.coasting is not None and self.step_coast(h):
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
        plant = self.plant if self.plant else plant_constants(self.code)
        coast = (self.coast_learned if self.coast_learned is not None
                 else ((plant.get("coast_s") or 0.0) if plant else 0.0))
        # The taper has to shed speed over the distance the BALL needs, which
        # is set by how fast it is really going -- not by the number on the
        # slider, which on this hardware is out by a factor of four.
        real, _ = self.sizing_speed()
        scale = max(real / max(float(self.speed), 1e-6), 1.0)
        v, target, done, note = pursue(self.path, pos, self.lookahead,
                                       self.speed, self.goal_tol,
                                       self.blob_radius_cm(),
                                       coast_s=(coast * scale) if self.taper else 0.0,
                                       min_speed=MIN_MOVING_CM_S)
        self.target_cm, self.drive_note = target, note

        # LET GO EARLY, on the last approach of an open path. The ball was
        # going to coast the final stretch whatever we did; cutting the motors
        # deliberately turns that into a measurement of how far it carries,
        # which is the one constant here still borrowed from another ball.
        if (COAST_LEARN and not done and self.path is not None
                and not self.path.closed and self.coasting is None):
            here = np.asarray(pos, dtype=float)
            end = self.path.ring[-1]
            gap = float(np.linalg.norm(end - here)) - self.goal_tol \
                - self.blob_radius_cm()
            now_speed = self.true_speed()
            if now_speed and 0.0 < gap <= now_speed * max(coast, 0.4):
                self.start_coast(gap, now_speed)
                self.step_coast(h)
                return
        if done:
            if self.step_patrol():
                return
            h.stop()
            self.disarm("arrived — stopped")
            return
        if self.style == "turn-go" or (self.style == "align"
                                       and not self.aligned):
            v = self.turn_and_go(v)
            # `turn_and_go` clears `self.turning` when the camera confirms the
            # ball is creeping the right way. Under `align` that is the whole
            # job: hand over to plain pursuit and do not come back, however far
            # the bearing wanders later.
            if self.style == "align" and self.turning is None:
                self.aligned = True
                self.drive_note = "aligned — pursuing"
        if self.stall_report() is not None:
            tries = (self.unstick or {}).get("tries", 0)
            if tries < UNSTICK_TRIES:
                self.start_unstick()
                return
            self.disarm(f"stuck after {tries} attempts to back off — "
                        "move it clear and re-arm")
            return
        if self._boot_pending:
            self._boot_pending = False
            self.start_bootstrap(pos)
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
                # The line the ball actually stops on: the goal grown by the
                # blob's own radius, because arrival is circle to circle.
                # Drawn as well as the goal so the difference is visible
                # rather than a number in a docstring.
                grown = self.blob_radius_cm()
                if grown > 0.1:
                    e2 = path_px([end + np.array([self.goal_tol + grown, 0.0])])[0]
                    pygame.draw.circle(self.screen, GREY, mid,
                                       int(max(5, abs(e2[0] - mid[0]))), 1)
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

        # Every connected robot, with the unselected ones drawn plainly and
        # the contended ones flagged. A marker that looks equally confident
        # whichever robot it is on is how a swap goes unnoticed.
        for code, bot in self.bots.items():
            if code == self.code:
                continue
            b, tr = bot.get("blob"), bot["track"]
            at = b["xy"] if b is not None else tr.predicted
            if at is None:
                continue
            c = px(at)
            hot = self.contended(code)
            named = bot.get("identified")
            tone = SUN if (hot or not named) else bot["marker"]
            # Hollow while unidentified or contended: the ring is this app's
            # claim about WHICH robot that is, and it must not look the same
            # when the claim is a guess.
            pygame.draw.circle(self.screen, tone, c, 13,
                               1 if (b is None or not named) else 2)
            self.text(code + ("" if named else " ?"), c[0] + 16, c[1] - 7,
                      tone, self.fs)

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
        if self.typing:
            box = pygame.Rect(x, H - 58, w, 22)
            card(self.screen, box)
            pygame.draw.rect(self.screen, CYAN, box, 1, border_radius=5)
            self.text("> " + self.typed + "_", box.x + 6, box.y + 4, CHALK,
                      self.fs)
        self.text("1 pt 2 line 3 poly 4 circ 5 free  g go", x, H - 36,
                  GREY, self.fs)
        self.text("c corners t target b scan  / ask  esc STOP", x, H - 22,
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
        want_jump = self.jump_advice()
        if want_jump:
            loose = self.track.max_jump > want_jump * 2.5
            y = self.row(f"max jump {self.track.max_jump} px — wants "
                         f"~{want_jump:.0f}" + ("  <-- LOOSE" if loose else ""),
                         x, y, SUN if loose else DIM, self.fs)
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
        stall = self.stall_report()
        if stall:
            for line in wrap(stall[0], 44):
                y = self.row(line, x, y, stall[1], self.fs)
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
            real, source = self.sizing_speed()
            want = latency_advice(real, plant)
            y = self.row(f"speed {real:.0f} cm/s ({source})   coasts "
                         f"{real * (plant.get('coast_s') or 0.5):.0f}cm",
                         x, y, SUN if source == "measured"
                         and abs(real - self.speed) > 4 else DIM, self.fs)
            if source == "measured" and abs(real - self.speed) > 4:
                # The slider is not centimetres per second and it matters.
                y = self.row(f"slider says {self.speed} — sizing off the "
                             "measurement", x, y, SUN, self.fs)
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

        y += 8
        y = self.sect(self.screen, self.fs, "agent", x, y, w,
                      (f"{self.agent_model} thinking…" if self.agent_busy else
                       (self.agent_model if self.agent_client else "no model")),
                      SUN if self.agent_busy else
                      (MINT if self.agent_client else DIM))
        if not self.agent_log:
            y = self.row("press / to ask", x, y, DIM, self.fs)
        for kind, text in self.agent_log[-6:]:
            tone = {"you": CHALK, "say": MINT, "error": CORAL}.get(kind, DIM)
            for line in wrap(f"{kind}: {text}", 44)[:3]:
                y = self.row(line, x, y, tone, self.fs)
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

        bot = self.bots.get(self.code)
        if bot is not None:
            rgb = tuple(int(v) for v in bot["rgb"])
            sw = pygame.Rect(x, y, w, 20)
            pygame.draw.rect(self.screen, rgb, sw, border_radius=4)
            pygame.draw.rect(self.screen, RULE, sw, 1, border_radius=4)
            y += 26
            hot = [n for n, v in zip("RGB", rgb) if v >= 250]
            blob = bot.get("blob")
            peak = None if blob is None else blob["peak"]
            y = self.row(f"{self.code}  rgb {rgb[0]},{rgb[1]},{rgb[2]}"
                         + (f"   peak {peak:.0f}" if peak is not None else ""),
                         x, y, CORAL if (peak or 0) >= 250 else DIM, self.fs)
            if hot:
                # A channel at the rail is a channel with no headroom -- the
                # core is blowing there whatever the camera does next.
                y = self.row(f"channel {'/'.join(hot)} at the rail — trim it",
                             x, y, CORAL, self.fs)
            y += 6

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
            y = self.row(f"{self.log.rows} rows — press log to reveal",
                         x, y, DIM, self.fs)
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
        # TYPING SWALLOWS EVERYTHING. Without this, typing "stop" would fire
        # save, target, taper and probe on the way past -- every letter in this
        # app is a shortcut, and a text box that leaves them live is a text box
        # that reconfigures the rig while you write to it.
        if self.typing:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                text, self.typed, self.typing = self.typed.strip(), "", False
                if text:
                    self.ask(text)
            elif e.key == pygame.K_ESCAPE:
                self.typed, self.typing = "", False
            elif e.key == pygame.K_BACKSPACE:
                self.typed = self.typed[:-1]
            elif e.unicode and e.unicode.isprintable():
                self.typed = (self.typed + e.unicode)[:120]
            return True
        if e.key == pygame.K_SLASH:
            self.typing, self.typed = True, ""
            return True

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
                   pygame.K_i: self.start_reid,
                   pygame.K_o: self.toggle_taper,
                   pygame.K_TAB if False else pygame.K_n: self.cycle_selected,
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
        # The bridge goes FIRST. It reads the tracker on its own thread, and
        # letting it run while the camera and the fleet are torn down beneath
        # it is how a clean exit turns into a traceback from a daemon thread.
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge.join(timeout=1.0)
        # Before anything else that matters: a window that closes while the
        # ball is still rolling is the worst possible exit.
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
    p.add_argument("--rpc", action="store_true",
                   help="publish position into the VLM framework's DataService "
                        f"as robot {RPC_ROBOT_ID} (its RPC server must be up)")
    p.add_argument("--model", default=None,
                   help="a preset from llm/models.yaml, e.g. qwen3.5:4b. "
                        "Default qwen3.5:9b; omit the model entirely with none")
    a = p.parse_args(argv)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    BlobTest(a.camera, size=size, exposure=a.exposure, code=a.robot,
             with_fleet=not a.no_fleet, model=a.model, rpc=a.rpc).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
