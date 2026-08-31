#!/usr/bin/env python3
"""Pose bench — read x, y and theta from the lights, and tune both ends at once.

    python pose_test.py --camera 0
    python pose_test.py --source sim        # no camera, no robots

The method, end to end, is four steps and no inference:

    1. threshold at V >= 200. On a correctly underexposed frame that is the
       LEDs and nothing else -- not the floor, not a window, not a shoe.
    2. connected components. One dot per component, with its centroid and its
       mean BGR.
    3. group dots within one ball diameter of each other. Each group is a robot.
    4. of the three dots in a group, the blue one is the tail and the other two
       are the tag.

From which: POSITION is the midpoint of the two tag dots -- exact, because they
are mounted symmetrically about the centre, so there is no offset to calibrate
and nothing that leans as the robot turns. HEADING is the direction from the
tail dot to that midpoint -- available every frame, with nothing driven, which
is the entire reason this exists. IDENTITY is the chromaticity of the two tag
dots against the palette in `vision/config.py`.

Why chromaticity and never a plain RGB sum: the palette's red is (255, 0, 0)
and its green is (0, 255, 0), so the two sum to exactly the same number and a
sum carries no information about which is which. And a blown pixel is
(255, 255, 255)
whatever LED made it -- which is why the LED brightness sliders are here beside
the threshold. Dividing each pixel by its own total throws away brightness and
keeps the ratio between the channels, which is the only part that names a robot.
Chromaticity still cannot save a fully clipped core (white normalises to a flat
third in every channel), so the dock reports the clipped fraction and the margin
between the best and second-best colour. Dim the LEDs until both look healthy;
that is the loop this app is built around.

IT REFUSES RATHER THAN GUESSES. A group that is not exactly one blue plus two
tag dots produces a reason on screen -- "two blues", "only 2 dots", "nothing
above threshold" -- and no pose. A silent frame is recoverable by the next
frame. A confident wrong heading is not: it goes into a controller, which drives
on it. Every refusal in this file is deliberate and none of them should be
softened into a best guess.

WHAT THIS DELIBERATELY DOES NOT DO is filter. No Kalman, no smoothing, no
carrying a pose forward through a bad frame. Those belong in a tracker and they
are wrong in a bench, because their whole purpose is to hide the frames this app
is for looking at.

Keys — s save the raw frame to runs/, l lock exposure, u unlock it,
v raw/mask view, space pause, esc quit.
"""

import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import pygame

from ui.theme import (CARD, CHALK, CORAL, CYAN, DIM, GREY, INK, MINT, PAD,
                      PANEL, RULE, SUN, Button, Slider, card, section)
from vision import config
from vision.homography import Homography

# -- the method's constants ------------------------------------------------

V_MIN = 200
"""Threshold on V. High on purpose: on a correctly exposed frame for this job
the only things above 200 are LED cores, so the component step gets dots and
not shapes. If the floor appears at this threshold the exposure is wrong, and
that is a camera problem to fix with the exposure controls rather than a number
to lower until the picture looks nice."""

GROUP_PX = 28
"""One ball diameter in pixels. Must span a robot's own three dots and stay
under the gap to the next robot -- so it follows the ball's size on screen and
belongs on a slider, not in the source."""

MIN_DOT_AREA = 4
"""Below this a component is sensor noise that cleared the threshold. Small,
because at range a real dot is only a handful of pixels and a floor that costs
us dots at range is worse than one stray blob we would refuse anyway."""

TAIL_HUE = (100, 130)
"""Hue window for the taillight, wider than `config.COLORS["blue"]` (102-122).

Deliberate: the palette entry is tuned for finding a whole lit SHELL, and this
reads a bright core instead. A core near clipping loses saturation and its hue
wanders several counts. The window has room to spare below magenta (150) and
above cyan (90), so widening it costs nothing and recovers frames that a
shell-tuned tolerance drops."""

CLIP_AT = 250
"""A channel at or above this is clipped for our purposes. Not 255: a mean over
a component is an average, so a dot whose core is fully blown still averages a
little under 255 once its edge pixels are included."""

TAG_NAMES = [n for n in config.COLORS if n != "blue"]
"""Blue belongs to the taillight and to nothing else, so no robot may be tagged
blue -- its nose and tail would be the same colour and there would be no
heading to read. The roster enforces this; this list is the detector's half."""

W, H = 1480, 940
MIN_W, MIN_H = 1100, 760
DOCK_W = 372
FLICKER = 8.0
"""Counts of mean brightness. Below this the picture did not move -- borrowed
from `vision/expose.py`, which graded exposure controls the same way."""

LIT_STEP = 200
"""Pixels. The other half of the same judgement, for a frame too dark for the
mean to say anything: an exposure change that is doing something moves at
least this many pixels across the threshold."""


# -- reading a pose --------------------------------------------------------

def reference_chromaticity(names=TAG_NAMES):
    """Where each palette colour sits once brightness is divided out.

    Built from `config.COLORS` so there is one palette in this project and not
    two. The saturation and value used to turn a hue into a BGR are arbitrary
    -- chromaticity normalises both away -- but they match `vision/dots.py` so
    the two agree about distances rather than merely about rankings.
    """
    out = {}
    for name in names:
        hue = config.COLORS[name]["hue"]
        bgr = cv2.cvtColor(np.uint8([[[hue, 235, 250]]]),
                           cv2.COLOR_HSV2BGR)[0, 0].astype(float)
        out[name] = bgr / max(bgr.sum(), 1e-6)
    return out


REFERENCE = reference_chromaticity()


def chromaticity(bgr):
    """bgr / bgr.sum(). Brightness out, the ratio between channels left."""
    bgr = np.asarray(bgr, dtype=float)
    return bgr / max(float(bgr.sum()), 1e-6)


def hue_of(bgr):
    """OpenCV hue (0-179) of one mean BGR triple."""
    px = np.uint8([[[int(round(np.clip(c, 0, 255))) for c in bgr]]])
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])


def find_dots(frame, v_min=V_MIN, min_area=MIN_DOT_AREA):
    """Every bright component, with its centroid and its mean BGR.

    The mean is taken over the component's own pixels -- the ones that cleared
    the threshold -- rather than over a disc around it. A disc drags in dark
    floor, which pulls every colour towards black and towards each other, and
    the whole identity step depends on those colours staying apart.
    """
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask = (v >= int(v_min)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return [], mask

    flat = labels.reshape(-1)
    px = frame.reshape(-1, 3).astype(np.float64)
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    sums = np.stack([np.bincount(flat, weights=px[:, c], minlength=n)
                     for c in range(3)], axis=1)

    dots = []
    for i in range(1, n):                       # 0 is the background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bgr = sums[i] / max(counts[i], 1.0)
        dots.append({"xy": np.array(cents[i], dtype=float),
                     "bgr": bgr,
                     "area": area,
                     "hue": hue_of(bgr),
                     "clipped": bool(bgr.max() >= CLIP_AT)})
    return dots, mask


def group_dots(dots, radius=GROUP_PX):
    """Union-find over "within one ball diameter". Each group is one robot.

    Transitive on purpose: a robot's three dots span more than the gap between
    neighbouring dots, so pairwise-within-radius has to chain along the line or
    a ball splits into a nose group and a tail group -- and each half then
    reports "only 2 dots" and "only 1 dot", which reads as the method failing
    when it is only this radius being too small.
    """
    parent = list(range(len(dots)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    r = float(radius)
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            if float(np.linalg.norm(dots[i]["xy"] - dots[j]["xy"])) <= r:
                parent[root(i)] = root(j)

    groups = {}
    for i, dot in enumerate(dots):
        groups.setdefault(root(i), []).append(dot)
    return list(groups.values())


def read_group(group, tail_hue=TAIL_HUE):
    """One group of dots -> a pose, or a reason there isn't one.

    Returns `(pose, why)`, exactly one of which is None. Every branch that
    returns a reason is a branch where guessing was possible and refused. Two
    blues could be resolved by taking the bluer one; three dots with no blue
    could be resolved by assuming the dimmest is the tail. Both of those
    produce a heading that is confidently backwards some of the time, and a
    backwards heading is the one failure this project cannot absorb.
    """
    n = len(group)
    if n < 3:
        return None, f"only {n} dot{'' if n == 1 else 's'}"

    # Counted before the colours are looked at, and the order matters. Two
    # balls inside one radius make a group of six with two blues in it, and
    # answering "two blues" there points at the taillights when the actual
    # problem is the radius -- a reason that sends you tuning the wrong knob is
    # barely better than no reason.
    if n > 3:
        return None, f"{n} dots in one group — two robots, or the radius is too big"

    lo, hi = tail_hue
    blues = [d for d in group if lo <= d["hue"] <= hi]
    tags = [d for d in group if not (lo <= d["hue"] <= hi)]

    if len(blues) > 1:
        return None, "two blues"
    if not blues:
        return None, "no blue — the taillight is off, or this robot is tagged blue"
    if len(tags) != 2:
        return None, f"{len(tags)} tag dot{'' if len(tags) == 1 else 's'}, need 2"

    tail = blues[0]
    centre = (tags[0]["xy"] + tags[1]["xy"]) / 2.0
    step = centre - tail["xy"]
    span = float(np.linalg.norm(step))
    if span < 2.0:
        return None, "the taillight sits on the centre — no direction in that"

    # Identity from both tag dots at once. Averaged as chromaticities rather
    # than as raw BGR, so the brighter of the two dots does not outvote the
    # other -- they are the same LED colour and deserve the same say.
    chroma = (chromaticity(tags[0]["bgr"]) + chromaticity(tags[1]["bgr"])) / 2.0
    ranked = sorted((float(np.linalg.norm(chroma - c)), name)
                    for name, c in REFERENCE.items())
    dist, name = ranked[0]
    # The gap to the runner-up, not the distance to the winner. Every colour is
    # far from a washed-out white, so a large distance alone does not mean the
    # match is wrong -- but a small gap always means it is a coin toss.
    margin = ranked[1][0] - dist if len(ranked) > 1 else float("inf")

    return {"centre": centre,
            "tail": tail["xy"],
            "tags": [t["xy"] for t in tags],
            "deg": math.degrees(math.atan2(step[1], step[0])) % 360.0,
            "span": span,
            "name": name,
            "dist": dist,
            "margin": margin,
            "clipped": sum(1 for d in group if d["clipped"]),
            "area": sum(d["area"] for d in group)}, None


def read_frame(frame, v_min=V_MIN, radius=GROUP_PX, tail_hue=TAIL_HUE):
    """Every robot in the frame, plus every reason one was refused.

    Refusals are always `(reason, dots)` pairs, even the frame-wide one, which
    carries an empty group. A caller that has to check the shape before reading
    it will one day forget on the rarest path -- and the rarest path here is
    "the threshold is above every dot", which is precisely the state somebody
    is in while dragging the threshold slider looking for the edge.
    """
    dots, mask = find_dots(frame, v_min)
    if not dots:
        return [], [("nothing above threshold", [])], dots, mask

    poses, why = [], []
    for group in group_dots(dots, radius):
        pose, reason = read_group(group, tail_hue)
        if pose:
            poses.append(pose)
        else:
            why.append((reason, group))
    return poses, why, dots, mask


def compass(image_deg):
    """Image degrees (x right, y down) as a Sphero heading (0 is +y, clockwise)."""
    return (90.0 - float(image_deg)) % 360.0


def exposure_verdict(seen):
    """Does this camera's exposure control do anything? `(message, tone)`, or None.

    `seen` maps an exposure value to the `(mean brightness, lit pixels)` the
    picture settled at. Judged across every value tried rather than on the last
    one: a single step may legitimately move the picture very little, while a
    sweep from -13 to 0 that moves nothing is a control that does nothing.

    Two signals, because neither alone covers both ends of this app's working
    range. On a lit room the mean brightness carries the answer. On the black
    arena this method actually wants, the mean is under a count and cannot --
    but the number of pixels over the threshold still moves. Requiring EITHER
    is what keeps the verdict honest in both.
    """
    if len(seen) < 2:
        return None
    means = [m for m, _ in seen.values()]
    lits = [n for _, n in seen.values()]
    spread = max(means) - min(means)
    lit_spread = max(lits) - min(lits)
    tried = len(seen)
    evidence = (f"{tried} exposure values moved brightness {spread:.0f} "
                f"counts and lit pixels by {lit_spread}")
    if spread >= FLICKER or lit_spread >= max(LIT_STEP, 0.2 * max(lits)):
        return (f"exposure works — {evidence}", MINT)
    return (f"THE CAMERA IGNORED IT — {evidence}. Use LOCK instead.", CORAL)


# -- the camera ------------------------------------------------------------

class Camera(threading.Thread):
    """Frames on their own thread.

    A blocking read on the render thread is a frozen window, and a frozen
    window during bring-up reads as a crash -- which sends you looking for a
    bug in the app when the camera is simply slow to hand over a frame.
    """

    def __init__(self, spec, size=None):
        super().__init__(daemon=True, name="pose-cam")
        self.source = None
        self.error = None
        try:
            if str(spec) in ("sim", "balls"):
                self.source = BallSource()
            else:
                from vision.synthetic import open_source
                self.source = open_source(spec, size=size)
        except Exception as e:
            # A camera that will not open must not cost you the window. On
            # macOS the usual cause is that this process has no camera
            # permission, or another app is holding the device, and a
            # traceback in a terminal is a worse way to learn either.
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
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self._stop.set()
        self.join(timeout=1.0)
        # Joined before releasing: `VideoCapture.release()` while the grab
        # thread is inside `read()` is a segfault on some backends.
        try:
            if self.source is not None:
                self.source.release()
        except Exception:
            pass


class BallSource:
    """A fake camera with the REAL light arrangement, for working without one.

    Draws through `vision/shots.py`, so the dots it renders are the three
    equal cores this app is built to read -- two tag, one blue, correctly
    spaced. `vision/synthetic.py` draws one coloured disc per robot instead,
    which is right for the colour tracker and has no heading in it at all.
    """

    def __init__(self, n=4, size=(1280, 720), px_cm=9.2, seed=0):
        from vision.shots import TAGS
        self.w, self.h = size
        self.px_cm = px_cm
        self.rng = np.random.default_rng(seed)
        self.tags = (list(TAGS) * 3)[:n]
        w_cm, h_cm = self.w / px_cm, self.h / px_cm
        self.pos = np.stack([self.rng.uniform(14, w_cm - 14, n),
                             self.rng.uniform(14, h_cm - 14, n)], axis=1)
        self.heading = self.rng.uniform(0, 360, n)
        self.turn = self.rng.uniform(-30, 30, n)
        self.bounds = (w_cm, h_cm)

    GAIN = 1.0 / 44.0
    """Exposure for the fake camera, chosen to sit just under clipping.

    `vision.shots.as_short` uses 1/50, which is tuned for the peak-finding
    method in `vision/facing.py` and leaves the cores peaking near 217 -- so
    only a handful of pixels per dot clear a threshold of 200, and a dot that
    loses a few to noise drops out entirely. That produces "only 2 dots"
    refusals that belong to the fake camera and not to the method, which is
    exactly the wrong thing for a bench to teach you.

    At 1/44 the cores peak near 247: comfortably above the threshold and not
    yet clipped, which is the frame this app is specified for and the one the
    LED brightness sliders exist to reach on real hardware.
    """

    def read(self):
        from vision.shots import draw_ball
        dt = 1 / 30.0
        self.heading = (self.heading + self.turn * dt) % 360.0
        step = np.stack([np.cos(np.radians(self.heading)),
                         np.sin(np.radians(self.heading))], axis=1) * 7.0 * dt
        self.pos += step
        for ax, hi in enumerate(self.bounds):
            out = (self.pos[:, ax] < 11) | (self.pos[:, ax] > hi - 11)
            self.heading[out] = (self.heading[out] + 150.0) % 360.0
            self.pos[:, ax] = np.clip(self.pos[:, ax], 11, hi - 11)

        canvas = np.zeros((self.h, self.w, 3), np.float32)
        canvas[:] = (6.0, 5.0, 4.0)
        for i, tag in enumerate(self.tags):
            draw_ball(canvas, self.pos[i, 0], self.pos[i, 1],
                      float(self.heading[i]), tag, self.px_cm)
        canvas += self.rng.normal(0.0, 1.6, canvas.shape).astype(np.float32)
        return True, np.clip(canvas * self.GAIN, 0, 255).astype(np.uint8)

    def release(self):
        pass


# -- the app ---------------------------------------------------------------

def to_surface(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")


def fit_to_display():
    global W, H
    try:
        sw, sh = pygame.display.get_desktop_sizes()[0]
    except Exception:
        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h
    if sw and sh:
        W = max(MIN_W, min(W, sw - 40))
        H = max(MIN_H, min(H, sh - 80))
    return W, H


class PoseTest:
    def __init__(self, spec="0", size=None, exposure=-7, with_fleet=True):
        pygame.init()
        pygame.display.set_caption("pose test — x, y, theta from the lights")
        fit_to_display()
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 17)

        self.v_min = V_MIN
        self.group_px = GROUP_PX
        self.exposure = int(exposure)
        self.paused = False
        self.view = "raw"
        self.note = ""
        self.note_tone = DIM
        self.frame = None
        self.mask = None
        self.poses, self.why, self.dots = [], [], []
        self.mean_v = 0.0
        self.lit_px = 0

        # Exposure bookkeeping. `exp_seen` maps an exposure value to the mean
        # brightness the picture settled at, which is the only evidence that
        # separates a control that works from one that is quietly ignored --
        # a camera that dropped the write reports back whatever you asked for.
        self.exp_took = None
        self.exp_seen = {}
        self._probe = None
        self.exposure_locked = False

        self.cam = Camera(spec, size=size)
        self.cam.start()
        if self.cam.error:
            self.say(self.cam.error, CORAL)

        self.homography = Homography.load()
        if not self.homography.ready:
            self.say("no calib/homography.json — positions stay in pixels", SUN)

        self.fleet = None
        self.leds = {}          # code -> {"color": name, "bright": int}
        self.tail_bright = 255
        if with_fleet:
            self.build_fleet()

        self.sliders, self.buttons = [], []
        self.build_dock()

    # -- fleet ------------------------------------------------------------

    def build_fleet(self):
        """The real robots, through the layer that already knows how to talk.

        Built here rather than reimplemented: `fleet` handles the BLE worker
        threads, the connect stagger and the write deduplication, and a bench
        that opened its own radio would be a second implementation to keep in
        step with the first. Failure is not fatal -- the pose half of this app
        is useful with no robots powered at all.
        """
        try:
            from fleet.manager import Fleet
            self.fleet = Fleet.from_roster()
        except Exception as e:
            self.say(f"no fleet: {type(e).__name__}: {e}", SUN)
            return
        if self.fleet.errors:
            self.say(f"roster: {self.fleet.errors[0]}", CORAL)
        for code, h in self.fleet.handles.items():
            self.leds[code] = {"color": h.color, "bright": config.LED_VALUE}
        # A robot tagged blue can never be read by this method, because blue is
        # how the tail is found -- its nose and its tail would be the same
        # colour and there would be no direction in the group at all. It shows
        # up as "two blues", which reads as a detector fault rather than as a
        # roster one, so it is worth naming here where the cause is.
        blue = [c for c, led in self.leds.items() if led["color"] == "blue"]
        if blue:
            self.say(f"{', '.join(blue)} tagged blue — blue is the taillight, "
                     "so this robot cannot be read. Click its code to relight it.",
                     CORAL)
        self.push_leds()

    def push_leds(self, codes=None):
        """Send the current colour and brightness to every robot named.

        One hue drives both what the ball is TOLD to glow and what this app
        HUNTS for, because `config.led_rgb` derives the RGB from the same
        palette entry the detector matches against. Two tables is how a robot
        ends up lit one colour and looked for as another, each internally
        consistent and nothing complaining.
        """
        if not self.fleet:
            return
        for code in (codes or list(self.leds)):
            h = self.fleet.get(code)
            if h is None:
                continue
            led = self.leds[code]
            try:
                h.set_led(config.led_rgb(config.COLORS[led["color"]]["hue"],
                                         value=led["bright"]))
                h.set_back_led(int(self.tail_bright))
            except Exception as e:
                self.say(f"{code}: {type(e).__name__}: {e}", CORAL)

    def cycle_color(self, code):
        led = self.leds.get(code)
        if not led:
            return
        i = TAG_NAMES.index(led["color"]) if led["color"] in TAG_NAMES else -1
        led["color"] = TAG_NAMES[(i + 1) % len(TAG_NAMES)]
        self.push_leds([code])
        self.say(f"{code} lit {led['color']}")

    def set_bright(self, code, value):
        self.leds[code]["bright"] = int(value)
        self.push_leds([code])

    def set_tail(self, value):
        self.tail_bright = int(value)
        self.push_leds()

    # -- camera controls --------------------------------------------------

    def set_exposure(self, value):
        """Ask for an exposure, and start finding out whether it took.

        Reading the property back is NOT enough and this is the trap the whole
        warning exists for: a camera that ignored the write happily reports the
        value you asked for. Only the picture settles it, so the readback is
        recorded and a brightness probe is started alongside it.
        """
        self.exposure = int(value)
        src = self.cam.source
        if src is None or not hasattr(src, "set"):
            self.say("this source has no camera controls", DIM)
            return
        self.exp_took = src.set("exposure", self.exposure)
        self._probe = {"asked": self.exposure, "mean": self.mean_v,
                       "lit": self.lit_px, "at": time.time()}

    def check_probe(self):
        """Did the picture move? Answered a beat after the write, not instantly.

        A change lands several frames later on a buffered stream, so measuring
        immediately reports "no effect" for a control that works.
        """
        if not self._probe or time.time() - self._probe["at"] < 0.6:
            return
        p, self._probe = self._probe, None
        self.exp_seen[p["asked"]] = (self.mean_v, self.lit_px)
        moved = abs(self.mean_v - p["mean"])
        lit = abs(self.lit_px - p["lit"])
        if moved >= FLICKER or lit >= max(LIT_STEP, 0.2 * p["lit"]):
            self.say(f"exposure {p['asked']}: brightness {moved:+.0f}, "
                     f"lit pixels {self.lit_px - p['lit']:+d}", MINT)

    @property
    def exposure_verdict(self):
        return exposure_verdict(self.exp_seen)

    def exposure_mode(self, lock):
        """Freeze the camera's own metering, or hand it back.

        On many macOS webcams this is the only exposure control that exists:
        OpenCV's AVFoundation backend accepts `CAP_PROP_EXPOSURE` and drops it,
        and the device reports `Custom` unsupported, so there is no shutter or
        ISO to set. `Locked` is still enough, because the problem was never the
        absolute value -- it is that a metering camera AMPLIFIES as you darken
        the room, which blows the cores into one blob. Expose against something
        bright, lock, then darken the room.
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
        got = src.manual()
        self.set_exposure(self.exposure)
        self.say(f"manual: {got}")

    def save_frame(self):
        """The RAW frame to runs/, never the overlay.

        What is worth keeping is what the camera actually delivered -- the
        overlay is this app's opinion about it, and an opinion baked into a
        pixel cannot be re-read later with a different threshold.
        """
        if self.frame is None:
            self.say("no frame to save", SUN)
            return
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"pose_{time.strftime('%m%d_%H%M%S')}.png")
        cv2.imwrite(path, self.frame)
        self.say(f"saved {os.path.relpath(path)}", MINT)

    def say(self, msg, tone=DIM):
        self.note, self.note_tone = msg, tone
        print(f"pose_test: {msg}")

    # -- loop -------------------------------------------------------------

    def tick(self):
        if not self.paused:
            frame = self.cam.latest()
            if frame is not None:
                self.frame = frame
        if self.frame is None:
            return
        # Subsampled: a mean over every fourth pixel tracks the exposure just
        # as well and costs a twentieth of the time, on the render thread.
        v = self.frame[::4, ::4].max(axis=2)
        self.mean_v = float(v.mean())
        # The second exposure signal, and on this app's frames the load-bearing
        # one. A correctly underexposed arena is BLACK apart from a few dots,
        # so its mean brightness is under a count and a real exposure change
        # barely moves it -- judging by the mean alone would report a working
        # control as ignored, which is the same lie in the other direction.
        # How much of the frame is lit does move, and keeps moving right up to
        # the point where the floor arrives.
        self.lit_px = int((v >= self.v_min).sum())
        self.check_probe()
        self.poses, self.why, self.dots, self.mask = read_frame(
            self.frame, self.v_min, self.group_px)

    def to_cm(self, xy):
        if not self.homography.ready:
            return None
        return self.homography.to_cm([xy])[0]

    # -- dock -------------------------------------------------------------

    def build_dock(self):
        self.sliders, self.buttons = [], []
        x = W - DOCK_W + PAD
        w = DOCK_W - 2 * PAD
        y = 52

        self.sliders.append(Slider((x, y, w, 18), "threshold", 40, 255,
                                   lambda: self.v_min, self.set_v_min))
        y += 26
        self.sliders.append(Slider((x, y, w, 18), "group px", 8, 160,
                                   lambda: self.group_px, self.set_group))
        y += 26
        self.sliders.append(Slider((x, y, w, 18), "exposure", -13, 0,
                                   lambda: self.exposure, self.set_exposure))
        y += 32
        bw = (w - 2 * 6) // 3
        self.buttons.append(Button((x, y, bw, 24), "lock",
                                   lambda: self.exposure_mode(True), tone=MINT))
        self.buttons.append(Button((x + bw + 6, y, bw, 24), "auto",
                                   lambda: self.exposure_mode(False)))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "manual",
                                   self.manual))
        y += 32
        self.buttons.append(Button((x, y, bw, 24), "save",
                                   self.save_frame, tone=CYAN))
        self.buttons.append(Button((x + bw + 6, y, bw, 24), "mask",
                                   self.toggle_view))
        self.buttons.append(Button((x + 2 * (bw + 6), y, bw, 24), "pause",
                                   self.toggle_pause))
        y += 40

        self.sliders.append(Slider((x, y, w, 18), "taillight", 0, 255,
                                   lambda: self.tail_bright, self.set_tail))
        y += 30
        for code in self.leds:
            self.buttons.append(Button((x, y, 58, 20), code,
                                       lambda c=code: self.cycle_color(c)))
            self.sliders.append(Slider(
                (x + 64, y + 1, w - 64, 18), "bright", 0, 255,
                lambda c=code: self.leds[c]["bright"],
                lambda v, c=code: self.set_bright(c, v)))
            y += 26
        self.led_bottom = y

    def set_v_min(self, v):
        self.v_min = int(v)

    def set_group(self, v):
        self.group_px = int(v)

    def toggle_view(self):
        self.view = "mask" if self.view == "raw" else "raw"

    def toggle_pause(self):
        self.paused = not self.paused

    # -- drawing ----------------------------------------------------------

    def text(self, s, x, y, col=CHALK, font=None):
        surf = (font or self.f).render(s, True, col)
        self.screen.blit(surf, (x, y))
        return y + (font or self.f).get_height() + 2

    def view_rect(self):
        return pygame.Rect(PAD, PAD, W - DOCK_W - 2 * PAD, H - 2 * PAD)

    def draw_view(self):
        r = self.view_rect()
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

        def px(p):
            return int(ox + p[0] * scale), int(oy + p[1] * scale)

        # Every dot that cleared the threshold, so a refusal can be SEEN and
        # not merely read. Clipped dots are ringed in coral: that is the state
        # the LED sliders exist to get you out of.
        for d in self.dots:
            pygame.draw.circle(self.screen, CORAL if d["clipped"] else GREY,
                               px(d["xy"]), 3, 1)

        for pose in self.poses:
            c = px(pose["centre"])
            tone = self.tone_for(pose["name"])
            radius = max(10, int(self.group_px * scale * 0.6))
            pygame.draw.circle(self.screen, tone, c, radius, 2)
            rad = math.radians(pose["deg"])
            tip = (c[0] + math.cos(rad) * radius * 2.0,
                   c[1] + math.sin(rad) * radius * 2.0)
            pygame.draw.line(self.screen, tone, c, tip, 2)
            for side in (150, -150):
                a = rad + math.radians(side)
                pygame.draw.line(self.screen, tone, tip,
                                 (tip[0] + math.cos(a) * 9,
                                  tip[1] + math.sin(a) * 9), 2)
            pygame.draw.circle(self.screen, (90, 140, 250), px(pose["tail"]), 4)
            cm = self.to_cm(pose["centre"])
            label = (f"{pose['name']}  {cm[0]:.1f}, {cm[1]:.1f} cm  "
                     f"{pose['deg']:.0f}°" if cm is not None else
                     f"{pose['name']}  {pose['centre'][0]:.0f}, "
                     f"{pose['centre'][1]:.0f} px  {pose['deg']:.0f}°")
            self.text(label, c[0] + radius + 8, c[1] - 8, tone, self.fs)

        # Refusals, drawn where they happened. A reason on screen next to the
        # dots that caused it is what turns "the tracker is flaky" into "that
        # one has two blues because its neighbour is inside the radius".
        for reason, group in self.why:
            if not group:                       # frame-wide, drawn below
                continue
            p = px(np.mean([d["xy"] for d in group], axis=0))
            pygame.draw.circle(self.screen, SUN, p, 14, 1)
            self.text(reason, p[0] + 18, p[1] - 7, SUN, self.fs)
        for reason, group in self.why:
            if not group:
                self.text(reason, r.x + 16, r.y + 16, SUN)

    def tone_for(self, name):
        c = config.COLORS.get(name, {}).get("draw", (200, 200, 200))
        return (c[2], c[1], c[0])       # the palette stores BGR; pygame wants RGB

    def draw_dock(self):
        x = W - DOCK_W + PAD
        w = DOCK_W - 2 * PAD
        pygame.draw.rect(self.screen, PANEL, (W - DOCK_W, 0, DOCK_W, H))
        pygame.draw.line(self.screen, RULE, (W - DOCK_W, 0), (W - DOCK_W, H))

        section(self.screen, self.fs, "detector", x, 24, w,
                f"{self.cam.fps:.0f} fps", DIM)
        for s in self.sliders:
            s.draw(self.screen, self.fs)
        for b in self.buttons:
            b.draw(self.screen, self.fs)

        y = self.led_bottom + 8
        if not self.fleet:
            y = self.text("no fleet — LED controls are inert", x, y, DIM, self.fs)
        y += 6

        y = section(self.screen, self.fs, "camera", x, y, w)
        verdict = self.exposure_verdict
        if verdict:
            msg, tone = verdict
            for line in wrap(msg, 44):
                y = self.text(line, x, y, tone, self.fs)
        else:
            y = self.text("move the exposure slider to test it", x, y, DIM,
                          self.fs)
        y = self.text(f"asked {self.exposure}   camera says {self.exp_took}",
                      x, y, DIM, self.fs)
        y = self.text("exposure LOCKED" if self.exposure_locked else
                      "exposure on auto — it will fight you", x, y,
                      MINT if self.exposure_locked else SUN, self.fs)
        y = self.text(f"mean brightness {self.mean_v:.1f}   "
                      f"lit {self.lit_px} px", x, y, DIM, self.fs)
        y += 8

        y = section(self.screen, self.fs, "pose", x, y, w,
                    f"{len(self.poses)} of {len(self.poses) + len(self.why)}",
                    DIM)
        for pose in self.poses:
            cm = self.to_cm(pose["centre"])
            tone = self.tone_for(pose["name"])
            r = pygame.Rect(x, y, w, 40)
            card(self.screen, r)
            self.text(pose["name"], x + 8, y + 5, tone, self.fs)
            where = (f"{cm[0]:7.1f} {cm[1]:7.1f} cm" if cm is not None
                     else f"{pose['centre'][0]:7.0f} {pose['centre'][1]:7.0f} px")
            self.text(where, x + 74, y + 5, CHALK, self.fs)
            self.text(f"{pose['deg']:5.1f}° img  {compass(pose['deg']):5.1f}° cmp",
                      x + 8, y + 21, DIM, self.fs)
            # Right-aligned, because the angles to its left are variable width
            # and a fixed column collides with them at three digits.
            flag = (f"m {pose['margin']:.2f}"
                    + ("  CLIPPED" if pose["clipped"] else ""))
            surf = self.fs.render(flag, True,
                                  CORAL if pose["clipped"] or pose["margin"] < 0.03
                                  else DIM)
            self.screen.blit(surf, (x + w - surf.get_width() - 8, y + 21))
            y += 44
        for reason, _ in self.why:
            y = self.text(f"refused: {reason}", x, y, SUN, self.fs)
        y += 8

        if self.note:
            for line in wrap(self.note, 44):
                y = self.text(line, x, y, self.note_tone, self.fs)

        self.text("s save   l lock   u auto   v mask   space pause   esc quit",
                  x, H - 22, GREY, self.fs)

    def draw(self):
        self.screen.fill(INK)
        self.draw_view()
        self.draw_dock()
        pygame.display.flip()

    # -- events -----------------------------------------------------------

    def key(self, e):
        if e.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if e.key == pygame.K_s:
            self.save_frame()
        elif e.key == pygame.K_l:
            self.exposure_mode(True)
        elif e.key == pygame.K_u:
            self.exposure_mode(False)
        elif e.key == pygame.K_v:
            self.toggle_view()
        elif e.key == pygame.K_SPACE:
            self.toggle_pause()
        elif e.key == pygame.K_LEFTBRACKET:
            self.set_exposure(max(-13, self.exposure - 1))
        elif e.key == pygame.K_RIGHTBRACKET:
            self.set_exposure(min(0, self.exposure + 1))
        return True

    def run(self):
        global W, H
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    W, H = max(MIN_W, e.w), max(MIN_H, e.h)
                    self.screen = pygame.display.set_mode((W, H),
                                                          pygame.RESIZABLE)
                    self.build_dock()
                elif e.type == pygame.KEYDOWN:
                    running = self.key(e)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for s in self.sliders:
                        if s.hit(e.pos):
                            break
                    else:
                        for b in self.buttons:
                            if b.hit(e.pos):
                                break
                elif e.type == pygame.MOUSEBUTTONUP:
                    for s in self.sliders:
                        s.dragging = False
                elif e.type == pygame.MOUSEMOTION:
                    for s in self.sliders:
                        if s.dragging:
                            s.drag(e.pos)
            self.tick()
            self.draw()
            self.clock.tick(60)
        self.close()

    def close(self):
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
    p.add_argument("--source", dest="camera",
                   help="same as --camera")
    p.add_argument("--size", default=None, help="e.g. 1280x720")
    p.add_argument("--exposure", type=int, default=-7)
    p.add_argument("--no-fleet", action="store_true",
                   help="skip the robots; tune the detector alone")
    a = p.parse_args(argv)
    size = None
    if a.size:
        size = tuple(int(v) for v in a.size.lower().split("x"))
    PoseTest(a.camera, size=size, exposure=a.exposure,
             with_fleet=not a.no_fleet).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
