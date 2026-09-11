"""A bench for one question: can the camera see which way a ball is pointing?

Everything else in this project infers heading by driving known legs and
watching where the ball went, because "a glowing sphere has no facing". That
premise is only true if you ignore the tail light. A Sphero carries a bright
main RGB LED and a dim blue aim LED at the back, and two lights on one shell
IS an orientation — readable every frame, with nothing driven and nothing to
go stale across a reconnect.

`vision/facing.py` does the reading. This is the lab you tune it in: one
camera, one ball, the arrow drawn where the maths thinks the robot points, and
a log of x/y/theta running underneath so you can watch it be right or wrong
while you change a light or a lens.

    ~/miniconda3/bin/python3 taillight.py --camera 0
    ~/miniconda3/bin/python3 taillight.py --source synthetic   # no hardware

MOVED THE CAMERA, OR CHANGED ITS RESOLUTION? Press `c` and click the four
arena corners — origin, +x, +x+y, +y. A homography calibrated at another
position or another frame size warps the wrong part of the picture and reports
centimetres that are fiction, and nothing in the view will tell you: the dots
still land on the ball. `w` writes it to `calib/homography.json`, which is live
state the tracker and the planner both read, so it only happens when you ask.

IDENTITY WITHOUT COLOUR. Press `i` and click the FRONT light of each ball in
turn. From then on the name and the facing are carried frame to frame by
association rather than re-read from a hue — see `vision/tracks.py` for why
that is a trade and not an upgrade, and for the LOST and CONTENDED states that
make the moment it breaks obvious. Once identity is not coming from the colour
there is no reason to spend two thirds of the LED getting a primary to the
camera: `white` drives every ball on all three dies, which is three times the
light and the bigger blobs that go with it.

DRIVING IT BY HAND. Left and right SWING THE AIM, up and down drive along it,
tab picks which ball. The yaw is deliberately gradual — a heading reader is
easy to fool with a ball that only ever sits at one of four angles, and what
has to be watched is whether it FOLLOWS: smoothly, through the quadrants,
without the front and back changing places. Down drives the opposite way,
which is the manoeuvre that exercises it hardest.

Commands go out RAW, with no heading offset applied, so `told` and `went` in
the readout are two independent numbers and the gap between them IS the ball's
aim offset — the figure `calib.py` drives known legs to measure.

Keys — arrows drive, tab switch ball, i assign identities by clicking, c pick
arena corners, w write them, backspace undo one, m mask, b brightness,
a all bots, f autofocus off and pin the lens, [ ] shutter one notch down/up,
space pause, x clear the log, q quit.

The `mask` button shows what the light finder is working from: every pixel
above the light floor, coloured by what becomes of it — mint for the cluster
the heading was read from, cyan for a light that is not in it, coral for a
region above the floor and thrown away on area. That last colour is the one to
watch while shortening the shutter, because floor creeping in arrives as ONE
enormous region rather than as many, and a plain white-on-black mask cannot
tell "lit" from "used". The bar underneath carries the percentage of the frame
that is lit, which turns red before any of the colours do.

The `brightness` button swaps the view for a heat map of where the light
actually falls, warped flat through the homography so it covers THE ARENA and
not the room — a bright window behind the floor is not a lighting problem, and
averaged into a whole-frame number it hides one that is. Clipped pixels are
drawn in flat red, because the top of a heat ramp reads as "bright" and the one
thing that must not be mistaken for bright is "no hue and no peaks left".

THE SHUTTER SLIDER IS THE FIRST THING TO MOVE, and it is a real one. Shorten
it until the floor goes black and the only things left in the frame are the two
tag LEDs and the tail — that is the picture every measurement below is easiest
in, because a lit ball stops being a hue problem and becomes three bright spots
on nothing. It goes out of band through `uvc-util`: macOS does not plumb UVC
exposure through AVFoundation for external cameras, so OpenCV accepts
`CAP_PROP_EXPOSURE` and drops it, and this slider used to be one of those. The
consequence to remember is that OpenCV never learns the exposure changed, so
the panel's own number is the only thing that knows. `exp save` writes it to
`calib/exposure.json`; `exp auto` hands the shutter back.

WHERE THE BALL IS, once you can see the three lights. The two tag LEDs straddle
the centre of the shell and the tail sits behind both, so the centre is the
MIDPOINT OF THE TAG PAIR — not the centroid of all three lights, which leans
about 7mm backwards along the heading. That offset rotates with the robot, so
no constant cancels it, and a controller reads it as a position that drifts
whichever way the ball happens to point. The readout names which of the two it
used, because with only one tag LED visible there is no midpoint to take and
the answer is biased forward instead.

Four things this is opinionated about, all learned elsewhere in this repo:

**It refuses out loud.** A reading that cannot be taken prints WHY — "one peak
— the lights are merging", "peaks 31px apart on a 22px ball". A bench that
says nothing when it fails leaves you turning knobs at random, and most of the
time lost in this project has been a measurement lying quietly rather than a
controller misbehaving.

**Defocus helps blobs and hurts headings, so they are separate knobs.** The
`unfocus` slider blurs only what the DETECTOR sees: a smeared ball is a rounder,
easier blob to find. The heading is read from the sharp frame, because blur is
exactly what merges two LEDs a few pixels apart into one. If you defocus the
LENS instead, you pay that cost for real — which is worth knowing, and is why
the focus slider is here next to it.

**One hue, not two.** The `hue` slider moves what the ball is TOLD to glow and
what the detector HUNTS at the same time. `vision/config.py` keeps those in one
table on purpose — two tables is how a robot ends up lit one colour and looked
for as another, each internally consistent and nothing complaining — and a lab
that split them would reintroduce that on the bench people use to diagnose it.

**Nothing here writes calibration.** No thresholds, no colours, no roster. It
reads `calib/` so the numbers mean the same thing they do on the bench, and
leaves it exactly as it found it. Tuning that turns out to be right goes back
into `calib.py`, deliberately, by a person. The ball is connected directly
rather than through a roster row for the same reason: `roster.json` is live
state, and a lab that edits it changes what the next real session connects to.
"""

import argparse
import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import pygame

from ui.theme import (CHALK, CORAL, CYAN, DIM, GREY, INK, MINT, RULE, SUN,
                      Button, Slider, card, section)
from vision import config, lighting, shutter
from vision.detect import Detector
from vision.facing import (ANNULUS, LIGHT_MIN_V, MIN_TAG_SAT,
                           SPAN_FOR_FULL_CONF, bearing_cm,
                           cluster_px, explain_cm, explain_px,
                           blob_axis_px, heading_from_lights, hue_err,
                           light_mask, lights_px, radial_profile)
from vision.homography import Homography
from vision.tracks import Tracks, wrap180
from vision.synthetic import open_source

W, H = 1480, 1030
"""The window. Taller than the view needs, because the control column is what
sets the height: camera, vision, a row per robot and a readout under them all.
Every row added to that column has to come from somewhere, and the thing it
silently took was the bottom of the panel — which is where `taillight` lives."""
VIEW = pygame.Rect(14, 48, 900, 506)          # 16:9, whatever the source is
LOG = pygame.Rect(14, 570, 620, 446)
# The radial profile sits beside the log rather than in the panel, because it
# is the instrument for choosing ANNULUS and a plot too small to read is not an
# instrument. It also belongs under the camera view, which is where a person
# tuning the exposure is already looking.
PROFILE = pygame.Rect(648, 570, 266, 446)
PANEL_X, PANEL_W = 930, 536

LOG_HZ = 5.0            # lines per second; faster than this is unreadable
LOG_KEEP = 600


class Grab(threading.Thread):
    """Camera reads on their own thread, latest frame wins.

    A 1080p read is tens of milliseconds and a dropped frame matters less here
    than a window that stops repainting — a bench that stutters gets blamed for
    the thing it is measuring.
    """

    def __init__(self, source):
        super().__init__(daemon=True, name="taillight-grab")
        self.source = source
        self.frame = None
        self.at = 0.0
        self.error = None
        self.count = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                ok, f = self.source.read()
            except Exception as e:
                self.error = str(e)
                time.sleep(0.2)
                continue
            if not ok or f is None:
                self.error = "camera returned no frame"
                time.sleep(0.05)
                continue
            self.error = None
            self.frame = f
            self.at = time.time()
            self.count += 1

    def stop(self):
        self._stop.set()


class Scan(threading.Thread):
    """Discover Spheros without freezing the window."""

    def __init__(self, timeout=7.0):
        super().__init__(daemon=True, name="taillight-scan")
        self.timeout = timeout
        self.names = []
        self.error = None
        self.finished = False

    def run(self):
        try:
            from spherov2 import scanner
            toys = scanner.find_toys(timeout=self.timeout)
            self.names = sorted({getattr(t, "name", None) or str(t) for t in toys})
            if not self.names:
                self.error = ("no Spheros answered — shake one awake, and check "
                              "it is not still paired to a phone")
        except Exception as e:
            self.error = str(e)
        finally:
            self.finished = True


class Lab:
    """The vision, with no pygame in it. `analyse` is the whole measurement."""

    def __init__(self, source="synthetic", size=(1920, 1080), color="red"):
        self.source_spec = str(source)
        self.source = open_source(self.source_spec, size=size)
        self.wanted_size = tuple(size)
        self.detector = Detector()
        self.hom = Homography.load()
        self.color = color
        # Blur lives on the detector, so this slider IS the detector's blur and
        # not a second one beside it. In memory only — see the module docstring.
        self.detector.thresh["blur"] = int(self.detector.thresh.get("blur", 5))
        self.floor = 55                 # percent of the peak, for facing
        # The peaks-first reader. `min_v` is what counts as a light at all and
        # `span_px` is how far apart two lights can be and still be one ball —
        # both want to be knobs, because the right values depend on the lens,
        # the mounting height and how bright the room is.
        self.mode = "lights"
        self.min_v = LIGHT_MIN_V
        self.span_px = 60
        self.robot = None
        # ble_name -> handle, and ble_name -> the colour slot THIS APP told it
        # to wear. Two dicts rather than one object because the assignment is
        # the more important of the two: it is what turns "a red cluster at
        # (46, 67)" into "SK-5640 at (46, 67)", and it is knowledge this app
        # has first-hand — it issued the command — rather than belief read out
        # of a file that something else may have edited.
        self.robots = {}
        self.assigned = {}
        # Colour slot -> the robot wearing it, so a cluster can be labelled
        # with the CODE a person recognises rather than a colour they have to
        # translate. READ ONLY, and the lab never writes `roster.json` — that
        # file is live state and editing it changes what the next real session
        # connects to. A missing or broken roster costs the labels and nothing
        # else, so it is caught rather than raised.
        # Identity handed over by a person instead of read off a hue. See
        # `vision/tracks.py` for why that is a trade and not an upgrade.
        self.tracks = Tracks()
        self.tag_mode = "colour"        # or "manual"
        self.wearer = {}
        try:
            from fleet.roster import Roster
            self.wearer = {e.color: e.code
                           for e in Roster.load().enabled_entries()}
        except Exception:
            pass

    # -- the measurement -------------------------------------------------

    @property
    def unfocus(self):
        return int(self.detector.thresh["blur"])

    @unfocus.setter
    def unfocus(self, v):
        self.detector.thresh["blur"] = max(1, int(v)) | 1

    def analyse(self, frame):
        """One frame in, one reading out. Never raises, always says why.

        Two routes to the same answer, and `mode` picks which. "lights" reads
        the bright spots directly and never asks the colour tracker anything;
        "blob" finds a hue blob first and reads the heading inside it. The
        first is the better instrument on a dark floor — a lit ball is the
        brightest thing in the room by a wide margin, while its hue has to
        survive a blown-out core — and it is the default for that reason.
        """
        if self.mode == "lights":
            return self.analyse_lights(frame)
        return self.analyse_blob(frame)

    def analyse_lights(self, frame):
        """Peaks first, colour second — colour only says which end is front."""
        out = self._blank()
        if frame is None:
            out["why"] = "no frame yet"
            return out

        lights = lights_px(frame, min_v=self.min_v)
        out["lights"] = lights
        if not lights:
            out["why"] = (f"nothing brighter than V={self.min_v} — lower the "
                          "light floor, or the LEDs are off")
            return out

        groups = cluster_px(lights, self.span_px)
        # The biggest cluster, not the brightest single light: a lone
        # reflection outshining a two-light ball is exactly the failure this
        # ordering avoids.
        groups.sort(key=lambda g: (-len(g), -max(l["peak"] for l in g)))
        group = groups[0]
        out["group"] = group

        # The detector's own signatures, so a name here means the same thing
        # it means on the bench. Passing them is what turns a heading into a
        # heading with a robot's name on it.
        got, why = heading_from_lights(group, signatures=self.detector.colors)
        if got is None:
            out["why"] = why
            if len(group) == 1:
                out["centre_px"] = (group[0]["x"], group[0]["y"])
            return out

        out.update(theta_img=got["deg"], conf=got["conf"],
                   centre_px=got["centre"], centre_from=got["centre_from"],
                   conf_span=got["conf_span"], conf_ends=got["conf_ends"],
                   certainty=got["certainty"], front=got["front"],
                   back=got["back"], by_colour=got["by_colour"],
                   span_px=got["span_px"], straightness=got["straightness"],
                   radius_px=max(4.0, got["span_px"] / 2.0),
                   color=got.get("color"), tag_conf=got.get("tag_conf"),
                   tag_why=got.get("tag_why"), by_tag=got.get("by_tag"))

        if self.hom is not None and self.hom.ready:
            pts = self.hom.to_cm([list(got["centre"])])
            out["xy_cm"] = tuple(float(v) for v in np.asarray(pts).ravel()[:2])
            out["theta_cm"] = bearing_cm((got["back"]["x"], got["back"]["y"]),
                                         (got["front"]["x"], got["front"]["y"]),
                                         self.hom)
        return out

    def analyse_all(self, frame):
        """Every cluster in the frame, with what vision believes each one is.

        `analyse_lights` answers about ONE ball — the biggest cluster — which
        is the right answer for a readout that has one row per field. This is
        the other question: with several balls lit at once, WHICH IS WHICH, and
        the honest answer per cluster includes the ones it cannot name.

        A cluster that fails to read is still returned, carrying its `why`.
        Dropping it would make a robot the reader is refusing to identify look
        exactly like a robot that is not there, and those need different fixes.
        """
        if frame is None:
            return []
        lights = lights_px(frame, min_v=self.min_v)
        out = []
        for group in cluster_px(lights, self.span_px):
            got, why = heading_from_lights(group,
                                           signatures=self.detector.colors)
            if got is None:
                centre = ((group[0]["x"], group[0]["y"])
                          if len(group) == 1 else None)
                row = {"group": group, "why": why, "centre": centre,
                       "deg": None, "conf": None, "color": None,
                       "code": None, "xy_cm": None, "axis": None}
                # A cluster the peak reader cannot use is not a cluster with
                # nothing in it. When the lights have bloomed together there
                # is still a streak, and a streak has an axis — which is all a
                # track needs, because it already knows which way round it is.
                if len(group) == 1:
                    axis = blob_axis_px(frame, group[0])
                    if axis is not None:
                        row["axis"], row["elongation"] = axis[0], axis[1]
                        row["why"] = (f"lights merged (x{axis[1]:.1f} long) — "
                                      "axis from the blob's shape, direction "
                                      "from the track")
                out.append(row)
                continue
            xy = None
            if self.hom is not None and self.hom.ready:
                pts = self.hom.to_cm([list(got["centre"])])
                xy = tuple(float(v) for v in np.asarray(pts).ravel()[:2])
            # The colour this app ASSIGNED wins over the roster's idea of
            # who wears what. This app issued the `set_led` — that is
            # first-hand knowledge, while `roster.json` is a file something
            # else may have edited since.
            colour = got.get("color")
            out.append({"group": group, "why": None, "centre": got["centre"],
                        "deg": got["deg"], "conf": got["conf"],
                        "front": got["front"], "back": got["back"],
                        "color": colour,
                        "conf_span": got["conf_span"],
                        "conf_ends": got["conf_ends"],
                        "span_px": got["span_px"],
                        "code": (self.wearing(colour)
                                 or self.wearer.get(colour)),
                        "centre_from": got["centre_from"],
                        "tag_why": got.get("tag_why"), "xy_cm": xy})
        # Biggest first, so the label of the ball you are looking at is drawn
        # over a neighbour's rather than under it.
        out.sort(key=lambda r: -len(r["group"]))
        return out

    def _blank(self):
        return {"color": self.color, "centre_px": None, "radius_px": None,
                "area": None, "theta_img": None, "theta_cm": None,
                "conf": None, "xy_cm": None, "why": None, "lights": [],
                "group": [], "front": None, "back": None, "by_colour": None,
                "centre_from": None, "conf_span": None, "conf_ends": None,
                "certainty": None,
                "span_px": None, "straightness": None, "tag_conf": None,
                "tag_why": None, "by_tag": None}

    def analyse_blob(self, frame):
        """The original route: find a hue blob, read the heading inside it.

        The blob comes from the BLURRED frame — `Detector` applies its own
        Gaussian and the `unfocus` slider drives it — and the heading comes
        from the frame as delivered. Two lights a few pixels apart do not
        survive the blur that makes a shell easy to find, so reading both off
        one image would mean choosing which of the two measurements to spoil.
        """
        out = self._blank()
        if frame is None:
            out["why"] = "no frame yet"
            return out

        cands = self.detector.candidates(frame, only=[self.color])
        found = cands.get(self.color)
        if not found:
            lim = self.detector.limits(self.color)
            out["why"] = (f"no {self.color} blob — hue "
                          f"{self.detector.colors[self.color]['hue']}"
                          f"±{self.detector.colors[self.color]['tol']}, "
                          f"s>{lim['s_min']} v>{lim['v_min']}, "
                          f"area {lim['min_area']}-{lim['max_area']}")
            return out

        x, y, area = found[0]
        radius = math.sqrt(max(area, 1.0) / math.pi)
        out.update(centre_px=(x, y), radius_px=radius, area=area)

        got, why = explain_px(frame, (x, y), radius, floor=self.floor / 100.0)
        if got is None:
            out["why"] = why
        else:
            out["theta_img"], out["conf"] = got

        if self.hom is not None and self.hom.ready:
            pts = self.hom.to_cm([[x, y]])
            out["xy_cm"] = tuple(float(v) for v in np.asarray(pts).ravel()[:2])
            arena, _ = explain_cm(frame, (x, y), radius, self.hom,
                                  floor=self.floor / 100.0)
            if arena is not None:
                out["theta_cm"] = arena[0]
        return out

    # -- lighting --------------------------------------------------------

    def arena_view(self, frame, out_w, out_h):
        """The frame warped flat to the arena. See `vision/lighting.py`."""
        return lighting.arena_view(frame, self.hom, out_w, out_h)

    def mask_view(self, frame, group=None):
        """What the light finder is actually working from, as a picture.

        Not a plain black-and-white threshold. The question this answers while
        you are pulling the shutter down is never "is anything lit" — it is
        "is the thing I can see being USED", and a white-on-black mask cannot
        tell those apart. A region that is above the threshold but outside the
        area gate looks identical to one that was accepted, so the floor
        creeping in reads as success right up until nothing tracks.

        So each region is coloured by what happens to it:

            mint    in the cluster the heading was read from
            cyan    a light, but not part of that cluster
            coral   above the threshold and thrown away on area — too big is
                    the floor joining in, too small is sensor noise

        Returns `(image, stats)`. `stats` is what the readout needs to say how
        much of the frame is lit, which is the number that tells you a shutter
        is still too long before any of the colours do.
        """
        grey, mask = light_mask(frame, self.min_v)
        n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)

        img = np.zeros((*grey.shape, 3), np.uint8)
        # The rejected floor first, so an accepted light drawn over it wins.
        # Painting by lookup rather than per-component: at 1080p a Python loop
        # over regions costs a full-resolution pass each, and the floor
        # creeping in is exactly when there are hundreds of them.
        tone = np.zeros((max(n, 1), 3), np.uint8)
        kept = {id(l) for l in (group or [])}
        chosen = np.zeros(max(n, 1), bool)
        for l in (group or []):
            lx, ly = int(round(l["x"])), int(round(l["y"]))
            if 0 <= ly < labels.shape[0] and 0 <= lx < labels.shape[1]:
                chosen[labels[ly, lx]] = True
        passed = 0
        for i in range(1, n):
            area = float(stats_cc[i, 4])
            if area < 3 or area > 6000:          # the gate `lights_px` uses
                tone[i] = (70, 70, 210)          # coral, in BGR
                continue
            passed += 1
            tone[i] = (150, 220, 130) if chosen[i] else (210, 190, 80)
        img[mask] = tone[labels[mask]]

        lit = float(mask.mean() * 100.0)
        return img, {"lit_pct": lit, "regions": n - 1, "passed": passed,
                     "max": int(grey.max())}

    def brightness_map(self, frame, out_w, out_h):
        """Where the light falls, over the arena. See `vision/lighting.py`."""
        return lighting.brightness_map(frame, self.hom, out_w, out_h)

    # -- the ball --------------------------------------------------------

    def free_color(self):
        """The slot furthest from every slot already in use.

        Not "the next one in the list". Identity here is a hue match with a
        tolerance and an ambiguity gap, so two balls wearing neighbouring
        slots are two balls that trade names when the light changes. Picking
        the slot with the largest minimum separation spends the palette in the
        order that keeps the fleet furthest apart for longest — with two balls
        it reaches for opposite sides of the wheel rather than for the first
        two rows of a table.
        """
        used = [self.detector.colors[c]["hue"] for c in self.assigned.values()
                if c in self.detector.colors]
        free = [c for c in self.detector.colors if c not in
                set(self.assigned.values())]
        if not free:
            return None
        if not used:
            return free[0]

        def gap(name):
            h = self.detector.colors[name]["hue"]
            return min(hue_err(h, u) for u in used)
        return max(free, key=gap)

    def connect(self, ble_name, color=None):
        """One more ball, for its lights only. No tracker, no workspace.

        ADDS rather than replaces: the whole point of tagging by colour is that
        several are lit at once, and a bench that can only hold one link can
        never show you the case where two get confused for each other.

        No roster row is written. `roster.json` is live state the bench and the
        app both read, and a lab that edits it changes what the next real
        session connects to — so the handle is constructed directly and the
        colour assignment lives in memory, where it belongs.
        """
        from fleet.real_handle import SpheroRobot
        if ble_name in self.robots:
            return self.robots[ble_name]
        color = color or self.free_color()
        if color is None:
            raise RuntimeError(
                f"every colour slot is taken by {len(self.assigned)} balls — "
                "release one, or add a slot to calib/colors.json")
        r = SpheroRobot(name=ble_name, code=ble_name, color=color,
                        ble_name=ble_name, tracker=None)
        self.robots[ble_name] = r
        self.assigned[ble_name] = color
        self.robot = r                  # the most recent, for the single-ball
        return r                        # sliders that still speak to just one

    def retry(self, ble_name):
        """Throw the handle away and make a fresh one, same colour.

        The handle retries on its own, but it backs off to thirty seconds
        between attempts — so a ball that failed once looks dead for half a
        minute, which is indistinguishable from a ball that is not coming
        back. Rebuilding it starts from a clean radio and a zero backoff, and
        it also clears a link that came up wedged.
        """
        color = self.assigned.get(ble_name)
        self.disconnect(ble_name)
        return self.connect(ble_name, color=color)

    def track(self, clusters, dt):
        """Carry the assigned identities onto this frame's clusters."""
        return self.tracks.update(clusters, dt)

    def px_per_cm(self, at_px):
        """Source pixels per centimetre, AT this point in the frame.

        Not a constant. A perspective map is only a scale factor at one point:
        on a tilted camera the far side of the arena has fewer pixels per
        centimetre than the near side, and quoting one number for the whole
        floor hides the corner where the reader will fail first.

        This is the number the whole rig turns on. Blob separation, the span
        term in the confidence, whether the lights resolve at all — every one
        of them is a distance on the ball multiplied by this.
        """
        if self.hom is None or not self.hom.ready or at_px is None:
            return None
        a = (float(at_px[0]), float(at_px[1]))
        b = (a[0] + 10.0, a[1])
        c = (a[0], a[1] + 10.0)
        try:
            pa, pb, pc = self.hom.to_cm([list(a), list(b), list(c)])
        except Exception:
            return None
        pa, pb, pc = (np.asarray(v, float)[:2] for v in (pa, pb, pc))
        dx = float(np.linalg.norm(pb - pa))
        dy = float(np.linalg.norm(pc - pa))
        if dx < 1e-9 or dy < 1e-9:
            return None
        # The geometric mean of the two directions: a perspective map squashes
        # one axis more than the other, and the reader cares about separation
        # along whichever way the ball happens to be pointing.
        return float(10.0 / math.sqrt(dx * dy))

    def frame_covers_cm(self, frame_w):
        """How much floor the frame spans, at the scale in its middle."""
        got = self.px_per_cm((frame_w / 2.0, frame_w * 0.28))
        return (frame_w / got) if got else None

    def cluster_at(self, px, clusters, within=140.0):
        """The cluster nearest a click, or None if the click missed.

        A radius rather than "always the nearest": clicking empty floor should
        assign nothing, and silently grabbing the far side of the arena is how
        a person ends up believing they told it something they did not.
        """
        best, best_d = None, within
        for c in clusters or []:
            centre = c.get("centre")
            if centre is None:
                continue
            d = math.dist(px, centre)
            if d <= best_d:
                best, best_d = c, d
        return best

    def link_trouble(self):
        """Why the fleet is not all up, in one line, or None.

        The handle logs its failures and the log is not where somebody
        watching a bench is looking. A controller hitting the radio's
        connection ceiling is the single most likely reason a second ball will
        not join, and it is not a retryable failure — so it is named rather
        than left to look like bad luck.
        """
        from fleet.real_handle import MAX_CONNECTIONS_HINT
        for ble, r in self.robots.items():
            if getattr(r, "max_connections_hit", False):
                return MAX_CONNECTIONS_HINT
        errs = [(ble, getattr(r, "last_error", None))
                for ble, r in self.robots.items()
                if not r.link_up and getattr(r, "last_error", None)]
        if errs:
            ble, err = errs[0]
            return f"{ble}: {err}"
        return None

    def wearing(self, color):
        """Which ball this app told to glow that colour, if any."""
        for ble, c in self.assigned.items():
            if c == color:
                return ble
        return None

    def disconnect(self, ble_name=None):
        """One ball, or all of them when `ble_name` is None."""
        names = [ble_name] if ble_name else list(self.robots)
        for n in names:
            r = self.robots.pop(n, None)
            self.assigned.pop(n, None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
        self.robot = next(iter(self.robots.values()), None)

    def close(self):
        self.disconnect()
        try:
            self.source.release()
        except Exception:
            pass


class App:
    """The window around `Lab`."""

    COLORS = list(config.COLORS)

    def __init__(self, source="synthetic", camera_size=(1920, 1080),
                 color="red", uvc_index=0):
        pygame.init()
        pygame.display.set_caption("taillight — heading from two LEDs")
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 16, bold=True)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)

        self.lab = Lab(source=source, size=camera_size, color=color)
        self.grab = Grab(self.lab.source)
        self.grab.start()

        self.reading = self.lab.analyse(None)
        self.lines = deque(maxlen=LOG_KEEP)
        self.last_log = 0.0
        self.paused = False
        self.notes = deque(maxlen=6)

        # LED state. Held here rather than read back off the robot, because a
        # Sphero cannot be asked what colour it is currently glowing.
        # From the DETECTOR's signature, not from `config.COLORS`. The live
        # calibration overrides the defaults — red is at hue 138 in this room
        # and 0 in the table — so seeding the slider from the table would put
        # the two out of step before anything had been touched.
        self.hue = self.lab.detector.colors[color]["hue"]
        self.bright = 255
        self.tail = 60

        self.scan = None
        self.found = []
        self.swatches = []
        self.link_rows = []
        self.hidden_rows = 0
        self._said_link = None
        # Four clicked corners, while picking. `_shot` is where the last frame
        # was blitted, which is the only way back from a screen click to a
        # frame pixel.
        self.picking = None
        self._shot = None
        self.list_scroll = 0
        self.list_total = 0
        self.list_room = 1
        self.list_rect = pygame.Rect(0, 0, 0, 0)
        # Manual drive. `driving` is which ball the arrows move; `held` is the
        # bearing currently commanded, or None when nothing is pressed.
        self.driving = None
        self.held = None                # bearing being commanded, or None
        self.aim = 0.0                  # the bearing the arrows steer
        self.drive_byte = 90
        self._drive_sent = None
        self._drive_at = 0.0
        self._tag_before = "colour"
        # Aim offsets read off completed drives: (bearing told, course made,
        # offset). Kept per ball, because the offset is a property of the
        # ball and averaging two balls together describes neither.
        self.aim_samples = {}
        self._last_aim = None
        # Who we are waiting to be told about, while assigning.
        self.assigning = None
        self.last_tick = time.time()
        # The shutter, out of band through uvc-util. macOS does not plumb
        # UVC exposure through AVFoundation for external cameras, so OpenCV
        # accepts `CAP_PROP_EXPOSURE` and drops it — which is what the slider
        # here used to drive, and why it moved without the picture moving.
        self.dial = shutter.Dial(uvc_index)
        # Focus goes out of band for the same reason: OpenCV's autofocus
        # property is accepted and dropped, which is why `af off` did nothing
        # while the lens kept hunting. On a dark floor that hunt is what merges
        # the two LEDs into one blob.
        self.focus_dial = shutter.FocusDial(uvc_index)
        self.view = "camera"            # or "mask" / "bright"
        self.show_all = False           # label every cluster, not just one
        self.all_reading = []
        self.light = None               # last brightness stats
        self.fps_at, self.fps_count, self.fps = time.time(), 0, 0.0
        self._focus_touched = False

        self.buttons, self.sliders = [], []
        self._build()
        self.say(f"source {self.lab.source_spec}, asked for "
                 f"{camera_size[0]}x{camera_size[1]}")
        got = self.camera_size()
        if got:
            self.say(f"camera is delivering {got[0]}x{got[1]}"
                     + ("" if got == tuple(camera_size) else "  — NOT what was asked"))
        if not (self.lab.hom is not None and self.lab.hom.ready):
            self.say("no arena calibration — x/y stay in pixels", CORAL)
        if self.lab.source_spec == "synthetic":
            # Otherwise the first run on no hardware looks broken. The
            # synthetic floor renders config.COLORS, and calib/colors.json is
            # tuned for a real room — red is at hue 138 there, not 0.
            self.say("synthetic frames use the DEFAULT hues while calib/ is "
                     "tuned for your room, so expect no blob here until the "
                     "hue slider matches", SUN)

    # -- helpers ---------------------------------------------------------

    def say(self, text, tone=None):
        self.notes.append((text, tone or DIM))

    def camera_size(self):
        get = getattr(self.lab.source, "size", None)
        if get is None:
            return None
        try:
            return tuple(get)
        except Exception:
            return None

    def cam_set(self, name, value):
        """Push a camera control and report what actually took."""
        setter = getattr(self.lab.source, "set", None)
        if setter is None:
            self.say(f"this source has no {name} control", SUN)
            return
        got = setter(name, value)
        if got is None:
            self.say(f"{name}: the camera ignored it", SUN)
        else:
            self.say(f"{name} -> {got:.0f}")

    def manual(self):
        fn = getattr(self.lab.source, "manual", None)
        if fn is None:
            self.say("this source has no automatics to turn off", SUN)
            return
        got = fn()
        self.say("manual: " + ", ".join(
            f"{k}={'?' if v is None else round(v, 2)}" for k, v in got.items()))
        self.say("autofocus and auto-exposure off — a dark floor makes both "
                 "of them fight you")

    # -- robot -----------------------------------------------------------

    def start_scan(self):
        if self.scan is not None and not self.scan.finished:
            return
        self.scan = Scan()
        self.scan.start()
        self.say("scanning for Spheros…")

    def drain_scan(self):
        if self.scan is None or not self.scan.finished:
            return
        if self.scan.error:
            self.say(self.scan.error, CORAL)
        self.found = list(self.scan.names)
        if self.found:
            self.say(f"found {', '.join(self.found)} — click one to connect")
        self.scan = None
        self._build()

    def connect(self, ble_name):
        """Add a ball and give it a colour nothing else is wearing."""
        try:
            r = self.lab.connect(ble_name)
        except Exception as e:
            self.say(f"connect failed: {e}", CORAL)
            return
        slot = self.lab.assigned[ble_name]
        self.say(f"{ble_name} joining as {slot.upper()} — lights follow once "
                 f"the link is up ({len(self.lab.robots)} held)")
        self.push_led(ble_name)
        self.push_tail(ble_name)
        self._build()

    def retry(self, ble_name):
        """Rebuild one link, keeping the colour."""
        try:
            self.lab.retry(ble_name)
        except Exception as e:
            self.say(f"retry failed: {e}", CORAL)
            self._build()
            return
        self.say(f"retrying {ble_name} as "
                 f"{self.lab.assigned.get(ble_name, '?').upper()}")
        self.push_led(ble_name)
        self.push_tail(ble_name)
        self._build()

    def release(self, ble_name=None):
        self.lab.disconnect(ble_name)
        self.say(f"released {ble_name}" if ble_name else "released every ball")
        self._build()

    def cycle_color(self, ble_name):
        """Move one ball to the next unused slot.

        Two balls that read as each other is the failure this bench exists to
        make visible, and the fix is always the same: put more hue between
        them. Doing it per ball beats re-scanning the fleet.
        """
        want = self.lab.free_color()
        if want is None:
            self.say("no free colour slot to move to", SUN)
            return
        self.lab.assigned[ble_name] = want
        r = self.lab.robots.get(ble_name)
        if r is not None:
            r.color = want
        self.say(f"{ble_name} is now {want.upper()}")
        self.push_led(ble_name)
        self._build()

    def _each(self, ble_name=None):
        if ble_name is not None:
            r = self.lab.robots.get(ble_name)
            return [(ble_name, r)] if r is not None else []
        return list(self.lab.robots.items())

    def push_led(self, ble_name=None):
        """Every ball glows the hue its OWN slot is calibrated at.

        Not the `hue` slider: that is the single-ball tuning knob and driving
        a fleet from it would light them all the same colour, which is the one
        arrangement in which colour cannot tag anything.
        """
        for ble, r in self._each(ble_name):
            slot = self.lab.assigned.get(ble)
            r.set_led(self.drive_for(slot))

    def drive_for(self, slot):
        """The RGB to send for a slot, dimmed by `bright`.

        Through `config.led_for`, so an explicit `rgb` on the signature wins
        over the hue — the same rule the rest of the fleet already follows.
        `bright` scales the triple linearly rather than going back through
        HSV: once somebody has set the channels by hand, a round trip through
        a colour space is exactly the step that would move them.
        """
        if self.lab.tag_mode == "manual":
            # White: all three dies, and nothing downstream is reading the
            # colour any more.
            k = max(0, min(255, int(self.bright)))
            return (k, k, k)
        if slot is None:
            return config.led_rgb(int(self.hue), value=self.bright)
        rgb = config.led_for(slot, self.lab.detector.colors)
        k = max(0, min(255, int(self.bright))) / 255.0
        return tuple(int(round(v * k)) for v in rgb)

    def push_tail(self, ble_name=None):
        for _, r in self._each(ble_name):
            r.set_back_led(int(self.tail))

    def toggle_mode(self):
        self.lab.mode = "blob" if self.lab.mode == "lights" else "lights"
        self.say("reading the LIGHTS directly — the colour tracker is not "
                 "consulted at all, it only says which end is the front"
                 if self.lab.mode == "lights" else
                 "reading inside a HUE BLOB — the heading is now only as good "
                 "as the colour tracking")
        self._build()

    # -- driving it by hand ------------------------------------------------

    # LEFT and RIGHT swing the aim; UP and DOWN drive along it. A Sphero has
    # one command — `roll(heading, speed)` — and a roll at speed zero turns the
    # ball on the spot, so "yaw" and "drive" here are that same command with
    # the speed byte set differently.
    #
    # Yawing rather than jumping between four fixed bearings is the point: a
    # heading reader is easy to fool with a ball that only ever sits at one of
    # four angles, and what has to be watched is whether it FOLLOWS — smoothly,
    # through the quadrants, without the front and back changing places.
    YAW_DEG_PER_S = 80.0
    DRIVE_PUSH_S = 0.1          # radio writes a second, at most
    DRIVE_TURN_DEADBAND = 4.0

    # Bearings are what the ball takes: zero is its own idea of forward and it
    # increases clockwise. Sent RAW, with no heading offset applied — the whole
    # point of driving by hand is to see what the ball's own zero actually
    # does, and a correction on the way out would hide exactly that.

    def drive_target(self):
        """Which ball the arrows move. The first held, unless one was picked."""
        if self.driving in self.lab.robots:
            return self.driving
        return next(iter(self.lab.robots), None)

    def cycle_drive(self):
        names = list(self.lab.robots)
        if not names:
            self.say("no ball to drive — scan and connect one", SUN)
            return
        at = names.index(self.driving) if self.driving in names else -1
        self.driving = names[(at + 1) % len(names)]
        self.say(f"arrows now drive {self.driving}")
        self._build()

    def drive_tick(self, dt):
        """Held arrows, once per frame. Left/right yaw, up/down drive.

        Read from the keyboard STATE rather than from key events, because a
        gradual turn is a thing that happens while a key is down and an event
        only says when it went down.
        """
        name = self.drive_target()
        if name is None:
            self.held = None
            return
        keys = pygame.key.get_pressed()
        turn = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        go = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        if turn:
            self.aim = (self.aim + turn * self.YAW_DEG_PER_S * dt) % 360.0
        if not turn and not go:
            if self.held is not None:
                # The drive just ended. If it went far enough to have a course,
                # that course is one independent measurement of the offset.
                if self._last_aim is not None:
                    self.aim_samples.setdefault(name, []).append(self._last_aim)
                    self._last_aim = None
                self.held = None
                self._drive_sent = None
                try:
                    self.lab.robots[name].stop()
                except Exception:
                    pass
            return

        # Down drives the opposite way: the ball turns round and goes, which is
        # the manoeuvre that exercises a heading reader hardest.
        bearing = self.aim if go >= 0 else (self.aim + 180.0) % 360.0
        # Only a DRIVE measures the offset. A yaw in place has no course.
        if go:
            got = self.aim_error()
            if got is not None:
                self._last_aim = got
        byte = int(self.drive_byte) if go else 0
        self.held = bearing
        self.driving = name

        now = time.perf_counter()
        last = self._drive_sent
        same = (last is not None
                and abs(wrap180(bearing - last[0])) < self.DRIVE_TURN_DEADBAND
                and last[1] == byte)
        if same or now - self._drive_at < self.DRIVE_PUSH_S:
            return
        self._drive_sent, self._drive_at = (bearing, byte), now
        try:
            self.lab.robots[name].drive_raw(bearing, byte)
        except Exception as e:
            self.say(f"{name}: {e}", CORAL)

    def drive_release(self):
        """Everything let go: stop, and forget what was last sent."""
        name = self.drive_target()
        self.held = None
        self._drive_sent = None
        if name is None:
            return
        try:
            self.lab.robots[name].stop()
        except Exception:
            pass

    def aim_error(self):
        """Commanded bearing versus the course actually made good.

        The gap between them IS the ball's heading offset — the number
        `calib.py` tries to measure by driving known legs, and the one whose
        absence is why nothing on this rig has a motion model. Driving by hand
        and reading it off the screen is the same measurement with a person in
        the loop.
        """
        name = self.drive_target()
        if name is None or self.held is None:
            return None
        t = self.lab.tracks.by_name.get(name)
        if t is None:
            return None
        if len(t._trail) < 3:
            return None
        a, b = t._trail[0][1], t._trail[-1][1]
        if math.dist(a, b) < 25.0:
            return None
        # THROUGH THE HOMOGRAPHY, as two points. The ball speaks a compass —
        # zero is +y in the arena, increasing clockwise — and the trail is in
        # image pixels, which is a different frame and, on a tilted camera, not
        # even a rotation of it. Comparing the two directly would report an
        # offset that is part perspective.
        made = bearing_cm(a, b, self.lab.hom)
        if made is None:
            return None
        return self.held, made, wrap180(self.held - made)

    # -- the aim offset, measured and saved ------------------------------

    # What a saved offset has to clear. Three drives is the least that can
    # disagree with itself; two DIRECTIONS is what makes that disagreement
    # mean something — a scale or perspective error changes the reading with
    # the bearing, while a true aim offset does not, and three drives the same
    # way cannot tell those apart.
    AIM_MIN_SAMPLES = 3
    AIM_MIN_SPREAD_DIRS = 60.0     # degrees between the widest two bearings
    AIM_MAX_SCATTER = 12.0         # degrees, circular std of the offsets

    def aim_summary(self, name=None):
        """The offset this ball's drives agree on, and whether they agree.

        Returns a dict, or None with no samples. `ok` is False with a `why`
        whenever saving would be a mistake — and the reasons are the ones that
        matter, because an offset that varies with direction is not an aim
        offset at all: it is a scale error, or a homography that is off, or a
        ball that slips, wearing an aim offset's clothes.
        """
        from fleet.heading import circular_mean

        name = name or self.drive_target()
        got = self.aim_samples.get(name) or []
        if not got:
            return None
        offs = [e for _, _, e in got]
        mean = circular_mean(offs)
        # Circular spread: how far the offsets scatter about their own mean.
        sx = sum(math.cos(math.radians(o)) for o in offs) / len(offs)
        sy = sum(math.sin(math.radians(o)) for o in offs) / len(offs)
        r = max(min(math.hypot(sx, sy), 1.0), 1e-9)
        scatter = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(r))))
        told = [t for t, _, _ in got]
        widest = max((abs(wrap180(a - b)) for a in told for b in told),
                     default=0.0)

        why = None
        if len(got) < self.AIM_MIN_SAMPLES:
            why = f"{len(got)} of {self.AIM_MIN_SAMPLES} drives — drive more"
        elif widest < self.AIM_MIN_SPREAD_DIRS:
            why = ("every drive went the same way — yaw round and drive a "
                   "different direction, or a scale error cannot be told "
                   "from a true offset")
        elif scatter > self.AIM_MAX_SCATTER:
            why = (f"the drives disagree by {scatter:.0f}deg — that is not a "
                   "constant offset. Check the arena calibration, or slip")
        return {"name": name, "mean": mean, "scatter": scatter, "n": len(got),
                "widest": widest, "ok": why is None, "why": why}

    def save_aim(self):
        """Write the agreed offset to this ball's roster row. Never automatic.

        `roster.json` is live state the fleet reads on its next connect, so it
        is written when somebody asks and only if the drives agreed — a wrong
        offset here is the exact failure that stopped the calibration battery,
        every stage of it, with "drove 12cm AWAY from the middle".

        The offset goes in as measured, with no sign change: the readout came
        from `drive_raw`, which applies none, and `set_velocity` ADDS
        `heading_offset` to what it sends. Worked through: a ball whose forward
        sits 30deg clockwise of +y reads told 0 went 30, an offset of -30, and
        to travel 90 it must be sent 60 = 90 + (-30).
        """
        from fleet.roster import Roster

        got = self.aim_summary()
        if got is None:
            self.say("no drives measured yet — hold up and let it travel", SUN)
            return
        if not got["ok"]:
            self.say(f"not saving: {got['why']}", SUN)
            return
        roster = Roster.load()
        entry = next((e for e in roster.entries
                      if e.ble_name == got["name"]), None)
        if entry is None:
            self.say(f"{got['name']} is not in roster.json, so there is no "
                     "row to put the offset on", CORAL)
            return
        was = entry.heading_offset
        entry.heading_offset = float(got["mean"]) % 360.0
        errors = roster.save()
        if errors:
            self.say(f"roster refused the write: {errors[0]}", CORAL)
            return
        self.say(f"{entry.code} heading_offset {was:.1f} -> "
                 f"{entry.heading_offset:.1f}  (from {got['n']} drives, "
                 f"scatter {got['scatter']:.1f}deg)", MINT)

    def forget_aim(self):
        self.aim_samples.pop(self.drive_target(), None)
        self.say("aim samples cleared for this ball")

    # -- identity by hand -------------------------------------------------

    def assignable(self):
        """Who there is to assign. Connected balls first — those are the ones
        this app can actually drive — and the roster otherwise, so the bench
        is usable with no radio at all."""
        if self.lab.robots:
            return list(self.lab.robots)
        return [c for c in self.lab.wearer.values()] or ["A", "B", "C"]

    def start_assigning(self):
        """Hand identity over by clicking, instead of reading it off a hue."""
        if not self.all_reading:
            self.say("nothing to assign — no clusters in frame", SUN)
            return
        self.show_all = True
        self.assigning = [n for n in self.assignable()]
        if not self.assigning:
            self.say("nobody to assign — scan and connect a ball first", SUN)
            self.assigning = None
            return
        # White is right for MEASURING and useless for CHOOSING: a floor of
        # identical white blobs is exactly the thing a person cannot pick from.
        # So the balls wear their colours while somebody is deciding, and go
        # back to white the moment the deciding is done.
        self._tag_before = self.lab.tag_mode
        self.light_for_assigning()
        self.say(f"click the FRONT light of {self.assigning[0]}"
                 "   (esc to stop)", CHALK)
        self._build()

    def light_for_assigning(self):
        """Colour them so they can be told apart, and single out the next one.

        Colour alone is not enough when the blobs are a few pixels across and
        two slots are neighbours on the wheel — so the ball being asked for is
        also the only one at full brightness. Somebody who cannot read the hue
        can still click the bright one.
        """
        if not self.assigning:
            return
        want = self.assigning[0]
        for ble, r in self.lab.robots.items():
            slot = self.lab.assigned.get(ble)
            rgb = (config.led_for(slot, self.lab.detector.colors)
                   if slot else (255, 255, 255))
            k = 1.0 if ble == want else 0.18
            try:
                r.set_led(tuple(int(round(v * k)) for v in rgb))
            except Exception:
                pass

    def assign_click(self, pos):
        """One click: which cluster, and which end of it is the front."""
        if not self.assigning or self._shot is None:
            return
        (ox, oy), k = self._shot
        px = ((pos[0] - ox) / k, (pos[1] - oy) / k)
        cluster = self.lab.cluster_at(px, self.all_reading)
        if cluster is None:
            self.say("no cluster there — click one of the ringed ones", SUN)
            return
        name = self.assigning[0]
        got, why = self.lab.tracks.assign(name, cluster, front_px=px)
        if got is None:
            self.say(f"{name}: {why}", CORAL)
            return
        self.say(f"{name} is that one, pointing {got.heading:.0f}deg", MINT)
        self.assigning = self.assigning[1:]
        if self.assigning:
            self.light_for_assigning()
            self.say(f"now click the FRONT light of {self.assigning[0]}")
        else:
            self.assigning = None
            self.lab.tag_mode = "manual"
            self.push_led()          # back to white, now that choosing is done
            self.say("all assigned — back to white. Identity is carried frame "
                     "to frame now, and a LOST track will say so", MINT)
        self._build()

    def scroll_list(self, by):
        """Move the fleet list, clamped to what there is."""
        most = max(0, self.list_total - self.list_room)
        want = max(0, min(self.list_scroll + int(by), most))
        if want != self.list_scroll:
            self.list_scroll = want
            self._build()

    def stop_assigning(self):
        self.assigning = None
        self.lab.tag_mode = getattr(self, "_tag_before", self.lab.tag_mode)
        self.push_led()
        self._build()

    def forget_tracks(self):
        self.lab.tracks.drop()
        self.lab.tag_mode = "colour"
        self.say("forgot every assignment — back to reading colour")
        self._build()

    def all_white(self):
        """Drive every ball white, which is three dies instead of one.

        Only sane once identity is not coming from the colour: a primary
        spends two thirds of the LED getting there, and on a rig whose lights
        are a few pixels across that is the difference between a blob the
        reader can work with and one it cannot.
        """
        self.lab.tag_mode = "manual"
        self.push_led()
        self.say("every ball driven WHITE — 3x the light of a primary. "
                 "Colour no longer identifies anything", MINT)
        self._build()

    # -- the arena ------------------------------------------------------

    def start_picking(self):
        """Re-pick the four arena corners, here rather than in another app.

        A homography calibrated at another resolution or another camera
        position warps the wrong part of the frame and reports centimetres
        that are fiction — and nothing in the picture will tell you. This is
        the whole cure, and it has to be to hand in the app you are already
        looking at.
        """
        self.picking = []
        # The brightness view has been through the homography, so a click on
        # it is a click on the OLD arena — which is the one being replaced.
        if self.view == "bright":
            self.view = "camera"
            self._build()
        self.say("click the arena corners: origin, +x, +x+y, +y  "
                 "(backspace undoes, w writes)", CHALK)

    def add_corner(self, pos):
        """One click, converted back into frame pixels."""
        if self.picking is None or self._shot is None:
            return
        frame = self.grab.frame
        if frame is None:
            return
        (ox, oy), k = self._shot
        x, y = (pos[0] - ox) / k, (pos[1] - oy) / k
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.picking.append((float(x), float(y)))
        self.say(f"{len(self.picking)}/4 corners")
        if len(self.picking) == 4:
            self.build_homography()

    def build_homography(self):
        """Four corners into a homography, sized from the workspace in cm.

        Held in memory only. `calib/homography.json` is live state the tracker,
        the planner and every bench read, so it is overwritten when somebody
        presses `w` and not a moment before.
        """
        from workspace.space import Workspace
        try:
            ws = Workspace.load()
            hom = Homography().set_rect(self.picking, ws.width, ws.height)
        except Exception as e:
            self.say(f"could not build a homography: {e}", CORAL)
            self.picking = None
            return
        self.lab.hom = hom
        self.picking = None
        got = self.camera_size()
        note = (f"arena set — {ws.width:.0f}x{ws.height:.0f}cm"
                + (f" over {got[0]}x{got[1]}" if got else ""))
        self.say(note + ".  NOT saved yet — press w", MINT)

    def save_homography(self):
        if self.lab.hom is None or not self.lab.hom.ready:
            self.say("no arena to save — press c and click four corners", SUN)
            return
        try:
            self.lab.hom.save()
        except Exception as e:
            self.say(f"could not write calib/homography.json: {e}", CORAL)
            return
        self.say("wrote calib/homography.json", MINT)

    def toggle_all(self):
        """Label every cluster, or just read the one."""
        self.show_all = not self.show_all
        if self.show_all:
            self.say("labelling every cluster — code and colour are what "
                     "VISION believes, and conf is how much of that belief "
                     "survives the front/back test", CHALK)
        self._build()

    def toggle_mask(self):
        self.view = "mask" if self.view != "mask" else "camera"
        if self.view == "mask":
            self.say("mask — mint is the cluster the heading came from, cyan "
                     "is a light that is not in it, coral is above the light "
                     "floor and thrown away on area", CHALK)
        self._build()

    def toggle_view(self):
        self.view = "bright" if self.view != "bright" else "camera"
        if self.view == "bright":
            self.say("brightness over the arena only — red is clipped, and "
                     "clipped is where hue and heading both die")
        self._build()

    def set_color(self, name):
        self.lab.color = name
        self.hue = self.lab.detector.colors[name]["hue"]
        self.push_led()
        self.say(f"hunting {name}")
        self._build()

    # -- layout ----------------------------------------------------------

    def _build(self):
        self.buttons, self.sliders = [], []
        self.head_y = {"camera": 62}
        x, w = PANEL_X, PANEL_W
        y = 62

        def btn(rect, label, cb, tone=None, on=False):
            b = Button(rect, label, cb, tone=tone)
            b.on = on
            self.buttons.append(b)
            return b

        def sl(label, lo, hi, get, set_, log=False):
            nonlocal y
            self.sliders.append(Slider((x, y, w, 20), label, lo, hi, get, set_,
                                       log=log))
            y += 26

        # CAMERA
        # Header plus THREE info lines — source, frames, shutter. The count is
        # here and the lines are drawn in `draw`, so a line added there has to
        # be paid for here or the buttons land on top of it.
        y += 24 + 51
        btn((x, y, 92, 26), "manual", self.manual)
        btn((x + 100, y, 110, 26), "af off", self.focus_off)
        btn((x + 218, y, 110, 26), "af on", self.focus_auto)
        btn((x + 336, y, 120, 26), "brightness", self.toggle_view,
            on=(self.view == "bright"))
        y += 32
        btn((x, y, 110, 26), "mask", self.toggle_mask,
            on=(self.view == "mask"))
        btn((x + 118, y, 118, 26), "all bots", self.toggle_all,
            on=self.show_all)
        y += 32
        held = bool(self.lab.tracks.names)
        btn((x, y, 110, 26), "assign", self.start_assigning,
            on=self.assigning is not None)
        btn((x + 118, y, 100, 26), "white", self.all_white,
            on=(self.lab.tag_mode == "manual"))
        if held:
            btn((x + 226, y, 100, 26), "forget", self.forget_tracks, tone=CORAL)
        y += 34
        sl("focus", shutter.FOCUS_MIN, shutter.FOCUS_MAX,
           lambda: self.focus_dial.value, self.focus_dial.set)
        # Logarithmic, because the useful end of 3..2047 for a dark floor
        # and bright LED cores is all under about 300 — a linear track spends
        # six sevenths of itself above anything you would choose.
        sl("shutter", shutter.EXP_MIN, shutter.EXP_MAX,
           lambda: self.dial.value, self.dial.set, log=True)
        y += 4
        btn((x, y, 110, 24), "exp auto", self.exposure_auto)
        btn((x + 118, y, 110, 24), "exp save", self.exposure_save)
        y += 34

        # VISION
        self.head_y["vision"] = y
        y += 24
        # In manual-identity mode NOTHING downstream reads a colour: the balls
        # are all white and the names came from a person. Every control below
        # that exists to tune a hue is then a knob that moves nothing, so it
        # is not shown — and the room it frees is exactly what the fleet list
        # needed.
        by_colour = self.lab.tag_mode == "colour"
        if by_colour:
            bw = (w - 5 * 6) // 6
            for i, name in enumerate(self.COLORS):
                btn((x + i * (bw + 6), y, bw, 24), name[:3],
                    (lambda n=name: self.set_color(n)),
                    on=(name == self.lab.color))
            y += 64
            btn((x, y - 32, 120, 24), "lights", self.toggle_mode,
                on=(self.lab.mode == "lights"))
        else:
            btn((x, y, 120, 24), "lights", self.toggle_mode,
                on=(self.lab.mode == "lights"))
            y += 32
        sl("light floor", 60, 254, lambda: self.lab.min_v,
           lambda v: setattr(self.lab, "min_v", int(v)))
        sl("ball span", 10, 300, lambda: self.lab.span_px,
           lambda v: setattr(self.lab, "span_px", int(v)))

        # WHAT THE BALL IS DRIVEN AT — set directly, in the units the ball
        # takes. Deriving it from the hunted hue assumes the LED emits what
        # the maths says and the sensor reads back what the LED emits; on this
        # bench a slot driven from hue 172 reads at 165, so the round trip
        # through HSV puts an error in rather than taking one out.
        if by_colour:
            for i, ch in enumerate("rgb"):
                sl(f"led {ch}", 0, 255,
                   (lambda k=i: self.led_channel(k)),
                   (lambda v, k=i: self.set_led_channel(k, v)))
            # ...and WHAT THE TRACKER HUNTS FOR, a different number.
            sl("hue", 0, 179, lambda: self.hue, self.set_hue)
            sl("tol", 1, 40,
               lambda: self.lab.detector.colors[self.lab.color]["tol"],
               self.set_tol)
        if self.lab.mode != "lights":
            # Blob-mode knobs, every one of them. `analyse_lights` reads only
            # `min_v` and `span_px` — showing the rest in lights mode is a
            # panel full of sliders that move nothing, which is precisely the
            # failure this bench exists to stop people chasing.
            sl("unfocus", 1, 31, lambda: self.lab.unfocus,
               lambda v: setattr(self.lab, "unfocus", v))
            sl("peak floor", 20, 95, lambda: self.lab.floor,
               lambda v: setattr(self.lab, "floor", int(v)))
            # Blob-mode knobs. In `lights` mode nothing reads them, and a
            # slider that does nothing is the thing this bench exists to stop
            # people turning.
            sl("s min", 0, 255, lambda: self.lab.detector.thresh["s_min"],
               lambda v: self.lab.detector.thresh.__setitem__("s_min", int(v)))
            sl("v min", 0, 255, lambda: self.lab.detector.thresh["v_min"],
               lambda v: self.lab.detector.thresh.__setitem__("v_min", int(v)))
            sl("min area", 10, 4000,
               lambda: self.lab.detector.thresh["min_area"],
               lambda v: self.lab.detector.thresh.__setitem__("min_area", int(v)))
        if by_colour:
            btn((x, y, 130, 24), "hunt seen", self.hunt_seen)
            y += 34
        else:
            y += 6

        # ROBOT
        self.head_y["robot"] = y
        y += 24
        held = self.lab.robots
        btn((x, y, 92, 26), "scan", self.start_scan)
        if held:
            btn((x + 100, y, 150, 26), "release all",
                lambda: self.release(None), tone=CORAL)
            up = sum(1 for r in held.values() if r.link_up)
            self.link_note = f"{up}/{len(held)} linked"
        else:
            self.link_note = "no ball"
        y += 34
        # The LED sliders sit ABOVE the ball list, not below it. The list grows
        # with the fleet and with whatever the last scan turned up, so anything
        # under it is one extra robot away from being off the window — which is
        # exactly how `taillight` ended up unreachable with three balls held.
        sl("bright", 0, 255, lambda: self.bright, self.set_bright)
        sl("taillight", 0, 255, lambda: self.tail, self.set_tail)
        sl("drive", 0, 255, lambda: self.drive_byte,
           lambda v: setattr(self, "drive_byte", int(v)))
        y += 6

        # The readout needs room whatever the fleet is doing, so the list is
        # given what is left and no more. A ball that does not fit is counted
        # rather than silently missing.
        # The readout's own budget, reserved before the list gets anything.
        # The fleet list then takes what is left and SCROLLS inside it, so the
        # panel below never moves however many balls are held or found — a
        # layout that reflows when a scan finishes is a layout where the thing
        # you were about to click has gone somewhere else.
        self.readout_top = H - 168
        self.list_rect = pygame.Rect(x, y, w, max(0, self.readout_top - 24 - y))
        room = max(1, self.list_rect.h // 28)

        # Connected balls first, then what the last scan saw and has not been
        # taken. One list, so one scrollbar.
        rows_all = ([("held", ble, r) for ble, r in held.items()]
                    + [("found", n, None) for n in self.found if n not in held])
        self.list_total = len(rows_all)
        top = max(0, min(self.list_scroll, max(0, self.list_total - room)))
        self.list_scroll = top
        self.list_room = room
        self.hidden_rows = max(0, self.list_total - room)
        rows_all = rows_all[top:top + room]

        # One row per CONNECTED ball, carrying the colour it was told to wear.
        # That colour is the whole identity story here, so it is shown as a
        # swatch on the row rather than left to be inferred from the overlay.
        self.swatches = []
        self.link_rows = []
        shown = [(ble, r) for kind, ble, r in rows_all if kind == "held"]
        for ble, r in shown:
            slot = self.lab.assigned.get(ble, "?")
            self.swatches.append((x, y, slot_rgb(self.lab.detector.colors,
                                                 slot)))
            btn((x + 22, y, 132, 24), f"{ble} {slot[:3]}",
                (lambda n=ble: self.cycle_color(n)), on=r.link_up)
            btn((x + 160, y, 62, 24), "retry",
                (lambda n=ble: self.retry(n)), tone=None)
            btn((x + 228, y, 56, 24), "drop",
                (lambda n=ble: self.release(n)), tone=CORAL)
            # What the link is DOING, beside the row that controls it. A ball
            # that is still trying and a ball that has given up look identical
            # otherwise, and the difference decides whether to press retry.
            if r.link_up:
                note, tone = "linked", MINT
            elif getattr(r, "max_connections_hit", False):
                note, tone = "radio full", CORAL
            elif getattr(r, "last_error", None):
                note, tone = f"try {getattr(r, 'attempts', 0)} — failing", CORAL
            else:
                note, tone = f"connecting… ({getattr(r, 'attempts', 0)})", SUN
            self.link_rows.append((x + 292, y + 5, note, tone))
            y += 28

        # ...and one per ball SEEN but not yet connected.
        for name in [n for kind, n, _ in rows_all if kind == "found"]:
            btn((x, y, 240, 24), name, (lambda n=name: self.connect(n)))
            y += 28

    # -- slider setters (they push to hardware, so they are not lambdas) --

    def focus_off(self):
        """Take the lens off autofocus and pin it where the slider says."""
        self.focus_dial.pending = self.focus_dial.value
        got = self.focus_dial.push(force=True)
        if got is None or got[0]:
            self.say(f"autofocus off, focus pinned at "
                     f"{self.focus_dial.value}", MINT)
        else:
            self.say(got[1], CORAL)

    def focus_auto(self):
        ok, msg = self.focus_dial.auto()
        self.say(msg, MINT if ok else CORAL)

    def exposure_auto(self):
        ok, msg = self.dial.auto()
        self.say(msg, MINT if ok else SUN)

    def exposure_save(self):
        self.say(self.dial.save(), MINT)

    def led_channel(self, k):
        """One channel of the RGB the selected slot is driven at."""
        return int(config.led_for(self.lab.color, self.lab.detector.colors)[k])

    def set_led_channel(self, k, v):
        """Set one channel, and light every ball wearing this slot.

        Writing the WHOLE triple onto the signature the first time a channel
        moves, rather than storing a partial one: a signature carrying `rgb`
        with a missing channel would be a slot whose colour depends on which
        slider was touched, and `led_for` cannot tell that from a deliberate
        zero.
        """
        rgb = list(config.led_for(self.lab.color, self.lab.detector.colors))
        rgb[k] = int(max(0, min(255, v)))
        self.lab.detector.colors[self.lab.color]["rgb"] = rgb
        self.push_led()

    def hunt_seen(self):
        """Hunt for the hue the CAMERA is reading, not the one we drove.

        The gap between the two is the whole reason this button exists. A slot
        driven from hue 172 reads at 165 on this bench, and 7 units is more
        than the tolerance — so the tag stops naming the ball, front and back
        stop being separable, and the heading flips 180 degrees. Reading the
        answer off the light itself beats typing a number at it.
        """
        r = self.reading
        light = r.get("front") or (r.get("group") or r.get("lights") or [None])[0]
        seen = (light or {}).get("hue")
        if seen is None:
            self.say("no readable hue on the light right now — the ring has "
                     "no colour left, so lengthen the shutter first", SUN)
            return
        was = self.lab.detector.colors[self.lab.color]["hue"]
        self.hue = int(round(seen))
        self.lab.detector.colors[self.lab.color]["hue"] = self.hue
        self.say(f"{self.lab.color} now hunted at {self.hue}, was {was} — "
                 f"the LED drive is unchanged", MINT)

    def set_hue(self, v):
        """One hue, moving the LED and the detector together.

        `vision/config.py` keeps the colour a ball is TOLD to glow and the hue
        the tracker HUNTS in one table on purpose: two tables is how a robot
        ends up lit one colour and looked for as another, each table
        internally consistent and nothing complaining. A lab with separate
        sliders for them would reintroduce exactly that, and would do it on
        the bench where people go to diagnose it.
        """
        self.hue = int(v)
        self.lab.detector.colors[self.lab.color]["hue"] = self.hue
        # Only relight when the slot has no explicit drive of its own. Once a
        # person has dialled the emitter by hand, moving the HUNTED hue must
        # not silently undo that — the two numbers are allowed to differ, and
        # on this camera they have to.
        if not self.lab.detector.colors[self.lab.color].get("rgb"):
            self.push_led()

    def set_tol(self, v):
        self.lab.detector.colors[self.lab.color]["tol"] = int(v)

    def set_bright(self, v):
        self.bright = int(v)
        self.push_led()

    def set_tail(self, v):
        self.tail = int(v)
        self.push_tail()

    # -- the loop --------------------------------------------------------

    def tick(self):
        self.drain_scan()
        # Rate limited inside `push`, so this is cheap on the frames where the
        # slider has not moved and smooth on the ones where it has.
        # Why the fleet is not all up. Said once per change: this runs from
        # the render loop, and a persistent fault would otherwise fill the log
        # with one message thirty times a second.
        trouble = self.lab.link_trouble()
        if trouble != self._said_link:
            self._said_link = trouble
            if trouble:
                self.say(trouble, CORAL)

        for d in (self.dial, self.focus_dial):
            pushed = d.push()
            if pushed is not None and not pushed[0]:
                self.say(pushed[1], CORAL)
        frame = self.grab.frame
        if not self.paused:
            self.reading = self.lab.analyse(frame)
            # Only when it is being shown: this walks every cluster in the
            # frame and the bench runs at 30fps with a 1080p read already.
            # The tracker needs every cluster, whether or not they are being
            # drawn — an identity that only survives while a view is open is
            # not an identity.
            self.all_reading = (self.lab.analyse_all(frame)
                                if (self.show_all or self.lab.tracks.names)
                                else [])
            now_t = time.time()
            step = now_t - self.last_tick
            if self.lab.tracks.names:
                self.lab.track(self.all_reading, step)
            self.last_tick = now_t
            self.drive_tick(max(min(step, 0.2), 1e-3))
            self.log_reading()
        now = time.time()
        self.fps_count += 1
        if now - self.fps_at >= 1.0:
            self.fps = self.fps_count / (now - self.fps_at)
            self.fps_at, self.fps_count = now, 0

    def log_reading(self):
        now = time.time()
        if now - self.last_log < 1.0 / LOG_HZ:
            return
        self.last_log = now
        r = self.reading
        stamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
        if r["theta_img"] is None:
            self.lines.append((stamp, None, r["why"] or "no reading"))
            return
        if r["xy_cm"] is not None:
            x, y = r["xy_cm"]
            where = f"x {x:7.1f}  y {y:7.1f} cm"
            th = r["theta_cm"]
            head = f"th {th:6.1f}deg" if th is not None else f"th {r['theta_img']:6.1f}img"
        else:
            x, y = r["centre_px"]
            where = f"x {x:7.1f}  y {y:7.1f} px"
            head = f"th {r['theta_img']:6.1f}img"
        self.lines.append((stamp, True,
                           f"{where}  {head}  conf {r['conf']:.2f}  "
                           f"r {r['radius_px']:.0f}px"))

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return self.close()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE and self.assigning:
                        self.stop_assigning()
                        self.say("stopped assigning")
                        continue
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return self.close()
                    if e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        self.say("paused" if self.paused else "running")
                    if e.key == pygame.K_TAB:
                        self.cycle_drive()
                    if e.key == pygame.K_g:
                        if e.mod & pygame.KMOD_SHIFT:
                            self.forget_aim()
                        else:
                            self.save_aim()
                    if e.key == pygame.K_i:
                        self.start_assigning()
                    if e.key == pygame.K_c:
                        self.start_picking()
                    if e.key == pygame.K_w:
                        self.save_homography()
                    if e.key == pygame.K_BACKSPACE and self.picking:
                        self.picking.pop()
                        self.say(f"{len(self.picking)}/4 corners")
                    if e.key == pygame.K_x:
                        self.lines.clear()
                    if e.key == pygame.K_m:
                        self.toggle_mask()
                    if e.key == pygame.K_b:
                        self.toggle_view()
                    if e.key == pygame.K_a:
                        self.toggle_all()
                    if e.key == pygame.K_f:
                        self.focus_off()
                    if e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        # One notch of the shutter, either way. A slider is
                        # hard to nudge by one step and the useful range here
                        # is narrow, so the keys step it instead.
                        step = -1 if e.key == pygame.K_LEFTBRACKET else 1
                        pos = shutter.us_to_slider(self.dial.value) + step
                        self.dial.set(shutter.slider_to_us(pos))
                if e.type == pygame.MOUSEBUTTONDOWN:
                    p = e.pos
                    if self.picking is not None and VIEW.collidepoint(p):
                        self.add_corner(p)
                        continue
                    if self.assigning and VIEW.collidepoint(p):
                        self.assign_click(p)
                        continue
                    if not any(b.hit(p) for b in self.buttons):
                        for s in self.sliders:
                            if s.hit(p):
                                break
                if e.type == pygame.MOUSEWHEEL:
                    if self.list_rect.collidepoint(pygame.mouse.get_pos()):
                        self.scroll_list(-e.y)
                if e.type == pygame.MOUSEBUTTONUP:
                    for s in self.sliders:
                        s.dragging = False
                if e.type == pygame.MOUSEMOTION:
                    for s in self.sliders:
                        if s.dragging:
                            s.drag(e.pos)
            self.tick()
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        # Join before releasing. `VideoCapture.release()` while another thread
        # is inside `read()` is a segfault in OpenCV, not an exception — and it
        # lands on the way out, where it reads as "the app crashed on quit"
        # rather than as the shutdown-order bug it is.
        self.grab.stop()
        self.grab.join(timeout=1.5)
        self.lab.close()
        pygame.quit()

    # -- drawing ---------------------------------------------------------

    def draw(self):
        s = self.screen
        s.fill(INK)
        s.blit(self.fb.render("taillight", True, CHALK), (14, 12))
        s.blit(self.f.render(
            "arrows drive · tab ball · i assign · c corners · w write · m mask · a all · q",
            True, GREY), (130, 16))
        self.draw_view(s)
        self.draw_log(s)
        self.draw_profile(s)
        self.draw_panel(s)

    def _blit_bgr(self, s, img):
        """Centre a BGR image in the view pane. Returns (origin, scale)."""
        fh, fw = img.shape[:2]
        scale = min(VIEW.w / fw, VIEW.h / fh)
        small = cv2.resize(img, (int(fw * scale), int(fh * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        at = (VIEW.x + (VIEW.w - surf.get_width()) // 2,
              VIEW.y + (VIEW.h - surf.get_height()) // 2)
        s.blit(surf, at)
        self._shot = (at, scale)
        return at, scale

    def draw_bright(self, s, frame):
        img, stats, why = self.lab.brightness_map(frame, VIEW.w, VIEW.h)
        self.light = stats
        if img is None:
            s.blit(self.f.render(why or "no brightness map", True, SUN),
                   (VIEW.x + 16, VIEW.y + VIEW.h // 2))
            return
        at, blit_scale = self._blit_bgr(s, img)
        ih, iw = img.shape[:2]

        # A light at frame (900, 400) is somewhere else entirely on a warped
        # floor, so the overlay needs the same transform the map went through
        # rather than the pane's plain scale.
        hom = self.lab.hom
        if hom is not None and hom.ready:
            m = min(VIEW.w / hom.width, VIEW.h / hom.height) * blit_scale

            def px_of(p):
                cm = np.asarray(hom.to_cm([list(p)])).ravel()[:2]
                out = (at[0] + float(cm[0]) * m, at[1] + float(cm[1]) * m)
                inside = (at[0] <= out[0] <= at[0] + iw * blit_scale
                          and at[1] <= out[1] <= at[1] + ih * blit_scale)
                return out if inside else None
        else:
            px_of = None
        self.draw_overlay(s, at, blit_scale, px_of=px_of)

        bar = pygame.Rect(VIEW.x + 10, VIEW.bottom - 26, VIEW.w - 20, 16)
        pygame.draw.rect(s, (6, 16, 28), bar, border_radius=3)
        blown, dark = stats["blown"], stats["dark"]
        line = (f"mean {stats['mean']:5.1f}   p05 {stats['p05']:5.1f}   "
                f"p95 {stats['p95']:5.1f}   max {stats['max']:3d}   "
                f"clipped {blown:5.2f}%   dark {dark:5.1f}%")
        tone = CORAL if blown > 0.5 else (SUN if blown > 0.05 else MINT)
        s.blit(self.fs.render(line, True, tone), (bar.x + 8, bar.y + 3))
        if why:
            s.blit(self.fs.render(why, True, SUN), (VIEW.x + 12, VIEW.y + 8))

    def draw_mask(self, s, frame):
        """The mask, at the frame's own scale so the overlay lines up.

        No blur and no homography: this is the image `lights_px` was handed,
        and putting it through either would make it a picture of something
        else — which is the one thing a mask view must not be.
        """
        img, stats = self.lab.mask_view(frame, self.reading.get("group"))
        at, scale = self._blit_bgr(s, img)
        self.draw_overlay(s, at, scale)

        bar = pygame.Rect(VIEW.x + 10, VIEW.bottom - 26, VIEW.w - 20, 16)
        pygame.draw.rect(s, (6, 16, 28), bar, border_radius=3)
        line = (f"light floor V>={self.lab.min_v}   lit {stats['lit_pct']:5.2f}%"
                f"   regions {stats['regions']:4d}   passed area "
                f"{stats['passed']:3d}   peak V {stats['max']:3d}")
        # A frame that is mostly lit is a shutter still too long, whatever the
        # regions say — the floor arrives as one huge region, not as many.
        tone = (CORAL if stats["lit_pct"] > 5.0 else
                SUN if stats["lit_pct"] > 1.0 else MINT)
        s.blit(self.fs.render(line, True, tone), (bar.x + 8, bar.y + 3))

    # How much confidence a heading needs before its label is drawn as an
    # answer rather than as a doubt. Measured, not chosen: swept over rendered
    # balls, every front/back REVERSAL sat below 0.3 while three quarters of
    # the correct readings sat above it. A reversal is the one error a
    # controller cannot recover from — it drives the ball at the wall — so the
    # gate is placed where the two populations actually separate.
    CONF_GATE = 0.3

    def draw_assigning(self, s, px_of, scale):
        """Every cluster, ringed, while somebody says which is which."""
        for c in self.all_reading:
            centre = c.get("centre")
            if centre is None:
                continue
            p = px_of(centre)
            if p is None:
                continue
            pygame.draw.circle(s, CHALK, (int(p[0]), int(p[1])), 22, 1)
            # The lights themselves, so it is obvious WHICH one to click.
            for light in c.get("group") or []:
                q = px_of((light["x"], light["y"]))
                if q is not None:
                    pygame.draw.circle(s, SUN, (int(q[0]), int(q[1])), 5, 1)
        want = self.assigning[0] if self.assigning else ""
        s.blit(self.fb.render(f"click the FRONT light of {want}", True, SUN),
               (VIEW.x + 16, VIEW.y + 14))
        s.blit(self.fs.render("esc to stop", True, DIM),
               (VIEW.x + 16, VIEW.y + 36))

    def draw_tracks(self, s, px_of, scale):
        """What this app believes, and how much it still believes it.

        A LOST track keeps its last position on screen but is drawn as a
        DOUBT — dashed, named, with the reason. Hiding it would be worse
        (a robot that vanished from the display is a robot nobody looks for)
        and drawing it as live would be worse still.
        """
        for t in self.lab.tracks.by_name.values():
            p = px_of(t.centre)
            if p is None:
                continue
            px, py = int(p[0]), int(p[1])
            tone = (CORAL if t.contended else SUN if t.lost else MINT)
            pygame.draw.circle(s, tone, (px, py), 20, 1 if t.lost else 2)
            if not t.lost:
                a = math.radians(t.heading)
                tip = (px + math.cos(a) * 40, py + math.sin(a) * 40)
                pygame.draw.line(s, tone, (px, py),
                                 (int(tip[0]), int(tip[1])), 2)
                pygame.draw.circle(s, tone, (int(tip[0]), int(tip[1])), 4)

            rows = [self.fb.render(
                t.name + ("  LOST" if t.lost else ""), True, tone)]
            if t.lost or t.why:
                rows.append(self.fs.render(
                    fit(self.fs, t.why or "", 240)[0] if t.why else "",
                    True, DIM))
            if not t.lost:
                rows.append(self.fs.render(
                    f"{t.heading:.0f}deg"
                    + ("  motion-checked" if t.confirmed_at else "")
                    + (f"  {t.flips_caught} flip fixed" if t.flips_caught
                       else ""), True, DIM))
            widest = max(r.get_width() for r in rows)
            lx = px + 28
            if lx + widest > VIEW.right - 6:
                lx = px - 28 - widest
            for k, row in enumerate(rows):
                s.blit(row, (max(VIEW.x + 4, lx), py - 18 + k * 15))

    def draw_labels(self, s, px_of, scale):
        """Which cluster vision thinks is which robot, drawn on each one.

        The label is what the CAMERA believes, never what the roster wishes.
        A cluster it cannot name says so, and one it names without enough
        front/back certainty is drawn in the doubt colour rather than left off
        — a bench that hides its uncertain answers is how a 180-degree heading
        reaches a controller unchallenged.
        """
        for i, c in enumerate(self.all_reading):
            centre = c.get("centre")
            if centre is None:
                continue
            p = px_of(centre)
            if p is None:
                continue
            px, py = int(p[0]), int(p[1])

            conf = c.get("conf")
            if c["why"] is not None:
                name, tone = "?", SUN
            elif conf is not None and conf < self.CONF_GATE:
                name = (c.get("code") or c.get("color") or "?") + " ??"
                tone = CORAL
            else:
                name = c.get("code") or c.get("color") or "unnamed"
                tone = MINT if c.get("code") else CYAN

            # A ring the size of the cluster, so two balls close together stay
            # visually separate, and the text clear of it.
            span = 12.0
            if len(c["group"]) > 1:
                xs = [l["x"] for l in c["group"]]
                ys = [l["y"] for l in c["group"]]
                span = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 8.0)
            rad = max(10, int(span * scale * 0.8))
            pygame.draw.circle(s, tone, (px, py), rad, 1)

            if c.get("color"):
                pygame.draw.circle(
                    s, slot_rgb(self.lab.detector.colors, c["color"]),
                    (px, py - rad - 9), 5)
            label = self.fb.render(name, True, tone)
            detail = (_wrap(c["why"], 30)[0] if c["why"] else
                      f"conf {conf:.2f}" + (f"  {c['color']}" if c.get("code")
                                            and c.get("color") else ""))
            rows = [label, self.fs.render(detail, True, DIM)]
            # WHICH TERM is holding this ball back, per ball. The product on
            # its own cannot say whether to move the camera or fix the colour,
            # and with several balls in frame they may need different answers.
            if c.get("conf_span") is not None:
                worst = ("span" if c["conf_span"] <= c["conf_ends"] else "ends")
                rows.append(self.fs.render(
                    f"span {c['conf_span']:.2f} ({c['span_px']:.0f}px)  "
                    f"ends {c['conf_ends']:.2f}  <- {worst}", True,
                    SUN if min(c["conf_span"], c["conf_ends"]) < 0.6 else GREY))
            if c.get("xy_cm") is not None:
                rows.append(self.fs.render(
                    f"{c['xy_cm'][0]:.0f}, {c['xy_cm'][1]:.0f} cm", True, GREY))

            # Flip to the left of the ring when the text would run off the
            # pane. Drawn past the edge it does not vanish — it lands on the
            # control panel, over the sliders, which reads as the app being
            # broken rather than as a label being too long.
            widest = max(r.get_width() for r in rows)
            lx = px + rad + 6
            if lx + widest > VIEW.right - 6:
                lx = px - rad - 6 - widest
            lx = max(VIEW.x + 4, lx)
            for k, row in enumerate(rows):
                s.blit(row, (lx, py - 18 + k * 15))

    def draw_view(self, s):
        card(s, VIEW, fill=(8, 20, 34))
        frame = self.grab.frame
        if frame is None:
            msg = self.grab.error or "waiting for the first frame…"
            s.blit(self.f.render(msg, True, SUN),
                   (VIEW.x + 16, VIEW.y + VIEW.h // 2))
            return

        if self.view == "bright":
            return self.draw_bright(s, frame)
        if self.view == "mask":
            return self.draw_mask(s, frame)

        # What the DETECTOR sees, not what the sensor sent — a bench that shows
        # you a sharp picture while hunting blobs in a blurred one is lying
        # about the thing you are tuning.
        b = self.lab.unfocus
        shown = cv2.GaussianBlur(frame, (b | 1, b | 1), 0) if b > 1 else frame
        at, scale = self._blit_bgr(s, shown)

        self.draw_overlay(s, at, scale)

    def draw_overlay(self, s, at, scale, px_of=None):
        """Every light, and the arrow. Drawn over whichever view is showing.

        `px_of` maps a source pixel into the pane; the default is the plain
        scale-and-offset the camera view uses. The brightness view passes its
        own, because that image has been through the homography and a light at
        frame (900, 400) is somewhere else entirely on the warped floor.
        """
        r = self.reading
        if px_of is None:
            def px_of(p):
                return (at[0] + p[0] * scale, at[1] + p[1] * scale)

        if self.picking is not None:
            pts = [px_of(q) for q in self.picking]
            pts = [(int(q[0]), int(q[1])) for q in pts if q is not None]
            for j, q in enumerate(pts):
                pygame.draw.circle(s, SUN, q, 5, 2)
                s.blit(self.fs.render("origin +x +x+y +y".split()[j], True, SUN),
                       (q[0] + 8, q[1] - 6))
            if len(pts) > 1:
                pygame.draw.lines(s, SUN, len(pts) == 4, pts, 1)
            return          # nothing else while picking: the clicks are the job

        if self.assigning:
            self.draw_assigning(s, px_of, scale)
        elif self.lab.tracks.names:
            self.draw_tracks(s, px_of, scale)
        elif self.show_all:
            self.draw_labels(s, px_of, scale)

        # Every light that passed the floor, whether or not it made a heading.
        # Seeing the four spots it found is what tells you the fifth is a
        # reflection; a view that draws only the answer cannot show you that.
        for light in r.get("lights") or []:
            p = px_of((light["x"], light["y"]))
            if p is None:
                continue
            rad = max(3, int(light["core_px"] * scale))
            in_group = any(g is light for g in (r.get("group") or []))
            pygame.draw.circle(s, CHALK if in_group else GREY,
                               (int(p[0]), int(p[1])), rad + 3, 1)

        if r["centre_px"] is None:
            return
        c = px_of(r["centre_px"])
        if c is None:
            return
        cx, cy = c
        rad = max(6.0, (r["radius_px"] or 6.0) * scale)

        if r["theta_img"] is None:
            pygame.draw.circle(s, SUN, (int(cx), int(cy)), int(rad), 1)
            return

        # The arrow is drawn from the IMAGE angle, which is what was actually
        # measured. The arena heading beside it has been through the
        # homography; if the two ever disagree about which way is which, the
        # calibration is the thing to doubt.
        # From the two mapped ENDPOINTS, not from the angle. `theta_img` is an
        # angle in the camera frame and the brightness view has been through
        # the homography, which does not preserve angles — drawing the image
        # angle over warped dots puts the arrow at an angle to its own lights.
        pf = px_of((r["front"]["x"], r["front"]["y"])) if r.get("front") else None
        pb = px_of((r["back"]["x"], r["back"]["y"])) if r.get("back") else None
        if pf is not None and pb is not None:
            ang = math.atan2(pf[1] - pb[1], pf[0] - pb[0])
            rad = max(6.0, math.hypot(pf[0] - pb[0], pf[1] - pb[1]) / 2.0)
        else:
            ang = math.radians(r["theta_img"])
        tip = (cx + math.cos(ang) * rad * 1.9, cy + math.sin(ang) * rad * 1.9)
        tail = (cx - math.cos(ang) * rad * 1.4, cy - math.sin(ang) * rad * 1.4)
        tone = MINT if (r["conf"] or 0) > 0.35 else SUN
        pygame.draw.line(s, tone, tail, tip, 3)
        for side in (150, -150):
            a2 = ang + math.radians(side)
            pygame.draw.line(s, tone, tip,
                             (tip[0] + math.cos(a2) * rad * 0.7,
                              tip[1] + math.sin(a2) * rad * 0.7), 3)

        # Mark which light was called the front and which the back, because
        # that decision is the one most likely to be wrong by exactly 180
        # degrees — and a 180 error is worse than no reading at all.
        for light, label, colour in ((r.get("front"), "F", MINT),
                                     (r.get("back"), "B", CORAL)):
            if not light:
                continue
            p = px_of((light["x"], light["y"]))
            if p is None:
                continue
            s.blit(self.fs.render(label, True, colour),
                   (int(p[0]) + 8, int(p[1]) - 6))

    def draw_profile(self, s):
        """Saturation and value against radius, for one light.

        ANNULUS is a guess — 1.1 to 2.4 times the core — and the right values
        depend on the lens, the exposure and how hard the LED is driven. This
        is how you replace the guess with a measurement: the white core is
        where SAT collapses, the halo is where SAT is high and VAL is still up,
        and floor is where VAL falls away. The shaded band is where the code is
        currently sampling; if it is not sitting on the halo, that is the bug.
        """
        card(s, PROFILE)
        # Clipped like the log, and for the same reason: `tag_why` is a
        # refusal message of unbounded length in a fixed card.
        s.set_clip(PROFILE.inflate(-8, -8))
        try:
            self._profile_body(s)
        finally:
            s.set_clip(None)

    def _profile_body(self, s):
        y = section(s, self.fs, "colour vs radius", PROFILE.x + 10,
                    PROFILE.y + 8, PROFILE.w - 20)
        r = self.reading
        light = r.get("front") or (r.get("group") or r.get("lights") or [None])[0]
        frame = self.grab.frame
        if light is None or frame is None:
            s.blit(self.f.render("no light to profile", True, GREY),
                   (PROFILE.x + 12, y + 10))
            return

        core = max(1.0, float(light.get("core_px") or 1.0))
        # Out to the light's own EXTENT, not to a multiple of its core.
        #
        # `core * 4` is fine for a crisp LED and blind for a bloomed one. On an
        # over-exposed frame the core thresholds tiny -- 2.4px -- while the
        # light itself bleeds out past a hundred, so the profile only ever
        # showed the innermost tenth of it: sat flat and low across the whole
        # plot, because every sample was still inside the white middle. The
        # colour was there, just further out than anything was looking.
        #
        # The floor of 40px is what makes this an instrument rather than a
        # confirmation of the guess it exists to test.
        area = float(light.get("area") or 0.0)
        spread = math.sqrt(max(area, 1.0) / math.pi)
        max_r = max(40.0, core * 4.0, spread * 4.0)
        prof = radial_profile(frame, (light["x"], light["y"]), max_r,
                              step=max(0.5, max_r / 40.0))
        if not prof:
            s.blit(self.f.render("nothing to profile", True, GREY),
                   (PROFILE.x + 12, y + 10))
            return

        plot = pygame.Rect(PROFILE.x + 34, y + 8, PROFILE.w - 50, 190)
        pygame.draw.rect(s, (8, 22, 36), plot)

        def at(row, key):
            fx = plot.x + (row["r"] / max_r) * plot.w
            fy = plot.bottom - (row[key] / 255.0) * plot.h
            return (fx, fy)

        # where the code is sampling now
        band = pygame.Rect(
            plot.x + (ANNULUS[0] * core / max_r) * plot.w, plot.y,
            max(2, ((ANNULUS[1] - ANNULUS[0]) * core / max_r) * plot.w), plot.h)
        band = band.clip(plot)
        if band.w:
            shade = pygame.Surface((band.w, band.h), pygame.SRCALPHA)
            shade.fill((99, 210, 232, 40))
            s.blit(shade, band.topleft)

        # the saturation floor a tag needs to clear
        fy = plot.bottom - (MIN_TAG_SAT / 255.0) * plot.h
        pygame.draw.line(s, (70, 60, 40), (plot.x, fy), (plot.right, fy), 1)

        for key, tone in (("val", CHALK), ("sat", CYAN)):
            pts = [at(row, key) for row in prof]
            if len(pts) > 1:
                pygame.draw.lines(s, tone, False, pts, 2)
        pygame.draw.rect(s, RULE, plot, 1)

        s.blit(self.fs.render("255", True, GREY), (PROFILE.x + 8, plot.y - 4))
        s.blit(self.fs.render("0", True, GREY), (PROFILE.x + 8, plot.bottom - 8))
        s.blit(self.fs.render(f"{max_r:.0f}px", True, GREY),
               (plot.right - 26, plot.bottom + 4))
        ty = plot.bottom + 20
        s.blit(self.fs.render("val", True, CHALK), (PROFILE.x + 12, ty))
        s.blit(self.fs.render("sat", True, CYAN), (PROFILE.x + 52, ty))
        s.blit(self.fs.render(f"band {ANNULUS[0] * core:.0f}-"
                              f"{ANNULUS[1] * core:.0f}px", True, DIM),
               (PROFILE.x + 96, ty))
        ty += 16
        for label, value in (("core", f"{core:.1f}px"),
                             ("ring", f"{light.get('ring_px', 0)}px"),
                             ("sat", f"{(light.get('sat') or 0):.0f}"),
                             ("hue", "—" if light.get("hue") is None
                              else f"{light['hue']:.0f}")):
            s.blit(self.fs.render(label, True, DIM), (PROFILE.x + 12, ty))
            s.blit(self.f.render(value, True, CHALK), (PROFILE.x + 70, ty - 2))
            ty += 17
        if r.get("color"):
            s.blit(self.f.render(f"tag: {r['color']}", True, MINT),
                   (PROFILE.x + 12, ty + 2))
        elif r.get("tag_why"):
            for ln in fit(self.fs, r["tag_why"], PROFILE.w - 24)[:4]:
                s.blit(self.fs.render(ln, True, SUN), (PROFILE.x + 12, ty))
                ty += 13

    def draw_log(self, s):
        card(s, LOG)
        y = section(s, self.fs, "log", LOG.x + 10, LOG.y + 8, LOG.w - 20,
                    f"{LOG_HZ:.0f} hz", GREY)
        rows = (LOG.bottom - y - 8) // 15
        # Clipped to the card. A refusal message is long and the honest ones
        # are the longest of all, so the log is exactly where text escapes —
        # and text that escapes lands on the panel next door.
        s.set_clip(LOG.inflate(-8, -8))
        room = LOG.right - (LOG.x + 96) - 10
        for stamp, ok, text in list(self.lines)[-rows:]:
            s.blit(self.fs.render(stamp, True, GREY), (LOG.x + 10, y))
            line = fit(self.f, text, room)
            s.blit(self.f.render(line[0] if line else "", True,
                                 CHALK if ok else SUN), (LOG.x + 96, y - 2))
            y += 15
        s.set_clip(None)

    def draw_panel(self, s):
        x, w = PANEL_X, PANEL_W
        y = 62
        # Everything in this column stays in this column. The readout and the
        # notes carry refusal messages of unbounded length, and without this
        # they run over the sliders and the camera view.
        s.set_clip(pygame.Rect(x - 4, 0, w + 8, H))

        y = section(s, self.fs, "camera", x, y, w, f"{self.fps:4.1f} fps",
                    MINT if self.fps > 12 else SUN)
        got = self.camera_size()
        s.blit(self.f.render(
            f"{self.lab.source_spec}   "
            + (f"{got[0]}x{got[1]}" if got else "size unknown"), True, CHALK), (x, y))
        y += 17
        err = self.grab.error
        s.blit(self.f.render(err or f"{self.grab.count} frames", True,
                             CORAL if err else DIM), (x, y))
        y += 17
        # OpenCV never learns the shutter changed, so `self.dial.value` is the
        # only thing that knows what was asked for — and `took` is the only
        # thing that knows whether it landed. A slider moving against a picture
        # that does not is this project's oldest failure; say which it is.
        if not self.dial.available:
            shut, tone = "no uvc-util — the shutter slider is dead", CORAL
        elif self.dial.took is False:
            shut, tone = f"shutter {self.dial.ms:.1f} ms — REFUSED", CORAL
        else:
            shut = f"shutter {self.dial.ms:.1f} ms ({self.dial.value})"
            tone = MINT if self.dial.took else DIM
        s.blit(self.f.render(shut, True, tone), (x, y))
        y += 17

        for b in self.buttons:
            b.draw(s, self.f)
        for sl in self.sliders:
            sl.draw(s, self.f)

        # Header positions are recorded by `_build` rather than derived here
        # from a widget index. Deriving them is how a panel silently
        # re-stacks itself the first time a row is added: the arithmetic stays
        # correct and stops describing the layout.
        section(s, self.fs, "vision", x, self.head_y["vision"], w,
                self.lab.color, CHALK)
        section(s, self.fs, "robot", x, self.head_y["robot"], w, self.link_note,
                MINT if "0/" not in self.link_note and self.lab.robots
                else GREY)
        # The colour each ball was told to wear, beside its row. This IS the
        # identity: the overlay names a cluster by the slot it is glowing, so
        # the two have to be checkable against each other by eye.
        for sx, sy, tone in self.swatches:
            pygame.draw.rect(s, tone, (sx, sy + 5, 14, 14), border_radius=3)
        for tx, ty, note, tone in getattr(self, "link_rows", []):
            s.blit(self.fs.render(note, True, tone), (tx, ty))
        if self.list_total > self.list_room:
            first = self.list_scroll + 1
            last = min(self.list_scroll + self.list_room, self.list_total)
            s.blit(self.fs.render(
                f"{first}-{last} of {self.list_total}   scroll to see the rest",
                True, DIM), (x, self.readout_top - 18))
            # A bar, so it is obvious there IS more rather than only stated.
            track = pygame.Rect(x + self.list_rect.w - 6, self.list_rect.y,
                                3, self.list_rect.h)
            pygame.draw.rect(s, RULE, track, border_radius=2)
            frac = self.list_room / max(self.list_total, 1)
            hh = max(18, int(track.h * frac))
            off = int((track.h - hh) * self.list_scroll
                      / max(1, self.list_total - self.list_room))
            pygame.draw.rect(s, GREY, (track.x, track.y + off, 3, hh),
                             border_radius=2)

        name = self.drive_target()
        if name is not None:
            aim = self.aim_error()
            if self.held is not None:
                line = (f"{name}  aim {self.aim:5.0f}  sent {self.held:5.0f}"
                        f"  byte {self.drive_byte}")
                tone = MINT
            else:
                line = (f"{name}: left/right yaw, up/down drive, tab switches"
                        f"   aim {self.aim:.0f}")
                tone = DIM
            s.blit(self.fs.render(line, True, tone),
                   (x, self.readout_top - 32))
            summ = self.aim_summary()
            if aim is not None:
                told, made, err = aim
                # The gap IS the ball's heading offset — the number `calib.py`
                # drives legs to measure, read off the screen instead.
                s.blit(self.fs.render(
                    f"told {told:.0f}  went {made:.0f}  aim offset "
                    f"{err:+.0f}deg", True, CHALK),
                    (x, self.readout_top - 18))
            elif summ is not None:
                line = (f"aim {summ['mean']:+.1f}deg  from {summ['n']} drives"
                        f"  scatter {summ['scatter']:.1f}  ")
                line += ("g to save" if summ["ok"] else summ["why"])
                s.blit(self.fs.render(fit(self.fs, line, w)[0], True,
                                      MINT if summ["ok"] else SUN),
                       (x, self.readout_top - 18))

        y = section(s, self.fs, "reading", x, self.readout_top, w)
        r = self.reading
        if r["theta_img"] is None:
            for ln in fit(self.f, r["why"] or "no reading", w):
                s.blit(self.f.render(ln, True, SUN), (x, y))
                y += 16
        else:
            rows = []
            if r["xy_cm"] is not None:
                rows += [("x", f"{r['xy_cm'][0]:.1f}", "cm"),
                         ("y", f"{r['xy_cm'][1]:.1f}", "cm")]
            else:
                rows += [("x", f"{r['centre_px'][0]:.0f}", "px"),
                         ("y", f"{r['centre_px'][1]:.0f}", "px")]
            th = r["theta_cm"]
            rows.append(("theta", f"{th:.1f}" if th is not None
                         else f"{r['theta_img']:.1f}",
                         "deg arena" if th is not None else "deg image"))
            # Both terms, always. `conf` alone cannot say whether to move the
            # camera or fix the colour, and those are the only two actions.
            rows.append(("conf", f"{r['conf']:.2f}", ""))
            if r.get("conf_span") is not None:
                span = r.get("span_px") or 0.0
                rows.append((" from span", f"{r['conf_span']:.2f}",
                             f"{span:.0f}px of {SPAN_FOR_FULL_CONF:.0f}"))
                how = ("tag" if r.get("by_tag") else
                       "blue" if r.get("by_colour") else "brightness")
                rows.append((" from ends", f"{r['conf_ends']:.2f}",
                             f"by {how}"))
            rows.append(("radius", f"{r['radius_px']:.1f}", "px"))
            # THE NUMBER THE RIG TURNS ON, and what it would have to become.
            # Every complaint that ends in "it cannot see it" is this times a
            # distance on the ball, so it is quoted next to the span it sets
            # rather than left to be worked out.
            scale = self.lab.px_per_cm(r["centre_px"])
            span = r.get("span_px") or 0.0
            if scale and span > 0:
                rows.append(("px per cm", f"{scale:.1f}", "at the ball"))
                span_cm = span / scale
                want = SPAN_FOR_FULL_CONF / max(span_cm, 1e-6)
                got = self.camera_size()
                covers = (got[0] / want) if got else None
                rows.append((" lights are", f"{span_cm:.1f}", "cm apart"))
                rows.append((" need", f"{want:.1f}", "px/cm for full conf"))
                if covers:
                    rows.append((" so frame", f"{covers:.0f}",
                                 f"cm wide, is {got[0] / scale:.0f}"))
            if r.get("centre_from"):
                pair = r["centre_from"] == "tag pair"
                rows.append(("centre", "tag pair" if pair else "1 LED",
                             "midpoint, tail out" if pair
                             else "biased forward"))
            for label, value, unit in rows:
                s.blit(self.f.render(label, True, DIM), (x, y))
                s.blit(self.fb.render(value, True, CHALK), (x + 90, y - 2))
                s.blit(self.fs.render(unit, True, GREY), (x + 190, y + 3))
                y += 20

        # Newest note first and stop at the bottom edge. Oldest-first runs off
        # the window exactly when there is most to say, which is the moment the
        # notes matter — and text drawn past the edge is not there at all.
        y += 8
        # Never `return` from here: the clip set at the top of this method has
        # to come off, or every later blit in the frame is confined to the
        # panel column and the camera view stops being drawn at all.
        for text, tone in reversed(list(self.notes)):
            for ln in fit(self.fs, text, w):
                if y > H - 16:
                    break
                s.blit(self.fs.render(ln, True, tone), (x, y))
                y += 13
            if y > H - 16:
                break
        s.set_clip(None)


def slot_rgb(sig, name, fallback=CHALK):
    """A colour signature's swatch, in pygame's byte order.

    `draw` is stored BGR because `vision/` hands it to cv2. Blitting it
    straight into pygame paints red as blue, and on a bench whose entire
    identity story is "which colour is that ball", a swatch that lies is worse
    than no swatch.
    """
    got = (sig.get(name) or {}).get("draw")
    return tuple(int(v) for v in reversed(got)) if got else fallback


def _wrap(text, n):
    out, line = [], ""
    for word in str(text).split():
        if len(line) + len(word) + 1 > n:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def fit(font, text, px):
    """Wrap to lines no wider than `px`, measured IN THE FONT.

    A character count is not a width. `SysFont` falls back to whatever the
    machine actually has, and on a fallback face — or at a different display
    scale — a "60 character" line is comfortably wider than the panel it was
    counted for. It then runs across the sliders and the log runs across the
    readout, which reads as the app being broken rather than as one number
    being wrong.

    A single word longer than `px` is broken mid-word rather than allowed to
    overhang, because the one thing this must never do is return a line that
    does not fit.
    """
    out, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and font.size(trial)[0] > px:
            out.append(line)
            line = word
        else:
            line = trial
        while font.size(line)[0] > px and len(line) > 1:
            cut = len(line)
            while cut > 1 and font.size(line[:cut])[0] > px:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
    if line:
        out.append(line)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, help="camera index")
    p.add_argument("--source", default=None,
                   help="'synthetic', a camera index, or a video file")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--color", default="red", choices=list(config.COLORS))
    p.add_argument("--uvc-index", type=int, default=0,
                   help="uvc-util's own index — not always the OpenCV one. "
                        "`python -m vision.shutter --list` shows them")
    a = p.parse_args(argv)

    source = a.source
    if source is None:
        source = str(a.camera) if a.camera is not None else "synthetic"
    App(source=source, camera_size=(a.width, a.height), color=a.color,
        uvc_index=a.uvc_index).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
