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

The `brightness` button swaps the view for a heat map of where the light
actually falls, warped flat through the homography so it covers THE ARENA and
not the room — a bright window behind the floor is not a lighting problem, and
averaged into a whole-frame number it hides one that is. Clipped pixels are
drawn in flat red, because the top of a heat ramp reads as "bright" and the one
thing that must not be mistaken for bright is "no hue and no peaks left".

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

from ui.theme import (CHALK, CORAL, DIM, GREY, INK, MINT, SUN, Button, Slider,
                      card, section)
from vision import config
from vision.detect import Detector
from vision.facing import (LIGHT_MIN_V, bearing_cm, cluster_px,
                           explain_cm, explain_px, heading_from_lights,
                           lights_px)
from vision.homography import Homography
from vision.synthetic import open_source

W, H = 1480, 940
VIEW = pygame.Rect(14, 48, 900, 506)          # 16:9, whatever the source is
LOG = pygame.Rect(14, 570, 900, 356)
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

        got, why = heading_from_lights(group)
        if got is None:
            out["why"] = why
            if len(group) == 1:
                out["centre_px"] = (group[0]["x"], group[0]["y"])
            return out

        out.update(theta_img=got["deg"], conf=got["conf"],
                   centre_px=got["centre"], front=got["front"],
                   back=got["back"], by_colour=got["by_colour"],
                   span_px=got["span_px"], straightness=got["straightness"],
                   radius_px=max(4.0, got["span_px"] / 2.0))

        if self.hom is not None and self.hom.ready:
            pts = self.hom.to_cm([list(got["centre"])])
            out["xy_cm"] = tuple(float(v) for v in np.asarray(pts).ravel()[:2])
            out["theta_cm"] = bearing_cm((got["back"]["x"], got["back"]["y"]),
                                         (got["front"]["x"], got["front"]["y"]),
                                         self.hom)
        return out

    def _blank(self):
        return {"color": self.color, "centre_px": None, "radius_px": None,
                "area": None, "theta_img": None, "theta_cm": None,
                "conf": None, "xy_cm": None, "why": None, "lights": [],
                "group": [], "front": None, "back": None, "by_colour": None,
                "span_px": None, "straightness": None}

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

    BLOWN = 250         # V at or above this has no colour left in it
    DARK = 30           # ...and below this there is nothing to key on

    def arena_view(self, frame, out_w, out_h):
        """The frame warped flat to the arena rectangle, top-down.

        Same composition the main app's floor view uses: the camera's
        pixel->cm matrix, then cm->local pixels. Restricting the lighting
        answer to the arena is the whole point — a bright window behind the
        floor is not a problem, and averaged into a whole-frame number it
        hides one that is.
        """
        if frame is None:
            return None, "no frame yet"
        if self.hom is None or not self.hom.ready:
            return None, "no arena calibration — this is the whole frame"
        scale = min(out_w / self.hom.width, out_h / self.hom.height)
        w, h = int(self.hom.width * scale), int(self.hom.height * scale)
        to_local = np.array([[scale, 0.0, 0.0],
                             [0.0, scale, 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            return cv2.warpPerspective(frame, to_local @ self.hom.M, (w, h)), None
        except Exception as e:
            return None, f"warp failed: {e}"

    def brightness_map(self, frame, out_w, out_h):
        """Where the light actually falls, over the floor the robots use.

        Value from HSV rather than a grey mix, because V is max(r,g,b) — which
        is precisely the channel that saturates. A shell reading 255 has no hue
        left for the detector and no separable peaks for a heading, so the
        number worth watching is not the average but the blown fraction.
        """
        warped, why = self.arena_view(frame, out_w, out_h)
        src = warped if warped is not None else frame
        if src is None:
            return None, None, why
        v = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)[:, :, 2]
        flat = v.reshape(-1)
        stats = {
            "mean": float(flat.mean()),
            "p05": float(np.percentile(flat, 5)),
            "p95": float(np.percentile(flat, 95)),
            "max": int(flat.max()),
            "blown": float((flat >= self.BLOWN).mean() * 100.0),
            "dark": float((flat <= self.DARK).mean() * 100.0),
        }
        img = cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)
        # Blown pixels in flat red, which is nowhere on the inferno ramp. The
        # ramp's own top end is a pale yellow that reads as "bright" — the one
        # thing that must not be mistaken for "bright" is "clipped".
        img[v >= self.BLOWN] = (0, 0, 255)
        return img, stats, why

    # -- the ball --------------------------------------------------------

    def connect(self, ble_name):
        """One ball, for its lights only. No tracker, no workspace, no roster.

        A roster row would have to be written to `roster.json`, which is live
        state the bench and the app both read — a lab that edits it changes
        what the next real session connects to. This constructs the handle
        directly instead, so nothing outside this process knows it happened.
        """
        from fleet.real_handle import SpheroRobot
        self.disconnect()
        self.robot = SpheroRobot(name="lab", code="LAB", color=self.color,
                                 ble_name=ble_name, tracker=None)
        return self.robot

    def disconnect(self):
        if self.robot is not None:
            try:
                self.robot.close()
            except Exception:
                pass
            self.robot = None

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
                 color="red"):
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
        self.view = "camera"            # or "bright"
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
        if self.lab.robot is not None:
            self.lab.disconnect()
            self.say("released")
            self._build()
            return
        try:
            self.lab.connect(ble_name)
        except Exception as e:
            self.say(f"connect failed: {e}", CORAL)
            return
        self.say(f"connecting to {ble_name} — lights follow once the link is up")
        self.push_led()
        self.push_tail()
        self._build()

    def push_led(self):
        if self.lab.robot is None:
            return
        self.lab.robot.set_led(config.led_rgb(self.hue, value=self.bright))

    def push_tail(self):
        if self.lab.robot is None:
            return
        self.lab.robot.set_back_led(int(self.tail))

    def toggle_mode(self):
        self.lab.mode = "blob" if self.lab.mode == "lights" else "lights"
        self.say("reading the LIGHTS directly — the colour tracker is not "
                 "consulted at all, it only says which end is the front"
                 if self.lab.mode == "lights" else
                 "reading inside a HUE BLOB — the heading is now only as good "
                 "as the colour tracking")
        self._build()

    def toggle_view(self):
        self.view = "bright" if self.view == "camera" else "camera"
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

        def sl(label, lo, hi, get, set_):
            nonlocal y
            self.sliders.append(Slider((x, y, w, 20), label, lo, hi, get, set_))
            y += 26

        # CAMERA
        y += 24 + 34                                    # header + two info lines
        btn((x, y, 92, 26), "manual", self.manual)
        btn((x + 100, y, 110, 26), "af off",
            lambda: self.cam_set("autofocus", 0))
        btn((x + 218, y, 110, 26), "af on",
            lambda: self.cam_set("autofocus", 1))
        btn((x + 336, y, 120, 26), "brightness", self.toggle_view,
            on=(self.view == "bright"))
        y += 34
        sl("focus", 0, 255, lambda: self.focus, self.set_focus)
        sl("exposure", 0, 255, lambda: self.exposure, self.set_exposure)
        y += 10

        # VISION
        self.head_y["vision"] = y
        y += 24
        bw = (w - 5 * 6) // 6
        for i, name in enumerate(self.COLORS):
            btn((x + i * (bw + 6), y, bw, 24), name[:3],
                (lambda n=name: self.set_color(n)),
                on=(name == self.lab.color))
        y += 64
        btn((x, y - 32, 120, 24), "lights", self.toggle_mode,
            on=(self.lab.mode == "lights"))
        sl("light floor", 60, 254, lambda: self.lab.min_v,
           lambda v: setattr(self.lab, "min_v", int(v)))
        sl("ball span", 10, 300, lambda: self.lab.span_px,
           lambda v: setattr(self.lab, "span_px", int(v)))
        sl("unfocus", 1, 31, lambda: self.lab.unfocus,
           lambda v: setattr(self.lab, "unfocus", v))
        sl("hue", 0, 179, lambda: self.hue, self.set_hue)
        sl("tol", 1, 40, lambda: self.lab.detector.colors[self.lab.color]["tol"],
           self.set_tol)
        sl("peak floor", 20, 95, lambda: self.lab.floor,
           lambda v: setattr(self.lab, "floor", int(v)))
        sl("s min", 0, 255, lambda: self.lab.detector.thresh["s_min"],
           lambda v: self.lab.detector.thresh.__setitem__("s_min", int(v)))
        sl("v min", 0, 255, lambda: self.lab.detector.thresh["v_min"],
           lambda v: self.lab.detector.thresh.__setitem__("v_min", int(v)))
        sl("min area", 10, 4000, lambda: self.lab.detector.thresh["min_area"],
           lambda v: self.lab.detector.thresh.__setitem__("min_area", int(v)))
        y += 10

        # ROBOT
        self.head_y["robot"] = y
        y += 24
        linked = self.lab.robot is not None
        btn((x, y, 92, 26), "scan", self.start_scan)
        if linked:
            btn((x + 100, y, 150, 26), "release",
                lambda: self.connect(None), tone=CORAL)
            up = self.lab.robot.link_up
            self.link_note = ("linked" if up else "connecting…")
        else:
            self.link_note = "no ball"
        y += 34
        for name in self.found[:4]:
            btn((x, y, 240, 24), name, (lambda n=name: self.connect(n)))
            y += 28
        sl("bright", 0, 255, lambda: self.bright, self.set_bright)
        sl("taillight", 0, 255, lambda: self.tail, self.set_tail)
        self.readout_top = y + 14

    # -- slider setters (they push to hardware, so they are not lambdas) --

    @property
    def focus(self):
        got = getattr(self.lab.source, "get", lambda _n: None)("focus")
        return int(got) if got is not None else 0

    @property
    def exposure(self):
        got = getattr(self.lab.source, "get", lambda _n: None)("exposure")
        return int(got) if got is not None else 0

    def set_focus(self, v):
        self.cam_set("focus", int(v))

    def set_exposure(self, v):
        self.cam_set("exposure", int(v))

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
        frame = self.grab.frame
        if not self.paused:
            self.reading = self.lab.analyse(frame)
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
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return self.close()
                    if e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        self.say("paused" if self.paused else "running")
                    if e.key == pygame.K_c:
                        self.lines.clear()
                if e.type == pygame.MOUSEBUTTONDOWN:
                    p = e.pos
                    if not any(b.hit(p) for b in self.buttons):
                        for s in self.sliders:
                            if s.hit(p):
                                break
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
            "heading from the two LEDs — space pauses, c clears the log, q quits",
            True, GREY), (130, 16))
        self.draw_view(s)
        self.draw_log(s)
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

    def draw_log(self, s):
        card(s, LOG)
        y = section(s, self.fs, "log", LOG.x + 10, LOG.y + 8, LOG.w - 20,
                    f"{LOG_HZ:.0f} hz", GREY)
        rows = (LOG.bottom - y - 8) // 15
        for stamp, ok, text in list(self.lines)[-rows:]:
            s.blit(self.fs.render(stamp, True, GREY), (LOG.x + 10, y))
            s.blit(self.f.render(text, True, CHALK if ok else SUN),
                   (LOG.x + 96, y - 2))
            y += 15

    def draw_panel(self, s):
        x, w = PANEL_X, PANEL_W
        y = 62

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
                MINT if self.link_note == "linked" else GREY)

        y = section(s, self.fs, "reading", x, self.readout_top, w)
        r = self.reading
        if r["theta_img"] is None:
            for ln in _wrap(r["why"] or "no reading", 52):
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
            rows += [("conf", f"{r['conf']:.2f}", ""),
                     ("radius", f"{r['radius_px']:.1f}", "px")]
            for label, value, unit in rows:
                s.blit(self.f.render(label, True, DIM), (x, y))
                s.blit(self.fb.render(value, True, CHALK), (x + 90, y - 2))
                s.blit(self.fs.render(unit, True, GREY), (x + 190, y + 3))
                y += 20

        # Newest note first and stop at the bottom edge. Oldest-first runs off
        # the window exactly when there is most to say, which is the moment the
        # notes matter — and text drawn past the edge is not there at all.
        y += 8
        for text, tone in reversed(list(self.notes)):
            for ln in _wrap(text, 60):
                if y > H - 16:
                    return
                s.blit(self.fs.render(ln, True, tone), (x, y))
                y += 13


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, help="camera index")
    p.add_argument("--source", default=None,
                   help="'synthetic', a camera index, or a video file")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--color", default="red", choices=list(config.COLORS))
    a = p.parse_args(argv)

    source = a.source
    if source is None:
        source = str(a.camera) if a.camera is not None else "synthetic"
    App(source=source, camera_size=(a.width, a.height), color=a.color).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
