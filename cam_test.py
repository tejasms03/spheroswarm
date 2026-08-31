#!/usr/bin/env python3
"""Point a camera at the floor and see whether the lights method works.

    python cam_test.py --camera 0
    python cam_test.py --camera 0 --exposure -7
    python cam_test.py --camera 0 --size 1920x1080
    python cam_test.py --source sim                # no hardware
    python cam_test.py --source clip.mov

Three questions, on one screen, live:

    CAN IT SEE THEM      every lit ball, ringed, with its size in pixels
    WHICH ONE IS IT      the tag colour read AT the LEDs, not off the shell
    WHICH WAY IS IT      an arrow, from the dots, with blue deciding the end

The exposure slider is the only setting that really matters and it is the first
thing to move. Too long and the LED cores blow to white -- no hue to identify
with, no separable peaks to take a heading from. Too short and there is nothing
above the noise. Somewhere between, all three questions answer at once, and
finding that window on YOUR camera is what this app is for.

It refuses out loud. A ball with no reading carries the reason -- "one dot",
"no blue at any dot", "too dark to read" -- because a bench that goes quiet
when it fails leaves you turning knobs at random.

Everything is measured INSIDE the arena and nowhere else. The frame is warped
flat through the homography first, so shelving, a lit room and a bright window
behind the floor cannot drag a brightness decision or offer up a reflection to
mistake for a robot. Press `a` for the whole frame if you want to see what is
being thrown away.

MOVED THE CAMERA? Press `c` and click the four arena corners -- origin, +x,
+x+y, +y. A homography calibrated at the old position warps the wrong part of
the frame and reports centimetres that are fiction, and nothing in the picture
will tell you. `w` writes the new one to calib/homography.json, which is live
state, so it only happens when you ask.

LOCKING THE EXPOSURE. OpenCV cannot set this camera's exposure -- the property
is accepted and dropped. AVFoundation can freeze it, which is enough: point the
camera at something BRIGHT so its own metering settles short, press `l`, then
point it back at the floor. It holds the short exposure instead of amplifying
until the mean is back where it likes it.

Keys — l lock/unlock exposure, c pick corners, w write them, v raw/gate,
a arena/whole frame,
m manual (autofocus + auto-exposure off), s save the frame,
[ ] exposure down/up, backspace undo a corner, esc quit.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from ui.theme import (CHALK, CORAL, CYAN, DIM, GAP, GREY, INK, MINT, PAD,
                      PANEL, RULE, SUN, Button, Slider, card, section)
from vision.dots import (explain_dots_px, explain_hue_px, find_balls,
                         gate_view, identify)
from vision.lighting import arena_scale, arena_view

W, H = 1360, 860
MIN_W, MIN_H = 1060, 660
DOCK = 330
BALL_CM = 7.4
ROOT = Path(__file__).resolve().parent


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


class Camera(threading.Thread):
    """Frames on their own thread. A blocking read on the render thread is a
    frozen window, and a frozen window during bring-up reads as a crash."""

    def __init__(self, spec, size=None):
        super().__init__(daemon=True, name="cam")
        self.source = None
        failed = None
        try:
            if str(spec) in ("sim", "balls"):
                self.source = LedSource()
            else:
                from vision.synthetic import open_source
                self.source = open_source(spec, size=size)
        except Exception as e:
            # A camera that will not open must not cost you the window. On
            # macOS the usual cause is that this process has no camera
            # permission, or that another app already holds the device, and a
            # traceback in a terminal is a worse way to learn either than a
            # line in the app saying so.
            failed = f"{type(e).__name__}: {e}"
        self.frame = None
        self.count = 0
        self.error = failed
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
        try:
            if self.source is not None:
                self.source.release()
        except Exception:
            pass


class LedSource:
    """A fake camera showing balls with the REAL light arrangement.

    `vision.synthetic.SyntheticSource` predates any of this and draws one
    coloured disc per robot, which is right for testing the colour tracker and
    useless here -- there are no dots in it to read a heading from. This draws
    what `vision/shots.py` draws, so the app can be exercised, and its refusals
    trusted, before a camera exists.
    """

    def __init__(self, n=4, size=(1280, 720), px_cm=6.5, seed=0):
        from vision.shots import TAGS
        self.w, self.h = size
        self.px_cm = px_cm
        self.rng = np.random.default_rng(seed)
        self.tags = (list(TAGS) * 3)[:n]
        w_cm, h_cm = self.w / px_cm, self.h / px_cm
        self.pos = np.stack([self.rng.uniform(12, w_cm - 12, n),
                             self.rng.uniform(12, h_cm - 12, n)], axis=1)
        self.heading = self.rng.uniform(0, 360, n)
        self.turn = self.rng.uniform(-25, 25, n)
        self.bounds = (w_cm, h_cm)
        self.t = 0.0

    def read(self):
        from vision.shots import as_short, draw_ball
        dt = 1 / 30.0
        self.t += dt
        self.heading = (self.heading + self.turn * dt) % 360.0
        step = np.stack([np.cos(np.radians(self.heading)),
                         np.sin(np.radians(self.heading))], axis=1) * 6.0 * dt
        self.pos += step
        for ax, hi in enumerate(self.bounds):
            out = (self.pos[:, ax] < 10) | (self.pos[:, ax] > hi - 10)
            self.heading[out] = (self.heading[out] + 150.0) % 360.0
            self.pos[:, ax] = np.clip(self.pos[:, ax], 10, hi - 10)

        canvas = np.zeros((self.h, self.w, 3), np.float32)
        canvas[:] = (6.0, 5.0, 4.0)
        for i, tag in enumerate(self.tags):
            draw_ball(canvas, self.pos[i, 0], self.pos[i, 1],
                      float(self.heading[i]), tag, self.px_cm)
        canvas += self.rng.normal(0.0, 1.6, canvas.shape).astype(np.float32)
        return True, as_short(canvas)

    def release(self):
        pass


class CamTest:
    def __init__(self, spec="0", exposure=None, px_cm=None, size=None):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("cam test — can the camera read the lights?")
        fit_to_display()
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15, bold=True)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)

        self.cam = Camera(spec, size=size)
        self.cam.start()
        self.view = "raw"
        # Everything outside the arena is furniture, shelving and a lit room,
        # and none of it is a robot. Warping flat to the arena first is not
        # cosmetic: a blown-out window behind the floor drags every brightness
        # decision made from a whole-frame statistic, and a reflection off a
        # box edge is a blob the detector has to be talked out of. Restricting
        # first means nothing has to be talked out of anything.
        self.arena_only = True
        self.hom = self._homography()
        # Corners picked here rather than in `vision.app`, because during
        # bring-up the camera moves -- and the moment it does, a homography
        # calibrated at the old position warps the wrong region of the frame
        # and every centimetre it reports is fiction. Re-picking has to be four
        # clicks in the app you are already looking at, or it will not happen.
        self.exposure_locked = False
        self.picking = None             # list of frame points while picking
        self._shot = None               # (offset, scale) of the last draw
        self.frame = None
        self.raw = None
        self.cropped = False
        self.rows = []
        self.note = f"opening {spec}"

        self.v_min = 60             # brightness a pixel needs to count as lit
        self.gate_lo, self.gate_hi = 40, 200
        self.exposure = exposure if exposure is not None else -7
        self.gain, self.brightness = 0, 128
        self.took = {}                  # what the camera reported back, per knob
        self.px_cm = px_cm or self._px_cm_from_homography()

        self.sliders, self.buttons = [], []
        self.dragging = None
        self.apply_size(*self.screen.get_size())
        if exposure is not None:
            self.manual()

    # -- scale -----------------------------------------------------------

    def start_picking(self):
        self.picking = []
        self.arena_only = False         # you cannot click corners on a crop
        self.note = "click the arena corners: origin, +x, +x+y, +y"

    def add_corner(self, screen_pos):
        """One click, converted back into frame pixels."""
        if self.picking is None or self._shot is None or self.frame is None:
            return
        (ox, oy), k = self._shot
        x, y = (screen_pos[0] - ox) / k, (screen_pos[1] - oy) / k
        h, w = self.raw.shape[:2] if self.raw is not None else self.frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.picking.append((float(x), float(y)))
        self.note = f"{len(self.picking)}/4 corners"
        if len(self.picking) == 4:
            self.build_homography()

    def build_homography(self):
        """Four corners into a homography, sized from the workspace in cm."""
        from vision.homography import Homography
        from workspace.space import Workspace
        try:
            ws = Workspace.load()
            hom = Homography().set_rect(self.picking, ws.width, ws.height)
        except Exception as e:
            self.note = f"could not build a homography: {e}"
            self.picking = None
            return
        self.hom = hom
        self.picking = None
        self.arena_only = True
        size = self.arena_size()
        self.px_cm = arena_scale(hom, *size) if size else None
        self.note = (f"arena set — {ws.width:.0f}x{ws.height:.0f}cm at "
                     f"{self.px_cm:.2f}px/cm, a ball is {7.4 * self.px_cm:.0f}px"
                     if self.px_cm else "arena set")

    def save_homography(self):
        """Write it to calib/homography.json. Never automatic.

        That file is live state the whole project reads -- the tracker, the
        planner and the bench all take their centimetres from it -- so it is
        overwritten only when somebody asks, and the old one is worth keeping
        until the new one has driven a robot.
        """
        if self.hom is None or not self.hom.ready:
            self.note = "no arena to save"
            return
        try:
            self.hom.save()
            self.note = "saved to calib/homography.json — re-check your colours"
            print("cam_test: wrote calib/homography.json")
        except Exception as e:
            self.note = f"could not save: {e}"

    def _homography(self):
        try:
            from vision.homography import Homography
            h = Homography.load()
            return h if h.ready else None
        except Exception as e:
            print(f"cam_test: could not load the homography: {e}")
            return None

    def arena_size(self):
        """The arena warped at its NATIVE pixel density, not stretched to fit.

        Warping to the window's size would upsample -- the readout would claim
        a bigger ball in more pixels per centimetre than the camera ever
        delivered, which is exactly the number being judged here.
        """
        native = self._px_cm_from_homography()
        if not native or self.hom is None:
            return None
        return (int(self.hom.width * native), int(self.hom.height * native))

    def _px_cm_from_homography(self):
        """How many pixels a centimetre is, if this arena has been calibrated.

        Only used to say how big a ball SHOULD be, which is the first thing to
        check when nothing is being detected. No homography just means that
        line reads unknown; nothing else depends on it.
        """
        try:
            from vision.homography import Homography
            h = Homography.load()
            if not h.ready:
                return None
            inv = np.linalg.inv(np.asarray(h.M, dtype=float))
            corners = np.array([[[0, 0]], [[h.width, 0]],
                                [[h.width, h.height]], [[0, h.height]]], np.float32)
            px = cv2.perspectiveTransform(corners, inv).reshape(-1, 2)
            across = (np.linalg.norm(px[1] - px[0]) + np.linalg.norm(px[2] - px[3])) / 2
            down = (np.linalg.norm(px[3] - px[0]) + np.linalg.norm(px[2] - px[1])) / 2
            return float((across / h.width + down / h.height) / 2)
        except Exception as e:
            # Loud, not silent. A bare pass here hid an attribute typo for long
            # enough that the scale line read "unknown" on an arena that had
            # been calibrated all along.
            print(f"cam_test: could not read the homography: {e}")
            return None

    # -- camera ----------------------------------------------------------

    def manual(self):
        """Autofocus and auto-exposure off, then apply the slider."""
        src = self.cam.source
        if src is None or not hasattr(src, "manual"):
            self.note = "this source has no manual controls (synthetic or video)"
            return
        got = src.manual()
        self.set_exposure(self.exposure)
        self.note = f"manual: {got}"

    def exposure_mode(self, lock):
        """Freeze the camera's own auto-exposure, or hand it back.

        The only exposure control this camera actually has. It will not take a
        shutter or an ISO -- `Custom` is unsupported -- but it will hold
        whatever its own metering last settled on, and that is the lever that
        matters: an unlocked camera AMPLIFIES when you darken the room, which
        blows the LED cores into one blob and is the opposite of what this
        method needs.
        """
        try:
            from vision import avcam
        except Exception as e:
            self.note = f"no AVFoundation control: {e}"
            return
        ok, msg = (avcam.lock() if lock else avcam.auto())
        self.exposure_locked = lock and ok
        self.note = (msg + ("  — now darken the room" if lock and ok else ""))
        print(f"cam_test: {msg}")

    def cam_set(self, name, value):
        """Ask for a property and record what the camera says it did.

        Recorded rather than assumed, because a `set` that is silently dropped
        and one that works look identical from here -- and on this backend the
        silent drop is the common case. The dock shows the readback beside each
        slider, so a knob that does nothing says so instead of leaving you
        turning it harder.
        """
        setattr(self, name, int(value))
        src = self.cam.source
        if src is None or not hasattr(src, "set"):
            return
        self.took[name] = src.set(name, int(value))

    def set_exposure(self, value):
        self.exposure = int(value)
        src = self.cam.source
        if src is None or not hasattr(src, "set"):
            return
        took = src.set("exposure", self.exposure)
        self.took["exposure"] = took
        self.note = (f"exposure {self.exposure} -> camera reports {took}"
                     if took is not None else
                     f"exposure {self.exposure}: the camera ignored it")

    def save(self):
        """Write the current frame out, for measuring off-line."""
        if self.frame is None:
            return
        out = ROOT / "runs"
        out.mkdir(exist_ok=True)
        name = out / f"cam-{time.strftime('%Y%m%d-%H%M%S')}-exp{self.exposure}.png"
        cv2.imwrite(str(name), getattr(self, "raw", self.frame))
        if getattr(self, "cropped", False):
            cv2.imwrite(str(name).replace(".png", "-arena.png"), self.frame)
        self.note = f"saved {name.name}"
        print(f"saved {name}")

    # -- analysis --------------------------------------------------------

    def analyse(self):
        """Always on the REAL capture, never on the gate rendering.

        The gate is false colour and false colour has no hue, so anything that
        reads a colour off it is reading the colourmap rather than the robot.
        """
        img = self.cam.latest()
        if img is None:
            return
        self.raw = img
        self.cropped = False
        if self.arena_only and self.hom is not None:
            size = self.arena_size()
            if size:
                warped, why = arena_view(img, self.hom, size[0], size[1])
                if warped is not None:
                    img, self.cropped = warped, True
                    self.px_cm = arena_scale(self.hom, size[0], size[1])
                elif why:
                    self.note = why
        self.frame = img
        ball_px = (BALL_CM * self.px_cm) if self.px_cm else None
        kw = {}
        if ball_px:
            kw = {"merge_px": max(6.0, ball_px * 1.2),
                  "sparse_min_area": max(1.0, (ball_px / 48.0) ** 2 * 24.0)}
        rows = []
        for centre, radius, area in find_balls(img, v_min=int(self.v_min), **kw):
            dots, dots_why = explain_dots_px(img, centre, radius)
            hue, hue_why = explain_hue_px(img, centre, radius)
            who = None
            if dots:
                who = identify(img, dots,
                               radius=max(1.0, (ball_px or 40) * 0.06))
            rows.append({"centre": centre, "radius": radius, "area": area,
                         "dots": dots, "dots_why": dots_why,
                         "hue": hue, "hue_why": hue_why, "who": who})
        self.rows = rows

    # -- layout ----------------------------------------------------------

    def apply_size(self, w, h):
        global W, H
        W, H = max(w, MIN_W), max(h, MIN_H)
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.view_rect = pygame.Rect(DOCK + PAD, PAD, W - DOCK - 2 * PAD,
                                     H - 2 * PAD - 20)
        self.build_dock()

    def build_dock(self):
        self.sliders, self.buttons = [], []
        x, w = PAD, DOCK - 2 * PAD
        self.sliders.append(Slider((x, 30, w, 18), "exposure", -13, 0,
                                   lambda: int(self.exposure), self.set_exposure))
        self.sliders.append(Slider((x, 54, w, 18), "lit above", 10, 200,
                                   lambda: int(self.v_min),
                                   lambda v: setattr(self, "v_min", v)))
        # Exposure is the one that matters and the one most likely to be
        # ignored. Gain and brightness are here because a camera that refuses
        # one property will sometimes honour another, and the only way to find
        # out is to move it and read it back.
        self.sliders.append(Slider((x, 78, w, 18), "gain", 0, 255,
                                   lambda: int(self.gain),
                                   lambda v: self.cam_set("gain", v)))
        self.sliders.append(Slider((x, 102, w, 18), "brightness", 0, 255,
                                   lambda: int(self.brightness),
                                   lambda v: self.cam_set("brightness", v)))
        self.sliders.append(Slider((x, 126, w, 18), "gate floor", 0, 250,
                                   lambda: int(self.gate_lo),
                                   lambda v: setattr(self, "gate_lo", v)))
        bw = (w - 2 * 6) // 3
        by = H - PAD - 36
        for i, (label, cb) in enumerate([("raw/gate", self.toggle_view),
                                         ("arena", lambda: self.key(
                                             type("K", (), {"key": pygame.K_a})())),
                                         ("save", self.save)]):
            self.buttons.append(Button((x + i * (bw + 6), by, bw, 26), label, cb))

    def toggle_view(self):
        self.view = "gate" if self.view == "raw" else "raw"

    # -- drawing ---------------------------------------------------------

    def text(self, s, x, y, col=CHALK, font=None):
        self.screen.blit((font or self.f).render(str(s), True, col), (x, y))

    def to_screen(self, p, shown, k):
        ox = self.view_rect.x + (self.view_rect.w - shown[0]) // 2
        oy = self.view_rect.y + (self.view_rect.h - shown[1]) // 2
        return (int(ox + p[0] * k), int(oy + p[1] * k))

    def draw_view(self):
        pygame.draw.rect(self.screen, (9, 22, 36), self.view_rect)
        pygame.draw.rect(self.screen, RULE, self.view_rect, 1)
        if self.frame is None:
            self.text(self.cam.error or "waiting for the first frame",
                      self.view_rect.x + 12, self.view_rect.centery, DIM)
            return
        img = (gate_view(self.frame, self.gate_lo, self.gate_hi)
               if self.view == "gate" else self.frame)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        surf = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")
        k = min(self.view_rect.w / rgb.shape[1], self.view_rect.h / rgb.shape[0], 1.0)
        shown = (int(rgb.shape[1] * k), int(rgb.shape[0] * k))
        origin = self.to_screen((0, 0), shown, k)
        self._shot = (origin, k)
        self.screen.blit(pygame.transform.scale(surf, shown), origin)

        if self.picking is not None:
            pts = [self.to_screen(q, shown, k) for q in self.picking]
            for j, q in enumerate(pts):
                pygame.draw.circle(self.screen, SUN, q, 5, 2)
                self.text("origin +x +x+y +y".split()[j], q[0] + 8, q[1] - 6,
                          SUN, self.fs)
            if len(pts) > 1:
                pygame.draw.lines(self.screen, SUN, False, pts, 1)
            return

        for i, r in enumerate(self.rows):
            at = self.to_screen(r["centre"], shown, k)
            rad = max(5, int(r["radius"] * k))
            got = r["dots"] or r["hue"]
            pygame.draw.circle(self.screen, (70, 110, 150), at, rad, 1)
            if r["dots"]:
                # The dots it actually used, so a wrong reading shows you WHY.
                for t in r["dots"]["tags"]:
                    pygame.draw.circle(self.screen, CYAN,
                                       self.to_screen(t, shown, k), 3, 1)
                pygame.draw.circle(self.screen, (90, 150, 255),
                                   self.to_screen(r["dots"]["tail"], shown, k), 3)
            if got:
                deg = r["dots"]["deg"] if r["dots"] else got[0]
                a = np.radians(deg)
                tip = (at[0] + np.cos(a) * rad * 2.0, at[1] + np.sin(a) * rad * 2.0)
                pygame.draw.line(self.screen, SUN, at, tip, 2)
                label = f"{i + 1}"
                if r["who"]:
                    label += f"  {r['who'][0]}"
                label += f"  {deg:.0f}deg"
                self.text(label, at[0] + rad + 5, at[1] - 8, CHALK, self.fs)
            else:
                self.text(f"{i + 1}  {(r['dots_why'] or '')[:30]}",
                          at[0] + rad + 5, at[1] - 6, CORAL, self.fs)

    def draw_dock(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, DOCK, H))
        pygame.draw.line(self.screen, RULE, (DOCK, 0), (DOCK, H))
        x, w = PAD, DOCK - 2 * PAD
        for s in self.sliders:
            s.draw(self.screen, self.f)

        y = 156
        y = section(self.screen, self.fs, "camera", x, y, w,
                    f"{self.cam.fps:.0f}fps", DIM)
        if self.frame is not None:
            h, wid = self.frame.shape[:2]
            # What ARRIVED, not what was asked for. A webcam offered a mode it
            # does not have picks its nearest and reports success, so the only
            # way to know you got 1080p is to look at a frame.
            want = getattr(self.cam.source, "wanted_size", None)
            got = f"{wid} x {h}"
            if want and tuple(want) != (wid, h):
                got += f"  (asked {want[0]}x{want[1]})"
            got += "  ARENA" if getattr(self, "cropped", False) else "  full frame"
            self.text(f"{got}   view {self.view}", x, y,
                      SUN if want and tuple(want) != (wid, h) else CHALK, self.fs)
        y += 14
        if self.px_cm:
            ball = BALL_CM * self.px_cm
            tone = MINT if ball >= 22 else SUN
            self.text(f"{self.px_cm:.2f} px/cm — a ball is {ball:.0f}px",
                      x, y, tone, self.fs)
            y += 13
            self.text("under 22px this stops reading" if ball < 30 else
                      "comfortably inside the readable range", x, y, DIM, self.fs)
        else:
            self.text("no homography — scale unknown", x, y, SUN, self.fs)
        y += 8
        # Which knobs this camera actually honours.
        self.text("exposure LOCKED" if self.exposure_locked else
                  "exposure on auto — it will fight you",
                  x, y, MINT if self.exposure_locked else SUN, self.fs)
        y += 14
        dead = [k for k, v in self.took.items() if v is None]
        if self.took:
            self.text("ignored: " + (", ".join(dead) if dead else "none"),
                      x, y, SUN if dead else MINT, self.fs)
            y += 13
            if dead:
                self.text("press l to lock exposure (AVFoundation)",
                          x, y, DIM, self.fs)
                y += 13
        y += 10

        y = section(self.screen, self.fs, "balls", x, y, w,
                    f"{len(self.rows)} found", DIM)
        for i, r in enumerate(self.rows):
            if y > H - 70:
                self.text(f"... {len(self.rows) - i} more", x, y, DIM, self.fs)
                break
            card(self.screen, pygame.Rect(x, y, w, 44))
            self.text(f"{i + 1}", x + 6, y + 4, CHALK, self.fb)
            self.text(f"{2 * r['radius']:.0f}px wide", x + 26, y + 5, DIM, self.fs)
            if r["dots"]:
                who = r["who"][0] if r["who"] else "?"
                self.text(f"{who}  {r['dots']['deg']:5.1f}deg  "
                          f"c{r['dots']['conf']:.2f}", x + 26, y + 18, MINT, self.fs)
                self.text("from the dots", x + 26, y + 30, DIM, self.fs)
            elif r["hue"]:
                self.text(f"{r['hue'][0]:5.1f}deg  from colour only",
                          x + 26, y + 18, CYAN, self.fs)
                self.text((r["dots_why"] or "")[:34], x + 26, y + 30, DIM, self.fs)
            else:
                self.text((r["dots_why"] or "no reading")[:34],
                          x + 26, y + 18, CORAL, self.fs)
                self.text((r["hue_why"] or "")[:34], x + 26, y + 30, DIM, self.fs)
            y += 50

        for b in self.buttons:
            b.on = (b.label == "raw/gate" and self.view == "gate")
            b.draw(self.screen, self.f)
        self.text((self.cam.error or self.note)[:46], PAD, H - PAD - 12,
                  CORAL if self.cam.error else DIM, self.fs)

    # -- loop ------------------------------------------------------------

    def key(self, e):
        if e.key == pygame.K_ESCAPE:
            return False
        elif e.key == pygame.K_v:
            self.toggle_view()
        elif e.key == pygame.K_m:
            self.manual()
        elif e.key == pygame.K_l:
            self.exposure_mode(not getattr(self, "exposure_locked", False))
        elif e.key == pygame.K_c:
            self.start_picking()
        elif e.key == pygame.K_BACKSPACE and self.picking:
            self.picking.pop()
            self.note = f"{len(self.picking)}/4 corners"
        elif e.key == pygame.K_w:
            self.save_homography()
        elif e.key == pygame.K_a:
            self.arena_only = not self.arena_only
            self.px_cm = (arena_scale(self.hom, *self.arena_size())
                          if self.arena_only and self.hom and self.arena_size()
                          else self._px_cm_from_homography())
            self.note = ("arena only — everything outside is discarded"
                         if self.arena_only else "whole frame, warts and all")
        elif e.key == pygame.K_s:
            self.save()
        elif e.key == pygame.K_LEFTBRACKET:
            self.set_exposure(self.exposure - 1)
        elif e.key == pygame.K_RIGHTBRACKET:
            self.set_exposure(self.exposure + 1)
        return True

    def run_loop(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if self.picking is not None and self.view_rect.collidepoint(e.pos):
                        self.add_corner(e.pos)
                        continue
                    for s in self.sliders:
                        if s.hit(e.pos):
                            self.dragging = s
                            break
                    else:
                        for b in self.buttons:
                            if b.hit(e.pos):
                                break
                elif e.type == pygame.MOUSEMOTION and self.dragging:
                    self.dragging.drag(e.pos)
                elif e.type == pygame.MOUSEBUTTONUP:
                    self.dragging = None
                elif e.type == pygame.KEYDOWN:
                    running = self.key(e)
                elif e.type == pygame.VIDEORESIZE:
                    self.apply_size(e.w, e.h)

            self.analyse()
            self.screen.fill(INK)
            self.draw_view()
            self.draw_dock()
            pygame.display.flip()
            self.clock.tick(30)
        self.cam.close()
        pygame.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default=None, help="camera index, e.g. 0")
    p.add_argument("--source", default=None,
                   help="'sim' (fake balls with real lights), 'synthetic', "
                        "a video path, or a camera index")
    p.add_argument("--exposure", type=int, default=None,
                   help="apply manual exposure at startup, e.g. -7")
    p.add_argument("--size", default=None,
                   help="request a capture size, e.g. 1920x1080. What actually "
                        "arrives is shown in the dock — cameras substitute.")
    p.add_argument("--px-cm", type=float, default=None,
                   help="override the scale instead of reading the homography")
    a = p.parse_args()
    spec = a.source or a.camera or "0"
    size = None
    if a.size:
        w, h = (int(v) for v in a.size.lower().split("x"))
        size = (w, h)
    CamTest(spec, exposure=a.exposure, px_cm=a.px_cm, size=size).run_loop()


if __name__ == "__main__":
    main()
