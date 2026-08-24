#!/usr/bin/env python3
"""Calibration bench — get one Sphero seen, aimed and measured.

    python calib.py                    # synthetic camera, nothing to plug in
    python calib.py --camera 0         # the overhead camera
    python calib.py --camera 0 --dry   # add sim robots, exercise the rig cold

Three jobs, in the order a bring-up session actually needs them:

  SEE     tune each colour signature until the tracker finds the ball. One
          button per colour, or one button for all six: the ball's own LED is
          driven off and on and the signature is learned from the difference,
          so the answer does not depend on anyone reading a hue histogram.
          Saved to `calib/colors.json`, which every Detector loads.

  AIM     the heading offset — the constant between the ball's aim frame and
          the camera's. Written back to `roster.json`.

  MEASURE the five numbers `fleet/sim_handle.py` currently guesses: position
          noise, the speed byte to cm/s map, the motor lag, the coast, and the
          loop delay. Saved to `calib/motion.json`.

The dock lists every robot in the roster. `real` connects one over Bluetooth;
`sim` drops in a simulated stand-in, which runs the whole battery with no
hardware present and is the way to check the rig before a session rather than
during one.
"""

import argparse
import json
import math
import threading
import time

import numpy as np
import pygame

from fleet import safety
import fleet.characterize as characterize
from fleet.heading import wrap180
from fleet.characterize import (Characterization, DriftWatch, Recenter,
                                load_motion)
from swarm.pd import (Circle, Line, PDController, Point, TurnAndGo,
                      gains_from_motion,
                      implied_heading_error, straightness, tracking_error)
from fleet.manager import Fleet
from fleet.roster import RobotEntry, Roster
from ui.theme import (CARD, CHALK, CORAL, CYAN, DIM, GAP, GREY, INK, LED_RGB,
                      MINT, PAD, PANEL, PANEL2, RULE, SUN, Button, Slider, card,
                      section, stat_tile)
from vision import config as vconfig
from vision import palette as vpalette
from workspace.space import Workspace

W, H = 1500, 950
DOCK = 400
MIN_W, MIN_H = 1150, 760
LOG_HISTORY = 300
TRAIL_LEN = 240         # ~8s of path at 30fps
SPEED_CAP_CM_S = 18     # nothing the bench drives goes faster than this


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


class BleScan(threading.Thread):
    """Discover Spheros without blocking the window.

    Scanning takes seconds and a frozen window during it reads as a crash, so
    it happens here and the result is picked up by the render loop.
    """

    def __init__(self, timeout=7.0):
        super().__init__(daemon=True, name="ble-scan")
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
                self.error = ("no Spheros answered. Wake one by shaking it, and "
                              "check it is not still paired to a phone")
        except Exception as e:
            self.error = str(e)
        finally:
            self.finished = True


class AutoTune:
    """Learn one colour signature from the ball's own LED.

    LED off, a burst of frames; LED on, another burst; whatever changed is the
    ball. Doing it this way rather than with sliders means the answer does not
    depend on the floor, the bulbs overhead, or the trainer's eye — and it is
    the only version that can be run for six colours in a row without anyone
    watching.
    """

    SETTLE = 0.7            # s for the LED change to reach the camera
    BURST = 6               # frames per burst

    def __init__(self, handle, colors, on_done, light=None):
        """`light(name) -> rgb` says what to glow for a colour slot.

        Passed in rather than looked up here, and that is the whole point. This
        used to light the ball from a fixed nominal table while the bench lit
        it from the hue actually being hunted — so a slot re-picked by
        `optimise hues` was tuned against the colour it USED to be, and the
        learned signature quietly put it back. The ball then glowed one colour
        while being hunted as another, which is a fault this codebase has
        already had once and did not need again.
        """
        self.handle = handle
        self.colors = list(colors)
        self.on_done = on_done
        self.light = light or (lambda name: LED_RGB.get(name, (255, 255, 255)))
        self.i = 0
        self.phase = "dark"
        self.t = 0.0
        self.off, self.on = [], []
        self.last_stamp = None
        self.done = False
        self.results = {}

    @property
    def color(self):
        return self.colors[self.i] if self.i < len(self.colors) else None

    @property
    def progress(self):
        if self.done:
            return "done"
        return f"{self.color}  ({self.i + 1}/{len(self.colors)})"

    def _grab(self, frame, bucket):
        """One frame per distinct camera frame, never the same one twice."""
        stamp = id(frame)
        if stamp == self.last_stamp:
            return False
        self.last_stamp = stamp
        bucket.append(frame.copy())
        return True

    def step(self, frame, dt):
        if self.done:
            return
        self.t += dt
        color = self.color
        if color is None:
            self.done = True
            self.on_done(self.results)
            return

        if self.phase == "dark":
            self.handle.set_led((0, 0, 0))
            if self.t >= self.SETTLE:
                self.phase, self.t, self.off = "off_burst", 0.0, []
            return

        if self.phase == "off_burst":
            if frame is not None:
                self._grab(frame, self.off)
            if len(self.off) >= self.BURST:
                self.handle.set_led(self.light(color))
                self.phase, self.t = "lit", 0.0
            return

        if self.phase == "lit":
            if self.t >= self.SETTLE:
                self.phase, self.t, self.on = "on_burst", 0.0, []
            return

        if self.phase == "on_burst":
            if frame is not None:
                self._grab(frame, self.on)
            if len(self.on) >= self.BURST:
                self._learn(color)
                self.i += 1
                self.phase, self.t = "dark", 0.0
                if self.i >= len(self.colors):
                    self.handle.set_led(self.light(self.colors[-1]))
                    self.done = True
                    self.on_done(self.results)
            return

    def _learn(self, color):
        from vision.detect import autotune_signature
        try:
            self.results[color] = autotune_signature(self.off, self.on)
        except Exception as e:
            self.results[color] = {"ok": False, "error": str(e)}


class SensorProbe(threading.Thread):
    """Ask the ball what it can report, without freezing the window.

    The probe is 150 blocking radio reads plus sixty drive writes across three
    streaming rates. Every one of those waits on an acknowledgement that takes
    about 230ms, so the whole thing is tens of seconds — and run on the render
    thread that is tens of seconds of a dead window, which reads as a crash and
    gets the app force-quit halfway through.

    Same shape as `BleScan`: do it here, pick the result up in the loop.
    """

    def __init__(self, api):
        super().__init__(daemon=True, name="sensor-probe")
        self.api = api
        self.report = None
        self.error = None
        self.finished = False

    def run(self):
        try:
            from fleet.sensors import probe
            self.report = probe(self.api)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.finished = True


class CalibApp:
    CAM_W = 460                 # camera panel width, source pixels scaled to it

    def __init__(self, camera="synthetic", dry=False, roster_path=None,
                 workspace_path=None):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Sphero calibration bench")
        fit_to_display()
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 16, bold=True)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)

        self.ws = Workspace.load(workspace_path) if workspace_path else Workspace.load()
        self.roster = Roster.load(roster_path) if roster_path else Roster.load()
        self.log = []
        self.log_scroll = 0

        self.tracker = None
        self.camera_spec = camera
        if camera is not None:
            self.start_camera(camera)

        # An empty fleet: robots join it when the trainer connects them, so
        # nothing is driven that the trainer did not ask for.
        # Same reasoning as app.py: a robot the battery has measured should
        # simulate like that robot, so a `--dry` rehearsal is a rehearsal of
        # the hardware rather than of a livelier machine that does not exist.
        self.fleet = Fleet(workspace=self.ws, tracker=self.tracker,
                           motion_path=characterize.MOTION_PATH)
        self.selected = self.roster.entries[0].code if self.roster.entries else None

        self.tab = "colour"
        self.scan = None
        self.found = []
        self.autotune = None
        self.probe = None               # a SensorProbe on its own thread
        self.run = None                 # a Characterization in progress
        self.run_code = None
        self.run_t = 0.0
        self.trail = []                 # recent tracked positions, in cm
        self.last_probe = None
        self._said_orbit = False
        self._said_arrival = False
        self._lost_for = 0.0
        self._said_lost = False
        self.pd = None                  # PDController while driving
        self.path = None                # Point / Line / Circle
        # Set when a drive starts circling and the bench stops to re-measure
        # the aim frame. See `start_recovery`.
        self.recover = None             # an ActiveCalibration, mid-drive
        self.recovered = 0              # corrections applied to THIS drive

        # "straight" aims once and commits; "pd" corrects continuously. A PD
        # loop re-aims every frame, so a rotated frame bends the path into a
        # circle it can never close — which is what a real ball does here. A
        # committed leg goes the WRONG WAY instead, in a straight line, and a
        # wrong direction is a measurement. See `watch_aim`.
        self.drive_mode = "straight"
        # Hand-tuning values. `None` means "whatever the measurement says";
        # touching a slider pins it, so a number a person chose is never
        # silently overwritten by the next calibration.
        self.tune_kp = self.tune_kd = self.tune_predict = None
        self._tune_seeded = False
        # Found by hand at the ball, and they are not arbitrary: 450ms is about
        # two command round trips on this radio, so the controller stops
        # issuing headings the robot has not finished acting on. Re-aiming
        # under 10deg of drift buys nothing a 230ms link can deliver.
        self.tune_retarget, self.tune_interval, self.tune_creep = 10.0, 0.45, 0.45
        self.aim_legs = []              # (commanded_deg, error_deg) this drive
        self.aim_from = None            # position when the current leg began
        self.aim_deg = None             # the bearing it committed to
        self.aim_fixes = 0              # frame corrections made this drive
        self.shape = "point"            # what a click builds
        self.pending = []               # clicks collected toward a shape
        # Driving and measuring want different speeds and get separate ones.
        # 6cm/s is what a person watching a ball actually wants: it lands where
        # it was pointed and there is time to react. The battery cannot use it
        # — a speed map confined to bytes 18-26 has no curve left to fit — so
        # the cap on runs stays where the measurement needs it.
        self.path_speed = 6
        self.radius = 45
        # How close counts as arrived. Seeded from the robot's own measurement
        # when it has one — an exact point is not a thing a Sphero can hold.
        self.arrive_cm = 5
        self.last_fit = None
        self.cam_surface, self.cam_stamp = None, None
        self.sliders = []
        self.buttons = []
        # Cleared here, in the one block every branch runs, rather than in each
        # branch separately. Per-branch resets are how the motion tab kept the
        # colour tab's geometry: adding a branch means remembering to clear
        # something in it, and the reminder is a crash three tabs later.
        self.palette_rects = {}
        self.wheel_rect = None
        self.motion_top = 14 + 26 + GAP + 28 + GAP
        self.drive_top = self.motion_top
        self.arena_rect = pygame.Rect(DOCK + PAD, self.drive_top, 400, 400)
        self.arena_scale = 2.0
        self.dragging_slider = None
        self.dry = dry
        self._wheel = None              # cached hue ring
        self._bg, self._bg_at, self._room = None, 0.0, None
        self.corner_mode = False
        self.corners = []
        self.checking = False
        self.checks = []
        self.recovering = False
        self.limiting = False
        # What the battery is allowed to do, and how much room it keeps.
        # `edge_margin` is in stopping distances: at 1.0 the ball may run right
        # up to the distance it needs to stop, at 4.0 it stays four times that
        # far out and crawls anywhere near a wall.
        # A hard ceiling on anything the bench commands, set from what the
        # arena can actually absorb rather than from what the hardware can do.
        # It costs the top of the speed curve — bytes above the cap are skipped
        # and the top speed becomes an extrapolation, which the fit says so
        # about — and it buys a ball that stays on the table.
        self.cap_cm_s = SPEED_CAP_CM_S
        self.edge_margin = 2.5
        x0, x1, y0, y1 = self.ws.bbox
        self.arena_w, self.arena_h = int(round(x1 - x0)), int(round(y1 - y0))

        if self.roster.errors:
            for e in self.roster.errors:
                self.say("error", e)
        self.check_calibration()
        self._build()
        if dry:
            # Rehearsal: the whole battery runs against simulated robots, which
            # is how you find out the rig is broken before a ball is on the
            # floor rather than twenty minutes into a session.
            for e in self.roster.enabled_entries():
                self.connect(e.code, "sim")
            self.say("warn", "dry run — these are simulated robots")
        self.say("info", "connect a robot, then AUTO-TUNE ALL to teach the "
                         "camera its colours")

    # -- plumbing --------------------------------------------------------

    def say(self, kind, text):
        self.log.append((kind, str(text)))
        del self.log[:-LOG_HISTORY]
        if self.log_scroll:
            self.log_scroll += 1

    def start_camera(self, spec):
        from fleet.vision_link import CameraTracker
        if self.tracker is not None:
            self.tracker.stop()
        self.tracker = CameraTracker(source=spec)
        errs = self.tracker.start()
        for e in errs:
            self.say("error", e)
        if not errs:
            self.say("ok", f"camera {spec} running")
        if getattr(self, "fleet", None) is not None:
            self.fleet.tracker = self.tracker
            for h in self.fleet.handles.values():
                if hasattr(h, "tracker"):
                    h.tracker = self.tracker

    def check_calibration(self):
        """The trap from HANDOFF section 9k, checked before anything moves."""
        from vision.homography import Homography
        h = Homography.load()
        x0, x1, y0, y1 = self.ws.bbox
        if not h.ready:
            self.say("warn", "no homography — the camera cannot report cm. "
                             "Run `python -m vision.app` to click the corners.")
        elif not h.matches(x1 - x0, y1 - y0):
            self.say("error", f"homography is {h.width:.0f}x{h.height:.0f}cm but "
                              f"the workspace is {x1 - x0:.0f}x{y1 - y0:.0f}cm — "
                              "the tracker and the planner disagree about the floor")
        else:
            self.say("ok", f"arena {x1 - x0:.0f}x{y1 - y0:.0f}cm, camera agrees")

    @property
    def homography(self):
        """Cached: to_px inverts a 3x3 every call, and this runs per frame."""
        if getattr(self, "_H", None) is None:
            from vision.homography import Homography
            self._H = Homography.load()
        return self._H

    def cm_to_panel(self, pts_cm, origin, f):
        """Arena centimetres -> pixels inside a drawn camera panel.

        Through the homography rather than from raw blob pixels, so a simulated
        robot — which the camera cannot see at all — draws in exactly the same
        place a real one would. One code path, and the trail stays meaningful
        during a dry run.
        """
        H = self.homography
        if not H.ready or not len(pts_cm):
            return []
        try:
            px = H.to_px(np.asarray(pts_cm, dtype=float).reshape(-1, 2))
        except Exception:
            return []
        return [(int(origin[0] + p[0] * f), int(origin[1] + p[1] * f)) for p in px]

    GRID_MINOR = 25.0
    GRID_MAJOR = 100.0

    def draw_arena_overlay(self, origin, f):
        """The workspace grid, warped through the homography onto the frame.

        Drawn through `to_px` rather than as a rectangle, because the camera
        looks at the floor from an angle and a rectangle is exactly what the
        arena is NOT in the image. What comes out is a trapezoid with lines
        converging toward the horizon — and if those lines sit on the tape,
        the calibration is right.

        This is a better check than any number. `calib/homography.json` can say
        200x200 with total confidence while describing a floor a metre from the
        one the robots are on, and no field in it looks wrong.
        """
        H = self.homography
        if not H.ready:
            return
        s = self.screen
        x0, x1, y0, y1 = self.ws.bbox
        w, h = x1 - x0, y1 - y0
        matched = H.matches(w, h)
        # A mismatch is drawn, not hidden: the grid is then showing the CAMERA's
        # idea of the floor against the PLANNER's, and seeing the two disagree
        # is the fastest way to understand what is wrong.
        minor = (34, 74, 110) if matched else (110, 70, 40)
        major = (60, 120, 165) if matched else (170, 110, 50)

        def line(a, b, colour, width=1):
            try:
                pts = H.to_px([a, b])
            except Exception:
                return
            p0 = (int(origin[0] + pts[0][0] * f), int(origin[1] + pts[0][1] * f))
            p1 = (int(origin[0] + pts[1][0] * f), int(origin[1] + pts[1][1] * f))
            pygame.draw.line(s, colour, p0, p1, width)

        def at(cm):
            q = H.to_px([cm])[0]
            return (int(origin[0] + q[0] * f), int(origin[1] + q[1] * f))

        step = self.GRID_MINOR if w <= 300 else self.GRID_MINOR * 2
        v = step
        while v < w:
            line((v, 0), (v, h), major if v % self.GRID_MAJOR == 0 else minor)
            v += step
        v = step
        while v < h:
            line((0, v), (w, v), major if v % self.GRID_MAJOR == 0 else minor)
            v += step

        outline = MINT if matched else CORAL
        for a, b in (((0, 0), (w, 0)), ((w, 0), (w, h)),
                     ((w, h), (0, h)), ((0, h), (0, 0))):
            line(a, b, outline, 2)

        # The axes, so "which way is +x" is answerable from the picture.
        line((0, 0), (min(40.0, w * 0.3), 0), CORAL, 3)
        line((0, 0), (0, min(40.0, h * 0.3)), MINT, 3)
        try:
            o = at((0.0, 0.0))
            pygame.draw.circle(s, CHALK, o, 5, 2)
            s.blit(self.fs.render("0,0", True, CHALK), (o[0] + 8, o[1] + 4))
            s.blit(self.fs.render("x", True, CORAL), at((min(46.0, w * 0.34), 0.0)))
            s.blit(self.fs.render("y", True, MINT), at((0.0, min(46.0, h * 0.34))))
            for cm in (self.GRID_MAJOR, 2 * self.GRID_MAJOR):
                if cm < w:
                    s.blit(self.fs.render(f"{cm:.0f}", True, major), at((cm, 0.0)))
                if cm < h:
                    s.blit(self.fs.render(f"{cm:.0f}", True, major), at((0.0, cm)))
        except Exception:
            pass
        if not matched:
            s.blit(self.fs.render(
                f"grid is the camera's {H.width:.0f}x{H.height:.0f}cm, workspace "
                f"is {w:.0f}x{h:.0f}cm", True, CORAL),
                (origin[0], origin[1] + 4))

    def draw_camera_panel(self, x, y, trail=(), caption=None):
        """The frame, the blobs the detector found, and where the ball has been."""
        s = self.screen
        surf, raw, f = self.camera_surface()
        if surf is None:
            card(s, pygame.Rect(x, y, self.CAM_W, int(self.CAM_W * 0.75)))
            s.blit(self.f.render("no camera frame", True, GREY), (x + 12, y + 12))
            return pygame.Rect(x, y, self.CAM_W, int(self.CAM_W * 0.75))

        rect = pygame.Rect(x, y, surf.get_width(), surf.get_height())
        s.blit(surf, (x, y))
        if not self.corner_mode:
            self.draw_arena_overlay((x, y), f)

        pts = self.cm_to_panel(trail, (x, y), f)
        if len(pts) > 1:
            clipped = [p for p in pts if rect.collidepoint(p)]
            if len(clipped) > 1:
                pygame.draw.lines(s, CYAN, False, clipped, 2)
                pygame.draw.circle(s, CHALK, clipped[-1], 4)

        for name, blob in (raw or {}).items():
            bx, by, area = blob
            px = (int(x + bx * f), int(y + by * f))
            rad = max(4, int(np.sqrt(max(area, 1.0) / np.pi) * f) + 2)
            pygame.draw.circle(s, LED_RGB.get(name, CHALK), px, rad, 2)

        pygame.draw.rect(s, RULE, rect, 1)
        if caption:
            s.blit(self.fs.render(caption, True, DIM), (x, y - 14))
        return rect

    @property
    def detector(self):
        """The live Detector inside the tracker, so edits take effect at once."""
        t = getattr(self.tracker, "tracker", None)
        return getattr(t, "det", None) if t else None

    @property
    def handle(self):
        return self.fleet.handles.get(self.selected)

    @property
    def entry(self):
        return self.roster.by_code(self.selected)

    @property
    def sig_color(self):
        e = self.entry
        return e.color if e else "red"

    # -- membership ------------------------------------------------------

    def blob_count(self):
        """(blobs the camera sees, robots the camera has a fix on).

        Handed to the tracking check so a ball that appears not to move can be
        told from a tracker that is not watching it. Both look identical in the
        nudge itself, and only one of them is worth re-tuning colours over.
        """
        if self.tracker is None:
            return None, None
        try:
            blobs = len(self.tracker.latest()[1] or {})
        except Exception:
            return None, None
        live = sum(1 for x in self.fleet.handles.values() if x.connected)
        return blobs, live

    def sync_tracked_colours(self):
        """Point the tracker at the colours the bench is actually wearing.

        Called whenever the fleet changes. Without it the tracker hunts all six
        hues forever, so a bench holding two robots reports blobs for four
        colours nobody is wearing — and each of those is something the tracker
        can lock onto and something downstream can be driven from.
        """
        if self.tracker is None:
            return
        worn = {h.color for h in self.fleet.handles.values() if h.color}
        self.tracker.set_colors(worn)

    def connect(self, code, kind):
        entry = self.roster.by_code(code)
        if entry is None:
            return
        if code in self.fleet.handles:
            self.fleet.remove(code)
            self.sync_tracked_colours()
            self.say("info", f"{code} released")
            self._build()
            return
        if kind == "real" and not entry.ble_name:
            self.say("error", f"{code} has no ble_name — SCAN, then click a "
                              "discovered Sphero to bind it")
            return
        e = RobotEntry(**{**entry.to_dict(), "kind": kind})
        errs = self.fleet.add(e)
        if errs:
            for x in errs:
                self.say("error", x)
            return
        self.selected = code
        h = self.fleet.handles[code]
        self.sync_tracked_colours()
        h.set_led(self.led_for(entry.color))
        self.say("ok", f"{code} joined as {kind}"
                       + (f" ({entry.ble_name})" if kind == "real" else ""))
        self._build()

    def bind_ble(self, ble_name):
        """Attach a discovered Sphero to the selected roster row."""
        e = self.entry
        if e is None:
            self.say("error", "select a robot first")
            return
        if e.code in self.fleet.handles:
            self.say("error", f"release {e.code} before rebinding it")
            return
        errs = self.roster.set_ble(e.code, ble_name)
        if errs:
            for x in errs:
                self.say("error", x)
            return
        stolen = list(getattr(self.roster, "stolen", []))
        for x in self.roster.save(allow_no_worse=True):
            self.say("error", x)
            return
        self.say("ok", f"{e.code} bound to {ble_name} — saved to roster.json")
        for code in stolen:
            self.say("warn", f"{code} was bound to the same ball and is now "
                             "unbound — one ball per row")
        self._build()

    def unbind(self, code):
        """Clear a row's ball. The route out of a roster with duplicates in it."""
        if code in self.fleet.handles:
            self.fleet.remove(code)
        errs = self.roster.clear_ble(code)
        if errs:
            for x in errs:
                self.say("error", x)
            return
        for x in self.roster.save(allow_no_worse=True):
            self.say("error", x)
            return
        self.say("ok", f"{code} unbound")
        self._build()

    def start_scan(self):
        if self.scan is not None and not self.scan.finished:
            return
        self.found = []
        self.scan = BleScan()
        self.scan.start()
        self.say("info", "scanning for Spheros…")

    def drain_scan(self):
        if self.scan is None or not self.scan.finished:
            return
        if self.scan.error:
            self.say("error", self.scan.error)
        self.found = list(self.scan.names)
        for n in self.found:
            self.say("ok", f"found {n}")
        self.scan = None
        self._build()

    # -- colour ----------------------------------------------------------

    def sig(self, name=None):
        d = self.detector
        name = name or self.sig_color
        if d is None:
            return dict(vconfig.COLORS.get(name, {}))
        return d.colors.setdefault(name, dict(vconfig.COLORS[name]))

    def set_slider(self, key, value):
        """A slider moved. Anything the LED depends on has to reach the LED."""
        self.set_sig(key, value)
        if key in ("hue", "led_value"):
            self.relight()

    def set_sig(self, key, value):
        d = self.detector
        if d is None:
            return
        d.colors.setdefault(self.sig_color, dict(vconfig.COLORS[self.sig_color]))
        d.colors[self.sig_color][key] = int(value)

    def get_sig(self, key, fallback):
        s = self.sig()
        d = self.detector
        if key in s:
            return int(s[key])
        if key == "led_value":
            return vconfig.LED_VALUE
        return int(d.thresh.get(key, fallback)) if d else fallback

    # -- staying on the floor --------------------------------------------
    #
    # Every stage aims its own legs at clear floor, and that is not enough on
    # real hardware. A leg aimed correctly still ends somewhere else when the
    # ball slips, when the heading offset is not yet known, or when the tracker
    # drops a few frames mid-leg — and by then the ball is at the edge of the
    # frame with a stage happily commanding it further out.
    #
    # So containment sits BELOW the stages and overrides them. Nothing a stage
    # asks for reaches a robot that is outside the safe margin; it gets driven
    # gently back instead, and the stage is paused meanwhile so the recovery
    # does not appear in the measurement.

    def stop_s(self):
        """This robot's measured stopping constant, or a cautious default."""
        return safety.stopping_seconds(load_motion(self.selected))

    def contain(self, h, cmd):
        """Vet one command. Returns what to send, and whether to pause the stage.

        A graduated limit, not a fence. The earlier version waited until the
        ball was already at the edge and then took over, which meant it had to
        be rescued by hand often enough to matter — and every rescue is a person
        picking up a robot mid-measurement, which is precisely the disturbance
        the measurement is trying to avoid.

        Now the ceiling falls smoothly with the floor left in front: full speed
        in the middle, a crawl near the edge, nothing at all into a wall it
        cannot stop before. A robot never arrives at the edge with any speed to
        carry it over, so there is nothing to recover from.
        """
        if not h.connected:
            # No fix means no idea where it is. Driving blind is how a ball
            # ends up under a desk.
            return None, True
        if cmd is None:
            return None, False

        heading, byte = cmd
        byte = min(int(byte), int(round(self.cap_cm_s / 60.0 * 255)))
        heading, byte, clamped = safety.limit(
            self.ws, h.pos, heading, byte, stop_s=self.stop_s(),
            max_speed=60.0, safety=self.edge_margin)
        if clamped and not self.limiting:
            self.limiting = True
            self.say("warn", f"{h.code} is near the edge — holding it slow")
        elif not clamped and self.limiting:
            self.limiting = False
        if byte <= 0:
            # Pinned with nowhere safe to go: hand back the wheel gently rather
            # than sitting on a stalled command.
            x0, x1, y0, y1 = self.ws.bbox
            to = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0]) - np.asarray(h.pos,
                                                                          dtype=float)
            d = float(np.linalg.norm(to))
            if d < 1e-6:
                return None, True
            course = float(math.degrees(math.atan2(to[0], to[1])) % 360.0)
            aim = self.run.aim if self.run is not None else (lambda c: c)
            if not self.recovering:
                self.recovering = True
                self.say("warn", f"{h.code} has no room — easing it back in")
            return (aim(course), 45), True
        if self.recovering:
            self.recovering = False
            self.say("ok", f"{h.code} has room again")
        return (heading, byte), False

    # -- the arena -------------------------------------------------------

    LOST_GIVE_UP_S = 3.0        # of no fix before a drive is abandoned entirely

    CORNER_LABELS = ("origin (0,0)", "+x axis", "far corner", "+y axis")

    def start_corners(self):
        """Pick the arena out of the camera image, and set the workspace to match.

        Both from the same four clicks. Keeping them apart is what allowed the
        homography to describe a 200x200 floor while `workspace.json` described
        a 240x180 one — robots then drove calmly off an arena they were nowhere
        near the edge of, and nothing errored because each file was internally
        fine. Written together they cannot disagree.
        """
        if self.tracker is None:
            self.say("error", "no camera — there is nothing to pick corners in")
            return
        # Checked before the first click rather than after the fourth. By the
        # fourth the trainer is at the far corner of the arena, and finding out
        # then that nothing could be saved wastes the whole pick.
        blocked = vconfig.writable("homography")
        if blocked:
            self.say("error", blocked)
            return
        # Seeded from the workspace each time, not once at startup: the sliders
        # should open showing the arena currently in force, so re-picking the
        # corners of an unchanged room does not silently resize it.
        x0, x1, y0, y1 = self.ws.bbox
        self.arena_w, self.arena_h = int(round(x1 - x0)), int(round(y1 - y0))
        self.corner_mode = True
        self.corners = []
        self.say("info", "put a ROBOT on each corner and click the ROBOT, not "
                         "the floor — see the log for why")
        self.say("info", f"click the {self.CORNER_LABELS[0]} of the arena "
                         "(then +x, the far corner, and +y). "
                         "backspace or right-click undoes, esc cancels")
        self._build()

    def undo_corner(self):
        """Take back the last point.

        A four-click sequence with no undo means one slip costs all four, and
        the slip is likeliest on the last one — by then the trainer is leaning
        over the arena rather than looking at the screen.
        """
        if not self.corner_mode or not self.corners:
            return
        self.corners.pop()
        n = len(self.corners)
        self.say("info", f"took that one back — click the {self.CORNER_LABELS[n]}")

    def cancel_corners(self, note="corner picking cancelled"):
        if not self.corner_mode:
            return
        self.corner_mode, self.corners = False, []
        self.say("warn", note)
        self._build()

    def add_corner(self, panel_xy):
        """One click on the camera panel, stored in SOURCE pixels."""
        surf, _, f = self.camera_surface()
        if surf is None or f <= 0:
            return
        self.corners.append((panel_xy[0] / f, panel_xy[1] / f))
        n = len(self.corners)
        if n < 4:
            self.say("info", f"now the {self.CORNER_LABELS[n]}")
            return
        self.apply_corners()

    def apply_corners(self):
        from vision.homography import Homography
        pts = self.corners[:4]
        w, h = float(self.arena_w), float(self.arena_h)
        # The sliders move in whole centimetres, and an arena measured with a
        # tape is not a whole number. If a slider has not been moved off the
        # size the workspace already holds, keep the workspace's exact figure
        # rather than the rounded one the slider can express — 138.8 stays
        # 138.8 instead of quietly becoming 139.
        x0, x1, y0, y1 = self.ws.bbox
        if abs(w - (x1 - x0)) < 1.0:
            w = float(x1 - x0)
        if abs(h - (y1 - y0)) < 1.0:
            h = float(y1 - y0)
        try:
            hom = Homography().set_rect(pts, w, h)
        except Exception as e:
            self.cancel_corners(f"those four points do not make a rectangle: {e}")
            return

        # Sanity: the clicked quadrilateral has to map back onto itself. A
        # degenerate pick — three points in a line, or a crossed order — still
        # produces a matrix, and it produces one that puts robots anywhere.
        # Against the order the homography actually used. `set_rect` re-reads
        # the four clicks so the arena is never mirrored, so checking the
        # clicked order here would fail every time the re-reading did its job.
        back = hom.to_cm(hom.corners or pts)
        want = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        if float(np.abs(back - want).max()) > 1.0:
            self.cancel_corners("those corners did not map cleanly — click them "
                                "in order: origin, +x, far corner, +y")
            return

        try:
            hom.save()
        except Exception as e:
            # A bench that dies because a file is unwritable is a bench that
            # dies in the middle of a session, having already asked for four
            # careful clicks. Say what happened and how to fix it.
            self.cancel_corners(f"could not write the calibration: {e}")
            self.say("error", "if a file there is owned by root from an old "
                              "sudo run, remove it and try again: "
                              "rm calib/homography.json")
            return
        self._H = hom
        self.cam_stamp = None

        # The workspace, from the same two numbers, keeping whatever was in it.
        old = self.ws
        self.ws = Workspace(bounds_cm=[[0, 0], [w, 0], [w, h], [0, h]],
                            obstacles=list(getattr(old, "obstacles", []) or []),
                            entities=getattr(old, "entities", None),
                            path=old.path)
        try:
            errs = self.ws.save()
        except Exception as e:
            errs = [f"could not write the workspace: {e}"]
        if errs:
            # The homography is already on disk and the workspace is not, which
            # is exactly the split this feature exists to prevent. Say so
            # plainly rather than leaving the two quietly disagreeing.
            for e in errs:
                self.say("error", e)
            self.say("error", "the camera calibration was saved but the "
                              "workspace was not — they now disagree. Fix the "
                              "file and set the arena again.")
            self.ws = old
        self.fleet.ws = self.ws
        for handle in self.fleet.handles.values():
            handle.ws = self.ws
        self.corner_mode, self.corners = False, []
        self.say("ok", f"arena set to {w:.0f}x{h:.0f}cm — homography and "
                       "workspace written together")
        self.check_calibration()
        self._build()

    # -- checking where the tracker thinks a robot is ---------------------

    def start_position_check(self):
        """Measure the gap between where a robot IS and where it is reported.

        The calibration maps a plane, and a ball's centre is not on the floor —
        it is a shell radius above it. Viewed from anything but straight down,
        that height projects the ball OUTWARD, away from the point beneath the
        camera, and the tracker reports it further away than it is. The effect
        is small (one to three centimetres at typical mounting heights) and
        entirely systematic, so it is invisible until something depends on
        absolute position and then it is baffling.

        Rather than model it, measure it: click where the robot really is and
        this reports the discrepancy. Several clicks at different places tell
        you whether it is a constant offset, a scale error, or the outward
        splay that means the calibration is on the wrong plane.
        """
        if self.tracker is None:
            self.say("error", "no camera")
            return
        h = self.handle
        if h is None or not h.connected:
            self.say("error", "connect a robot the tracker can see first")
            return
        self.checking = True
        self.checks = []
        self.say("info", f"click exactly where {h.code} really is, a few times "
                         "in different parts of the arena. esc when done")
        self._build()

    def add_position_check(self, panel_xy):
        surf, _, f = self.camera_surface()
        h = self.handle
        if surf is None or f <= 0 or h is None:
            return
        clicked_px = (panel_xy[0] / f, panel_xy[1] / f)
        H = self.homography
        if not H.ready:
            self.say("error", "no homography to check against")
            return
        try:
            clicked_cm = np.asarray(H.to_cm([clicked_px])[0], dtype=float)
        except Exception as e:
            self.say("error", f"could not map that point: {e}")
            return
        reported = np.asarray(h.pos, dtype=float)
        err = reported - clicked_cm
        self.checks.append({"clicked": clicked_cm, "reported": reported,
                            "error": err})
        self.say("info", f"#{len(self.checks)}: you say "
                         f"({clicked_cm[0]:.0f},{clicked_cm[1]:.0f}), tracker says "
                         f"({reported[0]:.0f},{reported[1]:.0f}) — "
                         f"{float(np.linalg.norm(err)):.1f}cm out")

    def finish_position_check(self):
        self.checking = False
        n = len(self.checks)
        if n == 0:
            self.say("warn", "position check cancelled")
            self._build()
            return
        errs = np.array([c["error"] for c in self.checks])
        mags = np.linalg.norm(errs, axis=1)
        self.say("ok", f"{n} point(s): mean {mags.mean():.1f}cm, "
                       f"worst {mags.max():.1f}cm")

        if n >= 3:
            # Is the error outward from a single point? That is the signature
            # of a calibration on the wrong plane, and it is fixable — recalibrate
            # by clicking a ROBOT on each corner instead of the floor.
            centre = np.array([c["clicked"] for c in self.checks]).mean(axis=0)
            radial = []
            for c in self.checks:
                v = c["clicked"] - centre
                d = float(np.linalg.norm(v))
                if d > 5.0:
                    radial.append(float(np.dot(c["error"], v / d)))
            # Is the mean displacement bigger than scatter alone would give?
            # Comparing it to a fraction of the average error is the tempting
            # test and it is not a test at all: pure noise passes it whenever
            # the sample happens to lean, and it then blames the corners for a
            # calibration that is fine.
            mean = errs.mean(axis=0)
            stderr = float(np.linalg.norm(errs.std(axis=0, ddof=1))) / np.sqrt(n)
            shifted = float(np.linalg.norm(mean)) > 2.0 * max(stderr, 1e-9)

            # Tested on its own terms, not via the mean displacement. A pure
            # outward splay has a mean of very nearly zero — the errors point
            # in opposite directions and cancel — so gating it on a mean shift
            # rejects precisely the case it exists to catch.
            splayed = False
            if len(radial) >= 3:
                rad = np.array(radial, dtype=float)
                rad_se = float(rad.std(ddof=1)) / np.sqrt(len(rad))
                splayed = rad.mean() > 2.0 * max(rad_se, 1e-9) and rad.min() > 0.0

            if splayed:
                self.say("error", "every error points away from the middle — the "
                                  "calibration is on the floor, but the balls "
                                  "ride above it. Set the arena again, clicking "
                                  "a ROBOT placed on each corner.")
                self.fit_parallax()
            elif shifted:
                self.say("warn", f"the error is a constant shift of "
                                 f"({mean[0]:+.1f},{mean[1]:+.1f})cm — the corners "
                                 "were probably clicked slightly off")
            else:
                self.say("ok", "no systematic pattern — that is measurement "
                               "noise, and it is as good as this camera gets")
        self._build()

    def flip_arena_y(self):
        """Mirror the arena, for a camera that delivers a mirrored image.

        The tell is never in the picture — a mirrored image is perfectly
        self-consistent, so corners map cleanly and the grid sits on the floor.
        It shows only when a robot drives: the aim error changes sign with
        direction, which no heading offset can cancel. See `mirrored_frame`.
        """
        hom = getattr(self.tracker.tracker, "H", None) if self.tracker else None
        if hom is None or not hom.ready:
            self.say("error", "no arena calibration to flip")
            return
        hom.flip_y()
        try:
            hom.save()
        except Exception as e:
            self.say("error", f"could not save the flipped arena: {e}")
            return
        for h in self.fleet.handles.values():
            h.heading_offset = 0.0      # measured in the old frame; worthless now
        for x in self.roster.save():
            self.say("error", x)
        self.say("ok", "arena y flipped and saved. Every heading offset has "
                       "been cleared — they were measured in the old frame. "
                       "Drive a point and the aim frame will re-measure itself; "
                       "if the error still changes sign, the flip was not the "
                       "problem and you can press it again to undo.")
        self._build()

    def fit_parallax(self):
        """Measure the ball-height offset from the points just checked, and
        correct for it rather than only naming it.

        Clicking a robot on each corner is the better fix and the log says so:
        it maps the plane the balls travel in and the parallax is gone rather
        than corrected. But the samples to do it the other way have just been
        taken, and a correction measured from the arena in front of you beats a
        recalibration nobody gets round to.
        """
        from vision import parallax as px

        hom = getattr(self.tracker.tracker, "H", None) if self.tracker else None
        if hom is None or not hom.ready:
            return
        # `reported` already has any previous correction applied, so a second
        # fit on top of the first would double it.
        old = hom.parallax
        hom.parallax = None
        try:
            fit = px.fit([(c["clicked"], c["reported"]) for c in self.checks])
        finally:
            if fit is None:
                hom.parallax = old

        if fit is None:
            self.say("warn", "not enough spread in those points to measure the "
                             "ball height — check three or more, well apart")
            return
        hom.parallax = fit
        try:
            hom.save()
        except Exception as e:
            self.say("error", f"could not save the parallax fit: {e}")
            return
        self.say("ok", f"measured it instead: everything sits {fit['scale']:.3f}x "
                       f"out from ({fit['nadir_cm'][0]:.0f},{fit['nadir_cm'][1]:.0f})"
                       f"cm, which is where the camera is looking straight down. "
                       f"Correcting for it takes the error from {fit['was_cm']}cm "
                       f"to {fit['residual_cm']}cm. Saved — re-picking the arena "
                       "clears it.")

    # -- the wheel -------------------------------------------------------

    def led_for(self, name):
        """What to tell a robot to glow, derived from the hue we look for."""
        d = self.detector
        return vconfig.led_for(name, d.colors if d is not None else None)

    def relight(self):
        """Push current hues out to every connected robot's LED."""
        for code, h in self.fleet.handles.items():
            e = self.roster.by_code(code)
            if e is not None:
                h.set_led(self.led_for(e.color))

    def wheel_surface(self, size):
        """The hue ring. Built once — it is the same picture every frame."""
        if getattr(self, "_wheel", None) is not None and self._wheel.get_width() == size:
            return self._wheel
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c, outer, inner = size / 2.0, size / 2.0 - 2, size / 2.0 - 26
        for hue in range(vpalette.HUES):
            a0 = np.radians(hue * 2.0 - 1.2)
            a1 = np.radians(hue * 2.0 + 1.2)
            pts = [(c + inner * np.cos(a0), c + inner * np.sin(a0)),
                   (c + outer * np.cos(a0), c + outer * np.sin(a0)),
                   (c + outer * np.cos(a1), c + outer * np.sin(a1)),
                   (c + inner * np.cos(a1), c + inner * np.sin(a1))]
            pygame.draw.polygon(surf, vconfig.led_rgb(hue), pts)
        self._wheel = surf
        return surf

    def wheel_hue(self, pos):
        """Screen point -> hue, or None if the click missed the ring."""
        r = self.wheel_rect
        c = np.array([r.centerx, r.centery], dtype=float)
        d = np.array(pos, dtype=float) - c
        radius = float(np.linalg.norm(d))
        if radius < r.w / 2.0 - 30 or radius > r.w / 2.0 + 2:
            return None
        return int(round(np.degrees(np.arctan2(d[1], d[0])) % 360.0 / 2.0)) % vpalette.HUES

    def set_hue(self, hue):
        """Move the selected robot's slot to a hue, and light it to match."""
        name = self.sig_color
        self.set_sig("hue", int(hue) % vpalette.HUES)
        self.relight()
        e = self.entry
        self.say("info", f"{name} ({e.code if e else '?'}) -> hue {int(hue)}  "
                         f"clarity {self.hue_clarity(hue):.2f}")

    def background(self):
        """The room's colour profile, refreshed a few times a second at most."""
        now_t = time.time()
        if getattr(self, "_bg", None) is not None and now_t - self._bg_at < 2.0:
            return self._bg
        frame = None
        if self.tracker is not None:
            try:
                frame, _ = self.tracker.latest()
            except Exception:
                frame = None
        if frame is None:
            self._bg, self._bg_at = np.zeros(vpalette.HUES), now_t
            return self._bg
        self._bg = vpalette.background_profile(frame)
        self._room = vpalette.room_light(frame)
        self._bg_at = now_t
        return self._bg

    def hue_clarity(self, hue):
        return vpalette.clarity(self.background(), hue)

    def palette_report(self):
        """How the palette in use scores, and what the best available would."""
        hist = self.background()
        used = [self.sig(e.color)["hue"] for e in self.roster.enabled_entries()]
        current = vpalette.score_existing(used, hist)
        return current, vpalette.summarise(current)

    def optimise_palette(self):
        """Re-pick every hue for the room the camera is looking at right now."""
        d = self.detector
        if d is None:
            self.say("error", "no camera, so no room to optimise for")
            return
        hist = self.background()
        entries = self.roster.enabled_entries()
        current, before = self.palette_report()
        chosen = vpalette.optimise(len(entries), hist)
        after = vpalette.summarise(chosen)

        if (after["worst_separation"] <= before["worst_separation"]
                and after["worst_clarity"] <= before["worst_clarity"]):
            self.say("ok", "the palette in use is already as good as anything "
                           "this room allows — leaving it alone")
            return

        # Assigned in hue order to the robots already in hue order, so the
        # colours shift as little as possible: a trainer who knows Caraxes is
        # the reddish one should still find it reddish afterwards.
        order = sorted(entries, key=lambda e: self.sig(e.color)["hue"])
        for e, pick in zip(order, sorted(chosen, key=lambda c: c["hue"])):
            d.colors.setdefault(e.color, dict(vconfig.COLORS[e.color]))["hue"] = pick["hue"]
        self.relight()
        self.save_signatures()
        room = getattr(self, "_room", None)
        if room:
            self.say("info", f"room: {room['verdict']} (brightness "
                             f"{room['brightness']:.0f})")
        self.say("ok", f"separation {before['worst_separation']}->"
                       f"{after['worst_separation']}, clarity "
                       f"{before['worst_clarity']:.2f}->{after['worst_clarity']:.2f}")
        for e, pick in zip(order, sorted(chosen, key=lambda c: c["hue"])):
            self.say("info", f"  {e.code} {e.color}: hue {pick['hue']} "
                             f"clarity {pick['clarity']:.2f}")

    def set_color(self, name):
        """Move the selected robot onto a different hue, and save it.

        Which hue a robot wears is a camera decision, not a naming one: if the
        overhead light cannot separate this room's cyan from its blue, the fix
        is to move a robot onto a hue that does separate, and that has to be
        changeable here rather than by hand-editing roster.json between runs.
        """
        e = self.entry
        if e is None:
            return
        was = e.color
        errs = self.roster.set_color(e.code, name)
        if errs:
            for x in errs:
                self.say("error", x)
            return
        for x in self.roster.save():
            self.say("error", x)
            return

        # The live fleet has to follow, or the tracker keeps looking for the
        # robot under its old hue and the LED keeps advertising it.
        for code, h in self.fleet.handles.items():
            entry = self.roster.by_code(code)
            if entry is not None and h.color != entry.color:
                h.color = entry.color
                h.set_led(self.led_for(entry.color))
        self.sync_tracked_colours()
        swapped = next((x.code for x in self.roster.entries
                        if x.color == was and x.code != e.code and x.enabled), None)
        self.say("ok", f"{e.code} is now {name}"
                       + (f", swapped with {swapped}" if swapped else "")
                       + " — saved to roster.json")
        self._build()

    def cycle_color(self, code):
        """Next hue along, for the swatch in the roster row."""
        e = self.roster.by_code(code)
        if e is None:
            return
        names = list(vconfig.COLORS)
        self.selected = code
        self.set_color(names[(names.index(e.color) + 1) % len(names)])

    def separation_report(self):
        """Are the tuned hues actually far enough apart for this camera?

        The question behind all of this. Two signatures whose hue windows
        overlap will swap identities the first time the balls pass close, and
        that failure looks like the tracker going mad rather than like a
        calibration that was never separable in the first place.
        """
        d = self.detector
        if d is None:
            return
        used = [e.color for e in self.roster.enabled_entries()]
        worst = []
        for i, a in enumerate(used):
            for b in used[i + 1:]:
                sa, sb = self.sig(a), self.sig(b)
                gap = abs((sa["hue"] - sb["hue"] + 90) % 180 - 90)
                margin = gap - (sa.get("tol", 8) + sb.get("tol", 8))
                worst.append((margin, a, b, gap))
        worst.sort()
        for margin, a, b, gap in worst[:3]:
            if margin < 0:
                self.say("error", f"{a} and {b} overlap: hues {gap} apart but "
                                  f"their windows need {gap - margin} — the "
                                  "tracker will confuse them")
            elif margin < 6:
                self.say("warn", f"{a} and {b} are only {margin} apart after "
                                 "tolerance — tight")
        if worst and worst[0][0] >= 6:
            self.say("ok", f"all six hues separate, closest pair "
                           f"{worst[0][1]}/{worst[0][2]} by {worst[0][0]}")

    def save_signatures(self):
        d = self.detector
        if d is None:
            self.say("error", "no camera, so nothing to save")
            return
        path = vconfig.save_signatures(d.colors)
        self.say("ok", f"colour signatures saved to {path.name} — every app that "
                       "builds a Detector now sees these")

    def sample_hue_at(self, px):
        """Click the camera panel to take the hue under the cursor.

        The fallback when auto-tune cannot find the ball — a shell that is not
        lit, or two robots overlapping. Never the first thing to reach for.
        """
        import cv2
        frame, _ = self.tracker.latest() if self.tracker else (None, {})
        if frame is None:
            return
        h, w = frame.shape[:2]
        f = self.CAM_W / float(w)
        x, y = int(px[0] / f), int(px[1] / f)
        if not (0 <= x < w and 0 <= y < h):
            return
        patch = frame[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        self.set_sig("hue", int(np.median(hsv[:, 0])))
        self.set_sig("s_min", int(max(30, np.percentile(hsv[:, 1], 10) * 0.8)))
        self.set_sig("v_min", int(max(30, np.percentile(hsv[:, 2], 10) * 0.8)))
        self.say("info", f"{self.sig_color}: sampled hue {self.get_sig('hue', 0)} "
                         "from the frame")

    def start_autotune(self, all_colors=False):
        h = self.handle
        if h is None:
            self.say("error", "connect a robot first — auto-tune drives its LED")
            return
        if self.tracker is None:
            self.say("error", "no camera")
            return
        colors = list(vconfig.COLORS) if all_colors else [self.sig_color]
        self.autotune = AutoTune(h, colors, self.finish_autotune,
                                 light=self.led_for)
        self.say("info", f"auto-tuning {len(colors)} colour(s) on {h.code} — "
                         "keep the ball still and in view")

    def finish_autotune(self, results):
        d = self.detector
        good = 0
        for name, r in results.items():
            if not r.get("ok"):
                self.say("error", f"{name}: {r.get('error', 'failed')}")
                continue
            if d is not None:
                d.colors.setdefault(name, dict(vconfig.COLORS[name])).update(
                    {k: r[k] for k in vconfig.SIGNATURE_KEYS if k in r})
            good += 1
            self.say("ok", f"{name}: hue {r['hue']}±{r['tol']}  s>{r['s_min']} "
                           f"v>{r['v_min']}  area {r['area']}px")
        self.autotune = None
        # Back onto the hues the tracker is looking for. Auto-tune leaves the
        # ball on whichever colour it finished with, and a ball left glowing
        # something nobody is hunting looks exactly like a ball the camera
        # cannot find.
        self.relight()
        if good:
            self.save_signatures()
        if good > 1:
            self.separation_report()
        # The ball ends lit in its own colour, which is what the tracker wants.
        self.relight()

    # -- driving ---------------------------------------------------------

    def gains(self):
        """Gains for the selected robot, from its measurement if it has one."""
        return gains_from_motion(load_motion(self.selected))

    def seed_arrive(self):
        """Set the arrival radius from what this robot can actually hold.

        An exact point is not something a Sphero can sit on: it cannot creep
        below its own deadband, and the camera reports it moving by the noise
        floor even parked. Asking for tighter than either is asking for a
        robot that never reports arriving and hunts around the target instead.
        """
        g = self.gains()
        if g.get("measured") and g.get("arrive_cm"):
            self.arrive_cm = int(round(g["arrive_cm"]))
            self.say("info", f"arrival radius set to {self.arrive_cm}cm from "
                             f"{self.selected}'s own measurement")
            self._build()

    def toggle_drive_mode(self):
        self.drive_mode = "pd" if self.drive_mode == "straight" else "straight"
        self.say("info",
                 "straight: aim once, drive a straight leg, stop. A committed "
                 "leg goes the wrong WAY if the frame is off, instead of "
                 "curving — and a wrong way is measurable."
                 if self.drive_mode == "straight" else
                 "PD loop: corrects every frame. Smoother into the target when "
                 "the aim frame is right, and circles when it is not.")
        if self.path is not None:
            self.start_path(self.path)      # rebuild under the new mode
        self._build()

    def tune_or_measured(self, name):
        """The pinned value if a person set one, else what the battery measured.

        The sliders exist before any drive has started, so they cannot read a
        value that only appears once a controller is built. Falling back to the
        measurement means a slider shows the number actually in force rather
        than a zero that would be a lie about what the robot is doing.
        """
        pinned = getattr(self, f"tune_{name}", None)
        if pinned is not None:
            return float(pinned)
        g = self.gains()
        return float({"kp": g["kp"], "kd": g["kd"],
                      "predict": g["predict_s"]}.get(name, 0.0))

    def set_tune(self, name, value):
        """Pin one gain by hand, and apply it to the drive already running."""
        setattr(self, f"tune_{name}", value)
        self.apply_tuning()

    def apply_tuning(self):
        """Push the hand-set values onto the live controller.

        A knob that only takes effect on the next drive is a settings page, not
        a tuning knob: half of tuning is feeling what a change did to a robot
        that is already moving.
        """
        c = self.pd
        if c is None:
            return
        if self.drive_mode == "straight":
            c.retarget = float(self.tune_retarget)
            c.min_interval = float(self.tune_interval)
            c.creep_frac = float(self.tune_creep)
            c.speed = float(self.path_speed)
            c.tol = float(self.arrive_cm)
        else:
            if self.tune_kp is not None:
                c.kp = float(self.tune_kp)
            if self.tune_kd is not None:
                c.kd = float(self.tune_kd)
            if self.tune_predict is not None:
                c.predict = float(self.tune_predict)
            c.tol = float(self.arrive_cm)

    def start_path(self, path):
        h = self.handle
        if h is None:
            self.say("error", "connect a robot first")
            return
        if self.run is not None:
            self.say("error", "a characterisation run is using this robot")
            return
        g = self.gains()
        if self.drive_mode == "straight":
            self.pd = TurnAndGo(speed=float(self.path_speed),
                                arrive_cm=float(self.arrive_cm))
        else:
            self.pd = PDController(g["kp"], g["kd"], g["max_speed"],
                                   g["deadband_cm_s"], tol=float(self.arrive_cm),
                                   predict_s=g["predict_s"])
        if self.tune_kp is None:
            self.tune_kp, self.tune_kd = g["kp"], g["kd"]
            self.tune_predict = g["predict_s"]
        self.apply_tuning()
        self.aim_from = self.aim_deg = None
        self.aim_fixes = 0
        self.aim_legs = []
        self.path = path
        self.trail = []
        self.recover = None
        self.recovered = 0
        self._said_orbit = False
        self._said_arrival = False
        self._lost_for = 0.0
        self._said_lost = False
        # Asking for less than the motors will accept is not a slow robot, it
        # is a stationary one that occasionally lurches — which looks exactly
        # like a controller that cannot hold a line.
        floor = g.get("deadband_cm_s") or 0.0
        want = getattr(path, "speed", None) or self.path_speed
        if floor and want < floor:
            self.say("error", f"{h.code} will not move below about {floor:.0f}cm/s "
                              f"and you have asked for {want:.0f}. Below the "
                              "deadband it sits still and then lurches; raise "
                              "the speed slider above it.")
        note = "measured" if g["measured"] else "UNMEASURED — run the battery for better gains"
        self.say("ok" if g["measured"] else "warn",
                 f"{h.code} -> {path.describe()}  "
                 f"[kp {g['kp']:.2f} kd {g['kd']:.2f} "
                 f"pred {g['predict_s'] * 1000:.0f}ms, stop within "
                 f"{self.arrive_cm}cm, {note}]")
        if g.get("delay_clamped"):
            self.say("warn", "the measured loop delay was over 450ms, which is "
                             "not a loop delay — it is a run taken against a "
                             "tracker that was not following the ball. Clamped, "
                             "but re-run the battery once the blob count is right.")

    def report_approach(self, h):
        """Say how straight the approach was, and what a curve means.

        The most useful diagnostic the bench has, and it comes free with every
        point-to-point move. A calibration leg drives ONE fixed heading and
        never re-aims, so it is straight whatever the frame offset is — it
        simply goes the wrong way. Driving re-aims every frame, so a rotated
        frame shows up as a curve instead. A trainer who notices that
        calibration paths are straight and driving paths are not has already
        found their heading offset; this saves them working out what it meant.
        """
        ratio = straightness(self.trail)
        if ratio is None:
            return
        deg = implied_heading_error(ratio)
        if deg is None:
            self.say("ok", f"{h.code} arrived, path {ratio:.2f}x straight-line "
                           "— the aim frame looks right")
            return
        self.say("warn", f"{h.code} arrived by a curve, {ratio:.2f}x the straight "
                         f"line. That is about {deg:.0f}deg of heading error: "
                         "driving re-aims every frame so a rotated frame bends "
                         "the path, while a calibration leg holds one heading "
                         "and stays straight however wrong it is.")
        if deg >= 25:
            self.say("warn", "run the heading calibration (MOTION tab) — it will "
                             "straighten this out")

    MAX_RECOVERIES = 2          # per drive; past this the fault is not the aim frame

    AIM_BASELINE_CM = 15.0      # travel before a leg's direction is worth reading
    AIM_TOLERANCE_DEG = 12.0    # below this, leave the frame alone
    MAX_AIM_FIXES = 3           # per drive

    MIRROR_SPREAD_DEG = 70.0    # of error swing across directions before we call it

    def mirrored_frame(self):
        """Does the aim error change sign with the direction driven?

        A rotated frame gives the SAME error whichever way the robot goes, and
        one number cancels it. A mirrored frame reflects every command, so the
        error is `2a - 2*heading` — it sweeps the whole circle as the heading
        changes, and no single offset can cancel it. Told 178 and driving 82 is
        -96; told 194 and driving 345 is +151. That is not noise and it is not
        a robot that needs a bigger correction, it is an arena the wrong way
        round.

        Needs legs pointing genuinely different ways: two legs 5 degrees apart
        cannot tell the two models apart, and calling a mirror on that evidence
        would send a trainer to re-pick corners that were fine.
        """
        legs = [l for l in self.aim_legs if l[0] is not None]
        if len(legs) < 2:
            return False
        spread = max(abs(wrap180(a[0] - b[0])) for a in legs for b in legs)
        if spread < 40.0:
            return False
        swing = max(abs(wrap180(a[1] - b[1])) for a in legs for b in legs)
        return swing > self.MIRROR_SPREAD_DEG

    def watch_aim(self, h):
        """Read the aim frame off whatever leg the robot is already driving.

        This is what committing to a bearing buys. A PD loop re-aims every
        frame, so a rotated frame shows up as a curve — and a curve tells you
        the frame is wrong without telling you by how much, because every
        sample was taken under a different heading. A committed leg holds ONE
        heading, so it travels in a straight line however wrong the frame is,
        and the gap between the bearing commanded and the bearing achieved is
        the error, directly, from an ordinary move.

        So there is no need to detect circling and then stop to measure. Every
        leg is a calibration leg, and the correction lands within a leg or two
        of the drive starting.

        Noise sets the baseline. The error in a direction measured over `d`
        centimetres is about sigma/d, so half a centimetre of position noise
        over 15cm is a couple of degrees — small against the twelve this will
        act on, and small against the eighty-five it is really looking for.
        """
        aim = getattr(self.pd, "aim", None)
        if aim is None:
            self.aim_from = self.aim_deg = None
            return
        if self.aim_deg is None or abs(wrap180(aim - self.aim_deg)) > 1.0:
            self.aim_from, self.aim_deg = np.asarray(h.pos, dtype=float).copy(), aim
            return
        if self.aim_from is None or self.aim_fixes >= self.MAX_AIM_FIXES:
            return

        went = np.asarray(h.pos, dtype=float) - self.aim_from
        travelled = float(np.linalg.norm(went))
        if travelled < self.AIM_BASELINE_CM:
            return

        actual = float(np.degrees(np.arctan2(went[1], went[0])))
        err = wrap180(actual - self.aim_deg)
        self.aim_from = np.asarray(h.pos, dtype=float).copy()
        if abs(err) <= self.AIM_TOLERANCE_DEG:
            return

        # The same arithmetic as every other correction here: what was measured
        # is the ERROR between commanded and achieved, so it is subtracted from
        # what is already in force rather than assigned over it.
        # Before correcting: is this an error one number CAN cancel?
        self.aim_legs.append((self.aim_deg, err))
        if self.mirrored_frame():
            self.aim_fixes = self.MAX_AIM_FIXES     # stop; corrections cannot help
            self.say("error",
                     "the aim error changes SIGN with direction, so it is a "
                     "mirrored frame, not a rotated one. A heading offset is "
                     "one number added to every command — it cancels a "
                     "constant error and can never cancel one that flips. "
                     "The camera image is probably mirrored, which no amount "
                     "of looking at the picture can reveal. Press `flip y` on "
                     "the COLOUR tab, or re-pick the arena with the camera "
                     "un-mirrored.")
            self.stop_path("stopped — correcting a mirrored frame walks the "
                           "offset round the compass forever")
            return

        self.aim_fixes += 1
        before = h.heading_offset
        h.heading_offset = (h.heading_offset - h.HEADING_SIGN * err) % 360.0
        e = self.roster.by_code(h.code)
        if e is not None:
            e.heading_offset = h.heading_offset
            for x in self.roster.save():
                self.say("error", x)
        told = self.aim_deg
        self.pd.aim = None              # re-aim in the corrected frame
        self.aim_from = self.aim_deg = None
        self.say("ok", f"{h.code} was told {told:.0f}deg and drove "
                       f"{travelled:.0f}cm at {actual:.0f}deg — that is "
                       f"{err:+.0f}deg of aim-frame error. Offset {before:.0f} "
                       f"-> {h.heading_offset:.0f}deg "
                       f"({self.aim_fixes}/{self.MAX_AIM_FIXES})")


    def start_recovery(self, h):
        """Circling means the aim frame is wrong. Stop and measure it.

        Why the bench has to intervene rather than letting the live estimator
        handle it: `HeadingEstimator` rejects any sample spanning more than
        `max_turn` degrees of yaw, because travel direction measured across a
        turn is meaningless. A robot going in circles is turning constantly, so
        every sample is rejected and the estimator never becomes ready — it is
        structurally blind in exactly the failure it exists to correct. Fed a
        20-second circle it rejects 581 samples and reports no offset at all.

        Breaking the circle is what makes the measurement possible. A
        calibration leg holds ONE heading and never re-aims, so it travels in a
        straight line however wrong the frame is — it simply goes the wrong
        way, and that wrongness is the number we need. Two legs rather than one
        so that disagreement between them is visible: legs that disagree mean
        slipping, or a tracker following the wrong ball, and neither is fixed
        by rotating a frame.
        """
        from fleet.heading import ActiveCalibration

        self.recovered += 1
        self.recover = ActiveCalibration(headings=(0.0, 90.0),
                                         workspace=self.ws)
        h.stop()
        self.say("warn", f"{h.code} is circling, so the aim frame is wrong. "
                         f"Stopping to re-measure it "
                         f"({self.recovered}/{self.MAX_RECOVERIES}) — two short "
                         "legs, then the drive continues on its own.")

    def step_recovery(self, dt):
        """Drive the recovery calibration. Returns True while it owns the robot."""
        if self.recover is None:
            return False
        h = self.handle
        if h is None:
            self.recover = None
            return False
        if not h.connected:
            h.stop()
            return True                 # wait for the fix; the drive is paused

        v = self.recover.step(h.pos, dt)
        if v is not None and not self.recover.done:
            h.set_velocity(v)
            return True

        h.stop()
        cal, self.recover = self.recover, None

        if cal.error or cal.offset is None:
            self.stop_path(f"{h.code}: could not re-measure the aim frame — "
                           f"{cal.error or 'no usable legs'}")
            return True

        # The measured value is the ERROR between commanded and achieved, so it
        # is subtracted from what is already in force rather than assigned over
        # it. Assigning it doubles the fault instead of cancelling it, which is
        # a mistake this codebase has made once already.
        before = h.heading_offset
        h.heading_offset = (h.heading_offset - float(cal.offset)) % 360.0
        e = self.roster.by_code(h.code)
        if e is not None:
            e.heading_offset = h.heading_offset
            for x in self.roster.save():
                self.say("error", x)

        self.say("ok", f"{h.code} aim frame {before:.0f}deg -> "
                       f"{h.heading_offset:.0f}deg (measured error "
                       f"{cal.offset:.0f}deg, legs agree within "
                       f"{cal.spread:.0f}deg) — saved, resuming the drive")
        if cal.spread is not None and cal.spread > 25.0:
            self.say("warn", f"the two legs disagreed by {cal.spread:.0f}deg. "
                             "That is not a rotated frame — it is slipping, "
                             "being pushed, or the tracker following a "
                             "different ball. Check the blob count.")
        # The orbit detector has been watching a drive that was doomed by the
        # frame rather than by the gains. Clear its history so it judges the
        # corrected drive on its own evidence.
        if getattr(self.pd, "orbiting", None) is not None:
            self.pd._orbit_t = 0.0
            self.pd._orbit_dist = []
            self.pd.orbiting = False
        self._said_orbit = False
        self.trail = []
        return True

    def stop_path(self, note="drive stopped"):
        if self.path is None:
            return
        err = tracking_error(self.path, self.trail[len(self.trail) // 3:]) \
            if len(self.trail) > 30 and self.path.kind != "point" else None
        self.path, self.pd = None, None
        h = self.handle
        if h is not None:
            h.stop()
        if err:
            self.say("info", f"tracking error {err['rms_cm']}cm rms, "
                             f"{err['max_cm']}cm worst")
        self.say("warn", note)

    def click_arena(self, pos_cm):
        """A click on the board. What it builds depends on the shape selected."""
        h = self.handle
        if h is None:
            self.say("error", "connect a robot first")
            return
        if self.shape == "point":
            self.start_path(Point(pos_cm))
            return
        self.pending.append(np.asarray(pos_cm, dtype=float))
        if self.shape == "circle":
            # One click is the centre; the radius comes from the slider, so a
            # circle is a single click rather than a click-and-drag nobody can
            # do accurately with a trackpad.
            centre = self.pending.pop()
            self.start_path(Circle(centre, self.radius, self.path_speed)
                            .start_phase(h.pos))
        elif self.shape == "line" and len(self.pending) >= 2:
            a, b = self.pending[-2], self.pending[-1]
            self.pending = []
            if float(np.linalg.norm(b - a)) < 15.0:
                self.say("error", "those two points are too close for a line")
                return
            self.start_path(Line(a, b, self.path_speed))
        elif self.shape == "line":
            self.say("info", "click the other end of the line")

    def set_shape(self, name):
        self.shape, self.pending = name, []
        self.say("info", {"point": "click anywhere to send the robot there",
                          "line": "click two points to shuttle between them",
                          "circle": "click a centre to orbit it"}[name])
        self._build()

    def step_path(self, dt):
        if self.path is None or self.pd is None:
            return
        h = self.handle
        if h is None:
            return
        # A recovery owns the robot until it is finished, and the drive it
        # interrupted resumes from wherever the calibration legs left the ball.
        if self.step_recovery(dt):
            return
        if not h.connected:
            # STOP, not return. A Sphero holds its last speed command until it
            # is given another one, so a loop that merely stops updating leaves
            # the ball driving on whatever it was last told — for as long as the
            # fix stays lost, which is how a point-to-point move becomes a
            # robot circling the room. The battery already did this correctly;
            # the drive path did not, and returned instead.
            h.stop()
            self._lost_for += dt
            if self._lost_for > self.LOST_GIVE_UP_S:
                self.stop_path(f"{h.code}: no camera fix for "
                               f"{self.LOST_GIVE_UP_S:.0f}s — drive abandoned")
            elif not self._said_lost:
                self._said_lost = True
                self.say("warn", f"{h.code} lost its camera fix — stopped. It "
                                 "will carry on if the tracker finds it again.")
            return
        if self._lost_for:
            self._lost_for = 0.0
            self._said_lost = False
        setpoint, ff = self.path.step(dt, h.pos)
        v = self.pd.step(h.pos, h.vel, setpoint, feedforward=ff)
        # The same ceiling the battery drives under. A click near a wall should
        # ease into it, not arrive at it.
        top = getattr(self.pd, "max_speed", None) or getattr(self.pd, "speed", 60.0)
        ceiling = safety.speed_ceiling(self.ws, h.pos, self.stop_s(),
                                       safety=self.edge_margin,
                                       cap=min(top, self.cap_cm_s))
        speed = float(np.linalg.norm(v))
        if speed > ceiling > 0:
            v = v / speed * ceiling
        h.set_velocity(v)
        self.trail.append(np.asarray(h.pos, dtype=float).copy())
        del self.trail[:-TRAIL_LEN]
        if self.path.kind == "point" and self.pd.arrived:
            h.stop()
            if not self._said_arrival:
                self._said_arrival = True
                self.report_approach(h)
        self.watch_aim(h)
        if getattr(self.pd, "orbiting", False) and not self._said_orbit:
            self._said_orbit = True
            # Circling is the aim frame, not the tolerance — and it is
            # measurable right here rather than being somebody else's errand.
            # Telling a trainer to go and run the MOTION battery is a poor
            # answer when the bench is already holding the robot, already has
            # the camera, and needs two short legs to find the number.
            if self.recovered < self.MAX_RECOVERIES:
                self.start_recovery(h)
            else:
                self.stop_path(
                    f"{h.code} is still circling after "
                    f"{self.MAX_RECOVERIES} aim-frame corrections, so the aim "
                    "frame is not what is wrong. Check that BLOBS equals the "
                    "number of connected robots — a tracker following a "
                    "different ball produces exactly this, and no correction "
                    "to this robot's frame can fix it.")

    # -- the battery -----------------------------------------------------

    def start_run(self, quick=False):
        h = self.handle
        if h is None:
            self.say("error", "connect a robot first")
            return
        if not h.connected:
            self.say("error", f"{h.code} has no camera fix — every measurement "
                              "here is a difference of camera positions, so the "
                              "tracker must be seeing it. Auto-tune its colour.")
            return
        if self.run is not None:
            return
        e = self.entry
        self.run = Characterization(
            blob_count=self.blob_count,
            workspace=self.ws, code=h.code,
            ble_name=getattr(e, "ble_name", None),
            heading_offset=getattr(h, "heading_offset", 0.0), quick=quick,
            max_byte=int(round(self.cap_cm_s / 60.0 * 255)),
            plan_safety=self.edge_margin)
        self.run_code = h.code
        self.run_t = 0.0
        self.trail = []
        self.last_fit = None
        self.say("info", f"characterising {h.code}"
                         f"{' (quick)' if quick else ''} at up to "
                         f"{self.cap_cm_s}cm/s — keep the arena clear")

    def start_drift(self, minutes=5.0):
        """The measurement that decides whether heading needs estimating live."""
        h = self.handle
        if h is None or not h.connected:
            self.say("error", "connect a robot with a camera fix first")
            return
        if self.run is not None:
            return
        self.run = Characterization(workspace=self.ws, code=h.code,
                                    heading_offset=getattr(h, "heading_offset", 0.0),
                                    max_byte=int(round(self.cap_cm_s / 60.0 * 255)),
                                    plan_safety=self.edge_margin)
        self.run.stages = [Recenter(self.run._centre(), aim=self.run.aim),
                           DriftWatch(minutes=minutes, workspace=self.ws,
                                      aim=self.run.aim)]
        self.run.drift_only = True
        self.run_code = h.code
        self.run_t = 0.0
        self.trail = []
        h.heading_tracking = False
        self.say("info", f"watching {h.code} for {minutes:.0f} minutes — "
                         "keep the arena clear, this one takes a while")

    def report_drift(self, res):
        rw = res.get("random_walk_deg_per_sqrt_min")
        if rw is None:
            self.say("error", res.get("error", "not enough legs to measure drift"))
            return
        wander = res.get("expected_wander_30min_deg")
        self.say("ok", f"{res['legs']} legs over {res['minutes']:.1f}min: "
                       f"drift {rw} deg/sqrt(min)")
        if not res.get("above_noise_floor"):
            self.say("ok", f"that is at the measurement floor "
                           f"({res['noise_floor_deg_per_sqrt_min']}) — this ball "
                           "holds its calibration; a static offset is enough")
        elif wander and wander > 30:
            self.say("error", f"expect ~{wander:.0f}deg of wander in 30min. Path "
                              "following breaks past 45deg, so this needs a "
                              "continuously estimated heading")
        else:
            self.say("warn", f"expect ~{wander:.0f}deg of wander in 30min — "
                             "recalibrate between sessions, or estimate it live")

    def probe_sensors(self):
        """What this ball actually reports, and what asking costs in airtime.

        Started on a thread and collected in `drain_probe`. Doing it inline
        froze the window for the whole probe — tens of seconds of blocking
        radio reads — which is indistinguishable from a hang.
        """
        if self.probe is not None:
            self.say("warn", "a sensor probe is already running")
            return
        h = self.handle
        api = getattr(h, "_api", None) if h is not None else None
        if api is None:
            self.say("error", "the sensor probe needs a real robot with its "
                              "radio up — connect one as `real` first")
            return
        self.probe = SensorProbe(api)
        self.probe.start()
        self.say("info", f"probing {h.code}'s sensors — about half a minute, "
                         "and the ball will twitch. The window stays live.")
        self._build()

    def drain_probe(self):
        if self.probe is None or not self.probe.finished:
            return
        report, err = self.probe.report, self.probe.error
        self.probe = None
        self._build()
        if err or report is None:
            self.say("error", f"probe failed: {err or 'no report'}")
            return
        self.report_sensors(report)

    def report_sensors(self, report):
        for name, r in report["reads"].items():
            if not r.get("available"):
                self.say("warn", f"{name}: {r.get('error', 'no')}")
            elif not r.get("changing"):
                self.say("warn", f"{name}: answers in {r['ms_mean']}ms but never "
                                 "changes — a cached value, not a reading")
            else:
                self.say("ok", f"{name}: {r['ms_mean']}ms, up to {r['hz_ceiling']}Hz")
        base = (report["streaming"].get("0", {}).get("drive") or {}).get("ms_mean")
        for hz, entry in report["streaming"].items():
            d = entry.get("drive") or {}
            if d.get("ms_mean") and base:
                self.say("info", f"drive write at {hz}Hz streaming: "
                                 f"{d['ms_mean']}ms ({d['ms_mean'] / base:.1f}x)")
        # Reached from the render loop now, so a probe that came back without
        # a verdict must cost a line of output rather than the window.
        v = report.get("verdict") or {}
        if v.get("branch"):
            self.say("ok", f"design branch: {v['branch']} — {v.get('why', '')}")
        else:
            self.say("warn", "the probe returned no verdict — nothing answered "
                             "well enough to choose a design branch")
        self.last_probe = report
        return report

    def stop_run(self, note="stopped"):
        if self.run is not None:
            self.run.cancel()
            self.run = None
        h = self.fleet.handles.get(self.run_code)
        if h is not None:
            h.heading_tracking = True
        for h in self.fleet.handles.values():
            try:
                h.stop()
            except Exception:
                pass
        self.say("warn", note)

    def step_run(self, dt):
        if self.run is None:
            return
        h = self.fleet.handles.get(self.run_code)
        if h is None:
            self.stop_run("robot went away mid-run")
            return
        if not h.connected:
            # A lost fix mid-run does not end it — the tracker drops a frame
            # now and then — but the stage must not be fed a stale position as
            # if it were a fresh one.
            h.stop()
            return

        self.trail.append(np.asarray(h.pos, dtype=float).copy())
        del self.trail[:-TRAIL_LEN]

        # Containment first. If it wants the robot back inside, the stage does
        # not run at all this frame — its clock does not advance and it sees no
        # samples, so the recovery leaves no trace in the measurement.
        cmd, paused = self.contain(h, None)
        if paused:
            if cmd is None:
                h.stop()
            else:
                h.drive_raw(cmd[0], cmd[1])
            return

        self.run_t += dt
        cmd, _ = self.contain(h, self.run.step(self.run_t, h.pos, dt))
        if cmd is None:
            h.stop()
        else:
            h.drive_raw(cmd[0], cmd[1])

        if not self.run.done:
            return

        h.stop()
        h.heading_tracking = True
        run, self.run = self.run, None
        if run.cancelled:
            return
        if getattr(run, "aborted", False):
            for n in run.notes:
                self.say("error", n)
            blobs = 0
            try:
                blobs = len((self.tracker.latest()[1] or {}))
            except Exception:
                pass
            live = sum(1 for x in self.fleet.handles.values() if x.connected)
            if blobs > live:
                self.say("error", f"the camera is finding {blobs} blobs for "
                                  f"{live} connected robot(s). The extra ones "
                                  "are phantoms and the tracker can lock onto "
                                  "them. COLOUR tab -> auto-tune -> check hues.")
            out = self.tracker.outside_arena if self.tracker else {}
            if out:
                names = ", ".join(f"{c} x{n}" for c, n in sorted(out.items()))
                self.say("warn", f"also rejected {names} outside the arena — "
                                 "something in the room wears that hue. "
                                 "`optimise hues` picks colours the room "
                                 "leaves free.")
            self.say("error", "nothing was saved, and nothing drove far. Fix "
                              "the tracking first.")
            return
        if getattr(run, "drift_only", False):
            self.report_drift(run.results.get("heading_drift", {}))
            return
        fitted, err = run.save()
        self.last_fit = fitted
        if err:
            self.say("error", err)
        for n in run.notes:
            self.say("warn", n)
        self.apply_heading(run, h)
        self.report(fitted)

    def apply_heading(self, run, h):
        """Write the measured aim offset back to the roster."""
        head = run.results.get("heading_offset") or {}
        if head.get("offset_deg") is None or head.get("error"):
            return
        # The measured value is the ERROR between commanded and achieved, so it
        # is subtracted from what was already in force, never assigned over it.
        h.heading_offset = (h.heading_offset - float(head["offset_deg"])) % 360.0
        e = self.roster.by_code(h.code)
        if e is not None:
            e.heading_offset = h.heading_offset
            for x in self.roster.save():
                self.say("error", x)
        self.say("ok", f"{h.code} heading offset {h.heading_offset:.0f}deg "
                       f"(spread {head.get('spread_deg')}deg) — saved to roster")

    def report(self, fitted):
        r = (fitted or {}).get("recommend", {})
        self.say("ok", f"saved calib/motion.json for {fitted.get('robot')}")
        if r.get("handle_max_speed_cm_s"):
            sm = (fitted or {}).get("speed_map", {})
            if sm.get("max_speed_measured"):
                self.say("info", f"top speed {r['handle_max_speed_cm_s']}cm/s "
                                 "at byte 255")
            else:
                self.say("warn", f"top speed {r['handle_max_speed_cm_s']}cm/s is "
                                 f"EXTRAPOLATED from bytes up to "
                                 f"{sm.get('highest_measured_byte')} — the cap "
                                 "kept it from driving fast enough to measure "
                                 "the top of the curve")
        if r.get("min_moving_cm_s"):
            self.say("info", f"deadband: nothing moves below {r['min_moving_cm_s']}cm/s")
        if fitted.get("latency", {}).get("loop_delay_s"):
            self.say("info", f"loop delay {fitted['latency']['loop_delay_s']*1000:.0f}ms "
                             "(BLE + motor + camera + filter)")
        if r.get("max_precise_speed_cm_s"):
            self.say("warn", f"stop-inside-6cm speed limit: "
                             f"{r['max_precise_speed_cm_s']}cm/s — above that, "
                             f"widen SLOW_RADIUS to {r.get('slow_radius_cm')}cm")

    # -- layout ----------------------------------------------------------

    def set_tab(self, name):
        """Switch tabs AND rebuild. Returns the callback, for a Button.

        The two tabs do not share a layout — `motion_top` and `palette_rects`
        are each created by one branch of `_build` and read by the other tab's
        draw. Flipping `self.tab` without rebuilding therefore hands the new
        tab the old tab's geometry, which is not a cosmetic problem: the very
        first attribute it reaches for does not exist yet.
        """
        def go():
            if self.tab != name:
                self.tab = name
                self._build()
        return go

    def apply_size(self, w, h):
        global W, H
        W, H = max(MIN_W, w), max(MIN_H, h)
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.cam_surface, self.cam_stamp = None, None
        self._build()

    def _build(self):
        self.buttons, self.sliders = [], []
        # Cleared here, in the one place every rebuild passes through, rather
        # than in each tab's branch. Per-branch resets are how the motion tab
        # inherited the colour tab's geometry: adding a branch means
        # remembering to clear something in it, and the reminder is a crash.
        self.palette_rects = {}
        self.wheel_rect = None
        x, w = PAD, DOCK - 2 * PAD

        def add(rect, label, cb, tone=None):
            b = Button(rect, label, cb, tone=tone)
            self.buttons.append(b)
            return b

        # camera + scan, top of the dock
        y = 34
        add((x, y, 84, 24), "camera", lambda: self.start_camera(self.camera_spec))
        add((x + 92, y, 84, 24), "scan", self.start_scan)
        add((x + 184, y, 100, 24), "save cols", self.save_signatures)

        # one row per roster entry
        y = 150
        self.row_rects = {}
        for e in self.roster.entries:
            r = pygame.Rect(x, y, w, 26)
            self.row_rects[e.code] = r
            code = e.code
            if e.ble_name:
                add((r.right - 144, y + 2, 44, 22), "unbind",
                    lambda c=code: self.unbind(c), tone=SUN)
            add((r.right - 96, y + 2, 44, 22), "real",
                lambda c=code: self.connect(c, "real"))
            add((r.right - 48, y + 2, 44, 22), "sim",
                lambda c=code: self.connect(c, "sim"))
            y += 28
        self.roster_bottom = y

        # discovered Spheros, if any
        self.found_rects = {}
        if self.found:
            y += 26
            for name in self.found:
                self.found_rects[name] = pygame.Rect(x, y, w, 22)
                y += 24
        self.log_top = y + 26

        # the working area
        ax = DOCK + PAD
        aw = W - DOCK - 2 * PAD
        # The frame and its mask sit side by side, and the mask is the point of
        # tuning by hand — a slider whose effect you cannot see is worse than no
        # slider at all. So the panels are sized to the room the window has,
        # rather than to a constant that stops fitting below 1500px.
        cam_w = max(220, min(460, (aw - GAP) // 2))
        if cam_w != self.CAM_W:
            self.CAM_W = cam_w
            self.cam_surface, self.cam_stamp = None, None
        self.tab_rects = {}
        for i, name in enumerate(("colour", "motion", "drive")):
            b = add((ax + i * 110, 14, 104, 26), name.upper(), self.set_tab(name))
            b.on = (self.tab == name)
            self.tab_rects[name] = b

        if self.tab == "colour":
            sy = 14 + 26 + GAP + int(self.CAM_W * 0.75) + GAP + 26
            # The real size, not a placeholder the first draw corrects. Hit
            # boxes that only become right once something has been rendered
            # make click handling depend on paint order, and a click that lands
            # nowhere is indistinguishable from one that did nothing.
            self.cam_rect = pygame.Rect(ax, 14 + 26 + GAP, self.CAM_W,
                                        int(self.CAM_W * 0.75))
            specs = [("hue", 0, 179), ("tol", 2, 40), ("s_min", 0, 255),
                     ("v_min", 0, 255), ("min_area", 10, 1200),
                     # How hard to drive the LED. A ball is a light source in a
                     # dim room and a coloured object in a bright one, and the
                     # two want different amounts: run it too hot under strong
                     # lighting and the shell blows out to white, which has no
                     # hue for the detector to find.
                     ("led_value", 40, 255)]
            for i, (key, lo, hi) in enumerate(specs):
                self.sliders.append(Slider(
                    (ax, sy + i * 26, min(300, aw - 20), 16), key, lo, hi,
                    lambda k=key: self.get_sig(k, lo),
                    lambda v, k=key: self.set_slider(k, v)))
            by = sy + len(specs) * 26 + GAP
            add((ax, by, 150, 26), "auto-tune this",
                lambda: self.start_autotune(False), tone=CYAN)
            add((ax + 158, by, 150, 26), "auto-tune all",
                lambda: self.start_autotune(True), tone=MINT)
            add((ax + 316, by, 130, 26), "check hues", self.separation_report)
            add((ax + 454, by, 130, 26),
                "cancel" if self.corner_mode else "set arena",
                self.cancel_corners if self.corner_mode else self.start_corners,
                tone=SUN if self.corner_mode else None)
            add((ax + 730, by, 110, 26), "flip y", self.flip_arena_y, tone=SUN)
            add((ax + 592, by, 130, 26),
                "done" if self.checking else "check pos",
                self.finish_position_check if self.checking
                else self.start_position_check,
                tone=CYAN if self.checking else None)
            if self.corner_mode:
                cy = by + 26 + GAP
                self.sliders.append(Slider((ax + 316, cy, 260, 16),
                                           "width cm", 50, 400,
                                           lambda: self.arena_w,
                                           lambda v: setattr(self, "arena_w", v)))
                self.sliders.append(Slider((ax + 316, cy + 26, 260, 16),
                                           "height cm", 50, 400,
                                           lambda: self.arena_h,
                                           lambda v: setattr(self, "arena_h", v)))

            # Which hue this robot wears. Six swatches for reassigning which
            # SLOT it uses, and a wheel for choosing what that slot actually
            # is — the two are different questions and were being conflated.
            py = by + 26 + GAP + 20
            for i, cname in enumerate(vconfig.COLORS):
                self.palette_rects[cname] = pygame.Rect(ax + i * 54, py, 46, 34)
            self.palette_y = py

            wsize = min(230, max(150, H - py - 130))
            # Clear of the swatch strip by more than its own radius: the robot
            # codes are drawn OUTSIDE the ring, so a rect that merely does not
            # overlap still collides on screen.
            wx = max(ax + 6 * 54 + 60, self.cam_rect.right + GAP)
            # Anchored below the button row rather than beside the swatches:
            # the labels ride outside the ring, so "level with the swatches"
            # still clips whatever is above it.
            self.wheel_rect = pygame.Rect(wx, by + 26 + GAP + 16, wsize, wsize)
            if self.wheel_rect.right + 34 > W - PAD:
                self.wheel_rect.x = max(ax, W - PAD - wsize - 34)
            add((self.wheel_rect.x, self.wheel_rect.bottom + 30, 150, 26),
                "optimise hues", self.optimise_palette, tone=MINT)
        elif self.tab == "motion":
            by = 14 + 26 + GAP
            add((ax, by, 130, 28), "run full", lambda: self.start_run(False), tone=MINT)
            add((ax + 138, by, 130, 28), "run quick", lambda: self.start_run(True))
            add((ax + 276, by, 110, 28), "STOP",
                lambda: self.stop_run("run aborted"), tone=CORAL)
            add((ax + 396, by, 120, 28), "drift 5min", self.start_drift, tone=SUN)
            add((ax + 524, by, 120, 28),
                "probing..." if self.probe is not None else "sensors",
                self.probe_sensors,
                tone=CYAN if self.probe is not None else None)
            sy = by + 28 + GAP
            self.sliders.append(Slider((ax, sy, 300, 16), "top speed", 6,
                                       SPEED_CAP_CM_S,
                                       lambda: self.cap_cm_s,
                                       lambda v: setattr(self, "cap_cm_s", v)))
            self.sliders.append(Slider((ax, sy + 26, 300, 16), "edge margin",
                                       10, 60,
                                       lambda: int(self.edge_margin * 10),
                                       lambda v: setattr(self, "edge_margin",
                                                         v / 10.0)))
            self.motion_top = sy + 52 + GAP
        else:
            by = 14 + 26 + GAP
            for i, name in enumerate(("point", "line", "circle")):
                b = add((ax + i * 92, by, 86, 28), name,
                        lambda n=name: self.set_shape(n))
                b.on = (self.shape == name)
            add((ax + 402, by, 120, 28), "auto radius", self.seed_arrive)
            add((ax + 292, by, 100, 28), "STOP",
                lambda: self.stop_path("drive stopped"), tone=CORAL)
            b = add((ax + 530, by, 120, 28),
                    "straight" if self.drive_mode == "straight" else "PD loop",
                    self.toggle_drive_mode,
                    tone=MINT if self.drive_mode == "straight" else CYAN)
            b.on = True
            self.drive_top = by + 28 + GAP
            # The arena, square-ish and as large as the pane allows: this is
            # the thing being clicked, so it gets the room.
            x0, x1, y0, y1 = self.ws.bbox
            avail_w, avail_h = aw - 320, H - self.drive_top - 30
            scale = min(avail_w / max(x1 - x0, 1e-6), avail_h / max(y1 - y0, 1e-6))
            self.arena_scale = scale
            self.arena_rect = pygame.Rect(ax, self.drive_top,
                                          int((x1 - x0) * scale),
                                          int((y1 - y0) * scale))
            sx = self.arena_rect.right + GAP
            # Same ceiling the battery runs under. A path speed the limiter
            # will not allow is a slider that lies: the setpoint runs away from
            # the robot and the tracking error is blamed on the controller.
            self.sliders.append(Slider((sx, self.drive_top + 40, 280, 16),
                                       "speed", 5, SPEED_CAP_CM_S,
                                       lambda: self.path_speed,
                                       lambda v: setattr(self, "path_speed", v)))
            self.sliders.append(Slider((sx, self.drive_top + 68, 280, 16),
                                       "radius", 10, 90,
                                       lambda: self.radius,
                                       lambda v: setattr(self, "radius", v)))
            self.sliders.append(Slider((sx, self.drive_top + 96, 280, 16),
                                       "stop within", 2, 20,
                                       lambda: self.arrive_cm,
                                       lambda v: setattr(self, "arrive_cm", v)))

            # Hand tuning. Every one of these is applied to the RUNNING
            # controller as well as the next one — see `apply_tuning`. A knob
            # that only takes effect on the next drive is not a tuning knob,
            # it is a settings page, and tuning by stop-edit-start loses the
            # feel of what the change did.
            ty = self.drive_top + 124
            if self.drive_mode == "straight":
                self.sliders.append(Slider((sx, ty, 280, 16),
                                           "re-aim deg", 4, 45,
                                           lambda: int(self.tune_retarget),
                                           lambda v: self.set_tune("retarget", v)))
                self.sliders.append(Slider((sx, ty + 28, 280, 16),
                                           "re-aim ms", 100, 1200,
                                           lambda: int(self.tune_interval * 1000),
                                           lambda v: self.set_tune("interval",
                                                                   v / 1000.0)))
                self.sliders.append(Slider((sx, ty + 56, 280, 16),
                                           "creep %", 20, 100,
                                           lambda: int(self.tune_creep * 100),
                                           lambda v: self.set_tune("creep",
                                                                   v / 100.0)))
            else:
                self.sliders.append(Slider((sx, ty, 280, 16),
                                           "kp x100", 10, 400,
                                           lambda: int(self.tune_or_measured("kp") * 100),
                                           lambda v: self.set_tune("kp", v / 100.0)))
                self.sliders.append(Slider((sx, ty + 28, 280, 16),
                                           "kd x100", 0, 300,
                                           lambda: int(self.tune_or_measured("kd") * 100),
                                           lambda v: self.set_tune("kd", v / 100.0)))
                self.sliders.append(Slider((sx, ty + 56, 280, 16),
                                           "predict ms", 0, 700,
                                           lambda: int(self.tune_or_measured("predict") * 1000),
                                           lambda v: self.set_tune("predict",
                                                                   v / 1000.0)))
            # The readout starts below whatever the sliders came to, so adding
            # or removing one can never draw the gains through them again.
            self.drive_gains_top = ty + 56 + 16 + GAP + 8

    # -- drawing ---------------------------------------------------------

    def camera_surface(self):
        if self.tracker is None:
            return None, {}, 1.0
        try:
            frame, raw = self.tracker.latest()
        except Exception:
            return None, {}, 1.0
        if frame is None:
            return None, {}, 1.0
        h, w = frame.shape[:2]
        f = self.CAM_W / float(w)
        if id(frame) != self.cam_stamp:
            import cv2
            small = cv2.resize(frame, (self.CAM_W, max(1, int(h * f))))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            self.cam_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self.cam_stamp = id(frame)
        return self.cam_surface, raw, f

    def mask_surface(self, name):
        """The mask this colour's signature produces, as the trainer sees it."""
        d = self.detector
        if d is None or self.tracker is None:
            return None
        frame, _ = self.tracker.latest()
        if frame is None:
            return None
        import cv2
        b = int(d.thresh.get("blur", 5)) | 1
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (b, b), 0), cv2.COLOR_BGR2HSV)
        try:
            m = d.mask_for(hsv, name)
        except Exception:
            return None
        h, w = m.shape[:2]
        f = self.CAM_W / float(w)
        small = cv2.resize(m, (self.CAM_W, max(1, int(h * f))))
        rgb = np.dstack([small] * 3)
        tint = np.array(LED_RGB.get(name, (255, 255, 255)), dtype=np.float32) / 255.0
        rgb = (rgb.astype(np.float32) * tint).astype(np.uint8)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def draw_dock(self):
        s = self.screen
        pygame.draw.rect(s, PANEL, (0, 0, DOCK, H))
        pygame.draw.line(s, RULE, (DOCK, 0), (DOCK, H))
        x, w = PAD, DOCK - 2 * PAD

        section(s, self.fs, "bench", x, 12, w,
                self.camera_spec if self.camera_spec else "no camera")

        # telemetry
        fps = getattr(self.tracker, "fps", 0.0) or 0.0
        _, raw, _ = self.camera_surface()
        real = sum(1 for h in self.fleet.handles.values() if h.kind == "real")
        fixed = sum(1 for h in self.fleet.handles.values() if h.connected)
        ty, tw = 68, (w - 3 * 8) // 4
        for i, (label, value, tone) in enumerate((
                ("cam", f"{fps:.0f}fps", MINT if fps > 5 else GREY),
                ("blobs", str(len(raw or {})), MINT if raw else GREY),
                ("fleet", str(len(self.fleet.handles)), CHALK),
                ("fix", str(fixed), MINT if fixed else CORAL if real else GREY))):
            stat_tile(s, self.fs, self.f,
                      pygame.Rect(x + i * (tw + 8), ty, tw, 38), label, value, tone)

        section(s, self.fs, "robots", x, 126, w, "real / sim")
        for e in self.roster.entries:
            r = self.row_rects.get(e.code)
            if r is None:
                continue
            h = self.fleet.handles.get(e.code)
            if e.code == self.selected:
                card(s, r, fill=PANEL2, edge=CYAN)
            pygame.draw.circle(s, LED_RGB.get(e.color, CHALK), (r.x + 12, r.centery), 6)
            tone = CHALK if h else DIM
            s.blit(self.f.render(f"{e.code}", True, tone), (r.x + 26, r.y + 5))
            note = e.ble_name or "unbound"
            if h is not None:
                if h.kind == "sim":
                    note, tone = "sim", MINT
                elif h.connected:
                    note, tone = "linked+fix", MINT
                elif getattr(h, "link_up", False):
                    note, tone = "linked, no fix", SUN
                else:
                    note, tone = (h.last_error or "connecting…")[:18], CORAL
            s.blit(self.fs.render(note[:20], True, tone), (r.x + 78, r.y + 7))

        if self.found:
            y = self.roster_bottom + 6
            section(s, self.fs, "found", x, y, w, "click to bind")
            for name, r in self.found_rects.items():
                card(s, r)
                s.blit(self.f.render(name[:34], True, CHALK), (r.x + 8, r.y + 3))

        self.draw_log(x, w, self.log_top)

    def draw_log(self, x, w, top):
        s = self.screen
        section(s, self.fs, "log", x, top, w)
        self.log_rect = pygame.Rect(x, top + 24, w, H - top - 34)
        tones = {"ok": MINT, "error": CORAL, "warn": SUN, "info": DIM}
        lines = []
        for kind, text in self.log:
            wrapped = _wrap(text, 46)
            lines += [(kind, ln) for ln in wrapped]
        fit = max(1, self.log_rect.h // 15)
        end = len(lines) - self.log_scroll
        for i, (kind, ln) in enumerate(lines[max(0, end - fit):end]):
            s.blit(self.fs.render(ln, True, tones.get(kind, DIM)),
                   (x, self.log_rect.y + i * 15))

    def draw_colour(self):
        s = self.screen
        ax = DOCK + PAD
        surf, raw, f = self.camera_surface()
        name = self.sig_color
        y = 14 + 26 + GAP

        if surf is None:
            s.blit(self.f.render("no camera frame yet", True, GREY), (ax, y))
        else:
            self.cam_rect = self.draw_camera_panel(
                ax, y, caption="click the frame to sample a hue")
            mask = self.mask_surface(name)
            mx = ax + self.cam_rect.width + GAP
            if mask is not None and mx + mask.get_width() <= W - PAD:
                s.blit(mask, (mx, y))
                pygame.draw.rect(s, RULE,
                                 (mx, y, mask.get_width(), mask.get_height()), 1)
                s.blit(self.fs.render(f"{name} mask — what the signature selects",
                                      True, DIM), (mx, y - 14))

        # What the ball will actually glow, at the size a person can judge. The
        # wheel says which hue; this says what that hue looks like once the
        # brightness is applied, which is the part a number cannot convey.
        rgb = self.led_for(name)
        sw = pygame.Rect(ax + 320, y + int(self.CAM_W * 0.75) + 30, 92, 92)
        if sw.right < W - PAD:
            pygame.draw.rect(s, rgb, sw, border_radius=6)
            pygame.draw.rect(s, RULE, sw, 1, border_radius=6)
            spec = self.sig(name)
            lines = [f"hue {spec.get('hue', 0)}  +-{spec.get('tol', 8)}",
                     f"rgb {rgb[0]},{rgb[1]},{rgb[2]}",
                     f"clarity {self.hue_clarity(spec.get('hue', 0)):.2f}"]
            for i, ln in enumerate(lines):
                s.blit(self.fs.render(ln, True, DIM), (sw.x, sw.bottom + 6 + i * 14))

        hdr = f"signature: {name}"
        if self.autotune is not None:
            hdr = f"auto-tuning {self.autotune.progress}"
        s.blit(self.f.render(hdr, True, CYAN if self.autotune else CHALK),
               (ax, y + int(self.CAM_W * 0.75) + 6))
        for sl in self.sliders:
            sl.draw(s, self.f)

        if getattr(self, "palette_rects", None):
            e = self.entry
            taken = {x.color: x.code for x in self.roster.enabled_entries()}
            section(s, self.fs, "hue for " + (e.code if e else "—"),
                    ax, self.palette_y - 20, 340, "click to reassign")
            for cname, r in self.palette_rects.items():
                mine = e is not None and e.color == cname
                pygame.draw.rect(s, LED_RGB[cname], r.inflate(-6, -12),
                                 border_radius=3)
                pygame.draw.rect(s, CHALK if mine else RULE, r, 2 if mine else 1,
                                 border_radius=4)
                holder = taken.get(cname)
                if holder and not mine:
                    # Under the swatch, not on it: a code printed over a
                    # saturated colour is unreadable at half the hues.
                    s.blit(self.fs.render(holder[:4], True, DIM),
                           (r.x + 2, r.bottom + 2))

        if self.corner_mode:
            self.draw_corners()
        elif self.checking:
            self.draw_checks()
        elif getattr(self, "wheel_rect", None):
            self.draw_wheel()

        found_it = name in (raw or {})
        msg = (f"tracker sees {name}" if found_it else
               f"{name} not detected — light the ball and auto-tune")
        s.blit(self.f.render(msg, True, MINT if found_it else CORAL),
               (ax, H - 40))

    def draw_corners(self):
        """What has been clicked so far, and what to click next."""
        s = self.screen
        r = getattr(self, "cam_rect", None)
        surf, _, f = self.camera_surface()
        if r is None or surf is None:
            return
        pts = [(int(r.x + px * f), int(r.y + py * f)) for px, py in self.corners]
        if len(pts) > 1:
            pygame.draw.lines(s, SUN, len(pts) == 4, pts, 2)
        for i, pt in enumerate(pts):
            pygame.draw.circle(s, SUN, pt, 7, 2)
            pygame.draw.circle(s, SUN, pt, 2)
            t = self.fs.render(self.CORNER_LABELS[i], True, SUN)
            s.blit(t, (pt[0] + 10, pt[1] - 6))

        n = len(self.corners)
        msg = ("click " + self.CORNER_LABELS[n]) if n < 4 else "applying…"
        s.blit(self.f.render(f"{msg}   ({n}/4)", True, SUN), (r.x, r.bottom + 8))
        s.blit(self.fs.render(f"arena will be {self.arena_w} x {self.arena_h} cm "
                              "— set the size before the fourth click",
                              True, DIM), (r.x, r.bottom + 28))
        s.blit(self.fs.render("backspace undoes the last point · esc cancels",
                              True, GREY), (r.x, r.bottom + 44))
        s.blit(self.fs.render("click a ROBOT sitting on each corner, not the "
                              "floor mark", True, SUN), (r.x, r.bottom + 58))

    def draw_checks(self):
        """Each click, the tracker's answer, and the line between them."""
        s = self.screen
        r = getattr(self, "cam_rect", None)
        surf, _, f = self.camera_surface()
        if r is None or surf is None:
            return
        H = self.homography
        for c in self.checks:
            try:
                a = H.to_px([c["clicked"]])[0]
                b = H.to_px([c["reported"]])[0]
            except Exception:
                continue
            pa = (int(r.x + a[0] * f), int(r.y + a[1] * f))
            pb = (int(r.x + b[0] * f), int(r.y + b[1] * f))
            pygame.draw.circle(s, CHALK, pa, 5, 1)
            pygame.draw.circle(s, CORAL, pb, 4)
            pygame.draw.line(s, CORAL, pa, pb, 1)
        h = self.handle
        who = h.code if h is not None else "?"
        s.blit(self.f.render(f"click exactly where {who} really is  "
                             f"({len(self.checks)} so far)", True, CYAN),
               (r.x, r.bottom + 8))
        s.blit(self.fs.render("white ring = your click · red dot = the tracker · "
                              "esc or DONE to finish", True, DIM),
               (r.x, r.bottom + 28))

    def draw_wheel(self):
        """Every robot's hue on one ring, with the width it actually occupies.

        The arcs are the point. A hue is a number and two numbers look far
        apart at a glance however close their tolerance windows are — and it is
        the windows that decide whether the tracker can tell two robots apart.
        Drawing the width each colour claims makes an overlap something you see
        rather than something you find out about when two dragons swap names
        mid-run.
        """
        s = self.screen
        r = self.wheel_rect
        s.blit(self.wheel_surface(r.w), (r.x, r.y))
        c = np.array([r.centerx, r.centery], dtype=float)
        radius = r.w / 2.0

        hist = self.background()
        # The room's own colours, drawn inside the ring: wherever the floor and
        # the walls already sit, a robot is competing rather than standing out.
        if hist is not None and float(hist.max()) > 0:
            peak = float(hist.max())
            for hue in range(0, vpalette.HUES, 2):
                w = hist[hue] / peak
                if w < 0.06:
                    continue
                a = np.radians(hue * 2.0)
                r0 = radius - 34
                r1 = r0 - 26 * w
                pygame.draw.line(s, vconfig.led_rgb(hue),
                                 (c[0] + r0 * np.cos(a), c[1] + r0 * np.sin(a)),
                                 (c[0] + r1 * np.cos(a), c[1] + r1 * np.sin(a)), 2)

        sel = self.sig_color
        for e in self.roster.enabled_entries():
            spec = self.sig(e.color)
            hue = int(spec.get("hue", 0))
            tol = int(spec.get("tol", 8))
            mine = e.color == sel
            band = CHALK if mine else (150, 180, 205)
            for d in range(-tol, tol + 1):
                a = np.radians(((hue + d) % vpalette.HUES) * 2.0)
                r0, r1 = radius + 1, radius + (8 if mine else 5)
                pygame.draw.line(s, band,
                                 (c[0] + r0 * np.cos(a), c[1] + r0 * np.sin(a)),
                                 (c[0] + r1 * np.cos(a), c[1] + r1 * np.sin(a)), 2)
            a = np.radians(hue * 2.0)
            tip = c + (radius - 13) * np.array([np.cos(a), np.sin(a)])
            pygame.draw.circle(s, CHALK if mine else INK, tuple(map(int, tip)),
                               7 if mine else 5)
            pygame.draw.circle(s, vconfig.led_rgb(hue), tuple(map(int, tip)),
                               5 if mine else 3)
            lab = c + (radius + 20) * np.array([np.cos(a), np.sin(a)])
            t = self.fs.render(e.code, True, CHALK if mine else DIM)
            s.blit(t, t.get_rect(center=tuple(map(int, lab))))

        entries, summary = self.palette_report()
        room = getattr(self, "_room", None)
        y = r.bottom + 68              # clear of the button beneath the ring
        sep, cla = summary["worst_separation"], summary["worst_clarity"]
        s.blit(self.fs.render("W O R S T  P A I R", True, (120, 156, 186)),
               (r.x, y))
        y += 16
        for label, value, good in (
                ("separation", f"{sep}" if sep is not None else "—",
                 sep is not None and sep >= vpalette.MIN_SEPARATION),
                ("clarity", f"{cla:.2f}" if cla is not None else "—",
                 cla is not None and cla > 0.5)):
            s.blit(self.f.render(label, True, DIM), (r.x, y))
            s.blit(self.fb.render(value, True, MINT if good else CORAL),
                   (r.x + 100, y - 2))
            y += 22
        if room:
            for ln in _wrap(f"room: {room['verdict']}", 34):
                s.blit(self.fs.render(ln, True, SUN if "workable" not in
                                      room["verdict"] else GREY), (r.x, y))
                y += 14

    def draw_motion(self):
        s = self.screen
        ax = DOCK + PAD
        aw = W - DOCK - 2 * PAD
        y = self.motion_top

        # The camera lives here too. During a run it is the only thing that
        # distinguishes "the ball is doing the legs" from "the tracker latched
        # onto a reflection and every number coming back is fiction" — and the
        # numbers alone cannot tell you which, because a mistracked run fits
        # perfectly well to the wrong ball.
        cam_x = ax + aw - self.CAM_W
        cap = None
        if self.run is not None and self.run.stage is not None:
            cap = f"{self.run.stage.label} — {self.run.stage.progress}"
        cam = self.draw_camera_panel(cam_x, y, trail=self.trail,
                                     caption=cap or "camera")
        aw = cam_x - ax - GAP

        # The pace controls, and what they mean in centimetres per second.
        # `row` is NOT called `y`: it was, and it shadowed the layout cursor —
        # every line below drew at y=1 instead of below the buttons, which is
        # why the whole tab came out stacked in the top corner.
        for sl in self.sliders:
            sl.draw(s, self.f)
        stop = self.stop_s()
        for row, d in enumerate((20.0, 40.0)):
            v = safety.speed_ceiling(self.ws, (d, d), stop,
                                     safety=self.edge_margin, cap=self.cap_cm_s)
            s.blit(self.fs.render(f"{d:.0f}cm from a wall -> {v:.0f}cm/s max",
                                  True, DIM),
                   (ax + 330, self.motion_top - 68 + row * 16))

        h = self.handle
        who = f"{self.selected}" + (f" · {h.kind}" if h else " · not connected")
        y = section(s, self.fs, "subject", ax, y, aw, who,
                    MINT if h and h.connected else GREY)

        if self.run is not None:
            st = self.run.stage
            card(s, pygame.Rect(ax, y, aw, 58), fill=PANEL2, edge=CYAN)
            s.blit(self.fb.render(st.label if st else "finishing", True, CHALK),
                   (ax + 12, y + 8))
            s.blit(self.f.render(st.detail if st else "", True, DIM), (ax + 12, y + 32))
            frac = self.run.i / max(len(self.run.stages), 1)
            bw = min(176, max(60, aw - 220))
            bar = pygame.Rect(ax + aw - bw - 14, y + 20, bw, 8)
            pygame.draw.rect(s, RULE, bar, border_radius=4)
            pygame.draw.rect(s, CYAN, (bar.x, bar.y, int(bar.w * frac), bar.h),
                             border_radius=4)
            s.blit(self.fs.render(f"{self.run.i + 1}/{len(self.run.stages)}",
                                  True, DIM), (bar.x, y + 34))
            y += 58 + GAP
            y = self._draw_results(ax, aw, y, self.run.results)
            return

        if self.last_fit is None:
            prev = load_motion(self.selected)
            if prev:
                s.blit(self.f.render("last run — press RUN FULL to remeasure",
                                     True, DIM), (ax, y))
                self._draw_fit(ax, aw, y + 22, prev)
            else:
                for i, line in enumerate((
                        "Nothing measured for this robot yet.",
                        "",
                        "1.  connect it  (real, or sim to rehearse the rig)",
                        "2.  COLOUR tab, auto-tune, until the tracker sees it",
                        "3.  put it near the middle of the arena",
                        "4.  RUN FULL — about three minutes, hands off",
                        "",
                        "It measures position noise, the heading offset, the",
                        "speed byte to cm/s map, the motor lag, the coast, and",
                        "the loop delay; then writes calib/motion.json.")):
                    s.blit(self.f.render(line, True, DIM if i > 1 else CHALK),
                           (ax, y + i * 20))
            return
        self._draw_fit(ax, aw, y, self.last_fit)

    def _draw_results(self, x, w, y, results):
        for key, res in results.items():
            label = key.replace("_", " ")
            tone = CORAL if res.get("error") else MINT
            s = ", ".join(f"{k}={v}" for k, v in res.items()
                          if k not in ("rows", "runs", "error") and v is not None)
            self.screen.blit(self.f.render(label, True, tone), (x, y))
            width = max(24, (w - 150) // 7)
            lines = _wrap(s or res.get("error", ""), width)
            for i, ln in enumerate(lines):
                self.screen.blit(self.fs.render(ln, True, DIM), (x + 150, y + i * 14))
            y += max(20, 14 * len(lines) + 6)
        return y

    def _draw_fit(self, x, w, y, fit):
        s = self.screen
        rec = fit.get("recommend", {})
        s.blit(self.fs.render(f"measured {fit.get('measured_at', '?')}", True, GREY),
               (x, y))
        y += 18

        rows = [
            ("top speed", rec.get("handle_max_speed_cm_s"), "cm/s at byte 255"),
            ("deadband", rec.get("min_moving_cm_s"), f"cm/s (byte {rec.get('min_moving_byte')})"),
            ("motor lag", fit.get("step_response", {}).get("tau_s"), "s"),
            ("loop delay", fit.get("latency", {}).get("loop_delay_s"), "s, BLE+motor+camera"),
            ("coast @45", fit.get("brake", {}).get("coast_at_45_cm"), "cm"),
            ("position noise", fit.get("position_noise", {}).get("sigma_cm"), "cm sigma"),
            ("heading offset", fit.get("heading", {}).get("offset_deg"), "deg"),
        ]
        for label, value, unit in rows:
            s.blit(self.f.render(label, True, DIM), (x, y))
            s.blit(self.fb.render("—" if value is None else str(value), True,
                                  CHALK if value is not None else GREY), (x + 160, y - 2))
            s.blit(self.fs.render(unit, True, GREY), (x + 250, y + 2))
            y += 24

        y += GAP
        y = section(s, self.fs, "what to change", x, y, w)
        lines = []
        if rec.get("max_precise_speed_cm_s"):
            lines.append((f"speed limit for a clean stop: "
                          f"{rec['max_precise_speed_cm_s']}cm/s", SUN))
        if rec.get("slow_radius_cm"):
            lines.append((f"at 45cm/s cruise, SLOW_RADIUS should be "
                          f"{rec['slow_radius_cm']}cm  ({rec.get('slow_radius_driver')})",
                          CHALK))
        if rec.get("handle_max_speed_cm_s"):
            lines.append((f"fleet/handle.py MAX_SPEED = {rec['handle_max_speed_cm_s']}",
                          CHALK))
        if rec.get("sim_tau_s"):
            lines.append((f"sim_handle: tau={rec['sim_tau_s']}s  "
                          f"latency={rec.get('sim_latency_steps')} steps  "
                          f"gain={rec.get('sim_gain')}", CHALK))
        if rec.get("kalman_r"):
            lines.append((f"vision/track.py Kalman2D r = {rec['kalman_r']}", CHALK))
        stop = rec.get("stopping_distance_cm") or {}
        if stop:
            lines.append(("stopping distance: " + "  ".join(
                f"{k}cm/s->{v}cm" for k, v in stop.items()), DIM))
        for text, tone in lines:
            for ln in _wrap(text, max(30, w // 8)):
                s.blit(self.f.render(ln, True, tone), (x, y))
                y += 19

    def to_px(self, p_cm):
        x0, x1, y0, y1 = self.ws.bbox
        r, k = self.arena_rect, self.arena_scale
        return (int(r.x + (p_cm[0] - x0) * k), int(r.y + (p_cm[1] - y0) * k))

    def to_cm(self, px):
        x0, x1, y0, y1 = self.ws.bbox
        r, k = self.arena_rect, self.arena_scale
        return np.array([x0 + (px[0] - r.x) / k, y0 + (px[1] - r.y) / k])

    def draw_drive(self):
        """A top-down board in centimetres, because that is what a click means.

        Not the camera image: a click has to become an arena coordinate, and
        going through the homography for that would put a robot the camera
        cannot see — a simulated one — somewhere it is not. The camera view
        lives on the other two tabs; this one is the planner's frame.
        """
        s, r = self.screen, self.arena_rect
        card(s, r, fill=(9, 28, 47), edge=RULE)
        for i in range(1, 5):
            gx = r.x + r.w * i // 5
            gy = r.y + r.h * i // 5
            pygame.draw.line(s, (22, 52, 78), (gx, r.y), (gx, r.bottom))
            pygame.draw.line(s, (22, 52, 78), (r.x, gy), (r.right, gy))
        for o in getattr(self.ws, "obstacles", []) or []:
            try:
                if o.get("type") == "circle":
                    pygame.draw.circle(s, CORAL, self.to_px(o["center"]),
                                       max(2, int(o["radius"] * self.arena_scale)), 1)
            except Exception:
                pass

        if self.path is not None:
            pts = [self.to_px(p) for p in self.path.preview(120)]
            if self.path.kind == "point":
                # The aura, at its real size. A target drawn as a dot invites
                # the assumption that the dot is what the robot is chasing, and
                # then any stop short of it looks like a failure rather than
                # the tolerance doing its job.
                rr = max(3, int(self.arrive_cm * self.arena_scale))
                inside = (self.pd is not None and self.pd.holding)
                pygame.draw.circle(s, MINT if inside else SUN, pts[0], rr, 1)
                pygame.draw.circle(s, SUN, pts[0], 2)
            else:
                pygame.draw.lines(s, (60, 110, 150), self.path.kind == "circle",
                                  pts, 1)
            sp, _ = self.path.step(0.0)
            pygame.draw.circle(s, SUN, self.to_px(sp), 5, 2)
        for pending in self.pending:
            pygame.draw.circle(s, DIM, self.to_px(pending), 5, 1)

        if len(self.trail) > 1:
            pygame.draw.lines(s, CYAN, False, [self.to_px(p) for p in self.trail], 2)

        for code, h in self.fleet.handles.items():
            px = self.to_px(h.pos)
            col = LED_RGB.get(h.color, CHALK)
            pygame.draw.circle(s, col, px, 7)
            if h.kind == "real":
                pygame.draw.circle(s, CHALK, px, 10, 1)
            if code == self.selected:
                pygame.draw.circle(s, CHALK, px, 13, 1)
            s.blit(self.fs.render(code, True, DIM), (px[0] + 12, px[1] - 6))

        # the readout
        x = r.right + GAP
        y = self.drive_top
        g = self.gains()
        s.blit(self.f.render(self.path.describe() if self.path else "idle",
                             True, CHALK if self.path else GREY), (x, y))
        for sl in self.sliders:
            sl.draw(s, self.f)

        y = getattr(self, "drive_gains_top", r.y + 110)
        y = section(s, self.fs, "gains", x, y, 300,
                    "measured" if g["measured"] else "defaults",
                    MINT if g["measured"] else SUN)
        rows = [("kp", f"{g['kp']:.2f}", "1/s"),
                ("kd", f"{g['kd']:.2f}", "s"),
                ("predict", f"{g['predict_s'] * 1000:.0f}", "ms ahead"),
                ("deadband", f"{g['deadband_cm_s']:.1f}", "cm/s"),
                ("ceiling", f"{g['max_speed']:.0f}", "cm/s")]
        for label, value, unit in rows:
            s.blit(self.f.render(label, True, DIM), (x, y))
            s.blit(self.fb.render(value, True, CHALK), (x + 110, y - 2))
            s.blit(self.fs.render(unit, True, GREY), (x + 180, y + 2))
            y += 22
        if not g["measured"]:
            for ln in _wrap("This robot has not been characterised, so it is "
                            "driving on conservative defaults with no delay "
                            "compensation. Run the battery on the MOTION tab.",
                            42):
                s.blit(self.fs.render(ln, True, SUN), (x, y))
                y += 14

        y += GAP
        if self.pd is not None and self.pd.error is not None:
            y = section(s, self.fs, "tracking", x, y, 300)
            s.blit(self.f.render(f"error {self.pd.error:5.1f} cm", True,
                                 MINT if self.pd.error < 6 else SUN), (x, y))
            y += 20
            if self.path.kind != "point" and len(self.trail) > 30:
                e = tracking_error(self.path, self.trail[len(self.trail) // 3:])
                if e:
                    s.blit(self.f.render(f"path  {e['rms_cm']:5.1f} cm rms",
                                         True, CHALK), (x, y))
                    y += 20
                    s.blit(self.fs.render(f"worst {e['max_cm']} cm", True, DIM),
                           (x, y))

    # -- interaction -----------------------------------------------------

    def click(self, pos):
        for sl in self.sliders:
            if sl.hit(pos):
                self.dragging_slider = sl
                return
        for b in self.buttons:
            if b.hit(pos):
                return
        if getattr(self, "wheel_rect", None) and self.wheel_rect.collidepoint(pos):
            hue = self.wheel_hue(pos)
            if hue is not None:
                self.set_hue(hue)
                return
        for cname, r in getattr(self, "palette_rects", {}).items():
            if r.collidepoint(pos):
                self.set_color(cname)
                return
        for code, r in self.row_rects.items():
            if r.collidepoint(pos):
                # The swatch itself cycles the hue; the rest of the row selects.
                if pos[0] < r.x + 24:
                    self.cycle_color(code)
                else:
                    self.selected = code
                    self._build()
                return
        for name, r in self.found_rects.items():
            if r.collidepoint(pos):
                self.bind_ble(name)
                return
        if self.tab == "colour" and getattr(self, "cam_rect", None) \
                and self.cam_rect.collidepoint(pos):
            local = (pos[0] - self.cam_rect.x, pos[1] - self.cam_rect.y)
            if self.corner_mode:
                self.add_corner(local)
            elif self.checking:
                self.add_position_check(local)
            else:
                self.sample_hue_at(local)
        elif self.tab == "drive" and getattr(self, "arena_rect", None) \
                and self.arena_rect.collidepoint(pos):
            self.click_arena(self.to_cm(pos))

    def key(self, e):
        if e.key == pygame.K_BACKSPACE and self.corner_mode:
            self.undo_corner()
        elif e.key == pygame.K_ESCAPE:
            if self.checking:
                self.finish_position_check()
            elif self.corner_mode:
                self.cancel_corners()
            elif self.run is not None:
                self.stop_run("run aborted")
            elif self.autotune is not None:
                self.autotune = None
                self.say("warn", "auto-tune aborted")
        elif e.key == pygame.K_TAB:
            order = ("colour", "motion", "drive")
            self.set_tab(order[(order.index(self.tab) + 1) % len(order)])()
        elif e.key == pygame.K_SPACE:
            self.stop_path("all stop")
            self.stop_run("all stop")
        elif e.key == pygame.K_s:
            self.save_signatures()
        elif e.key == pygame.K_PAGEUP:
            self.log_scroll += 8
        elif e.key == pygame.K_PAGEDOWN:
            self.log_scroll = max(0, self.log_scroll - 8)
        elif e.key == pygame.K_END:
            self.log_scroll = 0

    def step(self, dt):
        self.drain_scan()
        self.drain_probe()
        for h in list(self.fleet.handles.values()):
            try:
                h.step(dt)
            except Exception as e:
                self.say("error", f"{h.code}: {e}")
        if self.autotune is not None:
            frame, _ = self.tracker.latest() if self.tracker else (None, {})
            self.autotune.step(frame, dt)
        self.step_run(dt)
        self.step_path(dt)

    def run_loop(self):
        running = True
        while running:
            dt = 1.0 / 30.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 3 and self.corner_mode:
                        self.undo_corner()
                    else:
                        self.click(e.pos)
                elif e.type == pygame.MOUSEMOTION and self.dragging_slider:
                    self.dragging_slider.drag(e.pos)
                elif e.type == pygame.MOUSEBUTTONUP:
                    self.dragging_slider = None
                elif e.type == pygame.KEYDOWN:
                    self.key(e)
                elif e.type == pygame.VIDEORESIZE:
                    self.apply_size(e.w, e.h)
                elif e.type == pygame.MOUSEWHEEL:
                    if getattr(self, "log_rect", None) and \
                            self.log_rect.collidepoint(pygame.mouse.get_pos()):
                        self.log_scroll = max(0, self.log_scroll + e.y * 3)

            self.step(dt)
            self.screen.fill(INK)
            self.draw_dock()
            for b in self.buttons:
                b.draw(self.screen, self.f)
            {"colour": self.draw_colour,
             "motion": self.draw_motion,
             "drive": self.draw_drive}[self.tab]()
            pygame.display.flip()
            self.clock.tick(30)

        self.stop_path("closing")
        self.stop_run("closing")
        self.fleet.close()
        if self.tracker is not None:
            self.tracker.stop()
        pygame.quit()


def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="synthetic",
                   help="camera index, video path, or 'synthetic'")
    p.add_argument("--dry", action="store_true",
                   help="rehearse: sim robots run the whole battery, no hardware")
    p.add_argument("--roster", default=None)
    p.add_argument("--workspace", default=None)
    a = p.parse_args()
    CalibApp(camera=a.camera, dry=a.dry, roster_path=a.roster,
             workspace_path=a.workspace).run_loop()


if __name__ == "__main__":
    main()
