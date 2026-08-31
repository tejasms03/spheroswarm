#!/usr/bin/env python3
"""Tracker test — can reading the lights beat inferring the heading?

    python tracker_test.py                 # five balls drifting, live
    python tracker_test.py --bots 3 --seed 4
    python tracker_test.py shots/          # or step through stills instead

Balls wander at the slowest speed their motors will accept, with every
unreliability the simulator models: a per-ball speed gain, a heading bias, a
gyro that drifts, slip, and commands that land a fifth of a second late. They
are not meant to be accurate. They are meant to be as untrustworthy as the real
thing, so the question underneath can be answered honestly:

    Does reading the two lights give a heading sooner, and can you trust it?

So three headings are shown at once for every ball.

    LIGHTS   from the dots and their colour, this frame. Needs no history, no
             movement, and no assumption that the ball went where it pointed.
    TRAVEL   from where the ball actually moved over the last stretch. This is
             what the tracker does today. It needs a baseline before it means
             anything, it does not exist at all while the ball is stopped, and
             slip makes it wrong rather than merely late.
    TRUE     what the simulator knows. Neither method sees this.

Watch the error columns while a ball turns, stops, or slips. That gap is the
entire argument for the taillight.

Keys — 1/2/3 switch view, space pause, t truth arrows, k kick a ball,
r reset, left/right step stills, esc quit.
"""

import argparse
import glob
import math
import os
from collections import deque

import cv2
import numpy as np
import pygame

from fleet.heading import DEFAULT_MAX_TURN
from fleet.sim_handle import SimRobot
from swarm.trace import RollPlant, SpeedMap, heading_vector, wrap180
from ui.theme import (CHALK, CORAL, CYAN, DIM, GAP, GREY, INK, MINT, PAD,
                      PANEL, RULE, SUN, Button, Slider, card, section)
from vision.dots import read_all
from vision.shots import (BALL_CM, TAG_R_CM, TAGS, TAIL_R_CM, as_exposed,
                          as_gated, as_short, draw_ball)
from workspace.space import Workspace

W, H = 1440, 920
MIN_W, MIN_H = 1150, 700
DOCK = 360
PX_CM = 6.5             # what 1280x720 delivers over this arena; the real case
VIEWS = ("short", "exposure", "gate")
SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEADBAND_BYTE = 18      # the slowest byte the measured ball moves at
TRAVEL_CM = 8.0         # baseline the travel method needs before it will answer


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


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += [os.path.join(p, n) for n in sorted(os.listdir(p))
                    if n.lower().endswith(SUFFIXES)]
        elif any(c in p for c in "*?["):
            out += sorted(g for g in glob.glob(p) if g.lower().endswith(SUFFIXES))
        elif p.lower().endswith(SUFFIXES):
            out.append(p)
    return out


def to_surface(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")


def compass(image_deg):
    """Image degrees (x right, y down) as a Sphero heading (0 is +y, clockwise)."""
    return (90.0 - float(image_deg)) % 360.0


class Drifter:
    """One ball, wandering, with all the ways a real one misbehaves.

    Nothing here is tuned to be realistic in the sense of matching a particular
    floor. It is realistic in the sense that matters for this question: the
    heading it is COMMANDED, the heading it is POINTING and the direction it
    actually TRAVELS are three different numbers, and they come apart exactly
    where a tracker is most likely to believe them.
    """

    TURN_EVERY = (2.5, 6.0)         # s between course changes
    TURN_BY = 55.0                  # deg, at most, per change
    WALL_CM = 18.0                  # turn away inside this margin
    APART_CM = 22.0                 # steer away inside this, see `_steer`
    SKID_S = 2.4                    # how long a kick keeps it off the floor
    SKID_AWAY = (55.0, 115.0)       # deg between where it points and where it goes
    SKID_MIN_CM_S = 16.0            # a slide slow enough to miss is not a demo

    def __init__(self, code, tag, ws, speeds, seed, yaw_rate=180.0):
        self.code, self.tag = code, tag
        self.rng = np.random.default_rng(seed)
        self.robot = SimRobot(code, code, tag, workspace=ws, seed=seed,
                              randomize=True)
        self.robot.pos = np.asarray(ws.random_valid_point(self.rng), dtype=float)
        self.plant = RollPlant(self.robot, speeds=speeds, yaw_rate=yaw_rate)
        self.plant.heading = float(self.rng.uniform(0, 360))
        self.want = self.plant.heading
        self.until = float(self.rng.uniform(*self.TURN_EVERY))
        self.byte = DEADBAND_BYTE
        self.trail = deque(maxlen=180)          # (pos, t) for the travel method
        self.t = 0.0
        self.ws = ws
        self.skid_v = np.zeros(2)
        self.skid_until = 0.0
        self.base_slip = float(self.robot.slip)

    @property
    def skidding(self):
        return self.t < self.skid_until

    def kick(self):
        """Lose the floor: keep pointing one way, travel another.

        `SimRobot.slip` alone cannot show this and it is worth saying why. It
        multiplies the commanded velocity down -- the ball goes SLOWER along
        exactly the course it was driving -- so heading and travel stay
        agreed and there is nothing to see. Real slip is a direction failure,
        not a speed one: the shell breaks traction and the ball carries on
        across its own nose.

        That is the case the two methods disagree about. The lights keep
        reporting where the drive assembly points, because that is what they
        measure. Travel reports where the ball is sliding, confidently and
        wrongly, because travel has no way to know the difference.
        """
        away = float(self.rng.choice([-1.0, 1.0])) * float(
            self.rng.uniform(*self.SKID_AWAY))
        speed = max(float(np.linalg.norm(self.robot.vel)), self.SKID_MIN_CM_S)
        self.skid_v = heading_vector(self.plant.heading + away) * speed
        self.skid_until = self.t + self.SKID_S

    def _steer(self, dt, others=()):
        self.until -= dt
        x0, x1, y0, y1 = self.ws.bbox
        x, y = float(self.robot.pos[0]), float(self.robot.pos[1])

        # Keep them off each other. Not politeness: two balls closer than a
        # diameter are ONE blob to the detector, and every method reads that
        # blob as a single robot pointing somewhere between them. Left to
        # wander freely they collided constantly, and 38% of readings came back
        # over ten degrees out -- which measures the collisions, not the
        # method. Real robots are kept apart by a controller for the same
        # reason, so a test that lets them pile up is testing the wrong thing.
        close = None
        for other in others:
            gap = other - self.robot.pos
            d = float(np.linalg.norm(gap))
            if d < self.APART_CM and (close is None or d < close[0]):
                close = (d, gap)
        if close is not None:
            away = -close[1]
            self.want = float(np.degrees(np.arctan2(away[0], away[1])) % 360.0)
            self.until = max(self.until, 0.8)
            return

        near = (x - x0 < self.WALL_CM or x1 - x < self.WALL_CM
                or y - y0 < self.WALL_CM or y1 - y < self.WALL_CM)
        if near:
            # Point back at the middle rather than reflecting: a ball that
            # bounces off a wall it cannot feel looks like a simulation.
            mid = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            to = mid - self.robot.pos
            self.want = float(np.degrees(np.arctan2(to[0], to[1])) % 360.0)
            self.until = float(self.rng.uniform(*self.TURN_EVERY))
        elif self.until <= 0.0:
            self.want = (self.want
                         + float(self.rng.uniform(-self.TURN_BY, self.TURN_BY))) % 360.0
            self.until = float(self.rng.uniform(*self.TURN_EVERY))

    def step(self, dt, others=()):
        self.t += dt
        self._steer(dt, others)
        self.plant.roll(self.want, self.byte)

        if self.skidding:
            # The drive barely bites while it is sliding, and the slide itself
            # is applied to the POSITION -- the ball is being carried, not
            # driven, so it does not belong in the motor model.
            self.robot.slip = 0.95
            self.plant.step(dt)
            self.robot.pos = np.asarray(self.robot.pos, dtype=float) + self.skid_v * dt
            self.skid_v = self.skid_v * float(np.exp(-dt / 1.8))     # friction
            if not self.ws.is_valid_point(self.robot.pos):
                self.robot.pos = np.asarray(
                    self.ws.nearest_valid_point(self.robot.pos), dtype=float)
                self.skid_until = 0.0
        else:
            self.robot.slip = self.base_slip
            self.plant.step(dt)

        # NOTE: the trail is no longer filled here. It is filled by `observe`,
        # from what the tracker actually saw -- see the note there.

    def observe(self, pos_cm):
        """One camera fix, or None if the tracker lost it this frame.

        The travel method has to be built from THIS and not from where the
        simulator knows the ball is. Feeding it true positions was quietly
        rigging the comparison in its favour: it got a noiseless, gap-free
        history that no tracker has, while the lights method was made to work
        from the image like a real one. A method that needs position history
        inherits every error in that history, and that is part of its cost.
        """
        if pos_cm is None:
            return
        self.trail.append((np.asarray(pos_cm, dtype=float), self.t,
                           self.plant.heading))

    # -- what the two methods are compared against -----------------------

    @property
    def truth(self):
        """Where it is and where it POINTS. Only the simulator knows this."""
        return {"pos": np.array(self.robot.pos, dtype=float),
                "heading": self.plant.heading}

    def travel_heading(self):
        """Heading from where it went, which is what the tracker does today.

        Returns (deg, cm) or None. The centimetres are the baseline it found --
        a short one is a noisy answer, and no baseline at all is the honest
        state of a ball that has not moved far enough yet.

        The turn guard is `fleet/heading.py`'s and it is not optional. A
        baseline spanning a turn is a CHORD, and its direction is nobody's
        heading -- so the real estimator refuses those rather than answering
        wrongly. Leaving the guard out makes this comparison look far better
        for the lights than it honestly is: without it the travel method
        answered 91% of frames with a median error of 83 degrees, which is not
        the incumbent method failing, it is a strawman of it.
        """
        if len(self.trail) < 2:
            return None
        last, _, h_last = self.trail[-1]
        for pos, _, h_then in reversed(self.trail):
            if abs(wrap180(h_last - h_then)) > DEFAULT_MAX_TURN:
                return None             # it turned; the chord means nothing
            gap = float(np.linalg.norm(last - pos))
            if gap >= TRAVEL_CM:
                step = last - pos
                return (float(np.degrees(np.arctan2(step[0], step[1])) % 360.0),
                        gap)
        return None

    def render_state(self, px_cm):
        return {"x_cm": float(self.robot.pos[0]), "y_cm": float(self.robot.pos[1]),
                "heading_deg": (90.0 - self.plant.heading) % 360.0,
                "tag": self.tag}


class LiveArena:
    """The balls, a cached backdrop, and the three ways of reading a frame."""

    def __init__(self, ws, n=5, px_cm=PX_CM, seed=1, yaw_rate=180.0):
        self.ws, self.px_cm = ws, px_cm
        # 1.0 renders a 7.4cm shell at whatever `px_cm` says. Lower it and the
        # balls shrink while the floor stays put -- the same view a camera
        # further back would give, without moving the arena under the bots.
        self.ball_scale = 1.0
        self.speeds = SpeedMap(min_moving_byte=DEADBAND_BYTE)
        rng = np.random.default_rng(seed)
        tags = (TAGS * 4)[:n]
        self.bots = [Drifter(f"B{i + 1}", tags[i], ws, self.speeds,
                             int(rng.integers(1 << 30)), yaw_rate)
                     for i in range(n)]
        self.w = int(ws.width * px_cm)
        self.h = int(ws.height * px_cm)
        self._backdrop = self._make_backdrop(rng)

    def _make_backdrop(self, rng):
        """Floor, grid and sensor noise, built once.

        Rebuilding this every frame is most of the render cost and none of the
        information -- the noise is not the point of this app, and a fixed
        field costs nothing to look at.
        """
        canvas = np.zeros((self.h, self.w, 3), np.float32)
        canvas[:] = (6.0, 5.0, 4.0)
        for cm in np.arange(20.0, self.ws.width, 20.0):
            x = int(cm * self.px_cm)
            cv2.line(canvas, (x, 0), (x, self.h), (14, 13, 12), 2)
        for cm in np.arange(20.0, self.ws.height, 20.0):
            y = int(cm * self.px_cm)
            cv2.line(canvas, (0, y), (self.w, y), (14, 13, 12), 2)
        canvas += rng.normal(0.0, 1.6, canvas.shape).astype(np.float32)
        return canvas

    def step(self, dt):
        here = [np.array(b.robot.pos, dtype=float) for b in self.bots]
        for i, b in enumerate(self.bots):
            b.step(dt, [p for j, p in enumerate(here) if j != i])

    @property
    def ball_px(self):
        """The shell's diameter on screen — the number that decides everything."""
        return BALL_CM * self.px_cm * self.ball_scale

    @property
    def dot_gap_px(self):
        """Front tag LED to taillight — the span clustering has to bridge."""
        return (TAG_R_CM + TAIL_R_CM) * self.px_cm * self.ball_scale

    def read_kw(self):
        """Detector settings that have to follow the ball size, not be fixed.

        Both of these are lengths in pixels and both were constants tuned at
        one scale. `merge_px` must span the end dots or a ball splits into two;
        the area floor must sit under a dot or every dot is discarded as noise.
        Left fixed, the method would appear to fail at small sizes for reasons
        that are nothing to do with the method.
        """
        return {"merge_px": max(6.0, self.dot_gap_px * 1.4),
                "sparse_min_area": max(1.0, (self.ball_px / 48.0) ** 2 * 24.0)}

    def hdr(self):
        """The scene, drawn by `vision/shots.draw_ball` and nothing local.

        This used to draw the ball itself, which is how the live app carried on
        simulating one tag LED and a reflection for hours after the real
        arrangement -- two tag LEDs about the centre -- was corrected in the
        one place that generates the still frames.
        """
        canvas = self._backdrop.copy()
        for b in self.bots:
            s = b.render_state(self.px_cm)
            draw_ball(canvas, s["x_cm"], s["y_cm"], s["heading_deg"], s["tag"],
                      self.px_cm, scale=self.ball_scale)
        return canvas

    def readout(self, hdr, view):
        """One rendering of an already-built scene. See `TrackerTest.ANALYSE_ON`."""
        return {"short": as_short, "exposure": as_exposed,
                "gate": as_gated}[view](hdr)

    def frame(self, view):
        return self.readout(self.hdr(), view)


class Score:
    """Running answer to "can you trust it", per method.

    Availability first, accuracy second, and in that order deliberately: a
    method that is right whenever it speaks and silent half the time is not
    better than one that always answers within a few degrees. The tracker has
    to do something on the frames where nothing was read.
    """

    def __init__(self, keep=600):
        self.errs = deque(maxlen=keep)
        self.asked = 0
        self.answered = 0

    def add(self, err):
        self.asked += 1
        if err is None:
            return
        self.answered += 1
        self.errs.append(abs(err))

    @property
    def availability(self):
        return 100.0 * self.answered / max(self.asked, 1)

    @property
    def median(self):
        return float(np.median(self.errs)) if self.errs else None

    @property
    def worst(self):
        return max(self.errs) if self.errs else None


class TrackerTest:
    def __init__(self, paths=(), bots=5, seed=1, px_cm=PX_CM):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Tracker test — lights against travel")
        fit_to_display()
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15, bold=True)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)

        self.ws = Workspace.load()
        self.px_cm = px_cm
        self.n_bots, self.seed = bots, seed
        self.arena = LiveArena(self.ws, n=bots, px_cm=px_cm, seed=seed)
        self.view = "short"
        self.paused = False
        self.show_truth = True
        self.live = True

        self.stills = list(paths)
        self.still_index = 0
        self.frame = None
        self.surface = None
        self.rows = []
        self.readings = []              # per bot, this frame
        # One score per VIEW. They are different camera settings, not three
        # sensors, and averaging a run that switched between them describes no
        # camera anybody could build.
        self.score = {v: {"lights": Score(), "travel": Score()} for v in VIEWS}
        self.note = "live — balls drifting at the deadband speed"

        self.speed_cm_s = 4             # the deadband; the slowest it will move
        self.ball_px = 48               # the shell's diameter on screen
        self.sliders, self.buttons = [], []
        self.dragging = None
        self.apply_size(*self.screen.get_size())
        self.tick(0.0)

    # -- the loop --------------------------------------------------------

    # The gate is a way of LOOKING at a frame, not a way of capturing one. On
    # real hardware you threshold whatever the sensor gave you -- the colour is
    # still in those pixels, and you read it from there. Treating the gate as
    # its own capture meant analysing a false-coloured image, which has no hue
    # left in it, so every colour-dependent step refused and the gate scored
    # zero. That was this app inventing a camera nobody has.
    ANALYSE_ON = {"short": "short", "exposure": "exposure", "gate": "short"}

    def tick(self, dt):
        if self.live:
            if not self.paused:
                self.arena.step(dt)
            hdr = self.arena.hdr()
            self.frame = self.arena.readout(hdr, self.ANALYSE_ON[self.view])
            self.shown = self.arena.readout(hdr, self.view)
        elif self.frame is None:
            return
        self.surface = to_surface(getattr(self, "shown", None)
                                  if self.live else self.frame)
        self.rows = read_all(self.frame, stop_at_first=self.live,
                             use_peaks=not self.live,
                             **(self.arena.read_kw() if self.live else {}))
        self.match()

    def match(self):
        """Tie each reading to the ball it came from, and score both methods.

        Nearest-blob matching, which is only honest because the balls are kept
        apart. Two balls in one blob is a different problem and pretending to
        resolve it here would flatter both methods equally.
        """
        self.readings = []
        if not self.live:
            return
        for bot in self.arena.bots:
            truth = bot.truth
            want = truth["pos"] * self.px_cm
            near, best = None, 1e9
            for r in self.rows:
                d = math.hypot(r["centre"][0] - want[0], r["centre"][1] - want[1])
                if d < best:
                    near, best = r, d
            got = near if (near is not None and best < 4.0 * BALL_CM * self.px_cm) else None

            # Dots, then colour, and NEVER the brightness method. On this ball
            # the two brightest cores are the two tag LEDs -- equal, and
            # equidistant either side of the centre -- so `facing_px` recovers
            # the right axis and then guesses which way along it. Letting it
            # answer when the others decline turned a silent frame into a
            # confident 180deg error: at a 29px ball it dragged the p90 from
            # under two degrees to 178.
            lights = None
            if got and got["dots"]:
                lights = compass(got["dots"]["deg"])
            elif got and got["hue"]:
                lights = compass(got["hue"][0])

            travel = bot.travel_heading()
            pos_cm = None
            if got:
                pos_cm = np.array(got["centre"], dtype=float) / self.px_cm

            e_light = wrap180(lights - truth["heading"]) if lights is not None else None
            e_travel = wrap180(travel[0] - truth["heading"]) if travel else None
            here = self.score[self.view]
            here["lights"].add(e_light)
            here["travel"].add(e_travel)
            bot.observe(pos_cm)
            self.readings.append({
                "bot": bot, "blob": got, "lights": lights, "travel": travel,
                "pos_cm": pos_cm, "truth": truth,
                "e_light": e_light, "e_travel": e_travel,
                "pos_err": (float(np.linalg.norm(pos_cm - truth["pos"]))
                            if pos_cm is not None else None)})

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
        bw = (w - 2 * 6) // 3
        for i, v in enumerate(VIEWS):
            self.buttons.append(Button((x + i * (bw + 6), 30, bw, 26), v,
                                       lambda vv=v: self.set_view(vv)))
        # Live, and it is the one knob that makes the whole thing rowdier: a
        # faster ball covers more ground between course changes, hits walls and
        # neighbours sooner, and slips further when it slips. It also hands the
        # TRAVEL method more baseline per second, which is the fair way to
        # stress the comparison rather than a way to flatter the lights.
        self.sliders.append(Slider(
            (x, 62, w, 18), "speed cm/s", 4, 45,
            lambda: int(self.speed_cm_s), self.set_speed))
        self.sliders.append(Slider(
            (x, 86, w, 18), "ball px", 8, 80,
            lambda: int(self.ball_px), self.set_ball_px))
        bw4 = (w - 3 * 6) // 4
        by = H - PAD - 40
        for i, (label, cb) in enumerate([
                ("pause", self.toggle_pause), ("truth", self.toggle_truth),
                ("kick", self.kick), ("reset", self.reset)]):
            self.buttons.append(Button((x + i * (bw4 + 6), by, bw4, 26), label, cb))

    def set_speed(self, cm_s):
        self.speed_cm_s = int(cm_s)
        byte = self.arena.speeds.byte_for(float(cm_s))
        for b in self.arena.bots:
            b.byte = byte
        real = self.arena.speeds.cm_s_for(byte)
        self.note = (f"speed {real:.1f}cm/s (byte {byte}) — "
                     + ("the slowest these motors accept" if byte <= DEADBAND_BYTE
                        else "faster, and rowdier with it"))

    def set_ball_px(self, px):
        """Shrink the balls, leave the arena alone. Find where this breaks."""
        self.ball_px = int(px)
        self.arena.ball_scale = self.ball_px / (BALL_CM * self.px_cm)
        gap = self.arena.dot_gap_px
        self.note = (f"ball {self.ball_px}px across, end dots {gap:.0f}px apart"
                     + ("  — under the 3px floor" if gap < 6.0 else ""))

    def set_view(self, v):
        self.view = v
        self.note = {"short": "short exposure — dots sharp AND coloured",
                     "exposure": "normal exposure — the cores clip into the lobes",
                     "gate": "brightness gate — dots, but no colour left"}[v]

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_truth(self):
        self.show_truth = not self.show_truth

    def kick(self):
        for b in self.arena.bots:
            b.kick()
        self.note = ("KICKED — pointing one way, sliding another. "
                     "Watch the sun arrow leave the cyan one")

    def reset(self):
        self.seed += 1
        self.arena = LiveArena(self.ws, n=self.n_bots, px_cm=self.px_cm,
                               seed=self.seed)
        self.score = {v: {"lights": Score(), "travel": Score()} for v in VIEWS}
        self.live = True
        self.set_speed(self.speed_cm_s)
        self.set_ball_px(self.ball_px)
        self.note = "reset"

    # -- drawing ---------------------------------------------------------

    def text(self, s, x, y, col=CHALK, font=None):
        self.screen.blit((font or self.f).render(str(s), True, col), (x, y))

    @property
    def scale(self):
        if self.surface is None:
            return 1.0
        w, h = self.surface.get_size()
        return min(self.view_rect.w / max(w, 1), self.view_rect.h / max(h, 1), 1.0)

    def to_screen(self, p):
        k = self.scale
        w, h = self.surface.get_size()
        ox = self.view_rect.x + (self.view_rect.w - int(w * k)) // 2
        oy = self.view_rect.y + (self.view_rect.h - int(h * k)) // 2
        return (int(ox + p[0] * k), int(oy + p[1] * k))

    def arrow(self, at, compass_deg, length, colour, width=2):
        v = heading_vector(compass_deg)
        tip = (at[0] + v[0] * length, at[1] + v[1] * length)
        pygame.draw.line(self.screen, colour, at, tip, width)
        for side in (150.0, -150.0):
            a = math.radians(math.degrees(math.atan2(tip[1] - at[1],
                                                     tip[0] - at[0])) + side)
            pygame.draw.line(self.screen, colour, tip,
                             (tip[0] + math.cos(a) * 8, tip[1] + math.sin(a) * 8),
                             width)

    def draw_view(self):
        pygame.draw.rect(self.screen, (9, 22, 36), self.view_rect)
        pygame.draw.rect(self.screen, RULE, self.view_rect, 1)
        if self.surface is None:
            return
        k = self.scale
        w, h = self.surface.get_size()
        self.screen.blit(
            self.surface if k >= 0.999 else
            pygame.transform.scale(self.surface, (int(w * k), int(h * k))),
            self.to_screen((0, 0)))

        for i, r in enumerate(self.readings or []):
            truth = r["truth"]
            at = self.to_screen(truth["pos"] * self.px_cm)
            reach = max(BALL_CM * self.px_cm * k * 1.6, 22)
            if self.show_truth:
                self.arrow(at, truth["heading"], reach * 1.15, GREY, 1)
            if r["lights"] is not None:
                self.arrow(at, r["lights"], reach, CYAN, 2)
            if r["travel"]:
                self.arrow(at, r["travel"][0], reach * 0.8, SUN, 2)
            # x, y and theta, on the ball, live.
            p = r["pos_cm"] if r["pos_cm"] is not None else truth["pos"]
            th = r["lights"] if r["lights"] is not None else truth["heading"]
            label = f"{p[0]:.0f},{p[1]:.0f}  {th:.0f}deg"
            self.text(label, at[0] + 12, at[1] - 8, CHALK, self.fs)
            self.text(r["bot"].code, at[0] + 12, at[1] + 5, DIM, self.fs)
            if r["bot"].skidding:
                pygame.draw.circle(self.screen, CORAL, at,
                                   int(BALL_CM * self.px_cm * k * 1.1), 2)
                self.text("SLIP", at[0] + 12, at[1] + 17, CORAL, self.fs)

    def draw_dock(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, DOCK, H))
        pygame.draw.line(self.screen, RULE, (DOCK, 0), (DOCK, H))
        x, w = PAD, DOCK - 2 * PAD
        section(self.screen, self.fs, "view", x, PAD, w,
                f"{len(self.rows)} blobs", DIM)

        for sl in self.sliders:
            sl.draw(self.screen, self.f)

        y = 116
        y = section(self.screen, self.fs, "can you trust it", x, y, w,
                    f"on {self.view}", DIM)
        for name, colour in (("lights", CYAN), ("travel", SUN)):
            sc = self.score[self.view][name]
            card(self.screen, pygame.Rect(x, y, w, 34))
            self.text(name.upper(), x + 8, y + 4, colour, self.fb)
            self.text(f"answers {sc.availability:5.1f}% of frames",
                      x + 96, y + 3, CHALK, self.fs)
            med = sc.median
            self.text("median —" if med is None else
                      f"median {med:5.2f}deg   worst {sc.worst:5.1f}",
                      x + 96, y + 17,
                      MINT if (med is not None and med < 5) else SUN, self.fs)
            y += 40
        y += 4

        y = section(self.screen, self.fs, "per ball", x, y, w,
                    "cyan lights / sun travel", DIM)
        for r in (self.readings or []):
            if y > H - 90:
                break
            card(self.screen, pygame.Rect(x, y, w, 46))
            t = r["truth"]
            self.text(r["bot"].code, x + 6, y + 4,
                      CORAL if r["bot"].skidding else CHALK, self.fb)
            self.text(f"x {t['pos'][0]:5.1f}  y {t['pos'][1]:5.1f}  "
                      f"th {t['heading']:5.1f}", x + 40, y + 5, DIM, self.fs)
            self.text("lights " + ("—" if r["lights"] is None else
                                   f"{r['lights']:5.1f}  err {r['e_light']:+5.1f}"),
                      x + 40, y + 18, CYAN if r["lights"] is not None else GREY,
                      self.fs)
            if r["travel"]:
                self.text(f"travel {r['travel'][0]:5.1f}  err {r['e_travel']:+5.1f}"
                          f"  ({r['travel'][1]:.0f}cm)", x + 40, y + 31, SUN, self.fs)
            else:
                self.text("travel —  no baseline yet", x + 40, y + 31, GREY, self.fs)
            y += 52

        for b in self.buttons:
            b.on = (b.label == self.view or (b.label == "pause" and self.paused)
                    or (b.label == "truth" and self.show_truth))
            b.draw(self.screen, self.f)
        self.text(self.note[:50], PAD, H - PAD - 12, DIM, self.fs)

    # -- events ----------------------------------------------------------

    def load_still(self, index):
        if not self.stills:
            return
        self.still_index = index % len(self.stills)
        img = cv2.imread(self.stills[self.still_index], cv2.IMREAD_COLOR)
        if img is None:
            return
        self.live = False
        self.frame = img
        self.note = f"still: {os.path.basename(self.stills[self.still_index])}"
        self.tick(0.0)

    def key(self, e):
        if e.key == pygame.K_ESCAPE:
            return False
        elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.set_view(VIEWS[e.key - pygame.K_1])
        elif e.key == pygame.K_SPACE:
            self.toggle_pause()
        elif e.key == pygame.K_t:
            self.toggle_truth()
        elif e.key == pygame.K_k:
            self.kick()
        elif e.key == pygame.K_r:
            self.reset()
        elif e.key == pygame.K_RIGHT:
            self.load_still(self.still_index + 1)
        elif e.key == pygame.K_LEFT:
            self.load_still(self.still_index - 1)
        return True

    def run_loop(self):
        running = True
        while running:
            dt = min(self.clock.get_time() / 1000.0, 0.1) or 1 / 30.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.DROPFILE:
                    self.stills = expand([e.file]) + self.stills
                    self.load_still(0)
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    for sl in self.sliders:
                        if sl.hit(e.pos):
                            self.dragging = sl
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

            self.tick(dt)
            self.screen.fill(INK)
            self.draw_view()
            self.draw_dock()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="*", help="stills to step through instead")
    p.add_argument("--bots", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--px-cm", type=float, default=PX_CM)
    a = p.parse_args()
    app = TrackerTest(expand(a.images), bots=a.bots, seed=a.seed, px_cm=a.px_cm)
    if a.images:
        app.load_still(0)
    app.run_loop()


if __name__ == "__main__":
    main()
