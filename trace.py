#!/usr/bin/env python3
"""Draw a path, get the list of roll commands, watch one Sphero drive it.

    python trace.py                 # the arena from workspace.json, one sim ball
    python trace.py --robot SYRX    # a named ball, with its measured dynamics
    python trace.py --track         # start in closed-loop mode instead
    python trace.py --perfect       # no lag, no noise, no deadband: the fiction

Draw in the arena with the mouse. Freehand is a drag; polyline is a click per
corner and a right-click, double-click or Enter to finish. The right-hand panel
fills with the numbered list of commands that shape becomes, and the ball
starts driving them.

Every line of that list is one thing you could send a real ball over BLE:

     1) yaw   +90.0deg  to  90.0deg  for  0.62s   speed 0
     2) roll   90.0deg  at speed 123  for  4.53s   (20.0cm/s, 90cm)

Heading and speed go out together — that is the whole API — so a yaw is just a
roll with the speed byte at zero, turning the ball on the spot. Sharp corners
get one; gentle bends are taken while moving.

Two pilots, on the p key:

    PLAN    the list is sent on trust, open-loop, nothing watching. What a ball
            with no camera above it actually does.
    TRACK   the list is thrown away and the ball is steered off the camera,
            command by command. The panel then shows what it chose to send.

Keys — space run/pause, p plan/track, c clear, r re-run, f freehand,
l polyline, o closed loop, k calibrated speed map, s save the list,
backspace undo a corner, esc quit.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pygame

from fleet.sim_handle import SimRobot, from_motion
from swarm.pd import Polyline
from swarm.trace import (Camera, Drive, Program, RollFollower, SpeedMap,
                         compile_path, format_plan, heading_vector, simplify,
                         steps_from_log)
from ui.theme import (CHALK, CORAL, CYAN, DIM, GAP, INK, LED_RGB, MINT, PAD,
                      PANEL, RULE, SUN, Button, Slider, card, section,
                      stat_tile)
from workspace.space import Workspace

W, H = 1440, 900
MIN_W, MIN_H = 1180, 700
DOCK = 350              # controls, left
LIST = 330              # the command list, right
BALL_CM = 3.65          # a Bolt is 73mm across
TRAIL_MAX = 8000
MIN_DRAW_CM = 0.8       # freehand points closer than this are the same point
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


def load_fit(code):
    """One robot's `calib/motion.json` entry, or {} — never raising."""
    try:
        from fleet import characterize
        return characterize.load_motion(code, path=characterize.MOTION_PATH) or {}
    except Exception:
        return {}


def pick_robot(roster, wanted=None):
    """The requested robot, or the best-characterised one, or the first.

    Preferring a measured ball is not a detail. An uncharacterised one runs on
    the conservative guesses in `sim_handle`, and the entire argument for this
    window is that what you watch is the ball you own.
    """
    entries = [e for e in roster.entries if e.enabled] or roster.entries
    if wanted:
        for e in entries:
            if e.code.upper() == wanted.upper():
                return e, load_fit(e.code)
    best, best_fit, best_n = entries[0], {}, -1
    for e in entries:
        fit = load_fit(e.code)
        n = len(from_motion(fit))
        if n > best_n:
            best, best_fit, best_n = e, fit, n
    return best, best_fit


class TraceApp:
    def __init__(self, workspace_path=None, robot=None, perfect=False,
                 track=False, seed=7):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Sphero trace — draw it, compile it, drive it")
        fit_to_display()
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 16, bold=True)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)
        self.fl = pygame.font.SysFont("menlo,dejavusansmono,monospace", 12)

        self.ws = (Workspace.load(workspace_path) if workspace_path
                   else Workspace.load())
        from fleet.roster import Roster
        entry, fit = pick_robot(Roster.load(), robot)
        self.entry, self.fit = entry, fit
        self.measured = from_motion(fit)
        self.perfect = perfect
        self.seed = seed

        # A ball that has been measured simulates like the ball it was measured
        # from. `--perfect` throws all of that away on purpose: it is the
        # control condition, the simulator everybody writes first, and the
        # difference between the two trails is why the rest of this exists.
        self.robot = SimRobot(entry.name, entry.code, entry.color,
                              workspace=self.ws, seed=seed,
                              randomize=not perfect,
                              motion=None if perfect else self.measured)
        self.home = np.array([self.ws.width * 0.18, self.ws.height * 0.5])
        self.robot.pos = self.home.copy()
        # The ball's own aim error, before the slider adds any. Kept because
        # the slider sets `bias` outright and would otherwise quietly delete
        # the per-robot error the simulator generated.
        self.bias0 = float(self.robot.bias)

        # The TRUTH: what a byte does on this ball. The plant converts through
        # this, and the ball's own speed gain is applied under it.
        self.truth = SpeedMap.from_motion({} if perfect else fit)
        self.calibrated = False         # has anybody measured the speed map?

        self.path_speed = 20
        self.lookahead = 18
        self.cmd_hz = 6
        self.yaw_rate = 720 if perfect else 180
        self.aim_error = 0
        self.noise_mm = 0 if perfect else 3
        self.deadband = self.truth.min_moving_byte

        self.mode = "track" if track else "plan"
        self.draw_mode = "freehand"
        self.loop = False
        self.drawing = []               # cm, while the mouse is down
        self.pending = []               # polyline corners so far
        self.path = None
        self.steps, self.summary = [], {}
        self.drive = None
        self.running = False
        self.trail = []
        self.score = None
        self.list_scroll = 0
        self.note = "draw a path in the arena"

        self.sliders, self.buttons = [], []
        self.dragging = None
        self.apply_size(*self.screen.get_size())

    # -- the two speed maps ----------------------------------------------

    def belief(self):
        """What the PILOT thinks a byte is worth.

        Uncalibrated, it believes the nominal ceiling — which on this ball is
        a third more than the truth, and a plan compiled against it comes up a
        third short. Calibrated, it believes what a speed-map run would have
        found. The k key is the whole calibration story in one toggle.
        """
        if self.calibrated:
            return SpeedMap.truth_for(self.robot, self.deadband)
        return SpeedMap(min_moving_byte=self.deadband)

    def plant_map(self):
        return SpeedMap(min_moving_byte=self.deadband)

    def _dead_s(self):
        """The ball's command dead time, in seconds."""
        return float(getattr(self.robot, "latency_s", None)
                     or getattr(self.robot, "latency", 1) / 30.0)

    def _stop_s(self):
        """Dead time plus motor lag: how long it runs on after a command."""
        return max(float(getattr(self.robot, "tau", 0.35)) + self._dead_s(), 0.2)

    # -- layout ----------------------------------------------------------

    def apply_size(self, w, h):
        global W, H
        W, H = max(w, MIN_W), max(h, MIN_H)
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.list_rect = pygame.Rect(W - LIST, 0, LIST, H)
        area = pygame.Rect(DOCK + PAD, PAD, W - DOCK - LIST - 2 * PAD,
                           H - 2 * PAD - 26)
        # The arena keeps its real aspect ratio, always. A workspace stretched
        # to fill a window is one whose centimetres are two different lengths,
        # and every distance read off the screen is then wrong.
        self.scale = min(area.w / max(self.ws.width, 1e-6),
                         area.h / max(self.ws.height, 1e-6))
        w_px, h_px = int(self.ws.width * self.scale), int(self.ws.height * self.scale)
        self.arena = pygame.Rect(area.x + (area.w - w_px) // 2,
                                 area.y + (area.h - h_px) // 2, w_px, h_px)
        self.build_dock()

    def to_px(self, p):
        x0, _, y0, _ = self.ws.bbox
        return (int(self.arena.x + (float(p[0]) - x0) * self.scale),
                int(self.arena.y + (float(p[1]) - y0) * self.scale))

    def to_cm(self, p):
        x0, _, y0, _ = self.ws.bbox
        return np.array([x0 + (p[0] - self.arena.x) / self.scale,
                         y0 + (p[1] - self.arena.y) / self.scale])

    def build_dock(self):
        self.sliders, self.buttons = [], []
        x, w = PAD, DOCK - 2 * PAD
        y = 296

        def slider(label, lo, hi, attr):
            nonlocal y
            self.sliders.append(Slider((x, y, w, 18), label, lo, hi,
                                       lambda a=attr: getattr(self, a),
                                       lambda v, a=attr: setattr(self, a, v)))
            y += 23

        slider("speed cm/s", 5, 50, "path_speed")
        slider("lookahead cm", 3, 60, "lookahead")
        slider("cmds / s", 1, 30, "cmd_hz")
        slider("yaw deg/s", 30, 720, "yaw_rate")
        slider("aim error", -90, 90, "aim_error")
        slider("fix noise mm", 0, 30, "noise_mm")
        slider("deadband byte", 0, 60, "deadband")
        self.slider_bottom = y

        by = H - PAD - 118
        row = [("plan", lambda: self.set_mode("plan")),
               ("track", lambda: self.set_mode("track")),
               ("calibrated", self.toggle_calibrated)]
        bw = (w - 2 * 6) // 3
        for i, (label, cb) in enumerate(row):
            self.buttons.append(Button((x + i * (bw + 6), by, bw, 26), label, cb))
        row = [("freehand", lambda: self.set_draw("freehand")),
               ("polyline", lambda: self.set_draw("polyline")),
               ("closed", self.toggle_loop)]
        for i, (label, cb) in enumerate(row):
            self.buttons.append(Button((x + i * (bw + 6), by + 32, bw, 26), label, cb))
        row = [("run", self.toggle_run), ("re-run", self.rerun),
               ("clear", self.clear), ("save", self.save)]
        bw4 = (w - 3 * 6) // 4
        for i, (label, cb) in enumerate(row):
            self.buttons.append(Button((x + i * (bw4 + 6), by + 64, bw4, 26),
                                       label, cb))

    # -- modes -----------------------------------------------------------

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            if self.path is not None:
                self.rerun()

    def set_draw(self, mode):
        self.draw_mode = mode
        self.pending, self.drawing = [], []

    def toggle_calibrated(self):
        self.calibrated = not self.calibrated
        self.note = ("speed map measured — a byte now means what it does"
                     if self.calibrated else
                     "speed map back to nominal — the ball is slower than it says")
        if self.path is not None:
            self.rerun()

    def toggle_loop(self):
        self.loop = not self.loop
        if self.path is not None:
            self.commit(self.path.points)

    def clear(self):
        self.path = self.drive = None
        self.steps, self.summary = [], {}
        self.pending = self.drawing = []
        self.running = False
        self.trail, self.score = [], None
        self.note = "draw a path in the arena"

    def reset_ball(self):
        self.robot.pos = self.home.copy()
        self.robot.vel = np.zeros(2)
        self.trail = []

    # -- compiling -------------------------------------------------------

    def commit(self, points):
        """What was drawn becomes a path, a command list, and a drive."""
        pts = simplify(points, tol_cm=1.0)
        if len(pts) < 2:
            self.note = "that is one point, not a path"
            return
        path = Polyline(pts, speed=float(self.path_speed), loop=self.loop)
        if path.length < 5.0:
            self.note = "too short to drive"
            return
        self.path = path
        self.pending, self.drawing = [], []
        self.begin()

    def compile(self):
        """The numbered list, from where the ball is and where it points."""
        heading = self.drive.plant.heading if self.drive else 0.0
        self.steps, self.summary = compile_path(
            self.path, speed=float(self.path_speed), speeds=self.belief(),
            yaw_rate=float(self.yaw_rate), dead_s=self._dead_s(),
            start_pos=self.robot.pos, start_heading=heading)
        return self.steps

    def begin(self):
        """A fresh drive of the current path, from wherever the ball is."""
        if self.path is None:
            return
        self.path.speed = float(self.path_speed)
        self.compile()
        if self.mode == "plan":
            if not self.steps:
                self.note = self.summary.get("error", "nothing to send")
                return
            pilot = Program(self.steps)
        else:
            pilot = RollFollower(self.path, speed=float(self.path_speed),
                                 lookahead=float(self.lookahead),
                                 cmd_hz=float(self.cmd_hz), speeds=self.belief(),
                                 arrive_cm=6.0, stop_s=self._stop_s())
        heading = self.drive.plant.heading if self.drive else 0.0
        self.drive = Drive(self.path, self.robot, pilot, speeds=self.plant_map(),
                           yaw_rate=float(self.yaw_rate),
                           camera=Camera(sigma_cm=self.noise_mm / 10.0,
                                         seed=self.seed))
        # The assembly starts pointing wherever it was left. A real one does
        # too — nothing resets between runs short of power-cycling the ball.
        self.drive.plant.heading = heading
        self.drive.plant.commanded = heading
        self.trail, self.score = [], None
        self.list_scroll = 0
        self.running = True
        self.note = self.path.describe()
        print(f"\n{self.path.describe()}  [{self.mode}]")
        print(format_plan(self.steps, self.summary))

    def rerun(self):
        if self.path is None:
            return
        self.reset_ball()
        self.robot.pos = np.array(self.path.points[0], dtype=float)
        self.begin()

    def toggle_run(self):
        if self.path is None:
            self.note = "nothing drawn yet"
            return
        if self.drive is None or self.drive.done:
            self.begin()
        else:
            self.running = not self.running
            if not self.running:
                self.drive.plant.roll(self.drive.plant.commanded, 0)

    def save(self):
        """Write the list somewhere a person — or a robot — can read it."""
        steps = self.shown_steps()
        if not steps:
            self.note = "no commands to save"
            return
        out = ROOT / "runs"
        out.mkdir(exist_ok=True)
        name = out / f"trace-{time.strftime('%Y%m%d-%H%M%S')}-{self.mode}.txt"
        body = format_plan(steps, self.summary if self.mode == "plan" else None)
        name.write_text(f"# {self.entry.code} {self.mode} "
                        f"{self.path.describe() if self.path else ''}\n{body}\n")
        self.note = f"saved {name.name}"
        print(f"saved {name}")

    # -- the loop --------------------------------------------------------

    def apply_tuning(self):
        """Push the sliders into the live objects, every frame.

        Live rather than on the next run, because the interesting thing about a
        lookahead is what it does to the corner being taken right now.
        """
        self.robot.bias = self.bias0 + math.radians(float(self.aim_error))
        if self.drive is None:
            return
        self.drive.plant.speeds.min_moving_byte = int(self.deadband)
        self.drive.plant.yaw_rate = float(self.yaw_rate)
        self.drive.camera.sigma = self.noise_mm / 10.0
        pilot = self.drive.pilot
        if isinstance(pilot, RollFollower):
            pilot.speed = float(self.path_speed)
            pilot.lookahead = float(self.lookahead)
            pilot.cmd_hz = float(self.cmd_hz)
            pilot.stop_s = self._stop_s()
            pilot.speeds = self.belief()
            self.path.speed = float(self.path_speed)

    def step(self, dt):
        self.apply_tuning()
        if not (self.running and self.drive is not None):
            return
        if self.drive.done and self.drive.plant.speed < 1.0:
            self.finish()
            return
        self.drive.step(dt)
        self.trail.append(np.array(self.robot.pos, dtype=float))
        del self.trail[:-TRAIL_MAX]
        if self.drive.t > 240.0:
            self.running = False
            self.note = "gave up — four minutes is not a path being followed"

    def finish(self):
        self.running = False
        self.score = self.drive.score()
        self.note = "done — r to run it again"

    # -- rendering -------------------------------------------------------

    def text(self, s, x, y, col=CHALK, font=None):
        self.screen.blit((font or self.f).render(str(s), True, col), (x, y))

    def draw_arena(self):
        pygame.draw.rect(self.screen, (9, 28, 47), self.arena)
        pygame.draw.rect(self.screen, RULE, self.arena, 1)
        # A 20cm grid, so a distance on screen can be read rather than guessed.
        for cm in np.arange(20.0, max(self.ws.width, self.ws.height), 20.0):
            if cm < self.ws.width:
                x = self.to_px((cm, 0))[0]
                pygame.draw.line(self.screen, (18, 46, 70),
                                 (x, self.arena.top), (x, self.arena.bottom))
            if cm < self.ws.height:
                y = self.to_px((0, cm))[1]
                pygame.draw.line(self.screen, (18, 46, 70),
                                 (self.arena.left, y), (self.arena.right, y))
        self.text(f"{self.ws.width:.0f} x {self.ws.height:.0f} cm",
                  self.arena.x + 6, self.arena.bottom - 16, (70, 104, 134), self.fs)

    def draw_path(self):
        if self.path is not None:
            pts = [self.to_px(p) for p in self.path.preview(400)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (58, 96, 132), self.path.loop, pts, 5)
            for p in self.path.points:
                pygame.draw.circle(self.screen, (92, 134, 172), self.to_px(p), 3)
        live = list(self.drawing) or list(self.pending)
        if live:
            pts = [self.to_px(p) for p in live]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, SUN, False, pts, 2)
            for p in pts:
                pygame.draw.circle(self.screen, SUN, p, 3)
            if self.draw_mode == "polyline" and \
                    self.arena.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.line(self.screen, (120, 92, 40), pts[-1],
                                 pygame.mouse.get_pos(), 1)

    def draw_trail(self):
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, MINT, False,
                              [self.to_px(p) for p in self.trail], 2)

    def draw_robot(self):
        col = LED_RGB.get(self.entry.color, CYAN)
        p = self.to_px(self.robot.pos)
        r = max(4, int(BALL_CM * self.scale))
        if self.drive is not None and getattr(self.drive.pilot, "target", None) \
                is not None:
            t = self.to_px(self.drive.pilot.target)
            pygame.draw.line(self.screen, (58, 96, 132), p, t, 1)
            pygame.draw.circle(self.screen, CORAL, t, 4, 1)
        pygame.draw.circle(self.screen, col, p, r)
        pygame.draw.circle(self.screen, (240, 250, 255), p, r, 1)
        if self.drive is None:
            return
        plant = self.drive.plant
        # Two arrows, because the difference between them IS the yaw lag: the
        # course it was told, and the course it is actually driving.
        for deg, c, length, width in ((plant.commanded, CORAL, r * 3.2, 1),
                                      (plant.heading, CHALK, r * 2.2, 2)):
            v = heading_vector(deg)
            pygame.draw.line(self.screen, c, p,
                             (int(p[0] + v[0] * length), int(p[1] + v[1] * length)),
                             width)

    def shown_steps(self):
        """The list the panel is showing: the plan, or what got sent."""
        if self.mode == "plan":
            return self.steps
        if self.drive is None:
            return []
        return steps_from_log(self.drive.plant.log, self.plant_map(),
                              until=self.drive.plant.t)

    def current_index(self):
        if self.drive is None:
            return -1
        if self.mode == "plan":
            return getattr(self.drive.pilot, "i", -1)
        return len(self.drive.plant.log) - 1

    def draw_list(self):
        r = self.list_rect
        pygame.draw.rect(self.screen, PANEL, r)
        pygame.draw.line(self.screen, RULE, (r.x, 0), (r.x, H))
        x, w = r.x + PAD, r.w - 2 * PAD
        title = "the plan" if self.mode == "plan" else "what was sent"
        steps = self.shown_steps()
        y = section(self.screen, self.fs, title, x, PAD, w,
                    f"{len(steps)} cmds", DIM)

        rows = max(1, (H - y - 96) // 15)
        cur = self.current_index()
        # Follow the cursor, but never past the end of the list.
        top = max(0, min(cur - rows // 2, len(steps) - rows)) if cur >= 0 else 0
        top = max(0, top + self.list_scroll)
        for i in range(top, min(top + rows, len(steps))):
            st = steps[i]
            done = i < cur
            col = CHALK if i == cur else (GREY_DONE if done else DIM)
            if i == cur:
                pygame.draw.rect(self.screen, (24, 62, 92), (x - 4, y - 1, w + 8, 15))
            self.text(st.line(i + 1), x, y, col, self.fl)
            y += 15
        if len(steps) > top + rows:
            self.text(f"... {len(steps) - top - rows} more", x, y, GREY_DONE, self.fs)

        y = H - PAD - 84
        y = section(self.screen, self.fs, "totals", x, y, w)
        if self.mode == "plan" and self.summary:
            if self.summary.get("error"):
                self.text(self.summary["error"][:44], x, y, SUN, self.fs)
            else:
                self.text(f"{self.summary['steps']} commands  "
                          f"{self.summary['seconds']:.1f}s  "
                          f"{self.summary['distance_cm']:.0f}cm", x, y, CHALK, self.fs)
                y += 14
                self.text(f"byte {self.summary['byte']} = "
                          f"{self.summary['speed_cm_s']:.1f}cm/s, "
                          f"{self.summary['turns']} turns in place", x, y, DIM, self.fs)
                y += 14
                if self.summary.get("note"):
                    self.text(self.summary["note"][:46], x, y, SUN, self.fs)
        elif steps:
            total = sum(s.seconds for s in steps)
            self.text(f"{len(steps)} commands  {total:.1f}s  "
                      f"{sum(s.distance for s in steps):.0f}cm", x, y, CHALK, self.fs)

    def draw_dock(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, DOCK, H))
        pygame.draw.line(self.screen, RULE, (DOCK, 0), (DOCK, H))
        x, w = PAD, DOCK - 2 * PAD
        y = PAD

        kind = "perfect" if self.perfect else (
            "measured" if self.measured else "guessed")
        y = section(self.screen, self.fs, "robot", x, y, w, kind,
                    MINT if self.measured and not self.perfect else SUN)
        self.text(f"{self.entry.name}  {self.entry.code}", x, y, CHALK, self.fb)
        y += 21
        tau = getattr(self.robot, "tau", 0.0)
        self.text(f"motor lag {tau:.2f}s   link {self._dead_s():.2f}s   "
                  f"gain {getattr(self.robot, 'gain', 1.0):.2f}", x, y, DIM, self.fs)
        y += 13
        belief = self.belief()
        self.text(f"deadband {self.deadband} byte = "
                  f"{belief.min_moving_cm_s:.1f}cm/s   "
                  f"map {'measured' if self.calibrated else 'nominal'}",
                  x, y, DIM if self.calibrated else SUN, self.fs)
        y += 20

        y = section(self.screen, self.fs, "last roll command", x, y, w,
                    self.mode.upper(), CYAN)
        plant = self.drive.plant if self.drive else None
        cmd = plant.log[-1] if plant and plant.log else None
        card(self.screen, pygame.Rect(x, y, w, 62))
        if cmd:
            self.text(f"roll({cmd[1]:6.1f}°, {cmd[2]:3d})", x + 10, y + 7,
                      CORAL, self.fb)
            below = 0 < cmd[2] < self.deadband
            self.text(f"= {self.plant_map().cm_s_for(cmd[2]):.1f} cm/s"
                      f"{'   BELOW DEADBAND' if below else ''}",
                      x + 10, y + 28, SUN if below else DIM, self.fs)
            self.text(f"driving {plant.heading:5.1f}°, "
                      f"{plant.turning:4.0f}° still to swing",
                      x + 10, y + 42, DIM, self.fs)
        else:
            self.text("nothing sent yet", x + 10, y + 22, DIM, self.f)
        y += 62 + GAP

        pilot = self.drive.pilot if self.drive else None
        tiles = [("speed", f"{plant.speed:.0f}" if plant else "0"),
                 ("off path", f"{self.drive.cross_track:.1f}" if self.drive else "-"),
                 ("cmds", str(getattr(pilot, "commands", 0)) if pilot else "0"),
                 ("done", f"{self._progress():.0f}%")]
        tw = (w - 3 * 6) // 4
        for i, (label, value) in enumerate(tiles):
            stat_tile(self.screen, self.fs, self.fb,
                      pygame.Rect(x + i * (tw + 6), y, tw, 40), label, value)

        for s in self.sliders:
            s.draw(self.screen, self.f)

        ry = self.slider_bottom + 8
        if self.score:
            ry = section(self.screen, self.fs, "how it went", x, ry, w)
            rms = self.score.get("rms_cm")
            self.text(f"rms {rms}cm   worst {self.score.get('max_cm')}cm",
                      x, ry, MINT if (rms or 99) < 4 else SUN, self.f)
            ry += 16
            self.text(f"stopped {self.score.get('finished_cm')}cm from the end",
                      x, ry, DIM, self.fs)
            ry += 13
            self.text(f"{self.score['commands']} commands in "
                      f"{self.score['seconds']}s "
                      f"({self.score['cmd_hz']}/s)", x, ry, DIM, self.fs)

        for b in self.buttons:
            b.on = (b.label in (self.mode, self.draw_mode)
                    or (b.label == "closed" and self.loop)
                    or (b.label == "calibrated" and self.calibrated)
                    or (b.label == "run" and self.running))
            b.draw(self.screen, self.f)
        self.text(self.note[:52], PAD, H - PAD - 14, DIM, self.fs)

    def _progress(self):
        if not self.drive or self.path is None or self.path.length <= 0:
            return 0.0
        if self.mode == "plan":
            pilot = self.drive.pilot
            n = max(len(getattr(pilot, "steps", [])), 1)
            return 100.0 * max(getattr(pilot, "i", 0), 0) / n
        return min(100.0, 100.0 * self.drive.pilot.progress / self.path.length)

    # -- events ----------------------------------------------------------

    def click(self, pos, button):
        for s in self.sliders:
            if s.hit(pos):
                self.dragging = s
                return
        for b in self.buttons:
            if b.hit(pos):
                return
        if not self.arena.collidepoint(pos):
            return
        p = self.to_cm(pos)
        if self.draw_mode == "polyline":
            if button == 3 or (self.pending
                               and np.linalg.norm(p - self.pending[-1]) < 1.0):
                if len(self.pending) >= 2:
                    self.commit(self.pending)
                return
            self.pending.append(p)
            self.note = f"{len(self.pending)} corners — right-click to finish"
        else:
            self.drawing = [p]
            self.running = False

    def motion(self, pos, buttons):
        if self.dragging:
            self.dragging.drag(pos)
            return
        if self.draw_mode == "freehand" and buttons[0] and self.drawing \
                and self.arena.collidepoint(pos):
            p = self.to_cm(pos)
            if np.linalg.norm(p - self.drawing[-1]) >= MIN_DRAW_CM:
                self.drawing.append(p)

    def release(self, pos):
        self.dragging = None
        if self.draw_mode == "freehand" and len(self.drawing) >= 2:
            self.commit(self.drawing)
        self.drawing = []

    def key(self, e):
        if e.key == pygame.K_ESCAPE:
            return False
        elif e.key == pygame.K_SPACE:
            self.toggle_run()
        elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if len(self.pending) >= 2:
                self.commit(self.pending)
            else:
                self.rerun()
        elif e.key == pygame.K_BACKSPACE and self.pending:
            self.pending.pop()
        elif e.key == pygame.K_c:
            self.clear()
        elif e.key == pygame.K_r:
            self.rerun()
        elif e.key == pygame.K_p:
            self.set_mode("track" if self.mode == "plan" else "plan")
        elif e.key == pygame.K_f:
            self.set_draw("freehand")
        elif e.key == pygame.K_l:
            self.set_draw("polyline")
        elif e.key == pygame.K_o:
            self.toggle_loop()
        elif e.key == pygame.K_k:
            self.toggle_calibrated()
        elif e.key == pygame.K_s:
            self.save()
        return True

    def run_loop(self):
        running = True
        while running:
            dt = 1.0 / 30.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    self.click(e.pos, e.button)
                elif e.type == pygame.MOUSEMOTION:
                    self.motion(e.pos, e.buttons)
                elif e.type == pygame.MOUSEBUTTONUP:
                    self.release(e.pos)
                elif e.type == pygame.KEYDOWN:
                    running = self.key(e)
                elif e.type == pygame.VIDEORESIZE:
                    self.apply_size(e.w, e.h)
                elif e.type == pygame.MOUSEWHEEL:
                    if self.list_rect.collidepoint(pygame.mouse.get_pos()):
                        self.list_scroll -= e.y * 2

            self.step(dt)
            self.screen.fill(INK)
            self.draw_arena()
            self.draw_path()
            self.draw_trail()
            self.draw_robot()
            self.draw_dock()
            self.draw_list()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


GREY_DONE = (78, 108, 134)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default=None, help="roster code, e.g. SYRX")
    p.add_argument("--workspace", default=None)
    p.add_argument("--track", action="store_true",
                   help="start closed-loop instead of running the list on trust")
    p.add_argument("--perfect", action="store_true",
                   help="no lag, no noise, no deadband — the sim that lies")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    TraceApp(workspace_path=a.workspace, robot=a.robot, perfect=a.perfect,
             track=a.track, seed=a.seed).run_loop()


if __name__ == "__main__":
    main()
