#!/usr/bin/env python3
"""Swarm lab — workspace, fleet and tools in one window.

    python app.py                     # all-sim, no camera, no robots needed
    python app.py --camera 0          # run the tracker for real robots
    python app.py --controller boids

Left dock: status, controller, roster, formations, command bar, STOP.
Right: the workspace, in centimetres.

Keys — space pause, b/p/i/n controller, tab focus the command bar, esc unfocus.
1-9 take manual control of a robot: W/S drive, A/D turn, esc release.
v cycles the camera view: panel / floor (warped under the arena) / off.\nF11 fullscreen; the window is resizable and the dock re-lays itself out.
c calibrates a real robot's heading offset from how it just travelled.
"""

import argparse
import threading
import time

import numpy as np
import pygame
import torch

from fleet.manager import Fleet
from fleet.roster import RobotEntry, Roster
from swarm.boids import Boids, Idle
from swarm.navigate import Navigate
from swarm.policy import PolicyController, list_checkpoints
from swarm.train import train
from tools import SwarmContext
from tools.command import parses_as_tool
from tools.command import run as run_command
from tools.command import summarise
from vision.config import COLORS
from workspace.space import Workspace

torch.set_num_threads(1)

LOG_HISTORY = 400       # entries kept for scrollback

W, H = 1500, 950
DOCK = 400
INK = (11, 34, 57)
PANEL = (15, 44, 70)
PANEL2 = (20, 55, 84)
RULE = (44, 78, 108)
CHALK = (228, 240, 248)
DIM = (143, 179, 204)
CYAN = (99, 210, 232)
SUN = (245, 179, 66)
CORAL = (255, 107, 107)
MINT = (126, 226, 168)
GREY = (110, 124, 140)

# The window must fit the screen it opens on. At the hardcoded 950 the bottom
# strip — which is where STOP and the command bar live — falls below a 14"
# MacBook's usable height once the menu bar and title bar are subtracted, and
# the controls are simply unreachable.
CHROME_H = 80        # menu bar + title bar, approximately
CHROME_W = 40
MIN_W, MIN_H = 1100, 720


def _fit_to_display():
    """Shrink W/H to the actual desktop. Safe to call before set_mode()."""
    global W, H
    try:
        sizes = pygame.display.get_desktop_sizes()
        sw, sh = sizes[0]
    except Exception:
        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h
    if sw and sh:
        W = max(MIN_W, min(W, sw - CHROME_W))
        H = max(MIN_H, min(H, sh - CHROME_H))
    return W, H


# How fast robots actually drive, in cm/s at full controller output.
#
# THIS is the speed knob — edit this one number, or pass --speed.
#
# Not `fleet.handle.MAX_SPEED`, which looks like the obvious place and is not:
# that constant is also the cm/s-to-motor-byte calibration (byte = speed /
# MAX_SPEED * 255). Halving it would map 30cm/s onto byte 255 and a real ball
# would roll exactly as fast as before, having been told it was going half
# speed. This value scales what the controller asks for; that one converts the
# ask into hardware units, and they are different jobs.
CRUISE_SPEED = 45.0          # 75% of the 60 cm/s hardware ceiling

LED_RGB = {
    "red": (255, 60, 60), "yellow": (255, 210, 70), "green": (90, 220, 120),
    "cyan": (99, 210, 232), "blue": (90, 140, 250), "magenta": (225, 100, 215),
}


# One vertical rhythm for the whole dock. Panels used to each pick their own
# gaps, which is most of why it read as dense: nothing lined up, so the eye had
# no column to follow.
PAD = 14            # panel inner padding
GAP = 16            # between sections
CARD = (14, 40, 65)         # panel fill, a step up from the dock
CARD_EDGE = (28, 60, 90)


def section(surface, font, label, x, y, w, right_text=None, right_color=None):
    """An uppercase, letter-spaced section header. Returns the next y."""
    spaced = " ".join(label.upper())
    surface.blit(font.render(spaced, True, (120, 156, 186)), (x, y))
    if right_text:
        t = font.render(right_text, True, right_color or (120, 156, 186))
        surface.blit(t, (x + w - t.get_width(), y))
    pygame.draw.line(surface, (26, 56, 84), (x, y + 15), (x + w, y + 15))
    return y + 24


def card(surface, rect, fill=CARD, edge=CARD_EDGE):
    pygame.draw.rect(surface, fill, rect, border_radius=5)
    pygame.draw.rect(surface, edge, rect, 1, border_radius=5)
    return rect


def stat_tile(surface, small, big, rect, label, value, tone=None):
    """Label above, value below — the shape a number is easiest to read in."""
    card(surface, rect)
    surface.blit(small.render(" ".join(label.upper()), True, (108, 142, 172)),
                 (rect.x + 8, rect.y + 6))
    surface.blit(big.render(str(value), True, tone or CHALK),
                 (rect.x + 8, rect.y + 20))


class Button:
    def __init__(self, rect, label, cb, toggle=False, tone=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.cb = cb
        self.toggle = toggle
        self.tone = tone
        self.on = False
        self.enabled = True

    def draw(self, s, f):
        col = self.tone or (CYAN if self.on else RULE)
        bg = PANEL2 if self.on else PANEL
        if not self.enabled:
            col, bg = RULE, INK
        pygame.draw.rect(s, bg, self.rect, border_radius=3)
        pygame.draw.rect(s, col, self.rect, 1, border_radius=3)
        t = f.render(self.label, True, CHALK if self.enabled else RULE)
        s.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, p):
        if self.enabled and self.rect.collidepoint(p):
            self.cb()
            return True
        return False


def clipboard_text():
    """The system clipboard, as a single line. Never raises.

    Tries pygame's own scrap first and falls back to `pbpaste`, because scrap
    is not initialised on every platform/backend and a paste that silently does
    nothing is worse than one that costs a subprocess.
    """
    try:
        import pygame.scrap as scrap
        if not scrap.get_init():
            scrap.init()
        raw = scrap.get(pygame.SCRAP_TEXT)
        if raw:
            text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
            return text.replace("\x00", "").replace("\n", " ").strip()
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["pbpaste"], capture_output=True, text=True,
                             timeout=1.0)
        if out.returncode == 0:
            return out.stdout.replace("\n", " ").strip()
    except Exception:
        pass
    return ""


class TextField:
    def __init__(self, rect, placeholder=""):
        self.rect = pygame.Rect(rect)
        self.text = ""
        self.placeholder = placeholder
        self.focused = False
        self.caret = 0
        self.history = []
        self.hist_pos = None

    def draw(self, s, f):
        col = CYAN if self.focused else RULE
        pygame.draw.rect(s, PANEL, self.rect, border_radius=3)
        pygame.draw.rect(s, col, self.rect, 1, border_radius=3)
        body = self.text or self.placeholder
        colour = CHALK if self.text else RULE
        surf = f.render(body, True, colour)
        area = surf.get_rect()
        # keep the caret end visible on a long line
        if area.w > self.rect.w - 14:
            surf = surf.subsurface(pygame.Rect(area.w - (self.rect.w - 14), 0,
                                               self.rect.w - 14, area.h))
        s.blit(surf, (self.rect.x + 7, self.rect.y + 6))
        if self.focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            x = min(self.rect.x + 7 + f.size(self.text)[0], self.rect.right - 7)
            pygame.draw.line(s, CYAN, (x, self.rect.y + 5),
                             (x, self.rect.bottom - 5), 1)

    def key(self, e):
        """Returns a submitted line, or None."""
        if e.key == pygame.K_RETURN:
            line = self.text.strip()
            if line:
                self.history.append(line)
            self.text = ""
            self.hist_pos = None
            return line
        # Cmd+V / Ctrl+V. Checked before the printable branch, or `v` would
        # simply type itself and the paste would look like it did nothing.
        if e.key == pygame.K_v and (e.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)):
            self.text += clipboard_text()
            return None
        if e.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif e.key == pygame.K_ESCAPE:
            self.focused = False
        elif e.key == pygame.K_UP and self.history:
            self.hist_pos = (len(self.history) - 1 if self.hist_pos is None
                             else max(0, self.hist_pos - 1))
            self.text = self.history[self.hist_pos]
        elif e.key == pygame.K_DOWN and self.history:
            if self.hist_pos is not None and self.hist_pos < len(self.history) - 1:
                self.hist_pos += 1
                self.text = self.history[self.hist_pos]
            else:
                self.hist_pos = None
                self.text = ""
        elif e.unicode and e.unicode.isprintable():
            self.text += e.unicode
        return None


class App:
    def __init__(self, camera=None, controller="navigate", roster_path=None,
                 workspace_path=None, model="qwen3.5:9b", session=None,
                 ask=None, speed=None):
        self.speed = float(speed or CRUISE_SPEED)
        speed = self.speed
        pygame.display.init()
        pygame.font.init()
        pygame.key.set_repeat(400, 40)
        pygame.display.set_caption("Swarm lab")
        _fit_to_display()
        # Resizable, and F11 for fullscreen. The dock's panels are laid out
        # from W/H at build time, so a resize has to rebuild them — see
        # `apply_size`.
        self.fullscreen = False
        self.windowed_size = (W, H)
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.f = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.fb = pygame.font.SysFont("menlo,dejavusansmono,monospace", 16, bold=True)
        self.fs = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)

        self.ws = Workspace.load(workspace_path) if workspace_path else Workspace.load()
        self.roster = Roster.load(roster_path) if roster_path else Roster.load()
        self.tracker = None
        if camera is not None:
            from fleet.vision_link import CameraTracker
            self.tracker = CameraTracker(source=camera)
            self.tracker.start()

        self.homography = None
        self.fleet = Fleet.from_roster(self.roster, workspace=self.ws,
                                       tracker=self.tracker)
        self.ctx = SwarmContext(fleet=self.fleet, workspace=self.ws,
                                controller=Navigate(), max_speed=speed)
        self.ctx.realtime = True        # a person is watching; pace the sim
        # Learned precedents. Identifiers come from the roster so that "swap
        # SSMK and CRXS" matches a remembered "swap Seasmoke and Caraxes" —
        # which robots were named is an argument, not the intent.
        try:
            from llm.memory import Memory
            ids = [e.code for e in self.roster.entries]
            ids += [e.name for e in self.roster.entries]
            self.ctx.memory = Memory.load()
            self.ctx.memory.identifiers = set(ids)
        except Exception as e:
            self.log.append(("error", f"memory unavailable: {e}"))
        self.log = []
        # Beside the log itself: `say` runs during __init__, before the widget
        # state further down exists.
        self.log_scroll = 0          # lines back from the newest
        self.log_max_scroll = 0
        for e in (self.roster.errors + self.ws.errors + self.fleet.errors):
            self.log.append(("error", e))
        self.log.append(("info", f"{len(self.fleet)} robot(s) in the fleet."))
        # After the log exists — this reports into it, and running it earlier
        # crashed every camera start with AttributeError.
        if self.tracker is not None:
            self.check_calibration()

        # The simulation is advanced by the render loop *and* by tools such as
        # wait_until_settled on the agent thread. One lock, so they take turns —
        # and it is the context's own, held for a single tick at a time, so a
        # long-running tool never locks the user out of the other robots.
        self.sim_lock = self.ctx.lock
        self.model_preset = model
        self.fell_back_from = None
        self.session = session if session is not None else self._open_session(model)
        self.pending = None            # the command currently in flight
        self.started_at = None
        self.streamed = ""             # partial reply, shown while it thinks

        if self.session is not None and self.session.available:
            self.ask = True if ask is None else ask
            if self.fell_back_from:
                self.say("info", f"model {self.model_preset} ready (fallback "
                                 f"from {self.fell_back_from}) — type a request "
                                 "in plain English.")
            else:
                self.say("info", f"model {self.session.model_name} ready — "
                                 "type a request in plain English.")
        else:
            self.ask = False
            why = (self.session.error if self.session is not None
                   else "no model configured")
            self.say("info", f"no model ({why}) — literal tool calls only.")
        self.say("info", "e.g. move_to 60,40 120,40 180,40")

        self.boids = Boids()
        self.policy = None
        self.mode = controller
        self.paused = False
        self.checkpoints = list_checkpoints()
        self.sel = None

        self.train_thread = None
        self.train_stop = threading.Event()
        self.progress = None
        self.train_updates = 150

        self.roster_rows = []
        self.obstacle_rows = []
        self.entity_rows = []
        self.trails = {}             # entity id -> recent positions, for the tail

        # Manual drive. `manual_heading` is a direction in the WORKSPACE frame,
        # not the Sphero's own — the architecture keeps heading out of the tool
        # surface precisely because the camera frame and the ball's frame differ
        # by an uncalibrated offset. This one never leaves the UI: it is turned
        # straight into a cm/s velocity, the same API every controller uses.
        # `v` cycles: "panel" (picture in picture) -> "floor" (the camera warped
        # into the arena itself, so sim robots stand on the real surface) ->
        # "off". Ignored entirely without a tracker.
        self.camera_view = "panel"
        self.floor_surface = None
        self.floor_stamp = None
        self.cam_surface = None      # cached surface, rebuilt only on a new frame
        self.cam_stamp = None
        self.manual_code = None
        self.manual_heading = {}     # code -> degrees, 0 = +x, increasing = clockwise
        self.keys_held = set()
        self.manual_track = []       # [(pos, commanded_velocity)] while driving
        # Active calibration: drives a known pattern and reads the frame offset
        # straight off it. The passive estimator only tracks drift afterwards.
        self.calibrating = None      # (code, ActiveCalibration)
        self.estimators = {}         # code -> HeadingEstimator
        self.dragging_obstacle = None
        self.log_rect = pygame.Rect(0, 0, 0, 0)
        self.formation_rows = []
        self.ck_rows = []
        self.editing = None          # (code, "name" | "code")
        self.edit_buf = ""
        self._build()

    #: Tried in order when the requested preset cannot be reached. Local only —
    #: falling back from one unreachable hosted preset to another behind the
    #: same gateway would just fail twice.
    FALLBACK_PRESETS = ("qwen3.5:9b", "qwen3.5:4b")

    def check_calibration(self):
        """Warn when the camera and the planner disagree about the arena.

        These are two coordinate systems that have to describe the same
        rectangle. When they do not, nothing errors: the tracker reports
        positions in one frame while the controller plans in another, and
        robots calmly drive off an arena they are nowhere near the edge of.
        The symptom looks like a controller bug, so say it out loud at startup.
        """
        from vision.homography import Homography

        h = Homography.load()
        self.homography = h if h.ready else None
        fix = ("python -m workspace.make --from-camera --source 0 "
               f"--width {self.ws.width:.0f} --height {self.ws.height:.0f}")

        if not h.ready:
            self.log.append(("error", f"no camera calibration — run: {fix}"))
            return False

        if h.matches(self.ws.width, self.ws.height):
            return True

        self.log.append(("error",
                         f"camera is calibrated for {h.width:.0f}x{h.height:.0f}cm "
                         f"but the workspace is {self.ws.width:.0f}x"
                         f"{self.ws.height:.0f}cm — positions will not line up, "
                         "and robots will appear to drift off the arena"))
        self.log.append(("info", f"fix with: {fix}"))
        return False

    def _open_session(self, model, allow_fallback=True):
        """Never let a missing or broken model stop the app from starting.

        A hosted preset is unreachable the moment you are off the VPN, and
        losing plain English entirely for the rest of the session is a big
        punishment for a network hiccup. So an unreachable preset falls back to
        a local one — loudly, because silently answering as a different model
        than the one asked for is worse than the outage.
        """
        from llm.session import AgentSession

        def open_one(preset):
            try:
                return AgentSession(self.ctx, preset=preset,
                                    sim_lock=self.sim_lock, warm=True), None
            except Exception as e:
                return None, str(e)

        session, err = open_one(model)
        if session is not None and session.available:
            self.model_preset = model
            self.fell_back_from = None
            return session

        why = err or (session.error if session is not None else "unavailable")
        if not allow_fallback:
            self.log.append(("error", f"{model}: {why}"))
            self.model_preset = model
            return session

        for alt in self.FALLBACK_PRESETS:
            if alt == model:
                continue
            candidate, alt_err = open_one(alt)
            if candidate is not None and candidate.available:
                self.log.append(("error", f"{model} is unreachable ({why})"))
                self.log.append(("info", f"falling back to the local {alt} — "
                                         "reconnect the VPN and restart to use "
                                         f"{model}"))
                self.model_preset = alt
                self.fell_back_from = model
                return candidate

        self.log.append(("error", f"{model}: {why}"))
        self.model_preset = model
        self.fell_back_from = None
        return session

    def apply_size(self, width, height):
        """Resize the window and re-lay the dock out at the new size."""
        global W, H
        W = max(MIN_W, int(width))
        H = max(MIN_H, int(height))
        flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((W, H), flags)
        self._build()                # panels are positioned from W/H
        self.sync()
        # A cached warp is sized to the old arena rect and would be stretched.
        self.floor_surface = None
        self.floor_stamp = None
        self.cam_surface = None
        self.cam_stamp = None

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.fullscreen = False
            self.apply_size(*self.windowed_size)
            self.say("info", "windowed")
            return
        self.windowed_size = (W, H)
        self.fullscreen = True
        try:
            sw, sh = pygame.display.get_desktop_sizes()[0]
        except Exception:
            info = pygame.display.Info()
            sw, sh = info.current_w, info.current_h
        self.apply_size(sw, sh)
        self.say("info", "fullscreen — F11 or esc to return")

    # -- widgets ---------------------------------------------------------

    def _build(self):
        x, w = 14, DOCK - 28
        self.buttons = []

        self.modes = []
        labels = [("Nav", "navigate"), ("Boids", "boids"),
                  ("Policy", "policy"), ("Idle", "idle")]
        bw = (w - 3 * 4) // 4
        for i, (lab, m) in enumerate(labels):
            b = Button((x + i * (bw + 4), 84, bw, 26), lab,
                       lambda m=m: self.set_mode(m), toggle=True)
            self.modes.append(b)
            self.buttons.append(b)

        # Level with the "roster" heading, clear of the first row's toggles.
        self.add_btn = Button((x + w - 116, 118, 56, 22), "+ add", self.add_robot)
        self.buttons.append(self.add_btn)
        self.reload_btn = Button((x + w - 56, 118, 56, 22), "reload",
                                 self.reload_roster)
        self.buttons.append(self.reload_btn)

        # placed each frame by draw_obstacles_panel, which knows the layout
        self.obs_add_btn = Button((x + w - 56, 0, 56, 18), "+ obs",
                                  self.add_obstacle)
        self.ent_add_btn = Button((x + w - 56, 0, 56, 18), "+ ent",
                                  self.add_entity)


        self.cmd = TextField((x, H - 84, w, 26),
                             "move_to 60,40 120,40 180,40")

        self.stop_btn = Button((x, H - 50, w, 38), "STOP ALL", self.stop_all,
                               tone=CORAL)
        self.buttons.append(self.stop_btn)

        # ask / tool toggle, sitting directly above the command bar it governs
        self.ask_btn = Button((x, H - 112, 52, 22), "ask",
                              lambda: self.set_ask(True), toggle=True)
        self.tool_btn = Button((x + 56, H - 112, 52, 22), "tool",
                               lambda: self.set_ask(False), toggle=True)
        # Beside ask/tool rather than on the roster row, where "+ add" already
        # lives — the overlap tests caught that collision immediately.
        self.calib_btn = Button((x + 116, H - 112, 62, 22), "calib",
                                self.start_calibration)
        self.buttons += [self.ask_btn, self.tool_btn, self.calib_btn]

        self.train_btn = Button((x, H - 146, w - 92, 26), "Train new policy",
                                self.toggle_train)
        self.len_btn = Button((x + w - 88, H - 146, 88, 26), "150", self.cycle_len)
        self.buttons += [self.train_btn, self.len_btn]
        self.sync()

    def sync(self):
        for b in self.modes:
            b.on = {"Nav": "navigate", "Boids": "boids",
                    "Policy": "policy", "Idle": "idle"}[b.label] == self.mode
        has_model = self.session is not None and self.session.available
        self.ask_btn.on = self.ask
        self.tool_btn.on = not self.ask
        self.ask_btn.enabled = has_model
        self.cmd.placeholder = ("form a circle" if self.ask
                                else "move_to 60,40 120,40 180,40")

    def set_ask(self, on):
        if on and not (self.session is not None and self.session.available):
            self.say("error", "no model is reachable — staying in tool mode")
            return
        self.ask = bool(on)
        self.sync()

    # -- actions -----------------------------------------------------------

    def scroll_log(self, lines):
        """Positive scrolls back into history, negative returns to the newest."""
        self.log_scroll = max(0, min(self.log_scroll + int(lines),
                                     self.log_max_scroll))

    def say(self, kind, text):
        self.log.append((kind, text))
        del self.log[:-LOG_HISTORY]
        # Following the tail is the default; a new line while scrolled back
        # must not yank the view, or reading history during a run is hopeless.
        if self.log_scroll > 0:
            self.log_scroll = min(self.log_scroll + 1, LOG_HISTORY)

    def set_mode(self, m):
        if m == "policy" and self.policy is None:
            if not self.checkpoints:
                self.say("error", "no trained checkpoints in runs/ yet")
                return
            self.load(self.checkpoints[0])
        self.mode = m
        self.sync()

    def load(self, ck):
        try:
            self.policy = PolicyController(ck["path"])
            self.sel = ck["name"]
            self.mode = "policy"
            self.sync()
            self.say("info", f"loaded {ck['name']}")
        except Exception as e:
            self.say("error", f"could not load {ck['name']}: {e}")

    def controller(self):
        if self.mode == "boids":
            return self.boids
        if self.mode == "policy" and self.policy is not None:
            return self.policy
        if self.mode == "navigate":
            return self.ctx.controller
        return Idle()

    def step_world(self, dt):
        """One frame of simulation, on the render thread.

        Deliberately a single call into `ctx.tick` with this frame's controller
        rather than an inlined copy of it. It used to be inlined, and drifted:
        the copy never advanced moving entities or the controller's clock, so a
        flow field was frozen at t=0 and a patrolling obstacle sat still —
        except during a `wait_until_settled`, which ticks the context properly
        on the agent thread and made the world lurch forward only while a tool
        happened to be running.

        The lock is taken without blocking for the same reason: when a tool is
        already driving the world, waiting our turn here would freeze the
        window for as long as that tool runs.
        """
        if not self.sim_lock.acquire(blocking=False):
            return False
        try:
            # Manual drive goes in as an override so it lands after the
            # controller and before integration: a human holding W should not
            # be arguing with a controller for the same robot. Calibration
            # takes precedence over both — it is driving a measured pattern and
            # anything else steering would corrupt the measurement.
            def override():
                if not self.step_calibration(dt):
                    self.drive_manual(dt)

            self.ctx.tick(dt, controller=self.controller(), override=override)
        finally:
            self.sim_lock.release()
        return True

    # -- manual drive -----------------------------------------------------

    MANUAL_SPEED = CRUISE_SPEED * 0.6    # cm/s at full stick, tracking cruise
    MANUAL_TURN = 140.0          # deg/s
    MANUAL_KEYS = {"w", "s", "a", "d"}

    def manual_codes(self):
        return self.fleet.codes

    def select_manual(self, code):
        """Take a robot off the controller and drive it by hand.

        Its layers and target go first. Leaving a flow running underneath a
        human on the keyboard means two things steering one ball, and the
        result reads as the robot ignoring you.
        """
        if code not in self.fleet.handles:
            return
        if self.manual_code == code:
            self.release_manual()
            return
        if self.manual_code is not None:
            self.release_manual()

        self.manual_code = code
        self.manual_track = []
        self.manual_heading.setdefault(code, 0.0)
        self.ctx.stack.clear_layers(code)
        self.ctx.env.targets.pop(code, None)
        h = self.fleet.handles.get(code)
        if h is not None:
            h.target = None
        self.say("info", f"manual: {code} — W/S drive, A/D turn, esc release")

    def release_manual(self):
        code, self.manual_code = self.manual_code, None
        self.keys_held.clear()
        if code is None:
            return
        h = self.fleet.handles.get(code)
        if h is not None:
            h.set_velocity(np.zeros(2))
        self.say("info", f"released {code}")

    def manual_key_down(self, key):
        """Returns True if the key was consumed by manual drive.

        Number keys pick a robot. While one is selected WASD is swallowed —
        `a` is the ask/tool toggle otherwise, and turning left must not flip
        the model on and off.
        """
        if pygame.K_1 <= key <= pygame.K_9:
            i = key - pygame.K_1
            codes = self.manual_codes()
            if i < len(codes):
                self.select_manual(codes[i])
            return True
        if self.manual_code is None:
            return False
        if key == pygame.K_ESCAPE:
            self.release_manual()
            return True
        if key == pygame.K_c:
            self.calibrate_heading()
            return True
        name = pygame.key.name(key)
        if name in self.MANUAL_KEYS:
            self.keys_held.add(key)
            return True
        return False

    def drive_manual(self, dt):
        """Turn held keys into a velocity. Called once a frame, under the lock."""
        code = self.manual_code
        if code is None:
            return
        h = self.fleet.handles.get(code)
        if h is None:
            self.manual_code = None
            return

        held = {pygame.key.name(k) for k in self.keys_held}
        turn = (1.0 if "d" in held else 0.0) - (1.0 if "a" in held else 0.0)
        drive = (1.0 if "w" in held else 0.0) - (1.0 if "s" in held else 0.0)

        heading = self.manual_heading.get(code, 0.0) + turn * self.MANUAL_TURN * dt
        self.manual_heading[code] = heading % 360.0

        if drive == 0.0:
            h.set_velocity(np.zeros(2))
            return
        rad = np.radians(self.manual_heading[code])
        # y is down, so a positive angle reads as clockwise on screen and `d`
        # turns right, which is what a hand on WASD expects.
        direction = np.array([np.cos(rad), np.sin(rad)])
        commanded = direction * self.MANUAL_SPEED * drive
        h.set_velocity(commanded)

        # Log where it was told to go against where it actually went. This is
        # the only measurement of the aim-frame offset available: the camera
        # sees a glowing sphere with no facing, so heading can only be inferred
        # from travel.
        self.manual_track.append((np.asarray(h.pos, float).copy(),
                                  commanded.copy()))
        del self.manual_track[:-90]

    CALIBRATE_MIN_CM = 12.0      # travel needed before the direction is meaningful

    def start_calibration(self):
        """Drive a square and read this robot's frame offset off it.

        Preferred over the WASD-and-press-`c` route: four legs instead of one,
        so the noise averages down and — more usefully — disagreement between
        legs is diagnostic. Four legs that disagree mean the ball is slipping or
        the camera is following the wrong robot, which a single leg cannot tell
        you.
        """
        from fleet.heading import ActiveCalibration

        code = self.manual_code or next(
            (c for c, h in self.fleet.handles.items() if h.kind == "real"),
            next(iter(self.fleet.handles), None))
        if code is None:
            self.say("error", "no robots to calibrate")
            return
        h = self.fleet.handles.get(code)
        if h is None:
            return
        if not h.connected:
            self.say("error", f"{code} has no camera fix — calibration measures "
                              "where it actually went, so the tracker must see it")
            return

        self.select_manual(code)          # take it off the controller first
        self.keys_held.clear()
        # Measure the frame as it is, not as a live estimator is busy fixing
        # it. Leaving tracking on has the calibration converge on a robot that
        # has already corrected itself, so it reports no offset and the value
        # it writes to the roster is meaningless.
        h.heading_tracking = False
        self.calibrating = (code, ActiveCalibration(workspace=self.ws))
        self.say("info", f"calibrating {code} — driving a square, keep the area clear")

    def step_calibration(self, dt):
        """One frame of the calibration pattern. Returns True while running."""
        if self.calibrating is None:
            return False
        code, cal = self.calibrating
        h = self.fleet.handles.get(code)
        if h is None:
            self.calibrating = None
            return False

        v = cal.step(h.pos, dt)
        if v is not None:
            h.set_velocity(v)
            return True

        h.set_velocity(np.zeros(2))
        h.heading_tracking = True
        self.calibrating = None
        if cal.error:
            self.say("error", f"{code}: {cal.error}")
            if cal.offset is not None:
                self.say("info", f"measured {cal.offset:.0f}deg but did not "
                                 "apply it — fix the cause and run it again")
            return False

        # `cal.offset` is the ERROR between commanded and achieved, measured
        # with whatever offset was already in force. So it is subtracted from
        # the current value, not assigned over it — assigning stored the error
        # as if it were the correction and doubled the problem.
        h.heading_offset = (h.heading_offset - float(cal.offset)) % 360.0
        entry = self.roster.by_code(code)
        if entry is not None:
            entry.heading_offset = h.heading_offset
            self.roster.save()
        self.say("ok", f"{code} heading offset {h.heading_offset:.0f}deg "
                       f"(4 legs agreed within {cal.spread:.0f}deg) — saved")
        return False

    def calibrate_heading(self):
        """Solve this robot's aim-frame offset from how it actually travelled.

        Drive it with W for a second or two, then press `c`. The angle between
        where it was told to go and where the camera saw it go IS the offset;
        applying it makes every subsequent command land where intended, and
        removes the curved approach paths a stale offset produces.
        """
        from fleet.handle import velocity_to_command

        code = self.manual_code
        if code is None:
            return
        h = self.fleet.handles.get(code)
        if h is None:
            return
        if h.kind != "real":
            self.say("info", f"{code} is simulated — there is no aim frame to "
                             "calibrate")
            return

        track = [t for t in self.manual_track
                 if float(np.linalg.norm(t[1])) > 1e-6]
        if len(track) < 2:
            self.say("error", "drive it forward with W first, then press c")
            return

        travelled = track[-1][0] - track[0][0]
        if float(np.linalg.norm(travelled)) < self.CALIBRATE_MIN_CM:
            self.say("error", f"only moved {np.linalg.norm(travelled):.0f}cm — "
                              f"hold W until it has gone at least "
                              f"{self.CALIBRATE_MIN_CM:.0f}cm, then press c")
            return

        commanded = np.mean([t[1] for t in track], axis=0)
        if float(np.linalg.norm(commanded)) < 1e-6:
            self.say("error", "no consistent command to compare against")
            return

        want = velocity_to_command(commanded)[0]
        got = velocity_to_command(travelled)[0]
        error = (got - want + 180.0) % 360.0 - 180.0      # signed, -180..180
        new_offset = (h.heading_offset - error) % 360.0

        h.heading_offset = new_offset
        entry = self.roster.by_code(code)
        if entry is not None:
            entry.heading_offset = new_offset
            self.roster.save()
        self.manual_track = []
        self.say("ok", f"{code} heading offset {new_offset:.0f}deg "
                       f"(was off by {error:+.0f}deg over "
                       f"{np.linalg.norm(travelled):.0f}cm) — saved")

    def stop_all(self):
        """Halt the robots — and abort the agent, which would otherwise
        carry on issuing the rest of its plan into a stopped fleet."""
        if self.session is not None and self.session.busy:
            self.session.cancel()
            self.say("info", "STOP — aborting the model run")
        with self.sim_lock:
            self.ctx.stop_all()
        self.say("info", "STOP — every robot halted")

    def submit(self, line):
        self.say("cmd", "> " + line)
        if self.ask and self.session is not None and self.session.available:
            self.ask_model(line)
        else:
            self.run_tool(line)

    def run_tool(self, line):
        # Never a blocking acquire on the render thread. The lock is only ever
        # held for one tick, so this waits milliseconds in practice; the
        # timeout is there so that a tool wedged for any reason degrades to a
        # readable message instead of a frozen window.
        if not self.sim_lock.acquire(timeout=2.0):
            self.say("error", "the simulation is busy — try that again")
            return
        try:
            result = run_command(self.ctx, line)
        finally:
            self.sim_lock.release()
        self.say("ok" if result.get("ok") else "error", summarise(result))
        if result.get("ok") and self.mode != "navigate":
            self.mode = "navigate"        # a tool call means go where you were told
            self.sync()

    def ask_model(self, line):
        if self.session.busy:
            # The agent is one conversation on one thread, so a second natural
            # language request cannot start. A literal tool call is a different
            # matter: it touches the fleet directly, takes the shared lock for
            # a moment, and is exactly what someone reaches for when they want
            # to move another robot while a long command is still running.
            if parses_as_tool(line, codes=self.fleet.codes):
                self.say("info", "model is busy — running that as a direct "
                                 "tool call")
                self.run_tool(line)
                return
            self.say("error", "still working on the last request — press STOP "
                              "to abort it, or type a direct tool call "
                              "(e.g. move_to CRXS=60,40) to command another "
                              "robot meanwhile")
            return
        if not self.session.start(line):
            self.say("error", "could not start the model")
            return
        self.pending = line
        self.started_at = time.time()
        if self.mode != "navigate":
            self.mode = "navigate"
            self.sync()

    def drain_events(self):
        """Stream the agent's progress into the log. Called once a frame."""
        if self.session is None:
            return
        for e in self.session.poll():
            kind = e.get("type")
            if kind == "token":
                self.streamed = (self.streamed + e.get("text", ""))[-120:]
                continue
            if kind == "tool_start":
                self.streamed = ""
                args = e.get("args") or {}
                shown = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
                self.say("info", f"  {e['name']}({shown})")
            elif kind == "tool_result":
                r = e.get("result") or {}
                note = summarise(r) if r.get("ok") else (r.get("error") or "failed")
                self.say("ok" if e.get("ok") else "error",
                         f"    {note}  [{e.get('latency_s', 0):.1f}s]")
            elif kind == "reply":
                self.say("cmd", f"  {e['text']}")
            elif kind == "cap":
                self.say("error", f"  stopped at the {e['calls']}-call cap")
            elif kind == "error":
                self.say("error", f"  {e.get('error')}")
            elif kind == "done":
                self.pending, self.started_at, self.streamed = None, None, ""
                r = e.get("result")
                if r is not None and r.total_latency_s:
                    self.say("info", f"  {len(r.tool_calls)} call(s) in "
                                     f"{r.total_latency_s:.1f}s")

    # -- roster --------------------------------------------------------------

    def free_color(self):
        used = {h.color for h in self.fleet.handles.values()}
        for c in COLORS:
            if c not in used:
                return c
        return None

    def add_robot(self):
        color = self.free_color()
        if color is None:
            self.say("error", "all six tracked colours are in use — "
                              "the camera could not tell another robot apart")
            return
        n = len(self.roster.entries) + 1
        entry = RobotEntry(name=f"Robot{n}", code=f"R{n:03d}", kind="sim",
                           color=color)
        errors = self.fleet.add(entry)
        if errors:
            self.say("error", "; ".join(errors))
            return
        self.roster.entries.append(entry)
        self.roster.save()
        self.ctx.env.rebuild_if_needed()
        self.say("info", f"added {entry.code} ({color}) as sim")

    def remove_robot(self, code):
        errors = self.fleet.remove(code)
        if errors:
            self.say("error", "; ".join(errors))
            return
        self.roster.remove(code)
        self.roster.save()
        self.ctx.env.rebuild_if_needed()
        self.say("info", f"removed {code}")

    def toggle_kind(self, code):
        h = self.fleet.get(code)
        if h is None:
            return
        want = "sim" if h.kind == "real" else "real"
        entry = self.roster.by_code(code)
        ble = getattr(entry, "ble_name", None) or f"SK-{code[:4]}"
        errors = self.fleet.set_kind(code, want, ble_name=ble)
        if errors:
            self.say("error", "; ".join(errors))
            return
        if entry is not None:
            entry.kind = want
            entry.ble_name = ble if want == "real" else None
            self.roster.save()
        self.say("info", f"{code} is now {want}"
                         + (f" ({ble})" if want == "real" else ""))

    def start_edit(self, code, field):
        h = self.fleet.get(code)
        if h is None:
            return
        self.editing = (code, field)
        self.edit_buf = h.code if field == "code" else h.name
        self.cmd.focused = False

    def commit_edit(self):
        """Write an edited name or code back to the roster, or explain why not."""
        if self.editing is None:
            return
        code, field = self.editing
        value = self.edit_buf.strip()
        self.editing, self.edit_buf = None, ""

        entry = self.roster.by_code(code)
        if entry is None or not value:
            return
        old = entry.code if field == "code" else entry.name
        if value == old:
            return

        setattr(entry, field, value)
        errors = self.roster.save()          # validates names and codes together
        if errors:
            setattr(entry, field, old)
            self.say("error", "; ".join(errors))
            return

        if field == "code":
            errors = self.fleet.rename(code, new_code=value)
            if errors:
                setattr(entry, field, old)
                self.roster.save()
                self.say("error", "; ".join(errors))
                return
            self.ctx.env.rebuild_if_needed()
        else:
            self.fleet.rename(code, name=value)
        self.say("info", f"{old} renamed to {value}")

    def reload_roster(self):
        self.roster = Roster.load()
        if self.roster.errors:
            for e in self.roster.errors:
                self.say("error", e)
            return
        for code in list(self.fleet.handles):
            self.fleet.remove(code)
        for entry in self.roster.enabled_entries():
            errors = self.fleet.add(entry)
            if errors:
                self.say("error", "; ".join(errors))
        self.ctx.env.rebuild_if_needed()
        self.say("info", f"roster reloaded — {len(self.fleet)} robot(s)")

    # -- training -------------------------------------------------------------

    def cycle_len(self):
        opts = [60, 150, 400]
        self.train_updates = opts[(opts.index(self.train_updates) + 1) % len(opts)]
        self.len_btn.label = str(self.train_updates)

    def toggle_train(self):
        if self.train_thread and self.train_thread.is_alive():
            self.train_stop.set()
            return
        self.train_stop = threading.Event()
        self.progress = {"update": 0, "total": self.train_updates, "return": 0,
                         "coverage": 0, "collisions": 0, "history": []}
        cfg = dict(task="coverage", n_agents=max(len(self.fleet), 2),
                   updates=self.train_updates, n_envs=8, horizon=128)

        def go():
            try:
                train(cfg, on_progress=lambda s: setattr(self, "progress", s),
                      stop_event=self.train_stop)
            except Exception as e:
                self.say("error", f"training failed: {e}")
            finally:
                self.checkpoints = list_checkpoints()
                self.progress = None
                self.train_btn.label = "Train new policy"

        self.train_btn.label = "Stop training"
        self.train_thread = threading.Thread(target=go, daemon=True)
        self.train_thread.start()

    # -- geometry -------------------------------------------------------------

    def arena_rect(self):
        pad = 40
        avail_w, avail_h = W - DOCK - pad * 2, H - pad * 2 - 30
        scale = min(avail_w / max(self.ws.width, 1e-6),
                    avail_h / max(self.ws.height, 1e-6))
        w, h = self.ws.width * scale, self.ws.height * scale
        return pygame.Rect(DOCK + (W - DOCK - w) // 2, (H - h) // 2, int(w), int(h))

    def to_px(self, p):
        r = self.arena_rect()
        xmin, xmax, ymin, ymax = self.ws.bbox
        fx = (p[0] - xmin) / max(xmax - xmin, 1e-6)
        fy = (p[1] - ymin) / max(ymax - ymin, 1e-6)
        return (int(r.x + fx * r.w), int(r.y + fy * r.h))

    def to_cm(self, px):
        r = self.arena_rect()
        xmin, xmax, ymin, ymax = self.ws.bbox
        fx = (px[0] - r.x) / max(r.w, 1)
        fy = (px[1] - r.y) / max(r.h, 1)
        return np.array([xmin + fx * (xmax - xmin), ymin + fy * (ymax - ymin)])

    def scale_px(self):
        r = self.arena_rect()
        return r.w / max(self.ws.width, 1e-6)

    # -- drawing ---------------------------------------------------------------

    def draw_arena(self):
        s = self.screen
        r = self.arena_rect()

        poly = [self.to_px(p) for p in self.ws.bounds_cm]
        floor = self.floor_view()
        if floor is not None:
            r = self.arena_rect()
            s.blit(floor, (r.x, r.y))
        else:
            pygame.draw.polygon(s, PANEL, poly)

        step = 20.0
        xmin, xmax, ymin, ymax = self.ws.bbox
        x = xmin
        while x <= xmax + 1e-6:
            a, b = self.to_px((x, ymin)), self.to_px((x, ymax))
            pygame.draw.line(s, (26, 62, 94), a, b)
            x += step
        y = ymin
        while y <= ymax + 1e-6:
            a, b = self.to_px((xmin, y)), self.to_px((xmax, y))
            pygame.draw.line(s, (26, 62, 94), a, b)
            y += step
        pygame.draw.polygon(s, RULE, poly, 2)

        for o in self.ws.obstacles:
            self.draw_shape(o, CORAL, fill=(46, 30, 44))

        # Entities are drawn from the same shape dicts, at wherever they have
        # moved to. A target-only entity is outlined and not filled: robots
        # pass straight through it, and a filled shape reads as "do not enter".
        for e in self.ws.entities:
            solid = e.blocks
            edge = CORAL if solid else SUN
            self.draw_trail(e)
            self.draw_shape(e.shape(), edge, fill=(46, 30, 44) if solid else None)
            if e.role == "both":
                # Hatched: it is two things at once, and a single colour would
                # claim it is only one of them.
                self.draw_hatch(e, SUN)
            label = e.id if solid and e.role != "both" else f"{e.id} ({e.role})"
            s.blit(self.fs.render(label, True, edge),
                   self.to_px(e.pos + np.array([e.radius, -e.radius])))

        state = self.fleet.state()
        self.draw_motion(state)
        self.draw_manual()
        self.draw_camera_bounds()
        self.draw_camera()

        for code, d in state.items():
            target = d.get("target")
            if target is None:
                continue
            tp = self.to_px(target)
            rp = self.to_px(d["pos"])
            # No label here: the line already says whose target this is, and a
            # robot sitting on its target would print its code twice over.
            pygame.draw.line(s, (38, 74, 104), rp, tp, 1)
            pygame.draw.circle(s, SUN, tp, 7, 1)

        for code, d in state.items():
            self.draw_robot(code, d)

        info = [f"{self.ws.width:.0f}x{self.ws.height:.0f}cm",
                f"grid {step:.0f}cm"]
        if self.paused:
            info.append("PAUSED")
        s.blit(self.f.render("   ".join(info), True, DIM), (r.x, r.bottom + 10))
        s.blit(self.fb.render(f"controller: {self.mode}"
                              + (f" ({self.sel})" if self.mode == "policy" else ""),
                              True, CHALK), (r.x, r.y - 28))

    TRAIL_LEN = 45

    def draw_trail(self, e):
        """Where it has been. A moving thing drawn only at its current position
        is indistinguishable from a still one in a screenshot — and from a
        still one in the corner of your eye, which is worse."""
        if e.motion.get("kind", "static") == "static":
            return
        pts = self.trails.setdefault(e.id, [])
        p = (float(e.pos[0]), float(e.pos[1]))
        if not pts or abs(pts[-1][0] - p[0]) + abs(pts[-1][1] - p[1]) > 0.5:
            pts.append(p)
            del pts[:-self.TRAIL_LEN]
        if len(pts) < 2:
            return
        px = [self.to_px(q) for q in pts]
        pygame.draw.lines(self.screen, (58, 48, 62), False, px, 1)

    def draw_hatch(self, e, color):
        """Diagonal strokes across an entity that is both target and obstacle."""
        c = np.asarray(e.pos, dtype=float)
        r = e.radius
        for f in (-0.5, 0.0, 0.5):
            a = self.to_px(c + np.array([-r * 0.7, f * r]))
            b = self.to_px(c + np.array([r * 0.7, f * r]))
            pygame.draw.line(self.screen, color, a, b, 1)

    def draw_motion(self, state):
        """Paths, follow relationships and stalls — what the layers are doing.

        Reading this off the roster panel means holding six layer stacks in
        your head; drawn in the arena it is just there.
        """
        s = self.screen
        stack = self.ctx.stack
        for code in stack.codes():
            h = self.fleet.handles.get(code)
            if h is None:
                continue
            here = self.to_px(h.pos)
            for layer in stack.layers(code):
                p = layer.params or {}
                if layer.kind == "path":
                    wps = p.get("waypoints") or []
                    if len(wps) >= 2:
                        px = [self.to_px(q) for q in wps]
                        pygame.draw.lines(s, (52, 96, 128), False, px, 1)
                        leg = layer.state.get("leg", 0)
                        for i, q in enumerate(px):
                            pygame.draw.circle(s, SUN if i == leg else (52, 96, 128),
                                               q, 3, 0 if i == leg else 1)
                elif layer.kind == "follow":
                    target = p.get("target")
                    tp = self._point_of(target)
                    if tp is not None:
                        pygame.draw.line(s, (70, 110, 90), here, self.to_px(tp), 1)
                elif layer.kind == "flow" and p.get("entity"):
                    tp = self._point_of(p["entity"])
                    if tp is not None:
                        pygame.draw.line(s, (96, 80, 130), here, self.to_px(tp), 1)

            if stack.stalled(code):
                pygame.draw.circle(s, CORAL, here, 16, 2)
                s.blit(self.fs.render("stalled", True, CORAL),
                       (here[0] + 12, here[1] - 22))

    def draw_manual(self):
        """Ring the hand-driven robot and show the heading you are steering.

        Without the arrow there is no way to tell which way W will go: a
        uniformly glowing sphere looks identical at every heading, on screen
        exactly as it does to the camera.
        """
        code = self.manual_code
        if code is None:
            return
        h = self.fleet.handles.get(code)
        if h is None:
            return
        s = self.screen
        here = self.to_px(h.pos)
        pygame.draw.circle(s, MINT, here, 18, 2)

        rad = np.radians(self.manual_heading.get(code, 0.0))
        tip = h.pos + np.array([np.cos(rad), np.sin(rad)]) * 26.0
        pygame.draw.line(s, MINT, here, self.to_px(tip), 2)
        s.blit(self.fs.render(f"manual {code}", True, MINT),
               (here[0] + 14, here[1] + 12))

    def _point_of(self, ident):
        """Position of a robot code or an entity id, or None."""
        if not ident:
            return None
        e = self.ws.entities.by_id(ident)
        if e is not None:
            return e.pos
        h = self.fleet.handles.get(ident)
        return h.pos if h is not None else None

    CAMERA_VIEWS = ("panel", "floor", "off")

    def cycle_camera_view(self):
        i = self.CAMERA_VIEWS.index(self.camera_view)
        self.camera_view = self.CAMERA_VIEWS[(i + 1) % len(self.CAMERA_VIEWS)]
        self.say("info", f"camera view: {self.camera_view}")

    def floor_view(self):
        """The camera frame warped into the arena rectangle, or None.

        This is the arena the robots are actually on. Planning against an
        invented 240x180 rectangle that corresponds to nothing on the floor
        means every position is a guess; warping the real surface underneath
        the simulation puts sim robots and real ones in the same picture, and
        makes a bad calibration obvious the moment the grid stops lining up
        with the tape.
        """
        if self.camera_view != "floor" or self.tracker is None:
            return None
        h = self.homography
        if h is None or not getattr(h, "ready", False):
            return None
        try:
            frame, _ = self.tracker.latest()
        except Exception:
            return None
        if frame is None:
            return None

        stamp = id(frame)
        rect = self.arena_rect()
        if stamp == self.floor_stamp and self.floor_surface is not None \
                and self.floor_surface.get_size() == (rect.w, rect.h):
            return self.floor_surface

        import cv2

        xmin, ymin = self.ws.bbox[0], self.ws.bbox[2]
        scale = self.scale_px()
        # cm -> arena-local pixels, composed with the camera's pixel -> cm map.
        to_local = np.array([[scale, 0.0, -xmin * scale],
                             [0.0, scale, -ymin * scale],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            warped = cv2.warpPerspective(frame, to_local @ h.M, (rect.w, rect.h))
        except Exception:
            return None
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf.set_alpha(150)          # dimmed: the robots are the subject
        self.floor_surface, self.floor_stamp = surf, stamp
        return surf

    def draw_camera_bounds(self):
        """Outline the rectangle the camera is calibrated for, when it differs.

        Two coordinate systems that are supposed to describe the same
        rectangle, drawn on top of each other. When they disagree the picture
        says so instantly — a 200x200 calibration under a 240x180 arena shows
        as a square hanging off the bottom and stopping short of the right wall
        — where the numbers alone ("arena=200") mean nothing until someone goes
        and reads the code.
        """
        h = self.homography
        if h is None or not getattr(h, "ready", False):
            return
        if h.matches(self.ws.width, self.ws.height):
            return                      # they agree; drawing it twice says nothing

        s = self.screen
        xmin, ymin = self.ws.bbox[0], self.ws.bbox[2]
        corners = [(xmin, ymin), (xmin + h.width, ymin),
                   (xmin + h.width, ymin + h.height), (xmin, ymin + h.height)]
        px = [np.array(self.to_px(c), dtype=float) for c in corners]

        # Dashed, so it never reads as another obstacle.
        for i in range(4):
            a, b = px[i], px[(i + 1) % 4]
            seg = b - a
            steps = max(2, int(np.linalg.norm(seg) / 12))
            for k in range(0, steps, 2):
                pygame.draw.line(s, SUN, a + seg * (k / steps),
                                 a + seg * min((k + 1) / steps, 1.0), 1)
        s.blit(self.fs.render(f"camera sees {h.width:.0f}x{h.height:.0f}cm — "
                              f"arena is {self.ws.width:.0f}x{self.ws.height:.0f}",
                              True, SUN), (px[0][0] + 6, px[0][1] + 6))

    CAM_W = 300                 # px wide; the frame is scaled down to this

    def camera_surface(self):
        """Latest frame as a pygame surface, plus the blobs found in it.

        Converted at most once per new frame: the render loop runs faster than
        the camera, and re-converting the same image every frame is pure heat
        for no extra information.
        """
        if self.tracker is None:
            return None, {}, 1.0
        try:
            frame, raw = self.tracker.latest()
        except Exception:
            return None, {}, 1.0
        if frame is None:
            return None, {}, 1.0

        h, w = frame.shape[:2]
        f = self.CAM_W / float(w)              # source px -> panel px
        stamp = id(frame)
        if stamp != self.cam_stamp:
            import cv2
            small = cv2.resize(frame, (self.CAM_W, max(1, int(h * f))))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            # pygame wants (x, y, colour); OpenCV gives (row, col, colour).
            self.cam_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self.cam_stamp = stamp
        return self.cam_surface, raw, f

    def draw_camera(self):
        """Picture-in-picture of what the camera sees, with the blobs it found.

        Bring-up is mostly answering "is it detecting the ball at all?", and the
        top-down arena cannot answer that: a robot the detector has missed and a
        robot that is not there are drawn identically.
        """
        if self.camera_view != "panel":
            return
        surf, raw, f = self.camera_surface()
        if surf is None:
            return

        s = self.screen
        r = self.arena_rect()
        x = r.right - surf.get_width() - 8
        y = r.bottom - surf.get_height() - 8
        s.blit(surf, (x, y))
        pygame.draw.rect(s, RULE, (x, y, surf.get_width(), surf.get_height()), 1)

        for name, blob in (raw or {}).items():
            bx, by, area = blob
            px = (int(x + bx * f), int(y + by * f))
            radius = max(4, int(np.sqrt(max(area, 1.0) / np.pi) * f) + 2)
            pygame.draw.circle(s, LED_RGB.get(name, CHALK), px, radius, 2)

        fps = getattr(self.tracker, "fps", 0.0) or 0.0
        found = len(raw or {})
        s.blit(self.fs.render(f"camera {fps:.0f}fps  {found} blob(s)  v to hide",
                              True, DIM), (x, y - 14))
        if not found:
            s.blit(self.fs.render("nothing detected — light a robot (set_led), "
                                  "or tune thresholds", True, CORAL), (x, y - 27))

    def draw_shape(self, o, edge, fill=None):
        """One obstacle-shaped dict — used for static obstacles and entities alike."""
        s = self.screen
        if o.get("type") == "circle":
            c = self.to_px(o["center"])
            rad = max(2, int(o["radius"] * self.scale_px()))
            if fill is not None:
                pygame.draw.circle(s, fill, c, rad)
            pygame.draw.circle(s, edge, c, rad, 1)
        else:
            pts = [self.to_px(p) for p in o["points"]]
            if fill is not None:
                pygame.draw.polygon(s, fill, pts)
            pygame.draw.polygon(s, edge, pts, 1)

    def draw_robot(self, code, d):
        """The one place in the codebase that distinguishes sim from real."""
        s = self.screen
        c = np.array(self.to_px(d["pos"]), dtype=float)
        col = LED_RGB.get(d["color"], CHALK)
        live = d["connected"]
        v = np.array(d["vel"], dtype=float)
        a = np.arctan2(v[1], v[0]) if np.linalg.norm(v) > 2 else -np.pi / 2

        size = 11
        pts = [c + size * np.array([np.cos(a + k), np.sin(a + k)])
               for k in (0.0, 2.4, -2.4)]
        pts = [tuple(map(int, p)) for p in pts]

        if live:
            pygame.draw.polygon(s, col, pts)
        else:
            pygame.draw.polygon(s, GREY, pts, 1)      # hollow, last known place

        if d["kind"] == "real":
            pygame.draw.circle(s, col if live else GREY,
                               (int(c[0]), int(c[1])), size + 5, 1)

        self.draw_heading(c, d, col, live)

        label = code if live else f"{code} ?"
        s.blit(self.fs.render(label, True, col if live else GREY),
               (int(c[0]) + size + 6, int(c[1]) - 6))

    def draw_heading(self, c, d, col, live):
        """Which way the robot believes it is aimed, faded by how sure it is.

        Drawn because a stale heading and a fresh one are identical from the
        robot's behaviour until it is asked to follow a shape, at which point
        the shape comes out wrong and the cause is three layers away. A short
        arrow that visibly dims is the cheapest way to make that visible while
        it is still only a suspicion.
        """
        h = d.get("heading") or {}
        conf = float(h.get("confidence") or 0.0)
        if not live or conf <= 0.05:
            return
        s = self.screen
        # The aim frame in the camera's terms: the offset IS the rotation
        # between them, so it is the arrow.
        rad = np.radians(90.0 - float(h.get("applied_deg") or 0.0))
        tip = c + 26.0 * np.array([np.cos(rad), np.sin(rad)])
        fade = tuple(int(GREY[i] + (col[i] - GREY[i]) * min(1.0, conf))
                     for i in range(3))
        pygame.draw.line(s, fade, tuple(map(int, c)), tuple(map(int, tip)), 2)
        for k in (2.6, -2.6):
            barb = tip + 7.0 * np.array([np.cos(rad + k), np.sin(rad + k)])
            pygame.draw.line(s, fade, tuple(map(int, tip)), tuple(map(int, barb)), 2)
        if (h.get("drifted_deg") or 0.0) > 30.0:
            # Thirty degrees from the calibrated value is not drift any more.
            # It is a robot that was picked up, bumped, or is being tracked as
            # somebody else, and none of those is something the loop can fix.
            pygame.draw.circle(s, SUN, tuple(map(int, c)), 18, 1)

    # -- dock ------------------------------------------------------------------

    def draw_dock(self):
        s = self.screen
        pygame.draw.rect(s, (9, 28, 47), pygame.Rect(0, 0, DOCK, H))
        pygame.draw.line(s, RULE, (DOCK, 0), (DOCK, H))
        x, w = 14, DOCK - 28

        st = self.fleet.status()
        s.blit(self.fb.render("SWARM LAB", True, CHALK), (x, 12))

        # Connection dot and the one word that says whether anything is live.
        live = st["connected"] > 0
        pygame.draw.circle(s, MINT if live else GREY, (x + w - 78, 20), 4)
        s.blit(self.fs.render("CONNECTED" if live else "OFFLINE", True,
                              MINT if live else GREY), (x + w - 68, 14))

        y = self.draw_telemetry(x, w, 38)

        for b in self.buttons:
            b.draw(s, self.f)

        y = self.draw_roster(x, w, 124)
        y = self.draw_obstacles_panel(x, w, y + 12)
        y = self.draw_entities_panel(x, w, y + 10)
        y = self.draw_formations(x, w, y + 10)
        y = self.draw_checkpoints(x, w, y + 10)
        self.draw_log(x, w, y + 12)
        self.cmd.draw(self.screen, self.f)
        self.draw_thinking(x, w)
        self.draw_progress(x, w)

    def draw_telemetry(self, x, w, y):
        """The four numbers worth glancing at, as tiles rather than a sentence.

        They were three lines of run-together text — "6/6 connected   real 1/3
        model sonnet5   tracker 30fps" — which is everything the dashboard
        knows arranged so that none of it stands out.
        """
        s = self.screen
        st = self.fleet.status()
        tw = (w - 8) // 2
        rows = [
            ("robots", f"{st['connected']}/{st['robots']}", None),
            ("real linked", f"{st['real_linked']}/{st['real']}" if st["real"]
             else "—", CORAL if st["max_connections_hit"] else None),
            ("tracker", f"{st['tracker_fps']:.0f}fps" if st["tracker_fps"]
             else "off", None),
            ("ble", f"{st['mean_rtt_ms']:.0f}ms" if st["mean_rtt_ms"]
             else "idle", None),
        ]
        for i, (label, value, tone) in enumerate(rows):
            r = pygame.Rect(x + (i % 2) * (tw + 8), y + (i // 2) * 34, tw, 30)
            stat_tile(s, self.fs, self.f, r, label, value, tone)
        y += 72

        # The model line stays prose: which model answered is a sentence, and
        # a fallback needs the word "fallback" next to the name it replaced.
        if self.session is not None and self.session.available:
            text = f"model {self.model_preset}"
            tone = CHALK
            if self.fell_back_from:
                text += f"  (fallback from {self.fell_back_from})"
                tone = SUN
        else:
            text, tone = self.model_status(), GREY
        if self.manual_code:
            text += f"   MANUAL {self.manual_code}"
            tone = MINT
        s.blit(self.fs.render(text, True, tone), (x, y))
        if st["max_connections_hit"]:
            s.blit(self.fs.render("BLE max connections reached", True, CORAL),
                   (x, y + 13))
        return y + 16

    def model_status(self):
        if self.session is None:
            return "model none"
        name = self.session.model_name
        if not self.session.available:
            return f"model {name} unreachable"
        mean = self.session.mean_latency
        return f"model {name}" + (f"   {mean:.1f}s mean" if mean else "   ready")

    def draw_thinking(self, x, w):
        """A visibly working UI. At 9B a turn is seconds, and a frozen-looking
        window is the top cause of someone hammering the button."""
        if not (self.session is not None and self.session.busy):
            return
        y = H - 112
        elapsed = time.time() - (self.started_at or time.time())
        dots = "." * (int(elapsed * 3) % 4)
        label = f"thinking{dots}  {elapsed:.1f}s"
        if self.streamed:
            label = self.streamed.replace("\n", " ")[-46:]
        s = self.screen
        box = pygame.Rect(x + 114, y, w - 114, 22)
        pygame.draw.rect(s, PANEL2, box, border_radius=3)
        pygame.draw.rect(s, SUN, box, 1, border_radius=3)
        s.blit(self.fs.render(label, True, SUN), (box.x + 8, box.y + 5))
        span = int((elapsed * 60) % max(box.w - 40, 1))
        pygame.draw.rect(s, SUN, (box.x + 6 + span, box.y + 18, 28, 2))

    def draw_roster(self, x, w, y):
        s = self.screen
        y = section(s, self.fs, "roster", x, y, w) - 24
        s.blit(self.fs.render("", True, DIM), (x, y))
        y += 22
        self.roster_rows = []
        for code, h in self.fleet.handles.items():
            r = pygame.Rect(x, y, w, 24)
            pygame.draw.rect(s, PANEL, r, border_radius=3)

            swatch = pygame.Rect(x + 6, y + 7, 10, 10)
            pygame.draw.rect(s, LED_RGB.get(h.color, CHALK), swatch,
                             border_radius=2)

            code_rect = pygame.Rect(x + 22, y + 4, 40, 16)
            name_rect = pygame.Rect(x + 64, y + 4, 96, 16)
            for rect, field, text, tone in ((code_rect, "code", code, CHALK),
                                            (name_rect, "name", h.name[:12], DIM)):
                if self.editing == (code, field):
                    pygame.draw.rect(s, INK, rect, border_radius=2)
                    pygame.draw.rect(s, CYAN, rect, 1, border_radius=2)
                    shown = self.edit_buf[-12:] + "_"
                    s.blit(self.fs.render(shown, True, CHALK), (rect.x + 3, rect.y + 2))
                else:
                    s.blit(self.fs.render(text, True, tone), (rect.x + 2, rect.y + 2))

            kind_rect = pygame.Rect(x + w - 92, y + 3, 40, 18)
            on = h.kind == "real"
            pygame.draw.rect(s, PANEL2 if on else INK, kind_rect, border_radius=3)
            pygame.draw.rect(s, CYAN if on else RULE, kind_rect, 1, border_radius=3)
            s.blit(self.fs.render(h.kind, True, CHALK if on else DIM),
                   (kind_rect.x + 7, kind_rect.y + 3))

            dot = (x + w - 44, y + 12)
            pygame.draw.circle(s, MINT if h.connected else CORAL, dot, 4)

            head = h.heading_state()
            conf = float(head.get("confidence") or 0.0)
            drifted = float(head.get("drifted_deg") or 0.0)
            tone = (SUN if drifted > 30.0 else
                    MINT if conf > 0.4 else DIM if conf > 0.05 else RULE)
            s.blit(self.fs.render(f"{head.get('applied_deg', 0):.0f}\u00b0", True, tone),
                   (x + w - 138, y + 6))

            del_rect = pygame.Rect(x + w - 26, y + 3, 20, 18)
            pygame.draw.rect(s, INK, del_rect, border_radius=3)
            pygame.draw.rect(s, RULE, del_rect, 1, border_radius=3)
            s.blit(self.fs.render("x", True, DIM), (del_rect.x + 7, del_rect.y + 3))

            self.roster_rows.append({"kind": kind_rect, "del": del_rect,
                                      "code_field": code_rect,
                                      "name_field": name_rect, "code": code})
            y += 27
        if not self.fleet.handles:
            s.blit(self.fs.render("empty — press + add", True, RULE), (x, y))
            y += 24
        return y

    # -- obstacles ------------------------------------------------------------

    def add_obstacle(self):
        """Drop a new circle somewhere valid, clear of the robots."""
        from workspace.space import make_circle

        from workspace.space import obstacle_center, obstacle_size

        radius = 15.0
        xmin, xmax, ymin, ymax = self.ws.bbox
        # keep the whole body inside: a valid centre is not enough, the shape
        # around it has to fit too, or it hangs over the wall
        lo = np.array([xmin + radius + 2, ymin + radius + 2])
        hi = np.array([xmax - radius - 2, ymax - radius - 2])

        best, best_gap = None, -1.0
        for _ in range(80):
            p = np.clip(self.ws.random_valid_point(), lo, hi)
            gaps = [float(np.linalg.norm(p - h.pos)) for h in self.fleet.handles.values()]
            gaps += [float(np.linalg.norm(p - obstacle_center(o))) - obstacle_size(o)
                     for o in self.ws.obstacles]
            gap = min(gaps) if gaps else 1e9
            if gap > best_gap:
                best, best_gap = p, gap
        if best is None:
            best = np.clip(self.ws.centroid(), lo, hi)
        self.ws.obstacles.append(make_circle(best, radius))
        self.save_workspace(f"added a circle at ({best[0]:.0f}, {best[1]:.0f})")

    # -- entities ---------------------------------------------------------

    ENTITY_ROLES = ("target", "obstacle", "both")

    def add_entity(self):
        """Drop a moving entity that patrols across the arena.

        Defaults to `target` because that is the role with no other way in: a
        static thing to avoid is already one click away on the obstacles panel,
        whereas a followable thing previously meant hand-editing workspace.json.
        """
        from workspace.entities import Entity

        xmin, xmax, ymin, ymax = self.ws.bbox
        radius = 12.0
        my = (ymin + ymax) / 2.0
        a = (xmin + radius + 20, my)
        b = (xmax - radius - 20, my)

        n = 1
        existing = {e.id for e in self.ws.entities}
        while f"wanderer{n}" in existing:
            n += 1
        eid = "wanderer" if "wanderer" not in existing else f"wanderer{n}"

        entity = Entity(id=eid, role="target",
                        shape={"type": "circle", "center": list(a),
                               "radius": radius},
                        motion={"kind": "path", "waypoints": [list(a), list(b)],
                                "mode": "pingpong", "speed": 25.0})
        errors = self.ws.entities.add(entity)
        if errors:
            self.say("error", "; ".join(errors))
            return
        self.save_workspace(f"added {eid} — a followable entity patrolling the "
                            "middle. Try: follow the wanderer")

    def cycle_entity_role(self, eid):
        """target -> obstacle -> both. The role decides everything about it."""
        e = self.ws.entities.by_id(eid)
        if e is None:
            return
        i = self.ENTITY_ROLES.index(e.role) if e.role in self.ENTITY_ROLES else 0
        e.role = self.ENTITY_ROLES[(i + 1) % len(self.ENTITY_ROLES)]
        self.trails.pop(eid, None)
        self.save_workspace(f"{eid} is now {e.role}")

    def remove_entity(self, eid):
        errors = self.ws.entities.remove(eid)
        if errors:
            self.say("error", "; ".join(errors))
            return
        self.trails.pop(eid, None)
        # A layer aimed at something that no longer exists holds position by
        # design, but leaving it running is a robot waiting on a ghost.
        for code in list(self.ctx.stack.codes()):
            for layer in list(self.ctx.stack.layers(code)):
                p = layer.params or {}
                if eid in (p.get("target"), p.get("entity")):
                    self.ctx.stack.pop_layer(code, layer.name)
        self.save_workspace(f"removed {eid}")

    def remove_obstacle(self, i):
        if 0 <= i < len(self.ws.obstacles):
            del self.ws.obstacles[i]
            self.save_workspace(f"removed obstacle {i + 1}")

    def cycle_obstacle_shape(self, i):
        from workspace.space import obstacle_label, rebuild_obstacle

        o = self.ws.obstacles[i]
        kind = "poly" if o.get("type") == "circle" else "circle"
        self.ws.obstacles[i] = rebuild_obstacle(o, kind=kind)
        self.save_workspace(f"obstacle {i + 1} is now a "
                            f"{obstacle_label(self.ws.obstacles[i])}")

    def resize_obstacle(self, i, delta):
        from workspace.space import obstacle_center, obstacle_size, rebuild_obstacle

        o = self.ws.obstacles[i]
        # min, not max: a shape wider than the arena's *shorter* side can never
        # fit inside it, however you place the centre.
        size = max(3.0, min(obstacle_size(o) + delta, min(self.ws.width,
                                                           self.ws.height) / 2))
        # Growing near a wall would push the body through it. Clamping the size
        # alone is not enough — the centre has to move in as the shape grows.
        self.ws.obstacles[i] = rebuild_obstacle(
            o, center=self._fit_center(obstacle_center(o), size), size=size)
        self.save_workspace(f"obstacle {i + 1} size {size:.0f}cm")

    def _fit_center(self, center, size):
        """Nearest centre that keeps a shape of `size` wholly inside the arena."""
        xmin, xmax, ymin, ymax = self.ws.bbox
        lo = np.array([xmin + size, ymin + size])
        hi = np.array([xmax - size, ymax - size])
        lo = np.minimum(lo, hi)                 # arena smaller than the shape
        return np.clip(np.asarray(center, dtype=float), lo, hi)

    def move_obstacle(self, i, center):
        from workspace.space import obstacle_size, rebuild_obstacle

        o = self.ws.obstacles[i]
        c = self._fit_center(center, obstacle_size(o))
        self.ws.obstacles[i] = rebuild_obstacle(o, center=c)

    def save_workspace(self, note=None):
        """Persist and tell the user. Obstacles are shared state, not a view."""
        errors = self.ws.save()
        if errors:
            self.say("error", "; ".join(errors))
        elif note:
            self.say("info", note)

    def obstacle_at(self, pos_cm):
        """Index of the obstacle under a point in centimetres, or None."""
        from workspace.space import obstacle_center, obstacle_size

        for i, o in enumerate(self.ws.obstacles):
            if float(np.linalg.norm(np.asarray(pos_cm) - obstacle_center(o))) \
                    <= obstacle_size(o) + 2.0:
                return i
        return None

    def draw_obstacles_panel(self, x, w, y):
        from workspace.space import obstacle_label, obstacle_size

        s = self.screen
        y = section(s, self.fs, "obstacles", x, y, w) - 24
        s.blit(self.fs.render("", True, DIM), (x, y))
        self.obstacle_rows = []

        self.obs_add_btn.rect.topleft = (x + w - 56, y - 4)
        self.obs_add_btn.draw(s, self.fs)
        y += 20

        if not self.ws.obstacles:
            s.blit(self.fs.render("none — drag one in the arena to move it",
                                  True, RULE), (x, y + 2))
            return y + 18

        for i, o in enumerate(self.ws.obstacles[:5]):
            r = pygame.Rect(x, y, w, 20)
            pygame.draw.rect(s, PANEL, r, border_radius=3)

            shape = pygame.Rect(x + 4, y + 2, 54, 16)
            minus = pygame.Rect(x + 132, y + 2, 20, 16)
            plus = pygame.Rect(x + 156, y + 2, 20, 16)
            dele = pygame.Rect(x + w - 22, y + 2, 18, 16)

            pygame.draw.rect(s, PANEL2, shape, border_radius=2)
            s.blit(self.fs.render(obstacle_label(o)[:6], True, CHALK),
                   (shape.x + 5, shape.y + 2))
            s.blit(self.fs.render(f"{obstacle_size(o):.0f}cm", True, DIM),
                   (x + 66, y + 3))
            for rect, label in ((minus, "-"), (plus, "+")):
                pygame.draw.rect(s, PANEL2, rect, border_radius=2)
                t = self.fs.render(label, True, CHALK)
                s.blit(t, t.get_rect(center=rect.center))
            t = self.fs.render("x", True, CORAL)
            s.blit(t, t.get_rect(center=dele.center))

            self.obstacle_rows.append({"i": i, "shape": shape, "minus": minus,
                                        "plus": plus, "del": dele})
            y += 22

        if len(self.ws.obstacles) > 5:
            s.blit(self.fs.render(f"+{len(self.ws.obstacles) - 5} more in "
                                  "workspace.json", True, RULE), (x, y))
            y += 16
        return y

    def draw_entities_panel(self, x, w, y):
        s = self.screen
        y = section(s, self.fs, "entities", x, y, w) - 24
        s.blit(self.fs.render("", True, DIM), (x, y))
        self.entity_rows = []

        self.ent_add_btn.rect.topleft = (x + w - 56, y - 4)
        self.ent_add_btn.draw(s, self.fs)
        y += 20

        ents = list(self.ws.entities)
        if not ents:
            s.blit(self.fs.render("none — + ent adds a followable one",
                                  True, RULE), (x, y + 2))
            return y + 18

        for e in ents[:4]:
            r = pygame.Rect(x, y, w, 20)
            pygame.draw.rect(s, PANEL, r, border_radius=3)

            role = pygame.Rect(x + w - 92, y + 2, 62, 16)
            dele = pygame.Rect(x + w - 22, y + 2, 18, 16)

            s.blit(self.fs.render(e.id[:12], True, CHALK), (x + 5, y + 3))
            pygame.draw.rect(s, PANEL2, role, border_radius=2)
            t = self.fs.render(e.role, True, SUN if e.followable else CORAL)
            s.blit(t, t.get_rect(center=role.center))
            t = self.fs.render("x", True, CORAL)
            s.blit(t, t.get_rect(center=dele.center))

            self.entity_rows.append({"id": e.id, "role": role, "del": dele})
            y += 22

        if len(ents) > 4:
            s.blit(self.fs.render(f"+{len(ents) - 4} more in workspace.json",
                                  True, RULE), (x, y))
            y += 16
        return y

    def draw_formations(self, x, w, y):
        s = self.screen
        y = section(s, self.fs, "formations", x, y, w) - 24
        s.blit(self.fs.render("", True, DIM), (x, y))
        y += 22
        self.formation_rows = []
        entries = self.ctx.library.listing()
        if not entries:
            s.blit(self.fs.render("none saved yet", True, RULE), (x, y))
            return y + 20

        for f in entries[:4]:
            r = pygame.Rect(x, y, w, 34)
            pygame.draw.rect(s, PANEL, r, border_radius=3)
            pygame.draw.rect(s, RULE, r, 1, border_radius=3)

            thumb = pygame.Rect(x + 4, y + 4, 26, 26)
            pygame.draw.rect(s, INK, thumb, border_radius=2)
            pts = self.ctx.library.get(f["name"])["points"]
            for p in pts:
                px = int(thumb.centerx + p[0] * 10)
                py = int(thumb.centery + p[1] * 10)
                pygame.draw.circle(s, CYAN, (px, py), 2)

            s.blit(self.fs.render(f["name"][:16], True, CHALK), (x + 36, y + 4))
            s.blit(self.fs.render(f"{f['robots']} robots  {f['description'][:14]}",
                                  True, DIM), (x + 36, y + 19))

            btn = pygame.Rect(x + w - 58, y + 7, 52, 20)
            pygame.draw.rect(s, PANEL2, btn, border_radius=3)
            pygame.draw.rect(s, CYAN, btn, 1, border_radius=3)
            s.blit(self.fs.render("recall", True, CHALK), (btn.x + 8, btn.y + 4))

            self.formation_rows.append((btn, f["name"]))
            y += 38
        return y

    def draw_checkpoints(self, x, w, y):
        s = self.screen
        s.blit(self.fs.render("trained policies", True, DIM), (x, y))
        y += 18
        self.ck_rows = []
        if not self.checkpoints:
            s.blit(self.fs.render("none yet", True, RULE), (x, y))
            return y + 18
        for ck in self.checkpoints[:2]:
            r = pygame.Rect(x, y, w, 22)
            on = ck["name"] == self.sel
            pygame.draw.rect(s, PANEL2 if on else PANEL, r, border_radius=3)
            pygame.draw.rect(s, CYAN if on else RULE, r, 1, border_radius=3)
            s.blit(self.fs.render(ck["name"][:34], True, CHALK), (r.x + 6, r.y + 5))
            self.ck_rows.append((r, ck))
            y += 25
        return y

    def draw_log(self, x, w, top):
        s = self.screen
        # End above the topmost control below the log, derived from the widgets
        # themselves. A hardcoded offset here silently drew the log on top of
        # the ask/tool buttons the moment that row was added.
        bottom = min(b.rect.top for b in (self.ask_btn, self.tool_btn,
                                          self.train_btn, self.cmd,
                                          self.stop_btn)) - 8
        if self.progress:
            bottom -= 46

        # Never force a minimum height: on a short window that pushes the panel
        # straight back over the controls it was moved off. Better to show a
        # cramped log, or none, than to cover the buttons.
        height = bottom - top
        if height < 24:
            self.log_rect = pygame.Rect(x, top, w, 0)
            return
        area = pygame.Rect(x, top, w, height)
        self.log_rect = area
        pygame.draw.rect(s, (7, 24, 40), area, border_radius=3)

        colours = {"error": CORAL, "ok": MINT, "cmd": CYAN, "info": DIM}
        lines = []
        for kind, text in self.log[-LOG_HISTORY:]:
            for chunk in _wrap(text, 46):
                lines.append((kind, chunk))

        visible = max(1, area.h // 14)
        # `log_scroll` counts lines back from the newest. Clamped here rather
        # than at the scroll event, because the log grows underneath you and a
        # stale offset would scroll past the end.
        self.log_max_scroll = max(0, len(lines) - visible)
        self.log_scroll = max(0, min(self.log_scroll, self.log_max_scroll))
        end = len(lines) - self.log_scroll
        shown = lines[max(0, end - visible):end]

        y = area.y + 4
        for kind, text in shown:
            s.blit(self.fs.render(text, True, colours.get(kind, DIM)), (x + 5, y))
            y += 14

        if self.log_scroll > 0:
            tag = self.fs.render(f"scrolled back {self.log_scroll} line(s) — "
                                 "END to follow", True, SUN)
            s.blit(tag, (area.right - tag.get_width() - 6, area.bottom - 14))

    def draw_progress(self, x, w):
        p = self.progress
        if not p:
            return
        s = self.screen
        by = H - 126
        frac = p["update"] / max(p["total"], 1)
        pygame.draw.rect(s, PANEL, (x, by, w, 6), border_radius=3)
        pygame.draw.rect(s, SUN, (x, by, int(w * frac), 6), border_radius=3)
        s.blit(self.fs.render(f"{p['update']}/{p['total']}  ret {p['return']:.1f}",
                              True, SUN), (x, by - 14))

    # -- loop --------------------------------------------------------------------

    def edit_key(self, e):
        if e.key == pygame.K_RETURN:
            self.commit_edit()
        elif e.key == pygame.K_ESCAPE:
            self.editing, self.edit_buf = None, ""
        elif e.key == pygame.K_BACKSPACE:
            self.edit_buf = self.edit_buf[:-1]
        elif e.unicode and e.unicode.isprintable():
            self.edit_buf += e.unicode

    def click(self, pos):
        if any(b.hit(pos) for b in self.buttons):
            return
        if self.obs_add_btn.hit(pos) or self.ent_add_btn.hit(pos):
            return
        for row in self.entity_rows:
            if row["del"].collidepoint(pos):
                self.remove_entity(row["id"])
                return
            if row["role"].collidepoint(pos):
                self.cycle_entity_role(row["id"])
                return
        if self.cmd.rect.collidepoint(pos):
            self.cmd.focused = True
            return
        self.cmd.focused = False
        for row in self.roster_rows:
            code = row["code"]
            if row["kind"].collidepoint(pos):
                self.commit_edit()
                self.toggle_kind(code)
                return
            if row["del"].collidepoint(pos):
                self.commit_edit()
                self.remove_robot(code)
                return
            if row["code_field"].collidepoint(pos):
                self.commit_edit()
                self.start_edit(code, "code")
                return
            if row["name_field"].collidepoint(pos):
                self.commit_edit()
                self.start_edit(code, "name")
                return
        for row in self.obstacle_rows:
            i = row["i"]
            if row["shape"].collidepoint(pos):
                self.commit_edit(); self.cycle_obstacle_shape(i); return
            if row["minus"].collidepoint(pos):
                self.commit_edit(); self.resize_obstacle(i, -5); return
            if row["plus"].collidepoint(pos):
                self.commit_edit(); self.resize_obstacle(i, +5); return
            if row["del"].collidepoint(pos):
                self.commit_edit(); self.remove_obstacle(i); return

        # dragging an obstacle around the arena
        if self.arena_rect().collidepoint(pos):
            hit = self.obstacle_at(self.to_cm(pos))
            if hit is not None:
                self.commit_edit()
                self.dragging_obstacle = hit
                return

        if self.editing:
            self.commit_edit()
        for btn, name in self.formation_rows:
            if btn.collidepoint(pos):
                self.submit(f"recall_formation {name}")
                return
        for r, ck in self.ck_rows:
            if r.collidepoint(pos):
                self.load(ck)
                return

    def run(self):
        running = True
        while running:
            dt = 1.0 / 30.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    self.click(e.pos)
                elif e.type == pygame.MOUSEMOTION and self.dragging_obstacle is not None:
                    self.move_obstacle(self.dragging_obstacle, self.to_cm(e.pos))
                elif e.type == pygame.MOUSEBUTTONUP and self.dragging_obstacle is not None:
                    i, self.dragging_obstacle = self.dragging_obstacle, None
                    self.save_workspace(f"obstacle {i + 1} moved")
                elif e.type == pygame.KEYDOWN:
                    if self.editing:
                        self.edit_key(e)
                        continue
                    if self.cmd.focused:
                        line = self.cmd.key(e)
                        if line:
                            self.submit(line)
                        continue
                    if e.key == pygame.K_TAB:
                        self.cmd.focused = True
                    elif e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pygame.K_b:
                        self.set_mode("boids")
                    elif e.key == pygame.K_p:
                        self.set_mode("policy")
                    elif e.key == pygame.K_i:
                        self.set_mode("idle")
                    elif e.key == pygame.K_n:
                        self.set_mode("navigate")
                    elif e.key == pygame.K_v:
                        self.cycle_camera_view()
                    elif e.key == pygame.K_F11:
                        self.toggle_fullscreen()
                    elif e.key == pygame.K_ESCAPE and self.fullscreen:
                        self.toggle_fullscreen()
                    elif e.key == pygame.K_PAGEUP:
                        self.scroll_log(10)
                    elif e.key == pygame.K_PAGEDOWN:
                        self.scroll_log(-10)
                    elif e.key == pygame.K_END:
                        self.log_scroll = 0
                    elif self.manual_key_down(e.key):
                        pass          # WASD is being driven, not interpreted
                    elif e.key == pygame.K_a:
                        self.set_ask(not self.ask)
                elif e.type == pygame.VIDEORESIZE:
                    if not self.fullscreen:
                        self.apply_size(e.w, e.h)
                elif e.type == pygame.MOUSEWHEEL:
                    if self.log_rect.collidepoint(pygame.mouse.get_pos()):
                        self.scroll_log(e.y * 3)
                elif e.type == pygame.KEYUP:
                    self.keys_held.discard(e.key)

            self.drain_events()

            if not self.paused:
                self.step_world(dt)

            self.screen.fill(INK)
            self.draw_arena()
            self.draw_dock()
            pygame.display.flip()
            self.clock.tick(30)

        if self.session is not None and self.session.busy:
            self.session.cancel()
            self.session.join(timeout=2.0)
        self.fleet.close()
        if self.tracker is not None:
            self.tracker.stop()
        pygame.quit()


def _short(v, limit=34):
    """Tool arguments in the log: readable, never a wall of coordinates."""
    if isinstance(v, (list, tuple)):
        if len(v) > 3:
            return f"[{len(v)} points]"
        return "[" + ", ".join(_short(x, 12) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}:{_short(x, 10)}" for k, x in list(v.items())[:3]) + "}"
    if isinstance(v, float):
        return f"{v:g}"
    text = str(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default=None,
                   help="camera index, video path, or 'synthetic'")
    p.add_argument("--controller", default="navigate",
                   choices=["navigate", "boids", "policy", "idle"])
    p.add_argument("--model", default="qwen3.5:9b",
                   help="preset from llm/models.yaml")
    p.add_argument("--no-ask", action="store_true",
                   help="start in literal tool mode even if a model is up")
    p.add_argument("--speed", type=float, default=None,
                   help=f"cm/s at full controller output (default {CRUISE_SPEED:.0f})")
    a = p.parse_args()
    App(camera=a.camera, controller=a.controller, model=a.model, speed=a.speed,
        ask=False if a.no_ask else None).run()


if __name__ == "__main__":
    main()
