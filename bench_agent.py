"""A language model driving the taillight bench, through the bench's own jobs.

Every motion tool calls exactly the functions a click calls — `p2p_click`,
`line_click`, `finish_path`, `orbit_click`, `finish_patrol` — with centimetres
turned into the view position a click would have had. So the model drives the
same tested code a person does, and the job supervisor's retries cover it too.

Threading: the model runs on a worker thread so the window keeps drawing, but
it never touches the app. Each tool call is queued and run by `pump()` on the
main loop, and the worker waits for the answer. Only `wait` sleeps on the
worker, reading status through the same queue.

What the model does NOT get, on purpose: corner picking, calibration, spin
power, lookahead, connecting or dropping balls. It may say where a ball goes;
it may not reconfigure the rig.
"""

import json
import math
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from swarm import ball_calib

PRESET = "sonnet5"
MAX_CALLS = 10
WAIT_MAX_S = 90.0
EDGE_CM = 5.0
ENV_FILE = Path.home() / ".spheroswarm.env"

_POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2,
          "maxItems": 2}

TOOLS = [
    {"name": "get_state",
     "description": "Every connected ball: position (cm), which way it faces, "
                    "whether the camera sees it, whether it is calibrated, "
                    "which one is selected, what job is running, retries, and "
                    "the arena size.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "goto",
     "description": "Drive the ball to a point: it turns on the spot to face "
                    "it, then drives straight there. Replaces any running job.",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "ball": {"type": "string", "description": "which ball, e.g. "
                  "SK-914A; omit for the one already selected"},},
         "required": ["x", "y"]}},
    {"name": "follow_line",
     "description": "Drive to the line's start, face its end, then follow the "
                    "straight line to the end. Replaces any running job.",
     "parameters": {"type": "object", "properties": {
         "x1": {"type": "number"}, "y1": {"type": "number"},
         "x2": {"type": "number"}, "y2": {"type": "number"},
         "ball": {"type": "string", "description": "which ball, e.g. "
                  "SK-914A; omit for the one already selected"},},
         "required": ["x1", "y1", "x2", "y2"]}},
    {"name": "follow_path",
     "description": "Drive to the first point, then follow the route through "
                    "every point in order, once. smooth=true treats the points "
                    "as a hand-drawn curve (smoothed); false keeps sharp "
                    "corners. For shapes, give enough points (e.g. 24+ for a "
                    "curve). Replaces any running job.",
     "parameters": {"type": "object", "properties": {
         "points": {"type": "array", "items": _POINT, "minItems": 2},
         "smooth": {"type": "boolean"},
         "ball": {"type": "string", "description": "which ball, e.g. "
                  "SK-914A; omit for the one already selected"},},
         "required": ["points"]}},
    {"name": "orbit",
     "description": "Circle a centre point at a radius, until stopped. "
                    "direction 'cw' or 'ccw' as seen on screen (y down). "
                    "Radius at least 10cm; 25cm+ holds the circle best. "
                    "Replaces any running job.",
     "parameters": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "radius": {"type": "number"},
         "direction": {"type": "string", "enum": ["cw", "ccw"]},
         "ball": {"type": "string", "description": "which ball, e.g. "
                  "SK-914A; omit for the one already selected"},},
         "required": ["x", "y", "radius"]}},
    {"name": "patrol",
     "description": "Patrol points until stopped. style 'loop' goes round "
                    "1->2->...->1 (needs 3+ points); 'back_and_forth' drives to "
                    "the last point, turns round and comes back (2+ points). "
                    "Replaces any running job.",
     "parameters": {"type": "object", "properties": {
         "points": {"type": "array", "items": _POINT, "minItems": 2},
         "style": {"type": "string", "enum": ["loop", "back_and_forth"]},
         "ball": {"type": "string", "description": "which ball, e.g. "
                  "SK-914A; omit for the one already selected"},},
         "required": ["points"]}},
    {"name": "stop",
     "description": "Stop a job. Name a ball to stop that one; omit the ball "
                    "to stop every ball that is running.",
     "parameters": {"type": "object", "properties": {
         "ball": {"type": "string", "description": "which ball, e.g. "
                  "SK-914A; omit to stop them all"},}}},
    {"name": "wait",
     "description": "Wait for one ball's goto, follow_line or follow_path to "
                    "finish, up to timeout_s (default 30, max 90). Returns "
                    "status 'done', 'running' (still going when the timeout "
                    "ran out), or 'stopped'. Orbits and patrols never finish "
                    "— do not wait on them.",
     "parameters": {"type": "object", "properties": {
         "ball": {"type": "string", "description": "which ball to wait for, "
                  "e.g. SK-914A; omit for the one already selected"},
         "timeout_s": {"type": "number"}}}},
]


def load_env(path=ENV_FILE):
    """`export NAME=value` lines into the environment, without overwriting.
    The key never leaves this process and is never logged."""
    try:
        text = Path(path).read_text()
    except Exception:
        return
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line or line.startswith("#"):
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


class BenchAgent:
    def __init__(self, app, client=None, preset=PRESET):
        self.app = app
        self.client = client
        self.preset = preset
        self.busy = False
        self.log = []               # (kind, text), for the transcript
        self.typing = False
        self.text = ""
        self._calls = queue.Queue()

    # -- the model -----------------------------------------------------------

    def _client(self):
        if self.client is None:
            load_env()
            from llm.client import client_for
            self.client = client_for(self.preset)
        return self.client

    def ask(self, command):
        """Run a command on a worker thread."""
        if self.busy:
            self.app.say("agent: still working on the last command")
            return False
        self.busy = True
        self.log = [("you", command)]

        def work():
            try:
                self.run(command)
            except Exception as e:
                self.log.append(("error", f"{type(e).__name__}: {e}"))
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True, name="bench-agent").start()
        return True

    def run(self, command, call=None):
        """One command to the end. `call` runs a tool; by default it is queued
        to the main loop. Tests pass `self.dispatch` to run it directly."""
        call = call or self.call_on_main
        try:
            client = self._client()
        except Exception as e:
            self.log.append(("error", f"no model: {e}"))
            return self.log
        messages = [{"role": "system", "content": self.prompt(call)},
                    {"role": "user", "content": command}]
        for _ in range(MAX_CALLS):
            reply = client.chat(messages, TOOLS)
            if not reply.tool_calls:
                self.log.append(("say", reply.text or "(nothing)"))
                return self.log
            messages.append({"role": "assistant", "content": reply.text or "",
                             "tool_calls": [
                                 {"id": c.id, "type": "function",
                                  "function": {"name": c.name,
                                               "arguments": json.dumps(c.arguments)}}
                                 for c in reply.tool_calls]})
            for c in reply.tool_calls:
                result = call(c.name, c.arguments or {})
                self.log.append(("call", f"{c.name}({_short(c.arguments)}) -> "
                                         f"{_short(result)}"))
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(result)})
        self.log.append(("error", f"stopped after {MAX_CALLS} tool calls"))
        return self.log

    def prompt(self, call):
        st = call("get_state", {})
        return "\n".join([
            "You drive Sphero balls on a flat arena, seen by an overhead "
            "camera, using the tools. Each ball has its own job and they run "
            "at the same time.",
            "Coordinates are centimetres: x to the RIGHT, y DOWN the screen, "
            f"(0,0) the top-left corner, arena {st.get('arena_cm')}.",
            f"Keep points at least {EDGE_CM:.0f}cm inside the arena.",
            "",
            f"State now: {json.dumps(st)}",
            "",
            "Rules:",
            "- Use the tools; do not describe what you would do.",
            "- Every motion tool takes a `ball`. Give it whenever the user "
            "names a ball; leave it out to use the selected one.",
            "- Each ball has its own job and they run at the same time. "
            "Starting a job on one ball does NOT disturb another. To set "
            "several balls going together, call the motion tools one after "
            "another with a different `ball` each time and do not wait in "
            "between. A ball's own next job needs that ball finished first "
            "(wait on it), or stopped if it is an orbit or patrol.",
            "- NOTHING KEEPS THE BALLS APART. They will drive through each "
            "other if their paths cross. Choose jobs that stay clear of one "
            "another, and say so if the user asks for paths that cross.",
            "- orbit and patrol run until stop. goto, follow_line and "
            "follow_path finish: call wait if the user wants to know it got "
            "there, or before starting the next step of a sequence.",
            "- A job that fails is retried automatically from where the ball "
            "is; you do not need to restart it.",
            "- If a tool returns an error, read it and fix the call.",
            "- Compute shapes yourself as points (e.g. a square, a star, a "
            "figure-eight) and use follow_path or patrol.",
            "- Reply briefly once done.",
        ])

    # -- getting a tool call onto the main loop ---------------------------------

    def call_on_main(self, name, args, timeout=10.0):
        if name == "wait":
            return self._wait(args)
        box = {"done": threading.Event()}
        self._calls.put((name, args, box))
        if not box["done"].wait(timeout):
            return {"error": "the bench did not answer — is its window frozen?"}
        return box["result"]

    def pump(self):
        """Main loop: run queued tool calls."""
        while True:
            try:
                name, args, box = self._calls.get_nowait()
            except queue.Empty:
                return
            try:
                box["result"] = self.dispatch(name, args)
            except Exception as e:
                box["result"] = {"error": f"{type(e).__name__}: {e}"}
            box["done"].set()

    def _wait(self, args, poll=0.25):
        try:
            limit = float(args.get("timeout_s", 30.0))
        except (TypeError, ValueError):
            return {"error": "timeout_s must be a number"}
        limit = max(0.0, min(WAIT_MAX_S, limit))
        ball = args.get("ball")
        end = time.time() + limit
        while True:
            # Several balls run at once, so waiting means waiting on ONE of
            # them — named, or the selected one.
            st = self.call_on_main("status",
                                   {} if ball is None else {"ball": ball})
            if st.get("status") != "running" or time.time() >= end:
                return st
            time.sleep(poll)

    # -- the tools (main loop only) ------------------------------------------------

    def dispatch(self, name, args):
        fn = {"get_state": self.t_get_state, "goto": self.t_goto,
              "follow_line": self.t_follow_line,
              "follow_path": self.t_follow_path, "orbit": self.t_orbit,
              "patrol": self.t_patrol, "stop": self.t_stop,
              "status": self.t_status, "wait": self.t_status}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    def _ready(self, ball=None):
        """The ball a tool will act on. Naming one selects it first, the same
        as pressing Tab, because a job always belongs to the selected ball."""
        app = self.app
        if ball is not None:
            want = str(ball)
            if want not in app.lab.robots:
                return None, (f"{want} is not connected; connected: "
                              + (", ".join(sorted(app.lab.robots)) or "none"))
            app.driving = want
        name = app.drive_target()
        if name is None:
            return None, "no ball is connected"
        if name not in app.lab.tracks.by_name:
            return None, (f"{name} is not assigned on camera — a person has "
                          "to press i and click its front light")
        if app.lab.hom is None or not app.lab.hom.ready:
            return None, "the arena corners are not calibrated"
        if app._shot is None:
            return None, "the camera view is not up yet"
        if app.calib is not None:
            return None, "a calibration walk is running — wait for it"
        return name, None

    def _num(self, v, what):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what} must be a number, got {v!r}")
        if not math.isfinite(f):
            raise ValueError(f"{what} must be finite, got {v!r}")
        return f

    def _points(self, pts, what="points"):
        if not isinstance(pts, (list, tuple)):
            raise ValueError(f"{what} must be a list of [x, y]")
        out = []
        for i, p in enumerate(pts):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"{what}[{i}] must be [x, y], got {p!r}")
            out.append((self._num(p[0], f"{what}[{i}][0]"),
                        self._num(p[1], f"{what}[{i}][1]")))
        return out

    def _inside(self, pts):
        hom = self.app.lab.hom
        w, h = float(hom.width), float(hom.height)
        bad = [(round(x, 1), round(y, 1)) for x, y in pts
               if not (EDGE_CM <= x <= w - EDGE_CM
                       and EDGE_CM <= y <= h - EDGE_CM)]
        if bad:
            return (f"point(s) {bad[:4]} are outside the arena — keep x within "
                    f"{EDGE_CM:.0f}..{w - EDGE_CM:.0f} and y within "
                    f"{EDGE_CM:.0f}..{h - EDGE_CM:.0f}")
        return None

    def _view(self, x, y):
        """Centimetres to where a click on the view would be."""
        (ox, oy), k = self.app._shot
        px = self.app.lab.hom.to_px([[x, y]])[0]
        return (float(px[0]) * k + ox, float(px[1]) * k + oy)

    def _clear(self):
        """End whatever is running, exactly as the stop button does."""
        app = self.app
        app.orbit_run = None
        app.patrol_run = None
        for cancel, active in ((app.cancel_path, app.path_pick),
                               (app.cancel_orbit, app.orbit_pick),
                               (app.cancel_patrol, app.patrol_pick)):
            if active is not None:
                cancel("cancelled")
        app.p2p_pick = False
        app.line_pick = None
        app.stop_p2p("stopped")
        app.job = None

    def _refused(self, what):
        note = list(self.app.notes)[-1][0] if self.app.notes else "refused"
        return {"error": f"{what} did not start: {note}"}

    def _started(self, what, **extra):
        out = {"ok": True, "started": what}
        out.update(extra)
        return out

    def t_get_state(self):
        app = self.app
        hom = app.lab.hom
        name = app.drive_target()
        st = {"arena_cm": ([round(float(hom.width), 1),
                            round(float(hom.height), 1)]
                           if hom is not None and hom.ready else None),
              "selected": name,
              "connected": sorted(app.lab.robots),
              "balls": {n: self._ball_state(n) for n in sorted(app.lab.robots)}}
        if name is not None and name in app.lab.tracks.by_name:
            px = app.ball_px(name)
            if px is not None and hom is not None and hom.ready:
                p = np.asarray(hom.to_cm([list(px)]), float).ravel()[:2]
                st["position_cm"] = [round(float(p[0]), 1),
                                     round(float(p[1]), 1)]
            facing = app.arena_heading(name)
            st["facing_deg"] = None if facing is None else round(float(facing))
            st["seen"] = bool(app.ball_fresh(name))
            try:
                st["calibrated"] = ball_calib.load(name, hom.M)[0] is not None
            except Exception:
                st["calibrated"] = False
        elif name is not None:
            st["seen"] = False
            st["note"] = "not assigned on camera"
        st.update(self.t_status())
        return st

    def _ball_state(self, name):
        """One ball, as the model sees it. Read only."""
        app = self.app
        hom = app.lab.hom
        out = {"assigned": name in app.lab.tracks.by_name}
        if not out["assigned"]:
            out["note"] = "not assigned on camera"
            return out
        px = app.ball_px(name)
        if px is not None and hom is not None and hom.ready:
            p = np.asarray(hom.to_cm([list(px)]), float).ravel()[:2]
            out["position_cm"] = [round(float(p[0]), 1), round(float(p[1]), 1)]
        facing = app.arena_heading(name)
        out["facing_deg"] = None if facing is None else round(float(facing))
        out["seen"] = bool(app.ball_fresh(name))
        try:
            out["calibrated"] = ball_calib.load(name, hom.M)[0] is not None
        except Exception:
            out["calibrated"] = False
        # Its own job, because they run at the same time and a fleet-wide
        # "what is happening" has to name a ball for each answer.
        st = self.t_status(ball=name)
        out["status"] = st.get("status")
        if st.get("job"):
            out["job"] = st["job"]
        return out

    def t_status(self, ball=None, **_):
        """What one ball is doing. The selected one unless another is named.

        Read out of that ball's own job slot rather than the live attributes,
        because several balls run at once and the live ones belong to
        whichever is selected.
        """
        app = self.app
        name = str(ball) if ball is not None else app.drive_target()
        if name is None:
            return {"status": "idle", "retries": 0}
        s = app.ball_slot(name)
        if s["orbit_run"] is not None:
            r = s["orbit_run"]
            return {"status": "running", "job": "orbit", "ball": r["name"],
                    "detail": f"radius {r['radius']:.0f}cm {r['dir']}, "
                              f"{r['laps']} laps done — runs until stop",
                    "retries": s["retries"]}
        if s["patrol_run"] is not None:
            r = s["patrol_run"]
            return {"status": "running", "job": "patrol", "ball": r["name"],
                    "detail": f"{r['style']}, {r['rounds']} done — runs until "
                              "stop", "retries": s["retries"]}
        got = s["p2p"]
        if got is not None:
            job = ("follow_path" if got.get("path") is not None else
                   "follow_line" if got.get("kind") == "line" else "goto")
            return {"status": "running", "job": job, "ball": got.get("name"),
                    "stage": got.get("stage") or got.get("phase"),
                    "retries": s["retries"]}
        if s["job"] is not None:
            return {"status": "running", "job": "retrying",
                    "detail": (s["_last_stop"] or ("", None))[0][:100],
                    "retries": s["retries"]}
        last = (s["_last_stop"] or ("", None))[0]
        if not last:
            return {"status": "idle", "retries": s["retries"]}
        stopped = last in app.USER_STOPS
        return {"status": "stopped" if stopped else "done",
                "result": last[:160], "retries": s["retries"]}

    def t_stop(self, ball=None):
        """Stop one named ball, or every ball that is running."""
        app = self.app
        if ball is not None and str(ball) in app.lab.robots:
            app.driving = str(ball)
            self._clear()
            return {"ok": True, "stopped": [str(ball)]}
        busy = app.busy_balls() or ([app.drive_target()]
                                    if app.drive_target() else [])
        was = app.driving
        for name in busy:
            app.driving = name
            self._clear()
        if was is not None:
            app.driving = was
        return {"ok": True, "stopped": busy}

    def t_goto(self, x, y, ball=None):
        name, why = self._ready(ball)
        if why:
            return {"error": why}
        try:
            x, y = self._num(x, "x"), self._num(y, "y")
        except ValueError as e:
            return {"error": str(e)}
        bad = self._inside([(x, y)])
        if bad:
            return {"error": bad}
        self._clear()
        app = self.app
        app.start_p2p()
        if not app.p2p_pick:
            return self._refused("goto")
        app.p2p_click(self._view(x, y))
        if app.p2p is None:
            return self._refused("goto")
        return self._started("goto", ball=name, to=[x, y])

    def t_follow_line(self, x1, y1, x2, y2, ball=None):
        name, why = self._ready(ball)
        if why:
            return {"error": why}
        try:
            a = (self._num(x1, "x1"), self._num(y1, "y1"))
            b = (self._num(x2, "x2"), self._num(y2, "y2"))
        except ValueError as e:
            return {"error": str(e)}
        bad = self._inside([a, b])
        if bad:
            return {"error": bad}
        self._clear()
        app = self.app
        app.start_line()
        if app.line_pick is None:
            return self._refused("follow_line")
        app.line_click(self._view(*a))
        app.line_click(self._view(*b))
        if app.p2p is None:
            app.line_pick = None
            return self._refused("follow_line")
        return self._started("follow_line", ball=name,
                             length_cm=round(math.dist(a, b), 1))

    def t_follow_path(self, points, smooth=False, ball=None):
        name, why = self._ready(ball)
        if why:
            return {"error": why}
        try:
            pts = self._points(points)
        except ValueError as e:
            return {"error": str(e)}
        if len(pts) < 2:
            return {"error": "follow_path needs at least 2 points"}
        bad = self._inside(pts)
        if bad:
            return {"error": bad}
        self._clear()
        app = self.app
        kind = "free" if smooth else "poly"
        app.start_path(kind)
        if app.path_pick is None:
            return self._refused("follow_path")
        views = [self._view(*p) for p in pts]
        if kind == "poly":
            for v in views:
                app.path_press(v)
            app.finish_path()
        else:
            app.path_press(views[0])
            for v in views[1:]:
                app.path_drag(v)
            app.path_release()
        if app.p2p is None or app.p2p.get("path") is None:
            return self._refused("follow_path")
        return self._started("follow_path", ball=name,
                             length_cm=round(app.p2p["path"]["total"], 1),
                             smooth=bool(smooth))

    def t_orbit(self, x, y, radius, direction="ccw", ball=None):
        name, why = self._ready(ball)
        if why:
            return {"error": why}
        try:
            x, y = self._num(x, "x"), self._num(y, "y")
            r = self._num(radius, "radius")
        except ValueError as e:
            return {"error": str(e)}
        if direction not in ("cw", "ccw"):
            return {"error": "direction must be 'cw' or 'ccw'"}
        bad = self._inside([(x - r, y - r), (x + r, y + r)])
        if bad:
            return {"error": f"that circle leaves the arena ({bad})"}
        self._clear()
        app = self.app
        app.orbit_dir = direction
        app.start_orbit()
        if app.orbit_pick is None:
            return self._refused("orbit")
        app.orbit_click(self._view(x, y))
        app.orbit_click(self._view(x + r, y))
        if app.orbit_run is None:
            return self._refused("orbit")
        return self._started("orbit", ball=name, until="stop is called")

    def t_patrol(self, points, style="loop", ball=None):
        name, why = self._ready(ball)
        if why:
            return {"error": why}
        if style not in ("loop", "back_and_forth"):
            return {"error": "style must be 'loop' or 'back_and_forth'"}
        try:
            pts = self._points(points)
        except ValueError as e:
            return {"error": str(e)}
        need = 3 if style == "loop" else 2
        if len(pts) < need:
            return {"error": f"a {style} patrol needs at least {need} points"
                             + (" — use back_and_forth for 2" if need == 3
                                else "")}
        bad = self._inside(pts)
        if bad:
            return {"error": bad}
        self._clear()
        app = self.app
        app.patrol_style = "loop" if style == "loop" else "bounce"
        app.start_patrol()
        if app.patrol_pick is None:
            return self._refused("patrol")
        for p in pts:
            app.patrol_click(self._view(*p))
        app.finish_patrol()
        if app.patrol_run is None:
            if app.patrol_pick is not None:
                app.cancel_patrol("cancelled")
            return self._refused("patrol")
        return self._started("patrol", ball=name, style=style,
                             until="stop is called")

    # -- the command bar ---------------------------------------------------------------

    @staticmethod
    def clipboard():
        """Whatever is on the system clipboard, as one line.

        SDL's own clipboard goes through `pygame.scrap`, which needs the
        display initialised in a particular order and quietly returns nothing
        on macOS often enough not to rely on. `pbpaste` is the platform's own
        answer and is always there.
        """
        for argv in (["pbpaste"], ["xclip", "-selection", "clipboard", "-o"],
                     ["xsel", "--clipboard", "--output"]):
            try:
                got = subprocess.run(argv, capture_output=True, timeout=2)
            except (OSError, subprocess.SubprocessError):
                continue
            if got.returncode == 0:
                text = got.stdout.decode("utf-8", "replace")
                # A pasted command is one line: newlines would otherwise read
                # as Return and send it half typed.
                return " ".join(text.split())
        return ""

    def key(self, e, pygame):
        """A key while the bar is open. Returns True if it was consumed."""
        if e.key == pygame.K_ESCAPE:
            self.typing, self.text = False, ""
            return True
        # Paste. cmd-V on a Mac, ctrl-V everywhere else.
        if e.key == pygame.K_v and (e.mod & (pygame.KMOD_META
                                             | pygame.KMOD_CTRL)):
            self.text += self.clipboard()
            return True
        if e.key == pygame.K_u and (e.mod & pygame.KMOD_CTRL):
            self.text = ""            # the shell's own "clear the line"
            return True
        if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            command = self.text.strip()
            self.typing, self.text = False, ""
            if command:
                self.ask(command)
            return True
        if e.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True
        if e.unicode and e.unicode.isprintable():
            self.text += e.unicode
        return True


def _short(obj, limit=160):
    text = json.dumps(obj) if not isinstance(obj, str) else obj
    return text if len(text) <= limit else text[:limit - 1] + "…"
