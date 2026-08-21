"""What every tool is handed.

`SwarmContext` is the seam that makes the tools backend-agnostic: it holds a
fleet, a workspace and a controller, and knows nothing about whether any given
robot is a process or a ball. Swapping an all-sim fleet for a mixed one changes
nothing above this line.
"""

import threading
import time

import numpy as np

from fleet.manager import Fleet, FleetEnv
from swarm.layers import LayerStack
from swarm.navigate import Navigate
from workspace.space import Workspace

from .formations import FormationLibrary


class SwarmContext:
    def __init__(self, fleet=None, workspace=None, controller=None, library=None,
                 max_speed=None):
        self.ws = workspace or (fleet.ws if fleet is not None else Workspace.load())
        self.fleet = fleet if fleet is not None else Fleet(workspace=self.ws)
        # One stack for the whole context: tools push named layers onto it and
        # the controller blends them. Handed to Navigate so nothing has to look
        # it up through the context at tick rate.
        self.stack = LayerStack()
        self.controller = controller or Navigate(stack=self.stack)
        if getattr(self.controller, "stack", None) is None:
            self.controller.stack = self.stack
        self.library = library or FormationLibrary.load()
        self.env = FleetEnv(self.fleet)
        self.max_speed = max_speed
        # Keep the controller's idea of "full speed" in step with the fleet's,
        # so `stalled` measures the robots that are actually being driven.
        if max_speed and hasattr(self.controller, "speed_scale"):
            self.controller.speed_scale = float(max_speed)
        # True when a human is watching the arena. An all-sim fleet otherwise
        # advances as fast as the CPU allows, which is right for tests and evals
        # and wrong in front of a person: the robots cross the arena inside a
        # single tool call and the motion is never drawn.
        self.realtime = False
        # Learned precedents, injected into the prompt rather than fetched.
        # None means the feature is simply off, which is what tests want.
        self.memory = None
        self.log = []
        # Held for the duration of ONE tick, never longer. The UI shares it, so
        # that whoever is advancing the world does so alone — but a tool that
        # ticks for twenty seconds, like wait_until_settled, releases between
        # every tick. Holding it across a whole tool call instead is what made
        # a long wait freeze the window and lock out every other robot.
        self.lock = threading.RLock()

    # -- who is actually placeable ---------------------------------------

    def active_codes(self):
        """Enabled and connected. Everything a tool places is drawn from this."""
        return [c for c, h in self.fleet.handles.items() if h.connected]

    def active_positions(self):
        return np.array([self.fleet.handles[c].pos for c in self.active_codes()],
                        dtype=float).reshape(-1, 2)

    # -- driving ------------------------------------------------------------

    def apply_targets(self, mapping):
        """mapping: {code: (x, y)}. Sets the controller going; does not block."""
        self.env.rebuild_if_needed()
        self.env.targets = {c: np.asarray(p, dtype=float) for c, p in mapping.items()}
        for code, p in self.env.targets.items():
            h = self.fleet.handles.get(code)
            if h is not None:
                h.target = np.asarray(p, dtype=float)
        self.controller.targets = None       # per-robot targets come from the env

    def tick(self, dt, controller=None, override=None):
        """One control step: read the fleet, think, write velocities back.

        `controller` overrides the context's own for this step, which is what
        the UI needs — it swaps between navigate, boids, policy and idle. It is
        an argument rather than a second copy of this method in the app because
        a hand-rolled tick up there is exactly how moving entities and flow
        time stopped advancing on the render thread while still advancing
        inside `wait_until_settled`.
        """
        c = self.controller if controller is None else controller
        with self.lock:
            self.env.rebuild_if_needed()
            self.env.sync()
            self.ws.step(dt)                  # moving entities advance too
            if hasattr(c, "t"):
                c.t += dt
            if len(self.env.codes) == 0:
                return
            actions = c.act(self.env)
            self.env.apply(actions, max_speed=self.max_speed)
            # `override` gets the last word on velocities, after the controller
            # and before integration. Anything written after fleet.step would
            # simply be overwritten by the next tick's apply and never move a
            # robot at all — which is what a hand on WASD feels like.
            if override is not None:
                override()
            self.fleet.step(dt)

    def run_for(self, seconds, dt=0.1, sleep=False):
        """Advance the world. Sim robots need stepping; real ones need pacing."""
        steps = max(1, int(seconds / dt))
        for _ in range(steps):
            self.tick(dt)
            if sleep:
                time.sleep(dt)

    def stop_all(self):
        """The one call that reliably stops everything.

        Clearing the stack is the load-bearing part: without it a flow or a
        looping path survives the stop and starts driving again on the next
        tick, which is the worst possible response to someone pressing STOP.
        """
        self.stack.clear_layers()
        self.fleet.stop()
        self.env.targets = {}
        for h in self.fleet.handles.values():
            h.target = None

    # -- reporting -----------------------------------------------------------

    def state_summary(self):
        """The short status every tool result carries."""
        codes = self.active_codes()
        s = self.fleet.status()
        return {
            "robots": len(self.fleet),
            "active": len(codes),
            "codes": codes,
            "arena_cm": [round(self.ws.width, 1), round(self.ws.height, 1)],
            "obstacles": len(self.ws.obstacles),
            "formations": self.library.names(),
            "tracker_fps": s["tracker_fps"],
            "mean_rtt_ms": s["mean_rtt_ms"],
        }
