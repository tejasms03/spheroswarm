"""Named, weighted velocity layers.

A robot used to have one target. It now has a small stack of contributions —
`seek` toward a target, `orbit` from a flow field, `follow` after something —
blended by weight into one desired velocity.

Three rules carry the design:

**Layers are named.** Anonymous blending is additive only: once three things
are mixed, "stop doing the second one" has no answer. Names make every addition
reversible, which is what lets a ten-turn conversation stay coherent.

**Avoidance is not in here.** Neighbour and obstacle repulsion is applied by the
controller *after* the blend, at a fixed weight nothing can reach. A model that
can weight collision avoidance down will eventually do it, and then two robots
meet.

**Four layers per robot, maximum.** Blended fields cancel — seek pulling one
way, flow another — and a robot vibrating in place looks broken to everyone
watching.

Contributions are in the controller's normalised units, the same [-1, 1] space
`Navigate.act` returns, so a weight of 1.0 on a single seek layer reproduces the
old behaviour exactly.
"""

import time

import numpy as np

MAX_LAYERS = 4
DEFAULT_DURATION = 60.0        # s, what a continuous layer gets if unasked
MAX_DURATION = 300.0           # s, the hard ceiling
STALL_SPEED = 3.0              # cm/s, below this a robot is not really moving
STALL_SECONDS = 2.0            # s of that before we call it stalled

KINDS = frozenset({"seek", "path", "flow", "follow"})


def now():
    return time.monotonic()


class Layer:
    """One named contribution. `params` is kind-specific and never mutated."""

    def __init__(self, name, kind, params=None, weight=1.0, duration=None,
                 started=None):
        self.name = str(name)
        self.kind = kind
        self.params = dict(params or {})
        self.weight = float(weight)
        self.duration = None if duration is None else float(duration)
        self.started = now() if started is None else started
        self.state = {}                 # per-layer scratch (path leg, etc.)

    @property
    def age(self):
        return now() - self.started

    @property
    def remaining(self):
        if self.duration is None:
            return None
        return max(0.0, self.duration - self.age)

    @property
    def expired(self):
        return self.duration is not None and self.age >= self.duration

    def describe(self):
        out = {"name": self.name, "kind": self.kind,
               "weight": round(self.weight, 2)}
        if self.duration is not None:
            out["remaining"] = round(self.remaining, 1)
        return out


def clamp_duration(duration, default=DEFAULT_DURATION, maximum=MAX_DURATION):
    """Returns (duration, clamped). A continuous layer always gets a bound."""
    if duration is None:
        return default, False
    try:
        d = float(duration)
    except (TypeError, ValueError):
        return default, True
    if not np.isfinite(d) or d <= 0:
        return default, True
    if d > maximum:
        return maximum, True
    return d, False


class LayerStack:
    """Every robot's layers, keyed by code."""

    def __init__(self, max_layers=MAX_LAYERS):
        self.max_layers = max_layers
        self._layers = {}               # code -> [Layer]
        self._slow_since = {}           # code -> timestamp or None
        self._stalled = set()
        # Why each stalled robot is stalled: (asked_for, achieved) in cm/s.
        # "stalled" on its own sends you hunting; "asked for 31, achieving 0"
        # says immediately that something is cancelling it rather than that it
        # has nothing to do.
        self._stall_reason = {}

    # -- membership ---------------------------------------------------------

    def layers(self, code):
        return self._layers.get(code, [])

    def get(self, code, name):
        return next((l for l in self.layers(code) if l.name == name), None)

    def codes(self):
        return [c for c, ls in self._layers.items() if ls]

    def push_layer(self, code, name, kind, params=None, weight=1.0,
                   duration=None):
        """Add or replace a named layer. Returns (layer, errors)."""
        if kind not in KINDS:
            return None, [f"unknown layer kind {kind!r}; expected one of "
                          f"{sorted(KINDS)}"]

        existing = self.get(code, name)
        if existing is None and len(self.layers(code)) >= self.max_layers:
            return None, [f"{code} already has {self.max_layers} layers "
                          f"({', '.join(l.name for l in self.layers(code))}); "
                          "remove one before adding another"]

        layer = Layer(name, kind, params, weight, duration)
        stack = self._layers.setdefault(code, [])
        if existing is not None:
            # Replace in place: a layer whose name already exists is the same
            # layer being redefined, and reordering it would silently change
            # nothing visible but everything about the blend order.
            stack[stack.index(existing)] = layer
        else:
            stack.append(layer)
        return layer, []

    def pop_layer(self, code, name):
        stack = self._layers.get(code, [])
        found = next((l for l in stack if l.name == name), None)
        if found is None:
            return [f"{code} has no layer called {name!r}"]
        stack.remove(found)
        return []

    def set_weight(self, code, name, weight):
        layer = self.get(code, name)
        if layer is None:
            return [f"{code} has no layer called {name!r}"]
        try:
            w = float(weight)
        except (TypeError, ValueError):
            return [f"weight must be a number, got {weight!r}"]
        if not np.isfinite(w):
            return ["weight must be finite"]
        layer.weight = w
        return []

    def clear_layers(self, code=None):
        if code is None:
            self._layers.clear()
            self._slow_since.clear()
            self._stalled.clear()
            self._stall_reason.clear()
            return
        self._layers.pop(code, None)
        self._clear(code)

    def active_layers(self, code):
        return [l.describe() for l in self.layers(code)]

    def has_continuous(self, code=None):
        """True if anything is running that will never settle on its own."""
        codes = [code] if code else list(self._layers)
        for c in codes:
            for l in self.layers(c):
                if l.kind in ("flow", "follow"):
                    return True
                if l.kind == "path" and l.params.get("mode") in ("loop", "pingpong"):
                    return True
        return False

    # -- time ---------------------------------------------------------------

    def expire(self):
        """Drop finished layers. Returns [(code, name)] of what was removed."""
        dropped = []
        for code, stack in list(self._layers.items()):
            keep = []
            for l in stack:
                if l.expired:
                    dropped.append((code, l.name))
                else:
                    keep.append(l)
            if keep:
                self._layers[code] = keep
            else:
                self._layers.pop(code, None)
        return dropped

    # -- stall detection -----------------------------------------------------

    def note_speed(self, code, speed_cms, wanted_cms=None):
        """Feed the commanded speed each tick so `stalled` means something.

        `wanted_cms` is what the layers asked for before avoidance and
        clamping. Without it, "stalled" catches every robot whose layer is
        simply *satisfied* — a follow holding the correct radius on a
        stationary target, a `once` path sitting on its final waypoint — which
        is not a stall, it is the job being done. A stall is asking to move and
        going nowhere: layers cancelling each other, or a wall in the way.
        """
        if not self.layers(code):
            self._slow_since.pop(code, None)
            self._stalled.discard(code)
            return
        if wanted_cms is not None and wanted_cms < STALL_SPEED:
            self._clear(code)
            return
        if speed_cms >= STALL_SPEED:
            self._clear(code)
            return
        started = self._slow_since.get(code)
        if started is None:
            self._slow_since[code] = now()
        elif now() - started >= STALL_SECONDS:
            self._stalled.add(code)
            self._stall_reason[code] = (round(float(wanted_cms or 0.0), 1),
                                        round(float(speed_cms), 1))

    def _clear(self, code):
        self._slow_since.pop(code, None)
        self._stalled.discard(code)
        self._stall_reason.pop(code, None)

    def stalled(self, code):
        return code in self._stalled

    def stall_reason(self, code):
        """(asked_for_cms, achieved_cms) for a stalled robot, else None."""
        return self._stall_reason.get(code)

    def stalled_codes(self):
        return sorted(self._stalled)
