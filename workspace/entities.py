"""Things in the workspace that may move.

An obstacle and a followable target are the same object seen from two angles: a
shape, somewhere, possibly moving. Modelling them once with a `role` avoids the
alternative — two parallel systems that drift apart the first time something
needs to be both, like a robot you must not hit and must also follow.

Roles
    obstacle   avoid it
    target     follow it; robots pass straight through
    both       avoid it *and* follow it

Motion
    static     never moves (an old-schema obstacle loads as exactly this)
    path       waypoints walked at a speed, once / loop / pingpong
    flow       a sandboxed velocity field, compiled once at load

Nothing here raises: a malformed entity is reported and skipped, because a
workspace file with one bad entry should still open.
"""

import math

import numpy as np

VALID_ROLES = frozenset({"obstacle", "target", "both"})
VALID_KINDS = frozenset({"static", "path", "flow"})
VALID_MODES = frozenset({"once", "loop", "pingpong"})

MAX_SPEED = 200.0            # cm/s, an entity faster than this is a typo


def _point(p):
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def validate_entity(e, index=0):
    """Readable errors for one entity dict. Empty list means usable."""
    errors = []
    label = f"entities[{index}]"
    if not isinstance(e, dict):
        return [f"{label} must be an object"]

    if not e.get("id"):
        errors.append(f"{label} needs an id")
    label = f"entity {e.get('id', index)!r}"

    role = e.get("role", "obstacle")
    if role not in VALID_ROLES:
        errors.append(f"{label}: role must be one of {sorted(VALID_ROLES)}, got {role!r}")

    shape = e.get("shape")
    if not isinstance(shape, dict) or "type" not in shape:
        errors.append(f"{label} needs a shape with a type")
    else:
        t = shape.get("type")
        if t == "circle":
            if _point(shape.get("center")) is None:
                errors.append(f"{label}: circle needs a finite center")
            r = shape.get("radius")
            if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
                errors.append(f"{label}: circle needs a positive radius")
        elif t == "poly":
            pts = shape.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                errors.append(f"{label}: poly needs at least 3 points")
            elif any(_point(p) is None for p in pts):
                errors.append(f"{label}: poly has a non-finite point")
        else:
            errors.append(f"{label}: shape type must be 'circle' or 'poly', got {t!r}")

    motion = e.get("motion")
    if motion is not None:
        if not isinstance(motion, dict):
            errors.append(f"{label}: motion must be an object")
        else:
            kind = motion.get("kind", "static")
            if kind not in VALID_KINDS:
                errors.append(f"{label}: motion.kind must be one of "
                              f"{sorted(VALID_KINDS)}, got {kind!r}")
            speed = motion.get("speed", 0.0)
            if not isinstance(speed, (int, float)) or not math.isfinite(speed):
                errors.append(f"{label}: motion.speed must be a finite number")
            elif not 0 <= speed <= MAX_SPEED:
                errors.append(f"{label}: motion.speed {speed} is outside 0..{MAX_SPEED:.0f} cm/s")

            if kind == "path":
                wps = motion.get("waypoints")
                if not isinstance(wps, list) or len(wps) < 2:
                    errors.append(f"{label}: a path needs at least 2 waypoints")
                elif any(_point(p) is None for p in wps):
                    errors.append(f"{label}: a waypoint is not a finite point")
                mode = motion.get("mode", "loop")
                if mode not in VALID_MODES:
                    errors.append(f"{label}: motion.mode must be one of "
                                  f"{sorted(VALID_MODES)}, got {mode!r}")
            elif kind == "flow":
                if not isinstance(motion.get("expr"), str) or not motion["expr"].strip():
                    errors.append(f"{label}: a flow needs an 'expr' string")
    return errors


class Entity:
    """One thing in the workspace. Its shape is stored centred on `pos`."""

    def __init__(self, id, shape, role="obstacle", motion=None):
        self.id = id
        self.role = role
        self.motion = dict(motion or {"kind": "static"})
        self.motion.setdefault("kind", "static")

        self._shape_type = shape.get("type")
        if self._shape_type == "circle":
            self.pos = np.asarray(shape["center"], dtype=float)
            self.radius = float(shape["radius"])
            self._offsets = None
        else:
            pts = np.asarray(shape["points"], dtype=float)
            self.pos = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            self._offsets = pts - self.pos
            self.radius = float(np.linalg.norm(self._offsets, axis=1).max())

        self.home = self.pos.copy()
        self.vel = np.zeros(2)
        self.t = 0.0
        self._leg = 0
        self._dir = 1
        self._flow = None
        self.error = None

    # -- shape at the current position -------------------------------------

    @property
    def shape_type(self):
        return self._shape_type

    def shape(self):
        """The obstacle dict the rest of the codebase already understands."""
        if self._shape_type == "circle":
            return {"type": "circle", "center": [float(self.pos[0]), float(self.pos[1])],
                    "radius": self.radius}
        pts = self._offsets + self.pos
        return {"type": "poly", "points": [[float(p[0]), float(p[1])] for p in pts]}

    def to_dict(self):
        out = {"id": self.id, "shape": self.shape(), "role": self.role}
        if self.motion.get("kind", "static") != "static":
            out["motion"] = self.motion
        return out

    # -- roles ---------------------------------------------------------------

    @property
    def blocks(self):
        return self.role in ("obstacle", "both")

    @property
    def followable(self):
        return self.role in ("target", "both")

    # -- motion ---------------------------------------------------------------

    def compile_flow(self, compiler):
        """`compiler(expr)` -> callable(x, y, t) -> (vx, vy), or an error string."""
        if self.motion.get("kind") != "flow":
            return None
        fn, err = compiler(self.motion.get("expr", ""))
        self._flow = fn
        self.error = err
        return err

    def step(self, dt, bounds=None):
        """Advance one tick. Reflects off `bounds` rather than escaping it."""
        kind = self.motion.get("kind", "static")
        self.t += dt
        if kind == "static" or self.error:
            self.vel = np.zeros(2)
            return self.pos

        if kind == "path":
            self._step_path(dt)
        elif kind == "flow":
            self._step_flow(dt)

        if bounds is not None:
            self._reflect(bounds)
        return self.pos

    def _step_path(self, dt):
        wps = [np.asarray(w, dtype=float) for w in self.motion.get("waypoints", [])]
        if len(wps) < 2:
            self.vel = np.zeros(2)
            return
        speed = float(self.motion.get("speed", 0.0))
        mode = self.motion.get("mode", "loop")

        target = wps[self._leg]
        to = target - self.pos
        d = float(np.linalg.norm(to))
        if d < max(speed * dt, 1e-6):
            self.pos = target.copy()
            self._advance_leg(len(wps), mode)
            target = wps[self._leg]
            to = target - self.pos
            d = float(np.linalg.norm(to))
        if d > 1e-9:
            self.vel = to / d * speed
            self.pos = self.pos + self.vel * dt
        else:
            self.vel = np.zeros(2)

    def _advance_leg(self, n, mode):
        nxt = self._leg + self._dir
        if 0 <= nxt < n:
            self._leg = nxt
            return
        if mode == "loop":
            self._leg = 0 if self._dir > 0 else n - 1
        elif mode == "pingpong":
            self._dir *= -1
            self._leg = max(0, min(n - 1, self._leg + self._dir))
        else:                                   # once: stop at the end
            self._leg = max(0, min(n - 1, self._leg))
            self.motion = {**self.motion, "kind": "static"}

    def _step_flow(self, dt):
        if self._flow is None:
            self.vel = np.zeros(2)
            return
        try:
            vx, vy = self._flow(float(self.pos[0]), float(self.pos[1]), self.t)
        except Exception as e:
            self.error = f"flow failed: {type(e).__name__}: {e}"
            self.vel = np.zeros(2)
            return
        v = np.array([vx, vy], dtype=float)
        if not np.isfinite(v).all():
            self.error = "flow produced a non-finite velocity"
            self.vel = np.zeros(2)
            return
        speed = float(self.motion.get("speed", 0.0))
        mag = float(np.linalg.norm(v))
        if speed > 0 and mag > 1e-9:
            v = v / mag * speed
        self.vel = v
        self.pos = self.pos + v * dt

    def _reflect(self, bounds):
        """Bounce off the arena walls, keeping the whole body inside."""
        xmin, xmax, ymin, ymax = bounds
        r = self.radius
        for ax, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax))):
            if self.pos[ax] - r < lo:
                self.pos[ax] = lo + r
                self.vel[ax] = abs(self.vel[ax])
                self._bounce_path()
            elif self.pos[ax] + r > hi:
                self.pos[ax] = hi - r
                self.vel[ax] = -abs(self.vel[ax])
                self._bounce_path()

    def _bounce_path(self):
        """A path that hits a wall turns around rather than grinding into it."""
        if self.motion.get("kind") == "path" and self.motion.get("mode") == "pingpong":
            self._dir *= -1


class EntitySet:
    """All entities in a workspace, advanced together."""

    def __init__(self, entities=None):
        self.entities = list(entities or [])
        self.errors = []

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_data(cls, data, compiler=None):
        """`data` is the `entities` list; unusable entries are reported, not fatal."""
        out, errors = [], []
        seen = set()
        for i, raw in enumerate(data or []):
            errs = validate_entity(raw, i)
            if errs:
                errors.extend(errs)
                continue
            if raw["id"] in seen:
                errors.append(f"duplicate entity id {raw['id']!r}")
                continue
            seen.add(raw["id"])
            e = Entity(raw["id"], raw["shape"], raw.get("role", "obstacle"),
                       raw.get("motion"))
            if compiler is not None and e.motion.get("kind") == "flow":
                err = e.compile_flow(compiler)
                if err:
                    errors.append(f"entity {e.id!r}: {err}")
            out.append(e)
        s = cls(out)
        s.errors = errors
        return s

    @classmethod
    def from_obstacles(cls, obstacles):
        """Old schema: a plain list of shapes, all static obstacles."""
        out = []
        for i, shape in enumerate(obstacles or []):
            out.append(Entity(f"obstacle{i + 1}", shape, "obstacle", None))
        return cls(out)

    def to_data(self):
        return [e.to_dict() for e in self.entities]

    # -- queries ---------------------------------------------------------------

    def __len__(self):
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities)

    def by_id(self, entity_id):
        return next((e for e in self.entities if e.id == entity_id), None)

    def by_role(self, role):
        if role == "obstacle":
            return [e for e in self.entities if e.blocks]
        if role == "target":
            return [e for e in self.entities if e.followable]
        return [e for e in self.entities if e.role == role]

    def blocking_shapes(self):
        """Shapes of everything that should be avoided, at their current place."""
        return [e.shape() for e in self.entities if e.blocks]

    def positions(self):
        return {e.id: e.pos.copy() for e in self.entities}

    def velocities(self):
        return {e.id: e.vel.copy() for e in self.entities}

    def moving(self):
        return [e for e in self.entities
                if e.motion.get("kind", "static") != "static"]

    # -- mutation ---------------------------------------------------------------

    def add(self, entity):
        if self.by_id(entity.id) is not None:
            return [f"an entity called {entity.id!r} already exists"]
        self.entities.append(entity)
        return []

    def remove(self, entity_id):
        before = len(self.entities)
        self.entities = [e for e in self.entities if e.id != entity_id]
        return [] if len(self.entities) < before else [f"no entity {entity_id!r}"]

    # -- time ---------------------------------------------------------------------

    def step(self, dt, bounds=None):
        for e in self.entities:
            e.step(dt, bounds)
        return self.positions()
