"""The workspace: arena bounds and obstacles, in centimetres.

A polygon plus a handful of poly/circle exclusion zones. Everything a
controller or validator needs to reason about "is this point usable" lives
here. Pixels never appear past `workspace/make.py --from-camera`.
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "workspace.json"
DEFAULT_ORIGIN = "top-left, x right, y down, matching the camera frame"
DEFAULT_BOUNDS = [[0, 0], [200, 0], [200, 200], [0, 200]]

EPS = 1e-6
NUDGE = 0.5          # cm pushed past a boundary/obstacle edge to land cleanly inside/outside


def _to_point(p):
    """Best-effort (x, y) float tuple, or None if p isn't a finite 2D point."""
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def point_in_polygon(p, poly):
    """Even-odd ray casting. Boundary points may go either way; callers nudge."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or EPS)
            if x < x_int:
                inside = not inside
    return inside


def _dist_point_segment(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)), proj


def _nearest_boundary_point(p, poly):
    best_d, best_pt = math.inf, np.asarray(poly[0], float)
    n = len(poly)
    for i in range(n):
        d, proj = _dist_point_segment(p, poly[i], poly[(i + 1) % n])
        if d < best_d:
            best_d, best_pt = d, proj
    return best_pt, best_d


def _centroid(poly):
    pts = np.asarray(poly, float)
    return pts.mean(axis=0)


def _dump(data):
    """JSON with each [x, y] pair on one line — this file is meant to be hand-edited."""
    import re
    text = json.dumps(data, indent=2)
    return re.sub(r"\[\s+(-?[\d.]+),\s+(-?[\d.]+)\s+\]", r"[\1, \2]", text) + "\n"


def validate(data):
    """Validate a workspace dict as loaded from JSON. Returns a list of errors."""
    errors = []
    if not isinstance(data, dict):
        return ["workspace must be a JSON object"]

    bounds = data.get("bounds_cm")
    if not isinstance(bounds, list) or len(bounds) < 3:
        errors.append("bounds_cm must be a list of at least 3 [x, y] points")
    else:
        for i, p in enumerate(bounds):
            if _to_point(p) is None:
                errors.append(f"bounds_cm[{i}] is not a finite [x, y] point: {p!r}")

    ents = data.get("entities", [])
    if ents is not None and not isinstance(ents, list):
        errors.append("entities must be a list")

    obstacles = data.get("obstacles", [])
    if not isinstance(obstacles, list):
        errors.append("obstacles must be a list")
    else:
        for i, o in enumerate(obstacles):
            if not isinstance(o, dict) or "type" not in o:
                errors.append(f"obstacles[{i}] must be an object with a 'type'")
                continue
            t = o.get("type")
            if t == "poly":
                pts = o.get("points")
                if not isinstance(pts, list) or len(pts) < 3:
                    errors.append(f"obstacles[{i}] (poly) needs at least 3 points")
                else:
                    for j, p in enumerate(pts):
                        if _to_point(p) is None:
                            errors.append(f"obstacles[{i}].points[{j}] is not a finite point: {p!r}")
            elif t == "circle":
                c = _to_point(o.get("center"))
                if c is None:
                    errors.append(f"obstacles[{i}] (circle) needs a finite 'center'")
                r = o.get("radius")
                if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
                    errors.append(f"obstacles[{i}] (circle) needs a positive finite 'radius'")
            else:
                errors.append(f"obstacles[{i}] has unknown type {t!r} (want 'poly' or 'circle')")

    return errors


class Workspace:
    def __init__(self, bounds_cm=None, obstacles=None, origin=DEFAULT_ORIGIN,
                 path=DEFAULT_PATH, entities=None):
        self.bounds_cm = [tuple(map(float, p)) for p in (bounds_cm or DEFAULT_BOUNDS)]
        # Static obstacles stay a plain, directly mutable list: the UI editor
        # and everything written before entities existed edit it in place.
        self.obstacles = obstacles or []
        # Entities are the general case — a shape with a role and, optionally,
        # motion. An obstacle-role entity blocks exactly like a static one.
        from .entities import EntitySet
        self.entities = entities if entities is not None else EntitySet()
        self.origin = origin
        self.path = Path(path)
        self.errors = []

    # -- persistence -------------------------------------------------------

    def to_dict(self):
        out = {
            "bounds_cm": [list(p) for p in self.bounds_cm],
            "obstacles": self.obstacles,
            "origin": self.origin,
        }
        if len(self.entities):
            out["entities"] = self.entities.to_data()
        return out

    def step(self, dt):
        """Advance every moving entity. Static workspaces are unaffected."""
        if len(self.entities):
            self.entities.step(dt, self.bbox)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        path = Path(path)
        if not path.exists():
            ws = cls(path=path)
            ws.errors = [f"{path} does not exist; using a default {DEFAULT_BOUNDS[2][0]:.0f}cm square"]
            return ws
        try:
            raw = json.loads(path.read_text())
        except Exception as e:
            ws = cls(path=path)
            ws.errors = [f"could not parse {path}: {e}"]
            return ws

        errors = validate(raw)
        if errors:
            ws = cls(path=path)
            ws.errors = errors
            return ws

        from .entities import EntitySet
        from .flow import compile_flow

        entities = EntitySet.from_data(raw.get("entities"), compiler=compile_flow)
        ws = cls(bounds_cm=raw["bounds_cm"], obstacles=raw.get("obstacles", []),
                  origin=raw.get("origin", DEFAULT_ORIGIN), path=path,
                  entities=entities)
        ws.errors = list(entities.errors)
        return ws

    def save(self, path=None):
        errors = validate(self.to_dict())
        if errors:
            self.errors = errors
            return errors
        target = Path(path) if path else self.path
        try:
            target.write_text(_dump(self.to_dict()))
        except Exception as e:
            return [f"could not write {target}: {e}"]
        self.path = target
        self.errors = []
        return []

    # -- geometry ------------------------------------------------------------

    @property
    def bbox(self):
        pts = np.asarray(self.bounds_cm, float)
        return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max()))

    @property
    def width(self):
        xmin, xmax, _, _ = self.bbox
        return xmax - xmin

    @property
    def height(self):
        _, _, ymin, ymax = self.bbox
        return ymax - ymin

    def centroid(self):
        return _centroid(self.bounds_cm)

    def point_in_bounds(self, p):
        pt = _to_point(p)
        if pt is None:
            return False
        return point_in_polygon(pt, self.bounds_cm)

    def blocking(self):
        """Every shape that should be avoided, at its current position.

        Static obstacles plus obstacle-role entities. Single definition so a
        moving entity blocks everywhere a static one does — validation, the
        controller, nearest_valid_point — without any of them knowing entities
        exist.
        """
        return list(self.obstacles) + self.entities.blocking_shapes()

    def point_in_obstacle(self, p):
        pt = _to_point(p)
        if pt is None:
            return False
        for o in self.blocking():
            if o.get("type") == "poly":
                if point_in_polygon(pt, o["points"]):
                    return True
            elif o.get("type") == "circle":
                cx, cy = o["center"]
                if math.hypot(pt[0] - cx, pt[1] - cy) <= o["radius"]:
                    return True
        return False

    def is_valid_point(self, p):
        return self.point_in_bounds(p) and not self.point_in_obstacle(p)

    def _push_out_of_obstacles(self, p, clearance=0.0):
        """One pass: if p sits in an obstacle, push it past that obstacle's edge."""
        out = NUDGE + clearance
        for o in self.blocking():
            if o.get("type") == "poly" and point_in_polygon(p, o["points"]):
                boundary, _ = _nearest_boundary_point(p, o["points"])
                away = boundary - _centroid(o["points"])
                n = np.linalg.norm(away)
                away = away / n if n > EPS else np.array([1.0, 0.0])
                return boundary + away * out
            if o.get("type") == "circle":
                cx, cy = o["center"]
                center = np.array([cx, cy])
                d = np.linalg.norm(p - center)
                if d <= o["radius"] + clearance:
                    away = (p - center) / d if d > EPS else np.array([1.0, 0.0])
                    return center + away * (o["radius"] + out)
        return p

    def _push_off_walls(self, p, clearance, rounds=6):
        """Move p perpendicular away from every wall closer than `clearance`.

        Pushing toward the centroid does not work in a corner: the inward
        direction is diagonal, so it barely increases the gap from either wall
        and the search exhausts its iterations while the point is still
        unreachable. Pushing off each offending wall along its own normal
        converges immediately, and for a rectangle it is exactly a clamp.
        """
        cand = np.asarray(p, dtype=float).copy()
        centre = np.asarray(self.centroid(), dtype=float)
        n = len(self.bounds_cm)

        for _ in range(rounds):
            moved = False
            for i in range(n):
                a = np.asarray(self.bounds_cm[i], dtype=float)
                b = np.asarray(self.bounds_cm[(i + 1) % n], dtype=float)
                d, proj = _dist_point_segment(cand, a, b)
                if d >= clearance:
                    continue
                inward = centre - proj
                edge = b - a
                nrm = np.array([-edge[1], edge[0]], dtype=float)
                ln = float(np.linalg.norm(nrm))
                if ln < EPS:
                    continue
                nrm /= ln
                if float(nrm @ inward) < 0:
                    nrm = -nrm                     # point it into the arena
                cand = cand + nrm * (clearance - d + NUDGE)
                moved = True
            if not moved:
                break
        return cand

    def has_clearance(self, p, clearance):
        """True if p is at least `clearance` from every wall and obstacle."""
        return not self._too_close_to_edge(p, clearance)

    def _too_close_to_edge(self, p, clearance):
        """Inside the workspace, but within `clearance` of a wall or an obstacle."""
        if clearance <= 0:
            return False
        _, d_wall = _nearest_boundary_point(p, self.bounds_cm)
        if d_wall < clearance:
            return True
        for o in self.blocking():
            if o.get("type") == "circle":
                if np.linalg.norm(p - np.asarray(o["center"], float)) < o["radius"] + clearance:
                    return True
            elif o.get("type") == "poly":
                _, d = _nearest_boundary_point(p, o["points"])
                if d < clearance:
                    return True
        return False

    def nearest_valid_point(self, p, clearance=0.0):
        """Best-effort valid point closest to p. Never raises, always returns something usable.

        `clearance` keeps the result that far off every wall and obstacle. A
        target half a centimetre from a wall is geometrically valid and
        physically useless: a robot is a ball with a radius, and a controller
        that avoids obstacles will never settle on a point buried in its own
        repulsion field.
        """
        pt = _to_point(p)
        if pt is None:
            return np.asarray(self.centroid(), float)
        cand = np.asarray(pt, float)

        if (self.point_in_bounds(cand) and not self.point_in_obstacle(cand)
                and not self._too_close_to_edge(cand, clearance)):
            return cand

        for _ in range(8):
            if not point_in_polygon(cand, self.bounds_cm):
                boundary, _ = _nearest_boundary_point(cand, self.bounds_cm)
                inward = self.centroid() - boundary
                n = np.linalg.norm(inward)
                inward = inward / n if n > EPS else np.array([0.0, 0.0])
                cand = boundary + inward * (NUDGE + clearance)
            if clearance > 0 and self._wall_gap(cand) < clearance:
                cand = self._push_off_walls(cand, clearance)
            if self.point_in_obstacle(cand) or self._too_close_to_edge(cand, clearance):
                cand = self._push_out_of_obstacles(cand, clearance)
            if (self.point_in_bounds(cand) and not self.point_in_obstacle(cand)
                    and not self._too_close_to_edge(cand, clearance)):
                return cand

        # Nothing satisfied the clearance; fall back to merely valid.
        if clearance > 0:
            return self.nearest_valid_point(p, clearance=0.0)
        return self.random_valid_point()

    def _wall_gap(self, p):
        if not point_in_polygon(p, self.bounds_cm):
            return -1.0
        _, d = _nearest_boundary_point(p, self.bounds_cm)
        return d

    def segment_blocked(self, a, b, step_cm=1.0):
        """True if the straight line from a to b leaves bounds or crosses an obstacle."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        dist = float(np.linalg.norm(b - a))
        steps = max(2, int(dist / step_cm) + 1)
        for i in range(steps + 1):
            t = i / steps
            p = a + (b - a) * t
            if not self.point_in_bounds(p) or self.point_in_obstacle(p):
                return True
        return False

    def random_valid_point(self, rng=None, tries=5000):
        rng = rng or np.random.default_rng()
        xmin, xmax, ymin, ymax = self.bbox
        for _ in range(tries):
            p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            if self.point_in_bounds(p) and not self.point_in_obstacle(p):
                return p
        return np.asarray(self.centroid(), float)


# -- editing helpers -------------------------------------------------------
#
# The UI edits obstacles as "a shape, a place and a size". Circles carry that
# directly; rectangles are stored as 4-point polygons, so centre and half-size
# are derived rather than stored. Keeping the conversion here means the
# renderer never has to know how an obstacle is written to disk.

def make_circle(center, radius):
    return {"type": "circle", "center": [round(float(center[0]), 1),
                                          round(float(center[1]), 1)],
            "radius": round(float(max(radius, 2.0)), 1)}


def make_rect(center, half):
    cx, cy = float(center[0]), float(center[1])
    h = max(float(half), 2.0)
    return {"type": "poly", "points": [
        [round(cx - h, 1), round(cy - h, 1)],
        [round(cx + h, 1), round(cy - h, 1)],
        [round(cx + h, 1), round(cy + h, 1)],
        [round(cx - h, 1), round(cy + h, 1)],
    ]}


def obstacle_center(o):
    if o.get("type") == "circle":
        return np.asarray(o["center"], dtype=float)
    pts = np.asarray(o["points"], dtype=float)
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


def obstacle_size(o):
    """Radius for a circle, half-width for a rectangle."""
    if o.get("type") == "circle":
        return float(o["radius"])
    pts = np.asarray(o["points"], dtype=float)
    return float((pts.max(axis=0) - pts.min(axis=0)).max() / 2.0)


def rebuild_obstacle(o, center=None, size=None, kind=None):
    """Return a new obstacle with any of place, size or shape changed."""
    c = obstacle_center(o) if center is None else np.asarray(center, dtype=float)
    s = obstacle_size(o) if size is None else float(size)
    k = (o.get("type") if kind is None else kind)
    return make_circle(c, s) if k == "circle" else make_rect(c, s)


def obstacle_label(o):
    if o.get("type") == "circle":
        return "circle"
    pts = np.asarray(o["points"], dtype=float)
    w, h = (pts.max(axis=0) - pts.min(axis=0))
    return "square" if abs(w - h) < 1.0 else "rect"
