"""The formation library the model builds for itself.

`formations.json` starts empty and stays that way until a model arranges robots
and a user names the result. Nothing in here knows what a shape *is* — a wedge
and a smiley are both just point sets — which is exactly why an arbitrary
arrangement can be saved and recalled.

Points are stored normalised: centroid at the origin, scaled so the furthest
point sits at radius 1. That is what lets a shape saved small in one corner
come back anywhere, at any size, at any angle.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "formations.json"


def normalise(points):
    """Return (unit_points, centroid, radius). Radius 1 means "furthest point"."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    radius = float(np.linalg.norm(centred, axis=1).max())
    if radius < 1e-9:
        return centred, centroid, 0.0        # every robot in one spot
    return centred / radius, centroid, radius


def denormalise(unit_points, center, scale, rotation_deg=0.0):
    """Inverse of `normalise`, with an optional rotation about the centre."""
    pts = np.asarray(unit_points, dtype=float).reshape(-1, 2)
    if rotation_deg:
        a = math.radians(rotation_deg)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
    return pts * float(scale) + np.asarray(center, dtype=float)


class FormationLibrary:
    def __init__(self, formations=None, path=DEFAULT_PATH):
        self.path = Path(path)
        self.formations = formations or {}
        self.errors = []

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        path = Path(path)
        lib = cls(path=path)
        if not path.exists():
            return lib                        # empty is the correct first-run state
        try:
            raw = json.loads(path.read_text())
        except Exception as e:
            lib.errors = [f"could not parse {path}: {e}"]
            return lib
        if not isinstance(raw, dict):
            lib.errors = [f"{path} should hold an object keyed by formation name"]
            return lib
        lib.formations = raw
        return lib

    def save(self, path=None):
        target = Path(path) if path else self.path
        try:
            target.write_text(json.dumps(self.formations, indent=2))
        except Exception as e:
            return [f"could not write {target}: {e}"]
        return []

    # -- queries ------------------------------------------------------------

    def names(self):
        return sorted(self.formations)

    def get(self, name):
        return self.formations.get(name)

    def listing(self):
        out = []
        for name in self.names():
            f = self.formations[name]
            out.append({
                "name": name,
                "description": f.get("description", ""),
                "robots": f.get("robot_count", len(f.get("points", []))),
                "created": f.get("created", ""),
                "slots": f.get("slots", []),
                # So the model can tell "a shape" from "a shape that moves"
                # without recalling it to find out.
                "moves": bool(f.get("motion")),
            })
        return out

    # -- mutation ------------------------------------------------------------

    def save_formation(self, name, points, codes=None, description="",
                       motion=None):
        name = (name or "").strip()
        if not name:
            return ["a formation needs a name"]
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return ["nothing to save: no robot positions"]
        if not np.isfinite(pts).all():
            return ["cannot save a formation containing non-finite coordinates"]

        unit, centroid, radius = normalise(pts)
        entry = {
            "points": [[round(float(x), 5), round(float(y), 5)] for x, y in unit],
            "robot_count": int(len(pts)),
            "slots": list(codes) if codes else [],
            "description": description or "",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_radius_cm": round(radius, 1),
            "saved_center_cm": [round(float(centroid[0]), 1), round(float(centroid[1]), 1)],
        }
        if motion:
            # Stored in the same normalised frame as the points, so a patrol
            # taught small in one corner comes back anywhere, at any size, at
            # any angle — exactly what makes a saved shape reusable.
            entry["motion"] = motion
        self.formations[name] = entry
        return self.save()

    @staticmethod
    def normalise_motion(stack, centroid, radius):
        """Snapshot the running layers, in the shape frame.

        Only what is needed to replay it: a flow's expression is already
        parametric in `cx, cy, ex, ey` and rebinds itself on recall, a path's
        waypoints are geometry and must scale with the shape, and a follow is a
        target plus a distance.
        """
        r = max(float(radius), 1e-6)
        out = {}
        for code in stack.codes():
            layers = []
            for layer in stack.layers(code):
                if layer.kind == "seek":
                    continue
                p = dict(layer.params or {})
                p.pop("field", None)          # a compiled callable; rebuilt on recall
                if layer.kind == "path" and p.get("waypoints"):
                    p["waypoints"] = [
                        [round((float(x) - float(centroid[0])) / r, 5),
                         round((float(y) - float(centroid[1])) / r, 5)]
                        for x, y in p["waypoints"]]
                if layer.kind == "follow" and p.get("distance") is not None:
                    p["distance_norm"] = round(float(p["distance"]) / r, 5)
                layers.append({"name": layer.name, "kind": layer.kind,
                               "weight": round(float(layer.weight), 3),
                               "params": p})
            if layers:
                out[code] = layers
        return out or None

    def delete(self, name):
        if name not in self.formations:
            return [f"no formation named {name!r}"]
        del self.formations[name]
        return self.save()

    # -- recall ---------------------------------------------------------------

    def recall(self, name, robot_count, center, scale=None, rotation=0.0):
        """Denormalise a saved shape. Returns (points, errors).

        A mismatched robot count is refused rather than guessed at: resampling
        an arbitrary point set to a different N is ambiguous, and quietly
        producing the wrong shape is worse than declining.
        """
        f = self.formations.get(name)
        if f is None:
            known = ", ".join(self.names()) or "none saved yet"
            return None, [f"no formation named {name!r} (have: {known})"]

        saved = int(f.get("robot_count", len(f.get("points", []))))
        if saved != robot_count:
            return None, [f"formation {name!r} was saved with {saved} robots but "
                          f"{robot_count} are available — recall needs the same count"]

        unit = np.asarray(f["points"], dtype=float).reshape(-1, 2)
        if scale is None:
            scale = f.get("saved_radius_cm", 1.0)
        try:
            scale = float(scale)
            rotation = float(rotation or 0.0)
        except (TypeError, ValueError):
            return None, [f"scale and rotation must be numbers, got {scale!r}, {rotation!r}"]
        if not (math.isfinite(scale) and math.isfinite(rotation)):
            return None, ["scale and rotation must be finite numbers"]
        if scale <= 0:
            return None, [f"scale must be positive, got {scale}"]

        return denormalise(unit, center, scale, rotation), []
