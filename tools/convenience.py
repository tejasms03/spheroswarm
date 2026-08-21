"""Tools that take identifiers and do the geometry themselves.

These exist because of a measured limitation, not for tidiness. The deployment
model is a 9B on a laptop, and commands that require it to read two positions
out of state and construct a mapping from them fail reliably — it will happily
invent coordinates that are nowhere near either robot. "Swap Seasmoke and
Caraxes" is arithmetic on freshly-read state, and that is exactly the shape of
task it is worst at.

So the model names *who*, and this module works out *where*. Every result still
goes through `tools/validate.py` like any other target set: these are a
shortcut for the model, not a way around the safety layer.
"""

import math
import random

import numpy as np

from .movement import move_to
from .result import fail, ok

MIN_GAP = 20.0            # cm, the same floor the validator enforces
SPOT_TRIES = 12
SPOT_R_MIN = 25.0
SPOT_R_MAX = 60.0
SPREAD_ITERATIONS = 50
DIRECTIONS = {"left": (-1.0, 0.0), "right": (1.0, 0.0),
              "up": (0.0, -1.0), "down": (0.0, 1.0)}


# -- shared helpers -----------------------------------------------------------

def _robot(ctx, code):
    """(position, error). Accepts a code; names are resolved before this."""
    if code not in ctx.active_codes():
        return None, (f"unknown or disconnected robot {code!r}; "
                      f"connected: {', '.join(ctx.active_codes())}")
    return np.asarray(ctx.fleet[code].pos, dtype=float).copy(), None


def _anything(ctx, ident):
    """Position of a robot code or an entity id, or (None, error)."""
    if ident in ctx.active_codes():
        return np.asarray(ctx.fleet[ident].pos, dtype=float).copy(), None
    if hasattr(ctx.ws, "entities"):
        e = ctx.ws.entities.by_id(ident)
        if e is not None:
            return e.pos.copy(), None
    names = list(ctx.active_codes())
    if hasattr(ctx.ws, "entities"):
        names += [e.id for e in ctx.ws.entities]
    return None, f"nothing called {ident!r}; known: {', '.join(names) or 'none'}"


def _others(ctx, exclude):
    return [np.asarray(ctx.fleet[c].pos, dtype=float)
            for c in ctx.active_codes() if c not in exclude]


def free_spot_near(p, ctx, exclude=(), tries=SPOT_TRIES, r_min=SPOT_R_MIN,
                   r_max=SPOT_R_MAX, rng=None):
    """A valid, uncrowded point near `p`. Cheap on purpose.

    Twelve dart throws, first valid wins. Nobody cares exactly where a
    displaced robot lands, so searching properly would be effort spent on a
    question with no right answer. Falls back to the nearest valid point, which
    cannot fail.
    """
    rng = rng or random
    others = _others(ctx, set(exclude))
    p = np.asarray(p, dtype=float)

    for _ in range(tries):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(r_min, r_max)
        q = p + np.array([r * math.cos(a), r * math.sin(a)])
        if not ctx.ws.is_valid_point(q):
            continue
        if others and min(float(np.linalg.norm(q - o)) for o in others) <= MIN_GAP:
            continue
        return q

    # Darts are cheap but they can all miss — in a corner most land outside.
    # Sweep deterministically, and keep the roomiest point seen so that a
    # failure still returns the least bad answer instead of one sitting on
    # top of a robot.
    best, best_gap = None, -1.0
    for r in np.arange(r_min, max(r_max, r_min) + 140.0, 12.0):
        for k in range(24):
            a = 2 * math.pi * k / 24
            q = p + np.array([r * math.cos(a), r * math.sin(a)])
            if not ctx.ws.is_valid_point(q):
                continue
            gap = (min(float(np.linalg.norm(q - o)) for o in others)
                   if others else float("inf"))
            if gap > MIN_GAP:
                return q
            if gap > best_gap:
                best, best_gap = q, gap
    if best is not None:
        return best
    return np.asarray(ctx.ws.nearest_valid_point(p, clearance=12.0), dtype=float)


def _clear_of_others(ctx, point, moving, gap=MIN_GAP):
    """Shift `point` off any robot that is staying put.

    A reflection or a displacement is geometrically right and can still land
    on top of a robot nobody asked to move. The validator would reject the
    whole call; better to place it as close as possible and say so.
    """
    point = np.asarray(point, dtype=float)
    others = [np.asarray(ctx.fleet[c].pos, dtype=float)
              for c in ctx.active_codes() if c not in set(moving)]
    if not others:
        return point, False
    if min(float(np.linalg.norm(point - o)) for o in others) > gap:
        return point, False
    return free_spot_near(point, ctx, exclude=moving, r_min=gap * 1.1,
                          r_max=gap * 2.5), True


def _place(ctx, mapping, note=None, **extra):
    """Route a {code: point} mapping through the ordinary movement path."""
    result = move_to(ctx, assign={c: [float(p[0]), float(p[1])]
                                  for c, p in mapping.items()})
    if not result.get("ok"):
        return result
    if note:
        result["note"] = note
    result.update(extra)
    return result


# -- swap ---------------------------------------------------------------------

def swap(ctx, code_a=None, code_b=None):
    """Exchange two robots' positions."""
    if not code_a or not code_b:
        return fail(ctx, "swap needs two robot codes: code_a and code_b")
    if code_a == code_b:
        return fail(ctx, f"{code_a} cannot swap with itself")

    a, err = _robot(ctx, code_a)
    if err:
        return fail(ctx, err)
    b, err = _robot(ctx, code_b)
    if err:
        return fail(ctx, err)

    return _place(ctx, {code_a: b, code_b: a},
                  note=f"{code_a} and {code_b} exchanged places")


# -- displace -------------------------------------------------------------------

def displace(ctx, code_a=None, code_b=None):
    """A takes B's place; B steps aside to somewhere valid nearby."""
    if not code_a or not code_b:
        return fail(ctx, "displace needs code_a (the mover) and code_b (the one "
                         "being displaced)")
    if code_a == code_b:
        return fail(ctx, f"{code_a} is already where it is")

    a, err = _robot(ctx, code_a)
    if err:
        return fail(ctx, err)
    b, err = _robot(ctx, code_b)
    if err:
        return fail(ctx, err)

    spot = free_spot_near(b, ctx, exclude=(code_a, code_b))
    spot, _ = _clear_of_others(ctx, spot, moving=(code_a, code_b))
    return _place(ctx, {code_a: b, code_b: spot},
                  note=f"{code_a} took {code_b}'s place; {code_b} moved aside",
                  displaced_to=[round(float(spot[0]), 1), round(float(spot[1]), 1)])


# -- nudge -------------------------------------------------------------------------

def nudge(ctx, codes=None, direction=None, distance=30.0):
    """Shift robots a short way. `direction` may name a robot or entity."""
    if not direction:
        return fail(ctx, "nudge needs a direction: left, right, up, down, "
                         "toward:<id> or away:<id>")
    try:
        dist = float(distance)
    except (TypeError, ValueError):
        return fail(ctx, f"distance must be a number, got {distance!r}")
    if not np.isfinite(dist) or dist <= 0:
        return fail(ctx, "distance must be a positive number of centimetres")

    if isinstance(codes, str):
        codes = [codes]
    wanted = list(codes) if codes else list(ctx.active_codes())
    if not wanted:
        return fail(ctx, "no robots are connected")
    unknown = [c for c in wanted if c not in ctx.active_codes()]
    if unknown:
        return fail(ctx, f"unknown or disconnected robot(s): {', '.join(unknown)}")

    d = str(direction).strip().lower()
    mapping = {}

    if d in DIRECTIONS:
        vec = np.array(DIRECTIONS[d], dtype=float)
        for c in wanted:
            p, _ = _robot(ctx, c)
            mapping[c] = p + vec * dist
    elif d.startswith("toward:") or d.startswith("away:"):
        ident = direction.split(":", 1)[1].strip()
        anchor, err = _anything(ctx, ident)
        if err:
            return fail(ctx, err)
        sign = 1.0 if d.startswith("toward:") else -1.0
        for c in wanted:
            p, _ = _robot(ctx, c)
            v = anchor - p
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                mapping[c] = p
                continue
            mapping[c] = p + sign * (v / n) * dist
    else:
        return fail(ctx, f"unknown direction {direction!r}; use left, right, up, "
                         "down, toward:<id> or away:<id>")

    return _place(ctx, mapping, note=f"nudged {', '.join(mapping)} {d} by {dist:.0f}cm")


# -- gather --------------------------------------------------------------------------

def gather(ctx, codes=None, around=None, radius=45.0):
    """Cluster robots in a ring around a robot or entity."""
    if not around:
        return fail(ctx, "gather needs `around`: a robot code or entity id")
    centre, err = _anything(ctx, around)
    if err:
        return fail(ctx, err)

    try:
        r = float(radius)
    except (TypeError, ValueError):
        return fail(ctx, f"radius must be a number, got {radius!r}")
    if not np.isfinite(r) or r <= 0:
        return fail(ctx, "radius must be a positive number of centimetres")

    if isinstance(codes, str):
        codes = [codes]
    wanted = [c for c in (codes or ctx.active_codes()) if c != around]
    if not wanted:
        return fail(ctx, "no robots to gather")
    unknown = [c for c in wanted if c not in ctx.active_codes()]
    if unknown:
        return fail(ctx, f"unknown or disconnected robot(s): {', '.join(unknown)}")

    # A ring wide enough that the robots are not inside the validator's floor
    # Two constraints, both easy to miss: neighbours around the ring must clear
    # each other, and every robot on the ring must clear whatever is at its
    # centre — which is usually another robot.
    n = len(wanted)
    chord = (MIN_GAP * 1.15) / (2 * math.sin(math.pi / n)) if n > 1 else 0.0
    r = max(r, chord, MIN_GAP * 1.25)

    mapping = {}
    for i, c in enumerate(wanted):
        a = 2 * math.pi * i / n
        mapping[c] = centre + r * np.array([math.cos(a), math.sin(a)])

    return _place(ctx, mapping, note=f"gathered {n} robot(s) around {around} "
                                     f"at {r:.0f}cm", around=around, radius=round(r, 1))


# -- spread ----------------------------------------------------------------------------

def spread(ctx, codes=None, min_distance=60.0):
    """Push robots apart to at least `min_distance`, without naming places."""
    try:
        want = float(min_distance)
    except (TypeError, ValueError):
        return fail(ctx, f"min_distance must be a number, got {min_distance!r}")
    if not np.isfinite(want) or want <= 0:
        return fail(ctx, "min_distance must be a positive number of centimetres")

    if isinstance(codes, str):
        codes = [codes]
    wanted = list(codes) if codes else list(ctx.active_codes())
    if len(wanted) < 2:
        return fail(ctx, "spreading needs at least two robots")
    unknown = [c for c in wanted if c not in ctx.active_codes()]
    if unknown:
        return fail(ctx, f"unknown or disconnected robot(s): {', '.join(unknown)}")

    pts = np.array([ctx.fleet[c].pos for c in wanted], dtype=float)

    # Plain relaxation: push every too-close pair apart, clamp back inside,
    # repeat. Capped because it need not converge, only improve.
    for _ in range(SPREAD_ITERATIONS):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                v = pts[i] - pts[j]
                d = float(np.linalg.norm(v))
                if d >= want:
                    continue
                moved = True
                push = (v / d if d > 1e-6 else
                        np.array([math.cos(i), math.sin(i)])) * (want - d) / 2.0
                pts[i] = pts[i] + push
                pts[j] = pts[j] - push
        for k in range(len(pts)):
            pts[k] = ctx.ws.nearest_valid_point(pts[k], clearance=12.0)
        if not moved:
            break

    mapping = {c: pts[i] for i, c in enumerate(wanted)}
    gaps = [float(np.linalg.norm(pts[i] - pts[j]))
            for i in range(len(pts)) for j in range(i + 1, len(pts))]
    return _place(ctx, mapping,
                  note=f"spread to a minimum gap of {min(gaps):.0f}cm",
                  min_gap=round(min(gaps), 1))


# -- mirror ------------------------------------------------------------------------------

def mirror(ctx, code_a=None, code_b=None, axis="vertical"):
    """Place A as B's reflection: across an axis, or through a pivot."""
    if not code_a or not code_b:
        return fail(ctx, "mirror needs code_a (the one that moves) and code_b")
    if code_a == code_b:
        return fail(ctx, f"{code_a} cannot mirror itself")

    a, err = _robot(ctx, code_a)
    if err:
        return fail(ctx, err)
    b, err = _robot(ctx, code_b)
    if err:
        return fail(ctx, err)

    xmin, xmax, ymin, ymax = ctx.ws.bbox
    spec = str(axis).strip().lower()

    if spec == "vertical":
        target = np.array([xmin + xmax - b[0], b[1]])
    elif spec == "horizontal":
        target = np.array([b[0], ymin + ymax - b[1]])
    elif spec.startswith("through:"):
        ident = axis.split(":", 1)[1].strip()
        pivot, perr = _anything(ctx, ident)
        if perr:
            return fail(ctx, perr)
        # point reflection: the pivot ends up exactly between them
        target = 2 * pivot - b
    elif spec in ("centre", "center", "through"):
        target = 2 * np.asarray(ctx.ws.centroid(), dtype=float) - b
    else:
        return fail(ctx, f"unknown axis {axis!r}; use vertical, horizontal, "
                         "centre, or through:<id>")

    target = np.asarray(ctx.ws.nearest_valid_point(target, clearance=12.0),
                        dtype=float)
    target, shifted = _clear_of_others(ctx, target, moving=(code_a,))
    note = f"{code_a} placed as {code_b}'s reflection ({spec})"
    if shifted:
        note += " — nudged clear of a robot already standing there"
    return _place(ctx, {code_a: target}, note=note,
                  reflected_to=[round(float(target[0]), 1),
                                 round(float(target[1]), 1)])
