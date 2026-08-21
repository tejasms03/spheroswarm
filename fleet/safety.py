"""How fast a robot may go, given how much floor is left in front of it.

Recovering a ball that has already left the arena is the wrong mechanism. By
then it is off camera, the run is spoiled, and somebody is walking over to pick
it up — which is both the slowest part of a session and, done repeatedly, a
source of exactly the disturbance the session is trying to measure.

So speed is limited continuously instead. The limit is not a guess: a ball
takes `stop_s` seconds of travel to come to rest — the brake test measures
precisely that, delay and roll-out together — so a robot `d` centimetres from
the wall can safely be doing `d / stop_s` and no more. Divided by a margin,
because the measurement has an error bar and the floor does not move.

The result is a field rather than a switch. In the middle of the arena nothing
is limited at all; approaching an edge the ceiling falls smoothly to zero, and
a robot simply cannot be commanded into a wall it does not have room to stop
before. Nothing has to notice it is in trouble, because it never is.
"""

import numpy as np

# What a ball loses between being told to stop and stopping, per unit speed:
# the loop delay plus the roll-out. Replaced per robot from `calib/motion.json`
# — this is the conservative end of what the sim models.
DEFAULT_STOP_S = 0.55
SAFETY = 2.5            # of stopping distances of clearance to run flat out
FLOOR_CM_S = 4.0        # below this a Sphero does not move at all, so do not ask
HARD_STOP_CM = 6.0      # this close to a wall, nothing outward is allowed


def stopping_seconds(fit=None):
    """`stop_s` for one robot, measured if it has been.

    The evidence is checked here, not just the conclusion. This is the one
    consumer of a calibration that nobody reads: the gains panel puts its
    numbers in front of a person who can doubt them, while this quietly decides
    how fast every robot is allowed to move. A calibration written before the
    brake fit learned to distrust itself carries a published constant with no
    verdict attached, and taking that on faith is how a run that failed goes on
    governing the arena weeks later.

    Falling back is cheap and safe: `DEFAULT_STOP_S` is the conservative end of
    what the sim models, so an unmeasured robot is limited more than it needs
    to be rather than less.
    """
    fit = fit or {}
    rec = fit.get("recommend") or {}
    k = rec.get("stopping_distance_s_per_cm_s")
    try:
        k = float(k)
    except (TypeError, ValueError):
        return DEFAULT_STOP_S
    if k <= 0.05:
        return DEFAULT_STOP_S

    # Imported here rather than at module scope: `characterize` imports this
    # module for its own limiting, so the pair would not load.
    from .characterize import _brake_trusted
    brake = fit.get("brake")
    if brake and not _brake_trusted(brake):
        return DEFAULT_STOP_S
    return k


def clearance(ws, pos):
    """Centimetres to the nearest wall or obstacle. None when unknowable."""
    if ws is None:
        return None
    p = np.asarray(pos, dtype=float)
    try:
        if not ws.is_valid_point(p):
            return 0.0
    except Exception:
        return None
    # `has_clearance` answers yes/no, so the distance is found by bisection.
    # Twelve steps resolves a 300cm arena to under a millimetre, and it costs
    # twelve cheap geometry calls on a robot that is nowhere near anything.
    lo, hi = 0.0, 400.0
    try:
        if ws.has_clearance(p, hi):
            return hi
        for _ in range(12):
            mid = (lo + hi) / 2.0
            if ws.has_clearance(p, mid):
                lo = mid
            else:
                hi = mid
    except Exception:
        return None
    return lo


def speed_ceiling(ws, pos, stop_s=DEFAULT_STOP_S, safety=SAFETY, cap=60.0):
    """cm/s this robot may be commanded at, here. `cap` when there is room."""
    d = clearance(ws, pos)
    if d is None:
        return cap
    if d <= HARD_STOP_CM:
        return 0.0
    allowed = (d - HARD_STOP_CM) / max(stop_s * safety, 1e-6)
    return float(min(cap, allowed))


def limit(ws, pos, heading_deg, byte, stop_s=DEFAULT_STOP_S, max_speed=60.0,
          safety=SAFETY):
    """Clamp one (heading, byte) command to what the floor here allows.

    Direction is taken into account, and it matters more than it looks: a
    robot pinned against a wall must still be able to drive AWAY from it, and a
    limiter that only looks at distance would hold it there until somebody
    picked it up. Only the component heading further out is limited.
    """
    if byte <= 0:
        return heading_deg, 0, False
    d = clearance(ws, pos)
    if d is None:
        return heading_deg, int(byte), False

    requested = byte / 255.0 * max_speed
    ceiling = speed_ceiling(ws, pos, stop_s, safety, cap=max_speed)
    if requested <= ceiling:
        return heading_deg, int(byte), False

    if d <= HARD_STOP_CM * 2:
        # Close in, allow full speed as long as it is heading inward. The
        # alternative — clamping everything by distance alone — traps a ball at
        # the edge, which is the state we are trying to get out of.
        import math
        rad = math.radians(float(heading_deg))
        direction = np.array([math.sin(rad), math.cos(rad)])
        inward = _inward(ws, pos)
        if inward is not None and float(np.dot(direction, inward)) > 0.35:
            return heading_deg, int(byte), False

    allowed = max(ceiling, FLOOR_CM_S if ceiling > 0 else 0.0)
    return heading_deg, int(np.clip(allowed / max_speed * 255.0, 0, 255)), True


def _inward(ws, pos):
    """A unit vector pointing away from the nearest edge, or None."""
    try:
        x0, x1, y0, y1 = ws.bbox
    except Exception:
        return None
    p = np.asarray(pos, dtype=float)
    to = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0]) - p
    n = float(np.linalg.norm(to))
    return None if n < 1e-6 else to / n


def room_for(ws, pos, course_deg, speed_cm_s, stop_s=DEFAULT_STOP_S,
             leg_cm=0.0, safety=SAFETY):
    """Is there floor for a leg of this length AT this speed, this way?

    Stages call this instead of asking for a fixed clearance, so a fast leg
    demands more room than a slow one. Without it the limiter has to clamp a
    leg mid-measurement, and a speed map whose fastest points were quietly
    slowed down is worse than one with no fast points at all.
    """
    import math
    rad = math.radians(float(course_deg))
    reach = float(leg_cm) + float(speed_cm_s) * stop_s * safety
    end = (float(pos[0]) + math.sin(rad) * reach,
           float(pos[1]) + math.cos(rad) * reach)
    try:
        return ws is None or (ws.is_valid_point(end)
                              and ws.has_clearance(end, HARD_STOP_CM))
    except Exception:
        return True


def available_cm(ws, pos, course_deg, step=4.0, limit_cm=400.0):
    """How far a robot at `pos` can travel along `course_deg` and still stop.

    Marched rather than solved, because the arena is a polygon with obstacles
    in it and the first thing in the way is not always the nearest wall.
    """
    import math
    if ws is None:
        return limit_cm
    rad = math.radians(float(course_deg))
    d = np.array([math.sin(rad), math.cos(rad)])
    p = np.asarray(pos, dtype=float)
    travelled = 0.0
    while travelled < limit_cm:
        q = p + d * (travelled + step)
        try:
            if not (ws.is_valid_point(q) and ws.has_clearance(q, HARD_STOP_CM)):
                break
        except Exception:
            break
        travelled += step
    return travelled


def available_seconds(ws, pos, course_deg, speed_cm_s, stop_s=DEFAULT_STOP_S,
                      safety=SAFETY):
    """How long a robot may drive that way at that speed before it must stop.

    This is what turns a battery that works on paper into one that works in a
    200cm room. A leg specified as "1.1s settling then 1.4s measuring" is 150cm
    at full speed, and from the middle of a 200cm arena there is 100cm — so the
    fast end of a speed map was always going to end at a wall, however
    carefully the direction was chosen. Asking how much floor there is, and
    fitting the leg to it, is the difference.
    """
    speed = max(float(speed_cm_s), 1e-6)
    room = available_cm(ws, pos, course_deg) - speed * stop_s * safety
    return max(0.0, room / speed)
