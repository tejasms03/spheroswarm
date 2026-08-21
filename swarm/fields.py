"""What each layer kind contributes to a robot's velocity.

Separate from `layers.py` on purpose: that module is bookkeeping — names,
weights, durations, caps — and this one is geometry. Keeping them apart means
adding a layer kind never risks the rules that stop a swarm running away.

Every function returns a contribution in the controller's normalised units,
where magnitude 1.0 means "full speed in this direction". The controller sums
them by weight, then applies avoidance, then clamps.
"""

import numpy as np

LEAD_TIME = 0.4          # s, loop latency plus Sphero acceleration lag
ARRIVE_TOL = 6.0
SLOW_RADIUS = 25.0
HYSTERESIS = 12.0        # cm a slot must beat the current one to steal a robot


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


def _ease(to_target, slow_radius=SLOW_RADIUS, arrive_tol=ARRIVE_TOL):
    """Seek with the same easing the positional controller has always used."""
    d = float(np.linalg.norm(to_target))
    if d <= arrive_tol:
        return np.zeros(2), d
    return to_target / d * min(1.0, d / slow_radius), d


# -- seek --------------------------------------------------------------------

def seek(pos, target, slow_radius=SLOW_RADIUS, arrive_tol=ARRIVE_TOL):
    v, _ = _ease(np.asarray(target, float) - np.asarray(pos, float),
                 slow_radius, arrive_tol)
    return v


# -- path --------------------------------------------------------------------

def path(pos, layer, workspace=None):
    """Walk waypoints. Advances the layer's own leg counter as it arrives.

    Each waypoint is checked as it becomes active rather than once up front:
    a looping path outlives the world it was planned in, and an entity may have
    moved into a leg that was clear a minute ago.
    """
    wps = layer.params.get("waypoints") or []
    if len(wps) < 1:
        return np.zeros(2)

    mode = layer.params.get("mode", "once")
    leg = layer.state.get("leg", 0)
    direction = layer.state.get("dir", 1)

    leg = max(0, min(leg, len(wps) - 1))
    target = np.asarray(wps[leg], dtype=float)

    # skip a waypoint that has become unreachable rather than grinding into it
    if workspace is not None and not workspace.is_valid_point(target):
        blocked = layer.state.get("blocked", 0) + 1
        layer.state["blocked"] = blocked
        if blocked < len(wps):
            leg, direction = _advance(leg, direction, len(wps), mode, layer)
            layer.state["leg"], layer.state["dir"] = leg, direction
            return np.zeros(2)
    else:
        layer.state["blocked"] = 0

    v, d = _ease(target - np.asarray(pos, float))
    if d <= ARRIVE_TOL:
        leg, direction = _advance(leg, direction, len(wps), mode, layer)
        layer.state["leg"], layer.state["dir"] = leg, direction
        target = np.asarray(wps[max(0, min(leg, len(wps) - 1))], dtype=float)
        v, _ = _ease(target - np.asarray(pos, float))
    return v


def _advance(leg, direction, n, mode, layer):
    nxt = leg + direction
    if 0 <= nxt < n:
        return nxt, direction
    if mode == "loop":
        return (0 if direction > 0 else n - 1), direction
    if mode == "pingpong":
        direction = -direction
        return max(0, min(n - 1, leg + direction)), direction
    layer.state["done"] = True                 # once: hold the last waypoint
    return max(0, min(n - 1, leg)), direction


# -- flow --------------------------------------------------------------------

def flow(pos, layer, t, entity_pos=None):
    """Evaluate a compiled velocity field at this robot's position.

    `entity_pos` binds ex/ey, which is what lets a field orbit something that
    is itself moving — the reason follow and flow compose into "orbit the
    wanderer" without a bespoke tool.
    """
    fn = layer.params.get("field")
    if fn is None:
        return np.zeros(2)
    ex, ey = (float(entity_pos[0]), float(entity_pos[1])) \
        if entity_pos is not None else (0.0, 0.0)
    try:
        vx, vy = fn(float(pos[0]), float(pos[1]), float(t), ex, ey)
    except Exception as e:
        layer.state["error"] = f"{type(e).__name__}: {e}"
        return np.zeros(2)
    v = np.array([vx, vy], dtype=float)
    if not np.isfinite(v).all():
        layer.state["error"] = "flow produced a non-finite velocity"
        return np.zeros(2)
    n = float(np.linalg.norm(v))
    return v / n if n > 1.0 else v


# -- follow -------------------------------------------------------------------

def predict(target_pos, target_vel, lead_time=LEAD_TIME):
    """Where the target will be, not where it was.

    Without this, followers visibly lag and then overshoot when the target
    stops. It has to be built in: every gain downstream is tuned against it.
    """
    return (np.asarray(target_pos, float)
            + np.asarray(target_vel, float) * float(lead_time))


def ring_phase_of(target_pos, follower_positions, count):
    """Circular mean bearing of the followers, so the ring turns with them."""
    target_pos = np.asarray(target_pos, dtype=float)
    angles = []
    for i, p in enumerate(follower_positions):
        v = np.asarray(p, dtype=float) - target_pos
        if float(np.linalg.norm(v)) < 1e-6:
            continue
        angles.append(np.arctan2(v[1], v[0]) - 2 * np.pi * i / max(count, 1))
    if not angles:
        return 0.0
    angles = np.asarray(angles)
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def follow_point(mode, index, count, target_pos, target_vel, distance,
                 follower_pos=None, ring_phase=None):
    """Where this follower should be, for the given follow mode."""
    aim = np.asarray(target_pos, dtype=float)
    heading = _unit(np.asarray(target_vel, dtype=float))
    if float(np.linalg.norm(heading)) < 1e-9:
        heading = np.array([1.0, 0.0])
    side = np.array([-heading[1], heading[0]])

    if mode == "trail":
        lane = (index - (count - 1) / 2.0) * 22.0
        return aim - heading * distance + side * lane
    if mode == "flank":
        sign = 1.0 if index % 2 == 0 else -1.0
        rank = index // 2
        return aim + side * sign * distance - heading * rank * 22.0
    if mode == "mirror":
        return None                            # velocity copy, not a point
    # surround: an even ring whose phase comes from where the robots already
    # are. Pinning slots to robot index would hold every follower at a fixed
    # bearing, so a flow layer pushing tangentially could never turn the ring —
    # and "orbit the wanderer" is supposed to be follow + flow composed.
    phase = 0.0 if ring_phase is None else float(ring_phase)
    ang = phase + 2 * np.pi * index / max(count, 1)
    return aim + distance * np.array([np.cos(ang), np.sin(ang)])


def assign_slots(codes, positions, slots, previous=None,
                 hysteresis=HYSTERESIS):
    """Hungarian slot assignment that does not thrash.

    Re-solving every tick makes robots swap slots on noise. A robot keeps its
    slot unless another beats it by `hysteresis` centimetres.
    """
    from scipy.optimize import linear_sum_assignment

    if not codes:
        return {}
    P = np.asarray(positions, dtype=float).reshape(len(codes), 2)
    S = np.asarray(slots, dtype=float).reshape(-1, 2)
    cost = np.linalg.norm(P[:, None, :] - S[None, :, :], axis=2)

    if previous:
        for i, c in enumerate(codes):
            j = previous.get(c)
            if j is not None and 0 <= j < len(S):
                cost[i, j] -= hysteresis        # incumbent advantage

    rows, cols = linear_sum_assignment(cost)
    return {codes[i]: int(cols[k]) for k, i in enumerate(rows)}
