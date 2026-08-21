"""Go-to-target control: the controller every tool drives.

Each robot owns a target; output is a velocity toward it, bent around
neighbours and obstacles. Assignment of robots to an unordered set of target
points is Hungarian, so the swarm slots into a shape rather than crossing
through itself to get there.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from fleet.handle import MAX_SPEED

ARRIVE_TOL = 6.0        # cm, inside this a robot is "there"
SLOW_RADIUS = 25.0      # cm, start easing off here so it does not overshoot
SEP_DIST = 22.0         # cm, neighbour separation kicks in below this
OBSTACLE_MARGIN = 22.0  # cm, how far out an obstacle starts pushing
SEP_FLOOR = 0.45        # separation never fades below this: robots collide
MAX_FOLLOW_SPEED = 60.0 # cm/s a mirrored velocity is normalised against


def assign(positions, targets):
    """Hungarian assignment. Returns `order` where targets[order[i]] belongs to robot i.

    Minimises total distance travelled. The naive alternative — pairing by
    index — routinely sends robots straight through each other.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    if len(positions) == 0 or len(targets) == 0:
        return np.zeros(0, dtype=int)

    cost = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    order = np.full(len(positions), -1, dtype=int)
    order[rows] = cols
    return order


def assignment_cost(positions, targets, order):
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    total = 0.0
    for i, j in enumerate(order):
        if 0 <= j < len(targets):
            total += float(np.linalg.norm(positions[i] - targets[j]))
    return total


class Navigate:
    """Same `act(env)` contract as `Boids`, so it is a drop-in controller."""

    label = "navigate"

    def __init__(self, targets=None, separation=0.9, obstacle=1.4,
                 arrive_tol=ARRIVE_TOL, slow_radius=SLOW_RADIUS,
                 sep_dist=SEP_DIST, stack=None):
        self.targets = None if targets is None else np.asarray(targets, dtype=float)
        self.separation = separation
        self.obstacle = obstacle
        self.arrive_tol = arrive_tol
        self.slow_radius = slow_radius
        self.sep_dist = sep_dist
        # Optional LayerStack. With none attached — or with nothing in it but a
        # seek layer — the arithmetic below is exactly what it always was.
        self.stack = stack
        self.t = 0.0
        # What one unit of normalised output is worth in cm/s. Only the stall
        # detector needs it, but it needs it to be true: hardcoding 60 while
        # the fleet is actually driven at 30 makes every robot look twice as
        # fast as it is, and a stall is then reported late or not at all.
        self.speed_scale = MAX_SPEED

    # -- targets -----------------------------------------------------------

    def set_targets(self, targets, positions=None):
        """Assign an unordered point set to robots. Returns the per-robot targets."""
        t = np.asarray(targets, dtype=float).reshape(-1, 2)
        if positions is None:
            self.targets = t
            return self.targets
        order = assign(positions, t)
        out = np.array([t[j] if 0 <= j < len(t) else np.asarray(positions)[i]
                        for i, j in enumerate(order)], dtype=float)
        self.targets = out
        return out

    def target_for(self, env):
        if self.targets is not None and len(self.targets) == env.n:
            return self.targets
        return np.asarray(getattr(env, "target_array", lambda: env.pos)(), dtype=float)

    def arrived(self, env, tol=None):
        tol = self.arrive_tol if tol is None else tol
        t = self.target_for(env)
        d = np.linalg.norm(env.pos - t, axis=1)
        return d <= tol

    # -- control -----------------------------------------------------------

    def act(self, env, obs=None):
        n = env.n
        pos = np.asarray(env.pos, dtype=float)
        out = np.zeros((n, 2))
        wanted = np.zeros(n)
        if n == 0:
            return out

        targets = self.target_for(env)
        if targets.shape != (n, 2):
            targets = pos

        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        ws = getattr(env, "ws", None)

        codes = list(getattr(env, "codes", []))

        for i in range(n):
            to_target = targets[i] - pos[i]
            d = float(np.linalg.norm(to_target))

            if d <= self.arrive_tol:
                seek = np.zeros(2)
            else:
                # ease into the target so a laggy robot does not sail past it
                speed = min(1.0, d / self.slow_radius)
                seek = to_target / d * speed

            v = seek

            # Extra layers ride on top of seek. With an empty stack this loop
            # does nothing at all, which is what keeps the positional
            # behaviour bit-identical.
            if self.stack is not None:
                extra, moved = self._blend(env, i, pos[i])
                v = v + extra
                if moved:
                    d = max(d, self.slow_radius)   # a driven robot is not "arrived"

            # What was asked for, before avoidance and clamping get a say. The
            # stall detector needs it to tell "blocked" from "satisfied".
            wanted[i] = float(np.linalg.norm(v))

            # How far this robot still has to travel, 0..1. Both repulsions
            # fade with it: they exist to keep robots apart *in transit*, not
            # to argue with a destination the validator already approved.
            fade = min(1.0, d / self.slow_radius)

            close = np.where(dist[i] < self.sep_dist)[0]
            if len(close):
                # Targets are only guaranteed 20cm apart while separation
                # starts pushing at 22cm, so two robots on perfectly legal
                # targets sit inside each other's field. At full strength they
                # settle into a standoff several centimetres short of their
                # targets and stop dead — never arriving, never timing out
                # cleanly. Fading with distance-to-target lets them land.
                push = diff[i, close] / np.maximum(dist[i, close], 1e-6)[:, None]
                weight = (self.sep_dist - dist[i, close]) / self.sep_dist
                v = v + self.separation * max(fade, SEP_FLOOR) * \
                    (push * weight[:, None]).sum(axis=0)

            if ws is not None:
                v = v + self.obstacle * fade * self._avoid(ws, pos[i], seek)

            m = float(np.linalg.norm(v))
            out[i] = v / m if m > 1.0 else v

        if self.stack is not None:
            self.stack.expire()
            for i, code in enumerate(codes[:n]):
                self.stack.note_speed(
                    code, float(np.linalg.norm(out[i])) * self.speed_scale,
                    wanted_cms=float(wanted[i]) * self.speed_scale)
        return out

    def _blend(self, env, i, p):
        """Weighted sum of every non-seek layer on this robot.

        Returns (contribution, is_driven). `is_driven` says a continuous layer
        is steering, so the robot must not be treated as parked just because it
        happens to be sitting on an old target.
        """
        from . import fields

        codes = list(getattr(env, "codes", []))
        if i >= len(codes):
            return np.zeros(2), False
        code = codes[i]

        stack = self.stack
        layers = [l for l in stack.layers(code) if l.kind != "seek"]
        if not layers:
            return np.zeros(2), False

        ws = getattr(env, "ws", None)
        out = np.zeros(2)
        driven = False

        for layer in layers:
            if not self._condition_holds(env, code, layer, p):
                continue
            if layer.kind == "path":
                out = out + layer.weight * fields.path(p, layer, ws)
                driven = True
            elif layer.kind == "flow":
                ent = self._entity_pos(env, layer.params.get("entity"))
                out = out + layer.weight * fields.flow(p, layer, self.t, ent)
                driven = True
            elif layer.kind == "follow":
                out = out + layer.weight * self._follow(env, code, layer, p)
                driven = True
        return out, driven

    def _entity_pos(self, env, entity_id):
        if not entity_id:
            return None
        ws = getattr(env, "ws", None)
        if ws is not None and hasattr(ws, "entities"):
            e = ws.entities.by_id(entity_id)
            if e is not None:
                return e.pos
        codes = list(getattr(env, "codes", []))
        if entity_id in codes:
            return np.asarray(env.pos[codes.index(entity_id)], dtype=float)
        return None

    def _target_state(self, env, target_id):
        """(pos, vel) of a robot code or entity id, or (None, None) if stale."""
        ws = getattr(env, "ws", None)
        if ws is not None and hasattr(ws, "entities"):
            e = ws.entities.by_id(target_id)
            if e is not None:
                return e.pos.copy(), e.vel.copy()
        codes = list(getattr(env, "codes", []))
        if target_id in codes:
            j = codes.index(target_id)
            return (np.asarray(env.pos[j], dtype=float),
                    np.asarray(env.vel[j], dtype=float))
        return None, None

    HOLD_TICKS = 6          # ~0.2s at 30fps before a condition may flip

    def _condition_holds(self, env, code, layer, p):
        """Is this layer's `when` condition true for this robot, right now?

        A layer without one always applies, which is the old behaviour exactly.
        With one, the layer becomes a standing *rule* rather than a standing
        action — "follow while it is within 80cm", "orbit only while the target
        is moving" — re-decided every tick instead of once when the command was
        given.

        Debounced, because a condition evaluated at 30Hz on noisy positions
        flickers, and a robot that starts and stops thirty times a second is
        worse than one doing the wrong thing steadily.
        """
        pred = layer.params.get("_when")
        if pred is None:
            return True

        codes = list(getattr(env, "codes", []))
        tpos, tvel = self._target_state(env, layer.params.get("target"))
        d_target = float(np.linalg.norm(np.asarray(tpos, float) - p)) \
            if tpos is not None else 0.0

        others = [env.pos[i] for i, c in enumerate(codes) if c != code]
        d_nearest = min((float(np.linalg.norm(np.asarray(o, float) - p))
                         for o in others), default=0.0)

        rank = 0
        if tpos is not None and codes:
            dists = sorted((float(np.linalg.norm(np.asarray(env.pos[i], float)
                                                 - np.asarray(tpos, float))), c)
                           for i, c in enumerate(codes))
            rank = next((k for k, (_, c) in enumerate(dists) if c == code), 0)

        i = codes.index(code) if code in codes else 0
        speed = float(np.linalg.norm(env.vel[i])) if i < len(env.vel) else 0.0
        ent = self._entity_pos(env, layer.params.get("entity")
                               or layer.params.get("target"))
        ex, ey = (float(ent[0]), float(ent[1])) if ent is not None else (0.0, 0.0)

        try:
            now = bool(pred(x=float(p[0]), y=float(p[1]), t=self.t, speed=speed,
                            d_target=d_target, d_nearest=d_nearest, rank=rank,
                            ex=ex, ey=ey, elapsed=layer.age))
        except Exception as e:
            # A condition that raises is not a condition. Drop the layer rather
            # than silently driving on a broken rule.
            layer.params["_when"] = None
            layer.state["when_error"] = f"{type(e).__name__}: {e}"
            return False

        held = layer.state.get("when_held", 0)
        active = layer.state.get("when_active", now)
        held = held + 1 if now != active else 0
        if held >= self.HOLD_TICKS:
            active, held = now, 0
        layer.state["when_held"] = held
        layer.state["when_active"] = active
        return active

    def _followers_now(self, env, layer, code, tpos):
        """Who is following, this tick.

        A fixed list is resolved at push time and never revisited, which cannot
        express "whichever robot is closest". A selector is resolved here
        instead — every candidate carries the layer, and the ones not currently
        chosen contribute nothing.

        Hysteresis is not optional. Two robots a centimetre apart would swap
        the job every tick, and a robot that starts and stops thirty times a
        second is worse than the wrong robot doing it steadily.
        """
        from . import fields

        select = layer.params.get("select")
        if not select:
            return list(layer.params.get("followers") or [code])

        candidates = [c for c in layer.params.get("followers") or []
                      if c in getattr(env, "codes", [])]
        if not candidates:
            return [code]

        want = 1
        if ":" in str(select):
            try:
                want = max(1, int(str(select).split(":", 1)[1]))
            except ValueError:
                want = 1
        want = min(want, len(candidates))

        codes = list(env.codes)
        dist = {c: float(np.linalg.norm(env.pos[codes.index(c)] - tpos))
                for c in candidates}
        # Incumbents keep the job unless beaten by a clear margin.
        holding = [c for c in layer.state.get("chosen", []) if c in dist]
        for c in holding:
            dist[c] -= fields.HYSTERESIS

        chosen = sorted(dist, key=lambda c: dist[c])[:want]
        layer.state["chosen"] = chosen
        return chosen

    def _follow(self, env, code, layer, p):
        from . import fields

        target_id = layer.params.get("target")
        tpos, tvel = self._target_state(env, target_id)
        if tpos is None:
            # Stale target: hold position rather than chase a guess. Same rule
            # the fleet applies to a robot the camera has lost.
            layer.state["stale"] = True
            return np.zeros(2)
        layer.state["stale"] = False

        followers = self._followers_now(env, layer, code, tpos)
        if code not in followers:
            return np.zeros(2)       # a candidate, but not currently chosen
        idx = followers.index(code)
        mode = layer.params.get("mode", "trail")
        distance = float(layer.params.get("distance", 40.0))
        lead = float(layer.params.get("lead_time", fields.LEAD_TIME))

        aim = fields.predict(tpos, tvel, lead)
        if mode == "mirror":
            v = np.asarray(tvel, dtype=float) / max(MAX_FOLLOW_SPEED, 1e-9)
            nrm = float(np.linalg.norm(v))
            return v / nrm if nrm > 1.0 else v

        phase = None
        if mode == "surround":
            codes = list(getattr(env, "codes", []))
            pts = [env.pos[codes.index(c)] for c in followers if c in codes]
            phase = fields.ring_phase_of(aim, pts, len(followers))
        slot = fields.follow_point(mode, idx, len(followers), aim, tvel,
                                   distance, p, ring_phase=phase)
        if slot is None:
            return np.zeros(2)
        return fields.seek(p, slot, self.slow_radius, self.arrive_tol)

    def _avoid(self, ws, p, seek):
        """Push away from obstacles and out of the walls, in normalised units.

        `blocking()` rather than `obstacles`: it is the single definition of
        what must be avoided, and it evaluates moving entities at their current
        position. Iterating the static list alone made moving obstacles
        invisible here, and a robot was driven straight through one.
        """
        f = np.zeros(2)

        for o in (ws.blocking() if hasattr(ws, "blocking") else ws.obstacles):
            if o.get("type") == "circle":
                c = np.asarray(o["center"], dtype=float)
                reach = float(o["radius"]) + OBSTACLE_MARGIN
                away = p - c
                d = float(np.linalg.norm(away))
                if d < reach:
                    unit_away = away / d if d > 1e-6 else np.array([1.0, 0.0])
                    f += _deflect(unit_away, (reach - d) / OBSTACLE_MARGIN,
                                  seek, p - c)
            elif o.get("type") == "poly":
                pts = np.asarray(o["points"], dtype=float)
                centre = pts.mean(axis=0)
                near, d = _nearest_on_poly(p, pts)
                if _in_poly(p, pts):
                    away = p - centre
                    nrm = float(np.linalg.norm(away))
                    f += (away / nrm if nrm > 1e-6 else np.array([1.0, 0.0])) * 2.0
                elif d < OBSTACLE_MARGIN:
                    away = p - near
                    nrm = float(np.linalg.norm(away))
                    away = away / nrm if nrm > 1e-6 else _unit(p - centre)
                    f += _deflect(away, (OBSTACLE_MARGIN - d) / OBSTACLE_MARGIN,
                                  seek, p - centre)

        xmin, xmax, ymin, ymax = ws.bbox
        margin = OBSTACLE_MARGIN
        for ax, (lo, hi) in enumerate([(xmin, xmax), (ymin, ymax)]):
            if p[ax] < lo + margin:
                f[ax] += (lo + margin - p[ax]) / margin
            elif p[ax] > hi - margin:
                f[ax] -= (p[ax] - (hi - margin)) / margin
        return f


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([1.0, 0.0])


def _deflect(away, strength, seek, from_centre):
    """Radial push plus a tangential slide around the obstacle.

    Radial repulsion alone has a local minimum: a robot aimed straight at an
    obstacle feels a push exactly opposite its seek force, the two cancel, and
    it parks in front of it forever.

    The side to pass on comes from how far off the obstacle's centreline the
    robot already is, never from the seek direction. Seek points back at the
    target, so using it flips the tangent every time the robot drifts off
    course and produces a permanent oscillation instead of a way around.
    Committing to the side it is already on is stable, and it is also the
    shorter way round.
    """
    radial = away * strength
    if float(np.linalg.norm(seek)) < 1e-6:
        return radial
    u = _unit(seek)

    head_on = max(0.0, -float(u @ away))       # 1 when driving straight at it
    if head_on <= 0.0:
        return radial

    lateral = from_centre - float(from_centre @ u) * u
    n = float(np.linalg.norm(lateral))
    tangent = lateral / n if n > 1e-3 else np.array([-away[1], away[0]])
    return radial + tangent * strength * head_on * 1.6


def _in_poly(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-9):
                inside = not inside
    return inside


def _nearest_on_poly(p, poly):
    best_d, best = np.inf, poly[0]
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(p - proj))
        if d < best_d:
            best_d, best = d, proj
    return best, best_d
