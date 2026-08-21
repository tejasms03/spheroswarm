"""Motion over time: paths, flow fields, following, and managing what runs.

Everything here pushes a *named* layer onto the context's stack. The controller
blends them; these functions only validate and describe. Nothing raises — a bad
argument comes back as a readable error the model can act on, exactly like the
positional tools.
"""

import numpy as np

from swarm.layers import MAX_DURATION, clamp_duration
from workspace.flow import compile_flow

from .result import fail, ok
from .validate import validate_targets

FLOW_SIM_STEPS = 200        # how far ahead a flow field is simulated before it runs
FLOW_SIM_DT = 0.1
PINNED_TRAVEL_CM = 15.0     # movement over the last quarter below this = stuck
DEFAULT_FOLLOW_DISTANCE = 40.0
FOLLOW_MODES = ("trail", "surround", "flank", "mirror")
PATH_MODES = ("once", "loop", "pingpong")


def _codes(ctx, codes):
    """Resolve a requested robot list to connected codes, or an error string."""
    active = ctx.active_codes()
    if not active:
        return None, "no robots are connected"
    if codes is None:
        return list(active), None
    if isinstance(codes, str):
        codes = [codes]
    try:
        wanted = [str(c) for c in codes]
    except TypeError:
        return None, f"codes must be a list of robot codes, got {codes!r}"
    unknown = [c for c in wanted if c not in active]
    if unknown:
        return None, (f"unknown or disconnected robot(s): {', '.join(unknown)}; "
                      f"connected: {', '.join(active)}")
    return wanted, None


# -- set_path -----------------------------------------------------------------

def set_path(ctx, assignments=None, mode="once", speed=None, duration=None,
             append=False, weight=1.0):
    """Walk waypoints. `assignments` maps a robot code to a list of points."""
    if mode not in PATH_MODES:
        return fail(ctx, f"mode must be one of {list(PATH_MODES)}, got {mode!r}")
    if not isinstance(assignments, dict) or not assignments:
        return fail(ctx, 'assignments must be {"CODE": [[x, y], ...]}, '
                         f"got {type(assignments).__name__}")

    active = ctx.active_codes()
    unknown = [c for c in assignments if c not in active]
    if unknown:
        return fail(ctx, f"unknown or disconnected robot(s): {', '.join(unknown)}; "
                         f"connected: {', '.join(active)}")

    cleaned = {}
    for code, waypoints in assignments.items():
        report = validate_targets(waypoints, ctx.ws, min_separation=0.0)
        if not report["ok"]:
            return fail(ctx, f"{code}: {report['error']}", problems=report["problems"])
        if len(report["points"]) < 1:
            return fail(ctx, f"{code}: a path needs at least one waypoint")
        cleaned[code] = report["points"]

    seconds, clamped = clamp_duration(duration)
    pushed, errors = [], []
    for code, waypoints in cleaned.items():
        existing = ctx.stack.get(code, "path") if append else None
        if existing is not None:
            waypoints = list(existing.params.get("waypoints", [])) + waypoints

        layer, errs = ctx.stack.push_layer(
            code, "path", "path",
            {"waypoints": waypoints, "mode": mode, "speed": speed},
            weight=weight, duration=seconds)
        if errs:
            errors.extend(errs)
        else:
            pushed.append(code)

    if errors and not pushed:
        return fail(ctx, "; ".join(errors))

    return ok(ctx, codes=pushed, mode=mode, waypoints=cleaned,
              duration_s=seconds, duration_clamped=clamped,
              warnings=errors or None,
              note=(f"duration clamped to {seconds:.0f}s (max {MAX_DURATION:.0f})"
                    if clamped else None))


# -- set_flow -----------------------------------------------------------------

def set_flow(ctx, expr=None, robots=None, entity=None, weight=0.6, duration=None,
             name="flow", when=None):
    """Drive robots by a velocity field. Compiled and simulated before it runs."""
    codes, err = _codes(ctx, robots)
    if err:
        return fail(ctx, err)

    entity_pos = None
    if entity:
        entity_pos = _entity_position(ctx, entity)
        if entity_pos is None:
            return fail(ctx, f"no entity or robot called {entity!r}; "
                             f"known entities: {_entity_names(ctx) or 'none'}")

    centre = ctx.ws.centroid()
    field, cerr = compile_flow(expr, cx=float(centre[0]), cy=float(centre[1]),
                               n=len(codes))
    if cerr:
        return fail(ctx, cerr)

    # A syntactically perfect field can still drive everything into a wall, and
    # nothing about the source would tell you. Fly it before anyone rides it.
    escaped = _simulate_flow(ctx, field, codes, entity_pos)
    if escaped:
        return fail(ctx, "this field drives robot(s) into a wall and leaves "
                         f"them there: {', '.join(sorted(set(escaped)))}. Try a "
                         "field that circulates, like "
                         "vx = -(y-cy)*0.5 / vy = (x-cx)*0.5.")

    seconds, clamped = clamp_duration(duration)
    pushed, errors = [], []
    for code in codes:
        flow_params = {"field": field, "expr": expr, "entity": entity}
        werr = _compile_when(ctx, when, flow_params)
        if werr:
            return fail(ctx, werr)
        layer, errs = ctx.stack.push_layer(code, name, "flow", flow_params,
                                           weight=weight, duration=seconds)
        (errors.extend(errs) if errs else pushed.append(code))

    if errors and not pushed:
        return fail(ctx, "; ".join(errors))

    return ok(ctx, codes=pushed, expr=expr, entity=entity, weight=weight,
              duration_s=seconds, duration_clamped=clamped,
              warnings=errors or None,
              note=(f"duration clamped to {seconds:.0f}s" if clamped else None))


def _simulate_flow(ctx, field, codes, entity_pos):
    """Fly the field forward from each robot before anyone rides it.

    Two ways a field is unusable, and both have to be caught here because
    neither is visible in the source:

    * it leaves the arena — checked with wall repulsion applied, since the
      controller always applies it and a pre-flight without it condemns a plain
      rotation whose robots simply start further out than the arena is deep;
    * it pins robots against a wall — repulsion keeps them technically inside
      while they grind into the edge going nowhere, which is what "push
      everyone into the wall" actually looks like.
    """
    xmin, xmax, ymin, ymax = ctx.ws.bbox
    escaped = []
    ex, ey = (float(entity_pos[0]), float(entity_pos[1])) if entity_pos is not None \
        else (0.0, 0.0)

    for code in codes:
        h = ctx.fleet.handles.get(code)
        if h is None:
            continue
        p = np.asarray(h.pos, dtype=float).copy()
        t = 0.0
        track = [p.copy()]
        for _ in range(FLOW_SIM_STEPS):
            try:
                vx, vy = field(float(p[0]), float(p[1]), t, ex, ey)
            except Exception:
                escaped.append(code)
                break
            v = np.array([vx, vy], dtype=float)
            if not np.isfinite(v).all():
                escaped.append(code)
                break
            n = float(np.linalg.norm(v))
            if n > 1.0:
                v = v / n
            # The controller always applies wall repulsion after the blend, so
            # a pre-flight that leaves it out condemns fields that are fine in
            # practice — a plain rotation, for instance, whose robots start
            # further from the centre than the arena is deep.
            v = v + _wall_push(p, ctx.ws)
            n = float(np.linalg.norm(v))
            if n > 1.0:
                v = v / n
            p = p + v * 60.0 * FLOW_SIM_DT
            t += FLOW_SIM_DT
            track.append(p.copy())
            if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
                escaped.append(code)
                break
        else:
            # Survived the whole flight. Did it actually go anywhere near the
            # end, or is it just leaning on a wall?
            tail = np.asarray(track[-FLOW_SIM_STEPS // 4:], dtype=float)
            travelled = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum())
            if travelled < PINNED_TRAVEL_CM and _in_margin(p, ctx.ws):
                escaped.append(code)
    return escaped


def _in_margin(p, ws, margin=22.0):
    xmin, xmax, ymin, ymax = ws.bbox
    return (p[0] < xmin + margin or p[0] > xmax - margin
            or p[1] < ymin + margin or p[1] > ymax - margin)


def _wall_push(p, ws, margin=22.0, gain=1.4):
    """The same wall term the controller uses, for an honest pre-flight."""
    xmin, xmax, ymin, ymax = ws.bbox
    f = np.zeros(2)
    for ax, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax))):
        if p[ax] < lo + margin:
            f[ax] += (lo + margin - p[ax]) / margin
        elif p[ax] > hi - margin:
            f[ax] -= (p[ax] - (hi - margin)) / margin
    return f * gain


def _entity_position(ctx, entity_id):
    if hasattr(ctx.ws, "entities"):
        e = ctx.ws.entities.by_id(entity_id)
        if e is not None:
            return e.pos.copy()
    h = ctx.fleet.handles.get(entity_id)
    return np.asarray(h.pos, dtype=float) if h is not None else None


def _entity_names(ctx):
    if not hasattr(ctx.ws, "entities"):
        return []
    return [e.id for e in ctx.ws.entities]


def _compile_when(ctx, when, params):
    """Attach a compiled condition to a layer's params. Returns an error or None."""
    if not when:
        return None
    from workspace.flow import compile_predicate

    centre = ctx.ws.centroid()
    pred, err = compile_predicate(when, cx=float(centre[0]), cy=float(centre[1]),
                                  n=len(ctx.active_codes()))
    if err:
        return (f"condition {when!r} was rejected: {err}. A condition is one "
                "expression that is true or false, over x, y, t, speed, "
                "d_target, d_nearest, rank, elapsed, ex, ey, cx, cy, n — "
                "for example 'd_target < 80' or 'rank == 0'.")
    params["when"] = when
    params["_when"] = pred
    return None


# -- follow --------------------------------------------------------------------

def follow(ctx, target_id=None, followers=None, mode="trail",
           distance=DEFAULT_FOLLOW_DISTANCE, duration=None, weight=1.0,
           lead_time=None, select=None, when=None):
    """Chase a robot or a followable entity."""
    if mode not in FOLLOW_MODES:
        return fail(ctx, f"mode must be one of {list(FOLLOW_MODES)}, got {mode!r}")
    if not target_id:
        return fail(ctx, "follow needs a target_id — a robot code or an entity id")

    codes, err = _codes(ctx, followers)
    if err:
        return fail(ctx, err)

    target_is_robot = target_id in ctx.active_codes()
    entity = None
    if not target_is_robot and hasattr(ctx.ws, "entities"):
        entity = ctx.ws.entities.by_id(target_id)
        if entity is not None and not entity.followable:
            return fail(ctx, f"entity {target_id!r} has role {entity.role!r}; "
                             "only 'target' or 'both' entities can be followed")
    if not target_is_robot and entity is None:
        return fail(ctx, f"no robot or followable entity called {target_id!r}; "
                         f"entities: {_entity_names(ctx) or 'none'}")

    codes = [c for c in codes if c != target_id]
    if not codes:
        return fail(ctx, "a robot cannot follow itself")

    try:
        dist = float(distance)
    except (TypeError, ValueError):
        return fail(ctx, f"distance must be a number, got {distance!r}")
    if not np.isfinite(dist) or dist <= 0:
        return fail(ctx, "distance must be a positive number of centimetres")

    seconds, clamped = clamp_duration(duration)
    params = {"target": target_id, "mode": mode, "distance": dist,
              "followers": codes}
    err = _compile_when(ctx, when, params)
    if err:
        return fail(ctx, err)
    if select:
        # Membership is decided every tick from this list of candidates rather
        # than fixed now — "whichever is closest" is a standing rule, not a
        # choice made once at the moment the command was given.
        params["select"] = select
    if lead_time is not None:
        params["lead_time"] = lead_time

    pushed, errors = [], []
    for code in codes:
        layer, errs = ctx.stack.push_layer(code, "follow", "follow", params,
                                           weight=weight, duration=seconds)
        (errors.extend(errs) if errs else pushed.append(code))

    if errors and not pushed:
        return fail(ctx, "; ".join(errors))

    return ok(ctx, codes=pushed, target=target_id, mode=mode, distance=dist,
              duration_s=seconds, duration_clamped=clamped,
              warnings=errors or None,
              note=(f"duration clamped to {seconds:.0f}s" if clamped else None))


# -- motion_control -------------------------------------------------------------

def motion_control(ctx, action="list", codes=None, name=None, weight=None):
    """List, reweight, remove or clear layers."""
    if action == "list":
        active = {}
        for code in ctx.active_codes():
            layers = ctx.stack.active_layers(code)
            if layers:
                active[code] = layers
        return ok(ctx, layers=active, stalled=ctx.stack.stalled_codes(),
                  note="no motion layers are running" if not active else None)

    wanted, err = _codes(ctx, codes)
    if err:
        return fail(ctx, err)

    if action == "clear":
        for code in wanted:
            ctx.stack.clear_layers(code)
        return ok(ctx, cleared=wanted)

    if action in ("remove", "set_weight") and not name:
        return fail(ctx, f"{action} needs the layer `name` to act on; "
                         "call motion_control with action='list' to see them")

    if action == "remove":
        removed, errors = [], []
        for code in wanted:
            errs = ctx.stack.pop_layer(code, name)
            (errors.extend(errs) if errs else removed.append(code))
        if not removed:
            return fail(ctx, "; ".join(errors) or f"no layer {name!r} anywhere")
        return ok(ctx, removed=removed, name=name, warnings=errors or None)

    if action == "set_weight":
        if weight is None:
            return fail(ctx, "set_weight needs a `weight`")
        changed, errors = [], []
        for code in wanted:
            errs = ctx.stack.set_weight(code, name, weight)
            (errors.extend(errs) if errs else changed.append(code))
        if not changed:
            return fail(ctx, "; ".join(errors))
        return ok(ctx, changed=changed, name=name, weight=weight,
                  warnings=errors or None)

    return fail(ctx, f"unknown action {action!r}; expected list, set_weight, "
                     "remove or clear")
