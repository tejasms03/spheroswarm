"""Movement. `move_to` is the only primitive; every arrangement is a point set.

There is deliberately no circle function, no line function, no shape enum. The
model is told the arena size and the robot count and computes coordinates
itself, so a shape nobody anticipated is no harder than a row.
"""

import math

import numpy as np

from swarm.navigate import assign as hungarian

from .result import fail, ok
from .validate import MIN_SEPARATION, validate_targets


def _coerce_mapping(assign_map):
    if assign_map is None:
        return {}, None
    if not isinstance(assign_map, dict):
        return {}, f"the code-to-point mapping must be an object, got {type(assign_map).__name__}"
    return dict(assign_map), None


def move_to(ctx, points=None, assign=None, min_separation=MIN_SEPARATION):
    """Place robots at a set of points.

    `points` is unordered — the system assigns robots to their nearest slots.
    `assign` pins specific robots: {"SSMK": [100, 80]}. Both together means
    "put these robots exactly here, and fit the rest into what is left".
    """
    active = ctx.active_codes()
    if not active:
        return fail(ctx, "no robots are connected, so there is nothing to move")

    pinned, err = _coerce_mapping(assign)
    if err:
        return fail(ctx, err)

    unknown = [c for c in pinned if c not in ctx.fleet.handles]
    if unknown:
        return fail(ctx, f"unknown robot code(s): {', '.join(sorted(unknown))} "
                         f"(fleet has {', '.join(ctx.fleet.codes)})")

    offline = [c for c in pinned if c not in active]
    if offline:
        return fail(ctx, f"robot(s) {', '.join(sorted(offline))} are not connected")

    pinned_codes = list(pinned)
    rest_codes = [c for c in active if c not in pinned]

    if points is None and pinned:
        # Pinning some robots and saying nothing about the others means "move
        # these, leave the rest alone". Holding their current positions keeps
        # every robot accounted for, so the count check still means something
        # and a pin that would drive into a bystander is still caught.
        free_points = [ctx.fleet.handles[c].pos.copy() for c in rest_codes]
    else:
        free_points = [] if points is None else list(points)

    combined = [pinned[c] for c in pinned_codes] + free_points

    report = validate_targets(combined, ctx.ws, expected_count=len(active),
                              min_separation=min_separation)
    if not report["ok"]:
        return fail(ctx, report["error"], clamped=report["clamped"],
                    problems=report["problems"])

    placed = np.asarray(report["points"], dtype=float).reshape(-1, 2)
    n_pinned = len(pinned_codes)
    mapping = {c: placed[i] for i, c in enumerate(pinned_codes)}

    rest_points = placed[n_pinned:]
    if len(rest_codes):
        rest_pos = np.array([ctx.fleet.handles[c].pos for c in rest_codes],
                            dtype=float).reshape(-1, 2)
        order = hungarian(rest_pos, rest_points)
        for i, c in enumerate(rest_codes):
            j = order[i]
            if 0 <= j < len(rest_points):
                mapping[c] = rest_points[j]

    ctx.apply_targets(mapping)
    return ok(ctx,
              targets={c: [round(float(p[0]), 1), round(float(p[1]), 1)]
                       for c, p in mapping.items()},
              clamped=report["clamped"],
              pinned=pinned_codes)


def current_arrangement(ctx):
    """Where the swarm is going, or where it is if it was given nothing."""
    codes = ctx.active_codes()
    pts = []
    for c in codes:
        t = ctx.env.targets.get(c)
        pts.append(np.asarray(t, dtype=float) if t is not None
                   else ctx.fleet.handles[c].pos.copy())
    return codes, np.asarray(pts, dtype=float).reshape(-1, 2)


def transform(ctx, translate=None, rotate=None, scale=None,
              min_separation=MIN_SEPARATION):
    """Move the whole arrangement: shift it, spin it, spread it out.

    Robots keep their slots — this moves the shape, it does not reassign it.
    """
    codes, pts = current_arrangement(ctx)
    if not codes:
        return fail(ctx, "no robots are connected, so there is no arrangement to transform")

    out = pts.copy()
    centre = out.mean(axis=0)

    if scale is not None:
        try:
            s = float(scale)
        except (TypeError, ValueError):
            return fail(ctx, f"scale must be a number, got {scale!r}")
        if not math.isfinite(s) or s <= 0:
            return fail(ctx, f"scale must be a positive finite number, got {scale!r}")
        out = (out - centre) * s + centre

    if rotate is not None:
        try:
            deg = float(rotate)
        except (TypeError, ValueError):
            return fail(ctx, f"rotation must be a number of degrees, got {rotate!r}")
        if not math.isfinite(deg):
            return fail(ctx, "rotation must be a finite number of degrees")
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        out = (out - centre) @ rot.T + centre

    if translate is not None:
        try:
            dx, dy = (float(translate[0]), float(translate[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            return fail(ctx, f"translate must be [dx, dy], got {translate!r}")
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return fail(ctx, "translate must be two finite numbers")
        out = out + np.array([dx, dy])

    if translate is None and rotate is None and scale is None:
        return fail(ctx, "transform needs at least one of translate, rotate or scale")

    report = validate_targets(out, ctx.ws, expected_count=len(codes),
                              min_separation=min_separation)
    if not report["ok"]:
        return fail(ctx, report["error"], clamped=report["clamped"],
                    problems=report["problems"])

    placed = np.asarray(report["points"], dtype=float).reshape(-1, 2)
    mapping = {c: placed[i] for i, c in enumerate(codes)}
    ctx.apply_targets(mapping)
    return ok(ctx,
              targets={c: [round(float(p[0]), 1), round(float(p[1]), 1)]
                       for c, p in mapping.items()},
              clamped=report["clamped"])


def stop(ctx, codes=None):
    """Halt some or all robots. Its own tool so it is always one call away."""
    if codes is None:
        ctx.stop_all()
        return ok(ctx, stopped=ctx.fleet.codes)

    if isinstance(codes, str):
        codes = [codes]
    unknown = [c for c in codes if c not in ctx.fleet.handles]
    if unknown:
        return fail(ctx, f"unknown robot code(s): {', '.join(sorted(unknown))}")

    ctx.fleet.stop(codes)
    for c in codes:
        # Layers first: a robot whose flow layer survived a stop resumes on the
        # very next tick, and "stop" has to mean stopped for the named robots
        # exactly as much as it does for all of them.
        ctx.stack.clear_layers(c)
        ctx.env.targets.pop(c, None)
        h = ctx.fleet.handles.get(c)
        if h is not None:
            h.target = None
    return ok(ctx, stopped=list(codes))
