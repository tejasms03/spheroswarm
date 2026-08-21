"""Formation memory as tools: save, recall, list, delete."""

import numpy as np

from .movement import current_arrangement
from .result import fail, ok
from .validate import MIN_SEPARATION, validate_targets

DEFAULT_RECALL_DURATION = 60.0      # s; a replayed motion is still bounded


def save_formation(ctx, name, description="", with_motion=True):
    """Snapshot the current arrangement — and whatever it is currently doing.

    Saving only positions makes the library a book of statues. A patrol, an
    orbit or a follow is just as much "a thing the swarm knows how to do", and
    is far harder for a person to describe again from scratch than a shape is.
    """
    codes, pts = current_arrangement(ctx)
    if not codes:
        return fail(ctx, "no robots are connected, so there is no arrangement to save")

    motion = None
    stack = getattr(ctx, "stack", None)
    if with_motion and stack is not None and stack.codes():
        from .formations import normalise
        _, centroid, radius = normalise(pts)
        motion = ctx.library.normalise_motion(stack, centroid, radius)

    errors = ctx.library.save_formation(name, pts, codes=codes,
                                        description=description, motion=motion)
    if errors:
        return fail(ctx, "; ".join(errors))

    saved = ctx.library.get(name)
    moving = sorted(saved.get("motion") or {})
    return ok(ctx, name=name, robots=saved["robot_count"], slots=saved["slots"],
              normalised_points=saved["points"],
              motion_saved=moving or None,
              note=(f"also saved the motion running on {', '.join(moving)}"
                    if moving else None))


def _restore_motion(ctx, saved, center, scale, rotation):
    """Replay whatever this memory was doing, in the frame it is recalled into.

    Waypoints were stored normalised, so they scale and rotate with the shape.
    A flow's expression is already parametric in `cx, cy, ex, ey` and rebinds
    itself. A follow whose target has since vanished is reported, not replayed
    — silently following nothing is worse than saying so.
    """
    motion = saved.get("motion")
    stack = getattr(ctx, "stack", None)
    if not motion or stack is None:
        return []

    from workspace.flow import compile_flow

    radius = float(scale) if scale else float(saved.get("saved_radius_cm") or 1.0)
    rad = np.radians(float(rotation or 0.0))
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    centre = np.asarray(center, dtype=float)
    active = set(ctx.active_codes())
    restored, skipped = [], []

    for code, layers in motion.items():
        if code not in active:
            continue
        for spec in layers:
            params = dict(spec.get("params") or {})
            kind = spec.get("kind")

            if kind == "path" and params.get("waypoints"):
                params["waypoints"] = [
                    (centre + rot @ (np.asarray(w, dtype=float) * radius)).tolist()
                    for w in params["waypoints"]]
            elif kind == "flow":
                c = ctx.ws.centroid()
                field, err = compile_flow(params.get("expr", ""), cx=float(c[0]),
                                          cy=float(c[1]), n=len(active))
                if err:
                    skipped.append(f"{code}/{spec.get('name')}: {err}")
                    continue
                params["field"] = field
            elif kind == "follow":
                target = params.get("target")
                known = (target in ctx.fleet.handles
                         or (hasattr(ctx.ws, "entities")
                             and ctx.ws.entities.by_id(target) is not None))
                if not known:
                    skipped.append(f"{code}/{spec.get('name')}: no such target "
                                   f"{target!r} any more")
                    continue
                if params.get("distance_norm") is not None:
                    params["distance"] = float(params.pop("distance_norm")) * radius

            _, errs = stack.push_layer(code, spec.get("name", kind), kind, params,
                                       weight=float(spec.get("weight", 1.0)),
                                       duration=DEFAULT_RECALL_DURATION)
            if errs:
                skipped.append(f"{code}: {'; '.join(errs)}")
            else:
                restored.append(code)

    if skipped:
        ctx.log.append(("motion not fully restored", skipped))
    return sorted(set(restored))


def list_formations(ctx):
    """What the model already knows how to make."""
    return ok(ctx, formations=ctx.library.listing())


def delete_formation(ctx, name):
    errors = ctx.library.delete(name)
    if errors:
        return fail(ctx, "; ".join(errors))
    return ok(ctx, deleted=name, formations=ctx.library.names())


def recall_formation(ctx, name, center=None, scale=None, rotation=0.0,
                     min_separation=MIN_SEPARATION):
    """Place a saved shape: denormalise, validate, execute."""
    active = ctx.active_codes()
    if not active:
        return fail(ctx, "no robots are connected, so there is nothing to place")

    if center is None:
        center = ctx.ws.centroid()
    else:
        try:
            center = np.array([float(center[0]), float(center[1])])
        except (TypeError, ValueError, IndexError, KeyError):
            return fail(ctx, f"center must be [x, y], got {center!r}")
        if not np.isfinite(center).all():
            return fail(ctx, "center must be two finite numbers")

    pts, errors = ctx.library.recall(name, len(active), center, scale=scale,
                                     rotation=rotation)
    if errors:
        return fail(ctx, "; ".join(errors))

    report = validate_targets(pts, ctx.ws, expected_count=len(active),
                              min_separation=min_separation)
    if not report["ok"]:
        return fail(ctx, f"formation {name!r} does not fit here: {report['error']}",
                    clamped=report["clamped"], problems=report["problems"])

    placed = np.asarray(report["points"], dtype=float).reshape(-1, 2)

    # Prefer the slot each robot held when the shape was saved; fall back to
    # nearest-slot assignment for any robot that was not part of it.
    saved = ctx.library.get(name)
    slots = [c for c in saved.get("slots", []) if c in active]
    mapping = {}
    used = set()
    if len(slots) == len(placed):
        for i, code in enumerate(saved["slots"]):
            if code in active:
                mapping[code] = placed[i]
                used.add(i)

    leftover_codes = [c for c in active if c not in mapping]
    leftover_points = [placed[i] for i in range(len(placed)) if i not in used]
    if leftover_codes:
        from swarm.navigate import assign as hungarian
        pos = np.array([ctx.fleet.handles[c].pos for c in leftover_codes],
                       dtype=float).reshape(-1, 2)
        order = hungarian(pos, np.asarray(leftover_points).reshape(-1, 2))
        for i, c in enumerate(leftover_codes):
            j = order[i]
            if 0 <= j < len(leftover_points):
                mapping[c] = leftover_points[j]

    ctx.apply_targets(mapping)
    restored = _restore_motion(ctx, saved, center, scale, rotation)
    return ok(ctx, name=name,
              targets={c: [round(float(p[0]), 1), round(float(p[1]), 1)]
                       for c, p in mapping.items()},
              clamped=report["clamped"])
