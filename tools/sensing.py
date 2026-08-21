"""The model's only window into the world."""

import numpy as np

from .result import ok


def _entity_state(ctx):
    """Every entity at its current place, with the role that decides what it is."""
    out = []
    for e in getattr(ctx.ws, "entities", []):
        d = {"id": e.id, "role": e.role,
             "pos": [round(float(e.pos[0]), 1), round(float(e.pos[1]), 1)],
             "vel": [round(float(e.vel[0]), 1), round(float(e.vel[1]), 1)],
             "shape": e.shape_type, "radius": round(float(e.radius), 1),
             "moving": e.motion.get("kind", "static") != "static",
             "followable": e.followable}
        if e.error:
            d["error"] = e.error
        out.append(d)
    return out


def get_state(ctx):
    """Every robot, plus the arena it lives in and what shapes are already known."""
    stack = getattr(ctx, "stack", None)
    robots = []
    for code, d in ctx.fleet.state().items():
        # Layers and stall go in the per-robot dict rather than a parallel
        # structure: the model reads one object per robot, and anything it has
        # to cross-reference by hand it will eventually cross-reference wrong.
        if stack is not None:
            d["layers"] = stack.active_layers(code)
            d["stalled"] = stack.stalled(code)
            why = stack.stall_reason(code)
            if why:
                d["stall_reason"] = {"asked_for_cms": why[0], "achieving_cms": why[1]}
        robots.append(d)

    return ok(ctx, robots=robots, arena={
        "bounds_cm": [list(p) for p in ctx.ws.bounds_cm],
        "width_cm": round(ctx.ws.width, 1),
        "height_cm": round(ctx.ws.height, 1),
        "origin": ctx.ws.origin,
        "obstacles": ctx.ws.obstacles,
    }, entities=_entity_state(ctx), formations=ctx.library.listing())


def _corner_word(ws, p):
    xmin, xmax, ymin, ymax = ws.bbox
    fx = (p[0] - xmin) / max(xmax - xmin, 1e-6)
    fy = (p[1] - ymin) / max(ymax - ymin, 1e-6)
    vert = "top" if fy < 0.33 else ("bottom" if fy > 0.67 else "middle")
    horz = "left" if fx < 0.33 else ("right" if fx > 0.67 else "centre")
    if vert == "middle" and horz == "centre":
        return "the centre"
    return f"{vert}-{horz}"


def describe_scene(ctx):
    """The same facts as prose. Models reason better over sentences than arrays,
    and it costs a fraction of the tokens."""
    ws = ctx.ws
    lines = [f"The arena is {ws.width:.0f}cm wide and {ws.height:.0f}cm tall, "
             f"origin top-left, x to the right, y downward."]

    if ws.obstacles:
        parts = []
        for o in ws.obstacles:
            if o.get("type") == "circle":
                c, r = o["center"], o["radius"]
                parts.append(f"a circle of radius {r:.0f}cm at ({c[0]:.0f}, {c[1]:.0f})")
            else:
                pts = np.asarray(o["points"], dtype=float)
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                parts.append(f"a block from ({lo[0]:.0f}, {lo[1]:.0f}) to "
                             f"({hi[0]:.0f}, {hi[1]:.0f})")
        lines.append("Obstacles to avoid: " + "; ".join(parts) + ".")
    else:
        lines.append("There are no obstacles.")

    states = ctx.fleet.state()
    if not states:
        lines.append("There are no robots in the fleet.")
    else:
        active, idle = [], []
        for code, d in states.items():
            p = d["pos"]
            where = _corner_word(ws, p)
            phrase = (f"{code} ({d['name']}, {d['color']}) at "
                      f"({p[0]:.0f}, {p[1]:.0f}) in {where}")
            if not d["connected"]:
                idle.append(f"{phrase} — disconnected")
            elif d["speed"] > 3.0:
                active.append(f"{phrase}, moving at {d['speed']:.0f}cm/s")
            else:
                active.append(f"{phrase}, stationary")
        lines.append(f"{len(states)} robot(s): " + "; ".join(active + idle) + ".")

        placeable = ctx.active_codes()
        lines.append(f"{len(placeable)} can be commanded: "
                     f"{', '.join(placeable) if placeable else 'none'}.")

    lines += _entity_prose(ctx)
    lines += _motion_prose(ctx)

    names = ctx.library.names()
    lines.append("Saved formations: " + (", ".join(names) if names else "none yet") + ".")

    return ok(ctx, description=" ".join(lines))


def _entity_prose(ctx):
    ents = list(getattr(ctx.ws, "entities", []))
    if not ents:
        return []
    parts = []
    for e in ents:
        moving = "moving" if e.motion.get("kind", "static") != "static" else "still"
        what = {"obstacle": "to avoid", "target": "followable",
                "both": "followable and to avoid"}.get(e.role, e.role)
        parts.append(f"{e.id} ({what}, {moving}) at "
                     f"({e.pos[0]:.0f}, {e.pos[1]:.0f})")
    return [f"{len(ents)} entit{'y' if len(ents) == 1 else 'ies'}: "
            + "; ".join(parts) + "."]


def _motion_prose(ctx):
    """What is actually running, as a sentence.

    A list of layer dicts is a thing to decode; "four dragons orbiting the
    wanderer, SSMK trailing behind" is a thing to reason about, and reasoning
    is what the next turn needs.
    """
    stack = getattr(ctx, "stack", None)
    if stack is None or not stack.codes():
        return ["Nothing is moving under a continuous motion layer."]

    # Group by what the layer is, so six robots orbiting reads as one clause.
    groups = {}
    for code in stack.codes():
        for layer in stack.layers(code):
            if layer.kind == "seek":
                continue
            key = (layer.kind, layer.name, _layer_object(layer))
            groups.setdefault(key, []).append(code)

    if not groups:
        return ["Nothing is moving under a continuous motion layer."]

    verbs = {"flow": "driven by a flow field", "path": "walking a path",
             "follow": "following"}
    parts = []
    for (kind, name, obj), codes in groups.items():
        who = ", ".join(codes) if len(codes) < len(stack.codes()) else "all of them"
        phrase = verbs.get(kind, kind)
        if obj:
            phrase += f" {obj}"
        parts.append(f"{who} {phrase} (layer {name!r})")

    stalled = [c for c in stack.codes() if stack.stalled(c)]
    out = ["Active motion: " + "; ".join(parts) + "."]
    if stalled:
        bits = []
        for c in stalled:
            why = stack.stall_reason(c)
            bits.append(f"{c} (asked for {why[0]:.0f}cm/s, achieving {why[1]:.0f})"
                        if why else c)
        out.append("Stalled — being driven but going nowhere, so something is "
                   "cancelling them: " + "; ".join(bits) + ".")
    return out


def _layer_object(layer):
    """What a layer is aimed at, when it is aimed at something nameable."""
    p = layer.params or {}
    if layer.kind == "follow":
        mode = p.get("mode", "trail")
        target = p.get("target_id") or p.get("target") or "something"
        return f"{target} ({mode})"
    if layer.kind == "flow" and p.get("entity"):
        return f"about {p['entity']}"
    return ""
