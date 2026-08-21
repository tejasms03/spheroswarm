"""Optional: sandboxed parametric generation.

Secondary to `move_to` with literal coordinates. It exists for the case where a
model wants twenty points on a curve and would rather write the curve than the
twenty points.
"""

from .movement import move_to
from .result import fail, ok
from .validate import MIN_SEPARATION, SANDBOX_TIMEOUT
from .validate import compute_points as _compute


def compute_points(ctx, expression, execute=False, timeout=SANDBOX_TIMEOUT,
                   min_separation=MIN_SEPARATION):
    """Evaluate a short expression that assigns a list of (x, y) to `points`.

    Available names: the arena size and centre, the robot count, and a small
    maths allowlist. No imports, no attribute access beyond maths and list
    building, no `while`, one second.
    """
    active = ctx.active_codes()
    if not active:
        return fail(ctx, "no robots are connected")

    xmin, xmax, ymin, ymax = ctx.ws.bbox
    centre = ctx.ws.centroid()
    variables = {
        "n": len(active),
        "width": ctx.ws.width,
        "height": ctx.ws.height,
        "cx": float(centre[0]),
        "cy": float(centre[1]),
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
    }

    report = _compute(expression, ctx.ws, expected_count=len(active),
                      variables=variables, timeout=timeout,
                      min_separation=min_separation)
    if not report["ok"]:
        return fail(ctx, report["error"], clamped=report["clamped"],
                    problems=report["problems"])

    if execute:
        return move_to(ctx, points=report["points"], min_separation=min_separation)

    return ok(ctx, points=report["points"], clamped=report["clamped"],
              note="pass execute=true to move the robots to these points")
