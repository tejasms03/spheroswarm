"""Waiting. Without it, a sequence fires into a moving swarm and falls apart."""

import time

import numpy as np

from .result import fail, ok

DEFAULT_TOLERANCE = 8.0     # cm
DEFAULT_TIMEOUT = 20.0      # s


def wait_until_settled(ctx, tolerance=DEFAULT_TOLERANCE, timeout=DEFAULT_TIMEOUT,
                       dt=0.1, sleep=None):
    """Block until every robot is within `tolerance` of its target, or give up.

    `sleep` defaults to real time for a fleet with real robots in it and to
    free-running for an all-sim fleet, so tests and dry runs are not paced by
    a wall clock that nothing is waiting on.
    """
    try:
        tolerance = float(tolerance)
        timeout = float(timeout)
    except (TypeError, ValueError):
        return fail(ctx, f"tolerance and timeout must be numbers, got "
                         f"{tolerance!r}, {timeout!r}")
    if tolerance <= 0 or timeout <= 0:
        return fail(ctx, "tolerance and timeout must be positive")

    if sleep is None:
        sleep = (getattr(ctx, "realtime", False)
                 or any(h.kind == "real" for h in ctx.fleet.handles.values()))

    codes = ctx.active_codes()
    if not codes:
        return ok(ctx, settled=True, waited_s=0.0, remaining={},
                  note="no robots are connected")

    # A looping path, a flow and a follow have no settled state to reach, so
    # waiting for one burns the whole timeout and then reports failure. Saying
    # so up front is both faster and truthful — and it tells the model the one
    # thing it needs to know to get unstuck.
    stack = getattr(ctx, "stack", None)
    if stack is not None and stack.has_continuous():
        running = sorted({l["name"] for c in stack.codes()
                          for l in stack.active_layers(c)})
        return fail(ctx, "cannot wait for a settled state: continuous motion is "
                         f"running ({', '.join(running)}). Continuous layers "
                         "never settle — either let the duration expire, remove "
                         "the layer with motion_control, or call stop.")
    if not ctx.env.targets:
        return ok(ctx, settled=True, waited_s=0.0, remaining={},
                  note="no targets are set, so nothing to wait for")

    start = time.time()
    elapsed = 0.0
    steps = 0
    max_steps = max(1, int(timeout / dt))

    while steps < max_steps:
        ctx.tick(dt)
        steps += 1
        elapsed = (time.time() - start) if sleep else steps * dt
        if sleep:
            time.sleep(dt)

        remaining = _distances(ctx)
        if remaining and max(remaining.values()) <= tolerance:
            return ok(ctx, settled=True, waited_s=round(elapsed, 2),
                      remaining={c: round(d, 1) for c, d in remaining.items()})
        if elapsed >= timeout:
            break

    remaining = _distances(ctx)
    worst = max(remaining.values()) if remaining else 0.0
    return ok(ctx, settled=False, waited_s=round(elapsed, 2),
              remaining={c: round(d, 1) for c, d in remaining.items()},
              note=f"timed out after {elapsed:.1f}s with the furthest robot "
                   f"{worst:.1f}cm from its target (tolerance {tolerance:.0f}cm)")


def _distances(ctx):
    out = {}
    for code, target in ctx.env.targets.items():
        h = ctx.fleet.handles.get(code)
        if h is None or not h.connected:
            continue
        out[code] = float(np.linalg.norm(h.pos - np.asarray(target, dtype=float)))
    return out
