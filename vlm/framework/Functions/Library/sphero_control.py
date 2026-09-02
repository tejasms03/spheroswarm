"""Drive a Sphero, in place of `controller.py`.

Same two entry points the agents already call -- `exec_robot_create_thread`
and `stop_robot_thread` -- so an agent's `functions.json` only has to change
which library it names.

WHAT IS DIFFERENT, and why it has to be:

`controller.py` starts a PID thread here, in whatever process ran the agent,
and that thread drives the robot over serial. A Sphero is on BLE and is owned
by the tracking bench, in the bench's process, because the camera loop that
closes the loop around it lives there and two owners of one BLE link is not a
thing that can be made to work. So nothing here spawns a controller. These
functions post a request, the bench picks it up and drives with its own pure
pursuit, and the bench writes back how it ended.

That also means these return IMMEDIATELY. `wait_for_robot` is here because the
alternative is an agent polling `get_robot_state` in a loop and burning its
call budget on round trips -- measured on the bench's own tool layer, four of
six calls went to polling before the drive had finished.
"""

import math
import time

from rpc_system import RPCClient


client = RPCClient()


def _confirm(robot_id: int, posted: str, timeout_s: float = 4.0) -> str:
    """Wait for the BENCH to accept or refuse, then say which.

    Every drive tool here posts a request to a blackboard and returns at once,
    so what it returns is a receipt, not an outcome. The bench picks the
    request up on its next tick and can still refuse it -- a mirrored frame, no
    tracker lock, a curve that leaves the arena, no aim established. Returning
    the receipt reported a robot "looping continuously" while it sat still and
    nothing had been drawn, which is the same lie as reporting `arrived` for a
    drive that merely stopped.

    So: poll until the bench marks the robot moving, or writes a verdict, or
    the wait runs out. A few seconds at twenty ticks a second is generous.
    """
    rid = int(robot_id)
    until = time.time() + max(0.5, float(timeout_s))
    while time.time() < until:
        outcome = client.Robot.get_outcome(rid)
        if outcome:
            return (f"REFUSED — {outcome.get('reason') or outcome.get('outcome')}"
                    if outcome.get("outcome") == "refused"
                    else f"{outcome.get('outcome')}: {outcome.get('reason', '')}")
        if client.Robot.get_state(rid) == "moving":
            return posted
        time.sleep(0.1)
    return (f"{posted}\n\nBUT THE BENCH HAS NOT PICKED IT UP after "
            f"{timeout_s:.0f}s — it may not be running with --rpc. Nothing is "
            f"moving; do not report this as started.")


def exec_robot_create_thread(robot_id: int, robot_padding: int = 30) -> str:
    """Drive the path already generated for this robot.

    `robot_padding` is accepted and ignored: it is the radius, in pixels, at
    which their PID treats OTHER robots as obstacles, and this rig has one
    robot. Accepted rather than removed so an agent copying the call shape
    from `controller` does not fail on an unexpected argument.
    """
    said = client.Robot.request_drive(int(robot_id))
    if 'No robot is attached' in said or "doesn't exist" in said \
            or 'Generate path first' in said or 'needs at least' in said \
            or 'required' in said:
        return said
    return _confirm(robot_id, said)


def stop_robot_thread(robot_id: int, join_timeout: float = 5.0) -> str:
    """Stop the robot. `join_timeout` is accepted and ignored -- there is no
    thread here to join; the bench disarms on its next tick."""
    return client.Robot.request_stop(int(robot_id))


def get_robot_state(robot_id: int) -> str:
    """'moving' or 'halt'."""
    return client.Robot.get_state(int(robot_id))


def wait_for_robot(robot_id: int, timeout_s: float = 30.0) -> dict:
    """Block until the drive ends, then say HOW it ended.

    The distinction this exists to carry: a robot stops for arriving, for
    giving up stuck, for the tracker losing the ball, and for a person pressing
    escape. Reporting all four as success is how an agent announces it has
    finished over a ball twenty centimetres short of the goal.

    Returns `{"arrived": bool, "outcome": str, "reason": str, "timed_out": bool}`.
    """
    rid = int(robot_id)
    limit = max(0.5, min(float(timeout_s), 120.0))
    until = time.time() + limit

    while time.time() < until:
        outcome = client.Robot.get_outcome(rid)
        if outcome:
            return {"arrived": outcome.get("outcome") == "arrived",
                    "outcome": outcome.get("outcome") or "unknown",
                    "reason": outcome.get("reason") or "",
                    "timed_out": False}
        time.sleep(0.2)

    return {"arrived": False, "outcome": "still driving", "timed_out": True,
            "reason": f"no verdict within {limit:.0f}s — it may still be going; "
                      f"call stop_robot_thread to end it"}


def get_robot_position(robot_id: int) -> dict:
    """Where the tracker last SAW it, in arena pixels.

    Absent rather than stale when the ball is not tracked: the bench publishes
    nothing while its tracker is unlocked, so a missing robot here means "not
    seen", never "has not moved".
    """
    pose = client.Robot.get_all_robot_pose().get(int(robot_id))
    if not pose:
        return {"tracked": False,
                "reason": "the tracker cannot see this robot right now"}
    return {"tracked": True, "x": pose.get("x"), "y": pose.get("y"),
            "theta": pose.get("theta"),
            # The bearing is TRAVEL, not facing, and it is held while the ball
            # is at rest. Said here so a caller does not read a stale heading
            # as a measured one.
            "theta_is_measured": bool(pose.get("theta_fresh", False)),
            "theta_age_s": pose.get("theta_age_s", 0.0)}


def orbit(robot_id: int, x: float, y: float, radius: float) -> str:
    """Drive a real CIRCLE around a point, repeating until stopped.

    Use this instead of trace_targets whenever the ask is to orbit, circle,
    go round, or patrol a loop. `trace_targets` can only express a list of
    waypoints, so a circle asked for through it becomes a polygon: it has
    corners the lookahead cuts, and it ENDS rather than repeating.

    This never arrives. Call stop_robot_thread to end it.
    """
    said = client.Robot.request_orbit(int(robot_id), float(x), float(y),
                                      float(radius))
    if 'No robot is attached' in said or "doesn't exist" in said \
            or 'Generate path first' in said or 'needs at least' in said \
            or 'required' in said:
        return said
    return _confirm(robot_id, said)


def patrol(robot_id: int, x1: float, y1: float, x2: float, y2: float) -> str:
    """Patrol back and forth between two points, until stopped.

    Use this for any "patrol", "back and forth", "sweep", or "guard the edge"
    request. Do NOT build one out of waypoints with trace_targets: an
    out-and-back route lays both legs on the same line, the follower cannot
    tell them apart, and the ball reverses early and oscillates around one end
    instead of covering the run.

    This never arrives. Call stop_robot_thread to end it.
    """
    said = client.Robot.request_patrol(int(robot_id), float(x1), float(y1),
                                       float(x2), float(y2))
    if 'No robot is attached' in said or "doesn't exist" in said \
            or 'Generate path first' in said or 'needs at least' in said \
            or 'required' in said:
        return said
    return _confirm(robot_id, said)


def set_trajectory(robot_id: int, points: list) -> str:
    """Drive an explicit list of [x, y] waypoints, in order, with no planner.

    `trace_targets` runs A* and may move a goal it judges unreachable; this
    drives exactly the shape given. Use it when the shape itself matters.
    """
    said = client.Robot.request_trajectory(int(robot_id), points)
    if 'No robot is attached' in said or "doesn't exist" in said \
            or 'Generate path first' in said or 'needs at least' in said \
            or 'required' in said:
        return said
    return _confirm(robot_id, said)


def follow(robot_id: int, x: float, y: float) -> str:
    """Chase a point, and keep chasing it as you move it.

    Call again with a new x, y to move the target under a ball already on its
    way -- that updates the goal rather than starting a fresh drive.

    There is nothing on this rig for it to lock onto by itself: object
    detection needs SAM2 and a lit room while the ball tracker needs the room
    dark, and there is only one robot, so there is no second one to chase. You
    supply the target.
    """
    return client.Robot.set_follow_target(int(robot_id), float(x), float(y))


def set_flow(robot_id: int, expression: str, closed: bool = True) -> str:
    """Drive a curve you DESCRIBE, instead of listing its points.

    For shapes a waypoint list says badly: a figure eight, a spiral, a
    lissajous, a rose. Write a short Python expression that assigns a list of
    (x, y) to `points`, in arena pixels.

    Available names: cx, cy (arena centre), width, height, xmin, xmax, ymin,
    ymax, and a small maths allowlist including sin, cos, pi, sqrt. No
    imports, no attribute access, no `while`, one second.

    Figure eight around the centre:
      points = [(cx + 400*sin(2*pi*i/64), cy + 200*sin(4*pi*i/64))
                for i in range(64)]

    Spiral outwards:
      points = [(cx + 4*i*cos(i/6), cy + 4*i*sin(i/6)) for i in range(80)]

    With closed=True (the default) it repeats until stopped; call
    stop_robot_thread to end it. With closed=False it drives the curve once
    and arrives.
    """
    said = client.Robot.request_flow(int(robot_id), str(expression),
                                     bool(closed))
    if 'No robot is attached' in said or "doesn't exist" in said \
            or 'Generate path first' in said or 'needs at least' in said \
            or 'required' in said:
        return said
    return _confirm(robot_id, said)


def get_arena(robot_id: int = 2) -> dict:
    """The arena's real size, centre and safe bounds, in arena pixels.

    CALL THIS BEFORE PLACING ANY COORDINATE. The arena is four corners
    somebody clicked, mapped through a camera calibration, and it changes
    whenever either is redone — so any figure written into a prompt goes
    stale silently. It has already: a re-click moved the centre from
    (694, 554) to (712, 613) and an agent kept orbiting the old one.

    Returns width, height, centre, px_per_cm, width_cm, height_cm, and
    `safe` as [x_min, y_min, x_max, y_max] — keep every point inside `safe`
    or the drive is refused.
    """
    got = client.Robot.get_arena()
    if not got:
        return {"known": False,
                "reason": "the bench has not reported the arena yet — it is "
                          "not running, or not started with --rpc"}
    got["known"] = True
    return got


COMPASS = {"n": 0.0, "north": 0.0, "ne": 45.0, "northeast": 45.0,
           "e": 90.0, "east": 90.0, "se": 135.0, "southeast": 135.0,
           "s": 180.0, "south": 180.0, "sw": 225.0, "southwest": 225.0,
           "w": 270.0, "west": 270.0, "nw": 315.0, "northwest": 315.0}


def _bearing(direction):
    """A compass name or a number of degrees, as degrees clockwise from north.

    NORTH IS THE TOP OF THE ARENA, and nothing here knows or cares which way
    the Sphero's own compass points. These tools work out a POSITION; the
    bench's controller drives to it. Keeping the ball's convention out of this
    file is deliberate — the bench's own note says a second place that knows
    that convention is a second place to get it wrong, and a sign error there
    is the one that costs a fortnight because it circles whatever you correct.
    """
    if isinstance(direction, (int, float)):
        return float(direction) % 360.0
    key = str(direction).strip().lower().replace(" ", "").replace("-", "")
    if key in COMPASS:
        return COMPASS[key]
    try:
        return float(key) % 360.0
    except ValueError:
        return None


def move_to_cm(robot_id: int, x_cm: float, y_cm: float) -> str:
    """Drive to a position given in CENTIMETRES from the arena's top-left.

    x runs right, y runs DOWN — the same way the camera sees it.
    """
    arena = get_arena(robot_id)
    if not arena.get("known"):
        return arena.get("reason", "the arena is not known yet")
    k = float(arena["px_per_cm"])
    return _goto_px(robot_id, float(x_cm) * k, float(y_cm) * k,
                    f"({x_cm:.0f}, {y_cm:.0f}) cm")


def move_by_cm(robot_id: int, distance_cm: float, direction) -> str:
    """Move a distance in a COMPASS direction: "north", "NE", or degrees.

    North is the top of the arena, east is the right, and degrees run
    clockwise from north. This is an absolute direction in the room, not
    relative to the way the robot happens to be facing — a Sphero is a sphere
    and has no visible facing, so "forward" is not a direction anything here
    can resolve.
    """
    bearing = _bearing(direction)
    if bearing is None:
        return (f"{direction!r} is not a direction. Use north, NE, east, ... "
                f"or degrees clockwise from north.")
    where = get_robot_position(robot_id)
    if not where.get("tracked"):
        return f"cannot move relative to a position nobody can see: {where.get('reason')}"
    arena = get_arena(robot_id)
    if not arena.get("known"):
        return arena.get("reason", "the arena is not known yet")

    k = float(arena["px_per_cm"])
    step = float(distance_cm) * k
    rad = math.radians(bearing)
    # North is -y because the arena frame runs y DOWN, matching the camera.
    x = float(where["x"]) + step * math.sin(rad)
    y = float(where["y"]) - step * math.cos(rad)
    return _goto_px(robot_id, x, y,
                    f"{distance_cm:.0f} cm {str(direction).upper()}")


def _goto_px(robot_id: int, x_px: float, y_px: float, said: str) -> str:
    """Drive to one point in arena pixels, and confirm the bench took it."""
    rid = int(robot_id)
    arena = get_arena(rid)
    if arena.get("known"):
        x0, y0, x1, y1 = arena["safe"]
        if not (x0 <= x_px <= x1 and y0 <= y_px <= y1):
            return (f"{said} is ({x_px:.0f}, {y_px:.0f}) px, outside the safe "
                    f"box {arena['safe']}. The ball has width and reads its own "
                    f"position badly near a wall — pick somewhere further in.")
    client.Robot.set_path(rid, [[float(x_px), float(y_px), 0.0, 0]])
    posted = client.Robot.request_drive(rid)
    if "Started controller" not in posted:
        return posted
    return _confirm(rid, f"driving to {said}")
