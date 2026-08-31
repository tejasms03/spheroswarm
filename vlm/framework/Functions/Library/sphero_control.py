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

import time

from rpc_system import RPCClient


client = RPCClient()


def exec_robot_create_thread(robot_id: int, robot_padding: int = 30) -> str:
    """Drive the path already generated for this robot.

    `robot_padding` is accepted and ignored: it is the radius, in pixels, at
    which their PID treats OTHER robots as obstacles, and this rig has one
    robot. Accepted rather than removed so an agent copying the call shape
    from `controller` does not fail on an unexpected argument.
    """
    return client.Robot.request_drive(int(robot_id))


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
    return client.Robot.request_orbit(int(robot_id), float(x), float(y),
                                      float(radius))
