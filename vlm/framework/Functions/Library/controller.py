"""Their controller's entry points, pointed at the Sphero.

REPLACES the original, which spawned a PID thread here and drove a
differential-drive robot over serial through `PathControl.pp`. On this rig that
import chain does not apply and the robot is a rolling ball on BLE owned by the
tracking bench, so calling it raised inside the RPC worker, FastAPI turned that
into an HTML 500, and the React frontend died parsing it:

    Unexpected token 'I', "Internal S"... is not valid JSON

which tells you nothing.

Every agent in `Agents/` except `Sphero Home` names `controller` in its
`functions.json` -- Color Sorting, Go-HOME, Point Tracking, Line Following and
the rest -- so without this only one agent could move anything. Delegating here
means they all work, and none of their JSON needs editing.

The real implementation is `sphero_control`, beside this file. Read that one
for what actually happens; this exists so the old name keeps working.
"""

try:
    # How the agent dispatcher loads it: importlib on "Functions.Library.<lib>".
    from Functions.Library.sphero_control import (  # noqa: F401
        exec_robot_create_thread, stop_robot_thread, get_robot_state,
        wait_for_robot, get_robot_position, client)
except ImportError:
    # And how it resolves when this directory is itself on sys.path.
    from sphero_control import (  # noqa: F401
        exec_robot_create_thread, stop_robot_thread, get_robot_state,
        wait_for_robot, get_robot_position, client)


def run_controller(stop_event=None, robot_id=None, robot_padding=30):
    """Kept for signature compatibility. There is no thread to run.

    The bench owns the control loop, because it owns the camera that closes it.
    """
    return exec_robot_create_thread(robot_id, robot_padding)


def stop_all_robot_threads(join_timeout: float = 5.0):
    """Stop every robot. `backend/server.py:398` imports this by name."""
    return client.Robot.stop_all()
