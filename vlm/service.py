"""A RobotService the framework can talk to, backed by one Sphero.

Registers in place of `Functions/Utilities/services/robot_service.py`. Their
library code reaches it as `client.Robot.*`, so it MUST be registered under the
name `Robot` -- `RPCServer.register_class` names a service after the class
unless told otherwise, and `register_class(SpheroRobot(), class_name="Robot")`
is what makes `Functions/Library/planning.py` find it.

WHAT THIS IS NOT: a driver. It never opens a BLE link and never commands a
robot. The Sphero is owned by the bench, in the bench's process, where the
tracker that closes the loop around it also lives -- two owners of one BLE link
is not a thing that can be made to work. So this is a BLACKBOARD: their planner
writes a path and a request to drive, the bench reads them and does the
driving, and the bench writes back what happened.

That split is also why there is no `_move_robot_loop` here. Theirs sends every
25ms, which is a sane rate down a serial cable and roughly four times what a
Sphero's BLE link will accept; `fleet/real_handle.py` already paces writes and
has to stay the only thing that does.
"""

import threading
import time


HALT = "halt"
MOVING = "moving"


class Refused(Exception):
    """A command this rig cannot honestly carry out."""


class SpheroRobot:
    """Their RobotService surface, over one ball.

    The methods here are the ones `Functions/Library/` and `backend/` actually
    call, counted from the source rather than guessed: `get_all_robot_pose`,
    `get_all_path`, `set_path`, `get_path`, `path_list`, `get_state`,
    `set_state`, `stop_all`, `resume`, `set_command`, `get_name_from_id`,
    `get_id_from_name`, `set_name`.
    """

    def __init__(self, id_list=(2,), names=None, codes=None):
        self.id_list = [int(i) for i in id_list]
        self.names = list(names) if names else [f"Robot-{i}" for i in self.id_list]
        self.codes = dict(codes) if codes else {}
        self.server = None

        self._lock = threading.RLock()
        self._paths = {}
        self._states = {i: HALT for i in self.id_list}
        self._courses = {}
        self._requests = {}
        self._outcomes = {}
        self._arena = {}

        # Set by the bench when it attaches. Until then every drive request is
        # refused rather than queued: a path accepted by a service with nothing
        # behind it looks exactly like a path that is about to be driven, and
        # an agent told "ok" will sit waiting for a ball that was never asked
        # to move.
        self.attached_at = None

        # Not part of their schema. `pp.py` reads `tip_data_sim` off the
        # service in ten places for its simulator; present and None so an
        # attribute read does not explode before we have replaced that path.
        self.tip_data_sim = None

    # -- their dependency-injection hook ------------------------------------

    def _set_server_reference(self, server_instance):
        self.server = server_instance

    # -- what the bench does, and whether anything is listening -------------

    def attach(self):
        """Called by the bench to say a robot is actually behind this."""
        with self._lock:
            self.attached_at = time.time()
        return True

    def detach(self):
        with self._lock:
            self.attached_at = None
            for rid in self.id_list:
                self._states[rid] = HALT
        return True

    @property
    def attached(self):
        return self.attached_at is not None

    # -- poses, read through from DataService --------------------------------

    def get_all_robot_pose(self):
        """Whatever the bridge last published. NEVER a remembered pose.

        Theirs returns `self.server.Data.robot_poses.copy()` and so does this.
        The important part is what is upstream: `vlm.bridge` publishes an empty
        dict when the tracker is not locked, so a robot that cannot be seen is
        ABSENT here rather than present at a stale position.
        """
        if self.server is not None and hasattr(self.server, "Data"):
            return dict(getattr(self.server.Data, "robot_poses", {}) or {})
        return {}

    def get_pose(self, robot_id):
        """[x, y, theta] or [] -- the shape their `get_pose` returns."""
        pose = self.get_all_robot_pose().get(int(robot_id), {})
        if not pose:
            return []
        return [pose.get("x"), pose.get("y"), pose.get("theta")]

    # -- paths ---------------------------------------------------------------

    def set_path(self, robot_id, path):
        """`[[x_px, y_px, theta_rad, delay_ms], ...]`, as `trace_targets` builds."""
        with self._lock:
            self._paths[int(robot_id)] = list(path or [])
        return f"Path updated for robot {robot_id}."

    def get_path(self, robot_id):
        with self._lock:
            return list(self._paths.get(int(robot_id), []))

    def get_all_path(self):
        with self._lock:
            return {k: list(v) for k, v in self._paths.items()}

    def path_list(self):
        """A METHOD, where theirs is a plain dict attribute.

        `pp.py` line 721 writes `c.Robot.path_list()[id]`, which cannot work
        against an attribute over RPC: the proxy turns every name into a call,
        the server does `getattr(service, name)(*args)`, and calling a dict
        raises. As a method it does what that line plainly means.
        """
        return self.get_all_path()

    def clear_path(self, robot_id):
        with self._lock:
            self._paths.pop(int(robot_id), None)
        return True

    # -- state ---------------------------------------------------------------

    def get_state(self, robot_id):
        with self._lock:
            return self._states.get(int(robot_id))

    def set_state(self, robot_id, new):
        with self._lock:
            if int(robot_id) not in self._states:
                return False
            self._states[int(robot_id)] = new
        return True

    # -- commands ------------------------------------------------------------

    def set_command(self, robot_id, command):
        """REFUSED, deliberately and loudly.

        `[left, right, gripper]` with 90 neutral is a differential-drive
        command, and a Sphero is a rolling ball with an absolute course in its
        own aim frame. The conversion people reach for -- read (v, w) back out
        of the wheel pair and integrate w into a heading -- puts an open-loop
        dead-reckon underneath a controller that assumes closed-loop heading
        feedback, on a robot whose heading is not measured in the first place.
        It would run, and it would be wrong in a way that looks like tuning.

        Raising is the point. A caller that silently did nothing would leave an
        agent reporting success over a ball that never moved, which is the
        exact failure `wait_until_arrived` was rewritten to stop telling.
        """
        raise Refused(
            f"set_command{tuple(command) if command else ()} is differential "
            f"drive; robot {robot_id} is a Sphero. Use set_course(robot_id, "
            f"heading_deg, speed), or drive it with a path via set_path + "
            f"request_drive.")

    def set_course(self, robot_id, heading_deg, speed):
        """The Sphero-shaped command: an absolute course, and a speed.

        Recorded, not sent. The bench picks it up and hands it to the handle,
        which is the only thing holding the BLE link and the only thing that
        knows how fast it may be written to.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            raise Refused(f"robot {rid} is not on this rig ({self.id_list})")
        with self._lock:
            self._courses[rid] = {"heading_deg": float(heading_deg) % 360.0,
                                  "speed": float(speed), "at": time.time()}
        return True

    def take_course(self, robot_id):
        """Polled by the bench. The pending course, or None."""
        with self._lock:
            return self._courses.pop(int(robot_id), None)

    # -- drive requests, which the bench serves -------------------------------

    def request_drive(self, robot_id, note=None):
        """Ask the bench to drive the path already set for this robot.

        What `Functions/Library/sphero_control.exec_robot_create_thread` calls.
        Returns a string because that is what their `controller.py` returns and
        what an agent gets shown.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc before asking anything to drive.")
        with self._lock:
            if not self._paths.get(rid):
                return "Path for specific robot doesn't exist. Generate path first"
            self._requests[rid] = {"want": "drive", "at": time.time(),
                                   "note": note}
            self._outcomes.pop(rid, None)
        return f"Started controller thread for robot {rid}"

    def request_calibrate(self, robot_id):
        """Ask the bench to zero the aim and probe the frame.

        What the dashboard's Calibrate button now means on this rig. Theirs
        segments obstacles; ours answers the question no amount of reading the
        source can: which way this ball's compass runs relative to this camera.
        `start_probe` zeroes the aim itself before driving, so this is one
        request rather than two.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc first.")
        with self._lock:
            self._requests[rid] = {"want": "probe", "at": time.time()}
            self._outcomes.pop(rid, None)
        return ("Zeroing the aim and probing the frame — watch the bench "
                "window. This drives the ball in four directions.")

    def request_orbit(self, robot_id, x, y, radius):
        """Ask the bench to drive a real CIRCLE around a point.

        `trace_targets` can only express a list of waypoints, so an agent asked
        to orbit approximates one with a polygon -- which ends, and whose
        corners the lookahead cuts. The bench has `Path.circle`, and a CLOSED
        path repeats until it is stopped, which is what orbiting means.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc before asking anything to drive.")
        with self._lock:
            self._requests[rid] = {"want": "orbit", "at": time.time(),
                                   "centre": [float(x), float(y)],
                                   "radius": float(radius)}
            self._outcomes.pop(rid, None)
        return (f"Orbiting ({x:.0f}, {y:.0f}) at radius {radius:.0f}px. This "
                f"REPEATS until you call stop_robot_thread — it never arrives.")

    def request_patrol(self, robot_id, x1, y1, x2, y2):
        """Back and forth between two points, until stopped.

        NOT a path through `set_path`, and the difference is the whole reason
        this exists. An out-and-back route lays both legs on the same line, so
        the projection cannot tell them apart and the follower reverses on
        floating-point noise near the far end -- a patrol that oscillates
        around one corner and never covers the run. Pure pursuit cannot aim
        around a 180-degree reversal either, so the fix is not a better path:
        the bench drives two ordinary one-way legs and swaps the ends on
        arrival, where arriving is an ordinary arrival.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc before asking anything to drive.")
        with self._lock:
            self._requests[rid] = {"want": "patrol", "at": time.time(),
                                   "a": [float(x1), float(y1)],
                                   "b": [float(x2), float(y2)]}
            self._outcomes.pop(rid, None)
        return (f"Patrolling ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}). This "
                f"REPEATS until you call stop_robot_thread — it never arrives.")

    def request_trajectory(self, robot_id, points):
        """Drive an explicit list of waypoints, with no planner in between.

        `trace_targets` runs A* and may move a goal it thinks is unreachable;
        this drives exactly what was asked for. Useful when the shape matters.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc before asking anything to drive.")
        pts = [[float(a), float(b)] for a, b in (points or [])]
        if len(pts) < 2:
            return "A trajectory needs at least two points."
        with self._lock:
            self._paths[rid] = [[x, y, 0.0, 0] for x, y in pts]
            self._requests[rid] = {"want": "drive", "at": time.time()}
            self._outcomes.pop(rid, None)
        return f"Driving a {len(pts)}-point trajectory for robot {rid}."

    def set_follow_target(self, robot_id, x, y):
        """Chase a point, and keep chasing it as it is updated.

        A FOLLOWER, not a drive: calling this again while it is running moves
        the target under a ball already going for it, rather than starting
        again. There is nothing on this rig for it to lock onto by itself --
        object detection needs SAM2 and a lit room while the tracker needs it
        dark, and there is one robot, so there is no second one to chase. The
        target has to be told to it.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc before asking anything to drive.")
        with self._lock:
            self._requests[rid] = {"want": "follow", "at": time.time(),
                                   "target": [float(x), float(y)]}
            self._outcomes.pop(rid, None)
        return (f"Following ({x:.0f}, {y:.0f}). Call again to move the target; "
                f"call stop_robot_thread to stop.")

    def request_flow(self, robot_id, expression, closed=True):
        """Drive a curve the caller DESCRIBES rather than enumerates.

        For the shapes a waypoint list is a poor way to say: a figure eight, a
        spiral, a lissajous. `tools/generate.py` already exists for exactly
        this case -- "a model wants twenty points on a curve and would rather
        write the curve than the twenty points" -- and this is that, pointed at
        one robot's path instead of a formation.

        The expression is evaluated on the BENCH, not here, because the sandbox
        needs the workspace to check what it produced.
        """
        rid = int(robot_id)
        if rid not in self.id_list:
            return f"Selected ID doesn't exist ({self.id_list})"
        if not self.attached:
            return ("No robot is attached to this service — start the bench "
                    "with --rpc before asking anything to drive.")
        if not str(expression or "").strip():
            return "An expression is required."
        with self._lock:
            self._requests[rid] = {"want": "flow", "at": time.time(),
                                   "expression": str(expression),
                                   "closed": bool(closed)}
            self._outcomes.pop(rid, None)
        return ("Evaluating the curve and driving it." +
                (" This REPEATS until you call stop_robot_thread."
                 if closed else ""))

    # -- what the arena actually is -----------------------------------------

    def set_arena(self, arena):
        """Told by the bench, because only the bench knows.

        The arena is four corners a person clicked, mapped through a
        homography. It changes whenever either is redone, so anything that
        writes its size into a prompt is writing a number that goes stale
        silently -- which is exactly what happened: an agent kept orbiting
        (694, 554) after a recalibration moved the centre to (712, 613).
        """
        with self._lock:
            self._arena = dict(arena or {})
        return True

    def get_arena(self):
        """Size, centre and safe bounds, in arena pixels. Empty until told."""
        with self._lock:
            return dict(self._arena)

    def request_stop(self, robot_id):
        rid = int(robot_id)
        with self._lock:
            self._requests[rid] = {"want": "stop", "at": time.time()}
        return f"Stop requested for robot {rid}"

    def take_request(self, robot_id):
        """Polled by the bench. The pending request, or None."""
        with self._lock:
            return self._requests.pop(int(robot_id), None)

    def set_outcome(self, robot_id, outcome, reason=None):
        """The bench says how a drive ended. `arrived`, `stuck`, `lost`, ...

        Written back because "stopped" is not "arrived" and the difference is
        the thing an agent needs. Their own service has nowhere to put this,
        which is why `wait_until_arrived` on the bench had to grow one.
        """
        with self._lock:
            self._outcomes[int(robot_id)] = {
                "outcome": outcome, "reason": reason, "at": time.time()}
            self._states[int(robot_id)] = HALT
        return True

    def get_outcome(self, robot_id):
        with self._lock:
            got = self._outcomes.get(int(robot_id))
            return dict(got) if got else None

    # -- fleet-wide ----------------------------------------------------------

    def stop_all(self):
        with self._lock:
            for rid in self.id_list:
                self._requests[rid] = {"want": "stop", "at": time.time()}
                self._states[rid] = HALT
                self._courses.pop(rid, None)
        return "Stopping all robots..."

    def resume(self):
        """Nothing to restart. Theirs clears a stop_event on a send loop; the
        bench owns that loop and never stopped ticking."""
        return "Robot service resumed"

    # -- names ---------------------------------------------------------------

    def get_all_names(self):
        return list(self.names)

    def get_name_from_id(self, robot_id):
        try:
            return self.names[self.id_list.index(int(robot_id))]
        except (ValueError, IndexError):
            return None

    def get_id_from_name(self, robot_name):
        want = str(robot_name).strip().lower()
        for i, name in enumerate(self.names):
            if name.lower() == want:
                return self.id_list[i]
        # Roster codes too, because that is what a person working on this rig
        # actually calls the robot -- "CRXS", not "Robot-2".
        for rid, code in self.codes.items():
            if str(code).lower() == want:
                return int(rid)
        return None

    def set_name(self, robot_id, new_name):
        new_name = str(new_name).strip()
        if not new_name:
            raise ValueError("New name cannot be empty.")
        try:
            self.names[self.id_list.index(int(robot_id))] = new_name
        except (ValueError, IndexError):
            raise ValueError(f"Robot ID {robot_id} not found in {self.id_list}.")
        return f"Updated name for ID {robot_id} to '{new_name}'"
