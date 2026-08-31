"""Serve the framework's drive requests, using the bench's own controller.

The other half of `vlm.service`. That module is a blackboard living in the RPC
server; this is the thing in the bench process that reads it and actually does
something, because the bench is where the robot handle and the tracker that
closes the loop around it both live.

No control loop of its own. A drive request becomes `app.path` and a call to
`app.arm()` -- exactly what the GO button and the bench's own agent tools do,
and deliberately the same path so there is one controller to trust rather than
three that agree until they do not.

Called once per bridge tick, on the BRIDGE'S OWN THREAD and with the bridge's
client. That is not a detail: `RPCClient` holds a single ZMQ REQ socket, which
is not thread-safe, and a second thread calling into it would interleave sends
and receives on the same socket. One thread, one socket, no races.
"""

import numpy as np


class Driver:
    """Turns requests on the blackboard into drives on the bench."""

    def __init__(self, app, arena, robot_id=2, path_cls=None):
        self.app = app
        self.arena = arena
        self.robot_id = int(robot_id)
        self._path_cls = path_cls

        self.attached = False
        self.was_armed = False
        self.was_probing = False
        self.served = 0
        self.refused = 0
        self.last_note = ""

    @property
    def Path(self):
        """`fleet_test.Path`, imported late.

        The bench imports this module; importing the bench back at module scope
        would be a cycle. It is also what lets the tests hand in a stand-in
        rather than starting a pygame window to check an arithmetic conversion.
        """
        if self._path_cls is None:
            from fleet_test import Path
            self._path_cls = Path
        return self._path_cls

    # -- conversion ----------------------------------------------------------

    def path_to_cm(self, path_px):
        """`[[x_px, y_px, theta_rad, delay_ms], ...]` -> a list of cm points.

        Only x and y are read. The third element is a per-point heading their
        planner computes from consecutive waypoints, which is a property of the
        PATH rather than of the robot and which pure pursuit derives for itself
        from the lookahead; carrying it through would be storing an answer next
        to the question it is derived from.

        THE DELAYS ARE DROPPED, and that is a real narrowing rather than an
        oversight. Their fourth element is a dwell in milliseconds at each
        waypoint, which `pursue` has no notion of -- it follows a path, it does
        not stop partway along one. A route that asked for a pause gets driven
        straight through, so `sphero_control` says so where an agent can read
        it rather than leaving it to be discovered from behaviour.
        """
        out = []
        for point in path_px or []:
            if point is None or len(point) < 2:
                continue
            x, y = float(point[0]), float(point[1])
            out.append(np.array(self.arena.to_cm((x, y)), dtype=float))
        return out

    def wants_delay(self, path_px):
        """Did the route ask for a dwell we are about to ignore?"""
        return any(len(p) > 3 and p[3] not in (0, None, "")
                   for p in (path_px or []) if p is not None)

    # -- the tick ------------------------------------------------------------

    def serve(self, client):
        """One pass. Safe to call every tick; does nothing when nothing waits."""
        if client is None:
            return None
        if not self.attached:
            client.Robot.attach()
            self.attached = True

        self.report_finished(client)
        self.report_probe(client)

        request = client.Robot.take_request(self.robot_id)
        if request is not None:
            return self.obey(client, request)

        course = client.Robot.take_course(self.robot_id)
        if course is not None:
            return self.steer(course)
        return None

    def report_finished(self, client):
        """Say HOW a drive ended, once, on the tick it ends.

        Watched as a transition rather than polled as a state, because the
        verdict is only written at `disarm` and the next drive clears it. And
        the verdict is the point: `armed` goes false for arriving, for giving
        up stuck, for losing the ball and for a person pressing escape, so
        "stopped" carries none of the information a caller needs.
        """
        armed = bool(self.app.armed)
        if self.was_armed and not armed:
            client.Robot.set_outcome(
                self.robot_id,
                getattr(self.app, "last_outcome", None) or "unknown",
                getattr(self.app, "last_note", "") or "stopped for no stated reason")
        self.was_armed = armed

    def obey(self, client, request):
        want = request.get("want")
        if want == "stop":
            self.app.disarm("stopped by the framework")
            self.last_note = "stopped"
            return "stopped"
        if want == "probe":
            return self.probe()
        if want != "drive":
            self.last_note = f"unknown request {want!r}"
            return None
        return self.drive(client)

    def probe(self):
        """Zero the aim and probe the frame, via the bench's own routine.

        `start_probe` disarms, zeroes at rest, then drives each cardinal
        heading through `drive_raw` and reads back where the ball actually
        went. Its refusals -- nothing connected, no homography, no lock -- are
        left to it: they print in the bench window, which is where somebody
        running a calibration is looking.
        """
        self.app.start_probe()
        if getattr(self.app, "probe", None) is None:
            self.last_note = getattr(self.app, "note", "") or "probe refused"
            return None
        # Marked HERE, not left to the next tick's `report_probe`. That runs at
        # the top of `serve`, before the request is obeyed, so it would read
        # False on the tick the probe starts and the finish would never be seen
        # as a transition. `drive` sets `was_armed` for the same reason.
        self.was_probing = True
        self.last_note = "probing the frame"
        return self.last_note

    def report_probe(self, client):
        """Say how the probe ended, once, on the tick it finishes."""
        probing = getattr(self.app, "probe", None) is not None
        if self.was_probing and not probing:
            mirrored = bool(getattr(self.app, "mirrored", False))
            client.Robot.set_outcome(
                self.robot_id,
                "mirrored" if mirrored else "probed",
                getattr(self.app, "note", "") or
                ("the frame is MIRRORED — flip an axis and probe again; "
                 "nothing will converge until this is clean" if mirrored
                 else "the frame is a rotation, which is what it should be"))
        self.was_probing = probing

    def drive(self, client):
        """Build the bench's own Path from their waypoints, and arm."""
        path_px = client.Robot.get_path(self.robot_id)
        points = self.path_to_cm(path_px)
        if len(points) < 1:
            self.refused += 1
            self.last_note = "the path had no usable points"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None

        # A single waypoint is a GOAL, not a path. `Path.point` exists for
        # exactly this and behaves differently from a one-element polyline,
        # which has no length for the lookahead to run along.
        self.app.path = (self.Path.point(points[0]) if len(points) == 1
                         else self.Path([np.asarray(p, float) for p in points]))

        self.app._arm_source = "vlm"
        self.app.arm()
        if not self.app.armed:
            # `arm` refuses for reasons a caller has to see verbatim -- a
            # mirrored frame, no robot, no homography, an unestablished aim.
            # Swallowing that and reporting a start is how an agent ends up
            # planning on top of a rig that never moved.
            self.refused += 1
            self.last_note = getattr(self.app, "note", "") or "arm refused"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None

        self.served += 1
        self.was_armed = True
        self.last_note = f"driving {self.app.path.length:.0f} cm"
        if self.wants_delay(path_px):
            self.last_note += " (per-point delays ignored)"
        return self.last_note

    def steer(self, course):
        """Apply a raw course: an absolute bearing and a speed.

        Supersedes a path drive rather than fighting it. Two things commanding
        one ball is the kind of fault that reads as a tuning problem for a week
        -- the ball wanders, both sources look individually reasonable, and
        nothing in either log says the other one existed.
        """
        handle = self.handle()
        if handle is None:
            self.last_note = "no robot connected to steer"
            return None
        if self.app.armed:
            self.app.disarm("a direct course superseded the path")
        try:
            handle.drive_raw(float(course["heading_deg"]), int(course["speed"]))
        except Exception as e:
            self.last_note = f"could not steer: {type(e).__name__}: {e}"
            return None
        self.last_note = (f"course {course['heading_deg']:.0f} deg "
                          f"at {course['speed']:.0f}")
        return self.last_note

    def handle(self):
        fleet = getattr(self.app, "fleet", None)
        code = getattr(self.app, "code", None)
        if not fleet or not code:
            return None
        return fleet.handles.get(code)

    def close(self, client):
        if client is not None and self.attached:
            try:
                client.Robot.detach()
            except Exception:
                pass
        self.attached = False
