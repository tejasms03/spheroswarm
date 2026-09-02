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
        self.was_scaling = False
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
            # From the APP'S OWN module where it has one, not `fleet_test` by
            # name. The bench is forked -- `coast_test.py` is a copy kept
            # deliberately -- and handing a `fleet_test.Path` to a `coast_test`
            # controller works only for as long as the two happen to agree.
            # They are meant to diverge; that is what the fork is for.
            #
            # Falling back rather than insisting, because an app is not always
            # a bench: the end-to-end test drives a stand-in defined in the
            # test module, which has no `Path` of its own and does not need
            # one.
            import sys
            from fleet_test import Path
            own = sys.modules.get(type(self.app).__module__)
            self._path_cls = getattr(own, "Path", Path)
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
            client.Robot.set_arena(self.arena_facts())
            self.attached = True

        self.report_finished(client)
        self.report_probe(client)
        self.report_scale(client)

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
        if want == "scale":
            return self.scale_run(client, request)
        if want == "scale_fix":
            return self.scale_fix(client, request)
        if want == "flow":
            return self.flow(client, request)
        if want == "patrol":
            return self.patrol(client, request)
        if want == "follow":
            return self.follow(client, request)
        if want == "orbit":
            return self.orbit(client, request)
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

    def orbit(self, client, request):
        """A real circle, not a polygon that approximates one.

        Closed, so `pursue` runs it until something stops it. Nothing here
        reports an arrival because there is not one to report -- the caller is
        told that up front by `request_orbit` rather than discovering it when
        `wait_for_robot` times out.
        """
        centre = self.arena.to_cm(request["centre"])
        radius = self.arena.to_cm((request["radius"], 0.0))[0]
        self.app.path = self.anchored(
            self.Path.circle(np.asarray(centre, float), radius))
        self.app._arm_source = "vlm"
        self.app.arm()
        if not self.app.armed:
            self.refused += 1
            self.last_note = getattr(self.app, "note", "") or "arm refused"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None
        self.served += 1
        client.Robot.set_state(self.robot_id, "moving")
        self.was_armed = True
        self.last_note = f"orbiting r={radius:.0f}cm"
        return self.last_note

    def patrol(self, client, request):
        """The bench's own patrol, which is a state machine and not a shape.

        `start_patrol` drives one plain one-way leg and swaps the ends when it
        arrives. The obvious alternative -- a single `[a, b, a]` path -- is the
        thing that was already failing: both legs lie on the same line, the
        projection cannot tell them apart, and the follower reverses on
        floating-point noise near the far end. That is a patrol that
        oscillates around one corner instead of covering its run.
        """
        a = self.arena.to_cm(request["a"])
        b = self.arena.to_cm(request["b"])
        self.app.start_patrol(np.asarray(a, float), np.asarray(b, float))
        self.app._arm_source = "vlm"
        self.app.arm()
        if not self.app.armed:
            self.refused += 1
            self.last_note = getattr(self.app, "note", "") or "arm refused"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None
        self.served += 1
        client.Robot.set_state(self.robot_id, "moving")
        self.was_armed = True
        self.last_note = "patrolling"
        return self.last_note

    def follow(self, client, request):
        """Chase a point, and MOVE the target rather than restarting.

        Re-arming on every update would zero the aim and clear the escape
        budget each time the target twitched, which on a fast-moving target is
        a ball that spends its life in `zero_at_rest` and never travels.
        """
        goal = np.asarray(self.arena.to_cm(request["target"]), dtype=float)
        if self.app.armed and getattr(self.app.path, "kind", None) == "point":
            self.app.path = self.Path.point(goal)
            self.last_note = "target moved"
            return self.last_note

        self.app.path = self.Path.point(goal)
        self.app._arm_source = "vlm"
        self.app.arm()
        if not self.app.armed:
            self.refused += 1
            self.last_note = getattr(self.app, "note", "") or "arm refused"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None
        self.served += 1
        client.Robot.set_state(self.robot_id, "moving")
        self.was_armed = True
        self.last_note = "following"
        return self.last_note

    FLOW_VARS_PX = ("cx", "cy", "width", "height",
                    "xmin", "xmax", "ymin", "ymax")

    def flow(self, client, request):
        """Evaluate a parametric curve in the sandbox, then drive it.

        `tools/validate.compute_points` is the sandbox the formation tools
        already use -- no imports, no attribute access beyond maths, no
        `while`, one second. Two of its options are set differently here:
        `expected_count=None`, because a curve is not one point per robot, and
        `min_separation=0`, because consecutive samples along a curve are
        MEANT to be close and the formation rule that keeps robots apart would
        reject every trajectory.

        The expression works in ARENA PIXELS, like every other tool the agent
        has. Converting after evaluation rather than asking the model to think
        in two unit systems at once is the whole reason `cx`, `cy` and the
        bounds are handed in already scaled.
        """
        try:
            from tools.validate import run_sandboxed
        except Exception as e:
            self.last_note = f"no sandbox: {type(e).__name__}: {e}"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None

        w, h = self.arena.size_px
        variables = {"n": 1, "cx": w / 2.0, "cy": h / 2.0,
                     "width": float(w), "height": float(h),
                     "xmin": 0.0, "xmax": float(w),
                     "ymin": 0.0, "ymax": float(h)}
        # `run_sandboxed`, not `compute_points`. The latter validates what it
        # produced against the workspace, which is in CENTIMETRES -- feeding it
        # pixels clamps every point to one corner, which is exactly what it did
        # the first time. The checks that matter here are done below, in the
        # units they belong to.
        run = run_sandboxed(request["expression"], variables=variables)
        if not run["ok"] or run["value"] is None:
            self.refused += 1
            self.last_note = ("the curve was refused: " +
                              (run.get("error") or "it assigned no `points`"))
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None

        try:
            points = [np.asarray(self.arena.to_cm((float(x), float(y))),
                                 dtype=float) for x, y in run["value"]]
        except (TypeError, ValueError) as e:
            self.refused += 1
            self.last_note = f"`points` must be a list of (x, y): {e}"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None
        if len(points) < 2:
            self.refused += 1
            self.last_note = "a curve needs at least two points"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None

        # The bench's own boundary rule, not a second copy of it. It refuses a
        # goal the ball can only reach by shoving, and a curve is only as safe
        # as its worst point.
        bad = self.app.agent_check(points)
        if bad:
            self.refused += 1
            self.last_note = self.in_pixels(bad)
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None

        self.app.path = self.anchored(
            self.Path(points, closed=bool(request.get("closed")), kind="flow"))
        self.app._arm_source = "vlm"
        self.app.arm()
        if not self.app.armed:
            self.refused += 1
            self.last_note = getattr(self.app, "note", "") or "arm refused"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None
        self.served += 1
        client.Robot.set_state(self.robot_id, "moving")
        self.was_armed = True
        self.last_note = f"driving a {len(points)}-point curve"
        return self.last_note

    def in_pixels(self, refusal):
        """Restate a centimetre refusal in the units the caller wrote in.

        `agent_check` is the bench's own boundary rule and it speaks
        centimetres, because that is what the bench thinks in. Every tool the
        agent has speaks ARENA PIXELS. Handing back "aim inside x 10..129"
        after being given a curve spanning 294 to 1094 reads as a rig fault
        rather than an instruction -- an agent said so in as many words, and
        stopped rather than retry, which was the right call on the information
        it had.

        The centimetre wording is kept and the pixel bounds appended, because
        the numbers a person reads in the bench window are centimetres and the
        numbers the agent must aim in are pixels. Both are true; only one is
        actionable from where the agent sits.
        """
        bounds = getattr(self.app, "agent_bounds", None)
        margin = getattr(self.app, "goal_margin_cm", None)
        if bounds is None or margin is None:
            return refusal
        box = bounds()
        if not box:
            return refusal
        m = margin()
        lo = self.arena.to_px((box[0] + m, box[1] + m))
        hi = self.arena.to_px((box[2] - m, box[3] - m))
        return (f"{refusal} — IN ARENA PIXELS, which is what these tools take: "
                f"aim inside x {lo[0]:.0f}..{hi[0]:.0f}, "
                f"y {lo[1]:.0f}..{hi[1]:.0f}")

    def arena_facts(self):
        """What the arena IS, for anything that would otherwise guess.

        Published once on attach rather than written into a prompt. The arena
        is a clicked quad through a homography and moves whenever either is
        redone; a literal in a description is a number that goes stale without
        saying so.
        """
        w, h = self.arena.size_px
        margin = 100.0
        try:
            margin = max(margin, self.arena.scalar_to_px(
                self.app.goal_margin_cm()))
        except Exception:
            pass
        cx, cy = w / 2.0, h / 2.0
        # THE BIGGEST CIRCLE THAT FITS, from the centre to the nearest safe
        # edge. Without it the agent has only "10 pixels is 1cm" to go on and
        # picks something timid -- asked to orbit, it drew circles a tenth of
        # the arena across, which is not what anybody means by orbiting.
        max_radius = max(0.0, min(cx - margin, cy - margin,
                                  (w - margin) - cx, (h - margin) - cy))
        return {
            "width": w, "height": h,
            "centre": [round(cx, 1), round(cy, 1)],
            "max_radius": round(max_radius, 1),
            "px_per_cm": self.arena.px_per_cm,
            "width_cm": round(self.arena.width_cm, 1),
            "height_cm": round(self.arena.height_cm, 1),
            "safe": [round(margin, 1), round(margin, 1),
                     round(w - margin, 1), round(h - margin, 1)],
            "margin": round(margin, 1),
        }

    @staticmethod
    def anchored(path):
        """Drive this route from its START, not from the nearest bit of it.

        Pure pursuit locks on to the closest point when a run begins, which is
        what lets a goal be driven from wherever the ball happens to be. For a
        shape it is wrong: a ball sitting near the middle of a figure eight
        starts halfway round, the first half never gets driven, and nothing
        reports a fault because the follower did exactly what it was told.
        """
        path.anchor = True
        path.restart()
        return path

    def scale_run(self, client, request):
        """Drive a straight line so the operator can measure it with a tape."""
        self.app.start_scale_run(seconds=request.get("seconds", 3.0),
                                 speed=request.get("speed"))
        if getattr(self.app, "scale_run", None) is None:
            self.last_note = getattr(self.app, "note", "") or "scale run refused"
            client.Robot.set_outcome(self.robot_id, "refused", self.last_note)
            return None
        self.was_scaling = True
        self.last_note = "measuring the scale"
        return self.last_note

    def report_scale(self, client):
        """Publish the pixel measurement the instant the run settles."""
        running = getattr(self.app, "scale_run", None) is not None
        if self.was_scaling and not running:
            pend = getattr(self.app, "pending_scale", None)
            client.Robot.set_scale_result(dict(pend) if pend else
                                          {"error": getattr(self.app, "note", "")})
        self.was_scaling = running

    def scale_fix(self, client, request):
        """Fold the operator's measured distance into the homography.

        The arena moves with it, so the cached ArenaFrame has to go: every
        published position after this is in different centimetres, and holding
        the old rectangle would report the new measurements against the old
        floor.
        """
        said = self.app.set_measured_cm(request["measured_cm"])
        client.Robot.set_scale_result({"applied": said})
        self.arena = None
        self.attached = False          # re-attach republishes the arena
        self.last_note = said
        return said

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
        # A single waypoint is a goal and starts from wherever the ball is;
        # a list of them is a SHAPE and starts at its own beginning.
        self.app.path = (self.Path.point(points[0]) if len(points) == 1
                         else self.anchored(
                             self.Path([np.asarray(p, float) for p in points])))

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
        client.Robot.set_state(self.robot_id, "moving")
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
