"""The whole chain, with nothing stubbed but the camera and the ball's physics.

    their trace_targets (real A*, real RPC)
      -> client.Robot.set_path
        -> sphero_control.exec_robot_create_thread   (their library)
          -> SpheroRobot.request_drive               (ours, over real RPC)
            -> Driver.serve                          (ours, in the bench)
              -> the bench's own pursue()
                -> Bridge publishes the new position back
                  -> sphero_control.wait_for_robot reads the verdict

Every unit test beside this one mocks the seam it is testing, which is what
makes them fast and precise and is also exactly why they cannot catch a seam
that was never connected. One of them did not: the bridge published happily for
a whole commit while never serving a single drive request, because a
`str.replace` with a mismatched indent no-opped without complaining. This is
the test that would have caught it, so it earns its seconds.

SKIPPED when the framework is not on this machine. It is a separate project in
a separate directory and the bench must not depend on it being there.
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from fleet_test import BallSource, find_blobs, pursue
from vision.homography import Homography
from vlm.bridge import ArenaFrame, Bridge


FRAMEWORK = os.environ.get(
    "VLM_FRAMEWORK",
    os.path.expanduser("~/Downloads/Mobile-manipulation-with-VLMs-March/"
                       "Functions/Utilities"))
ROOT = os.path.dirname(os.path.dirname(FRAMEWORK.rstrip("/")))

pytestmark = pytest.mark.skipif(
    not os.path.isdir(FRAMEWORK) or
    not os.path.isfile(os.path.join(ROOT, "Functions/Library/sphero_control.py")),
    reason="the VLM framework (with sphero_control.py installed) is not here")

PORT = 5613
ARENA_W, ARENA_H = 138.8, 110.8
SIM_PX_CM = 9.2
SIM_SIZE = (1280, int(np.ceil(ARENA_H * SIM_PX_CM)))
ROBOT = 2


def _sim_homography():
    src = np.array([[0, 0], [ARENA_W * SIM_PX_CM, 0],
                    [ARENA_W * SIM_PX_CM, ARENA_H * SIM_PX_CM],
                    [0, ARENA_H * SIM_PX_CM]], dtype=np.float32)
    dst = np.array([[0, 0], [ARENA_W, 0], [ARENA_W, ARENA_H], [0, ARENA_H]],
                   dtype=np.float32)
    return Homography(cv2.getPerspectiveTransform(src, dst).astype(np.float64),
                      width=ARENA_W, height=ARENA_H)


class SimBench:
    """The bench's real tracker and real controller over a simulated ball.

    Faked: the camera (a rendered ball) and the plant (it moves at the
    commanded velocity). Real: detection, the homography, `pursue`, arrival.
    """

    def __init__(self):
        self.at = np.array([40.0, 30.0])
        self.src = BallSource(size=SIM_SIZE, px_cm=SIM_PX_CM, seed=0,
                              pose=lambda: [(self.at, 1.0)])
        self.homography = _sim_homography()
        self.track = type("T", (), {"locked": True, "status": "locked"})()
        self.blob = self.frame = self.path = None
        self.code, self.fleet = "CRXS", None
        self.armed = False
        self.note = self.last_note = ""
        self.last_outcome = None
        self._arm_source = "button"
        self.speed, self.goal_tol = 25.0, 6.0
        self._travel = None
        self.sense()

    def to_cm(self, xy):
        return self.homography.to_cm([xy])[0]

    def travel_readout(self):
        return self._travel

    def sense(self):
        _ok, frame = self.src.read()
        self.frame = frame
        blobs, _m, _n = find_blobs(frame)
        self.blob = blobs[0] if blobs else None
        self.track.locked = bool(blobs)

    def arm(self):
        if self.path is None:
            self.note, self.armed = "refused: no path", False
            return
        self.armed = True
        self.last_outcome, self.last_note = None, ""

    def disarm(self, note=None):
        if self.armed:
            text = (note or "").lower()
            self.last_outcome = ("arrived" if "arrived" in text else
                                 "stuck" if "stuck" in text else "stopped")
            self.last_note = note or ""
        self.armed = False

    def step(self, dt=0.1):
        self.sense()
        if not self.armed or self.path is None:
            return
        here = np.asarray(self.to_cm(self.blob["xy"]), dtype=float)
        v, _t, done, note = pursue(self.path, here, lookahead=15.0,
                                   speed=self.speed, goal_tol=self.goal_tol)
        if done:
            self.disarm(note or "arrived")
            self._travel = None
            return
        self.at = self.at + np.asarray(v, dtype=float) * dt
        if float(np.linalg.norm(v)) > 1.0:
            self._travel = (float(np.degrees(np.arctan2(v[1], v[0])) % 360.0),
                            None, float(np.linalg.norm(v)))


@pytest.fixture(scope="module")
def rig():
    """Their server with our robot behind it, plus a bench feeding it."""
    was = os.getcwd()
    for path in (FRAMEWORK, ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(ROOT)                       # their loaders use relative paths
    try:
        from vlm.server import build
        server = build(port=PORT, framework=FRAMEWORK, robot_id=ROBOT)
        threading.Thread(target=server.run, daemon=True).start()
        time.sleep(1.0)

        from rpc_system import RPCClient
        import Functions.Library.sphero_control as ctl
        import Functions.Library.planning as planning

        client = RPCClient(host="localhost", port=PORT)
        ctl.client = client
        planning.PlanningClient.__init__ = (
            lambda self, _c=client: setattr(self, "client", _c))

        bench = SimBench()
        arena = ArenaFrame(ARENA_W, ARENA_H)
        bridge = Bridge(bench, robot_id=ROBOT, arena=arena, client=client)

        def tick(n=1):
            for _ in range(n):
                bench.step()
                bridge.publish_once()

        tick(3)
        yield {"client": client, "ctl": ctl, "planning": planning,
               "bench": bench, "bridge": bridge, "arena": arena, "tick": tick}
    finally:
        os.chdir(was)


HOME_CM = (40.0, 30.0)


@pytest.fixture(autouse=True)
def fresh(rig):
    """Put the rig back to a known state before every test.

    The suite does not collect in definition order, so a test that assumed the
    ball had not moved yet passed alone and failed in company. Sharing an
    expensive server between tests is fine; sharing their leftovers is not.
    """
    bench, client, tick = rig["bench"], rig["client"], rig["tick"]
    client.Robot.request_stop(ROBOT)
    tick(2)
    client.Robot.clear_path(ROBOT)
    bench.armed, bench.path = False, None
    bench.last_outcome, bench.last_note = None, ""
    bench.at = np.array(HOME_CM)
    bench.track.locked = True
    bench._travel = None
    tick(3)
    return rig


def test_their_planner_can_see_our_robot(rig):
    poses = rig["client"].Robot.get_all_robot_pose()
    assert ROBOT in poses
    want = rig["arena"].to_px(HOME_CM)
    assert poses[ROBOT]["x"] == pytest.approx(want[0], abs=8.0)
    assert poses[ROBOT]["y"] == pytest.approx(want[1], abs=8.0)


def test_driving_before_planning_is_refused(rig):
    said = rig["ctl"].exec_robot_create_thread(robot_id=ROBOT)
    assert "Generate path first" in said


def test_the_whole_chain_drives_the_ball_to_a_planned_goal(rig):
    ctl, bench, arena, tick = rig["ctl"], rig["bench"], rig["arena"], rig["tick"]

    goal_px = [1000, 800]
    said = rig["planning"].trace_targets(
        robot_id=ROBOT, input_target_list=[goal_px], spacing=30, verbose=False)
    assert "successful" in said
    assert len(rig["client"].Robot.get_path(ROBOT)) >= 1

    assert "Started controller thread" in ctl.exec_robot_create_thread(robot_id=ROBOT)
    tick(2)
    assert bench.armed is True

    start = bench.at.copy()
    for _ in range(400):
        tick(1)
        if not bench.armed:
            break

    assert float(np.linalg.norm(bench.at - start)) > 10.0, "the ball never moved"
    goal_cm = np.array(arena.to_cm(goal_px))
    assert float(np.linalg.norm(bench.at - goal_cm)) < 10.0

    verdict = ctl.wait_for_robot(robot_id=ROBOT, timeout_s=5.0)
    assert verdict["arrived"] is True
    assert verdict["outcome"] == "arrived"


def test_position_reads_back_through_their_library(rig):
    where = rig["ctl"].get_robot_position(robot_id=ROBOT)
    assert where["tracked"] is True
    want = rig["arena"].to_px(HOME_CM)
    assert where["x"] == pytest.approx(want[0], abs=10.0)
    # The bearing is TRAVEL and the ball is at rest, so it must not claim to
    # have measured one.
    assert where["theta_is_measured"] is False


def test_a_lost_ball_reads_as_NOT_SEEN_through_their_library(rig):
    bench, bridge, ctl = rig["bench"], rig["bridge"], rig["ctl"]
    bench.track.locked = False
    bridge.publish_once()
    try:
        where = ctl.get_robot_position(robot_id=ROBOT)
        assert where["tracked"] is False
        assert "cannot see" in where["reason"]
    finally:
        bench.track.locked = True
        bridge.publish_once()


def test_a_differential_drive_command_is_refused_across_the_wire(rig):
    """The refusal has to survive RPC, not just work in-process."""
    with pytest.raises(Exception) as e:
        rig["client"].Robot.set_command(ROBOT, [120, 60, None])
    assert "differential" in str(e.value)


def test_stop_reaches_the_bench(rig):
    ctl, bench, tick = rig["ctl"], rig["bench"], rig["tick"]
    ctl.exec_robot_create_thread(robot_id=ROBOT)
    tick(2)
    ctl.stop_robot_thread(robot_id=ROBOT)
    tick(2)
    assert bench.armed is False
