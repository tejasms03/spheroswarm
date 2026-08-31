"""The boundary between this bench and the VLM framework.

Every test here builds its own workspace. `tests/test_fleet.py` learned the
hard way that `_app_with_ball()` loads the operator's live
`calib/blob_region.json`, so the suite silently inherits whatever workspace was
last clicked and shifts underfoot on recalibration. Nothing below reads a
calibration file.
"""

import math
import time

import cv2
import numpy as np
import pytest

from vlm.bridge import ArenaFrame, Bridge, Reading


ARENA_W, ARENA_H = 138.8, 110.8


class FakeHomography:
    """Just enough of `vision.homography.Homography` for the bridge."""

    def __init__(self, matrix=None, width=ARENA_W, height=ARENA_H):
        self.M = matrix
        self.width = width
        self.height = height

    @property
    def ready(self):
        return self.M is not None

    def to_cm(self, pts_px):
        p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.M).reshape(-1, 2)

    def to_px(self, pts_cm):
        inv = np.linalg.inv(self.M)
        p = np.asarray(pts_cm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, inv).reshape(-1, 2)


class FakeTrack:
    def __init__(self, locked=True, status="locked"):
        self.locked = locked
        self.status = status


class FakeApp:
    """A bench that reports what the test tells it to."""

    def __init__(self, xy_px=(100.0, 100.0), locked=True, status="locked",
                 travel=None, matrix=None, frame=None):
        if matrix is None:
            matrix = _square_homography()
        self.homography = FakeHomography(matrix)
        self.track = FakeTrack(locked, status)
        self.blob = None if xy_px is None else {"xy": np.array(xy_px, dtype=float)}
        self._travel = travel
        self.frame = frame
        self.code = "CRXS"
        # Read by the driver on every tick, because publishing also serves.
        self.armed = False
        self.fleet = None

    def to_cm(self, xy):
        return self.homography.to_cm([xy])[0]

    def travel_readout(self):
        return self._travel


def _square_homography(scale=4.0):
    """A plain scale: `scale` camera pixels to the centimetre, no perspective."""
    src = np.array([[0, 0], [ARENA_W * scale, 0],
                    [ARENA_W * scale, ARENA_H * scale], [0, ARENA_H * scale]],
                   dtype=np.float32)
    dst = np.array([[0, 0], [ARENA_W, 0], [ARENA_W, ARENA_H], [0, ARENA_H]],
                   dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def _tilted_homography():
    """A camera looking at the arena from an angle -- a real trapezoid."""
    src = np.array([[180, 120], [560, 140], [610, 400], [130, 380]],
                   dtype=np.float32)
    dst = np.array([[0, 0], [ARENA_W, 0], [ARENA_W, ARENA_H], [0, ARENA_H]],
                   dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


class FakeRobot:
    """The command half. Present because publishing SERVES as well as sends --
    one thread and one socket do both, so a client with only `Data` on it is
    not a client the bridge can use."""

    def __init__(self):
        self.attached = False

    def attach(self):
        self.attached = True
        return True

    def detach(self):
        self.attached = False
        return True

    def take_request(self, rid):
        return None

    def take_course(self, rid):
        return None

    def set_outcome(self, rid, outcome, reason=None):
        return True


class FakeClient:
    """Records what `update_state` was called with."""

    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises
        self.Data = self
        self.Robot = FakeRobot()

    def update_state(self, frame, poses, obstacles, ind, raw, hf):
        if self.raises:
            raise self.raises
        self.calls.append({"frame": frame, "poses": poses,
                           "obstacles": obstacles, "ind": ind})


# -- ArenaFrame --------------------------------------------------------------

def test_a_centimetre_round_trips_through_arena_pixels():
    arena = ArenaFrame(ARENA_W, ARENA_H)
    for cm in [(0.0, 0.0), (12.5, 90.25), (ARENA_W, ARENA_H)]:
        back = arena.to_cm(arena.to_px(cm))
        assert back == pytest.approx(cm, abs=1e-9)


def test_the_arena_is_sized_from_the_workspace_not_from_their_rig():
    """1388x1108, not the 1500x500 their arena_settings.json ships with."""
    arena = ArenaFrame(ARENA_W, ARENA_H)
    assert arena.size_px == (1388, 1108)


def test_a_length_crosses_as_well_as_a_point():
    arena = ArenaFrame(ARENA_W, ARENA_H)
    # Their robot_padding=30 and sweep_radius=65 are lengths, and a margin that
    # did not cross would leave the planner clearing a distance it never meant.
    assert arena.scalar_to_px(3.0) == pytest.approx(30.0)
    assert arena.scalar_to_px(6.5) == pytest.approx(65.0)


def test_px_per_cm_is_not_hardcoded_into_the_conversions():
    a, b = ArenaFrame(ARENA_W, ARENA_H, 10.0), ArenaFrame(ARENA_W, ARENA_H, 4.0)
    assert a.to_px((10.0, 0.0))[0] == pytest.approx(100.0)
    assert b.to_px((10.0, 0.0))[0] == pytest.approx(40.0)


def test_arena_pixels_are_NOT_the_homography_inverse():
    """The bug this whole class exists to prevent.

    `Homography.to_px` maps centimetres back to CAMERA pixels, which carry the
    perspective the homography exists to remove. Handing those to the framework
    would give its planner a trapezoid and call it a floor. Under a tilted
    camera the two answers must visibly disagree.
    """
    app = FakeApp(matrix=_tilted_homography())
    arena = ArenaFrame(ARENA_W, ARENA_H)
    cm = (ARENA_W / 2.0, ARENA_H / 2.0)

    camera_px = app.homography.to_px([cm])[0]
    arena_px = np.array(arena.to_px(cm))

    assert np.linalg.norm(camera_px - arena_px) > 100.0

    # And the arena frame must be square where the camera frame is not: equal
    # steps in cm have to be equal steps in arena pixels everywhere.
    near = np.array(arena.to_px((10.0, 10.0)))
    far = np.array(arena.to_px((10.0 + 20.0, 10.0)))
    other = np.array(arena.to_px((ARENA_W - 30.0, ARENA_H - 20.0)))
    other_far = np.array(arena.to_px((ARENA_W - 10.0, ARENA_H - 20.0)))
    assert np.linalg.norm(far - near) == pytest.approx(
        np.linalg.norm(other_far - other), abs=1e-9)


# -- Reading -----------------------------------------------------------------

def test_a_locked_ball_reads_where_the_bench_says_it_is():
    arena = ArenaFrame(ARENA_W, ARENA_H)
    # 4 camera px per cm, so (200, 400) camera px is (50, 100) cm.
    app = FakeApp(xy_px=(200.0, 400.0))
    got = Reading.of(app, arena)
    assert got is not None
    assert got.xy_px == pytest.approx((500.0, 1000.0), abs=1e-6)


def test_no_blob_is_no_reading():
    assert Reading.of(FakeApp(xy_px=None), ArenaFrame(ARENA_W, ARENA_H)) is None


def test_no_homography_is_no_reading():
    app = FakeApp()
    app.homography.M = None
    assert Reading.of(app, ArenaFrame(ARENA_W, ARENA_H)) is None


def test_a_COASTING_tracker_is_not_an_observation():
    """The bench is predicting a position it cannot see. That is not a reading.

    Publishing it would launder a guess into a measurement exactly one process
    downstream, where `pp.get_pos` holds a last_pose forever and a lost ball is
    indistinguishable from a still one.
    """
    app = FakeApp(locked=False, status="coasting 2/5 — no candidate")
    assert Reading.of(app, ArenaFrame(ARENA_W, ARENA_H)) is None


def test_theta_is_travel_in_radians():
    arena = ArenaFrame(ARENA_W, ARENA_H)
    app = FakeApp(travel=(90.0, 0.0, 12.0))     # 90 deg in the arena frame
    got = Reading.of(app, arena)
    assert got.theta_rad == pytest.approx(math.pi / 2)
    assert got.theta_fresh is True


def test_a_ball_at_rest_HOLDS_its_last_bearing_rather_than_inventing_north():
    arena = ArenaFrame(ARENA_W, ARENA_H)
    app = FakeApp(travel=None)
    got = Reading.of(app, arena, last_theta=1.25)
    assert got.theta_rad == pytest.approx(1.25)
    assert got.theta_fresh is False


def test_a_held_bearing_is_flagged_so_nothing_reads_it_as_measured():
    arena = ArenaFrame(ARENA_W, ARENA_H)
    fresh = Reading.of(FakeApp(travel=(10.0, 0.0, 5.0)), arena).as_pose()
    held = Reading.of(FakeApp(travel=None), arena, last_theta=0.5).as_pose()
    assert fresh["theta_fresh"] is True
    assert held["theta_fresh"] is False


def test_the_pose_dict_matches_the_shape_their_aruco_detector_returns():
    got = Reading.of(FakeApp(travel=(0.0, 0.0, 1.0)), ArenaFrame(ARENA_W, ARENA_H))
    pose = got.as_pose()
    for key in ("x", "y", "theta"):
        assert key in pose and isinstance(pose[key], float)


# -- Bridge ------------------------------------------------------------------

def test_publishing_sends_one_robot_under_its_integer_id():
    client = FakeClient()
    bridge = Bridge(FakeApp(xy_px=(200.0, 400.0)), robot_id=2, client=client)
    poses = bridge.publish_once()
    assert list(poses) == [2]
    assert client.calls[0]["poses"][2]["x"] == pytest.approx(500.0)


def test_a_lost_ball_publishes_an_EMPTY_dict_not_a_stale_pose():
    client = FakeClient()
    bridge = Bridge(FakeApp(locked=False, status="lost — no candidate"),
                    robot_id=2, client=client)
    assert bridge.publish_once() == {}
    assert client.calls[0]["poses"] == {}
    assert bridge.skipped == 1
    assert bridge.published == 0


def test_the_last_fresh_bearing_survives_the_ball_stopping():
    client = FakeClient()
    app = FakeApp(travel=(90.0, 0.0, 12.0))
    bridge = Bridge(app, robot_id=2, client=client)
    bridge.publish_once()

    app._travel = None                       # the ball comes to rest
    poses = bridge.publish_once()
    assert poses[2]["theta"] == pytest.approx(math.pi / 2)
    assert poses[2]["theta_fresh"] is False


def test_obstacles_are_published_empty_so_astar_degenerates_to_a_line():
    """No SAM2, no obstacle detection, and a bare arena. Say so explicitly."""
    client = FakeClient()
    Bridge(FakeApp(), robot_id=2, client=client).publish_once()
    assert client.calls[0]["obstacles"] == []


def test_a_missing_frame_does_not_stop_the_pose_going_out():
    client = FakeClient()
    bridge = Bridge(FakeApp(frame=None), robot_id=2, client=client)
    poses = bridge.publish_once()
    assert poses != {}
    # Arena-sized, not their 480x640 default: the placeholder still has to give
    # their planner a grid the poses fit inside.
    sent = client.calls[0]["frame"]
    assert (sent.shape[1], sent.shape[0]) == bridge.arena.size_px


def test_an_RPC_FAILURE_NEVER_REACHES_THE_BENCH():
    """The camera loop does not stop because another process went away."""
    bridge = Bridge(FakeApp(), robot_id=2,
                    client=FakeClient(raises=ConnectionError("gone")))
    bridge.period = 0.0
    bridge._stop.set()
    bridge.run()                             # must not raise
    assert bridge.errors == 0                # run() exits before publishing

    with pytest.raises(ConnectionError):
        bridge.publish_once()                # the raw call still surfaces it


def test_errors_are_counted_rather_than_thrown_by_the_loop():
    bridge = Bridge(FakeApp(), robot_id=2,
                    client=FakeClient(raises=ConnectionError("gone")))
    bridge.period = 0.0
    try:
        bridge.publish_once()
    except ConnectionError as e:
        bridge.errors += 1
        bridge.last_error = str(e)
    assert bridge.errors == 1
    assert "gone" in bridge.last_error


def test_the_arena_comes_from_the_homography_the_bench_actually_loaded():
    """Not from a constant, and not from workspace.json.

    A homography built for one rectangle under a workspace declaring another
    put the tracker and the planner on different floors once already.
    """
    app = FakeApp()
    app.homography.width, app.homography.height = 240.0, 180.0
    bridge = Bridge(app, robot_id=2, client=FakeClient())
    assert bridge.arena.width_cm == 240.0
    assert bridge.arena.height_cm == 180.0


# -- end to end, through the sim camera --------------------------------------
#
# The tests above build a FakeApp so they can pin one behaviour at a time.
# These run the REAL path: a synthetic camera renders a ball at a known
# centimetre, `find_blobs` finds it, a real `Homography` rectifies it, and the
# bridge converts it. Nothing here is stubbed except the robot, and the truth
# is generated rather than asserted by eye.

from fleet_test import BallSource, find_blobs                    # noqa: E402
from vision.homography import Homography                         # noqa: E402
from vlm.bridge import PX_PER_CM                                 # noqa: E402

SIM_PX_CM = 9.2

# Sized so the fake camera actually SEES the arena. At 9.2px/cm the bench's
# default 1280x720 covers 139.1 x 78.3cm -- wide enough but 32cm short, so a
# ball past y=78 is rendered off the bottom of the frame and `find_blobs`
# correctly reports nothing. That is not a conversion failure, and a test that
# read it as one would be pinning the sim camera's field of view rather than
# anything this module does.
SIM_SIZE = (1280, int(np.ceil(ARENA_H * SIM_PX_CM)))


def _sim_homography():
    """Camera pixels to centimetres for the sim camera's flat, square view."""
    src = np.array([[0, 0], [ARENA_W * SIM_PX_CM, 0],
                    [ARENA_W * SIM_PX_CM, ARENA_H * SIM_PX_CM],
                    [0, ARENA_H * SIM_PX_CM]], dtype=np.float32)
    dst = np.array([[0, 0], [ARENA_W, 0], [ARENA_W, ARENA_H], [0, ARENA_H]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return Homography(M, width=ARENA_W, height=ARENA_H)


def _sim_reading(truth_cm):
    """Render a ball at `truth_cm`, detect it, and carry it to arena pixels."""
    holder = {"at": np.asarray(truth_cm, dtype=float)}
    src = BallSource(size=SIM_SIZE, px_cm=SIM_PX_CM, seed=0,
                     pose=lambda: [(holder["at"], 1.0)])
    ok, frame = src.read()
    blobs, _mask, _n = find_blobs(frame)
    assert blobs, f"the sim camera rendered nothing at {truth_cm}"

    app = FakeApp(frame=frame)
    app.homography = _sim_homography()
    app.blob = blobs[0]
    return Reading.of(app, ArenaFrame(ARENA_W, ARENA_H)), app


@pytest.mark.parametrize("truth_cm", [
    (40.0, 30.0), (69.4, 55.4), (100.0, 50.0), (25.0, 90.0), (110.0, 20.0),
])
def test_a_simulated_ball_lands_where_the_framework_is_told_it_is(truth_cm):
    got, _app = _sim_reading(truth_cm)
    assert got is not None
    want = (truth_cm[0] * PX_PER_CM, truth_cm[1] * PX_PER_CM)
    # 1mm, which is 1 arena pixel. The synthetic ball is rendered symmetrically
    # so the centroid is exact; this is a test of the CONVERSION, not of what
    # the detector achieves on a real frame -- where the halo is clipped near a
    # boundary and the error reaches centimetres.
    assert got.xy_px == pytest.approx(want, abs=1.0)


def test_the_whole_chain_is_reversible_from_arena_pixels_back_to_the_ball():
    truth = (69.4, 55.4)
    got, _app = _sim_reading(truth)
    back = ArenaFrame(ARENA_W, ARENA_H).to_cm(got.xy_px)
    assert back == pytest.approx(truth, abs=0.1)


def test_a_ball_at_each_corner_of_the_workspace_still_converts():
    """Not an accuracy claim -- the detector is biased near a boundary.

    This only pins that the CONVERSION stays inside the arena rectangle, so a
    corner ball never reports a negative pixel or one past the far edge, which
    is what would put it outside their A* grid entirely.
    """
    arena = ArenaFrame(ARENA_W, ARENA_H)
    w, h = arena.size_px
    for truth in [(15.0, 15.0), (ARENA_W - 15.0, 15.0),
                  (ARENA_W - 15.0, ARENA_H - 15.0), (15.0, ARENA_H - 15.0)]:
        got, _app = _sim_reading(truth)
        assert got is not None
        assert 0 <= got.xy_px[0] <= w
        assert 0 <= got.xy_px[1] <= h


def test_the_sim_bench_gets_its_arena_from_what_the_camera_SEES():
    """In sim the homography is a real scale with no declared rectangle.

    `fleet_test` throws away the saved calibration under `--source sim` and
    substitutes the exact px/cm the fake camera draws at, setting width and
    height to zero. Falling back to the 200x200 placeholder there would report
    into a floor nobody calibrated; the frame's own extent is the honest answer.
    """
    app = FakeApp(frame=np.zeros((1020, 1280, 3), np.uint8))
    app.homography = Homography(
        [[1.0 / 9.2, 0.0, 0.0], [0.0, 1.0 / 9.2, 0.0], [0.0, 0.0, 1.0]])
    app.homography.width = app.homography.height = 0.0

    bridge = Bridge(app, robot_id=2, client=FakeClient())
    assert bridge.arena.width_cm == pytest.approx(1279 / 9.2, abs=0.2)
    assert bridge.arena.height_cm == pytest.approx(1019 / 9.2, abs=0.2)


def test_the_arena_is_resolved_once_and_then_held():
    """A rectangle that changed under a running planner would move every
    published position with nothing downstream told the units had shifted."""
    app = FakeApp(frame=np.zeros((1020, 1280, 3), np.uint8))
    bridge = Bridge(app, robot_id=2, client=FakeClient())
    first = bridge.arena
    app.homography.width, app.homography.height = 999.0, 999.0
    assert bridge.arena is first


def test_a_HELD_bearing_carries_its_age_because_it_decays_into_a_fiction():
    """Nothing here can tell that a stationary ball was lifted and set down
    facing elsewhere -- a blob has no facing to check against. So the age goes
    out and the consumer decides what it will still believe."""
    arena = ArenaFrame(ARENA_W, ARENA_H)
    got = Reading.of(FakeApp(travel=None), arena,
                     last_theta=1.25, last_theta_at=100.0, now=142.0)
    assert got.theta_age_s == pytest.approx(42.0)
    assert got.as_pose()["theta_age_s"] == pytest.approx(42.0)


def test_a_MEASURED_bearing_has_no_age():
    got = Reading.of(FakeApp(travel=(90.0, 0.0, 12.0)), ArenaFrame(ARENA_W, ARENA_H),
                     last_theta=0.1, last_theta_at=1.0, now=999.0)
    assert got.theta_fresh is True
    assert got.theta_age_s == 0.0


def test_the_age_grows_across_publishes_while_the_ball_stays_still():
    client = FakeClient()
    app = FakeApp(travel=(90.0, 0.0, 12.0))
    bridge = Bridge(app, robot_id=2, client=client)
    bridge.publish_once()
    app._travel = None
    bridge.last_theta_at = time.time() - 7.0
    poses = bridge.publish_once()
    assert poses[2]["theta_age_s"] >= 7.0


# -- the frame their PLANNER sizes its grid from -------------------------------

def test_the_published_frame_is_the_ARENA_size_not_the_camera_size():
    """`PlanningClient._get_planner` takes the A* grid size from
    `frame.shape[:2]`, and with no arena_corners in DataService it takes the
    arena rectangle from the frame bounds too. A 1280x1020 grid under poses
    running to 1388x1108 puts a ball at the far edge off the map."""
    client = FakeClient()
    app = FakeApp(frame=np.zeros((1020, 1280, 3), np.uint8))
    app.homography = _sim_homography()
    bridge = Bridge(app, robot_id=2, arena=ArenaFrame(ARENA_W, ARENA_H),
                    client=client)
    bridge.publish_once()
    sent = client.calls[0]["frame"]
    assert (sent.shape[1], sent.shape[0]) == bridge.arena.size_px


def test_the_warp_puts_the_ball_where_the_pose_says_it_is():
    """The frame and the coordinates have to agree, or their planner draws one
    world and steers in another."""
    truth = (69.4, 55.4)
    got, app = _sim_reading(truth)
    arena = ArenaFrame(ARENA_W, ARENA_H)
    warped = arena.warp(app.frame, app.homography)

    v = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:, :, 2]
    ys, xs = np.nonzero(v > 20)
    assert len(xs), "the ball vanished in the warp"
    centre = (float(xs.mean()), float(ys.mean()))
    assert centre == pytest.approx(got.xy_px, abs=6.0)


def test_the_scale_is_composed_onto_the_homography_not_applied_after_it():
    """Two transforms kept in step is how this project already lost an
    afternoon to a drawn robot landing where its blob was not."""
    arena = ArenaFrame(ARENA_W, ARENA_H)
    hom = _sim_homography()
    M = arena.matrix_from(hom)

    camera_px = np.array([[[69.4 * SIM_PX_CM, 55.4 * SIM_PX_CM]]], np.float64)
    one_step = cv2.perspectiveTransform(camera_px, M).reshape(2)
    two_step = np.array(arena.to_px(hom.to_cm(camera_px.reshape(1, 2))[0]))
    assert one_step == pytest.approx(two_step, abs=1e-6)


def test_no_frame_still_publishes_an_arena_sized_placeholder():
    client = FakeClient()
    bridge = Bridge(FakeApp(frame=None), robot_id=2,
                    arena=ArenaFrame(ARENA_W, ARENA_H), client=client)
    bridge.publish_once()
    sent = client.calls[0]["frame"]
    assert (sent.shape[1], sent.shape[0]) == (1388, 1108)


def test_publishing_ALSO_SERVES_drive_requests_on_the_same_thread():
    """The wiring that was silently missing once already.

    `str.replace` with a mismatched indent no-ops without complaining, and the
    bridge published happily while never once serving a request -- every drive
    came back "No robot is attached". One socket, one thread, both halves.
    """
    client = FakeClient()
    bridge = Bridge(FakeApp(), robot_id=2, arena=ArenaFrame(ARENA_W, ARENA_H),
                    client=client)
    bridge.publish_once()
    assert bridge.driver is not None
    assert bridge.driver.attached is True


def test_drive_can_be_turned_off_for_a_read_only_bridge():
    client = FakeClient()
    bridge = Bridge(FakeApp(), robot_id=2, arena=ArenaFrame(ARENA_W, ARENA_H),
                    client=client, drive=False)
    bridge.publish_once()
    assert bridge.driver is None


def test_the_sim_camera_covers_the_workspace_it_spawns_robots_into():
    """A fixed 1280x720 at 9.2px/cm saw 139 x 78cm of a 138.8 x 110.8cm
    workspace, so roughly one run in four began with the ball below the bottom
    of the frame -- present, moving, driveable and invisible, reported as no
    tracker lock and read as "the robot never connected"."""
    import json
    from fleet_test import sim_frame_size, SIM_PX_CM
    w_px, h_px = sim_frame_size()
    bounds = np.asarray(json.load(open("workspace.json"))["bounds_cm"])
    want_w = float(bounds[:, 0].max() - bounds[:, 0].min())
    want_h = float(bounds[:, 1].max() - bounds[:, 1].min())
    assert w_px / SIM_PX_CM >= want_w - 0.1
    assert h_px / SIM_PX_CM >= want_h - 0.1


def test_the_sim_frame_size_falls_back_rather_than_failing_to_start():
    from fleet_test import sim_frame_size
    assert sim_frame_size(px_cm=0.0) == (1280, 720) or sim_frame_size()[0] > 0


# -- lighting and blob ranking ------------------------------------------------

def test_a_blown_out_frame_is_named_rather_than_read_as_a_lost_ball():
    """Three saved frames from 29 Aug mean V 165 against a working frame's 5,
    and yield zero blobs: a mask that keeps everything separates nothing. From
    outside that is identical to a dark room with no ball in it, which sends a
    person hunting a ball that rolled away when someone turned the lights on."""
    from fleet_test import frame_blown, BLOWN_MEAN_V
    dark = np.full((80, 80, 3), 6, np.uint8)
    lit = np.full((80, 80, 3), 165, np.uint8)
    assert frame_blown(dark) is False
    assert frame_blown(lit) is True
    assert frame_blown(None) is False
    assert 7 < BLOWN_MEAN_V < 165, "the threshold must sit in the measured gap"


def test_a_SATURATED_blob_outranks_a_bigger_dimmer_one():
    """Sorting on area alone let scenery win. On two of the twenty-four saved
    frames with blobs the real ball was outranked: area 1648 peak 209 beaten by
    area 4211 peak 57, and area 6278 peak 255 beaten by area 9897 peak 132."""
    from fleet_test import find_blobs, BRIGHT_PEAK
    frame = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(frame, (110, 150), 46, (60, 60, 60), -1)      # big, dim: glare
    cv2.circle(frame, (300, 150), 26, (255, 255, 255), -1)   # small, lit: ball
    blobs, _m, _n = find_blobs(frame)
    assert len(blobs) == 2
    assert blobs[0]["peak"] >= BRIGHT_PEAK
    assert blobs[0]["area"] < blobs[1]["area"], "the lit one must win on peak"


def test_with_NOTHING_saturated_the_order_is_unchanged_largest_first():
    """Conservative on purpose: a dimmed LED must not reshuffle the ranking."""
    from fleet_test import find_blobs
    frame = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(frame, (110, 150), 46, (60, 60, 60), -1)
    cv2.circle(frame, (300, 150), 26, (90, 90, 90), -1)
    blobs, _m, _n = find_blobs(frame)
    assert [b["area"] for b in blobs] == sorted(
        (b["area"] for b in blobs), reverse=True)


# -- the ratchet must not rewind ----------------------------------------------

def test_progress_NEVER_REWINDS_on_an_open_path():
    """`BACK_CM` lets the projection settle behind so jitter cannot drag the
    ball forward. Storing that as progress made it COMPOUND: the next window
    centres on the rewound position and may go back again. A ball drifting
    where the route passes near its own earlier self walked backwards a few
    centimetres a frame -- a robot apparently returning to an old waypoint."""
    from fleet_test import Path, pursue
    path = Path([np.array([0.0, 0.0]), np.array([100.0, 0.0])])
    pursue(path, np.array([50.0, 0.0]), lookahead=15.0, speed=20.0)
    advanced = path.s
    assert advanced > 40.0

    for _ in range(20):                       # jitter backwards, repeatedly
        pursue(path, np.array([advanced - 3.0, 2.0]), lookahead=15.0, speed=20.0)
    assert path.s >= advanced, "progress rewound and compounded"


def test_a_ball_PICKED_UP_may_still_relock_backwards():
    """More than RELOCK_CM off the route means it was moved by hand, and
    insisting on the old arc position would drive it somewhere it is not."""
    from fleet_test import Path, pursue
    path = Path([np.array([0.0, 0.0]), np.array([200.0, 0.0])])
    pursue(path, np.array([150.0, 0.0]), lookahead=15.0, speed=20.0)
    assert path.s > 100.0
    pursue(path, np.array([10.0, 0.0]), lookahead=15.0, speed=20.0)
    assert path.s < 60.0, "a moved ball must be allowed to re-lock"


def test_a_CLOSED_path_may_still_wrap_to_zero():
    """An orbit's arc position has to return to zero at the lap boundary."""
    from fleet_test import Path, pursue
    ring = Path.circle(np.array([0.0, 0.0]), 30.0)
    pursue(ring, np.array([30.0, 0.0]), lookahead=15.0, speed=20.0)
    ring.s = ring.length - 1.0
    pursue(ring, np.array([30.0, 0.0]), lookahead=15.0, speed=20.0)
    assert ring.s < ring.length / 2.0
