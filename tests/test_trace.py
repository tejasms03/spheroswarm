"""Drawing a path, compiling it to roll commands, and driving it.

Two pilots are on trial here and they fail in opposite directions, which is
what most of these tests are pinning. An open-loop `Program` is exactly as good
as the model it was compiled against — perfect when the ball has been measured,
and wrong by whatever the calibration is wrong by when it has not. A
`RollFollower` does not care what it believes because it can see, but it pays
for that in radio traffic and cannot go faster than its fixes arrive.
"""

import math

import numpy as np
import pytest

from fleet.sim_handle import SimRobot
from swarm.pd import Polyline, tracking_error
from swarm.trace import (Camera, Drive, Program, RollFollower, RollPlant,
                         SpeedMap, Step, bearing, compile_path, format_plan,
                         heading_vector, run, simplify, steps_from_log, wrap180)
from workspace.space import Workspace

MEASURED = {"tau": 0.84, "latency_s": 1 / 30.0, "gain": 0.69}
DEAD_S = 1 / 30.0
RECT = [[20, 20], [110, 20], [110, 85], [20, 85]]


@pytest.fixture
def arena():
    return Workspace(bounds_cm=[[0, 0], [200, 0], [200, 160], [0, 160]])


def ball(arena, pos=(20, 20), bias_deg=0.0, seed=3):
    r = SimRobot("Syrax", "SYRX", "cyan", workspace=arena, pos=list(pos),
                 seed=seed, randomize=False, motion=dict(MEASURED))
    r.bias = math.radians(bias_deg)
    return r


def drive(arena, points, kind="plan", speed=20.0, bias_deg=0.0, calibrated=True,
          loop=False, deadband=18, **kw):
    """One run of one pilot, from the start of the shape."""
    path = Polyline(simplify(points, 1.0), speed=speed, loop=loop)
    r = ball(arena, pos=path.points[0], bias_deg=bias_deg)
    belief = (SpeedMap.truth_for(r, deadband) if calibrated
              else SpeedMap(min_moving_byte=deadband))
    if kind == "plan":
        steps, _ = compile_path(path, speed=speed, speeds=belief, dead_s=DEAD_S)
        pilot = Program(steps)
    else:
        pilot = RollFollower(path, speed=speed, lookahead=18.0, speeds=belief,
                             stop_s=MEASURED["tau"] + DEAD_S,
                             laps=1 if loop else None, **kw)
    return run(path, r, pilot, speeds=SpeedMap(min_moving_byte=deadband),
               camera=Camera(sigma_cm=0.35, seed=1), timeout_s=120)


# -- the shape -----------------------------------------------------------

def test_arclength_runs_along_the_shape_not_between_its_ends():
    p = Polyline([[0, 0], [100, 0], [100, 50]])
    assert p.length == pytest.approx(150.0)
    point, tangent = p.point_at(120.0)
    assert point == pytest.approx([100.0, 20.0])
    assert tangent == pytest.approx([0.0, 1.0])


def test_a_point_off_the_path_projects_onto_the_nearest_place_on_it():
    p = Polyline([[0, 0], [100, 0]])
    assert p.project([50, 12]) == pytest.approx(50.0)
    assert p.project([-30, 0]) == pytest.approx(0.0)
    assert p.project([300, 0]) == pytest.approx(100.0)


def test_projection_does_not_jump_to_the_other_branch_of_a_crossing():
    """A figure eight passes within a centimetre of itself at the middle.

    Without a forward window the nearest point there belongs to the other
    branch, and a follower takes the shortcut rather than the shape.
    """
    cross = Polyline([[0, 0], [50, 50], [100, 0], [100, 100], [50, 50], [0, 100]])
    near_middle = [51.0, 50.0]
    early = cross.project(near_middle, from_s=60.0, window=40.0)
    assert early == pytest.approx(cross.project(near_middle), abs=1.0)
    # Coming round the second time, the same point must read as further along.
    late = cross.project(near_middle, from_s=290.0, window=40.0)
    assert late > 300.0


def test_simplify_drops_the_wobble_and_keeps_the_corners():
    rng = np.random.default_rng(0)
    line = [[x, 0.2 * rng.normal()] for x in np.linspace(0, 100, 400)]
    corner = line + [[100, y] for y in np.linspace(2, 60, 200)]
    out = simplify(corner, tol_cm=1.0)
    assert len(out) <= 6, "a straight line and one corner is not 600 points"
    assert len(out) >= 3
    # And the shape survives it.
    err = tracking_error(Polyline(out), np.asarray(corner, dtype=float))
    assert err["max_cm"] < 1.5


# -- the units -----------------------------------------------------------

def test_a_speed_under_the_deadband_is_lifted_to_it_or_refused():
    speeds = SpeedMap(min_moving_byte=18)
    assert speeds.byte_for(0.0) == 0, "nothing asked for is nothing sent"
    assert speeds.byte_for(1.0) == 18, "a crawl becomes the slowest real speed"
    assert speeds.cm_s_for(10) == 0.0, "under the deadband the motors do nothing"
    assert speeds.cm_s_for(18) > 0.0


def test_the_believed_speed_map_and_the_true_one_can_differ():
    class Ball:
        gain = 0.69
    nominal, true = SpeedMap(), SpeedMap.truth_for(Ball())
    assert nominal.cm_s_for(85) == pytest.approx(20.0, abs=0.1)
    assert true.cm_s_for(85) == pytest.approx(13.8, abs=0.2)
    # Which is the entire sim-to-real gap on this ball, in one number.
    assert true.byte_for(20.0) > nominal.byte_for(20.0)


def test_a_heading_and_its_vector_round_trip():
    for deg in (0.0, 37.5, 90.0, 180.0, 271.0):
        assert bearing(heading_vector(deg)) == pytest.approx(deg, abs=1e-6)


# -- the list ------------------------------------------------------------

def test_a_drawn_shape_compiles_to_alternating_turns_and_legs():
    path = Polyline(RECT, speed=20.0)
    steps, summary = compile_path(path, speed=20.0,
                                  speeds=SpeedMap(min_moving_byte=18),
                                  dead_s=DEAD_S)
    kinds = [s.kind for s in steps]
    assert kinds == ["yaw", "roll", "yaw", "roll", "yaw", "roll"], \
        "three legs at right angles is three turns and three drives"
    assert all(s.seconds > 0 for s in steps)
    assert all(s.byte == 0 for s in steps if s.kind == "yaw"), \
        "a yaw is a roll at speed zero — that is how a Sphero turns on the spot"
    assert summary["distance_cm"] == pytest.approx(path.length, abs=1.0)
    assert "roll" in format_plan(steps, summary)


def test_a_gentle_bend_is_taken_while_moving_but_a_corner_is_not():
    gentle = Polyline([[0, 0], [100, 0], [190, 30]])       # ~18 degrees
    sharp = Polyline([[0, 0], [100, 0], [100, 90]])        # 90 degrees
    speeds = SpeedMap(min_moving_byte=18)
    bend, _ = compile_path(gentle, speeds=speeds, dead_s=DEAD_S)
    corner, _ = compile_path(sharp, speeds=speeds, dead_s=DEAD_S)
    assert [s.kind for s in bend].count("yaw") == 1, "only the initial aim"
    assert [s.kind for s in corner].count("yaw") == 2, "the corner earns one"


def test_a_leg_is_longer_than_distance_over_speed():
    """Because the ball spends the dead time not moving at all."""
    path = Polyline([[0, 0], [100, 0]], speed=20.0)
    steps, _ = compile_path(path, speed=20.0, speeds=SpeedMap(), dead_s=0.43)
    leg = [s for s in steps if s.kind == "roll"][0]
    assert leg.seconds == pytest.approx(100.0 / 20.0 + 0.43, abs=0.02)


def test_a_speed_under_the_deadband_becomes_the_slowest_the_ball_can_do():
    """There is no slower. Asking politely for 1cm/s gets a ball that sits
    there, so the plan asks for the deadband and says that it did."""
    path = Polyline(RECT, speed=1.0)
    steps, summary = compile_path(path, speed=1.0,
                                  speeds=SpeedMap(min_moving_byte=200),
                                  dead_s=DEAD_S)
    assert steps, "a plan that refuses to move is not a plan"
    assert summary["byte"] == 200
    assert summary["speed_cm_s"] > 1.0
    assert "deadband" in summary["note"]
    # And the durations are for the speed it will REALLY go, not the one asked
    # for, or every leg would be fifty times too long.
    leg = [s for s in steps if s.kind == "roll"][0]
    assert leg.seconds == pytest.approx(leg.distance / summary["speed_cm_s"]
                                        + DEAD_S, abs=0.05)


def test_a_plan_at_no_speed_at_all_is_refused():
    path = Polyline(RECT, speed=0.0)
    steps, summary = compile_path(path, speed=0.0, speeds=SpeedMap(),
                                  dead_s=DEAD_S)
    assert steps == []
    assert "deadband" in summary["error"]


# -- open loop -----------------------------------------------------------

def test_a_plan_compiled_against_a_measured_ball_lands_on_the_end(arena):
    d = drive(arena, RECT, kind="plan", calibrated=True)
    assert d.score()["finished_cm"] < 12.0
    assert d.score()["rms_cm"] < 6.0


def test_the_same_plan_falls_short_when_nobody_measured_the_speed(arena):
    """The classic. A byte means less than the datasheet says, and open loop
    has no way to find that out — so the whole shape comes out small."""
    measured = drive(arena, RECT, kind="plan", calibrated=True).score()
    guessed = drive(arena, RECT, kind="plan", calibrated=False).score()
    assert guessed["finished_cm"] > measured["finished_cm"] + 8.0
    assert guessed["rms_cm"] > measured["rms_cm"] + 3.0


def test_an_open_loop_plan_cannot_recover_from_an_aim_error(arena):
    """Every leg leaves at the wrong angle and nothing ever notices."""
    straight = drive(arena, RECT, kind="plan", bias_deg=0.0).score()
    skewed = drive(arena, RECT, kind="plan", bias_deg=15.0).score()
    assert skewed["rms_cm"] > straight["rms_cm"] + 4.0


def test_the_camera_pilot_survives_the_aim_error_that_ruins_the_plan(arena):
    planned = drive(arena, RECT, kind="plan", bias_deg=15.0).score()
    tracked = drive(arena, RECT, kind="track", bias_deg=15.0).score()
    assert tracked["rms_cm"] < planned["rms_cm"]
    assert tracked["rms_cm"] < 6.0


def test_the_plan_costs_a_fraction_of_the_radio_traffic(arena):
    planned = drive(arena, RECT, kind="plan").score()
    tracked = drive(arena, RECT, kind="track").score()
    assert planned["commands"] * 3 < tracked["commands"], \
        "open loop is a handful of commands; closed loop is a stream"


# -- the plant -----------------------------------------------------------

def test_the_ball_turns_at_a_finite_rate(arena):
    plant = RollPlant(ball(arena), speeds=SpeedMap(min_moving_byte=18),
                      yaw_rate=180.0)
    plant.roll(180.0, 100)
    plant.step(0.1)
    assert plant.turning > 100.0, "half a turn does not happen in a tenth of a second"
    for _ in range(30):
        plant.step(1 / 30.0)
    assert plant.turning == pytest.approx(0.0, abs=1.0)


def test_a_byte_below_the_deadband_moves_the_ball_nowhere(arena):
    plant = RollPlant(ball(arena), speeds=SpeedMap(min_moving_byte=18))
    start = np.array(plant.pos, dtype=float)
    plant.roll(90.0, 10)
    for _ in range(60):
        plant.step(1 / 30.0)
    assert float(np.linalg.norm(plant.pos - start)) < 0.5


def test_the_ball_coasts_after_the_last_command(arena):
    """Which is why a run that stops stepping at the last command lies."""
    plant = RollPlant(ball(arena), speeds=SpeedMap(min_moving_byte=18))
    plant.roll(90.0, 123)
    for _ in range(120):
        plant.step(1 / 30.0)
    stopped_at = np.array(plant.pos, dtype=float)
    plant.roll(90.0, 0)
    for _ in range(90):
        plant.step(1 / 30.0)
    assert float(np.linalg.norm(plant.pos - stopped_at)) > 5.0


# -- regressions ---------------------------------------------------------

def test_a_closed_loop_stops_after_one_lap(arena):
    """The seam bug: a projection landing exactly on the start read as a lap
    already completed, so the ball drove a second one looking for a finish
    line it had gone past."""
    circle = [[70 + 35 * math.cos(a), 70 + 35 * math.sin(a)]
              for a in np.linspace(0, 2 * math.pi, 40)]
    d = drive(arena, circle, kind="track", loop=True)
    assert d.pilot.laps == 1
    assert d.t < 2.5 * d.path.length / 20.0, "one lap, not two"
    assert d.score()["finished_cm"] < 20.0


def test_an_arrived_follower_stops_talking(arena):
    """It said stop once. Repeating it every frame is airtime, not control."""
    d = drive(arena, [[20, 20], [100, 20]], kind="track")
    assert d.pilot.done
    quiet = d.pilot.step(np.array([100.0, 20.0]), 1 / 30.0)
    assert quiet is None


def test_the_lead_in_is_part_of_the_plan(arena):
    """A ball is wherever the last run left it, and the list has to say so."""
    path = Polyline(RECT, speed=20.0)
    steps, _ = compile_path(path, speed=20.0, speeds=SpeedMap(min_moving_byte=18),
                            dead_s=DEAD_S, start_pos=[80.0, 120.0])
    assert len(steps) > 6
    assert steps[0].kind == "yaw"


def test_what_was_sent_can_be_read_back_as_a_list(arena):
    d = drive(arena, RECT, kind="track")
    steps = steps_from_log(d.plant.log, SpeedMap(min_moving_byte=18),
                           until=d.plant.t)
    assert len(steps) == len(d.plant.log)
    assert all(isinstance(s, Step) for s in steps)
    assert sum(s.seconds for s in steps) == pytest.approx(d.plant.t, abs=0.2)


def test_scoring_starts_where_the_ball_joined_the_path(arena):
    """Driving TO a shape is not driving it, and counting the lead-in makes a
    run look worse the further away the ball happened to start."""
    path = Polyline(RECT, speed=20.0)
    r = ball(arena, pos=(150, 140))
    pilot = RollFollower(path, speed=20.0, lookahead=18.0,
                         speeds=SpeedMap.truth_for(r, 18),
                         stop_s=MEASURED["tau"] + DEAD_S)
    d = run(path, r, pilot, speeds=SpeedMap(min_moving_byte=18),
            camera=Camera(sigma_cm=0.35, seed=1), timeout_s=120)
    assert d.score()["joined"]
    assert d.joined_at > 0
    assert d.score()["rms_cm"] < 6.0


# -- the corridor controller ---------------------------------------------

def track(**kw):
    kw.setdefault("speed", 20.0)
    kw.setdefault("arrive_cm", 6.0)
    from swarm.trace import TrackToPoint
    return TrackToPoint(**kw)


def test_the_corridor_controller_hands_back_a_velocity(arena):
    """Because that is what every handle and every drive loop here speaks,
    however roll-shaped the thing underneath is."""
    c = track()
    v = c.step([0.0, 0.0], [0.0, 0.0], [100.0, 0.0])
    assert v.shape == (2,)
    assert float(np.linalg.norm(v)) == pytest.approx(20.0, abs=1.0)
    assert v[0] > 0 and abs(v[1]) < 1.0, "pointing at the target"


def test_it_reports_the_bearing_it_committed_to_in_the_frame_watch_aim_reads():
    """Maths convention, anticlockwise, x-right — NOT a Sphero heading. The
    bench compares this against travel it measures with atan2(dy, dx), and the
    two conventions run opposite ways."""
    c = track()
    c.step([0.0, 0.0], [0.0, 0.0], [0.0, 100.0])     # straight down the screen
    # +90, not -90: the arena's y runs DOWN, and watch_aim reads travel with
    # atan2(dy, dx) in that same frame. Getting this backwards would have the
    # bench correcting every offset by twice the error, in the wrong direction.
    assert c.aim == pytest.approx(90.0, abs=2.0)
    c = track()
    c.step([0.0, 0.0], [0.0, 0.0], [100.0, 0.0])     # straight along +x
    assert c.aim == pytest.approx(0.0, abs=2.0)


def test_arriving_latches_until_the_error_grows_past_the_release():
    c = track(arrive_cm=6.0)
    assert not float(np.linalg.norm(c.step([0, 0], [0, 0], [100, 0]))) == 0.0
    stopped = c.step([98.0, 0.0], [0.0, 0.0], [100.0, 0.0])
    assert float(np.linalg.norm(stopped)) == 0.0
    assert c.holding and c.arrived
    # Just outside the circle but inside the release: still parked.
    assert float(np.linalg.norm(c.step([92.0, 0.0], [0, 0], [100, 0]))) == 0.0
    # Well outside: drives again.
    assert float(np.linalg.norm(c.step([80.0, 0.0], [0, 0], [100, 0]))) > 0.0


def test_a_setpoint_that_moves_a_little_keeps_the_corridor():
    c = track()
    c.step([0, 0], [0, 0], [100.0, 0.0])
    first = c.corridor
    c.step([5, 0], [20, 0], [102.0, 0.0])
    assert c.corridor is first, "two centimetres is not a new route"
    c.step([10, 0], [20, 0], [100.0, 40.0])
    assert c.corridor is not first


def test_re_laying_the_corridor_does_not_reset_the_rate_limiter():
    """A moving setpoint re-lays the route every few centimetres, and a fresh
    follower each time would think it had never spoken — which is a command
    per re-lay, on a link that has no room for them."""
    c = track(cmd_hz=4.0)
    c.step([0, 0], [0, 0], [100.0, 0.0])
    assert c.commands == 1
    for i in range(10):
        c.step([float(i), 0.0], [20.0, 0.0], [100.0, 10.0 * i], dt=1 / 30.0)
    assert c.commands <= 3, "ten re-lays inside a second is not ten commands"


def test_it_never_claims_to_be_circling():
    """The bench's aim recovery hangs off that flag, and a controller that
    converges onto a line cannot orbit a point."""
    c = track()
    for i in range(200):
        c.step([float(i) * 0.5, 0.0], [20.0, 0.0], [100.0, 0.0], dt=1 / 30.0)
        assert c.orbiting is False
