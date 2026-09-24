"""Two to four balls running jobs at the same time.

The bench has always been able to drive any one ball well. What it could not
do was drive two, because a run lived in one set of attributes on the app and
a second run would have landed on top of the first. These tests are about the
slot per ball that fixes that, and they check the thing that matters: that
running four balls does not change what any one of them does.

There is no traffic rule here on purpose. Balls in these tests are given jobs
that do not cross, because keeping them apart is a separate layer that does
not exist yet.
"""
import numpy as np
import pytest

from tests.test_taillight import (  # the doubles the bench tests already use
    BenchBall, Clock, app, fleet, lab, no_calib, px_at, seat_ball)

pytestmark = pytest.mark.usefixtures("no_calib")

ARENA_W, ARENA_H = 138.8, 110.8


@pytest.fixture(autouse=True)
def arena(app):
    """A calibrated floor, so cm and pixels mean something in these tests."""
    from vision.homography import Homography
    h = Homography()
    h.set_rect([(100, 80), (540, 80), (540, 420), (100, 420)],
               ARENA_W, ARENA_H)
    app.lab.hom = h
    return h


def seat(app, name, at_cm, facing=0.0, **kw):
    """Add one more ball without disturbing the ones already seated."""
    ball = BenchBall(ble_name=name, color="white", **kw)
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.lab.assigned[name] = "white"
    at = px_at(app, *at_cm)
    app.lab.tracks.assign(name, {"centre": tuple(at), "group": [
        {"x": at[0], "y": at[1], "area": 9.0, "peak": 255.0},
        {"x": at[0] + 8, "y": at[1], "area": 30.0, "peak": 255.0}]})
    app.saved_offset = lambda n: (0.0, "test")
    app.steer_sign.pop(name, None)
    app._shot = ((0, 0), 1.0)
    # Facing is asked for by name once there is more than one ball to ask
    # about; a single-ball lambda would answer for whichever was seated last.
    app.arena_heading = lambda n: app.lab.robots[n].facing
    return ball


def goto(app, name, x_cm, y_cm):
    """Give one named ball a p2p, the way a click or the agent would."""
    app.driving = name
    app.start_p2p()
    app.p2p_click(px_at(app, x_cm, y_cm))
    assert app.p2p is not None, f"{name} did not take the job"


def run(app, seconds, balls):
    """Tick every ball every frame, advancing each one's simulated floor."""
    import taillight as T
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        for _ in range(int(seconds * 30)):
            app.job_ticks()
            for name, ball in balls.items():
                track = app.lab.tracks.by_name[name]
                track.centre = ball.advance(
                    1 / 30.0, app.lab.hom, track.centre)
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf


def where(app, name):
    px = app.lab.tracks.by_name[name].centre
    return np.asarray(app.lab.hom.to_cm([list(px)]), float).ravel()[:2]


# -- the slots themselves ----------------------------------------------------

def test_a_second_ball_does_not_take_the_first_balls_job(app, fleet):
    a = seat(app, "SK-A", (30, 30))
    b = seat(app, "SK-B", (110, 80))
    goto(app, "SK-A", 30, 80)
    a_job = app.ball_slot("SK-A")["p2p"]
    assert a_job is not None

    goto(app, "SK-B", 110, 30)
    # Selecting B put B's job in the live attributes; A's is still A's.
    assert app.ball_slot("SK-A")["p2p"] is a_job
    assert app.ball_slot("SK-B")["p2p"] is not a_job
    assert sorted(app.busy_balls()) == ["SK-A", "SK-B"]


def test_selecting_a_ball_brings_its_own_job_back(app, fleet):
    seat(app, "SK-A", (30, 30))
    seat(app, "SK-B", (110, 80))
    goto(app, "SK-A", 30, 80)
    a_job = app.p2p
    goto(app, "SK-B", 110, 30)
    b_job = app.p2p
    assert a_job is not b_job

    app.driving = "SK-A"
    assert app.p2p is a_job
    app.driving = "SK-B"
    assert app.p2p is b_job


def test_a_job_started_before_anything_was_selected_is_not_lost(app, fleet):
    # `driving` is None until a person presses Tab or the agent names a ball,
    # and a job can be started before that. It must not be filed under nobody.
    seat(app, "SK-A", (30, 30))
    app._driving = None
    app._live = None
    app.start_p2p()
    app.p2p_click(px_at(app, 30, 80))
    job = app.p2p
    assert job is not None

    app.job_ticks()
    assert app.ball_slot("SK-A")["p2p"] is job


def test_an_idle_ball_keeps_an_empty_slot(app, fleet):
    seat(app, "SK-A", (30, 30))
    seat(app, "SK-B", (110, 80))
    goto(app, "SK-A", 30, 80)
    app.job_ticks()
    assert app.busy_balls() == ["SK-A"]
    assert app.ball_slot("SK-B")["p2p"] is None
    assert app.ball_slot("SK-B")["retries"] == 0


# -- actually moving at the same time ----------------------------------------

def test_two_balls_drive_to_their_own_targets_at_once(app, fleet):
    balls = {"SK-A": seat(app, "SK-A", (30, 25)),
             "SK-B": seat(app, "SK-B", (110, 85))}
    goto(app, "SK-A", 30, 85)
    goto(app, "SK-B", 110, 25)

    run(app, 40, balls)

    a, b = where(app, "SK-A"), where(app, "SK-B")
    assert np.hypot(*(a - np.array([30, 85]))) < 12, f"SK-A stopped at {a}"
    assert np.hypot(*(b - np.array([110, 25]))) < 12, f"SK-B stopped at {b}"


def test_four_balls_all_get_somewhere(app, fleet):
    starts = {"SK-A": (25, 25), "SK-B": (115, 25),
              "SK-C": (25, 85), "SK-D": (115, 85)}
    ends = {"SK-A": (25, 85), "SK-B": (115, 85),
            "SK-C": (25, 25), "SK-D": (115, 25)}
    balls = {n: seat(app, n, p) for n, p in starts.items()}
    for n, (x, y) in ends.items():
        goto(app, n, x, y)
    assert sorted(app.busy_balls()) == sorted(starts)

    run(app, 50, balls)

    for n, target in ends.items():
        got = where(app, n)
        assert np.hypot(*(got - np.array(target))) < 15, f"{n} stopped at {got}"


def test_one_ball_alone_behaves_the_same_as_it_always_did(app, fleet):
    # The point of the slots is that nothing changes for a single ball.
    balls = {"SK-A": seat(app, "SK-A", (30, 25))}
    goto(app, "SK-A", 30, 85)
    run(app, 40, balls)
    got = where(app, "SK-A")
    assert np.hypot(*(got - np.array([30, 85]))) < 12, got


def test_stopping_one_ball_leaves_the_others_running(app, fleet):
    balls = {"SK-A": seat(app, "SK-A", (30, 25)),
             "SK-B": seat(app, "SK-B", (110, 85))}
    goto(app, "SK-A", 30, 85)
    goto(app, "SK-B", 110, 25)

    app.driving = "SK-A"
    app.stop_p2p("stopped")
    app.job_ticks()

    assert app.ball_slot("SK-A")["p2p"] is None
    assert app.ball_slot("SK-B")["p2p"] is not None
    assert app.busy_balls() == ["SK-B"]


def test_the_whole_frame_runs_with_several_balls_going(app, fleet):
    # `job_ticks` on its own is not the bench: the real frame also grabs,
    # tracks, drives and records around it, and the swap has to survive all
    # of that and leave the selected ball's job where the drawing expects it.
    seat(app, "SK-A", (30, 25))
    seat(app, "SK-B", (110, 85))
    goto(app, "SK-A", 30, 85)
    goto(app, "SK-B", 110, 25)
    app.driving = "SK-A"
    a_job = app.p2p

    for _ in range(5):
        app.tick()

    assert app.driving == "SK-A"
    assert app.p2p is a_job, "the selected ball's job must still be live"
    assert sorted(app.busy_balls()) == ["SK-A", "SK-B"]


def test_every_balls_plan_is_drawn_not_just_the_selected_ones(app, fleet):
    import pygame
    import taillight as T

    seat(app, "SK-A", (30, 25))
    seat(app, "SK-B", (110, 85))
    goto(app, "SK-A", 30, 85)
    goto(app, "SK-B", 110, 25)
    app.driving = "SK-A"
    a_job = app.p2p

    drawn = []
    surface = pygame.Surface((T.W, T.H))
    real = app.draw_p2p
    app.draw_p2p = lambda s, px_of: drawn.append(app.drive_target())
    try:
        app.draw_plans(surface, lambda p: (100.0, 100.0))
    finally:
        app.draw_p2p = real

    assert sorted(drawn) == ["SK-A", "SK-B"], drawn
    assert drawn[-1] == "SK-A", "the selected ball is drawn last, on top"
    # Drawing must leave the bench exactly as it found it.
    assert app.driving == "SK-A" and app.p2p is a_job
    assert app.ball_slot("SK-B")["p2p"] is not None


def test_drawing_the_plans_does_not_repeat_the_pick_prompt(app, fleet):
    import pygame
    import taillight as T

    seat(app, "SK-A", (30, 25))
    seat(app, "SK-B", (110, 85))
    goto(app, "SK-B", 110, 25)
    app.driving = "SK-A"
    app.start_p2p()
    assert app.p2p_pick

    picking = []
    real = app.draw_p2p
    app.draw_p2p = lambda s, px_of: picking.append(app.p2p_pick)
    try:
        app.draw_plans(pygame.Surface((T.W, T.H)), lambda p: (10.0, 10.0))
    finally:
        app.draw_p2p = real

    assert picking.count(True) == 1, "the prompt belongs to the person, once"
    assert app.p2p_pick is True, "and the mode survives the drawing"


def test_a_retry_on_one_ball_does_not_count_against_another(app, fleet):
    seat(app, "SK-A", (30, 30))
    seat(app, "SK-B", (110, 80))
    goto(app, "SK-A", 30, 80)
    app.retries = 4                       # A has been round the loop a few times
    goto(app, "SK-B", 110, 30)
    assert app.retries == 0               # B starts its own count
    assert app.ball_slot("SK-A")["retries"] == 4
