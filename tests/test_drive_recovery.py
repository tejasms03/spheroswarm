"""Circling is a measurement the bench can take, not an errand for the trainer.

Recorded on the real ball: the tracking check commanded 135 degrees twice and
watched it travel 320.6 and 319.3 — a systematic ~185 degree frame error,
repeatable to within 1.3 degrees. Point-to-point cannot converge through that;
the handoff puts the limit near 60. The bench used to detect the circling and
then tell the trainer to go and run the MOTION battery, which is a poor answer
when it is already holding the robot and needs two short legs to find the
number.

The live estimator cannot rescue this on its own, and that is not a bug in it:
`HeadingEstimator` rejects any sample spanning more than `max_turn` degrees of
yaw, because travel direction measured across a turn means nothing. A robot
going in circles turns constantly, so every sample is rejected and it never
becomes ready. Fed a twenty-second circle it rejects 581 samples and reports no
offset at all. Breaking the circle is what makes the measurement possible.
"""

import numpy as np
import pytest

from fleet.heading import HeadingEstimator

pygame = pytest.importorskip("pygame")

from tests.test_calib import bench, render          # noqa: F401,E402


# -- the premise: why the bench has to intervene --------------------------

def _feed(est, radius_cm, speed=25.0, secs=20.0, dt=1 / 30):
    t = 0.0
    while t < secs:
        if radius_cm is None:
            x, y, course = speed * t, 0.0, 0.0
        else:
            w = speed / radius_cm
            x, y = radius_cm * np.cos(w * t), radius_cm * np.sin(w * t)
            course = np.degrees(w * t) + 90.0
        est.update(t, float(x), float(y), float(course))
        t += dt
    return est


def test_the_estimator_sees_a_straight_run():
    e = _feed(HeadingEstimator(), None)
    assert e.ready is True
    assert e.offset is not None


def test_the_estimator_is_blind_while_the_robot_circles():
    """The reason a circling robot never corrects itself."""
    e = _feed(HeadingEstimator(), 40.0)
    assert e.ready is False
    assert e.offset is None
    assert e.rejected["turning"] > 100, e.rejected


# -- the recovery ---------------------------------------------------------

class FakeCal:
    """Stands in for ActiveCalibration: drives for a few frames, then answers."""

    def __init__(self, offset=185.0, spread=3.0, error=None, frames=3):
        self.offset, self.spread, self.error = offset, spread, error
        self.done = False
        self._left = frames
        self.driven = 0

    def step(self, pos, dt):
        if self._left <= 0:
            self.done = True
            return None
        self._left -= 1
        self.driven += 1
        return np.array([5.0, 0.0])


def _circling(app):
    """Make the controller report circling and keep reporting it.

    `PDController.step` recomputes `orbiting` from the distance history every
    call, so a flag set by hand is gone by the time the bench looks at it.
    These tests are about what the BENCH does once the controller says the
    robot is circling, so the controller is held at that verdict.
    """
    real = app.pd.step

    def step(*a, **k):
        v = real(*a, **k)
        app.pd.orbiting = True
        return v

    app.pd.step = step


def _lose_the_camera(monkeypatch, h):
    """A sim robot is always connected, so losing a fix has to be arranged."""
    monkeypatch.setattr(type(h), "connected", property(lambda self: False))


def _drive(app, target=(120.0, 120.0)):
    from swarm.pd import Point
    app.connect("ONE", "sim")
    app.selected = "ONE"
    h = app.handle
    assert h is not None
    h.pos = np.array([40.0, 40.0])
    app.start_path(Point(np.array(target)))
    assert app.pd is not None, "the drive must have started"
    return h


def test_circling_starts_a_recovery_instead_of_an_errand(bench):
    app = bench
    h = _drive(app)
    assert app.recover is None

    _circling(app)
    app.step_path(1 / 30)

    assert app.recover is not None, "the bench must measure it, not delegate it"
    assert app.recovered == 1
    assert any("re-measure" in t for _, t in app.log), [t for _, t in app.log]


def test_the_recovery_applies_the_measured_error_by_subtracting_it(bench):
    """The measured value is the ERROR between commanded and achieved, so it is
    subtracted from what is in force. Assigning it doubles the fault instead of
    cancelling it — a mistake made once already in this codebase.

    The numbers here are chosen so the two are distinguishable. With an offset
    of 10 and a measured error of 185 they are not: (10 - 185) % 360 is 185,
    which is the measured value itself, and a test using those passes either
    way. That is precisely how the original sign error survived its test.
    """
    app = bench
    h = _drive(app)
    h.heading_offset = 25.0
    app.recover = FakeCal(offset=65.0, spread=3.0)

    for _ in range(8):
        app.step_path(1 / 30)

    assert app.recover is None
    assert h.heading_offset == pytest.approx(320.0)      # 25 - 65, wrapped
    assert h.heading_offset != pytest.approx(65.0), "assigned rather than subtracted"


def test_after_a_recovery_the_robot_goes_where_it_is_told(bench):
    """The question a sign error cannot answer correctly.

    The sim rotates every command by `bias + heading_offset`, exactly as a real
    ball does — the offset is applied on the way out and the world adds its own
    error afterwards. So a correct correction leaves the sum at zero, and the
    ball travels along the vector it was handed. Doubling the fault sends it 105
    degrees away, and no assertion about a stored number is needed to see it.
    """
    app = bench
    h = _drive(app)
    h.randomize = False
    h.bias = np.radians(40.0)               # the ball's own frame error
    h.drift_rate = 0.0
    h._drift = 0.0
    h.slip = 0.0
    h.heading_offset = 25.0

    # What two honest legs would measure: the total still outstanding.
    app.recover = FakeCal(offset=40.0 + 25.0, spread=2.0)
    for _ in range(8):
        app.step_path(1 / 30)
    assert app.recover is None

    h.pos = np.array([60.0, 60.0])
    h.vel = np.zeros(2)
    start = h.pos.copy()
    h.set_velocity(np.array([20.0, 0.0]))   # due +x
    for _ in range(60):
        h.step(1 / 30)

    went = h.pos - start
    course = np.degrees(np.arctan2(went[1], went[0])) % 360.0
    assert np.linalg.norm(went) > 3.0, "it has to actually move for this to mean anything"
    assert min(course, 360.0 - course) < 8.0, (
        f"told to go +x, travelled {course:.0f}deg — the correction did not cancel")


def test_the_corrected_offset_reaches_the_roster(bench):
    app = bench
    h = _drive(app)
    h.heading_offset = 0.0
    app.recover = FakeCal(offset=185.0)
    for _ in range(8):
        app.step_path(1 / 30)

    entry = app.roster.by_code(h.code)
    assert entry.heading_offset == pytest.approx(h.heading_offset)


def test_the_drive_continues_after_a_recovery(bench):
    app = bench
    _drive(app)
    app.recover = FakeCal(offset=185.0)
    for _ in range(8):
        app.step_path(1 / 30)

    assert app.path is not None, "the drive must resume, not be abandoned"
    assert app.pd is not None
    assert app.pd.orbiting is False, "the orbit history belongs to the old frame"


def test_the_recovery_owns_the_robot_while_it_runs(bench):
    """The PD controller must not be fighting the calibration legs."""
    app = bench
    h = _drive(app)
    cal = FakeCal(frames=5)
    app.recover = cal

    app.step_path(1 / 30)
    assert cal.driven == 1
    assert app.trail == [], "the drive's trail must not grow during a recovery"


def test_legs_that_disagree_are_called_out_rather_than_applied_quietly(bench):
    """Legs that disagree mean slipping, or a tracker on the wrong ball.
    Neither is fixed by rotating a frame, and both look like a big offset."""
    app = bench
    _drive(app)
    app.recover = FakeCal(offset=185.0, spread=48.0)
    for _ in range(8):
        app.step_path(1 / 30)

    assert any("disagreed" in t for _, t in app.log), [t for _, t in app.log]


def test_a_failed_recovery_stops_the_drive_and_says_why(bench):
    app = bench
    _drive(app)
    app.recover = FakeCal(error="not enough clear space to calibrate")
    for _ in range(8):
        app.step_path(1 / 30)

    assert app.path is None, "a drive that cannot be corrected must not continue"
    assert any("clear space" in t for _, t in app.log), [t for _, t in app.log]


def test_it_gives_up_rather_than_correcting_forever(bench):
    """Past a couple of corrections the aim frame is not what is wrong, and
    saying so beats rotating the frame round the compass."""
    app = bench
    _drive(app)
    app.recovered = app.MAX_RECOVERIES
    _circling(app)

    app.step_path(1 / 30)

    assert app.recover is None
    assert app.path is None
    assert any("BLOBS" in t for _, t in app.log), [t for _, t in app.log]


def test_a_lost_fix_pauses_the_recovery_rather_than_ruining_it(bench, monkeypatch):
    """Calibration legs are measured off camera positions. Continuing to drive
    without a fix would record a leg nobody watched."""
    app = bench
    h = _drive(app)
    cal = FakeCal(frames=5)
    app.recover = cal
    _lose_the_camera(monkeypatch, h)

    assert h.connected is False
    app.step_path(1 / 30)

    assert cal.driven == 0, "no leg may be driven blind"
    assert app.recover is cal, "and the recovery waits rather than failing"
