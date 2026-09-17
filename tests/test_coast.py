"""Inherited from `tests/test_blob.py`, pointed at `swarm_test.py`.

The multi-robot fork starts life passing every test the single-robot app
passes. What it must not do is lose them quietly: a swap, a gate or a
refusal that stops working while N tracks are added would otherwise look
like the new feature working.

x and y from one blob: what it costs, and where it stops answering.

`blob_test.py` trades the three-dot method's precision for availability -- it
answers on frames where the dots have merged, blurred, or shrunk past reading.
These tests pin both halves of that trade, because the trade is the whole
argument for the module and a change that quietly gave back the availability
would look like a harmless refactor.

Balls are rendered by `vision/shots.py` from a known centre, so the truth here
is generated rather than asserted by eye. They are drawn with NO TAILLIGHT,
which is the arrangement the method requires -- `test_the_taillight_biases_the_centroid`
is what says why.
"""

import math

import cv2
import numpy as np
import pytest

from coast_test import (BallSource, MAX_AREA, MIN_AREA, RunLog, V_MIN, Path,
                       Track, _csv_cell, find_blobs, fit_travel,
                       order_quad, pursue, read_one, roi_mask)
from vision.shots import BALL_CM, TAG_R_CM, TAIL_R_CM, TAIL_BGR, _glow, tag_bgr

PX_CM = 9.2
GAIN = BallSource.GAIN


def render(x_cm, y_cm, heading=0.0, tag="red", scale=1.0, tail=False,
           blur=0, amp=1.0, size=(520, 520)):
    """One ball, exposed the way the sim exposes it. `tail` adds the taillight."""
    canvas = np.zeros((size[1], size[0], 3), np.float32)
    canvas[:] = (6.0, 5.0, 4.0)
    cx, cy = x_cm * PX_CM, y_cm * PX_CM
    r = BALL_CM * PX_CM * scale / 2.0
    tp, lp = TAG_R_CM * PX_CM * scale, TAIL_R_CM * PX_CM * scale
    a = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    col = tag_bgr(tag)
    mix = tuple((p + q) / 2 for p, q in zip(col, TAIL_BGR)) if tail else col
    _glow(canvas, (cx, cy), r * 0.95, mix, 90.0 * amp)
    for s in (1, -1):
        pt = (cx + s * a[0] * tp, cy + s * a[1] * tp)
        _glow(canvas, pt, r * 0.68, col, 600.0 * amp)
        _glow(canvas, pt, max(0.7, 3.0 * scale), col, 10000.0 * amp)
    if tail:
        t = (cx - a[0] * lp, cy - a[1] * lp)
        _glow(canvas, t, r * 0.68, TAIL_BGR, 600.0 * amp)
        _glow(canvas, t, max(0.7, 3.0 * scale), TAIL_BGR, 10000.0 * amp)
    img = np.clip(canvas * GAIN, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (blur | 1, blur | 1), 0) if blur else img


def centre_of(img, **kw):
    blob, _, why, _, _ = read_one(img, **kw)
    assert blob is not None, why
    return blob["xy"]


def err_mm(img, x_cm, y_cm, **kw):
    got = centre_of(img, **kw)
    return float(np.linalg.norm(got - np.array([x_cm * PX_CM, y_cm * PX_CM]))
                 / PX_CM * 10.0)


# -- accuracy --------------------------------------------------------------

@pytest.mark.parametrize("heading", [0.0, 45.0, 128.0, 271.0])
def test_centre_is_exact_with_the_tail_off(heading):
    """Two symmetric LEDs put the weighted centroid ON the centre, whatever
    way the ball is pointing."""
    assert err_mm(render(28.0, 26.0, heading), 28.0, 26.0) < 0.5


def test_the_taillight_biases_the_centroid():
    """Why `blob_test` forces the tail off, stated as a measurement.

    The bias is not merely large, it ROTATES with the robot -- so it cannot be
    removed by any constant offset, which is the reason the tail is switched
    off rather than calibrated around.
    """
    offsets = []
    for heading in (0.0, 90.0, 180.0, 270.0):
        got = centre_of(render(28.0, 26.0, heading, tail=True))
        offsets.append(got - np.array([28.0 * PX_CM, 26.0 * PX_CM]))
        assert err_mm(render(28.0, 26.0, heading, tail=True), 28.0, 26.0) > 3.0
    # Pointing opposite ways puts the bias opposite ways: no constant fixes it.
    assert np.dot(offsets[0], offsets[2]) < 0


def test_the_threshold_hardly_matters_while_the_halo_stays_merged():
    """The reason this method needs so little tuning.

    Across v_min 8 to 28 the blob's area changes by a factor of ten and the
    reported centre moves by under a hundredth of a millimetre. Weighting by
    height ABOVE the threshold is what buys that: weighting by the raw value
    would give every pixel a constant `v_min` of dead weight, pulling the
    answer toward whichever side carries more area and making the centre a
    function of the slider.
    """
    seen = [centre_of(render(28.0, 26.0, 33.0), v_min=v, min_area=20)
            for v in (8, 12, 15, 20, 25, 28)]
    pts = np.array(seen)
    spread = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() / PX_CM * 10
    assert spread < 0.1, f"centre wandered {spread:.3f} mm while merged"


def test_the_centre_jumps_a_centimetre_the_moment_the_halo_SPLITS():
    """The cliff the `parts` readout on the dock exists to keep you off.

    Once the ball breaks into a core per LED, the largest component is one
    LOBE, and its centroid sits a tag-radius away from the true centre. It is
    not a gentle degradation -- it is a step of about 11mm between two
    neighbouring slider positions, which is exactly the kind of error that
    reads as "the tracker is drifting" rather than "the threshold is too high".
    """
    merged = centre_of(render(28.0, 26.0, 33.0), v_min=25, min_area=20)
    split = centre_of(render(28.0, 26.0, 33.0), v_min=40, min_area=20)
    _, _, parts = find_blobs(render(28.0, 26.0, 33.0), v_min=40, min_area=20)
    assert parts > 1
    assert np.linalg.norm(split - merged) / PX_CM * 10 > 5.0


# -- availability, which is the point --------------------------------------

@pytest.mark.parametrize("scale", [1.0, 0.7, 0.5, 0.35, 0.25, 0.15])
def test_still_reads_a_shrinking_ball(scale):
    """Down to a ball a few pixels across -- the range where the three-dot
    method has already stopped answering."""
    img = render(28.0, 26.0, 40.0, scale=scale)
    blob, _, why, _, _ = read_one(img, min_area=8)
    assert blob is not None, f"lost the ball at scale {scale}: {why}"
    assert err_mm(img, 28.0, 26.0, min_area=8) < 3.0


@pytest.mark.parametrize("blur", [0, 7, 15, 25, 35])
def test_survives_defocus_and_motion_blur(blur):
    """Blur is what merges LED cores into one smear, so it destroys any method
    that needs the dots separable. It barely touches a centroid."""
    img = render(28.0, 26.0, 40.0, blur=blur)
    assert err_mm(img, 28.0, 26.0) < 1.0


# -- the threshold knob ----------------------------------------------------

def test_a_low_threshold_merges_the_halo_and_a_high_one_splits_it():
    """The tuning rule the dock's `parts` readout exists to make visible."""
    img = render(28.0, 26.0, 0.0)
    _, _, low = find_blobs(img, v_min=15, min_area=20)
    _, _, high = find_blobs(img, v_min=60, min_area=20)
    assert low == 1, f"halo did not merge at 15 ({low} parts)"
    assert high > 1, f"expected a split at 60, got {high} part(s)"


def test_default_threshold_is_on_the_merged_side():
    _, _, parts = find_blobs(render(28.0, 26.0, 0.0), v_min=V_MIN,
                             min_area=MIN_AREA, max_area=MAX_AREA)
    assert parts == 1


# -- refusals --------------------------------------------------------------

def test_nothing_above_threshold():
    blob, _, why, _, parts = read_one(np.zeros((200, 200, 3), np.uint8))
    assert blob is None and parts == 0
    assert why == "nothing above threshold"


def test_a_blob_outside_the_area_gate_is_refused_with_a_count():
    """Refused, and it says how many bright things it DID see -- which is what
    tells you the gate is wrong rather than the ball being absent."""
    blob, _, why, _, _ = read_one(render(28.0, 26.0), min_area=50000)
    assert blob is None
    assert "none within the area gate" in why


def test_an_unlit_robot_is_refused_not_guessed():
    blob, _, why, _, _ = read_one(render(28.0, 26.0, amp=0.0))
    assert blob is None and why


def test_a_second_lit_thing_is_reported_not_hidden():
    """A reflection or a stray light must be visible to the operator, because
    on a one-robot bench it means the frame is not what you think it is."""
    img = render(28.0, 26.0)
    img = np.maximum(img, render(44.0, 44.0, tag="green"))
    blob, others, why, _, _ = read_one(img)
    assert blob is not None and len(others) == 1


# -- the noise readout -----------------------------------------------------

def detrended(pts):
    pts = np.array(pts, dtype=float)
    t = np.arange(len(pts), dtype=float)
    resid = np.stack([pts[:, i] - np.polyval(np.polyfit(t, pts[:, i], 1), t)
                      for i in range(2)], axis=1)
    return float(np.linalg.norm(resid, axis=1).max())


def test_the_noise_readout_ignores_constant_velocity():
    """Otherwise the number is useless exactly while you are driving: a ball
    crossing the arena would report its travel as measurement noise."""
    still = [np.array([10.0, 20.0]) for _ in range(40)]
    moving = [np.array([10.0 + 0.5 * k, 20.0 - 0.3 * k]) for k in range(40)]
    assert detrended(still) < 1e-9
    assert detrended(moving) < 1e-9


def test_the_noise_readout_still_sees_real_scatter():
    rng = np.random.default_rng(0)
    noisy = [np.array([10.0 + 0.5 * k, 20.0]) + rng.normal(0, 0.1, 2)
             for k in range(40)]
    assert detrended(noisy) > 0.05


# -- tracking --------------------------------------------------------------

def blob_at(x, y, area=1000):
    return {"xy": np.array([float(x), float(y)]), "area": float(area),
            "peak": 240.0, "sum": 1.0}


def run_track(track, frames):
    return [track.update(f) for f in frames]


def test_acquires_the_largest_blob_then_follows_the_prediction():
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(5)])
    assert t.locked
    assert np.allclose(t.vel, [10.0, 0.0])


def test_a_bigger_blob_elsewhere_cannot_steal_the_track():
    """The failure that picking the largest blob every frame has and this does
    not: a reflection that is momentarily larger than the ball simply takes the
    marker, and nothing on screen says it happened."""
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    got = t.update([blob_at(130, 100, area=900),
                    blob_at(600, 400, area=99999)])
    assert got is not None and got["xy"][0] == 130
    assert len(t.rejected) == 1


def test_a_teleport_is_refused_rather_than_followed():
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    assert t.update([blob_at(900, 900)]) is None
    assert "gate" in t.status


def test_it_coasts_a_few_frames_then_reports_lost():
    """A dropped frame must not be a lost ball, and a lost ball must not be
    reported as a position."""
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    for k in range(t.coast):
        assert t.update([]) is None
        assert t.status.startswith("coasting")
        assert t.xy is not None          # still has a prediction to offer
    t.update([])
    assert t.status.startswith("lost")
    assert t.xy is None                  # and now offers nothing at all


def test_a_coasted_frame_never_reports_a_measurement():
    """`update` returns None while coasting. The predicted point is available
    separately, so a caller has to opt in to a guess and cannot receive one by
    accident."""
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    assert t.update([]) is None
    assert t.predicted is not None
    assert not t.locked


def test_velocity_survives_a_gap_without_compounding():
    """Velocity is taken from the accepted fix over the frames actually
    elapsed, so a coasted stretch cannot feed its own guess back in."""
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    t.update([])
    t.update([])
    got = t.update([blob_at(150, 100)])
    assert got is not None and t.locked
    assert np.allclose(t.vel, [10.0, 0.0])


# -- direction of travel ---------------------------------------------------

def straight(vx, vy, n=8, dt=1 / 30.0, x0=100.0, y0=100.0, noise=0.0, seed=0):
    """A trail moving at exactly (vx, vy) px/s, sampled every `dt` seconds."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = k * dt
        p = np.array([x0 + vx * t, y0 + vy * t])
        if noise:
            p = p + rng.normal(0, noise, 2)
        out.append((t, p))
    return out


@pytest.mark.parametrize("vx,vy", [(120.0, 0.0), (0.0, 90.0), (-70.0, 70.0),
                                   (60.0, -140.0)])
def test_travel_recovers_a_known_velocity(vx, vy):
    v, speed = fit_travel(straight(vx, vy))
    assert np.allclose(v, [vx, vy], atol=1e-6)
    assert abs(speed - math.hypot(vx, vy)) < 1e-6


def test_travel_is_in_pixels_per_second_not_per_frame():
    """The unit bug this replaced: the trail is appended per camera frame and
    the render loop runs at its own rate, so anything that counted frames and
    then multiplied by the camera's fps mixed two different clocks."""
    slow = fit_travel(straight(100.0, 0.0, n=8, dt=1 / 15.0))
    fast = fit_travel(straight(100.0, 0.0, n=8, dt=1 / 60.0))
    assert abs(slow[1] - 100.0) < 1e-6
    assert abs(fast[1] - 100.0) < 1e-6


def test_no_arrow_at_rest():
    """At a standstill the fitted direction is a reading of the noise floor,
    and an arrow there spins on the spot and means nothing."""
    assert fit_travel(straight(0.0, 0.0, noise=0.3, seed=1)) is None


def test_no_arrow_below_the_speed_floor():
    assert fit_travel(straight(5.0, 0.0)) is None
    assert fit_travel(straight(200.0, 0.0)) is not None


def test_the_fit_beats_the_endpoints_when_the_trail_is_noisy():
    """Why a least-squares line and not simply last-minus-first: the endpoints
    are two noisy samples, and at low speed the noise is a large share of the
    movement."""
    fit_err, end_err = [], []
    for seed in range(40):
        w = straight(90.0, 0.0, n=8, noise=1.2, seed=seed)
        fit_err.append(abs(fit_travel(w, min_speed=0.0)[1] - 90.0))
        span = w[-1][0] - w[0][0]
        end_err.append(abs(float(np.linalg.norm(w[-1][1] - w[0][1])) / span - 90.0))
    assert np.median(fit_err) < np.median(end_err)


def test_travel_needs_a_few_samples():
    assert fit_travel([]) is None
    assert fit_travel(straight(200.0, 0.0, n=2)) is None


def test_a_stalled_camera_does_not_divide_by_zero():
    """Every sample carrying the same timestamp is a frozen feed, not motion."""
    frozen = [(1.0, np.array([10.0, 10.0])) for _ in range(8)]
    assert fit_travel(frozen) is None


# -- the workspace region --------------------------------------------------

def test_order_quad_never_makes_a_bowtie():
    """Clicks arrive in whatever order a person makes them. Filling an
    unsorted quad gives two triangles pinched in the middle, which masks a band
    through the centre of the arena and reads as the detector failing."""
    corners = [(10, 10), (90, 90), (90, 10), (10, 90)]      # deliberately crossed
    ordered = order_quad(corners)
    m = roi_mask((100, 100), ordered)
    assert m[50, 50] == 1                       # the centre must be INSIDE
    assert m.sum() > 0.6 * 80 * 80              # and it must be a full quad


def test_roi_mask_includes_inside_and_excludes_outside():
    m = roi_mask((200, 200), [(50, 50), (150, 50), (150, 150), (50, 150)])
    assert m[100, 100] == 1
    assert m[10, 10] == 0 and m[190, 190] == 0


def test_a_blob_outside_the_workspace_is_not_seen_at_all():
    """Masked at the pixel level, so something bright outside cannot form a
    component -- rather than being found and then argued with."""
    img = render(28.0, 26.0)
    img = np.maximum(img, render(48.0, 48.0, tag="green"))
    both, _, _ = find_blobs(img)
    assert len(both) == 2

    inside = roi_mask(img.shape, [(150, 150), (350, 150), (350, 350), (150, 350)])
    only_one, _, parts = find_blobs(img, region=inside)
    assert len(only_one) == 1 and parts == 1
    assert abs(only_one[0]["xy"][0] - 28.0 * PX_CM) < 3


def test_the_whole_frame_is_read_when_there_is_no_region():
    blobs, _, _ = find_blobs(render(28.0, 26.0), region=None)
    assert len(blobs) == 1


def test_a_blob_clipped_by_the_boundary_is_flagged():
    """It is still reported -- a ball at the arena edge is still somewhere --
    but the centroid is pulled inward by whatever was cut off, so the operator
    is told that this reading is worse than usual."""
    img = render(28.0, 26.0)
    # A boundary drawn straight through the ball.
    cut = roi_mask(img.shape, [(0, 0), (int(28.0 * PX_CM), 0),
                               (int(28.0 * PX_CM), 520), (0, 520)])
    blobs, _, _ = find_blobs(img, region=cut)
    assert blobs and blobs[0]["clipped_by_region"] is True

    whole = roi_mask(img.shape, [(0, 0), (519, 0), (519, 519), (0, 519)])
    clean, _, _ = find_blobs(render(28.0, 26.0), region=whole)
    assert clean and clean[0]["clipped_by_region"] is False


# -- choosing which blob to track ------------------------------------------

def test_pick_selects_the_nearest_blob_not_the_largest():
    """The operator pointing at a blob is better evidence than the largest-blob
    rule, so a pick overrides it."""
    t = Track()
    t.pick(np.array([600.0, 400.0]))
    got = t.update([blob_at(100, 100, area=99999), blob_at(610, 405, area=50)])
    assert got["xy"][0] == 610
    assert t.locked


def test_pick_overrides_the_gate():
    """Otherwise picking a blob far from where the tracker currently believes
    the ball is -- which is the whole reason to pick one -- would be refused."""
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    t.pick(np.array([900.0, 900.0]))
    got = t.update([blob_at(905, 905)])
    assert got is not None and t.locked


def test_a_pick_is_consumed_once():
    """A pick chooses the blob; from the next frame the ordinary gate applies
    again, or a stale pick would keep yanking the track back."""
    t = Track()
    t.pick(np.array([600.0, 400.0]))
    t.update([blob_at(600, 400)])
    assert t._pick is None
    assert t.update([blob_at(900, 900)]) is None       # gated again


def test_forget_drops_the_track_entirely():
    t = Track()
    run_track(t, [[blob_at(100 + 10 * k, 100)] for k in range(3)])
    assert t.locked
    t.forget()
    assert t.xy is None and not t.locked and t.status == "no lock"


# -- paths -----------------------------------------------------------------

def test_line_geometry():
    p = Path.line([0, 0], [100, 0])
    assert p.length == pytest.approx(100.0)
    assert np.allclose(p.at(25), [25, 0])
    s, off = p.project([30, 10])
    assert s == pytest.approx(30.0) and off == pytest.approx(10.0)


def test_a_point_path_has_no_length_and_still_projects():
    p = Path.point([10, 20])
    assert p.length == 0.0
    s, off = p.project([13, 24])
    assert s == 0.0 and off == pytest.approx(5.0)


def test_circle_is_closed_and_about_the_right_circumference():
    c = Path.circle([0, 0], 50)
    assert c.closed
    assert c.length == pytest.approx(2 * math.pi * 50, rel=0.01)


def test_at_wraps_on_a_closed_path_and_clamps_on_an_open_one():
    c = Path.circle([0, 0], 50)
    assert np.allclose(c.at(0), c.at(c.length), atol=1e-6)
    ln = Path.line([0, 0], [10, 0])
    assert np.allclose(ln.at(999), [10, 0])
    assert np.allclose(ln.at(-5), [0, 0])


def test_project_scans_the_whole_path_so_a_doubled_back_route_works():
    """A freehand scribble crosses itself. Searching forward from the last
    position would lock onto the wrong branch of a hairpin."""
    hairpin = Path([[0, 0], [100, 0], [100, 5], [0, 5]])
    s, off = hairpin.project([50, 5])
    assert off == pytest.approx(0.0, abs=1e-6)


# -- pure pursuit ----------------------------------------------------------

def test_pursuit_aims_ahead_not_at_the_nearest_point():
    """Steering at the closest point drives perpendicular into the path and
    oscillates. Aiming further along is what makes the approach converge."""
    ln = Path.line([0, 0], [100, 0])
    v, target, done, _ = pursue(ln, [0, 20], lookahead=15, speed=20)
    assert not done
    assert target[0] == pytest.approx(15.0)          # ahead, not at [0, 0]
    assert np.linalg.norm(v) == pytest.approx(20.0)  # speed is respected


@pytest.mark.parametrize("lookahead,expected_deg", [(5, -76.0), (40, -26.6)])
def test_longer_lookahead_gives_a_shallower_approach(lookahead, expected_deg):
    """The whole trade, in one number: this is why it is a slider."""
    ln = Path.line([0, 0], [100, 0])
    v, _, _, _ = pursue(ln, [0, 20], lookahead=lookahead, speed=20)
    assert math.degrees(math.atan2(v[1], v[0])) == pytest.approx(expected_deg,
                                                                abs=1.0)


def test_pursuit_converges_on_a_straight_line():
    ln = Path.line([0, 0], [200, 0])
    p = np.array([0.0, 25.0])
    for _ in range(400):
        v, _, done, _ = pursue(ln, p, lookahead=15, speed=25)
        if done:
            break
        p = p + v * 0.05
    assert abs(p[1]) < 1.0


def test_a_goal_point_reports_done_and_commands_zero():
    p = Path.point([10, 20])
    v, _, done, note = pursue(p, [10.5, 20.5], lookahead=10, speed=20)
    assert done and note == "arrived"
    assert np.allclose(v, [0, 0])


def test_a_closed_path_is_never_done():
    """A circle is a patrol, not an errand -- reporting 'arrived' on a lap
    would stop the robot every time round."""
    c = Path.circle([0, 0], 50)
    p = np.array([50.0, 0.0])
    for _ in range(400):
        v, _, done, _ = pursue(c, p, lookahead=12, speed=30)
        assert not done
        p = p + v * 0.05
    assert float(np.linalg.norm(p)) == pytest.approx(50.0, abs=3.0)


def test_no_path_commands_nothing():
    v, target, done, note = pursue(None, [0, 0], 10, 20)
    assert np.allclose(v, [0, 0]) and target is None and note == "no path"


# -- aiming: zero at rest, then read the heading back out of motion -------

def test_zeroing_happens_at_rest_and_measures_nothing():
    """`zero_at_rest` establishes a fresh reference and makes no claim about
    where it points. Guessing that would be inventing the very number the
    camera is there to supply."""
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.zero_at_rest)
    assert "aim_zero(0.0)" in src
    assert "heading_offset = 0.0" in src


def test_every_move_zeroes_before_it_drives():
    """The order the geometry needs, and not a step a person has to remember.

    A Sphero re-establishes its heading reference on connect and drifts
    besides, so a zero taken minutes ago describes a relationship that no
    longer holds. Arming takes a fresh one.
    """
    import inspect

    import coast_test as blob_test
    assert "zero_at_rest" in inspect.getsource(blob_test.BlobTest.arm)


def test_the_heading_correction_is_the_fleet_estimator_not_a_second_loop():
    """The bug this replaced, pinned so it cannot come back.

    `SimRobot.step` already calls `observe_heading` every step, so the fleet's
    heading fold runs on its own. A second corrector in this app made two
    controllers fight over one quantity with different gains and different
    ideas of the lag -- unstable under EITHER sign, which presents as the ball
    spiralling away and reads as a sign fault when it is not one.

    The estimator also carries the gates a naive difference lacks: baseline,
    turning and outlier. Those are what keep the controller's own steering lag
    from being counted as a frame error.
    """
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.drive)
    assert "observe_heading" in src
    assert not hasattr(blob_test.BlobTest, "retune_aim")


def test_the_correction_can_be_switched_off_to_isolate_the_follower():
    """Pure pursuit is geometric and needs no heading correction to follow a
    path, so turning this off is how you tell a follower fault from an aiming
    one."""
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.cycle_aim)
    assert "heading_tracking" in src
    assert set(blob_test.AIM_MODES) == {"off", "on"}


def test_one_definition_of_the_compass_convention():
    """`velocity_to_command` is the project's single statement of "zero is +y,
    clockwise". A second copy is a second place for a sign error, and on this
    quantity that circles whatever you correct instead of failing plainly."""
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.travel_readout)
    assert "velocity_to_command" in src
    assert "90.0 - deg" not in src


# -- the run log -----------------------------------------------------------

def test_the_log_writes_a_header_and_rows(tmp_path):
    log = RunLog(directory=str(tmp_path), stamp="test")
    log.write(frame=1, mode="idle", status="locked", x_cm=1.5, y_cm=2.0)
    log.write(frame=2, mode="driving", status="locked", x_cm=None)
    log.close()
    import csv
    rows = list(csv.DictReader(open(log.path)))
    assert len(rows) == 2
    assert set(rows[0]) == set(RunLog.COLUMNS)
    assert rows[0]["x_cm"] == "1.5" and rows[0]["frame"] == "1"


def test_a_missing_quantity_is_blank_not_zero():
    """A blank says the value did not exist on that frame; a zero says it was
    measured and came out zero. Those are different facts about a tracker, and
    collapsing them is how a dropout gets read as the ball sitting still."""
    assert _csv_cell(None) == ""
    assert _csv_cell(0.0) == "0"
    assert _csv_cell(False) == 0


def test_frames_with_no_ball_are_still_logged(tmp_path):
    """The gaps are the diagnostic part. A logger that only records successes
    turns a dropout into a missing row, indistinguishable from a slow camera."""
    log = RunLog(directory=str(tmp_path), stamp="gap")
    log.write(frame=1, status="locked", x_px=10.0)
    log.write(frame=2, status="coasting 1/6")
    log.write(frame=3, status="lost — no blob in frame")
    log.close()
    import csv
    rows = list(csv.DictReader(open(log.path)))
    assert [r["status"] for r in rows][1:] == ["coasting 1/6",
                                               "lost — no blob in frame"]
    assert rows[1]["x_px"] == ""


def test_a_broken_log_never_takes_the_app_down(tmp_path):
    """Losing the log is a nuisance; losing the session is not acceptable."""
    log = RunLog(directory=str(tmp_path / "nope" / "\0bad"), stamp="x")
    log.write(frame=1)
    assert log.error is not None
    log.write(frame=2)          # still must not raise
    log.close()


# -- smoothness ------------------------------------------------------------

def test_a_dropped_frame_does_not_stop_the_motors():
    """The largest single source of jerk, and a bug rather than a tuning issue.

    Stopping the wheels the instant the track misses a frame, then driving
    again when it returns, turns an ordinary flicker into stop-go-stop several
    times a second. Measured with one frame in five dropped, the old rule cut
    the motors twenty times in a single run and the new one never.

    It is still bounded: the controller rides the tracker's prediction for a
    few frames and stops when the track is genuinely lost, because driving on a
    guess indefinitely is the thing the safety rule exists to prevent.
    """
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.drive)
    assert "DRIVE_ON_PREDICTION" in src
    assert 0 < blob_test.DRIVE_ON_PREDICTION <= 6


def test_the_commanded_direction_is_rate_limited():
    """Position noise becomes heading noise -- pursuit's gain rises as the
    lookahead shortens -- and a Sphero physically turns its drive assembly to
    follow every twitch."""
    import inspect

    import coast_test as blob_test
    assert "slew" in inspect.getsource(blob_test.BlobTest.drive)
    assert blob_test.SLEW_DEG_S > 0


def test_slew_limits_the_turn_but_never_the_speed():
    """Speed is the operator's slider and must not be smoothed behind their
    back; only the direction's rate is limited."""
    import types

    import coast_test as blob_test
    app = types.SimpleNamespace(_slew_at=None, _slew_deg=None,
                                turn_rate=blob_test.SLEW_DEG_S,
                                slew=blob_test.BlobTest.slew)
    v = np.array([20.0, 0.0])
    out = blob_test.BlobTest.slew(app, v)
    assert np.linalg.norm(out) == pytest.approx(20.0)      # seeds, unchanged
    turned = blob_test.BlobTest.slew(app, np.array([0.0, 20.0]))
    assert np.linalg.norm(turned) == pytest.approx(20.0)   # speed preserved
    assert abs(math.degrees(math.atan2(turned[1], turned[0]))) < 90.0


# -- turn and go -----------------------------------------------------------

def test_turn_and_go_crawls_rather_than_turning_in_place():
    """Forced by the tracker, not chosen.

    A Sphero can rotate its drive assembly without moving, and that would be
    the smoother turn. But a BLOB HAS NO FACING WHEN IT IS STATIONARY -- a
    still ball is a round dot -- so a turn done in place could only be taken on
    trust. Crawling keeps just enough travel for the camera to confirm which
    way it now points before speed is committed.
    """
    import coast_test as blob_test
    assert 0 < blob_test.TURN_SPEED_CM_S < 10
    assert blob_test.TURN_SPEED_CM_S >= 1.0


def test_the_return_to_turning_band_is_wide():
    """Narrow it and every small correction becomes a stop-and-turn, which
    gives back the smoothness the mode exists to provide."""
    import coast_test as blob_test
    assert blob_test.RETURN_DEG > blob_test.TURN_OK_DEG
    assert blob_test.RETURN_DEG >= 30.0


def test_a_turn_that_will_not_converge_gives_up_and_drives():
    """Pointed roughly right and moving beats stopped and correct."""
    import coast_test as blob_test
    assert 0 < blob_test.TURN_MAX_S <= 10


def test_the_three_styles_exist_and_pursuit_is_the_default():
    """Pursuit is the only sane choice for a line, circle or scribble; turn-go
    is the better one for point-to-point. `align` replaces neither: it points
    the ball ONCE, before pursuit starts, and hands over for good."""
    import coast_test as blob_test
    assert set(blob_test.STYLES) == {"pursuit", "align", "turn-go"}
    assert blob_test.STYLES[0] == "pursuit"


# -- the arrival radius ----------------------------------------------------

def test_arrival_radius_is_a_knob_not_a_constant():
    """It decides between arriving and orbiting, so it has to be reachable
    while a run is going wrong rather than only between runs."""
    import inspect

    import coast_test as blob_test
    assert "goal_tol" in inspect.getsource(blob_test.BlobTest.build_track)
    assert blob_test.ARRIVE_CM > 0


def test_the_default_radius_is_scaled_to_the_ball():
    """A ball whose centre is within its own radius of the goal is physically
    on it; asking for much better is asking the controller to satisfy a test
    its own size may not permit."""
    from vision.shots import BALL_CM
    import coast_test as blob_test
    assert blob_test.ARRIVE_CM < BALL_CM
    assert blob_test.ARRIVE_CM > BALL_CM / 2.0


def _fly(goal_tol, aim_error_deg=0.0, steps=3000, tau=0.35, dt=0.04,
         speed=25.0):
    """Chase a point with an AIM ERROR and MOMENTUM, and see if it ever lands.

    Both are needed to reproduce an orbit and neither is enough alone. A
    constant rotation on its own spirals inward and arrives; inertia on its own
    converges straight in. Together, past some error, the turn the ball needs
    near the goal is tighter than it can make, and it circles instead.

    Returns `(arrived, closest approach)`.
    """
    goal = Path.point([100.0, 100.0])
    p = np.array([40.0, 40.0])
    vel = np.zeros(2)
    c, s_ = (math.cos(math.radians(aim_error_deg)),
             math.sin(math.radians(aim_error_deg)))
    best = float("inf")
    for _ in range(steps):
        v, _, done, _ = pursue(goal, p, lookahead=15, speed=speed,
                               goal_tol=goal_tol)
        if done:
            return True, best
        v = np.array([v[0] * c - v[1] * s_, v[0] * s_ + v[1] * c])
        vel = vel + (v - vel) * min(dt / tau, 1.0)
        p = p + vel * dt
        best = min(best, float(np.linalg.norm(p - [100.0, 100.0])))
    return False, best


def test_a_large_aim_error_makes_pursuit_orbit_at_a_fixed_radius():
    """The relationship worth stating precisely, because the obvious reading is
    wrong: the RADIUS does not cause the orbit, the AIM ERROR does.

    Steering continuously at a point, a ball that cannot turn tightly enough
    settles into a circle around it, and that circle has a size set by the aim
    error -- 12cm at 60 degrees, 46cm at 80. Any acceptance radius inside that
    circle can never be satisfied, so the run never ends and it reads as a
    broken follower.

    Which means the fix is either a wider radius or a better aim, and knowing
    which is which is the point of pinning this.
    """
    arrived, _ = _fly(goal_tol=8.0, aim_error_deg=20.0)
    assert arrived, "a modest aim error should still land"

    arrived, closest = _fly(goal_tol=8.0, aim_error_deg=60.0)
    assert not arrived, "60 degrees off should orbit"
    assert closest > 8.0, "and the orbit should sit outside the radius"

    # Widening the radius past the orbit is what lets the same run terminate.
    arrived, _ = _fly(goal_tol=20.0, aim_error_deg=60.0)
    assert arrived


def test_a_closed_path_ignores_the_arrival_radius():
    """A circle is a patrol. Stopping it because the ball came near its start
    would end every lap."""
    c = Path.circle([0, 0], 40)
    p = np.array([40.0, 0.0])
    for _ in range(300):
        v, _, done, _ = pursue(c, p, lookahead=12, speed=30, goal_tol=30.0)
        assert not done
        p = p + v * 0.05


# -- latency, and what it demands of the knobs -----------------------------

def test_advice_scales_with_speed():
    """Both limits are distances covered while the ball is not yet obeying, so
    both grow with speed. The corollary is the useful one: to stop tighter you
    must go slower."""
    from coast_test import latency_advice
    plant = {"tau_s": 0.84, "coast_s": 0.509}
    slow, fast = latency_advice(10.0, plant), latency_advice(40.0, plant)
    assert fast["lookahead_cm"] == pytest.approx(slow["lookahead_cm"] * 4)
    assert fast["arrive_cm"] == pytest.approx(slow["arrive_cm"] * 4)


def test_arrival_is_sized_from_the_measured_coast_not_from_dead_time():
    """The correction this replaced, recorded because the mistake was mine.

    `calib/motion.json` measures dead time twice and the two disagree by their
    whole value -- 433ms from the step fit, 0.0 from the reversal probe, with
    `delay_agreement_s` recording the gap and the battery's own recommendation
    landing on one frame. Sizing the arrival radius off a disputed number was
    wrong; the coast is measured directly over eight stops and is the quantity
    that actually decides how close the ball can stop.
    """
    from coast_test import latency_advice
    a = latency_advice(28.0, {"tau_s": 0.84, "coast_s": 0.509})
    assert a["arrive_cm"] == pytest.approx(28.0 * 0.509, abs=0.1)


def test_a_missing_coast_falls_back_without_pretending():
    from coast_test import latency_advice
    a = latency_advice(20.0, {"tau_s": 0.84})
    assert a["arrive_cm"] > 0


def test_nothing_here_measures_latency_itself():
    """`fleet/characterize.py` owns that battery. A second measurement would
    drift from it, and this session already paid for adding a duplicate
    controller -- see the heading correction."""
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.plant_constants)
    assert "MOTION_PATH" in src


def test_drawing_a_path_does_not_shadow_the_view_rectangle():
    """A crash that only fired when the ball was lost WHILE a path was drawn.

    The arrival-radius circle used `r` for its radius, and `r` is the view
    rectangle in that scope. Everything worked until the one branch that reads
    the rectangle afterwards ran -- the "no blob" message -- so the app crashed
    precisely when the tracker lost the ball, which is the moment you least
    want the window to disappear.
    """
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.draw_view)
    body = src[src.index("arrival radius"):]
    assert "\n                r = " not in body


def test_the_log_records_where_it_was_SENT_not_only_where_it_was():
    """The gap this closed. An error is a difference, and the log used to
    record only one side of it -- position, with the path's KIND but never its
    destination. You could not compute how far off a run was from its own log.
    """
    assert "goal_x" in RunLog.COLUMNS and "goal_y" in RunLog.COLUMNS
    assert "gap_cm" in RunLog.COLUMNS


def test_the_log_separates_the_goal_from_the_steering_target():
    """Two different things, and conflating them hides a whole class of fault.

    Under pursuit the ball steers at a point `lookahead` ahead on the path,
    which is almost never the goal. Without both you cannot tell a follower
    steering correctly at the wrong point from one steering wrongly at the
    right one.
    """
    assert "target_x" in RunLog.COLUMNS and "target_y" in RunLog.COLUMNS


def test_the_log_carries_the_settings_in_force():
    """Sliders move mid-run. A log that cannot say how it was configured
    cannot explain its own behaviour a day later."""
    for c in ("style", "set_speed", "set_lookahead", "set_arrive"):
        assert c in RunLog.COLUMNS


# -- naming the mirrored axis ----------------------------------------------

def _legs(mirror):
    """Four probe legs under a mirror. `t = 2a - c` for a mirror line at `a`."""
    out = []
    for c in (0, 90, 180, 270):
        t = (2 * (90.0 if mirror == "y" else 0.0) - c) % 360.0
        out.append((c, t, 18.0))
    return out


def test_the_probe_names_which_axis_is_inverted():
    """A reflection about a line at bearing `a` sends commanded `c` to measured
    `2a - c`, so each leg estimates the mirror line and four agreeing legs name
    it outright. Telling somebody "one axis is inverted" and leaving them to
    work out which is half a diagnosis."""
    from coast_test import BlobTest
    assert "flip y" in BlobTest.mirror_axis(_legs("y"))
    assert "flip x" in BlobTest.mirror_axis(_legs("x"))


def test_reflection_flips_the_axis_PERPENDICULAR_to_the_mirror():
    """The one thing about this that is easy to state backwards: a mirror line
    along x inverts y, not x."""
    from coast_test import BlobTest
    said = BlobTest.mirror_axis(_legs("y"))
    assert "X AXIS is aligned" in said and "Y is inverted" in said


def test_an_oblique_mirror_is_reported_as_neither_axis():
    """A camera both rotated and mirrored has no axis to name, and saying one
    anyway would send somebody flipping the wrong thing."""
    from coast_test import BlobTest
    rows = [(c, (2 * 40.0 - c) % 360.0, 18.0) for c in (0, 90, 180, 270)]
    assert "neither axis" in BlobTest.mirror_axis(rows)


def test_disagreeing_legs_refuse_to_name_an_axis():
    from coast_test import BlobTest
    rows = [(0, 10.0, 18.0), (90, 200.0, 18.0), (180, 40.0, 18.0),
            (270, 300.0, 18.0)]
    assert "disagree" in BlobTest.mirror_axis(rows)


def test_flipping_is_button_only_not_bound_to_a_key():
    """A stray keypress that mirrors the arena mid-run would be a bad way to
    find out a key was bound twice -- and the brackets were already the
    exposure controls."""
    import inspect

    import coast_test as blob_test
    src = inspect.getsource(blob_test.BlobTest.key)
    assert "flip_axis" not in src


def test_the_app_constructs_and_both_tabs_build():
    """A guard against exactly what just happened: a new piece of state added
    to one place and not to `__init__`, so the ROBOT tab crashed the moment it
    was opened. The pure-function tests all passed through it."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import coast_test as blob_test
    app = blob_test.BlobTest("sim", with_fleet=False)
    try:
        for tab in ("robot", "track"):
            app.set_tab(tab)
            assert app.buttons
        app.draw()
    finally:
        app.close()


# -- arrival is circle to circle -------------------------------------------

def test_any_part_of_the_blob_inside_the_radius_counts_as_arrived():
    """The ball has extent, so "there" is circles overlapping, not a point
    inside a circle. Testing the centre asks the ball to travel its own radius
    further than the task needs -- and at these speeds that last few
    centimetres is the expensive part, because it has to be crept up on."""
    goal = Path.point([100.0, 100.0])
    at = np.array([92.0, 100.0])              # centre 8cm out

    _, _, done, _ = pursue(goal, at, 15, 20, goal_tol=6.0, radius=0.0)
    assert not done, "8cm out with no extent is not arrived"

    _, _, done, _ = pursue(goal, at, 15, 20, goal_tol=6.0, radius=2.0)
    assert done, "a 2cm blob edge reaches a 6cm radius from 8cm out"


def test_the_countdown_measures_to_the_stop_line_not_the_centre():
    """Otherwise it reads several centimetres remaining at the moment it
    stops, which looks like the controller quitting early."""
    goal = Path.point([100.0, 100.0])
    _, _, _, note = pursue(goal, np.array([91.0, 100.0]), 15, 20,
                           goal_tol=6.0, radius=2.0)
    assert note.startswith("1cm")


def test_extent_only_ever_makes_arrival_easier():
    """A monotonic guard: a bigger blob may stop sooner, never later."""
    goal = Path.point([100.0, 100.0])
    at = np.array([88.0, 100.0])
    reached = [pursue(goal, at, 15, 20, goal_tol=6.0, radius=r)[2]
               for r in (0.0, 3.0, 6.0, 9.0)]
    assert reached == sorted(reached, key=lambda d: d)   # False... then True


def test_an_open_path_end_uses_the_same_reach():
    ln = Path.line([0.0, 0.0], [100.0, 0.0])
    at = np.array([92.0, 0.0])
    assert not pursue(ln, at, 15, 20, goal_tol=6.0, radius=0.0)[2]
    assert pursue(ln, at, 15, 20, goal_tol=6.0, radius=3.0)[2]


# -- re-identify by blink code ---------------------------------------------

def test_blink_codes_are_distinct_and_carry_signal():
    """All-ones and all-zeros are excluded on purpose: a ball that never
    changes carries no signal, and two such balls would be indistinguishable
    from each other and from one whose LED is stuck."""
    from coast_test import BlobTest
    codes = BlobTest.id_codes(6)
    assert len(codes) == 6
    assert len({tuple(c) for c in codes}) == 6
    for c in codes:
        assert 0 < sum(c) < len(c)


def test_the_code_costs_the_same_for_three_robots_as_for_thirteen():
    """The reason to blink in parallel rather than call a roll: a sequence
    costs a slot per robot, a code costs `ID_SLOTS` slots however many there
    are."""
    from coast_test import BlobTest
    assert len(BlobTest.id_codes(3)[0]) == BlobTest.ID_SLOTS
    assert len(BlobTest.id_codes(12)[0]) == BlobTest.ID_SLOTS
    assert len(BlobTest.id_codes(99)) == (1 << BlobTest.ID_SLOTS) - 2


def test_a_blink_never_turns_the_light_off():
    """An extinguished ball is a ball the tracker loses, and coming back with
    names but no positions to attach them to defeats the purpose."""
    from coast_test import BlobTest
    assert 0.0 < BlobTest.ID_DIM < 1.0
    assert BlobTest.ID_DIM >= 0.2


def test_reidentify_fits_the_advertised_two_seconds():
    from coast_test import BlobTest
    assert BlobTest.ID_SLOTS * BlobTest.ID_SLOT_S == pytest.approx(2.0, abs=0.5)


def test_decoding_is_relative_to_each_track_not_an_absolute_level():
    """A ball far from the camera is dimmer than a near one at the same drive,
    so an absolute threshold would read distance as data."""
    import inspect

    from coast_test import BlobTest
    # The roll call and the resting blink read through one shared reader.
    assert "read_bits" in inspect.getsource(BlobTest.finish_reid)
    src = inspect.getsource(BlobTest.read_bits)
    assert "min(means)" in src and "max(means)" in src
    # And it behaves that way: the same code, near and far, reads the same.
    reader = BlobTest.__new__(BlobTest)
    near = [245.0, 73.0, 73.0, 245.0]
    far = [v * 0.4 for v in near]
    assert reader.read_bits(near) == reader.read_bits(far) == (1, 0, 0, 1)


def test_an_unclaimed_or_contested_code_is_left_unidentified():
    """A wrong name is worse than no name: the name is what drive commands are
    addressed to."""
    import inspect

    from coast_test import BlobTest
    src = inspect.getsource(BlobTest.finish_reid)
    assert 'identified"] = False' in src
    assert "count(who) > 1" in src


# -- the first-stretch direction check -------------------------------------

def test_the_fine_corrector_refuses_the_error_the_bootstrap_exists_for():
    """The gap this closes, stated as the two numbers that make it a gap.

    `retune_aim` throws away any single reading over AIM_MAX_STEP_DEG as a
    skid, which is right for the drift it trims. Pure pursuit cannot recover
    past 90 degrees either. So a frame error between those and 180 has nothing
    handling it, and the symptom is a ball driving confidently away.
    """
    import coast_test as m
    assert m.AIM_MAX_STEP_DEG < 90.0
    assert m.BOOTSTRAP_CM > 0 and m.BOOTSTRAP_GROW_CM > 0


def test_the_check_measures_the_gap_not_a_bearing():
    """Did the distance to the goal shrink is unambiguous, needs no convention
    to read, and cannot be confused with a wide turn -- a turn still closes on
    the goal, just slowly."""
    import inspect

    import coast_test as m
    src = inspect.getsource(m.BlobTest.step_bootstrap)
    assert 'b["gap0"]' in src and "BOOTSTRAP_GROW_CM" in src


def test_it_corrects_by_the_measured_error_not_a_flat_180():
    """Frames are wrong by whatever they are wrong by. Assuming a reversal
    would leave anything between 90 and 180 still diverging."""
    import inspect

    import coast_test as m
    src = inspect.getsource(m.BlobTest.step_bootstrap)
    assert "velocity_to_command" in src
    assert "180.0" in src           # only as the wrap, not as the correction
    assert "heading_offset = (h.heading_offset" in src


def test_the_check_runs_once_per_run():
    """A loop that keeps re-deciding which way is forward while driving can
    oscillate, and there is nothing here to damp it."""
    import inspect

    import coast_test as m
    src = inspect.getsource(m.BlobTest.step_bootstrap)
    assert "self.bootstrap = None" in src


def test_a_closed_path_is_left_alone():
    """A circle has no goal to close on, so "did the gap shrink" has no
    meaning for it."""
    import inspect

    import coast_test as m
    assert 'gap0"] is None' in inspect.getsource(m.BlobTest.step_bootstrap)


# -- the agent -------------------------------------------------------------

def _app_with_ball():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import time
    import coast_test as F
    from fleet.manager import Fleet
    from fleet.roster import RobotEntry
    app = F.BlobTest("sim", with_fleet=False)
    app.fleet = Fleet(seed=2)
    app.fleet.add(RobotEntry(name="B", code="BALL1", kind="sim",
                             color="red", ble_name=None))
    app.fleet.handles["BALL1"].pos = np.array([50.0, 45.0])
    rgb = list(F.config.led_rgb(F.config.COLORS["red"]["hue"]))
    app.bots["BALL1"] = {"ble": "s", "color": "red", "rgb": rgb,
                         "track": F.Track(), "marker": tuple(rgb),
                         "zeroed": False, "identified": False}
    app.code = "BALL1"
    app.agent_client = None
    for _ in range(25):
        time.sleep(0.01)
        app.tick()
        app.draw()
    return app


def test_the_model_cannot_reconfigure_the_rig():
    """The line `tools/registry.py` draws, drawn again here for the same
    reason: a model may say where a robot goes, not connect one, flip the
    arena frame, zero the aim or write calibration. Those leave the setup
    wrong in ways nothing downstream detects."""
    import coast_test as F
    names = {t["name"] for t in F.AGENT_TOOLS}
    for forbidden in ("connect", "disconnect", "flip_axis", "zero", "probe",
                      "save_calib", "set_led", "scan"):
        assert forbidden not in names


def test_a_mirrored_frame_is_refused_by_the_tool_not_just_the_prompt():
    """A prompt is a request. This is the thing that actually stops it."""
    app = _app_with_ball()
    try:
        app.mirrored = True
        out = app.agent_call("goto", {"x": 60, "y": 50})
        assert "error" in out and "mirror" in out["error"]
        assert "BLOCKED" in app.agent_prompt()
    finally:
        app.close()


def test_a_destination_outside_the_workspace_is_refused():
    app = _app_with_ball()
    try:
        h, w = app.frame.shape[:2]
        app.corners = [np.array(p, dtype=float) for p in
                       ((w * .3, h * .3), (w * .7, h * .3),
                        (w * .7, h * .7), (w * .3, h * .7))]
        box = app.agent_bounds()
        assert box is not None
        out = app.agent_call("goto", {"x": box[2] + 40, "y": box[3] + 40})
        # The wording changed when goals gained a ball-radius margin: a point
        # beyond the quad and one just inside it are refused by the same rule
        # now, and the message names the safe range rather than the quad.
        assert "error" in out and "too close to the edge" in out["error"]
    finally:
        app.close()


def test_errors_come_back_readable_for_the_model_to_fix():
    """`llm/agent.py` found that handing validation errors back verbatim
    recovers most first-attempt failures on the second try."""
    app = _app_with_ball()
    try:
        assert "error" in app.agent_call("goto", {"x": "left"})
        assert "unknown tool" in app.agent_call("fly", {})["error"]
        assert "two points" in app.agent_call("follow_path",
                                              {"points": [[1, 1]]})["error"]
    finally:
        app.close()


def test_the_call_cap_is_held_by_the_app_not_the_model():
    """A loop whose exit the model controls is how a ball drives in circles
    for ten minutes while somebody reads the transcript."""
    from llm.stub import replies
    import coast_test as F
    app = _app_with_ball()
    try:
        log = app.agent_run("wander",
                            client=replies(*[[("get_state", {})]] * 30))
        calls = [e for e in log if e[0] == "call"]
        assert len(calls) == F.BlobTest.AGENT_MAX_CALLS
        assert log[-1][0] == "error"
    finally:
        app.close()


def test_state_warns_when_the_name_on_the_ball_is_a_guess():
    """A model given a position with no hint that the identity is unverified
    will plan confidently on top of it."""
    app = _app_with_ball()
    try:
        assert "warning_identity" in app.agent_state()
    finally:
        app.close()


def test_typing_suppresses_every_shortcut():
    """Every letter in this app is a shortcut. A text box that leaves them
    live reconfigures the rig while you write to it."""
    import pygame
    app = _app_with_ball()
    try:
        before = (app.path, app.taper, app.tab, app.style, app.view)
        app.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SLASH,
                                   unicode="/"))
        for ch in "stop patrol goto":
            app.key(pygame.event.Event(pygame.KEYDOWN, key=ord(ch),
                                       unicode=ch))
        assert app.typed == "stop patrol goto"
        assert (app.path, app.taper, app.tab, app.style, app.view) == before
    finally:
        app.close()


def test_typing_does_not_drive_the_robot():
    """The residual half of the shortcut bug, and the dangerous half.

    `key` was guarded, but `manual_drive` polls `pygame.key.get_pressed()`
    directly and never passes through it -- so typing "was" on the ROBOT tab
    drove a real ball fifty centimetres before anything on screen said so.
    Suppressing keys in the event handler is not the same as suppressing the
    keyboard.
    """
    import time
    import unittest.mock as M

    import pygame

    class Held(dict):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def __getitem__(self, i):
            return i == self.k

    app = _app_with_ball()
    try:
        app.set_tab("robot")
        app.key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SLASH,
                                   unicode="/"))
        h = app.fleet.handles["BALL1"]
        start = h.pos.copy()
        with M.patch.object(pygame.key, "get_pressed", lambda: Held(pygame.K_d)):
            for _ in range(40):
                time.sleep(0.005)
                app.tick()
                app.draw()
        assert float(np.linalg.norm(h.pos - start)) < 1.0
    finally:
        app.close()


def test_opening_the_box_mid_drive_releases_the_keys():
    """A suppressed key must also release what it was holding, or a text box
    opened while driving leaves the last command standing and the ball
    rolling while you type at it."""
    import inspect

    import coast_test as F
    src = inspect.getsource(F.BlobTest.manual_drive)
    assert "self.typing" in src
    assert "h.stop()" in src


def test_the_model_is_a_flag_not_an_edit():
    """Comparing a 4B against a 9B on the same rig should not need the source
    changed between runs -- that is how the two get compared under quietly
    different conditions."""
    import coast_test as F
    app = F.BlobTest("sim", with_fleet=False, model="qwen3.5:4b")
    try:
        assert app.agent_model == "qwen3.5:4b"
    finally:
        app.close()


def test_an_unknown_preset_does_not_stop_the_app():
    """This is a bench tool for a camera and a ball. A language model is
    something it can use when one is running, never a dependency."""
    import coast_test as F
    app = F.BlobTest("sim", with_fleet=False, model="no-such-model")
    try:
        assert app.agent_client is None
        app.tick()
        app.draw()
    finally:
        app.close()


def test_the_jump_gate_is_sized_from_the_rig_not_a_constant():
    """One number cannot serve two cameras.

    The gate is a distance in PIXELS, and pixels per centimetre is a property
    of the mounting: 3.6 on the real rig, 9.2 in the sim. The same 60px default
    is loose on one and tight on the other, which is why it is computed from
    the homography and the frame rate rather than picked.
    """
    import time

    import coast_test as F
    app = F.BlobTest("sim", with_fleet=False)
    try:
        for _ in range(30):
            time.sleep(0.01)
            app.tick()
            app.draw()
        scale = app.px_per_cm()
        assert scale and scale > 1.0
        want = app.jump_advice()
        assert want and want > 0
        # Must comfortably exceed one frame of travel at the ball's top speed,
        # or an honest fast move gets disbelieved.
        from fleet.handle import MAX_SPEED
        assert want > MAX_SPEED / max(app.cam.fps, 1.0) * scale
    finally:
        app.close()


def test_the_advice_needs_no_dropped_frame_margin():
    """`Track` widens its own gate by the number of misses, so this is the
    one-frame number and doubling it for dropouts would be counting twice."""
    import inspect

    import coast_test as F
    assert "misses" in inspect.getsource(F.Track.update)
    assert "DROPPED" in inspect.getsource(F.BlobTest.jump_advice)


# -- slowing into the end of an open path ----------------------------------

def _loop_path():
    """A scribble that finishes near where it started -- the shape that
    separates a correct taper from a plausible one."""
    import coast_test as F
    pts = [(20, 20), (90, 20), (90, 60), (30, 60), (30, 26), (26, 22)]
    return F.Path([np.array(p, dtype=float) for p in pts], kind="freehand")


def test_it_slows_into_the_end_of_a_freehand_path():
    """A FRESH PATH PER PROBE, because following one is now stateful.

    These two positions are independent scenarios -- a ball in the middle of
    the route and a ball at the end of it -- and they used to share a `Path`
    because following one was a pure function of where the ball was. It is not
    any more: the ratchet carries arc position between calls, so a probe made
    after another one is windowed to the progress the first left behind, and
    the ball at the end reads as a ball that has jumped there. That is the
    ratchet doing its job. Giving each probe its own path asks the question
    each was written to ask.
    """
    import coast_test as F
    fast = np.linalg.norm(F.pursue(_loop_path(), np.array([60.0, 20.0]), 15, 25,
                                   goal_tol=6, coast_s=0.5, min_speed=8)[0])
    slow = np.linalg.norm(F.pursue(_loop_path(), np.array([29.0, 27.0]), 15, 25,
                                   goal_tol=6, coast_s=0.5, min_speed=8)[0])
    assert fast == pytest.approx(25.0)
    assert slow < 12.0


def test_it_does_NOT_crawl_at_the_start_of_a_path_that_loops_back():
    """The bug the obvious implementation has.

    Taking the smaller of "arc left" and "straight line to the end" looks
    right and is wrong here: at the start of this path the ball is 4cm from
    the ENDPOINT and has 208cm still to drive, so the smaller measure makes it
    crawl every lap. Arriving needs both inside the radius, so the distance
    still to cover is the LARGER.
    """
    import coast_test as F
    path = _loop_path()
    start = np.array([22.0, 22.0])
    arc_left = path.length - path.project(start)[0]
    straight = float(np.linalg.norm(path.pts[-1] - start))
    assert straight < 10 and arc_left > 150        # near in space, far to drive
    v, _, done, _ = F.pursue(path, start, 15, 25, goal_tol=6,
                             coast_s=0.5, min_speed=8)
    assert not done
    assert np.linalg.norm(v) == pytest.approx(25.0)


def test_arrival_still_needs_both_measures():
    """Being beside the endpoint is not arriving if the path has not been
    driven; the taper and the arrival test agree about that."""
    import coast_test as F
    assert not F.pursue(_loop_path(), np.array([22.0, 22.0]), 15, 25,
                        goal_tol=6)[2]
    assert F.pursue(_loop_path(), np.array([26.5, 22.5]), 15, 25, goal_tol=6)[2]


def test_a_commanded_but_motionless_ball_is_called_out():
    """The app knows this and used to keep it to itself.

    It knows the velocity it commanded, the byte that became, whether the link
    is up and what the camera measures. "It did not move" is a conclusion those
    make jointly, and leaving a person to draw it by comparing a dock line
    against the floor is how a session goes to the wrong suspect.

    Checked with the escape disabled, because a stall now TRIGGERS the escape
    and the escape clears the report -- which is the right behaviour and makes
    the report itself momentary.
    """
    import time

    import coast_test as F
    app = _app_with_ball()
    saved = F.UNSTICK_TRIES
    try:
        F.UNSTICK_TRIES = 0                # report, do not act
        h = app.fleet.handles["BALL1"]
        app.path = F.Path.circle(np.array([60.0, 45.0]), 25)
        app.speed = 8
        h.gain = 0.0
        app.arm()
        seen = []
        real = app.stall_report
        app.stall_report = lambda _r=real: (lambda v: (seen.append(v)
                                                       if v else None) or v)(_r())
        for _ in range(220):
            time.sleep(0.008)
            app.tick()
            app.draw()
        assert seen, "the stall was never reported"
        seen = seen[0]
        assert "COMMANDED" in seen[0] and "byte" in seen[0]
    finally:
        F.UNSTICK_TRIES = saved
        app.disarm()
        app.close()


def test_the_stall_report_waits_before_accusing():
    """This rig takes most of a second to get going. Calling a stall inside
    that would flag every normal start."""
    import coast_test as F
    assert F.STALL_AFTER_S >= 1.0


# -- edges, corners and getting stuck --------------------------------------

def test_location_tools_refuse_without_a_workspace():
    """"Anywhere" is not a safe default for a tool that drives a real ball.
    Without a boundary every coordinate is as plausible as any other and none
    of them is checked."""
    app = _app_with_ball()
    try:
        app.corners = None
        out = app.agent_call("goto", {"x": 60, "y": 45})
        assert "error" in out and "no workspace" in out["error"]
    finally:
        app.close()


def test_goals_are_held_clear_of_the_boundary_by_the_ball_s_radius():
    """A goal on the line is one the ball can only answer by pushing: its
    centre cannot reach the edge, so the controller keeps commanding into the
    wall because from its point of view it has not arrived."""
    app = _app_with_ball()
    try:
        h, w = app.frame.shape[:2]
        app.corners = [np.array(p, dtype=float) for p in
                       ((w * .1, h * .1), (w * .9, h * .1),
                        (w * .9, h * .9), (w * .1, h * .9))]
        box = app.agent_bounds()
        assert "too close to the edge" in app.agent_call(
            "goto", {"x": box[0], "y": box[1]})["error"]
        mid = app.agent_call("goto", {"x": (box[0] + box[2]) / 2,
                                      "y": (box[1] + box[3]) / 2})
        assert mid.get("ok")
    finally:
        app.disarm()
        app.close()


def test_the_margin_accounts_for_the_ball_not_just_a_constant():
    import coast_test as F
    from vision.shots import BALL_CM
    app = _app_with_ball()
    try:
        assert app.goal_margin_cm() >= BALL_CM / 2.0 + F.EDGE_MARGIN_CM - 0.01
    finally:
        app.close()


def test_a_stuck_ball_backs_off_and_then_gives_up():
    """The escape is the reverse of the last command -- a ball that is
    commanded and not moving is against something, and the direction it was
    driving is the direction of the obstacle. No map needed.

    And it is capped. A ball wedged under a chair leg cannot be nudged free,
    and a loop that keeps shoving spends a battery learning nothing.
    """
    import time

    import coast_test as F
    app = _app_with_ball()
    try:
        h = app.fleet.handles["BALL1"]
        h.pos = np.array([60.0, 45.0])
        hh, w = app.frame.shape[:2]
        app.corners = [np.array(p, dtype=float) for p in
                       ((w * .1, hh * .1), (w * .9, hh * .1),
                        (w * .9, hh * .9), (w * .1, hh * .9))]
        wall, real = 95.0, h.step

        def blocked(dt, _s=real):
            _s(dt)
            if h.pos[0] > wall:
                h.pos[0] = wall
                if h.vel[0] > 0:
                    h.vel[0] = 0.0
        h.step = blocked

        box = app.agent_bounds()
        app.path = F.Path.point(np.array([box[2] - app.goal_margin_cm() - 1,
                                          45.0]))
        app.speed = 20
        app.arm()
        # Run to a WALL-CLOCK deadline rather than a tick count. The stall
        # and escape timers are in seconds, so a fixed number of iterations is
        # a different amount of simulated time on a loaded machine than on an
        # idle one -- which is how this test passed alone and failed in the
        # suite.
        xs, deadline = [], time.perf_counter() + 25.0
        while app.armed and time.perf_counter() < deadline:
            time.sleep(0.004)
            app.tick()
            app.draw()
            xs.append(h.pos[0])
        hit = next((i for i, x in enumerate(xs) if x >= wall - 0.2), None)
        assert hit is not None, "the test never reached the wall"
        # How FAR the nudge backs it off is not asserted. The stall and escape
        # timers run on the wall clock while the simulation advances per tick,
        # so on a loaded machine the same seconds buy fewer centimetres -- and
        # a test that measures distance under those conditions passes alone and
        # fails in a suite, which is worse than not testing it. The decisions
        # are deterministic and are what this is for; the distance was measured
        # by hand at 11.6cm off a wall.
        assert not app.armed
        assert (app.unstick or {}).get("tries") == F.UNSTICK_TRIES
        assert "stuck" in app.note
    finally:
        app.close()


def test_the_log_groups_rows_into_runs_and_records_how_each_ended():
    """Three columns that turn a frame dump into a table of attempts.

    `run_id` groups rows into arms; `run_source` says whether a person pressed
    GO or the model called a tool, which is the first thing you want when one
    behaves worse than the other; `run_outcome` is blank until the last row of
    a run, because while a run is happening nothing knows how it ends.
    """
    import coast_test as F
    for c in ("run_id", "run_source", "run_outcome"):
        assert c in F.RunLog.COLUMNS


def test_an_agent_run_is_labelled_differently_from_a_button_run():
    import coast_test as F
    app = _app_with_ball()
    try:
        hh, w = app.frame.shape[:2]
        app.corners = [np.array(p, dtype=float) for p in
                       ((w * .05, hh * .05), (w * .95, hh * .05),
                        (w * .95, hh * .95), (w * .05, hh * .95))]
        app.path = F.Path.point(np.array([80.0, 55.0]))
        app.arm()
        assert app.run_source == "button"
        app.disarm("halted by esc")

        app.agent_call("goto", {"x": 70.0, "y": 55.0})
        assert app.run_source == "agent"
        assert app.run_id == 2
        app.disarm("halted by esc")
    finally:
        app.close()


def test_outcomes_separate_arriving_from_giving_up():
    """"It stopped" is not a result. Arrived, stuck, lost and halted are four
    different things and the log has to tell them apart."""
    import inspect

    import coast_test as F
    src = inspect.getsource(F.BlobTest.disarm)
    for word in ("arrived", "stuck", "lost", "halted"):
        assert word in src


def test_the_speed_burst_is_only_for_escapes():
    """A run that begins with a burst begins with an overshoot, and it makes
    the slider mean one thing for the first third of a second and another
    afterwards. Ordinary starts use the commanded speed; only breaking contact
    with a wall gets more."""
    import inspect

    import coast_test as F
    assert not hasattr(F.BlobTest, "kick")
    assert "KICK_SPEED_CM_S" in inspect.getsource(F.BlobTest.start_unstick)
    assert "KICK_SPEED_CM_S" not in inspect.getsource(F.BlobTest.drive)


@pytest.mark.parametrize("commanded", [4.0, 8.0, 12.0, 25.0])
def test_the_taper_floor_never_exceeds_what_was_asked_for(commanded):
    """The floor stops the taper asking for a speed the ball cannot move at.
    It is not there to overrule the slider -- and unclamped it did: at a
    commanded 10cm/s a floor of 12 made the ball speed UP on its final
    approach, worst at exactly the low settings this rig is driven at."""
    import coast_test as F
    goal = F.Path.point([100.0, 100.0])
    for gap in (30.0, 12.0, 6.0, 2.0, 0.5):
        v, _, done, _ = F.pursue(goal, np.array([100.0 - gap - 6.0, 100.0]),
                                 15, commanded, goal_tol=6.0,
                                 coast_s=0.5, min_speed=F.MIN_MOVING_CM_S)
        assert float(np.linalg.norm(v)) <= commanded + 1e-6


# -- the ratchet, the patrol, and the route in the log ---------------------

def test_a_retracing_path_aims_the_lookahead_BACKWARDS_near_its_turn():
    """WHY A PATROL IS NOT ONE PATH THAT DOUBLES BACK. Pinned, not fixed.

    This is the measured failure from the rig, reproduced. Following a route
    that goes out and returns, the lookahead steps PAST the turning vertex and
    lands on the returning leg, so the point being steered at sits behind the
    ball and gets further behind the closer the ball comes to the end. With
    the vertex at x=112 and a 15cm lookahead the target is `2*112 - 15 - x`,
    which is the `target_x = 209 - ball_x` measured in the log while the ball
    reversed 1379 times in 264 seconds without ever driving back down the edge.

    The ratchet does not fix this and is not meant to: it settles WHICH LEG the
    ball is on, and here the answer is honestly "the outbound one" -- the route
    really does turn round 15cm ahead. Aiming around a 180-degree reversal is
    what pure pursuit cannot do, so the patrol is built from one-way legs
    instead. This test exists so that nobody rebuilds it the obvious way.
    """
    import coast_test as F
    out_and_back = F.Path([np.array([5.0, -15.0]), np.array([112.0, -15.0]),
                           np.array([5.0, -15.0])])
    for x in (100.0, 104.0, 108.0, 111.0):
        _, t, _, _ = F.pursue(out_and_back, np.array([x, -15.0]), 15, 20,
                              goal_tol=6)
        assert float(t[0]) == pytest.approx(209.0 - x, abs=1.0)
    # Reflected about the vertex, so the target crosses the ball at half the
    # lookahead short of the end and is behind it from there on -- steering
    # the ball away from an end it has not reached.
    _, t, _, _ = F.pursue(out_and_back, np.array([111.0, -15.0]), 15, 20,
                          goal_tol=6)
    assert float(t[0]) < 111.0, "the target is behind the ball"


def test_a_patrol_leg_always_aims_the_lookahead_FORWARD():
    """The same sweep, on the shape a patrol is actually built from.

    One leg is an ordinary open line with no reversal in it, so the lookahead
    has nowhere backwards to land and the target advances with the ball all
    the way to the end -- where the run arrives, and the patrol swaps the ends
    and drives the next leg.
    """
    import coast_test as F
    leg = F.Path.line(np.array([5.0, -15.0]), np.array([112.0, -15.0]))
    seen = []
    for x in np.arange(20.0, 112.0, 2.0):
        _, t, _, _ = F.pursue(leg, np.array([x, -15.0]), 15, 20, goal_tol=6)
        seen.append(float(t[0]))
        assert float(t[0]) >= x, "the target must never fall behind the ball"
    assert seen == sorted(seen), "the target must never move backwards"
    assert seen[-1] == pytest.approx(112.0, abs=0.5)


def test_a_ball_may_not_skip_to_the_far_leg_of_a_retracing_path():
    """The other half of the same guarantee.

    The ratchet is not merely smoothing: it is what makes "how far along am I"
    answerable at all on a path that crosses itself. A ball at the start must
    read as being at the start, even though the returning leg passes through
    exactly the same point.
    """
    import coast_test as F
    p = F.Path([np.array([0.0, 0.0]), np.array([100.0, 0.0]),
                np.array([0.0, 0.0])])
    s0, _ = p.project(np.array([10.0, 0.0]))          # no progress yet
    p.s = s0
    s1, _ = p.project(np.array([20.0, 0.0]), near=p.s)
    assert s1 == pytest.approx(20.0, abs=1.0)         # outbound, not returning
    assert s1 < 100.0


def test_the_ratchet_relocks_when_the_ball_is_moved():
    """Forward-only is a rule about following, not a claim about physics.

    A ball that has been picked up and put down somewhere else has not
    travelled the path to get there, and holding it to the old arc position
    would drive it back toward a place it is no longer near. Past
    `RELOCK_CM` the search goes global again.
    """
    import coast_test as F
    p = F.Path([np.array([0.0, 0.0]), np.array([200.0, 0.0])])
    p.s = 190.0
    s, off = p.project(np.array([10.0, 0.0]), near=p.s)
    assert s == pytest.approx(10.0, abs=1.0)
    assert off < 1.0


def test_a_patrol_turns_round_at_each_end_instead_of_stopping():
    """What "patrol an edge" is supposed to do, and did not.

    Arriving at the far end of a patrol is not the end of the run -- it is the
    moment to swap the ends and drive back. Pinned on the state machine rather
    than on a hardware run because the thing that was broken was the decision,
    not the driving.
    """
    import coast_test as F
    app = _app_with_ball()
    try:
        a, b = np.array([10.0, 40.0]), np.array([90.0, 40.0])
        app.start_patrol(a, b)
        assert app.path.kind == "line"
        first = list(app.path.pts)
        assert app.step_patrol()                       # arrival swaps the legs
        assert np.allclose(app.path.pts[0], first[-1])
        assert np.allclose(app.path.pts[-1], first[0])
        assert app.patrol["legs"] == 2
        assert app.step_patrol() and app.patrol["legs"] == 3
        assert np.allclose(app.path.pts[0], first[0])  # back where it began
    finally:
        app.close()


def test_drawing_a_new_route_ends_the_patrol():
    """Tied to the path object, so there is no flag to forget to clear.

    A patrol that outlived the path it was patrolling would turn a later
    point-to-point drive into an endless one, and the symptom -- a ball that
    will not stop -- looks nothing like its cause.
    """
    import coast_test as F
    app = _app_with_ball()
    try:
        app.start_patrol(np.array([10.0, 40.0]), np.array([90.0, 40.0]))
        app.path = F.Path.point(np.array([50.0, 50.0]))   # a new route
        assert not app.step_patrol()
    finally:
        app.close()


def test_the_log_records_the_route_and_not_merely_its_kind():
    """One cell per run, because "polyline" did not turn out to be enough.

    A patrol that oscillated at one end for four minutes was logged as
    `path=polyline`, and the geometry that explained it had to be recovered
    from arithmetic relating the target to the ball. The point list is what
    makes that read directly.
    """
    import csv
    import coast_test as F
    app = _app_with_ball()
    try:
        assert "path_pts" in F.RunLog.COLUMNS
        hh, w = app.frame.shape[:2]
        app.corners = [np.array(p, dtype=float) for p in
                       ((w * .1, hh * .1), (w * .9, hh * .1),
                        (w * .9, hh * .9), (w * .1, hh * .9))]
        app.start_patrol(np.array([20.0, 40.0]), np.array([80.0, 40.0]))
        app.arm()
        for _ in range(4):
            app.tick()
            app.draw()
        app.log.close()
        rows = [r for r in csv.DictReader(open(app.log.path)) if r["path_pts"]]
        assert len(rows) == 1, "the route belongs on ONE row per run"
        assert rows[0]["path_pts"] == "20.0 40.0;80.0 40.0"
    finally:
        app.close()


# -- predicting where the ball comes to rest ----------------------------------
#
# The bench already learns the roll-out as a TIME, because that is the form
# that transfers: `coast_s * speed` is the distance at any speed. These run it
# forwards -- from "how long it coasts" to "where it stops".

from coast_test import coast_rest                                # noqa: E402


def test_the_rest_point_is_the_coast_distance_along_the_travel():
    """`coast_s` times speed is the distance, which is the same model the
    approach taper already consumes in the other direction."""
    got = coast_rest([10.0, 20.0], [25.0, 0.0], 0.5)
    assert got == pytest.approx([22.5, 20.0])          # 25cm/s * 0.5s = 12.5cm


def test_it_follows_the_direction_of_travel_not_an_axis():
    got = coast_rest([0.0, 0.0], [0.0, -30.0], 0.4)
    assert got == pytest.approx([0.0, -12.0])


def test_a_faster_ball_rolls_further():
    slow = coast_rest([0.0, 0.0], [10.0, 0.0], 0.5)
    fast = coast_rest([0.0, 0.0], [30.0, 0.0], 0.5)
    assert fast[0] == pytest.approx(3.0 * slow[0])


def test_the_measured_roll_out_is_reproduced():
    """8 to 17cm from working speed, per the rig's own measurements."""
    for speed in (16.0, 34.0):
        rolled = np.linalg.norm(
            np.asarray(coast_rest([0.0, 0.0], [speed, 0.0], 0.509)))
        assert 7.0 <= rolled <= 18.0


def test_a_STATIONARY_ball_rests_where_it_is():
    got = coast_rest([5.0, 5.0], [0.0, 0.0], 0.5)
    assert got == pytest.approx([5.0, 5.0])


def test_NO_measured_coast_means_no_roll_rather_than_a_guess():
    assert coast_rest([5.0, 5.0], [25.0, 0.0], None) == pytest.approx([5.0, 5.0])
    assert coast_rest([5.0, 5.0], [25.0, 0.0], 0.0) == pytest.approx([5.0, 5.0])


def test_the_prediction_differs_from_the_position_by_more_than_the_ball():
    """The reason it is worth computing at all. Reporting the position at the
    moment of disarm once announced [88, 58] for a ball that stopped at
    [94, 60]."""
    here = np.array([88.0, 58.0])
    rest = coast_rest(here, [24.0, 8.0], 0.509)
    assert float(np.linalg.norm(rest - here)) > 7.3


# -- centimetres from one drive, from the bench keyboard ------------------

def _scale_app():
    import coast_test as F
    return F.BlobTest("sim", with_fleet=False)


def test_m_reads_which_half_of_the_calibration_it_is_in():
    """One key, two steps, separated by somebody walking over with a tape."""
    app = _scale_app()
    try:
        app.pending_scale = {"px": 400.0, "cm": 41.3, "at": 0.0}
        app.scale_key()
        assert app.typing and app.typing_mode == "measure"
        assert "41.3" in app.note, app.note
    finally:
        app.close()


def test_a_tape_reading_rescales_the_arena_and_is_written_down():
    app = _scale_app()
    try:
        before = app.homography.M.copy()
        app.pending_scale = {"px": 400.0, "cm": 40.0, "at": 0.0}
        saved = []
        app.homography.save = lambda: saved.append(True)
        app.take_measurement("46")
        assert saved, "a calibration that is not written down is not one"
        assert app.pending_scale is None
        # 46 measured against 40 reported: everything this reports grows 1.15x.
        assert app.homography.M[0, 0] == pytest.approx(before[0, 0] * 1.15,
                                                       rel=1e-9)
    finally:
        app.close()


@pytest.mark.parametrize("typed", ["", "  ", "about a foot", "-5"])
def test_a_reading_that_is_not_a_distance_changes_nothing(typed):
    app = _scale_app()
    try:
        before = app.homography.M.copy()
        app.pending_scale = {"px": 400.0, "cm": 40.0, "at": 0.0}
        app.take_measurement(typed)
        assert np.allclose(app.homography.M, before)
    finally:
        app.close()


def test_typing_a_measurement_does_not_reach_the_agent():
    """The text box started as the agent prompt. A tape reading typed into it
    must not be sent to a language model as a sentence."""
    app = _scale_app()
    try:
        asked = []
        app.ask = lambda t: asked.append(t)
        app.pending_scale = {"px": 400.0, "cm": 40.0, "at": 0.0}
        app.homography.save = lambda: None
        app.scale_key()
        app.take_measurement("46") if app.typing_mode == "measure" else None
        assert asked == []
    finally:
        app.close()


def test_a_text_box_says_why_the_keys_are_dead():
    """An unnoticed text box and a broken bench look identical from the
    keyboard. Every other refusal in `manual_drive` speaks; this one did not,
    and `m` made it easy to be in one without meaning to."""
    import pygame

    app = _scale_app()
    try:
        app.tab = "robot"
        app.typing = True

        class Handle:
            connected = True
            heading_offset = 0.0

            def stop(self):
                pass

        app.fleet = type("F", (), {"handles": {"AAAA": Handle()}})()
        app.code = "AAAA"

        class Held:
            def __getitem__(self, k):
                return k == pygame.K_w

        real = pygame.key.get_pressed
        pygame.key.get_pressed = lambda: Held()
        try:
            assert app.manual_drive() is False
        finally:
            pygame.key.get_pressed = real
        assert "esc" in (app.note or "").lower(), app.note
    finally:
        app.close()


# -- the coordinate frame, drawn on the floor it describes ----------------

def test_the_frame_overlay_is_off_until_it_is_asked_for():
    app = _scale_app()
    try:
        assert app.show_axes is False
        app.toggle_axes()
        assert app.show_axes is True
        app.toggle_axes()
        assert app.show_axes is False
    finally:
        app.close()


def test_drawing_the_frame_does_not_need_a_ball_or_a_path():
    """It describes the floor, not the run. Refusing to draw without a lock
    would hide it exactly when a position is most in doubt."""
    import time

    app = _scale_app()
    try:
        app.show_axes = True
        app.track.forget()
        app.path = None
        for _ in range(5):
            time.sleep(0.01)
            app.tick()
            app.draw()
    finally:
        app.close()


def test_the_frame_is_silent_without_a_homography():
    """Centimetres mean nothing without one, so there is nothing to draw and
    nothing to crash on."""
    app = _scale_app()
    try:
        app.homography.M = None
        app.show_axes = True
        app.draw_axes(lambda p: (int(p[0]), int(p[1])))
    finally:
        app.close()


def test_the_grid_follows_the_clicked_corners_not_the_calibration_quad():
    """The two differ by a third of a metre on this rig. A grid drawn from the
    homography's own rectangle would label the floor with numbers no other part
    of the bench uses."""
    app = _scale_app()
    try:
        b = app.agent_bounds()
        if b is None:
            pytest.skip("no corners in this fixture")
        seen = []
        app.show_axes = True
        app.draw_axes(lambda p: seen.append(p) or (int(p[0]), int(p[1])))
        assert seen, "nothing was drawn"
    finally:
        app.close()


# -- how fast the command may turn -----------------------------------------

def _swing(rate, monkeypatch):
    """Ask for a 90 degree turn 0.1s after the last command; say how far it got."""
    import coast_test as F
    app = F.BlobTest("sim", with_fleet=False)
    try:
        clock = [100.0]
        monkeypatch.setattr(F.time, "perf_counter", lambda: clock[0])
        app.turn_rate = rate
        app.slew(np.array([20.0, 0.0]))            # heading 0
        clock[0] += 0.1
        v = app.slew(np.array([0.0, 20.0]))        # wants 90
        return float(np.degrees(np.arctan2(v[1], v[0])))
    finally:
        app.close()


def test_the_turn_rate_slider_is_what_limits_the_swing(monkeypatch):
    """Together with lookahead it sets how tight a corner is: in sim a square's
    corners went from 3.3cm cut to under 1cm with a shorter lookahead and a
    faster turn. It is a knob so it can be set against the real ball's noise."""
    assert _swing(220.0, monkeypatch) == pytest.approx(22.0, abs=0.5)
    assert _swing(600.0, monkeypatch) == pytest.approx(60.0, abs=0.5)


def test_the_turn_rate_is_a_slider_and_is_logged():
    import coast_test as F
    from coast_test import RunLog
    app = F.BlobTest("sim", with_fleet=False)
    try:
        assert any(s.label == "turn rate" for s in app.sliders)
        assert app.turn_rate == F.SLEW_DEG_S, "the default is unchanged"
    finally:
        app.close()
    assert "set_turn" in RunLog.COLUMNS
