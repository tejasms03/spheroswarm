"""x and y from one blob: what it costs, and where it stops answering.

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

from blob_test import (BallSource, MAX_AREA, MIN_AREA, RunLog, V_MIN, Path,
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

    import blob_test
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

    import blob_test
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

    import blob_test
    src = inspect.getsource(blob_test.BlobTest.drive)
    assert "observe_heading" in src
    assert not hasattr(blob_test.BlobTest, "retune_aim")


def test_the_correction_can_be_switched_off_to_isolate_the_follower():
    """Pure pursuit is geometric and needs no heading correction to follow a
    path, so turning this off is how you tell a follower fault from an aiming
    one."""
    import inspect

    import blob_test
    src = inspect.getsource(blob_test.BlobTest.cycle_aim)
    assert "heading_tracking" in src
    assert set(blob_test.AIM_MODES) == {"off", "on"}


def test_one_definition_of_the_compass_convention():
    """`velocity_to_command` is the project's single statement of "zero is +y,
    clockwise". A second copy is a second place for a sign error, and on this
    quantity that circles whatever you correct instead of failing plainly."""
    import inspect

    import blob_test
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

    import blob_test
    src = inspect.getsource(blob_test.BlobTest.drive)
    assert "DRIVE_ON_PREDICTION" in src
    assert 0 < blob_test.DRIVE_ON_PREDICTION <= 6


def test_the_commanded_direction_is_rate_limited():
    """Position noise becomes heading noise -- pursuit's gain rises as the
    lookahead shortens -- and a Sphero physically turns its drive assembly to
    follow every twitch."""
    import inspect

    import blob_test
    assert "slew" in inspect.getsource(blob_test.BlobTest.drive)
    assert blob_test.SLEW_DEG_S > 0


def test_slew_limits_the_turn_but_never_the_speed():
    """Speed is the operator's slider and must not be smoothed behind their
    back; only the direction's rate is limited."""
    import types

    import blob_test
    app = types.SimpleNamespace(_slew_at=None, _slew_deg=None,
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
    import blob_test
    assert 0 < blob_test.TURN_SPEED_CM_S < 10
    assert blob_test.TURN_SPEED_CM_S >= 1.0


def test_the_return_to_turning_band_is_wide():
    """Narrow it and every small correction becomes a stop-and-turn, which
    gives back the smoothness the mode exists to provide."""
    import blob_test
    assert blob_test.RETURN_DEG > blob_test.TURN_OK_DEG
    assert blob_test.RETURN_DEG >= 30.0


def test_a_turn_that_will_not_converge_gives_up_and_drives():
    """Pointed roughly right and moving beats stopped and correct."""
    import blob_test
    assert 0 < blob_test.TURN_MAX_S <= 10


def test_both_styles_exist_and_pursuit_is_the_default():
    """Pursuit is the only sane choice for a line, circle or scribble; turn-go
    is the better one for point-to-point. Neither replaces the other."""
    import blob_test
    assert set(blob_test.STYLES) == {"pursuit", "turn-go"}
    assert blob_test.STYLES[0] == "pursuit"


# -- the arrival radius ----------------------------------------------------

def test_arrival_radius_is_a_knob_not_a_constant():
    """It decides between arriving and orbiting, so it has to be reachable
    while a run is going wrong rather than only between runs."""
    import inspect

    import blob_test
    assert "goal_tol" in inspect.getsource(blob_test.BlobTest.build_track)
    assert blob_test.ARRIVE_CM > 0


def test_the_default_radius_is_scaled_to_the_ball():
    """A ball whose centre is within its own radius of the goal is physically
    on it; asking for much better is asking the controller to satisfy a test
    its own size may not permit."""
    from vision.shots import BALL_CM
    import blob_test
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
    from blob_test import latency_advice
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
    from blob_test import latency_advice
    a = latency_advice(28.0, {"tau_s": 0.84, "coast_s": 0.509})
    assert a["arrive_cm"] == pytest.approx(28.0 * 0.509, abs=0.1)


def test_a_missing_coast_falls_back_without_pretending():
    from blob_test import latency_advice
    a = latency_advice(20.0, {"tau_s": 0.84})
    assert a["arrive_cm"] > 0


def test_nothing_here_measures_latency_itself():
    """`fleet/characterize.py` owns that battery. A second measurement would
    drift from it, and this session already paid for adding a duplicate
    controller -- see the heading correction."""
    import inspect

    import blob_test
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

    import blob_test
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
    from blob_test import BlobTest
    assert "flip y" in BlobTest.mirror_axis(_legs("y"))
    assert "flip x" in BlobTest.mirror_axis(_legs("x"))


def test_reflection_flips_the_axis_PERPENDICULAR_to_the_mirror():
    """The one thing about this that is easy to state backwards: a mirror line
    along x inverts y, not x."""
    from blob_test import BlobTest
    said = BlobTest.mirror_axis(_legs("y"))
    assert "X AXIS is aligned" in said and "Y is inverted" in said


def test_an_oblique_mirror_is_reported_as_neither_axis():
    """A camera both rotated and mirrored has no axis to name, and saying one
    anyway would send somebody flipping the wrong thing."""
    from blob_test import BlobTest
    rows = [(c, (2 * 40.0 - c) % 360.0, 18.0) for c in (0, 90, 180, 270)]
    assert "neither axis" in BlobTest.mirror_axis(rows)


def test_disagreeing_legs_refuse_to_name_an_axis():
    from blob_test import BlobTest
    rows = [(0, 10.0, 18.0), (90, 200.0, 18.0), (180, 40.0, 18.0),
            (270, 300.0, 18.0)]
    assert "disagree" in BlobTest.mirror_axis(rows)


def test_flipping_is_button_only_not_bound_to_a_key():
    """A stray keypress that mirrors the arena mid-run would be a bad way to
    find out a key was bound twice -- and the brackets were already the
    exposure controls."""
    import inspect

    import blob_test
    src = inspect.getsource(blob_test.BlobTest.key)
    assert "flip_axis" not in src


def test_the_app_constructs_and_both_tabs_build():
    """A guard against exactly what just happened: a new piece of state added
    to one place and not to `__init__`, so the ROBOT tab crashed the moment it
    was opened. The pure-function tests all passed through it."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import blob_test
    app = blob_test.BlobTest("sim", with_fleet=False)
    try:
        for tab in ("robot", "track"):
            app.set_tab(tab)
            assert app.buttons
        app.draw()
    finally:
        app.close()
