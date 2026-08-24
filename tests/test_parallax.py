"""The ball rides above the floor the homography was calibrated on.

A homography maps one plane. Calibrated on the floor, it reports every ball
pushed radially away from the point the camera looks straight down at, because
the ball's centre is ~3.7cm up and the camera sees it along a slanted ray.

Everything here is fitted from measurements the bench already takes, never from
a camera height typed in by hand — a number from a tape measure goes stale the
first time somebody nudges the tripod, and nothing would notice.
"""

import numpy as np
import pytest

from vision import parallax


def _samples(scale, nadir=(70.0, 55.0), noise=0.0, seed=0,
             pts=((20, 20), (120, 20), (120, 95), (20, 95), (70, 55), (100, 40))):
    rng = np.random.default_rng(seed)
    n = np.asarray(nadir, dtype=float)
    out = []
    for p in pts:
        p = np.asarray(p, dtype=float)
        obs = n + (p - n) * scale
        if noise:
            obs = obs + rng.normal(0, noise, 2)
        out.append((p, obs))
    return out


def test_it_recovers_a_geometry_it_was_not_told():
    cam, ball = 240.0, 3.7
    truth = cam / (cam - ball)
    f = parallax.fit(_samples(truth, noise=0.25))
    assert f["scale"] == pytest.approx(truth, abs=0.002)
    assert f["nadir_cm"][0] == pytest.approx(70.0, abs=2.0)
    assert f["nadir_cm"][1] == pytest.approx(55.0, abs=2.0)


def test_the_fitted_scale_implies_the_real_ball():
    """A sanity check for a person: a fit that implies 3-4cm is measuring a
    Sphero. One implying 30cm is measuring something else, however well it
    fits."""
    cam = 240.0
    f = parallax.fit(_samples(cam / (cam - 3.7), noise=0.2))
    assert parallax.ball_height_cm(f, cam) == pytest.approx(3.7, abs=0.6)


def test_correcting_undoes_the_error():
    truth = 1.05
    s = _samples(truth, noise=0.1, seed=3)
    f = parallax.fit(s)
    before = np.mean([np.linalg.norm(o - t) for t, o in s])
    after = np.mean([np.linalg.norm(parallax.correct(o, f) - t) for t, o in s])
    assert after < before / 3.0, f"{before:.2f}cm -> {after:.2f}cm"


# -- it has to refuse more often than it accepts -------------------------

def test_a_camera_with_no_parallax_gets_no_correction():
    """The common and correct answer for a camera mounted high or overhead.
    A fit that always finds something is fitting noise."""
    assert parallax.fit(_samples(1.0, noise=0.3)) is None


def test_too_few_points():
    assert parallax.fit(_samples(1.05)[:2]) is None


def test_points_all_in_one_place_say_nothing():
    p = np.array([70.0, 55.0])
    assert parallax.fit([(p, p + 1.0)] * 6) is None


def test_noise_that_the_model_does_not_explain_is_refused():
    rng = np.random.default_rng(1)
    s = [(np.array(p, dtype=float), np.array(p, dtype=float) + rng.normal(0, 4, 2))
         for p in ((20, 20), (120, 20), (120, 95), (20, 95), (70, 55))]
    f = parallax.fit(s)
    assert f is None or f["residual_cm"] < f["was_cm"]


def test_correct_is_a_no_op_without_a_fit():
    p = np.array([30.0, 40.0])
    assert np.allclose(parallax.correct(p, None), p)
    assert np.allclose(parallax.correct(p, {"scale": 1.0, "nadir_cm": [0, 0]}), p)
    assert np.allclose(parallax.correct(p, {"nonsense": True}), p)


# -- it must not outlive the calibration it was measured against ---------

def test_recalibrating_the_arena_clears_the_fit():
    """A correction measured against corners that no longer exist is worse
    than no correction."""
    from vision.homography import Homography

    h = Homography()
    h.set_rect([(100, 100), (500, 110), (510, 400), (90, 390)], 138.8, 110.8)
    h.parallax = {"nadir_cm": [70.0, 55.0], "scale": 1.05}
    h.set_rect([(90, 95), (505, 105), (515, 395), (85, 385)], 138.8, 110.8)
    assert h.parallax is None


def test_the_homography_applies_it_to_every_point():
    from vision.homography import Homography

    h = Homography()
    h.set_rect([(100, 100), (500, 110), (510, 400), (90, 390)], 138.8, 110.8)
    plain = h.to_cm([[300, 250], [200, 200]])
    h.parallax = {"nadir_cm": [69.4, 55.4], "scale": 1.08}
    pulled = h.to_cm([[300, 250], [200, 200]])
    n = np.array([69.4, 55.4])
    for a, b in zip(plain, pulled):
        assert np.linalg.norm(b - n) < np.linalg.norm(a - n) + 1e-9, "must pull inward"


def test_a_parallax_too_small_to_matter_is_refused():
    """`MIN_SCALE` on its own terms. A clean fit at 1.001 is a real geometric
    signal and still worth nothing: it moves a robot by a millimetre across
    the whole arena, which is inside the tracker's own noise. Applying it is
    fitting noise with extra steps, and it earns a correction step in every
    frame forever."""
    clean = parallax.fit(_samples(1.001, noise=0.0))
    assert clean is None, "a scale below MIN_SCALE must be refused"
    assert parallax.MIN_SCALE > 1.0

    # ...and one comfortably above it is taken, so the guard is a threshold
    # rather than a blanket refusal.
    assert parallax.fit(_samples(parallax.MIN_SCALE * 4, noise=0.0)) is not None


def test_to_px_is_the_exact_inverse_of_to_cm():
    """Or every overlay drawn from a tracked position lands where the blob is
    not — displaced by exactly the correction, which reads as a tracking fault
    and is not one."""
    from vision.homography import Homography

    h = Homography()
    h.set_rect([(100, 100), (500, 110), (510, 400), (90, 390)], 138.8, 110.8)
    px = [[300.0, 250.0], [180.0, 300.0], [420.0, 200.0]]

    assert np.allclose(h.to_px(h.to_cm(px)), px, atol=1e-6), "no parallax"

    h.parallax = {"nadir_cm": [69.4, 55.4], "scale": 1.08}
    assert np.allclose(h.to_px(h.to_cm(px)), px, atol=1e-6), "with parallax"


def test_uncorrect_undoes_correct():
    p = np.array([30.0, 90.0])
    fit = {"nadir_cm": [69.4, 55.4], "scale": 1.06}
    assert np.allclose(parallax.uncorrect(parallax.correct(p, fit), fit), p)


def test_uncorrect_is_a_no_op_without_a_fit():
    p = np.array([30.0, 40.0])
    assert np.allclose(parallax.uncorrect(p, None), p)
    assert np.allclose(parallax.uncorrect(p, {"scale": 1.0, "nadir_cm": [0, 0]}), p)
