"""Heading measured from the camera, not deduced from driving.

The project's premise is that a glowing sphere has no facing, so the aim frame
must be inferred by driving known legs. That inference is where the pain lives:
it needs clear floor, it needs a tracker following the right ball, it goes stale
on every reconnect, and it cannot separate a rotated frame from a mirrored one
without legs pointing several ways.

The premise is only true if you ignore the tail light. A Sphero has a main RGB
LED and a dim blue aim LED at the back, and two lights on one shell is an
orientation — measured every frame, with nothing driven.
"""

import math

import cv2
import numpy as np
import pytest

from vision.facing import facing_cm, facing_px


def ball(heading_deg, R=26, sep_frac=0.5, blur=0, one_light=False,
         front=(80, 80, 255), back=(200, 90, 40)):
    """A lit shell with a bright main LED and a dimmer tail. `heading_deg` is
    in IMAGE degrees: x right, y down."""
    f = np.zeros((200, 200, 3), np.uint8)
    c = np.array([100.0, 100.0])
    cv2.circle(f, tuple(c.astype(int)), R, (30, 30, 90), -1)
    r = math.radians(heading_deg)
    d = np.array([math.cos(r), math.sin(r)]) * R * sep_frac
    cv2.circle(f, tuple((c + d).astype(int)), 6, front, -1)
    if not one_light:
        cv2.circle(f, tuple((c - d).astype(int)), 5, back, -1)
    if blur:
        f = cv2.GaussianBlur(f, (blur | 1, blur | 1), 0)
    return f


@pytest.mark.parametrize("heading", list(range(0, 360, 30)))
def test_it_reads_the_heading_it_was_given(heading):
    got = facing_px(ball(heading), (100, 100), 26)
    assert got is not None, "two clear lights must produce a reading"
    deg, conf = got
    err = abs((deg - heading + 180) % 360 - 180)
    assert err < 3.0, f"told {heading}, read {deg:.1f}"
    assert conf > 0.0


def test_it_beats_the_driven_leg_estimator_on_accuracy():
    """The estimator this replaces manages 0.8deg median, 1.9deg at p90 — and
    only after the robot has travelled 25cm. This needs one frame."""
    errs = []
    for heading in range(0, 360, 15):
        got = facing_px(ball(heading), (100, 100), 26)
        assert got is not None
        errs.append(abs((got[0] - heading + 180) % 360 - 180))
    assert max(errs) < 2.0, f"worst {max(errs):.1f}deg"


def test_the_front_is_the_brighter_light():
    """Not the colour. The main LED is driven hard and the aim light is a dim
    marker, and keying on brightness keeps this working when somebody sets the
    main LED to blue."""
    lit = facing_px(ball(0.0), (100, 100), 26)
    flipped = facing_px(ball(0.0, front=(200, 90, 40), back=(80, 80, 255)),
                        (100, 100), 26)
    assert lit is not None and flipped is not None
    assert abs((lit[0] - flipped[0] + 180) % 360 - 180) > 150, (
        "swapping which light is brighter must reverse the reading")


# -- it has to refuse, and say nothing rather than something wrong -------

def test_one_light_is_not_an_orientation():
    assert facing_px(ball(45.0, one_light=True), (100, 100), 26) is None


def test_a_dark_frame_reads_nothing():
    assert facing_px(np.zeros((200, 200, 3), np.uint8), (100, 100), 26) is None


def test_lights_too_far_apart_are_two_robots():
    """A separation wider than the ball is not a ball.

    `sep_frac` is chosen so BOTH lights are still inside the patch window —
    otherwise this passes because only one light was visible, which tests the
    wrong thing and leaves the ceiling unpinned."""
    got = facing_px(ball(0.0, sep_frac=0.9), (100, 100), 26)
    assert got is None, f"separation ceiling did not bite: {got}"

    # ...and just inside the ceiling still reads, so it is a threshold and not
    # a blanket refusal.
    assert facing_px(ball(0.0, sep_frac=0.6), (100, 100), 26) is not None


def test_a_blob_at_the_frame_edge_is_refused_rather_than_guessed():
    f = ball(0.0)
    assert facing_px(f, (1, 1), 26) is None


# -- arena coordinates ---------------------------------------------------

def test_the_arena_reading_goes_through_the_homography():
    """As two POINTS, not as an angle. A perspective map does not preserve
    angles, so rotating a bearing by whatever the matrix does at the image
    centre is wrong everywhere else in a tilted view — and a tilted view is
    what an overhead camera on a tripod is."""
    from vision.homography import Homography

    h = Homography()
    h.set_rect([(0, 0), (200, 0), (200, 200), (0, 200)], 100.0, 100.0)
    got = facing_cm(ball(0.0), (100, 100), 26, h)
    assert got is not None
    deg, _ = got
    # image +x with this square calibration is arena +x, which is compass 90
    assert abs((deg - 90.0 + 180) % 360 - 180) < 5.0, deg


def test_no_homography_means_no_arena_reading():
    assert facing_cm(ball(0.0), (100, 100), 26, None) is None
