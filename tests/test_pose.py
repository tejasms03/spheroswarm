"""Pose from three dots, and every frame the method refuses to answer on.

The refusals are the point of these tests, not an afterthought to them.
`pose_test.py` is allowed to return nothing and is never allowed to return a
plausible-looking wrong heading, because a silent frame is recovered by the
next one and a confident backwards heading goes into a controller. So each
"cannot read this" branch gets a test that pins the REASON, not merely the
absence of an answer.

Scenes come from `vision/shots.py`, which renders from a known pose, so the
truth here is generated rather than asserted by eye.
"""

import numpy as np
import pytest

from pose_test import (BallSource, GROUP_PX, V_MIN, chromaticity,
                       exposure_verdict, find_dots, read_frame, read_group)
from vision import config
from vision.shots import draw_ball

PX_CM = 9.2
GAIN = BallSource.GAIN


def render(balls, size=(520, 520), px_cm=PX_CM, noise=0.0, seed=0):
    """`balls` is [(x_cm, y_cm, heading_deg, tag)] -> one exposed frame."""
    canvas = np.zeros((size[1], size[0], 3), np.float32)
    canvas[:] = (6.0, 5.0, 4.0)
    for x, y, heading, tag in balls:
        draw_ball(canvas, x, y, heading, tag, px_cm)
    if noise:
        canvas += np.random.default_rng(seed).normal(0.0, noise, canvas.shape)
    return np.clip(canvas * GAIN, 0, 255).astype(np.uint8)


def only(frame, **kw):
    """The single pose in a frame, failing loudly if there is not exactly one."""
    poses, why, _, _ = read_frame(frame, **kw)
    assert len(poses) == 1, f"expected one pose, got {len(poses)}; refused {why}"
    return poses[0]


# -- the happy path --------------------------------------------------------

@pytest.mark.parametrize("heading", [0.0, 37.0, 90.0, 168.0, 213.0, 299.0, 355.0])
def test_heading_is_read_from_the_tail(heading):
    pose = only(render([(28.0, 28.0, heading, "red")]))
    assert abs((pose["deg"] - heading + 180) % 360 - 180) < 2.0


@pytest.mark.parametrize("tag", ["red", "yellow", "green", "cyan", "magenta"])
def test_identity_from_chromaticity(tag):
    assert only(render([(28.0, 28.0, 45.0, tag)]))["name"] == tag


def test_position_is_the_midpoint_not_the_centroid():
    """The midpoint of the tag dots is the centre; the lit centroid is not.

    Three dots with the tail behind puts the centroid of the LIT REGION about
    a centimetre back along the axis, and that offset rotates with the robot --
    so it cannot be calibrated out with a constant. This is the reason the
    method takes the midpoint of two symmetric dots instead, and this test is
    what would catch a well-meaning change back to a centroid.
    """
    x_cm, y_cm = 28.0, 26.0
    pose = only(render([(x_cm, y_cm, 0.0, "red")]))
    truth = np.array([x_cm * PX_CM, y_cm * PX_CM])
    assert np.linalg.norm(pose["centre"] - truth) < 1.5      # px, ~1.6 mm

    dots, _ = find_dots(render([(x_cm, y_cm, 0.0, "red")]))
    lit = np.mean([d["xy"] for d in dots], axis=0)
    assert np.linalg.norm(lit - truth) > np.linalg.norm(pose["centre"] - truth)


def test_red_and_green_are_told_apart():
    """The case a plain brightness sum cannot do at all.

    The two LEDs the palette actually asks for are pure red and pure green, so
    their channel sums are not merely close, they are IDENTICAL -- a sum
    carries exactly no information about which of the two you are looking at.
    Chromaticity puts them a third of the space apart.
    """
    red = np.array(config.led_rgb(config.COLORS["red"]["hue"]), dtype=float)
    green = np.array(config.led_rgb(config.COLORS["green"]["hue"]), dtype=float)
    assert red.sum() == green.sum()
    assert np.linalg.norm(chromaticity(red) - chromaticity(green)) > 0.5


def test_every_palette_pair_is_separable():
    """No two tag colours may collide, or the roster could hand out a pair
    that the identity step can never tell apart."""
    names = [n for n in config.COLORS if n != "blue"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca = chromaticity(config.led_rgb(config.COLORS[a]["hue"]))
            cb = chromaticity(config.led_rgb(config.COLORS[b]["hue"]))
            assert np.linalg.norm(ca - cb) > 0.1, f"{a} and {b} collide"


def test_two_robots_are_two_poses():
    frame = render([(16.0, 26.0, 10.0, "red"), (40.0, 26.0, 200.0, "green")],
                   size=(620, 520))
    poses, why, _, _ = read_frame(frame)
    assert len(poses) == 2, why
    assert {p["name"] for p in poses} == {"red", "green"}


# -- the refusals ----------------------------------------------------------

def test_nothing_above_threshold():
    dark = np.zeros((200, 200, 3), np.uint8)
    poses, why, dots, _ = read_frame(dark)
    assert not poses and not dots
    assert why == [("nothing above threshold", [])]


def test_only_two_dots():
    """A ball with one dot lost below the threshold refuses and says so."""
    frame = render([(28.0, 28.0, 0.0, "red")])
    dots, _ = find_dots(frame)
    assert len(dots) == 3
    assert read_group(dots[:2])[1] == "only 2 dots"


def test_two_blues():
    """Two taillights in one group: which end is the back is unanswerable."""
    dots, _ = find_dots(render([(28.0, 28.0, 0.0, "red")]))
    blue = next(d for d in dots if 100 <= d["hue"] <= 130)
    tag = next(d for d in dots if not 100 <= d["hue"] <= 130)
    second = dict(blue, xy=tag["xy"])
    assert read_group([blue, second, tag])[1] == "two blues"


def test_no_blue_at_all():
    dots, _ = find_dots(render([(28.0, 28.0, 0.0, "red")]))
    tags = [d for d in dots if not 100 <= d["hue"] <= 130]
    pose, why = read_group(tags + [dict(tags[0])])
    assert pose is None and why.startswith("no blue")


def test_four_dots_in_one_group_refuses():
    """Two robots inside one grouping radius must not become one robot.

    Averaging four dots produces a centre between two balls and a heading that
    belongs to neither, and nothing downstream could tell that had happened.
    """
    frame = render([(26.0, 28.0, 0.0, "red"), (29.0, 28.0, 180.0, "green")])
    poses, why, dots, _ = read_frame(frame, radius=90)
    assert not poses
    assert "dots in one group" in why[0][0], why


def test_a_group_radius_that_is_too_small_splits_a_ball():
    """The failure mode the radius slider exists for, pinned so it stays legible.

    A ball whose dots do not chain together reports "only N dots" rather than a
    wrong pose -- which is the behaviour that makes the slider tunable by eye.
    """
    frame = render([(28.0, 28.0, 0.0, "red")])
    poses, why, _, _ = read_frame(frame, radius=4)
    assert not poses
    assert all("dot" in reason for reason, _ in why)


def test_every_refusal_is_a_reason_and_a_group():
    """The shape callers rely on, including on the frame-wide path.

    This is the bug the dock actually hit: "nothing above threshold" used to
    come back as a bare string while every other refusal was a pair, and the
    dock unpacked them all the same way -- so raising the threshold past the
    last dot crashed the app rather than telling you that is what you had done.
    """
    for frame, kw in ((np.zeros((200, 200, 3), np.uint8), {}),
                      (render([(28.0, 28.0, 0.0, "red")]), {"radius": 4}),
                      (render([(28.0, 28.0, 0.0, "red")]), {"v_min": 254})):
        _, why, _, _ = read_frame(frame, **kw)
        assert why
        for item in why:
            reason, group = item                 # must unpack, every time
            assert isinstance(reason, str) and isinstance(group, list)


# -- accuracy, against generated truth -------------------------------------

def test_accuracy_over_a_moving_scene():
    """What the method is worth when it does answer.

    Thresholds are deliberately loose against the measured numbers (0.12 mm
    and 0.17 deg median) -- this is a regression guard, not a restatement of
    today's result.
    """
    src = BallSource(n=4, seed=3)
    pos_err, ang_err, read, total, named = [], [], 0, 0, 0
    for _ in range(60):
        ok, frame = src.read()
        truth = [(src.pos[i] * src.px_cm, float(src.heading[i]), src.tags[i])
                 for i in range(len(src.tags))]
        poses, _, _, _ = read_frame(frame, v_min=V_MIN, radius=GROUP_PX)
        for tp, th, tag in truth:
            total += 1
            near = min(poses, key=lambda p: np.linalg.norm(p["centre"] - tp),
                       default=None)
            if near is None or np.linalg.norm(near["centre"] - tp) > 30:
                continue
            read += 1
            pos_err.append(np.linalg.norm(near["centre"] - tp) / src.px_cm)
            ang_err.append(abs((near["deg"] - th + 180) % 360 - 180))
            named += near["name"] == tag

    assert read / total > 0.95
    assert named == read
    assert np.median(pos_err) < 0.2                  # cm
    assert np.percentile(ang_err, 95) < 3.0          # degrees


# -- the exposure verdict --------------------------------------------------

def test_exposure_verdict_needs_two_samples():
    assert exposure_verdict({}) is None
    assert exposure_verdict({-7: (50.0, 900)}) is None


def test_exposure_that_is_ignored_says_so():
    msg, _ = exposure_verdict({-7: (50.0, 900), -3: (51.0, 910), 0: (50.5, 905)})
    assert "IGNORED" in msg


def test_exposure_that_works_is_recognised():
    msg, _ = exposure_verdict({-7: (20.0, 900), 0: (140.0, 4000)})
    assert "works" in msg


def test_a_dark_arena_does_not_look_like_an_ignored_control():
    """The regression this two-signal verdict exists for.

    A correctly underexposed frame is black, so its mean brightness moves by
    well under a count even when the exposure control is working perfectly.
    Judged on brightness alone this reads as "the camera ignored it", which
    would send somebody off to fight a control that was never broken.
    """
    msg, _ = exposure_verdict({-7: (0.4, 120), 0: (0.9, 1400)})
    assert "works" in msg
    msg, _ = exposure_verdict({-7: (0.4, 120), 0: (0.42, 124)})
    assert "IGNORED" in msg
