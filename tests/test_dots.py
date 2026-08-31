"""Heading from the two coloured lobes, and the frames that defeat it.

The premise of `vision/dots.py` is that an ordinary exposure CLIPS: the LED
cores saturate into the same flat colour as the shell around them, so the peak
method has nothing to find, while the colour of the two lobes survives intact.
The gate is the mirror image — peaks, no colour. These tests pin both halves,
because a method that quietly returns a plausible number on a frame it cannot
read is worse than one that fails.

Scenes are rendered by `vision/shots.py` from a known heading, so the truth
here is generated rather than asserted by eye.
"""

import numpy as np
import pytest

from vision.facing import explain_px
from vision.dots import (explain_hue_px, facing_hue_px, find_balls,
                             read_all, tail_range)
from vision.shots import as_exposed, as_gated, render_hdr

ARENA = (80.0, 60.0)
PX_CM = 9.2


def scene(bots, seed=0):
    """(exposure, gate) for one hand-written layout."""
    rng = np.random.default_rng(seed)
    hdr = render_hdr(bots, ARENA[0], ARENA[1], rng, px_cm=PX_CM)
    return as_exposed(hdr), as_gated(hdr)


def one(heading, tag="red", x=40.0, y=30.0):
    return [{"x_cm": x, "y_cm": y, "heading_deg": heading, "tag": tag}]


def at(x=40.0, y=30.0):
    return (x * PX_CM, y * PX_CM)


RADIUS = 7.4 * PX_CM / 2.0


# -- finding them --------------------------------------------------------

def test_it_finds_every_ball_in_both_views():
    bots = [{"x_cm": 20.0, "y_cm": 20.0, "heading_deg": 10.0, "tag": "red"},
            {"x_cm": 60.0, "y_cm": 22.0, "heading_deg": 200.0, "tag": "cyan"},
            {"x_cm": 40.0, "y_cm": 45.0, "heading_deg": 95.0, "tag": "green"}]
    exposure, gate = scene(bots)
    assert len(find_balls(exposure)) == 3
    assert len(find_balls(gate)) == 3


def test_balls_are_found_by_brightness_not_by_a_known_colour():
    """A robot nobody has calibrated is still a lit thing on the floor."""
    exposure, _ = scene(one(0.0, tag="magenta"))
    found = find_balls(exposure)
    assert len(found) == 1
    centre, radius, area = found[0]
    # The LIT region, not the shell: a glowing ball throws light past its own
    # edge, so this lands at roughly twice the 34px physical radius.
    assert 55.0 < radius < 130.0


def test_the_blob_centroid_is_not_the_ball_and_must_not_be_used_as_one():
    """It sits about a centimetre behind the centre, and it swings with the aim.

    The lights are not symmetric about the centre -- two tag LEDs are, but the
    taillight hangs off the back -- so the centroid of everything lit is pulled
    toward the tail. Measured at 8.7px, 0.94cm, on every heading tested.

    A constant offset would be calibratable. This one points wherever the robot
    points, so it is not: it is a centimetre of error that rotates. That is the
    whole reason position comes from the midpoint of the two SAME-COLOURED
    dots, which are symmetric by construction, and never from the blob.
    """
    import numpy as np
    offs = []
    for heading in range(0, 360, 30):
        exposure, _ = scene(one(float(heading)))
        centre, _, _ = find_balls(exposure)[0]
        offs.append(float(np.linalg.norm(np.array(centre) - np.array(at()))))
    assert min(offs) > 4.0, "if the blob centroid has become central, say so here"
    assert max(offs) < 15.0


# -- reading the heading -------------------------------------------------

@pytest.mark.parametrize("heading", list(range(0, 360, 30)))
def test_the_lobes_give_back_the_heading_they_were_drawn_with(heading):
    exposure, _ = scene(one(float(heading)))
    rows = read_all(exposure, use_peaks=False)
    assert len(rows) == 1 and rows[0]["hue"], "two lobes must produce a reading"
    err = abs((rows[0]["hue"][0] - heading + 180) % 360 - 180)
    # Measured across all twelve headings on this geometry: median 0.79deg,
    # worst 1.74. The bound is set from that with headroom rather than from
    # taste -- the old 1.0 was measured on the one-LED arrangement and simply
    # did not survive the geometry being corrected.
    assert err < 2.5, f"read {rows[0]['hue'][0]:.1f} for {heading}"


def test_the_patch_must_cover_the_taillight_or_the_heading_bends():
    """`find_balls` returns the LIT radius, and this is why it has to.

    The taillight sits further back than either tag LED, so a patch cut to the
    shell's own radius clips it -- and clipping one of the two things whose
    centroids define the heading bends the answer. Measured: 0.36deg worst on
    the lit radius against 8.8deg on the shell radius.

    This reversed once already. When the taillight was assumed to sit closer in,
    a tight patch was harmless and briefly looked preferable. Where the tail
    actually sits is a fact about the ball, so this test is pinned to the
    geometry in `vision/shots.py` and will move if that is corrected again.
    """
    worst_lit = worst_shell = 0.0
    for heading in range(0, 360, 30):
        exposure, _ = scene(one(float(heading)))
        lit = read_all(exposure, use_peaks=False)[0]["hue"]
        shell = facing_hue_px(exposure, at(), RADIUS)
        worst_lit = max(worst_lit, abs((lit[0] - heading + 180) % 360 - 180))
        if shell:
            worst_shell = max(worst_shell,
                              abs((shell[0] - heading + 180) % 360 - 180))
    assert worst_lit < 2.5
    assert worst_shell > worst_lit * 3.0, "a clipped taillight must show up"


def test_it_reads_the_exposure_that_defeats_the_peak_method():
    """The whole reason this module exists. Same frame, both methods."""
    hue = read_all(scene(one(35.0))[0], use_peaks=False)[0]["hue"]
    peaks, _ = explain_px(scene(one(35.0))[0], at(), RADIUS)
    assert hue is not None and abs((hue[0] - 35.0 + 180) % 360 - 180) < 2.5
    if peaks is not None:
        assert abs((peaks[0] - 35.0 + 180) % 360 - 180) > 15.0, \
            "if the peak method suddenly works here, this premise needs re-checking"


def test_brightness_ordering_cannot_find_the_front_of_this_ball():
    """`facing_px` decides the nose by which core is brighter. On this ball
    that is not a bias, it is an absence of information.

    The two brightest cores are the two TAG LEDs -- equal by construction and
    equidistant either side of the centre -- so the axis it recovers is right
    and the direction it puts on that axis is a coin toss. Measured here as a
    clean 180deg error rather than noise, which is the signature of a method
    reading a symmetry it has no way to break.

    Colour breaks it, because the odd dot out is the blue one and it is at the
    back. That is the whole reason this module exists alongside `facing.py`.
    """
    _, gate = scene(one(35.0))
    peaks, _ = explain_px(gate, at(), RADIUS)
    if peaks is not None:
        err = abs((peaks[0] - 35.0 + 180) % 360 - 180)
        assert err > 150.0, (f"read {peaks[0]:.1f} for 35 — if brightness has "
                             "started working here, this geometry has changed")
    # False colour has no hue in it, so the colour route declines rather than
    # guessing -- which is the honest answer on a gate frame.
    got, why = explain_hue_px(gate, at(), RADIUS)
    assert got is None and why


def test_every_tag_colour_works_and_none_of_them_is_blue():
    from vision.shots import TAGS
    assert "blue" not in TAGS, "blue is the taillight; a blue tag has no heading"
    for tag in TAGS:
        exposure, _ = scene(one(120.0, tag=tag))
        rows = read_all(exposure, use_peaks=False)
        assert rows and rows[0]["hue"], f"{tag} produced no reading"
        # magenta is the worst of the five at 1.58deg, green the best at 0.37.
        assert abs((rows[0]["hue"][0] - 120.0 + 180) % 360 - 180) < 2.5, tag


# -- refusing, out loud --------------------------------------------------

def test_a_ball_with_no_taillight_is_refused_rather_than_guessed():
    lo, hi = tail_range()
    exposure, _ = scene(one(0.0))
    hsv_free = exposure.copy()
    # Paint out the blue lobe: a tail that is switched off, or a robot tagged
    # blue, both arrive here as "one colour" and neither has a heading in it.
    import cv2
    hsv = cv2.cvtColor(hsv_free, cv2.COLOR_BGR2HSV)
    blue = (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
    hsv_free[blue] = (0, 0, 0)
    got, why = explain_hue_px(hsv_free, at(), RADIUS)
    assert got is None
    assert "blue" in why


def test_a_blob_off_the_frame_says_so():
    exposure, _ = scene(one(0.0))
    got, why = explain_hue_px(exposure, (-60, -60), RADIUS)
    assert got is None and "edge" in why


def test_empty_floor_produces_nothing_rather_than_a_reading():
    exposure, _ = scene([])
    assert find_balls(exposure) == []
    got, why = explain_hue_px(exposure, at(), RADIUS)
    assert got is None and why


# -- both together -------------------------------------------------------

def test_read_all_carries_the_disagreement_rather_than_resolving_it():
    exposure, _ = scene(one(35.0))
    rows = read_all(exposure)
    assert len(rows) == 1
    row = rows[0]
    assert row["hue"] is not None
    if row["peaks"] is not None:
        assert row["disagree_deg"] is not None and row["disagree_deg"] > 15.0
