"""Choosing hues for the room you are in, rather than the room the defaults assumed.

The optimiser trades two things against each other, and the tests are mostly
about that trade being made the right way round: separation is a hard
constraint — two robots inside each other's tolerance windows swap identities —
while clarity is soft, and a colour competing with the floor is merely harder to
detect. A version that buys clarity by giving up separation looks better on
every number and is worse on the floor.
"""

import cv2
import numpy as np
import pytest

from vision import config as vconfig
from vision.palette import (HUES, MIN_SEPARATION, background_profile, clarity,
                            optimise, room_light, score_existing, summarise)


def room(paint=None, base=(70, 70, 70), size=(240, 320)):
    img = np.full((size[0], size[1], 3), base, np.uint8)
    if paint:
        paint(img)
    return img


RED_MAT = lambda i: cv2.rectangle(i, (0, 150), (320, 240), (40, 40, 200), -1)
BLUE_CASE = lambda i: cv2.rectangle(i, (10, 10), (140, 130), (200, 90, 40), -1)


def sep(hues):
    def d(a, b):
        x = abs(a - b) % HUES
        return min(x, HUES - x)
    return min(d(a, b) for i, a in enumerate(hues) for b in hues[i + 1:])


# -- reading the room --------------------------------------------------------

def test_a_plain_room_occupies_no_hue():
    hist = background_profile(room())
    assert float(hist.max()) < 0.05, "grey should not read as a colour"


def test_a_red_mat_shows_up_where_red_is():
    hist = background_profile(room(RED_MAT))
    peak = int(np.argmax(hist))
    assert min(peak, HUES - peak) < 15, f"red should peak near 0, peaked at {peak}"


def test_a_beige_wall_is_not_mistaken_for_orange():
    """Weighting by saturation is what stops a pale wall claiming a third of the wheel."""
    pale = room(base=(180, 190, 200))          # washed out, barely coloured
    assert float(background_profile(pale).max()) < 0.05


def test_room_light_notices_a_dim_room_and_a_blown_one():
    assert "dim" in room_light(room(base=(20, 20, 20)))["verdict"]
    assert "blown" in room_light(room(base=(255, 255, 255)))["verdict"]
    assert room_light(room(base=(90, 90, 90)))["verdict"] == "workable"


# -- choosing ----------------------------------------------------------------

def test_with_nothing_in_the_way_it_spreads_evenly():
    hues = [e["hue"] for e in optimise(6, background_profile(room()))]
    assert sep(hues) >= 29, hues


def test_it_does_not_creep_downhill_on_a_tie():
    """Every hue ties on a plain floor; ranking on cost alone crams them low."""
    hues = [e["hue"] for e in optimise(6, background_profile(room()))]
    assert max(hues) > 120, f"the palette collapsed into the low hues: {hues}"


def test_it_moves_off_a_colour_the_room_already_has():
    hist = background_profile(room(RED_MAT))
    before = summarise(score_existing([c["hue"] for c in vconfig.COLORS.values()], hist))
    after = summarise(optimise(6, hist))
    assert after["worst_clarity"] > before["worst_clarity"]
    assert after["worst_separation"] >= before["worst_separation"]


def test_separation_is_never_traded_away_for_clarity():
    """The trade that must not happen, in the room most likely to provoke it."""
    hist = background_profile(room(lambda i: (RED_MAT(i), BLUE_CASE(i))))
    hues = [e["hue"] for e in optimise(6, hist)]
    assert sep(hues) >= MIN_SEPARATION, f"{hues} -> {sep(hues)}"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_it_works_for_any_fleet_size(n):
    entries = optimise(n, background_profile(room(RED_MAT)))
    assert len(entries) == n
    if n > 1:
        assert sep([e["hue"] for e in entries]) >= min(MIN_SEPARATION, HUES // n - 1)


def test_no_robots_is_not_an_error():
    assert optimise(0) == []


def test_it_runs_with_no_camera_at_all():
    """A bench with no frame yet must still be usable."""
    hues = [e["hue"] for e in optimise(6)]
    assert sep(hues) >= 29


# -- scoring -----------------------------------------------------------------

def test_clarity_is_one_where_the_room_is_empty():
    hist = background_profile(room(RED_MAT))
    assert clarity(hist, 90) > 0.9, "green is nowhere near a red mat"
    assert clarity(hist, 0) < 0.9, "red is exactly where the mat is"


def test_summarise_flags_a_palette_that_cannot_be_told_apart():
    unsafe = score_existing([10, 14, 60, 90, 120, 150], np.zeros(HUES))
    assert not summarise(unsafe)["safe"]
    assert summarise(unsafe)["worst_separation"] == 4


def test_the_defaults_are_reported_honestly_on_a_hostile_floor():
    hist = background_profile(room(RED_MAT))
    got = summarise(score_existing([c["hue"] for c in vconfig.COLORS.values()], hist))
    assert got["worst_clarity"] < 0.9, "red-on-red should not score clean"


# -- the LED follows the hue -------------------------------------------------

def test_the_led_colour_is_derived_from_the_hue_it_looks_for():
    """One table, not two. Two is how a ball glows one colour and is hunted as another."""
    for hue in (0, 30, 60, 90, 120, 150, 179):
        r, g, b = vconfig.led_rgb(hue)
        back = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        assert abs(int(back[0]) - hue) <= 1 or abs(int(back[0]) - hue) >= HUES - 1
        assert int(back[1]) > 200, "the LED should be driven saturated"


def test_led_for_follows_a_retuned_slot():
    colors = {n: dict(s) for n, s in vconfig.COLORS.items()}
    colors["green"]["hue"] = 105
    assert vconfig.led_for("green", colors) == vconfig.led_rgb(105)


def test_an_unknown_slot_does_not_explode():
    assert vconfig.led_for("chartreuse", {}) == (255, 255, 255)
