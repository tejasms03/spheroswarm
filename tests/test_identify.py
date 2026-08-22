"""Proving identity instead of inferring it.

The palette on disk has yellow at hue 60 +/- 9 and cyan at 62 +/- 10, so one
yellow ball is detected twice and two robots report one position. Nothing
static can separate those detections — they are the same pixels — so these
tests drive the LED and check that the blobs which follow are the ones that
should.

The palette here is built explicitly rather than loaded, so the tests describe
a collision rather than depending on the live calibration still having one.
"""

import cv2
import numpy as np
import pytest

from fleet.identify import (CONFIRMED, MISASSIGNED, PHANTOM, UNSEEN,
                            IdentityCheck, mask_areas, responded)
from vision.detect import Detector

# Yellow and cyan overlap, exactly as calib/colors.json does. Green is clear of
# both, so it is the control: a colour that must NOT follow anyone else's LED.
COLLIDING = {
    "yellow": {"hue": 60, "tol": 9, "s_min": 60, "v_min": 120},
    "cyan":   {"hue": 62, "tol": 10, "s_min": 60, "v_min": 120},
    "green":  {"hue": 85, "tol": 5, "s_min": 60, "v_min": 120},
}
SEPARATE = {
    "yellow": {"hue": 30, "tol": 8, "s_min": 60, "v_min": 120},
    "cyan":   {"hue": 90, "tol": 8, "s_min": 60, "v_min": 120},
    "green":  {"hue": 150, "tol": 8, "s_min": 60, "v_min": 120},
}


def _frame(balls):
    """`balls` is {hue: bool_lit}. A dark ball leaves the frame entirely, which
    is what a Sphero does when its LED goes out under a v_min floor."""
    f = np.zeros((240, 320, 3), np.uint8)
    for i, (hue, lit) in enumerate(balls.items()):
        if not lit:
            continue
        patch = np.full((1, 1, 3), (hue, 255, 255), np.uint8)
        bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.circle(f, (60 + i * 90, 120), 26, bgr, -1)
    return f


class FakeHandle:
    """A robot whose LED changes the picture — unless its ball is not there.

    `absent=True` models the case that matters most: the radio link is up and
    the commands are accepted, and nothing in the room changes. That is what a
    ball under a chair, out of frame, or with a dead LED looks like, and it is
    exactly the case a blob count cannot distinguish from a healthy robot.
    """

    def __init__(self, code, color, hue, world, absent=False):
        self.code, self.color, self.hue = code, color, hue
        self.rgb = (255, 255, 255)
        self.world = world              # {hue: lit}
        self.absent = absent
        self.led_calls = []

    def set_led(self, rgb, blink=None):
        self.led_calls.append(tuple(rgb))
        self.rgb = tuple(rgb)
        if not self.absent:
            self.world[self.hue] = any(rgb)


def _run(check, world, ticks=400, dt=0.05):
    """Step the check with a distinct stamp per frame, as the bench does from
    the tracker. Object identity is deliberately NOT used — see `_grab`."""
    for n in range(ticks):
        check.step(_frame(dict(world)), dt, stamp=n)
        if check.done:
            return True
    return False


# -- the response test itself --------------------------------------------

def test_a_blob_that_appears_with_the_led_responded():
    assert responded(0, 4000) is True


def test_a_blob_that_was_always_there_did_not():
    assert responded(4000, 4200) is False, "a reflection brightening slightly"


def test_a_few_flickering_pixels_did_not():
    assert responded(0, 12) is False


def test_a_blob_that_vanished_did_not():
    assert responded(4000, 0) is False


# -- measuring ------------------------------------------------------------

def test_mask_areas_uses_the_raw_mask_not_the_filtered_contour():
    """Half the palette has no min_area, so a check built on `detect()` would
    skip exactly the colours least likely to be right."""
    d = Detector(colors=COLLIDING)
    areas = mask_areas(d, _frame({60: True}))
    assert areas["yellow"] > 500
    assert areas["cyan"] > 500, "the collision is visible in the raw masks"
    assert areas["green"] == 0


# -- the verdicts ---------------------------------------------------------

def test_a_clean_palette_confirms_every_robot():
    world = {30: True, 90: True, 150: True}
    hs = [FakeHandle("SYRX", "yellow", 30, world),
          FakeHandle("SNFR", "cyan", 90, world),
          FakeHandle("VHGR", "green", 150, world)]
    c = IdentityCheck(hs, Detector(colors=SEPARATE), colors=list(SEPARATE))

    assert _run(c, world), "the check must finish"
    rep = c.report()
    assert rep["ok"] is True
    assert rep["confirmed"] == 3
    assert all(r["verdict"] == CONFIRMED for r in rep["results"].values())


def test_the_overlapping_palette_is_caught_red_handed():
    """The headline. ONE ball at hue 60, and two robots that both claim it."""
    world = {60: True}
    syrx = FakeHandle("SYRX", "yellow", 60, world)
    snfr = FakeHandle("SNFR", "cyan", 62, world, absent=True)
    c = IdentityCheck([syrx, snfr], Detector(colors=COLLIDING),
                      colors=list(COLLIDING))

    assert _run(c, world)
    rep = c.report()

    assert rep["ok"] is False
    assert rep["results"]["SYRX"]["verdict"] == PHANTOM
    assert set(rep["results"]["SYRX"]["followed"]) == {"yellow", "cyan"}
    assert "cyan" in rep["results"]["SYRX"]["why"]
    # And the robot that actually wears cyan is not seen at all, which is the
    # other half of the same fault.
    assert rep["results"]["SNFR"]["verdict"] == UNSEEN


def test_a_robot_the_tracker_cannot_see_is_named():
    world = {90: True}
    h = FakeHandle("SYRX", "yellow", 30, world, absent=True)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))

    assert _run(c, world)
    assert c.results["SYRX"]["verdict"] == UNSEEN
    assert "not watching" in c.results["SYRX"]["why"]


def test_a_swapped_roster_entry_is_named():
    """The robot is there and visible — wearing the wrong colour."""
    world = {90: True}
    h = FakeHandle("SYRX", "yellow", 90, world)        # roster says yellow
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))

    assert _run(c, world)
    r = c.results["SYRX"]
    assert r["verdict"] == MISASSIGNED
    assert r["followed"] == ["cyan"]


def test_a_contested_colour_is_reported_across_robots():
    """Two robots whose LEDs both move the same colour. Reported once, at the
    top, because it is a property of the pair rather than of either."""
    world = {60: True}
    a = FakeHandle("SYRX", "yellow", 60, world)
    b = FakeHandle("SNFR", "cyan", 60, world)          # same physical ball
    c = IdentityCheck([a, b], Detector(colors=COLLIDING), colors=list(COLLIDING))

    assert _run(c, world)
    rep = c.report()
    assert "yellow" in rep["contested_colours"]
    assert set(rep["contested_colours"]["yellow"]) == {"SYRX", "SNFR"}
    assert rep["ok"] is False


# -- behaviour of the pass itself -----------------------------------------

def test_the_led_is_given_back():
    world = {30: True}
    h = FakeHandle("SYRX", "yellow", 30, world)
    h.rgb = (12, 200, 90)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))

    assert _run(c, world)
    assert h.led_calls[0] == (0, 0, 0), "off first"
    assert h.rgb == (12, 200, 90), "and the robot's own colour restored"


def test_cancelling_mid_test_gives_the_led_back():
    world = {30: True}
    h = FakeHandle("SYRX", "yellow", 30, world)
    h.rgb = (12, 200, 90)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))

    for n in range(3):
        c.step(_frame(dict(world)), 0.05, stamp=n)
    assert h.rgb == (0, 0, 0), "mid-test, the LED is off"
    c.cancel()

    assert c.done and c.cancelled
    assert h.rgb == (12, 200, 90), "a cancelled check must not leave it dark"


def test_no_robot_is_ever_moved():
    """Safe to run with balls anywhere, including in someone's hand."""
    world = {30: True}
    h = FakeHandle("SYRX", "yellow", 30, world)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))
    _run(c, world)
    assert not hasattr(h, "_desired"), "set_velocity must never be called"


def test_repeated_reads_of_one_frame_are_not_two_samples():
    """A control loop outruns the camera. Counting reads rather than frames
    fills both bursts from a single instant, and every robot reads as UNSEEN."""
    world = {30: True}
    h = FakeHandle("SYRX", "yellow", 30, world)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))

    frame = _frame(dict(world))
    for _ in range(300):
        c.step(frame, 0.05, stamp=7)      # the same FRAME, 300 times
    assert not c.done, "one frame cannot complete a burst"


def test_object_identity_is_not_used_to_tell_frames_apart():
    """Freed frames are reallocated at the same address, so `id()` reports
    distinct pictures as repeats. The check must survive that."""
    world = {30: True}
    h = FakeHandle("SYRX", "yellow", 30, world)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))

    seen = set()
    for n in range(400):
        f = _frame(dict(world))           # temporary; ids will repeat
        seen.add(id(f))
        c.step(f, 0.05, stamp=n)
        if c.done:
            break
    assert len(seen) < 400, "the ids really do collide, which is the point"
    assert c.done, "and the check finished anyway"


def test_an_empty_fleet_is_done_immediately():
    c = IdentityCheck([], Detector(colors=SEPARATE))
    assert c.done is True
    assert c.report()["summary"] == "nothing checked"


def test_a_missing_frame_does_not_advance_anything():
    world = {30: True}
    h = FakeHandle("SYRX", "yellow", 30, world)
    c = IdentityCheck([h], Detector(colors=SEPARATE), colors=list(SEPARATE))
    for n in range(20):
        c.step(None, 0.05, stamp=n)
    assert not c.done
    assert h.led_calls == [], "no camera means no reason to touch the LED"
