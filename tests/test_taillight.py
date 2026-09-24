"""The taillight lab: the measurement, the lighting map, and what it refuses.

`Lab` is deliberately pygame-free so the vision can be tested without a window
— `App` is only the chrome around it. Everything here runs on synthetic frames
and never opens a camera, a Bluetooth link, or a calibration file for writing.
"""

import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import pygame
import pytest

from taillight import Lab, _wrap
from vision import config
from vision.tracks import wrap180
from vision.homography import Homography


def lit(heading_deg, centre=(300, 220), R=22, sep_frac=0.5, hue=0,
        size=(640, 480), one_light=False):
    """A lit shell of a given hue with a bright front LED and a dim tail.

    `heading_deg` is in IMAGE degrees — x right, y down — which is the frame
    `facing_px` answers in.
    """
    f = np.full((size[1], size[0], 3), 12, np.uint8)
    c = np.array(centre, dtype=float)
    body = np.uint8([[[hue, 235, 200]]])
    bgr = tuple(int(v) for v in cv2.cvtColor(body, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(f, tuple(c.astype(int)), R, bgr, -1)
    r = math.radians(heading_deg)
    d = np.array([math.cos(r), math.sin(r)]) * R * sep_frac
    cv2.circle(f, tuple((c + d).astype(int)), 5, (255, 255, 255), -1)
    if not one_light:
        cv2.circle(f, tuple((c - d).astype(int)), 4, (150, 150, 150), -1)
    return f


SIGNATURE = {"hue": 0, "tol": 10, "s_min": 90, "v_min": 70,
             "min_area": 60, "max_area": 20000}


def pin(lab):
    """Pin the colour signature instead of inheriting the live calibration.

    `calib/colors.json` is tuned for whatever room the camera last looked at —
    red currently sits at hue 138 there, not the 0 in `config.COLORS`. A test
    that inherits it passes or fails according to a file nobody edited today,
    which is how twelve tests in this repo once broke on a recalibration.
    """
    lab.detector.colors["red"] = dict(SIGNATURE)
    lab.detector.thresh.update({k: v for k, v in SIGNATURE.items()
                                if k not in ("hue", "tol")})
    lab.detector.thresh["blur"] = 5
    return lab


@pytest.fixture
def lab():
    """A lab on the synthetic source — no camera, no calibration written."""
    return pin(Lab(source="synthetic", color="red"))


@pytest.fixture
def blob(lab):
    """The same lab reading the older route: hue blob first, heading inside it.

    Kept tested alongside the default. It is the fallback when the lights
    cannot be separated, and a fallback nobody exercises is not a fallback.
    """
    lab.mode = "blob"
    return lab


# -- the reading ---------------------------------------------------------

@pytest.mark.parametrize("heading", [0, 45, 90, 180, 270, 315])
def test_it_reads_the_heading_off_a_lit_ball(blob, heading):
    r = blob.analyse(lit(heading))
    assert r["theta_img"] is not None, r["why"]
    err = abs((r["theta_img"] - heading + 180) % 360 - 180)
    assert err < 5.0, f"told {heading}, read {r['theta_img']:.1f}"
    assert r["centre_px"] is not None and r["radius_px"] > 4


def test_a_missing_ball_names_the_thresholds_rather_than_saying_nothing():
    """'no reading' leaves a person turning knobs at random. The refusal has
    to say which knob."""
    lab = pin(Lab(source="synthetic", color="red"))
    lab.mode = "blob"
    r = lab.analyse(np.zeros((480, 640, 3), np.uint8))
    assert r["theta_img"] is None
    assert "no red blob" in r["why"]
    assert "hue" in r["why"] and "area" in r["why"]


def test_one_light_is_refused_with_the_knob_to_turn(blob):
    r = blob.analyse(lit(0.0, one_light=True))
    assert r["theta_img"] is None
    assert "one peak" in r["why"] or "seen twice" in r["why"]
    assert "peak floor" in r["why"] or "focus" in r["why"] or "exposure" in r["why"]


def test_a_frame_that_never_arrived_is_not_an_error(lab):
    r = lab.analyse(None)
    assert r["theta_img"] is None and r["why"] == "no frame yet"


def test_the_blob_is_found_in_the_blurred_frame_and_the_heading_in_the_sharp_one(blob):
    """Defocus is what makes a shell an easy blob and what merges two LEDs
    into one. Reading both off one image would mean spoiling one of them."""
    blob.unfocus = 21
    r = blob.analyse(lit(90.0))
    assert r["centre_px"] is not None, "a heavy blur must still find the blob"
    assert r["theta_img"] is not None, (
        "the heading must be read from the sharp frame, not the blurred one")
    assert abs((r["theta_img"] - 90 + 180) % 360 - 180) < 5.0


def test_unfocus_is_the_detectors_own_blur_and_not_a_second_one(lab):
    lab.unfocus = 12
    assert lab.detector.thresh["blur"] == 13, "even blur kernels are illegal"
    assert lab.unfocus == 13


# -- arena coordinates ---------------------------------------------------

def test_without_a_calibration_it_stays_in_pixels_rather_than_inventing_cm(blob):
    blob.hom = Homography()
    r = blob.analyse(lit(0.0))
    assert r["xy_cm"] is None and r["theta_cm"] is None
    assert r["theta_img"] is not None, "the image reading still stands"


def test_with_a_calibration_it_reports_arena_position_and_a_compass_heading(blob):
    h = Homography()
    h.set_rect([(0, 0), (640, 0), (640, 480), (0, 480)], 138.8, 110.8)
    blob.hom = h
    r = blob.analyse(lit(0.0, centre=(320, 240)))
    assert r["xy_cm"] is not None
    x, y = r["xy_cm"]
    assert 60 < x < 80 and 45 < y < 65, (x, y)
    # image +x under this calibration is arena +x, which is compass 90
    assert abs((r["theta_cm"] - 90.0 + 180) % 360 - 180) < 6.0, r["theta_cm"]


# -- the brightness map --------------------------------------------------

def test_the_map_covers_the_arena_and_not_the_room(lab):
    """A bright window behind the floor is not a lighting problem, and
    averaged into a whole-frame number it hides one that is."""
    h = Homography()
    h.set_rect([(100, 100), (400, 100), (400, 300), (100, 300)], 138.8, 110.8)
    lab.hom = h
    frame = np.full((480, 640, 3), 10, np.uint8)
    frame[0:80, 0:640] = 255                    # a blown-out window, outside
    img, stats, why = lab.brightness_map(frame, 900, 506)
    assert why is None and img is not None
    assert stats["blown"] == 0.0, "the window is outside the arena"
    assert stats["mean"] < 30


def test_clipping_inside_the_arena_is_reported(lab):
    h = Homography()
    h.set_rect([(100, 100), (400, 100), (400, 300), (100, 300)], 138.8, 110.8)
    lab.hom = h
    frame = np.full((480, 640, 3), 10, np.uint8)
    frame[150:250, 150:350] = 255               # glare on the floor itself
    _, stats, _ = lab.brightness_map(frame, 900, 506)
    assert stats["blown"] > 5.0
    assert stats["max"] == 255


def test_without_a_calibration_the_map_falls_back_and_says_so(lab):
    lab.hom = Homography()
    img, stats, why = lab.brightness_map(np.full((480, 640, 3), 90, np.uint8),
                                         900, 506)
    assert img is not None and stats is not None
    assert "no arena calibration" in why
    assert 85 < stats["mean"] < 95


def test_the_map_is_an_image_the_size_it_was_asked_to_fit(lab):
    h = Homography()
    h.set_rect([(0, 0), (640, 0), (640, 480), (0, 480)], 138.8, 110.8)
    lab.hom = h
    img, _, _ = lab.brightness_map(np.full((480, 640, 3), 60, np.uint8), 900, 506)
    assert img.shape[2] == 3
    assert img.shape[1] <= 900 and img.shape[0] <= 506
    # 138.8 x 110.8 into a 900 x 506 pane: the arena is proportionally wider
    # than the pane, so HEIGHT is what binds and the map is letterboxed.
    assert img.shape[0] == 506
    assert abs(img.shape[1] / img.shape[0] - 138.8 / 110.8) < 0.02


# -- it must not touch live state ----------------------------------------

def test_the_lab_never_writes_calibration(lab, tmp_path, monkeypatch):
    """Thresholds moved here are for looking at, not for keeping. A lab that
    saves is a lab that changes what the next real session measures."""
    import vision.config as vc

    def refuse(*a, **k):
        raise AssertionError("the lab wrote a calibration file")

    monkeypatch.setattr(vc, "save", refuse)
    monkeypatch.setattr(vc, "save_signatures", refuse)
    lab.unfocus = 9
    lab.floor = 40
    lab.detector.thresh["s_min"] = 33
    lab.analyse(lit(0.0))


def test_wrapping_keeps_every_word():
    text = "peaks 31.0px apart on a 22px ball — that is two robots"
    assert " ".join(_wrap(text, 20)).split() == text.split()
    assert all(len(ln) <= 20 for ln in _wrap(text, 20))


# -- the window ----------------------------------------------------------

@pytest.fixture
def app():
    """A real App on the synthetic source, headless. Closed either way."""
    from taillight import App
    a = App(source="synthetic", camera_size=(640, 480), color="red")
    pin(a.lab)
    a.set_color(a.lab.color)     # re-sync the slider to the pinned signature
    try:
        yield a
    finally:
        a.close()


def test_every_control_is_inside_the_window(app):
    """Two buttons once shipped drawn off the right edge of the bench, and a
    test that calls the handler never discovers that no click can reach it."""
    from taillight import W, H

    window = pygame.Rect(0, 0, W, H)
    for b in app.buttons:
        assert window.contains(b.rect), f"{b.label} at {b.rect} is off-window"
    for s in app.sliders:
        assert window.contains(s.rect), f"{s.label} at {s.rect} is off-window"
        assert window.contains(s.track), f"{s.label} track is off-window"


def test_no_two_controls_overlap(app):
    """A hit box that covers another silently steals its clicks."""
    boxes = ([(b.label, b.rect) for b in app.buttons]
             + [(s.label, s.rect) for s in app.sliders])
    for i, (an, ar) in enumerate(boxes):
        for bn, br in boxes[i + 1:]:
            assert not ar.colliderect(br), f"{an} overlaps {bn}"


def test_controls_stay_inside_the_window_with_a_scan_result(app):
    """The robot list grows the panel downwards — the one direction that can
    push the sliders under it off the bottom."""
    from taillight import W, H

    app.found = ["SK-4C2C", "SK-67D7", "SK-1111", "SK-2222"]
    app._build()
    window = pygame.Rect(0, 0, W, H)
    for s in app.sliders:
        assert window.contains(s.rect), f"{s.label} fell off with a full list"
    for b in app.buttons:
        assert window.contains(b.rect), f"{b.label} fell off with a full list"


def test_the_brightness_view_toggles_and_draws(app):
    assert app.view == "camera"
    app.toggle_view()
    assert app.view == "bright"
    app.tick()
    app.draw()                      # must not raise with no calibration either
    app.toggle_view()
    assert app.view == "camera"


def test_the_log_records_the_refusal_rather_than_an_empty_line(app):
    """A blank log line is indistinguishable from a bench that has stopped."""
    app.lab.mode = "blob"
    app.lab.hom = Homography()
    app.reading = app.lab.analyse(np.zeros((480, 640, 3), np.uint8))
    app.last_log = 0.0
    app.log_reading()
    stamp, ok, text = app.lines[-1]
    assert ok is None and text.strip()
    assert "no red blob" in text


def test_the_log_reports_a_good_reading_with_its_units(app):
    app.lab.mode = "blob"
    app.lab.hom = Homography()
    app.reading = app.lab.analyse(lit(0.0))
    app.last_log = 0.0
    app.log_reading()
    _, ok, text = app.lines[-1]
    assert ok is True, text
    assert "px" in text and "img" in text, "no calibration means pixels, said so"
    assert "conf" in text


def test_a_lab_with_no_camera_controls_says_so_rather_than_pretending(app):
    """Focus goes out of band through uvc-util, exactly like the shutter:
    OpenCV accepts `CAP_PROP_AUTOFOCUS` for an external camera and drops it,
    which is a button that does nothing while the lens keeps hunting. With no
    tool underneath, saying so is the whole job — a control that silently does
    nothing is worse than one that is absent, because a person keeps turning
    it."""
    app.focus_dial.binary = None
    app.focus_off()
    assert any("uvc-util" in t for t, _ in app.notes), list(app.notes)


def test_focus_and_shutter_are_separate_controls_on_one_camera(app):
    """Two dials, two files, one push implementation. Pinning the lens must
    not disturb the exposure."""
    from vision import shutter

    assert app.focus_dial.SAVE_KEY == "focus"
    assert app.dial.SAVE_KEY == "exposure"
    app.focus_dial.set(9999)
    assert app.focus_dial.value == shutter.FOCUS_MAX
    before = app.dial.value
    app.focus_dial.set(80)
    assert app.dial.value == before, "the shutter moved when focus did"


def test_one_hue_moves_both_the_detector_and_the_led(app):
    """`vision/config.py` keeps the colour a ball is told to glow and the hue
    the tracker hunts in ONE table, because two tables is how a robot ends up
    lit one colour and looked for as another. A lab that split them would
    reintroduce that bug on the bench people use to diagnose it."""
    sent = []
    fake = type("Fake", (), {"set_led": lambda _s, rgb: sent.append(rgb),
                             "set_back_led": lambda _s, v: None,
                             "link_up": True, "color": None})()
    # A ball wearing the slot the slider is editing.
    app.lab.robots["SK-TEST"] = fake
    app.lab.assigned["SK-TEST"] = app.lab.color
    app.set_hue(140)
    assert app.lab.detector.colors[app.lab.color]["hue"] == 140
    assert sent, "the LED must follow the same slider"
    assert sent[-1] == config.led_rgb(140, value=app.bright)


def test_a_ball_wearing_another_slot_is_not_relit_by_that_slider(app):
    """The slider edits ONE slot. A fleet lit from it would be a fleet all the
    same colour, which is the one arrangement in which colour tags nothing."""
    sent = []
    fake = type("Fake", (), {"set_led": lambda _s, rgb: sent.append(rgb),
                             "set_back_led": lambda _s, v: None,
                             "link_up": True, "color": None})()
    other = next(c for c in app.lab.detector.colors if c != app.lab.color)
    app.lab.robots["SK-OTHER"] = fake
    app.lab.assigned["SK-OTHER"] = other
    app.set_hue(140)
    assert sent[-1] == config.led_rgb(
        app.lab.detector.colors[other]["hue"], value=app.bright), \
        "it was relit at its OWN slot, not at the slider's"


def test_the_hunted_hue_actually_changes_what_is_found(app):
    """Not just the number on the slider — a signature edit nothing reads is
    the same as no edit at all."""
    app.lab.mode = "blob"
    frame = lit(0.0, hue=90)
    app.set_hue(0)
    assert app.lab.analyse(frame)["centre_px"] is None
    app.set_hue(90)
    app.lab.detector.colors[app.lab.color]["tol"] = 10
    assert app.lab.analyse(frame)["centre_px"] is not None


def test_the_hue_slider_starts_where_the_detector_actually_is():
    """Not at the table default. The live calibration overrides it, and a
    slider seeded from the table shows one number while the detector hunts
    another — before anybody has touched anything.

    Built without `pin`, deliberately: the invariant is that the slider agrees
    with whatever signature the app actually loaded, whatever that is today.
    """
    from taillight import App

    a = App(source="synthetic", camera_size=(640, 480), color="red")
    try:
        assert a.hue == a.lab.detector.colors["red"]["hue"]
    finally:
        a.close()


def test_switching_colour_brings_the_slider_with_it(app):
    app.set_color("green")
    assert app.lab.color == "green"
    assert app.hue == app.lab.detector.colors["green"]["hue"]


# -- the lights route: no colour tracking involved -----------------------

def shell(heading_deg, centre=(300, 220), span=52, size=(640, 480),
          main=(40, 180, 255), tail=(255, 120, 40), middle=True):
    """Three lights along one axis, the way a lit ball actually looks.

    `main` is the bright LED and `tail` is the blue aim light, both BGR — so
    the default tail really is blue-dominant. `heading_deg` is image degrees.
    """
    f = np.zeros((size[1], size[0], 3), np.uint8)
    c = np.array(centre, dtype=float)
    r = math.radians(heading_deg)
    d = np.array([math.cos(r), math.sin(r)]) * span / 2.0
    cv2.circle(f, tuple((c + d).astype(int)), 7, main, -1)
    if middle:
        cv2.circle(f, tuple(c.astype(int)), 4, (60, 190, 250), -1)
    cv2.circle(f, tuple((c - d).astype(int)), 6, tail, -1)
    return cv2.GaussianBlur(f, (9, 9), 0)


@pytest.mark.parametrize("heading", [0, 45, 90, 137, 180, 270, 315])
def test_the_lights_route_reads_a_heading_with_no_colour_tracking(lab, heading):
    lab.min_v = 150
    r = lab.analyse(shell(heading))
    assert r["theta_img"] is not None, r["why"]
    assert abs((r["theta_img"] - heading + 180) % 360 - 180) < 5.0
    assert len(r["group"]) == 3, "all three lights belong to one ball"


def test_it_works_on_a_hue_the_detector_would_never_find(lab):
    """The point of the route: the colour tracker is not consulted at all."""
    lab.min_v = 150
    lab.detector.colors["red"]["hue"] = 90      # nowhere near the lit ball
    r = lab.analyse(shell(30.0))
    assert r["theta_img"] is not None, r["why"]
    assert abs((r["theta_img"] - 30 + 180) % 360 - 180) < 5.0


def test_the_blue_end_is_the_back(lab):
    """`set_back_led` is Color(0, 0, n) on every toy that has one, so the tail
    is the bluest thing on the shell. Swapping the two must reverse the arrow
    — a heading 180 out is worse than no heading."""
    lab.min_v = 150
    fwd = lab.analyse(shell(0.0))
    back = lab.analyse(shell(0.0, main=(255, 120, 40), tail=(40, 180, 255)))
    assert fwd["theta_img"] is not None and back["theta_img"] is not None
    assert fwd["by_colour"] and back["by_colour"]
    assert abs((fwd["theta_img"] - back["theta_img"] + 180) % 360 - 180) > 150


def test_two_lights_are_enough(lab):
    lab.min_v = 150
    r = lab.analyse(shell(60.0, middle=False))
    assert r["theta_img"] is not None, r["why"]
    assert abs((r["theta_img"] - 60 + 180) % 360 - 180) < 5.0


def test_one_light_is_a_position_and_not_an_orientation(lab):
    lab.min_v = 150
    f = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(f, (300, 220), 7, (40, 180, 255), -1)
    r = lab.analyse(cv2.GaussianBlur(f, (9, 9), 0))
    assert r["theta_img"] is None
    assert "one light" in r["why"]
    assert r["centre_px"] is not None, "we still know where it is"


def test_a_dark_frame_names_the_floor_it_failed(lab):
    r = lab.analyse(np.zeros((480, 640, 3), np.uint8))
    assert r["theta_img"] is None
    assert f"V={lab.min_v}" in r["why"]


def test_lights_too_far_apart_are_not_one_ball(lab):
    """Two robots across the arena must not be read as one very long one."""
    lab.min_v = 150
    lab.span_px = 30
    r = lab.analyse(shell(0.0, span=200, middle=False))
    assert r["theta_img"] is None or len(r["group"]) == 1


def test_a_scattered_triangle_is_refused_rather_than_averaged(lab):
    """Three lights on a shell are collinear. Three that are not are a
    reflection, or two robots that happen to be close."""
    lab.min_v = 150
    f = np.zeros((480, 640, 3), np.uint8)
    for p in ((300, 200), (340, 210), (315, 260)):
        cv2.circle(f, p, 6, (40, 180, 255), -1)
    r = lab.analyse(cv2.GaussianBlur(f, (9, 9), 0))
    assert r["theta_img"] is None
    assert "not in a line" in r["why"], r["why"]


def test_every_light_is_reported_even_the_ones_it_did_not_use(lab):
    """Seeing the four spots it found is what tells you the fifth is a
    reflection. A view that draws only the answer cannot show you that."""
    lab.min_v = 150
    lab.span_px = 40
    f = shell(0.0)
    cv2.circle(f, (100, 400), 6, (255, 255, 255), -1)      # a stray reflection
    r = lab.analyse(cv2.GaussianBlur(f, (5, 5), 0))
    assert len(r["lights"]) >= 4
    assert all(l in r["lights"] for l in r["group"])
    assert len(r["group"]) < len(r["lights"]), "the stray is not on the ball"


def test_the_arena_heading_goes_through_the_homography_as_two_points(lab):
    lab.min_v = 150
    h = Homography()
    h.set_rect([(0, 0), (640, 0), (640, 480), (0, 480)], 138.8, 110.8)
    lab.hom = h
    r = lab.analyse(shell(0.0, centre=(320, 240)))
    assert r["xy_cm"] is not None
    # image +x under this calibration is arena +x, which is compass 90
    assert abs((r["theta_cm"] - 90.0 + 180) % 360 - 180) < 6.0, r["theta_cm"]


def test_lights_is_the_default_route(lab):
    assert lab.mode == "lights"


MINT_BGR, CYAN_BGR, CORAL_BGR = (150, 220, 130), (210, 190, 80), (70, 70, 210)


def three_lights(size=(400, 300)):
    """Tail at x=100, tag LEDs at 140 and 180, all on one line."""
    f = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.circle(f, (100, 150), 5, (255, 60, 40), -1)
    cv2.circle(f, (140, 150), 6, (60, 255, 90), -1)
    cv2.circle(f, (180, 150), 6, (60, 255, 90), -1)
    return f


def count(img, bgr):
    return int((img == bgr).all(-1).sum())


def test_the_mask_is_the_threshold_the_detector_actually_used():
    """Not a second derivation of "what counts as a light". A mask view that
    computes its own threshold can reassure you about a picture the detector
    never saw, which is the one failure it must not have."""
    from vision.facing import light_mask

    lab = Lab(source="synthetic", size=(400, 300), color="green")
    f = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    lab.min_v = 180
    _, mask = light_mask(f, 180)
    _, stats = lab.mask_view(f)
    assert abs(stats["lit_pct"] - float(mask.mean() * 100.0)) < 1e-9


def test_the_mask_separates_used_from_merely_lit_from_thrown_away():
    """The question while pulling the shutter down is never "is anything lit"
    — it is "is what I can see being USED". A plain white-on-black mask cannot
    answer that, so each region is coloured by what happens to it."""
    lab = Lab(source="synthetic", size=(400, 300), color="green")
    f = three_lights()
    cv2.circle(f, (330, 60), 5, (250, 250, 250), -1)          # a reflection
    cv2.rectangle(f, (0, 250), (399, 299), (240, 240, 240), -1)  # lit floor
    f = cv2.GaussianBlur(f, (5, 5), 0)

    r = lab.analyse_lights(f)
    assert r["why"] is None, r["why"]
    img, stats = lab.mask_view(f, r["group"])

    assert count(img, MINT_BGR) > 0, "the cluster the heading came from"
    assert count(img, CYAN_BGR) > 0, "a light that is not in that cluster"
    assert count(img, CORAL_BGR) > 0, "the floor, thrown away on area"
    # The floor is one huge region, not many — which is why lit_pct is the
    # number that catches a shutter still too long, and the count is not.
    assert stats["regions"] == 5 and stats["passed"] == 4
    assert stats["lit_pct"] > 5.0


def test_a_dark_frame_reads_as_nothing_lit_rather_than_raising():
    lab = Lab(source="synthetic", size=(400, 300), color="green")
    img, stats = lab.mask_view(np.zeros((300, 400, 3), np.uint8))
    assert stats["regions"] == 0 and stats["lit_pct"] == 0.0
    assert img.max() == 0, "nothing painted"


def test_the_mask_keeps_the_frames_own_geometry():
    """No blur and no homography: the overlay is drawn over this at the plain
    scale, so a mask of something else would put the arrow in the wrong place.
    """
    lab = Lab(source="synthetic", size=(400, 300), color="green")
    f = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    img, _ = lab.mask_view(f)
    assert img.shape == f.shape


def ball(f, cx, cy, deg, bgr, span=54, tag=26):
    r = math.radians(deg)
    ux, uy = math.cos(r), math.sin(r)
    cv2.circle(f, (int(cx - ux * span / 2), int(cy - uy * span / 2)), 7,
               (255, 70, 45), -1)
    cv2.circle(f, (int(cx - ux * tag / 2), int(cy - uy * tag / 2)), 9, bgr, -1)
    cv2.circle(f, (int(cx + ux * tag / 2), int(cy + uy * tag / 2)), 9, bgr, -1)
    return f


def test_every_cluster_is_reported_not_just_the_biggest():
    """`analyse_lights` answers about ONE ball. This is the other question —
    with several lit at once, which is which."""
    lab = Lab(source="synthetic", size=(1280, 720), color="red")
    f = np.full((720, 1280, 3), 6, np.uint8)
    ball(f, 300, 200, 0, (60, 60, 245))
    ball(f, 800, 400, 120, (235, 200, 60))
    ball(f, 500, 560, 250, (90, 230, 90))
    f = cv2.GaussianBlur(f, (7, 7), 0)

    got = lab.analyse_all(f)
    assert len([c for c in got if c["deg"] is not None]) >= 3
    assert all(c["centre"] is not None for c in got if c["why"] is None)


def test_a_cluster_it_cannot_read_is_kept_with_its_reason():
    """Dropping it would make a robot the reader is REFUSING to identify look
    exactly like a robot that is not there, and those need different fixes."""
    lab = Lab(source="synthetic", size=(1280, 720), color="red")
    f = np.full((720, 1280, 3), 6, np.uint8)
    ball(f, 300, 200, 0, (60, 60, 245))
    cv2.circle(f, (1000, 150), 8, (245, 245, 245), -1)       # lone reflection
    f = cv2.GaussianBlur(f, (7, 7), 0)

    got = lab.analyse_all(f)
    lone = [c for c in got if c["why"] is not None]
    assert lone, "the reflection was dropped instead of explained"
    assert "one light" in lone[0]["why"]
    assert lone[0]["deg"] is None and lone[0]["code"] is None


def test_the_label_is_what_vision_read_not_what_the_roster_wishes():
    """The roster maps a colour slot to a code. A cluster whose colour was not
    read must stay unnamed rather than borrowing a neighbour's row."""
    lab = Lab(source="synthetic", size=(1280, 720), color="red")
    lab.wearer = {"green": "VHGR"}
    f = cv2.GaussianBlur(ball(np.full((720, 1280, 3), 6, np.uint8),
                              400, 300, 0, (60, 60, 245)), (7, 7), 0)
    for c in lab.analyse_all(f):
        if c["color"] is None:
            assert c["code"] is None


def test_an_empty_frame_is_no_clusters_rather_than_a_crash():
    lab = Lab(source="synthetic", size=(640, 480), color="red")
    assert lab.analyse_all(np.zeros((480, 640, 3), np.uint8)) == []
    assert lab.analyse_all(None) == []


def test_a_missing_roster_costs_the_labels_and_nothing_else():
    """`roster.json` is live state. The lab reads it and never writes it, and
    a broken one must not take the bench down with it."""
    lab = Lab(source="synthetic", size=(640, 480), color="red")
    assert isinstance(lab.wearer, dict)


# -- painting every branch ------------------------------------------------
#
# A NameError shipped in `draw_profile` because the branch it lived on — a
# cluster with a `tag_why` and no `color` — is the state the bench spends its
# time in on real hardware and the one no test had ever painted. Rendering is
# not exercised by asserting on numbers; it has to actually be drawn.

def paint(app):
    """Draw a whole frame, in every view, with and without the labels."""
    for show_all in (False, True):
        app.show_all = show_all
        for view in ("camera", "mask", "bright"):
            app.view = view
            app._build()
            app.draw()
            assert app.screen.get_clip() == app.screen.get_rect(), \
                f"{view}/{show_all} left the surface clipped"


def test_every_view_paints_when_the_tag_cannot_be_named(app):
    """hue outside every tolerance: a heading reads, the NAME does not. This
    is the live-hardware state, and the one that crashed."""
    f = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    app.grab.frame = f
    app.reading = app.lab.analyse_lights(f)
    app.reading["tag_why"] = ("no light matches a known colour within its "
                              "tolerance (nearest is red, 8 away, outside "
                              "its tolerance) — tune the signature")
    app.reading["color"] = None
    app.all_reading = app.lab.analyse_all(f)
    paint(app)


def test_every_view_paints_when_there_is_no_reading_at_all(app):
    app.grab.frame = np.zeros((300, 400, 3), np.uint8)
    app.reading = app.lab.analyse_lights(app.grab.frame)
    app.all_reading = []
    assert app.reading["why"], "this fixture must be a refusal"
    paint(app)


def test_every_view_paints_before_the_first_frame(app):
    app.grab.frame = None
    app.reading = app.lab.analyse(None)
    app.all_reading = []
    paint(app)


def test_every_view_paints_a_named_cluster(app):
    f = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    app.grab.frame = f
    app.reading = app.lab.analyse_lights(f)
    app.reading["color"] = "red"
    app.all_reading = app.lab.analyse_all(f)
    paint(app)


def test_a_long_refusal_does_not_escape_its_card(app):
    """Text drawn past a card edge does not vanish — it lands on the panel
    next door, over the sliders, which reads as the app being broken."""
    f = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    app.grab.frame = f
    app.reading = app.lab.analyse_lights(f)
    long = "cannot tell the front from the back: " + "no colour tag, " * 20
    app.reading["why"] = long
    app.reading["tag_why"] = long
    app.say(long)
    app.view = "camera"
    app._build()
    app.draw()

    import taillight as T
    arr = pygame.surfarray.array3d(app.screen)
    # The gutter between the camera view and the control panel must stay dark.
    gutter = arr[T.VIEW.right + 2:T.PANEL_X - 6, T.VIEW.y:T.VIEW.bottom]
    assert gutter.max() <= 60, f"text escaped into the gutter ({gutter.max()})"


# -- several balls, tagged by colour --------------------------------------

class FakeBall:
    """A handle that records what it was told to glow."""
    link_up = True

    def __init__(self, **kw):
        self.ble = kw.get("ble_name")
        self.color = kw.get("color")
        self.rgb = None
        self.tail = None

    def set_led(self, rgb, blink=None):
        self.rgb = rgb

    def set_back_led(self, v):
        self.tail = v

    def close(self):
        pass


@pytest.fixture
def fleet(monkeypatch):
    import fleet.real_handle as rh
    monkeypatch.setattr(rh, "SpheroRobot", FakeBall)
    return rh


def test_connecting_a_second_ball_keeps_the_first(app, fleet):
    """The whole point of tagging by colour is that several are lit at once.
    A bench that can only hold one link can never show you the case where two
    are confused for each other."""
    app.connect("SK-AAAA")
    app.connect("SK-BBBB")
    assert set(app.lab.robots) == {"SK-AAAA", "SK-BBBB"}
    assert len(set(app.lab.assigned.values())) == 2, "two balls, two colours"


def test_colours_are_spread_across_the_wheel_not_taken_in_order(app, fleet):
    """Identity is a hue match with a tolerance and an ambiguity gap, so two
    balls in neighbouring slots trade names when the light changes. Taking the
    table in order would do exactly that."""
    from vision.facing import hue_err

    for n in ("SK-A", "SK-B", "SK-C"):
        app.connect(n)
    hues = [app.lab.detector.colors[c]["hue"]
            for c in app.lab.assigned.values()]
    worst = min(hue_err(a, b) for i, a in enumerate(hues)
                for b in hues[i + 1:])
    in_order = [app.lab.detector.colors[c]["hue"]
                for c in list(app.lab.detector.colors)[:3]]
    naive = min(hue_err(a, b) for i, a in enumerate(in_order)
                for b in in_order[i + 1:])
    assert worst > naive, f"spread {worst} is no better than in-order {naive}"


def test_each_ball_is_lit_at_its_own_slot(app, fleet):
    """Through `config.led_for`, which is the fleet's one answer to "what
    should this slot be driven at" — so a slot carrying a calibrated
    `led_value` or an explicit `rgb` is honoured here exactly as it is
    everywhere else. Deriving it from the hue alone ignores both."""
    app.connect("SK-A")
    app.connect("SK-B")
    for ble, r in app.lab.robots.items():
        slot = app.lab.assigned[ble]
        assert r.rgb == app.drive_for(slot)
        if app.bright == 255:
            assert r.rgb == config.led_for(slot, app.lab.detector.colors)
    rgbs = [r.rgb for r in app.lab.robots.values()]
    assert len(set(rgbs)) == len(rgbs), "two balls lit the same colour tag nothing"


def test_a_calibrated_led_value_is_not_ignored(app, fleet):
    """`calib/colors.json` carries a per-slot LED level — red sits at 73 and
    green at 224 on this bench, a three-fold difference tuned so the slots
    read alike on camera. Driving every slot at full blows the dim ones out,
    which costs the ring the saturation the tag is read from."""
    colors = app.lab.detector.colors
    dim = next((c for c, spec in colors.items()
                if spec.get("led_value") and spec["led_value"] < 200), None)
    if dim is None:
        pytest.skip("no slot on this bench carries a reduced led_value")
    app.lab.assigned["SK-D"] = dim
    app.lab.robots["SK-D"] = FakeBall(ble_name="SK-D", color=dim)
    app.push_led("SK-D")
    assert max(app.lab.robots["SK-D"].rgb) <= colors[dim]["led_value"] + 1, \
        "driven brighter than the slot was calibrated for"


def test_the_camera_names_each_ball_by_the_colour_it_was_given(app, fleet):
    """The end-to-end claim: this app told each ball what to wear, so a
    cluster glowing that colour IS that ball."""
    for n in ("SK-A", "SK-B", "SK-C"):
        app.connect(n)
    sig = app.lab.detector.colors

    f = np.full((720, 1280, 3), 5, np.uint8)
    truth = {}
    for i, (ble, slot) in enumerate(app.lab.assigned.items()):
        cx, cy = 250 + i * 380, 240 + i * 110
        px = np.uint8([[[sig[slot]["hue"], 235, 255]]])
        bgr = tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(f, (cx - 34, cy), 7, (255, 90, 60), -1)      # the tail
        for t in (-14, 14):
            cv2.circle(f, (cx + t, cy), 9, bgr, -1)
        truth[ble] = (cx, cy)
    f = cv2.GaussianBlur(f, (7, 7), 0)

    named = {c["code"]: c["centre"] for c in app.lab.analyse_all(f)
             if c["code"]}
    assert set(named) == set(truth), f"named {set(named)}, expected {set(truth)}"
    for ble, (cx, cy) in truth.items():
        assert abs(named[ble][0] - cx) < 6 and abs(named[ble][1] - cy) < 6


def test_dropping_one_ball_leaves_the_others_alone(app, fleet):
    for n in ("SK-A", "SK-B", "SK-C"):
        app.connect(n)
    kept = dict(app.lab.assigned)
    app.release("SK-B")
    assert "SK-B" not in app.lab.robots and "SK-B" not in app.lab.assigned
    for n in ("SK-A", "SK-C"):
        assert app.lab.assigned[n] == kept[n], "a drop reshuffled the others"


def test_running_out_of_colours_refuses_out_loud(app, fleet):
    """Six slots, six balls. The seventh cannot be told apart from one of the
    others, and saying so beats lighting two the same."""
    for i in range(len(app.lab.detector.colors)):
        app.connect(f"SK-{i}")
    app.connect("SK-EXTRA")
    assert "SK-EXTRA" not in app.lab.robots
    assert any("colour slot" in t for t, _ in app.notes), list(app.notes)


def test_the_swatch_is_drawn_in_pygames_byte_order_not_opencvs():
    """`draw` is stored BGR because vision/ hands it to cv2. Blitting it
    straight into pygame paints red as blue — and a swatch that lies about the
    colour is worse than none on a bench whose identity story IS the colour."""
    from taillight import slot_rgb
    from vision import config

    red = slot_rgb(config.COLORS, "red")
    assert red[0] > red[2], f"red must be reddest in RGB, got {red}"
    blue = slot_rgb(config.COLORS, "blue")
    assert blue[2] > blue[0], f"blue must be bluest in RGB, got {blue}"
    assert slot_rgb(config.COLORS, "nosuch") is not None, "falls back"


# -- links that do not come up --------------------------------------------

def test_retry_rebuilds_the_link_and_keeps_the_colour(app, fleet):
    """The handle backs off to thirty seconds between its own attempts, so a
    ball that failed once looks dead for half a minute. Retry must start from
    a clean radio — and must not renumber the fleet's colours doing it."""
    app.connect("SK-A")
    app.connect("SK-B")
    before = dict(app.lab.assigned)
    first = app.lab.robots["SK-A"]

    app.retry("SK-A")
    assert app.lab.assigned == before, "retry reshuffled the colours"
    assert app.lab.robots["SK-A"] is not first, "the handle was not rebuilt"
    assert set(app.lab.robots) == {"SK-A", "SK-B"}


def test_a_retried_ball_is_relit_at_its_own_slot(app, fleet):
    app.connect("SK-A")
    slot = app.lab.assigned["SK-A"]
    app.retry("SK-A")
    hue = app.lab.detector.colors[slot]["hue"]
    assert app.lab.robots["SK-A"].rgb == config.led_rgb(hue, value=app.bright)


def test_the_radio_hitting_its_limit_is_named_not_left_to_look_like_bad_luck(
        app, fleet):
    """CBErrorDomain Code=11 is a hard ceiling on this machine, not a flaky
    ball. Retrying it forever is the wrong response and the message has to say
    so, because the handle only writes it to a log nobody is watching."""
    app.connect("SK-A")
    r = app.lab.robots["SK-A"]
    r.link_up = False
    r.max_connections_hit = True
    trouble = app.lab.link_trouble()
    assert trouble and "maximum" in trouble and "connection" in trouble

    app.tick()
    assert any("maximum" in t for t, _ in app.notes), list(app.notes)


def test_a_failing_link_reports_its_error_rather_than_going_quiet(app, fleet):
    app.connect("SK-A")
    r = app.lab.robots["SK-A"]
    r.link_up = False
    r.last_error = "no toy named SK-A answered"
    assert "SK-A" in (app.lab.link_trouble() or "")
    assert "answered" in (app.lab.link_trouble() or "")


def test_a_healthy_fleet_reports_no_trouble(app, fleet):
    app.connect("SK-A")
    app.connect("SK-B")
    assert app.lab.link_trouble() is None


def test_the_reason_is_said_once_not_every_frame(app, fleet):
    """This runs from the render loop. A persistent fault said every frame
    fills the log thirty times a second and pushes everything else off."""
    app.connect("SK-A")
    r = app.lab.robots["SK-A"]
    r.link_up = False
    r.last_error = "boom"
    for _ in range(12):
        app.tick()
    assert sum(1 for t, _ in app.notes if "boom" in t) == 1


# -- driving the LED by RGB -----------------------------------------------

def test_an_explicit_rgb_beats_the_hue_it_would_have_been_derived_from(app):
    """Deriving the drive from the hunted hue assumes the LED emits what the
    maths says and the sensor reads back what the LED emits. Neither holds:
    on this bench a slot driven from hue 172 is READ at 165."""
    slot = app.lab.color
    app.set_led_channel(0, 10)
    app.set_led_channel(1, 20)
    app.set_led_channel(2, 30)
    assert app.lab.detector.colors[slot]["rgb"] == [10, 20, 30]
    assert config.led_for(slot, app.lab.detector.colors) == (10, 20, 30)


def test_setting_one_channel_writes_a_whole_triple(app):
    """A signature carrying a partial `rgb` is a slot whose colour depends on
    which slider was touched, and `led_for` cannot tell a missing channel from
    a deliberate zero."""
    slot = app.lab.color
    app.set_led_channel(1, 77)
    got = app.lab.detector.colors[slot]["rgb"]
    assert len(got) == 3 and got[1] == 77
    assert all(isinstance(v, int) for v in got)


def test_moving_the_hunted_hue_does_not_undo_a_hand_set_drive(app, fleet):
    """The two numbers are allowed to differ, and on this camera they have to.
    Retuning what the TRACKER looks for must not silently relight the ball."""
    app.connect("SK-A")
    slot = app.lab.assigned["SK-A"]
    app.lab.color = slot
    app.set_led_channel(0, 200)
    lit = app.lab.robots["SK-A"].rgb
    app.set_hue(140)
    assert app.lab.detector.colors[slot]["hue"] == 140, "the hunt moved"
    assert app.lab.robots["SK-A"].rgb == lit, "but the drive did not"


def test_the_hue_slider_still_relights_a_slot_with_no_drive_of_its_own(app, fleet):
    """One table, still. A slot nobody has hand-tuned keeps the old behaviour,
    so the drive and the hunt cannot silently diverge by neglect."""
    app.connect("SK-A")
    slot = app.lab.assigned["SK-A"]
    app.lab.color = slot
    app.lab.detector.colors[slot].pop("rgb", None)
    app.set_hue(140)
    assert app.lab.robots["SK-A"].rgb == app.drive_for(slot)


def test_hunt_seen_moves_the_hunt_to_what_the_camera_reads(app):
    """The gap between driven and read is what breaks the tag: 7 units is more
    than the tolerance, so the ball stops being named and the heading flips."""
    f = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    app.grab.frame = f
    app.reading = app.lab.analyse_lights(f)
    light = (app.reading.get("front")
             or (app.reading.get("group") or [None])[0])
    assert light and light.get("hue") is not None, "fixture must have a hue"

    app.hunt_seen()
    assert app.lab.detector.colors[app.lab.color]["hue"] == \
        int(round(light["hue"]))


def test_hunt_seen_refuses_when_there_is_no_colour_left_to_read(app):
    app.grab.frame = np.zeros((300, 400, 3), np.uint8)
    app.reading = app.lab.analyse_lights(app.grab.frame)
    app.hunt_seen()
    assert any("no readable hue" in t for t, _ in app.notes), list(app.notes)


def test_the_rgb_survives_a_save_and_reload(tmp_path, monkeypatch):
    """`rgb` has to be a signature key, or it is dropped on the way to disk
    and the hand-tuning silently reverts on the next run."""
    from vision import config as vconfig

    assert "rgb" in vconfig.SIGNATURE_KEYS
    monkeypatch.setattr(vconfig, "CALIB", tmp_path)
    vconfig.save("colors", {"red": {"hue": 172, "tol": 5, "rgb": [255, 40, 0]}})
    back = vconfig.load("colors")
    assert back["red"]["rgb"] == [255, 40, 0]
    assert vconfig.led_for("red", back) == (255, 40, 0)


def test_the_led_sliders_stay_on_screen_however_many_balls_are_held(app, fleet):
    """The fleet list grows with the fleet and with whatever the last scan
    turned up. Anything laid out UNDER it is one extra robot away from being
    off the window, which is how `taillight` became unreachable."""
    import taillight as T

    for n in range(0, 7):
        if n:
            app.connect(f"SK-{n}")
        app.found = [f"FOUND-{i}" for i in range(8)]
        app._build()
        for s in app.sliders:
            assert s.rect.bottom <= T.H, \
                f"{s.label} is off the window with {n} balls held"
        for b in app.buttons:
            assert b.rect.bottom <= T.H, \
                f"button {b.label} is off the window with {n} balls held"


def test_the_readout_keeps_its_room_whatever_the_fleet_does(app, fleet):
    """The list scrolls inside a fixed viewport, so nothing below it moves."""
    import taillight as T

    before = None
    for n in range(1, 7):
        app.connect(f"SK-{n}")
        app.found = [f"FOUND-{i}" for i in range(8)]
        app._build()
        if before is None:
            before = app.readout_top
        assert app.readout_top == before, "the readout was pushed by the list"
    assert app.list_total > app.list_room, "and the overflow is scrollable"


def test_a_trimmed_fleet_list_says_so(app, fleet):
    """A ball missing from the list must not look like a ball that is gone."""
    for n in range(1, 7):
        app.connect(f"SK-{n}")
    app.found = [f"FOUND-{i}" for i in range(8)]
    app._build()
    app.draw()
    assert len(app.lab.robots) == 6, "all six are still held"
    assert len(app.swatches) <= app.list_room, "only a windowful is drawn"
    assert app.list_total == 6 + 8, "but the whole list is counted"


# -- picking the arena in the app you are already in -----------------------

def clicks_for(app, pts):
    """Frame pixels -> screen positions, through the last blit."""
    (ox, oy), k = app._shot
    return [(ox + x * k, oy + y * k) for x, y in pts]


def test_four_corners_become_an_arena(app):
    """A homography calibrated at another resolution warps the wrong part of
    the frame and reports centimetres that are fiction — and nothing in the
    picture says so, because the dots still land on the ball."""
    from workspace.space import Workspace

    app.grab.frame = np.full((480, 640, 3), 20, np.uint8)
    app.draw()
    app.start_picking()
    assert app.picking == []

    box = [(60, 40), (580, 40), (580, 440), (60, 440)]
    for pos in clicks_for(app, box):
        app.add_corner(pos)
    assert app.picking is None, "four corners should have built it"

    ws = Workspace.load()
    h = app.lab.hom
    assert h.ready
    got = np.asarray(h.to_cm([list(p) for p in box]))
    assert abs(got[0][0]) < 0.5 and abs(got[0][1]) < 0.5, "origin first"
    assert abs(got[1][0] - ws.width) < 0.5, "then +x"
    assert abs(got[3][1] - ws.height) < 0.5, "and +y last"


def test_picking_is_not_saved_until_asked(app, monkeypatch):
    """`calib/homography.json` is live state the tracker and planner read, so
    it is overwritten when somebody asks and not a moment before."""
    saved = []
    app.grab.frame = np.full((480, 640, 3), 20, np.uint8)
    app.draw()
    app.start_picking()
    for pos in clicks_for(app, [(60, 40), (580, 40), (580, 440), (60, 440)]):
        app.add_corner(pos)
    monkeypatch.setattr(type(app.lab.hom), "save",
                        lambda self: saved.append(True))
    assert not saved, "building must not write"
    assert any("NOT saved" in t for t, _ in app.notes)
    app.save_homography()
    assert saved == [True]


def test_a_corner_can_be_taken_back(app):
    app.grab.frame = np.full((480, 640, 3), 20, np.uint8)
    app.draw()
    app.start_picking()
    for pos in clicks_for(app, [(60, 40), (580, 40)]):
        app.add_corner(pos)
    assert len(app.picking) == 2
    app.picking.pop()
    assert len(app.picking) == 1, "and it did not build early"


def test_a_click_outside_the_frame_is_not_a_corner(app):
    app.grab.frame = np.full((480, 640, 3), 20, np.uint8)
    app.draw()
    app.start_picking()
    app.add_corner((0, 0))            # the window's corner, not the frame's
    assert app.picking == [], "off-frame clicks must not become corners"


def test_picking_leaves_the_warped_view(app):
    """The brightness view has been through the homography, so a click on it
    is a click on the OLD arena — the one being replaced."""
    app.view = "bright"
    app.start_picking()
    assert app.view == "camera"


def test_saving_with_no_arena_says_so(app):
    app.lab.hom = None
    app.save_homography()
    assert any("no arena to save" in t for t, _ in app.notes)


def test_every_view_paints_while_picking(app):
    app.grab.frame = cv2.GaussianBlur(three_lights(), (5, 5), 0)
    app.reading = app.lab.analyse_lights(app.grab.frame)
    app.draw()
    app.start_picking()
    for pos in clicks_for(app, [(60, 40), (380, 40)]):
        app.add_corner(pos)
    paint(app)


# -- identity by hand, in the bench ---------------------------------------

def lit_frame(spots, size=(1280, 720)):
    """A tail and a fused tag pair per ball, all in white."""
    f = np.full((size[1], size[0], 3), 5, np.uint8)
    for cx, cy, deg in spots:
        r = math.radians(deg)
        ux, uy = math.cos(r), math.sin(r)
        cv2.circle(f, (int(cx - ux * 30), int(cy - uy * 30)), 6,
                   (230, 230, 230), -1)
        cv2.circle(f, (int(cx + ux * 8), int(cy + uy * 8)), 10,
                   (255, 255, 255), -1)
    return cv2.GaussianBlur(f, (7, 7), 0)


def seat(app, spots):
    """Put a frame up and let the app read it.

    The grabber is stopped first: it runs on its own thread and would replace
    this frame with a synthetic one between the assignment and the click.
    """
    app.grab.stop()
    app.grab.join(timeout=1.0)
    app.grab.frame = lit_frame(spots)
    app.show_all = True
    app.tick()
    app.draw()                       # establishes _shot for click conversion
    return app.all_reading


def click_at(app, fx, fy):
    (ox, oy), k = app._shot
    return (ox + fx * k, oy + fy * k)


def test_clicking_the_front_light_names_a_ball_and_aims_it(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    assert app.assigning == ["SK-A"]

    app.assign_click(click_at(app, 408, 300))     # the tag end
    got = app.lab.tracks.by_name["SK-A"]
    assert abs(wrap180(got.heading - 0.0)) < 8.0
    assert app.assigning is None, "queue finished"


def test_clicking_the_other_end_aims_it_the_other_way(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 370, 300))     # the TAIL end
    got = app.lab.tracks.by_name["SK-A"]
    assert abs(wrap180(got.heading - 180.0)) < 8.0


def test_identity_is_carried_while_the_balls_drive(app, fleet):
    app.connect("SK-A")
    app.connect("SK-B")
    seat(app, [(300, 200, 0.0), (800, 500, 90.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 308, 200))
    app.assign_click(click_at(app, 800, 508))

    for i in range(1, 20):
        app.grab.frame = lit_frame([(300 + i * 6, 200, 0.0),
                                    (800, 500 + i * 6, 90.0)])
        app.tick()
    a, b = app.lab.tracks.by_name["SK-A"], app.lab.tracks.by_name["SK-B"]
    assert a.live and b.live
    assert a.centre[0] > 380, "A followed its ball"
    assert b.centre[1] > 580, "B followed its ball"
    assert abs(wrap180(a.heading)) < 12.0
    assert abs(wrap180(b.heading - 90.0)) < 12.0


def test_a_ball_leaving_the_frame_is_lost_not_drifted(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    was = app.lab.tracks.by_name["SK-A"].centre

    app.grab.frame = lit_frame([])            # the lights go out
    app.tick()
    got = app.lab.tracks.by_name["SK-A"]
    assert got.lost and got.centre == was, "it must not have moved"
    assert got.why


def test_clicking_empty_floor_assigns_nothing(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 60, 660))
    assert "SK-A" not in app.lab.tracks.by_name
    assert app.assigning == ["SK-A"], "still waiting for the same ball"


def test_white_drives_all_three_dies(app, fleet):
    """A primary spends two thirds of the LED. Once identity is not coming
    from the colour, there is no reason to pay that."""
    app.connect("SK-A")
    app.connect("SK-B")
    app.all_white()
    for r in app.lab.robots.values():
        assert r.rgb == (app.bright, app.bright, app.bright)
    assert app.lab.tag_mode == "manual"


def test_forgetting_goes_back_to_colour(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    assert app.lab.tracks.names
    app.forget_tracks()
    assert app.lab.tracks.names == []
    assert app.lab.tag_mode == "colour"


def test_assigning_paints_in_every_view(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    paint(app)


def test_tracked_balls_paint_in_every_view(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    paint(app)
    app.grab.frame = lit_frame([])       # and while LOST
    app.tick()
    paint(app)


# -- the fleet list ------------------------------------------------------

def test_the_fleet_list_scrolls_instead_of_reflowing_the_panel(app, fleet):
    """A layout that reflows when a scan finishes is a layout where the thing
    you were about to click has moved."""
    import taillight as T

    app.found = [f"SK-{i}" for i in range(9)]
    for i in range(3):
        app.connect(f"SK-{i}")
    app._build()
    was = (app.readout_top, [s.rect.y for s in app.sliders])

    app.scroll_list(3)
    assert app.list_scroll == 3
    assert app.readout_top == was[0], "the readout moved"
    assert [s.rect.y for s in app.sliders] == was[1], "the sliders moved"


def test_scrolling_is_clamped_at_both_ends(app, fleet):
    app.found = [f"SK-{i}" for i in range(9)]
    for i in range(3):
        app.connect(f"SK-{i}")
    app._build()
    app.scroll_list(500)
    assert app.list_scroll == max(0, app.list_total - app.list_room)
    app.scroll_list(-500)
    assert app.list_scroll == 0


def test_scrolling_shows_the_rows_further_down(app, fleet):
    app.found = [f"SK-{i}" for i in range(9)]
    for i in range(3):
        app.connect(f"SK-{i}")
    app._build()
    top = [b.label for b in app.buttons if b.label.startswith("SK-")]
    app.scroll_list(3)
    down = [b.label for b in app.buttons if b.label.startswith("SK-")]
    assert top and down and top != down
    assert not set(top) & set(down), "it scrolled rather than shuffling"


def test_manual_identity_hides_the_colour_controls(app, fleet):
    """Nothing downstream reads a colour once the names came from a person and
    the balls are white. A knob that moves nothing must not be on screen."""
    app.connect("SK-A")
    app._build()
    colour_knobs = {"hue", "tol", "led r", "led g", "led b"}
    assert colour_knobs <= {s.label for s in app.sliders}

    app.all_white()
    assert not (colour_knobs & {s.label for s in app.sliders})
    assert "hunt seen" not in {b.label for b in app.buttons}
    assert app.list_room > 3, "and the room it freed went to the fleet list"


def test_nothing_lands_off_window_in_either_tag_mode(app, fleet):
    import taillight as T

    app.found = [f"SK-{i}" for i in range(9)]
    for i in range(6):
        app.connect(f"SK-{i}")
    for mode in ("colour", "manual"):
        app.lab.tag_mode = mode
        app._build()
        app.draw()
        for w in list(app.sliders) + list(app.buttons):
            assert w.rect.bottom <= T.H, f"{w.label} off window in {mode}"
        assert app.list_rect.bottom <= app.readout_top


def test_balls_wear_their_colours_while_somebody_is_choosing(app, fleet):
    """White is right for MEASURING and useless for CHOOSING: a floor of
    identical white blobs is exactly what a person cannot pick from."""
    app.connect("SK-A")
    app.connect("SK-B")
    app.all_white()
    assert len({r.rgb for r in app.lab.robots.values()}) == 1, "all alike"

    seat(app, [(300, 200, 0.0), (800, 500, 90.0)])
    app.start_assigning()
    rgbs = {ble: r.rgb for ble, r in app.lab.robots.items()}
    assert len(set(rgbs.values())) == 2, "they must differ while choosing"


def test_the_ball_being_asked_for_is_the_bright_one(app, fleet):
    """Colour alone is not enough at a few pixels across. Somebody who cannot
    read the hue can still click the brightest thing on the floor."""
    app.connect("SK-A")
    app.connect("SK-B")
    seat(app, [(300, 200, 0.0), (800, 500, 90.0)])
    app.start_assigning()
    want = app.assigning[0]
    lit = {ble: max(r.rgb) for ble, r in app.lab.robots.items()}
    assert lit[want] == max(lit.values())
    assert lit[want] > 2 * max(v for b, v in lit.items() if b != want)


def test_the_highlight_moves_to_the_next_ball(app, fleet):
    app.connect("SK-A")
    app.connect("SK-B")
    seat(app, [(300, 200, 0.0), (800, 500, 90.0)])
    app.start_assigning()
    first = app.assigning[0]
    app.assign_click(click_at(app, 308, 200))
    second = app.assigning[0]
    assert second != first
    lit = {ble: max(r.rgb) for ble, r in app.lab.robots.items()}
    assert lit[second] > lit[first], "the highlight did not move on"


def test_finishing_puts_them_back_to_white(app, fleet):
    app.connect("SK-A")
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    assert app.assigning is None
    assert app.lab.tag_mode == "manual"
    assert app.lab.robots["SK-A"].rgb == (app.bright,) * 3


def test_giving_up_restores_whatever_the_lighting_was(app, fleet):
    app.connect("SK-A")
    app.all_white()
    seat(app, [(400, 300, 0.0)])
    app.start_assigning()
    app.stop_assigning()
    assert app.lab.tag_mode == "manual"
    assert app.lab.robots["SK-A"].rgb == (app.bright,) * 3


# -- driving by hand -------------------------------------------------------

class Keys:
    """Stand in for `pygame.key.get_pressed()`."""

    def __init__(self, *down):
        self.down = set(down)

    def __getitem__(self, k):
        return k in self.down


def press(app, monkeypatch, *keys):
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: Keys(*keys))


def test_left_and_right_swing_the_aim_gradually(app, fleet, monkeypatch):
    """A reader is easy to fool with a ball that only sits at four angles.
    What has to be watched is whether it FOLLOWS."""
    app.connect("SK-A")
    app.aim = 0.0
    press(app, monkeypatch, pygame.K_RIGHT)
    seen = []
    for _ in range(30):
        app.drive_tick(1 / 30.0)
        seen.append(app.aim)
    assert seen[-1] > seen[0], "it did not turn"
    steps = [b - a for a, b in zip(seen, seen[1:])]
    assert max(steps) < 6.0, "the yaw jumped rather than swung"
    assert abs(seen[-1] - app.YAW_DEG_PER_S) < 6.0, "about a second's worth"

    press(app, monkeypatch, pygame.K_LEFT)
    back = app.aim
    for _ in range(10):
        app.drive_tick(1 / 30.0)
    assert app.aim < back, "left must swing the other way"


def test_yawing_sends_a_zero_speed_so_it_turns_on_the_spot(app, fleet,
                                                           monkeypatch):
    app.connect("SK-A")
    sent = []
    app.lab.robots["SK-A"].drive_raw = lambda h, b: sent.append((h, b))
    press(app, monkeypatch, pygame.K_RIGHT)
    for _ in range(30):
        app.drive_tick(1 / 30.0)
    assert sent, "nothing reached the ball"
    assert all(b == 0 for _, b in sent), "a yaw must not drive"


def test_up_drives_along_the_aim_and_down_drives_the_other_way(app, fleet,
                                                               monkeypatch):
    app.connect("SK-A")
    app.aim = 40.0
    sent = []
    app.lab.robots["SK-A"].drive_raw = lambda h, b: sent.append((h, b))

    press(app, monkeypatch, pygame.K_UP)
    app.drive_tick(1 / 30.0)
    assert sent[-1] == (40.0, app.drive_byte)

    app._drive_at = 0.0
    press(app, monkeypatch, pygame.K_DOWN)
    app.drive_tick(1 / 30.0)
    assert abs(wrap180(sent[-1][0] - 220.0)) < 1e-6, "down is the other way"


def test_the_speed_slider_is_what_gets_sent(app, fleet, monkeypatch):
    app.connect("SK-A")
    sent = []
    app.lab.robots["SK-A"].drive_raw = lambda h, b: sent.append((h, b))
    app.drive_byte = 37
    press(app, monkeypatch, pygame.K_UP)
    app.drive_tick(1 / 30.0)
    assert sent[-1][1] == 37
    assert "drive" in {s.label for s in app.sliders}


def test_letting_go_stops_the_ball(app, fleet, monkeypatch):
    app.connect("SK-A")
    stopped = []
    app.lab.robots["SK-A"].stop = lambda: stopped.append(True)
    press(app, monkeypatch, pygame.K_UP)
    app.drive_tick(1 / 30.0)
    press(app, monkeypatch)                 # nothing held
    app.drive_tick(1 / 30.0)
    assert stopped, "a released key must stop the ball"
    assert app.held is None


def test_the_radio_is_not_flooded_while_a_key_is_held(app, fleet, monkeypatch):
    """Thirty writes a second is a queue, not a faster loop."""
    app.connect("SK-A")
    sent = []
    app.lab.robots["SK-A"].drive_raw = lambda h, b: sent.append((h, b))
    press(app, monkeypatch, pygame.K_UP)
    for _ in range(60):                     # two seconds of holding
        app.drive_tick(1 / 30.0)
    assert len(sent) <= 25, f"{len(sent)} writes for two seconds of holding"


def test_driving_with_no_ball_connected_does_not_raise(app, monkeypatch):
    press(app, monkeypatch, pygame.K_UP)
    app.drive_tick(1 / 30.0)
    assert app.held is None


# -- the aim offset -------------------------------------------------------

def banked(app, name, pairs):
    """Put completed drives in as (told, went) and let the app judge them."""
    app.aim_samples[name] = [(t, w, wrap180(t - w)) for t, w in pairs]
    app.driving = name


def test_the_offset_is_what_the_drives_agree_on(app, fleet):
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (90, 120), (200, 230)])
    got = app.aim_summary("SK-A")
    assert got["ok"], got["why"]
    assert abs(wrap180(got["mean"] + 30.0)) < 0.5
    assert got["scatter"] < 1.0


def test_too_few_drives_is_not_saved(app, fleet):
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (90, 120)])
    got = app.aim_summary("SK-A")
    assert not got["ok"] and "drive more" in got["why"]


def test_drives_all_one_way_cannot_tell_an_offset_from_a_scale_error(app,
                                                                     fleet):
    """A true aim offset is the same in every direction; a scale or
    perspective error is not. Three drives the same way cannot tell which."""
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (5, 35), (10, 40)])
    got = app.aim_summary("SK-A")
    assert not got["ok"] and "same way" in got["why"]


def test_drives_that_disagree_are_not_an_offset_at_all(app, fleet):
    """An offset that varies with direction is a scale error, a bad homography,
    or slip — wearing an aim offset's clothes."""
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (90, 150), (200, 205)])
    got = app.aim_summary("SK-A")
    assert not got["ok"] and "disagree" in got["why"]


def test_the_offset_is_averaged_the_short_way_round(app, fleet):
    """-179 and +179 are two degrees apart, not three hundred and fifty-eight."""
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 179), (90, 271), (200, 21)])
    got = app.aim_summary("SK-A")
    assert got["ok"], got["why"]
    assert abs(abs(wrap180(got["mean"])) - 180.0) < 2.0


class FakeEntry:
    def __init__(self, code, ble, off):
        self.code, self.ble_name, self.heading_offset = code, ble, off


class FakeRoster:
    saved = []

    def __init__(self, entries):
        self.entries = entries

    def save(self):
        FakeRoster.saved.append({e.code: e.heading_offset for e in self.entries})
        return []


@pytest.fixture
def roster(monkeypatch):
    import fleet.roster as fr
    entries = [FakeEntry("CRXS", "SK-A", 238.6), FakeEntry("OTHR", "SK-B", 12.0)]
    FakeRoster.saved = []
    monkeypatch.setattr(fr.Roster, "load",
                        classmethod(lambda cls, *a, **k: FakeRoster(entries)))
    return entries


def test_saving_writes_the_agreed_offset_to_that_balls_row(app, fleet, roster):
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (90, 120), (200, 230)])
    app.save_aim()
    assert FakeRoster.saved, "nothing was written"
    assert abs(wrap180(roster[0].heading_offset + 30.0)) < 0.5
    assert roster[1].heading_offset == 12.0, "another ball's row was touched"


def test_a_saved_offset_makes_set_velocity_go_where_it_was_asked(app, fleet,
                                                                 roster):
    """The sign, checked end to end rather than argued. A ball whose forward
    sits 30deg clockwise of +y reads an offset of -30; with that stored,
    asking for 90 must send 60 to the radio."""
    from fleet.handle import velocity_to_command

    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (90, 120), (200, 230)])
    app.save_aim()
    off = roster[0].heading_offset

    want = 90.0
    heading, _ = velocity_to_command(
        np.array([math.sin(math.radians(want)), math.cos(math.radians(want))]))
    sent = (heading + off) % 360.0
    travelled = (sent + 30.0) % 360.0       # this ball adds 30 to whatever it gets
    assert abs(wrap180(travelled - want)) < 0.5, \
        f"asked {want}, sent {sent:.1f}, it went {travelled:.1f}"


def test_disagreeing_drives_are_refused_at_save_time_too(app, fleet, roster):
    app.connect("SK-A")
    banked(app, "SK-A", [(0, 30), (90, 150), (200, 205)])
    app.save_aim()
    assert not FakeRoster.saved
    assert roster[0].heading_offset == 238.6, "a bad offset got written"


def test_a_ball_missing_from_the_roster_says_so(app, fleet, roster):
    app.connect("SK-Z")
    banked(app, "SK-Z", [(0, 30), (90, 120), (200, 230)])
    app.save_aim()
    assert not FakeRoster.saved
    assert any("not in roster" in t for t, _ in app.notes)


def test_yawing_in_place_banks_no_sample(app, fleet, monkeypatch):
    """A turn in place has no course, so it cannot measure an offset."""
    app.connect("SK-A")
    press(app, monkeypatch, pygame.K_RIGHT)
    for _ in range(20):
        app.drive_tick(1 / 30.0)
    press(app, monkeypatch)
    app.drive_tick(1 / 30.0)
    assert not app.aim_samples.get("SK-A")


def test_p_saves_the_raw_frame_not_the_window(app, tmp_path, monkeypatch):
    """`learn/` calibrates its synthetic footage against these, so they must be
    the camera's own pixels at full size, with the shutter they were read at."""
    import json
    import time as _time
    import cv2
    monkeypatch.chdir(tmp_path)
    deadline = _time.time() + 3.0
    while app.grab.frame is None and _time.time() < deadline:
        _time.sleep(0.02)
    app.save_raw()
    pngs = list((tmp_path / "runs" / "raw").glob("raw_*.png"))
    assert len(pngs) == 1
    img = cv2.imread(str(pngs[0]))
    assert img.shape == app.grab.frame.shape
    meta = json.loads(pngs[0].with_suffix(".json").read_text())
    assert meta["shape"] == list(img.shape)
    assert "shutter_us" in meta and "clusters" in meta


# -- the bench supplying the ball's own yaw -------------------------------

def test_the_yaw_read_is_a_cache_lookup_not_a_radio_trip(app, fleet):
    """`get_orientation` reads a streaming cache spherov2 refills about every
    150ms, so calling it every frame is a dictionary lookup."""
    app.connect("SK-A")
    r = app.lab.robots["SK-A"]
    calls = []
    r._api = type("Api", (), {
        "get_orientation": lambda _s: (calls.append(1),
                                       {"pitch": 1.0, "roll": 2.0,
                                        "yaw": 137.0})[1]})()
    assert app.lab.chassis_yaw(r) == 137.0
    assert app.lab.yaws() == {"SK-A": 137.0}
    assert len(calls) == 2


def test_no_packet_yet_reads_as_no_yaw_rather_than_a_number(app, fleet):
    """It returns None until the first packet lands, and None is the right
    answer then — a zero would be a heading the ball never reported."""
    app.connect("SK-A")
    r = app.lab.robots["SK-A"]
    r._api = type("Api", (), {"get_orientation": lambda _s: None})()
    assert app.lab.chassis_yaw(r) is None
    assert app.lab.yaws() == {}


def test_a_toy_without_attitude_is_simply_absent(app, fleet):
    app.connect("SK-A")
    app.lab.robots["SK-A"]._api = type("Api", (), {})()
    assert app.lab.chassis_yaw(app.lab.robots["SK-A"]) is None
    assert app.lab.yaws() == {}


def test_a_raising_sensor_does_not_take_the_bench_down(app, fleet):
    app.connect("SK-A")
    r = app.lab.robots["SK-A"]

    def boom(_s):
        raise RuntimeError("radio gone")
    r._api = type("Api", (), {"get_orientation": boom})()
    assert app.lab.chassis_yaw(r) is None
    app.tick()


def test_a_bridged_track_is_drawn_as_an_estimate(app, fleet):
    """An estimate that looks like a measurement is the failure this project
    keeps hitting."""
    seat(app, [(400, 300, 0.0)])
    app.connect("SK-A")
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    t = app.lab.tracks.by_name["SK-A"]
    t.bridged, t.lost = True, True
    t.why = "camera lost it — heading carried by the ball's own yaw for 0.4s"
    app.show_all = True
    app._build()
    app.draw()
    assert app.screen.get_clip() == app.screen.get_rect()


def test_every_read_cluster_gets_its_own_arrow(app, fleet):
    """Drawing only the primary reading's arrow looked exactly like the other
    balls never being read — a drawing gap wearing a detection failure's
    clothes."""
    import taillight as T

    f = lit_frame([(300, 200, 0.0), (900, 500, 90.0)], size=(1280, 720))
    app.grab.stop()
    app.grab.join(timeout=1.0)
    app.grab.frame = f
    app.show_all = True
    app.tick()

    read = [c for c in app.all_reading if c["why"] is None]
    assert len(read) >= 2, f"the fixture must give two readings, got {len(read)}"

    app.view = "camera"
    app._build()
    app.draw()

    # An arrow is drawn in the trust colours; count the balls that have one by
    # looking for arrow pixels near each cluster centre.
    arr = pygame.surfarray.array3d(app.screen)
    (ox, oy), k = app._shot
    with_arrow = 0
    for c in read:
        cx, cy = c["centre"]
        px, py = int(ox + cx * k), int(oy + cy * k)
        patch = arr[max(px - 60, 0):px + 60, max(py - 60, 0):py + 60]
        tones = {T.MINT, T.CYAN, T.CORAL}
        if any((patch == t).all(-1).any() for t in tones):
            with_arrow += 1
    assert with_arrow == len(read), (
        f"{with_arrow} of {len(read)} clusters were drawn with an arrow")


def test_the_grouping_knob_and_the_measured_span_are_named_apart(app):
    """One is a knob, the other a measurement, and they sit a few rows apart.
    Calling both 'span' is a panel that cannot be read correctly."""
    labels = {s.label for s in app.sliders}
    assert "group within" in labels
    assert "ball span" not in labels, "the old ambiguous name is back"


# -- does north actually go north -----------------------------------------

def aimed(app, name, image_deg):
    """Point an assigned track a given way, in image degrees."""
    app.lab.tracks.by_name[name].heading = float(image_deg)
    app.lab.tracks.by_name[name].lost = False


def ready_north(app, fleet, offset=0.0):
    seat(app, [(400, 300, 0.0)])
    app.connect("SK-A")
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    app.saved_offset = lambda ble: (offset, "test")
    return app.lab.robots["SK-A"]


def test_the_north_test_turns_in_place_and_never_drives(app, fleet):
    """The point of a heading you can read standing still: the aim frame can
    be checked without the ball travelling a centimetre, so it cannot reach a
    wall while being wrong."""
    r = ready_north(app, fleet, offset=40.0)
    sent = []
    r.drive_raw = lambda h, b: sent.append((h, b))
    app.start_north()
    assert sent, "nothing was commanded"
    assert all(b == 0 for _, b in sent), "a speed above zero would make it travel"
    assert abs(wrap180(sent[-1][0] - 40.0)) < 1e-6, "north plus the offset"


def test_a_correct_aim_frame_reads_as_north(app, fleet):
    r = ready_north(app, fleet)
    r.drive_raw = lambda h, b: None
    app.start_north()
    app.arena_heading = lambda name: 2.0          # it settles facing north
    for _ in range(6):
        app.north_tick()
    assert app.north_until is None, "it should have finished"
    got, err, settled = app.north_result
    assert settled and abs(err) < 8
    assert any("NORTH IS NORTH" in t for t, _ in app.notes)


def test_a_wrong_aim_frame_reports_the_correction(app, fleet):
    """The error IS the number still missing from heading_offset."""
    r = ready_north(app, fleet)
    r.drive_raw = lambda h, b: None
    app.start_north()
    app.arena_heading = lambda name: 137.0
    for _ in range(6):
        app.north_tick()
    got, err, _ = app.north_result
    assert abs(wrap180(err - 137.0)) < 1e-6
    assert any("aim frame is" in t and "out" in t for t, _ in app.notes)


def test_it_waits_for_the_turn_to_settle(app, fleet):
    r = ready_north(app, fleet)
    r.drive_raw = lambda h, b: None
    app.start_north()
    swinging = iter([10.0, 40.0, 80.0, 110.0, 130.0, 137.0, 137.0, 137.0, 137.0])
    app.arena_heading = lambda name: next(swinging, 137.0)
    for _ in range(5):
        app.north_tick()
        assert app.north_until is not None, "it stopped while still turning"
    for _ in range(4):
        app.north_tick()
    assert app.north_until is None
    assert app.north_result[2], "it should have settled, not timed out"


def test_an_unassigned_ball_is_refused(app, fleet):
    """Without a track there is no heading being followed, so there is nothing
    to read the answer off."""
    app.connect("SK-A")
    app.start_north()
    assert app.north_until is None
    assert any("not assigned" in t for t, _ in app.notes)


def test_the_arrows_do_not_fight_the_test_for_the_radio(app, fleet, monkeypatch):
    r = ready_north(app, fleet)
    sent = []
    r.drive_raw = lambda h, b: sent.append((h, b))
    app.start_north()
    sent.clear()
    press(app, monkeypatch, pygame.K_UP)
    app.drive_tick(1 / 30.0)
    assert not sent, "an arrow key drove during the north test"


def test_no_two_buttons_share_a_place(app, fleet):
    """`north` and `forget` were laid out on the same rect: one was invisible
    and the other could not be clicked, and which ran depended on the order
    they were added. So the visible label and the thing that happened were
    different buttons."""
    app.found = [f"SK-{i}" for i in range(6)]
    for i in range(3):
        app.connect(f"SK-{i}")
        app.lab.tracks.assign(f"SK-{i}", {
            "centre": (10 * i, 10), "group": [
                {"x": 0, "y": 0, "area": 9.0, "peak": 255.0},
                {"x": 20, "y": 20, "area": 30.0, "peak": 255.0}]})
    for mode in ("colour", "manual"):
        for assigning in (None, ["SK-0"]):
            app.lab.tag_mode, app.assigning = mode, assigning
            app._build()
            seen = []
            for b in app.buttons:
                for other, rect in seen:
                    assert not b.rect.colliderect(rect), (
                        f"{b.label} overlaps {other} in {mode}"
                        f"{'/assigning' if assigning else ''}")
                seen.append((b.label, b.rect))


def test_the_button_you_click_is_the_one_that_runs(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (10, 10), "group": [
        {"x": 0, "y": 0, "area": 9.0, "peak": 255.0},
        {"x": 20, "y": 20, "area": 30.0, "peak": 255.0}]})
    app._build()
    for b in app.buttons:
        hit = next((o for o in app.buttons if o.rect.collidepoint(b.rect.center)), None)
        assert hit is b, f"clicking {b.label} actually runs {hit.label}"


# -- point to point --------------------------------------------------------

class DrivenBall(FakeBall):
    """A ball that actually moves, so a leg can be run to completion.

    Turns toward whatever bearing it is sent at a finite rate and travels along
    the way it is FACING — which is what a Sphero does, and what makes an aim
    frame matter. `bias` is the ball's own zero being off from the arena's.
    """

    def __init__(self, bias=0.0, handed=1.0, **kw):
        super().__init__(**kw)
        self.bias = float(bias)
        # Which way round a HEADING runs against the arena's compass. A
        # Sphero's heading turns clockwise from above; whether the arena's
        # compass does depends on the camera and the calibration, so both are
        # modelled.
        self.handed = float(handed)
        self.facing = 0.0            # arena degrees
        self.want = 0.0
        self.byte = 0

    def drive_raw(self, heading, byte):
        self.want = wrap180(self.handed * float(heading) - self.bias) % 360.0
        self.byte = int(byte)

    def stop(self):
        self.byte = 0

    def step(self, dt, px_per_cm):
        err = wrap180(self.want - self.facing)
        step = max(-360.0 * dt, min(360.0 * dt, err))
        self.facing = (self.facing + step) % 360.0
        cm_s = self.byte / 255.0 * 60.0
        d = cm_s * dt * px_per_cm
        r = math.radians(self.facing)
        return (math.sin(r) * d, math.cos(r) * d)   # compass: 0 is +y


class Clock:
    """Wall clock that advances with the simulation.

    The controller waits real seconds between corrections — deliberately, so
    it does not argue with the drive's own lag. A test that steps a simulated
    ball thousands of times in a few milliseconds would never let it correct
    at all, so time here moves with the ball rather than with the wall.
    """

    def __init__(self):
        self.t = 1000.0

    def tick(self, dt):
        self.t += dt

    def __call__(self):
        return self.t


def run_p2p(app, start, target, bias=0.0, offset=None, steps=900):
    """Drive one leg in closed loop and report where it stopped."""
    ball = DrivenBall(bias=bias, ble_name="SK-A", color="red")
    app.lab.robots["SK-A"] = ball
    app.lab.assigned["SK-A"] = "red"
    app.driving = "SK-A"
    app.saved_offset = lambda n: ((bias if offset is None else offset), "test")

    pos = list(start)
    app.lab.tracks.assign("SK-A", {
        "centre": tuple(pos),
        "group": [{"x": pos[0], "y": pos[1], "area": 9.0, "peak": 255.0},
                  {"x": pos[0] + 8, "y": pos[1], "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing

    px_cm = app.lab.px_per_cm(start) or 10.0
    clock = Clock()
    import taillight as T
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app.p2p_pick = True
        app._shot = ((0, 0), 1.0)
        app.p2p_click(target)
        for _ in range(steps):
            if app.p2p is None:
                break
            app.p2p_tick()
            dx, dy = ball.step(1 / 30.0, px_cm)
            pos[0] += dx
            pos[1] += dy
            app.lab.tracks.by_name["SK-A"].centre = tuple(pos)
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    return np.asarray(pos), ball


def test_a_leg_arrives(app, fleet):
    start, target = (400.0, 300.0), (700.0, 520.0)
    end, ball = run_p2p(app, start, target)
    gap_px = float(np.linalg.norm(end - np.asarray(target)))
    gap_cm = gap_px / (app.lab.px_per_cm(start) or 10.0)
    assert app.p2p is None, "it never finished"
    assert gap_cm <= app.P2P_ARRIVE_CM + 2.0, f"stopped {gap_cm:.1f}cm away"
    assert ball.byte == 0, "it was left driving"
    assert any("arrived" in t for t, _ in app.notes)


def test_it_arrives_despite_the_balls_zero_being_rotated(app, fleet):
    """The aim frame doing its job: the ball's own north is 137deg off, and the
    saved offset is what makes 'go north' go north."""
    start, target = (400.0, 300.0), (250.0, 520.0)
    end, _ = run_p2p(app, start, target, bias=137.0)
    gap = float(np.linalg.norm(end - np.asarray(target)))
    gap_cm = gap / (app.lab.px_per_cm(start) or 10.0)
    assert gap_cm <= app.P2P_ARRIVE_CM + 2.0, f"stopped {gap_cm:.1f}cm away"


def test_it_arrives_with_a_badly_wrong_stored_offset(app, fleet):
    """The point of closing the loop on what the camera SEES: which way the
    ball thinks it is being sent does not matter. A stored offset that is 90
    degrees out costs a correction step, not the leg."""
    start, target = (400.0, 300.0), (700.0, 520.0)
    end, _ = run_p2p(app, start, target, bias=137.0, offset=90.0)
    gap_cm = (float(np.linalg.norm(end - np.asarray(target)))
              / (app.lab.px_per_cm(start) or 10.0))
    assert app.p2p is None, "it never finished"
    assert gap_cm <= app.P2P_ARRIVE_CM + 2.0, f"stopped {gap_cm:.1f}cm away"


def test_it_arrives_with_no_stored_offset_at_all(app, fleet):
    """Nothing saved, nothing calibrated, ball's own zero 137 degrees off."""
    start, target = (400.0, 300.0), (250.0, 480.0)
    end, _ = run_p2p(app, start, target, bias=137.0, offset=0.0)
    gap_cm = (float(np.linalg.norm(end - np.asarray(target)))
              / (app.lab.px_per_cm(start) or 10.0))
    assert gap_cm <= app.P2P_ARRIVE_CM + 2.0, f"stopped {gap_cm:.1f}cm away"


def test_it_refuses_rather_than_driving_on_an_unfinished_turn(app, fleet):
    """The failure seen on hardware: the turn stalled, the aim timed out, and
    it drove anyway on a heading it never reached."""
    ball = DrivenBall(ble_name="SK-A", color="red")
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    clock = Clock()
    import taillight as T
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._shot = ((0, 0), 1.0)
        app.p2p_pick = True
        app.p2p_click((400.0, 700.0))
        # Frozen facing the OPPOSITE way to the target. Which image direction
        # that is depends on the homography, so it is derived — picking one
        # can leave the ball already aimed, and then driving is correct.
        bearing, _ = app.p2p_geometry()
        app.arena_heading = lambda n: (bearing + 180.0) % 360.0
        drove = False
        for _ in range(2000):
            if app.p2p is None:
                break
            app.p2p_tick()
            if ball.byte:
                drove = True
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    assert not drove, "it drove on a turn that never finished"
    assert any("NOT driving" in t for t, _ in app.notes)


def test_the_command_is_refreshed_so_the_ball_does_not_stall(app, fleet):
    """A Sphero's roll command expires after a couple of seconds. Sending only
    when the bearing changes sends once during a turn in place — the ball turns
    partway, the command lapses, and it sits there. That was the freeze."""
    ball = DrivenBall(ble_name="SK-A", color="red")
    sent = []
    raw = ball.drive_raw
    ball.drive_raw = lambda h, b: (sent.append((h, b)), raw(h, b))[1]
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    clock = Clock()
    import taillight as T
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app._shot = ((0, 0), 1.0)
        app.p2p_pick = True
        app.p2p_click((400.0, 700.0))
        for _ in range(60):                    # two seconds of aiming
            app.p2p_tick()
            ball.step(1 / 30.0, 10.0)
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    assert len(sent) >= 8, (
        f"only {len(sent)} commands in two seconds — the ball would stall")


def test_it_turns_before_it_drives(app, fleet):
    """Turn and go: the aim phase commands speed zero, so it swings on the
    spot rather than curving away from wherever it happened to be facing."""
    ball = DrivenBall(ble_name="SK-A", color="red")
    sent = []
    raw = ball.drive_raw
    ball.drive_raw = lambda h, b: (sent.append((h, b)), raw(h, b))[1]
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    app._shot = ((0, 0), 1.0)
    app.p2p_pick = True
    app.p2p_click((400.0, 100.0))
    # Point it the OPPOSITE way to the target, in arena terms. Which image
    # direction that is depends on the homography, so it is derived rather
    # than assumed — assuming it starts the ball already aimed, and the turn
    # this is testing never happens.
    bearing, _ = app.p2p_geometry()
    ball.facing = (bearing + 180.0) % 360.0

    for _ in range(40):
        app.p2p_tick()
        ball.step(1 / 30.0, 10.0)
        app._drive_at -= app.DRIVE_PUSH_S
        if app.p2p and app.p2p["phase"] == "go":
            break
    assert sent, "nothing was commanded"
    assert all(b == 0 for _, b in sent), "it drove before it had aimed"
    # Against the BEARING, not against north: where the target lies in arena
    # degrees is whatever the homography says, and the claim is that it turned
    # to face the target before moving.
    assert abs(wrap180(ball.facing - bearing)) <= app.P2P_AIM_TOL + 2


def test_six_cm_per_second_is_the_byte_the_speed_map_gives(app, fleet):
    from swarm.trace import SpeedMap

    app.connect("SK-A")
    byte, sm = app.p2p_byte()
    assert byte == SpeedMap(min_moving_byte=18).byte_for(6.0)
    assert byte >= 18, "below the deadband a Sphero sits rather than creeping"


def test_losing_sight_stops_rather_than_driving_blind(app, fleet):
    start, target = (400.0, 300.0), (700.0, 520.0)
    ball = DrivenBall(ble_name="SK-A", color="red")
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": start, "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: 0.0
    app._shot = ((0, 0), 1.0)
    app.p2p_pick = True
    app.p2p_click(target)
    app.ball_px = lambda n: None            # no position at all, not merely stale
    app.p2p_tick()
    assert app.p2p is None and ball.byte == 0
    assert any("no position at all" in t for t, _ in app.notes)


def test_a_dropped_frame_does_not_abandon_the_leg(app, fleet):
    """32 dropouts of one or two frames in 35 seconds is what the camera
    actually does. A leg that gives up on the first one can never finish."""
    ball = DrivenBall(ble_name="SK-A", color="red")
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    app._shot = ((0, 0), 1.0)
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))

    t = app.lab.tracks.by_name["SK-A"]
    for i in range(30):
        t.lost = (i % 7 == 0)               # a missed frame every so often
        app.p2p_tick()
    assert app.p2p is not None, "one dropped frame killed the leg"


def test_it_coasts_while_blind_rather_than_driving_on_a_stale_fix(app, fleet):
    ball = DrivenBall(ble_name="SK-A", color="red")
    sent = []
    raw = ball.drive_raw
    ball.drive_raw = lambda h, b: (sent.append((h, b)), raw(h, b))[1]
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    app._shot = ((0, 0), 1.0)
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))

    app.lab.tracks.by_name["SK-A"].lost = True
    sent.clear()
    for _ in range(10):
        app.p2p_tick()
    assert not sent, "it kept commanding on a position nobody was measuring"


def test_a_long_blackout_does_give_up(app, fleet):
    ball = DrivenBall(ble_name="SK-A", color="red")
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    clock = Clock()
    import taillight as T
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._shot = ((0, 0), 1.0)
        app.p2p_pick = True
        app.p2p_click((700.0, 520.0))
        app.lab.tracks.by_name["SK-A"].lost = True
        for _ in range(int(30 * (app.P2P_BLIND_S + 1.0))):
            if app.p2p is None:
                break
            app.p2p_tick()
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    assert app.p2p is None
    assert any("could not see it" in t for t, _ in app.notes)


def test_pressing_it_again_cancels(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app.start_p2p()
    assert app.p2p_pick
    app.start_p2p()
    assert not app.p2p_pick and app.p2p is None


def test_the_target_is_drawn_where_it_was_clicked(app, fleet):
    """A target you cannot see is a target you cannot check. The click goes
    through the same pixel mapping as the corner picking, and if that were
    wrong the ball would drive somewhere nobody asked for with nothing on
    screen to say so."""
    import taillight as T

    seat(app, [(400, 300, 0.0)])
    app.connect("SK-A")
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    app.saved_offset = lambda n: (0.0, "test")
    app.arena_heading = lambda n: 0.0

    app.start_p2p()
    assert app.p2p_pick
    app.draw()                       # the "click where it should go" prompt
    app.p2p_click(click_at(app, 900, 520))
    app.draw()

    (ox, oy), k = app._shot
    want = (int(ox + 900 * k), int(oy + 520 * k))
    arr = pygame.surfarray.array3d(app.screen)
    patch = arr[want[0] - 24:want[0] + 24, want[1] - 24:want[1] + 24]
    assert any((patch == t).all(-1).any() for t in (T.SUN, T.MINT)), \
        "nothing was drawn at the clicked point"

    # ...and not somewhere else: the far corner must stay empty.
    far = arr[T.VIEW.x + 10:T.VIEW.x + 80, T.VIEW.y + 300:T.VIEW.y + 380]
    assert not any((far == t).all(-1).any() for t in (T.SUN, T.MINT))


def test_the_drawn_target_matches_the_one_it_will_drive_to(app, fleet):
    """The marker and the maths must come from the same number, or the picture
    reassures you about a leg that is not the one being driven."""
    seat(app, [(400, 300, 0.0)])
    app.connect("SK-A")
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    app.saved_offset = lambda n: (0.0, "test")

    app.start_p2p()
    app.p2p_click(click_at(app, 900, 520))
    assert abs(app.p2p["target"][0] - 900) < 1.5
    assert abs(app.p2p["target"][1] - 520) < 1.5


def test_the_prompt_shows_while_waiting_for_a_click(app, fleet):
    seat(app, [(400, 300, 0.0)])
    app.connect("SK-A")
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    app.start_p2p()
    app.view = "camera"
    app._build()
    app.draw()
    assert app.p2p_pick and app.screen.get_clip() == app.screen.get_rect()


# -- turning by spinning ------------------------------------------------------

class SpinBall(DrivenBall):
    """A ball that turns on raw motor power, the way the handle drives it.

    - `spin_raw(p)` sets a turn RATE; below `STATIC` the motors cannot beat
      friction and nothing turns. Which way a positive power turns it is
      `polarity`, which the controller is not told and has to learn.
    - `stop_raw()` re-zeroes where it points and stabilises, so the chassis
      coasts a little and is then HELD at the orientation it stopped at.
    - a roll queued while spinning stops the spin first — latest wins.
    - `drift` is creep in the ball's own heading reference while driving —
      the kind re-sending heading 0 cannot cure — to exercise re-aiming.

    It moves through the HOMOGRAPHY, not through an assumed image direction,
    so the test stays true whatever orientation the arena was calibrated in.
    """

    STATIC = 25.0
    GAIN = 3.0          # deg/s per unit of power above STATIC
    TAU = 0.08          # s, chassis spin inertia

    def __init__(self, polarity=1.0, drift=0.0, **kw):
        super().__init__(**kw)
        self.polarity = float(polarity)
        self.drift = float(drift)
        self.raw = 0
        self.rate = 0.0
        self.held = 0.0             # the heading it is holding, its own frame
        self.log = []

    @property
    def spinning(self):
        return self.raw != 0

    def spin_raw(self, power):
        self.log.append(("spin", int(power)))
        self.raw = int(power)
        self.raw_active = True          # the real handle: even power 0 is a spin
        self.byte = 0

    def stop_raw(self):
        self.log.append(("stop_raw",))
        if not self.raw and not getattr(self, "raw_active", False):
            return
        self.raw_active = False
        self.raw = 0
        self.held = 0.0
        self.bias = -self.facing            # heading 0 is now where it points
        self.want = self.facing             # and that is what it holds

    def stop(self):
        self.log.append(("stop",))
        if self.raw:
            self.stop_raw()
        self.byte = 0

    def drive_raw(self, heading, byte):
        self.log.append(("roll", float(heading), int(byte)))
        if self.raw:
            self.stop_raw()
        self.held = float(heading)
        super().drive_raw(heading, byte)

    def advance(self, dt, hom, pos_px):
        """One step. Returns the new pixel position."""
        target = 0.0
        if self.raw:
            mag = max(0.0, abs(self.raw) - self.STATIC) * self.GAIN
            target = self.polarity * (1 if self.raw > 0 else -1) * mag
        self.rate += (target - self.rate) * min(1.0, dt / self.TAU)
        if self.raw or abs(self.rate) > 0.5:
            self.facing = (self.facing + self.rate * dt) % 360.0
            return pos_px                   # spinning on the spot
        self.rate = 0.0
        if self.byte:
            # Drift lives in the ball's OWN heading reference. Re-sending
            # heading 0 does not cure it — the ball's idea of 0 has moved — so
            # it is applied to the frame, and the command re-reads it.
            self.bias = (self.bias - self.drift * dt) % 360.0
            self.want = wrap180(self.handed * self.held - self.bias) % 360.0
        err = wrap180(self.want - self.facing)
        self.facing = (self.facing + max(-360 * dt, min(360 * dt, err))) % 360.0
        if not self.byte:
            return pos_px
        cm_s = self.byte / 255.0 * 60.0
        here = np.asarray(hom.to_cm([list(pos_px)]), float).ravel()[:2]
        r = math.radians(self.facing)
        step = np.array([math.sin(r), math.cos(r)]) * cm_s * dt
        there = np.asarray(hom.to_px([list(here + step)]), float).ravel()[:2]
        return tuple(there)


def run_spin(app, start, target, facing=0.0, polarity=1.0, drift=0.0,
             power=50.0, band=8.0, steps=2400, name="SK-A"):
    """A whole rate-mode leg, closed loop, on a simulated clock."""
    import taillight as T

    ball = SpinBall(polarity=polarity, drift=drift, ble_name=name, color="red")
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.driving = name
    app.turn_mode = "rate"
    app.spin_power, app.aim_band = float(power), float(band)
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": tuple(start), "group": [
        {"x": start[0], "y": start[1], "area": 9.0, "peak": 255.0},
        {"x": start[0] + 8, "y": start[1], "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing

    pos = tuple(start)
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app._shot = ((0, 0), 1.0)
        app.p2p_pick = True
        app.p2p_click(target)
        for _ in range(steps):
            if app.p2p is None:
                break
            app.p2p_tick()
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    gap_cm = float(np.linalg.norm(
        np.asarray(app.lab.hom.to_cm([list(pos)]), float).ravel()[:2]
        - np.asarray(app.lab.hom.to_cm([list(target)]), float).ravel()[:2]))
    return gap_cm, ball


@pytest.mark.parametrize("facing", [0.0, 90.0, 200.0, 315.0])
@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_a_spun_leg_arrives_from_any_facing_either_motor_way_round(
        app, fleet, facing, polarity):
    gap, _ = run_spin(app, (400.0, 300.0), (700.0, 520.0),
                      facing=facing, polarity=polarity)
    assert app.p2p is None, "it never finished"
    assert gap <= app.P2P_ARRIVE_CM + 2.0, f"stopped {gap:.1f}cm away"
    assert any("arrived" in t for t, _ in app.notes), list(app.notes)[-3:]


def test_the_turn_commands_no_heading_at_all(app, fleet):
    """What was asked for: turn by spinning, not by telling it an angle.
    Every command before driving starts must be a spin or a stop."""
    gap, ball = run_spin(app, (400.0, 300.0), (700.0, 520.0), facing=250.0)
    first_drive = next(i for i, e in enumerate(ball.log)
                       if e[0] == "roll" and e[2] > 0)
    during_turn = ball.log[:first_drive]
    assert any(e[0] == "spin" for e in during_turn), "it never spun"
    assert not [e for e in during_turn if e[0] == "roll"], \
        "a heading was commanded during the turn"


def test_it_stops_spinning_before_it_drives(app, fleet):
    gap, ball = run_spin(app, (400.0, 300.0), (700.0, 520.0), facing=250.0)
    first_drive = next(i for i, e in enumerate(ball.log)
                       if e[0] == "roll" and e[2] > 0)
    last_spin = max(i for i, e in enumerate(ball.log[:first_drive])
                    if e[0] == "spin")
    assert any(e[0] == "stop_raw" for e in ball.log[last_spin:first_drive]), \
        "it went from spinning straight to driving"


def test_it_drives_straight_ahead_of_where_the_spin_stopped(app, fleet):
    """Stopping re-zeroes where it points, so straight on is heading 0 — no
    stored offset, nothing to get the wrong way round."""
    gap, ball = run_spin(app, (400.0, 300.0), (700.0, 520.0), facing=250.0)
    drives = [e for e in ball.log if e[0] == "roll" and e[2] > 0]
    assert drives and all(e[1] == 0.0 for e in drives), drives[:3]


def test_a_reversed_motor_is_learned_once_and_remembered(app, fleet):
    run_spin(app, (400.0, 300.0), (700.0, 520.0), facing=250.0, polarity=-1.0)
    assert app.spin_sign.get("SK-A") == -1
    assert "SK-A" in app.spin_sign_known
    flips = sum(1 for t, _ in app.notes if "spins the other way" in t)

    run_spin(app, (400.0, 300.0), (200.0, 150.0), facing=40.0, polarity=-1.0)
    again = sum(1 for t, _ in app.notes if "spins the other way" in t)
    assert again == flips, "a direction already confirmed was relearned"


def test_too_little_power_refuses_rather_than_driving(app, fleet):
    """Below friction the motors hum and nothing turns. That must stop the leg
    — not time out into a drive on a heading it never reached."""
    gap, ball = run_spin(app, (400.0, 300.0), (700.0, 520.0), facing=250.0,
                         power=10.0)
    assert app.p2p is None
    assert not [e for e in ball.log if e[0] == "roll" and e[2] > 0], \
        "it drove without having turned"
    assert any("raise spin power" in t for t, _ in app.notes)


def test_drift_while_driving_is_caught_and_re_aimed(app, fleet):
    """8deg/s of creep in the ball's own heading reference — several times a
    real Sphero's — must be noticed on camera and corrected, not driven along."""
    gap, ball = run_spin(app, (400.0, 300.0), (760.0, 560.0), facing=0.0,
                         drift=8.0)
    stops = sum(1 for e in ball.log if e[0] == "stop")
    assert stops >= 2, "it never stopped to re-aim"
    assert gap <= app.P2P_ARRIVE_CM + 2.0, f"stopped {gap:.1f}cm away"


def test_a_long_leg_is_not_abandoned_for_needing_many_re_aims(app, fleet):
    """A fixed cap on re-aims punishes distance. What must be stopped is
    re-aiming WITHOUT getting closer, and a long leg that closes in every
    time is doing its job."""
    gap, ball = run_spin(app, (150.0, 120.0), (1100.0, 640.0), facing=0.0,
                         drift=10.0, steps=6000)
    assert gap <= app.P2P_ARRIVE_CM + 2.0, f"gave up {gap:.1f}cm away"


def test_hunting_without_progress_gives_up_instead_of_looping(app, fleet):
    """Drift this violent cannot be driven through. It must stop, and say so,
    rather than turn and drive and turn for ever."""
    gap, ball = run_spin(app, (400.0, 300.0), (760.0, 560.0), facing=0.0,
                         drift=120.0, steps=9000)
    assert app.p2p is None, "it was still going"
    assert any("no closer" in t for t, _ in app.notes), list(app.notes)[-2:]


def test_a_ball_that_will_not_hold_still_is_refused(app, fleet):
    """If it cannot be brought to rest inside the band, the leg stops. Driving
    on a turn that never settled is how a leg sets off the wrong way."""
    import taillight as T

    ball = SpinBall(ble_name="SK-A", color="red")
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.turn_mode = "rate"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    wobble = {"t": 0.0}

    def restless(n):
        wobble["t"] += 1 / 30.0
        return (ball.facing + 40.0 * math.sin(wobble["t"] * 9.0)) % 360.0
    app.arena_heading = restless
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._shot = ((0, 0), 1.0)
        app.p2p_pick = True
        app.p2p_click((700.0, 520.0))
        for _ in range(int(30 * (app.P2P_AIM_TIMEOUT_S + 2))):
            if app.p2p is None:
                break
            app.p2p_tick()
            ball.advance(1 / 30.0, app.lab.hom, (400.0, 300.0))
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    assert app.p2p is None
    assert not [e for e in ball.log if e[0] == "roll" and e[2] > 0]


def test_going_blind_mid_spin_stops_the_motors(app, fleet):
    """The library keeps raw motors alive every 0.8s for ever. A spin left
    running while the camera cannot see would never end."""
    ball = SpinBall(ble_name="SK-A", color="red")
    ball.facing = ball.want = 250.0
    app.lab.robots["SK-A"] = ball
    app.driving = "SK-A"
    app.turn_mode = "rate"
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    app._drive_at = 0.0
    app._shot = ((0, 0), 1.0)
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))
    app.p2p_tick()
    assert ball.spinning, "the fixture should be mid-spin"
    app.lab.tracks.by_name["SK-A"].lost = True
    app.p2p_tick()
    assert not ball.spinning


def test_a_handle_that_cannot_spin_falls_back_to_heading_turns(app, fleet):
    """A simulated robot has no raw motors. It must still turn and go."""
    start, target = (400.0, 300.0), (700.0, 520.0)
    app.turn_mode = "rate"
    end, ball = run_p2p(app, start, target, bias=137.0)
    assert not hasattr(ball, "spin_raw")
    gap_cm = (float(np.linalg.norm(end - np.asarray(target)))
              / (app.lab.px_per_cm(start) or 10.0))
    assert gap_cm <= app.P2P_ARRIVE_CM + 2.0


def test_the_panel_shows_only_the_active_turns_knobs(app):
    app.turn_mode = "rate"
    app._build()
    labels = {s.label for s in app.sliders}
    assert "spin power" in labels and "turn lead" not in labels
    app.toggle_turn_mode()
    labels = {s.label for s in app.sliders}
    assert "turn lead" in labels and "spin power" not in labels


# -- following a line -----------------------------------------------------------

class NoSpinBall(SpinBall):
    """The same ball with no raw motors: a turn falls back to headings."""
    spin_raw = None


def run_line(app, ball_px, start, end, facing=0.0, polarity=1.0, handed=1.0,
             drift=0.0, steps=12000, cls=SpinBall, name="SK-A",
             keep_learned=False, during=None):
    """A whole line job — to start, face end, check, follow — closed loop."""
    import taillight as T

    ball = cls(polarity=polarity, handed=handed, drift=drift, ble_name=name,
               color="red")
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.driving = name
    app.saved_offset = lambda n: (0.0, "test")
    if not keep_learned:
        app.steer_sign.pop(name, None)
        app.spin_sign.pop(name, None)
        app.spin_sign_known.discard(name)
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": tuple(ball_px), "group": [
        {"x": ball_px[0], "y": ball_px[1], "area": 9.0, "peak": 255.0},
        {"x": ball_px[0] + 8, "y": ball_px[1], "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing

    pos = tuple(ball_px)
    trace = []
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app._shot = ((0, 0), 1.0)
        app.line_pick = []
        app.line_click(start)
        app.line_click(end)
        for _ in range(steps):
            if app.p2p is None:
                break
            app.p2p_tick()
            trace.append((app.p2p.get("stage") if app.p2p else None,
                          pos, ball.facing, ball.byte))
            if during is not None and app.p2p is not None:
                pos = during(app, ball, clock(), pos) or pos
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    return ball, trace


def cm(app, px):
    return np.asarray(app.lab.hom.to_cm([list(px)]), float).ravel()[:2]


LINE_S, LINE_E = (400.0, 300.0), (900.0, 560.0)


@pytest.mark.parametrize("handed", [1.0, -1.0])
@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_a_line_is_followed_to_its_end_either_way_round(app, fleet, handed,
                                                        polarity):
    """Both unknowns at once: which way the motors spin it, and which way a
    heading steers it. Neither is told to the controller."""
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0,
                           polarity=polarity, handed=handed)
    assert app.p2p is None, "it never finished"
    assert app.line_stats is not None, [t for t, _ in list(app.notes)[-3:]]
    assert app.line_stats["rms"] < 2.5, app.line_stats
    assert app.line_stats["end"] <= app.P2P_ARRIVE_CM + 2.0, app.line_stats


def test_it_goes_to_the_start_before_following(app, fleet):
    """The point of the whole design: a drawn path does not begin until the
    ball is at its start."""
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0)
    first_follow = next(i for i, t in enumerate(trace) if t[0] == "follow")
    at = cm(app, trace[first_follow][1])
    assert float(np.linalg.norm(at - cm(app, LINE_S))) <= app.P2P_ARRIVE_CM + 2.5


def test_it_faces_the_end_before_it_follows(app, fleet):
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0)
    first_follow = next(i for i, t in enumerate(trace) if t[0] == "follow")
    a, b = cm(app, LINE_S), cm(app, LINE_E)
    along = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0
    facing = trace[first_follow][2]
    assert abs(wrap180(facing - along)) <= app.aim_band + 25, \
        f"started following facing {facing:.0f}, line runs {along:.0f}"


def test_the_steering_check_never_drives(app, fleet):
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0)
    during = [t for t in trace if t[0] == "check"]
    assert during, "there was no check stage"
    assert all(t[3] == 0 for t in during), "it drove while checking"


def test_the_steering_direction_is_learned_once(app, fleet):
    run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0, handed=-1.0)
    assert app.steer_sign.get("SK-A") == -1
    app.lab.tracks.drop()
    ball, trace = run_line(app, (850.0, 600.0), (800.0, 520.0), (300.0, 280.0),
                           facing=90.0, handed=-1.0, keep_learned=True)
    assert not [t for t in trace if t[0] == "check"], "it checked again"
    assert app.line_stats is not None and app.line_stats["rms"] < 2.5


def test_creep_in_the_balls_heading_is_followed(app, fleet):
    """The ball's own idea of a heading slides while it drives. Held against
    the camera, the line is still held."""
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0,
                           drift=4.0)
    assert app.line_stats is not None, [t for t, _ in list(app.notes)[-3:]]
    assert app.line_stats["rms"] < 4.0, app.line_stats


def once_following(fault, after_s=1.0):
    """Run `fault(app, ball, pos)` once, a moment after following starts."""
    seen = {}

    def during(app, ball, now, pos):
        if app.p2p.get("stage") != "follow" or seen.get("done"):
            return None
        seen.setdefault("from", now)
        if now - seen["from"] >= after_s:
            seen["done"] = True
            return fault(app, ball, pos)
        return None
    return during


def shoved(app, ball, pos):
    """Picked up and set down well to one side of the line."""
    a, b = cm(app, LINE_S), cm(app, LINE_E)
    u = (b - a) / np.linalg.norm(b - a)
    side = cm(app, pos) + np.array([u[1], -u[0]]) * 35.0
    return tuple(np.asarray(app.lab.hom.to_px([list(side)]), float).ravel()[:2])


def reversed_steering(app, ball, pos):
    """Steering stops matching what was learned — every correction now
    pushes it further off."""
    ball.handed = -ball.handed


def test_the_line_recovers_from_a_knock_to_its_heading(app, fleet):
    def knock(app, ball, pos):
        ball.bias += 90.0
    run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0,
             during=once_following(knock))
    assert app.line_stats is not None, [t for t, _ in list(app.notes)[-2:]]
    assert app.line_stats["end"] <= app.P2P_ARRIVE_CM + 2.0


@pytest.mark.parametrize("fault,said", [(shoved, "off the line"),
                                        (reversed_steering, "")])
def test_a_ball_that_cannot_hold_the_line_is_stopped_not_left_wandering(
        app, fleet, fault, said):
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=200.0,
                           during=once_following(fault))
    assert "follow" in {t[0] for t in trace}, "it never got to following"
    assert app.p2p is None, "it was still going"
    assert app.line_stats is None, "it reported a finished line"
    last = list(app.notes)[-1][0]
    assert ("off the line" in last or "away from where" in last), last
    assert said in last, last


def test_a_ball_without_raw_motors_follows_too(app, fleet):
    ball, trace = run_line(app, (250.0, 470.0), LINE_S, LINE_E, facing=40.0,
                           cls=NoSpinBall)
    assert app.line_stats is not None, [t for t, _ in list(app.notes)[-3:]]
    assert app.line_stats["rms"] < 3.0, app.line_stats


def test_two_clicks_set_start_then_end(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_line()
    assert app.line_pick == []
    app.line_click(LINE_S)
    assert len(app.line_pick) == 1 and app.p2p is None
    app.line_click(LINE_E)
    assert app.line_pick is None
    assert app.p2p["kind"] == "line" and app.p2p["stage"] == "to_start"
    assert app.p2p["target"] == LINE_S, "it must head for the START first"
    app.start_line()
    assert app.p2p is None and app.line_pick is None, "pressing again cancels"


def test_a_line_too_short_to_follow_is_refused(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_line()
    app.line_click((400.0, 300.0))
    app.line_click((402.0, 301.0))
    assert app.p2p is None
    assert any("too short" in t for t, _ in app.notes)


def test_the_line_paints_while_picking_and_following(app, fleet):
    seat(app, [(400, 300, 0.0)])
    app.connect("SK-A")
    app.start_assigning()
    app.assign_click(click_at(app, 408, 300))
    app.start_line()
    paint(app)
    app.line_click(click_at(app, 400, 300))
    paint(app)
    app.line_click(click_at(app, 900, 560))
    app.p2p["stage"] = "follow"
    app.p2p["lookahead"] = cm(app, (600.0, 400.0))
    paint(app)


# -- calibrating a ball round the arena ----------------------------------------

class LagBall(SpinBall):
    """What the bench showed a real ball doing, which a plain SpinBall does not:

    - a heading change reaches the chassis `lag` seconds late;
    - STICTION: from rest it needs `breakaway` power to start turning, though
      once turning `STATIC` keeps it going — so a power low enough not to
      overshoot sticks after a stop;
    - it keeps turning after a stop (`TAU`), so a power high enough to break
      free overshoots.
    """

    def __init__(self, lag=0.0, breakaway=45.0, tau=0.25, **kw):
        super().__init__(**kw)
        self.lag, self.t, self.queue = float(lag), 0.0, []
        self.breakaway = float(breakaway)
        self.TAU = float(tau)

    def drive_raw(self, heading, byte):
        self.queue.append((self.t + self.lag, float(heading), int(byte)))

    def advance(self, dt, hom, pos_px):
        while self.queue and self.queue[0][0] <= self.t:
            _, h, b = self.queue.pop(0)
            SpinBall.drive_raw(self, h, b)
        self.t += dt
        if self.raw and abs(self.rate) < 0.5 and abs(self.raw) < self.breakaway:
            self.rate = 0.0
            return pos_px                    # stuck: not enough to break free
        return super().advance(dt, hom, pos_px)

    def stop(self):
        self.queue.clear()
        super().stop()

    def spin_raw(self, power):
        self.queue.clear()
        super().spin_raw(power)


def run_calib(app, tmp_path, monkeypatch, handed=1.0, polarity=1.0, lag=0.3,
              start_cm=(60.0, 50.0), stop_after=None, steps=30 * 600,
              cls=LagBall, noise=0.0, **kw):
    import taillight as T
    from vision import config

    monkeypatch.setattr(config, "CALIB", tmp_path)
    monkeypatch.setattr(T.App, "CALIB_LOG_DIR", tmp_path / "runs")
    name = "SK-A"
    ball = cls(lag=lag, polarity=polarity, handed=handed, ble_name=name,
               color="red", **kw)
    ball.facing = ball.want = 123.0
    app.lab.robots[name] = ball
    app.driving = name
    app.aim_band = 14.0
    start = tuple(app.lab.hom.to_px([list(start_cm)])[0])
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": tuple(start), "group": [
        {"x": start[0], "y": start[1], "area": 9.0, "peak": 255.0},
        {"x": start[0] + 8, "y": start[1], "area": 30.0, "peak": 255.0}]})
    rng = np.random.default_rng(0)
    app.arena_heading = lambda n: (ball.facing + rng.normal(0, noise)) % 360.0

    pos, trace = tuple(start), []
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    import bench_calib
    old_bc = bench_calib.time.time
    bench_calib.time.time = clock
    try:
        app.start_calib()
        for i in range(steps):
            if app.calib is None:
                break
            if stop_after is not None and i == stop_after:
                app.stop_calib("stopped")
                break
            app.calib_tick()
            trace.append(np.asarray(app.lab.hom.to_cm([list(pos)]),
                                    float).ravel()[:2])
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
        bench_calib.time.time = old_bc
    return ball, np.array(trace)


def last_note(app):
    return list(app.notes)[-1][0]


@pytest.mark.parametrize("handed", [1.0, -1.0])
@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_calib_measures_and_saves_either_way_round(app, fleet, tmp_path,
                                                   monkeypatch, handed,
                                                   polarity):
    from swarm import ball_calib

    ball, trace = run_calib(app, tmp_path, monkeypatch, handed=handed,
                            polarity=polarity)
    assert app.calib is None, "it never finished"
    assert app.calib_result is not None, last_note(app)
    rec, why = ball_calib.load("SK-A", app.lab.hom.M)
    assert why is None
    assert rec["sign"] == int(handed)
    assert rec["speed_cm_s"] == pytest.approx(rec["byte"] / 255.0 * 60.0,
                                              rel=0.1)
    assert rec["gain"] == pytest.approx(1.0, abs=0.2)
    assert rec["delay_s"] == pytest.approx(0.3, abs=0.3)
    assert rec["turn"]["breakaway_power"] >= 45.0
    assert list((tmp_path / "runs").glob("SK-A_*.json")), "no run log"


def test_calib_turn_breaks_stiction_without_overshooting(app, fleet, tmp_path,
                                                        monkeypatch):
    """The failure on the bench: 39 stuck after a stop, 50 overshot. The walk
    finds the power to break free and stops early for the coast."""
    ball, trace = run_calib(app, tmp_path, monkeypatch, breakaway=55.0,
                            tau=0.4)
    assert app.calib_result is not None, last_note(app)
    import json
    log = json.loads(next((tmp_path / "runs").glob("SK-A_*.json")).read_text())
    assert log["lead"] > 3.0, "it never learned to stop early"
    # Every turn ended inside the band without hunting back and forth.
    for t in log["turns"]:
        powers = [row[3] for row in t["samples"] if row[3]]
        reversals = sum(1 for a, b in zip(powers, powers[1:]) if a * b < 0)
        assert reversals <= 2, f"turn on leg {t['leg']} hunted {reversals}x"


def test_calib_visits_every_corner_and_the_centre(app, fleet, tmp_path,
                                                 monkeypatch):
    from bench_calib import CalibWalk
    ball, trace = run_calib(app, tmp_path, monkeypatch)
    hom, k = app.lab.hom, CalibWalk.INSET_CM
    w, h = hom.width, hom.height
    for c in [(k, k), (w - k, k), (w - k, h - k), (k, h - k), (w / 2, h / 2)]:
        near = np.min(np.linalg.norm(trace - np.array(c), axis=1))
        assert near < 12.0, f"never came within 12cm of {c} ({near:.0f})"


@pytest.mark.parametrize("handed", [1.0, -1.0])
def test_calib_never_leaves_the_arena(app, fleet, tmp_path, monkeypatch,
                                      handed):
    """Two real runs steered off the top edge and lost the ball."""
    ball, trace = run_calib(app, tmp_path, monkeypatch, handed=handed)
    hom = app.lab.hom
    assert trace[:, 0].min() > 3 and trace[:, 1].min() > 3
    assert trace[:, 0].max() < hom.width - 3
    assert trace[:, 1].max() < hom.height - 3


def test_calib_survives_a_noisy_heading(app, fleet, tmp_path, monkeypatch):
    run_calib(app, tmp_path, monkeypatch, noise=2.5)
    assert app.calib_result is not None, last_note(app)


def test_a_ball_that_ignores_steering_saves_nothing(app, fleet, tmp_path,
                                                    monkeypatch):
    class Deaf(LagBall):
        def drive_raw(self, heading, byte):
            super().drive_raw(0.0 if byte else heading, byte)

    run_calib(app, tmp_path, monkeypatch, cls=Deaf)
    assert app.calib is None and app.calib_result is None
    assert not list(tmp_path.glob("ball_*.json")), "it saved anyway"
    assert list((tmp_path / "runs").glob("SK-A_*.json")), "no run log"


def test_stopping_calib_part_way_saves_nothing_but_logs(app, fleet, tmp_path,
                                                         monkeypatch):
    run_calib(app, tmp_path, monkeypatch, stop_after=30 * 60)
    assert app.calib is None and app.calib_result is None
    assert not list(tmp_path.glob("ball_*.json"))
    import json
    got = json.loads(next((tmp_path / "runs").glob("SK-A_*.json")).read_text())
    assert got["saved"] is None and got["why"] == "stopped"
    assert got["legs"] and got["turns"], "the log has nothing in it"


def test_calib_does_not_start_over_a_p2p(app, fleet):
    app.p2p = {"kind": "p2p"}
    app.start_calib()
    assert app.calib is None


def test_calib_overlay_paints(app, fleet, tmp_path, monkeypatch):
    run_calib(app, tmp_path, monkeypatch, steps=30 * 8)
    assert app.calib is not None
    paint(app)
    app.stop_calib("stopped")


# -- the tapered turn -------------------------------------------------------------

class BenchBall(LagBall):
    """Fitted to 20 real turns of SK-914A (runs/calib/SK-914A_0916_215639):
    stuck below ~52 from rest; once free it keeps turning above ~35 and at
    power 55-60 turns 100-150 deg/s — stick-slip, not a smooth rate."""
    STATIC = 35.0
    GAIN = 5.5

    def __init__(self, breakaway=52.0, tau=0.2, **kw):
        super().__init__(breakaway=breakaway, tau=tau, **kw)


def run_turn(app, start, target, mode="taper", facing=0.0, polarity=1.0,
             breakaway=55.0, tau=0.4, power=45.0, band=14.0, noise=0.0,
             steps=30 * 60, name="SK-A", cls=None, latency=0.0):
    """One p2p leg with a ball like the bench's: sticks from rest below
    `breakaway`, runs on after a stop. Returns (gap, ball, aim log)."""
    import taillight as T

    ball = (cls or LagBall)(breakaway=breakaway, tau=tau, polarity=polarity,
                            ble_name=name, color="red")
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.driving = name
    app.turn_mode = mode
    app.spin_power, app.aim_band = float(power), float(band)
    app.saved_offset = lambda n: (0.0, "test")
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": tuple(start), "group": [
        {"x": start[0], "y": start[1], "area": 9.0, "peak": 255.0},
        {"x": start[0] + 8, "y": start[1], "area": 30.0, "peak": 255.0}]})
    rng = np.random.default_rng(1)
    seen = deque([ball.facing] * max(1, int(round(latency * 30)) + 1),
                 maxlen=max(1, int(round(latency * 30)) + 1))
    app.arena_heading = lambda n: (seen[0] + rng.normal(0, noise)) % 360.0

    pos, aim = tuple(start), []
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app._shot = ((0, 0), 1.0)
        app.p2p_pick = True
        app.p2p_click(target)
        for _ in range(steps):
            if app.p2p is None:
                break
            if app.p2p["phase"] == "aim":
                aim.append((clock(), ball.facing, ball.raw))
            app.p2p_tick()
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            seen.append(ball.facing)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    gap_cm = float(np.linalg.norm(
        np.asarray(app.lab.hom.to_cm([list(pos)]), float).ravel()[:2]
        - np.asarray(app.lab.hom.to_cm([list(target)]), float).ravel()[:2]))
    return gap_cm, ball, aim


def reversals(aim):
    powers = [r for _, _, r in aim if r]
    return sum(1 for a, b in zip(powers, powers[1:]) if a * b < 0)


@pytest.mark.parametrize("facing", [0.0, 90.0, 200.0, 315.0])
@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_taper_arrives_on_a_sticky_ball_from_any_facing(app, fleet, facing,
                                                        polarity):
    gap, ball, aim = run_turn(app, (400.0, 300.0), (700.0, 520.0),
                              facing=facing, polarity=polarity)
    assert app.p2p is None and gap <= app.P2P_ARRIVE_CM + 2.0, (
        gap, list(app.notes)[-2:])
    assert reversals(aim) <= 3, f"hunted: {reversals(aim)} reversals"


def test_taper_breaks_free_where_a_fixed_low_power_sticks(app, fleet):
    """The bench: 39 stuck. Spin power 35 is below this ball's breakaway."""
    gap, _, _ = run_turn(app, (400.0, 300.0), (700.0, 520.0), facing=200.0,
                         power=35.0)
    assert gap <= app.P2P_ARRIVE_CM + 2.0, list(app.notes)[-2:]
    assert app.taper["SK-A"]["kick"] >= 50.0


def test_taper_eases_the_power_off_near_the_angle(app, fleet):
    gap, ball, aim = run_turn(app, (400.0, 300.0), (700.0, 520.0),
                              facing=200.0, power=80.0)
    spun = [abs(r) for _, _, r in aim if r]
    assert spun, "it never spun"
    assert min(spun[len(spun) // 2:]) < max(spun[:len(spun) // 2]), \
        "power never came down as the angle closed"


def test_taper_hunts_less_than_a_fixed_power_that_overshoots(app, fleet):
    """The bench: 50 overshoots. Same ball, same power, both modes."""
    _, _, fixed = run_turn(app, (400.0, 300.0), (700.0, 520.0), mode="rate",
                           facing=200.0, power=60.0, steps=30 * 12)
    app.stop_p2p("next")
    _, _, taper = run_turn(app, (400.0, 300.0), (700.0, 520.0), facing=200.0,
                           power=60.0, steps=30 * 12)
    t_fixed = fixed[-1][0] - fixed[0][0] if fixed else 0.0
    t_taper = taper[-1][0] - taper[0][0]
    assert reversals(taper) <= reversals(fixed)
    assert t_taper <= t_fixed + 0.5, (t_taper, t_fixed)


def test_taper_learns_across_legs_and_turns_faster_next_time(app, fleet):
    run_turn(app, (400.0, 300.0), (700.0, 520.0), facing=200.0)
    kick = app.taper["SK-A"]["kick"]
    app.lab.tracks.drop()
    _, _, aim = run_turn(app, (700.0, 520.0), (400.0, 300.0), facing=20.0)
    assert app.taper["SK-A"]["kick"] == pytest.approx(kick, abs=15.0)
    assert reversals(aim) <= 2


def test_taper_survives_a_noisy_heading(app, fleet):
    gap, _, aim = run_turn(app, (400.0, 300.0), (700.0, 520.0), facing=200.0,
                           noise=2.5)
    assert gap <= app.P2P_ARRIVE_CM + 2.0, list(app.notes)[-2:]


def test_taper_a_dead_ball_is_refused_not_driven(app, fleet):
    gap, ball, _ = run_turn(app, (400.0, 300.0), (700.0, 520.0), facing=200.0,
                            breakaway=999.0)
    assert app.p2p is None
    assert not [e for e in ball.log if e[0] == "roll" and e[2] > 0], \
        "it drove on a turn that never happened"


# -- polyline and freehand paths ----------------------------------------------------

def px_at(app, x_cm, y_cm):
    return tuple(float(v) for v in app.lab.hom.to_px([[x_cm, y_cm]])[0])


def run_path(app, ball_cm, pts_cm, kind="poly", facing=200.0, handed=1.0,
             polarity=1.0, drift=0.0, steps=30 * 150, name="SK-A",
             cls=None):
    """Draw a path through the real picking calls, then drive it closed loop."""
    import taillight as T

    ball = (cls or SpinBall)(polarity=polarity, handed=handed, drift=drift,
                    ble_name=name, color="red")
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.driving = name
    app.saved_offset = lambda n: (0.0, "test")
    app.steer_sign.pop(name, None)
    app.spin_sign.pop(name, None)
    app.spin_sign_known.discard(name)
    start = px_at(app, *ball_cm)
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": start, "group": [
        {"x": start[0], "y": start[1], "area": 9.0, "peak": 255.0},
        {"x": start[0] + 8, "y": start[1], "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing

    pos, trace = start, []
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app._shot = ((0, 0), 1.0)
        app.start_path(kind)
        pts = [px_at(app, *c) for c in pts_cm]
        if kind == "poly":
            for q in pts:
                app.path_press(q)
            app.finish_path()
        else:
            app.path_press(pts[0])
            for q in pts[1:]:
                app.path_drag(q)
            app.path_release()
        for _ in range(steps):
            if app.p2p is None:
                break
            app.p2p_tick()
            here = np.asarray(app.lab.hom.to_cm([list(pos)]), float).ravel()[:2]
            trace.append((app.p2p.get("stage") if app.p2p else None, here))
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    return ball, trace


def dense(pts_cm, step=1.0):
    out = []
    for a, b in zip(pts_cm[:-1], pts_cm[1:]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                for k in range(n)]
    return out + [pts_cm[-1]]


def off_path(trace, pts_cm):
    """Worst distance of the FOLLOWED part of a run from the drawn path."""
    ref = np.asarray(dense(pts_cm, 0.5))
    followed = [p for st, p in trace if st == "follow"]
    return max(float(np.min(np.linalg.norm(ref - p, axis=1)))
               for p in followed)


L_SHAPE = [(30.0, 30.0), (100.0, 30.0), (100.0, 85.0)]
ZIGZAG = [(25.0, 25.0), (60.0, 60.0), (95.0, 25.0), (115.0, 70.0)]


@pytest.mark.parametrize("handed", [1.0, -1.0])
@pytest.mark.parametrize("shape", [L_SHAPE, ZIGZAG])
def test_a_polyline_is_followed_to_its_end(app, fleet, handed, shape):
    ball, trace = run_path(app, (70.0, 70.0), shape, handed=handed)
    assert app.p2p is None and app.path_stats is not None, \
        list(app.notes)[-2:]
    assert app.path_stats["end"] <= app.P2P_ARRIVE_CM + 2.0, app.path_stats
    assert app.path_stats["rms"] < 3.0, app.path_stats
    # Corners get cut by roughly the lookahead, never wildly.
    assert off_path(trace, shape) < app.lookahead_cm, off_path(trace, shape)


def test_a_freehand_curve_is_followed(app, fleet):
    curve = [(25.0 + 90.0 * t, 55.0 + 22.0 * math.sin(t * 2 * math.pi))
             for t in np.linspace(0, 1, 80)]
    rng = np.random.default_rng(3)
    wobbly = [(x + rng.normal(0, 0.4), y + rng.normal(0, 0.4))
              for x, y in curve]
    ball, trace = run_path(app, (70.0, 90.0), wobbly, kind="free")
    assert app.path_stats is not None, list(app.notes)[-2:]
    assert app.path_stats["kind"] == "free"
    assert app.path_stats["rms"] < 3.0, app.path_stats
    assert app.path_stats["end"] <= app.P2P_ARRIVE_CM + 2.0


def test_a_freehand_loop_is_not_short_cut_at_its_crossing(app, fleet):
    """A figure drawn back across itself: the ball must go round the loop,
    not jump to where the path crosses."""
    loop = ([(20.0, 55.0), (60.0, 55.0)]
            + [(80.0 + 20.0 * math.sin(a), 55.0 - 20.0 * math.cos(a))
               for a in np.linspace(-math.pi / 2, 1.5 * math.pi, 40)]
            + [(60.0, 55.0), (60.0, 90.0)])
    loop = [(x, y) for x, y in loop]
    ball, trace = run_path(app, (30.0, 90.0), loop, kind="free")
    assert app.path_stats is not None, list(app.notes)[-2:]
    far = np.array([100.0, 55.0])
    assert min(np.linalg.norm(p - far) for _, p in trace) < 8.0, \
        "it skipped the loop"


def test_a_path_goes_to_its_start_before_following(app, fleet):
    ball, trace = run_path(app, (70.0, 70.0), L_SHAPE)
    first = next(p for st, p in trace if st == "follow")
    assert np.linalg.norm(first - np.array(L_SHAPE[0])) <= app.P2P_ARRIVE_CM + 2.5


def test_a_path_survives_heading_creep(app, fleet):
    ball, trace = run_path(app, (70.0, 70.0), L_SHAPE, drift=4.0)
    assert app.path_stats is not None and app.path_stats["rms"] < 4.0, \
        (app.path_stats, list(app.notes)[-2:])


def test_a_path_the_ball_cannot_hold_is_stopped(app, fleet):
    """Steering that stops working once it is following: stopped, not a ball
    wandering off with a finished-path message."""
    class Deaf(SpinBall):
        def drive_raw(self, heading, byte):
            if byte and heading != 0.0:
                heading = 90.0      # drives to the start, then ignores steers
            super().drive_raw(heading, byte)

    ball, trace = run_path(app, (70.0, 70.0), ZIGZAG, cls=Deaf)
    assert app.p2p is None
    assert app.path_stats is None, "it reported a finished path"
    assert "path" in list(app.notes)[-1][0] or "away" in list(app.notes)[-1][0]


def test_poly_picking_clicks_backspace_and_done(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_path("poly")
    assert any(b.label == "done" for b in app.buttons)
    for q in [px_at(app, 30, 30), px_at(app, 90, 30), px_at(app, 5, 5)]:
        app.path_press(q)
    app.path_pick["pts"].pop()                        # backspace
    paint(app)
    app.finish_path()
    got = app.p2p
    assert got["kind"] == "line" and got["stage"] == "to_start"
    assert got["path"]["kind"] == "poly"
    assert got["path"]["total"] == pytest.approx(60.0, abs=1.0)
    assert np.allclose(app.lab.hom.to_cm([list(got["target"])])[0],
                       (30.0, 30.0), atol=0.5), "it must head for the START"
    paint(app)


def test_a_path_with_one_point_or_too_short_is_refused(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_path("poly")
    app.path_press(px_at(app, 30, 30))
    app.finish_path()
    assert app.p2p is None and "two points" in list(app.notes)[-1][0]
    app.start_path("free")
    app.path_press(px_at(app, 30, 30))
    app.path_drag(px_at(app, 33, 30))
    app.path_release()
    assert app.p2p is None and "too short" in list(app.notes)[-1][0]


def test_free_drawing_collects_while_dragging_and_starts_on_release(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_path("free")
    app.path_press(px_at(app, 30, 30))
    assert app.path_pick["drawing"]
    for x in range(32, 100, 2):
        app.path_drag(px_at(app, x, 30 + (x - 30) * 0.3))
    paint(app)
    assert app.p2p is None, "it must not start before the release"
    app.path_release()
    assert app.p2p is not None and app.p2p["path"]["kind"] == "free"
    assert app.path_pick is None


def test_space_and_stop_cancel_drawing(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app.start_path("poly")
    stop = next(b for b in app.buttons if b.label == "stop")
    stop.cb()
    assert app.path_pick is None and app.p2p is None


def test_path_buttons_fit_and_do_not_overlap(app):
    from taillight import W, H
    labels = [b.label for b in app.buttons]
    assert "poly" in labels and "free" in labels
    window = pygame.Rect(0, 0, W, H)
    for b in app.buttons:
        assert window.contains(b.rect), b.label
    for i, a in enumerate(app.buttons):
        for b in app.buttons[i + 1:]:
            assert not a.rect.colliderect(b.rect), (a.label, b.label)


# -- orbits ---------------------------------------------------------------------------

def run_orbit(app, ball_cm, centre_cm, radius_cm, direction="ccw", handed=1.0,
              facing=200.0, laps_per_run=1, seconds=90, drift=0.0,
              cls=None, stop_at_s=None, name="SK-A"):
    import taillight as T

    ball = (cls or SpinBall)(handed=handed, drift=drift, ble_name=name,
                             color="red")
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.driving = name
    app.saved_offset = lambda n: (0.0, "test")
    app.steer_sign.pop(name, None)
    app.ORBIT_LAPS_PER_RUN = laps_per_run
    app.orbit_dir = direction
    start = px_at(app, *ball_cm)
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": start, "group": [
        {"x": start[0], "y": start[1], "area": 9.0, "peak": 255.0},
        {"x": start[0] + 8, "y": start[1], "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing

    pos, trace = start, []
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    try:
        app._drive_at = 0.0
        app._shot = ((0, 0), 1.0)
        app.start_orbit()
        app.orbit_click(px_at(app, *centre_cm))
        app.orbit_click(px_at(app, centre_cm[0] + radius_cm, centre_cm[1]))
        for i in range(int(seconds * 30)):
            if app.orbit_run is None:
                break
            if stop_at_s is not None and i == int(stop_at_s * 30):
                app.stop_p2p("stopped")
            app.p2p_tick()
            app.orbit_tick()
            here = np.asarray(app.lab.hom.to_cm([list(pos)]), float).ravel()[:2]
            stage = app.p2p.get("stage") if app.p2p else None
            trace.append((stage, here, (app.p2p or {}).get("phase")))
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    return ball, trace


def swept_deg(trace, centre):
    """Signed angle swept round the centre while following (screen sense:
    positive is clockwise, since screen y runs down)."""
    ang = [math.atan2(p[1] - centre[1], p[0] - centre[0])
           for st, p, _ in trace if st == "follow"]
    return math.degrees(float(np.sum(np.diff(np.unwrap(ang))))) if ang else 0.0


C, R = (70.0, 55.0), 30.0


@pytest.mark.parametrize("handed", [1.0, -1.0])
@pytest.mark.parametrize("direction", ["ccw", "cw"])
def test_an_orbit_goes_round_the_right_way_and_holds_the_radius(
        app, fleet, handed, direction):
    ball, trace = run_orbit(app, (70.0, 100.0), C, R, direction=direction,
                            handed=handed, seconds=90)
    assert app.orbit_run is not None, list(app.notes)[-3:]
    swept = swept_deg(trace, C)
    assert abs(swept) > 1.5 * 360, f"only {swept:.0f}deg swept"
    assert (swept > 0) == (direction == "cw"), swept
    radii = [float(np.linalg.norm(p - np.array(C)))
             for st, p, _ in trace[len(trace) // 3:] if st == "follow"]
    err = np.asarray(radii) - R
    assert float(np.sqrt(np.mean(err ** 2))) < 3.0, float(np.sqrt(np.mean(err ** 2)))
    app.stop_p2p("done")


def test_an_orbit_carries_on_from_one_chunk_of_laps_to_the_next(app, fleet):
    ball, trace = run_orbit(app, (70.0, 100.0), C, R, laps_per_run=1,
                            seconds=90)
    assert app.orbit_run is not None and app.orbit_run["laps"] >= 2, \
        list(app.notes)[-3:]
    # Between chunks it does not go back to the start or spin for it.
    stages = [st for st, _, _ in trace]
    starts = sum(1 for a, b in zip(stages, stages[1:])
                 if b == "to_start" and a != "to_start")
    assert starts <= 1, f"went back to the start {starts} times"
    app.stop_p2p("done")


def test_an_orbit_starts_from_the_nearest_point_on_the_circle(app, fleet):
    ball, trace = run_orbit(app, (70.0, 100.0), C, R, seconds=40)
    first = next(p for st, p, _ in trace if st == "follow")
    nearest = np.array([70.0, 85.0])
    assert np.linalg.norm(first - nearest) <= app.P2P_ARRIVE_CM + 3.0
    app.stop_p2p("done")


def test_stopping_ends_the_orbit_it_does_not_restart(app, fleet):
    ball, trace = run_orbit(app, (70.0, 100.0), C, R, seconds=40, stop_at_s=20)
    assert app.orbit_run is None and app.p2p is None
    assert "ended" in list(app.notes)[-1][0]


def test_an_orbit_that_is_too_small_or_leaves_the_arena_is_refused(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_orbit()
    app.orbit_click(px_at(app, 70, 55))
    app.orbit_click(px_at(app, 75, 55))
    assert app.orbit_run is None and "too small" in list(app.notes)[-1][0]
    app.start_orbit()
    app.orbit_click(px_at(app, 20, 55))
    app.orbit_click(px_at(app, 60, 55))
    assert app.orbit_run is None and "leaves the arena" in list(app.notes)[-1][0]


def test_a_tight_orbit_warns_but_runs(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_orbit()
    app.orbit_click(px_at(app, 70, 55))
    assert any(b.label == "stop" for b in app.buttons)
    paint(app)
    app.orbit_click(px_at(app, 85, 55))
    assert app.orbit_run is not None
    assert any("tight" in t for t, _ in app.notes)
    paint(app)
    app.stop_p2p("stopped")
    app.orbit_tick()
    assert app.orbit_run is None


def test_orbit_direction_toggle_and_buttons_fit(app, fleet):
    from taillight import W, H
    app.orbit_dir = "ccw"
    app.toggle_orbit_dir()
    assert app.orbit_dir == "cw" and any(b.label == "cw" for b in app.buttons)
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app.start_path("poly")                   # "done" shares the row
    assert any(b.label == "done" for b in app.buttons)
    window = pygame.Rect(0, 0, W, H)
    for i, a in enumerate(app.buttons):
        assert window.contains(a.rect), a.label
        for b in app.buttons[i + 1:]:
            assert not a.rect.colliderect(b.rect), (a.label, b.label)


# -- never quit: the job supervisor -------------------------------------------------

def supervise(app, ball, name, pos, seconds, during=None):
    """The bench's own order each frame: supervisor, pursuit start, run,
    orbit, patrol."""
    import taillight as T
    clock = Clock()
    old_time, old_perf = T.time.time, T.time.perf_counter
    T.time.time, T.time.perf_counter = clock, clock
    trace = []
    try:
        app._drive_at = 0.0
        for i in range(int(seconds * 30)):
            if during is not None:
                pos = during(app, ball, i, pos) or pos
            app.job_tick()
            app.pursuit_tick()
            app.p2p_tick()
            app.orbit_tick()
            app.patrol_tick()
            if (app.p2p is None and app.job is None and app.orbit_run is None
                    and app.patrol_run is None and app._pursuit_pending is None):
                break
            here = np.asarray(app.lab.hom.to_cm([list(pos)]), float).ravel()[:2]
            trace.append(((app.p2p or {}).get("stage"), here))
            pos = ball.advance(1 / 30.0, app.lab.hom, pos)
            app.lab.tracks.by_name[name].centre = pos
            clock.tick(1 / 30.0)
    finally:
        T.time.time, T.time.perf_counter = old_time, old_perf
    return pos, trace


def seat_ball(app, ball, at_px, name="SK-A", facing=200.0):
    ball.facing = ball.want = float(facing)
    app.lab.robots[name] = ball
    app.driving = name
    app.saved_offset = lambda n: (0.0, "test")
    app.steer_sign.pop(name, None)
    app.lab.tracks.drop()
    app.lab.tracks.assign(name, {"centre": tuple(at_px), "group": [
        {"x": at_px[0], "y": at_px[1], "area": 9.0, "peak": 255.0},
        {"x": at_px[0] + 8, "y": at_px[1], "area": 30.0, "peak": 255.0}]})
    app.arena_heading = lambda n: ball.facing
    app._shot = ((0, 0), 1.0)


class PullBackBall(LagBall):
    """Runs on after a spin and is pulled back — what made the steering check
    read nothing on the bench and quit."""

    def __init__(self, **kw):
        kw.pop("lag", None)
        super().__init__(lag=0.0, breakaway=40.0, tau=0.25, **kw)


@pytest.fixture
def no_calib(tmp_path, monkeypatch):
    from vision import config
    monkeypatch.setattr(config, "CALIB", tmp_path)
    return tmp_path


def test_the_line_that_quit_on_the_bench_now_finishes(app, fleet, no_calib):
    """The exact failure: the check reads the pull-back, gives up. Now it
    retries from where it is until the line is done."""
    ball = PullBackBall(handed=-1.0, ble_name="SK-A", color="red")
    seat_ball(app, ball, (250.0, 470.0), facing=0.0)
    app.spin_power = 50.0
    app.line_pick = []
    app.line_click(LINE_S)
    app.line_click(LINE_E)
    supervise(app, ball, "SK-A", (250.0, 470.0), seconds=180)
    assert app.line_stats is not None, [t for t, _ in app.notes]
    assert app.job is None


def test_a_calibrated_ball_skips_the_steering_check(app, fleet, no_calib):
    from swarm import ball_calib
    ball_calib.save({"schema": ball_calib.SCHEMA, "name": "SK-A",
                     "homography": ball_calib.fingerprint(app.lab.hom.M),
                     "sign": -1})
    ball = PullBackBall(handed=-1.0, ble_name="SK-A", color="red")
    seat_ball(app, ball, (250.0, 470.0), facing=0.0)
    app.spin_power = 50.0
    app.line_pick = []
    app.line_click(LINE_S)
    app.line_click(LINE_E)
    pos, trace = supervise(app, ball, "SK-A", (250.0, 470.0), seconds=120)
    assert app.line_stats is not None, [t for t, _ in app.notes]
    assert not [st for st, _ in trace if st == "check"], "it checked anyway"
    assert app.steer_sign["SK-A"] == -1


def test_a_wrong_calibrated_direction_is_flipped_not_repeated(app, fleet,
                                                              no_calib):
    from swarm import ball_calib
    ball_calib.save({"schema": ball_calib.SCHEMA, "name": "SK-A",
                     "homography": ball_calib.fingerprint(app.lab.hom.M),
                     "sign": 1})
    ball = SpinBall(handed=-1.0, ble_name="SK-A", color="red")
    seat_ball(app, ball, (250.0, 470.0))
    app.line_pick = []
    app.line_click(LINE_S)
    app.line_click(LINE_E)
    supervise(app, ball, "SK-A", (250.0, 470.0), seconds=240)
    assert app.line_stats is not None, [t for t, _ in app.notes]
    assert app.steer_sign["SK-A"] == -1


def test_a_p2p_that_cannot_turn_raises_power_and_gets_there(app, fleet,
                                                             no_calib):
    ball = LagBall(breakaway=62.0, tau=0.1, ble_name="SK-A", color="red")
    seat_ball(app, ball, (400.0, 300.0))
    app.turn_mode, app.spin_power = "rate", 50.0
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))
    pos, _ = supervise(app, ball, "SK-A", (400.0, 300.0), seconds=120)
    assert any("arrived" in t for t, _ in app.notes), [t for t, _ in app.notes]
    assert app.spin_power > 50.0 and app.job is None


def test_a_line_retry_starts_from_where_the_ball_is_not_the_start(
        app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, (250.0, 470.0))
    app.line_pick = []
    app.line_click(LINE_S)
    app.line_click(LINE_E)
    shoved, retry_target = {}, []

    def shove(app, ball, i, pos):
        got = app.p2p or {}
        if shoved and not retry_target and got.get("stage") == "to_start":
            retry_target.append(cm(app, got["target"]))
        if got.get("stage") == "follow" and not shoved and got.get("cross") \
                and len(got["cross"]) > 120:
            shoved["at"] = np.asarray(app.lab.hom.to_cm([list(pos)]),
                                      float).ravel()[:2]
            u = np.subtract(cm(app, LINE_E), cm(app, LINE_S))
            u = u / np.linalg.norm(u)
            side = shoved["at"] + np.array([u[1], -u[0]]) * 35.0
            return tuple(app.lab.hom.to_px([list(side)])[0])
        return None

    pos, trace = supervise(app, ball, "SK-A", (250.0, 470.0), seconds=180,
                           during=shove)
    assert shoved, "the shove never happened"
    assert app.retries >= 1 and retry_target, "it never retried"
    assert app.line_stats is not None, [t for t, _ in app.notes]
    a, b = cm(app, LINE_S), cm(app, LINE_E)
    u = (b - a) / np.linalg.norm(b - a)
    along = float(np.dot(retry_target[0] - a, u))
    shoved_along = float(np.dot(shoved["at"] - a, u))
    assert along == pytest.approx(shoved_along, abs=6.0), \
        f"retry started {along:.0f}cm along, shoved at {shoved_along:.0f}cm"
    assert along > 10.0, "it went back to the start"


def test_stopping_a_job_does_not_retry(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, (400.0, 300.0))
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))

    def stop_soon(app, ball, i, pos):
        if i == 30:
            app.stop_p2p("stopped")

    supervise(app, ball, "SK-A", (400.0, 300.0), seconds=20, during=stop_soon)
    assert app.p2p is None and app.job is None
    assert app.retries == 0, "it restarted a job the user stopped"
    assert not any("arrived" in t for t, _ in app.notes)


def test_a_job_waits_while_the_ball_is_unseen_then_carries_on(app, fleet,
                                                              no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, (400.0, 300.0))
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))

    def hide(app, ball, i, pos):
        t = app.lab.tracks.by_name["SK-A"]
        t.lost = 60 <= i < 200            # ~4.5s gone: longer than p2p waits

    supervise(app, ball, "SK-A", (400.0, 300.0), seconds=90, during=hide)
    assert any("arrived" in t for t, _ in app.notes), [t for t, _ in app.notes]
    assert app.retries >= 1


def test_a_failing_orbit_is_retried_and_keeps_orbiting(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, px_at(app, 70.0, 100.0))
    app.ORBIT_LAPS_PER_RUN = 1
    app.start_orbit()
    app.orbit_click(px_at(app, *C))
    app.orbit_click(px_at(app, C[0] + R, C[1]))
    shoved = {}

    def shove(app, ball, i, pos):
        got = app.p2p or {}
        if got.get("stage") == "follow" and not shoved and len(
                got.get("cross") or []) > 60:
            shoved["i"] = i
            return px_at(app, 70.0, 55.0)         # dropped at the centre
        return None

    pos, trace = supervise(app, ball, "SK-A", px_at(app, 70.0, 100.0),
                           seconds=120, during=shove)
    assert shoved and app.orbit_run is not None, [t for t, _ in app.notes]
    assert app.retries >= 1
    late = [p for st, p in trace[-600:] if st == "follow"]
    assert late, "not orbiting again after the retry"
    app.stop_p2p("stopped")


# -- patrols ---------------------------------------------------------------------------

SQUARE = [(35.0, 30.0), (105.0, 30.0), (105.0, 80.0), (35.0, 80.0)]


def start_patrol_at(app, ball, pts_cm, style, rounds_per_run=1, at_cm=(70.0, 55.0)):
    seat_ball(app, ball, px_at(app, *at_cm))
    app.PATROL_ROUNDS_PER_RUN = rounds_per_run
    app.patrol_style = style
    app.start_patrol()
    for c in pts_cm:
        app.patrol_click(px_at(app, *c))
    app.finish_patrol()
    return px_at(app, *at_cm)


def visits(trace, point, within=8.0):
    """How many separate times the ball came within `within` of `point`."""
    near = [float(np.linalg.norm(p - np.array(point))) <= within
            for _, p in trace]
    return sum(1 for a, b in zip([False] + near, near) if b and not a)


def test_a_loop_patrol_goes_round_again_and_again(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    pos = start_patrol_at(app, ball, SQUARE, "loop")
    pos, trace = supervise(app, ball, "SK-A", pos, seconds=150)
    assert app.patrol_run is not None, [t for t, _ in app.notes]
    assert app.patrol_run["rounds"] >= 2, app.patrol_run["rounds"]
    for corner in SQUARE:
        assert visits(trace, corner, within=app.lookahead_cm) >= 2, corner
    # Between runs it does not drive a leg back to the start.
    stages = [st for st, _ in trace]
    legs = sum(1 for a, b in zip(stages, stages[1:])
               if b == "to_start" and a != "to_start")
    assert legs == 0, f"drove back to the start {legs} times"
    app.stop_p2p("stopped")
    app.patrol_tick()
    assert app.patrol_run is None


def test_a_back_and_forth_patrol_turns_round_at_each_end(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    route = [(30.0, 40.0), (70.0, 70.0), (110.0, 40.0)]
    pos = start_patrol_at(app, ball, route, "bounce", at_cm=(30.0, 80.0))
    pos, trace = supervise(app, ball, "SK-A", pos, seconds=150)
    assert app.patrol_run is not None and app.patrol_run["rounds"] >= 3, \
        [t for t, _ in app.notes]
    assert visits(trace, route[0]) >= 2 and visits(trace, route[-1]) >= 2
    # Each end is a turn round, not a drive back to the first point.
    stages = [st for st, _ in trace]
    legs = sum(1 for a_, b_ in zip(stages, stages[1:])
               if b_ == "to_start" and a_ != "to_start")
    assert legs == 0, f"drove back to the start {legs} times"
    # It stays on the route, never swinging wide at the ends.
    ref = np.asarray(dense(route, 0.5))
    followed = [p for st, p in trace if st == "follow"]
    worst = max(float(np.min(np.linalg.norm(ref - p, axis=1)))
                for p in followed)
    assert worst < app.lookahead_cm, worst
    app.stop_p2p("stopped")


def test_a_patrol_starts_at_its_first_point(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    pos = start_patrol_at(app, ball, SQUARE, "loop", at_cm=(70.0, 95.0))
    assert np.allclose(cm(app, app.p2p["target"]), SQUARE[0], atol=0.5)
    pos, trace = supervise(app, ball, "SK-A", pos, seconds=40)
    first = next(p for st, p in trace if st == "follow")
    assert np.linalg.norm(first - np.array(SQUARE[0])) <= app.P2P_ARRIVE_CM + 2.5
    app.stop_p2p("stopped")


def test_a_failed_patrol_run_is_retried_and_the_patrol_carries_on(
        app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    pos = start_patrol_at(app, ball, SQUARE, "loop")
    shoved = {}

    def shove(app, ball, i, pos):
        got = app.p2p or {}
        if got.get("stage") == "follow" and not shoved and len(
                got.get("cross") or []) > 150:
            shoved["i"] = i
            return px_at(app, 8.0, 105.0)     # dropped 39cm from the route
        return None

    pos, trace = supervise(app, ball, "SK-A", pos, seconds=200, during=shove)
    assert shoved and app.retries >= 1
    assert app.patrol_run is not None and app.patrol_run["rounds"] >= 1, \
        [t for t, _ in app.notes]
    app.stop_p2p("stopped")


def test_patrol_needs_enough_points(app, fleet):
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.patrol_style = "loop"
    app.start_patrol()
    app.patrol_click(px_at(app, 30, 30))
    app.patrol_click(px_at(app, 90, 30))
    app.finish_patrol()
    assert app.patrol_run is None and app.patrol_pick is not None, \
        "a 2-point loop must be refused and let you add a point"
    assert "back+forth" in list(app.notes)[-1][0]
    app.toggle_patrol_style()
    app.finish_patrol()
    assert app.patrol_run is not None and app.patrol_run["style"] == "bounce"
    app.start_patrol()                      # pressing patrol again ends it
    assert app.patrol_run is None and app.p2p is None


def test_patrol_picking_ui_paints_and_fits(app, fleet):
    from taillight import W, H
    app.connect("SK-A")
    app.lab.tracks.assign("SK-A", {"centre": (400.0, 300.0), "group": [
        {"x": 400.0, "y": 300.0, "area": 9.0, "peak": 255.0},
        {"x": 408.0, "y": 300.0, "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app._shot = ((0, 0), 1.0)
    app.start_patrol()
    for c in SQUARE:
        app.patrol_click(px_at(app, *c))
    labels = [b.label for b in app.buttons]
    assert "done" in labels and "stop" in labels and "patrol" in labels
    paint(app)
    window = pygame.Rect(0, 0, W, H)
    for i, a in enumerate(app.buttons):
        assert window.contains(a.rect), a.label
        for b in app.buttons[i + 1:]:
            assert not a.rect.colliderect(b.rect), (a.label, b.label)
    next(b for b in app.buttons if b.label == "stop").cb()
    assert app.patrol_pick is None


# -- the model's tools (bench_agent) -------------------------------------------------

def agent_ready(app, ball=None, at_cm=(70.0, 55.0)):
    ball = ball or SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, px_at(app, *at_cm))
    return ball


def test_agent_offers_only_the_bench_jobs(app):
    from bench_agent import TOOLS
    names = {t["name"] for t in TOOLS}
    assert names == {"get_state", "goto", "follow_line", "follow_path",
                     "orbit", "patrol", "stop", "wait"}
    for t in TOOLS:
        assert t["parameters"]["type"] == "object" and t["description"]
    for forbidden in ("calib", "corner", "spin", "lookahead", "connect"):
        assert not any(forbidden in n for n in names)


def test_agent_goto_drives_the_real_p2p_to_the_point(app, fleet, no_calib):
    ball = agent_ready(app)
    got = app.agent.dispatch("goto", {"x": 100.0, "y": 80.0})
    assert got.get("ok"), got
    assert app.p2p is not None and app.p2p.get("kind") is None
    assert np.allclose(cm(app, app.p2p["target"]), (100.0, 80.0), atol=0.5)
    supervise(app, ball, "SK-A", px_at(app, 70.0, 55.0), seconds=60)
    st = app.agent.dispatch("get_state", {})
    assert st["status"] == "done" and "arrived" in st["result"], st
    assert np.linalg.norm(np.subtract(st["position_cm"], (100.0, 80.0))) \
        <= app.P2P_ARRIVE_CM + 2.0


def test_agent_follow_line_goes_through_the_line_follower(app, fleet, no_calib):
    agent_ready(app)
    got = app.agent.dispatch("follow_line", {"x1": 30, "y1": 30, "x2": 100,
                                             "y2": 80})
    assert got.get("ok") and got["length_cm"] == pytest.approx(86.0, abs=0.5)
    assert app.p2p["kind"] == "line" and app.p2p.get("path") is None
    assert np.allclose(cm(app, app.p2p["target"]), (30, 30), atol=0.5)


@pytest.mark.parametrize("smooth,kind", [(False, "poly"), (True, "free")])
def test_agent_follow_path_poly_or_smooth(app, fleet, no_calib, smooth, kind):
    agent_ready(app)
    pts = [[30, 30], [100, 30], [100, 80]]
    got = app.agent.dispatch("follow_path", {"points": pts, "smooth": smooth})
    assert got.get("ok"), got
    assert app.p2p["path"]["kind"] == kind
    assert app.p2p["path"]["total"] == pytest.approx(120.0, abs=6.0)


def test_agent_orbit_and_patrol_set_up_the_real_jobs(app, fleet, no_calib):
    agent_ready(app)
    got = app.agent.dispatch("orbit", {"x": 70, "y": 55, "radius": 30,
                                       "direction": "cw"})
    assert got.get("ok"), got
    r = app.orbit_run
    assert r["dir"] == "cw" and r["radius"] == pytest.approx(30, abs=0.5)
    assert np.allclose(r["centre"], (70, 55), atol=0.5)
    got = app.agent.dispatch("patrol", {"points": [[30, 30], [100, 30],
                                                   [100, 80]]})
    assert got.get("ok") and app.orbit_run is None, "patrol must replace orbit"
    assert app.patrol_run["style"] == "loop"
    assert app.retries == 0 and app.job is None or app.job["tries"] == 0
    got = app.agent.dispatch("patrol", {"points": [[30, 30], [100, 30]],
                                        "style": "back_and_forth"})
    assert got.get("ok") and app.patrol_run["style"] == "bounce"


def test_agent_errors_are_readable_and_nothing_moves(app, fleet, no_calib):
    agent_ready(app)
    cases = [("goto", {"x": 500, "y": 20}, "outside the arena"),
             ("goto", {"x": "left", "y": 20}, "must be a number"),
             ("goto", {"x": float("nan"), "y": 20}, "finite"),
             ("patrol", {"points": [[30, 30], [90, 30]]}, "at least 3"),
             ("follow_path", {"points": [[30, 30]]}, "at least 2"),
             ("follow_path", {"points": [[30, 30, 1], [60, 60]]}, "[x, y]"),
             ("orbit", {"x": 10, "y": 55, "radius": 30}, "leaves the arena"),
             ("orbit", {"x": 70, "y": 55, "radius": 5}, "too small"),
             ("orbit", {"x": 70, "y": 55, "radius": 30, "direction": "up"},
              "direction"),
             ("goto", {"y": 3}, "bad arguments"),
             ("fly", {}, "unknown tool")]
    for name, args, words in cases:
        got = app.agent.dispatch(name, args)
        assert "error" in got and words in got["error"], (name, args, got)
        assert app.p2p is None and app.orbit_run is None \
            and app.patrol_run is None, (name, args)


def test_agent_refuses_when_the_rig_is_not_ready(app, fleet):
    got = app.agent.dispatch("goto", {"x": 50, "y": 50})
    assert "error" in got and "no ball" in got["error"]
    app.connect("SK-A")
    app.driving = "SK-A"
    got = app.agent.dispatch("goto", {"x": 50, "y": 50})
    assert "error" in got and "not assigned" in got["error"]


def test_agent_stop_ends_everything_and_nothing_retries(app, fleet, no_calib):
    ball = agent_ready(app)
    app.agent.dispatch("orbit", {"x": 70, "y": 55, "radius": 30})
    got = app.agent.dispatch("stop", {})
    assert got.get("ok")
    pos, _ = supervise(app, ball, "SK-A", px_at(app, 70.0, 55.0), seconds=5)
    assert app.p2p is None and app.orbit_run is None and app.job is None
    assert app.agent.dispatch("get_state", {})["status"] == "stopped"


def test_agent_loop_runs_tools_and_feeds_errors_back(app, fleet, no_calib):
    from llm.stub import StubClient
    agent_ready(app)
    stub = StubClient(script=[[("goto", {"x": 900, "y": 10})],
                              [("goto", {"x": 100, "y": 80})],
                              "On its way."])
    app.agent.client = stub
    log = app.agent.run("go to the bottom right", call=app.agent.dispatch)
    kinds = [k for k, _ in log]
    assert kinds.count("call") == 2 and log[-1] == ("say", "On its way.")
    # The model saw the arena and the error text verbatim.
    first = stub.calls[0]
    assert "arena" in first[0]["content"] and "SK-A" in first[0]["content"]
    tool_msgs = [m for m in stub.calls[1] if m["role"] == "tool"]
    assert "outside the arena" in tool_msgs[0]["content"]
    assert app.p2p is not None


def test_agent_calls_run_on_the_main_loop_not_the_worker(app, fleet, no_calib,
                                                         monkeypatch):
    agent_ready(app)
    ran_on, out = [], {}
    real = app.agent.dispatch

    def watched(name, args):
        ran_on.append(threading.current_thread())
        return real(name, args)

    monkeypatch.setattr(app.agent, "dispatch", watched)

    def worker():
        out["r"] = app.agent.call_on_main("goto", {"x": 90, "y": 60})

    t = threading.Thread(target=worker)
    t.start()
    deadline = time.time() + 3
    while t.is_alive() and time.time() < deadline:
        app.agent.pump()
        time.sleep(0.01)
    t.join(1)
    assert out["r"].get("ok"), out
    assert ran_on == [threading.main_thread()], ran_on


def test_agent_wait_returns_when_the_job_finishes(app, monkeypatch):
    from bench_agent import BenchAgent
    agent = BenchAgent(app)
    states = iter([{"status": "running"}, {"status": "running"},
                   {"status": "done", "result": "arrived"}])
    monkeypatch.setattr(agent, "call_on_main",
                        lambda name, args, timeout=10.0: next(states))
    got = agent._wait({"timeout_s": 5}, poll=0.0)
    assert got["status"] == "done"
    agent.call_on_main = lambda name, args, timeout=10.0: {"status": "running"}
    t0 = time.time()
    got = agent._wait({"timeout_s": 0.2}, poll=0.01)
    assert got["status"] == "running" and time.time() - t0 < 1.0


def test_command_bar_types_and_sends(app, monkeypatch):
    sent = []
    monkeypatch.setattr(app.agent, "ask", lambda c: sent.append(c) or True)
    app.agent.typing = True

    class K:
        def __init__(self, key, uni=""):
            self.key, self.unicode = key, uni

    for ch in "orbit q":
        app.agent.key(K(0, ch), pygame)
    app.agent.key(K(pygame.K_BACKSPACE), pygame)
    app.agent.key(K(0, "x"), pygame)
    assert app.agent.text == "orbit x"
    app.agent.key(K(pygame.K_RETURN), pygame)
    assert sent == ["orbit x"] and not app.agent.typing
    app.agent.typing = True
    app.agent.log = [("you", "orbit x"), ("call", "orbit(...) -> ok"),
                     ("error", "boom"), ("say", "done " * 60)]
    paint(app)
    app.agent.key(K(pygame.K_ESCAPE), pygame)
    assert not app.agent.typing


def test_env_file_is_loaded_without_overwriting(tmp_path, monkeypatch):
    from bench_agent import load_env
    f = tmp_path / "env"
    f.write_text("# c\nexport SPH_TEST_A='one'\nSPH_TEST_B=two\n")
    monkeypatch.setenv("SPH_TEST_B", "kept")
    monkeypatch.delenv("SPH_TEST_A", raising=False)
    load_env(f)
    assert os.environ["SPH_TEST_A"] == "one"
    assert os.environ["SPH_TEST_B"] == "kept"


# -- the spin direction: calibrated, and never locked wrong for good ----------------

def test_a_wrongly_locked_spin_direction_is_overturned(app, fleet, tmp_path):
    """The bench fault: direction locked the wrong way round, the ball rocks
    facing away from the target. Two wrong readings in a row now flip it."""
    app.recorder.log_dir = tmp_path
    app.spin_sign["SK-A"] = 1
    app.spin_sign_known.add("SK-A")                  # locked, and wrong
    gap, ball = run_spin(app, (400.0, 300.0), (700.0, 520.0), facing=200.0,
                         polarity=-1.0)
    assert gap <= app.P2P_ARRIVE_CM + 2.0, list(app.notes)[-3:]
    assert app.spin_sign["SK-A"] == -1 and "SK-A" in app.spin_sign_known


def test_one_wrong_reading_does_not_unlock_a_right_direction(app, fleet,
                                                             no_calib):
    ball = SpinBall(polarity=1.0, ble_name="SK-A", color="red")
    # Facing ~55deg off the target, so the bogus reading below does not also
    # carry the error across 180 (which would restart the probe instead).
    seat_ball(app, ball, (400.0, 300.0), facing=100.0)
    app.turn_mode, app.spin_power = "rate", 50.0
    app.spin_sign["SK-A"] = 1
    app.spin_sign_known.add("SK-A")                  # locked, and right
    said, seen = [], {"bogus": 0}
    real_say = app.say
    app.say = lambda text, tone=None: (said.append(text), real_say(text, tone))

    def jumpy(n):
        got = app.p2p or {}
        probe = got.get("probe")
        # While the first probe is open, report the ball swung hard AGAINST
        # the spin — one bad reading, then the truth again.
        if seen["bogus"] and not seen.get("counted") and app.spin_wrong.get("SK-A"):
            seen["counted"] = True
        if probe is not None and not seen["bogus"]:
            seen["bogus"] = 1
            return (probe[1] - 30.0 * probe[2]) % 360.0
        return ball.facing

    app.arena_heading = jumpy
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))
    supervise(app, ball, "SK-A", (400.0, 300.0), seconds=60)
    assert seen["bogus"], "the bogus reading was never injected"
    assert seen.get("counted"), "the bogus reading never reached the probe"
    assert any("arrived" in t for t in said), said[-3:]
    assert app.spin_sign["SK-A"] == 1
    assert not any("reversed" in t for t in said), said


def test_a_calibrated_ball_starts_with_its_measured_spin_direction(
        app, fleet, no_calib):
    from swarm import ball_calib
    ball_calib.save({"schema": ball_calib.SCHEMA, "name": "SK-A",
                     "homography": ball_calib.fingerprint(app.lab.hom.M),
                     "sign": -1, "turn": {"spin_dir_sign": -1}})
    ball = SpinBall(polarity=-1.0, ble_name="SK-A", color="red")
    seat_ball(app, ball, (400.0, 300.0))
    app.spin_sign.pop("SK-A", None)
    app.spin_sign_known.discard("SK-A")
    app.turn_mode = "rate"
    said = []
    real_say = app.say
    app.say = lambda text, tone=None: (said.append(text), real_say(text, tone))
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))
    supervise(app, ball, "SK-A", (400.0, 300.0), seconds=60)
    assert any("arrived" in t for t in said), said[-3:]
    assert app.spin_sign["SK-A"] == -1
    assert any("spin direction from its calibration" in t for t in said)
    assert not any("reversed" in t for t in said), "it had to learn it anyway"


def test_every_run_is_recorded_to_a_file(app, fleet, no_calib, tmp_path):
    import json
    app.recorder.log_dir = tmp_path / "rec"
    ball = SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, (400.0, 300.0))
    app.turn_mode = "rate"
    app.p2p_pick = True
    app.p2p_click((700.0, 520.0))

    def record(app, ball, i, pos):
        app.recorder.tick()

    supervise(app, ball, "SK-A", (400.0, 300.0), seconds=60, during=record)
    app.recorder.tick()                                # sees the run end
    files = list((tmp_path / "rec").glob("SK-A_*.json"))
    assert len(files) == 1, files
    got = json.loads(files[0].read_text())
    assert "arrived" in got["ended"] and got["kind"] == "p2p"
    rows = got["rows"]
    assert len(rows) > 30
    need = {"facing", "err", "gap_cm", "spin_sign", "sign_known", "spinning",
            "spin_power", "phase"}
    assert need <= set(rows[5]), set(rows[5])
    assert any(r["spinning"] for r in rows) and any(r["phase"] == "go"
                                                    for r in rows)


# -- pursuit p2p and the track recorder ---------------------------------------------

def pursuit_run(app, ball, start_cm, target_cm, seconds=60, during=None):
    seat_ball(app, ball, px_at(app, *start_cm), facing=ball.facing)
    app.turn_mode = "rate"
    app.pursuit_p2p = True
    app.p2p_pick = True

    def click_then(app, ball, i, pos):
        if i == 0:                  # inside the simulated clock
            app.pursuit_click(px_at(app, *target_cm))
        return during(app, ball, i, pos) if during else None

    return supervise(app, ball, "SK-A", px_at(app, *start_cm), seconds=seconds,
                     during=click_then)


def test_pursuit_facing_the_target_follows_without_a_spin(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    ball.facing = ball.want = 100.0                 # 10 degrees off due east
    pos, trace = pursuit_run(app, ball, (30.0, 55.0), (110.0, 55.0))
    assert app.path_stats is not None, [t for t, _ in app.notes]
    assert app.path_stats["kind"] == "pursuit"
    assert app.path_stats["end"] <= app.P2P_ARRIVE_CM + 2.0
    assert not [e for e in ball.log if e[0] == "spin" and e[1] != 0], "it spun"
    assert ("spin", 0) in ball.log and ("stop_raw",) in ball.log, \
        "it did not re-zero the heading before following"
    assert not [st for st, _ in trace if st == "to_start"], "drove a leg first"


def test_pursuit_facing_away_turns_first_then_follows(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    ball.facing = ball.want = 270.0                 # facing the wrong way
    pos, trace = pursuit_run(app, ball, (30.0, 55.0), (110.0, 55.0))
    assert app.path_stats is not None, [t for t, _ in app.notes]
    stages = [st for st, _ in trace if st is not None]
    assert stages[0] == "face_end" and "follow" in stages, stages[:5]
    assert app.path_stats["rms"] < 3.0


def test_pursuit_uses_the_calibrated_steering_direction(app, fleet, no_calib):
    from swarm import ball_calib
    ball_calib.save({"schema": ball_calib.SCHEMA, "name": "SK-A",
                     "homography": ball_calib.fingerprint(app.lab.hom.M),
                     "sign": -1})
    ball = SpinBall(handed=-1.0, ble_name="SK-A", color="red")
    ball.facing = ball.want = 90.0
    pos, trace = pursuit_run(app, ball, (30.0, 55.0), (110.0, 55.0))
    assert app.path_stats is not None, [t for t, _ in app.notes]
    assert not [st for st, _ in trace if st == "check"], "it wiggle-checked"


def test_pursuit_too_short_uses_the_spin_p2p(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    seat_ball(app, ball, px_at(app, 60.0, 55.0), facing=0.0)
    app.pursuit_p2p = True
    app.p2p_pick = True
    app.pursuit_click(px_at(app, 65.0, 55.0))
    assert app.p2p is not None and app.p2p.get("kind") is None, \
        "a 5cm target should be an ordinary p2p"


def test_a_shoved_pursuit_retries_from_where_it_is(app, fleet, no_calib):
    ball = SpinBall(ble_name="SK-A", color="red")
    ball.facing = ball.want = 90.0
    shoved = {}

    def shove(app, ball, i, pos):
        got = app.p2p or {}
        if got.get("stage") == "follow" and not shoved and len(
                got.get("cross") or []) > 90:
            shoved["i"] = i
            return px_at(app, 60.0, 90.0)           # 35cm off the path
        return None

    pos, trace = pursuit_run(app, ball, (30.0, 55.0), (110.0, 55.0),
                             seconds=120, during=shove)
    assert shoved and app.retries >= 1
    assert app.path_stats is not None, [t for t, _ in app.notes]


def test_pursuit_toggle_and_mode_readout(app, fleet):
    app.pursuit_p2p = False
    app.toggle_pursuit()
    assert app.pursuit_p2p and "PURSUIT" in list(app.notes)[-1][0]
    paint(app)
    app.toggle_pursuit()
    assert not app.pursuit_p2p


def test_track_recorder_writes_every_tracked_ball_each_frame(app, fleet,
                                                            tmp_path):
    import json
    app.track_recorder.log_dir = tmp_path
    for name, at in (("SK-A", (400.0, 300.0)), ("SK-B", (600.0, 300.0))):
        app.lab.robots[name] = SpinBall(ble_name=name, color="red")
        app.lab.tracks.assign(name, {"centre": at, "group": [
            {"x": at[0], "y": at[1], "area": 9.0, "peak": 255.0},
            {"x": at[0] + 8, "y": at[1], "area": 30.0, "peak": 255.0}]})
    assert not app.track_recorder.on
    app.track_recorder.tick()
    assert not list(tmp_path.glob("*.jsonl")), "wrote while switched off"
    app.toggle_track_recording()
    assert app.track_recorder.on
    for _ in range(5):
        app.track_recorder.tick()
    paint(app)
    app.toggle_track_recording()
    assert not app.track_recorder.on
    lines = [json.loads(l) for l in
             next(tmp_path.glob("tracks_*.jsonl")).read_text().splitlines()]
    assert lines[0]["meta"] and sorted(lines[0]["tracked"]) == ["SK-A", "SK-B"]
    rows = lines[1:]
    assert len(rows) == 5
    for r in rows:
        assert set(r["balls"]) == {"SK-A", "SK-B"}
        b = r["balls"]["SK-A"]
        for key in ("px", "cm", "facing", "lost", "contended", "connected"):
            assert key in b, key
    app.toggle_track_recording()
    app.close()                              # closing the bench closes the file
    assert not app.track_recorder.on


def test_agent_tools_take_a_ball_and_select_it(app, fleet, no_calib):
    for name, at in (("SK-A", (400.0, 300.0)), ("SK-B", (700.0, 420.0))):
        ball = SpinBall(ble_name=name, color="red")
        app.lab.robots[name] = ball
        app.lab.tracks.assign(name, {"centre": at, "group": [
            {"x": at[0], "y": at[1], "area": 9.0, "peak": 255.0},
            {"x": at[0] + 8, "y": at[1], "area": 30.0, "peak": 255.0}]})
    app.driving = "SK-A"
    app.saved_offset = lambda n: (0.0, "test")
    app._shot = ((0, 0), 1.0)

    got = app.agent.dispatch("orbit", {"x": 70, "y": 55, "radius": 30,
                                       "ball": "SK-B"})
    assert got.get("ok") and got["ball"] == "SK-B"
    assert app.driving == "SK-B" and app.orbit_run["name"] == "SK-B"
    assert app.agent.dispatch("get_state", {})["ball"] == "SK-B"

    got = app.agent.dispatch("goto", {"x": 100, "y": 80, "ball": "SK-A"})
    assert got.get("ok") and app.p2p["name"] == "SK-A"
    assert app.orbit_run is None, "the new job replaces the running one"

    st = app.agent.dispatch("get_state", {})
    assert st["selected"] == "SK-A" and st["status"] == "running"
    assert st["job"] == "goto" and st["ball"] == "SK-A"
    assert set(st["balls"]) == {"SK-A", "SK-B"}
    assert st["balls"]["SK-B"]["position_cm"] and st["balls"]["SK-B"]["assigned"]

    bad = app.agent.dispatch("goto", {"x": 50, "y": 50, "ball": "SK-Z"})
    assert "error" in bad and "not connected" in bad["error"]
    assert app.p2p is not None and app.p2p["name"] == "SK-A", "it kept going"

    assert app.agent.dispatch("stop", {"ball": "SK-B"}).get("ok")
    assert app.p2p is None and app.driving == "SK-B"
    # Stopping one ball is not stopping the bench: SK-A's job is still SK-A's.
    assert app.ball_slot("SK-A")["p2p"] is not None
    prompt = app.agent.prompt(app.agent.dispatch)
    assert "`ball`" in prompt and "run at the same time" in prompt
    assert "NOTHING KEEPS THE BALLS APART" in prompt
