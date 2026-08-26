"""The taillight lab: the measurement, the lighting map, and what it refuses.

`Lab` is deliberately pygame-free so the vision can be tested without a window
— `App` is only the chrome around it. Everything here runs on synthetic frames
and never opens a camera, a Bluetooth link, or a calibration file for writing.
"""

import math

import cv2
import numpy as np
import pygame
import pytest

from taillight import Lab, _wrap
from vision import config
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
    """The synthetic source has no focus. Turning the knob must report that
    it did nothing — a control that silently does nothing is worse than one
    that is absent, because a person will keep turning it."""
    app.set_focus(120)
    assert any("no focus control" in t or "ignored" in t
               for t, _ in app.notes), list(app.notes)


def test_one_hue_moves_both_the_detector_and_the_led(app):
    """`vision/config.py` keeps the colour a ball is told to glow and the hue
    the tracker hunts in ONE table, because two tables is how a robot ends up
    lit one colour and looked for as another. A lab that split them would
    reintroduce that bug on the bench people use to diagnose it."""
    sent = []
    app.lab.robot = type("Fake", (), {"set_led": lambda _s, rgb: sent.append(rgb),
                                      "set_back_led": lambda _s, v: None})()
    app.set_hue(140)
    assert app.lab.detector.colors[app.lab.color]["hue"] == 140
    assert sent, "the LED must follow the same slider"
    assert sent[-1] == config.led_rgb(140, value=app.bright)


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
