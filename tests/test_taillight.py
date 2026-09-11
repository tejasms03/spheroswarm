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
