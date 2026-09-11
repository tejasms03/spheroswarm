"""The shutter dial: what actually reaches the camera, and when.

`vision/shutter.py` is the only thing on macOS that can move this camera's
exposure — AVFoundation drops the write — so the failures worth pinning are the
ones that look identical from behind a slider: a value that was never sent, a
value sent while the camera is still metering for itself and quietly ignores
it, and a dial with no tool underneath it at all.
"""

import pytest

from vision import shutter


@pytest.fixture
def dial(monkeypatch, tmp_path):
    """A Dial with uvc-util replaced by a log of what it was asked to do."""
    calls = []

    def fake_mode(binary, index, mode):
        calls.append(("mode", index, mode))
        return True, "ok"

    def fake_exposure(binary, index, value):
        calls.append(("exposure", index, value))
        return True, "ok"

    monkeypatch.setattr(shutter, "set_mode", fake_mode)
    monkeypatch.setattr(shutter, "set_exposure", fake_exposure)
    monkeypatch.setattr(shutter.config, "load", lambda *a, **k: {})
    d = shutter.Dial(index=2, value=100)
    d.binary = "/fake/uvc-util"
    d.calls = calls
    return d


def test_setting_a_value_does_not_reach_the_camera_until_it_is_pushed(dial):
    """Each write is a subprocess and a dragged slider emits one event per
    pixel. `set` records; `push` sends."""
    dial.calls.clear()
    dial.set(400)
    assert dial.calls == [], "a set must not spawn a process"
    dial.push(force=True)
    assert ("exposure", 2, 400) in dial.calls


def test_a_drag_is_rate_limited_to_one_write(dial):
    dial.push(force=True)
    dial.calls.clear()
    for v in range(100, 140):
        dial.set(v)
        dial.push()
    sent = [c for c in dial.calls if c[0] == "exposure"]
    assert len(sent) <= 1, f"{len(sent)} writes from one drag"
    dial.push(force=True)
    assert dial.value == 139, "and the last value still lands"


def test_manual_mode_is_taken_before_the_first_write(dial):
    """In aperture priority the camera owns the shutter and drops these writes
    without complaint — the value reads back unchanged, which is
    indistinguishable from a dead dial."""
    dial.calls.clear()
    dial.set(250)
    dial.push(force=True)
    assert dial.calls[0] == ("mode", 2, shutter.MANUAL)


def test_handing_the_shutter_back_means_the_next_write_retakes_it(dial):
    dial.push(force=True)
    dial.auto()
    assert not dial.manual
    dial.calls.clear()
    dial.set(300)
    dial.push(force=True)
    assert ("mode", 2, shutter.MANUAL) in dial.calls, "or the write is dropped"


def test_a_refused_manual_mode_abandons_the_write(dial, monkeypatch):
    """Sending an exposure the camera will ignore, and reporting success, is
    the failure this whole module exists to make impossible."""
    monkeypatch.setattr(shutter, "set_mode",
                        lambda b, i, m: (False, "device busy"))
    dial.manual = False
    dial.calls.clear()
    dial.set(300)
    ok, msg = dial.push(force=True)
    assert not ok and "manual" in msg
    assert not [c for c in dial.calls if c[0] == "exposure"]


def test_no_tool_says_so_rather_than_going_quiet(monkeypatch):
    monkeypatch.setattr(shutter, "tool", lambda: None)
    monkeypatch.setattr(shutter.config, "load", lambda *a, **k: {})
    d = shutter.Dial()
    assert not d.available
    ok, msg = d.push(force=True)
    assert not ok and "uvc-util" in msg


def test_values_are_clamped_to_what_the_camera_accepts(dial):
    dial.set(99999)
    assert dial.value == shutter.EXP_MAX
    dial.set(-5)
    assert dial.value == shutter.EXP_MIN


def test_the_slider_mapping_round_trips(dial):
    for v in (3, 50, 156, 500, 2047):
        assert abs(shutter.slider_to_us(shutter.us_to_slider(v)) - v) <= v * 0.1
