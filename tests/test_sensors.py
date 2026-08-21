"""The §0 sensor probe: what the ball reports, and what asking for it costs.

Driven against fake APIs rather than hardware, because what is being tested is
the *judgement* — the probe has to tell a working sensor from a cached zero,
and an affordable stream from one that eats the command channel. Both of those
are decisions, and both are wrong in ways a real ball would not announce.
"""

import pytest

from fleet.sensors import probe, probe_drive_cost, probe_reads, verdict

# Enough samples to judge, few enough that the suite is not paying for sleeps.
FAST = {"samples": 6, "sleep": 0.0, "drive_samples": 6}


class FakeApi:
    """A toy that answers some calls and not others."""

    def __init__(self, gyro=True, heading=True, stuck=(), missing=(),
                 write_ms=0.0, stream_penalty=1.0):
        self.gyro, self.heading = gyro, heading
        self.stuck, self.missing = set(stuck), set(missing)
        self.n = 0
        self.write_ms = write_ms
        self.stream_penalty = stream_penalty
        self.streaming_hz = 0
        for name in missing:
            setattr(self, name, None)

    def _tick(self):
        self.n += 1
        return self.n

    def get_heading(self):
        if not self.heading:
            raise RuntimeError("not supported on this toy")
        return 0.0 if "get_heading" in self.stuck else (self._tick() * 3) % 360

    def get_gyroscope(self):
        if not self.gyro:
            raise RuntimeError("not supported on this toy")
        return {"x": 0.0, "y": 0.0, "z": float(self._tick())}

    def get_orientation(self):
        return {"pitch": 0.0, "roll": 0.0, "yaw": float(self._tick())}

    def get_acceleration(self):
        return {"x": 0.0, "y": 0.0, "z": 1.0}          # never changes

    def get_velocity(self):
        return {"x": 0.0, "y": 0.0}

    def get_location(self):
        return {"x": 0.0, "y": 0.0}

    def set_heading(self, h):
        self._sleep()

    def set_speed(self, v):
        pass

    def _sleep(self):
        import time
        ms = self.write_ms * (self.stream_penalty if self.streaming_hz else 1.0)
        if ms:
            time.sleep(ms / 1000.0)


def test_a_sensor_that_answers_and_changes_is_available():
    r = probe_reads(FakeApi(), samples=6, sleep=0)
    assert r["get_gyroscope"]["available"]
    assert r["get_gyroscope"]["changing"]
    assert r["get_gyroscope"]["hz_ceiling"] > 0


def test_a_sensor_that_never_changes_is_reported_as_not_changing():
    """A cached zero answers instantly and forever. Integrating one is a lie."""
    r = probe_reads(FakeApi(), samples=6, sleep=0)
    assert r["get_acceleration"]["available"]
    assert not r["get_acceleration"]["changing"]


def test_a_sensor_that_raises_is_reported_unavailable():
    r = probe_reads(FakeApi(gyro=False), samples=6, sleep=0)
    assert not r["get_gyroscope"]["available"]
    assert "not supported" in r["get_gyroscope"]["error"]


def test_a_call_the_api_does_not_have_at_all():
    api = FakeApi()
    del api.__class__.get_location
    try:
        r = probe_reads(api, samples=4, sleep=0)
        assert not r["get_location"]["available"]
    finally:
        FakeApi.get_location = lambda self: {"x": 0.0, "y": 0.0}


def test_drive_cost_is_measured_in_milliseconds():
    d = probe_drive_cost(FakeApi(write_ms=2.0), samples=5)
    assert d["ms_mean"] >= 1.5
    assert d["ms_max"] >= d["ms_mean"]


# -- the verdict, which is the part that decides the design ------------------

class RadioApi(FakeApi):
    """A fake whose reads take radio time, as a real sensor's would.

    `FakeApi` answers in-process and instantly, which the probe now correctly
    calls a cache. Representing a LIVE sensor therefore needs a fake that is
    slow — which is the whole point of the check, so the fixture has to respect
    it rather than being exempted from it.
    """

    DELAY_S = 0.004

    def _radio(self):
        import time
        time.sleep(self.DELAY_S)

    def get_heading(self):
        self._radio()
        return super().get_heading()

    def get_gyroscope(self):
        self._radio()
        return super().get_gyroscope()


def test_a_live_gyro_selects_the_full_design():
    v = probe(RadioApi(), rates=(0,), **FAST)["verdict"]
    assert v["branch"] == "full"


def test_no_gyro_but_a_working_heading_selects_the_heading_only_design():
    v = probe(RadioApi(gyro=False), rates=(0,), **FAST)["verdict"]
    assert v["branch"] == "heading-only"


def test_nothing_usable_falls_back_to_camera_only():
    v = probe(FakeApi(gyro=False, heading=False), rates=(0,), **FAST)["verdict"]
    assert v["branch"] == "camera-only"


def test_a_stuck_heading_is_not_mistaken_for_a_working_one():
    """It answers, quickly, every time — with the same number."""
    v = probe(RadioApi(gyro=False, stuck=("get_heading",)), rates=(0,), **FAST)["verdict"]
    assert v["branch"] == "camera-only"


def test_streaming_that_costs_too_much_airtime_is_refused():
    """Bandwidth is the binding constraint; a heading is not worth the fleet.

    Even with a perfectly good gyro, a stream that triples the time a drive
    command takes to write has to be turned down — six robots share one
    adapter, and the deadband in `real_handle.py` exists to protect exactly
    this.
    """
    api = FakeApi(write_ms=2.0, stream_penalty=4.0)
    real_set = None

    import fleet.sensors as sensors
    real_set = sensors.set_streaming

    def fake_set(a, hz):
        a.streaming_hz = hz
        return {"ok": True}

    sensors.set_streaming = fake_set
    try:
        report = probe(api, rates=(0, 10), **FAST)
    finally:
        sensors.set_streaming = real_set
    assert report["verdict"]["branch"] == "camera-only"
    assert "slower" in report["verdict"]["why"]


def test_set_streaming_never_raises_on_a_toy_without_it():
    from fleet.sensors import set_streaming
    r = set_streaming(FakeApi(), 10)
    assert r["ok"] is False and "set_data_streaming" in r["error"]


def test_the_probe_leaves_streaming_off():
    """It is a probe, not a configuration change."""
    api = FakeApi()
    import fleet.sensors as sensors
    seen = []
    real = sensors.set_streaming
    sensors.set_streaming = lambda a, hz: (seen.append(hz), {"ok": True})[1]
    try:
        probe(api, rates=(0, 10, 20), **FAST)
    finally:
        sensors.set_streaming = real
    assert seen[-1] == 0, f"probe left streaming at {seen[-1]}Hz"


def test_an_instant_answer_is_a_cache_not_a_sensor():
    """From a real probe: gyro samples at 596kHz while a drive write took 230ms.

    Nothing reached the robot. The verdict recommended integrating those "live
    gyro rates", which would have fed an estimator a number that never came
    from the ball — the worst kind of wrong, because it is smooth and
    plausible and completely unrelated to reality.
    """
    api = FakeApi()                 # answers instantly, in-process
    r = probe_reads(api, samples=6, sleep=0)
    assert r["get_gyroscope"]["cached"], r["get_gyroscope"]
    v = probe(api, rates=(0,), **FAST)["verdict"]
    assert v["branch"] == "camera-only", v
    assert "local cache" in v["why"]


def test_a_sensor_that_takes_radio_time_is_believed():
    import time

    class Slow(FakeApi):
        def get_gyroscope(self):
            time.sleep(0.004)       # a plausible BLE round trip
            return {"x": 0.0, "y": 0.0, "z": float(self._tick())}

    r = probe_reads(Slow(), samples=5, sleep=0)
    assert not r["get_gyroscope"]["cached"], r["get_gyroscope"]
