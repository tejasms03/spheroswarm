"""What does this Sphero actually tell us, and what does asking cost?

`spherov2` presents the same `SpheroEduAPI` surface for every toy it supports,
so `hasattr(api, "get_gyroscope")` is true regardless of whether the ball on
the floor can answer. The SPRK+ speaks a legacy protocol, and the honest way to
find out what it supports is to ask it and see.

Two questions, and the second matters as much as the first:

  WHAT     which of the sensor calls return finite, changing numbers, and how
           fast they can be read.
  WHAT FOR Bluetooth airtime is the binding constraint on this project — six
           robots share one adapter, and `real_handle.py` deadbands its writes
           specifically to protect it. A heading estimate that halves the
           command rate is not worth having, so the probe measures the drive
           round trip with streaming off and then on, and reports the damage.

Runs against the same connector `SpheroRobot` uses, so a test can drive it with
no radio present.
"""

import statistics
import time

# perf_counter, not time(): a BLE read is milliseconds and a cached one is
# nanoseconds, and time() cannot tell the second from zero on every platform.
clock = time.perf_counter

READS = ("get_heading", "get_orientation", "get_gyroscope", "get_acceleration",
         "get_velocity", "get_location")
# Below this a call did not go near the radio. Set well under any plausible
# BLE round trip and well over any local dictionary lookup.
LOCAL_CACHE_MS = 1.5
SAMPLES = 25
DRIVE_SAMPLES = 20


def _finite(value):
    """Did this call return something usable, and is it a number or a bag of them?"""
    if value is None:
        return False, None
    if isinstance(value, (int, float)):
        return value == value and abs(value) != float("inf"), float(value)
    if isinstance(value, dict):
        vals = [v for v in value.values() if isinstance(v, (int, float))]
        return bool(vals) and all(v == v for v in vals), dict(value)
    if isinstance(value, (list, tuple)):
        vals = [v for v in value if isinstance(v, (int, float))]
        return bool(vals) and all(v == v for v in vals), list(value)
    return True, repr(value)[:60]


def probe_reads(api, samples=SAMPLES, sleep=0.02):
    """Call each sensor repeatedly. Report rate, and whether it ever changes.

    A call that answers instantly with the same number every time is not
    working — it is a cached zero, and a heading estimator fed one would
    integrate a lie very smoothly.
    """
    out = {}
    for name in READS:
        fn = getattr(api, name, None)
        if fn is None:
            out[name] = {"available": False, "error": "not on this API"}
            continue
        values, times, error = [], [], None
        for _ in range(samples):
            t0 = clock()
            try:
                v = fn()
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                break
            times.append(clock() - t0)
            ok, parsed = _finite(v)
            if not ok:
                error = error or f"returned {v!r}"
            values.append(parsed)
            if sleep:
                time.sleep(sleep)

        if error and not times:
            out[name] = {"available": False, "error": error}
            continue
        distinct = len({repr(v) for v in values})
        # A read that comes back faster than the radio can possibly answer is
        # not a read. This toy returned gyroscope samples in 0.0ms — nominally
        # 596kHz — while a drive command to the same robot took 230ms, and the
        # verdict cheerfully recommended integrating those "live gyro rates".
        # They were a cached struct being handed back locally.
        mean_ms = statistics.mean(times) * 1000 if times else None
        cached = bool(mean_ms is not None and mean_ms < LOCAL_CACHE_MS)
        out[name] = {
            "available": bool(times) and error is None,
            "reads": len(times),
            "ms_mean": round(statistics.mean(times) * 1000, 1) if times else None,
            "ms_max": round(max(times) * 1000, 1) if times else None,
            # A read too fast to time is not a read that failed. Floored at a
            # microsecond so the ceiling is a number rather than a None that
            # every caller then has to special-case.
            "hz_ceiling": round(1.0 / max(statistics.mean(times), 1e-6), 1)
                          if times else None,
            "distinct_values": distinct,
            # One value for twenty-five reads means it is not really reading.
            "changing": distinct > 1,
            "cached": cached,
            "sample": values[-1] if values else None,
            "error": error,
        }
    return out


def probe_drive_cost(api, samples=DRIVE_SAMPLES):
    """How long does a drive command take to write? This is the scarce thing."""
    times = []
    for i in range(samples):
        heading = (i * 37) % 360
        t0 = clock()
        try:
            api.set_heading(heading)
            api.set_speed(0)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        times.append(clock() - t0)
    return {"writes": len(times),
            "ms_mean": round(statistics.mean(times) * 1000, 1),
            "ms_p90": round(sorted(times)[int(len(times) * 0.9)] * 1000, 1),
            "ms_max": round(max(times) * 1000, 1)}


def set_streaming(api, hz):
    """Turn sensor streaming on at a rate, or off with hz=0. Never raises."""
    toy = getattr(api, "_SpheroEduAPI__toy", None) or getattr(api, "toy", None)
    fn = getattr(toy, "set_data_streaming", None) if toy else None
    if fn is None:
        return {"ok": False, "error": "this toy has no set_data_streaming"}
    try:
        if hz <= 0:
            fn(interval=0, packet_count=0, sensor_masks=[])
        else:
            fn(interval=int(round(1000.0 / hz)), packet_count=0,
               sensor_masks=getattr(toy, "sensors", None) or [])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def probe(api, rates=(0, 10, 20), samples=SAMPLES, sleep=0.02,
          drive_samples=DRIVE_SAMPLES):
    """The whole §0 report: what answers, how fast, and what streaming costs."""
    report = {"reads": probe_reads(api, samples=samples, sleep=sleep),
              "streaming": {}}

    for hz in rates:
        entry = {}
        if hz:
            entry["enable"] = set_streaming(api, hz)
        else:
            entry["enable"] = set_streaming(api, 0)
        entry["drive"] = probe_drive_cost(api, samples=drive_samples)
        report["streaming"][str(hz)] = entry
    set_streaming(api, 0)

    report["verdict"] = verdict(report)
    return report


def verdict(report):
    """Which of the three designs the hardware actually supports.

    Stated as a branch rather than a score because the estimator is built
    differently in each case, and getting this wrong costs a rewrite rather
    than a tuning pass.
    """
    reads = report.get("reads", {})
    def works(name):
        r = reads.get(name) or {}
        return bool(r.get("available") and r.get("changing")
                    and not r.get("cached"))

    base = (report.get("streaming", {}).get("0", {}).get("drive") or {})
    base_ms = base.get("ms_mean")
    cost = None
    for hz in ("10", "20"):
        d = (report.get("streaming", {}).get(hz, {}).get("drive") or {})
        if d.get("ms_mean") and base_ms:
            cost = max(cost or 0.0, d["ms_mean"] / base_ms)

    if cost and cost > 2.0:
        return {"branch": "camera-only",
                "why": f"streaming makes drive commands {cost:.1f}x slower — "
                       "bandwidth is the binding constraint and this spends it"}
    if works("get_gyroscope"):
        return {"branch": "full",
                "why": "gyro rates are live; integrate them between camera fixes"}
    if works("get_heading"):
        return {"branch": "heading-only",
                "why": "no usable gyro, but get_heading changes — integrate its "
                       "deltas instead. Same architecture, slightly noisier"}
    if any((reads.get(n) or {}).get("cached") for n in READS):
        return {"branch": "camera-only",
                "why": "the sensor calls answer instantly, which means they are "
                       "returning a local cache rather than asking the robot. "
                       "Predict from the commanded heading and let the camera "
                       "correct it"}
    return {"branch": "camera-only",
            "why": "nothing on this toy reports orientation, so predict from "
                   "the commanded heading and let the camera correct it"}
