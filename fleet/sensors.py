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
# A tight burst with no delay between reads, used to tell a cache that is being
# REFRESHED from one that is being FABRICATED. See `probe_reads`.
BURST = 12
PER_CALL_CHANGE = 0.9       # above this, the value moves per call, not per tick
MIN_REFRESH_HZ = 4.0        # below this there is nothing to integrate


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
        # How often the value MOVED, per second of wall clock. This is the
        # number that matters for integrating between camera fixes, and it is
        # not the read rate: a cache read at 600kHz that a notification
        # refreshes ten times a second carries ten samples a second of
        # information and 599,990 repeats.
        moved = sum(1 for a, b in zip(values, values[1:]) if repr(a) != repr(b))
        span = sum(times) + (sleep * max(len(times) - 1, 0))
        refresh_hz = round(moved / span, 1) if span > 0 else None
        # A read that comes back faster than the radio can possibly answer is
        # not a read. This toy returned gyroscope samples in 0.0ms — nominally
        # 596kHz — while a drive command to the same robot took 230ms, and the
        # verdict cheerfully recommended integrating those "live gyro rates".
        # They were a cached struct being handed back locally.
        mean_ms = statistics.mean(times) * 1000 if times else None
        cached = bool(mean_ms is not None and mean_ms < LOCAL_CACHE_MS)

        # An instant answer is a local read. That is all it is, and treating it
        # as disqualifying on its own was wrong in a way this project nearly
        # shipped: `spherov2` serves every `get_*` from a cache that its own
        # background streaming handler refreshes, so a healthy gyro on a
        # working link answers in 0.0ms BY DESIGN, and reading it costs no
        # airtime at all, which is the best outcome available rather than the
        # worst.
        #
        # What the check was really reaching for is a value that never came
        # from the ball, and that has a signature: it moves once per CALL
        # rather than once per tick. So ask twice. A burst with no delay pins
        # anything that increments on being looked at; the timed pass above
        # pins anything that moves with the clock. A refreshed cache changes in
        # the second and not in the first, and only a fabricated one does both.
        per_call = None
        if cached and distinct > 1:
            burst = []
            try:
                for _ in range(BURST):
                    burst.append(repr(fn()))
            except Exception:
                burst = []
            if len(burst) > 1:
                per_call = round(sum(1 for a, b in zip(burst, burst[1:])
                                     if a != b) / (len(burst) - 1), 2)
        fabricated = bool(per_call is not None and per_call >= PER_CALL_CHANGE)
        usable = bool(times and error is None and distinct > 1
                      and not fabricated
                      and (not cached
                           or (refresh_hz or 0) >= MIN_REFRESH_HZ))
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
            "refresh_hz": refresh_hz,
            "per_call_change": per_call,
            "fabricated": fabricated,
            # The one field a caller should branch on. Everything above it is
            # the evidence; this is the finding.
            "usable": usable,
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


def probe_fast_writes(api, samples=DRIVE_SAMPLES):
    """The same measurement again, with the acknowledgement turned off.

    `probe_drive_cost` calls the library's setters, so it always measures the
    acked path however `real_handle` happens to be configured. This is the
    other half of the A/B, taken over the same link in the same session,
    because the only comparison worth anything is one where the room, the
    battery and the radio are held still between the two numbers.

    A word on what this can and cannot show. It times the *enqueue*, and a
    fire-and-forget write is expected to come back in microseconds — so a
    dramatic number here is not evidence the robot did anything. It says the
    ack is gone. Whether the ball still obeys is a question only the camera can
    answer, by watching it drive.
    """
    from . import sphero_fast

    writer = sphero_fast.attach(api)
    if writer is None:
        return {"error": "fast writes could not attach to this link"}

    times = []
    for i in range(samples):
        heading = (i * 37) % 360
        t0 = clock()
        try:
            writer.roll(0, heading)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        times.append(clock() - t0)
    return {"writes": len(times),
            "ms_mean": round(statistics.mean(times) * 1000, 3),
            "ms_p90": round(sorted(times)[int(len(times) * 0.9)] * 1000, 3),
            "ms_max": round(max(times) * 1000, 3)}


def compare_write_paths(api, samples=DRIVE_SAMPLES):
    """Acked versus fire-and-forget, back to back on one link.

    This is the number that decides whether `SPHERO_FAST_WRITES` is worth
    switching on, and it is deliberately the *first* thing a hardware session
    should run — every controller gain derived before it would have to be
    derived again afterwards.
    """
    acked = probe_drive_cost(api, samples=samples)
    fast = probe_fast_writes(api, samples=samples)
    out = {"acked": acked, "fast": fast}

    a, f = acked.get("ms_mean"), fast.get("ms_mean")
    # `is not None`, not truthiness: a fire-and-forget write is *supposed* to
    # measure near zero, and a guard that reads that as "no measurement" would
    # drop the result precisely when the change worked best.
    if a is not None and f is not None:
        out["speedup"] = round(a / f, 1) if f > 0 else None
        out["saved_ms"] = round(a - f, 1)
        # The ack is not the only cost. The library's writer thread still
        # pauses `cmd_safe_interval` between packets, so this is the ceiling
        # the command rate moves TOWARD, not one it reaches.
        out["ceiling_hz"] = round(1.0 / sphero_fast_gap(), 1)
        out["note"] = ("enqueue time, not proof of obedience — confirm with the "
                       "camera that the ball still drives before trusting it")
    return out


def sphero_fast_gap():
    from . import sphero_fast
    return sphero_fast.MIN_WRITE_GAP


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
    """The whole §0 report: what answers, how fast, and what streaming costs.

    The sensors are read at EVERY streaming rate, not just once at the start,
    and that is not thoroughness for its own sake. `spherov2`'s `get_*` calls
    do not ask the robot anything — they hand back the last value a streaming
    notification left in a local cache. Asked with streaming off, every one of
    them therefore answers instantly with a number that never changes, which is
    precisely the signature `probe_reads` reports as `cached`.

    So a single pass taken before streaming was ever enabled can only ever
    conclude "camera-only", on any toy, however good its gyro. That is what
    this used to do, and the verdict it produced was an artifact of the
    question rather than a fact about the hardware.
    """
    report = {"reads": probe_reads(api, samples=samples, sleep=sleep),
              "streaming": {}}

    for hz in rates:
        entry = {}
        entry["enable"] = set_streaming(api, hz if hz else 0)
        entry["drive"] = probe_drive_cost(api, samples=drive_samples)
        if hz and entry["enable"].get("ok"):
            # Half the samples: this pass happens once per rate, and the point
            # of it is whether the numbers MOVE, which a shorter run answers
            # just as well as a long one.
            entry["reads"] = probe_reads(api, samples=max(samples // 2, 4),
                                         sleep=sleep)
        report["streaming"][str(hz)] = entry
    set_streaming(api, 0)

    report["write_paths"] = compare_write_paths(api, samples=drive_samples)
    report["verdict"] = verdict(report)
    return report


def verdict(report):
    """Which of the three designs the hardware actually supports.

    Stated as a branch rather than a score because the estimator is built
    differently in each case, and getting this wrong costs a rewrite rather
    than a tuning pass.
    """
    reads = report.get("reads", {})
    streaming = report.get("streaming", {})

    def live(rds, name):
        r = (rds or {}).get(name) or {}
        return bool(r.get("usable"))

    def works(name):
        """Live in ANY pass — with streaming off, or at a rate that enabled.

        Any pass rather than the first one, because the first one is taken with
        streaming off and a cached getter cannot answer then. Which pass it was
        is reported, since "works, but only while streaming at 20Hz" is a
        different engineering position from "works".
        """
        if live(reads, name):
            return "off"
        for hz, entry in streaming.items():
            if hz != "0" and live(entry.get("reads"), name):
                return f"{hz}Hz"
        return None

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
    at = works("get_gyroscope")
    if at:
        return {"branch": "full", "streaming": at,
                "why": "gyro rates are live; integrate them between camera fixes"
                       + ("" if at == "off" else
                          f" — but only with streaming on at {at}, which has to "
                          "stay on for the whole session")}
    at = works("get_heading")
    if at:
        return {"branch": "heading-only", "streaming": at,
                "why": "no usable gyro, but get_heading changes — integrate its "
                       "deltas instead. Same architecture, slightly noisier"
                       + ("" if at == "off" else f" (needs streaming at {at})")}
    # Nothing answered in any pass. If streaming never even turned on, that is
    # the finding — not "this toy has no sensors", which is what it looks like.
    failed = [f"{hz}Hz: {(e.get('enable') or {}).get('error')}"
              for hz, e in streaming.items()
              if hz != "0" and not (e.get("enable") or {}).get("ok")]
    if failed and len(failed) == max(len(streaming) - 1, 0):
        return {"branch": "camera-only",
                "why": "streaming would not turn on at any rate, so the sensor "
                       f"getters have nothing to cache — {failed[0]}"}
    if any((reads.get(n) or {}).get("fabricated") for n in READS):
        return {"branch": "camera-only",
                "why": "the sensor calls change on every call rather than with "
                       "the clock, so the numbers are being made up locally "
                       "and never came from the ball. Predict from the "
                       "commanded heading and let the camera correct it"}
    slow = [n for n in READS
            if (reads.get(n) or {}).get("changing")
            and not (reads.get(n) or {}).get("usable")
            and (reads.get(n) or {}).get("cached")]
    if slow:
        hz = (reads.get(slow[0]) or {}).get("refresh_hz")
        return {"branch": "camera-only",
                "why": f"the sensor cache refreshes at only {hz}Hz — too slow "
                       "to integrate between camera fixes. Predict from the "
                       "commanded heading and let the camera correct it"}
    return {"branch": "camera-only",
            "why": "nothing on this toy reports orientation, so predict from "
                   "the commanded heading and let the camera correct it"}
