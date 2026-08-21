# The calibration bench — what exists, what we measured, what we concluded

A handoff for `calib.py` and the work around it. Paste into a fresh chat to
bring it up to speed. Companion to `HANDOFF.md`, which covers the swarm/LLM
stack this sits underneath.

---

## READ THIS FIRST, IF YOU ARE A NEW SESSION

**Do not start writing code. Start by reading this document and giving me the
plan.**

Specifically, before touching anything:

1. Read this whole file, then `HANDOFF.md` for the layers underneath.
2. Tell me back, in your own words: what state the project is in, what the
   last session concluded, and what you think the next step is.
3. Give me section 7's plan as *you* would order it — say plainly if you would
   sequence it differently and why.
4. Confirm the environment before trusting it: run the test suite, check
   whether the arena in `workspace.json` still matches `calib/homography.json`,
   and check whether `roster.json` is clean. Report what you find.
5. **Then wait.** Do not begin step 1 of the plan until I say so.

Two things that will otherwise waste your time and mine:

- **Most of the pain in this project has been measurement, not control.**
  Repeatedly, a controller looked broken and was fine, because it was being
  fed numbers from a camera that was not watching the robot. Suspect the
  measurement first. Section 5 is a list of times that was the answer.
- **Verify before you assert.** Nearly every conclusion in this document came
  from running something, not from reasoning about it — and several confident
  guesses along the way were wrong. If you catch yourself about to explain why
  something behaves a certain way, go and measure it instead.

```bash
cd ~/spheroswarm
~/miniconda3/bin/python3 calib.py --camera 0        # the bench
~/miniconda3/bin/python3 calib.py --dry             # rehearse with sim robots
SDL_VIDEODRIVER=dummy ~/miniconda3/bin/python3 -m pytest -q   # 995 tests, ~100s
```

**Restart the app after any code change.** Python loads `calib.py` once; a
running bench does not pick up edits, and most of a session was spent chasing
behaviour that had already been fixed on disk.

---

## 1. The rig

`calib.py` — a pygame bench with a robot dock and three tabs.

**COLOUR** — camera + live mask, a hue wheel, HSV sliders, LED brightness.
`auto-tune` drives the ball's own LED off then on and learns the signature from
the difference. `check hues` reports whether the palette is separable in this
room. `optimise hues` re-picks all six for the room the camera is looking at.
`set arena` picks the workspace out of the camera in four clicks. `check pos`
measures where the tracker thinks a robot is versus where it really is.

**MOTION** — the measurement battery. `run full` (~2.7 min), `run quick`,
`drift 5min`, `sensors`. Two pace controls: `top speed` (hard cap, 6–18 cm/s)
and `edge margin`.

**DRIVE** — click a point, or two for a line, or one for a circle. Sliders for
speed, circle radius and the arrival radius. `auto radius` sizes the last from
the robot's own measurement.

New modules: `fleet/characterize.py` (the battery), `fleet/safety.py` (the
speed field), `fleet/sensors.py` (the sensor probe), `swarm/pd.py` (PD, paths,
turn-and-go), `vision/palette.py` (hue selection), `ui/theme.py` (widgets).

---

## 2. What the hardware actually is — measured, not assumed

**The arena is 138.8 x 110.8 cm.** Smaller than the 200x200 everything was
written against, and small enough to change design decisions rather than just
parameters.

**A drive command takes ~230 ms.** Measured by `sensors`. This is the single
largest number in the control problem — bigger than the motor lag and the
camera put together — and it is a property of the protocol, not the robot.

**No usable onboard sensors.** `get_gyroscope` and friends answer in 0.0 ms at
a nominal 596 kHz. A read that returns faster than the radio can possibly
answer never went near the robot: it is a cached struct. The correct design
branch is **camera-only**, and the probe now says so.

**The colour tracking is the weak link.** Sessions repeatedly ran with 5 blobs
detected for 2 connected robots. Every phantom is something the tracker can
lock onto, and when it does, every safety mechanism downstream is guarding a
ghost while the real ball drives free.

---

## 3. The control-architecture conclusion

**We were using a teaching API as a real-time control interface, and that is
the root of most of the pain.**

`spherov2.sphero_edu.SpheroEduAPI` is designed for scripted sequences —
`roll(90, 200, 2)` meaning "go that way for two seconds". Underneath:

- **Every command blocks on an acknowledgement.** `Toy._execute` puts a packet
  on the queue and then waits on `_wait_packet`. That wait is the 230 ms.
- **It runs its own background thread** re-sending the speed every 0.8 s under
  a lock. Writing without that lock lets the two interleave on a slow link, and
  the loser raises `TimeoutError` from inside the library's thread. We saw
  exactly that.
- **`roll(h, v, duration)` is not a combined command.** It sets the speed,
  sleeps, then calls `stop_roll()`. Passing `duration=0` starts and stops the
  robot in the same breath. Using it as a "one packet instead of two"
  optimisation produced *no control at all*.
- **`set_heading` already carries the speed** — it calls
  `roll_start(heading, speed)`. Calling both setters sends two packets
  describing one intent.

The protocol itself is the classic Sphero API v1.2. Its SOP2 header byte
encodes per-message options, and **bit 0 is "answer requested"** — the sequence
number is ignored by the robot when that bit is clear. `spherov2` hardcodes
`0xFF`, so every command asks for an ack and blocks. **Fire-and-forget is
available in the protocol and simply not exposed by the library.**

**Is there a better C++ library? No — and the language is not the bottleneck.**
The options are Python (`spherov2`, `SpheroPy`), Go (`gobot`), or writing
directly against native BLE. A C++ client making the same blocking ack call
would be equally slow. The 230 ms is a protocol round trip, not interpreter
overhead. Fixing it means dropping the ack, not changing language.

### What is already done about it

`_write` now sends **one packet per command** (updates the stored speed, then
one `set_heading`), takes the library's own `__updating` lock so it cannot
interleave with the keepalive thread, and paces writes to the measured round
trip rather than queueing behind them.

### The ack bypass — built, unverified

`fleet/sphero_fast.py` builds v1.2 packets with the answer bit clear and puts
them straight on the toy's write queue. **Off by default** — set
`SPHERO_FAST_WRITES=1` or pass `fast_writes=True`. No physical ball has
received one.

The framing is pinned byte-for-byte against spherov2's own builder on the
acked case. That is the only evidence available without hardware, and it is
worth being clear about its limit: it shows the packet is well formed, not
that the robot obeys it. **From this side of the radio, "the ball obeys
instantly" and "the ball ignores us entirely" look identical** — both return
in microseconds. Only the camera can tell them apart.

Two hazards found in the wiring, both of which would have been blamed on the
controller:

- **spherov2's keepalive re-sends `roll_start(__heading, __speed)` every
  0.8s** whenever the speed is non-zero. Bypassing `set_heading` never updates
  that cache, so the library would have re-aimed the ball at a *stale heading*
  about once a second. The fast path keeps both values in sync, which turns
  the keepalive into a safety net: a fast path that silently stops being
  delivered degrades to driving correctly at 1.25Hz rather than to a ball that
  ignores us.
- **Pacing was derived from the measured round trip**, which now measures an
  enqueue. Without a floor the loop queues commands faster than the library's
  writer thread drains them — a backlog of stale intent, which latest-wins
  upstream cannot help once the packets are already queued. The ack was
  standing in for `cmd_safe_interval`, so the ceiling moves to **~16Hz, not
  infinity**. The 60ms writer pause remains and is now the binding constraint.

`sensors.compare_write_paths(api)` runs the A/B over one link. **Run it first
in the next hardware session**, before the battery: every gain derived on the
acked protocol would have to be derived again afterwards.

---

## 4. The measurement battery

Seven stages, each a state machine stepped once a frame so the window keeps
drawing. All of them run unchanged against a `SimRobot`, which is what makes
the rig testable with no hardware.

| stage | measures |
|---|---|
| tracking check | is the camera following THIS robot at all |
| position noise | the tracker's own error bar |
| heading offset | the ball's aim frame vs the camera's |
| speed map | speed byte to cm/s |
| step response | dead time and motor lag |
| brake test | how far it coasts — the speed limit, directly |
| loop latency | command issued to camera noticing |

Plus `drift 5min`, which reports **degrees per root-minute** — a wandering
heading is a random walk, and a slope fit through one finds nothing however
badly it wanders. Validated: 0/1/3/8/20 deg-per-min inputs read back as
2.0/1.9/3.1/7.7/17.5, with an explicit ~2 noise floor.

Accuracy in the real arena, capped at 18 cm/s: **r2 >= 0.998** on the speed
map, tau within ~10%, and the ball never closer than 12 cm to a wall.

---

## 5. Every bug found, and why it mattered

Ordered by how badly it misled us.

**a) The battery drove the ball out of the arena, repeatedly.** Legs were
specified as *durations*. 2.5 s at byte 255 is 150 cm; there is not 150 cm in
front of a ball in the middle of a 111 cm room. Legs are now fitted to the
floor actually in front, measured by marching along the course.

**b) A frozen tracker looked exactly like a stationary robot.** The tracker
locked onto a phantom and reported the same position forever. Every guard —
speed field, edge margin, runway planning — reads that position, so all of them
were faithfully protecting a ghost. The only tell is the contradiction:
commanding motion and seeing none. There are now three layers: the noise probe
refuses an impossibly quiet reading (0.017 cm sigma is not a measurement), a
per-stage watchdog stops after 25 cm of commanded travel with nothing seen, and
a pre-flight nudge refuses to start at all — it catches a dead tracker in 0.7 s
and 9 cm.

**c) `step_path` returned on a lost fix without stopping the robot.** A Sphero
holds its last speed command indefinitely, so "stop updating" is not "stop".
The ball kept driving on its last command for as long as the fix stayed lost.
This is why point-to-point moves ended as the ball circling the room.

**d) One unbounded extrapolation poisoned an entire calibration.** The speed
map reported a ceiling of **288 cm/s** — a slope from bytes 18-76 extended to
255. Because everything derives from it, the "deadband" of 20.4 cm/s was just
that ceiling scaled back down (18/255 x 288), and it sat *above* the speed cap,
so the ball could not legally be commanded to move. The fit now refuses to
publish a top speed it cannot stand behind and says so.

**e) Velocity was computed by differencing camera positions.** At 30 fps with
1 cm of position noise that is **41 cm/s** of velocity noise, and the
derivative gain multiplies it straight into the motors. The Kalman filter in
`vision/track.py` had been computing a proper estimate (0.8 cm/s) all along and
was never asked for it. Command jitter fell 26x.

**f) `roll(h, v, 0)` stopped the robot every command.** Mine. See section 3.

**g) The sensor probe's verdict was wrong.** It checked whether a value
*changed*, never whether the read was plausibly a radio round trip, and
recommended integrating "live gyro rates" that were a local cache.

**h) The LED colour and the detector hue were two independent tables.** Edit
one and a ball glows one colour while being hunted as another, with each table
internally consistent and nothing complaining. Now derived from one.

**i) `set_corners` forced a square arena.** Which is how the homography said
200x200 while `workspace.json` said 240x180. Both are now written from the same
four clicks and the same two numbers.

**j) A shadowed layout cursor.** `for d, y in (...)` rebound the `y` holding
the vertical position, so the whole MOTION tab drew stacked in the corner. And
`draw_motion` never drew its sliders at all — they existed, took clicks, and
were invisible.

**k) Tab buttons changed `self.tab` without rebuilding the layout.** Clicking
MOTION crashed on the first attribute the other branch had not created.

**l) Duplicate `ble_name` was never validated.** Three roster rows pointed at
one ball. Both open a link; one wins and the other sits in "connecting" forever.
Repair also had to be possible *while* the file was broken, which needed
comparing error *categories* rather than whole message strings.

**m) `calib/homography.json` was owned by root** from an old `sudo` run, and an
unguarded `hom.save()` took the whole app down from inside the fourth corner
click. Calibration writes are now atomic via rename, which needs only directory
permission — so the file was replaceable all along.

**n) Several tests silently depended on live state.** Changing the real
workspace to 138.8 x 110.8 broke twelve of them.

---

## 6. Things that are true and worth not rediscovering

- **A speed cap below the deadband makes the ball unable to move.** Driving and
  measuring want different speeds: 6 cm/s is right for driving, and a speed map
  confined to bytes 18-26 has no curve left to fit. Two sliders, two jobs.
- **A wide edge margin is free at 18 cm/s** — same ten bytes measured, same
  r2, nearly double the clearance. It only costs data when the cap is high.
- **Point-to-point does not need the heading offset**; path following does.
  Closing the loop on position absorbs a rotated command frame up to ~45 deg,
  at the cost of a curved route. Past ~60 deg it stops converging.
- **Calibration legs are straight and driving curves, on the same robot.** That
  is not a controller fault: a calibration leg holds ONE heading and never
  re-aims, so it is straight however wrong the frame is — it simply goes the
  wrong way. Driving re-aims every frame, so a rotated frame bends the path.
  Noticing that difference *is* the heading offset, and the bench now reports
  the curvature and the degrees it implies.
- **Hunting and circling are different faults.** Hunting is an arrival radius
  tighter than the robot can hold; widening it fixes that. Circling is the aim
  frame being wrong; no tolerance fixes it.
- **Turn-and-go is implemented** (`swarm/pd.TurnAndGo`) and lost to PD in
  simulation — 12.1 s vs 7.0 s to arrive, same straightness. It may still win
  on hardware where commands are *dropped* rather than merely delayed. Not yet
  demonstrated either way; do not assume.

---

## 7. The plan, in order

Everything below step 3 is downstream of tracking. Until the blob count is
right, nothing is measuring anything.

1. **Get `BLOBS` to equal the number of connected robots.** COLOUR tab,
   `auto-tune`, then `check hues`. This is the single highest-value action and
   every previous session was compromised by skipping it.
2. **Re-pick the arena, clicking a ROBOT on each corner** rather than the floor
   mark. The calibration then maps the plane the balls travel in, and the
   ball-height parallax (1-5 cm, always outward) vanishes exactly. Verify with
   `check pos`.
3. **`run full`.** The pre-flight refuses if the tracker still is not
   following, so a pass means the numbers are real. Gains derive themselves.
4. **Then** judge the controller. Not before — three of the five numbers on the
   gains panel were fiction, and tuning on top of that is tuning noise.
5. ~~The ack bypass~~ — **built, and moved to step 0.** It is written and
   tested but unproven, and it changes the very numbers `run full` measures,
   so it has to be settled *before* the battery rather than after. Open the
   session with `compare_write_paths`, switch it on, and drive the ball across
   the arena while watching it: obedience is the thing to confirm, and the
   camera is the only instrument that can. If it misbehaves, unset the flag
   and carry on down this list unchanged.
6. Wire the heading estimator's live correction into `SpheroRobot.set_velocity`
   properly, and decide `follow(mode="orbit")` — both carried over from
   `HANDOFF.md` section 14.

---

## 8. Testing

995 tests, ~100 s, no hardware, no model, no network.

Three traps specific to this work:

- **The bench writes to live state by design** — `calib/colors.json`,
  `calib/motion.json`, `roster.json`, `workspace.json`. An autouse fixture
  fails any test that writes into the real `calib/`, and distinguishes that
  from an external process (a developer with the bench open) changing it.
- **A tight loop samples one instant.** The camera thread produces frames in
  real time; a test loop that runs 3600 iterations in milliseconds sees one
  frame. Auto-tune and corner picking need pacing.
- **Assert behaviour, not the absence of an exception.** Several bugs here
  passed tests that only checked "it drew without throwing" — the stacked
  MOTION tab, the invisible sliders, the layout cursor. Where a defensive
  default can mask a fault, the test must check the fault, not the symptom.
