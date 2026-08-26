# The hardware sessions — what was wrong, and what was done about it

A handoff for the work of 21–24 August 2026. Paste into a fresh chat alongside
`HANDOFF_CALIB.md` (the bench) and `HANDOFF.md` (the stack underneath).

Twenty-two commits on a repo that had none. `git log` is the record; this is
the reasoning, which `git log` cannot carry on its own.

---

## READ THIS FIRST

**Almost every fault in this session was a measurement lying, and almost every
one first presented as a controller problem.** The bench was rebuilt around
that: guards that refuse rather than guess, and messages that name what to look
at rather than what to blame.

Three things that will otherwise cost you a day, all learned the hard way here:

1. **Restart the bench after any code change.** Python loads `calib.py` once.
   Hours went into behaviour that had already been fixed on disk.
2. **A test that calls the handler never discovers that no click can reach it.**
   Two buttons shipped drawn off the edge of the window. There is now a guard;
   see §6.
3. **Verify before asserting, and verify that the verification ran.** One
   measurement in this session was worthless because the edit implementing the
   thing being measured had silently failed to write. See §5.

```bash
cd ~/spheroswarm
~/miniconda3/bin/python3 calib.py --camera 0            # the bench
~/miniconda3/bin/python3 calib.py --dry                 # rehearse, sim robots
SDL_VIDEODRIVER=dummy ~/miniconda3/bin/python3 -m pytest -q   # ~1180 tests
```

---

## 1. The root cause, found late

**The arena frame was mirrored.** `workspace.json` declares *"top-left, x right,
y down"*; the calibration had **+y going up the image**.

That is the one fault a heading offset cannot absorb. A rotation gives the same
error whichever way the robot drives, and one number cancels it. A mirror
reflects every command, so the error **changes sign with direction** — measured
across eight compass points it swings 270°:

```
commanded    actually goes    error
    0deg  ->     180.0deg     -180.0
   45deg  ->     135.0deg      +90.0
   90deg  ->      90.0deg       +0.0
  135deg  ->      45.0deg      -90.0
```

`heading_offset` is one number added to every command. It cancels a constant
error and can never cancel one that flips.

**Which explains the whole preceding fortnight.** Four calibration legs
disagreeing by 46° and 86° were not slipping and not a tracker on the wrong
ball — legs pointing different ways get different errors, by construction. A
robot that circled whatever you corrected was a robot whose frame could not be
corrected by one number. The aim-frame recovery was measuring honestly and
applying a fix that could not work in principle.

**Fixed twice over.** `set_rect` no longer trusts the order the four corners
were clicked in: the corner nearest the image origin becomes the arena origin
and the order is forced clockwise. Previously three of eight readings produced a
rotated or mirrored frame, silently.

**And it can still hide.** `_orient` makes the arena agree with the *image*. A
camera that delivers a mirrored image is already a reflection of the world, so a
frame consistent with it is inconsistent with reality — and **every check
available inside the picture passes**. A test pins that: a flipped arena maps
its own corners exactly as cleanly. Only driving a robot can reveal it, which is
what `watch_aim` and `teach` now do. `flip y` on the COLOUR tab is the fix;
pressing it twice is the identity.

---

## 2. The measurement layer

Everything here was a case of the tracker being confidently wrong.

**A dead camera reported freshly tracked robots forever.** `CameraTracker._run`
caught its error and continued without clearing `_fixes`, so `read()` handed out
the last positions indefinitely and `last_seen` was refreshed by a value that
never changed. Demonstrated: camera dead for 1 s with `STALE_AFTER` at 0.5 s,
`connected` still `True`. **Staleness was counted in frames, and when frames
stop, nothing ages.** Fixes now carry the time they were produced.

That mattered more than it looks: `connected` is the single boolean the drive
loop's stop-on-lost-fix, the battery's watchdogs, and `FleetEnv.apply` all rest
on. Fixing `apply` alone would have achieved nothing while `connected` lied.

**One ball was detected as two robots.** Yellow at hue 60±9 and cyan at 62±10 —
18 units of overlap against `palette.py`'s own `MIN_SEPARATION = 22`. Running
the real detector against one yellow ball returned **yellow and cyan at the
identical pixel**. Four independent filters now stand between a blob and a
robot:

| filter | rejects |
|---|---|
| worn colours only | hues nothing on the bench wears |
| ball-sized only | a ceiling light, a red sock |
| inside arena + 40 cm | the room |
| nearest to prediction | the flip between two similar blobs |

Two details worth keeping. `detect()` used to collapse to the largest blob per
colour *before the tracker saw it*, so two similar blobs traded places whenever
their areas crossed and the tracker could only accept or reject — never choose.
And an **empty** fleet hunts *everything*, not nothing: tuning happens before a
robot is connected, and a camera view that blanks looks broken.

**A ball is judged against the ball it claims to be.** A Sphero is ~7.4 cm; the
homography converts that to an expected size anywhere in the frame, measured
from a one-pixel step at each blob so the local scale is used. This replaced
hand-tuned pixel areas that three of six colours did not have set.

**Nearest-neighbour alone is not enough**, and the test says so: with two
identical blobs it locks onto whichever was biggest in frame one and stays —
consistent, and consistently wrong, which is worse than flickering because
nothing flags it. `fleet/identify.py` exists for that case: drive one robot's
LED and see which blobs follow. Built, tested, **not wired to a button**.

---

## 3. The calibration layer

**A failed run was governing the live speed limiter.** Stage errors were
recorded and then ignored while `recommend` was assembled. The blast radius was
wider than the gains panel: `safety.stopping_seconds` reads
`stopping_distance_s_per_cm_s` from the same block and sets the ceiling every
robot drives under. Re-fitting the run on disk now publishes only `kalman_r`
and `sim_latency_steps` and withholds the rest **with reasons**.

The brake fit published from `len(rows) >= 2` with no check that the rows
spanned different speeds — the two on disk were both byte 26. Its sibling, the
speed-map fit, already had exactly that guard (`max_speed_trusted`), added after
one unbounded extrapolation poisoned a calibration. The fix existed and had been
applied to one stage of two.

**`gains_from_motion` still says `measured: True` from a failed run** —
`measured = bool(step.get("tau_s") or lat.get("loop_delay_s"))`, with no
reference to whether those stages errored. **This is open; see §7.**

**Recentring drove out of bounds.** Closed loop on position, *open* loop on aim
— with a wrong frame it drove away from the middle, and the only thing that
noticed was a 30 s timeout, by which point the ball had gone 776 cm and was
outside the arena. Two signals now, because a wrong frame fails two ways:

| frame error | before | now |
|---|---|---|
| 30° out | arrives | **still arrives** |
| 180° out | 30 s, 776 cm | **0.5 s, 12 cm** — it gets *further away* |
| 90° out | 30 s, 776 cm | **5.1 s, 133 cm** — it *orbits*, covering ground without arriving |

At 90° the ball drives tangentially: it never gets further away, so the first
check never fires. That is why there are two.

**The battery backs off and carries on** rather than driving through the
boundary. The threshold is 3 cm and it is *measured*: a healthy battery comes
within 9–21 cm of a wall on purpose — the brake test coasts toward one to
measure the coast — so the 16 cm first tried fired on good runs. Clamping the
speed near the edge was the other obvious answer and is worse: stages plan their
legs at `plan_safety`, so a limiter near that margin quietly slows the
measurements it is protecting, and a slowed leg is worse than a missing one
because nobody knows its numbers are wrong.

---

## 4. Things that were us

Worth recording separately, because each cost real time and each looked like
hardware.

**The blinking red ball.** `AutoTune` drove the LED off and on once per colour
from a *fixed nominal table* while the bench lit from the hue being hunted —
`LED_RGB["red"]` is real red, `led_for("red")` was (4, 116, 0). It started on
red and `finish_autotune` never relit, leaving the ball glowing a colour nothing
was looking for. That is bug (h) from `HANDOFF_CALIB.md` surviving in the one
path with its own table. Worse than cosmetic: auto-tune *learns* from whatever
it lit, so every run quietly undid `optimise hues`.

**The camera probe broke the camera.** `CameraSource.capabilities()` tested each
control **by writing to it** — focus, exposure, gain, autofocus — restoring only
those that reported a changed value. Testing a control by operating it is not a
safe way to ask what a camera can do. Reverted; `CameraSource` is back to
`read`/`release`.

**"The teach tab doesn't work"** was literal: the button spanned x 1066–1186 on a
1150-wide window.

**"A and D don't work"** was not: the logged drive turned 125° at one point. A
sphere has no visible front, so turning while stopped shows as nothing. There is
now a live readout and a yaw-rate slider.

**"6 cm/s is a safe speed"** — the ball cannot do 6. Commanded byte 26, the
camera saw 0.8 cm/s. A cap at or under the deadband is not a slow speed, it is a
stopped one, and the run then reports a tracker that cannot see the robot.
Safety is the **edge margin's** job: that scales speed with the floor in front,
so a workable cap is still slow near a wall.

---

## 5. Two mistakes of mine worth carrying

**I reverted the user's arena calibration.** A conftest warning said `calib/`
had changed and "this process did not write it". That was the user working, not
corruption. Recovered from a printed diff. Live state must never travel
backwards with a code revert — the same trap caught a revert later, which had
swept `calib/` and `roster.json` into its commits.

**I measured a thing that was not there.** Arguing about an integral term, I ran
an experiment showing `ki` made no difference at any value. The edit
implementing the integral had aborted before writing; the constructor accepted
`ki` and ignored it. Every number was reading a controller with no integral in
it. Implemented properly, on a sloped floor it takes the standing error from
6.7 cm to 3.3 cm.

The lesson is not "measure" — it was measured. It is **check that the thing
under test is actually present** before believing a null result.

---

## 6. What is new, and where

| where | what |
|---|---|
| `fleet/teach.py` | drive it yourself; the aim frame and the safe area read off your driving |
| `RobotHandle.aim_zero` | move the robot's own forward to match the arena |
| `fleet/identify.py` | prove identity by driving the LED — **no button yet** |
| `fleet/sphero_fast.py` | fire-and-forget writes, `SPHERO_FAST_WRITES=1`, **unverified on hardware** |
| MOTION tab | `teach`, `yaw deg/s`, threaded `sensors` |
| DRIVE tab | `straight` / `PD loop` toggle, live kp/ki/kd/predict sliders |
| COLOUR tab | `flip y`, and `check pos` now *corrects* a constant shift |

**Straight-line drive is the DRIVE default.** A PD loop re-aims every frame, so
a rotated frame bends the path into a circle it can never close. A committed leg
holds one heading and travels straight however wrong the frame is — it goes the
wrong *way*, and a wrong way is a measurement. `TurnAndGo` was in `swarm/pd.py`
unused; its own docstring already made the argument.

Tuned at the ball and adopted as defaults: **re-aim on 10° of drift, no more
often than 450 ms** — about two command round trips on this radio, which is the
point.

**Guards added that would have caught earlier bugs:**

- no control may be drawn outside the window (three tabs × three sizes)
- a calibration's own corners must map cleanly *in the order actually used*
- a mirrored frame must be refused, not averaged
- `to_cm` and `to_px` must be exact inverses

**Performance:** `gains()` read and parsed `motion.json` from disk on every
call, and slider getters call it while drawing — several file reads per frame at
30 fps. Memoised; the suite went from 12 minutes back to about 2.

---

## 7. Still open

1. **`gains_from_motion` reports `measured: True` from a failed run.** The
   fourth home of a fault fixed in three others. It decides how the bench
   drives. Do this first.
2. **The two-LED heading.** `vision/facing.py` reverted with the camera probe.
   0.5° worst case in synthetic tests, and your arena resolves the two lights at
   ~13 px. If it returns: read-only capability detection, default off. It would
   retire `heading_offset`, the calibration legs and the mirror ambiguity at
   once.
3. **The ack bypass.** Built, off, unverified. Settle it with
   `sensors.compare_write_paths` **before** the battery, since it changes the
   numbers the battery measures.
4. **The blur slider**, lost in the revert, camera-safe, uncontroversial.
5. **Does `get_location` work?** `HANDOFF_CALIB.md` established the gyro is a
   cached struct; the locator has never been checked and is already in the
   probe's read list. If it is real, position and heading come off the ball and
   the camera becomes a cross-check rather than the foundation.

---

## 8. The order to work in

Unchanged in shape from `HANDOFF_CALIB.md` §7, but the reasons are now measured
rather than suspected:

1. **Charge the balls.** Nothing measured on a sagging battery means anything.
2. **Blob count = robot count.** `optimise hues`, `auto-tune all`, `check hues`.
3. **Set the arena by clicking a ROBOT on each corner** — maps the plane the
   balls travel in, so ball-height parallax is *gone* rather than corrected.
   (The parallax correction was built and then removed: it only mattered on a
   distorted camera and cost a pair of transforms that had to stay each other's
   exact inverse.)
4. **`teach`** — drive until W sends the ball where the screen says, then stop.
   It now moves the BALL's zero rather than storing a correction (see below),
   so one leg is enough. Two axes still matter if you want the mirror check.
5. **`run full`**, with the top speed above the deadband.
6. **Then** judge the controller.


---

## 12. `aim_zero` — the correction moved into the robot

Added after the rest of this document. It changes how the aim frame is fixed,
so read it before trusting section 4.

`heading_offset` is a number added to every command for as long as the roster
holds it. A Sphero establishes its heading reference **when it connects**, so a
stored offset is stale the moment the link drops — which is why re-measuring it
never stuck across a session, and why the value kept coming back wrong.

The v1.2 protocol can do better and this codebase had never used it.
`SpheroEduAPI.reset_aim()` takes whichever way the drive assembly is currently
pointing and calls that zero. So `aim_zero(error_deg)`:

1. rotates the assembly by minus the measured error, **at speed zero** — which
   turns the assembly without moving the ball, so it needs no clear floor
2. calls `reset_aim()`, making that direction the robot's own forward
3. sets `heading_offset` to zero

The correction now lives in the robot. Nothing is applied on every command, so
there is no signed number left to apply the wrong way round — which is where
most of this project's aim-frame bugs have come from.

Measured in the simulator, one leg then four different courses:

| true frame error | measured | error afterwards, courses 0 / 90 / 200 / 315 |
|---:|---:|---|
| +40° | −40.0 | all under 0.02° |
| −70° | +70.0 | all under 0.02° |
| +150° | −150.0 | all under 0.02° |
| **−175°** | +175.0 | all under 0.02° |

The −175° row is the one that matters: a near-reversal is where wraparound and
sign errors bite hardest, and it is where every offset-based attempt failed.

`teach` uses it when the robot has one and falls back to storing an offset when
it does not — the offset is the fallback now, not the mechanism. `SimRobot`
implements the same contract by moving its own bias, so the behaviour is
testable with no hardware.

**Not verified on a ball.** The call exists and the path through `spherov2` is
the v1.2 one a SPRK+ speaks, but that is reading the library, not driving. Two
things to know before trying it: `reset_aim` briefly turns stabilisation off, so
the ball will not self-right for that moment; and it corrects a **rotation**
only — a mirrored arena is still `_orient` and `flip y`'s job.

Three traps found writing the tests, all worth keeping:

- **The live estimator folds mid-measurement.** An open-loop leg looked steady
  and then quietly changed direction at step 119. `heading_tracking = False`
  while measuring, which is what `start_calibration` already did and for the
  same reason.
- **A sim robot slides along the arena wall**, which corrupts a measured travel
  direction into something plausible and wrong. Measure in open space.
- **Compass and maths angles run opposite.** The sim's bias is a maths angle
  and the measured error is a compass bearing, so the correction that looks
  wrong is the one that works. Settled by driving afterwards, not by reasoning.
