# Sphero agents inside the VLM framework — integration brief

Paste this into a fresh chat to work on **the professor's framework**
(`~/Downloads/Mobile-manipulation-with-VLMs-March`). For work on the Sphero
codebase itself, use `HANDOFF.md` and `TOOLS.md` instead — keep the two
conversations separate; they touch different repos and different hardware.

**The task:** add Spheros to that framework as agents, with their own tools.
Not a migration. The Sphero repo stays as the implementation behind those
tools.

---

## 1. What that framework is

`~/Downloads/Mobile-manipulation-with-VLMs-March`, ~50,000 lines of Python.
A March snapshot — **check it is still what the lab runs before building on it.**

- **Robots:** differential drive with grippers, commanded over one serial port
  as JSON lines: `{"id": 4, "left": 90, "right": 90}` / `{"id": 4, "gripper": "open"}`
- **Pose:** ArUco markers → `(x, y, theta)`, written into `Data.robot_poses`
- **Planning:** A*, pure pursuit, CasADi optimisers, feasibility checks
- **Architecture:** multi-process, RPC services (`Robot`, `Data`, `Task`)
- **Model:** Gemini 2.5 Pro, function calling
- **Deps:** ROS 2 (`ament-*`, `cv-bridge`), CasADi, customtkinter — none installed
  on this machine, so **it has never been run here**

### Extension model — this is the part that matters

```
Functions/Library/<lib>.py          the callables
Functions/description/<lib>.json    the schemas the model sees (Gemini format)
Agents/<Agent Name>/functions.json  picks {"library": ..., "name": ...} pairs
```

Ten agents already exist this way: Color Sorting, Line Following, Go-HOME,
Pantry Pick n Place, Multi-Robot PickDropAWARE. **A Sphero agent is exactly
that shape** — this is a normal extension, not a fight with the architecture.

Schema format differs from the Sphero repo's: types are uppercase
(`"type": "OBJECT"`, `"INTEGER"`, `"NUMBER"`). Conversion is mechanical.

---

## 2. The plan

### Vertical slice first

Prove the whole path before committing to anything:

1. `Functions/Library/sphero.py` — thin wrappers calling the Sphero repo's
   `tools.call(ctx, name, args)`
2. `Functions/description/sphero.json` — three tools: `move_to`,
   `compute_points`, `stop`
3. `Agents/Sphero Swarm Agent/functions.json` — listing those three
4. Poses fed into `Data.robot_poses` from the Sphero colour tracker

If that drives one Sphero from a Gemini prompt, the integration works and the
rest is volume.

### Then the rest of the tools

The Sphero repo has 22, documented in `TOOLS.md`. They are already grouped
(sensing, movement, computation, timing, expression, formations, continuous
motion, convenience) and each group maps to one description file.

---

## 3. The hard part: `theta`

The framework's pose is `[x, y, theta]` and **theta is assumed everywhere** —
`pure_pursuit`, `robot_controller`, the feasibility checks all read it.

**Spheros cannot supply it the way that framework does.** A Sphero drives by
rotating its outer shell around a fixed internal chassis, so an ArUco marker
stuck on the shell tumbles as it moves. Colour-blob tracking gives `(x, y)`
only. This is a hardware fact and no amount of code changes it.

**Decision taken: fuse two sources. This is now BUILT** — `fleet/heading.py`
in the Sphero repo, transport-agnostic, so its output feeds straight into
`Data.robot_poses[id]["theta"]`.

| source | strength | weakness |
|---|---|---|
| gyro yaw — `get_orientation()['yaw']` over BLE | works at rest, instantaneous | drifts over minutes; in the ball's own frame |
| camera travel direction | drift-free, absolute in camera frame | only valid while moving |

```
offset ← lowpass( atan2(camera_travel) − yaw )   # only when speed > ~10 cm/s
heading_camera = yaw + offset                    # valid always, including at rest
```

One unknown, observed whenever the robot moves, applied even when it does not.
Self-correcting if someone lifts a ball and puts it down rotated.

**Constraint:** poll orientation at **1–2 Hz, not 10**. Radio airtime is the
scarce resource — the Sphero handle already deadbands motion writes because of
it, and competing with motion commands is a bad trade. The offset is a slowly
varying constant; it does not need to be fast.

### What exists now

**`HeadingEstimator`** — passive, continuous. Measured against known ground
truth in simulation: **0.8 deg median error, 1.9 deg at the 90th percentile**,
against 35.7 deg for the obvious frame-to-frame implementation. Twenty times
the gyro drift still gives 0.9 deg; 3cm of camera noise gives 2.8 deg.

Four things carry that, in order of leverage:

1. **Baseline length.** Angle noise is sigma/d. At 30fps and 45cm/s a
   consecutive-frame delta is 1.5cm, so 1cm of position noise is +-37 deg. Over
   a 25cm baseline it is +-2 deg. Waiting for distance is the whole game.
2. **Turning rejection** — travel across a turn is a chord, not a heading.
3. **Circular statistics** — the mean of 359 and 1 is 0, not 180.
4. **Outlier gating**, once settled.

The trail must span enough *time* to reach the baseline at the slowest speed
worth calibrating: at 40 samples it was 1.3s, only 20cm at 15cm/s, and a
creeping robot never calibrated at all.

**`ActiveCalibration`** — the `calib` button. Drives four known legs and reads
the offset straight off them. **Needs no gyro whatsoever**: commanding a
heading in the ball's frame and seeing where it went in the camera's frame *is*
the measurement. It solves the cold start, which the passive estimator cannot.

Four legs rather than one because **disagreement is diagnostic** — legs
differing by more than 25 deg mean slipping, pushing, or the tracker following
a different robot, and it reports that instead of averaging a meaningless
number. Also refuses on a stalled leg, on no room, and on no camera fix.

Both are exercised on **simulated** robots too, because `SimRobot` models a
heading bias and now a bounded drift — so the whole path is testable with no
hardware. 22 tests in `tests/test_heading.py`.

**Still not done:** the gyro poll itself. `get_orientation()['yaw']` has never
been read off a real ball, and the estimator has never seen real hardware.
Poll it at 1-2 Hz, not 10 — radio airtime is the scarce resource.

---

## 4. The other seams

**Robot IDs.** Theirs are integers (`id_list = [1, 2, 3, 4, 147, 248, 225, 699]`).
Spheros use four-letter codes (`SSMK`). Needs a mapping table — and keep the
codes in the tool surface, since models handle names better than integers.

**Actuation — a Sphero has no serial port at all.** BLE only, via
`spherov2`. That is a transport difference, not a blocker, and how it is
resolved decides how much of the lab's stack works on Spheros.

`RobotService.set_command(robot_id, [left, right, gripper])` feeds one 40Hz
serial loop, and **26 call sites hardcode `c.Robot.get_all_robot_pose`, 23
hardcode `c.Robot.set_command`**. So:

- *A parallel `Sphero` service* is clean and changes none of their code — but
  their existing tools all say `c.Robot.*`, so A*, pure pursuit, the path
  controller and the feasibility checks would **not** drive Spheros.
- *Routing by transport inside `RobotService`* — serial for wheeled robots, BLE
  for Spheros — makes the whole existing stack work on them. This is the one to
  build.

The second is possible because **a Sphero can emulate differential drive**. It
takes `(heading, speed)`, and the conversion is exact:

```
v = (left + right) / 2            w = (right - left) / wheelbase
```

integrate `w` into a heading, drive at `v`. Verified in simulation against the
real fleet layer: `[60, 60]` runs straight (0.0cm lateral drift over 110cm),
`[-40, 40]` spins in place (0.0cm translation), `[30, 60]` arcs, `[0, 0]`
stops, and a gripper field is accepted and ignored. The wheelbase and speed
scaling in that proof are notional and need calibrating against real geometry.

**The cost:** a Sphero is holonomic — it can move any direction instantly —
and forcing it through a differential-drive model throws that away, adding
turn-then-drive where none is needed. So keep both paths: `set_command` for
compatibility with their controllers, and the native velocity API for
Sphero-specific tools.

**This is the same problem as §3.** The adapter *carries* the heading rather
than measuring it, so if its integrated theta drifts from reality the robot
curves. The fusion in `fleet/heading.py` is what keeps the adapter honest.
Solve theta once and both problems close.

**One more option worth knowing:** `spherov2` exposes
`raw_motor(left, right, duration)` — literally the framework's wire format.
Tempting, but it **disables stabilization**, so the ball stops holding heading
and wobbles. The heading+speed conversion above is the right path; `raw_motor`
is for stunts.

There is **no path-following command on a Sphero**. `roll(heading, speed,
duration)` is timed and open-loop, `spin` rotates, and nothing takes a waypoint
list. Path following is the controller's job either way.

**Pose out.** `Data.robot_poses[id] = {"x": ..., "y": ..., "theta": ...}` is
what everything reads. That is the single write point for the Sphero tracker.

---

## 5. Things to know before touching it

- **Four active copies of `controller.py`** (156/166/128/362 lines, four
  different hashes) sharing the same private function names — diverged forks
  with no canonical version. Same for `motion.py` (4 active + 1 in `OLD/`) and
  `detection.py` (4). **Establish which one is live before editing any of them.**
- **13 test functions** across 17 scripts, no pytest config; several are camera
  probes (`raw_cam_test.py`, `test_cams.py`). There is no suite to protect a
  change, so verify by running.
- Requires ROS 2. Budget setup time.

---

## 6. Worth taking in the other direction

`Functions/Library/astar.py` imports only `numpy`, `cv2`, `heapq`, `math` — no
RPC, no ROS. The Sphero controller's avoidance is reactive potential-field only,
with a documented local-minimum problem (`HANDOFF.md` §9a) that cannot escape a
concave obstacle. Porting A* in as a planner behind `move_to` is roughly a day
and is the one clear win the framework offers back.

Also of interest later: **arena stitching** (multi-camera) if the arena
outgrows one camera, and **pure pursuit** if wheeled robots are ever added —
holonomic velocity commands do not map cleanly onto nonholonomic robots, which
is exactly where that control code earns its place. Spheros are holonomic, so
it is a non-issue for them.

---

## 7. Verified vs assumed

**Also portable:** conditional layers. `when` is a gated expression re-checked
every tick, so a behaviour can be expressed as a standing rule
(`follow(..., when="d_target < 80")`) instead of a decision the model has to
re-make by round trip. The lab framework has no equivalent — its controller
threads are started and stopped explicitly per robot — and the sandbox that
makes it safe (`tools/validate.check_source`) is self-contained.

**Possibly reusable in the other direction:** `llm/memory.py` in the Sphero
repo records commands that worked and injects the relevant ones into the
prompt, with no tool call involved. It is model- and robot-agnostic — it scores
text against text — so it would drop into a Gemini agent unchanged.

**Verified by reading and running:** the extension model, the tool schema
format, the serial wire format, ArUco as the pose source, the RPC service
shape, the duplication, the test count, `astar.py`'s import list. A
differential-drive robot written against the Sphero repo's `RobotHandle` — ~55
lines — was driven by 4 of 5 Sphero tool families unchanged, which is why the
Sphero tool layer is worth keeping as the implementation.

**Not verified:** the framework has never been run here (no ROS 2). Perhaps 2%
of 50,000 lines read. Unknown: what hardware the lab owns, whether this
snapshot is current, who else maintains it.

**If the research goal turns out to involve manipulation** — picking things up
— Spheros are the wrong platform and no integration fixes that. Worth settling
explicitly before investing.
