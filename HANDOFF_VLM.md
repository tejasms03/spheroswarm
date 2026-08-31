# Driving a Sphero from the VLM framework

A handoff for `vlm/` — the integration between this bench and the professor's
`Mobile-manipulation-with-VLMs`, done 30 August 2026. Read alongside
`HANDOFF_FLEET.md` (the bench underneath) and `HANDOFF_CALIB.md`.

---

## READ THIS FIRST

**It works in sim, end to end, and has never touched hardware.** Their real A*
plans a route, their real library starts the drive, our service carries it, and
the bench's own `pursue` puts a simulated ball 5.6 cm from a goal 70 cm away.
`tests/test_vlm_end_to_end.py` is that run, and it is the only test that would
have caught the wiring bug described in §5.

**Three of our files live inside THEIR tree, which is not version controlled.**
`Functions/Library/sphero_control.py`, `Functions/description/sphero_control.json`
and `Agents/Sphero Home/` have to be there because their loaders look there.
The authoritative copies are in `vlm/framework/`. A fresh clone of the
professor's repo silently deletes half the integration, and the only symptom is
an agent that cannot find its tools.

```bash
python3.13 -m vlm.install --check      # what is missing
python3.13 -m vlm.install              # copy it in
```

**The blockers from `HANDOFF_FLEET.md` §7 all still stand.** The frame is
mirrored, `arm()` refuses, and CRXS failed its battery. None of this integration
changes that, and none of it is trustworthy on hardware until they are fixed.

```bash
python3.13 -m vlm.server                                    # their RPC, our robot
python3.13 fleet_test.py --source sim --rpc                 # the bench, publishing
python3.13 -m vlm.monitor                                   # what the framework sees
python3.13 -m pytest tests/test_vlm*.py -q                  # 106 tests
```

---

## 1. The shape of it

Their framework is a ZMQ-RPC service mesh. Everything above `Functions/Library/`
reaches the world through two services, and that boundary is the whole
integration:

| theirs | ours | where it runs |
|---|---|---|
| `DataService` | unmodified | their server |
| `RobotService` | `vlm/service.py` | their server |
| `2_run_processing_services.py` (ArUco) | `vlm/bridge.py` | **the bench** |
| `PathControl/pp.py` (pure pursuit) | `vlm/driver.py` → `pursue()` | **the bench** |
| `TaskService` (SAM2) | not run | — |

**The service is a blackboard, not a driver.** It never opens a BLE link. Their
planner writes a path and a request; the bench reads them, drives, and writes
back the outcome. The alternative — the server process owning the Sphero — puts
two owners on one BLE link, and their `_move_robot_loop` sends every 25 ms,
about four times what the link takes. `fleet/real_handle.py` stays the only
thing pacing writes.

**Sensing and commanding share one thread and one socket.** `RPCClient` holds a
single ZMQ REQ socket and it is not thread-safe. `Bridge.publish_once` sends
the pose and then calls `Driver.serve` on the same client.

---

## 2. The two conversions, and how each is silently wrong

**Centimetres to arena pixels is a SCALE, not `Homography.to_px`.** That method
is the exact inverse of `to_cm` and lands back in *camera* pixels, which carry
the perspective the homography exists to remove. The chain is camera px → cm
(the homography, rectifying) → arena px (a scale, choosing units). Pinned by a
test that puts a tilted camera under both and requires them to disagree.

**`PX_PER_CM = 10.0` is a free choice that re-tunes their planner if changed.**
Their stack is pixels end to end and every constant in `PathControl/` is a pixel
count. At 10 px/cm our arena is 1388×1108 — the same order as the 1500×500
theirs was tuned against — so `robot_padding=30` reads as 3 cm and
`sweep_radius=65` as 6.5 cm.

**`theta` is TRAVEL, not facing.** A blob has no facing. Their control stack
reads `theta` as a body heading; a Sphero that slips or is nudged goes one way
while pointing another. At rest the last bearing is *held*, and every pose
carries `theta_fresh` and `theta_age_s` so nothing reads a remembered value as a
measured one.

**This is safe only because nothing on our path steers on it.** `pursue()` takes
no heading argument at all. `trace_targets` reads `theta` into `s_theta` and the
only line using `s_theta` is commented out; `line_following` computes a per-point
theta and then appends only `(x, y)`. `pick_and_drop` genuinely uses it and needs
a gripper, so it is out of scope. **If that ever changes, re-read this section.**

**The published frame must be the ARENA size.** `PlanningClient._get_planner`
takes the A* grid size from `frame.shape[:2]`, and with no `arena_corners` in
DataService it takes the arena rectangle from the frame bounds too. Publishing
the camera frame under arena-pixel poses gave their planner a 1280×1020 grid for
coordinates running to 1388×1108, and a ball at the far edge was off the map.
The frame is warped now, which also gives their UI the rectified top-down view
its own stitcher would have produced.

---

## 3. What is refused, and why refusing is the point

- **`set_command([left, right, gripper])` raises.** Reading (v, ω) back out of a
  wheel pair and integrating ω into a heading puts an open-loop dead-reckon
  under a controller that assumes closed-loop heading feedback, on a robot whose
  heading is not measured at all. It would run, and be wrong in a way that looks
  like tuning.
- **A drive request with no bench attached is refused, not queued.** An agent
  told "ok" waits for a ball nobody asked to move.
- **A coasting tracker publishes nothing**, and a lost ball clears the pose.
  Their `PoseFilter` holds a missing robot for five frames and `pp.get_pos`
  falls back to its last pose forever, so a quiet tracker looks exactly like a
  still ball.
- **`arm()`'s refusal is passed back verbatim** — mirrored frame, no homography,
  no aim. Swallowing it is how an agent plans on top of a rig that never moved.
- **The outcome is a verdict, not "stopped".** `armed` goes false for arriving,
  giving up stuck, losing the ball, and a person pressing escape.

---

## 4. Bugs in THEIR code, found on the way

Worth sending upstream. None are our doing and all four are live.

1. **The RPC layer is broken on cbor2 6.x.** Version 6 changed the `tag_hook`
   signature to pass the `CBORTag` first; `rpc_system.py:69` expects 5.x's
   `(decoder, tag)`, so `tag` binds to a bool and **every numpy array over RPC**
   fails with `error decoding semantic tag 42`. Their `pyproject.toml` leaves
   cbor2 unpinned, so anyone setting up fresh hits it. Pinned here to 5.9.0; the
   durable fix is a signature-agnostic hook.
2. **`1_run_server.py` will not start off their lab network.** It constructs
   `Tasks()` unconditionally, and `VLDetector.__init__` connects to a hardcoded
   `http://10.20.83.18:2021/v1`.
3. **`planning.py:309` treats radians as degrees** —
   `math.radians(float(s_theta_deg))` on a value their ArUco detector produces
   in radians. Dead in `trace_targets`, **live in `pick_and_drop`**.
4. **`pp.py:721` calls a dict.** `c.Robot.path_list()[id]` cannot work over RPC:
   the proxy turns every name into a call and the server calls the attribute.
   Ours is a method, so that line does what it plainly means.

---

## 5. Mistakes of mine worth carrying

1. **I wired the driver in with a `str.replace` whose indentation did not
   match.** It no-opped silently. The bridge published happily for a whole
   commit while never serving a single drive request, and every drive came back
   "No robot is attached". Every unit test passed throughout, because a mocked
   seam cannot catch a seam that was never connected. Use exact-match edits that
   fail loudly, and keep the end-to-end test. `HANDOFF_FLEET.md` §6.3 says the
   same thing about `str.replace` and I did it anyway.
2. **I published the camera frame under arena-pixel poses** and would have
   shipped a planner grid that disagreed with the coordinates. Caught only by
   reading `_get_planner` closely enough to notice it sizes the grid from the
   frame.
3. **I wrote order-dependent tests.** Collection here is not in definition
   order; two tests that assumed the ball had not moved passed alone and failed
   in company. Sharing an expensive server between tests is fine, sharing their
   leftovers is not.
4. **I told the operator to regenerate `Settings/arena_settings.json`.** It is
   a three-camera stitching config read only by their stitcher, their forwarding
   service and their Tkinter calibration screen — none of which we run. Same for
   `settings.json`'s `serial_port`. Both were busywork I proposed before
   checking who reads them.

---

## 6. Still open

- **Never run on hardware.** Everything above is sim.
- **Lighting is the real wall.** Their detection was ArUco under room lights;
  ours is a lit ball in the dark, and the two cannot coexist. That is why
  `TaskService` object detection is unreachable on this rig — finding objects
  needs the room lit and the tracker needs it dark. The first question is
  whether the LED clips at 255 while the floor sits near 150: if it does,
  exposure alone solves it (`python3.13 -m vision.expose --camera 0`, and note
  the lock matters as much as the value). If it does not, the options are a
  darker matte arena, blink-and-difference, or a bandpass filter. Putting the
  robot's IDENTITY in a blink pattern rather than in hue is the only approach
  that survives full room lighting for detection and identity at once.
- **Their React UI RUNS**, and shows the ball live. Three things it needed:
  `pip install pyserial` (`ReactGUI/port_utils` imports it at module scope even
  with no serial hardware); a CLEAN `npm install` in `frontend/` -- the shipped
  `node_modules` had a `vite` with no `dist/`, so `rm -rf node_modules
  package-lock.json` first; and note it serves on **8080**, not the 8000 their
  CLAUDE.md claims. Start it with `python3.13 backend/server.py` from the
  framework root, after `vlm.server` and the bench.
- **`vlm/server.py` names the robot Caraxes/CRXS by default**, so the UI labels
  it that whatever the bench actually connected. Pass `--name/--code` to match,
  or wire it to the roster.
- **Per-point dwell times are dropped.** Their fourth path element is a pause in
  milliseconds; `pursue` follows a path, it does not stop partway along one. A
  route asking for a pause is driven straight through and says so.
- **`is_path_feasable` is excluded from the agent** — it routes through
  `pp.feasibility_check`, which needs SAM masks from the `Tasks` service.
- **The agent layer has not been run against a live model.** The gateway is
  NYU-internal, so it needs VPN. `gemini.py` now defaults to
  `@vertexai/anthropic.claude-sonnet-5`, overridable with `VLM_AGENT_MODEL`.
- **One robot only**, by choice. `RPC_ROBOT_ID = 2`, matching the id their own
  sample `Data/robot_pos.txt` uses.

---

## 7. The order to work in

1. `python3.13 -m vlm.install --check`, whenever their tree has been re-cloned.
2. Fix the mirror (`flip y`, re-probe), zero the aim, get CRXS through its
   battery. Nothing below is trustworthy first.
3. Settle the lighting question — the LED-clipping test above. It decides
   whether this rig can ever use any of their vision.
4. Run the whole stack in sim with the real UI: `vlm.server`, the bench with
   `--rpc`, `backend/server.py`.
5. Then hardware, with the monitor open beside the bench.
