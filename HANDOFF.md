# Sphero swarm — what exists and how it fits together

A handoff document. Paste it into a fresh chat to bring it up to speed on the
codebase without reading ~20,000 lines.

Project root: `~/spheroswarm`. Python: `~/miniconda3/bin/python3` — the
`python3` on PATH is a bare Homebrew 3.14 with none of the dependencies.

```bash
cd ~/spheroswarm
~/miniconda3/bin/python3 app.py                        # all-sim, no robots, no camera
SDL_VIDEODRIVER=dummy ~/miniconda3/bin/python3 -m pytest -q   # 708 tests, ~44s, no hardware
~/miniconda3/bin/python3 -m evals.run --model stub     # eval harness, no model needed
source ~/.spheroswarm.env && ~/miniconda3/bin/python3 -m llm.smoke   # hosted gateway check
```

Two companion documents:

- **`TOOLS.md`** — all 22 tools, their arguments, and how the model is told to
  use each.
- **`HANDOFF_INTEGRATION.md`** — adding Spheros as agents inside the lab's
  VLM framework. **Keep that work in a separate chat**: different repo,
  different hardware, different assumptions.

---

## 1. What this is

Three things stacked on each other:

1. **A mixed-fleet swarm workspace.** Each robot is independently a simulated
   bot or a real Sphero over Bluetooth, freely mixed in one arena, driven
   through a tool layer. Everything in centimetres; pixels only in the renderer.
2. **A motion layer over that**: named, weighted velocity contributions, so a
   robot can seek a point *and* orbit a moving target at once, and either can
   be removed by name later.
3. **An LLM control layer over the tools**, local or hosted, plus an eval
   harness that measures whether a given model is good enough to drive it.

**Status:** all three work. The local model is measured (53% on 30 eval cases);
the hosted path is verified end to end but has no full eval run yet. Real
hardware is at first-light: one Sphero links over BLE and drives, the camera
pipeline runs, and the heading-offset calibration exists but has not been used
against a physical ball.

---

## 2. Layer map

```
roster.json          which robots exist (12 House of the Dragon dragons, 6 enabled)
memory.json          commands that worked, for the prompt to draw on
workspace.json       arena bounds, obstacles and entities, in cm
formations.json      shapes the model saved; starts empty, nothing hardcoded
calib/homography.json  pixel->cm mapping AND the rectangle it was built for

fleet/roster.py      load/validate/save; carries each robot's heading_offset
fleet/handle.py      RobotHandle — the one interface, sim or real
fleet/sim_handle.py  SimRobot: latency queue, motor lag, per-robot gain/bias
fleet/real_handle.py SpheroRobot: threaded BLE, deadband, reconnect, heading offset
fleet/manager.py     Fleet (add/remove/set_kind live) + FleetEnv
fleet/vision_link.py threads the camera; also hands the latest frame to the UI
fleet/heading.py     the offset estimator, and the active calibration routine

workspace/space.py   bounds/obstacle geometry, nearest_valid_point, clearance
workspace/entities.py  moving things: role (obstacle/target/both) + motion
workspace/flow.py    compiles a sandboxed velocity field once, not per tick
workspace/make.py    author a workspace, incl. --from-camera

swarm/layers.py      LayerStack: named weighted layers, caps, durations, stalls
swarm/fields.py      what each layer kind contributes (seek/path/flow/follow)
swarm/navigate.py    go-to-target controller, Hungarian assignment, blending
swarm/boids.py       unchanged behaviour, wall-aware of real bounds
swarm/policy.py      unchanged (trained actor-critic)
swarm/train.py       unchanged (PPO)

tools/validate.py    every target set passes through here before anything moves
tools/context.py     SwarmContext: the seam every tool is handed
tools/registry.py    JSON schemas + dispatcher + per-command schema gating
tools/command.py     text parser: "move_to 60,40 120,40"
tools/*.py           the 22 tools (see TOOLS.md)

llm/client.py        OpenAI-compatible client; backend-aware; tool-call recovery
llm/models.yaml      named presets, local and hosted
llm/prompt.py        system prompt, rebuilt per turn with live state injected
llm/agent.py         the agent loop: tools, retries, cap, cancellation, nudge
llm/session.py       runs the agent off the render thread for the UI
llm/smoke.py         standalone gateway check: does tool calling survive the hop?
llm/memory.py        learned precedents, injected — never fetched by a tool
llm/stub.py          script-replaying model for tests

evals/run.py         CLI: --model, --repeats, --compare, --only, --group
evals/harness.py     one fresh isolated world per case, scoring, reporting
evals/assertions.py  geometric checks (is_circle, matches_saved, ...)
evals/commands.yaml  the 30-case eval set
evals/stub_script.py a canned "good model" so the harness runs with no GPU

app.py               pygame UI: workspace, roster, camera view, command bar, STOP
```

**The central design rule:** nothing above `fleet/` branches on whether a robot
is real. Two tools consult `kind` for *pacing* and a *warning* — never for
behaviour — and the renderer rings real robots. A test asserts the uniform
state dict.

---

## 3. The fleet abstraction

Every robot — simulated or physical — presents the same surface:

```python
name, code, kind, color
pos, vel          # numpy (2,), centimetres and cm/s
connected, battery, last_seen
set_velocity(v)   # cm/s, the ONLY motion API
set_led(rgb); stop()
```

`SimRobot` integrates the same dynamics as the trainer: latency queue,
first-order motor lag, per-robot gain (0.8–1.2) and heading bias, plus battery
drain. **That gain matters when writing tests**: a robot commanded at 30cm/s
can legitimately reach 36. Assert on `_desired`, not on `vel`.

It also **drifts and slips**. The heading bias takes a bounded random walk
(~0.5–8 deg/min, clamped to ±34 deg) and up to 6% of each command is lost. Both
cost about 11% of a tick that already runs 190x faster than the render loop. The
point is not realism for its own sake: a *constant* bias is a robot you
calibrate once and forget, so without drift the heading estimator was being
tested against a world that could not fail it. `randomize=False` is still
perfectly deterministic, because tests that pin exact positions depend on it.
**The drift numbers are guesses** — no real ball has been characterised.

`SpheroRobot` runs a per-robot worker thread with a latest-wins command slot,
so `set_velocity` never blocks the control loop even when the radio stalls. It
has a fleet-wide connect lock with a 1.5s stagger, deadbanding, a reconnect
supervisor with backoff, and explicit detection of `CBErrorDomain Code=11`.

**Important semantic:** `connected` means *BLE link up **and** a tracker fix
within 0.5s*. A robot with a live radio the camera has lost reports
`connected=False`. `link_up` is separate, for the status line.

**Heading.** The camera measures position but never orientation — a glowing
sphere has no facing. Each real robot carries a `heading_offset` in the roster,
applied inside `set_velocity`. Nothing above `fleet/` learns heading exists.

`fleet/heading.py` supplies it two ways, and they are complementary:

- **`ActiveCalibration`** — the `calib` button. Drives four known legs and
  reads the offset straight off them; **needs no gyro at all**, because
  commanding a heading in the ball's frame and seeing where it went in the
  camera's frame *is* the measurement. Disagreement between legs is diagnostic:
  four legs that differ mean slipping, pushing, or the tracker following a
  different robot — which one leg cannot tell you.
- **`HeadingEstimator`** — passive, continuous, fuses gyro yaw with camera
  travel to track drift afterwards. Measured against known ground truth:
  **0.8 deg median, 1.9 deg at the 90th percentile**, versus 35.7 deg for the
  obvious frame-to-frame implementation.

Four things carry that accuracy, in order: **baseline length** (noise is
sigma/d, so a 1.5cm frame-to-frame delta with 1cm position noise is +-37 deg
while a 25cm baseline is +-2 deg), turning rejection, circular statistics, and
outlier gating. The trail must span enough *time* to reach the baseline at the
slowest speed worth calibrating — at 40 samples a robot creeping at 15cm/s
never calibrated at all.

---

## 4. The tool layer

22 tools in eight groups: sensing, movement, computation, timing, expression,
formations, continuous motion, convenience. **`TOOLS.md` documents every one.**

Two structural facts worth carrying in your head:

**Schemas are gated per command.** All 22 is ~2,660 tokens resent every round
trip, and more options means more wrong turns. `schemas(text)` sends the core
12 plus whatever the wording implies. A command with no trigger word never sees
`set_flow` — "go round and round in the middle" gets a static circle.

**No shape is hardcoded anywhere.** No circle function, no shape enum. The
model gets the bounds and the robot count and computes coordinates itself.
`move_to` is the only movement primitive.

---

## 5. Validation

`tools/validate.py` sits between a model's arithmetic and robots on a floor.
**Every** target set goes through it. Nothing raises; every failure is a
readable dict the model can act on:

- count matches connected robots — **and says how many to add or remove**
- coordinates finite (rejects NaN, inf, strings, nulls, wrong arity)
- out-of-bounds points clamped, with a per-point report
- points inside obstacles rejected or clamped
- minimum 20cm separation, naming the offending pair — and when several pairs
  near-miss, saying the shape is too small and by what factor to enlarge it
- clamped targets keep a 12cm standoff from walls and obstacles

`compute_points` adds a sandbox: AST allowlist executed in a **subprocess**
with a 1s timeout — the separate process is what makes the timeout real.
Verified against 15 escape attempts.

---

## 6. Layers and continuous motion

A robot holds up to **four** named, weighted contributions. Final velocity =
weighted sum → avoidance → clamp.

- **Layers are named**, so every addition is reversible.
- **A layer can be conditional.** `when` is a gated expression re-checked every
  tick — `d_target < 80`, `rank == 0`, `speed < 3` — so a layer becomes a
  standing *rule* rather than a standing action. Same sandbox as
  `compute_points`, compiled once, debounced ~0.2s because a condition
  evaluated at 30Hz on noisy positions flickers. This is the general mechanism
  for conditional control; `select: nearest` is one special case of it.
- **Avoidance is not a layer.** Fixed weight, applied after the blend, out of
  the model's reach. If it could be weighted down, eventually it would be.
- **Every layer carries a duration**: default 60s, hard max 300s.
- **Stall detection**: commanded speed under 3cm/s for 2s with layers active
  sets `stalled`. The conversion from normalised output to cm/s uses the fleet's
  real speed scale — hardcoding it was a bug.

---

## 6b. Two kinds of memory

**The formation library** is a filing cabinet: shapes saved by name, recalled by
name. It now stores **motion** as well as position — a patrol, an orbit or a
follow saved with the shape, in the same normalised frame, so a route taught
small in one corner comes back anywhere at any size and angle. `list_formations`
marks which entries move.

**`llm/memory.py` is the other kind, and nothing addresses it by name.**
Commands that worked are recorded automatically; the ones relevant to the
current request are injected into the prompt. The model never calls a tool to
read them — it simply finds a precedent already in front of it.

- **Only successes are kept.** A failure recalled later is a suggestion to fail
  the same way, and failures are already handled better by the validator
  handing its error back inside the same turn. A turn that only sensed is not a
  precedent either.
- **Relevance is scored, and silence is the default.** A list of loosely
  related past commands is worse than none: it invites copying the nearest
  thing rather than answering the question.
- **The budget is hard** — 420 characters. Memory is the one part of the prompt
  that grows without bound, so it is trimmed rather than allowed to push the
  arena and the robot positions out.
- **Robot identifiers are stripped before matching.** "swap SSMK and CRXS" must
  find the precedent set by "swap Seasmoke and Caraxes": which robots were
  named is an argument, the verb is the intent. Without this it missed entirely.

## 7. The LLM layer

### Client

Talks to an **OpenAI-compatible** endpoint. `backend` (`ollama` |
`openai_compatible`) decides which parameters are sent, because tolerance is not
symmetric:

| | ollama | hosted |
|---|---|---|
| `options.num_ctx` | yes | no |
| thinking toggle | yes | no |
| `max_tokens` | no | yes |
| `temperature` | yes | **only if the model accepts it** |

That last row is not hypothetical: Claude via Vertex answers a flat
`400 — temperature is deprecated for this model`. `temperature: null` means
omit, not zero.

Presets live in `llm/models.yaml`: local (`qwen3.5:9b`, `qwen3.5:4b`,
`qwen3:8b`, `glm-4.7-flash`) and hosted through the NYU Portkey gateway
(`sonnet5`, `haiku4.5`, `gemini-flash`). Hosted presets name an
`api_key_env` — the key is never written to the file.

Handles the **tool-call-as-text** failure: three shapes recovered with a
brace-depth scanner, each flagged `recovered_from_text=True` so the eval counts
how often a model needs rescuing.

### Prompt

Rebuilt fresh every turn, ~1,020 tokens against a 1,200 budget (raised from 800
when the motion toolkit landed).

- **State is injected, not fetched** — arena, obstacles, every connected robot
  as `CODE/Name (x,y)`, entities, saved formations. Disconnected robots are
  listed separately as "offline, ignore".
- **Entities must be listed or they cannot be referred to.** Asked to "follow
  the wanderer" against a prompt with no wanderer in it, a model invents one:
  it reads the phrase as *a robot that wanders*, picks one, and drives it.
- **`compute_points` is pushed as the primary path** — writing the formula is a
  language task; the sandbox does the arithmetic exactly.
- Dragons are robots, continuous motion needs a duration, prefer the
  convenience tools to arithmetic, and docking is a saved formation.

### Agent

- **Validation errors go back verbatim.** This loop does more work than any
  amount of prompt tuning.
- **The cap is hard, and hitting it stops the fleet.**
- **A completion claim with no tool call is nudged once.** Sonnet intermittently
  replies "Done." having called nothing — `ok=True`, nothing moved. Genuine
  refusals and plain answers are left alone.
- **Long-running tools are not wrapped in the shared lock.** They take it per
  tick instead; see §9c.
- Nothing escapes: every path returns an `AgentResult`.

---

## 8. What has and has not been verified

**Verified:** 708 tests, ~44s, against mocks and stubs. The eval harness runs
30/30 with the canned stub — *that is a harness test, not a model score.*

**Local model** (Ollama 0.32.9, `qwen3.5:9b`, M1 Pro 16GB, 2026-08-11):
16/30 (53%), 3.7 mean tool calls, 46 retries, 0 text recoveries, ~109s mean.
Weakest: sequence 1/3, memory 2/5. Safety 2/2.

`options.num_ctx` is **ignored** by Ollama's `/v1` endpoint; the only control
that works is the environment (`OLLAMA_CONTEXT_LENGTH`). Currently loading at
16384 with `OLLAMA_KV_CACHE_TYPE=q8_0`.

**Hosted gateway** (NYU Portkey, `@vertexai/anthropic.claude-sonnet-5`,
2026-08-17): `llm.smoke` passes all three checks — plain completion, **all 22
tool schemas through with native `tool_calls`** (the recovery path never fired),
and a `tool` role follow-up. ~1.5s per call. 23 models exposed.

Hand-checked: 11 of 12 motion and convenience commands right on the first call
with zero retries.

**Not verified:**

- **No full eval run against any hosted model.** The one on disk is quarantined
  in `evals/results/invalid/` — it scored 0/30 because the gateway was
  unreachable, not because the model failed.
- **Every real-BLE path** beyond first light — deadband, backoff, `Code=11`.
- **The heading-offset calibration against a physical ball.**
- The `thinking` toggle.

**The gateway is internal to NYU.** Off-campus it fails DNS resolution, not
connection. The app falls back to a local preset with a visible notice.

---

## 9. Bugs found during the build (read before changing these)

**a) Obstacle avoidance had two successive failure modes.** Pure radial
repulsion stalls a robot dead in front of an obstacle. Adding a tangential
slide created a worse bug: the side to pass on came from the seek vector, which
points at the target, so every drift flipped the tangent and the robot
oscillated forever. **The side must come from how far off the obstacle's
centreline the robot already is.**

**b) Validator and controller disagreed about reachability.** Clamped targets
landed inside the controller's repulsion field; robots orbited their own
targets forever. Fixed in two places: avoidance fades as a robot arrives, and
clamped targets keep a 12cm standoff.

**c) The render loop kept its own copy of the tick, and it drifted.** The copy
never advanced moving entities or the controller's clock, so flow fields were
frozen at `t=0` and patrolling entities sat still — *except* during
`wait_until_settled`, which ticks properly on the agent thread, so the world
lurched forward only while a tool happened to be running. Tests missed it
entirely because every test drives `ctx.tick` directly. The loop now calls
`ctx.tick(dt, controller=...)`; **do not re-inline it.**

**d) The shared lock was held for a whole tool call.** `wait_until_settled`
ticks for up to 20s, so it monopolised the lock for 20s: the window stopped
redrawing and no other robot could be commanded. The lock is now held for one
tick, and self-ticking tools are not wrapped in it.

**e) Unmocked BLE aborts the interpreter.** On macOS a real `connect` from a
worker thread under pytest kills the process — no traceback. `tests/conftest.py`
has an autouse fixture replacing `default_connector` with a raiser. **Never
remove it.** It earned its keep again this month, catching a fixture that read
the live `roster.json` after one robot was set to `real`.

**f) A UI hit box silently stole clicks.** There are now tests asserting no two
clickable regions overlap.

**g) The eval assertion parser rejected its own eval set.** `ast.literal_eval`
cannot read bare words, so nine cases failed with a parse error that looked
exactly like a model failure.

**h) `rotated_by` measured something unmeasurable.** A symmetric six-robot
circle turned 45° is the same point set as one turned −15°. The harness now
threads `{code: position}` maps into the assertion.

**i) `is_spiral` passed straight lines.** It now requires ≥200° of wrap around a
*searched-for* centre. It still passes ~5% of random 6-point clouds; a test
pins the rate below 10%.

**j) A circular import decided by import order.** `workspace.space` →
`workspace.flow` → `tools.validate` → `tools/__init__` → `tools.motion` → back
to the half-built `workspace.flow`. `app.py` imports `tools` first and never saw
it; any standalone script hit it immediately. `workspace/` now imports `tools`
lazily — it sits *below* tools and must not import upward at module load.

**k) The camera and the planner can disagree about the arena.** A homography
calibrated to a 200×200 square under a 240×180 workspace puts the tracker and
the controller in different coordinate systems: robots calmly drive off an
arena they are nowhere near the edge of, and nothing errors. `calib/homography.json`
now records the rectangle it was built for, and the app warns at startup.

**l) `workspace.make` silently deleted entities.** Recalibrating rebuilt the
workspace from scratch, dropping the wanderer — noticed only later, when a
`follow` said no such entity. It now carries them over.

---

**m) The prompt advertised tools the model was not given.** The motion rules
named `set_path`/`set_flow`/`follow`/`motion_control` unconditionally, but
schema gating only sends them when the wording contains a trigger word. "Go
round and round in the middle" matches none — so the model read about
`set_flow`, called it without a schema (dispatch does not check what was
offered), guessed the arguments and burned the whole call budget. The prompt is
now built from the same schema set that is sent; a test asserts the invariant
across seven commands.

**n) A reply claiming an action nobody performed.** Sonnet intermittently
answers "Done." — or nothing at all — with no tool call, `ok=True`, nothing
moved. A false success is worse than a failure because nobody goes looking. The
agent now nudges once when a turn calls nothing and either claims completion or
comes back empty, and contradicts a repeated claim. Refusals and plain answers
are untouched.

**o) "Stalled" could not tell blocked from satisfied.** An arrived robot
commands exactly zero, so a `follow` holding the correct radius on a stationary
target looked identical to one wedged against a wall. A stall now means *asked
to move and went nowhere*: the controller passes what the layers requested
before avoidance and clamping, and `get_state` reports the two numbers.

**p) The calibration stored the error as if it were the correction.** The
`calib` button assigned `heading_offset = cal.offset`, but that value is the
measured *error*, so it doubled the problem rather than cancelling it. The
`c`-key path had it right (`offset - error`). The test encoded the same
mistake and passed. It now asserts behaviour — *does the robot go where it is
told* — which a sign error cannot satisfy.

**q) The prompt could advertise a tool the model had no schema for.** See §9m;
mentioned again here because the same shape of bug is easy to reintroduce
whenever the rules and the schema set are edited separately.

## 10. Speed, and the knob that is not the knob

`CRUISE_SPEED` in `app.py` (currently **45 cm/s**, 75% of the ceiling) is what to edit, or
`--speed`. It sets `ctx.max_speed`, which scales controller output.

**Do not edit `fleet/handle.py: MAX_SPEED`.** It looks like the obvious place
and is not: that constant is also the cm/s-to-motor-byte calibration
(`byte = speed / MAX_SPEED * 255`). Halving it maps 30cm/s onto byte 255 and a
real ball rolls exactly as fast as before while reporting half speed.

`swarm/sim.py: MAX_SPEED` is a third thing again — the trained policy
normalises its observations by it, so changing it invalidates the checkpoints.

---

## 11. The UI

`app.py`, pygame, ~1,850 lines.

```
space  pause              b/p/i/n  controller     tab   focus command bar
1-9    take manual control         W/S drive  A/D turn  esc  release
c      calibrate heading offset from one observed leg (real robots only)
calib  button: drive a four-leg square and read the offset off it
v      camera view: panel / floor / off      a     ask / literal-tool mode
PgUp/PgDn  scroll the log      END  follow the tail   Cmd+V  paste
F11    fullscreen (esc returns); the window is resizable
calib  button: drive a four-leg square and read the offset off it
```

- **The camera can be the floor.** In `floor` mode the frame is warped through
  the homography into the arena rectangle and drawn under everything, so sim
  robots stand on the real surface. When the grid lines up with the tape on the
  floor, the calibration is right — which beats reading `arena=200` out of a
  JSON file.
- **A mismatched calibration is reported at startup** and drawn as a dashed
  amber rectangle over the arena.
- **Telemetry is four stat tiles** (robots, real linked, tracker, ble) rather
  than three lines of run-together text.
- **The log scrolls back** 400 lines, and a new line while you are scrolled back
  does not yank the view — reading history during a run has to be possible.
- **Resizing rebuilds the dock.** Panels are positioned from W/H at build time,
  so `apply_size` re-runs `_build` and drops the cached camera warps, which
  would otherwise be stretched to the old arena rectangle.

- **Manual drive** writes velocities directly through `set_velocity`, so it
  works on real balls **without a camera** — the quickest proof a radio
  round-trips. Taking control clears that robot's layers and targets.
- **The camera view** is picture-in-picture with every detected blob ringed in
  its own colour. A robot the detector has lost shows as an absence. Running
  `vision.app` alongside would fight for the same device.
- **A direct tool call still runs while the model is busy** — one agent on one
  thread cannot take a second natural-language request, but `move_to CRXS=60,40`
  goes through.
- **The model preset falls back** to a local one when a hosted preset is
  unreachable, loudly, with the active model always in the status line.

---

## 12. State files are live, not source

`roster.json`, `workspace.json` and `formations.json` are rewritten by the
running app. Anything constructing `App()` without explicit paths mutates the
real files; anything *reading* them in a test inherits whatever a developer left
behind. When scripting, pass `roster_path=` / `workspace_path=` and a scratch
`FormationLibrary(path=...)`. Two tests assert an eval run and an agent run
leave all three byte-identical.

`~/.spheroswarm.env` (mode 600, outside the repo) holds `PORTKEY_API_KEY`.
Source it before using a hosted preset. It is never written into `llm/models.yaml`.

---

## 13. Tests

708 tests, ~44s, no hardware, no model, no network. `SDL_VIDEODRIVER=dummy`.

```
test_app.py            109   UI headless: roster, camera, manual drive, calibration, locking
test_validate.py        62   NaN/strings/nulls, clamping, separation, the sandbox
test_evals.py           52   assertion helpers, world isolation, the CLI
test_llm_client.py      46   mocked HTTP: backends, num_ctx, all recovery shapes
test_tools.py           43   every tool, incl. integration against a live fleet
test_llm_agent.py       43   cap, retry, history, cancellation, the nudge, learning
test_fleet.py           42   mixed fleets, deadband, reconnect, heading offset, drift
test_llm_prompt.py      39   state injection, entities, precedents, token budget
test_motion_tools.py    45   set_path/set_flow/follow/motion_control, composition
test_layers.py          32   blending, caps, expiry, stalls, import order
test_llm_ui.py          27   session threading, ask/tool mode, STOP aborts a run
test_convenience.py     27   swap/displace/nudge/gather/spread/mirror
test_entities.py        24   path/flow motion, reflection, roles, old schema
test_heading.py         22   offset accuracy vs ground truth, active calibration
test_navigate.py        21   Hungarian optimality, obstacle rounding, convergence
test_workspace.py       18   geometry, nearest_valid_point, entity preservation
test_command.py         15   the text parser
test_memory.py          13   recall relevance, identifier stripping, the budget
test_roster.py          13   duplicates, ble_name, >6 enabled
test_sim_workspace.py    8   sim takes bounds from a workspace
test_policy_compat.py    7   trained checkpoints still load and act correctly
```

Three things that will bite you writing tests here:

- **Sim robots start at random positions**, and carry a random gain of 0.8–1.2.
  Pin positions explicitly, and assert on `_desired` rather than `vel`.
- **A tight loop samples one instant, not a duration.** Forty back-to-back
  lock attempts tell you where the other thread happened to be — a coin flip.
  Sample across real time.
- **Real-BLE behaviour is proven only against mocks.**

---

## 14. Sensible next steps

1. **Run a full eval against a hosted model.** `--model sonnet5`, then
   `--compare qwen3.5:9b sonnet5 haiku4.5 gemini-flash`. This is the first real
   number for the hosted path, and at ~2s per call it costs minutes, not hours.
   Requires the NYU VPN.
2. **The eval set does not cover the motion toolkit.** All 30 cases predate it.
   Sixteen more are specified but unwritten: orbits, patrols, follows, the
   composition case, and the convenience tools. Log the failure reason per
   case, distinguishing *wrong tool* from *right tool, bad arguments* — that
   distinction decides whether to add convenience tools or prune them.
3. **Close the stale-fix gap.** `FleetEnv.apply` writes velocities to every
   handle with no `connected` check, so a robot the camera has lost keeps being
   driven from its last known position. The fleet layer's own docstring says
   this is exactly what `connected` exists to prevent.
4. **Real-hardware bring-up, continued.** One ball links and drives. Next:
   recalibrate the arena from the camera, light the LED, confirm a tracker fix,
   then `1` / `W` / `c` to calibrate the heading offset.
5. **Decide on `follow(mode="orbit")`.** Neither model composes `follow` +
   `set_flow` unprompted; both reach for flow alone, which spirals apart. §5 of
   the motion spec says to build the composable version first and measure —
   that measurement is now in, and it argues for the shortcut.
6. **The UI still lacks layer chips and a flow-arrow grid** from the motion
   spec's §9.
7. There is **no git repository.** Consider `git init` before changing things.
