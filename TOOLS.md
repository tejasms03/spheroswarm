# The tool layer — every tool, and how the model uses it

Twenty-two tools. This is the entire surface a language model has over the
swarm: there is no other way for it to affect anything. Generated against
`tools/registry.py`; if the two disagree, the registry is right.

For the architecture around this, see `HANDOFF.md`.

---

## How a tool call actually happens

```
system prompt (rebuilt every turn, state injected)
        |
        v
model  --tool_call-->  tools/registry.py  -->  tools/validate.py  -->  controller
        ^                                            |
        |________________ error, verbatim ___________|
```

Four things are true of every tool:

1. **It takes a `SwarmContext`** — fleet, workspace, controller, formation
   library — and returns `{ok, error, state_summary, ...}`. Nothing raises.
2. **Robots are addressed by four-letter `code`** (`SSMK`), never by index.
   Names work too: the prompt carries `SSMK/Seasmoke` so both resolve.
3. **Every target set passes through `tools/validate.py`** before anything
   moves: count, finiteness, bounds, obstacles, 20cm separation.
4. **Validation errors go back to the model verbatim.** The validator names the
   offending pair, the clamped points, the count mismatch — and that recovers
   more first-attempt failures than any amount of prompt tuning.

### Not every tool is offered every turn

Schemas are the largest item in the prompt: 22 of them is ~2,660 tokens, about
four times the system prompt, resent on every round trip. They also make
selection harder — more options, more wrong turns. So `schemas(text)` sends the
**core 12** plus whatever the user's wording implies:

| Group | Sent | Triggered by words like |
|---|---|---|
| Core | always | — |
| Motion | on demand | orbit, patrol, follow, swirl, path, loop, sweep, trail, surround, flank, forever, back and forth, lap, track |
| Convenience | on demand | swap, exchange, displace, nudge, a bit, slightly, shift, gather, cluster, spread, apart, mirror, opposite, reflect, where |

**This has a practical consequence.** "Go round and round in the middle"
contains no trigger word, so `set_flow` is never offered and you get a static
circle instead. If a motion command behaves like a plain `move_to`, that is
why. Full list in `tools/registry.py:_MOTION_WORDS`.

---

## 1. Sensing — 2 tools

The model rarely needs these: the system prompt is rebuilt every turn with the
arena, every connected robot's position, entities and saved formation names
already in it. The prompt says so explicitly, and the agent short-circuits
repeat sensing calls after something has acted.

### `get_state`
Every robot's code, name, position, velocity, LED, connection and kind, plus
arena bounds, obstacles, entities and saved formations. Per robot it also
carries **`layers`** (what continuous motion is running) and **`stalled`**.

### `describe_scene`
The same facts as prose — *"four dragons orbiting the wanderer, SSMK trailing
behind"*. Cheaper to read and easier to reason over than coordinate arrays, and
it is where the model learns what motion is currently running.

---

## 2. Movement — 3 tools

### `move_to` — the only movement primitive
| arg | type | meaning |
|---|---|---|
| `points` | array | unordered targets; robots take the nearest (Hungarian) |
| `assign` | object | exact placements by code, `{"SSMK": [100, 80]}` |

Every arrangement, however strange, is a set of points the model computes
itself. **No shape is hardcoded anywhere** — no circle function, no shape enum.
Give both arguments to pin some robots and let the rest fill in.

### `transform`
`translate [dx,dy]`, `rotate` degrees clockwise, `scale` about the centre.

Keeps each robot in its slot. The prompt pushes this hard for *"bigger"*,
*"rotate it"*, *"shift it left"* — recomputing every point instead is a whole
extra round trip and usually gets the count wrong.

### `stop`
`codes` optional; omit for the fleet. **Clears layers as well as targets** — a
flow that survived a stop would start driving again on the very next tick.

---

## 3. Computation — 1 tool

### `compute_points` — the highest-leverage tool in the set
| arg | type | meaning |
|---|---|---|
| `expression` | string | must assign `points` |
| `execute` | boolean | also move the robots — pass `true` whenever they should move |

Predefined: `n, cx, cy, xmin, xmax, ymin, ymax, width, height`, plus
`sin cos sqrt pi atan2 hypot min max abs round`.

Ask a small model for six points on a circle and it returns points that are not
on a circle, not evenly spaced, or contain a NaN. Writing
`[(cx+r*cos(i*2*pi/n), ...) for i in range(n)]` is a *language* task, which it
is good at, and the sandbox does the arithmetic exactly.

**Sandbox:** AST allowlist (no imports, no dunders, no `while`, no attribute
access beyond `math.*`), executed in a **subprocess** with a 1s timeout — the
separate process is what makes the timeout real, since banning `while` does not
stop `range(10**9)`. Verified against 15 escape attempts.

**Two failure modes worth knowing.** A `#` comment hides everything after it on
its line, including the `points =` assignment — the validator has a specific
error for exactly this. And a hardcoded point count (5 points for a letter A
with 6 robots) is rejected; the error now says how many to add or remove.

---

## 4. Timing — 1 tool

### `wait_until_settled`
`tolerance` (cm, default 8), `timeout` (s, default 20).

Call it between steps of a sequence, or the next command fires into a swarm
that is still moving. **It refuses when continuous motion is running** — a
looping path never settles, so waiting for one burns the whole timeout and then
reports failure. The error names `motion_control` so the model can get unstuck.

---

## 5. Expression — 1 tool

### `set_led`
`color` (name or `[r,g,b]`), `codes` optional, `blink` ∈ none/slow/fast/pulse.

Not decoration: **the camera tracker keys on hue**, so recolouring a tracked
robot is a real decision and is reported back. It targets all fleet codes
rather than only connected ones, which matters during bring-up — a robot
defaults to white, white fails the detector's saturation floor, and a robot the
tracker cannot see never becomes `connected`.

---

## 6. Formations — 4 tools

### `save_formation` / `recall_formation` / `list_formations` / `delete_formation`

Formations are stored **normalised** — centroid at origin, furthest point at
radius 1 — which is what lets a shape saved small in one corner come back
anywhere, at any size, at any angle. `recall_formation` takes `center`, `scale`
and `rotation`; recall at a different robot count is refused with both numbers
named.

The library starts **empty**. Nothing is hardcoded; every shape in it is one
the model was asked to save.

**Docking is a saved formation, not a tool.** The prompt teaches that "dock",
"go home", "park" and "return" mean: check `list_formations` for something like
`cave` or `home` and recall it — and if there is none, say so and ask the user
to arrange the robots and save it first.

---

## 7. Continuous motion — 4 tools

These run over **time** rather than moving to a place. They push a named,
weighted layer onto the robot's stack; the controller blends the stack, then
applies avoidance on top at a fixed weight the model cannot reach.

Rules that hold across all of them:

- **Layers are named**, so every addition is reversible. Anonymous blending is
  additive only: once three things are mixed, "stop doing the second one" has
  no answer.
- **Max 4 layers per robot.** Blended fields cancel, and a robot vibrating in
  place looks broken.
- **Every layer needs a duration** — default 60s, hard max 300s. A looping path
  with no bound is a swarm that runs until someone finds the STOP button. If
  the user gives no duration the model is told to choose one and say which.
- **Avoidance is not a layer.** If collision avoidance could be weighted down,
  eventually it would be, and then two robots meet.

### `set_path`
`assignments {code: [[x,y],...]}` **required**, `mode` once/loop/pingpong,
`speed`, `duration`, `append`.

Patrols, sweeps, laps. Each waypoint is validated as it becomes active, so a
looping path never drives into an entity that has since moved into the way.

### `set_flow`
`expr` **required**, `robots`, `entity`, `weight` (default 0.6), `duration`.

A velocity field: an expression in `x, y, t` assigning `vx` and `vy`. With
`entity` given, `ex`/`ey` bind to that entity's **live** position — which is
what makes orbiting a moving target possible.

Same AST allowlist as `compute_points`, but compiled **once** into a callable
rather than run per tick: a subprocess at 10Hz per robot is not viable. Before
anyone rides it the field is flown forward 200 steps, and rejected if it would
drive a robot into a wall — a syntactically perfect field can do that and
nothing in the source would tell you.

### `follow`
`target_id` **required** (a robot code *or* an entity id), `followers`,
`mode` ∈ trail/surround/flank/mirror, `distance` (default 40), `duration`.

Aims at `target_pos + target_vel * 0.4s`, not at the target — without lead
prediction followers visibly trail and overshoot when the target stops. If the
target goes stale, followers hold position rather than chase a stale estimate.

### Conditional rules — `when`

`set_flow` and `follow` take a **`when`** condition, re-checked every tick. The
layer only contributes while it is true. That is the difference between a
standing *action* and a standing *rule*: without it, a layer is on for its whole
duration and every condition has to be re-decided by a round trip to the model.

```
follow(target_id="wanderer", when="d_target < 80")     chase it only when close
follow(target_id="wanderer", when="rank == 0")         only the nearest robot
set_flow(expr=ROT, when="speed < 3")                   swirl only when stalled
```

Available: `x, y, t, speed, d_target, d_nearest, rank, elapsed, ex, ey, cx, cy,
n`. `rank` is this robot's position when the fleet is sorted by distance to the
target, so `rank == 0` is "whichever is nearest" — and it re-decides as they
move.

**Same sandbox as `compute_points` and flow fields**: no imports, no `while`, no
attribute access, compiled once rather than per tick. A condition that raises
drops its layer and records why, rather than driving on a broken rule.

**Debounced by ~0.2s.** A condition evaluated at 30Hz on noisy positions
flickers, and a robot that starts and stops thirty times a second is worse than
one doing the wrong thing steadily.

`follow` also takes **`select`** (`"nearest"`, `"nearest:2"`) — the same idea
applied to membership rather than to whether the layer fires, with 12cm of
hysteresis so two equidistant robots do not swap the job every tick.

### `motion_control`
`action` ∈ list/set_weight/remove/clear, `codes`, `name`, `weight`.

One tool rather than a push/pop pair per layer type, to keep the count down.

---

## 8. Convenience — 6 tools

**These exist because `qwen3.5:9b` cannot reliably do the arithmetic.** Asking
it to read two robots' positions out of state and construct a swapped mapping
fails repeatedly. Each of these takes identifiers only and computes the
geometry internally, and the prompt tells the model to prefer them over doing
arithmetic on the positions it was given.

| tool | required | what it does |
|---|---|---|
| `swap` | `code_a`, `code_b` | true exchange of two positions |
| `displace` | `code_a`, `code_b` | A takes B's place; B steps aside to a free spot |
| `nudge` | `direction` | shift by `distance` (default 30cm); direction is `left/right/up/down/toward:<id>/away:<id>` |
| `gather` | `around` | cluster in a ring of `radius` (default 45) around a robot or entity |
| `spread` | — | push apart to at least `min_distance` (default 60) |
| `mirror` | `code_a`, `code_b` | reflect A across `vertical`/`horizontal`/`centre`/`through:<id>` |

`displace`'s free-spot search is twelve random darts, first valid wins, falling
back to `nearest_valid_point` — the spot is arbitrary and nobody cares where it
lands, so optimising it would be wasted work.

`mirror` with `through:<id>` is the most useful and the hardest for a model to
do by hand, since it needs `2*pivot - b_pos` on both components.

---

## Deliberately not exposed

A test asserts their absence:

- **motor or heading commands** — a Sphero is a sphere with no observable
  orientation; the camera sees a glowing blob with no facing. Exposing heading
  would reintroduce the camera-frame offset problem the architecture exists to
  avoid. What people mean by "turn rate" falls out of paths and flows for free.
- **BLE connect/disconnect**
- **roster mutation** — adding, removing or renaming robots
- **file or shell access**

Heading *is* handled, but strictly below the fleet layer: each real robot
carries a `heading_offset` calibrated from observed travel, applied inside
`SpheroRobot.set_velocity`. Nothing above `fleet/` learns that heading exists.

---

## Measured behaviour

| | `qwen3.5:9b` (local) | `sonnet5` (hosted) |
|---|---|---|
| eval pass rate | 16/30 (53%) | not yet run in full |
| mean tool calls | 3.7 | ~1 |
| retries over 30 cases | 46 | ~0 |
| text recoveries | 0 | 0 |
| mean latency | ~109s | ~2-5s |

On the ten motion and convenience tools, Sonnet picked the right tool with the
right arguments on the first call in 11 of 12 hand-checked commands.

Two failure modes to know:

- **qwen on tight letter shapes**: "make the letter A" succeeded 1 run in 3,
  hitting the 8-call cap on the others. The binding constraint is the 20cm
  separation, not the geometry — six robots in an A puts the crossbar points
  ~19cm apart, and it re-sends near-misses. The validator now says "redraw it
  about 1.1x larger" rather than only naming the pair.
- **Sonnet occasionally replies "Done." with no tool call at all** — `ok=True`,
  nothing moved. Caught in 2 of 4 runs. The agent now nudges once when a reply
  *claims* an action but called nothing; genuine refusals and plain answers are
  untouched.

Neither model composes `follow` + `set_flow` unprompted for "orbit each other".
Both reach for `set_flow` alone, which supplies rotation but nothing holds the
radius, so the pair spirals apart (40cm → 177cm measured). The composition does
work when asked for explicitly: separation holds 40-44cm over 7.6 revolutions.
