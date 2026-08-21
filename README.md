# Swarm lab

A mixed fleet of Spheros — each robot independently simulated or real over
Bluetooth, in the same workspace — with a tool layer an LLM can drive.
Everything is in centimetres; pixels exist only inside the renderer.

```bash
cd spheroswarm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                      # all-sim: no robots, no camera needed
python app.py --camera 0           # run the tracker for real robots
pytest                             # the suite runs with no hardware
```

## The window

Left dock: status, controller, roster, formations, a command bar, and STOP.
Right: the workspace, drawn from `workspace.json`.

- **Controller** — `navigate` (go-to-target, what the tools drive), `boids`,
  `policy` (a trained checkpoint), `idle`
- **Roster** — every robot with its colour, a per-robot sim/real toggle, and
  add/remove. Click a name or code to edit it; changes write to `roster.json`
  and take effect on the running fleet
- **Command bar** — typed tool calls, with a log of results:
  ```
  move_to 60,40 120,40 180,40
  move_to SSMK=100,80 CRXS=140,80
  transform rotate=45 scale=1.5
  save_formation wedge
  recall_formation wedge center=120,90 rotation=180
  ```
- **Formations** — saved shapes with a thumbnail and a recall button. Empty on
  first run: nothing is hardcoded, the library is built at runtime
- **STOP** — always enabled, halts every robot

Keys: `space` pause, `b`/`p`/`i`/`n` controller, `tab` focus the command bar.

## The layers

| | |
|---|---|
| `roster.json`, `fleet/roster.py` | which robots exist, named after dragons |
| `workspace.json`, `workspace/space.py` | bounds and obstacles in cm |
| `fleet/` | `RobotHandle` — one interface for sim and real alike |
| `swarm/navigate.py` | go-to-target with Hungarian assignment |
| `tools/validate.py` | everything between a model's arithmetic and the floor |
| `tools/` | the tool layer, with JSON schemas for function calling |

Nothing above `fleet/` branches on whether a robot is real. The renderer draws
a ring around real robots and that is the only exception in the codebase.

### Setting up a workspace

```bash
python -m workspace.make --width 240 --height 180      # a plain rectangle
python -m workspace.make --from-camera --source 0      # click corners, then obstacles
```

## Training from the terminal

```bash
python -m swarm.train --task coverage --agents 6 --updates 300 --name sweep_v1
```

Checkpoints land in `runs/` as `.pt` plus a `.json` of config and final metrics,
and appear in the dock immediately.

## What the sim models

The dynamics exist to make transfer work, not to be pretty:

- **Command latency** — 1-3 control steps, randomised per episode. Your real
  loop is camera (~35ms) plus BLE (~60ms). A policy trained at zero latency
  oscillates on hardware.
- **Motor lag** — first-order response, tau 0.25-0.5s
- **Per-robot gain** — 0.8-1.2x, because no two Spheros roll the same
- **Heading bias** — a few degrees of aim error per robot

Randomising all four each episode is what buys you sim-to-real. Turn it off with
`randomize=False` to see how much it costs.

## Rewards

Coverage rewards a robot for entering a grid cell nobody has visited, penalises
collisions, and charges a small time cost. Nothing in it mentions flocking,
spacing, or alignment — spreading behaviour has to be discovered. That's the
point: if you reward cohesion and separation directly, you've written boids in
a different notation.

## Going to hardware

```bash
python deploy.py --controller runs/coverage_0812_141233.pt --dry
```

`deploy.py` is the original headless bridge and still runs a controller against
a tracker. For live work prefer `app.py`, which does the same through the fleet
layer: set a robot's kind to `real` in the roster (or with the per-robot toggle),
give it the Sphero's advertised name as `ble_name`, and start the app with
`--camera 0`. Connections are serialised, dropped robots reconnect with backoff,
and a robot the camera cannot see reports as disconnected rather than handing the
controller a stale position.

The `offsets` dict is the per-robot correction between camera heading and Sphero
heading. Roll each robot forward at heading 0 once, measure the displacement
direction on camera, store the difference. Redo it whenever a robot is picked
up. Getting this wrong makes everything curve consistently to one side, and it
is the single most common cause of "the policy doesn't transfer".

## A realistic month

Assuming evenings and weekends, and buying 3-4 more robots early.

**Week 1 — perception.** Camera mounted, HSV blob tracking with one colour per
robot, four-point homography to centimetres, Kalman filter per robot. Deliverable:
a window showing live positions in cm at 30fps with identities holding through
crossings. This is the whole project's foundation and it will take longer than
you expect.

**Week 2 — closed loop.** Frame alignment, the offsets calibration, multi-robot
BLE threading under load. Run boids on real robots. Deliverable: three Spheros
visibly flocking. Also measure your true end-to-end latency and set the sim's
range to match.

**Week 3 — transfer.** Train coverage in sim, deploy, watch it fail, find out
why, fix the sim, retrain. Deliverable: a trained policy that covers the floor
better than boids does, with a side-by-side video.

**Week 4 — the interesting bit.** Pick one: emergent role specialisation on a
task where robots must split up; an LLM task-commander setting goals for the
policy; or reward search where an LLM rewrites the reward function overnight.
Deliverable: a writeup with sim and hardware results.

Deliberately not in this month: more than ~5 robots on one Mac's Bluetooth,
anything needing precise formation control, and vision-based obstacle avoidance.
Each is a project on its own.

The biggest risk isn't the RL. It's that SPRK+ batteries are old, and a robot
that browns out mid-episode looks exactly like a policy bug. Buy spares, keep
them charged, and test with LED-only commands when you're debugging software.

## Vision (task 1 — no robots needed)

```bash
pip install opencv-python scipy
python -m vision.app --source synthetic     # fake camera, five moving blobs
python -m vision.app --selftest             # accuracy against ground truth
python -m vision.app --source 0             # your webcam
python -m vision.app --source 0 --calibrate # click 4 arena corners
python -m vision.app --source 0 --tune      # tune colour thresholds live
```

Keys while running: `c` recalibrate, `t` tune, `m` mask view, `q` quit.

`Tracker.read()` returns `{name: (x_cm, y_cm)}` — the exact interface
`deploy.py` expects where `FakeTracker` currently sits.

Measured on synthetic ground truth, six targets, 3% frame dropout:
100% of robot-frames tracked, 1.0 cm mean error, 2.9 cm at the 95th percentile,
62 fps single-core at 960x720.
