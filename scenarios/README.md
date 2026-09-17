# Multi-ball scenarios and traffic simulation (overnight 2026-09-17)

Nothing in the bench (`taillight.py` and friends) was changed. This folder only adds:

- scenario files for the scenario runner (not built yet);
- the JavaScript simulation that produced their expected numbers.

## Scenario files

| file | what it tests |
|---|---|
| `base_cross.json` | two p2p legs through one crossing |
| `base_head.json` | two balls driving at each other on one line |
| `base_orbit.json` | an orbit with a parked ball sitting on the circle |
| `base_three.json` | three balls crossing the middle |
| `base_corridor.json` | two patrols in opposite directions on the same line |
| `base_orbits.json` | two orbits whose circles touch |
| `stress_2balls_seed11.json`, `stress_2balls_seed12.json` | 2 balls, 10 random jobs each, 180 s |
| `stress_3balls_seed21.json` | 3 balls, 10 random jobs each, 180 s |

**Format.** All positions are arena cm, x right, y down.

- `balls[]`: slot, start, facing, and whether the ball is parked.
- `jobs[]`: `ball`, `kind` (p2p / line / poly / orbit / patrol / park) and that kind's arguments.
- Each ball does its jobs in order, starting the next 3 s after the previous finishes.
- `expected_sim` holds the simulation's numbers at two settings. These are the numbers to compare a floor run against.

## The realistic ball

The simulated ball is built from real data:

- `calib/ball_SK-914A.json` and `calib/ball_SK-4C2C.json`;
- 68 recorded runs in `runs/p2p/`.

**Measured on the bench:**

| quantity | value |
|---|---|
| driving speed | median 14.3 cm/s, 11–19 cm/s spread |
| time to 80% speed | ~0.65 s |
| coast after stop | ~7 cm |
| spin rate | median 48°/s, p90 141°/s (stick-slip) |
| run-on after a spin | up to ~23° (SK-4C2C) |
| frame interval | 0.034 s |
| frames lost | 1.2% |
| orbit error | ~2.5 cm rms (only 1 run) |
| line error | ~1.4 cm rms (only 1 run) |

**Simulation parameters.** This uses the worse ball of the two, on purpose.

| parameter | value |
|---|---|
| command delay | 0.37 s |
| camera delay | 0.1 s |
| position noise | 1 cm |
| heading noise | 5° |
| frames lost | 1.2% |
| speed wander | ±25% |
| acceleration / deceleration time constants | 0.35 s / 0.5 s |
| spin rate | 40–140°/s, 30% chance of a 0.4 s stall |
| run-on | up to 23° |
| over-turn | 1.2× |
| heading drift | 6°/s, capped at ±15° |

The simulated follower gives 2.0 cm rms off an orbit and 2.2 cm off a line, measured through the camera. That is slightly worse than the real runs, on purpose.

**Traffic logic.** This is the path-timed planner. Balls follow their real smooth paths. The planner books space along each path in time and decides where to hold. A predictive brake sits underneath.

## Results (0 contacts in every run below)

### Base cases, realistic ball, 5 noise seeds each

| case | normal 12 cm/s | slow 9 cm/s, wide margins |
|---|---|---|
| cross | 10/10 jobs, 28 s, closest 35 cm | 10/10, 30 s, 39 cm |
| head-on | 10/10, 45 s, 16 cm | 10/10, 56 s, 22 cm |
| orbit + parked | 5/5, 64 s, 15 cm | 5/5, 65 s, 19 cm |
| three-way | 15/15, 38 s, 20 cm | 14/15, 122 s, 18 cm |
| corridor | 10/10, 107 s, 15 cm | 10/10, 124 s, 19 cm |
| crossing orbits | 10/10, 65 s, 16 cm | 10/10, 65 s, 34 cm |

### Random stress, 300 s

Jobs finished and closest pass. Seeds: 12 each for 2 and 3 balls, 6 each for 4 and 5.

| balls | clean ball | realistic, normal speed | realistic, slow + wide |
|---|---|---|---|
| 2 | 92%, 17 cm | 87%, 13 cm | 86%, 18 cm |
| 3 | 88%, 17 cm | 85%, 12 cm | 83%, 18 cm |
| 4 | 85%, 16 cm | 79%, 12 cm | 75%, 18 cm |
| 5 | 79%, 16 cm | 72%, 12 cm | 61%, 18 cm |

**Reading it:**

- Safety holds with a realistic ball: no contacts anywhere.
- At normal speed the closest passes shrink to about 12 cm. That is ball-to-ball clearance of about 5 cm, and it is tight.
- Slow speed with wide margins keeps passes at 18 cm or more, but finishes fewer jobs and spends more time waiting and stepping aside.

**Bugs found only with the realistic ball, all fixed in `sim/`:**

- Two balls could both end up "moving aside" for each other and wait forever.
- A ball moving aside that could not progress for more than 6 s now gives up and resumes its job.

## Morning data collection (about 15 minutes, before building traffic)

**Built overnight:** press **`r`** in the bench to record every tracked ball every frame to `runs/tracks/tracks_<time>.jsonl`. Each frame has:

- px and cm position, facing, lost/contended/bridged;
- connected and spinning;
- what manual driving sent, and the running job.

Press `r` again to stop. A red "REC tracks" readout shows in the top right while it runs. It only reads, and changes nothing about driving.

**Also built overnight: pursuit p2p.** Press **`o`** to switch p2p clicks to pursuit mode ("p2p: pursuit" in the top right). A click then:

1. re-zeros the ball's heading with a zero-power spin and stop, so nothing moves;
2. starts following a straight path to the target. There is no spin if the target is within 60° of where the ball faces; otherwise it turns to face the path first;
3. uses the calibrated steering direction when there is one.

Targets under 10 cm use the normal spin p2p. Press `o` again to go back. Retries and the run recorder work as for other paths.

**Try pursuit first:** about 5 clicks each with `o` off and on, with targets roughly ahead and behind.

Then the data collection, with `r` on for steps 2–6:

1. **Setup (you).** Charge both balls. Connect SK-914A and SK-4C2C. Press **white**, assign both balls, check exposure.
2. **Still noise (1 min).** Both balls still, more than 50 cm apart, for 30 s. Then 15 cm apart, then 8 cm apart, 10 s each. This gives real position and heading noise, and shows where the blobs merge or go CONTENDED.
3. **Second ball's follower (4 min).** On SK-4C2C: one line (~80 cm), one orbit (r 30, about 30 s), one patrol. This gives real follower accuracy for the worse ball, which today is only a model.
4. **Slow speed (3 min).** For each ball, one line at a lower speed setting. I add a temporary speed override for this. It shows whether 9 cm/s is actually reliable, since SK-4C2C's speed varied 62% leg to leg.
5. **Passing (4 min).** Park SK-4C2C. Drive SK-914A past it on a line at 40, 25 and 15 cm. Then drive both past each other in opposite directions (you on arrows, switching with Tab). This shows identity swaps, lost frames when close, and whether the bench stays responsive with two balls connected.
6. **Sanity p2p (1 min).** One p2p on each ball with both connected. This checks the Bluetooth command delay against the 0.22–0.37 s single-ball numbers.

After that, rerun `sim/` with the measured numbers and set the real traffic margins.

## Rerunning the simulation

From `scenarios/sim/`:

```bash
REAL=1 node expect.js realistic_normal
```

```bash
REAL=1 S=9 SEP=26 TMARG=1 PRED=20 node expect.js realistic_slow_wide
```

```bash
REAL=1 node realrun.js realistic_normal
```

`REAL=0` gives the clean ball.
