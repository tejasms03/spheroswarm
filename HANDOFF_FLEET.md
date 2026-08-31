# The camera benches — tracking a ball by its light, and driving it

A handoff for `pose_test.py`, `blob_test.py`, `swarm_test.py` and
`fleet_test.py` — the line of work from 27–30 August 2026. Paste into a fresh
chat alongside `HANDOFF_CALIB.md` (the calibration bench) and `HANDOFF.md`
(the stack underneath).

None of the other four handoffs mention these files. This is the only record.

---

## READ THIS FIRST

**Nothing in this line of work is committed.** `fleet_test.py`, `blob_test.py`,
`swarm_test.py`, `pose_test.py` and their tests are all untracked. There is no
`git log` to fall back on and no diff to read — if a file is lost it is lost.
Commit before doing anything else.

**Do not run cleanup globs over `runs/`.** In this session I ran
`rm -f runs/fleet_*.csv` to tidy up my own test artefacts and destroyed the
operator's real session logs — the exact data we then needed to diagnose a
fault. The `.png` frames survived only because most of those globs said
`*.csv`. Delete by explicit filename or not at all.

**The recurring theme, again.** As in `HANDOFF_SESSION.md`: almost every fault
here was a *measurement lying*, and almost every one first presented as a
controller problem. A ball pinned against a wall, an agent that gave up, a
patrol that oscillated at one end — three separate "the control is wrong"
reports, three measurement bugs.

```bash
cd ~/spheroswarm
python3.13 fleet_test.py                       # camera + robot + agent
python3.13 fleet_test.py --source sim          # no camera, no robot
python3.13 fleet_test.py --model qwen3.5:4b    # a smaller local model
python3.13 -m pytest tests/test_fleet.py tests/test_blob.py -q   # 246 tests
```

`python3.13`, never `python3` — the default is 3.14 with no cv2 and a broken
pip.

---

## 1. Four benches, and which one to touch

They are forks of each other, kept deliberately rather than refactored into
one. Each earlier one is a **working fallback** for the next, and the operator
asked for them to stay that way.

| file | what it does | status |
|---|---|---|
| `pose_test.py` | three lit dots → position *and* heading, identity by colour | reference |
| `blob_test.py` | one ball as one blob, no colour, no heading | **FROZEN** fallback |
| `swarm_test.py` | the same, several robots, roll-call naming | **FROZEN** fallback |
| `fleet_test.py` | swarm_test plus an LLM tool layer | **ACTIVE** |

Work in `fleet_test.py`. The frozen ones exist so that when the active bench
breaks mid-session there is something that still drives a ball.

`pose_test.py` is far more accurate — 100% read rate, 0.12 mm and 0.17° median
against generated truth — and is not what the rig uses, because three dots are
only visible when all three are visible. The blob method answers on frames
where the dot method refuses. Availability beat precision.

---

## 2. What the camera actually sees

**The ball is a halo, not a disc.** The lit region is roughly 20 cm across —
measured radius 9.9 cm — against a 7.3 cm shell. Everything that reasons about
the ball's size must use the halo, because the halo is what the mask cuts and
what the threshold finds.

**The threshold goes LOW, not high.** `V_MIN` is 20. A high threshold splits
the halo into several components; a low one keeps it as a single blob. This is
the opposite of the instinct and it is why `find_blobs` looks the way it does.

**The centroid is weighted by height ABOVE the threshold**, not by raw value.
Weighting by raw value makes the answer depend on where the threshold sits,
because every pixel carries a constant `v_min` of dead weight. Subtracting it
first cut centroid wander by 3.5× over a threshold drag from 170 to 230.

**The taillight biases the centroid** by about 7 mm in a way that rotates with
heading, so it cannot be calibrated out with a constant. It is forced off.

---

## 3. The wall problem — a measurement bug, not a control bug

**Reported symptom:** the ball gets stuck at edges and corners; the agent
drives it into a wall and keeps driving.

**Cause:** `find_blobs` applied the workspace mask to the *pixels*, before
measuring anything. A ball at the boundary kept only its inner half, and the
intensity-weighted centroid was therefore dragged **inward, away from the
wall**. Measured by sliding a boundary across a real ball from a saved frame:

| boundary vs ball centre | blob kept | reported error |
|---|---|---|
| +30 px (8.5 cm clear) | 94 % | 0.05 cm |
| +15 px | 74 % | 1.4 cm |
| 0 px (edge on centre) | 50 % | **3.6 cm** |
| −20 px | 19 % | **7.1 cm** |
| −40 px | — | **track lost** (under `MIN_AREA`) |

The ball only reads true when it is **8.5 cm clear of the line**. Inside that,
the reported position *saturates*: the ball keeps moving out, the reported
centre barely does. The loop sees commanded motion producing no measured
motion and keeps driving.

All nine tracked frames the operator saved were at 18–53 % kept, biased
**3.0–6.1 cm** inward. Every one of them was a stuck-at-the-wall moment.

**Second-order, and worse:** `goal_margin_cm` calls `blob_radius_cm`, which
measured the *current* blob. A clipped blob is smaller, so the safety margin
**shrank exactly as the ball got pinned** — 15.9 cm whole down to 10.3 cm at
19 % kept. Position error and margin error pointed the same way and added to
roughly 12 cm of optimism.

**The fix.** Detection now runs on a region grown by one halo radius, and the
true region only decides whether a blob's *centre* is in bounds. The halo stays
whole so the centroid is the ball's own; a lamp outside the arena still cannot
contribute, because the grown mask is one radius wider and not unbounded. On
the saved frames this recovers 1.9–4.1× more of the blob and removes
2.8–8.7 cm of bias. `blob_radius_cm` now holds the last measurement taken while
the blob was *whole*.

**A design decision worth keeping.** The first version *discarded* blobs whose
centre was outside the workspace. On the saved frames that turned six
stuck-at-the-wall moments into "no ball at all" — which loses the track and
leaves nothing to steer back in. It flags `outside_region` and keeps the
position instead. You cannot recover a ball you have stopped being able to see.

**Also seen in those frames:** the last three (19:21–19:22, 29 Aug) are fully
blown out — mean V 165, the entire frame above threshold, zero blobs. Room
lights. The tracker was blind, not confused. There is no guard for this yet.

---

## 4. The agent gave up before it got anywhere

Two independent bugs, both in the reporting rather than the driving. The tools
*do* drive the same controller as the buttons — `agent_call` sets `self.path`
and calls `self.arm()`, which is exactly what GO does.

**"Arrived" meant "stopped".** `wait_until_arrived` returned
`{"arrived": not self.armed}`. Every way of stopping clears `armed` — arriving,
giving up stuck, losing the ball, pressing esc — so all of them reported
success. The controller *knew* better: `disarm` was already working out a
verdict for the CSV and throwing it away.

```
before:  {'arrived': True,  'position_cm': [93.7, 45.1]}      # 20cm short
after:   {'arrived': False, 'outcome': 'stuck',
          'reason': 'stuck after 2 attempts to back off — move it clear…',
          'retry_hint': 'the escape budget is refreshed on the next drive…'}
```

**The escape budget never reset.** `self.unstick` was set to `None` only in
`__init__`. `step_unstick` deliberately keeps the count so that two failures in
one run give up rather than nudging forever — but nothing cleared it
afterwards, so it was **per process, not per run**:

| | tries at arm | escapes | time before giving up |
|---|---|---|---|
| run 1 | 0 | 2 | 12.6 s |
| run 2 (before fix) | 2 | **0** | **4.0 s** |
| run 2 (after fix) | 0 | 2 | 12.8 s |

From the agent's second `goto` onward the first stall was fatal.

**How these chain with §3.** `stall_report` fires when `true_speed()` reads
under 6 cm/s. Near a boundary the clipped centroid saturates, so measured speed
collapses toward zero *while the ball is moving*. The clipping manufactures a
false stall; the counter bug removes the escapes that would recover from it;
the reporting bug tells the model it arrived. That is the whole path from "ball
near a wall" to "the agent quits having gone nowhere".

---

## 5. "Patrol an edge" oscillated around one corner

**Reported symptom:** asked to patrol an edge, the ball oscillated around the
corner instead of running between the two ends.

The model was not at fault — it built a sensible bottom-edge route at y = −15
(the workspace spans y −26.3 .. 95.6, so that is a real edge).

**From the log:** the ball reached the far end x ≈ 112 at t = 29 s and for the
next **264 s never went back past x ≈ 40**, reversing direction **1379 times,
one every 0.4 s**. `gap` reached 0.2 cm repeatedly; the run never disarmed.

**The arithmetic that explains it.** While the ball was on the route the
lookahead target obeyed `target_x = 209 − ball_x` — moving *backward* as the
ball moved forward. With the turning vertex at x = 112 and a 15 cm lookahead,
`2 × 112 − 15 = 209`. The lookahead was stepping **past the turn** and landing
on the returning leg, so the controller aimed the ball back the way it came
before it had reached the end.

Two things follow, and they are different:

**(a) `Path.project` could not tell the legs apart.** An out-and-back route
lays both legs on the same line, so the ball is exactly equidistant from both
and which one wins is decided by the last bits of a float — it flips frame to
frame. The docstring had chosen a global search deliberately, to avoid getting
stuck on the wrong lap of a circle; that choice is precisely wrong for a route
that retraces itself. **Fixed** with a progress ratchet: `project(p, near=…)`
windows the search to the arc length already travelled (8 cm back, 80 cm
forward), and re-locks globally past 40 cm off-path, because a ball that has
been picked up and moved did not drive there.

**(b) The backward lookahead is NOT fixed, and cannot be.** On a route that
genuinely turns round, a lookahead that steps past the vertex is *correct* —
the route really does reverse 15 cm ahead. Aiming around a 180° reversal is
something pure pursuit cannot do. **I told the operator the ratchet would fix
this. It does not.** This is pinned as
`test_a_retracing_path_aims_the_lookahead_BACKWARDS_near_its_turn` so nobody
rebuilds a patrol the obvious way.

**Which is why a patrol is a run-level state machine, not a path shape.**
`start_patrol` / `step_patrol` drive A→B as an ordinary one-way line; arriving
swaps the ends and drives back. Each leg has no reversal in it, so there is no
ambiguity to resolve and arriving is an ordinary arrival. It is tied to the
path *object*, so drawing a new route or calling any other tool simply ends the
patrol — there is no flag to remember to clear.

Measured in the sim: **15 legs in 60 s, turning at x ≈ 103 and x ≈ 36 between
ends 110 and 30.** The ~7 cm shortfall is the arrival reach, and it is
symmetric.

**The log now records the route.** New `path_pts` column, written once per run,
capped at 200 points with the truncation stated. Only the path's *kind* used to
be recorded, and "polyline" was not enough to answer the question — the vertex
above had to be reconstructed from arithmetic, and the exact point list was
never recovered at all.

---

## 6. Mistakes of mine worth carrying

1. **I deleted the operator's run logs** with `rm -f runs/fleet_*.csv`, tidying
   my own artefacts with a glob that also matched theirs. Then had to diagnose
   the fault from twelve surviving `.png` frames. Delete by explicit filename.
2. **I claimed the ratchet would fix the backward lookahead.** It addresses a
   different half of the problem. Stating a fix before measuring it is how a
   session gets spent on the wrong suspect — see §5(b).
3. **I once corrupted `blob_test.py` to 341 MB** with a `str.replace` whose
   `old` was effectively empty, inserting a block between every character. Use
   exact-match edits, and never a replace whose target could be empty.
4. **My first patrol test failed for a reason that was not the code.**
   `_app_with_ball()` loads the operator's live `calib/blob_region.json`, so
   the ball drove outside a workspace left over from their camera. See §7.

---

## 7. Still open

- **The frame is MIRRORED.** The probe reported REFLECTION at 188° on 29 Aug.
  `arm()` refuses while `self.mirrored`, and it is right to: a reflection maps
  a commanded course `c` to a measured `2α − c`, so the error changes sign with
  direction and **every** controller diverges. Nothing downstream can converge
  until `flip y` → re-probe comes back clean. This is the same fault, on a
  different bench, as §1 of `HANDOFF_SESSION.md`.
- **Test isolation.** `_app_with_ball()` loads the live
  `calib/blob_region.json`, so the suite silently inherits whatever workspace
  was last clicked and will shift underfoot on recalibration. The app itself is
  fine — `click_view` clears the cached mask. Pin the tests to a fixed
  workspace.
- **Blown-out frames are not detected.** Three saved frames had the whole image
  above threshold and zero blobs. Nothing says "the lights came on".
- **Glare can outrank the ball.** The grown detection mask admits a band of
  scenery one halo wide. In one frame that band held a dim patch (area 4935,
  peak 40) that sorted *above* the real ball (area 3433, peak 255), because
  `find_blobs` sorts on area alone. Physical walls now keep the ball in bounds,
  so the clean guard is to sort in-bounds blobs first. Not done — it was
  offered and the operator moved on.
- **The agent tool layer is single-robot.** `get_state` and `goto` act on
  `self.code`. The bench drives several; the tools address one.
- **`run_outcome` is blank on most logged runs**, because arming a new path
  while already armed never disarms the previous run, so it never gets its end
  row.
- **The professor's framework** (`Mobile-manipulation-with-VLMs`) has not been
  looked at. The operator asked whether it can run on this stack, expecting
  navigation and tracking to need replacing.

---

## 8. The order to work in

1. **Commit everything.** None of it is tracked. Nothing below is safe until it
   is.
2. **`flip y`, then re-probe** until the probe reports a rotation and not a
   reflection. Nothing converges before this and no measurement taken before it
   is worth keeping.
3. Zero the aim, drive a straight line, and confirm the ball goes where it is
   sent.
4. Characterise CRXS. The battery has SYRX (tau 0.84 s, coast 0.509 s solid;
   dead time disputed between 0.433 s and 0.0, recorded in
   `delay_agreement_s`). CRXS FAILED its battery — it drove 12 cm *away* from
   the middle, which is the aim frame being wrong rather than the drive.
5. Only then put the agent on hardware.
