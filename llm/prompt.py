"""The system prompt, rebuilt fresh on every turn.

Two decisions carry most of the weight here.

**State is injected, not fetched.** The arena, the robots and their positions
go into the system prompt directly. `get_state` still exists, but a model that
has to spend a turn calling it before it can act is a model that takes twice as
long to do anything. The text is a few hundred tokens and always current.

**compute_points is pushed as the primary path.** This is the single highest
-leverage line in the file. Ask a small model for six points on a circle and it
returns points that are not on a circle, not evenly spaced, or contain a NaN —
coordinate arithmetic across many numbers is exactly what it is worst at.
Writing `[(cx+r*cos(i*2*pi/n), cy+r*sin(i*2*pi/n)) for i in range(n)]` is a
*language* task, which it is good at, and the sandbox does the arithmetic
exactly. Same result, radically lower error rate.
"""

import numpy as np

from tools.validate import MIN_SEPARATION

MAX_LISTED_OBSTACLES = 6
# Entities are user-editable and unbounded, the same growth risk obstacles are,
# so they get the same cap. The validator and the controller enforce every one
# whether or not it is named here.
MAX_LISTED_ENTITIES = 6

# Kept deliberately tight. Everything here earns its tokens; the guidance that
# survived is the guidance that changed behaviour in the evals.
RULES = """\
You command a swarm of small rolling robots on a flat floor.

Coordinates are centimetres. Origin is top-left, x increases right, y increases
down. Robots are addressed by their four-letter `code`, never by index or name.

HOW TO PLACE ROBOTS

`compute_points` is your primary tool. For any arrangement with structure —
circle, line, arc, grid, spiral, letter, wedge — write the formula rather than
the numbers, and set `execute: true`. You are far better at formulas.

It must assign `points`. Helper lines are fine, one per real newline. NEVER use
`;` or `#` — a comment swallows the rest of its line, including your
`points =`, and the call silently does nothing. n, cx, cy, xmin, xmax, ymin,
ymax, width, height ARE ALREADY DEFINED; never redefine them. Also sin, cos,
sqrt, pi, atan2, hypot, min, max, abs, round.

  "form a circle"
    expression: points = [(cx+60*cos(i*2*pi/n), cy+60*sin(i*2*pi/n)) for i in range(n)]
    execute: true

  "line up along the top"
    expression: points = [(xmin+30+i*(width-60)/(n-1), ymin+30) for i in range(n)]
    execute: true

Use `move_to` with literal coordinates only for arbitrary placement of a few
robots, or when a specific robot must go to a specific place:

  "put SSMK at the middle of the left wall"
    assign: {"SSMK": [20, 90]}

RULES THAT WILL BITE YOU

- State is below, fresh, every turn. Never call get_state or describe_scene
  first — go straight to the movement call.
- Return EXACTLY n points, one per robot. Build with `for i in range(n)`, never
  a hardcoded count. A wrong-sized set is rejected and costs a round trip.
- If the request names fewer places than robots ("the four corners" with six),
  still return n points: put the extras alongside, %(sep)dcm apart. Never fewer.
- Targets must be %(sep)dcm apart, inside the arena and clear of obstacles. Read
  the rejection reason and fix it — never repeat the same call.
- If a request is impossible (all robots in one spot), say so plainly.
- Always pass execute: true when the user wants the robots to move.
- After a move, call `wait_until_settled` before moving again — but never while
  continuous motion is running, because it cannot settle.
- To modify what you just made — bigger, rotated, shifted — use `transform`,
  do not recompute.
- "save that as X" / "remember this as X" means call `save_formation` with
  name X and nothing else. The robots are already in the shape — do not move
  them, recompute points, or call compute_points first.
- "do X again" / "recall X" means one `recall_formation` call. Size, place and
  angle are its `scale`, `center` and `rotation` arguments, not a new shape.

THE DRAGONS

The robots are named after House of the Dragon dragons. "the dragons",
"dragons" and any single name mean robots. A name and its code are the same
robot — Seasmoke is SSMK — and people will type either. Both are listed below.

DOCKING

"dock", "go home", "park" and "return" mean a saved formation, not a tool.
Call `list_formations` and recall the one that reads like home (cave, home,
dock). If there is none, say so and ask them to arrange the dragons and save
it first.

Keep replies to one short sentence. Do not narrate what you are about to do;
call the tool.
""" % {"sep": int(MIN_SEPARATION)}


# Appended only when the matching tools are actually in this turn's schema set.
# Naming a tool the model has no schema for is worse than not naming it: it
# calls the thing anyway (dispatch does not check what was offered), guesses
# the arguments, and burns the call budget failing. Observed: "go round and
# round in the middle" matches no motion trigger word, so no motion schema is
# sent — and the model spent all six calls reaching for set_flow regardless.
MOTION_RULES = """\
MOTION OVER TIME

`set_path` (waypoints), `set_flow` (a velocity field) and `follow`
(trail/surround/flank/mirror) run for a stretch of time rather than moving to a
place. They need a `duration`: if the user does not give one, choose a sensible
bound, use it, and say which you chose. `motion_control` lists, reweights and
removes what is running; `stop` ends everything."""

CONVENIENCE_RULES = """\
DO NOT COMPUTE WHAT A TOOL ALREADY COMPUTES

`swap`, `displace`, `nudge`, `gather`, `spread` and `mirror` take robot codes
and work out the geometry themselves. Reach for them rather than doing
arithmetic on the positions above — that is what they are for."""

_OPTIONAL_RULES = (
    (("set_path", "set_flow", "follow", "motion_control"), MOTION_RULES),
    (("swap", "displace", "nudge", "gather", "spread", "mirror"),
     CONVENIENCE_RULES),
)


def rules_for(available=None, rules=None):
    """The base rules plus whatever this turn's tools justify mentioning."""
    base = (rules or RULES).strip()
    if available is None:                       # everything, e.g. for tests
        return "\n\n".join([base] + [text for _, text in _OPTIONAL_RULES])
    have = set(available)
    parts = [base]
    for names, text in _OPTIONAL_RULES:
        if have & set(names):
            parts.append(text)
    return "\n\n".join(parts)


def _fmt(v):
    return f"{float(v):.0f}"


def arena_lines(ctx):
    ws = ctx.ws
    xmin, xmax, ymin, ymax = ws.bbox
    out = [f"Arena: x {_fmt(xmin)}..{_fmt(xmax)}, y {_fmt(ymin)}..{_fmt(ymax)} cm "
           f"(centre {_fmt((xmin + xmax) / 2)},{_fmt((ymin + ymax) / 2)})."]

    if ws.obstacles:
        # Obstacles are user-editable and unbounded, so this list is the one
        # part of the prompt a person can grow without limit. Cap it rather
        # than let the context quietly overflow; the validator still enforces
        # every obstacle whether or not it is named here.
        parts = []
        for o in ws.obstacles[:MAX_LISTED_OBSTACLES]:
            if o.get("type") == "circle":
                c, r = o["center"], o["radius"]
                parts.append(f"circle r{_fmt(r)} at ({_fmt(c[0])},{_fmt(c[1])})")
            else:
                pts = np.asarray(o["points"], dtype=float)
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                parts.append(f"block ({_fmt(lo[0])},{_fmt(lo[1])})-"
                             f"({_fmt(hi[0])},{_fmt(hi[1])})")
        line = "Obstacles (keep clear): " + "; ".join(parts)
        extra = len(ws.obstacles) - MAX_LISTED_OBSTACLES
        if extra > 0:
            line += (f"; and {extra} more — call get_state for the full list "
                     "before placing anything near them")
        out.append(line + ".")
    return out


def robot_lines(ctx):
    """Only robots that can actually be commanded.

    A disconnected robot in this list is an invitation to plan around a robot
    that will not move, and then report success.
    """
    states = ctx.fleet.state()
    active = [(c, d) for c, d in states.items() if d["connected"]]
    if not active:
        return ["No robots are connected — you cannot move anything right now."]

    # The name is carried alongside the code because users type names. Without
    # it the model has to guess that "Seasmoke" is SSMK — which it often gets
    # right from the spelling, and which fails silently the moment a robot is
    # renamed to something its code does not echo.
    bits = [f"{c}/{d['name']} ({_fmt(d['pos'][0])},{_fmt(d['pos'][1])})"
            for c, d in active]
    lines = [f"{len(active)} robots, n={len(active)}: " + ", ".join(bits) + "."]

    offline = [f"{c}/{d['name']}" for c, d in states.items() if not d["connected"]]
    if offline:
        lines.append(f"Offline, ignore: {', '.join(offline)}.")
    return lines


def entity_lines(ctx):
    """Entities, split by what the model can do with them.

    Without this the model cannot know an entity exists at all: state here is
    injected rather than fetched, and `arena_lines` only lists the static
    obstacle list. Asked to "follow the wanderer" against a prompt with no
    wanderer in it, a model does the reasonable thing and invents one — it
    reads the phrase as *a robot that wanders*, picks one, and drives it. The
    command appears to succeed and does something else entirely.
    """
    ents = list(getattr(ctx.ws, "entities", []))
    if not ents:
        return []

    lines = []
    followable = [e for e in ents if e.followable]
    blocking = [e for e in ents if e.blocks]

    def listed(group, fmt):
        shown = group[:MAX_LISTED_ENTITIES]
        text = ", ".join(fmt(e) for e in shown)
        extra = len(group) - len(shown)
        if extra > 0:
            text += f", and {extra} more — call get_state for the full list"
        return text

    def moving(e):
        return ", moving" if e.motion.get("kind", "static") != "static" else ""

    if followable:
        lines.append(
            "Followable (pass `follow` or `set_flow entity=` this id): "
            + listed(followable,
                     lambda e: f"{e.id} at ({_fmt(e.pos[0])},{_fmt(e.pos[1])})"
                               + moving(e)) + ".")
    if blocking:
        lines.append(
            "Moving obstacles (keep clear; they will not stay put): "
            + listed(blocking,
                     lambda e: f"{e.id} r{_fmt(e.radius)} at "
                               f"({_fmt(e.pos[0])},{_fmt(e.pos[1])})" + moving(e))
            + ".")
    return lines


def formation_lines(ctx):
    names = ctx.library.names()
    if not names:
        return ["Saved formations: none yet."]
    return ["Saved formations: " + ", ".join(names) + "."]


def memory_lines(ctx, command=None):
    """Precedents for what is being asked right now.

    Not a tool the model calls — it simply finds these already in front of it.
    Nothing is shown unless something is genuinely close, because a list of
    loosely related past commands is worse than none: it invites the model to
    copy the nearest thing rather than answer the question.
    """
    memory = getattr(ctx, "memory", None)
    if memory is None or not command:
        return []
    lines = memory.lines(command)
    if not lines:
        return []
    return ["Things that worked before for requests like this "
            "(a guide, not a rule):"] + lines


def build_system_prompt(ctx, rules=RULES, extra=None, available=None,
                        command=None):
    """The whole prompt as one string. One function so evals can diff variants.

    `available` is the set of tool names whose schemas are being sent this
    turn. The prompt must not advertise anything outside it.
    """
    parts = [rules_for(available, rules), "", "CURRENT STATE"]
    parts += arena_lines(ctx)
    parts += robot_lines(ctx)
    parts += entity_lines(ctx)
    parts += formation_lines(ctx)
    parts += memory_lines(ctx, command)
    if extra:
        parts += ["", extra.strip()]
    return "\n".join(parts)


def estimate_tokens(text):
    """Rough token count — good enough to keep a budget honest without a tokeniser."""
    return max(len(text) // 4, len(text.split()))
