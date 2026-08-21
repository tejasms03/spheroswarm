"""The tool registry: JSON schemas for function calling, and a dispatcher.

Deliberately absent: motor and heading commands, BLE connect/disconnect, roster
mutation, and anything touching files or a shell. A model gets to arrange
robots and remember arrangements. It does not get to reconfigure the rig.
"""

from . import (convenience, expression, generate, library, motion, movement,
               sensing, timing)

POINT = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
    "description": "[x, y] in centimetres",
}

TOOLS = [
    {
        "name": "get_state",
        "description": (
            "Every robot's code, name, position, velocity, LED, connection and "
            "kind, plus the arena bounds, the obstacles, and the names of saved "
            "formations. Call this before planning any arrangement."),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "fn": lambda ctx, **kw: sensing.get_state(ctx),
    },
    {
        "name": "describe_scene",
        "description": (
            "The same information as get_state, written as a short prose "
            "summary. Cheaper to read and easier to reason over than raw "
            "coordinate arrays."),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "fn": lambda ctx, **kw: sensing.describe_scene(ctx),
    },
    {
        "name": "move_to",
        "description": (
            "Move robots to a set of points, in centimetres. This is the only "
            "movement primitive: every arrangement, however simple or strange, "
            "is expressed as a set of points that you compute yourself. Give "
            "`points` as an unordered list and robots are assigned to their "
            "nearest slots; give `assign` to pin named robots to exact places; "
            "give both to pin some and let the rest fill in."),
        "parameters": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": POINT,
                    "description": "Unordered target points; robots take the nearest.",
                },
                "assign": {
                    "type": "object",
                    "additionalProperties": POINT,
                    "description": 'Exact placements by code, e.g. {"SSMK": [100, 80]}.',
                },
            },
            "required": [],
        },
        "fn": lambda ctx, **kw: movement.move_to(ctx, **kw),
    },
    {
        "name": "transform",
        "description": (
            "Translate, rotate and/or scale the current arrangement, keeping "
            "each robot in its slot. Use this for 'shift it left', 'spin it', "
            "'spread out' rather than recomputing every point."),
        "parameters": {
            "type": "object",
            "properties": {
                "translate": {**POINT, "description": "[dx, dy] shift in centimetres"},
                "rotate": {"type": "number",
                            "description": "degrees clockwise about the arrangement's centre"},
                "scale": {"type": "number",
                           "description": "multiplier about the centre; >1 spreads out"},
            },
            "required": [],
        },
        "fn": lambda ctx, **kw: movement.transform(ctx, **kw),
    },
    {
        "name": "stop",
        "description": "Halt robots immediately. Omit `codes` to stop all of them.",
        "parameters": {
            "type": "object",
            "properties": {
                "codes": {"type": "array", "items": {"type": "string"},
                           "description": "Robot codes; omit for the whole fleet."},
            },
            "required": [],
        },
        "fn": lambda ctx, **kw: movement.stop(ctx, **kw),
    },
    {
        "name": "wait_until_settled",
        "description": (
            "Block until every robot is within `tolerance` of its target, or "
            "until `timeout`. Call this between steps of a sequence, otherwise "
            "the next command fires into a swarm that is still moving."),
        "parameters": {
            "type": "object",
            "properties": {
                "tolerance": {"type": "number", "description": "cm, default 8"},
                "timeout": {"type": "number", "description": "seconds, default 20"},
            },
            "required": [],
        },
        "fn": lambda ctx, **kw: timing.wait_until_settled(ctx, **kw),
    },
    {
        "name": "set_led",
        "description": (
            "Set robot colour, optionally with a blink pattern. Colour is a "
            "name (red, yellow, green, cyan, blue, magenta, white, orange, "
            "purple, pink, off) or an [r, g, b] triple."),
        "parameters": {
            "type": "object",
            "properties": {
                "color": {"description": "colour name or [r, g, b]"},
                "codes": {"type": "array", "items": {"type": "string"},
                           "description": "Robot codes; omit for all."},
                "blink": {"type": "string", "enum": ["none", "slow", "fast", "pulse"]},
            },
            "required": ["color"],
        },
        "fn": lambda ctx, **kw: expression.set_led(ctx, **kw),
    },
    {
        "name": "save_formation",
        "description": (
            "Save the current arrangement under a name. Stored normalised — "
            "centred and scaled — so it can be recalled anywhere in the arena, "
            "at any size and any angle."),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string",
                                 "description": "optional note about what this shape is"},
            },
            "required": ["name"],
        },
        "fn": lambda ctx, **kw: library.save_formation(ctx, **kw),
    },
    {
        "name": "recall_formation",
        "description": (
            "Place a saved formation, optionally at a given centre, scale and "
            "rotation. Requires the same number of robots it was saved with."),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "center": {**POINT, "description": "where to centre it; defaults to the arena centre"},
                "scale": {"type": "number",
                           "description": "radius in cm of the furthest robot; defaults to the saved size"},
                "rotation": {"type": "number", "description": "degrees clockwise"},
            },
            "required": ["name"],
        },
        "fn": lambda ctx, **kw: library.recall_formation(ctx, **kw),
    },
    {
        "name": "list_formations",
        "description": "Names, descriptions, robot counts and creation times of saved formations.",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "fn": lambda ctx, **kw: library.list_formations(ctx),
    },
    {
        "name": "delete_formation",
        "description": "Forget a saved formation.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "fn": lambda ctx, **kw: library.delete_formation(ctx, **kw),
    },
    {
        "name": "compute_points",
        "description": (
            "Evaluate a short Python expression that builds a list of (x, y) "
            "points and assigns it to `points`. For parametric arrangements "
            "over many robots. Available: n, width, height, cx, cy, xmin, xmax, "
            "ymin, ymax, and basic maths (sin, cos, sqrt, pi, ...). No imports, "
            "no while loops, one second. Prefer literal coordinates with "
            "move_to for anything small."),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                                "description": "e.g. points = [(cx + 60*cos(i*2*pi/n), cy + 60*sin(i*2*pi/n)) for i in range(n)]"},
                "execute": {"type": "boolean",
                             "description": "move the robots there as well as returning the points; pass true whenever the robots should actually move"},
            },
            "required": ["expression"],
        },
        "fn": lambda ctx, **kw: generate.compute_points(ctx, **kw),
    },
    {
        "name": "set_path",
        "description": (
            "Walk robots along waypoints over time: patrols, sweeps, laps. "
            "`assignments` maps each robot code to its list of points. `mode` "
            "is once, loop or pingpong. Continuous modes need a duration."),
        "parameters": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "object",
                    "additionalProperties": {"type": "array", "items": POINT},
                    "description": 'e.g. {"SSMK": [[40,40],[200,40]]}',
                },
                "mode": {"type": "string", "enum": ["once", "loop", "pingpong"]},
                "speed": {"type": "number", "description": "cm/s; omit for full speed"},
                "duration": {"type": "number", "description": "seconds, max 300"},
                "append": {"type": "boolean",
                            "description": "extend the existing path instead of replacing it"},
            },
            "required": ["assignments"],
        },
        "fn": lambda ctx, **kw: motion.set_path(ctx, **kw),
    },
    {
        "name": "set_flow",
        "description": (
            "Drive robots by a velocity field: an expression in x, y and t "
            "assigning vx and vy. Use for orbits, drifts and swirls. Give "
            "`entity` to bind ex/ey to something that moves, so the field "
            "follows it. Rejected if it would drive anyone out of the arena."),
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string",
                          "description": "e.g. vx = -(y-cy)*0.5\nvy = (x-cx)*0.5"},
                "robots": {"type": "array", "items": {"type": "string"},
                            "description": "codes; omit for all"},
                "entity": {"type": "string",
                            "description": "bind ex/ey to this entity or robot"},
                "weight": {"type": "number", "description": "blend weight, default 0.6"},
                "duration": {"type": "number", "description": "seconds, max 300"},
                "when": {"type": "string",
                          "description": "a condition re-checked every tick; the "
                                         "layer only applies while it is true. "
                                         "Over x, y, t, speed, d_target, "
                                         "d_nearest, rank, elapsed, ex, ey, cx, "
                                         "cy, n. e.g. 'd_target < 80', "
                                         "'rank == 0', 'speed < 3'."},
            },
            "required": ["expr"],
        },
        "fn": lambda ctx, **kw: motion.set_flow(ctx, **kw),
    },
    {
        "name": "follow",
        "description": (
            "Follow a robot or a followable entity. Modes: trail (behind), "
            "surround (ring at `distance`), flank (to the sides), mirror "
            "(copy its velocity). Aims ahead of the target, not at it."),
        "parameters": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string",
                               "description": "a robot code or an entity id"},
                "followers": {"type": "array", "items": {"type": "string"},
                               "description": "codes; omit for all"},
                "mode": {"type": "string",
                          "enum": ["trail", "surround", "flank", "mirror"]},
                "distance": {"type": "number", "description": "cm, default 40"},
                "duration": {"type": "number", "description": "seconds, max 300"},
                "when": {"type": "string",
                          "description": "a condition re-checked every tick; the "
                                         "layer only applies while it is true. "
                                         "Over x, y, t, speed, d_target, "
                                         "d_nearest, rank, elapsed, ex, ey, cx, "
                                         "cy, n. e.g. 'd_target < 80', "
                                         "'rank == 0', 'speed < 3'."},
                "select": {"type": "string",
                            "description": "re-decide WHO follows every tick "
                                           "instead of fixing it now: 'nearest' "
                                           "or 'nearest:2'. Use for 'whichever "
                                           "robot is closest'."},
            },
            "required": ["target_id"],
        },
        "fn": lambda ctx, **kw: motion.follow(ctx, **kw),
    },
    {
        "name": "motion_control",
        "description": (
            "Manage running motion. action=list shows every active layer per "
            "robot; set_weight reweights one; remove drops one by name; clear "
            "removes all of them."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                            "enum": ["list", "set_weight", "remove", "clear"]},
                "codes": {"type": "array", "items": {"type": "string"},
                           "description": "robots to act on; omit for all"},
                "name": {"type": "string", "description": "layer name, e.g. path or flow"},
                "weight": {"type": "number"},
            },
            "required": ["action"],
        },
        "fn": lambda ctx, **kw: motion.motion_control(ctx, **kw),
    },
    {
        "name": "swap",
        "description": (
            "Exchange two robots' positions. Give the two codes; the geometry "
            "is worked out for you — do not compute coordinates for this."),
        "parameters": {
            "type": "object",
            "properties": {"code_a": {"type": "string"}, "code_b": {"type": "string"}},
            "required": ["code_a", "code_b"],
        },
        "fn": lambda ctx, **kw: convenience.swap(ctx, **kw),
    },
    {
        "name": "displace",
        "description": (
            "code_a takes code_b's place, and code_b steps aside to a free "
            "spot nearby. Use for 'put X where Y is'."),
        "parameters": {
            "type": "object",
            "properties": {"code_a": {"type": "string"}, "code_b": {"type": "string"}},
            "required": ["code_a", "code_b"],
        },
        "fn": lambda ctx, **kw: convenience.displace(ctx, **kw),
    },
    {
        "name": "nudge",
        "description": (
            "Shift robots a short way without computing a destination. "
            "direction is left, right, up, down, toward:<id> or away:<id>, "
            "where <id> is a robot code or entity id."),
        "parameters": {
            "type": "object",
            "properties": {
                "codes": {"type": "array", "items": {"type": "string"},
                           "description": "robots to move; omit for all"},
                "direction": {"type": "string"},
                "distance": {"type": "number", "description": "cm, default 30"},
            },
            "required": ["direction"],
        },
        "fn": lambda ctx, **kw: convenience.nudge(ctx, **kw),
    },
    {
        "name": "gather",
        "description": "Cluster robots in a ring around a robot or entity.",
        "parameters": {
            "type": "object",
            "properties": {
                "codes": {"type": "array", "items": {"type": "string"}},
                "around": {"type": "string",
                            "description": "robot code or entity id at the centre"},
                "radius": {"type": "number", "description": "cm, default 45"},
            },
            "required": ["around"],
        },
        "fn": lambda ctx, **kw: convenience.gather(ctx, **kw),
    },
    {
        "name": "spread",
        "description": (
            "Push robots apart to at least min_distance, without naming any "
            "destinations."),
        "parameters": {
            "type": "object",
            "properties": {
                "codes": {"type": "array", "items": {"type": "string"}},
                "min_distance": {"type": "number", "description": "cm, default 60"},
            },
            "required": [],
        },
        "fn": lambda ctx, **kw: convenience.spread(ctx, **kw),
    },
    {
        "name": "mirror",
        "description": (
            "Place code_a as code_b's reflection. axis is vertical, horizontal, "
            "centre, or through:<id> for a point reflection through a robot or "
            "entity — that last one puts code_a diametrically opposite."),
        "parameters": {
            "type": "object",
            "properties": {
                "code_a": {"type": "string", "description": "the robot that moves"},
                "code_b": {"type": "string", "description": "the one reflected"},
                "axis": {"type": "string",
                          "description": "vertical | horizontal | centre | through:<id>"},
            },
            "required": ["code_a", "code_b"],
        },
        "fn": lambda ctx, **kw: convenience.mirror(ctx, **kw),
    },
]

BY_NAME = {t["name"]: t for t in TOOLS}

# Schemas are the largest single item in every prompt: at 22 tools they are
# ~2,660 tokens, roughly four times the system prompt, re-sent on every round
# trip. They also make selection harder — more options, more wrong turns.
#
# CORE is what almost every command needs. The rest are loaded when the request
# plainly calls for them, so a "form a circle" never pays for the follow and
# mirror schemas.
CORE_TOOLS = (
    "move_to", "compute_points", "transform", "stop", "wait_until_settled",
    "set_led", "save_formation", "recall_formation", "list_formations",
    "delete_formation", "get_state", "describe_scene",
)

# word -> extra tools that word implies
_TRIGGERS = {
    "motion": ("set_path", "set_flow", "follow", "motion_control"),
    "convenience": ("swap", "displace", "nudge", "gather", "spread", "mirror"),
}

_MOTION_WORDS = ("orbit", "patrol", "follow", "chase", "circle around", "flow",
                 "swirl", "drift", "path", "route", "loop", "sweep", "trail",
                 "surround", "flank", "mirror the", "keep moving", "spin around",
                 "stop orbit", "stop following", "layer", "forever", "figure",
                 "back and forth", "lap", "track")
_CONVENIENCE_WORDS = ("swap", "exchange", "switch place", "displace", "nudge",
                      "a bit", "slightly", "shift", "gather", "cluster",
                      "spread", "apart", "mirror", "opposite", "reflect",
                      "where", "trade place")


def _strip(tool):
    return {k: v for k, v in tool.items() if k != "fn"}


_ALL_SCHEMAS = None
_CORE_SCHEMAS = None


def schemas(text=None, full=False):
    """JSON schemas for a function-calling API.

    `text` is the user's command; when given, only the core set plus whatever
    that command implies is returned. Built once and cached — the payload is
    identical on every call and rebuilding it per request is pure waste.
    """
    global _ALL_SCHEMAS, _CORE_SCHEMAS
    if _ALL_SCHEMAS is None:
        _ALL_SCHEMAS = [_strip(t) for t in TOOLS]
        _CORE_SCHEMAS = [s for s in _ALL_SCHEMAS if s["name"] in CORE_TOOLS]

    if full or text is None:
        return _ALL_SCHEMAS

    low = str(text).lower()
    wanted = set(CORE_TOOLS)
    if any(w in low for w in _MOTION_WORDS):
        wanted.update(_TRIGGERS["motion"])
    if any(w in low for w in _CONVENIENCE_WORDS):
        wanted.update(_TRIGGERS["convenience"])
    if wanted == set(CORE_TOOLS):
        return _CORE_SCHEMAS
    return [s for s in _ALL_SCHEMAS if s["name"] in wanted]


def call(ctx, name, arguments=None):
    """Dispatch by name. Unknown tools and bad arguments come back as errors."""
    from .result import fail

    tool = BY_NAME.get(name)
    if tool is None:
        return fail(ctx, f"unknown tool {name!r} — available: "
                         f"{', '.join(sorted(BY_NAME))}")

    args = arguments or {}
    if not isinstance(args, dict):
        return fail(ctx, f"arguments must be an object, got {type(args).__name__}")

    allowed = set(tool["parameters"].get("properties", {}))
    unexpected = set(args) - allowed
    if unexpected:
        return fail(ctx, f"{name} got unexpected argument(s): "
                         f"{', '.join(sorted(unexpected))}; accepts "
                         f"{', '.join(sorted(allowed)) or 'no arguments'}")

    missing = set(tool["parameters"].get("required", [])) - set(args)
    if missing:
        return fail(ctx, f"{name} is missing required argument(s): "
                         f"{', '.join(sorted(missing))}")

    try:
        return tool["fn"](ctx, **args)
    except Exception as e:
        return fail(ctx, f"{name} failed: {type(e).__name__}: {e}")
