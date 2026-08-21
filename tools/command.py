"""Typed tool calls: `move_to 60,40 120,40` and friends.

The same call surface a model will drive, reachable from a text box, so the
tool layer can be exercised by hand long before any model is wired up. Parsing
lives here rather than in the UI so it can be tested without a window.
"""

import shlex

from .registry import BY_NAME, call

# How bare words are read, per tool.
POSITIONAL = {
    "save_formation": ["name", "description"],
    "recall_formation": ["name"],
    "delete_formation": ["name"],
    "set_led": ["color"],
    "compute_points": ["expression"],
}

CODE_LIST_TOOLS = {"stop"}          # bare words are robot codes
POINT_TOOLS = {"move_to"}           # bare "x,y" pairs accumulate into `points`

VALUE_TYPES = {
    "center": "point", "translate": "point",
    "rotation": "number", "rotate": "number", "scale": "number",
    "timeout": "number", "tolerance": "number",
    "execute": "bool",
    "codes": "strlist",
    "color": "color",
    "name": "str", "description": "str", "expression": "str", "blink": "str",
}

# A word the user is likely to reach for, mapped to what the tool actually takes.
ALIASES = {
    "transform": {"rotation": "rotate", "move": "translate", "shift": "translate"},
    "recall_formation": {"rotate": "rotation", "at": "center", "size": "scale"},
    "wait_until_settled": {"tol": "tolerance"},
}


class ParseError(Exception):
    pass


def _number(text, field):
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ParseError(f"{field}: expected a number, got {text!r}")


def _point(text, field):
    parts = [p for p in text.replace(";", ",").split(",") if p != ""]
    if len(parts) != 2:
        raise ParseError(f"{field}: expected x,y — got {text!r}")
    return [_number(parts[0], field), _number(parts[1], field)]


def _looks_like_point(text):
    parts = text.split(",")
    if len(parts) != 2:
        return False
    try:
        float(parts[0]), float(parts[1])
    except ValueError:
        return False
    return True


def _coerce(key, text, tool):
    kind = VALUE_TYPES.get(key, "str")
    if kind == "point":
        return _point(text, key)
    if kind == "number":
        return _number(text, key)
    if kind == "bool":
        low = text.strip().lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0"):
            return False
        raise ParseError(f"{key}: expected true or false, got {text!r}")
    if kind == "strlist":
        return [p for p in text.replace(" ", "").split(",") if p]
    if kind == "color":
        if _looks_like_point(text):
            raise ParseError("color: an [r, g, b] colour needs three values")
        parts = text.split(",")
        if len(parts) == 3:
            return [int(_number(p, "color")) for p in parts]
        return text
    return text


def parse(text, codes=()):
    """Text in, (tool_name, arguments) out. Raises ParseError with a readable message."""
    try:
        tokens = shlex.split(text.strip())
    except ValueError as e:
        raise ParseError(f"could not read that line: {e}")
    if not tokens:
        raise ParseError("type a tool call, e.g. move_to 60,40 120,40 180,40")

    name, rest = tokens[0], tokens[1:]
    if name not in BY_NAME:
        raise ParseError(f"unknown tool {name!r} — try: {', '.join(sorted(BY_NAME))}")

    aliases = ALIASES.get(name, {})
    accepted = set(BY_NAME[name]["parameters"].get("properties", {}))

    if name == "compute_points":
        # An expression is a blob of code, not a token stream — it is full of
        # '=' and spaces, so anything that is not the execute flag is source.
        args, source = {}, []
        for tok in rest:
            if tok.startswith("execute="):
                args["execute"] = _coerce("execute", tok.split("=", 1)[1], name)
            else:
                source.append(tok)
        if source:
            args["expression"] = " ".join(source)
        return name, args

    args = {}
    points = []
    assign = {}
    positional = []
    codes = set(codes)

    for tok in rest:
        if "=" in tok and not tok.startswith("="):
            key, value = tok.split("=", 1)
            key = aliases.get(key, key)

            if key in codes or (name == "move_to" and key.isupper() and key not in accepted):
                assign[key] = _point(value, key)
                continue

            if key not in accepted:
                near = ", ".join(sorted(accepted)) or "no arguments"
                raise ParseError(f"{name} has no argument {key!r} — accepts {near}")
            args[key] = _coerce(key, value, name)
            continue

        if name in POINT_TOOLS:
            if _looks_like_point(tok):
                points.append(_point(tok, "points"))
                continue
            raise ParseError(f"{name} expects points as x,y — got {tok!r}")

        positional.append(tok)

    if points:
        args["points"] = points
    if assign:
        args["assign"] = assign

    if positional:
        if name in CODE_LIST_TOOLS:
            args["codes"] = positional
        else:
            slots = POSITIONAL.get(name)
            if not slots:
                raise ParseError(f"{name} takes no bare words, got {positional[0]!r}")
            if len(positional) > len(slots):
                raise ParseError(f"{name} takes at most {len(slots)} bare word(s), "
                                 f"got {len(positional)}")
            for slot, value in zip(slots, positional):
                args.setdefault(slot, _coerce(slot, value, name))

    return name, args


def parses_as_tool(text, codes=()):
    """Is this line a literal tool call, rather than something to ask a model?

    Used by the UI to decide whether a command typed while the agent is busy
    can be run directly. Parse-only: nothing is executed and nothing moves.
    """
    try:
        parse(text, codes=codes)
        return True
    except ParseError:
        return False


def run(ctx, text):
    """Parse and execute one line. Returns the tool result, errors included."""
    from .result import fail
    try:
        name, args = parse(text, codes=ctx.fleet.codes)
    except ParseError as e:
        return fail(ctx, str(e))
    return call(ctx, name, args)


def summarise(result):
    """One short line for the command log."""
    if not result.get("ok"):
        return f"error: {result.get('error')}"

    bits = []
    if "targets" in result:
        bits.append(f"{len(result['targets'])} target(s) set")
    if result.get("clamped"):
        bits.append(f"{len(result['clamped'])} clamped")
    if "settled" in result:
        bits.append("settled" if result["settled"] else "timed out")
        if result.get("waited_s") is not None:
            bits.append(f"{result['waited_s']:.1f}s")
    if "description" in result:
        return result["description"]
    if "formations" in result and isinstance(result["formations"], list):
        names = [f["name"] if isinstance(f, dict) else str(f)
                 for f in result["formations"]]
        bits.append("formations: " + (", ".join(names) or "none"))
    if "stopped" in result:
        bits.append("stopped " + (", ".join(result["stopped"]) or "nothing"))
    if "colored" in result:
        bits.append("led " + ", ".join(result["colored"]))
    if "name" in result and "targets" not in result:
        bits.append(f"saved {result['name']}")
    if "points" in result and "targets" not in result:
        bits.append(f"{len(result['points'])} point(s) computed")
    if result.get("note"):
        bits.append(result["note"])
    return "ok" + (" — " + "; ".join(bits) if bits else "")
