"""Everything between a model's arithmetic and robots on a floor.

Every target set passes through `validate_targets` before anything moves, no
matter what produced it — a literal list, a transform, a recalled formation, or
sandboxed code. Nothing in this module raises: a controller that crashes on bad
input is worse than one that reports what was wrong, because the model can act
on a report.

Models emit NaN, strings, nulls, wrong counts and points on top of each other
far more often than you would expect, so each of those is a named, tested case
rather than an afterthought.
"""

import ast
import json
import math
import subprocess
import sys

import numpy as np

MIN_SEPARATION = 20.0      # cm between any two targets
TARGET_CLEARANCE = 12.0    # cm a target must keep off walls and obstacles
SANDBOX_TIMEOUT = 1.0      # s


def _finite_point(p):
    """(x, y) floats, or None. Rejects NaN, inf, strings, nulls, wrong arity."""
    if isinstance(p, dict):
        if "x" not in p or "y" not in p:
            return None
        p = (p["x"], p["y"])
    if isinstance(p, (str, bytes)) or p is None:
        return None
    try:
        seq = list(p)
    except TypeError:
        return None
    if len(seq) != 2:
        return None
    out = []
    for v in seq:
        if isinstance(v, bool) or isinstance(v, (str, bytes)) or v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return (out[0], out[1])


def _describe(p):
    try:
        return repr(p)
    except Exception:
        return "<unrepresentable>"


def validate_targets(points, workspace, expected_count=None,
                     min_separation=MIN_SEPARATION, clamp=True,
                     clearance=TARGET_CLEARANCE):
    """Check a target set. Returns a report dict; never raises.

        {"ok": bool, "error": str|None, "points": [[x, y], ...],
         "clamped": [{"index", "from", "to"}], "problems": [...]}

    Out-of-bounds points are clamped to the nearest valid point and reported.
    Points that are too close together are *not* nudged — silently moving two
    robots apart hides the fact that the model's arrangement was wrong.
    """
    report = {"ok": False, "error": None, "points": [], "clamped": [], "problems": []}

    if points is None:
        report["error"] = "no points given"
        return report
    if isinstance(points, (str, bytes, dict)):
        report["error"] = f"expected a list of [x, y] points, got {type(points).__name__}"
        return report
    try:
        raw = list(points)
    except TypeError:
        report["error"] = f"expected a list of [x, y] points, got {type(points).__name__}"
        return report

    # -- shape of each point ------------------------------------------
    bad = []
    clean = []
    for i, p in enumerate(raw):
        pt = _finite_point(p)
        if pt is None:
            bad.append({"index": i, "value": _describe(p)})
        clean.append(pt)

    if bad:
        detail = ", ".join(f"index {b['index']} = {b['value']}" for b in bad)
        report["error"] = (f"{len(bad)} coordinate(s) are not finite numbers: {detail}")
        report["problems"] = bad
        return report

    # -- count ---------------------------------------------------------
    if expected_count is not None and len(clean) != expected_count:
        # Naming both numbers is not enough on its own. A small model given
        # "got 5, expected 6" for a letter shape will often just re-send the
        # same five points, because the shape it has in mind genuinely has
        # five corners. Saying what to *do* — add one more, or drop one — is
        # what turns this into a recoverable error rather than a loop.
        short = expected_count - len(clean)
        if short > 0:
            fix = (f"add {short} more point(s) so every robot has one — put the "
                   f"extras alongside the shape, at least {int(MIN_SEPARATION)}cm "
                   "from every other point. Never return fewer than "
                   f"{expected_count}")
        else:
            fix = (f"remove {-short} point(s); return exactly {expected_count}")
        report["error"] = (f"got {len(clean)} target(s) but there are "
                           f"{expected_count} robot(s) to place — {fix}.")
        return report

    if not clean:
        report["ok"] = True
        return report

    # -- bounds, then obstacles ----------------------------------------
    placed = []
    for i, pt in enumerate(clean):
        arr = np.array(pt, dtype=float)
        crowded = clearance > 0 and not workspace.has_clearance(arr, clearance)
        if workspace.is_valid_point(arr) and not crowded:
            placed.append(arr)
            continue

        # A point can be legal and still unreachable. The controller keeps a
        # margin off every wall and obstacle, so a target sitting on the
        # boundary — a corner, say — is one no robot can ever arrive at: it
        # parks short, never settles, and the caller retries forever.
        if crowded and workspace.is_valid_point(arr):
            reason = "too close to a wall or obstacle to be reachable"
        else:
            reason = ("inside an obstacle" if workspace.point_in_bounds(arr)
                      else "outside the arena")
        if not clamp:
            report["error"] = (f"target {i} at ({pt[0]:.1f}, {pt[1]:.1f}) is {reason}")
            report["problems"] = [{"index": i, "point": list(pt), "reason": reason}]
            return report

        fixed = np.asarray(workspace.nearest_valid_point(arr, clearance=clearance),
                           dtype=float)
        if not workspace.is_valid_point(fixed):
            report["error"] = (f"target {i} at ({pt[0]:.1f}, {pt[1]:.1f}) is {reason} "
                               "and no valid point could be found near it")
            return report
        report["clamped"].append({
            "index": i,
            "from": [round(float(pt[0]), 1), round(float(pt[1]), 1)],
            "to": [round(float(fixed[0]), 1), round(float(fixed[1]), 1)],
            "reason": reason,
        })
        placed.append(fixed)

    # -- separation, on the final positions -----------------------------
    too_close = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            d = float(np.linalg.norm(placed[i] - placed[j]))
            if d < min_separation:
                too_close.append({"pair": [i, j], "distance": round(d, 1),
                                  "points": [[round(float(v), 1) for v in placed[i]],
                                             [round(float(v), 1) for v in placed[j]]]})
    if too_close:
        detail = "; ".join(
            f"targets {t['pair'][0]} and {t['pair'][1]} are {t['distance']}cm apart"
            for t in too_close)
        # Naming the pair is what lets a model fix a single stray point, but a
        # near-miss across several pairs is almost never a stray point — it is
        # a shape drawn too small for this many robots, and nudging one target
        # just moves the collision. Say so, and say by how much: measured
        # against qwen, "17.5cm apart" alone produces another 18cm attempt.
        worst = min(t["distance"] for t in too_close)
        scale = min_separation / worst if worst > 0 else 2.0
        hint = (f" — the shape is too small for {len(placed)} robots at "
                f"{min_separation:.0f}cm spacing. Redraw it about "
                f"{scale:.1f}x larger rather than moving single points")
        report["error"] = (f"targets must be at least {min_separation:.0f}cm apart: "
                           f"{detail}{hint if len(too_close) > 1 or worst > min_separation * 0.6 else ''}")
        report["problems"] = too_close
        return report

    report["ok"] = True
    report["points"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in placed]
    return report


# -- the sandbox ---------------------------------------------------------

ALLOWED_MATH = {
    "pi": math.pi, "tau": math.tau, "e": math.e,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sqrt": math.sqrt, "hypot": math.hypot, "exp": math.exp, "log": math.log,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs, "pow": math.pow,
    "degrees": math.degrees, "radians": math.radians,
}

def _noop(*args, **kwargs):
    """`print` for the sandbox: accepted, discarded.

    Models reach for print reflexively. Rejecting the whole expression over a
    debugging line costs a full round trip — 11s on this hardware — so accept
    it and throw the output away. It must never reach real stdout, because
    that is the channel the sandbox returns its result on.
    """
    return None


ALLOWED_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "round": round, "sum": sum, "int": int, "float": float,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "list": list,
    "tuple": tuple, "reversed": reversed, "print": _noop,
}

ALLOWED_METHODS = frozenset({"append", "extend", "insert"})

_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Expression, ast.Assign, ast.AugAssign, ast.Return,
    ast.Load, ast.Store, ast.Name, ast.Constant, ast.Tuple, ast.List, ast.Dict,
    ast.Set, ast.Subscript, ast.Slice, ast.Index if hasattr(ast, "Index") else ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.If,
    ast.For, ast.Break, ast.Continue, ast.Pass, ast.Call, ast.keyword,
    ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.SetComp, ast.DictComp,
    ast.Starred, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

_FORBIDDEN_NODES = {
    ast.Import: "imports",
    ast.ImportFrom: "imports",
    ast.While: "while loops",
    ast.FunctionDef: "function definitions",
    ast.AsyncFunctionDef: "function definitions",
    ast.ClassDef: "class definitions",
    ast.Lambda: "lambdas",
    ast.Global: "global statements",
    ast.Nonlocal: "nonlocal statements",
    ast.With: "with blocks",
    ast.Try: "try blocks",
    ast.Raise: "raise statements",
    ast.Delete: "del statements",
    ast.Yield: "yield",
    ast.YieldFrom: "yield",
    ast.Await: "await",
}


def check_source(src, allow_bare_expression=False):
    """Static gate. Returns an error string, or None if the source is acceptable.

    `allow_bare_expression` is for conditions: `d_target < 60` is a whole valid
    rule and assigns nothing, where a points expression that assigns nothing is
    a mistake.
    """
    if not isinstance(src, str) or not src.strip():
        return "expression is empty"
    if len(src) > 4000:
        return "expression is too long (limit 4000 characters)"

    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"

    for node in ast.walk(tree):
        for bad, label in _FORBIDDEN_NODES.items():
            if isinstance(node, bad):
                return f"{label} are not allowed"

        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"dunder names are not allowed: {node.id}"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"dunder attributes are not allowed: .{node.attr}"
            if isinstance(node.value, ast.Name) and node.value.id == "math":
                if node.attr not in ALLOWED_MATH:
                    return f"math.{node.attr} is not in the allowed set"
            elif node.attr not in ALLOWED_METHODS:
                # Building a list in a loop needs .append; nothing else gets through.
                # With no imports and no builtins there is no dangerous object to
                # reach a method on in the first place.
                return (f"attribute access is only allowed on math, or "
                        f"{sorted(ALLOWED_METHODS)}, got .{node.attr}")

        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                if fn.id not in ALLOWED_BUILTINS and fn.id not in ALLOWED_MATH:
                    return f"{fn.id}() is not an allowed function"
            elif not isinstance(fn, ast.Attribute):
                return "only direct calls to allowed functions are permitted"

        if not isinstance(node, _ALLOWED_NODES):
            return f"{type(node).__name__} is not allowed here"

    return None


def _run_source(src, variables, pipe):
    try:
        env = dict(ALLOWED_BUILTINS)
        env.update(ALLOWED_MATH)
        env["math"] = _MathProxy()
        env.update(variables or {})
        scope = {"__builtins__": {}}
        scope.update(env)
        exec(compile(src, "<points>", "exec"), scope)  # noqa: S102 - gated by check_source
        pipe.send(("ok", scope.get("points")))
    except Exception as e:
        pipe.send(("error", f"{type(e).__name__}: {e}"))


class _MathProxy:
    """Exposes only the allowlisted names, so `math.system` cannot exist."""

    def __getattr__(self, name):
        if name in ALLOWED_MATH:
            return ALLOWED_MATH[name]
        raise AttributeError(f"math.{name} is not available")


_BOOTSTRAP = r"""
import json, math, sys

req = json.loads(sys.stdin.read())
ALLOWED_MATH = {name: getattr(math, name) for name in req["math_names"]
                if hasattr(math, name)}
for name, value in (("pi", math.pi), ("tau", math.tau), ("e", math.e)):
    if name in req["math_names"]:
        ALLOWED_MATH[name] = value


class MathProxy:
    def __getattr__(self, name):
        if name in ALLOWED_MATH:
            return ALLOWED_MATH[name]
        raise AttributeError("math.%s is not available" % name)


def _noop(*a, **k):
    return None


builtins_allowed = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "round": round, "sum": sum, "int": int, "float": float,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "list": list,
    "tuple": tuple, "reversed": reversed, "print": _noop,
}

scope = {"__builtins__": {}}
scope.update(builtins_allowed)
scope.update(ALLOWED_MATH)
scope["math"] = MathProxy()
scope.update(req["variables"])

try:
    exec(compile(req["src"], "<points>", "exec"), scope)
    value = scope.get("points")
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        value = [list(p) if hasattr(p, "__iter__") else p for p in value]
    sys.stdout.write(json.dumps({"status": "ok", "value": value}))
except Exception as e:
    sys.stdout.write(json.dumps(
        {"status": "error", "error": "%s: %s" % (type(e).__name__, e)}))
"""


_ESCAPES = (("\\n", "\n"), ("\\t", "\t"), ("\\r", "\r"), ('\\"', '"'), ("\\'", "'"))


def repair_source(src):
    """Undo a model's double-escaping. Returns (source, repaired).

    Models routinely send `\\n` as a literal backslash-n inside the JSON
    arguments instead of a real newline, which Python then reads as a line
    continuation followed by junk — "unexpected character after line
    continuation character". The source is otherwise perfectly good, so
    unescape and retry rather than making the model guess what we disliked.

    Only applied when the original does not compile and the repaired version
    does, so a legitimately escaped string is never rewritten.
    """
    if not isinstance(src, str) or "\\" not in src:
        return src, False
    try:
        compile(src, "<points>", "exec")
        return src, False               # it was fine; leave it alone
    except SyntaxError:
        pass

    fixed = src
    for bad, good in _ESCAPES:
        fixed = fixed.replace(bad, good)
    if fixed == src:
        return src, False
    try:
        compile(fixed, "<points>", "exec")
    except SyntaxError:
        return src, False               # repair did not help; report the real error
    return fixed, True


def run_sandboxed(src, variables=None, timeout=SANDBOX_TIMEOUT):
    """Execute `src` in a bare interpreter and return its `points`. Never raises.

    Returns {"ok", "error", "value"}. A separate process is what makes the
    timeout real — a tight loop in this interpreter cannot be interrupted
    reliably, and banning `while` is not sufficient on its own.

    It is a plain `subprocess`, not `multiprocessing`, deliberately. On macOS
    the spawn start method re-imports the parent's `__main__` in the child; run
    from the pygame app that means importing torch and pygame before a single
    line of the expression executes, which ate the entire budget and failed
    *every* call. Under pytest `__main__` is cheap, so tests never saw it. A
    bare `-I` interpreter imports nothing of ours, starts in tens of
    milliseconds, and is more isolated as a bonus.
    """
    src, repaired = repair_source(src)

    err = check_source(src)
    if err:
        if "syntax error" in err:
            err += ("  — write the whole thing as ONE line that assigns `points` "
                    "directly, e.g. points = [(cx+60*cos(i*2*pi/n), "
                    "cy+60*sin(i*2*pi/n)) for i in range(n)]")
        return {"ok": False, "error": err, "value": None, "repaired": repaired}

    payload = json.dumps({
        "src": src,
        "variables": _jsonable(variables or {}),
        "math_names": sorted(ALLOWED_MATH),
    })

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _BOOTSTRAP],
            input=payload, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False,
                "error": f"expression did not finish within {timeout:g}s",
                "value": None, "repaired": repaired}
    except Exception as e:
        return {"ok": False, "error": f"could not start sandbox: {e}", "value": None}

    if proc.returncode != 0 and not proc.stdout.strip():
        detail = (proc.stderr or "").strip().splitlines()
        return {"ok": False,
                "error": f"sandbox failed: {detail[-1] if detail else 'no output'}",
                "value": None}

    if not proc.stdout.strip():
        return {"ok": False, "error": "expression produced no result", "value": None}

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "sandbox returned unreadable output",
                "value": None}

    if result.get("status") == "error":
        return {"ok": False, "error": result.get("error"), "value": None,
                "repaired": repaired}

    value = result.get("value")
    if value is None:
        # Valid code that never assigned `points`. The usual cause is a `#`
        # comment earlier on the same line silently eating the assignment, and
        # nothing about the error would otherwise point at it.
        hint = ("your expression ran but never assigned `points`")
        if "#" in src:
            hint += (" — a `#` comment hides everything after it on that line, "
                     "including the assignment; remove the comment")
        elif ";" in src:
            hint += (" — put each statement on its own line instead of using `;`")
        return {"ok": False, "error": hint, "value": None, "repaired": repaired}

    return {"ok": True, "error": None, "value": value, "repaired": repaired}


def _jsonable(variables):
    """Only plain numbers cross the process boundary."""
    out = {}
    for k, v in variables.items():
        try:
            out[k] = float(v) if isinstance(v, (int, float, np.floating,
                                                 np.integer)) else v
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def compute_points(src, workspace, expected_count=None, variables=None,
                   timeout=SANDBOX_TIMEOUT, min_separation=MIN_SEPARATION):
    """Sandboxed generation, then every check a literal point list gets."""
    run = run_sandboxed(src, variables=variables, timeout=timeout)
    if not run["ok"]:
        return {"ok": False, "error": run["error"], "points": [], "clamped": [],
                "problems": []}

    value = run["value"]
    if value is None:
        return {"ok": False,
                "error": "expression must assign a list of (x, y) points to `points`",
                "points": [], "clamped": [], "problems": []}
    if isinstance(value, (str, bytes, dict)):
        return {"ok": False,
                "error": f"`points` must be a list of (x, y) tuples, got "
                         f"{type(value).__name__}",
                "points": [], "clamped": [], "problems": []}

    return validate_targets(value, workspace, expected_count=expected_count,
                            min_separation=min_separation)
