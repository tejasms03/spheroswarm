"""Compiling a velocity field once, instead of evaluating it per tick.

`compute_points` runs its expression in a subprocess, which is the only honest
way to bound a one-off computation. A flow field is different: it is evaluated
every tick, for every robot — a subprocess at 10Hz per robot is not viable.

So the same AST allowlist gates the source, and the *check* stays out of
process, but the accepted expression is compiled once into a plain callable.
The gate is what makes that safe: no imports, no attribute access beyond the
maths allowlist, no loops that could hang a tick.
"""

import math

FLOW_NAMES = ("x", "y", "t", "cx", "cy", "n", "ex", "ey")

# `tools` is imported lazily, inside the functions that need it. workspace sits
# *below* tools in the layering, and importing upward at module load makes the
# whole thing depend on import order: `import tools` first and everything
# works, but `from workspace.space import Workspace` in a fresh interpreter
# goes workspace.flow -> tools.validate -> tools/__init__ -> tools.motion ->
# back to the half-initialised workspace.flow, and raises. The app happened to
# import tools first and never saw it; any standalone script does not.


def _validate():
    from tools import validate
    return validate


class _MathProxy:
    def __getattr__(self, name):
        from tools.validate import ALLOWED_MATH
        if name in ALLOWED_MATH:
            return ALLOWED_MATH[name]
        raise AttributeError(f"math.{name} is not available")


def compile_flow(expr, cx=0.0, cy=0.0, n=0):
    """Return (callable(x, y, t) -> (vx, vy), error). Never raises."""
    if not isinstance(expr, str) or not expr.strip():
        return None, "flow expression is empty"

    validate = _validate()

    src, _ = validate.repair_source(expr)
    err = validate.check_source(src)
    if err:
        return None, err

    try:
        code = compile(src, "<flow>", "exec")
    except SyntaxError as e:
        return None, f"syntax error: {e.msg} (line {e.lineno})"

    base = dict(validate.ALLOWED_BUILTINS)
    base.update(validate.ALLOWED_MATH)
    base["math"] = _MathProxy()
    base["cx"], base["cy"], base["n"] = float(cx), float(cy), int(n)

    def field(x, y, t, ex=0.0, ey=0.0):
        scope = {"__builtins__": {}}
        scope.update(base)
        scope["x"], scope["y"], scope["t"] = float(x), float(y), float(t)
        # ex/ey are the followed entity's live position. Binding them here is
        # what lets one expression orbit something that is itself moving.
        scope["ex"], scope["ey"] = float(ex), float(ey)
        exec(code, scope)                       # noqa: S102 - gated by check_source
        if "vx" not in scope or "vy" not in scope:
            raise ValueError("flow must assign both vx and vy")
        return float(scope["vx"]), float(scope["vy"])

    # prove it runs and returns finite numbers before anyone depends on it
    try:
        vx, vy = field(cx or 1.0, cy or 1.0, 0.0)
    except Exception as e:
        return None, f"flow failed: {type(e).__name__}: {e}"
    if not (math.isfinite(vx) and math.isfinite(vy)):
        return None, "flow produced a non-finite velocity"

    return field, None


PREDICATE_NAMES = ("x", "y", "t", "speed", "n", "cx", "cy", "ex", "ey",
                   "d_target", "d_nearest", "rank", "elapsed")


def compile_predicate(expr, cx=0.0, cy=0.0, n=0):
    """Compile a gated boolean expression, evaluated per robot per tick.

    The same AST allowlist as a flow field, for the same reason: this is
    model-authored code running at tick rate, so it may not import, loop
    unboundedly, or reach for attributes. It is compiled once and then called,
    because a subprocess per robot per tick is not viable.

    This is what makes a standing *conditional* rule expressible at all —
    "follow while it is within 80cm", "orbit only while the target is moving",
    "spread out when anyone gets closer than 30". Without it a layer is either
    on or off for its whole duration, and every condition has to be re-decided
    by a round trip to the model.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None, "condition is empty"

    validate = _validate()
    src, _ = validate.repair_source(expr.strip())
    err = validate.check_source(src, allow_bare_expression=True)
    if err:
        return None, err

    body = src if "\n" not in src.strip() else src
    try:
        code = compile(body.strip(), "<when>", "eval")
    except SyntaxError:
        return None, ("a condition must be a single expression that is true or "
                      f"false, like 'd_target < 60' — could not read {expr!r}")

    base = dict(validate.ALLOWED_BUILTINS)
    base.update(validate.ALLOWED_MATH)
    base["math"] = _MathProxy()
    base["cx"], base["cy"], base["n"] = float(cx), float(cy), int(n)

    def predicate(**values):
        scope = {"__builtins__": {}}
        scope.update(base)
        for name in PREDICATE_NAMES:
            scope.setdefault(name, 0.0)
        scope.update(values)
        return bool(eval(code, scope))      # noqa: S307 - gated by check_source

    return predicate, None
