"""Run a whole routine from one tool call.

`MAX_TOOL_CALLS` is 8. A five-phase piece of choreography costs about fourteen
calls once the settles are counted, so the model runs out of budget halfway
through and the fleet stops mid-formation. Raising the cap is the wrong fix:
it exists so a model that has started reasoning in circles cannot drive the
swarm forever, and that is worth keeping.

The right fix is to stop spending the budget on steps that were already decided.
A sequence is ONE call carrying the whole routine, so the cap goes back to
limiting how long the model may think rather than how much choreography a
single intent may express -- fourteen actions become one call with seven left
for recovery.

Three properties keep this a longer leash rather than a hole in the fence.

**Nothing runs until everything validates.** Every step is checked against the
target tool's own schema first. A bad step fails the sequence before any robot
moves, where today a bad fourth step leaves the first three done and the swarm
in a shape nobody asked for.

**It stops when told.** A sequence checks for cancellation between steps, so
STOP still means stop. Without that this would be exactly the runaway the cap
was written to prevent.

**It is bounded.** `MAX_STEPS` caps the routine, and every step is a tool that
already has its own limits -- the waits time out, the paths take durations.
"""

MAX_STEPS = 16


def _registry():
    from . import registry
    return registry


def _plan(steps, reg):
    """Every step checked against its tool's schema. Returns (plan, errors).

    Deliberately a separate pass from running them. Validating as you go means
    discovering step 9 is malformed with steps 1-8 already on the floor, and
    the recovery for that is worse than the original mistake.
    """
    plan, errors = [], []
    if not isinstance(steps, list) or not steps:
        return [], ["steps must be a non-empty list"]
    if len(steps) > MAX_STEPS:
        return [], [f"{len(steps)} steps is over the limit of {MAX_STEPS} — "
                    "split the routine, or use a path or flow for the "
                    "repetitive part"]

    for i, step in enumerate(steps, 1):
        where = f"step {i}"
        if not isinstance(step, dict):
            errors.append(f"{where}: expected an object, got "
                          f"{type(step).__name__}")
            continue
        # `wait` is a shorthand, not a separate mechanism -- it is
        # wait_until_settled, spelled the way a routine reads.
        if "wait" in step and "tool" not in step:
            args = step.get("wait") or {}
            if not isinstance(args, dict):
                errors.append(f"{where}: wait takes an object of arguments")
                continue
            plan.append(("wait_until_settled", args, where))
            continue

        name = step.get("tool")
        args = step.get("args", {})
        if not isinstance(name, str):
            errors.append(f"{where}: needs a 'tool' name")
            continue
        if name == "run_sequence":
            errors.append(f"{where}: a sequence cannot run another sequence")
            continue
        tool = reg.BY_NAME.get(name)
        if tool is None:
            errors.append(f"{where}: unknown tool {name!r}")
            continue
        if not isinstance(args, dict):
            errors.append(f"{where}: args must be an object, got "
                          f"{type(args).__name__}")
            continue
        allowed = set(tool["parameters"].get("properties", {}))
        unexpected = set(args) - allowed
        if unexpected:
            errors.append(f"{where}: {name} got unexpected argument(s) "
                          f"{', '.join(sorted(unexpected))}; accepts "
                          f"{', '.join(sorted(allowed)) or 'no arguments'}")
            continue
        missing = set(tool["parameters"].get("required", [])) - set(args)
        if missing:
            errors.append(f"{where}: {name} is missing "
                          f"{', '.join(sorted(missing))}")
            continue
        plan.append((name, args, where))
    return plan, errors


def _cancelled(ctx):
    """True when something outside has asked for the run to stop.

    Read off the context rather than imported, because cancellation belongs to
    whoever is driving -- the agent session owns an Event, a test owns a flag,
    and a bare context has neither and simply runs.
    """
    flag = getattr(ctx, "cancel_event", None)
    if flag is None:
        return False
    try:
        return bool(flag.is_set())
    except AttributeError:
        return bool(flag)


def run_sequence(ctx, steps, stop_on_error=True):
    """Execute a list of tool calls in order. See the module docstring."""
    from .result import fail, ok

    reg = _registry()
    plan, errors = _plan(steps, reg)
    if errors:
        return fail(ctx, "the sequence was not run — " + "; ".join(errors[:4]))

    done = []
    for name, args, where in plan:
        if _cancelled(ctx):
            return ok(ctx, summary=f"stopped after {len(done)} of {len(plan)} "
                                   "steps", steps=done, cancelled=True)
        result = reg.call(ctx, name, args)
        entry = {"step": where, "tool": name}
        if isinstance(result, dict):
            entry["error"] = result.get("error")
            entry["summary"] = result.get("summary")
        done.append(entry)
        if stop_on_error and isinstance(result, dict) and result.get("error"):
            return fail(ctx, f"{where} ({name}) failed: {result['error']}",
                        steps=done)

    return ok(ctx, summary=f"ran {len(done)} steps", steps=done)


SCHEMA = {
    "name": "run_sequence",
    "description": (
        "Run a whole routine in one call: a list of steps, each naming a tool "
        "and its arguments, executed in order. Use this for anything with "
        "phases — form up, then split, then converge — instead of spending one "
        "tool call per step and running out of budget partway through. Every "
        "step is validated before any of them runs, so a mistake costs you "
        "nothing rather than leaving the swarm half-arranged. Write {\"wait\": "
        "{}} between steps that must finish before the next begins."),
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": (
                    "Ordered steps. Each is {\"tool\": name, \"args\": {...}}, "
                    "or {\"wait\": {\"tolerance\": 4}} to let the swarm settle."),
                "items": {"type": "object"},
            },
            "stop_on_error": {
                "type": "boolean",
                "description": (
                    "Stop at the first failing step (default) rather than "
                    "carrying on through the rest."),
            },
        },
        "required": ["steps"],
    },
    "fn": lambda ctx, **kw: run_sequence(ctx, **kw),
}
