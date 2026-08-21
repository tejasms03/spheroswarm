"""One result shape for every tool: {ok, error, state_summary, ...}."""


def ok(ctx, **extra):
    out = {"ok": True, "error": None, "state_summary": ctx.state_summary()}
    out.update(extra)
    return out


def fail(ctx, error, **extra):
    out = {"ok": False, "error": error, "state_summary": ctx.state_summary()}
    out.update(extra)
    return out
