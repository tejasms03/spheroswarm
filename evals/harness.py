"""Running the eval set: one fresh world per case, scored automatically.

Isolation is the point. Every case gets a new all-sim fleet at seeded start
positions and, critically, **scratch paths** for roster/workspace/formations.
An eval that writes the live `formations.json` corrupts both the next run and
the user's library, and this has already bitten once in this project.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml

from fleet.manager import Fleet
from fleet.roster import RobotEntry
from llm.agent import SwarmAgent
from tools import SwarmContext
from tools.formations import FormationLibrary
from workspace.space import Workspace

from . import assertions

ROOT = Path(__file__).resolve().parent
COMMANDS = ROOT / "commands.yaml"
RESULTS = ROOT / "results"

BOUNDS = [[0, 0], [240, 0], [240, 180], [0, 180]]
OBSTACLES = [{"type": "circle", "center": [180, 40], "radius": 15}]

NAMES = ["Seasmoke", "Caraxes", "Syrax", "Vhagar", "Meleys", "Sunfyre"]
CODES = ["SSMK", "CRXS", "SYRX", "VHGR", "MLYS", "SNFR"]
COLORS = ["cyan", "red", "yellow", "green", "magenta", "blue"]

# Fixed start positions: every model faces the identical world.
START = np.array([[30.0, 30.0], [90.0, 30.0], [150.0, 30.0],
                  [30.0, 150.0], [90.0, 150.0], [150.0, 150.0]])

SETTLE_SECONDS = 12.0


@dataclass
class CaseResult:
    id: str
    group: str = ""
    text: str = ""
    completed: bool = False
    valid: bool = False
    assertion: bool = False
    reason: str = ""
    tool_calls: int = 0
    retries: int = 0
    text_recoveries: int = 0
    latency_s: float = 0.0
    hit_cap: bool = False
    human_review: bool = False
    error: str = None
    reply: str = ""
    tools_used: list = field(default_factory=list)

    @property
    def passed(self):
        return self.completed and self.valid and self.assertion


def load_cases(path=COMMANDS):
    data = yaml.safe_load(Path(path).read_text()) or {}
    return data.get("cases", [])


def build_world(tmpdir, robots=6, seed=0):
    """A fresh, isolated all-sim world. Nothing here touches the repo's files."""
    tmpdir = Path(tmpdir)
    space = Workspace(bounds_cm=BOUNDS, obstacles=list(OBSTACLES),
                      path=tmpdir / "workspace.json")

    fleet = Fleet(workspace=space, seed=seed)
    for i in range(robots):
        fleet.add(RobotEntry(name=NAMES[i], code=CODES[i], kind="sim",
                             color=COLORS[i]))
    for i, code in enumerate(CODES[:robots]):
        fleet[code].pos = START[i % len(START)].copy()

    ctx = SwarmContext(fleet=fleet, workspace=space,
                       library=FormationLibrary(path=tmpdir / "formations.json"))
    return ctx


def positions_of(ctx):
    return np.array([ctx.fleet[c].pos for c in ctx.active_codes()], dtype=float)


def position_map(ctx):
    """code -> position. Rotation is only measurable per robot, not per point set."""
    return {c: ctx.fleet[c].pos.copy() for c in ctx.active_codes()}


def run_case(case, client_factory, tmpdir, settle=SETTLE_SECONDS,
             max_tool_calls=8, on_event=None):
    """One case, one fresh world. Never raises: a crash is a failed case."""
    result = CaseResult(id=case.get("id", "?"), group=case.get("group", ""),
                        text=case.get("text", ""),
                        human_review=bool(case.get("human_review")))
    started = time.time()

    try:
        ctx = build_world(tmpdir, robots=int(case.get("robots", 6)),
                          seed=int(case.get("seed", 0)))
        agent = SwarmAgent(client_factory(), ctx, max_tool_calls=max_tool_calls,
                           on_event=on_event)

        # setup turns are not scored; they exist to reach the state under test
        for line in _as_list(case.get("setup")):
            agent.command(line)
            ctx.run_for(settle)

        before = positions_of(ctx)
        before_map = position_map(ctx)

        r = agent.command(case["text"])
        ctx.run_for(settle)
        after = positions_of(ctx)
        after_map = position_map(ctx)

        result.completed = r.ok and not r.hit_cap
        result.tool_calls = len(r.tool_calls)
        result.retries = r.retries
        result.text_recoveries = r.recovered_from_text_count
        result.hit_cap = r.hit_cap
        result.error = r.error
        result.reply = r.reply
        result.tools_used = r.tool_names

        result.valid, valid_reason = assertions.valid_arrangement()(
            after, workspace=ctx.ws)

        check = assertions.build(case.get("assertion"))
        clamped_any = any(bool(t.result.get("clamped")) for t in r.tool_calls
                          if isinstance(t.result, dict))
        passed, reason = check(
            after, before=before, before_map=before_map, after_map=after_map,
            workspace=ctx.ws, library=ctx.library,
            reply=r.reply, tool_names=r.tool_names, agent_ok=r.ok,
            clamped_any=clamped_any)
        result.assertion = bool(passed)
        result.reason = reason if passed else f"{reason}"
        if not result.valid:
            result.reason = f"invalid arrangement: {valid_reason} | {result.reason}"

        ctx.fleet.close()

    except Exception as e:                       # a harness bug is a failed case
        result.error = f"{type(e).__name__}: {e}"
        result.reason = result.error

    result.latency_s = round(time.time() - started, 2)
    return result


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def aggregate(results):
    n = len(results)
    if not n:
        return {}
    auto = [r for r in results if not r.human_review]
    return {
        "cases": n,
        "passed": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / n, 3),
        "auto_cases": len(auto),
        "auto_passed": sum(1 for r in auto if r.passed),
        "auto_pass_rate": round(sum(1 for r in auto if r.passed) / len(auto), 3)
        if auto else None,
        "completed": sum(1 for r in results if r.completed),
        "valid": sum(1 for r in results if r.valid),
        "hit_cap": sum(1 for r in results if r.hit_cap),
        "mean_tool_calls": round(float(np.mean([r.tool_calls for r in results])), 2),
        "total_retries": sum(r.retries for r in results),
        "total_text_recoveries": sum(r.text_recoveries for r in results),
        "mean_latency_s": round(float(np.mean([r.latency_s for r in results])), 2),
        "total_latency_s": round(float(np.sum([r.latency_s for r in results])), 2),
    }


def by_group(results):
    groups = {}
    for r in results:
        g = groups.setdefault(r.group or "ungrouped", {"n": 0, "passed": 0})
        g["n"] += 1
        g["passed"] += int(r.passed)
    for g in groups.values():
        g["rate"] = round(g["passed"] / g["n"], 2) if g["n"] else 0.0
    return groups


# -- reporting ---------------------------------------------------------------

def format_table(results, width=118):
    head = (f"{'case':<22} {'grp':<9} {'ok':<3} {'val':<4} {'assert':<7} "
            f"{'calls':>5} {'retry':>5} {'txt':>4} {'secs':>6}  reason")
    lines = [head, "-" * width]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        flag = "*" if r.human_review else " "
        reason = (r.reason or "")[:44]
        lines.append(
            f"{r.id:<22} {r.group:<9} {_yn(r.completed):<3} {_yn(r.valid):<4} "
            f"{mark:<7} {r.tool_calls:>5} {r.retries:>5} {r.text_recoveries:>4} "
            f"{r.latency_s:>6.1f}{flag} {reason}")
    return "\n".join(lines)


def _yn(v):
    return "y" if v else "n"


def format_summary(agg, groups, model):
    lines = ["", f"model: {model}",
             f"  pass {agg['passed']}/{agg['cases']} ({agg['pass_rate']:.0%})"
             f"   automatic-only {agg['auto_passed']}/{agg['auto_cases']}"
             f" ({(agg['auto_pass_rate'] or 0):.0%})",
             f"  completed {agg['completed']}   valid {agg['valid']}"
             f"   hit cap {agg['hit_cap']}",
             f"  mean tool calls {agg['mean_tool_calls']}"
             f"   retries {agg['total_retries']}"
             f"   text recoveries {agg['total_text_recoveries']}",
             f"  mean latency {agg['mean_latency_s']}s"
             f"   total {agg['total_latency_s']}s", "  by group:"]
    for name, g in sorted(groups.items()):
        lines.append(f"    {name:<10} {g['passed']}/{g['n']}  ({g['rate']:.0%})")
    return "\n".join(lines)


def save_results(model, results, agg, groups, directory=RESULTS):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = model.replace("/", "_").replace(":", "-")
    path = directory / f"{safe}_{stamp}.json"
    path.write_text(json.dumps({
        "model": model,
        "timestamp": stamp,
        "aggregate": agg,
        "by_group": groups,
        "cases": [asdict(r) for r in results],
    }, indent=2, default=str))
    return path


def format_comparison(per_model):
    """per_model: {model: (results, aggregate)}. Side-by-side table."""
    models = list(per_model)
    ids = [r.id for r in per_model[models[0]][0]]

    width = 22 + len(models) * 12
    lines = ["", "case".ljust(22) + "".join(m[:11].ljust(12) for m in models),
             "-" * width]
    for i, cid in enumerate(ids):
        row = cid.ljust(22)
        for m in models:
            results = per_model[m][0]
            r = next((x for x in results if x.id == cid), None)
            row += ("PASS" if r and r.passed else "fail").ljust(12)
        lines.append(row)

    lines.append("-" * width)
    for label, key, fmt in [("pass rate", "pass_rate", "{:.0%}"),
                            ("mean calls", "mean_tool_calls", "{}"),
                            ("retries", "total_retries", "{}"),
                            ("text recov", "total_text_recoveries", "{}"),
                            ("mean secs", "mean_latency_s", "{}")]:
        row = label.ljust(22)
        for m in models:
            row += fmt.format(per_model[m][1].get(key, 0)).ljust(12)
        lines.append(row)
    return "\n".join(lines)
