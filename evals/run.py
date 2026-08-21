"""Eval runner.

    python -m evals.run --model qwen3.5:9b --repeats 3
    python -m evals.run --model frontier --repeats 1
    python -m evals.run --compare qwen3.5:9b qwen3:8b qwen3.5:4b
    python -m evals.run --model stub                 # no Ollama needed

`--model stub` replays a canned script, which is how the harness itself is
tested and how you check the plumbing before spending GPU time.
"""

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

from llm.client import ModelUnreachable, client_for

from .harness import (aggregate, by_group, format_comparison, format_summary,
                      format_table, load_cases, run_case, save_results)


def make_factory(model, num_ctx=None, thinking=None, temperature=None):
    if model == "stub":
        from .stub_script import build_stub_client
        return build_stub_client

    overrides = {}
    if num_ctx is not None:
        overrides["num_ctx"] = num_ctx
    if thinking is not None:
        overrides["thinking"] = thinking
    if temperature is not None:
        overrides["temperature"] = temperature

    def factory():
        return client_for(model, **overrides)

    return factory


def run_model(model, cases, repeats=1, settle=None, max_tool_calls=8,
              verbose=True, factory=None):
    factory = factory or make_factory(model)
    all_runs = []

    for rep in range(repeats):
        results = []
        for case in cases:
            with tempfile.TemporaryDirectory(prefix="swarm-eval-") as tmp:
                kwargs = {"max_tool_calls": max_tool_calls}
                if settle is not None:
                    kwargs["settle"] = settle
                r = run_case(case, factory, tmp, **kwargs)
            results.append(r)
            if verbose:
                mark = "PASS" if r.passed else "FAIL"
                extra = "" if r.passed else f"  {r.reason[:70]}"
                print(f"  [{rep + 1}/{repeats}] {r.id:<22} {mark}"
                      f"  {r.tool_calls} calls  {r.latency_s:.1f}s{extra}",
                      flush=True)
        all_runs.append(results)

    return all_runs


def merge_repeats(all_runs):
    """Flatten repeats, and report per-case pass counts when repeats > 1."""
    flat = [r for run in all_runs for r in run]
    if len(all_runs) == 1:
        return flat, {}

    counts = {}
    for run in all_runs:
        for r in run:
            c = counts.setdefault(r.id, {"passed": 0, "total": 0})
            c["passed"] += int(r.passed)
            c["total"] += 1
    return flat, counts


def main(argv=None):
    p = argparse.ArgumentParser(description="Score a model on the swarm command set.")
    p.add_argument("--model", default="stub",
                   help="preset from llm/models.yaml, or 'stub' for the canned model")
    p.add_argument("--compare", nargs="+", metavar="MODEL",
                   help="run several models and print them side by side")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--cases", help="path to a commands.yaml")
    p.add_argument("--only", nargs="+", help="run only these case ids")
    p.add_argument("--group", nargs="+", help="run only these groups")
    p.add_argument("--max-tool-calls", type=int, default=8)
    p.add_argument("--settle", type=float, default=None,
                   help="seconds of simulation after each command")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    cases = load_cases(a.cases) if a.cases else load_cases()
    if a.only:
        wanted = set(a.only)
        cases = [c for c in cases if c.get("id") in wanted]
    if a.group:
        wanted = set(a.group)
        cases = [c for c in cases if c.get("group") in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    models = a.compare or [a.model]
    per_model = {}

    for model in models:
        print(f"\n=== {model} — {len(cases)} case(s) x {a.repeats} ===", flush=True)
        try:
            runs = run_model(model, cases, repeats=a.repeats, settle=a.settle,
                             max_tool_calls=a.max_tool_calls,
                             verbose=not a.quiet)
        except ModelUnreachable as e:
            print(f"  {e}", file=sys.stderr)
            continue

        flat, counts = merge_repeats(runs)
        agg = aggregate(flat)
        groups = by_group(flat)
        per_model[model] = (flat, agg)

        print()
        print(format_table(runs[0] if a.repeats == 1 else flat))
        print(format_summary(agg, groups, model))

        if counts:
            unstable = {k: v for k, v in counts.items()
                        if 0 < v["passed"] < v["total"]}
            if unstable:
                print("\n  unstable across repeats:")
                for cid, v in sorted(unstable.items()):
                    print(f"    {cid:<22} {v['passed']}/{v['total']}")

        human = [r for r in flat if r.human_review]
        if human:
            print("\n  flagged for human review (scored for validity only):")
            for r in human:
                print(f"    {r.id:<22} {r.reply[:70]}")

        if not a.no_save:
            path = save_results(model, flat, agg, groups)
            print(f"\n  written to {path}")

    if len(per_model) > 1:
        print(format_comparison(per_model))

    if not per_model:
        return 1
    worst = min(agg["pass_rate"] for _, agg in per_model.values())
    return 0 if worst > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
