#!/usr/bin/env python3
"""Prove a hosted OpenAI-compatible gateway can actually drive this agent.

    source ~/.spheroswarm.env
    python -m llm.smoke                       # the Portkey gateway defaults
    python -m llm.smoke --preset qwen3.5:9b   # or any preset in models.yaml

Nothing here touches the agent, the fleet, or any state file. It answers one
question before a line of integration is written: *does tool calling survive
the gateway?* Four checks, in dependency order —

  1  a plain completion comes back as text
  2  a request carrying the real tool schemas comes back with a populated
     `tool_calls` object rather than a prose description of a tool call
  3  a follow-up turn carrying a `tool` role message is accepted
  4  latency for each, so the cost of the hop is visible up front

Check 2 is the one that matters. Gateways are notorious for dropping `tools` on
the floor or flattening it into text, and tool calling is this project's entire
critical path: if it fails here nothing downstream can work, and the honest
move is to stop rather than build a workaround on top of it.

The key is read from the environment and never printed. Every line this script
emits — including gateway error bodies, which we do not control — is scrubbed
of the key before it reaches the terminal.
"""

import argparse
import json
import os
import sys
import time

from llm.client import (DEFAULT_BASE_URL, ModelClient, ModelError,
                        ModelUnreachable, client_for)
from tools import registry

GATEWAY_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"
GATEWAY_MODEL = "@vertexai/anthropic.claude-sonnet-5"
KEY_ENV = "PORTKEY_API_KEY"

# A real command against a real schema. A toy schema would prove the gateway
# accepts *a* tool; this proves it accepts the 22 this project actually sends.
SYSTEM = ("You command a swarm of rolling robots on a floor, in centimetres. "
          "Robots are addressed by a four-letter code. Seasmoke is SSMK. "
          "Call a tool; do not describe what you would call.")
COMMAND = "Move Seasmoke to 100,80"

OK, BAD, WARN = "  ok ", " FAIL", " warn"


class Scrubber:
    """Redacts the key from anything on its way to the terminal.

    The gateway's error bodies are not ours to trust: a 401 that helpfully
    echoes the credential it rejected would otherwise print it.
    """

    def __init__(self, *secrets):
        self.secrets = [s for s in secrets if s and len(s) >= 8]

    def __call__(self, text):
        out = str(text)
        for s in self.secrets:
            out = out.replace(s, "<redacted>")
        return out


def _fmt(seconds):
    return f"{seconds:6.2f}s"


def check_plain(client, scrub, report):
    """1 — the endpoint answers at all, with text."""
    t0 = time.time()
    r = client.chat([{"role": "system", "content": "You are terse."},
                     {"role": "user", "content": "Reply with the single word: ready"}])
    dt = time.time() - t0

    if not r.text.strip():
        report(BAD, "plain completion", dt, "came back empty")
        return False, None
    report(OK, "plain completion", dt, repr(r.text.strip()[:60]))
    return True, r


def check_tool_call(client, scrub, report):
    """2 — the critical one. Schemas go out, a real tool_call comes back."""
    schemas = registry.schemas()
    t0 = time.time()
    r = client.chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": COMMAND}], tools=schemas)
    dt = time.time() - t0

    if not r.tool_calls:
        report(BAD, f"tool call ({len(schemas)} schemas)", dt,
               "no tool_calls in the response")
        print()
        print("  The gateway did not return a tool call. What came back instead:")
        print(f"    text: {scrub(r.text[:500]) or '<empty>'}")
        finish = (r.raw.get("choices") or [{}])[0].get("finish_reason")
        print(f"    finish_reason: {finish}")
        print()
        print("  Stopping here. Tool calling is the critical path for this")
        print("  project — every command the agent runs is a tool call — so")
        print("  there is nothing worth building on top of this result.")
        return False, r

    call = r.tool_calls[0]
    recovered = call.recovered_from_text
    label = f"tool call ({len(schemas)} schemas)"
    detail = f"{call.name}({json.dumps(call.arguments)[:80]})"

    if recovered:
        # Not a pass. It means the gateway returned prose that our recovery
        # code managed to parse — worth knowing, but a different world from a
        # properly populated tool_calls object.
        report(WARN, label, dt, detail + "  [recovered from TEXT, not native]")
    else:
        report(OK, label, dt, detail)

    if call.name != "move_to":
        print(f"       note: expected move_to, got {call.name} — the schemas got "
              "through, the model just chose differently")
    return True, r


def check_tool_roundtrip(client, scrub, report, prior):
    """3 — a tool result goes back in and the model continues the conversation."""
    call = prior.tool_calls[0]
    call_id = call.id or "call_smoke_1"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": COMMAND},
        {"role": "assistant", "content": prior.text or None,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": call.name,
                                      "arguments": json.dumps(call.arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "name": call.name,
         "content": json.dumps({"ok": True, "moved": ["SSMK"],
                                "state_summary": {"active": 1}})},
    ]

    t0 = time.time()
    r = client.chat(messages, tools=registry.schemas())
    dt = time.time() - t0

    if not (r.text.strip() or r.tool_calls):
        report(BAD, "tool role follow-up", dt, "accepted but answered with nothing")
        return False, r
    what = repr(r.text.strip()[:60]) if r.text.strip() else \
        f"another call: {r.tool_calls[0].name}"
    report(OK, "tool role follow-up", dt, what)
    return True, r


def build_client(args):
    """Returns (client, scrubber, error_message)."""
    if args.preset:
        try:
            return client_for(args.preset), Scrubber(), None
        except ModelError as e:
            return None, Scrubber(), str(e)

    key = os.environ.get(args.key_env, "").strip()
    if not key:
        return None, Scrubber(), (
            f"{args.key_env} is not set in the environment.\n"
            f"  Set it and re-run — never put the key in models.yaml or in source:\n"
            f"    export {args.key_env}='...'\n"
            f"  or:  source ~/.spheroswarm.env")

    client = ModelClient(
        model=args.model,
        base_url=args.base_url,
        api_key=key,
        # Backend-appropriate from the outset: num_ctx and the thinking toggle
        # are Ollama's, and sending them to a hosted model is at best ignored
        # and at worst a 400.
        num_ctx=0,
        thinking_style="none",
        max_tokens=args.max_tokens,
        temperature=args.temperature,      # None by default: this model 400s on it
        timeout=args.timeout,
    )
    return client, Scrubber(key), None


def explain(exc, client, scrub, key_env):
    """A readable line for the failure modes worth distinguishing."""
    text = scrub(str(exc))
    if isinstance(exc, ModelUnreachable):
        return (f"could not reach {client.base_url} — check the network or the "
                f"VPN.\n  The local `qwen3.5:9b` preset does not go through the "
                f"gateway and still works:\n    python -m llm.smoke "
                f"--preset qwen3.5:9b")
    if "401" in text or "403" in text:
        return (f"the gateway rejected the key in {key_env} (401/403). Check "
                f"that it is current — the value itself is not shown here.")
    if "404" in text:
        return (f"the gateway does not recognise the model string "
                f"{client.model!r} (404). Run with --list to see what it "
                f"reports as available.")
    if "429" in text:
        return "the gateway rate-limited this request (429). Wait and retry."
    return text


def list_models(client, scrub):
    """Ask the gateway what it exposes. Not every gateway implements this."""
    import requests
    try:
        r = client.session.get(f"{client.base_url}/models",
                               headers=client.headers(), timeout=15)
    except requests.RequestException as e:
        print(f"could not list models: {scrub(e)}")
        return 1
    if r.status_code >= 400:
        print(f"/models returned {r.status_code}: {scrub(r.text[:300])}")
        return 1
    try:
        data = r.json().get("data", [])
    except Exception:
        print(f"/models returned non-JSON: {scrub(r.text[:300])}")
        return 1
    if not data:
        print("the gateway reported no models")
        return 1
    print(f"{len(data)} model(s) reported by {client.base_url}:")
    for m in data:
        print("  " + str(m.get("id", m)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=GATEWAY_URL)
    p.add_argument("--model", default=GATEWAY_MODEL)
    p.add_argument("--key-env", default=KEY_ENV,
                   help="environment variable holding the gateway key")
    p.add_argument("--preset", default=None,
                   help="use a preset from models.yaml instead of the gateway")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None,
                   help="omitted entirely by default — Claude via Vertex "
                        "rejects it with a 400")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--list", action="store_true",
                   help="ask the gateway what models it exposes, then exit")
    args = p.parse_args(argv)

    client, scrub, err = build_client(args)
    if err:
        print(f"cannot start: {err}")
        return 2

    if args.list:
        return list_models(client, scrub)

    print(f"endpoint {client.base_url}")
    print(f"model    {client.model}")
    print(f"key      {args.key_env} found in the environment"
          if not args.preset else f"preset   {args.preset}")
    print()

    results = []

    def report(status, label, dt, detail=""):
        results.append((status, label))
        print(f"{status}  {label:<32} {_fmt(dt)}  {detail}")

    try:
        ok, _ = check_plain(client, scrub, report)
        if not ok:
            return 1

        ok, tool_response = check_tool_call(client, scrub, report)
        if not ok:
            return 1

        ok, _ = check_tool_roundtrip(client, scrub, report, tool_response)
        if not ok:
            return 1
    except (ModelUnreachable, ModelError) as e:
        print(f"\n FAIL  {explain(e, client, scrub, args.key_env)}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130

    warned = [l for s, l in results if s is WARN]
    print()
    if warned:
        print("PASSED WITH WARNINGS — tool calls arrived as text and had to be "
              "recovered.")
        print("The path works, but this model will need the recovery layer on "
              "every turn.")
    else:
        print("ALL CHECKS PASSED — the gateway passes tool schemas through and "
              "returns native tool calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
