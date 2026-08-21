"""The agent loop: natural language in, robots moving out.

Three behaviours here matter more than the rest of the file.

**Validation errors go back verbatim.** `tools/validate.py` already returns
readable dicts naming the offending pair, the clamped points, the count
mismatch. Handing that straight back to the model recovers most first-attempt
failures on the second try. This retry loop does more work than any amount of
prompt tuning.

**The cap is hard, and hitting it stops the fleet.** A loop whose exit condition
the model controls is how a swarm ends up doing something forever while someone
hunts for the STOP button.

**Nothing escapes.** `command()` returns an AgentResult on every path,
including a model that is unreachable, a tool that raises, and a cancel from
the UI thread.
"""

import threading
import time
from dataclasses import dataclass, field

from tools import call, schemas

from .client import ModelError, ModelUnreachable
from .prompt import build_system_prompt

MAX_TOOL_CALLS = 8
HISTORY_TURNS = 6
RESULT_CHARS = 600           # per tool result fed back to the model

# Sensing tools answer a question the system prompt has already answered: the
# arena, the obstacles and every robot position are injected fresh each turn.
# The model calls them anyway — three `describe_scene` calls in one measured
# recall — and each is a full round trip, ~4s on an M1 Pro. Serve the first one
# (it may genuinely want the prose) and short-circuit the rest.
SENSING_TOOLS = frozenset({"get_state", "describe_scene"})

# Tools that advance the simulation themselves, for an unbounded stretch. They
# do their own per-tick locking, so the agent must not wrap them in the shared
# lock — see the dispatch site below.
SELF_TICKING_TOOLS = frozenset({"wait_until_settled"})

# Sent once when a turn produces no tool call at all. Observed against Sonnet:
# "make the letter A" intermittently comes back as the single word "Done." with
# nothing called and nothing moved — a false success, which is worse than a
# failure because nobody goes looking for it. A refusal is still a legitimate
# answer ("all six in one spot" cannot be done), so this asks rather than
# forces: a model that meant to refuse restates the refusal, and one that
# simply forgot to act now acts.
NUDGE = ("You did not call any tool, so nothing has moved and nothing has "
         "changed. If the request needs the robots to do something, call the "
         "tool now. If it genuinely needs no action — it is impossible, or "
         "already satisfied — say so plainly in one sentence, and say why.")

# Words that make a reply a *claim about the world* rather than an answer to a
# question. Deliberately narrow: "hi" -> "hello" and "there are six robots" are
# both perfectly good tool-free turns, and nudging them costs a round trip and
# tells the model nothing moved when nothing needed to. What this catches is
# the reply that asserts an action nobody performed.
_CLAIM_WORDS = (
    "done", "formed", "moved", "placed", "positioned", "arranged", "sent",
    "swapped", "gathered", "spread out", "stopped", "recalled", "saved",
    "orbiting", "circling", "following", "trailing", "patrolling",
    "on its way", "on their way", "heading to", "heading toward",
)


def _needs_nudge(text):
    """Should a turn that called nothing be given a second chance?

    Two cases, and the empty one is the important one. A reply that *claims* an
    action nobody performed is a false success. A reply that is empty, with no
    tool call either, is not a turn at all — the model simply did nothing, and
    that is never a legitimate answer to anything.

    What is deliberately NOT nudged: a real answer ("there are six robots") and
    a refusal ("six robots cannot occupy one point"). Both are complete turns
    that happen to need no tool.
    """
    low = (text or "").strip().lower()
    if not low:
        return True
    return any(w in low for w in _CLAIM_WORDS)
MAX_SENSING_CALLS = 1


@dataclass
class ToolRecord:
    name: str
    args: dict
    result: dict
    latency_s: float
    recovered_from_text: bool = False

    @property
    def ok(self):
        return bool(self.result.get("ok"))


@dataclass
class ModelCall:
    """One round trip, so a slow command can be attributed rather than guessed."""
    latency_s: float = 0.0
    prompt_tokens: int = None
    completion_tokens: int = None
    thinking: bool = None
    ttft_s: float = None            # only meaningful when streaming


@dataclass
class Timings:
    """Where the wall clock went. `wait_s` is robots moving, not the model."""
    total_s: float = 0.0
    model_s: float = 0.0
    tool_exec_s: float = 0.0
    wait_s: float = 0.0
    prompt_build_s: float = 0.0

    @property
    def overhead_s(self):
        return max(0.0, self.total_s - self.model_s - self.tool_exec_s
                   - self.wait_s - self.prompt_build_s)

    def table(self):
        rows = [("model", self.model_s), ("tool_exec", self.tool_exec_s),
                ("wait (robots moving)", self.wait_s),
                ("prompt_build", self.prompt_build_s),
                ("overhead", self.overhead_s)]
        width = max(len(n) for n, _ in rows)
        out = [f"{'segment':<{width}}  {'seconds':>8}  {'share':>6}"]
        out.append("-" * (width + 18))
        for name, v in rows:
            share = (v / self.total_s * 100) if self.total_s else 0.0
            out.append(f"{name:<{width}}  {v:8.2f}  {share:5.1f}%")
        out.append("-" * (width + 18))
        out.append(f"{'TOTAL':<{width}}  {self.total_s:8.2f}  100.0%")
        return "\n".join(out)


@dataclass
class AgentResult:
    ok: bool = False
    reply: str = ""
    tool_calls: list = field(default_factory=list)
    total_latency_s: float = 0.0
    retries: int = 0
    recovered_from_text_count: int = 0
    hit_cap: bool = False
    nudged: bool = False        # the model answered without calling anything
    error: str = None
    cancelled: bool = False
    timings: Timings = field(default_factory=Timings)
    model_calls: list = field(default_factory=list)
    skipped_sensing: int = 0

    @property
    def tool_names(self):
        return [t.name for t in self.tool_calls]


class Cancelled(Exception):
    """Raised internally when the UI aborts a run."""


def _compact(result, limit=RESULT_CHARS):
    """Shrink a tool result for the transcript.

    Errors survive in full — they are the thing the model has to act on. The
    bulky success payloads (every point, every robot) get trimmed, since the
    next system prompt carries fresh state anyway.
    """
    if not isinstance(result, dict):
        return {"ok": False, "error": f"tool returned {type(result).__name__}"}

    out = {"ok": result.get("ok"), "error": result.get("error")}
    for key in ("clamped", "problems", "targets", "points", "settled",
                "remaining", "description", "formations", "name", "count",
                "saved_count", "robots", "waited_s"):
        if key in result and result[key] not in (None, [], {}):
            out[key] = result[key]

    text = repr(out)
    if len(text) <= limit:
        return out

    # Too big: keep the fields that drive the next decision.
    keep = {"ok": out.get("ok"), "error": out.get("error")}
    for key in ("clamped", "problems", "settled", "remaining", "name"):
        if key in out:
            keep[key] = out[key]
    if "description" in out:
        keep["description"] = str(out["description"])[:limit]
    if "points" in out and "points" not in keep:
        pts = out["points"]
        keep["points_count"] = len(pts) if hasattr(pts, "__len__") else None
    return keep


def _is_validation_failure(result):
    return isinstance(result, dict) and result.get("ok") is False


class SwarmAgent:
    def __init__(self, client, ctx, max_tool_calls=MAX_TOOL_CALLS,
                 history_turns=HISTORY_TURNS, on_event=None, prompt_rules=None,
                 sim_lock=None):
        self.client = client
        self.ctx = ctx
        self.max_tool_calls = max_tool_calls
        self.history_turns = history_turns
        self.on_event = on_event
        self.prompt_rules = prompt_rules
        # `wait_until_settled` advances the simulation itself. When a UI is also
        # stepping it on another thread, the two must not run at once, or they
        # race on the position arrays and time runs at double speed. Sharing one
        # lock makes tool calls and the render loop take turns.
        self.sim_lock = sim_lock or threading.RLock()
        self.history = []                  # [{"role": ..., "content": ...}]
        self._cancel = threading.Event()

    # -- cancellation ------------------------------------------------------

    def cancel(self):
        """Abort an in-flight run. Safe to call from another thread."""
        self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise Cancelled()

    # -- events -------------------------------------------------------------

    def _emit(self, kind, **data):
        if self.on_event is None:
            return
        try:
            self.on_event({"type": kind, **data})
        except Exception:
            pass                            # a broken UI callback must not kill a run

    # -- message assembly ---------------------------------------------------

    def _messages(self, text, available=None):
        system = build_system_prompt(
            self.ctx, available=available, command=text,
            **({"rules": self.prompt_rules} if self.prompt_rules else {}))
        msgs = [{"role": "system", "content": system}]
        msgs += self.history[-self.history_turns * 2:]
        msgs.append({"role": "user", "content": text})
        return msgs

    def _learn(self, text, records, reply):
        """File one command that worked, for next time.

        Only successes, and only ones that did something: a turn that called
        nothing is not a precedent for anything, and a failure recalled later
        is a suggestion to fail the same way.
        """
        memory = getattr(self.ctx, "memory", None)
        if memory is None:
            return
        acted = [r for r in records if r.ok and r.name not in SENSING_TOOLS]
        if not acted:
            return
        try:
            memory.record(text, [r.name for r in acted], summary=reply)
        except Exception:
            pass            # memory is a convenience; it must never break a run

    def _remember(self, user_text, summary):
        """Store the exchange compactly.

        Full tool payloads would blow the context in three turns, and the next
        system prompt carries live state anyway. What has to survive is enough
        for "now rotate it 45 degrees" to resolve.
        """
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": summary or "(no reply)"})
        del self.history[:-self.history_turns * 2]

    def reset(self):
        self.history.clear()

    # -- the loop ------------------------------------------------------------

    def command(self, text):
        self._cancel.clear()
        started = time.time()
        result = AgentResult()

        try:
            result = self._run(text, started)
        except Cancelled:
            self.ctx.stop_all()
            result = AgentResult(ok=False, cancelled=True, error="cancelled",
                                  reply="Stopped.",
                                  total_latency_s=time.time() - started)
            self._emit("cancelled")
        except ModelUnreachable as e:
            result = AgentResult(ok=False, error=str(e),
                                  reply="The model is unreachable.",
                                  total_latency_s=time.time() - started)
            self._emit("error", error=str(e))
        except ModelError as e:
            result = AgentResult(ok=False, error=str(e),
                                  reply="The model returned an error.",
                                  total_latency_s=time.time() - started)
            self._emit("error", error=str(e))
        except Exception as e:                # nothing escapes into the UI thread
            result = AgentResult(ok=False, error=f"{type(e).__name__}: {e}",
                                  reply="Something went wrong.",
                                  total_latency_s=time.time() - started)
            self._emit("error", error=str(e))

        self._emit("done", result=result)
        return result

    def _run(self, text, started):
        self._emit("start", text=text)
        timings = Timings()
        calls = []

        records, retries, recovered = [], 0, 0
        nudged = False
        sensing_used, skipped_sensing = 0, 0
        # Chosen once from the command, then reused: swapping the tool list
        # mid-conversation would invalidate the model's own earlier calls.
        tool_schemas = schemas(text)

        # The prompt is built from that same set, so it never advertises a tool
        # whose schema is not being sent. Naming one anyway is worse than
        # silence: dispatch does not check what was offered, so the model calls
        # it, guesses the arguments, and spends the whole budget failing.
        t0 = time.time()
        messages = self._messages(text, available=[t["name"] for t in tool_schemas])
        timings.prompt_build_s += time.time() - t0

        while True:
            self._check_cancel()
            self._emit("thinking", calls_so_far=len(records))

            t0 = time.time()
            response = self.client.chat(
                messages, tool_schemas,
                **({"on_token": lambda s: self._emit("token", text=s)}
                   if getattr(self.client, "stream", False) else {}))
            timings.model_s += time.time() - t0
            calls.append(ModelCall(
                latency_s=round(response.latency_s or (time.time() - t0), 3),
                prompt_tokens=(response.tokens or {}).get("prompt"),
                completion_tokens=(response.tokens or {}).get("completion"),
                thinking=getattr(self.client, "thinking", None),
                ttft_s=getattr(response, "ttft_s", None)))
            recovered += response.recovered_count

            if not response.tool_calls:
                # Nothing was called. Give it exactly one chance to either act
                # or own the fact that it is not going to.
                if not records and not nudged and _needs_nudge(response.text):
                    nudged = True
                    self._emit("nudge", text=response.text or "")
                    messages.append(self._assistant_message(response))
                    messages.append({"role": "user", "content": NUDGE})
                    continue

                reply = response.text or (
                    "No tool was called, so nothing moved."
                    if not records else "Done.")
                # Asked once to act or explain, and it did neither: still
                # claiming completion, or still saying nothing. Say plainly
                # that nothing happened rather than passing the claim on — a
                # false success is only harmful while nobody notices it.
                if nudged and not records and _needs_nudge(response.text):
                    reply = (reply.rstrip(". ")
                             + " — but no tool was called, so nothing moved.")
                self._remember(text, reply)
                self._learn(text, records, reply)
                self._emit("reply", text=reply)
                timings.total_s = time.time() - started
                return AgentResult(
                    ok=True, reply=reply, tool_calls=records,
                    total_latency_s=timings.total_s, retries=retries,
                    recovered_from_text_count=recovered, hit_cap=False,
                    nudged=nudged, timings=timings, model_calls=calls,
                    skipped_sensing=skipped_sensing)

            # A turn that only re-senses, after something has already acted, is
            # the model failing to notice it is done. Skipping the execution
            # does not help — the cost is this round trip and the next one. Cut
            # the loop instead: it is worth ~4s per redundant call and was the
            # difference between 3 and 8 calls on a plain formation recall.
            if (records and all(tc.name in SENSING_TOOLS for tc in response.tool_calls)
                    and any(r.name not in SENSING_TOOLS and r.ok for r in records)):
                reply = response.text or self._summarise(records)
                self._remember(text, reply)
                self._learn(text, records, reply)
                self._emit("reply", text=reply)
                timings.total_s = time.time() - started
                return AgentResult(
                    ok=True, reply=reply, tool_calls=records,
                    total_latency_s=timings.total_s, retries=retries,
                    recovered_from_text_count=recovered, hit_cap=False,
                    timings=timings, model_calls=calls,
                    skipped_sensing=skipped_sensing + len(response.tool_calls))

            messages.append(self._assistant_message(response))

            for tc in response.tool_calls:
                self._check_cancel()

                if len(records) >= self.max_tool_calls:
                    return self._give_up(text, records, started, retries,
                                         recovered, timings, calls, skipped_sensing)

                # Sensing after something has already acted is always waste:
                # the next system prompt carries the result anyway. Sensing
                # *first* can be legitimate ("what formations do you know?"),
                # so only the trailing ones are cut.
                acted = any(r.name not in SENSING_TOOLS for r in records)
                if tc.name in SENSING_TOOLS and (
                        acted or sensing_used >= MAX_SENSING_CALLS):
                    result = {
                        "ok": True, "error": None,
                        "state_summary": self.ctx.state_summary(),
                        "note": ("state is already in your system prompt, fresh "
                                 "every turn — this call was skipped. Act on "
                                 "what you have."),
                        "skipped": True,
                    }
                    records.append(ToolRecord(name=tc.name, args=tc.arguments,
                                              result=result, latency_s=0.0))
                    messages.append(self._tool_message(tc, result))
                    skipped_sensing += 1
                    self._emit("tool_result", name=tc.name, result=result,
                               latency_s=0.0, ok=True)
                    continue

                if tc.name in SENSING_TOOLS:
                    sensing_used += 1

                self._emit("tool_start", name=tc.name, args=tc.arguments)
                t0 = time.time()
                if tc.name in SELF_TICKING_TOOLS:
                    # This tool advances the world itself, for as long as it
                    # takes. Holding the lock around the whole call would mean
                    # holding it for twenty seconds — the window stops
                    # redrawing and no other robot can be commanded until it
                    # returns. It takes the lock per tick instead, inside
                    # ctx.tick, so everyone else gets a turn between them.
                    result = call(self.ctx, tc.name, tc.arguments)
                else:
                    with self.sim_lock:
                        result = call(self.ctx, tc.name, tc.arguments)
                latency = time.time() - t0
                # Waiting is robots physically moving. Counting it as agent
                # latency makes the model look slow for something no model
                # change can touch.
                if tc.name == "wait_until_settled":
                    timings.wait_s += latency
                else:
                    timings.tool_exec_s += latency

                if _is_validation_failure(result):
                    retries += 1

                rec = ToolRecord(name=tc.name, args=tc.arguments, result=result,
                                 latency_s=latency,
                                 recovered_from_text=tc.recovered_from_text)
                records.append(rec)
                self._emit("tool_result", name=tc.name, result=result,
                           latency_s=latency, ok=rec.ok)

                messages.append(self._tool_message(tc, result))

            if len(records) >= self.max_tool_calls:
                return self._give_up(text, records, started, retries, recovered,
                                     timings, calls, skipped_sensing)

    def _summarise(self, records):
        """A reply for a turn that ended without the model writing one."""
        did = [r.name for r in records if r.name not in SENSING_TOOLS and r.ok]
        return f"Done ({', '.join(did)})." if did else "Done."

    def _assistant_message(self, response):
        """Echo the model's tool calls back in the shape the API expects."""
        import json

        return {
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {"id": tc.id or f"call_{i}", "type": "function",
                 "function": {"name": tc.name,
                               "arguments": json.dumps(tc.arguments)}}
                for i, tc in enumerate(response.tool_calls)
            ],
        }

    def _tool_message(self, tool_call, result):
        import json

        return {
            "role": "tool",
            "tool_call_id": tool_call.id or f"call_{tool_call.name}",
            "name": tool_call.name,
            "content": json.dumps(_compact(result), default=str),
        }

    def _give_up(self, text, records, started, retries, recovered,
                 timings=None, calls=None, skipped=0):
        """Cap reached. Halt the fleet before returning — do not leave it rolling."""
        self.ctx.stop_all()
        reply = (f"Stopped after {len(records)} tool calls without finishing. "
                 f"The robots have been halted.")
        self._remember(text, reply)
        self._emit("cap", calls=len(records))
        timings = timings or Timings()
        timings.total_s = time.time() - started
        return AgentResult(ok=False, reply=reply, tool_calls=records,
                           total_latency_s=timings.total_s, retries=retries,
                           recovered_from_text_count=recovered, hit_cap=True,
                           error=f"hit the {self.max_tool_calls}-call cap",
                           timings=timings, model_calls=calls or [],
                           skipped_sensing=skipped)
