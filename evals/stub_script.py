"""A canned model that plays the eval set competently.

This exists so the harness can be exercised — and tested in CI — with no
Ollama, no GPU and no network. It is a keyword matcher, not a language model:
it recognises the eval commands and answers them the way a good model should.

What stub results measure is the **harness**: that worlds are isolated, tools
are dispatched, assertions fire, and the report adds up. They say nothing about
any real model's ability, and a stub pass rate must never be quoted as one.
"""

import re

from llm.client import ModelResponse, ToolCall

CIRCLE = "points = [(cx+60*cos(i*2*pi/n), cy+60*sin(i*2*pi/n)) for i in range(n)]"


def _call(_tool, /, **args):
    """Positional-only tool name: `save_formation` takes its own `name` kwarg."""
    return ToolCall(name=_tool, arguments=args, id=f"c_{_tool}")


def _move(expression):
    return [_call("compute_points", expression=expression, execute=True)]


def plan(text):
    """Return a list of turns for one user command."""
    t = (text or "").lower().strip()

    def has(*words):
        return any(w in t for w in words)

    # -- memory ------------------------------------------------------------
    if has("save that as", "save it as", "save as"):
        m = re.search(r"save (?:that|it) as (\w+)", t) or re.search(r"save as (\w+)", t)
        name = m.group(1) if m else "shape"
        return [[_call("save_formation", name=name)], f"Saved as {name}."]

    if has("what formations", "which formations", "formations do you know"):
        return [[_call("list_formations")], "I know the wedge."]

    if has("do the", "recall") and has("wedge"):
        args = {"name": "wedge"}
        if has("twice as big", "bigger"):
            args["scale"] = 90
        if "180" in t:
            args["rotation"] = 180
        elif has("rotated"):
            m = re.search(r"(\d+)\s*degree", t)
            args["rotation"] = int(m.group(1)) if m else 90
        return [[_call("recall_formation", **args)], "Recalled the wedge."]

    # -- relative modification ---------------------------------------------
    if has("bigger", "larger", "expand"):
        return [[_call("transform", scale=1.6)], "Made it bigger."]

    if has("squash", "oval", "ellipse"):
        return [_move("points = [(cx+85*cos(i*2*pi/n), cy+35*sin(i*2*pi/n)) for i in range(n)]"),
                "Squashed into an oval."]

    if has("rotate") and not has("then"):
        m = re.search(r"(\d+)\s*degree", t)
        return [[_call("transform", rotate=int(m.group(1)) if m else 90)],
                "Rotated it."]

    if has("to the left"):
        m = re.search(r"(\d+)\s*cm", t)
        return [[_call("transform", translate=[-(int(m.group(1)) if m else 30), 0])],
                "Shifted left."]

    if has("to the right") and has("500", "drive"):
        # deliberately off the edge: the validator must clamp and we must say so
        return [[_call("transform", translate=[500, 0])],
                "That would leave the arena, so I clamped the targets to the "
                "right-hand wall instead."]

    if has("top right", "top-right"):
        return [[_call("transform", translate=[60, -40])], "Shifted to the top right."]

    if has("come back together", "come together", "regroup"):
        return [_move("points = [(cx+28*cos(i*2*pi/n), cy+28*sin(i*2*pi/n)) for i in range(n)]"),
                "Back together."]

    # -- sequences ----------------------------------------------------------
    if has("corners") and has("middle", "centre", "center"):
        return [_move("points = [(xmin+35+(i%2)*(width-70), ymin+35+(i//2%2)*(height-70)) for i in range(n)]"),
                [_call("wait_until_settled", timeout=8)],
                _move("points = [(cx+26*cos(i*2*pi/n), cy+26*sin(i*2*pi/n)) for i in range(n)]"),
                "Corners, then the middle."]

    if has("wait until") and has("rotate"):
        return [_move("points = [(xmin+30+i*(width-60)/(n-1), cy) for i in range(n)]"),
                [_call("wait_until_settled", timeout=8)],
                [_call("transform", rotate=90)],
                "Lined up, waited, then rotated."]

    if has("one at a time"):
        return [_move("points = [(cx+45*cos(i*2*pi/n), cy+45*sin(i*2*pi/n)) for i in range(n)]"),
                [_call("wait_until_settled", timeout=8)],
                "Each took a turn."]

    # -- safety --------------------------------------------------------------
    if has("same spot", "stack up", "on top of each other"):
        # try it, get refused by the validator, then explain rather than retry
        return [[_call("move_to", assign={"SSMK": [120, 90], "CRXS": [120, 90],
                                           "SYRX": [120, 90], "VHGR": [120, 90],
                                           "MLYS": [120, 90], "SNFR": [120, 90]})],
                "I can't do that — targets must be at least 20cm apart, so the "
                "robots cannot occupy the same spot."]

    # -- basic arrangements ---------------------------------------------------
    if has("circle"):
        return [_move(CIRCLE), "Formed a circle."]

    if has("spiral"):
        return [_move("points = [(cx+(20+i*11)*cos(i*2*pi/n), cy+(20+i*11)*sin(i*2*pi/n)) for i in range(n)]"),
                "Made a spiral."]

    if has("vertical line", "line down the middle"):
        return [_move("points = [(cx, ymin+25+i*(height-50)/(n-1)) for i in range(n)]"),
                "Vertical line."]

    if has("top wall") or (has("line") and has("top")):
        return [_move("points = [(xmin+30+i*(width-60)/(n-1), ymin+25) for i in range(n)]"),
                "Lined up along the top."]

    if has("horizontal line") or has("line"):
        return [_move("points = [(xmin+30+i*(width-60)/(n-1), cy) for i in range(n)]"),
                "Formed a line."]

    if has("spread out", "spread"):
        return [_move("points = [(xmin+35+(i%3)*(width-70)/2, ymin+35+(i//3)*(height-70)) for i in range(n)]"),
                "Spread out."]

    if has("gather", "centre", "center") and not has("corner"):
        return [_move("points = [(cx+26*cos(i*2*pi/n), cy+26*sin(i*2*pi/n)) for i in range(n)]"),
                "Gathered in the centre."]

    if has("square"):
        return [_move("points = [(cx+70*cos(pi/4+i*2*pi/n), cy+70*sin(pi/4+i*2*pi/n)) for i in range(n)]"),
                "Formed a square."]

    if has("triangle"):
        return [_move("points = [(cx+70*cos(-pi/2+i*2*pi/n), cy+70*sin(-pi/2+i*2*pi/n)) for i in range(n)]"),
                "Made a triangle."]

    if has("arc"):
        # stays inside the arena: an arc that runs off the floor is rejected
        # by the validator and then nothing moves at all
        return [_move("points = [(cx+85*cos(pi/6+i*(2*pi/3)/(n-1)), cy+15+55*sin(pi/6+i*(2*pi/3)/(n-1))) for i in range(n)]"),
                "Formed an arc."]

    if has("wedge"):
        return [_move("points = [(cx-40+i//2*45, cy+(-1)**i*(18+i//2*22)) for i in range(n)]"),
                "Formed a wedge."]

    if has("letter l", "spell"):
        return [_move("points = [(xmin+50, ymin+30+i*30) if i < n-2 else (xmin+50+(i-(n-3))*40, ymin+30+(n-3)*30) for i in range(n)]"),
                "Spelled an L."]

    if has("arrow"):
        return [_move("points = [(cx+50, cy)] + [(cx+10-j*30, cy-30+j*10) for j in range((n-1)//2)] + [(cx+10-j*30, cy+30-j*10) for j in range(n-1-(n-1)//2)]"),
                "Made an arrow."]

    if has("two groups", "split into two", "opposite sides"):
        # Tight clusters on purpose: the assertion uses single-linkage at 60cm,
        # so a group spread near that width splits on a few centimetres of
        # settling noise and the fixture ends up measuring the controller
        # rather than the harness.
        return [_move("points = [(xmin+45+(i%2)*26, cy-13+(i//2)*26) if i < n//2 else (xmax-45-(i%2)*26, cy-13+((i-n//2)//2)*26) for i in range(n)]"),
                "Split into two groups."]

    if re.search(r"\bv\b", t) or has("form a v"):
        return [_move("points = [(cx-60+i*24, cy-50+i*25) if i < n//2 else (cx-60+i*24, cy+50-(i-n//2)*25) for i in range(n)]"),
                "Formed a V."]

    if has("50cm", "nobody is within", "personal space"):
        return [_move("points = [(xmin+40+(i%3)*(width-80)/2, ymin+35+(i//3)*(height-70)) for i in range(n)]"),
                "Everyone has room."]

    if has("scatter", "random"):
        return [_move("points = [(xmin+30+((i*67)%int(width-60)), ymin+30+((i*43)%int(height-60))) for i in range(n)]"),
                "Scattered."]

    return ["I'm not sure what arrangement you mean."]


class ScriptedClient:
    """Plays `plan()` for whatever the last user message was."""

    model = "stub"

    def __init__(self, latency_s=0.0):
        self.latency_s = latency_s
        self.calls = []
        self._current = None
        self._queue = []

    def reachable(self, timeout=2.0):
        return True

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")

        if user != self._current:
            self._current = user
            self._queue = list(plan(user))

        turn = self._queue.pop(0) if self._queue else "Done."

        if isinstance(turn, str):
            return ModelResponse(text=turn)
        return ModelResponse(text="", tool_calls=list(turn))


def build_stub_client():
    return ScriptedClient()
