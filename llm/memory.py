"""What the swarm has learned from being asked things.

The formation library is a filing cabinet: you have to know a shape exists and
ask for it by name. This is the other kind of memory — nobody files anything
and nobody looks anything up. Commands that worked are recorded, and the ones
relevant to what is being asked *right now* are injected into the system
prompt, which is rebuilt every turn anyway.

The model never calls a tool to read this. It simply finds, already in front of
it, that "make a circle" was answered before with a particular expression, and
that the last time someone said "smaller" they meant about half.

Three rules keep it from becoming a liability:

**Only successes are remembered.** A failed attempt recalled later is a
suggestion to fail the same way. Failures are already handled better by the
validator handing its error straight back inside the same turn.

**Relevance is scored, not assumed.** Everything ever done would be thousands
of tokens and would bury the state that actually matters. A handful of the
closest matches go in, and nothing at all when nothing is close.

**The budget is hard.** Memory is the one part of the prompt that grows without
bound, so it gets a fixed ceiling and is trimmed to fit rather than allowed to
push the arena and the robot positions out.
"""

import json
import re
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "memory.json"
MAX_ENTRIES = 300          # kept on disk
MAX_INJECTED = 4           # shown to the model in any one turn
MAX_CHARS = 420            # hard ceiling on the injected block
MIN_SCORE = 0.24           # below this, nothing is close enough to be worth tokens

_WORD = re.compile(r"[a-z0-9_]+")
# Words that appear in almost every command and so carry no signal about which
# past command is the relevant one.
_STOP = frozenset("""
the a an and or to at of in on for with please can you now all every each
robot robots dragon dragons them it that this go make put move set
""".split())


_CODE = re.compile(r"^[a-z]{4}$")


def _words(text, identifiers=()):
    """Content words, with robot identifiers removed.

    Which robots a command named is an argument, not an intent — "swap SSMK and
    CRXS" and "swap Seasmoke and Caraxes" are the same request, and scoring
    them as different ones means the precedent is never found. Codes and names
    are dropped so what remains is the verb and the shape of the ask.
    """
    ids = {i.lower() for i in identifiers}
    out = set()
    for w in _WORD.findall((text or "").lower()):
        if w in _STOP or w in ids:
            continue
        out.add(w)
    return out


class Memory:
    """Recorded successes, scored against whatever is being asked now."""

    def __init__(self, path=DEFAULT_PATH, entries=None, max_entries=MAX_ENTRIES,
                 identifiers=()):
        self.path = Path(path) if path else None
        self.max_entries = max_entries
        self.entries = list(entries or [])
        # Robot codes and names, so they can be ignored when matching. Set from
        # the roster; harmless when empty.
        self.identifiers = set(identifiers)
        self.errors = []

    # -- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        m = cls(path=path)
        if not m.path or not m.path.exists():
            return m
        try:
            raw = json.loads(m.path.read_text())
            m.entries = [e for e in raw if isinstance(e, dict) and e.get("text")]
        except Exception as e:
            m.errors.append(f"could not read {m.path.name}: {e}")
        return m

    def save(self):
        if not self.path:
            return []
        try:
            self.path.write_text(json.dumps(self.entries[-self.max_entries:],
                                            indent=1))
            return []
        except Exception as e:
            return [f"could not write {self.path.name}: {e}"]

    # -- learning --------------------------------------------------------

    def record(self, text, tools, summary=None, save=True):
        """Remember one command that worked. Later identical asks replace it."""
        text = (text or "").strip()
        if not text or not tools:
            return None

        entry = {"text": text, "tools": list(tools),
                 "summary": (summary or "").strip()[:110],
                 "when": time.strftime("%Y-%m-%d %H:%M"), "uses": 1}

        key = text.lower()
        for i, old in enumerate(self.entries):
            if old.get("text", "").lower() == key:
                entry["uses"] = int(old.get("uses", 1)) + 1
                self.entries[i] = entry
                break
        else:
            self.entries.append(entry)

        del self.entries[:-self.max_entries]
        if save:
            self.save()
        return entry

    # -- recall ----------------------------------------------------------

    def score(self, text, entry):
        """Jaccard overlap on content words, nudged by how often it is used."""
        a = _words(text, self.identifiers)
        b = _words(entry.get("text", ""), self.identifiers)
        if not a or not b:
            return 0.0
        overlap = len(a & b) / len(a | b)
        if overlap == 0.0:
            return 0.0
        # A command asked ten times is more likely the intended precedent than
        # one asked once, but never enough to beat a genuinely closer match.
        return overlap * (1.0 + min(int(entry.get("uses", 1)), 10) * 0.02)

    def relevant(self, text, limit=MAX_INJECTED, min_score=MIN_SCORE):
        scored = [(self.score(text, e), e) for e in self.entries]
        scored = [(s, e) for s, e in scored if s >= min_score]
        scored.sort(key=lambda se: (-se[0], -int(se[1].get("uses", 1))))
        return [e for _, e in scored[:limit]]

    def lines(self, text, limit=MAX_INJECTED, max_chars=MAX_CHARS):
        """The block to inject, already trimmed to the budget."""
        picked = self.relevant(text, limit=limit)
        if not picked:
            return []
        out, used = [], 0
        for e in picked:
            tools = ", ".join(e["tools"][:3])
            line = f'- "{e["text"][:60]}" -> {tools}'
            if e.get("summary"):
                line += f' ({e["summary"][:50]})'
            if used + len(line) > max_chars:
                break
            out.append(line)
            used += len(line)
        return out

    def forget(self, text):
        """Drop anything matching, for when a remembered answer turns out wrong."""
        key = (text or "").strip().lower()
        before = len(self.entries)
        self.entries = [e for e in self.entries
                        if key not in e.get("text", "").lower()]
        self.save()
        return before - len(self.entries)

    def __len__(self):
        return len(self.entries)
