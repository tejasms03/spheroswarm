"""The robot roster — a user-editable list of named robots, sim or real.

`roster.json` is the source of truth for what robots exist. Everything here is
read/validate/write; live fleet membership (`fleet/manager.py`) is a separate
concern that happens to start from this file.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vision.config import COLORS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "roster.json"

VALID_COLORS = frozenset(COLORS)
VALID_KINDS = frozenset({"sim", "real"})
MAX_ENABLED = 6


def _finite(value, default=0.0):
    """A bad offset in a hand-edited file must not stop the roster loading."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f % 360.0 if f == f and abs(f) != float("inf") else default


@dataclass
class RobotEntry:
    name: str
    code: str
    kind: str
    color: str
    ble_name: str | None = None
    enabled: bool = True
    # Degrees to add to every commanded heading for this robot. The camera
    # measures position but never orientation, and a Sphero drives in its own
    # aim frame, so the two differ by a constant nobody has measured. Left at 0
    # a robot still converges — the loop closes on position — but along a
    # curved path, and at ~90 degrees it orbits its target instead of arriving.
    heading_offset: float = 0.0

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name"),
            code=d.get("code"),
            kind=d.get("kind"),
            color=d.get("color"),
            ble_name=d.get("ble_name"),
            enabled=bool(d.get("enabled", True)),
            heading_offset=_finite(d.get("heading_offset"), 0.0),
        )

    def to_dict(self):
        return asdict(self)


def _err(msg):
    return {"ok": False, "error": msg}


def error_kinds(errors):
    """The category of each error, ignoring which items it names.

    Progress has to be recognisable. "duplicate ble_name: [A, B, C]" and
    "duplicate ble_name: [A, B]" are the same problem getting smaller, but as
    strings they are simply different — so comparing whole messages makes every
    partial repair look like a brand new fault, and the repair is refused one
    step before it can finish.
    """
    return {str(m).split(":", 1)[0] for m in errors}


def validate(entries):
    """Return a list of readable error strings. Empty list means valid."""
    errors = []

    for i, e in enumerate(entries):
        if not e.name:
            errors.append(f"entry {i}: missing name")
        if not e.code:
            errors.append(f"entry {i}: missing code")
        if e.kind not in VALID_KINDS:
            errors.append(f"{e.name or f'entry {i}'}: kind must be 'sim' or 'real', got {e.kind!r}")
        if e.color not in VALID_COLORS:
            errors.append(
                f"{e.name or f'entry {i}'}: color {e.color!r} is not one of {sorted(VALID_COLORS)}"
            )
        if e.kind == "real" and not e.ble_name:
            errors.append(f"{e.name or f'entry {i}'}: real robots need a ble_name")

    names = [e.name for e in entries if e.name]
    dup_names = {n for n in names if names.count(n) > 1}
    if dup_names:
        errors.append(f"duplicate names: {sorted(dup_names)}")

    codes = [e.code for e in entries if e.code]
    dup_codes = {c for c in codes if codes.count(c) > 1}
    if dup_codes:
        errors.append(f"duplicate codes: {sorted(dup_codes)}")

    # One physical ball, one roster slot. Two entries naming the same ble_name
    # both open a connection to it: the second either fails outright or wins the
    # link and leaves the first permanently "connecting…", and the fleet then
    # drives one ball from two sets of targets. Nothing downstream can detect
    # this — both entries look perfectly well-formed on their own.
    ble = [e.ble_name for e in entries if e.ble_name]
    dup_ble = {b for b in ble if ble.count(b) > 1}
    if dup_ble:
        errors.append(
            f"the same robot is bound to more than one entry: {sorted(dup_ble)}"
            " — rebind or clear the duplicates, one ball per row")

    enabled = [e for e in entries if e.enabled]
    if len(enabled) > MAX_ENABLED:
        errors.append(
            f"{len(enabled)} robots enabled, but the tracker only distinguishes "
            f"{MAX_ENABLED} hues — disable {len(enabled) - MAX_ENABLED} more"
        )

    enabled_colors = [e.color for e in enabled if e.color in VALID_COLORS]
    dup_colors = {c for c in enabled_colors if enabled_colors.count(c) > 1}
    if dup_colors:
        errors.append(f"duplicate colours among enabled robots: {sorted(dup_colors)}")

    return errors


class Roster:
    """In-memory roster backed by a JSON file. Never raises; check .errors."""

    def __init__(self, entries=None, path=DEFAULT_PATH):
        self.path = Path(path)
        self.entries = entries or []
        self.errors = []

    # -- persistence ---------------------------------------------------

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        path = Path(path)
        r = cls(path=path)
        if not path.exists():
            r.errors = [f"{path} does not exist"]
            return r
        try:
            raw = json.loads(path.read_text())
        except Exception as e:
            r.errors = [f"could not parse {path}: {e}"]
            return r
        try:
            r.entries = [RobotEntry.from_dict(d) for d in raw]
        except Exception as e:
            r.errors = [f"malformed roster entry: {e}"]
            r.entries = []
            return r
        r.errors = validate(r.entries)
        return r

    def save(self, path=None, allow_no_worse=False):
        """Validate then write. Returns a list of errors; writes only if empty.

        `allow_no_worse` lets a repair through: a roster that arrived on disk
        already broken can still be written as long as the write does not add
        a new problem. Without it the first bad file is permanent — every
        subsequent fix is refused by the very errors it is fixing.
        """
        errors = validate(self.entries)
        if errors and not (allow_no_worse
                           and error_kinds(errors) <= error_kinds(self.errors)):
            self.errors = errors
            return errors
        target = Path(path) if path else self.path
        try:
            target.write_text(json.dumps([e.to_dict() for e in self.entries], indent=2))
        except Exception as e:
            return [f"could not write {target}: {e}"]
        self.errors = errors
        return []

    # -- queries ---------------------------------------------------------

    def by_code(self, code):
        return next((e for e in self.entries if e.code == code), None)

    def enabled_entries(self):
        return [e for e in self.entries if e.enabled]

    # -- mutation (in-memory; caller decides when to .save()) ------------

    def add(self, entry):
        if isinstance(entry, dict):
            entry = RobotEntry.from_dict(entry)
        trial = [*self.entries, entry]
        errors = validate(trial)
        if errors:
            return errors
        self.entries = trial
        return []

    def remove(self, code):
        if self.by_code(code) is None:
            return [f"no robot with code {code!r}"]
        self.entries = [e for e in self.entries if e.code != code]
        return []

    def set_ble(self, code, ble_name):
        """Bind a physical ball to one entry, taking it off any other.

        Stealing rather than refusing: the trainer clicking a discovered ball
        onto a row means "this row is that ball", and any earlier claim on it
        is the stale one.

        Only errors this change *introduces* are reported. A roster that
        already has a duplicate in it elsewhere must not block the very
        operation that repairs duplicates — that is a file you can only fix by
        hand-editing, which is how it got into this state.
        """
        e = self.by_code(code)
        if e is None:
            return [f"no robot with code {code!r}"]

        before = validate(self.entries)
        trial, stolen = [], []
        for x in self.entries:
            d = x.to_dict()
            if x is e:
                d["ble_name"] = ble_name
                d["kind"] = "real" if ble_name else "sim"
            elif ble_name and x.ble_name == ble_name:
                # A real robot with no ball is not real.
                d["ble_name"], d["kind"] = None, "sim"
                stolen.append(x.code)
            trial.append(RobotEntry(**d))

        new = [m for m in validate(trial)
               if m.split(":", 1)[0] not in error_kinds(before)]
        if new:
            return new
        self.entries = trial
        self.stolen = stolen
        return []

    def clear_ble(self, code):
        """Unbind: the row keeps its name and colour, and becomes simulated."""
        e = self.by_code(code)
        if e is None:
            return [f"no robot with code {code!r}"]
        before = validate(self.entries)
        trial = [RobotEntry(**{**x.to_dict(), "ble_name": None, "kind": "sim"})
                 if x is e else x for x in self.entries]
        new = [m for m in validate(trial)
               if m.split(":", 1)[0] not in error_kinds(before)]
        if new:
            return new
        self.entries = trial
        self.stolen = []
        return []

    def set_color(self, code, color, swap=True):
        """Reassign which hue a robot wears. Returns a list of errors.

        Swaps by default, because refusing is useless at the point it matters.
        Six enabled robots hold all six hues, so every reassignment is a clash;
        a validator that says "cyan is taken" leaves the trainer disabling a
        robot to free a colour and re-enabling it afterwards. Swapping is what
        they meant, and it cannot fail validation — the set of hues in use is
        unchanged.
        """
        e = self.by_code(code)
        if e is None:
            return [f"no robot with code {code!r}"]
        if color not in VALID_COLORS:
            return [f"colour {color!r} is not one of {sorted(VALID_COLORS)}"]
        if e.color == color:
            return []

        # Only enabled robots contend for hues — the tracker never looks for
        # a disabled one — so a disabled robot simply takes the colour and
        # nobody has to give anything up.
        holder = (next((x for x in self.entries
                        if x is not e and x.color == color and x.enabled), None)
                  if e.enabled else None)
        trial = {x.code: x.color for x in self.entries}
        trial[e.code] = color
        if holder is not None:
            if not swap:
                return [f"{holder.code} already wears {color}"]
            trial[holder.code] = e.color

        candidate = [RobotEntry(**{**x.to_dict(), "color": trial[x.code]})
                     for x in self.entries]
        errors = validate(candidate)
        if errors:
            return errors
        self.entries = candidate
        return []

    def set_kind(self, code, kind, ble_name=None):
        e = self.by_code(code)
        if e is None:
            return [f"no robot with code {code!r}"]
        trial_entry = RobotEntry(**{**e.to_dict(), "kind": kind,
                                     "ble_name": ble_name if ble_name is not None else e.ble_name})
        trial = [trial_entry if x is e else x for x in self.entries]
        errors = validate(trial)
        if errors:
            return errors
        self.entries = trial
        return []
