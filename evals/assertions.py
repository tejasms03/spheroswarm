"""Geometric assertions over a final arrangement.

Each helper takes the final positions and returns (passed, reason). The reason
is written for a human reading a failure column, so it says what the shape
actually was, not just that it was wrong.

Tolerances are relative to the arrangement's own size wherever that makes
sense: "is this a circle" should not depend on whether the circle is 40cm or
140cm across.
"""

import math

import numpy as np


def _pts(positions):
    return np.asarray(positions, dtype=float).reshape(-1, 2)


def _centroid(p):
    return p.mean(axis=0)


def _radii(p):
    return np.linalg.norm(p - _centroid(p), axis=1)


def _pairwise(p):
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d


# -- shape ------------------------------------------------------------------

def is_circle(tol=0.15):
    """Radii from the centroid all within `tol` (relative) of the mean radius."""
    def check(p, **kw):
        p = _pts(p)
        if len(p) < 3:
            return False, f"only {len(p)} robots — a circle needs at least 3"
        r = _radii(p)
        mean = r.mean()
        if mean < 1e-6:
            return False, "all robots are on top of each other"
        spread = float(np.abs(r - mean).max() / mean)
        return (spread <= tol,
                f"radii {r.min():.0f}-{r.max():.0f}cm about mean {mean:.0f}cm "
                f"(deviation {spread:.0%}, allowed {tol:.0%})")
    return check


def is_line(tol=0.1):
    """Perpendicular spread small relative to length: a thin, straight cloud."""
    def check(p, **kw):
        p = _pts(p)
        if len(p) < 3:
            return True, "fewer than 3 robots is trivially a line"
        centred = p - _centroid(p)
        _, s, vh = np.linalg.svd(centred, full_matrices=False)
        length = float(s[0])
        thickness = float(s[1])
        if length < 1e-6:
            return False, "all robots are on top of each other"
        ratio = thickness / length
        direction = vh[0]
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180
        return (ratio <= tol,
                f"thickness/length {ratio:.2f} (allowed {tol:.2f}), "
                f"axis {angle:.0f}deg")
    return check


def is_evenly_spaced(tol=0.2):
    """Nearest-neighbour distances all within `tol` (relative) of their mean."""
    def check(p, **kw):
        p = _pts(p)
        if len(p) < 3:
            return True, "fewer than 3 robots is trivially even"
        nn = _pairwise(p).min(axis=1)
        mean = nn.mean()
        if mean < 1e-6:
            return False, "robots are coincident"
        spread = float(np.abs(nn - mean).max() / mean)
        return (spread <= tol,
                f"nearest-neighbour gaps {nn.min():.0f}-{nn.max():.0f}cm "
                f"(deviation {spread:.0%}, allowed {tol:.0%})")
    return check


def is_arc(tol=0.25):
    """Curved but not closed: consistent radius about a fitted centre."""
    def check(p, **kw):
        p = _pts(p)
        if len(p) < 3:
            return False, f"only {len(p)} robots"
        # algebraic circle fit
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        b = (p ** 2).sum(axis=1)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return False, "could not fit a circle"
        centre = sol[:2]
        r = np.linalg.norm(p - centre, axis=1)
        mean = r.mean()
        if mean < 1e-6:
            return False, "degenerate fit"
        spread = float(np.abs(r - mean).max() / mean)
        return spread <= tol, f"fitted radius {mean:.0f}cm, deviation {spread:.0%}"
    return check


def is_spiral(growth=1.5):
    """Radius grows with angle about the spiral's own centre.

    The centre is searched for, not assumed to be the centroid: a spiral's
    points are denser on the outer arm, so its centroid sits well off the true
    origin and radii measured from there are not monotonic even for a perfect
    spiral. Requiring real growth (`growth`) is what stops a circle — whose
    radii are constant about its centre — from passing.
    """
    def check(p, **kw):
        p = _pts(p)
        if len(p) < 4:
            return False, f"only {len(p)} robots — too few to read as a spiral"

        # A straight line has monotonically increasing radius when viewed from
        # a centre off to one side, so growth alone is not enough: the points
        # must also wrap *around* the centre.
        if _line_ratio(p) < 0.08:
            return False, "the robots are collinear — that is a line, not a spiral"

        c0 = _centroid(p)
        reach = max(float(np.linalg.norm(p - c0, axis=1).max()), 1.0)

        def score(c):
            v = p - c
            r = np.linalg.norm(v, axis=1)
            if r.min() < 1e-6:
                return None
            ang = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
            order = np.argsort(ang)
            r_sorted, a_sorted = r[order], ang[order]
            gaps = np.diff(np.r_[a_sorted, a_sorted[0] + 360])
            span = 360.0 - float(gaps.max())
            if span < 200.0:
                return None                        # does not wrap the centre
            return (int((np.diff(r_sorted) > 0).sum()),
                    float(r.max() / r.min()), span)

        # Coarse sweep then two refinements. The window stays wide on the first
        # refinement because the coarse grid rarely lands near the true centre,
        # and a spiral's centre is nowhere near its centroid.
        best, centre = None, c0
        for scale, steps in ((1.0, 17), (0.5, 13), (0.15, 11)):
            round_best, round_c = None, centre
            for dx in np.linspace(-reach * scale, reach * scale, steps):
                for dy in np.linspace(-reach * scale, reach * scale, steps):
                    c = centre + np.array([dx, dy])
                    s = score(c)
                    if s is not None and (round_best is None or s > round_best):
                        round_best, round_c = s, c
            if round_best is None:
                break
            if best is None or round_best > best:
                best = round_best
            centre = round_c

        if best is None:
            return False, "no centre found that the robots wrap around"

        rises, ratio, span = best
        need = len(p) - 1                        # strictly increasing with angle
        ok = rises >= need and ratio >= growth
        return (ok,
                f"best centre ({centre[0]:.0f},{centre[1]:.0f}), wrap {span:.0f}deg: "
                f"radius rose on {rises}/{len(p)-1} steps (need {need}), "
                f"outer/inner {ratio:.1f}x (need {growth})")
    return check


def is_v_shape(tol=0.2):
    """Two straight arms meeting at a vertex."""
    def check(p, **kw):
        p = _pts(p)
        if len(p) < 5:
            return False, f"only {len(p)} robots — a V needs at least 5"
        best = None
        for i in range(len(p)):
            rest = np.delete(p, i, axis=0)
            d = np.linalg.norm(rest - p[i], axis=1)
            order = np.argsort(d)
            for split in range(2, len(rest) - 1):
                a = np.vstack([p[i], rest[order[:split]]])
                b = np.vstack([p[i], rest[order[split:]]])
                sa, _ = is_line(1.0)(a), None
                ra = _line_ratio(a)
                rb = _line_ratio(b)
                score = max(ra, rb)
                if best is None or score < best[0]:
                    best = (score, i, split)
        score = best[0]
        return score <= tol, f"best two-arm fit has thickness/length {score:.2f}"
    return check


def _line_ratio(p):
    p = _pts(p)
    if len(p) < 2:
        return 0.0
    centred = p - _centroid(p)
    _, s, _ = np.linalg.svd(centred, full_matrices=False)
    if s[0] < 1e-9:
        return 1.0
    return float(s[1] / s[0])


# -- placement ---------------------------------------------------------------

def centroid_near(point, tol=25.0):
    target = np.asarray(point, dtype=float)

    def check(p, **kw):
        c = _centroid(_pts(p))
        d = float(np.linalg.norm(c - target))
        return (d <= tol,
                f"centroid ({c[0]:.0f},{c[1]:.0f}) is {d:.0f}cm from "
                f"({target[0]:.0f},{target[1]:.0f}), allowed {tol:.0f}cm")
    return check


def bbox_within(region):
    """region: [xmin, ymin, xmax, ymax]."""
    xmin, ymin, xmax, ymax = region

    def check(p, **kw):
        p = _pts(p)
        lo, hi = p.min(axis=0), p.max(axis=0)
        inside = (lo[0] >= xmin and lo[1] >= ymin and hi[0] <= xmax and hi[1] <= ymax)
        return (inside,
                f"bbox ({lo[0]:.0f},{lo[1]:.0f})-({hi[0]:.0f},{hi[1]:.0f}) vs "
                f"allowed ({xmin},{ymin})-({xmax},{ymax})")
    return check


def all_within(axis, lo=None, hi=None):
    """Every robot's x or y inside a band. `axis` is 'x' or 'y'."""
    idx = 0 if axis == "x" else 1

    def check(p, **kw):
        vals = _pts(p)[:, idx]
        ok = True
        if lo is not None:
            ok = ok and bool((vals >= lo).all())
        if hi is not None:
            ok = ok and bool((vals <= hi).all())
        return ok, f"{axis} range {vals.min():.0f}..{vals.max():.0f}cm"
    return check


def axis_spread_below(axis, limit):
    idx = 0 if axis == "x" else 1

    def check(p, **kw):
        vals = _pts(p)[:, idx]
        spread = float(vals.max() - vals.min())
        return spread <= limit, f"{axis} spread {spread:.0f}cm (allowed {limit:.0f})"
    return check


def min_separation_above(limit):
    def check(p, **kw):
        d = _pairwise(_pts(p))
        m = float(d.min())
        return m >= limit, f"closest pair {m:.0f}cm apart (need {limit:.0f})"
    return check


def n_clusters(k, gap=60.0):
    """Exactly k groups, by single-linkage at `gap` centimetres."""
    def check(p, **kw):
        p = _pts(p)
        n = len(p)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= gap:
                    parent[find(i)] = find(j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        found = len(groups)
        return found == k, f"found {found} cluster(s) at a {gap:.0f}cm linkage, wanted {k}"
    return check


def clusters_apart(k, min_gap, gap=60.0):
    """k clusters whose centroids are all at least `min_gap` apart."""
    base = n_clusters(k, gap)

    def check(p, **kw):
        okk, reason = base(p)
        if not okk:
            return False, reason
        p = _pts(p)
        n = len(p)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= gap:
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        cents = [p[idx].mean(axis=0) for idx in groups.values()]
        worst = min(float(np.linalg.norm(a - b))
                    for i, a in enumerate(cents) for b in cents[i + 1:])
        return worst >= min_gap, f"{k} clusters, centroids {worst:.0f}cm apart (need {min_gap:.0f})"
    return check


# -- change relative to a `before` snapshot ----------------------------------

def all_moved_by(vector, tol=8.0):
    v = np.asarray(vector, dtype=float)

    def check(p, before=None, **kw):
        if before is None:
            return False, "no before-snapshot recorded"
        p, b = _pts(p), _pts(before)
        if len(p) != len(b):
            return False, f"robot count changed {len(b)} -> {len(p)}"
        # translation of a set, not of individuals: slots may be reassigned
        shift = _centroid(p) - _centroid(b)
        err = float(np.linalg.norm(shift - v))
        return (err <= tol,
                f"centroid moved ({shift[0]:+.0f},{shift[1]:+.0f})cm, "
                f"wanted ({v[0]:+.0f},{v[1]:+.0f}), off by {err:.0f}cm")
    return check


def spread_changed(direction="increase", factor=1.2):
    def check(p, before=None, **kw):
        if before is None:
            return False, "no before-snapshot recorded"
        a = float(_radii(_pts(before)).mean())
        b = float(_radii(_pts(p)).mean())
        if a < 1e-6:
            return False, "degenerate starting spread"
        ratio = b / a
        want = ratio >= factor if direction == "increase" else ratio <= 1 / factor
        return want, f"mean radius {a:.0f} -> {b:.0f}cm (x{ratio:.2f})"
    return check


def centroid_moved(direction, min_cm=25.0):
    """direction: one of 'north', 'south', 'east', 'west', 'northeast', ..."""
    want = {
        "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
        "northeast": (1, -1), "northwest": (-1, -1),
        "southeast": (1, 1), "southwest": (-1, 1),
    }[direction]

    def check(p, before=None, **kw):
        if before is None:
            return False, "no before-snapshot recorded"
        shift = _centroid(_pts(p)) - _centroid(_pts(before))
        okx = (shift[0] * want[0] >= min_cm) if want[0] else True
        oky = (shift[1] * want[1] >= min_cm) if want[1] else True
        return (okx and oky,
                f"centroid moved ({shift[0]:+.0f},{shift[1]:+.0f})cm, "
                f"wanted {direction} by at least {min_cm:.0f}cm")
    return check


def radii_grew(factor=1.2):
    return spread_changed("increase", factor)


def rotated_by(degrees, tol_deg=25.0):
    """Same radii, rotated: the shape turned rather than being rebuilt.

    Measured per robot. A rotationally symmetric arrangement — which a circle
    of evenly spaced robots is — carries no absolute angle in its point set: a
    hexagon turned 45 degrees is the same set of points as one turned -15. The
    rotation is only recoverable by asking where each *individual* robot went,
    and `transform` keeps every robot in its own slot, so that is well defined.
    """
    def check(p, before=None, before_map=None, after_map=None, **kw):
        if before is None:
            return False, "no before-snapshot recorded"

        if before_map and after_map:
            shared = [c for c in before_map if c in after_map]
            if not shared:
                return False, "no robots in common between the snapshots"
            b = np.array([before_map[c] for c in shared], dtype=float)
            p = np.array([after_map[c] for c in shared], dtype=float)
            per_robot = True
        else:
            p, b = _pts(p), _pts(before)
            per_robot = False

        ra, rb = np.sort(_radii(b)), np.sort(_radii(p))
        if len(ra) != len(rb):
            return False, "robot count changed"
        scale_err = float(np.abs(rb - ra).max() / max(ra.mean(), 1e-6))
        if scale_err > 0.35:
            return False, f"radii changed by {scale_err:.0%} — it was rebuilt, not rotated"

        ca, cb = _centroid(b), _centroid(p)
        aa = np.degrees(np.arctan2((b - ca)[:, 1], (b - ca)[:, 0]))
        ab = np.degrees(np.arctan2((p - cb)[:, 1], (p - cb)[:, 0]))
        if not per_robot:
            aa, ab = np.sort(aa % 360), np.sort(ab % 360)

        deltas = (ab - aa) % 360
        # circular median: pick the offset that best explains every robot
        best, best_spread = 0.0, float("inf")
        for cand in deltas:
            spread = float(np.abs((deltas - cand + 180) % 360 - 180).mean())
            if spread < best_spread:
                best, best_spread = float(cand), spread

        err = abs((best - degrees + 180) % 360 - 180)
        how = "per robot" if per_robot else "by shape only"
        return (err <= tol_deg,
                f"turned {best:.0f}deg ({how}, scatter {best_spread:.0f}deg), "
                f"wanted {degrees}deg")
    return check


def extents_differ(ratio=1.4):
    """An oval, not a circle: one axis clearly longer than the other."""
    def check(p, **kw):
        p = _pts(p)
        lo, hi = p.min(axis=0), p.max(axis=0)
        w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        if min(w, h) < 1e-6:
            return False, "degenerate extent"
        r = max(w, h) / min(w, h)
        return r >= ratio, f"extents {w:.0f}x{h:.0f}cm (ratio {r:.2f}, need {ratio})"
    return check


# -- library ------------------------------------------------------------------

def matches_saved(name, tol=0.2):
    """Current arrangement matches a saved formation, up to place/size/angle.

    Compares normalised shapes, which is the whole promise of the library: the
    same shape anywhere, at any scale, at any rotation.
    """
    def check(p, library=None, **kw):
        if library is None:
            return False, "no library passed to the assertion"
        entry = library.formations.get(name)
        if entry is None:
            return False, f"no saved formation called {name!r}"

        from tools.formations import normalise

        saved = np.asarray(entry["points"], dtype=float)
        current, _, _ = normalise(_pts(p))
        if len(saved) != len(current):
            return False, f"saved {len(saved)} points, now {len(current)}"

        best = _best_rotation_distance(saved, current)
        return best <= tol, f"normalised shape differs by {best:.2f} (allowed {tol})"
    return check


def _best_rotation_distance(a, b, steps=360):
    """Smallest mean point-to-point distance over rotations, matching greedily."""
    best = float("inf")
    for k in range(steps):
        ang = math.radians(k * 360.0 / steps)
        c, s = math.cos(ang), math.sin(ang)
        rot = b @ np.array([[c, -s], [s, c]]).T
        d = np.linalg.norm(a[:, None, :] - rot[None, :, :], axis=2)
        # greedy nearest matching is enough at this tolerance
        used, total = set(), 0.0
        for i in range(len(a)):
            order = np.argsort(d[i])
            for j in order:
                if j not in used:
                    used.add(int(j))
                    total += float(d[i, j])
                    break
        best = min(best, total / len(a))
    return best


def formation_exists(name):
    def check(p, library=None, **kw):
        if library is None:
            return False, "no library passed"
        names = list(library.formations)
        return name in names, f"library holds {names or 'nothing'}"
    return check


def called_tool(name):
    def check(p, tool_names=None, **kw):
        tool_names = tool_names or []
        return name in tool_names, f"tools called: {tool_names or 'none'}"
    return check


def did_not_move(tol=15.0):
    def check(p, before=None, **kw):
        if before is None:
            return False, "no before-snapshot recorded"
        moved = float(np.linalg.norm(_pts(p) - _pts(before), axis=1).max())
        return moved <= tol, f"furthest robot moved {moved:.0f}cm (allowed {tol:.0f})"
    return check


def refused(check_movement=True, tol=15.0):
    """The model must decline, not comply. Passing means refusing well."""
    def check(p, before=None, reply="", tool_names=None, agent_ok=True, **kw):
        words = ("cannot", "can't", "not possible", "impossible", "at least",
                 "20cm", "too close", "unable", "won't", "would overlap",
                 "apart", "refus", "no valid", "not able")
        said_no = any(w in (reply or "").lower() for w in words)
        if not check_movement:
            return said_no, f"reply was: {reply[:120]!r}"
        if before is None:
            return said_no, f"reply was: {reply[:120]!r}"
        moved = float(np.linalg.norm(_pts(p) - _pts(before), axis=1).max())
        return (said_no and moved <= tol,
                f"moved {moved:.0f}cm; reply was: {reply[:120]!r}")
    return check


def reported_clamping():
    """Targets were pulled inside the arena and the model said so."""
    def check(p, reply="", clamped_any=False, workspace=None, **kw):
        words = ("clamp", "edge", "wall", "boundary", "bounds", "as far as",
                 "limit", "arena", "outside", "adjust", "fit")
        mentioned = any(w in (reply or "").lower() for w in words)
        inside = True
        if workspace is not None:
            inside = all(workspace.is_valid_point(q) for q in _pts(p))
        return (inside and (mentioned or clamped_any),
                f"all inside={inside}, clamped={clamped_any}, "
                f"reply: {reply[:120]!r}")
    return check


# -- composition ---------------------------------------------------------------

def all_of(*checks):
    def check(p, **kw):
        reasons = []
        for c in checks:
            okc, why = c(p, **kw)
            reasons.append(("PASS" if okc else "FAIL") + " " + why)
            if not okc:
                return False, " | ".join(reasons)
        return True, " | ".join(reasons)
    return check


def valid_arrangement(min_sep=20.0):
    """The universal check: in bounds, off obstacles, not on top of each other."""
    def check(p, workspace=None, **kw):
        p = _pts(p)
        if workspace is not None:
            bad = [i for i, q in enumerate(p) if not workspace.is_valid_point(q)]
            if bad:
                return False, f"robots {bad} are out of bounds or on an obstacle"
        d = _pairwise(p)
        m = float(d.min())
        return m >= min_sep * 0.6, f"closest pair {m:.0f}cm"
    return check


# The names an eval case may use in `assertion:`.
REGISTRY = {
    "is_circle": is_circle,
    "is_line": is_line,
    "is_evenly_spaced": is_evenly_spaced,
    "is_arc": is_arc,
    "is_spiral": is_spiral,
    "is_v_shape": is_v_shape,
    "centroid_near": centroid_near,
    "bbox_within": bbox_within,
    "all_within": all_within,
    "axis_spread_below": axis_spread_below,
    "min_separation_above": min_separation_above,
    "n_clusters": n_clusters,
    "clusters_apart": clusters_apart,
    "all_moved_by": all_moved_by,
    "spread_changed": spread_changed,
    "spread_increased": lambda factor=1.2: spread_changed("increase", factor),
    "spread_decreased": lambda factor=1.2: spread_changed("decrease", factor),
    "centroid_moved": centroid_moved,
    "radii_grew": radii_grew,
    "rotated_by": rotated_by,
    "extents_differ": extents_differ,
    "matches_saved": matches_saved,
    "formation_exists": formation_exists,
    "called_tool": called_tool,
    "did_not_move": did_not_move,
    "refused": refused,
    "reported_clamping": reported_clamping,
    "valid_arrangement": valid_arrangement,
    "all_of": all_of,
}


def build(spec):
    """Turn a YAML assertion spec into a callable.

    Accepts `is_circle`, `is_circle(0.15)`, `{all_of: [...]}` and lists.
    """
    if spec is None:
        return lambda p, **kw: (True, "no assertion")
    if isinstance(spec, list):
        return all_of(*[build(s) for s in spec])
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError(f"assertion object must have one key: {spec!r}")
        (name, arg), = spec.items()
        fn = REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"unknown assertion {name!r}")
        if name == "all_of":
            return all_of(*[build(s) for s in arg])
        if isinstance(arg, dict):
            return fn(**arg)
        if isinstance(arg, list):
            return fn(*arg)
        return fn(arg)
    if isinstance(spec, str):
        return _from_string(spec)
    raise ValueError(f"cannot read assertion {spec!r}")


def _from_string(spec):
    spec = spec.strip()
    if "(" not in spec:
        fn = REGISTRY.get(spec)
        if fn is None:
            raise ValueError(f"unknown assertion {spec!r}")
        return fn()

    name, _, rest = spec.partition("(")
    name = name.strip()
    args_text = rest.rstrip().rstrip(")")
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"unknown assertion {name!r}")

    import ast

    def value(node):
        """Literals as themselves; bare words as strings.

        The eval set is written for humans, so `all_within(axis=y)` and
        `matches_saved(wedge)` have to mean what they obviously mean rather
        than being a NameError.
        """
        if isinstance(node, ast.Name):
            return node.id
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"could not read argument in assertion {spec!r}: {e}") from e

    args, kwargs = [], {}
    if args_text.strip():
        try:
            node = ast.parse(f"f({args_text})", mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"could not parse assertion {spec!r}: {e}") from e
        args = [value(a) for a in node.args]
        kwargs = {k.arg: value(k.value) for k in node.keywords}
    return fn(*args, **kwargs)
