"""Put our files into the framework's tree, and check the ones we cannot copy.

    python3.13 -m vlm.install --check     # say what is missing, change nothing
    python3.13 -m vlm.install             # copy them in

The framework is a separate project in a separate directory, and it is not
version controlled here. Three files have to live inside it because that is
where its own loaders look:

    Functions/Library/sphero_control.py       agents dispatch by
                                              importlib.import_module(
                                                  "Functions.Library.<library>")
    Functions/description/sphero_description   how a tool's schema is found
    Agents/Sphero Home/                        an agent is a directory of files

`vlm/framework/` holds the authoritative copies, under version control here,
and this copies them across. Without that, a fresh `git clone` of the
professor's repository silently removes half the integration and the only
symptom is an agent that cannot find its tools.

One change cannot be copied because it is an edit to THEIR file rather than a
new one: `Functions/Library/Agent/gemini.py` defaults to Gemini 2.5 Pro. It is
reported by --check instead, with the one-line change to make.
"""

import argparse
import filecmp
import os
import shutil
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.join(HERE, "framework")

DEFAULT_ROOT = os.path.expanduser("~/Downloads/Mobile-manipulation-with-VLMs-March")


def framework_root(given=None):
    root = given or os.environ.get("VLM_FRAMEWORK_ROOT") or DEFAULT_ROOT
    return os.path.abspath(os.path.expanduser(root))


def pairs(root):
    """(source, destination) for every file we own inside their tree."""
    out = []
    for base, _dirs, files in os.walk(MIRROR):
        for name in files:
            src = os.path.join(base, name)
            out.append((src, os.path.join(root, os.path.relpath(src, MIRROR))))
    return sorted(out)


GEMINI = "Functions/Library/Agent/gemini.py"
GEMINI_WANT = "VLM_AGENT_MODEL"
GEMINI_FIX = """\
    edit {path}, in call_gemini_agent's signature:

      -  model_ver='@vertexai/gemini-2.5-pro'
      +  model_ver=os.getenv('VLM_AGENT_MODEL',
      +                      '@vertexai/anthropic.claude-sonnet-5')

    Their gateway exposes both, so this is a preference rather than a fix.\
"""


def report(root, check_only):
    if not os.path.isdir(root):
        print(f"the framework is not at {root!r}")
        print("pass --root, or set VLM_FRAMEWORK_ROOT")
        return 1

    missing = stale = copied = 0
    for src, dst in pairs(root):
        rel = os.path.relpath(dst, root)
        if not os.path.exists(dst):
            state, missing = "MISSING", missing + 1
        elif not filecmp.cmp(src, dst, shallow=False):
            state, stale = "DIFFERS", stale + 1
        else:
            state = "ok"

        if state == "ok" or check_only:
            print(f"  {state:8} {rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"  copied   {rel}")

    gemini = os.path.join(root, GEMINI)
    if os.path.isfile(gemini):
        with open(gemini) as f:
            pointed = GEMINI_WANT in f.read()
        print(f"  {'ok' if pointed else 'THEIRS':8} {GEMINI}"
              f"{'' if pointed else '  (still defaults to Gemini)'}")
        if not pointed:
            print("\n" + GEMINI_FIX.format(path=GEMINI))

    if check_only and (missing or stale):
        print(f"\n{missing} missing, {stale} differing. "
              f"Run without --check to copy them in.")
        return 1
    if copied:
        print(f"\n{copied} copied into {root}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="the framework's top directory "
                        "(or set VLM_FRAMEWORK_ROOT)")
    p.add_argument("--check", action="store_true",
                   help="report only, change nothing")
    a = p.parse_args(argv)
    return report(framework_root(a.root), a.check)


if __name__ == "__main__":
    raise SystemExit(main())
