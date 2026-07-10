"""Offline checks for _extract_notes (port of upstream 1bdad30). No keys, no Slack.

The bug being guarded: _web_search (Claude native), the grounded-Gemini !search
branch, and the !summarize path all returned model text WITHOUT stripping
[note:] working-note markers, so a note tacked onto a !search answer leaked into
the posted message. The old inline note_pattern was also case-sensitive, so
[Note:] / [NOTE:] were captured by neither memory nor the stripper.
"""
import sys, os, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaic import core

B = core.IsaicBot
ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL {name}: got {got!r}, want {want!r}")


class FakeWorking:
    def __init__(self): self.added = []
    def add(self, k, v): self.added.append((k, v)); return True


def bot_with_memory():
    """Minimal stand-in exposing just what _extract_notes touches."""
    working = FakeWorking()
    mem = types.SimpleNamespace(working=working)
    stub = types.SimpleNamespace(
        manager=types.SimpleNamespace(memories={1: mem}),
        _NOTE_RE=B._NOTE_RE,
    )
    return stub, working


def extract(text):
    stub, working = bot_with_memory()
    out = B._extract_notes(stub, text, 1)
    return out, working.added


print("capture + strip")
out, added = extract("Here you go. [note: deadline: late spring] Enjoy.")
check("note stripped from text", "[note:" in out, False)
check("note captured", added, [("deadline", "late spring")])
check("surrounding text kept", "Here you go." in out and "Enjoy." in out, True)

print("case-insensitivity (the old inline regex missed these)")
for tag in ("[Note: k: v]", "[NOTE: k: v]", "[NoTe: k: v]"):
    out, added = extract(f"answer {tag} tail")
    check(f"{tag} stripped", "note" in out.lower() and "k: v" in out, False)
    check(f"{tag} captured", added, [("k", "v")])

print("idempotency (it doubles as a safety net on already-clean paths)")
once, _ = extract("answer [note: a: b] tail")
stub, working = bot_with_memory()
twice = B._extract_notes(stub, once, 1)
check("second pass is a no-op", twice, once)
check("second pass captures nothing", working.added, [])

print("no-op fast path / edge cases")
check("clean text untouched", extract("just a normal answer")[0], "just a normal answer")
check("clean text captures nothing", extract("just a normal answer")[1], [])
check("empty string", extract("")[0], "")
check("None-safe", B._extract_notes(bot_with_memory()[0], None, 1), None)
# Key group excludes ']' (upstream widened it from [^:]+), so a bracketed tail
# can't swallow the terminator.
out, added = extract("x [note: key: value] y [note: k2: v2] z")
check("two notes captured", added, [("key", "value"), ("k2", "v2")])
check("both stripped", "[note:" in out, False)

print("multiline value stays inside the tag")
out, added = extract("a [note: k: v] b")
check("value not greedy past ]", added, [("k", "v")])

print("regex is compiled + case-insensitive")
import re as _re
check("_NOTE_RE has IGNORECASE", bool(B._NOTE_RE.flags & _re.IGNORECASE), True)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
