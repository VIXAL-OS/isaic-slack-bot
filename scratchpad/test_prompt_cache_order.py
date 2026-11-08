"""Offline checks for prompt-cache ordering (2026-11-07).

Guards the invariant that the SYSTEM PROMPT stays byte-identical across turns.
Every head but Claude does automatic prefix caching, and a prefix cache matches
the longest common prefix — so per-turn churn (thread-index ages, which tick
every minute; memory, which changes whenever a [note:] lands; the sparse-memory
nudge, which flips at the <3 boundary) placed in the system message invalidates
the cache for the whole prompt, conversation history included.

The churn now rides at the TAIL instead, via _append_volatile_context. These
checks fail if it is ever moved back ahead of the history.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_prompt_cache_order.py

Imports `bot` by default; set BOT_MODULE=isaic.core to run against the Slack
bot's copy (from the isaic-slack-bot repo root).
"""
import asyncio
import copy
import importlib
import os
import sys
import types

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))
from openai.types.chat import ChatCompletion  # noqa: E402

BotClass = getattr(bot, "ClaudeBot", None) or getattr(bot, "IsaicBot")
ok = fail = 0

STABLE = "You are a bot. Server context goes here."


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


# ---------------------------------------------------------------- the helper

append = BotClass._append_volatile_context

print("_append_volatile_context — shapes")

m = [{"role": "user", "content": "Sarah: hi"}]
check("string content → True, appended", append(m, "VOLATILE") is True
      and m[0]["content"].startswith("Sarah: hi") and "VOLATILE" in m[0]["content"])
check("appended block is framed as out-of-band context",
      "not part of the conversation" in m[0]["content"])

m = [{"role": "user", "content": [{"type": "text", "text": "Sarah: hi"}]}]
check("list content → True, extra text block appended", append(m, "VOLATILE") is True
      and len(m[0]["content"]) == 2 and m[0]["content"][0]["text"] == "Sarah: hi"
      and "VOLATILE" in m[0]["content"][1]["text"])

orig = [{"type": "text", "text": "Sarah: hi"}]
m = [{"role": "user", "content": orig}]
append(m, "VOLATILE")
check("list content: the caller's original list is not mutated", len(orig) == 1)

m = [{"role": "user", "content": "Sarah: hi"},
     {"role": "assistant", "content": "[Claude] hello"}]
before = copy.deepcopy(m)
check("last turn is assistant → False, messages untouched",
      append(m, "VOLATILE") is False and m == before)

check("empty messages → False", append([], "VOLATILE") is False)
check("empty volatile_context → False",
      append([{"role": "user", "content": "hi"}], "") is False)

m = [{"role": "user", "content": 42}]
check("non-str/non-list content → False", append(m, "VOLATILE") is False)

m = [{"role": "user", "content": "a"}, {"role": "user", "content": "Sarah: hi"}]
append(m, "VOLATILE")
check("only the NEWEST user turn is touched",
      m[0]["content"] == "a" and "VOLATILE" in m[1]["content"])


# ------------------------------------------------- the invariant, end to end

def chat_response() -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "pong"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    })


def run_turn(provider, thread_index, memory, note_count, history=None):
    """Drive the real _generate_response and capture the outgoing request."""
    captured = {}

    def create(**kw):
        captured.update(kw)
        return chat_response()

    B = BotClass.__new__(BotClass)
    B.tavily_client = None  # no tools → one call, no tool loop
    B._build_system_prompt = lambda *a, **k: STABLE
    B.clients = {provider.id: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))}

    mem = types.SimpleNamespace(
        get_context_string=lambda: memory,
        working=types.SimpleNamespace(notes=["n"] * note_count),
    )

    async def fetch_history(_channel):
        return copy.deepcopy(history or [{"role": "user", "content": "Sarah: hi"}])

    async def fetch_index(_channel):
        return thread_index

    B.manager = types.SimpleNamespace(
        fetch_thread_history=fetch_history,
        fetch_thread_index=fetch_index,
        memories={0: mem},
        reading_materials={},
    )
    channel = types.SimpleNamespace(id=1, parent_id=None)
    asyncio.run(B._generate_response(channel, 0, provider=provider, routing_reason=""))
    return captured


print("\nsystem prompt is stable across turns (shim path)")
p = copy.deepcopy(bot.DEEPSEEK_PROVIDER)
p.peak_windows_utc = ()

a = run_turn(p, "**Other recent threads**\n- **x** (5m ago)", "MEM: one note", 1)
b = run_turn(p, "**Other recent threads**\n- **x** (6m ago)", "MEM: one note", 1)
c = run_turn(p, "**Other recent threads**\n- **x** (6m ago)", "MEM: two notes", 5)

sys_a = a["messages"][0]
check("system message is first (it is the cached prefix)", sys_a["role"] == "system")
check("a thread age ticking 5m→6m does NOT change the system prompt",
      a["messages"][0]["content"] == b["messages"][0]["content"])
check("a new note + crossing the nudge boundary does NOT change it either",
      a["messages"][0]["content"] == c["messages"][0]["content"])
check("the system prompt is exactly the stable block",
      sys_a["content"] == STABLE)
check("no thread index leaked into the system prompt",
      "6m ago" not in c["messages"][0]["content"])
check("no memory leaked into the system prompt",
      "MEM:" not in c["messages"][0]["content"])
check("no sparse-memory nudge leaked into the system prompt",
      "working memory is pretty empty" not in a["messages"][0]["content"])

print("\nthe volatile context still reaches the model, at the tail")
tail = a["messages"][-1]["content"]
tail = tail if isinstance(tail, str) else " ".join(
    bl.get("text", "") for bl in tail)
check("thread index present in the final user turn", "5m ago" in tail)
check("memory present in the final user turn", "MEM: one note" in tail)
check("nudge present in the final user turn (1 note < 3)",
      "working memory is pretty empty" in tail)
nudged = run_turn(p, "", "", 5)["messages"][-1]["content"]
check("no nudge at 5 notes", "working memory is pretty empty" not in str(nudged))

print("\nfallback when the newest turn is not a user message")
hist = [{"role": "user", "content": "Sarah: hi"},
        {"role": "assistant", "content": "[Deepseek] hello"}]
f = run_turn(p, "IDX-FALLBACK", "MEM-FALLBACK", 1, history=hist)
check("volatile context falls back into the system prompt",
      "IDX-FALLBACK" in f["messages"][0]["content"]
      and "MEM-FALLBACK" in f["messages"][0]["content"])
check("fallback keeps the stable block first",
      f["messages"][0]["content"].startswith(STABLE))
check("fallback does not corrupt the assistant turn",
      f["messages"][-1]["content"] == "[Deepseek] hello")


# ------------------------------------------- Claude's explicit cache markers

print("\nClaude cache_control (explicit — Claude does not cache automatically)")


def claude_system_block(cache_system, reading_material=None):
    """Capture the `system` kwarg _generate_claude_response sends."""
    captured = {}

    class Boom(Exception):
        pass

    def create(**kw):
        captured.update(kw)
        raise Boom()

    B = BotClass.__new__(BotClass)
    B.claude_provider = copy.deepcopy(bot.CLAUDE_PROVIDER)
    B.claude_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create))
    B._build_reading_material_system_block = lambda rm: "FIC TEXT"
    try:
        asyncio.run(B._generate_claude_response(
            0, [{"role": "user", "content": "hi"}], STABLE,
            reading_material=reading_material, tools=[],
            cache_system=cache_system,
        ))
    except Exception:
        pass
    return captured.get("system")


s = claude_system_block(True)
check("cache_system=True, no fic → list of blocks", isinstance(s, list) and len(s) == 1)
check("cache_system=True, no fic → stable prefix carries cache_control",
      isinstance(s, list) and s[0].get("cache_control") == {"type": "ephemeral"}
      and s[0]["text"] == STABLE)

s = claude_system_block(False)
check("cache_system=False → bare string, no cache WRITE premium", s == STABLE)

s = claude_system_block(True, reading_material=object())
check("cache_system=True + fic → 2 blocks, fic first",
      isinstance(s, list) and len(s) == 2 and s[0]["text"] == "FIC TEXT")
check("cache_system=True + fic → BOTH blocks cached",
      isinstance(s, list)
      and s[0].get("cache_control") == {"type": "ephemeral"}
      and s[1].get("cache_control") == {"type": "ephemeral"})

s = claude_system_block(False, reading_material=object())
check("cache_system=False + fic → fic still cached (old behavior preserved)",
      isinstance(s, list) and s[0].get("cache_control") == {"type": "ephemeral"})
check("cache_system=False + fic → system block NOT cached",
      isinstance(s, list) and "cache_control" not in s[1])

print("\npanel/judge opt out of the system cache write")
import inspect  # noqa: E402
# _panel_complete became a thin selection-attribution wrapper on 2026-11-07,
# so the body to inspect lives in _panel_complete_dispatch where that exists.
# Read BOTH rather than only the wrapper: a future refactor that leaves a
# wrapper behind must fail here, not pass on an empty search.
panel_src = inspect.getsource(BotClass._panel_complete)
_dispatch = getattr(BotClass, "_panel_complete_dispatch", None)
if _dispatch is not None:
    panel_src += inspect.getsource(_dispatch)
check("_panel_complete passes cache_system=False on the Claude path",
      "cache_system=False" in panel_src)
check("and the cache-WRITE premium is named as the reason at the call",
      "cache" in panel_src.lower() and "write" in panel_src.lower())

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
