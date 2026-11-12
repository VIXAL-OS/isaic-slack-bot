"""Offline checks for the Claude streaming guard (2026-11-12).

A non-streaming Messages request whose max_tokens implies a worst case over
ten minutes is refused by the SDK BEFORE anything is sent, so the 64,000-token
xhigh/max effort tier killed every deep Claude turn:

    ValueError: Streaming is required for operations that may take longer
                than 10 minutes

These checks pin the boundary against the INSTALLED SDK (not against arithmetic
copied into a comment), pin which of our tiers sit either side of it, and pin
that both call sites in _generate_claude_response go through the one helper.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_claude_streaming.py

Imports `bot` by default; set BOT_MODULE=isaic.core for the Slack bot's copy.
"""
import asyncio
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))
BotClass = getattr(bot, "ClaudeBot", None) or getattr(bot, "IsaicBot")

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


CEILING = BotClass.CLAUDE_NONSTREAM_MAX_TOKENS

print("the boundary, as the INSTALLED SDK enforces it")
# Deliberately the private helper the live traceback named: it is the contract
# we depend on, so if a future SDK renames or re-tunes it, this fails here
# instead of in a Discord channel.
import anthropic  # noqa: E402

client = anthropic.Anthropic(api_key="not-a-real-key")
calc = getattr(client, "_calculate_nonstreaming_timeout", None)
check("the SDK still exposes _calculate_nonstreaming_timeout", calc is not None)
if calc is not None:
    check(f"it accepts our ceiling ({CEILING:,}) non-streaming",
          not raises(lambda: calc(CEILING, None)))
    check("and refuses one token above it",
          raises(lambda: calc(CEILING + 1, None)))
    check("so the ceiling is exactly the SDK's, not a guess",
          not raises(lambda: calc(CEILING, None))
          and raises(lambda: calc(CEILING + 1, None)))

print("\nwhich of our effort tiers sit either side of it")
plain = BotClass.CLAUDE_PLAIN_MAX_TOKENS
think = BotClass.CLAUDE_THINKING_MAX_TOKENS
check(f"plain turns ({plain:,}) are safe non-streaming", plain <= CEILING)
check(f"high-effort turns ({think:,}) are safe too", think <= CEILING)
check("but the xhigh/max tier (64,000) is NOT -- this was the live failure",
      64000 > CEILING)
if calc is not None:
    check("confirmed against the SDK: 64,000 non-streaming still raises",
          raises(lambda: calc(64000, None)))

src = inspect.getsource(BotClass._generate_claude_response)
check("the 64,000 tier is still the one bound to xhigh/max",
      "64000" in src and "xhigh" in src)

print("\n_claude_message routes on the budget")


class FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return "STREAMED"


class FakeMessages:
    def __init__(self):
        self.created, self.streamed = [], []

    def create(self, **kw):
        self.created.append(kw)
        return "CREATED"

    def stream(self, **kw):
        self.streamed.append(kw)
        return FakeStream()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def call(max_tokens, **extra):
    b = BotClass.__new__(BotClass)
    b.claude_client = FakeClient()
    out = asyncio.run(b._claude_message(max_tokens=max_tokens, **extra))
    return out, b.claude_client.messages


out, m = call(plain)
check("a plain budget uses create, not stream",
      out == "CREATED" and len(m.created) == 1 and not m.streamed)
out, m = call(CEILING)
check("exactly at the ceiling still uses create",
      out == "CREATED" and len(m.created) == 1 and not m.streamed)
out, m = call(CEILING + 1)
check("one token over switches to stream",
      out == "STREAMED" and len(m.streamed) == 1 and not m.created)
out, m = call(64000)
check("the xhigh/max tier streams",
      out == "STREAMED" and len(m.streamed) == 1 and not m.created)
check("and the stream's final message is what the caller gets",
      out == "STREAMED")

out, m = call(64000, messages=[{"role": "user", "content": "hi"}],
              system="sys", thinking={"type": "adaptive"},
              extra_body={"output_config": {"effort": "max"}})
check("every kwarg is forwarded to stream untouched",
      m.streamed[0]["system"] == "sys"
      and m.streamed[0]["thinking"] == {"type": "adaptive"}
      and m.streamed[0]["extra_body"]["output_config"]["effort"] == "max"
      and m.streamed[0]["max_tokens"] == 64000)

# A call with no max_tokens at all must not accidentally stream (nor crash).
b = BotClass.__new__(BotClass)
b.claude_client = FakeClient()
out = asyncio.run(b._claude_message(messages=[]))
check("a call without max_tokens defaults to create, not stream",
      out == "CREATED" and not b.claude_client.messages.streamed)

print("\nboth call sites go through the helper (the regression that matters)")
check("no bare messages.create survives in _generate_claude_response",
      "messages.create" not in src)
check("and it calls the helper twice -- first request AND tool-loop retry",
      src.count("_claude_message(") == 2)
# The helper itself is allowed to call create; nothing else on the Claude path
# should, so a third call site added later shows up here.
helper_src = inspect.getsource(BotClass._claude_message)
check("the helper is the only place that calls messages.create",
      "messages.create" in helper_src)
check("and the only place that calls messages.stream",
      "messages.stream" in helper_src and "messages.stream" not in src)

print("\nthe small fixed-budget calls were never affected")
for name, budget in (("!summarize", 4096), ("participation gate", 96)):
    if calc is not None:
        check(f"{name} ({budget:,}) was always fine non-streaming",
              not raises(lambda b=budget: calc(b, None)))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
