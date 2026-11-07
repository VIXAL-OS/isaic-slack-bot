"""Offline checks for the Kimi reasoning-effort tiers (2026-11-07).

Moonshot's docs: "The Kimi K3 series always enables thinking ... and uses the
top-level `reasoning_effort` field (`low`, `high`, or `max`; default `max`)."

So omitting the param on a plain turn is NOT "thinking off" — it is "thinking at
MAX", silently, on our priciest open head. The shim now sends an explicit tier
per turn: plain_reasoning_effort on a plain turn, think_reasoning_effort on a
`!think` turn.

That default is documented for MOONSHOT's hosted K3. Fireworks serves the open
weights and may not implement the param at all, and a param it rejects fails the
whole turn — so the plain tier is set on whichever route is Moonshot and left
None on the Fireworks route. The two repos invert which of those is the DEFAULT
(Hydra defaults to Moonshot with a fireworks backend; the Slack bot defaults to
Fireworks with a moonshot backend), so these checks resolve the routes by
base_url rather than assuming either arrangement.

Clearing a tier only works because the reasoning fields are in
_BACKEND_NULLABLE_FIELDS — without that, _apply_backend skips a None override
and the backend silently inherits the other route's tier.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_kimi_effort_tiers.py

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


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def resolved(backend=None):
    """KIMI_PROVIDER as the registry would leave it for `backend` (None=default)."""
    p = copy.deepcopy(bot.KIMI_PROVIDER)
    if backend is not None:
        bot.ProviderRegistry._apply_backend(None, p, backend, {})
    return p


def is_moonshot(p):
    return "moonshot.ai" in (p.base_url or "")


# Resolve both routes without assuming which repo we are in.
default_route = resolved()
other_name = next(iter(bot.KIMI_PROVIDER.backends))
other_route = resolved(other_name)
moonshot_route = default_route if is_moonshot(default_route) else other_route
fireworks_route = other_route if is_moonshot(default_route) else default_route

print(f"routes: default={'moonshot' if is_moonshot(default_route) else 'fireworks'}, "
      f"backend='{other_name}'")
check("one route is Moonshot and the other is not",
      is_moonshot(moonshot_route) and not is_moonshot(fireworks_route))

print("\nthe Moonshot route carries both tiers")
check("Moonshot thinking tier is max", moonshot_route.think_reasoning_effort == "max")
# WHICH tier a plain turn asks for is a per-bot product decision, not an
# invariant: the Discord bot is a chat bot and wants "low", the Slack lab bot
# wants "high". So pin that a tier IS set and that it is a value the docs say
# K3 accepts — pinning the literal would just encode one bot's choice.
PLAIN_TIER = moonshot_route.plain_reasoning_effort
check("Moonshot plain tier is SET (None would silently mean max)",
      PLAIN_TIER is not None)
check(f"the plain value ({PLAIN_TIER!r}) is one the docs say K3 accepts",
      PLAIN_TIER in ("low", "high", "max"))

print("\nthe Fireworks route sends no plain tier (unverified there)")
check("Fireworks plain tier is None",
      fireworks_route.plain_reasoning_effort is None)

print("\nno other head sends reasoning_effort at all")
for const in ("CLAUDE_PROVIDER", "GEMINI_PROVIDER", "DEEPSEEK_PROVIDER",
              "QWEN_PROVIDER", "GLM_PROVIDER", "MISTRAL_PROVIDER"):
    p = getattr(bot, const, None)
    if p is None:
        continue
    check(f"{const} has both tiers None",
          p.think_reasoning_effort is None and p.plain_reasoning_effort is None)

print("\nthe None-override trap")
check("reasoning tiers are in _BACKEND_NULLABLE_FIELDS",
      "think_reasoning_effort" in bot._BACKEND_NULLABLE_FIELDS
      and "plain_reasoning_effort" in bot._BACKEND_NULLABLE_FIELDS)
check("both tiers are overridable at all (in _BACKEND_FIELD_MAP)",
      "think_reasoning_effort" in bot._BACKEND_FIELD_MAP
      and "plain_reasoning_effort" in bot._BACKEND_FIELD_MAP)
check("the switched-to backend's other overrides still applied",
      other_route.base_url != bot.KIMI_PROVIDER.base_url
      and other_route.model_id is not None)

q = resolved("nonexistent-backend")
check("an unknown backend leaves the default tiers intact",
      q.think_reasoning_effort == bot.KIMI_PROVIDER.think_reasoning_effort
      and q.plain_reasoning_effort == bot.KIMI_PROVIDER.plain_reasoning_effort)

# Guard the skip-None rule it relies on: a partial dict must not blank a default.
r = copy.deepcopy(bot.KIMI_PROVIDER)
r.backends["probe"] = {"input_cost_per_million": None, "model": "m"}
bot.ProviderRegistry._apply_backend(None, r, "probe", {})
check("a None on a NON-nullable field is still skipped",
      r.input_cost_per_million == bot.KIMI_PROVIDER.input_cost_per_million
      and r.model_id == "m")

print("\nshim sends the right tier")


def captured_kwargs(provider, thinking):
    seen = {}

    def create(**kw):
        seen.update(kw)
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    B = BotClass.__new__(BotClass)
    B.tavily_client = None
    p = copy.deepcopy(provider)
    p.peak_windows_utc = ()
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    asyncio.run(B._generate_openai_compatible_response(
        client, p, 0, [{"role": "user", "content": "hi"}], "sys", thinking=thinking))
    return seen


kw = captured_kwargs(moonshot_route, thinking=False)
check(f"plain Kimi turn on Moonshot sends reasoning_effort={PLAIN_TIER!r}",
      kw.get("reasoning_effort") == PLAIN_TIER)
check("and that is NOT the thinking tier (the two tiers are distinct)",
      kw.get("reasoning_effort") != moonshot_route.think_reasoning_effort)
check("no extra_body {'thinking': ...} on it (K3 rejects that)",
      "thinking" not in (kw.get("extra_body") or {}))
kw = captured_kwargs(moonshot_route, thinking=True)
check("!think Kimi turn on Moonshot sends reasoning_effort='max'",
      kw.get("reasoning_effort") == "max")

kw = captured_kwargs(fireworks_route, thinking=False)
check("plain Kimi turn on Fireworks sends NO reasoning_effort",
      "reasoning_effort" not in kw)

kw = captured_kwargs(bot.DEEPSEEK_PROVIDER, thinking=False)
check("DeepSeek plain turn sends NO reasoning_effort",
      "reasoning_effort" not in kw)
kw = captured_kwargs(bot.DEEPSEEK_PROVIDER, thinking=True)
check("DeepSeek thinking turn sends NO reasoning_effort either",
      "reasoning_effort" not in kw)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
