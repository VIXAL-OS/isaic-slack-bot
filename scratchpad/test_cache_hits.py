"""Offline checks for _usage_cache_hits — prompt-cache hits read from BOTH the
DeepSeek field (`prompt_cache_hit_tokens`) and the OpenAI-standard one
(`prompt_tokens_details.cached_tokens`, Fireworks / Alibaba) — and for the
shim + simulator call sites that feed them into record_usage (2026-09-21).
No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_cache_hits.py

Imports `bot` by default; set BOT_MODULE=isaic.core to run against the Slack
bot's copy (from the isaic-slack-bot repo root).
"""
import asyncio
import copy
import importlib
import inspect
import os
import sys
import types

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))
from openai.types import Completion, CompletionUsage  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402

BotClass = getattr(bot, "ClaudeBot", None) or getattr(bot, "IsaicBot")
hits = bot._usage_cache_hits
ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def sdk_usage(**fields):
    """A real openai-SDK CompletionUsage, as the shim sees it (unknown
    provider fields like DeepSeek's ride along as pydantic extras)."""
    base = {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010}
    return CompletionUsage.model_validate({**base, **fields})


print("helper — shapes")
check("None usage → 0", hits(None) == 0)
check("no cache fields → 0", hits(sdk_usage()) == 0)
check("Fireworks shape (prompt_tokens_details.cached_tokens=800) → 800",
      hits(sdk_usage(prompt_tokens_details={"cached_tokens": 800})) == 800)
check("DeepSeek shape (prompt_cache_hit_tokens=128) → 128",
      hits(sdk_usage(prompt_cache_hit_tokens=128, prompt_cache_miss_tokens=872)) == 128)
check("DeepSeek reporting BOTH fields (128/128) → 128, not 256",
      hits(sdk_usage(prompt_cache_hit_tokens=128,
                     prompt_tokens_details={"cached_tokens": 128})) == 128)
check("DeepSeek field wins when both non-zero and differ",
      hits(sdk_usage(prompt_cache_hit_tokens=128,
                     prompt_tokens_details={"cached_tokens": 64})) == 128)
check("DeepSeek field 0 → falls through to the standard field",
      hits(sdk_usage(prompt_cache_hit_tokens=0,
                     prompt_tokens_details={"cached_tokens": 40})) == 40)
check("cached_tokens None (Fireworks sends nulls) → 0",
      hits(sdk_usage(prompt_tokens_details={"cached_tokens": None})) == 0)
check("prompt_tokens_details None → 0", hits(sdk_usage(prompt_tokens_details=None)) == 0)
check("dict shape, standard field → 300",
      hits({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 300}}) == 300)
check("dict shape, DeepSeek both fields → 50",
      hits({"prompt_cache_hit_tokens": 50, "prompt_tokens_details": {"cached_tokens": 50}}) == 50)
check("dict shape, no cache → 0", hits({"prompt_tokens": 10}) == 0)
check("attribute objects (SimpleNamespace) nested → 7",
      hits(types.SimpleNamespace(prompt_tokens_details=types.SimpleNamespace(cached_tokens=7))) == 7)
check("garbage value → 0", hits({"prompt_tokens_details": {"cached_tokens": "abc"}}) == 0)
check("negative value → 0", hits({"prompt_cache_hit_tokens": -5}) == 0)

print("every call site migrated")
src = inspect.getsource(bot)
check("no call site still reads prompt_cache_hit_tokens directly",
      'getattr(response.usage, "prompt_cache_hit_tokens"' not in src
      and 'getattr(usage, "prompt_cache_hit_tokens"' not in src)
check("≥4 call sites use _usage_cache_hits", src.count("_usage_cache_hits(") >= 5)  # def + 4 sites


def chat_response(usage: dict) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "pong"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010, **usage},
    })


def fake_client(resp, kind="chat"):
    create = types.SimpleNamespace(create=lambda **kw: resp)
    if kind == "chat":
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=create))
    return types.SimpleNamespace(completions=create)


def run_shim(provider, usage: dict):
    B = BotClass.__new__(BotClass)
    B.tavily_client = None  # no tools → one call, no tool loop
    p = copy.deepcopy(provider)
    p.peak_windows_utc = ()  # keep the cost math clock-independent
    text = asyncio.run(B._generate_openai_compatible_response(
        fake_client(chat_response(usage)), p, 0, [{"role": "user", "content": "hi"}], "sys",
    ))[0]
    return text, p


print("shim end-to-end (fake client)")
text, g = run_shim(bot.GLM_PROVIDER, {"prompt_tokens_details": {"cached_tokens": 800}})
check("GLM (Fireworks shape): 200 uncached + 800 cached recorded",
      text == "pong" and g.total_input_tokens == 200 and g.total_cached_input_tokens == 800
      and g.total_output_tokens == 10 and g.total_requests == 1)
exp = (200 * g.input_cost_per_million + 800 * g.cached_input_cost_per_million
       + 10 * g.output_cost_per_million) / 1e6
check(f"GLM cached tokens bill at the cache rate (${exp:.6f})", abs(g.get_cost() - exp) < 1e-12)
_, d = run_shim(bot.DEEPSEEK_PROVIDER, {"prompt_cache_hit_tokens": 128, "prompt_cache_miss_tokens": 872,
                                        "prompt_tokens_details": {"cached_tokens": 128}})
check("DeepSeek (both fields): 872 uncached + 128 cached, no double count",
      d.total_input_tokens == 872 and d.total_cached_input_tokens == 128)
_, n = run_shim(bot.GLM_PROVIDER, {})
check("no-cache usage: all 1000 input uncached",
      n.total_input_tokens == 1000 and n.total_cached_input_tokens == 0)

print("simulator end-to-end (fake /completions client)")
B = BotClass.__new__(BotClass)
sim = copy.deepcopy(bot.SIM_PROVIDER)
sim.sim_search = False
comp = Completion.model_validate({
    "id": "x", "object": "text_completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "text": " hello there", "logprobs": None}],
    "usage": {"prompt_tokens": 500, "completion_tokens": 5, "total_tokens": 505,
              "prompt_tokens_details": {"cached_tokens": 300}},
})
B.clients = {sim.id: fake_client(comp, kind="completions")}
asyncio.run(B._generate_simulator_response(
    sim, 0, [{"role": "user", "content": "Sarah: hi"}], "a channel log"))
check("simulator: 200 uncached + 300 cached recorded",
      sim.total_input_tokens == 200 and sim.total_cached_input_tokens == 300
      and sim.total_output_tokens == 5)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
