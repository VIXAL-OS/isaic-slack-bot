"""Offline checks for per-REQUEST server-side tool billing (2026-11-07).

DeepSeek's Search & Fetch APIs bill $0.75 per 1,000 successful requests each,
metered separately, live since 2026-10-31 16:00 UTC. We run search on Tavily
and never call them — this bucket exists so that enabling DeepSeek's native
search cannot silently under-report in !cost, the same shape as the 2026-09
cache-hit gap.

The load-bearing invariants: request-priced spend must NOT enter the token
buckets (which would corrupt $/token) or the energy/CO2 estimate (per token),
must survive a save/load round trip, and must not be multiplied by the
peak-hour surcharge.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_tool_request_cost.py

Imports `bot` by default; set BOT_MODULE=isaic.core to run against the Slack
bot's copy (from the isaic-slack-bot repo root).
"""
import copy
import importlib
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def ds():
    p = copy.deepcopy(bot.DEEPSEEK_PROVIDER)
    p.peak_windows_utc = ()  # keep the clock out of it unless a test wants it
    return p


print("rates are configured where they belong")
check("DeepSeek search rate is $0.75/1k", ds().search_cost_per_1k_requests == 0.75)
check("DeepSeek fetch rate is $0.75/1k", ds().fetch_cost_per_1k_requests == 0.75)
for const in ("CLAUDE_PROVIDER", "GEMINI_PROVIDER", "QWEN_PROVIDER",
              "GLM_PROVIDER", "MISTRAL_PROVIDER", "KIMI_PROVIDER"):
    p = getattr(bot, const, None)
    if p is None:
        continue
    check(f"{const} has no per-request tool billing (0.0 default)",
          p.search_cost_per_1k_requests == 0.0 and p.fetch_cost_per_1k_requests == 0.0)

print("\nmetering")
p = ds()
check("fresh provider: no tool spend", p.get_tool_request_cost() == 0.0)
p.record_tool_requests(search=1000)
check("1000 search = $0.75", abs(p.get_tool_request_cost() - 0.75) < 1e-12)
p = ds()
p.record_tool_requests(fetch=1000)
check("1000 fetch = $0.75", abs(p.get_tool_request_cost() - 0.75) < 1e-12)
p = ds()
p.record_tool_requests(search=500, fetch=500)
check("500 + 500 = $0.75 (billed separately, same rate)",
      abs(p.get_tool_request_cost() - 0.75) < 1e-12
      and p.total_search_requests == 500 and p.total_fetch_requests == 500)
p = ds()
p.record_tool_requests(search=1)
p.record_tool_requests(search=1)
check("accumulates across calls", p.total_search_requests == 2
      and abs(p.get_tool_request_cost() - 0.0015) < 1e-12)
p = ds()
p.record_tool_requests()
check("no-op call changes nothing",
      p.total_search_requests == 0 and p.get_tool_request_cost() == 0.0)

print("\nrequest spend stays OUT of the token and energy buckets")
p = ds()
p.record_tool_requests(search=5000, fetch=5000)
check("token buckets untouched",
      p.total_input_tokens == 0 and p.total_output_tokens == 0
      and p.total_cached_input_tokens == 0)
check("request count untouched (that counts model calls)", p.total_requests == 0)
check("energy untouched — request pricing is not per token",
      p.get_energy_wh() == 0.0)
check("CO2 untouched", p.get_co2_g(400.0) == 0.0)
check("but it DOES show up in get_cost()", abs(p.get_cost() - 7.5) < 1e-12)

print("\ncomposes with token cost and the peak surcharge")
p = ds()
p.record_usage(input_tokens=1_000_000, output_tokens=0)
token_only = p.get_cost()
p.record_tool_requests(search=1000)
check("get_cost = tokens + tool spend",
      abs(p.get_cost() - (token_only + 0.75)) < 1e-12)

p = copy.deepcopy(bot.DEEPSEEK_PROVIDER)  # keep real peak windows
peak_moment = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)  # Mon 02:00 UTC
check("sanity: that moment really is peak", p.is_peak(peak_moment) is True)
p.record_tool_requests(search=1000)
check("tool spend is NOT multiplied by the peak surcharge",
      p.total_peak_surcharge == 0.0 and abs(p.get_cost() - 0.75) < 1e-12)

print("\nlocal/self-hosted backends")
p = ds()
p.cost_mode = "local"
p.record_usage(input_tokens=1_000_000, output_tokens=1_000_000)
check("local: tokens are free", p.get_cost() == 0.0)
p.record_tool_requests(search=1000)
check("local: a HOSTED per-request tool still bills",
      abs(p.get_cost() - 0.75) < 1e-12)

print("\npersistence round trip")
p = ds()
p.record_tool_requests(search=1234, fetch=56)
blob = p.to_stats_dict()
check("to_stats_dict carries all three fields",
      blob["search_requests"] == 1234 and blob["fetch_requests"] == 56
      and abs(blob["tool_request_cost"] - (1234 + 56) / 1000.0 * 0.75) < 1e-12)
q = ds()
q.load_stats(blob)
check("load_stats restores counters and dollars",
      q.total_search_requests == 1234 and q.total_fetch_requests == 56
      and abs(q.get_tool_request_cost() - p.get_tool_request_cost()) < 1e-12)

q = ds()
q.load_stats({"input_tokens": 10, "output_tokens": 5, "requests": 1})
check("a pre-2026-11 save (no tool keys) loads as zero, not a crash",
      q.total_search_requests == 0 and q.total_fetch_requests == 0
      and q.get_tool_request_cost() == 0.0)

print("\ndollars are frozen at record time (no retroactive repricing)")
p = ds()
p.record_tool_requests(search=1000)
p.search_cost_per_1k_requests = 7.50  # provider hikes the price 10x
check("a later rate change does not reprice history",
      abs(p.get_tool_request_cost() - 0.75) < 1e-12)
p.record_tool_requests(search=1000)
check("new calls meter at the new rate",
      abs(p.get_tool_request_cost() - (0.75 + 7.50)) < 1e-12)

print("\n!cost surfaces it")
import inspect  # noqa: E402
BotClass = getattr(bot, "ClaudeBot", None) or getattr(bot, "IsaicBot")
mgr = getattr(bot, "ConversationManager", None)
src = inspect.getsource(mgr) if mgr else inspect.getsource(BotClass)
check("get_cost_summary mentions tool calls", "tool calls" in src)
check("get_cost_summary reads the per-request cost",
      "get_tool_request_cost()" in src)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
