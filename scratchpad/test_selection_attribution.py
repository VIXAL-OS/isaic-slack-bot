"""Offline checks for routing attribution — who chose the model (2026-11-07).

`!cost`'s rankings are volume x intensity and say nothing about whether anyone
MEANT to send that volume. A head the router keeps picking is a routing
decision to revisit; the same tokens chosen by hand are just what you wanted.
These checks pin the three buckets, the ContextVar mechanics that make them
exact without threading a parameter through eleven call sites, and the honest
treatment of saves that predate the counter.

No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_selection_attribution.py

Imports `bot` by default; set BOT_MODULE=isaic.core for the Slack bot's copy
(which does not have this feature yet — it will skip rather than fail).
"""
import asyncio
import copy
import importlib
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


if not hasattr(bot, "_SELECTION_MODE"):
    print("this build has no selection attribution — nothing to check")
    sys.exit(0)


def fresh(const=None):
    p = copy.deepcopy(const or bot.CLAUDE_PROVIDER)
    p.total_input_tokens = p.total_output_tokens = 0
    p.total_cached_input_tokens = 0
    p.total_requests = 0
    p.total_token_cost = None
    p.peak_windows_utc = ()
    p.total_tokens_by_selection = {}
    return p


def mgr():
    m = bot.ConversationManager.__new__(bot.ConversationManager)
    m.reading_materials = {}
    return m


print("the three buckets")
check("the modes are exactly auto / override / system",
      bot.SELECTION_MODES == ("auto", "override", "system"))
check("and each has a short display label",
      set(bot.SELECTION_LABELS) == set(bot.SELECTION_MODES))

print("\nrecord_usage attributes to whoever chose the provider")
p = fresh()
tok = bot._SELECTION_MODE.set("override")
p.record_usage(100, 10, cached_input_tokens=5)
bot._SELECTION_MODE.reset(tok)
check("an override turn lands in the override bucket",
      p.total_tokens_by_selection == {"override": 115})
check("the figure is uncached + cached + output, not just input",
      p.total_tokens_by_selection["override"] == 100 + 5 + 10)

p2 = fresh()
p2.record_usage(50, 5)
check("with nothing set it defaults to machinery, not to the user",
      p2.total_tokens_by_selection == {"system": 55})

p3 = fresh()
t3 = bot._SELECTION_MODE.set("not-a-real-mode")
p3.record_usage(10, 1)
bot._SELECTION_MODE.reset(t3)
check("an unrecognised value is treated as machinery, never trusted",
      p3.total_tokens_by_selection == {"system": 11})

print("\none logical request can bill several times (tool loop, retry)")
p4 = fresh()
t4 = bot._SELECTION_MODE.set("auto")
p4.record_usage(100, 10)       # first call
p4.record_usage(80, 20)        # tool-result round
p4.record_usage(30, 5)         # degraded retry
bot._SELECTION_MODE.reset(t4)
check("every call in the turn accumulates into the same bucket",
      p4.total_tokens_by_selection == {"auto": 245})
check("and the token totals still agree with the bucket",
      p4.total_input_tokens + p4.total_output_tokens == 245)

print("\nthe counter never invents history")
p5 = fresh()
p5.total_input_tokens = 5_000_000      # pre-existing, unattributed
p5.total_output_tokens = 100_000
t5 = bot._SELECTION_MODE.set("auto")
p5.record_usage(1_000, 100)
bot._SELECTION_MODE.reset(t5)
check("old tokens stay OUT of the buckets (sum < token total)",
      sum(p5.total_tokens_by_selection.values()) == 1_100
      and p5.total_input_tokens + p5.total_output_tokens > 5_000_000)

print("\nContextVar mechanics — why this needs no threaded parameter")


async def _concurrency():
    a, b = fresh(), fresh()

    async def turn(provider, mode, delay):
        # Each Discord event runs in its own Task, so each gets its own copy
        # of the context: interleaving must not cross-contaminate.
        bot._SELECTION_MODE.set(mode)
        await asyncio.sleep(delay)
        provider.record_usage(100, 10)
        await asyncio.sleep(delay)
        provider.record_usage(50, 5)
        return bot._SELECTION_MODE.get()

    seen = await asyncio.gather(
        turn(a, "auto", 0.01),
        turn(b, "override", 0.005),
    )
    return a, b, seen


pa, pb, modes = asyncio.run(_concurrency())
check("two interleaved turns keep their own modes",
      modes == ["auto", "override"])
check("and their tokens do not cross buckets",
      pa.total_tokens_by_selection == {"auto": 165}
      and pb.total_tokens_by_selection == {"override": 165})


async def _no_leak_outward():
    bot._SELECTION_MODE.set("override")

    async def child():
        bot._SELECTION_MODE.set("system")
        return bot._SELECTION_MODE.get()

    inner = await asyncio.create_task(child())
    return inner, bot._SELECTION_MODE.get()


inner, outer = asyncio.run(_no_leak_outward())
check("a child task's change does not leak back to its parent",
      inner == "system" and outer == "override")

print("\n_panel_complete bills machinery, then restores the turn's own mode")


async def _panel():
    b = BotClass.__new__(BotClass)
    seen = []

    async def spy(*a, **k):
        seen.append(bot._SELECTION_MODE.get())
        return "answer"

    b._panel_complete_dispatch = spy
    # A !research inside a chat turn: the turn already set its own mode.
    bot._SELECTION_MODE.set("override")
    out = await BotClass._panel_complete(b, None, 1, [], "sys")
    return seen, out, bot._SELECTION_MODE.get()


seen, answer, after = asyncio.run(_panel())
check("the panel call itself bills as machinery", seen == ["system"])
check("it still returns the model's text", answer == "answer")
check("and the chat turn's own mode survives it (reset, not overwritten)",
      after == "override")

print("\npersistence")
p6 = fresh()
p6.total_tokens_by_selection = {"auto": 1234, "override": 56}
round_trip = fresh()
round_trip.load_stats(p6.to_stats_dict())
check("the split survives a save/load round trip",
      round_trip.total_tokens_by_selection == {"auto": 1234, "override": 56})
legacy = fresh()
legacy.load_stats({"input_tokens": 999, "output_tokens": 9, "requests": 3})
check("a pre-2026-11-07 save loads as empty, not as all-auto",
      legacy.total_tokens_by_selection == {})
shared = fresh()
shared.load_stats(p6.to_stats_dict())
shared.total_tokens_by_selection["auto"] = 0
check("load_stats copies rather than aliasing the saved dict",
      p6.total_tokens_by_selection["auto"] == 1234)

print("\n_select_model reports who decided")


class _Chan:
    def __init__(self, cid=7):
        self.id = cid
        self.parent_id = None


class _Msg:
    def __init__(self, content="hello", cid=7):
        self.content = content
        self.attachments = []
        self.channel = _Chan(cid)


def _bot_for_selection(pref=None, default_model="claude"):
    b = BotClass.__new__(BotClass)
    b.providers = [copy.deepcopy(c) for c in
                   (bot.CLAUDE_PROVIDER, bot.DEEPSEEK_PROVIDER,
                    bot.QWEN_PROVIDER)]
    for prov in b.providers:
        prov.enabled = True
    b.claude_provider = b.providers[0]
    b.channel_preferences = {7: pref} if pref else {}
    bot.CONFIG.default_model = default_model
    return b


saved_default = bot.CONFIG.default_model
try:
    b = _bot_for_selection(default_model="")
    _, _, mode = asyncio.run(b._select_model(_Msg(), 1))
    check("the heuristic's own pick is 'auto'", mode == "auto")

    b = _bot_for_selection(pref="qwen", default_model="")
    prov, _, mode = asyncio.run(b._select_model(_Msg(), 1))
    check("a !prefer channel pin is 'override'",
          mode == "override" and prov.id == "qwen")

    b = _bot_for_selection(default_model="deepseek")
    prov, _, mode = asyncio.run(b._select_model(_Msg(), 1))
    check("an operator's config default_model is 'override' too",
          mode == "override" and prov.id == "deepseek")

    b = _bot_for_selection(default_model="")
    for prov in b.providers[1:]:
        prov.enabled = False
    _, _, mode = asyncio.run(b._select_model(_Msg(), 1))
    check("the only-head-up case is 'auto' (nobody chose it)", mode == "auto")
finally:
    bot.CONFIG.default_model = saved_default

print("\n!cost surfaces it honestly")
rows = [fresh(bot.CLAUDE_PROVIDER), fresh(bot.DEEPSEEK_PROVIDER)]
rows[0].total_requests = rows[1].total_requests = 10
rows[0].total_input_tokens = rows[1].total_input_tokens = 100_000
rows[0].total_output_tokens = rows[1].total_output_tokens = 5_000
plain = mgr().get_cost_summary(rows)
check("with no attributed tokens the line is absent, not a row of zeroes",
      "Chosen by" not in plain)

rows[0].total_tokens_by_selection = {"auto": 60_000, "override": 40_000}
withattr = mgr().get_cost_summary(rows)
check("once there is data the line appears", "Chosen by" in withattr)
line = [l for l in withattr.splitlines() if "Chosen by" in l][0]
check("shares are of the attributed subset (60/40 of 100k)",
      "auto 60%" in line and "yours 40%" in line)
check("a bucket with no tokens is omitted rather than shown as 0%",
      "machinery" not in line)
check("the unattributed remainder is named, not folded into auto",
      "unattributed" in line)
check("and rendered compactly so it cannot eat the char budget",
      "110k unattributed" in line)

rows[0].total_tokens_by_selection = {"auto": 105_000}
rows[1].total_tokens_by_selection = {"auto": 105_000}
fully = [l for l in mgr().get_cost_summary(rows).splitlines()
         if "Chosen by" in l][0]
check("fully attributed traffic drops the remainder clause",
      "unattributed" not in fully and "auto 100%" in fully)

print("\ncompact token rendering")
check("millions", bot._compact_tokens(61_500_000) == "61.5M")
check("thousands", bot._compact_tokens(110_000) == "110k")
check("and small counts stay exact", bot._compact_tokens(403) == "403")

print("\nthe message still fits Discord")
every = []
for c in (bot.CLAUDE_PROVIDER, bot.DEEPSEEK_PROVIDER, bot.GEMINI_PROVIDER,
          bot.MISTRAL_PROVIDER, bot.QWEN_PROVIDER, bot.GLM_PROVIDER,
          bot.KIMI_PROVIDER):
    q = fresh(c)
    q.total_requests = 50
    q.total_input_tokens, q.total_output_tokens = 500_000, 20_000
    q.total_cached_input_tokens = 100_000
    q.total_tokens_by_selection = {"auto": 400_000, "override": 150_000,
                                   "system": 70_000}
    every.append(q)
full = mgr().get_cost_summary(every)
check(f"7 heads with attribution render in {len(full)} chars (< 2000)",
      len(full) < 2000)
check("and the attribution line survives at that size",
      "Chosen by" in full)
# Per-provider detail left !cost for !costdetail on 2026-11-08, so what
# "survives at that size" now means is the pointer, not the section.
check("per-provider detail is no longer part of the summary",
      "**Detail**" not in full and "`!costdetail`" in full)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
