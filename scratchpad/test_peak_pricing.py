"""Offline checks for DeepSeek peak-hour pricing + the V4.1 Flash / Fireworks
slug refresh (2026-09-21). No network, no keys. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_peak_pricing.py

Imports `bot` by default; set BOT_MODULE=isaic.core to run against the Slack
bot's copy (from the isaic-slack-bot repo root).
"""
import copy
import importlib
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.getcwd())
bot = importlib.import_module(os.getenv("BOT_MODULE", "bot"))

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


D = bot.DEEPSEEK_PROVIDER

print("constants")
check("deepseek default model is deepseek-flash", D.model_id == "deepseek-flash")
check("deepseek off-peak rates 0.15/0.60/0.003",
      (D.input_cost_per_million, D.output_cost_per_million, D.cached_input_cost_per_million)
      == (0.15, 0.60, 0.003))
check("deepseek peak multiplier 2x", D.peak_multiplier == 2.0)
Q = bot.QWEN_PROVIDER
if Q.backend == "alibaba":  # Discord bot: Qwen 3.8 Flash on Alibaba US by default
    check("qwen default = qwen3.8-flash on dashscope-us",
          Q.model_id == "qwen3.8-flash" and "dashscope-us" in Q.base_url
          and Q.api_key_env == "DASHSCOPE_API_KEY")
    check("qwen fireworks backend = qwen3p8-max",
          Q.backends["fireworks"]["model"].endswith("/qwen3p8-max"))
else:  # isaic-slack-bot: Fireworks 3.8 Max is the default
    check("qwen slug qwen3p8-max", Q.model_id.endswith("/qwen3p8-max"))
check("glm slug glm-5p3", bot.GLM_PROVIDER.model_id.endswith("/glm-5p3"))

if Q.backend == "alibaba":
    print("qwen registry gating")
    # from_config mutates the module's provider constants in place, so build
    # each registry over fresh copies and restore the originals afterwards.
    orig_constants = bot._PROVIDER_CONSTANTS

    def fresh(cfg):
        bot._PROVIDER_CONSTANTS = [copy.deepcopy(p) for p in orig_constants]
        reg = bot.ProviderRegistry.from_config(cfg)
        return {p.id: p for p in reg.providers}

    saved = {k: os.environ.pop(k, None) for k in ("DASHSCOPE_API_KEY", "FIREWORKS_API_KEY")}
    try:
        os.environ["FIREWORKS_API_KEY"] = "fw-test"
        ps = fresh({})
        check("no DASHSCOPE key → Qwen off, GLM still on", not ps["qwen"].enabled and ps["glm"].enabled)
        os.environ["DASHSCOPE_API_KEY"] = "ds-test"
        q = fresh({})["qwen"]
        check("DASHSCOPE key → Qwen on, 1M ctx, Alibaba price",
              q.enabled and q.max_context_tokens == 1_000_000 and q.input_cost_per_million == 0.113)
        del os.environ["DASHSCOPE_API_KEY"]
        q = fresh({"providers": {"qwen": {"backend": "fireworks"}}})["qwen"]
        check("backend=fireworks → Max slug, Fireworks key, 256k ctx, Max energy",
              q.enabled and q.model_id.endswith("/qwen3p8-max")
              and q.api_key_env == "FIREWORKS_API_KEY" and q.max_context_tokens == 256_000
              and q.est_wh_per_1k_tokens == 0.35 and q.input_cost_per_million == 2.00)
    finally:
        bot._PROVIDER_CONSTANTS = orig_constants
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

print("peak windows (2026-09-21 is a Monday)")
check("Mon 02:30 UTC peak", D.is_peak(utc(2026, 9, 21, 2, 30)))
check("Mon 04:00 UTC off-peak (end exclusive)", not D.is_peak(utc(2026, 9, 21, 4, 0)))
check("Mon 05:59 UTC off-peak", not D.is_peak(utc(2026, 9, 21, 5, 59)))
check("Mon 06:00 UTC peak", D.is_peak(utc(2026, 9, 21, 6, 0)))
check("Mon 09:59 UTC peak", D.is_peak(utc(2026, 9, 21, 9, 59)))
check("Mon 10:00 UTC off-peak", not D.is_peak(utc(2026, 9, 21, 10, 0)))
check("Fri 07:00 UTC peak", D.is_peak(utc(2026, 9, 25, 7, 0)))
check("Sat 07:00 UTC off-peak", not D.is_peak(utc(2026, 9, 26, 7, 0)))
check("Sun 02:00 UTC off-peak", not D.is_peak(utc(2026, 9, 27, 2, 0)))
check("naive datetime treated as UTC", D.is_peak(datetime(2026, 9, 21, 1, 30)))
edt = timezone(timedelta(hours=-4))
check("Sun 21:30 EDT (= Mon 01:30 UTC) peak",
      D.is_peak(datetime(2026, 9, 20, 21, 30, tzinfo=edt)))
check("is_peak() with no arg doesn't raise", D.is_peak() in (True, False))

print("cost math")
base = 0.15 + 0.60 + 0.003
p = copy.deepcopy(D)
p.record_usage(1_000_000, 1_000_000, 1_000_000, when=utc(2026, 9, 26, 12, 0))  # Sat
check(f"off-peak request = ${base:.3f}", abs(p.get_cost() - base) < 1e-9 and p.total_peak_surcharge == 0)
p.record_usage(1_000_000, 1_000_000, 1_000_000, when=utc(2026, 9, 21, 7, 0))   # Mon peak
check(f"peak request billed 2x (total ${base * 3:.3f})", abs(p.get_cost() - base * 3) < 1e-9)
check("surcharge == one base request", abs(p.total_peak_surcharge - base) < 1e-9)
check("token buckets unaffected by surcharge",
      p.total_input_tokens == 2_000_000 and p.total_output_tokens == 2_000_000
      and p.total_cached_input_tokens == 2_000_000)
q = copy.deepcopy(D)
q.load_stats(p.to_stats_dict())
check("to_stats_dict/load_stats round-trips surcharge", abs(q.get_cost() - p.get_cost()) < 1e-9)
r = copy.deepcopy(D)
r.load_stats({"input_tokens": 5})
check("pre-surcharge stats dict loads as 0", r.total_peak_surcharge == 0.0)

print("flat providers + backends")
c = copy.deepcopy(bot.CLAUDE_PROVIDER)
check("Claude never peak", not c.is_peak(utc(2026, 9, 21, 7, 0)))
c.record_usage(1000, 1000, 0, when=utc(2026, 9, 21, 7, 0))
check("Claude cost formula unchanged",
      abs(c.get_cost() - (1000 * c.input_cost_per_million + 1000 * c.output_cost_per_million) / 1e6) < 1e-12)
g = copy.deepcopy(bot.GEMINI_PROVIDER)
g.record_usage(300_000, 1000, 0)
exp = (300_000 * (g.input_cost_per_million_above_tier or g.input_cost_per_million)
       + 1000 * (g.output_cost_per_million_above_tier or g.output_cost_per_million)) / 1e6
check("Gemini above-tier cost unchanged", abs(g.get_cost() - exp) < 1e-12)
reg = bot.ProviderRegistry.__new__(bot.ProviderRegistry)
fw = copy.deepcopy(D)
reg._apply_backend(fw, "fireworks", {})
check("fireworks backend is flat-priced", fw.peak_windows_utc == () and not fw.is_peak(utc(2026, 9, 21, 7, 0)))
check("fireworks backend slug deepseek-v4p1-flash",
      fw.model_id == "accounts/fireworks/models/deepseek-v4p1-flash")
sh = copy.deepcopy(D)
reg._apply_backend(sh, "self_hosted", {})
sh.record_usage(1000, 1000, 0, when=utc(2026, 9, 21, 7, 0))
check("self_hosted: local $0, no surcharge", sh.get_cost() == 0.0 and sh.total_peak_surcharge == 0.0)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
