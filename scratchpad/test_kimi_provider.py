# Offline validation for the Kimi K3 head, lab-bot flavor (ported 2026-08-02).
# Run from repo root: PYTHONIOENCODING=utf-8 python scratchpad/test_kimi_provider.py
#
# Adapted from upstream Opus-Deipseek's test_kimi_provider.py with the lab-bot
# delta: here Kimi's DEFAULT backend is Fireworks serverless (US/ZDR, shares
# FIREWORKS_API_KEY with Qwen/GLM); api.moonshot.ai exists only as the
# non-default "moonshot" backend and must never gate the lab bot's default.
# Scenarios B/C run in fresh-interpreter subprocesses because from_config
# resolves the module-level constants in place (one call per process, like prod).

import inspect
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Scenario B: fake FIREWORKS key => kimi enabled on Fireworks --------------
if len(sys.argv) > 1 and sys.argv[1] == "--scenario-b":
    os.environ["FIREWORKS_API_KEY"] = "fw-fake-for-offline-test"
    from isaic import core as bot
    reg = bot.ProviderRegistry.from_config({})
    kimi = reg.by_id("kimi")
    assert kimi is not None and kimi.enabled is True, "kimi not enabled with Fireworks key present"
    client = reg.clients.get("kimi")
    assert client is not None, "kimi client not built"
    assert "api.fireworks.ai" in str(client.base_url), f"bad base_url: {client.base_url}"
    assert reg.openai_compatible_clients.get("Kimi") is client, "no name-keyed Kimi client"
    print("scenario-b OK")
    sys.exit(0)

# --- Scenario C: config flips backend to moonshot (parity toggle) -------------
# Not for lab use, but the toggle must round-trip through _apply_backend so the
# inverted backends table is known-good.
if len(sys.argv) > 1 and sys.argv[1] == "--scenario-c":
    os.environ.pop("FIREWORKS_API_KEY", None)
    os.environ["MOONSHOT_API_KEY"] = "sk-fake-for-offline-test"
    from isaic import core as bot
    reg = bot.ProviderRegistry.from_config({"providers": {"kimi": {"backend": "moonshot"}}})
    kimi = reg.by_id("kimi")
    assert kimi is not None, "kimi missing"
    assert kimi.backend == "moonshot", f"backend not applied: {kimi.backend}"
    assert kimi.base_url == "https://api.moonshot.ai/v1", f"bad base_url: {kimi.base_url}"
    assert kimi.model_id == "kimi-k3", f"bad model: {kimi.model_id}"
    assert kimi.api_key_env == "MOONSHOT_API_KEY", f"bad key env: {kimi.api_key_env}"
    assert kimi.enabled is True, "kimi should gate on MOONSHOT_API_KEY under the moonshot backend"
    print("scenario-c OK")
    sys.exit(0)

# Scenario A must see no Fireworks key even if the operator's .env gains one later.
os.environ.pop("FIREWORKS_API_KEY", None)

from isaic import core as bot

PASS = 0
FAIL = 0

def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


print("== constant shape (lab flavor: Fireworks default) ==")
K = bot.KIMI_PROVIDER
check("name/id are Kimi/kimi", K.name == "Kimi" and K.id == "kimi")
check("openai_compatible on api.fireworks.ai (US/ZDR default)",
      K.sdk_type == "openai_compatible"
      and K.base_url == "https://api.fireworks.ai/inference/v1")
check("key env is FIREWORKS_API_KEY (shared with Qwen/GLM)",
      K.api_key_env == "FIREWORKS_API_KEY")
check("model slug is the Fireworks serverless one",
      K.model_id == "accounts/fireworks/models/kimi-k3")
check("pricing 3/15, cached 0.30 (Moonshot-identical)",
      K.input_cost_per_million == 3.0 and K.output_cost_per_million == 15.0
      and K.cached_input_cost_per_million == 0.30)
check("1M context", K.max_context_tokens == 1_000_000)
check("vision off, tavily search", K.supports_vision is False and K.search_backend == "tavily")
check("think quirk: reasoning_effort=max, no thinking-param quirks",
      K.think_reasoning_effort == "max"
      and K.requires_reasoning_echo is False
      and K.disables_thinking_by_default is False)
check("Fireworks grid (US fleet), not the CN default", K.grid_gco2_per_kwh == 400.0)
check("moonshot backend present as the NON-default toggle",
      K.backend == "fireworks" and "moonshot" in K.backends
      and K.backends["moonshot"]["model"] == "kimi-k3"
      and K.backends["moonshot"]["api_key_env"] == "MOONSHOT_API_KEY")

print("== canonical order + labels ==")
ids = [p.id for p in bot._PROVIDER_CONSTANTS]
check("kimi sits between glm and sim",
      ids.index("glm") < ids.index("kimi") < ids.index("sim"), str(ids))
check("MODEL_LABEL_NAMES includes Kimi", "|Kimi|" in f"|{bot.MODEL_LABEL_NAMES}|")
strip_re = re.compile(rf"^\[({bot.MODEL_LABEL_NAMES})\]\s*")
check("label regex strips a [Kimi] echo", strip_re.sub("", "[Kimi] hello") == "hello")

print("== cost math ==")
K.record_usage(1_000_000, 1_000_000, cached_input_tokens=1_000_000)
check("1M in + 1M out + 1M cached = $18.30", abs(K.get_cost() - 18.30) < 1e-9,
      f"got {K.get_cost()}")
K.total_input_tokens = K.total_output_tokens = K.total_cached_input_tokens = 0

print("== themes ==")
for tname, alias in (("eva", "!kaworu"), ("isaic", "!issachar"), ("nightvale", "!glowcloud")):
    fl = bot.THEMES[tname].flavors.get("kimi")
    check(f"{tname} has a kimi flavor with {alias}",
          fl is not None and alias in fl.aliases,
          str(fl))
check("nightvale also answers !allhail",
      "!allhail" in bot.THEMES["nightvale"].flavors["kimi"].aliases)
check("isaic (default skin) names it Issachar",
      bot.THEMES["isaic"].flavors["kimi"].display_name == "Issachar")
check("eva keeps canonical display name (skin invariant)",
      bot.THEMES["eva"].flavors["kimi"].display_name == "Kimi")

print("== IsaicBot wiring (static) ==")
prefixes = getattr(bot.IsaicBot, "CANONICAL_PREFIXES", {})
check("CANONICAL_PREFIXES kimi -> !kimi/!k3", prefixes.get("kimi") == ("!kimi", "!k3"),
      str(prefixes.get("kimi")))
check("HELP_ROLES has kimi", "kimi" in getattr(bot.IsaicBot, "HELP_ROLES", {}))
src = inspect.getsource(bot.IsaicBot)
check("guard tuple includes kimi",
      '"glm", "kimi")' in src or '"glm", "kimi", "sim")' in src)
check("panel_members_all includes Kimi", '"GLM", "Kimi",' in src)
check("_estimate_confidence has an override-only Kimi branch",
      'provider.name == "Kimi"' in inspect.getsource(bot.IsaicBot._estimate_confidence))
shim_src = inspect.getsource(bot.IsaicBot._generate_openai_compatible_response)
check("shim sends reasoning_effort when thinking + quirk set",
      'api_kwargs["reasoning_effort"] = provider.think_reasoning_effort' in shim_src)
check("identity prompt has a Kimi branch (Fireworks-served)",
      'made by Moonshot AI (served via Fireworks)' in src)

print("== registry, scenario A: no FIREWORKS_API_KEY ==")
regA = bot.ProviderRegistry.from_config({})
kimiA = regA.by_id("kimi")
check("kimi present but disabled", kimiA is not None and kimiA.enabled is False)
check("kimi client is None when disabled", regA.clients.get("kimi") is None)
others_ok = all(
    p.enabled == bool(os.getenv(p.api_key_env))
    for p in regA.providers
    if p.id not in ("kimi", "sim") and p.api_key_env
)
check("other heads' enabled state still tracks their own keys", others_ok,
      str([(p.id, p.enabled, p.api_key_env, bool(os.getenv(p.api_key_env or ''))) for p in regA.providers]))

print("== registry, scenario B: fake FIREWORKS_API_KEY (fresh interpreter) ==")
result = subprocess.run(
    [sys.executable, os.path.abspath(__file__), "--scenario-b"],
    capture_output=True, text=True,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)
check("kimi enabled + client wired to api.fireworks.ai with key present",
      result.returncode == 0 and "scenario-b OK" in result.stdout,
      (result.stdout + result.stderr)[-500:])

print("== registry, scenario C: backend=moonshot toggle (fresh interpreter) ==")
result = subprocess.run(
    [sys.executable, os.path.abspath(__file__), "--scenario-c"],
    capture_output=True, text=True,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)
check("moonshot backend toggle applies base_url/model/key-gating",
      result.returncode == 0 and "scenario-c OK" in result.stdout,
      (result.stdout + result.stderr)[-500:])

print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
