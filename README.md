# ISAIC — International System of AI Coopertition

> ⚠️ **UNTESTED — work in progress.** This bot has **not been run against a live Slack workspace yet**.
> It is syntax-validated and passes offline import/instantiation/integration checks, but the Slack
> adapter (Socket Mode events, file uploads, threading, reactions) has had **no live smoke test**.
> Expect to hit and fix rough edges once you wire real tokens. Not production-ready. See [Status](#status).

A **Slack-native, multi-model assistant**: Claude, Gemini, and Deepseek share one bot with smart
heuristic routing, shared two-tier memory, web search/grounding, prompt caching, and "bookclub" mode
for discussing long texts. Open-weight heads — **Mistral**, plus **Qwen**, **GLM**, and **Kimi K3**
on one Fireworks endpoint (US, zero-data-retention) — and a base-model **simulator** join when
configured.

The heads are skinned as the **twelve tribes** (the default `isaic` theme):

| Head | Model | Summon with |
|---|---|---|
| **Judah** | Claude (Opus 5.5) | `!judah` / `!claude` |
| **Joseph** | Gemini (3.1 Pro) | `!joseph` / `!gemini` |
| **Zebulun** | Deepseek (V4.1 Flash) | `!zebulun` / `!deepseek` |
| **Naphtali** | Mistral (Large 3) | `!naphtali` / `!mistral` |
| **Benjamin** | Qwen (Fireworks) | `!benjamin` / `!qwen` |
| **Gad** | GLM (Fireworks) | `!gad` / `!glm` |
| **Issachar** | Kimi K3 (Fireworks) | `!issachar` / `!kimi` / `!k3` |
| **Levi** | base-model simulator | `!levi` / `!sim` |

> This is a **Slack** port of a Discord bot. It is **architecturally complete but has not yet been
> smoke-tested against a live Slack workspace** — wire your tokens, invite the bot, and expect to
> shake out a few rough edges. See [Status](#status).

## How it works

Messages the bot can see (channels it's invited to, in `allowed_channels`) are routed by a heuristic
**argmax over the enabled heads** — no extra LLM call — with cost tiebreaks:

- Images → Claude or Gemini (the others are text-only here)
- CJK text → Deepseek (deeper Chinese training data)
- Novel reasoning / long-context synthesis → Gemini
- Complex code / careful epistemics → Claude
- Routine code / math → Qwen; French intent → Mistral
- Short factual / casual → Deepseek (much cheaper); ties → cheaper head wins

Override anything with a head's prefix (`!judah …`), pin a channel with `!prefer`, or stack flags
(`!think:max !judah …`). The canonical reply label is always the model name in brackets (e.g.
`[Claude]`) regardless of skin.

## Setup

1. **Create the Slack app** from the manifest: <https://api.slack.com/apps> → *Create New App* →
   *From a manifest* → paste [`slack-app-manifest.yaml`](slack-app-manifest.yaml). It enables Socket
   Mode (no public URL needed).
2. **Tokens** → copy into `.env` (see [`.env.example`](.env.example)):
   - *OAuth & Permissions* → Install to Workspace → **Bot User OAuth Token** (`xoxb-`) → `SLACK_BOT_TOKEN`
   - *Basic Information* → App-Level Tokens → generate one with `connections:write` (`xapp-`) → `SLACK_APP_TOKEN`
3. **Model keys** — set at least one in `.env` (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
   `DEEPSEEK_API_KEY`, `FIREWORKS_API_KEY`, `MISTRAL_API_KEY`). Missing keys just disable that head.
4. **Config** — `cp config.example.json config.json` and fill in `allowed_channels` (Slack channel
   IDs). The default `theme` is `isaic`; switch to `eva` or `nightvale` to re-skin.
5. **Install + run**:
   ```bash
   pip install -r requirements.txt
   python main.py            # add --check to validate config + objects without connecting
   ```
6. **Invite the bot** to each allowed channel: `/invite @ISAIC`.

## Commands

Talk to it in an allowed channel (the bot replies in a thread). Highlights:

- `!help` — full command list (themed)   ·   `!models` — heads + usage   ·   `!cost` — usage, cost & a rough energy/CO₂ estimate
- `!search <q>` — web search with citations   ·   `!research <q>` — multi-model panel + judge → one answer
- `!remember/!forget/!keep` — long-term + working memory   ·   `!prefer <head|auto>` — per-channel routing
- `!presence` — show session/ambient speaker state   ·   `!presence auto|session|ambient|tags` — override it
- `!load <ao3-url>` / `!load_text` — bookclub mode (pin a long text; `!scope`/`!chapters` per thread)
- `!speak <汉字>` — Mandarin TTS (forced tones)   ·   `!french <phrase>` — French TTS + IPA (needs Azure)
- React 👍/👎 to a reply to calibrate routing.

## Speaker-aware participation

The default `auto` mode stays conversational when only one human is active. If a second human
speaks within 15 minutes—or a Slack thread clearly becomes human-to-human—the channel enters
`ambient` mode for six hours. Mentions, model prefixes, commands, and a thread reply following the
bot's own turn always get through. Obvious replies between humans and lightweight backchannels stay
silent; only an ambiguous, unusually valuable intervention is sent to `claude-haiku-4-5`, with a
conservative 0.90 threshold and a 15-minute unsolicited-response cooldown.

Speaker state and Haiku usage survive restarts in `memories.json`. Parent-channel history and each
encountered thread are lazily reconstructed after downtime. Tune the windows, model, threshold, or
per-channel overrides in the `participation` block of `config.example.json`.

## Themes

Set `"theme"` in `config.json` (display-only — never changes routing or the canonical `[Claude]`
label). `isaic` (default, twelve tribes), `eva` (the EVA/MAGI cast), or `nightvale` (the five heads
of the dragon Hiram McDaniels: `!gold`/`!blue`/`!green`/`!violet`/`!gray` + `!carlos`/`!faceless`).

## Architecture

A `ChatPlatform` protocol ([`isaic/platform.py`](isaic/platform.py)) decouples the core bot from the
chat service; `SlackAdapter` ([`isaic/slack_adapter.py`](isaic/slack_adapter.py)) implements it on
`slack_bolt` in Socket Mode. The core logic — routing, generation, memory, commands, bookclub, TTS,
LaTeX, research — lives in [`isaic/core.py`](isaic/core.py); formatting (Markdown → Slack mrkdwn +
Block Kit) in [`isaic/formatting.py`](isaic/formatting.py); the flavor skins in
[`isaic/theme.py`](isaic/theme.py). Persistence (`memories.json`) is keyed per
`slack:{team}:{channel}`.

**Security / privacy**: never commit `.env`, `config.json`, or `memories.json` (all gitignored).
Model traffic goes to whichever providers you enable; DeepSeek's default endpoint is in China — use
its `fireworks` backend for US/ZDR (see `config.example.json`).

## Status

- ✅ Platform abstraction + Slack adapter (Socket Mode), ISAIC theme, full feature port, config +
  manifest, syntax-validated, import-smoke clean.
- ✅ Speaker-aware session/ambient participation gate with persistent state, Slack mention/thread
  semantics, deterministic human-to-human suppression, and a separately metered Haiku classifier.
- ⚠️ **Owes a live Slack smoke test**: @-mention reply, thread continuity, 👍/👎 calibration, a file
  upload (LaTeX/TTS), and `!load`. Expect to fix small Slack-API edges once tokens are wired.

## Prompt-cache ordering (2026-11-07)

Mirrored from the Hydra Discord bot. Every head but Claude does **automatic prefix
caching**, matched on the longest common prefix, so anything that churns between
turns must sit downstream of everything stable — a change near the front of the
prompt invalidates the cache for itself *and every token after it*, conversation
history included.

`_generate_response` was joining churning blocks into the system message ahead of
the history, and `openai_messages.insert(0, ...)` puts that at the very head of the
prompt. The memory context (changes whenever a `[note:]` lands) and the
sparse-memory nudge (flips at the `<3` boundary) therefore re-billed the whole
prompt at full input price. DeepSeek is $0.003/M cached vs $0.15/M miss (50×),
Kimi $0.30/M vs $3/M (10×), Qwen $0.014/M cached.

The prompt is now split: `stable_system` stays the cacheable prefix, and
`volatile_context` is appended to the newest user turn by `_append_volatile_context`.
One change at the assembly site covers every chat dispatch path, falling back to the
old system-prompt placement when the newest turn is not a user message. The
simulator deliberately keeps its volatile context in the preamble, since the tail of
a sim prompt is transcript body. Claude gains an opt-in `cache_system` breakpoint
(default on) because it caches only what carries `cache_control`; the panel and judge
pass `cache_system=False`, since a one-shot prompt pays the cache-write premium and
is never read back.

Slack-specific note: `fetch_thread_index` is still a stub returning `""`, so ISAIC
has none of the minute-granular thread-age churn the Discord bot had — only the
memory block and the nudge. If that stub is ever implemented with relative ages, it
already routes into `volatile_parts`, so it lands at the tail; just don't move it
into the system message. The invariant is documented at the stub.

Offline-validated with `scratchpad/test_prompt_cache_order.py` (31/31, run as
`BOT_MODULE=isaic.core python scratchpad/test_prompt_cache_order.py`); the full
pre-existing suite stays green. ⚠️ Owes a live Slack smoke test alongside the others.

## ⚠️ The 2026-11-07 work is OFFLINE-VALIDATED ONLY

Everything dated 2026-11-07 below — the prompt-cache split and breakpoint, the
Kimi reasoning-effort tiers, DeepSeek vision, the per-request tool-cost bucket —
has **not been exercised against live Slack or a live provider from this repo.**
It is published because the reasoning and the offline coverage are worth sharing,
not because it has been run in anger here.

What it does rest on:

- **Offline harnesses pass** — 10 of them, including the cross-turn prefix
  stability invariant driven through the real `_generate_response`, and the
  per-request tool-cost accounting. Run any of them with
  `BOT_MODULE=isaic.core python scratchpad/<name>.py`.
- **The cache claim is live-proven in the sibling Discord bot**, which shares
  this code: three real API calls there measured an 8,838-token prefix, read
  8,760 of it back on a warm turn while the volatile tail churned on every axis,
  and read **nothing** on a control call using the old prompt placement. That is
  an API-level fact about prefix caching, so it carries across — but it was
  measured there, not here.
- **This bot has no live `config.json` in this checkout**, which is the honest
  reason its Slack smoke tests have been outstanding: nothing has been wired up
  to run.

Still owed, and not blockable from here: an @-mention round trip, thread
continuity, 👍/👎 calibration, a file upload, `!load`, and one `!cost` showing
cache hits climbing across consecutive turns. Expect small Slack-API edges on
first contact.

## Carbon constants: Anthropic's grid figure was market-based (2026-11-07)

`!cost`'s per-provider `grid_gco2_per_kwh` is documented as the intensity of the
grid a provider's endpoint **physically** runs on. Claude's was 250 g/kWh,
justified in-comment as "AWS fleet: RECs + clean anchors" — which is
**market**-based accounting. Renewable energy certificates are a financial
instrument; they do not change which generators serve the load. So the premium
head was being flattered by the standard our own field definition sets.

Two facts make it worse rather than better:

- **2026-05-06: Anthropic leased all of Colossus 1** — Memphis, TN, 300+ MW,
  220k+ GPUs — from SpaceX, explicitly to serve Claude Pro and Max. That is
  *inference* capacity, which is exactly what this field prices. Leased, not
  bought.
- **That site runs mostly on on-site methane.** Roughly 35 simple-cycle gas
  turbines, about 422 MW, against a permit for 15 — the subject of a Clean Air
  Act complaint and an NAACP/SELC suit — topped up by ~150–250 MW of TVA grid.
  Simple-cycle gas is dirtier than the combined-cycle plants behind the IPCC AR5
  lifecycle median of ~490 g/kWh, and TVA's own 2025 system average is 721.69
  lb/MWh ≈ **327 g/kWh**.

Anthropic's AWS Trainium and GCP TPU capacity is cleaner, and the blend is not
knowable from outside, so the figure moves to **400 g/kWh** — the US average,
which is also this bot's global fallback. The residual error has a known
direction: if Memphis carries more of the serving load than the rest of the
fleet, 400 is still low. The participation classifier runs on the same fleet and
moves with it.

The same sweep re-checked China, where the figure was *not* wrong. The
wind/solar/hydro build-out has pulled the national average to ~530 g/kWh
(MEE/NBS official, 2023), ~526 (Ember 2025) or ~589 (IEA 2025), so DeepSeek's
550 stays. What was wrong was the **contrast**: 250-against-550 implied a ~2x
gap between a US and a Chinese endpoint where the honest figure is ~1.3x.

Not ported from the Discord bot: its ranked `!cost` layout and the volume-free
`Per Mtok` line, which is where that repo surfaces the more interesting point —
cumulative CO2e is *volume x intensity*, and volume is a routing artifact, so
per token the premium head is the dirty one. This repo keeps the older
registry-ordered summary for now.

## Fleet parity: DeepSeek vision, tool-cost bucket, field-map fix (2026-11-07)

Three things the Discord bot had and this one didn't, plus one correction.

- **DeepSeek `supports_vision=True`.** Native multimodal since 2026-09-10 —
  base64 / http(s) URL / Files API id, JPEG·PNG·GIF·WebP, an upper bound of
  1024 tokens per image, ≤600 images and 48 MiB per request, user messages
  only. No new wiring: `_convert_messages_to_openai_format` already emits
  OpenAI `image_url` blocks. **Routing effect:** `vision_pool` now includes
  Zebulun, so an image message is no longer forced to Claude or Gemini — at
  $0.15/M against $4/M that is the cheap eye.
- **Per-request tool cost bucket.** DeepSeek's Search & Fetch APIs bill $0.75
  per 1,000 successful requests *each*, metered separately, live since
  2026-10-31 16:00 UTC. `search_cost_per_1k_requests` /
  `fetch_cost_per_1k_requests` (0.0 on every other head), counters, and
  `record_tool_requests()` — deliberately **not** `record_usage`, because
  request-priced spend must stay out of the token buckets and the per-token
  energy/CO₂ estimate. Dollars are metered at record time so a rate change
  never reprices history, and it bills even under `cost_mode="local"` (a
  self-host pays nothing for inference, but Search/Fetch is hosted). Nothing
  calls it yet — search runs on Tavily — so it exists to stop a future switch
  under-reporting silently.
- **`_BACKEND_FIELD_MAP` gained `max_context_tokens`**, closing a drift from
  the Discord bot's September Qwen work. Without it a backend serving a
  different-sized model couldn't declare its own context window.
- **Correction: the Kimi plain tier is `high` here, not `low`.** The Discord
  bot wants `low` on plain turns because it is a chat bot; copying that value
  into the lab bot mirrored the wrong intent. Thoroughness is this bot's
  product, so its `moonshot` backend asks for `high`. Note also what the
  Fireworks default's `None` actually means — no param, so the provider's own
  default applies, and on Moonshot's K3 that is `max`. "Unset" here is the most
  thorough setting, not the least.

Offline-validated: `scratchpad/test_tool_request_cost.py` (31) ported, and
`test_kimi_effort_tiers.py` reworked to assert a tier is *set* and valid rather
than pinning one bot's choice, so the one file now covers both repos (23 each).
Full suite green.

## Kimi reasoning-effort tiers (2026-11-07)

Mirrored from Hydra. Moonshot's docs: the whole **K3 series always thinks** and
takes a top-level `reasoning_effort` of `low` | `high` | `max`, **defaulting to
`max`**. Our earlier note that K3 accepts only `"max"` was wrong. The consequence
is that omitting the param on a plain turn does not mean "thinking off" — it means
"thinking at max".

`ModelProvider` gained `plain_reasoning_effort` alongside `think_reasoning_effort`,
and the shim now picks a tier per turn rather than sending one only under `!think`.

This bot's **default Kimi backend is Fireworks**, serving the open weights, which
may not implement `reasoning_effort` at all — and a param the endpoint rejects
fails the whole turn. So the plain tier is left `None` on the Fireworks default
(a plain turn sends nothing, exactly as before) and set to `"low"` on the
non-default `moonshot` backend, where the documented behavior applies. To enable
it for Fireworks, smoke-test one plain and one `!think` Kimi turn, then add
`plain_reasoning_effort="low"` to the default spec.

Clearing a tier per backend required a new `_BACKEND_NULLABLE_FIELDS` list:
`_apply_backend` skips a `None` override by default, so without it a backend would
silently inherit the other route's tier. Both reasoning fields were also added to
`_BACKEND_FIELD_MAP`.

Offline-validated with `scratchpad/test_kimi_effort_tiers.py` (22/22, run as
`BOT_MODULE=isaic.core python scratchpad/test_kimi_effort_tiers.py`). It resolves
the Moonshot and Fireworks routes by `base_url`, so the one file covers both this
bot and Hydra despite their inverted defaults.

## Opus 5.5 migration (2026-10-05)

The Claude Opus head now uses `claude-opus-5-5`: $4/M uncached input,
$20/M output and $0.20/M cache reads. Five-minute cache writes cost $5/M;
one-hour writes cost $8/M. Persisted historical cost estimates retain the old
rates, while new calls accumulate dollars at their applicable rates.
Thinking is adaptive; tool continuations preserve signed thinking blocks,
the original system prompt and the full preceding conversation.
Plain chat/search and summaries use low effort; explicit reasoning keeps
its existing higher effort. Summaries select text blocks and reject empty or
truncated results. Qwen remains on Fireworks at its existing backend rates.

Sources: [migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide),
[Opus pricing](https://www.anthropic.com/claude-opus-5-5).
