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
| **Judah** | Claude (Opus 5) | `!judah` / `!claude` |
| **Joseph** | Gemini (3.1 Pro) | `!joseph` / `!gemini` |
| **Zebulun** | Deepseek (V4-Pro) | `!zebulun` / `!deepseek` |
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
