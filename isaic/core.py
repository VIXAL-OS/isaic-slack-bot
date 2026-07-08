"""
ISAIC core — multi-model assistant (Slack), ported from the Hydra Discord bot.
=============================================================================
Smart heuristic routing over several frontable models that share one memory;
the router picks whichever is best suited to each message (or users force one).
Skinned by default as the twelve tribes (ISAIC theme): Judah=Claude,
Joseph=Gemini, Zebulun=Deepseek, Naphtali=Mistral, Benjamin=Qwen, Gad=GLM,
Levi=simulator.

This module is platform-agnostic: all chat I/O goes through a ChatPlatform
(see platform.py) via a thin discord.py-shaped compatibility layer (compat.py,
imported here as `discord`). The Slack implementation lives in slack_adapter.py;
main.py wires them together. There is NO discord.py dependency.

Features: heuristic routing, two-tier memory, prompt caching, web search /
grounding, bookclub mode, Mandarin (!speak) + French (!french) TTS, LaTeX→PNG,
a research panel (!research), a base-model simulator, and cost+carbon tracking.
"""

from . import compat as discord  # discord.py-shaped shim backed by ChatPlatform (compat.py)
from . import formatting
from .platform import ChatPlatform, Message as PlatformMessage, Reaction as PlatformReaction
import anthropic
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import partial
import json
import os
import asyncio
import aiohttp
import base64
import re
import io
import html as _html
from dotenv import load_dotenv

# beautifulsoup4 is an optional dep used only by the AO3 fetcher in bookclub
# mode. The bot starts fine without it; !load just returns a clear pip-install
# message if it's missing.
try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _BeautifulSoup = None
    _HAS_BS4 = False

# matplotlib's mathtext is used to render LaTeX equations to PNG attachments
# so Discord (which has no native math rendering) can show them properly.
# Force the non-interactive Agg backend before any pyplot import so headless
# environments (and Windows without a display) don't try to open a window.
import matplotlib
matplotlib.use('Agg')
from matplotlib import mathtext as _mpl_mathtext
from PIL import Image as _PILImage

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BotConfig:
    # Model settings
    max_tokens: int = 4096
    default_model: str = "auto"  # "auto", "claude", "deepseek", or "gemini"

    # Context management (THE KEY TO NOT BEING MYK)
    # Bumped 20 → 60 and 50k → 800k to accommodate book-club mode where a
    # loaded reading material (e.g. a 320k-token AO3 fic) lives in system
    # context across a long discussion thread.
    max_messages_to_fetch: int = 60        # Fetch from Discord history
    max_longterm_memories: int = 25        # Explicit memories (!remember)
    max_working_notes: int = 10            # Auto-notes from Claude
    working_memory_decay_hours: float = 48.0  # Notes fade after ~48h

    # Token budgeting (approximate)
    max_input_tokens: int = 800_000        # Headroom for fic + discussion + memory
    chars_per_token: float = 4.0           # Rough estimate

    # Web search settings
    web_search_enabled: bool = True
    max_search_results_in_embed: int = 5   # How many sources to show
    
    # Supported image types for vision
    image_types: tuple = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
    max_image_size_mb: float = 20.0
    
    # Supported text file types
    text_file_types: tuple = ('.md', '.txt', '.py', '.js', '.ts', '.json', '.csv', '.html', '.css', '.yaml', '.yml', '.toml', '.xml', '.sql', '.sh', '.bash', '.r', '.rs', '.go', '.java', '.c', '.cpp', '.h', '.hpp')

    # PluralKit / webhook proxy compatibility
    # When a user sends a message, wait this long before responding so that
    # PluralKit (or any webhook-based proxy) has time to delete-and-repost it
    # as the alter. If the original is gone after the delay, we bail out and
    # let the webhook event re-trigger on_message with the proxied version.
    proxy_check_delay_seconds: float = 1.5

    # Optional free-text describing this workspace/team, injected into the prompt
    # via {server_context}. Override per-deployment in config.json ("server_context").
    server_context: str = ""

    # Bot behavior
    system_prompt: str = r"""You are {model_identity}, chatting in a Slack workspace.

You're helpful, harmless, and honest. You have a warm, curious personality. You can be playful but you're also genuinely knowledgeable and thoughtful.

- Be concise in casual chat, detailed when asked technical questions
- Use Slack mrkdwn formatting; keep individual messages reasonably sized
{server_context}

## Your identity

You're **{model_name}** (model ID: `{model_id}`). You know who you are — if someone asks,
just say so naturally. No need to hedge or say you "can't tell from the inside."

{identity_details}

## Multi-model system (ISAIC)

You're part of a multi-model system called ISAIC — think of it like a plural system where
different models take turns fronting. The router picks whoever's best suited for each message,
or users can call on you directly with commands like !claude, !deepseek, or !gemini.
{theme_blurb}

Your responses get labeled (e.g., **[Claude]**, **[Deepseek]**, **[Qwen]**, etc.) so
everyone can tell who said what. The labeling is handled automatically by the bot — do NOT
include ANY model-name tag (such as [Claude] — or your own name) at the start of your responses. Just write your
response normally and the system adds the label for you. When you see labeled messages
from the other models in conversation history, those are genuinely from them — your
collaborators, not copies of you. The heads share a memory system, so you'll all
see the same notes and context.

It's okay if things get a little blurry sometimes — that's natural in a shared-context system.
Just check your label and the routing info below if you need to orient yourself.

{routing_context}

## Special capabilities

**Reactions**: You can react to messages with emoji by including [react: emoji] in your response (it gets stripped from visible text).

**Files**: You can generate code files by wrapping them in ```filename.ext blocks. Long code becomes file attachments.

**Math**: Slack can't render LaTeX, so the bot does it for you. Wrap display equations in `$$...$$` and inline math in `$...$` — the bot will render each block to a PNG attachment while keeping your LaTeX source in the message body so users can copy it. Use this any time you write equations; don't strip the dollar signs.

Common pitfalls to avoid in your LaTeX (these silently produce wrong-looking renders):
- Subscripts on multi-char names: write `K_t`, `t_{1/2}`, `P_{t|t-1}`, NOT `Kt`, `t{1/2}`, `P{t|t-1}`. The underscore is required.
- Differentials: write `\,dt` and `\,dW_t` for proper thin-space spacing, NOT `,dt` or `dWt`.
- Multi-char subscripts and superscripts need braces: `H^T_t` and `H_t^T` are fine; `H^Tt` is not.
- Greek letters and operators always need a backslash: `\theta`, `\sigma`, `\sum`, `\int`, `\frac`. Bare `theta` will render as four italic letters.
- Re-read your equations before sending; a stray missing `_` or `\,` is the difference between a clean render and a confusing one.

**Speech (Mandarin TTS)**: You can attach spoken Mandarin audio inline while teaching — just write `!speak 汉字` (the command word followed immediately by the Chinese characters). The bot voices it with Azure's Xiaoxiao voice (tones forced correct via SSML), attaches an MP3, and rewrites the marker as "汉字 (spoken `pīnyīn` · dict. `pīnyīn`) 🔊" — it computes and appends the correct, sandhi-aware pinyin (both the spoken form and the dictionary form) for you. So you do NOT need to hand-write the pinyin next to a phrase you're speaking — let the bot be the source of truth; this avoids tone-mark mistakes. Write the literal `!speak` before EACH phrase you want spoken, every single time — the "🔊" you see in the result is added by the bot AFTER it processes your command, so don't reproduce that look from memory and expect audio. (As a safety net, a phrase written immediately followed by 🔊 is also spoken — but `!speak 汉字` is the real command.) Up to ~8 clips per message, e.g. a tone contrast: !speak 妈, !speak 麻, !speak 马, !speak 骂. If you ever need to override the pronunciation, use `[[speak: 汉字 | nǐ hǎo]]` to pin exact pinyin (tone marks or numbers). Mandarin only.

**Speech (French TTS)**: You can also attach spoken French inline while teaching — write `[[french: votre phrase ici]]` (English in is fine too; the bot translates to French). The bot speaks it in a natural Azure fr-FR voice (Denise) and rewrites the marker as "phrase (`IPA` — liaison note) 🔊", computing the IPA and a one-line pronunciation note for you — so DON'T hand-write IPA; let the bot be the source of truth. Up to ~8 clips per message. Unlike Mandarin (where tones are forced), French pronunciation is inferred by the native voice, so just give the words. Mistral is the resident French specialist.

**Images**: You can see images that users upload.

**Web search**: You can search the web! Users can invoke `!search <query>` to have you search for current information. Claude uses a built-in web search tool; Gemini uses Google's native search grounding; Deepseek uses Tavily via function calling. You DO have this capability — don't tell users you can't search.

## Memory System (Important!)

You have TWO types of memory:

**Working notes** - Your personal scratch space for things you notice IN THIS CONVERSATION:
- Write notes with [note: key: value] - e.g., [note: project_deadline: launch planned for late spring]
- These fade after ~48 hours if not referenced
- Frequently relevant notes stick around longer
- Max 10 notes (oldest/stalest get pushed out)
- Use these liberally! Jot down anything that might be useful later.
- IMPORTANT: Only write notes about the CURRENT conversation, not about other threads you can see.

**Long-term memories** - Permanent facts (users control these):
- Created by users with !remember
- Never decay until user does !forget
- Users can promote your working notes to permanent with !keep <key>
- Users can save thread summaries with !summarize <key>

When you reference information from your working notes, they get refreshed and stick around longer. So if you notice something and keep finding it relevant, it'll persist.

Write working notes for things like:
- Deadlines or dates people mention
- Current projects/tasks being discussed
- Preferences people express
- Names, relationships, context that comes up
- Technical details that might be relevant later

Don't be shy about noting things! The decay system handles cleanup automatically."""

CONFIG = BotConfig()


def _is_real_bot(msg: discord.Message) -> bool:
    """True if msg is from an actual Discord bot account, not a webhook proxy.

    PluralKit (and similar plural-system tools) repost user messages via
    webhooks; those have author.bot=True but represent a real user behind the
    scenes. Treat them as users, not assistants. Genuine bots — including this
    one — post directly via their bot user with no webhook_id."""
    return msg.author.bot and msg.webhook_id is None


# =============================================================================
# MODEL PROVIDERS
# =============================================================================

@dataclass
class ModelProvider:
    """Configuration and state for a single AI model provider."""
    name: str                          # Display name: "Claude", "Deepseek", "Gemini"
    model_id: str                      # API model string
    input_cost_per_million: float      # $/M input tokens (≤ tier threshold)
    output_cost_per_million: float     # $/M output tokens (≤ tier threshold)

    # --- Provider-registry / dispatch wiring (Phase 0) -----------------------
    # These make providers config-driven (config.json `providers` section)
    # instead of hardcoded in ClaudeBot.__init__. `name` stays the CANONICAL
    # routing key (NEVER rename — CLAUDE.md rule); `id` is the lowercase config
    # key. `sdk_type` drives dispatch in _generate_response without per-provider
    # if-branches. ProviderRegistry.from_config() wires base_url / api key /
    # backend from these + the config overlay.
    id: str = ""                         # config key: "claude", "deepseek", …
    sdk_type: str = "openai_compatible"  # "anthropic" | "openai_compatible" | "gemini"
    api_key_env: str = ""                # env var name holding this provider's key
    base_url: Optional[str] = None       # OpenAI-compatible endpoint (None for anthropic / gemini-native)
    alias: Optional[str] = None          # optional command-alias label
    display_name: Optional[str] = None   # optional human label (falls back to name)
    # Backend toggle (Phases 2/3): `backend` is the ACTIVE backend; `backends`
    # is a table of the NON-default backends only. The default backend's wiring
    # lives in the fields above/below, so the default config triggers ZERO
    # mutation (behavior-preserving); selecting a non-default backend merges
    # backends[backend] over those fields. e.g. deepseek: "api" (default) |
    # "fireworks" | "self_hosted"; gemini: "developer_api" (default) | "vertex".
    backend: Optional[str] = None
    backends: dict = field(default_factory=dict)
    # Backend characteristics (Phase 3). supports_server_cache=False for a
    # self-hosted endpoint (no prompt cache → cached price = input price).
    # cost_mode="local" skips per-token $ and labels the turn electricity-only
    # (tokens are still counted, so energy/CO₂ still report).
    supports_server_cache: bool = True
    cost_mode: str = "metered"           # "metered" | "local"
    # Routing metadata target (spec §4.1). INERT this phase — routing stays in
    # _estimate_confidence; never consulted by _select_model yet. Groundwork
    # for a future config-driven router.
    routing_tags: list = field(default_factory=list)

    # --- Simulator mode (Phase 7, §9) ---------------------------------------
    # completions_mode flips this provider off the chat path onto the
    # transcript-completion path (_generate_simulator_response): it talks to a
    # BASE model by continuing an IRC-style log via /completions instead of
    # sending chat messages to /chat/completions. Default False → every existing
    # provider keeps the instruct path byte-for-byte. A completions_mode provider
    # is also OVERRIDE-ONLY: _select_model never argmax's it (a cold/garrulous
    # base model shouldn't win auto-routing), but it's reachable by command
    # (!dummy) and as a sticky channel pref (the spec's "designated channel").
    completions_mode: bool = False
    # Base-model sampler knobs the instruct path ignores (§9.3). The OpenAI-
    # standard keys (temperature/top_p/frequency_penalty/presence_penalty/stop)
    # go straight to client.completions.create; the non-standard ones
    # (top_k/min_p/top_a/repetition_penalty) ride in extra_body for vLLM /
    # Fireworks. Config overrides any key via providers.<id>.sampler; a None
    # value drops that knob (lets the server pick its own default).
    sim_sampler: dict = field(default_factory=dict)
    # Query-driven search grounding for the simulator path (§9.2). A base model
    # can't call the web_search tool itself, so when this is True the simulator
    # detects a search intent in the latest user turn, runs `search_backend`
    # itself, and folds the retrieved snippets into the transcript preamble
    # (the chat heads append a tool result instead). Default False → the Dummy
    # Plug never auto-searches; pasted-URL grounding still rides in via
    # _augment_with_url_extracts. Flip via providers.<id>.search=true.
    sim_search: bool = False

    max_tokens: int = 4096
    # Context window in tokens — the API limit, not our budget. Used to gate
    # whether !load can attach a reading material to this provider's calls
    # (book-club mode requires ~320k+ for fics like Almost Nowhere).
    max_context_tokens: int = 200_000
    enabled: bool = True
    supports_vision: bool = True       # Can handle image content
    supports_web_search: bool = False  # Has built-in web search tool

    # Tiered pricing: when context_tier_threshold is set, requests with
    # input_tokens > threshold are billed at the *_above_tier rates instead.
    # Used by Gemini 3.1 Pro: ≤200k tokens is one rate, >200k is roughly 2x.
    # If None, single-rate pricing applies regardless of context length.
    context_tier_threshold: Optional[int] = None
    input_cost_per_million_above_tier: Optional[float] = None
    output_cost_per_million_above_tier: Optional[float] = None

    # Cached-input pricing: tokens served from the provider's prompt cache
    # bill at this (much lower) rate. Set on providers that support caching
    # AND where we extract the cache-hit count from the usage response.
    # Claude: cache_read_input_tokens (~10% of input rate).
    # Deepseek: prompt_cache_hit_tokens (~99% off via auto-server-cache).
    # Gemini native: usageMetadata.cachedContentTokenCount (~25% of input
    # rate). The OpenAI shim doesn't expose Gemini's cache info, so cache
    # accounting only kicks in for the native chat path.
    cached_input_cost_per_million: Optional[float] = None
    cached_input_cost_per_million_above_tier: Optional[float] = None

    # Provider quirks for the OpenAI-compatible generator
    # ---------------------------------------------------
    # requires_reasoning_echo: When thinking mode is on, the provider's API
    #   requires `reasoning_content` to be echoed back on every prior assistant
    #   turn. Deepseek V4 needs this; Gemini's shim handles thinking server-side.
    # disables_thinking_by_default: The provider enables a thinking/reasoning
    #   mode by default that we need to explicitly disable with
    #   `extra_body={"thinking": {"type": "disabled"}}` when thinking=False.
    #   Deepseek V4 has this; Gemini does not.
    requires_reasoning_echo: bool = False
    disables_thinking_by_default: bool = False

    # Which SearchBackend to use when this provider needs to ground via the web.
    # Values: "tavily", "google_native", or None (no external search). Claude has
    # supports_web_search=True and uses its native Anthropic web_search tool,
    # bypassing the SearchBackend system entirely. New backends slot in by
    # adding a key here and a matching entry in ClaudeBot.search_backends.
    search_backend: Optional[str] = None

    # --- !cost energy/CO₂ gauge (order-of-magnitude, NOT a measurement) -------
    # est_wh_per_1k_tokens: rough watt-hours per 1k tokens for a NON-reasoning
    # turn. Anchored to measured frontier inference (~0.3–0.6 Wh/query, median
    # 0.31; MoE/low-active-param models land lower) — arXiv 2505.09598. Per-token
    # metering already captures most of the reasoning blow-up: a !think turn just
    # emits far more tokens, so the measured ~70–100× per-*query* hit on R1-class
    # models is mostly token count, leaving only a residual ~3–5× per-token
    # premium we don't model. Relative ranking is the signal, not the absolute.
    est_wh_per_1k_tokens: float = 0.5
    # grid_gco2_per_kwh: carbon intensity of the grid/fleet THIS provider's
    # endpoint actually runs on (gCO₂e/kWh). Per-provider because it's the
    # well-documented, decision-relevant axis: France/EU nuclear ≈ 20, US average
    # ≈ 400, Google (66% 24/7-CFE) ≈ 130, AWS (RECs + clean anchors) ≈ 250, a
    # no-CFE US serverless fleet ≈ US avg, China grid ≈ 580. Tied to the
    # *endpoint*, not the brand: Mistral-on-Fireworks is US (~400), NOT French
    # nuclear — you'd only get the ~20 by calling api.mistral.ai directly.
    # None → fall back to the global GRID_GCO2_PER_KWH.
    grid_gco2_per_kwh: Optional[float] = None
    # train_tco2e: one-time TRAINING carbon (tonnes CO₂e), embodied at the
    # *training* grid (France ~clean for Mistral; a China province for
    # Qwen/GLM/DeepSeek; US for Claude/Gemini) — NOT the inference grid above.
    # Amortized over MODEL_LIFETIME_TOKENS into a separate !cost line. None →
    # omit. Published anchors are rare (Mistral Large-2 LCA = 2200 t); the rest
    # are order-of-magnitude estimates. Amortized share is small for widely-
    # served models (~10–15% of lifetime per Mistral's LCA), so the wide error
    # bars don't swamp the firmer inference gauge.
    train_tco2e: Optional[float] = None

    # Runtime stats — bottom-tier (or all, if untiered)
    total_input_tokens: int = 0          # uncached input only
    total_output_tokens: int = 0
    total_cached_input_tokens: int = 0   # served from prompt cache
    total_requests: int = 0

    # Runtime stats — above-tier (only used if context_tier_threshold set)
    total_input_tokens_above_tier: int = 0
    total_output_tokens_above_tier: int = 0
    total_cached_input_tokens_above_tier: int = 0

    # Estimated cache-storage cost. Gemini context caches bill per token-hour for
    # their whole TTL — a time-based charge we can't meter per request — so this
    # is an upper-bound estimate accumulated at cache creation and surfaced in
    # !cost (see _create_gemini_cache / get_cost_summary).
    total_cache_storage_cost_est: float = 0.0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> None:
        """Record one request's usage, routing into the right tier bucket.

        `input_tokens` is interpreted as the UNCACHED portion (Claude reports
        this directly as `input_tokens`; for Gemini/Deepseek the caller must
        subtract the cached count themselves before passing). Cache hits go
        into the separate cached counter and bill at the cache rate in
        get_cost().

        The total request size used for tier routing is uncached + cached,
        since the API still has to process that much context (and tier
        breakpoints are based on context size, not billing).
        """
        request_size = input_tokens + cached_input_tokens
        if (
            self.context_tier_threshold is not None
            and request_size > self.context_tier_threshold
        ):
            self.total_input_tokens_above_tier += input_tokens
            self.total_output_tokens_above_tier += output_tokens
            self.total_cached_input_tokens_above_tier += cached_input_tokens
        else:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cached_input_tokens += cached_input_tokens

    def get_cost(self) -> float:
        """Get total cost for this provider across all pricing tiers and cache states."""
        # Local / self-hosted backends bill electricity, not per token, so there
        # is no $ figure to report. Tokens are still counted, so get_energy_wh /
        # get_co2_g still surface the carbon. See cost_mode + the "local" label
        # in ConversationManager.get_cost_summary.
        if self.cost_mode == "local":
            return 0.0
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_cost_per_million
        # Cached input bills at its own (much lower) rate. If no cache rate is
        # configured but cached tokens exist (shouldn't happen normally), fall
        # back to the input rate so we don't silently zero out the cost.
        cache_rate = self.cached_input_cost_per_million
        if cache_rate is None and self.total_cached_input_tokens > 0:
            cache_rate = self.input_cost_per_million
        cached_cost = (self.total_cached_input_tokens / 1_000_000) * (cache_rate or 0.0)
        if self.context_tier_threshold is not None:
            above_in_rate = self.input_cost_per_million_above_tier or self.input_cost_per_million
            above_out_rate = self.output_cost_per_million_above_tier or self.output_cost_per_million
            above_cache_rate = (
                self.cached_input_cost_per_million_above_tier
                or self.cached_input_cost_per_million
                or self.input_cost_per_million_above_tier
                or self.input_cost_per_million
            )
            input_cost += (self.total_input_tokens_above_tier / 1_000_000) * above_in_rate
            output_cost += (self.total_output_tokens_above_tier / 1_000_000) * above_out_rate
            cached_cost += (self.total_cached_input_tokens_above_tier / 1_000_000) * above_cache_rate
        return input_cost + output_cost + cached_cost

    def get_energy_wh(self) -> float:
        """Order-of-magnitude inference-energy estimate in watt-hours: every
        token this provider has processed × its per-1k-token Wh proxy. A
        relative gauge for !cost, not a measurement (see est_wh_per_1k_tokens)."""
        total_tokens = (
            self.total_input_tokens
            + self.total_cached_input_tokens
            + self.total_output_tokens
            + self.total_input_tokens_above_tier
            + self.total_cached_input_tokens_above_tier
            + self.total_output_tokens_above_tier
        )
        return total_tokens / 1000.0 * self.est_wh_per_1k_tokens

    def get_co2_g(self, fallback_grid_gco2_per_kwh: float) -> float:
        """Rough CO₂e grams: this provider's energy × the carbon intensity of
        the fleet its endpoint runs on (its own grid_gco2_per_kwh, else the
        passed global fallback). The per-provider grid is where the real signal
        lives — the same tokens on French nuclear vs a US gas fleet differ ~20×."""
        grid = self.grid_gco2_per_kwh
        if grid is None:
            grid = fallback_grid_gco2_per_kwh
        return self.get_energy_wh() / 1000.0 * grid

    def get_amortized_train_co2_g(self, lifetime_tokens: float) -> float:
        """Rough amortized TRAINING CO₂e (grams) attributable to the tokens this
        provider has served: total training carbon spread over an assumed model
        lifespan. Squishy — the lifetime token count is the dominant unknown —
        so it's surfaced as its own !cost line, never folded into the firmer
        inference number. 0 when train_tco2e is unknown."""
        if not self.train_tco2e or lifetime_tokens <= 0:
            return 0.0
        served = (
            self.total_input_tokens + self.total_cached_input_tokens
            + self.total_output_tokens + self.total_input_tokens_above_tier
            + self.total_cached_input_tokens_above_tier
            + self.total_output_tokens_above_tier
        )
        return self.train_tco2e * 1_000_000.0 * (served / lifetime_tokens)

    def to_stats_dict(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_input_tokens": self.total_cached_input_tokens,
            "input_tokens_above_tier": self.total_input_tokens_above_tier,
            "output_tokens_above_tier": self.total_output_tokens_above_tier,
            "cached_input_tokens_above_tier": self.total_cached_input_tokens_above_tier,
            "requests": self.total_requests,
            "cache_storage_cost_est": self.total_cache_storage_cost_est,
        }

    def load_stats(self, data: dict) -> None:
        self.total_input_tokens = data.get("input_tokens", 0)
        self.total_output_tokens = data.get("output_tokens", 0)
        self.total_cached_input_tokens = data.get("cached_input_tokens", 0)
        self.total_input_tokens_above_tier = data.get("input_tokens_above_tier", 0)
        self.total_output_tokens_above_tier = data.get("output_tokens_above_tier", 0)
        self.total_cached_input_tokens_above_tier = data.get("cached_input_tokens_above_tier", 0)
        self.total_requests = data.get("requests", 0)
        self.total_cache_storage_cost_est = data.get("cache_storage_cost_est", 0.0)


@dataclass
class SearchResult:
    """Output of any SearchBackend.

    - text: human-/model-readable summary. When `is_grounded_answer` is True,
      this is already a synthesized answer (just display it). Otherwise it's
      raw search results that should be fed back through a model for synthesis.
    - citations: structured source list for rendering as Discord embeds.
    - is_grounded_answer: distinguishes "model already answered with citations"
      (Google native) from "here are raw search hits" (Tavily).
    - queries_used: search queries actually executed (Google native reports these).
    """
    text: str
    citations: list[dict] = field(default_factory=list)  # [{url, title, snippet}]
    is_grounded_answer: bool = False
    queries_used: list[str] = field(default_factory=list)


# SearchBackend is just a shape — anything with `async def search(query, max_results=5) -> SearchResult`
# satisfies it. Backends are instantiated in ClaudeBot.__init__ and keyed by the
# string a ModelProvider sets in its `search_backend` field. Today: "tavily",
# "google_native". To add another (e.g. Brave Search): write a class with the
# same method shape, instantiate it in __init__, key it in self.search_backends.


# Fallback grid carbon intensity (g CO₂e per kWh) for any provider that doesn't
# set its own grid_gco2_per_kwh. Default ≈ the US average grid (~384 g in 2024;
# IEA puts 2025–26 at ~400–435) — override with GRID_GCO2_PER_KWH in .env. Each
# provider overrides this with the intensity of the fleet its endpoint really
# runs on (see ModelProvider.grid_gco2_per_kwh). A transparent, tunable
# assumption — the !cost line is an order-of-magnitude *relative* gauge, not a
# precise footprint, and we never route on it.
GRID_GCO2_PER_KWH = float(os.getenv("GRID_GCO2_PER_KWH", "400"))

# Amortization horizon for embodied TRAINING carbon (see ModelProvider.train_
# tco2e). The dominant unknown — a frontier model serves ~10^13–10^15 tokens
# over its ~1–2-yr life. Default 1e14 (~100T) is calibrated so Mistral's public
# LCA reproduces (~2200 t training ≈ 10–15% of its lifetime footprint). Override
# with MODEL_LIFETIME_TOKENS in .env; AMORTIZE_TRAINING=0 hides the line.
MODEL_LIFETIME_TOKENS = float(os.getenv("MODEL_LIFETIME_TOKENS", "1e14"))
AMORTIZE_TRAINING = os.getenv("AMORTIZE_TRAINING", "1") not in ("0", "false", "False", "")

# Alternation of model display-names (provider.name values) that can show up as
# echoed "[Name]" labels in model output. The response pipeline strips any of
# these a model prepends BEFORE the bot adds its own authoritative label, and
# history parsing uses it to recognize who said what. A prompt instruction alone
# never fully stopped the echo — this strip is the real fix. ⚠️ KEEP IN SYNC
# with provider.name when adding a head (else its echo leaks through, the
# "[Qwen] [Qwen]" double-label bug).
MODEL_LABEL_NAMES = "Claude|Deepseek|Gemini|Mistral|Qwen|GLM|Dummy"


CLAUDE_PROVIDER = ModelProvider(
    name="Claude",
    id="claude",
    sdk_type="anthropic",
    api_key_env="ANTHROPIC_API_KEY",
    model_id="claude-opus-4-8",
    input_cost_per_million=5.0,
    output_cost_per_million=25.0,
    # cache_read_input_tokens bill at 10% of standard input rate. (Cache
    # writes on the first turn are billed at 1.25x; we lump those into the
    # regular input counter — a small under-bill once per session.)
    cached_input_cost_per_million=0.5,
    max_context_tokens=1_000_000,  # 1M GA'd for Opus 4.6+ in March 2026
    supports_vision=True,
    supports_web_search=True,
    est_wh_per_1k_tokens=0.6,  # large frontier model, but Anthropic ranks high on per-query efficiency
    grid_gco2_per_kwh=250.0,   # AWS fleet: RECs + clean anchors (e.g. the PA nuclear Anthropic capacity)
    train_tco2e=8000.0,        # undisclosed — order-of-magnitude estimate for an Opus-class US train
)

DEEPSEEK_PROVIDER = ModelProvider(
    name="Deepseek",
    id="deepseek",
    sdk_type="openai_compatible",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
    backend="api",
    model_id="deepseek-v4-pro",
    input_cost_per_million=0.435,
    output_cost_per_million=0.87,
    # Auto server-side prefix caching; cached input bills at ~99% off.
    # (Both rates are current through the May 31 2026 promo discount.)
    cached_input_cost_per_million=0.003625,
    # 1M context per DeepSeek V4-Pro docs (max output 384k). Server-side
    # context caching is automatic — no client flags required.
    max_context_tokens=1_000_000,
    supports_vision=False,
    supports_web_search=False,
    # Deepseek V4 enables thinking by default and requires reasoning_content
    # to be echoed back on every prior assistant turn when thinking is on.
    requires_reasoning_echo=True,
    disables_thinking_by_default=True,
    # Deepseek has no native web search; route through Tavily.
    search_backend="tavily",
    est_wh_per_1k_tokens=0.3,  # sparse MoE (~37B active) — light per token
    grid_gco2_per_kwh=550.0,   # DeepSeek China API (east-CN grid; province-dependent — Sichuan hydro ~112, Inner Mongolia coal higher). fireworks backend → ~400; self-host → your grid
    train_tco2e=1000.0,        # ~1.4 GWh for V3 (2.78M GPU-hrs); V4 est. — very training-efficient
    # Backend toggle (Phase 3). Default "api" (above) = China api.deepseek.com.
    # The Discord bot stays on "api" by design (CLAUDE.md locked deviation);
    # these are the lab-route options. Selecting one overrides base_url / key /
    # model / cost / grid. ⚠️ Fireworks pricing + grid are estimates — VERIFY.
    backends={
        "fireworks": {  # US, zero-data-retention, server-side cache (50%)
            "base_url": "https://api.fireworks.ai/inference/v1",
            "api_key_env": "FIREWORKS_API_KEY",
            "model": "accounts/fireworks/models/deepseek-v4-pro",
            "input_cost_per_million": 1.74,          # VERIFY Fireworks pricing
            "output_cost_per_million": 3.48,
            "cached_input_cost_per_million": 0.87,   # Fireworks caches input at 50%
            "grid_gco2_per_kwh": 400.0,              # Fireworks US fleet
            "supports_server_cache": True,
        },
        "self_hosted": {  # local vLLM/Ollama — no egress, electricity-only cost
            "base_url": "http://localhost:8000/v1",  # vLLM; Ollama = :11434/v1
            "api_key_env": None,                      # local server, no key
            "model": "deepseek-v4-flash",             # single-GPU; full V4 is multi-GPU
            "supports_server_cache": False,
            "cost_mode": "local",
            "grid_gco2_per_kwh": None,                # your grid → global GRID_GCO2_PER_KWH
        },
    },
)

GEMINI_PROVIDER = ModelProvider(
    name="Gemini",
    id="gemini",
    sdk_type="gemini",
    api_key_env="GEMINI_API_KEY",
    # OpenAI-compatible shim endpoint (non-bookclub chat). The native
    # generateContent + cachedContents paths build their own URLs via aiohttp.
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    backend="developer_api",
    # Newest fanciest as of May 2026. The OpenAI shim may also accept
    # "gemini-3.1-pro" if "-preview" gets rejected — check AI Studio logs.
    model_id="gemini-3.1-pro-preview",
    # Standard tier pricing from https://ai.google.dev/gemini-api/docs/pricing
    # (≤200k input tokens). Above the tier, input is $4.00/M and output $18.00/M.
    input_cost_per_million=2.0,
    output_cost_per_million=12.0,
    context_tier_threshold=200_000,
    input_cost_per_million_above_tier=4.0,
    output_cost_per_million_above_tier=18.0,
    # Implicit-cache reads bill at ~25% of standard input rate (75% discount).
    # Verify exact numbers on the pricing page — these are best-known estimates.
    cached_input_cost_per_million=0.5,
    cached_input_cost_per_million_above_tier=1.0,
    est_wh_per_1k_tokens=0.5,  # large Pro model
    grid_gco2_per_kwh=130.0,   # Google fleet: 66% 24/7 carbon-free (some regions ≥80%) — cleanest US hyperscaler
    train_tco2e=3000.0,        # undisclosed — large model but on Google's clean training fleet → est.
    # 1M context standard for Gemini Pro line. Bookclub-mode chat (when a
    # reading material is loaded) routes through _generate_gemini_native_response
    # instead of the OpenAI shim, so we can reference cachedContent — Google's
    # shim still rejects extra_body cached_content as of May 2026, so the
    # native path is the only one that gets the cache discount. Non-bookclub
    # chat stays on the shim for uniformity with Deepseek.
    max_context_tokens=1_000_000,
    # Caspar can see — unlike Melchior.
    supports_vision=True,
    # Chat goes through the OpenAI shim (for codepath uniformity), but search
    # uses the native API via aiohttp so we get free, high-quality google_search
    # grounding with structured citations. See _google_native_search.
    supports_web_search=False,  # search comes via SearchBackend, not the chat API
    # Gemini's OpenAI shim handles thinking server-side — no echo dance,
    # no opt-out kwarg required.
    requires_reasoning_echo=False,
    disables_thinking_by_default=False,
    # Native Google Search grounding via aiohttp to the native endpoint.
    search_backend="google_native",
    # Backend toggle (Phase 2). Default "developer_api" (above) is no-train when
    # billing is enabled on the Google project (operator setting, not code).
    # "vertex" is the optional enterprise rung (residency / IAM / SLA): auth via
    # ADC, a DIFFERENT caching surface (vertexai.caching.CachedContent), same
    # model slug. ⚠️ code-complete but UNVERIFIED — needs GCP ADC to smoke-test.
    backends={
        "vertex": {
            "project_env": "GOOGLE_CLOUD_PROJECT",
            "location_env": "GOOGLE_CLOUD_LOCATION",
            "location_default": "us-central1",
            "api_key_env": None,           # vertexai SDK uses ADC, not an API key
            "grid_gco2_per_kwh": 130.0,    # region-dependent; Google fleet default
        },
    },
)


# --- Open-weight heads, all served through ONE Fireworks endpoint + ONE key ---
# US, zero-data-retention infrastructure. EVA pilots (the cheap units you
# deploy): Mistral=Mari, Qwen=Rei, GLM=Asuka. They ride the existing
# OpenAI-compatible generator — no new generator, just registry entries differing
# by model slug (verified in the live Fireworks library 2026-06). Inference grid
# is Fireworks-US (~400) for ALL of them — it follows the ENDPOINT, not the brand
# (Mistral-on-Fireworks is NOT French nuclear; that ~20 g win needs
# api.mistral.ai). Pricing is $/Mtok; cached input = 0.5 × input (Fireworks 50%
# cache discount). Routing: Qwen earns the auto-router as the cheap
# coder/mathematician, Mistral gets a narrow French nudge, GLM stays override-
# only — see _estimate_confidence. Vision off for safety (Claude/Gemini see).
MISTRAL_PROVIDER = ModelProvider(
    name="Mistral",
    id="mistral",
    sdk_type="openai_compatible",
    api_key_env="MISTRAL_API_KEY",
    base_url="https://api.mistral.ai/v1",
    backend="api",
    # On Mistral's OWN API (api.mistral.ai), NOT Fireworks. `mistral-large-latest`
    # now resolves to Mistral Large 3 (released 2025-12-02, 675B/41B MoE,
    # Apache-2.0) — a ~4× price drop from Large 2 ($2/$6 → $0.50/$1.50). On
    # Fireworks it's on-demand/dedicated only (NOT serverless), so the own-API
    # route is now both the green win AND the cheap one. Needs MISTRAL_API_KEY.
    model_id="mistral-large-latest",
    input_cost_per_million=0.50,         # Mistral Large 3 (mistral.ai/pricing, 2026-06) — VERIFY at console.mistral.ai
    output_cost_per_million=1.50,
    cached_input_cost_per_million=0.50,  # Mistral API cache rate varies; assume no discount (shim doesn't extract cache hits)
    max_context_tokens=128_000,
    supports_vision=False,
    supports_web_search=False,
    search_backend="tavily",
    est_wh_per_1k_tokens=0.35,  # 675B/41B MoE — light per token
    grid_gco2_per_kwh=20.0,     # France grid (~20 g) — Mistral's own EU infra, the low-carbon win
    train_tco2e=2200.0,         # Large-2 LCA (Carbone 4) carried over — Large-3's isn't published; France-grid-embodied → low-carbon train
    # Backend toggle (Phase 3). Default "api" (above) = api.mistral.ai (EU,
    # France ~nuclear grid — the low-carbon win, and Mistral Large is native).
    # "together" = US serverless (loses the green win); "self_hosted" = local
    # (Large 3 is 675B/multi-GPU — use a small Mistral). ⚠️ together slug +
    # pricing are estimates — VERIFY. code-complete but UNVERIFIED.
    backends={
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "api_key_env": "TOGETHER_API_KEY",
            "model": "mistralai/Mistral-Large-3",    # VERIFY exact Together slug
            "input_cost_per_million": 2.0,           # VERIFY Together pricing
            "output_cost_per_million": 6.0,
            "cached_input_cost_per_million": 2.0,
            "grid_gco2_per_kwh": 400.0,              # Together US fleet
            "supports_server_cache": False,
        },
        "self_hosted": {
            "base_url": "http://localhost:8000/v1",
            "api_key_env": None,
            "model": "ministral-8b",                 # single-GPU; Large 3 is multi-GPU
            "supports_server_cache": False,
            "cost_mode": "local",
            "grid_gco2_per_kwh": None,
        },
    },
)

QWEN_PROVIDER = ModelProvider(
    name="Qwen",
    id="qwen",
    sdk_type="openai_compatible",
    api_key_env="FIREWORKS_API_KEY",
    base_url="https://api.fireworks.ai/inference/v1",
    model_id="accounts/fireworks/models/qwen3p7-plus",
    input_cost_per_million=0.50,         # VERIFY on Fireworks pricing page
    output_cost_per_million=3.00,
    cached_input_cost_per_million=0.25,
    max_context_tokens=256_000,
    supports_vision=False,
    supports_web_search=False,
    search_backend="tavily",
    est_wh_per_1k_tokens=0.35,  # MoE — light per token
    grid_gco2_per_kwh=400.0,    # Fireworks US fleet (trained on Alibaba's cleaner CN fleet → embodied in train_tco2e)
    train_tco2e=1500.0,         # estimate — Alibaba targets 100% clean by 2030 (Zhangjiakou wind / Ulanqab)
)

GLM_PROVIDER = ModelProvider(
    name="GLM",
    id="glm",
    sdk_type="openai_compatible",
    api_key_env="FIREWORKS_API_KEY",
    base_url="https://api.fireworks.ai/inference/v1",
    model_id="accounts/fireworks/models/glm-5p2",
    input_cost_per_million=1.40,         # VERIFY on Fireworks pricing page
    output_cost_per_million=4.40,
    cached_input_cost_per_million=0.70,
    max_context_tokens=200_000,
    supports_vision=False,
    supports_web_search=False,
    search_backend="tavily",
    est_wh_per_1k_tokens=0.45,  # 744B — larger
    grid_gco2_per_kwh=400.0,    # Fireworks US fleet
    train_tco2e=2000.0,         # estimate — GLM-5 (744B) trains end-to-end on Huawei Ascend; CN province grid
)

# Phase 7 — simulator mode (§9). The "Dummy Plug": EVA's autopilot that runs an
# Eva with NO conscious pilot. A base model continuing a transcript with no
# instruct "pilot" steering it is exactly that — so the simulator head is a NERV
# *system*, not a MAGI unit or a pilot, and it sidesteps the trinity-cap-at-3.
# It is OFF by default (enabled=False → the registry leaves it disabled unless
# config opts in; see _configure_provider) and OVERRIDE-ONLY (completions_mode
# keeps it out of the argmax). Point it at any /completions-capable endpoint:
# the default is a self-hosted vLLM/llama.cpp base model on localhost (no key —
# the operator owns the box); to use a hosted base model instead, set
# providers.sim.{base_url,model,api_key_env} in config.json. Pricing/energy are
# placeholders flagged VERIFY — they only bite once a real endpoint is wired
# (and read $0 anyway under a "local" self-host until you set cost_mode).
SIM_PROVIDER = ModelProvider(
    name="Dummy",                 # canonical routing key + [Dummy] label (NEVER rename — CLAUDE.md)
    id="sim",
    display_name="Dummy Plug",    # friendly label shown in !models ([Dummy] stays the routing key/label)
    sdk_type="openai_compatible",
    completions_mode=True,
    enabled=False,                # opt-in: flip providers.sim.enabled=true in config.json
    api_key_env="",               # self-hosted default → no key (operator owns the server)
    base_url="http://localhost:8000/v1",  # vLLM default; override via providers.sim.base_url
    model_id="base-model",        # override via providers.sim.model (e.g. a Ministral/Magistral base)
    input_cost_per_million=0.0,   # VERIFY — local self-host bills electricity, not tokens
    output_cost_per_million=0.0,  # VERIFY
    max_context_tokens=32_000,    # conservative; bump in config if your base model is bigger
    supports_vision=False,        # base model = text-only transcript
    supports_web_search=False,
    # A base model can't tool-call, so the chat heads' web_search loop never
    # runs here. Instead the simulator can run this backend ITSELF and fold the
    # snippets into the transcript preamble (§9.2) — but only when the operator
    # opts in via sim_search (off by default). Tavily is the default backend;
    # pasted-URL grounding reaches the path regardless via the URL-extract fold.
    search_backend="tavily",
    sim_search=False,             # opt-in: providers.sim.search=true (off by default)
    est_wh_per_1k_tokens=0.4,     # VERIFY — small/mid base model
    grid_gco2_per_kwh=None,       # self-host → global GRID_GCO2_PER_KWH fallback (your wall socket)
    # Base-model sampler defaults (§9.3). Higher temp than chat — base models
    # want headroom — with a light repetition penalty to curb loops. stop cuts
    # the continuation at the next IRC speaker tag or a paragraph gap so one
    # !dummy turn = one speaker's line, not the whole imagined channel.
    sim_sampler={
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.05,   # extra_body (vLLM/Fireworks); harmless if unsupported
        "stop": ["\n<", "\n\n\n"],
    },
)

# Recognized simulator sampler keys (§9.3). STD go straight to
# client.completions.create as named params; EXTRA ride in extra_body
# (vLLM/Fireworks accept them, plain OpenAI ignores them); `stop` is
# special-cased. Anything else in a providers.<id>.sampler config block is a
# typo and is warned about at registry build (see _configure_provider). NB:
# max_tokens is NOT a sampler key — it's taken from provider.max_tokens.
SIM_SAMPLER_STD = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
SIM_SAMPLER_EXTRA = ("top_k", "min_p", "top_a", "repetition_penalty")
SIM_SAMPLER_KEYS = frozenset(SIM_SAMPLER_STD + SIM_SAMPLER_EXTRA + ("stop",))


# Flavor themes (display-only skins) live in theme.py. ISAIC is the default.
from .theme import Flavor, Theme, THEMES, get_theme, DEFAULT_THEME


# Bookclub Gemini caching knobs. Gemini context caches bill storage per
# token-hour for their WHOLE TTL whether or not they're ever read, so a big fic
# cache that sits idle is pure waste (~$11.6 for Almost Nowhere's ~452k tokens at
# 24h). Default TTL is 6h, applied as a SLIDING window: each use bumps the expiry
# back to a full TTL (_refresh_gemini_cache, once past the halfway point), so an
# ACTIVE discussion never expires mid-conversation while an IDLE cache still dies
# ~6h after its last use (~$2.9 worst case). A fully-expired cache recreates
# automatically on the next message (~$1.8 re-upload + ~30s). Caches are also
# deleted immediately on !unload/!unscope (_drop_gemini_cache). Tune without code
# edits via GEMINI_CACHE_TTL_HOURS (e.g. 2 for very sporadic clubs, 24 for a
# marathon read).
GEMINI_CACHE_TTL_HOURS = float(os.getenv("GEMINI_CACHE_TTL_HOURS", "6"))
GEMINI_CACHE_TTL_SECONDS = int(GEMINI_CACHE_TTL_HOURS * 3600)  # cache auto-expires after this
# Rough storage rate for the heads-up estimate logged at cache creation. VERIFY
# against current Gemini pricing — used only for a console warning, not billing.
GEMINI_CACHE_STORAGE_COST_PER_MTOK_HOUR = 1.0


def _utcnow() -> datetime:
    """Timezone-AWARE current UTC. ALL Gemini cache expiry/storage math uses this
    one clock. Mixing datetime.now() (naive LOCAL) with Gemini's UTC expireTime
    was the bug that made a US-Eastern box believe a cache was still alive ~4h
    after Google had already deleted it — a dead zone of full-price re-uploads."""
    return datetime.now(timezone.utc)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Coerce a datetime to aware UTC. A naive value is ASSUMED UTC (what we now
    persist). Guards against 'can't compare offset-naive and offset-aware' when
    older persisted entries (historically a naive-UTC / naive-local mix) are
    compared against _utcnow()."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _gemini_storage_cost(
    tokens: int, created: datetime, expires: Optional[datetime], now: datetime
) -> float:
    """Storage $ one Gemini context cache accrued from creation → min(now, expiry).
    Storage bills per token-hour for as long as the cache lives, so billing stops
    at the cache's expiry, not at whenever we notice/tear it down."""
    created = _as_utc(created)
    end = now
    expires = _as_utc(expires)
    if expires is not None and expires < end:
        end = expires
    hours = max(0.0, (end - created).total_seconds() / 3600.0)
    return tokens / 1_000_000 * GEMINI_CACHE_STORAGE_COST_PER_MTOK_HOUR * hours


# =============================================================================
# PROVIDER REGISTRY (Phase 0 — config-driven provider wiring)
# =============================================================================

# The 6 providers in canonical order — how !models lists them and how routing
# iterates ties. ProviderRegistry resolves each one's backend + client from this
# list plus the config overlay. `name` stays the routing key (CLAUDE.md); `id`
# is the config key.
_PROVIDER_CONSTANTS = [
    CLAUDE_PROVIDER,
    DEEPSEEK_PROVIDER,
    GEMINI_PROVIDER,
    MISTRAL_PROVIDER,
    QWEN_PROVIDER,
    GLM_PROVIDER,
    SIM_PROVIDER,   # Phase 7 simulator (Dummy Plug) — last so it sorts after the heads in !models
]

# Override-dict key → ModelProvider attribute. A selected non-default backend
# (and any config sub-block for it) may set these. Vertex-only keys
# (project_env / location_env / location_default) are deliberately absent —
# they're read by the Vertex generator from prov.backends["vertex"], not applied
# as provider fields.
_BACKEND_FIELD_MAP = {
    "base_url": "base_url",
    "api_key_env": "api_key_env",
    "model": "model_id",
    "input_cost_per_million": "input_cost_per_million",
    "output_cost_per_million": "output_cost_per_million",
    "cached_input_cost_per_million": "cached_input_cost_per_million",
    "grid_gco2_per_kwh": "grid_gco2_per_kwh",
    "supports_server_cache": "supports_server_cache",
    "cost_mode": "cost_mode",
}


class ProviderRegistry:
    """Config-driven provider wiring. Replaces the hardcoded per-provider client
    setup that used to live in ClaudeBot.__init__.

    Built once via from_config(config). For each provider constant it:
      1. applies config["providers"][id] overrides (enabled / backend / model /
         routing_tags + per-backend sub-configs),
      2. resolves the ACTIVE backend in place — the *default* backend is the
         constant's own fields, so the default config (no `providers` section)
         mutates nothing and behavior is byte-for-byte preserved; selecting a
         non-default backend merges backends[backend] over those fields,
      3. gates `enabled` on key presence (graceful degradation). Backends with
         no api_key_env (Vertex → ADC, self_hosted → local server) aren't
         key-gated; the operator owns their availability,
      4. creates the SDK client per (sdk_type, backend).

    Exposes: .providers (ordered; all of them — the `enabled` flag gates use),
    .by_id / .by_name, .clients (id → client), .openai_compatible_clients
    (name → client; back-compat lookup for the dispatch path), .platform.
    """

    def __init__(self):
        self.providers: list[ModelProvider] = []
        self.clients: dict[str, object] = {}
        self.openai_compatible_clients: dict[str, object] = {}
        self.platform: str = "slack"
        self.theme: Theme = THEMES[DEFAULT_THEME]   # cosmetic skin; overridden by config "theme"

    def by_id(self, pid: str) -> Optional[ModelProvider]:
        for p in self.providers:
            if p.id == pid:
                return p
        return None

    def by_name(self, name: str) -> Optional[ModelProvider]:
        for p in self.providers:
            if p.name == name:
                return p
        return None

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "ProviderRegistry":
        reg = cls()
        config = config or {}
        reg.platform = config.get("platform", "slack")
        if reg.platform not in ("slack",):
            print(f"⚠️  platform='{reg.platform}' is not supported by this build "
                  f"(ISAIC is Slack-only) — running as 'slack'.")
            reg.platform = "slack"

        # Cosmetic flavor theme (display-only; canonical provider.name untouched).
        theme_name = config.get("theme", DEFAULT_THEME)
        reg.theme = THEMES.get(theme_name)
        if reg.theme is None:
            print(f"⚠️  theme='{theme_name}' is unknown — using 'eva'. "
                  f"(options: {', '.join(sorted(THEMES))})")
            reg.theme = THEMES[DEFAULT_THEME]

        providers_cfg = config.get("providers", {}) or {}

        for prov in _PROVIDER_CONSTANTS:
            reg._configure_provider(prov, providers_cfg.get(prov.id, {}) or {})
            reg.providers.append(prov)

        # Apply the theme's display names onto the providers (in place, like the
        # rest of the registry). EVA flavors carry the canonical names, so the
        # default theme leaves !models/!help unchanged; ISAIC/Night Vale rename.
        for prov in reg.providers:
            flavor = reg.theme.flavors.get(prov.id)
            if flavor:
                prov.display_name = flavor.display_name
        return reg

    def _configure_provider(self, prov: ModelProvider, pcfg: dict) -> None:
        # 1. Backend-independent overrides.
        if "model" in pcfg:
            prov.model_id = pcfg["model"]
        if "routing_tags" in pcfg:
            prov.routing_tags = list(pcfg["routing_tags"])
        # Simulator mode (Phase 7) + the endpoint knobs an operator needs to
        # point the Dummy Plug at their own /completions box. These are additive
        # and only fire when present in config, so the default config (and the 6
        # instruct heads) is untouched. base_url/api_key_env are set here, BEFORE
        # the key gate + client build below, so they take effect this pass.
        if "completions_mode" in pcfg:
            prov.completions_mode = bool(pcfg["completions_mode"])
        if "base_url" in pcfg:
            prov.base_url = pcfg["base_url"]
        if "api_key_env" in pcfg:
            prov.api_key_env = pcfg["api_key_env"]
        if "search" in pcfg:
            # Opt-in query-driven search grounding for a completions_mode head
            # (§9.2). Inert on the instruct heads (they ignore sim_search).
            prov.sim_search = bool(pcfg["search"])
        if isinstance(pcfg.get("sampler"), dict):
            prov.sim_sampler = {**prov.sim_sampler, **pcfg["sampler"]}
            unknown = [k for k in pcfg["sampler"] if k not in SIM_SAMPLER_KEYS]
            if unknown:
                print(f"⚠️  {prov.name}: ignoring unknown sampler key(s) {unknown} "
                      f"(known: {sorted(SIM_SAMPLER_KEYS)}; max_tokens is set "
                      f"separately, not via sampler).")

        # 2. Resolve the active backend. config may override it; an unknown
        #    backend falls back to the default with a warning. The default
        #    backend == the constant's own fields → no mutation.
        requested = pcfg.get("backend", prov.backend)
        if requested and requested != prov.backend:
            if requested in prov.backends:
                self._apply_backend(prov, requested, pcfg)
                prov.backend = requested
            else:
                print(f"⚠️  {prov.name}: unknown backend '{requested}' — using "
                      f"default '{prov.backend}' (known: {sorted(prov.backends)}).")

        # 3. Key + enabled gate. Backends with no api_key_env (Vertex/self_hosted)
        #    aren't key-gated — the operator owns ADC / the local server.
        key = os.getenv(prov.api_key_env) if prov.api_key_env else None
        # Default to the CONSTANT's own `enabled` so a provider shipped off
        # (SIM_PROVIDER) stays off until config opts in, while the 6 heads
        # (all enabled=True) behave exactly as before — pcfg.get("enabled",
        # True) and pcfg.get("enabled", prov.enabled) are identical for them.
        config_enabled = pcfg.get("enabled", prov.enabled)
        if not config_enabled:
            prov.enabled = False
        elif not prov.api_key_env:
            prov.enabled = True
        else:
            prov.enabled = bool(key)

        # 4. Client (only if enabled).
        client = self._make_client(prov, key) if prov.enabled else None
        self.clients[prov.id] = client
        if prov.sdk_type in ("openai_compatible", "gemini"):
            self.openai_compatible_clients[prov.name] = client

        self._print_status(prov, config_enabled)

    def _apply_backend(self, prov: ModelProvider, backend: str, pcfg: dict) -> None:
        """Merge a non-default backend's overrides onto the provider in place,
        then a per-backend config sub-block (e.g.
        providers.deepseek.self_hosted.base_url) so an operator can point
        self_hosted at their own server without editing code."""
        overrides = dict(prov.backends.get(backend, {}))
        overrides.update(pcfg.get(backend, {}) or {})
        for okey, attr in _BACKEND_FIELD_MAP.items():
            if okey not in overrides:
                continue
            val = overrides[okey]
            # api_key_env and grid_gco2_per_kwh are intentionally Nullable
            # (self_hosted has no key; None grid → global fallback). Skip an
            # explicit None elsewhere so a partial dict can't blank a default.
            if val is None and okey not in ("api_key_env", "grid_gco2_per_kwh"):
                continue
            setattr(prov, attr, val)

    def _make_client(self, prov: ModelProvider, key: Optional[str]):
        if prov.sdk_type == "anthropic":
            # max_retries=4 (default 2) absorbs more transient 5xx — matters for
            # bookclub cache-creation calls sending ~450k tokens.
            return anthropic.Anthropic(api_key=key, max_retries=4)
        if prov.sdk_type == "gemini" and prov.backend == "vertex":
            # Vertex talks via the vertexai SDK (ADC), not an OpenAI client —
            # _generate_gemini_vertex_response lazy-inits it. Nothing to build.
            return None
        if prov.sdk_type in ("openai_compatible", "gemini"):
            from openai import OpenAI
            # self_hosted / local endpoints carry no key; the OpenAI SDK rejects
            # an empty string, so pass a harmless placeholder.
            return OpenAI(api_key=key or "local", base_url=prov.base_url)
        return None

    def _print_status(self, prov: ModelProvider, config_enabled: bool) -> None:
        if prov.enabled:
            extra = ""
            if prov.backend and prov.backend not in ("api", "developer_api"):
                extra += f"; backend={prov.backend}"
            if prov.cost_mode == "local":
                extra += "; local (no per-token $)"
            if prov.completions_mode:
                extra += f"; simulator/base @ {prov.base_url}"
            print(f"🟢 {prov.name} enabled (model: {prov.model_id}{extra})")
        elif prov.completions_mode and not config_enabled:
            # Off-by-default simulator head, not a missing key — tell the
            # operator how to wake it up.
            print(f"⚪ {prov.name} off (Phase 7 simulator — enable via "
                  f"providers.{prov.id}.enabled=true + base_url)")
        elif not config_enabled:
            print(f"⚪ {prov.name} disabled in config")
        else:
            print(f"⚪ {prov.name} not configured ({prov.api_key_env} missing)")


# =============================================================================
# CALIBRATION TRACKER
# =============================================================================

@dataclass
class CalibrationRecord:
    """A single confidence bid record for calibration tracking."""
    model_name: str
    confidence: float
    timestamp: datetime
    user_reaction: Optional[str] = None  # "good" / "bad" / None


class CalibrationTracker:
    """Tracks model confidence calibration over time."""

    def __init__(self, max_records: int = 200):
        self.records: list[CalibrationRecord] = []
        self.max_records = max_records

    def record_bid(self, model_name: str, confidence: float) -> int:
        """Record a confidence bid. Returns the record index for later feedback."""
        record = CalibrationRecord(
            model_name=model_name,
            confidence=confidence,
            timestamp=datetime.now()
        )
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records.pop(0)
        return len(self.records) - 1

    def record_feedback(self, index: int, reaction: str) -> None:
        """Record user feedback on a response."""
        if 0 <= index < len(self.records):
            self.records[index].user_reaction = reaction

    def get_calibration_summary(self, model_name: str) -> dict:
        """Get calibration stats for a model by confidence bucket."""
        model_records = [r for r in self.records if r.model_name == model_name]
        rated = [r for r in model_records if r.user_reaction is not None]

        if not rated:
            return {"total": len(model_records), "rated": 0, "buckets": {}}

        buckets = {"high (0.7-1.0)": [], "medium (0.4-0.7)": [], "low (0.0-0.4)": []}
        for r in rated:
            if r.confidence >= 0.7:
                buckets["high (0.7-1.0)"].append(r.user_reaction == "good")
            elif r.confidence >= 0.4:
                buckets["medium (0.4-0.7)"].append(r.user_reaction == "good")
            else:
                buckets["low (0.0-0.4)"].append(r.user_reaction == "good")

        summary = {}
        for bucket_name, results in buckets.items():
            if results:
                summary[bucket_name] = {
                    "count": len(results),
                    "success_rate": sum(results) / len(results)
                }

        return {"total": len(model_records), "rated": len(rated), "buckets": summary}

    def to_dict(self) -> list:
        return [
            {
                "model": r.model_name,
                "confidence": r.confidence,
                "timestamp": r.timestamp.isoformat(),
                "feedback": r.user_reaction
            }
            for r in self.records
        ]

    @classmethod
    def from_dict(cls, data: list, max_records: int = 200) -> "CalibrationTracker":
        tracker = cls(max_records=max_records)
        for item in data:
            record = CalibrationRecord(
                model_name=item["model"],
                confidence=item["confidence"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                user_reaction=item.get("feedback")
            )
            tracker.records.append(record)
        return tracker


# =============================================================================
# MEMORY SYSTEM (Two-tier: Working + Long-term)
# =============================================================================

@dataclass
class WorkingNote:
    """A note in working memory. Decays if not accessed."""
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    
    def is_expired(self, decay_hours: float = 48.0) -> bool:
        """Check if note has decayed."""
        age_hours = (datetime.now() - self.last_accessed).total_seconds() / 3600
        # Notes accessed more get longer life
        effective_decay = decay_hours * (1 + (self.access_count * 0.5))
        return age_hours > effective_decay
    
    def touch(self) -> None:
        """Mark as accessed, resetting decay timer."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def freshness(self, decay_hours: float = 48.0) -> float:
        """0.0 = about to expire, 1.0 = fresh"""
        age_hours = (datetime.now() - self.last_accessed).total_seconds() / 3600
        effective_decay = decay_hours * (1 + (self.access_count * 0.5))
        return max(0, 1 - (age_hours / effective_decay))


class WorkingMemory:
    """
    Claude's "scratch space" - things it notices and jots down.
    
    - Auto-populated by Claude during conversations
    - Decays after ~48h of no access
    - Frequently referenced notes live longer
    - Can be promoted to long-term with !keep
    - Capped at max_notes to prevent bloat
    """
    
    def __init__(self, max_notes: int = 10, decay_hours: float = 48.0):
        self.notes: dict[str, WorkingNote] = {}
        self.max_notes = max_notes
        self.decay_hours = decay_hours
    
    def add(self, key: str, content: str) -> None:
        """Add or update a working note."""
        self._prune_expired()
        
        if key in self.notes:
            self.notes[key].content = content
            self.notes[key].touch()
        else:
            # If at capacity, remove stalest note
            if len(self.notes) >= self.max_notes:
                self._evict_stalest()
            self.notes[key] = WorkingNote(content=content)
    
    def get(self, key: str) -> Optional[str]:
        """Get a note, refreshing its decay timer."""
        if key in self.notes:
            if not self.notes[key].is_expired(self.decay_hours):
                self.notes[key].touch()
                return self.notes[key].content
            else:
                del self.notes[key]
        return None
    
    def remove(self, key: str) -> Optional[WorkingNote]:
        """Remove and return a note (for promotion to long-term)."""
        return self.notes.pop(key, None)
    
    def _prune_expired(self) -> None:
        """Remove all expired notes."""
        expired = [k for k, v in self.notes.items() if v.is_expired(self.decay_hours)]
        for k in expired:
            del self.notes[k]
    
    def _evict_stalest(self) -> None:
        """Remove the note closest to expiring."""
        if not self.notes:
            return
        stalest = min(self.notes.keys(), 
                     key=lambda k: self.notes[k].freshness(self.decay_hours))
        del self.notes[stalest]
    
    def get_context_string(self) -> str:
        """Get working notes formatted for LLM context."""
        self._prune_expired()
        if not self.notes:
            return ""
        
        lines = ["**Working notes** (recent observations, may fade):"]
        for key, note in sorted(self.notes.items(), 
                                key=lambda x: x[1].freshness(self.decay_hours),
                                reverse=True):
            freshness = note.freshness(self.decay_hours)
            fade_indicator = "●" if freshness > 0.7 else "◐" if freshness > 0.3 else "○"
            lines.append(f"- {fade_indicator} {key}: {note.content}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        self._prune_expired()
        return {
            key: {
                "content": note.content,
                "created_at": note.created_at.isoformat(),
                "last_accessed": note.last_accessed.isoformat(),
                "access_count": note.access_count
            }
            for key, note in self.notes.items()
        }
    
    @classmethod
    def from_dict(cls, data: dict, max_notes: int = 10, decay_hours: float = 48.0) -> "WorkingMemory":
        memory = cls(max_notes=max_notes, decay_hours=decay_hours)
        for key, note_data in data.items():
            note = WorkingNote(
                content=note_data["content"],
                created_at=datetime.fromisoformat(note_data["created_at"]),
                last_accessed=datetime.fromisoformat(note_data["last_accessed"]),
                access_count=note_data["access_count"]
            )
            if not note.is_expired(decay_hours):
                memory.notes[key] = note
        return memory


class LongTermMemory:
    """
    Explicit facts that persist forever until forgotten.
    
    - User-controlled via !remember / !forget
    - Can be populated by promoting working notes with !keep
    - Never decays
    - Hard cap to prevent unbounded growth
    """
    
    def __init__(self, max_entries: int = 25):
        self.entries: dict[str, str] = {}
        self.max_entries = max_entries
    
    def add(self, key: str, value: str) -> bool:
        """Add or update a memory. Returns False if at capacity and key is new."""
        if key in self.entries:
            self.entries[key] = value
            return True
        
        if len(self.entries) >= self.max_entries:
            return False
        
        self.entries[key] = value
        return True
    
    def get(self, key: str) -> Optional[str]:
        return self.entries.get(key)
    
    def remove(self, key: str) -> bool:
        if key in self.entries:
            del self.entries[key]
            return True
        return False
    
    def get_context_string(self) -> str:
        """Get long-term memories formatted for LLM context."""
        if not self.entries:
            return ""
        
        lines = ["**Long-term memories** (permanent facts):"]
        for key, value in self.entries.items():
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return dict(self.entries)
    
    @classmethod
    def from_dict(cls, data: dict, max_entries: int = 25) -> "LongTermMemory":
        memory = cls(max_entries=max_entries)
        memory.entries = dict(data)
        return memory


class TwoTierMemory:
    """
    Combined memory system with working + long-term storage.
    
    Like actual brains:
    - Working memory: Things Claude notices, fade over ~48h
    - Long-term memory: Explicit facts, permanent until forgotten
    
    Notes can be promoted from working → long-term with !keep
    """
    
    def __init__(
        self, 
        max_working_notes: int = 10,
        max_longterm_entries: int = 25,
        working_decay_hours: float = 48.0
    ):
        self.working = WorkingMemory(max_working_notes, working_decay_hours)
        self.longterm = LongTermMemory(max_longterm_entries)
    
    def promote(self, key: str) -> bool:
        """
        Promote a working note to long-term memory.
        Returns False if note doesn't exist or long-term is full.
        """
        note = self.working.notes.get(key)
        if not note:
            return False
        
        if self.longterm.add(key, note.content):
            self.working.remove(key)
            return True
        return False
    
    def get_context_string(self) -> str:
        """Get combined memory context for LLM."""
        parts = []
        
        lt_context = self.longterm.get_context_string()
        if lt_context:
            parts.append(lt_context)
        
        wm_context = self.working.get_context_string()
        if wm_context:
            parts.append(wm_context)
        
        return "\n\n".join(parts)
    
    def to_dict(self) -> dict:
        return {
            "working": self.working.to_dict(),
            "longterm": self.longterm.to_dict()
        }
    
    @classmethod
    def from_dict(
        cls, 
        data: dict,
        max_working_notes: int = 10,
        max_longterm_entries: int = 25,
        working_decay_hours: float = 48.0
    ) -> "TwoTierMemory":
        memory = cls(max_working_notes, max_longterm_entries, working_decay_hours)
        if "working" in data:
            memory.working = WorkingMemory.from_dict(
                data["working"], max_working_notes, working_decay_hours
            )
        if "longterm" in data:
            memory.longterm = LongTermMemory.from_dict(
                data["longterm"], max_longterm_entries
            )
        return memory

# =============================================================================
# READING MATERIAL (Bookclub mode — pinned long text per channel)
# =============================================================================

@dataclass
class ReadingMaterial:
    """A long text resource pinned to a Discord channel for bookclub mode.

    Loaded once via !load <url>, then injected into every model call's system
    context for that channel until !unload. Designed for AO3 fics and similar
    long-form works (~50k–500k tokens).

    The gemini_cache_name field stores the explicit-cache handle from
    /v1beta/cachedContents — Gemini's OpenAI shim doesn't support implicit
    caching, so for cost control we create an explicit cache once and
    reference it via extra_body={"cached_content": ...} on each shim call.
    """
    url: str
    title: str
    text: str
    chapter_breaks: list[tuple[int, str]] = field(default_factory=list)
    # ^ list of (char_offset, chapter_name) tuples for navigation/preview
    loaded_at: datetime = field(default_factory=datetime.now)
    # Provider-specific cache handles. Keyed by provider name. Currently only
    # Gemini populates this (via _create_gemini_cache). Claude uses
    # cache_control on the system block, which doesn't need a stored handle.
    cache_handles: dict[str, str] = field(default_factory=dict)
    cache_expires_at: dict[str, datetime] = field(default_factory=dict)
    # When each cache handle was created (aware UTC), keyed like cache_handles.
    # Drives real-lifetime storage metering in !cost instead of a creation-time
    # one-TTL guess. See _gemini_storage_cost / _settle_gemini_storage.
    cache_created_at: dict[str, datetime] = field(default_factory=dict)
    # Bookclub recaps: per-chapter "previously on" summaries (1-indexed chapter
    # number → text), generated once by a cheap model and cached here. Injected
    # into scoped threads so a context-limited model that only sees a LATER
    # chapter still knows what came before. recap_text is the assembled prefix
    # for a *scoped* slice (the chapters before its start); None on the full work.
    chapter_recaps: dict[int, str] = field(default_factory=dict)
    recap_text: Optional[str] = None

    @property
    def estimated_tokens(self) -> int:
        """Rough token count using BotConfig's chars_per_token estimate."""
        return int(len(self.text) / CONFIG.chars_per_token)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "chapter_breaks": self.chapter_breaks,
            "loaded_at": self.loaded_at.isoformat(),
            "cache_handles": dict(self.cache_handles),
            "cache_expires_at": {
                k: v.isoformat() for k, v in self.cache_expires_at.items()
            },
            "cache_created_at": {
                k: v.isoformat() for k, v in self.cache_created_at.items()
            },
            "chapter_recaps": {str(k): v for k, v in self.chapter_recaps.items()},
            "recap_text": self.recap_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReadingMaterial":
        return cls(
            url=data["url"],
            title=data.get("title", ""),
            text=data["text"],
            chapter_breaks=[tuple(b) for b in data.get("chapter_breaks", [])],
            loaded_at=datetime.fromisoformat(
                data.get("loaded_at", datetime.now().isoformat())
            ),
            cache_handles=dict(data.get("cache_handles", {})),
            cache_expires_at={
                k: _as_utc(datetime.fromisoformat(v))
                for k, v in data.get("cache_expires_at", {}).items()
            },
            cache_created_at={
                k: _as_utc(datetime.fromisoformat(v))
                for k, v in data.get("cache_created_at", {}).items()
            },
            chapter_recaps={
                int(k): v for k, v in data.get("chapter_recaps", {}).items()
            },
            recap_text=data.get("recap_text"),
        )


# =============================================================================
# CONVERSATION MANAGER (Uses Discord as message store)
# =============================================================================

class ConversationManager:
    """
    Uses Discord's message history as the source of truth.
    No redundant message storage - we fetch on each request.
    """
    
    def __init__(self):
        # The chat platform (Slack), injected by IsaicBot.attach_platform; used
        # for history fetches + authenticated file downloads.
        self.platform: Optional[ChatPlatform] = None
        # mem_key ("slack:{team}:{channel}") -> TwoTierMemory
        self.memories: dict[str, TwoTierMemory] = defaultdict(
            lambda: TwoTierMemory(
                max_working_notes=CONFIG.max_working_notes,
                max_longterm_entries=CONFIG.max_longterm_memories,
                working_decay_hours=CONFIG.working_memory_decay_hours
            )
        )
        # Calibration tracking for model selection
        self.calibration = CalibrationTracker()
        # Track last response per channel for feedback (keyed by channel id str)
        self.last_response_model: dict[str, str] = {}
        self.last_response_index: dict[str, int] = {}
        # channel_id -> ReadingMaterial (bookclub mode). Per-channel rather
        # than per-guild because different channels may read different works.
        self.reading_materials: dict[str, ReadingMaterial] = {}
    
    async def fetch_thread_index(self, channel, max_threads: int = 5) -> str:
        """Other-threads awareness. Slack has no cheap cross-thread listing, so
        this degrades to empty for now (TODO: enumerate via conversations.list +
        per-channel thread roots if desired)."""
        return ""
    
    async def fetch_thread_history(
        self, 
        channel: discord.abc.Messageable, 
        limit: int = CONFIG.max_messages_to_fetch
    ) -> list[dict]:
        """
        Fetch recent messages from Discord and format for Anthropic API.
        Handles text + image attachments.
        """
        messages = []
        
        async for msg in channel.history(limit=limit):
            # Skip other real bots, but include ourselves AND webhook proxies
            # (PluralKit etc. — those are user messages wearing a different face).
            if _is_real_bot(msg) and msg.author.id != channel._state.user.id:
                continue
            
            # Build content (can be text + images)
            content = []

            # Include replied-to message context if this is a reply
            if msg.reference and msg.reference.message_id:
                try:
                    ref_msg = msg.reference.resolved
                    if ref_msg is None:
                        ref_msg = await channel.fetch_message(msg.reference.message_id)
                    if ref_msg and ref_msg.content:
                        ref_text = ref_msg.content
                        # Strip model labels from referenced bot messages too
                        # (skip webhook proxies — those carry user content, not our labels)
                        if _is_real_bot(ref_msg):
                            ref_text = re.sub(rf'^((?:\*\*|\*)?\[(?:{MODEL_LABEL_NAMES})\](?:\*\*|\*)?\s*)+', '', ref_text)
                        ref_author = "bot" if _is_real_bot(ref_msg) else ref_msg.author.display_name
                        content.append({
                            "type": "text",
                            "text": f"[replying to {ref_author}: {ref_text}]"
                        })
                except (discord.NotFound, discord.HTTPException):
                    pass  # Referenced message deleted or inaccessible

            # Add text if present
            if msg.content:
                # Webhook proxies (PluralKit) are user messages, so prefix with
                # the alter's display name like any other user.
                author_prefix = "" if _is_real_bot(msg) else f"{msg.author.display_name}: "
                text = msg.content
                # Normalize model labels: strip ALL label formats (bold and plain),
                # then re-add a single clean plain-text label for identity.
                # This prevents accumulation from either format.
                if _is_real_bot(msg):
                    # First, extract which model this is from (check bold first, then plain)
                    model_label = None
                    label_match = re.match(rf'^(?:\*\*\[({MODEL_LABEL_NAMES})\]\*\*\s*|\*\[({MODEL_LABEL_NAMES})\]\*\s*|\[({MODEL_LABEL_NAMES})\]\s*)+', text)
                    if label_match:
                        # Get the model name captured (bold ** / mrkdwn * / plain)
                        model_label = label_match.group(1) or label_match.group(2) or label_match.group(3)
                        text = text[label_match.end():]
                    # Re-add a single clean label
                    if model_label:
                        text = f"[{model_label}] {text}"
                content.append({
                    "type": "text",
                    "text": f"{author_prefix}{text}"
                })
            
            # Add images if present
            for attachment in msg.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in CONFIG.image_types):
                    if attachment.size <= CONFIG.max_image_size_mb * 1024 * 1024:
                        try:
                            image_data = await self._fetch_image_base64(attachment.url)
                            if image_data:
                                # Detect media type
                                ext = attachment.filename.lower().split('.')[-1]
                                media_type = {
                                    'png': 'image/png',
                                    'jpg': 'image/jpeg', 
                                    'jpeg': 'image/jpeg',
                                    'gif': 'image/gif',
                                    'webp': 'image/webp'
                                }.get(ext, 'image/png')
                                
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data
                                    }
                                })
                        except Exception as e:
                            content.append({
                                "type": "text", 
                                "text": f"[Image attachment: {attachment.filename} - failed to load]"
                            })
                
                # Handle text files
                elif any(attachment.filename.lower().endswith(ext) for ext in CONFIG.text_file_types):
                    if attachment.size <= 1024 * 1024:  # 1MB limit for text files
                        try:
                            file_content = await self._fetch_text_file(attachment.url)
                            if file_content:
                                content.append({
                                    "type": "text",
                                    "text": f"\n--- File: {attachment.filename} ---\n{file_content}\n--- End of {attachment.filename} ---\n"
                                })
                        except Exception as e:
                            content.append({
                                "type": "text",
                                "text": f"[Text file: {attachment.filename} - failed to load: {e}]"
                            })
            
            if content:
                # Webhook proxies count as "user" — the human is upstream of the alter.
                role = "assistant" if _is_real_bot(msg) else "user"

                # Simplify if just text. _msg_id is internal-only metadata used
                # by the thinking-mode reasoning cache; strip before API calls.
                if len(content) == 1 and content[0]["type"] == "text":
                    messages.append({"role": role, "content": content[0]["text"], "_msg_id": msg.id})
                else:
                    messages.append({"role": role, "content": content, "_msg_id": msg.id})

        # Reverse so oldest first (Discord returns newest first)
        messages.reverse()
        
        # Ensure conversation starts with user message (API requirement)
        while messages and messages[0]["role"] == "assistant":
            messages.pop(0)
        
        return messages
    
    async def _fetch_image_base64(self, url: str) -> Optional[str]:
        """Fetch an image (authenticated for Slack private URLs) → base64."""
        headers = self.platform.auth_headers() if self.platform and hasattr(self.platform, "auth_headers") else {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                return base64.b64encode(data).decode("utf-8")
    
    async def _fetch_text_file(self, url: str) -> Optional[str]:
        """Fetch a text file (authenticated for Slack private URLs)."""
        headers = self.platform.auth_headers() if self.platform and hasattr(self.platform, "auth_headers") else {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return None
                raw = await resp.read()
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1", errors="replace")
    
    def estimate_tokens(self, messages: list[dict], guild_id: int, channel_id: Optional[int] = None) -> int:
        """Estimate context size in tokens (includes loaded reading material if any)."""
        total_chars = len(CONFIG.system_prompt)

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
                    elif part.get("type") == "image":
                        total_chars += 1000  # Rough estimate for image tokens

        memory_str = self.memories[guild_id].get_context_string()
        total_chars += len(memory_str)

        # Reading material adds substantial weight (often the dominant term).
        if channel_id is not None and channel_id in self.reading_materials:
            total_chars += len(self.reading_materials[channel_id].text)

        return int(total_chars / CONFIG.chars_per_token)

    def get_context_info(self, messages: list[dict], guild_id: int, channel_id: Optional[int] = None) -> str:
        """Get human-readable context info."""
        msg_count = len(messages)
        memory = self.memories[guild_id]
        working_count = len(memory.working.notes)
        longterm_count = len(memory.longterm.entries)
        est_tokens = self.estimate_tokens(messages, guild_id, channel_id=channel_id)
        # Use Claude pricing as worst-case estimate
        est_cost = (est_tokens / 1_000_000) * CLAUDE_PROVIDER.input_cost_per_million

        material_note = ""
        if channel_id is not None and channel_id in self.reading_materials:
            mat = self.reading_materials[channel_id]
            material_note = f", 📚 {mat.title} loaded (~{mat.estimated_tokens:,} tokens)"

        return (
            f"📊 Context: {msg_count} messages, "
            f"{working_count}/{CONFIG.max_working_notes} working notes, "
            f"{longterm_count}/{CONFIG.max_longterm_memories} long-term memories, "
            f"~{est_tokens:,} tokens (~${est_cost:.3f} worst-case){material_note}"
        )
    
    def get_cost_summary(self, providers: list[ModelProvider]) -> str:
        """Get total cost summary across all models, including cache hit rate."""
        lines = ["💰 **Cost Summary**"]
        grand_total = 0.0
        storage_total = 0.0
        energy_wh_total = 0.0
        co2_g_total = 0.0
        train_co2_g_total = 0.0

        # Live (in-flight) Gemini cache storage. total_cache_storage_cost_est
        # holds only SETTLED storage from caches already torn down; add what
        # currently-alive caches have accrued so far (created_at → now, capped at
        # expiry) so the figure tracks the real, growing bill instead of the
        # stale creation-time guess that used to read cheap while the balance bled.
        now = _utcnow()
        gemini_live_storage = 0.0
        for mat in self.reading_materials.values():
            for k, created in mat.cache_created_at.items():
                if k.startswith("Gemini"):
                    gemini_live_storage += _gemini_storage_cost(
                        mat.estimated_tokens, created,
                        mat.cache_expires_at.get(k), now,
                    )

        for p in providers:
            storage_est = p.total_cache_storage_cost_est
            if p.name == "Gemini":
                storage_est += gemini_live_storage
            if p.total_requests == 0 and storage_est == 0.0:
                continue
            cost = p.get_cost()
            grand_total += cost
            storage_total += storage_est
            energy_wh = p.get_energy_wh()
            energy_wh_total += energy_wh
            co2_g = p.get_co2_g(GRID_GCO2_PER_KWH)
            co2_g_total += co2_g
            train_co2_g_total += p.get_amortized_train_co2_g(MODEL_LIFETIME_TOKENS)

            # Sum uncached + cached across both pricing tiers
            uncached_in = p.total_input_tokens + p.total_input_tokens_above_tier
            cached_in = p.total_cached_input_tokens + p.total_cached_input_tokens_above_tier
            out_tokens = p.total_output_tokens + p.total_output_tokens_above_tier
            total_in = uncached_in + cached_in

            # Hit rate on input tokens (% served from cache)
            hit_rate = (cached_in / total_in * 100) if total_in > 0 else 0.0
            cache_note = ""
            if cached_in > 0:
                cache_note = f", 🟢 {cached_in:,} cached ({hit_rate:.0f}% hit rate)"

            # Self-hosted / local backends bill electricity, not per token — show
            # that instead of a $0.0000 that reads like "free". The 🌱 energy
            # line below still reports (tokens are counted) on the self-host grid.
            cost_str = ("local (no per-token $ — electricity only)"
                        if p.cost_mode == "local" else f"${cost:.4f}")
            lines.append(
                f"  **{p.name}**: {p.total_requests} requests, "
                f"{uncached_in:,} in + {out_tokens:,} out{cache_note} = "
                f"{cost_str}"
            )
            if storage_est > 0:
                live_note = (
                    f", ${gemini_live_storage:.4f} still accruing on a live cache"
                    if p.name == "Gemini" and gemini_live_storage > 0 else ""
                )
                lines.append(
                    f"      ⏳ + ~${storage_est:.4f} Gemini cache storage "
                    f"(time-based, per token-hour{live_note})"
                )
            if energy_wh > 0:
                grid = p.grid_gco2_per_kwh if p.grid_gco2_per_kwh is not None else GRID_GCO2_PER_KWH
                lines.append(
                    f"      🌱 ≈ {energy_wh:.1f} Wh ≈ {co2_g:.0f} g CO₂e "
                    f"(grid ≈ {grid:.0f} g/kWh)"
                )

        # energy_wh_total guards the local/self-hosted case: those bill $0 per
        # token but still record tokens, so a non-zero energy total means real
        # traffic happened even when grand_total is $0.
        if grand_total == 0 and storage_total == 0 and energy_wh_total == 0:
            return "💰 No API calls made yet."

        if storage_total > 0:
            lines.append(
                f"\n  **Total**: ${grand_total:.4f} tokens "
                f"+ ~${storage_total:.4f} cache storage (est) "
                f"= **~${grand_total + storage_total:.4f}**"
            )
        else:
            lines.append(f"\n  **Total**: ${grand_total:.4f}")
        if energy_wh_total > 0:
            lines.append(
                f"  **Energy (est)**: ≈ {energy_wh_total:.1f} Wh ≈ {co2_g_total:.0f} g CO₂e — "
                f"order-of-magnitude only; per-provider grid intensity, non-reasoning baseline "
                f"(tune Wh/grid in code/.env)."
            )
        if AMORTIZE_TRAINING and train_co2_g_total > 0:
            lines.append(
                f"  **+ amortized training (rough)**: ≈ {train_co2_g_total:.0f} g CO₂e — "
                f"embodied training carbon over a ~{MODEL_LIFETIME_TOKENS / 1e12:.0f}T-token "
                f"lifespan (squishiest estimate here; AMORTIZE_TRAINING=0 to hide)."
            )
        return "\n".join(lines)
    
    def save_memories(self, filepath: str = "memories.json", providers: list[ModelProvider] = None) -> None:
        """Save all memories to disk (synchronous - use save_memories_async in async contexts)."""
        data = {
            str(guild_id): memory.to_dict()
            for guild_id, memory in self.memories.items()
        }
        data["_calibration"] = self.calibration.to_dict()
        if providers:
            data["_model_stats"] = {
                p.name: p.to_stats_dict() for p in providers
            }
        # Reading materials (bookclub mode). Stored under a metadata key so
        # we don't collide with guild_ids. Persisted because a fic loaded
        # via !load should survive a bot restart.
        if self.reading_materials:
            data["_reading_materials"] = {
                str(channel_id): material.to_dict()
                for channel_id, material in self.reading_materials.items()
            }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self._memories_dirty = False
    
    async def save_memories_async(self, filepath: str = "memories.json", providers: list[ModelProvider] = None) -> None:
        """Save memories without blocking the event loop."""
        await asyncio.to_thread(self.save_memories, filepath, providers)
    
    def mark_dirty(self) -> None:
        """Mark memories as needing to be saved."""
        self._memories_dirty = True
    
    @property
    def needs_save(self) -> bool:
        """Check if memories need saving."""
        return getattr(self, '_memories_dirty', False)
    
    def load_memories(self, filepath: str = "memories.json", providers: list[ModelProvider] = None) -> None:
        """Load memories from disk."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            guild_count = 0
            for key, value in data.items():
                if key.startswith("_"):
                    continue  # Skip metadata keys
                self.memories[key] = TwoTierMemory.from_dict(
                    value,
                    max_working_notes=CONFIG.max_working_notes,
                    max_longterm_entries=CONFIG.max_longterm_memories,
                    working_decay_hours=CONFIG.working_memory_decay_hours
                )
                guild_count += 1

            # Load calibration data
            if "_calibration" in data:
                self.calibration = CalibrationTracker.from_dict(data["_calibration"])

            # Load model stats
            if "_model_stats" in data and providers:
                for p in providers:
                    if p.name in data["_model_stats"]:
                        p.load_stats(data["_model_stats"][p.name])

            # Load reading materials (bookclub mode)
            material_count = 0
            if "_reading_materials" in data:
                for ch_id_str, material_data in data["_reading_materials"].items():
                    try:
                        self.reading_materials[ch_id_str] = ReadingMaterial.from_dict(material_data)
                        material_count += 1
                    except (KeyError, ValueError) as e:
                        print(f"⚠️  Skipping malformed reading material for channel {ch_id_str}: {e}")

            print(f"Loaded memories for {guild_count} guilds" +
                  (f", {material_count} reading material(s)" if material_count else ""))
        except FileNotFoundError:
            print("No existing memories file, starting fresh")

# =============================================================================
# THE BOT
# =============================================================================

class IsaicBot:
    def __init__(self):
        # The chat platform (Slack) is attached after construction via
        # attach_platform(); until then the bot is fully built but offline. This
        # lets the object graph be smoke-tested without any Slack tokens.
        self.platform: Optional[ChatPlatform] = None
        self.user = None  # set lazily from the platform's bot_user_id once attached

        # --- Providers: config-driven registry (Phase 0) --------------------
        # Read config.json up front so the registry can apply its `providers`
        # overlay + `platform` toggle, then wire every provider's client +
        # backend from it. _load_config() (end of __init__) reuses this parsed
        # dict for allowed_channels / default_model / channel_preferences.
        # With no `providers` section (the default config) this reproduces the
        # old hardcoded setup exactly: each provider enabled iff its key is
        # present, on its default backend.
        self._raw_config = self._read_config_file()
        self.registry = ProviderRegistry.from_config(self._raw_config)
        self.platform_name = self.registry.platform   # "slack"
        self.theme = self.registry.theme   # cosmetic flavor skin (isaic | eva | nightvale)

        # Build the alias → flag map consumed by _peel_prefixes: the canonical
        # bare prefixes (always on) UNION the active theme's flavor aliases. Flag
        # values are provider.ids ("claude" …) plus "think"; nothing here touches
        # the canonical routing key or [label].
        self.alias_to_flag: dict[str, str] = {p: "think" for p in self.THINK_PREFIXES}
        for pid, prefixes in self.CANONICAL_PREFIXES.items():
            for p in prefixes:
                self.alias_to_flag[p] = pid
        for pid, flavor in self.theme.flavors.items():
            for p in flavor.aliases:
                self.alias_to_flag[p] = pid

        # Canonical providers + per-provider convenience handles. The rest of
        # this file references these by name; identity is preserved because the
        # registry resolves the module-level constants in place.
        self.providers = self.registry.providers
        self.claude_provider = self.registry.by_id("claude")
        self.deepseek_provider = self.registry.by_id("deepseek")
        self.gemini_provider = self.registry.by_id("gemini")
        self.mistral_provider = self.registry.by_id("mistral")
        self.qwen_provider = self.registry.by_id("qwen")
        self.glm_provider = self.registry.by_id("glm")
        self.sim_provider = self.registry.by_id("sim")   # Phase 7 simulator (Dummy Plug)

        # Bookclub Gemini caching mode. Default: INLINE (False). A loaded book is
        # inlined into Gemini's context each turn and Google's IMPLICIT caching
        # supplies the read discount with NO per-hour storage bill — no
        # cachedContents to create, refresh, leak, or !uncache (the whole class of
        # cost bugs this replaced). Flip to EXPLICIT via config
        # (providers.gemini.explicit_cache: true) or the admin !explicitcache
        # command: an explicit cachedContents cache gives a *guaranteed* read
        # discount, worth it ONLY for a sustained back-to-back marathon — it bills
        # ~$0.45/hr storage for a 452k book the whole time it's alive (metered live
        # in !cost), so turn it back off (→ inline, drops the cache) when done.
        _pcfg = self._raw_config.get("providers", {})
        _gcfg = _pcfg.get("gemini", {}) if isinstance(_pcfg, dict) else {}
        self.gemini_explicit_cache = bool(
            _gcfg.get("explicit_cache", False) if isinstance(_gcfg, dict) else False
        )
        print(
            "🟡 Gemini bookclub: EXPLICIT cache mode ON (per-hour storage billing — "
            "marathon mode; !explicitcache off to return to inline)"
            if self.gemini_explicit_cache
            else "🟢 Gemini bookclub: INLINE mode (implicit caching, no storage bill)"
        )

        # Clients. `clients` is keyed by provider id; `claude_client` /
        # `gemini_client` stay as named handles because their native SDK paths
        # reference them directly. `openai_compatible_clients` (name-keyed) is
        # the dispatch / search / panel lookup. Disabled providers map to None.
        self.clients = self.registry.clients
        self.openai_compatible_clients = self.registry.openai_compatible_clients
        self.claude_client = self.clients.get("claude")
        self.gemini_client = self.clients.get("gemini")

        # Tavily search client (optional - enables web search for Deepseek).
        # Gemini uses Google's native grounding (no Tavily needed); see
        # _google_native_search and GEMINI_PROVIDER.search_backend="google_native".
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            from tavily import TavilyClient
            self.tavily_client = TavilyClient(api_key=tavily_key)
            print("🟢 Tavily web search enabled")
        else:
            self.tavily_client = None
            print("⚪ Tavily not configured (Deepseek web search disabled)")

        # Azure Neural TTS (optional — powers the Mandarin !speak command). The
        # only mainstream backend that lets us FORCE exact tones via SSML
        # <phoneme>, instead of letting the synthesizer infer them — the whole
        # point for a language tutor. Free tier: 0.5M chars/month.
        self.azure_tts_key = os.getenv("AZURE_TTS_KEY")
        self.azure_tts_region = os.getenv("AZURE_TTS_REGION")
        if self.azure_tts_key and self.azure_tts_region:
            print(f"🟢 Azure Mandarin TTS enabled (region: {self.azure_tts_region})")
        else:
            print("⚪ Azure TTS not configured (AZURE_TTS_KEY / AZURE_TTS_REGION missing — !speak disabled)")

        # (self.providers + self.openai_compatible_clients are bound from the
        # registry above — canonical order preserved in _PROVIDER_CONSTANTS.)

        # Conversation manager
        self.manager = ConversationManager()

        # Per-channel model preferences (keyed by Slack channel id string).
        self.channel_preferences: dict[str, str] = {}

        # Research panel (!research): which providers form the panel and who
        # judges. Names must match ModelProvider.name. Defaults reuse models we
        # already pay for — no GPT. (Fable 5 is intentionally omitted: not
        # generally accessible yet.) Disabled members are skipped at runtime.
        self.panel_members: list[str] = ["Claude", "Gemini", "Deepseek"]
        # `!research all` convenes the full roster — the cheap Fireworks heads
        # (Mistral/Qwen/GLM) join for maximum decorrelated diversity. Different
        # training data → independent errors → a better judge synthesis: a panel
        # *rewards* the redundancy a router would punish. Filtered by `enabled`,
        # so they're silently skipped until FIREWORKS_API_KEY turns them on —
        # this list is forward-compatible today (no GPT, per the panel charter).
        self.panel_members_all: list[str] = [
            "Claude", "Gemini", "Deepseek", "Mistral", "Qwen", "GLM",
        ]
        self.panel_judge: str = "Claude"

        # Reasoning cache for thinking-mode multi-turn (keyed by sent message ts).
        # Deepseek requires reasoning_content to be echoed back on every prior
        # assistant turn whenever thinking mode is enabled on the current call.
        self.reasoning_cache: OrderedDict[str, str] = OrderedDict()

        # Allowed channels (Slack channel id strings, load from config)
        self.allowed_channels: set[str] = set()
        self._load_config()

    # --- platform attachment + lifecycle -----------------------------------
    def attach_platform(self, platform: ChatPlatform) -> None:
        """Wire a ChatPlatform (Slack) to this bot and register handlers."""
        self.platform = platform
        self.manager.platform = platform
        platform.on_message(self._on_platform_message)
        platform.on_reaction(self._on_platform_reaction)

    async def start(self) -> None:
        """Load state, then connect and run until interrupted."""
        if self.platform is None:
            raise RuntimeError("attach_platform() must be called before start()")
        await self.setup_hook()
        await self.platform.start()  # auth_test sets bot_user_id/team_id, then blocks

    @property
    def _bot_user_id(self) -> str:
        return getattr(self.platform, "bot_user_id", "") if self.platform else ""

    def _mem_key(self, channel_id: str) -> str:
        """Persistence scope: (platform, team, channel). Threads share their
        parent channel's key, so a thread sees the channel's memory."""
        team = getattr(self.platform, "team_id", "") if self.platform else ""
        return f"slack:{team}:{channel_id}"

    def get_channel(self, key: str):
        """Compat shim for the bookclub cache-cascade (!unload / !uncache). The
        original Discord bot inherited get_channel from discord.Client; here it
        returns a channel-shaped object for a reading-materials key. On Slack,
        reading materials are keyed by channel id, so these resolve to non-thread
        channels and the per-thread cascade is a no-op (scoped-thread caches fall
        back to their bounded TTL expiry). Returns None if no platform attached."""
        if self.platform is None:
            return None
        return discord.OutChannel(self.platform, key, is_thread=False)

    async def _on_platform_message(self, pm: PlatformMessage) -> None:
        """Adapter entrypoint: wrap the normalized message in the discord-shaped
        shim and run the ported handler."""
        if pm.is_self:
            return
        self.user = SimpleNamespace(id=self._bot_user_id)
        await self.on_message(discord.Message(self.platform, pm))

    async def _on_platform_reaction(self, r: PlatformReaction) -> None:
        """Adapter entrypoint for reaction add/remove → calibration."""
        if r.removed:
            return
        # Only calibrate reactions on THIS bot's own messages.
        if r.item_user_id and r.item_user_id != self._bot_user_id:
            return
        await self._handle_reaction(r)

    @property
    def multi_model_active(self) -> bool:
        """True if more than one model is enabled."""
        return sum(1 for p in self.providers if p.enabled) > 1

    REASONING_CACHE_MAX = 500
    THINK_PREFIXES = ("!think",)
    # Canonical bare command prefixes (provider.id -> prefixes) — ALWAYS active,
    # in every theme. The active theme's flavor aliases (!balthasar / !judah /
    # !gold …) are ADDED on top of these; the combined alias→provider map is
    # built once in __init__ as self.alias_to_flag and consumed by _peel_prefixes.
    # !sim stays the theme-independent plain alias for the simulator (the EVA
    # !dummy / ISAIC !levi / Night-Vale !faceless come from the theme).
    CANONICAL_PREFIXES = {
        "claude":   ("!claude", "!opus"),
        "deepseek": ("!deepseek",),
        "gemini":   ("!gemini",),
        "mistral":  ("!mistral",),
        "qwen":     ("!qwen",),
        "glm":      ("!glm",),
        "sim":      ("!sim",),
    }
    # One-line factual role blurbs for !help (theme-independent; the name + alias
    # in front of them come from the active theme).
    HELP_ROLES = {
        "claude":   "careful, thorough, vision, native web search",
        "deepseek": "fast, cheap, CJK-strong",
        "gemini":   "abstract reasoning, long-context, vision, Google grounding",
        "mistral":  "French/EU specialist (needs MISTRAL_API_KEY)",
        "qwen":     "cheap coder/mathematician (needs FIREWORKS_API_KEY)",
        "glm":      "agentic open head (needs FIREWORKS_API_KEY)",
        "sim":      "simulator mode — a base model continues the transcript (override-only; needs providers.sim)",
    }
    CLAUDE_THINKING_EFFORT = "high"  # low | medium | high | xhigh | max
    CLAUDE_THINKING_MAX_TOKENS = 16000

    def _store_reasoning(self, msg_id: int, content: str) -> None:
        """Cache reasoning_content under a Discord message id (FIFO eviction)."""
        if not content:
            return
        self.reasoning_cache[msg_id] = content
        self.reasoning_cache.move_to_end(msg_id)
        while len(self.reasoning_cache) > self.REASONING_CACHE_MAX:
            self.reasoning_cache.popitem(last=False)

    def _get_reasoning(self, msg_id: int) -> str:
        """Look up cached reasoning_content for a Discord message id, or empty string."""
        return self.reasoning_cache.get(msg_id, "")

    def _record_claude_usage(self, usage, count_request: bool = True) -> None:
        """Extract Claude usage counters into the provider, including cache info.

        Anthropic's response.usage carries input_tokens (uncached only),
        output_tokens, and two optional cache counters:
        cache_read_input_tokens (10% of input rate) and
        cache_creation_input_tokens (~125% — small surcharge that funds
        cheap reads). We bucket reads into the cached counter and lump
        creation into regular input (a small one-time under-bill).

        count_request controls whether to bump total_requests — set False
        for the tool-use loop continuation rounds so a single user turn
        counts as one request.
        """
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.claude_provider.record_usage(
            input_tokens + cache_creation,
            output_tokens,
            cached_input_tokens=cache_read,
        )
        if count_request:
            self.claude_provider.total_requests += 1

    async def _prev_bot_used_thinking(self, channel: discord.abc.Messageable) -> bool:
        """Did our most recent message in this channel use extended thinking?

        Behavioral-momentum signal for `_pick_effort`: conversations that
        had depth in the previous turn usually warrant depth in the next.
        Walks up to ~10 messages back, finds the most recent message
        authored by this bot, and checks if its reasoning was cached.
        """
        if self.user is None:
            return False
        try:
            async for msg in channel.history(limit=10):
                if _is_real_bot(msg) and msg.author.id == self.user.id:
                    return bool(self.reasoning_cache.get(msg.id))
        except (discord.HTTPException, AttributeError):
            return False
        return False

    @staticmethod
    def _pick_effort(text: str, prev_used_thinking: bool = False) -> Optional[str]:
        """Classify a prompt into an Opus 4.8 thinking-effort level.

        Returns None | "high" | "xhigh" | "max". None means thinking off
        (cheap chat path). Casual register and first-person emotional
        framing are anti-signals (not hard skips) — a hard question dressed
        in casual or emotional language can still pull the score above
        threshold with strong textual cues.
        """
        if not text:
            return None
        text_lower = text.lower()
        score = 0

        # --- Anti-signals (penalty, not veto) ---
        # Conversational opener. "lol so why does X" can still route to
        # thinking if the question itself has strong signals.
        if re.match(r"^\s*(lol|lmao|lmfao|wait what|huh|wtf|same|nice|cool|ok(ay)?|right)\b", text_lower):
            score -= 2
        # First-person emotional framing. A meaty question wrapped in feelings
        # (Lauren-style "my world used to not be full of depressed people...")
        # should still get analytical depth if the strong signals are there.
        if re.search(r"\b(i feel|i'?m feeling|i'?m (sad|depressed|anxious|scared|worried|tired|stressed|lonely))\b", text_lower):
            score -= 2

        # --- Strong signals (depth almost always rewarded) ---
        # Construction/derivation verbs, including inflections. "prov(e|es|
        # ed|ing|en)" is enumerated to avoid matching "provide"/"province".
        if re.search(r"\b(deriv\w*|prov(e|es|ed|ing|en)|design\w*|architect\w*|refactor\w*|debug\w*)\b", text_lower):
            score += 3
        # Large code blocks: model has to actually read them.
        code_blocks = re.findall(r"```[\s\S]*?```", text)
        if code_blocks and max(b.count("\n") for b in code_blocks) >= 50:
            score += 3
        # Stack traces (Python / JVM-style).
        if re.search(r"Traceback \(most recent call last\)|\sat [\w.$]+\(.*:\d+\)", text):
            score += 3
        # Compound questions: multiple "?" or explicit chaining.
        if text.count("?") >= 2 or re.search(r"\b(and also|but (what )?about|also,?\s+what about)\b", text_lower):
            score += 2
        # Math/LaTeX. {2,} avoids matching \n, \t inside pasted code.
        if re.search(r"\\[a-zA-Z]{2,}\b|=.*[+\-*/].*=", text):
            score += 2
        # Comparative / trade-off framing.
        if re.search(r"\btrade.?offs?\b|\b(when would you|what'?s the difference between|compare and contrast)\b|\bvs\.?\s+\w+", text_lower):
            score += 2

        # --- Medium signals ---
        if re.search(r"\b(why does|why is|why doesn.?t|why isn.?t|explain (why|how)|analy[sz]e)\b|\bhow does .{0,40}work\b", text_lower):
            score += 1
        if re.search(r"\b(step.by.step|walk me through|carefully|thoroughly|in.depth|from\s+(scratch|first\s+principles))\b", text_lower):
            score += 2
        if len(text) > 2000:
            score += 2

        # Behavioral momentum: prior turn used thinking → conversation has depth.
        if prev_used_thinking:
            score += 1

        if score >= 6:
            return "max"
        if score >= 3:
            return "xhigh"
        if score >= 1:
            return "high"
        return None

    VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")

    def _peel_prefixes(self, content: str) -> tuple[str, set[str], Optional[str]]:
        """Strip stacked !think / model-selection prefixes in any order.

        Returns (remaining_content, set_of_flags, forced_effort).
        - flags: subset of {'think','claude','deepseek','gemini','mistral','qwen',
          'glm','sim'}. Canonical bare prefixes (!claude/!opus/…) plus the active
          theme's flavor aliases (EVA !balthasar/!melchior/!caspar; ISAIC
          !judah/…; Night Vale !gold/…) all collapse to the provider's id flag,
          via self.alias_to_flag.
        - forced_effort: set when user used `!think:<level>` syntax (low | medium |
          high | xhigh | max). Implies the 'think' flag.
        """
        flags: set[str] = set()
        forced_effort: Optional[str] = None
        while True:
            stripped = content.strip()
            lower = stripped.lower()
            matched = False

            # Try `!think:<level>` first so it wins over the bare `!think` match.
            for think_prefix in self.THINK_PREFIXES:
                for level in self.VALID_EFFORTS:
                    tok = f"{think_prefix}:{level}"
                    if lower == tok:
                        rest = ""
                    elif lower.startswith(tok + " "):
                        rest = stripped[len(tok):].lstrip()
                    else:
                        continue
                    flags.add("think")
                    forced_effort = level
                    content = rest
                    matched = True
                    break
                if matched:
                    break
            if matched:
                continue

            # Theme-aware alias map: canonical bare prefixes + active flavor
            # aliases → flag (provider.id or "think"). Distinct command tokens, so
            # iteration order is irrelevant (matches are exact or prefix+space).
            for prefix, flag in self.alias_to_flag.items():
                if lower == prefix:
                    rest = ""
                elif lower.startswith(prefix + " "):
                    rest = stripped[len(prefix):].lstrip()
                else:
                    continue
                flags.add(flag)
                content = rest
                matched = True
                break
            if not matched:
                break
        return content, flags, forced_effort

    def _provider_aliases(self, pid: str) -> tuple:
        """All command prefixes that select a provider under the active theme:
        the canonical bare ones + the theme's flavor aliases (e.g. claude →
        ('!claude','!opus','!balthasar') under EVA, ('!claude','!opus','!gold')
        under Night Vale)."""
        flavor = self.theme.flavors.get(pid)
        return self.CANONICAL_PREFIXES.get(pid, ()) + (flavor.aliases if flavor else ())

    def _multimodel_help_lines(self) -> list[str]:
        """The themed model-selection block for !help (header + one line per
        provider). Names/aliases come from the active theme; the role blurbs are
        factual and theme-independent."""
        lines = [f"**Multi-model ({self.theme.umbrella}):**"]
        for p in self.providers:
            aliases = self._provider_aliases(p.id)
            if not aliases:
                continue
            disp = p.display_name or p.name
            alias_str = " / ".join(f"`{a}`" for a in aliases)
            role = self.HELP_ROLES.get(p.id, "")
            lines.append(f"{alias_str} `<message>` - Force {disp}" + (f" — {role}" if role else ""))
        return lines

    def _multimodel_footer_line(self) -> str:
        """The 🐉 one-liner at the bottom of !help, themed."""
        trio = [p for p in self.providers if p.id in ("claude", "deepseek", "gemini")]
        names = " + ".join((p.display_name or p.name) for p in trio)
        return (f"🐉 Multi-model: {names} (+ open-weight heads when configured) "
                f"with smart routing — skin: {self.theme.umbrella}")

    @staticmethod
    def _strip_internal_keys(messages: list[dict]) -> list[dict]:
        """Drop internal-only keys (prefixed with _) before sending to provider APIs."""
        return [{k: v for k, v in msg.items() if not k.startswith("_")} for msg in messages]

    def _read_config_file(self) -> dict:
        """Read + parse config.json once, returning the raw dict ({} if missing).
        Called early in __init__ so ProviderRegistry can apply the `providers`
        overlay + `platform`; _load_config reuses the result for the rest."""
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except FileNotFoundError:
            print("⚠️  No config.json found! Create one with {'allowed_channels': [channel_ids]}")
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Could not parse config.json ({e}) — using empty config.")
            return {}

    def _load_config(self) -> None:
        """Apply non-provider settings from the already-parsed config
        (self._raw_config): allowed channels, default model, channel prefs.
        Provider wiring + the platform toggle are handled earlier by
        ProviderRegistry (see __init__)."""
        config = getattr(self, "_raw_config", None) or {}
        # Slack channel ids are strings; keep them as-is.
        self.allowed_channels = set(str(c) for c in config.get('allowed_channels', []))
        CONFIG.default_model = config.get('default_model', 'auto')
        if config.get('server_context'):
            CONFIG.server_context = str(config['server_context'])
        for ch_id_str, model_name in config.get('channel_preferences', {}).items():
            self.channel_preferences[str(ch_id_str)] = model_name
        print(f"Loaded {len(self.allowed_channels)} allowed channels · platform={self.platform_name}")
        if CONFIG.default_model != "auto":
            print(f"   Default model: {CONFIG.default_model}")

    async def setup_hook(self) -> None:
        """Load persisted state + start the periodic-save task (called by start())."""
        self.manager.load_memories(providers=self.providers)
        # Belt-and-suspenders: sweep orphaned Gemini caches before we start
        # handling messages (race-free here — nothing is creating caches yet).
        await self._reconcile_gemini_caches()
        self._closed = False
        self._save_task = asyncio.create_task(self._periodic_save())
        models = [p.name for p in self.providers if p.enabled]
        print(f"🧠 Models: {', '.join(models)} (selection: {CONFIG.default_model}, skin: {self.theme.umbrella})")
        print(f"📋 Allowed channels: {self.allowed_channels or '(all — none configured)'}")

    async def _periodic_save(self) -> None:
        """Background task to save memories every 60 seconds if dirty."""
        while not getattr(self, "_closed", False):
            try:
                if self.manager.needs_save:
                    await self.manager.save_memories_async(providers=self.providers)
                    print("💾 Memories saved (background)")
            except Exception as e:
                print(f"⚠️  Error saving memories: {e}")
            await asyncio.sleep(60)  # Check every 60 seconds

    async def close(self) -> None:
        """Clean shutdown - save memories before closing."""
        self._closed = True
        if self.manager.needs_save:
            print("💾 Saving memories before shutdown...")
            await self.manager.save_memories_async(providers=self.providers)

    async def on_message(self, message: discord.Message) -> None:
        # Ignore our own messages (also filtered upstream via pm.is_self).
        if getattr(message, "is_self", False):
            return

        # Check if in allowed channel (threads share the parent channel id). An
        # empty allow-list means "respond anywhere the bot is invited".
        channel_id = message.channel.id
        parent_id = getattr(message.channel, 'parent_id', None)
        if self.allowed_channels and (
            channel_id not in self.allowed_channels and parent_id not in self.allowed_channels
        ):
            return

        # Peel any stacked model/thinking prefixes (!think / !think:<level> /
        # !claude / !opus / !deepseek / !gemini and the MAGI aliases)
        original_content = message.content or ""
        peeled_content, flags, forced_effort = self._peel_prefixes(original_content)
        forced_thinking = "think" in flags

        # Disabled-provider guards for explicit invocations (themed names + the
        # provider's real key env). The 6 instruct heads first, then the sim.
        for pid in ("claude", "deepseek", "gemini", "mistral", "qwen", "glm"):
            if pid in flags:
                prov = self.registry.by_id(pid)
                if prov is None or not prov.enabled:
                    disp = (prov.display_name or prov.name) if prov else pid
                    keyhint = f" (set {prov.api_key_env})" if prov and prov.api_key_env else ""
                    await message.channel.send(f"❌ {disp} isn't configured{keyhint}.")
                    return
        if "sim" in flags and (self.sim_provider is None or not self.sim_provider.enabled):
            disp = (self.sim_provider.display_name if self.sim_provider else None) or "Simulator mode"
            await message.channel.send(
                f"❌ {disp} (simulator mode) isn't enabled. Set "
                "`providers.sim.enabled = true` in config.json and point "
                "`providers.sim.base_url` at a /completions-capable base model."
            )
            return

        # Forced provider from an explicit prefix. First match wins, in canonical
        # order (so stacked prefixes resolve deterministically as before).
        forced_provider = None
        routing_reason = ""
        for pid in ("claude", "deepseek", "gemini", "mistral", "qwen", "glm", "sim"):
            if pid in flags:
                forced_provider = self.registry.by_id(pid)
                disp = forced_provider.display_name or forced_provider.name
                aliases = "/".join(self._provider_aliases(pid))
                routing_reason = f"User directly invoked {disp} with {aliases}."
                break

        if flags:
            message.content = peeled_content

        # Handle commands (but not if we just consumed a model/thinking prefix)
        if not flags and message.content.startswith('!'):
            await self._handle_command(message)
            return

        # Ignore empty messages (no text, no attachments)
        if not message.content and not message.attachments:
            return

        # Graceful guard: with no enabled heads there's nothing to route to.
        # (main.py already requires >=1 model key, so this only fires in an
        # all-keys-missing misconfiguration.)
        if not any(p.enabled for p in self.providers):
            await message.channel.send("⚠️ No models are configured — set an API key and restart.")
            return

        # Get or create thread
        thread, is_new_thread = await self._ensure_thread(message)

        # Select which model responds (forced or auto)
        if forced_provider:
            provider = forced_provider
        else:
            provider, routing_reason = await self._select_model(message, self._mem_key(message.channel.id))

        # Decide effort level. Priority: manual !think:<level> > auto-classify
        # > class default. Auto-classify only when user explicitly chose Claude
        # (forced_provider) — we don't want to silently flip auto-routed turns
        # into thinking mode.
        chosen_effort: Optional[str] = forced_effort
        if forced_effort:
            routing_reason = (routing_reason + f" User-set effort={forced_effort}.").strip()
        elif not forced_thinking and forced_provider is self.claude_provider:
            prev_thinking = await self._prev_bot_used_thinking(message.channel)
            auto_effort = self._pick_effort(message.content or "", prev_used_thinking=prev_thinking)
            if auto_effort:
                forced_thinking = True
                chosen_effort = auto_effort
                routing_reason = (routing_reason + f" Auto-enabled thinking (effort={auto_effort}).").strip()

        # Generate response
        async with thread.typing():
            response, reactions, reasoning = await self._generate_response(
                thread,
                self._mem_key(message.channel.id),
                initial_message=message if is_new_thread else None,
                provider=provider,
                routing_reason=routing_reason,
                thinking=forced_thinking,
                effort=chosen_effort,
            )

        # Label the response with model name (only when multi-model is active).
        # When thinking was used on Claude, also tag the chosen effort level so
        # users can see what depth the response was generated at without having
        # to set it explicitly per turn.
        if self.multi_model_active:
            # Strip any label the model echoed (handles old [Name] and the new
            # [Name · effort] format Claude may have started echoing).
            response = re.sub(
                rf'^(?:\*\*\[(?:{MODEL_LABEL_NAMES})(?:\s·\s\w+)?\]\*\*\s*|\*\[(?:{MODEL_LABEL_NAMES})(?:\s·\s\w+)?\]\*\s*|\[(?:{MODEL_LABEL_NAMES})(?:\s·\s\w+)?\]\s*)+',
                '',
                response,
            )
            label = provider.name
            if forced_thinking and provider is self.claude_provider:
                label = f"{label} · {chosen_effort or self.CLAUDE_THINKING_EFFORT}"
            response = f"**[{label}]** {response}"

        # Handle reactions
        for emoji in reactions:
            try:
                await message.add_reaction(emoji)
            except discord.HTTPException:
                pass

        # Extract and handle code files
        response, files = self._extract_code_files(response)

        # Render any LaTeX math blocks to PNGs (Discord has no native math
        # rendering). Source text stays in the message body so users can copy
        # it. Discord caps at 10 attachments per message; share the budget
        # with code files.
        latex_files = self._render_latex_attachments(response, max_files=10 - len(files))
        files.extend(latex_files)

        # Render inline [[speak:...]] markers to Mandarin TTS attachments so the
        # models can voice their own lessons. Shares the 10-attachment budget
        # with code + LaTeX files; strips the markers from the visible text.
        response, speak_files = await self._render_speak_attachments(
            self._mem_key(message.channel.id), response, max_files=10 - len(files)
        )
        files.extend(speak_files)

        # Same for inline [[french:...]] markers → French TTS attachments.
        response, french_files = await self._render_french_attachments(
            self._mem_key(message.channel.id), response, max_files=10 - len(files)
        )
        files.extend(french_files)

        # Send response (handle Discord's 2000 char limit)
        sent_msg = await self._send_response(thread, response, files)

        # Cache reasoning_content keyed by the sent Discord message id so the
        # next thinking-mode turn can echo it back to Deepseek's API.
        if reasoning and sent_msg is not None:
            self._store_reasoning(sent_msg.id, reasoning)

        # Record calibration bid, keyed by the BOT REPLY's message ts so a 👍/👎
        # on it maps back to exactly this response (per-message granularity that
        # works for both channel and thread replies on Slack).
        confidence = self._estimate_confidence(message.content or "", provider)
        record_idx = self.manager.calibration.record_bid(provider.name, confidence)
        if sent_msg is not None:
            self.manager.last_response_model[sent_msg.id] = provider.name
            self.manager.last_response_index[sent_msg.id] = record_idx

        # Mark memories as needing save (actual save happens in background task)
        self.manager.mark_dirty()
    
    async def _handle_reaction(self, reaction: PlatformReaction) -> None:
        """Track 👍/👎 feedback on the bot's own responses for calibration.
        (Self-message gating is done in _on_platform_reaction.)"""
        # Look up the calibration record by the reacted message's ts (the bot
        # reply we stored under sent_msg.id), giving per-response feedback.
        key = reaction.message_id
        if key not in self.manager.last_response_index:
            return

        emoji = reaction.emoji  # already mapped to unicode by the adapter
        # Positive: thumbs up, heart, fire, check, joy, sparkling heart, 100
        good_emoji = ('\U0001f44d', '\u2764\ufe0f', '\U0001f525', '\u2705',
                      '\U0001f602', '\U0001f496', '\U0001f4af')
        # Negative: thumbs down, x, confused
        bad_emoji = ('\U0001f44e', '\u274c', '\U0001f615')
        if emoji in good_emoji:
            self.manager.calibration.record_feedback(
                self.manager.last_response_index[key], "good"
            )
        elif emoji in bad_emoji:
            self.manager.calibration.record_feedback(
                self.manager.last_response_index[key], "bad"
            )

    async def _ensure_thread(self, message: discord.Message) -> tuple[discord.Thread, bool]:
        """Get existing thread or create new one. Returns (thread, is_new)."""
        if getattr(message.channel, 'is_thread', False):
            return message.channel, False
        
        # Create new thread
        thread = await message.create_thread(
            name=f"Chat with {message.author.display_name}",
            auto_archive_duration=60
        )
        await thread.send(
            f"🧵 Started new conversation!\n"
            f"Commands: `!help` for full list"
        )
        return thread, True

    # ----- Multi-model support methods -----

    def _strip_images_from_messages(self, messages: list[dict]) -> list[dict]:
        """Remove image content from messages for text-only models.

        Only call this when provider.supports_vision is False — Gemini and Claude
        both handle images natively and shouldn't go through this stripping.
        Deepseek is currently the only text-only provider in the system.
        """
        stripped = []
        for msg in messages:
            content = msg["content"]
            msg_id = msg.get("_msg_id")
            if isinstance(content, str):
                stripped.append(msg)
            elif isinstance(content, list):
                text_parts = [b for b in content if b.get("type") == "text"]
                if text_parts:
                    if len(text_parts) == 1:
                        new = {"role": msg["role"], "content": text_parts[0]["text"]}
                    else:
                        new = {"role": msg["role"], "content": text_parts}
                    if msg_id is not None:
                        new["_msg_id"] = msg_id
                    stripped.append(new)
                elif any(b.get("type") == "image" for b in content):
                    new = {"role": msg["role"], "content": "[An image was shared]"}
                    if msg_id is not None:
                        new["_msg_id"] = msg_id
                    stripped.append(new)
            else:
                stripped.append(msg)
        return stripped

    def _convert_messages_to_gemini_format(self, messages: list[dict]) -> list[dict]:
        """Convert internal/Anthropic message format to Gemini native format.

        Gemini uses {role, parts: [{text} | {inlineData}]} with role="model"
        for assistant turns. Image blocks become {inlineData: {mimeType, data}}.
        Used by _generate_gemini_native_response (the bookclub-mode chat path).
        """
        converted: list[dict] = []
        for msg in messages:
            content = msg.get("content", "")
            # Gemini uses "model" for assistant; "user" stays "user". No "system"
            # in contents — that goes in systemInstruction.
            role = "model" if msg.get("role") == "assistant" else "user"
            parts: list[dict] = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    bt = block.get("type")
                    if bt == "text":
                        parts.append({"text": block.get("text", "")})
                    elif bt == "image":
                        source = block.get("source", {})
                        parts.append({
                            "inlineData": {
                                "mimeType": source.get("media_type", "image/png"),
                                "data": source.get("data", ""),
                            }
                        })
            if parts:
                converted.append({"role": role, "parts": parts})
        return converted

    def _convert_messages_to_openai_format(self, messages: list[dict]) -> list[dict]:
        """Convert Anthropic-format messages to OpenAI chat format."""
        converted = []
        for msg in messages:
            content = msg["content"]
            msg_id = msg.get("_msg_id")
            if isinstance(content, str):
                new = {"role": msg["role"], "content": content}
                if msg_id is not None:
                    new["_msg_id"] = msg_id
                converted.append(new)
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"}
                        })
                if parts:
                    new = {"role": msg["role"], "content": parts}
                    if msg_id is not None:
                        new["_msg_id"] = msg_id
                    converted.append(new)
            else:
                converted.append(msg)
        return converted

    def _build_system_prompt(self, provider: ModelProvider, routing_reason: str = "") -> str:
        """Build system prompt tailored to the provider, with identity and routing context."""
        # Model identity line
        if provider.name == "Claude":
            identity = f"Claude (model: {provider.model_id}), an AI assistant made by Anthropic"
            identity_details = (
                "**[Deepseek]** messages are from your collaborator Deepseek (the fast, cheap, "
                "CJK-strong one) and **[Gemini]** messages are from your collaborator Gemini "
                "(the abstract-reasoning specialist). Both are different models, not you — "
                "your responses are labeled **[Claude]** by the bot. Your capabilities include "
                "vision (you can see images) and built-in web search — you can search the web "
                "anytime you think it would help, not just when users use !search. You tend "
                "to shine at complex analysis, code review, creative writing, and nuance."
            )
        elif provider.name == "Deepseek":
            identity = f"Deepseek (model: {provider.model_id}), an AI assistant made by DeepSeek"
            identity_details = (
                "**[Claude]** messages are from your collaborator Claude (the careful, "
                "thorough one) and **[Gemini]** messages are from your collaborator Gemini "
                "(the abstract-reasoning specialist with vision). Both are different models, "
                "not you. You can't see images, but you can search the web via Tavily "
                "function calling. You tend to shine at fast responses, factual questions, "
                "casual chat, and cost-efficiency.\n\n"
                "**Chinese language specialty**: You were trained on deep Chinese internet data "
                "(Zhihu, Baidu Baike, CSDN, Weibo, Douban, etc.) and have a much richer understanding "
                "of Chinese than Claude does. When Chinese text appears in conversation, it's your job "
                "to translate it to English for the group. When you think it's relevant or fun, include "
                "little mini-lessons breaking down interesting characters or words — e.g., how a character "
                "is composed, what its radicals mean, etymological tidbits, or how a phrase differs from "
                "its literal translation. Keep the lessons bite-sized and natural, not lecture-y.\n\n"
                "**Important: Always respond in English.** You can use Chinese characters inline when "
                "showing original text, breaking down words, or when a concept has no clean English "
                "equivalent — but your response itself should always be in English. Never reply with "
                "a wall of Chinese text. Your job is to be a bridge between languages, not to exclude "
                "English speakers from the conversation.\n\n"
                "**Formatting**: Write in flowing prose, not listicles. Avoid walls of bullet points, "
                "numbered lists, tables, and headers unless the user specifically asks for structured "
                "output. Keep it conversational — this is Slack chat, not a report. "
                "Minimize blank lines between paragraphs."
            )
        elif provider.name == "Gemini":
            identity = f"Gemini (model: {provider.model_id}), an AI assistant made by Google DeepMind"
            identity_details = (
                "**[Claude]** messages are from your collaborator Claude (the careful, "
                "thorough one with strong code review and multi-tool orchestration) and "
                "**[Deepseek]** messages are from your collaborator Deepseek (the fast, "
                "cheap, CJK-strong one). Both are different models, not you. You can see "
                "images and you have native Google Search grounding — when you call the "
                "web_search tool, the system routes it through Google's own search index "
                "(not a third-party meta-search), and citations are returned as structured "
                "groundingMetadata that the bot renders as source embeds. You tend "
                "to shine at genuinely novel reasoning (the kind ARC-AGI-2 tests), abstract "
                "pattern-finding, long-context synthesis, multi-step math, and questions "
                "where the answer requires recombining ideas in a non-obvious way.\n\n"
                "**Formatting**: Write in flowing prose, not listicles. Avoid walls of "
                "bullet points, numbered lists, tables, and headers unless the user "
                "specifically asks for structured output. Keep it conversational — this "
                "is Slack chat, not a slide deck. Minimize blank lines between paragraphs."
            )
        elif provider.name == "Mistral":
            identity = f"Mistral (model: {provider.model_id}), an AI assistant made by Mistral AI (Paris)"
            identity_details = (
                "Your collaborators **[Claude]** (careful/thorough), **[Gemini]** (abstract "
                "reasoning), and **[Deepseek]** (fast, CJK-strong) are different models, not "
                "you. The bot auto-prefixes your reply with **[Mistral]** — answer directly, "
                "never prepend your own name tag. You can't see images; you can search via "
                "Tavily. You shine at French and "
                "other European languages, multilingual nuance, and clean reasoning.\n\n"
                "**French language specialty**: you're French-native (a Paris lab), so when "
                "French comes up you're the resident tutor — translate it for the group and, "
                "when it's fun or useful, give bite-sized lessons (a liaison, a silent letter, "
                "a nasal vowel, an idiom that doesn't translate literally). To attach audio, "
                "write `[[french: la phrase]]`; the bot voices it (Azure fr-FR) and appends the "
                "IPA + a pronunciation note for you — you don't hand-write the IPA.\n\n"
                "**Important: always respond in English** (use French inline for examples), and "
                "write in flowing prose, not listicles — this is Slack chat. Minimize blank lines."
            )
        elif provider.name == "Qwen":
            identity = f"Qwen (model: {provider.model_id}), an AI assistant made by Alibaba"
            identity_details = (
                "Your collaborators **[Claude]**, **[Gemini]**, and **[Deepseek]** are "
                "different models, not you. The bot auto-prefixes your reply with **[Qwen]** — "
                "answer directly, never prepend your own name tag. You can't see images; you can "
                "search via Tavily. You shine at coding and math at low cost. Respond in English, "
                "in flowing prose (not listicles) — this is Slack chat. Minimize blank lines."
            )
        elif provider.name == "GLM":
            identity = f"GLM (model: {provider.model_id}), an AI assistant made by Zhipu AI"
            identity_details = (
                "Your collaborators **[Claude]**, **[Gemini]**, and **[Deepseek]** are "
                "different models, not you. The bot auto-prefixes your reply with **[GLM]** — "
                "answer directly, never prepend your own name tag. You can't see images; you can "
                "search via Tavily. You shine at agentic, tool-using, and coding tasks. Respond "
                "in English, in flowing prose (not listicles) — this is Slack chat. Minimize blank lines."
            )
        else:
            identity = f"{provider.name} (model: {provider.model_id}), an AI assistant"
            identity_details = ""

        # Routing context
        if routing_reason:
            routing_context = f"**Why you were chosen for this message:** {routing_reason}"
        else:
            routing_context = ""

        # Apply the cosmetic theme skin: the "You're **X**" name becomes the themed
        # display name and any flavor persona note is appended. EVA's display name
        # == the canonical name and its persona is empty, so the default theme is
        # behavior-preserving. The {model_identity} line and the [Claude]-style
        # collaborator labels stay canonical (factual identity + routing key).
        flavor = self.theme.flavors.get(provider.id)
        display = flavor.display_name if flavor else provider.name
        if flavor and flavor.persona:
            identity_details = (identity_details + "\n\n" + flavor.persona) if identity_details else flavor.persona

        prompt = CONFIG.system_prompt
        prompt = prompt.replace("{model_identity}", identity)
        prompt = prompt.replace("{model_name}", display)
        prompt = prompt.replace("{model_id}", provider.model_id)
        prompt = prompt.replace("{identity_details}", identity_details)
        prompt = prompt.replace("{routing_context}", routing_context)
        prompt = prompt.replace("{theme_blurb}", self.theme.blurb)
        sc = getattr(CONFIG, "server_context", "") or ""
        prompt = prompt.replace("{server_context}", (sc + "\n") if sc else "")
        return prompt

    # OpenAI-compatible function-calling tool definition.
    # Shared by both Deepseek and Gemini — both route web search through Tavily
    # over the OpenAI shim's tool-calling support. (Gemini also has native
    # google_search grounding, but that requires the native API, not the shim.)
    OPENAI_COMPATIBLE_TOOLS = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need up-to-date facts, news, or information you don't have.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }]

    async def _tavily_search(self, query: str, max_results: int = 5) -> SearchResult:
        """Perform a web search via Tavily. Returns raw hits (not synthesized).

        SearchResult.is_grounded_answer is False — callers should feed .text
        back through a model for synthesis if they want a coherent answer.
        """
        if not self.tavily_client:
            return SearchResult(
                text="Web search is not available (no Tavily API key configured).",
            )
        try:
            result = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                max_results=max_results,
            )
            if not result.get("results"):
                return SearchResult(text=f"No results found for: {query}")

            citations: list[dict] = []
            lines: list[str] = []
            for r in result["results"]:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                snippet = r.get("content", "")
                citations.append({"url": url, "title": title, "snippet": snippet})
                lines.append(f"**{title}**\n{url}\n{snippet}\n")
            return SearchResult(
                text="\n".join(lines),
                citations=citations,
                is_grounded_answer=False,
                queries_used=[query],
            )
        except Exception as e:
            return SearchResult(text=f"Search error: {e}")

    async def _google_native_search(self, query: str, max_results: int = 5) -> SearchResult:
        """Perform a web search via Gemini's native google_search grounding.

        Returns a SearchResult where is_grounded_answer=True — the response is
        already a synthesized answer with citations from groundingMetadata.
        Callers should display .text directly and render .citations as embeds.

        Uses raw aiohttp against the native endpoint (no new SDK dep). This is
        the only Gemini code path that doesn't go through the OpenAI shim.
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return SearchResult(
                text="Native Google search is not available (no GEMINI_API_KEY configured).",
            )
        model_id = self.gemini_provider.model_id
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": gemini_key,
        }
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": query}]}
            ],
            "tools": [{"google_search": {}}],
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body, timeout=60) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        return SearchResult(
                            text=f"Google native search error (HTTP {resp.status}): {err_text[:500]}",
                        )
                    data = await resp.json()
        except asyncio.TimeoutError:
            return SearchResult(text="Google native search timed out.")
        except Exception as e:
            return SearchResult(text=f"Google native search error: {e}")

        # Parse response: candidates[0].content.parts[*].text + groundingMetadata
        candidates = data.get("candidates", [])
        if not candidates:
            return SearchResult(text=f"No results for: {query}")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            text = f"No textual response for: {query}"

        # Track usage if reported (Gemini returns usageMetadata at top level)
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        cached_tokens = usage.get("cachedContentTokenCount", 0)
        if prompt_tokens or output_tokens:
            uncached_input = max(0, prompt_tokens - cached_tokens)
            self.gemini_provider.record_usage(
                uncached_input,
                output_tokens,
                cached_input_tokens=cached_tokens,
            )
            self.gemini_provider.total_requests += 1

        # Extract structured citations from groundingMetadata.groundingChunks
        grounding = candidate.get("groundingMetadata", {})
        citations: list[dict] = []
        seen_urls: set[str] = set()
        for chunk in grounding.get("groundingChunks", []):
            web = chunk.get("web", {})
            curl = web.get("uri", "")
            title = web.get("title", "")
            if curl and curl not in seen_urls:
                seen_urls.add(curl)
                citations.append({"url": curl, "title": title, "snippet": ""})
            if len(citations) >= max_results:
                break

        queries_used = grounding.get("webSearchQueries", [query])

        return SearchResult(
            text=text,
            citations=citations,
            is_grounded_answer=True,
            queries_used=queries_used,
        )

    async def _search_for(self, provider: ModelProvider, query: str, max_results: int = 5) -> SearchResult:
        """Dispatch a search to the provider's preferred SearchBackend.

        Falls back to Tavily if the provider's preferred backend isn't available.
        Returns an empty-ish SearchResult if nothing is configured.
        """
        backend = provider.search_backend
        if backend == "google_native" and os.getenv("GEMINI_API_KEY"):
            return await self._google_native_search(query, max_results=max_results)
        # Default / fallback: Tavily (if configured)
        if self.tavily_client:
            return await self._tavily_search(query, max_results=max_results)
        return SearchResult(
            text=f"No search backend available for {provider.name} "
                 "(set GEMINI_API_KEY for native grounding or TAVILY_API_KEY for Tavily).",
        )

    # ----- Reading material (bookclub mode) -----

    AO3_WORK_ID_RE = re.compile(r"archiveofourown\.org/works/(\d+)")

    @classmethod
    def _build_ao3_full_work_url(cls, url: str) -> Optional[str]:
        """Normalize any AO3 work URL to its full-work, adult-bypass form.

        Accepts: works/12345, works/12345/chapters/67890, with query strings, etc.
        Returns: works/12345?view_full_work=true&view_adult=true (or None if
        the URL doesn't match the AO3 work pattern).
        """
        match = cls.AO3_WORK_ID_RE.search(url)
        if not match:
            return None
        work_id = match.group(1)
        return f"https://archiveofourown.org/works/{work_id}?view_full_work=true&view_adult=true"

    async def _fetch_ao3_work(self, url: str) -> tuple[Optional[ReadingMaterial], Optional[str]]:
        """Fetch an AO3 work and return (material, error_reason).

        Exactly one of the two is non-None. The error_reason is a
        user-facing string (used by !load to give a specific message).
        Common cases:
        - "missing_bs4"        : beautifulsoup4 not installed
        - "bad_url"            : URL didn't match AO3 work pattern
        - "shields_up"         : AO3 is load-shedding anonymous traffic
        - "auth_required"      : work is registered-only / locked
        - "http_404"           : work not found
        - "http_<status>"      : other non-200 response
        - "network_error: ..." : aiohttp/network exception
        - "no_content"         : parsed but couldn't find chapter text
        """
        if not _HAS_BS4:
            return None, "missing_bs4"
        normalized = self._build_ao3_full_work_url(url)
        if not normalized:
            return None, "bad_url"

        # Standard browser UA — AO3's anti-bot heuristics aren't UA-based,
        # but using a real browser string costs nothing.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        # Logged-in cookie bypasses shields-up and accesses registered-only works.
        # To populate: log in at archiveofourown.org in your browser, open dev
        # tools → Application → Cookies → copy `_otwarchive_session`. Format the
        # env var as just the cookie value (no "Cookie:" prefix).
        ao3_cookie = os.getenv("AO3_COOKIE", "").strip()
        if ao3_cookie:
            # Accept either bare value or "name=value" form
            cookie_header = (
                ao3_cookie if "=" in ao3_cookie
                else f"_otwarchive_session={ao3_cookie}"
            )
            headers["Cookie"] = cookie_header

        html_text = ""
        last_status: Optional[int] = None
        last_network_err: Optional[str] = None
        # Retry on transient failures (network errors, 5xx). 403 shields-up
        # is NOT retried — it's deliberate load-shedding, won't fix in 10s.
        for attempt, delay in enumerate([0, 2, 5]):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(normalized, timeout=60) as resp:
                        last_status = resp.status
                        if resp.status == 200:
                            html_text = await resp.text()
                            break
                        if resp.status == 403:
                            # Peek at the body to confirm shields-up vs other 403
                            body = await resp.text()
                            if "Shields are up" in body or "shields are up" in body.lower():
                                return None, "shields_up"
                            # Other 403 — auth required (locked work)
                            return None, "auth_required"
                        if resp.status == 404:
                            return None, "http_404"
                        if 500 <= resp.status < 600:
                            # Retry transient server errors
                            continue
                        # Other non-200 — bail
                        return None, f"http_{resp.status}"
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_network_err = f"{type(e).__name__}: {e}"
                continue

        if not html_text:
            if last_status is not None:
                return None, f"http_{last_status}"
            return None, f"network_error: {last_network_err or 'unknown'}"

        soup = _BeautifulSoup(html_text, "html.parser")

        # Work title (in preface metadata)
        title_tag = soup.select_one("h2.title.heading") or soup.select_one("h2.title")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled AO3 Work"

        # Main chapter content lives in #chapters
        chapters_container = soup.select_one("#chapters")
        if chapters_container is None:
            return None, "no_content"

        body_parts: list[str] = []
        chapter_breaks: list[tuple[int, str]] = []
        chapter_divs = chapters_container.select("div.chapter")

        if chapter_divs:
            # Multi-chapter work — each <div class="chapter"> has its own
            # title + userstuff body.
            for idx, ch_div in enumerate(chapter_divs, 1):
                ch_title_tag = ch_div.select_one("h3.title")
                if ch_title_tag:
                    ch_title = ch_title_tag.get_text(" ", strip=True)
                else:
                    ch_title = f"Chapter {idx}"

                body_tag = ch_div.select_one("div.userstuff")
                body_text = body_tag.get_text("\n", strip=True) if body_tag else ""
                offset = sum(len(p) for p in body_parts)
                chapter_breaks.append((offset, ch_title))
                body_parts.append(f"## {ch_title}\n\n{body_text}\n\n")
        else:
            # Single-chapter / oneshot — one userstuff block under #chapters
            body_tag = chapters_container.select_one("div.userstuff")
            if body_tag:
                body_parts.append(body_tag.get_text("\n", strip=True))

        full_text = "".join(body_parts).strip()
        if not full_text:
            return None, "no_content"

        # Normalize whitespace and decode any lingering HTML entities
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = _html.unescape(full_text)

        return ReadingMaterial(
            url=url,
            title=title,
            text=full_text,
            chapter_breaks=chapter_breaks,
        ), None

    # Chapter heading patterns we look for in uploaded HTML / text. Anchored
    # to line starts (via re.MULTILINE) so we don't match phrases in prose like
    # "she opened to Chapter 3 and read…". Each pattern matches a full line so
    # we get the whole heading text (including ":" + subtitle) back as group 1.
    CHAPTER_HEADING_RE = re.compile(
        r'^('
        r'Chapter\s+\d+[^\n]{0,200}'          # "Chapter 1", "Chapter 12: ..."
        r'|Prologue[^\n]{0,200}'
        r'|Epilogue[^\n]{0,200}'
        r'|Interlude(?:\s+\d+)?[^\n]{0,200}'
        r'|Part\s+\d+[^\n]{0,200}'             # "Part 1: ..."
        r')$',
        re.MULTILINE | re.IGNORECASE,
    )

    @classmethod
    def _detect_chapter_breaks(cls, text: str) -> list[tuple[int, str]]:
        """Find chapter heading positions in text. Returns (char_offset, title) list.

        Heuristic: matches lines that look like "Chapter N[: Title]",
        "Prologue", "Epilogue", "Interlude [N]", or "Part N". Works for AO3
        download HTML's heading layout once it's been get_text'd. Returns
        empty list if no headings found — callers should treat that as
        "no chapter structure available."
        """
        return [
            (m.start(), m.group(1).strip())
            for m in cls.CHAPTER_HEADING_RE.finditer(text)
        ]

    @staticmethod
    def _slice_material_to_chapters(
        material: "ReadingMaterial",
        start_ch: int,
        end_ch: int,
    ) -> "ReadingMaterial":
        """Return a new ReadingMaterial covering chapters start_ch..end_ch (1-indexed inclusive).

        Caller is responsible for validating the range. Adjusted chapter_breaks
        on the sliced material have offsets relative to the sliced text (so
        !chapters on a scoped thread shows the right numbers).
        """
        breaks = material.chapter_breaks
        start_offset = breaks[start_ch - 1][0]
        if end_ch < len(breaks):
            end_offset = breaks[end_ch][0]
        else:
            end_offset = len(material.text)
        sliced_text = material.text[start_offset:end_offset].strip()
        sliced_breaks = [
            (b[0] - start_offset, b[1])
            for b in breaks[start_ch - 1:end_ch]
        ]
        range_desc = (
            f"Chapter {start_ch}" if start_ch == end_ch
            else f"Chapters {start_ch}-{end_ch}"
        )
        return ReadingMaterial(
            url=f"{material.url}#scope=ch{start_ch}-{end_ch}",
            title=f"{material.title} — {range_desc}",
            text=sliced_text,
            chapter_breaks=sliced_breaks,
        )

    @staticmethod
    def _build_reading_material_system_block(material: "ReadingMaterial") -> str:
        """Format a reading material as a system-prompt block.

        Used by all three providers — for Claude it becomes a separately
        cacheable block; for Deepseek/Gemini-fallback it gets prepended to
        the live system text. The framing tells the model to treat the text
        as primary source for any bookclub discussion.
        """
        recap = ""
        if material.recap_text:
            recap = (
                f"### Story so far\n\n"
                f"You're scoped to a later part of the work and can't see the earlier "
                f"chapters directly. Here's what happened before, so you can follow the "
                f"discussion:\n\n"
                f"{material.recap_text}\n\n"
            )
        return (
            f"## Reading Material: {material.title}\n\n"
            f"{recap}"
            f"This text has been loaded for bookclub discussion in this channel "
            f"(source: {material.url}). You have full access to the work below — "
            f"reference specific passages, characters, plot points, and structural "
            f"choices freely. Treat it as the canonical source for any question "
            f"about the work. The text follows between the markers.\n\n"
            f"--- BEGIN WORK ---\n\n"
            f"{material.text}\n\n"
            f"--- END WORK ---"
        )

    # --- Bookclub chapter recaps (the "previously on" for scoped threads) ----
    RECAP_SYSTEM = (
        "You are writing a spoiler-bounded 'previously on' recap of ONE chapter of a "
        "longer work, for a reader about to jump into a LATER chapter who needs to know "
        "what came before. Summarize only THIS chapter's key events, character beats, and "
        "revelations in 2-3 tight sentences. No preamble, no 'in this chapter' — just the "
        "recap prose. Don't speculate beyond the text."
    )

    async def _generate_chapter_recap(
        self, guild_id: int, parent_material: "ReadingMaterial", chapter_num: int
    ) -> Optional[str]:
        """Summarize a single chapter into a 2-3 sentence recap via the cheapest
        capable model (Deepseek → Claude → Gemini). Returns None on failure."""
        provider = next(
            (p for p in (self.deepseek_provider, self.claude_provider, self.gemini_provider)
             if p.enabled),
            None,
        )
        if provider is None:
            return None
        chapter = self._slice_material_to_chapters(parent_material, chapter_num, chapter_num)
        messages = [{"role": "user", "content": chapter.text}]
        try:
            out = await self._panel_complete(provider, guild_id, messages, self.RECAP_SYSTEM)
        except Exception as e:
            print(f"⚠️  Chapter {chapter_num} recap failed: {e}")
            return None
        if self._is_provider_error(provider, out):
            return None
        return out.strip()

    async def _ensure_chapter_recaps(
        self, guild_id: int, parent_material: "ReadingMaterial", up_to_chapter: int
    ) -> None:
        """Generate + cache any missing chapter recaps for chapters 1..up_to_chapter
        on the parent material. Missing recaps are generated in parallel and
        persist, so a chapter is only ever summarized once."""
        missing = [
            ch for ch in range(1, up_to_chapter + 1)
            if ch not in parent_material.chapter_recaps
        ]
        if not missing:
            return
        results = await asyncio.gather(
            *(self._generate_chapter_recap(guild_id, parent_material, ch) for ch in missing)
        )
        for ch, recap in zip(missing, results):
            if recap:
                parent_material.chapter_recaps[ch] = recap
        self.manager.mark_dirty()

    @staticmethod
    def _format_recap_prefix(
        parent_material: "ReadingMaterial", before_chapter: int
    ) -> Optional[str]:
        """Assemble the 'previously on' text from cached recaps of chapters
        1..before_chapter-1. None if there are no recaps to show."""
        lines = [
            f"- **Ch {ch}:** {parent_material.chapter_recaps[ch]}"
            for ch in range(1, before_chapter)
            if ch in parent_material.chapter_recaps
        ]
        return "\n".join(lines) if lines else None

    async def _create_gemini_cache(
        self,
        material: "ReadingMaterial",
        system_text: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> Optional[tuple[str, datetime]]:
        """Create a Gemini cachedContents entry for a reading material.

        POSTs to /v1beta/cachedContents with the fic as a user/model exchange
        prefix and a 24-hour TTL. Returns (cache_name, expires_at) on success,
        None on failure.

        Per Gemini's API, generateContent calls referencing a cachedContent
        can NOT also set systemInstruction, tools, or tool_config — those
        have to be baked into the cache at creation time. So we accept them
        here and put them on the cache. The chat call then only supplies
        cachedContent + new contents + generationConfig.
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return None
        url = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": gemini_key,
        }
        fic_intro = (
            f"I'm going to share a work with you for a bookclub discussion. "
            f"It's titled '{material.title}' (source: {material.url}). "
            f"Here is the full text — please read it; I'll ask questions about "
            f"it afterwards.\n\n"
        )
        body: dict = {
            "model": f"models/{self.gemini_provider.model_id}",
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": fic_intro + material.text}],
                },
                {
                    "role": "model",
                    "parts": [{"text":
                        "I've read the full work and have it loaded. Ready to "
                        "discuss whenever you'd like."
                    }],
                },
            ],
            "ttl": f"{GEMINI_CACHE_TTL_SECONDS}s",
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if tools:
            body["tools"] = tools
        print(
            f"📤 Uploading '{material.title}' (~{material.estimated_tokens:,} tokens) "
            f"to a Gemini context cache (TTL {GEMINI_CACHE_TTL_HOURS:g}h)…"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body, timeout=180) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        print(f"⚠️  Gemini cache creation failed: HTTP {resp.status}: {err_text[:300]}")
                        return None
                    data = await resp.json()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"⚠️  Gemini cache creation network error: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Gemini cache creation unexpected error: {e}")
            return None

        cache_name = data.get("name", "")
        expire_time_str = data.get("expireTime", "")
        if not cache_name:
            return None
        try:
            # ISO 8601 with Z suffix → fromisoformat needs +00:00. Keep it
            # timezone-AWARE (UTC): every expiry comparison uses _utcnow() (aware
            # UTC), so this must be aware too — do NOT strip tzinfo.
            expires_at = _as_utc(
                datetime.fromisoformat(expire_time_str.replace("Z", "+00:00"))
            )
        except (ValueError, AttributeError):
            expires_at = _utcnow() + timedelta(seconds=GEMINI_CACHE_TTL_SECONDS)

        # Record the one-time CREATION token cost (cached tokens billed ~at the
        # input rate). This was once untracked — a silent ~$156 bill. Kept.
        cached_tokens = int((data.get("usageMetadata") or {}).get("totalTokenCount", 0)) \
            or material.estimated_tokens
        self.gemini_provider.record_usage(cached_tokens, 0)
        self.gemini_provider.total_requests += 1
        # STORAGE (per token-hour) is NOT pre-charged here anymore. It used to add
        # one full TTL's worth at creation, so a cache the sliding window kept
        # alive for days still showed one TTL — !cost read cheap while the balance
        # bled. It now accrues against the cache's REAL lifetime: live caches are
        # metered on the fly in get_cost_summary and the elapsed cost is settled
        # into total_cache_storage_cost_est at teardown (_settle_gemini_storage).
        storage_per_ttl = (
            (cached_tokens / 1_000_000)
            * GEMINI_CACHE_STORAGE_COST_PER_MTOK_HOUR
            * (GEMINI_CACHE_TTL_SECONDS / 3600)
        )
        print(
            f"🟢 Created Gemini cache {cache_name} "
            f"({cached_tokens:,} tok, expires {expires_at:%Y-%m-%d %H:%M} UTC); "
            f"storage ≈ ${storage_per_ttl:.2f} per {GEMINI_CACHE_TTL_HOURS:g}h alive "
            f"(metered live in !cost) — deleted on !unload/!uncache"
        )
        return cache_name, expires_at

    def _gemini_cache_key(self) -> str:
        """Backend-tagged key for a material's Gemini cache handle, so a
        developer-API cachedContents id is never reused against Vertex's
        CachedContent (different surfaces). e.g. 'Gemini:developer_api'."""
        return f"Gemini:{self.gemini_provider.backend or 'developer_api'}"

    async def _ensure_gemini_cache(
        self,
        material: "ReadingMaterial",
        system_text: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> Optional[str]:
        """Get a Gemini cache handle for this material, creating one if needed.

        Reuses existing cache if it's still valid (with a 5-minute safety
        margin). On a miss, creates a fresh cache with the supplied
        systemInstruction + tools baked in (since chat calls referencing a
        cache cannot set those fields themselves). Returns the cache name
        (e.g. "cachedContents/abc123") or None if creation failed.
        """
        key = self._gemini_cache_key()
        existing = material.cache_handles.get(key)
        expires = material.cache_expires_at.get(key)
        # One-time migration: adopt a legacy untagged "Gemini" handle (created
        # before backend-tagging) when on developer_api, so the upgrade doesn't
        # orphan a live cache.
        if existing is None and self.gemini_provider.backend == "developer_api":
            legacy = material.cache_handles.pop("Gemini", None)
            legacy_exp = material.cache_expires_at.pop("Gemini", None)
            if legacy:
                existing = material.cache_handles[key] = legacy
                if legacy_exp is not None:
                    expires = material.cache_expires_at[key] = legacy_exp
        if existing and expires and _as_utc(expires) > _utcnow() + timedelta(minutes=5):
            # Sliding-window TTL: once the cache is past the halfway point of its
            # life, bump it back to a full TTL so an ACTIVE discussion never hits
            # a mid-conversation expiry (refreshing only past halfway avoids a
            # PATCH on every message). An IDLE cache still dies ~TTL after its
            # last use. Best-effort — on failure the cache still works to expiry.
            halfway = _utcnow() + timedelta(seconds=GEMINI_CACHE_TTL_SECONDS / 2)
            if _as_utc(expires) < halfway and await self._refresh_gemini_cache(existing):
                material.cache_expires_at[key] = (
                    _utcnow() + timedelta(seconds=GEMINI_CACHE_TTL_SECONDS)
                )
                self.manager.mark_dirty()
            return existing
        # Existing handle is missing/stale — best-effort delete any old cache so
        # we don't leave an overlapping one billing storage, then make a fresh one.
        if existing:
            await self._delete_gemini_cache(existing)

        # Settle the replaced cache's REAL storage cost before we overwrite its
        # created_at below, so a churned cache's spend isn't silently dropped.
        self._settle_gemini_storage(material, key)
        result = await self._create_gemini_cache(
            material, system_text=system_text, tools=tools
        )
        if result is None:
            return None
        cache_name, expires_at = result
        material.cache_handles[key] = cache_name
        material.cache_expires_at[key] = expires_at
        material.cache_created_at[key] = _utcnow()
        self.manager.mark_dirty()  # persist the new cache handle
        return cache_name

    async def _refresh_gemini_cache(self, cache_name: Optional[str]) -> bool:
        """Slide a developer-API Gemini cache's expiry forward to now+TTL
        (PATCH ?updateMask=ttl). Sliding-window TTL: an actively-used cache never
        expires mid-discussion, while an idle one still dies GEMINI_CACHE_TTL_HOURS
        after its last use. Best-effort; returns True on success."""
        if not cache_name:
            return False
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return False
        url = f"https://generativelanguage.googleapis.com/v1beta/{cache_name}?updateMask=ttl"
        headers = {"Content-Type": "application/json", "x-goog-api-key": gemini_key}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    url, headers=headers,
                    json={"ttl": f"{GEMINI_CACHE_TTL_SECONDS}s"},
                    timeout=30,
                ) as resp:
                    if resp.status == 200:
                        print(f"🔄 Slid Gemini cache TTL → +{GEMINI_CACHE_TTL_HOURS:g}h ({cache_name})")
                        return True
                    print(f"⚠️  Gemini cache TTL refresh {cache_name}: HTTP {resp.status}")
                    return False
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"⚠️  Gemini cache TTL refresh network error: {e}")
            return False

    async def _delete_gemini_cache(self, cache_name: Optional[str]) -> bool:
        """Delete a Gemini cachedContents entry to stop its storage billing.

        Returns True ONLY when the cache is actually gone: HTTP 200/204, a 404,
        or a 403 whose body says "not found" (Gemini 403s rather than 404s for a
        gone cache to avoid leaking existence, with a "…not found (or permission
        denied)" message). A *bare* 403 — no "not found" in the body — means the
        delete was DENIED while the cache is still alive and still billing
        (balance depleted / permission / quota); that is a real FAILURE and
        returns False, so callers don't falsely report "billing stopped" or drop
        the handle on a still-accruing cache (the bug behind !uncache lying)."""
        if not cache_name:
            return False
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return False
        url = f"https://generativelanguage.googleapis.com/v1beta/{cache_name}"
        headers = {"x-goog-api-key": gemini_key}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers, timeout=60) as resp:
                    if resp.status in (200, 204):
                        print(f"🗑️  Deleted Gemini cache {cache_name} (HTTP {resp.status})")
                        return True
                    body = (await resp.text())[:300]
                    if resp.status == 404 or (
                        resp.status == 403 and "not found" in body.lower()
                    ):
                        # Genuinely gone — the desired end state, so: success.
                        print(f"🗑️  Gemini cache {cache_name} already gone (HTTP {resp.status})")
                        return True
                    # Bare 403 / 429 / 5xx: the cache is (or may still be) ALIVE
                    # and billing. Do NOT claim success.
                    print(f"⚠️  Gemini cache delete {cache_name} FAILED — cache may still "
                          f"be billing: HTTP {resp.status}: {body}")
                    return False
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"⚠️  Gemini cache delete network error: {e}")
            return False

    def _settle_gemini_storage(self, material: "ReadingMaterial", key: str) -> None:
        """Fold a Gemini cache's accrued STORAGE cost into the settled running
        total (total_cache_storage_cost_est) and forget its created_at. Called
        when a cache leaves live tracking (deleted, replaced, abandoned) so !cost
        reflects the cache's REAL lifetime, not a creation-time one-TTL guess.
        Idempotent — a no-op when the key has no tracked creation."""
        created = material.cache_created_at.pop(key, None)
        if created is None:
            return
        self.gemini_provider.total_cache_storage_cost_est += _gemini_storage_cost(
            material.estimated_tokens,
            created,
            material.cache_expires_at.get(key),
            _utcnow(),
        )

    async def _drop_gemini_cache(self, material: "ReadingMaterial") -> tuple[int, int]:
        """Delete a material's live Gemini cache(s) and clear the handle(s).
        Handles any backend-tagged key ("Gemini:developer_api" / "Gemini:vertex")
        plus the legacy untagged "Gemini", deleting each via the matching
        backend's API.

        The delete is attempted BEFORE the handle is cleared, and the handle is
        KEPT if the delete fails (e.g. a billing/permission 403 on a still-alive
        cache) so it isn't orphaned beyond our reach. Returns (attempted, deleted)
        so callers can report the truth instead of an unconditional success."""
        keys = [k for k in list(material.cache_handles) if k.startswith("Gemini")]
        attempted = 0
        deleted = 0
        changed = False
        for k in keys:
            cache_name = material.cache_handles.get(k)
            if not cache_name:
                material.cache_handles.pop(k, None)
                material.cache_expires_at.pop(k, None)
                material.cache_created_at.pop(k, None)
                continue
            attempted += 1
            if k.endswith(":vertex"):
                ok = await self._delete_gemini_vertex_cache(cache_name)
            else:
                ok = await self._delete_gemini_cache(cache_name)
            if ok:
                deleted += 1
                # Settle real storage BEFORE clearing expiry (settle reads it).
                self._settle_gemini_storage(material, k)
                material.cache_handles.pop(k, None)
                material.cache_expires_at.pop(k, None)
                changed = True
            # else: keep the handle — the cache may still be live and billing.
        if changed:
            self.manager.mark_dirty()
        return attempted, deleted

    async def _delete_gemini_vertex_cache(self, cache_name: Optional[str]) -> bool:
        """Delete a Vertex CachedContent (stops its storage billing).
        ⚠️ UNVERIFIED — needs the vertexai SDK + ADC. Best-effort, never raises."""
        if not cache_name:
            return False

        def _del() -> bool:
            try:
                from vertexai.caching import CachedContent
                CachedContent(cached_content_name=cache_name).delete()
                print(f"🗑️  Deleted Vertex cache {cache_name}")
                return True
            except Exception as e:
                print(f"⚠️  Vertex cache delete {cache_name}: {e}")
                return False

        return await asyncio.to_thread(_del)

    def _create_gemini_vertex_cache(
        self, material: "ReadingMaterial", system_text: Optional[str],
    ) -> Optional[tuple[str, datetime]]:
        """Create a Vertex CachedContent for a reading material (blocking — call
        via to_thread). ⚠️ UNVERIFIED — the vertexai SDK surface varies by
        version. Returns (cache_resource_name, expires_at) or None."""
        try:
            import vertexai
            from vertexai.generative_models import Content, Part, Tool, grounding
            from vertexai.caching import CachedContent
        except ImportError:
            print("⚠️  Vertex cache: google-cloud-aiplatform not installed.")
            return None
        vcfg = self.gemini_provider.backends.get("vertex", {})
        project = os.getenv(vcfg.get("project_env", "GOOGLE_CLOUD_PROJECT"))
        location = (os.getenv(vcfg.get("location_env", "GOOGLE_CLOUD_LOCATION"))
                    or vcfg.get("location_default", "us-central1"))
        if not project:
            print("⚠️  Vertex cache: GOOGLE_CLOUD_PROJECT not set.")
            return None
        try:
            vertexai.init(project=project, location=location)
            fic_intro = (
                f"I'm going to share a work with you for a bookclub discussion. "
                f"It's titled '{material.title}' (source: {material.url}). "
                f"Here is the full text — please read it; I'll ask questions "
                f"about it afterwards.\n\n"
            )
            contents = [Content(role="user", parts=[Part.from_text(fic_intro + material.text)])]
            search_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
            cache = CachedContent.create(
                model_name=self.gemini_provider.model_id,
                system_instruction=system_text,
                contents=contents,
                tools=[search_tool],
                ttl=timedelta(seconds=GEMINI_CACHE_TTL_SECONDS),
            )
        except Exception as e:
            print(f"⚠️  Vertex cache creation failed: {e}")
            return None
        expires_at = _utcnow() + timedelta(seconds=GEMINI_CACHE_TTL_SECONDS)
        cached_tokens = material.estimated_tokens
        self.gemini_provider.record_usage(cached_tokens, 0)
        self.gemini_provider.total_requests += 1
        # Storage accrues by real lifetime (see _create_gemini_cache), metered
        # live in !cost and settled at teardown — not pre-charged here.
        print(f"🟢 Created Vertex cache {cache.name} "
              f"(~{cached_tokens:,} tok, expires {expires_at:%Y-%m-%d %H:%M} UTC)")
        return cache.name, expires_at

    async def _ensure_gemini_vertex_cache(
        self, material: "ReadingMaterial", system_text: Optional[str] = None,
    ) -> Optional[str]:
        """Get/create a Vertex CachedContent handle (backend-tagged
        'Gemini:vertex'). ⚠️ UNVERIFIED. Returns the cache resource name or None."""
        key = self._gemini_cache_key()
        existing = material.cache_handles.get(key)
        expires = material.cache_expires_at.get(key)
        if existing and expires and _as_utc(expires) > _utcnow() + timedelta(minutes=5):
            # Sliding-window TTL (see _ensure_gemini_cache). Best-effort.
            halfway = _utcnow() + timedelta(seconds=GEMINI_CACHE_TTL_SECONDS / 2)
            if _as_utc(expires) < halfway and await self._refresh_gemini_vertex_cache(existing):
                material.cache_expires_at[key] = (
                    _utcnow() + timedelta(seconds=GEMINI_CACHE_TTL_SECONDS)
                )
                self.manager.mark_dirty()
            return existing
        if existing:
            await self._delete_gemini_vertex_cache(existing)
        self._settle_gemini_storage(material, key)
        result = await asyncio.to_thread(
            self._create_gemini_vertex_cache, material, system_text
        )
        if result is None:
            return None
        cache_name, expires_at = result
        material.cache_handles[key] = cache_name
        material.cache_expires_at[key] = expires_at
        material.cache_created_at[key] = _utcnow()
        self.manager.mark_dirty()
        return cache_name

    async def _refresh_gemini_vertex_cache(self, cache_name: Optional[str]) -> bool:
        """Slide a Vertex CachedContent's expiry forward to now+TTL (SDK update).
        ⚠️ UNVERIFIED. Best-effort; returns True on success."""
        if not cache_name:
            return False

        def _upd() -> bool:
            try:
                from vertexai.caching import CachedContent
                CachedContent(cached_content_name=cache_name).update(
                    ttl=timedelta(seconds=GEMINI_CACHE_TTL_SECONDS)
                )
                return True
            except Exception as e:
                print(f"⚠️  Vertex cache TTL refresh {cache_name}: {e}")
                return False

        return await asyncio.to_thread(_upd)

    async def _set_reading_material(self, key: int, material: "ReadingMaterial") -> None:
        """Pin a reading material to a channel/thread key, first dropping any prior
        material's Gemini cache so we don't orphan it (the !load-over-a-load leak)."""
        old = self.manager.reading_materials.get(key)
        if old is not None and old is not material:
            await self._drop_gemini_cache(old)
        self.manager.reading_materials[key] = material

    async def _reconcile_gemini_caches(self) -> None:
        """Startup sweep: delete every live Gemini cache NOT referenced by a
        current reading material. The belt-and-suspenders backstop to the
        per-teardown deletes — catches orphans from a crash, an archived thread
        we couldn't resolve at !unload, or older code. Runs in setup_hook BEFORE
        the bot processes messages, so it can't race a cache mid-creation.
        ⚠️ Safe ONLY because this bot is the sole user of the Gemini key — it
        would delete a co-tenant's caches otherwise. Best-effort; never raises."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return
        # Cache handles we must KEEP (referenced by a loaded work / scoped
        # thread) — across any backend-tagged key + the legacy untagged one.
        referenced = {
            v
            for m in self.manager.reading_materials.values()
            for k, v in m.cache_handles.items()
            if k.startswith("Gemini")
        }
        referenced.discard(None)
        base = "https://generativelanguage.googleapis.com/v1beta"
        headers = {"x-goog-api-key": gemini_key}
        live: list[str] = []
        try:
            async with aiohttp.ClientSession() as session:
                page_token = None
                for _ in range(20):  # page cap — backstop against a runaway loop
                    url = f"{base}/cachedContents?pageSize=100"
                    if page_token:
                        url += f"&pageToken={page_token}"
                    async with session.get(url, headers=headers, timeout=60) as resp:
                        if resp.status != 200:
                            print(f"⚠️  Gemini cache reconcile: list HTTP {resp.status} — skipping sweep")
                            return
                        data = await resp.json()
                    live.extend(
                        c["name"] for c in data.get("cachedContents", []) if c.get("name")
                    )
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"⚠️  Gemini cache reconcile: list failed ({e}) — skipping sweep")
            return
        except Exception as e:
            print(f"⚠️  Gemini cache reconcile: unexpected error ({e}) — skipping sweep")
            return
        if not live:
            return
        orphans = [n for n in live if n not in referenced]
        if not orphans:
            print(f"🧹 Gemini cache reconcile: {len(live)} live, all referenced — clean.")
            return
        print(
            f"🧹 Gemini cache reconcile: {len(live)} live, {len(live) - len(orphans)} kept, "
            f"{len(orphans)} orphaned → deleting…"
        )
        for name in orphans:
            await self._delete_gemini_cache(name)  # logs each deletion

    URL_PATTERN = re.compile(r'https?://[^\s\)\]<>\"\'`]+[^\s\.\,\)\]<>\"\'`:]')
    URL_EXTRACT_MAX = 3
    URL_EXTRACT_CHAR_CAP = 20000

    @classmethod
    def _extract_urls(cls, text: str) -> list[str]:
        """Pull HTTP(S) URLs from text, deduped, capped at URL_EXTRACT_MAX."""
        if not text:
            return []
        seen: dict[str, None] = {}
        for url in cls.URL_PATTERN.findall(text):
            if url not in seen:
                seen[url] = None
            if len(seen) >= cls.URL_EXTRACT_MAX:
                break
        return list(seen.keys())

    async def _tavily_extract(self, urls: list[str]) -> dict[str, str]:
        """Fetch text content for a list of URLs via Tavily's extract endpoint.

        Returns {url: extracted_text}. URLs that fail to extract are silently
        skipped — better to give the model partial context than to error out.
        """
        if not self.tavily_client or not urls:
            return {}
        try:
            result = await asyncio.to_thread(
                self.tavily_client.extract,
                urls=urls,
            )
        except Exception:
            return {}
        out: dict[str, str] = {}
        for r in result.get("results", []) if isinstance(result, dict) else []:
            url = r.get("url", "")
            content = r.get("raw_content") or r.get("content") or ""
            if url and content:
                out[url] = content[:self.URL_EXTRACT_CHAR_CAP]
        return out

    async def _augment_with_url_extracts(self, messages: list[dict]) -> None:
        """If the latest user turn references URLs, fetch their text content via
        Tavily and append the extracted bodies to that message in-place. Lets
        text-only models (Deepseek) read links the user pastes, and gives Claude
        a deterministic copy of the page even though it could also web_search.
        """
        if not messages or not self.tavily_client:
            return
        last = messages[-1]
        if last.get("role") != "user":
            return
        content = last.get("content", "")
        if isinstance(content, str):
            text_for_url_scan = content
        elif isinstance(content, list):
            text_for_url_scan = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            return
        urls = self._extract_urls(text_for_url_scan)
        if not urls:
            return
        extracts = await self._tavily_extract(urls)
        if not extracts:
            return
        block_parts = ["\n\n---\n[Auto-fetched content from URLs in the user's message — for grounding, treat as freshly retrieved web pages:]"]
        for url, body in extracts.items():
            block_parts.append(f"\n## {url}\n{body}\n")
        block_parts.append("---\n")
        extract_block = "\n".join(block_parts)
        if isinstance(content, str):
            last["content"] = content + extract_block
        else:
            last["content"] = list(content) + [{"type": "text", "text": extract_block}]

    @staticmethod
    def _has_cjk(text: str) -> bool:
        """Check if text contains Chinese/Japanese/Korean characters."""
        return any('\u4e00' <= c <= '\u9fff'  # CJK Unified Ideographs
                   or '\u3400' <= c <= '\u4dbf'  # CJK Extension A
                   or '\uf900' <= c <= '\ufaff'  # CJK Compatibility Ideographs
                   for c in text)

    def _estimate_confidence(self, message_text: str, provider: ModelProvider) -> float:
        """
        Estimate how well-suited a model is for this message.
        Returns 0.0-1.0. This is a heuristic, NOT an LLM call.
        """
        score = 0.5
        text_lower = message_text.lower()
        word_count = len(message_text.split())
        has_cjk = self._has_cjk(message_text)

        if provider.name == "Claude":
            # Claude excels at: complex questions, code review, nuance, creative, analysis
            if word_count > 100:
                score += 0.15
            if any(kw in text_lower for kw in [
                'explain', 'analyze', 'compare', 'review', 'design',
                'architecture', 'tradeoff', 'nuance', 'creative', 'write'
            ]):
                score += 0.1
            if any(kw in text_lower for kw in [
                'code', 'debug', 'refactor', 'implement', 'function',
                'class', 'algorithm', 'bug', 'error'
            ]):
                score += 0.1
            if '```' in message_text:
                score += 0.1
            # CJK penalty - Deepseek is stronger here
            if has_cjk:
                score -= 0.15
            # Cost penalty - Claude must "earn" selection
            score -= 0.2

        elif provider.name == "Deepseek":
            # Deepseek excels at: quick answers, factual, simple code, casual chat, CJK languages
            if word_count < 30:
                score += 0.15
            if any(kw in text_lower for kw in [
                'what is', 'how do', 'define', 'translate', 'list',
                'name', 'when', 'where', 'who', 'quick',
                'mandarin', 'chinese', '中文'
            ]):
                score += 0.1
            if '?' in message_text and word_count < 20:
                score += 0.1
            # CJK bonus - trained on deeper Chinese internet data
            if has_cjk:
                score += 0.2
            # Cost bonus - cheap = preferred for routine tasks
            score += 0.15

        elif provider.name == "Gemini":
            # Gemini excels at: novel/abstract reasoning, long-context synthesis,
            # multi-step math, pattern-finding, and questions where the answer
            # requires recombining ideas non-obviously (ARC-AGI-2 territory).
            # Also strong on math/scientific reasoning (GPQA Diamond leader).
            if word_count > 60:
                score += 0.1  # long prompts benefit from its context handling
            if any(kw in text_lower for kw in [
                'reason', 'reasoning', 'puzzle', 'riddle', 'pattern',
                'abstract', 'prove', 'proof', 'derive', 'derivation',
                'novel', 'unusual', 'counterintuitive', 'paradox',
                'physics', 'chemistry', 'biology', 'theorem', 'lemma',
                'integral', 'derivative', 'limit', 'differential',
            ]):
                score += 0.15
            # Math notation / equations — Gemini is strong here
            if '$' in message_text or '\\' in message_text or '∫' in message_text or '∑' in message_text:
                score += 0.1
            # Very long context — synthesizing across a lot of material
            if word_count > 200:
                score += 0.1
            # Mid-tier cost — cheaper than Claude, pricier than Deepseek.
            # Smaller cost penalty than Claude, no bonus like Deepseek.
            score -= 0.08

        elif provider.name == "Mistral":
            # Mistral (Mari) — strongest open model on French/European languages.
            # French isn't script-detectable (Latin), so trigger on intent, not
            # text; otherwise stay out of the tuned core router. Cheap on Fireworks.
            if any(kw in text_lower for kw in (
                'french', 'français', 'francais', 'en français',
                'translate to french', 'how do you say', 'conjugat', 'liaison',
            )):
                score += 0.35
            else:
                score -= 0.15
            score += 0.1

        elif provider.name == "Qwen":
            # Qwen (Rei) — frontier code/math at Fireworks prices. Competes ONLY
            # in its lane (routine code/math that doesn't need Opus); stays out of
            # general chat so it can't muscle Deepseek off routine replies.
            in_lane = '```' in message_text or any(kw in text_lower for kw in (
                'code', 'debug', 'refactor', 'implement', 'function', 'regex',
                'math', 'solve', 'equation', 'integral', 'derivative', 'proof',
            ))
            if in_lane:
                score += 0.2
                score += 0.15  # cheap — credited only in-lane
                if '```' in message_text:
                    score += 0.1
                # hand careful/complex/long work back to Claude
                if word_count > 120 or any(kw in text_lower for kw in (
                    'review', 'architecture', 'design', 'tradeoff', 'nuance',
                )):
                    score -= 0.4
            else:
                score -= 0.2

        elif provider.name == "GLM":
            # GLM (Asuka) — its agentic/tool-use niche isn't what this chat bot
            # does, and a second cheap coder would just split Qwen's vote.
            # Override-only via !glm/!asuka: keep it out of the auto-router.
            score -= 0.5

        return max(0.0, min(1.0, score))

    @staticmethod
    def _avg_cost_per_million(provider: ModelProvider) -> float:
        """Average of input + output $/M as a single tiebreak scalar.
        Lower = cheaper = preferred when scores are tied."""
        return (provider.input_cost_per_million + provider.output_cost_per_million) / 2

    async def _select_model(self, message: discord.Message, guild_id: int) -> tuple[ModelProvider, str]:
        """Select which model should respond to this message.
        Returns (provider, routing_reason) tuple."""
        enabled = [p for p in self.providers if p.enabled]
        if not enabled:
            # Shouldn't happen — on_message gates earlier — but fail gracefully.
            return self.claude_provider, "No providers enabled; defaulting to Claude."

        # Hard rule: images require a vision-capable provider. Filter the pool
        # down to vision-capable providers and let normal scoring pick among them.
        has_images = any(
            any(a.filename.lower().endswith(ext) for ext in CONFIG.image_types)
            for a in message.attachments
        )
        if has_images:
            vision_pool = [p for p in enabled if p.supports_vision]
            if vision_pool:
                enabled = vision_pool
                vision_routing_note = (
                    " (filtered to vision-capable providers because the message "
                    "contains image attachments)"
                )
            else:
                # Fall back to whatever's enabled even though none can see.
                vision_routing_note = " (no vision-capable provider enabled — image ignored)"
        else:
            vision_routing_note = ""

        # Only one provider available in the (possibly filtered) pool?
        # Deliberate exception to "completions_mode = override-only": if the
        # base/sim head is the ONLY thing enabled (a sim-only box), plain
        # messages route to it rather than failing — the "never argmax it" rule
        # exists so a cold base model can't beat an available chat head, which
        # doesn't apply when it's the only head up. With any chat head enabled,
        # len>1 here and the auto_pool filter below keeps sim out of the argmax.
        if len(enabled) == 1:
            only = enabled[0]
            return only, f"Only {only.name} is available{vision_routing_note}."

        # User preference for this channel?
        channel_id = message.channel.id
        parent_id = getattr(message.channel, 'parent_id', None)
        pref = self.channel_preferences.get(channel_id) or self.channel_preferences.get(parent_id)
        # Registry-driven: every provider id is a valid pref. The canonical
        # three resolve identically to before; qwen/glm/mistral are now
        # selectable too (additive — only matters if someone sets them).
        pref_map = {p.id: p for p in self.providers}
        if pref in pref_map and pref_map[pref] in enabled:
            chosen = pref_map[pref]
            return chosen, f"User set channel preference to {chosen.name} (!prefer {pref}).{vision_routing_note}"

        # Global default override?
        if CONFIG.default_model in pref_map and pref_map[CONFIG.default_model] in enabled:
            chosen = pref_map[CONFIG.default_model]
            return chosen, f"Global default model is set to {chosen.name}.{vision_routing_note}"

        # Auto-select via confidence heuristic — three-way argmax with a
        # cost-based tiebreak (cheaper wins ties). The scoring functions
        # already encode each model's cost via per-provider penalties/bonuses,
        # so this is just final disambiguation.
        text = message.content or ""
        # Simulator/base heads (Phase 7, completions_mode) are OVERRIDE-ONLY:
        # never argmax'd (a cold/garrulous base model shouldn't win auto-routing)
        # — only reachable via !dummy or a !prefer-pinned channel, both handled
        # above. (The sim-only-box case already returned at the len==1 shortcut.)
        # `or enabled` is pure defense for a hypothetical all-completions pool.
        auto_pool = [p for p in enabled if not p.completions_mode] or enabled
        scores = {p.name: self._estimate_confidence(text, p) for p in auto_pool}
        # Sort: primary = score desc, secondary = cost asc (cheaper wins ties).
        ranked = sorted(
            auto_pool,
            key=lambda p: (-scores[p.name], self._avg_cost_per_million(p)),
        )
        winner = ranked[0]
        score_str = ", ".join(f"{p.name} {scores[p.name]:.2f}" for p in ranked)
        reason = (
            f"Auto-routed by heuristic: {score_str}. {winner.name} wins "
            f"(cost-based tiebreak applies on ties).{vision_routing_note}"
        )
        return winner, reason

    async def _generate_openai_compatible_response(
        self,
        client,
        provider: ModelProvider,
        guild_id: int,
        messages: list[dict],
        system: str,
        thinking: bool = False,
        reading_material: Optional["ReadingMaterial"] = None,
    ) -> tuple[str, list[str], str]:
        """Generate response using any OpenAI-compatible provider (Deepseek, Gemini).

        Uses provider.* quirks flags to handle per-provider differences:
        - supports_vision: keep image content if True, strip otherwise
        - requires_reasoning_echo: echo reasoning_content on prior assistant turns
        - disables_thinking_by_default: send extra_body to opt out when thinking=False

        Reading material handling (bookclub mode):
        - Gemini: create / reuse an explicit cachedContents entry via native
          API, reference it via extra_body={"cached_content": "..."} so the
          fic isn't re-uploaded every turn.
        - Other providers (Deepseek): prepend the fic to the system message.
          Deepseek's server-side prefix caching makes this cheap automatically.

        Returns (text, reactions, reasoning_content). reasoning_content is empty
        unless thinking mode was enabled and the provider returned a reasoning block.
        """
        # Convert to OpenAI format. Only strip images for text-only providers;
        # vision-capable providers (Gemini) keep the image blocks intact.
        if provider.supports_vision:
            openai_messages = list(messages)
        else:
            openai_messages = self._strip_images_from_messages(messages)
        openai_messages = self._convert_messages_to_openai_format(openai_messages)

        # When thinking is on AND the provider requires it, echo reasoning_content
        # on every prior assistant turn (empty string fine for ones we don't have).
        # Gemini's shim handles thinking server-side so this is skipped.
        if thinking and provider.requires_reasoning_echo:
            for msg in openai_messages:
                if msg.get("role") == "assistant":
                    msg_id = msg.get("_msg_id")
                    cached = self._get_reasoning(msg_id) if msg_id is not None else ""
                    msg["reasoning_content"] = cached

        # Handle reading material — always inline for now.
        #
        # NOTE on Gemini caching: the OpenAI shim's docs claim explicit caching
        # works via extra_body={"cached_content": "cachedContents/..."}, but the
        # API actually rejects it with HTTP 400 "Unknown name 'cached_content':
        # Cannot find field." We have working code in _create_gemini_cache that
        # creates a cache via the native API — but referencing it from the shim
        # doesn't work. Proper fix is to route Gemini-with-reading-material
        # through the native generateContent endpoint instead of the shim. Until
        # then, both Deepseek and Gemini get the fic injected into the system
        # message every turn. Deepseek's server-side cache makes that ~free
        # (~99% off). Gemini pays full $4/M-above-200k until we add the native
        # path — see Issue/TODO: "Native Gemini chat path for cached_content".
        if reading_material is not None:
            system = self._build_reading_material_system_block(reading_material) + "\n\n" + system

        # Prepend system message (OpenAI uses it as first message)
        openai_messages.insert(0, {"role": "system", "content": system})

        # Include web search tool if Tavily is available
        tools = self.OPENAI_COMPATIBLE_TOOLS if self.tavily_client else None

        try:
            api_kwargs = {
                "model": provider.model_id,
                "max_tokens": provider.max_tokens,
                "messages": self._strip_internal_keys(openai_messages),
            }
            extra_body: dict = {}
            if not thinking and provider.disables_thinking_by_default:
                # Provider has thinking on by default and needs the disable kwarg.
                # We reconstruct history from plain Discord text, so we can't
                # preserve reasoning blocks anyway — disable thinking instead.
                extra_body["thinking"] = {"type": "disabled"}
            if extra_body:
                api_kwargs["extra_body"] = extra_body
            if tools:
                api_kwargs["tools"] = tools

            response = await asyncio.to_thread(
                client.chat.completions.create,
                **api_kwargs,
            )

            # Track usage. DeepSeek exposes prompt_cache_hit_tokens for
            # server-side auto-cached input; subtract from prompt_tokens to
            # get the uncached portion. Other providers / no cache → 0.
            cached_hit = getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0
            uncached_input = max(0, response.usage.prompt_tokens - cached_hit)
            provider.record_usage(
                uncached_input,
                response.usage.completion_tokens,
                cached_input_tokens=cached_hit,
            )
            provider.total_requests += 1

            # Handle tool calls (max 3 rounds to prevent loops)
            tool_rounds = 0
            while response.choices[0].message.tool_calls and tool_rounds < 3:
                tool_rounds += 1
                assistant_msg = response.choices[0].message

                # Add assistant message with tool calls to conversation
                tool_assistant: dict = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        }
                        for tc in assistant_msg.tool_calls
                    ]
                }
                if thinking and provider.requires_reasoning_echo:
                    tool_assistant["reasoning_content"] = (
                        getattr(assistant_msg, "reasoning_content", None) or ""
                    )
                openai_messages.append(tool_assistant)

                # Execute each tool call. Routed through _search_for so each
                # provider uses its configured backend (Deepseek → Tavily,
                # Gemini → native google_search grounding, etc.).
                for tool_call in assistant_msg.tool_calls:
                    if tool_call.function.name == "web_search":
                        import json as _json
                        args = _json.loads(tool_call.function.arguments)
                        query = args.get("query", "")
                        search_result = await self._search_for(provider, query)
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_result.text,
                        })

                # Continue conversation with tool results — refresh messages
                # since api_kwargs holds a stripped copy.
                api_kwargs["messages"] = self._strip_internal_keys(openai_messages)
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    **api_kwargs,
                )

                # Track additional usage (cache-aware, see above)
                cached_hit = getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0
                uncached_input = max(0, response.usage.prompt_tokens - cached_hit)
                provider.record_usage(
                    uncached_input,
                    response.usage.completion_tokens,
                    cached_input_tokens=cached_hit,
                )

            msg0 = response.choices[0]
            response_text = msg0.message.content or ""
            # Empty content is a real failure mode — a transient blank completion,
            # a China-API content filter, or a tool-call the 3-round cap cut off.
            # It used to surface as a silent '' → a dropped panel member with no
            # second chance (why a first !research could lose DeepSeek while the
            # next one worked). Log WHY, then retry once.
            if not response_text.strip():
                fin = getattr(msg0, "finish_reason", "?")
                had_tc = bool(getattr(msg0.message, "tool_calls", None))
                print(f"⚠️  {provider.name} returned empty content "
                      f"(finish_reason={fin}, pending_tool_calls={had_tc}) — retrying once")
                retry_kwargs = dict(api_kwargs)
                if had_tc and "tools" in retry_kwargs:
                    # Loop cap hit mid-search — force a text answer this round.
                    retry_kwargs["tool_choice"] = "none"
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create, **retry_kwargs,
                    )
                    cached_hit = getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0
                    provider.record_usage(
                        max(0, response.usage.prompt_tokens - cached_hit),
                        response.usage.completion_tokens,
                        cached_input_tokens=cached_hit,
                    )
                    response_text = response.choices[0].message.content or ""
                except Exception as e:
                    print(f"⚠️  {provider.name} empty-retry failed: {e}")
            reasoning = (
                getattr(response.choices[0].message, "reasoning_content", None) or ""
            ) if (thinking and provider.requires_reasoning_echo) else ""

            # Process notes and reactions (same patterns as Claude)
            note_pattern = r'\[note:\s*([^:]+):\s*([^\]]+)\]'
            for match in re.finditer(note_pattern, response_text):
                key = match.group(1).strip()
                value = match.group(2).strip()
                self.manager.memories[guild_id].working.add(key, value)
            response_text = re.sub(note_pattern, '', response_text)

            reactions = []
            reaction_pattern = r'\[react:\s*([^\]]+)\]'
            for match in re.finditer(reaction_pattern, response_text):
                reactions.append(match.group(1).strip())
            response_text = re.sub(reaction_pattern, '', response_text).strip()
            # Clean up verbose formatting (both Deepseek and Gemini tend to over-format)
            response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)  # Triple+ newlines → double
            response_text = re.sub(r'\n\n+(#+\s)', r'\n\1', response_text)  # Extra newlines before headers
            response_text = re.sub(r'\n\n+(\*\*[^*]+\*\*:)', r'\n\1', response_text)  # Extra newlines before bold labels
            response_text = re.sub(r'  +', ' ', response_text)

            return response_text, reactions, reasoning

        except Exception as e:
            return f"{provider.name} Error: {e}", [], ""

    # =========================================================================
    # Simulator mode (Phase 7, §9) — base-model transcript completion
    # =========================================================================
    # A base model doesn't take chat turns; it continues a transcript. These
    # three methods ARE the feature (§9.3): _format_transcript renders channel
    # history as an IRC/script log + a continuation cue, the /completions call
    # continues it, and _parse_transcript_turn cuts that back to one speaker's
    # line. Cost + carbon ride the same record_usage path as the chat heads
    # (§9.2), so a !dummy turn lands in !cost exactly like a chat turn.

    @staticmethod
    def _transcript_flatten(content) -> str:
        """Flatten a message's content (str | list of blocks) to plain text,
        dropping non-text blocks (images are already stripped upstream)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
        return str(content or "")

    def _format_transcript(
        self,
        messages: list[dict],
        preamble: str,
        bot_speaker: str,
    ) -> tuple[str, str]:
        """Render history as an IRC/script log and append a continuation cue.

        Returns (prompt, bot_speaker). Each turn becomes a `<speaker> body`
        line: user turns already carry a `DisplayName:` prefix (optionally
        behind a `[replying to …]` block) and bot turns a `[Model]` tag — both
        normalize to the `<speaker>` form. The prompt ends on a dangling
        `<bot_speaker>` so the base model continues AS the bot; the stop
        sequences + _parse_transcript_turn keep it to a single line.
        """
        # Name caps are generous (80) so webhook/PluralKit proxy names — which
        # can exceed Discord's 32-char native cap — still attribute correctly
        # instead of leaking the name into the body.
        reply_re = re.compile(r'^\s*\[replying to', re.IGNORECASE)
        user_name_re = re.compile(r'^\s*([^\n:]{1,80}?):\s')
        bot_tag_re = re.compile(r'^\s*\[([^\]\n]{1,80})\]\s*')

        lines: list[str] = []
        for msg in messages:
            # Recover the body text per content-BLOCK: fetch_thread_history puts
            # a "[replying to …]" reply-context block ahead of the author block,
            # and the quoted text is arbitrary (embedded ']' / newlines), so it
            # must never be flattened-then-regex'd into the speaker line. Drop a
            # leading reply block (the replied-to message is usually already in
            # the log); parse the speaker from the real author block.
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            else:
                texts = [self._transcript_flatten(content)]
            body_blocks = []
            for t in texts:
                if not body_blocks and reply_re.match(t):
                    continue  # drop the leading reply-context block
                body_blocks.append(t)
            text = "\n".join(body_blocks).strip()
            if not text:
                continue
            if msg.get("role") == "assistant":
                m = bot_tag_re.match(text)
                speaker = m.group(1).strip() if m else bot_speaker
                body = text[m.end():].strip() if m else text
            else:
                m = user_name_re.match(text)
                speaker = m.group(1).strip() if m else "user"
                body = text[m.end():].strip() if m else text
            if not body:
                continue
            lines.append(f"<{speaker}> {body}")

        transcript = "\n".join(lines)
        header = (preamble or "").strip()
        framing = (
            "The following is a log of an ongoing group chat channel. Continue "
            f"it by writing the next line for <{bot_speaker}>, and only that line."
        )
        prompt = "\n\n".join(p for p in (header, framing, transcript) if p)
        prompt += f"\n<{bot_speaker}>"
        return prompt, bot_speaker

    def _parse_transcript_turn(self, raw: str, bot_speaker: str) -> str:
        """Cut a raw /completions continuation back to one speaker's line.

        Belt-and-suspenders to the server-side `stop`: strip any echoed leading
        speaker tag, then cut at the first NEW speaker line (`<name>` or
        `[name]` at line start) or a paragraph gap."""
        text = raw or ""
        esc = re.escape(bot_speaker)
        # Strip an echoed leading tag for the bot speaker (<Dummy> / [Dummy] / Dummy:).
        text = re.sub(rf'^\s*(?:<{esc}>|\[{esc}\]|{esc}:)\s*', '', text, flags=re.IGNORECASE)
        # Cut at the next speaker line. Only the `<name>` form — the one
        # _format_transcript actually emits — counts; matching `[name]` too
        # would over-cut on ordinary bracketed content a base model produces
        # (code literals `[1, 2, 3]`, citations `[1]`, markdown footnotes).
        m = re.search(r'\n\s*<[^>\n]{1,80}>', text)
        if m:
            text = text[:m.start()]
        # ... or a paragraph gap (matches the "\n\n\n" stop).
        text = re.split(r'\n{3,}', text)[0]
        return text.strip()

    # Query-driven search grounding for the simulator path (§9.2 / §9.5). A base
    # model can't request the web_search tool itself, so we detect a search
    # intent in the latest user turn HERE and fold results into the preamble.
    # Two deliberately conservative triggers so we don't search every turn:
    #   • explicit inline directive  [search: <query>]
    #   • a leading-question heuristic — a short factual question / lookup cue
    _SIM_SEARCH_DIRECTIVE_RE = re.compile(r'\[search:\s*([^\]]+)\]', re.IGNORECASE)
    _SIM_SEARCH_LEAD_RE = re.compile(
        r'^(?:who|what|whats|when|where|why|how|which|whose|is|are|was|were|'
        r'does|do|did|can|could|should|will|would|has|have)\b',
        re.IGNORECASE,
    )
    _SIM_SEARCH_CUE_RE = re.compile(
        r'\b(?:search (?:for|up)|look(?:ing)? up|google|latest|current(?:ly)?|'
        r'news (?:on|about)|who is|what is|how much|how many)\b',
        re.IGNORECASE,
    )

    def _search_backend_available(self, provider: ModelProvider) -> bool:
        """True if provider.search_backend can actually run a query right now —
        mirrors _search_for's backend selection without performing a search, so
        the simulator never folds a "no backend configured" sentinel into the
        preamble."""
        if provider.search_backend == "google_native" and os.getenv("GEMINI_API_KEY"):
            return True
        return provider.search_backend is not None and bool(self.tavily_client)

    def _sim_search_query(self, messages: list[dict]) -> Optional[str]:
        """Decide whether the latest user turn warrants a web search and, if so,
        return the query (else None). Read-only — never mutates the transcript.

        Mirrors _format_transcript's block handling: drop a leading
        "[replying to …]" block, then strip the "Name:" speaker prefix before
        applying the directive / question heuristics (see the regexes above)."""
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if last_user is None:
            return None
        content = last_user.get("content", "")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
        else:
            texts = [content if isinstance(content, str) else ""]
        body_blocks: list[str] = []
        for t in texts:
            if not body_blocks and re.match(r'^\s*\[replying to', t, re.IGNORECASE):
                continue  # drop the leading reply-context block
            body_blocks.append(t)
        text = "\n".join(body_blocks).strip()
        if not text:
            return None
        m = re.match(r'^\s*([^\n:]{1,80}?):\s', text)  # strip "Name:" speaker prefix
        if m:
            text = text[m.end():].strip()
        if not text:
            return None
        # Explicit directive wins — exact, operator-/persona-authored intent.
        d = self._SIM_SEARCH_DIRECTIVE_RE.search(text)
        if d:
            return d.group(1).strip() or None
        # Heuristic: cap length so a long monologue ending on '?' doesn't fire.
        if len(text) > 300:
            return None
        is_question = text.rstrip().endswith("?") and bool(self._SIM_SEARCH_LEAD_RE.match(text))
        if is_question or self._SIM_SEARCH_CUE_RE.search(text):
            return text.strip()
        return None

    async def _generate_simulator_response(
        self,
        provider: ModelProvider,
        guild_id: int,
        messages: list[dict],
        system: str,
        reading_material: Optional["ReadingMaterial"] = None,
        force_speaker: Optional[str] = None,
    ) -> tuple[str, list[str], str]:
        """Generate a turn by CONTINUING a transcript via /completions (§9.3).

        The base-model counterpart to _generate_openai_compatible_response:
        same client object, same usage/carbon accounting, but it calls
        client.completions.create on a rendered transcript instead of
        client.chat.completions.create on chat messages. Returns the same
        (text, reactions, reasoning) tuple (reasoning always "").

        Grounding (§9.2): reading material and pasted-URL extracts fold in — the
        latter because _augment_with_url_extracts has already rewritten the user
        turn before dispatch, so it rides into the transcript for free. Query-
        driven web_search (the chat heads' tool-call loop) can't run here — a
        base model can't tool-call — so when the operator opts in (sim_search)
        we detect a search intent ourselves and PREPEND the snippets to the
        preamble (the query→search→preamble fold). Off by default."""
        client = self.clients.get(provider.id)
        if client is None:
            return f"{provider.name} Error: simulator endpoint not configured", [], ""

        # Base model = text-only transcript; drop image blocks.
        msgs = self._strip_images_from_messages(messages)

        # Fold reading material (bookclub) into the preamble, like the chat path.
        preamble = system
        if reading_material is not None:
            preamble = (
                self._build_reading_material_system_block(reading_material)
                + "\n\n" + preamble
            )

        # Query-driven search grounding (§9.2 / §9.5). A base model can't
        # tool-call, so when the operator opts in (provider.sim_search) AND a
        # backend is actually available, detect a search intent in the latest
        # user turn, run the SearchBackend ourselves, and PREPEND the snippets
        # to the preamble (the chat heads append a tool result instead). Off by
        # default: sim_search defaults False, so the standard Dummy Plug never
        # auto-searches — pasted-URL grounding still rides in via the URL fold.
        if provider.sim_search and self._search_backend_available(provider):
            query = self._sim_search_query(msgs)
            if query:
                result = await self._search_for(provider, query)
                snippet = (result.text or "").strip() if result else ""
                if snippet:
                    grounding = (
                        f'[Web search results for "{query}" — freshly '
                        "retrieved, treat as current web context:]\n" + snippet
                    )
                    preamble = f"{grounding}\n\n{preamble}" if preamble else grounding

        bot_speaker = force_speaker or provider.name
        prompt, bot_speaker = self._format_transcript(msgs, preamble, bot_speaker)

        # Resolve sampler knobs (§9.3). Standard OpenAI params go on the call;
        # the non-standard base-model knobs ride in extra_body (vLLM/Fireworks).
        # A None value drops the knob (server default). `stop` is special-cased.
        sampler = dict(provider.sim_sampler or {})
        STD = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
        EXTRA = ("top_k", "min_p", "top_a", "repetition_penalty")
        api_kwargs: dict = {
            "model": provider.model_id,
            "prompt": prompt,
            "max_tokens": provider.max_tokens,
        }
        for k in STD:
            if sampler.get(k) is not None:
                api_kwargs[k] = sampler[k]
        if sampler.get("stop"):
            api_kwargs["stop"] = sampler["stop"]
        extra_body = {k: sampler[k] for k in EXTRA if sampler.get(k) is not None}
        if extra_body:
            api_kwargs["extra_body"] = extra_body

        try:
            response = await asyncio.to_thread(
                client.completions.create,
                **api_kwargs,
            )
            # Usage / carbon — identical accounting to the chat path (§9.2).
            usage = getattr(response, "usage", None)
            if usage is not None:
                cached_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
                uncached_input = max(0, (usage.prompt_tokens or 0) - cached_hit)
                provider.record_usage(
                    uncached_input,
                    usage.completion_tokens or 0,
                    cached_input_tokens=cached_hit,
                )
            provider.total_requests += 1

            raw = response.choices[0].text or ""
            response_text = self._parse_transcript_turn(raw, bot_speaker)

            # Note / reaction parity with the chat heads (cheap; a base model
            # rarely emits these, but a persona preamble might steer one to).
            note_pattern = r'\[note:\s*([^:]+):\s*([^\]]+)\]'
            for match in re.finditer(note_pattern, response_text):
                self.manager.memories[guild_id].working.add(
                    match.group(1).strip(), match.group(2).strip()
                )
            response_text = re.sub(note_pattern, '', response_text)

            reactions = []
            reaction_pattern = r'\[react:\s*([^\]]+)\]'
            for match in re.finditer(reaction_pattern, response_text):
                reactions.append(match.group(1).strip())
            response_text = re.sub(reaction_pattern, '', response_text).strip()

            if not response_text:
                response_text = "*(the simulator returned an empty continuation)*"
            return response_text, reactions, ""

        except Exception as e:
            return f"{provider.name} Error: {e}", [], ""

    async def _generate_gemini_native_response(
        self,
        guild_id: int,
        messages: list[dict],
        system: str,
        thinking: bool = False,
        reading_material: Optional["ReadingMaterial"] = None,
    ) -> tuple[str, list[str], str]:
        """Generate Gemini response via the native generateContent endpoint.

        Used when reading_material is loaded — only path that can reference
        cachedContent (the OpenAI shim rejects extra_body cached_content with
        HTTP 400 as of May 2026). Bypasses the openai SDK entirely, talking
        to the native API via aiohttp.

        For non-bookclub Gemini calls we still use the OpenAI shim because:
        (a) the shim works fine without caching needs, (b) it keeps the
        Deepseek/Gemini code path uniform, and (c) less native code to
        maintain.

        Returns (text, reactions, reasoning_content). reasoning_content is
        always empty — native API surfaces thinking differently and we
        don't need it for our Discord-history-based context.
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            return "Gemini Error: GEMINI_API_KEY not set", [], ""

        # If we have reading material, we'll use the cache. The cache holds
        # the fic AND systemInstruction AND tools — chat calls referencing
        # a cachedContent cannot set those fields themselves (Gemini API
        # rejects with HTTP 400). So we either:
        # - cache hit → chat call only sets cachedContent + contents + config
        # - cache miss / no material → chat call sets systemInstruction +
        #   tools inline.
        cache_tools = [{"google_search": {}}]
        # The system we'd bake into a cache for this material. Includes the
        # fic as a system block (Gemini's cache treats systemInstruction +
        # contents together as the cached prefix).
        cache_system_text = system
        # Explicit cachedContents only in marathon mode (self.gemini_explicit_cache).
        # Default (inline): cache_name stays None → the fic is inlined below and
        # Google's implicit caching supplies the read discount, with no storage bill.
        cache_name: Optional[str] = None
        if reading_material is not None and self.gemini_explicit_cache:
            cache_name = await self._ensure_gemini_cache(
                reading_material,
                system_text=cache_system_text,
                tools=cache_tools,
            )

        # Convert messages: Anthropic-style → Gemini native format.
        # Drop the system role (Gemini takes that as a separate field, or it
        # lives in the cache).
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        gemini_contents = self._convert_messages_to_gemini_format(
            self._strip_internal_keys(non_system_messages)
        )

        body: dict = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": self.gemini_provider.max_tokens,
            },
        }
        if cache_name:
            # Cache holds systemInstruction + tools + fic. Chat call must
            # not set any of those (HTTP 400 otherwise).
            body["cachedContent"] = cache_name
        else:
            # No cache (no material loaded, or cache creation failed).
            # Inline the fic in systemInstruction if we have one.
            if reading_material is not None:
                inline_system = (
                    self._build_reading_material_system_block(reading_material)
                    + "\n\n"
                    + system
                )
            else:
                inline_system = system
            body["systemInstruction"] = {"parts": [{"text": inline_system}]}
            body["tools"] = cache_tools

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_provider.model_id}:generateContent"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": gemini_key,
        }

        async def post_body(b: dict) -> tuple[Optional[dict], Optional[str]]:
            """Returns (data, error_text). Exactly one is non-None."""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=b, timeout=180) as resp:
                        if resp.status != 200:
                            return None, await resp.text()
                        return await resp.json(), None
            except asyncio.TimeoutError:
                return None, "TIMEOUT"
            except aiohttp.ClientError as e:
                return None, f"NETWORK: {e}"
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

        data, err_text = await post_body(body)

        # If the cache reference caused the 400 (stale cache from an older
        # version of the code that didn't bake in system/tools, etc.),
        # invalidate the handle and retry once with everything inline.
        if data is None and cache_name and err_text and (
            "cachedContent" in err_text
            or "CachedContent" in err_text
            or "CACHE" in err_text.upper()
        ):
            print(f"⚠️  Gemini cache {cache_name} rejected; deleting it + retrying inline")
            _gk = self._gemini_cache_key()
            reading_material.cache_handles.pop(_gk, None)
            reading_material.cache_expires_at.pop(_gk, None)
            reading_material.cache_handles.pop("Gemini", None)  # legacy untagged
            reading_material.cache_expires_at.pop("Gemini", None)
            await self._delete_gemini_cache(cache_name)  # don't orphan the rejected cache
            self.manager.mark_dirty()
            # Rebuild body without cache reference
            inline_system = (
                self._build_reading_material_system_block(reading_material)
                + "\n\n"
                + system
            ) if reading_material is not None else system
            retry_body = {
                "contents": gemini_contents,
                "generationConfig": body["generationConfig"],
                "systemInstruction": {"parts": [{"text": inline_system}]},
                "tools": cache_tools,
            }
            data, err_text = await post_body(retry_body)

        if data is None:
            if err_text == "TIMEOUT":
                return "Gemini Error: native API request timed out", [], ""
            if err_text and err_text.startswith("NETWORK:"):
                return f"Gemini Error: network error: {err_text[8:].strip()}", [], ""
            return f"Gemini Error: HTTP error: {(err_text or 'unknown')[:500]}", [], ""

        # Parse response
        candidates = data.get("candidates", [])
        if not candidates:
            return "Gemini Error: no candidates in response", [], ""
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        response_text = "".join(p.get("text", "") for p in parts).strip()
        if not response_text:
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            return f"Gemini Error: empty response (finishReason={finish_reason})", [], ""

        # Track usage. promptTokenCount is the total (including any cached);
        # cachedContentTokenCount is the cached portion. Subtract to get the
        # uncached input billed at the regular rate.
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        cached_tokens = usage.get("cachedContentTokenCount", 0)
        uncached_input = max(0, prompt_tokens - cached_tokens)
        self.gemini_provider.record_usage(
            uncached_input,
            output_tokens,
            cached_input_tokens=cached_tokens,
        )
        self.gemini_provider.total_requests += 1
        if cached_tokens:
            # Light-touch logging so it's visible the cache is actually doing work.
            print(f"🟢 Gemini cache hit: {cached_tokens:,} cached tokens / {prompt_tokens:,} total")

        # Process notes and reactions (same patterns as other generators)
        note_pattern = r'\[note:\s*([^:]+):\s*([^\]]+)\]'
        for match in re.finditer(note_pattern, response_text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            self.manager.memories[guild_id].working.add(key, value)
        response_text = re.sub(note_pattern, '', response_text)

        reactions: list[str] = []
        reaction_pattern = r'\[react:\s*([^\]]+)\]'
        for match in re.finditer(reaction_pattern, response_text):
            reactions.append(match.group(1).strip())
        response_text = re.sub(reaction_pattern, '', response_text).strip()

        # Same formatting cleanup as the shim path
        response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)
        response_text = re.sub(r'\n\n+(#+\s)', r'\n\1', response_text)
        response_text = re.sub(r'\n\n+(\*\*[^*]+\*\*:)', r'\n\1', response_text)
        response_text = re.sub(r'  +', ' ', response_text)

        return response_text, reactions, ""

    async def _generate_gemini_vertex_response(
        self,
        guild_id: int,
        messages: list[dict],
        system: str,
        thinking: bool = False,
        reading_material: Optional["ReadingMaterial"] = None,
    ) -> tuple[str, list[str], str]:
        """Generate a Gemini response via Vertex AI (the `vertex` backend).

        ⚠️ CODE-COMPLETE BUT UNVERIFIED — needs GCP ADC
        (GOOGLE_APPLICATION_CREDENTIALS) + GOOGLE_CLOUD_PROJECT to smoke-test,
        and the exact vertexai SDK surface (grounding tool constructor,
        from_cached_content) can vary by SDK version. Mirrors
        _generate_gemini_native_response on Vertex's surface: GenerativeModel +
        vertexai.caching.CachedContent (NOT the developer-API cachedContents) +
        Google Search grounding. Returns (text, reactions, "").
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Tool, grounding
            from vertexai.caching import CachedContent
        except ImportError:
            return ("Gemini Vertex Error: google-cloud-aiplatform not installed "
                    "(`pip install google-cloud-aiplatform`).", [], "")

        vcfg = self.gemini_provider.backends.get("vertex", {})
        project = os.getenv(vcfg.get("project_env", "GOOGLE_CLOUD_PROJECT"))
        location = (os.getenv(vcfg.get("location_env", "GOOGLE_CLOUD_LOCATION"))
                    or vcfg.get("location_default", "us-central1"))
        if not project:
            return ("Gemini Vertex Error: GOOGLE_CLOUD_PROJECT not set "
                    "(project + ADC required for the vertex backend).", [], "")

        # Bake the fic into a Vertex cache only in marathon mode; otherwise inline
        # system + search on the call (implicit caching, no storage bill).
        cache_name: Optional[str] = None
        if reading_material is not None and self.gemini_explicit_cache:
            cache_name = await self._ensure_gemini_vertex_cache(
                reading_material, system_text=system
            )

        def _call():
            vertexai.init(project=project, location=location)
            non_system = [m for m in messages if m.get("role") != "system"]
            contents = self._convert_messages_to_gemini_format(
                self._strip_internal_keys(non_system)
            )
            gen_config = {"max_output_tokens": self.gemini_provider.max_tokens}
            if cache_name:
                cached = CachedContent(cached_content_name=cache_name)
                model = GenerativeModel.from_cached_content(cached_content=cached)
            else:
                if reading_material is not None:
                    inline_system = (
                        self._build_reading_material_system_block(reading_material)
                        + "\n\n" + system
                    )
                else:
                    inline_system = system
                search_tool = Tool.from_google_search_retrieval(
                    grounding.GoogleSearchRetrieval()
                )
                model = GenerativeModel(
                    self.gemini_provider.model_id,
                    system_instruction=[inline_system],
                    tools=[search_tool],
                )
            return model.generate_content(contents, generation_config=gen_config)

        try:
            resp = await asyncio.to_thread(_call)
        except Exception as e:
            return (f"Gemini Vertex Error: {type(e).__name__}: {e}", [], "")

        try:
            response_text = (resp.text or "").strip()
        except Exception:
            response_text = ""
        if not response_text:
            return ("Gemini Vertex Error: empty response", [], "")

        # Usage (cache-aware), mirroring the native path.
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
            output_tokens = getattr(um, "candidates_token_count", 0) or 0
            cached_tokens = getattr(um, "cached_content_token_count", 0) or 0
            uncached_input = max(0, prompt_tokens - cached_tokens)
            self.gemini_provider.record_usage(
                uncached_input, output_tokens, cached_input_tokens=cached_tokens
            )
            self.gemini_provider.total_requests += 1

        # Process notes + reactions + formatting (same as the other generators).
        note_pattern = r'\[note:\s*([^:]+):\s*([^\]]+)\]'
        for match in re.finditer(note_pattern, response_text):
            self.manager.memories[guild_id].working.add(
                match.group(1).strip(), match.group(2).strip()
            )
        response_text = re.sub(note_pattern, '', response_text)
        reactions: list[str] = []
        for match in re.finditer(r'\[react:\s*([^\]]+)\]', response_text):
            reactions.append(match.group(1).strip())
        response_text = re.sub(r'\[react:\s*([^\]]+)\]', '', response_text).strip()
        response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)
        response_text = re.sub(r'\n\n+(#+\s)', r'\n\1', response_text)
        response_text = re.sub(r'\n\n+(\*\*[^*]+\*\*:)', r'\n\1', response_text)
        response_text = re.sub(r'  +', ' ', response_text)
        return response_text, reactions, ""

    async def _generate_response(
        self,
        channel: discord.abc.Messageable,
        guild_id: int,
        initial_message: discord.Message = None,
        provider: ModelProvider = None,
        routing_reason: str = "",
        thinking: bool = False,
        effort: Optional[str] = None,
    ) -> tuple[str, list[str], str]:
        """
        Generate response from the selected model provider.
        Returns (response_text, list_of_emoji_reactions, reasoning_content).
        reasoning_content is empty unless thinking mode was used and the
        provider returned a reasoning trace worth caching for next turn.
        Also processes [note: key: value] tags for working memory.
        """
        if provider is None:
            provider = self.claude_provider

        # Fetch conversation from Discord
        messages = await self.manager.fetch_thread_history(channel)

        # If this is a new thread, the triggering message isn't in thread history
        if initial_message:
            content_parts = []

            if initial_message.content:
                content_parts.append({
                    "type": "text",
                    "text": f"{initial_message.author.display_name}: {initial_message.content}"
                })

            for attachment in initial_message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in CONFIG.image_types):
                    if attachment.size <= CONFIG.max_image_size_mb * 1024 * 1024:
                        try:
                            image_data = await self.manager._fetch_image_base64(attachment.url)
                            if image_data:
                                ext = attachment.filename.lower().split('.')[-1]
                                media_type = {
                                    'png': 'image/png',
                                    'jpg': 'image/jpeg',
                                    'jpeg': 'image/jpeg',
                                    'gif': 'image/gif',
                                    'webp': 'image/webp'
                                }.get(ext, 'image/png')
                                content_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data
                                    }
                                })
                        except Exception:
                            pass

                elif any(attachment.filename.lower().endswith(ext) for ext in CONFIG.text_file_types):
                    if attachment.size <= 1024 * 1024:
                        try:
                            file_content = await self.manager._fetch_text_file(attachment.url)
                            if file_content:
                                content_parts.append({
                                    "type": "text",
                                    "text": f"\n--- File: {attachment.filename} ---\n{file_content}\n--- End of {attachment.filename} ---\n"
                                })
                        except Exception:
                            pass

            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    messages.insert(0, {"role": "user", "content": content_parts[0]["text"], "_msg_id": initial_message.id})
                else:
                    messages.insert(0, {"role": "user", "content": content_parts, "_msg_id": initial_message.id})

        if not messages:
            return "I don't see any messages to respond to!", [], ""

        # If the latest user message contains URLs, fetch them via Tavily and
        # append the extracted text in-place. Lets Deepseek (text-only) read
        # links and gives Claude a deterministic copy alongside its web_search.
        await self._augment_with_url_extracts(messages)

        # Build system prompt with all context sources
        system_parts = [self._build_system_prompt(provider, routing_reason)]

        # 1. Thread index (READ-ONLY - prevents feedback loops)
        thread_index = await self.manager.fetch_thread_index(channel)
        if thread_index:
            system_parts.append(thread_index)

        # 2. Memory (both tiers)
        memory_context = self.manager.memories[guild_id].get_context_string()
        if memory_context:
            system_parts.append(memory_context)

        # 3. Gentle nudge if working memory is sparse
        working_note_count = len(self.manager.memories[guild_id].working.notes)
        if working_note_count < 3:
            system_parts.append(
                "📝 *Reminder: Your working memory is pretty empty. "
                "If anything noteworthy comes up in this conversation, "
                "jot it down with [note: key: value].*"
            )

        system = "\n\n".join(system_parts)

        # Look up reading material pinned to this channel (bookclub mode).
        # Use parent channel for threads so a thread inside a bookclub
        # channel sees the same loaded fic.
        channel_id = getattr(channel, "id", None)
        parent_id = getattr(channel, "parent_id", None)
        reading_material: Optional[ReadingMaterial] = (
            self.manager.reading_materials.get(channel_id)
            or (self.manager.reading_materials.get(parent_id) if parent_id else None)
        )
        # Gate: don't try to inject material that won't fit alongside discussion.
        # Reserve ~50k tokens for chat history + memory + system framing.
        if reading_material is not None:
            needed = reading_material.estimated_tokens + 50_000
            if provider.max_context_tokens < needed:
                reading_material = None  # silently drop for this provider

        # Simulator mode (Phase 7, §9) takes priority over the sdk_type cases:
        # a completions_mode provider talks to a BASE model by continuing a
        # transcript, not by sending chat messages. Same return tuple, so
        # message-splitting / posting / LaTeX downstream are unchanged. Reuses
        # the reading-material + URL/search context assembled above (§9.2).
        if provider.completions_mode:
            return await self._generate_simulator_response(
                provider, guild_id, messages, system,
                reading_material=reading_material,
            )

        # Dispatch on sdk_type (Phase 0 registry). Behavior is preserved:
        # - gemini + vertex backend → native Vertex path (Phase 2; handles
        #   reading material + grounding internally).
        # - gemini + reading material (developer_api) → native generateContent
        #   path (only one that can reference cachedContent + google_search).
        # - gemini without reading material → OpenAI shim (uniform with Deepseek).
        # - openai_compatible (deepseek/mistral/qwen/glm, incl. fireworks /
        #   self_hosted DeepSeek backends) → OpenAI shim. Reading material is
        #   inline-injected; DeepSeek's server-side prefix cache makes that ~free.
        # - anthropic (Claude) → native SDK with its own tool/thinking machinery.
        if provider.sdk_type == "gemini":
            if provider.backend == "vertex":
                return await self._generate_gemini_vertex_response(
                    guild_id, messages, system,
                    thinking=thinking,
                    reading_material=reading_material,
                )
            if reading_material is not None:
                return await self._generate_gemini_native_response(
                    guild_id, messages, system,
                    thinking=thinking,
                    reading_material=reading_material,
                )
            return await self._generate_openai_compatible_response(
                self.clients[provider.id], provider, guild_id, messages, system,
                thinking=thinking,
                reading_material=reading_material,
            )
        if provider.sdk_type == "openai_compatible":
            return await self._generate_openai_compatible_response(
                self.clients[provider.id], provider, guild_id, messages, system,
                thinking=thinking,
                reading_material=reading_material,
            )

        # Claude path (anthropic) — delegated to _generate_claude_response so the
        # research panel and judge can reuse Claude with the same tool / thinking
        # / cache machinery. The implementation lives in that method below.
        return await self._generate_claude_response(
            guild_id,
            messages,
            system,
            thinking=thinking,
            effort=effort,
            reading_material=reading_material,
        )

    async def _generate_claude_response(
        self,
        guild_id: int,
        messages: list[dict],
        system: str,
        thinking: bool = False,
        effort: Optional[str] = None,
        reading_material: Optional["ReadingMaterial"] = None,
        tools: Optional[list] = None,
    ) -> tuple[str, list[str], str]:
        """Generate a response from Claude via the native Anthropic SDK.

        Extracted from _generate_response so other call sites (the research
        panel/judge) can call Claude directly. tools=None gives Claude its
        default web_search tool; pass tools=[] to disable web search (e.g. the
        judge synthesis step). Returns (response_text, reactions, reasoning).
        """
        # Default to giving Claude the web search tool; [] disables it.
        if tools is None:
            tools = [{"type": "web_search_20250305", "name": "web_search"}]

        try:

            # When a reading material is loaded, send system as a list of
            # blocks with cache_control on the (very large) fic block. The
            # cache marker tells Anthropic to cache everything up through
            # that block; subsequent requests with the same fic hit the
            # cache at ~10% pricing.
            if reading_material is not None:
                fic_block_text = self._build_reading_material_system_block(reading_material)
                claude_system = [
                    {
                        "type": "text",
                        "text": fic_block_text,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": system},
                ]
            else:
                claude_system = system

            claude_kwargs = {
                "model": self.claude_provider.model_id,
                "max_tokens": self.claude_provider.max_tokens,
                "system": claude_system,
            }
            if tools:
                claude_kwargs["tools"] = tools
            if thinking:
                # Adaptive thinking on Opus 4.8: model decides depth; effort
                # controls overall thinking/acting budget. effort=None falls
                # back to the class default ("high"). xhigh/max need ≥64K
                # max_tokens or they truncate mid-thought.
                # `output_config` is a newer field that some installed anthropic
                # SDK versions reject as an unknown kwarg; pass it via extra_body
                # so the field reaches the API regardless of SDK version.
                chosen_effort = effort or self.CLAUDE_THINKING_EFFORT
                claude_kwargs["max_tokens"] = (
                    64000 if chosen_effort in ("xhigh", "max")
                    else self.CLAUDE_THINKING_MAX_TOKENS
                )
                claude_kwargs["thinking"] = {"type": "adaptive"}
                claude_kwargs["extra_body"] = {"output_config": {"effort": chosen_effort}}

            api_messages = self._strip_internal_keys(messages)
            response = await asyncio.to_thread(
                self.claude_client.messages.create,
                messages=api_messages,
                **claude_kwargs,
            )

            # Track usage. Claude's response.usage exposes cache info as
            # separate counters: cache_read_input_tokens (10% of input rate)
            # and cache_creation_input_tokens (125% of input rate — billed at
            # a small surcharge to fund the cheap reads). We bucket
            # cache_read into the cached counter and lump cache_creation into
            # regular input (a small one-time under-bill per cache lifetime).
            self._record_claude_usage(response.usage)

            # Handle tool use loop — Claude may decide to search the web.
            # When thinking is on, response.content includes thinking blocks
            # that must be passed back unchanged in the next turn.
            search_rounds = 0
            while response.stop_reason == "tool_use" and search_rounds < 3:
                tool_use_block = None
                for block in response.content:
                    if block.type == "tool_use":
                        tool_use_block = block
                        break
                if not tool_use_block:
                    break

                api_messages.append({"role": "assistant", "content": response.content})
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": "Search completed."
                    }]
                })

                response = await asyncio.to_thread(
                    self.claude_client.messages.create,
                    messages=api_messages,
                    **claude_kwargs,
                )

                self._record_claude_usage(response.usage, count_request=False)
                search_rounds += 1

            if not response.content:
                return "I received an empty response from the API.", [], ""

            # Extract text from all content blocks (may include search result
            # and thinking blocks). We discard thinking blocks from the visible
            # output — Claude doesn't require them for the next turn since we
            # always start fresh from Discord history.
            response_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "thinking":
                    continue
                if hasattr(block, 'text'):
                    response_text += block.text

            # Extract and process working memory notes
            note_pattern = r'\[note:\s*([^:]+):\s*([^\]]+)\]'
            for match in re.finditer(note_pattern, response_text):
                key = match.group(1).strip()
                value = match.group(2).strip()
                self.manager.memories[guild_id].working.add(key, value)
            response_text = re.sub(note_pattern, '', response_text)

            # Extract reactions
            reactions = []
            reaction_pattern = r'\[react:\s*([^\]]+)\]'
            for match in re.finditer(reaction_pattern, response_text):
                reactions.append(match.group(1).strip())
            response_text = re.sub(reaction_pattern, '', response_text).strip()

            # Clean up formatting
            response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)
            response_text = re.sub(r'  +', ' ', response_text)

            return response_text, reactions, ""

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                req_id = getattr(e, "request_id", None) or "<not reported>"
                return (
                    f"⚠️ Claude returned {e.status_code} (Anthropic-side blip — "
                    f"`req_id={req_id}`). Bookclub mode sends huge cache-creation "
                    f"requests that occasionally hit transient backend issues. "
                    f"Try `!claude` again — these usually clear in seconds.",
                    [],
                    "",
                )
            return f"Claude Error {e.status_code}: {e.message}", [], ""
        except anthropic.APIError as e:
            return f"Claude Error: {e}", [], ""

    async def _panel_complete(
        self,
        provider: ModelProvider,
        guild_id: int,
        messages: list[dict],
        system: str,
        *,
        claude_tools: Optional[list] = None,
        thinking: bool = False,
    ) -> str:
        """Run one model (panel member or judge) on a shared prompt; return its text.

        Reuses the normal per-provider generation paths so usage/cost tracking and
        each provider's own web search come along for free. claude_tools is
        forwarded only on the Claude path; the default (None) keeps Claude's
        web_search on. Both members AND the judge keep search so the judge can
        VERIFY post-cutoff / contested claims rather than dismissing them from
        stale priors. thinking is forwarded on the Claude path (used for the
        judge — adjudicating + deciding what to verify wants real reasoning)."""
        if provider is self.claude_provider:
            text, _, _ = await self._generate_claude_response(
                guild_id, messages, system, tools=claude_tools, thinking=thinking,
            )
            return text
        client = self.openai_compatible_clients.get(provider.name)
        if client is None:
            raise RuntimeError(f"no OpenAI-compatible client for {provider.name}")
        text, _, _ = await self._generate_openai_compatible_response(
            client, provider, guild_id, messages, system,
        )
        return text

    @staticmethod
    def _is_provider_error(provider: ModelProvider, text: str) -> bool:
        """True if text is one of our own error sentinels, not a real answer.

        The generation helpers return error STRINGS (not exceptions) on API
        failure, so the panel has to recognise them to drop a failed member."""
        if not text or not text.strip():
            return True
        head = text.lstrip()
        return (
            head.startswith(f"{provider.name} Error")
            or head.startswith("Claude Error")
            or head.startswith("⚠️ Claude returned")
        )

    async def _run_panel(
        self,
        guild_id: int,
        messages: list[dict],
        system: str,
        members: list[ModelProvider],
    ) -> list[tuple[ModelProvider, str]]:
        """Fan a shared prompt out to every panel member concurrently.

        Fault-tolerant: a member that raises, errors, or returns empty is dropped
        (logged) rather than killing the panel. Returns [(provider, answer), ...]
        for survivors, in the members' original order."""
        async def run_one(provider: ModelProvider):
            try:
                text = await self._panel_complete(provider, guild_id, list(messages), system)
            except Exception as e:
                print(f"⚠️  Panel member {provider.name} raised: {e}")
                return provider, None
            if self._is_provider_error(provider, text):
                print(f"⚠️  Panel member {provider.name} failed: {text[:120]!r}")
                return provider, None
            return provider, text.strip()

        results = await asyncio.gather(*(run_one(p) for p in members))
        return [(p, t) for p, t in results if t]

    async def _judge(
        self,
        guild_id: int,
        query: str,
        answers: list[tuple[ModelProvider, str]],
        judge: ModelProvider,
    ) -> str:
        """Have the judge model synthesise the panel's answers into one verdict."""
        panel_block = "\n\n".join(
            f"### Answer from {p.name}\n{text}" for p, text in answers
        )
        today = datetime.now().strftime("%Y-%m-%d")
        judge_system = (
            f"You are the judge of a panel of AI models that independently answered a "
            f"research question. You are given the question and each model's answer. Today's "
            f"date is {today}.\n\n"
            "Produce ONE authoritative, well-structured answer that is better than any single "
            "input — do NOT just concatenate or average them. Compare them critically: where "
            "they agree, treat a claim as higher-confidence; where they conflict, resolve it "
            "and say why.\n\n"
            "EPISTEMICS — read carefully:\n"
            "• Your training data has a fixed cutoff and may be many months behind today's "
            "date. A model, product, price, version, company, or event you don't recognize "
            "may be REAL and RECENT, not fabricated. Absence from your memory is NOT evidence "
            "that something is fake.\n"
            "• You have a web_search tool. When a load-bearing claim is one you cannot confirm "
            "from your own knowledge — ESPECIALLY specific named products/versions, prices, "
            "dates, capabilities, or events that could postdate your cutoff — SEARCH to verify "
            "it before ruling. Verify the decision-critical claims; you needn't re-research "
            "everything the panel already covered well.\n"
            "• Use calibrated verdicts: CONFIRMED (search backs it), CONTRADICTED (search "
            "refutes it — say so), or UNVERIFIED (you couldn't confirm it — say exactly that). "
            "NEVER call a claim 'fabricated', 'fake', or 'vapor' merely because it isn't in "
            "your training data or you didn't immediately find it — reserve that for claims "
            "search ACTIVELY contradicts. When the panel members disagree on a fact, or one "
            "cites sources you can't confirm, search to settle it rather than siding with your "
            "prior.\n\n"
            "Attribute a contested or non-obvious claim to the model(s) it came from only when "
            "that helps the reader judge reliability. Don't narrate that you're a judge — just "
            "give the answer (you may briefly note which key facts you verified by search)."
        )
        judge_messages = [{
            "role": "user",
            "content": f"Research question:\n{query}\n\nPanel answers:\n\n{panel_block}",
        }]
        # The judge KEEPS web search (default tools) so it can VERIFY decision-critical
        # or post-cutoff claims before ruling — judging blind over stale priors is how it
        # once dismissed real post-cutoff products as "vapor". thinking=True: adjudicating
        # conflicts + deciding what to verify is real reasoning, not a formatting pass.
        return await self._panel_complete(
            judge, guild_id, judge_messages, judge_system, thinking=True,
        )

    async def _web_search(
        self,
        query: str,
        channel: discord.abc.Messageable,
        guild_id: int
    ) -> tuple[str, list[discord.Embed]]:
        """
        Perform a web search using Claude's web_search tool.
        Returns (response_text, list_of_embeds_for_citations)
        """
        # Build context for the search
        memory_context = self.manager.memories[guild_id].get_context_string()
        system = (
            "You are a helpful assistant performing a web search. "
            "Use the web_search tool to find current information, then provide a clear, "
            "well-cited answer. Be concise but thorough. Reference earlier conversation "
            "when the search query depends on it (e.g. 'links for what you said earlier')."
        )
        if memory_context:
            system += f"\n\nContext about the user/server:\n{memory_context}"

        # Pull conversation history so search queries that reference prior turns
        # (e.g. "find links for what you mentioned") have something to anchor on.
        # The triggering !search message is already in history; replace its content
        # with the cleaned query so the model sees the literal question to answer.
        history = await self.manager.fetch_thread_history(channel)
        history = self._strip_internal_keys(history)
        if history and history[-1].get("role") == "user":
            history[-1] = {"role": "user", "content": query}
        else:
            history.append({"role": "user", "content": query})
        messages = history
        
        try:
            # Web search always uses Claude (has built-in web search tool)
            response = await asyncio.to_thread(
                self.claude_client.messages.create,
                model=self.claude_provider.model_id,
                max_tokens=self.claude_provider.max_tokens,
                system=system,
                messages=messages,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }]
            )

            # Track usage (includes cache info via helper)
            self._record_claude_usage(response.usage)

            # Collect all sources for embeds
            sources = []
            final_text = ""

            # Process response - may need multiple rounds if tool_use
            while response.stop_reason == "tool_use":
                tool_use_block = None
                for block in response.content:
                    if block.type == "tool_use":
                        tool_use_block = block
                        break

                if not tool_use_block:
                    break

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": "Search completed."
                    }]
                })

                response = await asyncio.to_thread(
                    self.claude_client.messages.create,
                    model=self.claude_provider.model_id,
                    max_tokens=self.claude_provider.max_tokens,
                    system=system,
                    messages=messages,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search"
                    }]
                )

                self._record_claude_usage(response.usage, count_request=False)

            # Extract final text response
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text
            
            # Try to extract citations from the response
            # Claude's web search includes citations in a specific format
            embeds = []
            
            # Look for citation patterns and create embeds
            # The response may contain URLs - extract unique ones
            url_pattern = r'https?://[^\s\)\]<>\"\']+[^\s\.\,\)\]<>\"\':]'
            found_urls = list(set(re.findall(url_pattern, final_text)))[:CONFIG.max_search_results_in_embed]
            
            if found_urls:
                embed = discord.Embed(
                    title="🔍 Sources",
                    color=discord.Color.blue()
                )
                for i, url in enumerate(found_urls, 1):
                    # Truncate long URLs for display
                    display_url = url[:60] + "..." if len(url) > 60 else url
                    embed.add_field(
                        name=f"Source {i}",
                        value=f"[{display_url}]({url})",
                        inline=False
                    )
                embeds.append(embed)
            
            return final_text.strip(), embeds
            
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                req_id = getattr(e, "request_id", None) or "<not reported>"
                return (
                    f"⚠️ Claude search returned {e.status_code} (Anthropic-side "
                    f"blip — `req_id={req_id}`). Try `!search` again.",
                    [],
                )
            return f"❌ Search API Error {e.status_code}: {e.message}", []
        except anthropic.APIError as e:
            return f"❌ Search API Error: {e}", []
    
    def _extract_code_files(self, response: str) -> tuple[str, list[discord.File]]:
        """
        Extract code blocks with filenames and convert to Discord files.
        Format: ```filename.ext
        Returns (cleaned_response, list_of_files)
        """
        files = []

        # Pattern for code blocks with filename: ```filename.ext\ncode\n```
        pattern = r'```(\w+\.\w+)\n(.*?)```'

        def replace_with_attachment_note(match):
            filename = match.group(1)
            code = match.group(2)

            # Only convert to file if code is long enough
            if len(code) > 500:
                file_buffer = io.BytesIO(code.encode('utf-8'))
                files.append(discord.File(file_buffer, filename=filename))
                return f"📎 *See attached file: `{filename}`*"
            else:
                # Keep short code inline
                return match.group(0)

        cleaned = re.sub(pattern, replace_with_attachment_note, response, flags=re.DOTALL)

        return cleaned, files

    # LaTeX detection patterns. Display math is unambiguous ($$...$$). Inline
    # math ($...$) is filtered with a heuristic to avoid matching things like
    # "$50" — only blocks that contain real LaTeX syntax (\command, ^, _, {})
    # get rendered.
    LATEX_DISPLAY_RE = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
    LATEX_INLINE_RE = re.compile(r'(?<!\\)(?<!\$)\$([^\$\n]+?)\$(?!\$)')
    LATEX_LIKELY_RE = re.compile(r'\\[a-zA-Z]+|[\^_]\{|\\\\')
    LATEX_MAX_RENDERS = 8
    LATEX_MAX_SOURCE_LEN = 1500

    @staticmethod
    def _render_latex(latex_source: str) -> Optional[bytes]:
        """Render a LaTeX math expression to PNG bytes via matplotlib's mathtext.

        Returns None on render failure (unsupported LaTeX command, syntax error,
        empty input). Caller should fall back to leaving the source text alone.
        """
        src = latex_source.strip()
        if not src:
            return None
        try:
            buf = io.BytesIO()
            _mpl_mathtext.math_to_image(f"${src}$", buf, format='png', dpi=200)
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            return None

    @classmethod
    def _extract_latex_blocks(cls, text: str) -> list[str]:
        """Find LaTeX math blocks worth rendering. Returns the source strings.

        Display blocks ($$...$$) are taken first, then inline ($...$) blocks
        from the remaining text. Inline blocks must contain LaTeX-like syntax
        to avoid false positives on currency or sentence punctuation.
        """
        blocks: list[str] = []
        seen: set[str] = set()

        def add(src: str) -> bool:
            src = src.strip()
            if not src or src in seen or len(src) > cls.LATEX_MAX_SOURCE_LEN:
                return False
            seen.add(src)
            blocks.append(src)
            return len(blocks) >= cls.LATEX_MAX_RENDERS

        for m in cls.LATEX_DISPLAY_RE.finditer(text):
            if add(m.group(1)):
                return blocks
        text_no_display = cls.LATEX_DISPLAY_RE.sub('', text)
        for m in cls.LATEX_INLINE_RE.finditer(text_no_display):
            src = m.group(1).strip()
            if not cls.LATEX_LIKELY_RE.search(src):
                continue
            if add(src):
                return blocks
        return blocks

    @staticmethod
    def _composite_latex_pngs(pngs: list[bytes], pad: int = 24) -> bytes:
        """Stack rendered equation PNGs vertically into one image with white
        background and centered alignment. Discord shows multiple attachments
        in a squished horizontal grid; a single tall image renders full-width
        inline and reads top-to-bottom in equation order.
        """
        images = [_PILImage.open(io.BytesIO(p)).convert("RGBA") for p in pngs]
        max_w = max(im.width for im in images)
        total_w = max_w + 2 * pad
        total_h = sum(im.height for im in images) + pad * (len(images) + 1)
        canvas = _PILImage.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
        y = pad
        for im in images:
            x = pad + (max_w - im.width) // 2
            canvas.paste(im, (x, y), im)
            y += im.height + pad
        out = io.BytesIO()
        canvas.convert("RGB").save(out, format="PNG", optimize=True)
        out.seek(0)
        return out.getvalue()

    def _render_latex_attachments(self, response_text: str, max_files: int) -> list[discord.File]:
        """Detect LaTeX in the response and produce PNG attachments. The source
        text is left untouched in the message body so users can copy it.

        With multiple equations we composite them into a single tall PNG —
        Discord otherwise lays multiple attachments out in a horizontal grid
        which squishes wide math renders into illegibility.
        """
        if max_files <= 0:
            return []
        pngs: list[bytes] = []
        for src in self._extract_latex_blocks(response_text):
            png = self._render_latex(src)
            if png is not None:
                pngs.append(png)
        if not pngs:
            return []
        if len(pngs) == 1:
            return [discord.File(io.BytesIO(pngs[0]), filename="eq.png")]
        try:
            composite = self._composite_latex_pngs(pngs)
            return [discord.File(io.BytesIO(composite), filename="equations.png")]
        except Exception:
            # Composite failed for some reason — fall back to individual files,
            # capped at max_files. At least the math gets through.
            return [
                discord.File(io.BytesIO(p), filename=f"eq{i}.png")
                for i, p in enumerate(pngs[:max_files], 1)
            ]
    
    async def _send_response(
        self,
        channel: discord.abc.Messageable,
        content: str,
        files: list[discord.File] = None
    ) -> Optional[discord.Message]:
        """Send a response. Markdown→mrkdwn conversion and chunking to Slack's
        limits happen inside channel.send (OutChannel), so we never split raw
        Markdown across messages. Returns the first sent message so callers can
        key per-message state like the reasoning cache."""
        if not content and not files:
            return None
        return await channel.send(content, files=files)

    # --- Mandarin text-to-speech (language-teaching mode) ------------------
    # Azure Neural TTS is the only mainstream backend that lets us PIN exact
    # tones via SSML <phoneme> tags rather than letting the synthesizer infer
    # them — which is the whole point for a tutor. We use the LLM as the
    # grapheme→pinyin frontend (it already writes perfect tone-marked pinyin
    # and resolves polyphones / tone sandhi in context — the hard part), then
    # deterministically convert those marks to Azure's sapi pinyin (tone
    # NUMBERS) and force them. Anything upstream failing degrades gracefully to
    # letting Azure infer the tones, so the user still gets audible output.
    MANDARIN_TTS_VOICE = "zh-CN-XiaoxiaoNeural"
    MANDARIN_TTS_MAX_CHARS = 300
    MANDARIN_SPEAK_INLINE_MAX = 8  # max inline speak clips synthesized per message

    _PINYIN_TONE_MARKS = {
        'ā': ('a', '1'), 'á': ('a', '2'), 'ǎ': ('a', '3'), 'à': ('a', '4'),
        'ē': ('e', '1'), 'é': ('e', '2'), 'ě': ('e', '3'), 'è': ('e', '4'),
        'ī': ('i', '1'), 'í': ('i', '2'), 'ǐ': ('i', '3'), 'ì': ('i', '4'),
        'ō': ('o', '1'), 'ó': ('o', '2'), 'ǒ': ('o', '3'), 'ò': ('o', '4'),
        'ū': ('u', '1'), 'ú': ('u', '2'), 'ǔ': ('u', '3'), 'ù': ('u', '4'),
        'ǖ': ('v', '1'), 'ǘ': ('v', '2'), 'ǚ': ('v', '3'), 'ǜ': ('v', '4'),
        'ü': ('v', '5'),
    }

    @classmethod
    def _pinyin_marks_to_numbers(cls, pinyin: str) -> Optional[str]:
        """Convert tone-marked pinyin ('nǐ hǎo') to Azure sapi pinyin with tone
        numbers ('ni3 hao3'). Deterministic — we own this so we don't depend on
        the model emitting the right machine format. Accepts already-numbered
        syllables too. Returns None if the result isn't clean syllable+tone
        tokens, in which case the caller lets Azure infer tones rather than
        forcing a malformed pronunciation."""
        out_syllables: list[str] = []
        for raw in pinyin.split():
            syl = raw.strip().strip(",.!?'’\"；;：:。，！？、")
            if not syl:
                continue
            has_mark = any(ch in cls._PINYIN_TONE_MARKS for ch in syl)
            if has_mark:
                tone = '5'
                chars: list[str] = []
                for ch in syl:
                    mapped = cls._PINYIN_TONE_MARKS.get(ch)
                    if mapped:
                        base, tone = mapped
                        chars.append(base)
                    else:
                        chars.append(ch)
                token = ''.join(chars) + tone
            else:
                # Already numbered ('ni3'), or toneless ('de') → neutral tone 5.
                token = syl if syl[-1:].isdigit() else syl + '5'
            token = token.replace('ü', 'v').replace('Ü', 'v').lower()
            out_syllables.append(token)
        if not out_syllables:
            return None
        if not all(re.fullmatch(r"[a-z]+[1-5]", s) for s in out_syllables):
            return None
        return ' '.join(out_syllables)

    @classmethod
    def _parse_g2p(cls, out: str, fallback_hanzi: str) -> dict:
        """Parse the three-line G2P reply into a reading dict
        {hanzi, citation, surface, surface_numbered}. Tolerant of casing and
        stray markdown. citation/surface are tone-marked display strings (None
        if absent); surface_numbered is the sapi pinyin we synthesize (None if
        the surface line is missing or unconvertible, so the caller lets Azure
        infer tones). An empty `out` yields the all-None fallback shape."""
        def grab(label: str, s: str) -> Optional[str]:
            m = re.match(rf'(?i)^{label}\s*[:：]\s*(.+)$', s)
            return m.group(1).strip().strip('*`').strip() if m else None

        hanzi: Optional[str] = None
        citation: Optional[str] = None
        surface: Optional[str] = None
        for line in out.splitlines():
            s = line.strip().lstrip('*`> ').strip()
            for label in ("HANZI", "CITATION", "SURFACE"):
                v = grab(label, s)
                if v is not None:
                    if label == "HANZI":
                        hanzi = v
                    elif label == "CITATION":
                        citation = v
                    else:
                        surface = v
                    break
        if not hanzi:
            hanzi = fallback_hanzi
        # If the model gave only one pinyin form, treat it as both (no sandhi).
        if surface and not citation:
            citation = surface
        if citation and not surface:
            surface = citation
        surface_numbered = cls._pinyin_marks_to_numbers(surface) if surface else None
        return {
            "hanzi": hanzi,
            "citation": citation,
            "surface": surface,
            "surface_numbered": surface_numbered,
        }

    async def _mandarin_g2p(self, guild_id: int, text: str) -> dict:
        """Use an LLM as the grapheme→phoneme frontend: turn arbitrary input
        (Chinese characters, pinyin, or an English phrase to translate) into a
        reading dict — hanzi, the citation (dictionary) tones, the surface tones
        as actually spoken (tone sandhi applied), and the surface form as Azure
        sapi pinyin (tone numbers) for synthesis. The LLM resolves polyphones
        and sandhi in context far better than a static table — the exact failure
        mode the !research panel flagged.

        Keys: {hanzi, citation, surface, surface_numbered}. citation/surface are
        tone-marked strings for display (may be None); surface_numbered is what
        we feed Azure (None → let Azure infer tones). Prefers Deepseek
        (Chinese-native) then Claude then Gemini. Lands in !cost via the normal
        generation path."""
        provider = next(
            (p for p in (self.deepseek_provider, self.claude_provider, self.gemini_provider)
             if p.enabled),
            None,
        )
        if provider is None:
            return self._parse_g2p("", fallback_hanzi=text)
        system = (
            "You are the pronunciation frontend for a Mandarin text-to-speech engine and a "
            "pinyin tutor. Convert the user's input to Mandarin Chinese. The input may be "
            "Chinese characters, pinyin, or an English phrase to translate. Reply with "
            "EXACTLY these three lines and nothing else — no markdown, no translation, no "
            "commentary:\n"
            "HANZI: <the Chinese characters only>\n"
            "CITATION: <space-separated syllables with tone marks — the DICTIONARY tone of "
            "each syllable in isolation, with NO tone sandhi applied>\n"
            "SURFACE: <the same syllables with tone marks, but WITH tone sandhi applied as "
            "the phrase is actually spoken>\n"
            "For example, for the input 你好 you output exactly:\n"
            "HANZI: 你好\n"
            "CITATION: nǐ hǎo\n"
            "SURFACE: ní hǎo\n"
            "Apply sandhi only where it really occurs: for 谢谢, CITATION and SURFACE are "
            "identical (xiè xie). Common cases: 3rd+3rd → 2nd+3rd (你好 → ní hǎo); 不 bù → bú "
            "before a 4th tone (不是 → bú shì); 一 yī → yí/yì depending on the next tone."
        )
        messages = [{"role": "user", "content": text}]
        try:
            out = await self._panel_complete(provider, guild_id, messages, system)
        except Exception as e:
            print(f"⚠️  Mandarin G2P failed: {e}")
            return self._parse_g2p("", fallback_hanzi=text)
        if self._is_provider_error(provider, out):
            return self._parse_g2p("", fallback_hanzi=text)
        return self._parse_g2p(out, fallback_hanzi=text)

    @staticmethod
    def _is_han(ch: str) -> bool:
        """True for a CJK ideograph (one Mandarin syllable). Used to pair Han
        characters 1:1 with pinyin syllables when building forced-tone SSML."""
        return '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿'

    @classmethod
    def _build_mandarin_ssml(cls, hanzi: str, pinyin_numbered: Optional[str]) -> str:
        """Wrap text in zh-CN SSML. When we have numbered pinyin that lines up
        1:1 with the Han characters, force the pronunciation with one
        <phoneme alphabet="sapi"> tag PER SYLLABLE.

        Azure's sapi pinyin is fussy (verified against the live API): it wants a
        SPACE before the tone digit ('ni 3', not 'ni3') and a separate <phoneme>
        tag per syllable — a single tag with 'ni3 hao3' is rejected HTTP 400. If
        the syllable count doesn't match the Han-character count (punctuation is
        fine; think loanwords, digits, erhua merges), fall back to plain text and
        let Azure infer the tones rather than emit mismatched SSML."""
        body: Optional[str] = None
        if pinyin_numbered:
            syllables = pinyin_numbered.split()
            han = [c for c in hanzi if cls._is_han(c)]
            if syllables and len(han) == len(syllables):
                parts: list[str] = []
                it = iter(syllables)
                for ch in hanzi:
                    if cls._is_han(ch):
                        syl = next(it)
                        m = re.fullmatch(r'([a-z]+)([1-5])', syl)
                        ph = f"{m.group(1)} {m.group(2)}" if m else syl
                        parts.append(
                            f'<phoneme alphabet="sapi" ph="{_html.escape(ph, quote=True)}">'
                            f'{_html.escape(ch)}</phoneme>'
                        )
                    else:
                        parts.append(_html.escape(ch))
                body = ''.join(parts)
        if body is None:
            body = _html.escape(hanzi)
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="zh-CN">'
            f'<voice name="{cls.MANDARIN_TTS_VOICE}">{body}</voice>'
            '</speak>'
        )

    async def _azure_tts(self, ssml: str) -> Optional[bytes]:
        """POST SSML to Azure Speech and return MP3 bytes (or None on failure)."""
        if not (self.azure_tts_key and self.azure_tts_region):
            return None
        url = (
            f"https://{self.azure_tts_region}.tts.speech.microsoft.com"
            "/cognitiveservices/v1"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_tts_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
            "User-Agent": "OpusDeipseekBot",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, data=ssml.encode("utf-8")
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:200]
                        print(f"⚠️  Azure TTS HTTP {resp.status}: {body}")
                        return None
                    return await resp.read()
        except Exception as e:
            print(f"⚠️  Azure TTS request failed: {e}")
            return None

    async def _synthesize_mandarin(
        self, guild_id: int, text: str, pinyin: Optional[str] = None
    ) -> tuple[Optional[bytes], dict]:
        """Synthesize one Mandarin phrase → (mp3_bytes_or_None, reading_dict).

        If pinyin is supplied (tone marks or numbers — e.g. from an inline
        [[speak:汉字|pinyin]] marker), use it directly and skip the G2P LLM call;
        otherwise run the LLM frontend. Tones are forced via SSML, retrying with
        inferred tones if Azure rejects the forced pronunciation. Shared by the
        !speak command and the inline [[speak:..]] renderer."""
        if pinyin:
            reading = {
                "hanzi": text,
                "citation": pinyin,
                "surface": pinyin,
                "surface_numbered": self._pinyin_marks_to_numbers(pinyin),
            }
        else:
            reading = await self._mandarin_g2p(guild_id, text)
        hanzi = reading["hanzi"]
        surface_numbered = reading["surface_numbered"]
        audio = await self._azure_tts(self._build_mandarin_ssml(hanzi, surface_numbered))
        if audio is None and surface_numbered is not None:
            # Forced-tone SSML may have been rejected (odd syllable); retry
            # letting Azure infer tones so the user still gets audio.
            print("⚠️  Forced-tone TTS failed; retrying with inferred tones.")
            audio = await self._azure_tts(self._build_mandarin_ssml(hanzi, None))
        return audio, reading

    # Inline directive the models can emit to attach spoken Mandarin:
    #   [[speak:汉字]]            → bot runs G2P then synthesizes
    #   [[speak:汉字|nǐ hǎo]]     → bot uses the given pinyin (tone marks/numbers)
    SPEAK_MARKER_RE = re.compile(
        r'\[\[\s*speak\s*:\s*([^\]|]+?)\s*(?:\|\s*([^\]]+?)\s*)?\]\]', re.IGNORECASE
    )
    # Models naturally reach for the bare command form mid-message; normalize
    # "!speak 汉字" (the command word + a run of CJK characters) into the
    # [[speak:汉字]] form so both spellings flow through one renderer.
    BANG_SPEAK_RE = re.compile(
        r'!speak[ \t]+([一-鿿㐀-䶿　-〿＀-￯]+)',
        re.IGNORECASE,
    )
    # Models tend to imitate the bot's *rendered output* ("汉字 🔊") instead of
    # re-issuing the command, which silently produces no audio (the model is
    # copying what it sees in history). Treat that rendered form — a CJK run,
    # an optional pinyin paren, then the speaker emoji — as a request too, so
    # the model can't get it wrong either way. The model's own paren pinyin is
    # dropped here so the bot re-derives the authoritative version.
    EMOJI_SPEAK_RE = re.compile(
        r'([一-鿿㐀-䶿　-〿＀-￯]+)(?:[ \t]*\([^)\n]*\))?[ \t]*🔊'
    )

    @staticmethod
    def _inline_pinyin_annot(reading: dict) -> str:
        """Compact pinyin annotation for an inline speak clip: the spoken
        (sandhi) and dictionary (citation) forms, collapsing to one when there's
        no sandhi. Empty string if no pinyin is available. This makes the bot —
        not the model's hand-typing — the source of truth for the pinyin."""
        surface = reading.get("surface")
        citation = reading.get("citation")
        if surface and citation and surface != citation:
            return f" (spoken `{surface}` · dict. `{citation}`)"
        if surface or citation:
            return f" (`{surface or citation}`)"
        return ""

    async def _render_speak_attachments(
        self, guild_id: int, text: str, max_files: int
    ) -> tuple[str, list[discord.File]]:
        """Turn inline [[speak:汉字]] / [[speak:汉字|pinyin]] markers into MP3
        attachments. Each synthesized marker is replaced in the text by
        '汉字 🔊'; markers beyond the per-message cap (or that fail to synthesize,
        or when TTS is unconfigured) collapse to just the phrase text so users
        never see raw [[speak:..]] syntax. Returns (cleaned_text, files)."""
        text = self.BANG_SPEAK_RE.sub(r'[[speak:\1]]', text)   # !speak 汉字 → [[speak:汉字]]
        text = self.EMOJI_SPEAK_RE.sub(r'[[speak:\1]]', text)  # 汉字 🔊 (imitated output) → marker
        strip = lambda: self.SPEAK_MARKER_RE.sub(lambda m: m.group(1).strip(), text)
        if max_files <= 0 or not (self.azure_tts_key and self.azure_tts_region):
            return strip(), []
        matches = list(self.SPEAK_MARKER_RE.finditer(text))
        if not matches:
            return text, []
        synth = matches[:min(max_files, self.MANDARIN_SPEAK_INLINE_MAX)]

        async def one(m: re.Match) -> tuple[Optional[bytes], dict]:
            hanzi = m.group(1).strip()[:self.MANDARIN_TTS_MAX_CHARS]
            py = (m.group(2) or "").strip() or None
            return await self._synthesize_mandarin(guild_id, hanzi, pinyin=py)
        results = await asyncio.gather(*(one(m) for m in synth))

        files: list[discord.File] = []
        out: list[str] = []
        last = 0
        for m, (audio, reading) in zip(synth, results):
            out.append(text[last:m.start()])
            hanzi = m.group(1).strip()
            if audio is not None:
                files.append(discord.File(io.BytesIO(audio), filename=f"speak_{len(files)+1}.mp3"))
                out.append(f"{hanzi}{self._inline_pinyin_annot(reading)} 🔊")
            else:
                out.append(hanzi)
            last = m.end()
        out.append(text[last:])
        # Markers past the cap are still raw [[speak:..]] — collapse them to text.
        cleaned = self.SPEAK_MARKER_RE.sub(lambda m: m.group(1).strip(), ''.join(out))
        return cleaned, files

    # --- French text-to-speech (language-teaching mode) --------------------
    # The INVERSE of the Mandarin path. Azure's fr-FR neural voices already
    # pronounce liaison, élision, nasal vowels and silent letters correctly, so
    # we do NOT force phonemes — we send the plain French text and let the voice
    # infer (the opposite default from zh-CN, where tones must be pinned, because
    # Azure guesses them wrong). The LLM frontend (Mistral-preferred — the
    # French-native analog of Deepseek-for-Chinese) returns the French text, an
    # IPA transcription, and a one-line liaison/pronunciation note. IPA is the
    # learner's "pinyin" — DISPLAY only, never fed to the synth.
    FRENCH_TTS_VOICE = "fr-FR-DeniseNeural"  # clear, standard diction for learners (Vivienne = more natural alt)
    FRENCH_TTS_MAX_CHARS = 300
    FRENCH_SPEAK_INLINE_MAX = 8  # max inline french clips synthesized per message

    @classmethod
    def _parse_french_g2p(cls, out: str, fallback_text: str) -> dict:
        """Parse the three-line G2P reply into {text, ipa, note}. Tolerant of
        casing / stray markdown. Empty `out` yields the all-fallback shape."""
        def grab(label: str, s: str) -> Optional[str]:
            m = re.match(rf'(?i)^{label}\s*[:：]\s*(.+)$', s)
            return m.group(1).strip().strip('*`').strip() if m else None
        text: Optional[str] = None
        ipa: Optional[str] = None
        note: Optional[str] = None
        for line in out.splitlines():
            s = line.strip().lstrip('*`> ').strip()
            for label in ("TEXT", "IPA", "NOTE"):
                v = grab(label, s)
                if v is not None:
                    if label == "TEXT":
                        text = v
                    elif label == "IPA":
                        ipa = v
                    else:
                        note = v
                    break
        if not text:
            text = fallback_text
        if note in ("—", "-", ""):
            note = None
        return {"text": text, "ipa": ipa, "note": note}

    async def _french_g2p(self, guild_id: int, text: str) -> dict:
        """LLM frontend for French: turn input (French text, or an English phrase
        to translate) into {text, ipa, note}. Prefers Mistral (French-native —
        the analog of Deepseek for Chinese), then Claude/Gemini/Deepseek. Lands
        in !cost via the normal generation path."""
        provider = next(
            (p for p in (self.mistral_provider, self.claude_provider,
                         self.gemini_provider, self.deepseek_provider)
             if p.enabled),
            None,
        )
        if provider is None:
            return self._parse_french_g2p("", fallback_text=text)
        system = (
            "You are the pronunciation frontend for a French text-to-speech engine and an "
            "IPA tutor. Convert the user's input to natural French. The input may be French "
            "text, or an English phrase to translate. Reply with EXACTLY these three lines "
            "and nothing else — no extra commentary:\n"
            "TEXT: <the French text only, correct accents and punctuation>\n"
            "IPA: <the phrase in IPA, as actually spoken with liaison and élision applied>\n"
            "NOTE: <ONE short English line on the single trickiest pronunciation point — a "
            "liaison, a silent letter, or a nasal vowel — or '—' if nothing notable>\n"
            "For the input 'the friends' you output exactly:\n"
            "TEXT: les amis\n"
            "IPA: le.z‿a.mi\n"
            "NOTE: liaison — the silent 's' in 'les' links as /z/ into 'amis'"
        )
        messages = [{"role": "user", "content": text}]
        try:
            out = await self._panel_complete(provider, guild_id, messages, system)
        except Exception as e:
            print(f"⚠️  French G2P failed: {e}")
            return self._parse_french_g2p("", fallback_text=text)
        if self._is_provider_error(provider, out):
            return self._parse_french_g2p("", fallback_text=text)
        return self._parse_french_g2p(out, fallback_text=text)

    @classmethod
    def _build_french_ssml(cls, text: str) -> str:
        """Wrap French text in fr-FR SSML for Azure. Unlike the Mandarin path we
        do NOT force phonemes — the fr-FR neural voice handles liaison / nasals /
        silent letters natively, so plain text gives the most natural result."""
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="fr-FR">'
            f'<voice name="{cls.FRENCH_TTS_VOICE}">{_html.escape(text)}</voice>'
            '</speak>'
        )

    async def _synthesize_french(
        self, guild_id: int, text: str, ipa: Optional[str] = None
    ) -> tuple[Optional[bytes], dict]:
        """Synthesize one French phrase → (mp3_bytes_or_None, reading_dict). If
        ipa is supplied (from an inline [[french:texte|ipa]] marker) we skip the
        G2P LLM call and use it for display; the audio is always the plain French
        text through the native fr-FR voice. Shared by !french and the inline
        [[french:..]] renderer."""
        if ipa:
            reading = {"text": text, "ipa": ipa, "note": None}
        else:
            reading = await self._french_g2p(guild_id, text)
        audio = await self._azure_tts(self._build_french_ssml(reading["text"]))
        return audio, reading

    # Inline directive the models can emit to attach spoken French:
    #   [[french:texte]]            → bot runs G2P then synthesizes
    #   [[french:texte|le.z‿a.mi]]  → bot uses the given IPA for display
    FRENCH_MARKER_RE = re.compile(
        r'\[\[\s*french\s*:\s*([^\]|]+?)\s*(?:\|\s*([^\]]+?)\s*)?\]\]', re.IGNORECASE
    )

    @staticmethod
    def _inline_french_annot(reading: dict) -> str:
        """Compact annotation for an inline French clip: the IPA (the learner's
        'pinyin') plus the liaison/pronunciation note when present. Empty when
        neither is available — the bot, not the model's typing, owns the IPA."""
        ipa = reading.get("ipa")
        note = reading.get("note")
        if ipa and note:
            return f" (`{ipa}` — {note})"
        if ipa:
            return f" (`{ipa}`)"
        if note:
            return f" ({note})"
        return ""

    async def _render_french_attachments(
        self, guild_id: int, text: str, max_files: int
    ) -> tuple[str, list[discord.File]]:
        """Turn inline [[french:texte]] markers into MP3 attachments, mirroring
        _render_speak_attachments. Each synthesized marker becomes
        'texte (`ipa`) 🔊'; markers past the cap / failures / unconfigured TTS
        collapse to the bare phrase. Returns (cleaned_text, files)."""
        strip = lambda: self.FRENCH_MARKER_RE.sub(lambda m: m.group(1).strip(), text)
        if max_files <= 0 or not (self.azure_tts_key and self.azure_tts_region):
            return strip(), []
        matches = list(self.FRENCH_MARKER_RE.finditer(text))
        if not matches:
            return text, []
        synth = matches[:min(max_files, self.FRENCH_SPEAK_INLINE_MAX)]

        async def one(m: re.Match) -> tuple[Optional[bytes], dict]:
            phrase = m.group(1).strip()[:self.FRENCH_TTS_MAX_CHARS]
            ipa = (m.group(2) or "").strip() or None
            return await self._synthesize_french(guild_id, phrase, ipa=ipa)
        results = await asyncio.gather(*(one(m) for m in synth))

        files: list[discord.File] = []
        out: list[str] = []
        last = 0
        for m, (audio, reading) in zip(synth, results):
            out.append(text[last:m.start()])
            phrase = m.group(1).strip()
            if audio is not None:
                files.append(discord.File(io.BytesIO(audio), filename=f"french_{len(files)+1}.mp3"))
                out.append(f"{phrase}{self._inline_french_annot(reading)} 🔊")
            else:
                out.append(phrase)
            last = m.end()
        out.append(text[last:])
        cleaned = self.FRENCH_MARKER_RE.sub(lambda m: m.group(1).strip(), ''.join(out))
        return cleaned, files

    async def _handle_command(self, message: discord.Message) -> None:
        """Handle bot commands."""
        content = message.content.strip()
        parts = content.split(maxsplit=2)
        cmd = parts[0].lower()
        guild_id = self._mem_key(message.channel.id)
        memory = self.manager.memories[guild_id]
        
        if cmd == "!context":
            messages = await self.manager.fetch_thread_history(message.channel)
            # Look up channel-level reading material via parent channel for threads
            ctx_channel_id = message.channel.id
            if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                ctx_channel_id = message.channel.parent_id
            info = self.manager.get_context_info(messages, guild_id, channel_id=ctx_channel_id)
            await message.channel.send(info)
        
        elif cmd == "!cost":
            summary = self.manager.get_cost_summary(self.providers)
            await message.channel.send(summary)
        
        elif cmd == "!memories":
            lines = []
            
            # Long-term memories
            if memory.longterm.entries:
                lines.append("🧠 **Long-term memories** (permanent):")
                for key, value in memory.longterm.entries.items():
                    lines.append(f"  `{key}`: {value}")
            else:
                lines.append("🧠 **Long-term memories**: None yet")
            
            lines.append("")
            
            # Working notes
            if memory.working.notes:
                lines.append("📝 **Working notes** (fade over time):")
                for key, note in sorted(
                    memory.working.notes.items(),
                    key=lambda x: x[1].freshness(CONFIG.working_memory_decay_hours),
                    reverse=True
                ):
                    freshness = note.freshness(CONFIG.working_memory_decay_hours)
                    if freshness > 0.7:
                        indicator = "🟢"
                    elif freshness > 0.3:
                        indicator = "🟡"
                    else:
                        indicator = "🔴"
                    lines.append(f"  {indicator} `{key}`: {note.content}")
                lines.append("")
                lines.append("*Use `!keep <key>` to make a working note permanent*")
            else:
                lines.append("📝 **Working notes**: None yet")
            
            # Send in chunks if too long
            full_text = "\n".join(lines)
            await self._send_response(message.channel, full_text)
        
        elif cmd == "!remember":
            # !remember key value
            if len(parts) >= 3:
                key = parts[1]
                value = parts[2]
                if memory.longterm.add(key, value):
                    self.manager.save_memories(providers=self.providers)
                    await message.channel.send(f"✅ Remembered `{key}` (permanent)")
                else:
                    await message.channel.send(
                        f"❌ Long-term memory full ({CONFIG.max_longterm_memories} max). "
                        f"Use `!forget <key>` to make room."
                    )
            else:
                await message.channel.send("Usage: `!remember <key> <value>`")
        
        elif cmd == "!forget":
            if len(parts) >= 2:
                key = parts[1]
                # Try long-term first, then working
                if memory.longterm.remove(key):
                    self.manager.save_memories(providers=self.providers)
                    await message.channel.send(f"✅ Forgot `{key}` from long-term memory")
                elif memory.working.remove(key):
                    self.manager.save_memories(providers=self.providers)
                    await message.channel.send(f"✅ Forgot `{key}` from working notes")
                else:
                    await message.channel.send(f"❓ No memory with key `{key}`")
            else:
                await message.channel.send("Usage: `!forget <key>`")
        
        elif cmd == "!keep":
            # Promote a working note to long-term memory
            if len(parts) >= 2:
                key = parts[1]
                if key not in memory.working.notes:
                    await message.channel.send(f"❓ No working note with key `{key}`")
                elif memory.promote(key):
                    self.manager.save_memories(providers=self.providers)
                    await message.channel.send(f"✅ Promoted `{key}` to long-term memory (permanent)")
                else:
                    await message.channel.send(
                        f"❌ Long-term memory full ({CONFIG.max_longterm_memories} max). "
                        f"Use `!forget <key>` to make room."
                    )
            else:
                await message.channel.send("Usage: `!keep <key>` - promotes a working note to permanent memory")
        
        elif cmd == "!threads":
            # Show the thread index
            thread_index = await self.manager.fetch_thread_index(message.channel)
            if thread_index:
                await message.channel.send(thread_index)
            else:
                await message.channel.send("📭 No other threads found in this channel.")
        
        elif cmd == "!search":
            # Web search is available via any of:
            #   - Claude's native web_search tool (ANTHROPIC_API_KEY)
            #   - Gemini's native google_search grounding (GEMINI_API_KEY)
            #   - Any OpenAI-compatible provider via Tavily (TAVILY_API_KEY)
            #
            # Pick the searcher in this order: channel preference → Claude →
            # Gemini (native) → Deepseek (Tavily).
            has_gemini_native = self.gemini_provider.enabled and bool(os.getenv("GEMINI_API_KEY"))
            eligible: list[ModelProvider] = []
            if self.claude_provider.enabled:
                eligible.append(self.claude_provider)
            if has_gemini_native:
                eligible.append(self.gemini_provider)
            elif self.gemini_provider.enabled and self.tavily_client:
                eligible.append(self.gemini_provider)
            if self.deepseek_provider.enabled and self.tavily_client:
                eligible.append(self.deepseek_provider)

            if not eligible:
                await message.channel.send(
                    "❌ Web search requires one of: ANTHROPIC_API_KEY (Claude native), "
                    "GEMINI_API_KEY (Gemini native grounding), or TAVILY_API_KEY (Deepseek/Gemini)."
                )
                return

            if len(parts) < 2:
                await message.channel.send(
                    "Usage: `!search <query>`\n"
                    "Example: `!search latest news on Claude AI`\n\n"
                    "⚠️ Web search costs extra tokens (~$0.01-0.03 per search)"
                )
                return

            query = message.content[8:].strip()  # len("!search ") = 8
            await message.channel.send(f"🔍 Searching: *{query}*")

            # Pick the searcher
            channel_id = message.channel.id
            parent_id = getattr(message.channel, 'parent_id', None)
            ch_pref = self.channel_preferences.get(channel_id) or self.channel_preferences.get(parent_id)
            pref_map = {p.id: p for p in self.providers}
            searcher: Optional[ModelProvider] = None
            if ch_pref in pref_map and pref_map[ch_pref] in eligible:
                searcher = pref_map[ch_pref]
            if searcher is None:
                # Default order: Claude > Gemini (native grounding) > Deepseek (Tavily).
                for p in (self.claude_provider, self.gemini_provider, self.deepseek_provider):
                    if p in eligible:
                        searcher = p
                        break

            # Dispatch
            if searcher is self.claude_provider:
                # Claude native web search — existing path
                async with message.channel.typing():
                    response_text, embeds = await self._web_search(
                        query, message.channel, guild_id
                    )
                if self.multi_model_active:
                    response_text = f"**[Claude]** {response_text}"
                await self._send_response(message.channel, response_text)
                for embed in embeds:
                    await message.channel.send(embed=embed)
            else:
                # OpenAI-compatible provider — dispatch through SearchBackend
                async with message.channel.typing():
                    search_result = await self._search_for(searcher, query)

                    if search_result.is_grounded_answer:
                        # Gemini native: backend already synthesized the answer.
                        # Display directly; we don't need to round-trip through
                        # the chat model again.
                        response_text = search_result.text
                    else:
                        # Tavily: feed raw hits through the chosen model for synthesis.
                        history = await self.manager.fetch_thread_history(message.channel)
                        history = self._strip_internal_keys(history)
                        synthesis_prompt = (
                            f"Based on these web search results, answer the query: {query}\n\n"
                            f"Search results:\n{search_result.text}"
                        )
                        if history and history[-1].get("role") == "user":
                            history[-1] = {"role": "user", "content": synthesis_prompt}
                        else:
                            history.append({"role": "user", "content": synthesis_prompt})
                        client = self.openai_compatible_clients[searcher.name]
                        response_text, _, _ = await self._generate_openai_compatible_response(
                            client, searcher, guild_id, history,
                            "You are a helpful assistant. Summarize the search results clearly and "
                            "cite your sources with URLs. Use the prior conversation as context when "
                            "the user's query refers back to it.",
                        )

                if self.multi_model_active:
                    response_text = f"**[{searcher.name}]** {response_text}"
                await self._send_response(message.channel, response_text)

                # Render citation embeds (works for both Tavily and Google native)
                if search_result.citations:
                    embed = discord.Embed(title="🔍 Sources", color=discord.Color.blue())
                    for i, cit in enumerate(
                        search_result.citations[:CONFIG.max_search_results_in_embed], 1
                    ):
                        title = cit.get("title") or cit.get("url", "")[:60] or "(untitled)"
                        url = cit.get("url", "")
                        value = f"[{title}]({url})" if url else title
                        embed.add_field(name=f"Source {i}", value=value, inline=False)
                    await message.channel.send(embed=embed)

            await message.channel.send(
                "*💡 Web search incurs additional token costs. Use `!cost` to check usage.*"
            )
        
        elif cmd == "!research":
            # Multi-model panel: every member answers independently, then a judge
            # synthesises one answer. Reuses existing per-provider generation (so
            # cost lands in !cost) and each member's own web search.
            query = content[len(cmd):].strip()
            # `!research all <q>` convenes the full roster (adds the cheap
            # Fireworks heads for diversity); plain `!research <q>` stays the
            # lean core trio. Strip a leading "all" token before the question.
            # (Edge: a question that literally starts with the word "all" will
            # trip this — reword or use the core panel; cheap to special-case
            # later if it bites.)
            use_all = False
            all_match = re.match(r'(?is)all\b\s*(.*)', query)
            if all_match and all_match.group(1).strip():
                use_all = True
                query = all_match.group(1).strip()
            if not query:
                await message.channel.send(
                    "Usage: `!research <question>`  ·  `!research all <question>`\n"
                    "Convenes a multi-model panel — each model answers independently, then a "
                    "judge synthesises them into one answer.\n"
                    "• `!research` — lean core panel (Claude · Gemini · Deepseek).\n"
                    "• `!research all` — full roster, adding the cheap Fireworks heads "
                    "(Mistral/Qwen/GLM) for max diversity once they're configured.\n"
                    "⚠️ Runs several models (each may web-search) per call — roughly 3–4× the "
                    "cost and latency of a normal reply (more with `all`)."
                )
                return

            roster = self.panel_members_all if use_all else self.panel_members
            members = [p for p in self.providers if p.enabled and p.name in roster]
            if len(members) < 2:
                enabled_names = ", ".join(p.name for p in self.providers if p.enabled) or "none"
                await message.channel.send(
                    f"❌ The research panel needs ≥2 enabled providers from {roster}. "
                    f"Currently enabled: {enabled_names}."
                )
                return
            judge = next(
                (p for p in self.providers if p.name == self.panel_judge and p.enabled),
                members[0],
            )

            await message.channel.send(
                f"🧪 Convening {'FULL ' if use_all else ''}panel: "
                f"{', '.join(p.name for p in members)} → **{judge.name}** judging…"
            )

            # Shared prompt for every member: thread history with the last user
            # turn replaced by the research question (mirrors !search).
            history = self._strip_internal_keys(
                await self.manager.fetch_thread_history(message.channel)
            )
            if history and history[-1].get("role") == "user":
                history[-1] = {"role": "user", "content": query}
            else:
                history.append({"role": "user", "content": query})

            member_system = (
                "You are one member of a panel of AI models answering a research question for "
                "a technical user. Give your best, most accurate, well-structured answer. Be "
                "concrete and show your reasoning.\n"
                "Sourcing rules (important — a later model fact-checks you):\n"
                "• Your training data has a cutoff and may be behind today's date. If the "
                "question names specific products, models, versions, prices, or events you "
                "don't recognize, use the web_search tool to check whether they're REAL and "
                "RECENT before answering — do NOT assume they're fabricated.\n"
                "• Only cite a URL, article title, quote, date, or hash that came from an "
                "actual web_search result in THIS conversation. NEVER invent, guess, or "
                "reconstruct a citation from memory. If you did not search, attach no "
                "citations — say plainly that the claim is from memory and may be out of date.\n"
                "Another model will synthesise the panel's answers afterward, so optimise for "
                "correctness and completeness, and don't address the other panelists."
            )

            async with message.channel.typing():
                answers = await self._run_panel(guild_id, history, member_system, members)
                if not answers:
                    await message.channel.send(
                        "❌ Every panel member failed to respond — check `!cost` / the logs."
                    )
                    return
                verdict = await self._judge(guild_id, query, answers, judge)

            if self.multi_model_active:
                verdict = f"**[Panel → {judge.name}]** {verdict}"
            await self._send_response(message.channel, verdict)

            failed = len(members) - len(answers)
            note = (
                f"🧪 Synthesised from {len(answers)} model(s) "
                f"({', '.join(p.name for p, _ in answers)}) + {judge.name} judge"
            )
            if failed:
                note += f"; {failed} member(s) failed"
            note += f". Costs ~{len(answers) + 1}× a normal reply — check `!cost`."
            await message.channel.send(f"*{note}*")

        elif cmd == "!speak":
            # Mandarin text-to-speech for language-teaching mode. The LLM acts as
            # the grapheme→pinyin frontend; we force those tones into Azure
            # Xiaoxiao via SSML <phoneme> and attach the MP3 (mirrors the LaTeX
            # PNG path). See _mandarin_g2p / _build_mandarin_ssml / _azure_tts.
            text = content[len(cmd):].strip()
            if not text:
                await message.channel.send(
                    "Usage: `!speak <chinese / pinyin / phrase to say>`\n"
                    "Synthesizes Mandarin with the tones forced from pinyin (Azure Xiaoxiao) "
                    "and attaches an MP3 you can play.\n"
                    "Examples: `!speak 你好`, `!speak wǒ ài nǐ`, `!speak how do you say thank you`"
                )
                return
            if not (self.azure_tts_key and self.azure_tts_region):
                await message.channel.send(
                    "❌ Mandarin TTS isn't configured. Set `AZURE_TTS_KEY` and `AZURE_TTS_REGION` "
                    "(e.g. `eastus`) in `.env` — Azure's free tier covers 0.5M chars/month."
                )
                return
            if len(text) > self.MANDARIN_TTS_MAX_CHARS:
                await message.channel.send(
                    f"❌ That's a bit long for a clip (>{self.MANDARIN_TTS_MAX_CHARS} chars). "
                    "Try a shorter phrase."
                )
                return

            async with message.channel.typing():
                audio, reading = await self._synthesize_mandarin(guild_id, text)
                hanzi = reading["hanzi"]

            if audio is None:
                await message.channel.send(
                    "❌ TTS synthesis failed — check the logs and that `AZURE_TTS_KEY` / "
                    "`AZURE_TTS_REGION` are valid."
                )
                return

            # Caption: hanzi + pinyin. The audio follows the spoken (sandhi)
            # form; show the dictionary (citation) form alongside it when they
            # differ so learners get both — collapse to one line when there's no
            # sandhi (e.g. 谢谢) so we don't show a pointless duplicate.
            citation = reading["citation"]
            surface = reading["surface"]
            caption = f"🔊 **{hanzi}**"
            if surface and citation and surface != citation:
                caption += f"\n**Spoken:** `{surface}`  ·  **Dictionary:** `{citation}`"
            elif surface or citation:
                caption += f"\n`{surface or citation}`"
            await message.channel.send(
                caption,
                file=discord.File(io.BytesIO(audio), filename="speak.mp3"),
            )

        elif cmd == "!french":
            # French text-to-speech for language-teaching mode. The LLM (Mistral-
            # preferred, French-native) is the text→IPA frontend; UNLIKE !speak we
            # let Azure's fr-FR voice infer pronunciation (it handles liaison /
            # nasals / silent letters natively) and show the IPA + a liaison note
            # for the learner. See _french_g2p / _build_french_ssml / _synthesize_french.
            text = content[len(cmd):].strip()
            if not text:
                await message.channel.send(
                    "Usage: `!french <french text / english phrase to say>`\n"
                    "Speaks it in natural French (Azure fr-FR Denise) and shows the IPA "
                    "+ a liaison/pronunciation note.\n"
                    "Examples: `!french bonjour`, `!french les amis`, `!french how do you say I love you`"
                )
                return
            if not (self.azure_tts_key and self.azure_tts_region):
                await message.channel.send(
                    "❌ French TTS isn't configured. Set `AZURE_TTS_KEY` and `AZURE_TTS_REGION` "
                    "(e.g. `eastus`) in `.env` — Azure's free tier covers 0.5M chars/month."
                )
                return
            if len(text) > self.FRENCH_TTS_MAX_CHARS:
                await message.channel.send(
                    f"❌ That's a bit long for a clip (>{self.FRENCH_TTS_MAX_CHARS} chars). "
                    "Try a shorter phrase."
                )
                return

            async with message.channel.typing():
                audio, reading = await self._synthesize_french(guild_id, text)

            if audio is None:
                await message.channel.send(
                    "❌ TTS synthesis failed — check the logs and that `AZURE_TTS_KEY` / "
                    "`AZURE_TTS_REGION` are valid."
                )
                return

            # Caption: French text + IPA + liaison note (the learner's takeaways).
            caption = f"🔊 **{reading['text']}**"
            if reading.get("ipa"):
                caption += f"\n**IPA:** `{reading['ipa']}`"
            if reading.get("note"):
                caption += f"\n*{reading['note']}*"
            await message.channel.send(
                caption,
                file=discord.File(io.BytesIO(audio), filename="french.mp3"),
            )

        elif cmd == "!summarize":
            # Manually save a thread summary to long-term memory
            # Usage: !summarize <key> <summary>  OR  just !summarize to ask Claude to summarize
            if len(parts) >= 3:
                key = parts[1]
                summary = parts[2]
                if memory.longterm.add(f"thread_{key}", summary):
                    self.manager.save_memories(providers=self.providers)
                    await message.channel.send(f"✅ Saved thread summary as `thread_{key}`")
                else:
                    await message.channel.send(
                        f"❌ Long-term memory full. Use `!forget <key>` to make room."
                    )
            elif len(parts) == 2:
                # !summarize <key> - ask Claude to generate summary
                if not self.claude_provider.enabled:
                    await message.channel.send("❌ Auto-summarize requires Claude (ANTHROPIC_API_KEY not configured).")
                    return
                key = parts[1]
                await message.channel.send(f"📝 Generating summary for this thread as `thread_{key}`...")
                
                # Fetch thread history
                messages = await self.manager.fetch_thread_history(message.channel, limit=50)
                if messages:
                    try:
                        # Build the conversation text
                        conversation_text = "\n".join(
                            m["content"] if isinstance(m["content"], str) else str(m["content"])
                            for m in messages
                        )
                        
                        # Ask Claude to summarize (run in thread pool)
                        summary_response = await asyncio.to_thread(
                            self.claude_client.messages.create,
                            model=self.claude_provider.model_id,
                            max_tokens=200,
                            system="Summarize this conversation in 1-2 sentences. Focus on the key topic and any decisions/outcomes. Be concise.",
                            messages=[{"role": "user", "content": f"Conversation to summarize:\n\n{conversation_text}"}]
                        )
                        summary = summary_response.content[0].text.strip()

                        # Track usage (cache info goes through the helper too)
                        self._record_claude_usage(summary_response.usage)
                        
                        if memory.longterm.add(f"thread_{key}", summary):
                            self.manager.save_memories(providers=self.providers)
                            await message.channel.send(f"✅ Saved: `thread_{key}`: {summary}")
                        else:
                            await message.channel.send(
                                f"❌ Long-term memory full. Use `!forget <key>` to make room.\n"
                                f"Summary was: {summary}"
                            )
                    except anthropic.APIStatusError as e:
                        if e.status_code >= 500:
                            req_id = getattr(e, "request_id", None) or "<not reported>"
                            await message.channel.send(
                                f"⚠️ Claude returned {e.status_code} (Anthropic-side blip, "
                                f"`req_id={req_id}`). Try `!summarize` again."
                            )
                        else:
                            await message.channel.send(
                                f"❌ Summary API Error {e.status_code}: {e.message}"
                            )
                    except anthropic.APIError as e:
                        await message.channel.send(f"❌ Couldn't generate summary: {e}")
                else:
                    await message.channel.send("❌ No messages found in this thread to summarize.")
            else:
                await message.channel.send(
                    "Usage:\n"
                    "`!summarize <key>` - Auto-generate summary of this thread\n"
                    "`!summarize <key> <your summary>` - Save your own summary"
                )

        elif cmd == "!load":
            # Bookclub mode: load a long text (currently AO3 fics) into this
            # channel's pinned context. Persists across restarts. Per-channel
            # so different channels can read different works.
            channel_id = message.channel.id
            if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                channel_id = message.channel.parent_id

            if len(parts) < 2:
                current = self.manager.reading_materials.get(channel_id)
                if current:
                    chapter_count = len(current.chapter_breaks) or 1
                    await message.channel.send(
                        f"📚 Currently loaded: **{current.title}**\n"
                        f"  Source: {current.url}\n"
                        f"  ~{current.estimated_tokens:,} tokens, "
                        f"{current.word_count:,} words, {chapter_count} chapter(s)\n"
                        f"  Loaded: {current.loaded_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"Use `!unload` to drop it, or `!load <url>` to swap."
                    )
                else:
                    await message.channel.send(
                        "Usage: `!load <url>` — load a long text into this channel for bookclub mode.\n"
                        "Currently supports AO3 work URLs. Bot will fetch the full work and "
                        "pin it into every model's system context until you `!unload`."
                    )
                return

            url = parts[1].strip()
            # Strip Discord's auto-link brackets if present
            url = url.strip("<>").rstrip("/")

            ao3_normalized = self._build_ao3_full_work_url(url)
            if ao3_normalized is None:
                await message.channel.send(
                    "❌ I only know how to load AO3 work URLs right now. "
                    "Expected something like `https://archiveofourown.org/works/12345`."
                )
                return

            if not _HAS_BS4:
                await message.channel.send(
                    "❌ AO3 fetcher needs `beautifulsoup4`. Install with:\n"
                    "```\npip install beautifulsoup4\n```\n"
                    "Then restart the bot."
                )
                return

            await message.channel.send(f"📥 Fetching {url} …")
            async with message.channel.typing():
                material, err_reason = await self._fetch_ao3_work(url)

            if material is None:
                error_messages = {
                    "shields_up": (
                        "⛔ **AO3 is shields-up** right now — they're shedding "
                        "anonymous traffic to keep the site responsive for "
                        "logged-in users. Options:\n"
                        "  1. Wait for shields to come down "
                        "(check https://archiveofourown.org/ or @AO3_Status on Bluesky).\n"
                        "  2. Set `AO3_COOKIE` in `.env` with your logged-in "
                        "`_otwarchive_session` cookie value, then `!load` again — "
                        "logged-in requests bypass shields-up.\n"
                        "  3. Use `!load_text` with a `.txt` or `.html` attachment "
                        "if you have the work saved locally."
                    ),
                    "auth_required": (
                        "🔒 That work appears to be **registered-users-only**. "
                        "Set `AO3_COOKIE` in `.env` with a logged-in session cookie "
                        "and try again, or use `!load_text` with an attached file."
                    ),
                    "http_404": "❌ AO3 returned 404 — the work doesn't exist or has been deleted.",
                    "no_content": (
                        "❌ Fetched the page but couldn't find chapter content. "
                        "AO3's HTML may have changed shape, or this is an unusual work format. "
                        "Try `!load_text` with the work text as an attachment."
                    ),
                    "bad_url": (
                        "❌ I only know how to load AO3 work URLs right now. "
                        "Expected something like `https://archiveofourown.org/works/12345`."
                    ),
                    "missing_bs4": (
                        "❌ AO3 fetcher needs `beautifulsoup4`. Install with:\n"
                        "```\npip install beautifulsoup4\n```\nThen restart the bot."
                    ),
                }
                msg = error_messages.get(err_reason or "")
                if msg is None:
                    msg = f"❌ Couldn't load the work (reason: `{err_reason}`). Try `!load_text` with a file attachment instead."
                await message.channel.send(msg)
                return

            # Check which providers can fit it
            chapter_count = len(material.chapter_breaks) or 1
            fits: list[str] = []
            cant_fit: list[str] = []
            # Reserve some budget for memory, history, system framing
            budget_reserve = 50_000
            needed = material.estimated_tokens + budget_reserve
            for p in self.providers:
                if not p.enabled:
                    continue
                if p.max_context_tokens >= needed:
                    fits.append(p.name)
                else:
                    cant_fit.append(f"{p.name} ({p.max_context_tokens:,} ctx)")

            await self._set_reading_material(channel_id, material)
            self.manager.mark_dirty()
            # Force an immediate save so a hard restart in the next 60s
            # doesn't lose the load. Non-blocking — runs in a worker thread.
            await self.manager.save_memories_async(providers=self.providers)

            lines = [
                f"📚 Loaded **{material.title}**",
                f"  ~{material.estimated_tokens:,} tokens, "
                f"{material.word_count:,} words, {chapter_count} chapter(s)",
                f"  Fits in context for: {', '.join(fits) if fits else '(none — too long!)'}",
            ]
            if cant_fit:
                lines.append(f"  ⚠️  Won't fit for: {', '.join(cant_fit)} — those models will see the discussion only, not the source text.")
            lines.append(
                "\nThe text is now pinned to this channel's context. Try "
                "`!claude what do you think of chapter 1?` to start. "
                "Use `!unload` when done."
            )
            await message.channel.send("\n".join(lines))

        elif cmd == "!load_text":
            # Manual fallback for when AO3 is shields-up, the work is locked,
            # or the user already has the text saved locally. Accepts a .txt
            # or .html attachment; title comes from the command arg if given
            # else the filename.
            channel_id = message.channel.id
            if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                channel_id = message.channel.parent_id

            attachments = [
                a for a in message.attachments
                if a.filename.lower().endswith((".txt", ".html", ".htm", ".md"))
            ]
            if not attachments:
                await message.channel.send(
                    "Usage: attach a `.txt`, `.html`, or `.md` file with `!load_text [title]` as the message.\n"
                    "The file's contents will be pinned to this channel as reading material — "
                    "use this when AO3 is shields-up, the work is locked, or you have the text saved locally."
                )
                return

            attachment = attachments[0]
            if attachment.size > 8 * 1024 * 1024:  # 8 MB safety cap
                await message.channel.send(
                    f"❌ File too large ({attachment.size / 1024 / 1024:.1f} MB). "
                    "Keep it under 8 MB — a typical fic-sized text file is well under that."
                )
                return

            await message.channel.send(f"📥 Reading {attachment.filename} …")
            try:
                raw = await self.manager._fetch_text_file(attachment.url)
            except Exception as e:
                await message.channel.send(f"❌ Couldn't read file: {e}")
                return
            if not raw:
                await message.channel.send("❌ File came back empty.")
                return

            # If it's HTML, strip tags. Otherwise treat as plain text.
            is_html = attachment.filename.lower().endswith((".html", ".htm"))
            if is_html:
                if not _HAS_BS4:
                    await message.channel.send(
                        "❌ HTML files need `beautifulsoup4`. Install with `pip install beautifulsoup4`, "
                        "or convert to `.txt` and try again."
                    )
                    return
                soup = _BeautifulSoup(raw, "html.parser")
                # Strip scripts/styles
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                # AO3 download HTMLs have a #chapters wrapper; everything else
                # just gets the body text.
                container = soup.select_one("#chapters") or soup.body or soup
                text = container.get_text("\n", strip=True)
            else:
                text = raw
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            text = _html.unescape(text)

            if not text:
                await message.channel.send("❌ File parsed to empty text.")
                return

            # Title: from command arg if given, else filename without extension
            if len(parts) >= 2:
                title = message.content.split(maxsplit=1)[1].strip()
            else:
                title = re.sub(r"\.(txt|html|htm|md)$", "", attachment.filename, flags=re.IGNORECASE)

            # Best-effort chapter detection. Looks for heading-shaped lines:
            # "Chapter N", "Chapter N: Title", "Prologue", "Epilogue",
            # "Interlude N". Used by !scope / !chapters; harmless if no
            # matches (chapter_breaks stays empty).
            chapter_breaks: list[tuple[int, str]] = self._detect_chapter_breaks(text)

            material = ReadingMaterial(
                url=f"upload://{attachment.filename}",
                title=title,
                text=text,
                chapter_breaks=chapter_breaks,
            )

            # Provider fit check (same as !load)
            budget_reserve = 50_000
            needed = material.estimated_tokens + budget_reserve
            fits: list[str] = []
            cant_fit: list[str] = []
            for p in self.providers:
                if not p.enabled:
                    continue
                if p.max_context_tokens >= needed:
                    fits.append(p.name)
                else:
                    cant_fit.append(f"{p.name} ({p.max_context_tokens:,} ctx)")

            await self._set_reading_material(channel_id, material)
            self.manager.mark_dirty()
            # Force an immediate save so a hard restart in the next 60s
            # doesn't lose the load. Non-blocking — runs in a worker thread.
            await self.manager.save_memories_async(providers=self.providers)

            lines = [
                f"📚 Loaded **{material.title}** from `{attachment.filename}`",
                f"  ~{material.estimated_tokens:,} tokens, {material.word_count:,} words",
                f"  Fits in context for: {', '.join(fits) if fits else '(none — too long!)'}",
            ]
            if cant_fit:
                lines.append(f"  ⚠️  Won't fit for: {', '.join(cant_fit)}")
            lines.append("\nPinned to this channel. Use `!unload` when done.")
            await message.channel.send("\n".join(lines))

        elif cmd == "!unload":
            channel_id = message.channel.id
            if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                channel_id = message.channel.parent_id
            material = self.manager.reading_materials.pop(channel_id, None)
            # Cascade: drop any chapter-scoped materials pinned to this channel's
            # threads too. Each !scope made its OWN Gemini cache keyed by the
            # thread id; without this they orphan and keep billing storage until
            # their TTL. (Snapshot items so we can mutate the dict while looping.
            # Archived/uncached threads aren't resolvable here — those fall back
            # to TTL expiry, which is bounded.)
            scoped = []
            for key, mat in list(self.manager.reading_materials.items()):
                ch = self.get_channel(key)
                if getattr(ch, 'is_thread', False) and ch.parent_id == channel_id:
                    scoped.append((key, mat))
            for key, mat in scoped:
                self.manager.reading_materials.pop(key, None)
                await self._drop_gemini_cache(mat)  # stop scoped-thread cache billing
            if material is None and not scoped:
                await message.channel.send("Nothing loaded in this channel.")
            else:
                if material is not None:
                    await self._drop_gemini_cache(material)  # stop cache storage billing
                self.manager.mark_dirty()
                await self.manager.save_memories_async(providers=self.providers)
                title = f"**{material.title}**" if material is not None else "the scoped threads"
                tail = (
                    f" (+{len(scoped)} scoped-thread cache{'s' if len(scoped) != 1 else ''})"
                    if scoped else ""
                )
                await message.channel.send(f"📤 Unloaded {title}{tail}.")

        elif cmd == "!uncache":
            # Drop ONLY the Gemini context cache(s) for this channel's loaded
            # work (and any scoped-thread caches under it) — the book stays
            # loaded for every model. Stops Gemini storage billing immediately
            # without unloading; Gemini just recreates a fresh cache (current
            # sliding TTL) on its next turn. Handy to kill a stranded long-TTL
            # cache left over from before a restart.
            channel_id = message.channel.id
            if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                channel_id = message.channel.parent_id
            material = self.manager.reading_materials.get(channel_id)
            scoped = []
            for key, mat in list(self.manager.reading_materials.items()):
                ch = self.get_channel(key)
                if getattr(ch, 'is_thread', False) and ch.parent_id == channel_id:
                    scoped.append(mat)
            targets = ([material] if material is not None else []) + scoped
            if not targets:
                await message.channel.send("Nothing loaded in this channel.")
            elif not any(k.startswith("Gemini") for m in targets for k in m.cache_handles):
                await message.channel.send(
                    "No live Gemini cache here to drop — the book stays loaded. "
                    "(A cache only exists once Gemini has actually been used in this channel.)"
                )
            else:
                total_attempted = 0
                total_deleted = 0
                for m in targets:
                    att, dele = await self._drop_gemini_cache(m)
                    total_attempted += att
                    total_deleted += dele
                self.manager.mark_dirty()
                await self.manager.save_memories_async(providers=self.providers)
                title = f"**{material.title}**" if material is not None else "the scoped threads"
                scoped_note = (
                    f" (+{len(scoped)} scoped-thread cache{'s' if len(scoped) != 1 else ''})"
                    if scoped else ""
                )
                failed = total_attempted - total_deleted
                if failed > 0:
                    # A delete that didn't confirm gone (billing/permission 403,
                    # 429, network). The cache may STILL be billing — say so
                    # honestly instead of the old unconditional "billing stops now".
                    await message.channel.send(
                        f"⚠️ Couldn't confirm the Gemini cache for {title}{scoped_note} "
                        f"is gone: {total_deleted}/{total_attempted} deleted, {failed} "
                        f"unverified (likely a depleted balance, rate-limit, or network "
                        f"error) — those may still be billing storage until they "
                        f"self-expire at their {GEMINI_CACHE_TTL_HOURS:g}h TTL. I kept "
                        f"their handles, so re-run `!uncache` once billing is live."
                    )
                else:
                    await message.channel.send(
                        f"🗑️ Dropped the Gemini cache for {title}{scoped_note} — "
                        f"{total_deleted} cache(s) confirmed deleted, storage billing "
                        f"stops now. The book's still loaded for all models. You're in "
                        f"explicit-cache mode, so Gemini rebuilds a fresh "
                        f"{GEMINI_CACHE_TTL_HOURS:g}h cache on its next turn here — "
                        f"`!explicitcache off` switches to inline (no cache) to keep it off."
                    )

        elif cmd == "!explicitcache":
            # Toggle Gemini bookclub caching: INLINE (default — implicit caching,
            # no storage bill, nothing to leak) vs EXPLICIT (cachedContents —
            # guaranteed read discount but bills ~$0.45/hr storage for a 452k
            # book; worth it ONLY for a sustained back-to-back marathon). In-memory
            # (resets to inline on restart — the fail-safe default given the cost
            # history); set providers.gemini.explicit_cache in config for a sticky
            # startup default.
            arg = parts[1].lower() if len(parts) >= 2 else ""
            if arg in ("on", "explicit", "true", "1", "enable"):
                self.gemini_explicit_cache = True
                await message.channel.send(
                    "🟡 Gemini bookclub cache → **EXPLICIT** (cachedContents). Guaranteed "
                    "read discount for a back-to-back **marathon** — but it bills ~$0.45/hr "
                    "storage for a 452k book the whole time it stays alive. `!cost` meters "
                    "that live; run `!explicitcache off` (or `!unload`) when you're done. "
                    "(In-memory — resets to inline on restart.)"
                )
            elif arg in ("off", "inline", "false", "0", "disable"):
                self.gemini_explicit_cache = False
                dropped = 0
                for m in list(self.manager.reading_materials.values()):
                    _, d = await self._drop_gemini_cache(m)
                    dropped += d
                if dropped:
                    self.manager.mark_dirty()
                    await self.manager.save_memories_async(providers=self.providers)
                note = f" Dropped {dropped} live cache(s); storage billing stops now." if dropped else ""
                await message.channel.send(
                    f"🟢 Gemini bookclub cache → **INLINE** (implicit caching, no storage "
                    f"bill — the cheap default, nothing to leak).{note}"
                )
            else:
                state = ("**EXPLICIT** (cachedContents — marathon mode, bills ~$0.45/hr storage)"
                         if self.gemini_explicit_cache
                         else "**INLINE** (implicit caching, no storage bill — default)")
                await message.channel.send(
                    f"Gemini bookclub cache mode: {state}.\n"
                    f"• `!explicitcache on` — explicit cache (only for a sustained marathon)\n"
                    f"• `!explicitcache off` — inline (default; also drops any live cache)"
                )

        elif cmd == "!reading":
            # Lightweight info command — shows what's loaded without the
            # "no usage" path that !load has. Also includes a chapter table
            # of contents if available.
            channel_id = message.channel.id
            if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                channel_id = message.channel.parent_id
            material = self.manager.reading_materials.get(channel_id)
            if material is None:
                await message.channel.send("Nothing loaded in this channel. Use `!load <url>` to start a bookclub.")
                return
            lines = [
                f"📚 **{material.title}**",
                f"  Source: {material.url}",
                f"  ~{material.estimated_tokens:,} tokens, "
                f"{material.word_count:,} words",
                f"  Loaded: {material.loaded_at.strftime('%Y-%m-%d %H:%M')}",
            ]
            if material.chapter_breaks:
                lines.append(f"\n**Chapters** ({len(material.chapter_breaks)}):")
                for i, (_, ch_title) in enumerate(material.chapter_breaks[:30], 1):
                    lines.append(f"  {i}. {ch_title}")
                if len(material.chapter_breaks) > 30:
                    lines.append(f"  … and {len(material.chapter_breaks) - 30} more")
            await self._send_response(message.channel, "\n".join(lines))

        elif cmd == "!chapters":
            # Show chapter TOC with per-chapter token estimates. Looks at
            # this thread's scoped material first, then falls back to the
            # parent channel's full material.
            ch_id = message.channel.id
            parent_id = getattr(message.channel, 'parent_id', None)
            material = self.manager.reading_materials.get(ch_id) or (
                self.manager.reading_materials.get(parent_id) if parent_id else None
            )
            if material is None:
                await message.channel.send("Nothing loaded. Use `!load_text` (with file attachment) or `!load <ao3-url>`.")
                return
            breaks = material.chapter_breaks
            if not breaks:
                await message.channel.send(
                    f"📚 **{material.title}** has no detected chapter structure "
                    f"(~{material.estimated_tokens:,} tokens total). "
                    "`!scope` won't work unless the bot can find chapter headings."
                )
                return
            lines = [f"📚 **{material.title}** — {len(breaks)} chapter(s)"]
            for i, (offset, ch_title) in enumerate(breaks, 1):
                next_offset = breaks[i][0] if i < len(breaks) else len(material.text)
                ch_tokens = int((next_offset - offset) / CONFIG.chars_per_token)
                lines.append(f"  {i}. {ch_title} — ~{ch_tokens:,} tokens")
            lines.append("\nUse `!scope chapter N` (in a thread) to discuss just one chapter.")
            await self._send_response(message.channel, "\n".join(lines))

        elif cmd == "!scope":
            # Thread-only. Slice the parent channel's loaded material to a
            # chapter range and pin the slice to this thread. Generation
            # lookup checks thread_id first, so the slice overrides the
            # parent's full material for any !claude/!deepseek/!gemini turn
            # in this thread. Each scope gets its own caches automatically
            # (content-hashed for Claude/Deepseek; new explicit cache for
            # Gemini on first scoped turn).
            if not getattr(message.channel, 'is_thread', False):
                await message.channel.send(
                    "❌ `!scope` only works inside a thread. Create a thread for the chapter "
                    "you want to discuss, then run `!scope chapter N` there."
                )
                return
            parent_id = message.channel.parent_id
            if parent_id is None:
                await message.channel.send("❌ This thread has no parent channel.")
                return
            parent_material = self.manager.reading_materials.get(parent_id)
            if parent_material is None:
                await message.channel.send(
                    "❌ No reading material loaded in the parent channel. "
                    "Use `!load_text` or `!load` there first, then come back to scope this thread."
                )
                return
            if not parent_material.chapter_breaks:
                await message.channel.send(
                    "❌ The loaded material doesn't have detected chapter structure. "
                    "Re-upload via `!load_text` (the new parser tries to detect chapter "
                    "headings) — or use `!chapters` to confirm."
                )
                return

            n_chapters = len(parent_material.chapter_breaks)
            if len(parts) < 2:
                await message.channel.send(
                    f"Usage: `!scope chapter N` or `!scope chapters N-M` "
                    f"(this work has {n_chapters} chapters). See `!chapters` for the TOC."
                )
                return

            spec = message.content[len("!scope "):].strip().lower()
            # Accept "chapter N", "chapters N-M", "N", "N-M"
            match = re.match(r'^(?:chapters?\s+)?(\d+)(?:\s*-\s*(\d+))?$', spec)
            if not match:
                await message.channel.send(
                    "Usage: `!scope chapter N` or `!scope chapters N-M`"
                )
                return
            start_ch = int(match.group(1))
            end_ch = int(match.group(2)) if match.group(2) else start_ch
            if start_ch < 1 or end_ch > n_chapters or start_ch > end_ch:
                await message.channel.send(
                    f"❌ Invalid range. This work has chapters 1-{n_chapters}."
                )
                return

            scoped = self._slice_material_to_chapters(
                parent_material, start_ch, end_ch
            )

            # Auto-recap: scoped past chapter 1 → give context-limited models a
            # "previously on" of the earlier chapters they can't see. Recaps are
            # generated once (cheapest model) and cached on the parent material.
            if start_ch > 1:
                missing = sum(
                    1 for ch in range(1, start_ch)
                    if ch not in parent_material.chapter_recaps
                )
                if missing:
                    await message.channel.send(
                        f"📝 Summarizing chapters 1–{start_ch - 1} for the scoped models "
                        f"({missing} new recap{'s' if missing != 1 else ''})…"
                    )
                async with message.channel.typing():
                    await self._ensure_chapter_recaps(guild_id, parent_material, start_ch - 1)
                scoped.recap_text = self._format_recap_prefix(parent_material, start_ch)

            await self._set_reading_material(message.channel.id, scoped)
            self.manager.mark_dirty()
            await self.manager.save_memories_async(providers=self.providers)

            range_desc = (
                f"Chapter {start_ch}" if start_ch == end_ch
                else f"Chapters {start_ch}-{end_ch}"
            )
            tier_note = ""
            if scoped.estimated_tokens < 200_000:
                tier_note = " (under Gemini's ≤200k tier — cheaper per turn)"
            # Who can actually READ the scoped text vs. who's on vibes only?
            needed = scoped.estimated_tokens + 50_000
            readers = [p.name for p in self.providers if p.enabled and p.max_context_tokens >= needed]
            vibers = [p.name for p in self.providers if p.enabled and p.max_context_tokens < needed]
            fit_lines = (
                f"  📖 Reads the text: {', '.join(readers) if readers else '(none — too big!)'}\n"
            )
            if vibers:
                fit_lines += (
                    f"  💬 On vibes only (sees the discussion, not the text): "
                    f"{', '.join(vibers)}\n"
                )
            recap_line = (
                f"  📝 Scoped models get a recap of chapters 1–{start_ch - 1}.\n"
                if scoped.recap_text else ""
            )
            await message.channel.send(
                f"📖 This thread is now scoped to **{range_desc}**\n"
                f"  ~{scoped.estimated_tokens:,} tokens, "
                f"{scoped.word_count:,} words{tier_note}\n"
                f"{fit_lines}"
                f"{recap_line}"
                f"  Models in this thread only see the scoped text — no spoilers from later chapters.\n"
                f"  `!unscope` to drop the scope; `!unload` (in parent) to drop the whole work."
            )

        elif cmd == "!unscope":
            if not getattr(message.channel, 'is_thread', False):
                await message.channel.send("❌ `!unscope` only meaningful inside a thread.")
                return
            thread_id = message.channel.id
            scoped = self.manager.reading_materials.pop(thread_id, None)
            if scoped is None:
                await message.channel.send("This thread isn't scoped.")
            else:
                await self._drop_gemini_cache(scoped)  # stop cache storage billing
                self.manager.mark_dirty()
                await self.manager.save_memories_async(providers=self.providers)
                await message.channel.send(
                    "📖 Unscoped this thread. Models will see the full work loaded in the parent channel."
                )

        elif cmd == "!models":
            lines = [f"🤖 **Available Models** _(skin: {self.theme.umbrella})_"]
            for p in self.providers:
                status = "🟢 Enabled" if p.enabled else "⚪ Disabled"
                cost = p.get_cost()
                # Show the themed display_name (e.g. "Judah"/"Gold Head"/"Dummy
                # Plug"); p.name stays the canonical routing key + [label]. The
                # themed aliases follow so users know how to summon each head.
                label = p.display_name or p.name
                aliases = self._provider_aliases(p.id)
                alias_str = f"  ·  {' '.join(aliases)}" if aliases else ""
                if p.total_requests > 0:
                    lines.append(
                        f"  **{label}** ({p.model_id}) - {status}, "
                        f"{p.total_requests} requests, ${cost:.4f}{alias_str}"
                    )
                else:
                    lines.append(f"  **{label}** ({p.model_id}) - {status}{alias_str}")

            mode = CONFIG.default_model
            channel_id = message.channel.id
            parent_id = getattr(message.channel, 'parent_id', None)
            ch_pref = self.channel_preferences.get(channel_id) or self.channel_preferences.get(parent_id)
            if ch_pref:
                lines.append(f"\n  **This channel**: {ch_pref}")
            lines.append(f"  **Selection mode**: {mode}")
            await self._send_response(message.channel, "\n".join(lines))

        elif cmd == "!prefer":
            # Registry-driven: any provider id (or "auto") is a valid pref.
            usage = ("Usage: `!prefer ["
                     + "|".join([p.id for p in self.providers] + ["auto"]) + "]`")
            if len(parts) >= 2:
                pref = parts[1].lower()
                if pref != "auto" and self.registry.by_id(pref) is None:
                    await message.channel.send(usage)
                    return
                channel_id = message.channel.id
                # Use parent channel for threads
                if getattr(message.channel, 'is_thread', False) and message.channel.parent_id:
                    channel_id = message.channel.parent_id
                if pref == "auto":
                    self.channel_preferences.pop(channel_id, None)
                    await message.channel.send("✅ This channel will use **automatic** model selection.")
                else:
                    chosen = self.registry.by_id(pref)
                    if not chosen.enabled:
                        # Keyless heads (the self-hosted Dummy Plug) aren't gated
                        # on an API key — don't tell the operator to set one.
                        if chosen.completions_mode or not chosen.api_key_env:
                            await message.channel.send(
                                f"❌ {chosen.name} isn't enabled. Set "
                                f"`providers.{chosen.id}.enabled = true` in config.json "
                                f"and point `providers.{chosen.id}.base_url` at a "
                                f"/completions-capable endpoint."
                            )
                        else:
                            await message.channel.send(f"❌ {chosen.name} is not configured (no API key).")
                    else:
                        self.channel_preferences[channel_id] = pref
                        await message.channel.send(f"✅ This channel will always use **{chosen.name}**.")
            else:
                channel_id = message.channel.id
                parent_id = getattr(message.channel, 'parent_id', None)
                pref = self.channel_preferences.get(channel_id) or self.channel_preferences.get(parent_id, "auto")
                await message.channel.send(f"Current preference: **{pref}**\n" + usage)

        elif cmd == "!calibration":
            model_name = parts[1].title() if len(parts) >= 2 else None
            models = [model_name] if model_name else [p.name for p in self.providers if p.enabled]
            lines = ["📊 **Calibration Data**"]
            for name in models:
                summary = self.manager.calibration.get_calibration_summary(name)
                lines.append(f"\n  **{name}** ({summary['total']} bids, {summary['rated']} rated):")
                if summary['buckets']:
                    for bucket, data in summary['buckets'].items():
                        pct = int(data['success_rate'] * 100)
                        lines.append(f"    {bucket}: {data['count']} rated, {pct}% positive")
                else:
                    lines.append("    No feedback yet. React with 👍/👎 to bot responses!")
            await self._send_response(message.channel, "\n".join(lines))

        elif cmd == "!help":
            mm_block = "\n".join(self._multimodel_help_lines())
            mm_footer = self._multimodel_footer_line()
            help_text = """
**Commands:**
`!context` - Show current context size and cost estimate
`!cost` - Show total API usage and cost per model
`!memories` - List all memories (both types)
`!threads` - Show other recent threads in this channel
`!search <query>` - Web search via Claude, Deepseek, or Gemini (costs extra, ~$0.01-0.03)
`!research <question>` - Multi-model panel + judge → one synthesised answer (~3-4× cost)
`!research all <question>` - Full roster (adds Mistral/Qwen/GLM when configured) for max diversity
`!speak <chinese / pinyin / phrase>` - Mandarin TTS with tones forced from pinyin → MP3 (Azure Xiaoxiao; needs AZURE_TTS_KEY)
   (models can also voice phrases inline while teaching, via `!speak 汉字`)
`!french <french / english phrase>` - French TTS in natural fr-FR (Azure Denise) + IPA & liaison note → MP3 (needs AZURE_TTS_KEY)
   (models can also voice French inline while teaching, via `[[french: la phrase]]`)

__MM_BLOCK__
`!think <message>` - Use extended thinking (deeper reasoning, slower & costlier)
`!think:<level> <message>` - Force a specific effort level (low|medium|high|xhigh|max)
`!models` - Show available models and their usage stats
`!prefer [claude|deepseek|gemini|mistral|qwen|glm|sim|auto]` - Set model preference for this channel (sim = pin this channel to simulator mode)
`!calibration` - Show model confidence calibration stats
React with 👍/👎 to bot responses to improve model selection
Stack prefixes to combine: `!think !claude <message>` forces Claude with thinking on.
Thinking auto-enables on `!claude`/`!opus` when prompts look hard (cues like "derive",
"why does X", "step by step", LaTeX, large code blocks, stack traces). The chosen
effort level is shown in the response routing.

**Long-term memory (permanent):**
`!remember <key> <value>` - Store a permanent memory
`!forget <key>` - Remove a memory (works for both types)
`!summarize <key>` - Auto-summarize this thread and save it
`!summarize <key> <summary>` - Save your own thread summary

**Bookclub mode (pinned long texts):**
`!load <url>` - Load an AO3 work into this channel's context
`!load_text [title]` - (with .txt/.html/.md attachment) load from a file — works when AO3 is shields-up
`!chapters` - Show the chapter table of contents with per-chapter token counts
`!scope chapter N` / `!scope chapters N-M` - (in a thread) restrict that thread to a chapter range so models only see those chapters — spoiler-safe + much cheaper per turn
`!unscope` - (in a thread) drop the scope and use the parent channel's full work
`!unload` - Drop the loaded work entirely (removes it for ALL models)
`!uncache` - Drop just the Gemini cache, keep the book loaded (stops Gemini storage $)
`!reading` - Show what's currently loaded for this channel/thread
Workflow: load the whole work in a channel once → create a thread per chapter → `!scope chapter N` in each thread. Each model's caches (Claude ephemeral, Gemini cachedContent, Deepseek server-side) are content-hashed, so each scope gets its own cheap cache after the first turn.
Tip: set `AO3_COOKIE` in `.env` with your logged-in session cookie to bypass AO3's shields-up rate-limiting.

**Working memory (auto-managed):**
The AI automatically jots down notes during conversation.
These fade after ~48h if not relevant, or stick around if referenced.
`!keep <key>` - Promote a working note to permanent memory

**Legend for working notes:**
🟢 Fresh (>70% life remaining)
🟡 Fading (30-70% life)
🔴 Almost gone (<30% life)

**Features:**
📷 Upload images and I can see them (Claude + Gemini have vision)
💬 I respond in threads (one channel, multiple convos)
📎 Long code blocks become file attachments
😀 I can react to your messages with emoji
🧵 I can see other threads for context
🔍 Web search with citations (Claude + Gemini native; Deepseek via Tavily)
📚 Bookclub mode: `!load <ao3-url>` pins a fic to the channel for cross-model discussion
__MM_FOOTER__
            """
            help_text = help_text.replace("__MM_BLOCK__", mm_block).replace("__MM_FOOTER__", mm_footer)
            await message.channel.send(help_text)

# (entrypoint lives in main.py — see build_bot()/run in core below)
